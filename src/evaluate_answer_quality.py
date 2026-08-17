#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from chatbot_phase1 import (
    ANSWER_PROMPT_VARIANTS,
    answer_question,
    build_chunks,
    build_idf,
    collect_supported_files,
    ensure_default_pdf,
    prepare_index,
)
from language_classifier import DEFAULT_OLLAMA_URL, resolve_language_model

DEFAULT_LANGUAGE_MODEL = "llama3:latest"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_QUESTION = "Is White Sturgeon a species at risk? What is its current status?"
DEFAULT_GROUND_TRUTH = """Yes. Based only on the supplied SARA document, White Sturgeon (Acipenser transmontanus) is treated as a species at risk in Canada through four separately named populations.

Listed populations:
- Nechako River population
- Upper Columbia River population
- Upper Fraser River population
- Upper Kootenay River population

Status caveat:
The supplied excerpts confirm these White Sturgeon populations appear in SARA material, but do not show the legal risk category next to each entry.
Therefore, the defensible answer is that White Sturgeon is included in the SARA species-at-risk framework at the population level, while the exact current legal category cannot be verified from the available extract alone.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and iteratively improve phase1 answer prompt quality (RAGAS-style)."
    )
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--ground-truth-file", type=Path, default=None)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--lang-model", default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--contradiction-gate", type=float, default=0.49)
    return parser.parse_args()


def load_gemini_api_key(env_path: Path) -> str:
    if not env_path.exists():
        raise RuntimeError(f"Missing .env file at {env_path}")
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY":
            token = value.strip().strip("'").strip('"')
            if token:
                return token
    raise RuntimeError("GEMINI_API_KEY not found in .env")


def parse_gemini_json_response(body: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(body.decode("utf-8"))
        candidates = parsed.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini returned no content parts.")
        text = str(parts[0].get("text", "")).strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        raise RuntimeError("Gemini response parsing failed.") from exc


def try_call_gemini(api_key: str, model: str, prompt: str) -> dict[str, object]:
    normalized_model = model.strip()
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model[len("models/") :]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{normalized_model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        body = response.read()
    return parse_gemini_json_response(body)


def call_gemini_json(api_key: str, model: str, prompt: str) -> dict[str, object]:
    candidates = [model]
    if model != "gemini-flash-latest":
        candidates.append("gemini-flash-latest")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return try_call_gemini(api_key=api_key, model=candidate, prompt=prompt)
        except error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise RuntimeError(
                f"Failed to call Gemini API with model '{candidate}' (HTTP {exc.code})."
            ) from exc
        except (error.URLError, RuntimeError) as exc:
            last_error = exc
            continue
    raise RuntimeError("Failed to call Gemini API.") from last_error


def evaluate_with_gemini(
    api_key: str,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
) -> dict[str, object]:
    rubric_prompt = (
        "You are an evaluator for retrieval-grounded QA. "
        "Score the candidate answer against the reference answer.\n"
        "Use a RAGAS-style claim-centric rubric.\n"
        "Return JSON with fields:\n"
        "- weighted_claim_f1: float in [0,1]\n"
        "- unsupported_claim_rate: float in [0,1]\n"
        "- high_severity_contradiction: float in [0,1]\n"
        "- notes: short string\n\n"
        "Definitions:\n"
        "- weighted_claim_f1: weighted F1 across core claims in the reference.\n"
        "- unsupported_claim_rate: fraction of candidate claims not supported by reference/evidence intent.\n"
        "- high_severity_contradiction: severity of direct contradiction with core reference claims.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{ground_truth}\n\n"
        f"Candidate answer:\n{answer}\n"
    )
    result = call_gemini_json(api_key, model, rubric_prompt)
    required = ("weighted_claim_f1", "unsupported_claim_rate", "high_severity_contradiction")
    for key in required:
        if key not in result or not isinstance(result[key], (int, float)):
            raise RuntimeError(f"Gemini evaluation missing numeric {key}.")
    return result


def load_ground_truth(args: argparse.Namespace) -> str:
    if args.ground_truth_file:
        return args.ground_truth_file.read_text(encoding="utf-8")
    return DEFAULT_GROUND_TRUTH


def save_results(results_dir: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = results_dir / f"answer_eval_{stamp}.json"
    csv_path = results_dir / f"answer_eval_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "iteration",
                "prompt_version",
                "weighted_claim_f1",
                "unsupported_claim_rate",
                "high_severity_contradiction",
                "passes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "iteration": row["iteration"],
                    "prompt_version": row["prompt_version"],
                    "weighted_claim_f1": row["weighted_claim_f1"],
                    "unsupported_claim_rate": row["unsupported_claim_rate"],
                    "high_severity_contradiction": row["high_severity_contradiction"],
                    "passes": row["passes"],
                }
            )
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    ground_truth = load_ground_truth(args)
    gemini_key = load_gemini_api_key(Path(".env"))
    model = resolve_language_model(
        args.ollama_url, args.lang_model, fallback_model=DEFAULT_LANGUAGE_MODEL
    )
    answer_model = args.answer_model or model

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    try:
        ensure_default_pdf(args.docs_dir)
    except (error.URLError, OSError, RuntimeError) as exc:
        print(f"Error preparing docs: {exc}", file=sys.stderr)
        return 1

    files = collect_supported_files(args.docs_dir)
    if not files:
        print("Error: no supported files found in docs directory.", file=sys.stderr)
        return 1
    chunks = build_chunks(files, chunk_size=args.chunk_size)
    idf = build_idf(chunks)
    index = prepare_index(chunks, idf)

    max_iterations = min(max(args.iterations, 1), min(10, len(ANSWER_PROMPT_VARIANTS)))
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    for iteration in range(1, max_iterations + 1):
        try:
            _, answer = answer_question(
                question=args.question,
                index=index,
                idf=idf,
                top_k=max(args.top_k, 1),
                min_score=max(args.min_score, 0.0),
                ollama_url=args.ollama_url,
                lang_model=model,
                answer_model=answer_model,
                answer_prompt_version=iteration,
            )
        except RuntimeError as exc:
            print(f"Iteration {iteration}: answer generation failed: {exc}")
            continue
        eval_result = evaluate_with_gemini(
            api_key=gemini_key,
            model=args.gemini_model,
            question=args.question,
            ground_truth=ground_truth,
            answer=answer,
        )
        weighted_f1 = float(eval_result["weighted_claim_f1"])
        contradiction = float(eval_result["high_severity_contradiction"])
        unsupported = float(eval_result["unsupported_claim_rate"])
        passes = weighted_f1 >= 0.8 and contradiction < args.contradiction_gate
        row = {
            "iteration": iteration,
            "prompt_version": iteration,
            "weighted_claim_f1": weighted_f1,
            "unsupported_claim_rate": unsupported,
            "high_severity_contradiction": contradiction,
            "passes": passes,
            "notes": eval_result.get("notes", ""),
            "answer": answer,
        }
        rows.append(row)
        if best_row is None or weighted_f1 > float(best_row["weighted_claim_f1"]):
            best_row = row
        print(
            f"Iteration {iteration}: weighted_claim_f1={weighted_f1:.3f}, "
            f"unsupported_claim_rate={unsupported:.3f}, "
            f"high_severity_contradiction={contradiction:.3f}, passes={passes}"
        )
        if passes:
            break

    json_path, csv_path = save_results(args.results_dir, rows)
    if best_row is None:
        print("No evaluation rows produced.", file=sys.stderr)
        return 1

    print(
        f"\nBest prompt_version={best_row['prompt_version']} "
        f"weighted_claim_f1={best_row['weighted_claim_f1']:.3f} "
        f"(results: {json_path}, {csv_path})"
    )
    print("\nBest candidate answer:\n")
    print(best_row["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
