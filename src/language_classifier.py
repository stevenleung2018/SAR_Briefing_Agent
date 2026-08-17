from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
SIGNIFICANT_MARGIN = 0.15
SIGNIFICANT_TOP_SCORE = 0.60
LANGUAGE_BENCHMARK = [
    ("Is White Sturgeon a species at risk? What is its current status?", "en"),
    ("What are the recovery measures for Boreal Caribou?", "en"),
    ("Is the Red Knot listed as threatened under Canadian law?", "en"),
    ("What is the habitat requirement for Northern Saw-whet Owl?", "en"),
    ("Does the Atlantic Salmon have a recovery strategy?", "en"),
    ("What is the current conservation status of Wood Bison?", "en"),
    ("Is the Eastern Spotted Skunk considered endangered?", "en"),
    ("What is the population trend of Whooping Crane?", "en"),
    ("What threats affect the Pygmy Short-horned Lizard?", "en"),
    ("Is the Vancouver Island Marmot protected under federal law?", "en"),
    ("Le esturgeon blanc est-il une espèce en péril? Quel est son statut actuel?", "fr"),
    ("Quelles sont les mesures de rétablissement du caribou boréal?", "fr"),
    ("Le bécasseau maubèche est-il inscrit comme menacé en droit canadien?", "fr"),
    ("Quelle est l’exigence en matière d’habitat pour la chouette nivalis?", "fr"),
    ("Le saumon de l’Atlantique a-t-il un plan de rétablissement?", "fr"),
    ("Quel est le statut de conservation actuel du bison des bois?", "fr"),
    ("L’ondatra à pattes blanches est-il considéré comme en voie de disparition?", "fr"),
    ("Quelle est la tendance de la population de la grue blanche?", "fr"),
    ("Quelles sont les menaces qui pèsent sur le lézard nain à cornes courtes?", "fr"),
    ("Le marmotte de l’île de Vancouver est-elle protégée par la loi fédérale?", "fr"),
]


@dataclass
class LanguageScores:
    en: float
    fr: float
    other: float

    def top_label(self) -> str:
        items = {"en": self.en, "fr": self.fr, "other": self.other}
        return max(items, key=items.get)

    def normalized(self) -> "LanguageScores":
        total = self.en + self.fr + self.other
        if total <= 0.0:
            raise RuntimeError("Invalid language scores: total is zero.")
        return LanguageScores(self.en / total, self.fr / total, self.other / total)

    def with_boost(self, label: str, boost: float) -> "LanguageScores":
        en = self.en
        fr = self.fr
        other = self.other
        if label == "en":
            en += boost
        elif label == "fr":
            fr += boost
        elif label == "other":
            other += boost
        else:
            raise RuntimeError(f"Unsupported label for boost: {label}")
        return LanguageScores(en=en, fr=fr, other=other).normalized()

    def is_significant_top_language(self) -> bool:
        top = self.top_label()
        if top == "en":
            second = max(self.fr, self.other)
            return self.en >= SIGNIFICANT_TOP_SCORE and (self.en - second) >= SIGNIFICANT_MARGIN
        if top == "fr":
            second = max(self.en, self.other)
            return self.fr >= SIGNIFICANT_TOP_SCORE and (self.fr - second) >= SIGNIFICANT_MARGIN
        second = max(self.en, self.fr)
        return self.other >= SIGNIFICANT_TOP_SCORE and (self.other - second) >= SIGNIFICANT_MARGIN

    @staticmethod
    def one_hot(label: str) -> "LanguageScores":
        if label == "en":
            return LanguageScores(en=1.0, fr=0.0, other=0.0)
        if label == "fr":
            return LanguageScores(en=0.0, fr=1.0, other=0.0)
        if label == "other":
            return LanguageScores(en=0.0, fr=0.0, other=1.0)
        raise RuntimeError(f"Unsupported one-hot language label: {label}")


def request_ollama_response(ollama_url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as response:
            body = response.read()
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Unable to reach Ollama at {ollama_url}. Ensure Ollama is running."
        ) from exc

    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    return str(parsed.get("response", "")).strip()


def resolve_language_model(
    ollama_url: str, preferred_model: str | None, fallback_model: str
) -> str:
    if preferred_model:
        return preferred_model

    tags_url = (
        ollama_url.rsplit("/api/", 1)[0] + "/api/tags"
        if "/api/" in ollama_url
        else DEFAULT_OLLAMA_TAGS_URL
    )
    try:
        tags_req = request.Request(
            tags_url,
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with request.urlopen(tags_req, timeout=15) as response:
            tags_data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, json.JSONDecodeError, ValueError):
        return fallback_model

    models = tags_data.get("models") or []
    if models and isinstance(models, list):
        first_model = models[0]
        if isinstance(first_model, dict):
            name = first_model.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return fallback_model


def parse_language_scores(raw_text: str) -> LanguageScores:
    candidate = raw_text.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            candidate = match.group(0)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Language classifier returned invalid JSON: {raw_text}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Language classifier returned non-object JSON: {raw_text}")

    def get_score(key: str) -> float:
        value = parsed.get(key)
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"Language classifier JSON missing numeric '{key}' score: {raw_text}"
            )
        return float(value)

    en = max(0.0, get_score("en"))
    fr = max(0.0, get_score("fr"))
    other = max(0.0, get_score("other"))
    return LanguageScores(en=en, fr=fr, other=other).normalized()


def parse_language_label(raw_text: str) -> str:
    normalized = raw_text.strip().lower().split()[0] if raw_text.strip() else ""
    normalized = normalized.strip(".,;:!?'\"()[]{}")
    if normalized in {"en", "english"}:
        return "en"
    if normalized in {"fr", "french"}:
        return "fr"
    if normalized in {"other", "autre"}:
        return "other"
    raise RuntimeError(
        f"Language tie-break classifier returned '{raw_text}' (expected en/fr/other)."
    )


def classify_language_scores(ollama_url: str, model: str, prompt: str) -> LanguageScores:
    raw = request_ollama_response(ollama_url=ollama_url, model=model, prompt=prompt)
    return parse_language_scores(raw)


def detect_question_language(question: str, ollama_url: str, model: str) -> LanguageScores:
    prompts = [
        (
            "You are a language classifier.\n"
            "Given a user question, estimate probabilities for English, French, and Other.\n"
            "Return ONLY JSON with numeric values that sum approximately to 1.\n"
            "Required schema: {\"en\": <number>, \"fr\": <number>, \"other\": <number>}.\n"
            "Do not include any additional keys or text.\n\n"
            f"Question: {question}"
        ),
        (
            "Classify the language distribution of this user question.\n"
            "Output ONLY valid JSON: {\"en\": number, \"fr\": number, \"other\": number}.\n"
            "Examples:\n"
            "Q: Is White Sturgeon a species at risk?\n"
            "A: {\"en\": 0.99, \"fr\": 0.00, \"other\": 0.01}\n"
            "Q: Le caribou des bois est-il en péril?\n"
            "A: {\"en\": 0.00, \"fr\": 0.99, \"other\": 0.01}\n\n"
            f"Q: {question}\nA:"
        ),
        (
            "Estimate language probabilities for this text.\n"
            "Only output this JSON object: {\"en\":...,\"fr\":...,\"other\":...}\n"
            "No explanation.\n\n"
            f"Text: {question}"
        ),
    ]

    score_runs: list[LanguageScores] = []
    for prompt in prompts:
        try:
            score_runs.append(classify_language_scores(ollama_url, model, prompt))
        except RuntimeError:
            continue

    tie_break_prompt = (
        "Pick the predominant language of this question.\n"
        "Return exactly one token: en, fr, or other.\n"
        "No punctuation, no explanation.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    if not score_runs:
        tie_break_raw = request_ollama_response(
            ollama_url=ollama_url, model=model, prompt=tie_break_prompt
        )
        tie_break_label = parse_language_label(tie_break_raw)
        return LanguageScores.one_hot(tie_break_label)

    averaged = LanguageScores(
        en=sum(s.en for s in score_runs) / len(score_runs),
        fr=sum(s.fr for s in score_runs) / len(score_runs),
        other=sum(s.other for s in score_runs) / len(score_runs),
    ).normalized()

    if averaged.is_significant_top_language():
        return averaged

    tie_break_raw = request_ollama_response(
        ollama_url=ollama_url, model=model, prompt=tie_break_prompt
    )
    tie_break_label = parse_language_label(tie_break_raw)
    return averaged.with_boost(tie_break_label, boost=0.25)


def evaluate_classification_significance(scores: LanguageScores, expected_label: str) -> bool:
    if scores.top_label() != expected_label:
        return False
    candidate_score = getattr(scores, expected_label)
    second_best = max(
        getattr(scores, other_label)
        for other_label in ("en", "fr", "other")
        if other_label != expected_label
    )
    return candidate_score >= SIGNIFICANT_TOP_SCORE and (candidate_score - second_best) >= SIGNIFICANT_MARGIN


def run_language_benchmark(model_name: str, ollama_url: str) -> dict[str, object]:
    en_correct = 0
    en_significant = 0
    fr_correct = 0
    fr_significant = 0

    for question, expected in LANGUAGE_BENCHMARK:
        scores = detect_question_language(question, ollama_url=ollama_url, model=model_name)
        predicted = scores.top_label()
        is_significant = evaluate_classification_significance(scores, expected)
        if expected == "en":
            if predicted == "en":
                en_correct += 1
            if is_significant:
                en_significant += 1
        elif expected == "fr":
            if predicted == "fr":
                fr_correct += 1
            if is_significant:
                fr_significant += 1

    benchmark = {
        "model": model_name,
        "en_correct": en_correct,
        "en_significant": en_significant,
        "fr_correct": fr_correct,
        "fr_significant": fr_significant,
        "total_en": sum(1 for _, expected in LANGUAGE_BENCHMARK if expected == "en"),
        "total_fr": sum(1 for _, expected in LANGUAGE_BENCHMARK if expected == "fr"),
    }
    benchmark["passes"] = (
        benchmark["en_significant"] >= 8 and benchmark["fr_significant"] >= 8
    )
    return benchmark


def save_benchmark_result(model_name: str, benchmark: dict[str, object], root_dir: Path) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    output_path = root_dir / f"{model_name.replace(':', '_')}.json"
    output_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")


def load_benchmark_result(model_name: str, root_dir: Path) -> dict[str, object] | None:
    output_path = root_dir / f"{model_name.replace(':', '_')}.json"
    if not output_path.exists():
        return None
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_or_run_language_benchmark(
    model_name: str, ollama_url: str, results_dir: Path, force: bool = False
) -> dict[str, object]:
    if not force:
        cached = load_benchmark_result(model_name, results_dir)
        if cached is not None:
            return cached
    benchmark = run_language_benchmark(model_name, ollama_url)
    save_benchmark_result(model_name, benchmark, results_dir)
    return benchmark


def assert_model_can_classify(model_name: str, ollama_url: str, results_dir: Path) -> None:
    benchmark = get_or_run_language_benchmark(model_name, ollama_url, results_dir)
    if not benchmark["passes"]:
        raise RuntimeError(
            "The model available via Ollama is not strong enough for bilingual EN/FR classification. "
            f"Benchmark result for {model_name}: English significant accuracy={benchmark['en_significant']}/10, "
            f"French significant accuracy={benchmark['fr_significant']}/10. "
            "Please install or select a stronger multilingual model and try again."
        )
