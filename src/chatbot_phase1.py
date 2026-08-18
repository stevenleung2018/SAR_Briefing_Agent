#!/usr/bin/env python3
"""
Phase 1 local chatbot:
- Answers questions using only offline documents in a local data/ folder.
- Supports .txt, .md, and .pdf files.
- Uses TF-IDF retrieval and evidence-grounded responses.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import zipfile
from xml.etree import ElementTree
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib import error, request
from urllib.parse import urlparse

from pypdf import PdfReader

from constants import DEFAULT_RESULTS_DIR
from language_classifier import (
    LanguageScores,
    assert_model_can_classify,
    detect_question_language,
    request_ollama_response,
    resolve_language_model,
)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']+")
WHITESPACE_PATTERN = re.compile(r"\s+")
DEFAULT_PDF_URL = "https://laws-lois.justice.gc.ca/PDF/S-15.3.pdf"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_LANGUAGE_MODEL = "llama3:latest"
DEFAULT_ANSWER_PROMPT_VERSION = 2
RECOVERY_ANSWER_PROMPT_VERSION = 1
ANSWER_PROGRESS_STAGE_ORDER = (
    "language_detection",
    "evidence_retrieval",
    "answer_generation",
)
DEFAULT_ANSWER_STAGE_ETA_SECONDS = {
    "language_detection": 9.0,
    "evidence_retrieval": 1.5,
    "answer_generation": 14.0,
}
ANSWER_STAGE_ETA_SMOOTHING_ALPHA = 0.35
QUERY_STOPWORDS = {
    "is",
    "are",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "does",
    "do",
    "the",
    "a",
    "an",
    "at",
    "risk",
    "species",
    "est",
    "sont",
    "quoi",
    "quand",
    "où",
    "pourquoi",
    "comment",
    "peut",
    "les",
    "la",
    "le",
    "des",
    "une",
    "un",
    "espèce",
    "risque",
    "de",
    "du",
    "et",
    "of",
    "in",
    "on",
    "for",
    "to",
    "its",
    "current",
    "status",
    "their",
    "statut",
    "actuel",
    "actuelle",
    "courant",
    "quelle",
    "quel",
    "quelles",
    "quels",
}
POPULATION_LISTING_TERMS = {
    "population",
    "populations",
    "listed populations",
    "population list",
    "population-level",
    "population level",
    "population names",
    "populations listées",
    "population listée",
    "liste des populations",
    "nom des populations",
}
RECOVERY_INTENT_TERMS = {
    "recovery",
    "recover",
    "recovery strategy",
    "action plan",
    "critical habitat",
    "habitat",
    "mesures",
    "rétablissement",
    "programme de rétablissement",
    "plan d'action",
    "habitat essentiel",
}
RECOVERY_LEGAL_QUERY = (
    "recovery strategy action plan critical habitat schedule of studies "
    "technically and biologically feasible best available information COSEWIC "
    "threats to survival loss of habitat population and distribution objectives "
    "activities likely to result in destruction monitor recovery socio-economic costs benefits"
)
RECOVERY_CONCEPT_GROUPS = {
    "feasibility": [
        "technically and biologically feasible",
        "best available information",
        "cosewic",
    ],
    "threats": [
        "threats to the survival",
        "loss of habitat",
        "menaces à la survie",
        "perte de son habitat",
    ],
    "critical_habitat": ["critical habitat", "habitat essentiel"],
    "destruction_examples": [
        "activities that are likely to result in its destruction",
        "activités susceptibles d’entraîner sa destruction",
    ],
    "schedule_studies": ["schedule of studies", "calendrier des études"],
    "population_distribution": [
        "population and distribution objectives",
        "objectifs en matière de population et de dissémination",
    ],
    "action_plan": ["action plan", "plan d’action", "plan d'action"],
    "action_plan_timeline": [
        "when one or more action plans",
        "when these measures are to take place",
    ],
    "implementation_details": [
        "monitor the recovery",
        "socio-economic costs",
        "benefits to be derived from its implementation",
    ],
    "critical_habitat_protection": [
        "critical habitat that is identified in a recovery strategy or in an action plan must be protected",
        "recovery strategy or action plan that identified the critical habitat",
        "must, within 180 days",
    ],
}
ANSWER_PROMPT_VARIANTS = [
    (
        "You are a retrieval-grounded legal QA assistant for Canada's Species at Risk Act (SARA).\n"
        "Follow a claim-by-claim evidence policy inspired by RAGAS: every factual statement must be directly supported by the supplied excerpts, and unsupported statements must be removed or clearly labeled as not shown.\n"
        "Do not use outside knowledge, assumptions, or legal status beyond the snippets.\n"
        "Distinguish between (a) the species appears in SARA material and (b) the exact legal category is explicitly shown in the evidence.\n"
        "If the snippets do not show a status such as threatened, endangered, or special concern, say precisely that the legal category cannot be verified from the provided evidence.\n"
        "Return concise Markdown and follow the preferred structure headings provided later in this prompt.\n"
    ),
    (
        "You must answer from evidence only and optimize for precision over completeness.\n"
        "Every sentence must be traceable to one of the supplied snippets or clearly marked as 'not shown in the provided evidence'.\n"
        "Do NOT infer missing legal categories or status labels.\n"
        "When population-level listing is requested, include only populations explicitly tied to the asked species in the evidence.\n"
        "Provide: direct answer, evidence summary, and a caveat about status category if not explicitly shown.\n"
    ),
    (
        "Grounded QA task with evidence discipline.\n"
        "Use only evidence snippets below.\n"
        "For species questions, distinguish between 'appears in SARA material' and 'specific legal category shown'.\n"
        "If category is missing, say the exact legal status is not displayed in the provided excerpts and avoid guessing.\n"
        "Do not add any legal conclusion that the snippets do not state directly.\n"
    ),
    (
        "Evidence-locked response policy:\n"
        "- Allowed facts: only what appears in snippets.\n"
        "- Forbidden: inferred listing category not in snippets.\n"
        "- If populations are listed for the species, list them exactly once each.\n"
        "- If evidence supports inclusion in SARA framework, say yes with caveat if category absent.\n"
    ),
    (
        "Write an evidence-grounded answer in a compliance style.\n"
        "Prioritize precision over completeness.\n"
        "If evidence includes bilingual species rows, map only rows explicitly associated with the asked species.\n"
        "Never claim a listing category unless directly present in evidence.\n"
    ),
    (
        "Answer strictly from snippets.\n"
        "When asking about current status, require explicit status words (e.g., endangered, threatened, special concern).\n"
        "If none exist in snippets, state that current category cannot be verified from available extract.\n"
    ),
    (
        "You are scoring yourself for hallucination risk.\n"
        "Prefer caveated truth over unsupported detail.\n"
        "Output markdown with: Direct answer, Evidence summary, Status caveat.\n"
    ),
    (
        "Produce a conservative legal brief from provided snippets.\n"
        "Extract only species-specific population names and avoid mixing other species.\n"
        "If exact legal status category is absent, say so explicitly.\n"
    ),
    (
        "Use strict retrieval-grounded synthesis.\n"
        "If the question asks about population-level listing, mention it only when supported by evidence.\n"
        "Do not overstate certainty for category labels not shown.\n"
    ),
    (
        "Answer like a fact-checker.\n"
        "Claims must be directly traceable to snippets.\n"
        "If uncertain or absent in snippets, mark as unverifiable from provided extract.\n"
    ),
]


@dataclass
class Chunk:
    file_path: Path
    chunk_id: int
    text: str
    tokens: list[str]


@dataclass
class DocumentKnowledgeSummary:
    file_path: Path
    chunk_count: int
    token_count: int
    executive_summary: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options that configure the local document chatbot."""
    parser = argparse.ArgumentParser(
        description="Phase 1 local chatbot for offline documents only."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data"),
        help="Path to local documents directory (default: data).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence chunks to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.08,
        help="Minimum cosine similarity score for considering evidence (default: 0.08).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Approximate max characters per text chunk (default: 900).",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate API URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--lang-model",
        default=None,
        help="Ollama model for EN/FR language classification.",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Ollama model for grounded answer generation (default: same as language model).",
    )
    parser.add_argument(
        "--answer-prompt-version",
        type=int,
        default=None,
        help=(
            f"Force a fixed answer prompt variant number (1-{len(ANSWER_PROMPT_VARIANTS)}). "
            "If omitted, the chatbot auto-routes prompts by question intent."
        ),
    )
    parser.add_argument(
        "--debug-long",
        action="store_true",
        help="Print language classification debug output for each question.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the required language benchmark before launching the chatbot.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces in a text block."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split a text block into lowercase alphanumeric tokens suitable for retrieval."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def read_text_file(file_path: Path) -> str:
    """Read a UTF-8 document as plain text, replacing undecodable characters safely."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def read_pdf_file(file_path: Path) -> str:
    """Extract and flatten the text from a PDF into one searchable document string."""
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            pages.append(page_text)
    joined = "\n".join(pages)
    joined = joined.replace("-\n", "")
    joined = joined.replace("\n", " ")
    return joined


def read_document(file_path: Path) -> str:
    """Read a supported local document using the proper extractor for its file type."""
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return read_pdf_file(file_path)
    return read_text_file(file_path)


def count_document_pages(file_path: Path) -> int:
    """Count pages for PDF or DOCX files when summarizing the indexed corpus."""
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        return len(reader.pages)
    if extension != ".docx":
        return 1
    with zipfile.ZipFile(file_path) as archive:
        try:
            with archive.open("docProps/app.xml") as app_xml:
                root = ElementTree.parse(app_xml).getroot()
        except KeyError:
            return 0
    for element in root.iter():
        if element.tag.endswith("Pages") and element.text:
            try:
                return int(element.text)
            except ValueError:
                return 0
    return 0


def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split long document text into overlapping chunks that keep retrieval windows manageable."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    words_per_chunk = max(80, chunk_size // 6)
    overlap = max(20, words_per_chunk // 5)
    chunks: list[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(total, start + words_per_chunk)
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words).strip())
        if end >= total:
            break
        start = max(start + 1, end - overlap)
    return chunks


def collect_supported_files(docs_dir: Path) -> list[Path]:
    """Collect all text and PDF documents in the local docs directory that we can index."""
    files: list[Path] = []
    for path in docs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def ensure_default_pdf(docs_dir: Path) -> tuple[Path, bool]:
    """Ensure the default SARA PDF exists locally and download it if the workspace is empty."""
    pdf_filename = Path(urlparse(DEFAULT_PDF_URL).path).name
    target_path = docs_dir / pdf_filename
    if target_path.exists():
        return target_path, False

    with request.urlopen(DEFAULT_PDF_URL) as response:
        content = response.read()
    if not content:
        raise RuntimeError(f"Download returned no content from {DEFAULT_PDF_URL}")

    target_path.write_bytes(content)
    return target_path, True


def build_chunks(files: Iterable[Path], chunk_size: int) -> list[Chunk]:
    """Create retrieval chunks from each indexed document while preserving source metadata."""
    chunks: list[Chunk] = []
    for file_path in files:
        content = read_document(file_path)
        split = split_into_chunks(content, chunk_size=chunk_size)
        for idx, text in enumerate(split):
            tokens = tokenize(text)
            if tokens:
                chunks.append(Chunk(file_path=file_path, chunk_id=idx, text=text, tokens=tokens))
    return chunks


def build_idf(chunks: list[Chunk]) -> dict[str, float]:
    """Compute inverse document frequency values for the indexed chunk corpus."""
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        for term in set(chunk.tokens):
            doc_freq[term] += 1

    total_docs = max(len(chunks), 1)
    idf: dict[str, float] = {}
    for term, freq in doc_freq.items():
        idf[term] = math.log((1 + total_docs) / (1 + freq)) + 1.0
    return idf


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Build a TF-IDF vector from tokens so evidence matching can be scored numerically."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    vector: dict[str, float] = {}
    for term, count in counts.items():
        if term in idf:
            tf = count / total
            vector[term] = tf * idf[term]
    return vector


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Measure similarity between two sparse vectors using cosine distance."""
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for term, val_a in vec_a.items():
        dot += val_a * vec_b.get(term, 0.0)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def prepare_index(chunks: list[Chunk], idf: dict[str, float]) -> list[tuple[Chunk, dict[str, float]]]:
    """Precompute the chunk vectors needed for retrieval during each question."""
    return [(chunk, tfidf_vector(chunk.tokens, idf)) for chunk in chunks]


def is_recovery_measures_question(question: str) -> bool:
    """Detect whether the user is asking about recovery requirements rather than general status questions."""
    q = question.lower()
    return any(term in q for term in RECOVERY_INTENT_TERMS)


def is_population_listing_question(question: str) -> bool:
    """Detect whether the user explicitly asks for population-level listing details."""
    q = question.lower()
    return any(term in q for term in POPULATION_LISTING_TERMS)


def recovery_concept_match_count(chunk_text: str) -> int:
    """Count the number of recovery-related concept groups present in a chunk."""
    concept_hits = 0
    for phrases in RECOVERY_CONCEPT_GROUPS.values():
        if any(phrase in chunk_text for phrase in phrases):
            concept_hits += 1
    return concept_hits


def resolve_answer_prompt_version(question: str, forced_prompt_version: int | None) -> int:
    """Select the answer prompt variant that best fits the user's question and any explicit override."""
    if forced_prompt_version is not None:
        return min(max(forced_prompt_version, 1), len(ANSWER_PROMPT_VARIANTS))
    if is_recovery_measures_question(question):
        return RECOVERY_ANSWER_PROMPT_VERSION
    return DEFAULT_ANSWER_PROMPT_VERSION


def retrieve_evidence(
    question: str,
    index: list[tuple[Chunk, dict[str, float]]],
    idf: dict[str, float],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Score local chunks and return the strongest evidence for the current question."""
    query_tokens = tokenize(question)
    query_vec = tfidf_vector(query_tokens, idf)
    recovery_vec = tfidf_vector(tokenize(RECOVERY_LEGAL_QUERY), idf)
    recovery_intent = is_recovery_measures_question(question)
    focus_tokens = [t for t in query_tokens if len(t) > 2 and t not in QUERY_STOPWORDS]
    query_bigrams = {
        (focus_tokens[i], focus_tokens[i + 1])
        for i in range(len(focus_tokens) - 1)
    }
    scored: list[tuple[Chunk, float]] = []

    for chunk, chunk_vec in index:
        score = cosine_similarity(query_vec, chunk_vec)
        chunk_token_set = set(chunk.tokens)
        chunk_lower = chunk.text.lower()
        if focus_tokens:
            matches = sum(1 for token in focus_tokens if token in chunk_token_set)
            if matches > 0:
                score += 0.10 * (matches / len(focus_tokens))
                if query_bigrams:
                    chunk_bigrams = {
                        (chunk.tokens[i], chunk.tokens[i + 1])
                        for i in range(len(chunk.tokens) - 1)
                    }
                    bigram_matches = sum(1 for pair in query_bigrams if pair in chunk_bigrams)
                    if bigram_matches > 0:
                        score += 0.20 * (bigram_matches / len(query_bigrams))
                if matches == len(focus_tokens):
                    score += 0.25
        if recovery_intent:
            score += 0.45 * cosine_similarity(recovery_vec, chunk_vec)
            concept_matches = recovery_concept_match_count(chunk_lower)
            if concept_matches:
                score += 0.20 * concept_matches
        if score > 0.0:
            scored.append((chunk, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def summarize_evidence(evidence: list[tuple[Chunk, float]], language: str) -> str:
    """Render evidence in a concise, readable format for debugging or trace output."""
    lines = []
    for chunk, score in evidence:
        snippet = chunk.text
        if len(snippet) > 260:
            snippet = f"{snippet[:257]}..."
        source = f"{chunk.file_path.name} (chunk {chunk.chunk_id + 1})"
        if language == "fr":
            lines.append(f"- {snippet}\n  Source : {source}, pertinence={score:.2f}")
        else:
            lines.append(f"- {snippet}\n  Source: {source}, relevance={score:.2f}")
    return "\n".join(lines)


def format_evidence_for_prompt(evidence: list[tuple[Chunk, float]]) -> str:
    """Prepare the final evidence list as a prompt block containing source metadata and excerpts."""
    lines: list[str] = []
    for idx, (chunk, score) in enumerate(evidence, start=1):
        snippet = chunk.text.strip()
        if len(snippet) > 700:
            snippet = f"{snippet[:700]}..."
        source = f"{chunk.file_path.name} (chunk {chunk.chunk_id + 1})"
        lines.append(
            f"[Evidence {idx}] score={score:.3f} source={source}\n{snippet}"
        )
    return "\n\n".join(lines)


def build_answer_prompt(
    question: str, language: str, evidence_block: str, prompt_version: int
) -> str:
    """Build the final prompting template that constrains the answer to the evidence and target language."""
    idx = min(max(prompt_version, 1), len(ANSWER_PROMPT_VARIANTS)) - 1
    policy = ANSWER_PROMPT_VARIANTS[idx]
    recovery_intent = is_recovery_measures_question(question)
    population_listing_intent = is_population_listing_question(question)
    if recovery_intent:
        structure_en = "## Direct answer\n## Required recovery measures\n## Implementation and protection\n"
        structure_fr = "## Réponse directe\n## Mesures de rétablissement requises\n## Mise en œuvre et protection\n"
    elif population_listing_intent:
        structure_en = "## Direct answer\n## Listed populations\n## Status caveat\n"
        structure_fr = "## Réponse directe\n## Populations listées\n## Réserve sur le statut\n"
    else:
        structure_en = "## Direct answer\n## Evidence summary\n## Status caveat\n"
        structure_fr = "## Réponse directe\n## Résumé des preuves\n## Réserve sur le statut\n"
    recovery_addendum_en = (
        "Because this is a recovery-measures question, focus on statutory recovery requirements.\n"
        "Extract and list the concrete required elements if present in evidence:\n"
        "- technical/biological feasibility determination using best available information including COSEWIC\n"
        "- threats to survival including habitat loss\n"
        "- population and distribution objective\n"
        "- critical habitat identification (to extent possible)\n"
        "- examples of activities likely to destroy critical habitat\n"
        "- schedule of studies when critical habitat cannot be fully identified\n"
        "- broad strategies and timeline for action plan(s)\n"
        "- action plan implementation measures, monitoring methods, socio-economic costs/benefits\n"
        "- protection framework once critical habitat is identified\n"
        "Do NOT provide a 'listed populations' section unless the question explicitly asks for populations.\n"
        "Do NOT infer details not in evidence.\n"
    )
    recovery_addendum_fr = (
        "Puisqu'il s'agit d'une question sur les mesures de rétablissement, concentrez-vous sur les exigences prévues par la loi.\n"
        "Extrayez les éléments obligatoires présents dans les extraits (faisabilité, menaces, habitat essentiel, calendrier, plan d’action, suivi, coûts/bénéfices, protection).\n"
        "N'ajoutez pas de section 'populations listées' sauf si la question le demande explicitement.\n"
        "N'inférez rien qui n'est pas dans les extraits.\n"
    )
    population_addendum_en = (
        "Only include population names when the user explicitly asks for populations.\n"
        "If populations are requested, list only names directly supported by the evidence and tied to the asked species.\n"
    )
    population_addendum_fr = (
        "N'incluez des noms de populations que si l'utilisateur le demande explicitement.\n"
        "Si des populations sont demandées, listez uniquement les noms appuyés par les preuves et liés à l'espèce demandée.\n"
    )
    evidence_self_check_en = (
        "Evidence discipline checklist (apply silently before finalizing):\n"
        "- Every factual claim must be directly supported by a specific evidence snippet.\n"
        "- If a legal category is not explicitly shown, state 'not directly shown in the provided evidence' rather than inferring it.\n"
        "- Distinguish between species inclusion in SARA and exact status wording such as threatened/endangered/special concern.\n"
        "- Do not add population names, legal categories, or consequences that are not present in the excerpts.\n"
        "- When the evidence is incomplete, prefer a conservative, evidence-backed caveat over a stronger claim.\n"
    )
    evidence_self_check_fr = (
        "Liste de vérification de discipline des preuves (à appliquer silencieusement avant de finaliser) :\n"
        "- Chaque affirmation factuelle doit être directement appuyée par un extrait de preuve précis.\n"
        "- Si une catégorie juridique n’est pas explicitement indiquée, dites qu’elle n’est pas directement montrée dans les preuves fournies plutôt que de l’inférer.\n"
        "- Distinguez entre l’inclusion dans la LEP et le libellé exact du statut (menacée/en voie de disparition/préoccupante, etc.).\n"
        "- N’ajoutez pas de noms de populations, de catégories juridiques ou de conséquences qui ne figurent pas dans les extraits.\n"
        "- Quand les preuves sont incomplètes, privilégiez une réserve prudente et fondée sur les preuves.\n"
    )
    evidence_self_check = (
        evidence_self_check_fr if language == "fr" else evidence_self_check_en
    )
    if language == "fr":
        return (
            f"{policy}\n"
            f"{recovery_addendum_fr if recovery_intent else ''}"
            f"{population_addendum_fr if population_listing_intent else ''}"
            f"{evidence_self_check}\n"
            "Répondez uniquement en français.\n"
            "Structure recommandée:\n"
            f"{structure_fr}\n"
            "Question utilisateur:\n"
            f"{question}\n\n"
            "Extraits de preuve:\n"
            f"{evidence_block}\n\n"
            "Réponse:"
        )
    return (
        f"{policy}\n"
        f"{recovery_addendum_en if recovery_intent else ''}"
        f"{population_addendum_en if population_listing_intent else ''}"
        f"{evidence_self_check}\n"
        "Respond only in English.\n"
        "Preferred structure:\n"
        f"{structure_en}\n"
        "User question:\n"
        f"{question}\n\n"
        "Evidence snippets:\n"
        f"{evidence_block}\n\n"
        "Answer:"
    )


def generate_grounded_answer(
    question: str,
    language: str,
    evidence: list[tuple[Chunk, float]],
    ollama_url: str,
    model: str,
    prompt_version: int,
) -> str:
    """Generate a response grounded only in the retrieved local evidence snippets."""
    evidence_block = format_evidence_for_prompt(evidence)
    prompt = build_answer_prompt(
        question=question,
        language=language,
        evidence_block=evidence_block,
        prompt_version=prompt_version,
    )
    raw = request_ollama_response(ollama_url=ollama_url, model=model, prompt=prompt)
    answer = raw.strip()
    if answer.startswith("```"):
        answer = answer.strip("`").strip()
    if not answer:
        raise RuntimeError("Answer model returned an empty response.")
    return answer


def format_eta(seconds: float) -> str:
    """Convert seconds into a compact human-readable ETA like '~2m 15s'."""
    rounded = max(1, int(round(seconds)))
    if rounded < 60:
        return f"~{rounded}s"
    minutes, remaining_seconds = divmod(rounded, 60)
    if remaining_seconds == 0:
        return f"~{minutes}m"
    return f"~{minutes}m {remaining_seconds}s"


def estimate_answer_time_remaining(stage: str, stage_eta_seconds: dict[str, float]) -> float:
    """Compute the remaining ETA for the active answer pipeline based on the current stage and estimates."""
    if stage not in ANSWER_PROGRESS_STAGE_ORDER:
        return sum(
            max(stage_eta_seconds.get(name, DEFAULT_ANSWER_STAGE_ETA_SECONDS[name]), 0.1)
            for name in ANSWER_PROGRESS_STAGE_ORDER
        )
    start_idx = ANSWER_PROGRESS_STAGE_ORDER.index(stage)
    return sum(
        max(stage_eta_seconds.get(name, DEFAULT_ANSWER_STAGE_ETA_SECONDS[name]), 0.1)
        for name in ANSWER_PROGRESS_STAGE_ORDER[start_idx:]
    )


def update_answer_stage_eta(
    stage: str, elapsed_seconds: float, stage_eta_seconds: dict[str, float]
) -> None:
    """Update a stage ETA using the observed runtime and a weighted smoothing factor."""
    baseline = max(stage_eta_seconds.get(stage, DEFAULT_ANSWER_STAGE_ETA_SECONDS.get(stage, 1.0)), 0.1)
    observed = max(elapsed_seconds, 0.1)
    alpha = ANSWER_STAGE_ETA_SMOOTHING_ALPHA
    stage_eta_seconds[stage] = ((1.0 - alpha) * baseline) + (alpha * observed)


def answer_question(
    question: str,
    index: list[tuple[Chunk, dict[str, float]]],
    idf: dict[str, float],
    top_k: int,
    min_score: float,
    ollama_url: str,
    lang_model: str,
    answer_model: str,
    answer_prompt_version: int | None,
    progress_callback: Callable[[str], None] | None = None,
    stage_eta_seconds: dict[str, float] | None = None,
) -> tuple[LanguageScores, str]:
    """Classify the question, retrieve evidence, and answer grounded in the local legal corpus."""
    eta_state = stage_eta_seconds or dict(DEFAULT_ANSWER_STAGE_ETA_SECONDS)

    def report_progress(stage: str, message: str) -> None:
        """Emit a user-facing progress line with the adaptive ETA for the active stage."""
        if not progress_callback:
            return
        eta_seconds = estimate_answer_time_remaining(stage, eta_state)
        progress_callback(f"{message} (est. {format_eta(eta_seconds)} remaining)")

    report_progress("language_detection", "Detecting question language...")
    language_started_at = time.perf_counter()
    scores = detect_question_language(question, ollama_url=ollama_url, model=lang_model)
    if stage_eta_seconds is not None:
        update_answer_stage_eta(
            "language_detection", time.perf_counter() - language_started_at, stage_eta_seconds
        )
    language = scores.top_label()
    if language == "other":
        return (
            scores,
            "Please ask your question in English or French only.",
        )

    report_progress("evidence_retrieval", "Retrieving relevant evidence from local documents...")
    retrieval_started_at = time.perf_counter()
    retrieval_top_k = max(top_k, 10) if is_recovery_measures_question(question) else top_k
    evidence = retrieve_evidence(question, index, idf, top_k=retrieval_top_k)
    if stage_eta_seconds is not None:
        update_answer_stage_eta(
            "evidence_retrieval", time.perf_counter() - retrieval_started_at, stage_eta_seconds
        )
    if not evidence or evidence[0][1] < min_score:
        if language == "fr":
            return (
                scores,
                "Je ne trouve pas assez de preuves dans les documents locaux pour répondre avec confiance. "
                "Veuillez ajouter des documents plus pertinents dans data/ ou reformuler votre question.",
            )
        return (
            scores,
            "I can’t find enough evidence in the local documents to answer confidently. "
            "Please add more relevant documents to data/ or rephrase your question.",
        )

    report_progress("answer_generation", "Generating grounded answer from retrieved evidence...")
    selected_prompt_version = resolve_answer_prompt_version(question, answer_prompt_version)
    generation_started_at = time.perf_counter()
    generated_answer = generate_grounded_answer(
        question=question,
        language=language,
        evidence=evidence,
        ollama_url=ollama_url,
        model=answer_model,
        prompt_version=selected_prompt_version,
    )
    if stage_eta_seconds is not None:
        update_answer_stage_eta(
            "answer_generation", time.perf_counter() - generation_started_at, stage_eta_seconds
        )
    return (
        scores,
        generated_answer,
    )


def summarize_indexed_documents(files: list[Path], chunks: list[Chunk]) -> tuple[int, int]:
    """Return the total pages and token count for the indexed local document set."""
    total_pages = sum(
        count_document_pages(file_path)
        for file_path in files
        if file_path.suffix.lower() in {".pdf", ".docx"}
    )
    total_tokens = sum(len(chunk.tokens) for chunk in chunks)
    return total_pages, total_tokens


def build_document_summary_excerpt(file_chunks: list[Chunk], max_chars_per_excerpt: int = 420) -> str:
    """Build a small excerpt from the start, middle, and end of a document for summary generation."""
    if not file_chunks:
        return ""

    selected_indices = sorted({0, len(file_chunks) // 2, len(file_chunks) - 1})
    excerpts: list[str] = []
    for i in selected_indices:
        snippet = normalize_whitespace(file_chunks[i].text)
        if not snippet:
            continue
        clipped = snippet[:max_chars_per_excerpt]
        excerpts.append(f"[Excerpt {len(excerpts) + 1}] {clipped}")
    return "\n\n".join(excerpts)


def detect_document_identity_summary(excerpt: str) -> str | None:
    """Identify recognisable legal documents, especially bilingual SARA material, from a text excerpt."""
    lower_excerpt = excerpt.lower()
    has_sara_en = "species at risk act" in lower_excerpt
    has_sara_fr = "loi sur les espèces en péril" in lower_excerpt or "especes en peril" in lower_excerpt
    if has_sara_en and has_sara_fr:
        return (
            "This document is the Species at Risk Act (SARA), presented in English and French, "
            "covering legal protections, listing, and recovery requirements for at-risk wildlife."
        )
    if has_sara_en or has_sara_fr:
        return (
            "This document appears to be the Species at Risk Act (SARA), covering legal protections, "
            "listing, and recovery requirements for at-risk wildlife."
        )
    return None


def fallback_document_summary(file_path: Path, excerpt: str) -> str:
    """Generate a conservative summary when the document cannot be confidently identified from the excerpt."""
    identity_summary = detect_document_identity_summary(excerpt)
    if identity_summary:
        return identity_summary

    lower_excerpt = excerpt.lower()

    if "act" in lower_excerpt or "regulation" in lower_excerpt or "minister" in lower_excerpt:
        if "à jour" in lower_excerpt or "current to" in lower_excerpt:
            return (
                "This appears to be a legal or policy source document, likely containing statutory "
                "requirements and administrative provisions relevant to species-at-risk decisions."
            )
        return (
            "This appears to be a legal or policy source document with statutory requirements and "
            "administrative provisions relevant to the chatbot's domain."
        )

    return (
        f"This document ({file_path.name}) contains domain reference material used by the chatbot "
        "to ground answers from local evidence."
    )


def generate_document_executive_summary(
    file_path: Path,
    file_chunks: list[Chunk],
    ollama_url: str,
    model: str,
) -> tuple[str, bool]:
    """Generate a concise startup summary for one document using local excerpts and fallback detection."""
    excerpt = build_document_summary_excerpt(file_chunks)
    if not excerpt:
        return fallback_document_summary(file_path, excerpt), False
    identity_summary = detect_document_identity_summary(excerpt)
    if identity_summary:
        return identity_summary, False

    prompt = (
        "You create startup knowledge summaries for an offline QA system.\n"
        "Write exactly one concise executive-summary sentence (max 32 words).\n"
        "State what the document is and what knowledge it contains.\n"
        "Do not call it an 'executive summary' in the output sentence.\n"
        "If the excerpt shows both English and French text, explicitly mention it is bilingual.\n"
        "Do not include dates/version metadata unless central to the document identity.\n"
        "Do not repeat the file name in the output sentence.\n"
        "No markdown, no bullet points.\n\n"
        f"File name: {file_path.name}\n\n"
        "Document excerpts:\n"
        f"{excerpt}\n\n"
        "Executive summary:"
    )
    try:
        raw_summary = request_ollama_response(ollama_url=ollama_url, model=model, prompt=prompt)
    except RuntimeError:
        return fallback_document_summary(file_path, excerpt), False

    summary = normalize_whitespace(raw_summary.strip().strip("`"))
    if "file name:" in summary.lower():
        summary = summary.split("File name:", 1)[0].strip()
        summary = summary.split("file name:", 1)[0].strip()
    if not summary:
        return fallback_document_summary(file_path, excerpt), False
    return summary, True


def summarize_document_knowledge(
    files: list[Path],
    chunks: list[Chunk],
    ollama_url: str,
    model: str,
    progress_callback: Callable[[Path, int, int], None] | None = None,
) -> list[DocumentKnowledgeSummary]:
    """Summarize every document in the corpus so the startup output explains what the bot knows."""
    chunks_by_file: defaultdict[Path, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    summaries: list[DocumentKnowledgeSummary] = []
    for doc_index, file_path in enumerate(files, start=1):
        file_chunks = chunks_by_file.get(file_path, [])
        if not file_chunks:
            continue
        if progress_callback:
            progress_callback(file_path, doc_index, len(files))
        summary_text, _ = generate_document_executive_summary(
            file_path=file_path,
            file_chunks=file_chunks,
            ollama_url=ollama_url,
            model=model,
        )

        summaries.append(
            DocumentKnowledgeSummary(
                file_path=file_path,
                chunk_count=len(file_chunks),
                token_count=sum(len(chunk.tokens) for chunk in file_chunks),
                executive_summary=summary_text,
            )
        )
    return summaries


def print_startup_summary(
    files: list[Path], chunks: list[Chunk], ollama_url: str, answer_model: str
) -> None:
    """Display a startup summary that tells the user what knowledge has been loaded into the bot."""
    print("[Startup] Generating executive knowledge summaries...", flush=True)
    total_pages, total_tokens = summarize_indexed_documents(files, chunks)
    doc_summaries = summarize_document_knowledge(
        files,
        chunks,
        ollama_url=ollama_url,
        model=answer_model,
        progress_callback=lambda path, idx, total: print(
            f"[Startup] Summarizing document {idx}/{total}: {path.name}",
            flush=True,
        ),
    )
    print("Phase 1 local chatbot ready.")
    print(
        f"Indexed {len(files)} document(s), {total_pages} page(s) from PDF/DOCX files, "
        f"and {total_tokens} token(s)."
    )
    print("Knowledge loaded from documents:")
    for summary in doc_summaries:
        print(f"- {summary.file_path.name}: {summary.executive_summary}")
        print(f"  Coverage: {summary.chunk_count} chunk(s), {summary.token_count} token(s).")
    print("You can ask questions about these documents in either English or French.")
    print("Type your question. Type 'exit' or 'quit' to stop.\n")


def main() -> int:
    """Run the local chatbot, load documents, and answer user questions grounded in the indexed corpus."""
    args = parse_args()
    docs_dir = args.docs_dir

    print("[Startup] Initializing local document workspace...", flush=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    print("[Startup] Ensuring default reference document is available...", flush=True)
    try:
        default_pdf_path, was_downloaded = ensure_default_pdf(docs_dir)
    except (error.URLError, OSError, RuntimeError) as exc:
        print(f"Error: failed to ensure default PDF in {docs_dir}: {exc}", file=sys.stderr)
        return 1
    if was_downloaded:
        print(f"Downloaded default reference PDF: {default_pdf_path}")

    results_dir = DEFAULT_RESULTS_DIR
    print("[Startup] Resolving language model...", flush=True)
    selected_language_model = resolve_language_model(
        args.ollama_url, args.lang_model, fallback_model=DEFAULT_LANGUAGE_MODEL
    )
    selected_answer_model = args.answer_model or selected_language_model
    if not args.skip_benchmark:
        print("[Startup] Running language benchmark (this may take a moment)...", flush=True)
        try:
            assert_model_can_classify(selected_language_model, args.ollama_url, results_dir)
        except RuntimeError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 2
        print(f"Language benchmark passed for model: {selected_language_model}")

    print("[Startup] Scanning documents directory...", flush=True)
    files = collect_supported_files(docs_dir)
    if not files:
        print(
            f"Error: no supported documents found in {docs_dir} "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})",
            file=sys.stderr,
        )
        return 1

    print(f"[Startup] Building text chunks from {len(files)} document(s)...", flush=True)
    chunks = build_chunks(files, chunk_size=args.chunk_size)
    if not chunks:
        print("Error: documents were found, but no usable text could be extracted.", file=sys.stderr)
        return 1

    print("[Startup] Building retrieval index...", flush=True)
    idf = build_idf(chunks)
    index = prepare_index(chunks, idf)
    print_startup_summary(
        files,
        chunks,
        ollama_url=args.ollama_url,
        answer_model=selected_answer_model,
    )
    answer_stage_eta_seconds = dict(DEFAULT_ANSWER_STAGE_ETA_SECONDS)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        selected_prompt_version = resolve_answer_prompt_version(
            question, args.answer_prompt_version
        )
        full_eta = estimate_answer_time_remaining("language_detection", answer_stage_eta_seconds)
        print(f"[Answer] Processing your question... (est. {format_eta(full_eta)} remaining)", flush=True)
        try:
            language_scores, answer = answer_question(
                question=question,
                index=index,
                idf=idf,
                top_k=max(args.top_k, 1),
                min_score=max(args.min_score, 0.0),
                ollama_url=args.ollama_url,
                lang_model=selected_language_model,
                answer_model=selected_answer_model,
                answer_prompt_version=selected_prompt_version,
                progress_callback=lambda message: print(f"[Answer] {message}", flush=True),
                stage_eta_seconds=answer_stage_eta_seconds,
            )
        except RuntimeError as exc:
            print(f"\nBot error: {exc}\n")
            continue
        if args.debug_long:
            print(
                "\nDebug: "
                f"classified_language={language_scores.top_label()} "
                f"en={language_scores.en:.3f} "
                f"fr={language_scores.fr:.3f} "
                f"other={language_scores.other:.3f} "
                f"answer_prompt_version={selected_prompt_version}"
            )
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())
