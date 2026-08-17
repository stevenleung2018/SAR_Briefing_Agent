import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot_phase1 import (
    Chunk,
    build_answer_prompt,
    build_chunks,
    build_idf,
    collect_supported_files,
    prepare_index,
    retrieve_evidence,
    resolve_answer_prompt_version,
    tokenize,
)


def test_collect_supported_files_only_includes_supported_extensions(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "one.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "two.md").write_text("hi", encoding="utf-8")
    (tmp_path / "nested" / "three.pdf").write_bytes(b"fake-pdf")
    (tmp_path / "ignore.csv").write_text("nope", encoding="utf-8")

    files = collect_supported_files(tmp_path)
    assert {path.name for path in files} == {"one.txt", "two.md", "three.pdf"}


def test_build_idf_and_retrieve_evidence_prioritizes_species_chunk():
    species_text = "White Sturgeon is a species in danger and critical habitat is discussed."
    unrelated_text = "The forest contains cedar and pine, unrelated to species assessment."
    chunks = [
        Chunk(file_path=Path("species.txt"), chunk_id=0, text=species_text, tokens=tokenize(species_text)),
        Chunk(file_path=Path("other.txt"), chunk_id=0, text=unrelated_text, tokens=tokenize(unrelated_text)),
    ]

    idf = build_idf(chunks)
    index = prepare_index(chunks, idf)
    results = retrieve_evidence("Is White Sturgeon a species at risk?", index, idf, top_k=1)

    assert results[0][0].file_path.name == "species.txt"
    assert results[0][1] > 0.0


def test_build_answer_prompt_stays_in_english_and_mentions_recovery_requirements():
    prompt = build_answer_prompt(
        "What recovery measures are needed for Boreal Caribou?",
        "en",
        "[Evidence 1] score=0.91 source=doc.txt\nThe law requires recovery measures.",
        1,
    )

    assert "Respond only in English." in prompt
    assert "Because this is a recovery-measures question" in prompt
    assert "## Direct answer" in prompt


def test_resolve_answer_prompt_version_chooses_recovery_variant_when_needed():
    assert resolve_answer_prompt_version("What recovery measures are required?", None) == 1
    assert resolve_answer_prompt_version("Is White Sturgeon a species at risk?", None) == 2
    assert resolve_answer_prompt_version("Any question", 3) == 3


def test_build_chunks_creates_chunk_objects_from_txt_files(tmp_path):
    doc = tmp_path / "sample.txt"
    doc.write_text("Alpha beta alpha gamma delta", encoding="utf-8")

    chunks = build_chunks([doc], chunk_size=40)

    assert len(chunks) == 1
    assert chunks[0].file_path == doc
    assert chunks[0].tokens == ["alpha", "beta", "alpha", "gamma", "delta"]
