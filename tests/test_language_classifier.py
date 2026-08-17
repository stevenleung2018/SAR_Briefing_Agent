import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest

from language_classifier import (
    LanguageScores,
    parse_language_label,
    parse_language_scores,
    request_ollama_response,
    resolve_language_model,
)


class DummyResponse:
    def __init__(self, payload: str):
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


def test_language_scores_normalize_and_boost():
    scores = LanguageScores(en=0.7, fr=0.2, other=0.1)
    boosted = scores.with_boost("fr", 0.3)
    assert scores.top_label() == "en"
    assert scores.normalized().en == pytest.approx(0.7)
    assert boosted.fr == pytest.approx(0.3846153846)
    assert boosted.top_label() == "en"
    assert scores.is_significant_top_language() is True


def test_parse_language_scores_accepts_json_with_wrapping_text():
    raw = "prefix {\"en\": 0.8, \"fr\": 0.15, \"other\": 0.05} suffix"
    parsed = parse_language_scores(raw)
    assert parsed.en == pytest.approx(0.8)
    assert parsed.fr == pytest.approx(0.15)
    assert parsed.other == pytest.approx(0.05)


def test_parse_language_label_raises_for_unknown_label():
    with pytest.raises(RuntimeError):
        parse_language_label("spanish")


def test_request_ollama_response_returns_text(monkeypatch):
    def fake_urlopen(req, timeout=90):
        return DummyResponse(json.dumps({"response": "{\"en\": 0.9, \"fr\": 0.1, \"other\": 0.0}"}))

    import language_classifier

    monkeypatch.setattr(language_classifier.request, "urlopen", fake_urlopen)
    response = request_ollama_response("http://example.com/api/generate", "llama3", "hello")
    assert response == '{"en": 0.9, "fr": 0.1, "other": 0.0}'


def test_resolve_language_model_uses_ollama_model_list(monkeypatch):
    def fake_urlopen(req, timeout=15):
        return DummyResponse(json.dumps({"models": [{"name": "demo-model:latest"}]}))

    import language_classifier

    monkeypatch.setattr(language_classifier.request, "urlopen", fake_urlopen)
    assert resolve_language_model("http://example.com/api/generate", None, "fallback-model") == "demo-model:latest"
