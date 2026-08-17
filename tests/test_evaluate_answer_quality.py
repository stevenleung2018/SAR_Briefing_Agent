import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urllib import error

import evaluate_answer_quality as evaluator


def test_load_gemini_api_key_reads_expected_value(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY='abc123'\n", encoding="utf-8")

    assert evaluator.load_gemini_api_key(env_file) == "abc123"


def test_parse_gemini_json_response_extracts_candidate_text():
    payload = (
        b'{"candidates":[{"content":{"parts":[{"text":"{\\"weighted_claim_f1\\": 0.75, '
        b'\\"unsupported_claim_rate\\": 0.1, \\"high_severity_contradiction\\": 0.0}"}]}}]}'
    )

    parsed = evaluator.parse_gemini_json_response(payload)
    assert parsed["weighted_claim_f1"] == 0.75
    assert parsed["unsupported_claim_rate"] == 0.1


def test_call_gemini_json_retries_with_default_model(monkeypatch):
    calls = []

    def fake_try_call_gemini(api_key, model, prompt):
        calls.append(model)
        if model == "demo-model":
            raise error.HTTPError(
                url="https://example.com",
                code=404,
                msg="not found",
                hdrs=None,
                fp=None,
            )
        return {"weighted_claim_f1": 1.0, "unsupported_claim_rate": 0.0, "high_severity_contradiction": 0.0}

    monkeypatch.setattr(evaluator, "try_call_gemini", fake_try_call_gemini)
    result = evaluator.call_gemini_json("secret", "demo-model", "prompt")

    assert calls == ["demo-model", "gemini-flash-latest"]
    assert result["weighted_claim_f1"] == 1.0


def test_save_results_creates_json_and_csv(tmp_path):
    rows = [{
        "iteration": 1,
        "prompt_version": 2,
        "weighted_claim_f1": 0.8,
        "unsupported_claim_rate": 0.1,
        "high_severity_contradiction": 0.0,
        "passes": True,
    }]

    json_path, csv_path = evaluator.save_results(tmp_path, rows)

    assert json_path.exists()
    assert csv_path.exists()
    assert "iteration" in csv_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["passes"] is True
