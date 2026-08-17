import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import benchmark_language_model


def test_main_prints_benchmark_json(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_language_model, "resolve_language_model", lambda *args, **kwargs: "demo-model")
    monkeypatch.setattr(
        benchmark_language_model,
        "get_or_run_language_benchmark",
        lambda *args, **kwargs: {"passes": True, "en_significant": 10, "fr_significant": 9},
    )
    monkeypatch.setattr(sys, "argv", ["benchmark_language_model.py"])

    exit_code = benchmark_language_model.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"passes": true' in captured.out
