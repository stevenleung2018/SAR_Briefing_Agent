#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from constants import DEFAULT_RESULTS_DIR
from language_classifier import (
    DEFAULT_OLLAMA_URL,
    get_or_run_language_benchmark,
    resolve_language_model,
)

DEFAULT_LANGUAGE_MODEL = "llama3:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or reuse the EN/FR language classification benchmark."
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate API URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--lang-model",
        default=None,
        help="Ollama model for classification. If omitted, resolve from Ollama tags.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for benchmark cache files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and rerun benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = resolve_language_model(
        args.ollama_url, args.lang_model, fallback_model=DEFAULT_LANGUAGE_MODEL
    )
    try:
        benchmark = get_or_run_language_benchmark(
            model_name=model,
            ollama_url=args.ollama_url,
            results_dir=args.results_dir,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
