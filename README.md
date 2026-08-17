# Species at Risk Briefing Agent
Species at Risk Briefing Agent - an agentic AI project

(Note: This is a personal project and my employer has no involvement in this. It is therefore not representative of the position of my employer. All opinions are mine.)

## Local Chatbot (Phase 1)

Phase 1 implements a local, offline chatbot that answers questions using only documents in a local `docs/` folder.  The model being used is also complete local and offline. 

### Supported document types
- Markdown (`.md`)
- Text (`.txt`)
- PDF (`.pdf`)

### Setup
```bash
python3 -m pip install -r requirements.txt
```

### Run
```bash
python3 src/chatbot_phase1.py
```

Optional arguments:
```bash
python3 src/chatbot_phase1.py --docs-dir docs --top-k 3 --min-score 0.08 --chunk-size 900 --answer-prompt-version 1
```

### Language benchmark script
Run language-classification benchmark independently:
```bash
python3 src/benchmark_language_model.py --results-dir results
```

### Answer quality evaluation script
Run iterative prompt evaluation (up to 10 prompt variants) against a reference answer, scored with Gemini:
```bash
python3 src/evaluate_answer_quality.py --results-dir results --iterations 10
```

### Notes
- The chatbot is evidence-grounded: it only answers from local documents.
- If it cannot find enough local evidence, it will say so instead of guessing.
- On startup, it ensures `docs/S-15.3.pdf` exists by downloading it from `https://laws-lois.justice.gc.ca/PDF/S-15.3.pdf` if missing.
- The language classifier tries to use Ollama's default model automatically. If Ollama does not provide a default, it falls back to `llama3:latest`.
- Before launching the chatbot, it runs a 20-question EN/FR benchmark. The benchmark requires at least 8/10 correct and significant classifications in each language; otherwise it exits and tells the user the available Ollama model is not strong enough.
- If `other` has the highest score, it asks the user to submit the question in one of the two official languages.
- While generating each answer, the chatbot prints progress updates for language detection, evidence retrieval, and grounded answer generation, including a rough estimated time remaining that adapts over time.
- By default, no debug output is printed during Q&A. For debugging, use `--debug-long` to print classification details (`classified_language`, `en`, `fr`, `other`).
- Ensure Ollama is running locally before chat use (default API: `http://127.0.0.1:11434/api/generate`).
- You can override language-classification settings with `--ollama-url` and `--lang-model`, select a separate answer model with `--answer-model`, and skip startup benchmark with `--skip-benchmark` if desired.
- Prompt routing is automatic by default: recovery-measures questions use prompt version 1, while general/status questions use prompt version 2. Use `--answer-prompt-version` to force a fixed prompt variant.
- Benchmark cache files and answer-evaluation outputs are saved under `results/`.
