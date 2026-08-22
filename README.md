# Species at Risk Briefing Agent

![Last Commit](https://img.shields.io/github/last-commit/stevenleung2018/SAR_Briefing_Agent) ![CI](https://github.com/stevenleung2018/SAR_Briefing_Agent/actions/workflows/ci.yml/badge.svg)

Species at Risk Briefing Agent - an agentic AI project

(Disclaimers at the bottom of the page.)

## Executive summary

The Species at Risk Briefing Agent is an agentic AI chatbot intended to bridge this gap: it takes a plain-language question and returns a clear, evidence-grounded answer from the documents available in the current phase, so that no training or prior familiarity with the Registry is required. The project is being built in three phases, starting with a local chatbot that can run offline after its initial document download over a small document set (Phase 1), then adding agentic access to the live Public Registry and other online resources (Phase 2), and finally deploying to the public cloud so it is available to everyone (Phase 3).

## Objectives of this project
- Create a scalable application with a natural language interface that bridges the gap between the Species at Risk Public Registry's documents and listings and the direct, real-life questions the public actually has.
- Require no training or prior expertise to use: a member of the general public should be able to ask a plain-language, real-world question (e.g. "Can I do X in this area? Is it affecting any at-risk animals?") and get a clear, understandable answer, without needing to know where to look, which documents apply, or how to interpret them.
- Ground every answer in authoritative, up-to-date evidence from the Species at Risk Public Registry and related official sources, rather than relying on the model's general knowledge, so answers are trustworthy for real-life decisions.
- Be explicit about uncertainty and limitations: clearly state when the available evidence does not answer the question, direct the user to the appropriate official resource or contact, and avoid presenting speculation as fact.
- Support bilingual (English and French) interactions, reflecting the bilingual nature of the source registry and its public audience.
- The target user groups include:
    - Members of the general public planning an activity (e.g. construction, land use, recreation) who want to know if at-risk species may be affected in their area.
    - Landowners, contractors, and small business operators seeking a quick, plain-language answer about potential Species at Risk Act obligations before proceeding with an activity.
    - Community members, students, and researchers who want an accessible way to explore information about species at risk without needing prior familiarity with the registry or legal/technical terminology.

This project has 3 phases:

Phase 1: Local offline chatbot
Phase 2: Local chatbot with offline LLM model, offline documents and agentic access to additional resources online, including the [Species at Risk Public Registry](https://www.canada.ca/en/environment-climate-change/services/species-risk-public-registry.html).
Phase 3: Chatbot on a public cloud so that it is more powerful and available to everyone.

## Local Chatbot (Phase 1)

Phase 1 implements a local, offline chatbot that answers questions using only documents in a local `data/` folder.  The model being used is also completely local and offline.

### Supported document types

- Markdown (`.md`)
- Text (`.txt`)
- PDF (`.pdf`)

### Setup

Python 3.12 is recommended especially if you are using Windows.  I also recommend that you create a virtual environment with the platform of your choice, e.g. conda, venv, uv.

Please install [ollama](https://ollama.com/) yourself.

If you have not already downloaded the model, you can do that with the following command (using "llama3.1:latest" as the example):

```bash
ollama pull llama3.1:latest
```

For installing the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

### Run

Launch Ollama server.

```bash
ollama serve
```

Launch the chatbot.

```bash
python3 src/chatbot_phase1.py
```

Optional arguments:

```bash
python3 src/chatbot_phase1.py --data-dir data --top-k 3 --min-score 0.08 --chunk-size 900 --answer-prompt-version 1
```

### Language benchmark script

You can run the language-classification benchmark independently:

```bash
python3 src/benchmark_language_model.py --results-dir model_test_results
```

- Tests whether Ollama's default available model can classify 20 sample questions: 10 in English and 10 in French.
- Uses the LLM as a language-classifier baseline before it is used for chatbot interactions.
- Saves each tested model's result as a model-specific JSON cache file in the model_test_results directory and reuses it on future chatbot starts, so the same model is not retested every time.
- Requires at least 8 of 10 significant classifications in each language; otherwise, the chatbot does not launch and the user should use a better Ollama model.
- Supports `--force` when the benchmark should be rerun instead of using the cached result.

### Answer quality evaluation script

For fine-tuning the chatbot so that it gives the best answers with available information, you may run the iterative prompt evaluation (up to 10 prompt variants) against a reference answer, scored with Gemini:

```bash
python3 src/evaluate_answer_quality.py --results-dir model_test_results --iterations 10
```

- Follows the Ragas framework for evaluating retrieval-augmented generation. Ragas was chosen because it is the most widely cited framework in this area and provides an existing Python package (`ragas`, included in `requirements.txt`).
- Uses Google Gemini as the second LLM evaluator during prompt fine-tuning, scoring each candidate answer against the reference answer and retrieved-evidence intent.
- Compares prompt variants primarily by **weighted claim-level F1**, paired with **unsupported-claim rate** and a **high-severity contradiction gate**.
- Requires a prompt to meet the weighted claim-level F1 threshold and stay below the configured high-severity contradiction gate; unsupported-claim rate is reported as a companion metric for analysis.
- Stops iterating when a passing prompt is found; otherwise, it records the available prompt variants and reports the candidate with the highest weighted claim-level F1.
- Writes detailed JSON and CSV evaluation outputs, including the candidate answer and pass/fail metrics, to the model_test_results directory.
- Cites: S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAS: Automated Evaluation of Retrieval Augmented Generation,” *Proceedings of the EACL 2024 Demonstration Track*, pp. 150–158, 2024. <https://doi.org/10.18653/v1/2024.eacl-demo.16>

### Notes

- The chatbot is evidence-grounded: it only answers from local documents.
- If it cannot find enough local evidence, it will say so instead of guessing.
- On startup, it ensures `data/S-15.3.pdf` exists by downloading it from `https://laws-lois.justice.gc.ca/PDF/S-15.3.pdf` if missing.
- The language classifier tries to use Ollama's default model automatically. If Ollama does not provide a default, it falls back to `llama3:latest`.
- Before launching the chatbot, it runs a 20-question EN/FR benchmark. The benchmark requires at least 8/10 correct and significant classifications in each language; otherwise it exits and tells the user the available Ollama model is not strong enough.
- If `other` has the highest score, it asks the user to submit the question in one of the two official languages.
- While generating each answer, the chatbot prints progress updates for language detection, evidence retrieval, and grounded answer generation, including a rough estimated time remaining that adapts over time.
- By default, no debug output is printed during Q&A. For debugging, use `--debug-long` to print classification details (`classified_language`, `en`, `fr`, `other`).
- Ensure Ollama is running locally before chat use (default API: `http://127.0.0.1:11434/api/generate`).
- You can override language-classification settings with `--ollama-url` and `--lang-model`, select a separate answer model with `--answer-model`, and skip startup benchmark with `--skip-benchmark` if desired.
- Prompt routing is automatic by default: recovery-measures questions use prompt version 1, while general/status questions use prompt version 2. Use `--answer-prompt-version` to force a fixed prompt variant.
- Benchmark cache files and answer-evaluation outputs are saved under `model_test_results/`.

## Disclaimers

1. This is a personal project and my employer has no involvement in this. It is therefore not representative of the position of my employer. All opinions are mine.
2. All information in this repository or that the code uses is completely in the public domain. No protected or secret information is used and the code does not require any such information to work.
3. Even though I have reasonable methodology in fine-tuning the code to improve the quality of the answers, just as any other AI tool, the tool may make mistakes.
