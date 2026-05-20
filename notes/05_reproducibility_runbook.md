# Reproducibility Runbook

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llama,dev]"
python3 -m ipykernel install --user --name rankcloak --display-name "Python (rankcloak)"
```

If `llama-cpp-python` is difficult to build on the machine, install only the non-model dependencies and run codec-only tests:

```bash
pip install -e ".[dev]"
python3 scripts/run_experiment.py --profile codec-only --overwrite
```

## Local Model

Preferred model path:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Fallback path:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

The model is ignored by git. Do not commit GGUF files.

Download helper:

```bash
python3 - <<'PY'
from rankcloak.model_io import download_llama3_gguf
print(download_llama3_gguf())
PY
```

If Hugging Face requires authentication or license acceptance, use `huggingface-cli login`, accept any required model terms, or place the GGUF file manually under `models/llama3_8b/`.

## Standard Checks

Run before committing code changes:

```bash
python3 -m compileall rankcloak scripts
python3 -m pytest
```

Known recent status:

- `compileall`: passed after dialogue and payload granularity changes.
- `pytest`: 45 tests passed after dialogue and payload granularity changes.

## Core Experiment Commands

Fast no-model codec check:

```bash
python3 scripts/run_experiment.py --profile codec-only --overwrite
```

Fast model-backed smoke:

```bash
python3 scripts/run_experiment.py --profile smoke --overwrite
```

Small full-payload sweep:

```bash
python3 scripts/run_experiment.py \
  --profile small \
  --output-dir results/rankcloak_small_full \
  --overwrite
```

Strong prompt sweep:

```bash
python3 scripts/run_experiment.py \
  --profile strong-prompts \
  --output-dir results/rankcloak_strong_prompt_sweep \
  --overwrite
```

Dialogue key prompt pilot:

```bash
python3 scripts/run_experiment.py \
  --profile dialogue-key-pilot \
  --output-dir results/rankcloak_dialogue_key_pilot \
  --overwrite
```

Payload granularity pilot:

```bash
python3 scripts/run_experiment.py \
  --profile payload-granularity-pilot \
  --output-dir results/rankcloak_payload_granularity_pilot \
  --overwrite
```

## Reading Results

Start with:

- `summary.json` for machine-readable run status.
- `SUMMARY.md` for human-readable run status.
- `stegotext_recovery_trials.csv` for exact recovery and generation timings.
- `cover_text_features.csv` for plausibility/detectability proxy features.
- `cover_examples.jsonl` for generated RankCloak cover text.
- `baseline_cover_examples.jsonl` for ordinary greedy baseline cover text.
- `PROMPT_COMPARISON.md` or `DIALOGUE_PROMPT_COMPARISON.md` for curated manual inspection examples.
- `MANIFEST.json` for reproducibility metadata.

## Exact-Copy Requirements

Exact recovery currently requires:

- same GGUF model file;
- same tokenizer;
- same quantization;
- same prompt text;
- same deterministic rank ordering;
- same generated token ids;
- generated text preserved exactly if recovering from text;
- no platform edit, normalization, paraphrase, truncation, moderation rewrite, or formatting change.

## Git Notes

Small result files are intentionally committable:

- CSV;
- JSON;
- JSONL;
- Markdown;
- PNG figures.

Heavy artifacts are ignored:

- local model files;
- caches;
- virtual environments;
- build outputs;
- large binary result formats.

Before pushing, inspect:

```bash
git status --short
```

If the result directories are large, commit only the result subsets needed for the paper trail.
