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

### NVIDIA GPU Environment

The validated GPU setup uses the CUDA 12.4 llama-cpp-python wheel and pinned NVIDIA
runtime packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  llama-cpp-python==0.3.23 \
  nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cublas-cu12==12.4.5.8 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
python -m pip install -e ".[dev,analysis]"
```
Verify that the installed backend exposes GPU offload:

```bash
python - <<'PY'
from rankcloak.model_io import llama_cpp_gpu_offload_supported
print("GPU offload supported:", llama_cpp_gpu_offload_supported())
PY
```


Request full model offload with `--n-gpu-layers -1`:

```bash
python3 scripts/run_experiment.py \
  --profile smoke \
  --n-gpu-layers -1 \
  --overwrite
```

On a multi-GPU host, select the device before starting the process:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_experiment.py \
  --profile smoke \
  --n-gpu-layers -1 \
  --overwrite
```

GPU-backed ranks can differ from CPU-backed ranks, and ranks can also differ across
model artifacts or llama.cpp builds. Preserve the generated manifest and compare
exact trial IDs instead of overwriting historical results. See
`notes/21_gpu_support_and_validation.md` for the implementation, validated
environment, result comparison, and limitations.

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

- `compileall`: passed after GPU support and paper-matched validation.
- `pytest`: 89 tests passed after GPU support and paper-matched validation.

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

Segmented protocol pilot:

```bash
python3 scripts/run_experiment.py \
  --profile segmented-protocol-pilot \
  --output-dir results/rankcloak_segmented_protocol_pilot \
  --overwrite
```

Segmented quality-controls pilot:

```bash
python3 scripts/run_experiment.py \
  --profile segmented-quality-controls \
  --output-dir results/rankcloak_segmented_quality_controls \
  --overwrite
```

Paper-suite smoke:

```bash
python3 scripts/run_experiment.py \
  --profile paper-smoke \
  --output-dir results/rankcloak_paper_smoke \
  --overwrite
```

Continue the staged paper-main-pilot:

```bash
python3 scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

```bash
python3 scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

Refresh paper-main-pilot analysis artifacts after generation:

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
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
- `PAPER_RESULTS_SUMMARY.md`, `PAPER_COMPARISON_TABLES.md`, and
  `PAPER_FIGURE_INDEX.md` for paper-suite outputs.
- `RUN_PROGRESS.json` for staged paper-suite resume status.
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
