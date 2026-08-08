# Key Experiment Commands

This is the short command reference for reproducing the frozen paper-main baseline
and starting follow-up experiments on the validated machine. Run every command from
the repository root:

```text
/home/meow/Documents/repos/llm-rankcloak
```

The validated model is:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Its expected SHA-256 is:

```text
86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7
```

The authoritative completed result directory is:

```text
results/rankcloak_paper_gpu_main_rank_safe/
```

Do not use `--overwrite` on that directory. Use `--resume` to verify or safely
refresh it, and use a new output directory for every new experiment or independent
replication.

## 1. Verify The Environment

Confirm the GPUs and their indices:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

Confirm that the installed llama.cpp backend supports GPU offload:

```bash
.venv/bin/python -c "from rankcloak.model_io import llama_cpp_gpu_offload_supported; print(llama_cpp_gpu_offload_supported())"
```

The second command must print `True` for a GPU run. Confirm the exact model artifact:

```bash
sha256sum models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Show all runner options when needed:

```bash
.venv/bin/python scripts/run_experiment.py --help
```

## 2. Run A GPU Smoke Test

Use this before a new long run or after changing the CUDA, llama.cpp, driver, model,
or rank-generation code:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-smoke \
  --output-dir results/rankcloak_paper_gpu_smoke_check \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 \
  --overwrite
```

`--overwrite` is appropriate here only because this is a dedicated disposable smoke
directory. GPU mode automatically enables the rank-safe CUDA settings and
single-token `n_batch=1`, `n_ubatch=1` execution.

## 3. Verify Or Resume The Authoritative Paper-Main Run

This is the canonical command. It is safe to rerun because `--resume` skips all
existing stable trial IDs and refreshes the downstream summaries and figures:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_gpu_main_rank_safe \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 \
  --resume
```

On the completed package this command should report:

```text
paper-nonseg-generation: planned 475, existing 475, skipped 475, running 0
paper-segmented-generation: planned 75, existing 75, skipped 75, running 0
paper-baselines: planned 25, existing 25, running 0
```

## 4. Start An Independent Full GPU Replication

Use a new directory. The same command both starts and resumes the replication:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_gpu_main_replication_01 \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 \
  --resume
```

If interrupted, run the identical command again. Do not change the model, output
directory, GPU backend, prompt definitions, codec, or rank-stability settings during
one replication.

## 5. Run On CPU Instead

CPU is the default, but pass `0` explicitly in recorded research runs and use a
separate directory so CPU and GPU rows cannot be mixed:

```bash
.venv/bin/python scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_cpu_main_replication_01 \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers 0 \
  --resume
```

The backend switch is therefore:

- `--n-gpu-layers -1`: all model layers on the selected GPU.
- `--n-gpu-layers 0`: CPU only.
- `--n-gpu-layers N`: partial offload of `N` layers.

## 6. Run A Filtered GPU Batch

This example selects the frozen ASCII B=16, SHA-256 subset and runs at most five new
trials. Repeating the same command continues with the next five missing IDs:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_gpu_ascii_b16_sha256_batch \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 \
  --only-protocol-variant nonseg_ascii_b16 \
  --only-payload-class sha256_hex \
  --limit-trials 5 \
  --resume
```

Available selection controls are:

- `--only-protocol-variant NAME`
- `--only-payload-class NAME`
- `--only-prompt-name NAME`
- `--start-at-trial N`
- `--limit-trials N`

These options select rows from the existing frozen plan. A genuinely new payload,
prompt, codec, or protocol requires a new profile/configuration and a new output
directory; it should not be appended to the authoritative paper-main directory.

## 7. Monitor And Inspect A Run

Monitor GPU activity in another terminal:

```bash
watch -n 2 nvidia-smi
```

Inspect staged progress and the generated human-readable summary:

```bash
sed -n '1,120p' results/rankcloak_paper_gpu_main_rank_safe/RUN_PROGRESS.json
sed -n '1,160p' results/rankcloak_paper_gpu_main_rank_safe/SUMMARY.md
```

Check the recovery totals directly:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

root = "results/rankcloak_paper_gpu_main_rank_safe"
nonseg = pd.read_csv(f"{root}/paper_stegotext_trials.csv")
segmented = pd.read_csv(f"{root}/paper_segmented_trials.csv")
print("nonseg", len(nonseg), nonseg["exact_recovery"].sum())
print("segmented", len(segmented), segmented["exact_recovery"].sum())
PY
```

The authoritative output should print `nonseg 475 475` and `segmented 75 75`.

## 8. Run Repository Verification

Run these after changing experiment code and before trusting newly generated data:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q rankcloak scripts tests
git diff --check
git status --short
```

The validated repository state passed 97 tests. Review `MANIFEST.json` in every
result directory before comparing runs; model hash, backend, GPU selection, batching,
and CUDA settings are part of the exact-copy replay configuration.

## Operational Rules

- Keep the authoritative directory unchanged except for the canonical `--resume`
  command above.
- Give every new scientific question or independent replication a new output
  directory.
- Never resume CPU rows into a GPU directory or GPU rows into a CPU directory.
- Preserve `MANIFEST.json`, generated token IDs, JSONL examples, and all CSV rows.
- Do not compare or merge rows across different model hashes or inference backends as
  if they came from one experimental condition.
- Use `--overwrite` only for disposable smoke outputs or when replacement is
  deliberate and the target has been checked first.
