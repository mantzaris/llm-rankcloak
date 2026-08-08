# RankCloak GPU Validation Report

Date: 2026-08-08

## Verdict

The RTX run is scientifically consistent with the paper's main qualitative
findings, but it is not a bit-exact reproduction of the historical CPU pilot.
The exact trial IDs were matched: 20 non-segmented rows and 7 segmented rows.
GPU recovery was 24/27, compared with 26/27 in the published pilot.

These GPU results should remain a separate validation dataset. They should not
replace the manuscript's numeric tables without revising the paper's recovery
and provenance statements.

## GPU configuration

- Device: NVIDIA RTX 5000 Ada Generation, CUDA compute capability 8.9.
- Offload: all model layers requested with `--n-gpu-layers -1`.
- llama-cpp-python: 0.3.23 CUDA 12.4 wheel.
- Rank-stability controls: `GGML_CUDA_DISABLE_GRAPHS=1`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and a full KV-cache clear before each
  context replay.
- Device selection: `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`.
- Current GGUF SHA-256:
  `86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7`.

The full configuration is recorded in `MANIFEST.json`. NVIDIA documents
`:4096:8` as a deterministic cuBLAS workspace configuration:
https://docs.nvidia.com/cuda/archive/11.7.1/cublas/index.html

## Matched comparison

| Measure | Historical CPU pilot | GPU validation |
| --- | ---: | ---: |
| Non-segmented trial IDs | 20 | 20 |
| Non-segmented exact recovery | 20/20 | 19/20 |
| Segmented trial IDs | 7 | 7 |
| Segmented exact recovery | 6/7 | 5/7 |
| Overall exact recovery | 26/27 | 24/27 |
| Matched baseline rows | 22 | 20 |
| Detector dataset rows | 272 | 264 |
| Detector result rows | 57 | 57 |
| Statistical summary rows | 97 | 97 |
| Effect-size rows | 14 | 14 |
| Successful detector AUC range | 1.000--1.000 | 0.992--1.000 |

Recovery outcomes changed in four matched trials:

- `paper_sha256_hex_000`, non-segmented ASCII B=8, recipe-long prompt:
  pass on historical CPU, fail on GPU.
- `paper_sha256_hex_000`, segmented lead-in: fail on historical CPU, pass on
  GPU.
- `paper_sha256_hex_001`, segmented multi-topic: pass on historical CPU,
  fail on GPU.
- `paper_random_128_bit_hex_000`, segmented multi-topic: pass on historical
  CPU, fail on GPU.

## Findings that reproduce

- `paper_payloads.csv` and `paper_codec_comparison.csv` are byte-identical.
- The same 20 non-segmented and 7 segmented trial IDs were evaluated.
- ASCII B=16 has a lower mean token log-probability proxy than ASCII B=8:
  historical difference -0.377; GPU difference -0.263.
- Hex-nibble B=16 has a lower mean token log-probability proxy than ASCII
  B=16 on hex payloads: historical difference -0.728; GPU difference -0.765.
- Full segmented messages score substantially better than their payload-bearing
  forced prefixes because of the natural tails: historical gain 3.735; GPU
  gain 3.523.
- Segmented recovery remains a majority outcome (5/7).
- The feature-only detector conclusion reproduces: all 57 GPU detector rows
  completed, with AUCs from 0.992 to 1.000.

## Findings that do not reproduce exactly

- The manuscript's 20/20 non-segmented recovery statement is 19/20 on GPU.
- Segmented recovery is 5/7 rather than 6/7, and the failing variants differ.
- Direct-subword rank-pressure summaries differ for all 12 payloads.
- Two principal artifact-count effect directions reverse; artifact counts
  should not be transferred from the CPU tables to this GPU dataset.
- GPU-generated length buckets produce 20 rather than 22 matched baselines,
  and therefore 264 rather than 272 detector dataset rows.

## Provenance interpretation

The historical manifest records the same GGUF filename and byte size but omits
its SHA-256 because the file exceeded the former hash limit. The current run
records the full hash. A same-version local CPU diagnostic using the current
GGUF also differs from the paper's rank-pressure results for all 12 payloads;
the current CPU and GPU diagnostics differ for all 12 as well. Consequently,
the historical mismatch cannot be attributed to GPU arithmetic alone, and the
original model bytes cannot be proven identical.

This matters because RankCloak encodes information in exact token ordering:
small backend, build, quantization, or floating-point changes can move a token
across a rank boundary. GPU backend settings and the GGUF hash are therefore
part of the decoding configuration.

## Recorded model-time comparison

| Work | Historical CPU seconds | GPU seconds | Speedup |
| --- | ---: | ---: | ---: |
| Non-segmented generation | 979.22 | 53.14 | 18.4x |
| Non-segmented recovery | 906.31 | 29.52 | 30.7x |
| Segmented generation | 1142.98 | 47.75 | 23.9x |
| Segmented recovery | 518.23 | 7.71 | 67.2x |
| Combined | 3546.73 | 138.12 | 25.7x |

Timing values are the per-trial model times recorded in the result CSVs; they
exclude environment setup, model loading, hashing, plotting, and detector
analysis.

## Reproduction commands

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-diagnostics \
  --output-dir results/rankcloak_paper_gpu_validation \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 --resume

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_gpu_validation \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 --resume --limit-trials 20

# The published segmented subset uses planned rows 1--5 and 7--8.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_gpu_validation \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 --resume --start-at-trial 1 --limit-trials 5

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_gpu_validation \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 --resume --start-at-trial 7 --limit-trials 2

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python scripts/run_experiment.py \
  --profile paper-baselines \
  --output-dir results/rankcloak_paper_gpu_validation \
  --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
  --n-gpu-layers -1 --resume

.venv/bin/python scripts/run_experiment.py \
  --profile paper-detector \
  --output-dir results/rankcloak_paper_gpu_validation \
  --resume

.venv/bin/python scripts/run_experiment.py \
  --profile paper-statistics \
  --output-dir results/rankcloak_paper_gpu_validation \
  --resume
```
