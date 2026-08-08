# GPU Support And Validation

## Status

GPU execution was implemented and locally validated on 2026-08-08. CPU execution
remains the default, so existing commands retain their previous behavior.

The new --n-gpu-layers option controls llama.cpp layer offload:

| Value | Behavior |
| --- | --- |
| 0 | CPU-only execution; the default |
| -1 | Offload all model layers to the GPU |
| positive integer | Offload that many layers |
| less than -1 | Rejected as invalid |

The validation machine exposed an NVIDIA RTX 5000 Ada Generation GPU. Full offload
reported all 33 model layers loaded on the GPU.

## Public Interface

The option is available through both command entry points:

~~~bash
rankcloak run --profile smoke --n-gpu-layers -1 --overwrite

python3 scripts/run_experiment.py \
  --profile smoke \
  --n-gpu-layers -1 \
  --overwrite
~~~

Use CUDA_VISIBLE_DEVICES when a machine has more than one GPU:

~~~bash
CUDA_VISIBLE_DEVICES=0 rankcloak run \
  --profile smoke \
  --n-gpu-layers -1 \
  --overwrite
~~~

An explicit GPU request fails early if the installed llama-cpp-python package lacks
GPU-offload support. CPU-only execution does not require CUDA and continues to work
when the CUDA runtime is absent.

## Code Map

| Location | New responsibility |
| --- | --- |
| rankcloak/model_io.py | CUDA library discovery/loading, GPU capability check, layer-offload configuration, deterministic defaults, model metadata, and full KV-cache reset |
| rankcloak/experiments.py | Propagates n_gpu_layers into model loading and run configuration |
| rankcloak/cli.py | Defines and validates the --n-gpu-layers command option |
| rankcloak/reproducibility.py | Records GPU/offload state, CUDA package versions, and bounded model hashing |
| rankcloak/paper_suite.py | Derives segmented recovery-failure notes from observed data |
| pyproject.toml | Adds a pinned CUDA dependency extra |
| tests/test_gpu_support.py | Exercises GPU argument propagation, validation, manifest fields, and capability handling without loading the model |
| tests/test_paper_suite_resume.py | Covers the data-derived failure-note behavior |

## CUDA Runtime Loading

The reference repository demonstrated that its CUDA-enabled llama.cpp wheel needed
the NVIDIA pip-package libraries made available before importing llama_cpp. RankCloak
now performs that discovery locally in rankcloak.model_io.preload_pip_cuda_libraries.

The loader searches the installed NVIDIA Python-package namespace and preloads, in
dependency order:

1. libcudart
2. libcublasLt
3. libcublas

This avoids requiring a persistent LD_LIBRARY_PATH change in the user's shell. It
also preserves the CPU path: missing libraries are harmless until GPU offload is
explicitly requested.

## Rank-Stability Controls

RankCloak encodes information through exact candidate ranks, so small backend
differences can change the recovered payload even when the generated prose remains
plausible. GPU initialization therefore sets these defaults before llama.cpp is
loaded:

~~~text
GGML_CUDA_DISABLE_GRAPHS=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
~~~

Existing user values take precedence. Model reset now also clears the complete
llama.cpp KV cache, preventing stale prompt state from leaking between trials.

These controls reduce avoidable variation. They do not promise bit-identical CPU and
GPU logits, or bit-identical output across llama.cpp builds, drivers, quantizations,
and model files.

## GPU Environment

The validated environment used Python 3.10, llama-cpp-python 0.3.23, CUDA 12.4
runtime libraries, and the cu124 wheel:

~~~bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  llama-cpp-python==0.3.23 \
  nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cublas-cu12==12.4.5.8 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
python -m pip install -e ".[dev,analysis]"
~~~

The dependency versions are also available as the cuda project extra. Keep the
llama.cpp wheel, CUDA major version, driver, and model file fixed for comparisons.

## Reproducibility Manifest

MANIFEST.json now records:

- requested n_gpu_layers, detected GPU-offload support, and inferred backend-active state;
- CUDA device-order, visibility, and deterministic environment settings;
- installed llama-cpp-python and NVIDIA CUDA package versions;
- model path, byte size, and SHA-256 when the model is at most 8 GiB.

The locally validated model was
Meta-Llama-3-8B-Instruct.Q4_K_M.gguf with SHA-256
86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7.

## Paper-Matched Validation

The GPU validation used the exact trial IDs present in the historical
paper-main-pilot package, including segmented rows 1-5 and 7-8. Downstream baseline,
detector, statistical, and effect-size stages were regenerated from the GPU rows.

| Measure | Historical paper package | GPU validation |
| --- | ---: | ---: |
| Non-segmented exact recovery | 20/20 | 19/20 |
| Segmented exact recovery | 6/7 | 5/7 |
| Overall exact recovery | 26/27 | 24/27 |
| Baseline examples | 22 | 20 |
| Detector-feature rows | 272 | 264 |
| Detector result rows | 57 | 57 |
| Statistical summary rows | 97 | 97 |
| Effect-size rows | 14 | 14 |
| Detector ROC AUC range | 1.000-1.000 | 0.992-1.000 |

The validation reproduced the paper package's qualitative result directions:

- recovery remained high but not perfect;
- segmented generation remained the more fragile protocol;
- the lightweight detector still separated RankCloak and baseline samples almost
  perfectly;
- detector, bootstrap-statistics, and effect-size pipelines completed without
  structural failures.

Exact recovery was not identical. One historical non-segmented pass failed on the
GPU, the historical lead-in segmented failure passed on the GPU, and two historical
multi-topic segmented passes failed on the GPU. The specific artifact directions
therefore changed even though the aggregate conclusion did not.

## Complete GPU Pilot

After the paper-ID-matched check, the staged `paper-main-pilot-resume` orchestrator
was run in a fresh directory through the full planned pilot matrix. It completed every
generation and analysis stage:

| Protocol variant | Trials | Exact recoveries |
| --- | ---: | ---: |
| Non-segmented ASCII B=8 | 36 | 34 |
| Non-segmented ASCII B=16 | 36 | 33 |
| Non-segmented raw hex-nibble B=16 | 24 | 23 |
| Segmented single-topic | 8 | 6 |
| Segmented multi-topic | 8 | 8 |
| Segmented multi-topic with eight-token lead-in | 8 | 7 |
| Total | 120 | 111 |

The nine recovery failures are experiment outcomes, not runner failures. Every
planned trial ID is present, all generated rows and message artifacts remain
available for analysis, and the result tables explicitly report the failures.

The completed downstream package contains:

- 20 canonical length-matched greedy baselines;
- 668 cover-feature rows and 686 detector-dataset rows;
- 57 detector results, all with `status=ok`;
- 223 statistical summaries, all with `status=ok`;
- 14 effect-size results, all with `status=ok`;
- 10 nonempty paper figures.

The expanded run exposed three resume/reporting edge cases that the smaller subset
did not:

1. One generated token transiently replayed at rank 17 for a B=16 codec. Rank-domain
   decode errors are now retained as `exact_recovery=false` rows instead of causing
   the runner to omit the trial.
2. A changed median length target left an obsolete greedy baseline after resume.
   Baseline reconciliation now removes targets that are no longer in the current
   plan before regenerating detector and statistical artifacts.
3. The generated paper summary contained pilot-specific lead-in wording with a
   hard-coded failure count. Its lead-in result and run-scope text are now derived
   from the current result tables and resolved profile, with singular/plural
   regression coverage.

A final no-op resume skipped all 96 non-segmented and all 24 segmented trials, ran no
new baselines, and preserved the exact ID sets without duplicates. Full offload
(`-1`), one-layer partial offload (`1`), and CPU-only execution (`0`) were all
physically exercised. Two CPU context replays after KV-cache clearing produced
identical logits with maximum absolute delta 0.0.

## Why Exact Rows Differ

A diagnostic comparison found rank-pressure differences between the historical rows
and both newly generated CPU and GPU rows for all 12 sampled trials. The historical
manifest did not record a model hash. This means the observed mismatch cannot be
attributed to GPU execution alone: exact replay also depends on the model artifact,
llama.cpp build, tokenizer behavior, driver/backend, and generation configuration.

Treat the GPU output as a new, paper-matched validation dataset, not a byte-identical
replacement for the historical paper artifacts.

## Timing

For the same 27 trial IDs, recorded model time fell from 3546.73 seconds in the
historical CPU package to 138.12 seconds in the GPU validation, approximately a
25.7-fold speedup. This is an artifact-level comparison, not a controlled benchmark:
the runs were made on different machines and potentially different software/model
artifacts.

## Result Locations

- results/rankcloak_paper_gpu_validation/: GPU-generated validation package.
- results/rankcloak_paper_gpu_validation/GPU_VALIDATION_REPORT.md: exact
  historical/GPU comparison and caveats.
- results/rankcloak_paper_gpu_pilot_complete/: complete 96+24 GPU pilot with
  canonical baselines, detector/statistics/effects, summaries, and figures.
- results/rankcloak_paper_cpu_local_validation/: current-machine CPU diagnostic
  rows used to determine whether differences were GPU-specific.
- .paper/scientific_reports/: historical manuscript and supplemental result
  package; left unchanged.

## Verification

After the implementation:

- python3 -m compileall rankcloak scripts tests passed;
- python3 -m pytest passed all 93 tests;
- git diff --check passed.

The artifact audit also passed exact planned-ID, duplicate, critical-null, canonical
baseline, analysis-status, model-hash, backend-manifest, and nonempty-figure checks.
The GPU-focused unit tests use mocks, so the normal test suite remains runnable on a
CPU-only host.
