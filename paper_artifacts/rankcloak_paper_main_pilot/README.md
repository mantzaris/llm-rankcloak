# RankCloak Paper Main Pilot Artifact Package

This directory is a small manuscript-preparation package copied from the staged
`results/rankcloak_paper_main_pilot/` and `results/rankcloak_paper_analysis/`
outputs.

## Contents

- `PAPER_RESULTS_SUMMARY.md`: paper-oriented narrative summary for the current
  staged pilot outputs.
- `PAPER_COMPARISON_TABLES.md`: manuscript-friendly table drafts.
- `PAPER_FIGURE_INDEX.md`: generated figure index and caption placeholders.
- `SUMMARY.md` and `summary.json`: machine-readable and human-readable staged
  run summaries.
- `MANIFEST.json`: reproducibility metadata for the staged pilot.
- `statistical_summary.csv`: bootstrap summary rows for available trial rows.
- `effect_size_summary.csv`: simple effect-size rows for available comparisons.
- `detector_baseline.csv`: lightweight feature-only detector baseline results.
- `PAPER_ANALYSIS_SUMMARY.md` and `paper_analysis_summary.json`: cross-pilot
  aggregation summary.
- `figures/`: selected generated paper-pilot figures.

The local GGUF model file is not included.

## Current Status

This package is a partial staged paper-main-pilot package, not the full frozen
paper-main matrix. The current staged pilot contains:

- 12 payload rows.
- 12 direct rank-pressure rows.
- 36 codec-comparison rows.
- 20 non-segmented RankCloak trials out of 96 planned.
- 7 segmented RankCloak trials out of 24 planned.
- 15 greedy baseline examples.
- 240 detector dataset rows.
- 57 detector result rows.
- 97 statistical summary rows.
- 14 effect-size rows.

Exact recovery in the completed staged pilot rows is 26 pass and 1 fail.
The observed failure is in the experimental lead-in segmented variant,
`segmented_hex_multi_topic_leadin8_sentence_tail_filtered`.

## Paper Use

Main paper candidates:

- Payload representation and direct rank-pressure diagnostics.
- Exact recovery under tested exact-copy conditions.
- Bounded-rank versus direct-subword framing.
- Forced-prefix versus full-message metric separation.
- Artifact and detector metrics as lightweight pilot evidence.

Supplement candidates:

- Full generated cover examples.
- Detector split details.
- Effect-size and bootstrap tables from partial pilot data.
- Cross-pilot aggregation artifacts.

## Supported Claims

- The implemented methods recover deterministic synthetic payloads exactly under
  the tested exact-copy conditions.
- Payload representation changes rank count and rank pressure.
- Bounded-rank and hex-nibble encodings provide predictable rank constraints.
- Segmented variants separate payload-bearing forced-span metrics from
  full-message metrics.
- The detector outputs are lightweight feature-only baselines.

## Unsupported Claims

- No encryption, key exchange, authentication, signing, credential handling,
  cryptographic security, or undetectability is claimed.
- No edit robustness or paraphrase robustness is claimed.
- No cross-model generalization is claimed.
- No broad human naturalness conclusion is supported by this package.

## Rerun Commands

Continue non-segmented generation:

```bash
python3 scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

Continue segmented generation:

```bash
python3 scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

After additional generation batches:

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
```

## Scope

All payloads are deterministic synthetic examples. Exact recovery requires the same
model, tokenizer, quantization, prompt templates, rank ordering, payload codec,
segmentation and tail rules, token filter, and exact unmodified public text.
