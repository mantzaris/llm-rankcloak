# Paper Main Experiment Plan

## Purpose

The paper-main suite turns the pilot methodology into a locked results framework for
journal-style analysis. It compares implemented RankCloak payload representations and
protocol variants without introducing a broad new method search.

The suite remains an exact-copy measurement study over deterministic synthetic
payloads. It does not claim encryption, key exchange, authentication, signing,
credential handling, cryptographic security, or undetectability.

## Profiles

- `paper-main-pilot`: small enough to run first; writes to `results/rankcloak_paper_main_pilot/`.
- `paper-main`: frozen larger matrix for Alex to run when CPU time is available; writes to `results/rankcloak_paper_main/`.
- `paper-analysis`: aggregation profile that reads existing result directories without model generation; writes to `results/rankcloak_paper_analysis/`.
- `paper-smoke`: tiny staged end-to-end check that writes every expected paper artifact.
- `paper-diagnostics`: deterministic payload, rank-pressure, and codec comparison stage.
- `paper-nonseg-generation`: resumable non-segmented generation stage.
- `paper-segmented-generation`: resumable segmented generation stage.
- `paper-baselines`: greedy baseline generation stage.
- `paper-detector`: analysis-only detector dataset and baseline stage.
- `paper-statistics`: analysis-only bootstrap, effect-size, figures, and paper Markdown stage.
- `paper-main-pilot-resume`: staged pilot sequence that can be resumed and batched.

## Payload Suite

The paper payload suite is implemented in `rankcloak/paper_payloads.py`.

Pilot payloads use two instances each from:

- `sha256_hex`
- `random_128_bit_hex`
- `random_256_bit_hex`
- `nonce_96_bit_hex`
- `uuid_v4_like`
- `ciphertext_like_base64`

Full paper-main payloads use five instances each from:

- `sha256_hex`
- `random_128_bit_hex`
- `random_256_bit_hex`
- `nonce_96_bit_hex`
- `uuid_v4_like`
- `hmac_like_hex`
- `ciphertext_like_base64`

All payloads are deterministic and synthetic.

## Protocol Variants

- `raw_subword_rank_pressure`: diagnostic direct-token rank pressure, no cover generation.
- `nonseg_ascii_b8`: non-segmented ASCII-byte fixed-radix encoding with B=8.
- `nonseg_ascii_b16`: non-segmented ASCII-byte fixed-radix encoding with B=16.
- `nonseg_hex_nibble_b16`: non-segmented raw hex-nibble encoding for hex-like payloads.
- `segmented_hex_single_topic_sentence_tail_filtered`: segmented hex-nibble protocol with one recipe prompt, sentence tails, and `safe_text_filter_v1`.
- `segmented_hex_multi_topic_sentence_tail_filtered`: segmented hex-nibble protocol with a rotating ordinary-prose prompt schedule, sentence tails, and `safe_text_filter_v1`.
- `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`: experimental lead-in variant with an eight-token greedy lead-in before the forced span.

## Main Outputs

The paper profiles write payload tables, rank-pressure diagnostics, codec comparisons,
non-segmented trials, segmented trials, segmented message JSONL, baseline examples,
unified feature tables, detector datasets, detector baselines, bootstrap statistics,
effect sizes, manuscript-oriented Markdown summaries, and paper figures.

## Run Commands

Fast staged smoke:

```bash
python3 scripts/run_experiment.py \
  --profile paper-smoke \
  --output-dir results/rankcloak_paper_smoke \
  --overwrite
```

Resume the pilot in CPU-practical batches:

```bash
python3 scripts/run_experiment.py \
  --profile paper-diagnostics \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume
```

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

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
```

Legacy all-in-one pilot:

```bash
python3 scripts/run_experiment.py \
  --profile paper-main-pilot \
  --output-dir results/rankcloak_paper_main_pilot \
  --overwrite
```

```bash
python3 scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_main \
  --overwrite
```

```bash
python3 scripts/run_experiment.py \
  --profile paper-analysis \
  --output-dir results/rankcloak_paper_analysis \
  --overwrite
```

## Interpretation

Use the staged pilot to validate schemas, runtime, detector plumbing, statistics, and figures.
Use the full paper-main run for manuscript claims only after reviewing runtime and
result quality.

The staged runner writes `RUN_PROGRESS.json` after each stage and after each completed
generation trial where practical. `--resume` and `--skip-existing` skip stable
`trial_id` values already present in output CSV or JSONL files. `--limit-trials`
lets long CPU runs be completed in batches without duplicating rows.

## Current Staged Pilot State

Current directory:

```text
results/rankcloak_paper_main_pilot/
```

Current package:

```text
paper_artifacts/rankcloak_paper_main_pilot/
```

Current counts:

- Non-segmented trials: 20 complete out of 96 planned.
- Segmented trials: 7 complete out of 24 planned.
- Recovery: 26 pass and 1 fail.
- Baseline examples: 22.
- Detector dataset rows: 272.
- Statistical summary rows: 97.
- Effect-size rows: 14.

The failed row is in the experimental
`segmented_hex_multi_topic_leadin8_sentence_tail_filtered` variant. Treat this variant
as experimental until the failed row is rerun successfully or the variant is removed
from paper-main claims.

## Unsupported Claims

The suite does not support claims about secrecy, cryptographic security, robustness to
edits, cross-model portability, or undetectability.
