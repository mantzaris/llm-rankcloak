# Paper Main Results Summary

This note is updated by hand after running the paper-main profiles. It distinguishes
pilot validation results from the larger frozen paper-main matrix.

## Paper-Main Pilot

Expected output directory:

```text
results/rankcloak_paper_main_pilot/
```

Expected key files:

- `paper_payloads.csv`
- `paper_rank_pressure.csv`
- `paper_codec_comparison.csv`
- `paper_stegotext_trials.csv`
- `paper_segmented_trials.csv`
- `paper_segmented_messages.jsonl`
- `paper_baseline_examples.jsonl`
- `paper_cover_text_features.csv`
- `detector_dataset.csv`
- `detector_baseline.csv`
- `statistical_summary.csv`
- `effect_size_summary.csv`
- `PAPER_RESULTS_SUMMARY.md`
- `PAPER_COMPARISON_TABLES.md`
- `PAPER_FIGURE_INDEX.md`
- `summary.json`
- `SUMMARY.md`
- `MANIFEST.json`

Interpretation target:

- validate exact recovery under the paper-suite schemas;
- compare B=8 and B=16 non-segmented bounded-rank covers;
- compare ASCII fixed-radix and raw hex-nibble payload representations;
- compare segmented single-topic, multi-topic, and lead-in variants;
- verify detector and bootstrap output plumbing.

## Paper-Main Full Run

Expected output directory:

```text
results/rankcloak_paper_main/
```

The full profile uses a larger deterministic payload suite and five ordinary prose
prompts. It should be treated as the main manuscript candidate only after runtime and
quality are reviewed.

## Paper Analysis

Expected output directory:

```text
results/rankcloak_paper_analysis/
```

This profile aggregates prior pilot directories and paper-suite outputs without model
generation. It is useful for deciding which artifacts belong in the main paper versus
supplementary material.

## Current Status

Current implementation status:

- `paper-main-pilot` is implemented.
- Staged paper profiles are implemented so the pilot can be resumed instead of rerun
  as one large process.
- The staged `results/rankcloak_paper_main_pilot/` package now contains diagnostics,
  10 non-segmented trials, 2 segmented trials, 14 baselines, detector outputs,
  bootstrap/effect-size outputs, Markdown summaries, and figures.
- The completed staged rows record 12/12 exact recovery: 10 non-segmented pass,
  0 non-segmented fail, 2 segmented pass, and 0 segmented fail.
- The planned pilot generation matrix is still incomplete: 86 non-segmented rows and
  22 segmented rows remain.
- `paper-analysis` completed in `results/rankcloak_paper_analysis/` and aggregated
  existing pilot directories plus the current partial paper-main-pilot package.
- `paper-main` is implemented but has not been run.

For actual numbers, use `summary.json`, `PAPER_RESULTS_SUMMARY.md`, and the generated
CSV files when they exist. Do not fabricate missing values.

## Staged Resume Workflow

Continue the partial pilot with:

```bash
python3 scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

Then run:

```bash
python3 scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

After enough generation rows exist, run:

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
```

`RUN_PROGRESS.json` records planned trials, skipped existing rows, completed trials,
failures, remaining trials, and the last completed stable `trial_id`.

See `notes/19_final_paper_results_package.md` for final package row counts and
manuscript-use guidance.
