# RankCloak Notes

This directory documents the current RankCloak repository state, implemented code, completed experiments, and interpretation notes. It is intended as a durable project log separate from generated result files.

## Documents

- `01_project_context.md`: research goal, source context, safety boundaries, and core terminology.
- `02_codebase_overview.md`: package modules, scripts, notebook, tests, schemas, and result conventions.
- `03_experiment_profiles.md`: experiment profiles, payload/prompt/alphabet matrices, and run commands.
- `04_results_so_far.md`: completed runs, key metrics, and current interpretation of cover quality.
- `05_reproducibility_runbook.md`: setup, model path, commands, git behavior, and exact-copy requirements.
- `06_next_steps.md`: paper-readiness gaps and recommended next experiments.
- `07_segmented_protocol_pilot.md`: two-stage segmented multi-cover protocol design and pilot results.
- `08_segmented_quality_controls.md`: quality-control follow-up for forced-prefix metrics, sentence tails, control tails, and token filtering.
- `09_methodology_inventory.md`: paper-oriented inventory of implemented methods, assumptions, code locations, result directories, and unsupported claims.
- `10_results_index.md`: result-directory map with recovery summaries, key artifacts, figures, paper use, and caveats.
- `11_paper_methods_draft.md`: reusable methods-section draft covering payloads, codecs, rank ordering, prompts, segmented protocol, metrics, and limitations.
- `12_results_for_paper_draft.md`: result-oriented draft summarizing current pilot evidence with citations to result files.
- `13_paper_figures_tables_plan.md`: proposed paper figures and tables with source files, needed columns, current status, and interpretation notes.
- `14_submission_readiness_checklist.md`: practical checklist for methodology, results, reproducibility, statistics, detection, data, code, and responsible-use readiness.
- `15_paper_main_experiment_plan.md`: locked paper-main suite design, payload matrix, protocol variants, outputs, and commands.
- `16_paper_main_results_summary.md`: placeholder and index for paper-main pilot, full paper-main, and paper-analysis outputs.
- `17_detector_baseline_plan.md`: lightweight feature-only detector baseline design and caveats.
- `18_statistical_analysis_plan.md`: deterministic bootstrap and effect-size analysis plan.
- `19_final_paper_results_package.md`: final staged package status, row counts,
  recovery summary, detector/statistics caveats, figures, supported claims, and
  remaining commands.
- `20_current_experiment_artifact_inventory.md`: current consolidated inventory of
  implemented code, experiment profiles, result directories, row counts, paper package
  status, and remaining commands.
- `21_gpu_support_and_validation.md`: GPU-loading implementation, CUDA runtime
  setup, deterministic rank-replay controls, manifest changes, paper-matched
  validation, timing, and CPU/GPU consistency limits.
- Staged paper-suite resume behavior is documented in `15_paper_main_experiment_plan.md`
  and current staged run status is tracked in `16_paper_main_results_summary.md`.

## Current Status

RankCloak has moved from a notebook-only smoke prototype to a scriptable empirical framework. The repository now supports deterministic synthetic payload generation, bounded-rank codecs, direct subword rank audits, RankCloak stegotext generation/recovery, greedy baselines, lightweight cover-text feature extraction, reproducibility manifests, figures, and tests.

The latest completed pilots are:

- `results/rankcloak_strong_prompt_sweep/`: long prompt sweep over recipe, biology, car-buying, and forum prompts.
- `results/rankcloak_dialogue_key_pilot/`: dialogue-vs-monologue pilot at B=8 and B=16.
- `results/rankcloak_payload_granularity_pilot/`: payload-side representation comparison for ASCII fixed-radix, hex-nibble ranks, and direct subword ranks.
- `results/rankcloak_segmented_protocol_pilot/`: two-stage control-code and segmented multi-cover response pilot.
- `results/rankcloak_segmented_quality_controls/`: segmented quality-controls pilot output.
- `results/rankcloak_paper_smoke/`: tiny end-to-end paper-suite validation run.
- `results/rankcloak_paper_main_pilot/`: partial staged paper-main-pilot package with
  20/96 non-segmented trials, 7/24 segmented trials, 26 recovery passes, and 1
  experimental lead-in segmented failure.
- `results/rankcloak_paper_analysis/`: aggregation across current pilot and
  paper-suite artifacts.
- `results/rankcloak_paper_gpu_validation/`: paper-ID-matched RTX 5000 Ada
  validation with full GPU offload, downstream detector/statistics artifacts, and a
  dedicated consistency report.
- `results/rankcloak_paper_gpu_pilot_complete/`: complete RTX 5000 Ada pilot with
  96/96 non-segmented and 24/24 segmented rows, canonical baselines, downstream
  analyses, and all paper figures.

All payloads are deterministic synthetic examples. This project studies exact-copy concealment behavior; it is not encryption, key exchange, authentication, or credential handling.

## Paper Production Notes

For paper drafting, start with:

- `09_methodology_inventory.md` for method inventory and code/result mapping.
- `10_results_index.md` for artifact locations and result-directory meanings.
- `11_paper_methods_draft.md` for manuscript Methods text.
- `12_results_for_paper_draft.md` for current Results text and caveats.
- `13_paper_figures_tables_plan.md` for figure and table planning.
- `14_submission_readiness_checklist.md` for what remains before journal submission.
- `15_paper_main_experiment_plan.md` through `18_statistical_analysis_plan.md` for the paper-main results suite.
- `19_final_paper_results_package.md` for the current manuscript package status.
- `20_current_experiment_artifact_inventory.md` for the most compact current inventory.
- `21_gpu_support_and_validation.md` for GPU setup, implementation details, and
  interpretation of the paper-matched validation.
- Use `paper-smoke` to validate the staged pipeline quickly, then continue
  `results/rankcloak_paper_main_pilot/` with the staged `paper-*` profiles and
  `--resume --limit-trials N`.
