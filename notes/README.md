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
- Use `paper-smoke` to validate the staged pipeline quickly, then continue
  `results/rankcloak_paper_main_pilot/` with the staged `paper-*` profiles and
  `--resume --limit-trials N`.
