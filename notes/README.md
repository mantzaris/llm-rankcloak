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

## Current Status

RankCloak has moved from a notebook-only smoke prototype to a scriptable empirical framework. The repository now supports deterministic synthetic payload generation, bounded-rank codecs, direct subword rank audits, RankCloak stegotext generation/recovery, greedy baselines, lightweight cover-text feature extraction, reproducibility manifests, figures, and tests.

The latest completed pilots are:

- `results/rankcloak_strong_prompt_sweep/`: long prompt sweep over recipe, biology, car-buying, and forum prompts.
- `results/rankcloak_dialogue_key_pilot/`: dialogue-vs-monologue pilot at B=8 and B=16.
- `results/rankcloak_payload_granularity_pilot/`: payload-side representation comparison for ASCII fixed-radix, hex-nibble ranks, and direct subword ranks.
- `results/rankcloak_segmented_protocol_pilot/`: two-stage control-code and segmented multi-cover response pilot.
- `results/rankcloak_segmented_quality_controls/`: segmented quality-controls pilot output.

All payloads are deterministic synthetic examples. This project studies exact-copy concealment behavior; it is not encryption, key exchange, authentication, or credential handling.
