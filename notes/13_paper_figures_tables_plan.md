# Paper Figures And Tables Plan

This note lists candidate manuscript figures and tables, their purpose, source artifacts, required columns, current status, and interpretation notes.

## Proposed Figures

### Figure 1: RankCloak Concept Diagram

Purpose: show payload encoding, rank sequence, prompt-key cover generation, exact-copy channel, and recovery.

Source result directory: code-derived conceptual figure, no result directory required.

Source files: `rankcloak/rank_codec.py`, `rankcloak/prompts.py`, `notes/11_paper_methods_draft.md`.

Needed columns: none.

Current status: needs new diagram.

Notes for manuscript interpretation: include `K_common` assumptions and exact-copy requirement. State that the method is not encryption or key exchange.

### Figure 2: Payload Representation Comparison

Purpose: compare rank counts across ASCII-byte fixed-radix, raw hex-nibble, and raw subword direct representations.

Source result directory: `results/rankcloak_payload_granularity_pilot/`.

Source files: `payload_granularity_comparison.csv`, `figures/payload_representation_rank_count.png`.

Needed columns: `payload_name`, `representation_name`, `alphabet_size`, `rank_count`, `max_possible_rank`, `bits_per_rank_estimate`.

Current status: available now as a pilot figure; may need paper styling.

Notes for manuscript interpretation: emphasize that this is payload-side representation only. The cover-side tokenizer remains the model tokenizer.

### Figure 3: Direct Subword Rank Pressure

Purpose: show mean and p95 or max rank for direct subword payload tokens.

Source result directory: `results/rankcloak_small_full/`.

Source files: `rank_statistics.csv`, `figures/rank_summary_direct_subword.png`.

Needed columns: `payload_name`, `rank_count`, `mean_rank`, `p95_rank`, `max_rank`, `fraction_rank_le_16`, `fraction_rank_le_64`.

Current status: available now as a pilot figure.

Notes for manuscript interpretation: use to motivate bounded-rank encodings for high-entropy artifacts.

### Figure 4: Prompt Family And Alphabet-Size Effects

Purpose: show how B and prompt family affect token log probability, repetition, and rank pressure.

Source result directories: `results/rankcloak_small_full/`, `results/rankcloak_strong_prompt_sweep/`, `results/rankcloak_dialogue_key_pilot/`.

Source files: `cover_text_features.csv`, `stegotext_recovery_trials.csv`, prompt figures.

Needed columns: `alphabet_size`, `cover_prompt_name`, `prompt_family`, `mean_token_log_probability`, `repeated_token_fraction`, `punctuation_fraction`, `p95_generated_rank`, `exact_recovery`.

Current status: available now as pilot figures; needs consolidated paper-main analysis.

Notes for manuscript interpretation: avoid claiming prompt choice solves rank pressure. Current evidence suggests alphabet size is the dominant factor.

### Figure 5: Segmented Protocol Diagram

Purpose: illustrate the two-stage control code and segmented response flow.

Source result directory: `results/rankcloak_segmented_protocol_pilot/`.

Source files: `SEGMENTED_PROTOCOL_COMPARISON.md`, `control_request_trial.jsonl`, `segmented_protocol_messages.jsonl`.

Needed columns: `control_code`, `condition_name`, `segment_index`, `forced_rank_count`, `natural_tail_tokens`, `exact_segment_recovery`.

Current status: needs new conceptual diagram.

Notes for manuscript interpretation: show that User A and User B already share `K_common`; no key exchange is implemented.

### Figure 6: Forced-Prefix Versus Full-Message Quality

Purpose: compare payload-bearing forced-prefix metrics against full public-message metrics.

Source result directory: `results/rankcloak_segmented_quality_controls/`.

Source files: `segmented_quality_trials.csv`, `cover_text_features.csv`, `figures/quality_forced_vs_full_logprob.png`, `figures/quality_forced_vs_full_repetition.png`.

Needed columns: `condition_name`, `forced_prefix_mean_log_probability_mean`, `full_message_mean_log_probability_mean`, `forced_prefix_repetition_mean`, `full_message_repetition_mean`.

Current status: available now as a pilot figure.

Notes for manuscript interpretation: central figure for preventing tail-dominated quality overclaims.

### Figure 7: Safe-Text Filter Effect

Purpose: compare unfiltered and `safe_text_filter_v1` conditions on artifact counts, log probability, and recovery.

Source result directory: `results/rankcloak_segmented_quality_controls/`.

Source files: `segmented_quality_trials.csv`, `cover_text_features.csv`, `figures/quality_filter_effect_artifacts.png`, `figures/quality_filter_effect_logprob.png`.

Needed columns: `token_filter_name`, `condition_name`, `forced_prefix_artifact_count_mean`, `full_message_artifact_count_mean`, `full_message_mean_log_probability_mean`, `exact_recovery`.

Current status: available now as a pilot figure.

Notes for manuscript interpretation: filter is deterministic and heuristic; do not present it as a general detector or safety mechanism.

### Figure 8: Detector Baseline, Future Paper-Main Experiment

Purpose: report detectability using a trained baseline classifier if implemented later.

Source result directory: future result directory, not present yet.

Source files: future `detector_baseline.csv` or equivalent.

Needed columns: `source_type`, feature columns, train/test split, classifier type, AUC, confidence intervals if computed.

Current status: needs paper-main run.

Notes for manuscript interpretation: no detector AUC should be claimed from current artifacts.

## Proposed Tables

### Table 1: Methodological Components And Assumptions

Purpose: define payload codec, rank ordering, prompt key, model dependency, exact-copy channel, tails, filters, and segmented protocol assumptions.

Source result directory: documentation and code.

Source files: `notes/09_methodology_inventory.md`, `notes/11_paper_methods_draft.md`, `rankcloak/rank_codec.py`, `rankcloak/segmented_protocol.py`.

Needed columns: component, purpose, implementation, assumption, limitation.

Current status: available now from notes.

Notes for manuscript interpretation: include unsupported claims explicitly.

### Table 2: Synthetic Payload Classes

Purpose: list deterministic synthetic payload types and why they are included.

Source result directory: any directory with `tokenization_audit.csv`, preferably `results/rankcloak_small_full/`.

Source files: `rankcloak/synthetic_payloads.py`, `tokenization_audit.csv`.

Needed columns: `payload_name`, `payload_kind`, `description`, `character_length`, `byte_length`, `llm_token_count`.

Current status: available now.

Notes for manuscript interpretation: state that none are real secrets.

### Table 3: Experiment Profiles

Purpose: summarize profile matrices and intended questions.

Source result directory: all result directories.

Source files: `rankcloak/experiments.py`, `notes/03_experiment_profiles.md`, `notes/10_results_index.md`.

Needed columns: profile, payloads, prompts/conditions, alphabet sizes, output directory, purpose.

Current status: available now.

Notes for manuscript interpretation: distinguish smoke/pilot profiles from paper-main profiles.

### Table 4: Result Directories And Artifacts

Purpose: map each result directory to output files and paper use.

Source result directory: all current result directories.

Source files: `notes/10_results_index.md`, `summary.json`, `MANIFEST.json`.

Needed columns: directory, profile, key CSVs, key JSONL files, figures, comparison markdown, recovery summary.

Current status: available now.

Notes for manuscript interpretation: useful for data availability and supplementary materials.

### Table 5: Recovery Results By Condition

Purpose: summarize exact recovery across non-segmented and segmented pilots.

Source result directories: all result directories with recovery CSVs.

Source files: `summary.json`, `stegotext_recovery_trials.csv`, `segmented_protocol_trials.csv`, `segmented_quality_trials.csv`.

Needed columns: profile, condition or prompt group, payload count, trial count, recovery passes, recovery failures.

Current status: available now for pilots.

Notes for manuscript interpretation: recovery is exact-copy only and should not be treated as robustness.

### Table 6: Limitations And Unsupported Claims

Purpose: clearly separate what the study evaluates from what it does not claim.

Source result directory: documentation.

Source files: `notes/09_methodology_inventory.md`, `notes/11_paper_methods_draft.md`, `README.md`.

Needed columns: limitation, affected method, current mitigation, required future work.

Current status: available now.

Notes for manuscript interpretation: include no encryption, no key exchange, no authentication, no real secrets, no undetectability claim, and exact-copy dependency.
