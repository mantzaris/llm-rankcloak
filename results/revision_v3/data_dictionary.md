# Revision V3 data dictionary

## Corpus and partitions

- `row_id`: immutable detector observation identifier.
- `pair_id`: matched RankCloak/ordinary-control pair identifier.
- `payload_group_id`: payload instance used as the original grouping unit.
- `raw_text_sha256`: SHA-256 of the original UTF-8 visible text.
- `normalized_text_sha256`: SHA-256 after NFKC, case folding, whitespace collapse, and stripping.
- `dedup_cluster_id` / `split_group_id`: connected component of payload groups joined by exact or thresholded near-duplicate links; the indivisible split/bootstrap unit.
- `partition`: `train`, `validation`, or `test`; assigned only at component level.
- `label`: 1 for RankCloak and 0 for a control.
- `model_id`, `codec_id`, `payload_class`, `prompt_template_id`: preserved experimental factors.

## Predictions and detector metrics

- `evaluation_role`: validation or test row in a detector prediction ledger; training rows are not scored into this ledger.
- `evaluation_id`: `matched` or `leave_one_model__<model_id>`.
- `score`: detector positive-class score; larger values indicate RankCloak.
- `roc_auc`: tie-aware empirical ROC area.
- `partial_auc_fpr_0_01`: exact empirical false-positive-budget step area from FPR 0 to 0.01, divided by 0.01.
- `threshold_at_fpr_*`: threshold selected exclusively on validation scores and labels.
- `tpr_at_fpr_*` / `fpr_at_threshold_*`: test rates at the frozen validation threshold.
- `false_positives_at_fpr_*`: exact count of test controls above the frozen threshold.
- `validation_negative_count` / `test_negative_count`: denominators governing low-FPR resolution.
- `*_ci_low_95` / `*_ci_high_95`: 2,000-resample dedup-cluster bootstrap interval unless the source table states Wilson.
- `fpr_*_available` / `fpr_*_unavailable_reason`: empirical-resolution status and fail-closed explanation.

## Human-authored secondary controls

- `candidate_id` / `source_record_id`: stable Dolly candidate and pinned source-record identifiers; committed ledgers omit licensed response text.
- `message_text_sha256` / `canonical_text_sha256`: raw-display and canonical-response hashes.
- `relative_word_difference` / `matching_cost`: deterministic length/topic matching diagnostics.
- `human_fpr_at_llm_validation_threshold_0_01`: human-control false-positive rate at a threshold selected without human labels.

## Cover variability and recovery

- `normalized_character_edit_distance`: Levenshtein distance divided by the longer character length.
- `token_jaccard_similarity`: casefolded word-type intersection divided by union.
- `replay_mode`: saved IDs, greedy lead-in regeneration, or detokenized-text retokenization.
- `success_outcome_rows` / `observed_outcome_rows`: recovery numerator and denominator in the bounded historical robustness sample.
- `recovery_rate`: their ratio; its `ci_low` and `ci_high` are source-trial Wilson limits in the recovery table.

## Availability and provenance

- `unavailable_reason`: explicit reason an estimand or experiment could not be computed.
- `sha256`: content checksum over exact artifact bytes.
- `row_count`: data-row count excluding the CSV header; blank for non-tabular artifacts.
