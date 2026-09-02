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

## Model-backed generation

- `plan_id` / `pairing_unit_id` / `experimental_cell_id`: deterministic trial, matched quantization-observation, and six-row entropy-cell identifiers.
- `entropy_bits`: filtered next-token Shannon entropy in bits before the observed token.
- `eligible`: inclusive entropy-threshold decision recomputed by encoder and decoder.
- `token_role`: `payload`, `ordinary_sampled_skip`, or `ordinary_control`.
- `payload_rank`: consumed payload rank at an eligible position; missing at sampled skips and controls.
- `observed_rank`: exact model rank of the observed token under the relevant quantization and token mask.
- `token_surprisal_nats`: negative exact-model token log probability.
- `rank_pressure_log_probability_gap_nats`: greedy log probability minus observed-token log probability.
- `payload_completion`: whether all requested ranks were embedded before the maximum length.
- `fixed_payload_bits_per_generated_token`: serialized payload bits divided by full generated-token count.
- `fixed_token_budget_payload_fraction`: serialized payload fraction embedded within the paired ungated token budget.
- `ordinary_sampled_skip`: an entropy-ineligible top-p sample that does not consume a payload symbol.
- Ungated records retain the historical `forced_log_probabilities` field name; analysis aliases it to the same embedding-span trace represented by `embedding_log_probabilities` in gated records.
- Calibration `validation` maps contain the exclusion assertion `detector_outcomes_used=false`; a valid calibration record requires that value to be false while its token-count and finite-entropy assertions are true.
- `mean_entropy_q8_minus_q4_bits`: paired mean change when Q8 replays the identical historical Q4 token path.
- `observed_token_rank_changed_fraction` / `greedy_token_changed_fraction`: fraction of identical-path positions whose observed-token rank or greedy token differs across quantizations.
- `positionwise_generated_token_match_fraction`: same-position Q4/Q8 token agreement for independently generated paired outputs.

## Model-backed detector evaluations

- `entropy_gates`: historical payload-excluded train/validation data with the new entropy corpus locked as test.
- `q4_to_q8` / `q8_to_q4`: train and validation on one quantization, test on the other using held payloads.
- `pooled_quantizations`: both quantizations in fitting partitions and both in the held-payload test partition.
- `deduplication_before_feature_extraction`: assertion that locked exact/near-duplicate auditing preceded neural tokenization or model-aware feature construction.
- `training_quantizations` / `test_quantizations`: machine-readable holdout identity.
- `fpr_0_001_available`: false for model-backed evaluations whose validation or test control count is below 1,000; no interpolation substitutes for the unavailable estimand.
