# Final Paper Results Package

## Status

The staged paper-main-pilot package was advanced from diagnostics-only to a partial
manuscript-preparation package. The full planned generation matrix was not completed
because the remaining staged generation would take multiple additional hours on CPU.

Current package path:

```text
paper_artifacts/rankcloak_paper_main_pilot/
```

Primary result directory:

```text
results/rankcloak_paper_main_pilot/
```

Cross-pilot analysis directory:

```text
results/rankcloak_paper_analysis/
```

## Row Counts

| Artifact | Rows |
| --- | ---: |
| `paper_payloads.csv` | 12 |
| `paper_rank_pressure.csv` | 12 |
| `paper_codec_comparison.csv` | 36 |
| `paper_stegotext_trials.csv` | 20 |
| `paper_segmented_trials.csv` | 7 |
| `paper_segmented_messages.jsonl` | 48 |
| `paper_baseline_examples.jsonl` | 15 |
| `paper_cover_text_features.csv` | 227 |
| `detector_dataset.csv` | 240 |
| `detector_baseline.csv` | 57 |
| `statistical_summary.csv` | 97 |
| `effect_size_summary.csv` | 14 |

## Recovery Summary

- Non-segmented recovery: 20 pass, 0 fail.
- Segmented recovery: 6 pass, 1 fail.
- Total completed paper-main-pilot recovery: 26 pass, 1 fail.

These results support exact recovery only for the completed rows under exact-copy
conditions.

## Completed Matrix Coverage

Non-segmented completed rows:

- `nonseg_ascii_b8`: 8 rows.
- `nonseg_ascii_b16`: 6 rows.
- `nonseg_hex_nibble_b16`: 6 rows.

Segmented completed rows:

- `segmented_hex_single_topic_sentence_tail_filtered`: 3 rows.
- `segmented_hex_multi_topic_sentence_tail_filtered`: 3 rows.
- `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`: 1 row, with
  exact recovery failure.

Remaining planned generation:

- Non-segmented: 76 rows remaining from the 96-row pilot plan.
- Segmented: 17 rows remaining from the 24-row pilot plan.

## Detector Summary

The detector stage wrote 240 dataset rows and 57 result rows. The detector uses only
numeric and Boolean features and does not train on raw text content. Several detector
rows report high AUC values on the partial pilot, but these should be treated as
pipeline checks rather than strong steganalysis evidence because the sample is small
and incomplete.

## Statistics Summary

The statistics stage wrote 97 bootstrap summary rows and 14 effect-size rows. These
rows are useful for checking the analysis pipeline and for drafting table formats.
They should not be treated as final inferential results until the planned pilot matrix
or a clearly declared subset is completed.

## Figures Generated

The staged pilot generated 10 paper-oriented figures:

- `paper_payload_representation_rank_counts.png`
- `paper_direct_subword_rank_pressure.png`
- `paper_alphabet_capacity_quality.png`
- `paper_nonseg_protocol_variant_quality.png`
- `paper_segmented_protocol_variant_quality.png`
- `paper_forced_prefix_vs_full_message.png`
- `paper_artifact_counts_by_variant.png`
- `paper_recovery_by_variant.png`
- `paper_detector_auc.png`
- `paper_effect_sizes_summary.png`

Recommended main paper candidates after completing more rows:

- Payload representation rank counts.
- Direct subword rank pressure.
- Recovery by protocol variant.
- Forced-prefix versus full-message quality.
- Artifact counts by variant.

Recommended supplement candidates:

- Detector AUC.
- Effect sizes.
- Nonseg protocol variant quality.
- Segmented protocol variant quality.

## Supported Claims

- The completed rows recover exactly under tested exact-copy conditions.
- The paper payload suite, rank-pressure diagnostics, codec comparison, generation,
  detector, statistics, figure, and package paths are wired together.
- Payload representation and bounded-rank constraints can be measured reproducibly.
- Segmented trial outputs keep forced-prefix and full-message metrics separate.

## Claims Not Supported

- No cryptographic security claim.
- No encryption, key exchange, authentication, signing, credential handling, or
  undetectability claim.
- No edit robustness or paraphrase robustness claim.
- No cross-model generalization claim.
- No broad naturalness claim without further human or stronger plausibility study.

## Limitations

- The current paper-main-pilot package is partial.
- Completed generation rows are concentrated at the beginning of the deterministic
  planned trial order.
- Detector and bootstrap results are based on a small incomplete matrix.
- One experimental segmented lead-in row failed exact recovery.

## Exact Next Writing Tasks

- Decide whether the paper will use the current package as a pipeline-validation
  package or whether Alex will complete the remaining generation batches.
- If completing batches, rerun baselines, detector, statistics, and paper-analysis
  after the generation rows are complete.
- Use `notes/11_paper_methods_draft.md` as the Methods source.
- Use `notes/12_results_for_paper_draft.md` and this note as the Results source,
  but label incomplete pilot evidence clearly.
