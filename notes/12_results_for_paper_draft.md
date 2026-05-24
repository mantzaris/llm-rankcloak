# Results For Paper Draft

This note summarizes current evidence for later use in a Results section. The current
artifacts are pilots and should not be presented as final paper-main results.

## Current Recovery Evidence

All pre-paper model-backed stegotext pilots currently record exact recovery under
exact-copy conditions. The partial staged paper-main-pilot records one failure in the
experimental lead-in segmented variant and should be discussed separately.

Source artifacts:

- `results/rankcloak_crypto_artifact_exploration/summary.json`: 4/4 stegotext recovery, 63/63 codec roundtrip.
- `results/rankcloak_small_full/summary.json`: 64/64 stegotext recovery, 63/63 codec roundtrip.
- `results/rankcloak_strong_prompt_pilot/summary.json`: 16/16 stegotext recovery, 63/63 codec roundtrip.
- `results/rankcloak_strong_prompt_sweep/summary.json`: 60/60 stegotext recovery, 63/63 codec roundtrip.
- `results/rankcloak_dialogue_key_pilot/summary.json`: 24/24 stegotext recovery, 63/63 codec roundtrip.
- `results/rankcloak_segmented_protocol_pilot/summary.json`: control request exact recovery true, 10/10 response recovery.
- `results/rankcloak_segmented_quality_controls/summary.json`: 5/5 control recovery, 10/10 response recovery.
- `results/rankcloak_paper_main_pilot/summary.json`: partial staged paper-main-pilot
  package with 20/20 non-segmented recovery and 6/7 segmented recovery.

Interpretation: exact recovery is reliable in the current controlled local setting for
the completed non-lead-in pilots and non-segmented paper rows. The lead-in segmented
failure shows that experimental variants must be validated separately. All recovery
statements require the exact-copy, shared-configuration condition.

## Partial Paper-Main-Pilot Package

Source artifacts:

- `results/rankcloak_paper_main_pilot/PAPER_RESULTS_SUMMARY.md`
- `results/rankcloak_paper_main_pilot/PAPER_COMPARISON_TABLES.md`
- `results/rankcloak_paper_main_pilot/PAPER_FIGURE_INDEX.md`
- `paper_artifacts/rankcloak_paper_main_pilot/README.md`
- `notes/19_final_paper_results_package.md`

The staged paper-main-pilot package currently contains 12 payload rows, 12 direct
rank-pressure rows, 36 codec-comparison rows, 20 non-segmented stegotext rows,
7 segmented stegotext rows, 22 baseline rows, 272 detector dataset rows,
57 detector result rows, 97 bootstrap summary rows, and 14 effect-size rows.

Interpretation: this is a manuscript-preparation and pipeline-validation package, not
the complete planned pilot matrix. The remaining planned generation is 76
non-segmented rows and 17 segmented rows. One completed lead-in segmented row failed
exact recovery and should be reported as a limitation of that experimental variant.

## Direct Subword Rank-Pressure Evidence

Direct subword encoding uses fewer ranks but can create high rank pressure for
high-entropy payloads.

Source artifact: `results/rankcloak_small_full/rank_statistics.csv`.

Selected rows:

| Payload | Token count | Mean rank | Median rank | Max rank | P95 rank | Fraction rank <= 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sha256_public_test_string` | 37 | 212.1081 | 17.0 | 1986 | 1126.40 | 0.4865 |
| `random_128_bit_hex` | 18 | 504.5000 | 72.0 | 5216 | 2113.50 | 0.3889 |
| `random_256_bit_hex` | 38 | 234.6053 | 13.5 | 2966 | 1225.05 | 0.5526 |
| `synthetic_uuid_v4_like` | 21 | 6098.7143 | 13.0 | 123618 | 2747.00 | 0.5238 |

Interpretation: direct subword encoding can be compact in rank count but includes
extreme ranks that are poorly suited to plausible cover generation.

## Bounded-Rank Codec Evidence

Codec roundtrip succeeds for all current deterministic synthetic payloads and supported
alphabet sizes.

Source artifacts:

- `results/rankcloak_small_full/codec_roundtrip_trials.csv`
- `results/rankcloak_dialogue_key_pilot/codec_roundtrip_trials.csv`
- `results/rankcloak_payload_granularity_pilot/codec_roundtrip_trials.csv`

Each current summary records 63 codec passes and 0 codec failures. This covers 9
payloads across the fixed-radix sizes plus the hex-character codec row.

Interpretation: byte/rank transformation correctness is not the current bottleneck.

## Payload Granularity Evidence

The payload granularity pilot compares ASCII bytes, raw hex nibbles, and direct subword
ranks.

Source artifact:
`results/rankcloak_payload_granularity_pilot/payload_granularity_comparison.csv`.

Selected rows:

| Payload | Representation | B | Rank count | Max possible or observed rank | Bits per rank estimate |
| --- | --- | ---: | ---: | ---: | ---: |
| `sha256_public_test_string` | `ascii_bytes_fixed_radix` | 8 | 171 | 8 | 3 |
| `sha256_public_test_string` | `ascii_bytes_fixed_radix` | 16 | 128 | 16 | 4 |
| `sha256_public_test_string` | `raw_hex_nibbles` | 16 | 64 | 16 | 4 |
| `sha256_public_test_string` | `raw_subword_direct` | n/a | 37 | 1986 | n/a |
| `random_128_bit_hex` | `ascii_bytes_fixed_radix` | 8 | 86 | 8 | 3 |
| `random_128_bit_hex` | `ascii_bytes_fixed_radix` | 16 | 64 | 16 | 4 |
| `random_128_bit_hex` | `raw_hex_nibbles` | 16 | 32 | 16 | 4 |
| `random_128_bit_hex` | `raw_subword_direct` | n/a | 18 | 5216 | n/a |

Interpretation: raw hex-nibble coding halves the rank count versus ASCII-byte B=16 for
hex payloads while keeping ranks bounded by 16. Direct subword encoding is shorter but
has much larger observed ranks.

## Prompt Topic And Prompt Format Evidence

The small full sweep suggests alphabet size has a stronger effect than prompt choice in
current pilots.

Source artifacts:

- `results/rankcloak_small_full/cover_text_features.csv`
- `results/rankcloak_small_full/stegotext_recovery_trials.csv`

Mean RankCloak feature summary by alphabet:

| B | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean token count | Mean p95 generated rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -3.5833 | 0.2377 | 0.0555 | 131.00 | 7.0000 |
| 16 | -3.9146 | 0.1957 | 0.0389 | 98.00 | 8.9500 |
| 32 | -5.0398 | 0.1649 | 0.0494 | 79.00 | 27.0125 |
| 64 | -5.8190 | 0.0998 | 0.0618 | 65.75 | 54.1250 |

Interpretation: lower alphabets create longer covers but lower rank pressure and better
token log-probability metrics.

## Strong Prompt Sweep Findings

Source artifacts:

- `results/rankcloak_strong_prompt_sweep/summary.json`
- `results/rankcloak_strong_prompt_sweep/cover_text_features.csv`
- `results/rankcloak_strong_prompt_sweep/PROMPT_COMPARISON.md`

The sweep records 60/60 exact recovery. Mean RankCloak feature summary by alphabet:

| B | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean token count | Mean p95 generated rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -3.5805 | 0.2098 | 0.0428 | 117.6667 | 7.0000 |
| 16 | -3.9903 | 0.1696 | 0.0348 | 88.0000 | 8.9333 |
| 32 | -5.2156 | 0.1213 | 0.0582 | 71.0000 | 26.6833 |
| 64 | -6.0072 | 0.0924 | 0.0486 | 59.0000 | 54.2500 |

Prompt means by mean token log probability:

| Prompt | Mean token log probability | Repeated token fraction | Punctuation fraction |
| --- | ---: | ---: | ---: |
| `forum_reply` | -4.5919 | 0.1500 | 0.0496 |
| `recipe_long_specific` | -4.6302 | 0.1336 | 0.0297 |
| `car_buying_long_specific` | -4.7141 | 0.1514 | 0.0577 |
| `biology_long_specific` | -4.7634 | 0.1352 | 0.0514 |
| `recipe_blog` | -4.7925 | 0.1711 | 0.0420 |

Interpretation: longer prompts helped topical scaffolding, but larger alphabets still
degraded token log-probability metrics.

## Dialogue Prompt Findings

Source artifacts:

- `results/rankcloak_dialogue_key_pilot/summary.json`
- `results/rankcloak_dialogue_key_pilot/cover_text_features.csv`
- `results/rankcloak_dialogue_key_pilot/DIALOGUE_PROMPT_COMPARISON.md`

The pilot records 24/24 exact recovery. Prompt means:

| Prompt | Mean token log probability | Repeated token fraction | Punctuation fraction |
| --- | ---: | ---: | ---: |
| `recipe_dialogue_specific` | -3.4839 | 0.3180 | 0.0865 |
| `car_buying_dialogue_specific` | -3.6595 | 0.2148 | 0.0492 |
| `recipe_forum_exchange_specific` | -3.6627 | 0.2003 | 0.0575 |
| `recipe_long_specific` | -3.6887 | 0.1738 | 0.0302 |
| `recipe_blog` | -3.7439 | 0.2509 | 0.0296 |
| `biology_tutor_dialogue_specific` | -3.9211 | 0.1745 | 0.1033 |

Interpretation: dialogue style improved local log probability for the recipe dialogue
prompt but increased repetition and punctuation artifacts. Dialogue format alone is not
a complete solution.

## Segmented Protocol Findings

Source artifacts:

- `results/rankcloak_segmented_protocol_pilot/summary.json`
- `results/rankcloak_segmented_protocol_pilot/segmented_protocol_trials.csv`
- `results/rankcloak_segmented_protocol_pilot/SEGMENTED_PROTOCOL_COMPARISON.md`

The pilot records control request exact recovery true and 10/10 response recovery.

Condition means:

| Condition | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| `single_long_recipe_no_tail` | -4.5083 | 0.0625 | 0.0448 | 48.0 |
| `single_long_recipe_tail40` | -2.9080 | 0.2035 | 0.0390 | 88.0 |
| `segmented_single_topic_no_tail` | -4.6816 | 0.0000 | 0.0478 | 48.0 |
| `segmented_single_topic_tail40` | -1.3087 | 0.1966 | 0.0346 | 288.0 |
| `segmented_multi_topic_tail40` | -1.2328 | 0.1667 | 0.0501 | 288.0 |

Interpretation: segmentation and tails improve full-message metrics but increase message
count and total generated tokens. No-tail segmented messages are often too short to be
natural public messages.

## Segmented Quality-Control Findings

Source artifacts:

- `results/rankcloak_segmented_quality_controls/summary.json`
- `results/rankcloak_segmented_quality_controls/segmented_quality_trials.csv`
- `results/rankcloak_segmented_quality_controls/cover_text_features.csv`
- `results/rankcloak_segmented_quality_controls/SEGMENTED_QUALITY_COMPARISON.md`

The pilot records 5/5 control recovery and 10/10 response recovery.

Condition means:

| Condition | Filter | Tail policy | Avg tail tokens | Forced logprob | Full logprob | Full repetition | Full artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `segmented_single_topic_fixed_tail40_unfiltered` | none | `fixed_tail40` | 40.0000 | -4.6816 | -1.3087 | 0.1966 | 0.1875 |
| `segmented_single_topic_sentence_tail_unfiltered` | none | `sentence_tail_min20_max60` | 35.1250 | -4.6816 | -1.3972 | 0.1742 | 0.1875 |
| `segmented_multi_topic_sentence_tail_unfiltered` | none | `sentence_tail_min20_max60` | 35.8125 | -4.8773 | -1.2928 | 0.1457 | 0.1250 |
| `segmented_single_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 32.6250 | -4.6531 | -1.4824 | 0.1736 | 0.0000 |
| `segmented_multi_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 36.1250 | -4.7132 | -1.2813 | 0.1693 | 0.0000 |

Interpretation: metric separation confirms that full-message quality is much better than
forced-prefix quality because tails dominate the public message. The safe-text filter
eliminated tracked artifact flags in this pilot without breaking recovery.

## Forced-Prefix Versus Full-Message Quality

Source artifact:
`results/rankcloak_segmented_quality_controls/segmented_quality_trials.csv`.

The forced-prefix mean log probabilities remain near -4.65 to -4.88 across conditions,
while full-message mean log probabilities are near -1.28 to -1.48. This difference shows
why tail-heavy metrics must not be interpreted as payload-bearing-token quality.

## Safe-Text Filter Evidence

Source artifacts:

- `results/rankcloak_segmented_quality_controls/segmented_quality_trials.csv`
- `results/rankcloak_segmented_quality_controls/cover_text_features.csv`

The filtered conditions recorded 0.0000 mean tracked artifact counts for forced prefixes
and full messages in `segmented_quality_trials.csv`, compared with nonzero artifact
counts in the unfiltered conditions.

Interpretation: `safe_text_filter_v1` reduced tracked artifacts in this run. It also
changes the effective rank space and should be treated as a protocol parameter, not as a
general quality guarantee.

## Current Best-Performing Condition

By current segmented quality-control metrics,
`segmented_multi_topic_sentence_tail_filtered` is the strongest candidate for further
study because it combines:

- 10/10 exact response recovery across the quality-control profile when aggregated with the other condition rows;
- full-message mean log probability of -1.2813;
- full-message artifact count mean of 0.0000;
- sentence-boundary tails;
- deterministic safe-text filtering.

This is not a final best method claim. It is the current best-looking pilot condition
among the measured segmented quality-control variants.

## Current Unresolved Issues

- Forced-prefix quality remains much lower than full-message quality.
- Natural tails improve readability but increase public text length.
- Safe-text filtering may reduce artifacts while changing rank distributions and capacity.
- No detector AUC is implemented.
- No controlled human or LLM plausibility study has been run.
- No cross-model comparison has been run.
- Edit robustness is untested.

## Results That Still Need Larger Paper-Main Runs

- A larger B=8 and B=16 sweep with pre-registered prompt families.
- A segmented quality-control sweep with more payload classes and replicated prompt schedules.
- Detector baseline with train/test splits and AUC reporting.
- Model comparison across Llama, Phi, Mistral, Qwen, and Gemma families.
- Edit robustness tests for whitespace, punctuation, smart quotes, line wrapping, and paraphrase.
- Manual or LLM-assisted plausibility scoring with blinded samples.
