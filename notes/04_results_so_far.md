# Results So Far

This file summarizes the result directories currently present in `results/`. The numbers below are taken from `summary.json` and the generated CSV tables.

## Completed Runs

| Result directory | Profile | Model loaded | Payloads | Prompts | Alphabets | Stegotext passes | Stegotext failures |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| `results/rankcloak_crypto_artifact_exploration/` | `smoke` | yes | 1 | 2 | 16, 32 | 4 | 0 |
| `results/rankcloak_small_full/` | `small` | yes | 4 | 4 | 8, 16, 32, 64 | 64 | 0 |
| `results/rankcloak_strong_prompt_pilot/` | `strong-prompts-pilot` | yes | 2 | 4 | 16, 32 | 16 | 0 |
| `results/rankcloak_strong_prompt_sweep/` | `strong-prompts` | yes | 3 | 5 | 8, 16, 32, 64 | 60 | 0 |
| `results/rankcloak_dialogue_key_pilot/` | `dialogue-key-pilot` | yes | 2 | 6 | 8, 16 | 24 | 0 |
| `results/rankcloak_payload_granularity_pilot/` | `payload-granularity-pilot` | yes | 2 | 0 | 8, 16 | 0 | 0 |
| `results/rankcloak_segmented_protocol_pilot/` | `segmented-protocol-pilot` | yes | 2 | protocol-specific | 16 | 10 | 0 |

Codec roundtrip has passed in all current summaries. The model-backed full channel has also recovered exactly in every recorded stegotext trial so far.

## Small Full Sweep

Directory: `results/rankcloak_small_full/`

Matrix:

- 4 payloads.
- 4 cover prompts.
- B=8, B=16, B=32, B=64.
- 64 stegotext trials.

Outcome:

- Exact recovery: 64/64.
- B=8 had the best mean token log probability among bounded alphabets in this sweep.
- B=32 and B=64 showed substantially worse mean token log probability and higher rank pressure.
- Lower alphabet sizes require more generated tokens, but the cover distribution is less aggressively forced.

Mean RankCloak feature summary by alphabet:

| B | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean token count | Mean p95 generated rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -3.5833 | 0.2377 | 0.0555 | 131.00 | 7.0000 |
| 16 | -3.9146 | 0.1957 | 0.0389 | 98.00 | 8.9500 |
| 32 | -5.0398 | 0.1649 | 0.0494 | 79.00 | 27.0125 |
| 64 | -5.8190 | 0.0998 | 0.0618 | 65.75 | 54.1250 |

Interpretation:

- Exact recovery is not the bottleneck under exact-copy conditions.
- Plausibility degrades as B grows because the generator is forced farther into the tail of the next-token distribution.
- B=8 and B=16 are the most useful range for current cover-quality work.

## Strong Prompt Sweep

Directory: `results/rankcloak_strong_prompt_sweep/`

Matrix:

- 3 payloads.
- 5 prompts.
- B=8, B=16, B=32, B=64.
- 60 stegotext trials.

Outcome:

- Exact recovery: 60/60.
- Longer prompts helped topic anchoring but did not remove forced-rank artifacts.
- Alphabet size mattered more than prompt length.
- `forum_reply` and `recipe_long_specific` had the best mean token log probability among the tested strong prompts.

Mean RankCloak feature summary by alphabet:

| B | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean token count | Mean p95 generated rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -3.5805 | 0.2098 | 0.0428 | 117.6667 | 7.0000 |
| 16 | -3.9903 | 0.1696 | 0.0348 | 88.0000 | 8.9333 |
| 32 | -5.2156 | 0.1213 | 0.0582 | 71.0000 | 26.6833 |
| 64 | -6.0072 | 0.0924 | 0.0486 | 59.0000 | 54.2500 |

Top prompt means by mean token log probability:

| Prompt | Mean token log probability | Repeated token fraction | Punctuation fraction |
| --- | ---: | ---: | ---: |
| `forum_reply` | -4.5919 | 0.1500 | 0.0496 |
| `recipe_long_specific` | -4.6302 | 0.1336 | 0.0297 |
| `car_buying_long_specific` | -4.7141 | 0.1514 | 0.0577 |
| `biology_long_specific` | -4.7634 | 0.1352 | 0.0514 |
| `recipe_blog` | -4.7925 | 0.1711 | 0.0420 |

Interpretation:

- Stronger prompts provide better topical scaffolding.
- Long prompts do not solve low-level token damage from high ranks.
- Prompt comparisons should be restricted to B=8 and B=16 until distribution-matched coding is added.

## Dialogue Key Pilot

Directory: `results/rankcloak_dialogue_key_pilot/`

Matrix:

- 2 payloads.
- 6 prompts.
- B=8 and B=16 only.
- 24 stegotext trials.

Outcome:

- Exact recovery: 24/24.
- `recipe_dialogue_specific` had the best mean token log probability, but also the worst repeated-token fraction.
- `recipe_long_specific` looked cleaner by repetition and punctuation metrics.
- `recipe_forum_exchange_specific` was a reasonable middle ground but produced markup/link-like artifacts in some B=16 examples.
- `biology_tutor_dialogue_specific` showed the weakest quality by manual inspection and punctuation artifacts.

Mean RankCloak feature summary by alphabet:

| B | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean token count | Mean p95 generated rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -3.4528 | 0.2475 | 0.0607 | 128.5 | 7.0 |
| 16 | -3.9338 | 0.1966 | 0.0581 | 96.0 | 7.5 |

Prompt means:

| Prompt | Mean token log probability | Repeated token fraction | Punctuation fraction |
| --- | ---: | ---: | ---: |
| `recipe_dialogue_specific` | -3.4839 | 0.3180 | 0.0865 |
| `car_buying_dialogue_specific` | -3.6595 | 0.2148 | 0.0492 |
| `recipe_forum_exchange_specific` | -3.6627 | 0.2003 | 0.0575 |
| `recipe_long_specific` | -3.6887 | 0.1738 | 0.0302 |
| `recipe_blog` | -3.7439 | 0.2509 | 0.0296 |
| `biology_tutor_dialogue_specific` | -3.9211 | 0.1745 | 0.1033 |

Interpretation:

- Dialogue style alone is not a clear solution.
- A dialogue prompt can improve local token probability while increasing obvious repetition.
- For human-readable cover, `recipe_long_specific` and `recipe_forum_exchange_specific` currently look more promising than direct two-speaker recipe dialogue.

## Payload Granularity Pilot

Directory: `results/rankcloak_payload_granularity_pilot/`

Purpose: compare payload-side representations without changing the model tokenizer.

Key rows:

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

Interpretation:

- Hex-nibble coding is much more rank-efficient for hex payloads than encoding the ASCII bytes of the hex string.
- Direct subword encoding uses fewer ranks, but the observed rank pressure is far higher and therefore less compatible with plausible cover generation.
- The right payload representation depends on whether the priority is fewer cover tokens or lower rank pressure.

## Segmented Protocol Pilot

Directory: `results/rankcloak_segmented_protocol_pilot/`

Purpose: test whether breaking a raw-hex-nibble payload into several short cover messages reduces cover drift compared with one long forced-rank message.

Outcome:

- Control request exact recovery: true.
- Response exact recovery: 10/10.
- Payloads: `sha256_public_test_string`, `random_128_bit_hex`.
- Segment size: 8 ranks.
- Decode policy: forced prefix only.

Condition means:

| Condition | Mean token log probability | Repeated token fraction | Punctuation fraction | Mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| `single_long_recipe_no_tail` | -4.5083 | 0.0625 | 0.0448 | 48.0 |
| `single_long_recipe_tail40` | -2.9080 | 0.2035 | 0.0390 | 88.0 |
| `segmented_single_topic_no_tail` | -4.6816 | 0.0000 | 0.0478 | 48.0 |
| `segmented_single_topic_tail40` | -1.3087 | 0.1966 | 0.0346 | 288.0 |
| `segmented_multi_topic_tail40` | -1.2328 | 0.1667 | 0.0501 | 288.0 |

Interpretation:

- No-tail segmentation is recoverable but often fragmentary because each message has only 8 generated tokens.
- Tail40 segmentation makes messages look more natural by adding substantial greedy continuation after each forced prefix.
- The improved full-message log probability is partly a consequence of the high natural-tail-to-forced-token ratio.
- Segmentation may reduce visible drift, but it increases total cover length and message count.

## Current Overall Finding

The current evidence supports this working conclusion:

- Exact recovery is reliable under exact-copy conditions with the current local model and deterministic rank ordering.
- High-entropy artifacts are hard because direct token ranks can be very large.
- Bounded-rank coding keeps cover generation within a controlled rank band.
- B=8 and B=16 are the current practical range for plausibility.
- Better prompts help topical coherence, but rank pressure remains the main source of visible artifacts.
