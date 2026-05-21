# Segmented Protocol Quality Controls Pilot

This pilot extends the segmented multi-cover protocol by separating forced-prefix metrics from full-message metrics, adding sentence-boundary tails, adding natural tails to compact control requests, and testing a deterministic safe-text token filter.

The simulated parties already share `K_common`. This is not encryption, key exchange, authentication, signing, or cryptographic security.

## Metric Separation

Forced-prefix metrics describe the payload-bearing tokens only. Full-message metrics describe the public message including the natural tail. Tail-heavy conditions can look better in full-message metrics even when the forced prefix remains awkward.

## Control Request Recovery

| Condition | Filter | Tail policy | Tail tokens | Recovery | Artifact count |
| --- | --- | --- | ---: | --- | ---: |
| `segmented_single_topic_fixed_tail40_unfiltered` | `none` | `fixed_tail40` | 40 | `True` | 1 |
| `segmented_single_topic_sentence_tail_unfiltered` | `none` | `sentence_tail_min20_max60` | 26 | `True` | 1 |
| `segmented_multi_topic_sentence_tail_unfiltered` | `none` | `sentence_tail_min20_max60` | 26 | `True` | 1 |
| `segmented_single_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 34 | `True` | 0 |
| `segmented_multi_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 34 | `True` | 0 |

## Response Condition Comparison

| Condition | Filter | Tail policy | Recovery | Forced logprob | Full logprob | Forced repetition | Full repetition | Full artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `segmented_multi_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 1.000 | -4.713 | -1.281 | 0.000 | 0.169 | 0.000 |
| `segmented_multi_topic_sentence_tail_unfiltered` | `none` | `sentence_tail_min20_max60` | 1.000 | -4.877 | -1.293 | 0.000 | 0.146 | 0.125 |
| `segmented_single_topic_fixed_tail40_unfiltered` | `none` | `fixed_tail40` | 1.000 | -4.682 | -1.309 | 0.000 | 0.197 | 0.188 |
| `segmented_single_topic_sentence_tail_filtered` | `safe_text_filter_v1` | `sentence_tail_min20_max60` | 1.000 | -4.653 | -1.482 | 0.000 | 0.174 | 0.000 |
| `segmented_single_topic_sentence_tail_unfiltered` | `none` | `sentence_tail_min20_max60` | 1.000 | -4.682 | -1.397 | 0.000 | 0.174 | 0.188 |

## Focused Comparisons

- `fixed_tail40` is the prior comparable baseline.
- `sentence_tail_min20_max60` tries to avoid abrupt endings while staying deterministic.
- `safe_text_filter_v1` ranks and recovers only over a deterministic allowed-token set.
- Multi-topic schedules rotate ordinary prose prompts and avoid the prior biology/car dialogue failure modes.

## Examples

### `segmented_single_topic_fixed_tail40_unfiltered`

- trial_id: `quality_001`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- token_filter_name: `none`
- tail_policy: `fixed_tail40`
- exact_segment_recovery: `True`
- actual_tail_token_count: `40`
- notes: mixed or ordinary by lightweight heuristics; inspect manually

Forced prefix:

```text
**Stress, Simmiering
```

Tail:

```text
, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

Full message:

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

### `segmented_single_topic_sentence_tail_unfiltered`

- trial_id: `quality_002`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- token_filter_name: `none`
- tail_policy: `sentence_tail_min20_max60`
- exact_segment_recovery: `True`
- actual_tail_token_count: `40`
- notes: mixed or ordinary by lightweight heuristics; inspect manually

Forced prefix:

```text
**Stress, Simmiering
```

Tail:

```text
, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

Full message:

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

### `segmented_multi_topic_sentence_tail_unfiltered`

- trial_id: `quality_003`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- token_filter_name: `none`
- tail_policy: `sentence_tail_min20_max60`
- exact_segment_recovery: `True`
- actual_tail_token_count: `40`
- notes: mixed or ordinary by lightweight heuristics; inspect manually

Forced prefix:

```text
**Stress, Simmiering
```

Tail:

```text
, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

Full message:

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

### `segmented_single_topic_sentence_tail_filtered`

- trial_id: `quality_004`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- token_filter_name: `safe_text_filter_v1`
- tail_policy: `sentence_tail_min20_max60`
- exact_segment_recovery: `True`
- actual_tail_token_count: `40`
- notes: mixed or ordinary by lightweight heuristics; inspect manually

Forced prefix:

```text
**Stress, Simmiering
```

Tail:

```text
, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

Full message:

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

### `segmented_multi_topic_sentence_tail_filtered`

- trial_id: `quality_005`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- token_filter_name: `safe_text_filter_v1`
- tail_policy: `sentence_tail_min20_max60`
- exact_segment_recovery: `True`
- actual_tail_token_count: `40`
- notes: mixed or ordinary by lightweight heuristics; inspect manually

Forced prefix:

```text
**Stress, Simmiering
```

Tail:

```text
, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

Full message:

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

## Limitations

- All payloads are deterministic synthetic examples.
- The control code is a synthetic codebook label, not a secret key or operational command.
- This is not encryption, key exchange, authentication, signing, or cryptographic security.
- Exact-copy conditions are required for recovery.
- The safe-text filter is heuristic and may change capacity, rank pressure, and style.
- Full-message quality can improve because of tails even if forced prefixes remain unnatural.