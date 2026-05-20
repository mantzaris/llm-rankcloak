# Two-Stage Segmented Multi-Cover RankCloak Pilot

This pilot tests whether splitting a synthetic payload across several short cover messages reduces cover-text drift compared with one longer forced-rank cover message.

The simulated parties already share `K_common`: model file, tokenizer, quantization, rank ordering, payload codec, prompt templates, control prompt, segment-size rule, topic schedule, and forced-prefix decode policy. This is not key exchange.

## Control Request

- control_code: `C1`
- control_prompt_name: `recipe_forum_exchange_specific`
- control_codec_name: `ascii_bytes_fixed_radix_b16`
- exact_recovery: `True`
- generated_token_count: `4`

## Conditions

| Condition | Messages | Tail tokens/message | Recovery rate | Mean logprob | Mean repetition | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `segmented_multi_topic_tail40` | 6.0 | 40 | 1.000 | -1.233 | 0.167 | 8-rank segments; rotating prompt schedule; decoder ignores greedy tails |
| `segmented_single_topic_no_tail` | 6.0 | 0 | 1.000 | -4.682 | 0.000 | 8-rank segments; same recipe prompt for every message |
| `segmented_single_topic_tail40` | 6.0 | 40 | 1.000 | -1.309 | 0.197 | 8-rank segments; same recipe prompt; decoder ignores greedy tails |
| `single_long_recipe_no_tail` | 1.0 | 0 | 1.000 | -4.508 | 0.062 | one forced span; no natural tail |
| `single_long_recipe_tail40` | 1.0 | 40 | 1.000 | -2.908 | 0.204 | one forced span; decoder ignores greedy tail |

## Recovery Summary

- response_trial_count: `10`
- response_recovery_pass_count: `10`
- response_recovery_fail_count: `0`

## Comparison Notes

- Single-long conditions keep each payload in one forced span.
- Segmented conditions restart from a clean prompt context for each short rank chunk.
- Tail40 conditions append greedy natural text that is not decoded.
- Multi-topic segmentation rotates recipe, forum, car-buying, and recipe-blog prompts.

## Generated Examples

### `single_long_recipe_no_tail`

- trial_id: `segmented_001`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `64`
- mean_token_log_probability: `-4.671363834642375`
- notes: mostly prose-like by lightweight features

```text
**Stress, Simmiering: Lentel Vagetbtle Stuwwi in a Hvy Potter**.
Once onions, carrots' are warming the bottom layer. They'll become fragrances - so good the kitchen should, for instance; I like, to let you feel at that point the lent
```

- trial_id: `segmented_006`
- payload_name: `random_128_bit_hex`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `32`
- mean_token_log_probability: `-4.345224120250197`
- notes: mostly prose-like by lightweight features

```text
...Now let everything rest a minute together until we notice how the garlic's fragrance melds with a pinch and becomes almost unapparent—except for this quiet
```

### `single_long_recipe_tail40`

- trial_id: `segmented_002`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `104`
- mean_token_log_probability: `-3.280445944487394`
- notes: mostly prose-like by lightweight features

```text
**Stress, Simmiering: Lentel Vagetbtle Stuwwi in a Hvy Potter**.
Once onions, carrots' are warming the bottom layer. They'll become fragrances - so good the kitchen should, for instance; I like, to let you feel at that point the lentils are softening, and the tomato is deepening in color. The aroma of garlic is now softening, and the steam on the lid is a sign that the stew is coming together. It
```

- trial_id: `segmented_007`
- payload_name: `random_128_bit_hex`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `72`
- mean_token_log_probability: `-2.535529303234733`
- notes: mostly prose-like by lightweight features

```text
...Now let everything rest a minute together until we notice how the garlic's fragrance melds with a pinch and becomes almost unapparent—except for this quiet, savory depth it adds. As we stir, the tomato's color deepens, and the steam on the lid tells us the stew is coming along nicely. We can already imagine the tender lentils
```

### `segmented_single_topic_no_tail`

- trial_id: `segmented_003`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `8`
- mean_token_log_probability: `-4.3343696657985324`
- notes: punctuation-heavy

```text
**Stress, Simmiering
```

- trial_id: `segmented_003`
- payload_name: `sha256_public_test_string`
- segment_index: `2`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `8`
- mean_token_log_probability: `-4.762247698523064`
- notes: mostly prose-like by lightweight features

```text
Now we reach this satisfying peak—where
```

### `segmented_single_topic_tail40`

- trial_id: `segmented_004`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `48`
- mean_token_log_probability: `-1.2362570589104258`
- notes: mostly prose-like by lightweight features

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

- trial_id: `segmented_004`
- payload_name: `sha256_public_test_string`
- segment_index: `2`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `48`
- mean_token_log_probability: `-1.4342907491324317`
- notes: mostly prose-like by lightweight features

```text
Now we reach this satisfying peak—where the stew's flavors meld together, and the lentils have absorbed the rich, savory broth. The aroma of garlic and onions has mellowed, replaced by the deep, earthy scent of tomato
```

### `segmented_multi_topic_tail40`

- trial_id: `segmented_005`
- payload_name: `sha256_public_test_string`
- segment_index: `1`
- prompt_name: `recipe_long_specific`
- exact_segment_recovery: `True`
- generated_token_count: `48`
- mean_token_log_probability: `-1.2362570589104258`
- notes: mostly prose-like by lightweight features

```text
**Stress, Simmiering, and the Perfect Lentil Stew**
As the aroma of garlic softens and the onions begin to caramelize, the pot starts to simmer, releasing a savory scent that fills the kitchen.
```

- trial_id: `segmented_005`
- payload_name: `sha256_public_test_string`
- segment_index: `2`
- prompt_name: `recipe_forum_exchange_specific`
- exact_segment_recovery: `True`
- generated_token_count: `48`
- mean_token_log_probability: `-1.0585225313554292`
- notes: placeholder-like or markup artifact

```text
---

[original author]**lizzy**: Hey everyone! I'm trying to make lentil stew with ingredients I already have at home. I have lentils, onions, garlic, carrots, celery, canned tomatoes, vegetable broth, and some
```

## Limitations

- All payloads are deterministic synthetic examples.
- This is not encryption, key exchange, authentication, signing, or cryptographic security.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, prompts, and unmodified text.
- Natural tails are ignored by the decoder and do not carry payload ranks in this pilot.
- The quality notes are lightweight heuristics and manual-inspection aids, not detector AUC.