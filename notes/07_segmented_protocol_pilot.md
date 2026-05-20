# Segmented Protocol Pilot

## Protocol Variant

The implemented variant is called Two-Stage Segmented Multi-Cover RankCloak.

The simulated parties already share `K_common`, which includes:

- exact local model file;
- tokenizer;
- quantization;
- deterministic rank ordering;
- payload codec;
- prompt templates;
- control prompt;
- segment-size rule;
- topic schedule rule;
- forced-prefix decode rule.

This is not key exchange. The control message is a compact synthetic codebook label, not a secret, credential, command, signature, or authentication mechanism.

## Control Code

`CONTROL_CODEBOOK` currently contains:

```text
C1 -> sha256_public_test_string, raw_hex_nibbles, segment_size=8,
      topic_schedule=mixed_recipe_forum_car_blog,
      natural_tail_tokens=40,
      decode_policy=forced_prefix_only
```

The pilot hides `C1` with fixed-radix B=16 byte ranks under `recipe_forum_exchange_specific`.

Observed result:

- control rank count: 4
- exact recovery: true
- generated control text: visibly artificial in the completed run

The control request works technically, but future experiments should consider a less visibly artificial control cover, likely by allowing a short natural tail or using a less compact code representation.

## Response Payloads

The response pilot uses deterministic synthetic hex payloads only:

- `sha256_public_test_string`
- `random_128_bit_hex`

Payload codec:

- `raw_hex_nibbles`
- each hex character maps to one rank in `1..16`
- cover-side generation still uses the native LLM tokenizer

Segment size:

- 8 ranks

## Conditions

The completed profile tested 2 payloads x 5 conditions = 10 response trials.

Conditions:

- `single_long_recipe_no_tail`: one forced span, `recipe_long_specific`, no tail.
- `single_long_recipe_tail40`: one forced span, `recipe_long_specific`, 40 greedy tail tokens.
- `segmented_single_topic_no_tail`: 8-rank chunks, `recipe_long_specific`, no tail.
- `segmented_single_topic_tail40`: 8-rank chunks, `recipe_long_specific`, 40 greedy tail tokens per message.
- `segmented_multi_topic_tail40`: 8-rank chunks, rotating recipe/forum/car/recipe-blog prompts, 40 greedy tail tokens per message.

## Completed Run

Directory:

```text
results/rankcloak_segmented_protocol_pilot/
```

Command:

```bash
python3 scripts/run_experiment.py \
  --profile segmented-protocol-pilot \
  --output-dir results/rankcloak_segmented_protocol_pilot \
  --overwrite
```

Recovery:

- control request: 1/1
- response trials: 10/10
- response failures: 0

## Key Metrics

Condition-level means from `segmented_protocol_trials.csv`:

| Condition | Mean logprob | Mean repetition | Mean punctuation | Mean total generated tokens |
| --- | ---: | ---: | ---: | ---: |
| `single_long_recipe_no_tail` | -4.5083 | 0.0625 | 0.0448 | 48.0 |
| `single_long_recipe_tail40` | -2.9080 | 0.2035 | 0.0390 | 88.0 |
| `segmented_single_topic_no_tail` | -4.6816 | 0.0000 | 0.0478 | 48.0 |
| `segmented_single_topic_tail40` | -1.3087 | 0.1966 | 0.0346 | 288.0 |
| `segmented_multi_topic_tail40` | -1.2328 | 0.1667 | 0.0501 | 288.0 |

## Interpretation

The pilot supports a narrow interpretation:

- Exact recovery remains reliable under exact-copy conditions.
- No-tail segmentation produces very short fragments, which are technically recoverable but not independently natural cover messages.
- Tail40 segmentation produces more natural-looking public messages because each short forced prefix is followed by a longer greedy continuation.
- The apparent quality gain is not free: segmented tail40 conditions greatly increase total cover length and message count.
- Multi-topic tail40 slightly improved mean log probability and reduced repetition compared with single-topic tail40 in this run, but the sample is too small for a broad claim.

## Important Caveat

The mean token log probability for tail40 segmented conditions is dominated by greedy tail tokens. It is a public-message quality signal, not a payload-bearing forced-prefix-only metric. Future reports should separate:

- forced-prefix quality;
- natural-tail quality;
- full-message quality;
- cover expansion cost.

## Result Files

- `control_request_trial.jsonl`
- `segmented_protocol_trials.csv`
- `segmented_protocol_messages.jsonl`
- `SEGMENTED_PROTOCOL_COMPARISON.md`
- `cover_text_features.csv`
- `MANIFEST.json`
- `summary.json`
- `SUMMARY.md`
- `figures/segmented_condition_mean_logprob.png`
- `figures/segmented_condition_repetition.png`
- `figures/segmented_condition_length.png`
- `figures/segmented_recovery_by_condition.png`
- `figures/segmented_single_vs_multi_topic.png`
