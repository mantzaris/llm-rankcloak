# Segmented Quality Controls Pilot

## Why This Experiment Exists

The first segmented protocol pilot showed exact recovery for the compact control request and all response trials. It also showed that greedy tails make public messages look more natural, but the resulting full-message metrics are dominated by non-payload tail tokens. This follow-up makes that distinction explicit.

The pilot tests four focused quality controls:

- forced-prefix metrics separated from full-message metrics;
- sentence-boundary natural tails instead of fixed-length tails only;
- natural tails on compact control requests;
- deterministic `safe_text_filter_v1` token filtering to reduce obvious markup/code/placeholder artifacts.

## Conditions

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Payload codec:

- `raw_hex_nibbles`

Segment size:

- 8 ranks

Conditions:

- `segmented_single_topic_fixed_tail40_unfiltered`
- `segmented_single_topic_sentence_tail_unfiltered`
- `segmented_multi_topic_sentence_tail_unfiltered`
- `segmented_single_topic_sentence_tail_filtered`
- `segmented_multi_topic_sentence_tail_filtered`

Prompt set:

- `recipe_long_specific`
- `recipe_forum_exchange_specific`
- `recipe_blog`
- `grocery_planning_note_specific`
- `plant_care_note_specific`

## Expected Interpretation

An improvement should not be judged by exact recovery alone. Useful signals include:

- control requests recover exactly and look less like isolated artificial fragments;
- full-message quality improves without hiding a severely broken forced prefix;
- sentence tails reduce abrupt endings versus fixed tails;
- filtered runs reduce artifact counts without causing recovery failures;
- forced-prefix metrics remain visible and are not averaged away by tails.

## Limitations

- The control code is a synthetic codebook label, not a secret or operational command.
- The filter is heuristic and deterministic, not a learned detector.
- Full-message quality may improve because of tail tokens that do not carry payload ranks.
- Exact-copy conditions remain required.
- This is not encryption, key exchange, authentication, signing, or cryptographic security.

## Command

```bash
python3 scripts/run_experiment.py \
  --profile segmented-quality-controls \
  --output-dir results/rankcloak_segmented_quality_controls \
  --overwrite
```

## Completed Run

Directory:

```text
results/rankcloak_segmented_quality_controls/
```

Recovery:

- control request trials: 5/5
- response trials: 10/10
- response failures: 0

Condition means from `segmented_quality_trials.csv`:

| Condition | Filter | Tail policy | Forced logprob | Full logprob | Full repetition | Full artifacts |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `segmented_single_topic_fixed_tail40_unfiltered` | none | fixed_tail40 | -4.6816 | -1.3087 | 0.1966 | 0.1875 |
| `segmented_single_topic_sentence_tail_unfiltered` | none | sentence_tail_min20_max60 | -4.6816 | -1.3972 | 0.1742 | 0.1875 |
| `segmented_multi_topic_sentence_tail_unfiltered` | none | sentence_tail_min20_max60 | -4.8773 | -1.2928 | 0.1457 | 0.1250 |
| `segmented_single_topic_sentence_tail_filtered` | safe_text_filter_v1 | sentence_tail_min20_max60 | -4.6531 | -1.4824 | 0.1736 | 0.0000 |
| `segmented_multi_topic_sentence_tail_filtered` | safe_text_filter_v1 | sentence_tail_min20_max60 | -4.7132 | -1.2813 | 0.1693 | 0.0000 |

Initial interpretation:

- Separating metrics confirms that full-message quality is much better than forced-prefix quality because tails dominate the public text.
- Sentence tails reduced average tail length versus fixed 40-token tails in several conditions, but did not uniformly improve full-message log probability.
- `safe_text_filter_v1` eliminated the tracked artifact flags in this run without breaking recovery.
- Filtered controls looked better by artifact flags, though some markdown-like emphasis still remained because the first filter version does not reject asterisks.
