# Results Index

This note maps each current `results/` directory to its scientific meaning and paper use. All expected result directories listed in the paper-notes task were found.

## High-Level Index

| Result directory | Experiment type | Main question | Recovery result | Paper use |
| --- | --- | --- | --- | --- |
| `results/rankcloak_crypto_artifact_exploration/` | smoke prototype | Does the initial notebook and runner produce recoverable bounded-rank covers? | 4/4 stegotext, 63/63 codec | Early reproducibility and notebook artifact |
| `results/rankcloak_small_full/` | small sweep | How do payloads, prompts, and B=8..64 behave? | 64/64 stegotext, 63/63 codec | Main pilot for alphabet-size effects |
| `results/rankcloak_strong_prompt_pilot/` | prompt pilot | Do long specific prompts run correctly on a small matrix? | 16/16 stegotext, 63/63 codec | Prompt experiment sanity check |
| `results/rankcloak_strong_prompt_sweep/` | strong prompt sweep | Do longer prompts improve cover quality? | 60/60 stegotext, 63/63 codec | Prompt-specific evidence |
| `results/rankcloak_dialogue_key_pilot/` | dialogue pilot | Do dialogue or forum formats absorb damage better at B=8 and B=16? | 24/24 stegotext, 63/63 codec | Prompt-format evidence |
| `results/rankcloak_payload_granularity_pilot/` | representation pilot | How many ranks do different payload representations need? | 63/63 codec; no stegotext trials | Payload representation evidence |
| `results/rankcloak_segmented_protocol_pilot/` | segmented protocol | Does multi-cover segmentation recover and reduce drift? | control true, 10/10 responses | Protocol-variant evidence |
| `results/rankcloak_segmented_quality_controls/` | segmented quality controls | Do sentence tails, metric separation, and token filtering help? | 5/5 controls, 10/10 responses | Clearest current quality-control evidence |

## `results/rankcloak_crypto_artifact_exploration/`

Profile or experiment name: `smoke`.

Purpose: initial model-backed smoke test and notebook output directory.

Payloads used: one payload, `sha256_public_test_string`, sliced to the first 8 bytes for stegotext generation.

Prompts or conditions used: `play_dialogue`, `recipe_blog`.

Model status: loaded.

Recovery result summary: `summary.json` records 4 stegotext passes, 0 stegotext failures, 63 codec passes, and 0 codec failures.

Main metrics available: tokenization audit, direct rank statistics for one payload, codec roundtrip, stegotext recovery, cover features, baseline features.

Key output files: `summary.json`, `SUMMARY.md`, `MANIFEST.json`, `tokenization_audit.csv`, `rank_statistics.csv`, `codec_roundtrip_trials.csv`, legacy `recovery_trials.csv`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, `baseline_cover_examples.jsonl`, `cover_text_features.csv`.

Figures available: `token_count_by_payload.png`, `rank_summary_direct_subword.png`, `cover_length_vs_rank_alphabet.png`, `recovery_by_cover_prompt_and_alphabet.png`, `cover_text_feature_comparison.png`.

How this result can be used in the paper: cite as the first reproducible smoke artifact and as proof that the notebook and scripts write the standard outputs.

Known caveats: stegotext payload is intentionally short; use later result directories for substantive comparisons.

## `results/rankcloak_small_full/`

Profile or experiment name: `small`.

Purpose: first full-payload empirical sweep over four payloads, four prompts, and four alphabet sizes.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`, `random_256_bit_hex`, `synthetic_uuid_v4_like`.

Prompts or conditions used: `play_dialogue`, `recipe_blog`, `forum_reply`, `technical_documentation`.

Model status: loaded.

Recovery result summary: `summary.json` records 64 stegotext passes, 0 stegotext failures, 63 codec passes, and 0 codec failures.

Main metrics available: tokenization audit, direct subword rank statistics, codec roundtrip, stegotext recovery, cover-text features, greedy baselines.

Key output files: `summary.json`, `SUMMARY.md`, `MANIFEST.json`, `tokenization_audit.csv`, `rank_statistics.csv`, `codec_roundtrip_trials.csv`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, `baseline_cover_examples.jsonl`, `cover_text_features.csv`.

Figures available: standard five figures.

How this result can be used in the paper: primary pilot evidence that B=8 and B=16 have better cover-quality proxies than B=32 and B=64 while all current exact-copy trials recover.

Known caveats: no detector AUC; prompt set is small; one local model and quantization.

## `results/rankcloak_strong_prompt_pilot/`

Profile or experiment name: `strong-prompts-pilot`.

Purpose: small prompt-specific pilot before the full strong prompt sweep.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`.

Prompts or conditions used: `recipe_blog`, `recipe_long_specific`, `biology_long_specific`, `car_buying_long_specific`.

Model status: loaded.

Recovery result summary: `summary.json` records 16 stegotext passes, 0 stegotext failures, 63 codec passes, and 0 codec failures.

Main metrics available: rank statistics, cover features, prompt comparison examples, baseline examples.

Key output files: `summary.json`, `SUMMARY.md`, `MANIFEST.json`, `PROMPT_COMPARISON.md`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, `cover_text_features.csv`.

Figures available: standard figures plus strong prompt comparison figures.

How this result can be used in the paper: method-development artifact validating the long-prompt comparison code path.

Known caveats: pilot size is smaller than `rankcloak_strong_prompt_sweep`.

## `results/rankcloak_strong_prompt_sweep/`

Profile or experiment name: `strong-prompts`.

Purpose: compare current short prompt and long specific prompts across low and high bounded-rank alphabets.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`, `synthetic_uuid_v4_like`.

Prompts or conditions used: `recipe_blog`, `recipe_long_specific`, `biology_long_specific`, `car_buying_long_specific`, `forum_reply`.

Model status: loaded.

Recovery result summary: `summary.json` records 60 stegotext passes, 0 stegotext failures, 63 codec passes, and 0 codec failures.

Main metrics available: exact recovery, mean token log probability, repetition, punctuation, prompt length, rank pressure, baseline features.

Key output files: `summary.json`, `SUMMARY.md`, `MANIFEST.json`, `PROMPT_COMPARISON.md`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, `baseline_cover_examples.jsonl`, `cover_text_features.csv`.

Figures available: standard figures plus `strong_prompt_mean_logprob_by_prompt.png`, `strong_prompt_recovery_by_prompt.png`, `strong_prompt_length_by_prompt.png`, `strong_prompt_rank_pressure.png`.

How this result can be used in the paper: evidence that prompt specificity helps topic anchoring but does not overcome high rank pressure at larger alphabets.

Known caveats: quality notes are heuristic and manual-inspection oriented.

## `results/rankcloak_dialogue_key_pilot/`

Profile or experiment name: `dialogue-key-pilot`.

Purpose: compare dialogue and forum-exchange prompts against monologue prompts at B=8 and B=16.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`.

Prompts or conditions used: `recipe_blog`, `recipe_long_specific`, `recipe_dialogue_specific`, `recipe_forum_exchange_specific`, `car_buying_dialogue_specific`, `biology_tutor_dialogue_specific`.

Model status: loaded.

Recovery result summary: `summary.json` records 24 stegotext passes, 0 stegotext failures, 63 codec passes, and 0 codec failures.

Main metrics available: exact recovery, mean token log probability, repetition, punctuation, token count, prompt family, prompt length.

Key output files: `summary.json`, `SUMMARY.md`, `MANIFEST.json`, `DIALOGUE_PROMPT_COMPARISON.md`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, `cover_text_features.csv`.

Figures available: standard figures plus dialogue mean log-probability, repetition, length, and scatter figures.

How this result can be used in the paper: focused evidence that conversational prompts are not automatically cleaner than monologue prompts.

Known caveats: manual inspection remains necessary; B=32 and B=64 intentionally excluded.

## `results/rankcloak_payload_granularity_pilot/`

Profile or experiment name: `payload-granularity-pilot`.

Purpose: compare payload-side representations without changing the model tokenizer.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`.

Prompts or conditions used: none for cover generation.

Model status: loaded.

Recovery result summary: `summary.json` records 63 codec passes, 0 codec failures, and 0 stegotext trials.

Main metrics available: rank count, max possible or observed rank, bits per rank estimate, direct subword rank pressure.

Key output files: `payload_granularity_comparison.csv`, `rank_statistics.csv`, `codec_roundtrip_trials.csv`, `summary.json`, `MANIFEST.json`.

Figures available: standard placeholder/relevant figures plus `payload_representation_rank_count.png`.

How this result can be used in the paper: motivate hex-nibble coding for hex artifacts and distinguish payload-side granularity from changing the cover-side tokenizer.

Known caveats: this profile does not generate full covers for each representation.

## `results/rankcloak_segmented_protocol_pilot/`

Profile or experiment name: `segmented-protocol-pilot`.

Purpose: test a two-stage segmented multi-cover protocol with a synthetic control code and forced-prefix-only response decoding.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`.

Prompts or conditions used: `single_long_recipe_no_tail`, `single_long_recipe_tail40`, `segmented_single_topic_no_tail`, `segmented_single_topic_tail40`, `segmented_multi_topic_tail40`.

Model status: loaded.

Recovery result summary: `summary.json` records control request exact recovery as true and 10 response passes with 0 response failures.

Main metrics available: exact response recovery, message count, total generated token count, log probability, repetition, punctuation, alphabetic fraction.

Key output files: `control_request_trial.jsonl`, `segmented_protocol_trials.csv`, `segmented_protocol_messages.jsonl`, `cover_text_features.csv`, `SEGMENTED_PROTOCOL_COMPARISON.md`, `summary.json`, `MANIFEST.json`.

Figures available: segmented condition mean log-probability, repetition, length, recovery, and single-vs-multi-topic figures.

How this result can be used in the paper: protocol-variant pilot showing segmentation and natural tails under exact-copy recovery.

Known caveats: tail40 full-message metrics are dominated by natural greedy tails; no-tail messages are often fragmentary.

## `results/rankcloak_segmented_quality_controls/`

Profile or experiment name: `segmented-quality-controls`.

Purpose: refine segmented-protocol evaluation by separating forced-prefix and full-message metrics, adding sentence-boundary tails, adding natural control tails, and testing a deterministic safe-text filter.

Payloads used: `sha256_public_test_string`, `random_128_bit_hex`.

Prompts or conditions used: five segmented quality-control conditions using recipe, forum, grocery, and plant-care prompts.

Model status: loaded.

Recovery result summary: `summary.json` records 5 control passes, 0 control failures, 10 response passes, and 0 response failures.

Main metrics available: forced-prefix and full-message log probability, repetition, punctuation, alphabetic fraction, artifact counts, tail token counts, token filter labels, exact recovery.

Key output files: `control_request_trials.jsonl`, `segmented_quality_trials.csv`, `segmented_quality_messages.jsonl`, `cover_text_features.csv`, `SEGMENTED_QUALITY_COMPARISON.md`, `summary.json`, `MANIFEST.json`.

Figures available: `quality_forced_vs_full_logprob.png`, `quality_forced_vs_full_repetition.png`, `quality_tail_policy_logprob.png`, `quality_filter_effect_logprob.png`, `quality_filter_effect_artifacts.png`, `quality_recovery_by_condition.png`.

How this result can be used in the paper: current best diagnostic artifact for distinguishing payload-bearing quality from full-message quality.

Known caveats: safe-text filtering is heuristic; full-message improvements can still be tail-driven.

## Cross-Directory Caveats

- All current results use deterministic synthetic payloads.
- All current recovery results assume exact-copy text preservation.
- All current model-backed results use one local Llama 3 8B Instruct GGUF model family.
- No current result file records detector AUC.
- Human readability judgments have not been run as a controlled study.
