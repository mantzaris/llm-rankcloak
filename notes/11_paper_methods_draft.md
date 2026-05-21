# Paper Methods Draft

This draft is intended for later adaptation into a journal manuscript. It describes what the repository currently implements and avoids claims of cryptographic security.

## Overview

We evaluate RankCloak, a local LLM rank-transcoding framework for hiding deterministic synthetic cryptographic-artifact-like payloads in generated cover text under exact-copy conditions. The experiments compare raw subword payload ranks, bounded-rank payload encodings, prompt-key cover generation, prompt-family effects, segmented multi-cover variants, natural tails, and deterministic token filtering.

The method assumes that the sender and receiver already share `K_common`, which includes the model file, tokenizer, quantization, rank-ordering rule, prompt templates, payload codec, segment-size rule, optional token filter, tail policy, and decode policy. No key exchange is implemented or claimed. The compact control code used in segmented experiments is a simulated codebook label, not a secret key.

## Synthetic Payload Generation

All payloads are generated deterministically in `rankcloak/synthetic_payloads.py`. Payload classes include a SHA-256 digest of a public test string, fixed-seed random hex strings, a synthetic UUIDv4-like string, a fake bearer-token-like string, an invalid synthetic JWT-like string, a fake HMAC-like tag, a nonce-like hex string, and a base64 ciphertext-like block.

The payloads are designed to resemble high-entropy artifacts while avoiding real secrets, credentials, accounts, private keys, API tokens, or operational values.

## Payload Representations

### Direct Subword

In the direct subword representation, the artifact string is tokenized by the loaded LLM tokenizer. Each payload token is evaluated under the model and assigned its stable next-token rank. This representation often uses fewer ranks than bounded-rank coding, but high-entropy strings can produce very large ranks, which makes plausible cover generation difficult.

Implementation: `rankcloak/rank_codec.py` functions `direct_subword_ranks_for_text` and `rank_trace_from_token_ids`.

Primary artifacts: `rank_statistics.csv` and `payload_granularity_comparison.csv`.

### ASCII-Byte Fixed-Radix

In the ASCII-byte fixed-radix representation, the displayed artifact string is encoded as UTF-8 bytes. The byte stream is converted to base-B digits for B in 2, 4, 8, 16, 32, or 64. Each digit `d` maps to rank `d + 1`, so all generated payload ranks are bounded by B.

This representation provides direct control over rank pressure. Smaller B values produce more ranks and longer covers but keep generation closer to high-probability model tokens.

Implementation: `rankcloak/rank_codec.py` functions `encode_bytes_to_bounded_ranks` and `decode_bounded_ranks_to_bytes`.

Primary artifacts: `codec_roundtrip_trials.csv`, `stegotext_recovery_trials.csv`, and `cover_examples.jsonl`.

### Raw Hex-Nibble

For hex-like payloads, the raw hex-nibble representation maps each hex character directly to one rank from 1 through 16. Characters `0` through `f` map to ranks 1 through 16.

This representation is more efficient than encoding the ASCII bytes of a hex string. For example, a 64-character SHA-256 hex digest requires 64 ranks as raw hex nibbles, rather than 128 ranks as ASCII bytes at B=16.

Implementation: `rankcloak/rank_codec.py` functions `encode_hex_nibbles_to_ranks` and `decode_hex_nibble_ranks_to_text`.

Primary artifacts: `payload_granularity_comparison.csv`, `segmented_protocol_trials.csv`, and `segmented_quality_trials.csv`.

## Rank Ordering And Deterministic Tie-Breaking

Ranks are 1-indexed. For a logit vector, tokens are sorted by descending logit value. Ties are broken by token id in ascending order. Equivalently, a target token rank is one plus the number of tokens with higher logits plus the number of tied tokens with lower token ids.

Implementation: `rankcloak/rank_codec.py` functions `sorted_token_ids_from_logits`, `rank_of_token`, and `token_id_at_rank`.

This deterministic rule is required for exact recovery.

## Cover Generation With Prompt Keys

A cover prompt is tokenized to form the initial autoregressive context. For each payload rank, the model is evaluated at the current context, the token at that 1-indexed rank is selected, and the token is appended to the context. The generated token sequence is detokenized for public cover text.

Implementation: `rankcloak/rank_codec.py` function `generate_token_ids_from_ranks`; prompt registry in `rankcloak/prompts.py`.

The prompts are original and non-copyrighted. They include recipe writing, forum replies, technical documentation, dialogue, car-buying discussion, biology education, grocery planning, and plant-care notes.

## Exact Recovery Procedure

Recovery uses the same model, tokenizer, quantization, prompt, rank-ordering rule, and any optional token filter. The receiver evaluates the prompt context and each received token in sequence, recomputes the token rank at each step, and decodes the recovered rank sequence back to the payload representation.

Implementation: `rankcloak/rank_codec.py` function `recover_ranks_from_generated_ids`; filtered variant in `rankcloak/segmented_protocol.py`.

Exact recovery requires the public text channel to preserve the generated token sequence exactly.

## Prompt Families And Cover Genres

Prompt-family experiments compare short and long recipe prompts, biology explanations, car-buying posts, forum replies, technical documentation, stage dialogue, and dialogue-style prompt formats. The prompt registry also assigns prompt-family labels for analysis.

Implementation: `rankcloak/prompts.py`.

Primary artifacts: `PROMPT_COMPARISON.md`, `DIALOGUE_PROMPT_COMPARISON.md`, and `cover_text_features.csv`.

## Baseline Cover Generation

Baseline cover text is generated by selecting the rank-1 token greedily from the same prompt context for an approximate target token count. These baselines provide a local reference for feature extraction but are not a detector by themselves.

Implementation: `rankcloak/baselines.py`.

Primary artifacts: `baseline_cover_examples.jsonl` and baseline rows in `cover_text_features.csv`.

## Segmented Multi-Cover RankCloak

The segmented protocol tests whether splitting a payload into short rank chunks reduces visible drift compared with one long forced-rank message. A simulated sender first hides compact control code `C1`, which maps to a pre-agreed response configuration. The response payload is encoded with raw hex nibbles, split into fixed-size chunks, and sent as multiple cover messages.

The receiver decodes only the payload-bearing forced prefix of each message, ignores optional natural tails, concatenates recovered rank chunks, and decodes the payload.

Implementation: `rankcloak/segmented_protocol.py`.

Primary artifacts: `control_request_trial.jsonl`, `segmented_protocol_trials.csv`, `segmented_protocol_messages.jsonl`, and `SEGMENTED_PROTOCOL_COMPARISON.md`.

## Control-Code Simulation

The current codebook contains control code `C1`. It maps to a synthetic SHA-256 hex response, raw hex-nibble payload codec, segment size 8, a prompt schedule, natural tail setting, and forced-prefix-only decode policy.

The control code is a compact codebook label in a local simulation. It is not a secret key, not an operational command, and not a key-exchange mechanism.

Implementation: `CONTROL_CODEBOOK` in `rankcloak/segmented_protocol.py`.

## Natural Tails And Forced-Prefix Decoding

Segmented messages can include a forced prefix followed by greedy natural tail tokens. The forced prefix carries payload ranks. The tail is generated to make the public message less fragmentary and is ignored during decoding.

This design requires metrics to separate payload-bearing forced-prefix quality from full-message quality.

Implementation: `rankcloak/segmented_protocol.py` functions `generate_forced_prefix_with_natural_tail` and `generate_quality_message`.

## Sentence-Boundary Tail Policy

The `sentence_tail_min20_max60` policy generates at least 20 greedy tail tokens and then stops once the decoded tail appears to end at a likely sentence boundary, with a maximum of 60 tail tokens.

Likely boundaries include period, question mark, exclamation point, and simple closing quote forms after those punctuation marks.

Implementation: `rankcloak/segmented_protocol.py` functions `likely_sentence_boundary` and `should_stop_sentence_tail`.

## Deterministic Safe-Text Token Filtering

The optional `safe_text_filter_v1` rejects decoded token pieces that resemble code fences, markdown headings, LaTeX fragments, HTML fragments, URLs, bracket placeholders, repeated underscores, pipe-table fragments, control characters, or replacement characters.

When this filter is used, both generation and recovery rank over the same filtered token set. If a received token is not allowed under the filter, recovery fails cleanly rather than silently producing a wrong rank.

Implementation: `rankcloak/token_filters.py`.

## Metrics

### Exact Recovery

Exact recovery is recorded as a Boolean for codec roundtrips, stegotext recovery trials, segmented response trials, and control requests.

### Rank Count

Rank count is the number of payload ranks required by a representation. It is used as a cover-length proxy.

### Rank Pressure

Rank pressure is summarized by mean rank, median rank, p95 rank, max rank, and fractions below small rank thresholds. Larger ranks generally force lower-probability tokens.

### Token Log Probability

Generated token log probabilities are computed from model logits at selection time. Mean and median token log probabilities are used as lightweight quality proxies.

### Repetition

Unique-token and repeated-token fractions are computed from token ids where available.

### Punctuation And Artifact Flags

Character-level features include punctuation, whitespace, digit, and alphabetic fractions. Artifact flags detect simple patterns such as backticks, bracket placeholders, URL fragments, LaTeX fragments, HTML fragments, and markdown headings.

### Forced-Prefix Metrics

Forced-prefix metrics describe only payload-bearing generated tokens.

### Full-Message Metrics

Full-message metrics describe the forced prefix plus natural tail. These metrics can look better because tails are greedy high-probability text and do not carry payload ranks.

Implementation: `rankcloak/metrics.py`, `rankcloak/segmented_protocol.py`, and `rankcloak/schemas.py`.

## Reproducibility And Local CPU Execution

Experiments run locally with `llama-cpp-python` and a local Llama 3 8B Instruct GGUF model. The runner records manifests containing timestamp, git state, Python version, platform, CPU count, package versions, model identity, model size, model hash when available, profile, output directory, and command-line arguments.

Implementation: `rankcloak/reproducibility.py`; execution entry points in `scripts/run_experiment.py`, `scripts/run_smoke.py`, and `rankcloak/cli.py`.

## Safety, Scope, And Limitations

This study uses only deterministic synthetic payloads. It does not use real secrets, credentials, API keys, private keys, accounts, or services. It does not claim encryption, key exchange, authentication, signing, digital signatures, credential handling, or cryptographic security.

The current experiments measure exact-copy concealment behavior. They do not establish undetectability, edit robustness, cross-model portability, or human acceptability.
