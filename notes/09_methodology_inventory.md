# Methodology Inventory

This note inventories the RankCloak methods currently implemented in the repository and
maps each method to code, experiment profiles, result directories, and paper roles. It
is written for manuscript preparation, not as a claim of cryptographic security.

## Project Framing

RankCloak studies LLM rank-transcoding steganography for deterministic synthetic
cryptographic-artifact-like payloads. The project asks how high-entropy strings behave
when represented as model-token ranks or bounded rank sequences and then expressed as
generated cover text.

The work is a concealment and measurement study under exact-copy conditions. It does not
implement or claim encryption, key exchange, authentication, signing, credential
handling, digital signatures, or cryptographic security. All payloads are deterministic
synthetic examples.

## Shared Assumptions

The experiments assume the sender and receiver already share `K_common`. In this
repository, `K_common` means:

- exact model file;
- tokenizer;
- quantization;
- deterministic rank-ordering rule;
- payload codec;
- prompt templates;
- segmentation rule, where used;
- tail policy, where used;
- token filter, where used;
- forced-prefix decode rule, where used.

No key exchange is implemented. Control code `C1` is a simulated compact codebook label,
not a secret key and not an operational command.

## Exact-Copy Recovery Setting

Short name: exact-copy channel.

Purpose: test whether generated cover text can be decoded exactly when the receiver sees
the same unmodified token sequence.

Implemented location in code: `rankcloak/rank_codec.py`, `rankcloak/experiments.py`,
`rankcloak/segmented_protocol.py`.

Corresponding experiment profile names: `smoke`, `small`, `strong-prompts-pilot`,
`strong-prompts`, `dialogue-key-pilot`, `segmented-protocol-pilot`,
`segmented-quality-controls`.

Corresponding result directories: all model-backed directories under `results/`.

Key input variables: model path, prompt name, payload representation, alphabet size or
segment size, optional tail policy, optional token filter.

Key output files: `stegotext_recovery_trials.csv`, `segmented_protocol_trials.csv`,
`segmented_quality_trials.csv`, `cover_examples.jsonl`,
`segmented_protocol_messages.jsonl`, `segmented_quality_messages.jsonl`.

Intended paper role: define the controlled recovery setting and separate correctness
from cover quality.

Limitations: any text edit, tokenizer mismatch, model mismatch, quantization mismatch,
or rank-ordering mismatch can break recovery.

## Model And Tokenizer Dependence

Short name: local Llama GGUF dependency.

Purpose: use a local autoregressive model and its native tokenizer to define both token
ranks and generated cover text.

Implemented location in code: `rankcloak/model_io.py`, `rankcloak/reproducibility.py`.

Corresponding experiment profile name: every model-backed profile.

Corresponding result directories: model-loaded summaries are present in all current
result directories.

Key input variables: `models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`, context
length, thread count, `logits_all=True`.

Key output files: `MANIFEST.json`, `summary.json`.

Intended paper role: describe the exact model environment required for reproduction.

Limitations: results are model-specific and quantization-specific.

## Rank-Transcoding Background

Short name: Calgacus-style rank transcode.

Purpose: represent a payload as ranks under one context and generate a cover by
selecting tokens at matching ranks under a prompt-key context.

Implemented location in code: `rankcloak/rank_codec.py`.

Corresponding experiment profile names: all stegotext profiles.

Corresponding result directories: `results/rankcloak_crypto_artifact_exploration/`,
`results/rankcloak_small_full/`, prompt pilots, segmented pilots.

Key input variables: context token ids, target ranks, deterministic rank ordering.

Key output files: `rank_statistics.csv`, `cover_examples.jsonl`, recovery CSVs.

Intended paper role: provide the conceptual bridge from the Calgacus paper to
high-entropy cryptographic-artifact-like payloads.

Limitations: high-entropy payload tokens often land far down the next-token
distribution, which can damage cover plausibility.

## Direct Subword Encoding

Short name: raw subword direct.

Purpose: tokenize the artifact string directly with the model tokenizer and measure each
token rank under the model.

Implemented location in code: `rankcloak/rank_codec.py` function
`direct_subword_ranks_for_text`; `rankcloak/experiments.py` function
`run_rank_statistics`.

Corresponding experiment profile names: `audit-only`, `small`, `strong-prompts-pilot`,
`strong-prompts`, `dialogue-key-pilot`, `payload-granularity-pilot`.

Corresponding result directories: result directories containing `rank_statistics.csv`.

Key input variables: payload text, model tokenizer, BOS context.

Key output files: `rank_statistics.csv`, `payload_granularity_comparison.csv`.

Intended paper role: baseline showing why raw high-entropy strings are difficult
payloads.

Limitations: it does not necessarily generate a cover text in current pilots; it
measures rank pressure.

## ASCII-Byte Fixed-Radix Bounded-Rank Encoding

Short name: ASCII bytes fixed radix.

Purpose: encode artifact display bytes into base-B digits and map digits to 1-indexed
ranks from 1 through B.

Implemented location in code: `rankcloak/rank_codec.py` functions
`encode_bytes_to_bounded_ranks` and `decode_bounded_ranks_to_bytes`.

Corresponding experiment profile names: `codec-only`, `smoke`, `small`,
`strong-prompts-pilot`, `strong-prompts`, `dialogue-key-pilot`,
`payload-granularity-pilot`.

Corresponding result directories: all standard non-segmented result directories.

Key input variables: payload bytes, alphabet size B in 2, 4, 8, 16, 32, 64.

Key output files: `codec_roundtrip_trials.csv`, `stegotext_recovery_trials.csv`,
`cover_examples.jsonl`.

Intended paper role: controlled bounded-rank payload representation for exact recovery
and alphabet-size comparisons.

Limitations: lower B improves rank pressure but increases rank count and cover length.

## Hex-Nibble Character-Level Payload Encoding

Short name: raw hex nibbles.

Purpose: encode hex-like payload text one character at a time, mapping `0` through `f`
to ranks 1 through 16.

Implemented location in code: `rankcloak/rank_codec.py` functions
`encode_hex_nibbles_to_ranks` and `decode_hex_nibble_ranks_to_text`;
`rankcloak/segmented_protocol.py`.

Corresponding experiment profile names: `payload-granularity-pilot`,
`segmented-protocol-pilot`, `segmented-quality-controls`.

Corresponding result directories: `results/rankcloak_payload_granularity_pilot/`,
`results/rankcloak_segmented_protocol_pilot/`,
`results/rankcloak_segmented_quality_controls/`.

Key input variables: lowercase hex payload text, segment size where segmented.

Key output files: `payload_granularity_comparison.csv`, `segmented_protocol_trials.csv`,
`segmented_quality_trials.csv`.

Intended paper role: efficient bounded-rank representation for hex artifacts such as
hashes and random hex strings.

Limitations: only applies directly to hex-like payloads.

## Prompt-Key Cover Generation

Short name: prompt-key RankCloak cover.

Purpose: use an original prompt as the cover key context; for each requested rank,
select the token at that rank under the current context.

Implemented location in code: `rankcloak/rank_codec.py`, `rankcloak/prompts.py`,
`rankcloak/experiments.py`.

Corresponding experiment profile names: all stegotext profiles except
`payload-granularity-pilot` when it only summarizes representations.

Corresponding result directories: non-empty `cover_examples.jsonl`,
`segmented_protocol_messages.jsonl`, and `segmented_quality_messages.jsonl` directories.

Key input variables: prompt name, payload ranks, alphabet size, model logits.

Key output files: cover JSONL files and recovery CSV files.

Intended paper role: operational core for comparing prompt genres and payload encodings.

Limitations: prompt quality cannot fully compensate for large forced ranks.

## Prompt Topic And Prompt Format Experiments

Short name: prompt family sweep.

Purpose: test whether topic and format choices absorb forced-rank damage differently.

Implemented location in code: `rankcloak/prompts.py`, `rankcloak/experiments.py`,
`rankcloak/plotting.py`.

Corresponding experiment profile names: `small`, `strong-prompts-pilot`,
`strong-prompts`, `dialogue-key-pilot`.

Corresponding result directories: `results/rankcloak_small_full/`,
`results/rankcloak_strong_prompt_pilot/`, `results/rankcloak_strong_prompt_sweep/`,
`results/rankcloak_dialogue_key_pilot/`.

Key input variables: prompt name, prompt family, alphabet size, payload.

Key output files: `cover_text_features.csv`, `PROMPT_COMPARISON.md`,
`DIALOGUE_PROMPT_COMPARISON.md`, prompt figures.

Intended paper role: evaluate cover genre and prompt specificity as experimental
factors.

Limitations: current sweeps are pilots and not powered for statistical claims.

## Strong Prompt Sweep

Short name: strong-prompts.

Purpose: compare short prompts against longer, more specific prompts in recipe, biology,
car-buying, and forum styles.

Implemented location in code: `rankcloak/prompts.py`, `rankcloak/experiments.py`.

Corresponding experiment profile names: `strong-prompts-pilot`, `strong-prompts`.

Corresponding result directories: `results/rankcloak_strong_prompt_pilot/`,
`results/rankcloak_strong_prompt_sweep/`.

Key input variables: payloads, B values, prompt set.

Key output files: `PROMPT_COMPARISON.md`, `stegotext_recovery_trials.csv`,
`cover_text_features.csv`.

Intended paper role: show that prompt specificity helps topical anchoring but does not
remove rank-pressure artifacts.

Limitations: manual quality inspection is still required.

## Dialogue Key Prompt Pilot

Short name: dialogue-key-pilot.

Purpose: test whether dialogue and forum-exchange formats are more tolerant of low-rank
forced tokens than monologue prose.

Implemented location in code: `rankcloak/prompts.py`, `rankcloak/experiments.py`.

Corresponding experiment profile name: `dialogue-key-pilot`.

Corresponding result directory: `results/rankcloak_dialogue_key_pilot/`.

Key input variables: recipe, car-buying, and biology dialogue prompts; B=8 and B=16.

Key output files: `DIALOGUE_PROMPT_COMPARISON.md`, `cover_text_features.csv`, dialogue
figures.

Intended paper role: focused prompt-format comparison at the more plausible bounded-rank
settings.

Limitations: dialogue improved some log-probability metrics but also introduced
repetition and formatting artifacts.

## Payload Granularity Pilot

Short name: payload-granularity-pilot.

Purpose: compare the number of ranks needed for ASCII byte fixed-radix, raw hex-nibble,
and raw subword direct representations.

Implemented location in code: `rankcloak/experiments.py`, `rankcloak/rank_codec.py`.

Corresponding experiment profile name: `payload-granularity-pilot`.

Corresponding result directory: `results/rankcloak_payload_granularity_pilot/`.

Key input variables: selected hex payloads, B=8 and B=16, model availability for direct
subword ranks.

Key output files: `payload_granularity_comparison.csv`,
`figures/payload_representation_rank_count.png`.

Intended paper role: motivate payload-side representation choices without changing the
cover-side tokenizer.

Limitations: it does not run full cover generation for every representation.

## Two-Stage Segmented Multi-Cover RankCloak

Short name: segmented protocol.

Purpose: split a payload into short rank chunks and send multiple cover messages,
optionally with natural tails, to test whether segmentation reduces cover drift.

Implemented location in code: `rankcloak/segmented_protocol.py`.

Corresponding experiment profile name: `segmented-protocol-pilot`.

Corresponding result directory: `results/rankcloak_segmented_protocol_pilot/`.

Key input variables: `C1` control code, raw hex nibbles, segment size 8, prompt
schedule, tail length.

Key output files: `control_request_trial.jsonl`, `segmented_protocol_trials.csv`,
`segmented_protocol_messages.jsonl`, `SEGMENTED_PROTOCOL_COMPARISON.md`.

Intended paper role: protocol-variant demonstration separating long forced spans from
multiple short messages.

Limitations: more messages and tails increase public communication length; no-tail
segments can be fragmentary.

## Segmented Quality Controls

Short name: segmented-quality-controls.

Purpose: improve measurement of segmented covers by separating forced-prefix and
full-message metrics, adding sentence-boundary tails, adding control tails, and testing
`safe_text_filter_v1`.

Implemented location in code: `rankcloak/segmented_protocol.py`,
`rankcloak/token_filters.py`, `rankcloak/metrics.py`.

Corresponding experiment profile name: `segmented-quality-controls`.

Corresponding result directory: `results/rankcloak_segmented_quality_controls/`.

Key input variables: two hex payloads, segment size 8, five quality-control conditions.

Key output files: `control_request_trials.jsonl`, `segmented_quality_trials.csv`,
`segmented_quality_messages.jsonl`, `SEGMENTED_QUALITY_COMPARISON.md`, quality figures.

Intended paper role: scientifically cleaner quality assessment for segmented covers.

Limitations: tails can make full messages look better while forced prefixes remain
constrained.

## Natural Tail Policies

Short name: greedy natural tail.

Purpose: append rank-1 greedy tokens after the payload-bearing forced prefix so public
messages can continue more naturally.

Implemented location in code: `rankcloak/segmented_protocol.py`.

Corresponding experiment profile names: `segmented-protocol-pilot`,
`segmented-quality-controls`.

Corresponding result directories: segmented result directories.

Key input variables: natural tail token count or tail policy.

Key output files: segmented message JSONL files and trial CSVs.

Intended paper role: separate payload-bearing tokens from public-message smoothing
tokens.

Limitations: tails do not carry payload in current pilots and can dominate full-message
metrics.

## Sentence-Boundary Tail Policy

Short name: `sentence_tail_min20_max60`.

Purpose: stop greedy tails after at least 20 tokens once a likely sentence boundary is
reached, with a hard cap of 60 tokens.

Implemented location in code: `rankcloak/segmented_protocol.py` functions
`likely_sentence_boundary` and `should_stop_sentence_tail`.

Corresponding experiment profile name: `segmented-quality-controls`.

Corresponding result directory: `results/rankcloak_segmented_quality_controls/`.

Key input variables: generated tail text, minimum tail tokens, maximum tail tokens.

Key output files: `segmented_quality_messages.jsonl`, `segmented_quality_trials.csv`.

Intended paper role: avoid abrupt fixed-length endings in public messages.

Limitations: sentence-boundary heuristics are simple and language-specific.

## Deterministic Safe-Text Token Filtering

Short name: `safe_text_filter_v1`.

Purpose: restrict forced generation and recovery to a deterministic allowed-token set
that rejects obvious markup, code, URL, bracket-placeholder, and control-character
fragments.

Implemented location in code: `rankcloak/token_filters.py`; used by
`rankcloak/segmented_protocol.py`.

Corresponding experiment profile name: `segmented-quality-controls`.

Corresponding result directory: `results/rankcloak_segmented_quality_controls/`.

Key input variables: decoded token piece, filter name, allowed-token mask.

Key output files: `segmented_quality_trials.csv`, `cover_text_features.csv`,
`SEGMENTED_QUALITY_COMPARISON.md`.

Intended paper role: test whether a deterministic rank-space restriction reduces obvious
cover artifacts.

Limitations: filtering changes the effective rank distribution and may reduce capacity;
it is not a detector and not a safety classifier.

## Forced-Prefix Versus Full-Message Metrics

Short name: metric separation.

Purpose: compute features separately for payload-bearing forced tokens and the full
public message including tails.

Implemented location in code: `rankcloak/segmented_protocol.py`, `rankcloak/schemas.py`,
`rankcloak/metrics.py`.

Corresponding experiment profile name: `segmented-quality-controls`.

Corresponding result directory: `results/rankcloak_segmented_quality_controls/`.

Key input variables: forced token ids, tail token ids, full token ids, ranks, log
probabilities.

Key output files: `segmented_quality_messages.jsonl`, `segmented_quality_trials.csv`,
`cover_text_features.csv`.

Intended paper role: prevent tail-dominated metrics from being mistaken for
payload-bearing-token quality.

Limitations: metrics remain lightweight and do not replace human evaluation or
steganalysis.

## Baseline Cover Generation

Short name: greedy baseline.

Purpose: generate ordinary model text from the same prompts without payload ranks, using
rank-1 greedy continuation.

Implemented location in code: `rankcloak/baselines.py`, `rankcloak/experiments.py`.

Corresponding experiment profile names: standard non-segmented model-backed profiles.

Corresponding result directories: directories containing
`baseline_cover_examples.jsonl`.

Key input variables: prompt name, target token count.

Key output files: `baseline_cover_examples.jsonl`, `cover_text_features.csv`.

Intended paper role: provide a reference distribution for generated cover features.

Limitations: current baseline is greedy only; no detector AUC is implemented.

## Plausibility And Artifact Metrics

Short name: lightweight cover features.

Purpose: compute character, token, repetition, punctuation, digit, alphabetic, rank,
log-probability, and artifact-flag features.

Implemented location in code: `rankcloak/metrics.py`.

Corresponding experiment profile names: all profiles with generated cover text.

Corresponding result directories: directories containing `cover_text_features.csv`.

Key input variables: generated text, token ids, ranks, token log probabilities.

Key output files: `cover_text_features.csv`, comparison markdown files, figures.

Intended paper role: provide measurable quality proxies for pilot comparisons.

Limitations: these are feature measurements, not a trained detector and not a human
readability score.

## Reproducibility Manifest System

Short name: reproducibility manifest.

Purpose: record environment, package versions, git status, model path, model SHA-256,
profile, output directory, and command-line arguments.

Implemented location in code: `rankcloak/reproducibility.py`.

Corresponding experiment profile names: all current profiles.

Corresponding result directories: every current result directory contains
`MANIFEST.json`.

Key input variables: project root, model file, output directory, command-line args.

Key output files: `MANIFEST.json`.

Intended paper role: support reproducible local CPU execution and artifact provenance.

Limitations: manifests describe local runs; they do not guarantee bit-identical behavior
across all hardware and library versions.

## Current Limitations

- Current results are pilots, not final paper-main experiments.
- Exact-copy conditions are required.
- The current main model is one local Llama 3 8B Instruct GGUF quantization.
- Prompt comparisons are limited to selected original prompts.
- Detector AUC is not yet implemented.
- Human or LLM plausibility studies are not yet implemented.
- Safe-text filtering is deterministic and heuristic.
- Tails can improve full-message quality while leaving forced-prefix quality low.

## Claims Currently Supported By Methodology

- Rank ordering is deterministic and 1-indexed with token-id tie-breaking.
- Bounded-rank codecs roundtrip exactly in current tests.
- Current pilot runs recover exactly under exact-copy conditions.
- Direct subword payload representations can have high rank pressure for high-entropy artifacts.
- Lower bounded-rank alphabets reduce rank pressure at the cost of more ranks.
- Hex-nibble coding is efficient for hex-like artifacts.
- Prompt, tail, segmentation, and filter choices affect lightweight cover-quality metrics.

## Claims Not Supported And Should Not Be Made

- Do not claim cryptographic security.
- Do not claim encryption, key exchange, authentication, signing, or credential protection.
- Do not claim undetectability.
- Do not claim robustness to edits, paraphrase, re-tokenization, or copy-channel normalization.
- Do not claim cross-model portability.
- Do not claim real-secret handling.
- Do not claim detector performance until an actual detector is implemented and evaluated.
