# RankCloak Paper Results Summary

## Overview

Profile: `paper-main-pilot-resume`

This run evaluates deterministic synthetic payloads under exact-copy conditions. It does not claim encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or undetectability.

## Exact Recovery Summary

- Non-segmented recovery: 90 pass, 6 fail.
- Segmented recovery: 21 pass, 3 fail.

## Payload Representation Results

See `paper_codec_comparison.csv` and `paper_payloads.csv`.

## Direct Subword Rank Pressure

See `paper_rank_pressure.csv`. Direct subword rank pressure is a diagnostic baseline and does not generate cover text in this suite.

## Non-Segmented Bounded-Rank Results

See `paper_stegotext_trials.csv`. The main non-segmented variants are `nonseg_ascii_b8`, `nonseg_ascii_b16`, and `nonseg_hex_nibble_b16`.

## Hex-Nibble Results

Hex-nibble rows apply only to payloads marked `is_hex_like`.

## Prompt And Alphabet-Size Results

See `paper_cover_text_features.csv` and the figure index for prompt-family and alphabet-size comparisons.

## Segmented Protocol Results

See `paper_segmented_trials.csv` and `paper_segmented_messages.jsonl`.

## Lead-In Segmented Variant Results

The lead-in variant is implemented as `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`. The decoder ignores the greedy lead-in, decodes the forced span, and ignores the tail.

1 lead-in segmented exact-recovery failure was observed in this run; the variant remains experimental.

## Forced-Prefix Versus Full-Message Results

Segmented rows separate forced-prefix metrics from full-message metrics. Full-message quality can be tail-driven and should not be treated as payload-bearing-token quality.

## Safe-Text Filter And Artifact Results

Segmented paper variants use `safe_text_filter_v1`, a deterministic heuristic token filter. See artifact columns in `paper_cover_text_features.csv`.

## Detector Baseline Results

Detector rows: 57. These are lightweight feature baselines only and do not establish strong steganalysis.

## Statistical Uncertainty

Bootstrap summary rows: 223. Effect-size rows: 14.

## Recommended Main Paper Claims

- The suite measures exact-copy recovery and cover-quality proxies for deterministic synthetic artifacts.
- Bounded-rank and hex-nibble encodings provide controlled rank pressure compared with direct subword ranks.
- Segmented variants require separate forced-prefix and full-message metrics.

## Claims Not Supported

- No cryptographic security claim.
- No undetectability claim.
- No edit robustness claim.
- No cross-model portability claim.

## Limitations

The current run is `paper-main-pilot-resume`. This is a pilot-scale validation run before the larger frozen `paper-main` matrix.
