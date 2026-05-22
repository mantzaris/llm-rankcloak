# Paper Comparison Tables

## Table A: Payload Classes And Counts

| Payload class | Count |
| --- | ---: |
| `ciphertext_like_base64` | 2 |
| `nonce_96_bit_hex` | 2 |
| `random_128_bit_hex` | 2 |
| `random_256_bit_hex` | 2 |
| `sha256_hex` | 2 |
| `uuid_v4_like` | 2 |

## Table B: Protocol Variants

| Protocol variant | Role |
| --- | --- |
| `raw_subword_rank_pressure` | implemented paper-suite variant |
| `nonseg_ascii_b8` | implemented paper-suite variant |
| `nonseg_ascii_b16` | implemented paper-suite variant |
| `nonseg_hex_nibble_b16` | implemented paper-suite variant |
| `segmented_hex_single_topic_sentence_tail_filtered` | implemented paper-suite variant |
| `segmented_hex_multi_topic_sentence_tail_filtered` | implemented paper-suite variant |
| `segmented_hex_multi_topic_leadin8_sentence_tail_filtered` | implemented paper-suite variant |

## Table C: Recovery Summary By Protocol Variant

| Variant | Trials | Passes |
| --- | ---: | ---: |
| `nonseg_ascii_b16` | 6 | 6 |
| `nonseg_ascii_b8` | 8 | 8 |
| `nonseg_hex_nibble_b16` | 6 | 6 |
| `segmented_hex_multi_topic_leadin8_sentence_tail_filtered` | 1 | 0 |
| `segmented_hex_multi_topic_sentence_tail_filtered` | 3 | 3 |
| `segmented_hex_single_topic_sentence_tail_filtered` | 3 | 3 |

## Table D: Payload Representation Rank Counts

See `paper_codec_comparison.csv`.

## Table E: Cover-Quality Metrics By Protocol Variant

See `statistical_summary.csv`.

## Table F: Forced-Prefix Versus Full-Message Metrics

See `paper_segmented_trials.csv`.

## Table G: Detector Results

See `detector_baseline.csv`.

## Table H: Limitations And Unsupported Claims

- Synthetic payloads only.
- Exact-copy conditions only.
- No encryption, key exchange, authentication, signing, or cryptographic security claim.
- No undetectability claim.