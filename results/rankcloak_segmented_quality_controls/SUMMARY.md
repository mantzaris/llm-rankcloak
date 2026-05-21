# RankCloak Segmented Quality Controls Summary

- Profile: segmented-quality-controls
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Control recovery: 5 pass, 0 fail
- Response recovery: 10 pass, 0 fail
- Response trial count: 10

## Main Quality Observation

This profile separates payload-bearing forced-prefix metrics from full-message metrics. Sentence tails and safe-token filtering should be interpreted through both views: a full message may look natural because of its tail even when the forced prefix remains visibly constrained.

## Important Limitations

- Synthetic payloads only.
- No encryption, key exchange, authentication, signing, or cryptographic security is claimed.
- The control code is a compact pre-agreed label.
- The decoder ignores tails and recovers only forced prefixes.
- Exact-copy conditions are required.

## Next Recommended Experiment

Inspect `SEGMENTED_QUALITY_COMPARISON.md`, then decide whether to keep `safe_text_filter_v1` and whether forced-prefix quality justifies a smaller segment size or a distribution-matched rank code.
