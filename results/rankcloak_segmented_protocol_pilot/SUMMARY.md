# RankCloak Segmented Protocol Pilot Summary

- Profile: segmented-protocol-pilot
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Control request exact recovery: True
- Response recovery: 10 pass, 0 fail
- Response trial count: 10

## Main Quality Observation

Segmentation restarts each cover message from a clean prompt context and allows optional greedy tails, so this pilot compares cover drift against one longer forced span. The first-pass quality result should be read from `SEGMENTED_PROTOCOL_COMPARISON.md` and `cover_text_features.csv`; exact recovery alone is not a cover-quality score.

## Important Limitations

- This is not encryption, key exchange, authentication, signing, or cryptographic security.
- The control code is synthetic and maps to a pre-agreed local experiment configuration.
- The decoder recovers only the forced prefix and ignores natural tails.
- All examples are deterministic synthetic payloads.
- Exact-copy conditions are required.

## Next Recommended Experiment

If this pilot suggests lower drift, run a follow-up using raw-hex-nibble payload coding for more hex-like payloads and a small manual quality rubric.
