# External Source Note: LlmStenoExplore

Per the local instruction for this build, `https://github.com/mantzaris/LlmStenoExplore` was inspected through GitHub rather than cloned into this repository.

Patterns adapted conceptually:

- token-id-level rank tracing rather than relying on decoded text for correctness
- deterministic rank order by decreasing logit with token-id tie-breaks
- `llama-cpp-python` model loading with `logits_all=True`
- smoke profiles and manifest-style result outputs
- careful caveats that finite experiments do not prove steganographic security

No files from that repository are vendored here.

