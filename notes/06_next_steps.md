# Next Steps

## Immediate Research Priorities

1. Add manual quality scoring.

The current feature metrics are useful but incomplete. Mean token log probability, repetition, punctuation, and token count do not fully capture whether a text looks natural. Add a small blinded manual scoring rubric for coherence, topicality, formatting artifacts, and obvious forced-token damage.

2. Keep near-term generation at B=8 and B=16.

The current sweeps show that B=32 and B=64 increase rank pressure and visibly damage cover text. Larger alphabets can remain in measurement tables, but prompt-quality experiments should focus on B=8 and B=16 until distribution matching is implemented.

3. Separate payload representation from cover generation.

The payload granularity pilot shows that raw hex nibbles reduce rank count for hex payloads while keeping ranks bounded. This should become the default representation for hex-like artifacts in future cover-quality experiments.

4. Build a small human-readable comparison set.

For each promising prompt and B value, save a small fixed panel:

- original payload name only, not the payload value;
- prompt name;
- alphabet size;
- generated text excerpt;
- baseline excerpt of similar token length;
- exact recovery status;
- feature metrics.

This can become the basis for a human or LLM plausibility study.

## Engineering Priorities

1. Add a profile that uses `raw_hex_nibbles` for full stegotext generation.

Current full stegotext profiles use ASCII bytes through fixed-radix ranks. The payload granularity pilot suggests hex-nibble coding is a better payload-side representation for hex artifacts.

2. Add detector baselines only after the dataset is large enough.

Do not claim AUC yet. Once there are enough RankCloak and baseline examples, add a simple reproducible detector experiment with train/test splits and clear caveats.

3. Add runtime controls.

Helpful additions:

- max trials;
- prompt subset;
- payload subset;
- alphabet subset;
- resume/skip existing rows;
- CSV append mode with manifest update.

4. Stabilize notebook output.

The notebook should stay explanatory and load generated results rather than running long experiments inline.

## Future Experimental Questions

- Does raw hex-nibble cover generation improve quality by reducing length relative to ASCII fixed-radix B=16?
- Which cover genres tolerate repeated low-rank forcing best?
- Can distribution-matched rank coding reduce obvious artifacts while preserving exact recovery?
- How does quality change across Phi, Llama, Mistral, Qwen, and Gemma GGUF models?
- How sensitive is recovery to whitespace normalization, smart quotes, copied Markdown, and platform text transformations?
- Can a detector distinguish RankCloak cover text from greedy or sampled baseline cover text with reliable AUC?

## Current Working Hypotheses

- H1: Direct subword encoding is compact but creates high rank pressure for high-entropy artifacts.
- H2: Bounded-rank encodings trade more cover tokens for lower rank pressure and better plausibility.
- H3: B=8 and B=16 are the most useful current bounded alphabets for readable cover text.
- H4: Prompt strength improves topic anchoring but cannot by itself fix forced-rank artifacts.
- H5: Dialogue prompts can improve local log probability while increasing repetition and formatting artifacts.
- H6: Payload-side hex-nibble coding is likely better than ASCII byte coding for hex-like artifacts.

## Important Limits To Preserve In The Paper

- The method is not encryption.
- The method is not key exchange.
- The method is not authentication.
- The method does not protect real secrets.
- The current experiments rely on exact-copy text preservation.
- All examples are deterministic and synthetic.
- Generated text quality remains mixed and should not be overstated.
