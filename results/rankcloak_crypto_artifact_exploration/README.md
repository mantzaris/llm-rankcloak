# RankCloak Crypto Artifact Exploration Results

This directory contains smoke-test and notebook outputs for `notebooks/01_rankcloak_crypto_artifact_exploration.ipynb`.

Expected files:

- `tokenization_audit.csv`
- `rank_statistics.csv`
- `recovery_trials.csv`
- `cover_examples.jsonl`
- `summary.json`
- `SUMMARY.md`
- `figures/token_count_by_payload.png`
- `figures/rank_summary_direct_subword.png`
- `figures/cover_length_vs_rank_alphabet.png`

If the GGUF model or `llama-cpp-python` is unavailable, model-dependent CSV columns and cover examples are marked as skipped, while deterministic bounded-rank codec results are still written.

