# RankCloak Crypto Artifact Exploration Summary

- Run profile: smoke
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Codec roundtrip result: 63 pass, 0 fail
- Stegotext recovery result: 4 pass, 0 fail
- Baseline generation result: 2 rows
- Feature extraction result: 6 rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

- `results/rankcloak_crypto_artifact_exploration/tokenization_audit.csv`
- `results/rankcloak_crypto_artifact_exploration/rank_statistics.csv`
- `results/rankcloak_crypto_artifact_exploration/codec_roundtrip_trials.csv`
- `results/rankcloak_crypto_artifact_exploration/stegotext_recovery_trials.csv`
- `results/rankcloak_crypto_artifact_exploration/cover_examples.jsonl`
- `results/rankcloak_crypto_artifact_exploration/baseline_cover_examples.jsonl`
- `results/rankcloak_crypto_artifact_exploration/cover_text_features.csv`
- `results/rankcloak_crypto_artifact_exploration/MANIFEST.json`
- `results/rankcloak_crypto_artifact_exploration/figures/token_count_by_payload.png`
- `results/rankcloak_crypto_artifact_exploration/figures/rank_summary_direct_subword.png`
- `results/rankcloak_crypto_artifact_exploration/figures/cover_length_vs_rank_alphabet.png`
- `results/rankcloak_crypto_artifact_exploration/figures/recovery_by_cover_prompt_and_alphabet.png`
- `results/rankcloak_crypto_artifact_exploration/figures/cover_text_feature_comparison.png`
- `results/rankcloak_crypto_artifact_exploration/summary.json`
- `results/rankcloak_crypto_artifact_exploration/SUMMARY.md`
