# RankCloak Crypto Artifact Exploration Summary

- Run profile: small
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Codec roundtrip result: 63 pass, 0 fail
- Stegotext recovery result: 64 pass, 0 fail
- Baseline generation result: 4 rows
- Feature extraction result: 68 rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

- `results/rankcloak_small_full/tokenization_audit.csv`
- `results/rankcloak_small_full/rank_statistics.csv`
- `results/rankcloak_small_full/codec_roundtrip_trials.csv`
- `results/rankcloak_small_full/stegotext_recovery_trials.csv`
- `results/rankcloak_small_full/cover_examples.jsonl`
- `results/rankcloak_small_full/baseline_cover_examples.jsonl`
- `results/rankcloak_small_full/cover_text_features.csv`
- `results/rankcloak_small_full/MANIFEST.json`
- `results/rankcloak_small_full/figures/token_count_by_payload.png`
- `results/rankcloak_small_full/figures/rank_summary_direct_subword.png`
- `results/rankcloak_small_full/figures/cover_length_vs_rank_alphabet.png`
- `results/rankcloak_small_full/figures/recovery_by_cover_prompt_and_alphabet.png`
- `results/rankcloak_small_full/figures/cover_text_feature_comparison.png`
- `results/rankcloak_small_full/summary.json`
- `results/rankcloak_small_full/SUMMARY.md`
