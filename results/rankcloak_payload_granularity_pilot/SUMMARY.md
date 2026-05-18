# RankCloak Crypto Artifact Exploration Summary

- Run profile: payload-granularity-pilot
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Codec roundtrip result: 63 pass, 0 fail
- Stegotext recovery result: 0 pass, 0 fail
- Baseline generation result: 0 rows
- Feature extraction result: 0 rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

- `results/rankcloak_payload_granularity_pilot/tokenization_audit.csv`
- `results/rankcloak_payload_granularity_pilot/rank_statistics.csv`
- `results/rankcloak_payload_granularity_pilot/codec_roundtrip_trials.csv`
- `results/rankcloak_payload_granularity_pilot/stegotext_recovery_trials.csv`
- `results/rankcloak_payload_granularity_pilot/cover_examples.jsonl`
- `results/rankcloak_payload_granularity_pilot/baseline_cover_examples.jsonl`
- `results/rankcloak_payload_granularity_pilot/cover_text_features.csv`
- `results/rankcloak_payload_granularity_pilot/MANIFEST.json`
- `results/rankcloak_payload_granularity_pilot/figures/token_count_by_payload.png`
- `results/rankcloak_payload_granularity_pilot/figures/rank_summary_direct_subword.png`
- `results/rankcloak_payload_granularity_pilot/figures/cover_length_vs_rank_alphabet.png`
- `results/rankcloak_payload_granularity_pilot/figures/recovery_by_cover_prompt_and_alphabet.png`
- `results/rankcloak_payload_granularity_pilot/figures/cover_text_feature_comparison.png`
- `results/rankcloak_payload_granularity_pilot/payload_granularity_comparison.csv`
- `results/rankcloak_payload_granularity_pilot/figures/payload_representation_rank_count.png`
- `results/rankcloak_payload_granularity_pilot/summary.json`
- `results/rankcloak_payload_granularity_pilot/SUMMARY.md`
