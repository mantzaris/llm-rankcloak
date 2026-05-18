# RankCloak Crypto Artifact Exploration Summary

- Run profile: strong-prompts
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Codec roundtrip result: 63 pass, 0 fail
- Stegotext recovery result: 60 pass, 0 fail
- Baseline generation result: 5 rows
- Feature extraction result: 65 rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

- `results/rankcloak_strong_prompt_sweep/tokenization_audit.csv`
- `results/rankcloak_strong_prompt_sweep/rank_statistics.csv`
- `results/rankcloak_strong_prompt_sweep/codec_roundtrip_trials.csv`
- `results/rankcloak_strong_prompt_sweep/stegotext_recovery_trials.csv`
- `results/rankcloak_strong_prompt_sweep/cover_examples.jsonl`
- `results/rankcloak_strong_prompt_sweep/baseline_cover_examples.jsonl`
- `results/rankcloak_strong_prompt_sweep/cover_text_features.csv`
- `results/rankcloak_strong_prompt_sweep/MANIFEST.json`
- `results/rankcloak_strong_prompt_sweep/figures/token_count_by_payload.png`
- `results/rankcloak_strong_prompt_sweep/figures/rank_summary_direct_subword.png`
- `results/rankcloak_strong_prompt_sweep/figures/cover_length_vs_rank_alphabet.png`
- `results/rankcloak_strong_prompt_sweep/figures/recovery_by_cover_prompt_and_alphabet.png`
- `results/rankcloak_strong_prompt_sweep/figures/cover_text_feature_comparison.png`
- `results/rankcloak_strong_prompt_sweep/figures/strong_prompt_mean_logprob_by_prompt.png`
- `results/rankcloak_strong_prompt_sweep/figures/strong_prompt_recovery_by_prompt.png`
- `results/rankcloak_strong_prompt_sweep/figures/strong_prompt_length_by_prompt.png`
- `results/rankcloak_strong_prompt_sweep/figures/strong_prompt_rank_pressure.png`
- `results/rankcloak_strong_prompt_sweep/PROMPT_COMPARISON.md`
- `results/rankcloak_strong_prompt_sweep/summary.json`
- `results/rankcloak_strong_prompt_sweep/SUMMARY.md`
