# RankCloak Crypto Artifact Exploration Summary

- Run profile: dialogue-key-pilot
- Model status: loaded
- Model: QuantFactory/Meta-Llama-3-8B-Instruct-GGUF / Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
- Codec roundtrip result: 63 pass, 0 fail
- Stegotext recovery result: 24 pass, 0 fail
- Baseline generation result: 6 rows
- Feature extraction result: 30 rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

- `results/rankcloak_dialogue_key_pilot/tokenization_audit.csv`
- `results/rankcloak_dialogue_key_pilot/rank_statistics.csv`
- `results/rankcloak_dialogue_key_pilot/codec_roundtrip_trials.csv`
- `results/rankcloak_dialogue_key_pilot/stegotext_recovery_trials.csv`
- `results/rankcloak_dialogue_key_pilot/cover_examples.jsonl`
- `results/rankcloak_dialogue_key_pilot/baseline_cover_examples.jsonl`
- `results/rankcloak_dialogue_key_pilot/cover_text_features.csv`
- `results/rankcloak_dialogue_key_pilot/MANIFEST.json`
- `results/rankcloak_dialogue_key_pilot/figures/token_count_by_payload.png`
- `results/rankcloak_dialogue_key_pilot/figures/rank_summary_direct_subword.png`
- `results/rankcloak_dialogue_key_pilot/figures/cover_length_vs_rank_alphabet.png`
- `results/rankcloak_dialogue_key_pilot/figures/recovery_by_cover_prompt_and_alphabet.png`
- `results/rankcloak_dialogue_key_pilot/figures/cover_text_feature_comparison.png`
- `results/rankcloak_dialogue_key_pilot/figures/dialogue_prompt_mean_logprob.png`
- `results/rankcloak_dialogue_key_pilot/figures/dialogue_prompt_repetition.png`
- `results/rankcloak_dialogue_key_pilot/figures/dialogue_prompt_length.png`
- `results/rankcloak_dialogue_key_pilot/figures/dialogue_prompt_quality_scatter.png`
- `results/rankcloak_dialogue_key_pilot/DIALOGUE_PROMPT_COMPARISON.md`
- `results/rankcloak_dialogue_key_pilot/summary.json`
- `results/rankcloak_dialogue_key_pilot/SUMMARY.md`
