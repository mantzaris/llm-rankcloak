# RankCloak Staged Paper Suite Summary

- Profile: paper-main
- Stage: paper-statistics
- Model status: loaded
- Non-segmented trials: 475
- Non-segmented planned trials: 475
- Non-segmented remaining trials: 0
- Segmented trials: 75
- Segmented planned trials: 75
- Segmented remaining trials: 0
- Baseline rows: 25
- Recovery: 550 pass, 0 fail
- Detector dataset rows: 2445
- Statistical rows: 262
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental and should be reported separately.
No segmented exact-recovery failures were observed.

## Next Recommended Command

`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_experiment.py --profile paper-main --output-dir results/rankcloak_paper_gpu_main_rank_safe --model-path models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --n-gpu-layers -1 --resume`
