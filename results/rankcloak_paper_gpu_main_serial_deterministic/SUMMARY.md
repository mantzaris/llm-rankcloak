# RankCloak Staged Paper Suite Summary

- Profile: paper-main
- Stage: paper-diagnostics
- Model status: loaded
- Non-segmented trials: 0
- Non-segmented planned trials: 475
- Non-segmented remaining trials: 475
- Segmented trials: 0
- Segmented planned trials: 75
- Segmented remaining trials: 75
- Baseline rows: 0
- Recovery: 0 pass, 0 fail
- Detector dataset rows: 0
- Statistical rows: 0
- Effect-size rows: 0

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental and should be reported separately.
No segmented exact-recovery failures were observed.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_gpu_main_serial_deterministic --resume --limit-trials 10`
