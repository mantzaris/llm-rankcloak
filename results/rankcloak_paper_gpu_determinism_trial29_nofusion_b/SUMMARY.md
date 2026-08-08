# RankCloak Staged Paper Suite Summary

- Profile: paper-main
- Stage: paper-statistics
- Model status: loaded
- Non-segmented trials: 1
- Non-segmented planned trials: 475
- Non-segmented remaining trials: 474
- Segmented trials: 0
- Segmented planned trials: 75
- Segmented remaining trials: 75
- Baseline rows: 1
- Recovery: 1 pass, 0 fail
- Detector dataset rows: 6
- Statistical rows: 14
- Effect-size rows: 2

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental and should be reported separately.
No segmented exact-recovery failures were observed.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_gpu_determinism_trial29_nofusion_b --resume --limit-trials 10`
