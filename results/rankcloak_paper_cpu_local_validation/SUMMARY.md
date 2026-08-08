# RankCloak Staged Paper Suite Summary

- Profile: paper-diagnostics
- Stage: paper-diagnostics
- Model status: loaded
- Non-segmented trials: 0
- Non-segmented planned trials: 96
- Non-segmented remaining trials: 96
- Segmented trials: 0
- Segmented planned trials: 24
- Segmented remaining trials: 24
- Baseline rows: 0
- Recovery: 0 pass, 0 fail
- Detector dataset rows: 0
- Statistical rows: 0
- Effect-size rows: 0

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental. In the current partial pilot it produced one exact-recovery failure, so it should be reported separately from the non-lead-in segmented variants.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_cpu_local_validation --resume --limit-trials 10`
