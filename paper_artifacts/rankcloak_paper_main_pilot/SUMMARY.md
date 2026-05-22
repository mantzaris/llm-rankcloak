# RankCloak Staged Paper Suite Summary

- Profile: paper-statistics
- Stage: paper-statistics
- Model status: not_requested
- Non-segmented trials: 20
- Non-segmented planned trials: 96
- Non-segmented remaining trials: 76
- Segmented trials: 7
- Segmented planned trials: 24
- Segmented remaining trials: 17
- Baseline rows: 22
- Recovery: 26 pass, 1 fail
- Detector dataset rows: 272
- Statistical rows: 97
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental. In the current partial pilot it produced one exact-recovery failure, so it should be reported separately from the non-lead-in segmented variants.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_main_pilot --resume --limit-trials 10`
