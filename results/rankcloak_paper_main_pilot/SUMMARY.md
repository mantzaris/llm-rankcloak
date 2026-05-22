# RankCloak Staged Paper Suite Summary

- Profile: paper-statistics
- Stage: paper-statistics
- Model status: not_requested
- Non-segmented trials: 10
- Segmented trials: 2
- Baseline rows: 14
- Recovery: 12 pass, 0 fail
- Detector dataset rows: 124
- Statistical rows: 66
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

## Partial Pilot Status

This staged package is partial. Completed generation rows are:

- Non-segmented: 10 completed out of 96 planned, with 86 remaining.
- Segmented: 2 completed out of 24 planned, with 22 remaining.
- Completed recovery rows: 12 pass, 0 fail.

The detector and statistics files are valid for the current partial rows, but they
should not be treated as final paper-main evidence until the remaining staged
generation batches are completed or the subset is explicitly declared.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-nonseg-generation --output-dir results/rankcloak_paper_main_pilot --resume --limit-trials 10`
