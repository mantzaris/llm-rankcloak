# RankCloak Staged Paper Suite Summary

- Profile: paper-statistics
- Stage: paper-statistics
- Model status: not_requested
- Non-segmented trials: 20
- Segmented trials: 7
- Baseline rows: 15
- Recovery: 26 pass, 1 fail
- Detector dataset rows: 240
- Statistical rows: 97
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

## Partial Pilot Status

This staged package is partial. Completed generation rows are:

- Non-segmented: 20 completed out of 96 planned, with 76 remaining.
- Segmented: 7 completed out of 24 planned, with 17 remaining.
- Completed recovery rows: 26 pass, 1 fail.
- The observed failure is in `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`.

The detector and statistics files are valid for the current partial rows, but they
should not be treated as final paper-main evidence until the remaining staged
generation batches are completed or the subset is explicitly declared.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-nonseg-generation --output-dir results/rankcloak_paper_main_pilot --resume --limit-trials 10`
