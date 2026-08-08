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
- Baseline rows: 20
- Recovery: 24 pass, 3 fail
- Detector dataset rows: 264
- Statistical rows: 97
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental and should be reported separately.
2 Segmented exact-recovery failures observed in: `segmented_hex_multi_topic_sentence_tail_filtered`.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_gpu_validation --resume --limit-trials 10`
