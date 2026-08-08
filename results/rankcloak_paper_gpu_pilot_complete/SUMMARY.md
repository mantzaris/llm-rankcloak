# RankCloak Staged Paper Suite Summary

- Profile: paper-main-pilot-resume
- Stage: paper-statistics
- Model status: loaded
- Non-segmented trials: 96
- Non-segmented planned trials: 96
- Non-segmented remaining trials: 0
- Segmented trials: 24
- Segmented planned trials: 24
- Segmented remaining trials: 0
- Baseline rows: 20
- Recovery: 111 pass, 9 fail
- Detector dataset rows: 686
- Statistical rows: 223
- Effect-size rows: 14

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.

The lead-in segmented variant is experimental and should be reported separately.
3 Segmented exact-recovery failures observed in: `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`, `segmented_hex_single_topic_sentence_tail_filtered`.

## Next Recommended Command

`python3 scripts/run_experiment.py --profile paper-main-pilot-resume --output-dir results/rankcloak_paper_gpu_pilot_complete --resume --limit-trials 10`
