# Stage 1 entropy diagnostics

Exploratory analysis of retained trajectories. No historical generation was repeated.

All 360 RankCloak records reconcile with the 720-row table, which also contains 360 ordinary controls. Each gate retains 120 attempts. Ungated and moderate complete 120 each. Strict completes 114 and retains all six failures.

21 trajectories are selected. Twelve are the exact ungated and moderate matches to the six failures. Three strict successes are selected by the frozen hash rule within failed model, representation, and prompt strata. They are descriptive comparisons with different payloads.

| Trial suffix | Requested | Consumed | Positions | Longest below-threshold run | Eligible fraction |
| --- | --- | --- | --- | --- | --- |
| 2b5e682d0007193b56249891 | 128 | 110 | 768 | 50 | 0.143229 |
| 941eed91360afbdc8a26bf09 | 128 | 116 | 768 | 163 | 0.151042 |
| 9ae150681b7e7347c8241d5a | 64 | 54 | 384 | 39 | 0.140625 |
| a9ff67133eeaa189fa8be8b1 | 128 | 106 | 768 | 35 | 0.138021 |
| c4160dc01512e365fbecb170 | 128 | 59 | 768 | 267 | 0.076823 |
| cbb747058efe6a8b1fbd18ec | 128 | 116 | 768 | 51 | 0.151042 |

Every failed trajectory exhausts its 6L budget with fewer than L eligible positions. Thus its realized eligible fraction is below 1/6. This is exact capacity accounting. It does not establish that a linguistic pattern caused the low entropy.

Position tables retain entropy, inclusive eligibility, token role, observed rank, surprisal, rank pressure, cumulative consumption, remaining ranks, and trailing 32-position progress. Partial initial windows are explicitly labeled. The first low-progress window is the first full window with at most two consumed ranks. All maximal below-threshold runs are retained and longest-run ties choose the earliest.

The window table uses pinned vocabulary-only prefix detokenization and byte offsets when alignment is verified. It never concatenates isolated token pieces. Event bounds remain exact while display bounds can expand to avoid cutting UTF-8. Linguistic interpretations remain exploratory and must be compared with matched and successful traces.

The ordinary-development 75th percentile does not imply 25 percent eligibility along a forced generation trajectory. Different contexts and sampled skips change that trajectory. No causal claim follows from the selected examples.

Reproduce with `PYTHONPATH=. .venv-generation-v3/bin/python scripts/analyze_revision_v4_stage1.py --align-tokenizers`. The manifest binds input bytes and selection rules. Figure generation is a separate CPU-only command.
