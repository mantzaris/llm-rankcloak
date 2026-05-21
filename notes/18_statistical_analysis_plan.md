# Statistical Analysis Plan

## Purpose

The paper suite adds deterministic bootstrap summaries and simple effect-size tables so
pilot results can be reviewed with uncertainty estimates.

## Bootstrap Method

The implementation uses fixed-seed bootstrap resampling over recorded trial rows.

Default resamples:

- `paper-main-pilot`: 1000 resamples.
- `paper-main`: 2000 resamples.

If runtime becomes a concern, the resample count can be reduced and the reduction must
be reported in the summary.

## Statistical Summary Rows

The suite writes `statistical_summary.csv`.

Metrics include:

- exact recovery as 0/1;
- rank count;
- generated token count;
- mean token log probability;
- p95 generated rank;
- repeated-token fraction;
- artifact count;
- forced-prefix mean log probability;
- full-message mean log probability;
- total full-message token count.

Rows are grouped by protocol variant and, where available, by payload class.

## Effect Sizes

The suite writes `effect_size_summary.csv`.

Planned comparisons include:

- `nonseg_ascii_b8` versus `nonseg_ascii_b16`;
- `nonseg_ascii_b16` versus `nonseg_hex_nibble_b16` for hex payloads;
- segmented single-topic versus segmented multi-topic;
- forced-prefix versus full-message metrics;
- imported unfiltered versus filtered comparison from the segmented quality-controls pilot when available.

## Interpretation

Use bootstrap intervals as pilot uncertainty summaries, not as final inferential claims
until the paper-main matrix and analysis plan are frozen.

## Caveats

Some comparisons are not paired even when they share payloads. The current effect-size
implementation is intentionally simple and should be reviewed before manuscript
submission.
