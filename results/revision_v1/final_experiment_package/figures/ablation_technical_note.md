# Ablation-summary technical note

- Authoritative 60-row source: `results/revision_v1/final_experiment_package/statistics/ablation_analysis/ablation_canonical_contrasts.csv`.
- Evidence classification: exploratory post-outcome evidence extraction; `primary_inference=false`. Nine prespecified compact rows are plotted from the full 60-row table.
- The nine plotted rows are the compact contrasts already identified in the evidence records: 32-token lead-in, 32-rank segments, no filter, no tail, and fixed sentence tail for the stated outcomes.
- Every plotted estimand is a raw level-minus-canonical difference with a zero reference line. Hedges g is retained upstream but is not mixed onto the plotted scales; no ratio estimand is plotted.
- Bars are 95% payload-group bootstrap intervals (2,000 resamples). Exact Holm-adjusted p-values are retained below and in the plotted-source CSV; significance stars are not used.
- Null token-filter results are retained. The round-trip-stable filter cell was unavailable for 48 Mistral work units and is not treated as a recovery failure.

## Holm-adjusted p-values for plotted rows

| Outcome | Contrast | Holm-adjusted p-value |
|---|---|---:|
| Token log probability | 32-token lead-in | `1.0322418103744131e-32` |
| Token log probability | No token filter | `0.92356675669369737` |
| Payload rate | 32-rank segments | `3.7229362318000864e-19` |
| Payload rate | No token filter | `0.34883921949116231` |
| Payload rate | No tail | `2.509239517926144e-51` |
| Message length | 32-token lead-in | `2.4364512827558919e-11` |
| Message length | 32-rank segments | `2.4494272109777146e-12` |
| Message length | No tail | `1.3447765001040663e-16` |
| Message length | Fixed sentence tail | `6.5709431904093982e-05` |
