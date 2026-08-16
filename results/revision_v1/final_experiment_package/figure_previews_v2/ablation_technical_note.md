# Ablation-summary technical note

- Authoritative 60-row source: `results/revision_v1/final_experiment_package/statistics/ablation_analysis/ablation_canonical_contrasts.csv`.
- Figure status: exploratory post-outcome evidence extraction; `primary_inference=false`.
- The nine plotted rows are the compact contrasts already identified in the evidence records: 32-token lead-in, 32-rank segments, no filter, no tail, and fixed sentence tail for the stated outcomes.
- Every plotted estimand is a raw level-minus-canonical difference with a zero reference line. Hedges g is retained upstream but is not mixed onto the plotted scales; no ratio estimand is plotted.
- Bars are 95% payload-group bootstrap intervals (2,000 resamples); Holm-adjusted p-values are shown without significance stars.
- Null token-filter results are retained. The round-trip-stable filter cell was unavailable for 48 Mistral work units and is not treated as a recovery failure.
