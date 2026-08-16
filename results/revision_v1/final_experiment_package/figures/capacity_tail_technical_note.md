# Capacity and tail-overhead technical note

- Authoritative source: `results/revision_v1/final_experiment_package/theory/theory_empirical_validation.csv`.
- Capacity relationship: the theoretical forced-position requirement is the saved `theoretical_n_B` value derived from payload bits and the declared rank alphabet; it is compared directly with `observed_n_forced`.
- All nine aggregate capacity cells lie on the identity relation; maximum absolute forced-position residual is zero.
- Small screen-space offsets separate coincident aggregate markers only. Source values and axis coordinates remain recorded without jitter in the plotted-source CSV.
- Tail overhead is `tail_overhead_tokens`: cover-extension tokens beyond forced payload positions.
- Exact-zero tail cells are displayed on a separate linear panel. Positive medians and 5th-95th percentile ranges use a logarithmic axis.
- Points are medians; bars are 5th-95th percentile ranges, not confidence intervals.
