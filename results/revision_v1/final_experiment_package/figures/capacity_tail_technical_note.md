# Capacity and tail-overhead technical note

- Authoritative source: `results/revision_v1/final_experiment_package/theory/theory_empirical_validation.csv`.
- Evidence classification: supporting evidence; nine aggregate cells appear in both the capacity and tail panels (18 plotted-source rows).
- Capacity relationship: the theoretical forced-position requirement is the saved `theoretical_n_B` value derived from payload bits and the declared rank alphabet; it is compared directly with `observed_n_forced`.
- All nine aggregate capacity cells lie on the identity relation; maximum absolute forced-position residual is zero.
- Every capacity marker is plotted at its true data coordinates. Coincident cells are shown with concentric marker sizes, stage-specific shapes, transparent fills, and controlled z-order; no x, y, screen-space, pixel, point, or transform displacement is applied.
- Tail overhead is `tail_overhead_tokens`: cover-extension tokens beyond forced payload positions.
- Exact-zero tail cells are displayed on a separate linear panel. Positive medians and 5th-95th percentile ranges use a logarithmic axis.
- Points are medians; bars are 5th-95th percentile ranges, not confidence intervals.
