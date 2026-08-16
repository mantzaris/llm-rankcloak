# Capacity-quality validation technical note

Computational evidence artifact only; this is not manuscript text.

## Defined equations

- `n_B = ceil(H / log2(B))` (token_positions; identifier `minimum_forced_positions`).
- `R_B = H / n_B` (bits_per_forced_token; identifier `nominal_rate`).
- `R_effective = H / (n_forced + n_tail)` (bits_per_generated_token; identifier `effective_rate_with_tail`).
- `Q_B = -mean_t(log p(y_t | context_t))` (nats_per_forced_token; identifier `realized_surprisal`).
- `Delta_B = Q_B - Q_greedy` (nats_per_forced_token; identifier `quality_penalty`).
- `Q_greedy <= Q_B <= Q_rank_B` (nats_per_forced_token; identifier `same_context_rank_bounds`).

## Validation assumptions

- H is the saved representation-source bit count for the evaluated codec.
- B is the saved admissible rank bound and ranks are one-indexed.
- Tail tokens carry no additional payload bits in the stated effective-rate calculation.
- Quality endpoint inequalities require probabilities evaluated under identical saved contexts.
- Missing endpoint arrays are unavailable and are not reconstructed from aggregate means.
- Confidence intervals resample payload identities, not individual tokens.

## Empirical validation contract

- `theory_empirical_validation.csv` retains trial and condition identities and records capacity residuals, tail overhead, rates, same-context surprisal bounds, and missing endpoints.
- `theory_empirical_summary.csv` reports payload-clustered percentile-bootstrap intervals; token positions are not treated as independent observations.
- `theory_residual_plot_source.csv` is the figure source. Raw record files are referenced by hash and are not copied into this package.

## Known limits

- Observed rank replay does not prove the deterministic replay proposition when complete ranked token orders were not saved.
- The empirical summary describes the frozen saved corpus and does not identify causal quality effects.
