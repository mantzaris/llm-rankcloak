# Simulation-based power assumptions and sensitivity

## Purpose and status

This is a design calculation using synthetic ordinal ratings. It is not a result from
participants, not an IRB-approved sample-size decision, and not the final inferential
model. Re-run it after the final eight-condition contrast set and plausible variance
components are frozen.

## Baseline assumptions

- Seven ordered response categories arise by thresholding a cumulative-logit latent
  variable at `[-2.1, -1.3, -0.6, 0.1, 0.8, 1.6]`.
- Every message receives three independent ratings.
- Seventy-two messages per condition and 72 panel slots match the planned balanced
  incomplete-block schedule.
- Participant random-intercept SD is 0.65 and stimulus random-intercept SD is 0.50 on
  the logit scale.
- The planning effect is a common odds ratio of 1.50. This is an assumption to test,
  not an anticipated result or smallest effect asserted by evidence.
- The two-sided planning alpha is 0.025, conservatively reflecting two co-primary
  outcomes before the final Holm family is applied.
- Power is the proportion of simulations rejected by a stimulus-cluster comparison of
  condition means. Stimulus aggregation respects the message as an experimental unit;
  the simple Welch normal approximation is intentionally more portable and generally
  less efficient than the final crossed cumulative-link mixed model.

## Sensitivity grid

`config/power_scenarios.json` includes:

- a null common odds ratio of 1.0 to audit false-positive calibration;
- a smaller odds ratio of 1.25;
- the planning odds ratio of 1.50;
- higher participant/stimulus heterogeneity;
- 48 rather than 72 messages per condition; and
- 96 rather than 72 messages per condition.

`config/power_design_grid.json` extends this planning-only sensitivity analysis to
72--240 messages per condition and three or five ratings per message. Individual
scenarios may override the config-level rating and participant-slot counts; the
simulator validates both and supports at most five ratings because the balanced offset
table is frozen at five. The expanded grid is designed to locate rough planning
landmarks, not to choose a sample size automatically.

The full run uses 2,000 simulations per scenario. Monte Carlo standard error is
reported; increase to at least 10,000 for a final near-threshold decision. A design is
not declared adequately powered merely because one favorable scenario exceeds 80%.
The authors must justify the minimally relevant odds ratio, examine high-heterogeneity
and attrition cases, and use simulation-based precision for the primary contrasts.

## Limits

The planning test does not fit the final mixed model, estimate threshold uncertainty,
simulate rater-specific scale use beyond a random intercept, model missingness, or
represent every eight-condition contrast. It also assumes independent panel slots and
no carryover. The final analysis uses `ordinal::clmm` with crossed participant and
stimulus effects. If convergence or empirical variance differs materially, update the
simulation before recruitment rather than retrofitting the sample after outcomes.
