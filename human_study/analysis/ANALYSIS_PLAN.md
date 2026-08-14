# Human-rating statistical analysis plan — DRAFT

## Freeze and blinding

The plan, contrast matrix, exclusions, and model code must be versioned before
unblinding. Cleaning uses participant/stimulus blind IDs only. Condition labels are
joined after eligibility, duplicate, completion, range, schedule, and attention-check
rules are frozen and hashed.

## Outcomes

- Co-primary: overall naturalness and suspiciousness.
- Secondary: grammaticality, fluency, coherence, topic adherence, completeness.
- Attention checks and response time are quality-control fields, never outcomes.

Every rating is ordinal. Means may be shown descriptively with distributions, but the
primary effect model does not assume equal distance between 1--7 categories.

## Primary model

Fit a separate cumulative-logit mixed model for each outcome:

`ordered_rating ~ condition + prompt_category + (1 | participant_slot_id) + (1 | stimulus_blind_id)`

`ordinary_llm_control` is the reference condition. Participant and stimulus are
crossed random intercepts. Stimuli—not payload segments—are the message-level units;
no segment is treated as an independent human observation. Report log common-odds
effects, odds ratios, 95% confidence intervals, and adjusted p-values.

The co-primary condition-coefficient family is Holm adjusted across naturalness and
suspiciousness. Secondary condition coefficients are Holm adjusted as a separate
family. Emphasis is on prespecified contrasts and interval estimates, not a global
claim that every condition differs.

## Prespecified contrasts

1. Each RankCloak condition versus ordinary LLM control.
2. Direct subword Calgacus versus bounded RankCloak conditions.
3. Segmented forced span versus its full-message presentation.
4. RankCloak conditions versus licensed human-written control, interpreted as a
   naturalness benchmark rather than an equivalence claim.

The exact estimable contrast matrix must be added to the final script after final
condition names are frozen. Absence of significance is not evidence of equivalence;
any equivalence/noninferiority margin requires separate justification and power.

## Diagnostics and sensitivity

- Verify all seven response categories, three raters per stimulus, schedule balance,
  and crossed identifiers.
- Record optimizer convergence, gradient/Hessian warnings, boundary variances, and
  threshold order.
- If a CLMM fails, try prespecified optimizer controls and report failure; do not
  silently substitute a simpler model.
- Sensitivity: retain participants with one failed attention check (primary rule), then
  compare excluding them; include/exclude flagged extreme-speed rows without using
  condition labels; add payload random intercept when identifiable.
- Assess proportional-odds plausibility with a prespecified ordinal sensitivity model
  and category plots. Report material departures.
- Use no single imputation. Missing item ratings are absent rows; incomplete
  participants follow the frozen completion rule.

## Reporting

Provide condition sample sizes, complete rating distributions, effect sizes and 95%
CIs, Holm-adjusted values, rater/stimulus variance estimates, convergence status,
exclusion flow, and sensitivity results. Inter-rater agreement is descriptive and
reported with uncertainty; it is not used to delete individual ratings.
