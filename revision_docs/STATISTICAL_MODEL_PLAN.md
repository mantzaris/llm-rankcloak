# Prespecified confirmatory mixed-model plan

The machine-readable specification is
`analysis/revision_v1/confirmatory_model_plan.json`. It is frozen before the
confirmatory matrix is inspected. `analysis/revision_v1/mixed_effects_specs.json`
contains the narrower adapter-compatible recovery and continuous specifications
for `rankcloak.revision_statistics.run_mixed_effects_specs`; the dedicated R
driver implements the complete diagnostic and contrast contract.

## Analysis units and admissible inputs

The experimental unit is one payload trial under one frozen condition.
Segments are nested observations and are never treated as independent. The R
driver accepts only flat preprocessor outputs:

- primary `trials.csv` rows with evidence
  `confirmatory_primary_v2_payload_fidelity_after_manifest_freeze`, record type
  `rankcloak_trial`, and
  replay `saved_token_ids`;
- the v1 hash-checked held-out join's full-message feature CSV and manifest,
  collapsed once per trial using a token-weighted log-probability mean and
  summed artifact counts;
- `runtime.csv` trial-scope rows joined one-to-one to trial metadata;
- optional detector summaries for input auditing only (detector inference uses
  the grouped-bootstrap detector pipeline).

`condition_unavailable` and `dependent_unavailable` records never appear in
these estimand tables. They are counted separately by preprocessing and
reporting. Payload name and prompt-template ID are random-intercept grouping
variables; a segment cannot create another payload observation.

## Models

Primary recovery uses a binomial-logit `lme4::glmer` with model, protocol,
prompt category, their prespecified pairwise interactions, payload class, and
random intercepts for payload and prompt template. If every outcome is a
success (or every outcome is a failure), the GLMM is unidentified and is not
fit. The driver reports grouped Wilson intervals, an explicit complete-outcome
separation status, and no coefficient or fixed-effects substitute.

Artifact counts use `lme4::glmer.nb` with the same fixed/random structure and
an offset for log cover-token count. A Poisson GLMM dispersion ratio is a
prespecified diagnostic only; it does not select or replace the negative-
binomial primary model. All-zero counts produce an explicit not-fit status.

Gaussian `lme4::lmer` models cover effective artifact bits per full token,
token-weighted cover log probability, held-out-evaluator log probability, and
log payload throughput. Confirmatory execution requires the verified join, so
the held-out evaluator model must run; a missing column is a failed downstream
gate rather than permission to omit it. Human naturalness and suspiciousness remain external
until authorized ratings exist; the frozen future engine is `ordinal::clmm`
with participant, stimulus, and prompt-template random intercepts.

Every fitted model records optimizer messages, warnings, singularity at the
frozen tolerance, maximum gradient, Hessian eigenvalue, rank-deficient columns,
large-coefficient separation flags, exclusions for incomplete rows, and an
explicit `fixed_effects_fallback=false` value. Failure or nonidentifiability is
never converted to a simpler fixed-effects model.

## Contrasts and multiplicity

Prespecified `emmeans` contrasts compare protocol within model, model within
protocol, and prompt category for recovery; analogous protocol contrasts are
defined for artifact and continuous outcomes. Holm adjustment is applied
within each declared family (and separately by continuous outcome). Unplanned
post-hoc contrasts must be labeled exploratory rather than added to these
families.

## Locked R environment

`analysis/revision_v1/r_environment.lock.json` pins R 4.4.2 and exact package
versions. The no-install launcher resolves libraries in this declared order:

1. repository `.r_libs/revision_v1`: lme4 2.0.6 and ordinal 2026.7.26
   (DESCRIPTION version 2026.7-26);
2. existing R 4.4 user library: emmeans 1.10.5 and jsonlite 1.8.9;
3. R default libraries for dependencies.

Each package must resolve from its declared role. The launcher performs no
network request or installation, and the run manifest records every resolved
package path and version. The composite policy is necessary because base R
alone cannot see the project lme4/ordinal builds.

Validate a preprocessed trial table without fitting models:

```bash
analysis/revision_v1/run_with_locked_r.sh \
  scripts/run_revision_mixed_models.R \
  --plan analysis/revision_v1/confirmatory_model_plan.json \
  --environment-lock analysis/revision_v1/r_environment.lock.json \
  --trials results/revision_v1/analysis_inputs/primary/trials.csv \
  --features results/revision_v1/analysis_inputs/primary_v2_heldout_join_v1/primary_features_with_heldout_evaluator.csv \
  --feature-join-manifest results/revision_v1/analysis_inputs/primary_v2_heldout_join_v1/heldout_feature_join_manifest.json \
  --runtime results/revision_v1/analysis_inputs/primary/runtime.csv \
  --output-dir results/revision_v1/statistics/mixed_validation \
  --validate-only
```

Remove `--validate-only` and choose a new empty output directory for the
confirmatory fit. Non-validation execution refuses to proceed without the join
manifest and verifies that its declared CSV hash, 6,480 trial IDs, evidence
scope, nested-row evaluator identities, frozen models-config hash, and all
three evaluator artifact pins match `--features`. Outputs are
staged and committed by atomic directory rename:
coefficients, contrasts, diagnostics, Wilson sensitivity, Poisson dispersion,
model status, and a content-addressed run manifest. Existing output directories
are never overwritten.

Synthetic fixtures under `analysis/revision_v1/fixtures/` test complete-outcome
separation and all-zero-count handling only. They are not scientific evidence.
