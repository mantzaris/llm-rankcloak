# Submission Readiness Checklist

This checklist is for moving RankCloak from pilot repository to journal submission
package.

## Methodology Readiness

- [x] Deterministic rank ordering is implemented and tested.
- [x] Fixed-radix bounded-rank encoding is implemented and tested.
- [x] Raw hex-nibble encoding is implemented and tested.
- [x] Prompt-key cover generation is implemented.
- [x] Exact recovery is implemented for non-segmented covers.
- [x] Segmented multi-cover protocol is implemented.
- [x] Forced-prefix-only segmented decoding is implemented.
- [x] Sentence-boundary tail policy is implemented.
- [x] Deterministic safe-text token filter is implemented.
- [x] Paper-main pilot and full paper-main profiles are implemented in the runner.
- [x] Paper-analysis aggregation profile is implemented.
- [ ] Distribution-matched rank coding is not yet implemented.
- [ ] Cross-model method comparison is not yet implemented.
- [ ] Edit robustness protocol is not yet implemented.

## Results Readiness

- [x] Smoke results are present.
- [x] Small full sweep results are present.
- [x] Strong prompt sweep results are present.
- [x] Dialogue key prompt pilot results are present.
- [x] Payload granularity pilot results are present.
- [x] Segmented protocol pilot results are present.
- [x] Segmented quality-controls results are present.
- [x] Paper-main result schemas are defined.
- [x] Paper-main pilot and full output directories are defined.
- [ ] Paper-main repeated runs are not yet defined.
- [ ] Statistical confidence intervals are not yet computed.
- [ ] Human or LLM plausibility ratings are not yet collected.

## Reproducibility Readiness

- [x] Script-based runner exists.
- [x] CLI wrapper exists.
- [x] Notebook exists.
- [x] Reproducibility manifests are written.
- [x] Model path convention is documented.
- [x] Result directories contain `summary.json` and `MANIFEST.json`.
- [ ] External archived release or DOI is not yet created.
- [ ] Exact model acquisition and license instructions need final manuscript wording.
- [ ] Final paper-main command list should be frozen before submission.

## Statistical Analysis Readiness

- [x] Pilot tables record exact recovery.
- [x] Pilot tables record log probability, repetition, punctuation, and rank features.
- [x] Segmented quality controls separate forced-prefix and full-message metrics.
- [x] Deterministic bootstrap helpers are implemented for the paper-main suite.
- [x] Effect-size summary output is implemented for the paper-main suite.
- [ ] Statistical tests are not yet selected.
- [ ] Multiple-comparison plan is not yet written.
- [ ] Sample sizes for paper-main experiments are not yet justified.
- [ ] Confidence intervals or bootstrap intervals are not yet implemented.

## Detection And Steganalysis Readiness

- [x] Lightweight feature extraction exists.
- [x] Baseline cover generation exists for non-segmented profiles.
- [x] Lightweight detector dataset creation is implemented.
- [x] Dependency-free threshold detector baseline is implemented.
- [x] Optional scikit-learn detector baselines are attempted when installed.
- [x] Train/test split protocol is implemented for the lightweight baseline.
- [ ] Cross-prompt and cross-payload generalization tests are not implemented.
- [ ] Human or LLM detector comparison is not implemented.

## Data Availability

- [x] Small CSV, JSON, JSONL, Markdown, and PNG result artifacts are present.
- [x] Result directories are indexed in `notes/10_results_index.md`.
- [x] Synthetic payload generation is deterministic.
- [ ] Large model files are not committed and need external acquisition instructions.
- [ ] Final supplementary artifact list needs to be frozen.
- [ ] Any future large results should be separated from commit-friendly artifacts.

## Code Availability

- [x] Core code is in `rankcloak/`.
- [x] Experiment runner is in `scripts/run_experiment.py`.
- [x] Smoke script is in `scripts/run_smoke.py`.
- [x] Tests cover rank ordering, codecs, schemas, prompts, token filters, tail policy, and segmented protocol helpers.
- [ ] Final release tag is not yet created.
- [ ] Dependency versions should be pinned or recorded for paper-main reproduction.

## Responsible Use And Dual-Use Framing

- [x] README and notes frame the work as synthetic-payload research.
- [x] Notes state exact-copy assumptions.
- [x] Notes state shared `K_common` assumptions.
- [x] Notes state no key exchange.
- [x] Notes state no encryption, authentication, signing, or cryptographic-security claims.
- [x] Notes state no real secrets or credentials are used.
- [ ] Final paper should include a dedicated limitations and responsible-use section.
- [ ] Final paper should avoid operational evasion language.

## Manuscript Writing Tasks

- [x] Methods draft exists in `notes/11_paper_methods_draft.md`.
- [x] Results draft exists in `notes/12_results_for_paper_draft.md`.
- [x] Figures and tables plan exists in `notes/13_paper_figures_tables_plan.md`.
- [ ] Introduction and related work need drafting.
- [ ] Ethics and responsible-use statement need drafting.
- [ ] Abstract and title need drafting after paper-main results are selected.
- [ ] Supplementary material index needs drafting.

## What Must Be Added Before Scientific Reports Submission

- [x] Paper-main experiment matrix has an implemented pilot and full profile.
- [x] Lightweight detector baseline has been implemented.
- [x] Statistical analysis plan and bootstrap implementation have been added.
- [ ] Model comparison or strong rationale for single-model scope.
- [ ] Robustness tests or explicit limitation that results only apply to exact-copy channels.
- [ ] Final figure generation scripts or notebooks.
- [ ] Archived code and result artifacts with stable identifiers.
- [ ] Final responsible-use and limitations section.

## What Is Already Sufficient

- [x] The repository has a coherent experimental framework.
- [x] The main payload representations are implemented.
- [x] The exact recovery mechanism is demonstrated in multiple pilots.
- [x] The current notes identify methodology, result artifacts, paper methods, paper results, and figure/table plans.
- [x] The current documentation is sufficient for drafting a methods section and planning a paper-main experiment.
