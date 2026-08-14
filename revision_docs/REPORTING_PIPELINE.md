# Revision reporting pipeline

`rankcloak.revision_reporting` is the evidence-to-display boundary for the
Scientific Reports revision. It does not accept estimates, sample sizes, or
other numeric result overrides. Its only scientific inputs are paths to
machine-output manifests. Every generated row retains the source artifact and
SHA-256 digest.

## Safety contract

- Statistics and theory artifacts must match the hashes declared by their run
  manifests before any table is read.
- Generic runtime summaries must use a manifest with an `outputs` object; every
  output requires `path`, `sha256`, and (preferably) `bytes` and `row_count`.
- Preprocessing output manifests may supply `unavailable.csv`. Reporting
  verifies its digest, row count, and the preprocessing invariants that these
  rows are excluded from estimands and are not recovery failures.
- The version-1 detector runner did not place output hashes in its run
  manifest. The reporting layer checks its fixed output filenames and declared
  row counts, then pins their byte hashes in `report_source_manifest.json`.
  Subsequent builds cannot silently replace that source seal.
- Detector run-manifest v2 is preferred and every fixed output must match its
  manifest-declared SHA-256 and byte count. A non-smoke v2 run is accepted only
  with the complete primary 28-split/56-execution contract, zero skips,
  failures, or smoke fallbacks, matching metric/split IDs, exact class balance,
  and a live hash binding to the strict primary preprocessing manifest.
- Primary inferential effects come only from a verified, non-validation locked
  R `mixed_model_run_manifest.json`. Reporting rechecks its frozen plan and R
  environment hashes, every input/output hash, the held-out feature-join
  manifest, the exact repository R-driver hash, all prespecified model
  statuses, no-fallback flags, and Holm
  contrasts. Python pooled/pairwise effects are descriptive only and cause a
  hard error if marked primary.
- Recovery proportions, confidence-interval ordering, payload analysis units,
  effect-arm sizes, runtime intervals, theory summary counts, and detector
  prediction counts are checked for internal and cross-table consistency.
- Inputs labelled smoke, pilot, exploratory, or limited are excluded from
  ordinary report rows. A detector/statistics smoke artifact is rejected unless
  `fixture_mode=True`. Fixture mode exists only for automated tests and its
  products are labelled non-scientific.
- Missing results become an `unavailable` row with a reason. They are never
  imputed, reconstructed, simulated, or replaced with example effect sizes.
- The evaluator-unavailability manifest is self-hash checked against its four
  source files and 48 source-record hashes. Reporting discloses
  17,232 scored quality units plus 48 terminal excluded non-outcomes as the
  complete frozen 17,280-unit accounting; the 48 are never scored or imputed.
- Report products are atomic and write-once. An identical retry is a no-op; a
  differing retry raises `ReportArtifactConflict` before any output is changed.

## Inputs

The usual call is:

```bash
.venv/bin/python scripts/build_revision_reports.py \
  --statistics-manifest results/revision_v1/analysis/statistics_run_manifest.json \
  --theory-manifest results/revision_v1/theory/theory_validation_manifest.json \
  --detector-manifest results/revision_v1/detectors/detector_run_manifest.json \
  --mixed-model-manifest results/revision_v1/statistics/mixed_primary_v1/mixed_model_run_manifest.json \
  --evaluator-unavailability-manifest results/revision_v1/heldout_evaluator/upstream_dependent_unavailability_v1.json \
  --runtime-manifest results/revision_v1/runtime/runtime_report_manifest.json \
  --preprocessing-manifest results/revision_v1/analysis_inputs/ablation/preprocessing_output_manifest.json \
  --output-dir results/revision_v1/reports/final
```

The statistics manifest may already contain the runtime summaries in
`continuous_summary.csv`; in that case a separate runtime manifest is
optional. A generic runtime summary table should use the statistics column
vocabulary where possible: `model_id`, `protocol_variant`, `hardware_id`,
`outcome`, `mean`, `ci_low`, `ci_high`, and `n_payloads`. Supported main-table
outcomes include encoding and decoding time/throughput, payload bits per
second, peak RAM/GPU memory, and cover tokens per payload byte.

Runner `runtime_manifest.json` files are accepted as metadata-only sources.
They add environment provenance but cannot populate performance results because
they contain no measurements or output hashes.

Unavailable conditions are grouped and displayed in Supplementary Tables S7
and S13 with the explicit taxonomy
`condition_unavailable_not_recovery_failure`. They never enter either main
table or any recovery denominator.

## Outputs

Each build writes:

- `report_source_manifest.json`: hashes and sizes of every source manifest and
  evidence artifact.
- `report_output_manifest.json`: hashes and sizes of all generated products,
  excluding the output manifest itself to avoid a circular digest.
- `report_integrity.json`: source/hash/sample-size gates and the availability
  state of every planned table and figure.
- `display_registry.json`: stable Figure 1--5, Table 1--2, Supplementary Figure
  S1--S13, and Supplementary Table S1--S13 identifiers and LaTeX labels.
- `tables/*.csv` and `tables/*.tex`: machine source plus LaTeX for two main and
  thirteen supplementary tables.
- `plots/sources/*.csv`: one source file for each of five main and thirteen
  supplementary figures.
- `plots/plot_registry.csv` and `plots/plot_revision_figures.py`: a deterministic
  matplotlib renderer. The renderer displays an explicit unavailable panel when
  evidence is missing.

The fixed main registry contains five figures and two tables. Validation fails
if it exceeds the approved seven-display limit or if a number, ID, or LaTeX
label is duplicated.

Render one generated plot during a fixture smoke test with:

```bash
.venv/bin/python results/revision_v1/reports/fixture/plots/plot_revision_figures.py \
  --output-dir /tmp/rankcloak-report-plots \
  --format png \
  --only main_figure_1
```

Do not use `--fixture-mode` for manuscript evidence.

## Programmatic API

- `load_verified_sources(...) -> VerifiedSources`
- `build_revision_reports(...) -> ReportBuild`
- `display_registry() -> dict`
- `supplementary_table_rows(sources, index)`
- `plot_source_rows(sources, plot_id)`
- `verify_report_output_manifest(output_dir) -> dict`

The public build signature deliberately contains no scientific numeric fields.
Selection of compact main-table rows is deterministic; complete available rows
remain in the corresponding Supplementary tables and plot-source files.

## Planned evidence routing

| Display | Machine source |
|---|---|
| Main Figure 1 | Theory capacity and quality plot tables |
| Main Figure 2 | Recovery and effective-payload-rate summaries |
| Main Figure 3 | Human naturalness/suspiciousness summaries; unavailable before ratings |
| Main Figure 4 | Replay, lead-in, transformation, and mitigation summaries |
| Main Figure 5 | Detector grouped metrics |
| Main Table 1 | Aggregated successes and payload-condition counts from recovery summaries |
| Main Table 2 | Primary effect contrasts and representative runtime summaries |
| S1--S13 | Full protocol/model/corpus/recovery/model/ablation/robustness/human/detector/runtime/failure routes encoded in the module |

This layer prepares report components only. It does not edit `main.tex`,
`supplementary.tex`, or their revised counterparts.
