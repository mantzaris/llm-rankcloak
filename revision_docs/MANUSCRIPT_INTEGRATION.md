# Fail-closed manuscript integration

`scripts/revise_revision_manuscripts.py` is the only post-report manuscript
integrator. It is deterministic: it has no CLI option for an estimate, sample
size, confidence interval, or prose result. Scientific values can enter only
through hash-verified reporting products and the canonical progress and
unavailability manifests.

The command never edits `.paper/scientific_reports/main.tex`,
`.paper/scientific_reports/supplementary.tex`, either submitted PDF, or the
three revised source templates. Production execution pins the four submitted
artifacts to their pre-revision SHA-256 values and verifies them again after
the package is built.

## Production command

The confirmatory orchestrator invokes exactly:

```bash
.venv/bin/python scripts/revise_revision_manuscripts.py \
  --report-manifest results/revision_v1/reports/confirmatory_v2/report_output_manifest.json \
  --figures-dir results/revision_v1/reports/confirmatory_v2_figures \
  --statistics-manifest RESULTS_STATISTICS_MANIFEST.json \
  --mixed-model-manifest RESULTS_MIXED_MODEL_MANIFEST.json \
  --progress-manifest results/revision_v1/final_progress_snapshot_v1.json \
  --evaluator-unavailability-manifest RESULTS_EVALUATOR_UNAVAILABILITY.json \
  --manuscript-root .paper/scientific_reports \
  --output-dir results/revision_v1/manuscript_revision_v2
```

The uppercase paths above are orchestrator-resolved, hash-sealed inputs; they
are not manual values. `final_progress_snapshot_v1.json` is an atomic,
no-overwrite, byte-for-byte seal of a freshly generated and self-hash-verified
terminal progress snapshot. The orchestrator does not refresh canonical
progress after sealing it. `--fixture-mode` must never appear in a production
command.

For a no-write readiness check, append `--preflight-only`. Preflight verifies
the real report integrity and source seal, statistics and locked-R manifests,
canonical GPU accounting, evaluator-unavailability lineage, all 15 table and
18 plot identities, all 18 rendered PDFs, protected originals, deterministic
template replacement, centralized value reachability, and availability of
`latexmk`. It creates no output directory. Missing or fixture-labelled real
reports are fatal; preflight does not fall back to smoke or pilot products.

## Generated package

The output directory is created atomically and contains:

- `main2.tex`, `supplementary2.tex`, and `response_letter2.tex`;
- the centralized `generated_results.tex` vocabulary;
- `cross_document_value_manifest.json`, which records each value, its source
  product and SHA-256, derivation, formatting, macro, and document reachability;
- the verified 15 CSV/TeX tables and 18 rendered figure PDFs;
- bundled immutable copies of the report, statistics, locked-R, final-progress,
  and evaluator-unavailability provenance manifests;
- expanded audit-only TeX sources used for the journal word/display checks;
- `main2.pdf`, `supplementary2.pdf`, and `response_letter2.pdf`; and
- the self-hashed `manuscript_revision_manifest.json`.

Every computational placeholder is replaced by a centralized macro or a
required report table/figure before compilation. The five primary recovery
values (successes, total, rate, and Wilson interval endpoints) must be reachable
from all three documents. Result prose is mechanically derived from the same
macros, so no document can carry a separately typed sample size or estimate.

Human-study text is deliberately non-numeric and truthful: no participants
were recruited, no human outcome was estimated, and the materials and power
analysis were prepared. The package likewise states that no public DOI was
released and that deposit requires separate authorization. Automated quality
metrics are never relabelled as human outcomes.

## Fatal gates

The integrator stops without publishing a package if any of the following is
true:

- a source manifest, report product, table, figure, or submitted artifact is
  missing, symlinked, hash-mismatched, or inconsistent;
- the report source seal is not bound to the supplied statistics and locked-R
  manifests;
- a production display has any availability state other than the frozen
  computational set plus the explicitly unavailable human displays;
- an available table or plot row carries smoke, pilot, or exploratory identity;
- the primary table does not contain exactly 6,480 payload trials in production;
- canonical GPU use exceeds 165 GPU-hours or progress is incomplete/failed;
- a placeholder, stale unlaunched-computation statement, document-specific
  recovery value, or human/DOI claim survives materialization;
- the main paper exceeds 4,500 words, its abstract exceeds 200 words, its title
  exceeds 20 words, any figure legend exceeds 350 words, or it does not contain
  exactly seven display items; or
- any of the three LaTeX builds fails or does not emit a valid PDF.

Failed staging directories are non-authoritative because they lack a committed
manifest. A retry uses a new staging directory and never overwrites a completed
package with different inputs. Reuse of a completed package requires exact raw
SHA-256 equality for both `final_progress_snapshot_v1.json` and the
evaluator-unavailability manifest; timestamp or formatting drift is not
tolerated because the supplied progress file is immutable by contract.

## Tests

Run:

```bash
.venv/bin/python -m pytest -q tests/test_revision_manuscripts.py
```

The suite includes a clearly labelled non-scientific reporting fixture, all 18
fixture figure renders, all three PDF builds, centralized value tracing,
placeholder and protected-original checks, an anti-smoke test, and a
production-mode missing-report preflight that must fail without writing output.
