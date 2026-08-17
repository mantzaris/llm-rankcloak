# Revision-v1 offline release template

This directory defines the allowlist for a future DOI-assigning repository
deposit. It is a staging template only. Running the assembler does not access
the network, contact Zenodo, mint or reserve a DOI, publish a GitHub release, or
perform any other external action.

The checked-in specification is a pre-publication staging template. The
computational revision has a closed confirmatory index; human-study approval,
participant work, and any external deposit remain outside this package. The
allowlist assigns every raw-result source an explicit evidence role. Tokenizer
preflight, three-model `smoke_v3`, held-out
smoke evaluation, and the compute projection are validation/gate artifacts and
cannot be pooled as confirmatory evidence. The invalidation registry and prior
GPU-charge ledger are forensic only; the ledger is never runtime-rate evidence.
Only the exact finalized `primary_v2`, `ablation_v2`, `multilingual_v2`,
`robustness_v2`, held-out evaluator, preprocessing/join, detector, locked-R,
statistics, theory, report, figure, and manuscript products are marked
confirmatory or final documentation. Mere directory existence is never enough:
the assembler excludes all of those paths unless
`results/revision_v1/confirmatory_release_index_v1.json` exists and revalidates.
That self-hashed index is accepted only after the immutable final progress
snapshot is complete/failure-free and every supervisor completion validator
passes. The assembler copies confirmatory trees only while the finalized index
continues to revalidate.
The invalid legacy primary tree and all `smoke_v2` shard bytes are rejected by the
assembler. The verified offline environment snapshot is allowlisted at
`environment/revision_v1/`. A dry run reports every missing artifact, excluded
file, unresolved participant-material marker, model identifier, and license
notice.

The superseded atomic CPU detector attempt is represented only by its immutable
read-only audit, with a supporting-methodological role. The separately
predeclared detector acceleration policy binds checkpoint/resume, one-worker
CUDA execution, equivalence thresholds, the authorized GPU UUID, and per-fit
hard-ceiling reserves without changing the frozen detector data, splits, seeds,
architectures, comparisons, or metrics. Neither operational artifact is a
scientific result or confirmatory pooling input.

Detector completion is accepted only as a cross-bound, self-hashed closure:
the supervisor-published final manifest binds its closed accounting status,
finalization candidate, two passing equivalence reports, and pre-final GPU
ledger; the terminal receipt binds that candidate, closed accounting status,
published manifest, and exact GPU intervals; the terminal status binds the
manifest, receipt, and ledger-incorporation marker. The release map retains all 56 valid fit
checkpoints and completed permit receipts, the finalization records, terminal
status, CPU/CUDA and same-CUDA equivalence packages, two CUDA benchmark records,
the deduplicated signed ledger and incorporation proof, and the immutable
confirmatory event log. It excludes execution/supervisor locks, active
unconsumed fit permits, recovered-permit quarantine, caches, and model weights.
A missing or changed link fails the completion validator before the release
index can be created.

Preview the current missing-artifact and exclusion report with:

```bash
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-revision-v1-candidate \
  --dry-run
```

To stage a recoverable draft locally, omit `--dry-run`. The destination must not
already exist. To enforce all local quality gates, add `--require-final-ready`;
that option still creates only an offline candidate with status
`draft_external_action_prohibited`.

Before final-ready assembly:

1. Replace missing source paths with exact immutable artifacts or materialize
   the paths documented in `release_spec.json`.
2. After all final outputs exist, verify without writing:

   ```bash
   python scripts/build_revision_confirmatory_release_index.py --dry-run
   ```

   Then create the no-overwrite index once and immediately reverify it:

   ```bash
   python scripts/build_revision_confirmatory_release_index.py
   python scripts/build_revision_confirmatory_release_index.py --check
   ```

3. Reverify the allowlisted environment snapshot with
   `scripts/build_revision_environment_lock.py --check`.
4. Resolve the human-study institutional markers after the required ethics
   determination, without adding participant identifiers or raw responses.
   Current power analysis, instrument, randomization, and IRB materials are
   pre-recruitment planning artifacts, not empirical human results.
5. Recheck the separate model license/provenance registry. All three configured
   model chains have pinned license records and no model weights are copied.
   The historical patient-Huffman tag remains an excluded reference-only item:
   its license is unresolved and no checkout bytes may be redistributed.
6. Review the allowlist rather than broadening it to the repository root.
7. Run the final-ready build and independently verify `SHA256SUMS`,
   `PACKAGE_MANIFEST.sha256`, and `ARTIFACT_EVIDENCE_ROLES.json`.

Independently re-read every staged byte and require the offline draft to retain
`doi: null` with:

```bash
python scripts/verify_revision_release.py \
  /tmp/rankcloak-revision-v1-candidate --require-doi-null
```

Local content readiness never implies publication readiness. A null DOI,
unresolved human-material placeholders, and absent external authorization are
reported as explicit publication blockers. Public upload, DOI reservation, and
publication require separate explicit approval and a separately audited
external workflow.
