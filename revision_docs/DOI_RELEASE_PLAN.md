# DOI release-candidate plan

## Boundary of this workflow

`scripts/build_revision_release.py` assembles a local directory only. It has no
HTTP client, Zenodo integration, upload command, DOI reservation operation, or
publication operation. Every package produced by this workflow retains the
external status `draft_external_action_prohibited`, including a package whose
local content passes every readiness gate. Public deposit remains a separate,
explicitly approved and audited action.

The assembler never invents a DOI. With no DOI in the specification, the
package records `doi: null` and `doi_provenance: not_assigned`. A syntactically
valid DOI can only be copied from explicit user-supplied metadata and is labeled
`user_supplied_unverified`; the assembler neither verifies nor reserves it.

## Allowlist and staging model

The JSON specification enumerates every input under one of these groups:

- source code;
- frozen configurations;
- public deterministic payload corpus;
- raw machine-readable results;
- processed results;
- statistical outputs;
- generated figures and tables;
- participant-free human-study materials;
- environment inputs;
- supporting documentation.

Each entry maps a repository-relative source to a package-relative destination.
Directory entries are recursively expanded in deterministic lexical order, but
every resulting file is rechecked against the exclusion policy. Sources cannot
escape the repository, destinations cannot be absolute or contain traversal,
symlinks and special files are refused, destination collisions are errors, and
optional source hashes are enforced when supplied.

Confirmatory entries have an additional mandatory gate. The specification
declares `results/revision_v1/confirmatory_release_index_v1.json`, and the
assembler copies no confirmatory-path byte unless that index exists, has a
valid self-hash, matches the closed source/destination/evidence map implemented
by the content-pinned release-index validator and mirrored by the current
release specification, and matches every current file hash and size. The index
is generated with no overwrite only after the sealed final progress snapshot
verifies complete with zero failures/remaining units and the existing
supervisor validators accept all four preprocessing products plus evaluator
join, detector, Python statistics, locked-R mixed models, theory, reports, 18
figures, and the final manuscript package. The assembler re-runs those live
validators when it reads the index. Extra, missing, changed, symlinked, cached,
model-weight, smoke, legacy-primary, invalidated, or superseded bytes fail
closed.

Assembly occurs in a temporary sibling directory. Files are created without
overwrite, copied bytes are immediately rehashed, and the completed directory
is installed with Linux `renameat2(RENAME_NOREPLACE)` when available. Existing
destinations are never replaced. If an atomic no-replace directory rename is not
available, the assembler fails closed instead of using a racy fallback.

## Mandatory exclusions

The assembler excludes:

- `.git`, virtual environments, dependency caches, test caches, and bytecode;
- every `models` directory and common model-weight format, including GGUF,
  SafeTensors, PyTorch checkpoints, ONNX, and HDF5 weights;
- internals of `external_sources` checkouts;
- private-key and credential files;
- high-confidence private-key, cloud-key, GitHub-token, and API-token content
  signatures;
- participant identifier exports, raw human responses, signed consent files,
  platform exports, contact information, and unsupported human-data binaries.

Cache exclusions are informational. Sensitive data, external checkout, model
artifact, secret, and symlink exclusions are final-readiness blockers so an
overbroad allowlist cannot silently receive a passing status. Model identifiers,
revisions, quantization, licenses, and artifact hashes are inventoried without
copying the model files.

## Generated metadata

An assembled directory contains:

- `README.md`, stating the offline and externally unpublished status;
- the project `LICENSE` and `CITATION.cff`;
- `REPRODUCE.md`, generated from declared commands that the assembler does not
  execute;
- `THIRD_PARTY_INVENTORY.json`, combining configured model metadata with
  explicitly declared method or dependency records;
- `ENVIRONMENT_INPUTS.json`, recording supplied dependency inputs and whether
  an exact lock is present;
- `ARTIFACT_EVIDENCE_ROLES.json`, assigning validation-only, compute-gate,
  forensic-only, confirmatory, planning, environment, or documentation roles to
  every allowlisted partition;
- `ASSEMBLY_REPORT.json`, listing missing inputs, exclusions, group counts,
  evidence partitions, readiness blockers, and non-blocking notices;
- `SHA256SUMS`, covering all payload and generated audit files except the
  checksum and package-manifest files themselves;
- `PACKAGE_MANIFEST.json`, containing ordered file sizes and SHA-256 hashes;
- `PACKAGE_MANIFEST.sha256`, independently hashing the package manifest.

No generated file contains a timestamp, temporary path, or claimed DOI, so the
content manifest is stable for identical source bytes and specifications.

## Readiness gates

`--require-final-ready` refuses staging unless all of the following hold:

1. Title, version, description, creator names, project license, and citation
   metadata are present.
2. Every required group contains at least one safe file and every required
   allowlist entry exists.
3. No blocking exclusion or secret signature is present.
4. Participant-identifier and raw-response declarations are explicitly false.
5. Participant materials, configurations, documentation, and package metadata
   have no unresolved institutional markers such as double-braced fields.
   Pattern definitions and test literals in source code are not mistaken for
   unresolved release metadata.
6. Every redistributed third-party item has a recorded license. An unresolved
   reference-only method is a notice only when its bytes are excluded; changing
   `artifact_included` to true makes the missing license a blocker.
7. An exact environment lock input is supplied.
8. At least one non-empty reproduction command is declared.
9. The staged files pass a second placeholder/secret audit and every content
   hash verifies.
10. Every confirmatory artifact is present in the verified confirmatory release
    index and the independently staged index reproduces the exact candidate
    byte set.

Passing these gates sets only the content state
`final_ready_offline_candidate`; external publication status remains
`draft_external_action_prohibited`.
The readiness report separately retains `publication_ready: false` with
DOI-not-assigned, unresolved human-material, and absent-publication-authorization
blockers as applicable.

## Current template state

The checked-in template is deliberately a draft. Its raw-result allowlist is
fail-closed around the `payload_fidelity_v2` contract:

- tokenizer preflight, runner `smoke_v3`, held-out evaluator `smoke_v3`, and both
  compute projections are validation or compute-gate artifacts that cannot be
  pooled as confirmatory evidence; `compute_projection_v2.json` is the
  historical 150-hour no-GO audit, whereas `compute_projection_165h_v2.json`
  is the authorized 165-hour GO gate, and neither is a scientific outcome;
- the Qwen partial-primary invalidation record and the six-shard prior-charge
  ledger are forensic only, with the ledger explicitly barred from scientific
  and runtime-rate use;
- the invalid legacy `results/revision_v1/primary` tree and all `smoke_v2`
  shard directories are rejected rather than copied;
- only exact finalized `primary_v2`, `ablation_v2`, `multilingual_v2`,
  `robustness_v2`, held-out evaluator/unavailability, strict stage preprocessing,
  primary evaluator join, detector, Python statistics, locked-R models, theory,
  reporting, figure, and generated manuscript paths appear in the confirmatory
  release map;
- all confirmatory directories, including the sealed upstream shards, are
  deliberately resolved as missing/unverified and contribute zero staged files
  until the complete downstream DAG and final index exist;
- the read-only audit of the superseded atomic CPU detector is supporting
  methodological material only. The predeclared checkpoint/CUDA/equivalence
  policy is source provenance, not a scientific outcome or pooling input; it
  preserves the exact 56-fit data, split, seed, architecture, comparison, and
  metric contract while binding the authorized GPU and hard-ceiling reserves;
- detector release closure retains the canonical detector products, all 56
  valid per-fit checkpoints and consumed-permit receipts, the signed terminal
  status and finalization receipt, both architecture equivalence packages and
  reports, the two supervisor-finalized CUDA benchmarks, the deduplicated GPU
  accounting ledger and its incorporation marker, and the confirmatory event
  log. Execution locks, an unconsumed active permit, quarantined stale permits,
  caches, and every model-weight artifact remain excluded.

The detector closure is cross-bound rather than inferred from directory
existence. The supervisor-published run manifest has its own
`manifest_sha256`; it binds the closed accounting-status hash, finalization
candidate, two passing equivalence reports, and pre-final ledger identity. The
terminal receipt has `terminal_receipt_sha256` and binds that candidate, closed
status, published manifest, and exact non-overlapping GPU intervals. The final
status has `status_sha256` and binds the manifest, terminal receipt, and signed
ledger-incorporation marker. The ledger has `ledger_sha256`, and the marker
proves every benchmark/equivalence interval in that ledger is incorporated in
the canonical final receipt. Any missing or changed link prevents the detector
completion validator and therefore prevents creation of the confirmatory
release index.

The frozen 480-payload public export is materialized and pinned: corpus SHA-256
`caf0db84c814e02474a3cd2fc5588a8283cbe00fe41ce448764c6cdc67baa8c0`
and JSONL SHA-256
`6e45aafbc639f49c8fb07fc3055aa2d25e47025a2e945c4ffd2f5ac017322fd2`.
The remaining blocking work is the absent confirmatory/derived outputs, absent
confirmatory release index, and unresolved institutional placeholders in the
pre-recruitment human materials.
Current validation and forensic machine artifacts are staged as immutable
hash-verified bytes in their explicit non-confirmatory partitions.
No participant recruitment, pilot ratings, responses, identifiers, or raw human
data are present. The three configured model chains have separate
license/provenance entries and model weights remain excluded. The historical
patient-Huffman `acl-2019` checkout has no license in that tree; only our audit
and commit/tag identifiers may be packaged, so its unresolved license is a
reference-only notice rather than permission to redistribute third-party
bytes.

These are release-readiness findings, not permission to change ethics metadata,
download licenses, publish results, or contact a DOI service.

## Frozen environment input

`environment/revision_v1/` is the offline, content-hashed environment input for
the release candidate. It records the exact installed Python distributions and
their metadata fingerprints; R 4.4.2 and the composite locked-package
resolution (project `.r_libs/revision_v1` for lme4/ordinal and the declared user
R 4.4 library for emmeans/jsonlite); llama-cpp-python/native-library hashes;
the host driver, CUDA runtimes, GPU UUIDs, CPU, memory, OS, and tools; required
deterministic launch variables; and frozen config, 480-payload corpus, model, payload-fidelity-v2 protocol,
scientific-source, validation-artifact, forensic-record, compute-gate, and
participant-free human-planning identities. The complete current `human_study`
tree is content-pinned while caches and bytecode are excluded and raw or
identifying human data are forbidden. Configured GGUF sizes and SHA-256 pins
are recorded but their bytes are never in the bundle. Building the observation
may use size-only local checks; the explicit `--check --verify-model-files`
command streams and verifies current local hashes without copying them.

The current compute-gate pin is the authorized 165-GPU-hour GO projection,
`compute_projection_165h_v2.json`, self-hash
`35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3`;
the older 150-hour no-GO projection remains validation history rather than the
active authorization.

The bundle is observational and performs no dependency installation or network
access. Its `requirements-lock.txt` is an exact installed-version snapshot,
not a portable wheel-hash archive; the accompanying JSON records distribution
metadata/RECORD hashes so this limitation is explicit. `bundle_status.json`
records that no model copy or external publication occurred. Verify internal
checksums and current scientific pins with:

```bash
.venv/bin/python scripts/build_revision_environment_lock.py \
  --output-dir environment/revision_v1 --check
.venv/bin/python scripts/build_revision_environment_lock.py \
  --output-dir environment/revision_v1 --check --verify-model-files
```

The release allowlist includes `environment/revision_v1/` as an
`environment_inputs` directory, and the assembler independently verifies its
file set and semantic completion markers. It continues to exclude model weights
and the R package-library binaries themselves.

## Commands

After the complete confirmatory DAG exists, preview the exact index without
writing, create it once, and reverify every bound byte:

```bash
python scripts/build_revision_confirmatory_release_index.py --dry-run
python scripts/build_revision_confirmatory_release_index.py
python scripts/build_revision_confirmatory_release_index.py --check
```

Before that point the first command must fail on the missing/incomplete final
progress seal; it must never index the active partial primary tree.

Read-only planning with an exact missing-artifact report:

```bash
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-revision-v1-candidate \
  --dry-run
```

Local draft staging, still without external action:

```bash
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-revision-v1-candidate
```

Final local content gate after every required artifact and metadata field is
resolved:

```bash
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-revision-v1-final-candidate \
  --require-final-ready
```

After assembly, independently run `sha256sum --check SHA256SUMS` from inside the
candidate, verify `PACKAGE_MANIFEST.sha256`, and run:

```bash
python scripts/verify_revision_release.py \
  /tmp/rankcloak-revision-v1-candidate --require-doi-null
```

The verifier rejects unlisted or tampered files, symlinks, model weights,
secret signatures, inconsistent evidence roles, network/external-action claims,
and a non-null DOI when requested. Do not upload the directory or edit the
manuscript with a DOI until the user separately approves that irreversible
external publication action.
