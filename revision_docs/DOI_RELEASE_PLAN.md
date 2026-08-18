# Code-and-data deposit workflow

Reserved archive DOI: `10.5281/zenodo.21987450`

## Scope

The repository provides an offline assembler for the combined RankCloak
code-and-data deposit. It stages a local candidate only and has no network,
upload, publication, release, or DOI-minting capability. Zenodo has reserved
the DOI above, but the draft record is not yet published; the journal article
DOI will be a separate identifier.

The active specification is
release/revision_v1_template/release_spec.json. Despite the retained directory
name, it describes the completed computational revision rather than the earlier
prospective confirmatory workflow. It depends on tracked canonical paths,
the sealed final experiment package, excludes both paperV1 and paperV2, and
does not depend on private manuscript staging or a legacy confirmatory release
index.

## Integrity model

The allowlist rejects path traversal, symlinks, model weights, caches, external
checkouts, credentials, participant identifiers, and raw human responses. It
pins the sealed final-package index, validates every included source hash and
size, verifies the selected external result references needed by the reported
revision, and requires an exact environment lock input. Only tracked files are
eligible for inclusion.

Some retained result manifests record the machine-local paths present when the
completed analyses were run. The assembler verifies their original bytes first,
then rebases those path strings in the staged copy and records the source and
archive hashes in the portable final-evidence manifest. Numerical results,
tables, predictions, figures, classifications, and scientific conclusions are
not transformed.

The archive includes no participant outcomes because no human study was
conducted. It includes final CUDA detector predictions, metrics, split
identities, and same-device evidence represented by the sealed final package,
but it does not infer cross-device equivalence. Model
identifiers, exact revisions, licenses, and configured hashes are retained;
model weights are not copied.

## Local validation

Run a final-ready dry run, assemble into a new directory under /tmp, and invoke
the independent verifier as documented in the release-template README. The
candidate contains SHA256SUMS, PACKAGE_MANIFEST.json,
PACKAGE_MANIFEST.sha256, an evidence-role registry, environment and third-party
inventories, the source-to-portable evidence identities, and an assembly
report. The verifier rejects unlisted files, altered bytes, absolute home
paths, secrets, model weights, and inconsistent metadata.

## External action

After author review, a separate authorized workflow must upload the
independently verified candidate and publish the existing Zenodo draft. None of
those actions is performed or assumed by the local assembler, and the reserved
DOI may not resolve publicly until publication.
