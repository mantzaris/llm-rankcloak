# Code-and-data deposit workflow

The original published version 1.0.0 has DOI [10.5281/zenodo.21987450](https://doi.org/10.5281/zenodo.21987450). The latest published version checked on 22 September 2026 is version 2.0.0, DOI [10.5281/zenodo.22555497](https://doi.org/10.5281/zenodo.22555497). The shared concept DOI is [10.5281/zenodo.21987449](https://doi.org/10.5281/zenodo.21987449).

The Stage 1 audit verified the original ZIP against its public checksum and all 411 files listed in its embedded manifest. That version lacks the V3 extension. The newer ZIP matches its public checksum and all 8,342 files in Git commit `ce853d42d6ba64065cb63c6bdfc0d825c62734cd`. It includes V3 code, configurations, results, and 65 files under `paperV3`. Neither version contains V4 evidence. The newer public description says manuscripts are excluded, but the archive contains manuscript files. A future deposit should correct that description.

Metadata, archive inventories, checksums, and coverage findings are retained under `results/revision_v4/provenance`, including `archive_integrity.json` and `archive_v2_coverage.json`. Internal offline-candidate flags in the original archive describe its assembly state and do not negate its later publication.

## Scope

The repository provides an offline assembler for the combined RankCloak
code-and-data deposit. It stages a local candidate only and has no network,
upload, publication, release, or DOI-minting capability. Both archive versions
identified above are published. The journal article DOI is a separate identifier.

The earlier package specification is
release/revision_v1_template/release_spec.json. Despite the retained directory
name, it describes the completed computational revision rather than the earlier
prospective confirmatory workflow. It depends on tracked canonical paths,
the sealed final experiment package, excludes both paperV1 and paperV2, and
does not depend on private manuscript staging or a legacy confirmatory release
index. It is not yet a complete V4 release specification.

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

A later authorized workflow should create a new version under the existing
concept DOI, retain the V3 evidence, and add the completed V4 code, configurations,
diagnostic inputs, derived tables, figures, environment records, model and license
identities, reproduction commands, and verified portable manifests. Record the
final source commit and make the public description agree with the actual file
inventory. Verify the published version and contents before changing final
availability claims. No external upload, publication, or submission is part of
Stage 1.
