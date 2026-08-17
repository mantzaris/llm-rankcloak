# RankCloak code-and-data archive template

This directory contains the current allowlist for assembling one local
code-and-data archive for the completed RankCloak computational revision. The
assembler is offline: it cannot contact Zenodo, upload files, mint a DOI,
publish a GitHub release, or modify repository history.

The recipe uses only tracked files available in a clean checkout. It includes
the RankCloak package, analysis and verification scripts, tests, frozen
configuration and environment metadata, the public 480-payload corpus,
authoritative processed analysis tables, final detector predictions and
metrics, the final vector figures and plot-source tables, the completed
paperV2 publication package, and a portable derivative of
results/revision_v1/final_experiment_package. The source package index is pinned
by SHA-256 and its included files and selected external references are verified
before assembly. The environment bundle is an execution-time dependency and
hardware record, not a live-current-source attestation: its frozen source pins
legitimately retain the subsequently retired, non-scientific manuscript helper.

Machine-local paths in retained result metadata are rebased only in the staged
copy. The archive records both source and staged hashes for every rewritten
file. Source results are never edited. Duplicate raster previews, pre-final
diagnostics, internal process inventories, model weights, caches, virtual
environments, external checkouts, raw human responses, and participant
identifiers are excluded. No human participant outcomes were collected or
included. The final study used the retained CUDA detector results and makes no
cross-device equivalence claim.

Preview all source resolution and local readiness checks without writing an
archive:

```console
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-code-data-candidate \
  --dry-run --require-final-ready
```

Assemble a candidate into a new temporary destination:

```console
python scripts/build_revision_release.py \
  --spec release/revision_v1_template/release_spec.json \
  --output-dir /tmp/rankcloak-code-data-candidate \
  --require-final-ready
```

Then independently reread every staged byte:

```console
python -B scripts/verify_revision_release.py \
  /tmp/rankcloak-code-data-candidate --require-doi-null
```

A successful final-ready build means that the local content is internally
complete, portable, allowlisted, and hash-verified. It does not authorize or
perform an external deposit. Zenodo record creation, DOI assignment, upload,
and publication remain separate author-controlled actions.
