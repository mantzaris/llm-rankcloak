# Local V4 release specification

This is a preparation document. No deposit, public release or journal submission has been made in Stage 2.

The published version 2.0.0 remains DOI **10.5281/zenodo.22555497**. Its concept DOI is **10.5281/zenodo.21987449**. The earlier published version has DOI **10.5281/zenodo.21987450**. The Stage 1 archive audit verified that version 2.0.0 covers V3 source commit `ce853d42d6ba64065cb63c6bdfc0d825c62734cd`, including 65 files under paperV3, and does not cover V4. Its manuscript-exclusion description conflicts with those contents. Stage 2 keeps the valid published DOI and makes no V4-deposit claim.

## Proposed description

RankCloak research code and retained computational evidence for surface-form concealment of synthetic cryptographic artifacts in language-model-generated text. The archive includes historical V1 through V3 materials and the V4 revision sources, manuscript and supplement, reviewer response and cover-letter drafts, exact review wording, frozen configurations, model and environment identity records, reproducible analysis code, tables, figures, and execution and validation records. V4 includes retained-data entropy diagnostics, blinded boundary-control pilots including unsuccessful semantic validation, the prospectively selected exploratory boundary likelihood study, bounded transport decomposition, a canonical configuration compatibility API, and fixed-message Q4-to-Q8 decoding analysis. The study does not establish confidentiality, general transport robustness, statistical indistinguishability or human-perceived naturalness. The archive contains public synthetic artifacts and no collected human-participant judgments. Large pretrained model weights and local virtual environments are excluded. Exact acquisition references, revisions, hashes and applicable upstream model licenses are supplied instead.

Use this description only after the actual final inventory is checked. If manuscript files are deliberately excluded at release time, change both the packaging rule and this description. Do not leave contradictory metadata.

## Assembly contract

1. Finish the scientific and editorial review, including a decision on the still unsupported semantic/perceptual part of R1.4. Preserve the negative pilots and all adverse endpoints.
2. Commit the exact final source, evidence and document bytes. Record the full commit, tree ID and a clean-worktree check. A subsequent archive must be assembled from that commit, not from a mutable working directory.
3. Build a portable path, byte-count and SHA256 inventory for all selected committed files. Include Stage 1 manifests unchanged and Stage 2 raw/cache mappings, frozen plans, selections, metrics, failed attempts, GPU ledger, tests, source-based arguments and reproduction instructions. Do not include absolute local paths as the only way to locate required inputs. Local paths in provenance may remain as historical facts alongside pinned acquisition identities.
4. Include repository license and third-party attribution. Do not redistribute generator or semantic model weights, virtual environments, downloaded third-party source snapshots, local caches, credentials or incidental temporary files. Primary-source URLs and inspected revision/checksum metadata are sufficient for the comparator audit.
5. Package to a new local candidate directory, verify every payload file against the inventory, and verify every V4 claim's evidence links. Record archive SHA256 and file count. Keep packaging metadata separate from the payload manifest to avoid self-referential hashes.
6. Only in an explicitly authorized later release stage, create and publish a new version of the existing Zenodo concept record. Obtain its real version DOI and independently inspect the published files. Update manuscript Code Availability and the response using that verified DOI. Then rebuild and finalize the actual submission documents.

The local prospective inventory records the current release candidates. Its worktree hashes support review but are not a substitute for final commit-based assembly. The completion message identifies the pushed Stage 2 commit. The handoff report is versioned in that commit. No provisional DOI should be fabricated, and publishing the repository commit alone does not fulfill the V4 archive-coverage requirement.
