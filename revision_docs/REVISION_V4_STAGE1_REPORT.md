# V4 Stage 1 report

Stage 1 is complete. The V4 scaffold, source audit, initial offline diagnostics, local GPU check, and validation artifacts are available. The manuscript remains an internal working draft. Boundary coherence scores, final theoretical comparisons, completed reviewer answers, and the V4 archive version remain future work.

## Inspected state and preservation

The inspected checkout was `60d23fffe5962d4e0b652fa045d762ca10bc3dcc` on `main`. It matches the external inspection commit, but all findings below were checked against local files. The only initial local change was the untracked author plan at `paperV4/RankCloak_V4_Revision_Plan.md`. No applicable `AGENTS.md` was found in the repository or its ancestors. Work remained on `main`.

The audit covered the project documentation, author plan, complete current review letter, V3 manuscript, supplement, response, cover letter, bibliography, and the relevant codec, generation, diagnostics, protocol, filter, detector, evaluator, and release code. The old manuscripts and historical evidence were used as sources rather than regenerated.

The final preservation check matched all 7,256 protected file hashes, including paperV1 through paperV3, existing configurations, the audited V1 and V3 results, and the original requests file. The author's plan also retains its original bytes. The requests SHA-256 is `de663d74f7b91880f50b08836aa373022d5aaeddd281902a059c8d99db4136de`. The only edited file that was already tracked at the start is `revision_docs/DOI_RELEASE_PLAN.md`. The author's previously untracked plan is included unchanged with the new V4 work.

The initial state and final checks are recorded in [initial_state.json](../results/revision_v4/provenance/initial_state.json) and [stage1_validation.json](../results/revision_v4/provenance/stage1_validation.json).

## Deliverables

| Deliverable | Files and status |
| --- | --- |
| Main manuscript | [main4.tex](../paperV4/scientific_reports/main4.tex) and [main4.pdf](../paperV4/scientific_reports/main4.pdf), 23 pages. V3 baseline with a conspicuous draft marker, complete filter Methods, claim-specific watermarking citation, and corrected archive availability. |
| Supplement | [supplementary4.tex](../paperV4/scientific_reports/supplementary4.tex) and [supplementary4.pdf](../paperV4/scientific_reports/supplementary4.pdf), 31 pages. Includes initial six-case evidence in Note S16. |
| Response | [response_to_reviewers_v4.tex](../paperV4/response/response_to_reviewers_v4.tex) and [PDF](../paperV4/response/response_to_reviewers_v4.pdf), seven pages. The exact-block ledger is [review_comments.json](../paperV4/response/review_comments.json). |
| Cover letter | [cover_letter_v4.tex](../paperV4/cover_letter/cover_letter_v4.tex) and [PDF](../paperV4/cover_letter/cover_letter_v4.pdf), one page. Explicitly describes an unfinished internal draft. |
| Revision handoff | [REVISION_V4_PLAN.md](REVISION_V4_PLAN.md), [REVIEW_RESPONSE_MATRIX.md](../paperV4/REVIEW_RESPONSE_MATRIX.md), and [REVISION_V4_TECHNICAL_AUDIT.md](REVISION_V4_TECHNICAL_AUDIT.md). |
| Offline analysis | [revision_v4_stage1.py](../rankcloak/revision_v4_stage1.py), [analysis entry point](../scripts/analyze_revision_v4_stage1.py), and [frozen selection configuration](../configs/revision_v4/stage1_offline.json). |
| Evidence inventory | [audit_revision_v4_evidence.py](../scripts/audit_revision_v4_evidence.py), boundary and quantization CSVs, model identities, filter masks, and public archive inventories. |
| Reproduction and checks | Scripts to prepare the initial scaffold, plot the diagnostics, compile the documents, verify Stage 1 artifacts, and run the bounded GPU check. Focused tests are in [test_revision_v4_stage1.py](../tests/test_revision_v4_stage1.py). |

The scientific reports folder contains the necessary V3 style files, bibliography, figures, and table dependencies. Its two new inputs are `v4_filter_methods.tex` and `v4_stage1_entropy.tex`. The complete new-file inventory and hashes are recorded in `results/revision_v4/provenance/stage1_artifact_manifest.json`. Temporary LaTeX auxiliaries are ignored locally. PDFs and generated bibliographies are retained.

The response preserves 13 complete source blocks. These comprise the whole editor letter, Reviewer 2's acknowledgment, Reviewer 1's general assessment and wording instruction, and all ten numbered comments. There are 16 unresolved response sets, each containing a direct-answer placeholder, an evidence/change placeholder, and a location placeholder. They are explicitly internal pending work. All proposed section labels in the matrix exist in the V4 sources.

## Recovered entropy evidence

All 720 source rows reconcile with raw records. They contain 360 RankCloak attempts and 360 ordinary controls. Each gate level retains 120 RankCloak attempts. Ungated and moderate embedding each completed 120 payloads. Strict embedding completed 114 and retains all six failures.

The following are the complete failure identities. Intervals are zero-based and half-open.

| Trial suffix | Model | Requested ranks | Consumed ranks | Token budget | Longest below-threshold interval | Length |
| --- | --- | --- | --- | --- | --- | --- |
| 2b5e682d0007193b56249891 | Mistral | 128 | 110 | 768 | [382, 432) | 50 |
| 941eed91360afbdc8a26bf09 | Qwen | 128 | 116 | 768 | [419, 582) | 163 |
| 9ae150681b7e7347c8241d5a | Qwen | 64 | 54 | 384 | [36, 75) | 39 |
| a9ff67133eeaa189fa8be8b1 | Qwen | 128 | 106 | 768 | [322, 357) | 35 |
| c4160dc01512e365fbecb170 | Qwen | 128 | 59 | 768 | [501, 768) | 267 |
| cbb747058efe6a8b1fbd18ec | Qwen | 128 | 116 | 768 | [350, 401) | 51 |

Every case uses `ascii_b16`, exhausts its original 6L budget, and has fewer than L eligible positions. The retained lineage uses no token filter. The realized eligible fractions range from 0.076823 to 0.151042. This is direct capacity accounting. It does not identify a linguistic cause.

The analysis includes twelve exact ungated/moderate matches and three strict successes selected by a declared identity hash within the failed model, representation, and prompt strata. All 21 trajectories contribute to 6,682 position rows. Fields include token ID, entropy, threshold margin, eligibility, role, observed rank, surprisal, rank pressure, consumed and remaining ranks, rolling progress, and exact prefix byte offsets. All maximal below-threshold runs are retained. The first low-progress window uses 32 positions and at most two consumed ranks. Initial partial windows are labeled separately.

Pinned vocabulary-only detokenization reproduced all selected texts. It executed no likelihood evaluation. The aligned longest-run contexts include a public-services list, a repeated formatted marketing update, a water-source heading, library programs and user management, a revised CRM update with budget and closing text, and a project-proposal outline. These are descriptive observations, not causal categories. The implementation uses sampled ordinary skips under the gate, so the older proposed greedy-skip description would be inaccurate.

The retained outputs are [the six-case CSV](../results/revision_v4/source_tables/entropy_six_failures.csv), [all selected trajectories](../results/revision_v4/source_tables/entropy_selected_runs.csv), [position diagnostics](../results/revision_v4/source_tables/entropy_positions.csv), [aligned windows](../results/revision_v4/source_tables/entropy_text_windows.csv), [the concise report](../results/revision_v4/ENTROPY_FAILURE_REPORT.md), and [the figure](../results/revision_v4/figures/entropy_failure_traces.pdf). The [offline manifest](../results/revision_v4/provenance/offline_manifest.json) binds 734 input hashes and the derived outputs. No failed trial was regenerated or granted extra tokens.

## Other substantive findings

The complete filter export contains all 19 literal blocked substrings and the surrounding decoding, mask, ordering, exception, whitespace, control-character, backslash, and special-token contract. The main Methods contains the entire exclusion list. The historical filter has not changed. Its source SHA-256 is `8d09fdfd4670bead5d233cb9d7b4cb57ae465bf1decf37dd65b223c7d71808d8`. Selection fully sorts allowed tokens. Known-token recovery counts greater scores and lower-ID ties. The later complexity analysis must preserve this distinction and account for repeated diagnostic sorts, model prefills, vocabulary passes, and KV storage.

The implemented Markdown operation normalizes line endings, trims whitespace, and prefixes each line with `> `. Extraction then uses saved offsets on the new tokenization. Boundary shifts, retokenization, changed context ranks, and irreversible trimming are separate mechanisms. This does not show that every Markdown workflow fails. The audit specifies narrow missing comparisons rather than another complete robustness matrix.

The existing model-aware detector consumes saved token likelihoods. Reproducing them independently requires the model and runtime configuration, prompts and contexts, tokenization, segmentation, and probability convention. Visible-text scoring is a different access condition. Detection is not itself decoding, but an attacker with the full receiver configuration may recover the payload. No cryptographic secrecy is implied.

The Q4/Q8 audit verified identical contexts and observed token arrays for all 1,920 retained same-history pairs. Recomputed counts reconcile across 244,440 positions, including 69,528 observed-rank changes. These comparisons are distinct from divergence after independently generated paths split and from cross-quantization payload decoding. Raw neighbor-logit margins are not retained. A small conditional follow-up is specified below.

The coherence inventory contains 8,280 structural boundaries across 1,440 segmented trials and 240 payload groups. Each has an eligible ordinary prefix with the same generator and exact prompt context. Word-length and tokenizer alignment exclusions remain to be evaluated. The prospective design uses matched ordinary pseudo-boundaries, same-prompt shuffled tails, a semantic-relatedness proxy, and conditional compatibility from the existing independent evaluator map. It requires payload-cluster inference and reports donor reuse. Automated scores will not be described as proof of reader judgments.

The fallback in the technical audit is explicitly a proposal. It combines configuration identification, independently decodable blocks, integrity checks, bounded outer error/erasure correction, and retransmission. ECC cannot repair an arbitrary wrong model codebook or persistent synchronization loss.

V3 reference 16 is `Cai2025EntropyGuidedWatermarking`. The [arXiv record](https://arxiv.org/abs/2504.12108) has no journal reference, and a formal publication was not verified. The V4 bibliography adds [Lee et al., ACL 2024](https://aclanthology.org/2024.acl-long.268/) as peer-reviewed precedent for entropy-threshold selective watermarking. The text separates watermark detection from arbitrary-payload recovery.

The [original Zenodo release](https://zenodo.org/records/21987450) is published. Its public ZIP checksum and all 411 embedded manifest entries were verified. A newer [version 2.0.0](https://zenodo.org/records/22555497) is also published under concept DOI `10.5281/zenodo.21987449`. Its ZIP checksum matches and all 8,342 files match commit `ce853d42d6ba64065cb63c6bdfc0d825c62734cd`. It already contains V3 code, configurations, results, and manuscript files. V4 is absent. The newer public description says manuscripts are excluded, which conflicts with its inventory. The stale local deposit document is corrected. A later release must retain V3, add completed V4 evidence and portable provenance, and correct that description. No deposit or journal submission was performed.

## GPU readiness

The local compute device was an idle NVIDIA RTX 5000 Ada Generation with 32,760 MiB reported memory, driver 590.48.01, and UUID `GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf`. A separate Quadro T2000 served display activity. The check did not interrupt or compete with an active compute process.

All four configured GGUF files were present and matched their pinned sizes and SHA-256 hashes. The smoke used the existing Qwen 2.5 7B Q4_K_M file and `llama-cpp-python` 0.3.23 with CUDA runtime 12.4.127 and cuBLAS 12.4.5.8. The backend log records 29 of 29 layers offloaded to CUDA. Sixteen forced tokens were generated and all sixteen ranks were recovered exactly.

Total smoke wall time was 18.52 seconds, including the model hash check. Model loading, generation, and replay took 4.03 seconds. The sampled peak process allocation was 4,530 MiB, measured every 0.25 seconds with `nvidia-smi`. This is a sampled allocation peak rather than an allocator-exact maximum. The script had a 570-second watchdog. The Stage 1 GPU allowance was not approached. No dependencies were replaced, weights downloaded, paid compute acquired, or long CPU fallback started.

Evidence is retained in [gpu_smoke.json](../results/revision_v4/provenance/gpu_smoke.json), [gpu_backend.txt](../results/revision_v4/provenance/gpu_backend.txt), and [local_model_inventory.json](../results/revision_v4/provenance/local_model_inventory.json). This short check establishes local generation/replay readiness, not throughput for the future coherence evaluation. The semantic encoder's exact file and dependency hashes must still be frozen before its pilot.

## Validation

The focused suite passed 123 tests in 4.52 seconds. It combines 85 relevant existing checks with 38 new checks covering trace corruption, inclusive gating, comparator selection, alignment windows, genuine filter edge cases, and exact review preservation. [Pytest output](../results/revision_v4/validation/pytest_stage1.txt) and [JUnit results](../results/revision_v4/validation/pytest_stage1.xml) are retained.

The independent artifact validator passed preservation, review quotation, pending-marker, manifest, six-case, and per-token reconciliation checks. All four documents compiled with resolved references. The only final layout warning is an inherited underfull alignment in the supplement's threshold table. That table was visually inspected and is readable. No final overfull box was reported.

Seventeen rendered pages were inspected. They cover the changed main text and bibliography, supplement marker and new evidence, all seven response pages, and the cover letter. The page inventory and visual-check record are under `results/revision_v4/validation`. Rendered temporary previews are under `/tmp/rankcloak-v4-rendered`. The retained PDFs are the reviewable artifacts. Whitespace checks passed outside three declared evidence exemptions. The literal generated-text CSV and captured backend and LaTeX outputs retain their original trailing spaces. They were not trimmed to silence the check. The exact check scope and warnings are retained in `validation/whitespace_check.json`.

## Stage 2 implementation and execution handoff

The detailed design is [boundary_coherence_plan.json](../configs/revision_v4/boundary_coherence_plan.json). The following counts are work units rather than unmeasured runtime promises.

| Work unit | Scope and exit condition | Estimated execution |
| --- | --- | --- |
| Boundary instrumentation | Freeze exact text windows, pseudo-boundary positions, donor map, same-prompt shuffles, encoder hashes, evaluator joins, boundary-straddling tokens, and missingness rules. Add CPU tests. | One instrumented inventory of 8,280 boundaries. No model scoring before the plan is frozen. |
| Control-only metric pilot | Select one identity by fixed hash in each generator, schedule, and six-category cell. Score ordinary and shuffled ordinary transitions without examining RankCloak scores. | At most 36 identities, 72 transitions, 144 semantic window embeddings, and 144 conditional/reference LM passes. |
| Pilot decision | Require ordinary transitions to exceed shuffled controls with a positive payload-bootstrap 95 percent interval for each prespecified metric. The proposed base DeBERTa pooling proxy may fail. Record failure and amend before RankCloak scoring. | One blinded validation and throughput report. Measure time and memory here before forecasting the full workload. |
| Full coherence analysis | If the pilot validates the metrics, score every eligible boundary in three arms. Apply the frozen payload-cluster bootstrap and donor-support sensitivity. | At most 24,840 transitions, 49,680 encoder windows before deduplication, and 49,680 conditional/reference LM passes before exclusions. Use 2,000 payload-cluster bootstrap draws. |
| Transmission decomposition | Freeze at most one visible-text success and one failure per model by identity hash. Reuse original outcomes and add only missing isolated transformation arms. Include configuration-fingerprint acceptance and mismatch tests. | At most six source covers and thirty additional cover replays. Count their constituent segments before loading a model. |
| Quantization explanation | First reuse the retained same-history comparisons. Consider an offline Q4-to-Q8 bounded-codec diagnostic from saved observed ranks under a separate endpoint. Only collect raw neighbor margins if required for a specific claim. | Optional sixteen class-by-codec identities, two quantizations, at most 64 positions each. This is 32 replay prefixes and at most 2,048 evaluated positions. |
| Theory and integration | Complete the arithmetic-coding and grouping comparator audit, precise distributional/security discussion, configuration argument, and proposed fallback. Integrate checked evidence and resolve response sets with final locations. | One comparison table, one protocol proposal, targeted manuscript edits, and sixteen response sets. No broad detector retraining or experiment-matrix rerun. |
| Archive preparation | Extend the release specification for final V4 inputs, outputs, scripts, environment, model/license identities, source commit, and reproduction instructions. Independently verify a portable candidate and its description. | One local candidate after evidence is final. External publication and submission are separate future actions. |

Run one GPU process at a time and recheck occupancy before every load. Use the pinned local models. Preserve all historical outcomes and denominators. Do not run another entropy generation matrix to recover fields already saved.

## Reproduction notes

From the repository root, the following regenerate the offline diagnostics, inventory, and figure using existing environments.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-generation-v3/bin/python scripts/analyze_revision_v4_stage1.py --align-tokenizers
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-generation-v3/bin/python scripts/audit_revision_v4_evidence.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/plot_revision_v4_stage1.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/validate_revision_v4_stage1.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/build_revision_v4_documents.py
```

The offline analyzer has a position-only mode when `--align-tokenizers` is omitted. Use a separate output directory for such a run to avoid replacing the retained aligned analysis. The inventory script verifies the original public ZIP when `/tmp/rankcloak-v4-public-archive.zip` is present. The latest ZIP was downloaded to `/tmp/rankcloak-v4-public-archive-v2.zip` and checked against its published MD5 and the full named Git tree. Both inventories, metadata responses, and verification methods are retained in provenance files. Download URLs are in the saved official Zenodo metadata.

Do not rerun `prepare_revision_v4_stage1.py` over edited drafts. It is a create-only scaffolder and deliberately refuses differing targets. Rebuild the PDFs directly after conservative source edits. The GPU smoke is already complete and does not need repeating for the Stage 2 handoff.
