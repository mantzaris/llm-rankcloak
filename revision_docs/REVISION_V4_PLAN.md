# V4 revision plan

Stage 1 is a bounded audit, scaffold, and initial offline analysis. It does not establish submission readiness. The author plan at `paperV4/RankCloak_V4_Revision_Plan.md` and the unchanged review letter remain authoritative context. The inspected checkout is commit `60d23fffe5962d4e0b652fa045d762ca10bc3dcc` on `main`. The only initial local change was the untracked author plan. Its bytes and 7,256 protected files were recorded before implementation.

## Evidence and scope

The previous submission is `paperV3`. The primary evidence is the corrected `primary_v2` study and sealed package under `results/revision_v1/final_experiment_package`. The V3 extension is under `results/revision_v3`. Historical source names do not identify manuscript version numbers reliably. Preserve the older experiments, invalidated shards, pinned configurations, and original manuscripts.

Every new diagnostic must use a distinct V4 output namespace, stable identities, input hashes, explicit denominators, and a declared selection rule. Existing adverse outcomes remain in their original denominators. Do not repeat the full primary experiment or retrain the old detector matrix to explain existing results.

## Work groups

| Group | Existing evidence | Work type and remaining need |
| --- | --- | --- |
| Transformation, configuration, fallback | Robustness ledger and exact transformation/replay code | Theory and source audit now. At most thirty additional cover replays if the isolated comparisons are absent. Proposed fallback remains unimplemented. |
| Boundary coherence | Actual segment boundaries and prompt-matched ordinary controls | New scoring of retained text. Validate the proposed semantic and independent-model metrics on ordinary/shuffled controls first. |
| Model-aware detectability | Frozen features, splits, scores, and predictions | Explain trace access and distribution mismatch. Optional visible-text scoring is a separate access condition. No broad retraining. |
| Entropy failures | All six failures and retained per-token fields | Offline analysis completed in Stage 1. Interpret matched windows carefully. No regeneration or budget extension is needed now. |
| Computational complexity | Source-derived sort, counting, replay, and diagnostic operations | Mathematical accounting and primary-source comparator audit. Keep historical wall times unchanged. |
| Bibliography | V3 reference 16 and primary publication records | Verified key and publication boundary. Add claim-specific peer-reviewed selective-watermark precedent in the V4 copy. |
| Exact filter | Actual predicate, wrapper, masks, and tests | Complete rule export and main Methods insertion in Stage 1. Historical behavior remains unchanged. |
| Quantization divergence | Same-history Q4/Q8 traces and independent paths | Reuse retained comparisons. Optional maximum 2,048-position paired margin replay only if necessary. Cross-quantization decoding is a separate endpoint. |
| Archive coverage | Both public ZIPs, original manifest, and newer release Git tree | Published version 2.0.0 contains V3 and lacks V4. Prepare a later version retaining V3 and adding completed V4 evidence with portable provenance. Correct the description to match the inventory. No external deposit in Stage 1. |

## Stage 2 implementation and execution order

1. Validate and freeze the boundary inventory, donor mapping, local encoder hashes, evaluator tokenization, metric definitions, and missingness rules. The design is in `configs/revision_v4/boundary_coherence_plan.json`. Use actual inventory totals in `results/revision_v4/provenance/evidence_inventory.json`.
2. Implement CPU tests for byte boundaries, pseudo-boundary length matching, donor exclusion, payload grouping, and score-window accounting. Test corruption and ambiguous-token boundaries without loading a model.
3. Run at most 36 ordinary/shuffled pilot identities, yielding at most 72 transition pairs. The semantic encoder requires at most 144 window embeddings. Conditional compatibility and its prompt-only reference require at most 144 scoring passes. Use one GPU process at a time and verify occupancy immediately before loading it.
4. Evaluate the pilot against the declared validation rule. If a metric fails, report the failure and amend the metric plan before computing RankCloak outcomes. Do not select a metric based on an improvement for RankCloak. The local DeBERTa base encoder is a proposed proxy and may fail this validation.
5. If validated, score all eligible historical boundaries in three arms. The audited maximum before text-window exclusions is 24,840 transition scores, 49,680 encoder windows before deduplication, and 49,680 conditional/reference scoring passes. The actual workload must be updated from eligible windows and unique inputs. Derive elapsed-time forecasts from the pilot's measured scoring throughput and length distribution. The short Stage 1 generation smoke is not a valid scoring benchmark.
6. Implement the narrow transformation decomposition and configuration-fingerprint tests. Use at most six source covers and five genuinely missing transformation arms per cover. Count constituent segment replays before execution. Retain original controls and outcomes rather than reclassifying them.
7. Decide whether retained Q4/Q8 diagnostics fully support the explanation. If local margins add necessary evidence, freeze sixteen class-by-codec identities and at most 64 positions per identity for two quantizations. This is 32 replay prefixes and at most 2,048 evaluated positions. Otherwise omit that GPU work. Any new cross-quantization codec endpoint needs its own denominator and definition.
8. Produce grouped uncertainty, missingness tables, boundary illustrations, and compact figures. Complete comparator theory and the conceptual fallback. Integrate only verified findings into the draft and replace response placeholders only for genuinely completed points.

## Later completion checks

Check all numerical statements against source tables. Preserve every review quotation and current reviewer label. Resolve reference and figure warnings, inspect changed PDF pages, and update final page/line locations only after the layout is stable. Keep complete filter rules in the main Methods.

Prepare a new code/data archive version after the evidence is complete. Include the final source commit, V3/V4 configs, raw diagnostic inputs or portable references, derived tables, figures, analysis scripts, environment locks, model identifiers and licenses, and reproduction commands. Exclude model weights and restricted third-party text. Verify the public archive version and exact contents before updating availability claims. Journal submission and external publication remain outside Stage 1.

The review-to-evidence mapping is `paperV4/REVIEW_RESPONSE_MATRIX.md`. The substantive code and source audit is `revision_docs/REVISION_V4_TECHNICAL_AUDIT.md`. Stage 1 completion and actual validation results are recorded in `revision_docs/REVISION_V4_STAGE1_REPORT.md`.
