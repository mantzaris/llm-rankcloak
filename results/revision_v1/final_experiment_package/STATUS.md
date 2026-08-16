# Computational evidence status

Evidence artifact only. This file does not revise manuscript or response-letter text.

## Verified work-unit ledger

| Stage | Completed | Total | Successful | Unavailable | Failures | Remaining |
| --- | --- | --- | --- | --- | --- | --- |
| primary_v2 | 14400 | 14400 | 14400 | 0 | 0 | 0 |
| ablation_v2 | 1872 | 1872 | 1824 | 48 | 0 | 0 |
| robustness_v2 | 3744 | 3744 | 3408 | 336 | 0 | 0 |
| multilingual_v2 | 1152 | 1152 | 1152 | 0 | 0 | 0 |
| heldout_evaluator | 17280 | 17280 | 17232 | 48 | 0 | 0 |
| neural_detector | 56 | 56 | 56 | 0 | 0 | 0 |

## Audit state

| Item | Result | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Git HEAD | 223a06f649494eb1fefeca8832cf89a069d92cdc | confirmatory | analysis/revision_v1/evidence_specs/final_evidence_records.json | Exact historical baseline and origin/main relationship verified without reset, pull, checkout, or branch creation |
| Authoritative work-unit ledger | 38,504/38,504 complete; 38,072 successes; 432 unavailable; 0 failures; 0 remaining | confirmatory | results/revision_v1/final_progress_snapshot_v1.json | Stage totals and unavailability reconcile exactly; all 56 detector fits are included |
| Payload cryptographic validation | passed | confirmatory | release_inputs/revision_v1/PAYLOAD_MANIFEST.json | 480 rows, 8 classes, 60 per class, no invalid payload names |
| Held-out evaluator feature join | passed v5 | confirmatory | results/revision_v1/analysis_inputs/primary_heldout_join_v5/heldout_feature_join_manifest.json | 13,320 joined rows, 6,480 primary trials, 18/18 inputs valid, current algorithm-source provenance |
| Detector exact split leakage | 28/28 exact checks passed | diagnostic | results/revision_v1/final_experiment_package/detectors/leakage_audit/detector_leakage_audit_manifest.json | Payload, row, text hash, source trial, and pair identities are disjoint; separate lexical near-duplicate diagnostic is adverse |
| Same-device CUDA repeatability | passed for both detector architectures | diagnostic | results/revision_v1/final_experiment_package/overhead/detector_cuda_benchmark_manifest.json | Task design, prediction order, labels, metrics, scores, predictions, and model-state hashes were exact; CPU/GPU equivalence not tested |
| CPU neural feasibility evidence | preserved and excluded from detector results | diagnostic | results/revision_v1/supervisor/detector_live_readonly_audit_20260813T0145Z.json; results/revision_v1/supervisor/detector_live_readonly_audit_addendum_20260813T0151Z.json | No CPU neural training was resumed; CPU/GPU equivalence is not claimed |
| Human participant data | 0 participants; 0 ratings | external_gate | results/revision_v1/final_experiment_package/readability/human_evaluation_status.json | No human data collected or fabricated |
| Preservation and disk state | No deletion, move, cleanup, or quarantine performed | diagnostic | results/revision_v1/supervisor/detector_live_readonly_audit_addendum_20260813T0151Z.json | Evidence, diagnostics, checkpoints, caches, and user changes were preserved in place |
| Atomic final detector publication | 56/56 fits, 55,440 prediction rows, 0 failures, 0 retries | confirmatory | results/revision_v1/neural_detector/confirmatory_v2/detector_run_manifest.json; results/revision_v1/final_experiment_package/detectors/detector_reference_manifest.json | All fit checkpoints, final candidate, terminal receipt, complete status, GPU-ledger incorporation marker, products, and two required same-CUDA reports were hash-validated |
| Corrected locked mixed-model rerun | passed: 325 coefficients, 225 contrasts, 24 Wilson rows, 7 statuses, and 6 diagnostics | confirmatory | results/revision_v1/final_experiment_package/statistics/mixed_models/mixed_model_run_manifest.json; results/revision_v1/final_experiment_package/statistics/mixed_models_prefinal_v3_correction_audit.json | Final R source hash remained 6ff857647ef9c594a823a4786af1b441e0ff7274ed97f3201a7384e005dde0f9; no fixed-effects fallback; prefinal defective output remains excluded |
| Cumulative detector GPU budget | 75.406380 hours used; 89.593620 hours remaining below 165 hours | diagnostic | results/revision_v1/final_progress_snapshot_v1.json; results/revision_v1/detector_cuda_reproducibility_v2/gpu_accounting_ledger.json | The 62.478384-hour external historical floor was reconciled before adding 12.927996 measured detector GPU-hours and was never reset to zero |
| Scoped computational test suite | 597 passed; 0 failures; 0 errors; 1 prohibited CPU-neural test deselected | confirmatory | results/revision_v1/final_experiment_package/statistics/test_suite/test_suite_manifest.json; results/revision_v1/final_experiment_package/statistics/test_suite/junit.xml | The 629-test collection reconciles to 597 executed computational tests, 31 tests in three explicitly out-of-scope manuscript, bibliography, and release modules, and 1 CPU-neural training test prohibited by the frozen execution decision |
| Final payload-grouped Python statistical regeneration | passed in 3,519.071 wall-seconds; 3,696 recovery, 78,088 continuous, 8,446 effect-size, 29,520 quality, and 56 detector summary rows | confirmatory | results/revision_v1/final_experiment_package/statistics/python/statistics_run_manifest.json; results/revision_v1/final_experiment_package/statistics/python/statistics_integrity_report.json; results/revision_v1/final_experiment_package/statistics/python/statistics_execution_audit.json | All eight declared outputs hash-validated; 2,000 payload bootstraps were used; nested segments were collapsed; no numeric infinities or duplicate full output rows were observed; all 56 detector core point-metric rows exactly matched the saved fit metrics |
| Excluded prefinal Python statistics attempt | failed closed after approximately 3,051 seconds with no published output files | diagnostic | results/revision_v1/final_experiment_package/statistics/statistics_predetector_failure_20260816T0950Z.json | The prefinal protocol-stratification guard failure was corrected without changing the scientific design; the final source-bound run is AUD014 |
| Excluded detector benchmark packaging attempt | 2 failed post-fit packaging invocations; 111.000125 GPU-seconds retained in accounting | diagnostic | results/revision_v1/detector_cuda_reproducibility_v2/failed_attempts/task_0_packaging_failure.json | The checkpoint was not reused after the source fix; a fresh source-bound namespace was used, and this diagnostic is not a published benchmark or production fit failure |
| Final GPU, disk, process, lock, and temporary-diagnostic state | 0 relevant compute processes, 0 GPU compute applications, 9 preserved lock files with 0 active owners, and 53,935,910,912 disk bytes available | diagnostic | results/revision_v1/final_experiment_package/resource_and_process_audit.json | All 75 RankCloak-named temporary roots (466,367,788 regular-file bytes) were retained for reference; no deletion, move, cleanup, or quarantine was performed |
| Excluded prefinal evidence-summary packaging attempt | failed closed at a heterogeneous reference-table schema guard; corrected and regenerated with 14/14 outputs | diagnostic | results/revision_v1/final_experiment_package/evidence_summary_prefinal_failure_20260816T1736Z.json | The failure occurred after some atomic per-file writes but before a completed manifest; all declared outputs were overwritten by the validated rerun, and no scientific result was affected |
| Excluded prefinal package-index invocation | failed closed before publication because the output manifest was incorrectly required as a preexisting input | diagnostic | results/revision_v1/final_experiment_package/package_index_prefinal_failure_20260816T1746Z.json | No package index was written; the corrected invocation removed only the self-referential prerequisite and retained all component, external-reference, and required-file validations |

## GPU accounting

Historical floor: 62.4783840698 GPU-hours; new non-overlapping detector interval union: 12.9279961008; cumulative: 75.4063801706; remaining below the 165-hour ceiling: 89.5936198294.

## Unresolved or external items

| Item | Status | Evidence |
| --- | --- | --- |
| Human evaluation uncollected | external_gate | 0 participant rows, 0 rating rows, survey not deployed, recruitment not authorized |
| Real-world deployment untested | unresolved | Controlled offline experiment only; no deployed message or application sample |
| Incomplete requested perturbation coverage | unresolved | 3/11 requested classes untested and 3/11 only partially represented |
| Fresh generation replay unavailable in this checkout | unavailable | All three configured GGUF paths are absent; existing outputs remain manifest- and hash-bound |
| Frozen unavailable work units | unavailable | 48 ablation, 336 robustness, and 48 held-out evaluator work units are explicitly unavailable |
| Overhead measurement gaps | unavailable | CPU time and warm-up repeats unavailable; RAM/VRAM and wrapper timing scopes explicitly limited |
| Full recovery proposition not directly trace-evaluable | unavailable | Cascade-evaluable saved trace count 0 |
| Automated readability is not human naturalness | unresolved | All automated metric rows are labeled human_rating_substitute=false |
| Multilingual evidence limited to two secondary languages | unresolved | Spanish and Simplified Chinese exact-copy experiments only; English heuristics not transferred |
| Draft human instrument and model-identity gate | external_gate | The proposed instrument is DRAFT_NOT_IRB_APPROVED; manual safety review, participant scheduling, recruitment, and survey deployment are incomplete; the draft Llama 3.1 label does not match the executed Llama 3 8B identity |
