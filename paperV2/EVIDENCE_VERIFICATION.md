# Evidence verification audit

Audit date: 2026-08-16. Repository revision: `cfa19d3d41af8f869837c087ec6f33fcd9f51901`.

The V2 manuscript was written from retained, sealed artifacts only. No LLM generation, detector fitting, mixed model, bootstrap, neural training, or experimental pipeline was rerun. Where a narrative summary and an authoritative CSV differed in rounding, the CSV controlled.

## Principal numerical checks

| Manuscript topic | Authoritative artifact | Classification | Values verified for V2 |
|---|---|---|---|
| Payload corpus | `release_inputs/revision_v1/PAYLOAD_MANIFEST.json` | Confirmatory | 480 validated public deterministic artifacts; eight classes; 60 per class; zero validation errors |
| Model/prompt design and full ledger | `results/revision_v1/final_progress_snapshot_v1.json` plus the sealed model/prompt manifests | Confirmatory / diagnostic | Three model families; 18 English templates in six categories; 14,400 primary units; 38,504/38,504 reconciled, 38,072 success, 432 unavailable, zero failure, zero remaining |
| Primary recovery | `results/revision_v1/final_experiment_package/tables/primary_recovery/primary_recovery_endpoints.csv` | Confirmatory / supporting sensitivity | 6,480/6,480, Wilson 95% CI [0.999408, 1.000000]; 480/480 payload groups, CI [0.992061, 1.000000]; 1,440/1,440 model--payload groups, CI [0.997339, 1.000000] |
| Multilingual recovery | `results/revision_v1/final_experiment_package/tables/multilingual_recovery/multilingual_recovery_endpoints.csv` | Secondary | Spanish 288/288 and Simplified Chinese 288/288; 576/576 combined; per-language trial CI [0.986837, 1.000000] |
| Replay and transformations | `results/revision_v1/final_experiment_package/robustness/recovery_by_condition.csv` | Secondary | Saved IDs 144/144; visible-text retokenization 88/144; raw unchanged and declared tail-only truncation 144/144; NFKC 82/144; quote conversion 56/144; trim 28/144; markdown and paraphrase 0/144; cross-model 0/288 |
| Perturbation coverage | `results/revision_v1/final_experiment_package/robustness/perturbation_coverage_inventory.csv` | Diagnostic | Five requested classes direct, three partial, three untested; first-divergence labels treated as descriptive rather than causal |
| Automated readability | `results/revision_v1/final_experiment_package/readability/selected_stimulus_readability_summary.csv` | Supporting | 504 balanced stimuli; ordinary-control Flesch 61.940597, CI [54.897042, 68.143167]; segmented-full 60.551834, CI [56.873661, 64.135677] |
| Human-study status | `results/revision_v1/final_experiment_package/readability/human_evaluation_status.json` | External gate | Recruitment unauthorized; survey undeployed; participants 0; ratings 0 |
| Model-based quality and mixed models | `results/revision_v1/final_experiment_package/statistics/mixed_models/mixed_model_contrasts.csv` and `mixed_model_run_manifest.json` | Confirmatory with diagnostics | 42/45 Holm-significant contrasts for each quality endpoint, sign-dependent ranges; five estimable models; three singular; no fixed-effects fallback; recovery GLMM not estimable under complete separation |
| Ablations | `results/revision_v1/final_experiment_package/statistics/ablation_analysis/ablation_canonical_contrasts.csv` | Exploratory / post-outcome | All 60 rows transferred to Table S10; lead-in, segment, filter, and tail estimates and intervals checked; 48 Mistral roundtrip-stable-filter units unavailable |
| Topic contrasts | `results/revision_v1/final_experiment_package/statistics/topic_effects/topic_schedule_contrasts.csv` | Exploratory / post-outcome | All 15 rows transferred to Table S11; Llama artifact-count ratio 1.276361, CI [1.114743, 1.461409], Holm p=0.003294; other adjusted p-values at least 0.396248 |
| Capacity and cover overhead | `results/revision_v1/final_experiment_package/theory/theory_empirical_summary.csv` and `theory/TECHNICAL_NOTE.md` | Supporting / diagnostic | 17,424 records; 7,008 fully evaluable; zero forced-position, rate-bound, or supported quality-bound violations; 2,976 positive cover-length residuals; zero complete cascade traces |
| Neural detector core | `results/revision_v1/final_experiment_package/detectors/analysis/detector_extended_metrics.csv` | Confirmatory | 15,840 corpus rows; 56/56 fits; 55,440 predictions; matched TextCNN and DeBERTa ROC-AUC/balanced-accuracy values and grouped intervals; held-out ranges |
| Detector leakage and sensitivity | `results/revision_v1/final_experiment_package/detectors/leakage_audit/detector_leakage_audit_manifest.json` and `detectors/leakage_sensitivity/detector_leakage_sensitivity_manifest.json` | Diagnostic / exploratory | Exact identity checks passed; 53 lexical near-duplicate pairs, 44 cross-payload, 16 affected splits; reported restricted-minus-original metric ranges checked |
| Generation overhead | `results/revision_v1/final_experiment_package/overhead/overhead_summary.csv` | Supporting | 12,144 matched primary runtime rows; mean generation 1.674368--16.087628 s; throughput 13.631855--100.851772 bit/s; effective rate 0.710004--7.322003 bits/full token |
| Detector benchmark | `results/revision_v1/final_experiment_package/overhead/detector_cuda_benchmark_summary.csv` | Diagnostic | TextCNN 91.358113/9.497936 s and 306,247,680 bytes; DeBERTa 1,536.660910/1,425.063414 s and 6,380,851,200 bytes; same-device fixed-seed repeatability only |
| Worked examples | `results/revision_v1/primary_v2/llama3_8b_instruct_q4_k_m/records.jsonl` and `results/revision_v1/final_experiment_package/robustness/failure_taxonomy.csv` | Illustrative / diagnostic | SHA-256 test vector, first eight ranks, first-segment token IDs, semantic tail, exact saved-token recovery, and the reported retokenization and quote-conversion first divergences all match retained records |

## Source-resolution notes

- The canonical ablation CSV reports the selected no-filter mean-log-probability contrast as `-0.001797` with CI `[-0.037738, 0.035394]`. A rounded narrative summary reports `-0.001802` with slightly different last digits. V2 uses the canonical CSV value and labels it approximately `-0.0018` in prose.
- The complete ledger counts 3,744 robustness work units (3,408 successful and 336 unavailable). The robustness analysis manifest's 3,600-unit perturbation subset excludes the separate 144 saved-ID replay records. V2 uses the ledger total for work-unit reconciliation and condition-specific denominators for recovery results; it does not pool these scopes.
- The overhead package contains 21,192 runtime rows across scopes. V2's primary generation summaries use the authoritative 12,144 matched primary trial-runtime rows and identify that denominator explicitly.
- Unavailable combinations are retained as unavailable and excluded from estimands. They are never called execution or recovery failures.

## Machine-readable supplementary data integrity

`paperV2/scientific_reports/supplementary_tables/detector_split_metrics.csv` is a byte-for-byte copy of the authoritative detector extended-metrics table. Both files have SHA-256:

`256307b2951e46cf3d7ad4263dc3c9bf6490749e5004f92957811e349477598a`

The full ablation, topic, and overhead LaTeX row fragments were checked against their authoritative CSV tables and are included by the Supplement at compile time. Figure integrity is documented independently in `FIGURE_PROVENANCE.csv`.
