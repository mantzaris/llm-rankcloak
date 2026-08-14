import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256

from rankcloak.revision_reporting import (
    MAIN_DISPLAYS,
    ReportArtifactConflict,
    RevisionReportingError,
    build_revision_reports,
    display_registry,
    latex_escape,
    load_verified_sources,
    _validate_detector_preprocessing_binding,
    verify_report_output_manifest,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _declared(path, row_count=None):
    value = {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}
    if row_count is not None:
        value["row_count"] = row_count
    return value



def _mutate_statistics_csv(manifest_path, role, mutation):
    manifest = json.loads(manifest_path.read_text())
    artifact_path = manifest_path.parent / manifest["outputs"][role]["path"]
    with artifact_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mutation(rows)
    _write_csv(artifact_path, rows)
    manifest["outputs"][role] = _declared(artifact_path)
    _write_json(manifest_path, manifest)


def _statistics_fixture(root, *, smoke=True, bad_rate=False, bad_continuous_n=False):
    directory = root / "statistics"
    recovery_rows = [
        {
            "phase": "confirmatory",
            "evidence_status": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
            "study_phase": "primary_v2_confirmatory",
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "model_id": "model_a",
            "protocol_variant": "nonseg_ascii_b8",
            "prompt_category": "explanatory",
            "payload_class": "sha256",
            "replay_mode": "saved_token_ids",
            "transformation_id": "unmodified",
            "mitigation_id": "none",
            "analysis_unit": "payload",
            "n_payloads": 10,
            "recovery_outcome": "exact_payload_recovery",
            "recovery_outcome_semantics": "original_serialized_payload_bytes_sha256_v1",
            "exact_recovery_compatibility_alias": True,
            "payload_recovery_successes": 8,
            "successes": 8,
            "exact_payload_recovery_rate": 0.7 if bad_rate else 0.8,
            "exact_recovery_rate": 0.7 if bad_rate else 0.8,
            "rank_replay_n": 10,
            "rank_replay_successes": 10,
            "exact_rank_replay_rate": 1.0,
            "rank_replay_diagnostic_only": True,
            "wilson_ci_low": 0.49,
            "wilson_ci_high": 0.94,
            "confidence_level": 0.95,
        },
        {
            "phase": "confirmatory",
            "evidence_status": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
            "study_phase": "primary_v2_confirmatory",
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "model_id": "model_a",
            "protocol_variant": "nonseg_ascii_b16",
            "prompt_category": "explanatory",
            "payload_class": "sha256",
            "replay_mode": "saved_token_ids",
            "transformation_id": "unmodified",
            "mitigation_id": "none",
            "analysis_unit": "payload",
            "n_payloads": 10,
            "recovery_outcome": "exact_payload_recovery",
            "recovery_outcome_semantics": "original_serialized_payload_bytes_sha256_v1",
            "exact_recovery_compatibility_alias": True,
            "payload_recovery_successes": 9,
            "successes": 9,
            "exact_payload_recovery_rate": 0.9,
            "exact_recovery_rate": 0.9,
            "rank_replay_n": 10,
            "rank_replay_successes": 10,
            "exact_rank_replay_rate": 1.0,
            "rank_replay_diagnostic_only": True,
            "wilson_ci_low": 0.59,
            "wilson_ci_high": 0.99,
            "confidence_level": 0.95,
        },
    ]
    continuous_rows = [
        {
            "phase": "confirmatory",
            "evidence_status": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
            "study_phase": "primary_v2_confirmatory",
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "model_id": "model_a",
            "protocol_variant": "nonseg_ascii_b8",
            "prompt_category": "explanatory",
            "payload_class": "sha256",
            "replay_mode": "saved_token_ids",
            "transformation_id": "unmodified",
            "mitigation_id": "none",
            "outcome": "effective_payload_rate",
            "analysis_unit": "payload",
            "mean": 1.25,
            "n_payloads": 11 if bad_continuous_n else 10,
            "ci_low": 1.1,
            "ci_high": 1.4,
        },
        {
            "model_id": "model_a",
            "protocol_variant": "nonseg_ascii_b16",
            "payload_class": "sha256",
            "outcome": "encoding_seconds",
            "analysis_unit": "payload",
            "mean": 0.22,
            "n_payloads": 10,
            "ci_low": 0.20,
            "ci_high": 0.24,
            "hardware_id": "gpu_a",
        },
        {
            "model_id": "model_a",
            "protocol_variant": "nonseg_ascii_b16",
            "payload_class": "sha256",
            "outcome": "human_naturalness",
            "analysis_unit": "payload",
            "mean": 4.1,
            "n_payloads": 10,
            "ci_low": 3.8,
            "ci_high": 4.4,
            "text_view": "full_message",
        },
    ]
    effect_rows = [
        {
            "outcome": "exact_recovery",
            "recovery_outcome": "exact_payload_recovery",
            "recovery_outcome_semantics": "original_serialized_payload_bytes_sha256_v1",
            "exact_recovery_compatibility_alias": True,
            "exact_rank_replay_diagnostic_only": True,
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "evidence_status_scope": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
            "study_phase_scope": "primary_v2_confirmatory",
            "protocol_contract_revision_scope": "payload_fidelity_v2",
            "result_schema_revision_scope": "payload_aware_result_v2",
            "primary_inference": False,
            "inference_role": "descriptive_exploratory_pairwise",
            "inferential_p_value_supported": True,
            "comparison_design": "paired_payload",
            "factor": "protocol_variant",
            "level_first": "nonseg_ascii_b8",
            "level_second": "nonseg_ascii_b16",
            "analysis_unit": "payload",
            "n_payloads_first": 10,
            "n_payloads_second": 10,
            "n_payloads_paired": 10,
            "mean_difference_ci_low": -0.4,
            "mean_difference_ci_high": 0.2,
            "risk_difference": -0.1,
            "p_value_holm": 0.5,
        }
    ]
    paths = {}
    for name, rows in (
        ("recovery", recovery_rows),
        ("continuous", continuous_rows),
        ("effects", effect_rows),
        ("mixed", []),
        ("detectors", []),
    ):
        path = directory / {
            "recovery": "recovery_summary.csv",
            "continuous": "continuous_summary.csv",
            "effects": "effect_sizes.csv",
            "mixed": "mixed_effects_coefficients.csv",
            "detectors": "detector_summary.csv",
        }[name]
        _write_csv(path, rows)
        paths[name] = path
    integrity = {
        "status": "passed",
        "analysis_unit": "payload",
        "segments_as_independent_observations": False,
        "smoke_fixture": smoke,
        "payload_fidelity_contract": {
            "contract_version": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "semantics": "original_serialized_payload_bytes_sha256_v1",
            "primary_outcome": "exact_payload_recovery",
            "compatibility_alias": "exact_recovery",
            "alias_equality_validated": True,
            "exact_rank_replay_role": "diagnostic_only",
            "direct_rows": 0,
            "direct_rows_contract_verified": 0,
        },
        "primary_effect_scope": {
            "replay_mode": "saved_token_ids",
            "transformation_id": "unmodified",
            "mitigation_id": "none",
            "evidence_status": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
            "study_phase": "primary_v2_confirmatory",
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "diagnostic_replay_fallback": False,
            "pairwise_effects_are_primary_inference": False,
        },
        "independent_payloads": {"trials": 20, "features": 20, "runtime": 20},
    }
    integrity_path = directory / "statistics_integrity_report.json"
    _write_json(integrity_path, integrity)
    paths["integrity"] = integrity_path
    manifest = {
        "schema_version": "1.0",
        "inputs": [],
        "outputs": {name: _declared(path) for name, path in paths.items()},
    }
    manifest_path = directory / "statistics_run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _theory_fixture(root):
    directory = root / "theory"
    table_rows = {
        "capacity_validation": [
            {"trial_id": "t1", "model_id": "model_a", "protocol_variant": "nonseg_ascii_b8", "payload_name": "p1"}
        ],
        "capacity_plot": [
            {
                "trial_id": "t1",
                "model_id": "model_a",
                "protocol_variant": "nonseg_ascii_b8",
                "payload_name": "p1",
                "alphabet_size_B": 8,
                "R_B_bits_per_forced_token": 3.0,
                "R_effective_bits_per_forced_plus_tail_token": 2.0,
            }
        ],
        "quality_validation": [
            {"trial_id": "t1", "model_id": "model_a", "protocol_variant": "nonseg_ascii_b8", "payload_name": "p1"}
        ],
        "quality_plot": [
            {
                "trial_id": "t1",
                "model_id": "model_a",
                "protocol_variant": "nonseg_ascii_b8",
                "payload_name": "p1",
                "alphabet_size_B": 8,
                "Q_B_nats_per_forced_token": 1.2,
            }
        ],
        "exact_recovery": [{"trial_id": "t1", "proposition_confirmed": True}],
        "cascade": [],
    }
    tables = []
    for name, rows in table_rows.items():
        path = directory / (name + ".csv")
        _write_csv(path, rows)
        tables.append({"name": name, "path": path.name, "row_count": len(rows), "sha256": _sha(path)})
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "rankcloak_capacity_quality_theory_validation",
        "tables": tables,
        "summary": {
            "input_record_count": 1,
            "capacity_evaluable_count": 1,
            "quality_evaluable_count": 1,
            "quality_fully_bound_validated_count": 1,
            "exact_proposition_confirmed_count": 1,
            "exact_observed_only_count": 0,
            "cascade_evaluable_count": 0,
        },
    }
    path = directory / "theory_validation_manifest.json"
    _write_json(path, manifest)
    return path


def _detector_fixture(root, *, smoke=True, wrong_count=False, schema_v2=False):
    directory = root / "detector"
    metrics = [
        {
            "split_id": "matched",
            "regime": "matched",
            "detector_name": "textcnn",
            "test_rows": 4,
            "test_payload_groups": 4,
            "roc_auc": 0.75,
            "pr_auc": 0.8,
            "balanced_accuracy": 0.75,
        }
    ]
    predictions = [
        {"split_id": "matched", "detector_name": "textcnn", "payload_group_id": "p{}".format(i), "label": i % 2, "score": (0.8 if i % 2 else 0.2)}
        for i in range(4)
    ]
    dataset = [
        {"row_id": "r{}".format(i), "text_sha256": "a" * 64, "label": i % 2, "payload_group_id": "p{}".format(i), "prompt_template_id": "q", "model_id": "m", "codec_id": "c"}
        for i in range(4)
    ]
    _write_csv(directory / "detector_metrics.csv", metrics)
    _write_csv(directory / "detector_predictions.csv", predictions)
    _write_csv(directory / "detector_dataset_manifest.csv", dataset)
    _write_json(directory / "detector_split_manifest.json", {"splits": []})
    _write_json(directory / "detector_failures.json", [])
    manifest = {
        "schema_version": (
            "rankcloak-revision-detector-run-v2"
            if schema_v2
            else "rankcloak-revision-detector-run-v1"
        ),
        "smoke": smoke,
        "metric_rows": 2 if wrong_count else 1,
        "prediction_rows": 4,
        "normalized_rows": 4,
    }
    if schema_v2:
        manifest.update(
            {
                "execution_mode": "smoke" if smoke else "confirmatory",
                "output_files": {
                    filename: {
                        "sha256": _sha(directory / filename),
                        "size_bytes": (directory / filename).stat().st_size,
                    }
                    for filename in (
                        "detector_metrics.csv",
                        "detector_predictions.csv",
                        "detector_dataset_manifest.csv",
                        "detector_split_manifest.json",
                        "detector_failures.json",
                    )
                },
            }
        )
    path = directory / "detector_run_manifest.json"
    _write_json(path, manifest)
    return path


def _runtime_fixture(root):
    directory = root / "runtime"
    rows = [
        {
            "model_id": "model_b",
            "protocol_variant": "nonseg_ascii_b16",
            "outcome": "payload_bits_per_second",
            "mean": 40.0,
            "ci_low": 35.0,
            "ci_high": 45.0,
            "n_payloads": 8,
            "hardware_id": "gpu_b",
        }
    ]
    output = directory / "runtime_profile_summary.csv"
    _write_csv(output, rows)
    manifest = {
        "schema_version": "rankcloak-runtime-summary-v1",
        "outputs": {"runtime_profile": _declared(output, 1)},
    }
    path = directory / "runtime_report_manifest.json"
    _write_json(path, manifest)
    return path


def _mixed_model_fixture(root):
    project_root = Path(__file__).resolve().parents[1]
    directory = root / "mixed_model"
    directory.mkdir(parents=True, exist_ok=True)
    trials = directory / "trials.csv"
    features = directory / "primary_features_with_heldout_evaluator.csv"
    runtime = directory / "runtime.csv"
    _write_csv(trials, [{"trial_id": "fixture"}])
    _write_csv(features, [{"trial_id": "fixture", "heldout_evaluator_log_probability": -3.0}])
    _write_csv(runtime, [{"trial_id": "fixture"}])
    models_config = project_root / "configs" / "revision_v1" / "models.json"
    model_pins = {
        str(row["model_id"]): str(row["artifact_sha256"])
        for row in json.loads(models_config.read_text(encoding="utf-8"))["models"]
    }
    join_manifest = {
        "schema_version": "rankcloak-revision-heldout-feature-join-v1",
        "manifest_type": "rankcloak_revision_primary_heldout_feature_join",
        "primary_trial_count": 6480,
        "unmatched_primary_trials": 0,
        "source_record_hashes_recomputed": True,
        "evaluator_source_records_byte_identical_to_preprocessing": True,
        "evaluator_artifact_pins_verified": True,
        "models_config_sha256": _sha(models_config),
        "evaluator_artifact_pins": model_pins,
        "outputs": {
            "features": {
                "path": features.name,
                "sha256": _sha(features),
                "size_bytes": features.stat().st_size,
                "row_count": 1,
            }
        },
    }
    join_manifest_path = directory / "heldout_feature_join_manifest.json"
    _write_json(join_manifest_path, join_manifest)

    scientific_model_ids = (
        "primary_exact_recovery",
        "primary_artifact_counts",
        "primary_effective_artifact_rate",
        "primary_cover_log_probability",
        "primary_heldout_evaluator_log_probability",
        "primary_payload_throughput",
    )
    coefficient_rows = [
        {
            "model_id": model_id,
            "backend": "R_lme4",
            "family": "gaussian",
            "formula": "heldout_evaluator_log_probability ~ model_id + (1 | payload_name)",
            "term": "(Intercept)",
            "estimate": -3.0,
            "standard_error": 0.1,
            "statistic": -30.0,
            "p_value_raw": 0.001,
            "ci_low": -3.2,
            "ci_high": -2.8,
            "fixed_effects_fallback": False,
        }
        for model_id in scientific_model_ids
    ]
    contrast_families = {
        "primary_exact_recovery": (
            "recovery_protocol_within_model",
            "recovery_model_within_protocol",
            "recovery_prompt_category",
        ),
        "primary_artifact_counts": ("artifact_protocol_within_model",),
        "primary_effective_artifact_rate": ("continuous_protocol_within_model",),
        "primary_cover_log_probability": ("continuous_protocol_within_model",),
        "primary_heldout_evaluator_log_probability": (
            "continuous_protocol_within_model",
        ),
        "primary_payload_throughput": ("continuous_protocol_within_model",),
    }
    contrast_rows = [
        {
            "model_id": model_id,
            "multiplicity_family": family,
            "contrast": "nonseg_ascii_b8 - nonseg_ascii_b16",
            "protocol_variant": "nonseg_ascii_b8",
            "estimate": -0.2,
            "standard_error": 0.05,
            "statistic": -4.0,
            "p_value_raw": 0.002,
            "p_value_holm": 0.01,
            "ci_low": -0.3,
            "ci_high": -0.1,
            "adjustment": "holm",
            "scale": "model",
            "fixed_effects_fallback": False,
        }
        for model_id, families in contrast_families.items()
        for family in families
    ]
    statuses = []
    for model_id in scientific_model_ids:
        statuses.append(
            {
                "model_id": model_id,
                "status": "completed",
                "coefficient_rows": 1,
                "fixed_effects_fallback": False,
            }
        )
    statuses.append(
        {
            "model_id": "human_naturalness_and_suspiciousness_clmm",
            "status": "external_until_irb_approved_ratings_exist",
            "fixed_effects_fallback": False,
        }
    )
    output_rows = {
        "coefficients": ("mixed_model_coefficients.csv", coefficient_rows),
        "contrasts": ("mixed_model_contrasts.csv", contrast_rows),
        "wilson": ("recovery_wilson_sensitivity.csv", []),
        "dispersion": ("poisson_dispersion_check.csv", []),
    }
    outputs = {}
    for role, (filename, rows) in output_rows.items():
        path = directory / filename
        _write_csv(path, rows)
        outputs[role] = _declared(path, len(rows))
    diagnostics = directory / "mixed_model_diagnostics.json"
    status_path = directory / "model_status.json"
    _write_json(diagnostics, [{"fixed_effects_fallback": False}])
    _write_json(status_path, statuses)
    outputs["diagnostics"] = _declared(diagnostics)
    outputs["status"] = _declared(status_path)

    plan = project_root / "analysis" / "revision_v1" / "confirmatory_model_plan.json"
    lock = project_root / "analysis" / "revision_v1" / "r_environment.lock.json"
    driver = project_root / "scripts" / "run_revision_mixed_models.R"

    def input_declaration(role, path):
        return {
            "role": role,
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": "1.0",
        "manifest_type": "rankcloak_revision_v1_mixed_model_run",
        "plan_id": "rankcloak_revision_primary_v2_prespecified_confirmatory_models",
        "plan_sha256": _sha(plan),
        "environment_lock_sha256": _sha(lock),
        "validation_only": False,
        "analysis_unit": "payload_trial",
        "segments_as_independent_observations": False,
        "fixed_effects_fallback": False,
        "payload_fidelity_contract": {
            "contract_version": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
            "semantics": "original_serialized_payload_bytes_sha256_v1",
            "primary_outcome": "exact_payload_recovery",
            "compatibility_alias": "exact_recovery",
            "alias_equality_validated": True,
            "exact_rank_replay_role": "diagnostic_only",
            "direct_rows": 0,
            "direct_rows_contract_verified": 0,
        },
        "input_files": [
            input_declaration("driver_source", driver),
            input_declaration("plan", plan),
            input_declaration("environment_lock", lock),
            input_declaration("trials", trials),
            input_declaration("features", features),
            input_declaration("feature_join_manifest", join_manifest_path),
            input_declaration("runtime", runtime),
        ],
        "outputs": outputs,
    }
    path = directory / "mixed_model_run_manifest.json"
    _write_json(path, manifest)
    return path


def _evaluator_unavailability_fixture(root):
    directory = root / "evaluator_unavailability"
    directory.mkdir(parents=True, exist_ok=True)
    plan_rows = []
    record_rows = []
    units = []
    identifiers = []
    for index in range(48):
        work_id = "unavailable-{:02d}".format(index)
        identifiers.append(work_id)
        protocol = "nonseg_ascii_b16"
        payload = "payload-{:02d}".format(index)
        task = {
            "work_id": work_id,
            "work_kind": "rankcloak",
            "protocol_variant": protocol,
            "payload_name": payload,
        }
        record = {
            "work_id": work_id,
            "record_type": "condition_unavailable",
            "execution_status": "completed",
            "reason_code": "empty_isolated_roundtrip_vocabulary",
            "protocol_contract_revision": "payload_fidelity_v2",
            "result_schema_revision": "payload_aware_result_v2",
        }
        plan_rows.append(task)
        record_rows.append(record)
        units.append(
            {
                "terminal_status": "upstream_dependent_unavailable_not_scored",
                "source_stage": "ablation_v2",
                "source_work_id": work_id,
                "source_record_type": "condition_unavailable",
                "source_record_sha256": canonical_json_sha256(record),
                "reason_code": "empty_isolated_roundtrip_vocabulary",
                "generator_model_id": "mistral_7b_instruct_v0_3_q4_k_m",
                "evaluator_model_id": "llama3_8b_instruct_q4_k_m",
                "protocol_variant": protocol,
                "payload_name": payload,
                "scoring_attempted": False,
                "score_imputed": False,
            }
        )
    plan = directory / "plan.jsonl"
    records = directory / "records.jsonl"
    checkpoint = directory / "checkpoint.json"
    run_identity = directory / "run_identity.json"
    _write_jsonl(plan, plan_rows)
    _write_jsonl(records, record_rows)
    _write_json(checkpoint, {"completed_trial_ids": identifiers})
    _write_json(run_identity, {"stage": "ablation_v2", "fixture": True})
    source_files = [
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for role, path in (
            ("plan", plan),
            ("checkpoint", checkpoint),
            ("records", records),
            ("run_identity", run_identity),
        )
    ]
    manifest = {
        "schema_version": "rankcloak-heldout-evaluator-upstream-unavailability-v1",
        "manifest_type": "heldout_evaluator_upstream_dependent_unavailability",
        "protocol_contract_revision": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "authorized_projection_sha256": (
            "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
        ),
        "frozen_evaluator_target_units": 17280,
        "scoreable_evaluator_units": 17232,
        "upstream_dependent_unavailable_units": 48,
        "terminal_accounted_units": 17280,
        "scoring_attempted_for_unavailable_units": False,
        "scores_imputed_or_fabricated": False,
        "analysis_policy": (
            "terminal_design_units_excluded_from_quality_estimands_and_not_scored"
        ),
        "source_files": source_files,
        "source_files_sha256": canonical_json_sha256(source_files),
        "units": units,
        "units_sha256": canonical_json_sha256(units),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    path = directory / "upstream_dependent_unavailability_v1.json"
    _write_json(path, manifest)
    return path


def _detector_preprocessing_binding_fixture(root):
    directory = root / "detector_preprocessing_binding"
    directory.mkdir(parents=True, exist_ok=True)
    detector = directory / "detector_corpus.jsonl"
    detector.write_text("{}\n", encoding="utf-8")
    model_ids = [
        "llama3_8b_instruct_q4_k_m",
        "mistral_7b_instruct_v0_3_q4_k_m",
        "qwen2_5_7b_instruct_q4_k_m",
    ]
    input_manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_inputs",
        "strict_complete": True,
        "emitted_run_count": 3,
        "reference_run_count": 0,
        "input_files": [],
        "input_files_sha256": canonical_json_sha256([]),
        "run_shards": [
            {
                "role": "input",
                "stage": "primary_v2",
                "model_id": model_id,
                "evidence_status": (
                    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
                ),
                "planned_work_units": 4800,
                "completed_work_units": 4800,
            }
            for model_id in model_ids
        ],
    }
    input_path = directory / "preprocessing_input_manifest.json"
    _write_json(input_path, input_manifest)
    outputs = [
        {
            "role": "detector",
            "path": detector.name,
            "size_bytes": detector.stat().st_size,
            "sha256": _sha(detector),
            "row_count": 15840,
        },
        {
            "role": "input_manifest",
            "path": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": _sha(input_path),
            "row_count": None,
        },
    ]
    preprocessing = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_outputs",
        "input_manifest_sha256": _sha(input_path),
        "outputs": outputs,
        "outputs_sha256": canonical_json_sha256(outputs),
        "row_counts": {"detector": 15840},
        "invariants": {
            "detector_pair_count": 7920,
            "detector_grouping_unit": "payload_name",
        },
    }
    preprocessing_path = directory / "preprocessing_output_manifest.json"
    _write_json(preprocessing_path, preprocessing)
    binding = {
        "schema_version": (
            "rankcloak-revision-primary-detector-preprocessing-binding-v1"
        ),
        "preprocessing_manifest_path": str(preprocessing_path.resolve()),
        "preprocessing_manifest_sha256": _sha(preprocessing_path),
        "preprocessing_input_manifest_path": str(input_path.resolve()),
        "preprocessing_input_manifest_sha256": _sha(input_path),
        "detector_path": str(detector.resolve()),
        "detector_sha256": _sha(detector),
        "detector_size_bytes": detector.stat().st_size,
        "detector_row_count": 15840,
        "strict_complete": True,
        "primary_shard_count": 3,
        "primary_model_ids": model_ids,
    }
    detector_run = {
        "input_path": str(detector.resolve()),
        "input_sha256": _sha(detector),
        "preprocessing_manifest_path": str(preprocessing_path.resolve()),
        "preprocessing_manifest_sha256": _sha(preprocessing_path),
    }
    dataset_contract = {"model_ids": model_ids, "preprocessing_binding": binding}
    return detector_run, dataset_contract, detector


def _preprocessing_fixture(root):
    directory = root / "preprocessing"
    rows = [
        {
            "evidence_status": "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze",
            "record_type": "condition_unavailable",
            "model_id": "mistral_7b_instruct_v0_3_q4_k_m",
            "token_filter": "roundtrip_stable_filter_v1",
            "ablation_factor": "token_filter",
            "ablation_level": "roundtrip_stable_filter_v1",
            "reason_code": "empty_isolated_roundtrip_vocabulary",
            "root_condition_model_id": "mistral_7b_instruct_v0_3_q4_k_m",
            "excluded_from_estimands": True,
        }
        for _ in range(48)
    ]
    unavailable = directory / "unavailable.csv"
    _write_csv(unavailable, rows)
    declaration = _declared(unavailable, len(rows))
    declaration["role"] = "unavailable"
    manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_outputs",
        "outputs": [declaration],
        "row_counts": {"unavailable": len(rows)},
        "invariants": {
            "unavailable_rows_excluded_from_estimands": True,
            "unavailable_rows_are_not_recovery_failures": True,
            "payload_fidelity_contract": {
                "contract_version": "payload_fidelity_v2",
                "result_schema_revision": "payload_aware_result_v2",
                "semantics": "original_serialized_payload_bytes_sha256_v1",
                "primary_outcome": "exact_payload_recovery",
                "compatibility_alias": "exact_recovery",
                "alias_equality_validated": True,
                "exact_rank_replay_role": "diagnostic_only",
                "direct_rows": 0,
                "direct_rows_contract_verified": 0,
            },
        },
    }
    path = directory / "preprocessing_output_manifest.json"
    _write_json(path, manifest)
    return path


def _all_sources(tmp_path):
    return {
        "statistics_manifest": _statistics_fixture(tmp_path),
        "theory_manifest": _theory_fixture(tmp_path),
        "detector_manifest": _detector_fixture(tmp_path),
        "runtime_manifests": [_runtime_fixture(tmp_path)],
        "fixture_mode": True,
    }


def test_display_registry_is_stable_unique_and_at_seven_item_limit():
    registry = display_registry()
    assert len(MAIN_DISPLAYS) == 7
    assert registry["main_display_count"] == 7
    assert len({item["label"] for item in registry["main"] + registry["supplementary"]}) == 33
    assert [item["number"] for item in registry["main"] if item["type"] == "figure"] == ["1", "2", "3", "4", "5"]


def test_build_emits_all_tables_plot_sources_and_verified_manifests(tmp_path):
    build = build_revision_reports(output_dir=tmp_path / "reports", **_all_sources(tmp_path))
    assert build.integrity_report["status"] == "passed"
    assert build.integrity_report["main_display_count"] == 7
    assert verify_report_output_manifest(build.output_dir)["status"] == "ok"
    assert len(list((build.output_dir / "tables").glob("*.tex"))) == 15
    assert len(list((build.output_dir / "tables").glob("*.csv"))) == 15
    assert len(list((build.output_dir / "plots" / "sources").glob("*.csv"))) == 18
    table = (build.output_dir / "tables" / "main_table_1.csv").read_text(encoding="utf-8")
    assert "model_a" in table
    assert "source_sha256" in table
    assert "0.8" in table and "0.9" in table
    output_manifest = json.loads((build.output_dir / "report_output_manifest.json").read_text())
    assert len(output_manifest["files"]) == 53


def test_preprocessing_unavailable_rows_are_counted_in_s7_and_s13_only(tmp_path):
    sources = _all_sources(tmp_path)
    sources["preprocessing_manifests"] = [_preprocessing_fixture(tmp_path)]
    build = build_revision_reports(output_dir=tmp_path / "reports", **sources)
    s7 = (build.output_dir / "tables" / "supplementary_table_s7.csv").read_text()
    s13 = (build.output_dir / "tables" / "supplementary_table_s13.csv").read_text()
    assert "empty_isolated_roundtrip_vocabulary" in s7
    assert "condition_unavailable_not_recovery_failure" in s13
    assert "48" in s7 and "48" in s13
    main = (build.output_dir / "tables" / "main_table_1.csv").read_text()
    assert "condition_unavailable" not in main


def test_evaluator_unavailability_is_verified_counted_and_excluded(tmp_path):
    manifest = _evaluator_unavailability_fixture(tmp_path)
    build = build_revision_reports(
        output_dir=tmp_path / "reports",
        evaluator_unavailability_manifest=manifest,
        fixture_mode=True,
    )
    accounting = build.integrity_report["evaluator_unavailability_accounting"]
    assert accounting["scoreable_quality_estimand_units"] == 17232
    assert accounting["terminal_excluded_non_outcomes"] == 48
    assert accounting["terminal_accounted_units"] == 17280
    assert accounting["scores_imputed_or_fabricated"] is False
    s13 = (build.output_dir / "tables" / "supplementary_table_s13.csv").read_text()
    assert "heldout_evaluator_upstream_dependent_unavailable_not_scored" in s13
    assert "17232" in s13 and "48" in s13 and "17280" in s13

    source_records = manifest.parent / "records.jsonl"
    source_records.write_text(source_records.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(RevisionReportingError, match="SHA-256 mismatch"):
        load_verified_sources(
            evaluator_unavailability_manifest=manifest,
            fixture_mode=True,
        )


def test_report_cli_help_accepts_evaluator_unavailability_manifest():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_revision_reports.py"), "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--evaluator-unavailability-manifest" in completed.stdout


def test_plot_script_executes_matplotlib_on_fixture_source(tmp_path):
    pytest.importorskip("matplotlib")
    build = build_revision_reports(output_dir=tmp_path / "reports", **_all_sources(tmp_path))
    script = build.output_dir / "plots" / "plot_revision_figures.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path / "rendered"),
            "--format",
            "png",
            "--only",
            "main_figure_1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "rendered" / "main_figure_1.png").stat().st_size > 1000


def test_manifest_hash_mismatch_is_rejected_before_reading_results(tmp_path):
    manifest_path = _statistics_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["recovery"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(RevisionReportingError, match="SHA-256 mismatch"):
        load_verified_sources(statistics_manifest=manifest_path, fixture_mode=True)


def test_inconsistent_rate_and_cross_table_sample_size_are_rejected(tmp_path):
    bad_rate = _statistics_fixture(tmp_path / "rate", bad_rate=True)
    with pytest.raises(RevisionReportingError, match="rate is inconsistent"):
        load_verified_sources(statistics_manifest=bad_rate, fixture_mode=True)
    bad_n = _statistics_fixture(tmp_path / "n", bad_continuous_n=True)
    with pytest.raises(RevisionReportingError, match="sample sizes disagree"):
        load_verified_sources(statistics_manifest=bad_n, fixture_mode=True)



@pytest.mark.parametrize(
    "field,value,message",
    [
        ("result_schema_revision", "legacy_result_v1", "result_schema_revision"),
        ("evidence_status", "confirmatory_primary_after_manifest_freeze", "legacy or mismatched"),
        ("recovery_outcome_semantics", "rank_replay", "ambiguous recovery semantics"),
        ("successes", "7", "payload successes differ"),
    ],
)
def test_statistics_recovery_contract_rejects_legacy_or_ambiguous_rows(
    tmp_path, field, value, message
):
    manifest = _statistics_fixture(tmp_path / field)
    _mutate_statistics_csv(
        manifest,
        "recovery",
        lambda rows: rows[0].__setitem__(field, value),
    )
    with pytest.raises(RevisionReportingError, match=message):
        load_verified_sources(statistics_manifest=manifest, fixture_mode=True)


def test_smoke_v3_ordinary_control_phase_is_accepted_but_legacy_v2_is_rejected(
    tmp_path,
):
    manifest = _statistics_fixture(tmp_path / "smoke_v3_control")

    def set_smoke_control(rows):
        rows[0]["evidence_status"] = (
            "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling"
        )
        rows[0]["study_phase"] = "ordinary_llm_control_smoke_v3"

    _mutate_statistics_csv(manifest, "recovery", set_smoke_control)
    load_verified_sources(statistics_manifest=manifest, fixture_mode=True)

    _mutate_statistics_csv(
        manifest,
        "recovery",
        lambda rows: rows[0].__setitem__(
            "study_phase", "ordinary_llm_control_smoke_v2"
        ),
    )
    with pytest.raises(RevisionReportingError, match="legacy or mismatched"):
        load_verified_sources(statistics_manifest=manifest, fixture_mode=True)


def test_report_uses_payload_fidelity_when_rank_replay_is_exact(tmp_path):
    build = build_revision_reports(
        output_dir=tmp_path / "reports", **_all_sources(tmp_path)
    )
    with (build.output_dir / "tables" / "main_table_1.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    first = next(row for row in rows if row["protocol_variant"] == "nonseg_ascii_b8")
    assert first["payload_recovery_successes"] == "8"
    assert first["exact_payload_recovery_rate"] == "0.8"
    assert first["recovery_outcome"] == "exact_payload_recovery"
    assert first["recovery_outcome_semantics"] == "original_serialized_payload_bytes_sha256_v1"


def test_old_preprocessing_schema_and_contract_fail_closed(tmp_path):
    manifest = _preprocessing_fixture(tmp_path / "schema")
    value = json.loads(manifest.read_text())
    value["schema_version"] = "1.0"
    _write_json(manifest, value)
    with pytest.raises(RevisionReportingError, match="schema 2.0"):
        load_verified_sources(preprocessing_manifests=[manifest], fixture_mode=True)

    manifest = _preprocessing_fixture(tmp_path / "contract")
    value = json.loads(manifest.read_text())
    value["invariants"]["payload_fidelity_contract"]["result_schema_revision"] = "legacy"
    _write_json(manifest, value)
    with pytest.raises(RevisionReportingError, match="result_schema_revision"):
        load_verified_sources(preprocessing_manifests=[manifest], fixture_mode=True)

def test_detector_row_count_and_smoke_boundaries_are_enforced(tmp_path):
    bad = _detector_fixture(tmp_path / "bad", wrong_count=True)
    with pytest.raises(RevisionReportingError, match="Row-count mismatch"):
        load_verified_sources(detector_manifest=bad, fixture_mode=True)
    smoke = _detector_fixture(tmp_path / "smoke")
    with pytest.raises(RevisionReportingError, match="smoke output"):
        load_verified_sources(detector_manifest=smoke, fixture_mode=False)


def test_detector_manifest_v2_hashes_are_verified(tmp_path):
    manifest = _detector_fixture(tmp_path, schema_v2=True)
    sources = load_verified_sources(detector_manifest=manifest, fixture_mode=True)
    assert all(
        artifact.manifest_declared_sha256
        for key, artifact in sources.artifacts.items()
        if key.startswith("detector.")
    )
    metrics = manifest.parent / "detector_metrics.csv"
    metrics.write_text(metrics.read_text() + "tampered\n", encoding="utf-8")
    with pytest.raises(RevisionReportingError, match="SHA-256 mismatch"):
        load_verified_sources(detector_manifest=manifest, fixture_mode=True)


def test_reporting_reverifies_detector_primary_preprocessing_binding(tmp_path):
    detector_run, dataset_contract, detector = (
        _detector_preprocessing_binding_fixture(tmp_path)
    )
    _validate_detector_preprocessing_binding(detector_run, dataset_contract)
    detector.write_text(detector.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(RevisionReportingError, match="hash-mismatched"):
        _validate_detector_preprocessing_binding(detector_run, dataset_contract)


def test_locked_r_contrasts_are_primary_and_python_effects_never_fallback(tmp_path):
    statistics = _statistics_fixture(tmp_path)
    mixed = _mixed_model_fixture(tmp_path)
    build = build_revision_reports(
        output_dir=tmp_path / "reports",
        statistics_manifest=statistics,
        mixed_model_manifest=mixed,
        fixture_mode=True,
    )
    with (build.output_dir / "tables" / "main_table_2.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    effect_rows = [row for row in rows if row.get("section") == "effect"]
    assert len(effect_rows) == 6
    assert {row["source_artifact"] for row in effect_rows} == {
        "mixed_model.contrasts"
    }
    assert "heldout_evaluator_log_probability" in {
        row["outcome"] for row in effect_rows
    }
    assert build.integrity_report["primary_inference_source"] == (
        "locked_r_mixed_model_manifest"
    )

    def promote_python_effects(rows):
        rows[0]["primary_inference"] = "true"
        rows[0]["inference_role"] = "prespecified_mixed_effects"

    _mutate_statistics_csv(statistics, "effects", promote_python_effects)
    with pytest.raises(RevisionReportingError, match="locked R mixed-model manifest"):
        load_verified_sources(statistics_manifest=statistics, fixture_mode=True)


def test_reporting_rejects_declared_r_driver_other_than_locked_repository_source(tmp_path):
    mixed = _mixed_model_fixture(tmp_path)
    replacement = mixed.parent / "replacement_mixed_model_driver.R"
    replacement.write_text("# forged driver\n", encoding="utf-8")
    manifest = json.loads(mixed.read_text(encoding="utf-8"))
    driver = next(
        row for row in manifest["input_files"] if row["role"] == "driver_source"
    )
    driver.update(
        {
            "path": str(replacement.resolve()),
            "sha256": _sha(replacement),
            "size_bytes": replacement.stat().st_size,
        }
    )
    _write_json(mixed, manifest)
    with pytest.raises(RevisionReportingError, match="locked repository R source"):
        load_verified_sources(mixed_model_manifest=mixed, fixture_mode=True)


def test_generic_same_named_artifact_cannot_enter_primary_r_inference(tmp_path):
    statistics = _statistics_fixture(tmp_path)
    mixed = _mixed_model_fixture(tmp_path)
    runtime_dir = tmp_path / "runtime_collision"
    runtime_dir.mkdir()
    generic_contrasts = runtime_dir / "generic_contrasts.csv"
    _write_csv(
        generic_contrasts,
        [{"model_id": "generic_not_a_prespecified_mixed_model", "estimate": 999}],
    )
    runtime_manifest = runtime_dir / "runtime_manifest.json"
    _write_json(
        runtime_manifest,
        {
            "schema_version": "1.0",
            "outputs": {"contrasts": _declared(generic_contrasts, 1)},
        },
    )

    build = build_revision_reports(
        output_dir=tmp_path / "reports",
        statistics_manifest=statistics,
        mixed_model_manifest=mixed,
        runtime_manifests=(runtime_manifest,),
        fixture_mode=True,
    )
    with (build.output_dir / "tables" / "main_table_2.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        effect_rows = [
            row for row in csv.DictReader(handle) if row.get("section") == "effect"
        ]
    mixed_manifest = json.loads(mixed.read_text(encoding="utf-8"))
    assert len(effect_rows) == 6
    assert {row["source_sha256"] for row in effect_rows} == {
        mixed_manifest["outputs"]["contrasts"]["sha256"]
    }
    assert all("999" not in row["estimate"] for row in effect_rows)


def test_missing_outcomes_remain_explicitly_unavailable(tmp_path):
    theory = _theory_fixture(tmp_path)
    build = build_revision_reports(
        output_dir=tmp_path / "reports",
        theory_manifest=theory,
        fixture_mode=True,
    )
    human_table = (build.output_dir / "tables" / "supplementary_table_s10.csv").read_text()
    human_plot = (build.output_dir / "plots" / "sources" / "main_figure_3.csv").read_text()
    assert "unavailable" in human_table
    assert "unavailable" in human_plot
    assert "synthetic" not in human_table.lower()


def test_existing_different_report_is_never_overwritten(tmp_path):
    sources = _all_sources(tmp_path)
    output = tmp_path / "reports"
    build_revision_reports(output_dir=output, **sources)
    target = output / "tables" / "main_table_1.tex"
    original = target.read_bytes()
    target.write_text("manual edit", encoding="utf-8")
    with pytest.raises(ReportArtifactConflict, match="Refusing to overwrite"):
        build_revision_reports(output_dir=output, **sources)
    assert target.read_text(encoding="utf-8") == "manual edit"
    assert original != target.read_bytes()


def test_identical_retry_is_a_noop_and_generic_runtime_is_ingested(tmp_path):
    sources = _all_sources(tmp_path)
    output = tmp_path / "reports"
    first = build_revision_reports(output_dir=output, **sources)
    before = {key: path.stat().st_mtime_ns for key, path in first.files.items()}
    second = build_revision_reports(output_dir=output, **sources)
    after = {key: path.stat().st_mtime_ns for key, path in second.files.items()}
    assert before == after
    table = (output / "tables" / "main_table_2.csv").read_text()
    assert "model_b" in table
    assert "40.0" in table


def test_latex_escaping_does_not_allow_source_text_to_break_tables():
    assert latex_escape("a_b & 50% #1") == r"a\_b \& 50\% \#1"
