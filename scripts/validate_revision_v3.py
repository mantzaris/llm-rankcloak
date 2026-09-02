#!/usr/bin/env python3
"""Fail-closed validation of the complete revision-V3 result ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_revision_v3 import atomic_json, utc_now  # noqa: E402
from rankcloak.revision_v3_metrics import (  # noqa: E402
    empirical_partial_auc,
    frozen_threshold_counts,
    roc_auc,
    select_validation_threshold,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
MODELS = {
    "llama3_8b_instruct_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
}
DETECTORS = {"textcnn", "deberta", "surprisal"}


def close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))


def validate_prediction(
    prediction_path: Path,
    metric_path: Path,
    row_manifest: pd.DataFrame,
) -> list[str]:
    errors = []
    predictions = pd.read_csv(prediction_path, low_memory=False)
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    detector = str(metrics["detector"])
    evaluation = str(metrics["evaluation_id"])
    if detector not in DETECTORS:
        errors.append("unknown detector {}".format(detector))
    validation = predictions.loc[predictions["evaluation_role"].eq("validation")]
    test = predictions.loc[predictions["evaluation_role"].eq("test")]
    if len(validation) != int(metrics["validation_rows"]) or len(test) != int(metrics["test_rows"]):
        errors.append("prediction row counts differ for {} {}".format(detector, evaluation))
    if set(validation["row_id"]) & set(test["row_id"]):
        errors.append("row IDs cross validation/test for {} {}".format(detector, evaluation))
    if set(validation["dedup_cluster_id"]) & set(test["dedup_cluster_id"]):
        errors.append("dedup clusters cross validation/test for {} {}".format(detector, evaluation))
    if evaluation == "matched":
        expected_validation = set(
            row_manifest.loc[row_manifest["partition"].eq("validation"), "row_id"]
        )
        expected_test = set(row_manifest.loc[row_manifest["partition"].eq("test"), "row_id"])
        if set(validation["row_id"]) != expected_validation or set(test["row_id"]) != expected_test:
            errors.append("matched row identities differ for {}".format(detector))
    elif evaluation.startswith("leave_one_model__"):
        target = evaluation.replace("leave_one_model__", "", 1)
        if target not in MODELS or set(test["model_id"]) != {target}:
            errors.append("held-family test identity differs for {}".format(evaluation))
        if target in set(validation["model_id"]):
            errors.append("held model leaked into validation for {}".format(evaluation))
        if target in set(metrics["training_model_ids"]):
            errors.append("held model leaked into training for {}".format(evaluation))
    elif evaluation == "entropy_gates":
        if set(test["quantization"].astype(str)) != {"Q4_K_M"}:
            errors.append("entropy test contains an unexpected quantization")
        if set(test["gate_level"].astype(str)) != {"ungated", "moderate", "strict"}:
            errors.append("entropy test lacks a gate level")
        if metrics.get("deduplication_before_feature_extraction") is not True:
            errors.append("entropy detector was not deduplicated before features")
    elif evaluation in {"q4_to_q8", "q8_to_q4", "pooled_quantizations"}:
        expected = {
            "q4_to_q8": ({"Q4_K_M"}, {"Q8_0"}),
            "q8_to_q4": ({"Q8_0"}, {"Q4_K_M"}),
            "pooled_quantizations": ({"Q4_K_M", "Q8_0"}, {"Q4_K_M", "Q8_0"}),
        }[evaluation]
        if set(metrics["training_quantizations"]) != expected[0] or set(test["quantization"].astype(str)) != expected[1]:
            errors.append("quantization holdout identity differs for {}".format(evaluation))
        if metrics.get("deduplication_before_feature_extraction") is not True:
            errors.append("quantization detector was not deduplicated before features")
    else:
        errors.append("unknown evaluation {}".format(evaluation))
    labels = test["label"].to_numpy(dtype=int)
    scores = test["score"].to_numpy(dtype=float)
    if not close(roc_auc(labels, scores), metrics["roc_auc"]):
        errors.append("ROC-AUC does not reproduce for {} {}".format(detector, evaluation))
    if not close(empirical_partial_auc(labels, scores, 0.01), metrics["partial_auc_fpr_0_01"]):
        errors.append("partial AUC does not reproduce for {} {}".format(detector, evaluation))
    for target, suffix in ((0.01, "0_01"), (0.001, "0_001")):
        selection = select_validation_threshold(
            validation["label"], validation["score"], target
        )
        stored = metrics["threshold_selection"]["fpr_{}".format(suffix)]
        required = int(np.ceil(1.0 / target))
        expected_available = bool(
            selection["available"]
            and int(metrics["test_negative_count"]) >= required
        )
        if expected_available != bool(stored["available"]):
            errors.append("threshold availability differs for {} {} {}".format(detector, evaluation, target))
            continue
        if expected_available:
            if not close(selection["threshold"], stored["threshold"]):
                errors.append("validation threshold does not reproduce for {} {} {}".format(detector, evaluation, target))
            counts = frozen_threshold_counts(labels, scores, float(stored["threshold"]))
            if not close(counts["tpr"], metrics["tpr_at_fpr_{}".format(suffix)]):
                errors.append("frozen test TPR differs for {} {} {}".format(detector, evaluation, target))
            if int(counts["false_positives"]) != int(metrics["false_positives_at_fpr_{}".format(suffix)]):
                errors.append("false-positive count differs for {} {} {}".format(detector, evaluation, target))
        elif any(
            metrics.get(field.format(suffix)) is not None
            for field in (
                "threshold_at_fpr_{}",
                "tpr_at_fpr_{}",
                "fpr_at_threshold_{}",
                "false_positives_at_fpr_{}",
            )
        ):
            errors.append("unavailable low-FPR point contains a numeric estimate for {} {} {}".format(detector, evaluation, target))
    if metrics.get("threshold_selected_on_test") is not False:
        errors.append("no-test-tuning flag is absent for {} {}".format(detector, evaluation))
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    errors = []
    checks = {}
    row_manifest = pd.read_csv(
        output / "deduplication/deduplicated_row_manifest.csv", low_memory=False
    )
    leakage = json.loads((output / "deduplication/leakage_audit.json").read_text())
    checks["leakage_audit_pass"] = leakage.get("status") == "pass"
    if not checks["leakage_audit_pass"]:
        errors.append("strict leakage audit did not pass")
    counts = row_manifest.groupby(["partition", "label"]).size().unstack(fill_value=0)
    checks["every_partition_binary_balanced"] = bool(
        set(counts.index) == {"train", "validation", "test"}
        and (counts[0] == counts[1]).all()
    )
    if not checks["every_partition_binary_balanced"]:
        errors.append("a prepared partition is not exactly label balanced")
    prediction_files = sorted((output / "detector_predictions").glob("*.csv"))
    checks["prediction_file_count"] = len(prediction_files)
    if len(prediction_files) != 24:
        errors.append("expected 24 detector prediction ledgers")
    for prediction_path in prediction_files:
        metric_path = output / "metrics" / (prediction_path.stem + ".json")
        if not metric_path.is_file():
            errors.append("missing metric file for {}".format(prediction_path.name))
            continue
        errors.extend(validate_prediction(prediction_path, metric_path, row_manifest))
    fit_files = sorted((output / "provenance").glob("*__fit.json"))
    checks["fit_metadata_count"] = len(fit_files)
    if len(fit_files) != 24:
        errors.append("expected 24 independent fit metadata records")
    for path in fit_files:
        record = json.loads(path.read_text())
        if record.get("test_tuning") is not False:
            errors.append("fit metadata permits test tuning: {}".format(path.name))

    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_models = run_manifest.get("model_artifacts", [])
    manifest_tokenizers = run_manifest.get("tokenizers", [])
    manifest_datasets = run_manifest.get("datasets", {})
    generation_environment = run_manifest.get("generation_environment", {})
    model_backed_models = run_manifest.get("model_backed_model_artifacts", [])
    model_backed_tokenizers = run_manifest.get("model_backed_tokenizers", [])
    fit_seeds = {
        int(json.loads(path.read_text(encoding="utf-8"))["seed"])
        for path in fit_files
    }
    checks["run_manifest_self_contained"] = bool(
        len(manifest_models) == 4
        and {record.get("quantization") for record in manifest_models}
        == {"Q4_K_M", "Q8_0"}
        and len(manifest_tokenizers) == 5
        and manifest_datasets.get("authoritative_trials", {}).get("sha256")
        == "12b11bad2c7b6468d7f8e4ed12fde2a0ab29996cda8594d0cbcaa6b06797ae7b"
        and manifest_datasets.get("human_source", {}).get("revision")
        == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
        and fit_seeds.issubset(set(map(int, run_manifest.get("random_seeds", []))))
        and "inference_backends" in run_manifest
        and "device_information" in run_manifest
        and "software_environment" in run_manifest
        and len(model_backed_models) == 4
        and {record.get("sha256") for record in model_backed_models}
        == {
            "86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7",
            "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6",
            "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423",
            "9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b",
        }
        and len(model_backed_tokenizers) == 4
        and generation_environment.get(
            "matched_quantization_tokenizer_contract", {}
        ).get("upstream_identifier_exact")
        is True
        and generation_environment.get("remote_paid_compute_used") is False
        and len(generation_environment.get("execution_git_commits", [])) >= 1
        and float(run_manifest.get("run_duration_seconds", 0.0)) > 0.0
    )
    if not checks["run_manifest_self_contained"]:
        errors.append("run manifest is missing a required self-contained provenance field")


    entropy_plan = pd.read_csv(output / "provenance/entropy_generation_plan.csv")
    calibration_plan = pd.read_csv(output / "provenance/entropy_calibration_plan.csv")
    quantization_plan = pd.read_csv(output / "provenance/quantization_generation_plan.csv")
    checks["entropy_plan_rows"] = int(len(entropy_plan))
    checks["entropy_plan_all_new"] = bool(
        len(entropy_plan) == 720
        and entropy_plan["generation_required"].astype(bool).all()
        and not entropy_plan["exact_v2_baseline_compatible"].astype(bool).any()
        and entropy_plan["model_id"].nunique() == 3
        and entropy_plan["payload_class"].nunique() == 8
        and entropy_plan[["payload_class", "representation_name"]].drop_duplicates().shape[0] == 20
        and entropy_plan["prompt_template_id"].nunique() == 2
        and entropy_plan["gate_level"].nunique() == 3
        and entropy_plan["random_seed"].nunique() == 240
        and entropy_plan.groupby(["experimental_cell_id", "population"])["random_seed"].nunique().eq(1).all()
        and entropy_plan.groupby("experimental_cell_id")["random_seed"].nunique().eq(2).all()
    )
    if not checks["entropy_plan_all_new"]:
        errors.append("entropy generation plan is not the frozen 720-row all-new matrix")
    checks["entropy_calibration_plan_valid"] = bool(
        len(calibration_plan) == 18
        and calibration_plan["model_id"].nunique() == 3
        and calibration_plan["prompt_category"].nunique() == 6
        and not calibration_plan["detector_outcomes_used"].astype(bool).any()
    )
    if not checks["entropy_calibration_plan_valid"]:
        errors.append("entropy calibration plan is not the frozen 3-by-6 design")
    quantization_pairs = quantization_plan.groupby("pairing_unit_id")["quantization"].agg(set)
    checks["quantization_plan_valid"] = bool(
        len(quantization_plan) == 3840
        and len(quantization_pairs) == 1920
        and quantization_pairs.map(lambda value: value == {"Q4_K_M", "Q8_0"}).all()
        and not quantization_plan.loc[
            quantization_plan["quantization"].eq("Q4_K_M"), "generation_required"
        ].astype(bool).any()
        and quantization_plan.loc[
            quantization_plan["quantization"].eq("Q8_0"), "generation_required"
        ].astype(bool).all()
        and quantization_plan.loc[
            quantization_plan["quantization"].eq("Q4_K_M"), "source_row_id"
        ].notna().all()
        and quantization_plan["payload_name"].nunique() == 480
    )
    if not checks["quantization_plan_valid"]:
        errors.append("matched-quantization generation plan is not genuinely paired")
    generation_preflight = json.loads(
        (output / "provenance/generation_preflight.json").read_text(encoding="utf-8")
    )
    checks["generation_preflight_ready"] = bool(
        generation_preflight.get("status") == "ready"
        and generation_preflight.get("launch_performed") is False
        and generation_preflight.get("downloads_performed") is False
        and generation_preflight.get("entropy_experiment_ready") is True
        and generation_preflight.get("matched_quantization_experiment_ready") is True
        and generation_preflight.get("backend_cuda_valid") is True
        and all(item.get("valid") is True for item in generation_preflight.get("artifacts", []))
    )
    if not checks["generation_preflight_ready"]:
        errors.append("generation preflight does not verify the exact backend and artifacts")

    generation_validation = json.loads(
        (output / "provenance/generation_analysis_validation.json").read_text(encoding="utf-8")
    )
    checks["model_backed_generation_validation_pass"] = bool(
        generation_validation.get("status") == "pass"
        and generation_validation.get("counts", {}).get("calibration_traces") == 18
        and generation_validation.get("counts", {}).get("entropy_evaluations") == 720
        and generation_validation.get("counts", {}).get("quantization_q4_replays") == 1920
        and generation_validation.get("counts", {}).get("quantization_q8_generations") == 1920
        and generation_validation.get("counts", {}).get("real_model_smoke_records") == 8
        and generation_validation.get("checks", {}).get("real_model_smoke_matrix_exact")
        is True
    )
    if not checks["model_backed_generation_validation_pass"]:
        errors.append("model-backed generation validation did not pass exact counts")
    entropy_positions = pd.read_csv(
        output / "source_tables/entropy_position_summary.csv", low_memory=False
    )
    forced_positions = entropy_positions.loc[
        entropy_positions["analysis_id"].eq("entropy_position_overall")
        & entropy_positions["population"].eq("rankcloak")
        & entropy_positions["token_role"].eq("payload")
        & entropy_positions["metric"].isin(
            {
                "observed_rank",
                "token_surprisal_nats",
                "rank_pressure_log_probability_gap_nats",
            }
        )
    ]
    forced_table = output / "manuscript_tables/entropy_gate_forced_token_distribution.tex"
    model_table = output / "manuscript_tables/entropy_gate_by_model.tex"
    checks["entropy_forced_token_distribution_complete"] = bool(
        len(forced_positions) == 9
        and set(forced_positions["gate_level"].astype(str))
        == {"ungated", "moderate", "strict"}
        and forced_positions["position_count"].astype(int).gt(0).all()
        and forced_table.is_file()
        and forced_table.stat().st_size > 1000
        and model_table.is_file()
        and model_table.stat().st_size > 1000
    )
    if not checks["entropy_forced_token_distribution_complete"]:
        errors.append("forced-token rank/surprisal distribution artifact is incomplete")
    locked_audits = sorted(
        (output / "deduplication").glob("model_backed__*__leakage_audit.json")
    )
    checks["model_backed_locked_dedup_audit_count"] = len(locked_audits)
    checks["model_backed_locked_dedup_all_pass"] = bool(
        len(locked_audits) == 4
        and all(json.loads(path.read_text(encoding="utf-8")).get("status") == "pass" for path in locked_audits)
    )
    if not checks["model_backed_locked_dedup_all_pass"]:
        errors.append("one or more model-backed locked dedup audits is missing or failed")

    experiment_design = (output / "experiment_design.md").read_text(encoding="utf-8")
    checks["model_backed_experiment_design_section_exact"] = bool(
        experiment_design.count("<!-- revision-v3-model-backed-design:start -->") == 1
        and experiment_design.count("<!-- revision-v3-model-backed-design:end -->") == 1
        and experiment_design.count("The entropy-gate matrix used") == 1
        and experiment_design.count("The matched quantization matrix used") == 1
        and experiment_design.count("The recovery-mode comparison") == 1
    )
    if not checks["model_backed_experiment_design_section_exact"]:
        errors.append("model-backed experiment-design section is missing or duplicated")

    recovery = pd.read_csv(output / "source_tables/recovery_mode_comparison.csv")
    recovery_rates = recovery.set_index("replay_mode")["recovery_rate"].to_dict()
    checks["recovery_mode_source_valid"] = bool(
        len(recovery) == 3
        and set(recovery["observed_outcome_rows"].astype(int)) == {144}
        and close(recovery_rates.get("saved_token_ids"), 1.0)
        and close(recovery_rates.get("greedy_leadin_regeneration"), 1.0)
        and close(recovery_rates.get("detokenized_text_retokenized"), 88.0 / 144.0)
        and recovery["source_sha256"].eq(
            "bacd0e260ca9eb24638f9970a86a5d54ee51b77e48e43760a0abf940b296dd84"
        ).all()
    )
    if not checks["recovery_mode_source_valid"]:
        errors.append("bounded recovery-mode source table differs from immutable V1 evidence")

    human_import = json.loads(
        (output / "provenance/human_control_import_audit.json").read_text(encoding="utf-8")
    )
    checks["human_import_provenance_valid"] = bool(
        human_import.get("candidate_manifest_sha256")
        == "23913150fd9d7d46e2cab1c382eebba09454090bc90145187f7394b0a95619c9"
        and human_import.get("candidate_record_count") == 6606
        and human_import.get("network_access_performed_by_pipeline") is False
    )
    if not checks["human_import_provenance_valid"]:
        errors.append("human-control import provenance differs from the pinned audit")
    immutable_sources = {
        "authoritative_trial": (
            PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/trials.csv",
            "12b11bad2c7b6468d7f8e4ed12fde2a0ab29996cda8594d0cbcaa6b06797ae7b",
        ),
        "authoritative_detector": (
            PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl",
            "0688cc5904128582cdfad20bc1dbd9a6153dca4edf1a2e6ef1c54aeeb8fbfb76",
        ),
        "historical_recovery": (
            PROJECT_ROOT / "results/revision_v1/final_experiment_package/robustness/recovery_by_condition.csv",
            "bacd0e260ca9eb24638f9970a86a5d54ee51b77e48e43760a0abf940b296dd84",
        ),
    }
    for source_id, (source_path, expected_sha) in immutable_sources.items():
        digest = __import__("hashlib").sha256(source_path.read_bytes()).hexdigest()
        passed = digest == expected_sha
        checks["{}_sha256_unchanged".format(source_id)] = passed
        if not passed:
            errors.append("immutable source hash changed: {}".format(source_id))
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", "paperV2", "paperV3"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    checks["manuscript_directories_unmodified"] = not bool(diff.stdout.strip())
    if not checks["manuscript_directories_unmodified"]:
        errors.append("paperV2 or paperV3 contains a tracked modification")
    for stem in (
        "detector_model_family_generalization",
        "detector_low_fpr",
        "human_control_false_positives",
        "topic_conditioned_cover_variability",
        "entropy_gate_capacity_recovery",
        "entropy_gate_detector_performance",
        "matched_quantization_sensitivity",
        "matched_quantization_recovery",
        "matched_quantization_detector_transfer",
    ):
        for suffix in (".pdf", ".png"):
            path = output / "figures" / (stem + suffix)
            if not path.is_file() or path.stat().st_size < 1000:
                errors.append("missing or empty figure {}".format(path.name))
    artifact_manifest = pd.read_csv(output / "artifact_manifest.csv")
    manifest_paths = set(artifact_manifest["path"].astype(str))
    excluded_artifacts = {
        "artifact_manifest.csv",
        "provenance/validation_report.json",
    }
    actual_paths = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and str(path.relative_to(output)) not in excluded_artifacts
    }
    checks["artifact_manifest_path_set_exact"] = manifest_paths == actual_paths
    if not checks["artifact_manifest_path_set_exact"]:
        errors.append("artifact manifest path set is incomplete or contains stale paths")
    artifact_errors = []
    for row in artifact_manifest.to_dict("records"):
        path = output / str(row["path"])
        if not path.is_file():
            artifact_errors.append("missing {}".format(row["path"]))
            continue
        if int(row["size_bytes"]) != path.stat().st_size:
            artifact_errors.append("size {}".format(row["path"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(row["sha256"]):
            artifact_errors.append("sha256 {}".format(row["path"]))
        if path.suffix == ".csv" and pd.notna(row.get("row_count")):
            row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            if int(row["row_count"]) != row_count:
                artifact_errors.append("row_count {}".format(row["path"]))
    checks["artifact_manifest_entry_count"] = len(artifact_manifest)
    checks["artifact_manifest_checksums_valid"] = not artifact_errors
    if artifact_errors:
        errors.extend("artifact manifest mismatch: {}".format(item) for item in artifact_errors)
    checks["all_numeric_recomputations_pass"] = not any(
        "reproduce" in error or "differs" in error for error in errors
    )
    report = {
        "schema_version": "rankcloak-revision-v3-validation-report-v1",
        "status": "pass" if not errors else "fail",
        "completed_at": utc_now(),
        "checks": checks,
        "errors": errors,
    }
    atomic_json(output / "provenance/validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
