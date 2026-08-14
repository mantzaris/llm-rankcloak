import csv
import hashlib
import json
from pathlib import Path

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256, file_sha256
from rankcloak.revision_evaluator import EVALUATOR_BY_GENERATOR
from rankcloak.revision_evaluator_join import (
    EvaluatorFeatureJoinError,
    join_primary_heldout_evaluator_features,
)


PRIMARY_EVIDENCE = "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
EVALUATOR_EVIDENCE = (
    "confirmatory_heldout_evaluator_primary_v2_payload_fidelity_after_source_manifest_freeze"
)
MODELS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "revision_v1" / "models.json"
MODEL_PINS = {
    str(row["model_id"]): str(row["artifact_sha256"])
    for row in json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))["models"]
}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _declaration(path, role=None, rows=None):
    value = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if role is not None:
        value["role"] = role
    if rows is not None:
        value["row_count"] = rows
    return value


def _fixture(root: Path):
    generators = sorted(EVALUATOR_BY_GENERATOR)
    record_paths = []
    records_by_generator = {}
    feature_rows = []
    for model_index, generator in enumerate(generators):
        records = []
        for local_index in range(2160):
            trial_id = "trial-{}-{:04d}".format(model_index, local_index)
            text = "primary full message {}".format(trial_id)
            row = {
                "record_type": "rankcloak_trial",
                "work_id": trial_id,
                "trial_id": trial_id,
                "model_id": generator,
                "evidence_status": PRIMARY_EVIDENCE,
                "study_phase": "primary_v2_confirmatory",
                "protocol_contract_revision": "payload_fidelity_v2",
                "result_schema_revision": "payload_aware_result_v2",
                "payload_name": "payload-{:03d}".format(local_index % 480),
                "payload_class": "sha256_hex",
                "payload_split": "train",
                "protocol_variant": "nonseg_ascii_b16",
                "prompt_id": "prompt-{:02d}".format(local_index % 18),
                "prompt_category": "casual_conversation",
                "language": "en",
                "full_text": text,
            }
            records.append(row)
            feature_rows.append(
                {
                    "trial_id": trial_id,
                    "segment_index": 0,
                    "source_type": "rankcloak",
                    "text_view": "full_message",
                    "view": "full_message",
                    "text": text,
                    "evidence_status": PRIMARY_EVIDENCE,
                    "study_phase": "primary_v2_confirmatory",
                    "protocol_contract_revision": "payload_fidelity_v2",
                    "result_schema_revision": "payload_aware_result_v2",
                    "transformation_id": "unmodified",
                    "model_id": generator,
                    "payload_name": row["payload_name"],
                    "payload_class": row["payload_class"],
                    "payload_split": row["payload_split"],
                    "protocol_variant": row["protocol_variant"],
                    "prompt_id": row["prompt_id"],
                    "prompt_category": row["prompt_category"],
                    "language": row["language"],
                    "token_count": 4,
                    "mean_log_probability": -2.0,
                    "surface_flag_total": 0,
                }
            )
        records_path = root / "primary" / generator / "records.jsonl"
        _jsonl(records_path, records)
        record_paths.append(records_path)
        records_by_generator[generator] = records

    preprocessing = root / "preprocessed"
    preprocessing.mkdir(parents=True)
    features_path = preprocessing / "features.csv"
    with features_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    input_manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_inputs",
        "strict_complete": True,
        "run_shards": [
            {
                "stage": "primary_v2",
                "model_id": generator,
                "evidence_status": PRIMARY_EVIDENCE,
                "completed_work_units": 1,
                "planned_work_units": 1,
            }
            for generator in generators
        ],
        "input_files": [_declaration(path, role="records") for path in record_paths],
    }
    input_path = preprocessing / "preprocessing_input_manifest.json"
    _json(input_path, input_manifest)
    output_manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_outputs",
        "input_manifest_sha256": file_sha256(input_path),
        "outputs": [
            _declaration(features_path, role="features", rows=len(feature_rows)),
            _declaration(input_path, role="input_manifest"),
        ],
    }
    preprocessing_manifest = preprocessing / "preprocessing_output_manifest.json"
    _json(preprocessing_manifest, output_manifest)

    evaluator_manifests = []
    for generator in generators:
        evaluator = EVALUATOR_BY_GENERATOR[generator]
        directory = root / "evaluator" / evaluator
        directory.mkdir(parents=True, exist_ok=True)
        records_declaration = _declaration(
            root / "primary" / generator / "records.jsonl", role="records"
        )
        shard_files = [records_declaration]
        input_results = {
            "schema_version": "2.0",
            "manifest_type": "heldout_evaluator_inputs",
            "generator_model_id": generator,
            "evaluator_model_id": evaluator,
            "same_model_evaluation": False,
            "generator_artifact_opened_by_evaluator": False,
            "runner_shards": [
                {
                    "stage": "primary_v2",
                    "confirmatory_pooling_eligible": True,
                    "generator_model_id": generator,
                    "files": shard_files,
                    "files_sha256": canonical_json_sha256(shard_files),
                }
            ],
        }
        input_results["inputs_sha256"] = canonical_json_sha256(input_results)
        input_results_path = directory / "input_results_manifest.json"
        _json(input_results_path, input_results)
        input_results_hash = file_sha256(input_results_path)
        evaluator_rows = []
        for record in records_by_generator[generator]:
            text = record["full_text"]
            evaluator_rows.append(
                {
                    "row_id": "eval-" + record["trial_id"],
                    "source_work_id": record["work_id"],
                    "source_trial_id_raw": record["trial_id"],
                    "source_record_type": "rankcloak_trial",
                    "source_record_sha256": canonical_json_sha256(record),
                    "source_stage": "primary_v2",
                    "source_evidence_status": PRIMARY_EVIDENCE,
                    "evidence_status": EVALUATOR_EVIDENCE,
                    "study_phase": "primary_v2_confirmatory",
                    "confirmatory_pooling_eligible": True,
                    "protocol_contract_revision": "payload_fidelity_v2",
                    "result_schema_revision": "payload_aware_result_v2",
                    "same_model_evaluation": False,
                    "model_id": generator,
                    "generator_model_id": generator,
                    "evaluator_model_id": evaluator,
                    "payload_name": record["payload_name"],
                    "payload_class": record["payload_class"],
                    "payload_split": record["payload_split"],
                    "protocol_variant": record["protocol_variant"],
                    "prompt_id": record["prompt_id"],
                    "prompt_category": record["prompt_category"],
                    "language": record["language"],
                    "text_view": "full_message",
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "heldout_evaluator_log_probability": -3.25,
                    "evaluator_token_count": 4,
                    "evaluator_artifact_sha256": MODEL_PINS[evaluator],
                    "evaluator_artifact_actual_sha256": MODEL_PINS[evaluator],
                    "input_results_manifest_sha256": input_results_hash,
                }
            )
        evaluator_features = directory / "features.jsonl"
        continuous = directory / "continuous_quality.jsonl"
        _jsonl(evaluator_features, evaluator_rows)
        _jsonl(continuous, [])
        feature_manifest = {
            "schema_version": "2.0",
            "manifest_type": "heldout_evaluator_feature_table",
            "confirmatory_pooling_eligible": True,
            "nested_segments_are_not_independent": True,
            "evidence_statuses": [EVALUATOR_EVIDENCE],
            "path": str(evaluator_features.resolve()),
            "sha256": file_sha256(evaluator_features),
            "row_count": len(evaluator_rows),
            "continuous_quality_path": str(continuous.resolve()),
            "continuous_quality_sha256": file_sha256(continuous),
        }
        feature_manifest_path = directory / "features_manifest.json"
        _json(feature_manifest_path, feature_manifest)
        evaluator_manifests.append(feature_manifest_path)
    return preprocessing_manifest, evaluator_manifests


def test_primary_evaluator_join_is_complete_hash_checked_and_immutable(tmp_path):
    preprocessing, evaluator_manifests = _fixture(tmp_path)
    output = tmp_path / "joined"
    manifest = join_primary_heldout_evaluator_features(
        preprocessing_manifest=preprocessing,
        evaluator_feature_manifests=evaluator_manifests,
        output_dir=output,
    )
    assert manifest["primary_trial_count"] == 6480
    assert manifest["evaluator_score_rows_joined"] == 6480
    assert manifest["source_record_hashes_recomputed"] is True
    assert manifest["evaluator_artifact_pins_verified"] is True
    assert manifest["evaluator_artifact_pins"] == dict(sorted(MODEL_PINS.items()))
    with (output / "primary_features_with_heldout_evaluator.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6480
    assert {row["heldout_evaluator_log_probability"] for row in rows} == {"-3.25"}
    assert len({row["heldout_evaluator_model_id"] for row in rows}) == 3
    with pytest.raises(EvaluatorFeatureJoinError, match="already exists"):
        join_primary_heldout_evaluator_features(
            preprocessing_manifest=preprocessing,
            evaluator_feature_manifests=evaluator_manifests,
            output_dir=output,
        )

    feature_path = evaluator_manifests[0].parent / "features.jsonl"
    feature_path.write_text(feature_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(EvaluatorFeatureJoinError, match="SHA-256 mismatch"):
        join_primary_heldout_evaluator_features(
            preprocessing_manifest=preprocessing,
            evaluator_feature_manifests=evaluator_manifests,
            output_dir=tmp_path / "tampered_join",
        )


def test_primary_evaluator_join_rejects_forged_actual_artifact_pin(tmp_path):
    preprocessing, evaluator_manifests = _fixture(tmp_path)
    feature_path = evaluator_manifests[0].parent / "features.jsonl"
    rows = [json.loads(line) for line in feature_path.read_text().splitlines()]
    rows[0]["evaluator_artifact_actual_sha256"] = "0" * 64
    _jsonl(feature_path, rows)
    feature_manifest = json.loads(evaluator_manifests[0].read_text(encoding="utf-8"))
    feature_manifest["sha256"] = file_sha256(feature_path)
    _json(evaluator_manifests[0], feature_manifest)
    with pytest.raises(EvaluatorFeatureJoinError, match="frozen artifact pin"):
        join_primary_heldout_evaluator_features(
            preprocessing_manifest=preprocessing,
            evaluator_feature_manifests=evaluator_manifests,
            output_dir=tmp_path / "forged_pin_join",
        )
