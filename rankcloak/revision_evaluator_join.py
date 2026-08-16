"""Hash-verified held-out evaluator join for primary full-message features.

This adapter is deliberately narrower than the general preprocessing layer. It
accepts one complete primary preprocessing manifest and the three cyclic
held-out evaluator feature manifests, recomputes source-record hashes, and
writes only the primary RankCloak full-message feature rows needed by the
prospectively locked R models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .revision_artifacts import canonical_json_bytes, canonical_json_sha256, file_sha256
from .revision_evaluator import EVALUATOR_BY_GENERATOR
from .revision_runner import PROTOCOL_CONTRACT_REVISION, RESULT_SCHEMA_REVISION
from .revision_statistics import automated_text_quality_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "models.json"
JOIN_SCHEMA_VERSION = "rankcloak-revision-heldout-feature-join-v1"
JOIN_MANIFEST_TYPE = "rankcloak_revision_primary_heldout_feature_join"
OUTPUT_FILENAME = "primary_features_with_heldout_evaluator.csv"
MANIFEST_FILENAME = "heldout_feature_join_manifest.json"
PRIMARY_EVIDENCE_STATUS = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
PRIMARY_EVALUATOR_EVIDENCE_STATUS = (
    "confirmatory_heldout_evaluator_primary_v2_payload_fidelity_after_source_manifest_freeze"
)
PRIMARY_STUDY_PHASE = "primary_v2_confirmatory"
EXPECTED_PRIMARY_TRIALS = 6480
ARTIFACT_OUTCOME_CANDIDATES = (
    "artifact_count",
    "surface_flag_total",
    "artifact_like_fragment_count",
)
DERIVED_ARTIFACT_COLUMNS = (
    "surface_flag_total",
    "artifact_like_fragment_count",
)


class EvaluatorFeatureJoinError(ValueError):
    """Raised when evaluator/preprocessing lineage cannot support a safe join."""


def _frozen_evaluator_artifact_pins() -> Dict[str, str]:
    value = _read_json(MODELS_CONFIG)
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list):
        raise EvaluatorFeatureJoinError("Frozen evaluator model config is malformed")
    pins: Dict[str, str] = {}
    for model in models:
        if not isinstance(model, dict):
            raise EvaluatorFeatureJoinError("Frozen evaluator model entry is malformed")
        model_id = str(model.get("model_id", ""))
        artifact_sha256 = str(model.get("artifact_sha256", ""))
        if (
            not model_id
            or len(artifact_sha256) != 64
            or model_id in pins
        ):
            raise EvaluatorFeatureJoinError("Frozen evaluator artifact pin is malformed")
        pins[model_id] = artifact_sha256
    expected_models = set(EVALUATOR_BY_GENERATOR) | set(EVALUATOR_BY_GENERATOR.values())
    if set(pins) != expected_models:
        raise EvaluatorFeatureJoinError("Frozen evaluator pins do not cover the cyclic model set")
    return pins


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorFeatureJoinError("Cannot read JSON {}: {}".format(path, exc))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EvaluatorFeatureJoinError(
                        "JSONL row {}:{} is not an object".format(path, line_number)
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorFeatureJoinError("Cannot read JSONL {}: {}".format(path, exc))
    return rows


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvaluatorFeatureJoinError("CSV has no header: {}".format(path))
            return list(reader.fieldnames), [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise EvaluatorFeatureJoinError("Cannot read CSV {}: {}".format(path, exc))


def _resolve_path(manifest_path: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise EvaluatorFeatureJoinError(
            "Manifest {} contains an empty artifact path".format(manifest_path)
        )
    raw = Path(declared)
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend((manifest_path.parent / raw, manifest_path.parent / raw.name))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and not resolved.is_symlink():
            return resolved
    raise EvaluatorFeatureJoinError(
        "Manifest {} declares a missing artifact {}".format(manifest_path, declared)
    )


def _verify_declaration(manifest_path: Path, declaration: Mapping[str, Any]) -> Path:
    path = _resolve_path(manifest_path, declaration.get("path"))
    expected_hash = str(declaration.get("sha256", ""))
    if len(expected_hash) != 64 or file_sha256(path) != expected_hash:
        raise EvaluatorFeatureJoinError("SHA-256 mismatch for {}".format(path))
    expected_size = declaration.get("size_bytes", declaration.get("bytes"))
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise EvaluatorFeatureJoinError("Byte-count mismatch for {}".format(path))
    return path


def _output_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise EvaluatorFeatureJoinError("Preprocessing outputs must be a list")
    matches = [row for row in outputs if isinstance(row, dict) and row.get("role") == role]
    if len(matches) != 1:
        raise EvaluatorFeatureJoinError(
            "Preprocessing manifest must declare exactly one {} output".format(role)
        )
    return matches[0]


def _verified_primary_inputs(
    preprocessing_manifest_path: Path,
) -> Tuple[List[str], List[Dict[str, str]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, str], List[dict]]:
    manifest = _read_json(preprocessing_manifest_path)
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != "2.0"
        or manifest.get("manifest_type") != "revision_preprocessing_outputs"
    ):
        raise EvaluatorFeatureJoinError("Expected preprocessing output schema 2.0")
    feature_declaration = _output_by_role(manifest, "features")
    input_declaration = _output_by_role(manifest, "input_manifest")
    feature_path = _verify_declaration(preprocessing_manifest_path, feature_declaration)
    input_manifest_path = _verify_declaration(preprocessing_manifest_path, input_declaration)
    if manifest.get("input_manifest_sha256") != file_sha256(input_manifest_path):
        raise EvaluatorFeatureJoinError("Preprocessing input-manifest hash differs")
    columns, features = _read_csv(feature_path)
    if len(features) != int(feature_declaration.get("row_count", -1)):
        raise EvaluatorFeatureJoinError("Preprocessing feature row count differs")

    input_manifest = _read_json(input_manifest_path)
    if not isinstance(input_manifest, dict) or (
        input_manifest.get("schema_version") != "2.0"
        or input_manifest.get("manifest_type") != "revision_preprocessing_inputs"
        or input_manifest.get("strict_complete") is not True
    ):
        raise EvaluatorFeatureJoinError("Preprocessing input manifest is not strict schema 2.0")
    shards = input_manifest.get("run_shards")
    if not isinstance(shards, list) or len(shards) != 3:
        raise EvaluatorFeatureJoinError("Primary preprocessing must contain exactly three shards")
    for shard in shards:
        if not isinstance(shard, dict) or (
            shard.get("stage") != "primary_v2"
            or shard.get("evidence_status") != PRIMARY_EVIDENCE_STATUS
            or int(shard.get("completed_work_units", -1))
            != int(shard.get("planned_work_units", -2))
        ):
            raise EvaluatorFeatureJoinError("Preprocessing contains a non-primary/incomplete shard")

    files = input_manifest.get("input_files")
    if not isinstance(files, list):
        raise EvaluatorFeatureJoinError("Preprocessing input_files must be a list")
    record_declarations = [
        row for row in files if isinstance(row, dict) and row.get("role") == "records"
    ]
    if len(record_declarations) != 3:
        raise EvaluatorFeatureJoinError("Primary preprocessing must declare three records files")
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    record_file_hashes: Dict[str, str] = {}
    verified_inputs: List[dict] = [
        {
            "role": "preprocessing_output_manifest",
            "path": str(preprocessing_manifest_path),
            "sha256": file_sha256(preprocessing_manifest_path),
        },
        {
            "role": "preprocessing_input_manifest",
            "path": str(input_manifest_path),
            "sha256": file_sha256(input_manifest_path),
        },
        {
            "role": "preprocessing_features",
            "path": str(feature_path),
            "sha256": file_sha256(feature_path),
        },
    ]
    for declaration in record_declarations:
        path = _verify_declaration(input_manifest_path, declaration)
        digest = file_sha256(path)
        record_file_hashes[digest] = str(path)
        verified_inputs.append(
            {"role": "primary_records", "path": str(path), "sha256": digest}
        )
        for record in _read_jsonl(path):
            if record.get("record_type") != "rankcloak_trial":
                continue
            work_id = str(record.get("work_id", ""))
            digest = canonical_json_sha256(record)
            key = (work_id, digest)
            if not work_id or key in records:
                raise EvaluatorFeatureJoinError("Duplicate/empty primary source-record identity")
            records[key] = record
    return columns, features, records, record_file_hashes, verified_inputs


def _verify_evaluator_manifest(
    manifest_path: Path,
    source_records: Mapping[Tuple[str, str], Mapping[str, Any]],
    preprocessing_record_hashes: Mapping[str, str],
    evaluator_artifact_pins: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[dict]]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != "2.0"
        or manifest.get("manifest_type") != "heldout_evaluator_feature_table"
        or manifest.get("confirmatory_pooling_eligible") is not True
        or manifest.get("nested_segments_are_not_independent") is not True
        or manifest.get("evidence_statuses") != [PRIMARY_EVALUATOR_EVIDENCE_STATUS]
    ):
        raise EvaluatorFeatureJoinError("Evaluator feature manifest is not primary confirmatory")
    feature_declaration = {
        "path": manifest.get("path"),
        "sha256": manifest.get("sha256"),
    }
    feature_path = _verify_declaration(manifest_path, feature_declaration)
    continuous_path = _verify_declaration(
        manifest_path,
        {
            "path": manifest.get("continuous_quality_path"),
            "sha256": manifest.get("continuous_quality_sha256"),
        },
    )
    rows = _read_jsonl(feature_path)
    if len(rows) != int(manifest.get("row_count", -1)):
        raise EvaluatorFeatureJoinError("Evaluator feature row count differs")

    input_manifest_path = manifest_path.parent / "input_results_manifest.json"
    if not input_manifest_path.is_file() or input_manifest_path.is_symlink():
        raise EvaluatorFeatureJoinError("Evaluator input_results_manifest.json is missing")
    input_manifest = _read_json(input_manifest_path)
    if not isinstance(input_manifest, dict) or (
        input_manifest.get("schema_version") != "2.0"
        or input_manifest.get("manifest_type") != "heldout_evaluator_inputs"
        or input_manifest.get("same_model_evaluation") is not False
        or input_manifest.get("generator_artifact_opened_by_evaluator") is not False
    ):
        raise EvaluatorFeatureJoinError("Evaluator input manifest violates isolation policy")
    unsigned = dict(input_manifest)
    claimed_inputs_hash = unsigned.pop("inputs_sha256", None)
    if claimed_inputs_hash != canonical_json_sha256(unsigned):
        raise EvaluatorFeatureJoinError("Evaluator input manifest content hash differs")
    input_manifest_hash = file_sha256(input_manifest_path)
    generator = str(input_manifest.get("generator_model_id", ""))
    evaluator = str(input_manifest.get("evaluator_model_id", ""))
    if EVALUATOR_BY_GENERATOR.get(generator) != evaluator:
        raise EvaluatorFeatureJoinError("Evaluator/generator mapping is not the frozen cyclic map")
    runner_shards = input_manifest.get("runner_shards")
    if not isinstance(runner_shards, list) or len(runner_shards) != 1:
        raise EvaluatorFeatureJoinError("Primary evaluator manifest must reference one generator shard")
    shard = runner_shards[0]
    if not isinstance(shard, dict) or (
        shard.get("stage") != "primary_v2"
        or shard.get("confirmatory_pooling_eligible") is not True
        or shard.get("generator_model_id") != generator
    ):
        raise EvaluatorFeatureJoinError("Evaluator source shard is not primary confirmatory")
    shard_files = shard.get("files")
    if not isinstance(shard_files, list) or shard.get("files_sha256") != canonical_json_sha256(
        shard_files
    ):
        raise EvaluatorFeatureJoinError("Evaluator source file-list hash differs")
    source_record_files = []
    for declaration in shard_files:
        if not isinstance(declaration, dict):
            raise EvaluatorFeatureJoinError("Malformed evaluator source declaration")
        path = _verify_declaration(input_manifest_path, declaration)
        if declaration.get("role") == "records":
            source_record_files.append((path, str(declaration.get("sha256"))))
    if len(source_record_files) != 1 or source_record_files[0][1] not in preprocessing_record_hashes:
        raise EvaluatorFeatureJoinError(
            "Evaluator records source is not byte-identical to preprocessing lineage"
        )

    selected: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("input_results_manifest_sha256") != input_manifest_hash:
            raise EvaluatorFeatureJoinError("Evaluator row input-manifest hash differs")
        if row.get("source_record_type") != "rankcloak_trial":
            continue
        if (
            row.get("source_stage") != "primary_v2"
            or row.get("source_evidence_status") != PRIMARY_EVIDENCE_STATUS
            or row.get("evidence_status") != PRIMARY_EVALUATOR_EVIDENCE_STATUS
            or row.get("study_phase") != PRIMARY_STUDY_PHASE
            or row.get("confirmatory_pooling_eligible") is not True
            or row.get("text_view") != "full_message"
            or row.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION
            or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
            or row.get("same_model_evaluation") is not False
            or row.get("generator_model_id") != generator
            or row.get("evaluator_model_id") != evaluator
            or row.get("evaluator_artifact_sha256")
            != evaluator_artifact_pins.get(evaluator)
            or row.get("evaluator_artifact_actual_sha256")
            != evaluator_artifact_pins.get(evaluator)
        ):
            raise EvaluatorFeatureJoinError(
                "Evaluator RankCloak row violates primary scope or frozen artifact pin"
            )
        key = (str(row.get("source_work_id", "")), str(row.get("source_record_sha256", "")))
        source = source_records.get(key)
        if source is None:
            raise EvaluatorFeatureJoinError("Evaluator source-record hash is absent from preprocessing")
        raw_trial_id = str(source.get("trial_id") or source.get("work_id"))
        if row.get("source_trial_id_raw") != raw_trial_id:
            raise EvaluatorFeatureJoinError("Evaluator raw trial ID differs from source record")
        full_text = str(source.get("full_text", ""))
        if (
            row.get("text") != full_text
            or row.get("text_sha256")
            != hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        ):
            raise EvaluatorFeatureJoinError("Evaluator text differs from hashed source record")
        for field in (
            "model_id",
            "payload_name",
            "payload_class",
            "payload_split",
            "protocol_variant",
            "prompt_id",
            "prompt_category",
            "language",
        ):
            expected = generator if field == "model_id" else source.get(field)
            if str(row.get(field, "")) != str(expected if expected is not None else ""):
                raise EvaluatorFeatureJoinError(
                    "Evaluator/source metadata differs for {}".format(field)
                )
        try:
            score = float(row["heldout_evaluator_log_probability"])
            token_count = int(row["evaluator_token_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluatorFeatureJoinError("Evaluator score/token count is malformed") from exc
        if not math.isfinite(score) or token_count <= 0:
            raise EvaluatorFeatureJoinError("Evaluator score/token count is outside its domain")
        row = dict(row)
        row["_feature_manifest_sha256"] = file_sha256(manifest_path)
        selected.append(row)
    provenance = [
        {"role": "evaluator_features_manifest", "path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        {"role": "evaluator_features", "path": str(feature_path), "sha256": file_sha256(feature_path)},
        {"role": "evaluator_continuous_quality", "path": str(continuous_path), "sha256": file_sha256(continuous_path)},
        {"role": "evaluator_input_manifest", "path": str(input_manifest_path.resolve()), "sha256": input_manifest_hash},
    ]
    return selected, provenance


def _join_rows(
    columns: Sequence[str],
    features: Sequence[Mapping[str, str]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], int, Dict[str, Any]]:
    added = [
        "heldout_evaluator_log_probability",
        "heldout_evaluator_token_count",
        "heldout_evaluator_model_id",
        "heldout_evaluator_artifact_sha256",
        "heldout_evaluator_source_record_sha256",
        "heldout_evaluator_feature_row_id",
        "heldout_evaluator_feature_manifest_sha256",
        "heldout_evaluator_score_scope",
    ]
    collisions = sorted(set(columns) & set(added))
    if collisions:
        raise EvaluatorFeatureJoinError(
            "Preprocessing features already contain held-out join columns: {}".format(
                ", ".join(collisions)
            )
        )
    score_by_trial: Dict[str, Mapping[str, Any]] = {}
    for row in evaluator_rows:
        trial_id = str(row.get("source_trial_id_raw", ""))
        if not trial_id or trial_id in score_by_trial:
            raise EvaluatorFeatureJoinError("Evaluator scores are not one-to-one by primary trial")
        score_by_trial[trial_id] = row
    selected = [
        dict(row)
        for row in features
        if row.get("source_type") == "rankcloak"
        and row.get("text_view") == "full_message"
        and row.get("evidence_status") == PRIMARY_EVIDENCE_STATUS
        and row.get("study_phase") == PRIMARY_STUDY_PHASE
        and row.get("protocol_contract_revision") == PROTOCOL_CONTRACT_REVISION
        and row.get("result_schema_revision") == RESULT_SCHEMA_REVISION
        and row.get("transformation_id") == "unmodified"
    ]
    available_artifact_columns = [
        column for column in ARTIFACT_OUTCOME_CANDIDATES if column in columns
    ]
    if available_artifact_columns:
        artifact_diagnostics = {
            "status": "source_feature_column_preserved",
            "selected_source_column": available_artifact_columns[0],
            "outcome_candidates": list(ARTIFACT_OUTCOME_CANDIDATES),
            "derived_columns": [],
            "row_count": len(selected),
        }
    else:
        for feature in selected:
            metrics = automated_text_quality_metrics(
                str(feature.get("text", "")),
                str(feature.get("prompt_text", "")),
                language=str(feature.get("language", "")),
            )
            for column in DERIVED_ARTIFACT_COLUMNS:
                feature[column] = int(metrics[column])
        artifact_diagnostics = {
            "status": "derived_from_hash_bound_text_rows",
            "selected_source_column": "surface_flag_total",
            "outcome_candidates": list(ARTIFACT_OUTCOME_CANDIDATES),
            "derived_columns": list(DERIVED_ARTIFACT_COLUMNS),
            "algorithm_module": "rankcloak.revision_statistics",
            "algorithm_function": "automated_text_quality_metrics",
            "algorithm_source_path": str(
                (PROJECT_ROOT / "rankcloak" / "revision_statistics.py").resolve()
            ),
            "algorithm_source_sha256": file_sha256(
                PROJECT_ROOT / "rankcloak" / "revision_statistics.py"
            ),
            "row_count": len(selected),
        }
    feature_trials = {str(row.get("trial_id", "")) for row in selected}
    if "" in feature_trials or feature_trials != set(score_by_trial):
        missing = sorted(feature_trials - set(score_by_trial))[:5]
        extra = sorted(set(score_by_trial) - feature_trials)[:5]
        raise EvaluatorFeatureJoinError(
            "Primary feature/evaluator trial sets differ; missing_scores={}, "
            "scores_without_features={}".format(missing, extra)
        )
    joined: List[Dict[str, Any]] = []
    for feature in selected:
        score = score_by_trial[str(feature["trial_id"])]
        for field in (
            "model_id",
            "payload_name",
            "payload_class",
            "payload_split",
            "protocol_variant",
            "prompt_id",
            "prompt_category",
            "language",
        ):
            if str(feature.get(field, "")) != str(score.get(field, "")):
                raise EvaluatorFeatureJoinError(
                    "Feature/evaluator metadata differs for trial {} field {}".format(
                        feature["trial_id"], field
                    )
                )
        feature.update(
            {
                "heldout_evaluator_log_probability": score[
                    "heldout_evaluator_log_probability"
                ],
                "heldout_evaluator_token_count": score["evaluator_token_count"],
                "heldout_evaluator_model_id": score["evaluator_model_id"],
                "heldout_evaluator_artifact_sha256": score[
                    "evaluator_artifact_actual_sha256"
                ],
                "heldout_evaluator_source_record_sha256": score[
                    "source_record_sha256"
                ],
                "heldout_evaluator_feature_row_id": score["row_id"],
                "heldout_evaluator_feature_manifest_sha256": score[
                    "_feature_manifest_sha256"
                ],
                "heldout_evaluator_score_scope": (
                    "source_full_message_replicated_across_nested_segment_rows_v1"
                ),
            }
        )
        joined.append(feature)
    output_columns = list(columns)
    output_columns.extend(
        column
        for column in DERIVED_ARTIFACT_COLUMNS
        if column in artifact_diagnostics["derived_columns"]
        and column not in output_columns
    )
    output_columns.extend(column for column in added if column not in output_columns)
    return output_columns, joined, len(feature_trials), artifact_diagnostics


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in columns})
        handle.flush()
        os.fsync(handle.fileno())


def join_primary_heldout_evaluator_features(
    *,
    preprocessing_manifest: Path,
    evaluator_feature_manifests: Sequence[Path],
    output_dir: Path,
) -> Mapping[str, Any]:
    """Build one immutable, hash-addressed primary feature table for locked R."""

    preprocessing_manifest = Path(preprocessing_manifest).resolve()
    evaluator_paths = [Path(path).resolve() for path in evaluator_feature_manifests]
    if len(evaluator_paths) != 3 or len(set(evaluator_paths)) != 3:
        raise EvaluatorFeatureJoinError("Exactly three distinct evaluator manifests are required")
    columns, features, source_records, record_hashes, provenance = _verified_primary_inputs(
        preprocessing_manifest
    )
    evaluator_artifact_pins = _frozen_evaluator_artifact_pins()
    evaluator_rows: List[Dict[str, Any]] = []
    for path in evaluator_paths:
        rows, inputs = _verify_evaluator_manifest(
            path,
            source_records,
            record_hashes,
            evaluator_artifact_pins,
        )
        evaluator_rows.extend(rows)
        provenance.extend(inputs)
    generators = {str(row.get("generator_model_id", "")) for row in evaluator_rows}
    evaluators = {str(row.get("evaluator_model_id", "")) for row in evaluator_rows}
    if generators != set(EVALUATOR_BY_GENERATOR) or evaluators != set(
        EVALUATOR_BY_GENERATOR.values()
    ):
        raise EvaluatorFeatureJoinError("Evaluator rows do not cover all three frozen model families")
    output_columns, joined, trial_count, artifact_diagnostics = _join_rows(
        columns, features, evaluator_rows
    )
    if trial_count != EXPECTED_PRIMARY_TRIALS:
        raise EvaluatorFeatureJoinError(
            "Primary evaluator join requires {} trials, observed {}".format(
                EXPECTED_PRIMARY_TRIALS, trial_count
            )
        )

    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise EvaluatorFeatureJoinError("Output directory already exists: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.staging-".format(destination.name), dir=destination.parent)
    )
    committed = False
    try:
        output_path = staging / OUTPUT_FILENAME
        _write_csv(output_path, output_columns, joined)
        output_identity = {
            "path": OUTPUT_FILENAME,
            "sha256": file_sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "row_count": len(joined),
        }
        manifest = {
            "schema_version": JOIN_SCHEMA_VERSION,
            "manifest_type": JOIN_MANIFEST_TYPE,
            "analysis_unit": "primary_payload_trial_with_nested_segment_rows",
            "input_scope": "primary_v2_rankcloak_full_message_only",
            "primary_trial_count": trial_count,
            "primary_full_message_feature_rows": len(joined),
            "evaluator_score_rows_joined": len(evaluator_rows),
            "unmatched_primary_trials": 0,
            "duplicate_evaluator_trial_ids": 0,
            "source_record_hashes_recomputed": True,
            "evaluator_source_records_byte_identical_to_preprocessing": True,
            "evaluator_artifact_pins_verified": True,
            "models_config_sha256": file_sha256(MODELS_CONFIG),
            "evaluator_artifact_pins": dict(sorted(evaluator_artifact_pins.items())),
            "segments_as_independent_observations": False,
            "artifact_diagnostics": artifact_diagnostics,
            "score_scope": "source_full_message_replicated_across_nested_segment_rows_v1",
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "preprocessing_manifest": {
                "path": str(preprocessing_manifest),
                "sha256": file_sha256(preprocessing_manifest),
            },
            "input_files": sorted(
                provenance,
                key=lambda row: (str(row.get("role")), str(row.get("path"))),
            ),
            "outputs": {"features": output_identity},
        }
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, destination)
        committed = True
        return manifest
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "EvaluatorFeatureJoinError",
    "JOIN_MANIFEST_TYPE",
    "JOIN_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "OUTPUT_FILENAME",
    "join_primary_heldout_evaluator_features",
]
