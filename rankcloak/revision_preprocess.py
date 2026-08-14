"""Deterministic runner-to-analysis preprocessing for revision-v1 artifacts.

The runner intentionally writes rich, immutable JSON records.  This module is
the only adapter that turns those records into flat analysis inputs.  It
validates the run identity, plan, provenance manifests, durable attempt
history, evidence labels, reference joins, and sample-size arithmetic before
emitting any bytes.

No statistical analysis is performed here.  Missing measurements remain null
and receive an explicit availability label; they are never estimated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_artifacts import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    trial_ids_sha256,
    write_immutable_bytes,
    write_immutable_json,
    write_immutable_jsonl,
)


PREPROCESS_SCHEMA_VERSION = "2.0"
PAYLOAD_FIDELITY_CONTRACT_VERSION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
PAYLOAD_RECOVERY_SEMANTICS = "original_serialized_payload_bytes_sha256_v1"
DIRECT_SUBWORD_PROTOCOL = "direct_subword_calgacus"
UNAVAILABLE_NOT_RECORDED = "unavailable_not_recorded_by_runner_v1"
NOT_APPLICABLE = "not_applicable"

EVIDENCE_SMOKE = (
    "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling"
)
EVIDENCE_LIMITED = "exploratory_limited_not_for_confirmatory_pooling"
EVIDENCE_PRIMARY = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_ABLATION = (
    "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_MULTILINGUAL = (
    "secondary_supplementary_multilingual_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_ROBUSTNESS = (
    "confirmatory_supporting_robustness_v2_payload_fidelity_after_manifest_freeze"
)

EVIDENCE_FAMILIES = {
    EVIDENCE_SMOKE: "exploratory",
    EVIDENCE_LIMITED: "exploratory",
    EVIDENCE_PRIMARY: "confirmatory",
    EVIDENCE_ABLATION: "confirmatory",
    EVIDENCE_MULTILINGUAL: "secondary_supplementary",
    EVIDENCE_ROBUSTNESS: "confirmatory_supporting",
}

STAGE_STUDY_PHASES = {
    "smoke_v3": {
        "rankcloak": "smoke_v3_exploratory",
        "control": "ordinary_llm_control_smoke_v3",
    },
    "primary_v2": {
        "rankcloak": "primary_v2_confirmatory",
        "control": "ordinary_llm_control_primary_v2",
    },
    "ablation_v2": {
        "rankcloak": "ablation_v2_confirmatory",
        "reference": "ablation_v2_confirmatory",
    },
    "multilingual_v2": {
        "rankcloak": "multilingual_v2_secondary",
        "control": "ordinary_llm_control_multilingual_v2",
    },
    "robustness_v2": {
        "robustness_decode": "robustness_v2_confirmatory_supporting",
        "reference": "robustness_v2_confirmatory_supporting",
        "robustness_transform": "robustness_v2_transformation_generation",
    },
}

OUTPUT_FILENAMES = {
    "trials": "trials.csv",
    "features": "features.csv",
    "runtime": "runtime.csv",
    "failures": "failures.csv",
    "detector": "detector_corpus.jsonl",
    "unavailable": "unavailable.csv",
    "input_manifest": "preprocessing_input_manifest.json",
    "output_manifest": "preprocessing_output_manifest.json",
}

UNAVAILABLE_COLUMNS = (
    "preprocess_schema_version",
    "input_run_identity_sha256",
    "evidence_status",
    "evidence_family",
    "study_phase",
    "record_type",
    "work_kind",
    "trial_id",
    "control_id",
    "work_id",
    "source_trial_id",
    "source_record_sha256",
    "model_id",
    "source_model_id",
    "tokenizer_id",
    "tokenizer_revision",
    "tokenizer_artifact_sha256",
    "payload_name",
    "payload_class",
    "payload_split",
    "protocol_variant",
    "prompt_id",
    "prompt_category",
    "language",
    "token_filter",
    "ablation_factor",
    "ablation_level",
    "robustness_family",
    "replay_mode",
    "transformation_id",
    "reason_code",
    "reason",
    "safe_count",
    "stable_count",
    "vocabulary_size",
    "dependency_role",
    "dependency_record_type",
    "root_condition_work_id",
    "root_condition_trial_id",
    "root_condition_reason_code",
    "root_condition_model_id",
    "root_condition_tokenizer_id",
    "excluded_from_estimands",
)

TRIAL_COLUMNS = (
    "preprocess_schema_version",
    "input_run_identity_sha256",
    "evidence_status",
    "evidence_family",
    "study_phase",
    "record_type",
    "trial_id",
    "work_id",
    "source_trial_id",
    "source_record_sha256",
    "transform_work_id",
    "transformation_model_id",
    "transformation_record_sha256",
    "model_id",
    "source_model_id",
    "payload_name",
    "payload_class",
    "payload_split",
    "prompt_id",
    "prompt_category",
    "language",
    "protocol_variant",
    "representation_name",
    "codec_id",
    "alphabet_size_B",
    "segmented",
    "segment_count",
    "segment_size_ranks",
    "token_filter",
    "tail_policy",
    "leadin_tokens",
    "topic_schedule",
    "ablation_factor",
    "ablation_level",
    "robustness_family",
    "replay_mode",
    "transformation_id",
    "mitigation_id",
    "protocol_contract_revision",
    "result_schema_revision",
    "exact_rank_replay",
    "exact_payload_recovery",
    "recovery_outcome_semantics",
    "exact_recovery",
    "H_bits",
    "artifact_bit_length",
    "serialized_payload_bits",
    "effective_representation_bits_per_full_token",
    "effective_artifact_bits_per_full_token",
    "effective_serialized_bits_per_full_token",
    "effective_payload_rate",
    "forced_token_count",
    "tail_token_count",
    "full_token_count",
    "mean_log_probability",
    "unavailable_fields",
)

FEATURE_COLUMNS = (
    "preprocess_schema_version",
    "input_run_identity_sha256",
    "row_id",
    "trial_id",
    "work_id",
    "source_trial_id",
    "source_type",
    "view",
    "text_view",
    "segment_index",
    "text",
    "text_sha256",
    "prompt_text",
    "segment_prompt_id",
    "segment_prompt_category",
    "segment_prompt_text",
    "evidence_status",
    "evidence_family",
    "study_phase",
    "protocol_contract_revision",
    "result_schema_revision",
    "model_id",
    "source_model_id",
    "payload_name",
    "payload_class",
    "payload_split",
    "prompt_id",
    "prompt_category",
    "language",
    "protocol_variant",
    "representation_name",
    "codec_id",
    "control_view",
    "replay_mode",
    "transformation_id",
    "robustness_family",
    "transform_work_id",
    "transformation_model_id",
    "transformation_record_sha256",
    "token_count",
    "character_count",
    "mean_log_probability",
    "log_probability_availability",
    "effective_payload_rate",
    "nested_within_payload",
)

RUNTIME_COLUMNS = (
    "preprocess_schema_version",
    "input_run_identity_sha256",
    "trial_id",
    "work_id",
    "transform_work_id",
    "transformation_model_id",
    "transformation_record_sha256",
    "runtime_scope",
    "record_type",
    "evidence_status",
    "study_phase",
    "model_id",
    "payload_name",
    "payload_class",
    "protocol_variant",
    "hardware_id",
    "hardware_hash",
    "session_index",
    "model_load_seconds",
    "representation_seconds",
    "filter_setup_seconds",
    "generation_seconds",
    "encoding_seconds",
    "recovery_seconds",
    "decoding_seconds",
    "execution_seconds",
    "generation_tokens_per_second",
    "encoding_tokens_per_second",
    "decoding_tokens_per_second",
    "representation_bits_per_second",
    "payload_bits_per_second",
    "serialized_bits_per_second",
    "cover_tokens_per_payload_byte",
    "process_peak_rss_start_mib",
    "process_peak_rss_end_mib",
    "peak_ram_mib",
    "peak_gpu_memory_mib",
    "peak_ram_availability",
    "peak_gpu_memory_availability",
)

FAILURE_COLUMNS = (
    "preprocess_schema_version",
    "input_run_identity_sha256",
    "failure_id",
    "trial_id",
    "work_id",
    "source_trial_id",
    "record_type",
    "evidence_status",
    "study_phase",
    "model_id",
    "source_model_id",
    "payload_name",
    "payload_class",
    "protocol_variant",
    "robustness_family",
    "replay_mode",
    "transformation_id",
    "segment_index",
    "failure_category",
    "first_differing_position",
    "expected_token_id",
    "recovered_token_id",
    "expected_rank",
    "recovered_rank",
    "context_sha256",
    "boundary_start_offset",
    "boundary_end_offset",
    "expected_token_length",
    "recovered_token_length",
    "expected_rank_length",
    "recovered_rank_length",
    "execution_error_type",
    "execution_error_message",
    "divergence_fields_availability",
)


class RevisionPreprocessError(ValueError):
    """Raised when runner artifacts cannot support an auditable analysis table."""


@dataclass(frozen=True)
class PreprocessArtifacts:
    """Paths and row counts produced by :func:`preprocess_revision_results`."""

    output_dir: str
    files: Dict[str, str]
    row_counts: Dict[str, int]
    evidence_statuses: Tuple[str, ...]


@dataclass
class _RunShard:
    path: Path
    role: str
    identity: Dict[str, Any]
    plan: List[Dict[str, Any]]
    plan_by_id: Dict[str, Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    completions: Dict[str, Dict[str, Any]]
    model_manifest: Dict[str, Any]
    source_manifest: Dict[str, Any]
    payload_manifest: Dict[str, Any]
    evidence_status: str
    evidence_family: str
    stage: str
    model_id: str
    files: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    hardware_manifest: Optional[Dict[str, Any]]

    @property
    def identity_sha256(self) -> str:
        return str(self.identity["run_identity_sha256"])


def _require_mapping(value: object, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RevisionPreprocessError(f"{label} must be a JSON object")
    return dict(value)


def _load_json(path: Path, *, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RevisionPreprocessError(f"{label} is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RevisionPreprocessError(f"Invalid JSON in {label} {path}: {exc}") from exc
    return _require_mapping(value, label=label)


def _load_jsonl(path: Path, *, label: str, required: bool = True) -> List[Dict[str, Any]]:
    if not path.exists() and not required:
        return []
    if path.is_symlink() or not path.is_file():
        raise RevisionPreprocessError(f"{label} is missing or a symlink: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RevisionPreprocessError(
                    f"Invalid JSONL in {label} {path}:{line_number}: {exc}"
                ) from exc
            rows.append(_require_mapping(value, label=f"{label} row {line_number}"))
    return rows


def _require_fields(row: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise RevisionPreprocessError(f"{label} missing fields: {', '.join(missing)}")


def _nonempty(value: object, *, label: str) -> str:
    if value is None or not str(value).strip():
        raise RevisionPreprocessError(f"{label} must be non-empty")
    return str(value)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_record(path: Path, role: str, run_identity: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if run_identity is not None:
        row["run_identity_sha256"] = run_identity
    return row


def _parse_identity_args(identity: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    values = identity.get("command_line_args", [])
    if not isinstance(values, list):
        raise RevisionPreprocessError("run_identity.command_line_args must be a list")
    for raw in values:
        text = str(raw)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in result and result[key] != value:
                raise RevisionPreprocessError(f"Conflicting run-identity argument {key}")
            result[key] = value
    return result


def _verify_identity(identity: Mapping[str, Any], plan: Sequence[Mapping[str, Any]]) -> None:
    _require_fields(
        identity,
        (
            "manifest_type",
            "study_id",
            "config_manifest_sha256",
            "payload_manifest_sha256",
            "planned_trial_count",
            "planned_trial_ids_sha256",
            "model_artifacts",
            "command_line_args",
            "run_identity_sha256",
        ),
        label="run identity",
    )
    if identity["manifest_type"] != "revision_run_identity":
        raise RevisionPreprocessError("Unsupported run-identity manifest_type")
    embedded = str(identity["run_identity_sha256"])
    identity_body = dict(identity)
    identity_body.pop("run_identity_sha256", None)
    actual = canonical_json_sha256(identity_body)
    if embedded != actual:
        raise RevisionPreprocessError("run_identity_sha256 does not match identity content")
    ids = [_nonempty(row.get("work_id"), label="plan.work_id") for row in plan]
    if len(ids) != len(set(ids)):
        raise RevisionPreprocessError("Plan contains duplicate work_id values")
    if int(identity["planned_trial_count"]) != len(ids):
        raise RevisionPreprocessError("run identity planned_trial_count does not match plan")
    if str(identity["planned_trial_ids_sha256"]) != trial_ids_sha256(ids):
        raise RevisionPreprocessError("run identity planned ID hash does not match ordered plan")
    if not _is_sha256(identity["config_manifest_sha256"]):
        raise RevisionPreprocessError("run identity has invalid config manifest SHA-256")
    if not _is_sha256(identity["payload_manifest_sha256"]):
        raise RevisionPreprocessError("run identity has invalid payload manifest SHA-256")


def _verify_source_manifest(manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RevisionPreprocessError("source_manifest.files must be a non-empty list")
    if hashlib.sha256(canonical_json_bytes(files)).hexdigest() != manifest.get("files_sha256"):
        raise RevisionPreprocessError("source manifest file-list hash mismatch")
    paths: set[str] = set()
    for index, raw in enumerate(files):
        row = _require_mapping(raw, label=f"source manifest file {index}")
        path = _nonempty(row.get("path"), label="source manifest path")
        if path in paths:
            raise RevisionPreprocessError(f"Duplicate source manifest path: {path}")
        paths.add(path)
        if not _is_sha256(row.get("sha256")):
            raise RevisionPreprocessError(f"Invalid source hash for {path}")
        if int(row.get("size_bytes", -1)) < 0:
            raise RevisionPreprocessError(f"Invalid source size for {path}")


def _verify_model_manifest(
    manifest: Mapping[str, Any], identity: Mapping[str, Any], plan: Sequence[Mapping[str, Any]]
) -> str:
    artifacts = identity.get("model_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise RevisionPreprocessError("Each runner shard must bind exactly one model artifact")
    if canonical_json_sha256(artifacts[0]) != canonical_json_sha256(manifest):
        raise RevisionPreprocessError("model_manifest is not the artifact bound by run identity")
    configured = _require_mapping(manifest.get("configured_model"), label="configured model")
    model_id = _nonempty(configured.get("model_id"), label="configured model_id")
    configured_hash = configured.get("artifact_sha256")
    if not _is_sha256(configured_hash):
        raise RevisionPreprocessError("Configured model artifact SHA-256 is invalid")
    verification = _require_mapping(manifest.get("verification"), label="model verification")
    if verification.get("status") != "ok":
        raise RevisionPreprocessError("Runner model verification status is not ok")
    if not bool(verification.get("sha256_checked")):
        raise RevisionPreprocessError("Runner did not record a full model SHA-256 check")
    if verification.get("expected_sha256") != configured_hash or verification.get("actual_sha256") != configured_hash:
        raise RevisionPreprocessError("Runner model verification hash does not match configured artifact")
    configured_size = configured.get("artifact_size_bytes")
    if configured_size is None or int(verification.get("expected_size_bytes", -1)) != int(configured_size) or int(verification.get("actual_size_bytes", -1)) != int(configured_size):
        raise RevisionPreprocessError("Runner model verification size does not match configured artifact")
    plan_models = {_nonempty(row.get("model_id"), label="plan.model_id") for row in plan}
    if plan_models != {model_id}:
        raise RevisionPreprocessError(
            f"Plan models {sorted(plan_models)} do not match shard model {model_id}"
        )
    return model_id


def _verify_payload_manifest(manifest: Mapping[str, Any], identity: Mapping[str, Any], path: Path) -> None:
    if file_sha256(path) != identity.get("payload_manifest_sha256"):
        raise RevisionPreprocessError("payload manifest bytes do not match run identity")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RevisionPreprocessError("payload_manifest.records must be a list")
    if int(manifest.get("payload_count", -1)) != len(records):
        raise RevisionPreprocessError("payload manifest count does not match records")
    names = [str(row.get("payload_name")) for row in records if isinstance(row, dict)]
    if len(names) != len(records) or len(names) != len(set(names)):
        raise RevisionPreprocessError("payload manifest names must be unique objects")


def _validate_attempts(
    records: Sequence[Mapping[str, Any]],
    plan_by_id: Mapping[str, Mapping[str, Any]],
    *,
    strict_complete: bool,
) -> Dict[str, Dict[str, Any]]:
    attempts: set[Tuple[str, int]] = set()
    completions: Dict[str, Dict[str, Any]] = {}
    maximum_attempt: Dict[str, int] = {}
    for index, raw in enumerate(records):
        row = dict(raw)
        _require_fields(
            row,
            ("work_id", "evidence_status", "execution_status", "attempt_index", "record_type"),
            label=f"records row {index}",
        )
        work_id = _nonempty(row["work_id"], label="record.work_id")
        if work_id not in plan_by_id:
            raise RevisionPreprocessError(f"Record contains unplanned work_id {work_id}")
        try:
            attempt_index = int(row["attempt_index"])
        except (TypeError, ValueError) as exc:
            raise RevisionPreprocessError(f"Invalid attempt_index for {work_id}") from exc
        if attempt_index <= 0:
            raise RevisionPreprocessError(f"attempt_index must be positive for {work_id}")
        key = (work_id, attempt_index)
        if key in attempts:
            raise RevisionPreprocessError(f"Duplicate durable attempt {attempt_index} for {work_id}")
        attempts.add(key)
        maximum_attempt[work_id] = max(attempt_index, maximum_attempt.get(work_id, 0))
        task = plan_by_id[work_id]
        planned_evidence = str(task.get("evidence_status"))
        if str(row["evidence_status"]) != planned_evidence:
            raise RevisionPreprocessError(f"Evidence label differs between plan and record for {work_id}")
        planned_phase = _nonempty(
            task.get("study_phase"), label=f"plan {work_id} study_phase"
        )
        record_phase = _nonempty(
            row.get("study_phase"), label=f"record {work_id} study_phase"
        )
        if record_phase != planned_phase:
            raise RevisionPreprocessError(
                f"Study phase differs between plan and record for {work_id}"
            )
        status = str(row["execution_status"])
        if status == "completed":
            if work_id in completions:
                raise RevisionPreprocessError(f"Multiple durable completions for {work_id}")
            completions[work_id] = row
        elif status != "failed":
            raise RevisionPreprocessError(f"Invalid execution_status {status!r} for {work_id}")
    if strict_complete and set(completions) != set(plan_by_id):
        missing = sorted(set(plan_by_id) - set(completions))
        raise RevisionPreprocessError(
            f"Incomplete shard: {len(missing)} planned work units lack completion; first: "
            + ", ".join(missing[:3])
        )
    return completions


def _validate_record_types(
    plan_by_id: Mapping[str, Mapping[str, Any]], completions: Mapping[str, Mapping[str, Any]]
) -> None:
    expected = {
        "rankcloak": {"rankcloak_trial", "condition_unavailable"},
        "control": {"ordinary_control", "dependent_unavailable"},
        "robustness_decode": {"robustness_decode", "dependent_unavailable"},
        "robustness_transform": {
            "robustness_transform",
            "dependent_unavailable",
        },
        "reference": {
            "canonical_primary_reference",
            "robustness_reference",
            "dependent_unavailable",
        },
    }
    for work_id, record in completions.items():
        task = plan_by_id[work_id]
        kind = str(task.get("work_kind"))
        if kind not in expected:
            raise RevisionPreprocessError(f"Unknown plan work_kind {kind!r}")
        if str(record.get("record_type")) not in expected[kind]:
            raise RevisionPreprocessError(
                f"Record type {record.get('record_type')!r} is invalid for {kind} ({work_id})"
            )
        record_model_id = _coalesce(
            record.get("model_id"), record.get("transformation_model_id")
        )
        if str(record_model_id) != str(task.get("model_id")):
            raise RevisionPreprocessError(f"Record model_id differs from plan for {work_id}")
        record_trial = record.get("trial_id", record.get("control_id"))
        planned_trial = task.get("trial_id", task.get("control_id"))
        if record_trial is not None and planned_trial is not None and str(record_trial) != str(planned_trial):
            raise RevisionPreprocessError(f"Record trial/control ID differs from plan for {work_id}")


def _load_shard(path: Path, *, role: str, strict_complete: bool) -> _RunShard:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise RevisionPreprocessError(f"Run shard is missing or a symlink: {root}")
    paths = {
        "records": root / "records.jsonl",
        "plan": root / "plan.jsonl",
        "run_identity": root / "run_identity.json",
        "model_manifest": root / "model_manifest.json",
        "source_manifest": root / "source_manifest.json",
        "payload_manifest": root / "payload_manifest.json",
    }
    plan = _load_jsonl(paths["plan"], label="plan")
    identity = _load_json(paths["run_identity"], label="run identity")
    model_manifest = _load_json(paths["model_manifest"], label="model manifest")
    source_manifest = _load_json(paths["source_manifest"], label="source manifest")
    payload_manifest = _load_json(paths["payload_manifest"], label="payload manifest")
    records = _load_jsonl(paths["records"], label="records")
    _verify_identity(identity, plan)
    _verify_source_manifest(source_manifest)
    _verify_payload_manifest(payload_manifest, identity, paths["payload_manifest"])
    model_id = _verify_model_manifest(model_manifest, identity, plan)
    payload_rows = {
        str(row["payload_name"]): row
        for row in payload_manifest["records"]
        if isinstance(row, dict) and row.get("payload_name") is not None
    }
    for task in plan:
        payload_name = task.get("payload_name")
        if payload_name is not None and str(payload_name) not in payload_rows:
            raise RevisionPreprocessError(
                f"Plan payload {payload_name} is absent from the bound payload manifest"
            )
    plan_by_id = {str(row["work_id"]): row for row in plan}
    plan_evidence = {str(row.get("evidence_status")) for row in plan}
    if len(plan_evidence) != 1:
        raise RevisionPreprocessError("A runner shard must contain exactly one evidence label")
    evidence_status = next(iter(plan_evidence))
    if evidence_status not in EVIDENCE_FAMILIES:
        raise RevisionPreprocessError(f"Unrecognized evidence label: {evidence_status}")
    identity_args = _parse_identity_args(identity)
    stage = _nonempty(identity_args.get("stage"), label="run identity stage argument")
    allowed_stage_evidence = {
        "smoke_v3": {EVIDENCE_SMOKE},
        "primary_v2": {EVIDENCE_PRIMARY},
        "ablation_v2": {EVIDENCE_ABLATION},
        "multilingual_v2": {EVIDENCE_MULTILINGUAL},
        "robustness_v2": {EVIDENCE_ROBUSTNESS},
    }
    if stage not in allowed_stage_evidence or evidence_status not in allowed_stage_evidence[stage]:
        raise RevisionPreprocessError(
            f"Evidence label {evidence_status!r} is invalid for stage {stage!r}"
        )
    allowed_phases = STAGE_STUDY_PHASES[stage]
    for task in plan:
        work_id = task.get("work_id")
        work_kind = str(task.get("work_kind"))
        expected_phase = allowed_phases.get(work_kind)
        if expected_phase is None:
            raise RevisionPreprocessError(
                f"Work kind {work_kind!r} is invalid for superseding stage {stage!r}"
            )
        if task.get("study_phase") != expected_phase:
            raise RevisionPreprocessError(
                f"Plan work {work_id} study_phase is invalid for {stage}/{work_kind}"
            )
    if identity_args.get("model_id") != model_id:
        raise RevisionPreprocessError("run identity model_id argument does not match model manifest")
    if identity_args.get("evidence_status") != evidence_status:
        raise RevisionPreprocessError("run identity evidence_status does not match plan")
    if (
        identity.get("protocol_contract_revision")
        != PAYLOAD_FIDELITY_CONTRACT_VERSION
        or identity_args.get("protocol_contract_revision")
        != PAYLOAD_FIDELITY_CONTRACT_VERSION
    ):
        raise RevisionPreprocessError(
            "run identity is not bound to protocol_contract_revision="
            + PAYLOAD_FIDELITY_CONTRACT_VERSION
        )
    if (
        identity.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        or identity_args.get("result_schema_revision") != RESULT_SCHEMA_REVISION
    ):
        raise RevisionPreprocessError(
            "run identity is not bound to result_schema_revision="
            + RESULT_SCHEMA_REVISION
        )
    for task in plan:
        if task.get("protocol_contract_revision") != PAYLOAD_FIDELITY_CONTRACT_VERSION:
            raise RevisionPreprocessError(
                f"Plan work {task.get('work_id')} lacks the frozen protocol contract"
            )
        if task.get("result_schema_revision") != RESULT_SCHEMA_REVISION:
            raise RevisionPreprocessError(
                f"Plan work {task.get('work_id')} lacks the frozen result schema"
            )
    for record in records:
        if record.get("protocol_contract_revision") != PAYLOAD_FIDELITY_CONTRACT_VERSION:
            raise RevisionPreprocessError(
                f"Record {record.get('work_id')} lacks the frozen protocol contract"
            )
        if record.get("result_schema_revision") != RESULT_SCHEMA_REVISION:
            raise RevisionPreprocessError(
                f"Record {record.get('work_id')} lacks the frozen result schema"
            )
    expected_study = str(identity.get("study_id", ""))
    if not expected_study.endswith(f"/{stage}/{model_id}"):
        raise RevisionPreprocessError("run identity study_id does not match stage/model")
    completions = _validate_attempts(records, plan_by_id, strict_complete=strict_complete)
    _validate_record_types(plan_by_id, completions)

    files = [
        _file_record(path_value, name, str(identity["run_identity_sha256"]))
        for name, path_value in paths.items()
    ]
    events_path = root / "events.jsonl"
    events = _load_jsonl(events_path, label="events", required=False)
    if events_path.exists():
        files.append(_file_record(events_path, "events", str(identity["run_identity_sha256"])))
    hardware_path = root / "hardware_manifest.json"
    hardware = None
    if hardware_path.exists():
        hardware = _load_json(hardware_path, label="hardware manifest")
        files.append(_file_record(hardware_path, "hardware_manifest", str(identity["run_identity_sha256"])))
    input_results_path = root / "input_results_manifest.json"
    if input_results_path.exists():
        external = _load_json(input_results_path, label="input results manifest")
        if external.get("manifest_type") == "robustness_execution_inputs":
            body = dict(external)
            embedded_inputs_hash = body.pop("inputs_sha256", None)
            if embedded_inputs_hash != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
                raise RevisionPreprocessError("input_results_manifest inputs hash mismatch")
            children = [
                _require_mapping(external.get(name), label=f"input results {name}")
                for name in ("source_results", "transformation_results")
            ]
        else:
            children = [external]
        for child in children:
            external_files = child.get("files", [])
            if not isinstance(external_files, list) or hashlib.sha256(
                canonical_json_bytes(external_files)
            ).hexdigest() != child.get("files_sha256"):
                raise RevisionPreprocessError("input_results_manifest file-list hash mismatch")
            for raw in external_files:
                item = _require_mapping(raw, label="external source file")
                source_path = Path(_nonempty(item.get("path"), label="external source path"))
                if not source_path.is_file() or source_path.is_symlink():
                    raise RevisionPreprocessError(f"External source artifact unavailable: {source_path}")
                if source_path.stat().st_size != int(item.get("size_bytes", -1)) or file_sha256(source_path) != item.get("sha256"):
                    raise RevisionPreprocessError(f"External source artifact hash mismatch: {source_path}")
        files.append(_file_record(input_results_path, "input_results_manifest", str(identity["run_identity_sha256"])))

    return _RunShard(
        path=root,
        role=role,
        identity=identity,
        plan=plan,
        plan_by_id=plan_by_id,
        attempts=records,
        completions=completions,
        model_manifest=model_manifest,
        source_manifest=source_manifest,
        payload_manifest=payload_manifest,
        evidence_status=evidence_status,
        evidence_family=EVIDENCE_FAMILIES[evidence_status],
        stage=stage,
        model_id=model_id,
        files=files,
        events=events,
        hardware_manifest=hardware,
    )


def _record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def _mean(values: object) -> Optional[float]:
    if not isinstance(values, list) or not values:
        return None
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric)


def _bool_cell(value: object, *, label: str) -> int:
    if not isinstance(value, bool):
        raise RevisionPreprocessError(f"{label} must be a JSON boolean")
    return int(value)


def _is_direct_subword_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("protocol_variant") or "").strip() == DIRECT_SUBWORD_PROTOCOL


def _rank_replay_value(outcome: Mapping[str, Any], *, label: str) -> Optional[int]:
    for field in ("exact_rank_replay", "all_segment_ranks_exact"):
        if outcome.get(field) is not None:
            return _bool_cell(outcome.get(field), label=f"{label}.{field}")
    segments = outcome.get("segment_outcomes")
    if isinstance(segments, list) and segments:
        values = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise RevisionPreprocessError(
                    f"{label}.segment_outcomes[{index}] must be an object"
                )
            values.append(
                _bool_cell(
                    segment.get("exact_rank_replay"),
                    label=f"{label}.segment_outcomes[{index}].exact_rank_replay",
                )
            )
        return int(all(values))
    return None


def _flatten_recovery_contract(
    outcome: Mapping[str, Any],
    *,
    label: str,
    direct_subword: bool,
    protocol_contract_revision: object,
    result_schema_revision: object,
) -> Dict[str, Any]:
    """Normalize one replay outcome and reject ambiguous direct-subword rows."""

    if str(protocol_contract_revision or "") != PAYLOAD_FIDELITY_CONTRACT_VERSION:
        raise RevisionPreprocessError(
            f"{label} is not bound to protocol_contract_revision="
            f"{PAYLOAD_FIDELITY_CONTRACT_VERSION}"
        )
    if str(result_schema_revision or "") != RESULT_SCHEMA_REVISION:
        raise RevisionPreprocessError(
            f"{label} is not bound to result_schema_revision="
            f"{RESULT_SCHEMA_REVISION}"
        )
    alias = _bool_cell(outcome.get("exact_recovery"), label=f"{label}.exact_recovery")
    rank_replay = _rank_replay_value(outcome, label=label)
    decoded = outcome.get("decoded")
    decoded_mapping = decoded if isinstance(decoded, dict) else None

    decoded_payload_value = (
        decoded_mapping.get("exact_payload_recovery")
        if decoded_mapping is not None
        else None
    )
    decoded_payload = (
        _bool_cell(
            decoded_payload_value,
            label=f"{label}.decoded.exact_payload_recovery",
        )
        if decoded_payload_value is not None
        else None
    )
    if decoded_mapping is not None and decoded_mapping.get("exact_recovery") is not None:
        decoded_alias = _bool_cell(
            decoded_mapping.get("exact_recovery"),
            label=f"{label}.decoded.exact_recovery",
        )
        if decoded_payload is not None and decoded_alias != decoded_payload:
            raise RevisionPreprocessError(
                f"{label} decoded exact_recovery differs from exact_payload_recovery"
            )

    payload_value = outcome.get("exact_payload_recovery")
    if payload_value is not None:
        exact_payload = _bool_cell(
            payload_value, label=f"{label}.exact_payload_recovery"
        )
    elif rank_replay is not None and decoded_payload is not None:
        exact_payload = int(bool(rank_replay) and bool(decoded_payload))
    elif direct_subword:
        raise RevisionPreprocessError(
            f"{label} direct-subword outcome lacks exact_payload_recovery"
        )
    else:
        # Legacy bounded-codec rows already define exact_recovery as decoded
        # payload equality. This compatibility path is forbidden for direct
        # subword rows, where the historical field encoded rank replay only.
        exact_payload = alias

    semantics_value = (
        decoded_mapping.get("recovery_outcome_semantics")
        if decoded_mapping is not None
        else None
    )
    if semantics_value is None and not direct_subword:
        semantics_value = outcome.get("recovery_outcome_semantics")
    if semantics_value is None and direct_subword:
        raise RevisionPreprocessError(
            f"{label} direct-subword decoded outcome lacks recovery_outcome_semantics"
        )
    semantics = (
        str(semantics_value).strip()
        if semantics_value is not None
        else PAYLOAD_RECOVERY_SEMANTICS
    )
    if semantics != PAYLOAD_RECOVERY_SEMANTICS:
        raise RevisionPreprocessError(
            f"{label} recovery_outcome_semantics must equal "
            f"{PAYLOAD_RECOVERY_SEMANTICS!r}"
        )

    if direct_subword:
        if rank_replay is None:
            raise RevisionPreprocessError(
                f"{label} direct-subword outcome lacks exact_rank_replay"
            )
        if decoded_mapping is None or decoded_payload is None:
            raise RevisionPreprocessError(
                f"{label} direct-subword outcome lacks decoded payload fidelity"
            )
        original_hash = str(decoded_mapping.get("original_payload_sha256") or "")
        recovered_hash = str(decoded_mapping.get("recovered_payload_sha256") or "")
        for field, digest in (
            ("original_payload_sha256", original_hash),
            ("recovered_payload_sha256", recovered_hash),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise RevisionPreprocessError(
                    f"{label}.decoded.{field} is not a lowercase SHA-256"
                )
        if decoded_payload != int(original_hash == recovered_hash):
            raise RevisionPreprocessError(
                f"{label} decoded payload flag disagrees with payload SHA-256 equality"
            )

    if alias != exact_payload:
        raise RevisionPreprocessError(
            f"{label} exact_recovery compatibility alias differs from "
            "exact_payload_recovery"
        )
    return {
        "protocol_contract_revision": PAYLOAD_FIDELITY_CONTRACT_VERSION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "exact_rank_replay": rank_replay,
        "exact_payload_recovery": exact_payload,
        "recovery_outcome_semantics": semantics,
        "exact_recovery": exact_payload,
    }


def _validate_flattened_recovery_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    direct_count = 0
    for index, row in enumerate(rows):
        label = f"flattened trial row {index + 1}"
        if row.get("protocol_contract_revision") != PAYLOAD_FIDELITY_CONTRACT_VERSION:
            raise RevisionPreprocessError(f"{label} lacks the payload-fidelity-v2 contract")
        if row.get("result_schema_revision") != RESULT_SCHEMA_REVISION:
            raise RevisionPreprocessError(f"{label} lacks the payload-aware-v2 schema")
        if row.get("recovery_outcome_semantics") != PAYLOAD_RECOVERY_SEMANTICS:
            raise RevisionPreprocessError(f"{label} has ambiguous recovery semantics")
        if row.get("exact_recovery") != row.get("exact_payload_recovery"):
            raise RevisionPreprocessError(f"{label} recovery alias is inconsistent")
        if _is_direct_subword_row(row):
            direct_count += 1
            if row.get("exact_rank_replay") not in (0, 1):
                raise RevisionPreprocessError(
                    f"{label} direct-subword rank replay diagnostic is missing"
                )
    return direct_count


def _stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}__{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def _coalesce(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _representation_name(record: Mapping[str, Any], task: Mapping[str, Any]) -> Optional[str]:
    representation = record.get("representation")
    if isinstance(representation, dict) and representation.get("name") is not None:
        return str(representation["name"])
    value = task.get("representation_name")
    if value is not None:
        text = str(value)
        mapping = {
            "raw_subword_direct": "direct_subword",
            "ascii_bytes_fixed_radix": (
                "ascii_b8" if int(task.get("alphabet_size") or 0) == 8 else "ascii_b16"
            ),
            "raw_hex_nibbles": "hex_nibble",
        }
        return mapping.get(text, text)
    return None


def _common_trial(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    shard: _RunShard,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source_record = source or record
    representation = _representation_name(source_record, task)
    prompt_id = _coalesce(record.get("prompt_id"), task.get("prompt_id"), source_record.get("prompt_id"))
    protocol = _coalesce(task.get("protocol_variant"), record.get("protocol_variant"), source_record.get("protocol_variant"))
    unavailable: List[str] = []
    quality = source_record.get("quality") if isinstance(source_record.get("quality"), dict) else {}
    mean_logp = quality.get("mean_forced_token_log_probability")
    if mean_logp is None:
        unavailable.append("mean_log_probability")
    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "evidence_status": record.get("evidence_status", shard.evidence_status),
        "evidence_family": EVIDENCE_FAMILIES.get(str(record.get("evidence_status", shard.evidence_status))),
        "study_phase": _coalesce(record.get("study_phase"), task.get("study_phase")),
        "record_type": record.get("record_type"),
        "trial_id": _coalesce(record.get("trial_id"), record.get("control_id"), task.get("trial_id"), task.get("control_id")),
        "work_id": record.get("work_id"),
        "source_trial_id": _coalesce(record.get("source_trial_id"), task.get("source_trial_id")),
        "source_record_sha256": record.get("source_record_sha256"),
        "transform_work_id": record.get("transform_work_id"),
        "transformation_model_id": record.get("transformation_model_id"),
        "transformation_record_sha256": record.get("transformation_record_sha256"),
        "model_id": _coalesce(record.get("model_id"), record.get("transformation_model_id")),
        "source_model_id": _coalesce(record.get("source_model_id"), source_record.get("model_id")),
        "payload_name": _coalesce(record.get("payload_name"), task.get("payload_name"), source_record.get("payload_name")),
        "payload_class": _coalesce(record.get("payload_class"), task.get("payload_class"), source_record.get("payload_class")),
        "payload_split": _coalesce(record.get("payload_split"), task.get("payload_split"), source_record.get("payload_split")),
        "prompt_id": prompt_id,
        "prompt_category": _coalesce(record.get("prompt_category"), task.get("prompt_category"), source_record.get("prompt_category")),
        "language": _coalesce(record.get("language"), task.get("language"), source_record.get("language")),
        "protocol_variant": protocol,
        "representation_name": representation,
        "codec_id": representation,
        "alphabet_size_B": _coalesce(source_record.get("alphabet_size_B"), task.get("alphabet_size")),
        "segmented": source_record.get("segmented"),
        "segment_count": source_record.get("segment_count"),
        "segment_size_ranks": source_record.get("segment_size_ranks"),
        "token_filter": _coalesce(record.get("token_filter"), task.get("token_filter"), source_record.get("token_filter")),
        "tail_policy": _coalesce(task.get("tail_policy"), source_record.get("tail_policy")),
        "leadin_tokens": _coalesce(task.get("leadin_tokens"), source_record.get("leadin_tokens")),
        "topic_schedule": _coalesce(task.get("topic_schedule"), source_record.get("topic_schedule")),
        "ablation_factor": task.get("ablation_factor"),
        "ablation_level": task.get("ablation_level"),
        "robustness_family": record.get("robustness_family"),
        "H_bits": _coalesce(source_record.get("H_bits"), source_record.get("representation_source_bits")),
        "artifact_bit_length": source_record.get("artifact_bit_length"),
        "serialized_payload_bits": source_record.get("serialized_payload_bits"),
        "effective_representation_bits_per_full_token": source_record.get("effective_bits_per_full_token"),
        "effective_artifact_bits_per_full_token": source_record.get("effective_artifact_bits_per_full_token"),
        "effective_serialized_bits_per_full_token": source_record.get("effective_serialized_bits_per_full_token"),
        "effective_payload_rate": _coalesce(
            source_record.get("effective_artifact_bits_per_full_token"),
            source_record.get("effective_bits_per_full_token"),
        ),
        "forced_token_count": source_record.get("forced_token_count"),
        "tail_token_count": source_record.get("tail_token_count"),
        "full_token_count": source_record.get("full_token_count"),
        "mean_log_probability": mean_logp,
        "unavailable_fields": "|".join(unavailable),
    }


def _rankcloak_trial_rows(record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    outcomes = (
        ("saved_token_id_replay", "saved_token_ids", True),
        ("text_retokenization_replay", "detokenized_text_retokenized", False),
        ("greedy_leadin_replay", "greedy_leadin_regeneration", False),
    )
    for field, replay_mode, always in outcomes:
        outcome = record.get(field)
        if not isinstance(outcome, dict):
            if always:
                raise RevisionPreprocessError(f"RankCloak record {record.get('work_id')} lacks {field}")
            continue
        if not always and not bool(outcome.get("executed")):
            continue
        row = _common_trial(record, task, shard)
        row.update(
            {
                "replay_mode": replay_mode,
                "transformation_id": "unmodified",
                "mitigation_id": "none",
                **_flatten_recovery_contract(
                    outcome,
                    label=f"{record.get('work_id')}.{field}",
                    direct_subword=_is_direct_subword_row(row),
                    protocol_contract_revision=record.get(
                        "protocol_contract_revision"
                    ),
                    result_schema_revision=record.get("result_schema_revision"),
                ),
            }
        )
        rows.append(row)
    return rows


def _resolved_trial_row(
    record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard, source: Mapping[str, Any]
) -> Dict[str, Any]:
    source_outcome = source.get("saved_token_id_replay")
    if not isinstance(source_outcome, dict):
        raise RevisionPreprocessError("Reference source lacks saved-token replay outcome")
    row = _common_trial(record, task, shard, source=source)
    row.update(
        {
            "replay_mode": str(record.get("replay_mode", "saved_token_ids")),
            "transformation_id": str(record.get("transformation_id", "unmodified")),
            "mitigation_id": (
                "canonicalization_pipeline_v1"
                if record.get("robustness_family") == "limited_mitigation"
                else "none"
            ),
            **_flatten_recovery_contract(
                source_outcome,
                label=f"{record.get('work_id')}.source.saved_token_id_replay",
                direct_subword=_is_direct_subword_row(row),
                protocol_contract_revision=source.get(
                    "protocol_contract_revision"
                ),
                result_schema_revision=source.get("result_schema_revision"),
            ),
        }
    )
    return row


def _robustness_trial_row(
    record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard, source: Mapping[str, Any]
) -> Dict[str, Any]:
    row = _common_trial(record, task, shard, source=source)
    row.update(
        {
            "replay_mode": record.get("replay_mode"),
            "transformation_id": record.get("transformation_id"),
            "mitigation_id": (
                "canonicalization_pipeline_v1"
                if record.get("robustness_family") == "limited_mitigation"
                else "none"
            ),
            **_flatten_recovery_contract(
                record,
                label=str(record.get("work_id")),
                direct_subword=_is_direct_subword_row(row),
                protocol_contract_revision=record.get(
                    "protocol_contract_revision"
                ),
                result_schema_revision=record.get("result_schema_revision"),
            ),
        }
    )
    return row


def _joined_prompt_text(segments: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for segment in segments:
        prompt = segment.get("prompt")
        if isinstance(prompt, dict):
            values.append(str(prompt.get("prompt_text", "")))
    return "\n\n".join(values)


def _feature_base(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    shard: _RunShard,
    *,
    trial_id: str,
    source: Mapping[str, Any],
    source_type: str,
    view: str,
    segment_index: int,
    text: str,
    prompt_text: str,
    segment_prompt: Mapping[str, Any],
    token_count: Optional[int],
    mean_log_probability: Optional[float],
) -> Dict[str, Any]:
    representation = _representation_name(source, task)
    row_id = _stable_id("feature", shard.identity_sha256, trial_id, view, segment_index)
    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "row_id": row_id,
        "trial_id": trial_id,
        "work_id": record.get("work_id"),
        "source_trial_id": _coalesce(record.get("source_trial_id"), task.get("source_trial_id")),
        "source_type": source_type,
        "view": view,
        "text_view": view,
        "segment_index": segment_index,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_text": prompt_text,
        "segment_prompt_id": segment_prompt.get("prompt_id"),
        "segment_prompt_category": segment_prompt.get("prompt_category"),
        "segment_prompt_text": segment_prompt.get("prompt_text"),
        "evidence_status": record.get("evidence_status", shard.evidence_status),
        "evidence_family": EVIDENCE_FAMILIES.get(str(record.get("evidence_status", shard.evidence_status))),
        "study_phase": _coalesce(record.get("study_phase"), task.get("study_phase")),
        "protocol_contract_revision": record.get("protocol_contract_revision"),
        "result_schema_revision": record.get("result_schema_revision"),
        "model_id": record.get("model_id"),
        "source_model_id": _coalesce(record.get("source_model_id"), source.get("model_id")),
        "payload_name": _coalesce(record.get("payload_name"), task.get("payload_name"), source.get("payload_name")),
        "payload_class": _coalesce(record.get("payload_class"), task.get("payload_class"), source.get("payload_class")),
        "payload_split": _coalesce(record.get("payload_split"), task.get("payload_split"), source.get("payload_split")),
        "prompt_id": _coalesce(record.get("prompt_id"), task.get("prompt_id"), source.get("prompt_id")),
        "prompt_category": _coalesce(record.get("prompt_category"), task.get("prompt_category"), source.get("prompt_category")),
        "language": _coalesce(record.get("language"), task.get("language"), source.get("language")),
        "protocol_variant": _coalesce(task.get("protocol_variant"), source.get("protocol_variant")),
        "representation_name": representation,
        "codec_id": representation,
        "control_view": record.get("control_view"),
        "replay_mode": record.get("replay_mode"),
        "transformation_id": record.get("transformation_id", "unmodified"),
        "robustness_family": record.get("robustness_family"),
        "transform_work_id": record.get("transform_work_id"),
        "transformation_model_id": record.get("transformation_model_id"),
        "transformation_record_sha256": record.get("transformation_record_sha256"),
        "token_count": token_count,
        "character_count": len(text),
        "mean_log_probability": mean_log_probability,
        "log_probability_availability": (
            "recorded" if mean_log_probability is not None else UNAVAILABLE_NOT_RECORDED
        ),
        "effective_payload_rate": (
            source.get("effective_bits_per_full_token") if view == "full_message" else None
        ),
        "nested_within_payload": 1,
    }


def _rankcloak_feature_rows(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    shard: _RunShard,
    *,
    source: Optional[Mapping[str, Any]] = None,
    trial_id: Optional[str] = None,
    source_type: str = "rankcloak",
) -> List[Dict[str, Any]]:
    source_record = source or record
    segments = source_record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RevisionPreprocessError(f"RankCloak source {source_record.get('work_id')} has no segments")
    aggregate_prompt = _joined_prompt_text(segments)
    output: List[Dict[str, Any]] = []
    target_trial_id = str(trial_id or record.get("trial_id"))
    for index, raw_segment in enumerate(segments):
        segment = _require_mapping(raw_segment, label="RankCloak segment")
        segment_index = int(segment.get("segment_index", index))
        prompt = _require_mapping(segment.get("prompt"), label="segment prompt")
        forced_logp = _mean(segment.get("forced_log_probabilities"))
        all_logp: List[Any] = []
        for field in ("leadin_log_probabilities", "forced_log_probabilities", "tail_log_probabilities"):
            value = segment.get(field)
            if isinstance(value, list):
                all_logp.extend(value)
        full_logp = _mean(all_logp)
        forced_text = str(segment.get("forced_text", ""))
        full_text = str(segment.get("full_text", ""))
        output.append(
            _feature_base(
                record,
                task,
                shard,
                trial_id=target_trial_id,
                source=source_record,
                source_type=source_type,
                view="forced_span",
                segment_index=segment_index,
                text=forced_text,
                prompt_text=aggregate_prompt,
                segment_prompt=prompt,
                token_count=len(segment.get("forced_token_ids", [])),
                mean_log_probability=forced_logp,
            )
        )
        output.append(
            _feature_base(
                record,
                task,
                shard,
                trial_id=target_trial_id,
                source=source_record,
                source_type=source_type,
                view="full_message",
                segment_index=segment_index,
                text=full_text,
                prompt_text=aggregate_prompt,
                segment_prompt=prompt,
                token_count=len(segment.get("full_token_ids", [])),
                mean_log_probability=full_logp,
            )
        )
    return output


def _control_feature_row(
    record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard, source: Mapping[str, Any]
) -> Dict[str, Any]:
    generation = _require_mapping(record.get("generation"), label="control generation")
    segments = source.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RevisionPreprocessError("Control source lacks segment prompt metadata")
    first_prompt = _require_mapping(segments[0].get("prompt"), label="control source prompt")
    view = str(record.get("control_view"))
    expected_field = "full_token_count" if view == "full_message" else "forced_token_count"
    expected_length = source.get(expected_field)
    actual_length = record.get("full_token_count")
    if expected_length is None or int(expected_length) != int(actual_length):
        raise RevisionPreprocessError(
            f"Control {record.get('work_id')} length does not match {expected_field} of source"
        )
    if int(generation.get("target_token_count", -1)) != int(actual_length):
        raise RevisionPreprocessError(f"Control {record.get('work_id')} target length mismatch")
    text = str(record.get("full_text", ""))
    mean_logp = _mean(generation.get("token_log_probabilities"))
    row = _feature_base(
        record,
        task,
        shard,
        trial_id=str(record.get("control_id", record.get("work_id"))),
        source=source,
        source_type="ordinary_llm_control",
        view=view,
        segment_index=0,
        text=text,
        prompt_text=str(first_prompt.get("prompt_text", "")),
        segment_prompt=first_prompt,
        token_count=int(actual_length),
        mean_log_probability=mean_logp,
    )
    row["protocol_variant"] = "ordinary_llm_control"
    row["representation_name"] = _representation_name(source, task)
    row["codec_id"] = _representation_name(source, task)
    row["effective_payload_rate"] = None
    return row


def _robustness_feature_rows(
    record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard, source: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    outcomes = record.get("segment_outcomes")
    if record.get("record_type") == "robustness_reference":
        segments = source.get("segments")
        if not isinstance(segments, list):
            raise RevisionPreprocessError("Robustness reference source lacks segments")
        outcomes = [
            {
                "segment_index": segment.get("segment_index", index),
                "prompt": segment.get("prompt"),
                "observed_text": segment.get("full_text"),
                "observed_full_token_ids": segment.get("full_token_ids"),
            }
            for index, segment in enumerate(segments)
        ]
    if not isinstance(outcomes, list) or not outcomes:
        raise RevisionPreprocessError(f"Robustness record {record.get('work_id')} lacks segment outcomes")
    prompt_text = _joined_prompt_text(outcomes)
    output = []
    for index, raw in enumerate(outcomes):
        outcome = _require_mapping(raw, label="robustness segment outcome")
        prompt = _require_mapping(outcome.get("prompt"), label="robustness prompt")
        text = str(outcome.get("observed_text", ""))
        row = _feature_base(
            record,
            task,
            shard,
            trial_id=str(record.get("trial_id")),
            source=source,
            source_type="robustness_observed_text",
            view="transformed_full_message",
            segment_index=int(outcome.get("segment_index", index)),
            text=text,
            prompt_text=prompt_text,
            segment_prompt=prompt,
            token_count=len(outcome.get("observed_full_token_ids", [])),
            mean_log_probability=None,
        )
        row["effective_payload_rate"] = None
        output.append(row)
    return output


def _hardware_fields(shard: _RunShard) -> Tuple[str, str]:
    if shard.hardware_manifest is None:
        return UNAVAILABLE_NOT_RECORDED, UNAVAILABLE_NOT_RECORDED
    hardware_path = shard.path / "hardware_manifest.json"
    digest = file_sha256(hardware_path)
    selected = shard.hardware_manifest.get("selected_gpu_uuid")
    if selected:
        return str(selected), digest
    machine = shard.hardware_manifest.get("machine")
    return (str(machine) if machine else f"hardware_manifest_{digest[:16]}", digest)


def _runtime_base(record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard) -> Dict[str, Any]:
    hardware_id, hardware_hash = _hardware_fields(shard)
    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "trial_id": _coalesce(record.get("trial_id"), record.get("control_id"), record.get("work_id")),
        "work_id": record.get("work_id"),
        "transform_work_id": record.get("transform_work_id"),
        "transformation_model_id": record.get("transformation_model_id"),
        "transformation_record_sha256": record.get("transformation_record_sha256"),
        "runtime_scope": "trial",
        "record_type": record.get("record_type"),
        "evidence_status": record.get("evidence_status", shard.evidence_status),
        "study_phase": _coalesce(record.get("study_phase"), task.get("study_phase")),
        "model_id": _coalesce(record.get("model_id"), record.get("transformation_model_id")),
        "payload_name": _coalesce(record.get("payload_name"), task.get("payload_name")),
        "payload_class": _coalesce(record.get("payload_class"), task.get("payload_class")),
        "protocol_variant": _coalesce(record.get("protocol_variant"), task.get("protocol_variant"), "ordinary_llm_control" if record.get("record_type") == "ordinary_control" else None),
        "hardware_id": hardware_id,
        "hardware_hash": hardware_hash,
        "session_index": None,
        "model_load_seconds": None,
        "representation_seconds": None,
        "filter_setup_seconds": None,
        "generation_seconds": None,
        "encoding_seconds": None,
        "recovery_seconds": None,
        "decoding_seconds": None,
        "execution_seconds": record.get("execution_seconds"),
        "generation_tokens_per_second": None,
        "encoding_tokens_per_second": None,
        "decoding_tokens_per_second": None,
        "representation_bits_per_second": None,
        "payload_bits_per_second": None,
        "serialized_bits_per_second": None,
        "cover_tokens_per_payload_byte": None,
        "process_peak_rss_start_mib": None,
        "process_peak_rss_end_mib": None,
        "peak_ram_mib": None,
        "peak_gpu_memory_mib": None,
        "peak_ram_availability": UNAVAILABLE_NOT_RECORDED,
        "peak_gpu_memory_availability": UNAVAILABLE_NOT_RECORDED,
    }


def _runtime_row(record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard) -> Dict[str, Any]:
    row = _runtime_base(record, task, shard)
    if record.get("record_type") == "rankcloak_trial":
        timing = _require_mapping(record.get("timing"), label="RankCloak timing")
        row.update(
            {
                "representation_seconds": timing.get("representation_seconds"),
                "filter_setup_seconds": timing.get("filter_setup_seconds"),
                "generation_seconds": timing.get("generation_seconds"),
                "encoding_seconds": timing.get("encoding_seconds"),
                "recovery_seconds": timing.get("saved_token_id_replay_seconds"),
                "decoding_seconds": timing.get("supported_decoding_seconds"),
                "generation_tokens_per_second": timing.get("cover_tokens_per_generation_second"),
                "encoding_tokens_per_second": (
                    record.get("full_token_count") / timing.get("encoding_seconds")
                    if isinstance(record.get("full_token_count"), int)
                    and isinstance(timing.get("encoding_seconds"), (int, float))
                    and timing.get("encoding_seconds") > 0
                    else None
                ),
                "decoding_tokens_per_second": timing.get("forced_tokens_per_supported_decoding_second"),
                "representation_bits_per_second": timing.get("representation_bits_per_encoding_second"),
                "payload_bits_per_second": timing.get("payload_bits_per_encoding_second"),
                "serialized_bits_per_second": timing.get("serialized_bits_per_encoding_second"),
                "cover_tokens_per_payload_byte": record.get("cover_tokens_per_payload_display_byte"),
                "process_peak_rss_start_mib": (
                    timing.get("process_peak_rss_bytes_at_trial_start") / (1024 * 1024)
                    if isinstance(timing.get("process_peak_rss_bytes_at_trial_start"), (int, float))
                    else None
                ),
                "process_peak_rss_end_mib": (
                    timing.get("process_peak_rss_bytes_at_trial_end") / (1024 * 1024)
                    if isinstance(timing.get("process_peak_rss_bytes_at_trial_end"), (int, float))
                    else None
                ),
                "peak_ram_mib": (
                    timing.get("process_peak_rss_bytes_at_trial_end") / (1024 * 1024)
                    if isinstance(timing.get("process_peak_rss_bytes_at_trial_end"), (int, float))
                    else None
                ),
                "peak_ram_availability": (
                    "recorded_process_cumulative_peak_rss"
                    if isinstance(timing.get("process_peak_rss_bytes_at_trial_end"), (int, float))
                    else UNAVAILABLE_NOT_RECORDED
                ),
            }
        )
    elif record.get("record_type") == "ordinary_control":
        row["runtime_scope"] = "ordinary_control_generation"
        row["generation_seconds"] = record.get("execution_seconds")
        seconds = record.get("execution_seconds")
        count = record.get("full_token_count")
        if isinstance(seconds, (int, float)) and seconds > 0 and isinstance(count, int):
            row["generation_tokens_per_second"] = count / seconds
    elif record.get("record_type") == "robustness_decode":
        row["runtime_scope"] = "robustness_decode"
        row["recovery_seconds"] = record.get("execution_seconds")
        row["decoding_seconds"] = record.get("execution_seconds")
    elif record.get("record_type") == "robustness_transform":
        row["runtime_scope"] = "robustness_transform_generation"
        row["generation_seconds"] = record.get("execution_seconds")
        outputs = record.get("segment_outputs")
        token_count = sum(
            len(output.get("token_ids", []))
            for output in outputs
            if isinstance(output, dict)
        ) if isinstance(outputs, list) else 0
        if isinstance(record.get("execution_seconds"), (int, float)) and record.get("execution_seconds") > 0:
            row["generation_tokens_per_second"] = token_count / record.get("execution_seconds")
    elif record.get("record_type") == "condition_unavailable":
        row["runtime_scope"] = "condition_unavailable_no_execution"
    elif record.get("record_type") == "dependent_unavailable":
        row["runtime_scope"] = "dependent_unavailable_no_execution"
    else:
        row["runtime_scope"] = "reference_no_execution"
    return row


def _load_runtime_rows(shard: _RunShard) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hardware_id, hardware_hash = _hardware_fields(shard)
    model_events = [event for event in shard.events if event.get("event") == "model_loaded"]
    for index, event in enumerate(model_events):
        load_seconds = event.get("model_load_seconds")
        if not isinstance(load_seconds, (int, float)) or load_seconds < 0:
            raise RevisionPreprocessError("model_loaded event has invalid model_load_seconds")
        rows.append(
            {
                "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
                "input_run_identity_sha256": shard.identity_sha256,
                "trial_id": f"model_load::{shard.identity_sha256}::{index}",
                "work_id": f"model_load::{index}",
                "runtime_scope": "model_load_session",
                "record_type": "model_load_event",
                "evidence_status": shard.evidence_status,
                "study_phase": f"{shard.stage}_model_load",
                "model_id": shard.model_id,
                "payload_name": "not_applicable_model_load",
                "payload_class": "not_applicable_model_load",
                "protocol_variant": "not_applicable_model_load",
                "hardware_id": hardware_id,
                "hardware_hash": hardware_hash,
                "session_index": index,
                "model_load_seconds": load_seconds,
                "representation_seconds": None,
                "filter_setup_seconds": None,
                "generation_seconds": None,
                "encoding_seconds": None,
                "recovery_seconds": None,
                "decoding_seconds": None,
                "execution_seconds": None,
                "generation_tokens_per_second": None,
                "encoding_tokens_per_second": None,
                "decoding_tokens_per_second": None,
                "payload_bits_per_second": None,
                "cover_tokens_per_payload_byte": None,
                "peak_ram_mib": None,
                "peak_gpu_memory_mib": None,
                "peak_ram_availability": UNAVAILABLE_NOT_RECORDED,
                "peak_gpu_memory_availability": UNAVAILABLE_NOT_RECORDED,
            }
        )
    memory_events = [event for event in shard.events if event.get("event") == "memory_profile"]
    for index, event in enumerate(memory_events):
        sampled_rss = event.get("process_peak_rss_bytes_sampled")
        os_rss = event.get("process_peak_rss_bytes_os_high_water")
        gpu_peak = event.get("selected_gpu_peak_used_memory_mib_sampled")
        rows.append(
            {
                "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
                "input_run_identity_sha256": shard.identity_sha256,
                "trial_id": f"memory_profile::{shard.identity_sha256}::{index}",
                "work_id": f"memory_profile::{index}",
                "runtime_scope": "model_shard_memory_profile",
                "record_type": "memory_profile_event",
                "evidence_status": shard.evidence_status,
                "study_phase": f"{shard.stage}_memory_profile",
                "model_id": shard.model_id,
                "payload_name": "not_applicable_memory_profile",
                "payload_class": "not_applicable_memory_profile",
                "protocol_variant": "not_applicable_memory_profile",
                "hardware_id": hardware_id,
                "hardware_hash": hardware_hash,
                "session_index": index,
                "peak_ram_mib": (
                    os_rss / (1024 * 1024)
                    if isinstance(os_rss, (int, float))
                    else sampled_rss / (1024 * 1024)
                    if isinstance(sampled_rss, (int, float))
                    else None
                ),
                "peak_gpu_memory_mib": gpu_peak,
                "peak_ram_availability": (
                    "recorded_os_high_water_and_sampled_current_rss"
                    if isinstance(os_rss, (int, float)) and isinstance(sampled_rss, (int, float))
                    else "recorded_partial_memory_profile"
                    if isinstance(os_rss, (int, float)) or isinstance(sampled_rss, (int, float))
                    else UNAVAILABLE_NOT_RECORDED
                ),
                "peak_gpu_memory_availability": (
                    str(event.get("gpu_measurement_scope"))
                    if isinstance(gpu_peak, (int, float))
                    else UNAVAILABLE_NOT_RECORDED
                ),
            }
        )
    return rows


def _failure_row(
    failure: Mapping[str, Any],
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    shard: _RunShard,
    replay_mode: str,
    transformation_id: str,
) -> Dict[str, Any]:
    rank = failure.get("first_rank_divergence")
    token = failure.get("first_token_divergence")
    rank_map = dict(rank) if isinstance(rank, dict) else {}
    token_map = dict(token) if isinstance(token, dict) else {}
    required = (
        "failure_category",
        "expected_token_id",
        "recovered_token_id",
        "expected_rank",
        "recovered_rank",
        "context_sha256",
    )
    missing = [field for field in required if field not in failure]
    if missing:
        raise RevisionPreprocessError(
            f"Failure for {record.get('work_id')} lacks required fields: {', '.join(missing)}"
        )
    first_position = _coalesce(
        failure.get("first_differing_position"),
        rank_map.get("position_zero_based"),
        token_map.get("position_zero_based"),
    )
    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "failure_id": _stable_id("failure", record.get("work_id"), replay_mode, transformation_id),
        "trial_id": _coalesce(record.get("trial_id"), task.get("trial_id")),
        "work_id": record.get("work_id"),
        "source_trial_id": _coalesce(record.get("source_trial_id"), task.get("source_trial_id")),
        "record_type": record.get("record_type"),
        "evidence_status": record.get("evidence_status"),
        "study_phase": _coalesce(record.get("study_phase"), task.get("study_phase")),
        "model_id": record.get("model_id"),
        "source_model_id": record.get("source_model_id"),
        "payload_name": _coalesce(record.get("payload_name"), task.get("payload_name")),
        "payload_class": _coalesce(record.get("payload_class"), task.get("payload_class")),
        "protocol_variant": _coalesce(record.get("protocol_variant"), task.get("protocol_variant")),
        "robustness_family": record.get("robustness_family"),
        "replay_mode": replay_mode,
        "transformation_id": transformation_id,
        "segment_index": failure.get("segment_index"),
        "failure_category": failure.get("failure_category"),
        "first_differing_position": first_position,
        "expected_token_id": failure.get("expected_token_id"),
        "recovered_token_id": failure.get("recovered_token_id"),
        "expected_rank": failure.get("expected_rank"),
        "recovered_rank": failure.get("recovered_rank"),
        "context_sha256": failure.get("context_sha256"),
        "boundary_start_offset": _coalesce(failure.get("boundary_start_offset"), failure.get("boundary_start")),
        "boundary_end_offset": _coalesce(failure.get("boundary_end_offset"), failure.get("boundary_stop")),
        "expected_token_length": token_map.get("expected_length"),
        "recovered_token_length": token_map.get("observed_length"),
        "expected_rank_length": rank_map.get("expected_length"),
        "recovered_rank_length": rank_map.get("observed_length"),
        "execution_error_type": None,
        "execution_error_message": None,
        "divergence_fields_availability": "recorded",
    }


def _execution_failure_row(record: Mapping[str, Any], task: Mapping[str, Any], shard: _RunShard) -> Dict[str, Any]:
    error = record.get("error") if isinstance(record.get("error"), dict) else {}
    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "failure_id": _stable_id("execution_failure", record.get("work_id"), record.get("attempt_index")),
        "trial_id": _coalesce(task.get("trial_id"), task.get("control_id"), record.get("work_id")),
        "work_id": record.get("work_id"),
        "source_trial_id": task.get("source_trial_id"),
        "record_type": "execution_failure",
        "evidence_status": record.get("evidence_status"),
        "study_phase": task.get("study_phase"),
        "model_id": record.get("model_id"),
        "source_model_id": task.get("source_model_id"),
        "payload_name": task.get("payload_name"),
        "payload_class": task.get("payload_class"),
        "protocol_variant": task.get("protocol_variant"),
        "robustness_family": task.get("robustness_family"),
        "replay_mode": task.get("replay_mode"),
        "transformation_id": task.get("transformation_id"),
        "segment_index": None,
        "failure_category": f"execution_failure:{error.get('type', 'unknown')}",
        "first_differing_position": None,
        "expected_token_id": None,
        "recovered_token_id": None,
        "expected_rank": None,
        "recovered_rank": None,
        "context_sha256": None,
        "boundary_start_offset": None,
        "boundary_end_offset": None,
        "expected_token_length": None,
        "recovered_token_length": None,
        "expected_rank_length": None,
        "recovered_rank_length": None,
        "execution_error_type": error.get("type"),
        "execution_error_message": error.get("message"),
        "divergence_fields_availability": NOT_APPLICABLE,
    }


def _detector_pair(
    control: Mapping[str, Any],
    task: Mapping[str, Any],
    source: Mapping[str, Any],
    shard: _RunShard,
) -> List[Dict[str, Any]]:
    view = str(control.get("control_view"))
    stego_text = str(source.get("full_text" if view == "full_message" else "forced_text", ""))
    control_text = str(control.get("full_text", ""))
    if not stego_text.strip() or not control_text.strip():
        raise RevisionPreprocessError(f"Detector pair {control.get('work_id')} contains empty text")
    representation = _representation_name(source, task)
    if not representation:
        raise RevisionPreprocessError("Detector pair cannot determine codec_id")
    protocol_variant = _nonempty(
        _coalesce(source.get("protocol_variant"), task.get("protocol_variant")),
        label="detector protocol variant",
    )
    payload_name = _nonempty(source.get("payload_name"), label="detector payload group")
    prompt_id = _nonempty(source.get("prompt_id"), label="detector prompt template")
    pair_id = _stable_id("detector_pair", source.get("trial_id"), control.get("control_id"), view)
    common = {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "pair_id": pair_id,
        "payload_group_id": payload_name,
        "prompt_template_id": prompt_id,
        "model_id": source.get("model_id"),
        # The prespecified leave-one-codec analysis is over the six primary
        # protocol variants. Keep the implementation representation separately
        # so the three hex protocols are not collapsed into one detector level.
        "codec_id": protocol_variant,
        "protocol_variant": protocol_variant,
        "representation_name": representation,
        "source_trial_id": source.get("trial_id"),
        "control_id": control.get("control_id"),
        "view": view,
        "evidence_status": control.get("evidence_status", shard.evidence_status),
        "payload_class": source.get("payload_class"),
        "prompt_category": source.get("prompt_category"),
        "language": source.get("language"),
    }
    return [
        {**common, "row_id": f"{pair_id}__rankcloak", "text": stego_text, "label": 1},
        {**common, "row_id": f"{pair_id}__ordinary_control", "text": control_text, "label": 0},
    ]


def _unavailable_row(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    shard: _RunShard,
    source_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    record_type = str(record.get("record_type"))
    if record_type not in {"condition_unavailable", "dependent_unavailable"}:
        raise RevisionPreprocessError("Unavailable-row builder received an outcome row")
    if record.get("excluded_from_estimands") is not True:
        raise RevisionPreprocessError(
            f"Unavailable row {record.get('work_id')} is not marked excluded_from_estimands"
        )
    if record.get("exact_recovery") is not None:
        raise RevisionPreprocessError(
            f"Unavailable row {record.get('work_id')} must not encode a recovery outcome"
        )

    if record_type == "condition_unavailable":
        if record.get("reason_code") != "empty_isolated_roundtrip_vocabulary":
            raise RevisionPreprocessError("Unknown frozen condition-unavailable reason")
        if int(record.get("stable_count", -1)) != 0:
            raise RevisionPreprocessError("Empty round-trip vocabulary must record stable_count=0")
        if int(record.get("safe_count", 0)) <= 0:
            raise RevisionPreprocessError("Round-trip feasibility audit must record a nonempty safe mask")
        root = record
    else:
        if record.get("reason_code") != "source_condition_unavailable":
            raise RevisionPreprocessError("Unknown dependent-unavailable reason")
        source = _resolve_source(record, source_records, task)
        if str(source.get("record_type")) not in {
            "condition_unavailable",
            "dependent_unavailable",
        }:
            raise RevisionPreprocessError("Dependent-unavailable row resolves to an outcome source")
        embedded_root = record.get("dependency_root")
        if not isinstance(embedded_root, dict):
            raise RevisionPreprocessError("Dependent-unavailable row lacks dependency_root")
        root = (
            source.get("dependency_root")
            if isinstance(source.get("dependency_root"), dict)
            else source
        )
        for field in ("work_id", "trial_id", "reason_code", "model_id", "tokenizer_id"):
            if str(embedded_root.get(field)) != str(root.get(field)):
                raise RevisionPreprocessError(
                    f"Dependent-unavailable root {field} mismatch for {record.get('work_id')}"
                )
        if str(record.get("dependency_record_type")) != str(source.get("record_type")):
            raise RevisionPreprocessError("Dependent-unavailable source record type mismatch")

    return {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_run_identity_sha256": shard.identity_sha256,
        "evidence_status": record.get("evidence_status"),
        "evidence_family": shard.evidence_family,
        "study_phase": record.get("study_phase"),
        "record_type": record_type,
        "work_kind": record.get("work_kind", task.get("work_kind")),
        "trial_id": record.get("trial_id"),
        "control_id": record.get("control_id"),
        "work_id": record.get("work_id"),
        "source_trial_id": record.get("source_trial_id"),
        "source_record_sha256": record.get("source_record_sha256"),
        "model_id": record.get("model_id"),
        "source_model_id": _coalesce(record.get("source_model_id"), task.get("source_model_id")),
        "tokenizer_id": record.get("tokenizer_id"),
        "tokenizer_revision": record.get("tokenizer_revision"),
        "tokenizer_artifact_sha256": record.get("tokenizer_artifact_sha256"),
        "payload_name": _coalesce(record.get("payload_name"), task.get("payload_name")),
        "payload_class": _coalesce(record.get("payload_class"), task.get("payload_class")),
        "payload_split": _coalesce(record.get("payload_split"), task.get("payload_split")),
        "protocol_variant": _coalesce(record.get("protocol_variant"), task.get("protocol_variant")),
        "prompt_id": _coalesce(record.get("prompt_id"), task.get("prompt_id")),
        "prompt_category": _coalesce(record.get("prompt_category"), task.get("prompt_category")),
        "language": _coalesce(record.get("language"), task.get("language")),
        "token_filter": _coalesce(record.get("token_filter"), task.get("token_filter")),
        "ablation_factor": _coalesce(record.get("ablation_factor"), task.get("ablation_factor")),
        "ablation_level": _coalesce(record.get("ablation_level"), task.get("ablation_level")),
        "robustness_family": _coalesce(record.get("robustness_family"), task.get("robustness_family")),
        "replay_mode": _coalesce(record.get("replay_mode"), task.get("replay_mode")),
        "transformation_id": _coalesce(record.get("transformation_id"), task.get("transformation_id")),
        "reason_code": record.get("reason_code"),
        "reason": record.get("reason"),
        "safe_count": root.get("safe_count"),
        "stable_count": root.get("stable_count"),
        "vocabulary_size": root.get("vocabulary_size"),
        "dependency_role": record.get("dependency_role"),
        "dependency_record_type": record.get("dependency_record_type"),
        "root_condition_work_id": root.get("work_id"),
        "root_condition_trial_id": root.get("trial_id"),
        "root_condition_reason_code": root.get("reason_code"),
        "root_condition_model_id": root.get("model_id"),
        "root_condition_tokenizer_id": root.get("tokenizer_id"),
        "excluded_from_estimands": True,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return buffer.getvalue().encode("utf-8")


def _assert_unique(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *, label: str) -> None:
    seen: set[Tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(column)) for column in columns)
        if key in seen:
            raise RevisionPreprocessError(f"Duplicate {label} identity {dict(zip(columns, key))}")
        seen.add(key)


def _source_index(shards: Sequence[_RunShard]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for shard in shards:
        for record in shard.completions.values():
            if record.get("record_type") not in {
                "rankcloak_trial",
                "condition_unavailable",
                "dependent_unavailable",
            }:
                continue
            trial_id = _nonempty(record.get("trial_id"), label="source trial_id")
            existing = index.get(trial_id)
            if existing is not None and _record_sha256(existing) != _record_sha256(record):
                raise RevisionPreprocessError(f"Conflicting source records for trial {trial_id}")
            index[trial_id] = record
    return index


def _transform_index(shards: Sequence[_RunShard]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for shard in shards:
        for record in shard.completions.values():
            if record.get("record_type") != "robustness_transform":
                continue
            work_id = _nonempty(record.get("work_id"), label="transform work_id")
            existing = index.get(work_id)
            if existing is not None and _record_sha256(existing) != _record_sha256(record):
                raise RevisionPreprocessError(
                    f"Conflicting robustness transform records for {work_id}"
                )
            index[work_id] = record
    return index


def _resolve_transform(
    record: Mapping[str, Any],
    transform_records: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    transform_id = record.get("transform_work_id")
    embedded_hash = record.get("transformation_record_sha256")
    if transform_id is None:
        if embedded_hash is not None:
            raise RevisionPreprocessError(
                "Transformation record hash is present without transform_work_id"
            )
        return None
    work_id = _nonempty(transform_id, label="transform_work_id")
    if work_id not in transform_records:
        raise RevisionPreprocessError(
            f"Missing robustness transform provenance record {work_id}"
        )
    transform = transform_records[work_id]
    actual_hash = _record_sha256(transform)
    if str(embedded_hash) != actual_hash:
        raise RevisionPreprocessError(
            f"Transformation record hash mismatch for {work_id}"
        )
    if str(transform.get("source_trial_id")) != str(record.get("source_trial_id")):
        raise RevisionPreprocessError(f"Transformation source mismatch for {work_id}")
    if str(transform.get("transformation_model_id")) != str(record.get("transformation_model_id")):
        raise RevisionPreprocessError(f"Transformation model mismatch for {work_id}")
    if transform.get("transformation_id") != "paraphrase" or record.get("transformation_id") != "paraphrase":
        raise RevisionPreprocessError(f"Transform provenance is only valid for paraphrase: {work_id}")
    return transform


def _resolve_source(
    record: Mapping[str, Any],
    source_records: Mapping[str, Dict[str, Any]],
    task: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source_id = _nonempty(record.get("source_trial_id"), label="reference source_trial_id")
    if source_id not in source_records:
        raise RevisionPreprocessError(f"Missing referenced RankCloak source record {source_id}")
    source = source_records[source_id]
    embedded_hash = record.get("source_record_sha256")
    if embedded_hash is not None and str(embedded_hash) != _record_sha256(source):
        raise RevisionPreprocessError(f"Referenced source hash mismatch for {source_id}")
    planned = task or {}
    for field in (
        "payload_name",
        "payload_class",
        "payload_split",
        "prompt_id",
        "prompt_category",
        "language",
    ):
        expected = _coalesce(record.get(field), planned.get(field))
        if expected is not None and str(expected) != str(source.get(field)):
            raise RevisionPreprocessError(f"Reference {field} mismatch for {source_id}")
    expected_source_model = record.get("source_model_id")
    if expected_source_model is not None and str(expected_source_model) != str(source.get("model_id")):
        raise RevisionPreprocessError(f"Reference source_model_id mismatch for {source_id}")
    if record.get("record_type") in {"ordinary_control", "canonical_primary_reference"} and str(record.get("model_id")) != str(source.get("model_id")):
        raise RevisionPreprocessError(f"Within-model reference model mismatch for {source_id}")
    return source


def _validate_cross_shard(shards: Sequence[_RunShard], *, emitted_only: bool) -> None:
    selected = [shard for shard in shards if shard.role == "input"] if emitted_only else list(shards)
    paths = [str(shard.path) for shard in selected]
    if len(paths) != len(set(paths)):
        raise RevisionPreprocessError("The same run directory was supplied more than once")
    identities = [shard.identity_sha256 for shard in selected]
    if len(identities) != len(set(identities)):
        raise RevisionPreprocessError("Duplicate run identities would double-count a shard")
    config_hashes = {str(shard.identity["config_manifest_sha256"]) for shard in selected}
    if len(config_hashes) > 1:
        raise RevisionPreprocessError("Run shards bind different frozen config manifests")
    payload_hashes = {str(shard.payload_manifest.get("corpus_sha256")) for shard in selected}
    if len(payload_hashes) > 1:
        raise RevisionPreprocessError("Run shards bind different payload corpora")
    source_hashes = {str(shard.source_manifest.get("files_sha256")) for shard in selected}
    if len(source_hashes) > 1:
        raise RevisionPreprocessError("Run shards bind different runner source snapshots")
    if emitted_only:
        work_ids: set[str] = set()
        for shard in selected:
            overlap = work_ids.intersection(shard.plan_by_id)
            if overlap:
                raise RevisionPreprocessError(
                    "Work IDs occur in multiple emitted shards: " + ", ".join(sorted(overlap)[:3])
                )
            work_ids.update(shard.plan_by_id)
        families = {shard.evidence_family for shard in selected}
        if "exploratory" in families and len(families) > 1:
            raise RevisionPreprocessError(
                "Exploratory smoke/limited evidence cannot be mixed with confirmatory or secondary evidence"
            )


def _preflight_outputs(output_dir: Path) -> Dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise RevisionPreprocessError(f"Output path is not a normal directory: {output}")
    paths = {name: output / filename for name, filename in OUTPUT_FILENAMES.items()}
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise RevisionPreprocessError(
            "Refusing to overwrite preprocessing outputs: " + ", ".join(map(str, existing))
        )
    output.mkdir(parents=True, exist_ok=True)
    return paths


def preprocess_revision_results(
    *,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    reference_run_dirs: Sequence[str | Path] = (),
    strict_complete: bool = True,
) -> PreprocessArtifacts:
    """Validate runner shards and atomically emit flat analysis artifacts.

    ``run_dirs`` are emitted.  ``reference_run_dirs`` are validated and used
    only to resolve canonical-primary and robustness source records.  The
    default rejects incomplete runs so table row counts cannot silently drift.
    """

    if not run_dirs:
        raise RevisionPreprocessError("At least one --run-dir is required")
    emitted = [
        _load_shard(Path(path), role="input", strict_complete=strict_complete)
        for path in sorted(map(str, run_dirs))
    ]
    references = [
        _load_shard(Path(path), role="reference", strict_complete=strict_complete)
        for path in sorted(map(str, reference_run_dirs))
    ]
    all_shards = emitted + references
    _validate_cross_shard(all_shards, emitted_only=False)
    _validate_cross_shard(all_shards, emitted_only=True)
    source_records = _source_index(all_shards)
    transform_records = _transform_index(all_shards)

    trials: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []
    runtime: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    detector: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    join_counts = {
        "ordinary_control": 0,
        "canonical_primary_reference": 0,
        "robustness_source": 0,
        "robustness_transform_provenance": 0,
        "robustness_decode_transform_link": 0,
        "condition_unavailable": 0,
        "dependent_unavailable": 0,
    }

    for shard in emitted:
        runtime.extend(_load_runtime_rows(shard))
        for attempt in shard.attempts:
            if attempt.get("execution_status") == "failed":
                failures.append(
                    _execution_failure_row(attempt, shard.plan_by_id[str(attempt["work_id"])], shard)
                )
        for task in shard.plan:
            work_id = str(task["work_id"])
            record = shard.completions.get(work_id)
            if record is None:
                continue
            record_type = str(record.get("record_type"))
            runtime.append(_runtime_row(record, task, shard))
            if record_type in {"condition_unavailable", "dependent_unavailable"}:
                unavailable.append(
                    _unavailable_row(record, task, shard, source_records)
                )
                join_counts[record_type] += 1
            elif record_type == "rankcloak_trial":
                trials.extend(_rankcloak_trial_rows(record, task, shard))
                features.extend(_rankcloak_feature_rows(record, task, shard))
                for replay_field, replay_mode in (
                    ("saved_token_id_replay", "saved_token_ids"),
                    ("text_retokenization_replay", "detokenized_text_retokenized"),
                    ("greedy_leadin_replay", "greedy_leadin_regeneration"),
                ):
                    outcome = record.get(replay_field)
                    if not isinstance(outcome, dict):
                        continue
                    if replay_field != "saved_token_id_replay" and not bool(outcome.get("executed")):
                        continue
                    if outcome.get("exact_recovery") is False:
                        failure = outcome.get("failure")
                        if not isinstance(failure, dict):
                            raise RevisionPreprocessError(
                                f"Failed replay {work_id}/{replay_mode} lacks failure diagnostics"
                            )
                        failures.append(
                            _failure_row(failure, record, task, shard, replay_mode, "unmodified")
                        )
            elif record_type == "ordinary_control":
                source = _resolve_source(record, source_records, task)
                features.append(_control_feature_row(record, task, shard, source))
                detector.extend(_detector_pair(record, task, source, shard))
                join_counts["ordinary_control"] += 1
            elif record_type == "canonical_primary_reference":
                source = _resolve_source(record, source_records, task)
                trials.append(_resolved_trial_row(record, task, shard, source))
                features.extend(
                    _rankcloak_feature_rows(
                        record,
                        task,
                        shard,
                        source=source,
                        trial_id=str(record.get("trial_id")),
                        source_type="canonical_primary_reference",
                    )
                )
                join_counts["canonical_primary_reference"] += 1
            elif record_type == "robustness_transform":
                _resolve_source(record, source_records, task)
                join_counts["robustness_transform_provenance"] += 1
            elif record_type in {"robustness_decode", "robustness_reference"}:
                source = _resolve_source(record, source_records, task)
                transformation_record = _resolve_transform(record, transform_records)
                if transformation_record is not None:
                    join_counts["robustness_decode_transform_link"] += 1
                if record_type == "robustness_decode":
                    trials.append(_robustness_trial_row(record, task, shard, source))
                else:
                    trials.append(_resolved_trial_row(record, task, shard, source))
                features.extend(_robustness_feature_rows(record, task, shard, source))
                if record.get("exact_recovery") is False:
                    failure = record.get("failure")
                    if not isinstance(failure, dict):
                        raise RevisionPreprocessError(f"Failed robustness row {work_id} lacks diagnostics")
                    failures.append(
                        _failure_row(
                            failure,
                            record,
                            task,
                            shard,
                            str(record.get("replay_mode")),
                            str(record.get("transformation_id")),
                        )
                    )
                join_counts["robustness_source"] += 1
            else:  # guarded by _validate_record_types
                raise RevisionPreprocessError(f"Unsupported completed record_type {record_type}")

    direct_payload_fidelity_rows = _validate_flattened_recovery_rows(trials)
    _assert_unique(
        trials,
        ("trial_id", "replay_mode", "transformation_id", "mitigation_id"),
        label="trial-condition",
    )
    _assert_unique(features, ("row_id",), label="feature row")
    _assert_unique(runtime, ("trial_id", "hardware_hash"), label="runtime row")
    _assert_unique(failures, ("failure_id",), label="failure row")
    _assert_unique(detector, ("row_id",), label="detector row")
    _assert_unique(unavailable, ("work_id",), label="unavailable work unit")
    detector_pair_counts: Dict[str, set[int]] = {}
    for row in detector:
        detector_pair_counts.setdefault(str(row["pair_id"]), set()).add(int(row["label"]))
    invalid_pairs = [pair for pair, labels in detector_pair_counts.items() if labels != {0, 1}]
    if invalid_pairs:
        raise RevisionPreprocessError("Detector corpus contains incomplete matched pairs")

    paths = _preflight_outputs(Path(output_dir))
    input_files = [file for shard in all_shards for file in shard.files]
    input_files.sort(key=lambda row: (str(row["path"]), str(row["role"])))
    input_manifest = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "manifest_type": "revision_preprocessing_inputs",
        "strict_complete": bool(strict_complete),
        "mixing_policy": "exploratory smoke/limited runs are never pooled with non-exploratory runs",
        "emitted_run_count": len(emitted),
        "reference_run_count": len(references),
        "evidence_statuses": sorted({shard.evidence_status for shard in emitted}),
        "evidence_families": sorted({shard.evidence_family for shard in emitted}),
        "run_shards": [
            {
                "role": shard.role,
                "path": str(shard.path),
                "stage": shard.stage,
                "model_id": shard.model_id,
                "evidence_status": shard.evidence_status,
                "run_identity_sha256": shard.identity_sha256,
                "planned_work_units": len(shard.plan),
                "completed_work_units": len(shard.completions),
                "durable_attempt_rows": len(shard.attempts),
            }
            for shard in all_shards
        ],
        "input_files": input_files,
        "input_files_sha256": canonical_json_sha256(input_files),
        "reference_join_counts": join_counts,
    }
    write_immutable_json(paths["input_manifest"], input_manifest)
    write_immutable_bytes(paths["trials"], _csv_bytes(trials, TRIAL_COLUMNS))
    write_immutable_bytes(paths["features"], _csv_bytes(features, FEATURE_COLUMNS))
    write_immutable_bytes(paths["runtime"], _csv_bytes(runtime, RUNTIME_COLUMNS))
    write_immutable_bytes(paths["failures"], _csv_bytes(failures, FAILURE_COLUMNS))
    write_immutable_jsonl(paths["detector"], detector)
    write_immutable_bytes(
        paths["unavailable"], _csv_bytes(unavailable, UNAVAILABLE_COLUMNS)
    )

    row_counts = {
        "trials": len(trials),
        "features": len(features),
        "runtime": len(runtime),
        "failures": len(failures),
        "detector": len(detector),
        "unavailable": len(unavailable),
    }
    output_records = []
    for name in (
        "trials",
        "features",
        "runtime",
        "failures",
        "detector",
        "unavailable",
        "input_manifest",
    ):
        output_records.append(
            {
                "role": name,
                "path": paths[name].name,
                "size_bytes": paths[name].stat().st_size,
                "sha256": file_sha256(paths[name]),
                "row_count": row_counts.get(name),
            }
        )
    output_manifest = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "manifest_type": "revision_preprocessing_outputs",
        "input_manifest_sha256": file_sha256(paths["input_manifest"]),
        "outputs": output_records,
        "outputs_sha256": canonical_json_sha256(output_records),
        "row_counts": row_counts,
        "invariants": {
            "trial_unit": "one payload-condition/replay/transformation row",
            "segment_rows_nested": True,
            "detector_pair_count": len(detector_pair_counts),
            "detector_grouping_unit": "payload_name",
            "missing_values_imputed": False,
            "unavailable_rows_excluded_from_estimands": True,
            "unavailable_rows_are_not_recovery_failures": True,
            "payload_fidelity_contract": {
                "contract_version": PAYLOAD_FIDELITY_CONTRACT_VERSION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "semantics": PAYLOAD_RECOVERY_SEMANTICS,
                "primary_outcome": "exact_payload_recovery",
                "compatibility_alias": "exact_recovery",
                "alias_equality_validated": True,
                "exact_rank_replay_role": "diagnostic_only",
                "direct_rows": direct_payload_fidelity_rows,
                "direct_rows_contract_verified": direct_payload_fidelity_rows,
            },
        },
    }
    write_immutable_json(paths["output_manifest"], output_manifest)
    return PreprocessArtifacts(
        output_dir=str(Path(output_dir).resolve()),
        files={name: str(path.resolve()) for name, path in paths.items()},
        row_counts=row_counts,
        evidence_statuses=tuple(input_manifest["evidence_statuses"]),
    )


__all__ = [
    "EVIDENCE_ABLATION",
    "EVIDENCE_FAMILIES",
    "EVIDENCE_LIMITED",
    "EVIDENCE_MULTILINGUAL",
    "EVIDENCE_PRIMARY",
    "EVIDENCE_ROBUSTNESS",
    "EVIDENCE_SMOKE",
    "FAILURE_COLUMNS",
    "FEATURE_COLUMNS",
    "OUTPUT_FILENAMES",
    "PAYLOAD_FIDELITY_CONTRACT_VERSION",
    "PAYLOAD_RECOVERY_SEMANTICS",
    "PREPROCESS_SCHEMA_VERSION",
    "RESULT_SCHEMA_REVISION",
    "PreprocessArtifacts",
    "RUNTIME_COLUMNS",
    "RevisionPreprocessError",
    "TRIAL_COLUMNS",
    "UNAVAILABLE_COLUMNS",
    "preprocess_revision_results",
]
