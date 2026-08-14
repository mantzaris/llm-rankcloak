"""Auditable, read-only progress snapshots for revision-v1 confirmatory work.

The scanner never writes into a runner, evaluator, or detector shard.  Its
only mutation is an atomic replacement of the canonical progress snapshot.
Historical charges are re-verified through the frozen compute projector and
are kept separate from confirmatory throughput calculations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .revision_artifacts import canonical_json_sha256
from .revision_detection import RevisionDetectionError
from .revision_detector_execution import (
    detector_finalization_paths,
    detector_gpu_ledger_incorporation_path,
    read_detector_device_equivalence_report,
    read_detector_equivalence_fit_artifact,
    read_detector_finalization_candidate,
    read_detector_gpu_accounting_ledger,
    read_detector_gpu_ledger_incorporation_marker,
)


PROGRESS_SCHEMA_VERSION = "rankcloak-revision-confirmatory-progress-v1"
PROGRESS_MANIFEST_TYPE = "confirmatory_progress_snapshot"
PROGRESS_HASH_FIELD = "progress_sha256"
APPROVED_165H_PROJECTION_SHA256 = (
    "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
)
DEFAULT_PROGRESS_RELATIVE_PATH = Path("results/revision_v1/confirmatory_progress_v1.json")
EVALUATOR_UNAVAILABILITY_RELATIVE_PATH = Path(
    "heldout_evaluator/upstream_dependent_unavailability_v1.json"
)
EVALUATOR_UNAVAILABILITY_SCHEMA = (
    "rankcloak-heldout-evaluator-upstream-unavailability-v1"
)
EVALUATOR_UNAVAILABILITY_MANIFEST_TYPE = (
    "heldout_evaluator_upstream_dependent_unavailability"
)
EVALUATOR_SCOREABLE_UNITS = 17_232
EVALUATOR_UPSTREAM_UNAVAILABLE_UNITS = 48
EVALUATOR_FROZEN_TARGET_UNITS = 17_280
EVALUATOR_UNAVAILABLE_SOURCE_MODEL = "mistral_7b_instruct_v0_3_q4_k_m"
EVALUATOR_UNAVAILABLE_EVALUATOR_MODEL = "llama3_8b_instruct_q4_k_m"
AUTHORIZED_GPU_UUID = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"
DETECTOR_TOTAL_FITS = 56
DETECTOR_GPU_INTERVAL_POLICY = "detector_process_wall_span_v1"
DETECTOR_GPU_COLLECTION_POLICY = (
    "nonoverlapping_detector_process_wall_intervals_v1"
)
DETECTOR_EXECUTION_POLICY_SHA256 = (
    "75011cf80bd111e2d1c236aa7799610bf2819a18125d40dbf60d518a206f29e9"
)
DETECTOR_EXECUTION_POLICY_CONTENT_SHA256 = (
    "60152864388a21a37a5a2169145927537e76c7bc437a36e54bfc6bae041785d6"
)
DETECTOR_EXECUTION_POLICY_SUFFIX = Path(
    "operations/confirmatory_v2/detector_acceleration_policy_v1.json"
)
DETECTOR_EQUIVALENCE_RELATIVE_ROOT = Path("detector_equivalence_v1")
DETECTOR_GPU_LEDGER_RELATIVE_PATH = (
    DETECTOR_EQUIVALENCE_RELATIVE_ROOT / "gpu_accounting_ledger.json"
)
DETECTOR_GPU_LEDGER_DERIVATION_POLICY = (
    "terminal_receipt_interval_union_deduplicated_v1"
)
DETECTOR_TERMINAL_RECEIPT_SCHEMA = (
    "rankcloak-revision-detector-terminal-receipt-v1"
)
DETECTOR_FINALIZATION_CANDIDATE_SCHEMA = (
    "rankcloak-revision-detector-finalization-candidate-v1"
)
DETECTOR_LEDGER_INCORPORATION_SCHEMA = (
    "rankcloak-revision-detector-gpu-ledger-incorporation-v1"
)
DETECTOR_EQUIVALENCE_DETECTORS = {
    0: "published_textcnn_equivalent",
    1: "deberta_v3_base_classifier",
}
DETECTOR_EXPECTED_LEDGER_SOURCES = {
    "production_benchmark_task_0": ("benchmark_artifact", 0, "benchmark"),
    "production_benchmark_task_1": ("benchmark_artifact", 1, "benchmark"),
    "equivalence_cuda_task_0": ("equivalence_artifact", 0, "cuda"),
    "equivalence_cuda_task_1": ("equivalence_artifact", 1, "cuda"),
    "equivalence_cuda_repeat_task_0": (
        "equivalence_artifact",
        0,
        "cuda_repeat",
    ),
    "equivalence_cuda_repeat_task_1": (
        "equivalence_artifact",
        1,
        "cuda_repeat",
    ),
}
DETECTOR_ENVIRONMENT_RELATIVE_ROOT = Path("environment/revision_v1")
DETECTOR_ENVIRONMENT_FILES = {
    "environment_manifest.json",
    "scientific_pins.json",
    "CHECKSUMS.sha256",
}

RUNNER_STAGES = ("primary_v2", "ablation_v2", "robustness_v2", "multilingual_v2")
EVALUATOR_STAGE = "heldout_evaluator"
DETECTOR_STAGE = "neural_detector"
STAGE_ORDER = RUNNER_STAGES + (EVALUATOR_STAGE, DETECTOR_STAGE)
OBSERVED_PRIOR_STAGES = (
    "legacy_incurred_observed",
    "invalidated_shard_observed",
    "smoke_observed",
    "auxiliary_smoke_v3_observed",
)
UNAVAILABLE_RECORD_TYPES = {"condition_unavailable", "dependent_unavailable"}
CONDITION_FIELDS = (
    "work_kind",
    "protocol_variant",
    "representation_name",
    "token_filter",
    "segmented",
    "segment_size_ranks",
    "leadin_tokens",
    "tail_policy",
    "language",
    "prompt_category",
    "source_stage",
    "text_view",
    "generator_model_id",
    "evaluator_model_id",
    "replay_modes",
)

EVALUATOR_UNAVAILABILITY_FIELDS = {
    "schema_version",
    "manifest_type",
    "protocol_contract_revision",
    "result_schema_revision",
    "authorized_projection_sha256",
    "frozen_evaluator_target_units",
    "scoreable_evaluator_units",
    "upstream_dependent_unavailable_units",
    "terminal_accounted_units",
    "scoring_attempted_for_unavailable_units",
    "scores_imputed_or_fabricated",
    "analysis_policy",
    "source_files",
    "source_files_sha256",
    "units",
    "units_sha256",
    "manifest_sha256",
}
EVALUATOR_UNAVAILABILITY_UNIT_FIELDS = {
    "terminal_status",
    "source_stage",
    "source_work_id",
    "source_record_type",
    "source_record_sha256",
    "reason_code",
    "generator_model_id",
    "evaluator_model_id",
    "protocol_variant",
    "payload_name",
    "scoring_attempted",
    "score_imputed",
}


class RevisionProgressError(RuntimeError):
    """Raised when progress cannot be measured without guessing."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_digest(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop(PROGRESS_HASH_FIELD, None)
    return canonical_json_sha256(unsigned)


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RevisionProgressError("{} is missing".format(label))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RevisionProgressError("{} is not ISO-8601".format(label)) from exc
    if parsed.tzinfo is None:
        raise RevisionProgressError("{} lacks a timezone".format(label))
    return parsed.astimezone(timezone.utc)


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RevisionProgressError("{} is not numeric".format(label)) from exc
    if not math.isfinite(result) or result < 0:
        raise RevisionProgressError("{} must be finite and nonnegative".format(label))
    return result


def _reject_symlink_chain(path: Path, boundary: Path, label: str) -> None:
    path = Path(path)
    boundary = Path(boundary).resolve(strict=True)
    try:
        relative = path.absolute().relative_to(boundary)
    except ValueError as exc:
        raise RevisionProgressError("{} escapes the results root".format(label)) from exc
    cursor = boundary
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RevisionProgressError("{} contains a symlink: {}".format(label, cursor))


def _stable_bytes(path: Path, results_root: Path, label: str) -> bytes:
    _reject_symlink_chain(path, results_root, label)
    if not path.is_file():
        raise RevisionProgressError("{} is not a regular file: {}".format(label, path))
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(content) != after.st_size:
        raise RevisionProgressError("{} changed while being read: {}".format(label, path))
    return content


def _json_object(content: bytes, label: str) -> Dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RevisionProgressError("{} is not valid UTF-8 JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise RevisionProgressError("{} must contain a JSON object".format(label))
    return value


def _jsonl(content: bytes, label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RevisionProgressError("{} is not UTF-8".format(label)) from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RevisionProgressError(
                "{} has invalid JSON on line {}".format(label, line_number)
            ) from exc
        if not isinstance(value, dict):
            raise RevisionProgressError(
                "{} line {} is not an object".format(label, line_number)
            )
        rows.append(value)
    return rows


def _artifact(path: Path, content: bytes) -> Dict[str, object]:
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_bytes(content),
        "size_bytes": len(content),
    }


def _stable_progress_source_bytes(
    path: Path, results_root: Path, label: str
) -> bytes:
    """Allow only exact frozen detector policy/environment external sources."""

    root = Path(results_root).resolve(strict=True)
    project_root = root.parents[1]
    policy = (project_root / DETECTOR_EXECUTION_POLICY_SUFFIX).resolve()
    environment_sources = {
        (project_root / DETECTOR_ENVIRONMENT_RELATIVE_ROOT / name).resolve()
        for name in DETECTOR_ENVIRONMENT_FILES
    }
    resolved = Path(path).resolve()
    boundary = (
        project_root
        if resolved == policy or resolved in environment_sources
        else root
    )
    return _stable_bytes(path, boundary, label)


def _unit_id(row: Mapping[str, object], component: str) -> str:
    if component == "heldout_evaluator":
        return str(row.get("evaluation_id", ""))
    return str(row.get("work_id") or row.get("trial_id") or "")


def _bound_model_id(row: Mapping[str, object], component: str) -> str:
    if component == "heldout_evaluator":
        return str(row.get("evaluator_model_id", ""))
    return str(row.get("model_id", ""))


def _condition(row: Mapping[str, object]) -> Dict[str, object]:
    value = {field: row.get(field) for field in CONDITION_FIELDS if field in row}
    if "replay_modes" in value and isinstance(value["replay_modes"], list):
        value["replay_modes"] = list(map(str, value["replay_modes"]))
    return value


def _condition_key(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_unavailable(record: Mapping[str, object]) -> bool:
    record_type = str(record.get("record_type", "")).lower()
    status = str(record.get("execution_status", "")).lower()
    availability = str(record.get("availability_status", "")).lower()
    return (
        record_type in UNAVAILABLE_RECORD_TYPES
        or "unavailable" in status
        or "unavailable" in availability
    )


def _is_payload_bearing(row: Mapping[str, object]) -> bool:
    return str(row.get("work_kind", "")) in {"rankcloak", "robustness_decode"}


def _payload_recovery_outcome(record: Mapping[str, object]) -> Tuple[bool, bool]:
    """Return (attempted, successful) for one completed payload work unit."""

    if _is_unavailable(record):
        return False, False
    if record.get("record_type") == "robustness_decode":
        exact = record.get("exact_recovery")
        return isinstance(exact, bool), exact is True
    attempted = False
    successful = False
    for field in (
        "saved_token_id_replay",
        "text_retokenization_replay",
        "greedy_leadin_replay",
    ):
        replay = record.get(field)
        if not isinstance(replay, dict):
            continue
        exact = replay.get("exact_payload_recovery", replay.get("exact_recovery"))
        did_run = replay.get("executed") is True or isinstance(exact, bool)
        if did_run:
            attempted = True
            successful = successful or exact is True
    return attempted, successful


def _event_time(row: Mapping[str, object]) -> Optional[datetime]:
    for key in ("at", "completed_at", "updated_at", "created_at"):
        if row.get(key) is not None:
            return _timestamp(row[key], "event {}".format(key))
    return None


def _occupancy_intervals(
    events: Sequence[Mapping[str, object]],
    checkpoint: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    shard_path: Path,
    component: str,
    model_id: str,
) -> List[Dict[str, object]]:
    loads = []
    for index, event in enumerate(events):
        if str(event.get("event")) not in {"model_loaded", "evaluator_model_loaded"}:
            continue
        at = _timestamp(event.get("at"), "model-load event at")
        seconds = _finite_nonnegative(event.get("model_load_seconds"), "model_load_seconds")
        gpu_uuid = str(event.get("gpu_uuid", ""))
        if not gpu_uuid.startswith("GPU-"):
            raise RevisionProgressError("Model-load event lacks a bound GPU UUID")
        event_model = str(event.get("model_id", event.get("evaluator_model_id", model_id)))
        if event_model != model_id:
            raise RevisionProgressError("Model-load event model does not match its shard")
        loads.append((index, at, at - timedelta(seconds=seconds), gpu_uuid))

    durable_times: List[datetime] = []
    for event in events:
        at = _event_time(event)
        if at is not None:
            durable_times.append(at)
    if checkpoint.get("updated_at") is not None:
        durable_times.append(_timestamp(checkpoint["updated_at"], "checkpoint updated_at"))
    for record in records:
        if record.get("completed_at") is not None:
            durable_times.append(_timestamp(record["completed_at"], "record completed_at"))
    if (records or checkpoint.get("completed_trial_ids")) and not loads:
        raise RevisionProgressError(
            "Durable work exists without a model-load/GPU identity in {}".format(shard_path)
        )
    if not loads:
        return []

    intervals: List[Dict[str, object]] = []
    for load_number, (event_index, load_at, fallback_start, gpu_uuid) in enumerate(loads):
        next_load_at = loads[load_number + 1][1] if load_number + 1 < len(loads) else None
        window_events = []
        for candidate in events[event_index + 1 :]:
            candidate_at = _event_time(candidate)
            if candidate_at is None:
                continue
            if next_load_at is not None and candidate_at >= next_load_at:
                break
            window_events.append(candidate)
        profiles = [
            event for event in window_events if str(event.get("event")) == "memory_profile"
        ]
        if len(profiles) > 1:
            raise RevisionProgressError("A model-load session has multiple memory profiles")
        if profiles:
            profile = profiles[0]
            started = _timestamp(profile.get("started_at"), "memory profile started_at")
            ended = _timestamp(profile.get("at"), "memory profile at")
        else:
            started = fallback_start
            finish_events = [
                _event_time(event)
                for event in window_events
                if str(event.get("event")) in {"session_finished", "evaluator_session_finished"}
            ]
            finished = [time for time in finish_events if time is not None]
            if finished:
                ended = max(finished)
            else:
                candidates = [
                    time
                    for time in durable_times
                    if time >= load_at
                    and (next_load_at is None or time < next_load_at)
                ]
                if next_load_at is not None:
                    candidates.append(next_load_at)
                ended = max(candidates) if candidates else load_at
        if ended < load_at or ended <= started:
            raise RevisionProgressError("Invalid GPU occupancy interval in {}".format(shard_path))
        intervals.append(
            {
                "component": component,
                "model_id": model_id,
                "shard_path": str(shard_path.resolve(strict=True)),
                "gpu_uuid": gpu_uuid,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "seconds": (ended - started).total_seconds(),
                "derivation_policy": (
                    "memory_profile_wall_span_v1"
                    if profiles
                    else "model_load_to_last_durable_checkpoint_v1"
                ),
            }
        )
    return intervals


def _scan_shard(
    shard_path: Path, results_root: Path, stage: str, component: str
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    plan_path = shard_path / "plan.jsonl"
    checkpoint_path = shard_path / "checkpoint.json"
    records_path = shard_path / "records.jsonl"
    events_path = shard_path / "events.jsonl"
    contents: Dict[str, bytes] = {}
    for name, path in (
        ("plan", plan_path),
        ("checkpoint", checkpoint_path),
        ("records", records_path),
        ("events", events_path),
    ):
        if path.exists() or name in {"plan", "checkpoint"}:
            contents[name] = _stable_bytes(path, results_root, "{} {}".format(component, name))
    plan = _jsonl(contents["plan"], "plan")
    checkpoint = _json_object(contents["checkpoint"], "checkpoint")
    records = _jsonl(contents.get("records", b""), "records")
    events = _jsonl(contents.get("events", b""), "events")
    if not plan:
        raise RevisionProgressError("Empty shard plan: {}".format(plan_path))

    plan_by_id: Dict[str, Dict[str, object]] = {}
    for row in plan:
        # Every frozen unit has a unique work_id. Only payload-bearing rows
        # additionally have trial_id; controls use control_id.
        trial_id = _unit_id(row, component)
        if not trial_id or trial_id in plan_by_id:
            raise RevisionProgressError("Missing or duplicate trial_id in {}".format(plan_path))
        plan_by_id[trial_id] = row
    if int(checkpoint.get("planned_trial_count", -1)) != len(plan_by_id):
        raise RevisionProgressError("Checkpoint/plan count mismatch in {}".format(shard_path))
    completed_ids = set(map(str, checkpoint.get("completed_trial_ids", [])))
    failed_ids = set(map(str, checkpoint.get("failed_trial_ids", [])))
    if completed_ids & failed_ids or not completed_ids.issubset(plan_by_id) or not failed_ids.issubset(plan_by_id):
        raise RevisionProgressError("Checkpoint trial sets are inconsistent in {}".format(shard_path))

    latest_records: Dict[str, Dict[str, object]] = {}
    for record in records:
        trial_id = _unit_id(record, component)
        if not trial_id or trial_id not in plan_by_id:
            raise RevisionProgressError("Record has an unknown trial_id in {}".format(shard_path))
        attempt = int(record.get("attempt_index", 1))
        previous = latest_records.get(trial_id)
        if previous is None or attempt >= int(previous.get("attempt_index", 1)):
            latest_records[trial_id] = record
    missing_records = sorted(completed_ids - set(latest_records))
    if missing_records:
        raise RevisionProgressError("Completed checkpoint IDs lack durable records")

    unavailable_ids = {
        trial_id for trial_id in completed_ids if _is_unavailable(latest_records[trial_id])
    }
    success_ids = completed_ids - unavailable_ids
    payload_unavailable_ids = {
        work_id for work_id in unavailable_ids if _is_payload_bearing(plan_by_id[work_id])
    }
    payload_recovery_attempted = 0
    successful_payload_recoveries = 0
    for work_id in sorted(completed_ids - unavailable_ids):
        if not _is_payload_bearing(plan_by_id[work_id]):
            continue
        attempted, successful = _payload_recovery_outcome(latest_records[work_id])
        payload_recovery_attempted += int(attempted)
        successful_payload_recoveries += int(attempted and successful)
    payload_recovery_failures = (
        payload_recovery_attempted - successful_payload_recoveries
    )
    attempt_counts = checkpoint.get("attempt_counts", {})
    failure_details = checkpoint.get("failure_details", {})
    if not isinstance(attempt_counts, dict) or not isinstance(failure_details, dict):
        raise RevisionProgressError("Checkpoint retry metadata is malformed")
    recovered = []
    for trial_id in sorted(completed_ids):
        attempts = int(attempt_counts.get(trial_id, latest_records[trial_id].get("attempt_index", 1)))
        if attempts > 1 or trial_id in failure_details:
            recovered.append(
                {
                    "stage": stage,
                    "component": component,
                    "model_id": _bound_model_id(plan_by_id[trial_id], component),
                    "trial_id": trial_id,
                    "attempt_count": attempts,
                    "recovered_status": (
                        "unavailable" if trial_id in unavailable_ids else "success"
                    ),
                    "last_record_completed_at": latest_records[trial_id].get("completed_at"),
                    "retained_failure_detail": failure_details.get(trial_id),
                }
            )

    condition_counts: MutableMapping[str, Counter] = defaultdict(Counter)
    condition_values: Dict[str, Dict[str, object]] = {}
    for trial_id, row in plan_by_id.items():
        condition = _condition(row)
        key = _condition_key(condition)
        condition_values[key] = condition
        condition_counts[key]["total"] += 1
        if trial_id in completed_ids:
            condition_counts[key]["completed"] += 1
        if trial_id in success_ids:
            condition_counts[key]["successes"] += 1
        if trial_id in unavailable_ids:
            condition_counts[key]["unavailable"] += 1
        if trial_id in failed_ids:
            condition_counts[key]["failures"] += 1
    conditions = []
    for key in sorted(condition_counts):
        counts = condition_counts[key]
        total = counts["total"]
        terminal = counts["completed"] + counts["failures"]
        conditions.append(
            {
                "condition": condition_values[key],
                "completed": counts["completed"],
                "total": total,
                "successes": counts["successes"],
                "failures": counts["failures"],
                "unavailable": counts["unavailable"],
                "remaining": total - terminal,
            }
        )

    model_ids = {_bound_model_id(row, component) for row in plan}
    if len(model_ids) != 1 or "" in model_ids:
        raise RevisionProgressError("Shard plan does not bind exactly one model")
    model_id = next(iter(model_ids))
    pending_ids = [trial_id for trial_id in plan_by_id if trial_id not in completed_ids]
    current_trial_id = pending_ids[0] if pending_ids else None
    current_condition = _condition(plan_by_id[current_trial_id]) if current_trial_id else None
    intervals = _occupancy_intervals(
        events, checkpoint, records, shard_path, component, model_id
    )
    terminal_times = []
    for trial_id in completed_ids:
        value = latest_records[trial_id].get("completed_at")
        if value is not None:
            terminal_times.append(_timestamp(value, "record completed_at"))
    updated = _timestamp(checkpoint.get("updated_at"), "checkpoint updated_at")
    artifacts_by_name = []
    for name, content in contents.items():
        filename = "checkpoint.json" if name == "checkpoint" else name + ".jsonl"
        artifacts_by_name.append(_artifact(shard_path / filename, content))

    row = {
        "stage": stage,
        "component": component,
        "model_id": model_id,
        "path": str(shard_path.resolve(strict=True)),
        "state": (
            "complete"
            if len(completed_ids) == len(plan_by_id) and not failed_ids
            else "complete_with_failures"
            if len(completed_ids) + len(failed_ids) == len(plan_by_id)
            else "in_progress_or_paused"
        ),
        "completed": len(completed_ids),
        "total": len(plan_by_id),
        "successes": len(success_ids),
        "failures": len(failed_ids),
        "unavailable": len(unavailable_ids),
        "recovery_counts": {
            "payload_bearing_recovery_attempted": payload_recovery_attempted,
            "successful_payload_recoveries": successful_payload_recoveries,
            "payload_recovery_failures": payload_recovery_failures,
            "unavailable": len(payload_unavailable_ids),
        },
        "remaining": len(plan_by_id) - len(completed_ids) - len(failed_ids),
        "current_trial_id": current_trial_id,
        "current_condition": current_condition,
        "checkpoint_updated_at": updated.isoformat(),
        "checkpoint_sha256": _sha256_bytes(contents["checkpoint"]),
        "conditions": conditions,
        "terminal_completion_times": [time.isoformat() for time in sorted(terminal_times)],
        "measured_gpu_seconds": sum(float(item["seconds"]) for item in intervals),
        "gpu_accounting_policy": "sum_of_verified_nonoverlapping_durable_wall_spans",
    }
    return row, artifacts_by_name, recovered


def _discover_shards(results_root: Path) -> List[Tuple[str, str, Path]]:
    candidates: List[Tuple[str, str, Path]] = []
    for stage in RUNNER_STAGES:
        stage_root = results_root / stage
        if stage_root.is_dir() and not stage_root.is_symlink():
            for checkpoint in sorted(stage_root.glob("*/checkpoint.json")):
                candidates.append((stage, "runner", checkpoint.parent))
    evaluator_root = results_root / EVALUATOR_STAGE
    if evaluator_root.is_dir() and not evaluator_root.is_symlink():
        for source_stage in RUNNER_STAGES:
            stage_root = evaluator_root / source_stage
            if stage_root.is_dir() and not stage_root.is_symlink():
                for checkpoint in sorted(stage_root.glob("*/checkpoint.json")):
                    candidates.append((EVALUATOR_STAGE, "heldout_evaluator", checkpoint.parent))
    paths = [str(path.resolve(strict=True)) for _, _, path in candidates]
    if len(paths) != len(set(paths)):
        raise RevisionProgressError("A confirmatory shard was discovered more than once")
    return candidates


def _verified_baseline(results_root: Path) -> Dict[str, object]:
    projection_path = results_root / "compute_projection_165h_v2.json"
    projection_content = _stable_bytes(
        projection_path, results_root, "approved 165-GPU-hour projection"
    )
    report = _json_object(projection_content, "approved 165-GPU-hour projection")
    unsigned_saved = dict(report)
    saved_hash = unsigned_saved.pop("projection_sha256", None)
    if (
        saved_hash != APPROVED_165H_PROJECTION_SHA256
        or saved_hash != canonical_json_sha256(unsigned_saved)
    ):
        raise RevisionProgressError(
            "Approved 165-hour projection exact self-hash mismatch"
        )
    if (
        report.get("budget_gpu_hours") != 165.0
        or not isinstance(report.get("decision"), dict)
        or report["decision"].get("go") is not True
        or report["decision"].get("status") != "go_within_budget"
    ):
        raise RevisionProgressError("Approved 165-hour projection is not a GO artifact")
    if report.get("input_status") != "complete":
        raise RevisionProgressError(
            "Prior-charge verification failed: {}".format(report.get("incomplete_reasons"))
        )
    unsigned = dict(report)
    supplied_hash = unsigned.pop("projection_sha256", None)
    if supplied_hash != canonical_json_sha256(unsigned):
        raise RevisionProgressError("Verified compute projection self-hash mismatch")
    rows = {str(row["stage"]): row for row in report.get("stage_totals", [])}
    if not set(OBSERVED_PRIOR_STAGES).issubset(rows):
        raise RevisionProgressError("Verified compute projection lacks prior actual stages")
    prior_components = []
    prior_seconds = 0.0
    for stage in OBSERVED_PRIOR_STAGES:
        row = rows[stage]
        point = _finite_nonnegative(row.get("point_gpu_hours"), stage + " point hours")
        upper = _finite_nonnegative(row.get("upper_gpu_hours"), stage + " upper hours")
        if point != upper:
            raise RevisionProgressError("Observed prior charge has a projected interval")
        seconds = point * 3600.0
        prior_seconds += seconds
        charge_only = stage in {"legacy_incurred_observed", "invalidated_shard_observed"}
        prior_components.append(
            {
                "component": stage,
                "seconds": seconds,
                "gpu_hours": point,
                "verification": "project_revision_compute_fail_closed_v1",
                "charge_only_not_rate_evidence": charge_only,
                "scientific_result_evidence_allowed": False,
                "rate_evidence_allowed": stage in {
                    "smoke_observed", "auxiliary_smoke_v3_observed"
                },
            }
        )
    audit = report.get("combined_incurred_charge_audit")
    if not isinstance(audit, dict) or not str(audit.get("status", "")).startswith("ok_"):
        raise RevisionProgressError("Legacy/invalidation combined audit is not verified")
    ledger_and_invalid = sum(
        item["seconds"] for item in prior_components if item["charge_only_not_rate_evidence"]
    )
    if ledger_and_invalid != float(audit.get("combined_seconds")):
        raise RevisionProgressError("Prior charge totals do not reconcile")
    targets = {
        str(row["stage"]): int(row["target_work_units"])
        for row in report.get("stage_totals", [])
        if str(row.get("stage")) in STAGE_ORDER
    }
    if set(targets) != set(STAGE_ORDER):
        raise RevisionProgressError("Verified compute projection lacks confirmatory targets")
    return {
        "verified_prior_seconds": prior_seconds,
        "prior_components": prior_components,
        "targets": targets,
        "projection_sha256": supplied_hash,
        "projection_decision": report.get("decision"),
        "projection_artifact": _artifact(projection_path, projection_content),
    }


def _assert_nonoverlap(intervals: Sequence[Mapping[str, object]]) -> None:
    by_gpu: MutableMapping[str, List[Tuple[datetime, datetime, str]]] = defaultdict(list)
    seen = set()
    for item in intervals:
        identity = (str(item["shard_path"]), str(item["started_at"]), str(item["ended_at"]))
        if identity in seen:
            raise RevisionProgressError("Duplicate confirmatory GPU interval")
        seen.add(identity)
        started = _timestamp(item["started_at"], "GPU interval start")
        ended = _timestamp(item["ended_at"], "GPU interval end")
        if ended <= started:
            raise RevisionProgressError("GPU interval is not positive")
        by_gpu[str(item["gpu_uuid"])].append((started, ended, str(item["shard_path"])))
    for gpu_uuid, rows in by_gpu.items():
        ordered = sorted(rows)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise RevisionProgressError(
                    "Confirmatory GPU intervals overlap on {}: {} and {}".format(
                        gpu_uuid, previous[2], current[2]
                    )
                )


def _detector_gpu_intervals(
    manifest: Mapping[str, object], manifest_path: Path
) -> List[Dict[str, object]]:
    """Verify final CUDA provenance; retain legacy/CPU detector behavior."""

    accounting = manifest.get("gpu_accounting")
    device = manifest.get("device")
    if accounting is None:
        if device not in {None, "cpu"}:
            raise RevisionProgressError(
                "CUDA detector manifest lacks final GPU accounting"
            )
        return []
    expected_policy_path = (
        manifest_path.parents[4] / DETECTOR_EXECUTION_POLICY_SUFFIX
    ).resolve()
    expected_permit_path = manifest_path.parent.with_name(
        manifest_path.parent.name + ".fit_permit.json"
    ).resolve()
    run_identity = manifest.get("run_identity")
    if (
        not isinstance(accounting, dict)
        or set(accounting)
        != {
            "device",
            "gpu_uuid",
            "intervals",
            "cumulative_elapsed_seconds",
            "derivation_policy",
        }
        or device != "cuda:0"
        or manifest.get("gpu_uuid") != AUTHORIZED_GPU_UUID
        or int(manifest.get("workers", -1)) != 1
        or int(manifest.get("completed_fit_count", -1)) != DETECTOR_TOTAL_FITS
        or int(manifest.get("total_fit_count", -1)) != DETECTOR_TOTAL_FITS
        or accounting.get("device") != "cuda:0"
        or accounting.get("gpu_uuid") != AUTHORIZED_GPU_UUID
        or accounting.get("derivation_policy")
        != DETECTOR_GPU_COLLECTION_POLICY
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
        or Path(str(manifest.get("execution_policy_path", ""))).resolve()
        != expected_policy_path
        or manifest.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or manifest.get("execution_policy_content_sha256")
        != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or Path(str(manifest.get("fit_permit_file", ""))).resolve()
        != expected_permit_path
        or not isinstance(run_identity, dict)
        or manifest.get("run_identity_sha256")
        != canonical_json_sha256(run_identity)
        or Path(str(run_identity.get("execution_policy_path", ""))).resolve()
        != expected_policy_path
        or run_identity.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or Path(str(run_identity.get("fit_permit_file", ""))).resolve()
        != expected_permit_path
        or run_identity.get("require_fit_permit") is not True
    ):
        raise RevisionProgressError(
            "Detector final GPU accounting identity is invalid"
        )
    result: List[Dict[str, object]] = []
    identities: Set[Tuple[int, int]] = set()
    previous_end: Optional[datetime] = None
    total = 0.0
    for raw in accounting["intervals"]:
        if not isinstance(raw, dict) or set(raw) != {
            "pid",
            "process_start_ticks",
            "device",
            "gpu_uuid",
            "started_at_utc",
            "completed_at_utc",
            "last_observed_at_utc",
            "elapsed_seconds",
            "derivation_policy",
        }:
            raise RevisionProgressError("Detector GPU interval shape is invalid")
        pid = int(raw.get("pid", -1))
        start_ticks = int(raw.get("process_start_ticks", -1))
        identity = (pid, start_ticks)
        started = _timestamp(
            raw.get("started_at_utc"), "detector GPU interval start"
        )
        completed = _timestamp(
            raw.get("completed_at_utc"), "detector GPU interval completion"
        )
        observed = _timestamp(
            raw.get("last_observed_at_utc"), "detector GPU interval observation"
        )
        seconds = _finite_nonnegative(
            raw.get("elapsed_seconds"), "detector GPU interval seconds"
        )
        if (
            pid <= 0
            or start_ticks <= 0
            or identity in identities
            or raw.get("device") != "cuda:0"
            or raw.get("gpu_uuid") != AUTHORIZED_GPU_UUID
            or raw.get("derivation_policy") != DETECTOR_GPU_INTERVAL_POLICY
            or completed <= started
            or observed != completed
            or abs((completed - started).total_seconds() - seconds) > 1e-6
            or (previous_end is not None and started < previous_end)
        ):
            raise RevisionProgressError(
                "Detector GPU interval provenance is invalid"
            )
        identities.add(identity)
        previous_end = completed
        total += seconds
        result.append(
            {
                "component": DETECTOR_STAGE,
                "model_id": "checkpointed_56_fit_suite",
                "shard_path": str(manifest_path.parent.resolve(strict=True)),
                "gpu_uuid": AUTHORIZED_GPU_UUID,
                "started_at": started.isoformat(),
                "ended_at": completed.isoformat(),
                "seconds": seconds,
                "derivation_policy": DETECTOR_GPU_INTERVAL_POLICY,
                "source_pid": pid,
                "source_process_start_ticks": start_ticks,
                "collection_policy": DETECTOR_GPU_COLLECTION_POLICY,
            }
        )
    cumulative = _finite_nonnegative(
        accounting.get("cumulative_elapsed_seconds"),
        "detector cumulative GPU seconds",
    )
    if abs(cumulative - total) > 1e-6:
        raise RevisionProgressError(
            "Detector cumulative GPU time differs from its intervals"
        )
    return result


def _signed_detector_document(
    value: Mapping[str, object], hash_field: str, label: str
) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(hash_field, None)
    if not isinstance(claimed, str) or canonical_json_sha256(unsigned) != claimed:
        raise RevisionProgressError("{} self-hash differs".format(label))


def _unique_detector_artifact(
    artifacts: MutableMapping[str, Dict[str, object]],
    path: Path,
    content: bytes,
) -> None:
    artifact = _artifact(path, content)
    previous = artifacts.get(str(artifact["path"]))
    if previous is not None and previous != artifact:
        raise RevisionProgressError(
            "A detector accounting source changed during the progress scan"
        )
    artifacts[str(artifact["path"])] = artifact


def _detector_policy_identity_artifacts(
    value: object,
    results_root: Path,
) -> List[Dict[str, object]]:
    """Revalidate exact predeclared policy and frozen environment bytes."""

    if not isinstance(value, dict):
        raise RevisionProgressError(
            "Detector accounting source lacks policy/environment identity"
        )
    project_root = results_root.resolve(strict=True).parents[1]
    policy_path = (project_root / DETECTOR_EXECUTION_POLICY_SUFFIX).resolve()
    environment_root = (
        project_root / DETECTOR_ENVIRONMENT_RELATIVE_ROOT
    ).resolve()
    expected_keys = {
        "schema_version",
        "execution_policy_path",
        "execution_policy_sha256",
        "execution_policy_content_sha256",
        "environment_binding",
        "environment_binding_sha256",
        "equivalence_policy",
        "equivalence_policy_sha256",
    }
    environment = value.get("environment_binding")
    if (
        set(value) != expected_keys
        or value.get("schema_version")
        != "rankcloak-revision-detector-equivalence-policy-identity-v1"
        or Path(str(value.get("execution_policy_path", ""))).resolve()
        != policy_path
        or value.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or value.get("execution_policy_content_sha256")
        != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or not isinstance(environment, dict)
        or value.get("environment_binding_sha256")
        != canonical_json_sha256(environment)
        or environment.get("schema_version")
        != "rankcloak-revision-detector-environment-binding-v1"
        or Path(str(environment.get("root", ""))).resolve()
        != environment_root
        or environment.get("verification_status") != "ok"
    ):
        raise RevisionProgressError(
            "Detector accounting policy/environment identity differs"
        )
    policy_content = _stable_progress_source_bytes(
        policy_path, results_root, "detector acceleration policy"
    )
    policy = _json_object(policy_content, "detector acceleration policy")
    unsigned_policy = dict(policy)
    claimed_policy = unsigned_policy.pop("policy_sha256", None)
    if (
        _sha256_bytes(policy_content) != DETECTOR_EXECUTION_POLICY_SHA256
        or claimed_policy != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or canonical_json_sha256(unsigned_policy) != claimed_policy
        or value.get("equivalence_policy") != policy.get("equivalence")
        or value.get("equivalence_policy_sha256")
        != canonical_json_sha256(policy.get("equivalence"))
    ):
        raise RevisionProgressError(
            "Detector accounting acceleration policy differs"
        )
    required = environment.get("required_files")
    if not isinstance(required, dict) or set(required) != DETECTOR_ENVIRONMENT_FILES:
        raise RevisionProgressError(
            "Detector accounting environment declaration differs"
        )
    artifacts: Dict[str, Dict[str, object]] = {}
    _unique_detector_artifact(artifacts, policy_path, policy_content)
    required_contents: Dict[str, bytes] = {}
    for name in sorted(DETECTOR_ENVIRONMENT_FILES):
        path = environment_root / name
        content = _stable_progress_source_bytes(
            path, results_root, "detector frozen environment {}".format(name)
        )
        declaration = required[name]
        if not isinstance(declaration, dict) or declaration != {
            "path": str(path.resolve()),
            "sha256": _sha256_bytes(content),
            "size_bytes": len(content),
        }:
            raise RevisionProgressError(
                "Detector accounting frozen environment bytes differ"
            )
        required_contents[name] = content
        _unique_detector_artifact(artifacts, path, content)
    manifest = _json_object(
        required_contents["environment_manifest.json"],
        "detector frozen environment manifest",
    )
    records = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_type")
        != "rankcloak_revision_environment_file_set"
        or manifest.get("snapshot_status") != "complete"
        or not isinstance(records, list)
        or int(manifest.get("file_count", -1)) != len(records)
        or manifest.get("files_sha256") != canonical_json_sha256(records)
        or environment.get("environment_files_sha256")
        != manifest.get("files_sha256")
        or int(environment.get("verified_file_count", -1)) != len(records)
    ):
        raise RevisionProgressError(
            "Detector accounting frozen environment manifest differs"
        )
    listed: Set[str] = set()
    listed_contents: Dict[str, bytes] = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RevisionProgressError(
                "Detector accounting environment file row is malformed"
            )
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RevisionProgressError(
                "Detector accounting environment path is unsafe"
            )
        source = environment_root / relative
        content = _stable_bytes(
            source, environment_root, "detector frozen environment source"
        )
        if (
            relative.as_posix() in listed
            or row.get("sha256") != _sha256_bytes(content)
            or int(row.get("size_bytes", -1)) != len(content)
        ):
            raise RevisionProgressError(
                "Detector accounting environment source identity differs"
            )
        listed.add(relative.as_posix())
        listed_contents[relative.as_posix()] = content
    all_entries = list(environment_root.rglob("*"))
    if any(
        path.is_symlink() or not (path.is_file() or path.is_dir())
        for path in all_entries
    ):
        raise RevisionProgressError(
            "Detector accounting environment contains a symlink/special entry"
        )
    actual_files = {
        path.relative_to(environment_root).as_posix()
        for path in all_entries
        if path.is_file()
    }
    if actual_files != listed | {"environment_manifest.json"}:
        raise RevisionProgressError(
            "Detector accounting environment file set differs"
        )
    expected_directories = {
        parent.as_posix()
        for name in actual_files
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    actual_directories = {
        path.relative_to(environment_root).as_posix()
        for path in all_entries
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise RevisionProgressError(
            "Detector accounting environment directory set differs"
        )
    checksum_rows: Dict[str, str] = {}
    try:
        checksum_lines = required_contents["CHECKSUMS.sha256"].decode(
            "utf-8"
        ).splitlines()
    except UnicodeDecodeError as exc:
        raise RevisionProgressError(
            "Detector accounting environment CHECKSUMS is not UTF-8"
        ) from exc
    for line in checksum_lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match.group(2) in checksum_rows:
            raise RevisionProgressError(
                "Detector accounting environment CHECKSUMS is invalid"
            )
        checksum_rows[match.group(2)] = match.group(1)
    expected_checksum_paths = listed - {"CHECKSUMS.sha256"}
    if set(checksum_rows) != expected_checksum_paths or any(
        _sha256_bytes(listed_contents[relative]) != digest
        for relative, digest in checksum_rows.items()
    ):
        raise RevisionProgressError(
            "Detector accounting environment CHECKSUMS differs"
        )
    return list(artifacts.values())


def _detector_ledger_interval_rows(
    ledger: Mapping[str, object], ledger_path: Path
) -> List[Dict[str, object]]:
    if (
        ledger.get("device") != "cuda:0"
        or ledger.get("gpu_uuid") != AUTHORIZED_GPU_UUID
        or ledger.get("derivation_policy")
        != DETECTOR_GPU_LEDGER_DERIVATION_POLICY
        or not isinstance(ledger.get("intervals"), list)
    ):
        raise RevisionProgressError("Detector GPU ledger identity is invalid")
    result: List[Dict[str, object]] = []
    identities: Set[Tuple[int, int]] = set()
    prior_end: Optional[datetime] = None
    total = 0.0
    for raw in ledger["intervals"]:
        if not isinstance(raw, dict) or set(raw) != {
            "pid",
            "process_start_ticks",
            "device",
            "gpu_uuid",
            "started_at_utc",
            "completed_at_utc",
            "last_observed_at_utc",
            "elapsed_seconds",
            "derivation_policy",
        }:
            raise RevisionProgressError("Detector GPU ledger interval shape is invalid")
        pid = int(raw.get("pid", -1))
        ticks = int(raw.get("process_start_ticks", -1))
        started = _timestamp(raw.get("started_at_utc"), "ledger interval start")
        ended = _timestamp(raw.get("completed_at_utc"), "ledger interval end")
        observed = _timestamp(
            raw.get("last_observed_at_utc"), "ledger interval observation"
        )
        seconds = _finite_nonnegative(
            raw.get("elapsed_seconds"), "ledger interval seconds"
        )
        if (
            pid <= 0
            or ticks <= 0
            or (pid, ticks) in identities
            or raw.get("device") != "cuda:0"
            or raw.get("gpu_uuid") != AUTHORIZED_GPU_UUID
            or raw.get("derivation_policy") != DETECTOR_GPU_INTERVAL_POLICY
            or ended <= started
            or observed != ended
            or abs((ended - started).total_seconds() - seconds) > 1e-6
            or (prior_end is not None and started < prior_end)
        ):
            raise RevisionProgressError(
                "Detector GPU ledger interval provenance is invalid"
            )
        identities.add((pid, ticks))
        prior_end = ended
        total += seconds
        result.append(
            {
                "component": DETECTOR_STAGE,
                "model_id": "pre_final_detector_gpu_ledger",
                "shard_path": str(ledger_path.resolve(strict=True)),
                "gpu_uuid": AUTHORIZED_GPU_UUID,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "seconds": seconds,
                "derivation_policy": DETECTOR_GPU_INTERVAL_POLICY,
                "source_pid": pid,
                "source_process_start_ticks": ticks,
                "collection_policy": DETECTOR_GPU_LEDGER_DERIVATION_POLICY,
            }
        )
    cumulative = _finite_nonnegative(
        ledger.get("cumulative_elapsed_seconds"),
        "detector GPU ledger cumulative seconds",
    )
    if abs(cumulative - total) > 1e-6:
        raise RevisionProgressError(
            "Detector GPU ledger cumulative seconds differ"
        )
    return result


def _detector_ledger_expected_output(
    results_root: Path, source_id: str
) -> Path:
    kind, task_index, role = DETECTOR_EXPECTED_LEDGER_SOURCES[source_id]
    if kind == "benchmark_artifact":
        return (
            results_root
            / "supervisor"
            / "detector_benchmark_task_{}_cuda.json".format(task_index)
        ).resolve()
    return (
        results_root
        / DETECTOR_EQUIVALENCE_RELATIVE_ROOT
        / "task_{}".format(task_index)
        / "{}_artifact.json".format(role)
    ).resolve()


def _detector_ledger_expected_checkpoint(
    results_root: Path, source_id: str
) -> Path:
    kind, task_index, role = DETECTOR_EXPECTED_LEDGER_SOURCES[source_id]
    if kind == "benchmark_artifact" or role == "cuda":
        return (
            results_root
            / DETECTOR_STAGE
            / "confirmatory_v2.checkpoints"
        ).resolve()
    return (
        results_root
        / DETECTOR_EQUIVALENCE_RELATIVE_ROOT
        / "task_{}".format(task_index)
        / "cuda_repeat_run.checkpoints"
    ).resolve()


def _scan_detector_gpu_ledger(
    results_root: Path,
    *,
    final_manifest_path: Optional[Path],
    final_manifest: Optional[Mapping[str, object]],
) -> Tuple[
    Optional[Dict[str, object]],
    Optional[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Verify every ledger source and optionally its final incorporation triad."""

    ledger_path = results_root / DETECTOR_GPU_LEDGER_RELATIVE_PATH
    marker_path = detector_gpu_ledger_incorporation_path(ledger_path)
    if not ledger_path.exists() and not ledger_path.is_symlink():
        if marker_path.exists() or marker_path.is_symlink():
            raise RevisionProgressError(
                "Detector GPU incorporation marker exists without its ledger"
            )
        if final_manifest is not None and final_manifest.get("device") == "cuda:0":
            raise RevisionProgressError(
                "Final CUDA detector manifest lacks its pre-final GPU ledger"
            )
        return None, None, [], []
    ledger_content = _stable_bytes(
        ledger_path, results_root, "detector GPU accounting ledger"
    )
    parsed_ledger = _json_object(
        ledger_content, "detector GPU accounting ledger"
    )
    try:
        ledger = read_detector_gpu_accounting_ledger(ledger_path)
    except RevisionDetectionError as exc:
        raise RevisionProgressError(
            "Detector GPU ledger failed strict validation: {}".format(exc)
        ) from exc
    if ledger != parsed_ledger:
        raise RevisionProgressError(
            "Detector GPU ledger changed during strict validation"
        )
    artifacts: Dict[str, Dict[str, object]] = {}
    _unique_detector_artifact(artifacts, ledger_path, ledger_content)
    sources = ledger.get("sources")
    if not isinstance(sources, list):
        raise RevisionProgressError("Detector GPU ledger sources are malformed")
    source_ids = {str(row.get("source_id", "")) for row in sources}
    if not source_ids.issubset(DETECTOR_EXPECTED_LEDGER_SOURCES):
        raise RevisionProgressError("Detector GPU ledger has an unknown source")
    if final_manifest is not None and source_ids != set(
        DETECTOR_EXPECTED_LEDGER_SOURCES
    ):
        raise RevisionProgressError(
            "Final CUDA detector ledger does not contain all six pre-final sources"
        )
    expected_components = {
        "benchmark_artifact": "detector_production_benchmark",
        "equivalence_artifact": None,
    }
    for source in sources:
        source_id = str(source["source_id"])
        kind, task_index, role = DETECTOR_EXPECTED_LEDGER_SOURCES[source_id]
        expected_component = expected_components[kind]
        if kind == "equivalence_artifact":
            expected_component = "detector_device_equivalence_{}".format(role)
        if source.get("component") != expected_component:
            raise RevisionProgressError(
                "Detector GPU ledger source component differs"
            )
        receipt_identity = source.get("terminal_receipt")
        if not isinstance(receipt_identity, dict):
            raise RevisionProgressError(
                "Detector GPU ledger receipt declaration is malformed"
            )
        output_path = _detector_ledger_expected_output(results_root, source_id)
        checkpoint_dir = _detector_ledger_expected_checkpoint(
            results_root, source_id
        )
        candidate_path, expected_receipt_path = detector_finalization_paths(
            checkpoint_dir,
            kind=kind,
            requested_output_path=output_path,
            task_index=task_index,
            role=role,
        )
        receipt_path = Path(str(receipt_identity.get("path", ""))).resolve()
        if receipt_path != expected_receipt_path.resolve():
            raise RevisionProgressError(
                "Detector GPU ledger terminal receipt path differs"
            )
        receipt_content = _stable_bytes(
            receipt_path, results_root, "detector ledger terminal receipt"
        )
        receipt = _json_object(
            receipt_content, "detector ledger terminal receipt"
        )
        _signed_detector_document(
            receipt, "terminal_receipt_sha256", "Detector terminal receipt"
        )
        _unique_detector_artifact(artifacts, receipt_path, receipt_content)
        candidate_identity = receipt.get("candidate")
        published_identity = receipt.get("published_output")
        closed_status = receipt.get("closed_status")
        if (
            receipt.get("schema_version") != DETECTOR_TERMINAL_RECEIPT_SCHEMA
            or not isinstance(candidate_identity, dict)
            or not isinstance(published_identity, dict)
            or not isinstance(closed_status, dict)
            or Path(str(candidate_identity.get("path", ""))).resolve()
            != candidate_path.resolve()
            or Path(str(published_identity.get("path", ""))).resolve()
            != output_path
        ):
            raise RevisionProgressError(
                "Detector GPU ledger terminal receipt identity differs"
            )
        _signed_detector_document(
            closed_status, "status_sha256", "Detector ledger closed status"
        )
        if (
            closed_status.get("state") != "supervisor_observed_process_exit"
            or closed_status.get("status_sha256")
            != receipt.get("closed_status_sha256")
            or closed_status.get("gpu_accounting")
            != receipt.get("gpu_accounting")
        ):
            raise RevisionProgressError(
                "Detector ledger source was not closed by the supervisor"
            )
        candidate_content = _stable_bytes(
            candidate_path, results_root, "detector ledger finalization candidate"
        )
        parsed_candidate = _json_object(
            candidate_content, "detector ledger finalization candidate"
        )
        try:
            candidate = read_detector_finalization_candidate(candidate_path)
        except RevisionDetectionError as exc:
            raise RevisionProgressError(
                "Detector ledger candidate failed validation: {}".format(exc)
            ) from exc
        if (
            candidate != parsed_candidate
            or candidate.get("schema_version")
            != DETECTOR_FINALIZATION_CANDIDATE_SCHEMA
            or candidate.get("kind") != kind
            or candidate.get("run_identity_sha256")
            != receipt.get("run_identity_sha256")
            or Path(str(candidate.get("requested_output_path", ""))).resolve()
            != output_path
        ):
            raise RevisionProgressError(
                "Detector ledger candidate kind/task/output identity differs"
            )
        _unique_detector_artifact(artifacts, candidate_path, candidate_content)
        for child in candidate.get("output_files", {}).values():
            if not isinstance(child, dict):
                raise RevisionProgressError(
                    "Detector ledger candidate child identity is malformed"
                )
            child_path = Path(str(child.get("path", ""))).resolve()
            child_content = _stable_bytes(
                child_path, results_root, "detector ledger candidate child"
            )
            _unique_detector_artifact(artifacts, child_path, child_content)
        output_content = _stable_bytes(
            output_path, results_root, "detector ledger published output"
        )
        output = _json_object(output_content, "detector ledger published output")
        _unique_detector_artifact(artifacts, output_path, output_content)
        run_identity = closed_status.get("run_identity")
        lineage = None if not isinstance(run_identity, dict) else run_identity.get(
            "lineage"
        )
        execution_binding = (
            None if not isinstance(lineage, dict) else lineage.get("execution_policy")
        )
        environment_binding = (
            None if not isinstance(lineage, dict) else lineage.get("environment_binding")
        )
        if not isinstance(execution_binding, dict) or not isinstance(
            environment_binding, dict
        ):
            raise RevisionProgressError(
                "Detector ledger source run identity lacks frozen policy/environment"
            )
        policy_identity = {
            "schema_version": (
                "rankcloak-revision-detector-equivalence-policy-identity-v1"
            ),
            "execution_policy_path": execution_binding.get("path"),
            "execution_policy_sha256": execution_binding.get("sha256"),
            "execution_policy_content_sha256": execution_binding.get(
                "policy_sha256"
            ),
            "environment_binding": environment_binding,
            "environment_binding_sha256": canonical_json_sha256(
                environment_binding
            ),
            "equivalence_policy": execution_binding.get("policy", {}).get(
                "equivalence"
            ),
            "equivalence_policy_sha256": canonical_json_sha256(
                execution_binding.get("policy", {}).get("equivalence")
            ),
        }
        for artifact in _detector_policy_identity_artifacts(
            policy_identity, results_root
        ):
            artifacts[str(artifact["path"])] = artifact
        identity = output.get(
            "benchmark_task_identity"
            if kind == "benchmark_artifact"
            else "task_identity"
        )
        if (
            not isinstance(identity, dict)
            or int(identity.get("ordinal", -1)) != task_index
            or identity.get("detector_name")
            != DETECTOR_EQUIVALENCE_DETECTORS[task_index]
        ):
            raise RevisionProgressError(
                "Detector GPU ledger source scientific task identity differs"
            )
        if kind == "benchmark_artifact":
            _signed_detector_document(
                output, "benchmark_sha256", "Detector benchmark artifact"
            )
            if (
                output.get("schema_version")
                != "rankcloak-revision-detector-benchmark-v1"
                or int(output.get("benchmark_task_index", -1)) != task_index
                or output.get("device") != "cuda:0"
                or output.get("gpu_uuid") != AUTHORIZED_GPU_UUID
                or int(output.get("workers", -1)) != 1
            ):
                raise RevisionProgressError(
                    "Detector benchmark ledger source runtime differs"
                )
        else:
            try:
                artifact = read_detector_equivalence_fit_artifact(output_path)
            except RevisionDetectionError as exc:
                raise RevisionProgressError(
                    "Detector equivalence ledger source failed validation: {}".format(
                        exc
                    )
                ) from exc
            provenance = artifact.get("provenance")
            if (
                artifact != output
                or artifact.get("role") != role
                or int(artifact.get("task_index", -1)) != task_index
                or not isinstance(provenance, dict)
                or provenance.get("device") != "cuda:0"
                or provenance.get("gpu_uuid") != AUTHORIZED_GPU_UUID
                or int(provenance.get("workers", -1)) != 1
                or provenance.get("policy_identity") != policy_identity
            ):
                raise RevisionProgressError(
                    "Detector equivalence ledger source runtime/policy differs"
                )
    marker: Optional[Dict[str, object]] = None
    if marker_path.exists() or marker_path.is_symlink():
        if final_manifest is None or final_manifest_path is None:
            raise RevisionProgressError(
                "Detector GPU ledger is marked incorporated without a final manifest"
            )
        marker_content = _stable_bytes(
            marker_path, results_root, "detector GPU ledger incorporation marker"
        )
        parsed_marker = _json_object(
            marker_content, "detector GPU ledger incorporation marker"
        )
        try:
            marker = read_detector_gpu_ledger_incorporation_marker(marker_path)
        except RevisionDetectionError as exc:
            raise RevisionProgressError(
                "Detector GPU ledger incorporation failed validation: {}".format(
                    exc
                )
            ) from exc
        if (
            marker != parsed_marker
            or marker.get("schema_version")
            != DETECTOR_LEDGER_INCORPORATION_SCHEMA
            or marker.get("incorporated") is not True
            or Path(str(marker.get("ledger", {}).get("path", ""))).resolve()
            != ledger_path.resolve()
            or marker.get("ledger", {}).get("ledger_sha256")
            != ledger.get("ledger_sha256")
            or marker.get("ledger", {}).get("intervals_sha256")
            != ledger.get("intervals_sha256")
            or Path(
                str(marker.get("final_published_manifest", {}).get("path", ""))
            ).resolve()
            != final_manifest_path.resolve()
            or marker.get("final_published_manifest", {}).get("sha256")
            != _sha256_bytes(
                _stable_bytes(
                    final_manifest_path,
                    results_root,
                    "detector final incorporated manifest",
                )
            )
        ):
            raise RevisionProgressError(
                "Detector GPU ledger incorporation identity differs"
            )
        _unique_detector_artifact(artifacts, marker_path, marker_content)
    elif final_manifest is not None and final_manifest.get("device") == "cuda:0":
        raise RevisionProgressError(
            "Final CUDA detector manifest has an unincorporated GPU ledger"
        )
    intervals = (
        []
        if final_manifest is not None and final_manifest.get("device") == "cuda:0"
        else _detector_ledger_interval_rows(ledger, ledger_path)
    )
    return ledger, marker, list(artifacts.values()), intervals


def _validate_detector_report_for_progress(
    results_root: Path,
    task_index: int,
    declaration: Mapping[str, object],
    artifacts: MutableMapping[str, Dict[str, object]],
) -> Dict[str, object]:
    report_path = (
        results_root
        / DETECTOR_EQUIVALENCE_RELATIVE_ROOT
        / "task_{}".format(task_index)
        / "equivalence_report.json"
    ).resolve()
    report_content = _stable_bytes(
        report_path, results_root, "detector equivalence report"
    )
    parsed = _json_object(report_content, "detector equivalence report")
    if declaration != {
        "path": str(report_path),
        "sha256": _sha256_bytes(report_content),
        "size_bytes": len(report_content),
        "report_sha256": parsed.get("report_sha256"),
    }:
        raise RevisionProgressError(
            "Final detector equivalence report declaration differs"
        )
    policy_identity = parsed.get("policy_identity")
    policy_artifacts = _detector_policy_identity_artifacts(
        policy_identity, results_root
    )
    policy_path = (
        results_root.resolve(strict=True).parents[1]
        / DETECTOR_EXECUTION_POLICY_SUFFIX
    )
    policy = _json_object(
        _stable_progress_source_bytes(
            policy_path, results_root, "detector equivalence policy"
        ),
        "detector equivalence policy",
    )
    try:
        report = read_detector_device_equivalence_report(
            report_path,
            expected_task_index=task_index,
            expected_policy_identity=policy_identity,
            expected_equivalence_policy=policy.get("equivalence"),
        )
    except RevisionDetectionError as exc:
        raise RevisionProgressError(
            "Detector equivalence report failed strict validation: {}".format(exc)
        ) from exc
    if report != parsed or report.get("decision", {}).get("equivalent") is not True:
        raise RevisionProgressError(
            "Detector equivalence report is not an exact passing decision"
        )
    _unique_detector_artifact(artifacts, report_path, report_content)
    for artifact in policy_artifacts:
        artifacts[str(artifact["path"])] = artifact
    declarations = report.get("input_artifacts")
    if not isinstance(declarations, dict) or set(declarations) != {
        "cpu",
        "cuda",
        "cuda_repeat",
    }:
        raise RevisionProgressError(
            "Detector equivalence report input declaration differs"
        )
    for role in ("cpu", "cuda", "cuda_repeat"):
        artifact_path = (
            results_root
            / DETECTOR_EQUIVALENCE_RELATIVE_ROOT
            / "task_{}".format(task_index)
            / "{}_artifact.json".format(role)
        ).resolve()
        content = _stable_bytes(
            artifact_path, results_root, "detector equivalence report input"
        )
        artifact = _json_object(content, "detector equivalence report input")
        identity = declarations[role]
        if not isinstance(identity, dict) or identity != {
            "path": str(artifact_path),
            "sha256": _sha256_bytes(content),
            "size_bytes": len(content),
            "artifact_sha256": artifact.get("artifact_sha256"),
        }:
            raise RevisionProgressError(
                "Detector equivalence report input bytes differ"
            )
        try:
            verified = read_detector_equivalence_fit_artifact(artifact_path)
        except RevisionDetectionError as exc:
            raise RevisionProgressError(
                "Detector equivalence report input failed validation: {}".format(
                    exc
                )
            ) from exc
        identity_row = verified.get("task_identity")
        provenance = verified.get("provenance")
        if (
            verified != artifact
            or verified.get("role") != role
            or int(verified.get("task_index", -1)) != task_index
            or not isinstance(identity_row, dict)
            or identity_row.get("detector_name")
            != DETECTOR_EQUIVALENCE_DETECTORS[task_index]
            or not isinstance(provenance, dict)
            or provenance.get("policy_identity") != policy_identity
        ):
            raise RevisionProgressError(
                "Detector equivalence report scientific/runtime identity differs"
            )
        if role == "cpu":
            if (
                provenance.get("device") != "cpu"
                or provenance.get("gpu_uuid") is not None
                or provenance.get("gpu_accounting") is not None
            ):
                raise RevisionProgressError(
                    "Detector CPU equivalence reference is not zero-GPU"
                )
        elif (
            provenance.get("device") != "cuda:0"
            or provenance.get("gpu_uuid") != AUTHORIZED_GPU_UUID
            or int(provenance.get("workers", -1)) != 1
        ):
            raise RevisionProgressError(
                "Detector CUDA equivalence report input runtime differs"
            )
        _unique_detector_artifact(artifacts, artifact_path, content)
    return report


def _validate_detector_final_triad(
    results_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
    ledger: Mapping[str, object],
    marker: Mapping[str, object],
    artifact_rows: List[Dict[str, object]],
) -> None:
    """Require the sole-supervisor final manifest/receipt/marker/status seal."""

    artifacts: Dict[str, Dict[str, object]] = {
        str(row["path"]): dict(row) for row in artifact_rows
    }
    ledger_path = results_root / DETECTOR_GPU_LEDGER_RELATIVE_PATH
    marker_path = detector_gpu_ledger_incorporation_path(ledger_path)
    expected_ledger_identity = {
        "path": str(ledger_path.resolve()),
        "sha256": file_sha256(ledger_path),
        "size_bytes": int(ledger_path.stat().st_size),
        "ledger_sha256": ledger.get("ledger_sha256"),
        "sources_sha256": ledger.get("sources_sha256"),
        "intervals_sha256": ledger.get("intervals_sha256"),
        "cumulative_elapsed_seconds": ledger.get(
            "cumulative_elapsed_seconds"
        ),
    }
    if manifest.get("pre_final_gpu_accounting_ledger") != expected_ledger_identity:
        raise RevisionProgressError(
            "Final detector manifest did not incorporate the exact current ledger"
        )
    final_accounting = manifest.get("gpu_accounting")
    if not isinstance(final_accounting, dict) or not isinstance(
        final_accounting.get("intervals"), list
    ):
        raise RevisionProgressError(
            "Final detector accounting is malformed before ledger incorporation"
        )
    final_interval_hashes = {
        canonical_json_sha256(row) for row in final_accounting["intervals"]
    }
    ledger_interval_hashes = {
        canonical_json_sha256(row) for row in ledger.get("intervals", [])
    }
    if not ledger_interval_hashes.issubset(final_interval_hashes):
        raise RevisionProgressError(
            "Final detector accounting omits a pre-final ledger interval"
        )
    checkpoint_dir = Path(str(manifest.get("checkpoint_dir", ""))).resolve()
    candidate_path, receipt_path = detector_finalization_paths(
        checkpoint_dir,
        kind="detector_run_manifest",
        requested_output_path=manifest_path,
        role="suite",
    )
    marker_receipt = marker.get("final_terminal_receipt")
    if (
        not isinstance(marker_receipt, dict)
        or Path(str(marker_receipt.get("path", ""))).resolve()
        != receipt_path.resolve()
        or marker.get("incorporated_ledger_interval_count")
        != len(ledger.get("intervals", []))
        or marker.get("incorporated_ledger_intervals_sha256")
        != ledger.get("intervals_sha256")
        or marker.get("final_gpu_accounting_sha256")
        != canonical_json_sha256(manifest.get("gpu_accounting"))
    ):
        raise RevisionProgressError(
            "Final detector ledger incorporation marker differs"
        )
    receipt_content = _stable_bytes(
        receipt_path, results_root, "final detector terminal receipt"
    )
    receipt = _json_object(receipt_content, "final detector terminal receipt")
    _signed_detector_document(
        receipt, "terminal_receipt_sha256", "Final detector terminal receipt"
    )
    _unique_detector_artifact(artifacts, receipt_path, receipt_content)
    candidate_content = _stable_bytes(
        candidate_path, results_root, "final detector candidate"
    )
    candidate = _json_object(candidate_content, "final detector candidate")
    try:
        verified_candidate = read_detector_finalization_candidate(candidate_path)
    except RevisionDetectionError as exc:
        raise RevisionProgressError(
            "Final detector candidate failed strict validation: {}".format(exc)
        ) from exc
    _unique_detector_artifact(artifacts, candidate_path, candidate_content)
    expected_candidate_identity = {
        "path": str(candidate_path.resolve()),
        "sha256": _sha256_bytes(candidate_content),
        "size_bytes": len(candidate_content),
        "candidate_sha256": candidate.get("candidate_sha256"),
    }
    if (
        verified_candidate != candidate
        or candidate.get("kind") != "detector_run_manifest"
        or receipt.get("candidate") != expected_candidate_identity
        or manifest.get("finalization_candidate") != expected_candidate_identity
        or receipt.get("published_output")
        != {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "size_bytes": int(manifest_path.stat().st_size),
        }
        or receipt.get("gpu_accounting") != manifest.get("gpu_accounting")
        or receipt.get("closed_status_sha256")
        != manifest.get("terminal_accounting_status_sha256")
    ):
        raise RevisionProgressError(
            "Final detector candidate/receipt/manifest triad differs"
        )
    status_path = Path(str(manifest.get("status_file", ""))).resolve()
    status_content = _stable_bytes(
        status_path, results_root, "final detector complete status"
    )
    status = _json_object(status_content, "final detector complete status")
    _signed_detector_document(status, "status_sha256", "Final detector status")
    expected_receipt_identity = {
        "path": str(receipt_path.resolve()),
        "sha256": _sha256_bytes(receipt_content),
        "size_bytes": len(receipt_content),
        "terminal_receipt_sha256": receipt.get("terminal_receipt_sha256"),
    }
    marker_content = _stable_bytes(
        marker_path, results_root, "final detector incorporation marker"
    )
    expected_marker_identity = {
        "path": str(marker_path.resolve()),
        "sha256": _sha256_bytes(marker_content),
        "size_bytes": len(marker_content),
        "incorporation_sha256": marker.get("incorporation_sha256"),
    }
    if (
        status.get("state") != "complete"
        or status.get("run_identity_sha256")
        != manifest.get("run_identity_sha256")
        or status.get("gpu_accounting") != manifest.get("gpu_accounting")
        or status.get("terminal_receipt") != expected_receipt_identity
        or status.get("gpu_ledger_incorporation") != expected_marker_identity
        or status.get("final_manifest")
        != {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "size_bytes": int(manifest_path.stat().st_size),
        }
    ):
        raise RevisionProgressError(
            "Final detector complete status does not bind the terminal triad"
        )
    _unique_detector_artifact(artifacts, status_path, status_content)
    reports = manifest.get("required_equivalence_reports")
    run_identity = manifest.get("run_identity")
    lineage = None if not isinstance(run_identity, dict) else run_identity.get(
        "lineage"
    )
    if (
        not isinstance(reports, list)
        or len(reports) != 2
        or not isinstance(lineage, dict)
        or lineage.get("required_equivalence_reports") != reports
        or Path(str(lineage.get("pre_final_gpu_accounting_ledger_path", ""))).resolve()
        != ledger_path.resolve()
    ):
        raise RevisionProgressError(
            "Final detector manifest lacks two exact required equivalence reports"
        )
    for task_index, declaration in enumerate(reports):
        if not isinstance(declaration, dict):
            raise RevisionProgressError(
                "Final detector equivalence report declaration is malformed"
            )
        _validate_detector_report_for_progress(
            results_root, task_index, declaration, artifacts
        )
    artifact_rows[:] = list(artifacts.values())


def _scan_detector_outputs(
    results_root: Path,
) -> Tuple[
    Dict[str, int], List[Dict[str, object]], List[Dict[str, object]]
]:
    counts = {"completed": 0, "successes": 0, "failures": 0, "unavailable": 0}
    artifacts: List[Dict[str, object]] = []
    intervals: List[Dict[str, object]] = []
    detector_root = results_root / DETECTOR_STAGE
    if detector_root.exists() or detector_root.is_symlink():
        _reject_symlink_chain(detector_root, results_root, "detector output root")
        if detector_root.is_symlink() or not detector_root.is_dir():
            raise RevisionProgressError(
                "Detector output root is not a regular directory"
            )
        manifests = sorted(detector_root.rglob("detector_run_manifest.json"))
    else:
        manifests = []
    final_cuda_path: Optional[Path] = None
    final_cuda_manifest: Optional[Dict[str, object]] = None
    for path in manifests:
        content = _stable_bytes(path, results_root, "detector run manifest")
        manifest = _json_object(content, "detector run manifest")
        if (
            manifest.get("schema_version") != "rankcloak-revision-detector-run-v2"
            or manifest.get("execution_mode") != "confirmatory"
            or manifest.get("smoke") is not False
        ):
            raise RevisionProgressError("Non-confirmatory detector output in confirmatory root")
        failures = int(manifest.get("failure_count", -1))
        metrics = int(manifest.get("metric_rows", -1))
        if failures < 0 or metrics < 0:
            raise RevisionProgressError("Detector output counts are invalid")
        if manifest.get("confirmatory_complete") is not True:
            raise RevisionProgressError("Detector manifest is not confirmatory-complete")
        if manifest.get("device") == "cuda:0":
            final_cuda_path = path
            final_cuda_manifest = manifest
            canonical_manifest = (
                detector_root / "confirmatory_v2" / "detector_run_manifest.json"
            )
            allowed_top_level = {
                "confirmatory_v2",
                "confirmatory_v2.checkpoints",
                "confirmatory_v2.status.json",
            }
            observed_top_level = {entry.name for entry in detector_root.iterdir()}
            if (
                len(manifests) != 1
                or path.resolve() != canonical_manifest.resolve()
                or not observed_top_level.issubset(allowed_top_level)
                or failures != 0
                or metrics != DETECTOR_TOTAL_FITS
            ):
                raise RevisionProgressError(
                    "Complete CUDA detector output is not the one canonical 56-fit result"
                )
        counts["successes"] += metrics
        counts["failures"] += failures
        counts["completed"] += metrics
        artifacts.append(_artifact(path, content))
        products = manifest.get("output_files")
        if not isinstance(products, dict):
            raise RevisionProgressError("Detector manifest lacks product identities")
        for name, identity_row in sorted(products.items()):
            product = path.parent / str(name)
            product_content = _stable_bytes(product, results_root, "detector product")
            if not isinstance(identity_row, dict) or identity_row != {
                "sha256": _sha256_bytes(product_content),
                "size_bytes": len(product_content),
            }:
                raise RevisionProgressError("Detector product identity mismatch")
            artifacts.append(_artifact(product, product_content))
        intervals.extend(_detector_gpu_intervals(manifest, path))
        if manifest.get("device") == "cuda:0":
            project_root = results_root.resolve(strict=True).parents[1]
            policy_path = project_root / DETECTOR_EXECUTION_POLICY_SUFFIX
            policy_content = _stable_bytes(
                policy_path, project_root, "detector acceleration policy"
            )
            policy = _json_object(
                policy_content, "detector acceleration policy"
            )
            unsigned = dict(policy)
            claimed = unsigned.pop("policy_sha256", None)
            if (
                _sha256_bytes(policy_content)
                != DETECTOR_EXECUTION_POLICY_SHA256
                or claimed != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
                or canonical_json_sha256(unsigned) != claimed
            ):
                raise RevisionProgressError(
                    "Detector acceleration policy source identity differs"
                )
            artifacts.append(_artifact(policy_path, policy_content))
    ledger, marker, ledger_artifacts, ledger_intervals = (
        _scan_detector_gpu_ledger(
            results_root,
            final_manifest_path=final_cuda_path,
            final_manifest=final_cuda_manifest,
        )
    )
    artifacts.extend(ledger_artifacts)
    intervals.extend(ledger_intervals)
    if final_cuda_manifest is not None:
        if ledger is None or marker is None or final_cuda_path is None:
            raise RevisionProgressError(
                "Final CUDA detector output lacks exact GPU ledger incorporation"
            )
        _validate_detector_final_triad(
            results_root,
            final_cuda_path,
            final_cuda_manifest,
            ledger,
            marker,
            artifacts,
        )
    artifacts_by_path: Dict[str, Dict[str, object]] = {}
    for artifact in artifacts:
        path_key = str(artifact["path"])
        prior = artifacts_by_path.get(path_key)
        if prior is not None and prior != artifact:
            raise RevisionProgressError(
                "Detector source artifact changed during the progress scan"
            )
        artifacts_by_path[path_key] = artifact
    return counts, list(artifacts_by_path.values()), intervals


def _scan_evaluator_unavailability(
    results_root: Path,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]], Set[Tuple[str, str]]]:
    """Verify and bind the exact 48-unit evaluator-unavailability manifest."""

    manifest_path = results_root / EVALUATOR_UNAVAILABILITY_RELATIVE_PATH
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None, [], set()
    manifest_content = _stable_bytes(
        manifest_path, results_root, "held-out evaluator unavailability manifest"
    )
    manifest = _json_object(
        manifest_content, "held-out evaluator unavailability manifest"
    )
    if set(manifest) != EVALUATOR_UNAVAILABILITY_FIELDS:
        raise RevisionProgressError(
            "Held-out evaluator unavailability manifest fields mismatch"
        )
    unsigned = dict(manifest)
    claimed_manifest_hash = unsigned.pop("manifest_sha256", None)
    if (
        not isinstance(claimed_manifest_hash, str)
        or claimed_manifest_hash != canonical_json_sha256(unsigned)
    ):
        raise RevisionProgressError(
            "Held-out evaluator unavailability manifest self-hash mismatch"
        )
    integer_contract = all(
        type(manifest.get(field)) is int
        for field in (
            "frozen_evaluator_target_units",
            "scoreable_evaluator_units",
            "upstream_dependent_unavailable_units",
            "terminal_accounted_units",
        )
    )
    if (
        manifest.get("schema_version") != EVALUATOR_UNAVAILABILITY_SCHEMA
        or manifest.get("manifest_type")
        != EVALUATOR_UNAVAILABILITY_MANIFEST_TYPE
        or manifest.get("protocol_contract_revision") != "payload_fidelity_v2"
        or manifest.get("result_schema_revision") != "payload_aware_result_v2"
        or manifest.get("authorized_projection_sha256")
        != APPROVED_165H_PROJECTION_SHA256
        or not integer_contract
        or manifest.get("frozen_evaluator_target_units")
        != EVALUATOR_FROZEN_TARGET_UNITS
        or manifest.get("scoreable_evaluator_units") != EVALUATOR_SCOREABLE_UNITS
        or manifest.get("upstream_dependent_unavailable_units")
        != EVALUATOR_UPSTREAM_UNAVAILABLE_UNITS
        or manifest.get("terminal_accounted_units")
        != EVALUATOR_FROZEN_TARGET_UNITS
        or manifest.get("scoreable_evaluator_units")
        + manifest.get("upstream_dependent_unavailable_units")
        != manifest.get("terminal_accounted_units")
        or manifest.get("scoring_attempted_for_unavailable_units") is not False
        or manifest.get("scores_imputed_or_fabricated") is not False
        or manifest.get("analysis_policy")
        != "terminal_design_units_excluded_from_quality_estimands_and_not_scored"
    ):
        raise RevisionProgressError(
            "Held-out evaluator unavailability manifest contract mismatch"
        )

    source_files = manifest.get("source_files")
    if (
        not isinstance(source_files, list)
        or manifest.get("source_files_sha256")
        != canonical_json_sha256(source_files)
    ):
        raise RevisionProgressError(
            "Held-out evaluator unavailability source-files hash mismatch"
        )
    source_dir = (
        results_root
        / "ablation_v2"
        / EVALUATOR_UNAVAILABLE_SOURCE_MODEL
    )
    source_specs = (
        ("plan", "plan.jsonl"),
        ("checkpoint", "checkpoint.json"),
        ("records", "records.jsonl"),
        ("run_identity", "run_identity.json"),
    )
    if len(source_files) != len(source_specs):
        raise RevisionProgressError(
            "Held-out evaluator unavailability source-file cardinality mismatch"
        )
    source_contents: Dict[str, bytes] = {}
    source_artifacts: List[Dict[str, object]] = []
    for declaration, (role, filename) in zip(source_files, source_specs):
        if not isinstance(declaration, dict) or set(declaration) != {
            "role",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RevisionProgressError(
                "Held-out evaluator unavailability source declaration is malformed"
            )
        expected_path = source_dir / filename
        content = _stable_bytes(
            expected_path,
            results_root,
            "held-out evaluator unavailability source {}".format(role),
        )
        artifact = _artifact(expected_path, content)
        if declaration != {
            "role": role,
            "path": artifact["path"],
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        }:
            raise RevisionProgressError(
                "Held-out evaluator unavailability source identity mismatch"
            )
        source_contents[role] = content
        source_artifacts.append(artifact)

    plan = _jsonl(source_contents["plan"], "unavailability source plan")
    checkpoint = _json_object(
        source_contents["checkpoint"], "unavailability source checkpoint"
    )
    records = _jsonl(source_contents["records"], "unavailability source records")
    plan_by_id: Dict[str, Dict[str, object]] = {}
    for task in plan:
        work_id = str(task.get("work_id", ""))
        if not work_id or work_id in plan_by_id:
            raise RevisionProgressError(
                "Held-out evaluator unavailability source plan has missing/duplicate work_id"
            )
        plan_by_id[work_id] = task
    completed = checkpoint.get("completed_trial_ids")
    if not isinstance(completed, list):
        raise RevisionProgressError(
            "Held-out evaluator unavailability source checkpoint is malformed"
        )
    completed_ids = set(map(str, completed))
    if not completed_ids.issubset(plan_by_id):
        raise RevisionProgressError(
            "Held-out evaluator unavailability checkpoint has unknown work IDs"
        )

    expected_units: List[Dict[str, object]] = []
    seen_work_ids: Set[str] = set()
    for record in records:
        work_id = str(record.get("work_id", ""))
        task = plan_by_id.get(work_id)
        if (
            work_id not in completed_ids
            or record.get("execution_status") != "completed"
            or record.get("record_type") not in UNAVAILABLE_RECORD_TYPES
            or not isinstance(task, dict)
            or task.get("work_kind") != "rankcloak"
        ):
            continue
        if work_id in seen_work_ids:
            raise RevisionProgressError(
                "Duplicate upstream-unavailable source completion"
            )
        seen_work_ids.add(work_id)
        expected_units.append(
            {
                "terminal_status": "upstream_dependent_unavailable_not_scored",
                "source_stage": "ablation_v2",
                "source_work_id": work_id,
                "source_record_type": record.get("record_type"),
                "source_record_sha256": canonical_json_sha256(record),
                "reason_code": record.get("reason_code"),
                "generator_model_id": EVALUATOR_UNAVAILABLE_SOURCE_MODEL,
                "evaluator_model_id": EVALUATOR_UNAVAILABLE_EVALUATOR_MODEL,
                "protocol_variant": task.get("protocol_variant"),
                "payload_name": task.get("payload_name"),
                "scoring_attempted": False,
                "score_imputed": False,
            }
        )
    expected_units.sort(key=lambda row: str(row["source_work_id"]))
    if len(expected_units) != EVALUATOR_UPSTREAM_UNAVAILABLE_UNITS:
        raise RevisionProgressError(
            "Held-out evaluator unavailability does not derive exactly 48 source units"
        )
    units = manifest.get("units")
    if not isinstance(units, list) or any(
        not isinstance(unit, dict)
        or set(unit) != EVALUATOR_UNAVAILABILITY_UNIT_FIELDS
        for unit in units
    ):
        raise RevisionProgressError(
            "Held-out evaluator unavailability units are malformed"
        )
    if manifest.get("units_sha256") != canonical_json_sha256(units):
        raise RevisionProgressError(
            "Held-out evaluator unavailability units hash mismatch"
        )
    if units != expected_units:
        raise RevisionProgressError(
            "Held-out evaluator unavailability units differ from source records"
        )

    manifest_artifact = _artifact(manifest_path, manifest_content)
    binding = {
        "manifest_artifact": manifest_artifact,
        "manifest_sha256": claimed_manifest_hash,
        "source_files_sha256": manifest["source_files_sha256"],
        "source_file_artifacts": source_artifacts,
        "units_sha256": manifest["units_sha256"],
        "scoreable_evaluator_units": EVALUATOR_SCOREABLE_UNITS,
        "upstream_dependent_unavailable_units": (
            EVALUATOR_UPSTREAM_UNAVAILABLE_UNITS
        ),
        "terminal_accounted_units": EVALUATOR_FROZEN_TARGET_UNITS,
        "counting_policy": (
            "heldout_evaluator_completed_and_unavailable_only_no_success_"
            "recovery_failure_or_gpu_work"
        ),
    }
    source_keys = {
        (str(unit["source_stage"]), str(unit["source_work_id"]))
        for unit in units
    }
    return binding, [manifest_artifact] + source_artifacts, source_keys


def _assert_unavailability_not_in_evaluator_plans(
    results_root: Path,
    shards: Sequence[Mapping[str, object]],
    unavailable_source_keys: Set[Tuple[str, str]],
) -> None:
    if not unavailable_source_keys:
        return
    for shard in shards:
        if shard.get("component") != "heldout_evaluator":
            continue
        plan_path = Path(str(shard["path"])) / "plan.jsonl"
        plan = _jsonl(
            _stable_bytes(plan_path, results_root, "held-out evaluator plan"),
            "held-out evaluator plan",
        )
        for task in plan:
            key = (str(task.get("source_stage", "")), str(task.get("source_work_id", "")))
            if key in unavailable_source_keys:
                raise RevisionProgressError(
                    "An upstream-unavailable evaluator unit also appears in a scoring plan"
                )


def build_progress_snapshot(
    results_root: Path,
    generated_at: Optional[datetime] = None,
    _baseline: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Build a snapshot without writing or modifying any result shard."""

    root = Path(results_root)
    if root.is_symlink() or not root.is_dir():
        raise RevisionProgressError("results_root must be a real directory")
    root = root.resolve(strict=True)
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise RevisionProgressError("generated_at must be timezone-aware")
    now = now.astimezone(timezone.utc)
    baseline = dict(_baseline) if _baseline is not None else _verified_baseline(root)
    targets = {str(key): int(value) for key, value in dict(baseline["targets"]).items()}
    if set(targets) != set(STAGE_ORDER) or any(value < 0 for value in targets.values()):
        raise RevisionProgressError("Confirmatory target mapping is incomplete")

    shards: List[Dict[str, object]] = []
    artifacts: List[Dict[str, object]] = []
    if baseline.get("projection_artifact") is not None:
        artifacts.append(dict(baseline["projection_artifact"]))
    recovered_errors: List[Dict[str, object]] = []
    intervals: List[Dict[str, object]] = []
    for stage, component, path in _discover_shards(root):
        row, shard_artifacts, recovered = _scan_shard(path, root, stage, component)
        shards.append(row)
        artifacts.extend(shard_artifacts)
        recovered_errors.extend(recovered)
        # Reconstruct the already validated intervals once, without retaining
        # mutable raw event records in the public snapshot.
        event_path = path / "events.jsonl"
        checkpoint_path = path / "checkpoint.json"
        record_path = path / "records.jsonl"
        events = _jsonl(_stable_bytes(event_path, root, "events"), "events") if event_path.exists() else []
        checkpoint = _json_object(_stable_bytes(checkpoint_path, root, "checkpoint"), "checkpoint")
        records = _jsonl(_stable_bytes(record_path, root, "records"), "records") if record_path.exists() else []
        intervals.extend(
            _occupancy_intervals(events, checkpoint, records, path, component, str(row["model_id"]))
        )
    (
        evaluator_unavailability,
        evaluator_unavailability_artifacts,
        unavailable_source_keys,
    ) = _scan_evaluator_unavailability(root)
    _assert_unavailability_not_in_evaluator_plans(
        root, shards, unavailable_source_keys
    )
    if (
        evaluator_unavailability is not None
        and targets[EVALUATOR_STAGE] != EVALUATOR_FROZEN_TARGET_UNITS
    ):
        raise RevisionProgressError(
            "Held-out evaluator target disagrees with its unavailability manifest"
        )
    artifacts_by_path = {str(artifact["path"]): artifact for artifact in artifacts}
    for artifact in evaluator_unavailability_artifacts:
        path = str(artifact["path"])
        previous = artifacts_by_path.get(path)
        if previous is not None:
            if previous != artifact:
                raise RevisionProgressError(
                    "An unavailability lineage source changed during the progress scan"
                )
            continue
        artifacts.append(artifact)
        artifacts_by_path[path] = artifact

    (
        detector_counts,
        detector_artifacts,
        detector_intervals,
    ) = _scan_detector_outputs(root)
    artifacts.extend(detector_artifacts)
    intervals.extend(detector_intervals)
    _assert_nonoverlap(intervals)
    stage_rows = []
    aggregate_counts = Counter()
    aggregate_recovery = Counter()
    for stage in STAGE_ORDER:
        relevant = [row for row in shards if row["stage"] == stage]
        observed = Counter()
        stage_recovery = Counter()
        for row in relevant:
            for key in ("completed", "successes", "failures", "unavailable"):
                observed[key] += int(row[key])
            stage_recovery.update(row["recovery_counts"])
        unavailability_count = 0
        if stage == EVALUATOR_STAGE and evaluator_unavailability is not None:
            unavailability_count = int(
                evaluator_unavailability["upstream_dependent_unavailable_units"]
            )
            observed["completed"] += unavailability_count
            observed["unavailable"] += unavailability_count
        if stage == DETECTOR_STAGE:
            observed.update(detector_counts)
        total = targets[stage]
        if sum(int(row["total"]) for row in relevant) + unavailability_count > total:
            raise RevisionProgressError("Observed shard plans exceed the frozen stage target")
        if observed["completed"] + observed["failures"] > total:
            raise RevisionProgressError("Observed terminal work exceeds the frozen stage target")
        if observed["successes"] + observed["unavailable"] != observed["completed"]:
            raise RevisionProgressError("Completed work does not partition into success/unavailable")
        stage_row = {
            "stage": stage,
            "completed": observed["completed"],
            "total": total,
            "successes": observed["successes"],
            "failures": observed["failures"],
            "unavailable": observed["unavailable"],
            "recovery_counts": {
                key: int(stage_recovery[key])
                for key in (
                    "payload_bearing_recovery_attempted",
                    "successful_payload_recoveries",
                    "payload_recovery_failures",
                    "unavailable",
                )
            },
            "remaining": total - observed["completed"] - observed["failures"],
            "target_source": "fail_closed_verified_compute_projection",
        }
        stage_rows.append(stage_row)
        aggregate_counts.update({key: stage_row[key] for key in ("completed", "successes", "failures", "unavailable")})
        aggregate_counts["total"] += total
        aggregate_counts["remaining"] += stage_row["remaining"]
        aggregate_recovery.update(stage_row["recovery_counts"])

    current_candidates = [row for row in shards if row["state"] == "in_progress_or_paused"]
    current = None
    if current_candidates:
        selected = max(current_candidates, key=lambda row: str(row["checkpoint_updated_at"]))
        current = {
            "stage": selected["stage"],
            "component": selected["component"],
            "model_id": selected["model_id"],
            "condition": selected["current_condition"],
            "trial_id": selected["current_trial_id"],
            "checkpoint_updated_at": selected["checkpoint_updated_at"],
            "liveness_claim": "none_durable_state_only",
        }
    last_checkpoint = None
    if shards:
        selected = max(shards, key=lambda row: str(row["checkpoint_updated_at"]))
        last_checkpoint = {
            "path": str(Path(str(selected["path"])) / "checkpoint.json"),
            "sha256": selected["checkpoint_sha256"],
            "updated_at": selected["checkpoint_updated_at"],
            "stage": selected["stage"],
            "model_id": selected["model_id"],
        }

    monitored_seconds = sum(float(row["seconds"]) for row in intervals)
    verified_prior_seconds = _finite_nonnegative(
        baseline["verified_prior_seconds"], "verified prior seconds"
    )
    cumulative_seconds = verified_prior_seconds + monitored_seconds
    completion_times = sorted(
        _timestamp(value, "terminal completion time")
        for shard in shards
        for value in shard["terminal_completion_times"]
    )
    rolling_window = completion_times[-50:]
    rolling_rate = None
    rolling_seconds = None
    if len(rolling_window) >= 2 and rolling_window[-1] > rolling_window[0]:
        rolling_seconds = (rolling_window[-1] - rolling_window[0]).total_seconds()
        rolling_rate = (len(rolling_window) - 1) * 3600.0 / rolling_seconds
    elif aggregate_counts["completed"] and monitored_seconds > 0:
        rolling_seconds = monitored_seconds
        rolling_rate = aggregate_counts["completed"] * 3600.0 / monitored_seconds
    measured_completed_rate = (
        aggregate_counts["completed"] * 3600.0 / monitored_seconds
        if monitored_seconds > 0
        else None
    )
    measured_success_rate = (
        aggregate_counts["successes"] * 3600.0 / monitored_seconds
        if monitored_seconds > 0
        else None
    )
    if aggregate_counts["remaining"] == 0:
        eta = {
            "status": "complete",
            "rolling_eta_seconds": 0.0,
            "rolling_eta_hours": 0.0,
            "estimated_completion_at": now.isoformat(),
        }
    elif rolling_rate is None or rolling_rate <= 0:
        eta = {
            "status": "unavailable_no_confirmatory_throughput",
            "rolling_eta_seconds": None,
            "rolling_eta_hours": None,
            "estimated_completion_at": None,
        }
    else:
        eta_seconds = aggregate_counts["remaining"] / rolling_rate * 3600.0
        eta = {
            "status": "available_rolling_observed_rate",
            "rolling_eta_seconds": eta_seconds,
            "rolling_eta_hours": eta_seconds / 3600.0,
            "estimated_completion_at": (now + timedelta(seconds=eta_seconds)).isoformat(),
        }

    if aggregate_counts["completed"] == 0 and aggregate_counts["failures"] == 0:
        status = "not_started"
    elif aggregate_counts["remaining"] > 0:
        status = "in_progress_or_paused"
    elif aggregate_counts["failures"]:
        status = "complete_with_failures"
    else:
        status = "complete"

    artifacts = sorted(artifacts, key=lambda row: str(row["path"]))
    artifact_paths = [str(row["path"]) for row in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise RevisionProgressError("Duplicate source artifact in progress snapshot")
    value: Dict[str, object] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "manifest_type": PROGRESS_MANIFEST_TYPE,
        "generated_at": now.isoformat(),
        "results_root": str(root),
        "status": status,
        "status_semantics": "durable_checkpoint_state_only_no_process_liveness_claim",
        "protocol_contract_revision": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "current": current,
        "counts": dict(aggregate_counts),
        "recovery_counts": {
            key: int(aggregate_recovery[key])
            for key in (
                "payload_bearing_recovery_attempted",
                "successful_payload_recoveries",
                "payload_recovery_failures",
                "unavailable",
            )
        },
        "stage_progress": stage_rows,
        "shards": shards,
        "gpu": {
            "verified_prior_seconds": verified_prior_seconds,
            "verified_prior_gpu_hours": verified_prior_seconds / 3600.0,
            "prior_components": baseline["prior_components"],
            "monitored_confirmatory_seconds": monitored_seconds,
            "monitored_confirmatory_gpu_hours": monitored_seconds / 3600.0,
            "cumulative_actual_seconds": cumulative_seconds,
            "cumulative_actual_gpu_hours": cumulative_seconds / 3600.0,
            "confirmatory_intervals": intervals,
            "confirmatory_derivation": "durable_GPU_wall_spans_not_sum_of_trial_times",
            "legacy_and_invalidated_excluded_from_throughput": True,
        },
        "throughput": {
            "unit": "terminal_work_units_per_confirmatory_GPU_hour",
            "completed_per_gpu_hour": measured_completed_rate,
            "successful_per_gpu_hour": measured_success_rate,
            "rolling_completed_per_gpu_hour": rolling_rate,
            "rolling_window_completed": len(rolling_window),
            "rolling_window_seconds": rolling_seconds,
            "denominator_excludes_verified_prior_charges": True,
        },
        "eta": eta,
        "last_checkpoint": last_checkpoint,
        "recovered_errors": recovered_errors,
        "verified_compute_projection_sha256": baseline["projection_sha256"],
        "projection_decision_at_baseline": baseline.get("projection_decision"),
        "heldout_evaluator_upstream_unavailability": evaluator_unavailability,
        "source_artifacts": artifacts,
    }
    value[PROGRESS_HASH_FIELD] = _canonical_digest(value)

    # Detect a runner write that raced the scan.  The caller may retry safely.
    for artifact in artifacts:
        path = Path(str(artifact["path"]))
        content = _stable_progress_source_bytes(
            path, root, "progress source recheck"
        )
        if _artifact(path, content) != artifact:
            raise RevisionProgressError("A progress source changed during snapshot construction")
    return value


def atomic_write_progress_snapshot(
    path: Path, value: Mapping[str, object], results_root: Optional[Path] = None
) -> Path:
    """Atomically replace only the designated progress JSON file."""

    target = Path(path)
    root = Path(results_root or value.get("results_root", target.parent)).resolve(strict=True)
    _reject_symlink_chain(target, root, "progress output")
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise RevisionProgressError("Progress output is not a regular file")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RevisionProgressError("Progress output parent is not a real directory")
    document = dict(value)
    if document.get(PROGRESS_HASH_FIELD) != _canonical_digest(document):
        raise RevisionProgressError("Refusing to write an invalid progress self-hash")
    payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".confirmatory_progress_v1.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.resolve(strict=True)


def update_progress_snapshot(
    results_root: Path, output_path: Optional[Path] = None, retries: int = 3
) -> Dict[str, object]:
    """Build and atomically publish a fresh snapshot, retrying source races."""

    if retries < 1:
        raise RevisionProgressError("retries must be positive")
    root = Path(results_root).resolve(strict=True)
    target = Path(output_path) if output_path is not None else root / "confirmatory_progress_v1.json"
    last_error: Optional[Exception] = None
    for _ in range(retries):
        try:
            value = build_progress_snapshot(root)
            atomic_write_progress_snapshot(target, value, root)
            return value
        except RevisionProgressError as exc:
            last_error = exc
            if "changed" not in str(exc):
                raise
    raise RevisionProgressError("Progress sources remained unstable: {}".format(last_error))


def verify_progress_snapshot(path: Path) -> Dict[str, object]:
    """Fail closed on malformed, tampered, stale, or symlinked snapshots."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RevisionProgressError("Progress snapshot must be a regular non-symlink file")
    value = _json_object(target.read_bytes(), "progress snapshot")
    if value.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise RevisionProgressError("Progress schema mismatch")
    if value.get("manifest_type") != PROGRESS_MANIFEST_TYPE:
        raise RevisionProgressError("Progress manifest_type mismatch")
    if value.get(PROGRESS_HASH_FIELD) != _canonical_digest(value):
        raise RevisionProgressError("Progress snapshot self-hash mismatch")
    _timestamp(value.get("generated_at"), "progress generated_at")
    decision = value.get("projection_decision_at_baseline")
    if (
        value.get("verified_compute_projection_sha256")
        != APPROVED_165H_PROJECTION_SHA256
        or not isinstance(decision, dict)
        or decision.get("go") is not True
        or decision.get("status") != "go_within_budget"
    ):
        raise RevisionProgressError("Progress does not bind the approved 165-hour GO projection")
    root = Path(str(value.get("results_root", "")))
    if root.is_symlink() or not root.is_dir():
        raise RevisionProgressError("Progress results_root is unavailable")
    root = root.resolve(strict=True)
    (
        current_detector_counts,
        current_detector_artifacts,
        current_detector_intervals,
    ) = _scan_detector_outputs(root)
    (
        current_unavailability,
        current_unavailability_artifacts,
        unavailable_source_keys,
    ) = _scan_evaluator_unavailability(root)
    if value.get("heldout_evaluator_upstream_unavailability") != current_unavailability:
        raise RevisionProgressError(
            "Progress snapshot is stale: evaluator unavailability binding changed"
        )
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise RevisionProgressError("Progress counts are malformed")
    numeric = {key: int(counts.get(key, -1)) for key in ("completed", "total", "successes", "failures", "unavailable", "remaining")}
    if any(item < 0 for item in numeric.values()):
        raise RevisionProgressError("Progress counts must be nonnegative")
    if numeric["successes"] + numeric["unavailable"] != numeric["completed"]:
        raise RevisionProgressError("Progress success/unavailable partition mismatch")
    if numeric["completed"] + numeric["failures"] + numeric["remaining"] != numeric["total"]:
        raise RevisionProgressError("Progress terminal/remaining partition mismatch")
    stage_progress = value.get("stage_progress")
    if not isinstance(stage_progress, list) or len(stage_progress) != len(STAGE_ORDER):
        raise RevisionProgressError("Progress stage_progress is malformed")
    stages: Dict[str, Mapping[str, object]] = {}
    aggregate_from_stages = Counter()
    for row in stage_progress:
        if not isinstance(row, dict):
            raise RevisionProgressError("Progress stage row is malformed")
        stage = str(row.get("stage", ""))
        if stage not in STAGE_ORDER or stage in stages:
            raise RevisionProgressError("Progress stage identities are malformed")
        stages[stage] = row
        row_counts = {
            key: int(row.get(key, -1))
            for key in (
                "completed",
                "total",
                "successes",
                "failures",
                "unavailable",
                "remaining",
            )
        }
        if (
            min(row_counts.values()) < 0
            or row_counts["successes"] + row_counts["unavailable"]
            != row_counts["completed"]
            or row_counts["completed"]
            + row_counts["failures"]
            + row_counts["remaining"]
            != row_counts["total"]
        ):
            raise RevisionProgressError("Progress stage count partition mismatch")
        aggregate_from_stages.update(row_counts)
    if set(stages) != set(STAGE_ORDER) or any(
        aggregate_from_stages[key] != numeric[key]
        for key in numeric
    ):
        raise RevisionProgressError("Progress stage/aggregate counts mismatch")
    detector_row = stages[DETECTOR_STAGE]
    if any(
        int(detector_row.get(key, -1)) != current_detector_counts[key]
        for key in ("completed", "successes", "failures", "unavailable")
    ):
        raise RevisionProgressError(
            "Progress detector counts differ from final detector outputs"
        )
    shards = value.get("shards")
    if not isinstance(shards, list) or any(not isinstance(row, dict) for row in shards):
        raise RevisionProgressError("Progress shards are malformed")
    _assert_unavailability_not_in_evaluator_plans(
        root, shards, unavailable_source_keys
    )
    if current_unavailability is not None:
        evaluator_row = stages[EVALUATOR_STAGE]
        evaluator_shards = [
            row for row in shards if row.get("stage") == EVALUATOR_STAGE
        ]
        unavailable_count = EVALUATOR_UPSTREAM_UNAVAILABLE_UNITS
        expected_evaluator_counts = {
            "completed": sum(int(row.get("completed", 0)) for row in evaluator_shards)
            + unavailable_count,
            "successes": sum(int(row.get("successes", 0)) for row in evaluator_shards),
            "failures": sum(int(row.get("failures", 0)) for row in evaluator_shards),
            "unavailable": sum(int(row.get("unavailable", 0)) for row in evaluator_shards)
            + unavailable_count,
        }
        if (
            int(evaluator_row.get("total", -1)) != EVALUATOR_FROZEN_TARGET_UNITS
            or any(
                int(evaluator_row.get(key, -1)) != expected
                for key, expected in expected_evaluator_counts.items()
            )
        ):
            raise RevisionProgressError(
                "Progress evaluator unavailability counting mismatch"
            )
        expected_recovery = Counter()
        for shard in evaluator_shards:
            recovery = shard.get("recovery_counts")
            if not isinstance(recovery, dict):
                raise RevisionProgressError(
                    "Progress evaluator shard recovery counts are malformed"
                )
            expected_recovery.update(
                {key: int(value) for key, value in recovery.items()}
            )
        stage_recovery = evaluator_row.get("recovery_counts")
        if not isinstance(stage_recovery, dict) or any(
            int(stage_recovery.get(key, -1)) != expected_recovery[key]
            for key in (
                "payload_bearing_recovery_attempted",
                "successful_payload_recoveries",
                "payload_recovery_failures",
                "unavailable",
            )
        ):
            raise RevisionProgressError(
                "Progress evaluator unavailability was counted as recovery"
            )
    recovery_counts = value.get("recovery_counts")
    if not isinstance(recovery_counts, dict):
        raise RevisionProgressError("Progress recovery_counts are malformed")
    attempted = int(recovery_counts.get("payload_bearing_recovery_attempted", -1))
    recovered = int(recovery_counts.get("successful_payload_recoveries", -1))
    recovery_failures = int(recovery_counts.get("payload_recovery_failures", -1))
    recovery_unavailable = int(recovery_counts.get("unavailable", -1))
    if (
        min(attempted, recovered, recovery_failures, recovery_unavailable) < 0
        or recovered + recovery_failures != attempted
        or recovery_unavailable > numeric["unavailable"]
    ):
        raise RevisionProgressError("Progress payload-recovery partition mismatch")
    gpu = value.get("gpu")
    if not isinstance(gpu, dict):
        raise RevisionProgressError("Progress GPU accounting is malformed")
    prior = _finite_nonnegative(gpu.get("verified_prior_seconds"), "verified prior seconds")
    monitored = _finite_nonnegative(gpu.get("monitored_confirmatory_seconds"), "monitored seconds")
    cumulative = _finite_nonnegative(gpu.get("cumulative_actual_seconds"), "cumulative seconds")
    if cumulative != prior + monitored or gpu.get("cumulative_actual_gpu_hours") != cumulative / 3600.0:
        raise RevisionProgressError("Progress GPU total mismatch")
    intervals = gpu.get("confirmatory_intervals")
    if not isinstance(intervals, list):
        raise RevisionProgressError("Progress GPU intervals are malformed")
    _assert_nonoverlap(intervals)
    if sum(float(row["seconds"]) for row in intervals) != monitored:
        raise RevisionProgressError("Progress interval sum mismatch")
    stored_detector_intervals = [
        row
        for row in intervals
        if isinstance(row, dict) and row.get("component") == DETECTOR_STAGE
    ]
    if stored_detector_intervals != current_detector_intervals:
        raise RevisionProgressError(
            "Progress detector GPU provenance differs from the final manifest"
        )
    artifacts = value.get("source_artifacts")
    if not isinstance(artifacts, list):
        raise RevisionProgressError("Progress source_artifacts are malformed")
    paths = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RevisionProgressError("Progress source artifact is malformed")
        source = Path(str(artifact.get("path", "")))
        content = _stable_progress_source_bytes(
            source, root, "progress source artifact"
        )
        if artifact != _artifact(source, content):
            raise RevisionProgressError("Progress snapshot is stale: {} changed".format(source))
        paths.append(str(source.resolve(strict=True)))
    if len(paths) != len(set(paths)):
        raise RevisionProgressError("Progress source artifacts are duplicated")
    artifacts_by_path = {
        str(Path(str(artifact["path"])).resolve(strict=True)): artifact
        for artifact in artifacts
    }
    for expected in current_unavailability_artifacts:
        if artifacts_by_path.get(str(expected["path"])) != expected:
            raise RevisionProgressError(
                "Progress evaluator unavailability artifact binding is incomplete"
            )
    for expected in current_detector_artifacts:
        if artifacts_by_path.get(str(expected["path"])) != expected:
            raise RevisionProgressError(
                "Progress detector artifact binding is incomplete"
            )
    return {
        "status": "ok",
        "path": str(target.resolve(strict=True)),
        "progress_sha256": value[PROGRESS_HASH_FIELD],
        "generated_at": value["generated_at"],
        "execution_status": value["status"],
        "current": value["current"],
        "counts": value["counts"],
        "recovery_counts": value["recovery_counts"],
        "cumulative_actual_gpu_hours": gpu["cumulative_actual_gpu_hours"],
        "measured_completed_per_gpu_hour": value["throughput"]["completed_per_gpu_hour"],
        "rolling_eta_hours": value["eta"]["rolling_eta_hours"],
        "last_checkpoint": value["last_checkpoint"],
        "recovered_error_count": len(value["recovered_errors"]),
    }


__all__ = [
    "DEFAULT_PROGRESS_RELATIVE_PATH",
    "PROGRESS_HASH_FIELD",
    "PROGRESS_MANIFEST_TYPE",
    "PROGRESS_SCHEMA_VERSION",
    "RevisionProgressError",
    "atomic_write_progress_snapshot",
    "build_progress_snapshot",
    "update_progress_snapshot",
    "verify_progress_snapshot",
]
