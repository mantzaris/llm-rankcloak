"""Fail-closed, external invalidation records for immutable revision shards.

The invalidated shard is never written, moved, renamed, or deleted.  A caller
must attest that its writer has stopped before this module will snapshot it.
The resulting registry entry is atomically published outside the shard and can
later be re-verified against the shard's exact bytes and filesystem identity.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_artifacts import canonical_json_bytes, canonical_json_sha256


INVALIDATION_SCHEMA_VERSION = "1.0"
INVALIDATION_MANIFEST_TYPE = "revision_shard_invalidation"
INVALIDATION_HASH_FIELD = "invalidation_manifest_sha256"
REQUIRED_SHARD_FILES = (
    "checkpoint.json",
    "hardware_manifest.json",
    "model_manifest.json",
    "payload_manifest.json",
    "plan.jsonl",
    "run_identity.json",
    "runtime_manifest.json",
    "source_manifest.json",
)
_REASON_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RevisionInvalidationError(RuntimeError):
    """Base class for invalidation failures."""


class StoppedShardConfirmationError(RevisionInvalidationError):
    """Raised when the caller has not explicitly confirmed that writers stopped."""


class ShardIntegrityError(RevisionInvalidationError):
    """Raised when a shard is malformed, internally inconsistent, or changes."""


class InvalidationEntryExistsError(RevisionInvalidationError):
    """Raised when a registry path already exists and therefore cannot be replaced."""


def _load_json_object(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardIntegrityError("Cannot read JSON object {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ShardIntegrityError("JSON artifact is not an object: {}".format(path))
    return value


def _load_jsonl(path: Path, *, optional: bool = False) -> List[Dict[str, object]]:
    if optional and not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ShardIntegrityError(
                        "Invalid JSONL at {}:{}: {}".format(path, line_number, exc)
                    ) from exc
                if not isinstance(value, dict):
                    raise ShardIntegrityError(
                        "JSONL row is not an object at {}:{}".format(path, line_number)
                    )
                rows.append(value)
    except OSError as exc:
        raise ShardIntegrityError("Cannot read JSONL {}: {}".format(path, exc)) from exc
    return rows


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _validate_shard_path(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ShardIntegrityError("Shard path must not be a symlink: {}".format(path))
    try:
        resolved = path.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise ShardIntegrityError("Shard path does not resolve: {}".format(path)) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ShardIntegrityError("Shard path is not a real directory: {}".format(resolved))
    missing = [name for name in REQUIRED_SHARD_FILES if not (resolved / name).is_file()]
    if missing:
        raise ShardIntegrityError(
            "Shard is missing required artifacts: {}".format(", ".join(missing))
        )
    return resolved


def _stat_record(result: os.stat_result) -> Dict[str, int]:
    return {
        "device": int(result.st_dev),
        "inode": int(result.st_ino),
        "mode": int(stat.S_IMODE(result.st_mode)),
        "size_bytes": int(result.st_size),
        "mtime_ns": int(result.st_mtime_ns),
        "ctime_ns": int(result.st_ctime_ns),
    }


def _hash_regular_file(path: Path, expected: os.stat_result) -> Tuple[str, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ShardIntegrityError("Cannot safely open shard file {}: {}".format(path, exc)) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ShardIntegrityError("Shard entry is not a regular file: {}".format(path))
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise ShardIntegrityError("Shard file changed while opening: {}".format(path))
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if _stat_record(opened) != _stat_record(finished):
            raise ShardIntegrityError("Shard file changed while hashing: {}".format(path))
        return digest.hexdigest(), finished
    finally:
        os.close(descriptor)


def _walk_entries(root: Path) -> Tuple[List[Tuple[str, os.stat_result]], List[Tuple[str, os.stat_result]]]:
    directories: List[Tuple[str, os.stat_result]] = [(".", root.stat(follow_symlinks=False))]
    files: List[Tuple[str, os.stat_result]] = []
    pending: List[Tuple[Path, str]] = [(root, ".")]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ShardIntegrityError("Cannot enumerate shard directory {}: {}".format(directory, exc)) from exc
        for entry in entries:
            relative = entry.name if relative_directory == "." else "{}/{}".format(relative_directory, entry.name)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ShardIntegrityError("Cannot stat shard entry {}: {}".format(entry.path, exc)) from exc
            if stat.S_ISLNK(info.st_mode):
                raise ShardIntegrityError("Shard must not contain symlinks: {}".format(relative))
            if stat.S_ISDIR(info.st_mode):
                directories.append((relative, info))
                pending.append((Path(entry.path), relative))
            elif stat.S_ISREG(info.st_mode):
                files.append((relative, info))
            else:
                raise ShardIntegrityError("Shard contains a non-regular entry: {}".format(relative))
    directories.sort(key=lambda pair: pair[0])
    files.sort(key=lambda pair: pair[0])
    return directories, files


def snapshot_shard(shard_path: Path) -> Dict[str, object]:
    """Hash a shard tree without following links or accepting concurrent writes."""

    root = _validate_shard_path(Path(shard_path))
    directory_entries, file_entries = _walk_entries(root)
    directories = [
        {"path": relative, **_stat_record(info)}
        for relative, info in directory_entries
    ]
    files: List[Dict[str, object]] = []
    for relative, before in file_entries:
        digest, after = _hash_regular_file(root / relative, before)
        files.append({"path": relative, "sha256": digest, **_stat_record(after)})
    value: Dict[str, object] = {
        "schema_version": INVALIDATION_SCHEMA_VERSION,
        "directory_count": len(directories),
        "file_count": len(files),
        "directories": directories,
        "files": files,
        "directories_sha256": canonical_json_sha256(directories),
        "files_sha256": canonical_json_sha256(files),
    }
    value["shard_tree_sha256"] = canonical_json_sha256(
        {
            "directories_sha256": value["directories_sha256"],
            "files_sha256": value["files_sha256"],
        }
    )
    return value


def _snapshot_file_hash(snapshot: Mapping[str, object], relative_path: str) -> str:
    matches = [
        row
        for row in snapshot.get("files", [])
        if isinstance(row, dict) and row.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ShardIntegrityError("Snapshot does not contain exactly one {}".format(relative_path))
    return str(matches[0].get("sha256"))


def _require_sha256(value: object, label: str) -> str:
    rendered = str(value)
    if not _SHA256.fullmatch(rendered):
        raise ShardIntegrityError("{} is not a lowercase SHA-256".format(label))
    return rendered


def _parse_identity_args(values: object) -> Dict[str, str]:
    if not isinstance(values, list):
        raise ShardIntegrityError("run_identity command_line_args must be a list")
    result: Dict[str, str] = {}
    for raw in values:
        text = str(raw)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in result:
            raise ShardIntegrityError("Duplicate run-identity argument: {}".format(key))
        result[key] = value
    return result


def _verify_records(
    plan: Sequence[Mapping[str, object]],
    checkpoint: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    planned_ids = [str(row.get("work_id")) for row in plan]
    if any(value in {"", "None"} for value in planned_ids):
        raise ShardIntegrityError("Every plan row must have a nonempty work_id")
    if len(set(planned_ids)) != len(planned_ids):
        raise ShardIntegrityError("Plan contains duplicate work IDs")
    planned = set(planned_ids)
    attempts: Counter = Counter()
    seen_attempts = set()
    completions: Dict[str, Mapping[str, object]] = {}
    failures: Dict[str, Mapping[str, object]] = {}
    for record in records:
        work_id = str(record.get("work_id"))
        if work_id not in planned:
            raise ShardIntegrityError("Result contains unplanned work ID: {}".format(work_id))
        try:
            attempt_index = int(record.get("attempt_index", 1))
        except (TypeError, ValueError) as exc:
            raise ShardIntegrityError("Invalid attempt index for {}".format(work_id)) from exc
        if attempt_index < 1:
            raise ShardIntegrityError("Attempt indices must be positive")
        key = (work_id, attempt_index)
        if key in seen_attempts:
            raise ShardIntegrityError("Duplicate durable attempt {} for {}".format(attempt_index, work_id))
        seen_attempts.add(key)
        attempts[work_id] = max(attempts[work_id], attempt_index)
        status_value = record.get("execution_status")
        if status_value == "completed":
            if work_id in completions:
                raise ShardIntegrityError("Multiple durable completions for {}".format(work_id))
            completions[work_id] = record
        elif status_value == "failed":
            failures[work_id] = record
        else:
            raise ShardIntegrityError("Invalid execution_status for {}".format(work_id))

    checkpoint_completed = list(map(str, checkpoint.get("completed_trial_ids", [])))
    checkpoint_failed = list(map(str, checkpoint.get("failed_trial_ids", [])))
    if len(checkpoint_completed) != len(set(checkpoint_completed)):
        raise ShardIntegrityError("Checkpoint contains duplicate completed IDs")
    if len(checkpoint_failed) != len(set(checkpoint_failed)):
        raise ShardIntegrityError("Checkpoint contains duplicate failed IDs")
    if set(checkpoint_completed) != set(completions):
        raise ShardIntegrityError("Checkpoint completed IDs do not match durable completions")
    current_failures = set(failures) - set(completions)
    if set(checkpoint_failed) != current_failures:
        raise ShardIntegrityError("Checkpoint failed IDs do not match durable failures")
    checkpoint_attempts = {
        str(key): int(value)
        for key, value in dict(checkpoint.get("attempt_counts", {})).items()
    }
    if checkpoint_attempts != dict(attempts):
        raise ShardIntegrityError("Checkpoint attempt counts do not match durable records")
    return {
        "planned_work_units": len(planned_ids),
        "durable_attempt_rows": len(records),
        "completed_work_units": len(completions),
        "failed_work_units_current": len(current_failures),
    }


def _aware_timestamp(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ShardIntegrityError("{} must be an ISO-8601 timestamp".format(label)) from exc
    if parsed.tzinfo is None:
        raise ShardIntegrityError("{} must include a timezone".format(label))
    return parsed


def _nonnegative_seconds(value: object, label: str) -> float:
    try:
        rendered = float(value)
    except (TypeError, ValueError) as exc:
        raise ShardIntegrityError("{} must be numeric".format(label)) from exc
    if not math.isfinite(rendered) or rendered < 0:
        raise ShardIntegrityError("{} must be finite and nonnegative".format(label))
    return rendered


def inspect_shard_execution(
    shard_path: Path,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Derive stopped/incomplete state and exact observed invalid-run GPU charge."""

    root = _validate_shard_path(Path(shard_path))
    plan = _load_jsonl(root / "plan.jsonl")
    checkpoint = _load_json_object(root / "checkpoint.json")
    records = _load_jsonl(root / "records.jsonl", optional=True)
    progress = _verify_records(plan, checkpoint, records)
    events = _load_jsonl(root / "events.jsonl", optional=True)
    identity = _load_json_object(root / "run_identity.json")
    identity_args = _parse_identity_args(identity.get("command_line_args"))
    hardware = _load_json_object(root / "hardware_manifest.json")
    selected_gpu = hardware.get("selected_gpu_uuid") or identity_args.get("gpu_uuid")
    if selected_gpu in {None, "", "cpu"}:
        raise ShardIntegrityError("GPU invalidation charge requires a selected GPU UUID")
    selected_gpu = str(selected_gpu)
    identity_gpu = identity_args.get("gpu_uuid")
    if identity_gpu not in {None, selected_gpu}:
        raise ShardIntegrityError("Hardware and run-identity GPU UUIDs differ")

    profiles = [event for event in events if event.get("event") == "memory_profile"]
    if not profiles:
        raise ShardIntegrityError(
            "Stopped GPU shard lacks a final memory_profile event; incurred charge is unknowable"
        )
    spans: List[Dict[str, object]] = []
    incurred_gpu_seconds = 0.0
    for index, profile in enumerate(profiles):
        prefix = "memory_profile[{}]".format(index)
        started = _aware_timestamp(profile.get("started_at"), prefix + ".started_at")
        ended = _aware_timestamp(profile.get("at"), prefix + ".at")
        seconds = (ended - started).total_seconds()
        if seconds < 0:
            raise ShardIntegrityError("Memory-profile end precedes start")
        if str(profile.get("selected_gpu_uuid")) != selected_gpu:
            raise ShardIntegrityError("Memory-profile GPU UUID differs from run identity")
        poll_seconds = _nonnegative_seconds(
            profile.get("poll_interval_seconds"), prefix + ".poll_interval_seconds"
        )
        if poll_seconds <= 0:
            raise ShardIntegrityError("Memory-profile poll interval must be positive")
        sample_counts: Dict[str, int] = {}
        for field in (
            "sample_count",
            "selected_gpu_sample_count",
            "process_rss_sample_count",
        ):
            try:
                count = int(profile.get(field))
            except (TypeError, ValueError) as exc:
                raise ShardIntegrityError("{} must be an integer".format(prefix + "." + field)) from exc
            if count <= 0:
                raise ShardIntegrityError("{} must be positive".format(prefix + "." + field))
            sample_counts[field] = count
        memory_values = {}
        for field in (
            "selected_gpu_initial_used_memory_mib",
            "selected_gpu_final_used_memory_mib",
            "selected_gpu_peak_used_memory_mib_sampled",
            "process_peak_rss_bytes_sampled",
            "process_peak_rss_bytes_os_high_water",
        ):
            memory_values[field] = _nonnegative_seconds(
                profile.get(field), prefix + "." + field
            )
        if memory_values["selected_gpu_peak_used_memory_mib_sampled"] < max(
            memory_values["selected_gpu_initial_used_memory_mib"],
            memory_values["selected_gpu_final_used_memory_mib"],
        ):
            raise ShardIntegrityError("Memory-profile GPU peak is below an endpoint")
        incurred_gpu_seconds += seconds
        spans.append(
            {
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "elapsed_seconds": seconds,
                "selected_gpu_uuid": selected_gpu,
                "poll_interval_seconds": poll_seconds,
                **sample_counts,
                **memory_values,
            }
        )

    attempt_seconds = 0.0
    missing_attempt_timings = 0
    for index, record in enumerate(records):
        if record.get("execution_seconds") is None:
            missing_attempt_timings += 1
            continue
        attempt_seconds += _nonnegative_seconds(
            record["execution_seconds"], "records[{}].execution_seconds".format(index)
        )
    model_load_seconds = 0.0
    for index, event in enumerate(events):
        if event.get("event") == "model_loaded" and event.get("model_load_seconds") is not None:
            model_load_seconds += _nonnegative_seconds(
                event["model_load_seconds"], "events[{}].model_load_seconds".format(index)
            )
    if attempt_seconds + model_load_seconds > incurred_gpu_seconds + 1e-6:
        raise ShardIntegrityError(
            "Durable execution plus model-load time exceeds observed memory-profile span"
        )

    planned = int(progress["planned_work_units"])
    completed = int(progress["completed_work_units"])
    failed = int(progress["failed_work_units_current"])
    remaining = planned - completed
    if remaining < 0:
        raise ShardIntegrityError("Completed work exceeds planned work")
    finished_events = [event for event in events if event.get("event") == "session_finished"]
    last_finished_remaining = (
        int(finished_events[-1]["remaining"])
        if finished_events and finished_events[-1].get("remaining") is not None
        else None
    )
    evidence_statuses = sorted(
        {str(row.get("evidence_status")) for row in plan if row.get("evidence_status") is not None}
    )
    execution_state: Dict[str, object] = {
        "caller_confirmed_stopped": True,
        "terminal_state": "stopped_incomplete" if remaining or failed else "stopped_complete",
        "incomplete": bool(remaining or failed),
        "planned_work_units": planned,
        "completed_work_units": completed,
        "failed_work_units_current": failed,
        "remaining_work_units": remaining,
        "durable_attempt_rows": int(progress["durable_attempt_rows"]),
        "session_finished_event_observed": bool(finished_events),
        "session_finished_remaining": last_finished_remaining,
        "event_count": len(events),
        "original_evidence_statuses": evidence_statuses,
    }
    incurred_compute: Dict[str, object] = {
        "charge_status": "observed_incurred_invalidated_shard",
        "charge_policy": "memory_profile_wall_span_v1",
        "incurred_gpu_seconds": incurred_gpu_seconds,
        "selected_gpu_uuid": selected_gpu,
        "memory_profile_event_count": len(profiles),
        "memory_profile_spans": spans,
        "memory_profile_events_sha256": canonical_json_sha256(profiles),
        "durable_attempt_execution_seconds_sum": attempt_seconds,
        "durable_attempt_rows_missing_execution_seconds": missing_attempt_timings,
        "model_load_seconds_sum": model_load_seconds,
        "definition": (
            "exact sum of each hashed memory_profile interval (event.at - "
            "event.started_at); includes model load, all completed attempts, "
            "runner overhead, and any interrupted partial attempt while the "
            "selected GPU was monitored"
        ),
    }
    return execution_state, incurred_compute


def inspect_shard_identity(shard_path: Path, snapshot: Mapping[str, object]) -> Dict[str, object]:
    """Verify run/plan/checkpoint bindings and return the superseded identity."""

    root = _validate_shard_path(Path(shard_path))
    identity = _load_json_object(root / "run_identity.json")
    identity_without_hash = dict(identity)
    stored_identity_hash = _require_sha256(
        identity_without_hash.pop("run_identity_sha256", None),
        "run_identity_sha256",
    )
    if canonical_json_sha256(identity_without_hash) != stored_identity_hash:
        raise ShardIntegrityError("Run identity self-hash mismatch")

    plan = _load_jsonl(root / "plan.jsonl")
    planned_ids = [str(row.get("work_id")) for row in plan]
    planned_hash = canonical_json_sha256(planned_ids)
    if identity.get("planned_trial_count") != len(plan):
        raise ShardIntegrityError("Run identity planned count does not match plan")
    if identity.get("planned_trial_ids_sha256") != planned_hash:
        raise ShardIntegrityError("Run identity ordered-plan hash mismatch")

    checkpoint = _load_json_object(root / "checkpoint.json")
    for key in (
        "study_id",
        "config_manifest_sha256",
        "planned_trial_count",
        "planned_trial_ids_sha256",
    ):
        if checkpoint.get(key) != identity.get(key):
            raise ShardIntegrityError("Checkpoint/run-identity mismatch: {}".format(key))
    records = _load_jsonl(root / "records.jsonl", optional=True)
    progress = _verify_records(plan, checkpoint, records)

    payload_hash = _snapshot_file_hash(snapshot, "payload_manifest.json")
    if identity.get("payload_manifest_sha256") != payload_hash:
        raise ShardIntegrityError("Payload manifest is not bound by run identity")
    model_manifest = _load_json_object(root / "model_manifest.json")
    if identity.get("model_artifacts") != [model_manifest]:
        raise ShardIntegrityError("Model manifest differs from run-identity embedding")

    args = _parse_identity_args(identity.get("command_line_args"))
    bound_hashes: Dict[str, str] = {}
    for argument, filename in (
        ("source_manifest_sha256", "source_manifest.json"),
        ("runtime_manifest_sha256", "runtime_manifest.json"),
        ("hardware_manifest_sha256", "hardware_manifest.json"),
    ):
        actual = _snapshot_file_hash(snapshot, filename)
        expected = _require_sha256(args.get(argument), argument)
        if actual != expected:
            raise ShardIntegrityError("{} is not bound by run identity".format(filename))
        bound_hashes[argument] = actual

    source_manifest = _load_json_object(root / "source_manifest.json")
    source_files = source_manifest.get("files")
    if not isinstance(source_files, list):
        raise ShardIntegrityError("Source manifest files must be a list")
    source_files_hash = _require_sha256(
        source_manifest.get("files_sha256"), "source files_sha256"
    )
    if canonical_json_sha256(source_files) != source_files_hash:
        raise ShardIntegrityError("Source manifest file-list self-hash mismatch")

    config_hash = _require_sha256(
        identity.get("config_manifest_sha256"), "config_manifest_sha256"
    )
    return {
        "study_id": str(identity.get("study_id")),
        "run_identity_sha256": stored_identity_hash,
        "run_identity_file_sha256": _snapshot_file_hash(snapshot, "run_identity.json"),
        "config_manifest_sha256": config_hash,
        "payload_manifest_sha256": payload_hash,
        "planned_trial_count": len(plan),
        "planned_trial_ids_sha256": planned_hash,
        "plan_file_sha256": _snapshot_file_hash(snapshot, "plan.jsonl"),
        "checkpoint_file_sha256": _snapshot_file_hash(snapshot, "checkpoint.json"),
        "records_file_sha256": (
            _snapshot_file_hash(snapshot, "records.jsonl")
            if (root / "records.jsonl").is_file()
            else None
        ),
        "source_manifest_sha256": bound_hashes["source_manifest_sha256"],
        "source_files_sha256": source_files_hash,
        "runtime_manifest_sha256": bound_hashes["runtime_manifest_sha256"],
        "hardware_manifest_sha256": bound_hashes["hardware_manifest_sha256"],
        "model_artifacts_sha256": canonical_json_sha256(identity.get("model_artifacts")),
        "progress_at_invalidation": progress,
    }


def _validated_created_at(value: Optional[str]) -> str:
    rendered = value or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise RevisionInvalidationError("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise RevisionInvalidationError("created_at must include a timezone")
    return rendered


def _manifest_digest(value: Mapping[str, object]) -> str:
    unhashed = dict(value)
    unhashed.pop(INVALIDATION_HASH_FIELD, None)
    return canonical_json_sha256(unhashed)


def _atomic_create_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish a new file using a no-replace hard-link operation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise InvalidationEntryExistsError(
            "Invalidation registry entry already exists: {}".format(path)
        )
    content = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path), follow_symlinks=False)
        except FileExistsError as exc:
            raise InvalidationEntryExistsError(
                "Invalidation registry entry already exists: {}".format(path)
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise RevisionInvalidationError(
                    "Filesystem cannot atomically publish a no-replace registry entry"
                ) from exc
            raise
        try:
            directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_invalidation_manifest(
    shard_path: Path,
    *,
    reason_code: str,
    reason: str,
    superseding_target_namespace: str,
    superseding_stages: Sequence[str],
    confirm_stopped: bool,
    created_at: Optional[str] = None,
) -> Dict[str, object]:
    """Build a self-hashed invalidation manifest without publishing it."""

    if confirm_stopped is not True:
        raise StoppedShardConfirmationError(
            "Refusing to hash an invalidation snapshot without explicit stopped-shard confirmation"
        )
    if not _REASON_CODE.fullmatch(str(reason_code)):
        raise RevisionInvalidationError(
            "reason_code must match {}".format(_REASON_CODE.pattern)
        )
    rendered_reason = str(reason).strip()
    if not rendered_reason:
        raise RevisionInvalidationError("reason must be nonempty")
    target_namespace = str(superseding_target_namespace).strip()
    if not target_namespace:
        raise RevisionInvalidationError("superseding_target_namespace must be nonempty")
    stages = [str(stage).strip() for stage in superseding_stages]
    if not stages or any(not stage for stage in stages):
        raise RevisionInvalidationError("superseding_stages must contain nonempty values")
    if len(stages) != len(set(stages)):
        raise RevisionInvalidationError("superseding_stages must not contain duplicates")

    root = _validate_shard_path(Path(shard_path))
    first_snapshot = snapshot_shard(root)
    identity = inspect_shard_identity(root, first_snapshot)
    execution_state, incurred_compute = inspect_shard_execution(root)
    second_snapshot = snapshot_shard(root)
    if first_snapshot != second_snapshot:
        raise ShardIntegrityError("Shard changed while invalidation manifest was built")
    if target_namespace == identity["study_id"]:
        raise RevisionInvalidationError(
            "Superseding target namespace must differ from invalidated study_id"
        )

    manifest: Dict[str, object] = {
        "schema_version": INVALIDATION_SCHEMA_VERSION,
        "manifest_type": INVALIDATION_MANIFEST_TYPE,
        "created_at": _validated_created_at(created_at),
        "scientific_status": "invalidated_not_for_pooling",
        "scope": "entire_shard",
        "stop_attestation": {
            "confirmed_stopped": True,
            "method": "explicit_caller_confirmation",
        },
        "invalidation": {
            "reason_code": str(reason_code),
            "reason": rendered_reason,
        },
        "shard": {
            "absolute_path": str(root),
            "preservation_policy": "verify_in_place_never_edit_move_rename_or_delete",
            "snapshot": first_snapshot,
        },
        "superseded_identity": identity,
        "execution_state": execution_state,
        "incurred_compute": incurred_compute,
        "superseding_target": {
            "namespace": target_namespace,
            "stages": stages,
            "materialization_asserted": False,
        },
    }
    manifest[INVALIDATION_HASH_FIELD] = _manifest_digest(manifest)
    return manifest


def create_invalidation_entry(
    shard_path: Path,
    registry_entry_path: Path,
    *,
    reason_code: str,
    reason: str,
    superseding_target_namespace: str,
    superseding_stages: Sequence[str],
    confirm_stopped: bool,
    created_at: Optional[str] = None,
) -> Dict[str, object]:
    """Publish a new external registry entry and immediately verify it."""

    if confirm_stopped is not True:
        raise StoppedShardConfirmationError(
            "Refusing invalidation work without explicit stopped-shard confirmation"
        )
    root = _validate_shard_path(Path(shard_path))
    destination = Path(registry_entry_path).absolute()
    resolved_destination = destination.resolve(strict=False)
    if _is_within(resolved_destination, root):
        raise RevisionInvalidationError(
            "Invalidation registry entry must be external to the shard"
        )
    if destination.exists() or destination.is_symlink():
        raise InvalidationEntryExistsError(
            "Invalidation registry entry already exists: {}".format(destination)
        )
    manifest = build_invalidation_manifest(
        root,
        reason_code=reason_code,
        reason=reason,
        superseding_target_namespace=superseding_target_namespace,
        superseding_stages=superseding_stages,
        confirm_stopped=confirm_stopped,
        created_at=created_at,
    )
    # Close the build-to-publish interval before making the entry visible.
    if snapshot_shard(root) != manifest["shard"]["snapshot"]:
        raise ShardIntegrityError("Shard changed before invalidation entry publication")
    _atomic_create_json(destination, manifest)
    verify_invalidation_entry(destination)
    return manifest


def verify_invalidation_entry(registry_entry_path: Path) -> Dict[str, object]:
    """Fail closed unless an entry and its in-place shard remain exact."""

    entry_path = Path(registry_entry_path)
    if entry_path.is_symlink() or not entry_path.is_file():
        raise RevisionInvalidationError(
            "Invalidation registry entry must be a regular non-symlink file"
        )
    manifest = _load_json_object(entry_path)
    if manifest.get("schema_version") != INVALIDATION_SCHEMA_VERSION:
        raise RevisionInvalidationError("Unsupported invalidation schema version")
    if manifest.get("manifest_type") != INVALIDATION_MANIFEST_TYPE:
        raise RevisionInvalidationError("Invalid invalidation manifest_type")
    if manifest.get("scientific_status") != "invalidated_not_for_pooling":
        raise RevisionInvalidationError("Invalid invalidation scientific_status")
    if manifest.get("scope") != "entire_shard":
        raise RevisionInvalidationError("Invalid invalidation scope")
    invalidation = manifest.get("invalidation")
    if not isinstance(invalidation, dict):
        raise RevisionInvalidationError("Invalidation entry lacks a reason")
    if not _REASON_CODE.fullmatch(str(invalidation.get("reason_code", ""))):
        raise RevisionInvalidationError("Invalid invalidation reason_code")
    if not str(invalidation.get("reason", "")).strip():
        raise RevisionInvalidationError("Invalidation reason must be nonempty")
    stored_hash = _require_sha256(
        manifest.get(INVALIDATION_HASH_FIELD), INVALIDATION_HASH_FIELD
    )
    if _manifest_digest(manifest) != stored_hash:
        raise RevisionInvalidationError("Invalidation manifest self-hash mismatch")
    attestation = manifest.get("stop_attestation")
    if not isinstance(attestation, dict) or attestation.get("confirmed_stopped") is not True:
        raise RevisionInvalidationError("Invalidation entry lacks stopped-shard attestation")
    shard = manifest.get("shard")
    if not isinstance(shard, dict) or not shard.get("absolute_path"):
        raise RevisionInvalidationError("Invalidation entry lacks shard path")
    root = _validate_shard_path(Path(str(shard["absolute_path"])))
    expected_snapshot = shard.get("snapshot")
    if not isinstance(expected_snapshot, dict):
        raise RevisionInvalidationError("Invalidation entry lacks shard snapshot")
    actual_snapshot = snapshot_shard(root)
    if actual_snapshot != expected_snapshot:
        raise ShardIntegrityError("Invalidated shard changed after registry entry creation")
    actual_identity = inspect_shard_identity(root, actual_snapshot)
    if actual_identity != manifest.get("superseded_identity"):
        raise ShardIntegrityError("Invalidated shard identity no longer matches registry entry")
    actual_execution_state, actual_incurred_compute = inspect_shard_execution(root)
    if actual_execution_state != manifest.get("execution_state"):
        raise ShardIntegrityError("Invalidated shard execution state no longer matches registry entry")
    if actual_incurred_compute != manifest.get("incurred_compute"):
        raise ShardIntegrityError("Invalidated shard incurred-compute charge no longer matches registry entry")
    final_snapshot = snapshot_shard(root)
    if final_snapshot != actual_snapshot:
        raise ShardIntegrityError("Invalidated shard changed during registry verification")
    target = manifest.get("superseding_target")
    if not isinstance(target, dict) or not str(target.get("namespace", "")).strip():
        raise RevisionInvalidationError("Invalidation entry lacks superseding target namespace")
    stages = target.get("stages")
    if (
        not isinstance(stages, list)
        or not stages
        or any(not str(stage).strip() for stage in stages)
        or len(stages) != len(set(map(str, stages)))
    ):
        raise RevisionInvalidationError("Invalidation entry lacks valid superseding stages")
    return {
        "status": "ok",
        "registry_entry_path": str(entry_path.resolve()),
        "invalidation_manifest_sha256": stored_hash,
        "shard_path": str(root),
        "shard_tree_sha256": actual_snapshot["shard_tree_sha256"],
        "run_identity_sha256": actual_identity["run_identity_sha256"],
        "planned_trial_ids_sha256": actual_identity["planned_trial_ids_sha256"],
        "config_manifest_sha256": actual_identity["config_manifest_sha256"],
        "source_manifest_sha256": actual_identity["source_manifest_sha256"],
        "scientific_status": str(manifest["scientific_status"]),
        "execution_state": actual_execution_state,
        "incurred_gpu_seconds": actual_incurred_compute["incurred_gpu_seconds"],
        "charge_policy": actual_incurred_compute["charge_policy"],
        "superseding_target_namespace": str(target["namespace"]),
        "superseding_stages": list(stages),
    }


__all__ = [
    "INVALIDATION_HASH_FIELD",
    "INVALIDATION_MANIFEST_TYPE",
    "INVALIDATION_SCHEMA_VERSION",
    "InvalidationEntryExistsError",
    "RevisionInvalidationError",
    "ShardIntegrityError",
    "StoppedShardConfirmationError",
    "build_invalidation_manifest",
    "create_invalidation_entry",
    "inspect_shard_execution",
    "inspect_shard_identity",
    "snapshot_shard",
    "verify_invalidation_entry",
]
