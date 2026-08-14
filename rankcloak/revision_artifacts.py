"""Integrity and resumability primitives for revision experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ARTIFACT_MANIFEST_SCHEMA_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "1.0"


class ImmutableArtifactError(RuntimeError):
    """Raised when code attempts to replace a frozen artifact with new bytes."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact or checkpoint does not match its frozen identity."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_ids_sha256(trial_ids: Iterable[str]) -> str:
    """Hash ordered planned IDs; order is part of the resume contract."""

    return canonical_json_sha256([str(trial_id) for trial_id in trial_ids])


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactIntegrityError("Refusing to write through symlink: {}".format(path))
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_immutable_bytes(path: Path, content: bytes) -> bool:
    """Write once; identical retries are no-ops and differing retries fail."""

    path = Path(path)
    content = bytes(content)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise ImmutableArtifactError(
                "Refusing immutable write through symlink: {}".format(path)
            )
        existing = path.read_bytes()
        if hmac_compare_bytes(existing, content):
            return False
        raise ImmutableArtifactError(
            "Immutable artifact already exists with different content: {}".format(path)
        )
    _atomic_write_bytes(path, content)
    return True


def hmac_compare_bytes(left: bytes, right: bytes) -> bool:
    """Constant-time comparison without importing project cryptographic state."""

    import hmac

    return hmac.compare_digest(left, right)


def write_immutable_json(path: Path, value: object) -> bool:
    content = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    return write_immutable_bytes(path, content)


def write_immutable_jsonl(path: Path, rows: Iterable[object]) -> bool:
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return write_immutable_bytes(path, content)


def load_json_object(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            "Cannot load JSON artifact {}: {}".format(path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError("JSON artifact is not an object: {}".format(path))
    return value


def _normalized_exclusions(exclude_paths: Iterable[str]) -> set:
    return {str(Path(path).as_posix()).lstrip("./") for path in exclude_paths}


def build_directory_manifest(
    root: Path,
    relative_paths: Optional[Iterable[str]] = None,
    exclude_paths: Iterable[str] = ("config_manifest.json",),
) -> Dict[str, object]:
    """Build a content-addressed manifest without embedding absolute paths."""

    root = Path(root)
    exclusions = _normalized_exclusions(exclude_paths)
    if relative_paths is None:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
        relative = sorted(path.relative_to(root).as_posix() for path in candidates)
    else:
        relative = sorted({str(Path(path).as_posix()).lstrip("./") for path in relative_paths})
    relative = [path for path in relative if path not in exclusions]
    files: List[Dict[str, object]] = []
    for relative_path in relative:
        path = root / relative_path
        if path.is_symlink():
            raise ArtifactIntegrityError(
                "Manifest input may not be a symlink: {}".format(relative_path)
            )
        if not path.is_file():
            raise ArtifactIntegrityError(
                "Manifest input is missing or not a file: {}".format(relative_path)
            )
        files.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    files_sha256 = canonical_json_sha256(files)
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "sha256_file_set",
        "file_count": len(files),
        "files_sha256": files_sha256,
        "files": files,
    }


def write_immutable_directory_manifest(
    root: Path,
    output_name: str = "config_manifest.json",
    relative_paths: Optional[Iterable[str]] = None,
    exclude_paths: Iterable[str] = ("config_manifest.json",),
) -> Dict[str, object]:
    manifest = build_directory_manifest(
        root,
        relative_paths=relative_paths,
        exclude_paths=exclude_paths,
    )
    write_immutable_json(Path(root) / output_name, manifest)
    return manifest


def verify_directory_manifest(
    root: Path,
    manifest: Mapping[str, object],
    require_no_extra_files: bool = False,
    ignored_extra_paths: Iterable[str] = ("config_manifest.json",),
) -> Dict[str, object]:
    """Verify listed files and optionally reject unlisted files."""

    root = Path(root)
    errors: List[str] = []
    listed_paths = []
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return {"status": "error", "errors": ["manifest files must be a list"]}
    for record in files:
        if not isinstance(record, dict):
            errors.append("manifest file record is not an object")
            continue
        relative_path = str(record.get("path"))
        listed_paths.append(relative_path)
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            errors.append("missing or invalid file: {}".format(relative_path))
            continue
        actual_size = path.stat().st_size
        actual_hash = file_sha256(path)
        if actual_size != record.get("size_bytes"):
            errors.append("size mismatch: {}".format(relative_path))
        if actual_hash != record.get("sha256"):
            errors.append("SHA-256 mismatch: {}".format(relative_path))
    expected_files_hash = canonical_json_sha256(files)
    if expected_files_hash != manifest.get("files_sha256"):
        errors.append("manifest file-list hash mismatch")
    if len(files) != manifest.get("file_count"):
        errors.append("manifest file_count mismatch")
    if require_no_extra_files:
        ignored = _normalized_exclusions(ignored_extra_paths)
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        extras = sorted(actual_paths - set(listed_paths) - ignored)
        if extras:
            errors.append("unlisted files: {}".format(", ".join(extras)))
    return {
        "status": "ok" if not errors else "error",
        "verified_file_count": len(files),
        "errors": errors,
    }


def build_run_identity_manifest(
    study_id: str,
    config_manifest_sha256: str,
    payload_manifest_sha256: str,
    planned_trial_ids: Sequence[str],
    model_artifacts: Sequence[Mapping[str, object]],
    command_line_args: Sequence[str],
) -> Dict[str, object]:
    """Create immutable scientific inputs; runtime timestamps belong elsewhere."""

    identity = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "revision_run_identity",
        "study_id": str(study_id),
        "config_manifest_sha256": str(config_manifest_sha256),
        "payload_manifest_sha256": str(payload_manifest_sha256),
        "planned_trial_count": len(planned_trial_ids),
        "planned_trial_ids_sha256": trial_ids_sha256(planned_trial_ids),
        "model_artifacts": [dict(model) for model in model_artifacts],
        "command_line_args": [str(value) for value in command_line_args],
    }
    identity["run_identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _timestamp(value: Optional[str] = None) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def initialize_checkpoint(
    path: Path,
    study_id: str,
    config_manifest_sha256: str,
    planned_trial_ids: Sequence[str],
    timestamp: Optional[str] = None,
) -> Dict[str, object]:
    """Create or validate a mutable checkpoint bound to immutable inputs."""

    path = Path(path)
    planned_hash = trial_ids_sha256(planned_trial_ids)
    if path.exists():
        checkpoint = load_checkpoint(path)
        _validate_checkpoint_identity(
            checkpoint,
            study_id=study_id,
            config_manifest_sha256=config_manifest_sha256,
            planned_trial_ids_sha256=planned_hash,
            planned_trial_count=len(planned_trial_ids),
        )
        return checkpoint
    now = _timestamp(timestamp)
    checkpoint: Dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "study_id": str(study_id),
        "config_manifest_sha256": str(config_manifest_sha256),
        "planned_trial_count": len(planned_trial_ids),
        "planned_trial_ids_sha256": planned_hash,
        "completed_trial_ids": [],
        "failed_trial_ids": [],
        "failure_details": {},
        "attempt_counts": {},
        "created_at": now,
        "updated_at": now,
    }
    save_checkpoint(path, checkpoint)
    return checkpoint


def load_checkpoint(path: Path) -> Dict[str, object]:
    checkpoint = load_json_object(path)
    required = {
        "schema_version",
        "study_id",
        "config_manifest_sha256",
        "planned_trial_count",
        "planned_trial_ids_sha256",
        "completed_trial_ids",
        "failed_trial_ids",
        "failure_details",
        "attempt_counts",
        "created_at",
        "updated_at",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ArtifactIntegrityError(
            "Checkpoint is missing fields: {}".format(", ".join(missing))
        )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported checkpoint schema version")
    return checkpoint


def save_checkpoint(path: Path, checkpoint: Mapping[str, object]) -> None:
    """Atomically replace the mutable progress file."""

    content = json.dumps(
        dict(checkpoint),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(Path(path), content)


def _validate_checkpoint_identity(
    checkpoint: Mapping[str, object],
    study_id: str,
    config_manifest_sha256: str,
    planned_trial_ids_sha256: str,
    planned_trial_count: int,
) -> None:
    expected = {
        "study_id": str(study_id),
        "config_manifest_sha256": str(config_manifest_sha256),
        "planned_trial_ids_sha256": str(planned_trial_ids_sha256),
        "planned_trial_count": int(planned_trial_count),
    }
    mismatches = [
        key for key, value in expected.items() if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ArtifactIntegrityError(
            "Checkpoint identity mismatch: {}".format(", ".join(mismatches))
        )


def record_checkpoint_result(
    path: Path,
    trial_id: str,
    status: str,
    failure_detail: Optional[Mapping[str, object]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, object]:
    """Record a completed or failed attempt without losing retry history."""

    if status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'")
    checkpoint = load_checkpoint(path)
    trial_id = str(trial_id)
    completed = set(map(str, checkpoint["completed_trial_ids"]))
    failed = set(map(str, checkpoint["failed_trial_ids"]))
    attempts = {
        str(key): int(value)
        for key, value in dict(checkpoint["attempt_counts"]).items()
    }
    attempts[trial_id] = attempts.get(trial_id, 0) + 1
    failure_details = dict(checkpoint["failure_details"])
    if status == "completed":
        completed.add(trial_id)
        failed.discard(trial_id)
        failure_details.pop(trial_id, None)
    elif trial_id not in completed:
        failed.add(trial_id)
        failure_details[trial_id] = dict(failure_detail or {})
    checkpoint["completed_trial_ids"] = sorted(completed)
    checkpoint["failed_trial_ids"] = sorted(failed)
    checkpoint["failure_details"] = failure_details
    checkpoint["attempt_counts"] = attempts
    checkpoint["updated_at"] = _timestamp(timestamp)
    save_checkpoint(path, checkpoint)
    return checkpoint


def pending_trial_ids(
    planned_trial_ids: Sequence[str],
    checkpoint: Mapping[str, object],
) -> List[str]:
    """Return planned IDs not completed, preserving frozen plan order."""

    planned = [str(value) for value in planned_trial_ids]
    if checkpoint.get("planned_trial_ids_sha256") != trial_ids_sha256(planned):
        raise ArtifactIntegrityError("Checkpoint does not match planned trial IDs")
    if checkpoint.get("planned_trial_count") != len(planned):
        raise ArtifactIntegrityError("Checkpoint does not match planned trial count")
    completed = set(map(str, checkpoint.get("completed_trial_ids", [])))
    unknown = completed - set(planned)
    if unknown:
        raise ArtifactIntegrityError(
            "Checkpoint contains unknown completed trial IDs: {}".format(
                ", ".join(sorted(unknown))
            )
        )
    return [trial_id for trial_id in planned if trial_id not in completed]


def checkpoint_summary(checkpoint: Mapping[str, object]) -> Dict[str, int]:
    planned = int(checkpoint["planned_trial_count"])
    completed = len(set(map(str, checkpoint.get("completed_trial_ids", []))))
    failed = len(set(map(str, checkpoint.get("failed_trial_ids", []))))
    return {
        "planned": planned,
        "completed": completed,
        "failed_current": failed,
        "remaining": planned - completed,
    }
