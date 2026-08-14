"""Durable, identity-bound execution for the frozen revision detector matrix.

The scientific implementation remains in :mod:`rankcloak.revision_detection`.
This module adds an execution boundary around each ordered split/detector fit:
children are written atomically, a manifest is committed last, and a fit is
reused only after its complete lineage, task identity, and child hashes pass
validation.  Final publication-shaped products are deliberately left to the
CLI and may be assembled only after every task validates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import re
import signal
import sys
import tempfile
import threading
import time
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rankcloak.revision_detection import (
    DetectorSplit,
    DetectorSuiteResult,
    PreparedDetectorSuite,
    RevisionDetectionError,
    assemble_prepared_detector_result,
    detector_fit_seed,
    run_prepared_detector_fit,
)


CHECKPOINT_SCHEMA = "rankcloak-revision-detector-fit-checkpoint-v1"
ROW_ARTIFACT_SCHEMA = "rankcloak-revision-detector-fit-rows-v1"
STATUS_SCHEMA = "rankcloak-revision-detector-status-v1"
PLAN_SCHEMA = "rankcloak-revision-detector-execution-plan-v1"
PERMIT_SCHEMA = "rankcloak-revision-detector-fit-permit-v1"
PERMIT_RECEIPT_SCHEMA = "rankcloak-revision-detector-fit-permit-receipt-v1"
EQUIVALENCE_ARTIFACT_SCHEMA = (
    "rankcloak-revision-detector-equivalence-fit-artifact-v1"
)
EQUIVALENCE_REPORT_SCHEMA = "rankcloak-revision-detector-device-equivalence-v1"
FINALIZATION_CANDIDATE_SCHEMA = (
    "rankcloak-revision-detector-finalization-candidate-v1"
)
TERMINAL_RECEIPT_SCHEMA = "rankcloak-revision-detector-terminal-receipt-v1"
GPU_LEDGER_SCHEMA = "rankcloak-revision-detector-gpu-accounting-ledger-v1"
GPU_LEDGER_INCORPORATION_SCHEMA = (
    "rankcloak-revision-detector-gpu-ledger-incorporation-v1"
)
GPU_ACCOUNTING_POLICY = "detector_process_wall_span_v1"
GPU_ACCOUNTING_COLLECTION_POLICY = (
    "nonoverlapping_detector_process_wall_intervals_v1"
)
EXPECTED_GPU_LEDGER_SOURCES = {
    "production_benchmark_task_0": "detector_production_benchmark",
    "production_benchmark_task_1": "detector_production_benchmark",
    "equivalence_cuda_task_0": "detector_device_equivalence_cuda",
    "equivalence_cuda_task_1": "detector_device_equivalence_cuda",
    "equivalence_cuda_repeat_task_0": (
        "detector_device_equivalence_cuda_repeat"
    ),
    "equivalence_cuda_repeat_task_1": (
        "detector_device_equivalence_cuda_repeat"
    ),
}


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value using the project canonical encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RevisionDetectionError("Detector timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()


def _file_identity(path: Path) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RevisionDetectionError("Identity source is missing or unsafe: {}".format(path))
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def source_identity(paths: Sequence[Path]) -> dict:
    """Return a closed, ordered identity for detector implementation files."""

    records = [_file_identity(Path(path)) for path in paths]
    return {
        "files": records,
        "files_sha256": canonical_json_sha256(records),
    }


def _ensure_real_directory(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists():
        if absolute.is_symlink() or not absolute.is_dir():
            raise RevisionDetectionError(
                "Detector execution directory is not a real directory: {}".format(absolute)
            )
    else:
        absolute.mkdir(parents=True, exist_ok=False)
    return absolute


@contextmanager
def detector_execution_lease(checkpoint_dir: Path):
    """Hold a nonblocking exclusive lease for one checkpoint execution root."""

    directory = _ensure_real_directory(Path(checkpoint_dir))
    lease_path = directory / ".execution.lock"
    if lease_path.is_symlink() or (lease_path.exists() and not lease_path.is_file()):
        raise RevisionDetectionError(
            "Detector execution lease path is unsafe: {}".format(lease_path)
        )
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(lease_path), flags, 0o600)
    except OSError as exc:
        raise RevisionDetectionError(
            "Cannot open detector execution lease: {}".format(exc)
        ) from exc
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RevisionDetectionError(
                "Another detector invocation holds the checkpoint execution lease."
            ) from exc
        identity = {
            "schema_version": "rankcloak-revision-detector-execution-lease-v1",
            "pid": int(os.getpid()),
            "process_start_ticks": _process_start_ticks(),
            "acquired_at_utc": _utc_text(_utc_now()),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(identity, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    parent = _ensure_real_directory(Path(path).parent)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise RevisionDetectionError(
            "Refusing to replace unsafe detector artifact: {}".format(path)
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-{}-".format(path.name), dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(Path(path), encoded)


def write_detector_finalization_candidate(
    path: Path,
    *,
    kind: str,
    run_identity_sha256: str,
    payload: Mapping[str, object],
    output_files: Mapping[str, Mapping[str, object]],
    requested_output_path: Path,
) -> dict:
    """Persist a signed runner candidate; a supervisor owns CUDA publication."""

    if kind not in {
        "detector_run_manifest",
        "equivalence_artifact",
        "benchmark_artifact",
    }:
        raise RevisionDetectionError("Detector finalization candidate kind differs.")
    files = {str(name): dict(identity) for name, identity in output_files.items()}
    for name, identity in files.items():
        child = Path(str(identity.get("path", "")))
        if (
            child.is_symlink()
            or not child.is_file()
            or file_sha256(child) != identity.get("sha256")
            or int(child.stat().st_size) != int(identity.get("size_bytes", -1))
        ):
            raise RevisionDetectionError(
                "Detector finalization candidate child differs: {}".format(name)
            )
    value = {
        "schema_version": FINALIZATION_CANDIDATE_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "kind": kind,
        "run_identity_sha256": run_identity_sha256,
        "requested_output_path": str(Path(requested_output_path).resolve()),
        "payload": dict(payload),
        "payload_sha256": canonical_json_sha256(payload),
        "output_files": files,
        "output_files_sha256": canonical_json_sha256(files),
        "finalization_policy": (
            "supervisor_confirms_exact_pid_absence_closes_accounting_then_publishes_v1"
        ),
    }
    candidate = _signed_document(value, "candidate_sha256")
    atomic_write_json(Path(path), candidate)
    return candidate


def detector_finalization_paths(
    checkpoint_dir: Path,
    *,
    kind: str,
    requested_output_path: Path,
    task_index: Optional[int] = None,
    role: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Return immutable collision-free candidate/receipt paths for one output."""

    if kind not in {
        "detector_run_manifest",
        "equivalence_artifact",
        "benchmark_artifact",
    }:
        raise RevisionDetectionError("Detector finalization kind is invalid.")
    if task_index is not None and int(task_index) not in {0, 1}:
        raise RevisionDetectionError("Detector finalization task index is invalid.")
    role_text = "suite" if role is None else str(role)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", role_text):
        raise RevisionDetectionError("Detector finalization role is invalid.")
    output_hash = canonical_json_sha256(
        {"requested_output_path": str(Path(requested_output_path).resolve())}
    )[:16]
    task_text = "suite" if task_index is None else "task{}".format(int(task_index))
    stem = "{}-{}-{}-{}".format(kind, task_text, role_text, output_hash)
    root = Path(checkpoint_dir) / "finalization_candidates"
    return root / (stem + ".json"), root / (stem + ".terminal_receipt.json")


def read_detector_finalization_candidate(path: Path) -> dict:
    value = _read_json(Path(path), "detector finalization candidate")
    if value.get("schema_version") != FINALIZATION_CANDIDATE_SCHEMA:
        raise RevisionDetectionError("Detector finalization candidate schema differs.")
    _verify_signed_document(
        value, "candidate_sha256", "Detector finalization candidate"
    )
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "kind",
        "run_identity_sha256",
        "requested_output_path",
        "payload",
        "payload_sha256",
        "output_files",
        "output_files_sha256",
        "finalization_policy",
        "candidate_sha256",
    }
    if (
        set(value) != expected_keys
        or
        value.get("kind")
        not in {
            "detector_run_manifest",
            "equivalence_artifact",
            "benchmark_artifact",
        }
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("run_identity_sha256", ""))
        )
        or not isinstance(value.get("payload"), dict)
        or value.get("payload_sha256")
        != canonical_json_sha256(value["payload"])
        or not isinstance(value.get("output_files"), dict)
        or value.get("output_files_sha256")
        != canonical_json_sha256(value["output_files"])
        or value.get("finalization_policy")
        != "supervisor_confirms_exact_pid_absence_closes_accounting_then_publishes_v1"
    ):
        raise RevisionDetectionError("Detector finalization candidate is malformed.")
    for identity in value["output_files"].values():
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RevisionDetectionError(
                "Detector finalization candidate child identity differs."
            )
        child = Path(str(identity.get("path", "")))
        if (
            child.is_symlink()
            or not child.is_file()
            or file_sha256(child) != identity.get("sha256")
            or int(child.stat().st_size) != int(identity.get("size_bytes", -1))
        ):
            raise RevisionDetectionError(
                "Detector finalization candidate child bytes differ."
            )
    return value


def write_detector_terminal_receipt(
    path: Path,
    *,
    candidate_path: Path,
    published_output_path: Path,
    closed_status: Mapping[str, object],
    gpu_accounting: Optional[Mapping[str, object]] = None,
) -> dict:
    """Bind a supervisor-published output to exact closed terminal accounting."""

    candidate = read_detector_finalization_candidate(candidate_path)
    published = Path(published_output_path)
    if published.is_symlink() or not published.is_file():
        raise RevisionDetectionError("Supervisor-published detector output is unsafe.")
    _verify_signed_document(closed_status, "status_sha256", "Detector status")
    accounting = (
        closed_status.get("gpu_accounting")
        if gpu_accounting is None
        else dict(gpu_accounting)
    )
    if (
        closed_status.get("run_identity_sha256")
        != candidate["run_identity_sha256"]
        or not isinstance(accounting, dict)
    ):
        raise RevisionDetectionError(
            "Detector terminal status/accounting identity differs."
        )
    value = {
        "schema_version": TERMINAL_RECEIPT_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "candidate": {
            "path": str(Path(candidate_path).resolve()),
            "sha256": file_sha256(Path(candidate_path)),
            "size_bytes": int(Path(candidate_path).stat().st_size),
            "candidate_sha256": candidate["candidate_sha256"],
        },
        "published_output": {
            "path": str(published.resolve()),
            "sha256": file_sha256(published),
            "size_bytes": int(published.stat().st_size),
        },
        "closed_status": dict(closed_status),
        "closed_status_sha256": closed_status["status_sha256"],
        "run_identity_sha256": candidate["run_identity_sha256"],
        "gpu_accounting": accounting,
        "gpu_accounting_sha256": canonical_json_sha256(
            accounting
        ),
        "finalization_policy": (
            "supervisor_confirmed_exact_pid_absence_terminal_receipt_v1"
        ),
    }
    receipt = _signed_document(value, "terminal_receipt_sha256")
    atomic_write_json(Path(path), receipt)
    return receipt


def _validate_terminal_receipt(path: Path) -> dict:
    receipt_path = Path(path)
    receipt = _read_json(receipt_path, "detector terminal receipt")
    if receipt.get("schema_version") != TERMINAL_RECEIPT_SCHEMA:
        raise RevisionDetectionError("Detector terminal receipt schema differs.")
    _verify_signed_document(
        receipt, "terminal_receipt_sha256", "Detector terminal receipt"
    )
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "candidate",
        "published_output",
        "closed_status",
        "closed_status_sha256",
        "run_identity_sha256",
        "gpu_accounting",
        "gpu_accounting_sha256",
        "finalization_policy",
        "terminal_receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise RevisionDetectionError("Detector terminal receipt shape differs.")
    candidate_identity = receipt.get("candidate")
    output_identity = receipt.get("published_output")
    closed_status = receipt.get("closed_status")
    accounting = receipt.get("gpu_accounting")
    if (
        not isinstance(candidate_identity, dict)
        or not isinstance(output_identity, dict)
        or not isinstance(closed_status, dict)
        or not isinstance(accounting, dict)
        or receipt.get("gpu_accounting_sha256")
        != canonical_json_sha256(accounting)
        or receipt.get("finalization_policy")
        != "supervisor_confirmed_exact_pid_absence_terminal_receipt_v1"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("closed_status_sha256", ""))
        )
    ):
        raise RevisionDetectionError("Detector terminal receipt is malformed.")
    _verify_signed_document(closed_status, "status_sha256", "Detector status")
    if (
        closed_status.get("schema_version") != STATUS_SCHEMA
        or closed_status.get("state") != "supervisor_observed_process_exit"
        or
        closed_status.get("status_sha256")
        != receipt.get("closed_status_sha256")
        or closed_status.get("run_identity_sha256")
        != receipt.get("run_identity_sha256")
        or not isinstance(closed_status.get("gpu_accounting"), dict)
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt closed status identity differs."
        )
    candidate_path = Path(str(candidate_identity.get("path", "")))
    if (
        candidate_path.is_symlink()
        or not candidate_path.is_file()
        or file_sha256(candidate_path) != candidate_identity.get("sha256")
        or int(candidate_path.stat().st_size)
        != int(candidate_identity.get("size_bytes", -1))
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt candidate bytes differ."
        )
    candidate = read_detector_finalization_candidate(candidate_path)
    if (
        candidate.get("candidate_sha256")
        != candidate_identity.get("candidate_sha256")
        or candidate.get("run_identity_sha256")
        != receipt.get("run_identity_sha256")
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt candidate identity differs."
        )
    output_path = Path(str(output_identity.get("path", "")))
    if (
        output_path.is_symlink()
        or not output_path.is_file()
        or file_sha256(output_path) != output_identity.get("sha256")
        or int(output_path.stat().st_size)
        != int(output_identity.get("size_bytes", -1))
        or output_path.resolve()
        != Path(candidate["requested_output_path"]).resolve()
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt published bytes differ."
        )
    published_value = _read_json(
        output_path, "detector terminal receipt published output"
    )
    signature_field = {
        "equivalence_artifact": "artifact_sha256",
        "benchmark_artifact": "benchmark_sha256",
        "detector_run_manifest": "manifest_sha256",
    }[candidate["kind"]]
    _verify_signed_document(
        published_value, signature_field, "Supervisor-published detector output"
    )
    if (
        published_value.get("gpu_accounting") != accounting
        or published_value.get("terminal_accounting_status_sha256")
        != receipt.get("closed_status_sha256")
        or published_value.get("finalization_candidate", {}).get(
            "candidate_sha256"
        )
        != candidate["candidate_sha256"]
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt output/accounting triad differs."
        )
    if candidate["kind"] == "equivalence_artifact" and published_value.get(
        "provenance", {}
    ).get("gpu_accounting") != accounting:
        raise RevisionDetectionError(
            "CUDA equivalence artifact provenance accounting differs."
        )
    if (
        set(accounting)
        != {
            "device",
            "gpu_uuid",
            "intervals",
            "cumulative_elapsed_seconds",
            "derivation_policy",
        }
        or
        accounting.get("device") != "cuda:0"
        or not str(accounting.get("gpu_uuid", "")).startswith("GPU-")
        or accounting.get("derivation_policy")
        != GPU_ACCOUNTING_COLLECTION_POLICY
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt GPU accounting is malformed."
        )
    previous_end: Optional[datetime] = None
    elapsed_sum = 0.0
    for interval in accounting["intervals"]:
        if not isinstance(interval, dict) or set(interval) != {
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
            raise RevisionDetectionError(
                "Detector terminal receipt interval shape differs."
            )
        try:
            start = datetime.fromisoformat(str(interval["started_at_utc"]))
            end = datetime.fromisoformat(str(interval["completed_at_utc"]))
            last_observed = datetime.fromisoformat(
                str(interval["last_observed_at_utc"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RevisionDetectionError(
                "Detector terminal receipt interval timestamp is invalid."
            ) from exc
        elapsed = float((end - start).total_seconds())
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or end <= start
            or last_observed != end
            or int(interval.get("pid", -1)) <= 0
            or int(interval.get("process_start_ticks", -1)) <= 0
            or (previous_end is not None and start < previous_end)
            or interval.get("device") != "cuda:0"
            or interval.get("gpu_uuid") != accounting["gpu_uuid"]
            or interval.get("derivation_policy") != GPU_ACCOUNTING_POLICY
            or abs(float(interval.get("elapsed_seconds", -1.0)) - elapsed) > 1e-6
        ):
            raise RevisionDetectionError(
                "Detector terminal receipt intervals are malformed/overlapping."
            )
        elapsed_sum += elapsed
        previous_end = end
    if abs(
        float(accounting.get("cumulative_elapsed_seconds", -1.0)) - elapsed_sum
    ) > 1e-6:
        raise RevisionDetectionError(
            "Detector terminal receipt cumulative GPU time differs."
        )
    accounted_intervals = {
        canonical_json_sha256(interval) for interval in accounting["intervals"]
    }
    closed_intervals = closed_status["gpu_accounting"].get("intervals")
    if not isinstance(closed_intervals, list) or not {
        canonical_json_sha256(interval) for interval in closed_intervals
    }.issubset(accounted_intervals):
        raise RevisionDetectionError(
            "Detector terminal receipt omits closed-status GPU intervals."
        )
    ledger_identity = published_value.get("pre_final_gpu_accounting_ledger")
    if ledger_identity is not None:
        if not isinstance(ledger_identity, dict) or set(ledger_identity) != {
            "path",
            "sha256",
            "size_bytes",
            "ledger_sha256",
            "sources_sha256",
            "intervals_sha256",
            "cumulative_elapsed_seconds",
        }:
            raise RevisionDetectionError(
                "Detector terminal receipt ledger identity differs."
            )
        ledger_path = Path(str(ledger_identity["path"]))
        if (
            ledger_path.is_symlink()
            or not ledger_path.is_file()
            or file_sha256(ledger_path) != ledger_identity["sha256"]
            or int(ledger_path.stat().st_size)
            != int(ledger_identity["size_bytes"])
        ):
            raise RevisionDetectionError(
                "Detector terminal receipt ledger bytes differ."
            )
        ledger = read_detector_gpu_accounting_ledger(ledger_path)
        if (
            ledger["ledger_sha256"] != ledger_identity["ledger_sha256"]
            or ledger["sources_sha256"] != ledger_identity["sources_sha256"]
            or ledger["intervals_sha256"]
            != ledger_identity["intervals_sha256"]
            or float(ledger["cumulative_elapsed_seconds"])
            != float(ledger_identity["cumulative_elapsed_seconds"])
        ):
            raise RevisionDetectionError(
                "Detector terminal receipt ledger content differs."
            )
    expected_payload = dict(candidate["payload"])
    expected_payload["gpu_accounting"] = accounting
    expected_payload["pre_final_gpu_accounting_ledger"] = ledger_identity
    expected_payload["terminal_accounting_status_sha256"] = receipt[
        "closed_status_sha256"
    ]
    expected_payload["finalization_candidate"] = dict(receipt["candidate"])
    expected_payload["execution_completed_at_utc"] = accounting["intervals"][-1][
        "completed_at_utc"
    ]
    if candidate["kind"] == "equivalence_artifact":
        provenance = dict(expected_payload.get("provenance", {}))
        provenance["gpu_accounting"] = accounting
        provenance["terminal_accounting_status_sha256"] = receipt[
            "closed_status_sha256"
        ]
        expected_payload["provenance"] = provenance
        expected_payload["provenance_sha256"] = canonical_json_sha256(provenance)
    expected_published = _signed_document(expected_payload, signature_field)
    if published_value != expected_published:
        raise RevisionDetectionError(
            "Supervisor-published detector output differs from its candidate."
        )
    return receipt


def update_detector_gpu_accounting_ledger(
    ledger_path: Path,
    *,
    source_id: str,
    component: str,
    terminal_receipt_path: Path,
) -> dict:
    """Atomically add one terminal receipt's intervals to the union ledger."""

    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", source_id):
        raise RevisionDetectionError("Detector GPU ledger source id is invalid.")
    receipt_path = Path(terminal_receipt_path)
    receipt = _read_json(receipt_path, "detector terminal receipt")
    if receipt.get("schema_version") != TERMINAL_RECEIPT_SCHEMA:
        raise RevisionDetectionError("Detector terminal receipt schema differs.")
    _verify_signed_document(
        receipt, "terminal_receipt_sha256", "Detector terminal receipt"
    )
    accounting = receipt.get("gpu_accounting")
    if (
        not isinstance(accounting, dict)
        or accounting.get("device") != "cuda:0"
        or not str(accounting.get("gpu_uuid", "")).startswith("GPU-")
        or accounting.get("derivation_policy")
        != GPU_ACCOUNTING_COLLECTION_POLICY
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
        or receipt.get("gpu_accounting_sha256")
        != canonical_json_sha256(accounting)
    ):
        raise RevisionDetectionError(
            "Detector terminal receipt GPU accounting is malformed."
        )
    source = {
        "source_id": source_id,
        "component": component,
        "terminal_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": file_sha256(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
            "terminal_receipt_sha256": receipt["terminal_receipt_sha256"],
        },
        "gpu_accounting_sha256": receipt["gpu_accounting_sha256"],
    }
    path = Path(ledger_path)
    if path.exists():
        existing = read_detector_gpu_accounting_ledger(path)
        sources = [dict(value) for value in existing.get("sources", [])]
        previous = [value for value in sources if value.get("source_id") == source_id]
        if previous:
            if previous != [source]:
                raise RevisionDetectionError(
                    "Detector GPU ledger source identity changed."
                )
            return existing
        if existing.get("gpu_uuid") != accounting.get("gpu_uuid"):
            raise RevisionDetectionError("Detector GPU ledger UUID differs.")
    else:
        sources = []
    sources.append(source)
    interval_by_identity: Dict[Tuple[int, int, str, str], dict] = {}
    for source_value in sources:
        source_receipt_path = Path(
            str(source_value["terminal_receipt"]["path"])
        )
        source_receipt = _validate_terminal_receipt(source_receipt_path)
        if (
            file_sha256(source_receipt_path)
            != source_value["terminal_receipt"]["sha256"]
            or source_receipt["terminal_receipt_sha256"]
            != source_value["terminal_receipt"]["terminal_receipt_sha256"]
        ):
            raise RevisionDetectionError(
                "Detector GPU ledger terminal receipt bytes differ."
            )
        for interval in source_receipt["gpu_accounting"]["intervals"]:
            key = (
                int(interval["pid"]),
                int(interval["process_start_ticks"]),
                str(interval["started_at_utc"]),
                str(interval.get("completed_at_utc")),
            )
            prior = interval_by_identity.get(key)
            if prior is not None and prior != interval:
                raise RevisionDetectionError(
                    "Detector GPU ledger interval identity conflicts."
                )
            interval_by_identity[key] = dict(interval)
    intervals = sorted(
        interval_by_identity.values(), key=lambda value: value["started_at_utc"]
    )
    previous_end: Optional[datetime] = None
    for interval in intervals:
        start = datetime.fromisoformat(str(interval["started_at_utc"]))
        end = datetime.fromisoformat(str(interval["completed_at_utc"]))
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or end < start
            or (previous_end is not None and start < previous_end)
            or abs(
                float(interval.get("elapsed_seconds", -1.0))
                - float((end - start).total_seconds())
            )
            > 1e-6
        ):
            raise RevisionDetectionError(
                "Detector GPU ledger intervals overlap or are malformed."
            )
        previous_end = end
    value = {
        "schema_version": GPU_LEDGER_SCHEMA,
        "updated_at_utc": _utc_text(_utc_now()),
        "device": "cuda:0",
        "gpu_uuid": accounting["gpu_uuid"],
        "sources": sources,
        "sources_sha256": canonical_json_sha256(sources),
        "intervals": intervals,
        "intervals_sha256": canonical_json_sha256(intervals),
        "cumulative_elapsed_seconds": float(
            sum(float(interval["elapsed_seconds"]) for interval in intervals)
        ),
        "derivation_policy": "terminal_receipt_interval_union_deduplicated_v1",
    }
    signed = _signed_document(value, "ledger_sha256")
    atomic_write_json(path, signed)
    return signed


def read_detector_gpu_accounting_ledger(path: Path) -> dict:
    value = _read_json(Path(path), "detector GPU accounting ledger")
    if value.get("schema_version") != GPU_LEDGER_SCHEMA:
        raise RevisionDetectionError("Detector GPU ledger schema differs.")
    _verify_signed_document(value, "ledger_sha256", "Detector GPU ledger")
    expected_keys = {
        "schema_version",
        "updated_at_utc",
        "device",
        "gpu_uuid",
        "sources",
        "sources_sha256",
        "intervals",
        "intervals_sha256",
        "cumulative_elapsed_seconds",
        "derivation_policy",
        "ledger_sha256",
    }
    if (
        set(value) != expected_keys
        or
        value.get("device") != "cuda:0"
        or not str(value.get("gpu_uuid", "")).startswith("GPU-")
        or not isinstance(value.get("sources"), list)
        or value.get("sources_sha256") != canonical_json_sha256(value["sources"])
        or not isinstance(value.get("intervals"), list)
        or value.get("intervals_sha256")
        != canonical_json_sha256(value["intervals"])
        or value.get("derivation_policy")
        != "terminal_receipt_interval_union_deduplicated_v1"
    ):
        raise RevisionDetectionError("Detector GPU ledger is malformed.")
    source_ids = []
    interval_by_identity: Dict[Tuple[int, int, str, str], dict] = {}
    for source in value["sources"]:
        if (
            not isinstance(source, dict)
            or set(source)
            != {
                "source_id",
                "component",
                "terminal_receipt",
                "gpu_accounting_sha256",
            }
            or not re.fullmatch(
                r"[A-Za-z0-9_.:-]+", str(source.get("source_id", ""))
            )
            or not isinstance(source.get("component"), str)
        ):
            raise RevisionDetectionError("Detector GPU ledger source is malformed.")
        source_ids.append(source["source_id"])
        identity = source["terminal_receipt"]
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "sha256",
            "size_bytes",
            "terminal_receipt_sha256",
        }:
            raise RevisionDetectionError(
                "Detector GPU ledger receipt identity is malformed."
            )
        receipt_path = Path(str(identity["path"]))
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or file_sha256(receipt_path) != identity["sha256"]
            or int(receipt_path.stat().st_size) != int(identity["size_bytes"])
        ):
            raise RevisionDetectionError(
                "Detector GPU ledger receipt bytes differ."
            )
        receipt = _validate_terminal_receipt(receipt_path)
        if (
            receipt["terminal_receipt_sha256"]
            != identity["terminal_receipt_sha256"]
            or receipt["gpu_accounting_sha256"]
            != source["gpu_accounting_sha256"]
            or receipt["gpu_accounting"]["gpu_uuid"] != value["gpu_uuid"]
        ):
            raise RevisionDetectionError(
                "Detector GPU ledger receipt content differs."
            )
        for interval in receipt["gpu_accounting"]["intervals"]:
            key = (
                int(interval["pid"]),
                int(interval["process_start_ticks"]),
                str(interval["started_at_utc"]),
                str(interval["completed_at_utc"]),
            )
            prior = interval_by_identity.get(key)
            if prior is not None and prior != interval:
                raise RevisionDetectionError(
                    "Detector GPU ledger interval identity conflicts."
                )
            interval_by_identity[key] = dict(interval)
    if len(source_ids) != len(set(source_ids)):
        raise RevisionDetectionError("Detector GPU ledger source ids repeat.")
    derived_intervals = sorted(
        interval_by_identity.values(), key=lambda item: item["started_at_utc"]
    )
    if derived_intervals != value["intervals"]:
        raise RevisionDetectionError(
            "Detector GPU ledger interval union differs from its receipts."
        )
    previous_end: Optional[datetime] = None
    cumulative = 0.0
    for interval in derived_intervals:
        start = datetime.fromisoformat(str(interval["started_at_utc"]))
        end = datetime.fromisoformat(str(interval["completed_at_utc"]))
        if previous_end is not None and start < previous_end:
            raise RevisionDetectionError("Detector GPU ledger intervals overlap.")
        cumulative += float(interval["elapsed_seconds"])
        previous_end = end
    if abs(float(value.get("cumulative_elapsed_seconds", -1.0)) - cumulative) > 1e-6:
        raise RevisionDetectionError("Detector GPU ledger cumulative time differs.")
    return value


def detector_gpu_ledger_incorporation_path(ledger_path: Path) -> Path:
    return Path(str(Path(ledger_path)) + ".incorporation.json")


def write_detector_gpu_ledger_incorporation_marker(
    marker_path: Path,
    *,
    ledger_path: Path,
    final_terminal_receipt_path: Path,
) -> dict:
    """Prove every pre-final ledger interval is in the canonical final receipt."""

    ledger = read_detector_gpu_accounting_ledger(ledger_path)
    receipt = _validate_terminal_receipt(final_terminal_receipt_path)
    candidate_path = Path(str(receipt["candidate"]["path"]))
    candidate = read_detector_finalization_candidate(candidate_path)
    if candidate.get("kind") != "detector_run_manifest":
        raise RevisionDetectionError(
            "GPU ledger incorporation requires the final detector run manifest."
        )
    observed_sources = {
        str(source["source_id"]): str(source["component"])
        for source in ledger["sources"]
    }
    if observed_sources != EXPECTED_GPU_LEDGER_SOURCES:
        raise RevisionDetectionError(
            "Canonical detector completion requires the exact six pre-final GPU "
            "ledger sources."
        )
    final_accounting = receipt["gpu_accounting"]
    if final_accounting.get("gpu_uuid") != ledger.get("gpu_uuid"):
        raise RevisionDetectionError(
            "GPU ledger/final receipt UUID differs during incorporation."
        )
    final_intervals = {
        canonical_json_sha256(interval): interval
        for interval in final_accounting["intervals"]
    }
    missing = [
        interval
        for interval in ledger["intervals"]
        if canonical_json_sha256(interval) not in final_intervals
    ]
    if missing:
        raise RevisionDetectionError(
            "Canonical detector accounting omits pre-final ledger intervals."
        )
    ledger_identity = {
        "path": str(Path(ledger_path).resolve()),
        "sha256": file_sha256(Path(ledger_path)),
        "size_bytes": int(Path(ledger_path).stat().st_size),
        "ledger_sha256": ledger["ledger_sha256"],
        "intervals_sha256": ledger["intervals_sha256"],
        "cumulative_elapsed_seconds": ledger["cumulative_elapsed_seconds"],
    }
    receipt_identity = {
        "path": str(Path(final_terminal_receipt_path).resolve()),
        "sha256": file_sha256(Path(final_terminal_receipt_path)),
        "size_bytes": int(Path(final_terminal_receipt_path).stat().st_size),
        "terminal_receipt_sha256": receipt["terminal_receipt_sha256"],
    }
    value = {
        "schema_version": GPU_LEDGER_INCORPORATION_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "ledger": ledger_identity,
        "final_terminal_receipt": receipt_identity,
        "final_published_manifest": dict(receipt["published_output"]),
        "incorporated": True,
        "incorporated_ledger_interval_count": len(ledger["intervals"]),
        "incorporated_ledger_intervals_sha256": ledger["intervals_sha256"],
        "final_gpu_accounting_sha256": receipt["gpu_accounting_sha256"],
        "incorporation_policy": (
            "every_ledger_interval_exactly_present_in_final_accounting_v1"
        ),
    }
    marker = _signed_document(value, "incorporation_sha256")
    path = Path(marker_path)
    if path.exists():
        existing = _read_json(path, "detector GPU ledger incorporation marker")
        if existing != marker:
            # created_at is intentionally immutable; validate an existing marker
            # against the same current ledger/receipt instead of overwriting it.
            existing_unsigned = dict(existing)
            existing_unsigned.pop("incorporation_sha256", None)
            expected_without_time = dict(value)
            expected_without_time["created_at_utc"] = existing.get(
                "created_at_utc"
            )
            if existing_unsigned != expected_without_time:
                raise RevisionDetectionError(
                    "Existing GPU ledger incorporation marker differs."
                )
            _verify_signed_document(
                existing,
                "incorporation_sha256",
                "Detector GPU ledger incorporation marker",
            )
            return existing
    else:
        atomic_write_json(path, marker)
    return marker


def read_detector_gpu_ledger_incorporation_marker(path: Path) -> dict:
    marker = _read_json(Path(path), "detector GPU ledger incorporation marker")
    if marker.get("schema_version") != GPU_LEDGER_INCORPORATION_SCHEMA:
        raise RevisionDetectionError(
            "Detector GPU ledger incorporation marker schema differs."
        )
    _verify_signed_document(
        marker,
        "incorporation_sha256",
        "Detector GPU ledger incorporation marker",
    )
    regenerated = write_detector_gpu_ledger_incorporation_marker(
        Path(path),
        ledger_path=Path(str(marker.get("ledger", {}).get("path", ""))),
        final_terminal_receipt_path=Path(
            str(marker.get("final_terminal_receipt", {}).get("path", ""))
        ),
    )
    if regenerated != marker:
        raise RevisionDetectionError(
            "Detector GPU ledger incorporation marker bytes differ."
        )
    return marker


def finalize_detector_candidate_from_closed_status(
    candidate_path: Path,
    *,
    closed_status_file: Path,
    terminal_receipt_path: Path,
    gpu_accounting_ledger_path: Optional[Path] = None,
) -> dict:
    """Idempotently publish a CUDA candidate after exact PID absence is proven."""

    candidate = read_detector_finalization_candidate(candidate_path)
    if (
        candidate.get("kind") == "detector_run_manifest"
        and gpu_accounting_ledger_path is None
    ):
        raise RevisionDetectionError(
            "Canonical detector finalization requires the pre-final GPU ledger."
        )
    status = verify_status_file(closed_status_file)
    receipt_path = Path(terminal_receipt_path)
    if status.get("state") == "complete":
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise RevisionDetectionError(
                "Complete detector status lacks its terminal receipt."
            )
        receipt = _validate_terminal_receipt(receipt_path)
        if (
            receipt.get("candidate", {}).get("candidate_sha256")
            != candidate["candidate_sha256"]
            or status.get("terminal_receipt", {}).get("sha256")
            != file_sha256(receipt_path)
        ):
            raise RevisionDetectionError(
                "Complete detector status/terminal receipt identity differs."
            )
        marker_identity = None
        if (
            gpu_accounting_ledger_path is not None
            and candidate.get("kind") == "detector_run_manifest"
        ):
            marker_path = detector_gpu_ledger_incorporation_path(
                Path(gpu_accounting_ledger_path)
            )
            marker = write_detector_gpu_ledger_incorporation_marker(
                marker_path,
                ledger_path=Path(gpu_accounting_ledger_path),
                final_terminal_receipt_path=receipt_path,
            )
            mark_detector_execution_complete(
                closed_status_file,
                run_identity_sha256=candidate["run_identity_sha256"],
                final_manifest=Path(candidate["requested_output_path"]),
                terminal_receipt=receipt_path,
                ledger_incorporation_marker=marker_path,
            )
            marker_identity = {
                "path": str(marker_path.resolve()),
                "sha256": file_sha256(marker_path),
                "size_bytes": int(marker_path.stat().st_size),
                "incorporation_sha256": marker["incorporation_sha256"],
            }
        return {
            "published_output": receipt["published_output"],
            "terminal_receipt": dict(status["terminal_receipt"]),
            "gpu_accounting": receipt["gpu_accounting"],
            "gpu_ledger_incorporation": marker_identity,
        }
    accounting = status.get("gpu_accounting")
    if (
        status.get("run_identity_sha256") != candidate["run_identity_sha256"]
        or status.get("state") != "supervisor_observed_process_exit"
        or not isinstance(accounting, dict)
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
        or accounting["intervals"][-1].get("completed_at_utc") is None
    ):
        raise RevisionDetectionError(
            "Detector finalization requires closed supervisor-observed GPU status."
        )
    ledger_identity = None
    if gpu_accounting_ledger_path is not None:
        ledger_path = Path(gpu_accounting_ledger_path)
        ledger = read_detector_gpu_accounting_ledger(ledger_path)
        if candidate.get("kind") == "detector_run_manifest":
            observed_sources = {
                str(source["source_id"]): str(source["component"])
                for source in ledger["sources"]
            }
            if observed_sources != EXPECTED_GPU_LEDGER_SOURCES:
                raise RevisionDetectionError(
                    "Canonical detector finalization requires the exact six "
                    "pre-final GPU ledger sources."
                )
        if ledger.get("gpu_uuid") != accounting.get("gpu_uuid"):
            raise RevisionDetectionError(
                "Detector GPU ledger/final status UUID differs."
            )
        by_identity: Dict[Tuple[int, int, str, str], dict] = {}
        for interval in list(ledger["intervals"]) + list(
            accounting["intervals"]
        ):
            key = (
                int(interval["pid"]),
                int(interval["process_start_ticks"]),
                str(interval["started_at_utc"]),
                str(interval.get("completed_at_utc")),
            )
            prior = by_identity.get(key)
            if prior is not None and prior != interval:
                raise RevisionDetectionError(
                    "Detector final accounting interval identity conflicts."
                )
            by_identity[key] = dict(interval)
        intervals = sorted(
            by_identity.values(), key=lambda value: value["started_at_utc"]
        )
        previous_end: Optional[datetime] = None
        for interval in intervals:
            start = datetime.fromisoformat(str(interval["started_at_utc"]))
            end = datetime.fromisoformat(str(interval["completed_at_utc"]))
            if previous_end is not None and start < previous_end:
                raise RevisionDetectionError(
                    "Detector final GPU accounting intervals overlap."
                )
            if end < start:
                raise RevisionDetectionError(
                    "Detector final GPU accounting interval is negative."
                )
            previous_end = end
        accounting = {
            "device": "cuda:0",
            "gpu_uuid": accounting["gpu_uuid"],
            "intervals": intervals,
            "cumulative_elapsed_seconds": float(
                sum(float(value["elapsed_seconds"]) for value in intervals)
            ),
            "derivation_policy": GPU_ACCOUNTING_COLLECTION_POLICY,
        }
        ledger_identity = {
            "path": str(ledger_path.resolve()),
            "sha256": file_sha256(ledger_path),
            "size_bytes": int(ledger_path.stat().st_size),
            "ledger_sha256": ledger["ledger_sha256"],
            "sources_sha256": ledger["sources_sha256"],
            "intervals_sha256": ledger["intervals_sha256"],
            "cumulative_elapsed_seconds": ledger[
                "cumulative_elapsed_seconds"
            ],
        }
        if (
            candidate.get("kind") == "detector_run_manifest"
            and candidate.get("payload", {}).get(
                "pre_final_gpu_accounting_ledger"
            )
            != ledger_identity
        ):
            raise RevisionDetectionError(
                "Detector finalization ledger differs from the identity bound "
                "before production fitting."
            )
    payload = dict(candidate["payload"])
    payload["gpu_accounting"] = accounting
    payload["pre_final_gpu_accounting_ledger"] = ledger_identity
    payload["terminal_accounting_status_sha256"] = status["status_sha256"]
    payload["finalization_candidate"] = {
        "path": str(Path(candidate_path).resolve()),
        "sha256": file_sha256(Path(candidate_path)),
        "size_bytes": int(Path(candidate_path).stat().st_size),
        "candidate_sha256": candidate["candidate_sha256"],
    }
    payload["execution_completed_at_utc"] = accounting["intervals"][-1][
        "completed_at_utc"
    ]
    if candidate["kind"] == "equivalence_artifact":
        provenance = dict(payload.get("provenance", {}))
        provenance["gpu_accounting"] = accounting
        provenance["terminal_accounting_status_sha256"] = status[
            "status_sha256"
        ]
        payload["provenance"] = provenance
        payload["provenance_sha256"] = canonical_json_sha256(provenance)
        published = _signed_document(payload, "artifact_sha256")
    elif candidate["kind"] == "benchmark_artifact":
        published = _signed_document(payload, "benchmark_sha256")
    else:
        published = _signed_document(payload, "manifest_sha256")
    output_path = Path(candidate["requested_output_path"])
    if output_path.exists():
        existing = _read_json(output_path, "supervisor-published detector output")
        if existing != published:
            raise RevisionDetectionError(
                "Existing supervisor-published detector output differs."
            )
    else:
        atomic_write_json(output_path, published)
    if receipt_path.exists():
        existing_receipt = _validate_terminal_receipt(receipt_path)
        if (
            existing_receipt.get("published_output", {}).get("sha256")
            != file_sha256(output_path)
            or existing_receipt.get("closed_status_sha256")
            != status["status_sha256"]
        ):
            raise RevisionDetectionError(
                "Existing detector terminal receipt differs."
            )
        receipt = existing_receipt
    else:
        receipt = write_detector_terminal_receipt(
            receipt_path,
            candidate_path=candidate_path,
            published_output_path=output_path,
            closed_status=status,
            gpu_accounting=accounting,
        )
    if candidate["kind"] == "equivalence_artifact":
        read_detector_equivalence_fit_artifact(output_path)
    marker_path: Optional[Path] = None
    marker_identity = None
    if (
        gpu_accounting_ledger_path is not None
        and candidate.get("kind") == "detector_run_manifest"
    ):
        marker_path = detector_gpu_ledger_incorporation_path(
            Path(gpu_accounting_ledger_path)
        )
        marker = write_detector_gpu_ledger_incorporation_marker(
            marker_path,
            ledger_path=Path(gpu_accounting_ledger_path),
            final_terminal_receipt_path=receipt_path,
        )
        marker_identity = {
            "path": str(marker_path.resolve()),
            "sha256": file_sha256(marker_path),
            "size_bytes": int(marker_path.stat().st_size),
            "incorporation_sha256": marker["incorporation_sha256"],
        }
    mark_detector_execution_complete(
        closed_status_file,
        run_identity_sha256=candidate["run_identity_sha256"],
        final_manifest=output_path,
        terminal_receipt=receipt_path,
        ledger_incorporation_marker=marker_path,
    )
    return {
        "published_output": receipt["published_output"],
        "terminal_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": file_sha256(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
            "terminal_receipt_sha256": receipt["terminal_receipt_sha256"],
        },
        "gpu_accounting": accounting,
        "gpu_ledger_incorporation": marker_identity,
    }


def _sha256_int_sequence(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_text_sequence(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scientific_detector_config(config: Mapping[str, object]) -> dict:
    value = dict(config)
    value.pop("device", None)
    value.pop("torch_num_threads", None)
    return value


@dataclass(frozen=True)
class DetectorFitTask:
    ordinal: int
    split_position: int
    detector_position: int
    split: DetectorSplit
    detector_config: Mapping[str, object]
    seed: int
    identity: Mapping[str, object]
    identity_sha256: str

    @property
    def detector_name(self) -> str:
        return str(
            self.detector_config.get("name", self.detector_config.get("kind", ""))
        )

    @property
    def detector_kind(self) -> str:
        return str(self.detector_config.get("kind", ""))


def build_fit_tasks(prepared: PreparedDetectorSuite) -> List[DetectorFitTask]:
    """Build tasks in exact legacy split-outer/detector-inner order."""

    tasks: List[DetectorFitTask] = []
    detector_count = len(prepared.detector_configs)
    for split_position, split in enumerate(prepared.splits):
        train_row_ids = prepared.normalized_frame.iloc[
            list(split.train_indices)
        ]["row_id"].astype(str).tolist()
        test_row_ids = prepared.normalized_frame.iloc[
            list(split.test_indices)
        ]["row_id"].astype(str).tolist()
        for detector_position, detector_config in enumerate(
            prepared.detector_configs
        ):
            ordinal = split_position * detector_count + detector_position
            seed = detector_fit_seed(prepared, split, detector_config)
            effective_config = dict(detector_config)
            identity = {
                "ordinal": int(ordinal),
                "split_position": int(split_position),
                "detector_position": int(detector_position),
                "split_id": split.split_id,
                "regime": split.regime,
                "held_out_column": split.held_out_column,
                "held_out_value": split.held_out_value,
                "partition_policy": split.partition_policy,
                "purged_train_rows": int(split.purged_train_rows),
                "excluded_held_out_rows": int(split.excluded_held_out_rows),
                "train_row_count": int(len(split.train_indices)),
                "test_row_count": int(len(split.test_indices)),
                "train_indices_sha256": _sha256_int_sequence(split.train_indices),
                "test_indices_sha256": _sha256_int_sequence(split.test_indices),
                "train_row_ids_ordered_sha256": _sha256_text_sequence(train_row_ids),
                "test_row_ids_ordered_sha256": _sha256_text_sequence(test_row_ids),
                "detector_name": str(
                    detector_config.get("name", detector_config.get("kind", ""))
                ),
                "detector_kind": str(detector_config.get("kind", "")),
                "effective_detector_config": effective_config,
                "effective_detector_config_sha256": canonical_json_sha256(
                    effective_config
                ),
                "scientific_detector_config_sha256": canonical_json_sha256(
                    _scientific_detector_config(detector_config)
                ),
                "seed": int(seed),
                "bootstrap_resamples": int(prepared.bootstrap_resamples),
                "decision_threshold": float(prepared.threshold),
            }
            tasks.append(
                DetectorFitTask(
                    ordinal=ordinal,
                    split_position=split_position,
                    detector_position=detector_position,
                    split=split,
                    detector_config=detector_config,
                    seed=seed,
                    identity=identity,
                    identity_sha256=canonical_json_sha256(identity),
                )
            )
    return tasks


@dataclass
class DetectorExecutionContext:
    output_dir: Path
    checkpoint_dir: Path
    status_file: Path
    device: str
    gpu_uuid: Optional[str]
    workers: int
    lineage: Mapping[str, object]
    source: Mapping[str, object]
    resume: bool = True
    heartbeat_seconds: float = 30.0
    benchmark_seconds_per_fit: Optional[float] = None
    execution_policy: Optional[Mapping[str, object]] = None
    execution_policy_path: Optional[Path] = None
    execution_policy_sha256: Optional[str] = None
    fit_permit_file: Optional[Path] = None
    fit_permit_receipt_dir: Optional[Path] = None
    require_fit_permit: bool = False


@dataclass
class DetectorExecutionOutcome:
    result: Optional[DetectorSuiteResult]
    completed_fit_count: int
    total_fit_count: int
    resumed_fit_count: int
    recovered_errors: List[dict]
    fit_durations_seconds: List[float]
    run_identity: dict
    run_identity_sha256: str
    plan_sha256: str
    started_at_utc: str
    completed_at_utc: Optional[str]
    stopped_at_fit_boundary: bool = False
    last_completed_checkpoint: Optional[dict] = None
    gpu_accounting: Optional[dict] = None


def _validate_runtime(context: DetectorExecutionContext) -> None:
    if context.workers != 1:
        raise RevisionDetectionError(
            "Only --workers 1 is currently implemented safely; refusing unbounded "
            "or oversubscribed detector execution."
        )
    if context.device not in {"cpu", "cuda:0"}:
        raise RevisionDetectionError("Detector device must be cpu or cuda:0.")
    if context.device == "cuda:0":
        if not context.gpu_uuid or not str(context.gpu_uuid).startswith("GPU-"):
            raise RevisionDetectionError(
                "CUDA detector execution requires an exact --gpu-uuid beginning GPU-."
            )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(context.gpu_uuid):
            raise RevisionDetectionError(
                "CUDA_VISIBLE_DEVICES must equal the exact detector GPU UUID before "
                "execution; observed {!r}.".format(visible)
            )
    elif context.gpu_uuid is not None:
        raise RevisionDetectionError("--gpu-uuid is valid only with --device cuda:0.")
    if context.require_fit_permit and context.fit_permit_file is None:
        raise RevisionDetectionError(
            "Ceiling-gated detector execution requires a fit permit file."
        )
    if context.require_fit_permit and context.fit_permit_receipt_dir is None:
        raise RevisionDetectionError(
            "Ceiling-gated detector execution requires a fit permit receipt directory."
        )


def _next_fit_upper_seconds(
    context: DetectorExecutionContext, task: DetectorFitTask
) -> float:
    if context.execution_policy is None:
        raise RevisionDetectionError("Detector execution policy is unavailable.")
    try:
        value = float(
            context.execution_policy["ceiling"][
                "next_fit_upper_seconds_by_detector"
            ][task.detector_name]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RevisionDetectionError(
            "Detector execution policy lacks a next-fit bound for {}.".format(
                task.detector_name
            )
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RevisionDetectionError("Detector next-fit upper bound is invalid.")
    return value


def _fit_gate_nonce(
    run_identity_sha256: str,
    task: DetectorFitTask,
    last_checkpoint: Optional[Mapping[str, object]],
    *,
    invocation_pid: int,
    invocation_start_ticks: int,
) -> str:
    return canonical_json_sha256(
        {
            "run_identity_sha256": run_identity_sha256,
            "task_identity_sha256": task.identity_sha256,
            "prior_checkpoint_sha256": (
                None if last_checkpoint is None else last_checkpoint.get("sha256")
            ),
            "invocation_pid": int(invocation_pid),
            "invocation_start_ticks": int(invocation_start_ticks),
        }
    )


def fit_permit_receipt_path(
    context: DetectorExecutionContext, gate_nonce: str
) -> Path:
    """Return the deterministic durable receipt path checked by the issuer."""

    if context.fit_permit_receipt_dir is None or not re.fullmatch(
        r"[0-9a-f]{64}", str(gate_nonce)
    ):
        raise RevisionDetectionError("Detector fit receipt identity is invalid.")
    return Path(context.fit_permit_receipt_dir) / (str(gate_nonce) + ".json")


def _validate_fit_permit_receipt(
    path: Path,
    *,
    run_identity_sha256: str,
    task: DetectorFitTask,
    gate_nonce: str,
    invocation_pid: int,
    invocation_start_ticks: int,
    expected_upper: float,
) -> dict:
    receipt = _read_json(path, "detector fit permit receipt")
    if receipt.get("schema_version") != PERMIT_RECEIPT_SCHEMA:
        raise RevisionDetectionError("Detector fit permit receipt schema differs.")
    _verify_signed_document(
        receipt, "receipt_sha256", "Detector fit permit receipt"
    )
    if set(receipt) != {
        "schema_version",
        "run_identity_sha256",
        "task_identity_sha256",
        "fit_gate_nonce",
        "invocation_pid",
        "invocation_start_ticks",
        "next_fit_upper_seconds",
        "issued_permit_sha256",
        "issued_permit_file_sha256",
        "issued_permit_size_bytes",
        "consumed_at_utc",
        "receipt_sha256",
    }:
        raise RevisionDetectionError("Detector fit permit receipt shape differs.")
    if (
        receipt.get("run_identity_sha256") != run_identity_sha256
        or receipt.get("task_identity_sha256") != task.identity_sha256
        or receipt.get("fit_gate_nonce") != gate_nonce
        or int(receipt.get("invocation_pid", -1)) != int(invocation_pid)
        or int(receipt.get("invocation_start_ticks", -1))
        != int(invocation_start_ticks)
        or float(receipt.get("next_fit_upper_seconds", -1.0)) != expected_upper
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("issued_permit_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("issued_permit_file_sha256", ""))
        )
        or int(receipt.get("issued_permit_size_bytes", -1)) <= 0
    ):
        raise RevisionDetectionError("Detector fit permit receipt identity differs.")
    try:
        consumed = datetime.fromisoformat(str(receipt["consumed_at_utc"]))
    except ValueError as exc:
        raise RevisionDetectionError(
            "Detector fit permit receipt timestamp is invalid."
        ) from exc
    if consumed.tzinfo is None or consumed > _utc_now() + timedelta(minutes=5):
        raise RevisionDetectionError(
            "Detector fit permit receipt timestamp is invalid."
        )
    return receipt


def _consume_fit_permit(
    context: DetectorExecutionContext,
    *,
    run_identity_sha256: str,
    task: DetectorFitTask,
    gate_nonce: str,
    invocation_pid: int,
    invocation_start_ticks: int,
    recovered_errors: List[dict],
    stop_event: Optional[threading.Event] = None,
) -> Tuple[bool, Optional[dict]]:
    if not context.require_fit_permit:
        return True, None
    assert context.fit_permit_file is not None
    path = Path(context.fit_permit_file)
    while True:
        if path.is_symlink():
            raise RevisionDetectionError("Detector fit permit path is a symlink.")
        if path.is_file():
            permit = _read_json(path, "detector fit permit")
            if permit.get("schema_version") != PERMIT_SCHEMA:
                raise RevisionDetectionError("Detector fit permit schema differs.")
            _verify_signed_document(permit, "permit_sha256", "Detector fit permit")
            if set(permit) != {
                "schema_version",
                "run_identity_sha256",
                "task_identity_sha256",
                "fit_gate_nonce",
                "invocation_pid",
                "invocation_start_ticks",
                "next_fit_upper_seconds",
                "issued_at_utc",
                "permit_sha256",
            }:
                raise RevisionDetectionError("Detector fit permit shape differs.")
            expected_upper = _next_fit_upper_seconds(context, task)
            if (
                permit.get("run_identity_sha256") != run_identity_sha256
                or permit.get("task_identity_sha256") != task.identity_sha256
                or float(permit.get("next_fit_upper_seconds", -1.0)) != expected_upper
            ):
                raise RevisionDetectionError("Detector fit permit identity differs.")
            permit_pid = int(permit.get("invocation_pid", -1))
            permit_ticks = int(permit.get("invocation_start_ticks", -1))
            if (
                permit_pid != int(invocation_pid)
                or permit_ticks != int(invocation_start_ticks)
            ):
                stale_hash = file_sha256(path)
                recovered_directory = _ensure_real_directory(
                    Path(context.checkpoint_dir) / "recovered_fit_permits"
                )
                recovered_path = recovered_directory / (
                    "stale-{}-{}-{}.json".format(
                        permit_pid,
                        permit_ticks,
                        stale_hash,
                    )
                )
                if recovered_path.exists():
                    if (
                        recovered_path.is_symlink()
                        or not recovered_path.is_file()
                        or file_sha256(recovered_path) != stale_hash
                    ):
                        raise RevisionDetectionError(
                            "Recovered stale detector permit identity differs."
                        )
                    path.unlink()
                else:
                    os.replace(str(path), str(recovered_path))
                _fsync_directory(path.parent)
                _fsync_directory(recovered_directory)
                recovered_errors.append(
                    {
                        "type": "stale_fit_permit_quarantined",
                        "path": str(recovered_path.resolve()),
                        "sha256": stale_hash,
                        "stale_invocation_pid": permit_pid,
                        "stale_invocation_start_ticks": permit_ticks,
                        "current_invocation_pid": int(invocation_pid),
                        "current_invocation_start_ticks": int(
                            invocation_start_ticks
                        ),
                        "task_identity_sha256": task.identity_sha256,
                        "action": "fresh_ceiling_gate_required",
                    }
                )
                continue
            if permit.get("fit_gate_nonce") != gate_nonce:
                raise RevisionDetectionError("Detector fit permit nonce differs.")
            try:
                issued = datetime.fromisoformat(str(permit["issued_at_utc"]))
            except ValueError as exc:
                raise RevisionDetectionError(
                    "Detector fit permit timestamp is invalid."
                ) from exc
            if issued.tzinfo is None or issued > _utc_now() + timedelta(minutes=5):
                raise RevisionDetectionError(
                    "Detector fit permit timestamp is invalid."
                )
            permit_file_sha256 = file_sha256(path)
            permit_file_size = int(path.stat().st_size)
            receipt_path = fit_permit_receipt_path(context, gate_nonce)
            if receipt_path.exists():
                if receipt_path.is_symlink() or not receipt_path.is_file():
                    raise RevisionDetectionError(
                        "Detector fit permit receipt path is unsafe."
                    )
                receipt = _validate_fit_permit_receipt(
                    receipt_path,
                    run_identity_sha256=run_identity_sha256,
                    task=task,
                    gate_nonce=gate_nonce,
                    invocation_pid=invocation_pid,
                    invocation_start_ticks=invocation_start_ticks,
                    expected_upper=expected_upper,
                )
                if (
                    receipt["issued_permit_sha256"] != permit["permit_sha256"]
                    or receipt["issued_permit_file_sha256"]
                    != permit_file_sha256
                    or int(receipt["issued_permit_size_bytes"])
                    != permit_file_size
                ):
                    raise RevisionDetectionError(
                        "Existing detector fit permit receipt identifies different bytes."
                    )
            else:
                receipt = _signed_document(
                    {
                        "schema_version": PERMIT_RECEIPT_SCHEMA,
                        "run_identity_sha256": run_identity_sha256,
                        "task_identity_sha256": task.identity_sha256,
                        "fit_gate_nonce": gate_nonce,
                        "invocation_pid": int(invocation_pid),
                        "invocation_start_ticks": int(invocation_start_ticks),
                        "next_fit_upper_seconds": float(expected_upper),
                        "issued_permit_sha256": permit["permit_sha256"],
                        "issued_permit_file_sha256": permit_file_sha256,
                        "issued_permit_size_bytes": permit_file_size,
                        "consumed_at_utc": _utc_text(_utc_now()),
                    },
                    "receipt_sha256",
                )
                atomic_write_json(receipt_path, receipt)
                receipt = _validate_fit_permit_receipt(
                    receipt_path,
                    run_identity_sha256=run_identity_sha256,
                    task=task,
                    gate_nonce=gate_nonce,
                    invocation_pid=invocation_pid,
                    invocation_start_ticks=invocation_start_ticks,
                    expected_upper=expected_upper,
                )
            return True, {
                "path": str(receipt_path.resolve()),
                "sha256": file_sha256(receipt_path),
                "size_bytes": int(receipt_path.stat().st_size),
                "receipt_sha256": receipt["receipt_sha256"],
                "fit_gate_nonce": receipt["fit_gate_nonce"],
                "task_identity_sha256": receipt["task_identity_sha256"],
                "invocation_pid": int(receipt["invocation_pid"]),
                "invocation_start_ticks": int(
                    receipt["invocation_start_ticks"]
                ),
                "consumed_at_utc": receipt["consumed_at_utc"],
            }
        if stop_event is not None and stop_event.wait(0.25):
            return False, None
        time.sleep(0.25 if stop_event is None else 0.0)


def _retire_consumed_fit_permit(
    context: DetectorExecutionContext, receipt_identity: Mapping[str, object]
) -> None:
    """Remove a consumed permit only after signed status acknowledges its receipt."""

    if context.fit_permit_file is None:
        raise RevisionDetectionError("Detector fit permit path is unavailable.")
    receipt_path = Path(str(receipt_identity.get("path", "")))
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or file_sha256(receipt_path) != receipt_identity.get("sha256")
        or int(receipt_path.stat().st_size)
        != int(receipt_identity.get("size_bytes", -1))
    ):
        raise RevisionDetectionError(
            "Acknowledged detector fit permit receipt identity differs."
        )
    receipt = _read_json(receipt_path, "detector fit permit receipt")
    _verify_signed_document(
        receipt, "receipt_sha256", "Detector fit permit receipt"
    )
    if (
        receipt.get("receipt_sha256")
        != receipt_identity.get("receipt_sha256")
        or receipt.get("fit_gate_nonce")
        != receipt_identity.get("fit_gate_nonce")
    ):
        raise RevisionDetectionError(
            "Acknowledged detector fit permit receipt content differs."
        )
    permit_path = Path(context.fit_permit_file)
    if permit_path.is_symlink() or not permit_path.is_file():
        raise RevisionDetectionError(
            "Acknowledged detector fit permit disappeared before retirement."
        )
    if (
        file_sha256(permit_path) != receipt.get("issued_permit_file_sha256")
        or int(permit_path.stat().st_size)
        != int(receipt.get("issued_permit_size_bytes", -1))
    ):
        raise RevisionDetectionError(
            "Acknowledged detector fit permit bytes changed before retirement."
        )
    permit = _read_json(permit_path, "detector fit permit")
    _verify_signed_document(permit, "permit_sha256", "Detector fit permit")
    if (
        permit.get("permit_sha256") != receipt.get("issued_permit_sha256")
        or permit.get("fit_gate_nonce") != receipt.get("fit_gate_nonce")
    ):
        raise RevisionDetectionError(
            "Acknowledged detector fit permit content changed before retirement."
        )
    permit_path.unlink()
    _fsync_directory(permit_path.parent)


def _process_start_ticks() -> Optional[int]:
    try:
        text = Path("/proc/self/stat").read_text(encoding="utf-8")
        remainder = text[text.rfind(")") + 2 :].split()
        return int(remainder[19])
    except Exception:
        return None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def _peak_vram_bytes(device: str) -> int:
    if device != "cuda:0":
        return 0
    torch = sys.modules.get("torch")
    if torch is None:
        return 0
    try:
        return int(torch.cuda.max_memory_allocated(0))
    except Exception:
        return 0


def _build_plan(prepared: PreparedDetectorSuite, tasks: Sequence[DetectorFitTask]) -> dict:
    split_ids = [split.split_id for split in prepared.splits]
    detector_ids = [
        {
            "name": str(config.get("name", config.get("kind", ""))),
            "kind": str(config.get("kind", "")),
        }
        for config in prepared.detector_configs
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "split_count": int(len(prepared.splits)),
        "detector_count": int(len(prepared.detector_configs)),
        "total_fit_count": int(len(tasks)),
        "split_order": split_ids,
        "split_order_sha256": canonical_json_sha256(split_ids),
        "detector_order": detector_ids,
        "detector_order_sha256": canonical_json_sha256(detector_ids),
        "base_seed": int(prepared.seed),
        "bootstrap_resamples": int(prepared.bootstrap_resamples),
        "decision_threshold": float(prepared.threshold),
        "tasks": [dict(task.identity) for task in tasks],
        "task_identity_sha256_order": [task.identity_sha256 for task in tasks],
    }


def _build_run_identity(
    context: DetectorExecutionContext, plan: Mapping[str, object]
) -> Tuple[dict, str]:
    output_dir = Path(context.output_dir).resolve()
    checkpoint_dir = Path(context.checkpoint_dir).resolve()
    status_file = Path(context.status_file).resolve()
    plan_sha256 = canonical_json_sha256(plan)
    # Passing reports and the pre-final accounting ledger are publication gates,
    # not inputs to a detector fit.  Excluding only these explicitly enumerated
    # operational fields keeps an authorized production benchmark checkpoint
    # reusable for CUDA equivalence export and the remaining frozen matrix.
    operational_gate_fields = {
        "required_equivalence_reports",
        "pre_final_gpu_accounting_ledger_path",
        "pre_final_gpu_accounting_ledger",
    }
    checkpoint_lineage = {
        str(name): value
        for name, value in context.lineage.items()
        if name not in operational_gate_fields
    }
    identity = {
        "schema_version": "rankcloak-revision-detector-run-identity-v1",
        "input_sha256": context.lineage.get("input_sha256"),
        "preprocessing_manifest_sha256": context.lineage.get(
            "preprocessing_manifest_sha256"
        ),
        "config_sha256": context.lineage.get("config_sha256"),
        "confirmatory_plan_sha256": context.lineage.get(
            "confirmatory_plan_sha256"
        ),
        "lineage": checkpoint_lineage,
        "lineage_sha256": canonical_json_sha256(checkpoint_lineage),
        "excluded_operational_gate_fields": sorted(operational_gate_fields),
        "source": dict(context.source),
        "source_sha256": canonical_json_sha256(context.source),
        "plan_sha256": plan_sha256,
        "device": context.device,
        "gpu_uuid": context.gpu_uuid,
        "workers": int(context.workers),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "status_file": str(status_file),
        "execution_policy_path": (
            None
            if context.execution_policy_path is None
            else str(Path(context.execution_policy_path).resolve())
        ),
        "execution_policy_sha256": context.execution_policy_sha256,
        "fit_permit_file": (
            None
            if context.fit_permit_file is None
            else str(Path(context.fit_permit_file).resolve())
        ),
        "fit_permit_receipt_dir": (
            None
            if context.fit_permit_receipt_dir is None
            else str(Path(context.fit_permit_receipt_dir).resolve())
        ),
        "require_fit_permit": bool(context.require_fit_permit),
    }
    return identity, canonical_json_sha256(identity)


def _signed_document(value: Mapping[str, object], hash_field: str) -> dict:
    unsigned = dict(value)
    unsigned.pop(hash_field, None)
    unsigned[hash_field] = canonical_json_sha256(unsigned)
    return unsigned


def _verify_signed_document(
    value: Mapping[str, object], hash_field: str, label: str
) -> None:
    observed = value.get(hash_field)
    unsigned = dict(value)
    unsigned.pop(hash_field, None)
    expected = canonical_json_sha256(unsigned)
    if observed != expected:
        raise RevisionDetectionError("{} self-hash differs.".format(label))


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RevisionDetectionError("{} is missing or unsafe: {}".format(label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RevisionDetectionError("{} is invalid JSON: {}".format(label, exc)) from exc
    if not isinstance(value, dict):
        raise RevisionDetectionError("{} must be a JSON object.".format(label))
    return value


def _task_directory(checkpoint_dir: Path, task: DetectorFitTask) -> Path:
    return Path(checkpoint_dir) / "fits" / "{:04d}".format(task.ordinal)


def _row_payload(rows: Sequence[Mapping[str, object]]) -> dict:
    columns = [] if not rows else list(rows[0].keys())
    if any(list(row.keys()) != columns for row in rows):
        raise RevisionDetectionError("Detector checkpoint rows have inconsistent ordering.")
    return {
        "schema_version": ROW_ARTIFACT_SCHEMA,
        "columns": columns,
        "rows": [[row[column] for column in columns] for row in rows],
    }


def _payload_rows(payload: Mapping[str, object], label: str) -> List[dict]:
    if payload.get("schema_version") != ROW_ARTIFACT_SCHEMA:
        raise RevisionDetectionError("{} has an unexpected schema.".format(label))
    columns = payload.get("columns")
    rows = payload.get("rows")
    if (
        set(payload) != {"schema_version", "columns", "rows"}
        or
        not isinstance(columns, list)
        or len(columns) != len(set(map(str, columns)))
        or not isinstance(rows, list)
        or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
    ):
        raise RevisionDetectionError("{} row payload is malformed.".format(label))
    return [dict(zip(map(str, columns), row)) for row in rows]


def _checkpoint_manifest_path(checkpoint_dir: Path, task: DetectorFitTask) -> Path:
    return _task_directory(checkpoint_dir, task) / "manifest.json"


def _validate_checkpoint_semantics(
    prepared: PreparedDetectorSuite,
    task: DetectorFitTask,
    metrics: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> None:
    if len(metrics) != 1:
        raise RevisionDetectionError("Detector fit checkpoint must contain one metric row.")
    metric = metrics[0]
    expected = task.identity
    train = prepared.normalized_frame.iloc[list(task.split.train_indices)]
    expected_test = prepared.normalized_frame.iloc[
        list(task.split.test_indices)
    ]
    if (
        metric.get("split_id") != expected["split_id"]
        or metric.get("regime") != expected["regime"]
        or metric.get("held_out_column") != expected["held_out_column"]
        or metric.get("held_out_value") != expected["held_out_value"]
        or metric.get("detector_name") != expected["detector_name"]
        or metric.get("requested_kind") != expected["detector_kind"]
        or int(metric.get("seed", -1)) != int(expected["seed"])
        or int(metric.get("train_rows", -1)) != int(expected["train_row_count"])
        or int(metric.get("test_rows", -1)) != int(expected["test_row_count"])
        or int(metric.get("train_payload_groups", -1))
        != int(train["payload_group_id"].nunique())
        or int(metric.get("purged_train_rows", -1))
        != int(expected["purged_train_rows"])
        or float(metric.get("decision_threshold", float("nan")))
        != float(prepared.threshold)
        or metric.get("bootstrap_unit") != "payload_group_id"
        or int(metric.get("bootstrap_resamples_requested", -1))
        != int(prepared.bootstrap_resamples)
        or int(metric.get("test_payload_groups", -1))
        != int(expected_test["payload_group_id"].nunique())
    ):
        raise RevisionDetectionError("Detector metric checkpoint task identity differs.")
    if prepared.smoke:
        if metric.get("implementation_status") not in {
            "complete",
            "smoke_only",
            "smoke_fallback",
        }:
            raise RevisionDetectionError(
                "Detector smoke checkpoint implementation status differs."
            )
    elif (
        metric.get("implementation_status") != "complete"
        or metric.get("implementation_kind") != expected["detector_kind"]
    ):
        raise RevisionDetectionError(
            "Confirmatory detector checkpoint is not the requested complete implementation."
        )
    metadata = _implementation_metadata(metric)
    implemented_kind = str(metric.get("implementation_kind", ""))
    if implemented_kind in {"text_cnn", "pretrained_transformer"}:
        required_metadata = {
            "model_state_hash_algorithm": "rankcloak-torch-state-v1",
            "model_state_schema_hash_algorithm": "rankcloak-torch-state-schema-v1",
        }
        if (
            any(metadata.get(name) != value for name, value in required_metadata.items())
            or len(str(metadata.get("model_state_sha256", ""))) != 64
            or len(str(metadata.get("model_state_schema_sha256", ""))) != 64
            or metric.get("model_state_sha256") != metadata.get("model_state_sha256")
            or metric.get("model_state_hash_algorithm")
            != metadata.get("model_state_hash_algorithm")
        ):
            raise RevisionDetectionError(
                "Detector checkpoint trained-state provenance differs."
            )
        if implemented_kind == "pretrained_transformer" and (
            len(str(metadata.get("model_artifact_set_sha256", ""))) != 64
            or metric.get("model_artifact_set_sha256")
            != metadata.get("model_artifact_set_sha256")
        ):
            raise RevisionDetectionError(
                "Transformer checkpoint artifact provenance differs."
            )
    if len(predictions) != int(expected["test_row_count"]):
        raise RevisionDetectionError("Detector prediction checkpoint row count differs.")
    observed_row_ids = [str(row.get("row_id", "")) for row in predictions]
    expected_row_ids = expected_test["row_id"].astype(str).tolist()
    if observed_row_ids != expected_row_ids:
        raise RevisionDetectionError("Detector checkpoint prediction row order differs.")
    identity_columns = (
        "row_id",
        "payload_group_id",
        "prompt_template_id",
        "model_id",
        "codec_id",
        "label",
    )
    for row, (_, source_row) in zip(predictions, expected_test.iterrows()):
        score = float(row.get("score", float("nan")))
        if (
            row.get("split_id") != expected["split_id"]
            or row.get("regime") != expected["regime"]
            or row.get("held_out_value") != expected["held_out_value"]
            or row.get("detector_name") != expected["detector_name"]
            or row.get("requested_kind") != expected["detector_kind"]
            or row.get("implementation_kind")
            != metric.get("implementation_kind")
            or row.get("implementation_status")
            != metric.get("implementation_status")
            or any(row.get(name) != source_row[name] for name in identity_columns)
            or not math.isfinite(score)
            or score < 0.0
            or score > 1.0
            or int(row.get("prediction", -1))
            != int(score >= float(prepared.threshold))
        ):
            raise RevisionDetectionError("Detector prediction checkpoint is invalid.")


def _load_valid_checkpoint(
    prepared: PreparedDetectorSuite,
    context: DetectorExecutionContext,
    task: DetectorFitTask,
    run_identity_sha256: str,
    plan_sha256: str,
    recovered_errors: List[dict],
) -> Optional[Tuple[dict, List[dict], float, dict]]:
    directory = _task_directory(context.checkpoint_dir, task)
    manifest_path = directory / "manifest.json"
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise RevisionDetectionError(
            "Detector fit checkpoint path is unsafe: {}".format(directory)
        )
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    symlinks = [path.name for path in entries if path.is_symlink()]
    if symlinks:
        raise RevisionDetectionError(
            "Detector fit checkpoint contains symlinks: {}".format(", ".join(symlinks))
        )
    if not manifest_path.exists():
        allowed_partial = {"metric.json", "predictions.json"}
        temporary_pattern = re.compile(
            r"^\.tmp-(?:metric\.json|predictions\.json|manifest\.json)-[A-Za-z0-9_-]+$"
        )
        unexpected = [
            path.name
            for path in entries
            if path.name not in allowed_partial
            and temporary_pattern.fullmatch(path.name) is None
        ]
        if unexpected:
            raise RevisionDetectionError(
                "Incomplete detector checkpoint contains unexpected entries: {}".format(
                    ", ".join(unexpected)
                )
            )
        if entries:
            if any(path.is_symlink() or not path.is_file() for path in entries):
                raise RevisionDetectionError(
                    "Incomplete detector checkpoint contains unsafe orphan entries."
                )
            recovery_directory = _ensure_real_directory(
                Path(context.checkpoint_dir)
                / "recovered_orphan_fit_files"
                / "{:04d}".format(task.ordinal)
            )
            recovered_files = []
            for path in entries:
                digest = file_sha256(path)
                size = int(path.stat().st_size)
                destination = recovery_directory / "{}.{}.orphan".format(
                    path.name, digest
                )
                if destination.exists():
                    if (
                        destination.is_symlink()
                        or not destination.is_file()
                        or file_sha256(destination) != digest
                        or int(destination.stat().st_size) != size
                    ):
                        raise RevisionDetectionError(
                            "Recovered orphan detector artifact identity differs."
                        )
                    path.unlink()
                else:
                    os.replace(str(path), str(destination))
                recovered_files.append(
                    {
                        "source_name": path.name,
                        "preserved_path": str(destination.resolve()),
                        "sha256": digest,
                        "size_bytes": size,
                    }
                )
            _fsync_directory(directory)
            _fsync_directory(recovery_directory)
            recovered_errors.append(
                {
                    "type": "orphaned_incomplete_fit_checkpoint",
                    "task_ordinal": int(task.ordinal),
                    "path": str(directory),
                    "files": recovered_files,
                    "action": "orphans_preserved_and_fit_will_be_recomputed",
                }
            )
        return None
    allowed = {"metric.json", "predictions.json", "manifest.json"}
    observed = {path.name for path in entries}
    if observed != allowed or any(not path.is_file() for path in entries):
        raise RevisionDetectionError(
            "Committed detector checkpoint file set differs for task {}.".format(
                task.ordinal
            )
        )
    manifest = _read_json(manifest_path, "detector fit checkpoint manifest")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RevisionDetectionError("Detector fit checkpoint schema differs.")
    _verify_signed_document(manifest, "manifest_sha256", "Detector fit checkpoint")
    expected_manifest_keys = {
        "schema_version",
        "run_identity_sha256",
        "plan_sha256",
        "task_identity",
        "task_identity_sha256",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "children",
        "children_sha256",
        "manifest_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys
        or
        manifest.get("run_identity_sha256") != run_identity_sha256
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("task_identity") != dict(task.identity)
        or manifest.get("task_identity_sha256") != task.identity_sha256
    ):
        raise RevisionDetectionError(
            "Detector fit checkpoint lineage/task identity differs for task {}.".format(
                task.ordinal
            )
        )
    children = manifest.get("children")
    if (
        not isinstance(children, dict)
        or set(children) != {
        "metric.json",
        "predictions.json",
        }
        or manifest.get("children_sha256") != canonical_json_sha256(children)
    ):
        raise RevisionDetectionError("Detector fit checkpoint child manifest differs.")
    payloads: Dict[str, dict] = {}
    for name in ("metric.json", "predictions.json"):
        path = directory / name
        identity = children[name]
        if not isinstance(identity, dict) or set(identity) != {
            "sha256",
            "size_bytes",
            "row_count",
        } or (
            identity.get("sha256") != file_sha256(path)
            or int(identity.get("size_bytes", -1)) != int(path.stat().st_size)
        ):
            raise RevisionDetectionError(
                "Detector fit checkpoint child hash/size differs: {}".format(path)
            )
        payloads[name] = _read_json(path, "detector fit checkpoint child")
    metrics = _payload_rows(payloads["metric.json"], "detector metric checkpoint")
    predictions = _payload_rows(
        payloads["predictions.json"], "detector prediction checkpoint"
    )
    if (
        int(children["metric.json"].get("row_count", -1)) != len(metrics)
        or int(children["predictions.json"].get("row_count", -1))
        != len(predictions)
    ):
        raise RevisionDetectionError("Detector fit checkpoint child row counts differ.")
    _validate_checkpoint_semantics(prepared, task, metrics, predictions)
    elapsed = float(manifest.get("elapsed_seconds", -1.0))
    try:
        started_at = datetime.fromisoformat(str(manifest["started_at_utc"]))
        completed_at = datetime.fromisoformat(str(manifest["completed_at_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RevisionDetectionError(
            "Detector fit checkpoint timestamps are invalid."
        ) from exc
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or started_at.tzinfo is None
        or completed_at.tzinfo is None
        or completed_at < started_at
    ):
        raise RevisionDetectionError("Detector fit checkpoint elapsed time is invalid.")
    return metrics[0], predictions, elapsed, {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "task_ordinal": int(task.ordinal),
    }


def _write_fit_checkpoint(
    context: DetectorExecutionContext,
    task: DetectorFitTask,
    metric: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    run_identity_sha256: str,
    plan_sha256: str,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
) -> dict:
    directory = _ensure_real_directory(_task_directory(context.checkpoint_dir, task))
    metric_path = directory / "metric.json"
    predictions_path = directory / "predictions.json"
    manifest_path = directory / "manifest.json"
    atomic_write_json(metric_path, _row_payload([metric]))
    atomic_write_json(predictions_path, _row_payload(predictions))
    children = {
        "metric.json": {
            "sha256": file_sha256(metric_path),
            "size_bytes": int(metric_path.stat().st_size),
            "row_count": 1,
        },
        "predictions.json": {
            "sha256": file_sha256(predictions_path),
            "size_bytes": int(predictions_path.stat().st_size),
            "row_count": int(len(predictions)),
        },
    }
    manifest = _signed_document(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "run_identity_sha256": run_identity_sha256,
            "plan_sha256": plan_sha256,
            "task_identity": dict(task.identity),
            "task_identity_sha256": task.identity_sha256,
            "started_at_utc": _utc_text(started_at),
            "completed_at_utc": _utc_text(completed_at),
            "elapsed_seconds": float(elapsed_seconds),
            "children": children,
            "children_sha256": canonical_json_sha256(children),
        },
        "manifest_sha256",
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "task_ordinal": int(task.ordinal),
    }


class _StatusTracker:
    def __init__(
        self,
        context: DetectorExecutionContext,
        run_identity: Mapping[str, object],
        run_identity_sha256: str,
        total: int,
        recovered_errors: List[dict],
        started_at: datetime,
        started_monotonic: float,
        prior_gpu_intervals: Optional[Sequence[Mapping[str, object]]] = None,
        checkpoint_fit_seconds_at_process_start: float = 0.0,
    ) -> None:
        self.context = context
        self.run_identity = dict(run_identity)
        self.run_identity_sha256 = run_identity_sha256
        self.total = int(total)
        self.recovered_errors = recovered_errors
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.completed = 0
        self.current: Optional[DetectorFitTask] = None
        self.current_started_at: Optional[datetime] = None
        self.current_started_monotonic: Optional[float] = None
        self.next_fit: Optional[DetectorFitTask] = None
        self.next_fit_upper_seconds: Optional[float] = None
        self.fit_gate_nonce: Optional[str] = None
        self.last_consumed_fit_permit: Optional[dict] = None
        self.durations: List[float] = []
        self.last_checkpoint: Optional[dict] = None
        self.state = "initializing"
        self.invocation_completed_at: Optional[datetime] = None
        self.prior_gpu_intervals = [
            dict(value) for value in (prior_gpu_intervals or [])
        ]
        self.checkpoint_fit_seconds_at_process_start = float(
            checkpoint_fit_seconds_at_process_start
        )
        self.checkpoint_cumulative_fit_seconds = float(
            checkpoint_fit_seconds_at_process_start
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_heartbeat(self) -> None:
        interval = max(1.0, float(self.context.heartbeat_seconds))

        def heartbeat() -> None:
            while not self._stop.wait(interval):
                try:
                    self.write()
                except Exception:
                    # The foreground write at every boundary remains authoritative.
                    pass

        self._thread = threading.Thread(
            target=heartbeat, name="detector-status-heartbeat", daemon=True
        )
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _snapshot(self) -> dict:
        now = _utc_now()
        now_monotonic = time.monotonic()
        process_elapsed = max(0.0, now_monotonic - self.started_monotonic)
        current_elapsed = (
            None
            if self.current_started_monotonic is None
            else max(0.0, now_monotonic - self.current_started_monotonic)
        )
        rolling_values = list(self.durations[-8:])
        if not rolling_values and self.context.benchmark_seconds_per_fit:
            rolling_values = [float(self.context.benchmark_seconds_per_fit)]
        rolling_mean = (
            None if not rolling_values else float(sum(rolling_values) / len(rolling_values))
        )
        remaining = max(0, self.total - self.completed)
        eta = None if rolling_mean is None else float(remaining * rolling_mean)
        completion = None if eta is None else _utc_text(now + timedelta(seconds=eta))
        cumulative_fit_elapsed = float(self.checkpoint_cumulative_fit_seconds)
        if current_elapsed is not None:
            cumulative_fit_elapsed += float(current_elapsed)
        current = None
        if self.current is not None:
            current = {
                "ordinal": int(self.current.ordinal),
                "index": int(self.current.ordinal),
                "fit_number": int(self.current.ordinal + 1),
                "split_id": self.current.split.split_id,
                "regime": self.current.split.regime,
                "detector_name": self.current.detector_name,
                "detector_kind": self.current.detector_kind,
                "seed": int(self.current.seed),
                "task_identity_sha256": self.current.identity_sha256,
            }
        next_fit = None
        if self.next_fit is not None:
            next_fit = {
                "ordinal": int(self.next_fit.ordinal),
                "index": int(self.next_fit.ordinal),
                "fit_number": int(self.next_fit.ordinal + 1),
                "split_id": self.next_fit.split.split_id,
                "regime": self.next_fit.split.regime,
                "detector_name": self.next_fit.detector_name,
                "detector_kind": self.next_fit.detector_kind,
                "seed": int(self.next_fit.seed),
                "task_identity_sha256": self.next_fit.identity_sha256,
            }
        gpu_accounting = None
        if self.context.device == "cuda:0":
            interval_end = self.invocation_completed_at or now
            current_interval = {
                "pid": int(os.getpid()),
                "process_start_ticks": _process_start_ticks(),
                "device": "cuda:0",
                "gpu_uuid": self.context.gpu_uuid,
                "started_at_utc": _utc_text(self.started_at),
                "completed_at_utc": (
                    None
                    if self.invocation_completed_at is None
                    else _utc_text(self.invocation_completed_at)
                ),
                "last_observed_at_utc": _utc_text(interval_end),
                "elapsed_seconds": float(
                    (interval_end - self.started_at).total_seconds()
                ),
                "derivation_policy": GPU_ACCOUNTING_POLICY,
            }
            intervals = list(self.prior_gpu_intervals) + [current_interval]
            gpu_accounting = {
                "device": "cuda:0",
                "gpu_uuid": self.context.gpu_uuid,
                "intervals": intervals,
                "cumulative_elapsed_seconds": float(
                    sum(float(value["elapsed_seconds"]) for value in intervals)
                ),
                "derivation_policy": GPU_ACCOUNTING_COLLECTION_POLICY,
            }
        value = {
            "schema_version": STATUS_SCHEMA,
            "updated_at_utc": _utc_text(now),
            "state": self.state,
            "completed_fit_count": int(self.completed),
            "total_fit_count": int(self.total),
            "current_fit": current,
            "next_fit": next_fit,
            "next_fit_upper_seconds": self.next_fit_upper_seconds,
            "fit_gate_nonce": self.fit_gate_nonce,
            "fit_permit_file": (
                None
                if self.context.fit_permit_file is None
                else str(Path(self.context.fit_permit_file).resolve())
            ),
            "fit_permit_receipt_dir": (
                None
                if self.context.fit_permit_receipt_dir is None
                else str(Path(self.context.fit_permit_receipt_dir).resolve())
            ),
            "last_consumed_fit_permit": self.last_consumed_fit_permit,
            "global_started_at_utc": _utc_text(self.started_at),
            "global_elapsed_seconds": cumulative_fit_elapsed,
            "global_elapsed_policy": "sum_of_valid_fit_intervals_plus_active_fit_v1",
            "process_elapsed_seconds": float(process_elapsed),
            "checkpoint_fit_seconds_at_process_start": float(
                self.checkpoint_fit_seconds_at_process_start
            ),
            "checkpoint_cumulative_fit_seconds": float(
                self.checkpoint_cumulative_fit_seconds
            ),
            "current_fit_started_at_utc": (
                None
                if self.current_started_at is None
                else _utc_text(self.current_started_at)
            ),
            "current_fit_elapsed_seconds": current_elapsed,
            "fits_per_hour": (
                0.0
                if cumulative_fit_elapsed <= 0.0
                else float(3600.0 * self.completed / cumulative_fit_elapsed)
            ),
            "rolling_fits_per_hour": (
                None
                if rolling_mean is None or rolling_mean <= 0.0
                else float(3600.0 / rolling_mean)
            ),
            "rolling_eta_seconds": eta,
            "rolling_estimated_completion_utc": completion,
            "last_completed_checkpoint": self.last_checkpoint,
            "recovered_errors": list(self.recovered_errors),
            "device": self.context.device,
            "gpu_uuid": self.context.gpu_uuid,
            "workers": int(self.context.workers),
            "peak_rss_bytes": int(_peak_rss_bytes()),
            "peak_vram_bytes": int(_peak_vram_bytes(self.context.device)),
            "pid": int(os.getpid()),
            "process_start_ticks": _process_start_ticks(),
            "run_identity": self.run_identity,
            "run_identity_sha256": self.run_identity_sha256,
            "gpu_accounting": gpu_accounting,
        }
        return _signed_document(value, "status_sha256")

    def write(self) -> None:
        with self._lock:
            atomic_write_json(self.context.status_file, self._snapshot())

    def snapshot(self) -> dict:
        """Return one lock-consistent signed status snapshot."""

        with self._lock:
            return self._snapshot()

    def transition(self, **values: object) -> None:
        """Atomically update a multi-field state transition and publish it."""

        with self._lock:
            for name, value in values.items():
                if not hasattr(self, name):
                    raise RevisionDetectionError(
                        "Unknown detector status transition field: {}".format(name)
                    )
                setattr(self, name, value)
            atomic_write_json(self.context.status_file, self._snapshot())


def _validate_existing_status(
    context: DetectorExecutionContext,
    run_identity: Mapping[str, object],
    run_identity_sha256: str,
    recovered_errors: List[dict],
) -> List[dict]:
    path = Path(context.status_file)
    if not path.exists():
        fits_root = Path(context.checkpoint_dir) / "fits"
        if (
            context.device == "cuda:0"
            and fits_root.is_dir()
            and any(fits_root.iterdir())
        ):
            raise RevisionDetectionError(
                "CUDA detector status is missing while fit checkpoint state exists; "
                "prior GPU charge cannot be reconstructed safely."
            )
        return []
    try:
        value = _read_json(path, "detector status")
        if value.get("schema_version") != STATUS_SCHEMA:
            raise RevisionDetectionError("Detector status schema differs.")
        _verify_signed_document(value, "status_sha256", "Detector status")
        if (
            value.get("run_identity") != dict(run_identity)
            or value.get("run_identity_sha256") != run_identity_sha256
        ):
            raise RevisionDetectionError("Detector status run identity differs.")
        if context.device != "cuda:0":
            return []
        accounting = value.get("gpu_accounting")
        if not isinstance(accounting, dict) or (
            accounting.get("device") != "cuda:0"
            or accounting.get("gpu_uuid") != context.gpu_uuid
            or accounting.get("derivation_policy")
            != GPU_ACCOUNTING_COLLECTION_POLICY
            or not isinstance(accounting.get("intervals"), list)
        ):
            raise RevisionDetectionError("Detector status GPU accounting differs.")
        intervals: List[dict] = []
        previous_end: Optional[datetime] = None
        for raw in accounting["intervals"]:
            if not isinstance(raw, dict):
                raise RevisionDetectionError("Detector GPU interval is malformed.")
            interval = dict(raw)
            if (
                interval.get("device") != "cuda:0"
                or interval.get("gpu_uuid") != context.gpu_uuid
                or interval.get("derivation_policy") != GPU_ACCOUNTING_POLICY
            ):
                raise RevisionDetectionError("Detector GPU interval identity differs.")
            try:
                start = datetime.fromisoformat(str(interval["started_at_utc"]))
                end_text = interval.get("completed_at_utc") or interval.get(
                    "last_observed_at_utc"
                )
                end = datetime.fromisoformat(str(end_text))
            except (KeyError, TypeError, ValueError) as exc:
                raise RevisionDetectionError(
                    "Detector GPU interval timestamps are invalid."
                ) from exc
            if (
                start.tzinfo is None
                or end.tzinfo is None
                or end < start
                or (previous_end is not None and start < previous_end)
            ):
                raise RevisionDetectionError(
                    "Detector GPU intervals are not ordered and nonoverlapping."
                )
            elapsed = float((end - start).total_seconds())
            if abs(float(interval.get("elapsed_seconds", -1.0)) - elapsed) > 1e-6:
                raise RevisionDetectionError("Detector GPU interval elapsed time differs.")
            interval["completed_at_utc"] = _utc_text(end)
            interval["last_observed_at_utc"] = _utc_text(end)
            interval["elapsed_seconds"] = elapsed
            intervals.append(interval)
            previous_end = end
        return intervals
    except RevisionDetectionError as exc:
        if context.device == "cuda:0":
            raise RevisionDetectionError(
                "CUDA detector status/accounting is invalid; refusing to discard "
                "prior GPU charge: {}".format(exc)
            ) from exc
        recovered_errors.append(
            {
                "type": "stale_or_invalid_status_replaced",
                "path": str(path.resolve()),
                "error": str(exc),
                "action": "status_rebuilt_from_valid_fit_checkpoints",
            }
        )
        return []


FitRunner = Callable[
    [PreparedDetectorSuite, DetectorSplit, Mapping[str, object]],
    Tuple[dict, List[dict]],
]


def compare_detector_fit_outputs(
    reference_metric: Mapping[str, object],
    reference_predictions: Sequence[Mapping[str, object]],
    candidate_metric: Mapping[str, object],
    candidate_predictions: Sequence[Mapping[str, object]],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> dict:
    """Compare a direct/reference fit with a checkpoint-wrapper candidate.

    Identity, row order, labels, and thresholded predictions must match
    exactly. Score and floating metric fields use explicit tolerances so the
    same API can record deterministic same-device equality and measured
    CPU/GPU numerical agreement without weakening checkpoint validation.
    """

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise RevisionDetectionError("Equivalence tolerances must be nonnegative.")
    reference_rows = [dict(row) for row in reference_predictions]
    candidate_rows = [dict(row) for row in candidate_predictions]
    exact_prediction_fields = (
        "split_id",
        "regime",
        "held_out_value",
        "detector_name",
        "requested_kind",
        "implementation_kind",
        "implementation_status",
        "row_id",
        "payload_group_id",
        "prompt_template_id",
        "model_id",
        "codec_id",
        "label",
        "prediction",
    )
    exact_mismatches: List[str] = []
    if len(reference_rows) != len(candidate_rows):
        exact_mismatches.append("prediction_row_count")
    for index, (reference, candidate) in enumerate(
        zip(reference_rows, candidate_rows)
    ):
        for field_name in exact_prediction_fields:
            if reference.get(field_name) != candidate.get(field_name):
                exact_mismatches.append(
                    "prediction[{}].{}".format(index, field_name)
                )
    reference_scores = np.asarray(
        [float(row.get("score", float("nan"))) for row in reference_rows],
        dtype=np.float64,
    )
    candidate_scores = np.asarray(
        [float(row.get("score", float("nan"))) for row in candidate_rows],
        dtype=np.float64,
    )
    score_equivalent = bool(
        reference_scores.shape == candidate_scores.shape
        and np.isfinite(reference_scores).all()
        and np.isfinite(candidate_scores).all()
        and np.allclose(
            reference_scores,
            candidate_scores,
            atol=float(absolute_tolerance),
            rtol=float(relative_tolerance),
        )
    )
    max_score_difference = (
        None
        if reference_scores.shape != candidate_scores.shape or not len(reference_scores)
        else float(np.max(np.abs(reference_scores - candidate_scores)))
    )
    ignored_metric_fields = {
        "implementation_metadata_json",
        "model_state_sha256",
        "model_artifact_set_sha256",
    }
    metric_exact_mismatches: List[str] = []
    metric_numeric_differences: Dict[str, float] = {}
    metric_numeric_equivalent = True
    metric_keys = set(reference_metric) | set(candidate_metric)
    for field_name in sorted(metric_keys - ignored_metric_fields):
        reference = reference_metric.get(field_name)
        candidate = candidate_metric.get(field_name)
        if isinstance(reference, (int, float, np.number)) and isinstance(
            candidate, (int, float, np.number)
        ):
            reference_float = float(reference)
            candidate_float = float(candidate)
            if math.isnan(reference_float) and math.isnan(candidate_float):
                difference = 0.0
                equivalent = True
            else:
                difference = abs(reference_float - candidate_float)
                equivalent = math.isclose(
                    reference_float,
                    candidate_float,
                    abs_tol=float(absolute_tolerance),
                    rel_tol=float(relative_tolerance),
                )
            metric_numeric_differences[field_name] = difference
            metric_numeric_equivalent = metric_numeric_equivalent and equivalent
        elif reference != candidate:
            metric_exact_mismatches.append(field_name)
    equivalent = bool(
        not exact_mismatches
        and not metric_exact_mismatches
        and score_equivalent
        and metric_numeric_equivalent
    )
    return {
        "schema_version": "rankcloak-revision-detector-fit-equivalence-v1",
        "equivalent": equivalent,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "prediction_row_count": int(len(candidate_rows)),
        "exact_mismatches": exact_mismatches,
        "metric_exact_mismatches": metric_exact_mismatches,
        "score_equivalent": score_equivalent,
        "max_score_absolute_difference": max_score_difference,
        "metric_numeric_equivalent": metric_numeric_equivalent,
        "metric_numeric_absolute_differences": metric_numeric_differences,
        "ignored_device_specific_metric_fields": sorted(ignored_metric_fields),
    }


def _implementation_metadata(metric: Mapping[str, object]) -> dict:
    try:
        value = json.loads(str(metric.get("implementation_metadata_json", "{}")))
    except json.JSONDecodeError as exc:
        raise RevisionDetectionError(
            "Detector equivalence metric metadata is invalid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise RevisionDetectionError("Detector equivalence metadata is not an object.")
    return value


def evaluate_detector_device_equivalence(
    cpu_metric: Mapping[str, object],
    cpu_predictions: Sequence[Mapping[str, object]],
    cuda_metric: Mapping[str, object],
    cuda_predictions: Sequence[Mapping[str, object]],
    cuda_repeat_metric: Mapping[str, object],
    cuda_repeat_predictions: Sequence[Mapping[str, object]],
    equivalence_policy: Mapping[str, object],
) -> dict:
    """Apply the predeclared same-CUDA and CPU/CUDA neural-fit criteria."""

    same_policy = equivalence_policy.get("same_device_cuda")
    cross_policy = equivalence_policy.get("cpu_cuda")
    if not isinstance(same_policy, Mapping) or not isinstance(
        cross_policy, Mapping
    ):
        raise RevisionDetectionError("Detector equivalence policy is malformed.")
    cuda_rows = [dict(row) for row in cuda_predictions]
    repeat_rows = [dict(row) for row in cuda_repeat_predictions]
    cpu_rows = [dict(row) for row in cpu_predictions]
    cuda_same_scores = np.asarray(
        [float(row["score"]) for row in cuda_rows], dtype=np.float64
    )
    repeat_scores = np.asarray(
        [float(row["score"]) for row in repeat_rows], dtype=np.float64
    )
    cuda_metadata = _implementation_metadata(cuda_metric)
    repeat_metadata = _implementation_metadata(cuda_repeat_metric)
    same_checks = {
        "task_design_exact": dict(cuda_metric) == dict(cuda_repeat_metric),
        "row_identity_order_labels_exact": [
            (row.get("row_id"), row.get("label")) for row in cuda_rows
        ]
        == [(row.get("row_id"), row.get("label")) for row in repeat_rows],
        "model_state_sha256_exact": cuda_metadata.get("model_state_sha256")
        == repeat_metadata.get("model_state_sha256"),
        "scores_exact": bool(
            cuda_same_scores.shape == repeat_scores.shape
            and np.array_equal(cuda_same_scores, repeat_scores)
        ),
        "metrics_exact": dict(cuda_metric) == dict(cuda_repeat_metric),
        "predictions_exact": cuda_rows == repeat_rows,
    }
    same_pass = all(
        same_checks[name] is bool(required)
        for name, required in same_policy.items()
    )

    exact_prediction_fields = (
        "split_id",
        "regime",
        "held_out_value",
        "detector_name",
        "requested_kind",
        "implementation_kind",
        "implementation_status",
        "row_id",
        "payload_group_id",
        "prompt_template_id",
        "model_id",
        "codec_id",
        "label",
    )
    row_identity_exact = bool(
        len(cpu_rows) == len(cuda_rows)
        and all(
            all(cpu.get(name) == cuda.get(name) for name in exact_prediction_fields)
            for cpu, cuda in zip(cpu_rows, cuda_rows)
        )
    )
    exact_metric_fields = (
        "split_id",
        "regime",
        "held_out_column",
        "held_out_value",
        "detector_name",
        "requested_kind",
        "implementation_kind",
        "implementation_status",
        "train_rows",
        "test_rows",
        "train_payload_groups",
        "purged_train_rows",
        "decision_threshold",
        "seed",
        "bootstrap_unit",
        "bootstrap_resamples_requested",
        "test_payload_groups",
    )
    task_design_exact = all(
        cpu_metric.get(name) == cuda_metric.get(name) for name in exact_metric_fields
    )
    cpu_scores = np.asarray(
        [float(row["score"]) for row in cpu_rows], dtype=np.float64
    )
    cuda_scores = np.asarray(
        [float(row["score"]) for row in cuda_rows], dtype=np.float64
    )
    if (
        cpu_scores.shape != cuda_scores.shape
        or not np.isfinite(cpu_scores).all()
        or not np.isfinite(cuda_scores).all()
        or not len(cpu_scores)
    ):
        score_mae = float("inf")
        score_max_abs = float("inf")
        score_pearson = float("-inf")
    else:
        differences = np.abs(cpu_scores - cuda_scores)
        score_mae = float(np.mean(differences))
        score_max_abs = float(np.max(differences))
        if np.std(cpu_scores) == 0.0 or np.std(cuda_scores) == 0.0:
            score_pearson = 1.0 if np.array_equal(cpu_scores, cuda_scores) else -1.0
        else:
            score_pearson = float(np.corrcoef(cpu_scores, cuda_scores)[0, 1])
    prediction_agreement = (
        0.0
        if len(cpu_rows) != len(cuda_rows) or not cpu_rows
        else float(
            np.mean(
                np.asarray([row["prediction"] for row in cpu_rows], dtype=int)
                == np.asarray([row["prediction"] for row in cuda_rows], dtype=int)
            )
        )
    )
    metric_names = (
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    )
    metric_fields = [
        field_name
        for name in metric_names
        for field_name in (name, name + "_ci_low_95", name + "_ci_high_95")
    ]
    metric_differences: Dict[str, float] = {}
    for field_name in metric_fields:
        cpu_value = cpu_metric.get(field_name)
        cuda_value = cuda_metric.get(field_name)
        if cpu_value is None and cuda_value is None:
            metric_differences[field_name] = 0.0
        elif cpu_value is None or cuda_value is None:
            metric_differences[field_name] = float("inf")
        else:
            metric_differences[field_name] = abs(
                float(cpu_value) - float(cuda_value)
            )
    metric_max_abs = max(metric_differences.values(), default=0.0)
    cpu_metadata = _implementation_metadata(cpu_metric)
    tensor_schema_exact = (
        cpu_metadata.get("model_state_schema_hash_algorithm")
        == cuda_metadata.get("model_state_schema_hash_algorithm")
        and cpu_metadata.get("model_state_schema_sha256")
        == cuda_metadata.get("model_state_schema_sha256")
        and bool(cpu_metadata.get("model_state_schema_sha256"))
    )
    cross_measurements = {
        "task_design_exact": task_design_exact,
        "row_identity_order_labels_exact": row_identity_exact,
        "score_mae": score_mae,
        "score_max_abs": score_max_abs,
        "score_pearson": score_pearson,
        "prediction_agreement": prediction_agreement,
        "metric_max_abs": metric_max_abs,
        "metric_absolute_differences": metric_differences,
        "model_state_tensor_schema_exact": tensor_schema_exact,
        "cpu_model_state_sha256": cpu_metadata.get("model_state_sha256"),
        "cuda_model_state_sha256": cuda_metadata.get("model_state_sha256"),
        "model_state_hash_policy": "device_specific",
    }
    cross_checks = {
        "task_design_exact": task_design_exact is bool(
            cross_policy["task_design_exact"]
        ),
        "row_identity_order_labels_exact": row_identity_exact is bool(
            cross_policy["row_identity_order_labels_exact"]
        ),
        "score_mae_max": score_mae <= float(cross_policy["score_mae_max"]),
        "score_max_abs_max": score_max_abs
        <= float(cross_policy["score_max_abs_max"]),
        "score_pearson_min": score_pearson
        >= float(cross_policy["score_pearson_min"]),
        "prediction_agreement_min": prediction_agreement
        >= float(cross_policy["prediction_agreement_min"]),
        "metric_max_abs_max": metric_max_abs
        <= float(cross_policy["metric_max_abs_max"]),
        "model_state_tensor_schema_exact": tensor_schema_exact is bool(
            cross_policy["model_state_tensor_schema_exact"]
        ),
        "model_state_hash_policy": cross_policy["model_state_hash_policy"]
        == "device_specific",
    }
    cross_pass = all(cross_checks.values())
    return {
        "schema_version": EQUIVALENCE_REPORT_SCHEMA,
        "equivalent": bool(same_pass and cross_pass),
        "same_device_cuda": {
            "passed": same_pass,
            "predeclared_policy": dict(same_policy),
            "checks": same_checks,
        },
        "cpu_cuda": {
            "passed": cross_pass,
            "predeclared_policy": dict(cross_policy),
            "checks": cross_checks,
            "measurements": cross_measurements,
        },
    }


def write_detector_equivalence_fit_artifact(
    output_path: Path,
    *,
    role: str,
    task_identity: Mapping[str, object],
    metric: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict:
    """Self-finalize one CPU reference; CUDA publication belongs to supervisor."""

    if role not in {"cpu", "cuda", "cuda_repeat"}:
        raise RevisionDetectionError("Detector equivalence role is invalid.")
    if role != "cpu":
        raise RevisionDetectionError(
            "CUDA equivalence artifacts require supervisor terminal finalization."
        )
    ordinal = int(task_identity.get("ordinal", -1))
    if ordinal not in {0, 1}:
        raise RevisionDetectionError(
            "Production detector equivalence is restricted to frozen tasks 0 and 1."
        )
    provenance_value = dict(provenance)
    gpu_accounting = provenance_value.get("gpu_accounting")
    if role == "cpu":
        if provenance_value.get("device") != "cpu" or gpu_accounting is not None:
            raise RevisionDetectionError(
                "CPU equivalence artifact must declare zero GPU accounting."
            )
    value = {
        "schema_version": EQUIVALENCE_ARTIFACT_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "role": role,
        "task_index": ordinal,
        "task_identity": dict(task_identity),
        "task_identity_sha256": canonical_json_sha256(task_identity),
        "metric": dict(metric),
        "predictions": [dict(row) for row in predictions],
        "prediction_count": int(len(predictions)),
        "provenance": provenance_value,
        "provenance_sha256": canonical_json_sha256(provenance_value),
    }
    signed = _signed_document(value, "artifact_sha256")
    atomic_write_json(Path(output_path), signed)
    return signed


def read_detector_equivalence_fit_artifact(path: Path) -> dict:
    artifact_path = Path(path)
    value = _read_json(artifact_path, "detector equivalence fit artifact")
    if value.get("schema_version") != EQUIVALENCE_ARTIFACT_SCHEMA:
        raise RevisionDetectionError("Detector equivalence artifact schema differs.")
    _verify_signed_document(
        value, "artifact_sha256", "Detector equivalence fit artifact"
    )
    if (
        value.get("role") not in {"cpu", "cuda", "cuda_repeat"}
        or int(value.get("task_index", -1)) not in {0, 1}
        or not isinstance(value.get("task_identity"), dict)
        or int(value["task_identity"].get("ordinal", -1))
        != int(value.get("task_index", -2))
        or value.get("task_identity_sha256")
        != canonical_json_sha256(value["task_identity"])
        or not isinstance(value.get("metric"), dict)
        or not isinstance(value.get("predictions"), list)
        or int(value.get("prediction_count", -1)) != len(value["predictions"])
        or not isinstance(value.get("provenance"), dict)
        or value.get("provenance_sha256")
        != canonical_json_sha256(value["provenance"])
    ):
        raise RevisionDetectionError("Detector equivalence artifact is malformed.")
    role = str(value["role"])
    provenance = value["provenance"]
    accounting = provenance.get("gpu_accounting")
    if role == "cpu":
        if provenance.get("device") != "cpu" or accounting is not None:
            raise RevisionDetectionError(
                "CPU detector equivalence provenance is not zero-GPU."
            )
    elif (
        provenance.get("device") != "cuda:0"
        or not str(provenance.get("gpu_uuid", "")).startswith("GPU-")
        or not isinstance(accounting, dict)
        or accounting.get("device") != "cuda:0"
        or accounting.get("gpu_uuid") != provenance.get("gpu_uuid")
        or accounting.get("derivation_policy")
        != GPU_ACCOUNTING_COLLECTION_POLICY
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
    ):
        raise RevisionDetectionError(
            "CUDA detector equivalence provenance lacks GPU accounting."
        )
    if role != "cpu":
        candidate_identity = value.get("finalization_candidate")
        if not isinstance(candidate_identity, dict) or set(candidate_identity) != {
            "path",
            "sha256",
            "size_bytes",
            "candidate_sha256",
        }:
            raise RevisionDetectionError(
                "CUDA equivalence artifact lacks finalization identity."
            )
        candidate_path = Path(str(candidate_identity["path"]))
        if (
            candidate_path.is_symlink()
            or not candidate_path.is_file()
            or file_sha256(candidate_path) != candidate_identity["sha256"]
            or int(candidate_path.stat().st_size)
            != int(candidate_identity["size_bytes"])
        ):
            raise RevisionDetectionError(
                "CUDA equivalence finalization candidate bytes differ."
            )
        candidate = read_detector_finalization_candidate(candidate_path)
        if (
            candidate.get("kind") != "equivalence_artifact"
            or candidate.get("candidate_sha256")
            != candidate_identity["candidate_sha256"]
            or Path(candidate["requested_output_path"]).resolve()
            != artifact_path.resolve()
        ):
            raise RevisionDetectionError(
                "CUDA equivalence finalization candidate identity differs."
            )
        receipt_path = candidate_path.with_name(
            candidate_path.stem + ".terminal_receipt.json"
        )
        receipt = _validate_terminal_receipt(receipt_path)
        if (
            receipt.get("candidate", {}).get("candidate_sha256")
            != candidate["candidate_sha256"]
            or receipt.get("published_output", {}).get("sha256")
            != file_sha256(artifact_path)
            or receipt.get("gpu_accounting") != accounting
            or value.get("terminal_accounting_status_sha256")
            != receipt.get("closed_status_sha256")
            or provenance.get("terminal_accounting_status_sha256")
            != receipt.get("closed_status_sha256")
        ):
            raise RevisionDetectionError(
                "CUDA equivalence terminal receipt identity differs."
            )
    return value


def build_validated_checkpoint_equivalence_payload(
    prepared: PreparedDetectorSuite,
    context: DetectorExecutionContext,
    *,
    task_index: int,
    role: str,
    provenance: Mapping[str, object],
) -> dict:
    """Build one identity/hash/semantic-valid fit payload for finalization."""

    tasks = build_fit_tasks(prepared)
    if int(task_index) not in {0, 1} or int(task_index) >= len(tasks):
        raise RevisionDetectionError(
            "Production equivalence task index must be frozen task 0 or 1."
        )
    task = tasks[int(task_index)]
    plan = _build_plan(prepared, tasks)
    plan_sha256 = canonical_json_sha256(plan)
    run_identity, run_identity_sha256 = _build_run_identity(context, plan)
    recovered_errors: List[dict] = []
    loaded = _load_valid_checkpoint(
        prepared,
        context,
        task,
        run_identity_sha256,
        plan_sha256,
        recovered_errors,
    )
    if loaded is None or recovered_errors:
        raise RevisionDetectionError(
            "Detector equivalence checkpoint is absent or required recovery."
        )
    metric, predictions, elapsed_seconds, checkpoint = loaded
    provenance_value = {
        **dict(provenance),
        "task_index": int(task_index),
        "fit_elapsed_seconds": float(elapsed_seconds),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "execution_plan_sha256": plan_sha256,
        "checkpoint": checkpoint,
    }
    return {
        "schema_version": EQUIVALENCE_ARTIFACT_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "role": role,
        "task_index": int(task.ordinal),
        "task_identity": dict(task.identity),
        "task_identity_sha256": canonical_json_sha256(task.identity),
        "metric": dict(metric),
        "predictions": [dict(row) for row in predictions],
        "prediction_count": int(len(predictions)),
        "provenance": provenance_value,
        "provenance_sha256": canonical_json_sha256(provenance_value),
    }


def export_validated_checkpoint_equivalence_artifact(
    prepared: PreparedDetectorSuite,
    context: DetectorExecutionContext,
    *,
    task_index: int,
    role: str,
    output_path: Path,
    provenance: Mapping[str, object],
) -> dict:
    """Export one validated CPU fit; CUDA publication belongs to supervisor."""

    payload = build_validated_checkpoint_equivalence_payload(
        prepared,
        context,
        task_index=task_index,
        role=role,
        provenance=provenance,
    )
    if role != "cpu":
        raise RevisionDetectionError(
            "CUDA equivalence artifacts require supervisor terminal finalization."
        )
    signed = _signed_document(payload, "artifact_sha256")
    # Reuse the strict artifact validator's zero-GPU contract before publication.
    write_detector_equivalence_fit_artifact(
        output_path,
        role=role,
        task_identity=payload["task_identity"],
        metric=payload["metric"],
        predictions=payload["predictions"],
        provenance=payload["provenance"],
    )
    return read_detector_equivalence_fit_artifact(output_path)


def write_detector_device_equivalence_report(
    output_path: Path,
    *,
    cpu_artifact_path: Path,
    cuda_artifact_path: Path,
    cuda_repeat_artifact_path: Path,
    equivalence_policy: Mapping[str, object],
    policy_identity: Mapping[str, object],
) -> dict:
    """Verify three signed fit artifacts and persist a signed policy decision."""

    cpu = read_detector_equivalence_fit_artifact(cpu_artifact_path)
    cuda = read_detector_equivalence_fit_artifact(cuda_artifact_path)
    repeat = read_detector_equivalence_fit_artifact(cuda_repeat_artifact_path)
    if (cpu["role"], cuda["role"], repeat["role"]) != (
        "cpu",
        "cuda",
        "cuda_repeat",
    ):
        raise RevisionDetectionError("Detector equivalence artifact roles differ.")
    scientific_fields = (
        "ordinal",
        "split_position",
        "detector_position",
        "split_id",
        "regime",
        "held_out_column",
        "held_out_value",
        "partition_policy",
        "purged_train_rows",
        "excluded_held_out_rows",
        "train_row_count",
        "test_row_count",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_row_ids_ordered_sha256",
        "test_row_ids_ordered_sha256",
        "detector_name",
        "detector_kind",
        "scientific_detector_config_sha256",
        "seed",
        "bootstrap_resamples",
        "decision_threshold",
    )
    identities = [value["task_identity"] for value in (cpu, cuda, repeat)]
    scientific_task_identity_exact = all(
        identity.get(field_name) == identities[0].get(field_name)
        for identity in identities[1:]
        for field_name in scientific_fields
    )
    if any(
        value["provenance"].get("policy_identity") != dict(policy_identity)
        for value in (cpu, cuda, repeat)
    ):
        raise RevisionDetectionError(
            "Detector equivalence artifact policy/environment identity differs."
        )
    decision = evaluate_detector_device_equivalence(
        cpu["metric"],
        cpu["predictions"],
        cuda["metric"],
        cuda["predictions"],
        repeat["metric"],
        repeat["predictions"],
        equivalence_policy,
    )
    decision["scientific_task_identity_exact"] = scientific_task_identity_exact
    decision["equivalent"] = bool(
        decision["equivalent"] and scientific_task_identity_exact
    )
    inputs = {
        role: {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(Path(path)),
            "size_bytes": int(Path(path).stat().st_size),
            "artifact_sha256": value["artifact_sha256"],
        }
        for role, path, value in (
            ("cpu", cpu_artifact_path, cpu),
            ("cuda", cuda_artifact_path, cuda),
            ("cuda_repeat", cuda_repeat_artifact_path, repeat),
        )
    }
    report = {
        "schema_version": EQUIVALENCE_REPORT_SCHEMA,
        "created_at_utc": _utc_text(_utc_now()),
        "predeclared_policy": dict(equivalence_policy),
        "policy_identity": dict(policy_identity),
        "policy_identity_sha256": canonical_json_sha256(policy_identity),
        "input_artifacts": inputs,
        "input_artifacts_sha256": canonical_json_sha256(inputs),
        "decision": decision,
    }
    signed = _signed_document(report, "report_sha256")
    atomic_write_json(Path(output_path), signed)
    return signed


def read_detector_device_equivalence_report(
    path: Path,
    *,
    expected_task_index: Optional[int] = None,
    expected_policy_identity: Optional[Mapping[str, object]] = None,
    expected_equivalence_policy: Optional[Mapping[str, object]] = None,
) -> dict:
    """Strictly verify a signed passing production equivalence decision."""

    value = _read_json(Path(path), "detector device equivalence report")
    if value.get("schema_version") != EQUIVALENCE_REPORT_SCHEMA:
        raise RevisionDetectionError("Detector equivalence report schema differs.")
    _verify_signed_document(value, "report_sha256", "Detector equivalence report")
    if (
        not isinstance(value.get("input_artifacts"), dict)
        or value.get("input_artifacts_sha256")
        != canonical_json_sha256(value["input_artifacts"])
        or not isinstance(value.get("policy_identity"), dict)
        or value.get("policy_identity_sha256")
        != canonical_json_sha256(value["policy_identity"])
        or not isinstance(value.get("decision"), dict)
        or value["decision"].get("equivalent") is not True
        or value["decision"].get("scientific_task_identity_exact") is not True
    ):
        raise RevisionDetectionError(
            "Detector equivalence report is malformed or did not pass."
        )
    if expected_policy_identity is not None and value["policy_identity"] != dict(
        expected_policy_identity
    ):
        raise RevisionDetectionError("Detector equivalence policy identity differs.")
    if expected_equivalence_policy is not None and value.get(
        "predeclared_policy"
    ) != dict(expected_equivalence_policy):
        raise RevisionDetectionError(
            "Detector equivalence thresholds differ from the predeclared policy."
        )
    observed_task_indices = set()
    for role in ("cpu", "cuda", "cuda_repeat"):
        identity = value["input_artifacts"].get(role)
        if not isinstance(identity, dict):
            raise RevisionDetectionError(
                "Detector equivalence report input identity is missing."
            )
        artifact_path = Path(str(identity.get("path", "")))
        if (
            artifact_path.is_symlink()
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != identity.get("sha256")
            or int(artifact_path.stat().st_size)
            != int(identity.get("size_bytes", -1))
        ):
            raise RevisionDetectionError(
                "Detector equivalence report input artifact bytes differ."
            )
        artifact = read_detector_equivalence_fit_artifact(artifact_path)
        if (
            artifact.get("role") != role
            or artifact.get("artifact_sha256")
            != identity.get("artifact_sha256")
        ):
            raise RevisionDetectionError(
                "Detector equivalence report input artifact identity differs."
            )
        observed_task_indices.add(int(artifact["task_index"]))
    if len(observed_task_indices) != 1 or (
        expected_task_index is not None
        and observed_task_indices != {int(expected_task_index)}
    ):
        raise RevisionDetectionError(
            "Detector equivalence report task index differs."
        )
    return value


def _execute_checkpointed_detector_suite_locked(
    prepared: PreparedDetectorSuite,
    context: DetectorExecutionContext,
    *,
    fit_runner: FitRunner = run_prepared_detector_fit,
    stop_after_new_fits: Optional[int] = None,
    benchmark_task_index: Optional[int] = None,
) -> DetectorExecutionOutcome:
    """Execute/resume fits and return an aggregate only after all validate."""

    _validate_runtime(context)
    context.output_dir = _ensure_real_directory(context.output_dir)
    context.checkpoint_dir = _ensure_real_directory(context.checkpoint_dir)
    _ensure_real_directory(context.checkpoint_dir / "fits")
    _ensure_real_directory(Path(context.status_file).parent)
    tasks = build_fit_tasks(prepared)
    if benchmark_task_index is not None and (
        int(benchmark_task_index) < 0 or int(benchmark_task_index) >= len(tasks)
    ):
        raise RevisionDetectionError(
            "--benchmark-task-index must be between 0 and {}.".format(
                max(0, len(tasks) - 1)
            )
        )
    plan = _build_plan(prepared, tasks)
    plan_sha256 = canonical_json_sha256(plan)
    run_identity, run_identity_sha256 = _build_run_identity(context, plan)
    plan_path = context.checkpoint_dir / "execution_plan.json"
    if plan_path.exists():
        existing_plan = _read_json(plan_path, "detector execution plan")
        if existing_plan != plan:
            raise RevisionDetectionError(
                "Detector checkpoint execution plan differs; refusing unsafe resume."
            )
    else:
        atomic_write_json(plan_path, plan)

    if not context.resume:
        committed = list((context.checkpoint_dir / "fits").glob("*/manifest.json"))
        if committed:
            raise RevisionDetectionError(
                "Detector checkpoints already exist; pass --resume to reuse them."
            )

    recovered_errors: List[dict] = []
    prior_gpu_intervals = _validate_existing_status(
        context, run_identity, run_identity_sha256, recovered_errors
    )
    preexisting_checkpoints: Dict[
        int, Tuple[dict, List[dict], float, dict]
    ] = {}
    for task in tasks:
        loaded = _load_valid_checkpoint(
            prepared,
            context,
            task,
            run_identity_sha256,
            plan_sha256,
            recovered_errors,
        )
        if loaded is not None:
            preexisting_checkpoints[task.ordinal] = loaded
    checkpoint_fit_seconds_at_process_start = float(
        sum(value[2] for value in preexisting_checkpoints.values())
    )
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    invocation_pid = int(os.getpid())
    invocation_start_ticks = int(_process_start_ticks() or -1)
    if invocation_start_ticks <= 0:
        raise RevisionDetectionError(
            "Detector process start identity is unavailable."
        )
    tracker = _StatusTracker(
        context,
        run_identity,
        run_identity_sha256,
        len(tasks),
        recovered_errors,
        started_at,
        started_monotonic,
        prior_gpu_intervals,
        checkpoint_fit_seconds_at_process_start,
    )
    metrics: List[dict] = []
    predictions: List[dict] = []
    durations: List[float] = []
    resumed_count = 0
    last_checkpoint: Optional[dict] = None
    boundary_stop = threading.Event()
    previous_handlers: Dict[int, object] = {}

    def request_boundary_stop(signum: int, _frame: object) -> None:
        boundary_stop.set()
        try:
            tracker.transition(state="stop_requested_finishing_current_fit")
        except Exception:
            pass

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_boundary_stop)
    new_fit_count = 0
    try:
        tracker.transition(state="resuming")
        tracker.start_heartbeat()
        for task in tasks:
            loaded = preexisting_checkpoints.get(task.ordinal)
            if loaded is not None:
                metric, fit_predictions, elapsed, checkpoint = loaded
                metrics.append(metric)
                predictions.extend(fit_predictions)
                durations.append(elapsed)
                resumed_count += 1
                last_checkpoint = checkpoint
                tracker.transition(
                    completed=len(metrics),
                    durations=list(durations),
                    last_checkpoint=checkpoint,
                    state="resuming",
                )
                if (
                    benchmark_task_index is not None
                    and task.ordinal == int(benchmark_task_index)
                ):
                    tracker.transition(
                        state="stopped_at_fit_boundary",
                        invocation_completed_at=_utc_now(),
                    )
                    gpu_accounting = tracker.snapshot().get("gpu_accounting")
                    return DetectorExecutionOutcome(
                        result=None,
                        completed_fit_count=len(metrics),
                        total_fit_count=len(tasks),
                        resumed_fit_count=resumed_count,
                        recovered_errors=list(recovered_errors),
                        fit_durations_seconds=durations,
                        run_identity=run_identity,
                        run_identity_sha256=run_identity_sha256,
                        plan_sha256=plan_sha256,
                        started_at_utc=_utc_text(started_at),
                        completed_at_utc=None,
                        stopped_at_fit_boundary=True,
                        last_completed_checkpoint=last_checkpoint,
                        gpu_accounting=gpu_accounting,
                    )
                continue
            if (
                benchmark_task_index is not None
                and task.ordinal != int(benchmark_task_index)
            ):
                continue
            if boundary_stop.is_set():
                tracker.transition(
                    state="stopped_at_fit_boundary",
                    current=None,
                    current_started_at=None,
                    current_started_monotonic=None,
                    next_fit=None,
                    next_fit_upper_seconds=None,
                    fit_gate_nonce=None,
                    invocation_completed_at=_utc_now(),
                )
                gpu_accounting = tracker.snapshot().get("gpu_accounting")
                return DetectorExecutionOutcome(
                    result=None,
                    completed_fit_count=len(metrics),
                    total_fit_count=len(tasks),
                    resumed_fit_count=resumed_count,
                    recovered_errors=list(recovered_errors),
                    fit_durations_seconds=durations,
                    run_identity=run_identity,
                    run_identity_sha256=run_identity_sha256,
                    plan_sha256=plan_sha256,
                    started_at_utc=_utc_text(started_at),
                    completed_at_utc=None,
                    stopped_at_fit_boundary=True,
                    last_completed_checkpoint=last_checkpoint,
                    gpu_accounting=gpu_accounting,
                )
            if context.require_fit_permit:
                next_upper = _next_fit_upper_seconds(context, task)
                gate_nonce = _fit_gate_nonce(
                    run_identity_sha256,
                    task,
                    last_checkpoint,
                    invocation_pid=invocation_pid,
                    invocation_start_ticks=invocation_start_ticks,
                )
                tracker.transition(
                    next_fit=task,
                    next_fit_upper_seconds=next_upper,
                    fit_gate_nonce=gate_nonce,
                    state="awaiting_fit_ceiling_gate",
                )
                permit_granted, permit_receipt = _consume_fit_permit(
                    context,
                    run_identity_sha256=run_identity_sha256,
                    task=task,
                    gate_nonce=gate_nonce,
                    invocation_pid=invocation_pid,
                    invocation_start_ticks=invocation_start_ticks,
                    recovered_errors=recovered_errors,
                    stop_event=boundary_stop,
                )
                if not permit_granted or boundary_stop.is_set():
                    tracker.transition(
                        state="stopped_at_fit_boundary",
                        next_fit=None,
                        next_fit_upper_seconds=None,
                        fit_gate_nonce=None,
                        last_consumed_fit_permit=permit_receipt,
                        invocation_completed_at=_utc_now(),
                    )
                    if permit_receipt is not None:
                        _retire_consumed_fit_permit(context, permit_receipt)
                    gpu_accounting = tracker.snapshot().get("gpu_accounting")
                    return DetectorExecutionOutcome(
                        result=None,
                        completed_fit_count=len(metrics),
                        total_fit_count=len(tasks),
                        resumed_fit_count=resumed_count,
                        recovered_errors=list(recovered_errors),
                        fit_durations_seconds=durations,
                        run_identity=run_identity,
                        run_identity_sha256=run_identity_sha256,
                        plan_sha256=plan_sha256,
                        started_at_utc=_utc_text(started_at),
                        completed_at_utc=None,
                        stopped_at_fit_boundary=True,
                        last_completed_checkpoint=last_checkpoint,
                        gpu_accounting=gpu_accounting,
                    )
            else:
                permit_receipt = None
            fit_started_at = _utc_now()
            fit_started_monotonic = time.monotonic()
            tracker.transition(
                current=task,
                current_started_at=fit_started_at,
                current_started_monotonic=fit_started_monotonic,
                next_fit=None,
                next_fit_upper_seconds=None,
                fit_gate_nonce=None,
                last_consumed_fit_permit=permit_receipt,
                state="running_fit",
            )
            if permit_receipt is not None:
                _retire_consumed_fit_permit(context, permit_receipt)
            metric, fit_predictions = fit_runner(
                prepared, task.split, task.detector_config
            )
            _validate_checkpoint_semantics(
                prepared, task, [metric], fit_predictions
            )
            fit_completed_at = _utc_now()
            elapsed = max(0.0, time.monotonic() - fit_started_monotonic)
            checkpoint = _write_fit_checkpoint(
                context,
                task,
                metric,
                fit_predictions,
                run_identity_sha256,
                plan_sha256,
                fit_started_at,
                fit_completed_at,
                elapsed,
            )
            # Re-read the just-committed checkpoint before accepting it.
            loaded = _load_valid_checkpoint(
                prepared,
                context,
                task,
                run_identity_sha256,
                plan_sha256,
                recovered_errors,
            )
            if loaded is None:
                raise RevisionDetectionError(
                    "Committed detector checkpoint could not be reloaded."
                )
            metric, fit_predictions, elapsed, checkpoint = loaded
            metrics.append(metric)
            predictions.extend(fit_predictions)
            durations.append(elapsed)
            checkpoint_cumulative_fit_seconds = (
                tracker.checkpoint_cumulative_fit_seconds + float(elapsed)
            )
            last_checkpoint = checkpoint
            new_fit_count += 1
            tracker.transition(
                checkpoint_cumulative_fit_seconds=(
                    checkpoint_cumulative_fit_seconds
                ),
                completed=len(metrics),
                durations=list(durations),
                last_checkpoint=checkpoint,
                current=None,
                current_started_at=None,
                current_started_monotonic=None,
                state="fit_checkpointed",
            )
            if boundary_stop.is_set() or (
                stop_after_new_fits is not None
                and new_fit_count >= int(stop_after_new_fits)
            ) or benchmark_task_index is not None:
                tracker.transition(
                    state="stopped_at_fit_boundary",
                    invocation_completed_at=_utc_now(),
                )
                gpu_accounting = tracker.snapshot().get("gpu_accounting")
                return DetectorExecutionOutcome(
                    result=None,
                    completed_fit_count=len(metrics),
                    total_fit_count=len(tasks),
                    resumed_fit_count=resumed_count,
                    recovered_errors=list(recovered_errors),
                    fit_durations_seconds=durations,
                    run_identity=run_identity,
                    run_identity_sha256=run_identity_sha256,
                    plan_sha256=plan_sha256,
                    started_at_utc=_utc_text(started_at),
                    completed_at_utc=None,
                    stopped_at_fit_boundary=True,
                    last_completed_checkpoint=last_checkpoint,
                    gpu_accounting=gpu_accounting,
                )
        result = assemble_prepared_detector_result(prepared, metrics, predictions)
        if len(metrics) != len(tasks):
            raise RevisionDetectionError(
                "Detector aggregation attempted before every fit completed."
            )
        completed_at = _utc_now()
        tracker.transition(
            state="fits_complete_awaiting_final_manifest",
            current=None,
            current_started_at=None,
            current_started_monotonic=None,
            next_fit=None,
            next_fit_upper_seconds=None,
            fit_gate_nonce=None,
            invocation_completed_at=completed_at,
        )
        gpu_accounting = tracker.snapshot().get("gpu_accounting")
        return DetectorExecutionOutcome(
            result=result,
            completed_fit_count=len(metrics),
            total_fit_count=len(tasks),
            resumed_fit_count=resumed_count,
            recovered_errors=list(recovered_errors),
            fit_durations_seconds=durations,
            run_identity=run_identity,
            run_identity_sha256=run_identity_sha256,
            plan_sha256=plan_sha256,
            started_at_utc=_utc_text(started_at),
            completed_at_utc=_utc_text(completed_at),
            stopped_at_fit_boundary=False,
            last_completed_checkpoint=last_checkpoint,
            gpu_accounting=gpu_accounting,
        )
    except BaseException as exc:
        current_task_ordinal = (
            None if tracker.current is None else int(tracker.current.ordinal)
        )
        recovered_errors.append(
            {
                "type": "detector_execution_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "current_task_ordinal": current_task_ordinal,
            }
        )
        try:
            tracker.transition(
                state="error", invocation_completed_at=_utc_now()
            )
        except Exception:
            pass
        raise
    finally:
        tracker.stop_heartbeat()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def execute_checkpointed_detector_suite(
    prepared: PreparedDetectorSuite,
    context: DetectorExecutionContext,
    *,
    fit_runner: FitRunner = run_prepared_detector_fit,
    stop_after_new_fits: Optional[int] = None,
    benchmark_task_index: Optional[int] = None,
) -> DetectorExecutionOutcome:
    """Acquire the checkpoint-root lease, then execute/resume the suite."""

    with detector_execution_lease(context.checkpoint_dir):
        return _execute_checkpointed_detector_suite_locked(
            prepared,
            context,
            fit_runner=fit_runner,
            stop_after_new_fits=stop_after_new_fits,
            benchmark_task_index=benchmark_task_index,
        )


def mark_detector_execution_complete(
    status_file: Path,
    *,
    run_identity_sha256: str,
    final_manifest: Path,
    terminal_receipt: Optional[Path] = None,
    ledger_incorporation_marker: Optional[Path] = None,
) -> None:
    """Seal status only after the final detector run manifest exists and hashes."""

    status = _read_json(Path(status_file), "detector status")
    _verify_signed_document(status, "status_sha256", "Detector status")
    if status.get("run_identity_sha256") != run_identity_sha256:
        raise RevisionDetectionError("Detector completion status identity differs.")
    manifest_path = Path(final_manifest)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RevisionDetectionError("Final detector manifest is missing or unsafe.")
    status["state"] = "complete"
    status["updated_at_utc"] = _utc_text(_utc_now())
    status["final_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "size_bytes": int(manifest_path.stat().st_size),
    }
    if terminal_receipt is not None:
        receipt_path = Path(terminal_receipt)
        receipt = _validate_terminal_receipt(receipt_path)
        if (
            receipt.get("run_identity_sha256") != run_identity_sha256
            or receipt.get("published_output", {}).get("sha256")
            != status["final_manifest"]["sha256"]
        ):
            raise RevisionDetectionError(
                "Detector terminal receipt publication identity differs."
            )
        status["terminal_receipt"] = {
            "path": str(receipt_path.resolve()),
            "sha256": file_sha256(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
            "terminal_receipt_sha256": receipt["terminal_receipt_sha256"],
        }
        status["gpu_accounting"] = receipt["gpu_accounting"]
    if ledger_incorporation_marker is not None:
        if terminal_receipt is None:
            raise RevisionDetectionError(
                "Ledger incorporation requires a detector terminal receipt."
            )
        marker_path = Path(ledger_incorporation_marker)
        marker = read_detector_gpu_ledger_incorporation_marker(marker_path)
        if (
            marker.get("final_terminal_receipt", {}).get("sha256")
            != file_sha256(Path(terminal_receipt))
            or marker.get("final_published_manifest", {}).get("sha256")
            != status["final_manifest"]["sha256"]
        ):
            raise RevisionDetectionError(
                "Detector GPU ledger incorporation publication identity differs."
            )
        status["gpu_ledger_incorporation"] = {
            "path": str(marker_path.resolve()),
            "sha256": file_sha256(marker_path),
            "size_bytes": int(marker_path.stat().st_size),
            "incorporation_sha256": marker["incorporation_sha256"],
        }
    atomic_write_json(
        Path(status_file), _signed_document(status, "status_sha256")
    )


def mark_detector_awaiting_supervisor_finalization(
    status_file: Path,
    *,
    run_identity_sha256: str,
    candidate_path: Path,
) -> dict:
    """Publish the exact runner candidate in status, then allow process exit."""

    status = _read_json(Path(status_file), "detector status")
    _verify_signed_document(status, "status_sha256", "Detector status")
    candidate = read_detector_finalization_candidate(candidate_path)
    if (
        status.get("run_identity_sha256") != run_identity_sha256
        or candidate.get("run_identity_sha256") != run_identity_sha256
    ):
        raise RevisionDetectionError(
            "Detector finalization candidate/status identity differs."
        )
    status["state"] = "awaiting_supervisor_finalization"
    observed_at = _utc_now()
    status["updated_at_utc"] = _utc_text(observed_at)
    accounting = status.get("gpu_accounting")
    if status.get("device") == "cuda:0":
        if (
            not isinstance(accounting, dict)
            or not isinstance(accounting.get("intervals"), list)
            or not accounting["intervals"]
        ):
            raise RevisionDetectionError(
                "CUDA detector finalization status lacks accounting."
            )
        intervals = [dict(value) for value in accounting["intervals"]]
        current = intervals[-1]
        if (
            int(current.get("pid", -1)) != os.getpid()
            or int(current.get("process_start_ticks", -1))
            != int(_process_start_ticks() or -2)
        ):
            raise RevisionDetectionError(
                "CUDA detector finalization interval does not identify this process."
            )
        started = datetime.fromisoformat(str(current["started_at_utc"]))
        current["completed_at_utc"] = None
        current["last_observed_at_utc"] = _utc_text(observed_at)
        current["elapsed_seconds"] = float(
            (observed_at - started).total_seconds()
        )
        intervals[-1] = current
        status["gpu_accounting"] = {
            "device": "cuda:0",
            "gpu_uuid": status.get("gpu_uuid"),
            "intervals": intervals,
            "cumulative_elapsed_seconds": float(
                sum(float(value["elapsed_seconds"]) for value in intervals)
            ),
            "derivation_policy": GPU_ACCOUNTING_COLLECTION_POLICY,
        }
    status["finalization_candidate"] = {
        "path": str(Path(candidate_path).resolve()),
        "sha256": file_sha256(Path(candidate_path)),
        "size_bytes": int(Path(candidate_path).stat().st_size),
        "candidate_sha256": candidate["candidate_sha256"],
        "kind": candidate["kind"],
    }
    signed = _signed_document(status, "status_sha256")
    atomic_write_json(Path(status_file), signed)
    return signed


def verify_status_file(path: Path) -> dict:
    """Load and verify a status document for supervisor/tests."""

    value = _read_json(Path(path), "detector status")
    if value.get("schema_version") != STATUS_SCHEMA:
        raise RevisionDetectionError("Detector status schema differs.")
    _verify_signed_document(value, "status_sha256", "Detector status")
    return value
