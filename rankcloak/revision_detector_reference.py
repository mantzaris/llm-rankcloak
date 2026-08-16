"""Non-copying, hash-validated index of completed detector fit artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SCHEMA = "rankcloak-revision-detector-fit-checkpoint-v1"
ROW_SCHEMA = "rankcloak-revision-detector-fit-rows-v1"
OUTPUT_SCHEMA = "rankcloak-revision-detector-reference-index-v1"


class DetectorReferenceError(ValueError):
    """Raised when a final detector output or fit checkpoint is inconsistent."""


@dataclass(frozen=True)
class DetectorReferenceArtifacts:
    output_dir: str
    fit_index_path: str
    references_path: str
    manifest_path: str
    fit_count: int
    external_reference_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DetectorReferenceError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DetectorReferenceError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DetectorReferenceError(f"{label} must contain a JSON object")
    return value


def _verify_signed(value: Mapping[str, Any], field: str, *, label: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if observed != canonical_json_sha256(unsigned):
        raise DetectorReferenceError(f"{label} self-hash differs")


def _identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DetectorReferenceError(f"Missing or unsafe detector artifact: {path}")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _payload_rows(value: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    columns = value.get("columns")
    rows = value.get("rows")
    if (
        value.get("schema_version") != ROW_SCHEMA
        or set(value) != {"schema_version", "columns", "rows"}
        or not isinstance(columns, list)
        or len(columns) != len(set(map(str, columns)))
        or not isinstance(rows, list)
        or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
    ):
        raise DetectorReferenceError(f"{label} row artifact is malformed")
    return [dict(zip(map(str, columns), row)) for row in rows]


def _declared_final_outputs(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, dict[str, Any]]:
    output_dir = Path(str(manifest.get("output_dir", ""))).resolve()
    if output_dir != manifest_path.parent.resolve():
        raise DetectorReferenceError("Detector manifest output directory differs")
    declarations = manifest.get("output_files")
    expected = {
        "detector_metrics.csv",
        "detector_predictions.csv",
        "detector_dataset_manifest.csv",
        "detector_split_manifest.json",
        "detector_failures.json",
    }
    if not isinstance(declarations, Mapping) or set(declarations) != expected:
        raise DetectorReferenceError("Detector final output declaration set differs")
    results: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        declaration = declarations[name]
        if not isinstance(declaration, Mapping):
            raise DetectorReferenceError(f"Detector output declaration is malformed: {name}")
        path = output_dir / name
        rows = _csv_rows(path) if path.suffix == ".csv" else None
        identity = _identity(path, row_count=rows)
        if (
            identity["sha256"] != declaration.get("sha256")
            or identity["size_bytes"] != int(declaration.get("size_bytes", -1))
        ):
            raise DetectorReferenceError(f"Detector final output identity differs: {name}")
        results[name] = identity
    return results


def _validate_run_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _read_json(path, label="detector run manifest")
    _verify_signed(value, "manifest_sha256", label="detector run manifest")
    if (
        value.get("schema_version") != "rankcloak-revision-detector-run-v2"
        or value.get("execution_mode") != "confirmatory"
        or value.get("confirmatory_complete") is not True
        or value.get("device") != "cuda:0"
        or int(value.get("completed_fit_count", -1)) <= 0
        or int(value.get("completed_fit_count", -1))
        != int(value.get("total_fit_count", -2))
        or int(value.get("metric_rows", -1)) != int(value["total_fit_count"])
        or int(value.get("failure_count", -1)) != 0
        or int(value.get("smoke_fallback_metric_rows", -1)) != 0
        or not isinstance(value.get("gpu_accounting"), Mapping)
    ):
        raise DetectorReferenceError("Detector run is not a complete CUDA confirmatory matrix")
    return value, _declared_final_outputs(value, path)


def _validate_checkpoint(
    directory: Path,
    *,
    ordinal: int,
    run_identity_sha256: str,
    plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise DetectorReferenceError(f"Missing or unsafe checkpoint directory: {directory}")
    entries = sorted(path.name for path in directory.iterdir())
    if entries != ["manifest.json", "metric.json", "predictions.json"]:
        raise DetectorReferenceError(f"Checkpoint file set differs: {directory}")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path, label="fit checkpoint manifest")
    _verify_signed(manifest, "manifest_sha256", label="fit checkpoint manifest")
    task = manifest.get("task_identity")
    children = manifest.get("children")
    elapsed = float(manifest.get("elapsed_seconds", float("nan")))
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or manifest.get("run_identity_sha256") != run_identity_sha256
        or manifest.get("plan_sha256") != plan_sha256
        or not isinstance(task, Mapping)
        or int(task.get("ordinal", -1)) != ordinal
        or manifest.get("task_identity_sha256") != canonical_json_sha256(task)
        or not isinstance(children, Mapping)
        or set(children) != {"metric.json", "predictions.json"}
        or manifest.get("children_sha256") != canonical_json_sha256(children)
        or not math.isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise DetectorReferenceError(f"Checkpoint lineage differs for ordinal {ordinal}")

    child_results: dict[str, dict[str, Any]] = {}
    child_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("metric.json", "predictions.json"):
        declaration = children[name]
        if not isinstance(declaration, Mapping):
            raise DetectorReferenceError(f"Checkpoint child declaration differs: {name}")
        path = directory / name
        value = _read_json(path, label=f"checkpoint {name}")
        rows = _payload_rows(value, label=f"checkpoint {name}")
        identity = _identity(path, row_count=len(rows))
        if (
            identity["sha256"] != declaration.get("sha256")
            or identity["size_bytes"] != int(declaration.get("size_bytes", -1))
            or identity["row_count"] != int(declaration.get("row_count", -1))
        ):
            raise DetectorReferenceError(f"Checkpoint child identity differs: {path}")
        child_results[name] = identity
        child_rows[name] = rows
    if len(child_rows["metric.json"]) != 1:
        raise DetectorReferenceError(f"Checkpoint {ordinal} does not contain one metric row")
    metric = child_rows["metric.json"][0]
    if (
        metric.get("split_id") != task.get("split_id")
        or metric.get("regime") != task.get("regime")
        or metric.get("detector_name") != task.get("detector_name")
        or metric.get("requested_kind") != task.get("detector_kind")
        or metric.get("implementation_status") != "complete"
        or metric.get("implementation_kind") != task.get("detector_kind")
        or int(metric.get("seed", -1)) != int(task.get("seed", -2))
        or len(child_rows["predictions.json"]) != int(task.get("test_row_count", -1))
    ):
        raise DetectorReferenceError(f"Checkpoint metric task identity differs: {ordinal}")
    reference = {
        "checkpoint_manifest": _identity(manifest_path),
        "metric": child_results["metric.json"],
        "predictions": child_results["predictions.json"],
    }
    row = {
        "ordinal": ordinal,
        "detector_name": task["detector_name"],
        "detector_kind": task["detector_kind"],
        "split_id": task["split_id"],
        "regime": task["regime"],
        "held_out_column": task.get("held_out_column"),
        "held_out_value": task.get("held_out_value"),
        "seed": int(task["seed"]),
        "train_row_count": int(task["train_row_count"]),
        "test_row_count": int(task["test_row_count"]),
        "elapsed_seconds": elapsed,
        "model_state_sha256": metric.get("model_state_sha256"),
        "model_artifact_set_sha256": metric.get("model_artifact_set_sha256"),
        "checkpoint_manifest_path": reference["checkpoint_manifest"]["path"],
        "checkpoint_manifest_sha256": reference["checkpoint_manifest"]["sha256"],
        "metric_path": reference["metric"]["path"],
        "metric_sha256": reference["metric"]["sha256"],
        "predictions_path": reference["predictions"]["path"],
        "predictions_sha256": reference["predictions"]["sha256"],
    }
    return row, reference


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise DetectorReferenceError("Cannot write an empty detector fit index")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_detector_reference_index(
    *,
    detector_run_manifest: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> DetectorReferenceArtifacts:
    """Validate final outputs/checkpoints and publish references without copying them."""

    run_path = Path(detector_run_manifest).resolve()
    run, final_outputs = _validate_run_manifest(run_path)
    checkpoint_root = Path(str(run.get("checkpoint_dir", ""))).resolve()
    fits_root = checkpoint_root / "fits"
    total = int(run["total_fit_count"])
    expected_names = [f"{ordinal:04d}" for ordinal in range(total)]
    if (
        fits_root.is_symlink()
        or not fits_root.is_dir()
        or sorted(path.name for path in fits_root.iterdir()) != expected_names
    ):
        raise DetectorReferenceError("Detector checkpoint directory set differs")

    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for ordinal, name in enumerate(expected_names):
        row, reference = _validate_checkpoint(
            fits_root / name,
            ordinal=ordinal,
            run_identity_sha256=str(run["run_identity_sha256"]),
            plan_sha256=str(run["execution_plan_sha256"]),
        )
        rows.append(row)
        checkpoints.append({"ordinal": ordinal, **reference})
    if not math.isclose(
        sum(float(row["elapsed_seconds"]) for row in rows),
        float(run.get("checkpoint_cumulative_fit_seconds", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise DetectorReferenceError("Checkpoint duration total differs from final run")

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise DetectorReferenceError(f"Unsafe detector reference directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "fit_index": target / "detector_fit_reference_index.csv",
        "references": target / "detector_output_references.json",
        "manifest": target / "detector_reference_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise DetectorReferenceError(f"Refusing to overwrite detector reference: {path}")

    _atomic_csv(paths["fit_index"], rows)
    reference_value = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "passed",
        "large_detector_artifacts_copied": False,
        "detector_run_manifest": _identity(run_path),
        "final_outputs": final_outputs,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_references": checkpoints,
        "summary": {
            "fit_count": len(rows),
            "final_output_reference_count": len(final_outputs),
            "checkpoint_file_reference_count": 3 * len(checkpoints),
        },
    }
    _atomic_json(paths["references"], reference_value)
    outputs = {
        "fit_index": _identity(paths["fit_index"], row_count=len(rows)),
        "references": _identity(paths["references"]),
    }
    manifest_value = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "passed",
        "inputs": {"detector_run_manifest": _identity(run_path)},
        "outputs": outputs,
        "large_detector_artifacts_copied": False,
        "fit_count": len(rows),
        "generation_command": command,
    }
    _atomic_json(paths["manifest"], manifest_value)
    return DetectorReferenceArtifacts(
        output_dir=str(target),
        fit_index_path=str(paths["fit_index"]),
        references_path=str(paths["references"]),
        manifest_path=str(paths["manifest"]),
        fit_count=len(rows),
        external_reference_count=len(final_outputs) + 3 * len(checkpoints) + 1,
    )
