"""Hash-bound coverage map for requested transmission perturbations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CONFIG_SCHEMA = "rankcloak-robustness-coverage-map-v1"
MANIFEST_SCHEMA = "rankcloak-robustness-coverage-inventory-v1"
COVERAGE_STATUSES = {
    "directly_tested",
    "partially_represented",
    "not_tested",
}


class RobustnessCoverageError(ValueError):
    """Raised when the requested-condition crosswalk is inconsistent."""


@dataclass(frozen=True)
class RobustnessCoverageArtifacts:
    output_dir: str
    manifest_path: str
    request_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RobustnessCoverageError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RobustnessCoverageError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RobustnessCoverageError(f"{label} must contain a JSON object")
    return value


def _identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def _declared_conditions(
    manifest: Mapping[str, Any], manifest_path: Path
) -> Path:
    outputs = manifest.get("outputs")
    declaration = outputs.get("conditions") if isinstance(outputs, Mapping) else None
    if not isinstance(declaration, Mapping):
        raise RobustnessCoverageError("Robustness manifest lacks conditions output")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RobustnessCoverageError("Robustness conditions path is missing")
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RobustnessCoverageError(f"Robustness conditions are unsafe: {path}")
    if (
        file_sha256(path) != declaration.get("sha256")
        or int(path.stat().st_size) != int(declaration.get("size_bytes", -1))
    ):
        raise RobustnessCoverageError("Robustness conditions identity differs")
    return path


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_robustness_coverage_inventory(
    *,
    coverage_config: str | Path,
    robustness_config: str | Path,
    robustness_manifest: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> RobustnessCoverageArtifacts:
    """Cross-walk requested perturbations to frozen raw-transmission rows."""

    coverage_path = Path(coverage_config).resolve()
    robustness_config_path = Path(robustness_config).resolve()
    manifest_path = Path(robustness_manifest).resolve()
    coverage = _read_json(coverage_path, label="coverage config")
    design = _read_json(robustness_config_path, label="robustness config")
    manifest = _read_json(manifest_path, label="robustness manifest")
    if (
        coverage.get("schema_version") != CONFIG_SCHEMA
        or coverage.get("analysis_status")
        != "supporting_post_audit_coverage_inventory"
        or coverage.get("frozen_robustness_results_unchanged") is not True
    ):
        raise RobustnessCoverageError("Coverage disclosure differs")
    if (
        design.get("schema_version") != "1.0"
        or manifest.get("schema_version")
        != "rankcloak-revision-robustness-analysis-v1"
        or manifest.get("status") != "passed"
        or manifest.get("failure_taxonomy_scope")
        != "descriptive_first_divergence_not_causal_proof"
    ):
        raise RobustnessCoverageError("Frozen robustness identity differs")
    transformations = design.get("transformations")
    if not isinstance(transformations, list) or not transformations:
        raise RobustnessCoverageError("Robustness design has no transformations")
    transformation_ids = {
        str(row.get("transformation_id", ""))
        for row in transformations
        if isinstance(row, Mapping)
    }
    if "" in transformation_ids:
        raise RobustnessCoverageError("A robustness transformation lacks identity")

    conditions_path = _declared_conditions(manifest, manifest_path)
    conditions = pd.read_csv(conditions_path, low_memory=False)
    required = {
        "robustness_family",
        "transformation_id",
        "observed_outcome_rows",
        "unavailable_rows",
        "success_outcome_rows",
        "failure_outcome_rows",
        "recovery_rate",
        "analysis_unit",
    }
    if not required.issubset(conditions.columns):
        raise RobustnessCoverageError("Robustness conditions columns differ")
    raw = conditions.loc[
        conditions["robustness_family"].astype(str).eq("raw_transmission")
    ].copy()
    if raw["transformation_id"].astype(str).duplicated().any():
        raise RobustnessCoverageError("Raw transmission condition identity repeats")

    requests = coverage.get("requests")
    expected = int(coverage.get("expected_request_count", -1))
    if not isinstance(requests, list) or len(requests) != expected:
        raise RobustnessCoverageError("Requested perturbation count differs")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise RobustnessCoverageError(f"Coverage request {index} is malformed")
        request_id = str(request.get("request_id", ""))
        requested = str(request.get("requested_transformation", ""))
        status = str(request.get("coverage_status", ""))
        ids = request.get("transformation_ids")
        scope_note = str(request.get("scope_note", ""))
        limitation = str(request.get("limitation", ""))
        if (
            not request_id
            or request_id in seen
            or not requested
            or status not in COVERAGE_STATUSES
            or not isinstance(ids, list)
            or not scope_note
            or not limitation
        ):
            raise RobustnessCoverageError(f"Coverage request {index} differs")
        seen.add(request_id)
        mapped = list(map(str, ids))
        if len(mapped) != len(set(mapped)) or not set(mapped).issubset(
            transformation_ids
        ):
            raise RobustnessCoverageError(f"Coverage mapping differs for {request_id}")
        if (status == "not_tested") != (len(mapped) == 0):
            raise RobustnessCoverageError(
                f"Tested-status mapping differs for {request_id}"
            )
        cells = raw.loc[raw["transformation_id"].astype(str).isin(mapped)].copy()
        if len(cells) != len(mapped):
            raise RobustnessCoverageError(
                f"Frozen condition rows differ for {request_id}"
            )
        for column in (
            "observed_outcome_rows",
            "unavailable_rows",
            "success_outcome_rows",
            "failure_outcome_rows",
            "recovery_rate",
        ):
            values = pd.to_numeric(cells[column], errors="coerce")
            if len(values) and not values.map(math.isfinite).all():
                raise RobustnessCoverageError(
                    f"Non-finite {column} for {request_id}"
                )
            cells[column] = values
        observed = int(cells["observed_outcome_rows"].sum())
        successes = int(cells["success_outcome_rows"].sum())
        failures = int(cells["failure_outcome_rows"].sum())
        unavailable = int(cells["unavailable_rows"].sum())
        if observed != successes + failures:
            raise RobustnessCoverageError(
                f"Outcome arithmetic differs for {request_id}"
            )
        rates = list(map(float, cells["recovery_rate"]))
        rows.append(
            {
                "request_id": request_id,
                "requested_transformation": requested,
                "coverage_status": status,
                "frozen_transformation_ids": ";".join(mapped),
                "tested_condition_count": len(mapped),
                "observed_outcome_rows_across_conditions": observed,
                "success_rows_across_conditions": successes,
                "failure_rows_across_conditions": failures,
                "unavailable_rows_across_conditions": unavailable,
                "minimum_condition_recovery_rate": (
                    "" if not rates else min(rates)
                ),
                "maximum_condition_recovery_rate": (
                    "" if not rates else max(rates)
                ),
                "analysis_unit": (
                    "not_estimated"
                    if not len(cells)
                    else ";".join(sorted(set(map(str, cells["analysis_unit"]))))
                ),
                "scope_note": scope_note,
                "limitation": limitation,
            }
        )
    frame = pd.DataFrame(rows)

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise RobustnessCoverageError(f"Unsafe output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "perturbation_coverage_inventory.csv"
    output_manifest_path = target / "perturbation_coverage_manifest.json"
    for path in (output_path, output_manifest_path):
        if path.exists() and not overwrite:
            raise RobustnessCoverageError(f"Refusing to overwrite output: {path}")
    _atomic_csv(frame, output_path)
    output_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "analysis_status": coverage["analysis_status"],
        "frozen_robustness_results_unchanged": True,
        "inputs": {
            "coverage_config": _identity(coverage_path),
            "robustness_config": _identity(robustness_config_path),
            "robustness_manifest": _identity(manifest_path),
            "recovery_by_condition": _identity(
                conditions_path, row_count=len(conditions)
            ),
        },
        "outputs": {
            "perturbation_coverage_inventory": _identity(
                output_path, row_count=len(frame)
            )
        },
        "summary": {
            "requested_transformation_count": len(frame),
            "directly_tested_count": int(
                frame["coverage_status"].eq("directly_tested").sum()
            ),
            "partially_represented_count": int(
                frame["coverage_status"].eq("partially_represented").sum()
            ),
            "not_tested_count": int(
                frame["coverage_status"].eq("not_tested").sum()
            ),
        },
        "aggregation_warning": (
            "Counts spanning multiple mapped conditions are descriptive row sums, "
            "not pooled recovery estimands."
        ),
        "generation_command": command,
    }
    output_manifest["manifest_sha256"] = canonical_json_sha256(output_manifest)
    _atomic_json(output_manifest, output_manifest_path)
    return RobustnessCoverageArtifacts(
        output_dir=str(target),
        manifest_path=str(output_manifest_path),
        request_count=len(frame),
    )
