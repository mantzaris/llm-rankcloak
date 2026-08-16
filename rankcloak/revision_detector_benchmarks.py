"""Compact, hash-bound CUDA detector benchmark and repeatability evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


MANIFEST_SCHEMA = "rankcloak-detector-benchmark-evidence-v1"
EXPECTED_TASKS = {
    0: ("published_textcnn_equivalent", "text_cnn"),
    1: ("deberta_v3_base_classifier", "pretrained_transformer"),
}
EXACT_CHECKS = {
    "metrics_exact",
    "model_state_sha256_exact",
    "predictions_exact",
    "row_identity_order_labels_exact",
    "scores_exact",
    "task_design_exact",
}
PHASES = (
    "initialization_and_preprocessing",
    "training",
    "evaluation",
    "trained_state_hashing",
)


class DetectorBenchmarkError(ValueError):
    """Raised when benchmark or same-CUDA reproducibility evidence differs."""


@dataclass(frozen=True)
class DetectorBenchmarkArtifacts:
    output_dir: str
    manifest_path: str
    architecture_count: int


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


def _read_signed(path: Path, *, field: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DetectorBenchmarkError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DetectorBenchmarkError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DetectorBenchmarkError(f"{label} must contain a JSON object")
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if signature != canonical_json_sha256(unsigned):
        raise DetectorBenchmarkError(f"{label} self-hash differs")
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


def _finite(value: object, *, label: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectorBenchmarkError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise DetectorBenchmarkError(f"{label} is outside its allowed range")
    return result


def _phases(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DetectorBenchmarkError(f"{label} phases are malformed")
    result = {phase: _finite(value.get(phase), label=f"{label} {phase}") for phase in PHASES}
    total = _finite(value.get("total"), label=f"{label} total", positive=True)
    residual = total - sum(result.values())
    if residual < -1e-6 or residual > max(0.1, total * 1e-4):
        raise DetectorBenchmarkError(f"{label} phase total differs")
    result["total"] = total
    result["timer_residual"] = residual
    return result


def _validate_repro_artifacts(report: Mapping[str, Any]) -> None:
    inputs = report.get("input_artifacts")
    if not isinstance(inputs, Mapping) or set(inputs) != {"cuda", "cuda_repeat"}:
        raise DetectorBenchmarkError("Reproducibility input artifact set differs")
    for role, raw in inputs.items():
        if not isinstance(raw, Mapping):
            raise DetectorBenchmarkError(f"Reproducibility {role} identity is malformed")
        path = Path(str(raw.get("path", ""))).resolve()
        if path.is_symlink() or not path.is_file():
            raise DetectorBenchmarkError(f"Reproducibility {role} artifact is unsafe")
        if (
            file_sha256(path) != raw.get("sha256")
            or int(path.stat().st_size) != int(raw.get("size_bytes", -1))
        ):
            raise DetectorBenchmarkError(f"Reproducibility {role} identity differs")


def build_detector_benchmark_evidence(
    *,
    benchmark_paths: Sequence[str | Path],
    reproducibility_report_paths: Sequence[str | Path],
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> DetectorBenchmarkArtifacts:
    """Validate and compact the two frozen CUDA architecture gates."""

    if len(benchmark_paths) != 2 or len(reproducibility_report_paths) != 2:
        raise DetectorBenchmarkError("Exactly two benchmark/report pairs are required")
    benchmarks: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in benchmark_paths:
        path = Path(raw_path).resolve()
        value = _read_signed(path, field="benchmark_sha256", label="benchmark")
        index = int(value.get("benchmark_task_index", -1))
        if index in benchmarks:
            raise DetectorBenchmarkError(f"Duplicate benchmark task index: {index}")
        benchmarks[index] = (path, value)
    reports: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in reproducibility_report_paths:
        path = Path(raw_path).resolve()
        value = _read_signed(path, field="report_sha256", label="reproducibility report")
        artifacts = value.get("input_artifacts")
        if not isinstance(artifacts, Mapping):
            raise DetectorBenchmarkError("Reproducibility report input set differs")
        cuda = artifacts.get("cuda")
        if not isinstance(cuda, Mapping):
            raise DetectorBenchmarkError("Reproducibility report lacks CUDA artifact")
        artifact_path = Path(str(cuda.get("path", "")))
        try:
            index = int(artifact_path.parent.name.removeprefix("task_"))
        except ValueError as exc:
            raise DetectorBenchmarkError("Reproducibility task identity is unclear") from exc
        if index in reports:
            raise DetectorBenchmarkError(f"Duplicate reproducibility task index: {index}")
        reports[index] = (path, value)
    if set(benchmarks) != set(EXPECTED_TASKS) or set(reports) != set(EXPECTED_TASKS):
        raise DetectorBenchmarkError("Benchmark architecture task set differs")

    rows: list[dict[str, Any]] = []
    input_identities: dict[str, dict[str, Any]] = {}
    for index, (expected_name, expected_kind) in EXPECTED_TASKS.items():
        benchmark_path, benchmark = benchmarks[index]
        report_path, report = reports[index]
        task = benchmark.get("benchmark_task_identity")
        config = task.get("effective_detector_config") if isinstance(task, Mapping) else None
        if (
            benchmark.get("schema_version")
            != "rankcloak-revision-detector-benchmark-v1"
            or benchmark.get("device") != "cuda:0"
            or int(benchmark.get("workers", -1)) != 1
            or int(benchmark.get("new_fit_count", -1)) != 1
            or not isinstance(task, Mapping)
            or int(task.get("ordinal", -1)) != index
            or task.get("detector_name") != expected_name
            or task.get("detector_kind") != expected_kind
            or not isinstance(config, Mapping)
        ):
            raise DetectorBenchmarkError(f"Benchmark task {index} identity differs")
        phases = _phases(benchmark.get("phase_timings_seconds"), label="benchmark")
        non_model = _finite(
            benchmark.get("fit_non_model_analysis_seconds"), label="non-model analysis"
        )
        fit = _finite(benchmark.get("fit_elapsed_seconds"), label="fit elapsed", positive=True)
        checkpoint = _finite(benchmark.get("checkpoint_seconds"), label="checkpoint")
        invocation = _finite(
            benchmark.get("invocation_overhead_seconds"), label="invocation overhead"
        )
        wall = _finite(benchmark.get("wall_seconds"), label="wall", positive=True)
        if (
            not math.isclose(fit, phases["total"] + non_model, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                wall, fit + checkpoint + invocation, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise DetectorBenchmarkError(f"Benchmark task {index} timing arithmetic differs")

        if report.get("schema_version") != "rankcloak-revision-detector-cuda-reproducibility-v1":
            raise DetectorBenchmarkError(f"Reproducibility task {index} schema differs")
        decision = report.get("decision")
        same = decision.get("same_device_cuda") if isinstance(decision, Mapping) else None
        checks = same.get("checks") if isinstance(same, Mapping) else None
        measurements = same.get("measurements") if isinstance(same, Mapping) else None
        if (
            not isinstance(decision, Mapping)
            or decision.get("reproducible") is not True
            or decision.get("scientific_task_identity_exact") is not True
            or not isinstance(same, Mapping)
            or same.get("passed") is not True
            or not isinstance(checks, Mapping)
            or set(checks) != EXACT_CHECKS
            or not all(checks.values())
            or not isinstance(measurements, Mapping)
            or _finite(
                measurements.get("maximum_score_absolute_difference"),
                label="maximum score difference",
            )
            != 0.0
            or measurements.get("timings_excluded_from_scientific_equality") is not True
        ):
            raise DetectorBenchmarkError(f"Reproducibility task {index} did not pass exactly")
        primary_phases = _phases(
            measurements.get("cuda_phase_timings_seconds"), label="primary CUDA repeatability"
        )
        repeat_phases = _phases(
            measurements.get("cuda_repeat_phase_timings_seconds"), label="repeat CUDA"
        )
        if any(
            not math.isclose(phases[key], primary_phases[key], rel_tol=0.0, abs_tol=1e-9)
            for key in (*PHASES, "total")
        ):
            raise DetectorBenchmarkError(f"Benchmark/reproducibility phase identity differs for task {index}")
        _validate_repro_artifacts(report)
        input_identities[f"benchmark_task_{index}"] = _identity(benchmark_path)
        input_identities[f"reproducibility_task_{index}"] = _identity(report_path)
        rows.append(
            {
                "task_index": index,
                "detector_name": expected_name,
                "detector_kind": expected_kind,
                "train_rows": int(task["train_row_count"]),
                "test_rows": int(task["test_row_count"]),
                "batch_size": int(config["batch_size"]),
                "epochs": int(config["epochs"]),
                "initialization_and_preprocessing_seconds": phases["initialization_and_preprocessing"],
                "training_seconds": phases["training"],
                "evaluation_seconds": phases["evaluation"],
                "trained_state_hashing_seconds": phases["trained_state_hashing"],
                "phase_timer_residual_seconds": phases["timer_residual"],
                "model_phase_total_seconds": phases["total"],
                "non_model_analysis_seconds": non_model,
                "fit_elapsed_seconds": fit,
                "checkpoint_seconds": checkpoint,
                "invocation_overhead_seconds": invocation,
                "wall_seconds": wall,
                "peak_rss_bytes": int(benchmark["peak_rss_bytes"]),
                "peak_vram_bytes": int(benchmark["peak_vram_bytes"]),
                "repeat_model_phase_total_seconds": repeat_phases["total"],
                "prediction_rows_compared": int(measurements["prediction_row_count"]),
                "maximum_score_absolute_difference": 0.0,
                "all_predeclared_exact_checks_passed": True,
                "reproducibility_scope": "same_device_cuda_not_cpu_gpu_equivalence",
            }
        )

    frame = pd.DataFrame(rows).sort_values("task_index").reset_index(drop=True)
    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise DetectorBenchmarkError(f"Unsafe benchmark output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    table_path = target / "detector_cuda_benchmark_summary.csv"
    manifest_path = target / "detector_cuda_benchmark_manifest.json"
    for path in (table_path, manifest_path):
        if path.exists() and not overwrite:
            raise DetectorBenchmarkError(f"Refusing to overwrite benchmark output: {path}")
    temporary = table_path.with_name(f".{table_path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, table_path)
    output_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "device": "cuda:0",
        "architecture_count": len(frame),
        "same_device_cuda_reproducibility_passed": True,
        "cpu_gpu_equivalence_tested": False,
        "inputs": input_identities,
        "outputs": {
            "detector_cuda_benchmark_summary": _identity(
                table_path, row_count=len(frame)
            )
        },
        "timing_scope": {
            "fit_elapsed_seconds": "model phases plus post-fit metric and bootstrap analysis",
            "wall_seconds": "fit plus atomic checkpoint publication and invocation overhead",
            "phase_timer_residual_seconds": "outer phase timer minus the sum of individually instrumented phases",
        },
        "generation_command": command,
    }
    output_manifest["manifest_sha256"] = canonical_json_sha256(output_manifest)
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary_manifest.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return DetectorBenchmarkArtifacts(
        output_dir=str(target),
        manifest_path=str(manifest_path),
        architecture_count=len(frame),
    )
