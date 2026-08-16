#!/usr/bin/env python3
"""Build the signed, benchmark-derived CUDA detector budget gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_artifacts import canonical_json_sha256, file_sha256
from rankcloak.revision_detection import RevisionDetectionError
from rankcloak.revision_detector_execution import (
    EXPECTED_GPU_LEDGER_SOURCES,
    atomic_write_json,
    read_detector_failed_benchmark_attempt,
    read_detector_gpu_accounting_ledger,
)


SCHEMA = "rankcloak-revision-detector-cuda-budget-gate-v1"
POLICY_SCHEMA = "rankcloak-revision-detector-cuda-policy-v2"
BENCHMARK_SCHEMA = "rankcloak-revision-detector-benchmark-v1"
PLAN_SCHEMA = "rankcloak-revision-detector-execution-plan-v1"
ROOT = PROJECT_ROOT / "results" / "revision_v1" / "detector_cuda_reproducibility_v2"
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "operations"
    / "confirmatory_v2"
    / "detector_cuda_policy_v2.json"
)
DEFAULT_BENCHMARKS = (
    ROOT / "benchmarks" / "task_0_cuda.json",
    ROOT / "benchmarks" / "task_1_cuda.json",
)
DEFAULT_LEDGER = ROOT / "gpu_accounting_ledger.json"
DEFAULT_FAILED_ATTEMPT = (
    ROOT / "failed_attempts" / "task_0_packaging_failure.json"
)
DEFAULT_OUTPUT = ROOT / "cuda_budget_gate.json"
ARCHITECTURES = {
    0: "published_textcnn_equivalent",
    1: "deberta_v3_base_classifier",
}
PHASES = (
    "initialization_and_preprocessing",
    "training",
    "trained_state_hashing",
    "evaluation",
    "total",
)
STAGES = {
    "post_benchmark_pre_reproducibility",
    "post_reproducibility_preproduction",
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    declared = Path(path)
    if declared.is_symlink() or not declared.is_file():
        raise RevisionDetectionError(f"{label} is missing or unsafe: {declared}")
    try:
        value = json.loads(declared.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RevisionDetectionError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RevisionDetectionError(f"{label} must be a JSON object.")
    return value


def _identity(path: Path, *, content_key: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    value = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if content_key is not None:
        payload = _read_json(resolved, resolved.name)
        value[content_key] = payload.get(content_key)
    return value


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RevisionDetectionError(f"{label} is not numeric.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise RevisionDetectionError(f"{label} is negative or non-finite.")
    return result


def _verify_policy(path: Path) -> dict[str, Any]:
    policy = _read_json(path, "detector CUDA policy")
    unsigned = dict(policy)
    claimed = unsigned.pop("policy_sha256", None)
    benchmark = policy.get("benchmark")
    ceiling = policy.get("authorized_ceiling")
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or claimed != canonical_json_sha256(unsigned)
        or not isinstance(benchmark, dict)
        or not isinstance(ceiling, dict)
        or benchmark.get("task_indices") != [0, 1]
        or benchmark.get("checkpoint_reuse") is not True
        or int(benchmark.get("cuda_reproducibility_fit_count_per_architecture", -1))
        != 2
        or int(benchmark.get("allowed_failed_fit_retry_count_per_architecture", -1))
        != 1
        or float(benchmark.get("projection_safety_multiplier", -1.0)) != 1.5
        or benchmark.get("full_matrix_budget_gate_required") is not True
        or float(ceiling.get("gpu_hours", -1.0)) != 165.0
        or float(ceiling.get("historical_actual_gpu_hours_floor", -1.0))
        != 62.4783840698
    ):
        raise RevisionDetectionError("Detector CUDA policy budget contract differs.")
    return policy


def _verify_benchmark(
    path: Path, *, task_index: int
) -> tuple[dict[str, Any], dict[str, float]]:
    value = _read_json(path, f"detector benchmark task {task_index}")
    unsigned = dict(value)
    claimed = unsigned.pop("benchmark_sha256", None)
    identity = value.get("benchmark_task_identity")
    phases = value.get("phase_timings_seconds")
    accounting = value.get("gpu_accounting")
    if (
        value.get("schema_version") != BENCHMARK_SCHEMA
        or claimed != canonical_json_sha256(unsigned)
        or int(value.get("benchmark_task_index", -1)) != task_index
        or value.get("device") != "cuda:0"
        or int(value.get("workers", -1)) != 1
        or not isinstance(identity, dict)
        or int(identity.get("ordinal", -1)) != task_index
        or identity.get("detector_name") != ARCHITECTURES[task_index]
        or not isinstance(phases, dict)
        or set(phases) != set(PHASES)
        or not isinstance(accounting, dict)
        or not isinstance(accounting.get("intervals"), list)
        or not accounting["intervals"]
    ):
        raise RevisionDetectionError(
            f"Detector benchmark task {task_index} identity differs."
        )
    phase_values = {
        name: _finite(phases[name], f"task {task_index} phase {name}")
        for name in PHASES
    }
    component_sum = sum(phase_values[name] for name in PHASES if name != "total")
    if abs(component_sum - phase_values["total"]) > max(
        1e-6, 1e-6 * phase_values["total"]
    ):
        raise RevisionDetectionError(
            f"Detector benchmark task {task_index} phase times do not sum."
        )
    measurements = {
        **phase_values,
        "fit_elapsed": _finite(
            value.get("fit_elapsed_seconds"), f"task {task_index} fit elapsed"
        ),
        "fit_non_model_analysis": _finite(
            value.get("fit_non_model_analysis_seconds"),
            f"task {task_index} non-model analysis",
        ),
        "checkpoint": _finite(
            value.get("checkpoint_seconds"), f"task {task_index} checkpoint"
        ),
        "invocation_overhead": _finite(
            value.get("invocation_overhead_seconds"),
            f"task {task_index} invocation overhead",
        ),
        "benchmark_wall": _finite(
            value.get("wall_seconds"), f"task {task_index} benchmark wall"
        ),
        "process_wall": _finite(
            accounting["intervals"][-1].get("elapsed_seconds"),
            f"task {task_index} process GPU wall",
        ),
    }
    return value, measurements


def _verify_plan(benchmarks: Mapping[int, Mapping[str, Any]]) -> tuple[Path, dict]:
    checkpoint_paths = {
        Path(str(value.get("checkpoint_dir", ""))).resolve()
        for value in benchmarks.values()
    }
    if len(checkpoint_paths) != 1:
        raise RevisionDetectionError("Benchmark records do not share one checkpoint root.")
    checkpoint_root = next(iter(checkpoint_paths))
    plan_path = checkpoint_root / "execution_plan.json"
    plan = _read_json(plan_path, "detector execution plan")
    declared_hashes = {
        str(value.get("execution_plan_sha256", "")) for value in benchmarks.values()
    }
    plan_hash = canonical_json_sha256(plan)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or int(plan.get("total_fit_count", -1)) != 56
        or int(plan.get("detector_count", -1)) != 2
        or int(plan.get("split_count", -1)) != 28
        or not isinstance(plan.get("tasks"), list)
        or len(plan["tasks"]) != 56
        or declared_hashes != {plan_hash}
    ):
        raise RevisionDetectionError("Frozen detector execution plan identity differs.")
    for task_index, benchmark in benchmarks.items():
        if benchmark.get("benchmark_task_identity") != plan["tasks"][task_index]:
            raise RevisionDetectionError(
                f"Benchmark task {task_index} differs from execution plan."
            )
    return plan_path, plan


def _task_estimate_seconds(
    task: Mapping[str, Any],
    benchmark_task: Mapping[str, Any],
    measured: Mapping[str, float],
) -> dict[str, float]:
    train_ratio = float(task["train_row_count"]) / float(
        benchmark_task["train_row_count"]
    )
    test_ratio = float(task["test_row_count"]) / float(
        benchmark_task["test_row_count"]
    )
    analysis_ratio = max(1.0, test_ratio)
    component = (
        measured["initialization_and_preprocessing"]
        + measured["training"] * train_ratio
        + measured["trained_state_hashing"]
        + measured["evaluation"] * test_ratio
        + measured["fit_non_model_analysis"] * analysis_ratio
        + measured["checkpoint"]
        + measured["invocation_overhead"]
        + max(0.0, measured["process_wall"] - measured["benchmark_wall"])
    )
    envelope = measured["process_wall"] * max(1.0, train_ratio, test_ratio)
    return {
        "train_row_ratio": train_ratio,
        "test_row_ratio": test_ratio,
        "component_model_seconds": component,
        "process_wall_envelope_seconds": envelope,
        "selected_seconds": max(component, envelope),
    }


def verify_ledger_sources_for_stage(
    *, stage: str, sources: list[Mapping[str, Any]]
) -> dict[str, str]:
    observed = {
        str(row["source_id"]): str(row["component"]) for row in sources
    }
    benchmark_sources = {
        "production_benchmark_task_0": "detector_production_benchmark",
        "production_benchmark_task_1": "detector_production_benchmark",
    }
    if len(observed) != len(sources):
        raise RevisionDetectionError("GPU ledger contains duplicate source IDs.")
    if stage == "post_reproducibility_preproduction":
        if observed != EXPECTED_GPU_LEDGER_SOURCES:
            raise RevisionDetectionError(
                f"GPU ledger sources differ for budget gate stage {stage}."
            )
        return observed
    if (
        any(observed.get(key) != value for key, value in benchmark_sources.items())
        or any(
            EXPECTED_GPU_LEDGER_SOURCES.get(key) != value
            for key, value in observed.items()
        )
    ):
        raise RevisionDetectionError(
            f"GPU ledger sources differ for budget gate stage {stage}."
        )
    for task_index in ARCHITECTURES:
        export_source = f"equivalence_cuda_task_{task_index}"
        repeat_source = f"equivalence_cuda_repeat_task_{task_index}"
        if repeat_source in observed and export_source not in observed:
            raise RevisionDetectionError(
                "GPU ledger records a CUDA repeat before its checkpoint export."
            )
    return observed


def build_gate(
    *,
    stage: str,
    policy_path: Path,
    benchmark_paths: tuple[Path, Path],
    ledger_path: Path,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise RevisionDetectionError(f"Unsupported CUDA budget gate stage: {stage}")
    policy = _verify_policy(policy_path)
    failed_attempt = read_detector_failed_benchmark_attempt(
        DEFAULT_FAILED_ATTEMPT,
        expected_gpu_uuid="GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
    )
    failed_attempt_seconds = _finite(
        failed_attempt["gpu_accounting"]["cumulative_elapsed_seconds"],
        "failed benchmark attempt GPU seconds",
    )
    benchmarks: dict[int, dict[str, Any]] = {}
    measured: dict[int, dict[str, float]] = {}
    for task_index, path in enumerate(benchmark_paths):
        benchmarks[task_index], measured[task_index] = _verify_benchmark(
            path, task_index=task_index
        )
    plan_path, plan = _verify_plan(benchmarks)
    ledger = read_detector_gpu_accounting_ledger(ledger_path)
    observed_sources = verify_ledger_sources_for_stage(
        stage=stage, sources=ledger["sources"]
    )

    safety_multiplier = float(policy["benchmark"]["projection_safety_multiplier"])
    retry_count = int(
        policy["benchmark"]["allowed_failed_fit_retry_count_per_architecture"]
    )
    architecture_rows: dict[str, Any] = {}
    future_before_margin = 0.0
    watchdogs: dict[str, float] = {}
    for task_index, detector_name in ARCHITECTURES.items():
        benchmark_task = benchmarks[task_index]["benchmark_task_identity"]
        tasks = [
            task for task in plan["tasks"] if task["detector_name"] == detector_name
        ]
        if len(tasks) != 28:
            raise RevisionDetectionError(
                f"Frozen plan does not contain 28 tasks for {detector_name}."
            )
        task_estimates = [
            {
                "ordinal": int(task["ordinal"]),
                "split_id": str(task["split_id"]),
                **_task_estimate_seconds(
                    task, benchmark_task, measured[task_index]
                ),
            }
            for task in tasks
        ]
        production_remaining = [
            row for row in task_estimates if row["ordinal"] != task_index
        ]
        production_seconds = sum(
            float(row["selected_seconds"]) for row in production_remaining
        )
        export_pending = (
            f"equivalence_cuda_task_{task_index}" not in observed_sources
        )
        repeat_pending = (
            f"equivalence_cuda_repeat_task_{task_index}"
            not in observed_sources
        )
        reproducibility_seconds = (
            measured[task_index]["process_wall"] if repeat_pending else 0.0
        )
        export_seconds = (
            measured[task_index]["process_wall"] if export_pending else 0.0
        )
        retry_seconds = retry_count * max(
            float(row["selected_seconds"]) for row in task_estimates
        )
        subtotal = (
            production_seconds
            + reproducibility_seconds
            + export_seconds
            + retry_seconds
        )
        future_before_margin += subtotal
        watchdogs[detector_name] = min(
            float(
                policy["ceiling"]["next_fit_upper_seconds_by_detector"][
                    detector_name
                ]
            ),
            safety_multiplier
            * max(float(row["selected_seconds"]) for row in task_estimates),
        )
        architecture_rows[detector_name] = {
            "benchmark_task_index": task_index,
            "benchmark_train_rows": int(benchmark_task["train_row_count"]),
            "benchmark_test_rows": int(benchmark_task["test_row_count"]),
            "measured_seconds": measured[task_index],
            "remaining_production_fit_count": len(production_remaining),
            "remaining_production_seconds": production_seconds,
            "remaining_reproducibility_fit_count": int(repeat_pending),
            "remaining_reproducibility_seconds": reproducibility_seconds,
            "remaining_checkpoint_export_count": int(export_pending),
            "checkpoint_export_allowance_seconds": export_seconds,
            "failed_fit_retry_count": retry_count,
            "failed_fit_retry_allowance_seconds": retry_seconds,
            "subtotal_before_margin_seconds": subtotal,
            "benchmark_derived_fit_watchdog_seconds": watchdogs[detector_name],
            "task_estimates": task_estimates,
        }

    finalization_allowance = max(
        row["process_wall"] for row in measured.values()
    )
    future_before_margin += finalization_allowance
    projected_future_seconds = safety_multiplier * future_before_margin
    historical_floor = float(
        policy["authorized_ceiling"]["historical_actual_gpu_hours_floor"]
    )
    ledger_seconds = float(ledger["cumulative_elapsed_seconds"])
    new_actual_seconds = ledger_seconds + failed_attempt_seconds
    starting_hours = historical_floor + new_actual_seconds / 3600.0
    projected_hours = starting_hours + projected_future_seconds / 3600.0
    hard_ceiling = float(policy["authorized_ceiling"]["gpu_hours"])
    approved = projected_hours <= hard_ceiling
    projection = {
        "hard_ceiling_gpu_hours": hard_ceiling,
        "historical_actual_gpu_hours_floor": historical_floor,
        "new_gpu_ledger_actual_seconds": ledger_seconds,
        "new_gpu_ledger_actual_hours": ledger_seconds / 3600.0,
        "failed_benchmark_attempt_gpu_seconds": failed_attempt_seconds,
        "failed_benchmark_attempt_gpu_hours": failed_attempt_seconds / 3600.0,
        "new_total_actual_gpu_seconds": new_actual_seconds,
        "starting_cumulative_actual_gpu_hours": starting_hours,
        "projection_safety_multiplier": safety_multiplier,
        "architecture_estimates": architecture_rows,
        "finalization_allowance_seconds": finalization_allowance,
        "projected_future_seconds_before_margin": future_before_margin,
        "projected_future_gpu_seconds": projected_future_seconds,
        "projected_future_gpu_hours": projected_future_seconds / 3600.0,
        "projected_cumulative_gpu_hours": projected_hours,
        "projected_remaining_headroom_gpu_hours": hard_ceiling - projected_hours,
        "benchmark_derived_fit_watchdog_seconds": watchdogs,
        "internal_progress_stale_seconds": 900.0,
    }
    inputs = {
        "policy": _identity(policy_path, content_key="policy_sha256"),
        "execution_plan": {
            **_identity(plan_path),
            "execution_plan_sha256": canonical_json_sha256(plan),
        },
        "benchmarks": {
            detector_name: {
                **_identity(benchmark_paths[task_index]),
                "benchmark_sha256": benchmarks[task_index]["benchmark_sha256"],
            }
            for task_index, detector_name in ARCHITECTURES.items()
        },
        "failed_benchmark_attempt": {
            **_identity(DEFAULT_FAILED_ATTEMPT),
            "benchmark_sha256": failed_attempt["benchmark_sha256"],
        },
        "gpu_ledger": {
            **_identity(ledger_path),
            "ledger_sha256": ledger["ledger_sha256"],
            "sources_sha256": ledger["sources_sha256"],
            "intervals_sha256": ledger["intervals_sha256"],
        },
    }
    gate = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_stage": stage,
        "inputs": inputs,
        "inputs_sha256": canonical_json_sha256(inputs),
        "projection": projection,
        "projection_sha256": canonical_json_sha256(projection),
        "decision": {
            "approved": approved,
            "reason": (
                "benchmark_derived_projection_within_hard_ceiling"
                if approved
                else "benchmark_derived_projection_exceeds_hard_ceiling"
            ),
        },
    }
    gate["gate_sha256"] = canonical_json_sha256(gate)
    return gate


def read_gate(path: Path, *, expected_stage: str | None = None) -> dict[str, Any]:
    value = _read_json(path, "detector CUDA budget gate")
    unsigned = dict(value)
    claimed = unsigned.pop("gate_sha256", None)
    if (
        value.get("schema_version") != SCHEMA
        or claimed != canonical_json_sha256(unsigned)
        or value.get("inputs_sha256")
        != canonical_json_sha256(value.get("inputs"))
        or value.get("projection_sha256")
        != canonical_json_sha256(value.get("projection"))
        or value.get("gate_stage") not in STAGES
        or (
            expected_stage is not None
            and value.get("gate_stage") != expected_stage
        )
        or value.get("decision", {}).get("approved") is not True
    ):
        raise RevisionDetectionError(
            "Detector CUDA budget gate is malformed, wrong-stage, or not approved."
        )
    projection = value["projection"]
    if (
        float(projection.get("historical_actual_gpu_hours_floor", -1.0))
        != 62.4783840698
        or float(projection.get("hard_ceiling_gpu_hours", -1.0)) != 165.0
        or float(projection.get("projected_cumulative_gpu_hours", math.inf))
        > 165.0
    ):
        raise RevisionDetectionError("Detector CUDA budget gate ceiling differs.")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise RevisionDetectionError("Detector CUDA budget gate inputs are malformed.")
    identities = [
        inputs.get("policy"),
        inputs.get("execution_plan"),
        inputs.get("failed_benchmark_attempt"),
        inputs.get("gpu_ledger"),
        *(inputs.get("benchmarks", {}) or {}).values(),
    ]
    for identity in identities:
        if not isinstance(identity, dict):
            raise RevisionDetectionError("Detector CUDA budget input is missing.")
        input_path = Path(str(identity.get("path", "")))
        if (
            input_path.is_symlink()
            or not input_path.is_file()
            or file_sha256(input_path) != identity.get("sha256")
            or int(input_path.stat().st_size)
            != int(identity.get("size_bytes", -1))
        ):
            raise RevisionDetectionError(
                "Detector CUDA budget input bytes changed after projection."
            )
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(sorted(STAGES)), required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        default=[],
        help="Repeat exactly twice in task-index order; defaults to task 0/1.",
    )
    parser.add_argument("--gpu-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    benchmarks = tuple(args.benchmark) if args.benchmark else DEFAULT_BENCHMARKS
    if len(benchmarks) != 2:
        raise RevisionDetectionError("Exactly two benchmark paths are required.")
    if args.check:
        gate = read_gate(args.output, expected_stage=args.stage)
    else:
        gate = build_gate(
            stage=args.stage,
            policy_path=args.policy.resolve(),
            benchmark_paths=(benchmarks[0].resolve(), benchmarks[1].resolve()),
            ledger_path=args.gpu_ledger.resolve(),
        )
        history_path = (
            args.output.resolve().parent
            / "budget_gate_history"
            / "{}.{}.json".format(gate["gate_stage"], gate["gate_sha256"])
        )
        atomic_write_json(history_path, gate)
        atomic_write_json(args.output.resolve(), gate)
        if gate["decision"]["approved"] is not True:
            print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
            return 3
        gate = read_gate(args.output.resolve(), expected_stage=args.stage)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["decision"]["approved"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RevisionDetectionError as exc:
        print(f"detector CUDA budget gate error: {exc}", file=sys.stderr)
        raise SystemExit(2)
