#!/usr/bin/env python3
"""Fail-closed post-primary orchestrator for the frozen revision-v1 design.

This is operational tooling.  It does not edit runner, evaluator, protocol, or
configuration source and it never mutates a completed result record.  It waits
for all three primary_v2 shards, runs the supporting and held-out shards on one
pinned GPU, then executes hash-gated CPU analysis/reporting commands supplied
by a separate operational command contract.

Every GPU command is checkpoint/resume aware.  A conservative revised budget
is checked immediately before launch and while the child is running.  The
canonical progress updater remains the source of actual GPU time, throughput,
recovery counts, ETA, and last-checkpoint information.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_detector_cuda_budget_gate import (
    read_gate,
    verify_ledger_sources_for_stage,
)
from rankcloak.revision_artifacts import canonical_json_sha256, file_sha256
from rankcloak.revision_detection import RevisionDetectionError
from rankcloak.revision_detector_execution import (
    detector_finalization_paths,
    detector_gpu_ledger_incorporation_path,
    finalize_detector_candidate_from_closed_status,
    read_detector_cuda_reproducibility_report,
    read_detector_failed_benchmark_attempt,
    read_detector_equivalence_fit_artifact,
    read_detector_finalization_candidate,
    read_detector_gpu_accounting_ledger,
    read_detector_gpu_ledger_incorporation_marker,
    update_detector_gpu_accounting_ledger,
)
from rankcloak.revision_evaluator import verify_completed_runner_shard


GPU_UUID = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"
GPU_OCCUPANCY_TIMEOUT_SECONDS = 15.0
CONFIG_SHA256 = "dc0e7e022036e2681c87ad06446cbebd56d676faf81a0544a55d56375d4eadcd"
PROJECTION_SHA256 = "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
PROTOCOL_REVISION = "payload_fidelity_v2"
RESULT_REVISION = "payload_aware_result_v2"
BUDGET_GPU_HOURS = 165.0
DETECTOR_HISTORICAL_GPU_HOURS_FLOOR = 62.4783840698
DETECTOR_DEVICE = "cuda:0"
DETECTOR_WORKERS = 1
DETECTOR_TOTAL_FITS = 56
DETECTOR_STATUS_SCHEMA = "rankcloak-revision-detector-status-v1"
DETECTOR_FIT_PERMIT_SCHEMA = "rankcloak-revision-detector-fit-permit-v1"
DETECTOR_FIT_PERMIT_RECEIPT_SCHEMA = (
    "rankcloak-revision-detector-fit-permit-receipt-v1"
)
DETECTOR_GPU_DERIVATION = "detector_process_wall_span_v1"
DETECTOR_GPU_COLLECTION_DERIVATION = (
    "nonoverlapping_detector_process_wall_intervals_v1"
)
DETECTOR_STATUS_STALE_SECONDS = 15 * 60.0
DETECTOR_INTERNAL_PROGRESS_STALE_SECONDS = 15 * 60.0
DETECTOR_GRACEFUL_STOP_SECONDS = 30.0
DETECTOR_MONITOR_MAX_POLL_SECONDS = 5.0
DETECTOR_CPU_MONITOR_MAX_POLL_SECONDS = 30.0
DETECTOR_EXECUTION_POLICY_RELATIVE = Path(
    "operations/confirmatory_v2/detector_cuda_policy_v2.json"
)
DETECTOR_EXECUTION_POLICY_SHA256 = (
    "48e01759d4f2214047e7ee9e8a19937a9b45dce91300211b014ee067c34238e7"
)
DETECTOR_EXECUTION_POLICY_CONTENT_SHA256 = (
    "c52a2a6722c84a622727eeb38f2489960b656f91d51fd447dbf2ba40669def1d"
)
DETECTOR_NEXT_FIT_UPPER_SECONDS = {
    "published_textcnn_equivalent": 900.0,
    "deberta_v3_base_classifier": 7_200.0,
}
DETECTOR_BENCHMARK_TASKS = {
    0: "published_textcnn_equivalent",
    1: "deberta_v3_base_classifier",
}
DETECTOR_EQUIVALENCE_ROOT = (
    Path("results/revision_v1/detector_cuda_reproducibility_v2")
)
DETECTOR_FAILED_BENCHMARK_ATTEMPT = (
    DETECTOR_EQUIVALENCE_ROOT
    / "failed_attempts"
    / "task_0_packaging_failure.json"
)
DETECTOR_GPU_LEDGER_SCHEMA = (
    "rankcloak-revision-detector-gpu-accounting-ledger-v1"
)
DETECTOR_GPU_LEDGER_DERIVATION = (
    "append_only_finalized_cuda_equivalence_intervals_v1"
)

MODELS = (
    "llama3_8b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m",
)
QWEN = "qwen2_5_7b_instruct_q4_k_m"
EVALUATOR_BY_GENERATOR = {
    "llama3_8b_instruct_q4_k_m": "qwen2_5_7b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m": "mistral_7b_instruct_v0_3_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m": "llama3_8b_instruct_q4_k_m",
}
RUNNER_COUNTS = {
    "primary_v2": {model: 4800 for model in MODELS},
    "ablation_v2": {model: 624 for model in MODELS},
    "multilingual_v2": {model: 384 for model in MODELS},
    "robustness_v2": {
        "llama3_8b_instruct_q4_k_m": 1200,
        "qwen2_5_7b_instruct_q4_k_m": 1344,
        "mistral_7b_instruct_v0_3_q4_k_m": 1200,
    },
}
RUNNER_EVIDENCE = {
    "primary_v2": "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze",
    "ablation_v2": "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze",
    "multilingual_v2": "secondary_supplementary_multilingual_v2_payload_fidelity_after_manifest_freeze",
    "robustness_v2": "confirmatory_supporting_robustness_v2_payload_fidelity_after_manifest_freeze",
}
EVALUATOR_SOURCE_STAGES = ("primary_v2", "ablation_v2", "multilingual_v2")
EXPECTED_EVALUATOR_TASKS_WITHOUT_STRUCTURAL_UNAVAILABILITY = {
    "primary_v2": 4800,
    "ablation_v2": 576,
    "multilingual_v2": 384,
}
STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS = 48
FROZEN_EVALUATOR_TARGET_UNITS = 17_280
SCOREABLE_EVALUATOR_UNITS = (
    FROZEN_EVALUATOR_TARGET_UNITS - STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS
)
PRIMARY_EVALUATOR_JOIN_TRIALS = 6_480
PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS = 13_320
EVALUATOR_UNAVAILABILITY_SCHEMA = (
    "rankcloak-heldout-evaluator-upstream-unavailability-v1"
)
SUPPORT_ORDER = (
    *(("ablation_v2", model) for model in MODELS),
    *(("multilingual_v2", model) for model in MODELS),
    ("robustness_v2", QWEN),
    ("robustness_v2", "llama3_8b_instruct_q4_k_m"),
    ("robustness_v2", "mistral_7b_instruct_v0_3_q4_k_m"),
)

RESULTS_ROOT = Path("results/revision_v1")
SUPERVISOR_ROOT = RESULTS_ROOT / "supervisor"
STATE_PATH = SUPERVISOR_ROOT / "confirmatory_v2_orchestrator_state.json"
LOCK_PATH = SUPERVISOR_ROOT / "confirmatory_v2_orchestrator.lock"
ERROR_PATH = SUPERVISOR_ROOT / "confirmatory_v2_recovered_errors.jsonl"
EVENT_PATH = SUPERVISOR_ROOT / "confirmatory_v2_events.jsonl"
LOG_ROOT = SUPERVISOR_ROOT / "post_primary_logs"
MARKER_ROOT = SUPERVISOR_ROOT / "post_primary_markers"
PROJECTION_PATH = RESULTS_ROOT / "compute_projection_165h_v2.json"
PROGRESS_PATH = RESULTS_ROOT / "confirmatory_progress_v1.json"
FINAL_PROGRESS_PATH = RESULTS_ROOT / "final_progress_snapshot_v1.json"
EVALUATOR_UNAVAILABILITY_PATH = (
    RESULTS_ROOT
    / "heldout_evaluator"
    / "upstream_dependent_unavailability_v1.json"
)
DEFAULT_COMMAND_CONTRACT = (
    PROJECT_ROOT / "operations" / "confirmatory_v2" / "downstream_commands.json"
)

STATE_SCHEMA = "rankcloak-confirmatory-v2-orchestrator-state-v1"
CONTRACT_SCHEMA = "rankcloak-confirmatory-v2-downstream-command-contract-v1"
MARKER_SCHEMA = "rankcloak-confirmatory-v2-action-marker-v1"
FIGURE_MANIFEST_SCHEMA = "rankcloak-confirmatory-v2-figure-render-v1"
PROGRESS_SOURCE_RACE_WRAPPER = (
    "revision progress failed: Progress sources remained unstable: "
)
PROGRESS_SOURCE_RACE_FIXED_INNERS = (
    "A progress source changed during snapshot construction",
    "An unavailability lineage source changed during the progress scan",
)
PROGRESS_SOURCE_RACE_MESSAGE = (
    PROGRESS_SOURCE_RACE_WRAPPER + PROGRESS_SOURCE_RACE_FIXED_INNERS[0]
)
PROGRESS_SOURCE_RACE_LABELS = (
    "runner plan",
    "runner checkpoint",
    "runner records",
    "runner events",
    "heldout_evaluator plan",
    "heldout_evaluator checkpoint",
    "heldout_evaluator records",
    "heldout_evaluator events",
    "approved 165-GPU-hour projection",
    "detector run manifest",
    "detector product",
    "held-out evaluator unavailability manifest",
    "held-out evaluator unavailability source plan",
    "held-out evaluator unavailability source checkpoint",
    "held-out evaluator unavailability source records",
    "held-out evaluator unavailability source run_identity",
    "held-out evaluator plan",
    "events",
    "checkpoint",
    "records",
    "progress source recheck",
)
PROGRESS_OUTER_ATTEMPTS = 3
PROGRESS_RACE_BACKOFF_SECONDS = (2.0, 4.0)
PROGRESS_SUBPROCESS_TIMEOUT_SECONDS = 120.0


class OrchestratorError(RuntimeError):
    """Base fail-closed orchestration error."""


class MethodologicalHalt(OrchestratorError):
    """A frozen-design or integrity conflict makes continuation invalid."""


class BudgetHalt(OrchestratorError):
    """Actual or conservatively projected GPU use reaches the hard ceiling."""


class InterfaceHalt(OrchestratorError):
    """A parameterized downstream interface or result schema is unavailable."""


class TransientGPUOccupancy(OrchestratorError):
    """The pinned GPU became occupied between the wait and launch checks."""


class DeferrableProgressRefresh(OrchestratorError):
    """An operational progress refresh should defer without action mutation."""


class ProgressSourceRace(DeferrableProgressRefresh):
    """The canonical updater exhausted retries on its exact source-recheck race."""


class ProgressRefreshTimeout(DeferrableProgressRefresh):
    """The canonical updater exceeded its finite operational timeout."""


@dataclass(frozen=True)
class Action:
    action_id: str
    stage: str
    kind: str
    argv: tuple[str, ...]
    output_dir: Path | None = None
    model_id: str | None = None
    generator_model_id: str | None = None
    source_stage: str | None = None
    expected_count: int | None = None
    gpu: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"


def read_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorError(
            "cannot read {} at {}: {}".format(label or "JSON", path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise OrchestratorError("{} must contain a JSON object".format(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OrchestratorError(
                        "{}:{} is not a JSON object".format(path, line_number)
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorError("cannot read JSONL {}: {}".format(path, exc)) from exc
    return rows


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OrchestratorError("refusing atomic write through symlink: {}".format(path))
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise OrchestratorError("temporary path already exists: {}".format(temporary))
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short atomic write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, _json_bytes(dict(value)))


def atomic_publish_once_bytes(path: Path, content: bytes) -> None:
    """Atomically publish immutable bytes without ever replacing a target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("immutable target already exists: {}".format(path))
    temporary = path.with_name(".{}.{}.seal".format(path.name, os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise OrchestratorError("progress-seal temporary path exists: {}".format(temporary))
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable progress write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append by atomic replacement so readers never observe a torn JSON line."""

    path = Path(path)
    existing = b""
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise OrchestratorError("error/event log is not a regular file: {}".format(path))
        existing = path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            raise OrchestratorError("error/event log has a truncated final line: {}".format(path))
    line = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, existing + line)


def acquire_single_instance_lock() -> int:
    path = PROJECT_ROOT / LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise OrchestratorError("orchestrator lock path is unsafe: {}".format(path))
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise OrchestratorError(
            "another confirmatory-v2 orchestrator already holds {}".format(path)
        ) from exc
    return descriptor


def verify_projection(path: Path = PROJECT_ROOT / PROJECTION_PATH) -> dict[str, Any]:
    projection = read_json(path, label="authorized projection")
    unsigned = dict(projection)
    claimed = unsigned.pop("projection_sha256", None)
    if claimed != PROJECTION_SHA256 or canonical_json_sha256(unsigned) != claimed:
        raise MethodologicalHalt("authorized 165-hour projection self-hash mismatch")
    if projection.get("schema_version") != "rankcloak-revision-compute-projection-v1":
        raise MethodologicalHalt("unsupported authorized projection schema")
    if projection.get("budget_gpu_hours") != BUDGET_GPU_HOURS:
        raise MethodologicalHalt("projection is not bound to the 165 GPU-hour ceiling")
    decision = projection.get("decision")
    if not isinstance(decision, dict) or decision.get("go") is not True or decision.get("status") != "go_within_budget":
        raise MethodologicalHalt("authorized projection is not GO")
    totals = projection.get("totals")
    if not isinstance(totals, dict) or float(totals.get("upper_gpu_hours", 999.0)) >= BUDGET_GPU_HOURS:
        raise MethodologicalHalt("authorized projection lacks positive hard-ceiling headroom")
    frozen = projection.get("frozen_plan")
    if not isinstance(frozen, dict):
        raise MethodologicalHalt("projection lacks its frozen plan")
    for stage, expected in RUNNER_COUNTS.items():
        row = frozen.get(stage)
        counts = row.get("model_counts") if isinstance(row, dict) else None
        if counts != expected:
            raise MethodologicalHalt("frozen plan count mismatch for {}".format(stage))
    evaluator_targets = [
        int(row["target_work_units"])
        for row in projection.get("stage_totals", [])
        if isinstance(row, dict) and row.get("stage") == "heldout_evaluator"
    ]
    if evaluator_targets != [FROZEN_EVALUATOR_TARGET_UNITS]:
        raise MethodologicalHalt("frozen evaluator target is not exactly 17,280")
    return projection


def _relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _runner_argv(stage: str, model: str, output_dir: Path) -> tuple[str, ...]:
    argv = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_matrix.py",
        "--stage",
        stage,
        "--model",
        model,
        "--gpu-uuid",
        GPU_UUID,
        "--context",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--output-dir",
        _relative(output_dir),
    ]
    if stage == "robustness_v2":
        argv.extend(
            [
                "--primary-results-root",
                _relative(PROJECT_ROOT / RESULTS_ROOT / "primary_v2"),
                "--ablation-results-root",
                _relative(PROJECT_ROOT / RESULTS_ROOT / "ablation_v2"),
                "--robustness-results-root",
                _relative(PROJECT_ROOT / RESULTS_ROOT / "robustness_v2"),
            ]
        )
    return tuple(argv)


def _evaluator_argv(
    source_stage: str, generator: str, evaluator: str, output_dir: Path
) -> tuple[str, ...]:
    return (
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_evaluator.py",
        "--evaluator-model",
        evaluator,
        "--source-stage",
        source_stage,
        "--source-results-root",
        _relative(PROJECT_ROOT / RESULTS_ROOT),
        "--gpu-uuid",
        GPU_UUID,
        "--context",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--output-dir",
        _relative(output_dir),
    )


def build_gpu_actions() -> list[Action]:
    actions: list[Action] = []
    for stage, model in SUPPORT_ORDER:
        output = PROJECT_ROOT / RESULTS_ROOT / stage / model
        actions.append(
            Action(
                action_id="runner:{}:{}".format(stage, model),
                stage=stage,
                kind="runner",
                argv=_runner_argv(stage, model, output),
                output_dir=output,
                model_id=model,
                expected_count=RUNNER_COUNTS[stage][model],
                gpu=True,
            )
        )
    for source_stage in EVALUATOR_SOURCE_STAGES:
        for generator in MODELS:
            evaluator = EVALUATOR_BY_GENERATOR[generator]
            expected_count = EXPECTED_EVALUATOR_TASKS_WITHOUT_STRUCTURAL_UNAVAILABILITY[
                source_stage
            ]
            if source_stage == "ablation_v2" and generator == (
                "mistral_7b_instruct_v0_3_q4_k_m"
            ):
                expected_count -= STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS
            output = (
                PROJECT_ROOT
                / RESULTS_ROOT
                / "heldout_evaluator"
                / source_stage
                / evaluator
            )
            base = _evaluator_argv(source_stage, generator, evaluator, output)
            action_id = "evaluator:{}:{}".format(source_stage, evaluator)
            actions.append(
                Action(
                    action_id=action_id,
                    stage="heldout_evaluator",
                    kind="evaluator",
                    argv=base,
                    output_dir=output,
                    model_id=evaluator,
                    generator_model_id=generator,
                    source_stage=source_stage,
                    expected_count=expected_count,
                    gpu=True,
                )
            )
            actions.append(
                Action(
                    action_id=action_id + ":export",
                    stage="heldout_evaluator",
                    kind="evaluator_export",
                    argv=base + ("--resume",),
                    output_dir=output,
                    model_id=evaluator,
                    generator_model_id=generator,
                    source_stage=source_stage,
                    expected_count=expected_count,
                    gpu=False,
                )
            )
    return actions


def _identity_argument(identity: Mapping[str, Any], name: str) -> str | None:
    prefix = name + "="
    matches = [
        str(value)[len(prefix) :]
        for value in identity.get("command_line_args", [])
        if str(value).startswith(prefix)
    ]
    return matches[0] if len(matches) == 1 else None


def _unit_id(row: Mapping[str, Any], evaluator: bool) -> str:
    return str(row.get("evaluation_id") if evaluator else row.get("work_id"))


def shard_status(action: Action) -> str:
    """Return not_started/incomplete/complete after strict durable checks."""

    if action.output_dir is None:
        raise OrchestratorError("shard action has no output directory")
    output = action.output_dir
    checkpoint_path = output / "checkpoint.json"
    if not checkpoint_path.exists():
        if output.exists() and any(output.iterdir()):
            raise MethodologicalHalt("nonempty shard lacks checkpoint: {}".format(output))
        return "not_started"
    if checkpoint_path.is_symlink():
        raise MethodologicalHalt("checkpoint is a symlink: {}".format(checkpoint_path))
    evaluator = action.kind in {"evaluator", "evaluator_export"}
    plan_path = output / "plan.jsonl"
    identity_path = output / "run_identity.json"
    if not plan_path.is_file() or not identity_path.is_file():
        raise MethodologicalHalt("checkpointed shard lacks plan/identity: {}".format(output))
    plan = read_jsonl(plan_path)
    expected_count = action.expected_count
    if expected_count is not None and len(plan) != expected_count:
        raise MethodologicalHalt(
            "{} plan count {} differs from authorized expected {}".format(
                action.action_id, len(plan), expected_count
            )
        )
    identifiers = [_unit_id(row, evaluator) for row in plan]
    if any(value in {"", "None"} for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise MethodologicalHalt("missing or duplicate plan identity in {}".format(output))
    checkpoint = read_json(checkpoint_path, label="checkpoint")
    if checkpoint.get("planned_trial_count") != len(plan):
        raise MethodologicalHalt("checkpoint/plan count mismatch in {}".format(output))
    plan_ids_sha256 = canonical_json_sha256(identifiers)
    if checkpoint.get("planned_trial_ids_sha256") != plan_ids_sha256:
        raise MethodologicalHalt("checkpoint/plan identity hash mismatch in {}".format(output))
    if checkpoint.get("config_manifest_sha256") != CONFIG_SHA256:
        raise MethodologicalHalt("checkpoint config hash mismatch in {}".format(output))
    completed = list(map(str, checkpoint.get("completed_trial_ids", [])))
    failed = list(map(str, checkpoint.get("failed_trial_ids", [])))
    if len(completed) != len(set(completed)) or len(failed) != len(set(failed)):
        raise MethodologicalHalt("checkpoint contains duplicate terminal IDs")
    if set(completed) & set(failed) or not set(completed).issubset(identifiers) or not set(failed).issubset(identifiers):
        raise MethodologicalHalt("checkpoint terminal sets are inconsistent")
    identity = read_json(identity_path, label="run identity")
    claimed = identity.get("run_identity_sha256")
    unsigned = dict(identity)
    unsigned.pop("run_identity_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise MethodologicalHalt("run identity self-hash mismatch in {}".format(output))
    if identity.get("protocol_contract_revision") != PROTOCOL_REVISION or identity.get("result_schema_revision") != RESULT_REVISION:
        raise MethodologicalHalt("payload-fidelity contract mismatch in {}".format(output))
    if identity.get("config_manifest_sha256") != CONFIG_SHA256:
        raise MethodologicalHalt("run identity config hash mismatch in {}".format(output))
    if (
        identity.get("planned_trial_count") != len(plan)
        or identity.get("planned_trial_ids_sha256") != plan_ids_sha256
    ):
        raise MethodologicalHalt("run identity/plan hash mismatch in {}".format(output))
    expected_backend_arguments = {
        "context_limit": "4096",
        "gpu_uuid": GPU_UUID,
        "n_gpu_layers": "-1",
        "protocol_contract_revision": PROTOCOL_REVISION,
        "result_schema_revision": RESULT_REVISION,
    }
    for argument, expected_value in expected_backend_arguments.items():
        if _identity_argument(identity, argument) != expected_value:
            raise MethodologicalHalt(
                "run identity {} mismatch in {}".format(argument, output)
            )
    if evaluator:
        if _identity_argument(identity, "evaluator_model_id") != action.model_id:
            raise MethodologicalHalt("evaluator identity mismatch in {}".format(output))
        if _identity_argument(identity, "generator_model_id") != action.generator_model_id:
            raise MethodologicalHalt("generator identity mismatch in {}".format(output))
        if set(str(row.get("source_stage")) for row in plan) != {action.source_stage}:
            raise MethodologicalHalt("evaluator plan contains a different source stage")
        if _identity_argument(identity, "source_stages") != action.source_stage:
            raise MethodologicalHalt("evaluator source-stage identity mismatch")
    else:
        if _identity_argument(identity, "stage") != action.stage or _identity_argument(identity, "model_id") != action.model_id:
            raise MethodologicalHalt("runner stage/model identity mismatch in {}".format(output))
        if any(str(row.get("model_id")) != action.model_id for row in plan):
            raise MethodologicalHalt("runner plan is not bound to one expected model")
        expected_evidence = RUNNER_EVIDENCE[action.stage]
        if (
            _identity_argument(identity, "evidence_status") != expected_evidence
            or {str(row.get("evidence_status")) for row in plan}
            != {expected_evidence}
        ):
            raise MethodologicalHalt("runner evidence identity mismatch in {}".format(output))
    if len(completed) != len(plan) or failed:
        return "incomplete"
    records_path = output / "records.jsonl"
    records = read_jsonl(records_path)
    completion_counts: dict[str, int] = {}
    completed_records: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("execution_status") != "completed":
            continue
        identifier = _unit_id(row, evaluator)
        completion_counts[identifier] = completion_counts.get(identifier, 0) + 1
        completed_records[identifier] = row
    if set(completion_counts) != set(identifiers) or any(
        count != 1 for count in completion_counts.values()
    ):
        raise MethodologicalHalt("complete checkpoint lacks one-to-one durable completions")
    if any(
        row.get("protocol_contract_revision") != PROTOCOL_REVISION
        or row.get("result_schema_revision") != RESULT_REVISION
        for row in completed_records.values()
    ):
        raise MethodologicalHalt("completed records violate the payload-fidelity contract")
    if evaluator:
        verify_evaluator_exports(action)
    elif action.stage in EVALUATOR_SOURCE_STAGES:
        # Reuse the public immutable-source verifier for all stages that feed
        # held-out scoring.  It checks source/model/runtime manifests, ordered
        # identities, and exact one-to-one completion.
        verify_completed_runner_shard(
            output, action.stage, str(action.model_id), CONFIG_SHA256
        )
    return "complete"


def verify_evaluator_exports(action: Action) -> dict[str, Any]:
    assert action.output_dir is not None
    path = action.output_dir / "features_manifest.json"
    manifest = read_json(path, label="evaluator feature manifest")
    if (
        manifest.get("schema_version") != "2.0"
        or manifest.get("manifest_type") != "heldout_evaluator_feature_table"
        or manifest.get("confirmatory_pooling_eligible") is not True
    ):
        raise MethodologicalHalt("invalid confirmatory evaluator feature manifest: {}".format(path))
    expected_evidence = {
        "primary_v2": "confirmatory_heldout_evaluator_primary_v2_payload_fidelity_after_source_manifest_freeze",
        "ablation_v2": "confirmatory_supporting_heldout_evaluator_ablation_v2_payload_fidelity_after_source_manifest_freeze",
        "multilingual_v2": "secondary_supplementary_heldout_evaluator_multilingual_v2_payload_fidelity_after_source_manifest_freeze",
    }[str(action.source_stage)]
    if manifest.get("evidence_statuses") != [expected_evidence]:
        raise MethodologicalHalt("evaluator exports cross evidence-stage boundaries")
    for path_key, hash_key in (
        ("path", "sha256"),
        ("continuous_quality_path", "continuous_quality_sha256"),
    ):
        declared = Path(str(manifest.get(path_key, "")))
        candidate = declared if declared.is_absolute() else action.output_dir / declared.name
        if not candidate.is_file() or candidate.is_symlink() or file_sha256(candidate) != manifest.get(hash_key):
            raise MethodologicalHalt("evaluator export identity mismatch: {}".format(candidate))
    if int(manifest.get("row_count", -1)) != int(action.expected_count or -1):
        raise MethodologicalHalt("evaluator feature row count differs from the exact plan")
    return manifest


def primary_gate_status() -> tuple[bool, list[str]]:
    incomplete: list[str] = []
    for model in MODELS:
        output = PROJECT_ROOT / RESULTS_ROOT / "primary_v2" / model
        action = Action(
            action_id="primary_v2:{}".format(model),
            stage="primary_v2",
            kind="runner",
            argv=(),
            output_dir=output,
            model_id=model,
            expected_count=RUNNER_COUNTS["primary_v2"][model],
            gpu=True,
        )
        if shard_status(action) != "complete":
            incomplete.append(model)
    return not incomplete, incomplete


def marker_path(action: Action) -> Path:
    safe = action.action_id.replace(":", "__").replace("/", "_")
    return PROJECT_ROOT / MARKER_ROOT / (safe + ".json")


def export_marker_valid(action: Action) -> bool:
    path = marker_path(action)
    if not path.exists():
        return False
    marker = read_json(path, label="action marker")
    unsigned = dict(marker)
    claimed = unsigned.pop("marker_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise MethodologicalHalt("action marker self-hash mismatch: {}".format(path))
    if (
        marker.get("schema_version") != MARKER_SCHEMA
        or marker.get("action_id") != action.action_id
        or marker.get("argv_sha256") != canonical_json_sha256(list(action.argv))
    ):
        raise MethodologicalHalt("action marker identity mismatch: {}".format(path))
    manifest = verify_evaluator_exports(action)
    if marker.get("features_manifest_sha256") != file_sha256(
        action.output_dir / "features_manifest.json"  # type: ignore[operator]
    ):
        raise MethodologicalHalt("export marker no longer matches evaluator products")
    return bool(manifest)


def write_export_marker(action: Action) -> None:
    assert action.output_dir is not None
    value: dict[str, Any] = {
        "schema_version": MARKER_SCHEMA,
        "action_id": action.action_id,
        "completed_at": utc_now(),
        "argv": list(action.argv),
        "argv_sha256": canonical_json_sha256(list(action.argv)),
        "features_manifest_sha256": file_sha256(action.output_dir / "features_manifest.json"),
        "semantics": "identical_resume_export_pass_completed_without_pending_trials",
    }
    value["marker_sha256"] = canonical_json_sha256(value)
    atomic_write_json(marker_path(action), value)


def _record_stratum(task: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    record_type = str(record.get("record_type"))
    if record_type in {"condition_unavailable", "dependent_unavailable"}:
        return "planned_unavailable:" + record_type
    kind = str(task.get("work_kind"))
    if kind == "control":
        return "control:" + str(task.get("control_view"))
    if kind == "rankcloak":
        return "rankcloak:" + str(task.get("protocol_variant"))
    if kind == "reference":
        return "reference"
    if kind == "robustness_transform":
        return "robustness_transform"
    if kind == "robustness_decode":
        replay_mode = str(task.get("replay_mode"))
        if replay_mode == "greedy_leadin_regeneration":
            projected_replay_mode = replay_mode
        elif replay_mode in {
            "canonicalized_text_retokenized",
            "cross_model_text_retokenized",
            "detokenized_text_retokenized",
            "transformed_text_retokenized",
        }:
            # The authorized projection deliberately pools every frozen
            # text-retokenization replay under its smoke timing stratum.
            projected_replay_mode = "text_retokenized"
        else:
            raise MethodologicalHalt(
                "cannot map projected robustness replay_mode={!r}".format(
                    replay_mode
                )
            )
        return "robustness_decode:{}:{}".format(
            projected_replay_mode, task.get("protocol_variant")
        )
    raise MethodologicalHalt("cannot map projected stratum for work_kind={!r}".format(kind))


def _load_events(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def projected_consumption(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Compute projected upper seconds replaced by durable actual GPU time."""

    rows = [
        row
        for row in projection.get("projection_rows", [])
        if isinstance(row, dict)
        and row.get("resource_class") == "gpu"
        and row.get("stage") in {*RUNNER_COUNTS, "heldout_evaluator"}
    ]
    expected_keys = {
        (stage, model)
        for stage, counts in RUNNER_COUNTS.items()
        for model in counts
    } | {("heldout_evaluator", model) for model in MODELS}
    indexed = {(str(row["stage"]), str(row["model_id"])): row for row in rows}
    if set(indexed) != expected_keys:
        raise MethodologicalHalt("projection GPU row identity set is incomplete")

    consumed_seconds = 0.0
    consumption_rows: list[dict[str, Any]] = []
    for (stage, model), row in sorted(indexed.items()):
        matched: dict[str, int] = {}
        load_observed_count = 0
        if stage == "heldout_evaluator":
            completed = 0
            for source_stage in EVALUATOR_SOURCE_STAGES:
                shard = PROJECT_ROOT / RESULTS_ROOT / "heldout_evaluator" / source_stage / model
                load_observed_count += sum(
                    event.get("event") == "evaluator_model_loaded"
                    for event in _load_events(shard / "events.jsonl")
                )
                checkpoint_path = shard / "checkpoint.json"
                if checkpoint_path.exists():
                    checkpoint = read_json(checkpoint_path)
                    completed += len(set(map(str, checkpoint.get("completed_trial_ids", []))))
            if completed > int(row.get("target_work_units", -1)):
                raise MethodologicalHalt("evaluator completions exceed authorized projection")
            rate = row.get("rate")
            if not isinstance(rate, dict):
                raise MethodologicalHalt("evaluator projection rate is missing")
            task_seconds = completed * float(rate["upper_seconds_per_unit"])
            consumed_seconds += task_seconds
            matched["all_evaluator_tasks"] = completed
        else:
            shard = PROJECT_ROOT / RESULTS_ROOT / stage / model
            load_observed_count = sum(
                event.get("event") == "model_loaded"
                for event in _load_events(shard / "events.jsonl")
            )
            checkpoint_path = shard / "checkpoint.json"
            task_seconds = 0.0
            if checkpoint_path.exists():
                plan = read_jsonl(shard / "plan.jsonl")
                plan_by_id = {str(item.get("work_id")): item for item in plan}
                completed_ids = set(
                    map(str, read_json(checkpoint_path).get("completed_trial_ids", []))
                )
                completions: dict[str, dict[str, Any]] = {}
                for record in read_jsonl(shard / "records.jsonl"):
                    work_id = str(record.get("work_id"))
                    if record.get("execution_status") == "completed":
                        completions[work_id] = record
                if not completed_ids.issubset(completions):
                    raise MethodologicalHalt("budget scan found checkpoint/record disagreement")
                upper_by_stratum = {
                    str(item["stratum"]): item
                    for item in row.get("strata", [])
                    if isinstance(item, dict)
                }
                for work_id in completed_ids:
                    stratum = _record_stratum(plan_by_id[work_id], completions[work_id])
                    if stratum not in upper_by_stratum:
                        raise MethodologicalHalt(
                            "authorized projection has no stratum for {}/{}/{}".format(
                                stage, model, stratum
                            )
                        )
                    matched[stratum] = matched.get(stratum, 0) + 1
                    task_seconds += float(upper_by_stratum[stratum]["upper_seconds_per_unit"])
                for stratum, count in matched.items():
                    if count > int(upper_by_stratum[stratum].get("target_units", -1)):
                        raise MethodologicalHalt(
                            "durable {} count exceeds projected target for {}/{}/{}".format(
                                count, stage, model, stratum
                            )
                        )
            consumed_seconds += task_seconds
        load_seconds = 0.0
        planned_load_count = len(EVALUATOR_SOURCE_STAGES) if stage == "heldout_evaluator" else 1
        projected_loads_consumed = min(load_observed_count, planned_load_count)
        if projected_loads_consumed:
            load = row.get("model_load")
            if not isinstance(load, dict):
                raise MethodologicalHalt("projected model-load row is missing")
            load_seconds = (
                projected_loads_consumed * float(load["upper_seconds_per_unit"])
            )
            consumed_seconds += load_seconds
        consumption_rows.append(
            {
                "stage": stage,
                "model_id": model,
                "projected_upper_seconds_consumed": task_seconds + load_seconds,
                "observed_model_load_events": load_observed_count,
                "projected_model_loads_consumed": projected_loads_consumed,
                "planned_model_loads": planned_load_count,
                "completed_by_stratum": matched,
            }
        )
    return {"seconds": consumed_seconds, "rows": consumption_rows}


def operational_projection_adjustment_seconds(
    projection: Mapping[str, Any]
) -> float:
    """Add the six loads required by stage-isolated evaluator processes.

    The authorized workload projection aggregates all three evaluator source
    stages into one row per evaluator family and therefore includes one load
    per family.  Evidence isolation requires three processes per family.  The
    task rates are unchanged; only two additional conservative load bounds per
    evaluator must be added before replacing projected work with actual time.
    """

    rows = [
        row
        for row in projection.get("projection_rows", [])
        if isinstance(row, dict)
        and row.get("resource_class") == "gpu"
        and row.get("stage") == "heldout_evaluator"
    ]
    if {str(row.get("model_id")) for row in rows} != set(MODELS):
        raise MethodologicalHalt("evaluator load adjustment lacks all model rows")
    adjustment = 0.0
    for row in rows:
        load = row.get("model_load")
        if not isinstance(load, dict):
            raise MethodologicalHalt("evaluator projection lacks a load bound")
        adjustment += (len(EVALUATOR_SOURCE_STAGES) - 1) * float(
            load["upper_seconds_per_unit"]
        )
    return adjustment


def detector_failed_benchmark_gpu_seconds() -> float:
    """Return signed failed-attempt process-wall charge outside canonical progress."""

    path = PROJECT_ROOT / DETECTOR_FAILED_BENCHMARK_ATTEMPT
    if not path.exists():
        return 0.0
    try:
        artifact = read_detector_failed_benchmark_attempt(
            path, expected_gpu_uuid=GPU_UUID
        )
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "failed detector benchmark accounting artifact is invalid: {}".format(
                exc
            )
        ) from exc
    seconds = float(
        artifact["gpu_accounting"]["cumulative_elapsed_seconds"]
    )
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise MethodologicalHalt(
            "failed detector benchmark accounting seconds are invalid"
        )
    return seconds


def calculate_budget(
    projection: Mapping[str, Any],
    progress: Mapping[str, Any],
    *,
    live_detector_gpu_seconds: float = 0.0,
    live_detector_remaining_seconds: float = 0.0,
) -> dict[str, Any]:
    """Combine canonical actuals with caller-verified unaccounted detector time."""

    consumption = projected_consumption(projection)
    gpu = progress.get("gpu")
    if not isinstance(gpu, dict):
        raise MethodologicalHalt("canonical progress lacks GPU accounting")
    canonical_actual_confirmatory = float(
        gpu.get("monitored_confirmatory_gpu_hours", -1.0)
    )
    canonical_cumulative_reported = float(
        gpu.get("cumulative_actual_gpu_hours", -1.0)
    )
    canonical_cumulative_actual = max(
        canonical_cumulative_reported,
        DETECTOR_HISTORICAL_GPU_HOURS_FLOOR,
    )
    failed_detector_seconds = detector_failed_benchmark_gpu_seconds()
    live_detector_seconds = float(live_detector_gpu_seconds)
    remaining_detector_seconds = float(live_detector_remaining_seconds)
    if (
        canonical_actual_confirmatory < 0
        or canonical_cumulative_reported < 0
        or not math.isfinite(live_detector_seconds)
        or live_detector_seconds < 0
        or not math.isfinite(remaining_detector_seconds)
        or remaining_detector_seconds < 0
    ):
        raise MethodologicalHalt("canonical progress contains negative GPU accounting")
    # `_detector_unaccounted_gpu_seconds` compares exact PID/start/start-time
    # identities against canonical progress before every detector call.  A
    # coarse "any detector interval exists" rejection is invalid once the
    # append-only pre-final ledger is canonical while a later benchmark,
    # repeat, or production process is live.
    failed_detector_hours = failed_detector_seconds / 3600.0
    live_detector_hours = live_detector_seconds / 3600.0
    remaining_detector_hours = remaining_detector_seconds / 3600.0
    actual_confirmatory = (
        canonical_actual_confirmatory
        + failed_detector_hours
        + live_detector_hours
    )
    cumulative_actual = (
        canonical_cumulative_actual
        + failed_detector_hours
        + live_detector_hours
    )
    authorized_baseline = float(projection["totals"]["upper_gpu_hours"])
    operational_adjustment = operational_projection_adjustment_seconds(
        projection
    ) / 3600.0
    baseline = authorized_baseline + operational_adjustment
    consumed_hours = float(consumption["seconds"]) / 3600.0
    revised = max(
        baseline
        - consumed_hours
        + actual_confirmatory
        + remaining_detector_hours,
        cumulative_actual + remaining_detector_hours,
    )
    return {
        "hard_ceiling_gpu_hours": BUDGET_GPU_HOURS,
        "authorized_baseline_upper_gpu_hours": authorized_baseline,
        "stage_isolated_evaluator_load_adjustment_gpu_hours": operational_adjustment,
        "baseline_upper_gpu_hours": baseline,
        "projected_upper_consumed_gpu_hours": consumed_hours,
        "canonical_actual_confirmatory_gpu_hours": canonical_actual_confirmatory,
        "canonical_cumulative_reported_gpu_hours": canonical_cumulative_reported,
        "historical_actual_gpu_hours_floor": DETECTOR_HISTORICAL_GPU_HOURS_FLOOR,
        "failed_detector_benchmark_gpu_seconds": failed_detector_seconds,
        "failed_detector_benchmark_gpu_hours": failed_detector_hours,
        "canonical_cumulative_actual_gpu_hours": canonical_cumulative_actual,
        "live_detector_gpu_seconds": live_detector_seconds,
        "live_detector_gpu_hours": live_detector_hours,
        "live_detector_remaining_seconds": remaining_detector_seconds,
        "live_detector_remaining_gpu_hours": remaining_detector_hours,
        "actual_confirmatory_gpu_hours": actual_confirmatory,
        "cumulative_actual_gpu_hours": cumulative_actual,
        "revised_upper_gpu_hours": revised,
        "revised_headroom_gpu_hours": BUDGET_GPU_HOURS - revised,
        "consumption_rows": consumption["rows"],
    }


def canonical_progress_source_race(
    completed: subprocess.CompletedProcess[str],
) -> str | None:
    """Return the exact safe canonical race line, otherwise fail classification."""

    if completed.returncode != 1 or completed.stdout != "":
        return None
    stderr = completed.stderr
    if stderr.endswith("\r\n"):
        line = stderr[:-2]
    elif stderr.endswith("\n"):
        line = stderr[:-1]
    else:
        return None
    if "\r" in line or "\n" in line:
        return None
    if line in {
        PROGRESS_SOURCE_RACE_WRAPPER + inner
        for inner in PROGRESS_SOURCE_RACE_FIXED_INNERS
    }:
        return line
    if not line.startswith(PROGRESS_SOURCE_RACE_WRAPPER):
        return None
    inner = line[len(PROGRESS_SOURCE_RACE_WRAPPER) :]
    for label in PROGRESS_SOURCE_RACE_LABELS:
        marker = "{} changed while being read: ".format(label)
        if not inner.startswith(marker):
            continue
        path_text = inner[len(marker) :]
        candidate = Path(path_text)
        if not candidate.is_absolute() or path_text != str(candidate):
            return None
        try:
            results_root = (PROJECT_ROOT / RESULTS_ROOT).resolve(strict=True)
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(results_root)
        except (OSError, RuntimeError, ValueError):
            return None
        if not relative.parts or resolved != candidate:
            return None
        return line
    return None


def exact_progress_source_race(completed: subprocess.CompletedProcess[str]) -> bool:
    return canonical_progress_source_race(completed) is not None


def refresh_progress() -> dict[str, Any]:
    race_messages: list[str] = []
    for attempt in range(1, PROGRESS_OUTER_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                [
                    str(PROJECT_ROOT / ".venv/bin/python"),
                    "scripts/update_revision_progress.py",
                    "--write",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=PROGRESS_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProgressRefreshTimeout(
                "canonical progress updater timed out after {} seconds".format(
                    PROGRESS_SUBPROCESS_TIMEOUT_SECONDS
                )
            ) from exc
        if completed.returncode == 0:
            break
        source_race = canonical_progress_source_race(completed)
        if source_race is None:
            raise OrchestratorError(
                "canonical progress refresh failed: {}".format(
                    (completed.stderr or completed.stdout).strip()[-4000:]
                )
            )
        race_messages.append(source_race)
        if attempt < PROGRESS_OUTER_ATTEMPTS:
            time.sleep(PROGRESS_RACE_BACKOFF_SECONDS[attempt - 1])
            continue
        raise ProgressSourceRace(
            "canonical progress refresh deferred after {} outer attempts: {}".format(
                PROGRESS_OUTER_ATTEMPTS, race_messages[-1]
            )
        )
    else:
        raise AssertionError("unreachable progress refresh loop")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OrchestratorError("canonical progress updater returned invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "rankcloak-revision-confirmatory-progress-v1":
        raise OrchestratorError("canonical progress updater returned an unexpected schema")
    unsigned = dict(value)
    claimed = unsigned.pop("progress_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise MethodologicalHalt("canonical progress self-hash mismatch")
    if race_messages:
        emit_event(
            "canonical_progress_unstable_source_race_recovered",
            source_races_recovered=len(race_messages),
            last_source_race=race_messages[-1],
            progress_refresh_outer_attempts=attempt,
            progress_refresh_backoff_seconds=list(
                PROGRESS_RACE_BACKOFF_SECONDS[: max(0, attempt - 1)]
            ),
            action_retry_count_changed=False,
        )
    return value


def published_progress_after_source_race() -> dict[str, Any]:
    """Load the last atomically published snapshot solely for the ceiling gate.

    This is reachable only after the exact updater source race above.  It does
    not claim that the prior snapshot is current; it supplies the last durable
    accounting while the next refresh is deferred.
    """

    path = PROJECT_ROOT / PROGRESS_PATH
    if not path.is_file() or path.is_symlink():
        raise OrchestratorError(
            "canonical progress race occurred without a safe published snapshot"
        )
    value = read_json(path, label="last published canonical progress")
    if value.get("schema_version") != "rankcloak-revision-confirmatory-progress-v1":
        raise OrchestratorError(
            "last published canonical progress has an unexpected schema"
        )
    unsigned = dict(value)
    claimed = unsigned.pop("progress_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise MethodologicalHalt("last published canonical progress self-hash mismatch")
    return value


def verify_final_progress_snapshot() -> dict[str, Any]:
    path = PROJECT_ROOT / FINAL_PROGRESS_PATH
    if not path.is_file() or path.is_symlink():
        raise InterfaceHalt("immutable final progress snapshot is absent/unsafe")
    completed = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "scripts/update_revision_progress.py",
            "--check",
            "--output",
            str(path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MethodologicalHalt(
            "immutable final progress snapshot no longer verifies: {}".format(
                (completed.stderr or completed.stdout).strip()[-4000:]
            )
        )
    try:
        verification = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InterfaceHalt("final progress verifier returned invalid JSON") from exc
    value = read_json(path, label="immutable final progress snapshot")
    unsigned = dict(value)
    claimed = unsigned.pop("progress_sha256", None)
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "ok"
        or Path(str(verification.get("path", ""))).resolve() != path.resolve()
        or verification.get("progress_sha256") != claimed
        or verification.get("generated_at") != value.get("generated_at")
        or verification.get("counts") != value.get("counts")
        or value.get("schema_version")
        != "rankcloak-revision-confirmatory-progress-v1"
        or canonical_json_sha256(unsigned) != claimed
    ):
        raise MethodologicalHalt("immutable final progress self-hash mismatch")
    require_canonical_evaluator_completion(value)
    return value


def seal_final_progress_snapshot(progress: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the last canonical snapshot once, immediately before reporting."""

    require_canonical_evaluator_completion(progress)
    destination = PROJECT_ROOT / FINAL_PROGRESS_PATH
    if destination.exists() or destination.is_symlink():
        return verify_final_progress_snapshot()
    source = PROJECT_ROOT / PROGRESS_PATH
    if not source.is_file() or source.is_symlink():
        raise InterfaceHalt("canonical progress file is absent/unsafe before sealing")
    payload = source.read_bytes()
    try:
        observed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InterfaceHalt("canonical progress file is invalid before sealing") from exc
    if observed != dict(progress):
        raise MethodologicalHalt("canonical progress changed before immutable sealing")
    try:
        atomic_publish_once_bytes(destination, payload)
    except FileExistsError:
        # Another exact supervisor may have won the no-overwrite publication
        # race. Its bytes must pass the full source-bound verifier.
        pass
    sealed = verify_final_progress_snapshot()
    emit_event(
        "final_progress_snapshot_sealed",
        path=str(destination.resolve()),
        sha256=file_sha256(destination),
        progress_sha256=sealed["progress_sha256"],
    )
    return sealed


def operational_progress() -> dict[str, Any]:
    """Never rewrite canonical progress after the immutable final seal exists."""

    path = PROJECT_ROOT / FINAL_PROGRESS_PATH
    if path.exists() or path.is_symlink():
        return verify_final_progress_snapshot()
    return refresh_progress()


def enforce_budget(budget: Mapping[str, Any]) -> None:
    actual = float(budget["cumulative_actual_gpu_hours"])
    revised = float(budget["revised_upper_gpu_hours"])
    if not math.isfinite(actual) or not math.isfinite(revised):
        raise MethodologicalHalt("GPU accounting contains a non-finite value")
    if actual >= BUDGET_GPU_HOURS:
        raise BudgetHalt("actual cumulative GPU use reached the 165-hour hard ceiling")
    if revised > BUDGET_GPU_HOURS:
        raise BudgetHalt("revised conservative projection exceeds 165 GPU-hours")


def enforce_detector_next_fit_reserve(
    budget: Mapping[str, Any], upper_seconds: float
) -> None:
    """Reserve the immutable policy bound before allowing one detector fit."""

    reserve_hours = float(upper_seconds) / 3600.0
    actual_after = float(budget["cumulative_actual_gpu_hours"]) + reserve_hours
    estimated_remaining = float(
        budget.get("live_detector_remaining_gpu_hours", 0.0)
    )
    revised_after = (
        float(budget["revised_upper_gpu_hours"])
        - estimated_remaining
        + max(estimated_remaining, reserve_hours)
    )
    if not all(map(math.isfinite, (reserve_hours, actual_after, revised_after))):
        raise MethodologicalHalt("detector next-fit reserve is non-finite")
    if reserve_hours <= 0:
        raise MethodologicalHalt("detector next-fit reserve is not positive")
    if actual_after > BUDGET_GPU_HOURS:
        raise BudgetHalt(
            "next detector fit could exceed the 165 GPU-hour actual ceiling"
        )
    if revised_after > BUDGET_GPU_HOURS:
        raise BudgetHalt(
            "next detector fit could exceed the 165 GPU-hour revised projection"
        )


def _state_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("orchestrator_state_sha256", None)
    return canonical_json_sha256(unsigned)


def load_retry_counts() -> dict[str, int]:
    path = PROJECT_ROOT / STATE_PATH
    retry_counts: dict[str, int] = {}
    if path.exists():
        state = read_json(path, label="orchestrator state")
        if state.get("schema_version") != STATE_SCHEMA or _state_hash(state) != state.get("orchestrator_state_sha256"):
            raise MethodologicalHalt("orchestrator state schema/self-hash mismatch")
        retries = state.get("retry_counts", {})
        if not isinstance(retries, dict):
            raise MethodologicalHalt("orchestrator retry state is malformed")
        retry_counts.update({str(key): int(value) for key, value in retries.items()})
    # The atomic recovered-error log is an independent restart source.  This
    # closes the narrow crash window after logging a failed attempt but before
    # the next state snapshot.
    for row in read_jsonl(PROJECT_ROOT / ERROR_PATH):
        action_id = str(row.get("action_id", ""))
        retry_index = int(row.get("retry_index", 0))
        if action_id:
            retry_counts[action_id] = max(retry_counts.get(action_id, 0), retry_index)
    return retry_counts


def progress_refresh_recovery_summary() -> dict[str, Any]:
    rows = read_jsonl(PROJECT_ROOT / EVENT_PATH)
    recovered = [
        row
        for row in rows
        if row.get("event") == "canonical_progress_unstable_source_race_recovered"
    ]
    deferred_races = [
        row
        for row in rows
        if row.get("event") == "canonical_progress_unstable_source_race_deferred"
    ]
    deferred_timeouts = [
        row
        for row in rows
        if row.get("event") == "canonical_progress_updater_timeout_deferred"
    ]
    return {
        "successful_outer_recovery_events": len(recovered),
        "unstable_source_races_recovered": sum(
            int(row.get("source_races_recovered", 0)) for row in recovered
        ),
        "last_successful_outer_recovery": recovered[-1] if recovered else None,
        "deferred_unstable_source_race_events": len(deferred_races),
        "last_deferred_unstable_source_race": (
            deferred_races[-1] if deferred_races else None
        ),
        "deferred_updater_timeout_events": len(deferred_timeouts),
        "last_deferred_updater_timeout": (
            deferred_timeouts[-1] if deferred_timeouts else None
        ),
    }


def process_recovery_summary() -> dict[str, Any]:
    events = read_jsonl(PROJECT_ROOT / EVENT_PATH)
    recovered = [
        row for row in events if row.get("event") == "action_recovery_succeeded"
    ]
    errors = read_jsonl(PROJECT_ROOT / ERROR_PATH)
    return {
        "successful_recovery_events": len(recovered),
        "last_successful_recovery": recovered[-1] if recovered else None,
        "recoverable_error_events": len(errors),
        "last_recoverable_error": errors[-1] if errors else None,
    }


def write_state(
    *,
    status: str,
    message: str,
    action: Action | None,
    retries: Mapping[str, int],
    progress: Mapping[str, Any],
    budget: Mapping[str, Any],
    active_pid: int | None = None,
    detector_status: Mapping[str, Any] | None = None,
) -> None:
    evaluator_unavailability = None
    unavailability_path = PROJECT_ROOT / EVALUATOR_UNAVAILABILITY_PATH
    if unavailability_path.exists():
        manifest = verify_evaluator_unavailability_manifest()
        evaluator_unavailability = {
            "path": str(unavailability_path.resolve()),
            "sha256": file_sha256(unavailability_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "scoreable_evaluator_units": manifest["scoreable_evaluator_units"],
            "upstream_dependent_unavailable_units": manifest[
                "upstream_dependent_unavailable_units"
            ],
            "terminal_accounted_units": manifest["terminal_accounted_units"],
        }
    detector = dict(detector_status) if detector_status is not None else None
    detector_current = (
        detector.get("current_fit")
        if isinstance(detector, dict)
        and isinstance(detector.get("current_fit"), dict)
        else (
            detector.get("next_fit")
            if isinstance(detector, dict)
            and isinstance(detector.get("next_fit"), dict)
            else None
        )
    )
    detector_errors = (
        detector.get("recovered_errors")
        if isinstance(detector, dict)
        and isinstance(detector.get("recovered_errors"), list)
        else []
    )
    canonical_errors = progress.get("recovered_errors")
    if not isinstance(canonical_errors, list):
        canonical_errors = []
    process_errors = read_jsonl(PROJECT_ROOT / ERROR_PATH)
    process_recovery = process_recovery_summary()
    detector_completed = (
        detector.get("completed_fit_count") if detector is not None else None
    )
    detector_total = (
        detector.get("total_fit_count") if detector is not None else None
    )
    value: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "generated_at": utc_now(),
        "status": status,
        "message": message,
        "current_action_id": action.action_id if action else None,
        "current_stage": action.stage if action else (
            progress.get("current", {}).get("stage")
            if isinstance(progress.get("current"), dict)
            else None
        ),
        "current_condition": (
            {
                "split_id": detector_current.get("split_id"),
                "detector_name": detector_current.get("detector_name"),
                "detector_kind": detector_current.get("detector_kind"),
                "fit_index": detector_current.get("index"),
                "condition_state": (
                    detector.get("state") if detector is not None else None
                ),
                "next_fit_upper_seconds": (
                    detector.get("next_fit_upper_seconds")
                    if detector is not None
                    else None
                ),
            }
            if detector_current is not None
            else (
                progress.get("current", {}).get("condition")
                if isinstance(progress.get("current"), dict)
                else None
            )
        ),
        "active_pid": active_pid,
        "retry_counts": dict(sorted(retries.items())),
        "counts": progress.get("counts"),
        "stage_progress": progress.get("stage_progress"),
        "recovery_counts": progress.get("recovery_counts"),
        "successful_recoveries": {
            "payload": (
                progress.get("recovery_counts", {}).get(
                    "successful_payload_recoveries"
                )
                if isinstance(progress.get("recovery_counts"), dict)
                else None
            ),
            "process": process_recovery["successful_recovery_events"],
        },
        "failures": (
            progress.get("counts", {}).get("failures")
            if isinstance(progress.get("counts"), dict)
            else None
        ),
        "cumulative_actual_gpu_hours": budget.get("cumulative_actual_gpu_hours"),
        "measured_throughput": (
            {
                "detector_fits_per_hour": detector.get("fits_per_hour"),
                "detector_rolling_fits_per_hour": detector.get(
                    "rolling_fits_per_hour"
                ),
                "canonical": progress.get("throughput"),
            }
            if detector is not None
            else progress.get("throughput")
        ),
        "rolling_estimated_completion": (
            {
                "rolling_eta_seconds": detector.get("rolling_eta_seconds"),
                "estimated_completion_at": detector.get(
                    "rolling_estimated_completion_utc"
                ),
            }
            if detector is not None
            else progress.get("eta")
        ),
        "last_completed_checkpoint": (
            detector.get("last_completed_checkpoint")
            if detector is not None
            else progress.get("last_checkpoint")
        ),
        "automatically_recovered_errors": (
            canonical_errors + detector_errors + process_errors
        ),
        "detector_execution": detector,
        "detector_completed_fits": detector_completed,
        "detector_total_fits": detector_total,
        "detector_elapsed_seconds": (
            detector.get("global_elapsed_seconds") if detector is not None else None
        ),
        "detector_fits_per_hour": (
            detector.get("fits_per_hour") if detector is not None else None
        ),
        "detector_rolling_eta_seconds": (
            detector.get("rolling_eta_seconds") if detector is not None else None
        ),
        "detector_last_completed_checkpoint": (
            detector.get("last_completed_checkpoint")
            if detector is not None
            else None
        ),
        "progress_refresh_recovery": progress_refresh_recovery_summary(),
        "process_recovery": process_recovery,
        "evaluator_unavailability_accounting": evaluator_unavailability,
        "budget": budget,
        "canonical_progress_path": str((PROJECT_ROOT / PROGRESS_PATH).resolve()),
        "canonical_progress_sha256": progress.get("progress_sha256"),
        "recovered_error_log": str((PROJECT_ROOT / ERROR_PATH).resolve()),
        "progress_refresh_event_log": str((PROJECT_ROOT / EVENT_PATH).resolve()),
        "authorized_projection_sha256": PROJECTION_SHA256,
        "gpu_uuid": GPU_UUID,
    }
    value["orchestrator_state_sha256"] = _state_hash(value)
    atomic_write_json(PROJECT_ROOT / STATE_PATH, value)


def progress_refresh_deferral_kind(
    deferral: DeferrableProgressRefresh,
) -> tuple[str, str]:
    if isinstance(deferral, ProgressRefreshTimeout):
        return (
            "updater_timeout",
            "canonical_progress_updater_timeout_deferred",
        )
    return (
        "unstable_source_race",
        "canonical_progress_unstable_source_race_deferred",
    )


def emit_progress_refresh_deferral(
    deferral: DeferrableProgressRefresh,
    *,
    action: Action | None,
    retries: Mapping[str, int],
    active_pid: int | None,
    budget: Mapping[str, Any],
    occupancy_status: str | None = None,
) -> None:
    kind, event = progress_refresh_deferral_kind(deferral)
    emit_event(
        event,
        deferral_kind=kind,
        detail=str(deferral),
        action_id=action.action_id if action else None,
        active_pid=active_pid,
        occupancy_status=occupancy_status,
        progress_refresh_outer_attempts=(
            PROGRESS_OUTER_ATTEMPTS if isinstance(deferral, ProgressSourceRace) else 1
        ),
        progress_refresh_backoff_seconds=(
            list(PROGRESS_RACE_BACKOFF_SECONDS)
            if isinstance(deferral, ProgressSourceRace)
            else []
        ),
        retry_count_unchanged=(
            retries.get(action.action_id, 0) if action is not None else None
        ),
        cumulative_actual_gpu_hours=budget["cumulative_actual_gpu_hours"],
        revised_upper_gpu_hours=budget["revised_upper_gpu_hours"],
    )


def defer_progress_refresh(
    *,
    deferral: DeferrableProgressRefresh,
    projection: Mapping[str, Any],
    action: Action | None,
    retries: Mapping[str, int],
    status: str,
    message: str,
    active_pid: int | None = None,
    persist_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce the hard ceiling, then persist an operational-only deferral."""

    progress = published_progress_after_source_race()
    budget = calculate_budget(projection, progress)
    enforce_budget(budget)
    kind, _event = progress_refresh_deferral_kind(deferral)
    if kind == "updater_timeout":
        status = status.replace(
            "progress_snapshot_race", "progress_updater_timeout"
        )
    emit_progress_refresh_deferral(
        deferral,
        action=action,
        retries=retries,
        active_pid=active_pid,
        budget=budget,
    )
    if persist_state:
        write_state(
            status=status,
            message=message,
            action=action,
            retries=retries,
            progress=progress,
            budget=budget,
            active_pid=active_pid,
        )
    return progress, budget


def defer_progress_source_race(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible exact-race helper used by isolated operational tests."""

    return defer_progress_refresh(
        deferral=ProgressSourceRace(PROGRESS_SOURCE_RACE_MESSAGE), **kwargs
    )


def record_recoverable_error(
    action: Action, retry_index: int, event: str, detail: str
) -> None:
    atomic_append_jsonl(
        PROJECT_ROOT / ERROR_PATH,
        {
            "at": utc_now(),
            "action_id": action.action_id,
            "stage": action.stage,
            "model_id": action.model_id,
            "event": event,
            "detail": detail[-4000:],
            "retry_index": retry_index,
            "automatic_resume": True,
        },
    )


def emit_event(event: str, **fields: Any) -> None:
    value = {"at": utc_now(), "event": event, **fields}
    atomic_append_jsonl(PROJECT_ROOT / EVENT_PATH, value)
    print(json.dumps(value, sort_keys=True), flush=True)


def _arg_value(argv: Sequence[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = list(argv).index(flag)
    return argv[index + 1] if index + 1 < len(argv) else None


def validate_gpu_action(action: Action) -> None:
    argv = list(action.argv)
    if "--limit" in argv or "--max-pending" in argv:
        raise MethodologicalHalt("frozen GPU action contains a forbidden limiting option")
    if _arg_value(argv, "--gpu-uuid") != GPU_UUID:
        raise MethodologicalHalt("GPU action is not bound to the authorized UUID")
    if _arg_value(argv, "--context") != "4096" or _arg_value(argv, "--n-gpu-layers") != "-1":
        raise MethodologicalHalt("GPU action differs from frozen backend settings")
    if action.kind == "runner" and action.stage == "robustness_v2":
        expected = {
            "--primary-results-root": _relative(PROJECT_ROOT / RESULTS_ROOT / "primary_v2"),
            "--ablation-results-root": _relative(PROJECT_ROOT / RESULTS_ROOT / "ablation_v2"),
            "--robustness-results-root": _relative(PROJECT_ROOT / RESULTS_ROOT / "robustness_v2"),
        }
        for flag, value in expected.items():
            if _arg_value(argv, flag) != value:
                raise MethodologicalHalt("robustness action has a noncanonical {}".format(flag))
    script = PROJECT_ROOT / argv[1]
    required = (
        ("--stage", "--model", "--resume", "--gpu-uuid", "--output-dir")
        if action.kind == "runner"
        else ("--evaluator-model", "--source-stage", "--resume", "--gpu-uuid", "--output-dir")
    )
    _probe_python_cli(script, argv[0], required)


_PROBED_INTERFACES: set[tuple[str, tuple[str, ...]]] = set()


def _probe_python_cli(script: Path, python: str, required: Sequence[str]) -> None:
    key = (str(script), tuple(required))
    if key in _PROBED_INTERFACES:
        return
    if not script.is_file() or script.is_symlink():
        raise InterfaceHalt("required CLI is absent or unsafe: {}".format(script))
    probe = subprocess.run(
        [python, _relative(script), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = probe.stdout + "\n" + probe.stderr
    if probe.returncode != 0 or any(token not in help_text for token in required):
        raise InterfaceHalt(
            "CLI {} lacks the required interface: {}".format(
                script, ", ".join(required)
            )
        )
    _PROBED_INTERFACES.add(key)


def _process_cmdlines() -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if (
            "scripts/run_revision_matrix.py" in argv
            or "scripts/run_revision_evaluator.py" in argv
            or "scripts/run_revision_detectors.py" in argv
        ):
            found[int(entry.name)] = argv
    return found


def _detector_status_path(action: Action) -> Path:
    raw = _arg_value(action.argv, "--status-file")
    if raw is None:
        raise MethodologicalHalt("detector action lacks --status-file")
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _detector_checkpoint_dir(action: Action) -> Path:
    raw = _arg_value(action.argv, "--checkpoint-dir")
    if raw is None:
        raise MethodologicalHalt("detector action lacks --checkpoint-dir")
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _detector_fit_permit_path(action: Action) -> Path:
    raw = _arg_value(action.argv, "--fit-permit-file")
    if raw is None:
        raise MethodologicalHalt("detector action lacks --fit-permit-file")
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _detector_fit_permit_receipt_dir(action: Action) -> Path:
    raw = _arg_value(action.argv, "--fit-permit-receipt-dir")
    if raw is None:
        raise MethodologicalHalt(
            "detector action lacks --fit-permit-receipt-dir"
        )
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _detector_execution_policy_path(action: Action | None = None) -> Path:
    raw = (
        None if action is None else _arg_value(action.argv, "--execution-policy")
    )
    if action is not None and raw is None:
        raise MethodologicalHalt("detector action lacks --execution-policy")
    path = DETECTOR_EXECUTION_POLICY_RELATIVE if raw is None else Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _detector_finalization_contract(
    action: Action,
) -> tuple[str, int | None, str, Path, Path, Path]:
    """Resolve the one immutable candidate/output/receipt identity for an action."""

    checkpoint_dir = _detector_checkpoint_dir(action)
    benchmark_raw = _arg_value(action.argv, "--benchmark-task-index")
    equivalence_role = _arg_value(action.argv, "--equivalence-role")
    equivalence_task_raw = _arg_value(action.argv, "--equivalence-task-index")
    if benchmark_raw is not None and equivalence_role is not None:
        raise MethodologicalHalt(
            "detector benchmark and equivalence finalization modes overlap"
        )
    if benchmark_raw is not None:
        kind = "benchmark_artifact"
        task_index = int(benchmark_raw)
        role = "benchmark"
        output_raw = _arg_value(action.argv, "--benchmark-output")
    elif equivalence_role is not None:
        if equivalence_role not in {"cuda", "cuda_repeat"}:
            raise MethodologicalHalt(
                "only CUDA equivalence roles require supervisor finalization"
            )
        if equivalence_task_raw is None:
            raise MethodologicalHalt("detector equivalence action lacks task index")
        kind = "equivalence_artifact"
        task_index = int(equivalence_task_raw)
        role = equivalence_role
        output_raw = _arg_value(action.argv, "--equivalence-artifact")
    else:
        kind = "detector_run_manifest"
        task_index = None
        role = "suite"
        output_raw = str(action.output_dir / "detector_run_manifest.json")  # type: ignore[operator]
    if task_index is not None and task_index not in DETECTOR_BENCHMARK_TASKS:
        raise MethodologicalHalt("detector finalization task index differs")
    if output_raw is None:
        raise MethodologicalHalt("detector finalization output is absent")
    output = Path(output_raw)
    output = (output if output.is_absolute() else PROJECT_ROOT / output).resolve()
    try:
        candidate, receipt = detector_finalization_paths(
            checkpoint_dir,
            kind=kind,
            requested_output_path=output,
            task_index=task_index,
            role=role,
        )
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector finalization path contract is invalid: {}".format(exc)
        ) from exc
    return kind, task_index, role, output, candidate.resolve(), receipt.resolve()


def _detector_gpu_ledger_path() -> Path:
    return Path(_format_values()["detector_gpu_ledger"]).resolve()


def _detector_ledger_source_identity(
    *, kind: str, task_index: int | None, role: str
) -> tuple[str, str]:
    if kind == "benchmark_artifact":
        assert task_index is not None
        return (
            "production_benchmark_task_{}".format(task_index),
            "detector_production_benchmark",
        )
    if kind == "equivalence_artifact":
        assert task_index is not None
        return (
            "equivalence_{}_task_{}".format(role, task_index),
            "detector_device_equivalence_{}".format(role),
        )
    raise MethodologicalHalt("final detector suite is not a pre-final ledger source")


def _finalize_detector_after_confirmed_exit(
    action: Action, status: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Idempotently publish a runner candidate only after exact PID absence."""

    declared = status.get("finalization_candidate")
    if declared is None:
        return None
    kind, task_index, role, output, candidate_path, receipt_path = (
        _detector_finalization_contract(action)
    )
    if (
        not isinstance(declared, dict)
        or set(declared)
        != {"path", "sha256", "size_bytes", "candidate_sha256", "kind"}
        or Path(str(declared.get("path", ""))).resolve() != candidate_path
        or declared.get("kind") != kind
        or not candidate_path.is_file()
        or candidate_path.is_symlink()
        or file_sha256(candidate_path) != declared.get("sha256")
        or candidate_path.stat().st_size != int(declared.get("size_bytes", -1))
    ):
        raise MethodologicalHalt(
            "detector signed finalization candidate identity differs"
        )
    try:
        candidate = read_detector_finalization_candidate(candidate_path)
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector finalization candidate failed validation: {}".format(exc)
        ) from exc
    if (
        candidate.get("candidate_sha256") != declared.get("candidate_sha256")
        or candidate.get("kind") != kind
        or Path(str(candidate.get("requested_output_path", ""))).resolve()
        != output
        or candidate.get("run_identity_sha256")
        != status.get("run_identity_sha256")
        or status.get("state")
        not in {"supervisor_observed_process_exit", "complete"}
    ):
        raise MethodologicalHalt(
            "detector candidate/status/output finalization binding differs"
        )
    ledger_path = _detector_gpu_ledger_path()
    if kind == "detector_run_manifest" and (
        ledger_path.is_symlink() or not ledger_path.is_file()
    ):
        raise MethodologicalHalt(
            "final detector suite cannot publish before the equivalence GPU ledger"
        )
    try:
        result = finalize_detector_candidate_from_closed_status(
            candidate_path,
            closed_status_file=_detector_status_path(action),
            terminal_receipt_path=receipt_path,
            gpu_accounting_ledger_path=(
                ledger_path if kind == "detector_run_manifest" else None
            ),
        )
        if kind != "detector_run_manifest":
            source_id, component = _detector_ledger_source_identity(
                kind=kind, task_index=task_index, role=role
            )
            update_detector_gpu_accounting_ledger(
                ledger_path,
                source_id=source_id,
                component=component,
                terminal_receipt_path=receipt_path,
            )
            read_detector_gpu_accounting_ledger(ledger_path)
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector supervisor finalization failed closed: {}".format(exc)
        ) from exc
    emit_event(
        "detector_candidate_finalized_after_exit",
        action_id=action.action_id,
        kind=kind,
        task_index=task_index,
        role=role,
        candidate_sha256=candidate["candidate_sha256"],
        output_path=str(output),
        output_sha256=file_sha256(output),
        terminal_receipt_path=str(receipt_path),
        terminal_receipt_sha256=result["terminal_receipt"][
            "terminal_receipt_sha256"
        ],
    )
    return result


def _status_declares_expected_detector_candidate(
    action: Action, status: Mapping[str, Any]
) -> bool:
    declared = status.get("finalization_candidate")
    if declared is None:
        return False
    if not isinstance(declared, dict):
        raise MethodologicalHalt(
            "detector finalization candidate declaration is malformed"
        )
    _kind, _task, _role, _output, expected, _receipt = (
        _detector_finalization_contract(action)
    )
    observed = Path(str(declared.get("path", ""))).resolve()
    if observed == expected:
        return True
    if status.get("state") != "complete":
        raise MethodologicalHalt(
            "a different detector candidate remains unfinalized before resume"
        )
    return False


def _ensure_finalized_detector_ledger(action: Action) -> None:
    kind, task_index, role, _output, _candidate, receipt = (
        _detector_finalization_contract(action)
    )
    if kind == "detector_run_manifest":
        return
    if receipt.is_symlink() or not receipt.is_file():
        raise MethodologicalHalt(
            "finalized CUDA detector artifact lacks its terminal receipt"
        )
    source_id, component = _detector_ledger_source_identity(
        kind=kind, task_index=task_index, role=role
    )
    try:
        update_detector_gpu_accounting_ledger(
            _detector_gpu_ledger_path(),
            source_id=source_id,
            component=component,
            terminal_receipt_path=receipt,
        )
        read_detector_gpu_accounting_ledger(_detector_gpu_ledger_path())
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "finalized detector artifact ledger recovery failed: {}".format(exc)
        ) from exc


def verify_detector_execution_policy() -> dict[str, Any]:
    """Verify the immutable CUDA-only pre-benchmark operational policy."""

    path = _detector_execution_policy_path()
    if not path.is_file() or path.is_symlink():
        raise MethodologicalHalt("detector CUDA policy is absent or unsafe")
    if file_sha256(path) != DETECTOR_EXECUTION_POLICY_SHA256:
        raise MethodologicalHalt("detector CUDA policy byte hash differs")
    policy = read_json(path, label="detector CUDA policy")
    unsigned = dict(policy)
    claimed = unsigned.pop("policy_sha256", None)
    audit = policy.get("audit")
    diagnostic = (
        None
        if not isinstance(audit, dict)
        else PROJECT_ROOT / str(audit.get("diagnostic_path", ""))
    )
    expected_same_cuda = {
        "same_device_cuda": {
            "task_design_exact": True,
            "row_identity_order_labels_exact": True,
            "model_state_sha256_exact": True,
            "scores_exact": True,
            "metrics_exact": True,
            "predictions_exact": True,
        }
    }
    if (
        claimed != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or canonical_json_sha256(unsigned) != claimed
        or policy.get("schema_version")
        != "rankcloak-revision-detector-cuda-policy-v2"
        or policy.get("policy_status")
        != "cuda_only_predeclared_before_new_benchmarks"
        or policy.get("execution")
        != {
            "device": DETECTOR_DEVICE,
            "gpu_uuid": GPU_UUID,
            "workers": DETECTOR_WORKERS,
            "torch_num_threads": 1,
        }
        or policy.get("ceiling")
        != {
            "next_fit_upper_seconds_by_detector": (
                DETECTOR_NEXT_FIT_UPPER_SECONDS
            ),
            "post_benchmark_tighter_gate_required": True,
        }
        or policy.get("authorized_ceiling")
        != {
            "gpu_hours": BUDGET_GPU_HOURS,
            "historical_actual_gpu_hours_floor": (
                DETECTOR_HISTORICAL_GPU_HOURS_FLOOR
            ),
            "projection_path": str(PROJECTION_PATH),
            "projection_sha256": PROJECTION_SHA256,
        }
        or policy.get("benchmark")
        != {
            "task_indices": [0, 1],
            "checkpoint_reuse": True,
            "cuda_reproducibility_fit_count_per_architecture": 2,
            "allowed_failed_fit_retry_count_per_architecture": 1,
            "projection_safety_multiplier": 1.5,
            "full_matrix_budget_gate_required": True,
        }
        or policy.get("equivalence") != expected_same_cuda
        or not isinstance(audit, dict)
        or audit.get("cpu_diagnostics_status")
        != "preserved_feasibility_evidence_only"
        or audit.get("cpu_neural_training_authorized") is not False
        or audit.get("derivation")
        != "revision_takeover_2026-08-15_cuda_only_v2"
        or diagnostic is None
        or diagnostic.is_symlink()
        or not diagnostic.is_file()
        or file_sha256(diagnostic) != audit.get("diagnostic_sha256")
    ):
        raise MethodologicalHalt(
            "detector CUDA policy violates the authorized runtime contract"
        )
    return policy


def _detector_cuda_budget_gate_path() -> Path:
    return Path(_format_values()["detector_cuda_budget_gate"]).resolve()


def require_detector_cuda_budget_gate(
    *, expected_stage: str
) -> dict[str, Any]:
    path = _detector_cuda_budget_gate_path()
    try:
        gate = read_gate(path, expected_stage=expected_stage)
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector CUDA budget gate is absent, stale, or not approved: {}".format(
                exc
            )
        ) from exc
    inputs = gate.get("inputs")
    policy_identity = None if not isinstance(inputs, dict) else inputs.get("policy")
    ledger_identity = None if not isinstance(inputs, dict) else inputs.get("gpu_ledger")
    ledger_path = _detector_gpu_ledger_path()
    if (
        not isinstance(policy_identity, dict)
        or Path(str(policy_identity.get("path", ""))).resolve()
        != _detector_execution_policy_path()
        or policy_identity.get("sha256") != DETECTOR_EXECUTION_POLICY_SHA256
        or policy_identity.get("policy_sha256")
        != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or not isinstance(ledger_identity, dict)
        or Path(str(ledger_identity.get("path", ""))).resolve() != ledger_path
    ):
        raise MethodologicalHalt(
            "detector CUDA budget gate input identity differs"
        )
    ledger = read_detector_gpu_accounting_ledger(ledger_path)
    try:
        verify_ledger_sources_for_stage(
            stage=expected_stage, sources=ledger["sources"]
        )
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector CUDA budget gate ledger sources differ"
        ) from exc
    if (
        ledger_identity.get("sha256") != file_sha256(ledger_path)
        or ledger_identity.get("ledger_sha256") != ledger["ledger_sha256"]
    ):
        raise MethodologicalHalt(
            "detector CUDA budget gate ledger identity differs"
        )
    projection = gate["projection"]
    if (
        float(projection["starting_cumulative_actual_gpu_hours"])
        < DETECTOR_HISTORICAL_GPU_HOURS_FLOOR
        or float(projection["projected_cumulative_gpu_hours"])
        > BUDGET_GPU_HOURS
        or float(projection["projected_remaining_headroom_gpu_hours"]) < 0.0
    ):
        raise BudgetHalt(
            "benchmark-derived detector projection exceeds the GPU ceiling"
        )
    return gate


def _detector_fit_watchdog_seconds(action: Action, detector_name: str) -> float | None:
    policy_upper = DETECTOR_NEXT_FIT_UPPER_SECONDS.get(detector_name)
    if policy_upper is None:
        return None
    if _arg_value(action.argv, "--benchmark-task-index") is not None:
        return policy_upper
    stage = (
        "post_benchmark_pre_reproducibility"
        if _arg_value(action.argv, "--equivalence-role") is not None
        else "post_reproducibility_preproduction"
    )
    gate = require_detector_cuda_budget_gate(expected_stage=stage)
    measured = gate["projection"]["benchmark_derived_fit_watchdog_seconds"]
    derived = float(measured[detector_name])
    if not math.isfinite(derived) or derived <= 0.0 or derived > policy_upper:
        raise MethodologicalHalt(
            "benchmark-derived detector watchdog is invalid"
        )
    return derived


def build_detector_cuda_budget_gate(*, stage: str) -> dict[str, Any]:
    argv = (
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/build_detector_cuda_budget_gate.py",
        "--stage",
        stage,
        "--output",
        str(_detector_cuda_budget_gate_path()),
    )
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_detector_cpu_environment(),
    )
    action = Action(
        action_id="downstream:detector:cuda_budget_gate:{}".format(stage),
        stage="neural_detector",
        kind="downstream",
        argv=argv,
        output_dir=_detector_cuda_budget_gate_path().parent,
        gpu=False,
    )
    atomic_write_bytes(
        _log_path(action),
        (completed.stdout + "\n" + completed.stderr).encode(
            "utf-8", "replace"
        ),
    )
    if completed.returncode != 0:
        path = _detector_cuda_budget_gate_path()
        if path.is_file() and not path.is_symlink():
            value = read_json(path, label="failed detector CUDA budget gate")
            projected = float(
                value.get("projection", {}).get(
                    "projected_cumulative_gpu_hours", math.inf
                )
            )
            if projected > BUDGET_GPU_HOURS:
                raise BudgetHalt(
                    "benchmark-derived detector projection is {:.6f} GPU-hours"
                    .format(projected)
                )
        raise MethodologicalHalt(
            "detector CUDA budget gate construction failed: {}".format(
                (completed.stderr or completed.stdout)[-4000:]
            )
        )
    return require_detector_cuda_budget_gate(expected_stage=stage)


def _detector_checkpoint_state_exists(action: Action) -> bool:
    checkpoint_dir = _detector_checkpoint_dir(action)
    if not checkpoint_dir.exists():
        return False
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise MethodologicalHalt("detector checkpoint root is unsafe")
    try:
        return next(checkpoint_dir.rglob("*"), None) is not None
    except OSError as exc:
        raise MethodologicalHalt(
            "cannot inspect detector checkpoint root: {}".format(exc)
        ) from exc


def validate_detector_action(
    action: Action, *, benchmark_task_index: int | None = None
) -> None:
    """Require the exact checkpointed one-worker CUDA detector contract."""

    values = _format_values()
    expected_argv = (
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_detectors.py",
        "--input",
        values["primary_detector_corpus"],
        "--preprocessing-manifest",
        values["primary_preprocessing_manifest"],
        "--output-dir",
        values["detector_output_dir"],
        "--execution-policy",
        values["detector_execution_policy"],
        "--cuda-budget-gate",
        values["detector_cuda_budget_gate"],
        "--resume",
        "--overwrite",
        "--device",
        DETECTOR_DEVICE,
        "--gpu-uuid",
        GPU_UUID,
        "--workers",
        str(DETECTOR_WORKERS),
        "--checkpoint-dir",
        values["detector_checkpoint_dir"],
        "--status-file",
        values["detector_status_file"],
        "--fit-permit-file",
        values["detector_fit_permit_file"],
        "--fit-permit-receipt-dir",
        values["detector_fit_permit_receipt_dir"],
        "--equivalence-required-report",
        values["detector_equivalence_report_0"],
        "--equivalence-required-report",
        values["detector_equivalence_report_1"],
    )
    if benchmark_task_index is not None:
        if benchmark_task_index not in DETECTOR_BENCHMARK_TASKS:
            raise MethodologicalHalt("unsupported detector benchmark task index")
        expected_argv += (
            "--benchmark-one-fit",
            "--benchmark-task-index",
            str(benchmark_task_index),
            "--benchmark-output",
            values["detector_benchmark_output_{}".format(benchmark_task_index)],
        )
    expected_action_id = (
        "downstream:detector"
        if benchmark_task_index is None
        else "downstream:detector:benchmark:{}".format(benchmark_task_index)
    )
    if (
        action.action_id != expected_action_id
        or action.stage != "neural_detector"
        or action.kind != "downstream"
        or action.gpu is not True
        or action.argv != expected_argv
        or action.output_dir is None
        or _normalized_output_argument(_arg_value(action.argv, "--output-dir"))
        != action.output_dir.resolve()
    ):
        raise MethodologicalHalt(
            "detector action differs from the checkpointed one-worker CUDA contract"
        )
    status_path = _detector_status_path(action)
    checkpoint_dir = _detector_checkpoint_dir(action)
    permit_path = _detector_fit_permit_path(action)
    receipt_dir = _detector_fit_permit_receipt_dir(action)
    policy_path = _detector_execution_policy_path(action)
    output = action.output_dir.resolve()
    if (
        status_path != Path(values["detector_status_file"]).resolve()
        or checkpoint_dir != Path(values["detector_checkpoint_dir"]).resolve()
        or permit_path != Path(values["detector_fit_permit_file"]).resolve()
        or receipt_dir
        != Path(values["detector_fit_permit_receipt_dir"]).resolve()
        or policy_path != Path(values["detector_execution_policy"]).resolve()
        or status_path == output
        or checkpoint_dir == output
        or permit_path == output
        or status_path == checkpoint_dir
        or permit_path in {status_path, checkpoint_dir}
        or receipt_dir in {output, status_path, checkpoint_dir, permit_path}
        or status_path.is_relative_to(output)
        or checkpoint_dir.is_relative_to(output)
        or permit_path.is_relative_to(output)
        or receipt_dir.is_relative_to(output)
        or not receipt_dir.is_relative_to(checkpoint_dir)
    ):
        raise MethodologicalHalt(
            "detector checkpoint/status/permit/receipt paths violate the exact layout"
        )


def _parse_aware_time(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MethodologicalHalt("{} is not ISO-8601".format(label)) from exc
    if parsed.tzinfo is None:
        raise MethodologicalHalt("{} lacks a timezone".format(label))
    return parsed.astimezone(timezone.utc)


def _validate_detector_internal_progress(value: Any) -> datetime:
    if not isinstance(value, dict):
        raise MethodologicalHalt("detector internal progress is malformed")
    phase = str(value.get("phase", ""))
    base_keys = {"phase", "completed_units", "total_units", "updated_at_utc"}
    training_keys = base_keys | {
        "epoch",
        "epochs",
        "batch",
        "batches_per_epoch",
    }
    expected_keys = training_keys if phase == "training" else base_keys
    if (
        phase
        not in {
            "initialization_and_preprocessing",
            "training",
            "trained_state_hashing",
            "evaluation",
            "complete",
        }
        or set(value) != expected_keys
    ):
        raise MethodologicalHalt("detector internal progress shape differs")
    completed = int(value.get("completed_units", -1))
    total_raw = value.get("total_units")
    total = None if total_raw is None else int(total_raw)
    if completed < 0 or (total is not None and (total < 0 or completed > total)):
        raise MethodologicalHalt("detector internal progress counters are invalid")
    if phase == "training":
        epoch = int(value.get("epoch", -1))
        epochs = int(value.get("epochs", -1))
        batch = int(value.get("batch", -1))
        batches = int(value.get("batches_per_epoch", -1))
        if (
            epochs <= 0
            or batches <= 0
            or epoch < 0
            or epoch > epochs
            or batch < 0
            or batch > batches
        ):
            raise MethodologicalHalt(
                "detector internal epoch/batch counters are invalid"
            )
    updated = _parse_aware_time(
        value.get("updated_at_utc"), "detector internal progress updated_at_utc"
    )
    if updated > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise MethodologicalHalt(
            "detector internal progress timestamp is implausibly in the future"
        )
    return updated


def read_detector_status(
    action: Action, *, expected_pid: int | None = None
) -> dict[str, Any] | None:
    """Read and strictly verify the detector's atomically published heartbeat."""

    path = _detector_status_path(action)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise MethodologicalHalt("detector status is not a regular file")
    status = read_json(path, label="checkpointed detector status")
    unsigned = dict(status)
    claimed = unsigned.pop("status_sha256", None)
    if (
        status.get("schema_version") != DETECTOR_STATUS_SCHEMA
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
    ):
        raise MethodologicalHalt("detector status schema/self-hash mismatch")
    completed = int(status.get("completed_fit_count", -1))
    total = int(status.get("total_fit_count", -1))
    elapsed = float(status.get("global_elapsed_seconds", -1.0))
    current_elapsed_raw = status.get("current_fit_elapsed_seconds")
    current_elapsed = (
        0.0 if current_elapsed_raw is None else float(current_elapsed_raw)
    )
    process_elapsed = float(status.get("process_elapsed_seconds", -1.0))
    checkpoint_start = float(
        status.get("checkpoint_fit_seconds_at_process_start", -1.0)
    )
    checkpoint_cumulative = float(
        status.get("checkpoint_cumulative_fit_seconds", -1.0)
    )
    rate = status.get("fits_per_hour")
    eta = status.get("rolling_eta_seconds")
    if (
        total != DETECTOR_TOTAL_FITS
        or completed < 0
        or completed > total
        or not math.isfinite(elapsed)
        or elapsed < 0
        or not math.isfinite(current_elapsed)
        or current_elapsed < 0
        or not math.isfinite(process_elapsed)
        or process_elapsed < 0
        or not math.isfinite(checkpoint_start)
        or checkpoint_start < 0
        or not math.isfinite(checkpoint_cumulative)
        or checkpoint_cumulative < checkpoint_start
        or status.get("global_elapsed_policy")
        != "sum_of_valid_fit_intervals_plus_active_fit_v1"
        or (rate is not None and (not math.isfinite(float(rate)) or float(rate) < 0))
        or (eta is not None and (not math.isfinite(float(eta)) or float(eta) < 0))
        or status.get("device") != DETECTOR_DEVICE
        or status.get("gpu_uuid") != GPU_UUID
        or int(status.get("workers", -1)) != DETECTOR_WORKERS
        or status.get("state")
        not in {
            "resuming",
            "awaiting_fit_ceiling_gate",
            "running_fit",
            "fit_checkpointed",
            "stop_requested_finishing_current_fit",
            "stopped_at_fit_boundary",
            "fits_complete_awaiting_final_manifest",
            "awaiting_supervisor_finalization",
            "supervisor_observed_process_exit",
            "error",
            "complete",
        }
        or not isinstance(status.get("recovered_errors", []), list)
    ):
        raise MethodologicalHalt("detector status violates the frozen runtime contract")
    updated = _parse_aware_time(
        status.get("updated_at_utc"), "detector status updated_at_utc"
    )
    if updated > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise MethodologicalHalt("detector status timestamp is implausibly in the future")
    current = status.get("current_fit")
    if current is not None and (
        not isinstance(current, dict)
        or int(current.get("index", -1)) < 0
        or int(current.get("index", -1)) >= DETECTOR_TOTAL_FITS
        or int(current.get("ordinal", -1)) != int(current.get("index", -2))
        or int(current.get("fit_number", -1)) != int(current.get("index", -2)) + 1
        or not str(current.get("split_id", ""))
        or not str(current.get("detector_name", ""))
        or not str(current.get("detector_kind", ""))
        or int(current.get("seed", -1)) < 0
        or not str(current.get("task_identity_sha256", ""))
    ):
        raise MethodologicalHalt("detector status current_fit is malformed")
    internal_progress = status.get("current_fit_progress")
    if internal_progress is not None:
        _validate_detector_internal_progress(internal_progress)
    if (
        (current is None and internal_progress is not None)
        or (
            status.get("state") in {"running_fit", "stop_requested_finishing_current_fit"}
            and current is None
        )
    ):
        raise MethodologicalHalt(
            "detector current fit and internal progress identity differ"
        )
    next_fit = status.get("next_fit")
    next_upper = status.get("next_fit_upper_seconds")
    gate_nonce = status.get("fit_gate_nonce")
    gate_fields_present = any(
        value is not None for value in (next_fit, next_upper, gate_nonce)
    )
    if gate_fields_present:
        if (
            not isinstance(next_fit, dict)
            or int(next_fit.get("index", -1)) < 0
            or int(next_fit.get("index", -1)) >= DETECTOR_TOTAL_FITS
            or int(next_fit.get("ordinal", -1))
            != int(next_fit.get("index", -2))
            or int(next_fit.get("fit_number", -1))
            != int(next_fit.get("index", -2)) + 1
            or not str(next_fit.get("split_id", ""))
            or not str(next_fit.get("regime", ""))
            or next_fit.get("detector_name")
            not in DETECTOR_NEXT_FIT_UPPER_SECONDS
            or not str(next_fit.get("detector_kind", ""))
            or int(next_fit.get("seed", -1)) < 0
            or not str(next_fit.get("task_identity_sha256", ""))
            or float(next_upper)
            != DETECTOR_NEXT_FIT_UPPER_SECONDS[next_fit["detector_name"]]
            or not isinstance(gate_nonce, str)
            or len(gate_nonce) != 64
        ):
            raise MethodologicalHalt("detector next-fit ceiling gate is malformed")
    if (
        (status.get("state") == "awaiting_fit_ceiling_gate" and not gate_fields_present)
        or (status.get("state") == "awaiting_fit_ceiling_gate" and current is not None)
        or status.get("fit_permit_file")
        != str(_detector_fit_permit_path(action))
        or status.get("fit_permit_receipt_dir")
        != str(_detector_fit_permit_receipt_dir(action))
    ):
        raise MethodologicalHalt("detector next-fit ceiling gate identity differs")
    pid = int(status.get("pid", -1))
    start_ticks = int(status.get("process_start_ticks", -1))
    if pid <= 0 or start_ticks <= 0:
        raise MethodologicalHalt("detector status lacks a process identity")
    _validate_detector_gpu_accounting(status.get("gpu_accounting"), live=True)
    run_identity = status.get("run_identity")
    if (
        not isinstance(run_identity, dict)
        or status.get("run_identity_sha256") != canonical_json_sha256(run_identity)
        or run_identity.get("device") != DETECTOR_DEVICE
        or run_identity.get("gpu_uuid") != GPU_UUID
        or int(run_identity.get("workers", -1)) != DETECTOR_WORKERS
        or Path(str(run_identity.get("output_dir", ""))).resolve()
        != action.output_dir.resolve()  # type: ignore[union-attr]
        or Path(str(run_identity.get("checkpoint_dir", ""))).resolve()
        != _detector_checkpoint_dir(action)
        or Path(str(run_identity.get("status_file", ""))).resolve()
        != _detector_status_path(action)
        or Path(str(run_identity.get("fit_permit_file", ""))).resolve()
        != _detector_fit_permit_path(action)
        or Path(str(run_identity.get("fit_permit_receipt_dir", ""))).resolve()
        != _detector_fit_permit_receipt_dir(action)
        or Path(str(run_identity.get("execution_policy_path", ""))).resolve()
        != _detector_execution_policy_path(action)
        or run_identity.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or run_identity.get("require_fit_permit") is not True
    ):
        raise MethodologicalHalt("detector status run identity differs from the action")
    consumed_receipt = status.get("last_consumed_fit_permit")
    if consumed_receipt is not None:
        _validate_detector_fit_permit_receipt(
            action,
            status,
            identity=consumed_receipt,
        )
    if expected_pid is not None and pid == expected_pid:
        observed_ticks = _process_start_ticks(expected_pid)
        if observed_ticks is not None and start_ticks != observed_ticks:
            raise MethodologicalHalt(
                "detector status process-start identity differs from attached process"
            )
    return status


def _cpu_detector_default_permit_paths(action: Action) -> tuple[Path, Path]:
    """Resolve the runner defaults that remain identity-bound in CPU mode."""

    if action.output_dir is None:
        raise MethodologicalHalt("CPU detector action lacks an output directory")
    output = action.output_dir.resolve()
    permit = output.with_name(output.name + ".fit_permit.json")
    receipts = _detector_checkpoint_dir(action) / "fit_permit_receipts"
    return permit, receipts


def read_checkpointed_cpu_detector_status(
    action: Action, *, expected_pid: int | None = None
) -> dict[str, Any] | None:
    """Strictly bind a signed CPU-reference heartbeat to one exact action."""

    path = _detector_status_path(action)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise MethodologicalHalt("CPU detector status is not a regular file")
    status = read_json(path, label="checkpointed CPU detector status")
    unsigned = dict(status)
    claimed = unsigned.pop("status_sha256", None)
    if (
        status.get("schema_version") != DETECTOR_STATUS_SCHEMA
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
    ):
        raise MethodologicalHalt("CPU detector status schema/self-hash mismatch")
    try:
        completed = int(status.get("completed_fit_count", -1))
        total = int(status.get("total_fit_count", -1))
        elapsed = float(status.get("global_elapsed_seconds", -1.0))
        process_elapsed = float(status.get("process_elapsed_seconds", -1.0))
        checkpoint_start = float(
            status.get("checkpoint_fit_seconds_at_process_start", -1.0)
        )
        checkpoint_cumulative = float(
            status.get("checkpoint_cumulative_fit_seconds", -1.0)
        )
        pid = int(status.get("pid", -1))
        start_ticks = int(status.get("process_start_ticks", -1))
        task_index = int(_arg_value(action.argv, "--equivalence-task-index") or -1)
    except (TypeError, ValueError) as exc:
        raise MethodologicalHalt(
            "CPU detector status numeric fields are malformed"
        ) from exc
    current = status.get("current_fit")
    last_checkpoint = status.get("last_completed_checkpoint")
    if (
        total != DETECTOR_TOTAL_FITS
        or completed < 0
        or completed > total
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                elapsed,
                process_elapsed,
                checkpoint_start,
                checkpoint_cumulative,
            )
        )
        or checkpoint_cumulative < checkpoint_start
        or status.get("global_elapsed_policy")
        != "sum_of_valid_fit_intervals_plus_active_fit_v1"
        or status.get("device") != "cpu"
        or status.get("gpu_uuid") is not None
        or int(status.get("workers", -1)) != DETECTOR_WORKERS
        or status.get("gpu_accounting") is not None
        or status.get("state")
        not in {
            "resuming",
            "running_fit",
            "fit_checkpointed",
            "stop_requested_finishing_current_fit",
            "stopped_at_fit_boundary",
            "error",
        }
        or not isinstance(status.get("recovered_errors", []), list)
        or status.get("next_fit") is not None
        or status.get("next_fit_upper_seconds") is not None
        or status.get("fit_gate_nonce") is not None
        or status.get("last_consumed_fit_permit") is not None
        or pid <= 0
        or start_ticks <= 0
        or task_index not in DETECTOR_BENCHMARK_TASKS
    ):
        raise MethodologicalHalt(
            "CPU detector status violates the frozen runtime contract"
        )
    if current is not None and (
        not isinstance(current, dict)
        or int(current.get("index", -1)) != task_index
        or int(current.get("ordinal", -1)) != task_index
        or int(current.get("fit_number", -1)) != task_index + 1
        or current.get("detector_name") != DETECTOR_BENCHMARK_TASKS[task_index]
        or not str(current.get("split_id", ""))
        or not str(current.get("detector_kind", ""))
        or int(current.get("seed", -1)) < 0
        or not str(current.get("task_identity_sha256", ""))
    ):
        raise MethodologicalHalt("CPU detector status current fit differs")
    if completed and (
        not isinstance(last_checkpoint, dict)
        or int(last_checkpoint.get("task_ordinal", -1)) != task_index
    ):
        raise MethodologicalHalt(
            "CPU detector status checkpoint differs from the selected fit"
        )
    updated = _parse_aware_time(
        status.get("updated_at_utc"), "CPU detector status updated_at_utc"
    )
    if updated > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise MethodologicalHalt(
            "CPU detector status timestamp is implausibly in the future"
        )
    expected_permit, expected_receipts = _cpu_detector_default_permit_paths(
        action
    )
    run_identity = status.get("run_identity")
    if (
        status.get("fit_permit_file") != str(expected_permit)
        or status.get("fit_permit_receipt_dir") != str(expected_receipts)
        or not isinstance(run_identity, dict)
        or status.get("run_identity_sha256")
        != canonical_json_sha256(run_identity)
        or run_identity.get("device") != "cpu"
        or run_identity.get("gpu_uuid") is not None
        or int(run_identity.get("workers", -1)) != DETECTOR_WORKERS
        or Path(str(run_identity.get("output_dir", ""))).resolve()
        != action.output_dir.resolve()  # type: ignore[union-attr]
        or Path(str(run_identity.get("checkpoint_dir", ""))).resolve()
        != _detector_checkpoint_dir(action)
        or Path(str(run_identity.get("status_file", ""))).resolve() != path
        or Path(str(run_identity.get("fit_permit_file", ""))).resolve()
        != expected_permit
        or Path(str(run_identity.get("fit_permit_receipt_dir", ""))).resolve()
        != expected_receipts
        or Path(str(run_identity.get("execution_policy_path", ""))).resolve()
        != _detector_execution_policy_path(action)
        or run_identity.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or run_identity.get("require_fit_permit") is not False
    ):
        raise MethodologicalHalt(
            "CPU detector status run identity differs from the action"
        )
    if expected_pid is not None and pid == expected_pid:
        observed_ticks = _process_start_ticks(expected_pid)
        if observed_ticks is not None and observed_ticks != start_ticks:
            raise MethodologicalHalt(
                "CPU detector status process-start identity differs"
            )
    return status


def _validate_detector_gpu_accounting(
    accounting: Any, *, live: bool
) -> float:
    if not isinstance(accounting, dict) or set(accounting) != {
        "device",
        "gpu_uuid",
        "intervals",
        "cumulative_elapsed_seconds",
        "derivation_policy",
    }:
        raise MethodologicalHalt("detector GPU accounting shape is invalid")
    intervals = accounting.get("intervals")
    if (
        accounting.get("device") != DETECTOR_DEVICE
        or accounting.get("gpu_uuid") != GPU_UUID
        or accounting.get("derivation_policy")
        != DETECTOR_GPU_COLLECTION_DERIVATION
        or not isinstance(intervals, list)
        or not intervals
    ):
        raise MethodologicalHalt("detector GPU accounting identity is invalid")
    total = 0.0
    prior_end: datetime | None = None
    identities: set[tuple[int, int]] = set()
    for index, interval in enumerate(intervals):
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
            raise MethodologicalHalt("detector GPU interval shape is invalid")
        pid = int(interval.get("pid", -1))
        ticks = int(interval.get("process_start_ticks", -1))
        started = _parse_aware_time(
            interval.get("started_at_utc"), "detector GPU interval start"
        )
        observed = _parse_aware_time(
            interval.get("last_observed_at_utc"),
            "detector GPU interval observation",
        )
        completed_raw = interval.get("completed_at_utc")
        completed = (
            None
            if completed_raw is None
            else _parse_aware_time(completed_raw, "detector GPU interval completion")
        )
        elapsed_seconds = float(interval.get("elapsed_seconds", -1.0))
        end = completed or observed
        if (
            pid <= 0
            or ticks <= 0
            or (pid, ticks) in identities
            or interval.get("device") != DETECTOR_DEVICE
            or interval.get("gpu_uuid") != GPU_UUID
            or interval.get("derivation_policy") != DETECTOR_GPU_DERIVATION
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
            or end < started
            or abs((end - started).total_seconds() - elapsed_seconds) > 1e-6
            or (completed is not None and observed != completed)
            or (completed is None and (not live or index != len(intervals) - 1))
            or (prior_end is not None and started < prior_end)
        ):
            raise MethodologicalHalt("detector GPU interval provenance is invalid")
        identities.add((pid, ticks))
        prior_end = end
        total += elapsed_seconds
    cumulative = float(accounting.get("cumulative_elapsed_seconds", -1.0))
    if (
        not math.isfinite(cumulative)
        or cumulative < 0
        or abs(cumulative - total) > 1e-6
    ):
        raise MethodologicalHalt("detector cumulative GPU interval time is invalid")
    return cumulative


def _merge_detector_gpu_accounting(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return the exact nonoverlapping union, deduplicating only equal intervals."""

    if not values:
        return None
    by_identity: dict[tuple[int, int, str], dict[str, Any]] = {}
    for value in values:
        _validate_detector_gpu_accounting(value, live=False)
        for raw_interval in value["intervals"]:
            interval = dict(raw_interval)
            key = (
                int(interval["pid"]),
                int(interval["process_start_ticks"]),
                str(interval["started_at_utc"]),
            )
            prior = by_identity.get(key)
            if prior is not None and prior != interval:
                raise MethodologicalHalt(
                    "detector GPU ledgers disagree for one process interval"
                )
            by_identity[key] = interval
    intervals = sorted(
        by_identity.values(),
        key=lambda row: (
            _parse_aware_time(
                row["started_at_utc"], "merged detector GPU interval start"
            ),
            int(row["pid"]),
            int(row["process_start_ticks"]),
        ),
    )
    merged: dict[str, Any] = {
        "device": DETECTOR_DEVICE,
        "gpu_uuid": GPU_UUID,
        "intervals": intervals,
        "cumulative_elapsed_seconds": sum(
            float(interval["elapsed_seconds"]) for interval in intervals
        ),
        "derivation_policy": DETECTOR_GPU_COLLECTION_DERIVATION,
    }
    _validate_detector_gpu_accounting(merged, live=False)
    return merged


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        ).split()
        return int(fields[21])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


def _process_elapsed_seconds(pid: int, start_ticks: int) -> float | None:
    observed = _process_start_ticks(pid)
    if observed is None or observed != start_ticks:
        return None
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None
    return max(0.0, uptime - start_ticks / clock_ticks)


def _detector_processes() -> dict[int, list[str]]:
    return {
        pid: argv
        for pid, argv in _process_cmdlines().items()
        if "scripts/run_revision_detectors.py" in argv
    }


def _exact_detector_pid(action: Action) -> int | None:
    processes = _detector_processes()
    if len(processes) > 1:
        raise MethodologicalHalt(
            "multiple detector processes exist: {}".format(sorted(processes))
        )
    if not processes:
        return None
    pid, argv = next(iter(processes.items()))
    if tuple(argv) != action.argv:
        raise MethodologicalHalt(
            "live detector process does not match the exact checkpointed action"
        )
    return pid


def _detector_live_gpu_seconds(
    status: Mapping[str, Any] | None,
    *,
    pid: int,
    start_ticks: int,
) -> float:
    process_elapsed = _process_elapsed_seconds(pid, start_ticks)
    if status is None:
        return process_elapsed or 0.0
    accounting = status.get("gpu_accounting")
    status_total = _validate_detector_gpu_accounting(accounting, live=True)
    status_pid = int(status.get("pid", -1))
    status_ticks = int(status.get("process_start_ticks", -1))
    if status_pid != pid or status_ticks != start_ticks:
        return status_total + (process_elapsed or 0.0)
    intervals = accounting["intervals"]
    current = intervals[-1]
    prior = status_total - float(current["elapsed_seconds"])
    return prior + max(float(current["elapsed_seconds"]), process_elapsed or 0.0)


def _detector_unaccounted_gpu_seconds(
    progress: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    live_pid: int | None = None,
    live_start_ticks: int | None = None,
) -> float:
    """Charge status intervals not already present in canonical progress."""

    if status is None:
        if live_pid is None or live_start_ticks is None:
            return 0.0
        return _process_elapsed_seconds(live_pid, live_start_ticks) or 0.0
    accounting = status.get("gpu_accounting")
    _validate_detector_gpu_accounting(accounting, live=True)
    assert isinstance(accounting, dict)
    gpu = progress.get("gpu")
    canonical_rows = (
        gpu.get("confirmatory_intervals", []) if isinstance(gpu, dict) else []
    )
    canonical: dict[tuple[int, int, datetime], Mapping[str, Any]] = {}
    for row in canonical_rows:
        if not isinstance(row, dict) or row.get("component") != "neural_detector":
            continue
        key = (
            int(row.get("source_pid", -1)),
            int(row.get("source_process_start_ticks", -1)),
            _parse_aware_time(
                row.get("started_at"), "canonical detector GPU interval start"
            ),
        )
        if key in canonical:
            raise MethodologicalHalt(
                "canonical progress duplicates a detector GPU interval"
            )
        canonical[key] = row
    total = 0.0
    for interval in accounting["intervals"]:
        pid = int(interval["pid"])
        ticks = int(interval["process_start_ticks"])
        started = _parse_aware_time(
            interval["started_at_utc"], "detector status GPU interval start"
        )
        elapsed = float(interval["elapsed_seconds"])
        if pid == live_pid and ticks == live_start_ticks:
            elapsed = max(
                elapsed, _process_elapsed_seconds(pid, ticks) or 0.0
            )
        row = canonical.get((pid, ticks, started))
        if row is None:
            total += elapsed
            continue
        canonical_seconds = float(row.get("seconds", -1.0))
        canonical_end = _parse_aware_time(
            row.get("ended_at"), "canonical detector GPU interval end"
        )
        status_end = _parse_aware_time(
            interval.get("completed_at_utc")
            or interval.get("last_observed_at_utc"),
            "detector status GPU interval end",
        )
        if (
            canonical_seconds < 0
            or canonical_end < status_end
            or canonical_seconds + 1e-6 < elapsed
        ):
            raise MethodologicalHalt(
                "canonical detector accounting is shorter than signed status history"
            )
    return total


def _detector_status_gpu_seconds(status: Mapping[str, Any] | None) -> float:
    if status is None:
        return 0.0
    return _validate_detector_gpu_accounting(
        status.get("gpu_accounting"), live=True
    )


def _detector_remaining_gpu_seconds(
    status: Mapping[str, Any] | None,
) -> float:
    if status is None or status.get("state") in {
        "fits_complete_awaiting_final_manifest",
        "complete",
    }:
        return 0.0
    raw = status.get("rolling_eta_seconds")
    if raw is None:
        return 0.0
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise MethodologicalHalt("detector rolling ETA is invalid")
    return value


def _validate_detector_fit_permit(
    permit: Mapping[str, Any], status: Mapping[str, Any]
) -> None:
    unsigned = dict(permit)
    claimed = unsigned.pop("permit_sha256", None)
    next_fit = status.get("next_fit")
    if not isinstance(next_fit, dict):
        raise MethodologicalHalt("detector fit permit lacks a signed gate target")
    if (
        set(permit)
        != {
            "schema_version",
            "run_identity_sha256",
            "task_identity_sha256",
            "fit_gate_nonce",
            "invocation_pid",
            "invocation_start_ticks",
            "next_fit_upper_seconds",
            "issued_at_utc",
            "permit_sha256",
        }
        or permit.get("schema_version") != DETECTOR_FIT_PERMIT_SCHEMA
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
        or permit.get("run_identity_sha256")
        != status.get("run_identity_sha256")
        or permit.get("task_identity_sha256")
        != next_fit.get("task_identity_sha256")
        or permit.get("fit_gate_nonce") != status.get("fit_gate_nonce")
        or int(permit.get("invocation_pid", -1))
        != int(status.get("pid", -2))
        or int(permit.get("invocation_start_ticks", -1))
        != int(status.get("process_start_ticks", -2))
        or float(permit.get("next_fit_upper_seconds", -1.0))
        != float(status.get("next_fit_upper_seconds", -2.0))
    ):
        raise MethodologicalHalt("detector fit permit identity/self-hash differs")
    issued = _parse_aware_time(
        permit.get("issued_at_utc"), "detector fit permit issuance"
    )
    if issued > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise MethodologicalHalt("detector fit permit timestamp is in the future")


def _stable_regular_json(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str, int]:
    """Read immutable handshake bytes while rejecting replacement/symlink races."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise MethodologicalHalt("{} is absent or unsafe".format(label))
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise MethodologicalHalt("cannot read {}: {}".format(label, exc)) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(content) != after.st_size:
        raise MethodologicalHalt("{} changed while being read".format(label))
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodologicalHalt("{} is not valid UTF-8 JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise MethodologicalHalt("{} is not a JSON object".format(label))
    return value, hashlib.sha256(content).hexdigest(), len(content)


def _detector_fit_permit_receipt_path(action: Action, nonce: object) -> Path:
    text = str(nonce)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MethodologicalHalt("detector fit permit receipt nonce is malformed")
    return _detector_fit_permit_receipt_dir(action) / (text + ".json")


def _validate_detector_fit_permit_receipt(
    action: Action,
    status: Mapping[str, Any],
    *,
    identity: Any = None,
    gate_nonce: object | None = None,
    task_identity_sha256: object | None = None,
    next_fit_upper_seconds: object | None = None,
) -> dict[str, Any]:
    """Verify one durable consumed-permit receipt and its exact status binding."""

    identity_fields = {
        "path",
        "sha256",
        "size_bytes",
        "receipt_sha256",
        "fit_gate_nonce",
        "task_identity_sha256",
        "invocation_pid",
        "invocation_start_ticks",
        "consumed_at_utc",
    }
    if identity is not None:
        if not isinstance(identity, dict) or set(identity) != identity_fields:
            raise MethodologicalHalt(
                "detector consumed-permit receipt identity is malformed"
            )
        gate_nonce = identity["fit_gate_nonce"]
        task_identity_sha256 = identity["task_identity_sha256"]
    if gate_nonce is None or task_identity_sha256 is None:
        raise MethodologicalHalt("detector fit permit receipt target is missing")
    task_hash = str(task_identity_sha256)
    if len(task_hash) != 64 or any(
        character not in "0123456789abcdef" for character in task_hash
    ):
        raise MethodologicalHalt(
            "detector fit permit receipt task identity is malformed"
        )
    path = _detector_fit_permit_receipt_path(action, gate_nonce)
    receipt, raw_sha256, size_bytes = _stable_regular_json(
        path, label="detector fit permit receipt"
    )
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    exact_fields = {
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
    }
    upper = float(receipt.get("next_fit_upper_seconds", -1.0))
    expected_upper = (
        None
        if next_fit_upper_seconds is None
        else float(next_fit_upper_seconds)
    )
    hashes = (
        receipt.get("issued_permit_sha256"),
        receipt.get("issued_permit_file_sha256"),
    )
    if (
        set(receipt) != exact_fields
        or receipt.get("schema_version")
        != DETECTOR_FIT_PERMIT_RECEIPT_SCHEMA
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
        or receipt.get("run_identity_sha256")
        != status.get("run_identity_sha256")
        or receipt.get("task_identity_sha256") != task_hash
        or receipt.get("fit_gate_nonce") != str(gate_nonce)
        or int(receipt.get("invocation_pid", -1))
        != int(status.get("pid", -2))
        or int(receipt.get("invocation_start_ticks", -1))
        != int(status.get("process_start_ticks", -2))
        or not math.isfinite(upper)
        or upper not in set(DETECTOR_NEXT_FIT_UPPER_SECONDS.values())
        or (expected_upper is not None and upper != expected_upper)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
        or int(receipt.get("issued_permit_size_bytes", -1)) <= 0
    ):
        raise MethodologicalHalt(
            "detector fit permit receipt identity/self-hash differs"
        )
    consumed = _parse_aware_time(
        receipt.get("consumed_at_utc"), "detector fit permit receipt consumption"
    )
    if consumed > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise MethodologicalHalt(
            "detector fit permit receipt timestamp is in the future"
        )
    if identity is not None and (
        Path(str(identity.get("path", ""))).resolve() != path
        or identity.get("sha256") != raw_sha256
        or int(identity.get("size_bytes", -1)) != size_bytes
        or identity.get("receipt_sha256") != receipt.get("receipt_sha256")
        or int(identity.get("invocation_pid", -1))
        != int(receipt.get("invocation_pid", -2))
        or int(identity.get("invocation_start_ticks", -1))
        != int(receipt.get("invocation_start_ticks", -2))
        or identity.get("consumed_at_utc") != receipt.get("consumed_at_utc")
    ):
        raise MethodologicalHalt(
            "detector status receipt identity differs from durable bytes"
        )
    return receipt


def _detector_current_gate_receipt(
    action: Action, status: Mapping[str, Any]
) -> dict[str, Any] | None:
    next_fit = status.get("next_fit")
    if not isinstance(next_fit, dict) or status.get("fit_gate_nonce") is None:
        return None
    path = _detector_fit_permit_receipt_path(
        action, status["fit_gate_nonce"]
    )
    if not path.exists() and not path.is_symlink():
        return None
    return _validate_detector_fit_permit_receipt(
        action,
        status,
        gate_nonce=status["fit_gate_nonce"],
        task_identity_sha256=next_fit.get("task_identity_sha256"),
        next_fit_upper_seconds=status.get("next_fit_upper_seconds"),
    )


def issue_detector_fit_permit(
    action: Action,
    status: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    pid: int,
    start_ticks: int,
) -> dict[str, Any]:
    """Atomically issue one exact fit permit after lease, policy, and budget gates."""

    verify_detector_execution_policy()
    if not _detector_pid_is_live(action, pid, start_ticks):
        raise MethodologicalHalt(
            "detector process identity changed before fit-permit issuance"
        )
    freshest = read_detector_status(action, expected_pid=pid)
    if (
        freshest is None
        or freshest.get("state") != "awaiting_fit_ceiling_gate"
        or int(freshest.get("pid", -1)) != pid
        or int(freshest.get("process_start_ticks", -1)) != start_ticks
    ):
        raise MethodologicalHalt(
            "detector left its signed ceiling gate before permit issuance"
        )
    next_fit = freshest.get("next_fit")
    assert isinstance(next_fit, dict)
    upper_seconds = float(freshest["next_fit_upper_seconds"])
    expected_upper = DETECTOR_NEXT_FIT_UPPER_SECONDS.get(
        str(next_fit.get("detector_name"))
    )
    if expected_upper is None or upper_seconds != expected_upper:
        raise MethodologicalHalt("detector requested a non-policy next-fit reserve")
    enforce_budget(budget)
    enforce_detector_next_fit_reserve(budget, upper_seconds)

    # A receipt is a durable consumption acknowledgement.  It takes precedence
    # over a possibly stale awaiting status and permanently forbids reissuing
    # this nonce after either process or supervisor restart.
    consumed = _detector_current_gate_receipt(action, freshest)
    if consumed is not None:
        return consumed

    path = _detector_fit_permit_path(action)
    if path.exists() or path.is_symlink():
        try:
            existing, _raw_sha256, _size_bytes = _stable_regular_json(
                path, label="detector fit permit"
            )
            _validate_detector_fit_permit(existing, freshest)
            return existing
        except MethodologicalHalt:
            consumed = _detector_current_gate_receipt(action, freshest)
            if consumed is not None:
                return consumed
            raise

    # Re-read immediately before publication.  The detector retains a consumed
    # permit until a signed running status and receipt exist, so these three
    # observations close the stale-awaiting duplicate-publication window.
    latest = read_detector_status(action, expected_pid=pid)
    same_gate = bool(
        latest is not None
        and latest.get("state") == "awaiting_fit_ceiling_gate"
        and latest.get("run_identity_sha256")
        == freshest.get("run_identity_sha256")
        and latest.get("fit_gate_nonce") == freshest.get("fit_gate_nonce")
        and latest.get("next_fit") == freshest.get("next_fit")
        and float(latest.get("next_fit_upper_seconds", -1.0))
        == upper_seconds
        and int(latest.get("pid", -1)) == pid
        and int(latest.get("process_start_ticks", -1)) == start_ticks
    )
    if not same_gate:
        consumed = _detector_current_gate_receipt(action, freshest)
        if consumed is not None:
            return consumed
        raise MethodologicalHalt(
            "detector ceiling gate changed before permit publication"
        )
    assert latest is not None
    freshest = latest
    consumed = _detector_current_gate_receipt(action, freshest)
    if consumed is not None:
        return consumed
    if path.exists() or path.is_symlink():
        try:
            existing, _raw_sha256, _size_bytes = _stable_regular_json(
                path, label="detector fit permit"
            )
            _validate_detector_fit_permit(existing, freshest)
            return existing
        except MethodologicalHalt:
            consumed = _detector_current_gate_receipt(action, freshest)
            if consumed is not None:
                return consumed
            raise

    value: dict[str, Any] = {
        "schema_version": DETECTOR_FIT_PERMIT_SCHEMA,
        "run_identity_sha256": freshest["run_identity_sha256"],
        "task_identity_sha256": next_fit["task_identity_sha256"],
        "fit_gate_nonce": freshest["fit_gate_nonce"],
        "invocation_pid": pid,
        "invocation_start_ticks": start_ticks,
        "next_fit_upper_seconds": upper_seconds,
        "issued_at_utc": utc_now(),
    }
    value["permit_sha256"] = canonical_json_sha256(value)
    try:
        atomic_publish_once_bytes(path, _json_bytes(value))
    except FileExistsError:
        try:
            existing, _raw_sha256, _size_bytes = _stable_regular_json(
                path, label="detector fit permit"
            )
            _validate_detector_fit_permit(existing, freshest)
            return existing
        except MethodologicalHalt:
            consumed = _detector_current_gate_receipt(action, freshest)
            if consumed is not None:
                return consumed
            raise
    emit_event(
        "detector_fit_permit_issued",
        action_id=action.action_id,
        pid=pid,
        process_start_ticks=start_ticks,
        fit_index=next_fit["index"],
        detector_name=next_fit["detector_name"],
        task_identity_sha256=next_fit["task_identity_sha256"],
        fit_gate_nonce=freshest["fit_gate_nonce"],
        next_fit_upper_seconds=upper_seconds,
        permit_sha256=value["permit_sha256"],
        cumulative_actual_gpu_hours=budget["cumulative_actual_gpu_hours"],
        revised_upper_gpu_hours=budget["revised_upper_gpu_hours"],
    )
    return value


def close_detector_gpu_interval_after_exit(
    action: Action,
    status: Mapping[str, Any] | None,
    *,
    pid: int,
    start_ticks: int,
    observed_absent_at: datetime | None = None,
) -> dict[str, Any]:
    """Durably charge through the first identity-checked absent observation."""

    if status is None:
        raise MethodologicalHalt(
            "detector exited before publishing signed CUDA charge history"
        )
    value = dict(status)
    if (
        int(value.get("pid", -1)) != pid
        or int(value.get("process_start_ticks", -1)) != start_ticks
    ):
        raise MethodologicalHalt(
            "detector exit status does not match the observed process identity"
        )
    accounting = value.get("gpu_accounting")
    _validate_detector_gpu_accounting(accounting, live=True)
    assert isinstance(accounting, dict)
    intervals = [dict(row) for row in accounting["intervals"]]
    current = intervals[-1]
    if (
        int(current.get("pid", -1)) != pid
        or int(current.get("process_start_ticks", -1)) != start_ticks
    ):
        raise MethodologicalHalt(
            "detector live GPU interval does not match the exited process"
        )
    absence = (observed_absent_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    )
    started = _parse_aware_time(
        current.get("started_at_utc"), "detector exited interval start"
    )
    prior_end = _parse_aware_time(
        current.get("completed_at_utc")
        or current.get("last_observed_at_utc"),
        "detector exited interval prior observation",
    )
    # Never shrink a detector-authored interval if clock sampling or a supplied
    # absence timestamp is older.  The first confirmed-absent observation is a
    # conservative lower bound only when it extends the durable charge history.
    absence = max(absence, prior_end)
    if absence < started:
        raise MethodologicalHalt("detector absence precedes its GPU interval")
    closure = absence.isoformat()
    current["completed_at_utc"] = closure
    current["last_observed_at_utc"] = closure
    current["elapsed_seconds"] = (absence - started).total_seconds()
    intervals[-1] = current
    accounting = dict(accounting)
    accounting["intervals"] = intervals
    accounting["cumulative_elapsed_seconds"] = sum(
        float(row["elapsed_seconds"]) for row in intervals
    )
    value["gpu_accounting"] = accounting
    value["updated_at_utc"] = closure
    value["state"] = "supervisor_observed_process_exit"
    recovered = list(value.get("recovered_errors", []))
    recovered.append(
        {
            "type": "supervisor_closed_exited_gpu_interval",
            "pid": pid,
            "process_start_ticks": start_ticks,
            "observed_absent_at_utc": closure,
            "action": "resume_from_latest_valid_fit_checkpoint",
        }
    )
    value["recovered_errors"] = recovered
    value.pop("status_sha256", None)
    value["status_sha256"] = canonical_json_sha256(value)
    atomic_write_json(_detector_status_path(action), value)
    # Re-read the durable bytes and require a closed final interval.
    observed = read_detector_status(action)
    if observed is None:
        raise MethodologicalHalt("detector exit closure was not durably published")
    _validate_detector_gpu_accounting(observed.get("gpu_accounting"), live=False)
    emit_event(
        "detector_gpu_interval_closed_after_exit",
        action_id=action.action_id,
        pid=pid,
        process_start_ticks=start_ticks,
        observed_absent_at_utc=closure,
        cumulative_detector_gpu_seconds=accounting[
            "cumulative_elapsed_seconds"
        ],
    )
    return observed


def _gpu_compute_pids() -> set[int]:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=GPU_OCCUPANCY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise InterfaceHalt("pinned GPU occupancy query timed out") from exc
    if query.returncode != 0:
        raise InterfaceHalt("cannot query pinned GPU occupancy: {}".format(query.stderr.strip()))
    pids: set[int] = set()
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            raise InterfaceHalt("malformed nvidia-smi compute row: {!r}".format(line))
        if fields[0] == GPU_UUID:
            pids.add(int(fields[1]))
    return pids


def ensure_gpu_idle() -> None:
    occupied = _gpu_compute_pids()
    if occupied:
        raise TransientGPUOccupancy(
            "authorized GPU is occupied before launch by PID(s): {}".format(
                ", ".join(map(str, sorted(occupied)))
            )
        )


def _normalized_output_argument(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _process_matches_action(argv: Sequence[str], action: Action) -> bool:
    if action.output_dir is None:
        return False
    if _arg_value(argv, "--gpu-uuid") != GPU_UUID:
        return False
    if _arg_value(argv, "--context") != "4096" or _arg_value(argv, "--n-gpu-layers") != "-1":
        return False
    if "--limit" in argv or "--max-pending" in argv:
        return False
    if any(
        flag in argv
        for flag in (
            "--dry-run",
            "--config-dir",
            "--project-root",
            "--threads",
            "--verbose-model",
        )
    ):
        return False
    if _normalized_output_argument(_arg_value(argv, "--output-dir")) != action.output_dir.resolve():
        return False
    if action.kind == "runner":
        core_matches = (
            "scripts/run_revision_matrix.py" in argv
            and _arg_value(argv, "--stage") == action.stage
            and _arg_value(argv, "--model") == action.model_id
        )
        if not core_matches:
            return False
        # Robustness is scientifically bound to all three exact source roots.
        # Comparing these during orphan attachment prevents a manually started
        # process with a plausible output directory but different lineage from
        # being adopted as the frozen action.
        for flag in (
            "--primary-results-root",
            "--ablation-results-root",
            "--robustness-results-root",
        ):
            expected = _arg_value(action.argv, flag)
            observed = _arg_value(argv, flag)
            if _normalized_output_argument(observed) != _normalized_output_argument(
                expected
            ):
                return False
        return True
    return (
        "scripts/run_revision_evaluator.py" in argv
        and _arg_value(argv, "--evaluator-model") == action.model_id
        and _arg_value(argv, "--source-stage") == action.source_stage
        and _normalized_output_argument(_arg_value(argv, "--source-results-root"))
        == _normalized_output_argument(
            _arg_value(action.argv, "--source-results-root")
        )
    )


def _previous_attached_pid(action: Action) -> int | None:
    path = PROJECT_ROOT / STATE_PATH
    if not path.exists():
        return None
    state = read_json(path, label="orchestrator state")
    if state.get("schema_version") != STATE_SCHEMA or _state_hash(state) != state.get("orchestrator_state_sha256"):
        raise MethodologicalHalt("orchestrator state schema/self-hash mismatch")
    if (
        state.get("status") == "attached_existing_gpu_action"
        and state.get("current_action_id") == action.action_id
        and state.get("active_pid") is not None
    ):
        return int(state["active_pid"])
    return None


def wait_for_existing_gpu_occupancy(
    action: Action,
    *,
    projection: Mapping[str, Any],
    retries: dict[str, int],
    max_retries: int,
    poll_seconds: float,
) -> bool:
    """Attach to an exact orphan or wait for another process; never duplicate it."""

    occupied = _gpu_compute_pids()
    if not occupied:
        prior_pid = _previous_attached_pid(action)
        if prior_pid is not None and shard_status(action) != "complete":
            retries[action.action_id] = retries.get(action.action_id, 0) + 1
            record_recoverable_error(
                action,
                retries[action.action_id],
                "attached_runner_disappeared_before_completion",
                "Previously attached GPU PID {} is no longer active.".format(prior_pid),
            )
            if retries[action.action_id] > max_retries:
                raise OrchestratorError(
                    "retry ceiling exceeded for attached {}".format(action.action_id)
                )
        return False

    commands = _process_cmdlines()
    visible = {pid: commands[pid] for pid in occupied if pid in commands}
    managed = {
        pid: argv
        for pid, argv in visible.items()
        if "scripts/run_revision_matrix.py" in argv
        or "scripts/run_revision_evaluator.py" in argv
    }
    exact = [pid for pid, argv in managed.items() if _process_matches_action(argv, action)]
    primary = [
        pid
        for pid, argv in managed.items()
        if "scripts/run_revision_matrix.py" in argv
        and _arg_value(argv, "--stage") == "primary_v2"
        and _arg_value(argv, "--gpu-uuid") == GPU_UUID
        and _arg_value(argv, "--context") == "4096"
        and _arg_value(argv, "--n-gpu-layers") == "-1"
        and "--limit" not in argv
    ]
    unexpected_managed = set(managed) - set(exact) - set(primary)
    concurrent = set(occupied) - set(exact or primary)
    if (
        unexpected_managed
        or len(exact) > 1
        or len(primary) > 1
        or (exact and primary)
        or ((exact or primary) and concurrent)
    ):
        raise MethodologicalHalt(
            "out-of-order, overlapping, or multiply active GPU process(es): {}".format(
                ", ".join(map(str, sorted(occupied)))
            )
        )
    progress_deferral: DeferrableProgressRefresh | None = None
    try:
        progress = refresh_progress()
    except DeferrableProgressRefresh as exc:
        progress = published_progress_after_source_race()
        progress_deferral = exc
    budget = calculate_budget(projection, progress)
    enforce_budget(budget)
    if exact:
        status = "attached_existing_gpu_action"
        message = "Attached to an exact already-running checkpointed action."
        active_pid = exact[0]
    elif primary:
        status = "waiting_for_primary_process_finalization"
        message = "Primary checkpoint is complete; waiting for its process to release the GPU."
        active_pid = primary[0]
    else:
        status = "waiting_for_gpu_availability"
        message = "Pinned GPU is occupied by an external process; no duplicate launch attempted."
        active_pid = min(occupied)
    if progress_deferral is not None:
        message += (
            " Canonical progress refresh was operationally deferred; "
            "the attached/occupied process and action retry count are unchanged."
        )
        emit_progress_refresh_deferral(
            progress_deferral,
            action=action,
            retries=retries,
            active_pid=active_pid,
            budget=budget,
            occupancy_status=status,
        )
    write_state(
        status=status,
        message=message,
        action=action,
        retries=retries,
        progress=progress,
        budget=budget,
        active_pid=active_pid,
    )
    time.sleep(poll_seconds)
    return True


NONRECOVERABLE_TEXT = (
    "hash mismatch",
    "self-hash mismatch",
    "scientific identity",
    "payload-fidelity contract",
    "result schema",
    "license",
    "unsupported schema",
    "violates",
    "frozen plan",
    "artifact verification",
)


def _nonrecoverable_output(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in NONRECOVERABLE_TEXT)


def _log_path(action: Action) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = action.action_id.replace(":", "__").replace("/", "_")
    return PROJECT_ROOT / LOG_ROOT / "{}.{}.log".format(safe, stamp)


def run_gpu_process(
    action: Action,
    argv: Sequence[str],
    *,
    projection: Mapping[str, Any],
    retries: Mapping[str, int],
    poll_seconds: float,
) -> tuple[int, str]:
    validate_gpu_action(action)
    progress = refresh_progress()
    budget = calculate_budget(projection, progress)
    enforce_budget(budget)
    ensure_gpu_idle()
    log_path = _log_path(action)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(argv),
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    emit_event(
        "gpu_action_launched",
        action_id=action.action_id,
        pid=process.pid,
        argv=list(argv),
        log_path=str(log_path),
        revised_upper_gpu_hours=budget["revised_upper_gpu_hours"],
    )
    last_long_update = time.monotonic()
    while process.poll() is None:
        time.sleep(poll_seconds)
        progress_deferral: DeferrableProgressRefresh | None = None
        try:
            progress = refresh_progress()
        except DeferrableProgressRefresh as exc:
            progress = published_progress_after_source_race()
            progress_deferral = exc
        budget = calculate_budget(projection, progress)
        try:
            enforce_budget(budget)
        except BudgetHalt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
            write_state(
                status="halted_budget_ceiling",
                message="GPU child terminated because the hard ceiling gate closed.",
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
                active_pid=process.pid,
            )
            raise
        if progress_deferral is not None:
            emit_progress_refresh_deferral(
                progress_deferral,
                action=action,
                retries=retries,
                active_pid=process.pid,
                budget=budget,
                occupancy_status="running_managed_gpu_action",
            )
        write_state(
            status="running",
            message=(
                "Serial checkpointed GPU action is running; canonical progress "
                "refresh is operationally deferred."
                if progress_deferral is not None
                else "Serial checkpointed GPU action is running."
            ),
            action=action,
            retries=retries,
            progress=progress,
            budget=budget,
            active_pid=process.pid,
        )
        if time.monotonic() - last_long_update >= 6 * 3600:
            emit_event(
                "six_hour_progress",
                action_id=action.action_id,
                counts=progress.get("counts"),
                throughput=progress.get("throughput"),
                eta=progress.get("eta"),
                cumulative_actual_gpu_hours=budget["cumulative_actual_gpu_hours"],
            )
            last_long_update = time.monotonic()
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except OSError:
        tail = ""
    return int(process.returncode or 0), tail


def _detector_child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "RANKCLOAK_DETECTOR_DEVICE": DETECTOR_DEVICE,
            "RANKCLOAK_DETECTOR_GPU_UUID": GPU_UUID,
            "RANKCLOAK_DETECTOR_WORKERS": str(DETECTOR_WORKERS),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return environment


def _detector_pid_is_live(action: Action, pid: int, start_ticks: int) -> bool:
    observed_ticks = _process_start_ticks(pid)
    if observed_ticks is None or observed_ticks != start_ticks:
        return False
    exact = _exact_detector_pid(action)
    return exact == pid


def _terminate_detector_process(action: Action, pid: int, start_ticks: int) -> None:
    """Terminate only the exact identity-rechecked detector PID."""

    if not _detector_pid_is_live(action, pid, start_ticks):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + DETECTOR_GRACEFUL_STOP_SECONDS
    while time.monotonic() < deadline:
        if not _detector_pid_is_live(action, pid, start_ticks):
            return
        time.sleep(0.25)
    if _detector_pid_is_live(action, pid, start_ticks):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _kill_detector_at_hard_ceiling(
    action: Action, pid: int, start_ticks: int
) -> None:
    """Freeze then kill only the exact detector; never finish a fit past 165h."""

    if not _detector_pid_is_live(action, pid, start_ticks):
        return
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    try:
        emit_event(
            "detector_frozen_at_hard_ceiling",
            action_id=action.action_id,
            pid=pid,
            process_start_ticks=start_ticks,
            unfinished_fit_may_be_discarded=True,
            valid_fit_checkpoints_preserved=True,
        )
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_detector_absence(
    action: Action, pid: int, start_ticks: int, *, timeout: float = 10.0
) -> datetime:
    deadline = time.monotonic() + timeout
    while _detector_pid_is_live(action, pid, start_ticks):
        if time.monotonic() >= deadline:
            raise OrchestratorError(
                "detector process remained live after an exact stop request"
            )
        time.sleep(0.05)
    return datetime.now(timezone.utc)


def _stop_detector_and_close_interval(
    action: Action,
    *,
    pid: int,
    start_ticks: int,
    status: Mapping[str, Any] | None,
    immediate: bool,
) -> dict[str, Any] | None:
    if immediate:
        _kill_detector_at_hard_ceiling(action, pid, start_ticks)
    else:
        _terminate_detector_process(action, pid, start_ticks)
    absent_at = _wait_for_detector_absence(action, pid, start_ticks)
    freshest = read_detector_status(action) or (
        dict(status) if status is not None else None
    )
    if freshest is None:
        raise MethodologicalHalt(
            "stopped detector lacks signed CUDA interval history"
        )
    accounting = freshest.get("gpu_accounting")
    _validate_detector_gpu_accounting(accounting, live=True)
    assert isinstance(accounting, dict)
    freshest = close_detector_gpu_interval_after_exit(
        action,
        freshest,
        pid=pid,
        start_ticks=start_ticks,
        observed_absent_at=absent_at,
    )
    return freshest


def _stop_exact_detector_if_live(
    action: Action, *, immediate: bool = False
) -> None:
    pid = _exact_detector_pid(action)
    if pid is None:
        return
    start_ticks = _process_start_ticks(pid)
    if start_ticks is None:
        raise OrchestratorError("exact detector process identity disappeared")
    try:
        status = read_detector_status(action)
    except (MethodologicalHalt, OrchestratorError):
        if immediate:
            _kill_detector_at_hard_ceiling(action, pid, start_ticks)
        else:
            _terminate_detector_process(action, pid, start_ticks)
        _wait_for_detector_absence(action, pid, start_ticks)
        raise
    _stop_detector_and_close_interval(
        action,
        pid=pid,
        start_ticks=start_ticks,
        status=status,
        immediate=immediate,
    )


def run_checkpointed_detector_process(
    action: Action,
    *,
    projection: Mapping[str, Any],
    retries: Mapping[str, int],
    poll_seconds: float,
) -> tuple[int, str]:
    """Launch or attach to the exact detector and monitor its durable status."""

    benchmark_raw = _arg_value(action.argv, "--benchmark-task-index")
    benchmark_task_index = None if benchmark_raw is None else int(benchmark_raw)
    equivalence_role = _arg_value(action.argv, "--equivalence-role")
    if equivalence_role is None:
        validate_detector_action(
            action, benchmark_task_index=benchmark_task_index
        )
    else:
        equivalence_task_raw = _arg_value(
            action.argv, "--equivalence-task-index"
        )
        if equivalence_task_raw is None:
            raise MethodologicalHalt(
                "monitored detector equivalence action lacks task index"
            )
        validate_detector_equivalence_action(
            action,
            task_index=int(equivalence_task_raw),
            role=equivalence_role,
            contract=load_command_contract(DEFAULT_COMMAND_CONTRACT),
        )
    verify_detector_execution_policy()
    existing_pid = _exact_detector_pid(action)
    existing_start_ticks = (
        None if existing_pid is None else _process_start_ticks(existing_pid)
    )
    if existing_pid is not None and existing_start_ticks is None:
        raise OrchestratorError("exact detector process identity disappeared")
    status: dict[str, Any] | None = None
    try:
        status = read_detector_status(action)
        if existing_pid is None and status is not None:
            accounting = status.get("gpu_accounting")
            _validate_detector_gpu_accounting(accounting, live=True)
            assert isinstance(accounting, dict)
            final_interval = accounting["intervals"][-1]
            if final_interval.get("completed_at_utc") is None:
                status = close_detector_gpu_interval_after_exit(
                    action,
                    status,
                    pid=int(status["pid"]),
                    start_ticks=int(status["process_start_ticks"]),
                )
            if _status_declares_expected_detector_candidate(action, status):
                # The finalizer is deliberately idempotent for every crash
                # boundary: candidate, published output, receipt, marker, and
                # complete status.  Calling it even from complete repairs only
                # a missing deterministic post-publication seal and never
                # reruns a detector fit.
                _finalize_detector_after_confirmed_exit(action, status)
                _ensure_finalized_detector_ledger(action)
                return 0, "supervisor finalized the durable CUDA candidate"
        progress = refresh_progress()
        prior_live_seconds = _detector_unaccounted_gpu_seconds(
            progress, status
        )
        if (
            existing_pid is not None
            and existing_start_ticks is not None
        ):
            prior_live_seconds = _detector_unaccounted_gpu_seconds(
                progress,
                status,
                live_pid=existing_pid,
                live_start_ticks=existing_start_ticks,
            )
        budget = calculate_budget(
            projection,
            progress,
            live_detector_gpu_seconds=prior_live_seconds,
            live_detector_remaining_seconds=(
                _detector_remaining_gpu_seconds(status)
            ),
        )
        enforce_budget(budget)
    except DeferrableProgressRefresh:
        raise
    except BudgetHalt:
        if existing_pid is not None and existing_start_ticks is not None:
            _stop_detector_and_close_interval(
                action,
                pid=existing_pid,
                start_ticks=existing_start_ticks,
                status=status,
                immediate=True,
            )
        raise
    except (MethodologicalHalt, OrchestratorError):
        if existing_pid is not None and existing_start_ticks is not None:
            _stop_detector_and_close_interval(
                action,
                pid=existing_pid,
                start_ticks=existing_start_ticks,
                status=status,
                immediate=False,
            )
        raise

    process: subprocess.Popen[str] | None = None
    log_path = _log_path(action)
    if existing_pid is None:
        if status is None and _detector_checkpoint_state_exists(action):
            raise MethodologicalHalt(
                "detector checkpoints exist without signed CUDA charge history; "
                "refusing a resume that could omit prior GPU time"
            )
        ensure_gpu_idle()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                list(action.argv),
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=_detector_child_environment(),
            )
        pid = process.pid
        start_ticks = _process_start_ticks(pid)
        if start_ticks is None:
            raise OrchestratorError("launched detector process identity is unavailable")
        emit_event(
            "checkpointed_detector_launched",
            action_id=action.action_id,
            pid=pid,
            argv=list(action.argv),
            log_path=str(log_path),
            resumed=True,
            completed_fits=(status or {}).get("completed_fit_count", 0),
            total_fits=DETECTOR_TOTAL_FITS,
            revised_upper_gpu_hours=budget["revised_upper_gpu_hours"],
        )
    else:
        pid = existing_pid
        assert existing_start_ticks is not None
        start_ticks = existing_start_ticks
        emit_event(
            "checkpointed_detector_attached",
            action_id=action.action_id,
            pid=pid,
            argv=list(action.argv),
            completed_fits=(status or {}).get("completed_fit_count"),
            total_fits=DETECTOR_TOTAL_FITS,
        )

    last_long_update = time.monotonic()
    monitor_started = time.monotonic()
    while True:
        if process is not None:
            running = process.poll() is None
        else:
            running = _detector_pid_is_live(action, pid, start_ticks)
        if not running:
            break
        time.sleep(min(poll_seconds, DETECTOR_MONITOR_MAX_POLL_SECONDS))
        if process is not None and process.poll() is not None:
            break
        if process is None and not _detector_pid_is_live(action, pid, start_ticks):
            break

        try:
            verify_detector_execution_policy()
            status = read_detector_status(action, expected_pid=pid)
            status_matches_process = bool(
                status is not None
                and int(status.get("pid", -1)) == pid
                and int(status.get("process_start_ticks", -1)) == start_ticks
            )
            if status_matches_process:
                updated = _parse_aware_time(
                    status.get("updated_at_utc"), "detector status updated_at_utc"
                )
                status_age = (
                    datetime.now(timezone.utc) - updated
                ).total_seconds()
            else:
                status_age = time.monotonic() - monitor_started
            if status_age > DETECTOR_STATUS_STALE_SECONDS:
                status = _stop_detector_and_close_interval(
                    action,
                    pid=pid,
                    start_ticks=start_ticks,
                    status=status,
                    immediate=False,
                )
                if process is not None:
                    try:
                        process.wait(timeout=DETECTOR_GRACEFUL_STOP_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                return 75, (
                    "checkpointed detector heartbeat was absent or stale for "
                    "{:.1f} seconds; exact process stopped for checkpoint resume"
                ).format(status_age)

            if status_matches_process and status is not None and status.get(
                "state"
            ) in {"running_fit", "stop_requested_finishing_current_fit"}:
                current_fit = status.get("current_fit")
                assert isinstance(current_fit, dict)
                detector_name = str(current_fit.get("detector_name", ""))
                fit_upper = _detector_fit_watchdog_seconds(
                    action, detector_name
                )
                fit_elapsed = float(status.get("current_fit_elapsed_seconds", 0.0))
                if fit_upper is None or fit_elapsed > fit_upper:
                    status = _stop_detector_and_close_interval(
                        action,
                        pid=pid,
                        start_ticks=start_ticks,
                        status=status,
                        immediate=False,
                    )
                    return 75, (
                        "detector fit {} exceeded its {:.1f}-second watchdog at "
                        "{:.1f} seconds; process stopped for diagnosis"
                    ).format(detector_name, float(fit_upper or 0.0), fit_elapsed)
                internal = status.get("current_fit_progress")
                if internal is None and fit_elapsed > 60.0:
                    status = _stop_detector_and_close_interval(
                        action,
                        pid=pid,
                        start_ticks=start_ticks,
                        status=status,
                        immediate=False,
                    )
                    return 75, (
                        "detector fit produced no epoch/batch progress within "
                        "{:.1f} seconds; process stopped for diagnosis"
                    ).format(fit_elapsed)
                if internal is not None:
                    progress_updated = _validate_detector_internal_progress(
                        internal
                    )
                    internal_age = (
                        datetime.now(timezone.utc) - progress_updated
                    ).total_seconds()
                    if internal_age > DETECTOR_INTERNAL_PROGRESS_STALE_SECONDS:
                        status = _stop_detector_and_close_interval(
                            action,
                            pid=pid,
                            start_ticks=start_ticks,
                            status=status,
                            immediate=False,
                        )
                        return 75, (
                            "detector epoch/batch progress was unchanged for "
                            "{:.1f} seconds; process stopped for diagnosis"
                        ).format(internal_age)

            progress_deferral: DeferrableProgressRefresh | None = None
            try:
                progress = refresh_progress()
            except DeferrableProgressRefresh as exc:
                progress = published_progress_after_source_race()
                progress_deferral = exc
            live_seconds = _detector_unaccounted_gpu_seconds(
                progress,
                status,
                live_pid=pid,
                live_start_ticks=start_ticks,
            )
            budget = calculate_budget(
                projection,
                progress,
                live_detector_gpu_seconds=live_seconds,
                live_detector_remaining_seconds=(
                    _detector_remaining_gpu_seconds(status)
                ),
            )
            enforce_budget(budget)
        except BudgetHalt:
            status = _stop_detector_and_close_interval(
                action,
                pid=pid,
                start_ticks=start_ticks,
                status=status,
                immediate=True,
            )
            closed_seconds = _detector_unaccounted_gpu_seconds(
                progress, status
            )
            budget = calculate_budget(
                projection,
                progress,
                live_detector_gpu_seconds=closed_seconds,
                live_detector_remaining_seconds=(
                    _detector_remaining_gpu_seconds(status)
                ),
            )
            write_state(
                status="halted_budget_ceiling",
                message=(
                    "Checkpointed detector terminated because the 165 GPU-hour "
                    "hard ceiling gate closed."
                ),
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
                active_pid=pid,
                detector_status=status,
            )
            raise
        except (MethodologicalHalt, OrchestratorError):
            _stop_detector_and_close_interval(
                action,
                pid=pid,
                start_ticks=start_ticks,
                status=status,
                immediate=False,
            )
            raise

        occupancy_verified = True
        try:
            occupied = _gpu_compute_pids()
        except InterfaceHalt as exc:
            emit_event(
                "detector_gpu_occupancy_query_deferred",
                action_id=action.action_id,
                pid=pid,
                error=str(exc),
                action_retry_count_changed=False,
            )
            occupied = {pid}
            occupancy_verified = False
        unexpected = occupied - {pid}
        if unexpected:
            _stop_detector_and_close_interval(
                action,
                pid=pid,
                start_ticks=start_ticks,
                status=status,
                immediate=False,
            )
            raise MethodologicalHalt(
                "another process overlaps checkpointed detector GPU execution: {}".format(
                    sorted(unexpected)
                )
            )
        if (
            status is not None
            and status.get("state") == "awaiting_fit_ceiling_gate"
            and progress_deferral is None
            and occupancy_verified
        ):
            try:
                issue_detector_fit_permit(
                    action,
                    status,
                    budget,
                    pid=pid,
                    start_ticks=start_ticks,
                )
            except BudgetHalt:
                status = _stop_detector_and_close_interval(
                    action,
                    pid=pid,
                    start_ticks=start_ticks,
                    status=status,
                    immediate=True,
                )
                closed_seconds = _detector_unaccounted_gpu_seconds(
                    progress, status
                )
                budget = calculate_budget(
                    projection,
                    progress,
                    live_detector_gpu_seconds=closed_seconds,
                    live_detector_remaining_seconds=(
                        _detector_remaining_gpu_seconds(status)
                    ),
                )
                write_state(
                    status="halted_budget_ceiling",
                    message=(
                        "Checkpointed detector stopped before its next fit "
                        "because the immutable reserve would exceed 165 hours."
                    ),
                    action=action,
                    retries=retries,
                    progress=progress,
                    budget=budget,
                    active_pid=pid,
                    detector_status=status,
                )
                raise
            except (MethodologicalHalt, OrchestratorError):
                _stop_detector_and_close_interval(
                    action,
                    pid=pid,
                    start_ticks=start_ticks,
                    status=status,
                    immediate=False,
                )
                raise
            write_state(
                status="detector_fit_permitted",
                message=(
                    "One signed detector fit was permitted after exact lease, "
                    "GPU-exclusivity, and 165-hour reserve checks."
                ),
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
                active_pid=pid,
                detector_status=status,
            )
        if progress_deferral is not None:
            emit_progress_refresh_deferral(
                progress_deferral,
                action=action,
                retries=retries,
                active_pid=pid,
                budget=budget,
                occupancy_status="running_checkpointed_detector",
            )
        write_state(
            status=(
                "attached_checkpointed_detector"
                if process is None
                else "running_checkpointed_detector"
            ),
            message=(
                "Monitoring one exact checkpointed CUDA detector process; valid "
                "completed fits are immutable and skipped on resume."
            ),
            action=action,
            retries=retries,
            progress=progress,
            budget=budget,
            active_pid=pid,
            detector_status=status,
        )
        if time.monotonic() - last_long_update >= 6 * 3600:
            emit_event(
                "six_hour_progress",
                action_id=action.action_id,
                detector_completed_fits=(status or {}).get("completed_fit_count"),
                detector_total_fits=(status or {}).get("total_fit_count"),
                detector_current_fit=(status or {}).get("current_fit"),
                detector_fits_per_hour=(status or {}).get("fits_per_hour"),
                detector_eta_seconds=(status or {}).get("rolling_eta_seconds"),
                cumulative_actual_gpu_hours=budget["cumulative_actual_gpu_hours"],
            )
            last_long_update = time.monotonic()

    observed_absent_at = datetime.now(timezone.utc)
    status = read_detector_status(action)
    if status is not None:
        accounting = status.get("gpu_accounting")
        _validate_detector_gpu_accounting(accounting, live=True)
        if status.get("state") != "complete":
            status = close_detector_gpu_interval_after_exit(
                action,
                status,
                pid=pid,
                start_ticks=start_ticks,
                observed_absent_at=observed_absent_at,
            )
            if _status_declares_expected_detector_candidate(action, status):
                _finalize_detector_after_confirmed_exit(action, status)
                status = read_detector_status(action)
    else:
        raise MethodologicalHalt(
            "detector exited without signed CUDA interval history"
        )
    kind, _task_index, _role, final_output, _candidate, _receipt = (
        _detector_finalization_contract(action)
    )
    completed = bool(
        final_output.is_file()
        and status
        and status.get("state") == "complete"
        and (
            kind != "detector_run_manifest"
            or int(status.get("completed_fit_count", -1))
            == DETECTOR_TOTAL_FITS
        )
    )
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except OSError:
        tail = ""
    if process is not None:
        return int(process.returncode or 0), tail
    return (
        0
        if completed
        else 1
    ), tail


def execute_checkpointed_detector_action(
    action: Action,
    spec: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    retries: dict[str, int],
    max_retries: int,
    poll_seconds: float,
) -> None:
    substitutions = _format_values()
    try:
        if verify_completion(spec, substitutions):
            return
        if (PROJECT_ROOT / FINAL_PROGRESS_PATH).exists():
            raise MethodologicalHalt(
                "immutable final progress exists while the detector is incomplete"
            )
        verify_downstream_interface(spec, substitutions)
        validate_detector_action(action)
    except (MethodologicalHalt, InterfaceHalt, OrchestratorError):
        _stop_exact_detector_if_live(action)
        raise
    while True:
        try:
            code, detail = run_checkpointed_detector_process(
                action,
                projection=projection,
                retries=retries,
                poll_seconds=poll_seconds,
            )
        except DeferrableProgressRefresh as exc:
            progress = published_progress_after_source_race()
            status = read_detector_status(action)
            live_seconds = _detector_unaccounted_gpu_seconds(
                progress, status
            )
            budget = calculate_budget(
                projection,
                progress,
                live_detector_gpu_seconds=live_seconds,
                live_detector_remaining_seconds=(
                    _detector_remaining_gpu_seconds(status)
                ),
            )
            enforce_budget(budget)
            active_pid = _exact_detector_pid(action)
            emit_progress_refresh_deferral(
                exc,
                action=action,
                retries=retries,
                active_pid=active_pid,
                budget=budget,
                occupancy_status="detector_launch_deferred",
            )
            write_state(
                status="detector_launch_progress_refresh_deferred",
                message=(
                    "Canonical progress refresh is operationally deferred; "
                    "the detector was not launched and no retry was charged."
                ),
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
                active_pid=active_pid,
                detector_status=status,
            )
            time.sleep(poll_seconds)
            continue
        except TransientGPUOccupancy:
            time.sleep(poll_seconds)
            continue
        if code == 0:
            if verify_completion(spec, substitutions):
                retry_count = retries.get(action.action_id, 0)
                if retry_count:
                    emit_event(
                        "action_recovery_succeeded",
                        action_id=action.action_id,
                        retry_count=retry_count,
                        recovery="valid_detector_checkpoints_resumed",
                        completed_fits=DETECTOR_TOTAL_FITS,
                    )
                emit_event(
                    "downstream_action_complete",
                    action_id=action.action_id,
                    completed_fits=DETECTOR_TOTAL_FITS,
                )
                return
            raise InterfaceHalt(
                "detector exited zero without the exact 56-fit final manifest"
            )
        if _nonrecoverable_output(detail):
            raise MethodologicalHalt(
                "nonrecoverable checkpointed detector failure: {}".format(
                    detail[-2000:]
                )
            )
        retries[action.action_id] = retries.get(action.action_id, 0) + 1
        record_recoverable_error(
            action,
            retries[action.action_id],
            "checkpointed_detector_process_incomplete",
            detail,
        )
        if retries[action.action_id] > max_retries:
            raise OrchestratorError(
                "retry ceiling exceeded for {}".format(action.action_id)
            )
        retry_deferral: DeferrableProgressRefresh | None = None
        try:
            progress = refresh_progress()
        except DeferrableProgressRefresh as exc:
            progress = published_progress_after_source_race()
            retry_deferral = exc
        status = read_detector_status(action)
        live_seconds = _detector_unaccounted_gpu_seconds(progress, status)
        budget = calculate_budget(
            projection,
            progress,
            live_detector_gpu_seconds=live_seconds,
            live_detector_remaining_seconds=(
                _detector_remaining_gpu_seconds(status)
            ),
        )
        enforce_budget(budget)
        if retry_deferral is not None:
            emit_progress_refresh_deferral(
                retry_deferral,
                action=action,
                retries=retries,
                active_pid=None,
                budget=budget,
                occupancy_status="retrying_checkpointed_detector",
            )
        write_state(
            status="retrying_checkpointed_detector",
            message=(
                "Recoverable detector exit was durably recorded; restarting "
                "from the latest strictly valid per-fit checkpoint."
            ),
            action=action,
            retries=retries,
            progress=progress,
            budget=budget,
            detector_status=status,
        )
        time.sleep(poll_seconds)


def _benchmark_detector_action(
    contract: Mapping[str, Any], task_index: int
) -> tuple[Action, dict[str, Any], Path]:
    if task_index not in DETECTOR_BENCHMARK_TASKS:
        raise OrchestratorError(
            "--detector-benchmark-task-index must be exactly 0 or 1"
        )
    action, spec = next(
        (item for item in downstream_actions(contract) if item[1].get("operation_id") == "detector")
    )
    output = Path(
        _format_values()["detector_benchmark_output_{}".format(task_index)]
    ).resolve()
    argv = action.argv + (
        "--benchmark-one-fit",
        "--benchmark-task-index",
        str(task_index),
        "--benchmark-output",
        str(output),
    )
    benchmark = Action(
        action_id="downstream:detector:benchmark:{}".format(task_index),
        stage=action.stage,
        kind=action.kind,
        argv=argv,
        output_dir=action.output_dir,
        gpu=True,
    )
    validate_detector_action(
        benchmark, benchmark_task_index=task_index
    )
    return benchmark, spec, output


def _detector_equivalence_action(
    contract: Mapping[str, Any], *, task_index: int, role: str
) -> Action:
    if task_index not in DETECTOR_BENCHMARK_TASKS or role not in {
        "cuda",
        "cuda_repeat",
    }:
        raise MethodologicalHalt(
            "detector CUDA reproducibility action identity is invalid"
        )
    values = _format_values()
    prefix = "detector_equivalence_{}_{}".format(task_index, role)
    argv = _detector_equivalence_action_argv(
        contract, task_index=task_index, role=role
    )
    action = Action(
        action_id="downstream:detector:equivalence:{}:{}".format(
            task_index, role
        ),
        stage="neural_detector",
        kind="downstream",
        argv=argv,
        output_dir=Path(values[prefix + "_output_dir"]),
        gpu=True,
    )
    validate_detector_equivalence_action(
        action, task_index=task_index, role=role, contract=contract
    )
    return action


def validate_detector_equivalence_action(
    action: Action,
    *,
    task_index: int,
    role: str,
    contract: Mapping[str, Any],
) -> None:
    values = _format_values()
    prefix = "detector_equivalence_{}_{}".format(task_index, role)
    expected = _detector_equivalence_action_argv(
        contract, task_index=task_index, role=role
    )
    if (
        action.action_id
        != "downstream:detector:equivalence:{}:{}".format(task_index, role)
        or action.stage != "neural_detector"
        or action.kind != "downstream"
        or action.gpu is not True
        or action.argv != expected
        or action.output_dir is None
        or action.output_dir.resolve()
        != Path(values[prefix + "_output_dir"]).resolve()
    ):
        raise MethodologicalHalt(
            "detector CUDA reproducibility action differs from its isolated contract"
        )
    paths = {
        _detector_checkpoint_dir(action),
        _detector_status_path(action),
        _detector_fit_permit_path(action),
        _detector_fit_permit_receipt_dir(action),
        Path(values[prefix + "_artifact"]).resolve(),
    }
    if len(paths) != 5:
        raise MethodologicalHalt(
            "detector CUDA reproducibility paths overlap"
        )


def _detector_equivalence_action_argv(
    contract: Mapping[str, Any], *, task_index: int, role: str
) -> tuple[str, ...]:
    if task_index not in DETECTOR_BENCHMARK_TASKS or role not in {
        "cuda",
        "cuda_repeat",
    }:
        raise MethodologicalHalt(
            "detector CUDA reproducibility action identity is invalid"
        )
    values = _format_values()
    prefix = "detector_equivalence_{}_{}".format(task_index, role)
    if role == "cuda":
        base, _spec = next(
            item
            for item in downstream_actions(contract)
            if item[1].get("operation_id") == "detector"
        )
        argv = base.argv
    else:
        argv = (
            str(PROJECT_ROOT / ".venv/bin/python"),
            "scripts/run_revision_detectors.py",
            "--input",
            values["primary_detector_corpus"],
            "--preprocessing-manifest",
            values["primary_preprocessing_manifest"],
            "--output-dir",
            values[prefix + "_output_dir"],
            "--execution-policy",
            values["detector_execution_policy"],
            "--resume",
            "--overwrite",
            "--device",
            DETECTOR_DEVICE,
            "--workers",
            str(DETECTOR_WORKERS),
            "--checkpoint-dir",
            values[prefix + "_checkpoint_dir"],
            "--status-file",
            values[prefix + "_status_file"],
            "--gpu-uuid",
            GPU_UUID,
            "--fit-permit-file",
            values[prefix + "_fit_permit_file"],
            "--fit-permit-receipt-dir",
            values[prefix + "_fit_permit_receipt_dir"],
        )
    return argv + (
        "--equivalence-role",
        role,
        "--equivalence-task-index",
        str(task_index),
        "--equivalence-artifact",
        values[prefix + "_artifact"],
    )


def _verify_detector_benchmark_runtime_provenance(
    run_identity: Mapping[str, Any]
) -> None:
    lineage = run_identity.get("lineage")
    if not isinstance(lineage, dict):
        raise InterfaceHalt("detector benchmark run identity lacks lineage")
    policy = lineage.get("execution_policy")
    environment = lineage.get("environment_binding")
    policy_path = _detector_execution_policy_path()
    if (
        not isinstance(policy, dict)
        or Path(str(policy.get("path", ""))).resolve() != policy_path
        or policy.get("sha256") != DETECTOR_EXECUTION_POLICY_SHA256
        or policy.get("policy_sha256")
        != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or int(policy.get("size_bytes", -1)) != policy_path.stat().st_size
        or not isinstance(environment, dict)
        or environment.get("schema_version")
        != "rankcloak-revision-detector-environment-binding-v1"
        or environment.get("verification_status") != "ok"
        or not isinstance(environment.get("required_files"), dict)
        or set(environment["required_files"])
        != {
            "environment_manifest.json",
            "scientific_pins.json",
            "CHECKSUMS.sha256",
        }
    ):
        raise InterfaceHalt("detector benchmark policy/environment binding is invalid")
    for declaration in environment["required_files"].values():
        if not isinstance(declaration, dict):
            raise InterfaceHalt("detector benchmark environment declaration is invalid")
        source = Path(str(declaration.get("path", "")))
        if (
            not source.is_file()
            or source.is_symlink()
            or file_sha256(source) != declaration.get("sha256")
            or source.stat().st_size != int(declaration.get("size_bytes", -1))
        ):
            raise InterfaceHalt(
                "detector benchmark frozen environment identity changed"
            )


def _verify_detector_equivalence_policy_identity(value: Any) -> None:
    policy = verify_detector_execution_policy()
    if not isinstance(value, dict):
        raise MethodologicalHalt(
            "detector equivalence artifact lacks policy/environment identity"
        )
    environment = value.get("environment_binding")
    root = (PROJECT_ROOT / "environment/revision_v1").resolve()
    required = None if not isinstance(environment, dict) else environment.get(
        "required_files"
    )
    if (
        set(value)
        != {
            "schema_version",
            "execution_policy_path",
            "execution_policy_sha256",
            "execution_policy_content_sha256",
            "environment_binding",
            "environment_binding_sha256",
            "equivalence_policy",
            "equivalence_policy_sha256",
        }
        or value.get("schema_version")
        != "rankcloak-revision-detector-equivalence-policy-identity-v1"
        or Path(str(value.get("execution_policy_path", ""))).resolve()
        != _detector_execution_policy_path()
        or value.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
        or value.get("execution_policy_content_sha256")
        != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        or value.get("equivalence_policy") != policy.get("equivalence")
        or value.get("equivalence_policy_sha256")
        != canonical_json_sha256(policy["equivalence"])
        or not isinstance(environment, dict)
        or value.get("environment_binding_sha256")
        != canonical_json_sha256(environment)
        or environment.get("schema_version")
        != "rankcloak-revision-detector-environment-binding-v1"
        or Path(str(environment.get("root", ""))).resolve() != root
        or environment.get("verification_status") != "ok"
        or not isinstance(required, dict)
        or set(required)
        != {
            "environment_manifest.json",
            "scientific_pins.json",
            "CHECKSUMS.sha256",
        }
    ):
        raise MethodologicalHalt(
            "detector equivalence policy/environment identity differs"
        )
    for name, declaration in required.items():
        expected = root / name
        if (
            not isinstance(declaration, dict)
            or Path(str(declaration.get("path", ""))).resolve() != expected
            or expected.is_symlink()
            or not expected.is_file()
            or file_sha256(expected) != declaration.get("sha256")
            or expected.stat().st_size != int(declaration.get("size_bytes", -1))
        ):
            raise MethodologicalHalt(
                "detector equivalence environment source bytes differ"
            )
    manifest = read_json(
        root / "environment_manifest.json",
        label="detector frozen environment manifest",
    )
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("snapshot_status") != "complete"
        or environment.get("environment_files_sha256")
        != manifest.get("files_sha256")
    ):
        raise MethodologicalHalt(
            "detector equivalence environment manifest identity differs"
        )


def _verify_detector_equivalence_artifact(
    *, task_index: int, role: str
) -> dict[str, Any]:
    values = _format_values()
    path = Path(
        values["detector_equivalence_{}_{}_artifact".format(task_index, role)]
    ).resolve()
    if path.is_symlink() or not path.is_file():
        raise MethodologicalHalt(
            "detector equivalence {} task {} artifact is absent".format(
                role, task_index
            )
        )
    try:
        artifact = read_detector_equivalence_fit_artifact(path)
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector equivalence artifact failed validation: {}".format(exc)
        ) from exc
    identity = artifact.get("task_identity")
    provenance = artifact.get("provenance")
    if (
        artifact.get("role") != role
        or int(artifact.get("task_index", -1)) != task_index
        or not isinstance(identity, dict)
        or int(identity.get("ordinal", -1)) != task_index
        or identity.get("detector_name")
        != DETECTOR_BENCHMARK_TASKS[task_index]
        or not isinstance(provenance, dict)
        or provenance.get("equivalence_role") != role
        or int(provenance.get("workers", -1)) != DETECTOR_WORKERS
        or Path(str(provenance.get("execution_policy_path", ""))).resolve()
        != _detector_execution_policy_path()
        or provenance.get("execution_policy_sha256")
        != DETECTOR_EXECUTION_POLICY_SHA256
    ):
        raise MethodologicalHalt(
            "detector equivalence artifact role/task/runtime identity differs"
        )
    _verify_detector_equivalence_policy_identity(
        provenance.get("policy_identity")
    )
    if role not in {"cuda", "cuda_repeat"}:
        raise MethodologicalHalt(
            "CPU neural artifacts are prohibited by the CUDA-only gate"
        )
    if (
        provenance.get("device") != DETECTOR_DEVICE
        or provenance.get("gpu_uuid") != GPU_UUID
    ):
        raise MethodologicalHalt(
            "CUDA detector reproducibility artifact GPU identity differs"
        )
    _validate_detector_gpu_accounting(
        provenance.get("gpu_accounting"), live=False
    )
    contract = _detector_equivalence_action(
        load_command_contract(DEFAULT_COMMAND_CONTRACT),
        task_index=task_index,
        role=role,
    )
    _ensure_finalized_detector_ledger(contract)
    return artifact


def _verify_detector_equivalence_report(task_index: int) -> dict[str, Any]:
    values = _format_values()
    artifacts = {
        role: _verify_detector_equivalence_artifact(
            task_index=task_index, role=role
        )
        for role in ("cuda", "cuda_repeat")
    }
    policy_identity = artifacts["cuda"]["provenance"]["policy_identity"]
    if any(
        artifact["provenance"].get("policy_identity") != policy_identity
        for artifact in artifacts.values()
    ):
        raise MethodologicalHalt(
            "detector CUDA artifacts do not share one frozen policy identity"
        )
    report_path = Path(
        values["detector_equivalence_report_{}".format(task_index)]
    ).resolve()
    try:
        report = read_detector_cuda_reproducibility_report(
            report_path,
            expected_task_index=task_index,
            expected_policy_identity=policy_identity,
            expected_equivalence_policy=verify_detector_execution_policy()[
                "equivalence"
            ],
        )
    except RevisionDetectionError as exc:
        raise MethodologicalHalt(
            "detector task {} CUDA reproducibility report is absent, tampered, "
            "or failed: {}".format(task_index, exc)
        ) from exc
    expected_paths = {
        role: Path(
            values[
                "detector_equivalence_{}_{}_artifact".format(task_index, role)
            ]
        ).resolve()
        for role in artifacts
    }
    declarations = report.get("input_artifacts")
    if (
        not isinstance(declarations, dict)
        or set(declarations) != set(expected_paths)
        or any(
            Path(str(declarations[role].get("path", ""))).resolve()
            != expected_paths[role]
            for role in expected_paths
        )
        or report.get("decision", {}).get("reproducible") is not True
    ):
        raise MethodologicalHalt(
            "detector CUDA reproducibility paths/decision differ from the gate"
        )
    return report


def require_detector_equivalence_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    """Require both full signed task0/task1 reports before production continues."""

    reports = tuple(
        _verify_detector_equivalence_report(task_index)
        for task_index in sorted(DETECTOR_BENCHMARK_TASKS)
    )
    return reports  # type: ignore[return-value]


def _verify_detector_benchmark_checkpoint(
    value: Mapping[str, Any], *, task_index: int
) -> None:
    checkpoint_dir = Path(str(value.get("checkpoint_dir", ""))).resolve()
    last = value.get("last_completed_checkpoint")
    identity = value.get("benchmark_task_identity")
    if not isinstance(last, dict) or not isinstance(identity, dict):
        raise InterfaceHalt("detector benchmark checkpoint identity is absent")
    expected = checkpoint_dir / "fits" / "{:04d}".format(task_index) / "manifest.json"
    checkpoint_path = Path(str(last.get("path", ""))).resolve()
    if (
        checkpoint_path != expected.resolve()
        or not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or file_sha256(checkpoint_path) != last.get("sha256")
        or int(last.get("task_ordinal", -1)) != task_index
    ):
        raise InterfaceHalt("detector benchmark checkpoint reference differs")
    checkpoint = read_json(
        checkpoint_path, label="detector benchmark fit checkpoint"
    )
    unsigned = dict(checkpoint)
    claimed = unsigned.pop("manifest_sha256", None)
    children = checkpoint.get("children")
    plan_path = checkpoint_dir / "execution_plan.json"
    if (
        checkpoint.get("schema_version")
        != "rankcloak-revision-detector-fit-checkpoint-v1"
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
        or checkpoint.get("run_identity_sha256")
        != value.get("run_identity_sha256")
        or checkpoint.get("task_identity") != identity
        or checkpoint.get("task_identity_sha256")
        != canonical_json_sha256(identity)
        or checkpoint.get("detector_name") not in {None, identity.get("detector_name")}
        or int(identity.get("ordinal", -1)) != task_index
        or identity.get("detector_name") != DETECTOR_BENCHMARK_TASKS[task_index]
        or float(checkpoint.get("elapsed_seconds", -1.0))
        != float(value.get("fit_elapsed_seconds", -2.0))
        or not isinstance(children, dict)
        or set(children) != {"metric.json", "predictions.json"}
        or checkpoint.get("children_sha256") != canonical_json_sha256(children)
        or not plan_path.is_file()
        or plan_path.is_symlink()
        or checkpoint.get("plan_sha256")
        != value.get("execution_plan_sha256")
        or canonical_json_sha256(
            read_json(plan_path, label="detector benchmark execution plan")
        )
        != checkpoint.get("plan_sha256")
    ):
        raise InterfaceHalt("detector benchmark signed checkpoint identity is invalid")
    for name, declaration in children.items():
        child = checkpoint_path.parent / name
        if (
            not isinstance(declaration, dict)
            or not child.is_file()
            or child.is_symlink()
            or file_sha256(child) != declaration.get("sha256")
            or child.stat().st_size != int(declaration.get("size_bytes", -1))
        ):
            raise InterfaceHalt("detector benchmark checkpoint child differs")
        payload = read_json(child, label="detector benchmark checkpoint child")
        rows = payload.get("rows")
        if (
            payload.get("schema_version")
            != "rankcloak-revision-detector-fit-rows-v1"
            or not isinstance(payload.get("columns"), list)
            or not isinstance(rows, list)
            or len(rows) != int(declaration.get("row_count", -1))
        ):
            raise InterfaceHalt("detector benchmark checkpoint child rows differ")


def _verify_detector_benchmark(
    path: Path, *, task_index: int, action: Action
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise InterfaceHalt("detector benchmark record is absent or unsafe")
    value = read_json(path, label="detector benchmark record")
    unsigned = dict(value)
    claimed = unsigned.pop("benchmark_sha256", None)
    identity = value.get("benchmark_task_identity")
    if (
        value.get("schema_version")
        != "rankcloak-revision-detector-benchmark-v1"
        or not isinstance(claimed, str)
        or canonical_json_sha256(unsigned) != claimed
        or value.get("device") != DETECTOR_DEVICE
        or value.get("gpu_uuid") != GPU_UUID
        or int(value.get("workers", -1)) != DETECTOR_WORKERS
        or int(value.get("benchmark_task_index", -1)) != task_index
        or not isinstance(identity, dict)
        or int(identity.get("ordinal", -1)) != task_index
        or identity.get("detector_name") != DETECTOR_BENCHMARK_TASKS[task_index]
        or not isinstance(value.get("last_completed_checkpoint"), dict)
        or int(value["last_completed_checkpoint"].get("task_ordinal", -1))
        != task_index
        or Path(str(value.get("checkpoint_dir", ""))).resolve()
        != Path(_format_values()["detector_checkpoint_dir"]).resolve()
        or Path(str(value.get("status_file", ""))).resolve()
        != Path(_format_values()["detector_status_file"]).resolve()
    ):
        raise InterfaceHalt("detector benchmark record identity is invalid")
    _validate_detector_gpu_accounting(value.get("gpu_accounting"), live=False)
    status = read_detector_status(action)
    recorded_accounting = value.get("gpu_accounting")
    status_accounting = None if status is None else status.get("gpu_accounting")
    accounting_contains_record = False
    if isinstance(recorded_accounting, dict) and isinstance(status_accounting, dict):
        recorded_intervals = recorded_accounting.get("intervals")
        status_intervals = status_accounting.get("intervals")
        if (
            isinstance(recorded_intervals, list)
            and isinstance(status_intervals, list)
            and len(status_intervals) >= len(recorded_intervals)
        ):
            accounting_contains_record = True
            for recorded, current in zip(recorded_intervals, status_intervals):
                if not isinstance(recorded, dict) or not isinstance(current, dict):
                    accounting_contains_record = False
                    break
                recorded_end = _parse_aware_time(
                    recorded.get("completed_at_utc"),
                    "detector benchmark GPU interval completion",
                )
                current_end = _parse_aware_time(
                    current.get("completed_at_utc"),
                    "detector status GPU interval completion",
                )
                if (
                    any(
                        recorded.get(field) != current.get(field)
                        for field in (
                            "pid",
                            "process_start_ticks",
                            "device",
                            "gpu_uuid",
                            "started_at_utc",
                            "derivation_policy",
                        )
                    )
                    or current_end < recorded_end
                    or float(current.get("elapsed_seconds", -1.0))
                    < float(recorded.get("elapsed_seconds", 0.0))
                ):
                    accounting_contains_record = False
                    break
    if (
        status is None
        or status.get("state")
        not in {
            "stopped_at_fit_boundary",
            "supervisor_observed_process_exit",
            "resuming",
            "awaiting_fit_ceiling_gate",
            "running_fit",
            "fit_checkpointed",
            "fits_complete_awaiting_final_manifest",
            "complete",
        }
        or status.get("run_identity_sha256") != value.get("run_identity_sha256")
        or not accounting_contains_record
    ):
        raise InterfaceHalt("detector benchmark signed status differs")
    run_identity = status.get("run_identity")
    assert isinstance(run_identity, dict)
    _verify_detector_benchmark_runtime_provenance(run_identity)
    _verify_detector_benchmark_checkpoint(value, task_index=task_index)
    return value


def run_detector_benchmark(
    args: argparse.Namespace,
    projection: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> int:
    """Run one predeclared fit benchmark through the normal monitored gate."""

    task_index = int(args.detector_benchmark_task_index)
    action, spec, output = _benchmark_detector_action(contract, task_index)
    substitutions = _format_values()
    if verify_completion(spec, substitutions):
        raise MethodologicalHalt(
            "final detector products already exist; benchmark mode will not replace them"
        )
    verify_downstream_interface(spec, substitutions)
    if output.exists() or output.is_symlink():
        _verify_detector_benchmark(
            output, task_index=task_index, action=action
        )
        return 0
    retries = load_retry_counts()
    while True:
        code, detail = run_checkpointed_detector_process(
            action,
            projection=projection,
            retries=retries,
            poll_seconds=args.poll_seconds,
        )
        if code == 0:
            record = _verify_detector_benchmark(
                output, task_index=task_index, action=action
            )
            emit_event(
                "detector_benchmark_complete",
                action_id=action.action_id,
                task_index=task_index,
                detector_name=DETECTOR_BENCHMARK_TASKS[task_index],
                benchmark_sha256=record["benchmark_sha256"],
                fit_elapsed_seconds=record.get("fit_elapsed_seconds"),
            )
            return 0
        if _nonrecoverable_output(detail):
            raise MethodologicalHalt(
                "nonrecoverable detector benchmark failure: {}".format(
                    detail[-2000:]
                )
            )
        retries[action.action_id] = retries.get(action.action_id, 0) + 1
        record_recoverable_error(
            action,
            retries[action.action_id],
            "checkpointed_detector_benchmark_incomplete",
            detail,
        )
        if retries[action.action_id] > args.max_retries_per_action:
            raise OrchestratorError("detector benchmark retry ceiling exceeded")
        time.sleep(args.poll_seconds)


def _detector_equivalence_report_argv(task_index: int) -> tuple[str, ...]:
    values = _format_values()
    task_root = Path(
        values["detector_equivalence_root"]
    ) / "task_{}".format(task_index)
    return (
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_detectors.py",
        "--input",
        values["primary_detector_corpus"],
        "--preprocessing-manifest",
        values["primary_preprocessing_manifest"],
        "--output-dir",
        str(task_root / "report_work"),
        "--execution-policy",
        values["detector_execution_policy"],
        "--device",
        "cpu",
        "--workers",
        "1",
        "--equivalence-task-index",
        str(task_index),
        "--equivalence-report-output",
        values["detector_equivalence_report_{}".format(task_index)],
        "--equivalence-cuda-artifact",
        values[
            "detector_equivalence_{}_cuda_artifact".format(task_index)
        ],
        "--equivalence-cuda-repeat-artifact",
        values[
            "detector_equivalence_{}_cuda_repeat_artifact".format(task_index)
        ],
    )


def _detector_cpu_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "RANKCLOAK_DETECTOR_DEVICE": "cpu",
            "RANKCLOAK_DETECTOR_GPU_UUID": "",
            "RANKCLOAK_DETECTOR_WORKERS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def run_checkpointed_cpu_detector_process(
    action: Action,
    *,
    projection: Mapping[str, Any],
    retries: Mapping[str, int],
    poll_seconds: float,
) -> tuple[int, str]:
    """Launch or attach to one exact checkpointed CPU equivalence fit."""

    if (
        action.gpu
        or _arg_value(action.argv, "--device") != "cpu"
        or _arg_value(action.argv, "--equivalence-role") != "cpu"
    ):
        raise MethodologicalHalt(
            "CPU detector monitor received a non-CPU equivalence action"
        )
    monitor_poll = min(poll_seconds, DETECTOR_CPU_MONITOR_MAX_POLL_SECONDS)
    existing_pid: int | None = None
    existing_start_ticks: int | None = None
    progress: dict[str, Any]
    budget: dict[str, Any]

    # Do not launch through an exact canonical-progress source race. If an
    # orphan already exists, attach and keep monitoring it against the last
    # safe published accounting without charging an action retry.
    while True:
        existing_pid = _exact_detector_pid(action)
        existing_start_ticks = (
            None
            if existing_pid is None
            else _process_start_ticks(existing_pid)
        )
        if existing_pid is not None and existing_start_ticks is None:
            raise OrchestratorError(
                "exact CPU detector process identity disappeared"
            )
        try:
            progress = operational_progress()
            budget = calculate_budget(projection, progress)
            enforce_budget(budget)
            break
        except DeferrableProgressRefresh as exc:
            progress = published_progress_after_source_race()
            budget = calculate_budget(projection, progress)
            enforce_budget(budget)
            emit_progress_refresh_deferral(
                exc,
                action=action,
                retries=retries,
                active_pid=existing_pid,
                budget=budget,
                occupancy_status=(
                    "attached_checkpointed_cpu_detector"
                    if existing_pid is not None
                    else "cpu_detector_launch_deferred"
                ),
            )
            write_state(
                status=(
                    "attached_checkpointed_cpu_detector"
                    if existing_pid is not None
                    else "cpu_detector_launch_progress_refresh_deferred"
                ),
                message=(
                    "Monitoring the exact orphaned CPU reference while the "
                    "canonical progress refresh is deferred."
                    if existing_pid is not None
                    else "CPU reference launch is deferred until canonical "
                    "progress refresh succeeds."
                ),
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
                active_pid=existing_pid,
            )
            if existing_pid is not None:
                break
            time.sleep(monitor_poll)
            continue
        except BudgetHalt:
            if existing_pid is not None and existing_start_ticks is not None:
                _terminate_detector_process(
                    action, existing_pid, existing_start_ticks
                )
            raise

    # Recheck immediately before spawn. The single-supervisor lock plus this
    # exact argv check closes restart/launch races; the runner flock remains a
    # second fail-closed guard rather than the normal orphan-recovery path.
    if existing_pid is None:
        existing_pid = _exact_detector_pid(action)
        existing_start_ticks = (
            None
            if existing_pid is None
            else _process_start_ticks(existing_pid)
        )
        if existing_pid is not None and existing_start_ticks is None:
            raise OrchestratorError(
                "exact CPU detector process identity disappeared"
            )

    status: dict[str, Any] | None = None
    if existing_pid is not None:
        try:
            status = read_checkpointed_cpu_detector_status(
                action, expected_pid=existing_pid
            )
        except MethodologicalHalt:
            assert existing_start_ticks is not None
            _terminate_detector_process(
                action, existing_pid, existing_start_ticks
            )
            raise

    process: subprocess.Popen[str] | None = None
    log_path = _log_path(action)
    if existing_pid is None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                list(action.argv),
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=_detector_cpu_environment(),
            )
        pid = process.pid
        start_ticks = _process_start_ticks(pid)
        if start_ticks is None:
            raise OrchestratorError(
                "launched CPU detector process identity is unavailable"
            )
        emit_event(
            "checkpointed_cpu_detector_launched",
            action_id=action.action_id,
            pid=pid,
            process_start_ticks=start_ticks,
            argv=list(action.argv),
            log_path=str(log_path),
            resumed=True,
        )
    else:
        pid = existing_pid
        assert existing_start_ticks is not None
        start_ticks = existing_start_ticks
        emit_event(
            "checkpointed_cpu_detector_attached",
            action_id=action.action_id,
            pid=pid,
            process_start_ticks=start_ticks,
            argv=list(action.argv),
            duplicate_launch_prevented=True,
        )

    write_state(
        status=(
            "running_checkpointed_cpu_detector"
            if process is not None
            else "attached_checkpointed_cpu_detector"
        ),
        message=(
            "Monitoring one exact checkpointed CPU equivalence reference; "
            "restart will attach without launching a duplicate."
        ),
        action=action,
        retries=retries,
        progress=progress,
        budget=budget,
        active_pid=pid,
        detector_status=(
            status
            if status is not None
            and int(status.get("pid", -1)) == pid
            and int(status.get("process_start_ticks", -1)) == start_ticks
            else None
        ),
    )
    monitor_started = time.monotonic()
    last_long_update = monitor_started
    while True:
        running = (
            process.poll() is None
            if process is not None
            else _detector_pid_is_live(action, pid, start_ticks)
        )
        if not running:
            break
        time.sleep(monitor_poll)
        running = (
            process.poll() is None
            if process is not None
            else _detector_pid_is_live(action, pid, start_ticks)
        )
        if not running:
            break
        try:
            status = read_checkpointed_cpu_detector_status(
                action, expected_pid=pid
            )
        except MethodologicalHalt:
            _terminate_detector_process(action, pid, start_ticks)
            raise
        status_matches_process = bool(
            status is not None
            and int(status.get("pid", -1)) == pid
            and int(status.get("process_start_ticks", -1)) == start_ticks
        )
        status_age = (
            (
                datetime.now(timezone.utc)
                - _parse_aware_time(
                    status.get("updated_at_utc"),
                    "CPU detector status updated_at_utc",
                )
            ).total_seconds()
            if status_matches_process
            else time.monotonic() - monitor_started
        )
        if status_age > DETECTOR_STATUS_STALE_SECONDS:
            _terminate_detector_process(action, pid, start_ticks)
            return 75, (
                "checkpointed CPU detector heartbeat was absent or stale for "
                "{:.1f} seconds; exact process stopped for checkpoint resume"
            ).format(status_age)
        progress_deferral: DeferrableProgressRefresh | None = None
        try:
            progress = operational_progress()
        except DeferrableProgressRefresh as exc:
            progress = published_progress_after_source_race()
            progress_deferral = exc
        budget = calculate_budget(projection, progress)
        try:
            enforce_budget(budget)
        except BudgetHalt:
            _terminate_detector_process(action, pid, start_ticks)
            raise
        if progress_deferral is not None:
            emit_progress_refresh_deferral(
                progress_deferral,
                action=action,
                retries=retries,
                active_pid=pid,
                budget=budget,
                occupancy_status="running_checkpointed_cpu_detector",
            )
        write_state(
            status=(
                "running_checkpointed_cpu_detector"
                if process is not None
                else "attached_checkpointed_cpu_detector"
            ),
            message=(
                "Monitoring one exact checkpointed CPU equivalence reference; "
                "its signed heartbeat and manifest-last checkpoint remain fresh."
            ),
            action=action,
            retries=retries,
            progress=progress,
            budget=budget,
            active_pid=pid,
            detector_status=status if status_matches_process else None,
        )
        if time.monotonic() - last_long_update >= 6 * 3600:
            emit_event(
                "six_hour_progress",
                action_id=action.action_id,
                detector_completed_fits=(status or {}).get(
                    "completed_fit_count"
                ),
                detector_total_fits=(status or {}).get("total_fit_count"),
                detector_current_fit=(status or {}).get("current_fit"),
                detector_fits_per_hour=(status or {}).get("fits_per_hour"),
                detector_eta_seconds=(status or {}).get(
                    "rolling_eta_seconds"
                ),
                cumulative_actual_gpu_hours=budget[
                    "cumulative_actual_gpu_hours"
                ],
                execution_device="cpu",
            )
            last_long_update = time.monotonic()

    try:
        final_status = read_checkpointed_cpu_detector_status(action)
    except MethodologicalHalt:
        raise
    artifact_raw = _arg_value(action.argv, "--equivalence-artifact")
    if artifact_raw is None:
        raise MethodologicalHalt("CPU equivalence action lacks its artifact path")
    artifact = Path(artifact_raw)
    artifact = (
        artifact if artifact.is_absolute() else PROJECT_ROOT / artifact
    ).resolve()
    completed = artifact.is_file() and not artifact.is_symlink()
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except OSError:
        tail = ""
    if completed:
        return 0, tail
    if process is not None and process.returncode not in (None, 0):
        return int(process.returncode), tail
    return 1, (
        tail
        or "CPU detector exited without its final artifact; status={}".format(
            None if final_status is None else final_status.get("state")
        )
    )


def run_detector_equivalence_role(
    args: argparse.Namespace,
    projection: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    task_index: int,
    role: str,
) -> int:
    action = _detector_equivalence_action(
        contract, task_index=task_index, role=role
    )
    build_detector_cuda_budget_gate(
        stage="post_benchmark_pre_reproducibility"
    )
    artifact_path = Path(
        _format_values()[
            "detector_equivalence_{}_{}_artifact".format(task_index, role)
        ]
    )
    if artifact_path.exists() or artifact_path.is_symlink():
        _verify_detector_equivalence_artifact(
            task_index=task_index, role=role
        )
        return 0
    retries = load_retry_counts()
    while True:
        code, detail = run_checkpointed_detector_process(
            action,
            projection=projection,
            retries=retries,
            poll_seconds=args.poll_seconds,
        )
        if code == 0:
            artifact = _verify_detector_equivalence_artifact(
                task_index=task_index, role=role
            )
            emit_event(
                "detector_equivalence_artifact_complete",
                action_id=action.action_id,
                task_index=task_index,
                role=role,
                artifact_sha256=artifact["artifact_sha256"],
            )
            return 0
        if _nonrecoverable_output(detail):
            raise MethodologicalHalt(
                "nonrecoverable detector equivalence failure: {}".format(
                    detail[-2000:]
                )
            )
        retries[action.action_id] = retries.get(action.action_id, 0) + 1
        record_recoverable_error(
            action,
            retries[action.action_id],
            "checkpointed_detector_equivalence_incomplete",
            detail,
        )
        if retries[action.action_id] > args.max_retries_per_action:
            raise OrchestratorError(
                "detector equivalence retry ceiling exceeded"
            )
        time.sleep(args.poll_seconds)


def run_detector_equivalence_report(
    *, task_index: int
) -> dict[str, Any]:
    path = Path(
        _format_values()["detector_equivalence_report_{}".format(task_index)]
    )
    if path.exists() or path.is_symlink():
        return _verify_detector_equivalence_report(task_index)
    for role in ("cuda", "cuda_repeat"):
        _verify_detector_equivalence_artifact(
            task_index=task_index, role=role
        )
    argv = _detector_equivalence_report_argv(task_index)
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_detector_cpu_environment(),
    )
    action = Action(
        action_id="downstream:detector:equivalence:{}:report".format(
            task_index
        ),
        stage="neural_detector",
        kind="downstream",
        argv=argv,
        output_dir=path.parent,
        gpu=False,
    )
    atomic_write_bytes(
        _log_path(action),
        (completed.stdout + "\n" + completed.stderr).encode(
            "utf-8", "replace"
        ),
    )
    if completed.returncode != 0:
        # A signed numerical failure is the explicitly authorized
        # methodological stop, never a recoverable implementation retry.
        raise MethodologicalHalt(
            "detector device equivalence task {} did not pass: {}".format(
                task_index, (completed.stderr or completed.stdout)[-4000:]
            )
        )
    return _verify_detector_equivalence_report(task_index)


def ensure_detector_equivalence_gate(
    args: argparse.Namespace,
    projection: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Benchmark both architectures, then complete same-CUDA repeatability."""

    for task_index in sorted(DETECTOR_BENCHMARK_TASKS):
        benchmark_args = argparse.Namespace(**vars(args))
        benchmark_args.detector_benchmark_task_index = task_index
        run_detector_benchmark(benchmark_args, projection, contract)
    build_detector_cuda_budget_gate(
        stage="post_benchmark_pre_reproducibility"
    )
    for task_index in sorted(DETECTOR_BENCHMARK_TASKS):
        run_detector_equivalence_role(
            args,
            projection,
            contract,
            task_index=task_index,
            role="cuda",
        )
        run_detector_equivalence_role(
            args,
            projection,
            contract,
            task_index=task_index,
            role="cuda_repeat",
        )
        report = run_detector_equivalence_report(task_index=task_index)
        emit_event(
            "detector_cuda_reproducibility_report_passed",
            task_index=task_index,
            detector_name=DETECTOR_BENCHMARK_TASKS[task_index],
            report_sha256=report["report_sha256"],
        )
    require_detector_equivalence_gate()
    build_detector_cuda_budget_gate(
        stage="post_reproducibility_preproduction"
    )
    require_detector_cuda_budget_gate(
        expected_stage="post_reproducibility_preproduction"
    )


def run_cpu_process(action: Action, argv: Sequence[str]) -> tuple[int, str]:
    log_path = _log_path(action)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(argv),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_write_bytes(
        log_path,
        (completed.stdout + ("\n" if completed.stdout else "") + completed.stderr).encode(
            "utf-8", "replace"
        ),
    )
    return completed.returncode, (completed.stdout + "\n" + completed.stderr)[-8000:]


def evaluator_dry_run_count(action: Action) -> int:
    argv = [item for item in action.argv if item != "--resume"] + ["--dry-run"]
    completed = subprocess.run(
        argv, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise MethodologicalHalt(
            "exact evaluator dry run failed for {}: {}".format(
                action.action_id, (completed.stderr or completed.stdout)[-4000:]
            )
        )
    try:
        value = json.loads(completed.stdout)
        return int(value["evaluation_task_count"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise InterfaceHalt("evaluator dry run lacks evaluation_task_count") from exc


def _evaluator_unavailability_manifest_value(
    projection: Mapping[str, Any], scoreable_count: int
) -> dict[str, Any]:
    source_dir = (
        PROJECT_ROOT
        / RESULTS_ROOT
        / "ablation_v2"
        / "mistral_7b_instruct_v0_3_q4_k_m"
    )
    plan = read_jsonl(source_dir / "plan.jsonl")
    plan_by_id = {str(row.get("work_id")): row for row in plan}
    checkpoint = read_json(source_dir / "checkpoint.json", label="Mistral ablation checkpoint")
    completed_ids = set(map(str, checkpoint.get("completed_trial_ids", [])))
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in read_jsonl(source_dir / "records.jsonl"):
        work_id = str(record.get("work_id", ""))
        if (
            work_id not in completed_ids
            or record.get("execution_status") != "completed"
            or record.get("record_type") != "condition_unavailable"
            or record.get("reason_code")
            != "empty_isolated_roundtrip_vocabulary"
        ):
            continue
        task = plan_by_id.get(work_id)
        if not isinstance(task, dict) or task.get("work_kind") != "rankcloak":
            continue
        if work_id in seen:
            raise MethodologicalHalt(
                "duplicate upstream-unavailable completion in Mistral ablation"
            )
        seen.add(work_id)
        units.append(
            {
                "terminal_status": "upstream_dependent_unavailable_not_scored",
                "source_stage": "ablation_v2",
                "source_work_id": work_id,
                "source_record_type": record.get("record_type"),
                "source_record_sha256": canonical_json_sha256(record),
                "reason_code": record.get("reason_code"),
                "generator_model_id": "mistral_7b_instruct_v0_3_q4_k_m",
                "evaluator_model_id": "llama3_8b_instruct_q4_k_m",
                "protocol_variant": task.get("protocol_variant"),
                "payload_name": task.get("payload_name"),
                "scoring_attempted": False,
                "score_imputed": False,
            }
        )
    units.sort(key=lambda row: str(row["source_work_id"]))
    if len(units) != STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS:
        raise MethodologicalHalt(
            "expected exactly {} upstream-unavailable Mistral ablation evaluator units; found {}".format(
                STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS, len(units)
            )
        )
    files = []
    for name in ("plan.jsonl", "checkpoint.json", "records.jsonl", "run_identity.json"):
        path = source_dir / name
        if not path.is_file() or path.is_symlink():
            raise MethodologicalHalt(
                "upstream-unavailability lineage file is absent/unsafe: {}".format(path)
            )
        files.append(
            {
                "role": name.rsplit(".", 1)[0],
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    projected = next(
        int(row["target_work_units"])
        for row in projection.get("stage_totals", [])
        if isinstance(row, dict) and row.get("stage") == "heldout_evaluator"
    )
    manifest: dict[str, Any] = {
        "schema_version": EVALUATOR_UNAVAILABILITY_SCHEMA,
        "manifest_type": "heldout_evaluator_upstream_dependent_unavailability",
        "protocol_contract_revision": PROTOCOL_REVISION,
        "result_schema_revision": RESULT_REVISION,
        "authorized_projection_sha256": PROJECTION_SHA256,
        "frozen_evaluator_target_units": projected,
        "scoreable_evaluator_units": scoreable_count,
        "upstream_dependent_unavailable_units": len(units),
        "terminal_accounted_units": scoreable_count + len(units),
        "scoring_attempted_for_unavailable_units": False,
        "scores_imputed_or_fabricated": False,
        "analysis_policy": (
            "terminal_design_units_excluded_from_quality_estimands_and_not_scored"
        ),
        "source_files": files,
        "source_files_sha256": canonical_json_sha256(files),
        "units": units,
        "units_sha256": canonical_json_sha256(units),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def verify_or_write_evaluator_unavailability_manifest(
    projection: Mapping[str, Any], scoreable_count: int
) -> dict[str, Any]:
    expected = _evaluator_unavailability_manifest_value(projection, scoreable_count)
    path = PROJECT_ROOT / EVALUATOR_UNAVAILABILITY_PATH
    if path.exists():
        observed = read_json(path, label="evaluator unavailability accounting manifest")
        unsigned = dict(observed)
        claimed = unsigned.pop("manifest_sha256", None)
        if claimed != canonical_json_sha256(unsigned) or observed != expected:
            raise MethodologicalHalt(
                "evaluator unavailability accounting manifest differs from upstream lineage"
            )
    else:
        atomic_write_json(path, expected)
    return expected


def verify_evaluator_unavailability_manifest() -> dict[str, Any]:
    path = PROJECT_ROOT / EVALUATOR_UNAVAILABILITY_PATH
    manifest = read_json(path, label="evaluator unavailability accounting manifest")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise MethodologicalHalt("evaluator unavailability manifest self-hash mismatch")
    if (
        manifest.get("schema_version") != EVALUATOR_UNAVAILABILITY_SCHEMA
        or manifest.get("manifest_type")
        != "heldout_evaluator_upstream_dependent_unavailability"
        or manifest.get("protocol_contract_revision") != PROTOCOL_REVISION
        or manifest.get("result_schema_revision") != RESULT_REVISION
        or manifest.get("authorized_projection_sha256") != PROJECTION_SHA256
        or manifest.get("frozen_evaluator_target_units")
        != FROZEN_EVALUATOR_TARGET_UNITS
        or manifest.get("scoreable_evaluator_units")
        != SCOREABLE_EVALUATOR_UNITS
        or manifest.get("upstream_dependent_unavailable_units")
        != STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS
        or manifest.get("terminal_accounted_units")
        != SCOREABLE_EVALUATOR_UNITS
        + STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS
        or manifest.get("scoring_attempted_for_unavailable_units") is not False
        or manifest.get("scores_imputed_or_fabricated") is not False
        or manifest.get("analysis_policy")
        != "terminal_design_units_excluded_from_quality_estimands_and_not_scored"
    ):
        raise MethodologicalHalt("evaluator unavailability manifest contract mismatch")
    files = manifest.get("source_files")
    units = manifest.get("units")
    if (
        not isinstance(files, list)
        or len(files) != 4
        or manifest.get("source_files_sha256") != canonical_json_sha256(files)
        or not isinstance(units, list)
        or len(units) != STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS
        or manifest.get("units_sha256") != canonical_json_sha256(units)
    ):
        raise MethodologicalHalt("evaluator unavailability lineage hash mismatch")
    source_ids = [str(unit.get("source_work_id", "")) for unit in units]
    if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
        raise MethodologicalHalt("evaluator unavailability unit identities are not exact")
    for unit in units:
        if (
            unit.get("terminal_status")
            != "upstream_dependent_unavailable_not_scored"
            or unit.get("source_stage") != "ablation_v2"
            or unit.get("source_record_type") != "condition_unavailable"
            or unit.get("reason_code")
            != "empty_isolated_roundtrip_vocabulary"
            or unit.get("generator_model_id")
            != "mistral_7b_instruct_v0_3_q4_k_m"
            or unit.get("evaluator_model_id") != "llama3_8b_instruct_q4_k_m"
            or unit.get("scoring_attempted") is not False
            or unit.get("score_imputed") is not False
            or not unit.get("source_record_sha256")
        ):
            raise MethodologicalHalt(
                "evaluator unavailability unit violates terminal semantics"
            )
    try:
        _verify_declared_files(path, files)
    except InterfaceHalt as exc:
        raise MethodologicalHalt(
            "evaluator unavailability source identity mismatch: {}".format(exc)
        ) from exc
    expected = _evaluator_unavailability_manifest_value(
        {
            "stage_totals": [
                {
                    "stage": "heldout_evaluator",
                    "target_work_units": FROZEN_EVALUATOR_TARGET_UNITS,
                }
            ]
        },
        SCOREABLE_EVALUATOR_UNITS,
    )
    if manifest != expected:
        raise MethodologicalHalt(
            "evaluator unavailability manifest differs from exact source records"
        )
    return manifest


def evaluator_projection_gate(actions: Sequence[Action], projection: Mapping[str, Any]) -> None:
    evaluator_actions = [action for action in actions if action.kind == "evaluator"]
    observed: dict[str, int] = {}
    for action in evaluator_actions:
        count = evaluator_dry_run_count(action)
        observed[action.action_id] = count
        if count != action.expected_count:
            raise MethodologicalHalt(
                "exact evaluator plan count {} differs from the frozen expected {} for {}. "
                "Structurally unavailable source records cannot be scored or synthesized.".format(
                    count, action.expected_count, action.action_id
                )
            )
    projected = next(
        int(row["target_work_units"])
        for row in projection.get("stage_totals", [])
        if isinstance(row, dict) and row.get("stage") == "heldout_evaluator"
    )
    scoreable = sum(observed.values())
    if scoreable + STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS != projected:
        raise MethodologicalHalt(
            "exact evaluator DAG has {} scoreable plus {} upstream-unavailable units, but the authorized target is {}".format(
                scoreable, STRUCTURALLY_UNAVAILABLE_EVALUATOR_UNITS, projected
            )
        )
    verify_or_write_evaluator_unavailability_manifest(projection, scoreable)


def _completion_for_gpu_action(action: Action) -> bool:
    if action.kind == "evaluator_export":
        return shard_status(action) == "complete" and export_marker_valid(action)
    return shard_status(action) == "complete"


def _argv_for_gpu_action(action: Action) -> tuple[str, ...]:
    if action.kind == "evaluator_export":
        return action.argv
    status = shard_status(action)
    return action.argv + (("--resume",) if status == "incomplete" else ())


def _format_values() -> dict[str, str]:
    analysis_root = PROJECT_ROOT / RESULTS_ROOT / "analysis_inputs"
    evaluator_root = PROJECT_ROOT / RESULTS_ROOT / "heldout_evaluator"
    values: dict[str, str] = {
        "project_root": str(PROJECT_ROOT),
        "results_root": str(PROJECT_ROOT / RESULTS_ROOT),
        "primary_input_dir": str(analysis_root / "primary_v2"),
        "ablation_input_dir": str(analysis_root / "ablation_v2"),
        "multilingual_input_dir": str(analysis_root / "multilingual_v2"),
        "robustness_input_dir": str(analysis_root / "robustness_v2"),
        "primary_trials": str(analysis_root / "primary_v2" / "trials.csv"),
        "primary_features": str(analysis_root / "primary_v2" / "features.csv"),
        "primary_runtime": str(analysis_root / "primary_v2" / "runtime.csv"),
        "primary_detector_corpus": str(analysis_root / "primary_v2" / "detector_corpus.jsonl"),
        "primary_preprocessing_manifest": str(analysis_root / "primary_v2" / "preprocessing_output_manifest.json"),
        "ablation_trials": str(analysis_root / "ablation_v2" / "trials.csv"),
        "ablation_features": str(analysis_root / "ablation_v2" / "features.csv"),
        "ablation_runtime": str(analysis_root / "ablation_v2" / "runtime.csv"),
        "ablation_preprocessing_manifest": str(analysis_root / "ablation_v2" / "preprocessing_output_manifest.json"),
        "multilingual_trials": str(analysis_root / "multilingual_v2" / "trials.csv"),
        "multilingual_features": str(analysis_root / "multilingual_v2" / "features.csv"),
        "multilingual_runtime": str(analysis_root / "multilingual_v2" / "runtime.csv"),
        "multilingual_preprocessing_manifest": str(analysis_root / "multilingual_v2" / "preprocessing_output_manifest.json"),
        "robustness_trials": str(analysis_root / "robustness_v2" / "trials.csv"),
        "robustness_features": str(analysis_root / "robustness_v2" / "features.csv"),
        "robustness_runtime": str(analysis_root / "robustness_v2" / "runtime.csv"),
        "robustness_preprocessing_manifest": str(analysis_root / "robustness_v2" / "preprocessing_output_manifest.json"),
        "evaluator_join_dir": str(PROJECT_ROOT / RESULTS_ROOT / "analysis_inputs" / "primary_heldout_join_v2"),
        "evaluator_join_features": str(PROJECT_ROOT / RESULTS_ROOT / "analysis_inputs" / "primary_heldout_join_v2" / "primary_features_with_heldout_evaluator.csv"),
        "evaluator_join_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "analysis_inputs" / "primary_heldout_join_v2" / "heldout_feature_join_manifest.json"),
        "detector_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "neural_detector" / "confirmatory_v2"),
        "detector_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "neural_detector" / "confirmatory_v2" / "detector_run_manifest.json"),
        "detector_metrics": str(PROJECT_ROOT / RESULTS_ROOT / "neural_detector" / "confirmatory_v2" / "detector_metrics.csv"),
        "detector_predictions": str(PROJECT_ROOT / RESULTS_ROOT / "neural_detector" / "confirmatory_v2" / "detector_predictions.csv"),
        "detector_checkpoint_dir": str(PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "production_run_v2.checkpoints"),
        "detector_status_file": str(PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "production_run_v2.status.json"),
        "detector_fit_permit_file": str(PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "production_run_v2.fit_permit.json"),
        "detector_fit_permit_receipt_dir": str(PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "production_run_v2.checkpoints" / "fit_permit_receipts"),
        "detector_execution_policy": str(
            PROJECT_ROOT / DETECTOR_EXECUTION_POLICY_RELATIVE
        ),
        "detector_cuda_budget_gate": str(
            PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "cuda_budget_gate.json"
        ),
        "detector_benchmark_output_0": str(
            PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "benchmarks" / "task_0_cuda.json"
        ),
        "detector_benchmark_output_1": str(
            PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "benchmarks" / "task_1_cuda.json"
        ),
        "detector_equivalence_root": str(
            PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT
        ),
        "detector_equivalence_report_0": str(
            PROJECT_ROOT
            / DETECTOR_EQUIVALENCE_ROOT
            / "task_0"
            / "cuda_reproducibility_report.json"
        ),
        "detector_equivalence_report_1": str(
            PROJECT_ROOT
            / DETECTOR_EQUIVALENCE_ROOT
            / "task_1"
            / "cuda_reproducibility_report.json"
        ),
        "detector_gpu_ledger": str(
            PROJECT_ROOT / DETECTOR_EQUIVALENCE_ROOT / "gpu_accounting_ledger.json"
        ),
        "gpu_uuid": GPU_UUID,
        "statistics_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "analysis" / "statistics_v2"),
        "statistics_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "analysis" / "statistics_v2" / "statistics_run_manifest.json"),
        "mixed_models_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "analysis" / "mixed_models_v2"),
        "mixed_models_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "analysis" / "mixed_models_v2" / "mixed_model_run_manifest.json"),
        "theory_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "theory" / "confirmatory_v2"),
        "theory_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "theory" / "confirmatory_v2" / "theory_validation_manifest.json"),
        "report_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "reports" / "confirmatory_v2"),
        "report_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "reports" / "confirmatory_v2" / "report_output_manifest.json"),
        "figures_output_dir": str(PROJECT_ROOT / RESULTS_ROOT / "reports" / "confirmatory_v2_figures"),
        "figures_manifest": str(PROJECT_ROOT / RESULTS_ROOT / "reports" / "confirmatory_v2_figures" / "figure_render_manifest.json"),
        "evaluator_unavailability_manifest": str(
            PROJECT_ROOT / EVALUATOR_UNAVAILABILITY_PATH
        ),
        "progress_manifest": str(PROJECT_ROOT / FINAL_PROGRESS_PATH),
    }
    for task_index in DETECTOR_BENCHMARK_TASKS:
        task_root = (
            PROJECT_ROOT
            / DETECTOR_EQUIVALENCE_ROOT
            / "task_{}".format(task_index)
        )
        for role in ("cuda", "cuda_repeat"):
            prefix = "detector_equivalence_{}_{}".format(task_index, role)
            values[prefix + "_artifact"] = str(
                task_root / (role + "_artifact.json")
            )
            if role == "cuda":
                values[prefix + "_output_dir"] = values["detector_output_dir"]
                values[prefix + "_checkpoint_dir"] = values[
                    "detector_checkpoint_dir"
                ]
                values[prefix + "_status_file"] = values["detector_status_file"]
                values[prefix + "_fit_permit_file"] = values[
                    "detector_fit_permit_file"
                ]
                values[prefix + "_fit_permit_receipt_dir"] = values[
                    "detector_fit_permit_receipt_dir"
                ]
            else:
                role_root = task_root / (role + "_run")
                values[prefix + "_output_dir"] = str(role_root)
                values[prefix + "_checkpoint_dir"] = str(
                    role_root.with_name(role_root.name + ".checkpoints")
                )
                values[prefix + "_status_file"] = str(
                    role_root.with_name(role_root.name + ".status.json")
                )
                values[prefix + "_fit_permit_file"] = str(
                    role_root.with_name(role_root.name + ".fit_permit.json")
                )
                values[prefix + "_fit_permit_receipt_dir"] = str(
                    role_root.with_name(role_root.name + ".checkpoints")
                    / "fit_permit_receipts"
                )
    for source_stage in EVALUATOR_SOURCE_STAGES:
        for generator, evaluator in EVALUATOR_BY_GENERATOR.items():
            prefix = "evaluator_{}_{}".format(source_stage, generator)
            shard = evaluator_root / source_stage / evaluator
            values[prefix + "_feature_manifest"] = str(shard / "features_manifest.json")
            values[prefix + "_features"] = str(shard / "features.jsonl")
            values[prefix + "_continuous"] = str(shard / "continuous_quality.jsonl")
    return values


def load_command_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path, label="downstream command contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise InterfaceHalt("unsupported downstream command-contract schema")
    operations = contract.get("operations")
    if not isinstance(operations, list):
        raise InterfaceHalt("downstream contract operations must be a list")
    identifiers = [str(item.get("operation_id")) for item in operations if isinstance(item, dict)]
    required_order = [
        "primary_evaluator_join",
        "detector",
        "statistics",
        "mixed_models_r",
        "theory",
        "reports",
        "figures",
    ]
    if identifiers != required_order:
        raise InterfaceHalt(
            "downstream command contract is not the exact dependency order"
        )
    detector = operations[1]
    expected_execution = {
        "kind": "checkpointed_detector_gpu_v1",
        "device": DETECTOR_DEVICE,
        "gpu_uuid": "{gpu_uuid}",
        "workers": DETECTOR_WORKERS,
        "total_fits": DETECTOR_TOTAL_FITS,
        "checkpoint_dir": "{detector_checkpoint_dir}",
        "status_file": "{detector_status_file}",
        "fit_permit_file": "{detector_fit_permit_file}",
        "fit_permit_receipt_dir": "{detector_fit_permit_receipt_dir}",
        "required_equivalence_reports": [
            "{detector_equivalence_report_0}",
            "{detector_equivalence_report_1}",
        ],
        "gpu_accounting_ledger": "{detector_gpu_ledger}",
        "execution_policy": "{detector_execution_policy}",
        "cuda_budget_gate": "{detector_cuda_budget_gate}",
        "execution_policy_sha256": DETECTOR_EXECUTION_POLICY_SHA256,
        "execution_policy_content_sha256": (
            DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        ),
        "next_fit_upper_seconds_by_detector": (
            DETECTOR_NEXT_FIT_UPPER_SECONDS
        ),
        "ceiling_gate": "signed_single_use_per_fit_v1",
    }
    if (
        not isinstance(detector, dict)
        or detector.get("gpu") is not True
        or detector.get("execution") != expected_execution
    ):
        raise InterfaceHalt(
            "detector operation lacks the exact checkpointed CUDA execution contract"
        )
    verify_detector_execution_policy()
    return contract


def _expand_argv(values: Sequence[Any], substitutions: Mapping[str, str]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise InterfaceHalt("command argv entries must be strings")
        try:
            result.append(raw.format_map(substitutions))
        except KeyError as exc:
            raise InterfaceHalt("unknown command placeholder: {}".format(exc)) from exc
    if not result:
        raise InterfaceHalt("downstream command argv is empty")
    return tuple(result)


def downstream_actions(contract: Mapping[str, Any]) -> list[tuple[Action, dict[str, Any]]]:
    substitutions = _format_values()
    result: list[tuple[Action, dict[str, Any]]] = []
    for raw in contract["operations"]:
        if not isinstance(raw, dict):
            raise InterfaceHalt("downstream operation must be an object")
        operation_id = str(raw["operation_id"])
        argv = _expand_argv(raw.get("argv", []), substitutions)
        completion = raw.get("completion")
        interface = raw.get("interface")
        if not isinstance(completion, dict) or not isinstance(interface, dict):
            raise InterfaceHalt("{} lacks interface/completion contracts".format(operation_id))
        result.append(
            (
                Action(
                    action_id="downstream:" + operation_id,
                    stage=str(raw.get("stage", operation_id)),
                    kind="downstream",
                    argv=argv,
                    output_dir=(
                        Path(str(raw["output_dir"].format_map(substitutions)))
                        if raw.get("output_dir")
                        else None
                    ),
                    gpu=bool(raw.get("gpu", False)),
                ),
                dict(raw),
            )
        )
    return result


def verify_downstream_interface(spec: Mapping[str, Any], substitutions: Mapping[str, str]) -> None:
    interface = spec["interface"]
    path = Path(str(interface.get("path", "")).format_map(substitutions))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file() or path.is_symlink():
        raise InterfaceHalt("downstream entrypoint is absent or unsafe: {}".format(path))
    source_tokens = interface.get("required_source_tokens", [])
    if source_tokens:
        text = path.read_text(encoding="utf-8")
        missing = [str(token) for token in source_tokens if str(token) not in text]
        if missing:
            raise InterfaceHalt(
                "downstream source {} lacks required interface tokens: {}".format(
                    path, ", ".join(missing)
                )
            )
    probe = interface.get("probe_argv")
    if probe:
        argv = _expand_argv(probe, substitutions)
        completed = subprocess.run(
            argv, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        help_text = completed.stdout + "\n" + completed.stderr
        required = [str(item) for item in interface.get("required_help_tokens", [])]
        if completed.returncode != 0 or any(token not in help_text for token in required):
            raise InterfaceHalt("downstream CLI probe failed for {}".format(path))


def _verify_declared_files(
    manifest_path: Path,
    declarations: Iterable[Mapping[str, Any]],
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
    size_keys: Sequence[str] = ("size_bytes", "bytes"),
) -> None:
    for declaration in declarations:
        raw = Path(str(declaration.get(path_key, "")))
        candidate = raw if raw.is_absolute() else manifest_path.parent / raw
        if not candidate.is_file() or candidate.is_symlink():
            raise InterfaceHalt("declared output is missing/unsafe: {}".format(candidate))
        if file_sha256(candidate) != declaration.get(hash_key):
            raise InterfaceHalt("declared output hash mismatch: {}".format(candidate))
        expected_size = next(
            (declaration[key] for key in size_keys if declaration.get(key) is not None),
            None,
        )
        if expected_size is not None and candidate.stat().st_size != int(expected_size):
            raise InterfaceHalt("declared output size mismatch: {}".format(candidate))


def _resolved_declared_path(manifest_path: Path, raw_value: Any) -> Path:
    raw = Path(str(raw_value))
    if raw.is_absolute():
        return raw.resolve()
    candidates = [manifest_path.parent / raw, PROJECT_ROOT / raw]
    existing = [
        candidate.resolve()
        for candidate in candidates
        if candidate.is_file() and not candidate.is_symlink()
    ]
    unique = list(dict.fromkeys(existing))
    if len(unique) != 1:
        raise InterfaceHalt(
            "declared input path is missing or ambiguous: {}".format(raw)
        )
    return unique[0]


def verify_detector_final_publication(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> None:
    """Require the exact ledger/receipt/marker/status/report completion seal."""

    ledger_path = _detector_gpu_ledger_path()
    marker_path = detector_gpu_ledger_incorporation_path(ledger_path)
    try:
        ledger = read_detector_gpu_accounting_ledger(ledger_path)
        marker = read_detector_gpu_ledger_incorporation_marker(marker_path)
    except RevisionDetectionError as exc:
        raise InterfaceHalt(
            "detector final GPU ledger incorporation is invalid: {}".format(exc)
        ) from exc
    expected_ledger_identity = {
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
    terminal_identity = status.get("terminal_receipt")
    marker_identity = status.get("gpu_ledger_incorporation")
    if (
        manifest.get("pre_final_gpu_accounting_ledger")
        != expected_ledger_identity
        or not isinstance(terminal_identity, dict)
        or not isinstance(marker_identity, dict)
        or marker_identity
        != {
            "path": str(marker_path.resolve()),
            "sha256": file_sha256(marker_path),
            "size_bytes": int(marker_path.stat().st_size),
            "incorporation_sha256": marker["incorporation_sha256"],
        }
        or marker.get("ledger", {}).get("ledger_sha256")
        != ledger["ledger_sha256"]
        or marker.get("ledger", {}).get("intervals_sha256")
        != ledger["intervals_sha256"]
        or marker.get("incorporated") is not True
        or marker.get("incorporated_ledger_interval_count")
        != len(ledger["intervals"])
        or marker.get("incorporated_ledger_intervals_sha256")
        != ledger["intervals_sha256"]
        or marker.get("final_gpu_accounting_sha256")
        != canonical_json_sha256(manifest.get("gpu_accounting"))
        or marker.get("final_published_manifest")
        != {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "size_bytes": int(manifest_path.stat().st_size),
        }
        or marker.get("final_terminal_receipt") != terminal_identity
    ):
        raise InterfaceHalt(
            "detector final ledger/terminal receipt/incorporation seal differs"
        )
    final_accounting = manifest.get("gpu_accounting")
    if not isinstance(final_accounting, dict) or not isinstance(
        final_accounting.get("intervals"), list
    ):
        raise InterfaceHalt("detector final GPU accounting is malformed")
    final_hashes = {
        canonical_json_sha256(interval)
        for interval in final_accounting["intervals"]
    }
    ledger_hashes = {
        canonical_json_sha256(interval) for interval in ledger["intervals"]
    }
    if not ledger_hashes.issubset(final_hashes):
        raise InterfaceHalt(
            "detector final accounting omits a pre-final ledger interval"
        )
    reports = manifest.get("required_equivalence_reports")
    run_identity = manifest.get("run_identity")
    lineage = None if not isinstance(run_identity, dict) else run_identity.get(
        "lineage"
    )
    excluded_operational_fields = (
        None
        if not isinstance(run_identity, dict)
        else run_identity.get("excluded_operational_gate_fields")
    )
    expected_excluded_operational_fields = {
        "cuda_budget_gate",
        "pre_final_gpu_accounting_ledger",
        "pre_final_gpu_accounting_ledger_path",
        "required_equivalence_reports",
    }
    expected_report_paths = [
        Path(
            _format_values()["detector_equivalence_report_{}".format(index)]
        ).resolve()
        for index in sorted(DETECTOR_BENCHMARK_TASKS)
    ]
    if (
        not isinstance(reports, list)
        or len(reports) != 2
        or not isinstance(lineage, dict)
        or not isinstance(excluded_operational_fields, list)
        or set(excluded_operational_fields)
        != expected_excluded_operational_fields
        or any(field in lineage for field in expected_excluded_operational_fields)
        or Path(
            str(manifest.get("pre_final_gpu_accounting_ledger_path", ""))
        ).resolve()
        != ledger_path.resolve()
    ):
        raise InterfaceHalt(
            "detector final manifest does not bind two equivalence reports"
        )
    verified_reports = require_detector_equivalence_gate()
    for index, (declaration, verified, expected_path) in enumerate(
        zip(reports, verified_reports, expected_report_paths)
    ):
        if (
            not isinstance(declaration, dict)
            or declaration
            != {
                "path": str(expected_path),
                "sha256": file_sha256(expected_path),
                "size_bytes": int(expected_path.stat().st_size),
                "report_sha256": verified["report_sha256"],
            }
        ):
            raise InterfaceHalt(
                "detector final task {} equivalence report identity differs".format(
                    index
                )
            )


def detector_manifest_awaits_supervisor_finalization(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> bool:
    """Recognize only an exact recoverable crash between final publication steps."""

    if status.get("state") != "supervisor_observed_process_exit":
        return False
    candidate_identity = manifest.get("finalization_candidate")
    declared = status.get("finalization_candidate")
    if (
        not isinstance(candidate_identity, dict)
        or set(candidate_identity)
        != {"path", "sha256", "size_bytes", "candidate_sha256"}
        or not isinstance(declared, dict)
        or set(declared)
        != {"path", "sha256", "size_bytes", "candidate_sha256", "kind"}
        or declared.get("kind") != "detector_run_manifest"
        or {key: declared.get(key) for key in candidate_identity}
        != candidate_identity
        or manifest.get("run_identity_sha256")
        != status.get("run_identity_sha256")
        or manifest.get("terminal_accounting_status_sha256")
        != status.get("status_sha256")
    ):
        raise InterfaceHalt(
            "pending detector finalization candidate/status identity differs"
        )
    candidate_path = Path(str(candidate_identity["path"])).resolve()
    if (
        candidate_path.is_symlink()
        or not candidate_path.is_file()
        or file_sha256(candidate_path) != candidate_identity.get("sha256")
        or candidate_path.stat().st_size
        != int(candidate_identity.get("size_bytes", -1))
    ):
        raise InterfaceHalt(
            "pending detector finalization candidate bytes differ"
        )
    try:
        candidate = read_detector_finalization_candidate(candidate_path)
    except RevisionDetectionError as exc:
        raise InterfaceHalt(
            "pending detector finalization candidate is invalid: {}".format(exc)
        ) from exc
    if (
        candidate.get("candidate_sha256")
        != candidate_identity.get("candidate_sha256")
        or candidate.get("kind") != "detector_run_manifest"
        or candidate.get("run_identity_sha256")
        != manifest.get("run_identity_sha256")
        or Path(str(candidate.get("requested_output_path", ""))).resolve()
        != manifest_path.resolve()
    ):
        raise InterfaceHalt(
            "pending detector finalization scientific/output identity differs"
        )
    return True


def verify_completion(spec: Mapping[str, Any], substitutions: Mapping[str, str]) -> bool:
    completion = spec["completion"]
    kind = str(completion.get("kind"))
    path = Path(str(completion.get("path", "")).format_map(substitutions))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return False
    manifest = read_json(path, label="{} completion manifest".format(kind))
    if kind == "preprocess_v2":
        if manifest.get("schema_version") != "2.0" or manifest.get("manifest_type") != "revision_preprocessing_outputs":
            raise InterfaceHalt("invalid preprocessing output manifest")
        declarations = manifest.get("outputs")
        if not isinstance(declarations, list):
            raise InterfaceHalt("preprocessing outputs declaration is malformed")
        _verify_declared_files(path, declarations, size_keys=("size_bytes",))
        input_manifest = next(
            (row for row in declarations if row.get("role") == "input_manifest"), None
        )
        if not isinstance(input_manifest, dict):
            raise InterfaceHalt("preprocessing manifest lacks input lineage")
        input_path = path.parent / str(input_manifest["path"])
        inputs = read_json(input_path)
        expected_stage = completion.get("stage")
        run_shards = inputs.get("run_shards")
        input_files = inputs.get("input_files")
        if (
            inputs.get("schema_version") != "2.0"
            or inputs.get("manifest_type") != "revision_preprocessing_inputs"
            or not isinstance(run_shards, list)
            or not isinstance(input_files, list)
            or inputs.get("input_files_sha256")
            != canonical_json_sha256(input_files)
        ):
            raise InterfaceHalt("preprocessing input manifest lineage is malformed")
        _verify_declared_files(input_path, input_files)
        emitted = [row for row in run_shards if row.get("role") == "input"]
        references = [
            row for row in run_shards if row.get("role") == "reference"
        ]
        expected_reference_stages = {
            "primary_v2": set(),
            "ablation_v2": {"primary_v2"},
            "multilingual_v2": set(),
            "robustness_v2": {"primary_v2", "ablation_v2"},
        }[str(expected_stage)]
        if (
            inputs.get("strict_complete") is not True
            or len(emitted) != 3
            or {row.get("stage") for row in emitted} != {expected_stage}
            or {row.get("model_id") for row in emitted} != set(MODELS)
            or len(references) != 3 * len(expected_reference_stages)
            or {row.get("stage") for row in references}
            != expected_reference_stages
            or (
                expected_reference_stages
                and {
                    (row.get("stage"), row.get("model_id")) for row in references
                }
                != {
                    (stage, model)
                    for stage in expected_reference_stages
                    for model in MODELS
                }
            )
            or {
                (
                    row.get("role"),
                    row.get("stage"),
                    row.get("model_id"),
                    str(Path(str(row.get("path"))).resolve()),
                )
                for row in run_shards
            }
            != {
                (
                    role,
                    stage,
                    model,
                    str((PROJECT_ROOT / RESULTS_ROOT / stage / model).resolve()),
                )
                for role, stages in (
                    ("input", {str(expected_stage)}),
                    ("reference", expected_reference_stages),
                )
                for stage in stages
                for model in MODELS
            }
        ):
            raise InterfaceHalt("preprocessing output is not strict/stage-specific")
        return True
    if kind == "evaluator_join_v1":
        if manifest.get("schema_version") != "rankcloak-revision-heldout-feature-join-v1" or manifest.get("manifest_type") != "rankcloak_revision_primary_heldout_feature_join":
            raise InterfaceHalt("invalid held-out evaluator join manifest")
        outputs = manifest.get("outputs", {})
        if not isinstance(outputs, dict) or not isinstance(outputs.get("features"), dict):
            raise InterfaceHalt("held-out join lacks its feature identity")
        if (
            manifest.get("analysis_unit")
            != "primary_payload_trial_with_nested_segment_rows"
            or manifest.get("input_scope")
            != "primary_v2_rankcloak_full_message_only"
            or int(manifest.get("primary_trial_count", -1))
            != PRIMARY_EVALUATOR_JOIN_TRIALS
            or int(manifest.get("primary_full_message_feature_rows", -1))
            != PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
            or int(manifest.get("evaluator_score_rows_joined", -1))
            != PRIMARY_EVALUATOR_JOIN_TRIALS
            or int(manifest.get("unmatched_primary_trials", -1)) != 0
            or int(manifest.get("duplicate_evaluator_trial_ids", -1)) != 0
            or manifest.get("source_record_hashes_recomputed") is not True
            or manifest.get(
                "evaluator_source_records_byte_identical_to_preprocessing"
            )
            is not True
            or manifest.get("evaluator_artifact_pins_verified") is not True
            or manifest.get("segments_as_independent_observations") is not False
            or manifest.get("score_scope")
            != "source_full_message_replicated_across_nested_segment_rows_v1"
            or manifest.get("protocol_contract_revision") != PROTOCOL_REVISION
            or manifest.get("result_schema_revision") != RESULT_REVISION
            or int(outputs["features"].get("row_count", -1))
            != PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
        ):
            raise InterfaceHalt("held-out evaluator join is not the exact primary table")
        feature_declaration = outputs["features"]
        _verify_declared_files(path, [feature_declaration])
        feature_path = _resolved_declared_path(path, feature_declaration.get("path"))
        import csv

        physical_rows = 0
        trial_ids: set[str] = set()
        with feature_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "trial_id" not in reader.fieldnames:
                raise InterfaceHalt("held-out evaluator join lacks trial_id")
            for row in reader:
                trial_id = str(row.get("trial_id", "")).strip()
                if not trial_id:
                    raise InterfaceHalt("held-out evaluator join has an empty trial_id")
                physical_rows += 1
                trial_ids.add(trial_id)
        if (
            physical_rows != PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
            or len(trial_ids) != PRIMARY_EVALUATOR_JOIN_TRIALS
        ):
            raise InterfaceHalt(
                "held-out evaluator join CSV is not the exact nested primary table"
            )
        return True
    if kind == "detector_v2":
        if (
            manifest.get("schema_version") != "rankcloak-revision-detector-run-v2"
            or manifest.get("execution_mode") != "confirmatory"
            or manifest.get("smoke") is not False
            or manifest.get("confirmatory_complete") is not True
            or int(manifest.get("split_count", -1)) != 28
            or int(manifest.get("skipped_split_count", -1)) != 0
            or int(manifest.get("failure_count", -1)) != 0
            or int(manifest.get("metric_rows", -1)) != 56
            or int(manifest.get("smoke_fallback_metric_rows", -1)) != 0
            or manifest.get("device") != DETECTOR_DEVICE
            or manifest.get("gpu_uuid") != GPU_UUID
            or int(manifest.get("workers", -1)) != DETECTOR_WORKERS
            or int(manifest.get("completed_fit_count", -1))
            != DETECTOR_TOTAL_FITS
            or int(manifest.get("total_fit_count", -1)) != DETECTOR_TOTAL_FITS
            or int(manifest.get("resumed_fit_count", -1)) < 0
            or int(manifest.get("resumed_fit_count", -1)) > DETECTOR_TOTAL_FITS
            or not isinstance(manifest.get("recovered_errors"), list)
        ):
            raise InterfaceHalt(
                "detector manifest is not the exact complete 28-split/56-fit result"
            )
        outputs = manifest.get("output_files")
        expected_outputs = {
            "detector_metrics.csv",
            "detector_predictions.csv",
            "detector_dataset_manifest.csv",
            "detector_split_manifest.json",
            "detector_failures.json",
        }
        if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
            raise InterfaceHalt("detector manifest lacks output identities")
        expected_input = Path(substitutions["primary_detector_corpus"]).resolve()
        expected_preprocessing = Path(
            substitutions["primary_preprocessing_manifest"]
        ).resolve()
        expected_plan = (
            PROJECT_ROOT
            / "analysis/revision_v1/detector_confirmatory_plan.json"
        ).resolve()
        expected_checkpoint_dir = Path(
            _format_values()["detector_checkpoint_dir"]
        ).resolve()
        expected_status_file = Path(
            _format_values()["detector_status_file"]
        ).resolve()
        expected_permit_file = Path(
            _format_values()["detector_fit_permit_file"]
        ).resolve()
        expected_receipt_dir = Path(
            _format_values()["detector_fit_permit_receipt_dir"]
        ).resolve()
        expected_policy = Path(
            _format_values()["detector_execution_policy"]
        ).resolve()
        input_path = _resolved_declared_path(path, manifest.get("input_path"))
        preprocessing_path = _resolved_declared_path(
            path, manifest.get("preprocessing_manifest_path")
        )
        plan_path = _resolved_declared_path(
            path, manifest.get("confirmatory_plan_path")
        )
        if (
            input_path != expected_input
            or manifest.get("input_sha256") != file_sha256(input_path)
            or preprocessing_path != expected_preprocessing
            or manifest.get("preprocessing_manifest_sha256")
            != file_sha256(preprocessing_path)
            or plan_path != expected_plan
            or manifest.get("confirmatory_plan_sha256") != file_sha256(plan_path)
            or manifest.get("confirmatory_plan_schema_version")
            != "rankcloak-revision-detector-confirmatory-plan-v1"
            or Path(str(manifest.get("checkpoint_dir", ""))).resolve()
            != expected_checkpoint_dir
            or Path(str(manifest.get("status_file", ""))).resolve()
            != expected_status_file
            or Path(str(manifest.get("fit_permit_file", ""))).resolve()
            != expected_permit_file
            or Path(str(manifest.get("fit_permit_receipt_dir", ""))).resolve()
            != expected_receipt_dir
            or Path(str(manifest.get("execution_policy_path", ""))).resolve()
            != expected_policy
            or manifest.get("execution_policy_sha256")
            != DETECTOR_EXECUTION_POLICY_SHA256
            or manifest.get("execution_policy_content_sha256")
            != DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        ):
            raise InterfaceHalt(
                "detector manifest is not bound to the frozen primary preprocessing input"
            )
        _verify_declared_files(
            path,
            ({"path": name, **identity} for name, identity in outputs.items()),
        )
        started = _parse_aware_time(
            manifest.get("execution_started_at_utc"),
            "detector execution start",
        )
        completed = _parse_aware_time(
            manifest.get("execution_completed_at_utc"),
            "detector execution completion",
        )
        durations = manifest.get("fit_durations_seconds")
        checkpoint_seconds = float(
            manifest.get("checkpoint_cumulative_fit_seconds", -1.0)
        )
        if (
            completed < started
            or not isinstance(durations, list)
            or len(durations) != DETECTOR_TOTAL_FITS
            or any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in durations
            )
            or not math.isfinite(checkpoint_seconds)
            or checkpoint_seconds < 0
            or abs(sum(map(float, durations)) - checkpoint_seconds) > 1e-6
            or not isinstance(manifest.get("last_completed_checkpoint"), dict)
            or not str(manifest.get("execution_plan_sha256", ""))
        ):
            raise InterfaceHalt("detector checkpoint completion provenance is invalid")
        run_identity = manifest.get("run_identity")
        if (
            not isinstance(run_identity, dict)
            or manifest.get("run_identity_sha256")
            != canonical_json_sha256(run_identity)
            or run_identity.get("device") != DETECTOR_DEVICE
            or run_identity.get("gpu_uuid") != GPU_UUID
            or int(run_identity.get("workers", -1)) != DETECTOR_WORKERS
            or Path(str(run_identity.get("output_dir", ""))).resolve()
            != path.parent.resolve()
            or Path(str(run_identity.get("checkpoint_dir", ""))).resolve()
            != expected_checkpoint_dir
            or Path(str(run_identity.get("status_file", ""))).resolve()
            != expected_status_file
            or Path(str(run_identity.get("fit_permit_file", ""))).resolve()
            != expected_permit_file
            or Path(str(run_identity.get("fit_permit_receipt_dir", ""))).resolve()
            != expected_receipt_dir
            or Path(str(run_identity.get("execution_policy_path", ""))).resolve()
            != expected_policy
            or run_identity.get("execution_policy_sha256")
            != DETECTOR_EXECUTION_POLICY_SHA256
            or run_identity.get("require_fit_permit") is not True
        ):
            raise InterfaceHalt("detector final run identity is invalid")
        accounted_seconds = _validate_detector_gpu_accounting(
            manifest.get("gpu_accounting"), live=False
        )
        if accounted_seconds + 1e-6 < checkpoint_seconds:
            raise InterfaceHalt(
                "detector GPU wall accounting is shorter than committed fit time"
            )
        status_path = expected_status_file
        if status_path.is_symlink() or not status_path.is_file():
            raise InterfaceHalt("detector final signed status is absent or unsafe")
        status = read_json(status_path, label="detector final status")
        unsigned_status = dict(status)
        claimed_status = unsigned_status.pop("status_sha256", None)
        final_identity = status.get("final_manifest")
        if (
            status.get("schema_version") != DETECTOR_STATUS_SCHEMA
            or canonical_json_sha256(unsigned_status) != claimed_status
        ):
            raise InterfaceHalt("detector final signed status is invalid")
        if detector_manifest_awaits_supervisor_finalization(
            path, manifest, status
        ):
            return False
        if (
            status.get("state") != "complete"
            or int(status.get("completed_fit_count", -1))
            != DETECTOR_TOTAL_FITS
            or int(status.get("total_fit_count", -1))
            != DETECTOR_TOTAL_FITS
            or status.get("run_identity_sha256")
            != manifest.get("run_identity_sha256")
            or status.get("run_identity") != run_identity
            or status.get("gpu_accounting") != manifest.get("gpu_accounting")
            or not isinstance(final_identity, dict)
            or set(final_identity) != {"path", "sha256", "size_bytes"}
            or Path(str(final_identity.get("path", ""))).resolve()
            != path.resolve()
            or final_identity.get("sha256") != file_sha256(path)
            or int(final_identity.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise InterfaceHalt(
                "detector final status does not seal the exact manifest/accounting"
            )
        verify_detector_final_publication(path, manifest, status)
        return True
    if kind == "statistics_v1":
        if manifest.get("schema_version") != "1.0" or not isinstance(manifest.get("outputs"), dict):
            raise InterfaceHalt("invalid statistics run manifest")
        statistics_config = manifest.get("statistics_config")
        expected_statistics_config = (
            PROJECT_ROOT / "configs/revision_v1/statistics.json"
        ).resolve()
        if (
            not isinstance(statistics_config, dict)
            or _resolved_declared_path(path, statistics_config.get("path"))
            != expected_statistics_config
            or statistics_config.get("sha256")
            != file_sha256(expected_statistics_config)
            or set(manifest["outputs"])
            != {
                "continuous",
                "detectors",
                "effects",
                "integrity",
                "mixed",
                "mixed_status",
                "quality",
                "recovery",
            }
        ):
            raise InterfaceHalt("statistics manifest identity is not frozen/complete")
        input_rows = manifest.get("inputs")
        if not isinstance(input_rows, list):
            raise InterfaceHalt("statistics manifest lacks input lineage")
        _verify_declared_files(path, input_rows, size_keys=("bytes", "size_bytes"))
        expected_by_category = {
            "trials": {
                substitutions["primary_trials"],
                substitutions["ablation_trials"],
                substitutions["multilingual_trials"],
                substitutions["robustness_trials"],
                *(
                    substitutions[
                        "evaluator_{}_{}_continuous".format(stage, generator)
                    ]
                    for stage in EVALUATOR_SOURCE_STAGES
                    for generator in MODELS
                ),
            },
            "features": {
                substitutions["primary_features"],
                substitutions["ablation_features"],
                substitutions["multilingual_features"],
                substitutions["robustness_features"],
            },
            "detectors": {substitutions["detector_predictions"]},
            "runtime": {
                substitutions["primary_runtime"],
                substitutions["ablation_runtime"],
                substitutions["multilingual_runtime"],
                substitutions["robustness_runtime"],
            },
        }
        observed_by_category = {
            category: {
                str(_resolved_declared_path(path, row.get("path")))
                for row in input_rows
                if row.get("category") == category
            }
            for category in expected_by_category
        }
        if observed_by_category != {
            category: {str(Path(value).resolve()) for value in values}
            for category, values in expected_by_category.items()
        }:
            raise InterfaceHalt("statistics inputs differ from the complete frozen DAG")
        _verify_declared_files(path, manifest["outputs"].values())
        return True
    if kind == "mixed_models_v1":
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("manifest_type") != "rankcloak_revision_v1_mixed_model_run"
            or manifest.get("validation_only") is not False
            or manifest.get("fixed_effects_fallback") is not False
        ):
            raise InterfaceHalt("invalid confirmatory mixed-model manifest")
        outputs = manifest.get("outputs")
        declarations = outputs.values() if isinstance(outputs, dict) else outputs
        if not isinstance(declarations, Iterable):
            raise InterfaceHalt("mixed-model manifest lacks output identities")
        input_files = manifest.get("input_files")
        if not isinstance(input_files, list):
            raise InterfaceHalt("mixed-model manifest lacks input lineage")
        _verify_declared_files(path, input_files, size_keys=("size_bytes",))
        inputs_by_role = {
            str(row.get("role")): str(
                _resolved_declared_path(path, row.get("path"))
            )
            for row in input_files
        }
        expected_mixed_inputs = {
            "driver_source": str(
                (PROJECT_ROOT / "scripts/run_revision_mixed_models.R").resolve()
            ),
            "plan": str(
                (PROJECT_ROOT / "analysis/revision_v1/confirmatory_model_plan.json").resolve()
            ),
            "environment_lock": str(
                (PROJECT_ROOT / "analysis/revision_v1/r_environment.lock.json").resolve()
            ),
            "trials": str(Path(substitutions["primary_trials"]).resolve()),
            "features": str(Path(substitutions["evaluator_join_features"]).resolve()),
            "feature_join_manifest": str(Path(substitutions["evaluator_join_manifest"]).resolve()),
            "runtime": str(Path(substitutions["primary_runtime"]).resolve()),
            "detectors": str(Path(substitutions["detector_metrics"]).resolve()),
        }
        if inputs_by_role != expected_mixed_inputs:
            raise InterfaceHalt("mixed-model inputs differ from the locked primary DAG")
        _verify_declared_files(path, declarations, size_keys=("size_bytes",))
        return True
    if kind == "theory_v1":
        if manifest.get("schema_version") != "1.0" or manifest.get("artifact_type") != "rankcloak_capacity_quality_theory_validation":
            raise InterfaceHalt("invalid theory validation manifest")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 4:
            raise InterfaceHalt("theory manifest lacks the four frozen stage inputs")
        for row in inputs:
            candidate = _resolved_declared_path(path, row.get("path"))
            if file_sha256(candidate) != row.get("sha256"):
                raise InterfaceHalt("theory input hash mismatch: {}".format(candidate))
            if row.get("size_bytes") is not None and candidate.stat().st_size != int(row["size_bytes"]):
                raise InterfaceHalt("theory input size mismatch: {}".format(candidate))
        expected_theory_inputs = {
            str(Path(substitutions[key]).resolve())
            for key in (
                "primary_trials",
                "ablation_trials",
                "multilingual_trials",
                "robustness_trials",
            )
        }
        if {
            str(_resolved_declared_path(path, row.get("path"))) for row in inputs
        } != expected_theory_inputs:
            raise InterfaceHalt("theory inputs differ from the four frozen stages")
        _verify_declared_files(path, manifest.get("tables", []))
        return True
    if kind == "report_v1":
        if manifest.get("schema_version") != "rankcloak-revision-report-v1" or manifest.get("artifact_type") != "report_output_manifest":
            raise InterfaceHalt("invalid report output manifest")
        files = manifest.get("files")
        if not isinstance(files, list) or canonical_json_sha256(files) != manifest.get("files_sha256"):
            raise InterfaceHalt("report output file declaration is malformed")
        # The report uses a bytes digest of canonical JSON; individual hashes are
        # independently sufficient here and are rechecked by the report builder.
        _verify_declared_files(path, files)
        return True
    if kind == "figures_v1":
        registry = Path(str(completion["registry"].format_map(substitutions)))
        output_dir = Path(str(completion["output_dir"].format_map(substitutions)))
        renderer = Path(
            str(
                spec["interface"].get(
                    "runtime_path", spec["interface"]["path"]
                )
            ).format_map(substitutions)
        )
        if not renderer.is_absolute():
            renderer = PROJECT_ROOT / renderer
        if (
            not registry.is_file()
            or registry.is_symlink()
            or not renderer.is_file()
            or renderer.is_symlink()
            or not output_dir.is_dir()
        ):
            return False
        import csv

        with registry.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 18 or len({row.get("plot_id") for row in rows}) != 18:
            raise InterfaceHalt("plot registry is not the frozen 5+13 display registry")
        if (
            manifest.get("schema_version") != FIGURE_MANIFEST_SCHEMA
            or manifest.get("format") != str(completion.get("format", "pdf"))
            or manifest.get("plot_registry_sha256") != file_sha256(registry)
            or manifest.get("renderer_path") != str(renderer.resolve())
            or manifest.get("renderer_sha256") != file_sha256(renderer)
            or manifest.get("report_manifest_sha256")
            != file_sha256(Path(substitutions["report_manifest"]))
        ):
            raise InterfaceHalt("figure render manifest identity mismatch")
        unsigned = dict(manifest)
        claimed = unsigned.pop("manifest_sha256", None)
        if canonical_json_sha256(unsigned) != claimed:
            raise InterfaceHalt("figure render manifest self-hash mismatch")
        figures = manifest.get("figures")
        if (
            not isinstance(figures, list)
            or len(figures) != 18
            or {str(item.get("plot_id")) for item in figures}
            != {str(row.get("plot_id")) for row in rows}
            or manifest.get("figures_sha256") != canonical_json_sha256(figures)
        ):
            raise InterfaceHalt("figure render declaration is incomplete")
        _verify_declared_files(path, figures)
        for item in figures:
            figure = path.parent / str(item["path"])
            if figure.read_bytes()[:4] != b"%PDF":
                raise InterfaceHalt("rendered figure is not a PDF: {}".format(figure))
        return True
    raise InterfaceHalt("unknown completion-contract kind: {}".format(kind))


def preprocess_specs() -> list[dict[str, Any]]:
    substitutions = _format_values()
    specs: list[dict[str, Any]] = []
    for stage in ("primary_v2", "ablation_v2", "multilingual_v2", "robustness_v2"):
        argv = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "scripts/preprocess_revision_results.py",
        ]
        for model in MODELS:
            argv.extend(["--run-dir", str(PROJECT_ROOT / RESULTS_ROOT / stage / model)])
        if stage in {"ablation_v2", "robustness_v2"}:
            for model in MODELS:
                argv.extend(
                    [
                        "--reference-run-dir",
                        str(PROJECT_ROOT / RESULTS_ROOT / "primary_v2" / model),
                    ]
                )
        if stage == "robustness_v2":
            for model in MODELS:
                argv.extend(
                    [
                        "--reference-run-dir",
                        str(PROJECT_ROOT / RESULTS_ROOT / "ablation_v2" / model),
                    ]
                )
        output = Path(substitutions[stage.replace("_v2", "") + "_input_dir"])
        argv.extend(["--output-dir", str(output)])
        specs.append(
            {
                "operation_id": "preprocess_" + stage,
                "stage": "preprocessing",
                "argv": argv,
                "output_dir": str(output),
                "interface": {
                    "path": "scripts/preprocess_revision_results.py",
                    "probe_argv": [
                        str(PROJECT_ROOT / ".venv/bin/python"),
                        "scripts/preprocess_revision_results.py",
                        "--help",
                    ],
                    "required_help_tokens": [
                        "--run-dir",
                        "--reference-run-dir",
                        "--output-dir",
                    ],
                },
                "completion": {
                    "kind": "preprocess_v2",
                    "path": str(output / "preprocessing_output_manifest.json"),
                    "stage": stage,
                },
                "atomic_staging": True,
            }
        )
    return specs


def _staged_preprocess_argv(
    action: Action, spec: Mapping[str, Any], retry_index: int
) -> tuple[tuple[str, ...], Path]:
    assert action.output_dir is not None
    staging = action.output_dir.parent / ".{}.attempt-{}-{}".format(
        action.output_dir.name, retry_index, time.time_ns()
    )
    if staging.exists() or staging.is_symlink():
        raise InterfaceHalt("preprocessing staging path already exists: {}".format(staging))
    argv = list(action.argv)
    output_index = argv.index("--output-dir") + 1
    argv[output_index] = str(staging)
    return tuple(argv), staging


def execute_downstream_action(
    action: Action,
    spec: Mapping[str, Any],
    *,
    retries: dict[str, int],
    max_retries: int,
) -> None:
    substitutions = _format_values()
    if verify_completion(spec, substitutions):
        return
    verify_downstream_interface(spec, substitutions)
    while True:
        retry_index = retries.get(action.action_id, 0)
        argv = action.argv
        staging: Path | None = None
        if spec.get("atomic_staging"):
            argv, staging = _staged_preprocess_argv(action, spec, retry_index)
        code, detail = run_cpu_process(action, argv)
        if code == 0:
            if staging is not None:
                assert action.output_dir is not None
                if action.output_dir.exists():
                    raise InterfaceHalt("preprocessing destination appeared during staging")
                os.replace(staging, action.output_dir)
            if verify_completion(spec, substitutions):
                emit_event("downstream_action_complete", action_id=action.action_id)
                return
            raise InterfaceHalt("downstream command exited zero without its required manifest/schema")
        if _nonrecoverable_output(detail):
            raise MethodologicalHalt(
                "nonrecoverable downstream failure for {}: {}".format(
                    action.action_id, detail[-2000:]
                )
            )
        retries[action.action_id] = retry_index + 1
        record_recoverable_error(action, retries[action.action_id], "downstream_command_failed", detail)
        if retries[action.action_id] > max_retries:
            raise OrchestratorError("retry ceiling exceeded for {}".format(action.action_id))


def execute_figures(
    action: Action,
    spec: Mapping[str, Any],
    *,
    retries: dict[str, int],
    max_retries: int,
) -> None:
    substitutions = _format_values()
    if verify_completion(spec, substitutions):
        return
    verify_downstream_interface(spec, substitutions)
    completion = spec["completion"]
    registry = Path(str(completion["registry"].format_map(substitutions)))
    output_dir = Path(str(completion["output_dir"].format_map(substitutions)))
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with registry.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18 or len({row.get("plot_id") for row in rows}) != 18:
        raise InterfaceHalt("plot registry is not the frozen 5+13 display registry")
    for row in rows:
        plot_id = str(row["plot_id"])
        extension = str(completion.get("format", "pdf"))
        destination = output_dir / (plot_id + "." + extension)
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or destination.stat().st_size <= 0:
                raise InterfaceHalt("existing rendered figure is invalid: {}".format(destination))
            continue
        argv = _expand_argv(spec["argv"], substitutions) + ("--only", plot_id)
        code, detail = run_cpu_process(action, argv)
        if code != 0 or not destination.is_file() or destination.stat().st_size <= 0:
            retries[action.action_id] = retries.get(action.action_id, 0) + 1
            record_recoverable_error(action, retries[action.action_id], "figure_render_failed", detail)
            if retries[action.action_id] > max_retries:
                raise OrchestratorError("retry ceiling exceeded while rendering figures")
            if destination.exists():
                raise InterfaceHalt("failed renderer left an ambiguous figure artifact")
            return execute_figures(
                action, spec, retries=retries, max_retries=max_retries
            )
    figures = []
    for row in sorted(rows, key=lambda value: str(value["plot_id"])):
        plot_id = str(row["plot_id"])
        destination = output_dir / (plot_id + "." + extension)
        if destination.read_bytes()[:4] != b"%PDF":
            raise InterfaceHalt("renderer output is not a PDF: {}".format(destination))
        figures.append(
            {
                "plot_id": plot_id,
                "path": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        )
    renderer_path = Path(
        str(
            spec["interface"].get(
                "runtime_path", spec["interface"]["path"]
            )
        ).format_map(substitutions)
    )
    if not renderer_path.is_absolute():
        renderer_path = PROJECT_ROOT / renderer_path
    figure_manifest: dict[str, Any] = {
        "schema_version": FIGURE_MANIFEST_SCHEMA,
        "generated_at": utc_now(),
        "format": extension,
        "plot_registry_path": str(registry.resolve()),
        "plot_registry_sha256": file_sha256(registry),
        "renderer_path": str(renderer_path.resolve()),
        "renderer_sha256": file_sha256(renderer_path),
        "report_manifest_path": substitutions["report_manifest"],
        "report_manifest_sha256": file_sha256(
            Path(substitutions["report_manifest"])
        ),
        "figures": figures,
        "figures_sha256": canonical_json_sha256(figures),
    }
    figure_manifest["manifest_sha256"] = canonical_json_sha256(figure_manifest)
    manifest_path = Path(
        str(completion["path"].format_map(substitutions))
    )
    if manifest_path.exists():
        observed = read_json(manifest_path, label="figure render manifest")
        if observed != figure_manifest:
            # A manifest generated in an earlier process has a different
            # timestamp but must be verified, never silently replaced.
            if not verify_completion(spec, substitutions):
                raise InterfaceHalt("existing figure render manifest is invalid")
    else:
        atomic_write_json(manifest_path, figure_manifest)
    if not verify_completion(spec, substitutions):
        raise InterfaceHalt("figure rendering completed without the frozen registry outputs")
    emit_event("downstream_action_complete", action_id=action.action_id)


def _all_support_complete(actions: Sequence[Action]) -> bool:
    return all(
        shard_status(action) == "complete"
        for action in actions
        if action.kind == "runner"
    )


def require_canonical_evaluator_completion(progress: Mapping[str, Any]) -> None:
    manifest = verify_evaluator_unavailability_manifest()
    rows = [
        row
        for row in progress.get("stage_progress", [])
        if isinstance(row, dict) and row.get("stage") == "heldout_evaluator"
    ]
    if len(rows) != 1:
        raise InterfaceHalt("canonical progress lacks one heldout_evaluator stage row")
    row = rows[0]
    if (
        int(row.get("total", -1)) != int(manifest["frozen_evaluator_target_units"])
        or int(row.get("completed", -1))
        != int(manifest["terminal_accounted_units"])
        or int(row.get("unavailable", -1))
        < int(manifest["upstream_dependent_unavailable_units"])
        or int(row.get("remaining", -1)) != 0
    ):
        raise InterfaceHalt(
            "canonical progress has not incorporated the 48-unit held-out "
            "upstream-unavailability manifest"
        )


def run_orchestrator(args: argparse.Namespace) -> int:
    projection = verify_projection()
    contract = load_command_contract(args.command_contract)
    gpu_actions = build_gpu_actions()
    retries = load_retry_counts()
    evaluator_gate_checked = False

    while True:
        try:
            progress = operational_progress()
        except DeferrableProgressRefresh as exc:
            # Preserve any attached-action state already on disk.  The event is
            # durable, and the next loop re-observes occupancy before launch.
            defer_progress_refresh(
                deferral=exc,
                projection=projection,
                action=None,
                retries=retries,
                status="progress_snapshot_race_deferred",
                message=(
                    "Canonical progress refresh is operationally deferred; "
                    "no action state changed."
                ),
                persist_state=False,
            )
            time.sleep(args.poll_seconds)
            continue
        budget = calculate_budget(projection, progress)
        enforce_budget(budget)
        primary_complete, incomplete_primary = primary_gate_status()
        if not primary_complete:
            write_state(
                status="waiting_for_primary_v2",
                message="Post-primary DAG is gated on all three exact primary_v2 checkpoints: {}".format(
                    ", ".join(incomplete_primary)
                ),
                action=None,
                retries=retries,
                progress=progress,
                budget=budget,
            )
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
            continue

        if _all_support_complete(gpu_actions) and not evaluator_gate_checked:
            evaluator_projection_gate(gpu_actions, projection)
            evaluator_gate_checked = True

        next_action = next(
            (action for action in gpu_actions if not _completion_for_gpu_action(action)),
            None,
        )
        if next_action is not None:
            if (PROJECT_ROOT / FINAL_PROGRESS_PATH).exists():
                raise MethodologicalHalt(
                    "immutable final progress exists while a GPU action is incomplete"
                )
            if next_action.kind.startswith("evaluator") and not evaluator_gate_checked:
                # No evaluator GPU launch occurs until all support shards and all
                # exact evaluator dry runs agree with the authorized total.
                if not _all_support_complete(gpu_actions):
                    raise MethodologicalHalt("evaluator reached before supporting runner DAG completed")
                evaluator_projection_gate(gpu_actions, projection)
                evaluator_gate_checked = True
            if next_action.gpu and wait_for_existing_gpu_occupancy(
                next_action,
                projection=projection,
                retries=retries,
                max_retries=args.max_retries_per_action,
                poll_seconds=args.poll_seconds,
            ):
                continue
            # The attached process can finish between action selection and the
            # next occupancy check. Revalidate before constructing any launch
            # command so a newly complete shard is never started again.
            if _completion_for_gpu_action(next_action):
                continue
            if next_action.kind == "evaluator_export":
                code, detail = run_cpu_process(next_action, next_action.argv)
                if code == 0 and shard_status(next_action) == "complete":
                    write_export_marker(next_action)
                    emit_event("evaluator_export_pass_complete", action_id=next_action.action_id)
                    continue
            else:
                argv = _argv_for_gpu_action(next_action)
                try:
                    code, detail = run_gpu_process(
                        next_action,
                        argv,
                        projection=projection,
                        retries=retries,
                        poll_seconds=args.poll_seconds,
                    )
                except DeferrableProgressRefresh as exc:
                    # The launch-time ceiling refresh failed before Popen.
                    # Defer without launching or charging the action.
                    defer_progress_refresh(
                        deferral=exc,
                        projection=projection,
                        action=next_action,
                        retries=retries,
                        status="progress_snapshot_race_deferred_before_launch",
                        message=(
                            "Canonical progress refresh is operationally deferred; "
                            "no GPU launch occurred."
                        ),
                    )
                    time.sleep(args.poll_seconds)
                    continue
                except TransientGPUOccupancy:
                    # A process appeared after the preceding occupancy scan.
                    # Return to the wait/attach path without charging a retry.
                    continue
                if code == 0 and shard_status(next_action) == "complete":
                    emit_event(
                        "gpu_action_complete",
                        action_id=next_action.action_id,
                        stage=next_action.stage,
                        model_id=next_action.model_id,
                    )
                    continue
            if _nonrecoverable_output(detail):
                raise MethodologicalHalt(
                    "nonrecoverable GPU/export failure for {}: {}".format(
                        next_action.action_id, detail[-2000:]
                    )
                )
            retries[next_action.action_id] = retries.get(next_action.action_id, 0) + 1
            record_recoverable_error(
                next_action,
                retries[next_action.action_id],
                "gpu_or_export_process_incomplete",
                detail,
            )
            if retries[next_action.action_id] > args.max_retries_per_action:
                raise OrchestratorError(
                    "retry ceiling exceeded for {}".format(next_action.action_id)
                )
            try:
                progress = refresh_progress()
            except DeferrableProgressRefresh as exc:
                defer_progress_refresh(
                    deferral=exc,
                    projection=projection,
                    action=next_action,
                    retries=retries,
                    status="retrying_from_checkpoint_progress_snapshot_race_deferred",
                    message=(
                        "The process retry was already recorded; canonical progress "
                        "refresh is deferred without an additional action retry."
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            budget = calculate_budget(projection, progress)
            write_state(
                status="retrying_from_checkpoint",
                message="Recoverable process exit recorded; resuming latest valid checkpoint.",
                action=next_action,
                retries=retries,
                progress=progress,
                budget=budget,
            )
            continue

        downstream = []
        for spec in preprocess_specs():
            action = Action(
                action_id="downstream:" + str(spec["operation_id"]),
                stage="preprocessing",
                kind="downstream",
                argv=tuple(map(str, spec["argv"])),
                output_dir=Path(str(spec["output_dir"])),
            )
            downstream.append((action, spec))
        downstream.extend(downstream_actions(contract))
        downstream_deferred = False
        for action, spec in downstream:
            if spec.get("operation_id") == "reports":
                if (PROJECT_ROOT / FINAL_PROGRESS_PATH).exists():
                    progress = verify_final_progress_snapshot()
                else:
                    try:
                        progress = refresh_progress()
                    except DeferrableProgressRefresh as exc:
                        defer_progress_refresh(
                            deferral=exc,
                            projection=projection,
                            action=action,
                            retries=retries,
                            status="progress_snapshot_race_deferred_before_seal",
                            message=(
                                "Canonical progress refresh is deferred before "
                                "the immutable seal; no downstream action ran."
                            ),
                        )
                        time.sleep(args.poll_seconds)
                        downstream_deferred = True
                        break
                    budget = calculate_budget(projection, progress)
                    enforce_budget(budget)
                    write_state(
                        status="sealing_final_progress",
                        message=(
                            "Publishing the immutable source-bound progress snapshot "
                            "before reports and figures."
                        ),
                        action=action,
                        retries=retries,
                        progress=progress,
                        budget=budget,
                    )
                    progress = seal_final_progress_snapshot(progress)
            substitutions = _format_values()
            if verify_completion(spec, substitutions):
                continue
            if spec.get("operation_id") == "detector":
                ensure_detector_equivalence_gate(
                    args, projection, contract
                )
                require_detector_equivalence_gate()
                execute_checkpointed_detector_action(
                    action,
                    spec,
                    projection=projection,
                    retries=retries,
                    max_retries=args.max_retries_per_action,
                    poll_seconds=args.poll_seconds,
                )
                continue
            try:
                progress = operational_progress()
            except DeferrableProgressRefresh as exc:
                defer_progress_refresh(
                    deferral=exc,
                    projection=projection,
                    action=action,
                    retries=retries,
                    status="progress_snapshot_race_deferred_before_downstream",
                    message=(
                        "Canonical progress refresh is deferred before the "
                        "hash-gated downstream action; no action retry was charged."
                    ),
                )
                time.sleep(args.poll_seconds)
                downstream_deferred = True
                break
            budget = calculate_budget(projection, progress)
            enforce_budget(budget)
            write_state(
                status="running_downstream",
                message="Executing hash-gated CPU analysis/reporting DAG.",
                action=action,
                retries=retries,
                progress=progress,
                budget=budget,
            )
            if spec["completion"].get("kind") == "figures_v1":
                execute_figures(
                    action,
                    spec,
                    retries=retries,
                    max_retries=args.max_retries_per_action,
                )
            else:
                execute_downstream_action(
                    action,
                    spec,
                    retries=retries,
                    max_retries=args.max_retries_per_action,
                )

        if downstream_deferred:
            continue
        progress = verify_final_progress_snapshot()
        require_canonical_evaluator_completion(progress)
        budget = calculate_budget(projection, progress)
        write_state(
            status="complete",
            message="Full frozen post-primary computational DAG and figures are complete.",
            action=None,
            retries=retries,
            progress=progress,
            budget=budget,
        )
        emit_event("confirmatory_post_primary_complete", budget=budget)
        return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries-per-action", type=int, default=5)
    parser.add_argument("--command-contract", type=Path, default=DEFAULT_COMMAND_CONTRACT)
    parser.add_argument(
        "--detector-benchmark-task-index",
        type=int,
        choices=tuple(sorted(DETECTOR_BENCHMARK_TASKS)),
        help=(
            "Run only frozen detector task 0 or 1 through the signed budget gate, "
            "write its benchmark record, and stop at the durable fit boundary."
        ),
    )
    parser.add_argument(
        "--detector-equivalence-task-index",
        type=int,
        choices=tuple(sorted(DETECTOR_BENCHMARK_TASKS)),
        help="Run only one frozen task0/task1 detector equivalence step.",
    )
    parser.add_argument(
        "--detector-equivalence-role",
        choices=("cuda", "cuda_repeat", "report"),
        help=(
            "Select the benchmark-checkpoint CUDA export, independent CUDA "
            "repeat, or signed same-CUDA reproducibility report."
        ),
    )
    parser.add_argument(
        "--detector-production-only",
        action="store_true",
        help=(
            "After both signed reproducibility reports and the final budget gate, "
            "run only the frozen 56-fit detector matrix and exit."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Publish a waiting state once instead of polling for primary completion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable action DAG without launching or writing status.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not 5 <= args.poll_seconds <= 300:
        raise OrchestratorError("--poll-seconds must be between 5 and 300")
    if args.max_retries_per_action < 1:
        raise OrchestratorError("--max-retries-per-action must be positive")
    projection = verify_projection()
    contract = load_command_contract(args.command_contract)
    equivalence_mode = (
        args.detector_equivalence_task_index is not None
        or args.detector_equivalence_role is not None
    )
    if args.detector_benchmark_task_index is not None and (
        args.once or args.dry_run or equivalence_mode or args.detector_production_only
    ):
        raise OrchestratorError(
            "detector benchmark mode cannot be combined with other special modes"
        )
    if equivalence_mode and (
        args.detector_equivalence_task_index is None
        or args.detector_equivalence_role is None
        or args.once
        or args.dry_run
        or args.detector_production_only
    ):
        raise OrchestratorError(
            "detector equivalence mode requires both task index and role and "
            "cannot be combined with --once/--dry-run"
        )
    if args.detector_production_only and (args.once or args.dry_run):
        raise OrchestratorError(
            "detector production-only mode cannot be combined with --once/--dry-run"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "authorized_projection_sha256": projection["projection_sha256"],
                    "gpu_uuid": GPU_UUID,
                    "gpu_actions": [
                        {
                            "action_id": action.action_id,
                            "stage": action.stage,
                            "kind": action.kind,
                            "argv": list(action.argv),
                            "expected_count": action.expected_count,
                        }
                        for action in build_gpu_actions()
                    ],
                    "preprocessing_actions": [spec["operation_id"] for spec in preprocess_specs()],
                    "downstream_actions": [
                        spec["operation_id"] for spec in contract["operations"]
                    ],
                    "mutations_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    lock_descriptor = acquire_single_instance_lock()
    try:
        try:
            if args.detector_benchmark_task_index is not None:
                return run_detector_benchmark(args, projection, contract)
            if equivalence_mode:
                task_index = int(args.detector_equivalence_task_index)
                role = str(args.detector_equivalence_role)
                if role == "report":
                    run_detector_equivalence_report(task_index=task_index)
                    return 0
                return run_detector_equivalence_role(
                    args,
                    projection,
                    contract,
                    task_index=task_index,
                    role=role,
                )
            if args.detector_production_only:
                require_detector_equivalence_gate()
                require_detector_cuda_budget_gate(
                    expected_stage="post_reproducibility_preproduction"
                )
                action, spec = next(
                    item
                    for item in downstream_actions(contract)
                    if item[1].get("operation_id") == "detector"
                )
                execute_checkpointed_detector_action(
                    action,
                    spec,
                    projection=projection,
                    retries=load_retry_counts(),
                    max_retries=args.max_retries_per_action,
                    poll_seconds=args.poll_seconds,
                )
                return 0
            return run_orchestrator(args)
        except BaseException as exc:
            detector = None
            if equivalence_mode and args.detector_equivalence_role in {
                "cuda",
                "cuda_repeat",
            }:
                detector = _detector_equivalence_action(
                    contract,
                    task_index=int(args.detector_equivalence_task_index),
                    role=str(args.detector_equivalence_role),
                )
            elif not equivalence_mode and args.detector_benchmark_task_index is None:
                detector = next(
                    action
                    for action, spec in downstream_actions(contract)
                    if spec.get("operation_id") == "detector"
                )
            elif not equivalence_mode:
                detector, _spec, _output = _benchmark_detector_action(
                    contract, int(args.detector_benchmark_task_index)
                )
            if detector is not None:
                _stop_exact_detector_if_live(
                    detector, immediate=isinstance(exc, BudgetHalt)
                )
            raise
    finally:
        os.close(lock_descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetHalt as exc:
        print("confirmatory orchestration halted at budget ceiling: {}".format(exc), file=sys.stderr)
        raise SystemExit(3)
    except MethodologicalHalt as exc:
        print("confirmatory orchestration halted for methodological integrity: {}".format(exc), file=sys.stderr)
        raise SystemExit(4)
    except InterfaceHalt as exc:
        print("confirmatory orchestration halted at downstream interface: {}".format(exc), file=sys.stderr)
        raise SystemExit(5)
    except OrchestratorError as exc:
        print("confirmatory orchestration failed: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
