#!/usr/bin/env python3
"""Fail-closed supervisor for the three frozen primary_v2 model shards.

This script is operational tooling only.  It never edits a result shard.  It
attaches to an already-running primary_v2 process, refreshes the canonical
progress snapshot, resumes an incomplete shard after a recoverable process
exit, and starts the next model only after the previous checkpoint is exact
and complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_artifacts import canonical_json_sha256, trial_ids_sha256


GPU_UUID = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"
MODELS = (
    "llama3_8b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m",
)
PLANNED_PER_MODEL = 4800
CONFIG_SHA256 = "dc0e7e022036e2681c87ad06446cbebd56d676faf81a0544a55d56375d4eadcd"
PROTOCOL_REVISION = "payload_fidelity_v2"
RESULT_REVISION = "payload_aware_result_v2"
PROJECTION_SELF_SHA256 = "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
PROJECTION_PATH = Path("results/revision_v1/compute_projection_165h_v2.json")
RESULTS_ROOT = Path("results/revision_v1")
PRIMARY_ROOT = Path("results/revision_v1/primary_v2")
STATE_ROOT = Path("results/revision_v1/supervisor")
STATE_PATH = STATE_ROOT / "primary_v2_supervisor_state.json"
ERRORS_PATH = STATE_ROOT / "primary_v2_recovered_errors.jsonl"
LOG_ROOT = STATE_ROOT / "logs"
PROGRESS_COMMAND = (
    ".venv/bin/python",
    "scripts/update_revision_progress.py",
    "--write",
    "--compact",
)
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


class SupervisorError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"expected JSON object at {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if tmp.exists() or tmp.is_symlink():
        raise SupervisorError(f"temporary state path already exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_error(value: dict[str, Any]) -> None:
    path = PROJECT_ROOT / ERRORS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def progress_refresh_recovery_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / ERRORS_PATH
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise SupervisorError(
                            f"{path}:{line_number} is not a JSON object"
                        )
                    rows.append(value)
        except (OSError, json.JSONDecodeError) as exc:
            raise SupervisorError(f"cannot read recovered-error log {path}: {exc}") from exc
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


def verify_projection() -> dict[str, Any]:
    value = read_json(PROJECT_ROOT / PROJECTION_PATH)
    observed = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "projection_sha256"}
    )
    if observed != value.get("projection_sha256") or observed != PROJECTION_SELF_SHA256:
        raise SupervisorError("authorized 165-hour projection self-hash mismatch")
    if value.get("budget_gpu_hours") != 165.0:
        raise SupervisorError("projection is not bound to the 165 GPU-hour ceiling")
    decision = value.get("decision")
    if not isinstance(decision, dict) or decision.get("status") != "go_within_budget" or decision.get("go") is not True:
        raise SupervisorError("authorized compute projection is not GO")
    totals = value.get("totals")
    if not isinstance(totals, dict) or float(totals.get("upper_gpu_hours", 999.0)) >= 165.0:
        raise SupervisorError("authorized projection is not below 165 GPU-hours")
    return value


def proc_primary_commands() -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if "scripts/run_revision_matrix.py" not in args:
            continue
        if "--stage" not in args:
            continue
        index = args.index("--stage")
        if index + 1 < len(args) and args[index + 1] == "primary_v2":
            found[int(entry.name)] = args
    return found


def arg_value(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    return args[index + 1] if index + 1 < len(args) else None


def verify_active_processes(expected_model: str | None) -> dict[int, list[str]]:
    active = proc_primary_commands()
    if len(active) > 1:
        raise SupervisorError(f"multiple primary_v2 runner processes detected: {sorted(active)}")
    for pid, args in active.items():
        required_options = (
            "--stage",
            "--model",
            "--gpu-uuid",
            "--context",
            "--n-gpu-layers",
            "--output-dir",
        )
        for option in required_options:
            if args.count(option) != 1:
                raise SupervisorError(f"primary process {pid} has duplicate or missing {option}")
        allowed_options = set(required_options) | {"--resume"}
        unexpected_options = {
            item for item in args if item.startswith("--") and item not in allowed_options
        }
        if unexpected_options:
            raise SupervisorError(
                f"primary process {pid} has non-frozen options: {sorted(unexpected_options)}"
            )
        if args.count("--resume") > 1:
            raise SupervisorError(f"primary process {pid} repeats --resume")
        model = arg_value(args, "--model")
        if expected_model is not None and model != expected_model:
            raise SupervisorError(
                f"active primary process model {model!r} does not match expected {expected_model!r}"
            )
        if arg_value(args, "--gpu-uuid") != GPU_UUID:
            raise SupervisorError(f"primary process {pid} is not bound to the authorized GPU")
        if arg_value(args, "--context") != "4096" or arg_value(args, "--n-gpu-layers") != "-1":
            raise SupervisorError(f"primary process {pid} has a non-frozen execution configuration")
        expected_output = str(PRIMARY_ROOT / str(model))
        if arg_value(args, "--output-dir") != expected_output:
            raise SupervisorError(f"primary process {pid} has unexpected output directory")
        if "--limit" in args:
            raise SupervisorError(f"primary process {pid} uses forbidden --limit")
    return active


def verify_gpu_occupancy(active: dict[int, list[str]]) -> None:
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
            timeout=15.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise SupervisorError("GPU occupancy query timed out") from exc
    if query.returncode != 0:
        raise SupervisorError(f"cannot query GPU occupancy: {query.stderr.strip()}")
    authorized_pids: set[int] = set()
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise SupervisorError(f"malformed nvidia-smi row: {line!r}")
        gpu_uuid, pid_text = fields
        if gpu_uuid == GPU_UUID:
            authorized_pids.add(int(pid_text))
    active_pids = set(active)
    unexpected = authorized_pids - active_pids
    if unexpected:
        raise SupervisorError(
            f"unrelated compute process(es) occupy the authorized GPU: {sorted(unexpected)}"
        )
    if active_pids and authorized_pids != active_pids:
        raise SupervisorError("primary runner process is not visible on the authorized GPU")


def checkpoint_state(model: str) -> dict[str, Any] | None:
    shard = PROJECT_ROOT / PRIMARY_ROOT / model
    checkpoint_path = shard / "checkpoint.json"
    if not checkpoint_path.exists():
        if shard.exists() and any(shard.iterdir()):
            raise SupervisorError(f"nonempty shard lacks checkpoint: {shard}")
        return None
    checkpoint = read_json(checkpoint_path)
    expected_study_id = f"rankcloak_scientific_reports_revision_v1/primary_v2/{model}"
    if checkpoint.get("study_id") != expected_study_id:
        raise SupervisorError(f"checkpoint study identity mismatch for {model}")
    if checkpoint.get("planned_trial_count") != PLANNED_PER_MODEL:
        raise SupervisorError(f"wrong planned count in {checkpoint_path}")
    if checkpoint.get("config_manifest_sha256") != CONFIG_SHA256:
        raise SupervisorError(f"config hash mismatch in {checkpoint_path}")
    completed = checkpoint.get("completed_trial_ids")
    failed = checkpoint.get("failed_trial_ids")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        raise SupervisorError(f"invalid completed IDs in {checkpoint_path}")
    if not isinstance(failed, list) or len(failed) != len(set(failed)):
        raise SupervisorError(f"invalid failed IDs in {checkpoint_path}")
    if len(completed) > PLANNED_PER_MODEL:
        raise SupervisorError(f"too many completed IDs in {checkpoint_path}")
    identity = read_json(shard / "run_identity.json")
    identity_digest = canonical_json_sha256(
        {key: item for key, item in identity.items() if key != "run_identity_sha256"}
    )
    if identity.get("run_identity_sha256") != identity_digest:
        raise SupervisorError(f"run identity self-hash mismatch for {model}")
    if identity.get("study_id") != expected_study_id:
        raise SupervisorError(f"run identity study identity mismatch for {model}")
    if identity.get("config_manifest_sha256") != CONFIG_SHA256:
        raise SupervisorError(f"run identity config hash mismatch for {model}")
    model_artifacts = identity.get("model_artifacts")
    if not isinstance(model_artifacts, list) or len(model_artifacts) != 1:
        raise SupervisorError(f"run identity model artifact binding missing for {model}")
    configured_model = model_artifacts[0].get("configured_model")
    if not isinstance(configured_model, dict) or configured_model.get("model_id") != model:
        raise SupervisorError(f"run identity model binding mismatch for {model}")
    if identity.get("protocol_contract_revision") != PROTOCOL_REVISION:
        raise SupervisorError(f"protocol revision mismatch for {model}")
    if identity.get("result_schema_revision") != RESULT_REVISION:
        raise SupervisorError(f"result revision mismatch for {model}")
    if identity.get("planned_trial_count") != PLANNED_PER_MODEL:
        raise SupervisorError(f"run identity count mismatch for {model}")
    plan_ids: list[str] = []
    with (shard / "plan.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            work_id = row.get("work_id")
            if not isinstance(work_id, str) or not work_id:
                raise SupervisorError(f"plan row lacks work_id for {model}")
            plan_ids.append(work_id)
    if len(plan_ids) != PLANNED_PER_MODEL or len(plan_ids) != len(set(plan_ids)):
        raise SupervisorError(f"plan IDs are not exact and unique for {model}")
    plan_ids_digest = trial_ids_sha256(plan_ids)
    if checkpoint.get("planned_trial_ids_sha256") != plan_ids_digest:
        raise SupervisorError(f"checkpoint is not bound to the exact plan IDs for {model}")
    if identity.get("planned_trial_ids_sha256") != plan_ids_digest:
        raise SupervisorError(f"run identity is not bound to the exact plan IDs for {model}")
    plan_id_set = set(plan_ids)
    if not set(completed).issubset(plan_id_set) or not set(failed).issubset(plan_id_set):
        raise SupervisorError(f"checkpoint contains IDs outside the frozen plan for {model}")
    durable_completed: list[str] = []
    records_path = shard / "records.jsonl"
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") == "execution_failure":
                    continue
                work_id = row.get("work_id")
                if not isinstance(work_id, str):
                    raise SupervisorError(f"durable record lacks work_id for {model}")
                if work_id not in plan_id_set:
                    raise SupervisorError(f"durable record is outside the frozen plan for {model}")
                if row.get("execution_status") != "completed":
                    raise SupervisorError(f"nonfailure durable record is not completed for {model}")
                durable_completed.append(work_id)
    if len(durable_completed) != len(set(durable_completed)):
        raise SupervisorError(f"duplicate durable completed records for {model}")
    durable_set = set(durable_completed)
    completed_set = set(completed)
    if not completed_set.issubset(durable_set):
        raise SupervisorError(f"checkpoint-only completion detected for {model}")
    trailing_durable = durable_set - completed_set
    if len(trailing_durable) > 1:
        raise SupervisorError(f"more than one unreconciled durable completion for {model}")
    return {
        "completed": len(completed),
        "failed": len(failed),
        "checkpoint": checkpoint,
        "complete": len(completed) == PLANNED_PER_MODEL and not failed,
    }


def projection_primary_rows(projection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = projection.get("projection_rows")
    if not isinstance(rows, list):
        raise SupervisorError("projection rows missing")
    selected = {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("stage") == "primary_v2"
    }
    if set(selected) != set(MODELS):
        raise SupervisorError("projection does not contain all primary model rows")
    return selected


def record_stratum(record: dict[str, Any]) -> str:
    if record.get("record_type") == "ordinary_control":
        view = record.get("control_view")
        if view not in {"forced_span", "full_message"}:
            raise SupervisorError(f"unknown control view in durable record: {view!r}")
        return f"control:{view}"
    if record.get("record_type") == "rankcloak_trial":
        protocol = record.get("protocol_variant")
        return f"rankcloak:{protocol}"
    raise SupervisorError(f"unknown primary record type: {record.get('record_type')!r}")


def revised_upper_hours(projection: dict[str, Any]) -> dict[str, float]:
    baseline = float(projection["totals"]["upper_gpu_hours"])
    rows = projection_primary_rows(projection)
    removed_upper_seconds = 0.0
    actual_primary_seconds = 0.0
    for model in MODELS:
        shard = PROJECT_ROOT / PRIMARY_ROOT / model
        checkpoint_path = shard / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        checkpoint = read_json(checkpoint_path)
        created = datetime.fromisoformat(str(checkpoint["created_at"]))
        updated = datetime.fromisoformat(str(checkpoint["updated_at"]))
        actual_primary_seconds += max(0.0, (updated - created).total_seconds())
        model_row = rows[model]
        model_load = model_row.get("model_load")
        if not isinstance(model_load, dict):
            raise SupervisorError(f"projection model-load row missing for {model}")
        removed_upper_seconds += float(model_load["upper_seconds_per_unit"])
        strata_rows = model_row.get("strata")
        if not isinstance(strata_rows, list):
            raise SupervisorError(f"projection strata missing for {model}")
        upper_by_stratum = {
            str(item["stratum"]): float(item["upper_seconds_per_unit"])
            for item in strata_rows
            if isinstance(item, dict)
        }
        records_path = shard / "records.jsonl"
        if not records_path.exists():
            continue
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") == "execution_failure":
                    continue
                stratum = record_stratum(record)
                if stratum not in upper_by_stratum:
                    raise SupervisorError(f"no projected upper rate for {model}/{stratum}")
                removed_upper_seconds += upper_by_stratum[stratum]
    revised = baseline - removed_upper_seconds / 3600.0 + actual_primary_seconds / 3600.0
    return {
        "baseline_upper_gpu_hours": baseline,
        "projected_upper_consumed_gpu_hours": removed_upper_seconds / 3600.0,
        "actual_primary_gpu_hours": actual_primary_seconds / 3600.0,
        "revised_upper_gpu_hours": revised,
        "revised_headroom_gpu_hours": 165.0 - revised,
    }


def terminate_active(active: dict[int, list[str]]) -> None:
    for pid in active:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


def launch(model: str, resume: bool) -> subprocess.Popen[str]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_ROOT / f"{model}.{stamp}.log"
    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_matrix.py",
        "--stage",
        "primary_v2",
        "--model",
        model,
        "--gpu-uuid",
        GPU_UUID,
        "--context",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--output-dir",
        str(PRIMARY_ROOT / model),
    ]
    if resume:
        command.append("--resume")
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_handle.close()
    return process


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
        marker = f"{label} changed while being read: "
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
    """Refresh progress, with a bounded outer retry only for the exact race."""

    race_messages: list[str] = []
    for attempt in range(1, PROGRESS_OUTER_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                list(PROGRESS_COMMAND),
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=PROGRESS_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {
                "exit_code": None,
                "stdout": "",
                "stderr": (
                    "canonical progress updater timed out after "
                    f"{PROGRESS_SUBPROCESS_TIMEOUT_SECONDS:g} seconds"
                ),
                "outer_attempts": attempt,
                "observed_unstable_source_races": len(race_messages),
                "last_observed_unstable_source_race": (
                    race_messages[-1] if race_messages else None
                ),
                "recovered_unstable_source_races": 0,
                "last_recovered_unstable_source_race": None,
                "deferred_progress_refresh": True,
                "deferred_reason": "updater_timeout",
                "deferred_unstable_source_race": False,
            }
        source_race = canonical_progress_source_race(completed)
        if source_race is not None:
            race_messages.append(source_race)
        succeeded = completed.returncode == 0
        result = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip()[-4000:],
            "stderr": completed.stderr.strip()[-4000:],
            "outer_attempts": attempt,
            "observed_unstable_source_races": len(race_messages),
            "last_observed_unstable_source_race": (
                race_messages[-1] if race_messages else None
            ),
            "recovered_unstable_source_races": (
                len(race_messages) if succeeded else 0
            ),
            "last_recovered_unstable_source_race": (
                race_messages[-1] if succeeded and race_messages else None
            ),
            "deferred_progress_refresh": False,
            "deferred_reason": None,
            "deferred_unstable_source_race": False,
        }
        if succeeded or source_race is None:
            return result
        if attempt < PROGRESS_OUTER_ATTEMPTS:
            time.sleep(PROGRESS_RACE_BACKOFF_SECONDS[attempt - 1])
            continue
        result["deferred_progress_refresh"] = True
        result["deferred_reason"] = "unstable_source_race"
        result["deferred_unstable_source_race"] = True
        return result
    raise AssertionError("unreachable progress refresh loop")


def supervisor_state(
    *,
    status: str,
    current_model: str | None,
    retries: dict[str, int],
    active: dict[int, list[str]],
    budget: dict[str, float],
    progress: dict[str, Any],
    progress_recovery: dict[str, Any] | None = None,
    message: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "rankcloak-primary-v2-supervisor-v1",
        "generated_at": utc_now(),
        "status": status,
        "message": message,
        "current_model": current_model,
        "model_order": list(MODELS),
        "active_runner_pids": sorted(active),
        "retry_counts": dict(retries),
        "budget": budget,
        "progress_update": progress,
        "progress_refresh_recovery": progress_recovery or {
            "successful_outer_recovery_events": 0,
            "unstable_source_races_recovered": 0,
            "last_successful_outer_recovery": None,
            "deferred_unstable_source_race_events": 0,
            "last_deferred_unstable_source_race": None,
            "deferred_updater_timeout_events": 0,
            "last_deferred_updater_timeout": None,
        },
        "authorized_projection_sha256": PROJECTION_SELF_SHA256,
        "gpu_uuid": GPU_UUID,
    }
    value["supervisor_state_sha256"] = canonical_json_sha256(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries-per-model", type=int, default=5)
    parser.add_argument("--attach", action="store_true", help="Require an already-running first shard")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.poll_seconds > 300:
        raise SupervisorError("--poll-seconds must be between 5 and 300")
    if args.max_retries_per_model < 1:
        raise SupervisorError("--max-retries-per-model must be positive")

    projection = verify_projection()
    retries = {model: 0 for model in MODELS}
    fresh_launch_attempted: set[str] = set()
    initial_active = verify_active_processes(MODELS[0] if args.attach else None)
    if args.attach and not initial_active:
        raise SupervisorError("--attach requested but no live Llama primary_v2 process exists")
    if args.dry_run:
        budget = revised_upper_hours(projection)
        print(json.dumps({"status": "dry_run_ok", "active": sorted(initial_active), "budget": budget}, sort_keys=True))
        return 0

    while True:
        projection = verify_projection()
        states = {model: checkpoint_state(model) for model in MODELS}
        incomplete_models = [model for model in MODELS if not states[model] or not states[model]["complete"]]
        current_model = incomplete_models[0] if incomplete_models else None
        active = verify_active_processes(None)
        verify_gpu_occupancy(active)
        transitional_model: str | None = None
        if active:
            active_args = next(iter(active.values()))
            active_model = arg_value(active_args, "--model")
            if active_model != current_model:
                active_state = states.get(str(active_model))
                if not active_state or not active_state["complete"]:
                    raise SupervisorError(
                        f"active model {active_model!r} does not match next incomplete model {current_model!r}"
                    )
                transitional_model = str(active_model)
        budget = revised_upper_hours(projection)
        progress = refresh_progress()
        recovered_races = int(progress.get("recovered_unstable_source_races", 0))
        if recovered_races:
            append_error(
                {
                    "at": utc_now(),
                    "model_id": current_model,
                    "event": "canonical_progress_unstable_source_race_recovered",
                    "source_races_recovered": recovered_races,
                    "last_source_race": progress.get(
                        "last_recovered_unstable_source_race"
                    ),
                    "progress_refresh_outer_attempts": progress["outer_attempts"],
                    "progress_refresh_backoff_seconds": list(
                        PROGRESS_RACE_BACKOFF_SECONDS[
                            : max(0, int(progress["outer_attempts"]) - 1)
                        ]
                    ),
                    "model_retry_count_unchanged": (
                        retries[current_model] if current_model is not None else None
                    ),
                    "active_runner_pids_left_unchanged": sorted(active),
                    "automatic_resume": True,
                }
            )
        progress_recovery = progress_refresh_recovery_summary()
        progress_deferred = bool(
            progress.get("deferred_progress_refresh")
            or progress.get("deferred_unstable_source_race")
        )
        if progress["exit_code"] != 0 and not progress_deferred:
            raise SupervisorError(f"progress refresh failed: {progress}")
        if budget["revised_upper_gpu_hours"] > 165.0:
            terminate_active(active)
            state = supervisor_state(
                status="halted_budget_ceiling",
                current_model=current_model,
                retries=retries,
                active=active,
                budget=budget,
                progress=progress,
                progress_recovery=progress_recovery,
                message="Revised conservative projection exceeds 165 GPU-hours; active runner terminated.",
            )
            atomic_write_json(PROJECT_ROOT / STATE_PATH, state)
            return 3
        if progress_deferred:
            deferred_reason = str(
                progress.get("deferred_reason") or "unstable_source_race"
            )
            timeout_deferred = deferred_reason == "updater_timeout"
            append_error(
                {
                    "at": utc_now(),
                    "model_id": current_model,
                    "event": (
                        "canonical_progress_updater_timeout_deferred"
                        if timeout_deferred
                        else "canonical_progress_unstable_source_race_deferred"
                    ),
                    "detail": progress.get("stderr"),
                    "progress_refresh_outer_attempts": progress["outer_attempts"],
                    "progress_refresh_backoff_seconds": list(
                        PROGRESS_RACE_BACKOFF_SECONDS
                    ),
                    "model_retry_count_unchanged": (
                        retries[current_model] if current_model is not None else None
                    ),
                    "active_runner_pids_left_unchanged": sorted(active),
                    "automatic_resume": True,
                }
            )
            progress_recovery = progress_refresh_recovery_summary()
            state = supervisor_state(
                status=(
                    "progress_updater_timeout_deferred"
                    if timeout_deferred
                    else "progress_snapshot_race_deferred"
                ),
                current_model=current_model,
                retries=retries,
                active=active,
                budget=budget,
                progress=progress,
                progress_recovery=progress_recovery,
                message=(
                    "Canonical progress updater timed out; runner state is "
                    "unchanged and monitoring will resume automatically."
                    if timeout_deferred
                    else (
                        "Canonical progress sources remained unstable after three "
                        "outer attempts; runner state is unchanged and monitoring "
                        "will resume automatically."
                    )
                ),
            )
            atomic_write_json(PROJECT_ROOT / STATE_PATH, state)
            time.sleep(args.poll_seconds)
            continue
        if current_model is None and not active:
            state = supervisor_state(
                status="primary_v2_complete",
                current_model=None,
                retries=retries,
                active=active,
                budget=budget,
                progress=progress,
                progress_recovery=progress_recovery,
                message="All three primary_v2 checkpoints are exact and complete.",
            )
            atomic_write_json(PROJECT_ROOT / STATE_PATH, state)
            return 0
        if active:
            state = supervisor_state(
                status="finalizing_completed_shard" if transitional_model else "running",
                current_model=transitional_model or current_model,
                retries=retries,
                active=active,
                budget=budget,
                progress=progress,
                progress_recovery=progress_recovery,
                message=(
                    "Waiting for the completed shard process to finalize before advancing."
                    if transitional_model
                    else "Attached to the active frozen primary_v2 runner."
                ),
            )
            atomic_write_json(PROJECT_ROOT / STATE_PATH, state)
            time.sleep(args.poll_seconds)
            continue

        current_state = states[current_model]
        resume = current_state is not None
        if current_state is None:
            if current_model in fresh_launch_attempted:
                retries[current_model] += 1
                append_error(
                    {
                        "at": utc_now(),
                        "model_id": current_model,
                        "event": "runner_exited_before_checkpoint_creation",
                        "failed_count": 0,
                        "completed_count": 0,
                        "retry_index": retries[current_model],
                    }
                )
            else:
                fresh_launch_attempted.add(current_model)
        elif current_state["failed"]:
            retries[current_model] += 1
            append_error(
                {
                    "at": utc_now(),
                    "model_id": current_model,
                    "event": "recoverable_failed_ids_observed",
                    "failed_count": current_state["failed"],
                    "completed_count": current_state["completed"],
                    "retry_index": retries[current_model],
                }
            )
        elif current_state is not None:
            retries[current_model] += 1
            append_error(
                {
                    "at": utc_now(),
                    "model_id": current_model,
                    "event": "runner_disappeared_before_checkpoint_completion",
                    "failed_count": 0,
                    "completed_count": current_state["completed"],
                    "retry_index": retries[current_model],
                }
            )
        if retries[current_model] > args.max_retries_per_model:
            raise SupervisorError(f"retry ceiling exceeded for {current_model}")
        process = launch(current_model, resume=resume)
        state = supervisor_state(
            status="launched_resume" if resume else "launched_fresh",
            current_model=current_model,
            retries=retries,
            active={process.pid: []},
            budget=budget,
            progress=progress,
            progress_recovery=progress_recovery,
            message=f"Launched {current_model} with resume={resume}.",
        )
        atomic_write_json(PROJECT_ROOT / STATE_PATH, state)
        time.sleep(max(10.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupervisorError as exc:
        print(f"primary_v2 supervisor failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
