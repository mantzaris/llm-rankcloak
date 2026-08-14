"""Isolated safety tests for the live primary-v2 supervisor.

Every filesystem fixture is rooted under ``tmp_path`` and every process launch
is mocked.  These tests must never inspect or mutate a live result shard.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = PROJECT_ROOT / "scripts" / "supervise_primary_v2.py"


@pytest.fixture
def supervisor():
    spec = importlib.util.spec_from_file_location(
        "rankcloak_test_primary_v2_supervisor", SUPERVISOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _projection(supervisor, *, upper_gpu_hours: float = 158.0) -> dict:
    value = {
        "budget_gpu_hours": 165.0,
        "decision": {"go": True, "status": "go_within_budget"},
        "totals": {"upper_gpu_hours": upper_gpu_hours},
        "projection_rows": [],
    }
    value["projection_sha256"] = supervisor.canonical_json_sha256(value)
    return value


def _runner_args(supervisor, model: str) -> list[str]:
    return [
        str(supervisor.PROJECT_ROOT / ".venv/bin/python"),
        "scripts/run_revision_matrix.py",
        "--stage",
        "primary_v2",
        "--model",
        model,
        "--gpu-uuid",
        supervisor.GPU_UUID,
        "--context",
        "4096",
        "--n-gpu-layers",
        "-1",
        "--output-dir",
        str(supervisor.PRIMARY_ROOT / model),
    ]


def _install_shard(
    supervisor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    completed: list[str] | None = None,
) -> tuple[str, Path, list[str]]:
    model = "fixture_model"
    planned = ["trial-a", "trial-b"]
    completed = list(planned if completed is None else completed)
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "PRIMARY_ROOT", Path("primary"))
    monkeypatch.setattr(supervisor, "PLANNED_PER_MODEL", len(planned))
    monkeypatch.setattr(supervisor, "CONFIG_SHA256", "c" * 64)
    shard = tmp_path / "primary" / model
    shard.mkdir(parents=True)

    planned_hash = supervisor.canonical_json_sha256(planned)
    identity = {
        "schema_version": "1.0",
        "manifest_type": "revision_run_identity",
        "study_id": f"rankcloak_scientific_reports_revision_v1/primary_v2/{model}",
        "config_manifest_sha256": supervisor.CONFIG_SHA256,
        "payload_manifest_sha256": "p" * 64,
        "planned_trial_count": len(planned),
        "planned_trial_ids_sha256": planned_hash,
        "model_artifacts": [{"configured_model": {"model_id": model}}],
        "command_line_args": [
            "stage=primary_v2",
            f"model_id={model}",
            f"protocol_contract_revision={supervisor.PROTOCOL_REVISION}",
            f"result_schema_revision={supervisor.RESULT_REVISION}",
        ],
        "protocol_contract_revision": supervisor.PROTOCOL_REVISION,
        "result_schema_revision": supervisor.RESULT_REVISION,
    }
    identity["run_identity_sha256"] = supervisor.canonical_json_sha256(identity)
    checkpoint = {
        "schema_version": "1.0",
        "study_id": identity["study_id"],
        "config_manifest_sha256": supervisor.CONFIG_SHA256,
        "planned_trial_count": len(planned),
        "planned_trial_ids_sha256": planned_hash,
        "completed_trial_ids": completed,
        "failed_trial_ids": [],
        "failure_details": {},
        "attempt_counts": {trial_id: 1 for trial_id in completed},
        "created_at": "2026-08-09T00:00:00+00:00",
        "updated_at": "2026-08-09T00:01:00+00:00",
    }
    _write_json(shard / "run_identity.json", identity)
    _write_json(shard / "checkpoint.json", checkpoint)
    (shard / "plan.jsonl").write_text(
        "".join(json.dumps({"work_id": trial_id}) + "\n" for trial_id in planned),
        encoding="utf-8",
    )
    (shard / "records.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "record_type": "rankcloak_trial",
                    "execution_status": "completed",
                    "work_id": trial_id,
                    "protocol_variant": "fixture_protocol",
                }
            )
            + "\n"
            for trial_id in completed
            if trial_id in planned
        ),
        encoding="utf-8",
    )
    return model, shard, planned


def test_checked_in_projection_matches_both_self_hashes(supervisor):
    value = supervisor.verify_projection()
    unsigned = {key: item for key, item in value.items() if key != "projection_sha256"}
    observed = supervisor.canonical_json_sha256(unsigned)
    assert observed == value["projection_sha256"] == supervisor.PROJECTION_SELF_SHA256


def test_projection_tampering_fails_even_when_embedded_hash_is_recomputed(
    supervisor, monkeypatch, tmp_path
):
    value = _projection(supervisor)
    pinned = value["projection_sha256"]
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "PROJECTION_PATH", Path("projection.json"))
    monkeypatch.setattr(supervisor, "PROJECTION_SELF_SHA256", pinned)
    _write_json(tmp_path / "projection.json", value)
    assert supervisor.verify_projection()["projection_sha256"] == pinned

    value["totals"]["upper_gpu_hours"] = 157.0
    unsigned = {key: item for key, item in value.items() if key != "projection_sha256"}
    value["projection_sha256"] = supervisor.canonical_json_sha256(unsigned)
    _write_json(tmp_path / "projection.json", value)
    with pytest.raises(supervisor.SupervisorError, match="self-hash mismatch"):
        supervisor.verify_projection()


def test_exact_frozen_runner_command_is_accepted(supervisor, monkeypatch):
    model = supervisor.MODELS[0]
    monkeypatch.setattr(
        supervisor, "proc_primary_commands", lambda: {1234: _runner_args(supervisor, model)}
    )
    assert list(supervisor.verify_active_processes(model)) == [1234]


def test_gpu_occupancy_query_has_a_finite_timeout(supervisor, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    supervisor.verify_gpu_occupancy({})
    assert 0 < calls[0][1]["timeout"] <= 60


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--threads", "1"],
        ["--config-dir", "unfrozen-configs"],
        ["--model", "qwen2_5_7b_instruct_q4_k_m"],
    ],
)
def test_active_runner_rejects_unfrozen_or_duplicate_overrides(
    supervisor, monkeypatch, extra_args
):
    model = supervisor.MODELS[0]
    args = _runner_args(supervisor, model) + extra_args
    monkeypatch.setattr(supervisor, "proc_primary_commands", lambda: {1234: args})
    with pytest.raises(supervisor.SupervisorError):
        supervisor.verify_active_processes(model)


def test_launch_uses_exact_frozen_command_and_resume(supervisor, monkeypatch, tmp_path):
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        assert not kwargs["stdout"].closed
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "PRIMARY_ROOT", Path("primary"))
    monkeypatch.setattr(supervisor, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    process = supervisor.launch("fixture_model", resume=True)

    assert process.pid == 4321
    command, kwargs = calls[0]
    assert command == _runner_args(supervisor, "fixture_model") + ["--resume"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["start_new_session"] is True
    assert kwargs["stderr"] == supervisor.subprocess.STDOUT


def test_checkpoint_accepts_a_valid_complete_fixture(
    supervisor, monkeypatch, tmp_path
):
    model, _, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    state = supervisor.checkpoint_state(model)
    assert state is not None
    assert state["completed"] == 2
    assert state["failed"] == 0
    assert state["complete"] is True


def test_checkpoint_rejects_completion_ids_not_bound_to_frozen_plan(
    supervisor, monkeypatch, tmp_path
):
    model, _, _ = _install_shard(
        supervisor, monkeypatch, tmp_path, completed=["forged-a", "forged-b"]
    )
    with pytest.raises(supervisor.SupervisorError):
        supervisor.checkpoint_state(model)


def test_checkpoint_rejects_tampered_run_identity_self_hash(
    supervisor, monkeypatch, tmp_path
):
    model, shard, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    identity = json.loads((shard / "run_identity.json").read_text(encoding="utf-8"))
    identity["payload_manifest_sha256"] = "tampered"
    _write_json(shard / "run_identity.json", identity)
    with pytest.raises(supervisor.SupervisorError, match="identity|hash"):
        supervisor.checkpoint_state(model)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("identity", "config_manifest_sha256", "d" * 64),
        (
            "identity",
            "study_id",
            "rankcloak_scientific_reports_revision_v1/primary_v2/other_model",
        ),
        (
            "checkpoint",
            "study_id",
            "rankcloak_scientific_reports_revision_v1/primary_v2/other_model",
        ),
    ],
)
def test_checkpoint_and_identity_are_bound_to_config_stage_and_model(
    supervisor, monkeypatch, tmp_path, target, field, value
):
    model, shard, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    path = shard / f"{target}.json" if target == "checkpoint" else shard / "run_identity.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    if target == "identity":
        payload.pop("run_identity_sha256")
        payload["run_identity_sha256"] = supervisor.canonical_json_sha256(payload)
    _write_json(path, payload)
    with pytest.raises(supervisor.SupervisorError):
        supervisor.checkpoint_state(model)


def test_checkpoint_rejects_unknown_or_noncompleted_durable_rows(
    supervisor, monkeypatch, tmp_path
):
    model, shard, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    with (shard / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "rankcloak_trial",
                    "execution_status": "failed",
                    "work_id": "not-in-plan",
                    "protocol_variant": "fixture_protocol",
                }
            )
            + "\n"
        )
    with pytest.raises(supervisor.SupervisorError):
        supervisor.checkpoint_state(model)


def test_budget_replaces_completed_upper_rates_with_measured_elapsed_time(
    supervisor, monkeypatch, tmp_path
):
    model, shard, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "MODELS", (model,))
    (shard / "records.jsonl").write_text(
        json.dumps(
            {
                "record_type": "rankcloak_trial",
                "execution_status": "completed",
                "work_id": "trial-a",
                "protocol_variant": "fixture_protocol",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    projection = {
        "totals": {"upper_gpu_hours": 100.0},
        "projection_rows": [
            {
                "stage": "primary_v2",
                "model_id": model,
                "model_load": {"upper_seconds_per_unit": 20.0},
                "strata": [
                    {
                        "stratum": "rankcloak:fixture_protocol",
                        "upper_seconds_per_unit": 10.0,
                    }
                ],
            }
        ],
    }
    budget = supervisor.revised_upper_hours(projection)
    assert budget["projected_upper_consumed_gpu_hours"] == pytest.approx(30.0 / 3600.0)
    assert budget["actual_primary_gpu_hours"] == pytest.approx(60.0 / 3600.0)
    assert budget["revised_upper_gpu_hours"] == pytest.approx(100.0 + 30.0 / 3600.0)


def test_recoverable_failure_rows_do_not_break_budget_or_consume_planned_upper_rate(
    supervisor, monkeypatch, tmp_path
):
    model, shard, _ = _install_shard(supervisor, monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor, "MODELS", (model,))
    rows = [
        {
            "record_type": "execution_failure",
            "execution_status": "failed",
            "work_id": "trial-a",
            "attempt_index": 1,
        },
        {
            "record_type": "rankcloak_trial",
            "execution_status": "completed",
            "work_id": "trial-a",
            "attempt_index": 2,
            "protocol_variant": "fixture_protocol",
        },
    ]
    (shard / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    projection = {
        "totals": {"upper_gpu_hours": 100.0},
        "projection_rows": [
            {
                "stage": "primary_v2",
                "model_id": model,
                "model_load": {"upper_seconds_per_unit": 20.0},
                "strata": [
                    {
                        "stratum": "rankcloak:fixture_protocol",
                        "upper_seconds_per_unit": 10.0,
                    }
                ],
            }
        ],
    }
    budget = supervisor.revised_upper_hours(projection)
    assert budget["projected_upper_consumed_gpu_hours"] == pytest.approx(30.0 / 3600.0)


def test_precheckpoint_process_exits_consume_the_retry_allowance(
    supervisor, monkeypatch
):
    launches = []

    def fake_launch(model, resume):
        launches.append((model, resume))
        if len(launches) > 2:
            raise AssertionError("third launch bypassed the one-retry ceiling")
        return SimpleNamespace(pid=5000 + len(launches))

    monkeypatch.setattr(supervisor.sys, "argv", ["supervisor", "--poll-seconds", "5", "--max-retries-per-model", "1"])
    monkeypatch.setattr(supervisor, "verify_projection", lambda: {})
    monkeypatch.setattr(supervisor, "verify_active_processes", lambda expected: {})
    monkeypatch.setattr(supervisor, "verify_gpu_occupancy", lambda active: None)
    monkeypatch.setattr(supervisor, "checkpoint_state", lambda model: None)
    monkeypatch.setattr(
        supervisor,
        "revised_upper_hours",
        lambda projection: {"revised_upper_gpu_hours": 1.0},
    )
    monkeypatch.setattr(supervisor, "refresh_progress", lambda: {"exit_code": 0})
    monkeypatch.setattr(supervisor, "atomic_write_json", lambda path, value: None)
    monkeypatch.setattr(supervisor, "append_error", lambda value: None)
    monkeypatch.setattr(supervisor, "launch", fake_launch)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)

    with pytest.raises(supervisor.SupervisorError, match="retry ceiling"):
        supervisor.main()
    assert launches == [
        (supervisor.MODELS[0], False),
        (supervisor.MODELS[0], False),
    ]


def test_supervisor_state_self_hash_covers_every_unsigned_field(
    supervisor, monkeypatch
):
    monkeypatch.setattr(supervisor, "utc_now", lambda: "2026-08-09T00:00:00+00:00")
    state = supervisor.supervisor_state(
        status="running",
        current_model=supervisor.MODELS[0],
        retries={model: 0 for model in supervisor.MODELS},
        active={1234: []},
        budget={"revised_upper_gpu_hours": 158.0},
        progress={"exit_code": 0, "stdout": "{}", "stderr": ""},
        message="fixture",
    )
    digest = state.pop("supervisor_state_sha256")
    assert digest == supervisor.canonical_json_sha256(state)
    state["message"] = "tampered"
    assert digest != supervisor.canonical_json_sha256(state)


def _progress_result(supervisor, *, race: bool = False, code: int = 0):
    return SimpleNamespace(
        returncode=code,
        stdout="" if race else "{}\n",
        stderr=(supervisor.PROGRESS_SOURCE_RACE_MESSAGE + "\n") if race else "",
    )


def test_progress_refresh_uses_three_outer_attempts_with_two_four_backoff(
    supervisor, monkeypatch
):
    responses = [
        _progress_result(supervisor, race=True, code=1),
        _progress_result(supervisor, race=True, code=1),
        _progress_result(supervisor),
    ]
    sleeps = []
    monkeypatch.setattr(
        supervisor.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0)
    )
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)

    result = supervisor.refresh_progress()

    assert result["exit_code"] == 0
    assert result["outer_attempts"] == 3
    assert result["recovered_unstable_source_races"] == 2
    assert result["deferred_unstable_source_race"] is False
    assert sleeps == [2.0, 4.0]


def test_exact_progress_race_accepts_all_build_labels_only_for_normalized_in_root_paths(
    supervisor, monkeypatch, tmp_path
):
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / supervisor.RESULTS_ROOT
    results_root.mkdir(parents=True)
    source = results_root / "primary_v2/model/checkpoint.json"
    for label in supervisor.PROGRESS_SOURCE_RACE_LABELS:
        line = (
            supervisor.PROGRESS_SOURCE_RACE_WRAPPER
            + f"{label} changed while being read: {source}"
        )
        completed = SimpleNamespace(returncode=1, stdout="", stderr=line + "\n")
        assert supervisor.canonical_progress_source_race(completed) == line


def test_exact_progress_race_accepts_both_fixed_canonical_inners(supervisor):
    for inner in supervisor.PROGRESS_SOURCE_RACE_FIXED_INNERS:
        line = supervisor.PROGRESS_SOURCE_RACE_WRAPPER + inner
        completed = SimpleNamespace(returncode=1, stdout="", stderr=line + "\r\n")
        assert supervisor.canonical_progress_source_race(completed) == line


@pytest.mark.parametrize(
    "inner",
    [
        "unknown label changed while being read: {inside}",
        "checkpoint changed while being read: relative/checkpoint.json",
        "checkpoint changed while being read: {outside}",
        "checkpoint changed while being read: {inside}/../outside.json",
        "checkpoint changed while being read: {inside}\nextra",
    ],
)
def test_path_progress_race_rejects_unknown_relative_outside_or_unnormalized_paths(
    supervisor, monkeypatch, tmp_path, inner
):
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / supervisor.RESULTS_ROOT
    results_root.mkdir(parents=True)
    inside = results_root / "checkpoint.json"
    outside = tmp_path / "outside.json"
    line = supervisor.PROGRESS_SOURCE_RACE_WRAPPER + inner.format(
        inside=inside, outside=outside
    )
    completed = SimpleNamespace(returncode=1, stdout="", stderr=line + "\n")
    assert supervisor.canonical_progress_source_race(completed) is None


def test_path_progress_race_rejects_symlink_resolution_outside_results(
    supervisor, monkeypatch, tmp_path
):
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / supervisor.RESULTS_ROOT
    results_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = results_root / "link"
    link.symlink_to(outside, target_is_directory=True)
    source = link / "checkpoint.json"
    line = (
        supervisor.PROGRESS_SOURCE_RACE_WRAPPER
        + f"checkpoint changed while being read: {source}"
    )
    completed = SimpleNamespace(returncode=1, stdout="", stderr=line + "\n")
    assert supervisor.canonical_progress_source_race(completed) is None


def test_progress_refresh_defers_after_three_exact_races(supervisor, monkeypatch):
    calls = []
    sleeps = []

    def run(*_args, **_kwargs):
        calls.append(True)
        return _progress_result(supervisor, race=True, code=1)

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)

    result = supervisor.refresh_progress()

    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]
    assert result["exit_code"] == 1
    assert result["outer_attempts"] == 3
    assert result["deferred_unstable_source_race"] is True


def test_progress_refresh_timeout_is_finite_and_explicitly_deferred(
    supervisor, monkeypatch
):
    observed = []

    def run(*args, **kwargs):
        observed.append(kwargs["timeout"])
        raise supervisor.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    result = supervisor.refresh_progress()

    assert observed == [supervisor.PROGRESS_SUBPROCESS_TIMEOUT_SECONDS]
    assert result["deferred_progress_refresh"] is True
    assert result["deferred_reason"] == "updater_timeout"
    assert result["deferred_unstable_source_race"] is False
    assert result["recovered_unstable_source_races"] == 0


@pytest.mark.parametrize(
    ("code", "stdout", "stderr"),
    [
        (2, "", "{canonical}\n"),
        (1, "{}\n", "{canonical}\n"),
        (1, "", "{canonical}"),
        (1, "", "{canonical}\n\n"),
        (1, "", "{canonical} extra\n"),
        (
            1,
            "",
            "revision progress failed: Progress sources remained unstable: "
            "checkpoint changed while being read: /tmp/checkpoint.json\n",
        ),
    ],
)
def test_progress_refresh_fails_closed_on_every_near_match(
    supervisor, monkeypatch, code, stdout, stderr
):
    completed = SimpleNamespace(
        returncode=code,
        stdout=stdout,
        stderr=stderr.format(canonical=supervisor.PROGRESS_SOURCE_RACE_MESSAGE),
    )
    sleeps = []
    monkeypatch.setattr(
        supervisor.subprocess, "run", lambda *_args, **_kwargs: completed
    )
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)

    result = supervisor.refresh_progress()

    assert result["exit_code"] == code
    assert result["outer_attempts"] == 1
    assert result["deferred_unstable_source_race"] is False
    assert sleeps == []


def test_deferred_progress_race_persists_without_retry_launch_or_kill(
    supervisor, monkeypatch
):
    class StopAfterDeferredState(RuntimeError):
        pass

    states = []
    errors = []
    monkeypatch.setattr(
        supervisor.sys,
        "argv",
        ["supervisor", "--poll-seconds", "5", "--max-retries-per-model", "1"],
    )
    monkeypatch.setattr(supervisor, "verify_projection", lambda: {})
    monkeypatch.setattr(supervisor, "verify_active_processes", lambda _expected: {})
    monkeypatch.setattr(supervisor, "verify_gpu_occupancy", lambda _active: None)
    monkeypatch.setattr(supervisor, "checkpoint_state", lambda _model: None)
    monkeypatch.setattr(
        supervisor,
        "revised_upper_hours",
        lambda _projection: {"revised_upper_gpu_hours": 158.0},
    )
    monkeypatch.setattr(
        supervisor,
        "refresh_progress",
        lambda: {
            "exit_code": 1,
            "outer_attempts": 3,
            "deferred_unstable_source_race": True,
        },
    )
    monkeypatch.setattr(supervisor, "append_error", errors.append)
    monkeypatch.setattr(
        supervisor,
        "atomic_write_json",
        lambda _path, value: states.append(value),
    )
    monkeypatch.setattr(
        supervisor,
        "launch",
        lambda *_args, **_kwargs: pytest.fail("progress race launched a runner"),
    )
    monkeypatch.setattr(
        supervisor,
        "terminate_active",
        lambda _active: pytest.fail("progress race killed a runner"),
    )
    monkeypatch.setattr(
        supervisor.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopAfterDeferredState()),
    )

    with pytest.raises(StopAfterDeferredState):
        supervisor.main()

    assert states[-1]["status"] == "progress_snapshot_race_deferred"
    assert set(states[-1]["retry_counts"].values()) == {0}
    assert errors[-1]["event"] == "canonical_progress_unstable_source_race_deferred"
    assert errors[-1]["model_retry_count_unchanged"] == 0
    assert errors[-1]["active_runner_pids_left_unchanged"] == []
    assert len(errors) == 1


def test_successful_outer_recovery_is_cumulative_and_persistent_without_model_retry(
    supervisor, monkeypatch, tmp_path
):
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor.sys, "argv", ["supervisor", "--poll-seconds", "5"])
    monkeypatch.setattr(supervisor, "verify_projection", lambda: {})
    monkeypatch.setattr(supervisor, "verify_active_processes", lambda _expected: {})
    monkeypatch.setattr(supervisor, "verify_gpu_occupancy", lambda _active: None)
    monkeypatch.setattr(
        supervisor, "checkpoint_state", lambda _model: {"complete": True}
    )
    monkeypatch.setattr(
        supervisor,
        "revised_upper_hours",
        lambda _projection: {"revised_upper_gpu_hours": 158.0},
    )
    supervisor.append_error(
        {
            "at": "2026-08-10T00:00:00+00:00",
            "event": "canonical_progress_unstable_source_race_recovered",
            "source_races_recovered": 2,
            "last_source_race": "prior",
        }
    )
    monkeypatch.setattr(
        supervisor,
        "refresh_progress",
        lambda: {
            "exit_code": 0,
            "outer_attempts": 2,
            "recovered_unstable_source_races": 1,
            "last_recovered_unstable_source_race": supervisor.PROGRESS_SOURCE_RACE_MESSAGE,
            "deferred_progress_refresh": False,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "launch",
        lambda *_args, **_kwargs: pytest.fail("successful refresh recovery launched a runner"),
    )
    monkeypatch.setattr(
        supervisor,
        "terminate_active",
        lambda _active: pytest.fail("successful refresh recovery killed a runner"),
    )

    assert supervisor.main() == 0

    state = json.loads((tmp_path / supervisor.STATE_PATH).read_text(encoding="utf-8"))
    summary = state["progress_refresh_recovery"]
    assert summary["successful_outer_recovery_events"] == 2
    assert summary["unstable_source_races_recovered"] == 3
    assert summary["last_successful_outer_recovery"]["source_races_recovered"] == 1
    assert set(state["retry_counts"].values()) == {0}


def test_progress_updater_timeout_defers_without_retry_launch_or_kill(
    supervisor, monkeypatch
):
    class StopAfterTimeoutState(RuntimeError):
        pass

    states = []
    errors = []
    monkeypatch.setattr(supervisor.sys, "argv", ["supervisor", "--poll-seconds", "5"])
    monkeypatch.setattr(supervisor, "verify_projection", lambda: {})
    monkeypatch.setattr(supervisor, "verify_active_processes", lambda _expected: {})
    monkeypatch.setattr(supervisor, "verify_gpu_occupancy", lambda _active: None)
    monkeypatch.setattr(supervisor, "checkpoint_state", lambda _model: None)
    monkeypatch.setattr(
        supervisor,
        "revised_upper_hours",
        lambda _projection: {"revised_upper_gpu_hours": 158.0},
    )
    monkeypatch.setattr(
        supervisor,
        "refresh_progress",
        lambda: {
            "exit_code": None,
            "outer_attempts": 1,
            "recovered_unstable_source_races": 0,
            "deferred_progress_refresh": True,
            "deferred_reason": "updater_timeout",
            "stderr": "fixture timeout",
        },
    )
    monkeypatch.setattr(supervisor, "append_error", errors.append)
    monkeypatch.setattr(
        supervisor, "atomic_write_json", lambda _path, value: states.append(value)
    )
    monkeypatch.setattr(
        supervisor,
        "launch",
        lambda *_args, **_kwargs: pytest.fail("updater timeout launched a runner"),
    )
    monkeypatch.setattr(
        supervisor,
        "terminate_active",
        lambda _active: pytest.fail("updater timeout killed a runner"),
    )
    monkeypatch.setattr(
        supervisor.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopAfterTimeoutState()),
    )

    with pytest.raises(StopAfterTimeoutState):
        supervisor.main()

    assert len(errors) == 1
    assert errors[0]["event"] == "canonical_progress_updater_timeout_deferred"
    assert states[-1]["status"] == "progress_updater_timeout_deferred"
    assert set(states[-1]["retry_counts"].values()) == {0}


@pytest.mark.parametrize("deferred_reason", ["unstable_source_race", "updater_timeout"])
def test_budget_ceiling_preempts_progress_deferral(
    supervisor, monkeypatch, deferred_reason
):
    states = []
    errors = []
    terminations = []
    monkeypatch.setattr(supervisor.sys, "argv", ["supervisor", "--poll-seconds", "5"])
    monkeypatch.setattr(supervisor, "verify_projection", lambda: {})
    monkeypatch.setattr(supervisor, "verify_active_processes", lambda _expected: {})
    monkeypatch.setattr(supervisor, "verify_gpu_occupancy", lambda _active: None)
    monkeypatch.setattr(supervisor, "checkpoint_state", lambda _model: None)
    monkeypatch.setattr(
        supervisor,
        "revised_upper_hours",
        lambda _projection: {"revised_upper_gpu_hours": 165.01},
    )
    monkeypatch.setattr(
        supervisor,
        "refresh_progress",
        lambda: {
            "exit_code": None if deferred_reason == "updater_timeout" else 1,
            "outer_attempts": (
                1 if deferred_reason == "updater_timeout" else 3
            ),
            "recovered_unstable_source_races": 0,
            "deferred_progress_refresh": True,
            "deferred_reason": deferred_reason,
            "deferred_unstable_source_race": (
                deferred_reason == "unstable_source_race"
            ),
        },
    )
    monkeypatch.setattr(supervisor, "append_error", errors.append)
    monkeypatch.setattr(
        supervisor,
        "atomic_write_json",
        lambda _path, value: states.append(value),
    )
    monkeypatch.setattr(supervisor, "terminate_active", terminations.append)
    monkeypatch.setattr(
        supervisor,
        "launch",
        lambda *_args, **_kwargs: pytest.fail("budget halt launched a runner"),
    )

    assert supervisor.main() == 3
    assert terminations == [{}]
    assert errors == []
    assert states[-1]["status"] == "halted_budget_ceiling"
