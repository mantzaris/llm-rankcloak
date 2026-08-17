from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rankcloak.revision_runner import build_stage_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "supervise_confirmatory_v2.py"
SPEC = importlib.util.spec_from_file_location("confirmatory_v2_orchestrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = orchestrator
SPEC.loader.exec_module(orchestrator)


def _arg(argv: tuple[str, ...], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _detector_gpu_accounting(
    *, live: bool = False, seconds: float = 60.0, pid: int = 4321
) -> dict:
    started = datetime.now(timezone.utc) - timedelta(seconds=seconds + 5)
    ended = started + timedelta(seconds=seconds)
    interval = {
        "pid": pid,
        "process_start_ticks": 987654,
        "device": orchestrator.DETECTOR_DEVICE,
        "gpu_uuid": orchestrator.GPU_UUID,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None if live else ended.isoformat(),
        "last_observed_at_utc": ended.isoformat(),
        "elapsed_seconds": seconds,
        "derivation_policy": orchestrator.DETECTOR_GPU_DERIVATION,
    }
    return {
        "device": orchestrator.DETECTOR_DEVICE,
        "gpu_uuid": orchestrator.GPU_UUID,
        "intervals": [interval],
        "cumulative_elapsed_seconds": seconds,
        "derivation_policy": orchestrator.DETECTOR_GPU_COLLECTION_DERIVATION,
    }


def _detector_action() -> orchestrator.Action:
    values = orchestrator._format_values()
    output = Path(values["detector_output_dir"])
    return orchestrator.Action(
        action_id="downstream:detector",
        stage="neural_detector",
        kind="downstream",
        output_dir=output,
        gpu=True,
        argv=(
            str(orchestrator.PROJECT_ROOT / ".venv/bin/python"),
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
            orchestrator.DETECTOR_DEVICE,
            "--gpu-uuid",
            orchestrator.GPU_UUID,
            "--workers",
            "1",
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
        ),
    )


def _detector_status(action: orchestrator.Action, *, pid: int = 4321) -> dict:
    values = orchestrator._format_values()
    now = datetime.now(timezone.utc)
    run_identity = {
        "device": orchestrator.DETECTOR_DEVICE,
        "gpu_uuid": orchestrator.GPU_UUID,
        "workers": 1,
        "output_dir": str(action.output_dir.resolve()),
        "checkpoint_dir": values["detector_checkpoint_dir"],
        "status_file": values["detector_status_file"],
        "execution_policy_path": values["detector_execution_policy"],
        "execution_policy_sha256": (
            orchestrator.DETECTOR_EXECUTION_POLICY_SHA256
        ),
        "fit_permit_file": values["detector_fit_permit_file"],
        "fit_permit_receipt_dir": values["detector_fit_permit_receipt_dir"],
        "require_fit_permit": True,
    }
    value = {
        "schema_version": orchestrator.DETECTOR_STATUS_SCHEMA,
        "updated_at_utc": now.isoformat(),
        "state": "running_fit",
        "completed_fit_count": 7,
        "total_fit_count": orchestrator.DETECTOR_TOTAL_FITS,
        "current_fit": {
            "ordinal": 7,
            "index": 7,
            "fit_number": 8,
            "split_id": "split-04",
            "regime": "leave_one_model_out",
            "detector_name": "textcnn",
            "detector_kind": "textcnn",
            "seed": 123,
            "task_identity_sha256": "a" * 64,
        },
        "next_fit": None,
        "next_fit_upper_seconds": None,
        "fit_gate_nonce": None,
        "fit_permit_file": values["detector_fit_permit_file"],
        "fit_permit_receipt_dir": values["detector_fit_permit_receipt_dir"],
        "last_consumed_fit_permit": None,
        "global_started_at_utc": (now - timedelta(seconds=60)).isoformat(),
        "global_elapsed_seconds": 57.0,
        "global_elapsed_policy": "sum_of_valid_fit_intervals_plus_active_fit_v1",
        "process_elapsed_seconds": 60.0,
        "checkpoint_fit_seconds_at_process_start": 40.0,
        "checkpoint_cumulative_fit_seconds": 52.0,
        "current_fit_started_at_utc": (now - timedelta(seconds=5)).isoformat(),
        "current_fit_elapsed_seconds": 5.0,
        "fits_per_hour": 420.0,
        "rolling_fits_per_hour": 400.0,
        "rolling_eta_seconds": 441.0,
        "rolling_estimated_completion_utc": (now + timedelta(seconds=441)).isoformat(),
        "last_completed_checkpoint": {"ordinal": 6, "path": "fits/0006/manifest.json"},
        "recovered_errors": [{"type": "fixture_recovery"}],
        "device": orchestrator.DETECTOR_DEVICE,
        "gpu_uuid": orchestrator.GPU_UUID,
        "workers": 1,
        "peak_rss_bytes": 1024,
        "peak_vram_bytes": 2048,
        "pid": pid,
        "process_start_ticks": 987654,
        "run_identity": run_identity,
        "run_identity_sha256": orchestrator.canonical_json_sha256(run_identity),
        "gpu_accounting": _detector_gpu_accounting(live=True, pid=pid),
    }
    value["status_sha256"] = orchestrator.canonical_json_sha256(value)
    return value


def test_detector_equivalence_gate_requires_both_passing_reports_before_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    args = SimpleNamespace(
        poll_seconds=5.0,
        max_retries_per_action=3,
        detector_benchmark_task_index=None,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_detector_equivalence_role",
        lambda _args, _projection, _contract, *, task_index, role: calls.append(
            ("role", task_index, role)
        )
        or 0,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_detector_benchmark",
        lambda benchmark_args, _projection, _contract: calls.append(
            ("benchmark", benchmark_args.detector_benchmark_task_index)
        )
        or 0,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_detector_equivalence_report",
        lambda *, task_index: calls.append(("report", task_index))
        or {"report_sha256": str(task_index) * 64},
    )
    monkeypatch.setattr(
        orchestrator,
        "require_detector_equivalence_gate",
        lambda: calls.append(("require_both",)) or ({}, {}),
    )
    monkeypatch.setattr(
        orchestrator,
        "build_detector_cuda_budget_gate",
        lambda *, stage: calls.append(("build_budget_gate", stage)) or {},
    )
    monkeypatch.setattr(
        orchestrator,
        "require_detector_cuda_budget_gate",
        lambda *, expected_stage: calls.append(
            ("require_budget_gate", expected_stage)
        )
        or {},
    )
    monkeypatch.setattr(orchestrator, "emit_event", lambda *args, **kwargs: None)

    orchestrator.ensure_detector_equivalence_gate(args, {}, {})

    assert calls == [
        ("benchmark", 0),
        ("benchmark", 1),
        (
            "build_budget_gate",
            "post_benchmark_pre_reproducibility",
        ),
        ("role", 0, "cuda"),
        ("role", 0, "cuda_repeat"),
        ("report", 0),
        ("role", 1, "cuda"),
        ("role", 1, "cuda_repeat"),
        ("report", 1),
        ("require_both",),
        (
            "build_budget_gate",
            "post_reproducibility_preproduction",
        ),
        (
            "require_budget_gate",
            "post_reproducibility_preproduction",
        ),
    ]


def test_detector_cuda_budget_gate_accepts_valid_partial_equivalence_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = (tmp_path / "gpu_accounting_ledger.json").resolve()
    ledger_file_sha256 = "d" * 64
    ledger_content_sha256 = "e" * 64
    ledger = {
        "ledger_sha256": ledger_content_sha256,
        "sources": [
            {
                "source_id": "production_benchmark_task_0",
                "component": "detector_production_benchmark",
            },
            {
                "source_id": "production_benchmark_task_1",
                "component": "detector_production_benchmark",
            },
            {
                "source_id": "equivalence_cuda_task_0",
                "component": "detector_device_equivalence_cuda",
            },
        ],
    }
    gate = {
        "inputs": {
            "policy": {
                "path": str(orchestrator._detector_execution_policy_path()),
                "sha256": orchestrator.DETECTOR_EXECUTION_POLICY_SHA256,
                "policy_sha256": (
                    orchestrator.DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
                ),
            },
            "gpu_ledger": {
                "path": str(ledger_path),
                "sha256": ledger_file_sha256,
                "ledger_sha256": ledger_content_sha256,
            },
        },
        "projection": {
            "starting_cumulative_actual_gpu_hours": (
                orchestrator.DETECTOR_HISTORICAL_GPU_HOURS_FLOOR
            ),
            "projected_cumulative_gpu_hours": 100.0,
            "projected_remaining_headroom_gpu_hours": 65.0,
        },
    }
    monkeypatch.setattr(orchestrator, "read_gate", lambda *_args, **_kwargs: gate)
    monkeypatch.setattr(
        orchestrator, "_detector_gpu_ledger_path", lambda: ledger_path
    )
    monkeypatch.setattr(
        orchestrator,
        "read_detector_gpu_accounting_ledger",
        lambda _path: ledger,
    )
    monkeypatch.setattr(
        orchestrator, "file_sha256", lambda _path: ledger_file_sha256
    )

    assert orchestrator.require_detector_cuda_budget_gate(
        expected_stage="post_benchmark_pre_reproducibility"
    ) is gate


def test_detector_equivalence_false_is_methodological_halt_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = orchestrator._format_values()
    report_path = Path(values["detector_equivalence_report_0"])
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "_verify_detector_equivalence_artifact",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_detector_equivalence_report_argv",
        lambda _task: ("python", "report"),
    )
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=4,
            stdout="",
            stderr="methodological halt: equivalent=false",
        ),
    )
    monkeypatch.setattr(orchestrator, "atomic_write_bytes", lambda *args: None)
    with pytest.raises(orchestrator.MethodologicalHalt, match="did not pass"):
        orchestrator.run_detector_equivalence_report(task_index=0)
    assert report_path.name == "cuda_reproducibility_report.json"


def test_cpu_neural_equivalence_role_is_rejected_before_action_construction() -> None:
    contract = orchestrator.load_command_contract(
        PROJECT_ROOT / "operations/confirmatory_v2/downstream_commands.json"
    )
    with pytest.raises(orchestrator.MethodologicalHalt, match="CUDA reproducibility"):
        orchestrator._detector_equivalence_action(
            contract, task_index=0, role="cpu"
        )
    with pytest.raises(SystemExit):
        orchestrator.build_argument_parser().parse_args(
            [
                "--detector-equivalence-task-index",
                "0",
                "--detector-equivalence-role",
                "cpu",
            ]
        )


def test_gpu_dag_is_serial_complete_and_uses_exact_roots() -> None:
    actions = orchestrator.build_gpu_actions()
    runner = [action for action in actions if action.kind == "runner"]
    assert [(action.stage, action.model_id) for action in runner] == list(
        orchestrator.SUPPORT_ORDER
    )
    assert len(actions) == 27
    assert all("--limit" not in action.argv for action in actions)
    assert all("--max-pending" not in action.argv for action in actions)

    robustness = [action for action in runner if action.stage == "robustness_v2"]
    assert [action.model_id for action in robustness] == [
        orchestrator.QWEN,
        "llama3_8b_instruct_q4_k_m",
        "mistral_7b_instruct_v0_3_q4_k_m",
    ]
    for action in robustness:
        assert _arg(action.argv, "--primary-results-root") == (
            "results/revision_v1/primary_v2"
        )
        assert _arg(action.argv, "--ablation-results-root") == (
            "results/revision_v1/ablation_v2"
        )
        assert _arg(action.argv, "--robustness-results-root") == (
            "results/revision_v1/robustness_v2"
        )

    evaluator = [action for action in actions if action.kind == "evaluator"]
    exports = [action for action in actions if action.kind == "evaluator_export"]
    assert len(evaluator) == len(exports) == 9
    assert {action.source_stage for action in evaluator} == set(
        orchestrator.EVALUATOR_SOURCE_STAGES
    )
    assert sum(int(action.expected_count) for action in evaluator) == 17232
    mistral_ablation = next(
        action
        for action in evaluator
        if action.source_stage == "ablation_v2"
        and action.generator_model_id == "mistral_7b_instruct_v0_3_q4_k_m"
    )
    assert mistral_ablation.expected_count == 528
    for score, export in zip(evaluator, exports):
        assert export.action_id == score.action_id + ":export"
        assert export.argv == score.argv + ("--resume",)
        assert _arg(score.argv, "--output-dir") == (
            "results/revision_v1/heldout_evaluator/{}/{}".format(
                score.source_stage, score.model_id
            )
        )


def test_budget_adds_stage_isolated_evaluator_model_loads() -> None:
    projection = orchestrator.verify_projection()
    rows = [
        row
        for row in projection["projection_rows"]
        if row["stage"] == "heldout_evaluator"
    ]
    expected = sum(
        2.0 * float(row["model_load"]["upper_seconds_per_unit"])
        for row in rows
    )
    assert orchestrator.operational_projection_adjustment_seconds(projection) == pytest.approx(
        expected
    )
    assert expected > 0


@pytest.mark.parametrize(
    ("replay_mode", "projected_replay_mode"),
    [
        ("canonicalized_text_retokenized", "text_retokenized"),
        ("cross_model_text_retokenized", "text_retokenized"),
        ("detokenized_text_retokenized", "text_retokenized"),
        ("transformed_text_retokenized", "text_retokenized"),
        ("greedy_leadin_regeneration", "greedy_leadin_regeneration"),
    ],
)
def test_robustness_replay_modes_map_to_authorized_projection_strata(
    replay_mode: str, projected_replay_mode: str
) -> None:
    task = {
        "work_kind": "robustness_decode",
        "replay_mode": replay_mode,
        "protocol_variant": "segmented_hex_multi_topic",
    }
    assert orchestrator._record_stratum(
        task, {"record_type": "robustness_decode"}
    ) == "robustness_decode:{}:segmented_hex_multi_topic".format(
        projected_replay_mode
    )


def test_robustness_replay_mode_mapping_fails_closed_on_unknown_mode() -> None:
    with pytest.raises(
        orchestrator.MethodologicalHalt,
        match="cannot map projected robustness replay_mode",
    ):
        orchestrator._record_stratum(
            {
                "work_kind": "robustness_decode",
                "replay_mode": "unfrozen_future_replay",
                "protocol_variant": "segmented_hex_multi_topic",
            },
            {"record_type": "robustness_decode"},
        )


@pytest.mark.parametrize(
    ("record_type", "expected"),
    [
        ("condition_unavailable", "planned_unavailable:condition_unavailable"),
        ("dependent_unavailable", "planned_unavailable:dependent_unavailable"),
    ],
)
def test_projected_unavailability_record_types_override_task_stratum(
    record_type: str, expected: str
) -> None:
    assert orchestrator._record_stratum(
        {"work_kind": "unexecuted_by_contract"},
        {"record_type": record_type},
    ) == expected


def test_every_frozen_runner_plan_reconciles_to_authorized_stratum_targets() -> None:
    projection = orchestrator.verify_projection()
    projected = {
        (str(row["stage"]), str(row["model_id"])): Counter(
            {
                str(item["stratum"]): int(item["target_units"])
                for item in row.get("strata", [])
            }
        )
        for row in projection["projection_rows"]
        if row.get("resource_class") == "gpu"
        and row.get("stage") in orchestrator.RUNNER_COUNTS
    }
    expected_keys = {
        (stage, model)
        for stage, counts in orchestrator.RUNNER_COUNTS.items()
        for model in counts
    }
    assert set(projected) == expected_keys

    normal_record_types = {
        "control": "ordinary_control",
        "rankcloak": "rankcloak_trial",
        "reference": "robustness_reference",
        "robustness_transform": "robustness_transform",
        "robustness_decode": "robustness_decode",
    }
    unavailable_replacements = {
        (
            "ablation_v2",
            "mistral_7b_instruct_v0_3_q4_k_m",
        ): (
            Counter({"rankcloak:segmented_hex_multi_topic": 48}),
            "planned_unavailable:condition_unavailable",
        ),
        (
            "robustness_v2",
            "mistral_7b_instruct_v0_3_q4_k_m",
        ): (
            Counter(
                {
                    "reference": 48,
                    "robustness_decode:text_retokenized:segmented_hex_multi_topic": 288,
                }
            ),
            "planned_unavailable:dependent_unavailable",
        ),
    }
    observed_work_kinds: set[str] = set()
    observed_robustness_replay_modes: set[str] = set()

    for stage, counts in orchestrator.RUNNER_COUNTS.items():
        plan = build_stage_plan(stage)
        for model, total in counts.items():
            selected = [row for row in plan if row["model_id"] == model]
            assert len(selected) == total
            observed_work_kinds.update(str(row["work_kind"]) for row in selected)
            observed_robustness_replay_modes.update(
                str(row["replay_mode"])
                for row in selected
                if row["work_kind"] == "robustness_decode"
            )
            reconciled = Counter(
                orchestrator._record_stratum(
                    row,
                    {
                        "record_type": normal_record_types[str(row["work_kind"])]
                    },
                )
                for row in selected
            )
            replacement = unavailable_replacements.get((stage, model))
            if replacement is not None:
                removed, unavailable_stratum = replacement
                for stratum, count in removed.items():
                    assert reconciled[stratum] >= count
                    reconciled[stratum] -= count
                    if reconciled[stratum] == 0:
                        del reconciled[stratum]
                reconciled[unavailable_stratum] += sum(removed.values())
            assert reconciled == projected[(stage, model)]
            assert sum(projected[(stage, model)].values()) == total

    assert observed_work_kinds == set(normal_record_types)
    assert observed_robustness_replay_modes == {
        "canonicalized_text_retokenized",
        "cross_model_text_retokenized",
        "detokenized_text_retokenized",
        "greedy_leadin_regeneration",
        "transformed_text_retokenized",
    }


def test_single_instance_lock_rejects_duplicate_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    first = orchestrator.acquire_single_instance_lock()
    try:
        with pytest.raises(orchestrator.OrchestratorError, match="already holds"):
            orchestrator.acquire_single_instance_lock()
    finally:
        orchestrator.os.close(first)
    second = orchestrator.acquire_single_instance_lock()
    orchestrator.os.close(second)


def test_final_progress_snapshot_is_byte_exact_no_overwrite_and_stops_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    source = tmp_path / orchestrator.PROGRESS_PATH
    source.parent.mkdir(parents=True)
    progress = {
        "schema_version": "fixture-progress",
        "progress_sha256": "fixture-hash",
        "value": 1,
    }
    source.write_bytes(orchestrator._json_bytes(progress))
    monkeypatch.setattr(
        orchestrator, "require_canonical_evaluator_completion", lambda *_: None
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_final_progress_snapshot",
        lambda: orchestrator.read_json(tmp_path / orchestrator.FINAL_PROGRESS_PATH),
    )
    monkeypatch.setattr(orchestrator, "emit_event", lambda *_args, **_kwargs: None)
    assert orchestrator.seal_final_progress_snapshot(progress) == progress
    sealed = tmp_path / orchestrator.FINAL_PROGRESS_PATH
    assert sealed.read_bytes() == source.read_bytes()

    source.write_bytes(orchestrator._json_bytes({"value": 2}))
    assert orchestrator.seal_final_progress_snapshot({"value": 2}) == progress
    assert orchestrator.operational_progress() == progress
    with pytest.raises(FileExistsError):
        orchestrator.atomic_publish_once_bytes(sealed, b"replacement")


def test_orphan_attachment_requires_exact_scientific_roots() -> None:
    actions = orchestrator.build_gpu_actions()
    robustness = next(
        action
        for action in actions
        if action.kind == "runner" and action.stage == "robustness_v2"
    )
    assert orchestrator._process_matches_action(robustness.argv, robustness)
    bad_robustness = list(robustness.argv)
    index = bad_robustness.index("--ablation-results-root") + 1
    bad_robustness[index] = "results/revision_v1/wrong_ablation"
    assert not orchestrator._process_matches_action(bad_robustness, robustness)

    evaluator = next(action for action in actions if action.kind == "evaluator")
    assert orchestrator._process_matches_action(evaluator.argv, evaluator)
    bad_evaluator = list(evaluator.argv)
    index = bad_evaluator.index("--source-results-root") + 1
    bad_evaluator[index] = "results/revision_v1/wrong_sources"
    assert not orchestrator._process_matches_action(bad_evaluator, evaluator)
    assert not orchestrator._process_matches_action(
        evaluator.argv + ("--threads", "2"), evaluator
    )


def test_waiting_mode_rejects_overlap_but_waits_for_external_gpu_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = next(
        action
        for action in orchestrator.build_gpu_actions()
        if action.kind == "runner"
    )
    monkeypatch.setattr(orchestrator, "_gpu_compute_pids", lambda: {101, 202})
    monkeypatch.setattr(
        orchestrator,
        "_process_cmdlines",
        lambda: {101: action.argv, 202: ("external-cuda-job",)},
    )
    monkeypatch.setattr(orchestrator, "refresh_progress", lambda: {})
    monkeypatch.setattr(orchestrator, "calculate_budget", lambda *_: {})
    monkeypatch.setattr(orchestrator, "enforce_budget", lambda *_: None)
    with pytest.raises(orchestrator.MethodologicalHalt, match="overlapping"):
        orchestrator.wait_for_existing_gpu_occupancy(
            action,
            projection={},
            retries={},
            max_retries=2,
            poll_seconds=5,
        )

    writes: list[dict] = []
    monkeypatch.setattr(orchestrator, "_gpu_compute_pids", lambda: {202})
    monkeypatch.setattr(orchestrator, "write_state", lambda **kwargs: writes.append(kwargs))
    monkeypatch.setattr(orchestrator.time, "sleep", lambda *_: None)
    assert orchestrator.wait_for_existing_gpu_occupancy(
        action,
        projection={},
        retries={},
        max_retries=2,
        poll_seconds=5,
    )
    assert writes[-1]["status"] == "waiting_for_gpu_availability"


def test_evaluator_gate_reconciles_17232_scores_plus_48_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = orchestrator.build_gpu_actions()
    evaluator = [action for action in actions if action.kind == "evaluator"]
    expected = {action.action_id: int(action.expected_count) for action in evaluator}
    observed_write: dict[str, int] = {}
    monkeypatch.setattr(
        orchestrator,
        "evaluator_dry_run_count",
        lambda action: expected[action.action_id],
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_or_write_evaluator_unavailability_manifest",
        lambda projection, scoreable: observed_write.update(scoreable=scoreable) or {},
    )
    projection = {
        "stage_totals": [
            {"stage": "heldout_evaluator", "target_work_units": 17280}
        ]
    }
    orchestrator.evaluator_projection_gate(actions, projection)
    assert observed_write == {"scoreable": 17232}

    bad_id = evaluator[0].action_id
    monkeypatch.setattr(
        orchestrator,
        "evaluator_dry_run_count",
        lambda action: expected[action.action_id] - int(action.action_id == bad_id),
    )
    with pytest.raises(orchestrator.MethodologicalHalt, match="exact evaluator plan"):
        orchestrator.evaluator_projection_gate(actions, projection)


def test_unavailability_manifest_binds_exact_48_records_and_self_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    source = (
        tmp_path
        / "results/revision_v1/ablation_v2"
        / "mistral_7b_instruct_v0_3_q4_k_m"
    )
    plan = []
    records = []
    ids = []
    for index in range(48):
        work_id = "unavailable-{:02d}".format(index)
        ids.append(work_id)
        plan.append(
            {
                "work_id": work_id,
                "work_kind": "rankcloak",
                "protocol_variant": "segmented_hex_multi_topic",
                "payload_name": "payload-{:02d}".format(index),
            }
        )
        records.append(
            {
                "work_id": work_id,
                "record_type": "condition_unavailable",
                "execution_status": "completed",
                "reason_code": "empty_isolated_roundtrip_vocabulary",
                "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
                "result_schema_revision": orchestrator.RESULT_REVISION,
            }
        )
    _jsonl(source / "plan.jsonl", plan)
    _jsonl(source / "records.jsonl", records)
    _json(
        source / "checkpoint.json",
        {"completed_trial_ids": ids, "failed_trial_ids": []},
    )
    _json(source / "run_identity.json", {"fixture": True})
    projection = {
        "stage_totals": [
            {"stage": "heldout_evaluator", "target_work_units": 17280}
        ]
    }
    manifest = orchestrator.verify_or_write_evaluator_unavailability_manifest(
        projection, 17232
    )
    assert manifest["upstream_dependent_unavailable_units"] == 48
    assert manifest["terminal_accounted_units"] == 17280
    assert manifest["scores_imputed_or_fabricated"] is False
    assert len({row["source_work_id"] for row in manifest["units"]}) == 48
    assert manifest["units_sha256"] == orchestrator.canonical_json_sha256(
        manifest["units"]
    )
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256")
    assert claimed == orchestrator.canonical_json_sha256(unsigned)
    assert orchestrator.verify_evaluator_unavailability_manifest() == manifest

    records[0]["reason_code"] = "tampered"
    _jsonl(source / "records.jsonl", records)
    with pytest.raises(orchestrator.MethodologicalHalt, match="identity mismatch"):
        orchestrator.verify_evaluator_unavailability_manifest()


def test_strict_stage_specific_preprocessing_commands() -> None:
    by_id = {spec["operation_id"]: spec for spec in orchestrator.preprocess_specs()}
    assert set(by_id) == {
        "preprocess_primary_v2",
        "preprocess_ablation_v2",
        "preprocess_multilingual_v2",
        "preprocess_robustness_v2",
    }
    assert "--allow-incomplete" not in by_id["preprocess_primary_v2"]["argv"]
    assert by_id["preprocess_primary_v2"]["argv"].count("--run-dir") == 3
    assert by_id["preprocess_primary_v2"]["argv"].count(
        "--reference-run-dir"
    ) == 0
    assert by_id["preprocess_ablation_v2"]["argv"].count(
        "--reference-run-dir"
    ) == 3
    assert by_id["preprocess_robustness_v2"]["argv"].count(
        "--reference-run-dir"
    ) == 6
    for spec in by_id.values():
        assert spec["completion"]["stage"] in spec["operation_id"]
        assert spec["atomic_staging"] is True


def test_downstream_contract_pins_join_detector_r_report_and_manifest() -> None:
    contract = orchestrator.load_command_contract(
        PROJECT_ROOT / "operations/confirmatory_v2/downstream_commands.json"
    )
    by_id = {row["operation_id"]: row for row in contract["operations"]}
    assert list(by_id) == [
        "primary_evaluator_join",
        "detector",
        "statistics",
        "mixed_models_r",
        "theory",
        "reports",
        "figures",
    ]
    join = by_id["primary_evaluator_join"]["argv"]
    assert join.count("--evaluator-feature-manifest") == 3
    assert "{primary_preprocessing_manifest}" in join
    detector = by_id["detector"]
    assert detector["completion"]["kind"] == "detector_v2"
    assert "--overwrite" in detector["argv"]
    for token in (
        "--resume",
        "--device",
        "--gpu-uuid",
        "--workers",
        "--checkpoint-dir",
        "--status-file",
        "--execution-policy",
        "--fit-permit-file",
    ):
        assert token in detector["argv"]
        assert token in detector["interface"]["required_help_tokens"]
    assert detector["argv"][detector["argv"].index("--device") + 1] == "cuda:0"
    assert detector["argv"][detector["argv"].index("--workers") + 1] == "1"
    assert detector["execution"] == {
        "kind": "checkpointed_detector_gpu_v1",
        "device": "cuda:0",
        "gpu_uuid": "{gpu_uuid}",
        "workers": 1,
        "total_fits": 56,
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
        "execution_policy_sha256": (
            orchestrator.DETECTOR_EXECUTION_POLICY_SHA256
        ),
        "execution_policy_content_sha256": (
            orchestrator.DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        ),
        "next_fit_upper_seconds_by_detector": (
            orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS
        ),
        "ceiling_gate": "signed_single_use_per_fit_v1",
    }
    assert "--preprocessing-manifest" in detector["argv"]
    assert "--preprocessing-manifest" in detector["interface"][
        "required_help_tokens"
    ]
    statistics = by_id["statistics"]
    assert "--continuous-quality" in statistics["argv"]
    assert "--continuous-quality" in statistics["interface"][
        "required_help_tokens"
    ]
    evaluator_continuous = [
        value
        for value in statistics["argv"]
        if value.endswith("_continuous}")
    ]
    assert len(evaluator_continuous) == 9
    assert statistics["argv"].index("--continuous-quality") < min(
        statistics["argv"].index(value) for value in evaluator_continuous
    )
    assert max(
        statistics["argv"].index(value) for value in evaluator_continuous
    ) < statistics["argv"].index("--features")
    mixed = by_id["mixed_models_r"]["argv"]
    assert "--feature-join-manifest" in mixed
    assert "{evaluator_join_features}" in mixed
    report = by_id["reports"]
    assert "--mixed-model-manifest" in report["argv"]
    assert "--evaluator-unavailability-manifest" in report["argv"]
    assert "--evaluator-unavailability-manifest" in report["interface"][
        "required_help_tokens"
    ]
    assert by_id["figures"]["completion"]["path"] == "{figures_manifest}"


def test_current_downstream_interfaces_exist_and_manuscript_stage_is_retired() -> None:
    contract_path = PROJECT_ROOT / "operations/confirmatory_v2/downstream_commands.json"
    contract = orchestrator.load_command_contract(contract_path)
    substitutions = orchestrator._format_values()
    forbidden = (
        "scripts/revise_revision_manuscripts.py",
        "tests/test_revision_manuscripts.py",
        ".paper/scientific_reports",
        "results/revision_v1/manuscript_revision_v2",
    )
    serialized = json.dumps(contract, sort_keys=True)
    assert all(token not in serialized for token in forbidden)
    assert contract["operations"][-1]["operation_id"] == "figures"
    assert contract["operations"][-1]["completion"]["kind"] == "figures_v1"

    for operation in contract["operations"]:
        interface = operation["interface"]
        interface_path = Path(
            str(interface["path"]).format_map(substitutions)
        )
        if not interface_path.is_absolute():
            interface_path = PROJECT_ROOT / interface_path
        assert interface_path.is_file(), operation["operation_id"]
        assert not interface_path.is_symlink(), operation["operation_id"]

    figures = contract["operations"][-1]
    runtime_path = figures["interface"]["runtime_path"]
    assert runtime_path == figures["argv"][1]
    assert runtime_path.startswith("{report_output_dir}/")


def test_current_commands_and_documentation_exclude_retired_manuscript_paths() -> None:
    forbidden = (
        "scripts/revise_revision_manuscripts.py",
        "tests/test_revision_manuscripts.py",
        ".paper/scientific_reports",
        "results/revision_v1/manuscript_revision_v2",
    )
    active_paths = (
        "README.md",
        "operations/confirmatory_v2/downstream_commands.json",
        "scripts/supervise_confirmatory_v2.py",
        "release/revision_v1_template/release_spec.json",
        "release/revision_v1_template/README.md",
        "revision_docs/DOI_RELEASE_PLAN.md",
    )
    for relative in active_paths:
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), relative


def test_checkpointed_detector_action_and_status_are_exact_and_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = _detector_action()
    orchestrator.validate_detector_action(action)
    status = _detector_status(action)
    status_path = Path(orchestrator._format_values()["detector_status_file"])
    _json(status_path, status)
    observed = orchestrator.read_detector_status(action)
    assert observed["completed_fit_count"] == 7
    assert observed["total_fit_count"] == 56
    assert observed["current_fit"]["fit_number"] == 8
    assert orchestrator._detector_status_gpu_seconds(observed) == pytest.approx(60.0)

    status["completed_fit_count"] = 8
    _json(status_path, status)
    with pytest.raises(orchestrator.MethodologicalHalt, match="self-hash"):
        orchestrator.read_detector_status(action)


def test_live_detector_gpu_time_enters_hard_ceiling_exactly_once() -> None:
    projection = orchestrator.verify_projection()
    progress = {
        "gpu": {
            "monitored_confirmatory_gpu_hours": 10.0,
            "cumulative_actual_gpu_hours": 12.0,
            "confirmatory_intervals": [],
        }
    }
    without_live = orchestrator.calculate_budget(projection, progress)
    with_live = orchestrator.calculate_budget(
        projection, progress, live_detector_gpu_seconds=3600.0
    )
    failed_hours = (
        orchestrator.detector_failed_benchmark_gpu_seconds() / 3600.0
    )
    assert with_live["failed_detector_benchmark_gpu_hours"] == failed_hours
    assert with_live["cumulative_actual_gpu_hours"] == pytest.approx(
        orchestrator.DETECTOR_HISTORICAL_GPU_HOURS_FLOOR + failed_hours + 1.0
    )
    assert with_live["revised_upper_gpu_hours"] == pytest.approx(
        without_live["revised_upper_gpu_hours"] + 1.0
    )

    progress["gpu"]["confirmatory_intervals"] = [
        {
            "component": "neural_detector",
            "model_id": "pre_final_detector_gpu_ledger",
        }
    ]
    sequential = orchestrator.calculate_budget(
        projection, progress, live_detector_gpu_seconds=1.0
    )
    assert sequential["cumulative_actual_gpu_hours"] == pytest.approx(
        orchestrator.DETECTOR_HISTORICAL_GPU_HOURS_FLOOR
        + failed_hours
        + 1.0 / 3600.0
    )


def test_detector_resume_refuses_checkpoints_without_cuda_charge_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = _detector_action()
    checkpoint = Path(orchestrator._format_values()["detector_checkpoint_dir"])
    checkpoint.mkdir(parents=True)
    (checkpoint / "execution_plan.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_exact_detector_pid", lambda _action: None)
    monkeypatch.setattr(
        orchestrator, "verify_detector_execution_policy", lambda: {}
    )
    monkeypatch.setattr(orchestrator, "refresh_progress", lambda: {
        "gpu": {
            "monitored_confirmatory_gpu_hours": 0.0,
            "cumulative_actual_gpu_hours": 0.0,
            "confirmatory_intervals": [],
        }
    })
    monkeypatch.setattr(
        orchestrator,
        "calculate_budget",
        lambda *_args, **_kwargs: {
            "cumulative_actual_gpu_hours": 0.0,
            "revised_upper_gpu_hours": 1.0,
        },
    )
    monkeypatch.setattr(orchestrator, "enforce_budget", lambda _budget: None)
    with pytest.raises(orchestrator.MethodologicalHalt, match="charge history"):
        orchestrator.run_checkpointed_detector_process(
            action, projection={}, retries={}, poll_seconds=5
        )


def test_detector_child_environment_is_uuid_pinned_and_single_worker() -> None:
    environment = orchestrator._detector_child_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == orchestrator.GPU_UUID
    assert environment["RANKCLOAK_DETECTOR_DEVICE"] == "cuda:0"
    assert environment["RANKCLOAK_DETECTOR_GPU_UUID"] == orchestrator.GPU_UUID
    assert environment["RANKCLOAK_DETECTOR_WORKERS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"


def test_signed_detector_fit_permit_is_invocation_bound_and_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = _detector_action()
    status = _detector_status(action)
    status.update(
        {
            "state": "awaiting_fit_ceiling_gate",
            "current_fit": None,
            "current_fit_started_at_utc": None,
            "current_fit_elapsed_seconds": None,
            "next_fit": {
                "ordinal": 7,
                "index": 7,
                "fit_number": 8,
                "split_id": "split-04",
                "regime": "leave_one_model_out",
                "detector_name": "published_textcnn_equivalent",
                "detector_kind": "text_cnn",
                "seed": 123,
                "task_identity_sha256": "c" * 64,
            },
            "next_fit_upper_seconds": orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"],
            "fit_gate_nonce": "d" * 64,
        }
    )
    status["status_sha256"] = orchestrator.canonical_json_sha256(
        {key: value for key, value in status.items() if key != "status_sha256"}
    )
    status_path = Path(orchestrator._format_values()["detector_status_file"])
    _json(status_path, status)
    monkeypatch.setattr(
        orchestrator, "verify_detector_execution_policy", lambda: {}
    )
    monkeypatch.setattr(
        orchestrator,
        "_detector_pid_is_live",
        lambda _action, pid, ticks: pid == 4321 and ticks == 987654,
    )
    monkeypatch.setattr(orchestrator, "_process_start_ticks", lambda _pid: 987654)
    monkeypatch.setattr(orchestrator, "emit_event", lambda *_args, **_kwargs: None)
    budget = {
        "cumulative_actual_gpu_hours": 10.0,
        "revised_upper_gpu_hours": 20.0,
        "live_detector_remaining_gpu_hours": 1.0,
    }
    permit = orchestrator.issue_detector_fit_permit(
        action,
        status,
        budget,
        pid=4321,
        start_ticks=987654,
    )
    assert permit["invocation_pid"] == 4321
    assert permit["invocation_start_ticks"] == 987654
    assert permit["next_fit_upper_seconds"] == orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"]
    assert permit["permit_sha256"] == orchestrator.canonical_json_sha256(
        {key: value for key, value in permit.items() if key != "permit_sha256"}
    )
    assert orchestrator.issue_detector_fit_permit(
        action,
        status,
        budget,
        pid=4321,
        start_ticks=987654,
    ) == permit

    with pytest.raises(orchestrator.BudgetHalt, match="next detector fit"):
        orchestrator.enforce_detector_next_fit_reserve(
            {
                "cumulative_actual_gpu_hours": 164.9,
                "revised_upper_gpu_hours": 164.9,
                "live_detector_remaining_gpu_hours": 0.0,
            },
            orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"],
        )


def test_detector_rolling_eta_enters_revised_ceiling_without_double_reserve() -> None:
    projection = orchestrator.verify_projection()
    progress = {
        "gpu": {
            "monitored_confirmatory_gpu_hours": 10.0,
            "cumulative_actual_gpu_hours": 12.0,
            "confirmatory_intervals": [],
        }
    }
    baseline = orchestrator.calculate_budget(projection, progress)
    projected = orchestrator.calculate_budget(
        projection,
        progress,
        live_detector_remaining_seconds=7200.0,
    )
    assert projected["revised_upper_gpu_hours"] == pytest.approx(
        baseline["revised_upper_gpu_hours"] + 2.0
    )
    # A smaller next-fit reserve is already contained in the signed ETA.
    orchestrator.enforce_detector_next_fit_reserve(projected, 3600.0)


def test_consumed_detector_fit_permit_receipt_prevents_reissue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = _detector_action()
    status = _detector_status(action)
    status.update(
        {
            "state": "awaiting_fit_ceiling_gate",
            "current_fit": None,
            "current_fit_started_at_utc": None,
            "current_fit_elapsed_seconds": None,
            "next_fit": {
                "ordinal": 7,
                "index": 7,
                "fit_number": 8,
                "split_id": "split-04",
                "regime": "leave_one_model_out",
                "detector_name": "published_textcnn_equivalent",
                "detector_kind": "text_cnn",
                "seed": 123,
                "task_identity_sha256": "c" * 64,
            },
            "next_fit_upper_seconds": orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"],
            "fit_gate_nonce": "d" * 64,
        }
    )
    status["status_sha256"] = orchestrator.canonical_json_sha256(
        {key: value for key, value in status.items() if key != "status_sha256"}
    )
    _json(Path(orchestrator._format_values()["detector_status_file"]), status)
    permit = {
        "schema_version": orchestrator.DETECTOR_FIT_PERMIT_SCHEMA,
        "run_identity_sha256": status["run_identity_sha256"],
        "task_identity_sha256": "c" * 64,
        "fit_gate_nonce": "d" * 64,
        "invocation_pid": 4321,
        "invocation_start_ticks": 987654,
        "next_fit_upper_seconds": orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"],
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    permit["permit_sha256"] = orchestrator.canonical_json_sha256(permit)
    permit_path = Path(orchestrator._format_values()["detector_fit_permit_file"])
    _json(permit_path, permit)
    receipt = {
        "schema_version": orchestrator.DETECTOR_FIT_PERMIT_RECEIPT_SCHEMA,
        "run_identity_sha256": status["run_identity_sha256"],
        "task_identity_sha256": "c" * 64,
        "fit_gate_nonce": "d" * 64,
        "invocation_pid": 4321,
        "invocation_start_ticks": 987654,
        "next_fit_upper_seconds": orchestrator.DETECTOR_NEXT_FIT_UPPER_SECONDS["published_textcnn_equivalent"],
        "issued_permit_sha256": permit["permit_sha256"],
        "issued_permit_file_sha256": orchestrator.file_sha256(permit_path),
        "issued_permit_size_bytes": permit_path.stat().st_size,
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = orchestrator.canonical_json_sha256(receipt)
    receipt_path = (
        Path(orchestrator._format_values()["detector_fit_permit_receipt_dir"])
        / ("d" * 64 + ".json")
    )
    _json(receipt_path, receipt)
    permit_path.unlink()
    monkeypatch.setattr(
        orchestrator, "verify_detector_execution_policy", lambda: {}
    )
    monkeypatch.setattr(
        orchestrator,
        "_detector_pid_is_live",
        lambda _action, pid, ticks: pid == 4321 and ticks == 987654,
    )
    monkeypatch.setattr(orchestrator, "_process_start_ticks", lambda _pid: 987654)
    monkeypatch.setattr(
        orchestrator,
        "atomic_publish_once_bytes",
        lambda *_args, **_kwargs: pytest.fail("consumed nonce was reissued"),
    )
    observed = orchestrator.issue_detector_fit_permit(
        action,
        status,
        {
            "cumulative_actual_gpu_hours": 10.0,
            "revised_upper_gpu_hours": 20.0,
            "live_detector_remaining_gpu_hours": 1.0,
        },
        pid=4321,
        start_ticks=987654,
    )
    assert observed == receipt
    assert not permit_path.exists()


def test_detector_signals_are_exact_pid_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _detector_action()
    signals: list[tuple[int, int]] = []
    live = iter((True, False))
    monkeypatch.setattr(
        orchestrator,
        "_detector_pid_is_live",
        lambda *_args: next(live),
    )
    monkeypatch.setattr(
        orchestrator.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda *_args: pytest.fail("detector stop used a process-group signal"),
    )
    orchestrator._terminate_detector_process(action, 4321, 987654)
    assert signals == [(4321, orchestrator.signal.SIGTERM)]

    signals.clear()
    monkeypatch.setattr(
        orchestrator, "_detector_pid_is_live", lambda *_args: True
    )
    monkeypatch.setattr(orchestrator, "emit_event", lambda *_args, **_kwargs: None)
    orchestrator._kill_detector_at_hard_ceiling(action, 4321, 987654)
    assert signals == [
        (4321, orchestrator.signal.SIGSTOP),
        (4321, orchestrator.signal.SIGKILL),
    ]


def test_supervisor_closes_detector_crash_tail_durably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = _detector_action()
    status = _detector_status(action)
    accounting = _detector_gpu_accounting(live=False, seconds=60.0, pid=4321)
    status["gpu_accounting"] = accounting
    status["status_sha256"] = orchestrator.canonical_json_sha256(
        {key: value for key, value in status.items() if key != "status_sha256"}
    )
    _json(Path(orchestrator._format_values()["detector_status_file"]), status)
    prior_end = datetime.fromisoformat(
        accounting["intervals"][-1]["completed_at_utc"]
    )
    absent = prior_end + timedelta(seconds=5)
    observed = orchestrator.close_detector_gpu_interval_after_exit(
        action,
        status,
        pid=4321,
        start_ticks=987654,
        observed_absent_at=absent,
    )
    assert observed["gpu_accounting"]["cumulative_elapsed_seconds"] == 65.0
    assert observed["state"] == "supervisor_observed_process_exit"
    assert observed["recovered_errors"][-1]["type"] == (
        "supervisor_closed_exited_gpu_interval"
    )

    # A stale absence observation can never shrink already durable accounting.
    observed = orchestrator.close_detector_gpu_interval_after_exit(
        action,
        observed,
        pid=4321,
        start_ticks=987654,
        observed_absent_at=prior_end,
    )
    assert observed["gpu_accounting"]["cumulative_elapsed_seconds"] == 65.0


def test_complete_detector_status_reenters_idempotent_finalizer_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _detector_action()
    status = _detector_status(action)
    status["state"] = "complete"
    status["gpu_accounting"] = _detector_gpu_accounting(live=False)
    status["finalization_candidate"] = {
        "path": "fixture-candidate",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "candidate_sha256": "b" * 64,
        "kind": "detector_run_manifest",
    }
    status["status_sha256"] = orchestrator.canonical_json_sha256(
        {key: value for key, value in status.items() if key != "status_sha256"}
    )
    calls: list[str] = []
    monkeypatch.setattr(orchestrator, "validate_detector_action", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator, "verify_detector_execution_policy", lambda: {}
    )
    monkeypatch.setattr(orchestrator, "_exact_detector_pid", lambda _action: None)
    monkeypatch.setattr(
        orchestrator, "read_detector_status", lambda *a, **k: status
    )
    monkeypatch.setattr(
        orchestrator,
        "_status_declares_expected_detector_candidate",
        lambda _action, _status: True,
    )
    monkeypatch.setattr(
        orchestrator,
        "_finalize_detector_after_confirmed_exit",
        lambda _action, _status: calls.append("finalize") or {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_ensure_finalized_detector_ledger",
        lambda _action: calls.append("ledger"),
    )
    code, detail = orchestrator.run_checkpointed_detector_process(
        action, projection={}, retries={}, poll_seconds=5.0
    )
    assert code == 0
    assert "finalized" in detail
    assert calls == ["finalize", "ledger"]


def test_pending_detector_manifest_defers_only_for_exact_signed_candidate(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "detector_run_manifest.json"
    run_identity_sha = "a" * 64
    candidate_path = tmp_path / "finalization_candidate.json"
    candidate = {
        "schema_version": (
            "rankcloak-revision-detector-finalization-candidate-v1"
        ),
        "created_at_utc": "2026-08-13T00:00:00+00:00",
        "kind": "detector_run_manifest",
        "run_identity_sha256": run_identity_sha,
        "requested_output_path": str(manifest_path.resolve()),
        "payload": {},
        "payload_sha256": orchestrator.canonical_json_sha256({}),
        "output_files": {},
        "output_files_sha256": orchestrator.canonical_json_sha256({}),
        "finalization_policy": (
            "supervisor_confirms_exact_pid_absence_closes_accounting_then_publishes_v1"
        ),
    }
    candidate["candidate_sha256"] = orchestrator.canonical_json_sha256(
        candidate
    )
    _json(candidate_path, candidate)
    candidate_identity = {
        "path": str(candidate_path.resolve()),
        "sha256": orchestrator.file_sha256(candidate_path),
        "size_bytes": candidate_path.stat().st_size,
        "candidate_sha256": candidate["candidate_sha256"],
    }
    status = {
        "state": "supervisor_observed_process_exit",
        "run_identity_sha256": run_identity_sha,
        "finalization_candidate": {
            **candidate_identity,
            "kind": "detector_run_manifest",
        },
    }
    status["status_sha256"] = orchestrator.canonical_json_sha256(status)
    manifest = {
        "run_identity_sha256": run_identity_sha,
        "terminal_accounting_status_sha256": status["status_sha256"],
        "finalization_candidate": candidate_identity,
    }
    assert orchestrator.detector_manifest_awaits_supervisor_finalization(
        manifest_path, manifest, status
    )

    manifest["terminal_accounting_status_sha256"] = "b" * 64
    with pytest.raises(orchestrator.InterfaceHalt, match="identity differs"):
        orchestrator.detector_manifest_awaits_supervisor_finalization(
            manifest_path, manifest, status
        )


def test_benchmark_checkpoint_verifier_rehashes_children(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    fit_dir = checkpoint_dir / "fits" / "0000"
    fit_dir.mkdir(parents=True)
    identity = {
        "ordinal": 0,
        "detector_name": "published_textcnn_equivalent",
    }
    plan = {"tasks": [identity]}
    _json(checkpoint_dir / "execution_plan.json", plan)
    children = {}
    for name, columns, rows in (
        ("metric.json", ["detector_name"], [["published_textcnn_equivalent"]]),
        ("predictions.json", ["row_id", "score"], [["row-1", 0.5]]),
    ):
        child = fit_dir / name
        _json(
            child,
            {
                "schema_version": "rankcloak-revision-detector-fit-rows-v1",
                "columns": columns,
                "rows": rows,
            },
        )
        children[name] = {
            "sha256": orchestrator.file_sha256(child),
            "size_bytes": child.stat().st_size,
            "row_count": len(rows),
        }
    checkpoint = {
        "schema_version": "rankcloak-revision-detector-fit-checkpoint-v1",
        "run_identity_sha256": "a" * 64,
        "plan_sha256": orchestrator.canonical_json_sha256(plan),
        "task_identity": identity,
        "task_identity_sha256": orchestrator.canonical_json_sha256(identity),
        "started_at_utc": "2026-08-12T00:00:00+00:00",
        "completed_at_utc": "2026-08-12T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "children": children,
        "children_sha256": orchestrator.canonical_json_sha256(children),
    }
    checkpoint["manifest_sha256"] = orchestrator.canonical_json_sha256(checkpoint)
    checkpoint_path = fit_dir / "manifest.json"
    _json(checkpoint_path, checkpoint)
    benchmark = {
        "checkpoint_dir": str(checkpoint_dir),
        "last_completed_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": orchestrator.file_sha256(checkpoint_path),
            "task_ordinal": 0,
        },
        "benchmark_task_identity": identity,
        "run_identity_sha256": "a" * 64,
        "execution_plan_sha256": orchestrator.canonical_json_sha256(plan),
        "fit_elapsed_seconds": 1.0,
    }
    orchestrator._verify_detector_benchmark_checkpoint(benchmark, task_index=0)
    (fit_dir / "metric.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(orchestrator.InterfaceHalt, match="child differs"):
        orchestrator._verify_detector_benchmark_checkpoint(
            benchmark, task_index=0
        )


def test_evaluator_join_completion_requires_frozen_nested_feature_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "primary_features_with_heldout_evaluator.csv"

    def write_join_rows(row_count: int) -> None:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["trial_id", "segment_index"], lineterminator="\n"
            )
            writer.writeheader()
            for index in range(row_count):
                writer.writerow(
                    {
                        "trial_id": "trial-{:04d}".format(
                            index % orchestrator.PRIMARY_EVALUATOR_JOIN_TRIALS
                        ),
                        "segment_index": (
                            index // orchestrator.PRIMARY_EVALUATOR_JOIN_TRIALS
                        ),
                    }
                )

    write_join_rows(orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS)
    manifest = {
        "schema_version": "rankcloak-revision-heldout-feature-join-v1",
        "manifest_type": "rankcloak_revision_primary_heldout_feature_join",
        "analysis_unit": "primary_payload_trial_with_nested_segment_rows",
        "input_scope": "primary_v2_rankcloak_full_message_only",
        "primary_trial_count": orchestrator.PRIMARY_EVALUATOR_JOIN_TRIALS,
        "primary_full_message_feature_rows": (
            orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
        ),
        "evaluator_score_rows_joined": orchestrator.PRIMARY_EVALUATOR_JOIN_TRIALS,
        "unmatched_primary_trials": 0,
        "duplicate_evaluator_trial_ids": 0,
        "source_record_hashes_recomputed": True,
        "evaluator_source_records_byte_identical_to_preprocessing": True,
        "evaluator_artifact_pins_verified": True,
        "segments_as_independent_observations": False,
        "score_scope": (
            "source_full_message_replicated_across_nested_segment_rows_v1"
        ),
        "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
        "result_schema_revision": orchestrator.RESULT_REVISION,
        "outputs": {
            "features": {
                "path": output.name,
                "row_count": orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS,
                "sha256": orchestrator.file_sha256(output),
                "size_bytes": output.stat().st_size,
            }
        },
    }
    manifest_path = tmp_path / "heldout_feature_join_manifest.json"
    _json(manifest_path, manifest)
    spec = {"completion": {"kind": "evaluator_join_v1", "path": str(manifest_path)}}

    assert orchestrator.verify_completion(spec, {})

    write_join_rows(orchestrator.PRIMARY_EVALUATOR_JOIN_TRIALS)
    feature_declaration = manifest["outputs"]["features"]
    feature_declaration["sha256"] = orchestrator.file_sha256(output)
    feature_declaration["size_bytes"] = output.stat().st_size
    _json(manifest_path, manifest)
    with pytest.raises(orchestrator.InterfaceHalt, match="exact nested primary table"):
        orchestrator.verify_completion(spec, {})

    write_join_rows(orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS)
    feature_declaration["sha256"] = orchestrator.file_sha256(output)
    feature_declaration["size_bytes"] = output.stat().st_size
    manifest["primary_full_message_feature_rows"] = 6480
    _json(manifest_path, manifest)
    with pytest.raises(orchestrator.InterfaceHalt, match="exact primary table"):
        orchestrator.verify_completion(spec, {})

    manifest["primary_full_message_feature_rows"] = (
        orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
    )
    manifest["outputs"]["features"]["row_count"] = 6480
    _json(manifest_path, manifest)
    with pytest.raises(orchestrator.InterfaceHalt, match="exact primary table"):
        orchestrator.verify_completion(spec, {})

    manifest["outputs"]["features"]["row_count"] = (
        orchestrator.PRIMARY_EVALUATOR_JOIN_FEATURE_ROWS
    )
    manifest["source_record_hashes_recomputed"] = False
    _json(manifest_path, manifest)
    with pytest.raises(orchestrator.InterfaceHalt, match="exact primary table"):
        orchestrator.verify_completion(spec, {})


def test_detector_completion_requires_exact_28_by_2_and_primary_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        orchestrator,
        "verify_detector_final_publication",
        lambda _path, _manifest, _status: None,
    )
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    primary = tmp_path / "analysis_inputs/primary_v2"
    primary.mkdir(parents=True)
    corpus = primary / "detector_corpus.jsonl"
    preprocessing = primary / "preprocessing_output_manifest.json"
    corpus.write_text("{}\n", encoding="utf-8")
    _json(preprocessing, {"fixture": True})
    plan = tmp_path / "analysis/revision_v1/detector_confirmatory_plan.json"
    _json(plan, {"fixture": True})
    outputs = {}
    for name in (
        "detector_metrics.csv",
        "detector_predictions.csv",
        "detector_dataset_manifest.csv",
        "detector_split_manifest.json",
        "detector_failures.json",
    ):
        product = detector_dir / name
        product.write_text("fixture\n", encoding="utf-8")
        outputs[name] = {
            "sha256": orchestrator.file_sha256(product),
            "size_bytes": product.stat().st_size,
        }
    manifest = {
        "schema_version": "rankcloak-revision-detector-run-v2",
        "execution_mode": "confirmatory",
        "smoke": False,
        "confirmatory_complete": True,
        "split_count": 28,
        "skipped_split_count": 0,
        "failure_count": 0,
        "metric_rows": 56,
        "smoke_fallback_metric_rows": 0,
        "input_path": str(corpus),
        "input_sha256": orchestrator.file_sha256(corpus),
        "preprocessing_manifest_path": str(preprocessing),
        "preprocessing_manifest_sha256": orchestrator.file_sha256(preprocessing),
        "confirmatory_plan_path": str(plan),
        "confirmatory_plan_sha256": orchestrator.file_sha256(plan),
        "confirmatory_plan_schema_version": (
            "rankcloak-revision-detector-confirmatory-plan-v1"
        ),
        "output_files": outputs,
    }
    values = orchestrator._format_values()
    run_identity = {
        "device": orchestrator.DETECTOR_DEVICE,
        "gpu_uuid": orchestrator.GPU_UUID,
        "workers": 1,
        "output_dir": str(detector_dir.resolve()),
        "checkpoint_dir": values["detector_checkpoint_dir"],
        "status_file": values["detector_status_file"],
        "execution_policy_path": values["detector_execution_policy"],
        "execution_policy_sha256": (
            orchestrator.DETECTOR_EXECUTION_POLICY_SHA256
        ),
        "fit_permit_file": values["detector_fit_permit_file"],
        "fit_permit_receipt_dir": values["detector_fit_permit_receipt_dir"],
        "require_fit_permit": True,
    }
    manifest.update(
        {
            "device": orchestrator.DETECTOR_DEVICE,
            "gpu_uuid": orchestrator.GPU_UUID,
            "workers": 1,
            "checkpoint_dir": values["detector_checkpoint_dir"],
            "status_file": values["detector_status_file"],
            "fit_permit_file": values["detector_fit_permit_file"],
            "fit_permit_receipt_dir": values[
                "detector_fit_permit_receipt_dir"
            ],
            "execution_policy_path": values["detector_execution_policy"],
            "execution_policy_sha256": (
                orchestrator.DETECTOR_EXECUTION_POLICY_SHA256
            ),
            "execution_policy_content_sha256": (
                orchestrator.DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
            ),
            "completed_fit_count": 56,
            "total_fit_count": 56,
            "resumed_fit_count": 0,
            "recovered_errors": [],
            "execution_started_at_utc": "2026-08-12T00:00:00+00:00",
            "execution_completed_at_utc": "2026-08-12T00:01:00+00:00",
            "fit_durations_seconds": [1.0] * 56,
            "checkpoint_cumulative_fit_seconds": 56.0,
            "run_identity": run_identity,
            "run_identity_sha256": orchestrator.canonical_json_sha256(
                run_identity
            ),
            "execution_plan_sha256": "b" * 64,
            "last_completed_checkpoint": {"ordinal": 55},
            "gpu_accounting": _detector_gpu_accounting(seconds=60.0),
        }
    )
    manifest_path = detector_dir / "detector_run_manifest.json"
    _json(manifest_path, manifest)
    final_status = {
        "schema_version": orchestrator.DETECTOR_STATUS_SCHEMA,
        "state": "complete",
        "completed_fit_count": 56,
        "total_fit_count": 56,
        "run_identity": run_identity,
        "run_identity_sha256": manifest["run_identity_sha256"],
        "gpu_accounting": manifest["gpu_accounting"],
        "final_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": orchestrator.file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    final_status["status_sha256"] = orchestrator.canonical_json_sha256(
        final_status
    )
    _json(Path(values["detector_status_file"]), final_status)
    spec = {
        "completion": {"kind": "detector_v2", "path": str(manifest_path)}
    }
    substitutions = {
        "primary_detector_corpus": str(corpus),
        "primary_preprocessing_manifest": str(preprocessing),
    }
    assert orchestrator.verify_completion(spec, substitutions)
    manifest["split_count"] = 27
    _json(manifest_path, manifest)
    with pytest.raises(orchestrator.InterfaceHalt, match="28-split/56-fit"):
        orchestrator.verify_completion(spec, substitutions)


def test_detector_final_publication_requires_ledger_receipt_marker_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    values = orchestrator._format_values()
    ledger_path = Path(values["detector_gpu_ledger"])
    ledger_path.parent.mkdir(parents=True)
    _json(ledger_path, {"fixture": "ledger"})
    marker_path = orchestrator.detector_gpu_ledger_incorporation_path(
        ledger_path
    )
    _json(marker_path, {"fixture": "marker"})
    manifest_path = tmp_path / "detector_run_manifest.json"
    _json(manifest_path, {"fixture": "manifest"})
    receipt_path = tmp_path / "detector.terminal_receipt.json"
    _json(receipt_path, {"fixture": "receipt"})
    accounting = _detector_gpu_accounting(seconds=60.0)
    ledger = {
        "ledger_sha256": "a" * 64,
        "sources_sha256": "d" * 64,
        "intervals_sha256": orchestrator.canonical_json_sha256(
            accounting["intervals"]
        ),
        "intervals": accounting["intervals"],
        "cumulative_elapsed_seconds": accounting[
            "cumulative_elapsed_seconds"
        ],
    }
    terminal_identity = {
        "path": str(receipt_path.resolve()),
        "sha256": orchestrator.file_sha256(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "terminal_receipt_sha256": "b" * 64,
    }
    reports = []
    verified_reports = []
    for task_index in (0, 1):
        report_path = Path(
            values["detector_equivalence_report_{}".format(task_index)]
        )
        _json(report_path, {"task": task_index})
        report_sha = str(task_index + 1) * 64
        reports.append(
            {
                "path": str(report_path.resolve()),
                "sha256": orchestrator.file_sha256(report_path),
                "size_bytes": report_path.stat().st_size,
                "report_sha256": report_sha,
            }
        )
        verified_reports.append({"report_sha256": report_sha})
    manifest = {
        "gpu_accounting": accounting,
        "pre_final_gpu_accounting_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": orchestrator.file_sha256(ledger_path),
            "size_bytes": ledger_path.stat().st_size,
            "ledger_sha256": ledger["ledger_sha256"],
            "sources_sha256": ledger["sources_sha256"],
            "intervals_sha256": ledger["intervals_sha256"],
            "cumulative_elapsed_seconds": ledger[
                "cumulative_elapsed_seconds"
            ],
        },
        "required_equivalence_reports": reports,
        "pre_final_gpu_accounting_ledger_path": str(ledger_path.resolve()),
        "run_identity": {
            "lineage": {},
            "excluded_operational_gate_fields": [
                "cuda_budget_gate",
                "pre_final_gpu_accounting_ledger",
                "pre_final_gpu_accounting_ledger_path",
                "required_equivalence_reports",
            ],
        },
    }
    marker = {
        "incorporation_sha256": "c" * 64,
        "ledger": {
            "ledger_sha256": ledger["ledger_sha256"],
            "intervals_sha256": ledger["intervals_sha256"],
        },
        "incorporated": True,
        "incorporated_ledger_interval_count": 1,
        "incorporated_ledger_intervals_sha256": ledger["intervals_sha256"],
        "final_gpu_accounting_sha256": orchestrator.canonical_json_sha256(
            accounting
        ),
        "final_published_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": orchestrator.file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "final_terminal_receipt": terminal_identity,
    }
    status = {
        "terminal_receipt": terminal_identity,
        "gpu_ledger_incorporation": {
            "path": str(marker_path.resolve()),
            "sha256": orchestrator.file_sha256(marker_path),
            "size_bytes": marker_path.stat().st_size,
            "incorporation_sha256": marker["incorporation_sha256"],
        },
    }
    monkeypatch.setattr(
        orchestrator, "read_detector_gpu_accounting_ledger", lambda _path: ledger
    )
    monkeypatch.setattr(
        orchestrator,
        "read_detector_gpu_ledger_incorporation_marker",
        lambda _path: marker,
    )
    monkeypatch.setattr(
        orchestrator,
        "require_detector_equivalence_gate",
        lambda: tuple(verified_reports),
    )
    orchestrator.verify_detector_final_publication(
        manifest_path, manifest, status
    )

    manifest["gpu_accounting"] = {
        **accounting,
        "intervals": [],
        "cumulative_elapsed_seconds": 0.0,
    }
    marker["final_gpu_accounting_sha256"] = (
        orchestrator.canonical_json_sha256(manifest["gpu_accounting"])
    )
    with pytest.raises(orchestrator.InterfaceHalt, match="omits"):
        orchestrator.verify_detector_final_publication(
            manifest_path, manifest, status
        )


def test_figure_manifest_is_self_hashed_and_tamper_evident(tmp_path: Path) -> None:
    report_manifest = tmp_path / "report_output_manifest.json"
    _json(report_manifest, {"fixture": True})
    registry = tmp_path / "plot_registry.csv"
    renderer = tmp_path / "render_revision_figures.py"
    renderer.write_text("# fixture renderer\n", encoding="utf-8")
    rows = [
        {"plot_id": "plot_{:02d}".format(index)} for index in range(18)
    ]
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["plot_id"])
        writer.writeheader()
        writer.writerows(rows)
    figures = []
    for row in rows:
        path = tmp_path / (row["plot_id"] + ".pdf")
        path.write_bytes(b"%PDF-fixture\n")
        figures.append(
            {
                "plot_id": row["plot_id"],
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": orchestrator.file_sha256(path),
            }
        )
    manifest = {
        "schema_version": orchestrator.FIGURE_MANIFEST_SCHEMA,
        "format": "pdf",
        "plot_registry_sha256": orchestrator.file_sha256(registry),
        "renderer_path": str(renderer.resolve()),
        "renderer_sha256": orchestrator.file_sha256(renderer),
        "report_manifest_sha256": orchestrator.file_sha256(report_manifest),
        "figures": figures,
        "figures_sha256": orchestrator.canonical_json_sha256(figures),
    }
    manifest["manifest_sha256"] = orchestrator.canonical_json_sha256(manifest)
    manifest_path = tmp_path / "figure_render_manifest.json"
    _json(manifest_path, manifest)
    spec = {
        "interface": {"path": str(renderer)},
        "completion": {
            "kind": "figures_v1",
            "path": str(manifest_path),
            "registry": str(registry),
            "output_dir": str(tmp_path),
            "format": "pdf",
        }
    }
    substitutions = {"report_manifest": str(report_manifest)}
    assert orchestrator.verify_completion(spec, substitutions)
    (tmp_path / "plot_00.pdf").write_bytes(b"%PDF-tampered\n")
    with pytest.raises(orchestrator.InterfaceHalt, match="hash mismatch"):
        orchestrator.verify_completion(spec, substitutions)


def test_shard_status_requires_exact_one_to_one_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "primary_v2" / orchestrator.MODELS[0]
    output.mkdir(parents=True)
    plan = [
        {
            "work_id": "work-1",
            "model_id": orchestrator.MODELS[0],
            "evidence_status": orchestrator.RUNNER_EVIDENCE["primary_v2"],
            "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
            "result_schema_revision": orchestrator.RESULT_REVISION,
        },
        {
            "work_id": "work-2",
            "model_id": orchestrator.MODELS[0],
            "evidence_status": orchestrator.RUNNER_EVIDENCE["primary_v2"],
            "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
            "result_schema_revision": orchestrator.RESULT_REVISION,
        },
    ]
    _jsonl(output / "plan.jsonl", plan)
    identity = {
        "config_manifest_sha256": orchestrator.CONFIG_SHA256,
        "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
        "result_schema_revision": orchestrator.RESULT_REVISION,
        "command_line_args": [
            "stage=primary_v2",
            "model_id={}".format(orchestrator.MODELS[0]),
            "evidence_status={}".format(orchestrator.RUNNER_EVIDENCE["primary_v2"]),
            "context_limit=4096",
            "gpu_uuid={}".format(orchestrator.GPU_UUID),
            "n_gpu_layers=-1",
            "protocol_contract_revision={}".format(orchestrator.PROTOCOL_REVISION),
            "result_schema_revision={}".format(orchestrator.RESULT_REVISION),
        ],
        "planned_trial_count": 2,
        "planned_trial_ids_sha256": orchestrator.canonical_json_sha256(
            ["work-1", "work-2"]
        ),
    }
    identity["run_identity_sha256"] = orchestrator.canonical_json_sha256(identity)
    _json(output / "run_identity.json", identity)
    checkpoint = {
        "planned_trial_count": 2,
        "planned_trial_ids_sha256": orchestrator.canonical_json_sha256(
            ["work-1", "work-2"]
        ),
        "config_manifest_sha256": orchestrator.CONFIG_SHA256,
        "completed_trial_ids": ["work-1"],
        "failed_trial_ids": [],
    }
    _json(output / "checkpoint.json", checkpoint)
    action = orchestrator.Action(
        action_id="fixture",
        stage="primary_v2",
        kind="runner",
        argv=(),
        output_dir=output,
        model_id=orchestrator.MODELS[0],
        expected_count=2,
        gpu=True,
    )
    assert orchestrator.shard_status(action) == "incomplete"

    checkpoint["completed_trial_ids"] = ["work-1", "work-2"]
    _json(output / "checkpoint.json", checkpoint)
    record = {
        "work_id": "work-1",
        "execution_status": "completed",
        "protocol_contract_revision": orchestrator.PROTOCOL_REVISION,
        "result_schema_revision": orchestrator.RESULT_REVISION,
    }
    _jsonl(output / "records.jsonl", [record, {**record, "work_id": "work-2"}])
    monkeypatch.setattr(
        orchestrator, "verify_completed_runner_shard", lambda *args, **kwargs: ([], {})
    )
    assert orchestrator.shard_status(action) == "complete"

    _jsonl(
        output / "records.jsonl",
        [record, {**record, "work_id": "work-2"}, dict(record)],
    )
    with pytest.raises(orchestrator.MethodologicalHalt, match="one-to-one"):
        orchestrator.shard_status(action)


def test_retry_count_recovers_from_atomic_error_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    action = orchestrator.Action(
        action_id="runner:fixture",
        stage="fixture",
        kind="runner",
        argv=(),
    )
    orchestrator.record_recoverable_error(action, 2, "fixture_failure", "detail")
    assert orchestrator.load_retry_counts() == {"runner:fixture": 2}
    rows = orchestrator.read_jsonl(tmp_path / orchestrator.ERROR_PATH)
    assert rows[0]["automatic_resume"] is True


def _signed_progress(**fields):
    value = {
        "schema_version": "rankcloak-revision-confirmatory-progress-v1",
        **fields,
    }
    value["progress_sha256"] = orchestrator.canonical_json_sha256(value)
    return value


def _progress_process(*, code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def test_progress_refresh_uses_three_outer_attempts_with_two_four_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _signed_progress(status="fixture")
    race = _progress_process(
        code=1, stderr=orchestrator.PROGRESS_SOURCE_RACE_MESSAGE + "\n"
    )
    responses = [race, race, _progress_process(stdout=json.dumps(progress))]
    sleeps = []
    events = []
    monkeypatch.setattr(
        orchestrator.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0)
    )
    monkeypatch.setattr(orchestrator.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        orchestrator, "emit_event", lambda event, **fields: events.append((event, fields))
    )

    assert orchestrator.refresh_progress() == progress
    assert responses == []
    assert sleeps == [2.0, 4.0]
    assert events == [
        (
            "canonical_progress_unstable_source_race_recovered",
            {
                "source_races_recovered": 2,
                "last_source_race": orchestrator.PROGRESS_SOURCE_RACE_MESSAGE,
                "progress_refresh_outer_attempts": 3,
                "progress_refresh_backoff_seconds": [2.0, 4.0],
                "action_retry_count_changed": False,
            },
        )
    ]


def test_exact_progress_race_accepts_all_build_labels_only_for_normalized_in_root_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / orchestrator.RESULTS_ROOT
    results_root.mkdir(parents=True)
    source = results_root / "primary_v2/model/checkpoint.json"
    for label in orchestrator.PROGRESS_SOURCE_RACE_LABELS:
        line = (
            orchestrator.PROGRESS_SOURCE_RACE_WRAPPER
            + "{} changed while being read: {}".format(label, source)
        )
        completed = _progress_process(code=1, stderr=line + "\n")
        assert orchestrator.canonical_progress_source_race(completed) == line


def test_exact_progress_race_accepts_unavailability_lineage_fixed_inner() -> None:
    inner = "An unavailability lineage source changed during the progress scan"
    line = orchestrator.PROGRESS_SOURCE_RACE_WRAPPER + inner
    completed = _progress_process(code=1, stderr=line + "\r\n")
    assert orchestrator.canonical_progress_source_race(completed) == line


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inner: str
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / orchestrator.RESULTS_ROOT
    results_root.mkdir(parents=True)
    inside = results_root / "checkpoint.json"
    outside = tmp_path / "outside.json"
    line = orchestrator.PROGRESS_SOURCE_RACE_WRAPPER + inner.format(
        inside=inside, outside=outside
    )
    completed = _progress_process(code=1, stderr=line + "\n")
    assert orchestrator.canonical_progress_source_race(completed) is None


def test_path_progress_race_rejects_symlink_resolution_outside_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / orchestrator.RESULTS_ROOT
    results_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = results_root / "link"
    link.symlink_to(outside, target_is_directory=True)
    source = link / "checkpoint.json"
    line = (
        orchestrator.PROGRESS_SOURCE_RACE_WRAPPER
        + "checkpoint changed while being read: {}".format(source)
    )
    completed = _progress_process(code=1, stderr=line + "\n")
    assert orchestrator.canonical_progress_source_race(completed) is None


def test_progress_refresh_defers_after_three_exact_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sleeps = []

    def run(*_args, **_kwargs):
        calls.append(True)
        return _progress_process(
            code=1, stderr=orchestrator.PROGRESS_SOURCE_RACE_MESSAGE + "\n"
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    monkeypatch.setattr(orchestrator.time, "sleep", sleeps.append)

    with pytest.raises(orchestrator.ProgressSourceRace, match="three|3 outer"):
        orchestrator.refresh_progress()

    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_progress_refresh_timeout_is_finite_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def run(*args, **kwargs):
        observed.append(kwargs["timeout"])
        raise orchestrator.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(orchestrator.subprocess, "run", run)

    with pytest.raises(orchestrator.ProgressRefreshTimeout, match="timed out"):
        orchestrator.refresh_progress()

    assert observed == [orchestrator.PROGRESS_SUBPROCESS_TIMEOUT_SECONDS]


@pytest.mark.parametrize(
    ("code", "stdout", "stderr"),
    [
        (2, "", "{canonical}\n"),
        (1, "{{}}\n", "{canonical}\n"),
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
    monkeypatch: pytest.MonkeyPatch, code, stdout, stderr
) -> None:
    completed = _progress_process(
        code=code,
        stdout=stdout.format(canonical=orchestrator.PROGRESS_SOURCE_RACE_MESSAGE),
        stderr=stderr.format(canonical=orchestrator.PROGRESS_SOURCE_RACE_MESSAGE),
    )
    sleeps = []
    monkeypatch.setattr(
        orchestrator.subprocess, "run", lambda *_args, **_kwargs: completed
    )
    monkeypatch.setattr(orchestrator.time, "sleep", sleeps.append)

    with pytest.raises(orchestrator.OrchestratorError) as caught:
        orchestrator.refresh_progress()

    assert not isinstance(caught.value, orchestrator.ProgressSourceRace)
    assert sleeps == []


@pytest.mark.parametrize(
    "deferral_type",
    [orchestrator.ProgressSourceRace, orchestrator.ProgressRefreshTimeout],
)
def test_attached_action_progress_deferral_preserves_retry_and_classification(
    monkeypatch: pytest.MonkeyPatch, deferral_type,
) -> None:
    action = next(
        action
        for action in orchestrator.build_gpu_actions()
        if action.kind == "runner"
    )
    retries = {action.action_id: 2}
    states = []
    events = []
    enforced = []
    budget = {
        "cumulative_actual_gpu_hours": 12.0,
        "revised_upper_gpu_hours": 150.0,
    }
    monkeypatch.setattr(orchestrator, "_gpu_compute_pids", lambda: {101})
    monkeypatch.setattr(orchestrator, "_process_cmdlines", lambda: {101: action.argv})
    monkeypatch.setattr(
        orchestrator,
        "refresh_progress",
        lambda: (_ for _ in ()).throw(deferral_type("fixture")),
    )
    monkeypatch.setattr(
        orchestrator, "published_progress_after_source_race", lambda: {"fixture": True}
    )
    monkeypatch.setattr(orchestrator, "calculate_budget", lambda *_args: budget)
    monkeypatch.setattr(orchestrator, "enforce_budget", enforced.append)
    monkeypatch.setattr(
        orchestrator, "write_state", lambda **kwargs: states.append(kwargs)
    )
    monkeypatch.setattr(
        orchestrator, "record_recoverable_error", lambda *_args: pytest.fail(
            "progress race entered the action-retry ledger"
        )
    )
    monkeypatch.setattr(
        orchestrator, "emit_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _seconds: None)

    assert orchestrator.wait_for_existing_gpu_occupancy(
        action,
        projection={},
        retries=retries,
        max_retries=2,
        poll_seconds=5,
    )

    assert retries == {action.action_id: 2}
    assert enforced == [budget]
    assert states[-1]["status"] == "attached_existing_gpu_action"
    assert states[-1]["active_pid"] == 101
    assert "unchanged" in states[-1]["message"]
    expected_event = (
        "canonical_progress_updater_timeout_deferred"
        if deferral_type is orchestrator.ProgressRefreshTimeout
        else "canonical_progress_unstable_source_race_deferred"
    )
    assert events[-1][0] == expected_event
    assert events[-1][1]["retry_count_unchanged"] == 2
    assert len(events) == 1


def test_managed_gpu_child_is_not_killed_or_retried_for_progress_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = orchestrator.Action(
        action_id="runner:fixture",
        stage="fixture",
        kind="runner",
        argv=("fixture-runner",),
        gpu=True,
    )
    progress = {"fixture": True}
    budget = {
        "cumulative_actual_gpu_hours": 12.0,
        "revised_upper_gpu_hours": 150.0,
    }

    class Process:
        pid = 4321
        returncode = 0

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    process = Process()
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1
        if refreshes == 1:
            return progress
        raise orchestrator.ProgressSourceRace("fixture")

    states = []
    events = []
    retries = {action.action_id: 1}
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "LOG_ROOT", Path("logs"))
    monkeypatch.setattr(orchestrator, "validate_gpu_action", lambda _action: None)
    monkeypatch.setattr(orchestrator, "refresh_progress", refresh)
    monkeypatch.setattr(
        orchestrator, "published_progress_after_source_race", lambda: progress
    )
    monkeypatch.setattr(orchestrator, "calculate_budget", lambda *_args: budget)
    monkeypatch.setattr(orchestrator, "enforce_budget", lambda _budget: None)
    monkeypatch.setattr(orchestrator, "ensure_gpu_idle", lambda: None)
    monkeypatch.setattr(orchestrator.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda *_args: pytest.fail("progress race killed the managed child"),
    )
    monkeypatch.setattr(
        orchestrator, "write_state", lambda **kwargs: states.append(kwargs)
    )
    monkeypatch.setattr(
        orchestrator, "emit_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _seconds: None)

    assert orchestrator.run_gpu_process(
        action,
        action.argv,
        projection={},
        retries=retries,
        poll_seconds=5,
    ) == (0, "")

    assert retries == {action.action_id: 1}
    assert states[-1]["status"] == "running"
    assert states[-1]["active_pid"] == process.pid
    assert any(
        event == "canonical_progress_unstable_source_race_deferred"
        for event, _fields in events
    )


@pytest.mark.parametrize(
    "deferral_type",
    [orchestrator.ProgressSourceRace, orchestrator.ProgressRefreshTimeout],
)
def test_full_orchestrator_continues_after_top_level_progress_deferral(
    monkeypatch: pytest.MonkeyPatch, deferral_type,
) -> None:
    progress = {"fixture": True}
    budget = {
        "cumulative_actual_gpu_hours": 12.0,
        "revised_upper_gpu_hours": 150.0,
    }
    calls = 0

    def operational_progress():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise deferral_type("fixture")
        return progress

    events = []
    states = []
    sleeps = []
    monkeypatch.setattr(orchestrator, "verify_projection", lambda: {})
    monkeypatch.setattr(orchestrator, "load_command_contract", lambda _path: {})
    monkeypatch.setattr(orchestrator, "build_gpu_actions", lambda: [])
    monkeypatch.setattr(orchestrator, "load_retry_counts", lambda: {})
    monkeypatch.setattr(orchestrator, "operational_progress", operational_progress)
    monkeypatch.setattr(
        orchestrator, "published_progress_after_source_race", lambda: progress
    )
    monkeypatch.setattr(orchestrator, "calculate_budget", lambda *_args: budget)
    monkeypatch.setattr(orchestrator, "enforce_budget", lambda _budget: None)
    monkeypatch.setattr(
        orchestrator, "emit_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(
        orchestrator, "primary_gate_status", lambda: (False, ["fixture_model"])
    )
    monkeypatch.setattr(
        orchestrator, "write_state", lambda **kwargs: states.append(kwargs)
    )
    monkeypatch.setattr(orchestrator.time, "sleep", sleeps.append)
    args = SimpleNamespace(
        command_contract=Path("fixture-contract.json"),
        max_retries_per_action=2,
        once=True,
        poll_seconds=5,
    )

    assert orchestrator.run_orchestrator(args) == 0
    assert calls == 2
    assert sleeps == [5]
    expected_event = (
        "canonical_progress_updater_timeout_deferred"
        if deferral_type is orchestrator.ProgressRefreshTimeout
        else "canonical_progress_unstable_source_race_deferred"
    )
    assert events == [(expected_event, events[0][1])]
    assert states[-1]["status"] == "waiting_for_primary_v2"


@pytest.mark.parametrize(
    "deferral",
    [
        orchestrator.ProgressSourceRace("fixture race"),
        orchestrator.ProgressRefreshTimeout("fixture timeout"),
    ],
)
def test_progress_deferral_enforces_ceiling_before_state_or_event(
    monkeypatch: pytest.MonkeyPatch, deferral,
) -> None:
    budget = {
        "cumulative_actual_gpu_hours": 165.0,
        "revised_upper_gpu_hours": 165.0,
    }
    monkeypatch.setattr(
        orchestrator, "published_progress_after_source_race", lambda: {"fixture": True}
    )
    monkeypatch.setattr(orchestrator, "calculate_budget", lambda *_args: budget)
    monkeypatch.setattr(
        orchestrator,
        "write_state",
        lambda **_kwargs: pytest.fail("state was deferred before ceiling enforcement"),
    )
    monkeypatch.setattr(
        orchestrator,
        "emit_event",
        lambda *_args, **_kwargs: pytest.fail(
            "deferral event was emitted before ceiling enforcement"
        ),
    )

    with pytest.raises(orchestrator.BudgetHalt, match="hard ceiling"):
        orchestrator.defer_progress_refresh(
            deferral=deferral,
            projection={},
            action=None,
            retries={},
            status="fixture",
            message="fixture",
        )


def test_successful_progress_recovery_is_cumulative_and_persistent_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", tmp_path)
    events = [
        {
            "at": "2026-08-10T00:00:00+00:00",
            "event": "canonical_progress_unstable_source_race_recovered",
            "source_races_recovered": 2,
            "last_source_race": "first",
        },
        {
            "at": "2026-08-10T00:01:00+00:00",
            "event": "canonical_progress_unstable_source_race_recovered",
            "source_races_recovered": 1,
            "last_source_race": "last",
        },
    ]
    _jsonl(tmp_path / orchestrator.EVENT_PATH, events)
    orchestrator.write_state(
        status="running",
        message="fixture",
        action=None,
        retries={},
        progress={},
        budget={
            "cumulative_actual_gpu_hours": 12.0,
            "revised_upper_gpu_hours": 150.0,
        },
    )

    state = json.loads((tmp_path / orchestrator.STATE_PATH).read_text(encoding="utf-8"))
    summary = state["progress_refresh_recovery"]
    assert summary["successful_outer_recovery_events"] == 2
    assert summary["unstable_source_races_recovered"] == 3
    assert summary["last_successful_outer_recovery"] == events[-1]
    assert state["retry_counts"] == {}


def test_gpu_occupancy_query_has_finite_timeout_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def run(*args, **kwargs):
        observed.append(kwargs["timeout"])
        raise orchestrator.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(orchestrator.subprocess, "run", run)

    with pytest.raises(orchestrator.InterfaceHalt, match="occupancy query timed out"):
        orchestrator._gpu_compute_pids()

    assert observed == [orchestrator.GPU_OCCUPANCY_TIMEOUT_SECONDS]
