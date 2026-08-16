from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_detector_cuda_budget_gate.py"
SPEC = importlib.util.spec_from_file_location("detector_cuda_budget_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate_builder
SPEC.loader.exec_module(gate_builder)


def _install_projection_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    process_wall_seconds: float,
) -> dict:
    tasks = []
    for split in range(28):
        tasks.extend(
            [
                {
                    "ordinal": split * 2,
                    "split_id": f"split-{split:02d}",
                    "detector_name": "published_textcnn_equivalent",
                    "train_row_count": 100,
                    "test_row_count": 20,
                },
                {
                    "ordinal": split * 2 + 1,
                    "split_id": f"split-{split:02d}",
                    "detector_name": "deberta_v3_base_classifier",
                    "train_row_count": 100,
                    "test_row_count": 20,
                },
            ]
        )
    policy = {
        "benchmark": {
            "projection_safety_multiplier": 1.5,
            "allowed_failed_fit_retry_count_per_architecture": 1,
        },
        "authorized_ceiling": {
            "gpu_hours": 165.0,
            "historical_actual_gpu_hours_floor": 62.4783840698,
        },
        "ceiling": {
            "next_fit_upper_seconds_by_detector": {
                "published_textcnn_equivalent": 900.0,
                "deberta_v3_base_classifier": 7200.0,
            }
        },
    }
    records = {
        index: {
            "benchmark_task_identity": tasks[index],
            "benchmark_sha256": str(index) * 64,
        }
        for index in (0, 1)
    }
    measurements = {
        index: {
            "initialization_and_preprocessing": 2.0,
            "training": process_wall_seconds - 8.0,
            "trained_state_hashing": 1.0,
            "evaluation": 2.0,
            "total": process_wall_seconds - 3.0,
            "fit_elapsed": process_wall_seconds - 2.0,
            "fit_non_model_analysis": 1.0,
            "checkpoint": 1.0,
            "invocation_overhead": 1.0,
            "benchmark_wall": process_wall_seconds,
            "process_wall": process_wall_seconds,
        }
        for index in (0, 1)
    }
    ledger = {
        "sources": [
            {
                "source_id": f"production_benchmark_task_{index}",
                "component": "detector_production_benchmark",
            }
            for index in (0, 1)
        ],
        "cumulative_elapsed_seconds": 3600.0,
        "ledger_sha256": "a" * 64,
        "sources_sha256": "b" * 64,
        "intervals_sha256": "c" * 64,
    }
    monkeypatch.setattr(gate_builder, "_verify_policy", lambda _path: policy)
    monkeypatch.setattr(
        gate_builder,
        "read_detector_failed_benchmark_attempt",
        lambda *_args, **_kwargs: {
            "benchmark_sha256": "f" * 64,
            "gpu_accounting": {
                "cumulative_elapsed_seconds": 111.000125
            },
        },
    )
    monkeypatch.setattr(
        gate_builder,
        "_verify_benchmark",
        lambda _path, *, task_index: (
            records[task_index],
            measurements[task_index],
        ),
    )
    monkeypatch.setattr(
        gate_builder,
        "_verify_plan",
        lambda _benchmarks: (tmp_path / "execution_plan.json", {"tasks": tasks}),
    )
    monkeypatch.setattr(
        gate_builder,
        "read_detector_gpu_accounting_ledger",
        lambda _path: ledger,
    )

    def identity(path: Path, *, content_key: str | None = None) -> dict:
        value = {
            "path": str(Path(path).resolve()),
            "sha256": "d" * 64,
            "size_bytes": 1,
        }
        if content_key is not None:
            value[content_key] = "e" * 64
        return value

    monkeypatch.setattr(gate_builder, "_identity", identity)
    return ledger


def test_projection_includes_historical_floor_ledger_retries_and_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_projection_fixtures(
        monkeypatch, tmp_path, process_wall_seconds=200.0
    )
    gate = gate_builder.build_gate(
        stage="post_benchmark_pre_reproducibility",
        policy_path=tmp_path / "policy.json",
        benchmark_paths=(
            tmp_path / "benchmark-0.json",
            tmp_path / "benchmark-1.json",
        ),
        ledger_path=tmp_path / "ledger.json",
    )
    projection = gate["projection"]
    assert projection["starting_cumulative_actual_gpu_hours"] == pytest.approx(
        62.4783840698 + 1.0 + 111.000125 / 3600.0
    )
    assert projection["projection_safety_multiplier"] == 1.5
    assert projection["projected_future_gpu_seconds"] > 0.0
    assert gate["decision"]["approved"] is True
    for architecture in gate_builder.ARCHITECTURES.values():
        row = projection["architecture_estimates"][architecture]
        assert row["remaining_production_fit_count"] == 27
        assert row["remaining_reproducibility_fit_count"] == 1
        assert row["failed_fit_retry_count"] == 1


def test_projection_closes_gate_when_conservative_total_exceeds_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_projection_fixtures(
        monkeypatch, tmp_path, process_wall_seconds=250_000.0
    )
    gate = gate_builder.build_gate(
        stage="post_benchmark_pre_reproducibility",
        policy_path=tmp_path / "policy.json",
        benchmark_paths=(
            tmp_path / "benchmark-0.json",
            tmp_path / "benchmark-1.json",
        ),
        ledger_path=tmp_path / "ledger.json",
    )
    assert gate["decision"] == {
        "approved": False,
        "reason": "benchmark_derived_projection_exceeds_hard_ceiling",
    }
    assert gate["projection"]["projected_cumulative_gpu_hours"] > 165.0


def test_post_benchmark_gate_refresh_accepts_valid_partial_equivalence_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _install_projection_fixtures(
        monkeypatch, tmp_path, process_wall_seconds=200.0
    )
    initial = gate_builder.build_gate(
        stage="post_benchmark_pre_reproducibility",
        policy_path=tmp_path / "policy.json",
        benchmark_paths=(
            tmp_path / "benchmark-0.json",
            tmp_path / "benchmark-1.json",
        ),
        ledger_path=tmp_path / "ledger.json",
    )
    ledger["sources"].append(
        {
            "source_id": "equivalence_cuda_task_0",
            "component": "detector_device_equivalence_cuda",
        }
    )
    ledger["cumulative_elapsed_seconds"] += 20.0
    refreshed = gate_builder.build_gate(
        stage="post_benchmark_pre_reproducibility",
        policy_path=tmp_path / "policy.json",
        benchmark_paths=(
            tmp_path / "benchmark-0.json",
            tmp_path / "benchmark-1.json",
        ),
        ledger_path=tmp_path / "ledger.json",
    )
    before = initial["projection"]["architecture_estimates"]
    after = refreshed["projection"]["architecture_estimates"]
    textcnn = "published_textcnn_equivalent"
    assert before[textcnn]["remaining_checkpoint_export_count"] == 1
    assert after[textcnn]["remaining_checkpoint_export_count"] == 0
    assert after[textcnn]["checkpoint_export_allowance_seconds"] == 0.0
    assert after[textcnn]["remaining_reproducibility_fit_count"] == 1
    assert after["deberta_v3_base_classifier"][
        "remaining_checkpoint_export_count"
    ] == 1
    assert refreshed["decision"]["approved"] is True


def test_post_benchmark_gate_rejects_repeat_without_checkpoint_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _install_projection_fixtures(
        monkeypatch, tmp_path, process_wall_seconds=200.0
    )
    ledger["sources"].append(
        {
            "source_id": "equivalence_cuda_repeat_task_0",
            "component": "detector_device_equivalence_cuda_repeat",
        }
    )
    with pytest.raises(
        gate_builder.RevisionDetectionError,
        match="repeat before its checkpoint export",
    ):
        gate_builder.build_gate(
            stage="post_benchmark_pre_reproducibility",
            policy_path=tmp_path / "policy.json",
            benchmark_paths=(
                tmp_path / "benchmark-0.json",
                tmp_path / "benchmark-1.json",
            ),
            ledger_path=tmp_path / "ledger.json",
        )


def test_cli_publishes_unapproved_gate_and_returns_budget_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        stage="post_benchmark_pre_reproducibility",
        policy=tmp_path / "policy.json",
        benchmark=[],
        gpu_ledger=tmp_path / "ledger.json",
        output=tmp_path / "gate.json",
        check=False,
    )
    rejected = {
        "gate_stage": "post_benchmark_pre_reproducibility",
        "gate_sha256": "a" * 64,
        "decision": {"approved": False},
    }
    written = []
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(gate_builder, "build_argument_parser", lambda: parser)
    monkeypatch.setattr(gate_builder, "build_gate", lambda **_kwargs: rejected)
    monkeypatch.setattr(
        gate_builder,
        "atomic_write_json",
        lambda path, value: written.append((path, value)),
    )
    monkeypatch.setattr(
        gate_builder,
        "read_gate",
        lambda *_args, **_kwargs: pytest.fail(
            "an unapproved gate must not be accepted by read_gate"
        ),
    )

    assert gate_builder.main() == 3
    assert written == [
        (
            args.output.resolve().parent
            / "budget_gate_history"
            / "post_benchmark_pre_reproducibility.{}.json".format(
                rejected["gate_sha256"]
            ),
            rejected,
        ),
        (args.output.resolve(), rejected),
    ]
    assert json.loads(capsys.readouterr().out) == rejected


def test_signed_gate_rejects_changed_input_bytes(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "policy",
            "execution_plan",
            "failed_benchmark_attempt",
            "gpu_ledger",
            "benchmark",
        )
    }
    for name, path in paths.items():
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
    inputs = {
        "policy": gate_builder._identity(paths["policy"]),
        "execution_plan": gate_builder._identity(paths["execution_plan"]),
        "failed_benchmark_attempt": gate_builder._identity(
            paths["failed_benchmark_attempt"]
        ),
        "gpu_ledger": gate_builder._identity(paths["gpu_ledger"]),
        "benchmarks": {
            "published_textcnn_equivalent": gate_builder._identity(
                paths["benchmark"]
            )
        },
    }
    projection = {
        "historical_actual_gpu_hours_floor": 62.4783840698,
        "hard_ceiling_gpu_hours": 165.0,
        "projected_cumulative_gpu_hours": 64.0,
    }
    gate = {
        "schema_version": gate_builder.SCHEMA,
        "created_at_utc": "2026-08-15T00:00:00+00:00",
        "gate_stage": "post_benchmark_pre_reproducibility",
        "inputs": inputs,
        "inputs_sha256": canonical_json_sha256(inputs),
        "projection": projection,
        "projection_sha256": canonical_json_sha256(projection),
        "decision": {"approved": True, "reason": "fixture"},
    }
    gate["gate_sha256"] = canonical_json_sha256(gate)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    assert gate_builder.read_gate(gate_path)["decision"]["approved"] is True

    paths["benchmark"].write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(
        gate_builder.RevisionDetectionError,
        match="input bytes changed",
    ):
        gate_builder.read_gate(gate_path)
