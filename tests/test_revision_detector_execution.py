import json
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

import rankcloak.revision_detector_execution as detector_execution

from rankcloak.revision_detection import (
    DetectorSplit,
    PreparedDetectorSuite,
    RevisionDetectionError,
    assemble_prepared_detector_result,
    normalize_detector_frame,
    prepare_revision_detector_suite,
    run_prepared_detector_fit,
    run_revision_detector_suite,
)
from rankcloak.revision_detector_execution import (
    DetectorExecutionContext,
    atomic_write_json,
    build_fit_tasks,
    canonical_json_sha256,
    compare_detector_fit_outputs,
    detector_finalization_paths,
    detector_execution_lease,
    execute_checkpointed_detector_suite,
    evaluate_detector_device_equivalence,
    file_sha256,
    finalize_detector_candidate_from_closed_status,
    read_detector_gpu_accounting_ledger,
    read_detector_gpu_ledger_incorporation_marker,
    update_detector_gpu_accounting_ledger,
    verify_status_file,
    write_detector_device_equivalence_report,
    write_detector_equivalence_fit_artifact,
    write_detector_finalization_candidate,
)


def detector_frame(group_count=8):
    rows = []
    for group_index in range(group_count):
        for label in (0, 1):
            rows.append(
                {
                    "row_id": "row-{}-{}".format(group_index, label),
                    "text": "{} unique {} {}".format(
                        "ordinary cover" if label == 0 else "forced stego",
                        group_index,
                        label,
                    ),
                    "label": label,
                    "payload_group_id": "payload-{}".format(group_index),
                    "prompt_template_id": "template-{}".format(group_index % 2),
                    "model_id": "model-{}".format(group_index % 2),
                    "codec_id": "codec-{}".format(group_index % 2),
                }
            )
    return pd.DataFrame(rows)


def smoke_config(detector_count=2):
    detectors = [
        {
            "name": "smoke-{}".format(index),
            "kind": "hashed_ngram_smoke",
            "fallback_maximum_features": 128,
            "fallback_maximum_iterations": 50,
        }
        for index in range(detector_count)
    ]
    return {
        "schema_version": "rankcloak-revision-detectors-v1",
        "seed": 991,
        "columns": {},
        "decision_threshold": 0.5,
        "splits": {
            "regimes": ["matched"],
            "matched_test_fraction": 0.25,
            "assert_text_hash_disjoint": True,
            "minimum_train_rows": 4,
            "minimum_test_rows": 2,
            "fail_on_skipped_split": True,
        },
        "bootstrap": {"resamples": 10, "smoke_resamples": 10},
        "detectors": detectors,
    }


def context(tmp_path, *, resume=True):
    return DetectorExecutionContext(
        output_dir=tmp_path / "final",
        checkpoint_dir=tmp_path / "checkpoints",
        status_file=tmp_path / "status.json",
        fit_permit_receipt_dir=(
            tmp_path / "checkpoints" / "fit_permit_receipts"
        ),
        device="cpu",
        gpu_uuid=None,
        workers=1,
        lineage={
            "input_sha256": "a" * 64,
            "preprocessing_manifest_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "confirmatory_plan_sha256": "d" * 64,
        },
        source={"files_sha256": "e" * 64},
        resume=resume,
        heartbeat_seconds=3600.0,
    )


def signed_document(value, field):
    signed = dict(value)
    signed[field] = canonical_json_sha256(signed)
    return signed


def gpu_accounting(offset_seconds, *, duration_seconds=10.0, pid=None):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    started = datetime.fromtimestamp(
        start + float(offset_seconds), tz=timezone.utc
    )
    completed = datetime.fromtimestamp(
        start + float(offset_seconds) + float(duration_seconds),
        tz=timezone.utc,
    )
    interval = {
        "pid": int(1000 + offset_seconds if pid is None else pid),
        "process_start_ticks": int(2000 + offset_seconds),
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "last_observed_at_utc": completed.isoformat(),
        "elapsed_seconds": float(duration_seconds),
        "derivation_policy": "detector_process_wall_span_v1",
    }
    return {
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "intervals": [interval],
        "cumulative_elapsed_seconds": float(duration_seconds),
        "derivation_policy": (
            "nonoverlapping_detector_process_wall_intervals_v1"
        ),
    }


def finalization_fixture(tmp_path, name, *, offset_seconds=0.0):
    root = tmp_path / name
    checkpoint_dir = root / "checkpoints"
    child = root / "fit-child.json"
    atomic_write_json(child, {"sealed_scientific_child": name})
    output = root / "benchmark.json"
    run_identity_sha256 = canonical_json_sha256({"run": name})
    candidate_path, receipt_path = detector_finalization_paths(
        checkpoint_dir,
        kind="benchmark_artifact",
        requested_output_path=output,
        task_index=0,
        role=name,
    )
    payload = {
        "schema_version": "rankcloak-revision-detector-benchmark-v1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "scientific_value": "sealed-" + name,
    }
    candidate = write_detector_finalization_candidate(
        candidate_path,
        kind="benchmark_artifact",
        run_identity_sha256=run_identity_sha256,
        payload=payload,
        output_files={
            "fit_child": {
                "path": str(child.resolve()),
                "sha256": file_sha256(child),
                "size_bytes": int(child.stat().st_size),
            }
        },
        requested_output_path=output,
    )
    accounting = gpu_accounting(offset_seconds)
    status_file = root / "status.json"
    status = signed_document(
        {
            "schema_version": "rankcloak-revision-detector-status-v1",
            "updated_at_utc": accounting["intervals"][-1][
                "completed_at_utc"
            ],
            "state": "supervisor_observed_process_exit",
            "device": "cuda:0",
            "gpu_uuid": "GPU-fixture",
            "run_identity_sha256": run_identity_sha256,
            "gpu_accounting": accounting,
        },
        "status_sha256",
    )
    atomic_write_json(status_file, status)
    return {
        "candidate": candidate,
        "candidate_path": candidate_path,
        "receipt_path": receipt_path,
        "status_file": status_file,
        "output": output,
        "accounting": accounting,
        "run_identity_sha256": run_identity_sha256,
    }


def supervisor_finalize_equivalence_fixture(
    tmp_path,
    *,
    role,
    task,
    metric,
    predictions,
    policy_identity,
    offset_seconds,
):
    root = tmp_path / role
    child = root / "sealed-fit.json"
    atomic_write_json(child, {"role": role, "sealed": True})
    output = root / "artifact.json"
    run_identity_sha256 = canonical_json_sha256({"equivalence_role": role})
    provenance = {
        "environment_sha256": "2" * 64,
        "policy_identity": policy_identity,
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "gpu_accounting": "supervisor_closes_after_exit",
    }
    payload = {
        "schema_version": (
            "rankcloak-revision-detector-equivalence-fit-artifact-v1"
        ),
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "role": role,
        "task_index": int(task["ordinal"]),
        "task_identity": task,
        "task_identity_sha256": canonical_json_sha256(task),
        "metric": metric,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "provenance": provenance,
        "provenance_sha256": canonical_json_sha256(provenance),
    }
    checkpoint_dir = root / "checkpoints"
    candidate_path, receipt_path = detector_finalization_paths(
        checkpoint_dir,
        kind="equivalence_artifact",
        requested_output_path=output,
        task_index=int(task["ordinal"]),
        role=role,
    )
    write_detector_finalization_candidate(
        candidate_path,
        kind="equivalence_artifact",
        run_identity_sha256=run_identity_sha256,
        payload=payload,
        output_files={
            "sealed_fit": {
                "path": str(child.resolve()),
                "sha256": file_sha256(child),
                "size_bytes": int(child.stat().st_size),
            }
        },
        requested_output_path=output,
    )
    accounting = gpu_accounting(offset_seconds)
    status_file = root / "status.json"
    atomic_write_json(
        status_file,
        signed_document(
            {
                "schema_version": "rankcloak-revision-detector-status-v1",
                "updated_at_utc": accounting["intervals"][0][
                    "completed_at_utc"
                ],
                "state": "supervisor_observed_process_exit",
                "device": "cuda:0",
                "gpu_uuid": "GPU-fixture",
                "run_identity_sha256": run_identity_sha256,
                "gpu_accounting": accounting,
            },
            "status_sha256",
        ),
    )
    finalize_detector_candidate_from_closed_status(
        candidate_path,
        closed_status_file=status_file,
        terminal_receipt_path=receipt_path,
    )
    return output


def fake_fit(prepared, split, detector_config):
    detector_name = str(detector_config["name"])
    task = next(
        value
        for value in build_fit_tasks(prepared)
        if value.split.split_id == split.split_id
        and value.detector_name == detector_name
    )
    test = prepared.normalized_frame.iloc[list(split.test_indices)]
    train = prepared.normalized_frame.iloc[list(split.train_indices)]
    metadata = {
        "model_state_hash_algorithm": "rankcloak-torch-state-v1",
        "model_state_sha256": "f" * 64,
        "model_state_schema_hash_algorithm": "rankcloak-torch-state-schema-v1",
        "model_state_schema_sha256": "e" * 64,
    }
    if str(detector_config["kind"]) == "pretrained_transformer":
        metadata["model_artifact_set_sha256"] = "d" * 64
    metric = {
        "split_id": split.split_id,
        "regime": split.regime,
        "held_out_column": split.held_out_column,
        "held_out_value": split.held_out_value,
        "detector_name": detector_name,
        "requested_kind": str(detector_config["kind"]),
        "implementation_kind": str(detector_config["kind"]),
        "implementation_status": "complete",
        "train_rows": len(split.train_indices),
        "test_rows": len(split.test_indices),
        "train_payload_groups": int(train["payload_group_id"].nunique()),
        "purged_train_rows": int(split.purged_train_rows),
        "decision_threshold": prepared.threshold,
        "seed": task.seed,
        "notes": "fixture",
        "model_state_sha256": metadata["model_state_sha256"],
        "model_state_hash_algorithm": metadata["model_state_hash_algorithm"],
        "model_artifact_set_sha256": metadata.get("model_artifact_set_sha256"),
        "implementation_metadata_json": json.dumps(metadata, sort_keys=True),
        "bootstrap_unit": "payload_group_id",
        "bootstrap_resamples_requested": prepared.bootstrap_resamples,
        "test_payload_groups": int(test["payload_group_id"].nunique()),
        "roc_auc": 1.0,
    }
    predictions = []
    for position, (_, row) in enumerate(test.iterrows()):
        score = 0.2 if int(row["label"]) == 0 else 0.8
        predictions.append(
            {
                "split_id": split.split_id,
                "regime": split.regime,
                "held_out_value": split.held_out_value,
                "detector_name": detector_name,
                "requested_kind": str(detector_config["kind"]),
                "implementation_kind": str(detector_config["kind"]),
                "implementation_status": "complete",
                "row_id": row["row_id"],
                "payload_group_id": row["payload_group_id"],
                "prompt_template_id": row["prompt_template_id"],
                "model_id": row["model_id"],
                "codec_id": row["codec_id"],
                "label": int(row["label"]),
                "score": score,
                "prediction": int(score >= prepared.threshold),
            }
        )
    return metric, predictions


def prepared_56_fit_fixture():
    frame = normalize_detector_frame(detector_frame(group_count=4))
    splits = [
        DetectorSplit(
            split_id="split-{:02d}".format(index),
            regime="fixture",
            train_indices=(0, 1, 2, 3),
            test_indices=(4, 5, 6, 7),
        )
        for index in range(28)
    ]
    return PreparedDetectorSuite(
        normalized_frame=frame,
        splits=splits,
        skipped_splits=[],
        detector_configs=[
            {"name": "detector-a", "kind": "text_cnn", "epochs": 10},
            {
                "name": "detector-b",
                "kind": "pretrained_transformer",
                "epochs": 3,
            },
        ],
        seed=20260808,
        bootstrap_resamples=2000,
        threshold=0.5,
        smoke=False,
        allow_model_downloads=False,
        run_metadata={"execution_mode": "confirmatory"},
    )


def test_prepared_single_fit_wrapper_is_exactly_legacy_equivalent():
    frame = detector_frame()
    config = smoke_config()
    legacy = run_revision_detector_suite(frame, config, smoke=True)
    prepared = prepare_revision_detector_suite(frame, config, smoke=True)
    metrics = []
    predictions = []
    for split in prepared.splits:
        for detector in prepared.detector_configs:
            metric, rows = run_prepared_detector_fit(prepared, split, detector)
            metrics.append(metric)
            predictions.extend(rows)
    wrapped = assemble_prepared_detector_result(prepared, metrics, predictions)
    assert wrapped.metrics.to_dict(orient="records") == legacy.metrics.to_dict(
        orient="records"
    )
    assert wrapped.predictions.to_dict(
        orient="records"
    ) == legacy.predictions.to_dict(orient="records")
    report = compare_detector_fit_outputs(
        legacy.metrics.iloc[0].to_dict(),
        legacy.predictions[
            legacy.predictions["detector_name"] == legacy.metrics.iloc[0]["detector_name"]
        ].to_dict(orient="records"),
        wrapped.metrics.iloc[0].to_dict(),
        wrapped.predictions[
            wrapped.predictions["detector_name"] == wrapped.metrics.iloc[0]["detector_name"]
        ].to_dict(orient="records"),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    assert report["equivalent"] is True
    assert report["max_score_absolute_difference"] == 0.0


def test_interrupted_resume_skips_valid_fit_and_status_is_signed(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    first_calls = []

    def first_runner(*args):
        first_calls.append(args[1].split_id + "/" + args[2]["name"])
        return fake_fit(*args)

    first = execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=first_runner,
        stop_after_new_fits=1,
    )
    assert first.result is None
    assert first.completed_fit_count == 1
    assert first.total_fit_count == 2
    assert len(first_calls) == 1
    assert not (tmp_path / "final" / "detector_run_manifest.json").exists()
    status = verify_status_file(tmp_path / "status.json")
    assert status["completed_fit_count"] == 1
    assert status["total_fit_count"] == 2
    assert status["current_fit"] is None
    assert status["checkpoint_cumulative_fit_seconds"] >= 0.0
    assert status["process_elapsed_seconds"] >= 0.0
    unsigned = dict(status)
    observed_hash = unsigned.pop("status_sha256")
    assert observed_hash == canonical_json_sha256(unsigned)

    resumed_calls = []

    def resumed_runner(*args):
        resumed_calls.append(args[1].split_id + "/" + args[2]["name"])
        return fake_fit(*args)

    resumed = execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=resumed_runner,
    )
    assert resumed.result is not None
    assert resumed.completed_fit_count == 2
    assert resumed.resumed_fit_count == 1
    assert len(resumed_calls) == 1
    assert resumed.result.metrics["detector_name"].tolist() == [
        "smoke-0",
        "smoke-1",
    ]


def test_checkpoint_identity_is_stable_across_publication_gate_evidence(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    before = context(tmp_path)
    before.lineage.update(
        {
            "required_equivalence_reports": [],
            "pre_final_gpu_accounting_ledger_path": None,
            "pre_final_gpu_accounting_ledger": None,
        }
    )
    first = execute_checkpointed_detector_suite(
        prepared,
        before,
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    after = context(tmp_path)
    after.lineage.update(
        {
            "required_equivalence_reports": [{"report_sha256": "1" * 64}],
            "pre_final_gpu_accounting_ledger_path": "/sealed/ledger.json",
            "pre_final_gpu_accounting_ledger": {"ledger_sha256": "2" * 64},
        }
    )
    called = False

    def forbidden(*args):
        nonlocal called
        called = True
        raise AssertionError("valid benchmark fit must be reused")

    resumed = execute_checkpointed_detector_suite(
        prepared,
        after,
        fit_runner=forbidden,
        benchmark_task_index=0,
    )
    assert resumed.run_identity_sha256 == first.run_identity_sha256
    assert resumed.resumed_fit_count == 1
    assert called is False


def test_corrupt_committed_task_fails_closed_without_rerun(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    predictions = tmp_path / "checkpoints" / "fits" / "0000" / "predictions.json"
    predictions.write_text(predictions.read_text(encoding="utf-8") + " ", encoding="utf-8")
    called = False

    def forbidden(*args):
        nonlocal called
        called = True
        raise AssertionError("corrupt committed checkpoint must fail before a fit")

    with pytest.raises(RevisionDetectionError, match="child hash/size differs"):
        execute_checkpointed_detector_suite(
            prepared, context(tmp_path), fit_runner=forbidden
        )
    assert called is False


def test_torn_fit_children_and_atomic_temps_are_preserved_then_recomputed(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    task_dir = tmp_path / "checkpoints" / "fits" / "0000"
    task_dir.mkdir(parents=True)
    partial = task_dir / "predictions.json"
    temporary = task_dir / ".tmp-metric.json-deadbeef"
    partial.write_bytes(b'{"partial":true}\n')
    temporary.write_bytes(b"torn atomic temp")
    outcome = execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    assert outcome.completed_fit_count == 1
    recovered = [
        row
        for row in outcome.recovered_errors
        if row["type"] == "orphaned_incomplete_fit_checkpoint"
    ]
    assert len(recovered) == 1
    assert {row["source_name"] for row in recovered[0]["files"]} == {
        "predictions.json",
        ".tmp-metric.json-deadbeef",
    }
    for row in recovered[0]["files"]:
        preserved = os.path.realpath(row["preserved_path"])
        assert os.path.isfile(preserved)
        assert file_sha256(preserved) == row["sha256"]
    assert {path.name for path in task_dir.iterdir()} == {
        "metric.json",
        "predictions.json",
        "manifest.json",
    }


def test_self_consistently_rehashed_semantic_checkpoint_tamper_fails_closed(
    tmp_path,
):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    task_dir = tmp_path / "checkpoints" / "fits" / "0000"
    prediction_path = task_dir / "predictions.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    label_index = payload["columns"].index("payload_group_id")
    payload["rows"][0][label_index] = "self-consistent-tamper"
    atomic_write_json(prediction_path, payload)
    manifest_path = task_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["children"]["predictions.json"].update(
        {
            "sha256": file_sha256(prediction_path),
            "size_bytes": int(prediction_path.stat().st_size),
        }
    )
    manifest["children_sha256"] = canonical_json_sha256(manifest["children"])
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    called = False

    def forbidden(*args):
        nonlocal called
        called = True
        raise AssertionError("semantic tamper must fail before a fit")

    with pytest.raises(RevisionDetectionError, match="prediction checkpoint"):
        execute_checkpointed_detector_suite(
            prepared, context(tmp_path), fit_runner=forbidden
        )
    assert called is False


def test_frozen_28_split_two_detector_plan_aggregates_56_in_order(tmp_path):
    prepared = prepared_56_fit_fixture()
    tasks = build_fit_tasks(prepared)
    assert len(tasks) == 56
    assert [task.ordinal for task in tasks] == list(range(56))
    assert [(task.split.split_id, task.detector_name) for task in tasks[:4]] == [
        ("split-00", "detector-a"),
        ("split-00", "detector-b"),
        ("split-01", "detector-a"),
        ("split-01", "detector-b"),
    ]
    outcome = execute_checkpointed_detector_suite(
        prepared, context(tmp_path), fit_runner=fake_fit
    )
    assert outcome.completed_fit_count == 56
    assert outcome.total_fit_count == 56
    assert outcome.result is not None
    assert len(outcome.result.splits) == 28
    assert len(outcome.result.metrics) == 56
    assert outcome.result.run_metadata == {
        "execution_mode": "confirmatory",
        "expected_detector_split_executions": 56,
        "complete_detector_split_executions": 56,
        "confirmatory_complete": True,
    }
    assert list(
        zip(
            outcome.result.metrics["split_id"],
            outcome.result.metrics["detector_name"],
        )
    ) == [(task.split.split_id, task.detector_name) for task in tasks]
    committed = sorted((tmp_path / "checkpoints" / "fits").glob("*/manifest.json"))
    assert len(committed) == 56


def test_worker_count_fails_closed(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(detector_count=1), smoke=True
    )
    unsafe = context(tmp_path)
    unsafe.workers = 2
    with pytest.raises(RevisionDetectionError, match="Only --workers 1"):
        execute_checkpointed_detector_suite(prepared, unsafe, fit_runner=fake_fit)


def test_checkpoint_root_has_nonblocking_exclusive_lease(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(detector_count=1), smoke=True
    )
    execution_context = context(tmp_path)
    with detector_execution_lease(execution_context.checkpoint_dir):
        with pytest.raises(RevisionDetectionError, match="execution lease"):
            execute_checkpointed_detector_suite(
                prepared, execution_context, fit_runner=fake_fit
            )


def test_checkpoint_root_lease_rejects_a_second_process(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    with detector_execution_lease(checkpoint_dir):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from rankcloak.revision_detector_execution import "
                    "detector_execution_lease; "
                    "\nwith detector_execution_lease(Path(r'{}')): pass"
                ).format(checkpoint_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    assert completed.returncode != 0
    assert "execution lease" in completed.stderr


def test_corrupt_cuda_status_never_discards_prior_gpu_charge(tmp_path, monkeypatch):
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    execution_context = context(tmp_path)
    execution_context.device = "cuda:0"
    execution_context.gpu_uuid = gpu_uuid
    first = execute_checkpointed_detector_suite(
        prepared,
        execution_context,
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    assert first.completed_fit_count == 1
    status_path = tmp_path / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["gpu_accounting"]["cumulative_elapsed_seconds"] >= 0.0
    status["status_sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RevisionDetectionError, match="refusing to discard prior GPU charge"):
        execute_checkpointed_detector_suite(
            prepared, execution_context, fit_runner=fake_fit
        )


def test_missing_cuda_status_with_fit_state_fails_closed(tmp_path, monkeypatch):
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000002"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    execution_context = context(tmp_path)
    execution_context.device = "cuda:0"
    execution_context.gpu_uuid = gpu_uuid
    execute_checkpointed_detector_suite(
        prepared,
        execution_context,
        fit_runner=fake_fit,
        stop_after_new_fits=1,
    )
    (tmp_path / "status.json").unlink()
    with pytest.raises(RevisionDetectionError, match="prior GPU charge"):
        execute_checkpointed_detector_suite(
            prepared, execution_context, fit_runner=fake_fit
        )


def test_benchmark_task_index_runs_only_selected_frozen_task(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(), smoke=True
    )
    calls = []

    def record(*args):
        calls.append(args[2]["name"])
        return fake_fit(*args)

    outcome = execute_checkpointed_detector_suite(
        prepared,
        context(tmp_path),
        fit_runner=record,
        benchmark_task_index=1,
    )
    assert outcome.stopped_at_fit_boundary is True
    assert outcome.completed_fit_count == 1
    assert calls == ["smoke-1"]
    assert (tmp_path / "checkpoints" / "fits" / "0001" / "manifest.json").is_file()
    assert not (tmp_path / "checkpoints" / "fits" / "0000").exists()


def test_fit_waits_for_exact_one_use_ceiling_permit(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(detector_count=1), smoke=True
    )
    execution_context = context(tmp_path)
    execution_context.execution_policy = {
        "ceiling": {
            "next_fit_upper_seconds_by_detector": {"smoke-0": 123.0}
        }
    }
    execution_context.execution_policy_path = tmp_path / "policy.json"
    execution_context.execution_policy_path.write_text("{}", encoding="utf-8")
    execution_context.execution_policy_sha256 = "a" * 64
    execution_context.fit_permit_file = tmp_path / "permit.json"
    execution_context.require_fit_permit = True
    child = os.fork()
    if child == 0:
        try:
            outcome = execute_checkpointed_detector_suite(
                prepared,
                execution_context,
                fit_runner=fake_fit,
                stop_after_new_fits=1,
            )
            os._exit(0 if outcome.completed_fit_count == 1 else 7)
        except BaseException:
            os._exit(8)
    try:
        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            try:
                status = verify_status_file(tmp_path / "status.json")
            except RevisionDetectionError:
                time.sleep(0.05)
                continue
            if status["state"] == "awaiting_fit_ceiling_gate":
                break
            time.sleep(0.05)
        assert status is not None
        assert status["state"] == "awaiting_fit_ceiling_gate"
        assert status["next_fit_upper_seconds"] == 123.0
        assert not (tmp_path / "checkpoints" / "fits" / "0000").exists()
        permit = {
            "schema_version": "rankcloak-revision-detector-fit-permit-v1",
            "run_identity_sha256": status["run_identity_sha256"],
            "task_identity_sha256": status["next_fit"]["task_identity_sha256"],
            "fit_gate_nonce": status["fit_gate_nonce"],
            "invocation_pid": status["pid"],
            "invocation_start_ticks": status["process_start_ticks"],
            "next_fit_upper_seconds": 123.0,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        permit["permit_sha256"] = canonical_json_sha256(permit)
        execution_context.fit_permit_file.write_text(
            json.dumps(permit), encoding="utf-8"
        )
        _, wait_status = os.waitpid(child, 0)
        child = None
        assert os.waitstatus_to_exitcode(wait_status) == 0
        assert not execution_context.fit_permit_file.exists()
        assert (tmp_path / "checkpoints" / "fits" / "0000" / "manifest.json").is_file()
    finally:
        if child is not None:
            os.kill(child, 9)
            os.waitpid(child, 0)


def test_stale_prior_invocation_permit_is_quarantined_and_regated(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(detector_count=1), smoke=True
    )
    execution_context = context(tmp_path)
    execution_context.execution_policy = {
        "ceiling": {
            "next_fit_upper_seconds_by_detector": {"smoke-0": 123.0}
        }
    }
    execution_context.execution_policy_path = tmp_path / "policy.json"
    execution_context.execution_policy_path.write_text("{}", encoding="utf-8")
    execution_context.execution_policy_sha256 = "a" * 64
    execution_context.fit_permit_file = tmp_path / "permit.json"
    execution_context.require_fit_permit = True
    child = os.fork()
    if child == 0:
        try:
            outcome = execute_checkpointed_detector_suite(
                prepared,
                execution_context,
                fit_runner=fake_fit,
                stop_after_new_fits=1,
            )
            os._exit(0 if outcome.completed_fit_count == 1 else 7)
        except BaseException:
            os._exit(8)
    try:
        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            try:
                status = verify_status_file(tmp_path / "status.json")
            except RevisionDetectionError:
                time.sleep(0.05)
                continue
            if status["state"] == "awaiting_fit_ceiling_gate":
                break
            time.sleep(0.05)
        assert status is not None
        stale = {
            "schema_version": "rankcloak-revision-detector-fit-permit-v1",
            "run_identity_sha256": status["run_identity_sha256"],
            "task_identity_sha256": status["next_fit"]["task_identity_sha256"],
            "fit_gate_nonce": "f" * 64,
            "invocation_pid": status["pid"] + 1,
            "invocation_start_ticks": status["process_start_ticks"],
            "next_fit_upper_seconds": 123.0,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stale["permit_sha256"] = canonical_json_sha256(stale)
        execution_context.fit_permit_file.write_text(
            json.dumps(stale), encoding="utf-8"
        )
        recovered_root = tmp_path / "checkpoints" / "recovered_fit_permits"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not list(
            recovered_root.glob("stale-*.json") if recovered_root.exists() else []
        ):
            time.sleep(0.05)
        assert len(list(recovered_root.glob("stale-*.json"))) == 1
        assert not execution_context.fit_permit_file.exists()
        fresh = {
            "schema_version": "rankcloak-revision-detector-fit-permit-v1",
            "run_identity_sha256": status["run_identity_sha256"],
            "task_identity_sha256": status["next_fit"]["task_identity_sha256"],
            "fit_gate_nonce": status["fit_gate_nonce"],
            "invocation_pid": status["pid"],
            "invocation_start_ticks": status["process_start_ticks"],
            "next_fit_upper_seconds": 123.0,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        fresh["permit_sha256"] = canonical_json_sha256(fresh)
        execution_context.fit_permit_file.write_text(
            json.dumps(fresh), encoding="utf-8"
        )
        _, wait_status = os.waitpid(child, 0)
        child = None
        assert os.waitstatus_to_exitcode(wait_status) == 0
        final_status = verify_status_file(tmp_path / "status.json")
        assert any(
            row["type"] == "stale_fit_permit_quarantined"
            for row in final_status["recovered_errors"]
        )
    finally:
        if child is not None:
            os.kill(child, 9)
            os.waitpid(child, 0)


def test_signal_while_waiting_for_permit_stops_without_fit(tmp_path):
    prepared = prepare_revision_detector_suite(
        detector_frame(), smoke_config(detector_count=1), smoke=True
    )
    execution_context = context(tmp_path)
    execution_context.execution_policy = {
        "ceiling": {
            "next_fit_upper_seconds_by_detector": {"smoke-0": 123.0}
        }
    }
    execution_context.execution_policy_path = tmp_path / "policy.json"
    execution_context.execution_policy_path.write_text("{}", encoding="utf-8")
    execution_context.execution_policy_sha256 = "a" * 64
    execution_context.fit_permit_file = tmp_path / "permit.json"
    execution_context.require_fit_permit = True
    child = os.fork()
    if child == 0:
        try:
            outcome = execute_checkpointed_detector_suite(
                prepared, execution_context, fit_runner=fake_fit
            )
            os._exit(0 if outcome.stopped_at_fit_boundary else 7)
        except BaseException:
            os._exit(8)
    try:
        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            try:
                status = verify_status_file(tmp_path / "status.json")
            except RevisionDetectionError:
                time.sleep(0.05)
                continue
            if status["state"] == "awaiting_fit_ceiling_gate":
                break
            time.sleep(0.05)
        assert status is not None
        assert status["state"] == "awaiting_fit_ceiling_gate"
        os.kill(child, 15)
        _, wait_status = os.waitpid(child, 0)
        child = None
        assert os.waitstatus_to_exitcode(wait_status) == 0
        final_status = verify_status_file(tmp_path / "status.json")
        assert final_status["state"] == "stopped_at_fit_boundary"
        assert final_status["completed_fit_count"] == 0
        assert not (tmp_path / "checkpoints" / "fits" / "0000").exists()
    finally:
        if child is not None:
            os.kill(child, 9)
            os.waitpid(child, 0)


def test_numeric_equivalence_report_records_tolerance():
    reference_metric = {"split_id": "s", "roc_auc": 0.8, "seed": 3}
    candidate_metric = {"split_id": "s", "roc_auc": 0.8000005, "seed": 3}
    base = {
        "split_id": "s",
        "regime": "matched",
        "held_out_value": None,
        "detector_name": "d",
        "requested_kind": "text_cnn",
        "implementation_kind": "text_cnn",
        "implementation_status": "complete",
        "row_id": "r",
        "payload_group_id": "g",
        "prompt_template_id": "p",
        "model_id": "m",
        "codec_id": "c",
        "label": 1,
        "prediction": 1,
    }
    reference = [{**base, "score": 0.75}]
    candidate = [{**base, "score": 0.7500005}]
    report = compare_detector_fit_outputs(
        reference_metric,
        reference,
        candidate_metric,
        candidate,
        absolute_tolerance=1e-6,
        relative_tolerance=0.0,
    )
    assert report["equivalent"] is True
    assert report["max_score_absolute_difference"] == pytest.approx(5e-7)


def test_signed_predeclared_device_equivalence_report(tmp_path):
    task = {
        "ordinal": 0,
        "split_position": 0,
        "detector_position": 0,
        "split_id": "matched",
        "regime": "matched",
        "held_out_column": None,
        "held_out_value": None,
        "partition_policy": "full_held_out_condition",
        "purged_train_rows": 0,
        "excluded_held_out_rows": 0,
        "train_row_count": 6,
        "test_row_count": 2,
        "train_indices_sha256": "a" * 64,
        "test_indices_sha256": "b" * 64,
        "train_row_ids_ordered_sha256": "c" * 64,
        "test_row_ids_ordered_sha256": "d" * 64,
        "detector_name": "detector",
        "detector_kind": "text_cnn",
        "scientific_detector_config_sha256": "e" * 64,
        "seed": 5,
        "bootstrap_resamples": 10,
        "decision_threshold": 0.5,
    }
    metric_base = {
        "split_id": "matched",
        "regime": "matched",
        "held_out_column": None,
        "held_out_value": None,
        "detector_name": "detector",
        "requested_kind": "text_cnn",
        "implementation_kind": "text_cnn",
        "implementation_status": "complete",
        "train_rows": 6,
        "test_rows": 2,
        "train_payload_groups": 3,
        "purged_train_rows": 0,
        "decision_threshold": 0.5,
        "seed": 5,
        "bootstrap_unit": "payload_group_id",
        "bootstrap_resamples_requested": 10,
        "test_payload_groups": 1,
    }
    for name in (
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    ):
        metric_base[name] = 1.0
        metric_base[name + "_ci_low_95"] = 1.0
        metric_base[name + "_ci_high_95"] = 1.0
    metadata = {
        "model_state_sha256": "f" * 64,
        "model_state_schema_hash_algorithm": "rankcloak-torch-state-schema-v1",
        "model_state_schema_sha256": "1" * 64,
    }
    metric = {
        **metric_base,
        "implementation_metadata_json": json.dumps(metadata, sort_keys=True),
    }
    prediction_base = {
        "split_id": "matched",
        "regime": "matched",
        "held_out_value": None,
        "detector_name": "detector",
        "requested_kind": "text_cnn",
        "implementation_kind": "text_cnn",
        "implementation_status": "complete",
        "payload_group_id": "g",
        "prompt_template_id": "p",
        "model_id": "m",
        "codec_id": "c",
    }
    predictions = [
        {**prediction_base, "row_id": "r0", "label": 0, "score": 0.1, "prediction": 0},
        {**prediction_base, "row_id": "r1", "label": 1, "score": 0.9, "prediction": 1},
    ]
    policy_identity = {"policy_sha256": "3" * 64}
    paths = {}
    for role in ("cpu", "cuda", "cuda_repeat"):
        path = tmp_path / (role + ".json")
        if role == "cpu":
            write_detector_equivalence_fit_artifact(
                path,
                role=role,
                task_identity=task,
                metric=metric,
                predictions=predictions,
                provenance={
                    "environment_sha256": "2" * 64,
                    "policy_identity": policy_identity,
                    "device": "cpu",
                    "gpu_uuid": None,
                    "gpu_accounting": None,
                },
            )
        else:
            path = supervisor_finalize_equivalence_fixture(
                tmp_path,
                role=role,
                task=task,
                metric=metric,
                predictions=predictions,
                policy_identity=policy_identity,
                offset_seconds=0 if role == "cuda" else 20,
            )
        paths[role] = path
    policy = {
        "same_device_cuda": {
            "task_design_exact": True,
            "row_identity_order_labels_exact": True,
            "model_state_sha256_exact": True,
            "scores_exact": True,
            "metrics_exact": True,
            "predictions_exact": True,
        },
        "cpu_cuda": {
            "task_design_exact": True,
            "row_identity_order_labels_exact": True,
            "score_mae_max": 0.005,
            "score_max_abs_max": 0.05,
            "score_pearson_min": 0.999,
            "prediction_agreement_min": 0.995,
            "metric_max_abs_max": 0.02,
            "model_state_tensor_schema_exact": True,
            "model_state_hash_policy": "device_specific",
        },
    }
    report_path = tmp_path / "report.json"
    report = write_detector_device_equivalence_report(
        report_path,
        cpu_artifact_path=paths["cpu"],
        cuda_artifact_path=paths["cuda"],
        cuda_repeat_artifact_path=paths["cuda_repeat"],
        equivalence_policy=policy,
        policy_identity=policy_identity,
    )
    assert report["decision"]["equivalent"] is True
    assert report["decision"]["same_device_cuda"]["passed"] is True
    assert report["decision"]["cpu_cuda"]["passed"] is True
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    unsigned = dict(on_disk)
    claimed = unsigned.pop("report_sha256")
    assert claimed == canonical_json_sha256(unsigned)


def test_cpu_equivalence_self_finalizes_but_cuda_requires_supervisor(tmp_path):
    task = {
        "ordinal": 0,
        "split_position": 0,
        "detector_position": 0,
    }
    metric = {"split_id": "matched"}
    predictions = []
    cpu = write_detector_equivalence_fit_artifact(
        tmp_path / "cpu.json",
        role="cpu",
        task_identity=task,
        metric=metric,
        predictions=predictions,
        provenance={
            "device": "cpu",
            "gpu_uuid": None,
            "gpu_accounting": None,
        },
    )
    assert cpu["provenance"]["gpu_accounting"] is None
    with pytest.raises(
        RevisionDetectionError, match="supervisor terminal finalization"
    ):
        write_detector_equivalence_fit_artifact(
            tmp_path / "cuda.json",
            role="cuda",
            task_identity=task,
            metric=metric,
            predictions=predictions,
            provenance={
                "device": "cuda:0",
                "gpu_uuid": "GPU-fixture",
                "gpu_accounting": gpu_accounting(0),
            },
        )


def test_supervisor_finalization_is_idempotent_after_complete(tmp_path):
    fixture = finalization_fixture(tmp_path, "normal")
    first = finalize_detector_candidate_from_closed_status(
        fixture["candidate_path"],
        closed_status_file=fixture["status_file"],
        terminal_receipt_path=fixture["receipt_path"],
    )
    first_output = fixture["output"].read_bytes()
    first_receipt = fixture["receipt_path"].read_bytes()
    assert verify_status_file(fixture["status_file"])["state"] == "complete"
    second = finalize_detector_candidate_from_closed_status(
        fixture["candidate_path"],
        closed_status_file=fixture["status_file"],
        terminal_receipt_path=fixture["receipt_path"],
    )
    assert second["gpu_accounting"] == first["gpu_accounting"]
    assert fixture["output"].read_bytes() == first_output
    assert fixture["receipt_path"].read_bytes() == first_receipt


@pytest.mark.parametrize("failure_point", ["after_output", "after_receipt"])
def test_supervisor_finalization_recovers_each_crash_boundary(
    tmp_path, monkeypatch, failure_point
):
    fixture = finalization_fixture(tmp_path, failure_point)
    if failure_point == "after_output":
        original = detector_execution.write_detector_terminal_receipt

        def fail_after_output(*args, **kwargs):
            raise RuntimeError("simulated crash after output")

        monkeypatch.setattr(
            detector_execution, "write_detector_terminal_receipt", fail_after_output
        )
    else:
        original = detector_execution.mark_detector_execution_complete

        def fail_after_receipt(*args, **kwargs):
            raise RuntimeError("simulated crash after receipt")

        monkeypatch.setattr(
            detector_execution, "mark_detector_execution_complete", fail_after_receipt
        )
    with pytest.raises(RuntimeError, match="simulated crash"):
        finalize_detector_candidate_from_closed_status(
            fixture["candidate_path"],
            closed_status_file=fixture["status_file"],
            terminal_receipt_path=fixture["receipt_path"],
        )
    assert fixture["output"].is_file()
    if failure_point == "after_output":
        assert not fixture["receipt_path"].exists()
        monkeypatch.setattr(
            detector_execution, "write_detector_terminal_receipt", original
        )
    else:
        assert fixture["receipt_path"].is_file()
        assert verify_status_file(fixture["status_file"])["state"] != "complete"
        monkeypatch.setattr(
            detector_execution, "mark_detector_execution_complete", original
        )
    finalize_detector_candidate_from_closed_status(
        fixture["candidate_path"],
        closed_status_file=fixture["status_file"],
        terminal_receipt_path=fixture["receipt_path"],
    )
    assert verify_status_file(fixture["status_file"])["state"] == "complete"


def test_finalizer_rejects_runner_closed_awaiting_state(tmp_path):
    fixture = finalization_fixture(tmp_path, "not-observed")
    status = verify_status_file(fixture["status_file"])
    status.pop("status_sha256")
    status["state"] = "awaiting_supervisor_finalization"
    atomic_write_json(
        fixture["status_file"], signed_document(status, "status_sha256")
    )
    with pytest.raises(RevisionDetectionError, match="supervisor-observed"):
        finalize_detector_candidate_from_closed_status(
            fixture["candidate_path"],
            closed_status_file=fixture["status_file"],
            terminal_receipt_path=fixture["receipt_path"],
        )


def test_gpu_ledger_strictly_revalidates_receipts_and_deduplicates(tmp_path):
    fixture = finalization_fixture(tmp_path, "ledger-source")
    finalize_detector_candidate_from_closed_status(
        fixture["candidate_path"],
        closed_status_file=fixture["status_file"],
        terminal_receipt_path=fixture["receipt_path"],
    )
    ledger_path = tmp_path / "ledger.json"
    first = update_detector_gpu_accounting_ledger(
        ledger_path,
        source_id="production_benchmark_task_0",
        component="detector_production_benchmark",
        terminal_receipt_path=fixture["receipt_path"],
    )
    second = update_detector_gpu_accounting_ledger(
        ledger_path,
        source_id="production_benchmark_task_0",
        component="detector_production_benchmark",
        terminal_receipt_path=fixture["receipt_path"],
    )
    assert first == second
    assert len(read_detector_gpu_accounting_ledger(ledger_path)["intervals"]) == 1
    published = json.loads(fixture["output"].read_text(encoding="utf-8"))
    published["scientific_value"] = "self-consistently-rehashed-tamper"
    published.pop("benchmark_sha256")
    published["benchmark_sha256"] = canonical_json_sha256(published)
    atomic_write_json(fixture["output"], published)
    receipt = json.loads(fixture["receipt_path"].read_text(encoding="utf-8"))
    receipt["published_output"]["sha256"] = file_sha256(fixture["output"])
    receipt["published_output"]["size_bytes"] = int(fixture["output"].stat().st_size)
    receipt.pop("terminal_receipt_sha256")
    receipt["terminal_receipt_sha256"] = canonical_json_sha256(receipt)
    atomic_write_json(fixture["receipt_path"], receipt)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    identity = ledger["sources"][0]["terminal_receipt"]
    identity["sha256"] = file_sha256(fixture["receipt_path"])
    identity["size_bytes"] = int(fixture["receipt_path"].stat().st_size)
    identity["terminal_receipt_sha256"] = receipt["terminal_receipt_sha256"]
    ledger["sources_sha256"] = canonical_json_sha256(ledger["sources"])
    ledger.pop("ledger_sha256")
    ledger["ledger_sha256"] = canonical_json_sha256(ledger)
    atomic_write_json(ledger_path, ledger)
    with pytest.raises(RevisionDetectionError, match="differs from its candidate"):
        read_detector_gpu_accounting_ledger(ledger_path)


def test_gpu_ledger_rejects_overlapping_receipt_intervals(tmp_path):
    first = finalization_fixture(tmp_path, "overlap-a", offset_seconds=0)
    second = finalization_fixture(tmp_path, "overlap-b", offset_seconds=5)
    for fixture in (first, second):
        finalize_detector_candidate_from_closed_status(
            fixture["candidate_path"],
            closed_status_file=fixture["status_file"],
            terminal_receipt_path=fixture["receipt_path"],
        )
    ledger_path = tmp_path / "overlap-ledger.json"
    update_detector_gpu_accounting_ledger(
        ledger_path,
        source_id="production_benchmark_task_0",
        component="detector_production_benchmark",
        terminal_receipt_path=first["receipt_path"],
    )
    with pytest.raises(RevisionDetectionError, match="overlap"):
        update_detector_gpu_accounting_ledger(
            ledger_path,
            source_id="equivalence_cuda_task_0",
            component="detector_device_equivalence_cuda",
            terminal_receipt_path=second["receipt_path"],
        )


def test_final_manifest_requires_and_incorporates_exact_six_gpu_sources(tmp_path):
    sources = (
        ("production_benchmark_task_0", "detector_production_benchmark"),
        ("production_benchmark_task_1", "detector_production_benchmark"),
        ("equivalence_cuda_task_0", "detector_device_equivalence_cuda"),
        ("equivalence_cuda_task_1", "detector_device_equivalence_cuda"),
        (
            "equivalence_cuda_repeat_task_0",
            "detector_device_equivalence_cuda_repeat",
        ),
        (
            "equivalence_cuda_repeat_task_1",
            "detector_device_equivalence_cuda_repeat",
        ),
    )
    ledger_path = tmp_path / "six-source-ledger.json"
    for index, (source_id, component) in enumerate(sources):
        fixture = finalization_fixture(
            tmp_path,
            "source-{}".format(index),
            offset_seconds=float(index * 20),
        )
        finalize_detector_candidate_from_closed_status(
            fixture["candidate_path"],
            closed_status_file=fixture["status_file"],
            terminal_receipt_path=fixture["receipt_path"],
        )
        update_detector_gpu_accounting_ledger(
            ledger_path,
            source_id=source_id,
            component=component,
            terminal_receipt_path=fixture["receipt_path"],
        )
    ledger = read_detector_gpu_accounting_ledger(ledger_path)
    assert len(ledger["sources"]) == 6
    ledger_identity = {
        "path": str(ledger_path.resolve()),
        "sha256": file_sha256(ledger_path),
        "size_bytes": int(ledger_path.stat().st_size),
        "ledger_sha256": ledger["ledger_sha256"],
        "sources_sha256": ledger["sources_sha256"],
        "intervals_sha256": ledger["intervals_sha256"],
        "cumulative_elapsed_seconds": ledger["cumulative_elapsed_seconds"],
    }

    root = tmp_path / "final-suite"
    child = root / "detector_metrics.json"
    atomic_write_json(child, [{"sealed": True}])
    output = root / "detector_run_manifest.json"
    checkpoint_dir = root / "checkpoints"
    run_identity_sha256 = canonical_json_sha256({"run": "final-suite"})
    candidate_path, receipt_path = detector_finalization_paths(
        checkpoint_dir,
        kind="detector_run_manifest",
        requested_output_path=output,
        role="suite",
    )
    write_detector_finalization_candidate(
        candidate_path,
        kind="detector_run_manifest",
        run_identity_sha256=run_identity_sha256,
        payload={
            "schema_version": "rankcloak-revision-detector-run-v1",
            "completed_fit_count": 56,
            "total_fit_count": 56,
            "pre_final_gpu_accounting_ledger": ledger_identity,
        },
        output_files={
            "detector_metrics.json": {
                "path": str(child.resolve()),
                "sha256": file_sha256(child),
                "size_bytes": int(child.stat().st_size),
            }
        },
        requested_output_path=output,
    )
    accounting = gpu_accounting(140)
    status_file = root / "status.json"
    atomic_write_json(
        status_file,
        signed_document(
            {
                "schema_version": "rankcloak-revision-detector-status-v1",
                "updated_at_utc": accounting["intervals"][0][
                    "completed_at_utc"
                ],
                "state": "supervisor_observed_process_exit",
                "device": "cuda:0",
                "gpu_uuid": "GPU-fixture",
                "run_identity_sha256": run_identity_sha256,
                "gpu_accounting": accounting,
            },
            "status_sha256",
        ),
    )
    with pytest.raises(RevisionDetectionError, match="requires the pre-final"):
        finalize_detector_candidate_from_closed_status(
            candidate_path,
            closed_status_file=status_file,
            terminal_receipt_path=receipt_path,
        )
    assert not output.exists()
    assert not receipt_path.exists()
    result = finalize_detector_candidate_from_closed_status(
        candidate_path,
        closed_status_file=status_file,
        terminal_receipt_path=receipt_path,
        gpu_accounting_ledger_path=ledger_path,
    )
    assert len(result["gpu_accounting"]["intervals"]) == 7
    marker = read_detector_gpu_ledger_incorporation_marker(
        result["gpu_ledger_incorporation"]["path"]
    )
    assert marker["incorporated_ledger_interval_count"] == 6
    completed = verify_status_file(status_file)
    assert completed["state"] == "complete"
    assert completed["gpu_ledger_incorporation"]["sha256"] == file_sha256(
        result["gpu_ledger_incorporation"]["path"]
    )
