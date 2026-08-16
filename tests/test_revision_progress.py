from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256, file_sha256
import rankcloak.revision_progress as progress_module
from rankcloak.revision_progress import (
    AUTHORIZED_GPU_UUID,
    DETECTOR_EXECUTION_POLICY_CONTENT_SHA256,
    DETECTOR_EXECUTION_POLICY_SHA256,
    DETECTOR_GPU_COLLECTION_POLICY,
    DETECTOR_GPU_INTERVAL_POLICY,
    PROGRESS_SCHEMA_VERSION,
    RevisionProgressError,
    atomic_write_progress_snapshot,
    build_progress_snapshot,
    verify_progress_snapshot,
)


UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llama3_8b_instruct_q4_k_m"
MISTRAL = "mistral_7b_instruct_v0_3_q4_k_m"


def _baseline() -> dict:
    return {
        "verified_prior_seconds": 3600.0,
        "prior_components": [
            {
                "component": "fixture_prior",
                "seconds": 3600.0,
                "gpu_hours": 1.0,
                "verification": "fixture",
                "charge_only_not_rate_evidence": True,
                "scientific_result_evidence_allowed": False,
                "rate_evidence_allowed": False,
            }
        ],
        "targets": {
            "primary_v2": 2,
            "ablation_v2": 0,
            "robustness_v2": 0,
            "multilingual_v2": 0,
            "heldout_evaluator": 0,
            "neural_detector": 0,
        },
        "projection_sha256": (
            "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
        ),
        "projection_decision": {"go": True, "status": "go_within_budget"},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _fixture_root(tmp_path: Path) -> Path:
    policy_source = (
        PROJECT_ROOT
        / "operations/confirmatory_v2/detector_cuda_policy_v2.json"
    )
    policy_target = (
        tmp_path
        / "operations/confirmatory_v2/detector_cuda_policy_v2.json"
    )
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(policy_source.read_bytes())
    root = tmp_path / "results" / "revision_v1"
    shard = root / "primary_v2" / MODEL
    shard.mkdir(parents=True)
    plan = [
        {
            "work_id": "work-a",
            "trial_id": "work-a",
            "model_id": MODEL,
            "work_kind": "rankcloak",
            "protocol_variant": "hex_nibble_segmented",
            "representation_name": "hex_nibble",
            "language": "en",
            "prompt_category": "factual_explanation",
        },
        {
            "work_id": "work-b",
            "control_id": "work-b",
            "model_id": MODEL,
            "work_kind": "control",
            "language": "en",
            "prompt_category": "professional_communication",
        },
    ]
    _write_jsonl(shard / "plan.jsonl", plan)
    _write_jsonl(
        shard / "records.jsonl",
        [
            {
                "work_id": "work-a",
                "trial_id": "work-a",
                "attempt_index": 2,
                "execution_status": "completed",
                "record_type": "rankcloak_result",
                "saved_token_id_replay": {
                    "exact_payload_recovery": True,
                    "exact_recovery": True,
                },
                "completed_at": "2026-08-09T00:00:10+00:00",
            }
        ],
    )
    _write_jsonl(
        shard / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "at": "2026-08-09T00:00:02+00:00",
                "model_load_seconds": 2.0,
                "gpu_uuid": "GPU-fixture",
                "model_id": MODEL,
            }
        ],
    )
    _write_json(
        shard / "checkpoint.json",
        {
            "schema_version": "1.0",
            "study_id": "fixture/primary_v2/model",
            "planned_trial_count": 2,
            "completed_trial_ids": ["work-a"],
            "failed_trial_ids": [],
            "attempt_counts": {"work-a": 2},
            "failure_details": {},
            "created_at": "2026-08-09T00:00:00+00:00",
            "updated_at": "2026-08-09T00:00:10+00:00",
        },
    )
    return root


def _add_detector_manifest(
    root: Path,
    *,
    gpu: bool,
    metric_rows: int = 56,
    started: str = "2026-08-09T00:01:00+00:00",
    ended: str = "2026-08-09T00:03:00+00:00",
) -> Path:
    output = root / "neural_detector" / "confirmatory_v2"
    output.mkdir(parents=True)
    products = {}
    for name in (
        "detector_metrics.csv",
        "detector_predictions.csv",
        "detector_dataset_manifest.csv",
        "detector_split_manifest.json",
        "detector_failures.json",
    ):
        path = output / name
        path.write_text("fixture\n", encoding="utf-8")
        content = path.read_bytes()
        import hashlib

        products[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {
        "schema_version": "rankcloak-revision-detector-run-v2",
        "execution_mode": "confirmatory",
        "smoke": False,
        "confirmatory_complete": True,
        "failure_count": 0,
        "metric_rows": metric_rows,
        "output_files": products,
        "device": "cuda:0" if gpu else "cpu",
        "gpu_uuid": AUTHORIZED_GPU_UUID if gpu else None,
        "workers": 1,
        "completed_fit_count": metric_rows,
        "total_fit_count": metric_rows,
        "gpu_accounting": None,
    }
    if gpu:
        start_time = datetime.fromisoformat(started)
        end_time = datetime.fromisoformat(ended)
        seconds = (end_time - start_time).total_seconds()
        project_root = root.parents[1]
        policy_path = (
            project_root
            / "operations/confirmatory_v2/detector_cuda_policy_v2.json"
        ).resolve()
        permit_path = (
            root
            / "detector_cuda_reproducibility_v2/production_run_v2.fit_permit.json"
        ).resolve()
        run_identity = {
            "execution_policy_path": str(policy_path),
            "execution_policy_sha256": DETECTOR_EXECUTION_POLICY_SHA256,
            "fit_permit_file": str(permit_path),
            "require_fit_permit": True,
        }
        manifest["completed_fit_count"] = 56
        manifest["total_fit_count"] = 56
        manifest["execution_policy_path"] = str(policy_path)
        manifest["execution_policy_sha256"] = DETECTOR_EXECUTION_POLICY_SHA256
        manifest["execution_policy_content_sha256"] = (
            DETECTOR_EXECUTION_POLICY_CONTENT_SHA256
        )
        manifest["fit_permit_file"] = str(permit_path)
        manifest["run_identity"] = run_identity
        manifest["run_identity_sha256"] = canonical_json_sha256(run_identity)
        manifest["gpu_accounting"] = {
            "device": "cuda:0",
            "gpu_uuid": AUTHORIZED_GPU_UUID,
            "intervals": [
                {
                    "pid": 4321,
                    "process_start_ticks": 987654,
                    "device": "cuda:0",
                    "gpu_uuid": AUTHORIZED_GPU_UUID,
                    "started_at_utc": started,
                    "completed_at_utc": ended,
                    "last_observed_at_utc": ended,
                    "elapsed_seconds": seconds,
                    "derivation_policy": DETECTOR_GPU_INTERVAL_POLICY,
                }
            ],
            "cumulative_elapsed_seconds": seconds,
            "derivation_policy": DETECTOR_GPU_COLLECTION_POLICY,
        }
    path = output / "detector_run_manifest.json"
    _write_json(path, manifest)
    return path


def _add_evaluator_unavailability(root: Path) -> Path:
    source = root / "ablation_v2" / MISTRAL
    source.mkdir(parents=True)
    work_ids = ["upstream-unavailable-{:02d}".format(index) for index in range(48)]
    plans = [
        {
            "work_id": work_id,
            "trial_id": work_id,
            "model_id": MISTRAL,
            "work_kind": "rankcloak",
            "protocol_variant": "ablation-{:02d}".format(index % 4),
            "payload_name": "payload-{:02d}".format(index),
        }
        for index, work_id in enumerate(work_ids)
    ]
    records = [
        {
            "work_id": work_id,
            "trial_id": work_id,
            "attempt_index": 1,
            "execution_status": "completed",
            "record_type": "dependent_unavailable",
            "reason_code": "upstream_condition_unavailable",
            "completed_at": "2026-08-09T00:00:50+00:00",
        }
        for work_id in work_ids
    ]
    _write_jsonl(source / "plan.jsonl", plans)
    _write_jsonl(source / "records.jsonl", records)
    _write_jsonl(
        source / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "at": "2026-08-09T00:00:42+00:00",
                "model_load_seconds": 2.0,
                "gpu_uuid": "GPU-upstream-unavailability",
                "model_id": MISTRAL,
            }
        ],
    )
    _write_json(
        source / "checkpoint.json",
        {
            "planned_trial_count": 48,
            "completed_trial_ids": work_ids,
            "failed_trial_ids": [],
            "attempt_counts": {work_id: 1 for work_id in work_ids},
            "failure_details": {},
            "updated_at": "2026-08-09T00:01:00+00:00",
        },
    )
    _write_json(source / "run_identity.json", {"fixture": "unavailability-lineage"})

    files = []
    for role, filename in (
        ("plan", "plan.jsonl"),
        ("checkpoint", "checkpoint.json"),
        ("records", "records.jsonl"),
        ("run_identity", "run_identity.json"),
    ):
        path = source / filename
        files.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    records_by_id = {str(row["work_id"]): row for row in records}
    plans_by_id = {str(row["work_id"]): row for row in plans}
    units = []
    for work_id in sorted(work_ids):
        record = records_by_id[work_id]
        task = plans_by_id[work_id]
        units.append(
            {
                "terminal_status": "upstream_dependent_unavailable_not_scored",
                "source_stage": "ablation_v2",
                "source_work_id": work_id,
                "source_record_type": "dependent_unavailable",
                "source_record_sha256": canonical_json_sha256(record),
                "reason_code": "upstream_condition_unavailable",
                "generator_model_id": MISTRAL,
                "evaluator_model_id": MODEL,
                "protocol_variant": task["protocol_variant"],
                "payload_name": task["payload_name"],
                "scoring_attempted": False,
                "score_imputed": False,
            }
        )
    manifest = {
        "schema_version": "rankcloak-heldout-evaluator-upstream-unavailability-v1",
        "manifest_type": "heldout_evaluator_upstream_dependent_unavailability",
        "protocol_contract_revision": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "authorized_projection_sha256": (
            "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
        ),
        "frozen_evaluator_target_units": 17_280,
        "scoreable_evaluator_units": 17_232,
        "upstream_dependent_unavailable_units": 48,
        "terminal_accounted_units": 17_280,
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
    manifest_path = (
        root
        / "heldout_evaluator"
        / "upstream_dependent_unavailability_v1.json"
    )
    manifest_path.parent.mkdir(parents=True)
    _write_json(manifest_path, manifest)
    return manifest_path


def _unavailability_baseline() -> dict:
    baseline = _baseline()
    baseline["targets"]["ablation_v2"] = 48
    baseline["targets"]["heldout_evaluator"] = 17_280
    return baseline


def _resign_manifest(value: dict) -> None:
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_json_sha256(value)


def test_snapshot_counts_gpu_eta_current_and_recovered_error(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    value = build_progress_snapshot(
        root,
        generated_at=datetime(2026, 8, 9, 0, 0, 11, tzinfo=UTC),
        _baseline=_baseline(),
    )
    assert value["schema_version"] == PROGRESS_SCHEMA_VERSION
    assert value["status"] == "in_progress_or_paused"
    assert value["counts"] == {
        "completed": 1,
        "successes": 1,
        "failures": 0,
        "unavailable": 0,
        "total": 2,
        "remaining": 1,
    }
    assert value["current"]["stage"] == "primary_v2"
    assert value["current"]["model_id"] == MODEL
    assert value["current"]["trial_id"] == "work-b"
    assert value["current"]["condition"]["work_kind"] == "control"
    assert value["gpu"]["monitored_confirmatory_seconds"] == 10.0
    assert value["gpu"]["cumulative_actual_gpu_hours"] == 3610.0 / 3600.0
    assert value["throughput"]["completed_per_gpu_hour"] == 360.0
    assert value["eta"]["rolling_eta_seconds"] == 10.0
    assert value["recovery_counts"] == {
        "payload_bearing_recovery_attempted": 1,
        "successful_payload_recoveries": 1,
        "payload_recovery_failures": 0,
        "unavailable": 0,
    }
    assert value["recovered_errors"][0]["trial_id"] == "work-a"
    assert value["recovered_errors"][0]["attempt_count"] == 2


def test_atomic_write_and_compact_verifier_bind_sources(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    value = build_progress_snapshot(root, _baseline=_baseline())
    output = root / "confirmatory_progress_v1.json"
    atomic_write_progress_snapshot(output, value, root)
    report = verify_progress_snapshot(output)
    assert report["status"] == "ok"
    assert report["counts"]["completed"] == 1
    assert report["cumulative_actual_gpu_hours"] == 3610.0 / 3600.0

    records = root / "primary_v2" / MODEL / "records.jsonl"
    records.write_text(records.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RevisionProgressError, match="stale"):
        verify_progress_snapshot(output)


def test_cuda_detector_interval_is_counted_and_source_bound(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _add_detector_manifest(root, gpu=True)
    baseline = _baseline()
    baseline["targets"]["neural_detector"] = 56
    with pytest.raises(RevisionProgressError, match="pre-final GPU ledger"):
        build_progress_snapshot(root, _baseline=baseline)


def test_cpu_detector_preserves_zero_gpu_behavior(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _add_detector_manifest(root, gpu=False, metric_rows=3)
    baseline = _baseline()
    baseline["targets"]["neural_detector"] = 3
    value = build_progress_snapshot(root, _baseline=baseline)
    detector = next(
        row for row in value["stage_progress"] if row["stage"] == "neural_detector"
    )
    assert detector["completed"] == 3
    assert not any(
        row["component"] == "neural_detector"
        for row in value["gpu"]["confirmatory_intervals"]
    )
    assert value["gpu"]["monitored_confirmatory_seconds"] == 10.0


def _pre_final_ledger(
    root: Path,
    *,
    started: str = "2026-08-09T00:01:00+00:00",
    ended: str = "2026-08-09T00:03:00+00:00",
) -> tuple[Path, dict]:
    start = datetime.fromisoformat(started)
    end = datetime.fromisoformat(ended)
    seconds = (end - start).total_seconds()
    interval = {
        "pid": 4321,
        "process_start_ticks": 987654,
        "device": "cuda:0",
        "gpu_uuid": AUTHORIZED_GPU_UUID,
        "started_at_utc": started,
        "completed_at_utc": ended,
        "last_observed_at_utc": ended,
        "elapsed_seconds": seconds,
        "derivation_policy": DETECTOR_GPU_INTERVAL_POLICY,
    }
    ledger = {
        "schema_version": (
            "rankcloak-revision-detector-gpu-accounting-ledger-v1"
        ),
        "updated_at_utc": ended,
        "device": "cuda:0",
        "gpu_uuid": AUTHORIZED_GPU_UUID,
        "sources": [],
        "sources_sha256": canonical_json_sha256([]),
        "intervals": [interval],
        "intervals_sha256": canonical_json_sha256([interval]),
        "cumulative_elapsed_seconds": seconds,
        "derivation_policy": (
            "terminal_receipt_interval_union_deduplicated_v1"
        ),
    }
    ledger["ledger_sha256"] = canonical_json_sha256(ledger)
    path = root / "detector_cuda_reproducibility_v2/gpu_accounting_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, ledger)
    return path, ledger


def test_pre_final_gpu_ledger_is_included_in_every_progress_budget_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_root(tmp_path)
    ledger_path, ledger = _pre_final_ledger(root)
    monkeypatch.setattr(
        progress_module,
        "read_detector_gpu_accounting_ledger",
        lambda path: ledger if Path(path) == ledger_path else None,
    )
    value = build_progress_snapshot(root, _baseline=_baseline())
    detector_rows = [
        row
        for row in value["gpu"]["confirmatory_intervals"]
        if row["component"] == "neural_detector"
    ]
    assert len(detector_rows) == 1
    assert detector_rows[0]["model_id"] == "pre_final_detector_gpu_ledger"
    assert detector_rows[0]["seconds"] == 120.0
    assert value["gpu"]["monitored_confirmatory_seconds"] == 130.0
    output = root / "confirmatory_progress_v1.json"
    atomic_write_progress_snapshot(output, value, root)
    assert verify_progress_snapshot(output)["cumulative_actual_gpu_hours"] == (
        3730.0 / 3600.0
    )


def test_pre_final_gpu_ledger_tamper_and_overlap_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_root(tmp_path)
    ledger_path, ledger = _pre_final_ledger(root)
    monkeypatch.setattr(
        progress_module,
        "read_detector_gpu_accounting_ledger",
        lambda path: ledger if Path(path) == ledger_path else None,
    )
    value = build_progress_snapshot(root, _baseline=_baseline())
    output = root / "confirmatory_progress_v1.json"
    atomic_write_progress_snapshot(output, value, root)
    ledger["cumulative_elapsed_seconds"] += 1.0
    ledger.pop("ledger_sha256")
    ledger["ledger_sha256"] = canonical_json_sha256(ledger)
    _write_json(ledger_path, ledger)
    with pytest.raises(RevisionProgressError, match="cumulative"):
        build_progress_snapshot(root, _baseline=_baseline())

    ledger_path, ledger = _pre_final_ledger(
        root,
        started="2026-08-08T23:59:59+00:00",
        ended="2026-08-09T00:00:05+00:00",
    )
    events_path = root / "primary_v2" / MODEL / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["gpu_uuid"] = AUTHORIZED_GPU_UUID
    _write_jsonl(events_path, events)
    monkeypatch.setattr(
        progress_module,
        "read_detector_gpu_accounting_ledger",
        lambda path: ledger if Path(path) == ledger_path else None,
    )
    with pytest.raises(RevisionProgressError, match="overlap"):
        build_progress_snapshot(root, _baseline=_baseline())


def test_final_cuda_closure_requires_all_six_prefinal_ledger_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_root(tmp_path)
    ledger_path, ledger = _pre_final_ledger(root)
    monkeypatch.setattr(
        progress_module,
        "read_detector_gpu_accounting_ledger",
        lambda path: ledger if Path(path) == ledger_path else None,
    )
    final_path = root / "neural_detector/confirmatory_v2/detector_run_manifest.json"
    with pytest.raises(RevisionProgressError, match="all six"):
        progress_module._scan_detector_gpu_ledger(
            root,
            final_manifest_path=final_path,
            final_manifest={"device": "cuda:0"},
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("uuid", "identity"),
        ("elapsed", "provenance"),
        ("count", "56-fit"),
    ],
)
def test_cuda_detector_gpu_provenance_fails_closed(
    tmp_path: Path, tamper: str, message: str
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = _add_detector_manifest(root, gpu=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "uuid":
        manifest["gpu_accounting"]["gpu_uuid"] = "GPU-wrong"
    elif tamper == "elapsed":
        manifest["gpu_accounting"]["intervals"][0]["elapsed_seconds"] += 1.0
        manifest["gpu_accounting"]["cumulative_elapsed_seconds"] += 1.0
    else:
        manifest["metric_rows"] = 55
    _write_json(manifest_path, manifest)
    baseline = _baseline()
    baseline["targets"]["neural_detector"] = 56
    with pytest.raises(RevisionProgressError, match=message):
        build_progress_snapshot(root, _baseline=baseline)


def test_cuda_detector_scan_rejects_extra_or_incomplete_manifests(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = _add_detector_manifest(root, gpu=True)
    baseline = _baseline()
    baseline["targets"]["neural_detector"] = 56
    extra = root / "neural_detector" / "unexpected"
    extra.mkdir()
    (extra / "detector_run_manifest.json").write_bytes(manifest_path.read_bytes())
    with pytest.raises(RevisionProgressError, match="canonical 56-fit"):
        build_progress_snapshot(root, _baseline=baseline)

    (extra / "detector_run_manifest.json").unlink()
    extra.rmdir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["confirmatory_complete"] = False
    manifest["failure_count"] = 1
    _write_json(manifest_path, manifest)
    with pytest.raises(RevisionProgressError, match="confirmatory-complete"):
        build_progress_snapshot(root, _baseline=baseline)


def test_verifier_rejects_resigned_detector_interval_tampering(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    _add_detector_manifest(root, gpu=True)
    baseline = _baseline()
    baseline["targets"]["neural_detector"] = 56
    with pytest.raises(RevisionProgressError, match="pre-final GPU ledger"):
        build_progress_snapshot(root, _baseline=baseline)


def test_verifier_rejects_self_hash_tampering(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    output = root / "confirmatory_progress_v1.json"
    value = build_progress_snapshot(root, _baseline=_baseline())
    atomic_write_progress_snapshot(output, value, root)
    raw = json.loads(output.read_text())
    raw["counts"]["completed"] = 0
    _write_json(output, raw)
    with pytest.raises(RevisionProgressError, match="self-hash"):
        verify_progress_snapshot(output)


def test_rejects_overlapping_same_gpu_sessions(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    second = root / "primary_v2" / "qwen2_5_7b_instruct_q4_k_m"
    second.mkdir(parents=True)
    _write_jsonl(
        second / "plan.jsonl",
        [{"work_id": "q", "model_id": "qwen2_5_7b_instruct_q4_k_m"}],
    )
    _write_jsonl(second / "records.jsonl", [])
    _write_jsonl(
        second / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "at": "2026-08-09T00:00:06+00:00",
                "model_load_seconds": 1.0,
                "gpu_uuid": "GPU-fixture",
                "model_id": "qwen2_5_7b_instruct_q4_k_m",
            }
        ],
    )
    _write_json(
        second / "checkpoint.json",
        {
            "planned_trial_count": 1,
            "completed_trial_ids": [],
            "failed_trial_ids": [],
            "attempt_counts": {},
            "failure_details": {},
            "updated_at": "2026-08-09T00:00:08+00:00",
        },
    )
    baseline = _baseline()
    baseline["targets"]["primary_v2"] = 3
    with pytest.raises(RevisionProgressError, match="overlap"):
        build_progress_snapshot(root, _baseline=baseline)


def test_scans_heldout_evaluator_evaluation_ids_and_model_identity(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    evaluator = root / "heldout_evaluator" / "primary_v2" / MODEL
    evaluator.mkdir(parents=True)
    _write_jsonl(
        evaluator / "plan.jsonl",
        [
            {
                "evaluation_id": "eval-a",
                "trial_id": "source-trial",
                "evaluator_model_id": MODEL,
                "generator_model_id": "qwen2_5_7b_instruct_q4_k_m",
                "source_stage": "primary_v2",
                "text_view": "full_message",
            }
        ],
    )
    _write_jsonl(
        evaluator / "records.jsonl",
        [
            {
                "evaluation_id": "eval-a",
                "trial_id": "source-trial",
                "evaluator_model_id": MODEL,
                "model_id": "qwen2_5_7b_instruct_q4_k_m",
                "attempt_index": 1,
                "execution_status": "completed",
                "completed_at": "2026-08-09T00:00:30+00:00",
            }
        ],
    )
    _write_jsonl(
        evaluator / "events.jsonl",
        [
            {
                "event": "evaluator_model_loaded",
                "at": "2026-08-09T00:00:22+00:00",
                "model_load_seconds": 2.0,
                "gpu_uuid": "GPU-evaluator",
                "evaluator_model_id": MODEL,
            }
        ],
    )
    _write_json(
        evaluator / "checkpoint.json",
        {
            "planned_trial_count": 1,
            "completed_trial_ids": ["eval-a"],
            "failed_trial_ids": [],
            "attempt_counts": {"eval-a": 1},
            "failure_details": {},
            "updated_at": "2026-08-09T00:00:30+00:00",
        },
    )
    baseline = _baseline()
    baseline["targets"]["heldout_evaluator"] = 1
    value = build_progress_snapshot(root, _baseline=baseline)
    evaluator_row = next(
        row for row in value["shards"] if row["component"] == "heldout_evaluator"
    )
    assert evaluator_row["model_id"] == MODEL
    assert evaluator_row["completed"] == 1
    assert value["counts"]["completed"] == 2
    assert value["gpu"]["monitored_confirmatory_seconds"] == 20.0


def test_counts_and_binds_exact_upstream_evaluator_unavailability(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = _add_evaluator_unavailability(root)
    value = build_progress_snapshot(root, _baseline=_unavailability_baseline())
    evaluator = next(
        row for row in value["stage_progress"] if row["stage"] == "heldout_evaluator"
    )
    assert {
        key: evaluator[key]
        for key in ("completed", "successes", "failures", "unavailable")
    } == {
        "completed": 48,
        "successes": 0,
        "failures": 0,
        "unavailable": 48,
    }
    assert evaluator["recovery_counts"] == {
        "payload_bearing_recovery_attempted": 0,
        "successful_payload_recoveries": 0,
        "payload_recovery_failures": 0,
        "unavailable": 0,
    }
    binding = value["heldout_evaluator_upstream_unavailability"]
    assert binding["scoreable_evaluator_units"] == 17_232
    assert binding["upstream_dependent_unavailable_units"] == 48
    assert binding["terminal_accounted_units"] == 17_280
    assert binding["manifest_artifact"]["path"] == str(manifest_path.resolve())
    assert not any(
        interval["component"] == "heldout_evaluator"
        for interval in value["gpu"]["confirmatory_intervals"]
    )
    bound_paths = {row["path"] for row in value["source_artifacts"]}
    assert str(manifest_path.resolve()) in bound_paths
    assert str(
        (root / "ablation_v2" / MISTRAL / "run_identity.json").resolve()
    ) in bound_paths

    output = root / "confirmatory_progress_v1.json"
    atomic_write_progress_snapshot(output, value, root)
    report = verify_progress_snapshot(output)
    assert report["status"] == "ok"
    assert report["counts"]["completed"] == 97


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("self", "self-hash"),
        ("unit_hash", "units hash"),
        ("source", "source identity"),
    ],
)
def test_evaluator_unavailability_hashes_fail_closed(
    tmp_path: Path, tamper: str, message: str
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = _add_evaluator_unavailability(root)
    if tamper == "source":
        source = root / "ablation_v2" / MISTRAL / "run_identity.json"
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "self":
            manifest["manifest_sha256"] = "0" * 64
        else:
            manifest["units_sha256"] = "0" * 64
            _resign_manifest(manifest)
        _write_json(manifest_path, manifest)
    with pytest.raises(RevisionProgressError, match=message):
        build_progress_snapshot(root, _baseline=_unavailability_baseline())


def test_rejects_imputation_even_when_manifest_hashes_are_recomputed(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = _add_evaluator_unavailability(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["units"][0]["score_imputed"] = True
    manifest["units_sha256"] = canonical_json_sha256(manifest["units"])
    _resign_manifest(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(RevisionProgressError, match="differ from source records"):
        build_progress_snapshot(root, _baseline=_unavailability_baseline())


def test_rejects_double_counted_unavailable_evaluator_source(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _add_evaluator_unavailability(root)
    evaluator = root / "heldout_evaluator" / "ablation_v2" / MODEL
    evaluator.mkdir(parents=True)
    _write_jsonl(
        evaluator / "plan.jsonl",
        [
            {
                "evaluation_id": "eval-duplicate-unavailable",
                "trial_id": "source-trial",
                "evaluator_model_id": MODEL,
                "generator_model_id": MISTRAL,
                "source_stage": "ablation_v2",
                "source_work_id": "upstream-unavailable-00",
                "text_view": "full_message",
            }
        ],
    )
    _write_json(
        evaluator / "checkpoint.json",
        {
            "planned_trial_count": 1,
            "completed_trial_ids": [],
            "failed_trial_ids": [],
            "attempt_counts": {},
            "failure_details": {},
            "updated_at": "2026-08-09T00:01:10+00:00",
        },
    )
    with pytest.raises(RevisionProgressError, match="also appears in a scoring plan"):
        build_progress_snapshot(root, _baseline=_unavailability_baseline())


def test_rejects_symlinked_source(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    shard = root / "primary_v2" / MODEL
    real = shard / "plan.real"
    (shard / "plan.jsonl").rename(real)
    (shard / "plan.jsonl").symlink_to(real.name)
    with pytest.raises(RevisionProgressError, match="symlink"):
        build_progress_snapshot(root, _baseline=_baseline())
