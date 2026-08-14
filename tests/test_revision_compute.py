import json
from pathlib import Path

import pytest

import rankcloak.revision_compute as revision_compute
from rankcloak.revision_artifacts import (
    build_run_identity_manifest,
    canonical_json_sha256,
    file_sha256,
    trial_ids_sha256,
)
from rankcloak.revision_compute import (
    COMPUTE_SCHEMA_VERSION,
    EVALUATOR_UNITS_PER_MODEL,
    EXPECTED_MODELS,
    EXPECTED_PLAN_COUNTS,
    EXPECTED_SMOKE_STAGE,
    EXPECTED_SMOKE_UNAVAILABLE,
    EXPECTED_UNAVAILABLE_PROPAGATION,
    build_auxiliary_timing_record,
    load_auxiliary_timings,
    load_frozen_plans,
    project_revision_compute,
)
from rankcloak.revision_payloads import REVISION_CORPUS_ID
from rankcloak.revision_runner import (
    EVIDENCE_SMOKE_V3,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="session")
def frozen_plans():
    return load_frozen_plans()


def _rankcloak_record(task, scale):
    encoding = 0.8 * scale
    supported = 0.2 * scale
    return {
        "schema_version": "1.0",
        "record_type": "rankcloak_trial",
        "work_id": task["work_id"],
        "trial_id": task["trial_id"],
        "model_id": task["model_id"],
        "evidence_status": EVIDENCE_SMOKE_V3,
        "study_phase": task["study_phase"],
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "execution_status": "completed",
        "attempt_index": 1,
        "timing": {
            "encoding_seconds": encoding,
            "supported_decoding_seconds": supported,
            "saved_token_id_replay_seconds": 0.15 * scale,
            "saved_inverse_transcode_seconds": 0.05 * scale,
            "detokenized_text_retokenized_seconds": 0.25 * scale,
            "text_inverse_transcode_seconds": 0.05 * scale,
            "greedy_leadin_regeneration_seconds": 0.30 * scale,
            "greedy_inverse_transcode_seconds": 0.05 * scale,
            "total_seconds": 1.7 * scale,
        },
        "execution_seconds": 1.7 * scale,
    }


def _control_record(task, scale):
    return {
        "schema_version": "1.0",
        "record_type": "ordinary_control",
        "work_id": task["work_id"],
        "control_id": task["work_id"],
        "model_id": task["model_id"],
        "evidence_status": EVIDENCE_SMOKE_V3,
        "study_phase": task["study_phase"],
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "execution_status": "completed",
        "attempt_index": 1,
        "execution_seconds": 0.9 * scale,
    }


def _condition_unavailable_record(task):
    tokenizer_id = "fixture-tokenizer::{}".format(task["model_id"])
    return {
        "schema_version": "1.0",
        "record_type": "condition_unavailable",
        "work_id": task["work_id"],
        "trial_id": task["trial_id"],
        "work_kind": task["work_kind"],
        "model_id": task["model_id"],
        "evidence_status": EVIDENCE_SMOKE_V3,
        "study_phase": task["study_phase"],
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "execution_status": "completed",
        "attempt_index": 1,
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        "reason_code": "empty_isolated_roundtrip_vocabulary",
        "reason": "No isolated round-trip-stable safe token.",
        "token_filter": task["token_filter"],
        "protocol_variant": task["protocol_variant"],
        "ablation_factor": task["ablation_factor"],
        "ablation_level": task["ablation_level"],
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact_sha256": "3" * 64,
        "safe_count": 31_464,
        "stable_count": 0,
        "vocabulary_size": 32_768,
    }


def _dependent_unavailable_record(task, source):
    root = {
        field: source.get(field)
        for field in (
            "work_id",
            "trial_id",
            "record_type",
            "reason_code",
            "reason",
            "model_id",
            "tokenizer_id",
            "tokenizer_revision",
            "tokenizer_artifact_sha256",
            "safe_count",
            "stable_count",
        )
    }
    return {
        "schema_version": "1.0",
        "record_type": "dependent_unavailable",
        "work_id": task["work_id"],
        "trial_id": task["control_id"],
        "control_id": task["control_id"],
        "source_trial_id": task["source_trial_id"],
        "source_record_sha256": canonical_json_sha256(source),
        "work_kind": task["work_kind"],
        "model_id": task["model_id"],
        "evidence_status": EVIDENCE_SMOKE_V3,
        "study_phase": task["study_phase"],
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "execution_status": "completed",
        "attempt_index": 1,
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        "reason_code": "source_condition_unavailable",
        "reason": "Required source is unavailable.",
        "dependency_role": "rankcloak control source",
        "dependency_record_type": source["record_type"],
        "dependency_root": root,
        "tokenizer_id": source["tokenizer_id"],
        "tokenizer_revision": source["tokenizer_revision"],
        "tokenizer_artifact_sha256": source["tokenizer_artifact_sha256"],
    }


def make_smoke_shard(root: Path, plan, scale=1.0) -> Path:
    model_id = str(plan[0]["model_id"])
    shard = root / model_id
    shard.mkdir(parents=True)
    write_jsonl(shard / "plan.jsonl", plan)
    payload_manifest = {
        "schema_version": "1.0",
        "manifest_type": "revision_v1_public_payload_corpus",
        "corpus_id": REVISION_CORPUS_ID,
        "corpus_sha256": "a" * 64,
        "payload_count": 480,
        "records": [],
    }
    digest = ("1" if model_id == EXPECTED_MODELS[0] else
              "2" if model_id == EXPECTED_MODELS[1] else "3") * 64
    model_manifest = {
        "schema_version": "1.0",
        "configured_model": {
            "model_id": model_id,
            "artifact_sha256": digest,
        },
        "execution_policy": {"one_model_loaded_at_a_time": True},
        "verification": {
            "model_id": model_id,
            "status": "ok",
            "expected_sha256": digest,
            "actual_sha256": digest,
        },
    }
    source_files = [
        {"path": "rankcloak/revision_runner.py", "size_bytes": 1, "sha256": "4" * 64}
    ]
    source_manifest = {
        "schema_version": "1.0",
        "manifest_type": "revision_runner_source",
        "files": source_files,
        "files_sha256": canonical_json_sha256(source_files),
    }
    runtime_manifest = {"schema_version": "1.0", "fixture": True}
    hardware_manifest = {
        "schema_version": "1.0",
        "selected_gpu_uuid": "GPU-fixture-{}".format(model_id),
        "gpu_inventory": [],
    }
    write_json(shard / "payload_manifest.json", payload_manifest)
    write_json(shard / "model_manifest.json", model_manifest)
    write_json(shard / "source_manifest.json", source_manifest)
    write_json(shard / "runtime_manifest.json", runtime_manifest)
    write_json(shard / "hardware_manifest.json", hardware_manifest)
    config_hash = "c" * 64
    ids = [str(row["work_id"]) for row in plan]
    args = [
        "stage={}".format(EXPECTED_SMOKE_STAGE),
        "model_id={}".format(model_id),
        "evidence_status={}".format(EVIDENCE_SMOKE_V3),
        "context_limit=4096",
        "gpu_uuid=GPU-fixture-{}".format(model_id),
        "n_gpu_layers=-1",
        "protocol_contract_revision={}".format(PROTOCOL_CONTRACT_REVISION),
        "result_schema_revision={}".format(RESULT_SCHEMA_REVISION),
        "source_manifest_sha256={}".format(file_sha256(shard / "source_manifest.json")),
        "runtime_manifest_sha256={}".format(file_sha256(shard / "runtime_manifest.json")),
        "hardware_manifest_sha256={}".format(file_sha256(shard / "hardware_manifest.json")),
    ]
    identity = build_run_identity_manifest(
        study_id="{}/{}/{}".format(REVISION_CORPUS_ID, EXPECTED_SMOKE_STAGE, model_id),
        config_manifest_sha256=config_hash,
        payload_manifest_sha256=file_sha256(shard / "payload_manifest.json"),
        planned_trial_ids=ids,
        model_artifacts=[model_manifest],
        command_line_args=args,
    )
    identity["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
    identity["result_schema_revision"] = RESULT_SCHEMA_REVISION
    identity.pop("run_identity_sha256", None)
    identity["run_identity_sha256"] = canonical_json_sha256(identity)
    write_json(shard / "run_identity.json", identity)
    checkpoint = {
        "schema_version": "1.0",
        "study_id": identity["study_id"],
        "config_manifest_sha256": config_hash,
        "planned_trial_count": len(ids),
        "planned_trial_ids_sha256": trial_ids_sha256(ids),
        "completed_trial_ids": sorted(ids),
        "failed_trial_ids": [],
        "failure_details": {},
        "attempt_counts": {work_id: 1 for work_id in ids},
        "created_at": "2026-08-08T00:00:00+00:00",
        "updated_at": "2026-08-08T00:01:00+00:00",
    }
    write_json(shard / "checkpoint.json", checkpoint)
    unavailable_source = None
    if model_id == EXPECTED_MODELS[2]:
        unavailable_task = next(
            task
            for task in plan
            if task["work_kind"] == "rankcloak"
            and task.get("token_filter") == "roundtrip_stable_filter_v1"
        )
        unavailable_source = _condition_unavailable_record(unavailable_task)
    records = []
    for task in plan:
        if unavailable_source is not None and task["work_id"] == unavailable_source["work_id"]:
            records.append(unavailable_source)
        elif (
            unavailable_source is not None
            and task["work_kind"] == "control"
            and task.get("source_trial_id") == unavailable_source["trial_id"]
        ):
            records.append(_dependent_unavailable_record(task, unavailable_source))
        elif task["work_kind"] == "rankcloak":
            records.append(_rankcloak_record(task, scale))
        else:
            records.append(_control_record(task, scale))
    write_jsonl(shard / "records.jsonl", records)
    write_jsonl(
        shard / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "model_id": model_id,
                "model_load_seconds": 2.0 * scale,
                "gpu_uuid": "GPU-fixture-{}".format(model_id),
            },
            {
                "event": "memory_profile",
                "started_at": "2026-08-08T00:00:00+00:00",
                "at": "2026-08-08T00:01:00+00:00",
                "selected_gpu_uuid": "GPU-fixture-{}".format(model_id),
                "sample_count": 60,
                "selected_gpu_sample_count": 60,
                "process_rss_sample_count": 60,
            },
            {
                "event": "session_finished",
                "planned": len(plan),
                "completed": len(plan),
                "failed_current": 0,
                "remaining": 0,
            },
        ],
    )
    return shard


def make_all_shards(tmp_path, plans, scale=1.0):
    return [
        make_smoke_shard(
            tmp_path,
            [row for row in plans[EXPECTED_SMOKE_STAGE] if row["model_id"] == model],
            scale=scale,
        )
        for model in EXPECTED_MODELS
    ]


def make_required_auxiliary_timings(tmp_path):
    paths = []
    for model_id in EXPECTED_MODELS:
        record = build_auxiliary_timing_record(
            "evaluator",
            "heldout_evaluator_smoke_v3_{}_fixture".format(model_id),
            32,
            64.0,
            1,
            model_id=model_id,
            model_load_seconds=2.0,
        )
        path = tmp_path / "aux-{}.json".format(model_id)
        write_json(path, record)
        paths.append(path)
    return paths


def install_required_charge_verifiers(monkeypatch):
    legacy_entries = [
        {
            "component": "legacy-fixture",
            "model_id": EXPECTED_MODELS[index % len(EXPECTED_MODELS)],
            "incurred_gpu_seconds": 10.0,
            "charge_policy": "fixture_exact_observed",
            "scientific_use": "incurred_gpu_charge_only_never_rate_evidence",
            "absolute_path": "/fixture/legacy-{}".format(index),
            "run_identity_sha256": str(index + 1) * 64,
            "gpu_uuid": "GPU-fixture-ledger",
            "occupancy_started_at": "2026-08-07T00:{:02d}:00+00:00".format(index),
            "occupancy_ended_at": "2026-08-07T00:{:02d}:10+00:00".format(index),
        }
        for index in range(6)
    ]
    monkeypatch.setattr(
        revision_compute,
        "verify_legacy_incurred_charge_ledger",
        lambda _: {
            "status": "ok",
            "entries": legacy_entries,
            "total_incurred_gpu_seconds": 60.0,
            "scientific_use": "incurred_charge_only_never_timing_rate_or_evidence",
        },
    )
    monkeypatch.setattr(
        revision_compute,
        "verify_invalidation_entry",
        lambda _: {
            "status": "ok",
            "scientific_status": "invalidated_not_for_pooling",
            "charge_policy": "memory_profile_wall_span_v1",
            "incurred_gpu_seconds": 2189.687278,
            "invalidation_manifest_sha256": revision_compute.EXPECTED_INVALIDATION_MANIFEST_SHA256,
            "run_identity_sha256": revision_compute.EXPECTED_INVALIDATED_RUN_IDENTITY_SHA256,
            "shard_tree_sha256": revision_compute.EXPECTED_INVALIDATED_SHARD_TREE_SHA256,
            "shard_path": "/fixture/invalidated-primary-qwen",
            "registry_entry_path": "/fixture/invalidation.json",
            "occupancy_intervals": [
                {
                    "gpu_uuid": "GPU-fixture-ledger",
                    "started_at": "2026-08-08T00:00:00+00:00",
                    "ended_at": "2026-08-08T00:36:29.687278+00:00",
                }
            ],
            "superseding_stages": ["smoke_v3", "primary_v2"],
            "execution_state": {
                "terminal_state": "stopped_incomplete",
                "incomplete": True,
                "planned_work_units": 4800,
                "completed_work_units": 234,
                "remaining_work_units": 4566,
            },
        },
    )
    return Path("legacy-ledger-fixture.json"), [Path("invalidation-fixture.json")]



def project_with_required_inputs(shards, tmp_path, monkeypatch, budget=150.0):
    ledger, invalidations = install_required_charge_verifiers(monkeypatch)
    auxiliary = make_required_auxiliary_timings(tmp_path)
    return project_revision_compute(
        shards,
        auxiliary_timing_paths=auxiliary,
        invalidation_manifest_paths=invalidations,
        legacy_incurred_ledger_path=ledger,
        budget_gpu_hours=budget,
    )



def test_frozen_plan_counts_are_the_approved_matrix(frozen_plans):
    for stage, expected in EXPECTED_PLAN_COUNTS.items():
        assert len(frozen_plans[stage]) == expected["total"]
    assert EVALUATOR_UNITS_PER_MODEL == 5760


def test_complete_smoke_shards_produce_model_stage_point_and_upper_projection(
    tmp_path, frozen_plans, monkeypatch
):
    monkeypatch.setattr(
        revision_compute,
        "verify_config_manifest",
        lambda _: {"sha256": "c" * 64},
    )
    shards = make_all_shards(tmp_path, frozen_plans)
    report = project_with_required_inputs(shards, tmp_path, monkeypatch)
    assert report["schema_version"] == COMPUTE_SCHEMA_VERSION
    assert report["input_status"] == "complete", report["incomplete_reasons"]
    assert report["decision"]["go"] is True
    assert report["decision"]["status"] == "go_within_budget"
    assert report["totals"]["upper_gpu_hours"] >= report["totals"]["point_gpu_hours"]
    assert len(report["verified_smoke_shards"]) == 3
    assert len(report["projection_rows"]) == 6 + 1 + 3 + 3 + 12 + 3 + 2
    assert report["unavailability_propagation"]["smoke_unavailable_counts"] == (
        EXPECTED_SMOKE_UNAVAILABLE
    )
    assert report["unavailability_propagation"]["projected_counts_by_stage_model"] == (
        EXPECTED_UNAVAILABLE_PROPAGATION
    )
    ablation_mistral = next(
        row
        for row in report["projection_rows"]
        if row["stage"] == "ablation_v2" and row["model_id"] == EXPECTED_MODELS[2]
    )
    robustness_mistral = next(
        row
        for row in report["projection_rows"]
        if row["stage"] == "robustness_v2" and row["model_id"] == EXPECTED_MODELS[2]
    )
    assert ablation_mistral["projected_unavailable_work_units"] == 48
    assert robustness_mistral["projected_unavailable_work_units"] == 336
    assert any(
        item["stratum"] == "planned_unavailable:condition_unavailable"
        and item["target_units"] == 48
        and item["point_seconds"] == 0
        for item in ablation_mistral["strata"]
    )
    assert any(
        item["stratum"] == "planned_unavailable:dependent_unavailable"
        and item["target_units"] == 336
        and item["upper_seconds"] == 0
        for item in robustness_mistral["strata"]
    )
    primary = [row for row in report["projection_rows"] if row["stage"] == "primary_v2"]
    assert {row["model_id"] for row in primary} == set(EXPECTED_MODELS)
    assert sum(row["target_work_units"] for row in primary) == 14400
    digest = report.pop("projection_sha256")
    assert digest == canonical_json_sha256(report)


def test_gate_refuses_incomplete_and_over_budget_inputs(
    tmp_path, frozen_plans, monkeypatch
):
    monkeypatch.setattr(
        revision_compute,
        "verify_config_manifest",
        lambda _: {"sha256": "c" * 64},
    )
    shards = make_all_shards(tmp_path, frozen_plans, scale=2.0)
    incomplete = project_revision_compute(shards[:2])
    assert incomplete["decision"]["go"] is False
    assert incomplete["decision"]["status"] == "no_go_incomplete_inputs"
    over = project_with_required_inputs(
        shards, tmp_path, monkeypatch, budget=0.01
    )
    assert over["input_status"] == "complete", over["incomplete_reasons"]
    assert over["decision"]["status"] == "no_go_over_budget"
    assert over["totals"]["upper_gpu_hours"] > 0.01


def test_gate_rejects_non_smoke_record_label(tmp_path, frozen_plans, monkeypatch):
    monkeypatch.setattr(
        revision_compute,
        "verify_config_manifest",
        lambda _: {"sha256": "c" * 64},
    )
    shards = make_all_shards(tmp_path, frozen_plans)
    path = shards[0] / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["evidence_status"] = "confirmatory_after_manifest_freeze"
    write_jsonl(path, rows)
    report = project_revision_compute(shards)
    assert report["decision"]["status"] == "no_go_incomplete_inputs"
    assert "non-smoke evidence label" in report["incomplete_reasons"][0]


def test_gate_rejects_tampered_unavailable_dependency_root(
    tmp_path, frozen_plans, monkeypatch
):
    monkeypatch.setattr(
        revision_compute,
        "verify_config_manifest",
        lambda _: {"sha256": "c" * 64},
    )
    shards = make_all_shards(tmp_path, frozen_plans)
    path = shards[2] / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    dependent = next(row for row in rows if row["record_type"] == "dependent_unavailable")
    dependent["dependency_root"]["safe_count"] += 1
    write_jsonl(path, rows)
    report = project_revision_compute(shards)
    assert report["decision"]["status"] == "no_go_incomplete_inputs"
    assert "root safe_count mismatch" in report["incomplete_reasons"][0]


def test_auxiliary_timing_is_self_hashed_and_smoke_labelled(tmp_path):
    valid = build_auxiliary_timing_record(
        "evaluator",
        "heldout_evaluator_smoke_v3_{}_fixture".format(EXPECTED_MODELS[1]),
        12, 24.0, 1,
        model_id=EXPECTED_MODELS[1],
        model_load_seconds=2.0,
    )
    path = tmp_path / "timing.json"
    write_json(path, valid)
    loaded = load_auxiliary_timings([path])
    assert loaded[0]["completed_units"] == 12
    tampered = dict(valid)
    tampered["elapsed_seconds"] = 25.0
    write_json(path, tampered)
    with pytest.raises(Exception, match="self-hash mismatch"):
        load_auxiliary_timings([path])
