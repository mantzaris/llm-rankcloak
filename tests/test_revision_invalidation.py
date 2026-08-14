import json
import os
from pathlib import Path

import pytest

from rankcloak.revision_artifacts import (
    build_run_identity_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    initialize_checkpoint,
    record_checkpoint_result,
)
from rankcloak.revision_invalidation import (
    INVALIDATION_HASH_FIELD,
    InvalidationEntryExistsError,
    RevisionInvalidationError,
    ShardIntegrityError,
    StoppedShardConfirmationError,
    create_invalidation_entry,
    snapshot_shard,
    verify_invalidation_entry,
)
from scripts.invalidate_revision_shard import main


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _fixture_shard(tmp_path: Path) -> Path:
    shard = tmp_path / "active_results" / "fixture_model"
    shard.mkdir(parents=True)
    plan = [
        {
            "work_id": "fixture-work-001",
            "work_kind": "rankcloak",
            "model_id": "fixture_model",
            "evidence_status": "fixture",
        },
        {
            "work_id": "fixture-work-002",
            "work_kind": "control",
            "model_id": "fixture_model",
            "evidence_status": "fixture",
        },
    ]
    _write_jsonl(shard / "plan.jsonl", plan)
    _write_json(shard / "payload_manifest.json", {"records": [], "schema_version": "1.0"})
    model_manifest = {
        "schema_version": "1.0",
        "configured_model": {"model_id": "fixture_model"},
        "verification": {"status": "ok", "actual_sha256": "d" * 64},
    }
    _write_json(shard / "model_manifest.json", model_manifest)
    source_files = []
    _write_json(
        shard / "source_manifest.json",
        {
            "schema_version": "1.0",
            "manifest_type": "revision_runner_source",
            "files": source_files,
            "files_sha256": canonical_json_sha256(source_files),
        },
    )
    _write_json(shard / "runtime_manifest.json", {"schema_version": "1.0"})
    _write_json(
        shard / "hardware_manifest.json",
        {"schema_version": "1.0", "selected_gpu_uuid": "GPU-fixture"},
    )
    config_hash = "a" * 64
    study_id = "fixture_revision/primary/fixture_model"
    identity = build_run_identity_manifest(
        study_id=study_id,
        config_manifest_sha256=config_hash,
        payload_manifest_sha256=file_sha256(shard / "payload_manifest.json"),
        planned_trial_ids=[row["work_id"] for row in plan],
        model_artifacts=[model_manifest],
        command_line_args=[
            "stage=primary",
            "model_id=fixture_model",
            "gpu_uuid=GPU-fixture",
            "source_manifest_sha256={}".format(file_sha256(shard / "source_manifest.json")),
            "runtime_manifest_sha256={}".format(file_sha256(shard / "runtime_manifest.json")),
            "hardware_manifest_sha256={}".format(file_sha256(shard / "hardware_manifest.json")),
        ],
    )
    _write_json(shard / "run_identity.json", identity)
    checkpoint_path = shard / "checkpoint.json"
    initialize_checkpoint(
        checkpoint_path,
        study_id=study_id,
        config_manifest_sha256=config_hash,
        planned_trial_ids=[row["work_id"] for row in plan],
        timestamp="2026-08-08T00:00:00+00:00",
    )
    record_checkpoint_result(
        checkpoint_path,
        "fixture-work-001",
        "completed",
        timestamp="2026-08-08T00:01:00+00:00",
    )
    _write_jsonl(
        shard / "records.jsonl",
        [
            {
                "work_id": "fixture-work-001",
                "attempt_index": 1,
                "execution_status": "completed",
                "execution_seconds": 10.0,
                "completed_at": "2026-08-08T00:00:30+00:00",
                "record_type": "rankcloak_trial",
            }
        ],
    )
    _write_jsonl(
        shard / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "at": "2026-08-08T00:00:01+00:00",
                "model_load_seconds": 1.0,
            },
            {
                "event": "memory_profile",
                "started_at": "2026-08-08T00:00:00+00:00",
                "at": "2026-08-08T00:01:01+00:00",
                "selected_gpu_uuid": "GPU-fixture",
                "poll_interval_seconds": 1.0,
                "sample_count": 60,
                "selected_gpu_sample_count": 60,
                "process_rss_sample_count": 60,
                "selected_gpu_initial_used_memory_mib": 10,
                "selected_gpu_final_used_memory_mib": 11,
                "selected_gpu_peak_used_memory_mib_sampled": 20,
                "process_peak_rss_bytes_sampled": 1000,
                "process_peak_rss_bytes_os_high_water": 1100,
            },
        ],
    )
    return shard


def _create(shard: Path, entry: Path):
    return create_invalidation_entry(
        shard,
        entry,
        reason_code="fixture_methodological_defect",
        reason="Fixture recovery outcome was defined at the wrong semantic layer.",
        superseding_target_namespace="fixture_revision/primary_v2/fixture_model",
        superseding_stages=["smoke_v3", "primary_v2"],
        confirm_stopped=True,
        created_at="2026-08-08T01:00:00+00:00",
    )


def test_external_entry_is_self_hashed_no_replace_and_never_changes_shard(tmp_path):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    before = snapshot_shard(shard)

    manifest = _create(shard, entry)

    assert shard.is_dir()
    assert snapshot_shard(shard) == before
    assert entry.is_file()
    on_disk = json.loads(entry.read_text(encoding="utf-8"))
    assert on_disk == manifest
    unhashed = dict(on_disk)
    stored = unhashed.pop(INVALIDATION_HASH_FIELD)
    assert canonical_json_sha256(unhashed) == stored
    assert on_disk["scientific_status"] == "invalidated_not_for_pooling"
    assert on_disk["superseded_identity"]["run_identity_sha256"]
    assert on_disk["superseded_identity"]["planned_trial_ids_sha256"]
    assert on_disk["superseded_identity"]["source_manifest_sha256"]
    assert (
        on_disk["superseding_target"]["namespace"]
        == "fixture_revision/primary_v2/fixture_model"
    )
    assert on_disk["superseding_target"]["stages"] == ["smoke_v3", "primary_v2"]
    assert on_disk["execution_state"]["terminal_state"] == "stopped_incomplete"
    assert on_disk["execution_state"]["remaining_work_units"] == 1
    assert on_disk["incurred_compute"]["charge_policy"] == "memory_profile_wall_span_v1"
    assert on_disk["incurred_compute"]["incurred_gpu_seconds"] == 61.0
    report = verify_invalidation_entry(entry)
    assert report["status"] == "ok"
    assert report["shard_tree_sha256"] == before["shard_tree_sha256"]
    assert report["incurred_gpu_seconds"] == 61.0
    assert report["superseding_stages"] == ["smoke_v3", "primary_v2"]


def test_stopped_confirmation_is_mandatory_and_does_not_publish(tmp_path):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    with pytest.raises(StoppedShardConfirmationError, match="stopped-shard"):
        create_invalidation_entry(
            shard,
            entry,
            reason_code="fixture_methodological_defect",
            reason="A nonempty reason.",
            superseding_target_namespace="fixture_revision/primary_v2/fixture_model",
            superseding_stages=["smoke_v3", "primary_v2"],
            confirm_stopped=False,
        )
    assert not entry.exists()
    assert shard.is_dir()


def test_entry_must_be_external_and_existing_target_is_never_overwritten(tmp_path):
    shard = _fixture_shard(tmp_path)
    inside = shard / "invalidation.json"
    with pytest.raises(RevisionInvalidationError, match="external"):
        _create(shard, inside)
    assert not inside.exists()

    entry = tmp_path / "invalidations" / "fixture_model.json"
    entry.parent.mkdir()
    entry.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(InvalidationEntryExistsError, match="already exists"):
        _create(shard, entry)
    assert entry.read_text(encoding="utf-8") == "sentinel\n"
    assert shard.is_dir()


def test_verify_fails_closed_after_byte_or_metadata_mutation(tmp_path):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    _create(shard, entry)
    records = shard / "records.jsonl"
    records.write_bytes(records.read_bytes() + b" \n")
    with pytest.raises(ShardIntegrityError, match="changed"):
        verify_invalidation_entry(entry)

    shard_two = _fixture_shard(tmp_path / "second")
    entry_two = tmp_path / "invalidations" / "fixture_model_two.json"
    _create(shard_two, entry_two)
    source = shard_two / "source_manifest.json"
    current = source.stat()
    os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
    with pytest.raises(ShardIntegrityError, match="changed"):
        verify_invalidation_entry(entry_two)


def test_manifest_tampering_and_plan_identity_drift_fail_closed(tmp_path):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    _create(shard, entry)
    value = json.loads(entry.read_text(encoding="utf-8"))
    value["invalidation"]["reason"] = "tampered"
    _write_json(entry, value)
    with pytest.raises(RevisionInvalidationError, match="self-hash"):
        verify_invalidation_entry(entry)

    drifted = _fixture_shard(tmp_path / "drifted")
    plan = drifted / "plan.jsonl"
    rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
    rows.reverse()
    _write_jsonl(plan, rows)
    with pytest.raises(ShardIntegrityError, match="ordered-plan"):
        _create(drifted, tmp_path / "invalidations" / "drifted.json")


def test_symlink_inside_shard_is_rejected(tmp_path):
    shard = _fixture_shard(tmp_path)
    try:
        (shard / "linked-artifact").symlink_to(shard / "plan.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ShardIntegrityError, match="symlinks"):
        _create(shard, tmp_path / "invalidations" / "fixture_model.json")


def test_cli_requires_explicit_confirm_stopped(tmp_path, capsys):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    status = main(
        [
            "create",
            "--shard",
            str(shard),
            "--registry-entry",
            str(entry),
            "--reason-code",
            "fixture_methodological_defect",
            "--reason",
            "Fixture reason.",
            "--superseding-target-namespace",
            "fixture_revision/primary_v2/fixture_model",
            "--superseding-stage",
            "smoke_v3",
            "--superseding-stage",
            "primary_v2",
        ]
    )
    assert status == 2
    assert "stopped-shard" in capsys.readouterr().err
    assert not entry.exists()


def test_memory_profile_charge_is_required_and_gpu_bound(tmp_path):
    shard = _fixture_shard(tmp_path / "missing")
    events_path = shard / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(events_path, [event for event in events if event["event"] != "memory_profile"])
    entry = tmp_path / "invalidations" / "missing.json"
    with pytest.raises(ShardIntegrityError, match="lacks a final memory_profile"):
        _create(shard, entry)
    assert not entry.exists()

    mismatch = _fixture_shard(tmp_path / "mismatch")
    mismatch_events = mismatch / "events.jsonl"
    rows = [json.loads(line) for line in mismatch_events.read_text(encoding="utf-8").splitlines()]
    rows[-1]["selected_gpu_uuid"] = "GPU-wrong"
    _write_jsonl(mismatch_events, rows)
    with pytest.raises(ShardIntegrityError, match="GPU UUID differs"):
        _create(mismatch, tmp_path / "invalidations" / "mismatch.json")


def test_superseding_stages_are_nonempty_and_unique(tmp_path):
    shard = _fixture_shard(tmp_path)
    with pytest.raises(RevisionInvalidationError, match="duplicates"):
        create_invalidation_entry(
            shard,
            tmp_path / "invalidations" / "duplicate.json",
            reason_code="fixture_methodological_defect",
            reason="Fixture reason.",
            superseding_target_namespace="fixture_revision/primary_v2/fixture_model",
            superseding_stages=["smoke_v3", "smoke_v3"],
            confirm_stopped=True,
        )


def test_cli_create_and_verify_fixture_entry(tmp_path, capsys):
    shard = _fixture_shard(tmp_path)
    entry = tmp_path / "invalidations" / "fixture_model.json"
    create_status = main(
        [
            "create",
            "--shard",
            str(shard),
            "--registry-entry",
            str(entry),
            "--reason-code",
            "fixture_methodological_defect",
            "--reason",
            "Fixture reason.",
            "--superseding-target-namespace",
            "fixture_revision/primary_v2/fixture_model",
            "--superseding-stage",
            "smoke_v3",
            "--superseding-stage",
            "primary_v2",
            "--confirm-stopped",
        ]
    )
    assert create_status == 0
    assert entry.is_file()
    capsys.readouterr()
    verify_status = main(["verify", "--registry-entry", str(entry)])
    assert verify_status == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["incurred_gpu_seconds"] == 61.0
    assert rendered["charge_policy"] == "memory_profile_wall_span_v1"
