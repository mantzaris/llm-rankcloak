import json

import pytest

from rankcloak.revision_artifacts import (
    ArtifactIntegrityError,
    ImmutableArtifactError,
    build_directory_manifest,
    build_run_identity_manifest,
    canonical_json_sha256,
    checkpoint_summary,
    initialize_checkpoint,
    pending_trial_ids,
    record_checkpoint_result,
    verify_directory_manifest,
    write_immutable_json,
)


def test_canonical_json_hash_ignores_mapping_order():
    assert canonical_json_sha256({"a": 1, "b": [2, 3]}) == canonical_json_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_immutable_json_allows_identical_retry_and_rejects_change(tmp_path):
    path = tmp_path / "frozen.json"
    assert write_immutable_json(path, {"value": 1}) is True
    assert write_immutable_json(path, {"value": 1}) is False
    with pytest.raises(ImmutableArtifactError):
        write_immutable_json(path, {"value": 2})


def test_directory_manifest_detects_tampering_and_extras(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    manifest = build_directory_manifest(tmp_path)

    assert verify_directory_manifest(
        tmp_path, manifest, require_no_extra_files=True
    )["status"] == "ok"
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    report = verify_directory_manifest(tmp_path, manifest)
    assert report["status"] == "error"
    assert any("a.txt" in error for error in report["errors"])

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    report = verify_directory_manifest(
        tmp_path, manifest, require_no_extra_files=True
    )
    assert any("unlisted files" in error for error in report["errors"])


def test_checkpoint_resume_retries_failed_rows_and_refuses_plan_drift(tmp_path):
    path = tmp_path / "checkpoint.json"
    planned = ["trial_a", "trial_b", "trial_c"]
    checkpoint = initialize_checkpoint(
        path,
        study_id="study",
        config_manifest_sha256="a" * 64,
        planned_trial_ids=planned,
        timestamp="2026-08-08T00:00:00+00:00",
    )
    assert pending_trial_ids(planned, checkpoint) == planned

    checkpoint = record_checkpoint_result(
        path,
        "trial_a",
        "completed",
        timestamp="2026-08-08T00:01:00+00:00",
    )
    checkpoint = record_checkpoint_result(
        path,
        "trial_b",
        "failed",
        failure_detail={"exception_type": "TestFailure"},
        timestamp="2026-08-08T00:02:00+00:00",
    )
    assert pending_trial_ids(planned, checkpoint) == ["trial_b", "trial_c"]
    assert checkpoint_summary(checkpoint) == {
        "planned": 3,
        "completed": 1,
        "failed_current": 1,
        "remaining": 2,
    }

    checkpoint = record_checkpoint_result(
        path,
        "trial_b",
        "completed",
        timestamp="2026-08-08T00:03:00+00:00",
    )
    assert pending_trial_ids(planned, checkpoint) == ["trial_c"]
    assert checkpoint["attempt_counts"]["trial_b"] == 2
    assert checkpoint["failed_trial_ids"] == []
    with pytest.raises(ArtifactIntegrityError):
        initialize_checkpoint(
            path,
            study_id="study",
            config_manifest_sha256="b" * 64,
            planned_trial_ids=planned,
        )
    with pytest.raises(ArtifactIntegrityError):
        pending_trial_ids(["trial_a", "trial_b"], checkpoint)


def test_run_identity_manifest_is_deterministic_and_self_identifying():
    first = build_run_identity_manifest(
        study_id="study",
        config_manifest_sha256="a" * 64,
        payload_manifest_sha256="b" * 64,
        planned_trial_ids=["one", "two"],
        model_artifacts=[{"model_id": "model", "sha256": "c" * 64}],
        command_line_args=["--resume"],
    )
    second = build_run_identity_manifest(
        study_id="study",
        config_manifest_sha256="a" * 64,
        payload_manifest_sha256="b" * 64,
        planned_trial_ids=["one", "two"],
        model_artifacts=[{"model_id": "model", "sha256": "c" * 64}],
        command_line_args=["--resume"],
    )
    assert first == second
    assert len(first["run_identity_sha256"]) == 64
    assert json.loads(json.dumps(first)) == first
