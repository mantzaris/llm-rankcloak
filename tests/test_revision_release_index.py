import json
from pathlib import Path

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256
from rankcloak import revision_release_index as release_index


def write(path: Path, content: str = "verified artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def synthetic_project(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir(parents=True)
    for relative in release_index.VALIDATOR_SOURCE_PATHS:
        if relative.endswith("detector_acceleration_policy_v1.json"):
            policy = {
                "schema_version": "rankcloak-revision-detector-acceleration-policy-v1"
            }
            policy["policy_sha256"] = canonical_json_sha256(policy)
            write(root / relative, json.dumps(policy) + "\n")
        else:
            write(root / relative, "validator {}\n".format(relative))
    for _group, source, _destination, _role, _kind in release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS:
        path = root / source
        if source in release_index.CONFIRMATORY_FILE_SOURCES:
            if source.endswith("confirmatory_v2_events.jsonl"):
                write(path, '{"event":"confirmatory_post_primary_complete"}\n')
            elif source in release_index.DETECTOR_SEMANTIC_SOURCES:
                write(path, "{}\n")
            else:
                write(path, "{}\n".format(source))
        else:
            write(path / "verified.txt", "{}\n".format(source))
    manifest_fixture = release_index._file_record(
        root / release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS[0][1] / "verified.txt",
        root,
    )
    actions = [
        {
            "operation_id": operation_id,
            "completion_kind": completion_kind,
            "manifest": dict(manifest_fixture),
        }
        for operation_id, completion_kind in zip(
            release_index.LEGACY_EXPECTED_ACTION_IDS,
            release_index.LEGACY_EXPECTED_ACTION_KINDS,
        )
    ]
    verification = {
        "status": "verified_complete",
        "final_progress": {
            "execution_status": "complete",
            "counts": {"completed": 1, "total": 1, "failures": 0, "remaining": 0},
        },
        "actions": actions,
        "actions_sha256": canonical_json_sha256(actions),
    }
    validators = [
        release_index._file_record(root / relative, root)
        for relative in release_index.VALIDATOR_SOURCE_PATHS
    ]
    artifacts = []
    for group, source, destination, role, kind in release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS:
        files = release_index._artifact_files(root, source)
        artifacts.append({
            "group": group,
            "source": source,
            "destination": destination,
            "evidence_role": role,
            "completion_kind": kind,
            "file_count": len(files),
            "files": files,
            "files_sha256": canonical_json_sha256(files),
        })
    value = {
        "schema_version": release_index.INDEX_SCHEMA,
        "manifest_type": release_index.INDEX_TYPE,
        "status": "verified_complete",
        "source_project_root": str(root),
        "protocol_contract_revision": release_index.PROTOCOL_REVISION,
        "result_schema_revision": release_index.RESULT_REVISION,
        "authorized_projection_sha256": release_index.AUTHORIZED_PROJECTION_SHA256,
        "network_access_used": False,
        "external_publication_performed": False,
        "doi_minted_or_reserved": False,
        "model_weights_included": False,
        "invalidated_or_superseded_outputs_included": False,
        "verification": verification,
        "validator_sources": validators,
        "validator_sources_sha256": canonical_json_sha256(validators),
        "artifacts": artifacts,
        "artifacts_sha256": canonical_json_sha256(artifacts),
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    return root, value, verification


def test_index_binds_exact_safe_confirmatory_file_sets(tmp_path, monkeypatch):
    root, value, verification = synthetic_project(tmp_path)
    target = root / release_index.DEFAULT_INDEX
    release_index.write_confirmatory_release_index(target, value)
    monkeypatch.setattr(
        release_index, "verify_legacy_confirmatory_pipeline", lambda _root: verification
    )
    report = release_index.verify_confirmatory_release_index(target, root)
    assert report["status"] == "verified_complete"
    assert report["artifact_count"] == len(release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS)
    assert report["manifest_sha256"] == value["manifest_sha256"]


def test_index_rejects_artifact_or_validator_drift(tmp_path):
    root, value, _verification = synthetic_project(tmp_path)
    first_source = root / release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS[0][1]
    (first_source / "verified.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="bytes differ"):
        release_index._verify_index_document(value, root, verify_live=False)

    root, value, _verification = synthetic_project(tmp_path / "validator")
    (root / release_index.VALIDATOR_SOURCE_PATHS[0]).write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="validator source changed"):
        release_index._verify_index_document(value, root, verify_live=False)


def test_index_refuses_weights_and_no_overwrite(tmp_path):
    root, value, _verification = synthetic_project(tmp_path)
    first_source = root / release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS[0][1]
    write(first_source / "model.gguf", "not real weights")
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="model weights"):
        release_index._artifact_files(
            root, release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS[0][1]
        )
    (first_source / "model.gguf").unlink()
    target = root / release_index.DEFAULT_INDEX
    release_index.write_confirmatory_release_index(target, value)
    before = target.read_bytes()
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="already exists"):
        release_index.write_confirmatory_release_index(target, value)
    assert target.read_bytes() == before


def test_checkpoint_transients_are_excluded_and_active_bytes_fail_closed(tmp_path):
    root, _value, _verification = synthetic_project(tmp_path)
    source_text = (
        "results/revision_v1/neural_detector/confirmatory_v2.checkpoints"
    )
    source = root / source_text
    write(source / ".execution.lock", "pid\n")
    write(source / "recovered_orphan_fit_files/stale.json", "{}\n")
    rows = release_index._artifact_files(root, source_text)
    assert all(".execution.lock" not in row["path"] for row in rows)
    assert all("recovered_orphan_fit_files" not in row["path"] for row in rows)
    write(source / ".tmp-active", "pending\n")
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="active or temporary"):
        release_index._artifact_files(root, source_text)
    (source / ".tmp-active").unlink()
    write(source / "confirmatory_v2.fit_permit.json", "{}\n")
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="active or temporary"):
        release_index._artifact_files(root, source_text)


def test_staged_verifier_rechecks_remapped_candidate_bytes(tmp_path):
    source_root, value, _verification = synthetic_project(tmp_path / "source")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for source, destination in release_index.VALIDATOR_CANDIDATE_PATHS.items():
        target = candidate / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / source).read_bytes())
    for artifact in value["artifacts"]:
        source = Path(artifact["source"])
        destination = Path(artifact["destination"])
        source_is_file = source.as_posix() in release_index.CONFIRMATORY_FILE_SOURCES
        for row in artifact["files"]:
            original = source_root / row["path"]
            target = (
                candidate / destination
                if source_is_file
                else candidate / destination / Path(row["path"]).relative_to(source)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
    index_path = candidate / "CONFIRMATORY_ARTIFACT_INDEX.json"
    index_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    report = release_index.verify_staged_confirmatory_release_index(
        index_path, candidate
    )
    assert report["artifact_count"] == len(
        release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS
    )
    victim = candidate / value["artifacts"][0]["destination"] / "verified.txt"
    victim.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(
        release_index.ConfirmatoryReleaseIndexError, match="hash mismatch"
    ):
        release_index.verify_staged_confirmatory_release_index(index_path, candidate)


def test_staged_resolver_uses_candidate_bytes_without_original_host_fallback(tmp_path):
    source_root, value, _verification = synthetic_project(tmp_path / "source")
    action_manifest = source_root / value["artifacts"][0]["files"][0]["path"]
    action = value["verification"]["actions"][0]
    action["manifest"] = release_index._file_record(action_manifest, source_root)
    value["verification"]["actions_sha256"] = canonical_json_sha256(
        value["verification"]["actions"]
    )
    value["manifest_sha256"] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for source, destination in release_index.VALIDATOR_CANDIDATE_PATHS.items():
        target = candidate / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / source).read_bytes())
    for artifact in value["artifacts"]:
        source = Path(artifact["source"])
        destination = Path(artifact["destination"])
        is_file = source.as_posix() in release_index.CONFIRMATORY_FILE_SOURCES
        for row in artifact["files"]:
            target = candidate / destination
            if not is_file:
                target /= Path(row["path"]).relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source_root / row["path"]).read_bytes())
    index_path = candidate / "CONFIRMATORY_ARTIFACT_INDEX.json"
    index_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    moved = tmp_path / "original-unavailable"
    source_root.rename(moved)
    assert release_index.verify_staged_confirmatory_release_index(
        index_path, candidate
    )["status"] == "verified_complete"
    victim = candidate / value["artifacts"][0]["destination"] / "verified.txt"
    victim.write_text("changed\n", encoding="utf-8")
    with pytest.raises(release_index.ConfirmatoryReleaseIndexError, match="hash mismatch"):
        release_index.verify_staged_confirmatory_release_index(index_path, candidate)


def test_release_map_excludes_superseded_invalid_smoke_and_model_paths():
    sources = {row[1] for row in release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS}
    assert "results/revision_v1/primary" not in sources
    assert not any("smoke" in source for source in sources)
    assert not any("invalidations" in source for source in sources)
    assert not any("models" in Path(source).parts for source in sources)
    assert "results/revision_v1/manuscript_revision_v2" not in sources
    assert not any(
        row[4] == "manuscript_revision_v1"
        for row in release_index.LEGACY_CONFIRMATORY_ARTIFACT_SPECS
    )
    assert "results/revision_v1/final_progress_snapshot_v1.json" in sources


def test_cli_help_exposes_only_offline_build_check_and_dry_run(capsys):
    with pytest.raises(SystemExit) as exc:
        release_index.build_argument_parser().parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--check" in help_text
    assert "--dry-run" in help_text
    assert "--output" in help_text
    assert "upload" not in help_text.lower()
    assert "publish" not in help_text.lower()
