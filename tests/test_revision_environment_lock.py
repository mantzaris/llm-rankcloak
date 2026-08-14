import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_revision_environment_lock.py"
SPEC = importlib.util.spec_from_file_location("revision_environment_lock", SCRIPT)
assert SPEC and SPEC.loader
envlock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(envlock)


def fixture_snapshot():
    return {
        "schema_version": "1.0",
        "snapshot_id": "fixture_environment",
        "status": "complete",
        "requirements_lock_bytes": b"fixture==1.2.3\n",
        "python": {"packages": [{"name": "fixture", "version": "1.2.3"}]},
        "r": {"runtime_version": "4.4.2", "status": "complete"},
        "backend_system": {"selected_execution_gpu_uuid": "GPU-fixture"},
        "determinism": envlock.build_determinism_record("GPU-fixture"),
        "scientific_pins": {
            "config": {"manifest_sha256": "a" * 64},
            "corpus": {"canonical_record_array_sha256": "b" * 64},
            "models": [
                {
                    "model_id": "fixture-model",
                    "artifact_sha256": "c" * 64,
                    "model_weight_included": False,
                }
            ],
            "model_weights_included": False,
        },
    }


def test_render_and_write_are_deterministic_and_content_hashed(tmp_path):
    first = envlock.render_bundle(fixture_snapshot())
    second = envlock.render_bundle(fixture_snapshot())
    assert first == second
    target = tmp_path / "environment"
    envlock.write_bundle(target, first)
    report = envlock.verify_bundle(target)
    assert report == {"status": "ok", "verified_file_count": 10, "errors": []}
    manifest = json.loads((target / "environment_manifest.json").read_text())
    assert manifest["file_count"] == 10
    assert (target / "bundle_status.json").is_file()
    assert not any(path.suffix == ".gguf" for path in target.rglob("*"))


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    with pytest.raises(envlock.EnvironmentLockError, match="already exists"):
        envlock.write_bundle(target, envlock.render_bundle(fixture_snapshot()))
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_checksum_tamper_and_unlisted_file_are_detected(tmp_path):
    target = tmp_path / "tampered"
    envlock.write_bundle(target, envlock.render_bundle(fixture_snapshot()))
    (target / "python_environment.json").write_text("{}\n", encoding="utf-8")
    report = envlock.verify_bundle(target)
    assert report["status"] == "error"
    assert any("python_environment.json" in error for error in report["errors"])
    (target / "extra.txt").write_text("extra\n", encoding="utf-8")
    report = envlock.verify_bundle(target)
    assert any("file-set mismatch" in error for error in report["errors"])


def test_forbidden_paths_secrets_and_model_weights_fail_closed():
    with pytest.raises(envlock.EnvironmentLockError, match="absolute user path"):
        envlock._scan_for_forbidden_content({"x.json": b'{"path":"/home/person/secret"}'})
    with pytest.raises(envlock.EnvironmentLockError, match="private-key"):
        envlock._scan_for_forbidden_content({"x.txt": b"-----BEGIN PRIVATE KEY-----"})
    with pytest.raises(envlock.EnvironmentLockError, match="model-weight"):
        envlock._scan_for_forbidden_content({"weights/model.gguf": b"not real weights"})


def test_deterministic_launch_contract_pins_gpu_uuid_and_serial_backend(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "observed-value")
    record = envlock.build_determinism_record("GPU-exact")
    required = record["required_for_gpu_runs"]
    assert required["CUDA_VISIBLE_DEVICES"] == "GPU-exact"
    assert required["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert required["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert required["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert record["observed_while_snapshot_was_built"]["CUDA_VISIBLE_DEVICES"] == "observed-value"
    assert record["model_loading_policy"]["rank_replay_n_batch"] == 1
    assert record["model_loading_policy"]["rank_replay_n_ubatch"] == 1


def test_composite_r_resolution_records_each_library_and_r_version_equivalence():
    lock = {
        "packages": {
            "ordinal": {"version": "2026.7-26"},
            "emmeans": {"version": "1.10.5"},
        },
        "policy": {"system_or_user_copy_does_not_satisfy_locked_package": False},
    }
    observed = [
        {
            "name": "ordinal", "version": "2026.7.26", "library_index": 1,
            "library_path": ".r_libs/revision_v1", "library_path_kind": "repository_relative",
        },
        {
            "name": "emmeans", "version": "1.10.5", "library_index": 2,
            "library_path": "$HOME/R/x86_64-pc-linux-gnu-library/4.4",
            "library_path_kind": "home_relative",
        },
    ]
    report = envlock.resolve_required_r_packages(lock, observed)
    assert report["complete"] is True
    by_name = {row["package"]: row for row in report["records"]}
    assert by_name["ordinal"]["resolved_library_path"] == ".r_libs/revision_v1"
    assert by_name["emmeans"]["resolved_library_path"].startswith("$HOME/")
    lock["policy"]["system_or_user_copy_does_not_satisfy_locked_package"] = True
    strict = envlock.resolve_required_r_packages(lock, observed)
    assert strict["complete"] is False
    assert "emmeans:scope_mismatch" in strict["errors"]


def test_project_r_library_inventory_hashes_description_without_copying(tmp_path):
    root = tmp_path / "project"
    description = root / ".r_libs/revision_v1/lme4/DESCRIPTION"
    description.parent.mkdir(parents=True)
    description.write_text("Package: lme4\nVersion: 2.0.6\n", encoding="utf-8")
    lock = {"library": {"relative_path": ".r_libs/revision_v1"}}
    report = envlock.inspect_r_project_library(root, lock)
    assert report["directory_present"] is True
    assert report["observed_package_count"] == 1
    assert report["records"][0]["package"] == "lme4"
    assert len(report["records"][0]["description_sha256"]) == 64


def test_python_inventory_sanitizes_editable_source_paths():
    record, requirements = envlock.collect_python_environment(Path.cwd())
    serialized = json.dumps(record, sort_keys=True)
    assert "/home/" not in serialized
    assert "file://" not in serialized
    assert record["package_count"] > 0
    assert b"-e ." in requirements
    rankcloak = [row for row in record["packages"] if row["name"] == "rankcloak"]
    assert any(row["editable"] is True for row in rankcloak)
    assert any(row["source_kind"] == "editable_local_directory" for row in rankcloak)


def test_default_check_ignores_only_local_model_presence_fields():
    value = fixture_snapshot()["scientific_pins"]
    value["model_file_sha256_verification_requested"] = True
    value["model_verification_status"] = "ok"
    value["models"][0]["local_verification"] = {"actual_sha256": "c" * 64}
    stripped = envlock._without_local_model_verification(value)
    assert "model_file_sha256_verification_requested" not in stripped
    assert "model_verification_status" not in stripped
    assert "local_verification" not in stripped["models"][0]
    assert stripped["models"][0]["artifact_sha256"] == "c" * 64


def test_optional_live_model_rehash_is_independent_of_snapshot_observation(tmp_path, monkeypatch):
    expected = fixture_snapshot()["scientific_pins"]
    expected["model_file_sha256_verification_requested"] = False
    expected["model_verification_status"] = "size_checked_only"
    output = tmp_path / "environment"
    output.mkdir()
    (output / "scientific_pins.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    actual = json.loads(json.dumps(expected))
    actual["model_file_sha256_verification_requested"] = True
    actual["model_verification_status"] = "ok"
    actual["models"][0]["local_verification"] = {
        "status": "ok", "actual_sha256": "c" * 64
    }
    monkeypatch.setattr(envlock, "collect_scientific_pins", lambda *_args: actual)
    report = envlock.verify_live_scientific_pins(
        output, tmp_path, verify_model_files=True
    )
    assert report["status"] == "ok"
    assert report["model_verification_status"] == "ok"


def test_bundle_status_never_claims_install_network_copy_or_publication():
    files = envlock.render_bundle(fixture_snapshot())
    status = json.loads(files["bundle_status.json"])
    assert status["network_access_used"] is False
    assert status["installation_performed"] is False
    assert status["model_weights_copied"] is False
    assert status["external_publication_performed"] is False
    readme = files["README.md"].decode("utf-8")
    assert "self-hashed final manifest/status/receipt/ledger" in readme
    assert "checkpoint, equivalence, benchmark, incorporation, and event-log" in readme
    assert "execution locks, active permits" in readme
    assert "quarantined stale permits, caches, and model weights remain excluded" in readme



def test_payload_fidelity_validation_pins_are_verified_and_partitioned():
    project_root = Path(__file__).resolve().parents[1]
    rows = envlock.collect_validation_result_pins(project_root)
    assert len(rows) == 18
    by_path = {row["path"]: row for row in rows}
    assert all(row["semantic_status"] == "verified" for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert by_path[
        "results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json"
    ]["role"] == "forensic_charge_only_not_rate_evidence"
    assert by_path[
        "results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json"
    ]["role"] == "forensic_invalidated_not_for_pooling"
    assert by_path[
        "results/revision_v1/compute_projection_165h_v2.json"
    ]["role"] == "authorized_compute_gate_not_scientific_outcome"
    assert by_path[
        "results/revision_v1/supervisor/detector_live_readonly_audit_20260813T0145Z.json"
    ]["role"] == "operational_diagnostic_not_scientific_evidence"
    assert by_path[
        "results/revision_v1/supervisor/detector_live_readonly_audit_addendum_20260813T0151Z.json"
    ]["role"] == "operational_diagnostic_not_scientific_evidence"
    assert not any("/primary/" in path or "/smoke_v2/" in path for path in by_path)
    derived = [row for row in rows if "/smoke_v3_" in row["path"]]
    assert len(derived) == 5
    assert all(row["role"] == "exploratory_validation_not_for_confirmatory_pooling" for row in derived)
    assert all(row["directory_file_count"] > 0 for row in derived)
    assert all(len(row["directory_files_sha256"]) == 64 for row in derived)
    assert not any("statistics_v1" in path or "statistics_v2" in path or "reports_v1" in path or "reports_v2" in path for path in by_path)


def test_validation_self_hash_tampering_fails_closed():
    value = {"status": "pass", "payload": [1, 2, 3]}
    value["self_sha256"] = envlock.sha256_bytes(envlock.canonical_json_bytes(value))
    envlock._verify_self_hash(value, "self_sha256", "fixture")
    value["payload"].append(4)
    with pytest.raises(envlock.EnvironmentLockError, match="self-hash mismatch"):
        envlock._verify_self_hash(value, "self_sha256", "fixture")


def test_scientific_source_contract_includes_payload_fidelity_and_power_grid():
    required = {
        "rankcloak/revision_artifacts.py",
        "rankcloak/revision_detection.py",
        "rankcloak/revision_protocol.py",
        "rankcloak/revision_tokenizer_preflight.py",
        "rankcloak/revision_invalidation.py",
        "rankcloak/revision_compute.py",
        "scripts/manage_legacy_gpu_ledger.py",
        "rankcloak/revision_evaluator_join.py",
        "rankcloak/revision_progress.py",
        "rankcloak/revision_detector_execution.py",
        "rankcloak/revision_release.py",
        "rankcloak/revision_release_index.py",
        "scripts/run_revision_detectors.py",
        "scripts/join_revision_evaluator_features.py",
        "scripts/run_revision_mixed_models.R",
        "scripts/update_revision_progress.py",
        "scripts/supervise_primary_v2.py",
        "scripts/supervise_confirmatory_v2.py",
        "scripts/revise_revision_manuscripts.py",
        "scripts/build_revision_environment_lock.py",
        "scripts/build_revision_release.py",
        "scripts/build_revision_confirmatory_release_index.py",
        "scripts/verify_revision_release.py",
        "operations/confirmatory_v2/downstream_commands.json",
        "operations/confirmatory_v2/detector_acceleration_policy_v1.json",
        "release/revision_v1_template/release_spec.json",
        "release/revision_v1_template/README.md",
        "revision_docs/DOI_RELEASE_PLAN.md",
        "revision_docs/NEURAL_DETECTOR_EXECUTION.md",
        "tests/test_revision_detection.py",
        "tests/test_revision_detector_execution.py",
        "tests/test_revision_progress.py",
        "tests/test_confirmatory_v2_orchestrator.py",
        "tests/test_revision_release.py",
        "tests/test_revision_release_index.py",
        "tests/test_revision_environment_lock.py",
        "human_study/config/power_design_grid.json",
        "human_study/power/planning_power_design_grid.csv",
        "human_study/power/simulate_power.py",
        "human_study/tests/test_power_design_grid.py",
        "human_study/power/PLANNING_RESULTS.md",
        "human_study/power/ASSUMPTIONS_AND_SENSITIVITY.md",
        "human_study/README.md",
    }
    assert required.issubset(set(envlock.SCIENTIFIC_SOURCE_PATHS))
    project_root = Path(__file__).resolve().parents[1]
    observed = {
        relative: envlock.sha256_file(project_root / relative)
        for relative in required
    }
    assert observed["human_study/config/power_design_grid.json"] == "675a5d7d303149ac2d4e8c23c6383934a790c10b299226a4259decb36d34aee6"
    assert observed["human_study/power/planning_power_design_grid.csv"] == "88c02ab24c66c128404a4dd3d710644c614859b5997d14da20798eb118fa3832"
    assert observed["human_study/power/PLANNING_RESULTS.md"] == "228c11c01deaefc4bcbeef17fd2815527bc0b7e944f76d1474da0fa9229eaccd"


def test_detector_release_closure_sources_are_explicit_and_content_pinned():
    project_root = Path(__file__).resolve().parents[1]
    scientific = envlock.collect_scientific_pins(
        project_root, verify_model_files=False
    )
    contract = scientific["detector_release_closure_source_contract"]
    expected = list(envlock.DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS)
    assert contract == {
        "paths": expected,
        "paths_sha256": envlock.sha256_bytes(
            envlock.canonical_json_bytes(expected)
        ),
        "all_paths_content_pinned_in_source_files": True,
        "generated_detector_results_copied_into_environment_bundle": False,
        "model_weights_included": False,
    }
    assert len(expected) == len(set(expected))
    assert set(expected).issubset(set(envlock.SCIENTIFIC_SOURCE_PATHS))
    pinned = {row["path"]: row for row in scientific["source_files"]}
    for relative in expected:
        assert pinned[relative] == {
            "path": relative,
            "size_bytes": (project_root / relative).stat().st_size,
            "sha256": envlock.sha256_file(project_root / relative),
        }


def test_detector_release_closure_rejects_an_unpinned_source(monkeypatch):
    monkeypatch.setattr(
        envlock,
        "DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS",
        envlock.DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS
        + ("revision_docs/unpinned_detector_closure.md",),
    )
    with pytest.raises(
        envlock.EnvironmentLockError,
        match="release-closure source contract is duplicated or unpinned",
    ):
        envlock.collect_scientific_pins(
            Path(__file__).resolve().parents[1], verify_model_files=False
        )


def test_current_environment_gate_is_authorized_165h_projection():
    project_root = Path(__file__).resolve().parents[1]
    scientific = envlock.collect_scientific_pins(
        project_root, verify_model_files=False
    )
    gate = scientific["current_compute_gate_decision"]
    assert gate["path"] == "results/revision_v1/compute_projection_165h_v2.json"
    assert gate["projection_sha256"] == (
        "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
    )
    assert gate["budget_gpu_hours"] == 165.0
    assert gate["decision"]["status"] == "go_within_budget"


def test_complete_current_human_material_tree_is_hashed_as_planning_only():
    project_root = Path(__file__).resolve().parents[1]
    scientific = envlock.collect_scientific_pins(project_root, verify_model_files=False)
    paths = {row["path"] for row in scientific["source_files"]}
    assert "human_study/irb/DRAFT_PROTOCOL.md" in paths
    assert "human_study/irb/MATERIALS_CHECKLIST.md" in paths
    assert "human_study/power/PLANNING_RESULTS.md" in paths
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in paths)
    policy = scientific["source_directory_contract"]
    assert policy["human_results_included"] is False
    assert policy["human_material_role"] == "pre_recruitment_planning_not_empirical_results"
    assert policy["raw_or_identifying_human_data_forbidden"] is True
