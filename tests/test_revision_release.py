import hashlib
import json
from pathlib import Path

import pytest

import rankcloak.revision_release as revision_release
from rankcloak.revision_release import (
    DRAFT_CONTENT_STATUS,
    EXTERNAL_STATUS,
    FINAL_CONTENT_STATUS,
    RevisionReleaseError,
    build_release_candidate,
    plan_release,
    publish_release,
    validate_release_spec,
    verify_release_candidate,
)


REQUIRED_GROUPS = [
    "source_code",
    "configs",
    "public_payload_corpus",
    "raw_results",
    "processed_results",
    "statistics_outputs",
    "figure_table_outputs",
    "human_materials",
    "environment_inputs",
]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def authoritative_reference_fixture(
    tmp_path: Path, checkout_name: str = "fixture-checkout"
):
    root = tmp_path / checkout_name
    root.mkdir(parents=True)
    source = "results/revision_v1/final_experiment_package"
    package_relative = "verified.txt"
    package_file = write(root / source / package_relative, "sealed package file\n")
    external_relative = "tracked/external-reference.txt"
    external_file = write(root / external_relative, "required external bytes\n")
    historical_path = (
        "/" + "home/historical-user/Documents/repos/llm-rankcloak/"
        + external_relative
    )
    manifest = {
        "schema_version": "rankcloak-final-experiment-package-index-v1",
        "status": "passed",
        "package_files": [
            {
                "path": package_relative,
                "size_bytes": package_file.stat().st_size,
                "sha256": hashlib.sha256(package_file.read_bytes()).hexdigest(),
            }
        ],
        "external_references": [
            {
                "label": "required-reference",
                "path": historical_path,
                "size_bytes": external_file.stat().st_size,
                "sha256": hashlib.sha256(external_file.read_bytes()).hexdigest(),
            }
        ],
        "summary": {"package_file_count": 1, "external_reference_count": 1},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = write(
        root / source / "manifest.json",
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    spec = {
        "authoritative_evidence_package": {
            "source": source,
            "manifest": source + "/manifest.json",
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "status": "passed",
            "required_external_labels": ["required-reference"],
            "external_reference_mappings": [
                {
                    "label": "required-reference",
                    "repository_path": external_relative,
                }
            ],
        },
        "artifacts": {
            group: [] for group in revision_release.ARTIFACT_GROUPS
        },
    }
    spec["artifacts"]["figure_table_outputs"] = [
        {"source": source, "destination": source, "exclude_paths": []}
    ]
    included_files = [
        {
            "source": source + "/" + package_relative,
            "size_bytes": package_file.stat().st_size,
            "sha256": hashlib.sha256(package_file.read_bytes()).hexdigest(),
        }
    ]
    return root, spec, included_files, {external_relative}


def resign_authoritative_manifest(root: Path, spec, mutate) -> None:
    declaration = spec["authoritative_evidence_package"]
    path = root / declaration["manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    declaration["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def add_verified_environment_snapshot(root: Path, spec, r_status="complete") -> Path:
    directory = root / "environment/fixture_revision_v1"
    requirements = write(directory / "requirements-lock.txt", "-e .\n")
    packages = [{"name": "fixture", "version": "1.2.3"}]
    for relative in revision_release._environment_lock_source_contract()[0]:
        path = root / relative
        if not path.exists():
            write(path, "fixture scientific source {}\n".format(relative))
    expected_sources = revision_release._expected_environment_scientific_source_paths(
        root
    )
    assert expected_sources is not None
    source_files = [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        }
        for relative in sorted(expected_sources)
    ]
    live_root = Path(__file__).resolve().parents[1]
    projection_source = (
        live_root / "results/revision_v1/compute_projection_165h_v2.json"
    )
    projection_path = root / "results/revision_v1/compute_projection_165h_v2.json"
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection_path.write_bytes(projection_source.read_bytes())
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    authorized_hash = projection["projection_sha256"]
    decision = projection["decision"]
    validation_results = [{
        "path": "results/revision_v1/compute_projection_165h_v2.json",
        "role": "authorized_compute_gate_not_scientific_outcome",
        "size_bytes": projection_path.stat().st_size,
        "sha256": hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        "semantic_status": "verified",
    }]
    payloads = {
        "README.md": "fixture environment\n",
        "REPRODUCE.md": "offline verification only\n",
        "CHECKSUMS.sha256": "fixture checksums\n",
        "backend_cuda_hardware.json": "{}\n",
        "determinism.json": "{}\n",
        "bundle_status.json": json.dumps(
            {
                "status": "complete",
                "network_access_used": False,
                "installation_performed": False,
                "model_weights_copied": False,
                "external_publication_performed": False,
            }
        )
        + "\n",
        "python_environment.json": json.dumps(
            {
                "packages": packages,
                "packages_sha256": canonical_sha256(packages),
                "requirements_lock_sha256": hashlib.sha256(
                    requirements.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        "r_environment.json": json.dumps({"status": r_status}) + "\n",
        "scientific_pins.json": json.dumps({
            "model_weights_included": False,
            "source_files": source_files,
            "source_files_sha256": canonical_sha256(source_files),
            "validation_result_artifacts": validation_results,
            "validation_result_artifacts_sha256": canonical_sha256(
                validation_results
            ),
            "current_compute_gate_decision": {
                "path": "results/revision_v1/compute_projection_165h_v2.json",
                "projection_sha256": authorized_hash,
                "budget_gpu_hours": 165.0,
                "decision": decision,
            },
        })
        + "\n",
    }
    for name, content in payloads.items():
        write(directory / name, content)
    records = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "environment_manifest.json"
    ]
    write(
        directory / "environment_manifest.json",
        json.dumps(
            {
                "manifest_type": "rankcloak_revision_environment_file_set",
                "snapshot_status": "complete",
                "file_count": len(records),
                "files_sha256": canonical_sha256(records),
                "files": records,
            },
            sort_keys=True,
        )
        + "\n",
    )
    spec["artifacts"]["environment_inputs"] = [
        {
            "source": "environment/fixture_revision_v1",
            "destination": "environment/revision_v1",
            "required": True,
        }
    ]
    return directory


def fixture_project(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir(parents=True)
    write(root / "LICENSE", "Fixture permissive license\n")
    write(
        root / "CITATION.cff",
        "cff-version: 1.2.0\ntitle: Fixture release\ntype: software\n",
    )
    artifacts = {}
    for index, group in enumerate(REQUIRED_GROUPS):
        if group == "environment_inputs":
            source = "inputs/requirements-lock.txt"
            content = "fixture==1.0 --hash=sha256:{}\n".format("a" * 64)
        else:
            source = "inputs/{}/artifact-{}.txt".format(group, index)
            content = "deterministic {} fixture\n".format(group)
        write(root / source, content)
        entry = {
            "source": source,
            "destination": "{}/artifact-{}.txt".format(group, index),
            "required": True,
        }
        if group == "raw_results":
            entry["evidence_role"] = (
                "exploratory_validation_not_for_confirmatory_pooling"
            )
        artifacts[group] = [entry]
    artifacts["documentation"] = []
    spec = {
        "schema_version": "1.0",
        "package_id": "fixture-release",
        "release_status": EXTERNAL_STATUS,
        "metadata": {
            "title": "Fixture reproducibility package",
            "version": "1.0.0",
            "description": "A deterministic offline fixture package.",
            "creators": [{"name": "Fixture Author"}],
            "doi": None,
            "direct_participant_identifiers_included": False,
            "raw_human_response_data_included": False,
            "human_participant_outcomes_collected": False,
        },
        "license_file": "LICENSE",
        "citation_file": "CITATION.cff",
        "required_groups": list(REQUIRED_GROUPS),
        "artifacts": artifacts,
        "third_party": [
            {
                "id": "fixture_dependency",
                "name": "Fixture Dependency",
                "license": "MIT",
                "revision": "1.0",
                "artifact_included": False,
            }
        ],
        "reproduction_commands": [
            {"label": "Run fixture", "command": "python -m pytest -q"}
        ],
    }
    return root, spec


def test_final_ready_fixture_builds_hashed_offline_candidate(tmp_path):
    root, spec = fixture_project(tmp_path)
    target = tmp_path / "candidate"
    report = build_release_candidate(
        spec, target, project_root=root, require_final_ready=True
    )
    assert report["readiness"]["content_readiness"] == FINAL_CONTENT_STATUS
    assert report["readiness"]["publication_ready"] is False
    assert {row["code"] for row in report["readiness"]["publication_blockers"]} == {
        "external_publication_authorization_absent",
        "doi_not_assigned",
    }
    assert report["release_status"] == EXTERNAL_STATUS
    assert report["external_action_performed"] is False
    assert report["network_access_used"] is False
    manifest = json.loads((target / "PACKAGE_MANIFEST.json").read_text())
    assert manifest["release_status"] == EXTERNAL_STATUS
    assert manifest["doi"] is None
    assert manifest["doi_provenance"] == "not_assigned"
    assert not (target / "models").exists()
    for row in manifest["files"]:
        digest = hashlib.sha256((target / row["path"]).read_bytes()).hexdigest()
        assert digest == row["sha256"]
    manifest_digest = hashlib.sha256(
        (target / "PACKAGE_MANIFEST.json").read_bytes()
    ).hexdigest()
    assert (target / "PACKAGE_MANIFEST.sha256").read_text().startswith(
        manifest_digest
    )
    continuous_metadata = json.loads((target / "ASSEMBLY_REPORT.json").read_text())
    assert continuous_metadata["doi_minted_or_reserved_by_assembler"] is False
    verification = verify_release_candidate(target)
    assert verification["status"] == "ok"
    assert verification["doi"] is None
    assert verification["model_weights_included"] is False
    assert verification["package_manifest_sha256"] == manifest_digest


def test_dry_run_reports_missing_without_creating_output(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["artifacts"]["raw_results"][0]["source"] = "missing/raw.jsonl"
    target = tmp_path / "never-created"
    report = build_release_candidate(
        spec, target, project_root=root, dry_run=True
    )
    assert report["dry_run"] is True
    assert report["output_created"] is False
    assert not target.exists()
    assert report["resolution"]["missing_artifacts"] == [
        {"group": "raw_results", "source": "missing/raw.jsonl", "required": True}
    ]
    assert report["readiness"]["content_readiness"] == DRAFT_CONTENT_STATUS


def test_requirements_lock_name_alone_does_not_pass_exact_lock_gate(tmp_path):
    root, spec = fixture_project(tmp_path)
    lock = root / spec["artifacts"]["environment_inputs"][0]["source"]
    lock.write_text("fixture>=1.0\n", encoding="utf-8")
    report = plan_release(spec, project_root=root)
    assert report["environment_inventory"]["exact_lock_input_present"] is False
    assert any(
        blocker["code"] == "missing_exact_environment_lock"
        for blocker in report["readiness"]["blockers"]
    )


def test_verified_environment_snapshot_satisfies_gate_and_tampering_fails(tmp_path):
    root, spec = fixture_project(tmp_path)
    directory = add_verified_environment_snapshot(root, spec)
    report = plan_release(spec, project_root=root)
    inventory = report["environment_inventory"]
    assert inventory["exact_lock_input_present"] is True
    assert inventory["verified_environment_snapshot_present"] is True
    assert inventory["lock_status"] == "verified_observed_environment_snapshot_supplied"
    assert inventory["verified_environment_snapshot_sources"] == [
        "environment/fixture_revision_v1/environment_manifest.json"
    ]
    scientific_source = root / "rankcloak/revision_release.py"
    original_source = scientific_source.read_text(encoding="utf-8")
    scientific_source.write_text("stale\n", encoding="utf-8")
    stale = plan_release(spec, project_root=root)
    assert stale["environment_inventory"]["verified_environment_snapshot_present"] is False
    scientific_source.write_text(original_source, encoding="utf-8")
    (directory / "r_environment.json").write_text(
        json.dumps({"status": "incomplete"}) + "\n", encoding="utf-8"
    )
    tampered = plan_release(spec, project_root=root)
    assert tampered["environment_inventory"]["exact_lock_input_present"] is False
    assert any(
        blocker["code"] == "missing_exact_environment_lock"
        for blocker in tampered["readiness"]["blockers"]
    )


def test_explicit_external_reference_mapping_validates(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    report = revision_release.verify_authoritative_evidence_package(
        spec, root, included, tracked_files=tracked
    )
    assert report["status"] == "verified_complete"
    assert report["verified_external_reference_count"] == 1


def test_explicit_mapping_is_independent_of_checkout_basename(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(
        tmp_path, checkout_name="unrelated-portability-name"
    )
    manifest_path = root / spec["authoritative_evidence_package"]["manifest"]
    historical = json.loads(manifest_path.read_text())["external_references"][0][
        "path"
    ]
    assert "/llm-rankcloak/" in historical
    assert root.name not in historical
    assert revision_release.verify_authoritative_evidence_package(
        spec, root, included, tracked_files=tracked
    )["status"] == "verified_complete"


def test_explicit_mapping_rejects_wrong_repository_path(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    wrong = write(root / "tracked/wrong.txt", "different bytes\n")
    mapping = spec["authoritative_evidence_package"][
        "external_reference_mappings"
    ][0]
    mapping["repository_path"] = "tracked/wrong.txt"
    tracked.add("tracked/wrong.txt")
    assert wrong.is_file()
    with pytest.raises(RevisionReleaseError, match="external reference differs"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_explicit_mapping_rejects_wrong_frozen_hash(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)

    def mutate(manifest):
        manifest["external_references"][0]["sha256"] = "0" * 64

    resign_authoritative_manifest(root, spec, mutate)
    with pytest.raises(RevisionReleaseError, match="external reference differs"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_explicit_mapping_rejects_duplicate_label(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    mappings = spec["authoritative_evidence_package"][
        "external_reference_mappings"
    ]
    mappings.append(dict(mappings[0]))
    with pytest.raises(RevisionReleaseError, match="Duplicate external-reference"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_explicit_mapping_rejects_required_label_without_mapping(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    spec["authoritative_evidence_package"]["required_external_labels"].append(
        "undeclared-required-reference"
    )
    with pytest.raises(RevisionReleaseError, match="lack explicit mappings"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_explicit_mapping_rejects_mapping_not_declared_required(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    spec["authoritative_evidence_package"]["external_reference_mappings"].append(
        {"label": "extra", "repository_path": "tracked/external-reference.txt"}
    )
    with pytest.raises(RevisionReleaseError, match="not declared required"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


@pytest.mark.parametrize("unsafe", ["/absolute/reference.txt", "../escape.txt"])
def test_explicit_mapping_rejects_unsafe_path(tmp_path, unsafe):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    spec["authoritative_evidence_package"]["external_reference_mappings"][0][
        "repository_path"
    ] = unsafe
    with pytest.raises(RevisionReleaseError, match="safe relative path"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_explicit_mapping_rejects_untracked_file(tmp_path):
    root, spec, included, _tracked = authoritative_reference_fixture(tmp_path)
    with pytest.raises(RevisionReleaseError, match="is not tracked"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=set()
        )


def test_explicit_mapping_rejects_symlink(tmp_path):
    root, spec, included, tracked = authoritative_reference_fixture(tmp_path)
    link_relative = "tracked/external-link.txt"
    (root / link_relative).symlink_to(root / "tracked/external-reference.txt")
    spec["authoritative_evidence_package"]["external_reference_mappings"][0][
        "repository_path"
    ] = link_relative
    tracked.add(link_relative)
    with pytest.raises(RevisionReleaseError, match="not a regular tracked file"):
        revision_release.verify_authoritative_evidence_package(
            spec, root, included, tracked_files=tracked
        )


def test_real_release_template_uses_tracked_environment_and_final_evidence():
    project_root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (project_root / "release/revision_v1_template/release_spec.json").read_text()
    )
    environment_entries = spec["artifacts"]["environment_inputs"]
    assert len(environment_entries) == 1
    assert environment_entries[0]["source"] == "environment/revision_v1"
    assert environment_entries[0]["destination"] == "environment/revision_v1"
    assert environment_entries[0]["required"] is True
    report = plan_release(spec, project_root=project_root)
    assert report["environment_inventory"]["exact_lock_input_present"] is True
    authoritative = report["authoritative_evidence_verification"]
    assert authoritative["status"] == "verified_complete"
    assert authoritative["included_package_file_count"] == 130
    assert authoritative["source_package_file_count"] == 203
    assert authoritative["verified_external_reference_count"] == 14
    assert report["confirmatory_artifact_verification"]["status"] == "not_required"
    assert report["readiness"]["final_ready"] is True
    assert report["readiness"]["blockers"] == []


def test_no_overwrite_is_enforced_and_existing_bytes_survive(tmp_path):
    root, spec = fixture_project(tmp_path)
    target = tmp_path / "candidate"
    target.mkdir()
    sentinel = write(target / "keep.txt", "keep me\n")
    with pytest.raises(RevisionReleaseError, match="already exists"):
        build_release_candidate(spec, target, project_root=root)
    assert sentinel.read_text() == "keep me\n"


def test_model_weights_external_checkouts_caches_and_raw_human_data_are_excluded(tmp_path):
    root, spec = fixture_project(tmp_path)
    source_dir = root / "bundle"
    write(source_dir / "safe.py", "print('safe')\n")
    write(source_dir / "models" / "model.gguf", "weight bytes")
    write(source_dir / "external_sources" / "repo" / "code.py", "vendored")
    write(source_dir / "__pycache__" / "safe.pyc", "cache")
    write(source_dir / "raw_responses.csv", "participant_id,response\n1,text\n")
    spec["artifacts"]["source_code"] = [
        {"source": "bundle", "destination": "source/bundle", "required": True}
    ]
    report = plan_release(spec, project_root=root)
    included = {row["source"] for row in report["resolution"]["files"]}
    assert "bundle/safe.py" in included
    assert not any(path.endswith(".gguf") for path in included)
    reasons = {row["reason"] for row in report["resolution"]["exclusions"]}
    assert "model_weight" in reasons or any(
        reason.startswith("forbidden_path_component:models") for reason in reasons
    )
    assert "forbidden_path_component:external_sources" in reasons
    assert "forbidden_path_component:__pycache__" in reasons
    assert "participant_identifier_or_raw_human_data" in reasons
    assert report["readiness"]["final_ready"] is False


def test_secret_signatures_and_symlinks_are_never_copied(tmp_path):
    root, spec = fixture_project(tmp_path)
    write(
        root / "sensitive" / "secret.txt",
        "-----BEGIN PRIVATE KEY-----\n"
        + ("A" * 120)
        + "\n-----END PRIVATE KEY-----\n",
    )
    safe = write(root / "sensitive" / "safe.txt", "safe\n")
    (root / "sensitive" / "linked.txt").symlink_to(safe)
    spec["artifacts"]["source_code"] = [
        {"source": "sensitive", "destination": "source/sensitive", "required": True}
    ]
    report = plan_release(spec, project_root=root)
    reasons = {row["reason"] for row in report["resolution"]["exclusions"]}
    assert "high_confidence_secret_signature" in reasons
    assert "symlink" in reasons
    target = tmp_path / "draft"
    build_release_candidate(spec, target, project_root=root)
    assert (target / "source/sensitive/safe.txt").is_file()
    assert not (target / "source/sensitive/secret.txt").exists()
    assert not (target / "source/sensitive/linked.txt").exists()


def test_placeholders_block_final_ready_but_are_reported_in_draft(tmp_path):
    root, spec = fixture_project(tmp_path)
    source = root / spec["artifacts"]["human_materials"][0]["source"]
    source.write_text("Contact {{STUDY_CONTACT_EMAIL}}\n", encoding="utf-8")
    plan = plan_release(spec, project_root=root)
    assert any(
        row["code"] == "human_material_placeholder_unresolved"
        for row in plan["readiness"]["publication_blockers"]
    )
    assert any(
        blocker["code"] == "unresolved_placeholder"
        for blocker in plan["readiness"]["blockers"]
    )
    with pytest.raises(RevisionReleaseError, match="not final-ready"):
        build_release_candidate(
            spec, tmp_path / "blocked", project_root=root,
            require_final_ready=True,
        )
    draft = tmp_path / "draft"
    build_release_candidate(spec, draft, project_root=root)
    assembly = json.loads((draft / "ASSEMBLY_REPORT.json").read_text())
    assert assembly["content_readiness"]["final_ready"] is False


def test_network_publication_keys_and_publication_function_are_refused(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["zenodo"] = {"deposit": True}
    with pytest.raises(RevisionReleaseError, match="prohibited external action"):
        validate_release_spec(spec)
    with pytest.raises(RevisionReleaseError, match="External release actions are prohibited"):
        publish_release("anything")


def test_doi_is_never_fabricated_and_only_valid_user_input_is_preserved(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["metadata"]["doi"] = "invent-this-later"
    with pytest.raises(RevisionReleaseError, match="not syntactically valid"):
        plan_release(spec, project_root=root)
    spec["metadata"]["doi"] = "10.1234/user.supplied"
    target = tmp_path / "candidate"
    build_release_candidate(spec, target, project_root=root, require_final_ready=True)
    manifest = json.loads((target / "PACKAGE_MANIFEST.json").read_text())
    assert manifest["doi"] == "10.1234/user.supplied"
    assert manifest["doi_provenance"] == "user_supplied_unverified"


def test_destination_collisions_and_path_traversal_are_rejected(tmp_path):
    root, spec = fixture_project(tmp_path)
    write(root / "inputs/other.txt", "other\n")
    spec["artifacts"]["source_code"].append(
        {
            "source": "inputs/other.txt",
            "destination": spec["artifacts"]["source_code"][0]["destination"],
            "required": True,
        }
    )
    with pytest.raises(RevisionReleaseError, match="Two allowlist inputs"):
        plan_release(spec, project_root=root)
    root, spec = fixture_project(tmp_path / "second")
    spec["artifacts"]["source_code"][0]["source"] = "../outside"
    with pytest.raises(RevisionReleaseError, match="safe relative"):
        plan_release(spec, project_root=root)


def test_evidence_roles_are_validated_propagated_and_staged(tmp_path):
    root, spec = fixture_project(tmp_path)
    entry = spec["artifacts"]["raw_results"][0]
    entry["evidence_role"] = "exploratory_validation_not_for_confirmatory_pooling"
    plan = plan_release(spec, project_root=root)
    partition = next(
        row for row in plan["artifact_evidence_partitions"]
        if row["group"] == "raw_results"
    )
    assert partition["evidence_role"] == "exploratory_validation_not_for_confirmatory_pooling"
    assert plan["resolution"]["files"][0].get("evidence_role") is not None or any(
        row.get("evidence_role") == "exploratory_validation_not_for_confirmatory_pooling"
        for row in plan["resolution"]["files"]
    )
    target = tmp_path / "role-candidate"
    build_release_candidate(spec, target, project_root=root, require_final_ready=True)
    staged = json.loads((target / "ARTIFACT_EVIDENCE_ROLES.json").read_text())
    assert any(
        row["evidence_role"] == "exploratory_validation_not_for_confirmatory_pooling"
        for row in staged
    )
    assembly = json.loads((target / "ASSEMBLY_REPORT.json").read_text())
    assert assembly["artifact_evidence_partitions"] == staged


def test_raw_result_roles_and_legacy_exclusions_fail_closed(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["artifacts"]["raw_results"][0].pop("evidence_role")
    with pytest.raises(RevisionReleaseError, match="requires an explicit evidence_role"):
        validate_release_spec(spec)

    root, spec = fixture_project(tmp_path / "bad-role")
    spec["artifacts"]["raw_results"][0]["evidence_role"] = "scientific-ish"
    with pytest.raises(RevisionReleaseError, match="Unsupported evidence_role"):
        validate_release_spec(spec)

    root, spec = fixture_project(tmp_path / "legacy")
    entry = spec["artifacts"]["raw_results"][0]
    entry["source"] = "results/revision_v1/smoke_v2/model"
    entry["evidence_role"] = "exploratory_validation_not_for_confirmatory_pooling"
    with pytest.raises(RevisionReleaseError, match="Legacy smoke_v2"):
        validate_release_spec(spec)

    root, spec = fixture_project(tmp_path / "invalid-primary")
    entry = spec["artifacts"]["raw_results"][0]
    entry["source"] = "results/revision_v1/primary"
    with pytest.raises(RevisionReleaseError, match="invalidated legacy primary"):
        validate_release_spec(spec)


def test_real_template_uses_current_tracked_sources_and_authoritative_package():
    project_root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (project_root / "release/revision_v1_template/release_spec.json").read_text()
    )
    serialized = json.dumps(spec, sort_keys=True).lower()
    assert spec["tracked_sources_only"] is True
    assert "confirmatory_artifact_index" not in spec
    assert ".paper/" not in serialized
    assert "confirmatory_release_index_v1.json" not in serialized
    assert "detector_equivalence_v1" not in serialized
    assert "equivalence" not in serialized
    assert "human_materials" not in spec["required_groups"]
    commands = [row["command"] for row in spec["reproduction_commands"]]
    assert any(
        "not textcnn_reports_reproducible_post_training_state_hash" in command
        for command in commands
    )
    assert "python -m pytest -q" not in commands
    assert "python -B scripts/verify_revision_release.py . --require-doi-null" in commands

    entries = [
        row
        for group in revision_release.ARTIFACT_GROUPS
        for row in spec["artifacts"].get(group, [])
    ]
    assert entries
    assert all(not Path(row["source"]).is_absolute() for row in entries)
    assert all((project_root / row["source"]).exists() for row in entries)
    assert all(not row["source"].startswith(".paper/") for row in entries)
    final_package = next(
        row for row in entries
        if row["source"] == "results/revision_v1/final_experiment_package"
    )
    assert final_package["evidence_role"] == "authoritative_final_evidence"
    assert final_package["portable_paths"] is True
    authoritative = spec["authoritative_evidence_package"]
    assert authoritative["manifest_sha256"] == (
        "b7ec0c47a59fa6a9a33de2fd072f9e2e4db4c38328cc8757aa5fa562451e2349"
    )
    mappings = {
        row["label"]: row["repository_path"]
        for row in authoritative["external_reference_mappings"]
    }
    assert set(mappings) == set(authoritative["required_external_labels"])
    assert all(
        not Path(path).is_absolute() and ".." not in Path(path).parts
        for path in mappings.values()
    )
    tracked = revision_release._tracked_repository_files(project_root)
    assert all(path in tracked for path in mappings.values())


def test_real_template_resolves_clean_checkout_and_dry_runs_final_ready(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (project_root / "release/revision_v1_template/release_spec.json").read_text()
    )
    target = tmp_path / "not-created"
    report = build_release_candidate(
        spec, target, project_root=project_root, dry_run=True,
        require_final_ready=True,
    )
    assert report["dry_run"] is True
    assert report["output_created"] is False
    assert not target.exists()
    assert report["resolution"]["missing_artifacts"] == []
    tracked = revision_release._tracked_repository_files(project_root)
    assert all(
        row["source"] in tracked for row in report["resolution"]["files"]
    )
    assert report["readiness"]["content_readiness"] == FINAL_CONTENT_STATUS
    assert report["readiness"]["blockers"] == []
    assert report["authoritative_evidence_verification"]["status"] == (
        "verified_complete"
    )
    assert report["confirmatory_artifact_verification"]["status"] == "not_required"
    assert {
        row["code"] for row in report["readiness"]["publication_blockers"]
    } == {"external_publication_authorization_absent", "doi_not_assigned"}


def test_confirmatory_role_without_exact_index_map_is_rejected(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["artifacts"]["raw_results"][0]["evidence_role"] = (
        "confirmatory_scientific_evidence"
    )
    with pytest.raises(RevisionReleaseError, match="confirmatory_artifact_index"):
        validate_release_spec(spec)


def test_unlicensed_reference_only_bytes_are_notice_but_redistribution_blocks(tmp_path):
    root, spec = fixture_project(tmp_path)
    spec["third_party"][0]["license"] = "license_not_recorded"
    spec["third_party"][0]["artifact_included"] = False
    plan = plan_release(spec, project_root=root)
    assert plan["readiness"]["final_ready"] is True
    assert plan["readiness"]["notices"] == [
        {
            "code": "third_party_license_unresolved_reference_only_bytes_excluded",
            "detail": "fixture_dependency",
        }
    ]
    spec["third_party"][0]["artifact_included"] = True
    blocked = plan_release(spec, project_root=root)
    assert any(
        row["code"] == "third_party_license_unresolved"
        for row in blocked["readiness"]["blockers"]
    )


def test_real_template_model_license_registry_clears_model_gaps_only():
    project_root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (project_root / "release/revision_v1_template/release_spec.json").read_text()
    )
    plan = plan_release(spec, project_root=project_root)
    inventory = {row["id"]: row for row in plan["third_party_inventory"]}
    assert inventory["llama3_8b_instruct_q4_k_m"]["license"] == "Meta Llama 3 Community License (release 2024-04-18)"
    assert inventory["qwen2_5_7b_instruct_q4_k_m"]["license"] == "Apache-2.0"
    assert inventory["mistral_7b_instruct_v0_3_q4_k_m"]["license"] == "Apache-2.0"
    assert all(inventory[model_id]["artifact_included"] is False for model_id in (
        "llama3_8b_instruct_q4_k_m",
        "qwen2_5_7b_instruct_q4_k_m",
        "mistral_7b_instruct_v0_3_q4_k_m",
    ))
    unresolved = [
        row for row in plan["readiness"]["notices"]
        if row["code"] == "third_party_license_unresolved_reference_only_bytes_excluded"
    ]
    assert unresolved == [{
        "code": "third_party_license_unresolved_reference_only_bytes_excluded",
        "detail": "published_comparator_patient_huffman_acl2019",
    }]
    assert not any(
        row["code"] == "third_party_license_unresolved"
        for row in plan["readiness"]["blockers"]
    )


def test_independent_candidate_verifier_rejects_tamper_unlisted_weights_and_symlinks(tmp_path):
    root, spec = fixture_project(tmp_path)
    target = tmp_path / "verified-candidate"
    build_release_candidate(spec, target, project_root=root, require_final_ready=True)
    manifested = json.loads((target / "PACKAGE_MANIFEST.json").read_text())["files"]
    victim = target / manifested[0]["path"]
    victim.write_bytes(victim.read_bytes() + b"tamper")
    with pytest.raises(RevisionReleaseError, match="hash or size mismatch"):
        verify_release_candidate(target)

    target2 = tmp_path / "weight-candidate"
    build_release_candidate(spec, target2, project_root=root, require_final_ready=True)
    write(target2 / "unexpected/model.gguf", "not real weights")
    with pytest.raises(RevisionReleaseError, match="Prohibited staged path"):
        verify_release_candidate(target2)

    target3 = tmp_path / "symlink-candidate"
    build_release_candidate(spec, target3, project_root=root, require_final_ready=True)
    (target3 / "linked").symlink_to(target3 / "README.md")
    with pytest.raises(RevisionReleaseError, match="Symlink in staged candidate"):
        verify_release_candidate(target3)


def test_bibtex_double_braces_are_not_mistaken_for_release_placeholders(tmp_path):
    root, spec = fixture_project(tmp_path)
    bib = write(root / "paper/references.bib", "@string{pmlr = {{PMLR}}}\n")
    spec["artifacts"]["documentation"] = [{
        "source": "paper/references.bib",
        "destination": "manuscript/references.bib",
        "required": True,
        "evidence_role": "documentation_not_scientific_result",
    }]
    plan = plan_release(spec, project_root=root)
    assert not any(
        row["code"] == "unresolved_placeholder" and row["detail"]["path"] == "paper/references.bib"
        for row in plan["readiness"]["blockers"]
    )
    bib.write_text("@misc{x, note = {[INSERT DOI]}}\n", encoding="utf-8")
    blocked = plan_release(spec, project_root=root)
    assert any(
        row["code"] == "unresolved_placeholder" and row["detail"]["path"] == "paper/references.bib"
        for row in blocked["readiness"]["blockers"]
    )
