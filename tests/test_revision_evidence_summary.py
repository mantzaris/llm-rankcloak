from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rankcloak.revision_evidence_summary import (
    EvidenceSummaryError,
    HISTORICAL_GPU_HOURS_FLOOR,
    build_evidence_summary,
    canonical_json_sha256,
    file_sha256,
)


HEAD = "a" * 40


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    package = root / "results" / "revision_v1" / "final_experiment_package"
    package.mkdir(parents=True)
    evidence = root / "results" / "evidence.csv"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("outcome,estimate\nrecovery,1.0\n", encoding="utf-8")
    corpus = root / "release_inputs" / "payloads.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "payload_class": payload_class,
                    "algorithm": algorithm,
                    "artifact_bit_length": bits,
                }
            )
            + "\n"
            for payload_class, algorithm, bits in (
                ("digest", "SHA-256", 256),
                ("ciphertext", "AES-256-GCM", 384),
            )
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    corpus_sha = "c" * 64
    payload_manifest = _write(
        root / "release_inputs" / "PAYLOAD_MANIFEST.json",
        {
            "schema_version": "1.0",
            "manifest_type": "revision_payload_corpus",
            "payload_count": 4,
            "class_counts": {"ciphertext": 2, "digest": 2},
            "payload_file_sha256": file_sha256(corpus),
            "payload_file_size_bytes": corpus.stat().st_size,
            "corpus_sha256": corpus_sha,
            "cryptography_version": "46.0.7",
            "validation": {
                "status": "ok",
                "errors": [],
                "invalid_payload_names": [],
                "payload_count": 4,
                "class_counts": {"ciphertext": 2, "digest": 2},
                "corpus_sha256": corpus_sha,
            },
        },
    )
    progress = _write(
        root / "results" / "progress.json",
        {
            "counts": {
                "total": 16,
                "completed": 16,
                "successes": 15,
                "unavailable": 1,
                "failures": 0,
                "remaining": 0,
            },
            "stage_progress": [
                {
                    "stage": "primary_v2",
                    "total": 12,
                    "completed": 12,
                    "successes": 11,
                    "unavailable": 1,
                    "failures": 0,
                    "remaining": 0,
                },
                {
                    "stage": "neural_detector",
                    "total": 4,
                    "completed": 4,
                    "successes": 4,
                    "unavailable": 0,
                    "failures": 0,
                    "remaining": 0,
                },
            ],
        },
    )
    prompts = _write(
        root / "configs" / "prompts.json",
        {"category_count": 2, "template_count": 6},
    )
    models = _write(
        root / "configs" / "models.json",
        {"planned_model_count": 2, "models": [{"model_id": "a"}, {"model_id": "b"}]},
    )
    detector = _write(
        root / "results" / "detector_run_manifest.json",
        {
            "schema_version": "rankcloak-revision-detector-run-v2",
            "execution_mode": "confirmatory",
            "confirmatory_complete": True,
            "device": "cuda:0",
            "failure_count": 0,
            "completed_fit_count": 4,
            "total_fit_count": 4,
            "gpu_accounting": {
                "derivation_policy": "fixture_wall_intervals",
                "gpu_uuid": "GPU-fixture",
                "cumulative_elapsed_seconds": 7200.0,
                "intervals": [{"elapsed_seconds": 3600.0}, {"elapsed_seconds": 3600.0}],
            },
        },
    )
    human_status = _write(
        root / "results" / "human_evaluation_status.json",
        {
            "schema_version": "rankcloak-human-evaluation-status-v1",
            "status": "UNCOLLECTED_BLOCKED_NO_HUMAN_PARTICIPANT_DATA",
            "human_participant_rows": 0,
            "human_rating_rows": 0,
            "human_outcomes_estimated": False,
            "automated_metrics_are_human_rating_substitutes": False,
            "recruitment_authorized": False,
            "survey_deployed": False,
        },
    )
    base_record = {
        "status": "confirmatory",
        "artifacts": [
            str(evidence.relative_to(root)),
            str(progress.relative_to(root)),
        ],
    }
    spec = _write(
        root / "configs" / "evidence.json",
        {
            "schema_version": "rankcloak-revision-evidence-records-v1",
            "expected_head": HEAD,
            "historical_gpu_hours_floor": HISTORICAL_GPU_HOURS_FLOOR,
            "gpu_hours_ceiling": 165.0,
            "design_expectations": {
                "payload_count": 4,
                "class_count": 2,
                "per_class": 2,
                "payload_corpus_sha256": corpus_sha,
                "model_families": 2,
                "english_prompt_categories": 2,
                "english_prompt_templates": 6,
                "secondary_languages": 2,
            },
            "findings": [
                {
                    **base_record,
                    "finding_id": "F1",
                    "topic": "Recovery",
                    "configuration": "fixture",
                    "sample_size": "n=4 payloads",
                    "estimate": "1.0",
                    "uncertainty": "95% CI 0.5-1.0",
                    "effect_size": "risk difference 0.0",
                    "limitation": "fixture",
                }
            ],
            "reviewer_concerns": [
                {
                    **base_record,
                    "concern_id": "R1",
                    "request": "Expanded corpus",
                    "configuration": "fixture",
                    "sample_size": "n=4 payloads",
                    "principal_numeric_result": "4 payloads",
                    "uncertainty_or_test": "descriptive count",
                    "limitation": "fixture",
                }
            ],
            "limitations": [
                {
                    "limitation_id": "L1",
                    "item": "Human ratings",
                    "status": "unresolved",
                    "evidence": "zero ratings",
                    "artifacts": [str(evidence.relative_to(root))],
                }
            ],
            "artifact_priorities": [
                {
                    "artifact_id": "A1",
                    "artifact": "Recovery table",
                    "classification": "core evidence",
                    "rationale": "primary endpoint",
                    "source_data": str(evidence.relative_to(root)),
                    "generation_command": "fixture command",
                    "artifacts": [str(evidence.relative_to(root))],
                }
            ],
            "audit_items": [
                {
                    "audit_id": "AUD1",
                    "item": "HEAD",
                    "result": HEAD,
                    "notes": "exact baseline",
                    **base_record,
                }
            ],
        },
    )
    return {
        "root": root,
        "package": package,
        "spec": spec,
        "progress": progress,
        "corpus": corpus,
        "payload_manifest": payload_manifest,
        "prompts": prompts,
        "models": models,
        "detector": detector,
        "human_status": human_status,
    }


def _build(paths: dict[str, Path], **kwargs):
    return build_evidence_summary(
        project_root=paths["root"],
        package_root=paths["package"],
        evidence_spec=paths["spec"],
        progress_ledger=paths["progress"],
        payload_corpus=paths["corpus"],
        payload_manifest=paths["payload_manifest"],
        prompts_config=paths["prompts"],
        models_config=paths["models"],
        detector_run_manifest=paths["detector"],
        human_evaluation_status=paths["human_status"],
        observed_head=kwargs.pop("observed_head", HEAD),
        **kwargs,
    )


def test_evidence_summary_emits_hash_bound_nonpublication_artifacts(tmp_path):
    paths = _fixture(tmp_path)
    artifacts = _build(paths, command="fixture")
    assert artifacts.finding_count == 1
    assert artifacts.reviewer_concern_count == 1
    assert artifacts.total_gpu_hours == pytest.approx(HISTORICAL_GPU_HOURS_FLOOR + 2.0)
    manifest_path = Path(artifacts.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    assert manifest["scope"] == "computational_evidence_only_no_manuscript_or_response_text"
    for declaration in manifest["outputs"].values():
        path = paths["root"] / declaration["path"]
        assert file_sha256(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["size_bytes"]
    budget = json.loads((paths["package"] / "gpu_budget.json").read_text())
    budget_signature = budget.pop("budget_sha256")
    assert budget_signature == canonical_json_sha256(budget)
    assert budget["historical_floor_was_not_reset"] is True
    assert "response-letter wording" in (
        paths["package"] / "reviewer_evidence_matrix.md"
    ).read_text()
    with (paths["package"] / "tables" / "evidence_artifact_references.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        references = {row["path"]: row for row in csv.DictReader(handle)}
    assert references["results/evidence.csv"]["row_count"] == "1"
    assert references["results/progress.json"]["row_count"] == ""


def test_evidence_summary_rejects_head_mismatch(tmp_path):
    paths = _fixture(tmp_path)
    with pytest.raises(EvidenceSummaryError, match="HEAD differs"):
        _build(paths, observed_head="b" * 40)


def test_evidence_summary_rejects_gpu_ceiling_violation(tmp_path):
    paths = _fixture(tmp_path)
    detector = json.loads(paths["detector"].read_text())
    seconds = (165.0 - HISTORICAL_GPU_HOURS_FLOOR + 1.0) * 3600.0
    detector["gpu_accounting"]["cumulative_elapsed_seconds"] = seconds
    detector["gpu_accounting"]["intervals"] = [{"elapsed_seconds": seconds}]
    paths["detector"].write_text(json.dumps(detector), encoding="utf-8")
    with pytest.raises(EvidenceSummaryError, match="exceeds the hard ceiling"):
        _build(paths)


def test_evidence_summary_unions_pre_final_and_terminal_gpu_intervals(tmp_path):
    paths = _fixture(tmp_path)
    detector = json.loads(paths["detector"].read_text())
    first = {
        "pid": 101,
        "process_start_ticks": 1001,
        "started_at_utc": "2026-08-16T00:00:00+00:00",
        "completed_at_utc": "2026-08-16T01:00:00+00:00",
        "elapsed_seconds": 3600.0,
    }
    second = {
        "pid": 102,
        "process_start_ticks": 1002,
        "started_at_utc": "2026-08-16T01:00:00+00:00",
        "completed_at_utc": "2026-08-16T02:00:00+00:00",
        "elapsed_seconds": 3600.0,
    }
    third = {
        "pid": 103,
        "process_start_ticks": 1003,
        "started_at_utc": "2026-08-15T23:30:00+00:00",
        "completed_at_utc": "2026-08-16T00:00:00+00:00",
        "elapsed_seconds": 1800.0,
    }
    detector["gpu_accounting"] = {
        "derivation_policy": "terminal_fixture",
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "cumulative_elapsed_seconds": 7200.0,
        "intervals": [first, second],
    }
    detector["pre_final_gpu_accounting_ledger"] = {
        "derivation_policy": "pre_final_fixture",
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "cumulative_elapsed_seconds": 5400.0,
        "intervals": [third, first],
    }
    paths["detector"].write_text(json.dumps(detector), encoding="utf-8")

    artifacts = _build(paths)
    assert artifacts.total_gpu_hours == pytest.approx(
        HISTORICAL_GPU_HOURS_FLOOR + 2.5
    )
    budget = json.loads((paths["package"] / "gpu_budget.json").read_text())
    assert budget["new_detector_accounting_seconds"] == pytest.approx(9000.0)
    assert budget["accounting_components"] == {
        "pre_final_seconds": 5400.0,
        "pre_final_interval_count": 2,
        "terminal_seconds": 7200.0,
        "terminal_interval_count": 2,
        "duplicate_interval_count": 1,
        "union_interval_count": 3,
    }


def test_evidence_summary_validates_terminal_incorporation_of_signed_ledger(tmp_path):
    paths = _fixture(tmp_path)
    detector = json.loads(paths["detector"].read_text())
    first = {
        "pid": 101,
        "process_start_ticks": 1001,
        "started_at_utc": "2026-08-16T00:00:00+00:00",
        "completed_at_utc": "2026-08-16T01:00:00+00:00",
        "elapsed_seconds": 3600.0,
    }
    second = {
        "pid": 102,
        "process_start_ticks": 1002,
        "started_at_utc": "2026-08-16T01:00:00+00:00",
        "completed_at_utc": "2026-08-16T02:00:00+00:00",
        "elapsed_seconds": 3600.0,
    }
    detector["gpu_accounting"] = {
        "derivation_policy": "terminal_fixture",
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "cumulative_elapsed_seconds": 7200.0,
        "intervals": [first, second],
    }
    ledger = {
        "schema_version": "fixture-ledger-v1",
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "derivation_policy": "pre_final_fixture",
        "cumulative_elapsed_seconds": 3600.0,
        "intervals": [first],
        "sources": [{"source_id": "fixture"}],
        "sources_sha256": "s" * 64,
        "intervals_sha256": "i" * 64,
    }
    ledger["ledger_sha256"] = canonical_json_sha256(ledger)
    ledger_path = _write(paths["root"] / "results" / "gpu_ledger.json", ledger)
    detector["pre_final_gpu_accounting_ledger"] = {
        "path": str(ledger_path.resolve()),
        "sha256": file_sha256(ledger_path),
        "size_bytes": ledger_path.stat().st_size,
        "ledger_sha256": ledger["ledger_sha256"],
        "sources_sha256": ledger["sources_sha256"],
        "intervals_sha256": ledger["intervals_sha256"],
        "cumulative_elapsed_seconds": 3600.0,
    }
    paths["detector"].write_text(json.dumps(detector), encoding="utf-8")

    artifacts = _build(paths)
    assert artifacts.total_gpu_hours == pytest.approx(
        HISTORICAL_GPU_HOURS_FLOOR + 2.0
    )
    budget = json.loads((paths["package"] / "gpu_budget.json").read_text())
    assert budget["accounting_components"]["duplicate_interval_count"] == 1
    assert budget["accounting_components"]["union_interval_count"] == 2


def test_evidence_summary_rejects_conflicting_duplicate_gpu_interval(tmp_path):
    paths = _fixture(tmp_path)
    detector = json.loads(paths["detector"].read_text())
    first = {
        "pid": 101,
        "process_start_ticks": 1001,
        "started_at_utc": "2026-08-16T00:00:00+00:00",
        "completed_at_utc": "2026-08-16T01:00:00+00:00",
        "elapsed_seconds": 3600.0,
    }
    detector["gpu_accounting"] = {
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "cumulative_elapsed_seconds": 3600.0,
        "intervals": [first],
    }
    conflicting = dict(first)
    conflicting["elapsed_seconds"] = 3599.0
    detector["pre_final_gpu_accounting_ledger"] = {
        "device": "cuda:0",
        "gpu_uuid": "GPU-fixture",
        "cumulative_elapsed_seconds": 3599.0,
        "intervals": [conflicting],
    }
    paths["detector"].write_text(json.dumps(detector), encoding="utf-8")

    with pytest.raises(EvidenceSummaryError, match="interval differs"):
        _build(paths)


def test_evidence_summary_rejects_publication_reference(tmp_path):
    paths = _fixture(tmp_path)
    forbidden = paths["root"] / ".paper" / "main.tex"
    forbidden.parent.mkdir()
    forbidden.write_text("manuscript", encoding="utf-8")
    spec = json.loads(paths["spec"].read_text())
    spec["findings"][0]["artifacts"] = [".paper/main.tex"]
    paths["spec"].write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(EvidenceSummaryError, match="out of scope"):
        _build(paths)


def test_evidence_summary_rejects_claimed_human_ratings(tmp_path):
    paths = _fixture(tmp_path)
    status = json.loads(paths["human_status"].read_text())
    status["human_rating_rows"] = 1
    paths["human_status"].write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(EvidenceSummaryError, match="zero uncollected ratings"):
        _build(paths)


def test_evidence_summary_rejects_unvalidated_payload_manifest(tmp_path):
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["payload_manifest"].read_text())
    manifest["validation"]["invalid_payload_names"] = ["bad-payload"]
    paths["payload_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvidenceSummaryError, match="cryptographic validation"):
        _build(paths)


def test_evidence_summary_refuses_unrequested_overwrite(tmp_path):
    paths = _fixture(tmp_path)
    _build(paths)
    with pytest.raises(EvidenceSummaryError, match="Refusing to overwrite"):
        _build(paths)
