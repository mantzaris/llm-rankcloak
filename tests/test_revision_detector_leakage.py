from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_detection import build_evaluation_splits, normalize_detector_frame
from rankcloak.revision_detector_leakage import (
    DetectorLeakageAuditError,
    _sha256_int_sequence,
    _sha256_text_sequence,
    build_detector_leakage_audit,
    build_detector_leakage_sensitivity,
    canonical_json_sha256,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    # Keep enough held-out payload groups that removing the deliberately
    # implicated group cannot collapse either detector class in the
    # sensitivity fixture.
    for group in range(20):
        for label in (0, 1):
            rows.append(
                {
                    "row_id": f"row-{group}-{label}",
                    "pair_id": f"pair-{group}",
                    "source_trial_id": f"trial-{group}",
                    "text": f"Unique detector sentence for group {group} and label {label}.",
                    "label": label,
                    "payload_group_id": f"payload-{group}",
                    "prompt_template_id": "template-a",
                    "model_id": "model-a",
                    "codec_id": "codec-a",
                }
            )
    config = {
        "schema_version": "rankcloak-revision-detectors-v1",
        "seed": 91,
        "columns": {},
        "splits": {
            "regimes": ["matched"],
            "matched_test_fraction": 0.25,
            "assert_text_hash_disjoint": True,
            "minimum_train_rows": 2,
            "minimum_test_rows": 2,
        },
    }
    frame = normalize_detector_frame(pd.DataFrame(rows))
    splits, _ = build_evaluation_splits(
        frame,
        regimes=["matched"],
        test_fraction=0.25,
        seed=91,
        check_text_hash=True,
        minimum_train_rows=2,
        minimum_test_rows=2,
    )
    train_position = int(splits[0].train_indices[0])
    test_position = int(splits[0].test_indices[0])
    base = (
        "A deliberately long near duplicate sentence contains stable lexical material "
        "for the detector leakage audit and differs only at its final punctuation"
    )
    rows[train_position]["text"] = base + "."
    rows[test_position]["text"] = base + "!"
    frame = normalize_detector_frame(pd.DataFrame(rows))
    splits, _ = build_evaluation_splits(
        frame,
        regimes=["matched"],
        test_fraction=0.25,
        seed=91,
        check_text_hash=True,
        minimum_train_rows=2,
        minimum_test_rows=2,
    )
    split = splits[0]
    train = list(map(int, split.train_indices))
    test = list(map(int, split.test_indices))
    plan = {
        "schema_version": "rankcloak-revision-detector-execution-plan-v1",
        "split_count": 1,
        "detector_count": 1,
        "total_fit_count": 1,
        "tasks": [
            {
                "split_id": "matched",
                "train_row_count": len(train),
                "test_row_count": len(test),
                "train_indices_sha256": _sha256_int_sequence(train),
                "test_indices_sha256": _sha256_int_sequence(test),
                "train_row_ids_ordered_sha256": _sha256_text_sequence(
                    frame.iloc[train]["row_id"].tolist()
                ),
                "test_row_ids_ordered_sha256": _sha256_text_sequence(
                    frame.iloc[test]["row_id"].tolist()
                ),
            }
        ],
    }
    corpus_path = tmp_path / "detector.jsonl"
    pd.DataFrame(rows).to_json(corpus_path, orient="records", lines=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return corpus_path, config_path, plan_path


def test_detector_leakage_audit_is_hash_bound_and_adverse(tmp_path: Path) -> None:
    corpus, config, plan = _fixture(tmp_path)
    output = tmp_path / "audit"
    artifacts = build_detector_leakage_audit(
        detector_corpus=corpus,
        detector_config=config,
        execution_plan=plan,
        output_dir=output,
        command="fixture audit",
    )
    manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    assert manifest["status"] == "adverse_near_duplicate_overlap_detected"
    assert manifest["summary"]["affected_split_count"] == 1
    assert manifest["summary"]["exact_leakage_failed_split_count"] == 0
    assert artifacts.near_duplicate_pair_count >= 1
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["size_bytes"]


def test_detector_leakage_audit_rejects_execution_plan_drift(tmp_path: Path) -> None:
    corpus, config, plan = _fixture(tmp_path)
    value = json.loads(plan.read_text())
    value["tasks"][0]["test_row_count"] += 2
    plan.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DetectorLeakageAuditError, match="partition identity"):
        build_detector_leakage_audit(
            detector_corpus=corpus,
            detector_config=config,
            execution_plan=plan,
            output_dir=tmp_path / "audit",
        )


def test_detector_leakage_sensitivity_excludes_complete_payload_groups(
    tmp_path: Path,
) -> None:
    corpus, config, plan = _fixture(tmp_path)
    audit_dir = tmp_path / "audit"
    audit = build_detector_leakage_audit(
        detector_corpus=corpus,
        detector_config=config,
        execution_plan=plan,
        output_dir=audit_dir,
    )
    raw = pd.read_json(corpus, lines=True)
    normalized = normalize_detector_frame(raw)
    splits, _ = build_evaluation_splits(
        normalized,
        regimes=["matched"],
        test_fraction=0.25,
        seed=91,
        check_text_hash=True,
        minimum_train_rows=2,
        minimum_test_rows=2,
    )
    test = normalized.iloc[list(splits[0].test_indices)].copy()
    predictions = pd.DataFrame(
        {
            "split_id": "matched",
            "regime": "matched",
            "held_out_value": "not_applicable",
            "detector_name": "fixture_detector",
            "requested_kind": "fixture",
            "implementation_kind": "fixture",
            "implementation_status": "complete",
            "row_id": test["row_id"].tolist(),
            "payload_group_id": test["payload_group_id"].tolist(),
            "label": test["label"].astype(int).tolist(),
            "score": [0.9 if value == 1 else 0.1 for value in test["label"]],
        }
    )
    prediction_path = tmp_path / "detector_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    run = {
        "schema_version": "rankcloak-revision-detector-run-v2",
        "execution_mode": "confirmatory",
        "confirmatory_complete": True,
        "completed_fit_count": 1,
        "total_fit_count": 1,
        "failure_count": 0,
        "smoke_fallback_metric_rows": 0,
        "device": "cuda:0",
        "output_dir": str(tmp_path),
        "output_files": {
            "detector_predictions.csv": {
                "sha256": file_sha256(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
            }
        },
    }
    run["manifest_sha256"] = canonical_json_sha256(run)
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    sensitivity_config = {
        "schema_version": "rankcloak-detector-near-duplicate-sensitivity-config-v1",
        "analysis_status": "exploratory_post_partial_checkpoint_leakage_diagnostic",
        "partial_checkpoint_outcomes_seen_before_extension": True,
        "frozen_training_design_unchanged": True,
        "frozen_split_design_unchanged": True,
        "expected_fit_count": 1,
        "cosine_similarity_threshold": 0.95,
        "exclusion_unit": "test_payload_group_id",
        "exclusion_rule": "exclude_entire_test_payload_group_if_any_test_row_has_declared_near_duplicate_in_training",
        "decision_threshold": 0.5,
        "low_false_positive_rates": [0.01, 0.05],
        "bootstrap": {
            "unit": "payload_group_id",
            "resamples": 20,
            "seed": 42,
            "confidence_level": 0.95,
        },
    }
    sensitivity_config_path = tmp_path / "sensitivity.json"
    sensitivity_config_path.write_text(json.dumps(sensitivity_config), encoding="utf-8")
    artifacts = build_detector_leakage_sensitivity(
        detector_run_manifest=run_path,
        leakage_audit_manifest=audit.manifest_path,
        sensitivity_config=sensitivity_config_path,
        output_dir=tmp_path / "sensitivity-output",
    )
    manifest = json.loads(Path(artifacts.manifest_path).read_text())
    assert artifacts.fit_count == 1
    assert artifacts.affected_fit_count == 1
    assert artifacts.metric_row_count == 10
    metrics = pd.read_csv(manifest["outputs"]["metrics"]["path"])
    assert set(metrics["excluded_payload_groups"]) == {1}
    assert set(metrics["excluded_test_rows"]) == {2}
    assert (metrics["restricted_minus_original"].abs() < 1e-12).all()
