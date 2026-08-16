import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_detector_analysis import (
    DetectorAnalysisError,
    analyze_detector_outputs,
    file_sha256,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    raw.mkdir()
    prediction_rows = []
    metric_rows = []
    cells = [
        ("matched", "matched", "", "textcnn", "text_cnn"),
        ("matched", "matched", "", "deberta", "pretrained_transformer"),
        (
            "held_out_template:template_a",
            "held_out_template",
            "template_a",
            "textcnn",
            "text_cnn",
        ),
        (
            "held_out_template:template_a",
            "held_out_template",
            "template_a",
            "deberta",
            "pretrained_transformer",
        ),
    ]
    for split_id, regime, held_out, detector, kind in cells:
        for payload_index in range(4):
            # A payload group can contain multiple case/control pairs in the
            # frozen corpus; it is the clustering unit, not a two-row pair ID.
            payload = f"payload-{payload_index // 2}"
            for label, suffix, score in (
                (1, "rankcloak", 0.8 + 0.1 * (payload_index % 2)),
                (0, "control", 0.1 + 0.1 * (payload_index % 2)),
            ):
                prediction_rows.append(
                    {
                        "split_id": split_id,
                        "regime": regime,
                        "held_out_value": held_out,
                        "detector_name": detector,
                        "requested_kind": kind,
                        "implementation_kind": kind,
                        "implementation_status": "complete",
                        "row_id": (
                            f"{split_id}-{detector}-pair-{payload_index}-{suffix}"
                        ),
                        "payload_group_id": payload,
                        "prompt_template_id": "template_a",
                        "model_id": "model_a",
                        "codec_id": "codec_a",
                        "label": label,
                        "score": score,
                        "prediction": int(score >= 0.5),
                    }
                )
        row = {
            "split_id": split_id,
            "regime": regime,
            "held_out_column": (
                "prompt_template_id" if held_out else ""
            ),
            "held_out_value": held_out,
            "detector_name": detector,
            "requested_kind": kind,
            "implementation_kind": kind,
            "implementation_status": "complete",
            "train_rows": 24,
            "test_rows": 8,
            "train_payload_groups": 12,
            "purged_train_rows": 0,
            "decision_threshold": 0.5,
            "seed": 1,
            "notes": "",
            "model_state_sha256": "a" * 64,
            "model_state_hash_algorithm": "rankcloak-torch-state-v1",
            "model_artifact_set_sha256": "",
            "implementation_metadata_json": "{}",
            "bootstrap_unit": "payload_group_id",
            "bootstrap_resamples_requested": 50,
            "test_payload_groups": 4,
        }
        for metric in (
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "f1",
            "sensitivity",
            "specificity",
        ):
            row[metric] = 1.0
            row[f"{metric}_bootstrap_valid"] = 50
            row[f"{metric}_ci_low_95"] = 1.0
            row[f"{metric}_ci_high_95"] = 1.0
        metric_rows.append(row)
    predictions_path = raw / "detector_predictions.csv"
    metrics_path = raw / "detector_metrics.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": "rankcloak-revision-detector-run-v2",
        "execution_mode": "confirmatory",
        "confirmatory_complete": True,
        "completed_fit_count": 4,
        "total_fit_count": 4,
        "failure_count": 0,
        "smoke_fallback_metric_rows": 0,
        "device": "cuda:0",
        "output_dir": str(raw.resolve()),
        "output_files": {
            "detector_predictions.csv": {
                "sha256": file_sha256(predictions_path),
                "size_bytes": predictions_path.stat().st_size,
            },
            "detector_metrics.csv": {
                "sha256": file_sha256(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            },
        },
    }
    manifest_path = raw / "detector_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": "rankcloak-detector-supplementary-metrics-v1",
        "analysis_status": "supplementary_exploratory_post_confirmatory_freeze",
        "partial_checkpoint_outcomes_seen_before_extension": True,
        "frozen_training_design_unchanged": True,
        "expected_fit_count": 4,
        "decision_threshold": 0.5,
        "precision_zero_division": 0,
        "low_false_positive_rates": [0.01, 0.05],
        "supplementary_metrics": [
            "precision",
            "brier_score",
            "tpr_at_fpr_0.01",
            "tpr_at_fpr_0.05",
        ],
        "bootstrap": {
            "unit": "payload_group_id",
            "resamples": 50,
            "seed": 17,
            "confidence_level": 0.95,
        },
        "low_fpr_interpretation": (
            "descriptive_test_set_roc_curve_summary_not_a_deployed_operating_threshold"
        ),
        "regime_summary": (
            "median_and_range_across_prespecified_splits_without_cross_split_confidence_interval"
        ),
    }
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, config_path


def test_detector_analysis_emits_hash_bound_exploratory_metrics(tmp_path):
    manifest_path, config_path = _write_fixture(tmp_path)
    output = tmp_path / "analysis"
    artifacts = analyze_detector_outputs(
        detector_run_manifest=manifest_path,
        analysis_config=config_path,
        output_dir=output,
    )
    assert artifacts.summary["fit_count"] == 4
    assert artifacts.summary["metric_rows"] == 40
    assert artifacts.summary["supplementary_metric_rows"] == 16
    metrics = pd.read_csv(output / "detector_extended_metrics.csv")
    assert set(metrics["metric"]) == {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "recall",
        "specificity",
        "precision",
        "brier_score",
        "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.05",
    }
    supplement = metrics.loc[
        metrics["evidence_status"].eq(
            "supplementary_exploratory_post_freeze"
        )
    ]
    assert len(supplement) == 16
    assert supplement["bootstrap_resamples_valid"].eq(50).all()
    assert metrics.loc[metrics["metric"].eq("precision"), "estimate"].eq(1.0).all()
    assert np.isclose(
        metrics.loc[metrics["metric"].eq("brier_score"), "estimate"],
        0.025,
        rtol=0.0,
        atol=1e-15,
    ).all()
    manifest = json.loads(
        (output / "detector_analysis_manifest.json").read_text()
    )
    assert manifest["partial_checkpoint_outcomes_seen_before_extension"] is True
    assert manifest["frozen_training_design_unchanged"] is True
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["size_bytes"]


def test_detector_analysis_rejects_tampered_declared_predictions(tmp_path):
    manifest_path, config_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    prediction_path = Path(manifest["output_dir"]) / "detector_predictions.csv"
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DetectorAnalysisError, match="hash mismatch"):
        analyze_detector_outputs(
            detector_run_manifest=manifest_path,
            analysis_config=config_path,
            output_dir=tmp_path / "analysis",
        )


def test_detector_analysis_rejects_incomplete_run(tmp_path):
    manifest_path, config_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmatory_complete"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DetectorAnalysisError, match="complete frozen CUDA"):
        analyze_detector_outputs(
            detector_run_manifest=manifest_path,
            analysis_config=config_path,
            output_dir=tmp_path / "analysis",
        )
