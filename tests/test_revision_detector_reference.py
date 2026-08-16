from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rankcloak.revision_detector_reference import (
    DetectorReferenceError,
    build_detector_reference_index,
    canonical_json_sha256,
    file_sha256,
)


def _signed(value):
    result = dict(value)
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def _row_payload(rows):
    columns = list(rows[0])
    return {
        "schema_version": "rankcloak-revision-detector-fit-rows-v1",
        "columns": columns,
        "rows": [[row[column] for column in columns] for row in rows],
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, fit_count: int = 4) -> Path:
    output = tmp_path / "detector"
    checkpoint = tmp_path / "checkpoints"
    (checkpoint / "fits").mkdir(parents=True)
    output.mkdir()
    run_identity = "a" * 64
    plan_sha = "b" * 64
    metric_rows = []
    prediction_rows = []
    durations = []
    for ordinal in range(fit_count):
        detector_name = "textcnn" if ordinal % 2 == 0 else "deberta"
        kind = "text_cnn" if ordinal % 2 == 0 else "pretrained_transformer"
        task = {
            "ordinal": ordinal,
            "detector_name": detector_name,
            "detector_kind": kind,
            "split_id": f"split-{ordinal // 2}",
            "regime": "matched" if ordinal < 2 else "held_out_template",
            "held_out_column": None if ordinal < 2 else "prompt_template_id",
            "held_out_value": None if ordinal < 2 else "template-a",
            "seed": 10 + ordinal,
            "train_row_count": 6,
            "test_row_count": 2,
        }
        metric = {
            "split_id": task["split_id"],
            "regime": task["regime"],
            "detector_name": detector_name,
            "requested_kind": kind,
            "implementation_kind": kind,
            "implementation_status": "complete",
            "seed": task["seed"],
            "model_state_sha256": str(ordinal) * 64,
            "model_artifact_set_sha256": "c" * 64 if kind == "pretrained_transformer" else None,
            "roc_auc": 0.9,
        }
        predictions = [
            {
                "split_id": task["split_id"],
                "detector_name": detector_name,
                "row_id": f"row-{ordinal}-{index}",
                "label": index,
                "score": float(index),
            }
            for index in range(2)
        ]
        directory = checkpoint / "fits" / f"{ordinal:04d}"
        directory.mkdir()
        metric_path = directory / "metric.json"
        predictions_path = directory / "predictions.json"
        _write_json(metric_path, _row_payload([metric]))
        _write_json(predictions_path, _row_payload(predictions))
        children = {
            "metric.json": {
                "sha256": file_sha256(metric_path),
                "size_bytes": metric_path.stat().st_size,
                "row_count": 1,
            },
            "predictions.json": {
                "sha256": file_sha256(predictions_path),
                "size_bytes": predictions_path.stat().st_size,
                "row_count": 2,
            },
        }
        elapsed = 1.0 + ordinal
        durations.append(elapsed)
        fit_manifest = _signed(
            {
                "schema_version": "rankcloak-revision-detector-fit-checkpoint-v1",
                "run_identity_sha256": run_identity,
                "plan_sha256": plan_sha,
                "task_identity": task,
                "task_identity_sha256": canonical_json_sha256(task),
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "completed_at_utc": "2026-01-01T00:00:01+00:00",
                "elapsed_seconds": elapsed,
                "children": children,
                "children_sha256": canonical_json_sha256(children),
            }
        )
        _write_json(directory / "manifest.json", fit_manifest)
        metric_rows.append(metric)
        prediction_rows.extend(predictions)

    def write_csv(name, rows):
        path = output / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    files = {
        "detector_metrics.csv": write_csv("detector_metrics.csv", metric_rows),
        "detector_predictions.csv": write_csv("detector_predictions.csv", prediction_rows),
        "detector_dataset_manifest.csv": write_csv(
            "detector_dataset_manifest.csv", [{"row_id": "row", "label": 1}]
        ),
    }
    for name, value in (
        ("detector_split_manifest.json", {"splits": []}),
        ("detector_failures.json", []),
    ):
        path = output / name
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        files[name] = path
    declarations = {
        name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for name, path in files.items()
    }
    run = _signed(
        {
            "schema_version": "rankcloak-revision-detector-run-v2",
            "execution_mode": "confirmatory",
            "confirmatory_complete": True,
            "device": "cuda:0",
            "completed_fit_count": fit_count,
            "total_fit_count": fit_count,
            "metric_rows": fit_count,
            "failure_count": 0,
            "smoke_fallback_metric_rows": 0,
            "gpu_accounting": {"cumulative_elapsed_seconds": 10.0},
            "output_dir": str(output),
            "checkpoint_dir": str(checkpoint),
            "run_identity_sha256": run_identity,
            "execution_plan_sha256": plan_sha,
            "checkpoint_cumulative_fit_seconds": sum(durations),
            "output_files": declarations,
        }
    )
    manifest_path = output / "detector_run_manifest.json"
    _write_json(manifest_path, run)
    return manifest_path


def test_detector_reference_indexes_all_fits_without_copying(tmp_path):
    run = _fixture(tmp_path)
    output = tmp_path / "package" / "detectors"
    artifacts = build_detector_reference_index(
        detector_run_manifest=run,
        output_dir=output,
        command="fixture",
    )
    assert artifacts.fit_count == 4
    assert artifacts.external_reference_count == 18
    with (output / "detector_fit_reference_index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["detector_kind"] for row in rows} == {"text_cnn", "pretrained_transformer"}
    references = json.loads((output / "detector_output_references.json").read_text())
    assert references["large_detector_artifacts_copied"] is False
    assert len(references["checkpoint_references"]) == 4
    manifest = json.loads((output / "detector_reference_manifest.json").read_text())
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]


def test_detector_reference_rejects_tampered_checkpoint_child(tmp_path):
    run = _fixture(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "fits" / "0001" / "predictions.json"
    checkpoint.write_text(checkpoint.read_text() + "\n", encoding="utf-8")
    with pytest.raises(DetectorReferenceError, match="identity differs"):
        build_detector_reference_index(
            detector_run_manifest=run, output_dir=tmp_path / "package"
        )


def test_detector_reference_rejects_incomplete_run(tmp_path):
    run_path = _fixture(tmp_path)
    run = json.loads(run_path.read_text())
    run.pop("manifest_sha256")
    run["confirmatory_complete"] = False
    _write_json(run_path, _signed(run))
    with pytest.raises(DetectorReferenceError, match="complete CUDA"):
        build_detector_reference_index(
            detector_run_manifest=run_path, output_dir=tmp_path / "package"
        )


def test_detector_reference_refuses_unrequested_overwrite(tmp_path):
    run = _fixture(tmp_path)
    output = tmp_path / "package"
    build_detector_reference_index(detector_run_manifest=run, output_dir=output)
    with pytest.raises(DetectorReferenceError, match="Refusing to overwrite"):
        build_detector_reference_index(detector_run_manifest=run, output_dir=output)
