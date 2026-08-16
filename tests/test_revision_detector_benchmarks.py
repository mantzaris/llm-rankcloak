from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_detector_benchmarks import (
    DetectorBenchmarkError,
    build_detector_benchmark_evidence,
    canonical_json_sha256,
    file_sha256,
)


def _signed(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    benchmarks = []
    reports = []
    for index, name, kind in (
        (0, "published_textcnn_equivalent", "text_cnn"),
        (1, "deberta_v3_base_classifier", "pretrained_transformer"),
    ):
        phases = {
            "initialization_and_preprocessing": 1.0,
            "training": 2.0,
            "evaluation": 0.5,
            "trained_state_hashing": 0.25,
            "total": 3.75,
        }
        benchmark = {
            "schema_version": "rankcloak-revision-detector-benchmark-v1",
            "benchmark_task_index": index,
            "benchmark_task_identity": {
                "ordinal": index,
                "detector_name": name,
                "detector_kind": kind,
                "train_row_count": 12,
                "test_row_count": 4,
                "effective_detector_config": {"batch_size": 2, "epochs": 1},
            },
            "device": "cuda:0",
            "workers": 1,
            "new_fit_count": 1,
            "phase_timings_seconds": phases,
            "fit_non_model_analysis_seconds": 1.25,
            "fit_elapsed_seconds": 5.0,
            "checkpoint_seconds": 0.1,
            "invocation_overhead_seconds": 0.2,
            "wall_seconds": 5.3,
            "peak_rss_bytes": 100,
            "peak_vram_bytes": 200,
        }
        benchmark_path = _signed(
            tmp_path / f"benchmark-{index}.json", benchmark, "benchmark_sha256"
        )
        artifact_dir = tmp_path / f"task_{index}"
        artifact_dir.mkdir()
        input_identities = {}
        for role in ("cuda", "cuda_repeat"):
            artifact = artifact_dir / f"{role}.json"
            artifact.write_text(f"{role}-{index}", encoding="utf-8")
            input_identities[role] = {
                "path": str(artifact.resolve()),
                "sha256": file_sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        checks = {
            "metrics_exact": True,
            "model_state_sha256_exact": True,
            "predictions_exact": True,
            "row_identity_order_labels_exact": True,
            "scores_exact": True,
            "task_design_exact": True,
        }
        report = {
            "schema_version": "rankcloak-revision-detector-cuda-reproducibility-v1",
            "input_artifacts": input_identities,
            "decision": {
                "reproducible": True,
                "scientific_task_identity_exact": True,
                "same_device_cuda": {
                    "passed": True,
                    "checks": checks,
                    "measurements": {
                        "cuda_phase_timings_seconds": phases,
                        "cuda_repeat_phase_timings_seconds": phases,
                        "maximum_score_absolute_difference": 0.0,
                        "prediction_row_count": 4,
                        "timings_excluded_from_scientific_equality": True,
                    },
                },
            },
        }
        report_path = _signed(
            artifact_dir / "cuda_reproducibility_report.json",
            report,
            "report_sha256",
        )
        benchmarks.append(benchmark_path)
        reports.append(report_path)
    return benchmarks, reports


def test_detector_benchmark_evidence_preserves_phase_scope(tmp_path: Path) -> None:
    benchmarks, reports = _fixture(tmp_path)
    artifacts = build_detector_benchmark_evidence(
        benchmark_paths=benchmarks,
        reproducibility_report_paths=reports,
        output_dir=tmp_path / "output",
        command="fixture",
    )
    assert artifacts.architecture_count == 2
    manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    frame = pd.read_csv(
        manifest["outputs"]["detector_cuda_benchmark_summary"]["path"]
    )
    assert frame["fit_elapsed_seconds"].eq(5.0).all()
    assert frame["wall_seconds"].eq(5.3).all()
    assert frame["all_predeclared_exact_checks_passed"].eq(True).all()
    assert manifest["cpu_gpu_equivalence_tested"] is False


def test_detector_benchmark_evidence_rejects_timing_mismatch(tmp_path: Path) -> None:
    benchmarks, reports = _fixture(tmp_path)
    value = json.loads(benchmarks[0].read_text())
    value.pop("benchmark_sha256")
    value["wall_seconds"] = 9.0
    _signed(benchmarks[0], value, "benchmark_sha256")
    with pytest.raises(DetectorBenchmarkError, match="timing arithmetic"):
        build_detector_benchmark_evidence(
            benchmark_paths=benchmarks,
            reproducibility_report_paths=reports,
            output_dir=tmp_path / "output",
        )
