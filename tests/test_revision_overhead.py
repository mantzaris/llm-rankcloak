from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_overhead import (
    OverheadAnalysisError,
    build_overhead_artifacts,
    derive_overhead_metrics,
)
from rankcloak.revision_statistics import file_sha256, synthetic_smoke_frames


def test_overhead_totals_respect_inclusive_runner_timings():
    frame = pd.DataFrame(
        {
            "representation_seconds": [1.0, 1.0],
            "filter_setup_seconds": [2.0, np.nan],
            "encoding_seconds": [21.0, 21.0],
            "recovery_seconds": [4.0, 4.0],
            "decoding_seconds": [9.0, 9.0],
            "generation_seconds": [18.0, 18.0],
        }
    )
    derived = derive_overhead_metrics(frame)
    assert derived.iloc[0]["encoding_overhead_seconds"] == pytest.approx(3.0)
    assert derived.iloc[0]["decoding_overhead_seconds"] == pytest.approx(9.0)
    assert derived.iloc[0]["inverse_transcode_seconds"] == pytest.approx(5.0)
    assert derived.iloc[0]["non_generation_overhead_seconds"] == pytest.approx(
        12.0
    )
    assert derived.iloc[0]["overhead_to_generation_ratio"] == pytest.approx(
        12 / 18
    )
    assert derived.iloc[0]["encoding_component_residual_seconds"] == pytest.approx(
        0.0
    )
    assert np.isnan(derived.iloc[1]["encoding_component_residual_seconds"])
    assert derived.iloc[1]["encoding_overhead_seconds"] == pytest.approx(3.0)
    assert derived.iloc[1]["non_generation_overhead_seconds"] == pytest.approx(12.0)
    with pytest.raises(OverheadAnalysisError, match="negative or nonfinite"):
        derive_overhead_metrics(pd.DataFrame({"generation_seconds": [-1.0]}))
    with pytest.raises(OverheadAnalysisError, match="inclusive timing semantics"):
        derive_overhead_metrics(
            pd.DataFrame({"generation_seconds": [2.0], "encoding_seconds": [1.0]})
        )


def _saved_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    sources = synthetic_smoke_frames(seed=7)
    trials = sources["trials"].iloc[:4].copy()
    runtime = sources["runtime"].iloc[:4].copy()
    runtime["runtime_scope"] = "trial"
    runtime["record_type"] = "trial_runtime"
    runtime["representation_seconds"] = 0.1
    runtime["filter_setup_seconds"] = 0.2
    runtime["generation_seconds"] = 1.0
    runtime["encoding_seconds"] = 1.3
    runtime["recovery_seconds"] = 0.4
    runtime["decoding_seconds"] = 0.5
    runtime["execution_seconds"] = 2.5
    runtime["generation_tokens_per_second"] = 20.0
    runtime["encoding_tokens_per_second"] = 100.0
    runtime["decoding_tokens_per_second"] = 120.0
    runtime["representation_bits_per_second"] = 500.0
    runtime["serialized_bits_per_second"] = 600.0
    runtime["cover_tokens_per_payload_byte"] = 2.0
    load = {
        **runtime.iloc[0].to_dict(),
        "trial_id": "model_load::0",
        "payload_name": "not_applicable_model_load",
        "protocol_variant": "not_applicable_model_load",
        "runtime_scope": "model_load_session",
        "record_type": "model_load_event",
        "model_load_seconds": 2.0,
    }
    memory = {
        **runtime.iloc[0].to_dict(),
        "trial_id": "memory_profile::0",
        "payload_name": "not_applicable_memory_profile",
        "protocol_variant": "not_applicable_memory_profile",
        "runtime_scope": "model_shard_memory_profile",
        "record_type": "memory_profile_event",
        "peak_ram_mib": 1024.0,
        "peak_gpu_memory_mib": 2048.0,
        "peak_ram_availability": "recorded_test_fixture",
        "peak_gpu_memory_availability": "recorded_test_fixture",
    }
    for row in (load, memory):
        for column in (
            "representation_seconds",
            "filter_setup_seconds",
            "generation_seconds",
            "encoding_seconds",
            "recovery_seconds",
            "decoding_seconds",
            "execution_seconds",
            "generation_tokens_per_second",
            "encoding_tokens_per_second",
            "decoding_tokens_per_second",
            "representation_bits_per_second",
            "payload_bits_per_second",
            "serialized_bits_per_second",
            "cover_tokens_per_payload_byte",
        ):
            row[column] = np.nan
    runtime = pd.concat(
        [runtime, pd.DataFrame([load, memory])], ignore_index=True
    )
    trials_path = tmp_path / "trials.csv"
    runtime_path = tmp_path / "runtime.csv"
    config_path = tmp_path / "statistics.json"
    trials.to_csv(trials_path, index=False)
    runtime.to_csv(runtime_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "intervals": {
                    "confidence_level": 0.95,
                    "bootstrap_resamples": 10,
                    "bootstrap_seed": 9,
                }
            }
        ),
        encoding="utf-8",
    )
    return trials_path, runtime_path, config_path


def test_build_overhead_artifacts_records_sessions_limits_and_hashes(tmp_path):
    trials_path, runtime_path, config_path = _saved_fixture(tmp_path)
    output_dir = tmp_path / "output"
    artifacts = build_overhead_artifacts(
        trial_paths={"primary": trials_path},
        runtime_paths={"primary": runtime_path},
        statistics_config=config_path,
        output_dir=output_dir,
    )
    assert artifacts.summary["trial_rows"] == 4
    assert artifacts.summary["runtime_rows"] == 6
    assert artifacts.summary["matched_trial_runtime_rows"] == 4
    assert artifacts.summary["unmatched_nontrial_runtime_rows"] == 2
    assert artifacts.summary["model_load_rows"] == 1
    assert artifacts.summary["memory_profile_rows"] == 1
    summary = pd.read_csv(artifacts.files["summary"])
    assert "encoding_overhead_seconds" in set(summary["outcome"])
    initialization = pd.read_csv(artifacts.files["initialization"])
    memory = pd.read_csv(artifacts.files["memory"])
    assert initialization.iloc[0]["model_load_seconds"] == pytest.approx(2.0)
    assert memory.iloc[0]["peak_gpu_memory_mib"] == pytest.approx(2048.0)
    limitations = json.loads(Path(artifacts.files["limitations"]).read_text())
    assert limitations["cpu_time"]["status"] == "unavailable_not_recorded"
    assert limitations["new_generation_runs"] == 0
    manifest = json.loads(Path(artifacts.files["manifest"]).read_text())
    for output in manifest["outputs"].values():
        assert file_sha256(output["path"]) == output["sha256"]

    with pytest.raises(OverheadAnalysisError, match="Refusing to overwrite"):
        build_overhead_artifacts(
            trial_paths={"primary": trials_path},
            runtime_paths={"primary": runtime_path},
            statistics_config=config_path,
            output_dir=output_dir,
        )


def test_overhead_stage_labels_must_match(tmp_path):
    trials_path, runtime_path, config_path = _saved_fixture(tmp_path)
    with pytest.raises(OverheadAnalysisError, match="labels must match"):
        build_overhead_artifacts(
            trial_paths={"primary": trials_path},
            runtime_paths={"other": runtime_path},
            statistics_config=config_path,
            output_dir=tmp_path / "unused",
        )
