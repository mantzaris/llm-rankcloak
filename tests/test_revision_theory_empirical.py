from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_theory_empirical import (
    OUTPUT_FILENAMES,
    EmpiricalTheoryError,
    build_empirical_theory_artifacts,
    empirical_validation_rows,
    file_sha256,
)


def _record(index: int) -> dict[str, object]:
    return {
        "record_type": "rankcloak_trial",
        "study_phase": "primary_v2_confirmatory",
        "trial_id": f"trial_{index}",
        "work_id": f"work_{index}",
        "model_id": "model_a",
        "protocol_variant": "nonseg_ascii_b8",
        "payload_name": f"payload_{index}",
        "payload_class": "sha256_hex",
        "language": "en",
        "segmented": False,
        "segment_count": 1,
        "tail_policy": "fixed_two",
        "token_filter": "none",
        "leadin_tokens": 0,
        "topic_schedule": None,
        "H_bits": 8,
        "alphabet_size_B": 8,
        "artifact_bit_length": 8,
        "representation": {
            "name": "ascii_b8",
            "expected_ranks": [1, 2, 3],
            "metadata": {
                "alphabet_size": 8,
                "original_byte_length": 1,
                "padding_bits": 1,
            },
        },
        "forced_token_count": 3,
        "tail_token_count": 2,
        "segments": [
            {
                "expected_ranks": [1, 2, 3],
                "forced_log_probabilities": [-0.1, -0.5, -0.8],
                "greedy_log_probabilities": [-0.1, -0.2, -0.2],
                "rank_B_log_probabilities": [-1.5, -1.5, -1.5],
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "primary_v2" / "model_a" / "records.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "".join(json.dumps(_record(index)) + "\n" for index in (1, 2)),
        encoding="utf-8",
    )
    statistics = tmp_path / "statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "intervals": {
                    "confidence_level": 0.95,
                    "bootstrap_resamples": 25,
                    "bootstrap_seed": 20260808,
                }
            }
        ),
        encoding="utf-8",
    )
    return source, statistics


def test_empirical_rows_preserve_conditions_and_derive_residuals():
    record = _record(1)
    record["_theory_source_file"] = "/saved/primary_v2/model/records.jsonl"
    record["_theory_source_row"] = 1
    row = empirical_validation_rows([record])[0]
    assert row["source_stage"] == "primary_v2"
    assert row["forced_position_residual_tokens"] == 0
    assert row["tail_overhead_tokens"] == 2
    assert row["observed_total_cover_tokens"] == 5
    assert row["cover_length_residual_tokens"] == 2
    assert row["rate_bound_holds"] is True
    assert row["quality_status"] == "validated"
    assert row["greedy_lower_bound_holds_per_context"] is True
    assert row["rank_B_upper_bound_holds_per_context"] is True


def test_builder_emits_hashed_residuals_uncertainty_and_assumptions(tmp_path: Path):
    source, statistics = _fixture(tmp_path)
    output = tmp_path / "output"
    artifacts = build_empirical_theory_artifacts(
        input_paths=[source],
        statistics_config=statistics,
        output_dir=output,
    )
    assert artifacts.summary["input_record_count"] == 2
    assert artifacts.summary["capacity_evaluable_count"] == 2
    assert artifacts.summary["forced_position_residual_nonzero_count"] == 0
    assert artifacts.summary["cover_length_residual_positive_count"] == 2
    assert artifacts.summary["quality_fully_bound_validated_count"] == 2

    summary = pd.read_csv(output / OUTPUT_FILENAMES["summary"])
    residual = summary.loc[
        summary["outcome"].eq("forced_position_residual_tokens")
    ].iloc[0]
    assert residual["n"] == 2
    assert residual["payload_units"] == 2
    assert residual["mean"] == 0
    assert residual["ci_low"] == 0
    assert residual["ci_high"] == 0

    assumptions = json.loads(
        (output / OUTPUT_FILENAMES["assumptions"]).read_text(encoding="utf-8")
    )
    assert any(
        equation["id"] == "minimum_forced_positions"
        for equation in assumptions["equations"]
    )
    technical_note = (output / OUTPUT_FILENAMES["technical_note"]).read_text(
        encoding="utf-8"
    )
    assert "Computational evidence artifact only" in technical_note
    assert "`n_B = ceil(H / log2(B))`" in technical_note
    assert "payload-clustered percentile-bootstrap intervals" in technical_note
    assert "token positions are not treated as independent" in technical_note
    manifest = json.loads(
        (output / OUTPUT_FILENAMES["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["raw_record_files_copied"] is False
    assert manifest["resampling_unit"] == "payload_name"
    assert "technical_note" in manifest["outputs"]
    for key, entry in manifest["outputs"].items():
        assert file_sha256(entry["path"]) == entry["sha256"], key


def test_duplicate_raw_input_path_fails_before_double_counting(tmp_path: Path):
    source, statistics = _fixture(tmp_path)
    with pytest.raises(EmpiricalTheoryError, match="unique"):
        build_empirical_theory_artifacts(
            input_paths=[source, source],
            statistics_config=statistics,
            output_dir=tmp_path / "output",
        )
