import json

import numpy as np
import pytest

from rankcloak.revision_v3_diagnostics import trace_observed_tokens
from rankcloak.revision_v3_generation import (
    GenerationExecutionError,
    atomic_json,
    dry_run_summary,
    immutable_result,
    paired_distribution_comparison,
    sort_phase_rows,
)


class TinyModel:
    def __init__(self):
        self.n_tokens = 0
        self.scores = np.zeros((64, 4), dtype=float)

    def reset(self):
        self.n_tokens = 0

    def eval(self, token_ids):
        for _token_id in token_ids:
            self.scores[self.n_tokens] = [0.0, -1.0, -2.0, -3.0]
            self.n_tokens += 1


def test_observed_trace_records_entropy_rank_and_pressure_at_every_position():
    trace = trace_observed_tokens(TinyModel(), [0], [1, 2, 3])
    assert trace["position_count"] == 3
    assert trace["observed_ranks"] == [2, 3, 4]
    assert trace["greedy_token_ids"] == [0, 0, 0]
    assert trace["observed_surprisals_nats"] == pytest.approx(
        [-value for value in trace["observed_log_probabilities"]]
    )
    assert len(trace["entropy_bits"]) == 3
    assert np.isfinite(trace["entropy_bits"]).all()


def test_paired_distribution_comparison_uses_same_position_path():
    q4 = {
        "entropy_bits": [1.0, 2.0],
        "observed_ranks": [1, 2],
        "greedy_token_ids": [4, 5],
    }
    q8 = {
        "entropy_bits": [1.5, 1.5],
        "observed_ranks": [1, 3],
        "greedy_token_ids": [4, 6],
    }
    result = paired_distribution_comparison(q4, q8)
    assert result["position_count"] == 2
    assert result["mean_entropy_q8_minus_q4_bits"] == pytest.approx(0.0)
    assert result["observed_token_rank_changed_count"] == 1
    assert result["greedy_token_changed_fraction"] == pytest.approx(0.5)


def test_entropy_rows_are_ordered_by_gate_then_rankcloak_before_control():
    rows = [
        {
            "experimental_cell_id": "cell",
            "gate_level": gate,
            "population": population,
            "plan_id": gate + population,
        }
        for gate in ("strict", "ungated", "moderate")
        for population in ("ordinary_control", "rankcloak")
    ]
    ordered = sort_phase_rows("entropy", rows)
    assert [(row["gate_level"], row["population"]) for row in ordered] == [
        ("ungated", "rankcloak"),
        ("ungated", "ordinary_control"),
        ("moderate", "rankcloak"),
        ("moderate", "ordinary_control"),
        ("strict", "rankcloak"),
        ("strict", "ordinary_control"),
    ]


def test_dry_run_selects_executable_ledgers_without_loading_models(tmp_path):
    calibration = dry_run_summary(
        "entropy_calibration",
        "llama3_8b_instruct_q4_k_m",
        tmp_path,
        False,
        None,
    )
    quantization = dry_run_summary(
        "quantization",
        "qwen2_5_7b_instruct_q8_0",
        tmp_path,
        False,
        None,
    )
    assert calibration["model_loaded"] is False
    assert calibration["selected_row_count"] == 6
    assert quantization["selected_row_count"] == 1920


def test_immutable_result_refuses_completed_trial_overwrite(tmp_path):
    path = tmp_path / "trial.json"
    atomic_json(path, {"value": 1})
    assert json.loads(path.read_text())["value"] == 1
    with pytest.raises(GenerationExecutionError, match="overwrite"):
        immutable_result(path, {"value": 2})
