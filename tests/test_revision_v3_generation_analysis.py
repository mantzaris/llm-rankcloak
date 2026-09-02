import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_revision_v3_generation import (  # noqa: E402
    entropy_control_differences,
    payload_rank_trace_alignment_passes,
    position_summary,
    record_validation_passes,
)
from finalize_revision_v3_model_backed import (  # noqa: E402
    MODEL_BACKED_DESIGN_END,
    MODEL_BACKED_DESIGN_START,
    render_model_backed_experiment_design,
)


def test_calibration_negative_assertion_is_a_passing_validation():
    record = {
        "record_type": "entropy_calibration_trace",
        "validation": {
            "target_token_count_exact": True,
            "finite_entropy_at_every_position": True,
            "detector_outcomes_used": False,
        },
    }
    assert record_validation_passes(record)
    record["validation"]["detector_outcomes_used"] = True
    assert not record_validation_passes(record)
    record["validation"]["detector_outcomes_used"] = False
    record["validation"]["unexpected_assertion"] = False
    assert not record_validation_passes(record)


def test_noncalibration_validation_requires_all_positive_assertions():
    record = {
        "record_type": "entropy_rankcloak_trial",
        "validation": {
            "saved_payload_exact": True,
            "encoder_decoder_gate_positions_exact": True,
        },
    }
    assert record_validation_passes(record)
    record["validation"]["saved_payload_exact"] = False
    assert not record_validation_passes(record)


def test_payload_rank_alignment_ignores_only_sampled_skip_markers():
    generation = {
        "embedding_eligible_mask": [True, False, True, False, True],
        "realized_ranks": [7, None, 3, None, 11],
        "consumed_payload_rank_count": 3,
        "consumed_payload_rank_indices": [0, 1, 2],
    }
    assert payload_rank_trace_alignment_passes(generation, [7, 3, 11])
    generation["realized_ranks"][3] = 5
    assert not payload_rank_trace_alignment_passes(generation, [7, 3, 11])


def test_entropy_control_differences_are_strictly_paired():
    metric_names = (
        "generated_token_count",
        "mean_entropy_bits",
        "mean_observed_rank",
        "mean_token_surprisal_nats",
        "mean_rank_pressure_log_probability_gap_nats",
        "word_count",
        "sentence_count",
        "character_count",
        "flesch_reading_ease_heuristic",
        "flesch_kincaid_grade_heuristic",
        "coleman_liau_index",
        "unique_word_fraction",
        "repeated_bigram_fraction",
        "repeated_trigram_fraction",
        "maximum_identical_word_run",
        "surface_flag_total",
        "artifact_like_fragment_count",
        "tfidf_prompt_similarity",
    )
    identity = {
        "pairing_unit_id": "pair-1",
        "experimental_cell_id": "cell-1",
        "model_id": "model",
        "payload_name": "payload",
        "payload_class": "class",
        "representation_name": "representation",
        "prompt_template_id": "template",
        "gate_level": "moderate",
    }
    encoded = {**identity, "population": "rankcloak"}
    control = {**identity, "population": "ordinary_control"}
    for index, metric in enumerate(metric_names, start=1):
        encoded[metric] = float(index + 2)
        control[metric] = float(index)
    result = entropy_control_differences(pd.DataFrame([control, encoded]))
    assert len(result) == 1
    assert result["word_count_rankcloak_minus_control"].iloc[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="not one matched pair"):
        entropy_control_differences(pd.DataFrame([encoded]))


def test_position_summary_reports_forced_token_rank_distribution():
    frame = pd.DataFrame(
        {
            "population": ["rankcloak"] * 4,
            "gate_level": ["moderate"] * 4,
            "token_role": ["payload"] * 4,
            "entropy_bits": [1.0, 2.0, 3.0, 4.0],
            "observed_rank": [1, 3, 7, 9],
            "token_surprisal_nats": [0.1, 0.3, 0.7, 0.9],
            "rank_pressure_log_probability_gap_nats": [0.0, 0.2, 0.6, 0.8],
        }
    )
    summary = position_summary(
        frame, ["population", "gate_level", "token_role"]
    )
    ranks = summary.loc[summary["metric"].eq("observed_rank")].iloc[0]
    assert ranks["position_count"] == 4
    assert ranks["mean"] == pytest.approx(5.0)
    assert ranks["median"] == pytest.approx(5.0)
    assert ranks["p95"] == pytest.approx(8.7)


def test_model_backed_experiment_design_is_idempotent():
    original = """# Design

Stable detector design.

The bounded entropy-gate matrix is a dry-run plan.

The recovery-mode comparison reuses 144 historical trials.
"""
    first = render_model_backed_experiment_design(original)
    second = render_model_backed_experiment_design(first)
    assert second == first
    assert first.count(MODEL_BACKED_DESIGN_START) == 1
    assert first.count(MODEL_BACKED_DESIGN_END) == 1
    assert first.count("The entropy-gate matrix used") == 1
    assert first.count("The recovery-mode comparison") == 1
