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
    record_validation_passes,
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
        "prompt_word_jaccard",
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
