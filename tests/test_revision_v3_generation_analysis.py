import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_revision_v3_generation import record_validation_passes  # noqa: E402


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
