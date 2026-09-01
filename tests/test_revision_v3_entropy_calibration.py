import pytest

from rankcloak.revision_v3_entropy import (
    EntropyGateError,
    calibrate_entropy_gate_thresholds,
)


def test_entropy_thresholds_are_clean_development_quantiles_only():
    result = calibrate_entropy_gate_thresholds([0.0, 1.0, 2.0, 3.0, 4.0])
    assert result["moderate_threshold_bits"] == 2.0
    assert result["strict_threshold_bits"] == 3.0
    assert result["detector_outcomes_used"] is False
    assert result["quantile_method"] == "numpy_linear"


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [-0.1, 1.0]])
def test_entropy_threshold_calibration_fails_closed_on_invalid_traces(values):
    with pytest.raises(EntropyGateError):
        calibrate_entropy_gate_thresholds(values)
