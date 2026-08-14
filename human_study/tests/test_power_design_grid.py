import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "power" / "simulate_power.py"))


def test_power_design_grid_uses_scenario_specific_counts():
    config = json.loads(
        (ROOT / "config" / "power_design_grid.json").read_text(encoding="utf-8")
    )
    rows = MODULE["run"](config, simulations=10, seed=17)
    by_id = {row["scenario_id"]: row for row in rows}
    assert by_id["or1_5_typical_72x3"]["ratings_per_stimulus"] == 3
    assert by_id["or1_5_typical_72x3"]["participant_slots"] == 72
    assert by_id["or1_5_typical_120x5"]["ratings_per_stimulus"] == 5
    assert by_id["or1_5_typical_120x5"]["participant_slots"] == 200


def test_power_grid_rejects_unsupported_rating_replication():
    config = json.loads(
        (ROOT / "config" / "power_design_grid.json").read_text(encoding="utf-8")
    )
    bad = dict(config)
    bad["scenarios"] = [dict(config["scenarios"][0], ratings_per_stimulus=6)]
    with pytest.raises(ValueError, match="between 1 and 5"):
        MODULE["run"](bad, simulations=10, seed=19)
