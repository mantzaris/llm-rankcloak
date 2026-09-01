import json
from pathlib import Path

import pandas as pd

from scripts.build_revision_v3_generation_plans import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_amended_generation_plans_bind_paired_seeds_and_historical_q4(tmp_path):
    assert main(["--output-dir", str(tmp_path)]) == 0
    provenance = tmp_path / "provenance"
    entropy = pd.read_csv(provenance / "entropy_generation_plan.csv")
    calibration = pd.read_csv(provenance / "entropy_calibration_plan.csv")
    quantization = pd.read_csv(provenance / "quantization_generation_plan.csv")

    assert len(entropy) == 720
    assert entropy["experimental_cell_id"].nunique() == 120
    assert (
        entropy.groupby(["experimental_cell_id", "population"])["random_seed"]
        .nunique()
        .eq(1)
        .all()
    )
    assert entropy.groupby("population")["random_seed"].nunique().to_dict() == {
        "ordinary_control": 120,
        "rankcloak": 120,
    }
    assert entropy["seed_shared_across_gate_levels"].astype(bool).all()
    assert set(entropy["ordinary_sampler"]) == {
        "numpy_pcg64_serial_top_p_v1_token_id_tiebreak"
    }

    assert len(calibration) == 18
    assert set(calibration["generation_method"]) == {
        "ordinary_top_p_clean_development_trace"
    }
    assert set(calibration["target_token_count"]) == {128}
    assert set(calibration["temperature"]) == {0.8}
    assert set(calibration["top_p"]) == {0.95}
    assert not calibration["detector_outcomes_used"].astype(bool).any()

    assert len(quantization) == 3840
    grouped = quantization.groupby("pairing_unit_id")
    assert grouped["quantization"].nunique().eq(2).all()
    assert grouped["historical_control_sampling_seed"].nunique().eq(1).all()
    assert grouped["non_quantization_contract_sha256"].nunique().eq(1).all()
    assert quantization["random_seed"].eq(
        quantization["historical_control_sampling_seed"]
    ).all()

    records_path = (
        PROJECT_ROOT
        / "results/revision_v1/primary_v2/qwen2_5_7b_instruct_q4_k_m/records.jsonl"
    )
    historical_seeds = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("record_type") == "ordinary_control"
                and record.get("control_view") == "full_message"
            ):
                historical_seeds[str(record["source_trial_id"])] = int(
                    record["generation"]["sampling_seed"]
                )
    for row in quantization.itertuples():
        assert int(row.historical_control_sampling_seed) == historical_seeds[
            str(row.reference_q4_trial_id)
        ]

    summary = json.loads(
        (provenance / "generation_plan_summary.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (provenance / "generation_preflight.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"].endswith("v2")
    assert preflight["schema_version"].endswith("v2")
    assert preflight["launch_performed"] is False
    assert preflight["downloads_performed"] is False
