from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_ablation_evidence import (
    AblationEvidenceError,
    build_ablation_evidence,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    for payload_index in range(4):
        payload = f"p{payload_index}"
        prompt = f"prompt{payload_index}"
        for model in ("a", "b"):
            base = 10.0 + payload_index + (model == "b")
            rows.append(
                {
                    "trial_id": f"canonical-{payload}-{model}",
                    "model_id": model,
                    "payload_name": payload,
                    "prompt_id": prompt,
                    "ablation_factor": "canonical",
                    "ablation_level": "canonical",
                    "condition": "base",
                    "exact_payload_recovery": 1,
                    "metric": base,
                    "evidence_status": "fixture_status",
                    "study_phase": "fixture_phase",
                }
            )
            rows.append(
                {
                    "trial_id": f"factor-x-{payload}-{model}",
                    "model_id": model,
                    "payload_name": payload,
                    "prompt_id": prompt,
                    "ablation_factor": "factor",
                    "ablation_level": "x",
                    "condition": "x",
                    "exact_payload_recovery": 1,
                    "metric": base + 2.0,
                    "evidence_status": "fixture_status",
                    "study_phase": "fixture_phase",
                }
            )
        rows.append(
            {
                "trial_id": f"factor-y-{payload}-a",
                "model_id": "a",
                "payload_name": payload,
                "prompt_id": prompt,
                "ablation_factor": "factor",
                "ablation_level": "y",
                "condition": "y",
                "exact_payload_recovery": 1,
                "metric": 13.0 + payload_index,
                "evidence_status": "fixture_status",
                "study_phase": "fixture_phase",
            }
        )
    trials = tmp_path / "trials.csv"
    pd.DataFrame(rows).to_csv(trials, index=False)
    unavailable = tmp_path / "unavailable.csv"
    pd.DataFrame(
        [
            {
                "model_id": "b",
                "payload_name": f"p{index}",
                "ablation_factor": "factor",
                "ablation_level": "y",
                "reason_code": "fixture_unavailable",
            }
            for index in range(4)
        ]
    ).to_csv(unavailable, index=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "rankcloak-revision-ablation-evidence-analysis-v1",
                "analysis_status": "exploratory_post_outcome_evidence_extraction",
                "confirmatory_generation_design_unchanged": True,
                "outcomes_were_available_before_analysis_specification": True,
                "expected_input": {
                    "trial_rows": len(rows),
                    "unavailable_rows": 4,
                    "planned_work_units": len(rows) + 4,
                    "payload_groups": 4,
                    "evidence_status": "fixture_status",
                    "study_phase": "fixture_phase",
                },
                "canonical_selector": {
                    "ablation_factor": "canonical",
                    "ablation_level": "canonical",
                },
                "factors": [
                    {
                        "factor": "factor",
                        "condition_column": "condition",
                        "canonical_value": "base",
                        "levels": ["x", "y"],
                    }
                ],
                "continuous_outcomes": ["metric"],
                "inference": {
                    "analysis_unit": "payload_name",
                    "bootstrap_resamples": 100,
                    "bootstrap_seed": 7,
                    "confidence_level": 0.95,
                    "primary_inference": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return trials, unavailable, config


def test_ablation_evidence_uses_payload_pairs_and_shared_models(tmp_path: Path):
    trials, unavailable, config = _fixture(tmp_path)
    output = tmp_path / "out"
    artifacts = build_ablation_evidence(
        trials_path=trials,
        unavailable_path=unavailable,
        config_path=config,
        output_dir=output,
        command="fixture",
    )
    assert artifacts.configuration_rows == 3
    assert artifacts.contrast_rows == 2
    contrasts = pd.read_csv(output / "ablation_canonical_contrasts.csv")
    x = contrasts.loc[contrasts["level"].eq("x")].iloc[0]
    y = contrasts.loc[contrasts["level"].eq("y")].iloc[0]
    assert x["level_minus_canonical"] == pytest.approx(2.0)
    assert x["shared_model_count"] == 2
    assert y["shared_model_count"] == 1
    assert y["paired_payload_groups"] == 4
    summary = pd.read_csv(output / "ablation_configuration_summary.csv")
    y_summary = summary.loc[summary["level"].eq("y")].iloc[0]
    assert y_summary["unavailable_work_units"] == 4
    manifest = json.loads((output / "ablation_evidence_manifest.json").read_text())
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]


def test_ablation_evidence_rejects_unexpected_level(tmp_path: Path):
    trials, unavailable, config = _fixture(tmp_path)
    value = json.loads(config.read_text())
    value["factors"][0]["levels"] = ["x"]
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AblationEvidenceError, match="levels differ"):
        build_ablation_evidence(
            trials_path=trials,
            unavailable_path=unavailable,
            config_path=config,
            output_dir=tmp_path / "out",
        )
