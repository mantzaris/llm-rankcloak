from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_robustness_coverage import (
    RobustnessCoverageError,
    build_robustness_coverage_inventory,
    canonical_json_sha256,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    conditions = tmp_path / "conditions.csv"
    pd.DataFrame(
        [
            {
                "robustness_family": "raw_transmission",
                "transformation_id": "trim",
                "observed_outcome_rows": 10,
                "unavailable_rows": 0,
                "success_outcome_rows": 2,
                "failure_outcome_rows": 8,
                "recovery_rate": 0.2,
                "analysis_unit": "source_trial_id",
            },
            {
                "robustness_family": "raw_transmission",
                "transformation_id": "quotes",
                "observed_outcome_rows": 10,
                "unavailable_rows": 0,
                "success_outcome_rows": 4,
                "failure_outcome_rows": 6,
                "recovery_rate": 0.4,
                "analysis_unit": "source_trial_id",
            },
        ]
    ).to_csv(conditions, index=False)
    design = tmp_path / "robustness.json"
    design.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "transformations": [
                    {"transformation_id": "trim"},
                    {"transformation_id": "quotes"},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "rankcloak-revision-robustness-analysis-v1",
                "status": "passed",
                "failure_taxonomy_scope": (
                    "descriptive_first_divergence_not_causal_proof"
                ),
                "outputs": {
                    "conditions": {
                        "path": conditions.name,
                        "sha256": file_sha256(conditions),
                        "size_bytes": conditions.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "coverage.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "rankcloak-robustness-coverage-map-v1",
                "analysis_status": "supporting_post_audit_coverage_inventory",
                "frozen_robustness_results_unchanged": True,
                "expected_request_count": 3,
                "requests": [
                    {
                        "request_id": "whitespace",
                        "requested_transformation": "Whitespace",
                        "coverage_status": "directly_tested",
                        "transformation_ids": ["trim"],
                        "scope_note": "trim tested",
                        "limitation": "addition untested",
                    },
                    {
                        "request_id": "quotation",
                        "requested_transformation": "Quotation",
                        "coverage_status": "partially_represented",
                        "transformation_ids": ["quotes"],
                        "scope_note": "one direction",
                        "limitation": "reverse untested",
                    },
                    {
                        "request_id": "case",
                        "requested_transformation": "Case",
                        "coverage_status": "not_tested",
                        "transformation_ids": [],
                        "scope_note": "absent",
                        "limitation": "no estimate",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config, design, manifest


def test_coverage_inventory_preserves_tested_and_untested_status(tmp_path: Path) -> None:
    config, design, manifest = _fixture(tmp_path)
    artifacts = build_robustness_coverage_inventory(
        coverage_config=config,
        robustness_config=design,
        robustness_manifest=manifest,
        output_dir=tmp_path / "output",
        command="fixture",
    )
    assert artifacts.request_count == 3
    output_manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = output_manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(output_manifest)
    assert output_manifest["summary"] == {
        "directly_tested_count": 1,
        "not_tested_count": 1,
        "partially_represented_count": 1,
        "requested_transformation_count": 3,
    }
    frame = pd.read_csv(
        output_manifest["outputs"]["perturbation_coverage_inventory"]["path"]
    )
    case = frame.loc[frame["request_id"].eq("case")].iloc[0]
    assert case["observed_outcome_rows_across_conditions"] == 0
    assert case["analysis_unit"] == "not_estimated"


def test_coverage_inventory_rejects_false_not_tested_mapping(tmp_path: Path) -> None:
    config, design, manifest = _fixture(tmp_path)
    value = json.loads(config.read_text())
    value["requests"][2]["transformation_ids"] = ["trim"]
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RobustnessCoverageError, match="Tested-status mapping"):
        build_robustness_coverage_inventory(
            coverage_config=config,
            robustness_config=design,
            robustness_manifest=manifest,
            output_dir=tmp_path / "output",
        )
