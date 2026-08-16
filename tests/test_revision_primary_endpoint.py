from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_primary_endpoint import (
    PrimaryEndpointError,
    build_primary_endpoint,
    canonical_json_sha256,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    for payload in range(4):
        for model in range(2):
            success = int(not (payload == 3 and model == 1))
            rows.append(
                {
                    "trial_id": f"trial-{payload}-{model}",
                    "record_type": "rankcloak_trial",
                    "evidence_status": "primary-evidence",
                    "study_phase": "primary-phase",
                    "replay_mode": "saved_token_ids",
                    "protocol_contract_revision": "payload_fidelity_v2",
                    "result_schema_revision": "payload_aware_result_v2",
                    "exact_rank_replay": success,
                    "exact_payload_recovery": success,
                    "exact_recovery": success,
                    "recovery_outcome_semantics": (
                        "original_serialized_payload_bytes_sha256_v1"
                    ),
                    "payload_name": f"payload-{payload}",
                    "payload_class": f"class-{payload % 2}",
                    "model_id": f"model-{model}",
                    "prompt_id": f"prompt-{model}",
                    "prompt_category": f"category-{model}",
                    "protocol_variant": f"protocol-{model}",
                }
            )
    trials = tmp_path / "trials.csv"
    pd.DataFrame(rows).to_csv(trials, index=False)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "plan_id": (
                    "rankcloak_revision_primary_v2_prespecified_confirmatory_models"
                ),
                "experimental_unit": "payload_trial",
                "segments_as_independent_observations_forbidden": True,
                "filters": {
                    "primary_trials": {
                        "evidence_status": "primary-evidence",
                        "study_phase": "primary-phase",
                        "record_type": "rankcloak_trial",
                        "replay_mode": "saved_token_ids",
                        "protocol_contract_revision": "payload_fidelity_v2",
                        "result_schema_revision": "payload_aware_result_v2",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": (
                    "rankcloak-primary-recovery-endpoint-config-v1"
                ),
                "analysis_status": (
                    "post_audit_payload_group_sensitivity_not_prespecified_replacement"
                ),
                "partial_outcomes_seen_before_addition": True,
                "frozen_generation_design_unchanged": True,
                "strict_payload_success_rule": (
                    "all_observed_primary_trials_for_payload_must_recover"
                ),
                "confidence_level": 0.95,
                "expected_design": {
                    "primary_trial_rows": 8,
                    "payloads": 4,
                    "model_payload_groups": 8,
                    "model_families": 2,
                    "prompt_templates": 2,
                    "prompt_categories": 2,
                    "payload_classes": 2,
                    "protocol_variants": 2,
                },
                "expected_payload_trial_count_distribution": {"2": 4},
            }
        ),
        encoding="utf-8",
    )
    return trials, plan, config


def test_primary_endpoint_is_grouped_hash_bound_and_disclosed(
    tmp_path: Path,
) -> None:
    trials, plan, config = _fixture(tmp_path)
    artifacts = build_primary_endpoint(
        primary_trials=trials,
        confirmatory_plan=plan,
        endpoint_config=config,
        output_dir=tmp_path / "output",
        command="fixture command",
    )
    manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    endpoints = pd.read_csv(manifest["outputs"]["endpoints"]["path"])
    by_id = endpoints.set_index("endpoint_id")
    assert by_id.loc["primary_payload_trial_frozen_endpoint", "successes"] == 7
    assert (
        by_id.loc[
            "primary_payload_strict_all_observed_configurations", "successes"
        ]
        == 3
    )
    assert by_id.loc["primary_model_payload_strict", "successes"] == 7
    assert artifacts.payload_row_count == 4
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["size_bytes"]


def test_primary_endpoint_rejects_recovery_alias_drift(tmp_path: Path) -> None:
    trials, plan, config = _fixture(tmp_path)
    frame = pd.read_csv(trials)
    frame.loc[0, "exact_recovery"] = 1 - frame.loc[0, "exact_payload_recovery"]
    frame.to_csv(trials, index=False)
    with pytest.raises(PrimaryEndpointError, match="compatibility alias"):
        build_primary_endpoint(
            primary_trials=trials,
            confirmatory_plan=plan,
            endpoint_config=config,
            output_dir=tmp_path / "output",
        )
