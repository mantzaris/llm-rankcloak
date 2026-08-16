from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_multilingual_endpoint import (
    MultilingualEndpointError,
    build_multilingual_endpoint,
    canonical_json_sha256,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for language in ("es", "zh_hans"):
        for payload in range(2):
            for model in range(2):
                for protocol in range(2):
                    success = int(
                        not (
                            language == "es"
                            and payload == 1
                            and model == 1
                            and protocol == 1
                        )
                    )
                    rows.append(
                        {
                            "trial_id": (
                                f"{language}-{payload}-{model}-{protocol}"
                            ),
                            "record_type": "rankcloak_trial",
                            "evidence_status": "fixture_evidence",
                            "study_phase": "fixture_phase",
                            "replay_mode": "saved_token_ids",
                            "protocol_contract_revision": "payload_fidelity_v2",
                            "result_schema_revision": "payload_aware_result_v2",
                            "exact_rank_replay": success,
                            "exact_payload_recovery": success,
                            "exact_recovery": success,
                            "recovery_outcome_semantics": (
                                "original_serialized_payload_bytes_sha256_v1"
                            ),
                            "language": language,
                            "payload_name": f"payload-{payload}",
                            "payload_class": f"class-{payload}",
                            "model_id": f"model-{model}",
                            "prompt_id": f"{language}-prompt-{protocol}",
                            "prompt_category": f"category-{protocol}",
                            "protocol_variant": f"protocol-{protocol}",
                        }
                    )
    trials = tmp_path / "trials.csv"
    pd.DataFrame(rows).to_csv(trials, index=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": (
                    "rankcloak-multilingual-recovery-endpoint-config-v1"
                ),
                "analysis_status": (
                    "secondary_post_audit_language_group_sensitivity"
                ),
                "outcomes_were_available_before_grouping_specification": True,
                "frozen_generation_design_unchanged": True,
                "strict_payload_success_rule": (
                    "all_observed_multilingual_trials_for_language_payload_must_recover"
                ),
                "confidence_level": 0.95,
                "frozen_filter": {
                    "record_type": "rankcloak_trial",
                    "evidence_status": "fixture_evidence",
                    "study_phase": "fixture_phase",
                    "replay_mode": "saved_token_ids",
                    "protocol_contract_revision": "payload_fidelity_v2",
                    "result_schema_revision": "payload_aware_result_v2",
                },
                "expected_design": {
                    "trial_rows": 16,
                    "payloads": 2,
                    "language_payload_groups": 4,
                    "language_model_payload_groups": 8,
                    "model_families": 2,
                    "prompt_templates": 4,
                    "prompt_categories": 2,
                    "payload_classes": 2,
                    "protocol_variants": 2,
                    "languages": ["es", "zh_hans"],
                    "trials_per_language_payload": 4,
                    "trials_per_language_model_payload": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return trials, config


def test_multilingual_endpoint_groups_by_language_and_payload(
    tmp_path: Path,
) -> None:
    trials, config = _fixture(tmp_path)
    artifacts = build_multilingual_endpoint(
        multilingual_trials=trials,
        endpoint_config=config,
        output_dir=tmp_path / "output",
        command="fixture",
    )
    assert artifacts.endpoint_row_count == 6
    assert artifacts.language_payload_row_count == 4
    manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    endpoints = pd.read_csv(manifest["outputs"]["endpoints"]["path"])
    by_id = endpoints.set_index("endpoint_id")
    assert by_id.loc["multilingual_es_payload_trial", "successes"] == 7
    assert by_id.loc["multilingual_es_strict_payload", "successes"] == 1
    assert (
        by_id.loc["multilingual_es_strict_model_payload", "successes"] == 3
    )
    assert by_id.loc["multilingual_zh_hans_payload_trial", "successes"] == 8
    for declaration in manifest["outputs"].values():
        path = Path(declaration["path"])
        assert file_sha256(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["size_bytes"]


def test_multilingual_endpoint_rejects_recovery_alias_drift(
    tmp_path: Path,
) -> None:
    trials, config = _fixture(tmp_path)
    frame = pd.read_csv(trials)
    frame.loc[0, "exact_recovery"] = 1 - frame.loc[
        0, "exact_payload_recovery"
    ]
    frame.to_csv(trials, index=False)
    with pytest.raises(
        MultilingualEndpointError, match="compatibility alias"
    ):
        build_multilingual_endpoint(
            multilingual_trials=trials,
            endpoint_config=config,
            output_dir=tmp_path / "output",
        )
