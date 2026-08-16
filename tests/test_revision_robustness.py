from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_robustness import (
    CONDITION_COLUMNS,
    RobustnessAnalysisError,
    build_robustness_artifacts,
    classify_failure_mechanism,
    file_sha256,
    summarize_recovery_conditions,
)


def _saved_fixture(tmp_path: Path) -> dict[str, Path]:
    trials = pd.DataFrame(
        [
            {
                "trial_id": "replay_success",
                "source_trial_id": "source_1",
                "exact_payload_recovery": 1,
                "exact_recovery": 1,
                "payload_name": "payload_1",
                "payload_class": "sha256_hex",
                "prompt_category": "conversation",
                "robustness_family": "replay_modes",
                "replay_mode": "saved_token_ids",
                "transformation_id": "unmodified",
                "mitigation_id": "none",
                "source_model_id": "model_a",
                "model_id": "model_a",
                "alphabet_size_B": 16,
            },
            {
                "trial_id": "raw_failure",
                "source_trial_id": "source_2",
                "exact_payload_recovery": 0,
                "exact_recovery": 0,
                "payload_name": "payload_2",
                "payload_class": "sha256_hex",
                "prompt_category": "conversation",
                "robustness_family": "raw_transmission",
                "replay_mode": "transformed_text_retokenized",
                "transformation_id": "whitespace_trim",
                "mitigation_id": "none",
                "source_model_id": "model_a",
                "model_id": "model_a",
                "alphabet_size_B": 16,
            },
            {
                "trial_id": "cross_failure",
                "source_trial_id": "source_3",
                "exact_payload_recovery": 0,
                "exact_recovery": 0,
                "payload_name": "payload_3",
                "payload_class": "sha256_hex",
                "prompt_category": "conversation",
                "robustness_family": "cross_model_mismatch",
                "replay_mode": "cross_model_text_retokenized",
                "transformation_id": "unmodified",
                "mitigation_id": "none",
                "source_model_id": "model_a",
                "model_id": "model_b",
                "alphabet_size_B": 16,
            },
        ]
    )
    failures = pd.DataFrame(
        [
            {
                "failure_id": "failure_raw",
                "trial_id": "raw_failure",
                "source_trial_id": "source_2",
                "payload_name": "payload_2",
                "failure_category": "raw_transmission",
                "first_differing_position": 2,
                "expected_token_id": 10,
                "recovered_token_id": 11,
                "expected_rank": 16,
                "recovered_rank": 17,
                "context_sha256": "a" * 64,
                "boundary_start_offset": 0,
                "boundary_end_offset": 8,
                "expected_token_length": 8,
                "recovered_token_length": 8,
                "expected_rank_length": 8,
                "recovered_rank_length": 8,
                "divergence_fields_availability": "recorded",
                "execution_error_type": np.nan,
                "robustness_family": "raw_transmission",
                "replay_mode": "transformed_text_retokenized",
                "transformation_id": "whitespace_trim",
                "source_model_id": "model_a",
                "model_id": "model_a",
            },
            {
                "failure_id": "failure_cross",
                "trial_id": "cross_failure",
                "source_trial_id": "source_3",
                "payload_name": "payload_3",
                "failure_category": "cross_model_mismatch",
                "first_differing_position": 0,
                "expected_token_id": 20,
                "recovered_token_id": 21,
                "expected_rank": 8,
                "recovered_rank": 144,
                "context_sha256": "b" * 64,
                "boundary_start_offset": 0,
                "boundary_end_offset": 8,
                "expected_token_length": 8,
                "recovered_token_length": 8,
                "expected_rank_length": 8,
                "recovered_rank_length": 8,
                "divergence_fields_availability": "recorded",
                "execution_error_type": np.nan,
                "robustness_family": "cross_model_mismatch",
                "replay_mode": "cross_model_text_retokenized",
                "transformation_id": "unmodified",
                "source_model_id": "model_a",
                "model_id": "model_b",
            },
        ]
    )
    unavailable = pd.DataFrame(
        [
            {
                "work_id": "limited_unavailable",
                "source_trial_id": "source_4",
                "payload_name": "payload_4",
                "reason_code": "source_condition_unavailable",
                "root_condition_reason_code": "empty_roundtrip_vocabulary",
                "excluded_from_estimands": True,
                "robustness_family": "limited_mitigation",
                "replay_mode": "canonicalized_text_retokenized",
                "transformation_id": "unmodified",
                "source_model_id": "model_c",
                "model_id": "model_c",
            }
        ]
    )
    robustness_config = {
        "failure_record_required_fields": [
            "trial_id",
            "first_differing_position",
            "expected_token_id",
            "recovered_token_id",
            "expected_rank",
            "recovered_rank",
            "context_sha256",
            "boundary_start_offset",
            "boundary_end_offset",
        ],
        "replay_modes": {"outcome_rows": 1},
        "raw_transmission": {"outcome_rows": 1},
        "limited_mitigation": {"outcome_rows": 1},
        "cross_model_mismatch": {"outcome_rows": 1},
        "expected_counts": {"robustness_outcome_rows": 4},
    }
    statistics_config = {
        "intervals": {
            "confidence_level": 0.95,
            "bootstrap_resamples": 25,
            "bootstrap_seed": 42,
        }
    }
    paths = {
        "trials": tmp_path / "trials.csv",
        "failures": tmp_path / "failures.csv",
        "unavailable": tmp_path / "unavailable.csv",
        "robustness_config": tmp_path / "robustness.json",
        "statistics_config": tmp_path / "statistics.json",
    }
    trials.to_csv(paths["trials"], index=False)
    failures.to_csv(paths["failures"], index=False)
    unavailable.to_csv(paths["unavailable"], index=False)
    paths["robustness_config"].write_text(
        json.dumps(robustness_config), encoding="utf-8"
    )
    paths["statistics_config"].write_text(
        json.dumps(statistics_config), encoding="utf-8"
    )
    return paths


def test_failure_mechanism_labels_are_conservative():
    assert classify_failure_mechanism(
        {
            "robustness_family": "cross_model_mismatch",
            "transformation_id": "unmodified",
        }
    )[0] == "model_or_tokenizer_identity_mismatch"
    assert classify_failure_mechanism(
        {
            "robustness_family": "raw_transmission",
            "transformation_id": "paraphrase",
        }
    )[0] == "semantic_rewrite_divergence"
    assert classify_failure_mechanism(
        {
            "robustness_family": "replay_modes",
            "replay_mode": "detokenized_text_retokenized",
            "transformation_id": "unmodified",
        }
    )[0] == "detokenization_retokenization_divergence"


def test_source_cover_interval_selection():
    binary = pd.DataFrame(
        {
            "robustness_family": ["raw_transmission"],
            "replay_mode": ["transformed_text_retokenized"],
            "transformation_id": ["unmodified"],
            "source_trial_id": ["source"],
            "exact_payload_recovery": [1],
        }
    )
    summary = summarize_recovery_conditions(
        binary,
        binary.iloc[0:0],
        group_columns=CONDITION_COLUMNS,
        confidence_level=0.95,
        n_resamples=10,
        seed=1,
    )
    assert summary.iloc[0]["interval_method"] == "source_cover_wilson"
    assert summary.iloc[0]["ci_low"] < 1.0

    repeated = pd.concat(
        [binary.assign(exact_payload_recovery=0), binary], ignore_index=True
    )
    summary = summarize_recovery_conditions(
        repeated,
        binary.iloc[0:0],
        group_columns=CONDITION_COLUMNS,
        confidence_level=0.95,
        n_resamples=10,
        seed=1,
    )
    assert summary.iloc[0]["interval_method"] == (
        "source_cover_grouped_percentile_bootstrap"
    )
    assert summary.iloc[0]["recovery_rate"] == pytest.approx(0.5)


def test_build_robustness_artifacts_is_hashed_and_fail_closed(tmp_path):
    paths = _saved_fixture(tmp_path)
    output_dir = tmp_path / "output"
    artifacts = build_robustness_artifacts(
        trials_path=paths["trials"],
        failures_path=paths["failures"],
        unavailable_path=paths["unavailable"],
        robustness_config=paths["robustness_config"],
        statistics_config=paths["statistics_config"],
        output_dir=output_dir,
    )
    assert artifacts.summary == {
        "observed_rows": 3,
        "failure_rows": 2,
        "unavailable_rows": 1,
        "planned_rows": 4,
        "success_rows": 1,
        "recovery_failure_rows": 2,
        "execution_failure_rows": 0,
        "family_counts": {
            "cross_model_mismatch": 1,
            "limited_mitigation": 1,
            "raw_transmission": 1,
            "replay_modes": 1,
        },
        "unavailable_rows_are_not_recovery_failures": True,
    }
    conditions = pd.read_csv(artifacts.files["conditions"])
    unavailable = conditions[conditions["status"].eq("unavailable")]
    assert len(unavailable) == 1
    assert np.isnan(unavailable.iloc[0]["recovery_rate"])
    taxonomy = pd.read_csv(artifacts.files["failure_taxonomy"])
    assert taxonomy["recovered_rank_out_of_bound"].all()
    assert set(taxonomy["failure_mechanism"]) == {
        "model_or_tokenizer_identity_mismatch",
        "whitespace_or_markup_tokenization_divergence",
    }
    manifest = json.loads(Path(artifacts.files["manifest"]).read_text())
    for output in manifest["outputs"].values():
        assert file_sha256(output["path"]) == output["sha256"]
    with pytest.raises(RobustnessAnalysisError, match="Refusing to overwrite"):
        build_robustness_artifacts(
            trials_path=paths["trials"],
            failures_path=paths["failures"],
            unavailable_path=paths["unavailable"],
            robustness_config=paths["robustness_config"],
            statistics_config=paths["statistics_config"],
            output_dir=output_dir,
        )
