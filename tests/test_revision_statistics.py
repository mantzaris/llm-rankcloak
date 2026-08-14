import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_statistics import (
    PAYLOAD_RECOVERY_SEMANTICS,
    MixedEffectsUnavailable,
    RevisionStatisticsError,
    adjust_pvalues,
    automated_text_quality_metrics,
    build_trial_quality_table,
    exact_binomial_interval,
    fit_r_lme4,
    fit_statsmodels_mixedlm,
    grouped_payload_bootstrap_ci,
    pairwise_effect_sizes,
    run_mixed_effects_specs,
    run_statistics_analysis,
    summarize_detector_results,
    summarize_recovery,
    synthetic_smoke_frames,
    validate_detector_rows,
    validate_trial_results,
    wilson_interval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATISTICS_CONFIG = (
    PROJECT_ROOT / "configs" / "revision_v1" / "statistics.json"
)


def _direct_payload_pathology_trials() -> pd.DataFrame:
    """Reproduce the three direct-subword fidelity pathologies from smoke_v2."""

    cases = (
        (
            "llama3_8b_instruct_q4_k_m",
            "07Zonz3zYvY4payload",
            " 07Zonz3zYvY4payload",
        ),
        (
            "mistral_7b_instruct_v0_3_q4_k_m",
            "Zq2k9payload",
            "  Zq2k9payload",
        ),
        (
            "qwen2_5_7b_instruct_q4_k_m",
            "/mGs9payload",
            "mGs9payload",
        ),
    )
    base = synthetic_smoke_frames()["trials"].iloc[0].to_dict()
    rows = []
    for index, (model_id, original, recovered) in enumerate(cases):
        original_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
        recovered_hash = hashlib.sha256(recovered.encode("utf-8")).hexdigest()
        exact_payload = int(original_hash == recovered_hash)
        rows.append(
            {
                **base,
                "trial_id": f"direct_payload_pathology_{index}",
                "payload_name": f"direct_payload_{index}",
                "model_id": model_id,
                "protocol_variant": "direct_subword_calgacus",
                "representation_name": "direct_subword",
                "codec_id": "raw_subword_direct",
                "replay_mode": "saved_token_ids",
                "transformation_id": "unmodified",
                "mitigation_id": "none",
                "protocol_contract_revision": "payload_fidelity_v2",
                "result_schema_revision": "payload_aware_result_v2",
                "exact_rank_replay": 1,
                "exact_payload_recovery": exact_payload,
                "exact_recovery": exact_payload,
                "recovery_outcome_semantics": PAYLOAD_RECOVERY_SEMANTICS,
                "original_payload_sha256": original_hash,
                "recovered_payload_sha256": recovered_hash,
            }
        )
    return pd.DataFrame(rows)


def test_binomial_intervals_cover_observed_rate_and_handle_boundaries():
    wilson_low, wilson_high = wilson_interval(8, 10)
    exact_low, exact_high = exact_binomial_interval(8, 10)
    assert wilson_low < 0.8 < wilson_high
    assert exact_low < 0.8 < exact_high
    assert exact_binomial_interval(0, 10)[0] == 0.0
    assert exact_binomial_interval(10, 10)[1] == 1.0
    with pytest.raises(RevisionStatisticsError):
        wilson_interval(11, 10)


def test_grouped_bootstrap_equal_weights_payloads_not_nested_rows():
    result = grouped_payload_bootstrap_ci(
        [0.0, 0.0, 10.0],
        ["payload_a", "payload_a", "payload_b"],
        n_resamples=100,
        seed=22,
    )
    assert result["mean"] == 5.0
    assert result["n_payloads"] == 2
    assert result == grouped_payload_bootstrap_ci(
        [0.0, 0.0, 10.0],
        ["payload_a", "payload_a", "payload_b"],
        n_resamples=100,
        seed=22,
    )


def test_holm_and_bh_adjustments_are_monotone_and_preserve_missing():
    raw = [0.01, 0.04, 0.03, None]
    assert adjust_pvalues(raw, "holm") == pytest.approx(
        [0.03, 0.06, 0.06, None], nan_ok=True
    )
    assert adjust_pvalues(raw, "bh") == pytest.approx(
        [0.03, 0.04, 0.04, None], nan_ok=True
    )


def test_recovery_summary_uses_payload_units_and_rejects_duplicates():
    trials = synthetic_smoke_frames()["trials"]
    result = summarize_recovery(trials)
    assert set(result["analysis_unit"]) == {"payload"}
    assert set(result["n_payloads"]) == {12}
    duplicated = pd.concat([trials, trials.iloc[[0]]], ignore_index=True)
    with pytest.raises(RevisionStatisticsError, match="Duplicate trial-condition"):
        validate_trial_results(duplicated)


def test_direct_subword_recovery_uses_payload_fidelity_not_rank_replay():
    trials = _direct_payload_pathology_trials()
    validated, _ = validate_trial_results(trials)
    contract = validated.attrs["payload_fidelity_contract"]
    assert contract == {
        "contract_version": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "semantics": PAYLOAD_RECOVERY_SEMANTICS,
        "primary_outcome": "exact_payload_recovery",
        "compatibility_alias": "exact_recovery",
        "alias_equality_validated": True,
        "exact_rank_replay_role": "diagnostic_only",
        "direct_rows": 3,
        "direct_rows_contract_verified": 3,
    }
    assert validated["exact_rank_replay"].astype(int).eq(1).all()
    assert validated["exact_payload_recovery"].astype(int).eq(0).all()
    assert validated["exact_recovery"].eq(0).all()

    summary = summarize_recovery(
        trials, group_columns=["protocol_variant", "replay_mode"]
    ).iloc[0]
    assert summary["protocol_contract_revision"] == "payload_fidelity_v2"
    assert summary["result_schema_revision"] == "payload_aware_result_v2"
    assert summary["recovery_outcome"] == "exact_payload_recovery"
    assert summary["recovery_outcome_semantics"] == PAYLOAD_RECOVERY_SEMANTICS
    assert summary["payload_recovery_successes"] == 0
    assert summary["exact_payload_recovery_rate"] == 0.0
    assert summary["successes"] == 0
    assert summary["exact_recovery_rate"] == 0.0
    assert summary["rank_replay_n"] == 3
    assert summary["rank_replay_successes"] == 3
    assert summary["exact_rank_replay_rate"] == 1.0
    assert bool(summary["rank_replay_diagnostic_only"])


def test_direct_subword_payload_fidelity_contract_fails_closed():
    valid = _direct_payload_pathology_trials()

    legacy = valid.drop(
        columns=[
            "exact_rank_replay",
            "exact_payload_recovery",
            "recovery_outcome_semantics",
        ]
    )
    with pytest.raises(RevisionStatisticsError, match="contract missing columns"):
        validate_trial_results(legacy)
    with pytest.raises(RevisionStatisticsError, match="contract missing columns"):
        pairwise_effect_sizes(
            legacy,
            outcome="exact_recovery",
            factor="model_id",
            binary=True,
            n_resamples=10,
        )

    wrong_semantics = valid.copy()
    wrong_semantics["recovery_outcome_semantics"] = "rank_token_ids_equal"
    with pytest.raises(RevisionStatisticsError, match="must equal"):
        validate_trial_results(wrong_semantics)

    alias_mismatch = valid.copy()
    alias_mismatch.loc[alias_mismatch.index[0], "exact_recovery"] = 1
    with pytest.raises(RevisionStatisticsError, match="compatibility alias differs"):
        validate_trial_results(alias_mismatch)

    missing_payload_value = valid.copy()
    missing_payload_value.loc[
        missing_payload_value.index[0], "exact_payload_recovery"
    ] = np.nan
    with pytest.raises(RevisionStatisticsError, match="missing exact_payload_recovery"):
        validate_trial_results(missing_payload_value)


def test_heldout_direct_quality_rows_are_not_recovery_observations():
    recovery = _direct_payload_pathology_trials()
    recovery["preprocess_schema_version"] = "2.0"
    recovery["evidence_status"] = (
        "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling"
    )
    recovery["study_phase"] = "smoke_v3_exploratory"
    quality = recovery.iloc[[0]].copy()
    quality["trial_id"] = "heldout_quality_direct_0"
    quality["record_type"] = "heldout_evaluator_feature"
    quality["study_phase"] = "ordinary_llm_control_smoke_v3"
    quality["heldout_evaluator_log_probability"] = -3.25
    quality = quality.drop(
        columns=[
            "exact_rank_replay",
            "exact_payload_recovery",
            "exact_recovery",
            "recovery_outcome_semantics",
        ]
    )

    combined = pd.concat([recovery, quality], ignore_index=True, sort=False)
    validated, _ = validate_trial_results(combined)
    contract = validated.attrs["payload_fidelity_contract"]
    assert contract["direct_rows"] == len(recovery)
    assert contract["direct_rows_contract_verified"] == len(recovery)
    observed_quality = validated[
        validated["record_type"].eq("heldout_evaluator_feature")
    ]
    assert len(observed_quality) == 1
    assert observed_quality.iloc[0]["heldout_evaluator_log_probability"] == -3.25

    wrong_phase = combined.copy()
    wrong_phase.loc[wrong_phase["record_type"].eq("heldout_evaluator_feature"), "study_phase"] = (
        "ordinary_llm_control_smoke_v2"
    )
    with pytest.raises(RevisionStatisticsError, match="legacy or mismatched"):
        validate_trial_results(wrong_phase)


def test_preprocessed_trials_require_superseding_stage_and_schema_labels():
    valid = _direct_payload_pathology_trials()
    valid["preprocess_schema_version"] = "2.0"
    valid["evidence_status"] = (
        "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
    )
    valid["study_phase"] = "primary_v2_confirmatory"
    validate_trial_results(valid)

    old_evidence = valid.copy()
    old_evidence["evidence_status"] = "confirmatory_after_manifest_freeze"
    with pytest.raises(RevisionStatisticsError, match="legacy or mismatched"):
        validate_trial_results(old_evidence)

    old_phase = valid.copy()
    old_phase["study_phase"] = "primary_confirmatory"
    with pytest.raises(RevisionStatisticsError, match="legacy or mismatched"):
        validate_trial_results(old_phase)

    missing_result_revision = valid.drop(columns=["result_schema_revision"])
    with pytest.raises(RevisionStatisticsError, match="missing columns"):
        validate_trial_results(missing_result_revision)

    wrong_result_revision = valid.copy()
    wrong_result_revision["result_schema_revision"] = "rank_only_result_v1"
    with pytest.raises(RevisionStatisticsError, match="payload_aware_result_v2"):
        validate_trial_results(wrong_result_revision)


def test_valid_direct_payload_effects_carry_explicit_semantics():
    effects = pairwise_effect_sizes(
        _direct_payload_pathology_trials(),
        outcome="exact_recovery",
        factor="model_id",
        binary=True,
        n_resamples=10,
    )
    assert not effects.empty
    assert set(effects["protocol_contract_revision"]) == {
        "payload_fidelity_v2"
    }
    assert set(effects["result_schema_revision"]) == {
        "payload_aware_result_v2"
    }
    assert set(effects["recovery_outcome"]) == {"exact_payload_recovery"}
    assert set(effects["recovery_outcome_semantics"]) == {
        PAYLOAD_RECOVERY_SEMANTICS
    }
    assert effects["exact_recovery_compatibility_alias"].astype(bool).all()
    assert effects["exact_rank_replay_diagnostic_only"].astype(bool).all()
    assert effects["mean_first"].eq(0.0).all()
    assert effects["mean_second"].eq(0.0).all()


def test_recovery_preserves_replay_strata_and_excludes_unavailable_rows():
    saved = (
        synthetic_smoke_frames()["trials"]
        .drop_duplicates(["payload_name", "model_id"])
        .iloc[:4]
        .copy()
    )
    saved["replay_mode"] = "saved_token_ids"
    saved["exact_recovery"] = 1
    diagnostic = saved.copy()
    diagnostic["trial_id"] += "__retokenized"
    diagnostic["replay_mode"] = "detokenized_text_retokenized"
    diagnostic["exact_recovery"] = 0
    unavailable = saved.iloc[[0]].copy()
    unavailable["trial_id"] = "scientifically_unavailable"
    unavailable["payload_name"] = "payload_unavailable"
    unavailable["exact_recovery"] = np.nan
    unavailable["record_type"] = "condition_unavailable"
    unavailable["excluded_from_estimands"] = True
    combined = pd.concat([saved, diagnostic, unavailable], ignore_index=True)

    summary = summarize_recovery(combined, group_columns=["model_id"])
    assert set(summary["replay_mode"]) == {
        "saved_token_ids",
        "detokenized_text_retokenized",
    }
    assert summary["n_payloads"].sum() == 8
    assert summary.loc[
        summary["replay_mode"].eq("saved_token_ids"), "successes"
    ].sum() == 4
    assert summary.loc[
        summary["replay_mode"].eq("detokenized_text_retokenized"), "successes"
    ].sum() == 0


def test_pairwise_effects_reject_implicit_replay_pooling():
    saved = synthetic_smoke_frames()["trials"].copy()
    saved["replay_mode"] = "saved_token_ids"
    diagnostic = saved.copy()
    diagnostic["trial_id"] += "__retokenized"
    diagnostic["replay_mode"] = "detokenized_text_retokenized"
    with pytest.raises(RevisionStatisticsError, match="would pool replay_mode"):
        pairwise_effect_sizes(
            pd.concat([saved, diagnostic], ignore_index=True),
            outcome="exact_recovery",
            factor="protocol_variant",
            binary=True,
            n_resamples=10,
        )


def test_quality_collapses_nested_segments_and_labels_surface_scope():
    features = synthetic_smoke_frames()["features"]
    quality = build_trial_quality_table(features)
    assert len(quality) == 24
    assert set(quality["nested_segment_count"]) == {2}
    assert not quality["human_rating_substitute"].any()
    assert set(quality["readability_scope"]) == {"english_surface_heuristic"}
    assert quality["tfidf_prompt_similarity"].between(0.0, 1.0).all()
    flags = automated_text_quality_metrics(
        'this sentence starts low  with spaces and an unmatched ( quote "',
        "A prompt",
    )
    assert flags["surface_flag_total"] > 0
    assert flags["human_rating_substitute"] is False


def test_quality_preserves_exact_segment_join_and_token_weights_log_probability():
    features = pd.DataFrame(
        {
            "trial_id": ["trial", "trial"],
            "payload_name": ["payload", "payload"],
            "model_id": ["model", "model"],
            "protocol_variant": ["segmented", "segmented"],
            "prompt_id": ["prompt", "prompt"],
            "text_view": ["full_message", "full_message"],
            "segment_index": [0, 1],
            "text": [" leading segment ", "second segment\n"],
            "prompt_text": ["Prompt", "Prompt"],
            "token_count": [1, 3],
            "mean_log_probability": [-8.0, -2.0],
        }
    )
    quality = build_trial_quality_table(features)
    assert len(quality) == 1
    expected_text = " leading segment \n\nsecond segment\n"
    assert quality.iloc[0]["text_sha256"] == hashlib.sha256(
        expected_text.encode("utf-8")
    ).hexdigest()
    assert quality.iloc[0]["source_mean_log_probability"] == pytest.approx(-3.5)


def test_nested_log_probability_fails_closed_without_complete_weights():
    features = pd.DataFrame(
        {
            "trial_id": ["trial", "trial"],
            "payload_name": ["payload", "payload"],
            "model_id": ["model", "model"],
            "protocol_variant": ["segmented", "segmented"],
            "prompt_id": ["prompt", "prompt"],
            "text_view": ["full_message", "full_message"],
            "segment_index": [0, 1],
            "text": ["first", "second"],
            "prompt_text": ["Prompt", "Prompt"],
            "token_count": [1, None],
            "mean_log_probability": [-8.0, -2.0],
        }
    )
    with pytest.raises(RevisionStatisticsError, match="token_count weights"):
        build_trial_quality_table(features)


def test_partition_leakage_and_detector_duplicate_guards():
    leaked = pd.DataFrame(
        {
            "payload_group_id": ["p1", "p1"],
            "partition": ["train", "test"],
            "row_id": ["a", "b"],
            "label": [0, 1],
            "score": [0.1, 0.9],
        }
    )
    with pytest.raises(RevisionStatisticsError, match="leakage"):
        validate_detector_rows(leaked)


def test_detector_prediction_summary_is_payload_group_bootstrapped():
    predictions = synthetic_smoke_frames()["detectors"]
    result = summarize_detector_results(predictions, n_resamples=50, seed=9)
    assert len(result) == 1
    assert result.iloc[0]["bootstrap_unit"] == "payload_group_id"
    assert result.iloc[0]["analysis_unit"] == "payload_group"
    assert result.iloc[0]["roc_auc"] == 1.0


def test_paired_payload_effect_sizes_and_multiplicity_columns():
    trials = synthetic_smoke_frames()["trials"]
    effects = pairwise_effect_sizes(
        trials,
        outcome="effective_payload_rate",
        factor="protocol_variant",
        binary=False,
        n_resamples=50,
        seed=8,
    )
    assert len(effects) == 1
    assert effects.iloc[0]["comparison_design"] == "paired_payload"
    assert effects.iloc[0]["n_payloads_paired"] == 12
    assert "hedges_g" in effects
    assert "p_value_holm" in effects
    assert "p_value_bh" in effects


def test_quality_effects_require_an_explicit_single_view_scope():
    full_message = synthetic_smoke_frames()["features"].copy()
    forced_span = full_message.copy()
    forced_span["trial_id"] += "__forced_span"
    forced_span["text_view"] = "forced_span"
    quality = build_trial_quality_table(
        pd.concat([full_message, forced_span], ignore_index=True)
    )
    assert set(quality["view"]) == {"forced_span", "full_message"}
    with pytest.raises(RevisionStatisticsError, match="would pool view levels"):
        pairwise_effect_sizes(
            quality,
            outcome="flesch_reading_ease_heuristic",
            factor="protocol_variant",
            n_resamples=10,
        )


def test_quality_effects_are_stratified_by_smoke_source_phase(tmp_path):
    rankcloak = synthetic_smoke_frames()["features"].copy()
    rankcloak["evidence_status"] = (
        "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling"
    )
    rankcloak["study_phase"] = "smoke_v3_exploratory"
    rankcloak["protocol_contract_revision"] = "payload_fidelity_v2"
    rankcloak["result_schema_revision"] = "payload_aware_result_v2"
    controls = rankcloak.copy()
    controls["trial_id"] += "__control"
    controls["protocol_variant"] = "ordinary_llm_control"
    controls["study_phase"] = "ordinary_llm_control_smoke_v3"
    features = pd.concat([rankcloak, controls], ignore_index=True)
    path = tmp_path / "features.csv"
    features.to_csv(path, index=False)

    artifacts = run_statistics_analysis(
        output_dir=tmp_path / "analysis",
        feature_paths=[path],
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    effects = pd.read_csv(artifacts.files["effects"])
    quality_effects = effects[
        effects["outcome"].isin(
            {
                "flesch_reading_ease_heuristic",
                "repeated_bigram_fraction",
                "surface_flag_total",
                "tfidf_prompt_similarity",
            }
        )
    ]
    assert not quality_effects.empty
    assert not quality_effects["study_phase_scope"].isna().any()
    assert set(quality_effects["study_phase_scope"]) <= {
        "smoke_v3_exploratory",
        "ordinary_llm_control_smoke_v3",
    }


def test_partial_payload_overlap_retains_descriptive_effect_but_omits_pvalue():
    frame = pd.DataFrame(
        {
            "payload_name": ["p1", "p2", "p2", "p3"],
            "condition": ["a", "a", "b", "b"],
            "quality": [1.0, 3.0, 2.0, 4.0],
            "study_phase": ["exploratory"] * 4,
        }
    )
    result = pairwise_effect_sizes(
        frame,
        outcome="quality",
        factor="condition",
        n_resamples=100,
        seed=45,
    )
    row = result.iloc[0]
    assert row["comparison_design"] == "partially_overlapping_payload"
    assert row["n_payloads_overlap"] == 1
    assert row["n_payloads_paired"] == 0
    assert row["mean_first"] == 2.0
    assert row["mean_second"] == 3.0
    assert row["mean_difference"] == -1.0
    assert row["bootstrap_design"] == "payload_cluster_partial_overlap"
    assert row["bootstrap_resamples_valid"] > 0
    assert np.isfinite(row["mean_difference_ci_low"])
    assert np.isfinite(row["mean_difference_ci_high"])
    assert row["test"] == "unsupported_partial_payload_overlap"
    assert pd.isna(row["p_value_raw"])
    assert pd.isna(row["p_value_holm"])
    assert pd.isna(row["p_value_bh"])
    assert not bool(row["inferential_p_value_supported"])
    assert not bool(row["primary_inference"])
    assert row["inference_role"] == "descriptive_exploratory_pairwise"


def test_statsmodels_adapter_executes_random_intercept_model():
    rng = np.random.default_rng(51)
    rows = []
    for payload_index in range(24):
        random_intercept = rng.normal(0.0, 0.8)
        for repetition in range(4):
            treatment = repetition % 2
            rows.append(
                {
                    "payload_id": f"p{payload_index:02d}",
                    "treatment": treatment,
                    "quality": (
                        1.5
                        + 1.8 * treatment
                        + random_intercept
                        + rng.normal(0.0, 0.25)
                    ),
                }
            )
    coefficients = fit_statsmodels_mixedlm(
        pd.DataFrame(rows),
        outcome="quality",
        fixed_effects=["treatment"],
        group_column="payload_id",
        name="synthetic_quality",
    )
    treatment = coefficients[coefficients["term"] == "treatment"].iloc[0]
    assert treatment["estimate"] == pytest.approx(1.8, abs=0.2)
    assert bool(treatment["converged"])
    assert not bool(treatment["fixed_effects_fallback"])


def test_mixed_model_requests_fail_explicitly_without_fixed_fallback():
    trials = synthetic_smoke_frames()["trials"]
    spec = {
        "name": "logistic_requires_r",
        "backend": "statsmodels",
        "family": "binomial",
        "data_source": "trials",
        "outcome": "exact_recovery",
        "fixed_effects": ["protocol_variant"],
        "group_column": "payload_name",
        "required": False,
    }
    coefficients, statuses = run_mixed_effects_specs(
        {"trials": trials}, [spec]
    )
    assert coefficients.empty
    assert statuses[0]["status"] == "unavailable_or_failed"
    assert statuses[0]["fixed_effects_fallback"] is False
    assert "Gaussian MixedLM only" in statuses[0]["error"]
    with pytest.raises(MixedEffectsUnavailable, match="no fixed-effects"):
        run_mixed_effects_specs(
            {"trials": trials}, [{**spec, "required": True}]
        )
    with pytest.raises(MixedEffectsUnavailable, match="unavailable"):
        fit_r_lme4(
            trials,
            formula="exact_recovery ~ protocol_variant + (1 | payload_name)",
            family="binomial",
            name="missing_r",
            rscript="definitely-not-an-rscript",
        )


def test_mixed_recovery_model_rejects_legacy_direct_outcome_contract():
    legacy = _direct_payload_pathology_trials().drop(
        columns=[
            "exact_rank_replay",
            "exact_payload_recovery",
            "recovery_outcome_semantics",
        ]
    )
    spec = {
        "name": "legacy_direct_recovery",
        "backend": "statsmodels",
        "family": "binomial",
        "data_source": "trials",
        "outcome": "exact_recovery",
        "fixed_effects": ["model_id"],
        "group_column": "payload_name",
        "required": False,
    }
    coefficients, statuses = run_mixed_effects_specs(
        {"trials": legacy}, [spec], fail_required=False
    )
    assert coefficients.empty
    assert statuses[0]["status"] == "unavailable_or_failed"
    assert statuses[0]["error_type"] == "RevisionStatisticsError"
    assert "payload-fidelity contract missing columns" in statuses[0]["error"]


def test_end_to_end_valid_direct_contract_is_recorded(tmp_path):
    trials_path = tmp_path / "valid_direct_trials.csv"
    _direct_payload_pathology_trials().to_csv(trials_path, index=False)
    artifacts = run_statistics_analysis(
        output_dir=tmp_path / "valid-direct-analysis",
        trial_paths=[trials_path],
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    contract = artifacts.integrity_report["payload_fidelity_contract"]
    assert contract["direct_rows"] == 3
    assert contract["direct_rows_contract_verified"] == 3
    assert contract["primary_outcome"] == "exact_payload_recovery"
    assert contract["exact_rank_replay_role"] == "diagnostic_only"
    recovery = pd.read_csv(artifacts.files["recovery"])
    assert recovery["payload_recovery_successes"].sum() == 0
    assert recovery["rank_replay_successes"].sum() == 3


def test_end_to_end_smoke_writes_hashed_machine_outputs(tmp_path):
    output_dir = tmp_path / "analysis"
    artifacts = run_statistics_analysis(
        output_dir=output_dir,
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    assert artifacts.integrity_report["analysis_unit"] == "payload"
    assert (
        artifacts.integrity_report["segments_as_independent_observations"]
        is False
    )
    assert artifacts.integrity_report["payload_fidelity_contract"] == {
        "contract_version": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "semantics": PAYLOAD_RECOVERY_SEMANTICS,
        "primary_outcome": "exact_payload_recovery",
        "compatibility_alias": "exact_recovery",
        "alias_equality_validated": True,
        "exact_rank_replay_role": "diagnostic_only",
        "direct_rows": 0,
        "direct_rows_contract_verified": 0,
    }
    for path in artifacts.files.values():
        assert Path(path).is_file()
    manifest = json.loads(
        (output_dir / "statistics_run_manifest.json").read_text()
    )
    assert manifest["statistics_config"]["sha256"]
    assert manifest["outputs"]["recovery"]["sha256"]
    quality = pd.read_csv(output_dir / "quality_trial_metrics.csv")
    assert not quality["human_rating_substitute"].any()
    recovery = pd.read_csv(output_dir / "recovery_summary.csv")
    assert set(recovery["protocol_contract_revision"]) == {
        "payload_fidelity_v2"
    }
    assert set(recovery["result_schema_revision"]) == {
        "payload_aware_result_v2"
    }
    assert set(recovery["recovery_outcome"]) == {"exact_payload_recovery"}
    assert set(recovery["recovery_outcome_semantics"]) == {
        PAYLOAD_RECOVERY_SEMANTICS
    }
    effects = pd.read_csv(output_dir / "effect_sizes.csv")
    quality_effects = effects[
        effects["outcome"].isin(
            {
                "flesch_reading_ease_heuristic",
                "repeated_bigram_fraction",
                "surface_flag_total",
                "tfidf_prompt_similarity",
            }
        )
    ]
    assert not quality_effects.empty
    assert set(quality_effects["view_scope"]) == {"full_message"}
    assert not quality_effects["view_scope"].isna().any()
    assert not quality_effects["primary_inference"].astype(bool).any()
    assert artifacts.integrity_report["quality_effect_scope"] == {
        "view_column": "view",
        "view_levels": ["full_message"],
        "view_stratified": True,
        "effect_rows": len(quality_effects),
        "unscoped_effect_rows": 0,
        "exclusion_reason": None,
    }
    with pytest.raises(RevisionStatisticsError, match="overwrite"):
        run_statistics_analysis(
            output_dir=output_dir,
            statistics_config=STATISTICS_CONFIG,
            smoke=True,
        )


def test_primary_binary_effects_do_not_average_diagnostic_replay_modes(tmp_path):
    saved = synthetic_smoke_frames()["trials"].copy()
    saved["replay_mode"] = "saved_token_ids"
    saved["exact_recovery"] = 1
    diagnostic = saved.copy()
    diagnostic["replay_mode"] = "detokenized_text_retokenized"
    diagnostic["exact_recovery"] = np.arange(len(diagnostic)) % 2
    path = tmp_path / "mixed_replay_trials.csv"
    pd.concat([saved, diagnostic], ignore_index=True).to_csv(path, index=False)

    artifacts = run_statistics_analysis(
        output_dir=tmp_path / "mixed-replay-analysis",
        trial_paths=[path],
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    effects = pd.read_csv(artifacts.files["effects"])
    recovery_effects = effects[effects["outcome"].eq("exact_recovery")]
    assert not recovery_effects.empty
    assert recovery_effects["mean_first"].eq(1.0).all()
    assert recovery_effects["mean_second"].eq(1.0).all()
    assert recovery_effects["replay_mode_scope"].eq("saved_token_ids").all()


def test_primary_effects_never_fall_back_to_diagnostic_replay(tmp_path):
    diagnostic = synthetic_smoke_frames()["trials"].copy()
    diagnostic["replay_mode"] = "detokenized_text_retokenized"
    path = tmp_path / "diagnostic_only_trials.csv"
    diagnostic.to_csv(path, index=False)

    artifacts = run_statistics_analysis(
        output_dir=tmp_path / "diagnostic-only-analysis",
        trial_paths=[path],
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    effects = pd.read_csv(artifacts.files["effects"])
    assert "outcome" not in effects or not effects["outcome"].eq(
        "exact_recovery"
    ).any()
    scope = artifacts.integrity_report["primary_effect_scope"]
    assert scope["replay_mode"] == "saved_token_ids"
    assert scope["eligible_trial_rows"] == 0
    assert scope["diagnostic_replay_fallback"] is False


def test_statistics_analysis_excludes_explicitly_unavailable_trials(tmp_path):
    trials = synthetic_smoke_frames()["trials"].copy()
    trials["replay_mode"] = "saved_token_ids"
    unavailable = trials.iloc[[0]].copy()
    unavailable["trial_id"] = "condition_unavailable"
    unavailable["payload_name"] = "payload_unavailable"
    unavailable["exact_recovery"] = np.nan
    unavailable["record_type"] = "condition_unavailable"
    unavailable["excluded_from_estimands"] = True
    path = tmp_path / "trials_with_unavailable.csv"
    pd.concat([trials, unavailable], ignore_index=True).to_csv(path, index=False)

    artifacts = run_statistics_analysis(
        output_dir=tmp_path / "unavailable-analysis",
        trial_paths=[path],
        statistics_config=STATISTICS_CONFIG,
        smoke=True,
    )
    recovery = pd.read_csv(artifacts.files["recovery"])
    assert recovery["n_payloads"].sum() == len(trials)
    assert artifacts.integrity_report["estimand_exclusions"] == {
        "unavailable_trial_rows": 1,
        "unavailable_rows_counted": False,
    }


def test_cli_smoke_fixture(tmp_path):
    output_dir = tmp_path / "cli"
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_revision_statistics.py"),
            "--smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["analysis_unit"] == "payload"
    assert report["segments_as_independent_observations"] is False
