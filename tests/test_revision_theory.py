import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from rankcloak.revision_theory import (
    OUTPUT_FILENAMES,
    TOKEN_ID_TIE_BREAK_RULE,
    Delta_B,
    Q_B,
    R_B,
    R_effective,
    TheoryValidationError,
    build_theory_artifacts,
    capacity_metrics,
    deterministic_rank_order,
    diagnose_cascading_context_edit,
    n_B,
    validate_quality_bounds,
    verify_exact_recovery_trace,
)


def test_capacity_equations_exact_fit_and_tail_rate():
    assert n_B(256, 16) == 64
    assert R_B(256, 16) == pytest.approx(4.0)
    assert R_effective(256, 64, 64) == pytest.approx(2.0)
    metrics = capacity_metrics(
        256, 16, observed_n_forced=64, observed_n_tail=64
    )
    assert metrics.theoretical_n_B == 64
    assert metrics.R_B_bits_per_forced_token == pytest.approx(4.0)
    assert metrics.rate_upper_bound_bits_per_forced_token == pytest.approx(4.0)
    assert metrics.rate_bound_holds is True
    assert metrics.observed_forced_count_feasible is True
    assert metrics.code_space_slack_bits == pytest.approx(0.0)
    assert metrics.literal_padding_bits == 0
    assert metrics.finite_padding_case == "exact_symbol_fit"


def test_finite_padding_nondivisible_nonbinary_and_empty_cases():
    binary = capacity_metrics(8, 8, observed_n_forced=3, observed_n_tail=0)
    assert binary.theoretical_n_B == 3
    assert binary.R_B_bits_per_forced_token == pytest.approx(8 / 3)
    assert binary.R_B_bits_per_forced_token <= math.log2(8)
    assert binary.literal_padding_bits == 1
    assert binary.code_space_utilization == pytest.approx(0.5)
    assert binary.finite_padding_case == "finite_binary_padding"

    ternary = capacity_metrics(1, 3)
    assert ternary.theoretical_n_B == 1
    assert ternary.literal_padding_bits is None
    assert ternary.code_space_slack_bits == pytest.approx(math.log2(3) - 1)
    assert ternary.code_space_utilization == pytest.approx(2 / 3)
    assert ternary.finite_padding_case == "non_power_of_two_code_space_slack"

    fractional_entropy = capacity_metrics(1.5, 4)
    assert fractional_entropy.literal_padding_bits is None

    empty = capacity_metrics(0, 16, observed_n_forced=0, observed_n_tail=0)
    assert n_B(0, 16) == 0
    assert R_B(0, 16) == 0.0
    assert R_effective(0, 0, 0) == 0.0
    assert empty.code_space_utilization == 1.0
    assert empty.finite_padding_case == "empty_payload"


def test_capacity_validation_rejects_invalid_domains_and_flags_impossible_count():
    with pytest.raises(TheoryValidationError):
        n_B(8, 1)
    with pytest.raises(TheoryValidationError):
        n_B(-1, 8)
    with pytest.raises(TheoryValidationError):
        R_effective(8, 0, 0)
    metrics = capacity_metrics(8, 8, observed_n_forced=2)
    assert metrics.observed_forced_count_feasible is False
    assert metrics.observed_rate_bits_per_forced_token == pytest.approx(4.0)


def test_empirical_quality_and_same_context_bounds():
    realized = [-0.5, -0.6]
    greedy = [-0.1, -0.2]
    rank_b = [-1.0, -1.2]
    metrics = validate_quality_bounds(
        realized,
        greedy_log_probabilities=greedy,
        rank_B_log_probabilities=rank_b,
        realized_ranks=[2, 3],
        alphabet_size=4,
    )
    assert Q_B(realized) == pytest.approx(0.55)
    assert metrics.Q_B_nats_per_forced_token == pytest.approx(0.55)
    assert metrics.Q_greedy_nats_per_forced_token == pytest.approx(0.15)
    assert metrics.Q_rank_B_nats_per_forced_token == pytest.approx(1.1)
    assert Delta_B(realized, greedy) == pytest.approx(0.4)
    assert metrics.Delta_B_nats_per_forced_token == pytest.approx(0.4)
    assert metrics.greedy_lower_bound_holds_per_context is True
    assert metrics.rank_B_upper_bound_holds_per_context is True
    assert metrics.rank_range_holds is True
    assert metrics.quality_status == "validated"


def test_quality_missing_endpoints_is_unavailable_not_imputed_and_violations_fail():
    unavailable = validate_quality_bounds([-0.5, -0.7])
    assert unavailable.Q_B_nats_per_forced_token == pytest.approx(0.6)
    assert unavailable.Q_greedy_nats_per_forced_token is None
    assert unavailable.Q_rank_B_nats_per_forced_token is None
    assert unavailable.Delta_B_nats_per_forced_token is None
    assert unavailable.all_available_checks_hold is None
    assert unavailable.quality_status == "not_evaluable_missing_endpoint_probabilities"

    failed = validate_quality_bounds(
        [-0.5],
        greedy_log_probabilities=[-0.8],
        rank_B_log_probabilities=[-0.4],
        realized_ranks=[5],
        alphabet_size=4,
    )
    assert failed.greedy_lower_bound_holds_per_context is False
    assert failed.rank_B_upper_bound_holds_per_context is False
    assert failed.rank_range_holds is False
    assert failed.quality_status == "failed"


def _config(model="model-sha"):
    return {
        "model_identity": model,
        "tokenizer_identity": "tokenizer-sha",
        "inference_config_identity": "config-sha",
        "prompt_token_ids": [1, 2, 3],
        "admissible_token_set_identity": "filter-sha",
        "tie_break_rule": TOKEN_ID_TIE_BREAK_RULE,
    }


def _exact_traces():
    encoder = [
        {
            "context_token_ids": [1, 2, 3],
            "ranked_token_ids": [4, 7, 9],
            "selected_token_id": 7,
            "expected_rank": 2,
        },
        {
            "context_token_ids": [1, 2, 3, 7],
            "ranked_token_ids": [9, 4, 7],
            "selected_token_id": 9,
            "expected_rank": 1,
        },
    ]
    decoder = [
        {
            "context_token_ids": [1, 2, 3],
            "ranked_token_ids": [4, 7, 9],
            "observed_token_id": 7,
        },
        {
            "context_token_ids": [1, 2, 3, 7],
            "ranked_token_ids": [9, 4, 7],
            "observed_token_id": 9,
        },
    ]
    return encoder, decoder


def test_token_id_tie_break_and_exact_recovery_proposition():
    assert deterministic_rank_order([0.1, 2.0, 2.0, -1.0, 0.1]) == [1, 2, 0, 4, 3]
    assert deterministic_rank_order([1.0, 1.0, 2.0], [1, 0]) == [0, 1]
    encoder, decoder = _exact_traces()
    report = verify_exact_recovery_trace(encoder, decoder, _config(), _config())
    assert report["configurations_identical"] is True
    assert report["supported_tie_break_identical"] is True
    assert report["context_token_ids_identical"] is True
    assert report["forced_token_ids_identical"] is True
    assert report["guarantee_conditions_satisfied"] is True
    assert report["recovered_ranks_equal_expected"] is True
    assert report["proposition_confirmed"] is True
    assert report["proposition_violation"] is False


def test_coincidental_recovery_does_not_validate_mismatched_configuration():
    encoder, decoder = _exact_traces()
    report = verify_exact_recovery_trace(
        encoder, decoder, _config(), _config(model="different-model-sha")
    )
    assert report["observed_exact_recovery"] is True
    assert report["configurations_identical"] is False
    assert report["guarantee_conditions_satisfied"] is False
    assert report["proposition_confirmed"] is False
    assert report["proposition_violation"] is False


def test_context_edit_diagnostic_records_autoregressive_cascade():
    reference = [
        {
            "context_token_ids": [0, 1],
            "ranked_token_ids": [10, 11, 12],
            "observed_token_id": 11,
        },
        {
            "context_token_ids": [0, 1, 11],
            "ranked_token_ids": [20, 21, 22],
            "observed_token_id": 21,
        },
        {
            "context_token_ids": [0, 1, 11, 21],
            "ranked_token_ids": [30, 31, 32],
            "observed_token_id": 31,
        },
    ]
    edited = [
        {
            "context_token_ids": [0, 99, 1],
            "ranked_token_ids": [11, 10, 12],
            "observed_token_id": 11,
        },
        {
            "context_token_ids": [0, 99, 1, 11],
            "ranked_token_ids": [21, 20, 22],
            "observed_token_id": 21,
        },
        {
            "context_token_ids": [0, 99, 1, 11, 21],
            "ranked_token_ids": [31, 30, 32],
            "observed_token_id": 31,
        },
    ]
    report = diagnose_cascading_context_edit(reference, edited)
    assert report["first_context_divergence_step"] == 0
    assert report["first_context_token_difference"] == 1
    assert report["first_rank_order_divergence_step"] == 0
    assert report["first_rank_error_step"] == 0
    assert report["rank_error_count"] == 3
    assert report["post_initial_rank_error_count"] == 2
    assert report["cascade_observed"] is True
    assert report["cascade_status"] == "context_edit_with_downstream_cascade"


def _saved_trial_record():
    encoder, decoder = _exact_traces()
    reference = [
        {
            "context_token_ids": [1],
            "ranked_token_ids": [4, 5, 6],
            "observed_token_id": 5,
        },
        {
            "context_token_ids": [1, 5],
            "ranked_token_ids": [7, 8, 9],
            "observed_token_id": 8,
        },
    ]
    edited = [
        {
            "context_token_ids": [1, 99],
            "ranked_token_ids": [5, 4, 6],
            "observed_token_id": 5,
        },
        {
            "context_token_ids": [1, 99, 5],
            "ranked_token_ids": [8, 7, 9],
            "observed_token_id": 8,
        },
    ]
    return {
        "record_type": "rankcloak_trial",
        "trial_id": "trial-001",
        "model_id": "fake-model",
        "protocol_variant": "nonseg_ascii_b8",
        "payload_name": "payload-001",
        "artifact_bit_length": 8,
        "representation": {
            "name": "ascii_b8",
            "expected_ranks": [1, 2, 3],
            "metadata": {
                "alphabet_size": 8,
                "original_byte_length": 1,
                "padding_bits": 1,
            },
        },
        "forced_token_count": 3,
        "tail_token_count": 2,
        "segments": [
            {
                "expected_ranks": [1, 2, 3],
                "forced_log_probabilities": [-0.1, -0.5, -0.8],
                "greedy_log_probabilities": [-0.1, -0.2, -0.2],
                "rank_B_log_probabilities": [-1.5, -1.5, -1.5],
            }
        ],
        "saved_token_id_replay": {
            "recovered_ranks": [1, 2, 3],
            "exact_recovery": True,
        },
        "encoder_trace": encoder,
        "decoder_trace": decoder,
        "encoder_configuration": _config(),
        "decoder_configuration": _config(),
        "reference_trace": reference,
        "edited_trace": edited,
    }


def test_artifact_builder_emits_data_only_validation_and_plot_tables(tmp_path):
    source = tmp_path / "trials.jsonl"
    source.write_text(json.dumps(_saved_trial_record()) + "\n", encoding="utf-8")
    output = tmp_path / "theory"
    artifacts = build_theory_artifacts([source], output)
    assert artifacts.summary["input_record_count"] == 1
    assert artifacts.summary["capacity_evaluable_count"] == 1
    assert artifacts.summary["quality_fully_bound_validated_count"] == 1
    assert artifacts.summary["exact_proposition_confirmed_count"] == 1
    assert artifacts.summary["cascade_evaluable_count"] == 1

    for filename in OUTPUT_FILENAMES.values():
        assert (output / filename).is_file()
    with (output / OUTPUT_FILENAMES["capacity_validation"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        capacity = list(csv.DictReader(handle))
    assert capacity[0]["H_bits"] == "8.0"
    assert capacity[0]["H_source"] == "codec_metadata_original_byte_length"
    assert capacity[0]["theoretical_n_B"] == "3"
    assert capacity[0]["R_effective_bits_per_forced_plus_tail_token"] == "1.6"

    manifest = json.loads(
        (output / OUTPUT_FILENAMES["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["missing_result_policy"].startswith("no imputation")
    assert manifest["inputs"][0]["sha256"]
    assert {table["row_count"] for table in manifest["tables"]} == {1}


def test_csv_trial_mean_does_not_fabricate_endpoint_bounds(tmp_path):
    source = tmp_path / "means.csv"
    source.write_text(
        "trial_id,model_id,protocol_variant,alphabet_size,H_bits,forced_token_count,tail_token_count,mean_token_log_probability\n"
        "mean-1,m,p,16,16,4,0,-0.75\n",
        encoding="utf-8",
    )
    output = tmp_path / "tables"
    build_theory_artifacts([source], output)
    with (output / OUTPUT_FILENAMES["quality_validation"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["Q_B_nats_per_forced_token"] == "0.75"
    assert row["quality_evidence_level"] == "trial_mean_only"
    assert row["Q_greedy_nats_per_forced_token"] == ""
    assert row["Q_rank_B_nats_per_forced_token"] == ""
    assert row["Delta_B_nats_per_forced_token"] == ""
    assert row["quality_status"] == "not_evaluable_missing_endpoint_probabilities"


def test_saved_aggregate_endpoint_means_are_not_promoted_to_per_context_bounds(tmp_path):
    source = tmp_path / "aggregate-means.csv"
    source.write_text(
        "trial_id,alphabet_size,H_bits,forced_token_count,tail_token_count,"
        "mean_token_log_probability,mean_greedy_log_probability,mean_rank_B_log_probability\n"
        "mean-2,16,16,4,0,-0.75,-0.25,-1.25\n",
        encoding="utf-8",
    )
    output = tmp_path / "aggregate-tables"
    artifacts = build_theory_artifacts([source], output)
    with (output / OUTPUT_FILENAMES["quality_validation"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["forced_context_count"] == "4"
    assert row["greedy_lower_bound_holds_per_context"] == ""
    assert row["rank_B_upper_bound_holds_per_context"] == ""
    assert row["greedy_lower_bound_holds_in_expectation"] == "True"
    assert row["rank_B_upper_bound_holds_in_expectation"] == "True"
    assert row["quality_status"] == (
        "aggregate_means_only_per_context_bounds_not_evaluable"
    )
    assert artifacts.summary["quality_fully_bound_validated_count"] == 0


def test_cli_consumes_saved_jsonl_and_reports_outputs(tmp_path):
    source = tmp_path / "trials.jsonl"
    source.write_text(json.dumps(_saved_trial_record()) + "\n", encoding="utf-8")
    output = tmp_path / "cli-output"
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_revision_theory.py"),
            "--trials",
            str(source),
            "--output-dir",
            str(output),
        ],
        cwd=str(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "completed"
    assert report["summary"]["input_record_count"] == 1
    assert report["empirical_missing_value_policy"] == "no imputation"

