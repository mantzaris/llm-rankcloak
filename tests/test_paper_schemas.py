from rankcloak.schemas import (
    DETECTOR_BASELINE_COLUMNS,
    DETECTOR_DATASET_COLUMNS,
    EFFECT_SIZE_SUMMARY_COLUMNS,
    PAPER_CODEC_COMPARISON_COLUMNS,
    PAPER_COVER_TEXT_FEATURE_COLUMNS,
    PAPER_PAYLOAD_COLUMNS,
    PAPER_RANK_PRESSURE_COLUMNS,
    PAPER_SEGMENTED_TRIAL_COLUMNS,
    PAPER_STEGOTEXT_TRIAL_COLUMNS,
    STATISTICAL_SUMMARY_COLUMNS,
)


def test_paper_payload_schema_columns_exist():
    for column in [
        "payload_name",
        "payload_class",
        "artifact_text_character_length",
        "artifact_bit_length_if_known",
        "representation_hint",
        "is_hex_like",
    ]:
        assert column in PAPER_PAYLOAD_COLUMNS


def test_paper_trial_schema_columns_exist():
    for column in [
        "trial_id",
        "protocol_variant",
        "payload_name",
        "representation_name",
        "exact_recovery",
        "mean_token_log_probability",
        "artifact_count_total",
    ]:
        assert column in PAPER_STEGOTEXT_TRIAL_COLUMNS


def test_paper_segmented_schema_columns_exist():
    for column in [
        "trial_id",
        "protocol_variant",
        "leadin_policy",
        "tail_policy",
        "token_filter_name",
        "forced_prefix_mean_log_probability_mean",
        "full_message_mean_log_probability_mean",
    ]:
        assert column in PAPER_SEGMENTED_TRIAL_COLUMNS


def test_paper_analysis_schema_columns_exist():
    for columns in [
        PAPER_RANK_PRESSURE_COLUMNS,
        PAPER_CODEC_COMPARISON_COLUMNS,
        PAPER_COVER_TEXT_FEATURE_COLUMNS,
        DETECTOR_DATASET_COLUMNS,
        DETECTOR_BASELINE_COLUMNS,
        STATISTICAL_SUMMARY_COLUMNS,
        EFFECT_SIZE_SUMMARY_COLUMNS,
    ]:
        assert "notes" in columns


def test_paper_analysis_status_columns_exist():
    for columns in [
        DETECTOR_BASELINE_COLUMNS,
        STATISTICAL_SUMMARY_COLUMNS,
        EFFECT_SIZE_SUMMARY_COLUMNS,
    ]:
        assert "status" in columns
