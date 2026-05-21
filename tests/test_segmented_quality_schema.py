from rankcloak.schemas import SEGMENTED_QUALITY_FEATURE_COLUMNS, SEGMENTED_QUALITY_TRIAL_COLUMNS


def test_segmented_quality_trial_schema_has_required_fields():
    for column in [
        "trial_id",
        "payload_name",
        "payload_codec_name",
        "condition_name",
        "tail_policy",
        "token_filter_name",
        "total_forced_prefix_token_count",
        "total_full_message_token_count",
        "forced_prefix_mean_log_probability_mean",
        "full_message_mean_log_probability_mean",
        "forced_prefix_repetition_mean",
        "full_message_repetition_mean",
        "exact_recovery",
    ]:
        assert column in SEGMENTED_QUALITY_TRIAL_COLUMNS


def test_segmented_quality_feature_schema_has_required_fields():
    for column in [
        "source_type",
        "trial_id",
        "condition_name",
        "prompt_name",
        "token_filter_name",
        "tail_policy",
        "token_count",
        "mean_token_log_probability",
        "repeated_token_fraction",
        "artifact_count_total",
    ]:
        assert column in SEGMENTED_QUALITY_FEATURE_COLUMNS
