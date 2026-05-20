from rankcloak.rank_codec import codec_roundtrip_rows
from rankcloak.schemas import (
    CODEC_ROUNDTRIP_COLUMNS,
    COVER_TEXT_FEATURE_COLUMNS,
    SEGMENTED_PROTOCOL_TRIAL_COLUMNS,
    STEGOTEXT_RECOVERY_COLUMNS,
)
from rankcloak.synthetic_payloads import generate_synthetic_payloads


def test_schema_constants_include_required_codec_columns():
    for column in [
        "payload_name",
        "payload_kind",
        "payload_byte_length",
        "alphabet_size",
        "rank_count",
        "exact_roundtrip",
        "notes",
    ]:
        assert column in CODEC_ROUNDTRIP_COLUMNS


def test_schema_constants_include_required_stegotext_columns():
    for column in [
        "payload_name",
        "payload_kind",
        "payload_byte_length",
        "payload_slice_description",
        "encoding_name",
        "alphabet_size",
        "cover_prompt_name",
        "rank_count",
        "generated_token_count",
        "generated_character_count",
        "exact_recovery",
        "generation_seconds",
        "recovery_seconds",
        "mean_generated_rank",
        "median_generated_rank",
        "p95_generated_rank",
        "max_generated_rank",
        "fraction_generated_rank_le_16",
        "fraction_generated_rank_le_64",
        "model_repo_id",
        "model_filename",
        "model_path_relative",
        "notes",
    ]:
        assert column in STEGOTEXT_RECOVERY_COLUMNS


def test_feature_schema_has_rankcloak_and_baseline_context_columns():
    for column in [
        "source_type",
        "source_id",
        "cover_prompt_name",
        "prompt_family",
        "prompt_length_characters",
        "prompt_length_tokens",
        "character_count",
    ]:
        assert column in COVER_TEXT_FEATURE_COLUMNS


def test_segmented_protocol_schema_has_required_columns():
    for column in [
        "trial_id",
        "payload_name",
        "payload_kind",
        "payload_codec_name",
        "condition_name",
        "segment_size",
        "segment_count",
        "message_count",
        "topic_schedule_name",
        "natural_tail_tokens_per_message",
        "total_forced_rank_count",
        "total_generated_token_count",
        "exact_recovery",
        "mean_token_log_probability",
        "mean_generated_rank",
        "p95_generated_rank",
        "model_repo_id",
        "model_filename",
        "model_path_relative",
        "notes",
    ]:
        assert column in SEGMENTED_PROTOCOL_TRIAL_COLUMNS


def test_codec_roundtrip_rows_match_schema_and_pass():
    payload = generate_synthetic_payloads()[:1]
    rows = codec_roundtrip_rows(payload, [2, 16, 64])
    assert rows
    for row in rows:
        assert set(CODEC_ROUNDTRIP_COLUMNS).issubset(row.keys())
        assert row["exact_roundtrip"] is True
