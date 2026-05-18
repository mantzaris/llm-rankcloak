"""Result schema constants for RankCloak output files."""

from __future__ import annotations


CODEC_ROUNDTRIP_COLUMNS = [
    "payload_name",
    "payload_kind",
    "payload_byte_length",
    "encoding_name",
    "alphabet_size",
    "rank_count",
    "exact_roundtrip",
    "notes",
]

STEGOTEXT_RECOVERY_COLUMNS = [
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
]

BASELINE_COVER_COLUMNS = [
    "cover_prompt_name",
    "baseline_mode",
    "generated_text",
    "generated_token_count",
    "generated_character_count",
    "generation_seconds",
    "model_repo_id",
    "model_filename",
    "notes",
]

COVER_TEXT_FEATURE_COLUMNS = [
    "source_type",
    "source_id",
    "payload_name",
    "cover_prompt_name",
    "prompt_family",
    "prompt_length_characters",
    "prompt_length_tokens",
    "alphabet_size",
    "baseline_mode",
    "character_count",
    "token_count",
    "line_count",
    "whitespace_fraction",
    "punctuation_fraction",
    "digit_fraction",
    "alphabetic_fraction",
    "unique_token_fraction",
    "repeated_token_fraction",
    "mean_token_log_probability",
    "median_token_log_probability",
    "mean_generated_rank",
    "p95_generated_rank",
    "fraction_rank_le_16",
    "fraction_rank_le_64",
]
