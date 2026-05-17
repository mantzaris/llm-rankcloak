"""Metric helpers for RankCloak experiments."""

from __future__ import annotations

import string
from typing import List, Optional, Sequence

import numpy as np


def summarize_rank_sequence(payload_name: str, ranks: Sequence[int]) -> dict:
    """Summarize 1-indexed ranks for direct subword encoding."""

    if not ranks:
        return {
            "payload_name": payload_name,
            "rank_count": 0,
            "mean_rank": None,
            "median_rank": None,
            "max_rank": None,
            "p90_rank": None,
            "p95_rank": None,
            "fraction_rank_le_1": None,
            "fraction_rank_le_5": None,
            "fraction_rank_le_16": None,
            "fraction_rank_le_64": None,
        }
    values = np.asarray(list(map(int, ranks)), dtype=np.float64)
    return {
        "payload_name": payload_name,
        "rank_count": int(values.size),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
        "max_rank": int(np.max(values)),
        "p90_rank": float(np.percentile(values, 90)),
        "p95_rank": float(np.percentile(values, 95)),
        "fraction_rank_le_1": float(np.mean(values <= 1)),
        "fraction_rank_le_5": float(np.mean(values <= 5)),
        "fraction_rank_le_16": float(np.mean(values <= 16)),
        "fraction_rank_le_64": float(np.mean(values <= 64)),
    }


def summarize_many_rank_sequences(items: Sequence[dict]) -> List[dict]:
    rows = []
    for item in items:
        rows.append(summarize_rank_sequence(item["payload_name"], item.get("ranks", [])))
    return rows


def safe_fraction(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def summarize_optional_ranks(ranks: Optional[Sequence[int]]) -> dict:
    if not ranks:
        return {
            "mean_generated_rank": None,
            "median_generated_rank": None,
            "p95_generated_rank": None,
            "max_generated_rank": None,
            "fraction_generated_rank_le_16": None,
            "fraction_generated_rank_le_64": None,
        }
    values = np.asarray(list(map(int, ranks)), dtype=np.float64)
    return {
        "mean_generated_rank": float(np.mean(values)),
        "median_generated_rank": float(np.median(values)),
        "p95_generated_rank": float(np.percentile(values, 95)),
        "max_generated_rank": int(np.max(values)),
        "fraction_generated_rank_le_16": float(np.mean(values <= 16)),
        "fraction_generated_rank_le_64": float(np.mean(values <= 64)),
    }


def summarize_optional_log_probabilities(log_probabilities: Optional[Sequence[float]]) -> dict:
    if not log_probabilities:
        return {
            "mean_token_log_probability": None,
            "median_token_log_probability": None,
        }
    values = np.asarray(list(map(float, log_probabilities)), dtype=np.float64)
    return {
        "mean_token_log_probability": float(np.mean(values)),
        "median_token_log_probability": float(np.median(values)),
    }


def extract_text_features(
    text: str,
    token_ids: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
    token_log_probabilities: Optional[Sequence[float]] = None,
) -> dict:
    """Extract lightweight plausibility/detectability features for cover text."""

    text = text or ""
    character_count = len(text)
    token_count = len(token_ids) if token_ids is not None else None
    whitespace_count = sum(1 for character in text if character.isspace())
    punctuation_count = sum(1 for character in text if character in string.punctuation)
    digit_count = sum(1 for character in text if character.isdigit())
    alphabetic_count = sum(1 for character in text if character.isalpha())
    line_count = 0 if character_count == 0 else text.count("\n") + 1

    unique_token_fraction = None
    repeated_token_fraction = None
    if token_ids:
        unique_token_fraction = len(set(map(int, token_ids))) / float(len(token_ids))
        repeated_token_fraction = 1.0 - unique_token_fraction

    row = {
        "character_count": character_count,
        "token_count": token_count,
        "line_count": line_count,
        "whitespace_fraction": safe_fraction(whitespace_count, character_count),
        "punctuation_fraction": safe_fraction(punctuation_count, character_count),
        "digit_fraction": safe_fraction(digit_count, character_count),
        "alphabetic_fraction": safe_fraction(alphabetic_count, character_count),
        "unique_token_fraction": unique_token_fraction,
        "repeated_token_fraction": repeated_token_fraction,
    }
    row.update(summarize_optional_log_probabilities(token_log_probabilities))
    rank_summary = summarize_optional_ranks(ranks)
    row.update(
        {
            "mean_generated_rank": rank_summary["mean_generated_rank"],
            "p95_generated_rank": rank_summary["p95_generated_rank"],
            "fraction_rank_le_16": rank_summary["fraction_generated_rank_le_16"],
            "fraction_rank_le_64": rank_summary["fraction_generated_rank_le_64"],
        }
    )
    return row
