"""Metric helpers for RankCloak experiments."""

from __future__ import annotations

from typing import List, Sequence

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

