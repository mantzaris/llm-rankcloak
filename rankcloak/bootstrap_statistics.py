"""Deterministic bootstrap helpers for paper-oriented summaries."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def numeric_values(values: Iterable[object]) -> np.ndarray:
    cleaned = []
    for value in values:
        if value is None:
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.asarray(cleaned, dtype=np.float64)


def bootstrap_mean_ci(
    values: Iterable[object],
    n_resamples: int = 1000,
    seed: int = 20260521,
) -> dict:
    """Return deterministic mean, standard deviation, and bootstrap CI."""

    array = numeric_values(values)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "bootstrap_ci_low_95": None,
            "bootstrap_ci_high_95": None,
        }
    mean_value = float(np.mean(array))
    if array.size == 1:
        return {
            "n": 1,
            "mean": mean_value,
            "standard_deviation": 0.0,
            "bootstrap_ci_low_95": mean_value,
            "bootstrap_ci_high_95": mean_value,
        }
    rng = np.random.default_rng(int(seed))
    bootstrap_means = []
    for _ in range(int(n_resamples)):
        sample = rng.choice(array, size=array.size, replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n": int(array.size),
        "mean": mean_value,
        "standard_deviation": float(np.std(array, ddof=1)),
        "bootstrap_ci_low_95": float(min(low, mean_value)),
        "bootstrap_ci_high_95": float(max(high, mean_value)),
    }


def bootstrap_difference_ci(
    group_a: Iterable[object],
    group_b: Iterable[object],
    n_resamples: int = 1000,
    seed: int = 20260521,
) -> dict:
    """Return deterministic bootstrap CI for mean(group_b) - mean(group_a)."""

    array_a = numeric_values(group_a)
    array_b = numeric_values(group_b)
    if array_a.size == 0 or array_b.size == 0:
        return {
            "n_a": int(array_a.size),
            "n_b": int(array_b.size),
            "mean_a": None if array_a.size == 0 else float(np.mean(array_a)),
            "mean_b": None if array_b.size == 0 else float(np.mean(array_b)),
            "difference_b_minus_a": None,
            "ratio_b_over_a": None,
            "bootstrap_ci_low_95": None,
            "bootstrap_ci_high_95": None,
        }
    mean_a = float(np.mean(array_a))
    mean_b = float(np.mean(array_b))
    difference = mean_b - mean_a
    ratio = None if mean_a == 0 else mean_b / mean_a
    if array_a.size == 1 and array_b.size == 1:
        low = high = difference
    else:
        rng = np.random.default_rng(int(seed))
        differences = []
        for _ in range(int(n_resamples)):
            sample_a = rng.choice(array_a, size=array_a.size, replace=True)
            sample_b = rng.choice(array_b, size=array_b.size, replace=True)
            differences.append(float(np.mean(sample_b) - np.mean(sample_a)))
        low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "n_a": int(array_a.size),
        "n_b": int(array_b.size),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_b_minus_a": difference,
        "ratio_b_over_a": ratio,
        "bootstrap_ci_low_95": float(min(low, difference)),
        "bootstrap_ci_high_95": float(max(high, difference)),
    }
