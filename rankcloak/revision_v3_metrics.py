"""Validation-frozen detector metrics for revision V3.

Low-FPR thresholds are selected from validation labels only.  The partial AUC
uses an empirical false-positive-count envelope rather than interpolation
between score values.  Confidence intervals resample complete payload-linked
deduplication clusters.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "rankcloak-revision-v3-detector-metrics-v1"
DEFAULT_SEED = 20260831
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
PARTIAL_AUC_MAX_FPR = 0.01
LOW_FPR_TARGETS = (0.01, 0.001)


class RevisionV3MetricError(ValueError):
    """Raised when detector scores cannot support a declared estimand."""


def _validate_binary_scores(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise RevisionV3MetricError("labels and scores must be aligned non-empty vectors")
    if not np.isin(y, [0, 1]).all() or y.min() == y.max():
        raise RevisionV3MetricError("both binary labels are required")
    if not np.isfinite(s).all():
        raise RevisionV3MetricError("all scores must be finite")
    return y, s


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware Mann-Whitney ROC AUC."""

    y, s = _validate_binary_scores(labels, scores)
    positive_count = int(np.count_nonzero(y == 1))
    negative_count = int(np.count_nonzero(y == 0))
    order = np.argsort(s, kind="mergesort")
    ordered_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        # Average one-based rank for a tied score block.
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(mann_whitney / (positive_count * negative_count))


def _threshold_states(
    labels: np.ndarray, scores: np.ndarray
) -> list[Tuple[float, int, int]]:
    """Return achievable (threshold, FP, TP) states for score >= threshold."""

    maximum = float(np.max(scores))
    states: list[Tuple[float, int, int]] = [
        (float(np.nextafter(maximum, math.inf)), 0, 0)
    ]
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    fp = 0
    tp = 0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        block = ordered_labels[start:stop]
        fp += int(np.count_nonzero(block == 0))
        tp += int(np.count_nonzero(block == 1))
        states.append((float(ordered_scores[start]), fp, tp))
        start = stop
    return states


def select_validation_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    target_fpr: float,
    minimum_negative_count: Optional[int] = None,
) -> Mapping[str, object]:
    """Select the highest-validation-TPR threshold within an exact FP budget."""

    y, s = _validate_binary_scores(labels, scores)
    target = float(target_fpr)
    if not 0.0 < target < 1.0:
        raise RevisionV3MetricError("target_fpr must be in (0, 1)")
    negatives = int(np.count_nonzero(y == 0))
    positives = int(np.count_nonzero(y == 1))
    required = int(minimum_negative_count or math.ceil(1.0 / target))
    if negatives < required:
        return {
            "available": False,
            "target_fpr": target,
            "validation_negative_count": negatives,
            "validation_positive_count": positives,
            "minimum_negative_count": required,
            "reason": "validation negatives do not support the requested empirical resolution",
            "threshold": None,
        }
    allowed_fp = int(math.floor(target * negatives + 1e-12))
    feasible = [state for state in _threshold_states(y, s) if state[1] <= allowed_fp]
    threshold, fp, tp = max(
        feasible,
        key=lambda state: (state[2], -state[1], state[0]),
    )
    return {
        "available": True,
        "target_fpr": target,
        "validation_negative_count": negatives,
        "validation_positive_count": positives,
        "minimum_negative_count": required,
        "allowed_false_positives": allowed_fp,
        "threshold": float(threshold),
        "validation_false_positives": int(fp),
        "validation_true_positives": int(tp),
        "validation_empirical_fpr": float(fp / negatives),
        "validation_empirical_tpr": float(tp / positives),
        "selection_rule": (
            "maximize validation TPR among observed score thresholds with exact "
            "validation FP count at or below floor(target_fpr*n_negative); break "
            "ties by fewer FP then higher threshold"
        ),
    }


def empirical_partial_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    max_fpr: float = PARTIAL_AUC_MAX_FPR,
) -> float:
    """Normalized pAUC of the exact empirical FP-budget step envelope.

    For each integer false-positive budget k, the ordinate is the greatest TPR
    available at an observed score threshold with FP <= k.  That ordinate is
    constant on [k/n_negative, (k+1)/n_negative).  The area is divided by
    max_fpr, so the result ranges from zero to one.  No scores or ROC vertices
    are interpolated.
    """

    y, s = _validate_binary_scores(labels, scores)
    alpha = float(max_fpr)
    if not 0.0 < alpha <= 1.0:
        raise RevisionV3MetricError("max_fpr must be in (0, 1]")
    negative_count = int(np.count_nonzero(y == 0))
    positive_count = int(np.count_nonzero(y == 1))
    states = _threshold_states(y, s)
    maximum_budget = int(math.floor(alpha * negative_count + 1e-12))
    best_tpr = np.zeros(maximum_budget + 1, dtype=np.float64)
    for budget in range(maximum_budget + 1):
        best_tp = max(tp for _, fp, tp in states if fp <= budget)
        best_tpr[budget] = float(best_tp / positive_count)
    area = 0.0
    position = 0.0
    budget = 0
    step = 1.0 / negative_count
    while position < alpha - 1e-15:
        width = min(step, alpha - position)
        safe_budget = min(budget, maximum_budget)
        area += float(best_tpr[safe_budget]) * width
        position += width
        budget += 1
    return float(area / alpha)


def frozen_threshold_counts(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Mapping[str, object]:
    y, s = _validate_binary_scores(labels, scores)
    predicted = s >= float(threshold)
    tp = int(np.count_nonzero((y == 1) & predicted))
    fp = int(np.count_nonzero((y == 0) & predicted))
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    return {
        "threshold": float(threshold),
        "positive_count": positives,
        "negative_count": negatives,
        "true_positives": tp,
        "false_positives": fp,
        "tpr": float(tp / positives),
        "fpr": float(fp / negatives),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Two-sided Wilson score interval for an empirical proportion."""

    n = int(trials)
    k = int(successes)
    if n <= 0 or not 0 <= k <= n:
        raise RevisionV3MetricError("Wilson interval requires 0 <= successes <= trials")
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _stable_seed(seed: int, *parts: object) -> int:
    material = "\x1f".join([str(int(seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def _bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: Mapping[float, float],
    n_resamples: int,
    seed: int,
) -> Mapping[str, list[float]]:
    unique_groups = np.asarray(sorted(set(groups.astype(str))), dtype=object)
    if len(unique_groups) < 2:
        raise RevisionV3MetricError("grouped bootstrap requires at least two groups")
    positions = {
        group: np.flatnonzero(groups.astype(str) == group) for group in unique_groups
    }
    names = ["roc_auc", "partial_auc_fpr_0_01"]
    for target in thresholds:
        suffix = "0_01" if math.isclose(target, 0.01) else "0_001"
        names.extend(["tpr_at_fpr_{}".format(suffix), "fpr_at_threshold_{}".format(suffix)])
    samples: Dict[str, list[float]] = {name: [] for name in names}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_resamples)):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([positions[group] for group in selected])
        y = labels[indices]
        s = scores[indices]
        if len(np.unique(y)) != 2:
            continue
        samples["roc_auc"].append(roc_auc(y, s))
        samples["partial_auc_fpr_0_01"].append(empirical_partial_auc(y, s, 0.01))
        for target, threshold in thresholds.items():
            counts = frozen_threshold_counts(y, s, threshold)
            suffix = "0_01" if math.isclose(target, 0.01) else "0_001"
            samples["tpr_at_fpr_{}".format(suffix)].append(float(counts["tpr"]))
            samples["fpr_at_threshold_{}".format(suffix)].append(float(counts["fpr"]))
    return samples


def evaluate_validation_frozen_detector(
    validation_labels: Sequence[int],
    validation_scores: Sequence[float],
    test_labels: Sequence[int],
    test_scores: Sequence[float],
    test_groups: Sequence[object],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Mapping[str, object]:
    """Evaluate one detector without consulting test labels for thresholds."""

    val_y, val_s = _validate_binary_scores(validation_labels, validation_scores)
    test_y, test_s = _validate_binary_scores(test_labels, test_scores)
    groups = np.asarray(list(map(str, test_groups)), dtype=object)
    if len(groups) != len(test_y):
        raise RevisionV3MetricError("test_groups must align with test scores")
    negative_count = int(np.count_nonzero(test_y == 0))
    positive_count = int(np.count_nonzero(test_y == 1))
    point: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "unit_of_analysis": "detector corpus observation",
        "uncertainty_unit": "dedup_cluster_id",
        "validation_rows": int(len(val_y)),
        "validation_negative_count": int(np.count_nonzero(val_y == 0)),
        "validation_positive_count": int(np.count_nonzero(val_y == 1)),
        "test_rows": int(len(test_y)),
        "test_negative_count": negative_count,
        "test_positive_count": positive_count,
        "test_group_count": int(len(set(groups))),
        "roc_auc": roc_auc(test_y, test_s),
        "partial_auc_fpr_0_01": empirical_partial_auc(test_y, test_s, 0.01),
        "partial_auc_normalization": "raw empirical step area divided by 0.01",
        "partial_auc_interpolation": "none; exact FP-budget step envelope",
        "bootstrap_resamples_requested": int(bootstrap_resamples),
    }
    selected_thresholds: Dict[float, float] = {}
    selections: Dict[str, object] = {}
    for target in LOW_FPR_TARGETS:
        suffix = "0_01" if math.isclose(target, 0.01) else "0_001"
        required = int(math.ceil(1.0 / target))
        selection = dict(
            select_validation_threshold(
                val_y, val_s, target, minimum_negative_count=required
            )
        )
        if negative_count < required:
            selection = {
                **selection,
                "available": False,
                "test_negative_count": negative_count,
                "reason": "test negatives do not support the requested empirical resolution",
            }
        selections["fpr_{}".format(suffix)] = selection
        if bool(selection.get("available")):
            threshold = float(selection["threshold"])
            selected_thresholds[target] = threshold
            counts = frozen_threshold_counts(test_y, test_s, threshold)
            point["threshold_at_fpr_{}".format(suffix)] = threshold
            point["tpr_at_fpr_{}".format(suffix)] = float(counts["tpr"])
            point["fpr_at_threshold_{}".format(suffix)] = float(counts["fpr"])
            point["false_positives_at_fpr_{}".format(suffix)] = int(
                counts["false_positives"]
            )
            point["true_positives_at_fpr_{}".format(suffix)] = int(
                counts["true_positives"]
            )
        else:
            point["threshold_at_fpr_{}".format(suffix)] = None
            point["tpr_at_fpr_{}".format(suffix)] = None
            point["fpr_at_threshold_{}".format(suffix)] = None
            point["false_positives_at_fpr_{}".format(suffix)] = None
            point["true_positives_at_fpr_{}".format(suffix)] = None
    point["threshold_selection"] = selections

    samples = _bootstrap_metrics(
        test_y,
        test_s,
        groups,
        selected_thresholds,
        int(bootstrap_resamples),
        _stable_seed(seed, "detector_metrics"),
    )
    metric_names = ["roc_auc", "partial_auc_fpr_0_01"] + [
        name
        for target in selected_thresholds
        for name in (
            "tpr_at_fpr_{}".format(
                "0_01" if math.isclose(target, 0.01) else "0_001"
            ),
            "fpr_at_threshold_{}".format(
                "0_01" if math.isclose(target, 0.01) else "0_001"
            ),
        )
    ]
    for name in metric_names:
        values = np.asarray(samples.get(name, []), dtype=np.float64)
        point["{}_bootstrap_valid".format(name)] = int(len(values))
        if len(values):
            low, high = np.percentile(values, [2.5, 97.5])
            point["{}_ci_low_95".format(name)] = float(low)
            point["{}_ci_high_95".format(name)] = float(high)
        else:
            point["{}_ci_low_95".format(name)] = None
            point["{}_ci_high_95".format(name)] = None
    return point


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_SEED",
    "LOW_FPR_TARGETS",
    "PARTIAL_AUC_MAX_FPR",
    "RevisionV3MetricError",
    "SCHEMA_VERSION",
    "empirical_partial_auc",
    "evaluate_validation_frozen_detector",
    "frozen_threshold_counts",
    "roc_auc",
    "select_validation_threshold",
    "wilson_interval",
]
