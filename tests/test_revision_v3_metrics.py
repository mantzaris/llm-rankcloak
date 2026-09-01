import numpy as np

from rankcloak.revision_v3_metrics import (
    empirical_partial_auc,
    evaluate_validation_frozen_detector,
    frozen_threshold_counts,
    roc_auc,
    select_validation_threshold,
    wilson_interval,
)


def test_perfect_ranking_has_unit_auc_and_normalized_partial_auc():
    labels = np.asarray([0] * 100 + [1] * 100)
    scores = np.asarray(list(np.linspace(0.0, 0.4, 100)) + list(np.linspace(0.6, 1.0, 100)))
    assert roc_auc(labels, scores) == 1.0
    assert empirical_partial_auc(labels, scores, 0.01) == 1.0


def test_threshold_is_selected_by_exact_validation_false_positive_budget():
    labels = np.asarray([0] * 100 + [1] * 20)
    negative_scores = np.linspace(0.0, 0.5, 100)
    positive_scores = np.linspace(0.6, 0.9, 20)
    selected = select_validation_threshold(
        labels, np.concatenate([negative_scores, positive_scores]), 0.01
    )
    assert selected["available"] is True
    assert selected["validation_false_positives"] <= 1
    counts = frozen_threshold_counts(
        labels,
        np.concatenate([negative_scores, positive_scores]),
        selected["threshold"],
    )
    assert counts["tpr"] == 1.0


def test_point_one_percent_is_unavailable_without_one_thousand_negatives():
    labels = np.asarray([0] * 999 + [1] * 10)
    scores = np.linspace(0.0, 1.0, len(labels))
    selected = select_validation_threshold(labels, scores, 0.001)
    assert selected["available"] is False
    assert selected["minimum_negative_count"] == 1000


def test_validation_frozen_evaluation_reports_counts_intervals_and_warning():
    validation_labels = np.asarray([0] * 100 + [1] * 100)
    validation_scores = np.asarray([0.1] * 100 + [0.9] * 100)
    test_labels = np.asarray([0] * 100 + [1] * 100)
    test_scores = np.asarray([0.2] * 100 + [0.95] * 100)
    groups = ["group-{}".format(index // 2) for index in range(200)]
    result = evaluate_validation_frozen_detector(
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
        groups,
        bootstrap_resamples=20,
        seed=7,
    )
    assert result["test_negative_count"] == 100
    assert result["tpr_at_fpr_0_01"] == 1.0
    assert result["threshold_selection"]["fpr_0_001"]["available"] is False
    assert result["roc_auc_bootstrap_valid"] > 0


def test_wilson_interval_contains_empirical_proportion():
    low, high = wilson_interval(1, 100)
    assert low < 0.01 < high
