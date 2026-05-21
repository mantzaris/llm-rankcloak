"""Lightweight detector baselines for paper-oriented RankCloak features."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DETECTOR_NUMERIC_FEATURES = [
    "token_count",
    "character_count",
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
    "artifact_count_total",
    "contains_backtick",
    "contains_bracket_placeholder",
    "contains_url_fragment",
    "contains_latex_fragment",
    "contains_html_fragment",
    "contains_markdown_heading",
    "contains_markdown_emphasis",
    "contains_separator_line",
]


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Compute ROC AUC from labels and scores without external dependencies."""

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    positive_scores = s[y == 1]
    negative_scores = s[y == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return None
    wins = 0.0
    total = float(positive_scores.size * negative_scores.size)
    for positive in positive_scores:
        wins += float(np.count_nonzero(positive > negative_scores))
        wins += 0.5 * float(np.count_nonzero(positive == negative_scores))
    return wins / total


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(predictions, dtype=int)
    if y.size == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    true_positive = int(np.count_nonzero((y == 1) & (p == 1)))
    true_negative = int(np.count_nonzero((y == 0) & (p == 0)))
    false_positive = int(np.count_nonzero((y == 0) & (p == 1)))
    false_negative = int(np.count_nonzero((y == 1) & (p == 0)))
    precision = (
        true_positive / float(true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / float(true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / float(precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "accuracy": (true_positive + true_negative) / float(y.size),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def deterministic_stratified_split(
    frame: pd.DataFrame,
    label_column: str = "label",
    test_fraction: float = 0.35,
    seed: int = 20260521,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train and test positional indices with deterministic stratification."""

    rng = np.random.default_rng(int(seed))
    train_indices: List[int] = []
    test_indices: List[int] = []
    for label_value in sorted(frame[label_column].dropna().unique()):
        positions = np.flatnonzero(frame[label_column].to_numpy() == label_value)
        rng.shuffle(positions)
        test_count = max(1, int(round(len(positions) * float(test_fraction))))
        if len(positions) - test_count <= 0 and len(positions) > 1:
            test_count = len(positions) - 1
        test_indices.extend(map(int, positions[:test_count]))
        train_indices.extend(map(int, positions[test_count:]))
    return np.asarray(sorted(train_indices), dtype=int), np.asarray(sorted(test_indices), dtype=int)


def prepare_detector_dataset(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Create detector datasets from the unified paper feature table."""

    if feature_frame.empty:
        return pd.DataFrame()
    rows = []
    dataset_sources = {
        "detector_full_message": {"baseline", "nonseg_rankcloak", "segmented_full_message"},
        "detector_forced_prefix": {"baseline", "segmented_forced_prefix"},
        "detector_nonseg_only": {"baseline", "nonseg_rankcloak"},
        "detector_segmented_full_only": {"baseline", "segmented_full_message"},
    }
    for dataset_name, source_types in dataset_sources.items():
        subset = feature_frame[feature_frame["source_type"].isin(source_types)].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            output = row.to_dict()
            output["dataset_name"] = dataset_name
            output["label"] = 0 if row.get("source_type") == "baseline" else 1
            rows.append(output)
    return pd.DataFrame(rows)


def _feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    matrix = frame.reindex(columns=list(features)).copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float64)


def _threshold_detector_row(
    dataset_name: str,
    frame: pd.DataFrame,
    split_name: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    score_column: str,
) -> dict:
    train = frame.iloc[train_indices]
    test = frame.iloc[test_indices]
    train_scores = -pd.to_numeric(train[score_column], errors="coerce").fillna(0.0).to_numpy()
    test_scores = -pd.to_numeric(test[score_column], errors="coerce").fillna(0.0).to_numpy()
    train_labels = train["label"].astype(int).to_numpy()
    test_labels = test["label"].astype(int).to_numpy()
    if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
        threshold = float(np.median(train_scores)) if train_scores.size else 0.0
    else:
        positive_mean = float(np.mean(train_scores[train_labels == 1]))
        negative_mean = float(np.mean(train_scores[train_labels == 0]))
        threshold = (positive_mean + negative_mean) / 2.0
    predictions = (test_scores >= threshold).astype(int)
    metrics = classification_metrics(test_labels, predictions)
    return {
        "split_name": split_name,
        "detector_name": "threshold_mean_token_log_probability",
        "dataset_name": dataset_name,
        "feature_set": score_column,
        "auc": binary_auc(test_labels, test_scores),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "notes": "Dependency-free threshold baseline; higher score means lower token log probability.",
    }


def _sklearn_rows(
    dataset_name: str,
    frame: pd.DataFrame,
    split_name: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    features: Sequence[str],
) -> List[dict]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return []

    train = frame.iloc[train_indices]
    test = frame.iloc[test_indices]
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return []
    x_train = _feature_matrix(train, features)
    x_test = _feature_matrix(test, features)
    models = [
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=20260521),
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=100, random_state=20260521, max_depth=4),
        ),
    ]
    rows = []
    for detector_name, model in models:
        try:
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            predictions = model.predict(x_test)
            rows.append(
                {
                    "split_name": split_name,
                    "detector_name": detector_name,
                    "dataset_name": dataset_name,
                    "feature_set": "numeric_cover_features",
                    "auc": float(roc_auc_score(y_test, probabilities)),
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "precision": float(precision_score(y_test, predictions, zero_division=0)),
                    "recall": float(recall_score(y_test, predictions, zero_division=0)),
                    "f1": float(f1_score(y_test, predictions, zero_division=0)),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "notes": "Lightweight sklearn feature detector; no text content features.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "split_name": split_name,
                    "detector_name": detector_name,
                    "dataset_name": dataset_name,
                    "feature_set": "numeric_cover_features",
                    "auc": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "notes": "sklearn detector failed: {}".format(exc),
                }
            )
    return rows


def run_detector_baselines(detector_dataset: pd.DataFrame) -> pd.DataFrame:
    """Run modest detector baselines without using text content directly."""

    rows = []
    if detector_dataset.empty:
        return pd.DataFrame(rows)
    features = [feature for feature in DETECTOR_NUMERIC_FEATURES if feature in detector_dataset.columns]
    for dataset_name, dataset_frame in detector_dataset.groupby("dataset_name"):
        frame = dataset_frame.copy().reset_index(drop=True)
        if len(frame) < 4 or len(frame["label"].unique()) < 2:
            continue
        train_indices, test_indices = deterministic_stratified_split(frame)
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        if "mean_token_log_probability" in frame.columns:
            rows.append(
                _threshold_detector_row(
                    dataset_name,
                    frame,
                    "random_stratified",
                    train_indices,
                    test_indices,
                    "mean_token_log_probability",
                )
            )
        rows.extend(
            _sklearn_rows(
                dataset_name,
                frame,
                "random_stratified",
                train_indices,
                test_indices,
                features,
            )
        )
        for split_column in ("prompt_family", "payload_class"):
            if split_column not in frame.columns:
                continue
            for held_out in sorted(value for value in frame[split_column].dropna().unique()):
                test_mask = frame[split_column] == held_out
                train_mask = ~test_mask
                if (
                    frame.loc[test_mask, "label"].nunique() < 2
                    or frame.loc[train_mask, "label"].nunique() < 2
                ):
                    continue
                train_positions = np.flatnonzero(train_mask.to_numpy())
                test_positions = np.flatnonzero(test_mask.to_numpy())
                split_name = "leave_one_{}_out:{}".format(split_column, held_out)
                if "mean_token_log_probability" in frame.columns:
                    rows.append(
                        _threshold_detector_row(
                            dataset_name,
                            frame,
                            split_name,
                            train_positions,
                            test_positions,
                            "mean_token_log_probability",
                        )
                    )
                rows.extend(
                    _sklearn_rows(
                        dataset_name,
                        frame,
                        split_name,
                        train_positions,
                        test_positions,
                        features,
                    )
                )
    return pd.DataFrame(rows)
