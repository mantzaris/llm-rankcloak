"""Supplementary payload-grouped diagnostics for frozen neural detector outputs.

This analysis never trains or selects a detector. It reuses the frozen upstream
metrics, checks their point estimates against saved predictions, and adds the
reviewer-requested precision, Brier, and low-FPR summaries. The added metrics
are explicitly exploratory because they were specified after partial checkpoint
outcomes had been inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


OUTPUT_FILENAMES = {
    "metrics": "detector_extended_metrics.csv",
    "regimes": "detector_regime_summary.csv",
    "plot_source": "detector_plot_source.csv",
    "manifest": "detector_analysis_manifest.json",
}
CORE_METRICS = {
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "recall": "sensitivity",
    "specificity": "specificity",
}
SUPPLEMENTARY_METRICS = (
    "precision",
    "brier_score",
    "tpr_at_fpr_0.01",
    "tpr_at_fpr_0.05",
)
PLOT_METRICS = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "brier_score",
    "tpr_at_fpr_0.01",
)


class DetectorAnalysisError(ValueError):
    """Raised when frozen detector outputs cannot support the analysis."""


@dataclass(frozen=True)
class DetectorAnalysisArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise DetectorAnalysisError(f"Missing or unsafe {label}: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DetectorAnalysisError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DetectorAnalysisError(f"{label} must contain a JSON object")
    return value


def _read_csv(path: str | Path, *, label: str) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise DetectorAnalysisError(f"Missing or unsafe {label}: {candidate}")
    try:
        return pd.read_csv(candidate, low_memory=False)
    except Exception as exc:
        raise DetectorAnalysisError(f"Could not read {label}: {exc}") from exc


def _declared_output(
    manifest: Mapping[str, Any], manifest_path: Path, filename: str
) -> Path:
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(filename), Mapping):
        raise DetectorAnalysisError(f"Detector run manifest lacks {filename}")
    declaration = outputs[filename]
    raw = declaration.get("path")
    if raw:
        candidate = Path(str(raw))
        path = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    else:
        output_dir = manifest.get("output_dir")
        if not output_dir:
            raise DetectorAnalysisError("Detector run manifest lacks output_dir")
        path = Path(str(output_dir)) / filename
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise DetectorAnalysisError(f"Declared detector output is missing: {path}")
    if file_sha256(path) != declaration.get("sha256"):
        raise DetectorAnalysisError(f"Detector output hash mismatch: {path}")
    if int(path.stat().st_size) != int(declaration.get("size_bytes", -1)):
        raise DetectorAnalysisError(f"Detector output size mismatch: {path}")
    return path


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DetectorAnalysisError(
            f"{label} lacks required columns: {', '.join(missing)}"
        )


def _binary(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise DetectorAnalysisError(f"{label} must contain only zero and one")
    return numeric.astype(int).to_numpy()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _confusion(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    predicted = scores >= threshold
    true_positive = float(weights[(labels == 1) & predicted].sum())
    false_positive = float(weights[(labels == 0) & predicted].sum())
    true_negative = float(weights[(labels == 0) & ~predicted].sum())
    false_negative = float(weights[(labels == 1) & ~predicted].sum())
    return true_positive, false_positive, true_negative, false_negative


def _core_points(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    weights = np.ones(len(labels), dtype=float)
    tp, fp, tn, fn = _confusion(labels, scores, threshold, weights)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(2.0 * tp / (2.0 * tp + fp + fn)),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def _supplementary_points(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    low_fprs: tuple[float, ...],
    weights: np.ndarray,
) -> dict[str, float]:
    tp, fp, _tn, _fn = _confusion(labels, scores, threshold, weights)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    brier = float(np.average((scores - labels) ** 2, weights=weights))
    fpr, tpr, _ = roc_curve(
        labels,
        scores,
        sample_weight=weights,
        drop_intermediate=False,
    )
    values = {
        "precision": float(precision),
        "brier_score": brier,
    }
    for target in low_fprs:
        eligible = tpr[fpr <= target + 1e-12]
        values[f"tpr_at_fpr_{target:.2f}"] = (
            float(np.max(eligible)) if len(eligible) else float("nan")
        )
    return values


def _bootstrap_supplementary(
    cell: pd.DataFrame,
    *,
    threshold: float,
    low_fprs: tuple[float, ...],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float, int]]]:
    labels = _binary(cell["label"], label="detector label")
    scores = pd.to_numeric(cell["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise DetectorAnalysisError("Detector scores must be finite values in [0, 1]")
    groups = cell["payload_group_id"].astype(str)
    unique_groups = sorted(groups.unique())
    if len(unique_groups) < 2:
        raise DetectorAnalysisError("Detector cell has fewer than two payload groups")
    group_index = {value: index for index, value in enumerate(unique_groups)}
    row_group = groups.map(group_index).to_numpy(dtype=int)
    for _group, rows in cell.groupby("payload_group_id", sort=False):
        group_labels = _binary(rows["label"], label="payload-group label")
        positive = int(group_labels.sum())
        negative = int(len(group_labels) - positive)
        if positive < 1 or positive != negative:
            raise DetectorAnalysisError(
                "Each detector payload group must contain balanced positive and control rows"
            )

    point = _supplementary_points(
        labels,
        scores,
        threshold,
        low_fprs,
        np.ones(len(cell), dtype=float),
    )
    draws = {metric: [] for metric in point}
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(unique_groups), 1.0 / len(unique_groups))
    for _ in range(resamples):
        group_weights = rng.multinomial(len(unique_groups), probabilities)
        weights = group_weights[row_group].astype(float)
        values = _supplementary_points(
            labels, scores, threshold, low_fprs, weights
        )
        for metric, value in values.items():
            if math.isfinite(value):
                draws[metric].append(value)
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, tuple[float, float, int]] = {}
    for metric, values in draws.items():
        if values:
            low, high = np.quantile(values, [alpha, 1.0 - alpha])
            intervals[metric] = (float(low), float(high), len(values))
        else:
            intervals[metric] = (float("nan"), float("nan"), 0)
    return point, intervals


def _validate_run(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Path, Path]:
    expected = int(config.get("expected_fit_count", -1))
    if (
        manifest.get("schema_version") != "rankcloak-revision-detector-run-v2"
        or manifest.get("execution_mode") != "confirmatory"
        or manifest.get("confirmatory_complete") is not True
        or int(manifest.get("completed_fit_count", -1)) != expected
        or int(manifest.get("total_fit_count", -1)) != expected
        or int(manifest.get("failure_count", -1)) != 0
        or int(manifest.get("smoke_fallback_metric_rows", -1)) != 0
        or manifest.get("device") != "cuda:0"
    ):
        raise DetectorAnalysisError(
            "Detector run manifest is not a complete frozen CUDA confirmatory run"
        )
    predictions = _declared_output(
        manifest, manifest_path, "detector_predictions.csv"
    )
    metrics = _declared_output(manifest, manifest_path, "detector_metrics.csv")
    return predictions, metrics


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def _atomic_json(value: Mapping[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def analyze_detector_outputs(
    *,
    detector_run_manifest: str | Path,
    analysis_config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> DetectorAnalysisArtifacts:
    """Validate the frozen run and emit supplementary grouped diagnostics."""

    manifest_path = Path(detector_run_manifest).resolve()
    config_path = Path(analysis_config).resolve()
    manifest = _read_json(manifest_path, label="detector run manifest")
    config = _read_json(config_path, label="supplementary detector config")
    if (
        config.get("schema_version")
        != "rankcloak-detector-supplementary-metrics-v1"
        or config.get("analysis_status")
        != "supplementary_exploratory_post_confirmatory_freeze"
        or config.get("partial_checkpoint_outcomes_seen_before_extension") is not True
        or config.get("frozen_training_design_unchanged") is not True
    ):
        raise DetectorAnalysisError("Supplementary detector config disclosure differs")
    predictions_path, metrics_path = _validate_run(
        manifest_path, manifest, config
    )
    predictions = _read_csv(predictions_path, label="detector predictions")
    upstream = _read_csv(metrics_path, label="detector metrics")
    _require_columns(
        predictions,
        (
            "split_id",
            "regime",
            "held_out_value",
            "detector_name",
            "requested_kind",
            "implementation_kind",
            "implementation_status",
            "row_id",
            "payload_group_id",
            "label",
            "score",
            "prediction",
        ),
        label="detector predictions",
    )
    _require_columns(
        upstream,
        (
            "split_id",
            "regime",
            "detector_name",
            "implementation_status",
            "bootstrap_unit",
            "bootstrap_resamples_requested",
            *tuple(CORE_METRICS.values()),
            *tuple(
                column
                for upstream_name in set(CORE_METRICS.values())
                for column in (
                    f"{upstream_name}_bootstrap_valid",
                    f"{upstream_name}_ci_low_95",
                    f"{upstream_name}_ci_high_95",
                )
            ),
        ),
        label="detector metrics",
    )
    if predictions["row_id"].isna().any() or predictions.duplicated(
        ["split_id", "detector_name", "row_id"]
    ).any():
        raise DetectorAnalysisError("Detector prediction row identities are not unique")
    if not predictions["implementation_status"].astype(str).eq("complete").all():
        raise DetectorAnalysisError("Incomplete detector predictions are present")
    expected_fit_count = int(config["expected_fit_count"])
    upstream_keys = ["split_id", "detector_name"]
    if len(upstream) != expected_fit_count or upstream.duplicated(upstream_keys).any():
        raise DetectorAnalysisError("Upstream detector metric matrix is incomplete")
    if not upstream["implementation_status"].astype(str).eq("complete").all():
        raise DetectorAnalysisError("Upstream detector metrics are incomplete")
    if not upstream["bootstrap_unit"].astype(str).eq("payload_group_id").all():
        raise DetectorAnalysisError("Upstream metrics use the wrong bootstrap unit")

    threshold = float(config["decision_threshold"])
    low_fprs = tuple(float(value) for value in config["low_false_positive_rates"])
    bootstrap = config["bootstrap"]
    resamples = int(bootstrap["resamples"])
    seed = int(bootstrap["seed"])
    confidence_level = float(bootstrap["confidence_level"])
    if (
        tuple(low_fprs) != (0.01, 0.05)
        or threshold != 0.5
        or bootstrap.get("unit") != "payload_group_id"
        or int(config.get("precision_zero_division", -1)) != 0
        or resamples <= 0
        or not 0.0 < confidence_level < 1.0
    ):
        raise DetectorAnalysisError("Supplementary detector settings differ")

    metric_rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(
        [
            "split_id",
            "regime",
            "held_out_value",
            "detector_name",
            "requested_kind",
            "implementation_kind",
        ],
        dropna=False,
        sort=True,
    )
    if grouped.ngroups != expected_fit_count:
        raise DetectorAnalysisError("Prediction detector matrix is incomplete")
    for keys, cell in grouped:
        (
            split_id,
            regime,
            held_out_value,
            detector_name,
            requested_kind,
            implementation_kind,
        ) = keys
        upstream_cell = upstream.loc[
            upstream["split_id"].astype(str).eq(str(split_id))
            & upstream["detector_name"].astype(str).eq(str(detector_name))
        ]
        if len(upstream_cell) != 1:
            raise DetectorAnalysisError("Prediction cell lacks one upstream metric row")
        upstream_row = upstream_cell.iloc[0]
        labels = _binary(cell["label"], label="detector label")
        scores = pd.to_numeric(cell["score"], errors="coerce").to_numpy(dtype=float)
        predicted = _binary(cell["prediction"], label="detector prediction")
        if not np.array_equal(predicted, (scores >= threshold).astype(int)):
            raise DetectorAnalysisError("Saved predictions differ from fixed threshold")
        core_points = _core_points(labels, scores, threshold)
        common = {
            "split_id": str(split_id),
            "regime": str(regime),
            "held_out_value": (
                "not_applicable"
                if pd.isna(held_out_value)
                else str(held_out_value)
            ),
            "detector_name": str(detector_name),
            "requested_kind": str(requested_kind),
            "implementation_kind": str(implementation_kind),
            "n_test_rows": int(len(cell)),
            "n_payload_groups": int(cell["payload_group_id"].nunique()),
            "positive_rows": int(labels.sum()),
            "negative_rows": int(len(labels) - labels.sum()),
            "analysis_unit": "payload_group_id",
            "confidence_level": confidence_level,
        }
        for metric, upstream_name in CORE_METRICS.items():
            observed = float(upstream_row[upstream_name])
            ci_low = float(upstream_row[f"{upstream_name}_ci_low_95"])
            ci_high = float(upstream_row[f"{upstream_name}_ci_high_95"])
            valid = int(upstream_row[f"{upstream_name}_bootstrap_valid"])
            if (
                not all(math.isfinite(value) for value in (observed, ci_low, ci_high))
                or not 0.0 <= observed <= 1.0
                or not 0.0 <= ci_low <= ci_high <= 1.0
                or valid <= 0
            ):
                raise DetectorAnalysisError(
                    f"Upstream {metric} confidence interval is invalid"
                )
            if not math.isclose(observed, core_points[metric], rel_tol=0.0, abs_tol=1e-12):
                raise DetectorAnalysisError(
                    f"Upstream {metric} differs from saved predictions"
                )
            metric_rows.append(
                {
                    **common,
                    "metric": metric,
                    "estimate": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_resamples_requested": int(
                        upstream_row["bootstrap_resamples_requested"]
                    ),
                    "bootstrap_resamples_valid": valid,
                    "evidence_status": "confirmatory_frozen_upstream",
                    "metric_source": "frozen_upstream_payload_group_bootstrap",
                    "decision_threshold": (
                        threshold
                        if metric
                        in {"balanced_accuracy", "f1", "recall", "specificity"}
                        else np.nan
                    ),
                    "false_positive_rate_constraint": np.nan,
                    "higher_is_better": True,
                }
            )
        supplementary, intervals = _bootstrap_supplementary(
            cell,
            threshold=threshold,
            low_fprs=low_fprs,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=_stable_seed(seed, split_id, detector_name),
        )
        for metric in SUPPLEMENTARY_METRICS:
            low, high, valid = intervals[metric]
            metric_rows.append(
                {
                    **common,
                    "metric": metric,
                    "estimate": supplementary[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_resamples_requested": resamples,
                    "bootstrap_resamples_valid": valid,
                    "evidence_status": "supplementary_exploratory_post_freeze",
                    "metric_source": "recomputed_payload_group_bootstrap",
                    "decision_threshold": (
                        threshold if metric == "precision" else np.nan
                    ),
                    "false_positive_rate_constraint": (
                        0.01
                        if metric == "tpr_at_fpr_0.01"
                        else 0.05
                        if metric == "tpr_at_fpr_0.05"
                        else np.nan
                    ),
                    "higher_is_better": metric != "brier_score",
                }
            )

    extended = pd.DataFrame(metric_rows).sort_values(
        ["regime", "held_out_value", "detector_name", "metric"]
    ).reset_index(drop=True)
    if extended[["estimate"]].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise DetectorAnalysisError("A detector point estimate is unavailable")
    regime_rows: list[dict[str, Any]] = []
    for keys, cell in extended.groupby(
        ["detector_name", "regime", "metric", "evidence_status", "higher_is_better"],
        dropna=False,
        sort=True,
    ):
        detector_name, regime, metric, evidence_status, higher_is_better = keys
        regime_rows.append(
            {
                "detector_name": detector_name,
                "regime": regime,
                "metric": metric,
                "evidence_status": evidence_status,
                "higher_is_better": bool(higher_is_better),
                "split_count": int(cell["split_id"].nunique()),
                "estimate_median_across_splits": float(cell["estimate"].median()),
                "estimate_min_across_splits": float(cell["estimate"].min()),
                "estimate_max_across_splits": float(cell["estimate"].max()),
                "test_rows_sum_across_splits": int(cell["n_test_rows"].sum()),
                "payload_groups_sum_across_splits": int(
                    cell["n_payload_groups"].sum()
                ),
                "cross_split_interval": "not_computed_heterogeneous_prespecified_splits",
            }
        )
    regimes = pd.DataFrame(regime_rows)
    plot_source = regimes.loc[regimes["metric"].isin(PLOT_METRICS)].copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise DetectorAnalysisError(
            "Refusing to overwrite detector analysis outputs: "
            + ", ".join(str(path) for path in existing)
        )
    _atomic_csv(extended, targets["metrics"])
    _atomic_csv(regimes, targets["regimes"])
    _atomic_csv(plot_source, targets["plot_source"])
    summary = {
        "fit_count": expected_fit_count,
        "detector_count": int(extended["detector_name"].nunique()),
        "split_count": int(extended["split_id"].nunique()),
        "metric_count": int(extended["metric"].nunique()),
        "metric_rows": int(len(extended)),
        "regime_summary_rows": int(len(regimes)),
        "supplementary_metric_rows": int(
            extended["evidence_status"]
            .eq("supplementary_exploratory_post_freeze")
            .sum()
        ),
        "upstream_point_estimate_mismatches": 0,
    }
    output_manifest: dict[str, Any] = {
        "schema_version": "rankcloak-detector-supplementary-analysis-v1",
        "status": "passed",
        "analysis_status": config["analysis_status"],
        "partial_checkpoint_outcomes_seen_before_extension": True,
        "frozen_training_design_unchanged": True,
        "inputs": {
            "detector_run_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
            },
            "detector_predictions": {
                "path": str(predictions_path),
                "sha256": file_sha256(predictions_path),
                "size_bytes": predictions_path.stat().st_size,
            },
            "detector_metrics": {
                "path": str(metrics_path),
                "sha256": file_sha256(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            },
            "analysis_config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
                "size_bytes": config_path.stat().st_size,
            },
        },
        "inference": {
            "bootstrap_unit": "payload_group_id",
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "confidence_level": confidence_level,
            "precision_zero_division": 0,
            "low_fpr_interpretation": config["low_fpr_interpretation"],
            "regime_summary": config["regime_summary"],
        },
        "outputs": {},
        "summary": summary,
    }
    for key in ("metrics", "regimes", "plot_source"):
        target = targets[key]
        output_manifest["outputs"][key] = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
            "row_count": int(
                len(
                    extended
                    if key == "metrics"
                    else regimes
                    if key == "regimes"
                    else plot_source
                )
            ),
        }
    _atomic_json(output_manifest, targets["manifest"])
    return DetectorAnalysisArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=summary,
    )
