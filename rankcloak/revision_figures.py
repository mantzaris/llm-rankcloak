"""Data-only computational figures for the final revision evidence package."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


OUTPUT_FILENAMES = {
    "robustness_pdf": "robustness_recovery.pdf",
    "robustness_png": "robustness_recovery.png",
    "robustness_source": "robustness_figure_source.csv",
    "theory_pdf": "capacity_tail_validation.pdf",
    "theory_png": "capacity_tail_validation.png",
    "theory_source": "theory_figure_source.csv",
    "readability_pdf": "automated_readability.pdf",
    "readability_png": "automated_readability.png",
    "readability_source": "readability_figure_source.csv",
    "overhead_pdf": "computational_overhead.pdf",
    "overhead_png": "computational_overhead.png",
    "overhead_source": "overhead_figure_source.csv",
    "detector_pdf": "neural_detector_performance.pdf",
    "detector_png": "neural_detector_performance.png",
    "detector_source": "detector_figure_source.csv",
    "commands": "generation_commands.txt",
    "manifest": "figure_manifest.json",
}

FAMILY_ORDER = (
    "replay_modes",
    "raw_transmission",
    "limited_mitigation",
    "cross_model_mismatch",
)
CONDITION_ORDER = (
    "ordinary_llm_control",
    "rankcloak_ascii_b8",
    "rankcloak_ascii_b16",
    "rankcloak_hex_nibble",
    "direct_subword_calgacus",
    "rankcloak_segmented_forced_span",
    "rankcloak_segmented_full_message",
)
READABILITY_OUTCOMES = (
    "flesch_reading_ease_heuristic",
    "surface_flag_total",
    "tfidf_prompt_similarity",
)
OVERHEAD_OUTCOMES = (
    "generation_seconds",
    "encoding_overhead_seconds",
    "decoding_overhead_seconds",
    "payload_bits_per_second",
)
OVERHEAD_PROTOCOL_ORDER = (
    "direct_subword_calgacus",
    "nonseg_ascii_b16",
    "nonseg_ascii_b8",
    "nonseg_hex_nibble_b16",
    "segmented_hex_multi_topic",
    "segmented_hex_single_topic",
)
OVERHEAD_MODEL_LABELS = {
    "llama3_8b_instruct_q4_k_m": "Llama 3 8B",
    "mistral_7b_instruct_v0_3_q4_k_m": "Mistral 7B",
    "qwen2_5_7b_instruct_q4_k_m": "Qwen 2.5 7B",
}
DETECTOR_REGIME_ORDER = (
    "matched",
    "held_out_template",
    "leave_one_model",
    "leave_one_codec",
)
DETECTOR_ORDER = (
    "published_textcnn_equivalent",
    "deberta_v3_base_classifier",
)
DETECTOR_LABELS = {
    "published_textcnn_equivalent": "TextCNN",
    "deberta_v3_base_classifier": "DeBERTa-v3-base",
}
DETECTOR_OUTCOMES = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "brier_score",
    "tpr_at_fpr_0.01",
)


class FigureEvidenceError(ValueError):
    """Raised when a source manifest or plot table is inconsistent."""


@dataclass(frozen=True)
class FigureArtifacts:
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
    resolved = Path(path)
    if not resolved.is_file():
        raise FigureEvidenceError(f"Missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FigureEvidenceError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FigureEvidenceError(f"{label} must contain a JSON object")
    return value


def _declared_output(
    manifest: Mapping[str, Any], manifest_path: Path, key: str
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(key), Mapping):
        raise FigureEvidenceError(f"Manifest lacks declared output {key}")
    entry = outputs[key]
    raw_path = Path(str(entry.get("path", "")))
    path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    if not path.is_file():
        raise FigureEvidenceError(f"Declared output does not exist: {path}")
    observed = file_sha256(path)
    if observed != entry.get("sha256"):
        raise FigureEvidenceError(f"Declared output hash mismatch: {path}")
    return path


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FigureEvidenceError(f"{label} lacks columns: {', '.join(missing)}")


def _display(value: Any) -> str:
    return str(value).replace("_", " ")


def _interval_errors(
    point: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.concatenate([point, low, high])
    if not np.isfinite(values).all():
        raise FigureEvidenceError(f"{label} contains non-finite interval values")
    left = point - low
    right = high - point
    tolerance = 1e-12
    if (left < -tolerance).any() or (right < -tolerance).any():
        raise FigureEvidenceError(
            f"{label} contains an interval excluding its estimate"
        )
    return np.vstack([np.maximum(left, 0.0), np.maximum(right, 0.0)])


def _robustness_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        (
            "robustness_family",
            "replay_mode",
            "transformation_id",
            "observed_outcome_rows",
            "unavailable_rows",
            "recovery_rate",
            "ci_low",
            "ci_high",
            "status",
        ),
        label="robustness plot source",
    )
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILY_ORDER):
        cell = frame.loc[frame["robustness_family"].eq(family)].copy()
        if cell.empty:
            continue
        cell = cell.sort_values(["transformation_id", "replay_mode"])
        for within_index, (_, row) in enumerate(cell.iterrows()):
            transformation = str(row["transformation_id"])
            replay = str(row["replay_mode"])
            label = transformation
            if transformation in {"", "nan", "not_applicable", "unmodified"}:
                label = replay
            elif replay not in {"", "nan", "not_applicable"}:
                label = f"{transformation} | {replay}"
            rows.append(
                {
                    **row.to_dict(),
                    "family_order": family_index,
                    "within_family_order": within_index,
                    "display_label": _display(label),
                }
            )
    return pd.DataFrame(rows)


def _plot_robustness(source: pd.DataFrame) -> plt.Figure:
    families = [family for family in FAMILY_ORDER if family in set(source["robustness_family"])]
    fig, axes = plt.subplots(
        len(families), 1, figsize=(8.2, 2.0 + 0.36 * len(source)), squeeze=False
    )
    color = "#0072B2"
    for axis, family in zip(axes[:, 0], families):
        cell = source.loc[source["robustness_family"].eq(family)].sort_values(
            "within_family_order"
        )
        y = np.arange(len(cell))
        point = pd.to_numeric(cell["recovery_rate"], errors="coerce").to_numpy()
        low = pd.to_numeric(cell["ci_low"], errors="coerce").to_numpy()
        high = pd.to_numeric(cell["ci_high"], errors="coerce").to_numpy()
        axis.errorbar(
            point,
            y,
            xerr=_interval_errors(point, low, high, label=f"robustness {family}"),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=4.5,
        )
        axis.set_yticks(y, cell["display_label"], fontsize=7.5)
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.6)
        axis.set_title(_display(family), loc="left", fontsize=9, fontweight="bold")
        axis.invert_yaxis()
    axes[-1, 0].set_xlabel("Exact payload recovery probability (95% source-cover CI)")
    fig.suptitle("Transmission and replay perturbation outcomes", fontsize=11)
    fig.tight_layout()
    return fig


def _theory_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        (
            "source_stage",
            "model_id",
            "protocol_variant",
            "tail_policy",
            "theoretical_n_B",
            "observed_n_forced",
            "tail_overhead_tokens",
        ),
        label="theory validation",
    )
    frame["theoretical_n_B"] = pd.to_numeric(frame["theoretical_n_B"], errors="coerce")
    frame["observed_n_forced"] = pd.to_numeric(
        frame["observed_n_forced"], errors="coerce"
    )
    frame["tail_overhead_tokens"] = pd.to_numeric(
        frame["tail_overhead_tokens"], errors="coerce"
    )
    frame = frame.loc[frame["theoretical_n_B"].notna()].copy()
    groups = ["source_stage", "protocol_variant", "tail_policy"]
    rows: list[dict[str, Any]] = []
    for keys, cell in frame.groupby(groups, dropna=False, sort=True):
        rows.append(
            {
                **dict(zip(groups, keys)),
                "n": len(cell),
                "mean_theoretical_n_B": float(cell["theoretical_n_B"].mean()),
                "mean_observed_n_forced": float(cell["observed_n_forced"].mean()),
                "forced_residual_max_abs": float(
                    (cell["observed_n_forced"] - cell["theoretical_n_B"]).abs().max()
                ),
                "tail_median": float(cell["tail_overhead_tokens"].median()),
                "tail_q05": float(cell["tail_overhead_tokens"].quantile(0.05)),
                "tail_q95": float(cell["tail_overhead_tokens"].quantile(0.95)),
                "tail_max": float(cell["tail_overhead_tokens"].max()),
                "display_label": _display(" | ".join(str(value) for value in keys)),
            }
        )
    return pd.DataFrame(rows)


def _plot_theory(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, max(4.3, 0.36 * len(source))))
    colors = {
        stage: color
        for stage, color in zip(
            sorted(source["source_stage"].unique()),
            ("#0072B2", "#D55E00", "#009E73", "#CC79A7"),
        )
    }
    for stage, cell in source.groupby("source_stage", sort=True):
        axes[0].scatter(
            cell["mean_theoretical_n_B"],
            cell["mean_observed_n_forced"],
            s=np.clip(np.sqrt(cell["n"]) * 4, 18, 90),
            label=_display(stage),
            color=colors[stage],
            alpha=0.8,
        )
    maximum = max(
        source["mean_theoretical_n_B"].max(), source["mean_observed_n_forced"].max()
    )
    axes[0].plot([0, maximum], [0, maximum], color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Theoretical minimum forced positions")
    axes[0].set_ylabel("Observed forced positions")
    axes[0].set_title("A  Capacity-position validation", loc="left", fontweight="bold")
    axes[0].grid(color="#e5e5e5", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    ordered = source.sort_values(["tail_q95", "tail_median"])
    y = np.arange(len(ordered))
    median = np.log1p(ordered["tail_median"].to_numpy())
    q05 = np.log1p(ordered["tail_q05"].to_numpy())
    q95 = np.log1p(ordered["tail_q95"].to_numpy())
    axes[1].errorbar(
        median,
        y,
        xerr=_interval_errors(median, q05, q95, label="theory tail quantiles"),
        fmt="o",
        color="#D55E00",
        ecolor="#D55E00",
        capsize=2.5,
        markersize=4.5,
    )
    axes[1].set_yticks(y, ordered["display_label"], fontsize=7.2)
    ticks = np.asarray([0, 1, 10, 100, 1000])
    axes[1].set_xticks(np.log1p(ticks), [str(value) for value in ticks])
    axes[1].set_xlabel("Tail tokens per message (median and 5th–95th percentiles)")
    axes[1].set_title("B  Segmentation/tail overhead", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#e5e5e5", linewidth=0.6)
    fig.suptitle("Capacity prediction and empirical cover-length overhead", fontsize=11)
    fig.tight_layout()
    return fig


def _readability_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        ("condition", "outcome", "n", "mean", "ci_low", "ci_high"),
        label="readability summary",
    )
    result = frame.loc[frame["outcome"].isin(READABILITY_OUTCOMES)].copy()
    order = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    result["condition_order"] = result["condition"].map(order).fillna(len(order))
    result["display_condition"] = result["condition"].map(_display)
    return result.sort_values(["outcome", "condition_order"]).reset_index(drop=True)


def _plot_readability(source: pd.DataFrame) -> plt.Figure:
    labels = {
        "flesch_reading_ease_heuristic": "Flesch ease (heuristic)",
        "surface_flag_total": "Surface-flag count",
        "tfidf_prompt_similarity": "Prompt similarity (TF–IDF)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0))
    for axis, outcome in zip(axes, READABILITY_OUTCOMES):
        cell = source.loc[source["outcome"].eq(outcome)].sort_values("condition_order")
        y = np.arange(len(cell))
        point = pd.to_numeric(cell["mean"], errors="coerce").to_numpy()
        low = pd.to_numeric(cell["ci_low"], errors="coerce").to_numpy()
        high = pd.to_numeric(cell["ci_high"], errors="coerce").to_numpy()
        axis.errorbar(
            point,
            y,
            xerr=_interval_errors(point, low, high, label=f"readability {outcome}"),
            fmt="o",
            color="#009E73",
            ecolor="#009E73",
            capsize=2.5,
            markersize=4.5,
        )
        axis.set_yticks(y, cell["display_condition"], fontsize=7.5)
        axis.invert_yaxis()
        axis.set_xlabel(labels[outcome])
        axis.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    fig.suptitle(
        "Automated surface diagnostics for the selected computational sample\n"
        "(95% prompt-template-cluster CIs; not human ratings)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _overhead_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        (
            "source_stage",
            "runtime_scope",
            "model_id",
            "protocol_variant",
            "outcome",
            "mean",
            "ci_low",
            "ci_high",
            "n_payloads",
        ),
        label="overhead plot source",
    )
    result = frame.loc[
        frame["source_stage"].eq("primary_v2")
        & frame["runtime_scope"].eq("trial")
        & frame["outcome"].isin(OVERHEAD_OUTCOMES)
    ].copy()
    if result.empty:
        raise FigureEvidenceError("overhead plot source has no primary trial rows")
    for column in ("mean", "ci_low", "ci_high", "n_payloads"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(
        result[["mean", "ci_low", "ci_high", "n_payloads"]].to_numpy()
    ).all():
        raise FigureEvidenceError("overhead plot source contains non-finite values")
    duplicate = result.duplicated(
        ["model_id", "protocol_variant", "outcome"], keep=False
    )
    if duplicate.any():
        raise FigureEvidenceError("overhead plot source contains duplicate cells")
    expected_cells = {
        (model_id, protocol, outcome)
        for model_id in OVERHEAD_MODEL_LABELS
        for protocol in OVERHEAD_PROTOCOL_ORDER
        for outcome in OVERHEAD_OUTCOMES
    }
    observed_cells = set(
        result[["model_id", "protocol_variant", "outcome"]].itertuples(
            index=False, name=None
        )
    )
    if observed_cells != expected_cells:
        raise FigureEvidenceError(
            "overhead plot source does not contain the complete frozen primary grid"
        )
    if (result["n_payloads"] <= 0).any():
        raise FigureEvidenceError("overhead plot source has nonpositive sample sizes")
    protocols = {value: index for index, value in enumerate(OVERHEAD_PROTOCOL_ORDER)}
    models = {value: index for index, value in enumerate(OVERHEAD_MODEL_LABELS)}
    result["protocol_order"] = result["protocol_variant"].map(protocols)
    result["model_order"] = result["model_id"].map(models)
    if result[["protocol_order", "model_order"]].isna().any().any():
        raise FigureEvidenceError(
            "overhead plot source contains an unrecognized model or protocol"
        )
    result["row_order"] = (
        result["protocol_order"].astype(int) * len(models)
        + result["model_order"].astype(int)
    )
    result["display_label"] = result.apply(
        lambda row: (
            f"{_display(row['protocol_variant'])} — "
            f"{OVERHEAD_MODEL_LABELS[str(row['model_id'])]}"
        ),
        axis=1,
    )
    return result.sort_values(["outcome", "row_order"]).reset_index(drop=True)


def _plot_overhead(source: pd.DataFrame) -> plt.Figure:
    titles = {
        "generation_seconds": "A  Generation time (s)",
        "encoding_overhead_seconds": "B  Encoding setup overhead (s)",
        "decoding_overhead_seconds": "C  Supported decoding time (s)",
        "payload_bits_per_second": "D  Artifact payload throughput (bit/s)",
    }
    colors = {
        "generation_seconds": "#0072B2",
        "encoding_overhead_seconds": "#E69F00",
        "decoding_overhead_seconds": "#D55E00",
        "payload_bits_per_second": "#009E73",
    }
    labels = (
        source[["row_order", "display_label"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    expected_rows = len(labels)
    y = np.arange(expected_rows)
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 7.4), sharey=True)
    for axis, outcome in zip(axes, OVERHEAD_OUTCOMES):
        cell = source.loc[source["outcome"].eq(outcome)].sort_values("row_order")
        if len(cell) != expected_rows or cell["row_order"].nunique() != expected_rows:
            raise FigureEvidenceError(
                f"overhead plot source has incomplete rows for {outcome}"
            )
        point = cell["mean"].to_numpy(dtype=float)
        low = cell["ci_low"].to_numpy(dtype=float)
        high = cell["ci_high"].to_numpy(dtype=float)
        if outcome == "encoding_overhead_seconds" and (low <= 0).any():
            raise FigureEvidenceError(
                "encoding-overhead confidence bounds must be positive for log scale"
            )
        axis.errorbar(
            point,
            y,
            xerr=_interval_errors(point, low, high, label=f"overhead {outcome}"),
            fmt="o",
            color=colors[outcome],
            ecolor=colors[outcome],
            capsize=2.2,
            markersize=4.2,
        )
        if outcome == "encoding_overhead_seconds":
            axis.set_xscale("log")
        axis.set_title(titles[outcome], loc="left", fontsize=9, fontweight="bold")
        axis.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        for boundary in range(
            len(OVERHEAD_MODEL_LABELS),
            expected_rows,
            len(OVERHEAD_MODEL_LABELS),
        ):
            axis.axhline(boundary - 0.5, color="#d0d0d0", linewidth=0.6)
    axes[0].set_yticks(y, labels["display_label"], fontsize=7.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Primary-run computational timing and throughput\n"
        "(95% payload-cluster bootstrap confidence intervals)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _detector_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        (
            "detector_name",
            "regime",
            "metric",
            "evidence_status",
            "higher_is_better",
            "split_count",
            "estimate_median_across_splits",
            "estimate_min_across_splits",
            "estimate_max_across_splits",
            "test_rows_sum_across_splits",
            "payload_groups_sum_across_splits",
            "cross_split_interval",
        ),
        label="detector plot source",
    )
    expected = {
        (detector, regime, outcome)
        for detector in DETECTOR_ORDER
        for regime in DETECTOR_REGIME_ORDER
        for outcome in DETECTOR_OUTCOMES
    }
    observed = set(
        frame[["detector_name", "regime", "metric"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or len(frame) != len(expected):
        raise FigureEvidenceError(
            "detector plot source does not contain the complete frozen grid"
        )
    numeric = (
        "estimate_median_across_splits",
        "estimate_min_across_splits",
        "estimate_max_across_splits",
        "split_count",
        "test_rows_sum_across_splits",
        "payload_groups_sum_across_splits",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(numeric)].to_numpy()).all():
        raise FigureEvidenceError("detector plot source contains non-finite values")
    point = frame["estimate_median_across_splits"].to_numpy(dtype=float)
    low = frame["estimate_min_across_splits"].to_numpy(dtype=float)
    high = frame["estimate_max_across_splits"].to_numpy(dtype=float)
    _interval_errors(point, low, high, label="detector cross-split ranges")
    if ((low < 0) | (high > 1)).any():
        raise FigureEvidenceError("detector plot source contains values outside [0, 1]")
    if (frame["split_count"] <= 0).any():
        raise FigureEvidenceError("detector plot source has nonpositive split counts")
    if not frame["cross_split_interval"].astype(str).eq(
        "not_computed_heterogeneous_prespecified_splits"
    ).all():
        raise FigureEvidenceError("detector plot source misstates cross-split intervals")
    confirmatory = frame["metric"].isin(
        ("roc_auc", "pr_auc", "balanced_accuracy")
    )
    if not frame.loc[confirmatory, "evidence_status"].astype(str).eq(
        "confirmatory_frozen_upstream"
    ).all() or not frame.loc[~confirmatory, "evidence_status"].astype(str).eq(
        "supplementary_exploratory_post_freeze"
    ).all():
        raise FigureEvidenceError("detector metric evidence status is inconsistent")
    regime_order = {
        value: index for index, value in enumerate(DETECTOR_REGIME_ORDER)
    }
    detector_order = {value: index for index, value in enumerate(DETECTOR_ORDER)}
    outcome_order = {value: index for index, value in enumerate(DETECTOR_OUTCOMES)}
    frame["regime_order"] = frame["regime"].map(regime_order)
    frame["detector_order"] = frame["detector_name"].map(detector_order)
    frame["outcome_order"] = frame["metric"].map(outcome_order)
    frame["display_detector"] = frame["detector_name"].map(DETECTOR_LABELS)
    return frame.sort_values(
        ["outcome_order", "regime_order", "detector_order"]
    ).reset_index(drop=True)


def _plot_detectors(source: pd.DataFrame) -> plt.Figure:
    titles = {
        "roc_auc": "A  ROC–AUC",
        "pr_auc": "B  PR–AUC",
        "balanced_accuracy": "C  Balanced accuracy",
        "precision": "D  Precision at 0.5",
        "brier_score": "E  Brier score (lower is better)",
        "tpr_at_fpr_0.01": "F  TPR at FPR ≤ 1%",
    }
    colors = {
        "published_textcnn_equivalent": "#0072B2",
        "deberta_v3_base_classifier": "#D55E00",
    }
    markers = {
        "published_textcnn_equivalent": "o",
        "deberta_v3_base_classifier": "s",
    }
    x = np.arange(len(DETECTOR_REGIME_ORDER), dtype=float)
    offsets = (-0.10, 0.10)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for axis, outcome in zip(axes.flat, DETECTOR_OUTCOMES):
        for detector, offset in zip(DETECTOR_ORDER, offsets):
            cell = source.loc[
                source["metric"].eq(outcome)
                & source["detector_name"].eq(detector)
            ].sort_values("regime_order")
            if len(cell) != len(DETECTOR_REGIME_ORDER):
                raise FigureEvidenceError(
                    f"detector plot source has incomplete rows for {outcome}"
                )
            point = cell["estimate_median_across_splits"].to_numpy(dtype=float)
            low = cell["estimate_min_across_splits"].to_numpy(dtype=float)
            high = cell["estimate_max_across_splits"].to_numpy(dtype=float)
            axis.errorbar(
                x + offset,
                point,
                yerr=_interval_errors(
                    point, low, high, label=f"detector {detector} {outcome}"
                ),
                fmt=markers[detector],
                color=colors[detector],
                ecolor=colors[detector],
                capsize=2.5,
                markersize=4.8,
                label=DETECTOR_LABELS[detector],
            )
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_title(titles[outcome], loc="left", fontsize=9, fontweight="bold")
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        axis.set_xticks(x, [_display(value) for value in DETECTOR_REGIME_ORDER])
        axis.tick_params(axis="x", labelrotation=24, labelsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Frozen neural-detector regimes: median and range across prespecified splits\n"
        "Panels D–F are supplementary exploratory post-freeze summaries; bars are ranges, not CIs",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _atomic_write_csv(frame: pd.DataFrame, target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)
    return target


def _atomic_write_text(value: str, target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _atomic_save_figure(figure: plt.Figure, target: Path, *, dpi: int = 300) -> Path:
    temporary = target.with_name(
        f".{target.stem}.tmp-{uuid.uuid4().hex}{target.suffix}"
    )
    metadata = {"Creator": "RankCloak computational evidence pipeline"}
    if target.suffix.lower() == ".pdf":
        metadata.update({"CreationDate": None, "ModDate": None})
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", metadata=metadata)
    os.replace(temporary, target)
    return target


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def build_core_figures(
    *,
    robustness_manifest: str | Path,
    theory_manifest: str | Path,
    readability_manifest: str | Path,
    overhead_manifest: str | Path,
    detector_manifest: str | Path,
    output_dir: str | Path,
    command: str,
    overwrite: bool = False,
) -> FigureArtifacts:
    """Build plots only from hash-validated computational source tables."""

    robustness_manifest_path = Path(robustness_manifest).resolve()
    theory_manifest_path = Path(theory_manifest).resolve()
    readability_manifest_path = Path(readability_manifest).resolve()
    overhead_manifest_path = Path(overhead_manifest).resolve()
    detector_manifest_path = Path(detector_manifest).resolve()
    robustness = _read_json(robustness_manifest_path, label="robustness manifest")
    theory = _read_json(theory_manifest_path, label="empirical theory manifest")
    readability = _read_json(readability_manifest_path, label="readability manifest")
    overhead = _read_json(overhead_manifest_path, label="overhead manifest")
    detector = _read_json(detector_manifest_path, label="detector analysis manifest")
    robustness_path = _declared_output(
        robustness, robustness_manifest_path, "plot_source"
    )
    theory_path = _declared_output(theory, theory_manifest_path, "validation")
    readability_path = _declared_output(readability, readability_manifest_path, "summary")
    overhead_path = _declared_output(overhead, overhead_manifest_path, "plot_source")
    detector_path = _declared_output(
        detector, detector_manifest_path, "plot_source"
    )

    sources = {
        "robustness_source": _robustness_source(robustness_path),
        "theory_source": _theory_source(theory_path),
        "readability_source": _readability_source(readability_path),
        "overhead_source": _overhead_source(overhead_path),
        "detector_source": _detector_source(detector_path),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FigureEvidenceError(
            "Refusing to overwrite figure outputs: "
            + ", ".join(str(path) for path in existing)
        )
    for key, frame in sources.items():
        _atomic_write_csv(frame, targets[key])

    figures = {
        "robustness": _plot_robustness(sources["robustness_source"]),
        "theory": _plot_theory(sources["theory_source"]),
        "readability": _plot_readability(sources["readability_source"]),
        "overhead": _plot_overhead(sources["overhead_source"]),
        "detector": _plot_detectors(sources["detector_source"]),
    }
    try:
        for name, figure in figures.items():
            _atomic_save_figure(figure, targets[f"{name}_pdf"])
            _atomic_save_figure(figure, targets[f"{name}_png"])
    finally:
        for figure in figures.values():
            plt.close(figure)
    _atomic_write_text(command.rstrip() + "\n", targets["commands"])

    summary = {
        "figure_count": 5,
        "rendered_file_count": 10,
        "source_table_count": 5,
    }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-computational-figure-evidence-v1",
        "status": "passed",
        "inputs": {
            "robustness_manifest": {
                "path": str(robustness_manifest_path),
                "sha256": file_sha256(robustness_manifest_path),
            },
            "theory_manifest": {
                "path": str(theory_manifest_path),
                "sha256": file_sha256(theory_manifest_path),
            },
            "readability_manifest": {
                "path": str(readability_manifest_path),
                "sha256": file_sha256(readability_manifest_path),
            },
            "overhead_manifest": {
                "path": str(overhead_manifest_path),
                "sha256": file_sha256(overhead_manifest_path),
            },
            "detector_manifest": {
                "path": str(detector_manifest_path),
                "sha256": file_sha256(detector_manifest_path),
            },
        },
        "outputs": {},
        "summary": summary,
        "human_rating_figures_emitted": False,
        "readability_figure_scope": (
            "automated_surface_diagnostics_not_human_judgements"
        ),
        "source_tables_hash_validated_before_plotting": True,
        "detector_cross_split_bars": "minimum_to_maximum_range_not_confidence_interval",
        "detector_supplementary_panels_status": "exploratory_post_freeze",
    }
    for key, target in targets.items():
        if key == "manifest":
            continue
        entry = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        if key in sources:
            entry["row_count"] = len(sources[key])
        manifest["outputs"][key] = entry
    _atomic_write_json(manifest, targets["manifest"])
    return FigureArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=summary,
    )
