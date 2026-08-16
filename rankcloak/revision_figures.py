"""Data-only computational figures for the final revision evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_WIDTH_INCHES = 180.0 / 25.4
PNG_DPI = 300

FIGURE_TITLES = {
    "robustness_compact": "Exact-copy fragility",
    "robustness_full": "Exact-copy fragility",
    "detector_compact": "Neural detectability",
    "detector_full": "Neural detectability",
    "capacity_tail": "Capacity and tail overhead",
    "readability": "Automated readability diagnostics",
    "overhead_compact": "Computational overhead",
    "overhead_full": "Computational overhead",
    "ablation": "Exploratory ablation contrasts",
}

FIGURE_HEIGHT_INCHES = {
    "robustness_compact": 4.10,
    "robustness_full": 6.10,
    "detector_compact": 3.45,
    "detector_full": 5.85,
    "capacity_tail": 4.55,
    "readability": 3.75,
    "overhead_compact": 3.55,
    "overhead_full": 5.15,
    "ablation": 3.55,
}

PROHIBITED_PLOT_PHRASES = (
    "compact view",
    "complete view",
    "points and 95% intervals",
    "no pooled recovery estimate",
    "higher detection means weaker concealment",
    "similar surface scores do not establish naturalness",
    "tail tokens are not forced payload positions",
)

# Okabe-Ito-derived colors plus grayscale-distinguishable marker shapes.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
    "light_gray": "#E6E6E6",
}

OUTPUT_FILENAMES = {
    "robustness_compact_pdf": "robustness_recovery_compact.pdf",
    "robustness_compact_png": "robustness_recovery_compact.png",
    "robustness_compact_source": "robustness_compact_source.csv",
    "robustness_pdf": "robustness_recovery.pdf",
    "robustness_png": "robustness_recovery.png",
    "robustness_source": "robustness_figure_source.csv",
    "robustness_note": "robustness_technical_note.md",
    "theory_pdf": "capacity_tail_validation.pdf",
    "theory_png": "capacity_tail_validation.png",
    "theory_source": "theory_figure_source.csv",
    "theory_note": "capacity_tail_technical_note.md",
    "readability_pdf": "automated_readability.pdf",
    "readability_png": "automated_readability.png",
    "readability_source": "readability_figure_source.csv",
    "readability_note": "automated_readability_technical_note.md",
    "overhead_compact_pdf": "computational_overhead_compact.pdf",
    "overhead_compact_png": "computational_overhead_compact.png",
    "overhead_compact_source": "overhead_compact_source.csv",
    "overhead_pdf": "computational_overhead.pdf",
    "overhead_png": "computational_overhead.png",
    "overhead_source": "overhead_figure_source.csv",
    "overhead_note": "computational_overhead_technical_note.md",
    "detector_compact_pdf": "neural_detector_performance_compact.pdf",
    "detector_compact_png": "neural_detector_performance_compact.png",
    "detector_compact_source": "detector_compact_source.csv",
    "detector_pdf": "neural_detector_performance.pdf",
    "detector_png": "neural_detector_performance.png",
    "detector_source": "detector_figure_source.csv",
    "detector_note": "neural_detector_technical_note.md",
    "ablation_pdf": "ablation_summary.pdf",
    "ablation_png": "ablation_summary.png",
    "ablation_source": "ablation_figure_source.csv",
    "ablation_note": "ablation_technical_note.md",
    "inventory": "figure_inventory.csv",
    "validation": "figure_validation.json",
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
DETECTOR_COMPACT_OUTCOMES = (
    "roc_auc",
    "balanced_accuracy",
    "tpr_at_fpr_0.01",
)

MODEL_STYLES = {
    "llama3_8b_instruct_q4_k_m": ("Llama 3 8B", PALETTE["blue"], "o"),
    "mistral_7b_instruct_v0_3_q4_k_m": ("Mistral 7B", PALETTE["green"], "s"),
    "qwen2_5_7b_instruct_q4_k_m": ("Qwen 2.5 7B", PALETTE["orange"], "^"),
}

ROBUSTNESS_FAMILY_LABELS = {
    "replay_modes": "Replay channels",
    "raw_transmission": "Transmission transformations",
    "limited_mitigation": "Limited canonicalization",
    "cross_model_mismatch": "Model mismatch",
}
ROBUSTNESS_REPLAY_LABELS = {
    "saved_token_ids": "Saved token IDs",
    "detokenized_text_retokenized": "Unchanged text retokenized",
    "greedy_leadin_regeneration": "Greedy lead-in regeneration",
}
ROBUSTNESS_TRANSFORMATION_LABELS = {
    "unmodified": "Unchanged transmission text",
    "character_deletion": "Character deletion",
    "character_insertion": "Character insertion",
    "character_substitution": "Character substitution",
    "line_endings": "Line-ending conversion",
    "markdown_copy_paste": "Markdown copy/paste",
    "paraphrase": "Paraphrase",
    "quote_conversion": "Quote conversion",
    "token_deletion": "Token deletion",
    "truncation": "Final 10% tail-only truncation",
    "unicode_normalization": "Unicode normalization (NFKC)",
    "whitespace_collapse": "Whitespace collapse",
    "whitespace_trim": "Whitespace trim",
}
ROBUSTNESS_RAW_ORDER = (
    "unmodified",
    "whitespace_trim",
    "whitespace_collapse",
    "line_endings",
    "unicode_normalization",
    "quote_conversion",
    "character_insertion",
    "character_deletion",
    "character_substitution",
    "token_deletion",
    "truncation",
    "markdown_copy_paste",
    "paraphrase",
)
ROBUSTNESS_REPLAY_ORDER = (
    "saved_token_ids",
    "detokenized_text_retokenized",
    "greedy_leadin_regeneration",
)
ROBUSTNESS_LIMITED_ORDER = (
    "unmodified",
    "whitespace_trim",
    "whitespace_collapse",
    "line_endings",
    "unicode_normalization",
    "quote_conversion",
    "markdown_copy_paste",
)
ROBUSTNESS_COMPACT_KEYS = (
    ("replay_modes", "saved_token_ids", "unmodified"),
    ("replay_modes", "detokenized_text_retokenized", "unmodified"),
    ("replay_modes", "greedy_leadin_regeneration", "unmodified"),
    ("raw_transmission", "transformed_text_retokenized", "unmodified"),
    ("raw_transmission", "transformed_text_retokenized", "whitespace_trim"),
    ("raw_transmission", "transformed_text_retokenized", "unicode_normalization"),
    ("raw_transmission", "transformed_text_retokenized", "quote_conversion"),
    ("raw_transmission", "transformed_text_retokenized", "markdown_copy_paste"),
    ("raw_transmission", "transformed_text_retokenized", "truncation"),
    ("raw_transmission", "transformed_text_retokenized", "paraphrase"),
    ("cross_model_mismatch", "cross_model_text_retokenized", "unmodified"),
)

READABILITY_LABELS = {
    "ordinary_llm_control": "Ordinary LLM control",
    "rankcloak_ascii_b8": "ASCII B=8",
    "rankcloak_ascii_b16": "ASCII B=16",
    "rankcloak_hex_nibble": "Hex nibble B=16",
    "direct_subword_calgacus": "Direct subword",
    "rankcloak_segmented_forced_span": "Segmented forced span",
    "rankcloak_segmented_full_message": "Segmented full message",
}
READABILITY_GROUPS = {
    "ordinary_llm_control": "Ordinary control",
    "rankcloak_ascii_b8": "Nonsegmented RankCloak",
    "rankcloak_ascii_b16": "Nonsegmented RankCloak",
    "rankcloak_hex_nibble": "Nonsegmented RankCloak",
    "direct_subword_calgacus": "Direct subword",
    "rankcloak_segmented_forced_span": "Segmented RankCloak",
    "rankcloak_segmented_full_message": "Segmented RankCloak",
}
PROTOCOL_LABELS = {
    "direct_subword_calgacus": "Direct subword",
    "nonseg_ascii_b16": "ASCII B=16",
    "nonseg_ascii_b8": "ASCII B=8",
    "nonseg_hex_nibble_b16": "Hex nibble B=16",
    "segmented_hex_multi_topic": "Segmented multi-topic",
    "segmented_hex_single_topic": "Segmented single-topic",
}
PROTOCOL_FAMILIES = {
    "direct_subword_calgacus": "Direct subword",
    "nonseg_ascii_b16": "Nonsegmented",
    "nonseg_ascii_b8": "Nonsegmented",
    "nonseg_hex_nibble_b16": "Nonsegmented",
    "segmented_hex_multi_topic": "Segmented",
    "segmented_hex_single_topic": "Segmented",
}

ABLATION_SELECTION = (
    ("leadin_tokens", "32", "mean_log_probability", "32-token lead-in", 0),
    ("token_filter", "none", "mean_log_probability", "No token filter", 1),
    ("segment_size_ranks", "32", "effective_artifact_bits_per_full_token", "32-rank segments", 0),
    ("token_filter", "none", "effective_artifact_bits_per_full_token", "No token filter", 1),
    ("tail_policy", "none", "effective_artifact_bits_per_full_token", "No tail", 2),
    ("leadin_tokens", "32", "full_token_count", "32-token lead-in", 0),
    ("segment_size_ranks", "32", "full_token_count", "32-rank segments", 1),
    ("tail_policy", "none", "full_token_count", "No tail", 2),
    ("tail_policy", "sentence_tail_min20_max60", "full_token_count", "Fixed sentence tail", 3),
)


class FigureEvidenceError(ValueError):
    """Raised when a source manifest or plot table is inconsistent."""


@dataclass(frozen=True)
class FigureArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FigureParentRefreshArtifacts:
    evidence_summary_manifest: str
    reference_table: str
    updated_reference_count: int
    manifest_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.titlesize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _set_figure_header(
    figure: plt.Figure, figure_id: str, *, note: str | None = None
) -> None:
    title = FIGURE_TITLES[figure_id]
    figure.suptitle(
        title,
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    if note:
        figure.text(
            0.99,
            0.992,
            note,
            ha="right",
            va="top",
            fontsize=7.0,
            color=PALETTE["gray"],
            fontweight="normal",
        )


def _validate_figure_presentation(
    figures: Mapping[str, plt.Figure],
) -> None:
    figure_ids = {
        "robustness_compact": "robustness_compact",
        "robustness": "robustness_full",
        "theory": "capacity_tail",
        "readability": "readability",
        "overhead_compact": "overhead_compact",
        "overhead": "overhead_full",
        "detector_compact": "detector_compact",
        "detector": "detector_full",
        "ablation": "ablation",
    }
    if set(figures) != set(figure_ids):
        raise FigureEvidenceError("figure presentation validation set is incomplete")
    for render_key, figure in figures.items():
        figure_id = figure_ids[render_key]
        title = figure._suptitle
        if title is None or title.get_text() != FIGURE_TITLES[figure_id]:
            raise FigureEvidenceError(
                f"figure {figure_id} does not use its concise frozen title"
            )
        visible_text = "\n".join(
            artist.get_text()
            for artist in figure.findobj()
            if hasattr(artist, "get_text")
        ).lower()
        for phrase in PROHIBITED_PLOT_PHRASES:
            if phrase in visible_text:
                raise FigureEvidenceError(
                    f"figure {figure_id} contains prohibited plot phrase: {phrase}"
                )


def _portable_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise FigureEvidenceError(
            f"Figure-related path is outside the repository root: {resolved}"
        ) from exc


def _finite_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise FigureEvidenceError(f"{label} contains non-finite plotted values")


def _format_holm_p(value: float) -> str:
    if value < 0.001:
        return f"Holm p={value:.1e}"
    return f"Holm p={value:.3f}"


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
            "source_cover_units",
            "recovery_rate",
            "ci_low",
            "ci_high",
            "status",
        ),
        label="robustness plot source",
    )
    numeric = (
        "observed_outcome_rows",
        "unavailable_rows",
        "source_cover_units",
        "recovery_rate",
        "ci_low",
        "ci_high",
    )
    _finite_columns(frame, numeric, label="robustness plot source")
    if frame.duplicated(
        ["robustness_family", "replay_mode", "transformation_id"]
    ).any():
        raise FigureEvidenceError("robustness plot source contains duplicate cells")
    point = frame["recovery_rate"].to_numpy(dtype=float)
    low = frame["ci_low"].to_numpy(dtype=float)
    high = frame["ci_high"].to_numpy(dtype=float)
    _interval_errors(point, low, high, label="robustness Wilson intervals")
    if ((low < 0.0) | (high > 1.0)).any():
        raise FigureEvidenceError("robustness intervals fall outside [0, 1]")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        family = str(row["robustness_family"])
        replay = str(row["replay_mode"])
        transformation = str(row["transformation_id"])
        if family not in ROBUSTNESS_FAMILY_LABELS:
            raise FigureEvidenceError(f"Unknown robustness family: {family}")
        if family == "replay_modes":
            if replay not in ROBUSTNESS_REPLAY_LABELS:
                raise FigureEvidenceError(f"Unknown replay channel: {replay}")
            display_label = ROBUSTNESS_REPLAY_LABELS[replay]
            within_order = ROBUSTNESS_REPLAY_ORDER.index(replay)
        elif family == "raw_transmission":
            if transformation not in ROBUSTNESS_TRANSFORMATION_LABELS:
                raise FigureEvidenceError(
                    f"Unknown transmission transformation: {transformation}"
                )
            display_label = ROBUSTNESS_TRANSFORMATION_LABELS[transformation]
            within_order = ROBUSTNESS_RAW_ORDER.index(transformation)
        elif family == "limited_mitigation":
            if transformation not in ROBUSTNESS_TRANSFORMATION_LABELS:
                raise FigureEvidenceError(
                    f"Unknown canonicalization transformation: {transformation}"
                )
            display_label = (
                "Unchanged filtered text"
                if transformation == "unmodified"
                else ROBUSTNESS_TRANSFORMATION_LABELS[transformation]
            )
            within_order = ROBUSTNESS_LIMITED_ORDER.index(transformation)
        else:
            display_label = "Cross-model decoding"
            within_order = 0
        baseline = transformation == "unmodified" and family != "cross_model_mismatch"
        rows.append(
            {
                **row.to_dict(),
                "family_order": FAMILY_ORDER.index(family),
                "within_family_order": within_order,
                "display_family": ROBUSTNESS_FAMILY_LABELS[family],
                "display_label": display_label,
                "condition_role": (
                    "unchanged_or_exact_baseline" if baseline else "modified_condition"
                ),
                "interval_semantics": "source_cover_wilson_95_ci",
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["family_order", "within_family_order"]
    )
    return result.reset_index(drop=True)


def _robustness_compact_source(source: pd.DataFrame) -> pd.DataFrame:
    keyed = {
        (str(row.robustness_family), str(row.replay_mode), str(row.transformation_id)): row
        for row in source.itertuples(index=False)
    }
    missing = [key for key in ROBUSTNESS_COMPACT_KEYS if key not in keyed]
    if missing:
        raise FigureEvidenceError(
            "robustness compact selection is incomplete: " + repr(missing)
        )
    selected = pd.DataFrame([keyed[key]._asdict() for key in ROBUSTNESS_COMPACT_KEYS])
    selected["compact_order"] = np.arange(len(selected))
    return selected.reset_index(drop=True)


def _robustness_family_style(family: str, baseline: bool) -> tuple[str, str]:
    colors = {
        "replay_modes": PALETTE["blue"],
        "raw_transmission": PALETTE["vermillion"],
        "limited_mitigation": PALETTE["purple"],
        "cross_model_mismatch": PALETTE["black"],
    }
    if family == "cross_model_mismatch":
        marker = "s"
    elif baseline:
        marker = "o"
    elif family == "limited_mitigation":
        marker = "D"
    else:
        marker = "X"
    return colors[family], marker


def _draw_robustness_axis(
    axis: plt.Axes, cell: pd.DataFrame, *, title: str, panel: str
) -> None:
    ordered = cell.sort_values("within_family_order")
    y = np.arange(len(ordered))
    for position, (_, row) in enumerate(ordered.iterrows()):
        baseline = str(row["condition_role"]) == "unchanged_or_exact_baseline"
        color, marker = _robustness_family_style(
            str(row["robustness_family"]), baseline
        )
        point = np.asarray([float(row["recovery_rate"])])
        low = np.asarray([float(row["ci_low"])])
        high = np.asarray([float(row["ci_high"])])
        axis.errorbar(
            point,
            [position],
            xerr=_interval_errors(point, low, high, label=title),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.4,
            elinewidth=1.1,
            markersize=4.8,
            markeredgecolor=PALETTE["black"] if baseline else color,
            markerfacecolor="white" if baseline else color,
            zorder=3,
            clip_on=False,
        )
    axis.set_yticks(y, ordered["display_label"], fontsize=6.8)
    axis.invert_yaxis()
    axis.set_xlim(-0.045, 1.045)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6, zorder=0)
    axis.axvline(0.0, color=PALETTE["gray"], linewidth=0.6, zorder=1)
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")


def _plot_robustness(source: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["robustness_full"]),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        3, 2, width_ratios=(0.92, 1.35), height_ratios=(1.25, 2.55, 0.9)
    )
    axes = {
        "replay_modes": fig.add_subplot(grid[0, 0]),
        "limited_mitigation": fig.add_subplot(grid[1, 0]),
        "cross_model_mismatch": fig.add_subplot(grid[2, 0]),
        "raw_transmission": fig.add_subplot(grid[:, 1]),
    }
    panels = {
        "replay_modes": "A",
        "raw_transmission": "B",
        "limited_mitigation": "C",
        "cross_model_mismatch": "D",
    }
    for family in FAMILY_ORDER:
        _draw_robustness_axis(
            axes[family],
            source.loc[source["robustness_family"].eq(family)],
            title=ROBUSTNESS_FAMILY_LABELS[family],
            panel=panels[family],
        )
    axes["raw_transmission"].set_xlabel("Exact recovery probability")
    axes["cross_model_mismatch"].set_xlabel("Exact recovery probability")
    _set_figure_header(fig, "robustness_full")
    return fig


def _plot_robustness_compact(source: pd.DataFrame) -> plt.Figure:
    ordered = source.sort_values("compact_order")
    fig, axis = plt.subplots(
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["robustness_compact"]),
        constrained_layout=True,
    )
    y = np.arange(len(ordered))
    for position, (_, row) in enumerate(ordered.iterrows()):
        baseline = str(row["condition_role"]) == "unchanged_or_exact_baseline"
        color, marker = _robustness_family_style(
            str(row["robustness_family"]), baseline
        )
        point = np.asarray([float(row["recovery_rate"])])
        low = np.asarray([float(row["ci_low"])])
        high = np.asarray([float(row["ci_high"])])
        axis.errorbar(
            point,
            [position],
            xerr=_interval_errors(point, low, high, label="compact robustness"),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.5,
            elinewidth=1.2,
            markersize=5.2,
            markerfacecolor="white" if baseline else color,
            markeredgecolor=PALETTE["black"] if baseline else color,
            clip_on=False,
            zorder=3,
        )
    for boundary in (2.5, 9.5):
        axis.axhline(boundary, color="#BDBDBD", linewidth=0.7)
    axis.set_yticks(y, ordered["display_label"], fontsize=7.4)
    axis.invert_yaxis()
    axis.set_xlim(-0.045, 1.045)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.axvline(0.0, color=PALETTE["gray"], linewidth=0.6)
    axis.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6)
    axis.set_xlabel("Exact payload recovery probability")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PALETTE["black"], label="Exact/unchanged"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=PALETTE["vermillion"], markeredgecolor=PALETTE["vermillion"], label="Modified transmission"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["black"], markeredgecolor=PALETTE["black"], label="Model mismatch"),
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.8)
    _set_figure_header(fig, "robustness_compact")
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
    frame["theoretical_n_B"] = pd.to_numeric(
        frame["theoretical_n_B"], errors="coerce"
    )
    frame = frame.loc[frame["theoretical_n_B"].notna()].copy()
    _finite_columns(
        frame,
        ("theoretical_n_B", "observed_n_forced", "tail_overhead_tokens"),
        label="theory validation",
    )
    if (
        frame[["theoretical_n_B", "observed_n_forced", "tail_overhead_tokens"]]
        .lt(0)
        .any()
        .any()
    ):
        raise FigureEvidenceError("theory validation contains negative counts")
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
            }
        )
    base = pd.DataFrame(rows)
    if len(base) != 9:
        raise FigureEvidenceError(
            f"theory plot requires the nine frozen aggregate cells, observed {len(base)}"
        )
    _finite_columns(
        base,
        (
            "n",
            "mean_theoretical_n_B",
            "mean_observed_n_forced",
            "forced_residual_max_abs",
            "tail_median",
            "tail_q05",
            "tail_q95",
            "tail_max",
        ),
        label="theory aggregate source",
    )
    _interval_errors(
        base["tail_median"].to_numpy(dtype=float),
        base["tail_q05"].to_numpy(dtype=float),
        base["tail_q95"].to_numpy(dtype=float),
        label="tail 5th-95th percentile ranges",
    )
    stage_labels = {
        "primary_v2": "Primary",
        "ablation_v2": "Ablation",
        "multilingual_v2": "Multilingual",
    }
    protocol_labels = {
        "nonseg_ascii_b8": "ASCII B=8",
        "nonseg_ascii_b16": "ASCII B=16",
        "nonseg_hex_nibble_b16": "Hex nibble B=16",
        "segmented_hex_multi_topic": "Segmented multi-topic",
        "segmented_hex_single_topic": "Segmented single-topic",
    }
    tail_labels = {
        "none": "No tail",
        "dynamic_completion_v1": "Dynamic completion",
        "sentence_tail_min20_max60": "Fixed sentence tail",
    }
    base["display_stage"] = base["source_stage"].map(stage_labels)
    base["display_protocol"] = base["protocol_variant"].map(protocol_labels)
    base["display_tail"] = base["tail_policy"].map(tail_labels)
    if base[["display_stage", "display_protocol", "display_tail"]].isna().any().any():
        raise FigureEvidenceError("theory source contains an unknown display category")
    base["display_label"] = base.apply(
        lambda row: (
            f"{row['display_stage']} · {row['display_protocol']} · "
            f"{row['display_tail']}"
        ),
        axis=1,
    )
    base["display_tail_label"] = base.apply(
        lambda row: (
            f"{row['display_stage']} · "
            + {
                "segmented_hex_multi_topic": "Multi-topic",
                "segmented_hex_single_topic": "Single-topic",
            }.get(row["protocol_variant"], row["display_protocol"])
            + (
                ""
                if row["tail_policy"] == "none"
                else (
                    " · Dynamic"
                    if row["tail_policy"] == "dynamic_completion_v1"
                    else " · Fixed"
                )
            )
        ),
        axis=1,
    )
    capacity = base.copy()
    capacity["panel"] = "capacity"
    capacity["panel_order"] = np.arange(len(capacity))
    tail = base.sort_values(
        ["tail_median", "tail_q95", "source_stage", "protocol_variant"]
    ).copy()
    tail["panel"] = "tail_overhead"
    tail["panel_order"] = np.arange(len(tail))
    return pd.concat([capacity, tail], ignore_index=True)


def _plot_theory(source: pd.DataFrame) -> plt.Figure:
    capacity = source.loc[source["panel"].eq("capacity")].copy()
    tail = source.loc[source["panel"].eq("tail_overhead")].copy()
    if len(capacity) != 9 or len(tail) != 9:
        raise FigureEvidenceError("theory figure requires 9 capacity and 9 tail rows")
    fig = plt.figure(
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["capacity_tail"]),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.55),
        height_ratios=(1.0, 1.0),
    )
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    ]
    stage_styles = {
        "primary_v2": (PALETTE["blue"], "o"),
        "ablation_v2": (PALETTE["vermillion"], "s"),
        "multilingual_v2": (PALETTE["green"], "^"),
    }
    grouped_capacity = capacity.groupby(
        ["mean_theoretical_n_B", "mean_observed_n_forced"],
        sort=True,
        dropna=False,
    )
    for _, coincident in grouped_capacity:
        ordered = coincident.sort_values(
            ["source_stage", "protocol_variant", "tail_policy"]
        )
        marker_sizes = np.linspace(9.4, 4.6, len(ordered))
        for layer, ((_, row), marker_size) in enumerate(
            zip(ordered.iterrows(), marker_sizes)
        ):
            color, marker = stage_styles[str(row["source_stage"])]
            axes[0].plot(
                [float(row["mean_theoretical_n_B"])],
                [float(row["mean_observed_n_forced"])],
                marker=marker,
                markersize=float(marker_size),
                markerfacecolor="white",
                markeredgewidth=1.0,
                markeredgecolor=color,
                linestyle="none",
                alpha=0.92,
                zorder=3 + layer * 0.01,
            )
    maximum = max(
        capacity["mean_theoretical_n_B"].max(),
        capacity["mean_observed_n_forced"].max(),
    )
    axes[0].plot(
        [0, maximum * 1.05],
        [0, maximum * 1.05],
        color=PALETTE["black"],
        linestyle="--",
        linewidth=0.9,
        label="Identity",
    )
    axes[0].set_xlim(0, maximum * 1.08)
    axes[0].set_ylim(0, maximum * 1.08)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Theoretical minimum forced positions")
    axes[0].set_ylabel("Observed forced positions")
    axes[0].set_title("A  Capacity validation", loc="left", fontweight="bold")
    axes[0].grid(color=PALETTE["light_gray"], linewidth=0.6)
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="white",
            markeredgecolor=color,
            label=label,
        )
        for (stage, (color, marker)), label in zip(
            stage_styles.items(), ("Primary", "Ablation", "Multilingual")
        )
    ]
    axes[0].legend(
        handles=[
            *stage_handles,
            Line2D(
                [0],
                [0],
                color=PALETTE["black"],
                linestyle="--",
                linewidth=0.9,
                label="Identity",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=6.4,
        ncol=2,
    )
    axes[0].text(
        0.02,
        0.95,
        "All cells on identity line",
        transform=axes[0].transAxes,
        va="top",
        fontsize=6.8,
    )

    zero = tail.loc[tail["tail_median"].eq(0)].sort_values("panel_order")
    positive = tail.loc[tail["tail_median"].gt(0)].sort_values("panel_order")
    zero_y = np.arange(len(zero))
    axes[1].scatter(
        np.zeros(len(zero)),
        zero_y,
        marker="o",
        facecolors="white",
        edgecolors=PALETTE["blue"],
        s=26,
        zorder=3,
    )
    axes[1].set_yticks(zero_y, zero["display_tail_label"], fontsize=6.7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.12, 0.45)
    axes[1].set_xticks([0], ["0"])
    axes[1].set_xlabel("Tail-overhead tokens")
    axes[1].set_title("B1  Zero tail overhead", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6)

    positive_y = np.arange(len(positive))
    point = positive["tail_median"].to_numpy(dtype=float)
    low = positive["tail_q05"].to_numpy(dtype=float)
    high = positive["tail_q95"].to_numpy(dtype=float)
    axes[2].errorbar(
        point,
        positive_y,
        xerr=_interval_errors(point, low, high, label="positive tail ranges"),
        fmt="o",
        color=PALETTE["vermillion"],
        ecolor=PALETTE["vermillion"],
        capsize=2.7,
        elinewidth=1.1,
        markersize=5.0,
    )
    axes[2].set_xscale("log")
    axes[2].set_yticks(
        positive_y, positive["display_tail_label"], fontsize=6.7
    )
    axes[2].invert_yaxis()
    axes[2].set_xlim(max(10.0, float(low.min()) * 0.72), float(high.max()) * 1.25)
    axes[2].set_xlabel("Tail-overhead tokens (log scale)")
    axes[2].set_title(
        "B2  Positive tail overhead", loc="left", fontweight="bold"
    )
    axes[2].grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6)
    _set_figure_header(fig, "capacity_tail")
    return fig


def _readability_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        ("condition", "outcome", "n", "mean", "ci_low", "ci_high"),
        label="readability summary",
    )
    result = frame.loc[frame["outcome"].isin(READABILITY_OUTCOMES)].copy()
    expected = {
        (condition, outcome)
        for condition in CONDITION_ORDER
        for outcome in READABILITY_OUTCOMES
    }
    observed = set(
        result[["condition", "outcome"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(result) != len(expected):
        raise FigureEvidenceError(
            "readability source does not contain the complete 21-cell plot grid"
        )
    _finite_columns(
        result,
        ("n", "mean", "ci_low", "ci_high"),
        label="readability summary",
    )
    _interval_errors(
        result["mean"].to_numpy(dtype=float),
        result["ci_low"].to_numpy(dtype=float),
        result["ci_high"].to_numpy(dtype=float),
        label="readability confidence intervals",
    )
    if "human_rating_substitute" in result.columns and not result[
        "human_rating_substitute"
    ].astype(str).str.lower().eq("false").all():
        raise FigureEvidenceError(
            "readability source incorrectly treats automated metrics as human ratings"
        )
    if "interval_method" in result.columns and not result["interval_method"].astype(
        str
    ).eq("prompt_template_cluster_percentile_bootstrap").all():
        raise FigureEvidenceError("readability interval method is inconsistent")
    order = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    result["condition_order"] = result["condition"].map(order)
    result["display_condition"] = result["condition"].map(READABILITY_LABELS)
    result["display_group"] = result["condition"].map(READABILITY_GROUPS)
    result["interval_semantics"] = "prompt_template_cluster_bootstrap_95_ci"
    result["evidence_scope"] = "automated_surface_diagnostic_not_human_rating"
    return result.sort_values(["outcome", "condition_order"]).reset_index(drop=True)


def _plot_readability(source: pd.DataFrame) -> plt.Figure:
    labels = {
        "flesch_reading_ease_heuristic": "Flesch ease (heuristic)",
        "surface_flag_total": "Surface-flag count",
        "tfidf_prompt_similarity": "Prompt similarity (TF–IDF)",
    }
    panel_titles = {
        "flesch_reading_ease_heuristic": "A  Flesch ease heuristic",
        "surface_flag_total": "B  Surface-flag count",
        "tfidf_prompt_similarity": "C  Prompt similarity",
    }
    group_styles = {
        "Ordinary control": (PALETTE["black"], "D"),
        "Nonsegmented RankCloak": (PALETTE["blue"], "o"),
        "Direct subword": (PALETTE["orange"], "s"),
        "Segmented RankCloak": (PALETTE["green"], "^"),
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["readability"]),
        sharey=True,
        constrained_layout=True,
    )
    for axis_index, (axis, outcome) in enumerate(zip(axes, READABILITY_OUTCOMES)):
        cell = source.loc[source["outcome"].eq(outcome)].sort_values("condition_order")
        y = np.arange(len(cell))
        for position, (_, row) in enumerate(cell.iterrows()):
            color, marker = group_styles[str(row["display_group"])]
            point = np.asarray([float(row["mean"])])
            low = np.asarray([float(row["ci_low"])])
            high = np.asarray([float(row["ci_high"])])
            axis.errorbar(
                point,
                [position],
                xerr=_interval_errors(
                    point, low, high, label=f"readability {outcome}"
                ),
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=2.2,
                elinewidth=1.0,
                markersize=4.8,
                markerfacecolor=(
                    color if row["condition"] == "ordinary_llm_control" else "white"
                ),
                markeredgecolor=color,
                markeredgewidth=1.0,
                zorder=3,
            )
        axis.set_yticks(y, cell["display_condition"], fontsize=7.0)
        axis.invert_yaxis()
        axis.set_xlabel(labels[outcome])
        axis.set_title(panel_titles[outcome], loc="left", fontweight="bold")
        axis.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6)
        axis.set_xlim(left=0.0)
        for boundary in (0.5, 3.5, 4.5):
            axis.axhline(boundary, color="#C8C8C8", linewidth=0.6, zorder=0)
        if axis_index > 0:
            axis.tick_params(axis="y", labelleft=False)
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=(color if group == "Ordinary control" else "white"),
            markeredgecolor=color,
            label=group,
        )
        for group, (color, marker) in group_styles.items()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "readability",
        note="Not human ratings",
    )
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
    _finite_columns(
        result,
        ("mean", "ci_low", "ci_high", "n_payloads"),
        label="overhead plot source",
    )
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
    _interval_errors(
        result["mean"].to_numpy(dtype=float),
        result["ci_low"].to_numpy(dtype=float),
        result["ci_high"].to_numpy(dtype=float),
        label="overhead confidence intervals",
    )
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
    result["display_protocol"] = result["protocol_variant"].map(PROTOCOL_LABELS)
    result["display_family"] = result["protocol_variant"].map(PROTOCOL_FAMILIES)
    result["display_model"] = result["model_id"].map(OVERHEAD_MODEL_LABELS)
    result["interval_semantics"] = "payload_group_bootstrap_95_ci"
    result["timing_scope_note"] = "inclusive_wrapper_measurement"
    return result.sort_values(["outcome", "row_order"]).reset_index(drop=True)


def _overhead_compact_source(source: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[
        source["outcome"].isin(
            ("generation_seconds", "decoding_overhead_seconds", "payload_bits_per_second")
        )
    ].copy()
    if len(selected) != 54:
        raise FigureEvidenceError(
            f"compact overhead source requires 54 rows, observed {len(selected)}"
        )
    return selected.reset_index(drop=True)


def _draw_overhead_panel(
    axis: plt.Axes,
    source: pd.DataFrame,
    *,
    outcome: str,
    title: str,
    show_labels: bool,
) -> None:
    cell = source.loc[source["outcome"].eq(outcome)].copy()
    expected = len(OVERHEAD_PROTOCOL_ORDER) * len(OVERHEAD_MODEL_LABELS)
    if len(cell) != expected:
        raise FigureEvidenceError(
            f"overhead plot source has incomplete rows for {outcome}"
        )
    protocol_y = np.arange(len(OVERHEAD_PROTOCOL_ORDER), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    for (model_id, (label, color, marker)), offset in zip(
        MODEL_STYLES.items(), offsets
    ):
        model_cell = cell.loc[cell["model_id"].eq(model_id)].sort_values(
            "protocol_order"
        )
        point = model_cell["mean"].to_numpy(dtype=float)
        low = model_cell["ci_low"].to_numpy(dtype=float)
        high = model_cell["ci_high"].to_numpy(dtype=float)
        axis.errorbar(
            point,
            protocol_y + offset,
            xerr=_interval_errors(
                point, low, high, label=f"overhead {outcome} {model_id}"
            ),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.0,
            elinewidth=0.9,
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            label=label,
            zorder=3,
        )
    axis.axhspan(-0.5, 0.5, color=PALETTE["orange"], alpha=0.06, zorder=0)
    axis.axhspan(0.5, 3.5, color=PALETTE["blue"], alpha=0.045, zorder=0)
    axis.axhspan(3.5, 5.5, color=PALETTE["green"], alpha=0.05, zorder=0)
    for boundary in (0.5, 3.5):
        axis.axhline(boundary, color="#BDBDBD", linewidth=0.6, zorder=1)
    labels = [PROTOCOL_LABELS[value] for value in OVERHEAD_PROTOCOL_ORDER]
    axis.set_yticks(protocol_y, labels if show_labels else [])
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6, zorder=0)
    if outcome == "encoding_overhead_seconds":
        if (cell["ci_low"] <= 0).any():
            raise FigureEvidenceError(
                "encoding-overhead confidence bounds must be positive for log scale"
            )
        axis.set_xscale("log")
        axis.set_xlabel("Seconds (log scale)")
    elif outcome == "payload_bits_per_second":
        axis.set_xlim(left=0.0)
        axis.set_xlabel("Payload bits per second")
    else:
        axis.set_xlim(left=0.0)
        axis.set_xlabel("Seconds")


def _overhead_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="white",
            markeredgecolor=color,
            label=label,
        )
        for label, color, marker in MODEL_STYLES.values()
    ]


def _plot_overhead(source: pd.DataFrame) -> plt.Figure:
    titles = {
        "generation_seconds": "A  Generation time",
        "encoding_overhead_seconds": "B  Encoding setup",
        "decoding_overhead_seconds": "C  Decoding wrapper",
        "payload_bits_per_second": "D  Payload throughput",
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["overhead_full"]),
        constrained_layout=True,
    )
    for index, (axis, outcome) in enumerate(zip(axes.flat, OVERHEAD_OUTCOMES)):
        _draw_overhead_panel(
            axis,
            source,
            outcome=outcome,
            title=titles[outcome],
            show_labels=index % 2 == 0,
        )
    fig.legend(
        handles=_overhead_legend_handles(),
        frameon=False,
        ncol=3,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "overhead_full",
        note="Inclusive wrapper timing",
    )
    return fig


def _plot_overhead_compact(source: pd.DataFrame) -> plt.Figure:
    outcomes = (
        "generation_seconds",
        "payload_bits_per_second",
        "decoding_overhead_seconds",
    )
    titles = {
        "generation_seconds": "A  Generation",
        "payload_bits_per_second": "B  Payload throughput",
        "decoding_overhead_seconds": "C  Decoding wrapper",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["overhead_compact"]),
        constrained_layout=True,
    )
    for index, (axis, outcome) in enumerate(zip(axes, outcomes)):
        _draw_overhead_panel(
            axis,
            source,
            outcome=outcome,
            title=titles[outcome],
            show_labels=index == 0,
        )
    fig.legend(
        handles=_overhead_legend_handles(),
        frameon=False,
        ncol=3,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "overhead_compact",
        note="Inclusive wrapper timing",
    )
    return fig


def _detector_source(path: Path, metrics_path: Path) -> pd.DataFrame:
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
    regime_labels = {
        "matched": "Matched",
        "held_out_template": "Held-out\ntemplate",
        "leave_one_model": "Leave-one\nmodel",
        "leave_one_codec": "Leave-one\ncodec",
    }
    frame["display_regime"] = frame["regime"].map(regime_labels)

    metrics = pd.read_csv(metrics_path, low_memory=False)
    _require_columns(
        metrics,
        (
            "split_id",
            "detector_name",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "confidence_level",
            "evidence_status",
        ),
        label="detector extended metrics",
    )
    matched = metrics.loc[
        metrics["split_id"].eq("matched")
        & metrics["detector_name"].isin(DETECTOR_ORDER)
        & metrics["metric"].isin(DETECTOR_OUTCOMES)
    ].copy()
    if len(matched) != len(DETECTOR_ORDER) * len(DETECTOR_OUTCOMES):
        raise FigureEvidenceError(
            "detector extended metrics lack the 12 matched interval rows"
        )
    _finite_columns(
        matched,
        ("estimate", "ci_low", "ci_high", "confidence_level"),
        label="matched detector intervals",
    )
    _interval_errors(
        matched["estimate"].to_numpy(dtype=float),
        matched["ci_low"].to_numpy(dtype=float),
        matched["ci_high"].to_numpy(dtype=float),
        label="matched detector confidence intervals",
    )
    matched_lookup = {
        (str(row.detector_name), str(row.metric)): row
        for row in matched.itertuples(index=False)
    }
    interval_low: list[float] = []
    interval_high: list[float] = []
    interval_semantics: list[str] = []
    for row in frame.itertuples(index=False):
        if row.regime == "matched":
            match = matched_lookup[(str(row.detector_name), str(row.metric))]
            if not np.isclose(
                float(row.estimate_median_across_splits),
                float(match.estimate),
                atol=1e-12,
                rtol=0.0,
            ):
                raise FigureEvidenceError(
                    "matched detector point differs between authoritative tables"
                )
            interval_low.append(float(match.ci_low))
            interval_high.append(float(match.ci_high))
            interval_semantics.append("payload_group_bootstrap_95_ci")
        else:
            interval_low.append(float(row.estimate_min_across_splits))
            interval_high.append(float(row.estimate_max_across_splits))
            interval_semantics.append("heterogeneous_split_min_max_range")
    frame["interval_low"] = interval_low
    frame["interval_high"] = interval_high
    frame["interval_semantics"] = interval_semantics
    _interval_errors(
        frame["estimate_median_across_splits"].to_numpy(dtype=float),
        frame["interval_low"].to_numpy(dtype=float),
        frame["interval_high"].to_numpy(dtype=float),
        label="detector displayed intervals",
    )
    return frame.sort_values(
        ["outcome_order", "regime_order", "detector_order"]
    ).reset_index(drop=True)


def _detector_compact_source(source: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[source["metric"].isin(DETECTOR_COMPACT_OUTCOMES)].copy()
    if len(selected) != 24:
        raise FigureEvidenceError(
            f"compact detector source requires 24 rows, observed {len(selected)}"
        )
    return selected.reset_index(drop=True)


def _draw_detector_panel(
    axis: plt.Axes, source: pd.DataFrame, *, outcome: str, title: str
) -> None:
    colors = {
        "published_textcnn_equivalent": PALETTE["blue"],
        "deberta_v3_base_classifier": PALETTE["vermillion"],
    }
    markers = {
        "published_textcnn_equivalent": "o",
        "deberta_v3_base_classifier": "s",
    }
    x = np.arange(len(DETECTOR_REGIME_ORDER), dtype=float)
    offsets = (-0.10, 0.10)
    for detector, offset in zip(DETECTOR_ORDER, offsets):
        cell = source.loc[
            source["metric"].eq(outcome)
            & source["detector_name"].eq(detector)
        ].sort_values("regime_order")
        if len(cell) != len(DETECTOR_REGIME_ORDER):
            raise FigureEvidenceError(
                f"detector plot source has incomplete rows for {outcome}"
            )
        positions = x + offset
        matched = cell["regime"].eq("matched").to_numpy()
        held_out = ~matched
        point = cell["estimate_median_across_splits"].to_numpy(dtype=float)
        low = cell["interval_low"].to_numpy(dtype=float)
        high = cell["interval_high"].to_numpy(dtype=float)
        axis.errorbar(
            positions[matched],
            point[matched],
            yerr=_interval_errors(
                point[matched],
                low[matched],
                high[matched],
                label=f"matched {detector} {outcome}",
            ),
            fmt=markers[detector],
            color=colors[detector],
            ecolor=colors[detector],
            capsize=3.0,
            elinewidth=1.5,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=4,
            clip_on=False,
        )
        axis.vlines(
            positions[held_out],
            low[held_out],
            high[held_out],
            colors=colors[detector],
            linestyles=(0, (2.0, 1.7)),
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            positions[held_out],
            point[held_out],
            marker=markers[detector],
            facecolors=colors[detector],
            edgecolors=colors[detector],
            s=20,
            zorder=4,
            clip_on=False,
        )
        axis.scatter(
            np.repeat(positions[held_out], 2),
            np.column_stack([low[held_out], high[held_out]]).reshape(-1),
            marker="_",
            color=colors[detector],
            s=23,
            linewidths=0.9,
            zorder=3,
        )
    if outcome == "brier_score":
        axis.set_ylim(-0.012, 0.46)
        axis.set_yticks(np.arange(0.0, 0.41, 0.1))
    else:
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.6)
    axis.set_xticks(
        x,
        [
            "Matched",
            "Held-out\ntemplate",
            "Leave-one-\nmodel",
            "Leave-one-\ncodec",
        ],
    )
    axis.tick_params(axis="x", labelsize=6.8)


def _detector_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color=PALETTE["blue"], markerfacecolor="white", linestyle="none", label="TextCNN"),
        Line2D([0], [0], marker="s", color=PALETTE["vermillion"], markerfacecolor="white", linestyle="none", label="DeBERTa-v3-base"),
        Line2D([0], [0], color=PALETTE["black"], linewidth=1.5, marker="|", markersize=8, label="Matched: 95% CI"),
        Line2D([0], [0], color=PALETTE["black"], linewidth=1.0, linestyle=(0, (2.0, 1.7)), label="Held-out: min–max"),
    ]


def _plot_detectors(source: pd.DataFrame) -> plt.Figure:
    titles = {
        "roc_auc": "A  ROC–AUC",
        "pr_auc": "B  PR–AUC",
        "balanced_accuracy": "C  Balanced accuracy",
        "precision": "D  Precision at 0.5",
        "brier_score": "E  Brier score: lower is better",
        "tpr_at_fpr_0.01": "F  TPR at FPR ≤ 1%",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["detector_full"]),
        constrained_layout=True,
    )
    for axis, outcome in zip(axes.flat, DETECTOR_OUTCOMES):
        _draw_detector_panel(axis, source, outcome=outcome, title=titles[outcome])
    fig.legend(
        handles=_detector_legend_handles(),
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "detector_full",
        note="Higher = easier detection · D–F exploratory",
    )
    return fig


def _plot_detectors_compact(source: pd.DataFrame) -> plt.Figure:
    titles = {
        "roc_auc": "A  ROC–AUC",
        "balanced_accuracy": "B  Balanced accuracy",
        "tpr_at_fpr_0.01": "C  TPR at FPR ≤ 1%",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["detector_compact"]),
        constrained_layout=True,
    )
    for axis, outcome in zip(axes, DETECTOR_COMPACT_OUTCOMES):
        _draw_detector_panel(axis, source, outcome=outcome, title=titles[outcome])
    fig.legend(
        handles=_detector_legend_handles(),
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "detector_compact",
        note="Higher = easier detection · C exploratory",
    )
    return fig


def _normalized_level(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _ablation_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    _require_columns(
        frame,
        (
            "factor",
            "level",
            "canonical_value",
            "outcome",
            "shared_models",
            "paired_payload_groups",
            "level_minus_canonical",
            "ci_low",
            "ci_high",
            "p_value_holm",
            "inferential_p_value_supported",
            "bootstrap_unit",
            "bootstrap_resamples",
            "confidence_level",
            "primary_inference",
            "evidence_status",
        ),
        label="ablation canonical contrasts",
    )
    if len(frame) != 60:
        raise FigureEvidenceError(
            f"ablation source must retain the full 60-row evidence table, observed {len(frame)}"
        )
    frame["normalized_level"] = frame["level"].map(_normalized_level)
    if not frame["evidence_status"].astype(str).eq(
        "exploratory_post_outcome_evidence_extraction"
    ).all():
        raise FigureEvidenceError("ablation source evidence status is inconsistent")
    if frame["primary_inference"].astype(str).str.lower().eq("true").any():
        raise FigureEvidenceError("ablation source incorrectly claims primary inference")
    selected_rows: list[dict[str, Any]] = []
    for factor, level, outcome, label, panel_order in ABLATION_SELECTION:
        cell = frame.loc[
            frame["factor"].eq(factor)
            & frame["normalized_level"].eq(level)
            & frame["outcome"].eq(outcome)
        ]
        if len(cell) != 1:
            raise FigureEvidenceError(
                "ablation compact selection has missing or duplicate row: "
                f"{factor}/{level}/{outcome}"
            )
        row = cell.iloc[0].to_dict()
        row.update(
            {
                "display_label": label,
                "panel_order": panel_order,
                "estimand_type": "raw_difference_level_minus_canonical",
                "reference_value": 0.0,
                "interval_semantics": "payload_group_bootstrap_95_ci",
                "holm_label": _format_holm_p(float(row["p_value_holm"])),
            }
        )
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    _finite_columns(
        selected,
        (
            "paired_payload_groups",
            "level_minus_canonical",
            "ci_low",
            "ci_high",
            "p_value_holm",
            "bootstrap_resamples",
            "confidence_level",
            "reference_value",
        ),
        label="selected ablation contrasts",
    )
    _interval_errors(
        selected["level_minus_canonical"].to_numpy(dtype=float),
        selected["ci_low"].to_numpy(dtype=float),
        selected["ci_high"].to_numpy(dtype=float),
        label="ablation confidence intervals",
    )
    if not selected["paired_payload_groups"].eq(48).all():
        raise FigureEvidenceError("ablation selected rows do not use 48 payload groups")
    if not selected["bootstrap_resamples"].eq(2000).all():
        raise FigureEvidenceError("ablation selected rows do not use 2,000 bootstraps")
    if not selected["confidence_level"].eq(0.95).all():
        raise FigureEvidenceError("ablation selected rows do not use 95% intervals")
    if not selected["inferential_p_value_supported"].astype(str).str.lower().eq(
        "true"
    ).all():
        raise FigureEvidenceError(
            "ablation compact selection includes unsupported inferential p-values"
        )
    outcome_order = {
        "mean_log_probability": 0,
        "effective_artifact_bits_per_full_token": 1,
        "full_token_count": 2,
    }
    selected["outcome_order"] = selected["outcome"].map(outcome_order)
    return selected.sort_values(["outcome_order", "panel_order"]).reset_index(drop=True)


def _plot_ablation(source: pd.DataFrame) -> plt.Figure:
    outcomes = (
        "mean_log_probability",
        "effective_artifact_bits_per_full_token",
        "full_token_count",
    )
    titles = {
        "mean_log_probability": "A  Token log probability",
        "effective_artifact_bits_per_full_token": "B  Payload rate",
        "full_token_count": "C  Message length",
    }
    labels = {
        "mean_log_probability": "Difference in mean log probability",
        "effective_artifact_bits_per_full_token": "Difference (bits per full token)",
        "full_token_count": "Difference in tokens",
    }
    factor_styles = {
        "leadin_tokens": (PALETTE["vermillion"], "v"),
        "segment_size_ranks": (PALETTE["blue"], "s"),
        "token_filter": (PALETTE["gray"], "D"),
        "tail_policy": (PALETTE["green"], "o"),
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES["ablation"]),
        constrained_layout=True,
    )
    for axis, outcome in zip(axes, outcomes):
        cell = source.loc[source["outcome"].eq(outcome)].sort_values("panel_order")
        y = (4.0 - len(cell)) / 2.0 + np.arange(len(cell), dtype=float)
        for position, (_, row) in zip(y, cell.iterrows()):
            color, marker = factor_styles[str(row["factor"])]
            point = np.asarray([float(row["level_minus_canonical"])])
            low = np.asarray([float(row["ci_low"])])
            high = np.asarray([float(row["ci_high"])])
            axis.errorbar(
                point,
                [position],
                xerr=_interval_errors(
                    point, low, high, label=f"ablation {outcome}"
                ),
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=2.5,
                elinewidth=1.1,
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=color,
                zorder=3,
            )
        axis.set_yticks(y, cell["display_label"], fontsize=6.9)
        axis.set_ylim(3.5, -0.5)
        axis.axvline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.9)
        axis.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.6)
        minimum = min(float(cell["ci_low"].min()), 0.0)
        maximum = max(float(cell["ci_high"].max()), 0.0)
        span = max(maximum - minimum, 1e-6)
        axis.set_xlim(minimum - 0.08 * span, maximum + 0.08 * span)
        axis.set_title(titles[outcome], loc="left", fontweight="bold")
        axis.set_xlabel(labels[outcome])
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="white",
            markeredgecolor=color,
            label=label,
        )
        for label, (color, marker) in zip(
            ("Lead-in", "Segment length", "Token filter", "Tail policy"),
            factor_styles.values(),
        )
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="outside lower center",
    )
    _set_figure_header(
        fig,
        "ablation",
        note="Post-outcome",
    )
    return fig


def _ablation_pvalue_note_lines(source: pd.DataFrame) -> list[str]:
    outcome_labels = {
        "mean_log_probability": "Token log probability",
        "effective_artifact_bits_per_full_token": "Payload rate",
        "full_token_count": "Message length",
    }
    lines = [
        "",
        "## Holm-adjusted p-values for plotted rows",
        "",
        "| Outcome | Contrast | Holm-adjusted p-value |",
        "|---|---|---:|",
    ]
    for row in source.sort_values(["outcome_order", "panel_order"]).itertuples(
        index=False
    ):
        lines.append(
            f"| {outcome_labels[str(row.outcome)]} | {row.display_label} | "
            f"`{float(row.p_value_holm):.17g}` |"
        )
    return lines


def _technical_notes(
    upstream_paths: Mapping[str, str], ablation_source: pd.DataFrame
) -> dict[str, str]:
    return {
        "robustness_note": "\n".join(
            [
                "# Robustness figure technical note",
                "",
                f"- Authoritative source: `{upstream_paths['robustness']}`.",
                "- Evidence classification: secondary evidence with diagnostic scope; 11 rows in the compact core candidate and 24 rows in the full supporting candidate.",
                "- Analysis unit: source cover; intervals: Wilson 95% confidence intervals.",
                "- Replay, raw-transmission, limited-canonicalization, and cross-model channels remain separate; no pooled recovery estimate is plotted.",
                "- `Final 10% tail-only truncation` removes `ceil(10%)` of final token IDs. It does not test arbitrary truncation of payload-bearing positions.",
                "- Limited canonicalization has partial availability: 96 observed and 48 unavailable source-cover units per displayed cell.",
                "- Untested requested classes: isolated punctuation changes, arbitrary prefixes/suffixes, and case conversion.",
                "- Partially represented classes: line wrapping (line-ending proxy only), email quoting (Markdown blockquote proxy only), and isolated sentence-boundary edits (paraphrase proxy only).",
                "- Failure-mechanism labels in the upstream evidence are descriptive first-divergence categories, not causal proof.",
                "",
            ]
        ),
        "detector_note": "\n".join(
            [
                "# Neural-detector figure technical note",
                "",
                f"- Regime source: `{upstream_paths['detector_plot']}`.",
                f"- Matched confidence-interval source: `{upstream_paths['detector_metrics']}`.",
                "- Evidence classification: confirmatory frozen endpoints plus explicitly exploratory post-freeze endpoints; 24 rows in the compact candidate and 48 rows in the full supporting candidate.",
                "- Matched bars are payload-group bootstrap 95% confidence intervals.",
                "- Held-out bars are minimum-to-maximum ranges across heterogeneous prespecified splits; they are not confidence intervals.",
                "- ROC-AUC, PR-AUC, and balanced accuracy retain confirmatory frozen-upstream status.",
                "- Precision, Brier score, and TPR at FPR <= 1% are supplementary exploratory post-freeze metrics.",
                "- Brier score is lower-is-better. High values in the other detector panels indicate weaker concealment, not a favorable RankCloak outcome.",
                "- The lexical near-duplicate sensitivity analysis is not plotted here and does not remove the underlying limitation.",
                "",
            ]
        ),
        "readability_note": "\n".join(
            [
                "# Automated-readability figure technical note",
                "",
                f"- Authoritative source: `{upstream_paths['readability']}`.",
                "- Evidence classification: supporting automated surface diagnostics, not human evaluation; 21 plotted rows (seven conditions by three outcomes).",
                "- All plotted rows have `human_rating_substitute=false`.",
                "- Intervals are 95% prompt-template-cluster percentile-bootstrap intervals (18 prompt-template units; 72 stimuli per condition).",
                "- `Surface-flag count` sums unmatched brackets, double-quote imbalance, repeated punctuation, whitespace flags, lowercase sentence starts, sentences longer than 40 words, missing terminal punctuation, and long hexadecimal/base64-like fragments.",
                "- `Prompt similarity` is TF-IDF cosine similarity between cover text and prompt text.",
                "- Flesch ease is an English surface heuristic. Similar values do not establish naturalness or human-perceived quality.",
                "- No composite naturalness score or synthetic human score is computed.",
                "",
            ]
        ),
        "theory_note": "\n".join(
            [
                "# Capacity and tail-overhead technical note",
                "",
                f"- Authoritative source: `{upstream_paths['theory']}`.",
                "- Evidence classification: supporting evidence; nine aggregate cells appear in both the capacity and tail panels (18 plotted-source rows).",
                "- Capacity relationship: the theoretical forced-position requirement is the saved `theoretical_n_B` value derived from payload bits and the declared rank alphabet; it is compared directly with `observed_n_forced`.",
                "- All nine aggregate capacity cells lie on the identity relation; maximum absolute forced-position residual is zero.",
                "- Every capacity marker is plotted at its true data coordinates. Coincident cells are shown with concentric marker sizes, stage-specific shapes, transparent fills, and controlled z-order; no x, y, screen-space, pixel, point, or transform displacement is applied.",
                "- Tail overhead is `tail_overhead_tokens`: cover-extension tokens beyond forced payload positions.",
                "- Exact-zero tail cells are displayed on a separate linear panel. Positive medians and 5th-95th percentile ranges use a logarithmic axis.",
                "- Points are medians; bars are 5th-95th percentile ranges, not confidence intervals.",
                "",
            ]
        ),
        "overhead_note": "\n".join(
            [
                "# Computational-overhead figure technical note",
                "",
                f"- Authoritative source: `{upstream_paths['overhead']}`.",
                "- Evidence classification: supporting computational evidence; 54 rows in the compact candidate and 72 rows in the full supporting candidate (18 model-protocol cells per panel).",
                "- Plotted cells are primary-stage, trial-scope, payload-group summaries with 95% payload-bootstrap intervals.",
                "- Timing fields are inclusive wrapper measurements; encoding, generation, and supported decoding are not asserted to be perfectly isolated.",
                "- Encoding setup uses a log axis only in the complete figure because values span several orders of magnitude.",
                "- CPU time and repeated warm-up measurements were unavailable; wall time is not substituted for CPU time.",
                "- Saved RAM/VRAM scopes are limited, and no kernel-exact VRAM peak is claimed.",
                "",
            ]
        ),
        "ablation_note": "\n".join(
            [
                "# Ablation-summary technical note",
                "",
                f"- Authoritative 60-row source: `{upstream_paths['ablation']}`.",
                "- Evidence classification: exploratory post-outcome evidence extraction; `primary_inference=false`. Nine prespecified compact rows are plotted from the full 60-row table.",
                "- The nine plotted rows are the compact contrasts already identified in the evidence records: 32-token lead-in, 32-rank segments, no filter, no tail, and fixed sentence tail for the stated outcomes.",
                "- Every plotted estimand is a raw level-minus-canonical difference with a zero reference line. Hedges g is retained upstream but is not mixed onto the plotted scales; no ratio estimand is plotted.",
                "- Bars are 95% payload-group bootstrap intervals (2,000 resamples). Exact Holm-adjusted p-values are retained below and in the plotted-source CSV; significance stars are not used.",
                "- Null token-filter results are retained. The round-trip-stable filter cell was unavailable for 48 Mistral work units and is not treated as a recovery failure.",
            ]
            + _ablation_pvalue_note_lines(ablation_source)
            + [""]
        ),
    }


def _figure_specs() -> dict[str, dict[str, Any]]:
    return {
        "robustness_compact": {
            "title": FIGURE_TITLES["robustness_compact"],
            "classification": "Core compact candidate",
            "evidence_status": "secondary_with_diagnostic_scope",
            "panel_count": 1,
            "plotted_row_count": 11,
            "source_key": "robustness_compact_source",
            "pdf_key": "robustness_compact_pdf",
            "png_key": "robustness_compact_png",
            "note_key": "robustness_note",
            "uncertainty": "source-cover Wilson 95% confidence intervals",
        },
        "robustness_full": {
            "title": FIGURE_TITLES["robustness_full"],
            "classification": "Full supporting candidate",
            "evidence_status": "secondary_with_diagnostic_scope",
            "panel_count": 4,
            "plotted_row_count": 24,
            "source_key": "robustness_source",
            "pdf_key": "robustness_pdf",
            "png_key": "robustness_png",
            "note_key": "robustness_note",
            "uncertainty": "source-cover Wilson 95% confidence intervals",
        },
        "detector_compact": {
            "title": FIGURE_TITLES["detector_compact"],
            "classification": "Core compact candidate",
            "evidence_status": "confirmatory_endpoints_plus_exploratory_low_fpr_metric",
            "panel_count": 3,
            "plotted_row_count": 24,
            "source_key": "detector_compact_source",
            "pdf_key": "detector_compact_pdf",
            "png_key": "detector_compact_png",
            "note_key": "detector_note",
            "uncertainty": "matched 95% CIs; held-out cross-split min-max ranges",
        },
        "detector_full": {
            "title": FIGURE_TITLES["detector_full"],
            "classification": "Full supporting candidate",
            "evidence_status": "confirmatory_and_exploratory_panels_separated",
            "panel_count": 6,
            "plotted_row_count": 48,
            "source_key": "detector_source",
            "pdf_key": "detector_pdf",
            "png_key": "detector_png",
            "note_key": "detector_note",
            "uncertainty": "matched 95% CIs; held-out cross-split min-max ranges",
        },
        "capacity_tail": {
            "title": FIGURE_TITLES["capacity_tail"],
            "classification": "Core compact candidate",
            "evidence_status": "supporting",
            "panel_count": 3,
            "plotted_row_count": 18,
            "source_key": "theory_source",
            "pdf_key": "theory_pdf",
            "png_key": "theory_png",
            "note_key": "theory_note",
            "uncertainty": "tail medians with 5th-95th percentile ranges",
        },
        "readability": {
            "title": FIGURE_TITLES["readability"],
            "classification": "Full supporting candidate",
            "evidence_status": "automated_not_human_rating",
            "panel_count": 3,
            "plotted_row_count": 21,
            "source_key": "readability_source",
            "pdf_key": "readability_pdf",
            "png_key": "readability_png",
            "note_key": "readability_note",
            "uncertainty": "prompt-template-cluster bootstrap 95% confidence intervals",
        },
        "overhead_compact": {
            "title": FIGURE_TITLES["overhead_compact"],
            "classification": "Full supporting candidate",
            "evidence_status": "supporting",
            "panel_count": 3,
            "plotted_row_count": 54,
            "source_key": "overhead_compact_source",
            "pdf_key": "overhead_compact_pdf",
            "png_key": "overhead_compact_png",
            "note_key": "overhead_note",
            "uncertainty": "payload-group bootstrap 95% confidence intervals",
        },
        "overhead_full": {
            "title": FIGURE_TITLES["overhead_full"],
            "classification": "Full supporting candidate",
            "evidence_status": "supporting",
            "panel_count": 4,
            "plotted_row_count": 72,
            "source_key": "overhead_source",
            "pdf_key": "overhead_pdf",
            "png_key": "overhead_png",
            "note_key": "overhead_note",
            "uncertainty": "payload-group bootstrap 95% confidence intervals",
        },
        "ablation": {
            "title": FIGURE_TITLES["ablation"],
            "classification": "Full supporting candidate",
            "evidence_status": "exploratory_post_outcome",
            "panel_count": 3,
            "plotted_row_count": 9,
            "source_key": "ablation_source",
            "pdf_key": "ablation_pdf",
            "png_key": "ablation_png",
            "note_key": "ablation_note",
            "uncertainty": "payload-group bootstrap 95% confidence intervals",
        },
    }


def _figure_inventory(
    targets: Mapping[str, Path], project_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure_id, spec in _figure_specs().items():
        rows.append(
            {
                "figure_id": figure_id,
                "title": spec["title"],
                "classification": spec["classification"],
                "evidence_status": spec["evidence_status"],
                "panel_count": spec["panel_count"],
                "plotted_row_count": spec["plotted_row_count"],
                "width_mm": 180.0,
                "height_mm": round(FIGURE_HEIGHT_INCHES[figure_id] * 25.4, 2),
                "primary_pdf": _portable_path(targets[spec["pdf_key"]], project_root),
                "inspection_png_300dpi": _portable_path(
                    targets[spec["png_key"]], project_root
                ),
                "plotted_source_csv": _portable_path(
                    targets[spec["source_key"]], project_root
                ),
                "technical_note": _portable_path(
                    targets[spec["note_key"]], project_root
                ),
                "uncertainty": spec["uncertainty"],
            }
        )
    return pd.DataFrame(rows)


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


def _atomic_save_figure(
    figure: plt.Figure, target: Path, *, dpi: int = PNG_DPI
) -> Path:
    temporary = target.with_name(
        f".{target.stem}.tmp-{uuid.uuid4().hex}{target.suffix}"
    )
    metadata = {"Creator": "RankCloak computational evidence pipeline"}
    if target.suffix.lower() == ".pdf":
        metadata.update({"CreationDate": None, "ModDate": None})
    figure.savefig(
        temporary,
        dpi=dpi,
        metadata=metadata,
        facecolor="white",
    )
    os.replace(temporary, target)
    return target


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def _source_row_count(path: Path) -> int | None:
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle, delimiter=delimiter)) - 1, 0)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


def refresh_evidence_summary_figure_hashes(
    *,
    project_root: str | Path,
    evidence_summary_manifest: str | Path,
    reference_table: str | Path,
    figure_dir: str | Path,
    figure_manifest: str | Path,
    refresh_head: str,
    command: str,
) -> FigureParentRefreshArtifacts:
    """Refresh only existing figure identities in a sealed evidence summary."""

    root = Path(project_root).resolve()
    if str(root) in command or command.lstrip().startswith("/"):
        raise FigureEvidenceError(
            "Figure-parent refresh command must be repository-relative and portable"
        )
    manifest_path = Path(evidence_summary_manifest).resolve()
    table_path = Path(reference_table).resolve()
    figures_path = Path(figure_dir).resolve()
    figure_manifest_path = Path(figure_manifest).resolve()
    for path in (manifest_path, table_path, figures_path, figure_manifest_path):
        _portable_path(path, root)
    if not figures_path.is_dir():
        raise FigureEvidenceError(f"Missing canonical figure directory: {figures_path}")
    if figure_manifest_path.parent != figures_path or not figure_manifest_path.is_file():
        raise FigureEvidenceError(
            "Canonical figure manifest must exist directly under the figure directory"
        )

    manifest = _read_json(manifest_path, label="evidence summary manifest")
    signature = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if signature != canonical_json_sha256(unsigned):
        raise FigureEvidenceError("Evidence summary manifest self-hash mismatch")
    outputs = manifest.get("outputs")
    table_key = "tables/evidence_artifact_references.csv"
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(table_key), Mapping
    ):
        raise FigureEvidenceError(
            "Evidence summary manifest lacks the artifact-reference output"
        )
    table_entry = outputs[table_key]
    if _portable_path(table_path, root) != table_entry.get("path"):
        raise FigureEvidenceError("Artifact-reference table path differs from manifest")
    if file_sha256(table_path) != table_entry.get("sha256"):
        raise FigureEvidenceError(
            "Artifact-reference table is already stale relative to its manifest"
        )

    frame = pd.read_csv(table_path, dtype=str, keep_default_na=False)
    _require_columns(
        frame,
        ("path", "sha256", "size_bytes", "row_count"),
        label="evidence artifact references",
    )
    if frame["path"].duplicated().any():
        raise FigureEvidenceError("Evidence artifact references contain duplicate paths")
    figure_prefix = _portable_path(figures_path, root).rstrip("/") + "/"
    selected = frame["path"].str.startswith(figure_prefix)
    if not selected.any():
        raise FigureEvidenceError(
            "Evidence artifact references contain no canonical figure rows"
        )
    preserved = frame.loc[~selected].copy(deep=True)
    updated_paths: list[str] = []
    for index in frame.index[selected]:
        relative = str(frame.at[index, "path"])
        target = (root / relative).resolve()
        try:
            target.relative_to(figures_path)
        except ValueError as exc:
            raise FigureEvidenceError(
                f"Figure reference escapes the canonical directory: {relative}"
            ) from exc
        if target.is_symlink() or not target.is_file():
            raise FigureEvidenceError(f"Missing or unsafe figure reference: {relative}")
        frame.at[index, "sha256"] = file_sha256(target)
        frame.at[index, "size_bytes"] = str(target.stat().st_size)
        rows = _source_row_count(target)
        frame.at[index, "row_count"] = "" if rows is None else str(rows)
        updated_paths.append(relative)
    pd.testing.assert_frame_equal(
        preserved,
        frame.loc[~selected],
        check_dtype=False,
        check_exact=True,
    )
    _atomic_write_csv(frame, table_path)

    reference_lookup = {
        str(row["path"]): row for row in frame.to_dict(orient="records")
    }
    manifest_references = manifest.get("referenced_artifacts")
    if not isinstance(manifest_references, list):
        raise FigureEvidenceError("Evidence summary manifest references are malformed")
    refreshed_count = 0
    for entry in manifest_references:
        if not isinstance(entry, dict):
            raise FigureEvidenceError("Evidence summary manifest reference is malformed")
        relative = str(entry.get("path", ""))
        if relative not in updated_paths:
            continue
        row = reference_lookup[relative]
        entry["sha256"] = row["sha256"]
        entry["size_bytes"] = int(row["size_bytes"])
        entry["row_count"] = (
            None if row["row_count"] == "" else int(row["row_count"])
        )
        refreshed_count += 1
    if refreshed_count != len(updated_paths):
        raise FigureEvidenceError(
            "Evidence summary manifest does not contain every figure table row"
        )

    output_entry = manifest["outputs"][table_key]
    output_entry["sha256"] = file_sha256(table_path)
    output_entry["size_bytes"] = table_path.stat().st_size
    output_entry["row_count"] = len(frame)
    manifest["figure_hash_refresh"] = {
        "scope": "existing_figure_reference_identities_only",
        "sealed_evidence_git_head_preserved": manifest.get("git_head"),
        "refresh_git_head": refresh_head,
        "working_tree_context": "local_uncommitted_figure_revision",
        "updated_reference_count": refreshed_count,
        "scientific_estimates_changed": False,
        "historical_generation_command_preserved": True,
        "historical_absolute_path_provenance_preserved": True,
        "canonical_figure_manifest": {
            "path": _portable_path(figure_manifest_path, root),
            "sha256": file_sha256(figure_manifest_path),
        },
        "refresh_command": command.rstrip(),
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_write_json(manifest, manifest_path)

    reread = _read_json(manifest_path, label="refreshed evidence summary manifest")
    reread_signature = reread.pop("manifest_sha256", None)
    if reread_signature != canonical_json_sha256(reread):
        raise FigureEvidenceError("Refreshed evidence summary self-hash mismatch")
    if file_sha256(table_path) != reread["outputs"][table_key]["sha256"]:
        raise FigureEvidenceError("Refreshed reference-table output hash mismatch")
    return FigureParentRefreshArtifacts(
        evidence_summary_manifest=str(manifest_path),
        reference_table=str(table_path),
        updated_reference_count=refreshed_count,
        manifest_sha256=str(reread_signature),
    )


def build_core_figures(
    *,
    robustness_manifest: str | Path,
    theory_manifest: str | Path,
    readability_manifest: str | Path,
    overhead_manifest: str | Path,
    detector_manifest: str | Path,
    ablation_manifest: str | Path,
    output_dir: str | Path,
    command: str,
    project_root: str | Path = PROJECT_ROOT,
    generator_sources: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> FigureArtifacts:
    """Build plots only from hash-validated computational source tables."""

    root = Path(project_root).resolve()
    if str(root) in command or command.lstrip().startswith("/"):
        raise FigureEvidenceError(
            "Figure generation command must be repository-relative and portable"
        )
    robustness_manifest_path = Path(robustness_manifest).resolve()
    theory_manifest_path = Path(theory_manifest).resolve()
    readability_manifest_path = Path(readability_manifest).resolve()
    overhead_manifest_path = Path(overhead_manifest).resolve()
    detector_manifest_path = Path(detector_manifest).resolve()
    ablation_manifest_path = Path(ablation_manifest).resolve()
    robustness = _read_json(robustness_manifest_path, label="robustness manifest")
    theory = _read_json(theory_manifest_path, label="empirical theory manifest")
    readability = _read_json(readability_manifest_path, label="readability manifest")
    overhead = _read_json(overhead_manifest_path, label="overhead manifest")
    detector = _read_json(detector_manifest_path, label="detector analysis manifest")
    ablation = _read_json(ablation_manifest_path, label="ablation evidence manifest")
    robustness_path = _declared_output(
        robustness, robustness_manifest_path, "plot_source"
    )
    theory_path = _declared_output(theory, theory_manifest_path, "validation")
    readability_path = _declared_output(readability, readability_manifest_path, "summary")
    overhead_path = _declared_output(overhead, overhead_manifest_path, "plot_source")
    detector_path = _declared_output(
        detector, detector_manifest_path, "plot_source"
    )
    detector_metrics_path = _declared_output(
        detector, detector_manifest_path, "metrics"
    )
    ablation_path = _declared_output(
        ablation, ablation_manifest_path, "contrasts"
    )

    _configure_style()
    robustness_full = _robustness_source(robustness_path)
    overhead_full = _overhead_source(overhead_path)
    detector_full = _detector_source(detector_path, detector_metrics_path)
    sources: dict[str, pd.DataFrame] = {
        "robustness_compact_source": _robustness_compact_source(robustness_full),
        "robustness_source": robustness_full,
        "theory_source": _theory_source(theory_path),
        "readability_source": _readability_source(readability_path),
        "overhead_compact_source": _overhead_compact_source(overhead_full),
        "overhead_source": overhead_full,
        "detector_compact_source": _detector_compact_source(detector_full),
        "detector_source": detector_full,
        "ablation_source": _ablation_source(ablation_path),
    }
    output_path = Path(output_dir).resolve()
    _portable_path(output_path, root)
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
        "robustness_compact": _plot_robustness_compact(
            sources["robustness_compact_source"]
        ),
        "robustness": _plot_robustness(sources["robustness_source"]),
        "theory": _plot_theory(sources["theory_source"]),
        "readability": _plot_readability(sources["readability_source"]),
        "overhead_compact": _plot_overhead_compact(
            sources["overhead_compact_source"]
        ),
        "overhead": _plot_overhead(sources["overhead_source"]),
        "detector_compact": _plot_detectors_compact(
            sources["detector_compact_source"]
        ),
        "detector": _plot_detectors(sources["detector_source"]),
        "ablation": _plot_ablation(sources["ablation_source"]),
    }
    _validate_figure_presentation(figures)
    try:
        for name, figure in figures.items():
            _atomic_save_figure(figure, targets[f"{name}_pdf"])
            _atomic_save_figure(figure, targets[f"{name}_png"])
    finally:
        for figure in figures.values():
            plt.close(figure)
    _atomic_write_text(command.rstrip() + "\n", targets["commands"])

    upstream_paths = {
        "robustness": _portable_path(robustness_path, root),
        "theory": _portable_path(theory_path, root),
        "readability": _portable_path(readability_path, root),
        "overhead": _portable_path(overhead_path, root),
        "detector_plot": _portable_path(detector_path, root),
        "detector_metrics": _portable_path(detector_metrics_path, root),
        "ablation": _portable_path(ablation_path, root),
    }
    notes = _technical_notes(upstream_paths, sources["ablation_source"])
    for key, value in notes.items():
        _atomic_write_text(value, targets[key])
    inventory = _figure_inventory(targets, root)
    _atomic_write_csv(inventory, targets["inventory"])

    figure_specs = _figure_specs()
    validation = {
        "schema_version": "rankcloak-computational-figure-validation-v2",
        "status": "passed",
        "authoritative_upstream_sources": {
            label: {
                "path": path,
                "sha256": file_sha256(root / path),
            }
            for label, path in upstream_paths.items()
        },
        "source_tables": {
            key: {
                "path": _portable_path(targets[key], root),
                "sha256": file_sha256(targets[key]),
                "row_count": len(frame),
                "all_plotted_values_finite": True,
                "interval_order_validated": True,
                "validated_derivation_from_hash_validated_upstream": True,
            }
            for key, frame in sources.items()
        },
        "figures": {
            figure_id: {
                "title": spec["title"],
                "panel_count": spec["panel_count"],
                "plotted_row_count": spec["plotted_row_count"],
                "width_mm": 180.0,
                "height_mm": round(
                    FIGURE_HEIGHT_INCHES[figure_id] * 25.4, 2
                ),
                "evidence_status": spec["evidence_status"],
                "classification": spec["classification"],
                "uncertainty": spec["uncertainty"],
            }
            for figure_id, spec in figure_specs.items()
        },
        "checks": {
            "no_nan_or_infinite_plotted_values": True,
            "intervals_contain_estimates": True,
            "confidence_intervals_and_ranges_distinguished": True,
            "automated_readability_not_human_rating": True,
            "ablation_exploratory_status_preserved": True,
            "concise_plot_titles_validated": True,
            "prohibited_plot_phrases_absent": True,
            "capacity_markers_use_true_data_coordinates": True,
            "capacity_marker_displacement_applied": False,
            "absolute_local_paths_emitted": False,
            "pdf_primary_png_300dpi_secondary": True,
        },
    }
    _atomic_write_json(validation, targets["validation"])

    summary = {
        "figure_count": 9,
        "rendered_file_count": 18,
        "source_table_count": 9,
        "technical_note_count": 6,
        "inventory_row_count": 9,
    }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-computational-figure-evidence-v2",
        "status": "passed",
        "inputs": {
            "robustness_manifest": {
                "path": _portable_path(robustness_manifest_path, root),
                "sha256": file_sha256(robustness_manifest_path),
            },
            "theory_manifest": {
                "path": _portable_path(theory_manifest_path, root),
                "sha256": file_sha256(theory_manifest_path),
            },
            "readability_manifest": {
                "path": _portable_path(readability_manifest_path, root),
                "sha256": file_sha256(readability_manifest_path),
            },
            "overhead_manifest": {
                "path": _portable_path(overhead_manifest_path, root),
                "sha256": file_sha256(overhead_manifest_path),
            },
            "detector_manifest": {
                "path": _portable_path(detector_manifest_path, root),
                "sha256": file_sha256(detector_manifest_path),
            },
            "ablation_manifest": {
                "path": _portable_path(ablation_manifest_path, root),
                "sha256": file_sha256(ablation_manifest_path),
            },
        },
        "outputs": {},
        "figures": {},
        "summary": summary,
        "generation_command": command.rstrip(),
        "project_root_assumption": "run_from_repository_root",
        "portable_repository_relative_paths": True,
        "absolute_local_paths_emitted": False,
        "primary_output_format": "vector_pdf",
        "inspection_output_format": "png_300_dpi",
        "human_rating_figures_emitted": False,
        "readability_figure_scope": (
            "automated_surface_diagnostics_not_human_judgements"
        ),
        "source_tables_hash_validated_before_plotting": True,
        "detector_matched_bars": "payload_group_bootstrap_95_confidence_interval",
        "detector_cross_split_bars": "minimum_to_maximum_range_not_confidence_interval",
        "detector_supplementary_panels_status": "exploratory_post_freeze",
        "ablation_figure_status": "exploratory_post_outcome",
        "historical_parent_manifest_note": (
            "Parent package manifests may retain sealed historical absolute-path provenance; "
            "this figure manifest and all new figure-related references are repository-relative."
        ),
    }
    if generator_sources:
        manifest["inputs"]["generator_sources"] = {
            label: {
                "path": _portable_path(path, root),
                "sha256": file_sha256(path),
            }
            for label, path in sorted(generator_sources.items())
        }
    for key, target in targets.items():
        if key == "manifest":
            continue
        entry = {
            "path": _portable_path(target, root),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        if key in sources:
            entry["row_count"] = len(sources[key])
        manifest["outputs"][key] = entry
    for figure_id, spec in figure_specs.items():
        manifest["figures"][figure_id] = {
            "title": spec["title"],
            "classification": spec["classification"],
            "evidence_status": spec["evidence_status"],
            "panel_count": spec["panel_count"],
            "plotted_row_count": spec["plotted_row_count"],
            "width_mm": 180.0,
            "height_mm": round(FIGURE_HEIGHT_INCHES[figure_id] * 25.4, 2),
            "uncertainty": spec["uncertainty"],
            "primary_pdf": _portable_path(targets[spec["pdf_key"]], root),
            "inspection_png_300dpi": _portable_path(
                targets[spec["png_key"]], root
            ),
            "plotted_source_csv": _portable_path(
                targets[spec["source_key"]], root
            ),
            "technical_note": _portable_path(targets[spec["note_key"]], root),
        }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_write_json(manifest, targets["manifest"])
    return FigureArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=summary,
    )
