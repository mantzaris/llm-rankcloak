"""Matplotlib plotting helpers for RankCloak outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _save_placeholder(path: Path, title: str, message: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_token_count_by_payload(tokenization_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = tokenization_frame.copy()
    if "llm_token_count" not in frame or frame["llm_token_count"].dropna().empty:
        return _save_placeholder(
            output_path,
            "Token Count By Payload",
            "LLM tokenizer was unavailable; rerun after loading the GGUF model.",
        )
    frame = frame.sort_values("llm_token_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(frame["payload_name"], frame["llm_token_count"], color="#2b6f73")
    ax.set_xlabel("LLM token count")
    ax.set_ylabel("Payload")
    ax.set_title("Direct Subword Token Count By Payload")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_rank_summary_direct_subword(rank_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rank_frame.empty or "mean_rank" not in rank_frame or rank_frame["mean_rank"].dropna().empty:
        return _save_placeholder(
            output_path,
            "Direct Subword Rank Summary",
            "Model logits were unavailable; rerun after installing llama-cpp-python and loading the GGUF.",
        )
    frame = rank_frame.sort_values("mean_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(frame["payload_name"], frame["mean_rank"], color="#8a5a28", label="mean")
    ax.scatter(frame["p95_rank"], frame["payload_name"], color="#1f1f1f", label="p95")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Payload")
    ax.set_title("Direct Subword Rank Summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_cover_length_vs_rank_alphabet(recovery_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = recovery_frame.copy()
    if "encoding_name" in frame:
        frame = frame[frame["encoding_name"] == "fixed_radix_bits"].copy()
    elif "encoding" in frame:
        frame = frame[frame["encoding"] == "fixed_radix_bits"].copy()
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Cover Length Vs Rank Alphabet",
            "No fixed-radix recovery rows were available.",
        )
    grouped = frame.groupby("alphabet_size", as_index=False)["rank_count"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["alphabet_size"], grouped["rank_count"], marker="o", color="#304c89")
    try:
        ax.set_xscale("log", base=2)
    except (TypeError, ValueError):
        ax.set_xscale("log", basex=2)
    ax.set_xlabel("Bounded rank alphabet size")
    ax.set_ylabel("Mean rank count")
    ax.set_title("Mean Cover Length Proxy Vs Rank Alphabet")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_recovery_by_cover_prompt_and_alphabet(recovery_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if recovery_frame.empty or "exact_recovery" not in recovery_frame:
        return _save_placeholder(
            output_path,
            "Recovery By Cover Prompt And Alphabet",
            "No stegotext recovery rows were available.",
        )
    frame = recovery_frame.copy()
    frame = frame.dropna(subset=["exact_recovery"])
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Recovery By Cover Prompt And Alphabet",
            "Stegotext recovery was skipped because the model was unavailable.",
        )
    grouped = (
        frame.groupby(["cover_prompt_name", "alphabet_size"], as_index=False)["exact_recovery"]
        .mean()
        .sort_values(["cover_prompt_name", "alphabet_size"])
    )
    labels = [
        "{} / B={}".format(row["cover_prompt_name"], int(row["alphabet_size"]))
        for _, row in grouped.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(grouped)), grouped["exact_recovery"], color="#52796f")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact recovery rate")
    ax.set_title("Exact Recovery By Cover Prompt And Rank Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_cover_text_feature_comparison(feature_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_frame.empty or "source_type" not in feature_frame:
        return _save_placeholder(
            output_path,
            "Cover Text Feature Comparison",
            "No cover text feature rows were available.",
        )
    metrics = [
        "whitespace_fraction",
        "punctuation_fraction",
        "digit_fraction",
        "alphabetic_fraction",
        "unique_token_fraction",
    ]
    available_metrics = [metric for metric in metrics if metric in feature_frame]
    if not available_metrics:
        return _save_placeholder(
            output_path,
            "Cover Text Feature Comparison",
            "No comparable feature columns were available.",
        )
    grouped = feature_frame[["source_type"] + available_metrics].groupby("source_type").mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean feature value")
    ax.set_title("RankCloak Vs Baseline Cover Text Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_strong_prompt_mean_logprob(feature_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_frame.empty or "mean_token_log_probability" not in feature_frame:
        return _save_placeholder(
            output_path,
            "Strong Prompt Mean Log Probability",
            "No feature rows with token log probabilities were available.",
        )
    frame = feature_frame[feature_frame["source_type"] == "rankcloak"].copy()
    frame = frame.dropna(subset=["mean_token_log_probability"])
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Strong Prompt Mean Log Probability",
            "RankCloak feature rows were unavailable.",
        )
    pivot = frame.pivot_table(
        index="cover_prompt_name",
        columns="alphabet_size",
        values="mean_token_log_probability",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean token log probability")
    ax.set_xlabel("Prompt")
    ax.set_title("Mean Token Log Probability By Prompt And Alphabet")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_strong_prompt_recovery(recovery_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if recovery_frame.empty or "exact_recovery" not in recovery_frame:
        return _save_placeholder(
            output_path,
            "Strong Prompt Recovery",
            "No stegotext recovery rows were available.",
        )
    frame = recovery_frame.dropna(subset=["exact_recovery"]).copy()
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Strong Prompt Recovery",
            "Stegotext recovery rows were unavailable.",
        )
    pivot = frame.pivot_table(
        index="cover_prompt_name",
        columns="alphabet_size",
        values="exact_recovery",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact recovery rate")
    ax.set_xlabel("Prompt")
    ax.set_title("Exact Recovery By Strong Prompt And Alphabet")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_strong_prompt_length(recovery_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if recovery_frame.empty or "generated_token_count" not in recovery_frame:
        return _save_placeholder(
            output_path,
            "Strong Prompt Length",
            "No generated length rows were available.",
        )
    pivot = recovery_frame.pivot_table(
        index="cover_prompt_name",
        columns="alphabet_size",
        values="generated_token_count",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean generated token count")
    ax.set_xlabel("Prompt")
    ax.set_title("Generated Length By Strong Prompt And Alphabet")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_strong_prompt_rank_pressure(recovery_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric = "p95_generated_rank"
    if recovery_frame.empty or metric not in recovery_frame:
        return _save_placeholder(
            output_path,
            "Strong Prompt Rank Pressure",
            "No rank-pressure rows were available.",
        )
    pivot = recovery_frame.pivot_table(
        index="cover_prompt_name",
        columns="alphabet_size",
        values=metric,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean p95 generated rank")
    ax.set_xlabel("Prompt")
    ax.set_title("Rank Pressure By Strong Prompt And Alphabet")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_dialogue_prompt_repetition(feature_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_frame.empty or "repeated_token_fraction" not in feature_frame:
        return _save_placeholder(
            output_path,
            "Dialogue Prompt Repetition",
            "No repeated-token feature rows were available.",
        )
    frame = feature_frame[feature_frame["source_type"] == "rankcloak"].dropna(
        subset=["repeated_token_fraction"]
    )
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Dialogue Prompt Repetition",
            "No RankCloak repeated-token feature rows were available.",
        )
    pivot = frame.pivot_table(
        index="cover_prompt_name",
        columns="alphabet_size",
        values="repeated_token_fraction",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean repeated-token fraction")
    ax.set_xlabel("Prompt")
    ax.set_title("Dialogue Prompt Repetition By Alphabet")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Alphabet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_dialogue_prompt_quality_scatter(feature_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"mean_token_log_probability", "repeated_token_fraction", "prompt_family"}
    if feature_frame.empty or not required.issubset(set(feature_frame.columns)):
        return _save_placeholder(
            output_path,
            "Dialogue Prompt Quality Scatter",
            "No feature rows were available for the quality scatter.",
        )
    frame = feature_frame[feature_frame["source_type"] == "rankcloak"].dropna(
        subset=["mean_token_log_probability", "repeated_token_fraction"]
    )
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Dialogue Prompt Quality Scatter",
            "No RankCloak feature rows were available for the quality scatter.",
        )
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, family_frame in frame.groupby("prompt_family"):
        ax.scatter(
            family_frame["mean_token_log_probability"],
            family_frame["repeated_token_fraction"],
            label=family,
            alpha=0.8,
        )
    ax.set_xlabel("Mean token log probability")
    ax.set_ylabel("Repeated-token fraction")
    ax.set_title("Dialogue Prompt Quality Feature Scatter")
    ax.legend(title="Prompt family", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_payload_representation_rank_count(comparison_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_frame.empty or "rank_count" not in comparison_frame:
        return _save_placeholder(
            output_path,
            "Payload Representation Rank Count",
            "No payload granularity comparison rows were available.",
        )
    frame = comparison_frame.copy()
    frame["label"] = frame["representation_name"].astype(str)
    if "alphabet_size" in frame:
        frame["label"] = frame.apply(
            lambda row: (
                "{} B={}".format(row["representation_name"], int(row["alphabet_size"]))
                if row.get("alphabet_size") == row.get("alphabet_size")
                else str(row["representation_name"])
            ),
            axis=1,
        )
    pivot = frame.pivot_table(
        index="payload_name",
        columns="label",
        values="rank_count",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rank count")
    ax.set_xlabel("Payload")
    ax.set_title("Payload Representation Rank Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Representation", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_segmented_condition_mean_logprob(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "mean_token_log_probability" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Segmented Condition Mean Log Probability",
            "No segmented protocol trial rows were available.",
        )
    frame = trial_frame.dropna(subset=["mean_token_log_probability"]).copy()
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Segmented Condition Mean Log Probability",
            "No token log probability values were available.",
        )
    grouped = frame.groupby("condition_name", as_index=False)["mean_token_log_probability"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(grouped["condition_name"], grouped["mean_token_log_probability"], color="#2b6f73")
    ax.set_ylabel("Mean token log probability")
    ax.set_xlabel("Condition")
    ax.set_title("Segmented Protocol Mean Token Log Probability")
    ax.set_xticklabels(grouped["condition_name"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_segmented_condition_repetition(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "repeated_token_fraction_mean" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Segmented Condition Repetition",
            "No repeated-token summary rows were available.",
        )
    frame = trial_frame.dropna(subset=["repeated_token_fraction_mean"]).copy()
    if frame.empty:
        return _save_placeholder(
            output_path,
            "Segmented Condition Repetition",
            "No repeated-token values were available.",
        )
    grouped = frame.groupby("condition_name", as_index=False)["repeated_token_fraction_mean"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(grouped["condition_name"], grouped["repeated_token_fraction_mean"], color="#8a5a28")
    ax.set_ylabel("Mean repeated-token fraction")
    ax.set_xlabel("Condition")
    ax.set_title("Segmented Protocol Repetition By Condition")
    ax.set_xticklabels(grouped["condition_name"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_segmented_condition_length(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "total_generated_token_count" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Segmented Condition Length",
            "No generated length rows were available.",
        )
    grouped = trial_frame.groupby("condition_name", as_index=False)["total_generated_token_count"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(grouped["condition_name"], grouped["total_generated_token_count"], color="#304c89")
    ax.set_ylabel("Mean generated token count")
    ax.set_xlabel("Condition")
    ax.set_title("Segmented Protocol Generated Length")
    ax.set_xticklabels(grouped["condition_name"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_segmented_recovery_by_condition(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "exact_recovery" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Segmented Recovery By Condition",
            "No recovery rows were available.",
        )
    grouped = trial_frame.groupby("condition_name", as_index=False)["exact_recovery"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(grouped["condition_name"], grouped["exact_recovery"], color="#52796f")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact recovery rate")
    ax.set_xlabel("Condition")
    ax.set_title("Segmented Protocol Recovery By Condition")
    ax.set_xticklabels(grouped["condition_name"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_segmented_single_vs_multi_topic(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"condition_name", "mean_token_log_probability", "repeated_token_fraction_mean"}
    if trial_frame.empty or not required.issubset(set(trial_frame.columns)):
        return _save_placeholder(
            output_path,
            "Segmented Single Vs Multi Topic",
            "No comparable segmented protocol rows were available.",
        )
    frame = trial_frame.copy()
    frame["condition_group"] = frame["condition_name"].map(
        lambda name: (
            "single_long"
            if str(name).startswith("single_long")
            else "segmented_multi_topic"
            if "multi_topic" in str(name)
            else "segmented_single_topic"
        )
    )
    grouped = frame.groupby("condition_group", as_index=False)[
        ["mean_token_log_probability", "repeated_token_fraction_mean"]
    ].mean()
    fig, ax_left = plt.subplots(figsize=(9, 5))
    positions = list(range(len(grouped)))
    ax_left.bar(
        [position - 0.18 for position in positions],
        grouped["mean_token_log_probability"],
        width=0.36,
        color="#2b6f73",
        label="mean logprob",
    )
    ax_right = ax_left.twinx()
    ax_right.bar(
        [position + 0.18 for position in positions],
        grouped["repeated_token_fraction_mean"],
        width=0.36,
        color="#8a5a28",
        label="repetition",
    )
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(grouped["condition_group"], rotation=20, ha="right")
    ax_left.set_ylabel("Mean token log probability")
    ax_right.set_ylabel("Mean repeated-token fraction")
    ax_left.set_title("Single Long Vs Segmented Topic Schedules")
    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_forced_vs_full_logprob(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "condition_name",
        "forced_prefix_mean_log_probability_mean",
        "full_message_mean_log_probability_mean",
    }
    if trial_frame.empty or not required.issubset(set(trial_frame.columns)):
        return _save_placeholder(
            output_path,
            "Forced Prefix Vs Full Message Log Probability",
            "No quality-control trial rows were available.",
        )
    grouped = trial_frame.groupby("condition_name", as_index=False)[
        [
            "forced_prefix_mean_log_probability_mean",
            "full_message_mean_log_probability_mean",
        ]
    ].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    grouped.set_index("condition_name").plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean token log probability")
    ax.set_xlabel("Condition")
    ax.set_title("Forced Prefix Vs Full Message Log Probability")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(["forced prefix", "full message"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_forced_vs_full_repetition(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "condition_name",
        "forced_prefix_repetition_mean",
        "full_message_repetition_mean",
    }
    if trial_frame.empty or not required.issubset(set(trial_frame.columns)):
        return _save_placeholder(
            output_path,
            "Forced Prefix Vs Full Message Repetition",
            "No quality-control repetition rows were available.",
        )
    grouped = trial_frame.groupby("condition_name", as_index=False)[
        ["forced_prefix_repetition_mean", "full_message_repetition_mean"]
    ].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    grouped.set_index("condition_name").plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean repeated-token fraction")
    ax.set_xlabel("Condition")
    ax.set_title("Forced Prefix Vs Full Message Repetition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(["forced prefix", "full message"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_tail_policy_logprob(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "tail_policy" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Tail Policy Log Probability",
            "No tail policy rows were available.",
        )
    grouped = trial_frame.groupby("tail_policy", as_index=False)[
        [
            "forced_prefix_mean_log_probability_mean",
            "full_message_mean_log_probability_mean",
        ]
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.set_index("tail_policy").plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean token log probability")
    ax.set_xlabel("Tail policy")
    ax.set_title("Tail Policy Mean Log Probability")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(["forced prefix", "full message"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_filter_effect_logprob(trial_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trial_frame.empty or "token_filter_name" not in trial_frame:
        return _save_placeholder(
            output_path,
            "Token Filter Log Probability",
            "No token filter rows were available.",
        )
    grouped = trial_frame.groupby("token_filter_name", as_index=False)[
        [
            "forced_prefix_mean_log_probability_mean",
            "full_message_mean_log_probability_mean",
        ]
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.set_index("token_filter_name").plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean token log probability")
    ax.set_xlabel("Token filter")
    ax.set_title("Token Filter Mean Log Probability")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(["forced prefix", "full message"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_filter_effect_artifacts(feature_frame: Any, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"token_filter_name", "artifact_count_total", "punctuation_fraction"}
    if feature_frame.empty or not required.issubset(set(feature_frame.columns)):
        return _save_placeholder(
            output_path,
            "Token Filter Artifact Features",
            "No artifact feature rows were available.",
        )
    frame = feature_frame[feature_frame["source_type"].astype(str).str.contains("full_message")]
    if frame.empty:
        frame = feature_frame
    grouped = frame.groupby("token_filter_name", as_index=False)[
        ["artifact_count_total", "punctuation_fraction", "repeated_token_fraction"]
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.set_index("token_filter_name").plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean feature value")
    ax.set_xlabel("Token filter")
    ax.set_title("Token Filter Artifact And Repetition Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_quality_recovery_by_condition(trial_frame: Any, output_path: Path) -> Path:
    return plot_segmented_recovery_by_condition(trial_frame, output_path)
