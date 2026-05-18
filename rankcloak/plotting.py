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
