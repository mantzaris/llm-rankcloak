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
    frame = recovery_frame[recovery_frame["encoding"] == "fixed_radix_bits"].copy()
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
