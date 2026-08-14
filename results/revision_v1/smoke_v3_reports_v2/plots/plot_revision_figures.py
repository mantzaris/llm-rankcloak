#!/usr/bin/env python3
"""Render hash-sealed RankCloak report plot sources; generated, do not edit."""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render(source, destination, title):
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    available = [row for row in rows if row.get("report_status") == "available"]
    if not available:
        reason = rows[0].get("reason", "Machine evidence unavailable") if rows else "Machine evidence unavailable"
        axis.axis("off")
        axis.text(0.5, 0.5, "Unavailable\n" + reason, ha="center", va="center", wrap=True)
    else:
        candidates = (
            "exact_payload_recovery_rate", "exact_recovery_rate", "mean", "roc_auc", "pr_auc", "balanced_accuracy",
            "R_effective_bits_per_forced_plus_tail_token", "R_B_bits_per_forced_token",
            "Q_B_nats_per_forced_token", "true_positive_rate", "precision",
        )
        y_column = next((name for name in candidates if any(_float(row.get(name)) is not None for row in available)), None)
        if y_column is None:
            axis.axis("off")
            axis.text(0.5, 0.5, "Verified rows available; no numeric plotting field in this source.", ha="center", va="center", wrap=True)
        else:
            values = [_float(row.get(y_column)) for row in available]
            positions = [index for index, value in enumerate(values) if value is not None]
            numeric = [value for value in values if value is not None]
            axis.plot(positions, numeric, marker="o", linewidth=1.2)
            axis.set_xlabel("Verified source-row order")
            axis.set_ylabel(y_column.replace("_", " "))
            axis.grid(alpha=0.25)
    axis.set_title(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        plt.close(fig)
        raise FileExistsError("Refusing to overwrite rendered figure: {}".format(destination))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + destination.stem + ".",
            suffix=destination.suffix,
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, dpi=180)
        plt.close(fig)
        os.replace(str(temporary), str(destination))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("plot_registry.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="pdf")
    parser.add_argument("--only")
    args = parser.parse_args()
    with args.registry.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        if args.only and record["plot_id"] != args.only:
            continue
        source = args.registry.parent / record["source_csv"]
        destination = args.output_dir / (record["plot_id"] + "." + args.format)
        render(source, destination, record["title"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
