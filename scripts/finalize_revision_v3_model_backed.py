#!/usr/bin/env python3
"""Complete the V3 handoff from authoritative model-backed result ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DETECTORS = ("textcnn", "deberta", "surprisal")
DETECTOR_LABELS = {
    "textcnn": "TextCNN",
    "deberta": "DeBERTa-v3-base",
    "surprisal": "Model-aware surprisal",
}
EVALUATIONS = ("entropy_gates", "q4_to_q8", "q8_to_q4", "pooled_quantizations")
MODEL_BACKED_DESIGN_START = "<!-- revision-v3-model-backed-design:start -->"
MODEL_BACKED_DESIGN_END = "<!-- revision-v3-model-backed-design:end -->"


import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_revision_v3 import atomic_csv, atomic_json, atomic_text, utc_now  # noqa: E402
from rankcloak.revision_v3_analysis import file_sha256  # noqa: E402


def load_metric_documents(output: Path) -> list[Mapping[str, object]]:
    documents = []
    for evaluation in EVALUATIONS:
        for detector in DETECTORS:
            path = output / f"metrics/{detector}__model_backed__{evaluation}.json"
            if not path.is_file():
                raise SystemExit(f"missing model-backed detector metrics: {path}")
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("threshold_selected_on_test") is not False:
                raise SystemExit(f"test-tuned metric ledger: {path}")
            documents.append(document)
    return documents


def detector_metric_table(documents: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    fields = (
        "roc_auc", "roc_auc_ci_low_95", "roc_auc_ci_high_95",
        "partial_auc_fpr_0_01", "partial_auc_fpr_0_01_ci_low_95",
        "partial_auc_fpr_0_01_ci_high_95", "threshold_at_fpr_0_01",
        "tpr_at_fpr_0_01", "tpr_at_fpr_0_01_ci_low_95", "tpr_at_fpr_0_01_ci_high_95",
        "fpr_at_threshold_0_01", "fpr_at_threshold_0_01_ci_low_95",
        "fpr_at_threshold_0_01_ci_high_95", "false_positives_at_fpr_0_01",
        "true_positives_at_fpr_0_01", "threshold_at_fpr_0_001",
        "tpr_at_fpr_0_001", "fpr_at_threshold_0_001",
        "false_positives_at_fpr_0_001",
    )
    rows = []
    for document in documents:
        row = {
            "detector": document["detector"],
            "detector_label": DETECTOR_LABELS[str(document["detector"])],
            "generation_study": document["generation_study"],
            "evaluation_id": document["evaluation_id"],
            "implementation_kind": document["implementation_kind"],
            "training_rows": int(document["training_rows"]),
            "validation_rows": int(document["validation_rows"]),
            "test_rows": int(document["test_rows"]),
            "validation_negative_count": int(document["validation_negative_count"]),
            "test_negative_count": int(document["test_negative_count"]),
            "test_positive_count": int(document["test_positive_count"]),
            "test_group_count": int(document["test_group_count"]),
            "training_quantizations": ";".join(document["training_quantizations"]),
            "test_quantizations": ";".join(document["test_quantizations"]),
            "seed": int(document["seed"]),
            "threshold_selected_on_test": False,
            "partial_auc_normalization": document["partial_auc_normalization"],
            "bootstrap_resamples": int(document["bootstrap_resamples_requested"]),
            "fpr_0_001_available": bool(document["threshold_selection"]["fpr_0_001"]["available"]),
            "fpr_0_001_unavailable_reason": document["threshold_selection"]["fpr_0_001"].get("reason"),
        }
        for field in fields:
            row[field] = document.get(field)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["generation_study", "evaluation_id", "detector"]).reset_index(drop=True)


def subgroup_table(output: Path) -> pd.DataFrame:
    frames = []
    for evaluation in EVALUATIONS:
        for detector in DETECTORS:
            path = output / f"source_tables/{detector}__model_backed__{evaluation}__subgroups.csv"
            frame = pd.read_csv(path, low_memory=False)
            frame.insert(0, "evaluation_id", evaluation)
            frame.insert(0, "generation_study", "entropy" if evaluation == "entropy_gates" else "quantization")
            frame.insert(0, "detector_label", DETECTOR_LABELS[detector])
            frame.insert(0, "detector", detector)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        na_rep="--",
        caption=caption,
        label=label,
    )


def save_figure(fig: object, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight",
        metadata={"Creator": "RankCloak revision V3", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".png"), dpi=220, bbox_inches="tight",
        metadata={"Software": "RankCloak revision V3"},
    )
    plt.close(fig)


def render_model_backed_experiment_design(design: str) -> str:
    """Replace the model-backed design section without accumulating rerun text."""

    recovery_anchor = "\nThe recovery-mode comparison"
    if recovery_anchor not in design:
        raise ValueError("experiment design lacks the recovery-mode anchor")
    recovery_tail = design.split(recovery_anchor, 1)[1]
    recovery_paragraph = (
        "The recovery-mode comparison"
        + recovery_tail.split("\n\n", 1)[0].rstrip()
    )

    if MODEL_BACKED_DESIGN_START in design:
        prefix = design.split(MODEL_BACKED_DESIGN_START, 1)[0].rstrip()
    elif "\nThe bounded entropy-gate matrix" in design:
        prefix = design.split("\nThe bounded entropy-gate matrix", 1)[0].rstrip()
    elif "\nThe entropy-gate matrix used" in design:
        prefix = design.split("\nThe entropy-gate matrix used", 1)[0].rstrip()
    else:
        raise ValueError("experiment design lacks an entropy-design anchor")

    model_backed = """The entropy-gate matrix used 120 paired experimental cells and three gate levels, producing 360 RankCloak and 360 length-matched ordinary-control generations. Eighteen independent ordinary top-p calibration traces fixed model-specific median and 75th-percentile thresholds before evaluation. Ineligible positions were ordinary top-p samples, not greedy or rank-1 choices. RankCloak and control seeds were each shared across gate levels within a cell. Both fixed-payload and fixed-token-budget estimands were computed from the same records.

The matched quantization matrix used all 1,920 historical Qwen Q4 rows and 1,920 newly generated Q8 counterparts. Historical ordinary-control seeds were read from raw Q4 records. Non-quantization contracts were hash-bound across every pair. Predeclared payload splits support Q4-to-Q8, Q8-to-Q4, and pooled held-payload detector evaluations. Generated detector corpora were locked-partition deduplicated before feature extraction, with complete matched-pair and payload-group removal used to resolve any cross-boundary duplicate component.

The two model-backed studies completed on the local RTX 5000 Ada with exact pinned GGUFs and CUDA-enabled llama-cpp-python 0.3.23. Every outcome, failure status, runtime, and peak-memory value is retained in atomic JSON ledgers; aggregate results are derived rather than manually copied."""
    return (
        prefix
        + "\n\n"
        + MODEL_BACKED_DESIGN_START
        + "\n\n"
        + model_backed
        + "\n\n"
        + MODEL_BACKED_DESIGN_END
        + "\n\n"
        + recovery_paragraph
        + "\n"
    )


def write_detector_artifacts(
    output: Path,
    metrics: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> None:
    atomic_csv(output / "source_tables/model_backed_detector_metrics.csv", metrics)
    atomic_csv(output / "source_tables/model_backed_detector_subgroups.csv", subgroups)
    entropy = subgroups.loc[
        subgroups["evaluation_id"].eq("entropy_gates")
        & subgroups["dimension"].eq("gate_level")
        & subgroups["available"].astype(bool)
    ][[
        "detector_label", "level", "positive_count", "negative_count", "roc_auc",
        "roc_auc_ci_low_95", "roc_auc_ci_high_95", "partial_auc_fpr_0_01",
        "tpr_at_fpr_0_01", "fpr_at_threshold_0_01", "false_positives_at_fpr_0_01",
    ]].copy()
    quantization = metrics.loc[metrics["generation_study"].eq("quantization"), [
        "detector_label", "evaluation_id", "test_positive_count", "test_negative_count",
        "roc_auc", "roc_auc_ci_low_95", "roc_auc_ci_high_95",
        "partial_auc_fpr_0_01", "tpr_at_fpr_0_01", "fpr_at_threshold_0_01",
        "false_positives_at_fpr_0_01",
    ]].copy()
    atomic_text(
        output / "manuscript_tables/entropy_gate_detectors.tex",
        latex_table(entropy, "Detector performance by entropy-gate level on locked, deduplicated test data.", "tab:entropy-gate-detectors"),
    )
    atomic_text(
        output / "manuscript_tables/matched_quantization_detectors.tex",
        latex_table(quantization, "Detector transfer across matched Q4_K_M and Q8_0 conditions.", "tab:matched-quantization-detectors"),
    )

    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42}
    )
    gates = ["ungated", "moderate", "strict"]
    gate_labels = ["Ungated", "Median", "75th percentile"]
    x = np.arange(len(gates), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    for detector_index, detector in enumerate(DETECTORS):
        cell = entropy.loc[entropy["detector_label"].eq(DETECTOR_LABELS[detector])].set_index("level")
        for axis, metric, title in (
            (axes[0], "roc_auc", "ROC-AUC"),
            (axes[1], "partial_auc_fpr_0_01", "Normalized pAUC (FPR 0–1%)"),
        ):
            values = [float(cell.loc[gate, metric]) for gate in gates]
            axis.bar(x + (detector_index - 1) * width, values, width, label=DETECTOR_LABELS[detector])
            axis.set_title(title)
            axis.set_ylim(0, 1.03)
            axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, gate_labels, rotation=15, ha="right")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3)
    fig.suptitle("Entropy-gate detector performance")
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    save_figure(fig, output / "figures/entropy_gate_detector_performance")

    evaluations = ["q4_to_q8", "q8_to_q4", "pooled_quantizations"]
    eval_labels = ["Q4→Q8", "Q8→Q4", "Pooled"]
    x = np.arange(len(evaluations), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    for detector_index, detector in enumerate(DETECTORS):
        cell = metrics.loc[
            metrics["detector"].eq(detector) & metrics["evaluation_id"].isin(evaluations)
        ].set_index("evaluation_id")
        for axis, metric, title in (
            (axes[0], "roc_auc", "ROC-AUC"),
            (axes[1], "tpr_at_fpr_0_01", "TPR at frozen 1% FPR threshold"),
        ):
            values = [float(cell.loc[evaluation, metric]) for evaluation in evaluations]
            axis.bar(x + (detector_index - 1) * width, values, width, label=DETECTOR_LABELS[detector])
            axis.set_title(title)
            axis.set_ylim(0, 1.03)
            axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, eval_labels, rotation=12, ha="right")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3)
    fig.suptitle("Matched-quantization detector transfer")
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    save_figure(fig, output / "figures/matched_quantization_detector_transfer")


def summary_row(
    frame: pd.DataFrame,
    *,
    analysis_id: str,
    metric: str,
    **criteria: str,
) -> Mapping[str, object]:
    selected = frame.loc[
        frame["analysis_id"].eq(analysis_id) & frame["metric"].eq(metric)
    ]
    for field, value in criteria.items():
        selected = selected.loc[selected[field].astype(str).eq(str(value))]
    if len(selected) != 1:
        raise SystemExit(
            f"expected one summary row for {analysis_id} {metric} {criteria}; found {len(selected)}"
        )
    return selected.iloc[0].to_dict()


def detector_subgroup(
    subgroups: pd.DataFrame, detector: str, dimension: str, level: str
) -> Mapping[str, object]:
    selected = subgroups.loc[
        subgroups["detector"].eq(detector)
        & subgroups["evaluation_id"].eq("entropy_gates")
        & subgroups["dimension"].eq(dimension)
        & subgroups["level"].astype(str).eq(level)
    ]
    if len(selected) != 1:
        raise SystemExit(f"missing entropy detector subgroup {detector} {dimension} {level}")
    return selected.iloc[0].to_dict()


def detector_main(
    metrics: pd.DataFrame, detector: str, evaluation: str
) -> Mapping[str, object]:
    selected = metrics.loc[
        metrics["detector"].eq(detector) & metrics["evaluation_id"].eq(evaluation)
    ]
    if len(selected) != 1:
        raise SystemExit(f"missing detector metric {detector} {evaluation}")
    return selected.iloc[0].to_dict()


def true_count(values: pd.Series) -> int:
    return int(values.map(lambda value: str(value).strip().lower() == "true").sum())


def unavailable_conditions(audit: Mapping[str, object]) -> str:
    conditions = list(audit.get("conditions_unavailable_after_deduplication", []))
    return "none" if not conditions else ", ".join(map(str, conditions))


def write_handoff_docs(
    output: Path,
    metrics: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> None:
    entropy = pd.read_csv(output / "source_tables/entropy_generation_summary.csv", low_memory=False)
    entropy_trials = pd.read_csv(
        output / "source_tables/entropy_generation_trials.csv", low_memory=False
    )
    entropy_positions = pd.read_csv(
        output / "source_tables/entropy_position_summary.csv", low_memory=False
    )
    control_quality = pd.read_csv(
        output
        / "source_tables/entropy_rankcloak_control_difference_summary.csv",
        low_memory=False,
    )
    gate_quality = pd.read_csv(
        output / "source_tables/entropy_paired_difference_summary.csv",
        low_memory=False,
    )
    quantization = pd.read_csv(output / "source_tables/quantization_generation_summary.csv", low_memory=False)
    quantization_trials = pd.read_csv(
        output / "source_tables/quantization_generation_trials.csv",
        low_memory=False,
    )
    pairs = pd.read_csv(output / "source_tables/quantization_pair_summary.csv", low_memory=False)
    thresholds = pd.read_csv(output / "source_tables/entropy_calibration_thresholds.csv", low_memory=False)
    generation_audit = json.loads(
        (output / "provenance/generation_analysis_validation.json").read_text(encoding="utf-8")
    )
    entropy_dedup = json.loads(
        (output / "deduplication/model_backed__entropy_gates__leakage_audit.json").read_text(encoding="utf-8")
    )
    quant_dedup = {
        evaluation: json.loads(
            (output / f"deduplication/model_backed__{evaluation}__leakage_audit.json").read_text(encoding="utf-8")
        )
        for evaluation in ("q4_to_q8", "q8_to_q4", "pooled_quantizations")
    }

    threshold_text = "; ".join(
        f"{row.model_id}: median {row.moderate_threshold_bits:.6f} bits, 75th percentile {row.strict_threshold_bits:.6f} bits"
        for row in thresholds.itertuples(index=False)
    )
    methods_path = output / "methods_for_manuscript.md"
    old_methods = methods_path.read_text(encoding="utf-8")
    methods_prefix = old_methods.split("\nThe entropy-gate design was informed", 1)[0].rstrip()
    methods = methods_prefix + f"""

The entropy-gated extension was informed by Cai, Ding, and Tao (arXiv:2504.12108) but is not the same task: Cai et al. study watermarking, whereas RankCloak carries arbitrary payload ranks. Six 128-token ordinary top-p development traces per model used temperature 0.8, top-p 0.95, and the `numpy_pcg64_serial_top_p_v1_token_id_tiebreak` sampler. Their 768 positions per model fixed the following thresholds before evaluation: {threshold_text}. Detector outcomes did not enter calibration.

At every embedding-span step, the encoder computed filtered next-token Shannon entropy. An eligible position consumed and forced the next payload rank. An ineligible position used ordinary top-p sampling under the same allowed-token mask and did not consume a payload rank. The replay-assisted decoder recomputed entropy before every observed token, consumed a rank only at eligible positions, ignored the observed rank at sampled skips, and appended every observed token to context; it received no saved gate-position metadata. Token roles were recorded as `payload` or `ordinary_sampled_skip`. The ungated, moderate, and strict conditions shared one stable RankCloak seed per experimental cell. Their length-matched ordinary controls shared a separate ordinary-generation seed, so shorter controls were deterministic prefixes of longer controls when target lengths differed.

The entropy evaluation comprised 120 RankCloak trials per gate and 120 paired ordinary controls per gate across all three model families, eight artifact classes, all 20 supported artifact/representation cells, and two frozen templates. Fixed-payload generation continued until complete or the predeclared maximum; fixed-token-budget capacity was evaluated at the paired ungated span length. Saved-ID replay and independent visible-text detokenization/retokenization were both evaluated. Confidence intervals resampled complete payload instances, thereby retaining all repeated model and template observations for a selected payload in each bootstrap draw. The generated detector evaluation used an external locked test partition. Historical train and validation rows carrying any of the 16 entropy-study payload instances were excluded, then exact and 0.95-threshold near-duplicate auditing was rerun across fitting and test rows before tokenization or feature extraction. Complete matched pairs or payload groups were removed as needed to resolve leakage; the retained counts are in the audit.

The matched-quantization analysis held Qwen2.5-7B-Instruct revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031, its embedded tokenizer, rendered prompt, payload, codec, token filter, target length, temperature, top-p, llama-cpp-python 0.3.23 backend, and historical ordinary-control seed fixed. It model-backed-replayed all 1,920 historical Q4_K_M rows and generated 1,920 paired Q8_0 rows. Each Q8 record also replayed the historical Q4 token path, allowing entropy, observed-rank, greedy-token, and rank-pressure changes to be compared at identical contexts and observed tokens. Q8 RankCloak saved-ID and visible-text recovery were evaluated independently.

Quantization detector fits used the frozen payload train/validation/test assignment. Q4-to-Q8 trained and selected thresholds only on Q4, then tested Q8; Q8-to-Q4 reversed the direction; the pooled fit included both quantizations in train and validation and tested both only on held payloads. Every evaluation was exact- and near-duplicate audited before feature extraction. TextCNN and DeBERTa retained their predeclared architectures. The model-aware logistic attacker retained the same shared per-token surprisal feature definition for both labels. All detector thresholds were selected from validation data, then frozen for test; 0.1% results remained unavailable wherever validation or test had fewer than 1,000 negatives.
"""
    atomic_text(methods_path, methods)

    entropy_lines = []
    for gate, label in (("ungated", "Ungated"), ("moderate", "Median gate"), ("strict", "75th-percentile gate")):
        gate_trials = entropy_trials.loc[
            entropy_trials["population"].eq("rankcloak")
            & entropy_trials["gate_level"].eq(gate)
        ]
        completed_count = true_count(gate_trials["payload_completion"])
        saved_count = true_count(gate_trials["saved_id_exact_payload_recovery"])
        visible_count = true_count(
            gate_trials["visible_text_exact_payload_recovery"]
        )
        completion = summary_row(entropy, analysis_id="entropy_overall", metric="payload_completion", population="rankcloak", gate_level=gate)
        capacity = summary_row(entropy, analysis_id="entropy_overall", metric="fixed_payload_bits_per_generated_token", population="rankcloak", gate_level=gate)
        budget = summary_row(entropy, analysis_id="entropy_overall", metric="fixed_token_budget_payload_fraction", population="rankcloak", gate_level=gate)
        length = summary_row(entropy, analysis_id="entropy_overall", metric="length_ratio_vs_ungated", population="rankcloak", gate_level=gate)
        saved = summary_row(entropy, analysis_id="entropy_overall", metric="saved_id_exact_payload_recovery", population="rankcloak", gate_level=gate)
        visible = summary_row(entropy, analysis_id="entropy_overall", metric="visible_text_exact_payload_recovery", population="rankcloak", gate_level=gate)
        eligibility = summary_row(entropy, analysis_id="entropy_overall", metric="eligible_position_fraction", population="rankcloak", gate_level=gate)
        surprisal = summary_row(entropy, analysis_id="entropy_overall", metric="mean_token_surprisal_nats", population="rankcloak", gate_level=gate)
        pressure = summary_row(entropy, analysis_id="entropy_overall", metric="mean_rank_pressure_log_probability_gap_nats", population="rankcloak", gate_level=gate)
        forced_rank = summary_row(
            entropy_positions,
            analysis_id="entropy_position_overall",
            metric="observed_rank",
            population="rankcloak",
            gate_level=gate,
            token_role="payload",
        )
        forced_surprisal = summary_row(
            entropy_positions,
            analysis_id="entropy_position_overall",
            metric="token_surprisal_nats",
            population="rankcloak",
            gate_level=gate,
            token_role="payload",
        )
        forced_pressure = summary_row(
            entropy_positions,
            analysis_id="entropy_position_overall",
            metric="rank_pressure_log_probability_gap_nats",
            population="rankcloak",
            gate_level=gate,
            token_role="payload",
        )
        entropy_lines.append(
            f"{label}: payload completion {completed_count}/{len(gate_trials)} ({completion['mean']:.4f}; {len(gate_trials) - completed_count} maximum-budget failures; payload-group bootstrap 95% CI {completion['ci_low_95']:.4f}–{completion['ci_high_95']:.4f}), mean fixed-payload capacity {capacity['mean']:.4f} bits/token, mean fixed-budget payload fraction {budget['mean']:.4f}, mean length ratio versus ungated {length['mean']:.4f}, eligible-position fraction {eligibility['mean']:.4f}, all-position mean token surprisal {surprisal['mean']:.4f} nats and rank pressure {pressure['mean']:.4f} nats, forced-payload-token rank mean {forced_rank['mean']:.4f} (median {forced_rank['median']:.4f}, IQR {forced_rank['p25']:.4f}–{forced_rank['p75']:.4f}, p95 {forced_rank['p95']:.4f}, {int(forced_rank['position_count'])} positions), forced-token surprisal {forced_surprisal['mean']:.4f} nats, forced-token rank pressure {forced_pressure['mean']:.4f} nats, saved-ID exact recovery {saved_count}/{len(gate_trials)} ({saved['mean']:.4f}), and visible-text exact recovery {visible_count}/{len(gate_trials)} ({visible['mean']:.4f})."
        )
    quality_lines = []
    quality_metrics = (
        ("word_count_rankcloak_minus_control", "word count"),
        ("unique_word_fraction_rankcloak_minus_control", "unique-word fraction"),
        ("repeated_bigram_fraction_rankcloak_minus_control", "repeated-bigram fraction"),
        ("surface_flag_total_rankcloak_minus_control", "surface-flag count"),
        ("tfidf_prompt_similarity_rankcloak_minus_control", "TF-IDF prompt similarity"),
    )
    for gate, label in (("ungated", "Ungated"), ("moderate", "Median gate"), ("strict", "75th-percentile gate")):
        values = [
            (
                text,
                summary_row(
                    control_quality,
                    analysis_id="paired_rankcloak_control_difference",
                    metric=metric,
                    gate_level=gate,
                ),
            )
            for metric, text in quality_metrics
        ]
        quality_lines.append(
            label
            + " paired RankCloak-minus-control mean differences were "
            + ", ".join(f"{text} {row['mean']:.4f}" for text, row in values)
            + "."
        )
    gate_quality_lines = []
    gate_quality_metrics = (
        ("word_count_difference_vs_ungated", "word count"),
        ("unique_word_fraction_difference_vs_ungated", "unique-word fraction"),
        ("repeated_bigram_fraction_difference_vs_ungated", "repeated-bigram fraction"),
        ("surface_flag_total_difference_vs_ungated", "surface-flag count"),
    )
    for gate, label in (("moderate", "Median gate"), ("strict", "75th-percentile gate")):
        values = [
            (
                text,
                summary_row(
                    gate_quality,
                    analysis_id="paired_entropy_difference",
                    metric=metric,
                    gate_level=gate,
                ),
            )
            for metric, text in gate_quality_metrics
        ]
        gate_quality_lines.append(
            label
            + " paired gated-minus-ungated mean differences were "
            + ", ".join(f"{text} {row['mean']:.4f}" for text, row in values)
            + "."
        )
    model_lines = []
    for model_id in sorted(entropy_trials["model_id"].astype(str).unique()):
        model_trials = entropy_trials.loc[
            entropy_trials["population"].eq("rankcloak")
            & entropy_trials["model_id"].astype(str).eq(model_id)
        ]
        strict_trials = model_trials.loc[model_trials["gate_level"].eq("strict")]
        visible_counts = {
            gate: true_count(
                model_trials.loc[
                    model_trials["gate_level"].eq(gate),
                    "visible_text_exact_payload_recovery",
                ]
            )
            for gate in ("ungated", "moderate", "strict")
        }
        per_gate_count = {
            gate: int(model_trials["gate_level"].eq(gate).sum())
            for gate in ("ungated", "moderate", "strict")
        }
        model_lines.append(
            f"{model_id}: strict payload completion {true_count(strict_trials['payload_completion'])}/{len(strict_trials)}; visible-text exact recovery ungated/median/strict {visible_counts['ungated']}/{per_gate_count['ungated']}, {visible_counts['moderate']}/{per_gate_count['moderate']}, and {visible_counts['strict']}/{per_gate_count['strict']}."
        )
    detector_lines = []
    for detector in DETECTORS:
        cells = [detector_subgroup(subgroups, detector, "gate_level", gate) for gate in ("ungated", "moderate", "strict")]
        detector_lines.append(
            f"{DETECTOR_LABELS[detector]} gate-specific ROC-AUC was "
            + ", ".join(f"{gate} {cell['roc_auc']:.4f}" for gate, cell in zip(("ungated", "median", "75th-percentile"), cells))
            + "; normalized 0–1% pAUC was "
            + ", ".join(f"{cell['partial_auc_fpr_0_01']:.4f}" for cell in cells)
            + ", respectively. At the single historical-validation-frozen 1% threshold, gate-specific TPRs were "
            + ", ".join(f"{cell['tpr_at_fpr_0_01']:.4f}" for cell in cells)
            + "."
        )

    q8_saved = summary_row(quantization, analysis_id="quantization_overall", metric="saved_id_exact_payload_recovery", quantization="Q8_0", population="rankcloak")
    q8_visible = summary_row(quantization, analysis_id="quantization_overall", metric="visible_text_exact_payload_recovery", quantization="Q8_0", population="rankcloak")
    q4_saved = summary_row(quantization, analysis_id="quantization_overall", metric="saved_id_exact_payload_recovery", quantization="Q4_K_M", population="rankcloak")
    entropy_change = summary_row(pairs, analysis_id="quantization_pair_overall", metric="mean_entropy_q8_minus_q4_bits", population="rankcloak")
    rank_change = summary_row(pairs, analysis_id="quantization_pair_overall", metric="observed_token_rank_changed_fraction", population="rankcloak")
    greedy_change = summary_row(pairs, analysis_id="quantization_pair_overall", metric="greedy_token_changed_fraction", population="rankcloak")
    token_match = summary_row(pairs, analysis_id="quantization_pair_overall", metric="positionwise_generated_token_match_fraction", population="rankcloak")
    q4_rankcloak_trials = quantization_trials.loc[
        quantization_trials["quantization"].eq("Q4_K_M")
        & quantization_trials["population"].eq("rankcloak")
    ]
    q8_rankcloak_trials = quantization_trials.loc[
        quantization_trials["quantization"].eq("Q8_0")
        & quantization_trials["population"].eq("rankcloak")
    ]
    q4_saved_count = true_count(
        q4_rankcloak_trials["saved_id_exact_payload_recovery"]
    )
    q8_saved_count = true_count(
        q8_rankcloak_trials["saved_id_exact_payload_recovery"]
    )
    q8_visible_count = true_count(
        q8_rankcloak_trials["visible_text_exact_payload_recovery"]
    )
    quant_detector_lines = []
    for evaluation, label in (("q4_to_q8", "Q4→Q8"), ("q8_to_q4", "Q8→Q4"), ("pooled_quantizations", "pooled")):
        parts = []
        for detector in DETECTORS:
            row = detector_main(metrics, detector, evaluation)
            parts.append(
                f"{DETECTOR_LABELS[detector]} ROC-AUC {row['roc_auc']:.4f}, pAUC {row['partial_auc_fpr_0_01']:.4f}, TPR {row['tpr_at_fpr_0_01']:.4f} at observed FPR {row['fpr_at_threshold_0_01']:.4f} ({int(row['false_positives_at_fpr_0_01'])}/{int(row['test_negative_count'])} false positives)"
            )
        quant_detector_lines.append(label + ": " + "; ".join(parts) + ".")

    results_path = output / "results_for_manuscript.md"
    old_results = results_path.read_text(encoding="utf-8")
    results_prefix = old_results.split("\nNo model-backed entropy or matched-quantization outcome was produced.", 1)[0].rstrip()
    results = results_prefix + f"""

All 18 entropy calibration traces and all 720 entropy evaluation generations completed. The model-backed ledger audit found zero execution or ledger-validation failures, exact agreement between encoder and replay-derived gate positions, and deterministic prefix agreement for all length-matched control groups. {' '.join(entropy_lines)} Model-stratified outcomes were: {' '.join(model_lines)} {' '.join(gate_quality_lines)} {' '.join(quality_lines)} The quality measures are transparent surface heuristics rather than human judgments. These outcomes quantify capacity, length, recovery, distributional, and heuristic-quality changes; they do not by themselves establish reduced detectability or robust visible-text transport.

{' '.join(detector_lines)} The entropy detector corpus retained {entropy_dedup['retained_rows']:,}/{entropy_dedup['original_rows']:,} rows after locked pre-feature deduplication, with {entropy_dedup['removed_rows']} rows removed and zero audited cross-partition links. Conditions unavailable after this audit: {unavailable_conditions(entropy_dedup)}. Each gate subgroup had fewer than 1,000 test controls after deduplication, so 0.1% TPR was not estimated.

All 1,920 historical Q4 rows were successfully replayed under the exact pinned Q4 model, and all 1,920 paired Q8 generations completed. Saved-ID recovery among RankCloak rows was {q4_saved_count}/{len(q4_rankcloak_trials)} ({q4_saved['mean']:.4f}) for Q4 replay and {q8_saved_count}/{len(q8_rankcloak_trials)} ({q8_saved['mean']:.4f}) for Q8; Q8 visible-text recovery was {q8_visible_count}/{len(q8_rankcloak_trials)} ({q8_visible['mean']:.4f}). On the identical historical Q4 token path for RankCloak rows, Q8 minus Q4 mean next-token entropy was {entropy_change['mean']:.4f} bits (95% CI {entropy_change['ci_low_95']:.4f}–{entropy_change['ci_high_95']:.4f}); the observed-token rank changed at a mean fraction {rank_change['mean']:.4f} of positions and the greedy token changed at {greedy_change['mean']:.4f}. Independently generated Q4/Q8 RankCloak outputs matched positionwise at mean fraction {token_match['mean']:.4f}. These are paired quantization sensitivities for one pinned Qwen model, not cross-family comparisons.

{' '.join(quant_detector_lines)} Locked deduplication retained Q4→Q8 {quant_dedup['q4_to_q8']['retained_rows']:,}/{quant_dedup['q4_to_q8']['original_rows']:,}, Q8→Q4 {quant_dedup['q8_to_q4']['retained_rows']:,}/{quant_dedup['q8_to_q4']['original_rows']:,}, and pooled {quant_dedup['pooled_quantizations']['retained_rows']:,}/{quant_dedup['pooled_quantizations']['original_rows']:,} rows, with zero audited leakage. Conditions unavailable after deduplication were Q4→Q8: {unavailable_conditions(quant_dedup['q4_to_q8'])}; Q8→Q4: {unavailable_conditions(quant_dedup['q8_to_q4'])}; pooled: {unavailable_conditions(quant_dedup['pooled_quantizations'])}. None of these test sets contained 1,000 negatives, so 0.1% operating points were unavailable rather than interpolated.
"""
    atomic_text(results_path, results)

    limitations_path = output / "limitations_for_manuscript.md"
    lines = limitations_path.read_text(encoding="utf-8").splitlines()
    stale_prefixes = (
        "- The model-aware detector is a bounded adaptive threat model",
        "- The entropy-gated protocol is implemented and unit-tested",
        "- The matched quantization experiment was not run",
    )
    retained = [line for line in lines if not line.startswith(stale_prefixes)]
    additions = [
        "- The model-aware detector remains a bounded adaptive threat model using shared exact-model surprisal summaries. The new generation traces also support descriptive entropy, rank, and rank-pressure analyses, but those RankCloak-process diagnostics do not constitute a comprehensive attacker model.",
        "- The entropy study is a bounded 120-payload-cell comparison per gate using two templates and model-specific thresholds from 768 development positions. It estimates capacity, length, recovery, quality, and detector effects in that matrix only. Entropy eligibility does not eliminate exact-model replay dependence, and any visible-text failures remain failures.",
        "- The quantization analysis isolates Q4_K_M versus Q8_0 only for one pinned Qwen2.5-7B-Instruct revision and llama-cpp-python 0.3.23. It does not generalize to other bit widths, quantizers, backends, base models, or hardware. Q4 visible-text recovery was not recomputed by the model-backed Q4 path audit and is unavailable in this comparison; Q8 visible-text recovery is reported separately.",
        "- Model-backed low-FPR thresholds were validation-frozen and exact-count based. The entropy and quantization test sets support 1% but not 0.1% empirical resolution; grouped bootstrap uncertainty cannot supply missing false-positive resolution.",
    ]
    retained.extend(additions)
    atomic_text(limitations_path, "\n".join(retained).rstrip() + "\n")


def write_claim_matrix(output: Path) -> None:
    existing = pd.read_csv(output / "claim_evidence_matrix.csv", low_memory=False)
    existing = existing.loc[
        ~existing["proposed_claim"].isin(
            {
                "Matched quantization sensitivity",
                "Entropy-gated RankCloak has replay-consistent protocol primitives",
                "Entropy gating improves detectability, capacity, recovery, or quality",
            }
        )
    ]
    additions = pd.DataFrame(
        [
            {
                "proposed_claim": "Entropy-gated RankCloak was evaluated under a replay-consistent ordinary-sampled-skip protocol",
                "evidence_status": "fully_addressed_for_bounded_720_trial_matrix",
                "evidence_artifacts": "source_tables/entropy_generation_summary.csv;provenance/generation_analysis_validation.json;figures/entropy_gate_capacity_recovery.pdf",
                "qualification": "Does not remove replay dependence or imply successful visible-text recovery",
            },
            {
                "proposed_claim": "Entropy gating changes payload capacity, output length, recovery, quality, and detector behavior",
                "evidence_status": "supported_descriptively_as_quantified",
                "evidence_artifacts": "source_tables/entropy_generation_summary.csv;source_tables/entropy_position_summary.csv;source_tables/entropy_paired_difference_summary.csv;source_tables/entropy_rankcloak_control_difference_summary.csv;source_tables/model_backed_detector_subgroups.csv;manuscript_tables/entropy_gate_generation.tex;manuscript_tables/entropy_gate_by_model.tex;manuscript_tables/entropy_gate_forced_token_distribution.tex;manuscript_tables/entropy_gate_paired_changes.tex;manuscript_tables/entropy_gate_quality.tex;manuscript_tables/entropy_gate_detectors.tex",
                "qualification": "Direction and magnitude must be stated from the tables; no universal improvement claim",
            },
            {
                "proposed_claim": "Matched Q4_K_M versus Q8_0 sensitivity was isolated for one pinned Qwen revision",
                "evidence_status": "fully_addressed_for_targeted_1920_pair_matrix",
                "evidence_artifacts": "source_tables/quantization_pair_summary.csv;source_tables/model_backed_detector_metrics.csv;figures/matched_quantization_sensitivity.pdf",
                "qualification": "One model revision, two quantizations, and one inference backend only",
            },
            {
                "proposed_claim": "Detector thresholds transfer across matched Q4 and Q8 quantizations",
                "evidence_status": "supported_or_contradicted_exactly_as_reported",
                "evidence_artifacts": "source_tables/model_backed_detector_metrics.csv;manuscript_tables/matched_quantization_detectors.tex;figures/matched_quantization_detector_transfer.pdf",
                "qualification": "Frozen-threshold observed FPR may differ from nominal 1%; 0.1% unavailable",
            },
        ]
    )
    atomic_csv(output / "claim_evidence_matrix.csv", pd.concat([existing, additions], ignore_index=True))


def write_index_and_dictionary(output: Path) -> None:
    requirements = json.loads(
        (PROJECT_ROOT / "configs/revision_v3/generation_requirements.json").read_text(encoding="utf-8")
    )
    downloads = "\n".join(item["download_command"] for item in requirements["artifacts"])
    gpu = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"
    readme = f"""# RankCloak Scientific Reports revision V3 computational handoff

This directory is the authoritative V3 computational extension. It reuses immutable V2 sources without overwriting them, includes the completed model-backed entropy and matched-quantization studies, and contains no manuscript edits.

Primary entry points are `methods_for_manuscript.md`, `results_for_manuscript.md`, `limitations_for_manuscript.md`, `claim_evidence_matrix.csv`, `run_manifest.json`, and `test_report.md`. Open source tables are under `source_tables`; generated LaTeX tables under `manuscript_tables`; vector figures and PNG previews under `figures`; row-level detector predictions under `detector_predictions`; raw atomic model-backed records under `generation/raw`; and exact/near-duplicate audits under `deduplication`.

## Complete rerun guide

Use a fresh output directory for corpus preparation. The four model downloads are exact, pinned inputs and must pass the configured size and SHA-256 checks; substitutions fail closed.

```bash
huggingface-cli download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl --repo-type dataset --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a --local-dir /tmp/rankcloak_revision_v3_sources_repro
sha256sum /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl
.venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import_repro
.venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/build_revision_v3_generation_plans.py --output-dir /tmp/rankcloak_revision_v3_repro
{requirements['required_backend']['environment_creation_command']}
{requirements['required_backend']['installation_command']}
{requirements['required_backend']['runtime_installation_command']}
{downloads}
{requirements['post_download_verification_command']}
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id llama3_8b_instruct_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id mistral_7b_instruct_v0_3_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id llama3_8b_instruct_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id mistral_7b_instruct_v0_3_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q8_0 --gpu-uuid {gpu} --output-dir /tmp/rankcloak_revision_v3_repro/generation
.venv/bin/python scripts/analyze_revision_v3_generation.py --output-dir /tmp/rankcloak_revision_v3_repro --generation-dir /tmp/rankcloak_revision_v3_repro/generation
.venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --prepare-only --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector surprisal --output-dir /tmp/rankcloak_revision_v3_repro
CUDA_VISIBLE_DEVICES={gpu} .venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector textcnn --output-dir /tmp/rankcloak_revision_v3_repro
CUDA_VISIBLE_DEVICES={gpu} .venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector deberta --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python -m pytest -q tests/test_revision_v3_*.py --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_v3_focused.xml
.venv/bin/python -m pytest -q --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_full.xml
.venv/bin/python scripts/finalize_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/validate_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
```

The Dolly checksum is `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`. The model sizes and hashes are recorded in `configs/revision_v3/generation_requirements.json` and verified again in `provenance/generation_preflight.json`. Generation is resumable: completed trial records are immutable and the runner refuses silent overwrite. Run Q4 model-backed quantization replay before Q8 because each Q8 record binds to the exact paired Q4 replay hash. No paid remote compute is part of this workflow.

Re-running analysis and finalization from unchanged authoritative ledgers regenerates source tables, LaTeX, figures, prose, source maps, and checksums. `provenance/artifact_source_map.csv` maps every publication artifact to its source and command.
"""
    atomic_text(output / "README.md", readme)

    design_path = output / "experiment_design.md"
    design = design_path.read_text(encoding="utf-8")
    design = render_model_backed_experiment_design(design)
    atomic_text(design_path, design)

    dictionary_path = output / "data_dictionary.md"
    dictionary = dictionary_path.read_text(encoding="utf-8")
    dictionary = dictionary.split("\n## Model-backed generation", 1)[0].rstrip() + """

## Model-backed generation

- `plan_id` / `pairing_unit_id` / `experimental_cell_id`: deterministic trial, matched quantization-observation, and six-row entropy-cell identifiers.
- `entropy_bits`: filtered next-token Shannon entropy in bits before the observed token.
- `eligible`: inclusive entropy-threshold decision recomputed by encoder and decoder.
- `token_role`: `payload`, `ordinary_sampled_skip`, or `ordinary_control`.
- `payload_rank`: consumed payload rank at an eligible position; missing at sampled skips and controls.
- `observed_rank`: exact model rank of the observed token under the relevant quantization and token mask.
- `token_surprisal_nats`: negative exact-model token log probability.
- `rank_pressure_log_probability_gap_nats`: greedy log probability minus observed-token log probability.
- `payload_completion`: whether all requested ranks were embedded before the maximum length.
- `fixed_payload_bits_per_generated_token`: serialized payload bits divided by full generated-token count.
- `fixed_token_budget_payload_fraction`: serialized payload fraction embedded within the paired ungated token budget.
- `ordinary_sampled_skip`: an entropy-ineligible top-p sample that does not consume a payload symbol.
- Ungated records retain the historical `forced_log_probabilities` field name; analysis aliases it to the same embedding-span trace represented by `embedding_log_probabilities` in gated records.
- Calibration `validation` maps contain the exclusion assertion `detector_outcomes_used=false`; a valid calibration record requires that value to be false while its token-count and finite-entropy assertions are true.
- `mean_entropy_q8_minus_q4_bits`: paired mean change when Q8 replays the identical historical Q4 token path.
- `observed_token_rank_changed_fraction` / `greedy_token_changed_fraction`: fraction of identical-path positions whose observed-token rank or greedy token differs across quantizations.
- `positionwise_generated_token_match_fraction`: same-position Q4/Q8 token agreement for independently generated paired outputs.

## Model-backed detector evaluations

- `entropy_gates`: historical payload-excluded train/validation data with the new entropy corpus locked as test.
- `q4_to_q8` / `q8_to_q4`: train and validation on one quantization, test on the other using held payloads.
- `pooled_quantizations`: both quantizations in fitting partitions and both in the held-payload test partition.
- `deduplication_before_feature_extraction`: assertion that locked exact/near-duplicate auditing preceded neural tokenization or model-aware feature construction.
- `training_quantizations` / `test_quantizations`: machine-readable holdout identity.
- `fpr_0_001_available`: false for model-backed evaluations whose validation or test control count is below 1,000; no interpolation substitutes for the unavailable estimand.
"""
    atomic_text(dictionary_path, dictionary)


def write_source_map(output: Path) -> None:
    path = output / "provenance/artifact_source_map.csv"
    existing = pd.read_csv(path, low_memory=False)
    rows = [
        ("manuscript_tables/entropy_calibration_thresholds.tex", "source_tables/entropy_calibration_thresholds.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_generation.tex", "source_tables/entropy_generation_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_by_model.tex", "source_tables/entropy_generation_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_forced_token_distribution.tex", "source_tables/entropy_position_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_quality.tex", "source_tables/entropy_rankcloak_control_difference_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_paired_changes.tex", "source_tables/entropy_paired_difference_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/entropy_gate_detectors.tex", "source_tables/model_backed_detector_subgroups.csv", ".venv/bin/python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/matched_quantization_generation.tex", "source_tables/quantization_generation_summary.csv;source_tables/quantization_pair_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("manuscript_tables/matched_quantization_detectors.tex", "source_tables/model_backed_detector_metrics.csv", ".venv/bin/python scripts/finalize_revision_v3.py"),
        ("figures/entropy_gate_capacity_recovery.pdf", "source_tables/entropy_generation_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("figures/entropy_gate_detector_performance.pdf", "source_tables/model_backed_detector_subgroups.csv", ".venv/bin/python scripts/finalize_revision_v3.py"),
        ("figures/matched_quantization_sensitivity.pdf", "source_tables/quantization_pair_comparison.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("figures/matched_quantization_recovery.pdf", "source_tables/quantization_generation_summary.csv", ".venv/bin/python scripts/analyze_revision_v3_generation.py"),
        ("figures/matched_quantization_detector_transfer.pdf", "source_tables/model_backed_detector_metrics.csv", ".venv/bin/python scripts/finalize_revision_v3.py"),
    ]
    added = pd.DataFrame(rows, columns=existing.columns)
    combined = pd.concat([existing, added], ignore_index=True).drop_duplicates("artifact", keep="last")
    atomic_csv(path, combined.sort_values("artifact").reset_index(drop=True))


def update_manifest(output: Path) -> None:
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = json.loads(
        (output / "provenance/generation_environment.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (output / "provenance/generation_analysis_validation.json").read_text(encoding="utf-8")
    )
    fit_files = sorted((output / "provenance").glob("*__model_backed__*__fit.json"))
    fits = [json.loads(path.read_text(encoding="utf-8")) for path in fit_files]
    entropy_trials = pd.read_csv(output / "source_tables/entropy_generation_trials.csv", low_memory=False)
    quant_trials = pd.read_csv(output / "source_tables/quantization_generation_trials.csv", low_memory=False)
    calibration = pd.read_csv(output / "source_tables/entropy_calibration_traces.csv", low_memory=False)
    execution_seconds = float(
        entropy_trials["execution_seconds"].sum()
        + quant_trials["execution_seconds"].sum()
        + calibration["execution_seconds"].sum()
    )
    experiment_status = [
        {"experiment": "A", "name": "strict_deduplication", "status": "completed"},
        {"experiment": "B", "name": "human_controls", "status": "completed_bounded_secondary"},
        {"experiment": "C", "name": "unseen_model_family", "status": "completed"},
        {"experiment": "D", "name": "matched_quantization", "status": "completed_targeted_qwen_q4_q8"},
        {"experiment": "E", "name": "adaptive_detectors", "status": "completed_bounded_threat_models"},
        {"experiment": "F", "name": "low_fpr", "status": "completed_with_resolution_warnings"},
        {"experiment": "G", "name": "entropy_gate", "status": "completed_bounded_720_trial_matrix"},
        {"experiment": "H", "name": "topic_variability", "status": "completed"},
    ]
    trial_counts = dict(manifest["trial_counts"])
    trial_counts.update(
        {
            "new_model_generation_trials": 2640,
            "new_entropy_evaluation_generations": 720,
            "new_entropy_calibration_traces": 18,
            "new_matched_quantization_q8_generations": 1920,
            "model_backed_historical_q4_replays": 1920,
            "matched_quantization_q4_trials_reused": 1920,
            "real_model_smoke_records": int(
                environment["real_model_smoke_record_count"]
            ),
            "real_model_smoke_new_generation_records": int(
                environment["real_model_smoke_new_generation_count"]
            ),
            "real_model_smoke_q4_replay_records": int(
                environment["real_model_smoke_q4_replay_count"]
            ),
            "new_model_backed_detector_fits": 12,
        }
    )
    model_backed_commands = [
        "PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id <each-q4-model> --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
        "PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id <each-q4-model> --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
        "PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
        "PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q8_0 --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
        ".venv/bin/python scripts/analyze_revision_v3_generation.py",
        ".venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --prepare-only",
        ".venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector <surprisal|textcnn|deberta>",
        ".venv/bin/python -m pytest -q tests/test_revision_v3_*.py --junitxml=results/revision_v3/logs/pytest_v3_focused.xml",
    ]
    completion_time = max(
        [environment["execution_completed_at"], *(record["completed_at"] for record in fits)]
    )
    run_duration_seconds = float(
        (
            pd.Timestamp(completion_time)
            - pd.Timestamp(manifest["start_time"])
        ).total_seconds()
    )
    manifest.pop("blocked_generation", None)
    manifest.update(
        {
            "status": "complete",
            "completion_time": completion_time,
            "run_duration_seconds": run_duration_seconds,
            "git_commit_at_generation_analysis": validation["analysis_git_commit"],
            "trial_counts": trial_counts,
            "detector_fit_count": 24,
            "new_model_backed_detector_fit_count": len(fits),
            "new_model_backed_detector_fit_elapsed_seconds_sum": float(sum(float(record["elapsed_seconds"]) for record in fits)),
            "generation_execution": {
                **validation["counts"],
                "new_generation_trial_count": 2640,
                "historical_q4_model_backed_replay_count": 1920,
                "execution_git_commits": environment["execution_git_commits"],
                "sum_per_record_execution_seconds": execution_seconds,
                "real_model_smoke_record_count": int(
                    environment["real_model_smoke_record_count"]
                ),
                "real_model_smoke_execution_seconds_sum": float(
                    environment["real_model_smoke_execution_seconds_sum"]
                ),
                "started_at": environment["execution_started_at"],
                "completed_at": environment["execution_completed_at"],
                "remote_paid_compute_used": False,
                "failed_trial_count": validation["checks"]["failure_record_count"],
            },
            "generation_environment": environment,
            "model_backed_model_artifacts": environment["model_artifacts"],
            "model_backed_tokenizers": environment["tokenizer_identifiers"],
            "model_backed_inference_backend": {
                "packages": environment["packages"],
                "llama_cpp_system_info": environment["llama_cpp_system_info"],
                "gpu_offload_supported": environment["gpu_offload_supported"],
            },
            "model_backed_device_information": environment["gpu_inventory"],
            "generation_preflight_status": "ready",
            "commands": list(dict.fromkeys([*manifest.get("commands", []), *model_backed_commands])),
            "random_seeds": sorted(
                set(map(int, manifest.get("random_seeds", [])))
                | {int(record["seed"]) for record in fits}
            ),
            "experiment_status": experiment_status,
        }
    )
    atomic_csv(output / "provenance/experiment_status.csv", pd.DataFrame(experiment_status))
    atomic_json(manifest_path, manifest)


def artifact_manifest(output: Path) -> pd.DataFrame:
    rows = []
    excluded = {"artifact_manifest.csv", "provenance/validation_report.json"}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded:
            continue
        row_count = None
        if path.suffix == ".csv":
            try:
                row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            except Exception:
                row_count = None
        rows.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": file_sha256(path), "row_count": row_count}
        )
    return pd.DataFrame(rows)


def complete_handoff(output: Path) -> Mapping[str, object]:
    output = output.resolve()
    generation_validation = json.loads(
        (output / "provenance/generation_analysis_validation.json").read_text(encoding="utf-8")
    )
    if generation_validation.get("status") != "pass":
        raise SystemExit("model-backed generation validation has not passed")
    documents = load_metric_documents(output)
    metrics = detector_metric_table(documents)
    subgroups = subgroup_table(output)
    write_detector_artifacts(output, metrics, subgroups)
    write_handoff_docs(output, metrics, subgroups)
    write_claim_matrix(output)
    write_index_and_dictionary(output)
    write_source_map(output)
    update_manifest(output)
    atomic_csv(output / "artifact_manifest.csv", artifact_manifest(output))
    return {
        "status": "complete",
        "model_backed_detector_metric_rows": len(metrics),
        "model_backed_subgroup_rows": len(subgroups),
        "artifact_count": len(pd.read_csv(output / "artifact_manifest.csv")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(complete_handoff(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
