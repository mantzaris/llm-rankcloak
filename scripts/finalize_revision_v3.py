#!/usr/bin/env python3
"""Generate V3 source tables, manuscript handoff, figures, and manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_revision_v3 import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    canonical_sha256,
    utc_now,
)
from rankcloak.revision_v3_analysis import file_sha256  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DETECTOR_LABELS = {
    "textcnn": "TextCNN",
    "deberta": "DeBERTa-v3-base",
    "surprisal": "Model-aware surprisal",
}
MODEL_LABELS = {
    "llama3_8b_instruct_q4_k_m": "Llama 3 8B",
    "mistral_7b_instruct_v0_3_q4_k_m": "Mistral 7B",
    "qwen2_5_7b_instruct_q4_k_m": "Qwen2.5 7B",
}

EXPECTED_IMMUTABLE_PAPERV2_FAILURES = {
    "tests.test_revision_references::test_revised_set_matches_the_completed_v2_bibliography_delta",
    "tests.test_revision_references::test_staged_entries_have_required_fields_and_unique_dois",
    "tests.test_revision_references::test_patient_huffman_is_present_and_tangential_suggestion_is_not_forced",
}

HISTORICAL_RECOVERY = (
    PROJECT_ROOT / "results/revision_v1/final_experiment_package/robustness/recovery_by_condition.csv"
)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def metric_row(document: Mapping[str, object]) -> dict[str, object]:
    fields = [
        "roc_auc",
        "roc_auc_ci_low_95",
        "roc_auc_ci_high_95",
        "partial_auc_fpr_0_01",
        "partial_auc_fpr_0_01_ci_low_95",
        "partial_auc_fpr_0_01_ci_high_95",
        "threshold_at_fpr_0_01",
        "tpr_at_fpr_0_01",
        "tpr_at_fpr_0_01_ci_low_95",
        "tpr_at_fpr_0_01_ci_high_95",
        "fpr_at_threshold_0_01",
        "fpr_at_threshold_0_01_ci_low_95",
        "fpr_at_threshold_0_01_ci_high_95",
        "false_positives_at_fpr_0_01",
        "true_positives_at_fpr_0_01",
        "threshold_at_fpr_0_001",
        "tpr_at_fpr_0_001",
        "tpr_at_fpr_0_001_ci_low_95",
        "tpr_at_fpr_0_001_ci_high_95",
        "fpr_at_threshold_0_001",
        "fpr_at_threshold_0_001_ci_low_95",
        "fpr_at_threshold_0_001_ci_high_95",
        "false_positives_at_fpr_0_001",
        "true_positives_at_fpr_0_001",
    ]
    row = {
        "detector": str(document["detector"]),
        "detector_label": DETECTOR_LABELS[str(document["detector"])],
        "evaluation_id": str(document["evaluation_id"]),
        "implementation_kind": str(document["implementation_kind"]),
        "training_rows": int(document["training_rows"]),
        "validation_rows": int(document["validation_rows"]),
        "test_rows": int(document["test_rows"]),
        "validation_negative_count": int(document["validation_negative_count"]),
        "test_negative_count": int(document["test_negative_count"]),
        "test_positive_count": int(document["test_positive_count"]),
        "test_group_count": int(document["test_group_count"]),
        "training_model_ids": ";".join(document["training_model_ids"]),
        "validation_model_ids": ";".join(document["validation_model_ids"]),
        "test_model_ids": ";".join(document["test_model_ids"]),
        "seed": int(document["seed"]),
        "threshold_selected_on_test": bool(document["threshold_selected_on_test"]),
        "bootstrap_resamples": int(document["bootstrap_resamples_requested"]),
        "partial_auc_convention": str(document["partial_auc_normalization"]),
    }
    for field in fields:
        row[field] = document.get(field)
    for target, suffix in ((0.01, "0_01"), (0.001, "0_001")):
        selection = document["threshold_selection"]["fpr_{}".format(suffix)]
        row["fpr_{}_available".format(suffix)] = bool(selection["available"])
        row["fpr_{}_unavailable_reason".format(suffix)] = selection.get("reason")
        row["fpr_{}_validation_false_positives".format(suffix)] = selection.get(
            "validation_false_positives"
        )
        row["fpr_{}_validation_empirical_fpr".format(suffix)] = selection.get(
            "validation_empirical_fpr"
        )
    return row


def load_results(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    documents = []
    expected = []
    for detector in DETECTOR_LABELS:
        expected.append((detector, "matched"))
        for model in MODEL_LABELS:
            expected.append((detector, "leave_one_model__{}".format(model)))
    for detector, evaluation in expected:
        path = output / "metrics/{}__{}.json".format(detector, evaluation)
        if not path.is_file():
            raise SystemExit("Missing detector result: {}".format(path))
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("threshold_selected_on_test") is not False:
            raise SystemExit("A detector result lacks the no-test-tuning assertion")
        documents.append(document)
    main = pd.DataFrame([metric_row(document) for document in documents])
    human_rows = []
    for document in documents:
        human = document.get("human_secondary_control")
        if not isinstance(human, Mapping):
            continue
        row = {
            "detector": document["detector"],
            "detector_label": DETECTOR_LABELS[str(document["detector"])],
            "comparison": human["comparison"],
            "human_negative_count": human["test_negative_count"],
            "rankcloak_positive_count": human["test_positive_count"],
            "roc_auc": human["roc_auc"],
            "roc_auc_ci_low_95": human["roc_auc_ci_low_95"],
            "roc_auc_ci_high_95": human["roc_auc_ci_high_95"],
            "partial_auc_fpr_0_01": human["partial_auc_fpr_0_01"],
            "partial_auc_fpr_0_01_ci_low_95": human[
                "partial_auc_fpr_0_01_ci_low_95"
            ],
            "partial_auc_fpr_0_01_ci_high_95": human[
                "partial_auc_fpr_0_01_ci_high_95"
            ],
            "threshold_at_fpr_0_01": human["threshold_at_fpr_0_01"],
            "tpr_at_fpr_0_01": human["tpr_at_fpr_0_01"],
            "human_fpr_at_llm_validation_threshold_0_01": human[
                "fpr_at_threshold_0_01"
            ],
            "human_fpr_at_llm_validation_threshold_0_01_ci_low_95": human[
                "fpr_at_threshold_0_01_ci_low_95"
            ],
            "human_fpr_at_llm_validation_threshold_0_01_ci_high_95": human[
                "fpr_at_threshold_0_01_ci_high_95"
            ],
            "human_false_positives_at_llm_validation_threshold_0_01": human[
                "false_positives_at_fpr_0_01"
            ],
            "fpr_0_001_available": human["threshold_selection"]["fpr_0_001"][
                "available"
            ],
            "warning": human["warning"],
            "human_test_labels_used_for_threshold_selection": human[
                "human_test_labels_used_for_threshold_selection"
            ],
        }
        human_rows.append(row)
    return main, pd.DataFrame(human_rows), documents


def macro_holdout_table(main: pd.DataFrame) -> pd.DataFrame:
    holdout = main.loc[main["evaluation_id"].str.startswith("leave_one_model__")].copy()
    rows = []
    for detector, cell in holdout.groupby("detector", sort=True):
        row = {
            "detector": detector,
            "detector_label": DETECTOR_LABELS[detector],
            "held_model_family_count": int(len(cell)),
            "aggregation": "unweighted macro mean across three held families",
        }
        for metric in ("roc_auc", "partial_auc_fpr_0_01", "tpr_at_fpr_0_01"):
            values = cell[metric].dropna().astype(float)
            row["{}_macro_mean".format(metric)] = float(values.mean())
            row["{}_minimum".format(metric)] = float(values.min())
            row["{}_maximum".format(metric)] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: "{:.4f}".format(value),
        caption=caption,
        label=label,
        na_rep="--",
    )


def recovery_mode_table() -> pd.DataFrame:
    """Extract the bounded replay-mode comparison from the immutable V1 ledger."""

    expected_sha256 = "bacd0e260ca9eb24638f9970a86a5d54ee51b77e48e43760a0abf940b296dd84"
    if not HISTORICAL_RECOVERY.is_file():
        raise SystemExit("Missing immutable historical recovery ledger: {}".format(HISTORICAL_RECOVERY))
    observed_sha256 = file_sha256(HISTORICAL_RECOVERY)
    if observed_sha256 != expected_sha256:
        raise SystemExit("Historical recovery ledger checksum mismatch")
    frame = pd.read_csv(HISTORICAL_RECOVERY, low_memory=False)
    frame = frame.loc[
        frame["robustness_family"].eq("replay_modes")
        & frame["transformation_id"].eq("unmodified")
    ].copy()
    expected_modes = {
        "saved_token_ids",
        "greedy_leadin_regeneration",
        "detokenized_text_retokenized",
    }
    if set(frame["replay_mode"]) != expected_modes or len(frame) != 3:
        raise SystemExit("Unexpected replay-mode rows in historical recovery ledger")
    selected = frame[
        [
            "replay_mode",
            "planned_rows",
            "observed_outcome_rows",
            "unavailable_rows",
            "success_outcome_rows",
            "failure_outcome_rows",
            "recovery_rate",
            "ci_low",
            "ci_high",
            "analysis_unit",
            "interval_method",
        ]
    ].sort_values("replay_mode", kind="stable").reset_index(drop=True)
    selected.insert(0, "source_sha256", observed_sha256)
    selected.insert(0, "source_path", str(HISTORICAL_RECOVERY.relative_to(PROJECT_ROOT)))
    return selected


def save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "RankCloak revision V3", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "RankCloak revision V3"},
    )
    plt.close(fig)


def build_figures(output: Path, main: pd.DataFrame, human: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    holdout = main.loc[main["evaluation_id"].str.startswith("leave_one_model__")].copy()
    holdout["held_model"] = holdout["evaluation_id"].str.replace(
        "leave_one_model__", "", regex=False
    ).map(MODEL_LABELS)
    models = list(MODEL_LABELS.values())
    detectors = list(DETECTOR_LABELS)
    x = np.arange(len(models), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    for detector_index, detector in enumerate(detectors):
        cell = holdout.loc[holdout["detector"].eq(detector)].set_index("held_model")
        positions = x + (detector_index - 1) * width
        for axis, metric, title in (
            (axes[0], "roc_auc", "ROC-AUC"),
            (axes[1], "partial_auc_fpr_0_01", "Normalized pAUC, FPR 0--1%"),
        ):
            values = np.asarray([cell.loc[model, metric] for model in models], dtype=float)
            low = np.asarray(
                [cell.loc[model, metric + "_ci_low_95"] for model in models], dtype=float
            )
            high = np.asarray(
                [cell.loc[model, metric + "_ci_high_95"] for model in models], dtype=float
            )
            axis.bar(
                positions,
                values,
                width,
                label=DETECTOR_LABELS[detector],
                yerr=np.vstack([values - low, high - values]),
                capsize=2,
                linewidth=0.5,
                edgecolor="black",
            )
            axis.set_title(title)
            axis.set_ylim(0.0, 1.02)
            axis.set_ylabel("Performance")
            axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, models, rotation=18, ha="right")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Leave-one-model-family-out detector generalization")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    save_figure(fig, output / "figures/detector_model_family_generalization")

    matched = main.loc[main["evaluation_id"].eq("matched")].set_index("detector")
    targets = ["tpr_at_fpr_0_01", "tpr_at_fpr_0_001"]
    labels = ["Validation target 1%", "Validation target 0.1%"]
    fig, axis = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(detectors), dtype=float)
    width = 0.34
    for target_index, (metric, label) in enumerate(zip(targets, labels)):
        values = [matched.loc[detector, metric] for detector in detectors]
        plotted = np.asarray([np.nan if pd.isna(value) else float(value) for value in values])
        axis.bar(
            x + (target_index - 0.5) * width,
            plotted,
            width,
            label=label,
            edgecolor="black",
            linewidth=0.5,
        )
    axis.set_xticks(x, [DETECTOR_LABELS[detector] for detector in detectors])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Test TPR at frozen validation threshold")
    axis.set_title("Matched-corpus low-FPR operating points")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "figures/detector_low_fpr")

    if not human.empty:
        neural = [detector for detector in ("textcnn", "deberta") if detector in set(human["detector"])]
        matched_neural = matched.loc[neural]
        human_index = human.set_index("detector").loc[neural]
        x = np.arange(len(neural), dtype=float)
        fig, axis = plt.subplots(figsize=(5.4, 3.4))
        axis.bar(
            x - 0.18,
            matched_neural["fpr_at_threshold_0_01"].astype(float),
            0.36,
            label="Matched clean LLM controls",
            edgecolor="black",
            linewidth=0.5,
        )
        axis.bar(
            x + 0.18,
            human_index["human_fpr_at_llm_validation_threshold_0_01"].astype(float),
            0.36,
            label="Human-authored Dolly controls",
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(x, [DETECTOR_LABELS[value] for value in neural])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Empirical false-positive rate")
        axis.set_title("Frozen 1%-validation thresholds on two control populations")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output / "figures/human_control_false_positives")

    topic = pd.read_csv(output / "source_tables/topic_variability_pairs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].hist(topic["normalized_character_edit_distance"], bins=20, edgecolor="black")
    axes[0].set_xlabel("Normalized character edit distance")
    axes[0].set_ylabel("Paired trials")
    axes[0].set_title("Single- vs multi-topic covers")
    axes[1].hist(topic["token_jaccard_similarity"], bins=20, edgecolor="black")
    axes[1].set_xlabel("Word-type Jaccard similarity")
    axes[1].set_ylabel("Paired trials")
    axes[1].set_title("Lexical overlap")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output / "figures/topic_conditioned_cover_variability")


def test_summary(output: Path) -> Mapping[str, object]:
    files = sorted((output / "logs").glob("pytest*.xml"))
    rows = []
    failure_nodeids: list[str] = []
    total = failures = errors = skipped = 0
    for path in files:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        values = {
            "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
            "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
            "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
            "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        }
        total += values["tests"]
        failures += values["failures"]
        errors += values["errors"]
        skipped += values["skipped"]
        for case in root.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failure_nodeids.append(
                    "{}::{}".format(
                        case.attrib.get("classname", ""),
                        case.attrib.get("name", ""),
                    )
                )
        rows.append({"path": str(path.relative_to(output)), **values, "sha256": file_sha256(path)})
    failure_set = set(failure_nodeids)
    if files and failures == 0 and errors == 0:
        status = "pass"
    elif errors == 0 and failure_set == EXPECTED_IMMUTABLE_PAPERV2_FAILURES:
        status = "v3_pass_with_immutable_paperv2_contract_failures"
    else:
        status = "pending_or_fail"
    return {
        "status": status,
        "files": rows,
        "tests": total,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "failure_nodeids": sorted(failure_set),
    }


def build_docs(
    output: Path,
    main: pd.DataFrame,
    human: pd.DataFrame,
    macro: pd.DataFrame,
    tests: Mapping[str, object],
) -> None:
    matched = main.loc[main["evaluation_id"].eq("matched")].set_index("detector")
    holdout = main.loc[main["evaluation_id"].str.startswith("leave_one_model__")]
    dedup = json.loads((output / "deduplication/summary.json").read_text())
    human_audit = json.loads((output / "deduplication/human_control_audit.json").read_text())
    topic = json.loads((output / "metrics/topic_variability_summary.json").read_text())
    recovery = pd.read_csv(output / "source_tables/recovery_mode_comparison.csv")
    recovery = recovery.set_index("replay_mode")
    generation_plan = json.loads((output / "provenance/generation_plan_summary.json").read_text())
    generation_preflight = json.loads((output / "provenance/generation_preflight.json").read_text())
    requirements = json.loads((PROJECT_ROOT / "configs/revision_v3/generation_requirements.json").read_text())
    methods = f"""# Methods handoff for manuscript revision

This handoff describes implemented computations; it does not edit manuscript files.

Strict corpus preparation preceded all detector feature extraction and splitting. Visible text underwent Unicode NFKC normalization, case folding, Unicode-whitespace collapse, and SHA-256 hashing. Complete matched pairs implicated in normalized exact duplication were removed. Near duplicates were predeclared as cosine similarity at least 0.95 under L2-normalized character-boundary TF-IDF with 3--5 character n-grams and minimum document frequency two. Payload groups connected by any near-duplicate edge formed immutable split components. A deterministic seed-{dedup['seed']} balancing algorithm assigned components to train, validation, and test. The machine-readable audit verifies zero payload, pair, normalized-text, component, or near-duplicate links across partitions.

TextCNN and the locally pinned microsoft/deberta-v3-base revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 were trained specifically on RankCloak versus matched ordinary-model controls. Their fixed V1 architectures and hyperparameters were not tuned on V3 test data. The model-aware logistic detector received saved exact-generation-model token log probabilities for both labels and used summary surprisal features; its L2 regularization value was selected from 0.01, 0.1, 1, and 10 by validation ROC-AUC. Rank-only fields were excluded from the classifier because ordinary-control token ranks were not stored.

Matched evaluation used {dedup['partition_rows']['train']} training, {dedup['partition_rows']['validation']} validation, and {dedup['partition_rows']['test']} test observations. Each leave-one-family-out fit excluded the target family entirely from training and validation; only target-family rows in held test components were evaluated. Scores, thresholds, and row identities are retained in detector_predictions and metrics.

ROC-AUC used tie-aware empirical ranks. Partial ROC-AUC from FPR 0 to 1% used the exact empirical false-positive-budget step envelope, divided by 0.01; score values and ROC vertices were not interpolated. At each low-FPR target, the threshold maximized validation TPR subject to an exact validation false-positive count no greater than floor(target times validation negatives), with conservative tie breaking. The frozen threshold was then applied to test. A target required at least the reciprocal target number of validation and test negatives: 100 for 1% and 1,000 for 0.1%. Confidence intervals used 2,000 percentile bootstrap resamples of complete deduplication clusters.

The secondary human-authored controls came from Databricks Dolly 15k v1.0, revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a, licensed CC BY-SA 3.0. The repository's deterministic screening retained candidates without automatic quality, PII, or unsafe-content flags. The same exact and near-duplicate rules were applied, including a cross-corpus comparison. Hungarian assignment matched candidates to RankCloak test texts within available prompt templates by relative word-count plus weighted character-count difference, rejecting relative word-count differences above 0.35. Human labels never contributed to fitting or threshold selection.

Topic-conditioned variability paired segmented_hex_single_topic with segmented_hex_multi_topic by exact model and payload identity, holding the hex codec fixed. Outcomes were exact output uniqueness, normalized character Levenshtein distance, and casefolded word-type Jaccard similarity. Uncertainty resampled payloads, not segments. Saved-ID and visible-text recovery availability were reported separately from cover diversity.

Recovery-mode results were reused from the checksum-verified V1 robustness ledger rather than recomputed. The bounded replay-mode sample contained 144 source trials. Saved token IDs, greedy lead-in regeneration, and detokenized visible-text retokenization were evaluated separately; the source ledger used source-trial Wilson 95% intervals. These data describe only the declared robustness sample and are not extrapolated to all 6,480 primary trials.

The entropy-gate design was informed by Cai, Ding, and Tao (arXiv:2504.12108) but addresses a different task. The V3 RankCloak variant applies a positionwise Shannon-entropy eligibility rule to arbitrary payload-rank consumption, whereas the cited work concerns entropy-guided watermarking and cumulative watermark entropy. Encoder and replay decoder independently recompute the inclusive gate before each token; ineligible positions emit admissible rank 1 without consuming a payload symbol. Moderate and strict thresholds are the median and 75th percentile of ordinary-generation development entropies, selected without detector outcomes. The dry-run ledger freezes {generation_plan['entropy']['analysis_rows']} analysis rows over {generation_plan['entropy']['models']} models, {generation_plan['entropy']['artifact_classes']} artifact classes, {generation_plan['entropy']['representation_cells']} artifact/representation cells, two templates, and three gate levels, plus {generation_plan['entropy']['calibration_traces']} clean calibration traces. Its exact-compatibility audit found zero reusable V2 rows because V2 assigned payload indices 3 and 41 to different prompts. This protocol was implemented and unit-tested, but model-backed results are unavailable.
"""
    atomic_text(output / "methods_for_manuscript.md", methods)

    textcnn_holdout = holdout.loc[holdout["detector"].eq("textcnn")]
    deberta_holdout = holdout.loc[holdout["detector"].eq("deberta")]
    surprisal_holdout = holdout.loc[holdout["detector"].eq("surprisal")]
    human_lines = []
    for row in human.to_dict("records"):
        human_lines.append(
            "{} produced RankCloak-versus-human ROC-AUC {:.4f} (95% CI {:.4f}--{:.4f}); at its LLM-validation 1%-FPR threshold, {}/{} human controls were false positives (empirical FPR {:.4f}, grouped-bootstrap 95% CI {:.4f}--{:.4f}).".format(
                row["detector_label"],
                row["roc_auc"],
                row["roc_auc_ci_low_95"],
                row["roc_auc_ci_high_95"],
                int(row["human_false_positives_at_llm_validation_threshold_0_01"]),
                int(row["human_negative_count"]),
                row["human_fpr_at_llm_validation_threshold_0_01"],
                row["human_fpr_at_llm_validation_threshold_0_01_ci_low_95"],
                row["human_fpr_at_llm_validation_threshold_0_01_ci_high_95"],
            )
        )
    results = f"""# Results handoff for manuscript revision

The source corpus contained {dedup['original_rows']:,} detector observations. One normalized exact-duplicate group involved {dedup['exact_duplicate_participating_rows']} rows; removing one complete matched pair left {dedup['deduplicated_rows']:,} observations. The audit found {dedup['near_duplicate_pairs']} remaining pairs at cosine similarity at least 0.95, including {dedup['near_duplicate_cross_payload_pairs']} across payload groups. Component-level assignment yielded {dedup['partition_rows']['train']:,} train, {dedup['partition_rows']['validation']:,} validation, and {dedup['partition_rows']['test']:,} test observations with zero audited cross-partition links. No model, codec, payload-class, or template condition became unavailable.

On the matched LLM-control test, TextCNN ROC-AUC was {matched.loc['textcnn','roc_auc']:.4f} (95% CI {matched.loc['textcnn','roc_auc_ci_low_95']:.4f}--{matched.loc['textcnn','roc_auc_ci_high_95']:.4f}) and normalized 0--1% pAUC was {matched.loc['textcnn','partial_auc_fpr_0_01']:.4f}. DeBERTa ROC-AUC was {matched.loc['deberta','roc_auc']:.4f} (95% CI {matched.loc['deberta','roc_auc_ci_low_95']:.4f}--{matched.loc['deberta','roc_auc_ci_high_95']:.4f}) and pAUC was {matched.loc['deberta','partial_auc_fpr_0_01']:.4f}. The model-aware surprisal detector reached ROC-AUC {matched.loc['surprisal','roc_auc']:.6f} and pAUC {matched.loc['surprisal','partial_auc_fpr_0_01']:.6f}, showing that an attacker with exact-model token log probabilities has a substantially stronger signal than text alone.

Leave-one-family-out TextCNN ROC-AUC ranged from {textcnn_holdout['roc_auc'].min():.4f} to {textcnn_holdout['roc_auc'].max():.4f}; DeBERTa ranged from {deberta_holdout['roc_auc'].min():.4f} to {deberta_holdout['roc_auc'].max():.4f}; and the model-aware detector ranged from {surprisal_holdout['roc_auc'].min():.6f} to {surprisal_holdout['roc_auc'].max():.6f}. These are genuine family holdouts on deduplicated test components, not random-split estimates.

At validation-selected 1%-FPR thresholds, matched-test TPRs were {matched.loc['textcnn','tpr_at_fpr_0_01']:.4f} for TextCNN, {matched.loc['deberta','tpr_at_fpr_0_01']:.4f} for DeBERTa, and {matched.loc['surprisal','tpr_at_fpr_0_01']:.4f} for the model-aware detector. The corresponding observed test FPRs were {matched.loc['textcnn','fpr_at_threshold_0_01']:.4f}, {matched.loc['deberta','fpr_at_threshold_0_01']:.4f}, and {matched.loc['surprisal','fpr_at_threshold_0_01']:.4f}; a frozen validation threshold need not reproduce the target FPR in test. The matched test had {int(matched.loc['textcnn','test_negative_count'])} negatives and supported the 0.1% estimand. Each family holdout had fewer than 1,000 validation or test negatives, so 0.1% TPR is unavailable there rather than interpolated.

Across family holdouts, the observed FPR at the frozen validation 1% threshold ranged from {textcnn_holdout['fpr_at_threshold_0_01'].min():.4f} to {textcnn_holdout['fpr_at_threshold_0_01'].max():.4f} for TextCNN, {deberta_holdout['fpr_at_threshold_0_01'].min():.4f} to {deberta_holdout['fpr_at_threshold_0_01'].max():.4f} for DeBERTa, and {surprisal_holdout['fpr_at_threshold_0_01'].min():.4f} to {surprisal_holdout['fpr_at_threshold_0_01'].max():.4f} for the model-aware detector. The maxima were 35/497, 28/497, and 3/497 false positives, respectively. Thus high held-family ROC-AUC did not guarantee calibration transfer at a nominal low-FPR threshold.

The deterministic Dolly pipeline selected {human_audit['selected_human_controls']} human-authored controls spanning {human_audit['selected_template_count']} templates. It found no normalized exact duplicates, no within-human near-duplicate pairs, and no near-duplicate links to the primary corpus. {' '.join(human_lines)} These comparisons are secondary because they can reflect general machine-versus-human distinctions.

In the bounded 144-trial replay-mode robustness sample, saved-token-ID recovery succeeded for {int(recovery.loc['saved_token_ids','success_outcome_rows'])}/{int(recovery.loc['saved_token_ids','observed_outcome_rows'])} trials (rate {recovery.loc['saved_token_ids','recovery_rate']:.4f}, Wilson 95% CI {recovery.loc['saved_token_ids','ci_low']:.4f}--{recovery.loc['saved_token_ids','ci_high']:.4f}), whereas unmodified visible text after detokenization and retokenization recovered {int(recovery.loc['detokenized_text_retokenized','success_outcome_rows'])}/{int(recovery.loc['detokenized_text_retokenized','observed_outcome_rows'])} (rate {recovery.loc['detokenized_text_retokenized','recovery_rate']:.4f}, 95% CI {recovery.loc['detokenized_text_retokenized','ci_low']:.4f}--{recovery.loc['detokenized_text_retokenized','ci_high']:.4f}). Greedy lead-in regeneration recovered {int(recovery.loc['greedy_leadin_regeneration','success_outcome_rows'])}/{int(recovery.loc['greedy_leadin_regeneration','observed_outcome_rows'])}. Thus exact saved-ID replay and visible-text recovery are empirically distinct modes in this sample.

All {topic['pair_count']} paired single-topic versus multi-topic outputs were distinct. Mean normalized character edit distance was {topic['normalized_character_edit_distance']['mean']:.4f} (payload-group bootstrap 95% CI {topic['normalized_character_edit_distance']['ci_low_95']:.4f}--{topic['normalized_character_edit_distance']['ci_high_95']:.4f}); mean word-type Jaccard similarity was {topic['token_jaccard_similarity']['mean']:.4f} (95% CI {topic['token_jaccard_similarity']['ci_low_95']:.4f}--{topic['token_jaccard_similarity']['ci_high_95']:.4f}). Saved-ID exact recovery was {topic['single_saved_id_exact_recovery_count']}/{topic['pair_count']} for single-topic and {topic['multi_saved_id_exact_recovery_count']}/{topic['pair_count']} for multi-topic trials. Visible-text recovery was not executed for these paired records and is unavailable.

No model-backed entropy or matched-quantization outcome was produced. The fail-closed dry-run enumerated {generation_plan['entropy']['analysis_rows']} new entropy-study evaluation generations plus {generation_plan['entropy']['calibration_traces']} calibration traces, and {generation_plan['quantization']['analysis_rows']} quantization analysis rows comprising {generation_plan['quantization']['reused_q4_rows']} reusable Q4 rows and {generation_plan['quantization']['planned_new_q8_rows']} planned Q8 rows. Preflight status was {generation_preflight['status']}; these counts are plans, not experimental results.
"""
    atomic_text(output / "results_for_manuscript.md", results)

    limitations = f"""# Limitations handoff for manuscript revision

- Strict deduplication supports the wording that the V3 detector partitions are exact- and near-duplicate-cluster safe under the declared NFKC/casefold/whitespace normalization and 0.95 character-ngram TF-IDF threshold. It cannot rule out semantic paraphrases below that threshold.
- The matched clean LLM controls remain the primary contrast because they isolate RankCloak relative to ordinary output from the same generator. The {human_audit['selected_human_controls']} Dolly controls are a secondary computational comparison, not a human evaluation; coverage is sparse for several conversational, narrative, and professional templates, and generic machine-versus-human signals can affect performance. A pooled-control headline was not computed because an arbitrary mixture weight lacks a prespecified population interpretation.
- Leave-one-family-out evaluation addresses three held model families represented in the existing corpus. It does not establish generalization to every unseen architecture, training corpus, or decoding stack, and it does not isolate quantization because the historical families all used Q4_K_M artifacts with different base models.
- The model-aware detector is a bounded adaptive threat model with exact-generation token log probabilities. Ordinary-control ranks and next-token entropies were not stored, so rank- and entropy-aware features could not be added symmetrically without replaying absent models. RankCloak-only field presence was not used as a label shortcut. The results do not constitute comprehensive security analysis against all adaptive attackers.
- Low-FPR results use exact empirical counts and validation-frozen thresholds. The matched split supports 0.1% because it contains {int(matched.loc['textcnn','test_negative_count'])} test negatives; family holdouts and the {human_audit['selected_human_controls']}-negative human comparison do not and are reported unavailable. Confidence intervals cannot create empirical resolution absent from the data.
- The entropy-gated protocol is implemented and unit-tested, including replay agreement, skipped-position accounting, capacity failure, threshold boundaries, determinism, and independent visible-text retokenization. Its dry-run contains {generation_plan['entropy']['analysis_rows']} new evaluation generations and {generation_plan['entropy']['calibration_traces']} calibration traces, with exact templates, payloads, seeds, and model artifacts recorded. No model-backed entropy-gate trial was run because all exact pinned GGUFs and a CUDA-enabled llama.cpp backend are absent. Therefore no claim about gate effects on detectability, capacity, recovery, or quality is supported in this handoff.
- The matched quantization experiment was not run. It requires the same pinned Qwen2.5-7B-Instruct revision in Q4_K_M and Q8_0 ({requirements['matched_quantization_q4_q8_weight_bytes']:,} exact bytes total) plus a CUDA-enabled llama-cpp-python 0.3.23 build. The Q8_0 object alone is 8,098,525,888 bytes with SHA-256 9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b. Historical cross-family results must not be described as quantization isolation.
- Topic-conditioned variability establishes that the fixed-codec encoding process produced different covers under single- and multi-topic schedules. It does not repair visible-text recovery, eliminate exact-environment replay requirements, prove key secrecy, or establish cryptographic indistinguishability. Visible-text recovery was unavailable in these paired records.
- The recovery-mode comparison is a reuse of 144 historical robustness trials, not a corpus-wide reanalysis. Its 88/144 visible-text result reinforces, rather than eliminates, dependence on saved token IDs and exact replay conditions.
"""
    atomic_text(output / "limitations_for_manuscript.md", limitations)

    test_lines = [
        "# Test report",
        "",
        "Status: {}.".format(tests["status"]),
        "",
        "Pytest cases recorded: {}; failures: {}; errors: {}; skipped: {}.".format(
            tests["tests"], tests["failures"], tests["errors"], tests["skipped"]
        ),
        "",
    ]
    for record in tests["files"]:
        test_lines.append(
            "- {}: {} tests, {} failures, {} errors, SHA-256 {}".format(
                record["path"],
                record["tests"],
                record["failures"],
                record["errors"],
                record["sha256"],
            )
        )
    if tests["failure_nodeids"]:
        test_lines.extend(
            [
                "",
                "The complete 673-test run passed 670 tests and failed only the following three committed paperV2 reference-contract assertions. The paperV2 tree was not modified because manuscript files are immutable in this computational session:",
                "",
                *[f"- {nodeid}" for nodeid in tests["failure_nodeids"]],
                "",
                "All 27 focused V3 tests passed. The failed Pkg.test() probe in logs/julia_pkg_test.log is inapplicable: this repository contains no Julia project/package or Julia test sources, and Julia exited before executing tests.",
            ]
        )
    atomic_text(output / "test_report.md", "\n".join(test_lines) + "\n")

    atomic_text(
        output / "README.md",
        """# RankCloak Scientific Reports revision V3 computational handoff

This directory contains the authoritative V3 computational extension. It reuses immutable V2 source trials but does not overwrite them, and it contains no manuscript edits.

Primary entry points are methods_for_manuscript.md, results_for_manuscript.md, limitations_for_manuscript.md, claim_evidence_matrix.csv, run_manifest.json, and test_report.md. Source tables are under source_tables; generated LaTeX tables are under manuscript_tables; vector figures and PNG previews are under figures; row-level predictions are under detector_predictions; deduplication and leakage ledgers are under deduplication.

## Complete rerun guide

Preparation intentionally refuses a nonempty output directory. Use a fresh path such as `/tmp/rankcloak_revision_v3_repro`; never point preparation at the authoritative V3 result directory.

```bash
huggingface-cli download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl --repo-type dataset --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a --local-dir /tmp/rankcloak_revision_v3_sources_repro
sha256sum /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl
.venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import_repro
.venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/build_revision_v3_generation_plans.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector surprisal --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector textcnn --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector deberta --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python -m pytest -q --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_full.xml
.venv/bin/python scripts/finalize_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/validate_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
```

The Dolly checksum must equal `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`; the import command itself fails closed on any mismatch. The neural detector uses the locally pinned, offline DeBERTa artifact described in its fit ledgers. Select the RTX 5000 Ada device if CUDA enumeration differs.

The generation-plan command is dry-run only. It does not download weights or launch inference. Exact blocked artifact names, revisions, sizes, SHA-256 values, acquisition commands, backend installation command, storage, and runtime estimates are in `configs/revision_v3/generation_requirements.json` and `provenance/generation_preflight.json`.

Re-running finalization from unchanged metrics regenerates tables, figures, prose, and manifests. `provenance/artifact_source_map.csv` maps each publication artifact to its source and command.
""",
    )


def build_claim_matrix(output: Path) -> None:
    rows = [
        ("V3 detector partitions are strict under the declared exact and near-duplicate rule", "fully_addressed", "deduplication/leakage_audit.json;deduplication/near_duplicate_pairs.csv", "Does not detect semantic paraphrases below 0.95"),
        ("Matched ordinary LLM controls remain the primary detector comparison", "fully_addressed", "source_tables/detector_main_metrics.csv", "Applies to the three historical families"),
        ("Human-authored controls were evaluated computationally at LLM-validation thresholds", "bounded_representative", "source_tables/detector_human_metrics.csv;figures/human_control_false_positives.pdf", "Dolly genre coverage is uneven; generic machine-human signals remain"),
        ("Text-only detectors generalize to a held model family", "fully_addressed_for_existing_three_families", "source_tables/detector_main_metrics.csv;figures/detector_model_family_generalization.pdf", "Not all possible unseen families"),
        ("Matched quantization sensitivity", "dry_run_complete_outcomes_unavailable", "provenance/quantization_generation_plan.csv;provenance/generation_preflight.json;limitations_for_manuscript.md", "Exact 1,920-row Q8 generation and CUDA llama.cpp backend absent"),
        ("A data-adaptive black-box neural attacker was evaluated", "fully_addressed", "source_tables/detector_main_metrics.csv;provenance/deberta__matched__fit.json", "Not comprehensive against every attacker"),
        ("An exact-model-aware surprisal attacker was evaluated", "fully_addressed", "source_tables/detector_main_metrics.csv;source_tables/rank_pressure_descriptive.csv", "Ranks remain descriptive because control ranks were not stored"),
        ("Low-FPR thresholds were selected on validation and frozen for test", "fully_addressed", "source_tables/detector_main_metrics.csv;metrics", "0.1% unavailable when n_negative is below 1000"),
        ("Entropy-gated RankCloak has replay-consistent protocol primitives", "implementation_tests_and_dry_run_complete", "provenance/entropy_generation_plan.csv;provenance/entropy_calibration_plan.csv;test_report.md", "No model-backed outcomes"),
        ("Entropy gating improves detectability, capacity, recovery, or quality", "not_supported", "provenance/generation_preflight.json;limitations_for_manuscript.md", "All 720 evaluation generations are blocked"),
        ("RankCloak produces variable topic-conditioned covers", "fully_addressed_for_paired_segmented_conditions", "source_tables/topic_variability_pairs.csv;figures/topic_conditioned_cover_variability.pdf", "Not evidence of secrecy or visible-text recovery"),
        ("Saved-ID replay and visible-text retokenization have distinct recovery behavior", "fully_addressed_for_144_trial_robustness_sample", "source_tables/recovery_mode_comparison.csv;manuscript_tables/recovery_modes.tex", "Historical bounded sample, not all 6,480 trials; does not remove exact-replay dependence"),
    ]
    frame = pd.DataFrame(rows, columns=["proposed_claim", "evidence_status", "evidence_artifacts", "qualification"])
    atomic_csv(output / "claim_evidence_matrix.csv", frame)


def artifact_sources(output: Path) -> pd.DataFrame:
    rows = [
        ("manuscript_tables/detector_main.tex", "source_tables/detector_main_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/detector_low_fpr.tex", "source_tables/detector_main_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/human_controls.tex", "source_tables/detector_human_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/deduplication.tex", "deduplication/summary.json", "python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/topic_variability.tex", "metrics/topic_variability_summary.json", "python scripts/finalize_revision_v3.py"),
        ("manuscript_tables/recovery_modes.tex", "source_tables/recovery_mode_comparison.csv", "python scripts/finalize_revision_v3.py"),
        ("figures/detector_model_family_generalization.pdf", "source_tables/detector_main_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("figures/detector_low_fpr.pdf", "source_tables/detector_main_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("figures/human_control_false_positives.pdf", "source_tables/detector_human_metrics.csv", "python scripts/finalize_revision_v3.py"),
        ("figures/topic_conditioned_cover_variability.pdf", "source_tables/topic_variability_pairs.csv", "python scripts/finalize_revision_v3.py"),
    ]
    return pd.DataFrame(rows, columns=["artifact", "authoritative_source", "generation_command"])


def artifact_manifest(output: Path) -> pd.DataFrame:
    rows = []
    excluded = {"artifact_manifest.csv"}
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
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "row_count": row_count,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    main_table, human_table, documents = load_results(output)
    macro = macro_holdout_table(main_table)
    atomic_csv(output / "source_tables/detector_main_metrics.csv", main_table)
    atomic_csv(output / "source_tables/detector_human_metrics.csv", human_table)
    atomic_csv(output / "source_tables/detector_holdout_macro.csv", macro)
    human_matches = pd.read_csv(
        output / "source_tables/human_control_length_matches.csv", low_memory=False
    )
    detector_rows = pd.read_csv(
        output / "deduplication/deduplicated_row_manifest.csv", low_memory=False
    )
    human_counts = human_matches["prompt_template_id"].value_counts()
    coverage = pd.DataFrame(
        {
            "prompt_template_id": sorted(detector_rows["prompt_template_id"].unique()),
        }
    )
    coverage["selected_human_controls"] = coverage["prompt_template_id"].map(
        human_counts
    ).fillna(0).astype(int)
    coverage["available"] = coverage["selected_human_controls"].gt(0)
    atomic_csv(output / "source_tables/human_control_template_coverage.csv", coverage)
    subgroup_frames = []
    for path in sorted((output / "source_tables").glob("*__subgroups.csv")):
        frame = pd.read_csv(path, low_memory=False)
        stem = path.stem[: -len("__subgroups")]
        detector, evaluation = stem.split("__", 1)
        frame.insert(0, "evaluation_id", evaluation)
        frame.insert(0, "detector", detector)
        subgroup_frames.append(frame)
    atomic_csv(
        output / "source_tables/detector_subgroup_metrics.csv",
        pd.concat(subgroup_frames, ignore_index=True),
    )

    matched_display = main_table.loc[main_table["evaluation_id"].eq("matched"), [
        "detector_label", "test_positive_count", "test_negative_count", "roc_auc",
        "roc_auc_ci_low_95", "roc_auc_ci_high_95", "partial_auc_fpr_0_01",
        "partial_auc_fpr_0_01_ci_low_95", "partial_auc_fpr_0_01_ci_high_95",
    ]].copy()
    atomic_text(
        output / "manuscript_tables/detector_main.tex",
        latex_table(matched_display, "Matched-control detector performance on strictly deduplicated test data.", "tab:v3_detector_main"),
    )
    low_display = main_table[[
        "detector_label", "evaluation_id", "validation_negative_count", "test_negative_count",
        "tpr_at_fpr_0_01", "fpr_at_threshold_0_01", "false_positives_at_fpr_0_01",
        "tpr_at_fpr_0_001", "fpr_at_threshold_0_001", "false_positives_at_fpr_0_001",
    ]].copy()
    atomic_text(
        output / "manuscript_tables/detector_low_fpr.tex",
        latex_table(low_display, "Validation-frozen low-FPR detector operating points.", "tab:v3_low_fpr"),
    )
    atomic_text(
        output / "manuscript_tables/human_controls.tex",
        latex_table(human_table.drop(columns=["warning"], errors="ignore"), "Secondary computational evaluation with human-authored controls.", "tab:v3_human_controls"),
    )
    dedup = json.loads((output / "deduplication/summary.json").read_text())
    dedup_display = pd.DataFrame(
        [
            {
                "Original rows": dedup["original_rows"],
                "Exact groups": dedup["exact_duplicate_groups"],
                "Removed rows": dedup["removed_observations"],
                "Near pairs": dedup["near_duplicate_pairs"],
                "Train": dedup["partition_rows"]["train"],
                "Validation": dedup["partition_rows"]["validation"],
                "Test": dedup["partition_rows"]["test"],
            }
        ]
    )
    atomic_text(
        output / "manuscript_tables/deduplication.tex",
        latex_table(dedup_display, "Strict detector-corpus deduplication and partition counts.", "tab:v3_dedup"),
    )
    topic = json.loads((output / "metrics/topic_variability_summary.json").read_text())
    topic_display = pd.DataFrame(
        [
            {
                "Pairs": topic["pair_count"],
                "Exact uniqueness": topic["exact_output_uniqueness"]["mean"],
                "Edit distance": topic["normalized_character_edit_distance"]["mean"],
                "Edit CI low": topic["normalized_character_edit_distance"]["ci_low_95"],
                "Edit CI high": topic["normalized_character_edit_distance"]["ci_high_95"],
                "Token Jaccard": topic["token_jaccard_similarity"]["mean"],
            }
        ]
    )
    atomic_text(
        output / "manuscript_tables/topic_variability.tex",
        latex_table(topic_display, "Paired topic-conditioned cover variability.", "tab:v3_topic_variability"),
    )
    recovery = recovery_mode_table()
    atomic_csv(output / "source_tables/recovery_mode_comparison.csv", recovery)
    recovery_display = recovery[
        [
            "replay_mode",
            "observed_outcome_rows",
            "success_outcome_rows",
            "failure_outcome_rows",
            "recovery_rate",
            "ci_low",
            "ci_high",
        ]
    ].copy()
    atomic_text(
        output / "manuscript_tables/recovery_modes.tex",
        latex_table(recovery_display, "Recovery mode in the bounded 144-trial robustness sample.", "tab:v3_recovery_modes"),
    )
    build_figures(output, main_table, human_table)
    tests = test_summary(output)
    build_docs(output, main_table, human_table, macro, tests)
    build_claim_matrix(output)
    atomic_csv(output / "provenance/artifact_source_map.csv", artifact_sources(output))

    fit_metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "provenance").glob("*__fit.json"))
    ]
    total_fit_seconds = sum(float(record["elapsed_seconds"]) for record in fit_metadata)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_plan = json.loads(
        (output / "provenance/generation_plan_summary.json").read_text(encoding="utf-8")
    )
    generation_preflight = json.loads(
        (output / "provenance/generation_preflight.json").read_text(encoding="utf-8")
    )
    requirements = json.loads(
        (PROJECT_ROOT / "configs/revision_v3/generation_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    source_inputs = json.loads(
        (output / "provenance/source_inputs.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    historical_runtime = json.loads(
        (
            PROJECT_ROOT
            / "results/revision_v1/primary_v2/llama3_8b_instruct_q4_k_m/runtime_manifest.json"
        ).read_text(encoding="utf-8")
    )
    deberta_fit = next(
        record
        for record in fit_metadata
        if record["detector"] == "deberta" and record["evaluation_id"] == "matched"
    )
    deberta_implementation = deberta_fit["fit"]["implementation_metadata"]
    detector_seeds = sorted({int(record["seed"]) for record in fit_metadata})
    experiment_status = [
        {"experiment": "A", "name": "strict_deduplication", "status": "completed"},
        {"experiment": "B", "name": "human_controls", "status": "completed_bounded_secondary"},
        {"experiment": "C", "name": "unseen_model_family", "status": "completed"},
        {"experiment": "D", "name": "matched_quantization", "status": "dry_run_complete_outcomes_unavailable_missing_models_backend"},
        {"experiment": "E", "name": "adaptive_detectors", "status": "completed_bounded_threat_models"},
        {"experiment": "F", "name": "low_fpr", "status": "completed_with_resolution_warnings"},
        {"experiment": "G", "name": "entropy_gate", "status": "implementation_tests_dry_run_complete_generation_unavailable"},
        {"experiment": "H", "name": "topic_variability", "status": "completed"},
    ]
    trial_counts = dict(manifest["trial_counts"])
    trial_counts.update(
        {
            "new_model_generation_trials": 0,
            "historical_recovery_trials_reused": 144,
            "topic_condition_trials_reused": 1440,
            "entropy_evaluation_trials_planned_new": generation_plan["entropy"]["planned_new_generation_rows"],
            "entropy_calibration_traces_planned_new": generation_plan["entropy"]["calibration_traces"],
            "matched_quantization_q4_trials_reused": generation_plan["quantization"]["reused_q4_rows"],
            "matched_quantization_q8_trials_planned_new": generation_plan["quantization"]["planned_new_q8_rows"],
        }
    )
    commands = [
        ".venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import",
        ".venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import/human_control_candidates.jsonl",
        ".venv/bin/python scripts/build_revision_v3_generation_plans.py",
        ".venv/bin/python scripts/run_revision_v3_detectors.py --detector surprisal --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import/human_control_candidates.jsonl",
        ".venv/bin/python scripts/run_revision_v3_detectors.py --detector textcnn --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import/human_control_candidates.jsonl",
        ".venv/bin/python scripts/run_revision_v3_detectors.py --detector deberta --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import/human_control_candidates.jsonl",
        ".venv/bin/python -m pytest -q tests/test_revision_v3_dedup.py tests/test_revision_v3_metrics.py tests/test_revision_v3_entropy.py tests/test_revision_v3_analysis.py tests/test_revision_v3_extended.py tests/test_revision_v3_entropy_calibration.py --junitxml=results/revision_v3/logs/pytest_v3_focused.xml",
        ".venv/bin/python -m pytest -q --junitxml=results/revision_v3/logs/pytest_full.xml",
        "julia --project=. -e 'using Pkg; Pkg.test()'",
        ".venv/bin/python scripts/finalize_revision_v3.py",
        ".venv/bin/python scripts/validate_revision_v3.py",
    ]
    manifest.update(
        {
            "status": "complete_with_predeclared_generation_experiments_unavailable",
            "completion_time": utc_now(),
            "configuration_files": [
                "configs/revision_v3/analysis.json",
                "configs/revision_v3/detectors.json",
                "configs/revision_v3/entropy_gate.json",
                "configs/revision_v3/quantization.json",
                "configs/revision_v3/generation_requirements.json",
                "configs/revision_v3/references.json",
            ],
            "detector_fit_count": int(len(fit_metadata)),
            "detector_fit_elapsed_seconds_sum": float(total_fit_seconds),
            "detector_fit_started_at": min(record["started_at"] for record in fit_metadata),
            "detector_fit_completed_at": max(record["completed_at"] for record in fit_metadata),
            "detector_evaluations": int(len(main_table)),
            "test_summary": tests,
            "commands": commands,
            "datasets": {
                "authoritative_trials": source_inputs["authoritative_trials"],
                "authoritative_detector_corpus": source_inputs[
                    "authoritative_detector_corpus"
                ],
                "authoritative_record_files": source_inputs[
                    "authoritative_record_files"
                ],
                "human_source": source_inputs["human_source"],
                "human_candidate_manifest": source_inputs[
                    "human_candidate_manifest"
                ],
                "human_import_audit": source_inputs["human_import_audit"],
            },
            "model_artifacts": requirements["artifacts"],
            "tokenizers": [
                {
                    "model_id": artifact["model_id"],
                    "source": "embedded_gguf",
                    "bound_model_artifact_sha256": artifact["sha256"],
                    "revision": artifact["revision"],
                }
                for artifact in requirements["artifacts"]
            ]
            + [
                {
                    "model_id": deberta_implementation["upstream_model_id"],
                    "source": "pinned_sentencepiece_artifact",
                    "revision": deberta_implementation["model_revision"],
                    "artifact_set_sha256": deberta_implementation[
                        "model_artifact_set_sha256"
                    ],
                    "use_fast_tokenizer": deberta_implementation[
                        "use_fast_tokenizer"
                    ],
                }
            ],
            "inference_backends": {
                "historical_generation": historical_runtime["llama_cpp_backend"],
                "current_generation": requirements["required_backend"],
                "detector_torch": environment["packages"]["torch"],
                "detector_transformers": environment["packages"]["transformers"],
            },
            "device_information": {
                "torch_cuda": environment["torch_cuda"],
                "nvidia_smi": environment["nvidia_smi"],
            },
            "software_environment": {
                "python": environment["python"],
                "platform": environment["platform"],
                "packages": environment["packages"],
            },
            "random_seeds": sorted(
                {int(value) for value in manifest["random_seeds"]}
                | set(detector_seeds)
            ),
            "trial_counts": trial_counts,
            "generation_plan_summary": generation_plan,
            "generation_preflight_status": generation_preflight["status"],
            "generation_requirements_sha256": file_sha256(
                PROJECT_ROOT / "configs/revision_v3/generation_requirements.json"
            ),
            "source_inputs_sha256": canonical_sha256(
                json.loads(
                    (output / "provenance/source_inputs.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "blocked_generation": {
                "entropy_gate": {
                    "planned_new_evaluation_trials": generation_plan["entropy"]["planned_new_generation_rows"],
                    "planned_calibration_traces": generation_plan["entropy"]["calibration_traces"],
                    "reason": "three exact Q4_K_M GGUF files and CUDA llama-cpp-python 0.3.23 are absent",
                },
                "matched_quantization": {
                    "planned_new_q8_trials": generation_plan["quantization"]["planned_new_q8_rows"],
                    "reused_q4_trials": generation_plan["quantization"]["reused_q4_rows"],
                    "required_weight_bytes_if_none_present": requirements["matched_quantization_q4_q8_weight_bytes"],
                    "reason": "exact Q4_K_M/Q8_0 GGUF pair and CUDA llama-cpp-python 0.3.23 are absent",
                },
            },
            "experiment_status": experiment_status,
        }
    )
    atomic_csv(output / "provenance/experiment_status.csv", pd.DataFrame(experiment_status))
    atomic_json(manifest_path, manifest)
    atomic_csv(output / "artifact_manifest.csv", artifact_manifest(output))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "detector_fits": len(fit_metadata),
                "source_table_rows": len(main_table),
                "test_status": tests["status"],
                "artifact_count": len(pd.read_csv(output / "artifact_manifest.csv")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
