#!/usr/bin/env python3
"""Fit dedup-safe detectors for V3 entropy and quantization generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_revision_v3 import atomic_csv, atomic_json, utc_now  # noqa: E402
from rankcloak.revision_detection import run_configured_detector  # noqa: E402
from rankcloak.revision_v3_analysis import (  # noqa: E402
    fit_surprisal_detector,
    surprisal_features_from_log_probabilities,
)
from rankcloak.revision_v3_generation import canonical_sha256, file_sha256  # noqa: E402
from rankcloak.revision_v3_generation_detection import (  # noqa: E402
    locked_partition_deduplicate,
)
from rankcloak.revision_v3_metrics import (  # noqa: E402
    evaluate_validation_frozen_detector,
    frozen_threshold_counts,
    roc_auc,
    empirical_partial_auc,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/revision_v3/detectors.json"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "configs/revision_v3/analysis.json"
HISTORICAL_CORPUS = (
    PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl"
)
STUDY_EVALUATIONS = {
    "entropy": ("entropy_gates",),
    "quantization": ("q4_to_q8", "q8_to_q4", "pooled_quantizations"),
}


def stable_seed(seed: int, *parts: object) -> int:
    material = "\x1f".join([str(int(seed)), *(str(part) for part in parts)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def load_json_records(root: Path) -> dict[str, Mapping[str, object]]:
    records: dict[str, Mapping[str, object]] = {}
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        plan_id = str(record["plan_id"])
        if plan_id in records:
            raise SystemExit(f"duplicate model-backed generation record: {plan_id}")
        records[plan_id] = record
    return records


def load_historical_frame(output: Path) -> pd.DataFrame:
    source = pd.read_json(HISTORICAL_CORPUS, lines=True)
    manifest = pd.read_csv(
        output / "deduplication/deduplicated_row_manifest.csv", low_memory=False
    )
    extras = manifest[
        ["row_id", "normalized_text_sha256", "dedup_cluster_id", "partition"]
    ]
    frame = source.merge(extras, on="row_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise SystemExit("historical prepared frame does not match its dedup manifest")
    return frame.sort_values("row_id").reset_index(drop=True)


def entropy_source_frame(output: Path) -> pd.DataFrame:
    generated_path = output / "generation/analysis/entropy_detector_corpus.csv"
    generated = pd.read_csv(generated_path, low_memory=False)
    if len(generated) != 720 or set(generated["population"]) != {"rankcloak", "ordinary_control"}:
        raise SystemExit("entropy detector corpus is not the complete 720-row evaluation")
    historical = load_historical_frame(output)
    excluded_payloads = set(generated["payload_name"].astype(str))
    historical = historical.loc[
        historical["partition"].isin(["train", "validation"])
        & ~historical["payload_group_id"].astype(str).isin(excluded_payloads)
    ].copy()
    historical["generation_study"] = "entropy"
    historical["evaluation_id"] = "entropy_gates"
    historical["study_source"] = "historical_deduplicated_fit"
    historical["gate_level"] = "not_applicable_historical_fit"
    historical["quantization"] = "historical_mixed_q4"
    historical["representation_name"] = historical["representation_name"].astype(str)

    new = pd.DataFrame(
        {
            "row_id": "entropy_generated__" + generated["plan_id"].astype(str),
            "pair_id": generated["pairing_unit_id"].astype(str),
            "payload_group_id": generated["payload_name"].astype(str),
            "text": generated["full_text"].astype(str),
            "label": generated["label"].astype(int),
            "partition": "test",
            "model_id": generated["model_id"].astype(str),
            "codec_id": generated["codec_id"].astype(str),
            "payload_class": generated["payload_class"].astype(str),
            "prompt_template_id": generated["prompt_template_id"].astype(str),
            "representation_name": generated["representation_name"].astype(str),
            "generation_study": "entropy",
            "evaluation_id": "entropy_gates",
            "study_source": "new_model_backed_test",
            "gate_level": generated["gate_level"].astype(str),
            "quantization": generated["quantization"].astype(str),
            "plan_id": generated["plan_id"].astype(str),
            "experimental_cell_id": generated["experimental_cell_id"].astype(str),
        }
    )
    paired = new.groupby("pair_id")["label"].agg(lambda values: set(map(int, values)))
    if not paired.map(lambda values: values == {0, 1}).all():
        raise SystemExit("an entropy matched pair does not contain both labels")
    return pd.concat([historical, new], ignore_index=True, sort=False)


def quantization_source_frame(output: Path, evaluation: str) -> pd.DataFrame:
    generated = pd.read_csv(
        output / "generation/analysis/quantization_detector_corpus.csv", low_memory=False
    )
    if len(generated) != 3840 or set(generated["quantization"]) != {"Q4_K_M", "Q8_0"}:
        raise SystemExit("quantization detector corpus is not the complete paired evaluation")
    if evaluation == "q4_to_q8":
        keep = (
            generated["quantization"].eq("Q4_K_M")
            & generated["payload_split"].isin(["train", "validation"])
        ) | (
            generated["quantization"].eq("Q8_0")
            & generated["payload_split"].eq("test")
        )
    elif evaluation == "q8_to_q4":
        keep = (
            generated["quantization"].eq("Q8_0")
            & generated["payload_split"].isin(["train", "validation"])
        ) | (
            generated["quantization"].eq("Q4_K_M")
            & generated["payload_split"].eq("test")
        )
    elif evaluation == "pooled_quantizations":
        keep = pd.Series(True, index=generated.index)
    else:
        raise SystemExit(f"unknown quantization evaluation: {evaluation}")
    generated = generated.loc[keep].copy()
    frame = pd.DataFrame(
        {
            "row_id": "quantization_generated__" + generated["plan_id"].astype(str),
            "pair_id": generated["reference_q4_pair_id"].astype(str),
            "payload_group_id": generated["payload_name"].astype(str),
            "text": generated["full_text"].astype(str),
            "label": generated["label"].astype(int),
            "partition": generated["payload_split"].astype(str),
            "model_id": generated["model_id"].astype(str),
            "codec_id": generated["codec_id"].astype(str),
            "payload_class": generated["payload_class"].astype(str),
            "prompt_template_id": generated["prompt_template_id"].astype(str),
            "representation_name": generated["representation_name"].astype(str),
            "generation_study": "quantization",
            "evaluation_id": evaluation,
            "study_source": "model_backed_quantization",
            "gate_level": "not_applicable",
            "quantization": generated["quantization"].astype(str),
            "plan_id": generated["plan_id"].astype(str),
            "pairing_unit_id": generated["pairing_unit_id"].astype(str),
        }
    )
    for partition, cell in frame.groupby("partition"):
        if set(cell["label"].astype(int)) != {0, 1}:
            raise SystemExit(f"quantization {partition} lacks one detector label")
    return frame


def preparation_paths(output: Path, evaluation: str) -> Mapping[str, Path]:
    stem = f"model_backed__{evaluation}"
    return {
        "corpus": output / f"generation/analysis/{stem}__deduplicated_corpus.csv",
        "manifest": output / f"deduplication/{stem}__row_manifest.csv",
        "removed": output / f"deduplication/{stem}__removed_rows.csv",
        "exact": output / f"deduplication/{stem}__exact_groups.csv",
        "near": output / f"deduplication/{stem}__near_pairs.csv",
        "clusters": output / f"deduplication/{stem}__cluster_manifest.csv",
        "audit": output / f"deduplication/{stem}__leakage_audit.json",
    }


def prepare_evaluation(
    output: Path,
    study: str,
    evaluation: str,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    paths = preparation_paths(output, evaluation)
    if all(path.is_file() for path in paths.values()) and not overwrite:
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise SystemExit(f"stored locked-dedup audit failed: {evaluation}")
        return pd.read_csv(paths["corpus"], low_memory=False)
    if any(path.exists() for path in paths.values()) and not overwrite:
        raise SystemExit(f"incomplete prepared output exists for {evaluation}; use --overwrite")
    source = entropy_source_frame(output) if study == "entropy" else quantization_source_frame(output, evaluation)
    result = locked_partition_deduplicate(source, threshold=0.95)
    for partition in ("train", "validation", "test"):
        cell = result.frame.loc[result.frame["partition"].eq(partition)]
        if cell.empty or set(cell["label"].astype(int)) != {0, 1}:
            raise SystemExit(f"deduplication left {evaluation} {partition} unusable")
    paths["corpus"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(paths["corpus"], result.frame)
    atomic_csv(paths["manifest"], result.frame.drop(columns=["text", "normalized_text"], errors="ignore"))
    atomic_csv(paths["removed"], result.removed_rows.drop(columns=["text", "normalized_text"], errors="ignore"))
    atomic_csv(paths["exact"], result.exact_groups)
    atomic_csv(paths["near"], result.near_pairs)
    atomic_csv(paths["clusters"], result.cluster_manifest)
    audit = {
        **result.audit,
        "generation_study": study,
        "evaluation_id": evaluation,
        "source_file_sha256": file_sha256(
            output / (
                "generation/analysis/entropy_detector_corpus.csv"
                if study == "entropy"
                else "generation/analysis/quantization_detector_corpus.csv"
            )
        ),
        "payload_grouping": "payload_name",
        "pair_grouping": "entropy pairing_unit_id or historical reference_q4_pair_id",
        "deduplication_before_detector_feature_extraction": True,
    }
    atomic_json(paths["audit"], audit)
    return result.frame


def generated_log_probabilities(record: Mapping[str, object]) -> Sequence[float]:
    record_type = str(record["record_type"])
    population = str(record.get("population", "unknown"))
    if record_type == "entropy_rankcloak_trial":
        return record["generation"]["embedding_log_probabilities"]
    if record_type == "entropy_ordinary_control":
        return record["generation"]["token_log_probabilities"]
    if record_type == "quantization_q4_model_backed_replay":
        return record["distribution_trace"]["observed_log_probabilities"]
    if record_type == "quantization_q8_generation":
        return record["q8_own_path_distribution_trace"]["observed_log_probabilities"]
    raise SystemExit(f"unsupported generated feature record: {record_type} {population}")


def feature_frame(output: Path, frame: pd.DataFrame, study: str) -> pd.DataFrame:
    historical = pd.read_csv(
        output / "provenance/generation_surprisal_features.csv", low_memory=False
    ).set_index("row_id", verify_integrity=True)
    records = load_json_records(
        output / f"generation/raw/{'entropy' if study == 'entropy' else 'quantization'}"
    )
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        row_id = str(row["row_id"])
        plan_id = row.get("plan_id")
        if str(row.get("study_source")) == "historical_deduplicated_fit":
            if row_id not in historical.index:
                raise SystemExit(f"historical feature row is missing: {row_id}")
            source = historical.loc[row_id].to_dict()
            rows.append({"row_id": row_id, **source})
            continue
        if plan_id is None or (isinstance(plan_id, float) and math.isnan(plan_id)):
            raise SystemExit(f"generated detector row lacks plan_id: {row_id}")
        record = records.get(str(plan_id))
        if record is None:
            raise SystemExit(f"generated trace is missing: {plan_id}")
        rows.append(
            {
                "row_id": row_id,
                "label": int(row["label"]),
                "source_record_kind": str(record["record_type"]),
                **surprisal_features_from_log_probabilities(
                    generated_log_probabilities(record)
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(frame) or result["row_id"].duplicated().any():
        raise SystemExit("model-aware feature extraction is not one-to-one")
    feature_columns = sorted(
        set(result.columns) - {"row_id", "label", "source_record_kind"}
    )
    if result[feature_columns].isna().any().any():
        raise SystemExit("model-aware feature extraction produced missing values")
    return result


def select_features(features: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    indexed = features.set_index("row_id", verify_integrity=True)
    selected = indexed.loc[rows["row_id"].astype(str)].reset_index()
    if selected["row_id"].astype(str).tolist() != rows["row_id"].astype(str).tolist():
        raise SystemExit("model-aware feature order differs from detector rows")
    return selected


def prediction_frame(
    frame: pd.DataFrame,
    scores: Sequence[float],
    *,
    detector: str,
    study: str,
    evaluation: str,
    role: str,
) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise SystemExit("detector score count differs from requested rows")
    columns = [
        "row_id", "label", "dedup_cluster_id", "payload_group_id", "pair_id",
        "model_id", "codec_id", "payload_class", "prompt_template_id",
        "representation_name", "gate_level", "quantization", "study_source",
    ]
    result = frame[columns].copy()
    result["score"] = np.asarray(scores, dtype=float)
    result["detector"] = detector
    result["generation_study"] = study
    result["evaluation_id"] = evaluation
    result["evaluation_role"] = role
    return result


class ProgressRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_phase: str | None = None

    def __call__(self, event: Mapping[str, object]) -> None:
        phase = str(event.get("phase", "unknown"))
        batch = int(event.get("batch", 0) or 0)
        if phase == self.last_phase and batch != 1 and (not batch or batch % 100):
            return
        self.last_phase = phase
        payload = {"timestamp": utc_now(), **dict(event)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print("[generation-detector] " + json.dumps(payload, sort_keys=True), flush=True)


def evaluate_slices(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    study: str,
    evaluation: str,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[Mapping[str, object], pd.DataFrame]:
    main = dict(
        evaluate_validation_frozen_detector(
            validation["label"], validation["score"],
            test["label"], test["score"], test["dedup_cluster_id"],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
    )
    dimensions = ["model_id", "codec_id", "payload_class", "representation_name"]
    if study == "entropy":
        dimensions.insert(0, "gate_level")
    elif evaluation == "pooled_quantizations":
        dimensions.insert(0, "quantization")
    rows: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    for dimension in dimensions:
        for level, cell in test.groupby(dimension, dropna=False, sort=True):
            labels = set(cell["label"].astype(int))
            if labels != {0, 1}:
                rows.append(
                    {
                        "dimension": dimension, "level": str(level),
                        "row_count": len(cell), "available": False,
                        "unavailable_reason": "subgroup lacks one detector label",
                    }
                )
                continue
            metrics = dict(
                evaluate_validation_frozen_detector(
                    validation["label"], validation["score"],
                    cell["label"], cell["score"], cell["dedup_cluster_id"],
                    bootstrap_resamples=bootstrap_resamples,
                    seed=stable_seed(seed, dimension, level),
                )
            )
            metrics.update({"dimension": dimension, "level": str(level)})
            detail.append(metrics)
            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "row_count": int(len(cell)),
                    "positive_count": int(cell["label"].eq(1).sum()),
                    "negative_count": int(cell["label"].eq(0).sum()),
                    "group_count": int(cell["dedup_cluster_id"].nunique()),
                    "available": True,
                    "roc_auc": metrics["roc_auc"],
                    "roc_auc_ci_low_95": metrics["roc_auc_ci_low_95"],
                    "roc_auc_ci_high_95": metrics["roc_auc_ci_high_95"],
                    "partial_auc_fpr_0_01": metrics["partial_auc_fpr_0_01"],
                    "partial_auc_fpr_0_01_ci_low_95": metrics["partial_auc_fpr_0_01_ci_low_95"],
                    "partial_auc_fpr_0_01_ci_high_95": metrics["partial_auc_fpr_0_01_ci_high_95"],
                    "threshold_at_fpr_0_01": metrics["threshold_at_fpr_0_01"],
                    "tpr_at_fpr_0_01": metrics["tpr_at_fpr_0_01"],
                    "fpr_at_threshold_0_01": metrics["fpr_at_threshold_0_01"],
                    "false_positives_at_fpr_0_01": metrics["false_positives_at_fpr_0_01"],
                    "fpr_0_001_available": metrics["threshold_selection"]["fpr_0_001"]["available"],
                    "low_fpr_warning": metrics["threshold_selection"]["fpr_0_001"].get("reason"),
                }
            )
    main["subgroup_metrics"] = detail
    return main, pd.DataFrame(rows)


def fit_paths(output: Path, detector: str, evaluation: str) -> Mapping[str, Path]:
    stem = f"{detector}__model_backed__{evaluation}"
    return {
        "predictions": output / f"detector_predictions/{stem}.csv",
        "metrics": output / f"metrics/{stem}.json",
        "subgroups": output / f"source_tables/{stem}__subgroups.csv",
        "metadata": output / f"provenance/{stem}__fit.json",
        "log": output / f"logs/{stem}.jsonl",
    }


def run_fit(
    output: Path,
    frame: pd.DataFrame,
    *,
    study: str,
    evaluation: str,
    detector: str,
    detector_config: Mapping[str, object],
    analysis_config: Mapping[str, object],
    overwrite: bool,
) -> None:
    paths = fit_paths(output, detector, evaluation)
    targets = [paths["predictions"], paths["metrics"], paths["subgroups"], paths["metadata"]]
    if all(path.is_file() for path in targets) and not overwrite:
        print(json.dumps({"status": "already_complete", "detector": detector, "evaluation": evaluation}))
        return
    if any(path.exists() for path in targets) and not overwrite:
        raise SystemExit(f"partial detector fit exists for {detector} {evaluation}; use --overwrite-fits")
    if overwrite:
        for path in targets:
            if path.exists():
                path.unlink()
    partitions = {
        name: frame.loc[frame["partition"].eq(name)].sort_values("row_id").reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    boundaries = [set(partitions[name]["dedup_cluster_id"].astype(str)) for name in partitions]
    if boundaries[0] & boundaries[1] or boundaries[0] & boundaries[2] or boundaries[1] & boundaries[2]:
        raise SystemExit("dedup clusters cross detector fit boundaries")
    if study == "entropy":
        test_payloads = set(partitions["test"]["payload_group_id"].astype(str))
        if test_payloads & set(partitions["train"]["payload_group_id"].astype(str)) or test_payloads & set(partitions["validation"]["payload_group_id"].astype(str)):
            raise SystemExit("entropy test payload leaked into fitting data")
    if evaluation == "q4_to_q8" and not (
        set(partitions["train"]["quantization"]) == {"Q4_K_M"}
        and set(partitions["validation"]["quantization"]) == {"Q4_K_M"}
        and set(partitions["test"]["quantization"]) == {"Q8_0"}
    ):
        raise SystemExit("Q4-to-Q8 holdout is not genuine")
    if evaluation == "q8_to_q4" and not (
        set(partitions["train"]["quantization"]) == {"Q8_0"}
        and set(partitions["validation"]["quantization"]) == {"Q8_0"}
        and set(partitions["test"]["quantization"]) == {"Q4_K_M"}
    ):
        raise SystemExit("Q8-to-Q4 holdout is not genuine")

    seed = stable_seed(int(analysis_config["seed"]), "model_backed", detector, evaluation)
    started_at = utc_now()
    started = time.perf_counter()
    if detector == "surprisal":
        features = feature_frame(output, frame, study)
        fit = fit_surprisal_detector(
            select_features(features, partitions["train"]),
            select_features(features, partitions["validation"]),
            select_features(features, partitions["test"]),
            c_grid=detector_config["C_grid"],
            seed=seed,
        )
        validation_scores = np.asarray(fit.pop("validation_scores"), dtype=float)
        test_scores = np.asarray(fit.pop("test_scores"), dtype=float)
        fit_metadata = dict(fit)
        implementation_kind = "saved_generation_trace_logistic"
    else:
        combined = pd.concat([partitions["validation"], partitions["test"]], ignore_index=True, sort=False)
        detector_output = run_configured_detector(
            partitions["train"], combined, detector_config, seed,
            smoke=False, allow_model_downloads=False,
            progress_callback=ProgressRecorder(paths["log"]),
        )
        validation_count = len(partitions["validation"])
        validation_scores = detector_output.scores[:validation_count]
        test_scores = detector_output.scores[validation_count:]
        if len(test_scores) != len(partitions["test"]):
            raise SystemExit("neural detector score slicing failed")
        fit_metadata = {
            "detector_name": detector_output.detector_name,
            "requested_kind": detector_output.requested_kind,
            "implementation_kind": detector_output.implementation_kind,
            "implementation_status": detector_output.implementation_status,
            "notes": detector_output.notes,
            "implementation_metadata": detector_output.metadata,
        }
        implementation_kind = detector_output.implementation_kind

    validation_predictions = prediction_frame(
        partitions["validation"], validation_scores, detector=detector,
        study=study, evaluation=evaluation, role="validation",
    )
    test_predictions = prediction_frame(
        partitions["test"], test_scores, detector=detector,
        study=study, evaluation=evaluation, role="test",
    )
    metrics, subgroup = evaluate_slices(
        validation_predictions, test_predictions,
        study=study, evaluation=evaluation,
        bootstrap_resamples=int(analysis_config["detector_metrics"]["bootstrap_resamples"]),
        seed=seed,
    )
    metrics.update(
        {
            "detector": detector,
            "generation_study": study,
            "evaluation_id": evaluation,
            "implementation_kind": implementation_kind,
            "training_rows": len(partitions["train"]),
            "training_model_ids": sorted(partitions["train"]["model_id"].astype(str).unique()),
            "validation_model_ids": sorted(partitions["validation"]["model_id"].astype(str).unique()),
            "test_model_ids": sorted(partitions["test"]["model_id"].astype(str).unique()),
            "training_quantizations": sorted(partitions["train"]["quantization"].astype(str).unique()),
            "validation_quantizations": sorted(partitions["validation"]["quantization"].astype(str).unique()),
            "test_quantizations": sorted(partitions["test"]["quantization"].astype(str).unique()),
            "seed": seed,
            "threshold_selected_on_test": False,
            "deduplication_before_feature_extraction": True,
            "dedup_audit_path": str(preparation_paths(output, evaluation)["audit"].relative_to(PROJECT_ROOT)),
        }
    )
    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    elapsed = max(0.0, time.perf_counter() - started)
    metadata = {
        "schema_version": "rankcloak-revision-v3-model-backed-detector-fit-v1",
        "detector": detector,
        "generation_study": study,
        "evaluation_id": evaluation,
        "seed": seed,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "train_row_count": len(partitions["train"]),
        "validation_row_count": len(partitions["validation"]),
        "test_row_count": len(partitions["test"]),
        "fit": fit_metadata,
        "test_tuning": False,
        "source_hashes": {
            str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)): file_sha256(DEFAULT_CONFIG),
            str(DEFAULT_ANALYSIS_CONFIG.relative_to(PROJECT_ROOT)): file_sha256(DEFAULT_ANALYSIS_CONFIG),
            str(preparation_paths(output, evaluation)["corpus"].relative_to(PROJECT_ROOT)): file_sha256(preparation_paths(output, evaluation)["corpus"]),
            str(Path(__file__).relative_to(PROJECT_ROOT)): file_sha256(Path(__file__)),
        },
    }
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(paths["predictions"], predictions)
    atomic_json(paths["metrics"], metrics)
    atomic_csv(paths["subgroups"], subgroup)
    atomic_json(paths["metadata"], metadata)
    print(
        json.dumps(
            {
                "status": "complete", "detector": detector, "evaluation": evaluation,
                "roc_auc": metrics["roc_auc"],
                "partial_auc_fpr_0_01": metrics["partial_auc_fpr_0_01"],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("entropy", "quantization", "all"), default="all")
    parser.add_argument("--evaluation", default="all")
    parser.add_argument("--detector", choices=("textcnn", "deberta", "surprisal", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-preparation", action="store_true")
    parser.add_argument("--overwrite-fits", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    studies = ["entropy", "quantization"] if args.study == "all" else [args.study]
    selected: list[tuple[str, str]] = []
    for study in studies:
        evaluations = STUDY_EVALUATIONS[study]
        if args.evaluation != "all":
            if args.evaluation not in evaluations:
                continue
            evaluations = (args.evaluation,)
        selected.extend((study, evaluation) for evaluation in evaluations)
    if not selected:
        raise SystemExit("the requested study/evaluation combination is empty")
    detector_document = json.loads(args.detector_config.read_text(encoding="utf-8"))
    analysis_document = json.loads(args.analysis_config.read_text(encoding="utf-8"))
    detectors = ["textcnn", "deberta", "surprisal"] if args.detector == "all" else [args.detector]
    for study, evaluation in selected:
        frame = prepare_evaluation(
            output, study, evaluation, overwrite=args.overwrite_preparation
        )
        print(
            json.dumps(
                {
                    "status": "prepared", "study": study, "evaluation": evaluation,
                    "rows": len(frame),
                    "partition_counts": frame["partition"].value_counts().sort_index().to_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.prepare_only:
            continue
        for detector in detectors:
            run_fit(
                output, frame, study=study, evaluation=evaluation, detector=detector,
                detector_config=detector_document["detectors"][detector],
                analysis_config=analysis_document, overwrite=args.overwrite_fits,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
