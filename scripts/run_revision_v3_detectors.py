#!/usr/bin/env python3
"""Fit revision-V3 detectors on immutable deduplicated partitions."""

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
    read_jsonl,
)
from rankcloak.revision_v3_dedup import leave_one_model_partitions  # noqa: E402
from rankcloak.revision_v3_metrics import (  # noqa: E402
    empirical_partial_auc,
    evaluate_validation_frozen_detector,
    frozen_threshold_counts,
    roc_auc,
    wilson_interval,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DEFAULT_CORPUS = (
    PROJECT_ROOT
    / "results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/revision_v3/detectors.json"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "configs/revision_v3/analysis.json"


def stable_seed(seed: int, *parts: object) -> int:
    material = "\x1f".join([str(int(seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def load_prepared_frame(output: Path) -> pd.DataFrame:
    source = pd.read_json(DEFAULT_CORPUS, lines=True)
    manifest_path = output / "deduplication/deduplicated_row_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit("Run scripts/prepare_revision_v3.py first")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    extras = manifest[
        ["row_id", "normalized_text_sha256", "dedup_cluster_id", "partition"]
    ].copy()
    frame = source.merge(extras, on="row_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise SystemExit("Prepared row manifest does not match the source corpus")
    for column in ("label", "model_id", "codec_id", "payload_class", "prompt_template_id"):
        expected = manifest.set_index("row_id")[column].astype(str)
        observed = frame.set_index("row_id")[column].astype(str)
        if not observed.equals(expected.reindex(observed.index)):
            raise SystemExit("Prepared source identity differs for {}".format(column))
    return frame.sort_values("row_id").reset_index(drop=True)


def load_human_frame(candidate_path: Path, output: Path) -> pd.DataFrame:
    selection_path = output / "provenance/human_control_selection_manifest.csv"
    selection = pd.read_csv(selection_path, low_memory=False)
    candidates = {
        str(row["candidate_id"]): row
        for row in read_jsonl(candidate_path)
        if row.get("eligible_for_manual_review") is True
    }
    rows = []
    for selected in selection.to_dict("records"):
        candidate = candidates.get(str(selected["candidate_id"]))
        if candidate is None:
            raise SystemExit("Selected human candidate is missing from the verified input")
        text = str(candidate["message_text"])
        observed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if observed_hash != str(selected["message_text_sha256"]):
            raise SystemExit("Selected human text hash differs")
        rows.append(
            {
                **selected,
                "text": text,
                "label": 0,
                "payload_group_id": str(selected["dedup_cluster_id"]),
                "pair_id": str(selected["candidate_id"]),
                "model_id": "human_databricks_dolly_15k",
                "codec_id": "not_applicable_human_control",
                "payload_class": "not_applicable_human_control",
                "partition": "human_test_only",
            }
        )
    return pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)


def evaluation_partitions(frame: pd.DataFrame, evaluation: str) -> Mapping[str, pd.DataFrame]:
    if evaluation == "matched":
        return {
            partition: frame.loc[frame["partition"].eq(partition)].copy()
            for partition in ("train", "validation", "test")
        }
    prefix = "leave_one_model__"
    if not evaluation.startswith(prefix):
        raise SystemExit("Unknown evaluation: {}".format(evaluation))
    model_id = evaluation[len(prefix) :]
    return leave_one_model_partitions(frame, model_id)


def prediction_rows(
    frame: pd.DataFrame,
    scores: Sequence[float],
    detector: str,
    evaluation: str,
    role: str,
    population: str,
) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise SystemExit("Detector score count differs from evaluation rows")
    result = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str).tolist(),
            "label": frame["label"].astype(int).tolist(),
            "score": np.asarray(scores, dtype=np.float64),
            "detector": detector,
            "evaluation_id": evaluation,
            "evaluation_role": role,
            "control_population": population,
            "dedup_cluster_id": frame["dedup_cluster_id"].astype(str).tolist(),
            "payload_group_id": frame["payload_group_id"].astype(str).tolist(),
            "model_id": frame["model_id"].astype(str).tolist(),
            "codec_id": frame["codec_id"].astype(str).tolist(),
            "payload_class": frame["payload_class"].astype(str).tolist(),
            "prompt_template_id": frame["prompt_template_id"].astype(str).tolist(),
        }
    )
    return result


def subgroup_metrics(
    test_predictions: pd.DataFrame, main_metrics: Mapping[str, object]
) -> pd.DataFrame:
    rows = []
    for dimension in ("codec_id", "payload_class"):
        for level, cell in test_predictions.groupby(dimension, sort=True):
            labels = cell["label"].to_numpy(dtype=int)
            scores = cell["score"].to_numpy(dtype=float)
            positive_count = int(np.count_nonzero(labels == 1))
            negative_count = int(np.count_nonzero(labels == 0))
            row: dict[str, Any] = {
                "dimension": dimension,
                "level": str(level),
                "row_count": int(len(cell)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "roc_auc": (
                    roc_auc(labels, scores)
                    if positive_count > 0 and negative_count > 0
                    else None
                ),
                "partial_auc_fpr_0_01": (
                    empirical_partial_auc(labels, scores, 0.01)
                    if positive_count > 0 and negative_count >= 100
                    else None
                ),
                "partial_auc_unavailable_reason": (
                    None
                    if positive_count > 0 and negative_count >= 100
                    else "fewer than 100 negative observations or one class absent"
                ),
            }
            for suffix, required in (("0_01", 100), ("0_001", 1000)):
                threshold = main_metrics.get("threshold_at_fpr_{}".format(suffix))
                if threshold is None or negative_count < required or positive_count == 0:
                    row["tpr_at_main_threshold_fpr_{}".format(suffix)] = None
                    row["fpr_at_main_threshold_fpr_{}".format(suffix)] = None
                    row["low_fpr_{}_unavailable_reason".format(suffix)] = (
                        "main threshold unavailable or subgroup negative count below {}".format(
                            required
                        )
                    )
                    continue
                counts = frozen_threshold_counts(labels, scores, float(threshold))
                tpr_low, tpr_high = wilson_interval(
                    int(counts["true_positives"]), positive_count
                )
                fpr_low, fpr_high = wilson_interval(
                    int(counts["false_positives"]), negative_count
                )
                row["tpr_at_main_threshold_fpr_{}".format(suffix)] = counts["tpr"]
                row["tpr_at_main_threshold_fpr_{}_ci_low_95".format(suffix)] = tpr_low
                row["tpr_at_main_threshold_fpr_{}_ci_high_95".format(suffix)] = tpr_high
                row["fpr_at_main_threshold_fpr_{}".format(suffix)] = counts["fpr"]
                row["fpr_at_main_threshold_fpr_{}_ci_low_95".format(suffix)] = fpr_low
                row["fpr_at_main_threshold_fpr_{}_ci_high_95".format(suffix)] = fpr_high
                row["low_fpr_{}_unavailable_reason".format(suffix)] = None
            rows.append(row)
    return pd.DataFrame(rows)


class ProgressRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_phase = None

    def __call__(self, event: Mapping[str, object]) -> None:
        phase = str(event.get("phase", "unknown"))
        batch = int(event.get("batch", 0) or 0)
        should_emit = phase != self.last_phase or batch == 1 or (batch and batch % 100 == 0)
        if not should_emit:
            return
        self.last_phase = phase
        payload = {"timestamp": utc_now(), **dict(event)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print("[detector] {}".format(json.dumps(payload, sort_keys=True)), flush=True)


def feature_partition(feature_frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    indexed = feature_frame.set_index("row_id", verify_integrity=True)
    selected = indexed.loc[rows["row_id"].astype(str)].reset_index()
    if selected["row_id"].astype(str).tolist() != rows["row_id"].astype(str).tolist():
        raise SystemExit("Generation feature order differs from detector rows")
    return selected


def run_one(
    detector_key: str,
    evaluation: str,
    frame: pd.DataFrame,
    human: pd.DataFrame,
    detector_config: Mapping[str, object],
    analysis_config: Mapping[str, object],
    output: Path,
    overwrite: bool,
) -> None:
    prediction_path = output / "detector_predictions/{}__{}.csv".format(
        detector_key, evaluation
    )
    metric_path = output / "metrics/{}__{}.json".format(detector_key, evaluation)
    subgroup_path = output / "source_tables/{}__{}__subgroups.csv".format(
        detector_key, evaluation
    )
    metadata_path = output / "provenance/{}__{}__fit.json".format(
        detector_key, evaluation
    )
    log_path = output / "logs/{}__{}.jsonl".format(detector_key, evaluation)
    targets = [prediction_path, metric_path, subgroup_path, metadata_path]
    if any(path.exists() for path in targets) and not overwrite:
        raise SystemExit("Refusing to overwrite an existing fit: {} {}".format(detector_key, evaluation))
    partitions = evaluation_partitions(frame, evaluation)
    for name in partitions:
        partitions[name] = partitions[name].sort_values("row_id").reset_index(drop=True)
    seed = stable_seed(int(analysis_config["seed"]), detector_key, evaluation)
    started = utc_now()
    clock = time.perf_counter()
    human_predictions = None
    if detector_key == "surprisal":
        all_features = pd.read_csv(
            output / "provenance/generation_surprisal_features.csv", low_memory=False
        )
        train_features = feature_partition(all_features, partitions["train"])
        validation_features = feature_partition(all_features, partitions["validation"])
        test_features = feature_partition(all_features, partitions["test"])
        fit = fit_surprisal_detector(
            train_features,
            validation_features,
            test_features,
            c_grid=detector_config["C_grid"],
            seed=seed,
        )
        validation_scores = np.asarray(fit.pop("validation_scores"), dtype=float)
        test_scores = np.asarray(fit.pop("test_scores"), dtype=float)
        fit_metadata = fit
        implementation_kind = "saved_generation_trace_logistic"
    else:
        evaluation_frames = [partitions["validation"], partitions["test"]]
        roles = [
            ("validation", len(partitions["validation"])),
            ("test", len(partitions["test"])),
        ]
        if evaluation == "matched" and not human.empty:
            evaluation_frames.append(human)
            roles.append(("human_test", len(human)))
        combined = pd.concat(evaluation_frames, ignore_index=True, sort=False)
        progress = ProgressRecorder(log_path)
        detector_output = run_configured_detector(
            partitions["train"],
            combined,
            detector_config,
            seed,
            smoke=False,
            allow_model_downloads=False,
            progress_callback=progress,
        )
        cursor = 0
        score_by_role = {}
        for role, length in roles:
            score_by_role[role] = detector_output.scores[cursor : cursor + length]
            cursor += length
        if cursor != len(detector_output.scores):
            raise SystemExit("Neural detector score slicing failed")
        validation_scores = np.asarray(score_by_role["validation"], dtype=float)
        test_scores = np.asarray(score_by_role["test"], dtype=float)
        if "human_test" in score_by_role:
            human_predictions = prediction_rows(
                human,
                score_by_role["human_test"],
                detector_key,
                evaluation,
                "human_test",
                "human_authored_dolly",
            )
        fit_metadata = {
            "detector_name": detector_output.detector_name,
            "requested_kind": detector_output.requested_kind,
            "implementation_kind": detector_output.implementation_kind,
            "implementation_status": detector_output.implementation_status,
            "notes": detector_output.notes,
            "implementation_metadata": detector_output.metadata,
        }
        implementation_kind = detector_output.implementation_kind

    validation_predictions = prediction_rows(
        partitions["validation"],
        validation_scores,
        detector_key,
        evaluation,
        "validation",
        "matched_clean_llm",
    )
    test_predictions = prediction_rows(
        partitions["test"],
        test_scores,
        detector_key,
        evaluation,
        "test",
        "matched_clean_llm",
    )
    predictions = [validation_predictions, test_predictions]
    if human_predictions is not None:
        predictions.append(human_predictions)
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics = dict(
        evaluate_validation_frozen_detector(
            validation_predictions["label"],
            validation_predictions["score"],
            test_predictions["label"],
            test_predictions["score"],
            test_predictions["dedup_cluster_id"],
            bootstrap_resamples=int(
                analysis_config["detector_metrics"]["bootstrap_resamples"]
            ),
            seed=seed,
        )
    )
    metrics.update(
        {
            "detector": detector_key,
            "evaluation_id": evaluation,
            "implementation_kind": implementation_kind,
            "training_rows": int(len(partitions["train"])),
            "training_model_ids": sorted(partitions["train"]["model_id"].astype(str).unique()),
            "validation_model_ids": sorted(partitions["validation"]["model_id"].astype(str).unique()),
            "test_model_ids": sorted(partitions["test"]["model_id"].astype(str).unique()),
            "seed": int(seed),
            "threshold_selected_on_test": False,
        }
    )
    human_metrics = None
    if human_predictions is not None:
        positive_test = test_predictions.loc[test_predictions["label"].eq(1)]
        rankcloak_vs_human = pd.concat(
            [positive_test, human_predictions], ignore_index=True, sort=False
        )
        human_metrics = dict(
            evaluate_validation_frozen_detector(
                validation_predictions["label"],
                validation_predictions["score"],
                rankcloak_vs_human["label"],
                rankcloak_vs_human["score"],
                rankcloak_vs_human["dedup_cluster_id"],
                bootstrap_resamples=int(
                    analysis_config["detector_metrics"]["bootstrap_resamples"]
                ),
                seed=stable_seed(seed, "human"),
            )
        )
        human_metrics.update(
            {
                "comparison": "rankcloak_vs_human_authored_dolly",
                "warning": (
                    "Performance may include generic machine-versus-human signals; "
                    "thresholds were selected on matched LLM validation controls only."
                ),
                "human_test_labels_used_for_threshold_selection": False,
            }
        )
        metrics["human_secondary_control"] = human_metrics
    elapsed = max(0.0, time.perf_counter() - clock)
    metadata = {
        "schema_version": "rankcloak-revision-v3-detector-fit-v1",
        "detector": detector_key,
        "evaluation_id": evaluation,
        "seed": int(seed),
        "started_at": started,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "train_row_count": int(len(partitions["train"])),
        "validation_row_count": int(len(partitions["validation"])),
        "test_row_count": int(len(partitions["test"])),
        "human_row_count": int(len(human_predictions)) if human_predictions is not None else 0,
        "fit": fit_metadata,
        "test_tuning": False,
    }
    atomic_csv(prediction_path, all_predictions)
    atomic_json(metric_path, metrics)
    atomic_csv(subgroup_path, subgroup_metrics(test_predictions, metrics))
    atomic_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "status": "complete",
                "detector": detector_key,
                "evaluation": evaluation,
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
    parser.add_argument("--detector", choices=("textcnn", "deberta", "surprisal"), required=True)
    parser.add_argument("--evaluation", default="all")
    parser.add_argument("--human-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    frame = load_prepared_frame(output)
    human = load_human_frame(args.human_candidates, output) if args.detector != "surprisal" else pd.DataFrame()
    detector_document = json.loads(args.detector_config.read_text(encoding="utf-8"))
    analysis_document = json.loads(args.analysis_config.read_text(encoding="utf-8"))
    detector_config = detector_document["detectors"][args.detector]
    model_ids = sorted(frame["model_id"].astype(str).unique())
    evaluations = ["matched"] + ["leave_one_model__{}".format(model) for model in model_ids]
    if args.evaluation != "all":
        if args.evaluation not in evaluations:
            raise SystemExit(
                "evaluation must be one of: {}".format(", ".join(evaluations))
            )
        evaluations = [args.evaluation]
    for evaluation in evaluations:
        run_one(
            args.detector,
            evaluation,
            frame,
            human,
            detector_config,
            analysis_document,
            output,
            args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
