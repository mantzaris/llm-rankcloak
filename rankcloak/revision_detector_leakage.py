"""Deterministic detector-corpus split and near-duplicate leakage audit.

This diagnostic is deliberately separate from detector fitting.  It validates the
frozen execution-plan partitions, repeats the exact row/payload/text assertions,
and applies a declared character-ngram similarity rule without changing any split.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .revision_detection import (
    DetectorColumns,
    DetectorSplit,
    build_evaluation_splits,
    grouped_bootstrap_detector_metrics,
    normalize_detector_frame,
    read_detector_frame,
)
from .revision_detector_analysis import (
    _bootstrap_supplementary,
    _stable_seed as _analysis_stable_seed,
)


SCHEMA_VERSION = "rankcloak-detector-leakage-audit-v1"
SENSITIVITY_SCHEMA_VERSION = "rankcloak-detector-leakage-sensitivity-v1"
SIMILARITY_THRESHOLD = 0.95


class DetectorLeakageAuditError(ValueError):
    """Raised when corpus, split, plan, or audit identities disagree."""


@dataclass(frozen=True)
class DetectorLeakageAuditArtifacts:
    output_dir: str
    manifest_path: str
    near_duplicate_pair_count: int
    affected_split_count: int
    affected_test_payload_group_count_across_splits: int


@dataclass(frozen=True)
class DetectorLeakageSensitivityArtifacts:
    output_dir: str
    manifest_path: str
    fit_count: int
    metric_row_count: int
    affected_fit_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DetectorLeakageAuditError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DetectorLeakageAuditError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DetectorLeakageAuditError(f"{label} must contain a JSON object")
    return value


def _file_identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def _sha256_int_sequence(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_text_sequence(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_near_duplicate_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _near_duplicate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    normalized_text = frame["text"].map(_normalize_near_duplicate_text)
    if normalized_text.eq("").any():
        raise DetectorLeakageAuditError("Near-duplicate normalization produced empty text")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(normalized_text.tolist())
    neighbors = NearestNeighbors(
        metric="cosine", algorithm="brute", n_jobs=1, radius=0.05
    ).fit(matrix)
    distances, indices = neighbors.radius_neighbors(
        matrix, return_distance=True, sort_results=True
    )
    normalized_sha = normalized_text.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    rows: list[dict[str, Any]] = []
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, raw_right in zip(row_distances, row_indices):
            right = int(raw_right)
            if right <= left:
                continue
            similarity = 1.0 - float(distance)
            if similarity + 1e-12 < SIMILARITY_THRESHOLD:
                continue
            a = frame.iloc[left]
            b = frame.iloc[right]
            rows.append(
                {
                    "pair_number": len(rows) + 1,
                    "left_position": int(left),
                    "right_position": int(right),
                    "left_row_id": str(a["row_id"]),
                    "right_row_id": str(b["row_id"]),
                    "left_payload_group_id": str(a["payload_group_id"]),
                    "right_payload_group_id": str(b["payload_group_id"]),
                    "same_payload_group": bool(
                        str(a["payload_group_id"]) == str(b["payload_group_id"])
                    ),
                    "left_label": int(a["label"]),
                    "right_label": int(b["label"]),
                    "left_text_sha256": str(a["text_sha256"]),
                    "right_text_sha256": str(b["text_sha256"]),
                    "left_normalized_text_sha256": str(normalized_sha.iloc[left]),
                    "right_normalized_text_sha256": str(normalized_sha.iloc[right]),
                    "normalized_text_exact": bool(
                        normalized_sha.iloc[left] == normalized_sha.iloc[right]
                    ),
                    "cosine_similarity": similarity,
                    "left_prompt_template_id": str(a["prompt_template_id"]),
                    "right_prompt_template_id": str(b["prompt_template_id"]),
                    "left_model_id": str(a["model_id"]),
                    "right_model_id": str(b["model_id"]),
                    "left_codec_id": str(a["codec_id"]),
                    "right_codec_id": str(b["codec_id"]),
                }
            )
    columns = [
        "pair_number",
        "left_position",
        "right_position",
        "left_row_id",
        "right_row_id",
        "left_payload_group_id",
        "right_payload_group_id",
        "same_payload_group",
        "left_label",
        "right_label",
        "left_text_sha256",
        "right_text_sha256",
        "left_normalized_text_sha256",
        "right_normalized_text_sha256",
        "normalized_text_exact",
        "cosine_similarity",
        "left_prompt_template_id",
        "right_prompt_template_id",
        "left_model_id",
        "right_model_id",
        "left_codec_id",
        "right_codec_id",
    ]
    return pd.DataFrame(rows, columns=columns), int(matrix.shape[1])


def _build_splits(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[DetectorSplit]:
    split_config = dict(config.get("splits", {}))
    regimes = split_config.get(
        "regimes", ["matched", "held_out_template", "leave_one_model", "leave_one_codec"]
    )
    splits, skipped = build_evaluation_splits(
        frame,
        regimes=list(map(str, regimes)),
        test_fraction=float(split_config.get("matched_test_fraction", 0.25)),
        seed=int(config.get("seed", -1)),
        check_text_hash=bool(split_config.get("assert_text_hash_disjoint", True)),
        minimum_train_rows=int(split_config.get("minimum_train_rows", 4)),
        minimum_test_rows=int(split_config.get("minimum_test_rows", 2)),
    )
    if skipped:
        raise DetectorLeakageAuditError(
            "Frozen detector audit unexpectedly skipped splits: "
            + "; ".join(item.split_id for item in skipped)
        )
    return splits


def _validate_execution_plan(
    plan: Mapping[str, Any], frame: pd.DataFrame, splits: Sequence[DetectorSplit]
) -> None:
    tasks = plan.get("tasks")
    if (
        plan.get("schema_version") != "rankcloak-revision-detector-execution-plan-v1"
        or not isinstance(tasks, list)
        or int(plan.get("split_count", -1)) != len(splits)
        or int(plan.get("total_fit_count", -1)) != len(tasks)
        or int(plan.get("detector_count", -1)) <= 0
    ):
        raise DetectorLeakageAuditError("Frozen detector execution plan structure differs")
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise DetectorLeakageAuditError("Execution-plan task is malformed")
        by_split.setdefault(str(raw.get("split_id", "")), []).append(raw)
    if set(by_split) != {split.split_id for split in splits}:
        raise DetectorLeakageAuditError("Execution-plan split identities differ")
    expected_per_split = int(plan["detector_count"])
    for split in splits:
        train = list(map(int, split.train_indices))
        test = list(map(int, split.test_indices))
        expected = {
            "train_row_count": len(train),
            "test_row_count": len(test),
            "train_indices_sha256": _sha256_int_sequence(train),
            "test_indices_sha256": _sha256_int_sequence(test),
            "train_row_ids_ordered_sha256": _sha256_text_sequence(
                frame.iloc[train]["row_id"].astype(str).tolist()
            ),
            "test_row_ids_ordered_sha256": _sha256_text_sequence(
                frame.iloc[test]["row_id"].astype(str).tolist()
            ),
        }
        if len(by_split[split.split_id]) != expected_per_split:
            raise DetectorLeakageAuditError(
                f"Execution-plan detector count differs for {split.split_id}"
            )
        for task in by_split[split.split_id]:
            if any(task.get(key) != value for key, value in expected.items()):
                raise DetectorLeakageAuditError(
                    f"Execution-plan partition identity differs for {split.split_id}"
                )


def _split_audit_tables(
    frame: pd.DataFrame,
    splits: Sequence[DetectorSplit],
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_records = pairs.to_dict("records")
    summaries: list[dict[str, Any]] = []
    affected_rows: list[dict[str, Any]] = []
    for split in splits:
        train_positions = set(map(int, split.train_indices))
        test_positions = set(map(int, split.test_indices))
        train = frame.iloc[sorted(train_positions)]
        test = frame.iloc[sorted(test_positions)]
        cross_pairs: list[tuple[dict[str, Any], int, int]] = []
        for pair in pair_records:
            left = int(pair["left_position"])
            right = int(pair["right_position"])
            if left in train_positions and right in test_positions:
                cross_pairs.append((pair, left, right))
            elif right in train_positions and left in test_positions:
                cross_pairs.append((pair, right, left))
        by_test_row: dict[int, list[tuple[dict[str, Any], int]]] = {}
        for pair, train_position, test_position in cross_pairs:
            by_test_row.setdefault(test_position, []).append((pair, train_position))
        for test_position, hits in sorted(by_test_row.items()):
            test_row = frame.iloc[test_position]
            affected_rows.append(
                {
                    "split_id": split.split_id,
                    "regime": split.regime,
                    "test_position": int(test_position),
                    "test_row_id": str(test_row["row_id"]),
                    "test_payload_group_id": str(test_row["payload_group_id"]),
                    "test_label": int(test_row["label"]),
                    "near_duplicate_train_pair_count": len(hits),
                    "maximum_cosine_similarity": max(
                        float(pair["cosine_similarity"]) for pair, _ in hits
                    ),
                    "train_row_ids": ";".join(
                        sorted(str(frame.iloc[position]["row_id"]) for _, position in hits)
                    ),
                }
            )
        source_trial_overlap = (
            len(set(train["source_trial_id"].astype(str)) & set(test["source_trial_id"].astype(str)))
            if "source_trial_id" in frame.columns
            else -1
        )
        pair_id_overlap = (
            len(set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str)))
            if "pair_id" in frame.columns
            else -1
        )
        affected_payloads = {
            str(frame.iloc[position]["payload_group_id"]) for position in by_test_row
        }
        summaries.append(
            {
                "split_id": split.split_id,
                "regime": split.regime,
                "held_out_column": split.held_out_column or "not_applicable",
                "held_out_value": split.held_out_value or "not_applicable",
                "train_rows": len(train),
                "test_rows": len(test),
                "row_position_overlap": len(train_positions & test_positions),
                "payload_group_overlap": len(
                    set(train["payload_group_id"]) & set(test["payload_group_id"])
                ),
                "row_id_overlap": len(set(train["row_id"]) & set(test["row_id"])),
                "raw_text_sha256_overlap": len(
                    set(train["text_sha256"]) & set(test["text_sha256"])
                ),
                "source_trial_id_overlap": source_trial_overlap,
                "pair_id_overlap": pair_id_overlap,
                "cross_partition_near_duplicate_pairs": len(cross_pairs),
                "affected_test_rows": len(by_test_row),
                "affected_test_payload_groups": len(affected_payloads),
                "maximum_cross_partition_similarity": (
                    max(float(pair["cosine_similarity"]) for pair, _, _ in cross_pairs)
                    if cross_pairs
                    else np.nan
                ),
                "exact_leakage_checks_passed": bool(
                    len(train_positions & test_positions) == 0
                    and not (set(train["payload_group_id"]) & set(test["payload_group_id"]))
                    and not (set(train["row_id"]) & set(test["row_id"]))
                    and not (set(train["text_sha256"]) & set(test["text_sha256"]))
                    and source_trial_overlap in {-1, 0}
                    and pair_id_overlap in {-1, 0}
                ),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(affected_rows, columns=[
        "split_id",
        "regime",
        "test_position",
        "test_row_id",
        "test_payload_group_id",
        "test_label",
        "near_duplicate_train_pair_count",
        "maximum_cosine_similarity",
        "train_row_ids",
    ])


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def _atomic_json(value: Mapping[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def build_detector_leakage_audit(
    *,
    detector_corpus: str | Path,
    detector_config: str | Path,
    execution_plan: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> DetectorLeakageAuditArtifacts:
    """Validate frozen partitions and publish exact/near-duplicate diagnostics."""

    corpus_path = Path(detector_corpus).resolve()
    config_path = Path(detector_config).resolve()
    plan_path = Path(execution_plan).resolve()
    config = _read_json(config_path, label="detector config")
    plan = _read_json(plan_path, label="detector execution plan")
    columns = DetectorColumns.from_mapping(config.get("columns", {}))
    try:
        frame = normalize_detector_frame(read_detector_frame(corpus_path), columns=columns)
        splits = _build_splits(frame, config)
    except Exception as exc:
        raise DetectorLeakageAuditError(f"Detector split reconstruction failed: {exc}") from exc
    _validate_execution_plan(plan, frame, splits)
    pairs, vocabulary_features = _near_duplicate_pairs(frame)
    split_summary, affected = _split_audit_tables(frame, splits, pairs)
    if not split_summary["exact_leakage_checks_passed"].all():
        raise DetectorLeakageAuditError("An exact detector leakage invariant failed")

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise DetectorLeakageAuditError(f"Unsafe output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "near_duplicate_pairs": target / "near_duplicate_pairs.csv",
        "split_summary": target / "split_leakage_summary.csv",
        "affected_test_rows": target / "affected_test_rows.csv",
        "manifest": target / "detector_leakage_audit_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise DetectorLeakageAuditError(f"Refusing to overwrite audit output: {path}")

    _atomic_csv(pairs, paths["near_duplicate_pairs"])
    _atomic_csv(split_summary, paths["split_summary"])
    _atomic_csv(affected, paths["affected_test_rows"])
    affected_split_count = int(
        (split_summary["cross_partition_near_duplicate_pairs"] > 0).sum()
    )
    affected_payloads_across_splits = int(
        split_summary["affected_test_payload_groups"].sum()
    )
    output_identities = {
        "near_duplicate_pairs": _file_identity(
            paths["near_duplicate_pairs"], row_count=len(pairs)
        ),
        "split_summary": _file_identity(
            paths["split_summary"], row_count=len(split_summary)
        ),
        "affected_test_rows": _file_identity(
            paths["affected_test_rows"], row_count=len(affected)
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "adverse_near_duplicate_overlap_detected"
            if affected_split_count
            else "passed_no_declared_near_duplicate_overlap"
        ),
        "analysis_role": "diagnostic_split_leakage_audit_frozen_splits_unchanged",
        "near_duplicate_definition": {
            "normalization": "Unicode_NFKC_casefold_whitespace_collapse",
            "vectorizer": "character_word_boundary_tfidf_ngrams_3_5_min_df_2_l2_float32",
            "cosine_similarity_threshold": SIMILARITY_THRESHOLD,
            "neighbor_search": "exhaustive_radius_brute_single_cpu_worker",
            "semantic_equivalence_claimed": False,
        },
        "inputs": {
            "detector_corpus": _file_identity(corpus_path, row_count=len(frame)),
            "detector_config": _file_identity(config_path),
            "execution_plan": _file_identity(plan_path),
        },
        "outputs": output_identities,
        "summary": {
            "corpus_rows": int(len(frame)),
            "payload_groups": int(frame["payload_group_id"].nunique()),
            "pair_ids": int(frame["pair_id"].nunique()) if "pair_id" in frame else None,
            "split_count": int(len(split_summary)),
            "execution_plan_fit_count": int(plan["total_fit_count"]),
            "exact_leakage_failed_split_count": int(
                (~split_summary["exact_leakage_checks_passed"]).sum()
            ),
            "tfidf_vocabulary_features": vocabulary_features,
            "near_duplicate_pair_count": int(len(pairs)),
            "cross_payload_near_duplicate_pair_count": int(
                (~pairs["same_payload_group"]).sum()
            ),
            "within_payload_near_duplicate_pair_count": int(
                pairs["same_payload_group"].sum()
            ),
            "normalized_exact_pair_count": int(pairs["normalized_text_exact"].sum()),
            "affected_split_count": affected_split_count,
            "cross_partition_pair_occurrences_across_splits": int(
                split_summary["cross_partition_near_duplicate_pairs"].sum()
            ),
            "affected_test_row_occurrences_across_splits": int(len(affected)),
            "affected_test_payload_group_count_across_splits": (
                affected_payloads_across_splits
            ),
        },
        "limitations": [
            "The 0.95 threshold is a declared lexical diagnostic, not a universal semantic-near-duplicate definition.",
            "The frozen confirmatory detector splits and completed or active fits were not changed by this audit.",
            "Detector outcomes from affected splits require an explicit leakage limitation and sensitivity analysis.",
        ],
        "generation_command": command,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(manifest, paths["manifest"])
    return DetectorLeakageAuditArtifacts(
        output_dir=str(target),
        manifest_path=str(paths["manifest"]),
        near_duplicate_pair_count=int(len(pairs)),
        affected_split_count=affected_split_count,
        affected_test_payload_group_count_across_splits=affected_payloads_across_splits,
    )


def _verify_signed(value: Mapping[str, Any], field: str, *, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_json_sha256(payload):
        raise DetectorLeakageAuditError(f"{label} self-hash differs")


def _declared_file(
    manifest_path: Path,
    outputs: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> Path:
    declaration = outputs.get(key)
    if not isinstance(declaration, Mapping):
        raise DetectorLeakageAuditError(f"{label} lacks declared output {key}")
    raw = declaration.get("path")
    if isinstance(raw, str) and raw:
        candidate = Path(raw)
        path = (
            candidate
            if candidate.is_absolute()
            else manifest_path.parent / candidate
        )
    else:
        output_dir = manifest_path.parent
        path = output_dir / key
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise DetectorLeakageAuditError(f"Missing or unsafe {label} output: {path}")
    if (
        file_sha256(path) != declaration.get("sha256")
        or int(path.stat().st_size) != int(declaration.get("size_bytes", -1))
    ):
        raise DetectorLeakageAuditError(f"{label} output identity differs: {path}")
    return path


def _detector_prediction_path(
    manifest_path: Path, manifest: Mapping[str, Any], expected_fits: int
) -> Path:
    if (
        manifest.get("schema_version") != "rankcloak-revision-detector-run-v2"
        or manifest.get("execution_mode") != "confirmatory"
        or manifest.get("confirmatory_complete") is not True
        or int(manifest.get("completed_fit_count", -1)) != expected_fits
        or int(manifest.get("total_fit_count", -1)) != expected_fits
        or int(manifest.get("failure_count", -1)) != 0
        or int(manifest.get("smoke_fallback_metric_rows", -1)) != 0
        or manifest.get("device") != "cuda:0"
    ):
        raise DetectorLeakageAuditError(
            "Detector run is not the complete frozen CUDA confirmatory matrix"
        )
    if "manifest_sha256" in manifest:
        _verify_signed(manifest, "manifest_sha256", label="detector run manifest")
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping):
        raise DetectorLeakageAuditError("Detector run lacks output_files")
    return _declared_file(
        manifest_path,
        outputs,
        "detector_predictions.csv",
        label="detector run",
    )


def _validate_sensitivity_config(value: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = value.get("bootstrap")
    if (
        value.get("schema_version")
        != "rankcloak-detector-near-duplicate-sensitivity-config-v1"
        or value.get("analysis_status")
        != "exploratory_post_partial_checkpoint_leakage_diagnostic"
        or value.get("partial_checkpoint_outcomes_seen_before_extension") is not True
        or value.get("frozen_training_design_unchanged") is not True
        or value.get("frozen_split_design_unchanged") is not True
        or value.get("exclusion_unit") != "test_payload_group_id"
        or value.get("exclusion_rule")
        != "exclude_entire_test_payload_group_if_any_test_row_has_declared_near_duplicate_in_training"
        or float(value.get("cosine_similarity_threshold", -1))
        != SIMILARITY_THRESHOLD
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("unit") != "payload_group_id"
        or int(bootstrap.get("resamples", 0)) <= 0
        or not 0.0 < float(bootstrap.get("confidence_level", 0.0)) < 1.0
    ):
        raise DetectorLeakageAuditError("Leakage sensitivity disclosure config differs")
    return dict(bootstrap)


def build_detector_leakage_sensitivity(
    *,
    detector_run_manifest: str | Path,
    leakage_audit_manifest: str | Path,
    sensitivity_config: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> DetectorLeakageSensitivityArtifacts:
    """Re-evaluate final predictions after excluding implicated test payload groups."""

    run_path = Path(detector_run_manifest).resolve()
    audit_path = Path(leakage_audit_manifest).resolve()
    config_path = Path(sensitivity_config).resolve()
    run = _read_json(run_path, label="detector run manifest")
    audit = _read_json(audit_path, label="detector leakage audit manifest")
    config = _read_json(config_path, label="detector leakage sensitivity config")
    bootstrap = _validate_sensitivity_config(config)
    expected_fits = int(config.get("expected_fit_count", -1))
    predictions_path = _detector_prediction_path(run_path, run, expected_fits)
    _verify_signed(audit, "manifest_sha256", label="detector leakage audit manifest")
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("status") != "adverse_near_duplicate_overlap_detected"
        or float(audit.get("near_duplicate_definition", {}).get(
            "cosine_similarity_threshold", -1
        )) != SIMILARITY_THRESHOLD
    ):
        raise DetectorLeakageAuditError("Leakage audit is not the declared adverse audit")
    audit_outputs = audit.get("outputs")
    if not isinstance(audit_outputs, Mapping):
        raise DetectorLeakageAuditError("Leakage audit lacks outputs")
    affected_path = _declared_file(
        audit_path,
        audit_outputs,
        "affected_test_rows",
        label="detector leakage audit",
    )
    predictions = pd.read_csv(predictions_path, low_memory=False)
    affected = pd.read_csv(affected_path, low_memory=False)
    required = {
        "split_id",
        "regime",
        "held_out_value",
        "detector_name",
        "requested_kind",
        "implementation_kind",
        "implementation_status",
        "row_id",
        "payload_group_id",
        "label",
        "score",
    }
    if not required.issubset(predictions.columns):
        raise DetectorLeakageAuditError("Detector predictions lack sensitivity columns")
    if not {"split_id", "test_payload_group_id"}.issubset(affected.columns):
        raise DetectorLeakageAuditError("Leakage audit lacks affected payload identities")
    if predictions.duplicated(["split_id", "detector_name", "row_id"]).any():
        raise DetectorLeakageAuditError("Detector prediction row identities repeat")
    if not predictions["implementation_status"].astype(str).eq("complete").all():
        raise DetectorLeakageAuditError("Detector predictions include incomplete fits")

    affected_groups = {
        str(split_id): set(cell["test_payload_group_id"].astype(str))
        for split_id, cell in affected.groupby("split_id", sort=False)
    }
    threshold = float(config["decision_threshold"])
    low_fprs = tuple(float(value) for value in config["low_false_positive_rates"])
    resamples = int(bootstrap["resamples"])
    seed = int(bootstrap["seed"])
    confidence = float(bootstrap["confidence_level"])
    if threshold != 0.5 or low_fprs != (0.01, 0.05):
        raise DetectorLeakageAuditError("Leakage sensitivity metric settings differ")

    metric_rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(
        [
            "split_id",
            "regime",
            "held_out_value",
            "detector_name",
            "requested_kind",
            "implementation_kind",
        ],
        dropna=False,
        sort=True,
    )
    if grouped.ngroups != expected_fits:
        raise DetectorLeakageAuditError("Leakage sensitivity fit matrix differs")
    core_names = (
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    )
    supplementary_names = (
        "precision",
        "brier_score",
        "tpr_at_fpr_0.01",
        "tpr_at_fpr_0.05",
    )
    for keys, cell in grouped:
        split_id, regime, held_out_value, detector_name, requested_kind, implementation_kind = keys
        split_id = str(split_id)
        excluded_groups = affected_groups.get(split_id, set())
        restricted = cell.loc[
            ~cell["payload_group_id"].astype(str).isin(excluded_groups)
        ].copy()
        observed_excluded_groups = set(cell["payload_group_id"].astype(str)) & excluded_groups
        if observed_excluded_groups != excluded_groups:
            raise DetectorLeakageAuditError(
                f"Affected payload groups are absent from predictions for {split_id}"
            )
        if len(restricted) == 0 or restricted["label"].nunique() != 2:
            raise DetectorLeakageAuditError(
                f"Leakage restriction removes a detector class for {split_id}"
            )
        for _group, rows in restricted.groupby("payload_group_id", sort=False):
            labels = pd.to_numeric(rows["label"], errors="coerce")
            positive = int(labels.sum()) if not labels.isna().any() else -1
            negative = int(len(labels) - positive) if positive >= 0 else -1
            if positive < 1 or positive != negative:
                raise DetectorLeakageAuditError(
                    f"Leakage restriction breaks balanced payload groups for {split_id}"
                )
        original_core = grouped_bootstrap_detector_metrics(
            cell["label"].astype(int),
            cell["score"].astype(float),
            cell["payload_group_id"].astype(str),
            n_resamples=0,
            seed=_analysis_stable_seed(seed, split_id, detector_name, "original"),
            threshold=threshold,
        )
        restricted_core = grouped_bootstrap_detector_metrics(
            restricted["label"].astype(int),
            restricted["score"].astype(float),
            restricted["payload_group_id"].astype(str),
            n_resamples=resamples,
            seed=_analysis_stable_seed(seed, split_id, detector_name, "restricted"),
            threshold=threshold,
        )
        original_supplementary, _ = _bootstrap_supplementary(
            cell,
            threshold=threshold,
            low_fprs=low_fprs,
            resamples=1,
            confidence_level=confidence,
            seed=_analysis_stable_seed(seed, split_id, detector_name, "original_supplementary"),
        )
        restricted_supplementary, restricted_intervals = _bootstrap_supplementary(
            restricted,
            threshold=threshold,
            low_fprs=low_fprs,
            resamples=resamples,
            confidence_level=confidence,
            seed=_analysis_stable_seed(seed, split_id, detector_name, "restricted_supplementary"),
        )
        common = {
            "split_id": split_id,
            "regime": str(regime),
            "held_out_value": (
                "not_applicable" if pd.isna(held_out_value) else str(held_out_value)
            ),
            "detector_name": str(detector_name),
            "requested_kind": str(requested_kind),
            "implementation_kind": str(implementation_kind),
            "original_test_rows": int(len(cell)),
            "restricted_test_rows": int(len(restricted)),
            "excluded_test_rows": int(len(cell) - len(restricted)),
            "original_payload_groups": int(cell["payload_group_id"].nunique()),
            "restricted_payload_groups": int(restricted["payload_group_id"].nunique()),
            "excluded_payload_groups": int(len(excluded_groups)),
            "affected_split": bool(excluded_groups),
            "analysis_unit": "payload_group_id",
            "confidence_level": confidence,
            "bootstrap_resamples_requested": resamples,
            "evidence_status": "exploratory_post_partial_checkpoint_leakage_sensitivity",
        }
        for metric in core_names:
            original_estimate = float(original_core[metric])
            restricted_estimate = float(restricted_core[metric])
            metric_rows.append(
                {
                    **common,
                    "metric": metric,
                    "original_estimate": original_estimate,
                    "restricted_estimate": restricted_estimate,
                    "restricted_minus_original": restricted_estimate - original_estimate,
                    "restricted_ci_low": float(restricted_core[f"{metric}_ci_low_95"]),
                    "restricted_ci_high": float(restricted_core[f"{metric}_ci_high_95"]),
                    "bootstrap_resamples_valid": int(
                        restricted_core[f"{metric}_bootstrap_valid"]
                    ),
                    "higher_is_better": True,
                }
            )
        for metric in supplementary_names:
            low, high, valid = restricted_intervals[metric]
            original_estimate = float(original_supplementary[metric])
            restricted_estimate = float(restricted_supplementary[metric])
            metric_rows.append(
                {
                    **common,
                    "metric": metric,
                    "original_estimate": original_estimate,
                    "restricted_estimate": restricted_estimate,
                    "restricted_minus_original": restricted_estimate - original_estimate,
                    "restricted_ci_low": float(low),
                    "restricted_ci_high": float(high),
                    "bootstrap_resamples_valid": int(valid),
                    "higher_is_better": metric != "brier_score",
                }
            )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["regime", "held_out_value", "detector_name", "metric"]
    ).reset_index(drop=True)
    numeric = metrics[
        [
            "original_estimate",
            "restricted_estimate",
            "restricted_minus_original",
            "restricted_ci_low",
            "restricted_ci_high",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DetectorLeakageAuditError("Leakage sensitivity produced nonfinite metrics")
    summary_rows: list[dict[str, Any]] = []
    for keys, cell in metrics.groupby(
        ["detector_name", "regime", "metric", "higher_is_better"], sort=True
    ):
        detector_name, regime, metric, higher_is_better = keys
        summary_rows.append(
            {
                "detector_name": detector_name,
                "regime": regime,
                "metric": metric,
                "higher_is_better": bool(higher_is_better),
                "fit_count": int(cell["split_id"].nunique()),
                "affected_fit_count": int(cell["affected_split"].sum()),
                "excluded_payload_groups_sum_across_splits": int(
                    cell["excluded_payload_groups"].sum()
                ),
                "original_estimate_median": float(cell["original_estimate"].median()),
                "restricted_estimate_median": float(
                    cell["restricted_estimate"].median()
                ),
                "restricted_minus_original_median": float(
                    cell["restricted_minus_original"].median()
                ),
                "maximum_absolute_change": float(
                    cell["restricted_minus_original"].abs().max()
                ),
                "cross_split_interval": "not_computed_heterogeneous_prespecified_splits",
            }
        )
    summary = pd.DataFrame(summary_rows)

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise DetectorLeakageAuditError(f"Unsafe sensitivity output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": target / "detector_leakage_sensitivity_metrics.csv",
        "summary": target / "detector_leakage_sensitivity_summary.csv",
        "manifest": target / "detector_leakage_sensitivity_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise DetectorLeakageAuditError(
                f"Refusing to overwrite leakage sensitivity output: {path}"
            )
    _atomic_csv(metrics, paths["metrics"])
    _atomic_csv(summary, paths["summary"])
    outputs = {
        "metrics": _file_identity(paths["metrics"], row_count=len(metrics)),
        "summary": _file_identity(paths["summary"], row_count=len(summary)),
    }
    affected_fit_count = int(
        metrics.loc[metrics["metric"].eq("roc_auc"), "affected_split"].sum()
    )
    result: dict[str, Any] = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "status": "completed_exploratory_leakage_sensitivity",
        "analysis_status": config["analysis_status"],
        "frozen_training_design_unchanged": True,
        "frozen_split_design_unchanged": True,
        "partial_checkpoint_outcomes_seen_before_extension": True,
        "exclusion_unit": config["exclusion_unit"],
        "exclusion_rule": config["exclusion_rule"],
        "inputs": {
            "detector_run_manifest": _file_identity(run_path),
            "detector_predictions": _file_identity(
                predictions_path, row_count=len(predictions)
            ),
            "leakage_audit_manifest": _file_identity(audit_path),
            "affected_test_rows": _file_identity(affected_path, row_count=len(affected)),
            "sensitivity_config": _file_identity(config_path),
        },
        "outputs": outputs,
        "summary": {
            "fit_count": int(grouped.ngroups),
            "affected_fit_count": affected_fit_count,
            "unaffected_fit_count": int(grouped.ngroups - affected_fit_count),
            "metric_row_count": int(len(metrics)),
            "summary_row_count": int(len(summary)),
            "affected_test_payload_group_occurrences": int(
                audit["summary"]["affected_test_payload_group_count_across_splits"]
            ),
            "bootstrap_resamples": resamples,
        },
        "limitations": [
            "This post-partial-checkpoint sensitivity is exploratory, not a replacement confirmatory estimand.",
            "Training was not repeated; the sensitivity removes implicated payload groups only from evaluation.",
            "The lexical threshold does not exhaust all possible semantic near duplicates.",
        ],
        "generation_command": command,
    }
    result["manifest_sha256"] = canonical_json_sha256(result)
    _atomic_json(result, paths["manifest"])
    return DetectorLeakageSensitivityArtifacts(
        output_dir=str(target),
        manifest_path=str(paths["manifest"]),
        fit_count=int(grouped.ngroups),
        metric_row_count=int(len(metrics)),
        affected_fit_count=affected_fit_count,
    )
