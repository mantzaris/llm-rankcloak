"""Strict deduplication and group-safe three-way splits for revision V3.

The V1 detector analysis audited duplicate leakage after constructing splits.
This module changes the order of operations: visible text is normalized and
deduplicated first, near-duplicate links are then converted to connected
components, and only those components are assigned to train, validation, and
test partitions.  Payload groups and matched pairs are never divided.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .revision_detection import normalize_detector_frame


SCHEMA_VERSION = "rankcloak-revision-v3-dedup-v1"
NORMALIZATION_VERSION = "unicode_nfkc_casefold_whitespace_collapse_v1"
NEAR_DUPLICATE_METHOD = "char_wb_tfidf_cosine_ngrams_3_5_min_df_2_v1"
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.95
DEFAULT_SEED = 20260831
PARTITIONS = ("train", "validation", "test")


class RevisionV3DedupError(ValueError):
    """Raised when a purported strict partition has unresolved leakage."""


@dataclass(frozen=True)
class StrictDedupResult:
    """Authoritative deduplicated frame and its audit tables."""

    frame: pd.DataFrame
    removed_rows: pd.DataFrame
    exact_groups: pd.DataFrame
    near_pairs: pd.DataFrame
    cluster_manifest: pd.DataFrame
    partition_manifest: pd.DataFrame
    leakage_audit: Mapping[str, object]
    summary: Mapping[str, object]


def _sha256(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize_visible_text(value: object) -> str:
    """Apply the frozen exact-deduplication normalization.

    NFKC is intentionally followed by case folding and Unicode whitespace
    collapse.  The original text remains untouched in result artifacts.
    """

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(str(left))
        root_right = self.find(str(right))
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self.parent[second] = first


def _exact_pair_deduplicate(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove complete matched pairs implicated by normalized duplicates."""

    work = frame.copy()
    if "pair_id" not in work.columns:
        work["pair_id"] = work["row_id"].astype(str)
    work["normalized_text"] = work["text"].map(normalize_visible_text)
    if work["normalized_text"].eq("").any():
        raise RevisionV3DedupError("Text normalization produced an empty value")
    work["normalized_text_sha256"] = work["normalized_text"].map(_sha256)

    duplicate_hashes = sorted(
        work.loc[
            work["normalized_text_sha256"].duplicated(False),
            "normalized_text_sha256",
        ].unique()
    )
    dropped_pairs: set[str] = set()
    group_rows = []
    for group_number, text_hash in enumerate(duplicate_hashes, start=1):
        members = work.loc[work["normalized_text_sha256"].eq(text_hash)].copy()
        pairs = sorted(members["pair_id"].astype(str).unique())
        counts = members.groupby(members["pair_id"].astype(str)).size().to_dict()
        safe_retain = [pair for pair in pairs if int(counts[pair]) == 1]
        retained_pair = safe_retain[0] if safe_retain else None
        group_drops = [pair for pair in pairs if pair != retained_pair]
        if retained_pair is None:
            group_drops = pairs
        dropped_pairs.update(group_drops)
        group_rows.append(
            {
                "exact_group_number": int(group_number),
                "normalized_text_sha256": str(text_hash),
                "participating_rows": int(len(members)),
                "participating_pair_count": int(len(pairs)),
                "participating_row_ids": ";".join(
                    sorted(members["row_id"].astype(str))
                ),
                "participating_pair_ids": ";".join(pairs),
                "retained_pair_id": retained_pair,
                "removed_pair_ids": ";".join(sorted(group_drops)),
            }
        )

    removed = work.loc[work["pair_id"].astype(str).isin(dropped_pairs)].copy()
    retained = work.loc[~work["pair_id"].astype(str).isin(dropped_pairs)].copy()
    retained = retained.reset_index(drop=True)
    removed = removed.reset_index(drop=True)
    if retained["normalized_text_sha256"].duplicated(False).any():
        raise RevisionV3DedupError(
            "Exact pair-level deduplication left a normalized duplicate"
        )
    exact_groups = pd.DataFrame(
        group_rows,
        columns=[
            "exact_group_number",
            "normalized_text_sha256",
            "participating_rows",
            "participating_pair_count",
            "participating_row_ids",
            "participating_pair_ids",
            "retained_pair_id",
            "removed_pair_ids",
        ],
    )
    return retained, removed, exact_groups


def find_near_duplicate_pairs(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> Tuple[pd.DataFrame, int]:
    """Find all normalized char-ngram TF-IDF cosine neighbors at threshold."""

    if not 0.0 < float(threshold) <= 1.0:
        raise RevisionV3DedupError("near-duplicate threshold must be in (0, 1]")
    normalized = (
        frame["normalized_text"]
        if "normalized_text" in frame.columns
        else frame["text"].map(normalize_visible_text)
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(normalized.astype(str).tolist())
    radius = max(0.0, 1.0 - float(threshold)) + 1e-12
    neighbors = NearestNeighbors(
        metric="cosine", algorithm="brute", n_jobs=1, radius=radius
    ).fit(matrix)
    distances, indices = neighbors.radius_neighbors(
        matrix, return_distance=True, sort_results=True
    )
    rows = []
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, raw_right in zip(row_distances, row_indices):
            right = int(raw_right)
            if right <= left:
                continue
            similarity = 1.0 - float(distance)
            if similarity + 1e-12 < float(threshold):
                continue
            a = frame.iloc[left]
            b = frame.iloc[right]
            rows.append(
                {
                    "near_pair_number": int(len(rows) + 1),
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
                    "cosine_similarity": float(similarity),
                    "threshold": float(threshold),
                }
            )
    columns = [
        "near_pair_number",
        "left_position",
        "right_position",
        "left_row_id",
        "right_row_id",
        "left_payload_group_id",
        "right_payload_group_id",
        "same_payload_group",
        "left_label",
        "right_label",
        "cosine_similarity",
        "threshold",
    ]
    return pd.DataFrame(rows, columns=columns), int(matrix.shape[1])


def _attach_duplicate_clusters(
    frame: pd.DataFrame, near_pairs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = sorted(frame["payload_group_id"].astype(str).unique())
    union = _UnionFind(groups)
    for pair in near_pairs.to_dict("records"):
        union.union(
            str(pair["left_payload_group_id"]),
            str(pair["right_payload_group_id"]),
        )
    components: Dict[str, list[str]] = {}
    for group in groups:
        components.setdefault(union.find(group), []).append(group)
    group_to_cluster: Dict[str, str] = {}
    rows = []
    for members in sorted((sorted(value) for value in components.values())):
        identity = _sha256("\n".join(members))[:24]
        cluster_id = "dedup-cluster-{}".format(identity)
        for group in members:
            group_to_cluster[group] = cluster_id
        cluster_frame = frame.loc[frame["payload_group_id"].astype(str).isin(members)]
        rows.append(
            {
                "dedup_cluster_id": cluster_id,
                "payload_group_count": int(len(members)),
                "row_count": int(len(cluster_frame)),
                "positive_rows": int(cluster_frame["label"].eq(1).sum()),
                "negative_rows": int(cluster_frame["label"].eq(0).sum()),
                "payload_group_ids": ";".join(members),
            }
        )
    result = frame.copy()
    result["dedup_cluster_id"] = result["payload_group_id"].astype(str).map(
        group_to_cluster
    )
    return result, pd.DataFrame(rows).sort_values("dedup_cluster_id").reset_index(drop=True)


def _balance_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "label",
        "model_id",
        "codec_id",
        "payload_class",
        "prompt_template_id",
    ]
    return [column for column in candidates if column in frame.columns]


def assign_three_way_partitions(
    frame: pd.DataFrame,
    fractions: Mapping[str, float] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily balance immutable dedup clusters across three partitions."""

    fractions = dict(
        fractions or {"train": 0.60, "validation": 0.20, "test": 0.20}
    )
    if set(fractions) != set(PARTITIONS):
        raise RevisionV3DedupError("fractions must define train, validation, and test")
    if any(float(value) <= 0.0 for value in fractions.values()) or not math.isclose(
        sum(map(float, fractions.values())), 1.0, abs_tol=1e-12
    ):
        raise RevisionV3DedupError("partition fractions must be positive and sum to one")
    if "dedup_cluster_id" not in frame.columns:
        raise RevisionV3DedupError("dedup_cluster_id is required before splitting")

    factors = _balance_columns(frame)
    feature_names = ["rows"]
    for column in factors:
        feature_names.extend(
            "{}={}".format(column, value)
            for value in sorted(frame[column].astype(str).unique())
        )
    feature_index = {name: index for index, name in enumerate(feature_names)}

    def vector(subset: pd.DataFrame) -> np.ndarray:
        values = np.zeros(len(feature_names), dtype=np.float64)
        values[feature_index["rows"]] = float(len(subset))
        for column in factors:
            counts = subset[column].astype(str).value_counts()
            for level, count in counts.items():
                values[feature_index["{}={}".format(column, level)]] = float(count)
        return values

    total = vector(frame)
    targets = {
        partition: total * float(fractions[partition]) for partition in PARTITIONS
    }
    current = {
        partition: np.zeros_like(total, dtype=np.float64) for partition in PARTITIONS
    }
    cluster_frames = {
        str(cluster): subset.copy()
        for cluster, subset in frame.groupby("dedup_cluster_id", sort=True)
    }
    cluster_vectors = {
        cluster: vector(subset) for cluster, subset in cluster_frames.items()
    }
    ordering = sorted(
        cluster_frames,
        key=lambda cluster: (
            -len(cluster_frames[cluster]),
            _sha256("{}\x1f{}".format(int(seed), cluster)),
        ),
    )
    assignments: Dict[str, str] = {}
    scale = np.maximum(total, 1.0)
    for cluster in ordering:
        candidate_scores = []
        for partition in PARTITIONS:
            proposed = {name: current[name].copy() for name in PARTITIONS}
            proposed[partition] += cluster_vectors[cluster]
            loss = sum(
                float(np.sum(((proposed[name] - targets[name]) ** 2) / scale))
                for name in PARTITIONS
            )
            tie = _sha256("{}\x1f{}\x1f{}".format(int(seed), cluster, partition))
            candidate_scores.append((loss, tie, partition))
        _, _, selected = min(candidate_scores)
        assignments[cluster] = selected
        current[selected] += cluster_vectors[cluster]

    result = frame.copy()
    result["partition"] = result["dedup_cluster_id"].map(assignments)
    manifest_rows = []
    for cluster in sorted(assignments):
        subset = cluster_frames[cluster]
        manifest_rows.append(
            {
                "dedup_cluster_id": cluster,
                "partition": assignments[cluster],
                "row_count": int(len(subset)),
                "positive_rows": int(subset["label"].eq(1).sum()),
                "negative_rows": int(subset["label"].eq(0).sum()),
                "payload_group_count": int(subset["payload_group_id"].nunique()),
                "row_ids_sha256": _sha256(
                    "\n".join(sorted(subset["row_id"].astype(str)))
                ),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    for partition in PARTITIONS:
        subset = result.loc[result["partition"].eq(partition)]
        if subset.empty or subset["label"].nunique() != 2:
            raise RevisionV3DedupError(
                "Partition {} is empty or lacks one label".format(partition)
            )
    return result, manifest


def validate_strict_partitions(
    frame: pd.DataFrame, near_pairs: pd.DataFrame
) -> Dict[str, object]:
    """Return a machine-readable leakage audit and fail on any boundary leak."""

    if "partition" not in frame.columns:
        raise RevisionV3DedupError("partition column is missing")
    observed = set(frame["partition"].astype(str))
    if observed != set(PARTITIONS):
        raise RevisionV3DedupError("all three partitions must be present")
    checks: Dict[str, int] = {}
    for field, name in [
        ("row_id", "row_id_cross_partition"),
        ("pair_id", "pair_id_cross_partition"),
        ("payload_group_id", "payload_group_cross_partition"),
        ("dedup_cluster_id", "dedup_cluster_cross_partition"),
        ("normalized_text_sha256", "normalized_text_cross_partition"),
    ]:
        if field not in frame.columns:
            continue
        counts = frame.groupby(field, dropna=False)["partition"].nunique()
        checks[name] = int(counts.gt(1).sum())
    row_partition = frame.set_index("row_id")["partition"].astype(str).to_dict()
    crossed_near = 0
    for pair in near_pairs.to_dict("records"):
        if row_partition[str(pair["left_row_id"])] != row_partition[
            str(pair["right_row_id"])
        ]:
            crossed_near += 1
    checks["near_duplicate_pair_cross_partition"] = int(crossed_near)
    violations = {name: count for name, count in checks.items() if int(count) != 0}
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not violations else "fail",
        "normalization": NORMALIZATION_VERSION,
        "near_duplicate_method": NEAR_DUPLICATE_METHOD,
        "near_duplicate_threshold": float(
            near_pairs["threshold"].iloc[0]
            if len(near_pairs)
            else DEFAULT_NEAR_DUPLICATE_THRESHOLD
        ),
        "checks": checks,
        "violations": violations,
    }
    if violations:
        raise RevisionV3DedupError(
            "Strict partition leakage audit failed: {}".format(violations)
        )
    return result


def _factor_count_records(frame: pd.DataFrame, stage: str) -> list[dict]:
    rows = []
    dimensions = [
        "label",
        "model_id",
        "codec_id",
        "payload_class",
        "prompt_template_id",
    ]
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        for value, count in frame[dimension].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "stage": stage,
                    "dimension": dimension,
                    "level": str(value),
                    "row_count": int(count),
                }
            )
    return rows


def build_strict_deduplicated_corpus(
    source: pd.DataFrame,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    seed: int = DEFAULT_SEED,
    fractions: Optional[Mapping[str, float]] = None,
) -> StrictDedupResult:
    """Execute the complete pre-split strict deduplication pipeline."""

    normalized = normalize_detector_frame(source)
    retained, removed, exact_groups = _exact_pair_deduplicate(normalized)
    near_pairs, feature_count = find_near_duplicate_pairs(retained, threshold=threshold)
    clustered, cluster_manifest = _attach_duplicate_clusters(retained, near_pairs)
    partitioned, partition_manifest = assign_three_way_partitions(
        clustered, fractions=fractions, seed=seed
    )
    leakage = validate_strict_partitions(partitioned, near_pairs)
    factor_counts = _factor_count_records(normalized, "original")
    factor_counts.extend(_factor_count_records(partitioned, "deduplicated"))
    for partition in PARTITIONS:
        factor_counts.extend(
            _factor_count_records(
                partitioned.loc[partitioned["partition"].eq(partition)], partition
            )
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "near_duplicate_method": NEAR_DUPLICATE_METHOD,
        "near_duplicate_threshold": float(threshold),
        "seed": int(seed),
        "original_rows": int(len(normalized)),
        "original_pairs": int(normalized["pair_id"].nunique())
        if "pair_id" in normalized.columns
        else None,
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_participating_rows": int(
            exact_groups["participating_rows"].sum() if len(exact_groups) else 0
        ),
        "removed_observations": int(len(removed)),
        "removed_pairs": int(removed["pair_id"].nunique())
        if len(removed) and "pair_id" in removed.columns
        else 0,
        "deduplicated_rows": int(len(partitioned)),
        "near_duplicate_pairs": int(len(near_pairs)),
        "near_duplicate_cross_payload_pairs": int(
            (~near_pairs["same_payload_group"]).sum() if len(near_pairs) else 0
        ),
        "near_duplicate_clusters": int(cluster_manifest["payload_group_count"].gt(1).sum()),
        "dedup_components": int(len(cluster_manifest)),
        "tfidf_feature_count": int(feature_count),
        "partition_rows": {
            partition: int(partitioned["partition"].eq(partition).sum())
            for partition in PARTITIONS
        },
        "partition_positive_rows": {
            partition: int(
                partitioned.loc[partitioned["partition"].eq(partition), "label"].sum()
            )
            for partition in PARTITIONS
        },
        "factor_counts": factor_counts,
        "conditions_unavailable_after_deduplication": [],
        "leakage_audit_status": leakage["status"],
    }
    return StrictDedupResult(
        frame=partitioned,
        removed_rows=removed,
        exact_groups=exact_groups,
        near_pairs=near_pairs,
        cluster_manifest=cluster_manifest,
        partition_manifest=partition_manifest,
        leakage_audit=leakage,
        summary=summary,
    )


def leave_one_model_partitions(
    frame: pd.DataFrame, held_out_model: str
) -> Dict[str, pd.DataFrame]:
    """Create a genuine model-family holdout with disjoint dedup clusters.

    Target-family rows are used only from test clusters.  Training and
    validation use other families from their respective clusters.  Remaining
    rows are deliberately unused so neither model identity nor payload-linked
    duplicate components cross the evaluation boundary.
    """

    target = str(held_out_model)
    if target not in set(frame["model_id"].astype(str)):
        raise RevisionV3DedupError("Unknown held-out model: {}".format(target))
    result = {
        "train": frame.loc[
            frame["partition"].eq("train")
            & ~frame["model_id"].astype(str).eq(target)
        ].copy(),
        "validation": frame.loc[
            frame["partition"].eq("validation")
            & ~frame["model_id"].astype(str).eq(target)
        ].copy(),
        "test": frame.loc[
            frame["partition"].eq("test")
            & frame["model_id"].astype(str).eq(target)
        ].copy(),
    }
    for name, subset in result.items():
        if subset.empty or subset["label"].nunique() != 2:
            raise RevisionV3DedupError(
                "Leave-one-model {} partition {} cannot support evaluation".format(
                    target, name
                )
            )
    if result["train"]["model_id"].astype(str).eq(target).any() or result[
        "validation"
    ]["model_id"].astype(str).eq(target).any():
        raise RevisionV3DedupError("Held-out model leaked into fitting data")
    if not result["test"]["model_id"].astype(str).eq(target).all():
        raise RevisionV3DedupError("Non-target model leaked into holdout test")
    boundaries = [
        set(result[name]["dedup_cluster_id"].astype(str)) for name in PARTITIONS
    ]
    if boundaries[0] & boundaries[1] or boundaries[0] & boundaries[2] or boundaries[1] & boundaries[2]:
        raise RevisionV3DedupError("Dedup clusters cross leave-one-model boundaries")
    return result


__all__ = [
    "DEFAULT_NEAR_DUPLICATE_THRESHOLD",
    "DEFAULT_SEED",
    "NEAR_DUPLICATE_METHOD",
    "NORMALIZATION_VERSION",
    "PARTITIONS",
    "RevisionV3DedupError",
    "SCHEMA_VERSION",
    "StrictDedupResult",
    "assign_three_way_partitions",
    "build_strict_deduplicated_corpus",
    "find_near_duplicate_pairs",
    "leave_one_model_partitions",
    "normalize_visible_text",
    "validate_strict_partitions",
]
