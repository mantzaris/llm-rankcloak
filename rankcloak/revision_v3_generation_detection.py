"""Leakage-safe locked partitions for V3 model-backed detector studies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .revision_v3_dedup import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_METHOD,
    NORMALIZATION_VERSION,
    find_near_duplicate_pairs,
    normalize_visible_text,
)


SCHEMA_VERSION = "rankcloak-revision-v3-locked-dedup-v1"
DEFAULT_PARTITION_PRIORITY = ("test", "validation", "train")


class LockedPartitionDedupError(ValueError):
    """Raised when fixed detector partitions cannot be made leakage-safe."""


@dataclass(frozen=True)
class LockedPartitionDedupResult:
    frame: pd.DataFrame
    removed_rows: pd.DataFrame
    exact_groups: pd.DataFrame
    near_pairs: pd.DataFrame
    cluster_manifest: pd.DataFrame
    audit: Mapping[str, object]


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _components(frame: pd.DataFrame, near_pairs: pd.DataFrame) -> list[list[str]]:
    groups = sorted(frame["payload_group_id"].astype(str).unique())
    union = _UnionFind(groups)
    for row in near_pairs.to_dict("records"):
        union.union(str(row["left_payload_group_id"]), str(row["right_payload_group_id"]))
    collected: dict[str, list[str]] = {}
    for group in groups:
        collected.setdefault(union.find(group), []).append(group)
    return sorted(sorted(members) for members in collected.values())


def _near_pairs(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    if len(frame) < 2:
        return pd.DataFrame(), 0
    try:
        return find_near_duplicate_pairs(frame, threshold=threshold)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return pd.DataFrame(), 0


def locked_partition_deduplicate(
    source: pd.DataFrame,
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    partition_priority: Sequence[str] = DEFAULT_PARTITION_PRIORITY,
) -> LockedPartitionDedupResult:
    """Deduplicate fixed train/validation/test rows without reassigning a row.

    Complete matched pairs are removed when one member is an exact duplicate.
    For a near-duplicate component spanning locked partitions, the component's
    highest-priority partition is retained and whole payload groups from other
    partitions are removed. This preserves the test population when auditing a
    prospectively frozen external test set while ensuring no textual component
    crosses a fitting boundary.
    """

    required = {"row_id", "pair_id", "payload_group_id", "text", "label", "partition"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise LockedPartitionDedupError(f"locked dedup source lacks columns: {missing}")
    frame = source.copy().reset_index(drop=True)
    if frame["row_id"].astype(str).duplicated().any():
        raise LockedPartitionDedupError("row_id must be unique before deduplication")
    if not set(frame["label"].astype(int)).issubset({0, 1}):
        raise LockedPartitionDedupError("labels must be binary")
    priority = {str(name): index for index, name in enumerate(partition_priority)}
    unknown = sorted(set(frame["partition"].astype(str)) - set(priority))
    if unknown:
        raise LockedPartitionDedupError(f"partition priority omits: {unknown}")
    for field in ("pair_id", "payload_group_id"):
        crossing = frame.groupby(field)["partition"].nunique().gt(1)
        if crossing.any():
            raise LockedPartitionDedupError(f"{field} crosses locked partitions before deduplication")

    frame["normalized_text"] = frame["text"].map(normalize_visible_text)
    if frame["normalized_text"].eq("").any():
        raise LockedPartitionDedupError("normalization produced empty text")
    frame["normalized_text_sha256"] = frame["normalized_text"].map(_hash)
    removed_parts: list[pd.DataFrame] = []
    exact_rows: list[dict[str, object]] = []
    duplicate_hashes = sorted(
        frame.loc[frame["normalized_text_sha256"].duplicated(False), "normalized_text_sha256"].unique()
    )
    dropped_pairs: set[str] = set()
    for group_number, digest in enumerate(duplicate_hashes, start=1):
        members = frame.loc[frame["normalized_text_sha256"].eq(digest)]
        pair_counts = members.groupby(members["pair_id"].astype(str)).size().to_dict()
        singleton_pairs = [pair for pair, count in pair_counts.items() if int(count) == 1]
        retained_pair = None
        if singleton_pairs:
            retained_pair = min(
                singleton_pairs,
                key=lambda pair: (
                    priority[str(members.loc[members["pair_id"].astype(str).eq(pair), "partition"].iloc[0])],
                    str(pair),
                ),
            )
        group_drops = sorted(pair for pair in pair_counts if pair != retained_pair)
        dropped_pairs.update(group_drops)
        exact_rows.append(
            {
                "exact_group_number": group_number,
                "normalized_text_sha256": digest,
                "participating_row_count": int(len(members)),
                "participating_pair_ids": ";".join(sorted(pair_counts)),
                "retained_pair_id": retained_pair,
                "removed_pair_ids": ";".join(group_drops),
            }
        )
    if dropped_pairs:
        removed = frame.loc[frame["pair_id"].astype(str).isin(dropped_pairs)].copy()
        removed["removal_reason"] = "exact_normalized_duplicate_complete_pair"
        removed_parts.append(removed)
        frame = frame.loc[~frame["pair_id"].astype(str).isin(dropped_pairs)].copy()
    if frame["normalized_text_sha256"].duplicated().any():
        raise LockedPartitionDedupError("exact duplicate remained after pair-level removal")

    preliminary_near, preliminary_features = _near_pairs(frame, float(threshold))
    groups_to_drop: set[str] = set()
    cross_component_rows: list[dict[str, object]] = []
    for members in _components(frame, preliminary_near):
        cell = frame.loc[frame["payload_group_id"].astype(str).isin(members)]
        partitions = sorted(cell["partition"].astype(str).unique(), key=lambda value: priority[value])
        if len(partitions) <= 1:
            continue
        retained_partition = partitions[0]
        dropped = sorted(
            cell.loc[~cell["partition"].astype(str).eq(retained_partition), "payload_group_id"].astype(str).unique()
        )
        groups_to_drop.update(dropped)
        cross_component_rows.append(
            {
                "payload_group_ids": ";".join(members),
                "partitions": ";".join(partitions),
                "retained_partition": retained_partition,
                "removed_payload_group_ids": ";".join(dropped),
            }
        )
    if groups_to_drop:
        removed = frame.loc[frame["payload_group_id"].astype(str).isin(groups_to_drop)].copy()
        removed["removal_reason"] = "near_duplicate_cross_partition_complete_payload_group"
        removed_parts.append(removed)
        frame = frame.loc[~frame["payload_group_id"].astype(str).isin(groups_to_drop)].copy()

    near_pairs, final_features = _near_pairs(frame.reset_index(drop=True), float(threshold))
    components = _components(frame, near_pairs)
    group_to_cluster: dict[str, str] = {}
    cluster_rows: list[dict[str, object]] = []
    for members in components:
        cluster_id = "locked-dedup-cluster-" + _hash("\n".join(members))[:24]
        for group in members:
            group_to_cluster[group] = cluster_id
        cell = frame.loc[frame["payload_group_id"].astype(str).isin(members)]
        partitions = sorted(cell["partition"].astype(str).unique())
        if len(partitions) != 1:
            raise LockedPartitionDedupError("a final near-duplicate component crosses partitions")
        cluster_rows.append(
            {
                "dedup_cluster_id": cluster_id,
                "partition": partitions[0],
                "payload_group_count": len(members),
                "row_count": len(cell),
                "positive_rows": int(cell["label"].astype(int).sum()),
                "negative_rows": int(len(cell) - cell["label"].astype(int).sum()),
                "payload_group_ids": ";".join(members),
            }
        )
    frame["dedup_cluster_id"] = frame["payload_group_id"].astype(str).map(group_to_cluster)
    frame = frame.sort_values("row_id").reset_index(drop=True)
    removed_rows = (
        pd.concat(removed_parts, ignore_index=True, sort=False).sort_values("row_id").reset_index(drop=True)
        if removed_parts
        else pd.DataFrame(columns=[*source.columns, "removal_reason"])
    )

    checks: dict[str, int] = {}
    for field in ("row_id", "pair_id", "payload_group_id", "normalized_text_sha256", "dedup_cluster_id"):
        checks[f"{field}_cross_partition"] = int(frame.groupby(field)["partition"].nunique().gt(1).sum())
    partition_by_row = frame.set_index("row_id")["partition"].astype(str).to_dict()
    checks["near_pair_cross_partition"] = int(
        sum(
            partition_by_row[str(row["left_row_id"])] != partition_by_row[str(row["right_row_id"])]
            for row in near_pairs.to_dict("records")
        )
    )
    violations = {key: value for key, value in checks.items() if value}
    if violations:
        raise LockedPartitionDedupError(f"locked partition audit failed: {violations}")
    unavailable = []
    factor_columns = [
        column for column in ("gate_level", "quantization", "model_id", "representation_name", "payload_class")
        if column in source.columns
    ]
    for column in factor_columns:
        lost = sorted(set(source[column].astype(str)) - set(frame[column].astype(str)))
        unavailable.extend(f"{column}={value}" for value in lost)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "normalization": NORMALIZATION_VERSION,
        "near_duplicate_method": NEAR_DUPLICATE_METHOD,
        "near_duplicate_threshold": float(threshold),
        "partition_priority": list(partition_priority),
        "original_rows": int(len(source)),
        "retained_rows": int(len(frame)),
        "removed_rows": int(len(removed_rows)),
        "removed_pairs": int(removed_rows["pair_id"].nunique()) if len(removed_rows) else 0,
        "removed_payload_groups": int(removed_rows["payload_group_id"].nunique()) if len(removed_rows) else 0,
        "exact_duplicate_groups": len(exact_rows),
        "preliminary_near_duplicate_pairs": int(len(preliminary_near)),
        "final_near_duplicate_pairs": int(len(near_pairs)),
        "cross_partition_near_components_resolved": len(cross_component_rows),
        "preliminary_tfidf_feature_count": int(preliminary_features),
        "final_tfidf_feature_count": int(final_features),
        "retained_counts_by_partition_and_label": [
            {"partition": str(partition), "label": int(label), "row_count": int(len(cell))}
            for (partition, label), cell in frame.groupby(["partition", "label"], sort=True)
        ],
        "conditions_unavailable_after_deduplication": unavailable,
        "checks": checks,
        "cross_partition_component_actions": cross_component_rows,
    }
    return LockedPartitionDedupResult(
        frame=frame,
        removed_rows=removed_rows,
        exact_groups=pd.DataFrame(exact_rows),
        near_pairs=near_pairs,
        cluster_manifest=pd.DataFrame(cluster_rows).sort_values("dedup_cluster_id").reset_index(drop=True),
        audit=audit,
    )


__all__ = [
    "DEFAULT_PARTITION_PRIORITY",
    "LockedPartitionDedupError",
    "LockedPartitionDedupResult",
    "SCHEMA_VERSION",
    "locked_partition_deduplicate",
]
