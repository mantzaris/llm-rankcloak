"""Leakage-resistant raw-text detectors for the RankCloak revision study.

This module is intentionally separate from :mod:`rankcloak.detection`, which
implements the feature-only exploratory detector used by the submitted pilot.
The revision pipeline operates on raw text and requires a payload grouping key
so that repeated payloads cannot cross an evaluation boundary.

Neural dependencies are imported lazily.  A dependency-light hashed n-gram
classifier is available only as an explicitly labelled smoke/offline fallback;
its results must not be reported as neural-detector results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import stat
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_SEED = 20260808
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_TRANSFORMER_MODEL_ID = "microsoft/deberta-v3-base"
CONFIRMATORY_TRANSFORMER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
CONFIRMATORY_TRANSFORMER_RELATIVE_PATH = "models/detectors/deberta_v3_base"
CONFIRMATORY_TRANSFORMER_DIRECTORY = (
    PROJECT_ROOT / CONFIRMATORY_TRANSFORMER_RELATIVE_PATH
)
CONFIRMATORY_TRANSFORMER_ARTIFACTS: Dict[str, Dict[str, object]] = {
    "config.json": {
        "sha256": "649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b",
        "size_bytes": 579,
    },
    "pytorch_model.bin": {
        "sha256": "691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33",
        "size_bytes": 371146213,
    },
    "spm.model": {
        "sha256": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
        "size_bytes": 2464616,
    },
    "tokenizer_config.json": {
        "sha256": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
        "size_bytes": 52,
    },
}
MODEL_STATE_HASH_ALGORITHM = "rankcloak-torch-state-v1"
MODEL_STATE_SCHEMA_HASH_ALGORITHM = "rankcloak-torch-state-schema-v1"
CANONICAL_COLUMNS = (
    "row_id",
    "text",
    "label",
    "payload_group_id",
    "prompt_template_id",
    "model_id",
    "codec_id",
)
REQUIRED_SOURCE_FIELDS = tuple(column for column in CANONICAL_COLUMNS if column != "row_id")
REGIME_COLUMNS = {
    "held_out_template": "prompt_template_id",
    "leave_one_model": "model_id",
    "leave_one_codec": "codec_id",
}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class RevisionDetectionError(ValueError):
    """Raised when detector input or evaluation design is unsafe."""


@dataclass(frozen=True)
class DetectorColumns:
    """Source-column mapping for a revision detector dataset."""

    row_id: str = "row_id"
    text: str = "text"
    label: str = "label"
    payload_group_id: str = "payload_group_id"
    prompt_template_id: str = "prompt_template_id"
    model_id: str = "model_id"
    codec_id: str = "codec_id"

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, object]] = None) -> "DetectorColumns":
        mapping = dict(values or {})
        unknown = sorted(set(mapping) - set(CANONICAL_COLUMNS))
        if unknown:
            raise RevisionDetectionError(
                "Unknown detector column mapping keys: {}".format(", ".join(unknown))
            )
        instance = cls(**{key: str(value) for key, value in mapping.items()})
        source_names = [instance.source_name(name) for name in CANONICAL_COLUMNS]
        if len(source_names) != len(set(source_names)):
            raise RevisionDetectionError(
                "Detector column mappings must use a different source column for every field."
            )
        return instance

    def source_name(self, canonical_name: str) -> str:
        return str(getattr(self, canonical_name))


@dataclass(frozen=True)
class DetectorSplit:
    """One immutable train/test split represented by positional indices."""

    split_id: str
    regime: str
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    held_out_column: Optional[str] = None
    held_out_value: Optional[str] = None
    purged_train_rows: int = 0
    partition_policy: str = "full_held_out_condition"
    excluded_held_out_rows: int = 0


@dataclass(frozen=True)
class SkippedSplit:
    """A requested split that could not support binary evaluation."""

    split_id: str
    regime: str
    reason: str
    held_out_column: Optional[str] = None
    held_out_value: Optional[str] = None


@dataclass
class DetectorOutput:
    """Scores and provenance returned by one detector on one split."""

    scores: np.ndarray
    detector_name: str
    requested_kind: str
    implementation_kind: str
    implementation_status: str
    notes: str
    metadata: Dict[str, object]


@dataclass
class DetectorSuiteResult:
    """Machine-readable products from a complete detector suite."""

    normalized_frame: pd.DataFrame
    splits: List[DetectorSplit]
    skipped_splits: List[SkippedSplit]
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    failures: List[dict]
    run_metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PreparedDetectorSuite:
    """Scientific inputs for ordered, independently durable detector fits.

    The checkpoint layer consumes this object one ``(split, detector)`` pair
    at a time. Preparation stays here so resumable and legacy runs share the
    exact split construction, ordering, seed derivation, and row schemas.
    """

    normalized_frame: pd.DataFrame
    splits: List[DetectorSplit]
    skipped_splits: List[SkippedSplit]
    detector_configs: List[dict]
    seed: int
    bootstrap_resamples: int
    threshold: float
    smoke: bool
    allow_model_downloads: bool
    run_metadata: Dict[str, object] = field(default_factory=dict)


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_pinned_model_artifacts(
    model_directory: Path,
    expected_artifacts: Mapping[str, object],
    *,
    required_directory: Optional[Path] = None,
    model_id: Optional[str] = None,
    upstream_revision: Optional[str] = None,
) -> Dict[str, object]:
    """Verify a frozen, closed-world local model-artifact directory.

    The directory and its allowed artifacts must be real, non-symlink
    filesystem objects. Missing files, extra entries, non-regular files, size
    mismatches, and SHA-256 mismatches all fail before a model loader is called.
    A real ``.cache`` directory is the sole ignored top-level metadata entry;
    it is never treated as a model artifact and may not contain symlinks.
    """

    directory = Path(os.path.abspath(os.fspath(Path(model_directory))))
    required: Optional[Path] = None
    if required_directory is not None:
        required = Path(os.path.abspath(os.fspath(Path(required_directory))))
        if directory != required:
            raise RevisionDetectionError(
                "Pinned model directory must be exactly {}; got {}.".format(
                    required, directory
                )
            )
    if directory.is_symlink():
        raise RevisionDetectionError(
            "Pinned model directory must not be a symlink: {}".format(directory)
        )
    try:
        directory_stat = directory.lstat()
    except FileNotFoundError as exc:
        raise RevisionDetectionError(
            "Pinned model directory does not exist: {}".format(directory)
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise RevisionDetectionError(
            "Pinned model path is not a directory: {}".format(directory)
        )

    if required is not None:
        try:
            relative_parts = required.relative_to(PROJECT_ROOT).parts
        except ValueError:
            relative_parts = ()
        current = PROJECT_ROOT
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise RevisionDetectionError(
                    "Pinned model path component must not be a symlink: {}".format(
                        current
                    )
                )

    expected_names = set(map(str, expected_artifacts))
    entries = list(directory.iterdir())
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if symlinks:
        raise RevisionDetectionError(
            "Pinned model artifacts must not be symlinks: {}.".format(
                ", ".join(symlinks)
            )
        )
    cache_entry = directory / ".cache"
    if cache_entry.exists():
        if not cache_entry.is_dir():
            raise RevisionDetectionError(
                "Ignored model cache entry must be a real directory: {}".format(
                    cache_entry
                )
            )
        for cache_root, cache_directories, cache_files in os.walk(
            cache_entry, followlinks=False
        ):
            for cache_name in list(cache_directories) + list(cache_files):
                cache_path = Path(cache_root) / cache_name
                if cache_path.is_symlink():
                    raise RevisionDetectionError(
                        "Pinned model cache metadata must not contain symlinks: {}".format(
                            cache_path
                        )
                    )
    actual_names = {entry.name for entry in entries if entry.name != ".cache"}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        raise RevisionDetectionError(
            "Pinned model directory is missing required artifacts: {}.".format(
                ", ".join(missing)
            )
        )
    if extra:
        raise RevisionDetectionError(
            "Pinned model directory contains disallowed extra entries: {}.".format(
                ", ".join(extra)
            )
        )

    artifact_rows: List[dict] = []
    for name in sorted(expected_names):
        artifact_path = directory / name
        file_stat = artifact_path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise RevisionDetectionError(
                "Pinned model artifact must be a regular file: {}".format(artifact_path)
            )
        specification = expected_artifacts[name]
        if isinstance(specification, Mapping):
            expected_sha256 = str(specification.get("sha256", "")).lower()
            expected_size = specification.get("size_bytes")
        else:
            expected_sha256 = str(specification).lower()
            expected_size = None
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise RevisionDetectionError(
                "Pinned SHA-256 for {!r} is invalid.".format(name)
            )
        if expected_size is not None and int(file_stat.st_size) != int(expected_size):
            raise RevisionDetectionError(
                "Pinned model artifact size mismatch for {}: expected {}, got {}.".format(
                    name, int(expected_size), int(file_stat.st_size)
                )
            )
        actual_sha256 = _sha256_file(artifact_path)
        if actual_sha256 != expected_sha256:
            raise RevisionDetectionError(
                "Pinned model artifact SHA-256 mismatch for {}: expected {}, got {}.".format(
                    name, expected_sha256, actual_sha256
                )
            )
        artifact_rows.append(
            {
                "path": name,
                "size_bytes": int(file_stat.st_size),
                "sha256": actual_sha256,
            }
        )

    frozen_identity = {
        "model_id": model_id,
        "upstream_revision": upstream_revision,
        "artifacts": artifact_rows,
    }
    return {
        "schema_version": "rankcloak-model-artifact-pin-v1",
        "policy": "exact_regular_files_no_symlinks_cache_metadata_ignored",
        "verification_status": "verified",
        "model_directory": str(directory),
        "ignored_top_level_entries": [".cache"] if cache_entry.exists() else [],
        **frozen_identity,
        "artifact_set_sha256": _canonical_json_sha256(frozen_identity),
    }


def deterministic_model_state_sha256(model_or_state: object) -> str:
    """Hash sorted tensor names, dtypes, shapes, and contiguous CPU bytes."""

    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise RevisionDetectionError("Model state must be a non-empty tensor mapping.")

    digest = hashlib.sha256()
    digest.update((MODEL_STATE_HASH_ALGORITHM + "\0").encode("ascii"))

    def update_field(value: bytes) -> None:
        digest.update(struct.pack(">Q", len(value)))
        digest.update(value)

    for name in sorted(map(str, state)):
        tensor = state[name]
        if not all(hasattr(tensor, attribute) for attribute in ("detach", "cpu", "shape")):
            raise RevisionDetectionError(
                "Model state entry {!r} is not a tensor.".format(name)
            )
        normalized = tensor.detach().cpu().contiguous()
        update_field(name.encode("utf-8"))
        update_field(str(normalized.dtype).encode("ascii"))
        shape = tuple(map(int, normalized.shape))
        update_field(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        try:
            raw_bytes = normalized.numpy().tobytes(order="C")
        except TypeError:
            # NumPy has no native bfloat16 representation. A byte view retains
            # the tensor's exact contiguous storage without numeric conversion.
            try:
                import torch

                raw_bytes = normalized.view(torch.uint8).numpy().tobytes(order="C")
            except Exception as exc:  # pragma: no cover - unusual tensor dtype
                raise RevisionDetectionError(
                    "Cannot serialize model state tensor {!r}: {}".format(name, exc)
                ) from exc
        update_field(raw_bytes)
    return digest.hexdigest()


def deterministic_model_state_schema_sha256(model_or_state: object) -> str:
    """Hash sorted tensor names, dtypes, and shapes without trained values."""

    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise RevisionDetectionError("Model state must be a non-empty tensor mapping.")
    rows = []
    for name in sorted(map(str, state)):
        tensor = state[name]
        if not all(hasattr(tensor, attribute) for attribute in ("detach", "shape")):
            raise RevisionDetectionError(
                "Model state entry {!r} is not a tensor.".format(name)
            )
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.detach().dtype),
                "shape": list(map(int, tensor.shape)),
            }
        )
    return _canonical_json_sha256(
        {
            "algorithm": MODEL_STATE_SCHEMA_HASH_ALGORITHM,
            "tensors": rows,
        }
    )


def _stable_seed(seed: int, *parts: object) -> int:
    material = "\x1f".join([str(int(seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def _nonempty_string_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


def normalize_detector_frame(
    frame: pd.DataFrame,
    columns: Optional[DetectorColumns] = None,
) -> pd.DataFrame:
    """Validate and normalize raw detector data to the canonical schema.

    ``payload_group_id`` is mandatory.  Controls should be assigned to the
    same group as their payload-matched stego sample, or to another documented
    independent group; empty synthetic placeholders are rejected.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RevisionDetectionError("Detector input must be a non-empty pandas DataFrame.")
    spec = columns or DetectorColumns()
    missing = [
        spec.source_name(name)
        for name in REQUIRED_SOURCE_FIELDS
        if spec.source_name(name) not in frame.columns
    ]
    if missing:
        raise RevisionDetectionError(
            "Detector input is missing required columns: {}".format(", ".join(sorted(missing)))
        )

    rename_map = {
        spec.source_name(name): name
        for name in CANONICAL_COLUMNS
        if spec.source_name(name) in frame.columns
    }
    normalized = frame.rename(columns=rename_map).copy().reset_index(drop=True)
    for column in (
        "text",
        "payload_group_id",
        "prompt_template_id",
        "model_id",
        "codec_id",
    ):
        if normalized[column].isna().any():
            raise RevisionDetectionError("Column {!r} contains missing values.".format(column))
        normalized[column] = _nonempty_string_series(normalized[column])
        empty_positions = np.flatnonzero(normalized[column].eq("").to_numpy())
        if empty_positions.size:
            raise RevisionDetectionError(
                "Column {!r} contains empty values at rows {}.".format(
                    column, empty_positions[:10].tolist()
                )
            )

    labels = pd.to_numeric(normalized["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise RevisionDetectionError("Detector labels must be numeric binary values 0 or 1.")
    normalized["label"] = labels.astype(int)
    if normalized["label"].nunique() != 2:
        raise RevisionDetectionError("Detector input must contain both cover and stego labels.")

    normalized["text_sha256"] = normalized["text"].map(_sha256_text)
    if "row_id" not in normalized.columns:
        identity_columns = [
            "text_sha256",
            "label",
            "payload_group_id",
            "prompt_template_id",
            "model_id",
            "codec_id",
        ]
        normalized["row_id"] = normalized[identity_columns].apply(
            lambda row: "auto-{}".format(
                _sha256_text("\x1f".join(map(str, row.tolist())))[:24]
            ),
            axis=1,
        )
    else:
        if normalized["row_id"].isna().any():
            raise RevisionDetectionError("row_id values must not be missing.")
        normalized["row_id"] = _nonempty_string_series(normalized["row_id"])
        if normalized["row_id"].eq("").any():
            raise RevisionDetectionError("row_id values must be non-empty.")
    duplicated_ids = normalized.loc[normalized["row_id"].duplicated(False), "row_id"].unique()
    if len(duplicated_ids):
        raise RevisionDetectionError(
            "row_id values must be unique; duplicated values include {}.".format(
                sorted(map(str, duplicated_ids))[:5]
            )
        )
    return normalized


def read_detector_frame(path: Path) -> pd.DataFrame:
    """Read a CSV or JSON Lines detector dataset without inferring a split."""

    path = Path(path)
    if not path.is_file():
        raise RevisionDetectionError("Detector input does not exist: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise RevisionDetectionError("Detector input must be CSV or JSON Lines: {}".format(path))


def _has_both_labels(frame: pd.DataFrame, indices: Sequence[int]) -> bool:
    if len(indices) == 0:
        return False
    return frame.iloc[list(indices)]["label"].nunique() == 2


def _has_exact_label_balance(frame: pd.DataFrame, indices: Sequence[int]) -> bool:
    if len(indices) == 0:
        return False
    counts = frame.iloc[list(indices)]["label"].value_counts()
    return int(counts.get(0, 0)) == int(counts.get(1, 0)) > 0


def deterministic_payload_group_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.25,
    seed: int = DEFAULT_SEED,
    attempts: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic exactly balanced split of whole payload groups.

    Candidate assignments are ranked by deviation of the test-set positive
    fraction from the complete dataset.  Row order never influences the group
    assignment.
    """

    if not 0.0 < float(test_fraction) < 1.0:
        raise RevisionDetectionError("test_fraction must be strictly between 0 and 1.")
    groups = sorted(frame["payload_group_id"].astype(str).unique())
    if len(groups) < 2:
        raise RevisionDetectionError("At least two payload groups are required for a split.")
    test_group_count = int(round(len(groups) * float(test_fraction)))
    test_group_count = min(len(groups) - 1, max(1, test_group_count))
    overall_rate = float(frame["label"].mean())
    candidates: List[Tuple[Tuple[float, float, str], Tuple[str, ...]]] = []
    max_attempts = max(1, int(attempts))
    for attempt in range(max_attempts):
        ordered = sorted(
            groups,
            key=lambda group: _sha256_text(
                "{}\x1f{}\x1f{}".format(int(seed), int(attempt), group)
            ),
        )
        test_groups = tuple(sorted(ordered[:test_group_count]))
        test_mask = frame["payload_group_id"].astype(str).isin(test_groups)
        train_mask = ~test_mask
        train_indices = tuple(map(int, np.flatnonzero(train_mask.to_numpy())))
        test_indices = tuple(map(int, np.flatnonzero(test_mask.to_numpy())))
        if not _has_both_labels(frame, train_indices) or not _has_both_labels(frame, test_indices):
            continue
        if not _has_exact_label_balance(
            frame, train_indices
        ) or not _has_exact_label_balance(frame, test_indices):
            continue
        test_rate = float(frame.iloc[list(test_indices)]["label"].mean())
        row_fraction = len(test_indices) / float(len(frame))
        score = (
            abs(test_rate - overall_rate),
            abs(row_fraction - float(test_fraction)),
            "|".join(test_groups),
        )
        candidates.append((score, test_groups))
    if not candidates:
        raise RevisionDetectionError(
            "Could not form exactly class-balanced train/test payload-group partitions."
        )
    _, selected_test_groups = min(candidates, key=lambda item: item[0])
    selected_mask = frame["payload_group_id"].astype(str).isin(selected_test_groups)
    train = np.flatnonzero((~selected_mask).to_numpy()).astype(int)
    test = np.flatnonzero(selected_mask.to_numpy()).astype(int)
    return train, test


def _partition_crossed_held_out_condition(
    frame: pd.DataFrame,
    *,
    held_out_column: str,
    held_out_value: str,
    test_fraction: float,
    seed: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """Partition payloads before evaluating a crossed held-out condition.

    In the primary matrix every payload occurs under every model and under the
    all-payload codec conditions. Testing all rows for one such condition and
    purging its payloads would therefore remove the entire training set. This
    helper first chooses a deterministic, label-balanced subset of payload
    groups within the held-out condition. The test set contains the held-out
    condition only for those groups; training contains non-held-out conditions
    only for the complementary groups. No payload can cross the boundary and
    the held-out value remains absent from training.
    """

    held_out_mask = frame[held_out_column].astype(str).eq(str(held_out_value))
    held_out_positions = np.flatnonzero(held_out_mask.to_numpy()).astype(int)
    if not len(held_out_positions):
        raise RevisionDetectionError(
            "Held-out value {!r} is absent from {}.".format(
                held_out_value, held_out_column
            )
        )
    held_out_frame = frame.iloc[held_out_positions].reset_index(drop=True)
    _, local_test_indices = deterministic_payload_group_split(
        held_out_frame,
        test_fraction=test_fraction,
        seed=seed,
    )
    selected_test_groups = set(
        held_out_frame.iloc[local_test_indices]["payload_group_id"].astype(str)
    )
    group_is_test = frame["payload_group_id"].astype(str).isin(selected_test_groups)
    candidate_train_mask = ~held_out_mask
    purged_train_mask = candidate_train_mask & group_is_test
    train_mask = candidate_train_mask & ~group_is_test
    test_mask = held_out_mask & group_is_test
    excluded_held_out_mask = held_out_mask & ~group_is_test
    return (
        tuple(map(int, np.flatnonzero(train_mask.to_numpy()))),
        tuple(map(int, np.flatnonzero(test_mask.to_numpy()))),
        int(purged_train_mask.sum()),
        int(excluded_held_out_mask.sum()),
    )


def validate_confirmatory_detector_frame(
    frame: pd.DataFrame,
    contract: Mapping[str, object],
) -> Tuple[str, ...]:
    """Validate the closed-world primary detector corpus and split identities."""

    required_counts = {
        "rows": int(contract["rows"]),
        "payload_groups": int(contract["payload_groups"]),
        "positive_rows": int(contract["positive_rows"]),
        "negative_rows": int(contract["negative_rows"]),
    }
    observed_counts = {
        "rows": int(len(frame)),
        "payload_groups": int(frame["payload_group_id"].nunique()),
        "positive_rows": int(frame["label"].eq(1).sum()),
        "negative_rows": int(frame["label"].eq(0).sum()),
    }
    if observed_counts != required_counts:
        raise RevisionDetectionError(
            "Confirmatory detector input is not the complete primary corpus: "
            "expected {}, observed {}.".format(required_counts, observed_counts)
        )
    expected_by_column = {
        "prompt_template_id": tuple(
            sorted(map(str, contract["prompt_template_ids"]))
        ),
        "model_id": tuple(sorted(map(str, contract["model_ids"]))),
        "codec_id": tuple(sorted(map(str, contract["codec_ids"]))),
    }
    for column, expected in expected_by_column.items():
        observed = tuple(sorted(frame[column].astype(str).unique()))
        if observed != expected:
            raise RevisionDetectionError(
                "Confirmatory primary detector {} levels differ: expected {}, "
                "observed {}.".format(column, list(expected), list(observed))
            )
        counts = frame.groupby(column, dropna=False)["label"].nunique()
        if not counts.eq(2).all():
            raise RevisionDetectionError(
                "Every confirmatory primary {} level must contain both labels.".format(
                    column
                )
            )
    group_label_counts = frame.groupby("payload_group_id", dropna=False)[
        "label"
    ].nunique()
    if not group_label_counts.eq(2).all():
        raise RevisionDetectionError(
            "Every confirmatory primary payload group must contain both labels."
        )
    group_class_counts = frame.groupby(
        ["payload_group_id", "label"], dropna=False
    ).size().unstack(fill_value=0)
    if (
        0 not in group_class_counts.columns
        or 1 not in group_class_counts.columns
        or not group_class_counts[0].eq(group_class_counts[1]).all()
    ):
        raise RevisionDetectionError(
            "Every confirmatory primary payload group must contain exactly balanced labels."
        )
    split_ids = ["matched"]
    split_ids.extend(
        "held_out_template:{}".format(value)
        for value in expected_by_column["prompt_template_id"]
    )
    split_ids.extend(
        "leave_one_model:{}".format(value)
        for value in expected_by_column["model_id"]
    )
    split_ids.extend(
        "leave_one_codec:{}".format(value)
        for value in expected_by_column["codec_id"]
    )
    expected_split_count = int(contract.get("split_count", len(split_ids)))
    if len(split_ids) != expected_split_count:
        raise RevisionDetectionError(
            "Confirmatory detector contract expected {} split identities but "
            "factor levels imply {}.".format(expected_split_count, len(split_ids))
        )
    return tuple(split_ids)


def assert_no_split_leakage(
    frame: pd.DataFrame,
    split: DetectorSplit,
    check_text_hash: bool = True,
) -> None:
    """Fail if rows, payloads, or optionally identical texts cross a split."""

    train_positions = set(map(int, split.train_indices))
    test_positions = set(map(int, split.test_indices))
    row_overlap = train_positions & test_positions
    if row_overlap:
        raise RevisionDetectionError(
            "Split {} repeats positional rows across train and test.".format(split.split_id)
        )
    train = frame.iloc[list(split.train_indices)]
    test = frame.iloc[list(split.test_indices)]
    group_overlap = set(train["payload_group_id"]) & set(test["payload_group_id"])
    if group_overlap:
        raise RevisionDetectionError(
            "Split {} leaks payload groups: {}".format(
                split.split_id, sorted(map(str, group_overlap))[:10]
            )
        )
    id_overlap = set(train["row_id"]) & set(test["row_id"])
    if id_overlap:
        raise RevisionDetectionError(
            "Split {} leaks row IDs: {}".format(split.split_id, sorted(id_overlap)[:10])
        )
    if check_text_hash:
        text_overlap = set(train["text_sha256"]) & set(test["text_sha256"])
        if text_overlap:
            raise RevisionDetectionError(
                "Split {} contains identical raw text across train and test ({} hashes).".format(
                    split.split_id, len(text_overlap)
                )
            )
    if split.held_out_column:
        held_out = str(split.held_out_value)
        if train[split.held_out_column].astype(str).eq(held_out).any():
            raise RevisionDetectionError(
                "Split {} leaves held-out value {!r} in training.".format(
                    split.split_id, held_out
                )
            )
        if not test[split.held_out_column].astype(str).eq(held_out).all():
            raise RevisionDetectionError(
                "Split {} test rows do not all have held-out value {!r}.".format(
                    split.split_id, held_out
                )
            )


def build_evaluation_splits(
    frame: pd.DataFrame,
    regimes: Sequence[str] = (
        "matched",
        "held_out_template",
        "leave_one_model",
        "leave_one_codec",
    ),
    test_fraction: float = 0.25,
    seed: int = DEFAULT_SEED,
    check_text_hash: bool = True,
    minimum_train_rows: int = 4,
    minimum_test_rows: int = 2,
) -> Tuple[List[DetectorSplit], List[SkippedSplit]]:
    """Build matched and out-of-domain splits without payload leakage.

    Model and codec conditions are crossed with payloads in the primary matrix,
    so those regimes use deterministic disjoint payload partitions. Template
    conditions retain the stronger full-condition test whenever their payload
    groups are not fully crossed.
    """

    requested = list(dict.fromkeys(map(str, regimes)))
    unknown = sorted(set(requested) - ({"matched"} | set(REGIME_COLUMNS)))
    if unknown:
        raise RevisionDetectionError("Unknown split regimes: {}".format(", ".join(unknown)))
    splits: List[DetectorSplit] = []
    skipped: List[SkippedSplit] = []
    if "matched" in requested:
        train_indices, test_indices = deterministic_payload_group_split(
            frame, test_fraction=test_fraction, seed=seed
        )
        split = DetectorSplit(
            split_id="matched",
            regime="matched",
            train_indices=tuple(map(int, train_indices)),
            test_indices=tuple(map(int, test_indices)),
        )
        if not _has_exact_label_balance(
            frame, split.train_indices
        ) or not _has_exact_label_balance(frame, split.test_indices):
            raise RevisionDetectionError(
                "Matched detector split is not exactly class-balanced."
            )
        assert_no_split_leakage(frame, split, check_text_hash=check_text_hash)
        splits.append(split)

    for regime in requested:
        if regime == "matched":
            continue
        held_out_column = REGIME_COLUMNS[regime]
        values = sorted(frame[held_out_column].astype(str).unique())
        for value in values:
            split_id = "{}:{}".format(regime, value)
            partition_policy = "full_held_out_condition"
            excluded_held_out_rows = 0
            if regime in {"leave_one_model", "leave_one_codec"}:
                (
                    train_indices,
                    test_indices,
                    purged_train_rows,
                    excluded_held_out_rows,
                ) = _partition_crossed_held_out_condition(
                    frame,
                    held_out_column=held_out_column,
                    held_out_value=value,
                    test_fraction=test_fraction,
                    seed=_stable_seed(seed, regime, value, "payload_partition"),
                )
                partition_policy = "deterministic_disjoint_payload_partition_v1"
            else:
                test_mask = frame[held_out_column].astype(str).eq(value)
                test_groups = set(
                    frame.loc[test_mask, "payload_group_id"].astype(str)
                )
                candidate_train_mask = ~test_mask
                purged_mask = candidate_train_mask & frame[
                    "payload_group_id"
                ].astype(str).isin(test_groups)
                train_mask = candidate_train_mask & ~purged_mask
                train_indices = tuple(
                    map(int, np.flatnonzero(train_mask.to_numpy()))
                )
                test_indices = tuple(map(int, np.flatnonzero(test_mask.to_numpy())))
                purged_train_rows = int(purged_mask.sum())
            reason = None
            if len(train_indices) < int(minimum_train_rows):
                reason = "fewer than {} training rows after payload-group purge".format(
                    int(minimum_train_rows)
                )
            elif len(test_indices) < int(minimum_test_rows):
                reason = "fewer than {} test rows".format(int(minimum_test_rows))
            elif not _has_both_labels(frame, train_indices):
                reason = "training partition does not contain both labels"
            elif not _has_both_labels(frame, test_indices):
                reason = "test partition does not contain both labels"
            elif not _has_exact_label_balance(frame, train_indices):
                reason = "training partition is not exactly class-balanced"
            elif not _has_exact_label_balance(frame, test_indices):
                reason = "test partition is not exactly class-balanced"
            if reason is not None:
                skipped.append(
                    SkippedSplit(
                        split_id=split_id,
                        regime=regime,
                        held_out_column=held_out_column,
                        held_out_value=value,
                        reason=reason,
                    )
                )
                continue
            split = DetectorSplit(
                split_id=split_id,
                regime=regime,
                train_indices=train_indices,
                test_indices=test_indices,
                held_out_column=held_out_column,
                held_out_value=value,
                purged_train_rows=int(purged_train_rows),
                partition_policy=partition_policy,
                excluded_held_out_rows=int(excluded_held_out_rows),
            )
            assert_no_split_leakage(frame, split, check_text_hash=check_text_hash)
            splits.append(split)
    if not splits:
        raise RevisionDetectionError("No requested detector split could be constructed.")
    return splits, skipped


def split_manifest_rows(frame: pd.DataFrame, splits: Sequence[DetectorSplit]) -> List[dict]:
    """Return concise, auditable split metadata without model-dependent state."""

    rows = []
    for split in splits:
        train = frame.iloc[list(split.train_indices)]
        test = frame.iloc[list(split.test_indices)]
        rows.append(
            {
                "split_id": split.split_id,
                "regime": split.regime,
                "held_out_column": split.held_out_column,
                "held_out_value": split.held_out_value,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_payload_groups": int(train["payload_group_id"].nunique()),
                "test_payload_groups": int(test["payload_group_id"].nunique()),
                "train_positive_rows": int(train["label"].sum()),
                "test_positive_rows": int(test["label"].sum()),
                "purged_train_rows": int(split.purged_train_rows),
                "partition_policy": split.partition_policy,
                "excluded_held_out_rows": int(split.excluded_held_out_rows),
                "train_row_ids_sha256": _sha256_text(
                    "\n".join(sorted(train["row_id"].astype(str)))
                ),
                "test_row_ids_sha256": _sha256_text(
                    "\n".join(sorted(test["row_id"].astype(str)))
                ),
            }
        )
    return rows


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    wins = 0.0
    for positive in positives:
        wins += float(np.count_nonzero(positive > negatives))
        wins += 0.5 * float(np.count_nonzero(positive == negatives))
    return wins / float(positives.size * negatives.size)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positive_count = int(np.count_nonzero(labels == 1))
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    cumulative_positive = np.cumsum(ordered_labels == 1)
    distinct_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    precisions = cumulative_positive[distinct_ends] / (distinct_ends + 1.0)
    recalls = cumulative_positive[distinct_ends] / float(positive_count)
    recall_increments = np.diff(np.r_[0.0, recalls])
    return float(np.sum(recall_increments * precisions))


def binary_detector_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Compute prespecified detector metrics from positive-class scores."""

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size or y.size == 0:
        raise RevisionDetectionError("labels and scores must be non-empty equal-length vectors.")
    if not np.isin(y, [0, 1]).all():
        raise RevisionDetectionError("Detector labels must contain only 0 and 1.")
    if not np.isfinite(s).all():
        raise RevisionDetectionError("Detector scores must all be finite.")
    predictions = (s >= float(threshold)).astype(int)
    tp = int(np.count_nonzero((y == 1) & (predictions == 1)))
    tn = int(np.count_nonzero((y == 0) & (predictions == 0)))
    fp = int(np.count_nonzero((y == 0) & (predictions == 1)))
    fn = int(np.count_nonzero((y == 1) & (predictions == 0)))
    sensitivity = tp / float(tp + fn) if tp + fn else None
    specificity = tn / float(tn + fp) if tn + fp else None
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * sensitivity / float(precision + sensitivity)
        if sensitivity is not None and precision + sensitivity > 0
        else 0.0
    )
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "roc_auc": _roc_auc(y, s),
        "pr_auc": _average_precision(y, s),
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def grouped_bootstrap_detector_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    payload_groups: Sequence[object],
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
    threshold: float = 0.5,
) -> dict:
    """Calculate percentile intervals by resampling whole payload groups."""

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(list(map(str, payload_groups)), dtype=object)
    if y.size != s.size or y.size != groups.size or y.size == 0:
        raise RevisionDetectionError("labels, scores, and payload_groups must align.")
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if unique_groups.size < 2:
        raise RevisionDetectionError("Grouped bootstrap requires at least two payload groups.")
    point = binary_detector_metrics(y, s, threshold=threshold)
    samples: Dict[str, List[float]] = {name: [] for name in point}
    rng = np.random.default_rng(int(seed))
    group_positions = {
        group: np.flatnonzero(groups == group).astype(int) for group in unique_groups
    }
    for _ in range(max(0, int(n_resamples))):
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        positions = np.concatenate([group_positions[group] for group in sampled_groups])
        replicate = binary_detector_metrics(y[positions], s[positions], threshold=threshold)
        for name, value in replicate.items():
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    result: Dict[str, object] = {
        "bootstrap_unit": "payload_group_id",
        "bootstrap_resamples_requested": int(n_resamples),
        "test_payload_groups": int(unique_groups.size),
    }
    for name, value in point.items():
        result[name] = value
        values = samples[name]
        result["{}_bootstrap_valid".format(name)] = int(len(values))
        if values:
            low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
            result["{}_ci_low_95".format(name)] = float(low)
            result["{}_ci_high_95".format(name)] = float(high)
        else:
            result["{}_ci_low_95".format(name)] = None
            result["{}_ci_high_95".format(name)] = None
    return result


def _tokenize_words(text: object, lowercase: bool = True) -> List[str]:
    value = str(text)
    if lowercase:
        value = value.lower()
    return TOKEN_PATTERN.findall(value)


def _build_train_vocabulary(
    texts: Sequence[str],
    minimum_frequency: int,
    maximum_vocabulary_size: int,
    lowercase: bool,
) -> Dict[str, int]:
    counts: Counter = Counter()
    for text in texts:
        counts.update(_tokenize_words(text, lowercase=lowercase))
    eligible = [
        (token, count)
        for token, count in counts.items()
        if int(count) >= int(minimum_frequency)
    ]
    eligible.sort(key=lambda item: (-int(item[1]), str(item[0])))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    limit = max(2, int(maximum_vocabulary_size))
    for token, _ in eligible[: max(0, limit - len(vocabulary))]:
        vocabulary[str(token)] = len(vocabulary)
    return vocabulary


def _encode_word_texts(
    texts: Sequence[str],
    vocabulary: Mapping[str, int],
    maximum_length: int,
    minimum_length: int,
    lowercase: bool,
) -> np.ndarray:
    width = max(int(minimum_length), int(maximum_length))
    encoded = np.zeros((len(texts), width), dtype=np.int64)
    unknown = int(vocabulary["<UNK>"])
    for row_index, text in enumerate(texts):
        token_ids = [
            int(vocabulary.get(token, unknown))
            for token in _tokenize_words(text, lowercase=lowercase)[:width]
        ]
        if token_ids:
            encoded[row_index, : len(token_ids)] = token_ids
    return encoded


def _run_hashed_ngram_logistic(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    config: Mapping[str, object],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit the explicitly non-neural smoke/offline fallback."""

    try:
        import sklearn
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RevisionDetectionError(
            "The smoke fallback requires scikit-learn: {}".format(exc)
        ) from exc
    labels = np.asarray(train_labels, dtype=int)
    if len(np.unique(labels)) != 2:
        raise RevisionDetectionError("Fallback training data must contain both labels.")
    ngram_min = int(config.get("fallback_ngram_min", 3))
    ngram_max = int(config.get("fallback_ngram_max", 5))
    feature_count = int(config.get("fallback_maximum_features", 20000))
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=bool(config.get("lowercase", True)),
        n_features=feature_count,
        alternate_sign=False,
        norm="l2",
        dtype=np.float64,
    )
    train_matrix = vectorizer.transform(list(map(str, train_texts)))
    test_matrix = vectorizer.transform(list(map(str, test_texts)))
    classifier = LogisticRegression(
        C=float(config.get("fallback_c", 1.0)),
        max_iter=int(config.get("fallback_maximum_iterations", 1000)),
        random_state=int(seed),
        solver="liblinear",
    )
    classifier.fit(train_matrix, labels)
    scores = classifier.predict_proba(test_matrix)[:, 1].astype(np.float64)
    return scores, {
        "scikit_learn_version": str(sklearn.__version__),
        "fallback_hash_features": feature_count,
        "fallback_ngram_range": [ngram_min, ngram_max],
    }


def _configure_torch_determinism(torch: Any, seed: int, config: Mapping[str, object]) -> None:
    # Must be set before the first CUDA/cuBLAS operation when deterministic
    # algorithms are enforced.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    requested_device = str(config.get("device", "cpu"))
    if (
        requested_device.startswith("cuda")
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(max(1, int(config.get("torch_num_threads", 1))))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _torch_device(torch: Any, config: Mapping[str, object]) -> Any:
    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RevisionDetectionError("CUDA was requested for a detector but is unavailable.")
    return torch.device(requested)


def _run_torch_text_cnn(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    config: Mapping[str, object],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Train an independent TS-CNN/TextCNN-equivalent raw-text baseline.

    The architecture uses train-only word vocabulary construction, a learned
    embedding, parallel one-dimensional convolutions, ReLU, global max pooling,
    dropout, and binary logits. These are published TextCNN ingredients used
    by linguistic-steganalysis comparators; this is not copied official code.
    """

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RevisionDetectionError("PyTorch is unavailable: {}".format(exc)) from exc
    _configure_torch_determinism(torch, seed, config)
    filter_sizes = tuple(map(int, config.get("filter_sizes", [3, 5])))
    if not filter_sizes or min(filter_sizes) < 1:
        raise RevisionDetectionError("text_cnn filter_sizes must contain positive integers.")
    lowercase = bool(config.get("lowercase", True))
    vocabulary = _build_train_vocabulary(
        list(map(str, train_texts)),
        minimum_frequency=int(config.get("minimum_token_frequency", 1)),
        maximum_vocabulary_size=int(config.get("maximum_vocabulary_size", 30000)),
        lowercase=lowercase,
    )
    maximum_length = int(config.get("maximum_length", 256))
    train_ids = _encode_word_texts(
        list(map(str, train_texts)),
        vocabulary,
        maximum_length=maximum_length,
        minimum_length=max(filter_sizes),
        lowercase=lowercase,
    )
    test_ids = _encode_word_texts(
        list(map(str, test_texts)),
        vocabulary,
        maximum_length=maximum_length,
        minimum_length=max(filter_sizes),
        lowercase=lowercase,
    )
    labels = np.asarray(train_labels, dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise RevisionDetectionError("text_cnn training data must contain both labels.")
    embedding_dimension = int(config.get("embedding_dimension", 128))
    filters_per_width = int(config.get("filters_per_width", 100))
    dropout = float(config.get("dropout", 0.5))

    class TextCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                len(vocabulary), embedding_dimension, padding_idx=0
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=embedding_dimension,
                        out_channels=filters_per_width,
                        kernel_size=width,
                    )
                    for width in filter_sizes
                ]
            )
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(filters_per_width * len(filter_sizes), 2)

        def forward(self, token_ids: Any) -> Any:
            embedded = self.embedding(token_ids).transpose(1, 2)
            pooled = [
                torch.max(torch.relu(convolution(embedded)), dim=2).values
                for convolution in self.convolutions
            ]
            return self.output(self.dropout(torch.cat(pooled, dim=1)))

    device = _torch_device(torch, config)
    model = TextCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = max(1, int(config.get("epochs", 10)))
    batch_size = max(1, int(config.get("batch_size", 64)))
    rng = np.random.default_rng(int(seed))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_ids))
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            batch_ids = torch.as_tensor(train_ids[positions], dtype=torch.long, device=device)
            batch_labels = torch.as_tensor(labels[positions], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    model_state_sha256 = deterministic_model_state_sha256(model)
    model_state_schema_sha256 = deterministic_model_state_schema_sha256(model)
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(test_ids), batch_size):
            batch_ids = torch.as_tensor(
                test_ids[start : start + batch_size], dtype=torch.long, device=device
            )
            batches.append(torch.softmax(model(batch_ids), dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(batches).astype(np.float64)
    return scores, {
        "torch_version": str(torch.__version__),
        "architecture": "word_embedding_parallel_conv_relu_global_max_pool_dropout",
        "vocabulary_size": int(len(vocabulary)),
        "embedding_dimension": embedding_dimension,
        "filter_sizes": list(filter_sizes),
        "filters_per_width": filters_per_width,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "maximum_length": int(train_ids.shape[1]),
        "device": str(device),
        "test_tuning": False,
        "model_state_hash_algorithm": MODEL_STATE_HASH_ALGORITHM,
        "model_state_sha256": model_state_sha256,
        "model_state_schema_hash_algorithm": MODEL_STATE_SCHEMA_HASH_ALGORITHM,
        "model_state_schema_sha256": model_state_schema_sha256,
    }


def _run_pretrained_transformer(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    config: Mapping[str, object],
    seed: int,
    allow_model_downloads: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fine-tune the frozen local DeBERTa-v3-base sequence classifier."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RevisionDetectionError(
            "PyTorch/transformers dependencies are unavailable: {}".format(exc)
        ) from exc
    if allow_model_downloads:
        raise RevisionDetectionError(
            "Confirmatory transformer model downloads are prohibited; stage the frozen "
            "artifacts at {}.".format(CONFIRMATORY_TRANSFORMER_RELATIVE_PATH)
        )
    artifact_verification = verify_pinned_model_artifacts(
        CONFIRMATORY_TRANSFORMER_DIRECTORY,
        CONFIRMATORY_TRANSFORMER_ARTIFACTS,
        required_directory=CONFIRMATORY_TRANSFORMER_DIRECTORY,
        model_id=CONFIRMATORY_TRANSFORMER_MODEL_ID,
        upstream_revision=CONFIRMATORY_TRANSFORMER_REVISION,
    )
    _configure_torch_determinism(torch, seed, config)
    configured_model_name = str(
        config.get("model_name_or_path", CONFIRMATORY_TRANSFORMER_MODEL_ID)
    )
    model_name = str(CONFIRMATORY_TRANSFORMER_DIRECTORY)
    local_files_only = True
    downloads_enabled = False
    revision = CONFIRMATORY_TRANSFORMER_REVISION
    common_kwargs: Dict[str, object] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    use_fast_tokenizer = bool(config.get("use_fast_tokenizer", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast_tokenizer, **common_kwargs
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=False, **common_kwargs
    )
    device = _torch_device(torch, config)
    model.to(device)
    maximum_length = int(config.get("maximum_length", 256))
    train_encodings = tokenizer(
        list(map(str, train_texts)),
        padding=True,
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    test_encodings = tokenizer(
        list(map(str, test_texts)),
        padding=True,
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    labels_array = np.asarray(train_labels, dtype=np.int64)
    if len(np.unique(labels_array)) != 2:
        raise RevisionDetectionError("Transformer training data must contain both labels.")
    labels = torch.as_tensor(labels_array, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = max(1, int(config.get("epochs", 3)))
    batch_size = max(1, int(config.get("batch_size", 8)))
    rng = np.random.default_rng(int(seed))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            batch = {
                key: value[positions].to(device) for key, value in train_encodings.items()
            }
            batch_labels = labels[positions].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, labels=batch_labels)
            output.loss.backward()
            optimizer.step()
    model_state_sha256 = deterministic_model_state_sha256(model)
    model_state_schema_sha256 = deterministic_model_state_schema_sha256(model)
    model.eval()
    score_batches = []
    with torch.no_grad():
        for start in range(0, len(test_texts), batch_size):
            batch = {
                key: value[start : start + batch_size].to(device)
                for key, value in test_encodings.items()
            }
            logits = model(**batch).logits
            score_batches.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(score_batches).astype(np.float64)
    return scores, {
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "model_name_or_path": model_name,
        "configured_model_name_or_path": configured_model_name,
        "upstream_model_id": CONFIRMATORY_TRANSFORMER_MODEL_ID,
        "model_revision": revision,
        "artifact_verification": artifact_verification,
        "model_artifact_set_sha256": artifact_verification["artifact_set_sha256"],
        "use_fast_tokenizer": use_fast_tokenizer,
        "local_files_only": local_files_only,
        "downloads_enabled": downloads_enabled,
        "maximum_length": maximum_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "test_tuning": False,
        "model_state_hash_algorithm": MODEL_STATE_HASH_ALGORITHM,
        "model_state_sha256": model_state_sha256,
        "model_state_schema_hash_algorithm": MODEL_STATE_SCHEMA_HASH_ALGORITHM,
        "model_state_schema_sha256": model_state_schema_sha256,
    }


def run_configured_detector(
    train: pd.DataFrame,
    test: pd.DataFrame,
    detector_config: Mapping[str, object],
    seed: int,
    smoke: bool = False,
    allow_model_downloads: bool = False,
) -> DetectorOutput:
    """Run one configured detector, preserving requested versus actual method."""

    requested_kind = str(detector_config.get("kind", "")).strip()
    detector_name = str(detector_config.get("name", requested_kind)).strip()
    supported = {"text_cnn", "pretrained_transformer", "hashed_ngram_smoke"}
    if requested_kind not in supported:
        raise RevisionDetectionError(
            "Unknown detector kind {!r} for {!r}.".format(requested_kind, detector_name)
        )
    if not detector_name:
        raise RevisionDetectionError("Every detector configuration requires a non-empty name.")
    train_texts = train["text"].astype(str).tolist()
    train_labels = train["label"].astype(int).to_numpy()
    test_texts = test["text"].astype(str).tolist()
    fallback = str(detector_config.get("fallback", "hashed_ngram_smoke"))
    force_fallback = bool(smoke and requested_kind != "hashed_ngram_smoke")
    try:
        if force_fallback:
            raise RevisionDetectionError("smoke mode requested the dependency-light fallback")
        if requested_kind == "text_cnn":
            scores, metadata = _run_torch_text_cnn(
                train_texts, train_labels, test_texts, detector_config, seed
            )
        elif requested_kind == "pretrained_transformer":
            scores, metadata = _run_pretrained_transformer(
                train_texts,
                train_labels,
                test_texts,
                detector_config,
                seed,
                allow_model_downloads=allow_model_downloads,
            )
        else:
            scores, metadata = _run_hashed_ngram_logistic(
                train_texts, train_labels, test_texts, detector_config, seed
            )
        return DetectorOutput(
            scores=np.asarray(scores, dtype=np.float64),
            detector_name=detector_name,
            requested_kind=requested_kind,
            implementation_kind=requested_kind,
            implementation_status="complete" if not smoke else "smoke_only",
            notes="Prespecified detector completed without test-set tuning.",
            metadata=metadata,
        )
    except Exception as exc:
        if (
            not smoke
            or requested_kind == "hashed_ngram_smoke"
            or fallback != "hashed_ngram_smoke"
        ):
            raise
        scores, metadata = _run_hashed_ngram_logistic(
            train_texts, train_labels, test_texts, detector_config, seed
        )
        metadata["fallback_reason"] = "{}: {}".format(type(exc).__name__, exc)
        return DetectorOutput(
            scores=np.asarray(scores, dtype=np.float64),
            detector_name=detector_name,
            requested_kind=requested_kind,
            implementation_kind="hashed_ngram_smoke",
            implementation_status="smoke_fallback",
            notes=(
                "Requested neural detector was not run; scores are from a feature-hashed "
                "character n-gram logistic smoke fallback and must not be reported as neural."
            ),
            metadata=metadata,
        )


def _detector_configs(config: Mapping[str, object]) -> List[dict]:
    values = config.get("detectors", [])
    if not isinstance(values, list) or not values:
        raise RevisionDetectionError("Detector config must contain a non-empty detectors list.")
    enabled = [dict(value) for value in values if bool(dict(value).get("enabled", True))]
    if not enabled:
        raise RevisionDetectionError("No detectors are enabled in the detector config.")
    names = [str(value.get("name", value.get("kind", ""))) for value in enabled]
    if len(names) != len(set(names)):
        raise RevisionDetectionError("Enabled detector names must be unique.")
    return enabled


def _confirmatory_detector_preflight(
    detector_configs: Sequence[Mapping[str, object]],
    allow_model_downloads: bool,
) -> Dict[str, object]:
    """Validate the fixed confirmatory implementations before any training."""

    if allow_model_downloads:
        raise RevisionDetectionError(
            "Confirmatory detector execution prohibits model downloads."
        )
    identities = {
        (str(value.get("name", "")), str(value.get("kind", "")))
        for value in detector_configs
    }
    required = {
        ("published_textcnn_equivalent", "text_cnn"),
        ("deberta_v3_base_classifier", "pretrained_transformer"),
    }
    if identities != required:
        raise RevisionDetectionError(
            "Confirmatory execution requires exactly these detector identities: {}.".format(
                ", ".join("{}/{}".format(name, kind) for name, kind in sorted(required))
            )
        )
    transformer_config = next(
        value
        for value in detector_configs
        if str(value.get("kind", "")) == "pretrained_transformer"
    )
    configured_model = str(transformer_config.get("model_name_or_path", ""))
    if configured_model != CONFIRMATORY_TRANSFORMER_MODEL_ID:
        raise RevisionDetectionError(
            "Confirmatory transformer config must identify upstream model {!r}.".format(
                CONFIRMATORY_TRANSFORMER_MODEL_ID
            )
        )
    configured_revision = transformer_config.get("model_revision")
    if configured_revision not in (None, CONFIRMATORY_TRANSFORMER_REVISION):
        raise RevisionDetectionError(
            "Confirmatory transformer config has an incompatible upstream revision."
        )
    if not bool(transformer_config.get("offline_only", True)):
        raise RevisionDetectionError("Confirmatory transformer must be offline_only.")
    if bool(transformer_config.get("allow_downloads", False)):
        raise RevisionDetectionError(
            "Confirmatory transformer config must prohibit downloads."
        )
    return verify_pinned_model_artifacts(
        CONFIRMATORY_TRANSFORMER_DIRECTORY,
        CONFIRMATORY_TRANSFORMER_ARTIFACTS,
        required_directory=CONFIRMATORY_TRANSFORMER_DIRECTORY,
        model_id=CONFIRMATORY_TRANSFORMER_MODEL_ID,
        upstream_revision=CONFIRMATORY_TRANSFORMER_REVISION,
    )


def run_revision_detector_suite(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    smoke: bool = False,
    allow_model_downloads: bool = False,
    confirmatory_dataset_contract: Optional[Mapping[str, object]] = None,
) -> DetectorSuiteResult:
    """Run every detector and split, returning predictions and grouped metrics."""

    seed = int(config.get("seed", DEFAULT_SEED))
    detectors = _detector_configs(config)
    run_metadata: Dict[str, object] = {
        "execution_mode": "smoke" if smoke else "confirmatory",
        "confirmatory_complete": False if not smoke else None,
        "smoke_fallback_allowed": bool(smoke),
        "artifact_pin_verification_required": not smoke,
    }
    if not smoke:
        run_metadata["transformer_artifact_verification"] = (
            _confirmatory_detector_preflight(detectors, allow_model_downloads)
        )
    columns = DetectorColumns.from_mapping(config.get("columns", {}))
    normalized = normalize_detector_frame(frame, columns=columns)
    expected_split_ids: Optional[Tuple[str, ...]] = None
    if not smoke and confirmatory_dataset_contract is not None:
        expected_split_ids = validate_confirmatory_detector_frame(
            normalized, confirmatory_dataset_contract
        )
    split_config = dict(config.get("splits", {}))
    regimes = split_config.get(
        "regimes",
        ["matched", "held_out_template", "leave_one_model", "leave_one_codec"],
    )
    splits, skipped = build_evaluation_splits(
        normalized,
        regimes=list(map(str, regimes)),
        test_fraction=float(split_config.get("matched_test_fraction", 0.25)),
        seed=seed,
        check_text_hash=bool(split_config.get("assert_text_hash_disjoint", True)),
        minimum_train_rows=int(split_config.get("minimum_train_rows", 4)),
        minimum_test_rows=int(split_config.get("minimum_test_rows", 2)),
    )
    actual_split_ids = tuple(split.split_id for split in splits)
    missing_split_ids = (
        []
        if expected_split_ids is None
        else sorted(set(expected_split_ids) - set(actual_split_ids))
    )
    unexpected_split_ids = (
        []
        if expected_split_ids is None
        else sorted(set(actual_split_ids) - set(expected_split_ids))
    )
    if (not smoke or bool(split_config.get("fail_on_skipped_split", False))) and skipped:
        raise RevisionDetectionError(
            "Requested evaluation splits were skipped: {}".format(
                "; ".join("{} ({})".format(item.split_id, item.reason) for item in skipped)
            )
        )
    if missing_split_ids or unexpected_split_ids:
        raise RevisionDetectionError(
            "Confirmatory detector split identities differ from the primary contract; "
            "missing={}, unexpected={}.".format(
                missing_split_ids, unexpected_split_ids
            )
        )
    run_metadata.update(
        {
            "split_contract": (
                None
                if expected_split_ids is None
                else {
                    "schema_version": "rankcloak-revision-detector-splits-v2",
                    "input_scope": "primary_full_detector_corpus_only",
                    "expected_split_count": int(len(expected_split_ids)),
                    "expected_split_ids": list(expected_split_ids),
                    "expected_split_ids_sha256": _sha256_text(
                        "\n".join(sorted(expected_split_ids))
                    ),
                    "dataset_contract": dict(confirmatory_dataset_contract),
                    "missing_split_ids": missing_split_ids,
                    "unexpected_split_ids": unexpected_split_ids,
                }
            )
        }
    )
    bootstrap_config = dict(config.get("bootstrap", {}))
    bootstrap_resamples = int(bootstrap_config.get("resamples", 2000))
    if smoke:
        bootstrap_resamples = min(
            bootstrap_resamples, int(bootstrap_config.get("smoke_resamples", 100))
        )
    threshold = float(config.get("decision_threshold", 0.5))
    predictions: List[dict] = []
    metric_rows: List[dict] = []
    failures: List[dict] = []
    for split in splits:
        train = normalized.iloc[list(split.train_indices)].copy()
        test = normalized.iloc[list(split.test_indices)].copy()
        for detector_config in detectors:
            detector_name = str(detector_config.get("name", detector_config.get("kind", "")))
            detector_seed = _stable_seed(seed, split.split_id, detector_name)
            try:
                output = run_configured_detector(
                    train,
                    test,
                    detector_config,
                    seed=detector_seed,
                    smoke=smoke,
                    allow_model_downloads=allow_model_downloads,
                )
                if len(output.scores) != len(test):
                    raise RevisionDetectionError(
                        "Detector {} returned {} scores for {} test rows.".format(
                            detector_name, len(output.scores), len(test)
                        )
                    )
                metric = grouped_bootstrap_detector_metrics(
                    test["label"].astype(int).to_numpy(),
                    output.scores,
                    test["payload_group_id"].astype(str).to_numpy(),
                    n_resamples=bootstrap_resamples,
                    seed=_stable_seed(detector_seed, "grouped_bootstrap"),
                    threshold=threshold,
                )
                metric_rows.append(
                    {
                        "split_id": split.split_id,
                        "regime": split.regime,
                        "held_out_column": split.held_out_column,
                        "held_out_value": split.held_out_value,
                        "detector_name": output.detector_name,
                        "requested_kind": output.requested_kind,
                        "implementation_kind": output.implementation_kind,
                        "implementation_status": output.implementation_status,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_payload_groups": int(train["payload_group_id"].nunique()),
                        "purged_train_rows": int(split.purged_train_rows),
                        "decision_threshold": threshold,
                        "seed": int(detector_seed),
                        "notes": output.notes,
                        "model_state_sha256": output.metadata.get(
                            "model_state_sha256"
                        ),
                        "model_state_hash_algorithm": output.metadata.get(
                            "model_state_hash_algorithm"
                        ),
                        "model_artifact_set_sha256": output.metadata.get(
                            "model_artifact_set_sha256"
                        ),
                        "implementation_metadata_json": json.dumps(
                            output.metadata, sort_keys=True, default=str
                        ),
                        **metric,
                    }
                )
                for (_, row), score in zip(test.iterrows(), output.scores):
                    predictions.append(
                        {
                            "split_id": split.split_id,
                            "regime": split.regime,
                            "held_out_value": split.held_out_value,
                            "detector_name": output.detector_name,
                            "requested_kind": output.requested_kind,
                            "implementation_kind": output.implementation_kind,
                            "implementation_status": output.implementation_status,
                            "row_id": row["row_id"],
                            "payload_group_id": row["payload_group_id"],
                            "prompt_template_id": row["prompt_template_id"],
                            "model_id": row["model_id"],
                            "codec_id": row["codec_id"],
                            "label": int(row["label"]),
                            "score": float(score),
                            "prediction": int(float(score) >= threshold),
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "split_id": split.split_id,
                        "regime": split.regime,
                        "detector_name": detector_name,
                        "requested_kind": str(detector_config.get("kind", "")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    expected_executions = int(len(splits) * len(detectors))
    complete_executions = sum(
        1
        for row in metric_rows
        if row["implementation_status"] == "complete"
        and row["implementation_kind"] == row["requested_kind"]
    )
    run_metadata.update(
        {
            "expected_detector_split_executions": expected_executions,
            "complete_detector_split_executions": int(complete_executions),
            "confirmatory_complete": (
                None
                if smoke
                else bool(
                    not failures
                    and not skipped
                    and complete_executions == expected_executions
                    and expected_executions > 0
                )
            ),
        }
    )
    return DetectorSuiteResult(
        normalized_frame=normalized,
        splits=splits,
        skipped_splits=skipped,
        predictions=pd.DataFrame(predictions),
        metrics=pd.DataFrame(metric_rows),
        failures=failures,
        run_metadata=run_metadata,
    )


def prepare_revision_detector_suite(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    smoke: bool = False,
    allow_model_downloads: bool = False,
    confirmatory_dataset_contract: Optional[Mapping[str, object]] = None,
) -> PreparedDetectorSuite:
    """Prepare the legacy detector matrix without fitting a model.

    This is the single-fit execution boundary used by the durable runner.  Its
    setup intentionally mirrors :func:`run_revision_detector_suite`; focused
    equivalence tests guard both paths against drift.
    """

    seed = int(config.get("seed", DEFAULT_SEED))
    detectors = _detector_configs(config)
    run_metadata: Dict[str, object] = {
        "execution_mode": "smoke" if smoke else "confirmatory",
        "confirmatory_complete": False if not smoke else None,
        "smoke_fallback_allowed": bool(smoke),
        "artifact_pin_verification_required": not smoke,
    }
    if not smoke:
        run_metadata["transformer_artifact_verification"] = (
            _confirmatory_detector_preflight(detectors, allow_model_downloads)
        )
    columns = DetectorColumns.from_mapping(config.get("columns", {}))
    normalized = normalize_detector_frame(frame, columns=columns)
    expected_split_ids: Optional[Tuple[str, ...]] = None
    if not smoke and confirmatory_dataset_contract is not None:
        expected_split_ids = validate_confirmatory_detector_frame(
            normalized, confirmatory_dataset_contract
        )
    split_config = dict(config.get("splits", {}))
    regimes = split_config.get(
        "regimes",
        ["matched", "held_out_template", "leave_one_model", "leave_one_codec"],
    )
    splits, skipped = build_evaluation_splits(
        normalized,
        regimes=list(map(str, regimes)),
        test_fraction=float(split_config.get("matched_test_fraction", 0.25)),
        seed=seed,
        check_text_hash=bool(split_config.get("assert_text_hash_disjoint", True)),
        minimum_train_rows=int(split_config.get("minimum_train_rows", 4)),
        minimum_test_rows=int(split_config.get("minimum_test_rows", 2)),
    )
    actual_split_ids = tuple(split.split_id for split in splits)
    missing_split_ids = (
        []
        if expected_split_ids is None
        else sorted(set(expected_split_ids) - set(actual_split_ids))
    )
    unexpected_split_ids = (
        []
        if expected_split_ids is None
        else sorted(set(actual_split_ids) - set(expected_split_ids))
    )
    if (not smoke or bool(split_config.get("fail_on_skipped_split", False))) and skipped:
        raise RevisionDetectionError(
            "Requested evaluation splits were skipped: {}".format(
                "; ".join(
                    "{} ({})".format(item.split_id, item.reason) for item in skipped
                )
            )
        )
    if missing_split_ids or unexpected_split_ids:
        raise RevisionDetectionError(
            "Confirmatory detector split identities differ from the primary contract; "
            "missing={}, unexpected={}.".format(
                missing_split_ids, unexpected_split_ids
            )
        )
    run_metadata["split_contract"] = (
        None
        if expected_split_ids is None
        else {
            "schema_version": "rankcloak-revision-detector-splits-v2",
            "input_scope": "primary_full_detector_corpus_only",
            "expected_split_count": int(len(expected_split_ids)),
            "expected_split_ids": list(expected_split_ids),
            "expected_split_ids_sha256": _sha256_text(
                "\n".join(sorted(expected_split_ids))
            ),
            "dataset_contract": dict(confirmatory_dataset_contract),
            "missing_split_ids": missing_split_ids,
            "unexpected_split_ids": unexpected_split_ids,
        }
    )
    bootstrap_config = dict(config.get("bootstrap", {}))
    bootstrap_resamples = int(bootstrap_config.get("resamples", 2000))
    if smoke:
        bootstrap_resamples = min(
            bootstrap_resamples, int(bootstrap_config.get("smoke_resamples", 100))
        )
    return PreparedDetectorSuite(
        normalized_frame=normalized,
        splits=splits,
        skipped_splits=skipped,
        detector_configs=detectors,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        threshold=float(config.get("decision_threshold", 0.5)),
        smoke=bool(smoke),
        allow_model_downloads=bool(allow_model_downloads),
        run_metadata=run_metadata,
    )


def detector_fit_seed(
    prepared: PreparedDetectorSuite,
    split: DetectorSplit,
    detector_config: Mapping[str, object],
) -> int:
    """Return the stable seed used by the frozen all-at-once implementation."""

    detector_name = str(
        detector_config.get("name", detector_config.get("kind", ""))
    )
    return _stable_seed(prepared.seed, split.split_id, detector_name)


def run_prepared_detector_fit(
    prepared: PreparedDetectorSuite,
    split: DetectorSplit,
    detector_config: Mapping[str, object],
) -> Tuple[dict, List[dict]]:
    """Fit one detector/split and return the exact legacy row schemas."""

    normalized = prepared.normalized_frame
    train = normalized.iloc[list(split.train_indices)].copy()
    test = normalized.iloc[list(split.test_indices)].copy()
    detector_name = str(
        detector_config.get("name", detector_config.get("kind", ""))
    )
    detector_seed = detector_fit_seed(prepared, split, detector_config)
    output = run_configured_detector(
        train,
        test,
        detector_config,
        seed=detector_seed,
        smoke=prepared.smoke,
        allow_model_downloads=prepared.allow_model_downloads,
    )
    if len(output.scores) != len(test):
        raise RevisionDetectionError(
            "Detector {} returned {} scores for {} test rows.".format(
                detector_name, len(output.scores), len(test)
            )
        )
    metric = grouped_bootstrap_detector_metrics(
        test["label"].astype(int).to_numpy(),
        output.scores,
        test["payload_group_id"].astype(str).to_numpy(),
        n_resamples=prepared.bootstrap_resamples,
        seed=_stable_seed(detector_seed, "grouped_bootstrap"),
        threshold=prepared.threshold,
    )
    metric_row = {
        "split_id": split.split_id,
        "regime": split.regime,
        "held_out_column": split.held_out_column,
        "held_out_value": split.held_out_value,
        "detector_name": output.detector_name,
        "requested_kind": output.requested_kind,
        "implementation_kind": output.implementation_kind,
        "implementation_status": output.implementation_status,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_payload_groups": int(train["payload_group_id"].nunique()),
        "purged_train_rows": int(split.purged_train_rows),
        "decision_threshold": prepared.threshold,
        "seed": int(detector_seed),
        "notes": output.notes,
        "model_state_sha256": output.metadata.get("model_state_sha256"),
        "model_state_hash_algorithm": output.metadata.get(
            "model_state_hash_algorithm"
        ),
        "model_artifact_set_sha256": output.metadata.get(
            "model_artifact_set_sha256"
        ),
        "implementation_metadata_json": json.dumps(
            output.metadata, sort_keys=True, default=str
        ),
        **metric,
    }
    prediction_rows: List[dict] = []
    for (_, row), score in zip(test.iterrows(), output.scores):
        prediction_rows.append(
            {
                "split_id": split.split_id,
                "regime": split.regime,
                "held_out_value": split.held_out_value,
                "detector_name": output.detector_name,
                "requested_kind": output.requested_kind,
                "implementation_kind": output.implementation_kind,
                "implementation_status": output.implementation_status,
                "row_id": row["row_id"],
                "payload_group_id": row["payload_group_id"],
                "prompt_template_id": row["prompt_template_id"],
                "model_id": row["model_id"],
                "codec_id": row["codec_id"],
                "label": int(row["label"]),
                "score": float(score),
                "prediction": int(float(score) >= prepared.threshold),
            }
        )
    return metric_row, prediction_rows


def assemble_prepared_detector_result(
    prepared: PreparedDetectorSuite,
    metric_rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]] = (),
) -> DetectorSuiteResult:
    """Assemble a legacy suite result from rows in frozen task order."""

    metric_values = [dict(row) for row in metric_rows]
    prediction_values = [dict(row) for row in predictions]
    failure_values = [dict(row) for row in failures]
    run_metadata = dict(prepared.run_metadata)
    expected_executions = int(
        len(prepared.splits) * len(prepared.detector_configs)
    )
    complete_executions = sum(
        1
        for row in metric_values
        if row["implementation_status"] == "complete"
        and row["implementation_kind"] == row["requested_kind"]
    )
    run_metadata.update(
        {
            "expected_detector_split_executions": expected_executions,
            "complete_detector_split_executions": int(complete_executions),
            "confirmatory_complete": (
                None
                if prepared.smoke
                else bool(
                    not failure_values
                    and not prepared.skipped_splits
                    and complete_executions == expected_executions
                    and expected_executions > 0
                )
            ),
        }
    )
    return DetectorSuiteResult(
        normalized_frame=prepared.normalized_frame,
        splits=prepared.splits,
        skipped_splits=prepared.skipped_splits,
        predictions=pd.DataFrame(prediction_values),
        metrics=pd.DataFrame(metric_values),
        failures=failure_values,
        run_metadata=run_metadata,
    )


def load_detector_config(path: Path) -> dict:
    """Load and minimally validate a JSON revision-detector configuration."""

    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RevisionDetectionError("Detector config does not exist: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise RevisionDetectionError("Detector config is invalid JSON: {}".format(exc)) from exc
    if not isinstance(config, dict):
        raise RevisionDetectionError("Detector config root must be a JSON object.")
    if str(config.get("schema_version", "")) != "rankcloak-revision-detectors-v1":
        raise RevisionDetectionError(
            "Detector config schema_version must be rankcloak-revision-detectors-v1."
        )
    _detector_configs(config)
    return config
