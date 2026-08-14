"""Statistical analysis and deterministic text-quality helpers for revision studies.

This module treats payloads—not segments—as the experimental units.  It reads
only saved CSV/JSONL artifacts and never launches generation or downloads
models.  Surface-quality metrics are diagnostics and are explicitly not a
substitute for blinded human ratings.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 2_026_080_801

PAYLOAD_RECOVERY_OUTCOME = "exact_payload_recovery"
PAYLOAD_RECOVERY_SEMANTICS = (
    "original_serialized_payload_bytes_sha256_v1"
)
PROTOCOL_CONTRACT_REVISION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
PRIMARY_V2_EVIDENCE_STATUS = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
SUPERSEDING_EVIDENCE_PHASES = {
    "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling": {
        "smoke_v3_exploratory",
        "ordinary_llm_control_smoke_v3",
    },
    PRIMARY_V2_EVIDENCE_STATUS: {"primary_v2_confirmatory"},
    "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze": {
        "ablation_v2_confirmatory",
    },
    "secondary_supplementary_multilingual_v2_payload_fidelity_after_manifest_freeze": {
        "multilingual_v2_secondary",
    },
    "confirmatory_supporting_robustness_v2_payload_fidelity_after_manifest_freeze": {
        "robustness_v2_confirmatory_supporting",
    },
}
DIRECT_SUBWORD_PROTOCOL_VARIANTS = frozenset({"direct_subword_calgacus"})
DIRECT_SUBWORD_REPRESENTATIONS = frozenset(
    {"direct_subword", "raw_subword_direct"}
)

PAYLOAD_COLUMN_CANDIDATES = ("payload_name", "payload_id", "payload_group_id")
TEXT_COLUMN_CANDIDATES = (
    "text",
    "full_text",
    "generated_text",
    "message_text",
    "cover_text",
)
PROMPT_TEXT_COLUMN_CANDIDATES = ("prompt_text", "prompt", "source_prompt")
VIEW_COLUMN_CANDIDATES = ("view", "text_view", "span_type")
PARTITION_COLUMN_CANDIDATES = ("partition", "split", "dataset_split")

RECOVERY_GROUP_CANDIDATES = (
    "evidence_status",
    "study_phase",
    "phase",
    "model_id",
    "protocol_variant",
    "prompt_category",
    "payload_class",
    "replay_mode",
    "transformation_id",
    "mitigation_id",
    "protocol_contract_revision",
    "result_schema_revision",
)
INFERENCE_CONDITION_COLUMNS = (
    "replay_mode",
    "transformation_id",
    "mitigation_id",
)
EFFECT_STRATUM_COLUMNS = INFERENCE_CONDITION_COLUMNS + (
    "view",
    "text_view",
    "span_type",
    "evidence_status",
    "study_phase",
    "protocol_contract_revision",
    "result_schema_revision",
)
PRIMARY_EFFECT_CONDITIONS = {
    "replay_mode": "saved_token_ids",
    "transformation_id": "unmodified",
    "mitigation_id": "none",
    "evidence_status": PRIMARY_V2_EVIDENCE_STATUS,
    "study_phase": "primary_v2_confirmatory",
    "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
    "result_schema_revision": RESULT_SCHEMA_REVISION,
}
UNAVAILABLE_RECORD_TYPES = frozenset(
    {
        "condition_unavailable",
        "dependent_unavailable",
        "scientific_unavailable",
        "unavailable",
    }
)
CONTINUOUS_OUTCOME_CANDIDATES = (
    "heldout_evaluator_log_probability",
    "mean_log_probability",
    "effective_payload_rate",
    "payload_bits_per_second",
    "encoding_tokens_per_second",
    "decoding_tokens_per_second",
    "encoding_seconds",
    "decoding_seconds",
    "peak_ram_mib",
    "peak_gpu_memory_mib",
    "cover_tokens_per_payload_byte",
)
SOURCE_NUMERIC_QUALITY_COLUMNS = (
    "mean_log_probability",
    "artifact_count",
    "heldout_evaluator_log_probability",
    "effective_payload_rate",
)
QUALITY_EFFECT_OUTCOMES = (
    "flesch_reading_ease_heuristic",
    "repeated_bigram_fraction",
    "surface_flag_total",
    "tfidf_prompt_similarity",
)

OUTPUT_FILENAMES = {
    "recovery": "recovery_summary.csv",
    "continuous": "continuous_summary.csv",
    "effects": "effect_sizes.csv",
    "quality": "quality_trial_metrics.csv",
    "detectors": "detector_summary.csv",
    "mixed": "mixed_effects_coefficients.csv",
    "mixed_status": "mixed_effects_status.json",
    "integrity": "statistics_integrity_report.json",
    "manifest": "statistics_run_manifest.json",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


class RevisionStatisticsError(ValueError):
    """Raised when saved analysis data violate the revision design."""


class MixedEffectsUnavailable(RuntimeError):
    """Raised when a requested mixed-effects backend cannot be used."""


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Paths emitted by :func:`run_statistics_analysis`."""

    output_dir: str
    files: dict[str, str]
    integrity_report: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_saved_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSONL artifact without type-specific silent fallbacks."""

    source = Path(path)
    if not source.is_file():
        raise RevisionStatisticsError(f"Input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    raise RevisionStatisticsError(
        f"Unsupported saved-artifact format {suffix!r}; expected .csv or .jsonl"
    )


def load_saved_tables(paths: Sequence[str | Path], *, label: str) -> pd.DataFrame:
    """Load and combine saved tables, rejecting path and byte duplicates."""

    if not paths:
        return pd.DataFrame()
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RevisionStatisticsError(f"Repeated {label} input path")
    hashes = [file_sha256(path) for path in resolved]
    duplicates = sorted({value for value in hashes if hashes.count(value) > 1})
    if duplicates:
        raise RevisionStatisticsError(
            f"Byte-identical {label} inputs would double-count observations: "
            + ", ".join(duplicates)
        )
    frames: list[pd.DataFrame] = []
    for path, digest in zip(resolved, hashes):
        frame = read_saved_table(path).copy()
        frame["_source_file"] = str(path)
        frame["_source_sha256"] = digest
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def first_present_column(
    frame: pd.DataFrame, candidates: Sequence[str], *, required: bool = False
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise RevisionStatisticsError(
            "Missing required column; expected one of " + ", ".join(candidates)
        )
    return None


def _require_nonempty(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RevisionStatisticsError(f"{label} missing columns: {', '.join(missing)}")
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise RevisionStatisticsError(f"{label}.{column} contains missing values")


def _binary_series(values: pd.Series, *, label: str) -> pd.Series:
    mapping = {
        True: 1,
        False: 0,
        1: 1,
        0: 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    }
    normalized = values.map(
        lambda value: mapping.get(value, mapping.get(str(value).strip().lower()))
        if not pd.isna(value)
        else np.nan
    )
    if normalized.isna().any():
        examples = values[normalized.isna()].astype(str).head(3).tolist()
        raise RevisionStatisticsError(f"{label} is not binary; examples: {examples}")
    return normalized.astype(int)


def _direct_subword_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify payload-bearing direct-subword rows without matching controls.

    ``protocol_variant`` is authoritative when present. The representation and
    codec fallbacks support narrow legacy tables that omit that column; they do
    not match rows explicitly labelled as another protocol (for example an
    ordinary-LLM control that inherited payload metadata). Held-out evaluator
    features retain the source protocol solely as quality provenance; they are
    not recovery observations and therefore cannot satisfy or violate the
    payload-recovery contract.
    """

    mask = pd.Series(False, index=frame.index, dtype=bool)
    protocol_present = pd.Series(False, index=frame.index, dtype=bool)
    if "protocol_variant" in frame.columns:
        protocol = frame["protocol_variant"].fillna("").astype(str).str.strip()
        protocol_present = protocol.ne("")
        mask |= protocol.isin(DIRECT_SUBWORD_PROTOCOL_VARIANTS)

    unresolved = ~protocol_present
    representation_present = pd.Series(False, index=frame.index, dtype=bool)
    if "representation_name" in frame.columns:
        representation = (
            frame["representation_name"].fillna("").astype(str).str.strip()
        )
        representation_present = representation.ne("")
        mask |= unresolved & representation.isin(DIRECT_SUBWORD_REPRESENTATIONS)

    if "codec_id" in frame.columns:
        codec = frame["codec_id"].fillna("").astype(str).str.strip()
        mask |= (
            unresolved
            & ~representation_present
            & codec.isin(DIRECT_SUBWORD_REPRESENTATIONS)
        )
    if "record_type" in frame.columns:
        record_type = frame["record_type"].fillna("").astype(str).str.strip()
        mask &= ~record_type.eq("heldout_evaluator_feature")
    return mask


def _validate_payload_fidelity_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed where exact rank replay could mask payload corruption.

    Direct-subword rank equality is only a replay diagnostic. Its scientific
    recovery outcome is equality of the original serialized payload bytes,
    represented by ``exact_payload_recovery``. ``exact_recovery`` is retained
    solely as a compatibility alias and must agree with that outcome.
    """

    result = frame.copy()
    direct_mask = _direct_subword_mask(result)
    direct_count = int(direct_mask.sum())
    contract = {
        "contract_version": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "semantics": PAYLOAD_RECOVERY_SEMANTICS,
        "primary_outcome": PAYLOAD_RECOVERY_OUTCOME,
        "compatibility_alias": "exact_recovery",
        "alias_equality_validated": True,
        "exact_rank_replay_role": "diagnostic_only",
        "direct_rows": direct_count,
        "direct_rows_contract_verified": 0,
    }

    if "exact_recovery" in result.columns:
        supplied_alias = result["exact_recovery"].notna()
        if supplied_alias.any():
            result.loc[supplied_alias, "exact_recovery"] = _binary_series(
                result.loc[supplied_alias, "exact_recovery"],
                label="exact_recovery",
            )
        if direct_count and result.loc[direct_mask, "exact_recovery"].isna().any():
            raise RevisionStatisticsError(
                "Direct-subword payload-fidelity contract has missing exact_recovery"
            )
    elif direct_count:
        raise RevisionStatisticsError(
            "Direct-subword trial results lack the exact_recovery "
            "compatibility alias"
        )

    required_direct_columns = (
        "protocol_contract_revision",
        "result_schema_revision",
        "exact_rank_replay",
        PAYLOAD_RECOVERY_OUTCOME,
        "recovery_outcome_semantics",
    )
    missing = [
        column for column in required_direct_columns if column not in result
    ]
    if direct_count and missing:
        raise RevisionStatisticsError(
            "Direct-subword payload-fidelity contract missing columns: "
            + ", ".join(missing)
        )

    for column in ("exact_rank_replay", PAYLOAD_RECOVERY_OUTCOME):
        if column not in result.columns:
            continue
        present = result[column].notna()
        if present.any():
            result.loc[present, column] = _binary_series(
                result.loc[present, column], label=column
            )

    preprocessed = "preprocess_schema_version" in result.columns
    if preprocessed:
        required_preprocessed = (
            "protocol_contract_revision",
            "result_schema_revision",
            "evidence_status",
            "study_phase",
            "exact_rank_replay",
            PAYLOAD_RECOVERY_OUTCOME,
            "recovery_outcome_semantics",
        )
        missing_preprocessed = [
            column for column in required_preprocessed if column not in result
        ]
        if missing_preprocessed:
            raise RevisionStatisticsError(
                "Payload-aware preprocessed trials missing columns: "
                + ", ".join(missing_preprocessed)
            )
        if not result["protocol_contract_revision"].astype(str).eq(
            PROTOCOL_CONTRACT_REVISION
        ).all():
            raise RevisionStatisticsError(
                "Preprocessed trials are not payload_fidelity_v2 evidence"
            )
        if not result["result_schema_revision"].astype(str).eq(
            RESULT_SCHEMA_REVISION
        ).all():
            raise RevisionStatisticsError(
                "Preprocessed trials are not payload_aware_result_v2 evidence"
            )
        for index, row in result.iterrows():
            evidence = str(row["evidence_status"])
            phase = str(row["study_phase"])
            if phase not in SUPERSEDING_EVIDENCE_PHASES.get(evidence, set()):
                raise RevisionStatisticsError(
                    "Preprocessed trial row {} has a legacy or mismatched "
                    "evidence/study-phase label".format(index)
                )

    for column, expected_value in (
        ("protocol_contract_revision", PROTOCOL_CONTRACT_REVISION),
        ("result_schema_revision", RESULT_SCHEMA_REVISION),
    ):
        if column in result.columns:
            supplied = result[column].notna()
            if supplied.any() and not result.loc[supplied, column].astype(str).eq(
                expected_value
            ).all():
                raise RevisionStatisticsError(
                    f"{column} must equal {expected_value!r}"
                )

    if direct_count:
        for column in required_direct_columns:
            values = result.loc[direct_mask, column]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise RevisionStatisticsError(
                    f"Direct-subword payload-fidelity contract has missing {column}"
                )

        semantics = (
            result.loc[direct_mask, "recovery_outcome_semantics"]
            .astype(str)
            .str.strip()
        )
        if not semantics.eq(PAYLOAD_RECOVERY_SEMANTICS).all():
            observed = sorted(semantics.unique().tolist())[:3]
            raise RevisionStatisticsError(
                "Direct-subword recovery_outcome_semantics must equal "
                f"{PAYLOAD_RECOVERY_SEMANTICS!r}; observed {observed!r}"
            )

        alias = result.loc[direct_mask, "exact_recovery"].astype(int)
        payload = result.loc[direct_mask, PAYLOAD_RECOVERY_OUTCOME].astype(int)
        mismatch = alias.ne(payload)
        if mismatch.any():
            examples = result.loc[
                alias.index[mismatch],
                [
                    column
                    for column in (
                        "trial_id",
                        "model_id",
                        "replay_mode",
                        "exact_recovery",
                        PAYLOAD_RECOVERY_OUTCOME,
                        "exact_rank_replay",
                    )
                    if column in result.columns
                ],
            ].head(3)
            raise RevisionStatisticsError(
                "exact_recovery compatibility alias differs from "
                f"{PAYLOAD_RECOVERY_OUTCOME}: "
                + repr(examples.to_dict(orient="records"))
            )
        contract["direct_rows_contract_verified"] = direct_count

    # If a non-direct producer supplies the new payload outcome too, never
    # permit the compatibility alias to disagree silently.
    if PAYLOAD_RECOVERY_OUTCOME in result.columns and "exact_recovery" in result:
        supplied = result[PAYLOAD_RECOVERY_OUTCOME].notna()
        if supplied.any():
            alias = result.loc[supplied, "exact_recovery"].astype(int)
            payload = result.loc[supplied, PAYLOAD_RECOVERY_OUTCOME].astype(int)
            if alias.ne(payload).any():
                raise RevisionStatisticsError(
                    "exact_recovery compatibility alias differs from "
                    f"{PAYLOAD_RECOVERY_OUTCOME}"
                )

    result.attrs["payload_fidelity_contract"] = contract
    return result


def _unavailable_estimand_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify rows explicitly declared unavailable for scientific estimands.

    The preprocessing contract writes unavailable work to a separate table, but
    this fail-safe prevents accidental counting if tables are concatenated by a
    downstream caller. It intentionally uses explicit schema fields rather than
    inferring availability from a missing outcome.
    """

    mask = pd.Series(False, index=frame.index, dtype=bool)
    if "excluded_from_estimands" in frame.columns:
        values = frame["excluded_from_estimands"]
        nonmissing = values.notna()
        normalized = values.astype(str).str.strip().str.lower()
        recognized = normalized.isin(
            {"true", "false", "1", "0", "yes", "no"}
        )
        if (nonmissing & ~recognized).any():
            examples = values[nonmissing & ~recognized].astype(str).head(3).tolist()
            raise RevisionStatisticsError(
                "excluded_from_estimands is not boolean; examples: "
                + repr(examples)
            )
        mask |= nonmissing & normalized.isin({"true", "1", "yes"})

    if "record_type" in frame.columns:
        record_types = (
            frame["record_type"].fillna("").astype(str).str.strip().str.lower()
        )
        mask |= record_types.isin(UNAVAILABLE_RECORD_TYPES)

    for column in ("availability_status", "condition_status", "record_status"):
        if column not in frame.columns:
            continue
        statuses = frame[column].fillna("").astype(str).str.strip().str.lower()
        mask |= statuses.isin(UNAVAILABLE_RECORD_TYPES)
    return mask


def _exclude_unavailable_estimand_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return estimand-eligible rows and retain an auditable exclusion count."""

    if frame.empty:
        result = frame.copy()
        result.attrs["excluded_unavailable_rows"] = 0
        return result
    mask = _unavailable_estimand_mask(frame)
    result = frame.loc[~mask].copy()
    result.attrs["excluded_unavailable_rows"] = int(mask.sum())
    return result


def _condition_levels(frame: pd.DataFrame, column: str) -> list[str]:
    """Return observed condition levels, representing mixed missingness."""

    if column not in frame.columns or frame.empty:
        return []
    values = frame[column]
    levels = sorted(values.dropna().astype(str).unique().tolist())
    if values.isna().any() and levels:
        levels.append("<missing>")
    return levels


def _assert_effect_condition_not_pooled(
    frame: pd.DataFrame, *, factor: str, outcome: str
) -> dict[str, str]:
    """Reject implicit averaging across replay or other estimand strata."""

    relevant = frame.copy()
    relevant[outcome] = pd.to_numeric(relevant[outcome], errors="coerce")
    relevant = relevant[np.isfinite(relevant[outcome])]
    scopes: dict[str, str] = {}
    for column in EFFECT_STRATUM_COLUMNS:
        levels = _condition_levels(relevant, column)
        if len(levels) > 1 and factor != column:
            raise RevisionStatisticsError(
                f"Pairwise {outcome} effects would pool {column} levels "
                f"{levels}; filter to one level or compare {column} directly"
            )
        if len(levels) == 1:
            scopes[f"{column}_scope"] = levels[0]
    return scopes


def _condition_identity_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["trial_id"]
    for candidate in (
        "replay_mode",
        "transformation_id",
        "mitigation_id",
        "replicate_id",
        "text_view",
        "view",
    ):
        if candidate in frame.columns:
            columns.append(candidate)
    return columns


def _assert_metadata_consistency(
    frame: pd.DataFrame,
    *,
    key: str,
    columns: Sequence[str],
    label: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame.groupby(key, dropna=False)[column].nunique(dropna=False)
        bad = counts[counts > 1]
        if not bad.empty:
            raise RevisionStatisticsError(
                f"{label}: {column} changes within {key}; examples: "
                + ", ".join(map(str, bad.index[:3]))
            )


def assert_partition_group_disjoint(
    frame: pd.DataFrame,
    *,
    group_column: str,
    partition_column: str,
) -> None:
    """Reject payload groups observed in more than one data partition."""

    _require_nonempty(
        frame, [group_column, partition_column], label="partitioned observations"
    )
    counts = frame.groupby(group_column, dropna=False)[partition_column].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise RevisionStatisticsError(
            "Payload-group leakage across partitions: "
            + ", ".join(map(str, leaked.index[:5]))
        )


def validate_trial_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate recovery trials and return data plus the payload-unit column."""

    if frame.empty:
        return frame.copy(), "payload_name"
    payload_column = first_present_column(
        frame, PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    result = _exclude_unavailable_estimand_rows(frame)
    excluded_unavailable_rows = int(
        result.attrs.get("excluded_unavailable_rows", 0)
    )
    if result.empty:
        result.attrs["excluded_unavailable_rows"] = excluded_unavailable_rows
        return result, payload_column
    _require_nonempty(result, ["trial_id", payload_column], label="trial results")
    _assert_metadata_consistency(
        result,
        key="trial_id",
        columns=(
            payload_column,
            "model_id",
            "protocol_variant",
            "prompt_id",
            "prompt_category",
            "payload_class",
        ),
        label="trial results",
    )
    identity = _condition_identity_columns(result)
    if result.duplicated(identity, keep=False).any():
        examples = result.loc[
            result.duplicated(identity, keep=False), identity
        ].head(3)
        raise RevisionStatisticsError(
            "Duplicate trial-condition rows would inflate sample size: "
            + examples.to_dict(orient="records").__repr__()
        )
    result = _validate_payload_fidelity_contract(result)
    result.attrs["excluded_unavailable_rows"] = excluded_unavailable_rows
    return result, payload_column


def validate_runtime_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        return frame.copy(), "payload_name"
    payload_column = first_present_column(
        frame, PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    _require_nonempty(frame, ["trial_id", payload_column], label="runtime results")
    identity = ["trial_id"]
    for column in ("hardware_id", "hardware_hash", "replicate_id"):
        if column in frame.columns:
            identity.append(column)
    if frame.duplicated(identity, keep=False).any():
        raise RevisionStatisticsError(
            f"Duplicate runtime analysis units for identity {identity}"
        )
    return frame.copy(), payload_column


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_feature_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate nested feature rows without treating segments as independent."""

    if frame.empty:
        return frame.copy(), "payload_name"
    payload_column = first_present_column(
        frame, PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    _require_nonempty(frame, ["trial_id", payload_column], label="feature rows")
    _assert_metadata_consistency(
        frame,
        key="trial_id",
        columns=(payload_column, "model_id", "protocol_variant", "prompt_id"),
        label="feature rows",
    )
    view_column = first_present_column(frame, VIEW_COLUMN_CANDIDATES)
    identity = ["trial_id"]
    if view_column:
        identity.append(view_column)
    if "segment_index" in frame.columns:
        identity.append("segment_index")
    if frame.duplicated(identity, keep=False).any():
        raise RevisionStatisticsError(
            f"Duplicate nested feature rows for identity {identity}"
        )

    result = frame.copy()
    text_column = first_present_column(result, TEXT_COLUMN_CANDIDATES)
    if text_column:
        hashes = result[text_column].fillna("").astype(str).map(_text_sha256)
        audit = pd.DataFrame(
            {"hash": hashes, "payload": result[payload_column].astype(str)}
        )
        duplicated_across_payloads = audit.groupby("hash")["payload"].nunique()
        bad = duplicated_across_payloads[duplicated_across_payloads > 1]
        if not bad.empty:
            raise RevisionStatisticsError(
                "Identical cover text assigned to multiple payload groups; hashes: "
                + ", ".join(bad.index[:3])
            )
    return result, payload_column


def validate_detector_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        return frame.copy(), "payload_name"
    payload_column = first_present_column(
        frame, ("payload_group_id",) + PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    _require_nonempty(frame, [payload_column], label="detector rows")
    partition_column = first_present_column(frame, PARTITION_COLUMN_CANDIDATES)
    if partition_column:
        assert_partition_group_disjoint(
            frame,
            group_column=payload_column,
            partition_column=partition_column,
        )
    identity = [
        column
        for column in (
            "detector",
            "detector_name",
            "regime",
            "split_id",
            "row_id",
            "trial_id",
            payload_column,
        )
        if column in frame.columns
    ]
    if len(identity) > 1 and frame.duplicated(identity, keep=False).any():
        raise RevisionStatisticsError(
            f"Duplicate detector prediction rows for identity {identity}"
        )
    return frame.copy(), payload_column


def _syllable_count(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    if not normalized:
        return 0
    groups = len(_VOWEL_GROUP_RE.findall(normalized))
    if normalized.endswith("e") and groups > 1 and not normalized.endswith(("le", "ye")):
        groups -= 1
    return max(1, groups)


def _maximum_word_run(words: Sequence[str]) -> int:
    if not words:
        return 0
    maximum = current = 1
    for previous, word in zip(words, words[1:]):
        if word == previous:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 1
    return maximum


def _repeated_ngram_fraction(words: Sequence[str], size: int) -> float:
    ngrams = [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def _prompt_similarity(text: str, prompt: str) -> float:
    if not text.strip() or not prompt.strip():
        return float("nan")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RevisionStatisticsError(
            "TF-IDF prompt similarity requires scikit-learn"
        ) from exc
    matrix = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    ).fit_transform([prompt, text])
    return float((matrix[0] @ matrix[1].T).toarray()[0, 0])


def automated_text_quality_metrics(text: str, prompt: str = "") -> dict[str, Any]:
    """Compute deterministic, offline surface diagnostics for one cover text.

    Flesch measures use a documented English syllable heuristic; they are not
    validated for multilingual text.  Grammar-like counts flag surface forms
    only and are not a grammar checker or substitutes for human judgements.
    """

    text = "" if text is None else str(text)
    prompt = "" if prompt is None else str(prompt)
    raw_words = _WORD_RE.findall(text)
    words = [word.lower() for word in raw_words]
    sentences = [part.strip() for part in _SENTENCE_RE.findall(text) if part.strip()]
    word_count = len(words)
    sentence_count = max(1, len(sentences)) if word_count else 0
    syllables = sum(_syllable_count(word) for word in raw_words)
    characters = sum(len(re.sub(r"[^A-Za-z0-9]", "", word)) for word in raw_words)

    if word_count:
        sentence_denominator = max(1, sentence_count)
        words_per_sentence = word_count / sentence_denominator
        syllables_per_word = syllables / word_count
        flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        coleman_liau = (
            0.0588 * (characters / word_count * 100)
            - 0.296 * (sentence_denominator / word_count * 100)
            - 15.8
        )
    else:
        flesch = fk_grade = coleman_liau = float("nan")

    sentence_keys = [re.sub(r"\s+", " ", sentence.lower()).strip() for sentence in sentences]
    duplicate_sentences = len(sentence_keys) - len(set(sentence_keys))
    opening_chars = [sentence.lstrip()[0] for sentence in sentences if sentence.lstrip()]
    lowercase_starts = sum(character.islower() for character in opening_chars)
    long_sentences = sum(len(_WORD_RE.findall(sentence)) > 40 for sentence in sentences)
    terminal_missing = int(bool(text.strip()) and text.rstrip()[-1] not in ".!?")
    unmatched_brackets = sum(abs(text.count(left) - text.count(right)) for left, right in "() [] {}".split())
    quote_imbalance = text.count('"') % 2
    repeated_punctuation = len(re.findall(r"([!?.,;:])\1{1,}", text))
    whitespace_flags = len(re.findall(r"[ \t]{2,}|\s+[,.!?;:]", text))
    artifact_fragments = len(
        re.findall(r"(?:[A-Fa-f0-9]{24,}|[A-Za-z0-9+/]{24,}={0,2})", text)
    )

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "character_count": len(text),
        "flesch_reading_ease_heuristic": flesch,
        "flesch_kincaid_grade_heuristic": fk_grade,
        "coleman_liau_index": coleman_liau,
        "unique_word_fraction": len(set(words)) / word_count if word_count else float("nan"),
        "repeated_bigram_fraction": _repeated_ngram_fraction(words, 2),
        "repeated_trigram_fraction": _repeated_ngram_fraction(words, 3),
        "maximum_identical_word_run": _maximum_word_run(words),
        "duplicate_sentence_count": duplicate_sentences,
        "unmatched_bracket_count": unmatched_brackets,
        "unmatched_double_quote_count": quote_imbalance,
        "repeated_punctuation_count": repeated_punctuation,
        "whitespace_surface_flag_count": whitespace_flags,
        "lowercase_sentence_start_count": lowercase_starts,
        "long_sentence_count_gt40": long_sentences,
        "missing_terminal_punctuation": terminal_missing,
        "artifact_like_fragment_count": artifact_fragments,
        "surface_flag_total": (
            unmatched_brackets
            + quote_imbalance
            + repeated_punctuation
            + whitespace_flags
            + lowercase_starts
            + long_sentences
            + terminal_missing
            + artifact_fragments
        ),
        "tfidf_prompt_similarity": _prompt_similarity(text, prompt),
        "human_rating_substitute": False,
        "readability_scope": "english_surface_heuristic",
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise RevisionStatisticsError("Wilson interval requires 0 <= successes <= total")
    if not 0.0 < confidence_level < 1.0:
        raise RevisionStatisticsError("confidence_level must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return max(0.0, (centre - radius) / denominator), min(
        1.0, (centre + radius) / denominator
    )


def exact_binomial_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Two-sided Clopper--Pearson exact binomial interval."""

    if total <= 0 or successes < 0 or successes > total:
        raise RevisionStatisticsError(
            "Exact binomial interval requires 0 <= successes <= total"
        )
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RevisionStatisticsError("Exact intervals require scipy") from exc
    alpha = 1.0 - confidence_level
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return lower, upper


def grouped_payload_bootstrap_ci(
    values: Sequence[float],
    payload_groups: Sequence[object],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Mean and percentile CI after equal-weight payload aggregation/resampling."""

    if len(values) != len(payload_groups) or not values:
        raise RevisionStatisticsError(
            "values and payload_groups must align and be non-empty"
        )
    data = pd.DataFrame(
        {
            "value": pd.to_numeric(pd.Series(values), errors="coerce"),
            "payload": list(map(str, payload_groups)),
        }
    ).dropna(subset=["value"])
    if data.empty:
        raise RevisionStatisticsError("No finite values remain for grouped bootstrap")
    payload_values = data.groupby("payload", sort=True)["value"].mean()
    finite = payload_values[np.isfinite(payload_values.to_numpy(dtype=float))]
    if finite.empty:
        raise RevisionStatisticsError("No finite payload-level values remain")
    point = float(finite.mean())
    result: dict[str, float | int] = {
        "mean": point,
        "n_payloads": int(len(finite)),
        "bootstrap_resamples_requested": int(n_resamples),
    }
    if len(finite) < 2 or n_resamples <= 0:
        result.update(
            {
                "ci_low": point,
                "ci_high": point,
                "bootstrap_resamples_valid": 0,
            }
        )
        return result
    rng = np.random.default_rng(int(seed))
    array = finite.to_numpy(dtype=float)
    samples = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        samples[index] = float(
            rng.choice(array, size=len(array), replace=True).mean()
        )
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    result.update(
        {
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_resamples_valid": int(len(samples)),
        }
    )
    return result


def adjust_pvalues(
    pvalues: Sequence[float | None], method: str
) -> list[float | None]:
    """Return Holm family-wise or Benjamini--Hochberg adjusted p-values."""

    normalized: list[float | None] = []
    for value in pvalues:
        if value is None or not math.isfinite(float(value)):
            normalized.append(None)
        elif not 0.0 <= float(value) <= 1.0:
            raise RevisionStatisticsError(f"Invalid p-value: {value}")
        else:
            normalized.append(float(value))
    valid = [
        (index, value)
        for index, value in enumerate(normalized)
        if value is not None
    ]
    if not valid:
        return normalized
    ordered = sorted(valid, key=lambda item: float(item[1]))
    count = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    if method.lower() == "holm":
        running = 0.0
        for rank, (index, value) in enumerate(ordered):
            candidate = (count - rank) * float(value)
            running = max(running, candidate)
            adjusted_by_index[index] = min(1.0, running)
    elif method.lower() in {"bh", "fdr_bh", "benjamini-hochberg"}:
        running = 1.0
        for reverse_rank in range(count - 1, -1, -1):
            index, value = ordered[reverse_rank]
            candidate = float(value) * count / (reverse_rank + 1)
            running = min(running, candidate)
            adjusted_by_index[index] = min(1.0, running)
    else:
        raise RevisionStatisticsError(
            f"Unknown p-value adjustment method: {method}"
        )
    return [adjusted_by_index.get(index) for index in range(len(normalized))]


def _group_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for column in candidates:
        if column in frame.columns and frame[column].notna().any():
            columns.append(column)
    return columns


def _iter_groups(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    if not group_columns:
        yield tuple(), frame
        return
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for keys, group in frame.groupby(grouper, dropna=False, sort=True):
        if len(group_columns) == 1:
            keys = (keys,)
        yield tuple(keys), group


def summarize_recovery(
    trials: pd.DataFrame,
    *,
    payload_column: str | None = None,
    group_columns: Sequence[str] | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Summarize exact recovery with one independent payload per condition cell."""

    if trials.empty or "exact_recovery" not in trials.columns:
        return pd.DataFrame()
    validated, detected_payload_column = validate_trial_results(trials)
    payload_column = payload_column or detected_payload_column
    groups = (
        list(group_columns)
        if group_columns is not None
        else _group_columns(validated, RECOVERY_GROUP_CANDIDATES)
    )
    # A caller may request a coarse summary, but replay/transformation strata
    # remain part of the scientific condition and must never be averaged away.
    for column in INFERENCE_CONDITION_COLUMNS:
        if _condition_levels(validated, column) and column not in groups:
            groups.append(column)
    rows: list[dict[str, Any]] = []
    for keys, cell in _iter_groups(validated, groups):
        counts = cell.groupby(payload_column, dropna=False)["exact_recovery"].size()
        if (counts > 1).any():
            raise RevisionStatisticsError(
                "Repeated payload observations within a recovery condition; "
                "stratify replay/transformation/replicate conditions before inference"
            )
        successes = int(cell["exact_recovery"].sum())
        total = int(cell[payload_column].nunique())
        rank_replay_n = (
            int(cell["exact_rank_replay"].notna().sum())
            if "exact_rank_replay" in cell.columns
            else 0
        )
        rank_replay_successes = (
            int(pd.to_numeric(cell["exact_rank_replay"], errors="coerce").sum())
            if rank_replay_n
            else 0
        )
        wilson_low, wilson_high = wilson_interval(
            successes, total, confidence_level=confidence_level
        )
        exact_low, exact_high = exact_binomial_interval(
            successes, total, confidence_level=confidence_level
        )
        row = dict(zip(groups, keys))
        row.update(
            {
                "analysis_unit": "payload",
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "recovery_outcome": PAYLOAD_RECOVERY_OUTCOME,
                "recovery_outcome_semantics": PAYLOAD_RECOVERY_SEMANTICS,
                "exact_recovery_compatibility_alias": True,
                "n_payloads": total,
                "payload_recovery_successes": successes,
                "exact_payload_recovery_rate": successes / total,
                "successes": successes,
                "exact_recovery_rate": successes / total,
                "rank_replay_n": rank_replay_n,
                "rank_replay_successes": rank_replay_successes,
                "exact_rank_replay_rate": (
                    rank_replay_successes / rank_replay_n
                    if rank_replay_n
                    else float("nan")
                ),
                "rank_replay_diagnostic_only": True,
                "wilson_ci_low": wilson_low,
                "wilson_ci_high": wilson_high,
                "exact_ci_low": exact_low,
                "exact_ci_high": exact_high,
                "confidence_level": confidence_level,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _metadata_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return None
    values = group[column].dropna().unique()
    if len(values) > 1:
        raise RevisionStatisticsError(
            f"{column} changes within one nested trial/view feature group"
        )
    return values[0] if len(values) else None


def build_trial_quality_table(features: pd.DataFrame) -> pd.DataFrame:
    """Collapse nested segments to trial/view texts, then compute diagnostics."""

    if features.empty:
        return pd.DataFrame()
    validated, payload_column = validate_feature_rows(features)
    view_column = first_present_column(validated, VIEW_COLUMN_CANDIDATES)
    group_columns = ["trial_id"] + ([view_column] if view_column else [])
    text_column = first_present_column(validated, TEXT_COLUMN_CANDIDATES)
    prompt_column = first_present_column(
        validated, PROMPT_TEXT_COLUMN_CANDIDATES
    )
    rows: list[dict[str, Any]] = []
    for keys, group in _iter_groups(validated, group_columns):
        ordered = (
            group.sort_values("segment_index", kind="stable")
            if "segment_index" in group
            else group
        )
        if text_column:
            text_parts = ordered[text_column].fillna("").astype(str).tolist()
            separator = "\n\n" if "segment_index" in ordered.columns else ""
            text = separator.join(text_parts)
        else:
            text = ""
        prompt = ""
        if prompt_column:
            prompt_values = ordered[prompt_column].dropna().astype(str).unique()
            if len(prompt_values) > 1:
                raise RevisionStatisticsError(
                    "Prompt text changes within a trial/view"
                )
            prompt = prompt_values[0] if len(prompt_values) else ""
        row = dict(zip(group_columns, keys))
        if view_column:
            row["view"] = row[view_column]
        for column in (
            payload_column,
            "model_id",
            "protocol_variant",
            "prompt_id",
            "prompt_category",
            "payload_class",
            "language",
            "evidence_status",
            "study_phase",
            "protocol_contract_revision",
            "result_schema_revision",
        ):
            if column in ordered.columns:
                row[column] = _metadata_value(ordered, column)
        row.update(
            {
                "analysis_unit": "payload_trial_view",
                "nested_segment_count": int(len(ordered)),
                "text_available": bool(text_column and text),
                "text_sha256": _text_sha256(text) if text else None,
                **automated_text_quality_metrics(text, prompt),
            }
        )
        for column in SOURCE_NUMERIC_QUALITY_COLUMNS:
            if column in ordered.columns:
                numeric_all = pd.to_numeric(ordered[column], errors="coerce")
                numeric = numeric_all.dropna()
                if (
                    column == "mean_log_probability"
                    and len(ordered) > 1
                    and not numeric.empty
                ):
                    if len(numeric) != len(ordered):
                        raise RevisionStatisticsError(
                            "mean_log_probability is only partially recorded within one "
                            "nested trial/view"
                        )
                    if "token_count" not in ordered.columns:
                        raise RevisionStatisticsError(
                            "Nested mean_log_probability requires token_count weights"
                        )
                    weights = pd.to_numeric(ordered["token_count"], errors="coerce")
                    if weights.isna().any() or (weights <= 0).any():
                        raise RevisionStatisticsError(
                            "Nested mean_log_probability has missing or nonpositive "
                            "token_count weights"
                        )
                    row[f"source_{column}"] = float(
                        np.average(numeric_all.to_numpy(), weights=weights.to_numpy())
                    )
                else:
                    row[f"source_{column}"] = (
                        float(numeric.mean()) if not numeric.empty else np.nan
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_continuous_outcomes(
    frame: pd.DataFrame,
    *,
    outcomes: Sequence[str] | None = None,
    payload_column: str | None = None,
    group_columns: Sequence[str] | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Payload-grouped bootstrap summaries for continuous saved outcomes."""

    if frame.empty:
        return pd.DataFrame()
    payload_column = payload_column or first_present_column(
        frame, PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    selected_outcomes = (
        list(outcomes)
        if outcomes is not None
        else [
            column
            for column in itertools.chain(
                CONTINUOUS_OUTCOME_CANDIDATES,
                (
                    "flesch_reading_ease_heuristic",
                    "flesch_kincaid_grade_heuristic",
                    "coleman_liau_index",
                    "unique_word_fraction",
                    "repeated_bigram_fraction",
                    "surface_flag_total",
                    "tfidf_prompt_similarity",
                ),
            )
            if column in frame.columns
        ]
    )
    groups = (
        list(group_columns)
        if group_columns is not None
        else _group_columns(
            frame,
            RECOVERY_GROUP_CANDIDATES
            + ("view", "text_view", "span_type", "hardware_id"),
        )
    )
    rows: list[dict[str, Any]] = []
    for outcome in selected_outcomes:
        numeric_frame = frame.copy()
        numeric_frame[outcome] = pd.to_numeric(
            numeric_frame[outcome], errors="coerce"
        )
        numeric_frame = numeric_frame[
            np.isfinite(numeric_frame[outcome])
        ]
        for keys, cell in _iter_groups(numeric_frame, groups):
            if cell.empty:
                continue
            summary = grouped_payload_bootstrap_ci(
                cell[outcome].astype(float).tolist(),
                cell[payload_column].astype(str).tolist(),
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed,
            )
            payload_means = cell.groupby(payload_column)[outcome].mean()
            row = dict(zip(groups, keys))
            row.update(
                {
                    "outcome": outcome,
                    "analysis_unit": "payload",
                    "mean": summary["mean"],
                    "standard_deviation": (
                        float(payload_means.std(ddof=1))
                        if len(payload_means) > 1
                        else np.nan
                    ),
                    "median": float(payload_means.median()),
                    **{
                        key: value
                        for key, value in summary.items()
                        if key != "mean"
                    },
                    "confidence_level": confidence_level,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _stable_seed(seed: int, *parts: object) -> int:
    material = "||".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (
        2**32 - 1
    )


def _payload_level_condition_values(
    frame: pd.DataFrame,
    *,
    payload_column: str,
    factor: str,
    outcome: str,
) -> pd.DataFrame:
    data = frame[[payload_column, factor, outcome]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data[np.isfinite(data[outcome])]
    if data.empty:
        return data
    return (
        data.groupby([payload_column, factor], dropna=False, sort=True)[outcome]
        .mean()
        .reset_index()
    )


def _bootstrap_mean_difference(
    first: pd.Series,
    second: pd.Series,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    paired: bool,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    if paired:
        shared = sorted(set(first.index) & set(second.index))
        differences = (
            first.loc[shared].to_numpy(dtype=float)
            - second.loc[shared].to_numpy(dtype=float)
        )
        if len(differences) < 2:
            point = float(differences.mean()) if len(differences) else float("nan")
            return point, point
        samples = np.asarray(
            [
                rng.choice(differences, size=len(differences), replace=True).mean()
                for _ in range(int(n_resamples))
            ],
            dtype=float,
        )
    else:
        first_values = first.to_numpy(dtype=float)
        second_values = second.to_numpy(dtype=float)
        if len(first_values) < 2 or len(second_values) < 2:
            point = float(first_values.mean() - second_values.mean())
            return point, point
        samples = np.asarray(
            [
                rng.choice(
                    first_values, size=len(first_values), replace=True
                ).mean()
                - rng.choice(
                    second_values, size=len(second_values), replace=True
                ).mean()
                for _ in range(int(n_resamples))
            ],
            dtype=float,
        )
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _bootstrap_partially_overlapping_mean_difference(
    first: pd.Series,
    second: pd.Series,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, int]:
    """Cluster-bootstrap a descriptive contrast over the payload-ID union.

    Shared payload IDs are resampled together, while condition-specific payloads
    retain their observed missingness pattern. This supplies an exploratory
    interval without pretending that partially overlapping samples are either
    fully paired or independent.
    """

    first_values = {str(key): float(value) for key, value in first.items()}
    second_values = {str(key): float(value) for key, value in second.items()}
    payload_ids = np.asarray(
        sorted(set(first_values) | set(second_values)), dtype=object
    )
    point = float(
        np.mean(list(first_values.values()))
        - np.mean(list(second_values.values()))
    )
    if len(payload_ids) < 2 or n_resamples <= 0:
        return point, point, 0

    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        selected = rng.choice(payload_ids, size=len(payload_ids), replace=True)
        sampled_first = [
            first_values[value] for value in selected if value in first_values
        ]
        sampled_second = [
            second_values[value] for value in selected if value in second_values
        ]
        if sampled_first and sampled_second:
            samples.append(float(np.mean(sampled_first) - np.mean(sampled_second)))
    if not samples:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha])
    return float(low), float(high), len(samples)


def _hedges_g(
    first: np.ndarray, second: np.ndarray, *, paired: bool
) -> float:
    if paired:
        differences = first - second
        standard_deviation = (
            float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        )
        uncorrected = (
            float(differences.mean() / standard_deviation)
            if standard_deviation > 0
            else float("nan")
        )
        degrees_freedom = len(differences) - 1
    else:
        degrees_freedom = len(first) + len(second) - 2
        if degrees_freedom <= 0:
            return float("nan")
        pooled_variance = (
            (len(first) - 1) * float(first.var(ddof=1))
            + (len(second) - 1) * float(second.var(ddof=1))
        ) / degrees_freedom
        uncorrected = (
            float((first.mean() - second.mean()) / math.sqrt(pooled_variance))
            if pooled_variance > 0
            else float("nan")
        )
    correction = (
        1.0 - 3.0 / (4.0 * degrees_freedom - 1.0)
        if degrees_freedom > 1
        else 1.0
    )
    return correction * uncorrected


def pairwise_effect_sizes(
    frame: pd.DataFrame,
    *,
    outcome: str,
    factor: str,
    payload_column: str | None = None,
    binary: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Payload-level descriptive contrasts with grouped exploratory CIs.

    These generic pairwise outputs are never designated as primary inference.
    Prespecified mixed-effects models provide primary inferential estimates.
    """

    recovery_outcome = outcome in {
        "exact_recovery",
        PAYLOAD_RECOVERY_OUTCOME,
    }
    if factor not in frame:
        return pd.DataFrame()
    if recovery_outcome:
        frame, detected_payload_column = validate_trial_results(frame)
        payload_column = payload_column or detected_payload_column
    else:
        frame = _exclude_unavailable_estimand_rows(frame)
    if frame.empty or outcome not in frame:
        return pd.DataFrame()
    condition_scopes = _assert_effect_condition_not_pooled(
        frame, factor=factor, outcome=outcome
    )
    payload_column = payload_column or first_present_column(
        frame, PAYLOAD_COLUMN_CANDIDATES, required=True
    )
    assert payload_column is not None
    data = _payload_level_condition_values(
        frame,
        payload_column=payload_column,
        factor=factor,
        outcome=outcome,
    )
    levels = sorted(map(str, data[factor].dropna().unique()))
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RevisionStatisticsError("Effect tests require scipy") from exc
    rows: list[dict[str, Any]] = []
    for first_level, second_level in itertools.combinations(levels, 2):
        first = (
            data[data[factor].astype(str) == first_level]
            .set_index(payload_column)[outcome]
            .astype(float)
        )
        second = (
            data[data[factor].astype(str) == second_level]
            .set_index(payload_column)[outcome]
            .astype(float)
        )
        first.index = first.index.astype(str)
        second.index = second.index.astype(str)
        first_ids = set(first.index)
        second_ids = set(second.index)
        shared = sorted(first_ids & second_ids)
        same_payload_set = first_ids == second_ids and len(shared) >= 2
        partially_overlapping = bool(shared) and not same_payload_set
        if same_payload_set:
            comparison_design = "paired_payload"
            first_array = first.loc[shared].to_numpy(dtype=float)
            second_array = second.loc[shared].to_numpy(dtype=float)
        elif partially_overlapping:
            comparison_design = "partially_overlapping_payload"
            first_array = first.to_numpy(dtype=float)
            second_array = second.to_numpy(dtype=float)
        else:
            comparison_design = "independent_payload"
            first_array = first.to_numpy(dtype=float)
            second_array = second.to_numpy(dtype=float)

        difference = float(first_array.mean() - second_array.mean())
        contrast_seed = _stable_seed(
            seed, outcome, factor, first_level, second_level
        )
        if partially_overlapping:
            ci_low, ci_high, bootstrap_resamples_valid = (
                _bootstrap_partially_overlapping_mean_difference(
                    first,
                    second,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    seed=contrast_seed,
                )
            )
            bootstrap_design = "payload_cluster_partial_overlap"
        else:
            ci_low, ci_high = _bootstrap_mean_difference(
                first,
                second,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=contrast_seed,
                paired=same_payload_set,
            )
            enough_for_resampling = (
                len(shared) >= 2
                if same_payload_set
                else len(first) >= 2 and len(second) >= 2
            )
            bootstrap_resamples_valid = (
                int(n_resamples) if enough_for_resampling else 0
            )
            bootstrap_design = (
                "paired_payload" if same_payload_set else "independent_payload"
            )

        row: dict[str, Any] = {
            "outcome": outcome,
            "factor": factor,
            "level_first": first_level,
            "level_second": second_level,
            "analysis_unit": "payload",
            "comparison_design": comparison_design,
            "n_payloads_first": int(len(first)),
            "n_payloads_second": int(len(second)),
            "n_payloads_overlap": int(len(shared)),
            "n_payloads_paired": int(len(shared)) if same_payload_set else 0,
            "mean_first": float(first_array.mean()),
            "mean_second": float(second_array.mean()),
            "mean_difference": difference,
            "mean_difference_ci_low": ci_low,
            "mean_difference_ci_high": ci_high,
            "confidence_level": confidence_level,
            "bootstrap_unit": "payload",
            "bootstrap_design": bootstrap_design,
            "bootstrap_resamples": int(n_resamples),
            "bootstrap_resamples_valid": int(bootstrap_resamples_valid),
            "primary_inference": False,
            "inference_role": "descriptive_exploratory_pairwise",
            "inferential_p_value_supported": not partially_overlapping,
            **condition_scopes,
        }
        if recovery_outcome:
            row.update(
                {
                    "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                    "result_schema_revision": RESULT_SCHEMA_REVISION,
                    "recovery_outcome": PAYLOAD_RECOVERY_OUTCOME,
                    "recovery_outcome_semantics": PAYLOAD_RECOVERY_SEMANTICS,
                    "exact_recovery_compatibility_alias": True,
                    "exact_rank_replay_diagnostic_only": True,
                }
            )
        if binary:
            if not np.isin(first_array, [0, 1]).all() or not np.isin(
                second_array, [0, 1]
            ).all():
                raise RevisionStatisticsError(
                    f"{outcome} declared binary but contains non-binary payload values"
                )
            first_risk = float(first_array.mean())
            second_risk = float(second_array.mean())
            row["effect_type"] = "risk_difference"
            row["risk_difference"] = first_risk - second_risk
            row["risk_ratio"] = (
                first_risk / second_risk
                if second_risk > 0
                else (float("inf") if first_risk > 0 else float("nan"))
            )
            first_success = int(first_array.sum())
            second_success = int(second_array.sum())
            cells = np.asarray(
                [
                    [first_success, len(first_array) - first_success],
                    [second_success, len(second_array) - second_success],
                ],
                dtype=float,
            )
            corrected = cells + (0.5 if np.any(cells == 0) else 0.0)
            row["odds_ratio_haldane_anscombe"] = float(
                corrected[0, 0]
                * corrected[1, 1]
                / (corrected[0, 1] * corrected[1, 0])
            )
            if partially_overlapping:
                row["p_value_raw"] = np.nan
                row["test"] = "unsupported_partial_payload_overlap"
            elif same_payload_set:
                discordant = first_array != second_array
                discordant_count = int(discordant.sum())
                if discordant_count:
                    first_only = int(
                        ((first_array == 1) & (second_array == 0)).sum()
                    )
                    row["p_value_raw"] = float(
                        stats.binomtest(
                            first_only, discordant_count, 0.5
                        ).pvalue
                    )
                else:
                    row["p_value_raw"] = 1.0
                row["test"] = "exact_mcnemar_binomial"
            else:
                row["p_value_raw"] = float(stats.fisher_exact(cells)[1])
                row["test"] = "fisher_exact"
        else:
            row["effect_type"] = "mean_difference"
            row["hedges_g"] = _hedges_g(
                first_array, second_array, paired=same_payload_set
            )
            if partially_overlapping:
                row["p_value_raw"] = np.nan
                row["test"] = "unsupported_partial_payload_overlap"
            elif same_payload_set:
                row["p_value_raw"] = float(
                    stats.ttest_rel(
                        first_array, second_array, nan_policy="raise"
                    ).pvalue
                )
                row["test"] = "paired_t"
            else:
                row["p_value_raw"] = float(
                    stats.ttest_ind(
                        first_array,
                        second_array,
                        equal_var=False,
                        nan_policy="raise",
                    ).pvalue
                )
                row["test"] = "welch_t"
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        raw = result["p_value_raw"].tolist()
        result["p_value_holm"] = adjust_pvalues(raw, "holm")
        result["p_value_bh"] = adjust_pvalues(raw, "bh")
    return result


def summarize_detector_results(
    frame: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Summarize prediction rows or preserve saved upstream metric rows."""

    if frame.empty:
        return pd.DataFrame()
    if {"label", "score"}.issubset(frame.columns):
        validated, payload_column = validate_detector_rows(frame)
        from rankcloak.revision_detection import (
            grouped_bootstrap_detector_metrics,
        )

        groups = _group_columns(
            validated,
            (
                "split_id",
                "regime",
                "detector_name",
                "requested_kind",
                "implementation_kind",
                "held_out_value",
            ),
        )
        rows: list[dict[str, Any]] = []
        for keys, cell in _iter_groups(validated, groups):
            metrics = grouped_bootstrap_detector_metrics(
                _binary_series(cell["label"], label="detector label").tolist(),
                pd.to_numeric(cell["score"], errors="raise").astype(float).tolist(),
                cell[payload_column].astype(str).tolist(),
                n_resamples=n_resamples,
                seed=_stable_seed(seed, *keys),
            )
            rows.append(
                {
                    **dict(zip(groups, keys)),
                    "analysis_unit": "payload_group",
                    "metric_source": "recomputed_from_predictions",
                    **metrics,
                }
            )
        return pd.DataFrame(rows)
    metric_columns = {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
    }
    if metric_columns & set(frame.columns):
        result = frame.copy()
        result["metric_source"] = "saved_upstream_grouped_metrics"
        if "bootstrap_unit" in result.columns and not result[
            "bootstrap_unit"
        ].astype(str).eq("payload_group_id").all():
            raise RevisionStatisticsError(
                "Saved detector metrics were not payload-group bootstrapped"
            )
        return result
    raise RevisionStatisticsError(
        "Detector input must contain label/score predictions or saved metric columns"
    )


def fit_statsmodels_mixedlm(
    frame: pd.DataFrame,
    *,
    outcome: str,
    fixed_effects: Sequence[str],
    group_column: str,
    variance_component_columns: Sequence[str] = (),
    name: str = "mixedlm",
) -> pd.DataFrame:
    """Fit a Gaussian random-intercept model; never substitute fixed effects."""

    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MixedEffectsUnavailable(
            "Requested statsmodels MixedLM, but statsmodels is unavailable"
        ) from exc
    required = [outcome, group_column, *fixed_effects, *variance_component_columns]
    _require_nonempty(frame, required, label=f"mixed model {name}")
    if frame[group_column].nunique() < 2:
        raise MixedEffectsUnavailable(
            f"Mixed model {name} requires at least two {group_column} groups"
        )
    formula = outcome + (
        " ~ " + " + ".join(fixed_effects) if fixed_effects else " ~ 1"
    )
    variance_components = {
        f"vc_{column}": f"0 + C({column})"
        for column in variance_component_columns
    }
    try:
        model = smf.mixedlm(
            formula,
            frame,
            groups=frame[group_column],
            re_formula="1",
            vc_formula=variance_components or None,
        )
        result = model.fit(reml=False, method="lbfgs", disp=False)
    except Exception as exc:
        raise MixedEffectsUnavailable(
            f"statsmodels MixedLM {name} failed; no fixed-effects fallback used: {exc}"
        ) from exc
    confidence = result.conf_int()
    rows: list[dict[str, Any]] = []
    for term, estimate in result.fe_params.items():
        rows.append(
            {
                "model_name": name,
                "backend": "statsmodels_mixedlm",
                "family": "gaussian",
                "formula": formula,
                "group_column": group_column,
                "term": str(term),
                "estimate": float(estimate),
                "standard_error": float(result.bse_fe.loc[term]),
                "ci_low": float(confidence.loc[term, 0]),
                "ci_high": float(confidence.loc[term, 1]),
                "converged": bool(result.converged),
                "fixed_effects_fallback": False,
            }
        )
    return pd.DataFrame(rows)


def fit_r_lme4(
    frame: pd.DataFrame,
    *,
    formula: str,
    family: str,
    name: str,
    rscript: str = "Rscript",
) -> pd.DataFrame:
    """Fit lme4::lmer/glmer through R or fail explicitly when unavailable."""

    executable = shutil.which(rscript)
    if executable is None:
        raise MixedEffectsUnavailable(
            f"Requested R mixed model {name}, but {rscript} is unavailable"
        )
    if family not in {"gaussian", "binomial"}:
        raise MixedEffectsUnavailable(
            f"R adapter supports gaussian/binomial only; requested {family!r}"
        )
    with tempfile.TemporaryDirectory(prefix="rankcloak_mixed_") as directory:
        directory_path = Path(directory)
        input_path = directory_path / "input.csv"
        output_path = directory_path / "coefficients.csv"
        script_path = directory_path / "fit.R"
        frame.to_csv(input_path, index=False)
        script = """
args <- commandArgs(trailingOnly=TRUE)
if (!requireNamespace("lme4", quietly=TRUE)) {
  stop("Required R package 'lme4' is unavailable; no fixed-effects fallback used")
}
d <- read.csv(args[1], check.names=FALSE)
formula_value <- as.formula(args[3])
if (args[4] == "binomial") {
  fit <- lme4::glmer(formula_value, data=d, family=binomial())
} else {
  fit <- lme4::lmer(formula_value, data=d, REML=FALSE)
}
coefficients <- as.data.frame(summary(fit)$coefficients)
coefficients$term <- rownames(coefficients)
coefficients$converged <- is.null(fit@optinfo$conv$lme4$messages)
write.csv(coefficients, args[2], row.names=FALSE)
"""
        script_path.write_text(script, encoding="utf-8")
        process = subprocess.run(
            [
                executable,
                str(script_path),
                str(input_path),
                str(output_path),
                formula,
                family,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not output_path.is_file():
            detail = (process.stderr or process.stdout).strip()
            raise MixedEffectsUnavailable(
                f"R lme4 model {name} failed; no fixed-effects fallback used: {detail}"
            )
        coefficients = pd.read_csv(output_path)
    rename = {
        "Estimate": "estimate",
        "Std..Error": "standard_error",
        "Std. Error": "standard_error",
        "Pr...z..": "p_value_raw",
        "Pr...t..": "p_value_raw",
    }
    coefficients = coefficients.rename(columns=rename)
    coefficients.insert(0, "model_name", name)
    coefficients.insert(1, "backend", "R_lme4")
    coefficients.insert(2, "family", family)
    coefficients.insert(3, "formula", formula)
    coefficients["fixed_effects_fallback"] = False
    return coefficients


def load_analysis_config(path: str | Path | None) -> dict[str, Any]:
    """Load the frozen statistical design, or return documented defaults."""

    if path is None:
        return {
            "intervals": {
                "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
                "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
                "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            },
            "multiplicity": {
                "primary_family": "Holm",
                "exploratory_supplementary": "Benjamini-Hochberg",
            },
            "reporting": {
                "segments_as_independent_observations_forbidden": True
            },
        }
    config_path = Path(path)
    if not config_path.is_file():
        raise RevisionStatisticsError(
            f"Statistics configuration does not exist: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    reporting = config.get("reporting", {})
    if not reporting.get("segments_as_independent_observations_forbidden", False):
        raise RevisionStatisticsError(
            "Analysis configuration must forbid segments as independent observations"
        )
    return config


def run_mixed_effects_specs(
    sources: Mapping[str, pd.DataFrame],
    specs: Sequence[Mapping[str, Any]],
    *,
    fail_required: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run explicitly requested mixed models and record unavailable backends."""

    coefficient_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    required_failures: list[str] = []
    for index, raw_spec in enumerate(specs):
        spec = dict(raw_spec)
        name = str(spec.get("name", f"mixed_model_{index + 1}"))
        backend = str(spec.get("backend", "statsmodels")).lower()
        source_name = str(spec.get("data_source", "trials"))
        required = bool(spec.get("required", False))
        source = sources.get(source_name)
        status: dict[str, Any] = {
            "model_name": name,
            "backend": backend,
            "data_source": source_name,
            "required": required,
            "fixed_effects_fallback": False,
        }
        try:
            if source is None or source.empty:
                raise MixedEffectsUnavailable(
                    f"Mixed model {name} source {source_name!r} is empty"
                )
            outcome = str(spec["outcome"])
            if source_name == "trials" and outcome in {
                "exact_recovery",
                PAYLOAD_RECOVERY_OUTCOME,
            }:
                source, _ = validate_trial_results(source)
            family = str(spec.get("family", "gaussian")).lower()
            if backend in {"statsmodels", "statsmodels_mixedlm"}:
                if family != "gaussian":
                    raise MixedEffectsUnavailable(
                        "statsmodels adapter implements Gaussian MixedLM only; "
                        "request R lme4 for a logistic mixed model"
                    )
                coefficients = fit_statsmodels_mixedlm(
                    source,
                    outcome=str(spec["outcome"]),
                    fixed_effects=list(spec.get("fixed_effects", [])),
                    group_column=str(spec["group_column"]),
                    variance_component_columns=list(
                        spec.get("variance_component_columns", [])
                    ),
                    name=name,
                )
            elif backend in {"r", "r_lme4", "lme4"}:
                coefficients = fit_r_lme4(
                    source,
                    formula=str(spec["formula"]),
                    family=family,
                    name=name,
                    rscript=str(spec.get("rscript", "Rscript")),
                )
            else:
                raise MixedEffectsUnavailable(
                    f"Unknown mixed-effects backend {backend!r}"
                )
            coefficient_frames.append(coefficients)
            status.update(
                {
                    "status": "completed",
                    "coefficient_rows": int(len(coefficients)),
                }
            )
        except (KeyError, RevisionStatisticsError, MixedEffectsUnavailable) as exc:
            status.update(
                {
                    "status": "unavailable_or_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if required:
                required_failures.append(f"{name}: {exc}")
        statuses.append(status)
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True, sort=False)
        if coefficient_frames
        else pd.DataFrame()
    )
    if required_failures and fail_required:
        raise MixedEffectsUnavailable(
            "Required mixed-effects analyses failed; no fixed-effects "
            "substitution was made: " + "; ".join(required_failures)
        )
    return coefficients, statuses


def synthetic_smoke_frames(
    *, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, pd.DataFrame]:
    """Small deterministic fixture exercising nested and grouped analyses."""

    rng = np.random.default_rng(int(seed))
    trial_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for payload_index in range(12):
        payload = f"payload_{payload_index:02d}"
        for protocol_index, protocol in enumerate(("ascii_b8", "hex_nibble")):
            trial_id = f"{payload}_{protocol}"
            exact = int((payload_index + protocol_index) % 5 != 0)
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "phase": "confirmatory",
                    "payload_name": payload,
                    "payload_class": "sha256",
                    "model_id": "smoke_model",
                    "protocol_variant": protocol,
                    "prompt_id": f"prompt_{payload_index % 3}",
                    "prompt_category": "explanatory",
                    "replay_mode": "saved_token_ids",
                    "transformation_id": "unmodified",
                    "mitigation_id": "none",
                    "exact_recovery": exact,
                    "effective_payload_rate": 1.2
                    + protocol_index * 0.25
                    + rng.normal(0.0, 0.04),
                }
            )
            for segment_index in range(2):
                feature_rows.append(
                    {
                        "trial_id": trial_id,
                        "payload_name": payload,
                        "payload_class": "sha256",
                        "model_id": "smoke_model",
                        "protocol_variant": protocol,
                        "prompt_id": f"prompt_{payload_index % 3}",
                        "prompt_category": "explanatory",
                        "text_view": "full_message",
                        "segment_index": segment_index,
                        "text": (
                            f"Example {payload_index} protocol {protocol_index} "
                            f"segment {segment_index} explains a deterministic idea."
                        ),
                        "prompt_text": "Explain one deterministic idea.",
                    }
                )
            runtime_rows.append(
                {
                    "trial_id": trial_id,
                    "payload_name": payload,
                    "payload_class": "sha256",
                    "model_id": "smoke_model",
                    "protocol_variant": protocol,
                    "hardware_id": "cpu_smoke",
                    "encoding_seconds": 0.2 + protocol_index * 0.03,
                    "decoding_seconds": 0.1 + protocol_index * 0.02,
                    "payload_bits_per_second": 100.0 - protocol_index * 8.0,
                }
            )
    detector_rows: list[dict[str, Any]] = []
    for index in range(20):
        label = index % 2
        detector_rows.append(
            {
                "split_id": "matched_smoke",
                "regime": "matched",
                "detector_name": "smoke_scores",
                "row_id": f"detector_{index:02d}",
                "payload_group_id": f"detector_payload_{index:02d}",
                "label": label,
                "score": 0.75 if label else 0.25,
            }
        )
    return {
        "trials": pd.DataFrame(trial_rows),
        "features": pd.DataFrame(feature_rows),
        "runtime": pd.DataFrame(runtime_rows),
        "detectors": pd.DataFrame(detector_rows),
    }


def _input_manifest(
    categorized_paths: Mapping[str, Sequence[str | Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_hashes: dict[str, Path] = {}
    for category, paths in categorized_paths.items():
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if path in seen_paths:
                raise RevisionStatisticsError(
                    f"Input path reused across analysis categories: {path}"
                )
            digest = file_sha256(path)
            if digest in seen_hashes:
                raise RevisionStatisticsError(
                    "Byte-identical inputs reused across categories: "
                    f"{seen_hashes[digest]} and {path}"
                )
            seen_paths.add(path)
            seen_hashes[digest] = path
            rows.append(
                {
                    "category": category,
                    "path": str(path),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def _write_machine_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=["status"])
    frame.to_csv(path, index=False)


def run_statistics_analysis(
    *,
    output_dir: str | Path,
    trial_paths: Sequence[str | Path] = (),
    feature_paths: Sequence[str | Path] = (),
    detector_paths: Sequence[str | Path] = (),
    runtime_paths: Sequence[str | Path] = (),
    statistics_config: str | Path | None = None,
    mixed_effects_specs: Sequence[Mapping[str, Any]] = (),
    smoke: bool = False,
    overwrite: bool = False,
) -> AnalysisArtifacts:
    """Validate saved artifacts and emit all machine-readable analysis products."""

    config = load_analysis_config(statistics_config)
    intervals = config.get("intervals", {})
    confidence_level = float(
        intervals.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
    )
    n_resamples = int(
        intervals.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
    )
    if smoke:
        n_resamples = min(n_resamples, 100)
    seed = int(intervals.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_paths = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in target_paths.values() if path.exists()]
    if existing and not overwrite:
        raise RevisionStatisticsError(
            "Refusing to overwrite analysis outputs: "
            + ", ".join(map(str, existing))
        )

    categorized = {
        "trials": list(trial_paths),
        "features": list(feature_paths),
        "detectors": list(detector_paths),
        "runtime": list(runtime_paths),
    }
    input_rows = _input_manifest(categorized)
    if smoke and not any(categorized.values()):
        sources = synthetic_smoke_frames(seed=seed)
    else:
        sources = {
            category: load_saved_tables(paths, label=category)
            for category, paths in categorized.items()
        }
    if not any(not frame.empty for frame in sources.values()):
        raise RevisionStatisticsError(
            "No saved analysis inputs supplied (use smoke=True for the fixture)"
        )

    raw_input_rows = {
        name: int(len(frame)) for name, frame in sources.items()
    }
    trials, trial_payload = validate_trial_results(sources["trials"])
    excluded_unavailable_trial_rows = int(
        trials.attrs.get("excluded_unavailable_rows", 0)
    )
    payload_fidelity_contract = dict(
        trials.attrs.get(
            "payload_fidelity_contract",
            {
                "contract_version": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "semantics": PAYLOAD_RECOVERY_SEMANTICS,
                "primary_outcome": PAYLOAD_RECOVERY_OUTCOME,
                "compatibility_alias": "exact_recovery",
                "alias_equality_validated": True,
                "exact_rank_replay_role": "diagnostic_only",
                "direct_rows": 0,
                "direct_rows_contract_verified": 0,
            },
        )
    )
    features, feature_payload = validate_feature_rows(sources["features"])
    runtime, runtime_payload = validate_runtime_results(sources["runtime"])
    detectors, _ = validate_detector_rows(sources["detectors"])
    sources = {
        "trials": trials,
        "features": features,
        "runtime": runtime,
        "detectors": detectors,
    }

    recovery = summarize_recovery(
        trials,
        payload_column=trial_payload,
        confidence_level=confidence_level,
    )
    quality = build_trial_quality_table(features)
    continuous_frames: list[pd.DataFrame] = []
    if not trials.empty:
        continuous_frames.append(
            summarize_continuous_outcomes(
                trials,
                payload_column=trial_payload,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    if not quality.empty:
        continuous_frames.append(
            summarize_continuous_outcomes(
                quality,
                payload_column=feature_payload,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    if not runtime.empty:
        continuous_frames.append(
            summarize_continuous_outcomes(
                runtime,
                payload_column=runtime_payload,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    continuous = (
        pd.concat(
            [frame for frame in continuous_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in continuous_frames)
        else pd.DataFrame()
    )

    # The runner emits smoke/robustness diagnostics for several replay modes,
    # but the supported confirmatory estimand is saved-token-ID replay.  Do not
    # average binary outcomes across replay modes within a payload before
    # computing descriptive contrasts for the primary outcome.
    effect_trials = trials
    primary_effect_scope: dict[str, str] = {}
    for column, required_level in PRIMARY_EFFECT_CONDITIONS.items():
        if column not in effect_trials.columns:
            continue
        effect_trials = effect_trials[
            effect_trials[column].astype(str).eq(required_level)
        ]
        primary_effect_scope[column] = required_level

    quality_view_column = first_present_column(quality, VIEW_COLUMN_CANDIDATES)
    quality_view_levels: list[str] = []
    quality_effect_cells: list[tuple[tuple[object, ...], pd.DataFrame]] = []
    quality_effect_exclusion_reason: str | None = None
    if not quality.empty:
        if quality_view_column is None:
            quality_effect_exclusion_reason = "missing_explicit_text_view"
        elif quality[quality_view_column].isna().any():
            quality_effect_exclusion_reason = "missing_values_in_text_view"
        else:
            quality_view_levels = sorted(
                quality[quality_view_column].astype(str).unique().tolist()
            )
            quality_effect_strata = [quality_view_column] + [
                column
                for column in EFFECT_STRATUM_COLUMNS
                if column != quality_view_column
                and column in quality.columns
                and quality[column].notna().any()
            ]
            quality_effect_cells = [
                (keys, cell.copy())
                for keys, cell in _iter_groups(quality, quality_effect_strata)
            ]

    effect_frames: list[pd.DataFrame] = []
    for factor in ("protocol_variant", "model_id", "prompt_category"):
        if not effect_trials.empty and factor in effect_trials:
            effect_frames.append(
                pairwise_effect_sizes(
                    effect_trials,
                    outcome="exact_recovery",
                    factor=factor,
                    payload_column=trial_payload,
                    binary=True,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
            for outcome in CONTINUOUS_OUTCOME_CANDIDATES:
                if outcome in effect_trials:
                    effect_frames.append(
                        pairwise_effect_sizes(
                            effect_trials,
                            outcome=outcome,
                            factor=factor,
                            payload_column=trial_payload,
                            confidence_level=confidence_level,
                            n_resamples=n_resamples,
                            seed=seed,
                        )
                    )
        for view_level, quality_cell in quality_effect_cells:
            if factor not in quality_cell:
                continue
            for outcome in QUALITY_EFFECT_OUTCOMES:
                effect_frames.append(
                    pairwise_effect_sizes(
                        quality_cell,
                        outcome=outcome,
                        factor=factor,
                        payload_column=feature_payload,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        seed=_stable_seed(
                            seed,
                            "quality_view",
                            quality_view_column,
                            view_level,
                        ),
                    )
                )
    effects = (
        pd.concat(
            [
                frame.dropna(axis=1, how="all")
                for frame in effect_frames
                if not frame.empty
            ],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in effect_frames)
        else pd.DataFrame()
    )
    quality_effect_row_count = 0
    unscoped_quality_effect_row_count = 0
    if not effects.empty:
        if "p_value_raw" not in effects.columns:
            effects["p_value_raw"] = np.nan
        effects["p_value_holm"] = adjust_pvalues(
            effects["p_value_raw"].tolist(), "holm"
        )
        effects["p_value_bh"] = adjust_pvalues(
            effects["p_value_raw"].tolist(), "bh"
        )
        quality_effect_mask = effects["outcome"].isin(QUALITY_EFFECT_OUTCOMES)
        quality_effect_row_count = int(quality_effect_mask.sum())
        if quality_effect_row_count:
            expected_scope_column = f"{quality_view_column}_scope"
            if expected_scope_column not in effects.columns:
                raise RevisionStatisticsError(
                    "Quality effect rows lack an explicit text-view scope"
                )
            unscoped_quality_effect_row_count = int(
                effects.loc[quality_effect_mask, expected_scope_column]
                .isna()
                .sum()
            )
            if unscoped_quality_effect_row_count:
                raise RevisionStatisticsError(
                    "Quality effect rows would marginalize text views"
                )
    detector_summary = summarize_detector_results(
        detectors, n_resamples=n_resamples, seed=seed
    )
    mixed_sources = {
        **sources,
        "quality": quality,
    }
    mixed_coefficients, mixed_status = run_mixed_effects_specs(
        mixed_sources, mixed_effects_specs, fail_required=False
    )

    integrity = {
        "status": "passed",
        "analysis_unit": "payload",
        "segments_as_independent_observations": False,
        "nested_segments_collapsed_before_quality_inference": True,
        "confidence_level": confidence_level,
        "bootstrap_unit": "payload",
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
        "smoke_fixture": smoke and not any(categorized.values()),
        "input_rows": {
            name: int(len(frame)) for name, frame in sources.items()
        },
        "raw_input_rows": raw_input_rows,
        "estimand_exclusions": {
            "unavailable_trial_rows": excluded_unavailable_trial_rows,
            "unavailable_rows_counted": False,
        },
        "payload_fidelity_contract": payload_fidelity_contract,
        "primary_effect_scope": {
            **primary_effect_scope,
            "eligible_trial_rows": int(len(effect_trials)),
            "diagnostic_replay_fallback": False,
            "pairwise_effects_are_primary_inference": False,
        },
        "quality_effect_scope": {
            "view_column": quality_view_column,
            "view_levels": quality_view_levels,
            "view_stratified": bool(quality_effect_cells),
            "effect_rows": quality_effect_row_count,
            "unscoped_effect_rows": unscoped_quality_effect_row_count,
            "exclusion_reason": quality_effect_exclusion_reason,
        },
        "independent_payloads": {
            "trials": int(trials[trial_payload].nunique())
            if not trials.empty
            else 0,
            "features": int(features[feature_payload].nunique())
            if not features.empty
            else 0,
            "runtime": int(runtime[runtime_payload].nunique())
            if not runtime.empty
            else 0,
        },
        "quality_metric_notice": (
            "Readability, repetition, surface flags, and TF-IDF similarity are "
            "deterministic diagnostics and not substitutes for human ratings."
        ),
        "mixed_effects": mixed_status
        or [
            {
                "status": "not_requested",
                "fixed_effects_fallback": False,
            }
        ],
    }
    _write_machine_csv(recovery, target_paths["recovery"])
    _write_machine_csv(continuous, target_paths["continuous"])
    _write_machine_csv(effects, target_paths["effects"])
    _write_machine_csv(quality, target_paths["quality"])
    _write_machine_csv(detector_summary, target_paths["detectors"])
    _write_machine_csv(mixed_coefficients, target_paths["mixed"])
    target_paths["mixed_status"].write_text(
        json.dumps(
            mixed_status
            or [
                {
                    "status": "not_requested",
                    "fixed_effects_fallback": False,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    target_paths["integrity"].write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "statistics_config": (
            {
                "path": str(Path(statistics_config).resolve()),
                "sha256": file_sha256(statistics_config),
            }
            if statistics_config is not None
            else {"path": None, "defaults_embedded": True}
        ),
        "inputs": input_rows,
        "outputs": {},
        "determinism": {
            "bootstrap_seed": seed,
            "bootstrap_resamples": n_resamples,
        },
    }
    for key, path in target_paths.items():
        if key == "manifest":
            continue
        manifest["outputs"][key] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    target_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    required_failed = [
        status
        for status in mixed_status
        if status.get("required") and status.get("status") != "completed"
    ]
    if required_failed:
        raise MixedEffectsUnavailable(
            "Required mixed-effects analyses failed after diagnostic outputs "
            "were written; no fixed-effects substitution was made: "
            + "; ".join(
                f"{status['model_name']}: {status.get('error', 'unknown error')}"
                for status in required_failed
            )
        )
    return AnalysisArtifacts(
        output_dir=str(output_path.resolve()),
        files={
            key: str(path.resolve()) for key, path in target_paths.items()
        },
        integrity_report=integrity,
    )
