"""Computational-overhead summaries from immutable revision runtime records."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rankcloak.revision_statistics import (
    RevisionStatisticsError,
    enrich_runtime_conditions,
    file_sha256,
    summarize_continuous_outcomes,
    validate_runtime_results,
    validate_trial_results,
)


OUTPUT_FILENAMES = {
    "summary": "overhead_summary.csv",
    "scaling": "overhead_scaling_summary.csv",
    "inventory": "runtime_scope_inventory.csv",
    "initialization": "model_initialization.csv",
    "memory": "memory_profiles.csv",
    "plot_source": "overhead_plot_source.csv",
    "limitations": "measurement_limitations.json",
    "manifest": "overhead_analysis_manifest.json",
}

SOURCE_RUNTIME_METRICS = (
    "model_load_seconds",
    "representation_seconds",
    "filter_setup_seconds",
    "generation_seconds",
    "encoding_seconds",
    "recovery_seconds",
    "decoding_seconds",
    "execution_seconds",
    "generation_tokens_per_second",
    "encoding_tokens_per_second",
    "decoding_tokens_per_second",
    "representation_bits_per_second",
    "payload_bits_per_second",
    "serialized_bits_per_second",
    "cover_tokens_per_payload_byte",
    "peak_ram_mib",
    "peak_gpu_memory_mib",
)

SUMMARY_OUTCOMES = (
    "generation_seconds",
    "representation_seconds",
    "filter_setup_seconds",
    "encoding_seconds",
    "recovery_seconds",
    "decoding_seconds",
    "inverse_transcode_seconds",
    "execution_seconds",
    "encoding_overhead_seconds",
    "decoding_overhead_seconds",
    "non_generation_overhead_seconds",
    "overhead_to_generation_ratio",
    "generation_tokens_per_second",
    "encoding_tokens_per_second",
    "decoding_tokens_per_second",
    "representation_bits_per_second",
    "payload_bits_per_second",
    "serialized_bits_per_second",
    "cover_tokens_per_payload_byte",
    "effective_artifact_bits_per_full_token",
)

MAIN_GROUP_COLUMNS = (
    "source_stage",
    "runtime_scope",
    "model_id",
    "protocol_variant",
    "hardware_id",
)

SCALING_GROUP_COLUMNS = MAIN_GROUP_COLUMNS + (
    "language",
    "payload_class",
    "artifact_bit_length",
    "alphabet_size_B",
    "segmented",
    "segment_size_ranks",
    "token_filter",
    "tail_policy",
    "leadin_tokens",
    "topic_schedule",
    "ablation_factor",
    "ablation_level",
    "robustness_family",
    "replay_mode",
    "transformation_id",
)


class OverheadAnalysisError(ValueError):
    """Raised when saved runtime evidence is malformed or inconsistently bound."""


@dataclass(frozen=True)
class OverheadArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


def _read_csv(path: str | Path, *, label: str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.is_file():
        raise OverheadAnalysisError(f"Missing {label}: {resolved}")
    try:
        return pd.read_csv(resolved, low_memory=False)
    except Exception as exc:
        raise OverheadAnalysisError(f"Could not read {label}: {exc}") from exc


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise OverheadAnalysisError(f"Missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OverheadAnalysisError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise OverheadAnalysisError(f"{label} must contain a JSON object")
    return value


def _load_staged_tables(
    paths: Mapping[str, str | Path], *, label: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage, path in sorted(paths.items()):
        if not stage or "=" in stage:
            raise OverheadAnalysisError(f"Invalid stage label {stage!r}")
        frame = _read_csv(path, label=f"{stage} {label}")
        frame["source_stage"] = stage
        frames.append(frame)
    if not frames:
        raise OverheadAnalysisError(f"No {label} inputs were supplied")
    return pd.concat(frames, ignore_index=True, sort=False)


def _numeric_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in SOURCE_RUNTIME_METRICS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
        finite = result[column].dropna()
        if not np.isfinite(finite).all() or (finite < 0).any():
            raise OverheadAnalysisError(
                f"Runtime metric {column} contains negative or nonfinite values"
            )
    return result


def _complete_sum(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame[list(columns)]
    complete = values.notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[complete] = values.loc[complete].sum(axis=1)
    return result


def _nonnegative_difference(
    frame: pd.DataFrame,
    minuend: str,
    subtrahend: str,
    *,
    label: str,
) -> pd.Series:
    """Subtract recorded inclusive timings while tolerating float roundoff only."""

    complete = frame[[minuend, subtrahend]].notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[complete] = (
        frame.loc[complete, minuend] - frame.loc[complete, subtrahend]
    )
    tolerance = 1e-12
    materially_negative = result < -tolerance
    if materially_negative.any():
        raise OverheadAnalysisError(
            f"{label} is negative for {int(materially_negative.sum())} rows; "
            "inclusive timing semantics are inconsistent"
        )
    result.loc[result.between(-tolerance, 0.0, inclusive="left")] = 0.0
    return result


def derive_overhead_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate non-generation work from the runner's inclusive wrappers."""

    result = _numeric_runtime(frame)
    result["encoding_overhead_seconds"] = _nonnegative_difference(
        result,
        "encoding_seconds",
        "generation_seconds",
        label="encoding_seconds - generation_seconds",
    )
    result["inverse_transcode_seconds"] = _nonnegative_difference(
        result,
        "decoding_seconds",
        "recovery_seconds",
        label="decoding_seconds - recovery_seconds",
    )
    # supported_decoding_seconds is already the inclusive saved-replay plus
    # inverse-transcode wrapper, so it must not be added to recovery_seconds.
    result["decoding_overhead_seconds"] = result["decoding_seconds"]
    result["non_generation_overhead_seconds"] = _complete_sum(
        result, ("encoding_overhead_seconds", "decoding_overhead_seconds")
    )

    encoding_components = _complete_sum(
        result,
        ("representation_seconds", "filter_setup_seconds", "generation_seconds"),
    )
    component_complete = (
        encoding_components.notna() & result["encoding_seconds"].notna()
    )
    result["encoding_component_residual_seconds"] = np.nan
    result.loc[component_complete, "encoding_component_residual_seconds"] = (
        result.loc[component_complete, "encoding_seconds"]
        - encoding_components.loc[component_complete]
    )
    inconsistent = result["encoding_component_residual_seconds"].abs() > 1e-9
    if inconsistent.any():
        raise OverheadAnalysisError(
            "encoding_seconds does not equal representation + filter setup + "
            f"generation for {int(inconsistent.sum())} rows"
        )
    denominator = result["generation_seconds"]
    ratio_mask = (
        result["non_generation_overhead_seconds"].notna()
        & denominator.notna()
        & denominator.gt(0)
    )
    result["overhead_to_generation_ratio"] = np.nan
    result.loc[ratio_mask, "overhead_to_generation_ratio"] = (
        result.loc[ratio_mask, "non_generation_overhead_seconds"]
        / denominator.loc[ratio_mask]
    )
    result["overhead_component_policy"] = (
        "complete_case_sum_representation_filter_encoding_and_recovery_decoding"
    )
    return result


def _group_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [
        column
        for column in candidates
        if column in frame.columns and frame[column].notna().any()
    ]


def _runtime_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = _group_columns(
        frame,
        ("source_stage", "runtime_scope", "record_type", "evidence_status"),
    )
    rows: list[dict[str, Any]] = []
    for keys, cell in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_tuple))
        row.update(
            {
                "row_count": int(len(cell)),
                "trial_metadata_matched_rows": int(
                    cell["runtime_trial_metadata_matched"].sum()
                ),
                "payload_units": int(cell["payload_name"].nunique()),
            }
        )
        for metric in SOURCE_RUNTIME_METRICS:
            row[f"{metric}_recorded_rows"] = int(cell[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _session_rows(
    frame: pd.DataFrame,
    scope: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    selected = frame[
        frame.get(
            "runtime_scope", pd.Series(index=frame.index, dtype=object)
        ).eq(scope)
    ]
    available = [column for column in columns if column in selected.columns]
    return selected[available].copy().reset_index(drop=True)


def _atomic_write_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def build_overhead_artifacts(
    *,
    trial_paths: Mapping[str, str | Path],
    runtime_paths: Mapping[str, str | Path],
    statistics_config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> OverheadArtifacts:
    """Build grouped overhead, scaling, session, and measurement-limit tables."""

    if set(trial_paths) != set(runtime_paths):
        raise OverheadAnalysisError(
            "Trial and runtime stage labels must match exactly"
        )
    trials_raw = _load_staged_tables(trial_paths, label="trials")
    runtime_raw = _load_staged_tables(runtime_paths, label="runtime")
    try:
        trials, _ = validate_trial_results(trials_raw)
        runtime, payload_column = validate_runtime_results(runtime_raw)
        runtime = enrich_runtime_conditions(runtime, trials)
    except RevisionStatisticsError as exc:
        raise OverheadAnalysisError(
            f"Runtime/trial integrity validation failed: {exc}"
        ) from exc
    runtime = derive_overhead_metrics(runtime)

    config = _read_json(statistics_config, label="statistics config")
    intervals = config.get("intervals", {})
    confidence_level = float(intervals.get("confidence_level", 0.95))
    n_resamples = int(intervals.get("bootstrap_resamples", 2_000))
    seed = int(intervals.get("bootstrap_seed", 2_026_080_801))
    if not 0 < confidence_level < 1 or n_resamples <= 0:
        raise OverheadAnalysisError("Invalid frozen interval configuration")

    summary = summarize_continuous_outcomes(
        runtime,
        outcomes=[column for column in SUMMARY_OUTCOMES if column in runtime],
        payload_column=payload_column,
        group_columns=_group_columns(runtime, MAIN_GROUP_COLUMNS),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    matched = runtime[runtime["runtime_trial_metadata_matched"]].copy()
    scaling = summarize_continuous_outcomes(
        matched,
        outcomes=[column for column in SUMMARY_OUTCOMES if column in matched],
        payload_column=payload_column,
        group_columns=_group_columns(matched, SCALING_GROUP_COLUMNS),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    inventory = _runtime_inventory(runtime)
    initialization = _session_rows(
        runtime,
        "model_load_session",
        (
            "source_stage",
            "model_id",
            "hardware_id",
            "hardware_hash",
            "session_index",
            "model_load_seconds",
            "peak_ram_availability",
            "peak_gpu_memory_availability",
        ),
    )
    memory = _session_rows(
        runtime,
        "model_shard_memory_profile",
        (
            "source_stage",
            "model_id",
            "hardware_id",
            "hardware_hash",
            "session_index",
            "peak_ram_mib",
            "peak_gpu_memory_mib",
            "peak_ram_availability",
            "peak_gpu_memory_availability",
        ),
    )
    plot_outcomes = {
        "generation_seconds",
        "encoding_overhead_seconds",
        "decoding_overhead_seconds",
        "payload_bits_per_second",
        "generation_tokens_per_second",
        "effective_artifact_bits_per_full_token",
    }
    plot_source = summary[summary["outcome"].isin(plot_outcomes)].copy()
    limitations: dict[str, Any] = {
        "schema_version": "rankcloak-revision-overhead-limitations-v1",
        "cpu_time": {
            "status": "unavailable_not_recorded",
            "wall_clock_substitution": False,
        },
        "warmup_repeated_microbenchmark": {
            "status": "unavailable_not_run",
            "reason": "saved confirmatory executions were validated without rerunning generation",
        },
        "ram": {
            "profile_rows": int(len(memory)),
            "limitation": (
                "OS process high-water and sampled current RSS; trial rows are not "
                "treated as independent peak-memory measurements"
            ),
        },
        "vram": {
            "profile_rows": int(memory["peak_gpu_memory_mib"].notna().sum())
            if "peak_gpu_memory_mib" in memory
            else 0,
            "limitation": (
                "one-second total selected-device memory.used sample; includes "
                "co-tenant processes and is not a kernel-exact allocation peak"
            ),
        },
        "timing_scope": (
            "recorded runner components; no claim that wrapper and driver overhead "
            "are perfectly isolated from base-model execution"
        ),
        "timing_semantics": {
            "encoding_seconds": (
                "inclusive representation + filter setup + generation wrapper"
            ),
            "encoding_overhead_seconds": (
                "encoding_seconds minus generation_seconds"
            ),
            "decoding_seconds": (
                "inclusive saved-token replay + inverse transcode wrapper"
            ),
            "decoding_overhead_seconds": "same inclusive decoding wrapper",
            "inverse_transcode_seconds": (
                "decoding_seconds minus saved-token replay seconds"
            ),
        },
        "new_generation_runs": 0,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise OverheadAnalysisError(
            "Refusing to overwrite overhead outputs: "
            + ", ".join(str(path) for path in existing)
        )
    frames = {
        "summary": summary,
        "scaling": scaling,
        "inventory": inventory,
        "initialization": initialization,
        "memory": memory,
        "plot_source": plot_source,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, targets[key])
    _atomic_write_json(limitations, targets["limitations"])

    inputs: dict[str, Any] = {"trials": {}, "runtime": {}}
    for category, paths in (("trials", trial_paths), ("runtime", runtime_paths)):
        for stage, raw_path in sorted(paths.items()):
            resolved = Path(raw_path).resolve()
            inputs[category][stage] = {
                "path": str(resolved),
                "sha256": file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
    config_path = Path(statistics_config).resolve()
    inputs["statistics_config"] = {
        "path": str(config_path),
        "sha256": file_sha256(config_path),
        "size_bytes": config_path.stat().st_size,
    }
    audit_summary = {
        "trial_rows": int(len(trials)),
        "runtime_rows": int(len(runtime)),
        "matched_trial_runtime_rows": int(
            runtime["runtime_trial_metadata_matched"].sum()
        ),
        "unmatched_nontrial_runtime_rows": int(
            (~runtime["runtime_trial_metadata_matched"]).sum()
        ),
        "model_load_rows": int(len(initialization)),
        "memory_profile_rows": int(len(memory)),
        "cpu_time_rows": 0,
        "new_generation_runs": 0,
    }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-revision-overhead-analysis-v1",
        "status": "passed",
        "inputs": inputs,
        "outputs": {},
        "summary": audit_summary,
        "inference": {
            "analysis_unit": "payload_name",
            "confidence_level": confidence_level,
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": seed,
        },
        "component_policy": (
            "inclusive wrapper timings are differenced only when both operands are "
            "recorded; derived sums require every named component and no missing "
            "component is treated as zero"
        ),
    }
    for key, target in targets.items():
        if key == "manifest":
            continue
        manifest["outputs"][key] = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
            "row_count": int(len(frames[key])) if key in frames else None,
        }
    _atomic_write_json(manifest, targets["manifest"])
    return OverheadArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=audit_summary,
    )
