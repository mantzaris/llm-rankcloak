"""Validated transmission-robustness summaries for revision evidence.

The builder consumes only saved preprocessing artifacts.  It treats each source
cover as the resampling unit, keeps unavailable conditions out of recovery
denominators, and records failure mechanisms as descriptive diagnostics rather
than causal proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rankcloak.revision_statistics import (
    grouped_payload_bootstrap_ci,
    wilson_interval,
)


OUTPUT_FILENAMES = {
    "conditions": "recovery_by_condition.csv",
    "models": "recovery_by_model_condition.csv",
    "failure_taxonomy": "failure_taxonomy.csv",
    "failure_summary": "failure_mechanism_summary.csv",
    "unavailable": "unavailable_summary.csv",
    "plot_source": "robustness_recovery_plot_source.csv",
    "manifest": "robustness_analysis_manifest.json",
}

CONDITION_COLUMNS = (
    "robustness_family",
    "replay_mode",
    "transformation_id",
)
MODEL_CONDITION_COLUMNS = CONDITION_COLUMNS + (
    "source_model_id",
    "model_id",
)


class RobustnessAnalysisError(ValueError):
    """Raised when saved robustness evidence violates the frozen design."""


@dataclass(frozen=True)
class RobustnessArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: str | Path, *, label: str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.is_file():
        raise RobustnessAnalysisError(f"Missing {label} input: {resolved}")
    try:
        return pd.read_csv(resolved, low_memory=False)
    except Exception as exc:
        raise RobustnessAnalysisError(
            f"Could not read {label} input {resolved}: {exc}"
        ) from exc


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise RobustnessAnalysisError(f"Missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RobustnessAnalysisError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RobustnessAnalysisError(f"{label} must contain a JSON object")
    return value


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RobustnessAnalysisError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def _binary(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise RobustnessAnalysisError(f"{label} must contain only zero or one")
    return numeric.astype(int)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _condition_key_frame(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = "not_applicable"
        result[column] = result[column].fillna("not_applicable").astype(str)
    return result


def _expected_family_counts(config: Mapping[str, Any]) -> dict[str, int]:
    mapping = {
        "replay_modes": "replay_modes",
        "raw_transmission": "raw_transmission",
        "limited_mitigation": "limited_mitigation",
        "cross_model_mismatch": "cross_model_mismatch",
    }
    result: dict[str, int] = {}
    for family, section_name in mapping.items():
        section = config.get(section_name)
        if not isinstance(section, Mapping) or "outcome_rows" not in section:
            raise RobustnessAnalysisError(
                f"Robustness config lacks {section_name}.outcome_rows"
            )
        result[family] = int(section["outcome_rows"])
    return result


def validate_robustness_inputs(
    trials: pd.DataFrame,
    failures: pd.DataFrame,
    unavailable: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate row identities, frozen counts, and failure/unavailability roles."""

    _require_columns(
        trials,
        (
            "trial_id",
            "source_trial_id",
            "exact_payload_recovery",
            "exact_recovery",
            "payload_name",
            *MODEL_CONDITION_COLUMNS,
        ),
        label="robustness trials",
    )
    _require_columns(
        failures,
        (
            "failure_id",
            "trial_id",
            "source_trial_id",
            "payload_name",
            "failure_category",
            "first_differing_position",
            "expected_token_id",
            "recovered_token_id",
            "expected_rank",
            "recovered_rank",
            "context_sha256",
            "boundary_start_offset",
            "boundary_end_offset",
            *MODEL_CONDITION_COLUMNS,
        ),
        label="robustness failures",
    )
    _require_columns(
        unavailable,
        (
            "work_id",
            "source_trial_id",
            "reason_code",
            "excluded_from_estimands",
            *MODEL_CONDITION_COLUMNS,
        ),
        label="robustness unavailable rows",
    )
    configured_failure_fields = config.get("failure_record_required_fields", [])
    if not isinstance(configured_failure_fields, list):
        raise RobustnessAnalysisError(
            "failure_record_required_fields must be a JSON list"
        )
    _require_columns(
        failures,
        [str(value) for value in configured_failure_fields],
        label="robustness failures required by config",
    )

    if trials.empty:
        raise RobustnessAnalysisError("Robustness trials are empty")
    if trials["trial_id"].isna().any() or trials["trial_id"].duplicated().any():
        raise RobustnessAnalysisError("Robustness trial_id values must be unique")
    if failures["failure_id"].isna().any() or failures["failure_id"].duplicated().any():
        raise RobustnessAnalysisError("Robustness failure_id values must be unique")
    if failures["trial_id"].isna().any() or failures["trial_id"].duplicated().any():
        raise RobustnessAnalysisError(
            "Each failed robustness trial must have exactly one failure row"
        )
    if unavailable["work_id"].isna().any() or unavailable["work_id"].duplicated().any():
        raise RobustnessAnalysisError("Unavailable work_id values must be unique")

    recovery = _binary(
        trials["exact_payload_recovery"], label="exact_payload_recovery"
    )
    alias = _binary(trials["exact_recovery"], label="exact_recovery")
    if not recovery.equals(alias):
        raise RobustnessAnalysisError(
            "exact_recovery does not equal exact_payload_recovery"
        )
    failed_ids = set(trials.loc[recovery.eq(0), "trial_id"].astype(str))
    detail_ids = set(failures["trial_id"].astype(str))
    if failed_ids != detail_ids:
        raise RobustnessAnalysisError(
            "Failure details do not match the exact set of failed trials"
        )
    unavailable_ids = set(unavailable["work_id"].astype(str))
    if set(trials["trial_id"].astype(str)) & unavailable_ids:
        raise RobustnessAnalysisError(
            "Observed trial IDs overlap explicitly unavailable work IDs"
        )
    excluded = unavailable["excluded_from_estimands"]
    if not excluded.map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    ).all():
        raise RobustnessAnalysisError(
            "Every unavailable row must be excluded_from_estimands"
        )
    if "execution_error_type" in failures.columns:
        execution_errors = failures["execution_error_type"].dropna().astype(str).str.strip()
        if execution_errors.ne("").any():
            raise RobustnessAnalysisError(
                "Execution errors are present in the recovery-failure table"
            )

    expected_total = int(
        config.get("expected_counts", {}).get("robustness_outcome_rows", -1)
    )
    actual_total = int(len(trials) + len(unavailable))
    if expected_total < 0 or actual_total != expected_total:
        raise RobustnessAnalysisError(
            f"Frozen robustness total is {expected_total}, observed plan total is {actual_total}"
        )
    combined = pd.concat(
        [
            trials[["robustness_family"]],
            unavailable[["robustness_family"]],
        ],
        ignore_index=True,
    )
    actual_family_counts = {
        str(key): int(value)
        for key, value in combined["robustness_family"].value_counts().items()
    }
    expected_family_counts = _expected_family_counts(config)
    if actual_family_counts != expected_family_counts:
        raise RobustnessAnalysisError(
            "Frozen robustness family counts differ: "
            f"expected {expected_family_counts}, observed {actual_family_counts}"
        )

    return {
        "observed_rows": int(len(trials)),
        "failure_rows": int(len(failures)),
        "unavailable_rows": int(len(unavailable)),
        "planned_rows": actual_total,
        "success_rows": int(recovery.sum()),
        "recovery_failure_rows": int((1 - recovery).sum()),
        "execution_failure_rows": 0,
        "family_counts": actual_family_counts,
        "unavailable_rows_are_not_recovery_failures": True,
    }


def summarize_recovery_conditions(
    trials: pd.DataFrame,
    unavailable: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize recovery with source-cover grouped percentile intervals."""

    observed = _condition_key_frame(trials, group_columns)
    missing = _condition_key_frame(unavailable, group_columns)
    observed_groups = {
        tuple(keys if isinstance(keys, tuple) else (keys,)): cell
        for keys, cell in observed.groupby(list(group_columns), sort=True, dropna=False)
    }
    unavailable_groups = {
        tuple(keys if isinstance(keys, tuple) else (keys,)): cell
        for keys, cell in missing.groupby(list(group_columns), sort=True, dropna=False)
    }
    rows: list[dict[str, Any]] = []
    for keys in sorted(set(observed_groups) | set(unavailable_groups)):
        cell = observed_groups.get(keys, observed.iloc[0:0])
        unavailable_cell = unavailable_groups.get(keys, missing.iloc[0:0])
        row = dict(zip(group_columns, keys))
        n_observed = int(len(cell))
        n_unavailable = int(len(unavailable_cell))
        source_units = int(cell["source_trial_id"].nunique()) if n_observed else 0
        successes = (
            int(_binary(cell["exact_payload_recovery"], label="recovery").sum())
            if n_observed
            else 0
        )
        row.update(
            {
                "observed_outcome_rows": n_observed,
                "unavailable_rows": n_unavailable,
                "planned_rows": n_observed + n_unavailable,
                "source_cover_units": source_units,
                "success_outcome_rows": successes,
                "failure_outcome_rows": n_observed - successes,
                "row_level_recovery_rate": (
                    successes / n_observed if n_observed else np.nan
                ),
                "unavailable_not_counted_as_failure": True,
                "analysis_unit": "source_trial_id",
                "confidence_level": confidence_level,
                "interval_method": (
                    "pending_source_cover_interval_selection"
                    if n_observed
                    else "not_estimable_all_unavailable"
                ),
                "status": (
                    "observed_with_partial_unavailability"
                    if n_observed and n_unavailable
                    else "observed"
                    if n_observed
                    else "unavailable"
                ),
            }
        )
        if n_observed:
            values = _binary(cell["exact_payload_recovery"], label="recovery")
            unit_values = pd.DataFrame(
                {
                    "value": values.to_numpy(),
                    "source_trial_id": cell["source_trial_id"].astype(str).to_numpy(),
                }
            ).groupby("source_trial_id", sort=True)["value"].mean()
            if unit_values.isin([0.0, 1.0]).all():
                unit_successes = int(unit_values.sum())
                low, high = wilson_interval(
                    unit_successes,
                    len(unit_values),
                    confidence_level=confidence_level,
                )
                row.update(
                    {
                        "recovery_rate": float(unit_values.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "interval_method": "source_cover_wilson",
                        "source_cover_success_units": unit_successes,
                        "bootstrap_resamples_requested": 0,
                        "bootstrap_resamples_valid": 0,
                    }
                )
            else:
                summary = grouped_payload_bootstrap_ci(
                    values.tolist(),
                    cell["source_trial_id"].astype(str).tolist(),
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    seed=_stable_seed(seed, *keys),
                )
                row.update(
                    {
                        "recovery_rate": summary["mean"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "interval_method": (
                            "source_cover_grouped_percentile_bootstrap"
                        ),
                        "source_cover_success_units": np.nan,
                        "bootstrap_resamples_requested": summary[
                            "bootstrap_resamples_requested"
                        ],
                        "bootstrap_resamples_valid": summary[
                            "bootstrap_resamples_valid"
                        ],
                    }
                )
            row["outcome_rows_per_source_cover"] = n_observed / source_units
        else:
            row.update(
                {
                    "recovery_rate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "source_cover_success_units": np.nan,
                    "bootstrap_resamples_requested": 0,
                    "bootstrap_resamples_valid": 0,
                    "outcome_rows_per_source_cover": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_failure_mechanism(row: Mapping[str, Any]) -> tuple[str, str]:
    """Map a saved first-divergence record to a conservative mechanism label."""

    execution_error = row.get("execution_error_type")
    if execution_error is not None and str(execution_error).strip() not in {"", "nan"}:
        return "execution_error", "nonempty execution_error_type"
    family = str(row.get("robustness_family", ""))
    replay_mode = str(row.get("replay_mode", ""))
    transformation = str(row.get("transformation_id", ""))
    if family == "cross_model_mismatch":
        return (
            "model_or_tokenizer_identity_mismatch",
            "cross_model_mismatch family",
        )
    if family == "replay_modes":
        if replay_mode == "detokenized_text_retokenized":
            return (
                "detokenization_retokenization_divergence",
                "detokenized text replay mode",
            )
        if replay_mode == "greedy_leadin_regeneration":
            return "leadin_regeneration_divergence", "greedy lead-in replay mode"
        return "exact_replay_divergence", f"replay mode {replay_mode}"
    if transformation == "paraphrase":
        return "semantic_rewrite_divergence", "paraphrase transformation"
    if transformation == "truncation":
        return "truncation_or_boundary_loss", "truncation transformation"
    if transformation in {
        "character_insertion",
        "character_deletion",
        "character_substitution",
        "token_deletion",
    }:
        return "content_edit_tokenization_divergence", transformation
    if transformation == "unicode_normalization":
        return (
            "unicode_normalization_tokenization_divergence",
            transformation,
        )
    if transformation == "quote_conversion":
        return "quotation_tokenization_divergence", transformation
    if transformation in {
        "line_endings",
        "whitespace_trim",
        "whitespace_collapse",
        "markdown_copy_paste",
    }:
        return "whitespace_or_markup_tokenization_divergence", transformation
    return "other_recorded_replay_divergence", transformation or family


def build_failure_taxonomy(
    failures: pd.DataFrame, trials: pd.DataFrame
) -> pd.DataFrame:
    """Attach condition metadata and descriptive first-divergence flags."""

    condition_columns = [
        "trial_id",
        "alphabet_size_B",
        "exact_payload_recovery",
    ]
    for column in (
        "prompt_category",
        "payload_class",
        "mitigation_id",
    ):
        if column in trials.columns:
            condition_columns.append(column)
    joined = failures.merge(
        trials[condition_columns], on="trial_id", how="left", validate="one_to_one"
    )
    if joined["exact_payload_recovery"].isna().any():
        raise RobustnessAnalysisError(
            "Failure rows could not be joined one-to-one to observed trials"
        )
    if _binary(joined["exact_payload_recovery"], label="failed trial recovery").any():
        raise RobustnessAnalysisError("A failure record is attached to a successful trial")
    mechanisms = joined.apply(classify_failure_mechanism, axis=1)
    joined["failure_mechanism"] = [value[0] for value in mechanisms]
    joined["mechanism_basis"] = [value[1] for value in mechanisms]
    recovered_rank = pd.to_numeric(joined["recovered_rank"], errors="coerce")
    expected_rank = pd.to_numeric(joined["expected_rank"], errors="coerce")
    alphabet_size = pd.to_numeric(joined["alphabet_size_B"], errors="coerce")
    joined["recovered_rank_out_of_bound"] = (
        recovered_rank.notna()
        & alphabet_size.notna()
        & ((recovered_rank < 1) | (recovered_rank > alphabet_size))
    )
    joined["expected_rank_out_of_bound"] = (
        expected_rank.notna()
        & alphabet_size.notna()
        & ((expected_rank < 1) | (expected_rank > alphabet_size))
    )
    if joined["expected_rank_out_of_bound"].any():
        raise RobustnessAnalysisError("An expected encoded rank is out of bound")
    for prefix, left, right in (
        ("token", "expected_token_length", "recovered_token_length"),
        ("rank", "expected_rank_length", "recovered_rank_length"),
    ):
        if left in joined.columns and right in joined.columns:
            joined[f"{prefix}_length_changed"] = pd.to_numeric(
                joined[left], errors="coerce"
            ).ne(pd.to_numeric(joined[right], errors="coerce"))
        else:
            joined[f"{prefix}_length_changed"] = False
    joined["mechanism_scope"] = (
        "descriptive_first_divergence_not_causal_proof"
    )
    preferred = [
        "failure_id",
        "trial_id",
        "source_trial_id",
        "robustness_family",
        "replay_mode",
        "transformation_id",
        "mitigation_id",
        "source_model_id",
        "model_id",
        "payload_name",
        "payload_class",
        "prompt_category",
        "segment_index",
        "failure_category",
        "failure_mechanism",
        "mechanism_basis",
        "mechanism_scope",
        "first_differing_position",
        "expected_token_id",
        "recovered_token_id",
        "expected_rank",
        "recovered_rank",
        "alphabet_size_B",
        "recovered_rank_out_of_bound",
        "token_length_changed",
        "rank_length_changed",
        "context_sha256",
        "boundary_start_offset",
        "boundary_end_offset",
        "divergence_fields_availability",
    ]
    return joined[[column for column in preferred if column in joined.columns]].copy()


def summarize_failure_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "robustness_family",
        "replay_mode",
        "transformation_id",
        "failure_mechanism",
    ]
    rows: list[dict[str, Any]] = []
    for keys, cell in taxonomy.groupby(group_columns, sort=True, dropna=False):
        first = pd.to_numeric(cell["first_differing_position"], errors="coerce")
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "failure_rows": int(len(cell)),
                "source_cover_units": int(cell["source_trial_id"].nunique()),
                "payload_units": int(cell["payload_name"].nunique()),
                "recovered_rank_out_of_bound_rows": int(
                    cell["recovered_rank_out_of_bound"].sum()
                ),
                "token_length_changed_rows": int(cell["token_length_changed"].sum()),
                "rank_length_changed_rows": int(cell["rank_length_changed"].sum()),
                "first_difference_recorded_rows": int(first.notna().sum()),
                "first_difference_median": (
                    float(first.median()) if first.notna().any() else np.nan
                ),
                "first_difference_q25": (
                    float(first.quantile(0.25)) if first.notna().any() else np.nan
                ),
                "first_difference_q75": (
                    float(first.quantile(0.75)) if first.notna().any() else np.nan
                ),
                "first_difference_max": (
                    float(first.max()) if first.notna().any() else np.nan
                ),
                "mechanism_scope": "descriptive_first_divergence_not_causal_proof",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_unavailable(unavailable: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "robustness_family",
        "replay_mode",
        "transformation_id",
        "source_model_id",
        "model_id",
        "reason_code",
        "root_condition_reason_code",
    ]
    available_columns = [column for column in group_columns if column in unavailable]
    rows: list[dict[str, Any]] = []
    for keys, cell in unavailable.groupby(
        available_columns, sort=True, dropna=False
    ):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(available_columns, key_tuple))
        row.update(
            {
                "unavailable_rows": int(len(cell)),
                "source_cover_units": int(cell["source_trial_id"].nunique()),
                "payload_units": int(cell["payload_name"].nunique())
                if "payload_name" in cell
                else np.nan,
                "excluded_from_estimands": True,
                "counted_as_recovery_failure": False,
                "taxonomy_class": "condition_unavailable_not_recovery_failure",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _atomic_write_csv(frame: pd.DataFrame, target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)
    return target


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def build_robustness_artifacts(
    *,
    trials_path: str | Path,
    failures_path: str | Path,
    unavailable_path: str | Path,
    robustness_config: str | Path,
    statistics_config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> RobustnessArtifacts:
    """Validate saved robustness evidence and emit compact derived artifacts."""

    trials = _read_csv(trials_path, label="robustness trials")
    failures = _read_csv(failures_path, label="robustness failures")
    unavailable = _read_csv(unavailable_path, label="robustness unavailable rows")
    robustness = _read_json(robustness_config, label="robustness config")
    statistics = _read_json(statistics_config, label="statistics config")
    intervals = statistics.get("intervals", {})
    confidence_level = float(intervals.get("confidence_level", 0.95))
    n_resamples = int(intervals.get("bootstrap_resamples", 2_000))
    seed = int(intervals.get("bootstrap_seed", 2_026_080_801))
    if not 0 < confidence_level < 1 or n_resamples <= 0:
        raise RobustnessAnalysisError("Invalid frozen interval configuration")

    validated_summary = validate_robustness_inputs(
        trials, failures, unavailable, robustness
    )
    condition_summary = summarize_recovery_conditions(
        trials,
        unavailable,
        group_columns=CONDITION_COLUMNS,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    model_summary = summarize_recovery_conditions(
        trials,
        unavailable,
        group_columns=MODEL_CONDITION_COLUMNS,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    taxonomy = build_failure_taxonomy(failures, trials)
    failure_summary = summarize_failure_taxonomy(taxonomy)
    unavailable_summary = summarize_unavailable(unavailable)
    plot_columns = [
        *CONDITION_COLUMNS,
        "observed_outcome_rows",
        "unavailable_rows",
        "source_cover_units",
        "recovery_rate",
        "ci_low",
        "ci_high",
        "status",
    ]
    plot_source = condition_summary[plot_columns].copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise RobustnessAnalysisError(
            "Refusing to overwrite robustness outputs: "
            + ", ".join(str(path) for path in existing)
        )
    frames = {
        "conditions": condition_summary,
        "models": model_summary,
        "failure_taxonomy": taxonomy,
        "failure_summary": failure_summary,
        "unavailable": unavailable_summary,
        "plot_source": plot_source,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, targets[key])

    inputs = {}
    for key, raw_path in (
        ("trials", trials_path),
        ("failures", failures_path),
        ("unavailable", unavailable_path),
        ("robustness_config", robustness_config),
        ("statistics_config", statistics_config),
    ):
        resolved = Path(raw_path).resolve()
        inputs[key] = {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-revision-robustness-analysis-v1",
        "status": "passed",
        "inputs": inputs,
        "outputs": {},
        "summary": validated_summary,
        "inference": {
            "analysis_unit": "source_trial_id",
            "confidence_level": confidence_level,
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": seed,
            "unavailable_rows_counted_as_failures": False,
        },
        "failure_taxonomy_scope": (
            "descriptive_first_divergence_not_causal_proof"
        ),
    }
    for key, target in targets.items():
        if key == "manifest":
            continue
        manifest["outputs"][key] = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
            "row_count": int(len(frames[key])),
        }
    _atomic_write_json(manifest, targets["manifest"])
    return RobustnessArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=validated_summary,
    )
