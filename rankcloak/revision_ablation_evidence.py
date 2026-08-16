"""Evidence-only summaries of frozen ablation levels versus the canonical run.

The generation design is confirmatory and immutable.  This extraction is
explicitly exploratory because its cross-level summary was specified after
saved outcomes were available.  Comparisons use payload groups, never tokens
or nested segments, and restrict each canonical comparison to the model set
available for the corresponding ablation level.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .revision_statistics import (
    adjust_pvalues,
    grouped_payload_bootstrap_ci,
    pairwise_effect_sizes,
    wilson_interval,
)


SCHEMA_VERSION = "rankcloak-revision-ablation-evidence-analysis-v1"
MANIFEST_SCHEMA = "rankcloak-revision-ablation-evidence-manifest-v1"
OUTPUT_FILENAMES = {
    "configurations": "ablation_configuration_summary.csv",
    "continuous": "ablation_continuous_summary.csv",
    "contrasts": "ablation_canonical_contrasts.csv",
    "unavailable": "ablation_unavailable_summary.csv",
    "manifest": "ablation_evidence_manifest.json",
}


class AblationEvidenceError(ValueError):
    """Raised when frozen ablation evidence cannot be summarized safely."""


@dataclass(frozen=True)
class AblationEvidenceArtifacts:
    output_dir: str
    manifest_path: str
    configuration_rows: int
    contrast_rows: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AblationEvidenceError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AblationEvidenceError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AblationEvidenceError(f"{label} must contain an object")
    return value


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise AblationEvidenceError(f"Missing or unsafe {label}: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise AblationEvidenceError(f"Could not read {label}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AblationEvidenceError(
            f"{label} lacks required columns: {', '.join(missing)}"
        )


def _text_levels(values: pd.Series) -> set[str]:
    return {str(value) for value in values.dropna().tolist()}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _identity(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if rows is not None:
        result["row_count"] = int(rows)
    return result


def _configuration_row(
    cell: pd.DataFrame,
    *,
    factor: str,
    level: str,
    canonical: bool,
    confidence_level: float,
    unavailable_rows: int,
) -> dict[str, Any]:
    numeric = pd.to_numeric(cell["exact_payload_recovery"], errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise AblationEvidenceError("Ablation payload recovery is not binary")
    strict = cell.assign(_recovery=numeric).groupby("payload_name")["_recovery"].min()
    successes = int(strict.sum())
    total = int(len(strict))
    low, high = wilson_interval(successes, total, confidence_level=confidence_level)
    return {
        "factor": factor,
        "level": level,
        "canonical_reference": bool(canonical),
        "observed_trial_rows": int(len(cell)),
        "model_count": int(cell["model_id"].nunique()),
        "payload_groups": total,
        "strict_payload_successes": successes,
        "strict_payload_recovery_rate": successes / total,
        "wilson_ci_low": low,
        "wilson_ci_high": high,
        "confidence_level": confidence_level,
        "unavailable_work_units": int(unavailable_rows),
        "unavailable_not_counted_as_failure": True,
        "analysis_unit": "payload_name",
    }


def _continuous_rows(
    cell: pd.DataFrame,
    *,
    factor: str,
    level: str,
    canonical: bool,
    outcomes: list[str],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        values = pd.to_numeric(cell[outcome], errors="coerce")
        valid = cell.loc[np.isfinite(values)].copy()
        valid[outcome] = values.loc[valid.index].astype(float)
        if valid.empty:
            raise AblationEvidenceError(f"Ablation outcome is empty: {outcome}")
        summary = grouped_payload_bootstrap_ci(
            valid[outcome].tolist(),
            valid["payload_name"].astype(str).tolist(),
            confidence_level=confidence_level,
            n_resamples=resamples,
            seed=seed,
        )
        payload_means = valid.groupby("payload_name", sort=True)[outcome].mean()
        rows.append(
            {
                "factor": factor,
                "level": level,
                "canonical_reference": bool(canonical),
                "outcome": outcome,
                "observed_trial_rows": int(len(valid)),
                "model_count": int(valid["model_id"].nunique()),
                "payload_groups": int(len(payload_means)),
                "mean": summary["mean"],
                "standard_deviation": float(payload_means.std(ddof=1)),
                "median": float(payload_means.median()),
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "bootstrap_resamples_requested": int(resamples),
                "bootstrap_resamples_valid": summary[
                    "bootstrap_resamples_valid"
                ],
                "confidence_level": confidence_level,
                "analysis_unit": "payload_name",
                "evidence_status": "exploratory_post_outcome_evidence_extraction",
            }
        )
    return rows


def _contrast_row(
    canonical: pd.DataFrame,
    level_cell: pd.DataFrame,
    *,
    factor: str,
    level: str,
    outcome: str,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    models = sorted(_text_levels(level_cell["model_id"]))
    baseline = canonical[canonical["model_id"].astype(str).isin(models)].copy()
    keys = ["model_id", "payload_name", "prompt_id"]
    baseline_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    level_keys = set(map(tuple, level_cell[keys].astype(str).to_numpy()))
    if baseline_keys != level_keys or not baseline_keys:
        raise AblationEvidenceError(
            f"Canonical pairing keys differ for {factor}={level}, outcome={outcome}"
        )
    left = baseline[["payload_name", outcome]].copy()
    left["_comparison_condition"] = "0_canonical"
    right = level_cell[["payload_name", outcome]].copy()
    right["_comparison_condition"] = "1_level"
    contrast = pairwise_effect_sizes(
        pd.concat([left, right], ignore_index=True),
        outcome=outcome,
        factor="_comparison_condition",
        payload_column="payload_name",
        binary=False,
        confidence_level=confidence_level,
        n_resamples=resamples,
        seed=seed,
    )
    if len(contrast) != 1:
        raise AblationEvidenceError(
            f"Expected one canonical contrast for {factor}={level}, {outcome}"
        )
    source = contrast.iloc[0]
    raw_p = source.get("p_value_raw")
    return {
        "factor": factor,
        "level": level,
        "canonical_value": "canonical",
        "outcome": outcome,
        "shared_models": ";".join(models),
        "shared_model_count": len(models),
        "paired_payload_groups": int(source["n_payloads_paired"]),
        "canonical_mean": float(source["mean_first"]),
        "level_mean": float(source["mean_second"]),
        "level_minus_canonical": -float(source["mean_difference"]),
        "ci_low": -float(source["mean_difference_ci_high"]),
        "ci_high": -float(source["mean_difference_ci_low"]),
        "hedges_g_level_minus_canonical": (
            -float(source["hedges_g"])
            if math.isfinite(float(source.get("hedges_g", float("nan"))))
            else np.nan
        ),
        "p_value_raw": (
            float(raw_p) if raw_p is not None and math.isfinite(float(raw_p)) else np.nan
        ),
        "p_value_holm": np.nan,
        "p_value_bh": np.nan,
        "test": str(source["test"]),
        "inferential_p_value_supported": bool(
            source["inferential_p_value_supported"]
        ),
        "bootstrap_unit": "payload_name",
        "bootstrap_resamples": int(resamples),
        "confidence_level": confidence_level,
        "primary_inference": False,
        "evidence_status": "exploratory_post_outcome_evidence_extraction",
    }


def build_ablation_evidence(
    *,
    trials_path: str | Path,
    unavailable_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> AblationEvidenceArtifacts:
    trials_file = Path(trials_path).resolve()
    unavailable_file = Path(unavailable_path).resolve()
    config_file = Path(config_path).resolve()
    trials = _read_csv(trials_file, label="ablation trials")
    unavailable = _read_csv(unavailable_file, label="ablation unavailable records")
    config = _read_json(config_file, label="ablation evidence config")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("analysis_status")
        != "exploratory_post_outcome_evidence_extraction"
        or config.get("confirmatory_generation_design_unchanged") is not True
        or config.get("outcomes_were_available_before_analysis_specification")
        is not True
    ):
        raise AblationEvidenceError("Ablation analysis disclosure differs")
    expected = config.get("expected_input", {})
    factors = config.get("factors")
    outcomes = list(config.get("continuous_outcomes", []))
    inference = config.get("inference", {})
    if not isinstance(factors, list) or not factors or not outcomes:
        raise AblationEvidenceError("Ablation factor or outcome specification is empty")
    required = [
        "trial_id",
        "model_id",
        "payload_name",
        "prompt_id",
        "ablation_factor",
        "ablation_level",
        "exact_payload_recovery",
        *outcomes,
    ]
    _require_columns(trials, required, label="ablation trials")
    _require_columns(
        unavailable,
        ["model_id", "payload_name", "ablation_factor", "ablation_level", "reason_code"],
        label="ablation unavailable records",
    )
    if (
        len(trials) != int(expected.get("trial_rows", -1))
        or len(unavailable) != int(expected.get("unavailable_rows", -1))
        or len(trials) + len(unavailable)
        != int(expected.get("planned_work_units", -1))
        or trials["payload_name"].nunique()
        != int(expected.get("payload_groups", -1))
        or _text_levels(trials["evidence_status"])
        != {str(expected.get("evidence_status"))}
        or _text_levels(trials["study_phase"])
        != {str(expected.get("study_phase"))}
    ):
        raise AblationEvidenceError("Frozen ablation input identity/counts differ")
    if trials.duplicated(["model_id", "payload_name", "prompt_id", "ablation_factor", "ablation_level"]).any():
        raise AblationEvidenceError("Ablation trial pairing identity is duplicated")

    selector = config["canonical_selector"]
    canonical = trials[
        trials["ablation_factor"].astype(str).eq(str(selector["ablation_factor"]))
        & trials["ablation_level"].astype(str).eq(str(selector["ablation_level"]))
    ].copy()
    if canonical.empty:
        raise AblationEvidenceError("Canonical ablation reference is absent")
    confidence = float(inference["confidence_level"])
    resamples = int(inference["bootstrap_resamples"])
    seed = int(inference["bootstrap_seed"])

    configuration_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    expected_factor_levels = {"canonical"}
    for factor_spec in factors:
        factor = str(factor_spec["factor"])
        column = str(factor_spec["condition_column"])
        levels = list(map(str, factor_spec["levels"]))
        expected_factor_levels.add(factor)
        if column not in trials:
            raise AblationEvidenceError(f"Ablation condition column is absent: {column}")
        factor_frame = trials[trials["ablation_factor"].astype(str).eq(factor)]
        observed_levels = _text_levels(factor_frame["ablation_level"])
        if observed_levels != set(levels):
            raise AblationEvidenceError(f"Ablation levels differ for {factor}")
        canonical_value = _text_levels(canonical[column])
        if canonical_value != {str(factor_spec["canonical_value"])}:
            raise AblationEvidenceError(f"Canonical condition value differs for {factor}")
        unavailable_factor = unavailable[
            unavailable["ablation_factor"].astype(str).eq(factor)
        ]
        configuration_rows.append(
            _configuration_row(
                canonical,
                factor=factor,
                level="canonical",
                canonical=True,
                confidence_level=confidence,
                unavailable_rows=0,
            )
        )
        continuous_rows.extend(
            _continuous_rows(
                canonical,
                factor=factor,
                level="canonical",
                canonical=True,
                outcomes=outcomes,
                confidence_level=confidence,
                resamples=resamples,
                seed=seed,
            )
        )
        for level in levels:
            cell = factor_frame[factor_frame["ablation_level"].astype(str).eq(level)].copy()
            if _text_levels(cell[column]) != {level}:
                raise AblationEvidenceError(f"Condition value differs for {factor}={level}")
            missing_count = int(
                unavailable_factor["ablation_level"].astype(str).eq(level).sum()
            )
            configuration_rows.append(
                _configuration_row(
                    cell,
                    factor=factor,
                    level=level,
                    canonical=False,
                    confidence_level=confidence,
                    unavailable_rows=missing_count,
                )
            )
            continuous_rows.extend(
                _continuous_rows(
                    cell,
                    factor=factor,
                    level=level,
                    canonical=False,
                    outcomes=outcomes,
                    confidence_level=confidence,
                    resamples=resamples,
                    seed=seed,
                )
            )
            for outcome in outcomes:
                contrast_rows.append(
                    _contrast_row(
                        canonical,
                        cell,
                        factor=factor,
                        level=level,
                        outcome=outcome,
                        confidence_level=confidence,
                        resamples=resamples,
                        seed=seed,
                    )
                )
    if _text_levels(trials["ablation_factor"]) != expected_factor_levels:
        raise AblationEvidenceError("Unexpected frozen ablation factor is present")

    configurations = pd.DataFrame(configuration_rows).sort_values(["factor", "canonical_reference", "level"], ascending=[True, False, True])
    continuous = pd.DataFrame(continuous_rows).sort_values(["factor", "outcome", "canonical_reference", "level"], ascending=[True, True, False, True])
    contrasts = pd.DataFrame(contrast_rows).sort_values(["outcome", "factor", "level"]).reset_index(drop=True)
    for outcome, indices in contrasts.groupby("outcome", sort=True).groups.items():
        positions = list(indices)
        raw = contrasts.loc[positions, "p_value_raw"].tolist()
        holm = [
            np.nan if value is None else float(value)
            for value in adjust_pvalues(raw, "holm")
        ]
        bh = [
            np.nan if value is None else float(value)
            for value in adjust_pvalues(raw, "bh")
        ]
        contrasts.loc[positions, "p_value_holm"] = holm
        contrasts.loc[positions, "p_value_bh"] = bh
    unavailable_summary = (
        unavailable.groupby(
            ["ablation_factor", "ablation_level", "model_id", "reason_code"],
            dropna=False,
            sort=True,
        )
        .agg(
            unavailable_work_units=("payload_name", "size"),
            payload_groups=("payload_name", "nunique"),
        )
        .reset_index()
    )
    unavailable_summary["excluded_from_estimands"] = True
    unavailable_summary["counted_as_recovery_failure"] = False

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = {key: output / name for key, name in OUTPUT_FILENAMES.items()}
    if any(path.exists() for path in targets.values()) and not overwrite:
        raise AblationEvidenceError("Refusing to overwrite ablation evidence outputs")
    _atomic_csv(configurations, targets["configurations"])
    _atomic_csv(continuous, targets["continuous"])
    _atomic_csv(contrasts, targets["contrasts"])
    _atomic_csv(unavailable_summary, targets["unavailable"])
    unsupported = int((~contrasts["inferential_p_value_supported"]).sum())
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "analysis_status": config["analysis_status"],
        "confirmatory_generation_design_unchanged": True,
        "outcomes_were_available_before_analysis_specification": True,
        "primary_inference": False,
        "inputs": {
            "trials": _identity(trials_file, rows=len(trials)),
            "unavailable": _identity(unavailable_file, rows=len(unavailable)),
            "config": _identity(config_file),
        },
        "inference": inference,
        "outputs": {
            "configurations": _identity(targets["configurations"], rows=len(configurations)),
            "continuous": _identity(targets["continuous"], rows=len(continuous)),
            "contrasts": _identity(targets["contrasts"], rows=len(contrasts)),
            "unavailable": _identity(targets["unavailable"], rows=len(unavailable_summary)),
        },
        "summary": {
            "configuration_rows": len(configurations),
            "continuous_summary_rows": len(continuous),
            "canonical_contrast_rows": len(contrasts),
            "unavailable_summary_rows": len(unavailable_summary),
            "unsupported_numerical_test_rows": unsupported,
            "observed_trial_rows": len(trials),
            "unavailable_work_units": len(unavailable),
        },
        "generation_command": command,
    }
    _atomic_json(manifest, targets["manifest"])
    return AblationEvidenceArtifacts(
        output_dir=str(output),
        manifest_path=str(targets["manifest"]),
        configuration_rows=len(configurations),
        contrast_rows=len(contrasts),
    )
