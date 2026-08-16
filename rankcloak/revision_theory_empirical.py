"""Empirical residual and uncertainty summaries for capacity--quality theory.

The companion builder reuses the exact theory calculations in
``revision_theory`` and adds condition identities, empirical residuals, and
payload-clustered confidence intervals.  It consumes saved records only and
never imputes absent probabilities or replay traces.
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

from rankcloak.revision_theory import (
    capacity_validation_rows,
    quality_validation_rows,
    read_trial_records,
)


OUTPUT_FILENAMES = {
    "validation": "theory_empirical_validation.csv",
    "summary": "theory_empirical_summary.csv",
    "plot_source": "theory_residual_plot_source.csv",
    "assumptions": "theory_assumptions.json",
    "technical_note": "TECHNICAL_NOTE.md",
    "manifest": "theory_empirical_manifest.json",
}

KNOWN_STAGES = ("primary_v2", "ablation_v2", "multilingual_v2", "robustness_v2")
GROUP_COLUMNS = (
    "source_stage",
    "study_phase",
    "language",
    "model_id",
    "protocol_variant",
    "representation_name",
    "alphabet_size_B",
    "segmented",
    "tail_policy",
    "token_filter",
    "leadin_tokens",
    "topic_schedule",
)
OUTCOMES = {
    "forced_position_residual_tokens": "capacity_residual",
    "tail_overhead_tokens": "capacity_tail_overhead",
    "cover_length_residual_tokens": "capacity_total_overhead",
    "observed_rate_bits_per_forced_token": "capacity_rate",
    "R_effective_bits_per_forced_plus_tail_token": "capacity_effective_rate",
    "code_space_utilization": "capacity_code_space",
    "unused_codeword_fraction": "capacity_code_space",
    "Q_B_nats_per_forced_token": "quality_surprisal",
    "Q_greedy_nats_per_forced_token": "quality_greedy_endpoint",
    "Q_rank_B_nats_per_forced_token": "quality_rank_B_endpoint",
    "Delta_B_nats_per_forced_token": "quality_penalty",
    "minimum_lower_margin_nats": "quality_bound_margin",
    "minimum_upper_margin_nats": "quality_bound_margin",
}
VALIDATION_COLUMNS = (
    "source_file",
    "source_row",
    "source_stage",
    "study_phase",
    "record_type",
    "trial_id",
    "model_id",
    "protocol_variant",
    "payload_name",
    "payload_class",
    "language",
    "representation_name",
    "alphabet_size_B",
    "segmented",
    "segment_count",
    "tail_policy",
    "token_filter",
    "leadin_tokens",
    "topic_schedule",
    "capacity_status",
    "quality_status",
    "quality_evidence_level",
    "H_bits",
    "theoretical_n_B",
    "observed_n_forced",
    "observed_n_tail",
    "tail_overhead_tokens",
    "forced_position_residual_tokens",
    "observed_total_cover_tokens",
    "cover_length_residual_tokens",
    "R_B_bits_per_forced_token",
    "observed_rate_bits_per_forced_token",
    "R_effective_bits_per_forced_plus_tail_token",
    "rate_upper_bound_bits_per_forced_token",
    "rate_bound_holds",
    "observed_forced_count_feasible",
    "code_space_slack_bits",
    "literal_padding_bits",
    "code_space_utilization",
    "unused_codeword_fraction",
    "forced_context_count",
    "Q_B_nats_per_forced_token",
    "Q_greedy_nats_per_forced_token",
    "Q_rank_B_nats_per_forced_token",
    "Delta_B_nats_per_forced_token",
    "greedy_lower_bound_holds_per_context",
    "rank_B_upper_bound_holds_per_context",
    "minimum_lower_margin_nats",
    "minimum_upper_margin_nats",
)
SUMMARY_COLUMNS = (
    "evidence_domain",
    *GROUP_COLUMNS,
    "outcome",
    "group_record_count",
    "n",
    "missing_n",
    "payload_units",
    "mean",
    "standard_deviation",
    "median",
    "minimum",
    "maximum",
    "ci_low",
    "ci_high",
    "confidence_level",
    "interval_method",
    "bootstrap_resamples",
    "bootstrap_seed",
)


class EmpiricalTheoryError(ValueError):
    """Raised when empirical theory inputs or identities are inconsistent."""


@dataclass(frozen=True)
class EmpiricalTheoryArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise EmpiricalTheoryError(f"Missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EmpiricalTheoryError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EmpiricalTheoryError(f"{label} must contain a JSON object")
    return value


def _source_stage(source_file: Any) -> str:
    parts = Path(str(source_file or "")).parts
    for stage in KNOWN_STAGES:
        if stage in parts:
            return stage
    return "unspecified"


def _number(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _condition_value(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    if value is None and name == "leadin_tokens":
        value = record.get("leadin_token_count")
    if value is None or value == "":
        return "not_applicable"
    return value


def empirical_validation_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join theory outputs to saved condition identities and derive residuals."""

    capacity = capacity_validation_rows(records)
    quality = quality_validation_rows(records)
    if not (len(records) == len(capacity) == len(quality)):
        raise EmpiricalTheoryError("Theory validators changed record ordering or counts")
    rows: list[dict[str, Any]] = []
    for record, capacity_row, quality_row in zip(records, capacity, quality):
        theoretical = _number(capacity_row.get("theoretical_n_B"))
        forced = _number(capacity_row.get("observed_n_forced"))
        tail = _number(capacity_row.get("observed_n_tail"))
        forced_residual = (
            forced - theoretical
            if np.isfinite(forced) and np.isfinite(theoretical)
            else np.nan
        )
        total_cover = forced + tail if np.isfinite(forced) and np.isfinite(tail) else np.nan
        cover_residual = (
            total_cover - theoretical
            if np.isfinite(total_cover) and np.isfinite(theoretical)
            else np.nan
        )
        row: dict[str, Any] = {
            "source_file": capacity_row.get("source_file"),
            "source_row": capacity_row.get("source_row"),
            "source_stage": _source_stage(capacity_row.get("source_file")),
            "study_phase": _condition_value(record, "study_phase"),
            "record_type": _condition_value(record, "record_type"),
            "trial_id": capacity_row.get("trial_id"),
            "model_id": capacity_row.get("model_id"),
            "protocol_variant": capacity_row.get("protocol_variant") or "not_applicable",
            "payload_name": capacity_row.get("payload_name"),
            "payload_class": _condition_value(record, "payload_class"),
            "language": _condition_value(record, "language"),
            "representation_name": capacity_row.get("representation_name") or "not_applicable",
            "alphabet_size_B": capacity_row.get("alphabet_size_B"),
            "segmented": _condition_value(record, "segmented"),
            "segment_count": record.get("segment_count"),
            "tail_policy": _condition_value(record, "tail_policy"),
            "token_filter": _condition_value(record, "token_filter"),
            "leadin_tokens": _condition_value(record, "leadin_tokens"),
            "topic_schedule": _condition_value(record, "topic_schedule"),
            "capacity_status": capacity_row.get("capacity_status"),
            "quality_status": quality_row.get("quality_status"),
            "quality_evidence_level": quality_row.get("quality_evidence_level"),
            "tail_overhead_tokens": tail,
            "forced_position_residual_tokens": forced_residual,
            "observed_total_cover_tokens": total_cover,
            "cover_length_residual_tokens": cover_residual,
        }
        for column in VALIDATION_COLUMNS:
            if column in row:
                continue
            if column in quality_row and quality_row.get(column) is not None:
                row[column] = quality_row.get(column)
            else:
                row[column] = capacity_row.get(column)
        rows.append({column: row.get(column) for column in VALIDATION_COLUMNS})
    return rows


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def summarize_empirical_theory(
    validation: pd.DataFrame,
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize outcomes using payload names as the resampling clusters."""

    if not 0.0 < confidence_level < 1.0 or bootstrap_resamples <= 0:
        raise EmpiricalTheoryError("Invalid empirical interval configuration")
    frame = validation.copy()
    for column in GROUP_COLUMNS:
        frame[column] = frame[column].fillna("not_applicable").astype(str)
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - confidence_level) / 2.0
    for keys, cell in frame.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(GROUP_COLUMNS, key_values))
        payload_fallback = cell["trial_id"].fillna(cell["source_row"].astype(str))
        clusters = cell["payload_name"].fillna(payload_fallback).astype(str)
        for outcome, domain in OUTCOMES.items():
            values = pd.to_numeric(cell[outcome], errors="coerce")
            finite_mask = np.isfinite(values.to_numpy(dtype=float))
            finite = values.loc[finite_mask]
            if finite.empty:
                continue
            cluster_values: dict[str, np.ndarray] = {}
            for cluster in sorted(clusters.loc[finite_mask].unique()):
                array = values.loc[finite_mask & clusters.eq(cluster)].to_numpy(dtype=float)
                cluster_values[cluster] = array[np.isfinite(array)]
            labels = sorted(cluster_values)
            estimates: list[float] = []
            if len(labels) > 1:
                rng = np.random.default_rng(
                    _stable_seed(bootstrap_seed, *key_values, outcome)
                )
                sums = np.asarray([cluster_values[label].sum() for label in labels])
                counts = np.asarray([len(cluster_values[label]) for label in labels])
                for _ in range(bootstrap_resamples):
                    draw = rng.integers(0, len(labels), size=len(labels))
                    estimates.append(float(sums[draw].sum() / counts[draw].sum()))
            rows.append(
                {
                    "evidence_domain": domain,
                    **identity,
                    "outcome": outcome,
                    "group_record_count": int(len(cell)),
                    "n": int(len(finite)),
                    "missing_n": int(len(cell) - len(finite)),
                    "payload_units": len(labels),
                    "mean": float(finite.mean()),
                    "standard_deviation": (
                        float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
                    ),
                    "median": float(finite.median()),
                    "minimum": float(finite.min()),
                    "maximum": float(finite.max()),
                    "ci_low": (
                        float(np.quantile(estimates, alpha)) if estimates else np.nan
                    ),
                    "ci_high": (
                        float(np.quantile(estimates, 1.0 - alpha))
                        if estimates else np.nan
                    ),
                    "confidence_level": confidence_level,
                    "interval_method": "payload_cluster_percentile_bootstrap",
                    "bootstrap_resamples": bootstrap_resamples,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


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


def _atomic_write_text(value: str, target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _technical_note(assumptions: Mapping[str, Any]) -> str:
    equation_lines = [
        f"- `{row['expression']}` ({row['units']}; identifier `{row['id']}`)."
        for row in assumptions["equations"]
    ]
    assumption_lines = [f"- {value}" for value in assumptions["assumptions"]]
    limitation_lines = [f"- {value}" for value in assumptions["limitations"]]
    return "\n".join(
        [
            "# Capacity-quality validation technical note",
            "",
            "Computational evidence artifact only; this is not manuscript text.",
            "",
            "## Defined equations",
            "",
            *equation_lines,
            "",
            "## Validation assumptions",
            "",
            *assumption_lines,
            "",
            "## Empirical validation contract",
            "",
            "- `theory_empirical_validation.csv` retains trial and condition identities and records capacity residuals, tail overhead, rates, same-context surprisal bounds, and missing endpoints.",
            "- `theory_empirical_summary.csv` reports payload-clustered percentile-bootstrap intervals; token positions are not treated as independent observations.",
            "- `theory_residual_plot_source.csv` is the figure source. Raw record files are referenced by hash and are not copied into this package.",
            "",
            "## Known limits",
            "",
            *limitation_lines,
            "",
        ]
    )


def _input_entry(path: str | Path, *, record_count: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    entry: dict[str, Any] = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if record_count is not None:
        entry["record_count"] = record_count
    return entry


def build_empirical_theory_artifacts(
    *,
    input_paths: Sequence[str | Path],
    statistics_config: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> EmpiricalTheoryArtifacts:
    """Build residual, uncertainty, assumption, and provenance artifacts."""

    resolved_inputs = [Path(path).resolve() for path in input_paths]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise EmpiricalTheoryError("Theory input paths must be unique")
    statistics = _read_json(statistics_config, label="statistics config")
    interval = statistics.get("intervals", {})
    confidence_level = float(interval.get("confidence_level", 0.95))
    bootstrap_resamples = int(interval.get("bootstrap_resamples", 2_000))
    bootstrap_seed = int(interval.get("bootstrap_seed", 2_026_080_801))
    try:
        records, sources = read_trial_records(resolved_inputs)
    except Exception as exc:
        raise EmpiricalTheoryError(f"Could not read theory records: {exc}") from exc
    validation_rows = empirical_validation_rows(records)
    validation = pd.DataFrame(validation_rows, columns=VALIDATION_COLUMNS)
    empirical_summary = summarize_empirical_theory(
        validation,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    capacity_plot = validation.loc[
        pd.to_numeric(validation["theoretical_n_B"], errors="coerce").notna(),
        [
            *GROUP_COLUMNS,
            "trial_id",
            "payload_name",
            "payload_class",
            "H_bits",
            "theoretical_n_B",
            "observed_n_forced",
            "observed_n_tail",
            "forced_position_residual_tokens",
            "cover_length_residual_tokens",
            "observed_rate_bits_per_forced_token",
            "R_effective_bits_per_forced_plus_tail_token",
            "Q_B_nats_per_forced_token",
            "Delta_B_nats_per_forced_token",
        ],
    ].copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise EmpiricalTheoryError(
            "Refusing to overwrite empirical theory outputs: "
            + ", ".join(str(path) for path in existing)
        )
    frames = {
        "validation": validation,
        "summary": empirical_summary,
        "plot_source": capacity_plot,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, targets[key])

    assumptions = {
        "schema_version": "rankcloak-capacity-quality-assumptions-v1",
        "status": "technical_evidence_not_manuscript_text",
        "equations": [
            {
                "id": "minimum_forced_positions",
                "expression": "n_B = ceil(H / log2(B))",
                "units": "token_positions",
            },
            {
                "id": "nominal_rate",
                "expression": "R_B = H / n_B",
                "units": "bits_per_forced_token",
            },
            {
                "id": "effective_rate_with_tail",
                "expression": "R_effective = H / (n_forced + n_tail)",
                "units": "bits_per_generated_token",
            },
            {
                "id": "realized_surprisal",
                "expression": "Q_B = -mean_t(log p(y_t | context_t))",
                "units": "nats_per_forced_token",
            },
            {
                "id": "quality_penalty",
                "expression": "Delta_B = Q_B - Q_greedy",
                "units": "nats_per_forced_token",
            },
            {
                "id": "same_context_rank_bounds",
                "expression": "Q_greedy <= Q_B <= Q_rank_B",
                "units": "nats_per_forced_token",
            },
        ],
        "assumptions": [
            "H is the saved representation-source bit count for the evaluated codec.",
            "B is the saved admissible rank bound and ranks are one-indexed.",
            "Tail tokens carry no additional payload bits in the stated effective-rate calculation.",
            "Quality endpoint inequalities require probabilities evaluated under identical saved contexts.",
            "Missing endpoint arrays are unavailable and are not reconstructed from aggregate means.",
            "Confidence intervals resample payload identities, not individual tokens.",
        ],
        "limitations": [
            "Observed rank replay does not prove the deterministic replay proposition when complete ranked token orders were not saved.",
            "The empirical summary describes the frozen saved corpus and does not identify causal quality effects.",
        ],
    }
    _atomic_write_json(assumptions, targets["assumptions"])
    _atomic_write_text(_technical_note(assumptions), targets["technical_note"])

    numeric_forced_residual = pd.to_numeric(
        validation["forced_position_residual_tokens"], errors="coerce"
    )
    numeric_cover_residual = pd.to_numeric(
        validation["cover_length_residual_tokens"], errors="coerce"
    )
    summary = {
        "input_record_count": len(records),
        "capacity_evaluable_count": int(numeric_forced_residual.notna().sum()),
        "forced_position_residual_nonzero_count": int(
            numeric_forced_residual.dropna().ne(0).sum()
        ),
        "cover_length_residual_positive_count": int(
            numeric_cover_residual.dropna().gt(0).sum()
        ),
        "rate_bound_failure_count": int(
            validation["rate_bound_holds"].eq(False).sum()
        ),
        "quality_fully_bound_validated_count": int(
            validation["quality_status"].eq("validated").sum()
        ),
        "quality_failed_check_count": int(
            validation["quality_status"].astype(str).str.startswith("failed").sum()
        ),
        "empirical_summary_rows": len(empirical_summary),
    }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-empirical-theory-evidence-v1",
        "status": "passed",
        "inputs": {
            "records": sources,
            "statistics_config": _input_entry(statistics_config),
        },
        "outputs": {},
        "summary": summary,
        "missing_value_policy": "no_imputation_no_endpoint_reconstruction",
        "resampling_unit": "payload_name",
        "confidence_level": confidence_level,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "raw_record_files_copied": False,
    }
    for key, target in targets.items():
        if key == "manifest":
            continue
        entry = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        if key in frames:
            entry["row_count"] = int(len(frames[key]))
        manifest["outputs"][key] = entry
    _atomic_write_json(manifest, targets["manifest"])
    return EmpiricalTheoryArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=summary,
    )
