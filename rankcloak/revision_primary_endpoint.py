"""Hash-bound primary recovery endpoints at trial and repeated-payload units."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "rankcloak-primary-recovery-endpoint-v1"
CONFIG_SCHEMA = "rankcloak-primary-recovery-endpoint-config-v1"
RECOVERY_SEMANTICS = "original_serialized_payload_bytes_sha256_v1"


class PrimaryEndpointError(ValueError):
    """Raised when the primary endpoint inputs or design identity differ."""


@dataclass(frozen=True)
class PrimaryEndpointArtifacts:
    output_dir: str
    manifest_path: str
    endpoint_row_count: int
    payload_row_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrimaryEndpointError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PrimaryEndpointError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PrimaryEndpointError(f"{label} must contain a JSON object")
    return value


def _identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def _binary(series: pd.Series, *, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise PrimaryEndpointError(f"{label} must contain only zero and one")
    return values.astype(int)


def _wilson(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise PrimaryEndpointError("Wilson counts are invalid")
    # The authorized configuration fixes 95%; this avoids an optional SciPy
    # dependency while retaining the same double-precision calculation as R.
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-15):
        raise PrimaryEndpointError("Primary endpoint confidence level must be 0.95")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    centre = estimate + z * z / (2.0 * total)
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    )
    return (
        max(0.0, (centre - radius) / denominator),
        min(1.0, (centre + radius) / denominator),
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _endpoint_row(
    *,
    endpoint_id: str,
    analysis_unit: str,
    evidence_status: str,
    successes: int,
    total: int,
    minimum_trials: int,
    maximum_trials: int,
    confidence: float,
) -> dict[str, Any]:
    low, high = _wilson(successes, total, confidence)
    return {
        "endpoint_id": endpoint_id,
        "analysis_unit": analysis_unit,
        "evidence_status": evidence_status,
        "success_definition": (
            "exact original serialized payload recovery in every row belonging "
            "to the declared analysis unit"
        ),
        "successes": int(successes),
        "total_units": int(total),
        "estimate": float(successes / total),
        "confidence_level": confidence,
        "wilson_ci_low": low,
        "wilson_ci_high": high,
        "minimum_primary_trials_per_unit": int(minimum_trials),
        "maximum_primary_trials_per_unit": int(maximum_trials),
        "recovery_outcome": "exact_payload_recovery",
        "recovery_outcome_semantics": RECOVERY_SEMANTICS,
        "exact_rank_replay_role": "diagnostic_only",
    }


def build_primary_endpoint(
    *,
    primary_trials: str | Path,
    confirmatory_plan: str | Path,
    endpoint_config: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> PrimaryEndpointArtifacts:
    """Validate the primary rows and publish trial/payload recovery summaries."""

    trials_path = Path(primary_trials).resolve()
    plan_path = Path(confirmatory_plan).resolve()
    config_path = Path(endpoint_config).resolve()
    if trials_path.is_symlink() or not trials_path.is_file():
        raise PrimaryEndpointError(f"Missing or unsafe primary trials: {trials_path}")
    plan = _read_json(plan_path, label="confirmatory model plan")
    config = _read_json(config_path, label="primary endpoint config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("analysis_status")
        != "post_audit_payload_group_sensitivity_not_prespecified_replacement"
        or config.get("partial_outcomes_seen_before_addition") is not True
        or config.get("frozen_generation_design_unchanged") is not True
        or config.get("strict_payload_success_rule")
        != "all_observed_primary_trials_for_payload_must_recover"
    ):
        raise PrimaryEndpointError("Primary endpoint disclosure config differs")
    if (
        plan.get("plan_id")
        != "rankcloak_revision_primary_v2_prespecified_confirmatory_models"
        or plan.get("experimental_unit") != "payload_trial"
        or plan.get("segments_as_independent_observations_forbidden") is not True
    ):
        raise PrimaryEndpointError("Confirmatory model plan identity differs")
    frozen_filter = plan.get("filters", {}).get("primary_trials")
    if not isinstance(frozen_filter, Mapping):
        raise PrimaryEndpointError("Confirmatory plan lacks its primary filter")

    frame = pd.read_csv(trials_path, low_memory=False)
    required = {
        "trial_id",
        "record_type",
        "evidence_status",
        "study_phase",
        "replay_mode",
        "protocol_contract_revision",
        "result_schema_revision",
        "exact_rank_replay",
        "exact_payload_recovery",
        "exact_recovery",
        "recovery_outcome_semantics",
        "payload_name",
        "payload_class",
        "model_id",
        "prompt_id",
        "prompt_category",
        "protocol_variant",
    }
    if not required.issubset(frame.columns):
        raise PrimaryEndpointError("Primary trials lack required endpoint columns")
    selected = frame.copy()
    for column, expected in frozen_filter.items():
        if column not in selected:
            raise PrimaryEndpointError(f"Primary filter column is absent: {column}")
        selected = selected.loc[selected[column].astype(str).eq(str(expected))]
    selected = selected.copy()
    if selected.empty or selected["trial_id"].isna().any():
        raise PrimaryEndpointError("No identified primary rows remain")
    if selected["trial_id"].astype(str).duplicated().any():
        raise PrimaryEndpointError("Primary trial identities repeat")
    selected["exact_payload_recovery"] = _binary(
        selected["exact_payload_recovery"], label="exact_payload_recovery"
    )
    selected["exact_recovery"] = _binary(
        selected["exact_recovery"], label="exact_recovery"
    )
    selected["exact_rank_replay"] = _binary(
        selected["exact_rank_replay"], label="exact_rank_replay"
    )
    if not selected["exact_recovery"].equals(selected["exact_payload_recovery"]):
        raise PrimaryEndpointError("Recovery compatibility alias differs")
    if not selected["recovery_outcome_semantics"].astype(str).eq(
        RECOVERY_SEMANTICS
    ).all():
        raise PrimaryEndpointError("Primary recovery semantics differ")

    expected = config.get("expected_design")
    if not isinstance(expected, Mapping):
        raise PrimaryEndpointError("Primary endpoint config lacks expected_design")
    observed = {
        "primary_trial_rows": int(len(selected)),
        "payloads": int(selected["payload_name"].nunique()),
        "model_families": int(selected["model_id"].nunique()),
        "prompt_templates": int(selected["prompt_id"].nunique()),
        "prompt_categories": int(selected["prompt_category"].nunique()),
        "payload_classes": int(selected["payload_class"].nunique()),
        "protocol_variants": int(selected["protocol_variant"].nunique()),
    }
    if any(int(expected.get(key, -1)) != value for key, value in observed.items()):
        raise PrimaryEndpointError("Observed primary endpoint design counts differ")

    payload_grouped = selected.groupby("payload_name", sort=True, dropna=False)
    payload_rows = payload_grouped.agg(
        payload_class=("payload_class", "first"),
        primary_trial_count=("trial_id", "size"),
        model_count=("model_id", "nunique"),
        prompt_template_count=("prompt_id", "nunique"),
        protocol_variant_count=("protocol_variant", "nunique"),
        exact_payload_recovery_all=("exact_payload_recovery", "min"),
        exact_rank_replay_all=("exact_rank_replay", "min"),
    ).reset_index()
    if payload_rows["payload_name"].isna().any():
        raise PrimaryEndpointError("A primary payload identity is missing")
    class_counts = selected.groupby("payload_name")["payload_class"].nunique()
    if not class_counts.eq(1).all():
        raise PrimaryEndpointError("A payload maps to multiple payload classes")
    distribution = {
        str(int(trials)): int(count)
        for trials, count in payload_rows["primary_trial_count"]
        .value_counts()
        .sort_index()
        .items()
    }
    declared_distribution = {
        str(key): int(value)
        for key, value in dict(
            config.get("expected_payload_trial_count_distribution", {})
        ).items()
    }
    if distribution != declared_distribution:
        raise PrimaryEndpointError("Payload trial-count distribution differs")

    model_payload = selected.groupby(
        ["payload_name", "model_id"], sort=True, dropna=False
    ).agg(
        trial_count=("trial_id", "size"),
        exact_payload_recovery_all=("exact_payload_recovery", "min"),
    )
    expected_model_payload_groups = int(expected.get("model_payload_groups", -1))
    if len(model_payload) != expected_model_payload_groups:
        raise PrimaryEndpointError("Model-payload group count differs")

    confidence = float(config.get("confidence_level", 0.0))
    endpoint_rows = [
        _endpoint_row(
            endpoint_id="primary_payload_trial_frozen_endpoint",
            analysis_unit="payload_trial",
            evidence_status="confirmatory_frozen_endpoint",
            successes=int(selected["exact_payload_recovery"].sum()),
            total=len(selected),
            minimum_trials=1,
            maximum_trials=1,
            confidence=confidence,
        ),
        _endpoint_row(
            endpoint_id="primary_payload_strict_all_observed_configurations",
            analysis_unit="payload_name",
            evidence_status="supporting_post_audit_strict_payload_sensitivity",
            successes=int(payload_rows["exact_payload_recovery_all"].sum()),
            total=len(payload_rows),
            minimum_trials=int(payload_rows["primary_trial_count"].min()),
            maximum_trials=int(payload_rows["primary_trial_count"].max()),
            confidence=confidence,
        ),
        _endpoint_row(
            endpoint_id="primary_model_payload_strict",
            analysis_unit="payload_name+model_id",
            evidence_status="supporting_post_audit_grouped_sensitivity",
            successes=int(model_payload["exact_payload_recovery_all"].sum()),
            total=len(model_payload),
            minimum_trials=int(model_payload["trial_count"].min()),
            maximum_trials=int(model_payload["trial_count"].max()),
            confidence=confidence,
        ),
    ]
    endpoints = pd.DataFrame(endpoint_rows)

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise PrimaryEndpointError(f"Unsafe primary endpoint directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "endpoints": target / "primary_recovery_endpoints.csv",
        "payloads": target / "primary_payload_group_outcomes.csv",
        "manifest": target / "primary_recovery_endpoint_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise PrimaryEndpointError(
                f"Refusing to overwrite primary endpoint output: {path}"
            )
    _atomic_csv(endpoints, paths["endpoints"])
    _atomic_csv(payload_rows, paths["payloads"])
    outputs = {
        "endpoints": _identity(paths["endpoints"], row_count=len(endpoints)),
        "payload_group_outcomes": _identity(
            paths["payloads"], row_count=len(payload_rows)
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "analysis_status": config["analysis_status"],
        "partial_outcomes_seen_before_addition": True,
        "frozen_generation_design_unchanged": True,
        "inputs": {
            "primary_trials": _identity(trials_path, row_count=len(frame)),
            "confirmatory_plan": _identity(plan_path),
            "endpoint_config": _identity(config_path),
        },
        "observed_design": observed,
        "payload_trial_count_distribution": distribution,
        "outputs": outputs,
        "summary": {
            "endpoint_rows": int(len(endpoints)),
            "payload_rows": int(len(payload_rows)),
            "primary_trial_successes": int(
                selected["exact_payload_recovery"].sum()
            ),
            "strict_payload_successes": int(
                payload_rows["exact_payload_recovery_all"].sum()
            ),
            "strict_model_payload_successes": int(
                model_payload["exact_payload_recovery_all"].sum()
            ),
        },
        "limitations": [
            "The strict payload-group sensitivity was added after partial outcomes were available.",
            "It does not replace the frozen payload-trial endpoint.",
            "Wilson intervals summarize binary grouped outcomes without modeling cross-configuration heterogeneity.",
        ],
        "generation_command": command,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(manifest, paths["manifest"])
    return PrimaryEndpointArtifacts(
        output_dir=str(target),
        manifest_path=str(paths["manifest"]),
        endpoint_row_count=len(endpoints),
        payload_row_count=len(payload_rows),
    )
