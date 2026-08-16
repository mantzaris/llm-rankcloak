"""Hash-bound multilingual recovery endpoints at language and payload units."""

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


CONFIG_SCHEMA = "rankcloak-multilingual-recovery-endpoint-config-v1"
MANIFEST_SCHEMA = "rankcloak-multilingual-recovery-endpoint-v1"
RECOVERY_SEMANTICS = "original_serialized_payload_bytes_sha256_v1"


class MultilingualEndpointError(ValueError):
    """Raised when multilingual endpoint inputs or frozen identities differ."""


@dataclass(frozen=True)
class MultilingualEndpointArtifacts:
    output_dir: str
    manifest_path: str
    endpoint_row_count: int
    language_payload_row_count: int


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
        raise MultilingualEndpointError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MultilingualEndpointError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise MultilingualEndpointError(f"{label} must contain a JSON object")
    return value


def _identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        value["row_count"] = int(row_count)
    return value


def _binary(series: pd.Series, *, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise MultilingualEndpointError(f"{label} must contain only zero and one")
    return values.astype(int)


def _wilson(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise MultilingualEndpointError("Wilson counts are invalid")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-15):
        raise MultilingualEndpointError(
            "Multilingual endpoint confidence level must be 0.95"
        )
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    centre = estimate + z * z / (2.0 * total)
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / total
        + z * z / (4.0 * total * total)
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
    language: str,
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
        "language": language,
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
        "minimum_trials_per_unit": int(minimum_trials),
        "maximum_trials_per_unit": int(maximum_trials),
        "recovery_outcome": "exact_payload_recovery",
        "recovery_outcome_semantics": RECOVERY_SEMANTICS,
        "exact_rank_replay_role": "diagnostic_only",
    }


def build_multilingual_endpoint(
    *,
    multilingual_trials: str | Path,
    endpoint_config: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> MultilingualEndpointArtifacts:
    """Validate saved multilingual rows and publish language-group endpoints."""

    trials_path = Path(multilingual_trials).resolve()
    config_path = Path(endpoint_config).resolve()
    if trials_path.is_symlink() or not trials_path.is_file():
        raise MultilingualEndpointError(
            f"Missing or unsafe multilingual trials: {trials_path}"
        )
    config = _read_json(config_path, label="multilingual endpoint config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("analysis_status")
        != "secondary_post_audit_language_group_sensitivity"
        or config.get("outcomes_were_available_before_grouping_specification")
        is not True
        or config.get("frozen_generation_design_unchanged") is not True
        or config.get("strict_payload_success_rule")
        != "all_observed_multilingual_trials_for_language_payload_must_recover"
    ):
        raise MultilingualEndpointError(
            "Multilingual endpoint disclosure config differs"
        )
    frozen_filter = config.get("frozen_filter")
    expected = config.get("expected_design")
    if not isinstance(frozen_filter, Mapping) or not isinstance(expected, Mapping):
        raise MultilingualEndpointError(
            "Multilingual endpoint config lacks filter or design"
        )

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
        "language",
        "payload_name",
        "payload_class",
        "model_id",
        "prompt_id",
        "prompt_category",
        "protocol_variant",
    }
    if not required.issubset(frame.columns):
        raise MultilingualEndpointError(
            "Multilingual trials lack required endpoint columns"
        )
    selected = frame.copy()
    for column, expected_value in frozen_filter.items():
        if column not in selected:
            raise MultilingualEndpointError(
                f"Multilingual filter column is absent: {column}"
            )
        selected = selected.loc[
            selected[column].astype(str).eq(str(expected_value))
        ]
    selected = selected.copy()
    if selected.empty or selected["trial_id"].isna().any():
        raise MultilingualEndpointError("No identified multilingual rows remain")
    if selected["trial_id"].astype(str).duplicated().any():
        raise MultilingualEndpointError("Multilingual trial identities repeat")
    for column in (
        "exact_payload_recovery",
        "exact_recovery",
        "exact_rank_replay",
    ):
        selected[column] = _binary(selected[column], label=column)
    if not selected["exact_recovery"].equals(
        selected["exact_payload_recovery"]
    ):
        raise MultilingualEndpointError("Recovery compatibility alias differs")
    if not selected["recovery_outcome_semantics"].astype(str).eq(
        RECOVERY_SEMANTICS
    ).all():
        raise MultilingualEndpointError(
            "Multilingual recovery semantics differ"
        )

    observed = {
        "trial_rows": int(len(selected)),
        "payloads": int(selected["payload_name"].nunique()),
        "model_families": int(selected["model_id"].nunique()),
        "prompt_templates": int(selected["prompt_id"].nunique()),
        "prompt_categories": int(selected["prompt_category"].nunique()),
        "payload_classes": int(selected["payload_class"].nunique()),
        "protocol_variants": int(selected["protocol_variant"].nunique()),
        "languages": sorted(selected["language"].astype(str).unique()),
    }
    for key in (
        "trial_rows",
        "payloads",
        "model_families",
        "prompt_templates",
        "prompt_categories",
        "payload_classes",
        "protocol_variants",
    ):
        if int(expected.get(key, -1)) != observed[key]:
            raise MultilingualEndpointError(
                "Observed multilingual endpoint design counts differ"
            )
    if list(expected.get("languages", [])) != observed["languages"]:
        raise MultilingualEndpointError("Observed multilingual languages differ")

    language_payload = (
        selected.groupby(
            ["language", "payload_name"], sort=True, dropna=False
        )
        .agg(
            payload_class=("payload_class", "first"),
            trial_count=("trial_id", "size"),
            model_count=("model_id", "nunique"),
            prompt_template_count=("prompt_id", "nunique"),
            protocol_variant_count=("protocol_variant", "nunique"),
            exact_payload_recovery_all=("exact_payload_recovery", "min"),
            exact_rank_replay_all=("exact_rank_replay", "min"),
        )
        .reset_index()
    )
    language_model_payload = (
        selected.groupby(
            ["language", "model_id", "payload_name"],
            sort=True,
            dropna=False,
        )
        .agg(
            trial_count=("trial_id", "size"),
            exact_payload_recovery_all=("exact_payload_recovery", "min"),
        )
        .reset_index()
    )
    if (
        len(language_payload)
        != int(expected.get("language_payload_groups", -1))
        or len(language_model_payload)
        != int(expected.get("language_model_payload_groups", -1))
        or not language_payload["trial_count"].eq(
            int(expected.get("trials_per_language_payload", -1))
        ).all()
        or not language_model_payload["trial_count"].eq(
            int(expected.get("trials_per_language_model_payload", -1))
        ).all()
    ):
        raise MultilingualEndpointError(
            "Multilingual repeated-payload distribution differs"
        )
    if (
        selected.groupby(["language", "payload_name"])["payload_class"]
        .nunique()
        .ne(1)
        .any()
    ):
        raise MultilingualEndpointError(
            "A language-payload group maps to multiple payload classes"
        )

    confidence = float(config.get("confidence_level", 0.0))
    endpoint_rows: list[dict[str, Any]] = []
    for language in observed["languages"]:
        trials_cell = selected[
            selected["language"].astype(str).eq(language)
        ]
        payload_cell = language_payload[
            language_payload["language"].astype(str).eq(language)
        ]
        model_payload_cell = language_model_payload[
            language_model_payload["language"].astype(str).eq(language)
        ]
        endpoint_rows.extend(
            [
                _endpoint_row(
                    endpoint_id=f"multilingual_{language}_payload_trial",
                    language=language,
                    analysis_unit="payload_trial",
                    evidence_status="secondary_frozen_generation_endpoint",
                    successes=int(trials_cell["exact_payload_recovery"].sum()),
                    total=len(trials_cell),
                    minimum_trials=1,
                    maximum_trials=1,
                    confidence=confidence,
                ),
                _endpoint_row(
                    endpoint_id=f"multilingual_{language}_strict_payload",
                    language=language,
                    analysis_unit="language+payload_name",
                    evidence_status=(
                        "supporting_post_audit_strict_payload_sensitivity"
                    ),
                    successes=int(
                        payload_cell["exact_payload_recovery_all"].sum()
                    ),
                    total=len(payload_cell),
                    minimum_trials=int(payload_cell["trial_count"].min()),
                    maximum_trials=int(payload_cell["trial_count"].max()),
                    confidence=confidence,
                ),
                _endpoint_row(
                    endpoint_id=(
                        f"multilingual_{language}_strict_model_payload"
                    ),
                    language=language,
                    analysis_unit="language+model_id+payload_name",
                    evidence_status=(
                        "supporting_post_audit_grouped_sensitivity"
                    ),
                    successes=int(
                        model_payload_cell[
                            "exact_payload_recovery_all"
                        ].sum()
                    ),
                    total=len(model_payload_cell),
                    minimum_trials=int(
                        model_payload_cell["trial_count"].min()
                    ),
                    maximum_trials=int(
                        model_payload_cell["trial_count"].max()
                    ),
                    confidence=confidence,
                ),
            ]
        )
    endpoints = pd.DataFrame(endpoint_rows)

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise MultilingualEndpointError(
            f"Unsafe multilingual endpoint directory: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "endpoints": target / "multilingual_recovery_endpoints.csv",
        "payloads": target / "multilingual_language_payload_outcomes.csv",
        "manifest": target / "multilingual_recovery_endpoint_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise MultilingualEndpointError(
                f"Refusing to overwrite multilingual endpoint output: {path}"
            )
    _atomic_csv(endpoints, paths["endpoints"])
    _atomic_csv(language_payload, paths["payloads"])
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "analysis_status": config["analysis_status"],
        "outcomes_were_available_before_grouping_specification": True,
        "frozen_generation_design_unchanged": True,
        "inputs": {
            "multilingual_trials": _identity(
                trials_path, row_count=len(frame)
            ),
            "endpoint_config": _identity(config_path),
        },
        "observed_design": observed,
        "outputs": {
            "endpoints": _identity(
                paths["endpoints"], row_count=len(endpoints)
            ),
            "language_payload_outcomes": _identity(
                paths["payloads"], row_count=len(language_payload)
            ),
        },
        "summary": {
            "endpoint_rows": int(len(endpoints)),
            "language_payload_rows": int(len(language_payload)),
            "trial_successes": int(
                selected["exact_payload_recovery"].sum()
            ),
            "strict_language_payload_successes": int(
                language_payload["exact_payload_recovery_all"].sum()
            ),
            "strict_language_model_payload_successes": int(
                language_model_payload["exact_payload_recovery_all"].sum()
            ),
        },
        "limitations": [
            "The language-group sensitivity was specified after saved outcomes were available.",
            "It does not replace the frozen secondary payload-trial endpoint.",
            "Only Spanish and Simplified Chinese secondary conditions were evaluated.",
            "Wilson intervals do not model cross-configuration heterogeneity."
        ],
        "generation_command": command,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(manifest, paths["manifest"])
    return MultilingualEndpointArtifacts(
        output_dir=str(target),
        manifest_path=str(paths["manifest"]),
        endpoint_row_count=len(endpoints),
        language_payload_row_count=len(language_payload),
    )
