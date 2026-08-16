"""Compact, hash-bound extraction of locked single- versus multi-topic contrasts."""

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


CONFIG_SCHEMA = "rankcloak-topic-effect-extraction-config-v1"
MANIFEST_SCHEMA = "rankcloak-topic-effect-extraction-v1"


class TopicEffectError(ValueError):
    """Raised when locked topic contrasts cannot be selected exactly."""


@dataclass(frozen=True)
class TopicEffectArtifacts:
    output_dir: str
    manifest_path: str
    contrast_row_count: int


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
        raise TopicEffectError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TopicEffectError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TopicEffectError(f"{label} must contain a JSON object")
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


def _declared_output(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    name: str,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TopicEffectError("Mixed-model manifest lacks outputs")
    declaration = outputs.get(name)
    if not isinstance(declaration, Mapping):
        raise TopicEffectError(f"Mixed-model manifest lacks {name}")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TopicEffectError(f"Mixed-model output {name} lacks a path")
    candidate = Path(raw_path)
    path = (
        candidate
        if candidate.is_absolute()
        else manifest_path.parent / candidate
    ).resolve()
    if path.is_symlink() or not path.is_file():
        raise TopicEffectError(f"Mixed-model output is missing or unsafe: {path}")
    if (
        file_sha256(path) != declaration.get("sha256")
        or int(path.stat().st_size)
        != int(declaration.get("size_bytes", -1))
    ):
        raise TopicEffectError(f"Mixed-model output identity differs: {path}")
    return path


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


def build_topic_effect_extraction(
    *,
    mixed_model_manifest: str | Path,
    extraction_config: str | Path,
    output_dir: str | Path,
    command: str | None = None,
    overwrite: bool = False,
) -> TopicEffectArtifacts:
    """Select the exact locked topic contrasts without refitting any model."""

    manifest_path = Path(mixed_model_manifest).resolve()
    config_path = Path(extraction_config).resolve()
    manifest = _read_json(manifest_path, label="mixed-model manifest")
    config = _read_json(config_path, label="topic-effect extraction config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("analysis_status")
        != "exploratory_post_outcome_locked_contrast_extraction"
        or config.get("outcomes_were_available_before_extraction_specification")
        is not True
        or config.get("locked_mixed_models_unchanged") is not True
    ):
        raise TopicEffectError("Topic-effect extraction disclosure differs")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_type")
        != "rankcloak_revision_v1_mixed_model_run"
        or manifest.get("plan_id")
        != "rankcloak_revision_primary_v2_prespecified_confirmatory_models"
        or manifest.get("validation_only") is not False
        or manifest.get("fixed_effects_fallback") is not False
        or manifest.get("analysis_unit") != "payload_trial"
        or manifest.get("segments_as_independent_observations") is not False
    ):
        raise TopicEffectError("Mixed-model run identity differs")

    contrasts_path = _declared_output(
        manifest, manifest_path, "contrasts"
    )
    contrasts = pd.read_csv(contrasts_path, low_memory=False)
    required = {
        "model_id",
        "stratum_model_id",
        "multiplicity_family",
        "contrast",
        "estimate",
        "standard_error",
        "statistic",
        "p_value_raw",
        "p_value_holm",
        "ci_low",
        "ci_high",
        "adjustment",
        "scale",
        "fixed_effects_fallback",
    }
    if not required.issubset(contrasts.columns):
        raise TopicEffectError(
            "Mixed-model contrasts lack corrected analysis/stratum identities"
        )
    selections = config.get("selections")
    strata = list(map(str, config.get("stratum_model_ids", [])))
    if not isinstance(selections, list) or not selections or not strata:
        raise TopicEffectError("Topic-effect selections or strata are empty")

    selected_rows: list[pd.DataFrame] = []
    for raw in selections:
        if not isinstance(raw, Mapping):
            raise TopicEffectError("A topic-effect selection is malformed")
        model_id = str(raw.get("model_id", ""))
        family = str(raw.get("multiplicity_family", ""))
        contrast = str(raw.get("contrast", ""))
        scale = str(raw.get("scale", ""))
        semantics = str(raw.get("effect_semantics", ""))
        if not all((model_id, family, contrast, scale, semantics)):
            raise TopicEffectError("A topic-effect selection is incomplete")
        cell = contrasts.loc[
            contrasts["model_id"].astype(str).eq(model_id)
            & contrasts["multiplicity_family"].astype(str).eq(family)
            & contrasts["contrast"].astype(str).eq(contrast)
            & contrasts["scale"].astype(str).eq(scale)
        ].copy()
        if (
            len(cell) != len(strata)
            or sorted(cell["stratum_model_id"].astype(str)) != sorted(strata)
        ):
            raise TopicEffectError(
                f"Topic-effect contrast grid differs for {model_id}"
            )
        cell["effect_semantics"] = semantics
        selected_rows.append(cell)
    selected = pd.concat(selected_rows, ignore_index=True)
    expected_rows = int(config.get("expected_row_count", -1))
    if len(selected) != expected_rows or selected.duplicated(
        ["model_id", "stratum_model_id"]
    ).any():
        raise TopicEffectError("Topic-effect selected row identity differs")
    numeric_columns = (
        "estimate",
        "standard_error",
        "statistic",
        "p_value_raw",
        "p_value_holm",
        "ci_low",
        "ci_high",
    )
    for column in numeric_columns:
        values = pd.to_numeric(selected[column], errors="coerce")
        if not values.map(math.isfinite).all():
            raise TopicEffectError(
                f"Topic-effect contrast contains non-finite {column}"
            )
        selected[column] = values.astype(float)
    if (
        selected["fixed_effects_fallback"].astype(str).str.lower().ne("false").any()
        or selected["adjustment"].astype(str).ne("holm").any()
        or (selected["p_value_raw"] < 0).any()
        or (selected["p_value_raw"] > 1).any()
        or (selected["p_value_holm"] < 0).any()
        or (selected["p_value_holm"] > 1).any()
        or (selected["ci_low"] > selected["ci_high"]).any()
    ):
        raise TopicEffectError(
            "Topic-effect inference metadata or bounds differ"
        )
    selected["analysis_status"] = config["analysis_status"]
    selected["new_model_fit_performed"] = False
    selected = selected.sort_values(
        ["model_id", "stratum_model_id"]
    ).reset_index(drop=True)

    target = Path(output_dir).resolve()
    if target.is_symlink():
        raise TopicEffectError(f"Unsafe topic-effect output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": target / "topic_schedule_contrasts.csv",
        "manifest": target / "topic_effect_extraction_manifest.json",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise TopicEffectError(
                f"Refusing to overwrite topic-effect output: {path}"
            )
    _atomic_csv(selected, paths["contrasts"])
    output_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "analysis_status": config["analysis_status"],
        "outcomes_were_available_before_extraction_specification": True,
        "locked_mixed_models_unchanged": True,
        "new_model_fit_performed": False,
        "inputs": {
            "mixed_model_manifest": _identity(manifest_path),
            "mixed_model_contrasts": _identity(
                contrasts_path, row_count=len(contrasts)
            ),
            "extraction_config": _identity(config_path),
        },
        "outputs": {
            "topic_schedule_contrasts": _identity(
                paths["contrasts"], row_count=len(selected)
            )
        },
        "summary": {
            "selected_contrast_rows": int(len(selected)),
            "analysis_model_count": int(selected["model_id"].nunique()),
            "stratum_model_count": int(
                selected["stratum_model_id"].nunique()
            ),
            "holm_adjusted_rows": int(
                selected["adjustment"].astype(str).eq("holm").sum()
            ),
        },
        "limitations": [
            "The compact extraction was specified after saved outcomes were available.",
            "It preserves locked model estimates and multiplicity adjustments but is not a new confirmatory model.",
            "Singular-fit warnings in the source mixed-model diagnostics remain applicable."
        ],
        "generation_command": command,
    }
    output_manifest["manifest_sha256"] = canonical_json_sha256(
        output_manifest
    )
    _atomic_json(output_manifest, paths["manifest"])
    return TopicEffectArtifacts(
        output_dir=str(target),
        manifest_path=str(paths["manifest"]),
        contrast_row_count=len(selected),
    )
