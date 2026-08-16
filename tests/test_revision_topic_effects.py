from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_topic_effects import (
    TopicEffectError,
    build_topic_effect_extraction,
    canonical_json_sha256,
    file_sha256,
)


MODELS = ("analysis_artifact", "analysis_quality")
STRATA = ("llama", "mistral", "qwen")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for model_id in MODELS:
        is_artifact = model_id == "analysis_artifact"
        for position, stratum in enumerate(STRATA):
            rows.append(
                {
                    "model_id": model_id,
                    "stratum_model_id": stratum,
                    "multiplicity_family": (
                        "artifact_family" if is_artifact else "continuous_family"
                    ),
                    "contrast": (
                        "multi / single" if is_artifact else "multi - single"
                    ),
                    "estimate": 1.1 + position if is_artifact else 0.1 + position,
                    "standard_error": 0.05,
                    "statistic": 2.0,
                    "p_value_raw": 0.02,
                    "p_value_holm": 0.04,
                    "ci_low": 0.01,
                    "ci_high": 2.5,
                    "adjustment": "holm",
                    "scale": "response" if is_artifact else "model",
                    "fixed_effects_fallback": False,
                }
            )
    rows.append(
        {
            **rows[0],
            "contrast": "irrelevant",
        }
    )
    contrasts = tmp_path / "mixed_model_contrasts.csv"
    pd.DataFrame(rows).to_csv(contrasts, index=False)
    manifest = tmp_path / "mixed_model_run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_type": "rankcloak_revision_v1_mixed_model_run",
                "plan_id": (
                    "rankcloak_revision_primary_v2_prespecified_confirmatory_models"
                ),
                "validation_only": False,
                "fixed_effects_fallback": False,
                "analysis_unit": "payload_trial",
                "segments_as_independent_observations": False,
                "outputs": {
                    "contrasts": {
                        "path": contrasts.name,
                        "sha256": file_sha256(contrasts),
                        "size_bytes": contrasts.stat().st_size,
                        "row_count": len(rows),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": (
                    "rankcloak-topic-effect-extraction-config-v1"
                ),
                "analysis_status": (
                    "exploratory_post_outcome_locked_contrast_extraction"
                ),
                "outcomes_were_available_before_extraction_specification": True,
                "locked_mixed_models_unchanged": True,
                "expected_row_count": 6,
                "stratum_model_ids": list(STRATA),
                "selections": [
                    {
                        "model_id": "analysis_artifact",
                        "multiplicity_family": "artifact_family",
                        "contrast": "multi / single",
                        "scale": "response",
                        "effect_semantics": "ratio",
                    },
                    {
                        "model_id": "analysis_quality",
                        "multiplicity_family": "continuous_family",
                        "contrast": "multi - single",
                        "scale": "model",
                        "effect_semantics": "difference",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, config


def test_topic_effect_extraction_preserves_locked_rows_and_hashes(
    tmp_path: Path,
) -> None:
    manifest, config = _fixture(tmp_path)
    artifacts = build_topic_effect_extraction(
        mixed_model_manifest=manifest,
        extraction_config=config,
        output_dir=tmp_path / "output",
        command="fixture",
    )
    assert artifacts.contrast_row_count == 6
    output_manifest = json.loads(Path(artifacts.manifest_path).read_text())
    signature = output_manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(output_manifest)
    selected_path = Path(
        output_manifest["outputs"]["topic_schedule_contrasts"]["path"]
    )
    selected = pd.read_csv(selected_path)
    assert set(selected["contrast"]) == {"multi / single", "multi - single"}
    assert selected["new_model_fit_performed"].eq(False).all()
    assert file_sha256(selected_path) == output_manifest["outputs"][
        "topic_schedule_contrasts"
    ]["sha256"]


def test_topic_effect_extraction_rejects_missing_stratum(
    tmp_path: Path,
) -> None:
    manifest, config = _fixture(tmp_path)
    manifest_value = json.loads(manifest.read_text())
    path = tmp_path / manifest_value["outputs"]["contrasts"]["path"]
    frame = pd.read_csv(path).iloc[1:]
    frame.to_csv(path, index=False)
    manifest_value["outputs"]["contrasts"]["sha256"] = file_sha256(path)
    manifest_value["outputs"]["contrasts"]["size_bytes"] = path.stat().st_size
    manifest_value["outputs"]["contrasts"]["row_count"] = len(frame)
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    with pytest.raises(TopicEffectError, match="contrast grid differs"):
        build_topic_effect_extraction(
            mixed_model_manifest=manifest,
            extraction_config=config,
            output_dir=tmp_path / "output",
        )
