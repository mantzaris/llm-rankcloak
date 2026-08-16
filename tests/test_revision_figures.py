from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_figures import (
    OUTPUT_FILENAMES,
    FigureEvidenceError,
    build_core_figures,
    canonical_json_sha256,
    file_sha256,
    refresh_evidence_summary_figure_hashes,
)


EXPECTED_FIGURES = {
    "robustness_compact": (1, 11),
    "robustness_full": (4, 24),
    "detector_compact": (3, 24),
    "detector_full": (6, 48),
    "capacity_tail": (3, 18),
    "readability": (3, 21),
    "overhead_compact": (3, 54),
    "overhead_full": (4, 72),
    "ablation": (3, 9),
}


def _manifest(path: Path, outputs: Mapping[str, Path]) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "outputs": {
                    key: {
                        "path": source.name,
                        "sha256": file_sha256(source),
                    }
                    for key, source in outputs.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    robustness_source = tmp_path / "robustness.csv"
    robustness_rows: list[dict[str, Any]] = []

    def add_robustness(
        family: str,
        replay: str,
        transformation: str,
        point: float,
    ) -> None:
        robustness_rows.append(
            {
                "robustness_family": family,
                "replay_mode": replay,
                "transformation_id": transformation,
                "source_cover_units": 12,
                "observed_outcome_rows": 12,
                "unavailable_rows": 0,
                "recovery_rate": point,
                "ci_low": max(0.0, point - 0.10),
                "ci_high": min(1.0, point + 0.10),
                "status": "observed",
            }
        )

    for replay, point in (
        ("saved_token_ids", 1.0),
        ("detokenized_text_retokenized", 0.8),
        ("greedy_leadin_regeneration", 0.5),
    ):
        add_robustness("replay_modes", replay, "unmodified", point)
    raw_transformations = (
        "unmodified",
        "character_deletion",
        "character_insertion",
        "character_substitution",
        "line_endings",
        "markdown_copy_paste",
        "paraphrase",
        "quote_conversion",
        "token_deletion",
        "truncation",
        "unicode_normalization",
        "whitespace_collapse",
        "whitespace_trim",
    )
    for index, transformation in enumerate(raw_transformations):
        point = 0.9 if transformation == "unmodified" else 0.05 + index * 0.02
        add_robustness(
            "raw_transmission",
            "transformed_text_retokenized",
            transformation,
            min(point, 0.95),
        )
    for index, transformation in enumerate(
        (
            "unmodified",
            "whitespace_trim",
            "whitespace_collapse",
            "line_endings",
            "unicode_normalization",
            "quote_conversion",
            "markdown_copy_paste",
        )
    ):
        add_robustness(
            "limited_mitigation",
            "canonicalized_text_retokenized",
            transformation,
            0.7 if transformation == "unmodified" else 0.12 + index * 0.02,
        )
    add_robustness(
        "cross_model_mismatch",
        "cross_model_text_retokenized",
        "unmodified",
        0.0,
    )
    pd.DataFrame(robustness_rows).to_csv(robustness_source, index=False)

    theory_source = tmp_path / "theory.csv"
    theory_cells = (
        ("primary_v2", "nonseg_ascii_b8", "none", 8, 0),
        ("primary_v2", "nonseg_ascii_b16", "none", 7, 0),
        ("primary_v2", "nonseg_hex_nibble_b16", "none", 6, 0),
        (
            "primary_v2",
            "segmented_hex_multi_topic",
            "dynamic_completion_v1",
            10,
            180,
        ),
        (
            "primary_v2",
            "segmented_hex_single_topic",
            "sentence_tail_min20_max60",
            10,
            40,
        ),
        ("ablation_v2", "nonseg_ascii_b8", "none", 9, 0),
        (
            "ablation_v2",
            "segmented_hex_multi_topic",
            "dynamic_completion_v1",
            11,
            220,
        ),
        ("multilingual_v2", "nonseg_ascii_b8", "none", 12, 0),
        (
            "multilingual_v2",
            "segmented_hex_single_topic",
            "sentence_tail_min20_max60",
            13,
            45,
        ),
    )
    pd.DataFrame(
        [
            {
                "source_stage": stage,
                "model_id": "llama3_8b_instruct_q4_k_m",
                "protocol_variant": protocol,
                "tail_policy": tail,
                "theoretical_n_B": forced,
                "observed_n_forced": forced,
                "tail_overhead_tokens": tail_tokens,
            }
            for stage, protocol, tail, forced, tail_tokens in theory_cells
        ]
    ).to_csv(theory_source, index=False)

    readability_source = tmp_path / "readability.csv"
    conditions = (
        "ordinary_llm_control",
        "rankcloak_ascii_b8",
        "rankcloak_ascii_b16",
        "rankcloak_hex_nibble",
        "direct_subword_calgacus",
        "rankcloak_segmented_forced_span",
        "rankcloak_segmented_full_message",
    )
    outcomes = (
        "flesch_reading_ease_heuristic",
        "surface_flag_total",
        "tfidf_prompt_similarity",
    )
    pd.DataFrame(
        [
            {
                "condition": condition,
                "outcome": outcome,
                "n": 72,
                "mean": 1.0 + condition_index * 0.2,
                "ci_low": 0.9 + condition_index * 0.2,
                "ci_high": 1.1 + condition_index * 0.2,
                "interval_method": "prompt_template_cluster_percentile_bootstrap",
                "human_rating_substitute": False,
            }
            for condition_index, condition in enumerate(conditions)
            for outcome in outcomes
        ]
    ).to_csv(readability_source, index=False)

    overhead_source = tmp_path / "overhead.csv"
    overhead_points = {
        "generation_seconds": 4.0,
        "encoding_overhead_seconds": 0.2,
        "decoding_overhead_seconds": 2.0,
        "payload_bits_per_second": 40.0,
    }
    models = (
        "llama3_8b_instruct_q4_k_m",
        "mistral_7b_instruct_v0_3_q4_k_m",
        "qwen2_5_7b_instruct_q4_k_m",
    )
    protocols = (
        "direct_subword_calgacus",
        "nonseg_ascii_b16",
        "nonseg_ascii_b8",
        "nonseg_hex_nibble_b16",
        "segmented_hex_multi_topic",
        "segmented_hex_single_topic",
    )
    pd.DataFrame(
        [
            {
                "source_stage": "primary_v2",
                "runtime_scope": "trial",
                "model_id": model_id,
                "protocol_variant": protocol,
                "outcome": outcome,
                "mean": point,
                "ci_low": point * 0.9,
                "ci_high": point * 1.1,
                "n_payloads": 12,
            }
            for model_id in models
            for protocol in protocols
            for outcome, point in overhead_points.items()
        ]
    ).to_csv(overhead_source, index=False)

    detector_source = tmp_path / "detector.csv"
    detectors = (
        "published_textcnn_equivalent",
        "deberta_v3_base_classifier",
    )
    regimes = (
        "matched",
        "held_out_template",
        "leave_one_model",
        "leave_one_codec",
    )
    detector_outcomes = (
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "brier_score",
        "tpr_at_fpr_0.01",
    )
    detector_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for detector_index, detector in enumerate(detectors):
        for regime_index, regime in enumerate(regimes):
            for metric in detector_outcomes:
                point = 0.18 if metric == "brier_score" else 0.72
                point += detector_index * 0.03 + regime_index * 0.01
                detector_rows.append(
                    {
                        "detector_name": detector,
                        "regime": regime,
                        "metric": metric,
                        "evidence_status": (
                            "confirmatory_frozen_upstream"
                            if metric in {"roc_auc", "pr_auc", "balanced_accuracy"}
                            else "supplementary_exploratory_post_freeze"
                        ),
                        "higher_is_better": metric != "brier_score",
                        "split_count": 1 if regime == "matched" else 2,
                        "estimate_median_across_splits": point,
                        "estimate_min_across_splits": (
                            point if regime == "matched" else point - 0.03
                        ),
                        "estimate_max_across_splits": (
                            point if regime == "matched" else point + 0.03
                        ),
                        "test_rows_sum_across_splits": 80,
                        "payload_groups_sum_across_splits": 40,
                        "cross_split_interval": (
                            "not_computed_heterogeneous_prespecified_splits"
                        ),
                    }
                )
                if regime == "matched":
                    metrics_rows.append(
                        {
                            "split_id": "matched",
                            "detector_name": detector,
                            "metric": metric,
                            "estimate": point,
                            "ci_low": max(0.0, point - 0.04),
                            "ci_high": min(1.0, point + 0.04),
                            "confidence_level": 0.95,
                            "evidence_status": (
                                "confirmatory_frozen_upstream"
                                if metric
                                in {"roc_auc", "pr_auc", "balanced_accuracy"}
                                else "supplementary_exploratory_post_freeze"
                            ),
                        }
                    )
    pd.DataFrame(detector_rows).to_csv(detector_source, index=False)
    detector_metrics = tmp_path / "detector_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(detector_metrics, index=False)

    ablation_source = tmp_path / "ablation.csv"
    factor_levels = {
        "leadin_tokens": ("0", "8", "16", "32", "64"),
        "segment_size_ranks": ("16", "32", "64"),
        "token_filter": ("none", "roundtrip_stable_v1"),
        "tail_policy": ("none", "sentence_tail_min20_max60"),
    }
    canonical = {
        "leadin_tokens": "16",
        "segment_size_ranks": "64",
        "token_filter": "roundtrip_stable_v1",
        "tail_policy": "sentence_tail_min20_max60",
    }
    ablation_outcomes = (
        "mean_log_probability",
        "effective_artifact_bits_per_full_token",
        "full_token_count",
        "recovery_rate",
        "artifact_flag_rate",
    )
    ablation_rows: list[dict[str, Any]] = []
    for factor_index, (factor, levels) in enumerate(factor_levels.items()):
        for level_index, level in enumerate(levels):
            for outcome_index, outcome in enumerate(ablation_outcomes):
                estimate = (factor_index - 1.5) * 0.05 + level_index * 0.02
                ablation_rows.append(
                    {
                        "factor": factor,
                        "level": level,
                        "canonical_value": canonical[factor],
                        "outcome": outcome,
                        "shared_models": "all_available",
                        "paired_payload_groups": 48,
                        "level_minus_canonical": estimate,
                        "ci_low": estimate - 0.1,
                        "ci_high": estimate + 0.1,
                        "p_value_holm": 0.5,
                        "inferential_p_value_supported": True,
                        "bootstrap_unit": "payload_group",
                        "bootstrap_resamples": 2000,
                        "confidence_level": 0.95,
                        "primary_inference": False,
                        "evidence_status": (
                            "exploratory_post_outcome_evidence_extraction"
                        ),
                    }
                )
    assert len(ablation_rows) == 60
    pd.DataFrame(ablation_rows).to_csv(ablation_source, index=False)

    return {
        "robustness": _manifest(
            tmp_path / "robustness_manifest.json",
            {"plot_source": robustness_source},
        ),
        "theory": _manifest(
            tmp_path / "theory_manifest.json", {"validation": theory_source}
        ),
        "readability": _manifest(
            tmp_path / "readability_manifest.json", {"summary": readability_source}
        ),
        "overhead": _manifest(
            tmp_path / "overhead_manifest.json", {"plot_source": overhead_source}
        ),
        "detector": _manifest(
            tmp_path / "detector_manifest.json",
            {"plot_source": detector_source, "metrics": detector_metrics},
        ),
        "ablation": _manifest(
            tmp_path / "ablation_manifest.json", {"contrasts": ablation_source}
        ),
    }


def _build(tmp_path: Path, manifests: Mapping[str, Path], **kwargs: Any):
    return build_core_figures(
        robustness_manifest=manifests["robustness"],
        theory_manifest=manifests["theory"],
        readability_manifest=manifests["readability"],
        overhead_manifest=manifests["overhead"],
        detector_manifest=manifests["detector"],
        ablation_manifest=manifests["ablation"],
        output_dir=kwargs.pop("output_dir", tmp_path / "figures"),
        command=kwargs.pop(
            "command", "scripts/build_revision_figures.py --fixture"
        ),
        project_root=tmp_path,
        **kwargs,
    )


def _assert_portable_paths(value: Any, project_root: Path) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {
                "path",
                "primary_pdf",
                "inspection_png_300dpi",
                "plotted_source_csv",
                "technical_note",
            }:
                assert not Path(str(nested)).is_absolute()
                assert str(project_root) not in str(nested)
            _assert_portable_paths(nested, project_root)
    elif isinstance(value, list):
        for nested in value:
            _assert_portable_paths(nested, project_root)


def test_builder_emits_validated_portable_figure_package(tmp_path: Path):
    manifests = _fixture(tmp_path)
    output = tmp_path / "figures"
    artifacts = _build(tmp_path, manifests, output_dir=output)

    assert artifacts.summary == {
        "figure_count": 9,
        "rendered_file_count": 18,
        "source_table_count": 9,
        "technical_note_count": 6,
        "inventory_row_count": 9,
    }
    assert set(OUTPUT_FILENAMES) == set(artifacts.files)
    for key, filename in OUTPUT_FILENAMES.items():
        target = output / filename
        assert target.is_file(), key
        if key.endswith("_pdf"):
            assert target.read_bytes().startswith(b"%PDF")
        if key.endswith("_png"):
            assert target.read_bytes().startswith(b"\x89PNG")

    manifest = json.loads(
        (output / OUTPUT_FILENAMES["manifest"]).read_text(encoding="utf-8")
    )
    validation = json.loads(
        (output / OUTPUT_FILENAMES["validation"]).read_text(encoding="utf-8")
    )
    inventory = pd.read_csv(output / OUTPUT_FILENAMES["inventory"])
    assert manifest["schema_version"].endswith("figure-evidence-v2")
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    assert manifest["portable_repository_relative_paths"] is True
    assert manifest["human_rating_figures_emitted"] is False
    assert manifest["ablation_figure_status"] == "exploratory_post_outcome"
    assert validation["status"] == "passed"
    assert validation["checks"]["confidence_intervals_and_ranges_distinguished"]
    assert validation["checks"]["automated_readability_not_human_rating"]
    assert validation["checks"]["ablation_exploratory_status_preserved"]
    _assert_portable_paths(manifest, tmp_path)
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in (output / OUTPUT_FILENAMES["commands"]).read_text()

    assert len(inventory) == 9
    assert set(inventory["figure_id"]) == set(EXPECTED_FIGURES)
    for figure_id, (panels, rows) in EXPECTED_FIGURES.items():
        figure = manifest["figures"][figure_id]
        assert figure["panel_count"] == panels
        assert figure["plotted_row_count"] == rows
        inventory_row = inventory.loc[inventory["figure_id"].eq(figure_id)].iloc[0]
        assert inventory_row["panel_count"] == panels
        assert inventory_row["plotted_row_count"] == rows

    assert manifest["figures"]["readability"]["evidence_status"] == (
        "automated_not_human_rating"
    )
    assert manifest["figures"]["ablation"]["evidence_status"] == (
        "exploratory_post_outcome"
    )
    assert "exploratory" in manifest["figures"]["detector_compact"][
        "evidence_status"
    ]

    for key, entry in manifest["outputs"].items():
        resolved = tmp_path / entry["path"]
        assert file_sha256(resolved) == entry["sha256"], key
        if key.endswith("_source"):
            frame = pd.read_csv(resolved)
            numeric = frame.select_dtypes(include=[np.number])
            assert np.isfinite(numeric.to_numpy()).all(), key

    for key, point_column, low_column, high_column in (
        ("robustness_source", "recovery_rate", "ci_low", "ci_high"),
        ("readability_source", "mean", "ci_low", "ci_high"),
        ("overhead_source", "mean", "ci_low", "ci_high"),
        (
            "detector_source",
            "estimate_median_across_splits",
            "interval_low",
            "interval_high",
        ),
        (
            "ablation_source",
            "level_minus_canonical",
            "ci_low",
            "ci_high",
        ),
    ):
        frame = pd.read_csv(tmp_path / manifest["outputs"][key]["path"])
        assert (frame[low_column] <= frame[point_column]).all(), key
        assert (frame[point_column] <= frame[high_column]).all(), key


def test_builder_rejects_tampered_declared_source(tmp_path: Path):
    manifests = _fixture(tmp_path)
    (tmp_path / "robustness.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FigureEvidenceError, match="hash mismatch"):
        _build(tmp_path, manifests)


def test_builder_refuses_unrequested_overwrite(tmp_path: Path):
    manifests = _fixture(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    (output / OUTPUT_FILENAMES["manifest"]).write_text(
        "preserve", encoding="utf-8"
    )
    with pytest.raises(FigureEvidenceError, match="Refusing to overwrite"):
        _build(tmp_path, manifests, output_dir=output)


def test_builder_rejects_interval_that_excludes_estimate(tmp_path: Path):
    manifests = _fixture(tmp_path)
    source = tmp_path / "robustness.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "ci_high"] = 0.20
    frame.to_csv(source, index=False)
    manifests["robustness"] = _manifest(
        tmp_path / "robustness_manifest.json", {"plot_source": source}
    )
    with pytest.raises(FigureEvidenceError, match="excluding its estimate"):
        _build(tmp_path, manifests)


def test_builder_rejects_incomplete_frozen_overhead_grid(tmp_path: Path):
    manifests = _fixture(tmp_path)
    source = tmp_path / "overhead.csv"
    pd.read_csv(source).iloc[1:].to_csv(source, index=False)
    manifests["overhead"] = _manifest(
        tmp_path / "overhead_manifest.json", {"plot_source": source}
    )
    with pytest.raises(FigureEvidenceError, match="complete frozen primary grid"):
        _build(tmp_path, manifests)


def test_builder_rejects_incomplete_detector_grid(tmp_path: Path):
    manifests = _fixture(tmp_path)
    source = tmp_path / "detector.csv"
    pd.read_csv(source).iloc[1:].to_csv(source, index=False)
    manifests["detector"] = _manifest(
        tmp_path / "detector_manifest.json",
        {"plot_source": source, "metrics": tmp_path / "detector_metrics.csv"},
    )
    with pytest.raises(FigureEvidenceError, match="complete frozen grid"):
        _build(tmp_path, manifests)


def test_builder_rejects_nonportable_command(tmp_path: Path):
    manifests = _fixture(tmp_path)
    with pytest.raises(FigureEvidenceError, match="repository-relative"):
        _build(tmp_path, manifests, command=f"{tmp_path}/scripts/build.py")


def test_parent_refresh_changes_only_existing_figure_identities(tmp_path: Path):
    figures = tmp_path / "results" / "package" / "figures"
    tables = tmp_path / "results" / "package" / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    figure = figures / "figure.pdf"
    figure.write_bytes(b"%PDF-new-figure")
    multiline = figures / "multiline.csv"
    pd.DataFrame([{"display_label": "line one\nline two"}]).to_csv(
        multiline, index=False
    )
    figure_manifest = figures / "figure_manifest.json"
    figure_manifest.write_text("{}\n", encoding="utf-8")
    preserved = tmp_path / "raw.csv"
    preserved.write_text("value\n1\n", encoding="utf-8")
    reference_table = tables / "evidence_artifact_references.csv"
    rows = [
        {
            "path": figure.relative_to(tmp_path).as_posix(),
            "sha256": "0" * 64,
            "size_bytes": "1",
            "row_count": "",
        },
        {
            "path": multiline.relative_to(tmp_path).as_posix(),
            "sha256": "0" * 64,
            "size_bytes": "1",
            "row_count": "99",
        },
        {
            "path": preserved.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(preserved),
            "size_bytes": str(preserved.stat().st_size),
            "row_count": "1",
        },
    ]
    pd.DataFrame(rows).to_csv(reference_table, index=False)
    manifest = {
        "schema_version": "fixture-v1",
        "git_head": "sealed-head",
        "generation_command": "/historical/absolute/command",
        "referenced_artifacts": [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "size_bytes": int(row["size_bytes"]),
                "row_count": (
                    None if row["row_count"] == "" else int(row["row_count"])
                ),
            }
            for row in rows
        ],
        "outputs": {
            "tables/evidence_artifact_references.csv": {
                "path": reference_table.relative_to(tmp_path).as_posix(),
                "sha256": file_sha256(reference_table),
                "size_bytes": reference_table.stat().st_size,
                "row_count": 3,
            }
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "results" / "package" / "evidence_summary_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = refresh_evidence_summary_figure_hashes(
        project_root=tmp_path,
        evidence_summary_manifest=manifest_path,
        reference_table=reference_table,
        figure_dir=figures,
        figure_manifest=figure_manifest,
        refresh_head="refresh-head",
        command="scripts/refresh_revision_figure_parent_hashes.py --fixture",
    )

    refreshed_table = pd.read_csv(reference_table, dtype=str, keep_default_na=False)
    figure_row = refreshed_table.loc[
        refreshed_table["path"].eq(figure.relative_to(tmp_path).as_posix())
    ].iloc[0]
    preserved_row = refreshed_table.loc[
        refreshed_table["path"].eq(preserved.relative_to(tmp_path).as_posix())
    ].iloc[0]
    multiline_row = refreshed_table.loc[
        refreshed_table["path"].eq(multiline.relative_to(tmp_path).as_posix())
    ].iloc[0]
    assert figure_row["sha256"] == file_sha256(figure)
    assert multiline_row["row_count"] == "1"
    assert preserved_row.to_dict() == rows[2]
    refreshed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = refreshed_manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(refreshed_manifest)
    assert refreshed_manifest["git_head"] == "sealed-head"
    assert refreshed_manifest["generation_command"] == (
        "/historical/absolute/command"
    )
    assert refreshed_manifest["figure_hash_refresh"]["refresh_git_head"] == (
        "refresh-head"
    )
    assert artifacts.updated_reference_count == 2
