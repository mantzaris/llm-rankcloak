from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_figures import (
    OUTPUT_FILENAMES,
    FigureEvidenceError,
    build_core_figures,
    file_sha256,
)


def _manifest(path: Path, key: str, source: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "outputs": {
                    key: {
                        "path": str(source.resolve()),
                        "sha256": file_sha256(source),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    robustness_source = tmp_path / "robustness.csv"
    pd.DataFrame(
        [
            {
                "robustness_family": "raw_transmission",
                "replay_mode": "raw_text_retokenized",
                "transformation_id": "unicode_nfc",
                "observed_outcome_rows": 12,
                "unavailable_rows": 0,
                "recovery_rate": 0.25,
                "ci_low": 0.10,
                "ci_high": 0.45,
                "status": "observed",
            }
        ]
    ).to_csv(robustness_source, index=False)

    theory_source = tmp_path / "theory.csv"
    pd.DataFrame(
        [
            {
                "source_stage": "primary_v2",
                "model_id": "model_a",
                "protocol_variant": "nonseg_ascii_b8",
                "tail_policy": "fixed_two",
                "theoretical_n_B": 3,
                "observed_n_forced": 3,
                "tail_overhead_tokens": 2,
            },
            {
                "source_stage": "primary_v2",
                "model_id": "model_b",
                "protocol_variant": "nonseg_ascii_b8",
                "tail_policy": "fixed_two",
                "theoretical_n_B": 4,
                "observed_n_forced": 4,
                "tail_overhead_tokens": 2,
            },
        ]
    ).to_csv(theory_source, index=False)

    readability_source = tmp_path / "readability.csv"
    pd.DataFrame(
        [
            {
                "condition": condition,
                "outcome": outcome,
                "n": 18,
                "mean": point,
                "ci_low": point - 0.1,
                "ci_high": point + 0.1,
            }
            for condition, point in (
                ("ordinary_llm_control", 1.0),
                ("rankcloak_ascii_b8", 1.2),
            )
            for outcome in (
                "flesch_reading_ease_heuristic",
                "surface_flag_total",
                "tfidf_prompt_similarity",
            )
        ]
    ).to_csv(readability_source, index=False)

    overhead_source = tmp_path / "overhead.csv"
    overhead_points = {
        "generation_seconds": 4.0,
        "encoding_overhead_seconds": 0.2,
        "decoding_overhead_seconds": 2.0,
        "payload_bits_per_second": 40.0,
    }
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
            for model_id in (
                "llama3_8b_instruct_q4_k_m",
                "mistral_7b_instruct_v0_3_q4_k_m",
                "qwen2_5_7b_instruct_q4_k_m",
            )
            for protocol in (
                "direct_subword_calgacus",
                "nonseg_ascii_b16",
                "nonseg_ascii_b8",
                "nonseg_hex_nibble_b16",
                "segmented_hex_multi_topic",
                "segmented_hex_single_topic",
            )
            for outcome, point in overhead_points.items()
        ]
    ).to_csv(overhead_source, index=False)

    detector_source = tmp_path / "detector.csv"
    pd.DataFrame(
        [
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
                "split_count": 2,
                "estimate_median_across_splits": 0.75,
                "estimate_min_across_splits": 0.70,
                "estimate_max_across_splits": 0.80,
                "test_rows_sum_across_splits": 80,
                "payload_groups_sum_across_splits": 40,
                "cross_split_interval": (
                    "not_computed_heterogeneous_prespecified_splits"
                ),
            }
            for detector in (
                "published_textcnn_equivalent",
                "deberta_v3_base_classifier",
            )
            for regime in (
                "matched",
                "held_out_template",
                "leave_one_model",
                "leave_one_codec",
            )
            for metric in (
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "precision",
                "brier_score",
                "tpr_at_fpr_0.01",
            )
        ]
    ).to_csv(detector_source, index=False)

    return (
        _manifest(tmp_path / "robustness_manifest.json", "plot_source", robustness_source),
        _manifest(tmp_path / "theory_manifest.json", "validation", theory_source),
        _manifest(tmp_path / "readability_manifest.json", "summary", readability_source),
        _manifest(tmp_path / "overhead_manifest.json", "plot_source", overhead_source),
        _manifest(tmp_path / "detector_manifest.json", "plot_source", detector_source),
    )


def test_builder_emits_hashed_figures_and_compact_source_tables(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    output = tmp_path / "figures"
    artifacts = build_core_figures(
        robustness_manifest=robustness,
        theory_manifest=theory,
        readability_manifest=readability,
        overhead_manifest=overhead,
        detector_manifest=detector,
        output_dir=output,
        command="python scripts/build_revision_figures.py --fixture",
    )

    assert artifacts.summary == {
        "figure_count": 5,
        "rendered_file_count": 10,
        "source_table_count": 5,
    }
    assert (output / OUTPUT_FILENAMES["robustness_pdf"]).read_bytes().startswith(
        b"%PDF"
    )
    assert (output / OUTPUT_FILENAMES["theory_png"]).read_bytes().startswith(
        b"\x89PNG"
    )
    assert (output / OUTPUT_FILENAMES["overhead_pdf"]).read_bytes().startswith(
        b"%PDF"
    )
    assert (output / OUTPUT_FILENAMES["detector_png"]).read_bytes().startswith(
        b"\x89PNG"
    )
    manifest = json.loads(
        (output / OUTPUT_FILENAMES["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["human_rating_figures_emitted"] is False
    assert (
        manifest["readability_figure_scope"]
        == "automated_surface_diagnostics_not_human_judgements"
    )
    assert manifest["source_tables_hash_validated_before_plotting"] is True
    assert manifest["detector_cross_split_bars"].endswith("not_confidence_interval")
    assert (
        manifest["detector_supplementary_panels_status"]
        == "exploratory_post_freeze"
    )
    for key, entry in manifest["outputs"].items():
        assert file_sha256(entry["path"]) == entry["sha256"], key


def test_builder_rejects_tampered_declared_source(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    robustness_source = tmp_path / "robustness.csv"
    robustness_source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FigureEvidenceError, match="hash mismatch"):
        build_core_figures(
            robustness_manifest=robustness,
            theory_manifest=theory,
            readability_manifest=readability,
            overhead_manifest=overhead,
            detector_manifest=detector,
            output_dir=tmp_path / "figures",
            command="fixture",
        )


def test_builder_refuses_unrequested_overwrite(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    (output / OUTPUT_FILENAMES["manifest"]).write_text("preserve", encoding="utf-8")
    with pytest.raises(FigureEvidenceError, match="Refusing to overwrite"):
        build_core_figures(
            robustness_manifest=robustness,
            theory_manifest=theory,
            readability_manifest=readability,
            overhead_manifest=overhead,
            detector_manifest=detector,
            output_dir=output,
            command="fixture",
        )


def test_builder_rejects_interval_that_excludes_estimate(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    robustness_source = tmp_path / "robustness.csv"
    frame = pd.read_csv(robustness_source)
    frame.loc[0, "ci_high"] = 0.20
    frame.to_csv(robustness_source, index=False)
    robustness = _manifest(
        tmp_path / "robustness_manifest.json", "plot_source", robustness_source
    )
    with pytest.raises(FigureEvidenceError, match="excluding its estimate"):
        build_core_figures(
            robustness_manifest=robustness,
            theory_manifest=theory,
            readability_manifest=readability,
            overhead_manifest=overhead,
            detector_manifest=detector,
            output_dir=tmp_path / "figures",
            command="fixture",
        )


def test_builder_rejects_incomplete_frozen_overhead_grid(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    overhead_source = tmp_path / "overhead.csv"
    frame = pd.read_csv(overhead_source).iloc[1:].copy()
    frame.to_csv(overhead_source, index=False)
    overhead = _manifest(
        tmp_path / "overhead_manifest.json", "plot_source", overhead_source
    )
    with pytest.raises(FigureEvidenceError, match="complete frozen primary grid"):
        build_core_figures(
            robustness_manifest=robustness,
            theory_manifest=theory,
            readability_manifest=readability,
            overhead_manifest=overhead,
            detector_manifest=detector,
            output_dir=tmp_path / "figures",
            command="fixture",
        )


def test_builder_rejects_incomplete_detector_grid(tmp_path: Path):
    robustness, theory, readability, overhead, detector = _fixture(tmp_path)
    detector_source = tmp_path / "detector.csv"
    pd.read_csv(detector_source).iloc[1:].to_csv(detector_source, index=False)
    detector = _manifest(
        tmp_path / "detector_manifest.json", "plot_source", detector_source
    )
    with pytest.raises(FigureEvidenceError, match="complete frozen grid"):
        build_core_figures(
            robustness_manifest=robustness,
            theory_manifest=theory,
            readability_manifest=readability,
            overhead_manifest=overhead,
            detector_manifest=detector,
            output_dir=tmp_path / "figures",
            command="fixture",
        )
