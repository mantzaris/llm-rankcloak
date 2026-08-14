import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rankcloak.revision_artifacts import canonical_json_sha256
from rankcloak.revision_reporting import build_revision_reports
from scripts.revise_revision_manuscripts import (
    GENERATED_RESULTS_NAME,
    PLACEHOLDER_MARKERS,
    ManuscriptRevisionError,
    _verify_no_exploratory_report_rows,
    build_manuscript_package,
    finalize_sources,
    verify_manuscript_manifest,
)
from tests.test_revision_reporting import (
    _detector_fixture,
    _evaluator_unavailability_fixture,
    _mixed_model_fixture,
    _mutate_statistics_csv,
    _preprocessing_fixture,
    _runtime_fixture,
    _statistics_fixture,
    _theory_fixture,
    _write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = PROJECT_ROOT / ".paper" / "scientific_reports"
PROTECTED_NAMES = (
    "main.tex",
    "supplementary.tex",
    "rankcloak_scientific_reports_manuscript.pdf",
    "rankcloak_scientific_reports_supplementary.pdf",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _progress_fixture(root):
    manifest = {
        "schema_version": "rankcloak-revision-confirmatory-progress-v1",
        "counts": {"remaining": 0, "failures": 0},
        "gpu": {"cumulative_actual_gpu_hours": 12.375},
    }
    manifest["progress_sha256"] = canonical_json_sha256(manifest)
    path = root / "progress" / "canonical_progress.json"
    _write_json(path, manifest)
    return path


def _report_fixture(root):
    statistics = _statistics_fixture(root)

    # Reporting fixtures exercise numeric plumbing, but this manuscript fixture
    # must model the authorized study state: no participant-derived outcome.
    _mutate_statistics_csv(
        statistics,
        "continuous",
        lambda rows: rows.__setitem__(
            slice(None),
            [row for row in rows if row.get("outcome") != "human_naturalness"],
        ),
    )
    mixed_model = _mixed_model_fixture(root)
    unavailability = _evaluator_unavailability_fixture(root)
    build = build_revision_reports(
        output_dir=root / "reports",
        statistics_manifest=statistics,
        theory_manifest=_theory_fixture(root),
        detector_manifest=_detector_fixture(root),
        mixed_model_manifest=mixed_model,
        evaluator_unavailability_manifest=unavailability,
        runtime_manifests=(_runtime_fixture(root),),
        preprocessing_manifests=(_preprocessing_fixture(root),),
        fixture_mode=True,
    )
    return build, statistics, mixed_model, unavailability


def _render_fixture_figures(report_root, figures_dir):
    figures_dir.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(report_root / "plots" / "plot_revision_figures.py"),
            "--registry",
            str(report_root / "plots" / "plot_registry.csv"),
            "--output-dir",
            str(figures_dir),
            "--format",
            "pdf",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )


def test_actual_templates_materialize_without_computational_placeholders():
    sources = finalize_sources(manuscript_root=MANUSCRIPT_ROOT)
    assert set(sources) == {"main2", "supplementary2", "response_letter"}
    for source in sources.values():
        assert source.count(r"\input{" + GENERATED_RESULTS_NAME + "}") == 1
        assert not any(marker in source for marker in PLACEHOLDER_MARKERS)
        assert "PendingResult" not in source
        assert "PendingSI" not in source
        assert "PendingValue" not in source


def test_real_preflight_fails_closed_when_confirmatory_reports_are_missing(tmp_path):
    output = tmp_path / "must_not_exist"
    absent = tmp_path / "missing" / "report_output_manifest.json"
    with pytest.raises(ManuscriptRevisionError, match="cannot read report output manifest"):
        build_manuscript_package(
            report_manifest=absent,
            figures_dir=tmp_path / "missing_figures",
            statistics_manifest=tmp_path / "missing_statistics.json",
            mixed_model_manifest=tmp_path / "missing_mixed_models.json",
            progress_manifest=tmp_path / "missing_progress.json",
            evaluator_unavailability_manifest=tmp_path / "missing_unavailability.json",
            manuscript_root=MANUSCRIPT_ROOT,
            output_dir=output,
            preflight_only=True,
        )
    assert not output.exists()


@pytest.mark.skipif(
    shutil.which("latexmk") is None,
    reason="latexmk is required by the production manuscript gate",
)
def test_fixture_package_centralizes_values_compiles_and_preserves_originals(tmp_path):
    build, statistics, mixed_model, unavailability = _report_fixture(tmp_path)
    figures = tmp_path / "figures"
    _render_fixture_figures(build.output_dir, figures)
    progress = _progress_fixture(tmp_path)
    output = tmp_path / "manuscript_package"
    protected_before = {
        name: _sha256(MANUSCRIPT_ROOT / name) for name in PROTECTED_NAMES
    }

    manifest = build_manuscript_package(
        report_manifest=build.output_dir / "report_output_manifest.json",
        figures_dir=figures,
        statistics_manifest=statistics,
        mixed_model_manifest=mixed_model,
        progress_manifest=progress,
        evaluator_unavailability_manifest=unavailability,
        manuscript_root=MANUSCRIPT_ROOT,
        output_dir=output,
        fixture_mode=True,
    )

    assert manifest == verify_manuscript_manifest(output)
    assert manifest["fixture_mode"] is True
    assert manifest["non_scientific_fixture"] is True
    assert manifest["originals_preserved"] is True
    assert manifest["all_computational_placeholders_resolved"] is True
    assert manifest["cross_document_values_consistent"] is True
    assert manifest["main_word_limit_satisfied"] is True
    assert manifest["main_display_limit_satisfied"] is True
    assert manifest["pdf_compilation_passed"] is True
    assert manifest["final_progress_snapshot_sha256"] == _sha256(progress)
    assert manifest["evaluator_unavailability_manifest_sha256"] == _sha256(
        unavailability
    )
    assert {row["role"] for row in manifest["provenance_manifests"]} == {
        "report_manifest",
        "report_integrity",
        "statistics_manifest",
        "mixed_model_manifest",
        "final_progress_snapshot",
        "evaluator_unavailability",
    }
    assert {row["role"] for row in manifest["compiled_pdfs"]} == {
        "compiled_pdf:main2",
        "compiled_pdf:supplementary2",
        "compiled_pdf:response_letter2",
    }
    assert protected_before == {
        name: _sha256(MANUSCRIPT_ROOT / name) for name in PROTECTED_NAMES
    }

    value_manifest = json.loads(
        (output / "cross_document_value_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    recovery = {
        row["value_id"]: row for row in value_manifest["values"]
        if row["value_id"].startswith("primary_recovery_")
    }
    assert len(recovery) == 5
    assert all(
        row["documents"] == ["main2", "response_letter", "supplementary2"]
        for row in recovery.values()
    )
    assert value_manifest["human_participants_recruited"] is False
    assert value_manifest["human_outcomes_estimated"] is False
    assert value_manifest["public_doi_released"] is False

    generated = (output / GENERATED_RESULTS_NAME).read_text(encoding="utf-8")
    assert "No participants were recruited and no human outcome was estimated" in generated
    assert "no public DOI was released" in generated
    for name in ("main2.tex", "supplementary2.tex", "response_letter2.tex"):
        source = (output / name).read_text(encoding="utf-8")
        assert not any(marker in source for marker in PLACEHOLDER_MARKERS)
        assert source.count(r"\input{" + GENERATED_RESULTS_NAME + "}") == 1

    build_arguments = {
        "report_manifest": build.output_dir / "report_output_manifest.json",
        "figures_dir": figures,
        "statistics_manifest": statistics,
        "mixed_model_manifest": mixed_model,
        "progress_manifest": progress,
        "evaluator_unavailability_manifest": unavailability,
        "manuscript_root": MANUSCRIPT_ROOT,
        "output_dir": output,
        "fixture_mode": True,
    }
    progress_bytes = progress.read_bytes()
    progress.write_bytes(progress_bytes + b"\n")
    with pytest.raises(ManuscriptRevisionError, match="different sealed inputs"):
        build_manuscript_package(**build_arguments)
    progress.write_bytes(progress_bytes)

    unavailability_bytes = unavailability.read_bytes()
    unavailability.write_bytes(unavailability_bytes + b"\n")
    with pytest.raises(ManuscriptRevisionError, match="different sealed inputs"):
        build_manuscript_package(**build_arguments)
    unavailability.write_bytes(unavailability_bytes)
    assert verify_manuscript_manifest(output)["manifest_sha256"] == manifest[
        "manifest_sha256"
    ]

    # The anti-pooling gate is independently deterministic and scans the exact
    # 15-table/18-plot package. A single available smoke-labelled row is fatal.
    table_path = build.output_dir / "tables" / "main_table_1.csv"
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["phase"] = "smoke"
    _write_csv(table_path, rows)
    with pytest.raises(ManuscriptRevisionError, match="exploratory evidence leaked"):
        _verify_no_exploratory_report_rows(build.output_dir)
