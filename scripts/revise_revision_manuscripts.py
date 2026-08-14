#!/usr/bin/env python3
"""Build the final revision TeX package from sealed machine products.

The command is deliberately deterministic and non-generative.  It does not
invent a result or mutate the author manuscript sources.  Numeric recovery
prose is derived from the hash-verified generated main table; all other
computational estimates stay in the generated tables and figures.  Human
outcomes and a public DOI remain explicitly unavailable because neither action
is authorized by the confirmatory execution approval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_artifacts import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
)
from rankcloak.revision_reporting import (  # noqa: E402
    verify_report_output_manifest,
)
from tools.count_scientific_reports import (  # noqa: E402
    build_report,
    command_arguments,
)


SCHEMA_VERSION = "rankcloak-revision-manuscript-run-v1"
REPORT_SCHEMA = "rankcloak-revision-report-v1"
STATISTICS_SCHEMA = "1.0"
MIXED_MODEL_TYPE = "rankcloak_revision_v1_mixed_model_run"
UNAVAILABILITY_SCHEMA = (
    "rankcloak-heldout-evaluator-upstream-unavailability-v1"
)
PROGRESS_SCHEMA = "rankcloak-revision-confirmatory-progress-v1"
HARD_GPU_CEILING = 165.0
FROZEN_EVALUATOR_TARGET = 17_280
SCOREABLE_EVALUATOR_ROWS = 17_232
UPSTREAM_UNAVAILABLE_ROWS = 48
MAIN_DISPLAY_LIMIT = 7
MAIN_WORD_LIMIT = 4_500
ABSTRACT_WORD_LIMIT = 200
TITLE_WORD_LIMIT = 20
FIGURE_LEGEND_WORD_LIMIT = 350
VALUE_MANIFEST_SCHEMA = "rankcloak-revision-cross-document-values-v1"
GENERATED_RESULTS_NAME = "generated_results.tex"
VALUE_MANIFEST_NAME = "cross_document_value_manifest.json"

TEMPLATE_NAMES = ("main2.tex", "supplementary2.tex", "response_letter2.tex")
SUPPORT_NAMES = (
    "wlscirep.cls",
    "references2.bib",
    "naturemag-doi.bst",
    "jabbrv.sty",
    "jabbrv-ltwa-en.ldf",
    "jabbrv-ltwa-all.ldf",
)
PERMITTED_UNAVAILABLE_TABLES = {"supplementary_table_s10"}
PERMITTED_UNAVAILABLE_PLOTS = {
    "main_figure_3",
    "supplementary_figure_s8",
}
PROTECTED_SUBMITTED_ARTIFACTS = {
    "main.tex": "17e1045c098184b9472ded03e2dc16a26e451d8676e1ec982114ed8a9a545d74",
    "supplementary.tex": "93208ad95613d4fcd9eb60d5307c364fa783505408ca384ed9fd00bd2d75b995",
    "rankcloak_scientific_reports_manuscript.pdf": (
        "ac90fd962f48117b8549e5488a543f42b777d58f773a81aeb5fca1038de74703"
    ),
    "rankcloak_scientific_reports_supplementary.pdf": (
        "d1e9f57ddbfaf4daaaf92caeea33412813796adae8aa64f539d74f3ddf2bf219"
    ),
}
PLACEHOLDER_MARKERS = (
    r"\PendingResult{",
    r"\PendingSI{",
    r"\PendingValue{",
    r"\PendingPanel{",
    r"\PendingSIPanel{",
    "INTERNAL GENERATED-RESULT PLACEHOLDER",
    "INTERNAL GENERATED-EVIDENCE PLACEHOLDER",
    "INTERNAL PLACEHOLDER---NOT A RESULT",
    "machine-generated values unavailable",
    "This placeholder will be replaced",
)
EXPLORATORY_EVIDENCE_RE = re.compile(
    r"(?:^|[_/\s-])(?:smoke|pilot|exploratory)(?:$|[_/\s-])",
    re.IGNORECASE,
)


class ManuscriptRevisionError(RuntimeError):
    """Fail-closed manuscript packaging error."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManuscriptRevisionError(
            "cannot read {} at {}: {}".format(label, path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise ManuscriptRevisionError("{} must be a JSON object".format(path))
    return value


def _file_identity(path: Path, *, role: str, relative_to: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ManuscriptRevisionError("required regular file is absent: {}".format(path))
    declared = (
        str(path.relative_to(relative_to))
        if relative_to is not None
        else str(path.resolve())
    )
    return {
        "role": role,
        "path": declared,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _resolve_declared(manifest_path: Path, raw: object) -> Path:
    candidate = Path(str(raw))
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _verify_declarations(
    manifest_path: Path,
    declarations: Iterable[Mapping[str, Any]],
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> None:
    for declaration in declarations:
        path = _resolve_declared(manifest_path, declaration.get(path_key, ""))
        if not path.is_file() or path.is_symlink():
            raise ManuscriptRevisionError(
                "declared source is missing or unsafe: {}".format(path)
            )
        if file_sha256(path) != declaration.get(hash_key):
            raise ManuscriptRevisionError(
                "declared source hash mismatch: {}".format(path)
            )
        size = declaration.get("size_bytes", declaration.get("bytes"))
        if size is not None and path.stat().st_size != int(size):
            raise ManuscriptRevisionError(
                "declared source size mismatch: {}".format(path)
            )


def _verify_self_hash(manifest: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(manifest)
    claimed = unsigned.pop(field, None)
    if not claimed or canonical_json_sha256(unsigned) != claimed:
        raise ManuscriptRevisionError("{} self-hash mismatch".format(label))


def _protected_snapshot(
    manuscript_root: Path, *, fixture_mode: bool
) -> list[dict[str, Any]]:
    """Hash the four immutable submitted artifacts.

    Production builds require the hashes recorded before revision work began.
    Fixture builds still require all four files and compare their before/after
    snapshots, but do not require fixture bytes to equal the production pins.
    """

    records: list[dict[str, Any]] = []
    for name, expected_sha256 in PROTECTED_SUBMITTED_ARTIFACTS.items():
        path = manuscript_root / name
        record = _file_identity(path, role="protected_submitted:" + name)
        record["expected_sha256"] = (
            record["sha256"] if fixture_mode else expected_sha256
        )
        if record["sha256"] != record["expected_sha256"]:
            raise ManuscriptRevisionError(
                "protected submitted artifact hash mismatch: {}".format(path)
            )
        records.append(record)
    return records


def _verify_protected_unchanged(
    before: Sequence[Mapping[str, Any]],
    manuscript_root: Path,
    *,
    fixture_mode: bool,
) -> list[dict[str, Any]]:
    after = _protected_snapshot(manuscript_root, fixture_mode=fixture_mode)
    before_identity = [
        (row["role"], row["path"], row["size_bytes"], row["sha256"])
        for row in before
    ]
    after_identity = [
        (row["role"], row["path"], row["size_bytes"], row["sha256"])
        for row in after
    ]
    if before_identity != after_identity:
        raise ManuscriptRevisionError(
            "a protected submitted source or PDF changed during manuscript build"
        )
    return after


def _verify_report_source_seal(
    report_manifest: Path,
    *,
    statistics_manifest: Path,
    mixed_model_manifest: Path,
    fixture_mode: bool,
) -> dict[str, Any]:
    seal = _read_json(
        report_manifest.parent / "report_source_manifest.json",
        "report source seal",
    )
    recovery = seal.get("recovery_reporting_contract")
    if (
        seal.get("schema_version") != REPORT_SCHEMA
        or seal.get("artifact_type") != "report_source_seal"
        or seal.get("fixture_mode") is not fixture_mode
        or seal.get("numeric_input_policy")
        != "manifest-addressed machine artifacts only; no numeric overrides"
        or not isinstance(recovery, dict)
        or recovery.get("primary_outcome") != "exact_payload_recovery"
        or recovery.get("semantics")
        != "original_serialized_payload_bytes_sha256_v1"
        or recovery.get("exact_rank_replay_role") != "diagnostic_only"
    ):
        raise ManuscriptRevisionError(
            "report source seal violates the result-provenance contract"
        )
    manifests = seal.get("manifests")
    if not isinstance(manifests, list):
        raise ManuscriptRevisionError("report source seal lacks manifest identities")
    by_key = {
        str(row.get("source_key")): row
        for row in manifests
        if isinstance(row, dict)
    }
    expected = {
        "statistics": statistics_manifest,
        "mixed_model": mixed_model_manifest,
    }
    for source_key, path in expected.items():
        row = by_key.get(source_key)
        if row is None or row.get("sha256") != file_sha256(path):
            raise ManuscriptRevisionError(
                "report source seal is not bound to supplied {} manifest".format(
                    source_key
                )
            )
    return seal


def _verify_no_exploratory_report_rows(report_root: Path) -> None:
    """Reject any available display row carrying smoke/pilot identity."""

    identity_fields = (
        "phase",
        "study_phase",
        "evidence_status",
        "trial_id",
        "source_trial_id",
        "work_id",
    )
    csv_paths = sorted((report_root / "tables").glob("*.csv"))
    csv_paths.extend(sorted((report_root / "plots" / "sources").glob("*.csv")))
    if len(csv_paths) != 33:
        raise ManuscriptRevisionError(
            "report package is not the exact 15-table/18-plot product set"
        )
    for path in csv_paths:
        for index, row in enumerate(_read_csv(path), start=2):
            if row.get("report_status") != "available":
                continue
            identity = " ".join(row.get(field, "") for field in identity_fields)
            if EXPLORATORY_EVIDENCE_RE.search(identity):
                raise ManuscriptRevisionError(
                    "exploratory evidence leaked into {} row {}".format(path, index)
                )


def _verify_statistics(path: Path) -> dict[str, Any]:
    manifest = _read_json(path, "statistics manifest")
    if manifest.get("schema_version") != STATISTICS_SCHEMA:
        raise ManuscriptRevisionError("unsupported statistics manifest schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ManuscriptRevisionError("statistics manifest lacks output identities")
    _verify_declarations(path, outputs.values())
    return manifest


def _verify_mixed_models(path: Path) -> dict[str, Any]:
    manifest = _read_json(path, "mixed-model manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_type") != MIXED_MODEL_TYPE
        or manifest.get("validation_only") is not False
        or manifest.get("fixed_effects_fallback") is not False
    ):
        raise ManuscriptRevisionError(
            "mixed-model manifest is not the locked confirmatory result"
        )
    outputs = manifest.get("outputs")
    declarations = outputs.values() if isinstance(outputs, dict) else outputs
    if not isinstance(declarations, list) and not isinstance(declarations, tuple):
        try:
            declarations = list(declarations)
        except TypeError as exc:
            raise ManuscriptRevisionError(
                "mixed-model manifest lacks output identities"
            ) from exc
    if not declarations:
        raise ManuscriptRevisionError("mixed-model manifest has no outputs")
    _verify_declarations(path, declarations)
    return manifest


def _verify_unavailability(path: Path) -> dict[str, Any]:
    manifest = _read_json(path, "held-out evaluator unavailability manifest")
    _verify_self_hash(manifest, "manifest_sha256", "unavailability manifest")
    if (
        manifest.get("schema_version") != UNAVAILABILITY_SCHEMA
        or manifest.get("manifest_type")
        != "heldout_evaluator_upstream_dependent_unavailability"
        or manifest.get("frozen_evaluator_target_units")
        != FROZEN_EVALUATOR_TARGET
        or manifest.get("scoreable_evaluator_units")
        != SCOREABLE_EVALUATOR_ROWS
        or manifest.get("upstream_dependent_unavailable_units")
        != UPSTREAM_UNAVAILABLE_ROWS
        or manifest.get("terminal_accounted_units")
        != FROZEN_EVALUATOR_TARGET
        or manifest.get("scoring_attempted_for_unavailable_units") is not False
        or manifest.get("scores_imputed_or_fabricated") is not False
    ):
        raise ManuscriptRevisionError(
            "held-out evaluator unavailability accounting is not exact"
        )
    files = manifest.get("source_files")
    units = manifest.get("units")
    if (
        not isinstance(files, list)
        or len(files) != 4
        or manifest.get("source_files_sha256") != canonical_json_sha256(files)
        or not isinstance(units, list)
        or len(units) != UPSTREAM_UNAVAILABLE_ROWS
        or manifest.get("units_sha256") != canonical_json_sha256(units)
    ):
        raise ManuscriptRevisionError("unavailability lineage declaration is malformed")
    _verify_declarations(path, files)
    return manifest


def _verify_progress(path: Path, *, fixture_mode: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ManuscriptRevisionError(
            "immutable final progress snapshot is missing or unsafe: {}".format(
                path
            )
        )
    if not fixture_mode and path.name != "final_progress_snapshot_v1.json":
        raise ManuscriptRevisionError(
            "production manuscript input must be final_progress_snapshot_v1.json"
        )
    progress = _read_json(path, "canonical progress manifest")
    _verify_self_hash(progress, "progress_sha256", "canonical progress")
    counts = progress.get("counts")
    gpu = progress.get("gpu")
    if (
        progress.get("schema_version") != PROGRESS_SCHEMA
        or not isinstance(counts, dict)
        or int(counts.get("remaining", -1)) != 0
        or int(counts.get("failures", -1)) != 0
        or not isinstance(gpu, dict)
    ):
        raise ManuscriptRevisionError(
            "canonical progress does not describe complete failure-free execution"
        )
    actual = float(gpu.get("cumulative_actual_gpu_hours", -1.0))
    if not 0.0 <= actual <= HARD_GPU_CEILING:
        raise ManuscriptRevisionError("canonical actual GPU use violates the ceiling")
    return progress


def _verify_report(
    path: Path, *, fixture_mode: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(path, "report output manifest")
    if (
        manifest.get("schema_version") != REPORT_SCHEMA
        or manifest.get("artifact_type") != "report_output_manifest"
    ):
        raise ManuscriptRevisionError("unsupported report output manifest")
    verification = verify_report_output_manifest(path.parent)
    if verification.get("status") != "ok":
        raise ManuscriptRevisionError(
            "report output verification failed: {}".format(
                "; ".join(map(str, verification.get("errors", [])))
            )
        )
    integrity_path = path.parent / "report_integrity.json"
    integrity = _read_json(integrity_path, "report integrity manifest")
    if (
        integrity.get("schema_version") != REPORT_SCHEMA
        or integrity.get("status") != "passed"
        or integrity.get("fixture_mode") is not fixture_mode
        or integrity.get("sample_size_consistency") != "passed"
        or integrity.get("source_hash_validation") != "passed"
        or integrity.get("main_display_count") != MAIN_DISPLAY_LIMIT
        or integrity.get("main_display_limit") != MAIN_DISPLAY_LIMIT
        or integrity.get("primary_inference_source")
        != "locked_r_mixed_model_manifest"
        or integrity.get("python_pooled_effects_are_primary_inference") is not False
    ):
        raise ManuscriptRevisionError(
            "report integrity is not complete confirmatory evidence"
        )
    table_status = integrity.get("table_status")
    plot_status = integrity.get("plot_status")
    if not isinstance(table_status, dict) or not isinstance(plot_status, dict):
        raise ManuscriptRevisionError("report integrity lacks display status maps")
    if len(table_status) != 15 or len(plot_status) != 18:
        raise ManuscriptRevisionError("report display registry is incomplete")
    if fixture_mode:
        bad_status = [
            "{}={!r}".format(display_id, status)
            for display_id, status in {**table_status, **plot_status}.items()
            if status not in {"available", "unavailable"}
        ]
        if bad_status:
            raise ManuscriptRevisionError(
                "fixture report has invalid display states: {}".format(
                    ", ".join(bad_status)
                )
            )
    else:
        for display_id, status in table_status.items():
            expected = (
                "unavailable"
                if display_id in PERMITTED_UNAVAILABLE_TABLES
                else "available"
            )
            if status != expected:
                raise ManuscriptRevisionError(
                    "{} has status {!r}; expected {!r}".format(
                        display_id, status, expected
                    )
                )
        for display_id, status in plot_status.items():
            expected = (
                "unavailable"
                if display_id in PERMITTED_UNAVAILABLE_PLOTS
                else "available"
            )
            if status != expected:
                raise ManuscriptRevisionError(
                    "{} has status {!r}; expected {!r}".format(
                        display_id, status, expected
                    )
                )
    return manifest, integrity


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ManuscriptRevisionError("cannot read generated table {}".format(path)) from exc


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ManuscriptRevisionError("invalid primary recovery totals")
    z = 1.959963984540054
    proportion = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def primary_recovery_summary(
    report_root: Path, *, fixture_mode: bool
) -> dict[str, Any]:
    rows = _read_csv(report_root / "tables" / "main_table_1.csv")
    if not rows or any(row.get("report_status") != "available" for row in rows):
        raise ManuscriptRevisionError("main Table 1 is not fully available")
    try:
        total = sum(int(row["n_payloads"]) for row in rows)
        successes = sum(int(row["payload_recovery_successes"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ManuscriptRevisionError(
            "main Table 1 lacks exact recovery counts"
        ) from exc
    if not fixture_mode and total != 6_480:
        raise ManuscriptRevisionError(
            "primary recovery table has {}, not 6,480 payload trials".format(total)
        )
    low, high = _wilson(successes, total)
    rate = successes / float(total)
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "latex": (
            "{:,}/{:,} primary trials recovered the original payload "
            "({:.2f}\\%; 95\\% Wilson CI {:.2f}--{:.2f}\\%)"
        ).format(
            successes,
            total,
            100.0 * rate,
            100.0 * low,
            100.0 * high,
        ),
    }


def _replace_command_uses(
    source: str, command: str, replacements: Sequence[str]
) -> str:
    uses = list(command_arguments(source, command))
    if len(uses) != len(replacements):
        raise ManuscriptRevisionError(
            "expected {} uses of \\{}, found {}".format(
                len(replacements), command, len(uses)
            )
        )
    result = source
    for (start, end, _), replacement in zip(reversed(uses), reversed(replacements)):
        result = result[:start] + replacement + result[end:]
    return result


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ManuscriptRevisionError(
            "{} expected exactly once in manuscript template".format(label)
        )
    return source.replace(old, new, 1)


def _braced_argument(source: str, opening: int) -> tuple[str, int]:
    while opening < len(source) and source[opening].isspace():
        opening += 1
    if opening >= len(source) or source[opening] != "{":
        raise ManuscriptRevisionError("expected a braced LaTeX argument")
    depth = 0
    index = opening
    while index < len(source):
        character = source[index]
        escaped = index > 0 and source[index - 1] == "\\"
        if not escaped and character == "{":
            depth += 1
        elif not escaped and character == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
        index += 1
    raise ManuscriptRevisionError("unbalanced LaTeX argument")


def _required_file_branches(source: str) -> str:
    """Resolve every final-package IfFileExists to its required true branch."""

    pattern = re.compile(r"\\IfFileExists")
    cursor = 0
    output: list[str] = []
    count = 0
    while True:
        match = pattern.search(source, cursor)
        if match is None:
            output.append(source[cursor:])
            break
        output.append(source[cursor : match.start()])
        first, first_end = _braced_argument(source, match.end())
        second, second_end = _braced_argument(source, first_end)
        _, third_end = _braced_argument(source, second_end)
        if not (
            first.startswith(r"\ReportRoot/")
            or first.startswith(r"\ReportFigureRoot/")
        ):
            raise ManuscriptRevisionError(
                "unexpected conditional file dependency in final template: {}".format(
                    first
                )
            )
        output.append(second)
        cursor = third_end
        count += 1
    if count == 0:
        raise ManuscriptRevisionError(
            "manuscript template had no generated-product file hooks"
        )
    return "".join(output)


def _expand_supplement_display_hooks(source: str) -> str:
    def replace(command: str, render: Any, expected: int) -> str:
        nonlocal source
        pattern = re.compile(r"\\" + re.escape(command))
        cursor = 0
        output: list[str] = []
        count = 0
        while True:
            match = pattern.search(source, cursor)
            if match is None:
                output.append(source[cursor:])
                break
            try:
                first, first_end = _braced_argument(source, match.end())
                second, second_end = _braced_argument(source, first_end)
                third, third_end = _braced_argument(source, second_end)
            except ManuscriptRevisionError:
                # The command name inside its own \newcommand declaration is not
                # a three-argument invocation.
                output.append(source[cursor : match.end()])
                cursor = match.end()
                continue
            output.append(source[cursor : match.start()])
            output.append(render(first, second, third))
            cursor = third_end
            count += 1
        source = "".join(output)
        if count != expected:
            raise ManuscriptRevisionError(
                "expected {} {} display hooks, found {}".format(
                    expected, command, count
                )
            )
        return source

    replace(
        "SuppTableHook",
        lambda display_id, _title, _label: (
            r"\input{\ReportRoot/tables/" + display_id + ".tex}"
        ),
        13,
    )
    replace(
        "SuppFigureHook",
        lambda display_id, title, label: "\n".join(
            (
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\includegraphics[width=0.95\linewidth]{\ReportFigureRoot/"
                + display_id
                + ".pdf}",
                r"\caption{\label{" + label + "}" + title + "}",
                r"\end{figure}",
            )
        ),
        13,
    )
    return source


def _main_replacements() -> list[str]:
    return [
        r"\RCAbstractResultSentence",
        r"\RCExecutionStatus",
        r"\RCCapacityResult",
        r"\RCPrimaryResult",
        r"\RCHumanStatus",
        r"\RCRobustnessResult",
        r"\RCDetectorRuntimeResult",
        r"\RCDiscussionResult",
    ]


def _supplement_replacements() -> list[str]:
    return [
        r"\RCSuppPrimaryResult",
        r"\RCSuppEffectsResult",
        r"\RCSuppRobustnessResult",
        r"\RCSuppHumanStatus",
        r"\RCSuppDetectorRuntimeResult",
    ]


def generated_result_macros(
    *,
    summary: Mapping[str, Any],
    gpu_hours: float,
    unavailable: int,
) -> tuple[bytes, dict[str, str], dict[str, dict[str, Any]]]:
    """Create the one authoritative LaTeX result vocabulary."""

    values: dict[str, dict[str, Any]] = {
        "primary_recovery_successes": {
            "macro": "RCPrimaryRecoverySuccesses",
            "raw_value": int(summary["successes"]),
            "formatted_latex": "{:,}".format(int(summary["successes"])),
            "derivation": "sum(payload_recovery_successes) over main_table_1.csv",
            "source_product": "tables/main_table_1.csv",
        },
        "primary_recovery_total": {
            "macro": "RCPrimaryRecoveryTotal",
            "raw_value": int(summary["total"]),
            "formatted_latex": "{:,}".format(int(summary["total"])),
            "derivation": "sum(n_payloads) over main_table_1.csv",
            "source_product": "tables/main_table_1.csv",
        },
        "primary_recovery_percent": {
            "macro": "RCPrimaryRecoveryPercent",
            "raw_value": 100.0 * float(summary["rate"]),
            "formatted_latex": "{:.2f}".format(100.0 * float(summary["rate"])),
            "derivation": "100 * successes / total",
            "source_product": "tables/main_table_1.csv",
        },
        "primary_recovery_ci_low_percent": {
            "macro": "RCPrimaryRecoveryCILowPercent",
            "raw_value": 100.0 * float(summary["ci_low"]),
            "formatted_latex": "{:.2f}".format(100.0 * float(summary["ci_low"])),
            "derivation": "two-sided 95 percent Wilson interval from table totals",
            "source_product": "tables/main_table_1.csv",
        },
        "primary_recovery_ci_high_percent": {
            "macro": "RCPrimaryRecoveryCIHighPercent",
            "raw_value": 100.0 * float(summary["ci_high"]),
            "formatted_latex": "{:.2f}".format(100.0 * float(summary["ci_high"])),
            "derivation": "two-sided 95 percent Wilson interval from table totals",
            "source_product": "tables/main_table_1.csv",
        },
        "cumulative_actual_gpu_hours": {
            "macro": "RCCumulativeActualGPUHours",
            "raw_value": float(gpu_hours),
            "formatted_latex": "{:.3f}".format(float(gpu_hours)),
            "derivation": "canonical progress cumulative_actual_gpu_hours",
            "source_product": "canonical_progress.json",
        },
        "hard_gpu_hour_ceiling": {
            "macro": "RCHardGPUHourCeiling",
            "raw_value": HARD_GPU_CEILING,
            "formatted_latex": "{:.0f}".format(HARD_GPU_CEILING),
            "derivation": "authorized hard-ceiling contract",
            "source_product": "canonical_progress.json",
        },
        "evaluator_target_units": {
            "macro": "RCEvaluatorTargetUnits",
            "raw_value": FROZEN_EVALUATOR_TARGET,
            "formatted_latex": "{:,}".format(FROZEN_EVALUATOR_TARGET),
            "derivation": "frozen held-out evaluator target",
            "source_product": "evaluator_unavailability_manifest.json",
        },
        "evaluator_scoreable_units": {
            "macro": "RCEvaluatorScoreableUnits",
            "raw_value": SCOREABLE_EVALUATOR_ROWS,
            "formatted_latex": "{:,}".format(SCOREABLE_EVALUATOR_ROWS),
            "derivation": "verified scoreable evaluator units",
            "source_product": "evaluator_unavailability_manifest.json",
        },
        "evaluator_unavailable_units": {
            "macro": "RCEvaluatorUnavailableUnits",
            "raw_value": int(unavailable),
            "formatted_latex": "{:,}".format(int(unavailable)),
            "derivation": "verified upstream-dependent unavailable evaluator units",
            "source_product": "evaluator_unavailability_manifest.json",
        },
    }
    prose: dict[str, str] = {
        "RCPrimaryRecoverySummary": (
            r"\RCPrimaryRecoverySuccesses/\RCPrimaryRecoveryTotal{} primary trials "
            r"recovered the original payload (\RCPrimaryRecoveryPercent\%; 95\% "
            r"Wilson CI \RCPrimaryRecoveryCILowPercent--"
            r"\RCPrimaryRecoveryCIHighPercent\%)"
        ),
        "RCAbstractResultSentence": r"\RCPrimaryRecoverySummary.",
        "RCExecutionStatus": (
            r"All frozen computational stages completed under the authorized "
            r"\RCHardGPUHourCeiling{} GPU-hour ceiling; canonical accounting "
            r"records \RCCumulativeActualGPUHours{} cumulative GPU-hours. "
            r"Exploratory smoke and the invalidated shard remain excluded."
        ),
        "RCCapacityResult": (
            r"Manifest-verified capacity and same-context quality validation is "
            r"reported in Fig.~\ref{fig:capacity-quality-framework}, "
            r"Supplementary Figs. S2--S3, and Supplementary Table S3."
        ),
        "RCPrimaryResult": (
            r"\RCPrimaryRecoverySummary; complete model--protocol estimates and "
            r"prespecified effects appear in Tables~\ref{tab:study-recovery} and "
            r"\ref{tab:effects-runtime} and Supplementary Tables S4--S6. "
            r"Rank/representation replay remains a separate diagnostic."
        ),
        "RCHumanStatus": (
            r"No participants were recruited and no human outcome was estimated; "
            r"the blinded materials and power analysis were prepared but remain "
            r"externally gated. Automated forced-span, full-message, readability, "
            r"and held-out-evaluator results are reported separately in "
            r"Supplementary Figs. S6 and S9 and Supplementary Tables S7 and S10."
        ),
        "RCRobustnessResult": (
            r"Saved-ID, retokenization, lead-in, transformation, and mitigation "
            r"results with payload-level uncertainty and first-divergence "
            r"classifications are reported in Supplementary Tables S7--S9 and "
            r"Supplementary Figs. S4--S7."
        ),
        "RCDetectorRuntimeResult": (
            r"Matched and held-out detector metrics with grouped uncertainty, "
            r"together with throughput and memory summaries, are reported in "
            r"Supplementary Tables S11--S12 and Supplementary Figs. S10--S12."
        ),
        "RCDiscussionResult": (
            r"\RCPrimaryRecoverySummary. The remaining quality, robustness, "
            r"detector, multilingual, and runtime evidence is reported with its "
            r"uncertainty in the sealed displays; human evidence remains unavailable."
        ),
        "RCSuppPrimaryResult": (
            r"\RCPrimaryRecoverySummary. Payload-trial denominators and Wilson "
            r"intervals appear in Table S4; representation replay is diagnostic. "
            r"\RCEvaluatorUnavailableUnits{} structurally unscoreable held-out-"
            r"evaluator units are terminal unavailable outcomes excluded from "
            r"quality estimands."
        ),
        "RCSuppEffectsResult": (
            r"Complete prespecified locked-R contrasts, interactions, and sensitivity "
            r"analyses are reported in Tables S5--S6 and Figs. S2--S3 and S13; "
            r"exploratory smoke is not used for inference."
        ),
        "RCSuppRobustnessResult": (
            r"Paired filter, lead-in, tail, segment-size, replay, transformation, "
            r"and first-divergence results are reported in Tables S7--S9 and "
            r"Figs. S4--S7. Condition-unavailable rows remain outside recovery "
            r"denominators."
        ),
        "RCSuppHumanStatus": (
            r"No participants were recruited and no human outcome was estimated. "
            r"Blinded materials and power analysis were prepared; automated "
            r"readability and held-out evaluator results are reported in Fig. S9 "
            r"and are not described as human naturalness."
        ),
        "RCSuppDetectorRuntimeResult": (
            r"Matched and held-out detector metrics, curves, calibration, and leakage "
            r"checks are reported in Table S11 and Figs. S10--S11; per-model/protocol "
            r"timing and memory appear in Table S12 and Fig. S12."
        ),
        "CapacityResult": (
            r"the capacity--quality validation in Figure 1 and Supplementary Table S3"
        ),
        "PrimaryRecoveryResult": (
            r"\RCPrimaryRecoverySummary (Table 1 and Supplementary Table S4)"
        ),
        "PrimaryQualityResult": (
            r"the locked primary quality and effective-rate contrasts in Table 2"
        ),
        "PromptResult": (
            r"the prespecified prompt-category contrasts in Supplementary Tables S5--S6"
        ),
        "FilterResult": (
            r"the filter-ablation estimates in Supplementary Table S7, with "
            r"unavailable conditions excluded"
        ),
        "LeadInResult": (
            r"the lead-in and replay estimates and first-divergence taxonomy in "
            r"Supplementary Table S8"
        ),
        "TailResult": r"the tail-policy contrasts in Supplementary Table S7",
        "HumanNaturalnessResult": (
            r"not estimated because no participants were recruited; blinded "
            r"materials and power analysis were prepared"
        ),
        "HumanSuspiciousnessResult": (
            r"not estimated because no participants were recruited; blinded "
            r"materials and power analysis were prepared"
        ),
        "DetectorResult": (
            r"the matched and held-out detector metrics in Supplementary Table S11"
        ),
        "RuntimeResult": (
            r"the throughput and memory summary in Supplementary Table S12"
        ),
        "MultilingualResult": (
            r"the Spanish and Mandarin computational estimates in Supplementary "
            r"Table S6 and Figure S13"
        ),
        "DOIResult": (
            r"no public DOI was released; deposit requires separate authorization"
        ),
    }
    lines = [
        "% Generated from hash-verified confirmatory report products; do not edit.",
        "% No human result and no public DOI are asserted by this file.",
    ]
    for value_id in sorted(values):
        value = values[value_id]
        lines.append(
            r"\newcommand{{\{}}}{{{}}}".format(
                value["macro"], value["formatted_latex"]
            )
        )
    for macro in sorted(prose):
        lines.append(r"\newcommand{{\{}}}{{{}}}".format(macro, prose[macro]))
    lines.append("")
    return "\n".join(lines).encode("utf-8"), prose, values


RESPONSE_STATUS_SPECS = (
    (
        "prose revision and evidence-governance",
        "The computational revision and evidence-governance audit are complete. Confirmatory values are bound to the sealed generated reports; human evidence and a public DOI remain separately unavailable.",
    ),
    (
        "Corpus generation, immutable manifesting",
        "Corpus generation, immutable manifesting, locked statistical analysis, and sealed reporting are complete. Computational estimates appear only in the generated tables and figures.",
    ),
    (
        "models are pinned and license-audited",
        "The models are pinned and license-audited, tokenizer preflight passed, and all primary, supporting, robustness, and held-out-evaluator shards completed under the authorized ceiling. Exploratory smoke remains excluded.",
    ),
    (
        "Computational and human-study preparation",
        "Confirmatory computation, detector training, and statistical reporting are complete. Human evidence remains unavailable: there was no recruitment, usability pilot, exposure, payment, or rating collection.",
    ),
    (
        "proposition, transformation framework",
        "The proposition, transformation framework, deterministic edits, provenance logging, failure taxonomy, and confirmatory robustness analysis are complete: \\LeadInResult.",
    ),
    (
        "Release packaging and validation",
        "Release packaging and validation are prepared; no public Zenodo deposit or DOI publication occurred. This editor requirement remains externally gated until explicit publication approval.",
    ),
    (
        "final transformation estimates",
        "Implemented; the sealed transformation estimates and uncertainty intervals appear in Supplementary Table S9 and Figs. S7.",
    ),
    (
        "Corpus implemented",
        "Corpus and primary execution are complete: \\PrimaryRecoveryResult.",
    ),
    (
        "Materials, power code",
        "Materials, power code, control-ingest tooling, randomization, and analysis code are implemented; recruitment and ratings did not occur.",
    ),
    (
        "machine-readable unavailable/dependency",
        "Implemented, including machine-readable unavailable/dependency records and exact source propagation. The confirmatory filter results are reported in Supplementary Table S7 and Fig. S4.",
    ),
    (
        "Historical diagnosis corrected",
        "Historical diagnosis corrected; serial execution and the complete confirmatory lead-in sweep are reported in Supplementary Table S8 and Fig. S5.",
    ),
    (
        "confirmatory quality, completion",
        "Implemented; confirmatory quality, completion, censoring, and effective-rate results are sealed in the generated reports.",
    ),
    (
        "Pipelines, split contracts",
        "Pipelines, split contracts, leakage checks, primary detector corpus, detector training, and all confirmatory metric fits are complete.",
    ),
    (
        "Balanced assignment and statistical",
        "Balanced assignment and the locked confirmatory statistical analysis are complete; prespecified estimates appear in Tables 1--2 and Supplementary Tables S4--S6.",
    ),
    (
        "Theory, endpoint-trace instrumentation",
        "Theory, endpoint-trace instrumentation, tests, and confirmatory empirical validation are complete; generated evidence appears in Figure 1 and Supplementary Table S3.",
    ),
    (
        "Implemented and hash-manifested",
        "Implemented and hash-manifested.",
    ),
    (
        "Three-model infrastructure",
        "Three-model infrastructure, tokenizer preflight, primary execution, and all supporting computational stages completed with exact manifest accounting.",
    ),
    (
        "Instrumentation and report generation",
        "Instrumentation, confirmatory profiling, and report generation are complete; representative throughput and memory results appear in Supplementary Table S12 and Fig. S12.",
    ),
    (
        "Reference audit and revised bibliography",
        "Reference audit and revised bibliography are implemented.",
    ),
    (
        "Designs, prompts, models",
        "Designs, prompts, models, manifests, transformations, and confirmatory multilingual execution are complete; results appear in Supplementary Table S6 and Fig. S13.",
    ),
    (
        "Materials and computational pipelines",
        "Computational pipelines and detector metrics are complete. Human materials remain externally gated, and no human outcome is claimed.",
    ),
    (
        "Conceptual comparisons",
        "Conceptual comparisons and two reproducibility/compatibility audits are complete. No patient-Huffman numeric cover-text result is claimed.",
    ),
)


def _response_status_replacements(source: str) -> list[str]:
    uses = list(command_arguments(source, "Status"))
    if len(uses) != len(RESPONSE_STATUS_SPECS):
        raise ManuscriptRevisionError(
            "response template status block count changed unexpectedly"
        )
    result = []
    for (_, _, observed), (needle, replacement) in zip(uses, RESPONSE_STATUS_SPECS):
        if needle not in observed:
            raise ManuscriptRevisionError(
                "response status order/content changed near {!r}".format(needle)
            )
        result.append("\\Status{" + replacement + "}")
    return result


def finalize_sources(
    *,
    manuscript_root: Path,
) -> dict[str, str]:
    main = (manuscript_root / "main2.tex").read_text(encoding="utf-8")
    supplement = (manuscript_root / "supplementary2.tex").read_text(encoding="utf-8")
    response = (manuscript_root / "response_letter2.tex").read_text(encoding="utf-8")

    main = _replace_command_uses(main, "PendingResult", _main_replacements())
    main = _replace_once(
        main,
        r"\newcommand{\ReportRoot}{../../results/revision_v1/reports/final}",
        r"\newcommand{\ReportRoot}{report}",
        "main report root",
    )
    main = _replace_once(
        main,
        r"\newcommand{\ReportFigureRoot}{\ReportRoot/plots/rendered}",
        r"\newcommand{\ReportFigureRoot}{figures}",
        "main figure root",
    )
    main = _replace_once(
        main,
        "\\usepackage{url}\n",
        "\\usepackage{url}\n\\input{" + GENERATED_RESULTS_NAME + "}\n",
        "main generated-result input",
    )
    main = _replace_once(
        main,
        r"\newcommand{\PendingResult}[1]{\textbf{[INTERNAL GENERATED-RESULT PLACEHOLDER: #1]}}",
        "",
        "main pending-result definition",
    )
    main = _replace_once(
        main,
        "\\newcommand{\\PendingPanel}[1]{%\n"
        "  \\fbox{\\parbox[c][38mm][c]{0.90\\linewidth}{\\centering\n"
        "  \\textbf{INTERNAL PLACEHOLDER---NOT A RESULT}\\\\[2mm]#1}}}",
        "\\newcommand{\\MissingGeneratedFigure}[1]{%\n"
        "  \\PackageError{rankcloak}{Required generated figure #1 is missing}%\n"
        "  {Run the sealed figure renderer before compiling.}}",
        "main pending-panel definition",
    )
    main = re.sub(
        r"\\PendingPanel\{Generated Main Figure ([1-5]):[^{}]*\}",
        r"\\MissingGeneratedFigure{\1}",
        main,
    )
    main = main.replace(
        "INTERNAL PLACEHOLDER---machine-generated values unavailable",
        "ERROR---required machine-generated table missing",
    )
    main = main.replace(
        "The generated table will report",
        "The generated table reports",
    )
    main = main.replace(
        "This placeholder will be replaced by the hash-verified generated table; "
        "unavailable conditions will remain outside recovery denominators.",
        "The hash-verified generated table keeps unavailable conditions outside "
        "recovery denominators.",
    )
    main = main.replace(
        "The final machine-generated panels will",
        "The machine-generated panels",
    )
    main = main.replace("Final panels will", "Panels")
    main = main.replace("The final figure will", "The figure")
    main = main.replace("Failures will be localized", "Failures are localized")
    main = main.replace(
        "The resulting measurements will describe",
        "The resulting measurements describe",
    )
    main = _replace_once(
        main,
        (
            "Primary outcomes are exact original-payload recovery, effective payload "
            "rate, human naturalness and suspiciousness, held-out probability, detector "
            "ROC-AUC, and encoding and decoding throughput."
        ),
        (
            "Primary outcomes span recovery, rate, quality, detection, and throughput; "
            "human outcomes remain unavailable."
        ),
        "abstract outcome sentence",
    )
    main = _replace_once(
        main,
        (
            "\\textbf{DOI: [INTERNAL PLACEHOLDER---NO PUBLIC DOI DEPOSIT OR RELEASE "
            "HAS BEEN MADE; REPLACE ONLY AFTER EXPLICIT PUBLICATION APPROVAL].} "
            "The final manuscript must not be submitted with this placeholder."
        ),
        (
            "\\textbf{No public DOI has been released.} Archival deposit and public "
            "release require separate authorization and are outside this confirmatory "
            "execution."
        ),
        "main DOI status",
    )

    supplement = _replace_command_uses(
        supplement, "PendingSI", _supplement_replacements()
    )
    supplement = _replace_once(
        supplement,
        r"\newcommand{\ReportRoot}{../../results/revision_v1/reports/final}",
        r"\newcommand{\ReportRoot}{report}",
        "supplement report root",
    )
    supplement = _replace_once(
        supplement,
        r"\newcommand{\ReportFigureRoot}{\ReportRoot/plots/rendered}",
        r"\newcommand{\ReportFigureRoot}{figures}",
        "supplement figure root",
    )
    supplement = _replace_once(
        supplement,
        "\\usepackage{url}\n",
        "\\usepackage{url}\n\\input{" + GENERATED_RESULTS_NAME + "}\n",
        "supplement generated-result input",
    )
    supplement = _replace_once(
        supplement,
        r"\newcommand{\PendingSI}[1]{\textbf{[INTERNAL GENERATED-EVIDENCE PLACEHOLDER: #1]}}",
        "",
        "supplement pending-result definition",
    )
    supplement = _replace_once(
        supplement,
        "\\newcommand{\\PendingSIPanel}[1]{%\n"
        "  \\fbox{\\parbox[c][35mm][c]{0.90\\linewidth}{\\centering\n"
        "  \\textbf{INTERNAL PLACEHOLDER---NOT A RESULT}\\\\[2mm]#1}}}",
        "\\newcommand{\\MissingGeneratedSIFigure}[1]{%\n"
        "  \\PackageError{rankcloak}{Required generated SI figure #1 is missing}%\n"
        "  {Run the sealed figure renderer before compiling.}}",
        "supplement pending-panel definition",
    )
    supplement = supplement.replace(
        r"{\PendingSIPanel{#2}}",
        r"{\MissingGeneratedSIFigure{#1}}",
    )
    supplement = supplement.replace(
        r"\caption{#2 \textbf{Internal draft: machine-generated panels are not yet available.}}",
        r"\caption{#2}",
    )
    supplement = supplement.replace(
        r"\textbf{Internal placeholder.} Values will be included only from the hash-verified reporting pipeline.",
        r"\textbf{Error.} The required hash-verified generated table is missing.",
    )
    supplement = supplement.replace(
        r"\caption{#2 \textbf{Internal draft: machine-generated values are not yet available.}}",
        r"\caption{#2}",
    )
    old_projection = (
        "The resulting compute projection was a prespecified no-go: 56.017000 point "
        "GPU-hours and 158.508216 conservative GPU-hours against the approved "
        "150-hour ceiling. Consequently no \\code{primary_v2} shard has been launched, "
        "and all confirmatory result hooks below remain unavailable. The projection "
        "and preflight manifest self-hashes are recorded in the reproduction audit "
        "rather than treated as result values."
    )
    new_projection = (
        "The original conservative projection was 158.508216 GPU-hours. Full execution "
        "was subsequently authorized under a hard \\RCHardGPUHourCeiling{} GPU-hour "
        "ceiling and completed with \\RCCumulativeActualGPUHours{} cumulative GPU-hours "
        "in the canonical accounting manifest. "
        "Projection and preflight self-hashes remain provenance, not scientific "
        "outcomes."
    )
    supplement = _replace_once(
        supplement, old_projection, new_projection, "supplement compute status"
    )
    supplement = supplement.replace("The final flow diagram will", "The flow diagram")
    supplement = supplement.replace("The final figure will", "The figure")
    supplement = supplement.replace("The final panels will", "The panels")
    supplement = supplement.replace("Final panels will", "Panels")
    supplement = supplement.replace("Panels will show", "Panels show")
    supplement = supplement.replace("will be plotted", "are plotted")
    supplement = supplement.replace("will be paired", "are paired")
    supplement = supplement.replace("tails will never", "tails do not")
    supplement = supplement.replace("The final matrix will", "The matrix")
    supplement = supplement.replace("Curves will include", "Curves include")
    supplement = supplement.replace("Every panel will identify", "Every panel identifies")
    supplement = supplement.replace("Final panels will separate", "Panels separate")
    supplement = supplement.replace("panels will report", "panels report")

    response = _replace_command_uses(
        response, "Status", _response_status_replacements(response)
    )
    response = _replace_once(
        response,
        "\\usepackage{xcolor}\n",
        "\\usepackage{xcolor}\n\\input{" + GENERATED_RESULTS_NAME + "}\n",
        "response generated-result input",
    )
    response = _replace_once(
        response,
        r"\newcommand{\PendingValue}[1]{\textcolor{internalred}{\textbf{[INTERNAL GENERATED-RESULT PLACEHOLDER: #1]}}}",
        "",
        "response pending-value definition",
    )
    for macro in (
        "CapacityResult",
        "PrimaryRecoveryResult",
        "PrimaryQualityResult",
        "PromptResult",
        "FilterResult",
        "LeadInResult",
        "TailResult",
        "HumanNaturalnessResult",
        "HumanSuspiciousnessResult",
        "DetectorResult",
        "RuntimeResult",
        "MultilingualResult",
        "DOIResult",
    ):
        pattern = re.compile(
            r"\\newcommand\{\\" + re.escape(macro) + r"\}\{\\PendingValue\{[^{}]*\}\}"
        )
        response, count = pattern.subn("", response)
        if count != 1:
            raise ManuscriptRevisionError(
                "response macro {} changed unexpectedly".format(macro)
            )
    response = _replace_once(
        response,
        r"\date{Major-revision working draft}",
        r"\date{Major-revision response}",
        "response date",
    )
    notice_pattern = re.compile(
        r"\\section\*\{Internal completion notice\}\n.*?(?=\\section\*\{Overview of the revision\})",
        re.DOTALL,
    )
    notice = (
        "\\section*{Evidence provenance notice}\n\n"
        "All computational values in this response are routed to the sealed, "
        "hash-verified reporting products. The full frozen computational design "
        "completed under the authorized \\RCHardGPUHourCeiling{} GPU-hour ceiling; "
        "canonical accounting records \\RCCumulativeActualGPUHours{} cumulative "
        "GPU-hours. Exploratory smoke and the invalidated "
        "Qwen shard remain excluded from confirmatory inference. Human evidence is "
        "unavailable because no recruitment or data collection occurred. No public "
        "deposit or DOI release was performed.\n\n"
    )
    response, count = notice_pattern.subn(lambda _: notice, response)
    if count != 1:
        raise ManuscriptRevisionError("response completion notice changed unexpectedly")
    response = response.replace(
        "Results will be inserted only from generated reports:",
        "Results are drawn only from generated reports:",
    )
    response = response.replace(
        "Every red generated-result macro is replaced exclusively from the sealed reporting products, and sample sizes and values agree across the manuscript, SI, and letter.",
        "Every computational result is bound to the sealed reporting products, and sample sizes and values agree across the manuscript, SI, and letter.",
    )
    response = response.replace(
        "No confirmatory value is inserted until the budget gate permits disjoint \\texttt{primary\\_v2} and supporting execution; exploratory smoke and the whole invalidated Qwen shard remain excluded.",
        "Every confirmatory value derives from the completed disjoint \\texttt{primary\\_v2} and supporting execution; exploratory smoke and the whole invalidated Qwen shard remain excluded.",
    )
    response = response.replace(
        "Before this letter is submitted, we will require all of the following:",
        "The completed revision satisfies the following internal gates; human evidence and public release remain separately gated:",
    )

    main = _required_file_branches(main)
    supplement = _required_file_branches(supplement)
    supplement = _expand_supplement_display_hooks(supplement)
    main = _replace_once(
        main,
        "\\newcommand{\\MissingGeneratedFigure}[1]{%\n"
        "  \\PackageError{rankcloak}{Required generated figure #1 is missing}%\n"
        "  {Run the sealed figure renderer before compiling.}}",
        "",
        "dead main figure fallback",
    )
    supplement = _replace_once(
        supplement,
        "\\newcommand{\\MissingGeneratedSIFigure}[1]{%\n"
        "  \\PackageError{rankcloak}{Required generated SI figure #1 is missing}%\n"
        "  {Run the sealed figure renderer before compiling.}}",
        "",
        "dead supplementary figure fallback",
    )
    supplement = _replace_once(
        supplement,
        "\\newcommand{\\SuppTableHook}[3]{%\n"
        "\\input{\\ReportRoot/tables/#1.tex}}",
        "",
        "materialized supplementary table hook definition",
    )
    supplement = _replace_once(
        supplement,
        "\\newcommand{\\SuppFigureHook}[3]{%\n"
        "\\begin{figure}[H]\n"
        "\\centering\n"
        "\\includegraphics[width=0.95\\linewidth]{\\ReportFigureRoot/#1.pdf}\n"
        "\\caption{#2}\n"
        "\\label{#3}\n"
        "\\end{figure}}",
        "",
        "materialized supplementary figure hook definition",
    )

    forbidden = PLACEHOLDER_MARKERS + (
        "no \\texttt{primary\\_v2} shard has been launched",
        "confirmatory execution has not begun",
        "confirmatory sweep has not run",
        "confirmatory empirical validation has not run",
        "blocked at the budget gate",
        "compute gate is no-go",
        "unlaunched primary corpus",
    )
    for name, source in (
        ("main2", main),
        ("supplementary2", supplement),
        ("response_letter", response),
    ):
        present = [needle for needle in forbidden if needle in source]
        if present:
            raise ManuscriptRevisionError(
                "{} retains stale computational placeholders: {}".format(
                    name, ", ".join(present)
                )
            )
    return {
        "main2": main,
        "supplementary2": supplement,
        "response_letter": response,
    }


def _macro_uses(source: str, macro: str) -> int:
    return len(
        re.findall(
            r"\\" + re.escape(macro) + r"(?:\{\})?(?![A-Za-z@])",
            source,
        )
    )


def _macro_dependencies(text: str, names: Iterable[str]) -> set[str]:
    return {name for name in names if _macro_uses(text, name)}


def _expand_generated_macros(
    source: str,
    *,
    prose: Mapping[str, str],
    values: Mapping[str, Mapping[str, Any]],
) -> str:
    expanded = source.replace(r"\input{" + GENERATED_RESULTS_NAME + "}", "")
    macros = {
        str(value["macro"]): str(value["formatted_latex"])
        for value in values.values()
    }
    macros.update({str(key): str(value) for key, value in prose.items()})
    for _ in range(len(macros) + 2):
        changed = False
        for macro in sorted(macros, key=len, reverse=True):
            pattern = re.compile(
                r"\\" + re.escape(macro) + r"(?:\{\})?(?![A-Za-z@])"
            )
            expanded, count = pattern.subn(
                lambda _match, replacement=macros[macro]: replacement,
                expanded,
            )
            changed = changed or bool(count)
        if not changed:
            break
    unresolved = [
        macro for macro in macros if _macro_uses(expanded, macro)
    ]
    if unresolved:
        raise ManuscriptRevisionError(
            "generated macro expansion did not converge: {}".format(
                ", ".join(sorted(unresolved))
            )
        )
    return expanded


def _materialize_table_inputs(source: str, table_root: Path) -> str:
    result = source
    for table in sorted(table_root.glob("*.tex")):
        marker = r"\input{\ReportRoot/tables/" + table.name + "}"
        if marker not in result:
            continue
        result = result.replace(marker, table.read_text(encoding="utf-8"))
    if r"\input{\ReportRoot/tables/" in result:
        raise ManuscriptRevisionError(
            "word audit could not materialize every generated table"
        )
    return result


def materialized_audit_sources(
    *,
    sources: Mapping[str, str],
    prose: Mapping[str, str],
    values: Mapping[str, Mapping[str, Any]],
    table_root: Path,
) -> dict[str, str]:
    main = _expand_generated_macros(
        sources["main2"], prose=prose, values=values
    )
    supplement = _expand_generated_macros(
        sources["supplementary2"], prose=prose, values=values
    )
    return {
        "main2": _materialize_table_inputs(main, table_root),
        "supplementary2": _materialize_table_inputs(supplement, table_root),
    }


def cross_document_value_manifest(
    *,
    sources: Mapping[str, str],
    prose: Mapping[str, str],
    values: Mapping[str, Mapping[str, Any]],
    generated_results_sha256: str,
    report_manifest: Path,
    statistics_manifest: Path,
    mixed_model_manifest: Path,
    progress_manifest: Path,
    evaluator_unavailability_manifest: Path,
    fixture_mode: bool,
) -> dict[str, Any]:
    document_sources = {
        "main2": sources["main2"],
        "supplementary2": sources["supplementary2"],
        "response_letter": sources["response_letter"],
    }
    for role, source in document_sources.items():
        if source.count(r"\input{" + GENERATED_RESULTS_NAME + "}") != 1:
            raise ManuscriptRevisionError(
                "{} does not include the centralized result file exactly once".format(
                    role
                )
            )

    all_names = set(prose)
    all_names.update(str(value["macro"]) for value in values.values())
    dependencies = {
        macro: _macro_dependencies(text, all_names)
        for macro, text in prose.items()
    }
    direct_by_document = {
        role: {
            macro: _macro_uses(source, macro)
            for macro in all_names
            if _macro_uses(source, macro)
        }
        for role, source in document_sources.items()
    }

    def reachable(seed: Iterable[str]) -> set[str]:
        seen = set(seed)
        frontier = list(seed)
        while frontier:
            macro = frontier.pop()
            for dependency in dependencies.get(macro, set()):
                if dependency not in seen:
                    seen.add(dependency)
                    frontier.append(dependency)
        return seen

    reachable_by_document = {
        role: reachable(uses)
        for role, uses in direct_by_document.items()
    }
    recovery_value_ids = {
        "primary_recovery_successes",
        "primary_recovery_total",
        "primary_recovery_percent",
        "primary_recovery_ci_low_percent",
        "primary_recovery_ci_high_percent",
    }
    for value_id in recovery_value_ids:
        macro = str(values[value_id]["macro"])
        missing = [
            role
            for role, reachable_macros in reachable_by_document.items()
            if macro not in reachable_macros
        ]
        if missing:
            raise ManuscriptRevisionError(
                "{} is not centralized across all documents: {}".format(
                    value_id, ", ".join(missing)
                )
            )

    source_paths = {
        "tables/main_table_1.csv": report_manifest.parent
        / "tables"
        / "main_table_1.csv",
        "canonical_progress.json": progress_manifest,
        "evaluator_unavailability_manifest.json": (
            evaluator_unavailability_manifest
        ),
    }
    value_rows = []
    for value_id in sorted(values):
        item = dict(values[value_id])
        source_path = source_paths[str(item["source_product"])]
        item.update(
            {
                "value_id": value_id,
                "source_sha256": file_sha256(source_path),
                "documents": sorted(
                    role
                    for role, reachable_macros in reachable_by_document.items()
                    if str(item["macro"]) in reachable_macros
                ),
            }
        )
        value_rows.append(item)

    prose_rows = [
        {
            "macro": macro,
            "template": prose[macro],
            "direct_document_uses": {
                role: uses[macro]
                for role, uses in direct_by_document.items()
                if macro in uses
            },
            "dependencies": sorted(dependencies[macro]),
        }
        for macro in sorted(prose)
    ]
    manifest: dict[str, Any] = {
        "schema_version": VALUE_MANIFEST_SCHEMA,
        "artifact_type": "cross_document_result_value_manifest",
        "fixture_mode": fixture_mode,
        "generated_results_sha256": generated_results_sha256,
        "report_manifest_sha256": file_sha256(report_manifest),
        "statistics_manifest_sha256": file_sha256(statistics_manifest),
        "mixed_model_manifest_sha256": file_sha256(mixed_model_manifest),
        "final_progress_snapshot_sha256": file_sha256(progress_manifest),
        "evaluator_unavailability_manifest_sha256": file_sha256(
            evaluator_unavailability_manifest
        ),
        "values": value_rows,
        "values_sha256": canonical_json_sha256(value_rows),
        "prose_macros": prose_rows,
        "prose_macros_sha256": canonical_json_sha256(prose_rows),
        "document_macro_usage": direct_by_document,
        "primary_recovery_values_shared_by_all_documents": True,
        "numeric_result_entry_policy": (
            "machine-derived macros only; no document-specific numeric entry"
        ),
        "smoke_or_pilot_values_used": False,
        "human_participants_recruited": False,
        "human_outcomes_estimated": False,
        "public_doi_released": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _verify_pdf(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size < 5
        or path.read_bytes()[:5] != b"%PDF-"
    ):
        raise ManuscriptRevisionError(
            "LaTeX compilation did not produce a valid PDF: {}".format(path)
        )
    return _file_identity(path, role="compiled_pdf:" + path.stem)


def compile_manuscripts(staging: Path) -> list[dict[str, Any]]:
    """Compile all three documents and return hash identities for their PDFs."""

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise ManuscriptRevisionError(
            "latexmk is required for the final three-document PDF gate"
        )
    for name in TEMPLATE_NAMES:
        completed = subprocess.run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                name,
            ],
            cwd=str(staging),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=900,
        )
        if completed.returncode != 0:
            tail = completed.stdout[-8000:]
            raise ManuscriptRevisionError(
                "LaTeX compilation failed for {}:\n{}".format(name, tail)
            )
    records = []
    for name in TEMPLATE_NAMES:
        pdf = staging / (Path(name).stem + ".pdf")
        _verify_pdf(pdf)
        records.append(
            _file_identity(
                pdf,
                role="compiled_pdf:" + Path(name).stem,
                relative_to=staging,
            )
        )
    return records


def _copy_verified(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ManuscriptRevisionError("unsafe copy source: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    if file_sha256(destination) != file_sha256(source):
        raise ManuscriptRevisionError("copied file hash mismatch: {}".format(destination))


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def verify_manuscript_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    path = output_dir / "manuscript_revision_manifest.json"
    manifest = _read_json(path, "manuscript revision manifest")
    _verify_self_hash(manifest, "manifest_sha256", "manuscript revision manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != "rankcloak_revision_manuscript_package"
        or manifest.get("originals_preserved") is not True
        or manifest.get("main_word_limit_satisfied") is not True
        or manifest.get("main_display_limit_satisfied") is not True
        or manifest.get("no_fabricated_results") is not True
        or manifest.get("all_computational_placeholders_resolved") is not True
        or manifest.get("cross_document_values_consistent") is not True
        or manifest.get("pdf_compilation_passed") is not True
        or manifest.get("smoke_or_pilot_values_used") is not False
        or manifest.get("human_recruitment_performed") is not False
        or manifest.get("human_outcomes_estimated") is not False
        or manifest.get("public_release_performed") is not False
    ):
        raise ManuscriptRevisionError("manuscript revision manifest contract mismatch")
    outputs = manifest.get("outputs")
    if (
        not isinstance(outputs, list)
        or {row.get("role") for row in outputs}
        != {"main2", "supplementary2", "response_letter"}
    ):
        raise ManuscriptRevisionError("manuscript output identity set is incomplete")
    _verify_declarations(path, outputs)
    products = manifest.get("product_files")
    compiled = manifest.get("compiled_pdfs")
    protected = manifest.get("protected_submitted_artifacts_after")
    provenance = manifest.get("provenance_manifests")
    if (
        not isinstance(products, list)
        or not isinstance(compiled, list)
        or len(compiled) != 3
        or not isinstance(protected, list)
        or len(protected) != 4
        or not isinstance(provenance, list)
        or len(provenance) != 6
        or manifest.get("provenance_manifests_sha256")
        != canonical_json_sha256(provenance)
    ):
        raise ManuscriptRevisionError(
            "manuscript package lacks generated products, PDFs, protected pins, "
            "or sealed provenance"
        )
    _verify_declarations(path, products)
    _verify_declarations(path, compiled)
    _verify_declarations(path, protected)
    _verify_declarations(path, provenance)
    provenance_by_role = {
        str(row.get("role")): row for row in provenance
    }
    expected_provenance_roles = {
        "report_manifest",
        "report_integrity",
        "statistics_manifest",
        "mixed_model_manifest",
        "final_progress_snapshot",
        "evaluator_unavailability",
    }
    if set(provenance_by_role) != expected_provenance_roles:
        raise ManuscriptRevisionError(
            "bundled manuscript provenance role set is incomplete"
        )
    declared_hash_fields = {
        "report_manifest": "report_manifest_sha256",
        "report_integrity": "report_integrity_sha256",
        "statistics_manifest": "statistics_manifest_sha256",
        "mixed_model_manifest": "mixed_model_manifest_sha256",
        "final_progress_snapshot": "final_progress_snapshot_sha256",
        "evaluator_unavailability": (
            "evaluator_unavailability_manifest_sha256"
        ),
    }
    for role, field in declared_hash_fields.items():
        if provenance_by_role[role].get("sha256") != manifest.get(field):
            raise ManuscriptRevisionError(
                "bundled {} does not match {}".format(role, field)
            )
    bundled_progress = _resolve_declared(
        path, provenance_by_role["final_progress_snapshot"]["path"]
    )
    bundled_unavailability = _resolve_declared(
        path, provenance_by_role["evaluator_unavailability"]["path"]
    )
    _verify_progress(
        bundled_progress, fixture_mode=bool(manifest.get("fixture_mode"))
    )
    _verify_unavailability(bundled_unavailability)
    if not manifest.get("fixture_mode"):
        protected_by_name = {
            str(row.get("role", "")).split(":", 1)[-1]: row
            for row in protected
        }
        for name, expected_sha256 in PROTECTED_SUBMITTED_ARTIFACTS.items():
            if protected_by_name.get(name, {}).get("sha256") != expected_sha256:
                raise ManuscriptRevisionError(
                    "production protected-artifact pin is invalid: {}".format(name)
                )
    for record in compiled:
        _verify_pdf(_resolve_declared(path, record["path"]))
    generated_results = output_dir / GENERATED_RESULTS_NAME
    value_manifest_path = output_dir / VALUE_MANIFEST_NAME
    value_manifest = _read_json(
        value_manifest_path, "cross-document value manifest"
    )
    _verify_self_hash(
        value_manifest, "manifest_sha256", "cross-document value manifest"
    )
    if (
        value_manifest.get("schema_version") != VALUE_MANIFEST_SCHEMA
        or value_manifest.get("generated_results_sha256")
        != file_sha256(generated_results)
        or value_manifest.get("final_progress_snapshot_sha256")
        != manifest.get("final_progress_snapshot_sha256")
        or value_manifest.get("evaluator_unavailability_manifest_sha256")
        != manifest.get("evaluator_unavailability_manifest_sha256")
        or value_manifest.get("primary_recovery_values_shared_by_all_documents")
        is not True
        or value_manifest.get("smoke_or_pilot_values_used") is not False
        or value_manifest.get("human_participants_recruited") is not False
        or value_manifest.get("human_outcomes_estimated") is not False
        or value_manifest.get("public_doi_released") is not False
    ):
        raise ManuscriptRevisionError(
            "cross-document value manifest violates the final reporting contract"
        )
    for record in outputs:
        source = _resolve_declared(path, record["path"]).read_text(encoding="utf-8")
        if any(marker in source for marker in PLACEHOLDER_MARKERS):
            raise ManuscriptRevisionError("revised source contains a result placeholder")
        if source.count(r"\input{" + GENERATED_RESULTS_NAME + "}") != 1:
            raise ManuscriptRevisionError(
                "revised source is not bound once to generated_results.tex"
            )
    return manifest


def build_manuscript_package(
    *,
    report_manifest: Path,
    figures_dir: Path,
    statistics_manifest: Path,
    mixed_model_manifest: Path,
    progress_manifest: Path,
    evaluator_unavailability_manifest: Path,
    manuscript_root: Path,
    output_dir: Path,
    fixture_mode: bool = False,
    preflight_only: bool = False,
    compile_documents: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if not compile_documents and not fixture_mode:
        raise ManuscriptRevisionError(
            "PDF compilation may be skipped only for non-scientific fixtures"
        )

    report_manifest = Path(report_manifest).resolve()
    figures_dir = Path(figures_dir).resolve()
    statistics_manifest = Path(statistics_manifest).resolve()
    mixed_model_manifest = Path(mixed_model_manifest).resolve()
    progress_manifest = Path(progress_manifest).resolve()
    evaluator_unavailability_manifest = Path(
        evaluator_unavailability_manifest
    ).resolve()
    manuscript_root = Path(manuscript_root).resolve()

    protected_before = _protected_snapshot(
        manuscript_root, fixture_mode=fixture_mode
    )
    report, integrity = _verify_report(
        report_manifest, fixture_mode=fixture_mode
    )
    _verify_statistics(statistics_manifest)
    _verify_mixed_models(mixed_model_manifest)
    _verify_report_source_seal(
        report_manifest,
        statistics_manifest=statistics_manifest,
        mixed_model_manifest=mixed_model_manifest,
        fixture_mode=fixture_mode,
    )
    if not fixture_mode:
        _verify_no_exploratory_report_rows(report_manifest.parent)
    progress = _verify_progress(progress_manifest, fixture_mode=fixture_mode)
    unavailability = _verify_unavailability(evaluator_unavailability_manifest)
    summary = primary_recovery_summary(
        report_manifest.parent, fixture_mode=fixture_mode
    )
    gpu_hours = float(progress["gpu"]["cumulative_actual_gpu_hours"])

    templates = []
    for name in TEMPLATE_NAMES:
        templates.append(
            _file_identity(manuscript_root / name, role="source_template:" + name)
        )
    support_sources = []
    for name in SUPPORT_NAMES:
        support_sources.append(
            _file_identity(manuscript_root / name, role="latex_support:" + name)
        )

    registry_path = report_manifest.parent / "plots" / "plot_registry.csv"
    registry_rows = _read_csv(registry_path)
    plot_ids = [str(row.get("plot_id", "")) for row in registry_rows]
    if len(plot_ids) != 18 or len(set(plot_ids)) != 18 or any(not value for value in plot_ids):
        raise ManuscriptRevisionError("plot registry is not the exact 5+13 registry")
    figure_sources = []
    for plot_id in sorted(plot_ids):
        figure_sources.append(
            _file_identity(
                figures_dir / (plot_id + ".pdf"),
                role="rendered_figure:" + plot_id,
            )
        )

    macro_bytes, prose, values = generated_result_macros(
        summary=summary,
        gpu_hours=gpu_hours,
        unavailable=int(unavailability["upstream_dependent_unavailable_units"]),
    )
    sources = finalize_sources(manuscript_root=manuscript_root)
    value_manifest = cross_document_value_manifest(
        sources=sources,
        prose=prose,
        values=values,
        generated_results_sha256=hashlib.sha256(macro_bytes).hexdigest(),
        report_manifest=report_manifest,
        statistics_manifest=statistics_manifest,
        mixed_model_manifest=mixed_model_manifest,
        progress_manifest=progress_manifest,
        evaluator_unavailability_manifest=evaluator_unavailability_manifest,
        fixture_mode=fixture_mode,
    )
    _verify_protected_unchanged(
        protected_before, manuscript_root, fixture_mode=fixture_mode
    )
    if preflight_only:
        if compile_documents and shutil.which("latexmk") is None:
            raise ManuscriptRevisionError(
                "latexmk is absent; final PDF compilation would fail"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight_passed",
            "preflight_only": True,
            "fixture_mode": fixture_mode,
            "output_written": False,
            "report_manifest_sha256": file_sha256(report_manifest),
            "final_progress_snapshot_sha256": file_sha256(progress_manifest),
            "evaluator_unavailability_manifest_sha256": file_sha256(
                evaluator_unavailability_manifest
            ),
            "generated_results_sha256": hashlib.sha256(macro_bytes).hexdigest(),
            "protected_originals_verified": True,
            "figure_count": len(figure_sources),
            "table_count": len(integrity["table_status"]),
        }
    if not compile_documents:
        raise ManuscriptRevisionError(
            "a manuscript package cannot be finalized without compiling all PDFs"
        )
    if output_dir.exists():
        existing = verify_manuscript_manifest(output_dir)
        if (
            existing.get("fixture_mode") is not fixture_mode
            or existing.get("report_manifest_sha256")
            != file_sha256(report_manifest)
            or existing.get("statistics_manifest_sha256")
            != file_sha256(statistics_manifest)
            or existing.get("mixed_model_manifest_sha256")
            != file_sha256(mixed_model_manifest)
            or existing.get("final_progress_snapshot_sha256")
            != file_sha256(progress_manifest)
            or existing.get("evaluator_unavailability_manifest_sha256")
            != file_sha256(evaluator_unavailability_manifest)
        ):
            raise ManuscriptRevisionError(
                "existing manuscript package belongs to different sealed inputs"
            )
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}.staging-".format(output_dir.name),
            dir=str(output_dir.parent),
        )
    )
    committed = False
    try:
        _write_new(staging / "main2.tex", sources["main2"].encode("utf-8"))
        _write_new(
            staging / "supplementary2.tex",
            sources["supplementary2"].encode("utf-8"),
        )
        _write_new(
            staging / "response_letter2.tex",
            sources["response_letter"].encode("utf-8"),
        )
        _write_new(staging / GENERATED_RESULTS_NAME, macro_bytes)
        _write_new(
            staging / VALUE_MANIFEST_NAME,
            (
                json.dumps(
                    value_manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        for name in SUPPORT_NAMES:
            _copy_verified(manuscript_root / name, staging / name)
        provenance_copies = {
            "statistics_manifest": (
                statistics_manifest,
                staging / "provenance" / "statistics_manifest.json",
            ),
            "mixed_model_manifest": (
                mixed_model_manifest,
                staging / "provenance" / "mixed_model_manifest.json",
            ),
            "final_progress_snapshot": (
                progress_manifest,
                staging / "provenance" / "final_progress_snapshot_v1.json",
            ),
            "evaluator_unavailability": (
                evaluator_unavailability_manifest,
                staging
                / "provenance"
                / "evaluator_unavailability_manifest.json",
            ),
        }
        for source_path, copied_path in provenance_copies.values():
            _copy_verified(source_path, copied_path)
        copied_progress = provenance_copies["final_progress_snapshot"][1]
        copied_unavailability = provenance_copies[
            "evaluator_unavailability"
        ][1]
        _verify_progress(copied_progress, fixture_mode=fixture_mode)
        _verify_unavailability(copied_unavailability)
        for name in (
            "report_output_manifest.json",
            "report_integrity.json",
            "report_source_manifest.json",
            "display_registry.json",
        ):
            _copy_verified(
                report_manifest.parent / name,
                staging / "report" / name,
            )
        _copy_verified(
            report_manifest.parent / "plots" / "plot_registry.csv",
            staging / "report" / "plots" / "plot_registry.csv",
        )
        for suffix in ("*.csv", "*.tex"):
            for table in sorted((report_manifest.parent / "tables").glob(suffix)):
                _copy_verified(
                    table, staging / "report" / "tables" / table.name
                )
        if len(list((staging / "report" / "tables").glob("*.tex"))) != 15:
            raise ManuscriptRevisionError("manuscript package lacks the 2+13 tables")
        if len(list((staging / "report" / "tables").glob("*.csv"))) != 15:
            raise ManuscriptRevisionError(
                "manuscript package lacks the 2+13 table value sources"
            )
        for plot_id in sorted(plot_ids):
            _copy_verified(
                figures_dir / (plot_id + ".pdf"),
                staging / "figures" / (plot_id + ".pdf"),
            )

        audit_sources = materialized_audit_sources(
            sources=sources,
            prose=prose,
            values=values,
            table_root=staging / "report" / "tables",
        )
        _write_new(
            staging / "audits" / "main2_expanded_for_count.tex",
            audit_sources["main2"].encode("utf-8"),
        )
        _write_new(
            staging / "audits" / "supplementary2_expanded_for_count.tex",
            audit_sources["supplementary2"].encode("utf-8"),
        )
        count_report = build_report(
            staging / "audits" / "main2_expanded_for_count.tex",
            staging / "audits" / "supplementary2_expanded_for_count.tex",
        )
        counts = asdict(count_report)
        word_ok = (
            count_report.journal_main_text_words <= MAIN_WORD_LIMIT
            and count_report.abstract_words <= ABSTRACT_WORD_LIMIT
            and count_report.title_words <= TITLE_WORD_LIMIT
            and all(
                legend.words <= FIGURE_LEGEND_WORD_LIMIT
                for legend in count_report.figure_legends
            )
        )
        display_ok = count_report.main_display_items == MAIN_DISPLAY_LIMIT
        if not word_ok or not display_ok:
            raise ManuscriptRevisionError(
                "generated manuscript violates word/display limits: {}".format(counts)
            )
        compiled_pdfs = compile_manuscripts(staging)
        protected_after = _verify_protected_unchanged(
            protected_before, manuscript_root, fixture_mode=fixture_mode
        )

        outputs = [
            _file_identity(staging / "main2.tex", role="main2", relative_to=staging),
            _file_identity(
                staging / "supplementary2.tex",
                role="supplementary2",
                relative_to=staging,
            ),
            _file_identity(
                staging / "response_letter2.tex",
                role="response_letter",
                relative_to=staging,
            ),
        ]
        support_paths = sorted(
            [
                *(staging / name for name in SUPPORT_NAMES),
                *(staging / "report" / "tables").glob("*"),
                staging / "report" / "report_output_manifest.json",
                staging / "report" / "report_integrity.json",
                staging / "report" / "report_source_manifest.json",
                staging / "report" / "display_registry.json",
                staging / "report" / "plots" / "plot_registry.csv",
                *(staging / "figures").glob("*.pdf"),
            ]
        )
        support = [
            _file_identity(path, role="support", relative_to=staging)
            for path in support_paths
        ]
        product_paths = [
            staging / GENERATED_RESULTS_NAME,
            staging / VALUE_MANIFEST_NAME,
            staging / "audits" / "main2_expanded_for_count.tex",
            staging / "audits" / "supplementary2_expanded_for_count.tex",
        ]
        products = [
            _file_identity(path, role="generated_product", relative_to=staging)
            for path in product_paths
        ]
        provenance = [
            _file_identity(
                staging / "report" / "report_output_manifest.json",
                role="report_manifest",
                relative_to=staging,
            ),
            _file_identity(
                staging / "report" / "report_integrity.json",
                role="report_integrity",
                relative_to=staging,
            ),
            _file_identity(
                provenance_copies["statistics_manifest"][1],
                role="statistics_manifest",
                relative_to=staging,
            ),
            _file_identity(
                provenance_copies["mixed_model_manifest"][1],
                role="mixed_model_manifest",
                relative_to=staging,
            ),
            _file_identity(
                copied_progress,
                role="final_progress_snapshot",
                relative_to=staging,
            ),
            _file_identity(
                copied_unavailability,
                role="evaluator_unavailability",
                relative_to=staging,
            ),
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "rankcloak_revision_manuscript_package",
            "fixture_mode": fixture_mode,
            "non_scientific_fixture": fixture_mode,
            "originals_preserved": True,
            "protected_submitted_artifacts_before": protected_before,
            "protected_submitted_artifacts_after": protected_after,
            "source_templates": templates,
            "source_templates_sha256": canonical_json_sha256(templates),
            "provenance_manifests": provenance,
            "provenance_manifests_sha256": canonical_json_sha256(provenance),
            "report_integrity_sha256": file_sha256(
                report_manifest.parent / "report_integrity.json"
            ),
            "report_manifest_sha256": file_sha256(report_manifest),
            "statistics_manifest_sha256": file_sha256(statistics_manifest),
            "mixed_model_manifest_sha256": file_sha256(mixed_model_manifest),
            "final_progress_snapshot_sha256": file_sha256(progress_manifest),
            "evaluator_unavailability_manifest_sha256": file_sha256(
                evaluator_unavailability_manifest
            ),
            "report_display_status": {
                "table_status": integrity["table_status"],
                "plot_status": integrity["plot_status"],
            },
            "primary_recovery_summary": summary,
            "evaluator_accounting": {
                "frozen_target": FROZEN_EVALUATOR_TARGET,
                "scoreable_rows": SCOREABLE_EVALUATOR_ROWS,
                "terminal_upstream_unavailable": UPSTREAM_UNAVAILABLE_ROWS,
                "scores_imputed_or_fabricated": False,
            },
            "cumulative_actual_gpu_hours": gpu_hours,
            "journal_limit_audit": counts,
            "main_word_limit_satisfied": word_ok,
            "main_display_limit_satisfied": display_ok,
            "no_fabricated_results": True,
            "all_computational_placeholders_resolved": True,
            "cross_document_values_consistent": True,
            "pdf_compilation_passed": True,
            "smoke_or_pilot_values_used": False,
            "numeric_prose_policy": (
                "primary recovery counts derived mechanically from hash-verified "
                "main_table_1.csv; all other computational numbers remain in "
                "machine-generated tables and figures"
            ),
            "human_recruitment_performed": False,
            "human_outcomes_estimated": False,
            "public_release_performed": False,
            "outputs": outputs,
            "product_files": products,
            "product_files_sha256": canonical_json_sha256(products),
            "compiled_pdfs": compiled_pdfs,
            "compiled_pdfs_sha256": canonical_json_sha256(compiled_pdfs),
            "support_files": support,
            "support_files_sha256": canonical_json_sha256(support),
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write_new(
            staging / "manuscript_revision_manifest.json",
            (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        if output_dir.exists():
            raise ManuscriptRevisionError(
                "manuscript destination appeared during atomic staging"
            )
        os.replace(staging, output_dir)
        committed = True
        return verify_manuscript_manifest(output_dir)
    finally:
        if not committed and staging.exists():
            # A failed staging directory contains no authoritative manifest and
            # is intentionally retained for diagnosis; a retry uses a new name.
            pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-manifest", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--statistics-manifest", type=Path, required=True)
    parser.add_argument("--mixed-model-manifest", type=Path, required=True)
    parser.add_argument("--progress-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluator-unavailability-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=PROJECT_ROOT / ".paper" / "scientific_reports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify all sealed inputs, templates, protected originals, figures, "
            "and deterministic replacements without creating the output directory."
        ),
    )
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help=(
            "Permit an explicitly fixture-labelled report for automated tests. "
            "Fixture products are non-scientific and production reports are rejected "
            "under this flag."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = build_manuscript_package(
            report_manifest=args.report_manifest,
            figures_dir=args.figures_dir,
            statistics_manifest=args.statistics_manifest,
            mixed_model_manifest=args.mixed_model_manifest,
            progress_manifest=args.progress_manifest,
            evaluator_unavailability_manifest=(
                args.evaluator_unavailability_manifest
            ),
            manuscript_root=args.manuscript_root,
            output_dir=args.output_dir,
            fixture_mode=bool(args.fixture_mode),
            preflight_only=bool(args.preflight_only),
        )
    except (
        ManuscriptRevisionError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise SystemExit("manuscript revision failed: {}".format(exc))
    if args.preflight_only:
        payload = dict(manifest)
        payload["output_dir"] = str(args.output_dir.resolve())
    else:
        payload = {
            "status": "completed",
            "output_dir": str(args.output_dir.resolve()),
            "manifest_sha256": manifest["manifest_sha256"],
            "main_word_limit_satisfied": manifest[
                "main_word_limit_satisfied"
            ],
            "main_display_limit_satisfied": manifest[
                "main_display_limit_satisfied"
            ],
            "no_fabricated_results": manifest["no_fabricated_results"],
            "cross_document_values_consistent": manifest[
                "cross_document_values_consistent"
            ],
            "pdf_compilation_passed": manifest["pdf_compilation_passed"],
            "human_recruitment_performed": False,
            "human_outcomes_estimated": False,
            "public_release_performed": False,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
