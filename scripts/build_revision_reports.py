#!/usr/bin/env python3
"""Build hash-verified Scientific Reports tables and plot sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_reporting import (  # noqa: E402
    ReportArtifactConflict,
    RevisionReportingError,
    build_revision_reports,
    verify_report_output_manifest,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate revision tables and figure-source artifacts exclusively "
            "from manifest-addressed machine outputs."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statistics-manifest", type=Path)
    parser.add_argument("--theory-manifest", type=Path)
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument(
        "--mixed-model-manifest",
        type=Path,
        help=(
            "Locked R mixed_model_run_manifest.json. This is the only accepted "
            "source of primary inferential effects."
        ),
    )
    parser.add_argument(
        "--evaluator-unavailability-manifest",
        type=Path,
        help=(
            "Hash-addressed accounting for the 48 upstream-dependent evaluator "
            "non-outcomes; verified and excluded from quality estimands."
        ),
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional metadata-only runner runtime manifest or generic hashed "
            "runtime-output manifest; may be repeated for model shards."
        ),
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional preprocessing_output_manifest.json; may be repeated to "
            "display explicit condition-unavailable counts."
        ),
    )
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help=(
            "Permit explicitly labelled deterministic test fixtures. These "
            "outputs are not scientific evidence and must not be cited."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.statistics_manifest,
            args.theory_manifest,
            args.detector_manifest,
            args.mixed_model_manifest,
            args.evaluator_unavailability_manifest,
            args.runtime_manifest,
            args.preprocessing_manifest,
        )
    ):
        parser.error("provide at least one machine-output manifest")
    try:
        build = build_revision_reports(
            output_dir=args.output_dir,
            statistics_manifest=args.statistics_manifest,
            theory_manifest=args.theory_manifest,
            detector_manifest=args.detector_manifest,
            mixed_model_manifest=args.mixed_model_manifest,
            evaluator_unavailability_manifest=(
                args.evaluator_unavailability_manifest
            ),
            runtime_manifests=args.runtime_manifest,
            preprocessing_manifests=args.preprocessing_manifest,
            fixture_mode=bool(args.fixture_mode),
        )
        verification = verify_report_output_manifest(build.output_dir)
        if verification["status"] != "ok":
            raise RevisionReportingError(
                "Generated report failed its output-manifest verification: {}".format(
                    "; ".join(verification["errors"])
                )
            )
    except (RevisionReportingError, ReportArtifactConflict) as exc:
        parser.exit(2, "revision reporting failed: {}\n".format(exc))
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(build.output_dir),
                "output_file_count": verification["verified_file_count"],
                "main_display_count": build.integrity_report["main_display_count"],
                "main_display_limit": build.integrity_report["main_display_limit"],
                "fixture_mode": build.integrity_report["fixture_mode"],
                "missing_results_are_explicit": True,
                "numeric_override_interface": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
