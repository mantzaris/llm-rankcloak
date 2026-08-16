#!/usr/bin/env python3
"""Build supplementary payload-grouped diagnostics for frozen detector outputs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_detector_analysis import (  # noqa: E402
    DetectorAnalysisError,
    analyze_detector_outputs,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "analysis"
    / "revision_v1"
    / "evidence_specs"
    / "detector_supplementary_metrics.json"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-run-manifest", type=Path, required=True)
    parser.add_argument(
        "--analysis-config", type=Path, default=DEFAULT_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [
        str(Path(__file__).resolve()),
        "--detector-run-manifest",
        str(args.detector_run_manifest.resolve()),
        "--analysis-config",
        str(args.analysis_config.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = analyze_detector_outputs(
            detector_run_manifest=args.detector_run_manifest,
            analysis_config=args.analysis_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except DetectorAnalysisError as exc:
        raise SystemExit(f"detector analysis failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "files": artifacts.files,
                "summary": artifacts.summary,
                "reproducible_command": _command(args),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
