#!/usr/bin/env python3
"""Build exploratory final-detector metrics excluding near-duplicate test groups."""

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

from rankcloak.revision_detector_leakage import (  # noqa: E402
    DetectorLeakageAuditError,
    build_detector_leakage_sensitivity,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "analysis"
    / "revision_v1"
    / "evidence_specs"
    / "detector_leakage_sensitivity.json"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-run-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--detector-run-manifest", args.detector_run_manifest),
        ("--leakage-audit-manifest", args.leakage_audit_manifest),
        ("--sensitivity-config", args.sensitivity_config),
        ("--output-dir", args.output_dir),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_detector_leakage_sensitivity(
            detector_run_manifest=args.detector_run_manifest,
            leakage_audit_manifest=args.leakage_audit_manifest,
            sensitivity_config=args.sensitivity_config,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except DetectorLeakageAuditError as exc:
        raise SystemExit(f"detector leakage sensitivity failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "fit_count": artifacts.fit_count,
                "metric_row_count": artifacts.metric_row_count,
                "affected_fit_count": artifacts.affected_fit_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
