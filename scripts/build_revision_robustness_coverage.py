#!/usr/bin/env python3
"""Build the requested-to-tested transmission perturbation crosswalk."""

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

from rankcloak.revision_robustness_coverage import (  # noqa: E402
    RobustnessCoverageError,
    build_robustness_coverage_inventory,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/revision_v1/evidence_specs/robustness_coverage_map.json",
    )
    parser.add_argument(
        "--robustness-config",
        type=Path,
        default=PROJECT_ROOT / "configs/revision_v1/robustness.json",
    )
    parser.add_argument("--robustness-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--coverage-config", args.coverage_config),
        ("--robustness-config", args.robustness_config),
        ("--robustness-manifest", args.robustness_manifest),
        ("--output-dir", args.output_dir),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_robustness_coverage_inventory(
            coverage_config=args.coverage_config,
            robustness_config=args.robustness_config,
            robustness_manifest=args.robustness_manifest,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except RobustnessCoverageError as exc:
        raise SystemExit(f"robustness coverage build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "request_count": artifacts.request_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
