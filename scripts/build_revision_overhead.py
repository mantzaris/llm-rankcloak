#!/usr/bin/env python3
"""Build data-only computational-overhead evidence from saved runtime rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_overhead import (  # noqa: E402
    OverheadAnalysisError,
    build_overhead_artifacts,
)


def _stage_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise OverheadAnalysisError(f"{label} must use STAGE=PATH: {value}")
        stage, raw_path = value.split("=", 1)
        if not stage or stage in result or not raw_path:
            raise OverheadAnalysisError(f"Invalid or duplicate {label}: {value}")
        result[stage] = Path(raw_path)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", action="append", required=True, metavar="STAGE=PATH")
    parser.add_argument("--runtime", action="append", required=True, metavar="STAGE=PATH")
    parser.add_argument(
        "--statistics-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "revision_v1" / "statistics.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_overhead_artifacts(
            trial_paths=_stage_paths(args.trials, label="--trials"),
            runtime_paths=_stage_paths(args.runtime, label="--runtime"),
            statistics_config=args.statistics_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except OverheadAnalysisError as exc:
        raise SystemExit(f"revision overhead analysis failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "files": artifacts.files,
                "summary": artifacts.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
