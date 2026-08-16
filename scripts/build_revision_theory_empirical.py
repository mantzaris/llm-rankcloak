#!/usr/bin/env python3
"""Build empirical residual and uncertainty artifacts for RankCloak theory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_theory_empirical import (  # noqa: E402
    EmpiricalTheoryError,
    build_empirical_theory_artifacts,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="One or more unique saved .csv, .jsonl, or .ndjson trial files.",
    )
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
    inputs = [path for group in args.trials for path in group]
    try:
        artifacts = build_empirical_theory_artifacts(
            input_paths=inputs,
            statistics_config=args.statistics_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except EmpiricalTheoryError as exc:
        raise SystemExit(f"empirical theory build failed: {exc}") from exc
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
