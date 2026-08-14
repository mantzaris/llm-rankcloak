#!/usr/bin/env python3
"""Build RankCloak capacity--quality theory tables from saved trial data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_theory import (  # noqa: E402
    TheoryValidationError,
    build_theory_artifacts,
)


def _flatten(values: Optional[Sequence[Sequence[Path]]]) -> List[Path]:
    return [path for group in (values or []) for path in group]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate bounded-codec capacity rates, validate empirical "
            "same-context surprisal bounds, and audit exact-replay traces. "
            "Only saved CSV/JSONL evidence is consumed; missing endpoints are "
            "reported as unavailable and are never imputed."
        )
    )
    parser.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help="One or more saved .csv, .jsonl, or .ndjson trial files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for machine-readable validation and plot-source tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known theory outputs in the target directory.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = build_theory_artifacts(
            _flatten(args.trials), args.output_dir, overwrite=args.overwrite
        )
    except TheoryValidationError as exc:
        parser.exit(2, "revision theory build failed: {}\n".format(exc))
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "files": dict(artifacts.files),
                "summary": dict(artifacts.summary),
                "empirical_missing_value_policy": "no imputation",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

