#!/usr/bin/env python3
"""Build data-only robustness summaries from immutable revision artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_robustness import (  # noqa: E402
    RobustnessAnalysisError,
    build_robustness_artifacts,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate saved transmission/replay outcomes and emit condition, "
            "unavailability, and first-divergence source tables."
        )
    )
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--unavailable", type=Path, required=True)
    parser.add_argument(
        "--robustness-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "revision_v1" / "robustness.json",
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
    try:
        artifacts = build_robustness_artifacts(
            trials_path=args.trials,
            failures_path=args.failures,
            unavailable_path=args.unavailable,
            robustness_config=args.robustness_config,
            statistics_config=args.statistics_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except RobustnessAnalysisError as exc:
        raise SystemExit(f"revision robustness analysis failed: {exc}") from exc
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
