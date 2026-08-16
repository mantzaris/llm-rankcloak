#!/usr/bin/env python3
"""Build participant-free readability and blinded-stimulus evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_readability import (  # noqa: E402
    ReadabilityEvidenceError,
    build_readability_artifacts,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="Ordered frozen primary records.jsonl files, one per model.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=PROJECT_ROOT / "human_study" / "config" / "design.json",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROJECT_ROOT / "configs" / "revision_v1" / "prompts.json",
    )
    parser.add_argument(
        "--human-control-audit",
        type=Path,
        default=(
            PROJECT_ROOT / "human_study" / "controls" / "dolly_coverage_audit.json"
        ),
    )
    parser.add_argument(
        "--rating-instrument",
        type=Path,
        default=PROJECT_ROOT / "human_study" / "survey" / "instrument.json",
    )
    parser.add_argument(
        "--power-planning",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "human_study" / "power" / "planning_power_design_grid.csv",
            PROJECT_ROOT / "human_study" / "power" / "PLANNING_RESULTS.md",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_readability_artifacts(
            records_paths=args.records,
            design_path=args.design,
            prompt_config_path=args.prompts,
            control_audit_path=args.human_control_audit,
            instrument_path=args.rating_instrument,
            power_paths=args.power_planning,
            output_dir=args.output_dir,
            confidence_level=args.confidence_level,
            bootstrap_resamples=args.bootstrap_resamples,
            overwrite=args.overwrite,
        )
    except ReadabilityEvidenceError as exc:
        raise SystemExit(f"revision readability build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "computational_complete_human_evaluation_uncollected",
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
