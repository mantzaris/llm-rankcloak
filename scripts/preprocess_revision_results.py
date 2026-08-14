#!/usr/bin/env python3
"""Validate immutable revision runner shards and build flat analysis inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rankcloak.revision_preprocess import preprocess_revision_results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Leaf runner directory containing records.jsonl and frozen manifests; repeatable.",
    )
    parser.add_argument(
        "--reference-run-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Validated source shard used only for canonical/robustness joins; "
            "its rows are not emitted unless also supplied as --run-dir."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit planned work without a durable completion. This is for diagnostic "
            "inspection only and must not be used for final confirmatory tables."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    artifacts = preprocess_revision_results(
        run_dirs=args.run_dir,
        reference_run_dirs=args.reference_run_dir,
        output_dir=args.output_dir,
        strict_complete=not args.allow_incomplete,
    )
    print(
        json.dumps(
            {
                "output_dir": artifacts.output_dir,
                "files": artifacts.files,
                "row_counts": artifacts.row_counts,
                "evidence_statuses": list(artifacts.evidence_statuses),
                "strict_complete": not args.allow_incomplete,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
