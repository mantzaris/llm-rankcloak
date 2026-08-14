#!/usr/bin/env python3
"""Poll, atomically update, or verify revision-v1 confirmatory progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rankcloak.revision_progress import (
    RevisionProgressError,
    build_progress_snapshot,
    update_progress_snapshot,
    verify_progress_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_v1"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "confirmatory_progress_v1.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true",
        help="Atomically replace the canonical status JSON after scanning all shards.",
    )
    mode.add_argument(
        "--check", action="store_true",
        help="Verify the existing status JSON and every bound source artifact.",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--compact", action="store_true",
        help="Print a single-line polling summary rather than the complete JSON object.",
    )
    return parser.parse_args(argv)


def _compact(value: dict) -> dict:
    if value.get("status") == "ok":
        return value
    gpu = value["gpu"]
    return {
        "status": value["status"],
        "generated_at": value["generated_at"],
        "current": value["current"],
        "counts": value["counts"],
        "recovery_counts": value["recovery_counts"],
        "cumulative_actual_gpu_hours": gpu["cumulative_actual_gpu_hours"],
        "completed_per_gpu_hour": value["throughput"]["completed_per_gpu_hour"],
        "rolling_eta_hours": value["eta"]["rolling_eta_hours"],
        "last_checkpoint": value["last_checkpoint"],
        "recovered_error_count": len(value["recovered_errors"]),
        "progress_sha256": value["progress_sha256"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            value = verify_progress_snapshot(args.output)
        elif args.write:
            value = update_progress_snapshot(
                args.results_root, output_path=args.output, retries=args.retries
            )
        else:
            value = build_progress_snapshot(args.results_root)
    except (OSError, RevisionProgressError, ValueError) as exc:
        raise SystemExit("revision progress failed: {}".format(exc))
    output = _compact(value) if args.compact else value
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
