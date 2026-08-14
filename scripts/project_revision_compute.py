#!/usr/bin/env python3
"""Project the frozen revision-v1 workload against the 150 GPU-hour gate."""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_compute import (
    DEFAULT_BUDGET_GPU_HOURS,
    discover_smoke_shards,
    project_revision_compute,
)


class ProjectionPublicationError(RuntimeError):
    """Raised when a versioned compute projection cannot be published once."""


def _atomic_write_json(path: Path, value: object) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ProjectionPublicationError("Projection parent must not be a symlink")
    if destination.exists() or destination.is_symlink():
        raise ProjectionPublicationError(
            "Projection output already exists: {}".format(destination)
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(destination), follow_symlinks=False)
        except FileExistsError as exc:
            raise ProjectionPublicationError(
                "Projection output already exists: {}".format(destination)
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise ProjectionPublicationError(
                    "Filesystem cannot publish a no-overwrite projection"
                ) from exc
            raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify three completed exploratory smoke shards and make a fail-closed "
            "point/conservative revision-v1 GPU-budget projection."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--smoke-root",
        type=Path,
        help="Root containing the three completed model smoke shard directories.",
    )
    inputs.add_argument(
        "--smoke-shard",
        type=Path,
        action="append",
        help="One completed model smoke shard; pass exactly three times.",
    )
    parser.add_argument(
        "--auxiliary-timing",
        type=Path,
        action="append",
        default=[],
        help="Optional self-hashed evaluator/detector exploratory timing JSON.",
    )
    parser.add_argument(
        "--legacy-incurred-ledger",
        type=Path,
        required=True,
        help="Verified six-entry charge-only legacy ledger.",
    )
    parser.add_argument(
        "--invalidation-manifest",
        type=Path,
        action="append",
        required=True,
        help="Verified stopped-shard invalidation manifest; exactly one is required.",
    )
    parser.add_argument(
        "--budget-gpu-hours",
        type=float,
        default=DEFAULT_BUDGET_GPU_HOURS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "revision_v1" / "compute_projection_v2.json",
        help="No-overwrite versioned machine-readable JSON report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    shards = (
        discover_smoke_shards(args.smoke_root)
        if args.smoke_root is not None
        else list(args.smoke_shard or [])
    )
    report = project_revision_compute(
        shards,
        auxiliary_timing_paths=args.auxiliary_timing,
        invalidation_manifest_paths=args.invalidation_manifest,
        legacy_incurred_ledger_path=args.legacy_incurred_ledger,
        budget_gpu_hours=args.budget_gpu_hours,
    )
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    try:
        _atomic_write_json(args.output, report)
    except (OSError, ProjectionPublicationError) as exc:
        print("compute projection publication error: {}".format(exc), file=sys.stderr)
        return 2
    print(rendered)
    status = report["decision"]["status"]
    if status == "go_within_budget":
        return 0
    if status == "no_go_over_budget":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
