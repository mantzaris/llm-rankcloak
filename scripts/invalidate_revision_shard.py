#!/usr/bin/env python3
"""Create or verify an external immutable revision-shard invalidation entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_invalidation import (
    RevisionInvalidationError,
    create_invalidation_entry,
    verify_invalidation_entry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a stopped revision shard in an external, self-hashed "
            "invalidation registry without editing or moving the shard."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser(
        "create",
        help="Hash a stopped shard in place and atomically create a registry entry.",
    )
    create.add_argument("--shard", type=Path, required=True)
    create.add_argument("--registry-entry", type=Path, required=True)
    create.add_argument("--reason-code", required=True)
    create.add_argument("--reason", required=True)
    create.add_argument("--superseding-target-namespace", required=True)
    create.add_argument(
        "--superseding-stage",
        action="append",
        required=True,
        help="A replacement stage identifier; repeat for every superseding stage.",
    )
    create.add_argument(
        "--confirm-stopped",
        action="store_true",
        help=(
            "Required caller attestation that every shard writer has stopped. "
            "The utility does not stop or discover processes."
        ),
    )

    verify = commands.add_parser(
        "verify",
        help="Re-hash the in-place shard and fail if it or the entry changed.",
    )
    verify.add_argument("--registry-entry", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            report = create_invalidation_entry(
                args.shard,
                args.registry_entry,
                reason_code=args.reason_code,
                reason=args.reason,
                superseding_target_namespace=args.superseding_target_namespace,
                superseding_stages=args.superseding_stage,
                confirm_stopped=args.confirm_stopped,
            )
        else:
            report = verify_invalidation_entry(args.registry_entry)
    except RevisionInvalidationError as exc:
        print("invalidation error: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
