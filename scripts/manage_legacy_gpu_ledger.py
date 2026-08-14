#!/usr/bin/env python3
"""Create or verify the no-overwrite legacy incurred-GPU charge ledger."""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_compute import (
    EXPECTED_MODELS,
    RevisionComputeError,
    build_legacy_incurred_charge_ledger,
    verify_legacy_gpu_ledger,
)


class LedgerPublicationError(RuntimeError):
    """Raised when an immutable ledger cannot be published safely."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def publish_no_overwrite(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish JSON with link-based no-replace semantics."""

    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise LedgerPublicationError("Ledger parent must not be a symlink")
    if destination.exists() or destination.is_symlink():
        raise LedgerPublicationError("Ledger already exists: {}".format(destination))
    content = json.dumps(
        dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(destination), follow_symlinks=False)
        except FileExistsError as exc:
            raise LedgerPublicationError(
                "Ledger already exists: {}".format(destination)
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise LedgerPublicationError(
                    "Filesystem cannot atomically publish a no-overwrite ledger"
                ) from exc
            raise
        try:
            parent_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            parent_descriptor = None
        if parent_descriptor is not None:
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _model_directories(root: Path) -> list[Path]:
    return [Path(root) / model_id for model_id in EXPECTED_MODELS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Account for completed legacy smoke/evaluator GPU time without "
            "using the superseded artifacts as scientific or rate evidence."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Verify six stopped legacy sources and publish one immutable ledger."
    )
    create.add_argument("--smoke-root", type=Path, required=True)
    create.add_argument("--evaluator-root", type=Path, required=True)
    create.add_argument("--ledger", type=Path, required=True)
    create.add_argument(
        "--created-at",
        required=True,
        help="Explicit timezone-aware ISO-8601 creation time for reproducible provenance.",
    )
    verify = commands.add_parser(
        "verify", help="Re-hash the ledger and all six source trees; fail on any drift."
    )
    verify.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            sources = _model_directories(args.smoke_root) + _model_directories(
                args.evaluator_root
            )
            destination = args.ledger.absolute()
            destination_resolved = destination.resolve(strict=False)
            for source in sources:
                source_resolved = source.resolve(strict=True)
                if _is_within(destination_resolved, source_resolved):
                    raise LedgerPublicationError(
                        "Ledger must be external to every charged source directory"
                    )
            value = build_legacy_incurred_charge_ledger(
                _model_directories(args.smoke_root),
                _model_directories(args.evaluator_root),
                created_at=args.created_at,
            )
            publish_no_overwrite(destination, value)
            report = verify_legacy_gpu_ledger(destination)
        else:
            report = verify_legacy_gpu_ledger(args.ledger)
    except (OSError, RevisionComputeError, LedgerPublicationError) as exc:
        print("legacy GPU ledger error: {}".format(exc), file=sys.stderr)
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
