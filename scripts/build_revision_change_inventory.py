#!/usr/bin/env python3
"""Build the read-only local change/generated-file preservation inventory."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_change_inventory import (  # noqa: E402
    ChangeInventoryError,
    build_change_inventory,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preexisting-path", action="append", default=[])
    parser.add_argument("--extra-path", type=Path, action="append", default=[])
    parser.add_argument("--planned-output-path", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve()), "--output", str(args.output.resolve()), "--manifest", str(args.manifest.resolve())]
    for path in args.preexisting_path:
        values.extend(("--preexisting-path", path))
    for path in args.extra_path:
        values.extend(("--extra-path", str(path.resolve())))
    for path in args.planned_output_path:
        values.extend(("--planned-output-path", str(path.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_change_inventory(
            project_root=PROJECT_ROOT,
            output_path=args.output,
            manifest_path=args.manifest,
            preexisting_paths=args.preexisting_path,
            extra_paths=args.extra_path,
            planned_output_paths=args.planned_output_path,
            command=_command(args),
            overwrite=bool(args.overwrite),
        )
    except ChangeInventoryError as exc:
        raise SystemExit(f"change inventory build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "inventory_path": artifacts.inventory_path,
                "manifest_path": artifacts.manifest_path,
                "repository_entry_count": artifacts.repository_entry_count,
                "extra_entry_count": artifacts.extra_entry_count,
                "total_entry_count": artifacts.total_entry_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
