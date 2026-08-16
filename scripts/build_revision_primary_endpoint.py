#!/usr/bin/env python3
"""Build hash-bound primary trial and strict repeated-payload recovery endpoints."""

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

from rankcloak.revision_primary_endpoint import (  # noqa: E402
    PrimaryEndpointError,
    build_primary_endpoint,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-trials", type=Path, required=True)
    parser.add_argument(
        "--confirmatory-plan",
        type=Path,
        default=PROJECT_ROOT / "analysis/revision_v1/confirmatory_model_plan.json",
    )
    parser.add_argument(
        "--endpoint-config",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/revision_v1/evidence_specs/primary_recovery_endpoint.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--primary-trials", args.primary_trials),
        ("--confirmatory-plan", args.confirmatory_plan),
        ("--endpoint-config", args.endpoint_config),
        ("--output-dir", args.output_dir),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_primary_endpoint(
            primary_trials=args.primary_trials,
            confirmatory_plan=args.confirmatory_plan,
            endpoint_config=args.endpoint_config,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except PrimaryEndpointError as exc:
        raise SystemExit(f"primary endpoint build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "endpoint_row_count": artifacts.endpoint_row_count,
                "payload_row_count": artifacts.payload_row_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
