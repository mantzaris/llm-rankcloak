#!/usr/bin/env python3
"""Build the frozen detector split and near-duplicate leakage audit."""

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

from rankcloak.revision_detector_leakage import (  # noqa: E402
    DetectorLeakageAuditError,
    build_detector_leakage_audit,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-corpus", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--detector-corpus", args.detector_corpus),
        ("--detector-config", args.detector_config),
        ("--execution-plan", args.execution_plan),
        ("--output-dir", args.output_dir),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_detector_leakage_audit(
            detector_corpus=args.detector_corpus,
            detector_config=args.detector_config,
            execution_plan=args.execution_plan,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except DetectorLeakageAuditError as exc:
        raise SystemExit(f"detector leakage audit failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "near_duplicate_pair_count": artifacts.near_duplicate_pair_count,
                "affected_split_count": artifacts.affected_split_count,
                "affected_test_payload_group_count_across_splits": (
                    artifacts.affected_test_payload_group_count_across_splits
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
