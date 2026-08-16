#!/usr/bin/env python3
"""Build compact phase-timing and same-CUDA detector reproducibility evidence."""

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

from rankcloak.revision_detector_benchmarks import (  # noqa: E402
    DetectorBenchmarkError,
    build_detector_benchmark_evidence,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument(
        "--reproducibility-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for path in args.benchmark:
        values.extend(("--benchmark", str(path.resolve())))
    for path in args.reproducibility_report:
        values.extend(("--reproducibility-report", str(path.resolve())))
    values.extend(("--output-dir", str(args.output_dir.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_detector_benchmark_evidence(
            benchmark_paths=args.benchmark,
            reproducibility_report_paths=args.reproducibility_report,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except DetectorBenchmarkError as exc:
        raise SystemExit(f"detector benchmark evidence failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "architecture_count": artifacts.architecture_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
