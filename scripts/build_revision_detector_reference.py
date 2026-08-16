#!/usr/bin/env python3
"""Build a non-copying reference index for final neural-detector artifacts."""

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

from rankcloak.revision_detector_reference import (  # noqa: E402
    DetectorReferenceError,
    build_detector_reference_index,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [
        str(Path(__file__).resolve()),
        "--detector-run-manifest",
        str(args.detector_run_manifest.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_detector_reference_index(
            detector_run_manifest=args.detector_run_manifest,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=bool(args.overwrite),
        )
    except DetectorReferenceError as exc:
        raise SystemExit(f"detector reference build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "fit_count": artifacts.fit_count,
                "external_reference_count": artifacts.external_reference_count,
                "manifest_path": artifacts.manifest_path,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
