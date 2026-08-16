#!/usr/bin/env python3
"""Build hash-bound computational figures for the revision evidence package."""

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

from rankcloak.revision_figures import (  # noqa: E402
    FigureEvidenceError,
    build_core_figures,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness-manifest", type=Path, required=True)
    parser.add_argument("--theory-manifest", type=Path, required=True)
    parser.add_argument("--readability-manifest", type=Path, required=True)
    parser.add_argument("--overhead-manifest", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise FigureEvidenceError(
            f"Figure-related path is outside the repository root: {resolved}"
        ) from exc


def _reproducible_command(args: argparse.Namespace) -> str:
    values = [
        "scripts/build_revision_figures.py",
        "--robustness-manifest",
        _repository_relative(args.robustness_manifest),
        "--theory-manifest",
        _repository_relative(args.theory_manifest),
        "--readability-manifest",
        _repository_relative(args.readability_manifest),
        "--overhead-manifest",
        _repository_relative(args.overhead_manifest),
        "--detector-manifest",
        _repository_relative(args.detector_manifest),
        "--ablation-manifest",
        _repository_relative(args.ablation_manifest),
        "--output-dir",
        _repository_relative(args.output_dir),
    ]
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_core_figures(
            robustness_manifest=args.robustness_manifest,
            theory_manifest=args.theory_manifest,
            readability_manifest=args.readability_manifest,
            overhead_manifest=args.overhead_manifest,
            detector_manifest=args.detector_manifest,
            ablation_manifest=args.ablation_manifest,
            output_dir=args.output_dir,
            command=_reproducible_command(args),
            project_root=PROJECT_ROOT,
            generator_sources={
                "implementation": PROJECT_ROOT / "rankcloak" / "revision_figures.py",
                "cli": Path(__file__).resolve(),
            },
            overwrite=args.overwrite,
        )
    except FigureEvidenceError as exc:
        raise SystemExit(f"figure build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "files": artifacts.files,
                "summary": artifacts.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
