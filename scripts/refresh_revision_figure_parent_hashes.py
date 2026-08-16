#!/usr/bin/env python3
"""Refresh sealed parent identities for changed computational figures only."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_figures import (  # noqa: E402
    FigureEvidenceError,
    refresh_evidence_summary_figure_hashes,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-summary-manifest", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    return parser


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise FigureEvidenceError(
            f"Figure-parent path is outside the repository: {path.resolve()}"
        ) from exc


def _command(args: argparse.Namespace) -> str:
    values = ["scripts/refresh_revision_figure_parent_hashes.py"]
    for option, value in (
        ("--evidence-summary-manifest", args.evidence_summary_manifest),
        ("--reference-table", args.reference_table),
        ("--figure-dir", args.figure_dir),
        ("--figure-manifest", args.figure_manifest),
    ):
        values.extend((option, _relative(value)))
    return " ".join(shlex.quote(value) for value in values)


def _head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = refresh_evidence_summary_figure_hashes(
            project_root=PROJECT_ROOT,
            evidence_summary_manifest=args.evidence_summary_manifest,
            reference_table=args.reference_table,
            figure_dir=args.figure_dir,
            figure_manifest=args.figure_manifest,
            refresh_head=_head(),
            command=_command(args),
        )
    except (FigureEvidenceError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"figure-parent refresh failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "evidence_summary_manifest": artifacts.evidence_summary_manifest,
                "reference_table": artifacts.reference_table,
                "updated_reference_count": artifacts.updated_reference_count,
                "manifest_sha256": artifacts.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
