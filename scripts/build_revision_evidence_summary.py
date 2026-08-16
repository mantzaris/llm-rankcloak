#!/usr/bin/env python3
"""Build evidence-only revision status, maps, source tables, and GPU budget."""

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

from rankcloak.revision_evidence_summary import (  # noqa: E402
    EvidenceSummaryError,
    build_evidence_summary,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--evidence-spec", type=Path, required=True)
    parser.add_argument("--progress-ledger", type=Path, required=True)
    parser.add_argument("--payload-corpus", type=Path, required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--prompts-config", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--detector-run-manifest", type=Path, required=True)
    parser.add_argument("--human-evaluation-status", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--package-root", args.package_root),
        ("--evidence-spec", args.evidence_spec),
        ("--progress-ledger", args.progress_ledger),
        ("--payload-corpus", args.payload_corpus),
        ("--payload-manifest", args.payload_manifest),
        ("--prompts-config", args.prompts_config),
        ("--models-config", args.models_config),
        ("--detector-run-manifest", args.detector_run_manifest),
        ("--human-evaluation-status", args.human_evaluation_status),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
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
        artifacts = build_evidence_summary(
            project_root=PROJECT_ROOT,
            package_root=args.package_root,
            evidence_spec=args.evidence_spec,
            progress_ledger=args.progress_ledger,
            payload_corpus=args.payload_corpus,
            payload_manifest=args.payload_manifest,
            prompts_config=args.prompts_config,
            models_config=args.models_config,
            detector_run_manifest=args.detector_run_manifest,
            human_evaluation_status=args.human_evaluation_status,
            observed_head=_head(),
            command=_command(args),
            overwrite=bool(args.overwrite),
        )
    except (EvidenceSummaryError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"evidence summary build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "output_count": len(artifacts.output_paths),
                "finding_count": artifacts.finding_count,
                "reviewer_concern_count": artifacts.reviewer_concern_count,
                "stage_count": artifacts.stage_count,
                "cumulative_gpu_hours": artifacts.total_gpu_hours,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
