#!/usr/bin/env python3
"""Build payload-grouped evidence tables for the frozen ablation matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_ablation_evidence import (  # noqa: E402
    AblationEvidenceError,
    build_ablation_evidence,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--trials", type=Path, required=True)
    value.add_argument("--unavailable", type=Path, required=True)
    value.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/revision_v1/evidence_specs/ablation_evidence_analysis.json",
    )
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    command = " ".join(
        shlex.quote(part)
        for part in (
            str(Path(__file__).resolve()),
            "--trials",
            str(args.trials.resolve()),
            "--unavailable",
            str(args.unavailable.resolve()),
            "--config",
            str(args.config.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            *(('--overwrite',) if args.overwrite else ()),
        )
    )
    try:
        artifacts = build_ablation_evidence(
            trials_path=args.trials,
            unavailable_path=args.unavailable,
            config_path=args.config,
            output_dir=args.output_dir,
            command=command,
            overwrite=args.overwrite,
        )
    except AblationEvidenceError as exc:
        raise SystemExit(f"ablation evidence build failed: {exc}") from exc
    print(json.dumps(artifacts.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
