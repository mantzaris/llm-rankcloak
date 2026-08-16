#!/usr/bin/env python3
"""Extract locked single-topic versus multi-topic mixed-model contrasts."""

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

from rankcloak.revision_topic_effects import (  # noqa: E402
    TopicEffectError,
    build_topic_effect_extraction,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-model-manifest", type=Path, required=True)
    parser.add_argument(
        "--extraction-config",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/revision_v1/evidence_specs/topic_effect_extraction.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [str(Path(__file__).resolve())]
    for option, value in (
        ("--mixed-model-manifest", args.mixed_model_manifest),
        ("--extraction-config", args.extraction_config),
        ("--output-dir", args.output_dir),
    ):
        values.extend((option, str(value.resolve())))
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_topic_effect_extraction(
            mixed_model_manifest=args.mixed_model_manifest,
            extraction_config=args.extraction_config,
            output_dir=args.output_dir,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except TopicEffectError as exc:
        raise SystemExit(f"topic-effect extraction failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "contrast_row_count": artifacts.contrast_row_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
