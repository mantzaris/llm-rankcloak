#!/usr/bin/env python3
"""Run or verify the immutable payload-fidelity-v2 tokenizer preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from rankcloak.revision_tokenizer_preflight import (
    MANIFEST_HASH_FIELD,
    run_tokenizer_preflight,
    verify_preflight_output,
    write_preflight_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "revision_v1" / "tokenizer_preflight_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all pinned GGUF tokenizers against all 480 payloads and "
            "30 English/Spanish/Mandarin prompts without model evaluation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an already published bundle without loading tokenizers.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_existing:
            report = verify_preflight_output(args.output_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["scientific_status"] == "pass" else 2
        manifest, records = run_tokenizer_preflight(
            project_root=args.project_root,
            config_dir=args.config_dir,
        )
        published = write_preflight_output(args.output_dir, manifest, records)
        report = verify_preflight_output(args.output_dir)
        print(
            json.dumps(
                {
                    "status": published["status"],
                    "output_dir": str(args.output_dir),
                    MANIFEST_HASH_FIELD: published[MANIFEST_HASH_FIELD],
                    "record_count": report["record_count"],
                    "failure_count": report["failure_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if published["status"] == "pass" else 2
    except Exception as exc:
        print("tokenizer preflight failed: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

