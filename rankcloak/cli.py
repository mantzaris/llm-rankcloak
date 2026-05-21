"""RankCloak command-line interface."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .experiments import main as experiment_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rankcloak", description="RankCloak research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment profile")
    run_parser.add_argument(
        "--profile",
        choices=[
            "audit-only",
            "codec-only",
            "small",
            "smoke",
            "dialogue-key-pilot",
            "payload-granularity-pilot",
            "paper-analysis",
            "paper-main",
            "paper-main-pilot",
            "segmented-protocol-pilot",
            "segmented-quality-controls",
            "strong-prompts",
            "strong-prompts-pilot",
        ],
        default="smoke",
    )
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--model-path", default=None)
    run_parser.add_argument("--max-payload-bytes", type=int, default=None)
    run_parser.add_argument("--skip-model-download", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        experiment_args = [
            "--profile",
            args.profile,
        ]
        if args.output_dir:
            experiment_args.extend(["--output-dir", args.output_dir])
        if args.model_path:
            experiment_args.extend(["--model-path", args.model_path])
        if args.max_payload_bytes is not None:
            experiment_args.extend(["--max-payload-bytes", str(args.max_payload_bytes)])
        if args.skip_model_download:
            experiment_args.append("--skip-model-download")
        if args.overwrite:
            experiment_args.append("--overwrite")
        return experiment_main(experiment_args)
    parser.error("Unknown command")


if __name__ == "__main__":
    main()
