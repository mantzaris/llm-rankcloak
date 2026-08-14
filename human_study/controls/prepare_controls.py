#!/usr/bin/env python3
"""Offline import, audit, and selection for licensed human-written controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from control_pipeline import (  # noqa: E402
    DEFAULT_DESIGN,
    DEFAULT_PROFILES,
    DEFAULT_REGISTRY,
    HumanControlError,
    import_registered_source,
    load_json,
    load_reviews,
    read_csv,
    read_jsonl,
    select_human_controls,
    validate_profiles,
    validate_source_registry,
    write_import_artifacts,
    write_selection_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "import", help="Verify and screen a pinned local corpus without network access."
    )
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--source-id", required=True)
    audit.add_argument("--acquisition-date", required=True, help="ISO date, YYYY-MM-DD.")
    audit.add_argument("--source-registry", type=Path, default=DEFAULT_REGISTRY)
    audit.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    audit.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument(
        "--audit-only",
        action="store_true",
        help="Write only aggregate coverage/provenance audit; do not write licensed text.",
    )

    select = subparsers.add_parser(
        "select", help="Length-match manually approved candidates into all 72 strata."
    )
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--reviews", type=Path, required=True)
    select.add_argument("--targets", type=Path, required=True)
    select.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--max-relative-word-difference", type=float, default=0.35)
    return parser


def run_import(args: argparse.Namespace) -> int:
    registry_document = load_json(args.source_registry)
    sources = validate_source_registry(registry_document)
    if args.source_id not in sources:
        raise HumanControlError("Unregistered source_id: {}".format(args.source_id))
    design = load_json(args.design)
    profiles = validate_profiles(load_json(args.profiles), design)
    candidates, audit = import_registered_source(
        args.input,
        sources[args.source_id],
        profiles,
        args.acquisition_date,
        candidate_pool_minimum_per_template=(
            len(design["eligible_payload_classes"])
            * int(design["candidate_replicates_per_stratum"])
        ),
    )
    result = write_import_artifacts(
        args.output_dir, candidates, audit, audit_only=bool(args.audit_only)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    # Coverage audit success means the source was reproducible, not that stimuli
    # are approved. The audit embeds the blocking pre-recruitment state.
    return 0


def run_select(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates)
    reviews = load_reviews(args.reviews)
    targets = read_csv(args.targets)
    design = load_json(args.design)
    selected, manifest = select_human_controls(
        candidates,
        reviews,
        targets,
        design,
        max_relative_word_difference=args.max_relative_word_difference,
    )
    result = write_selection_artifacts(args.output_dir, selected, manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "import":
            return run_import(args)
        if args.command == "select":
            return run_select(args)
        raise HumanControlError("Unknown command")
    except HumanControlError as exc:
        print("human-control error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
