#!/usr/bin/env python3
"""Independently verify an offline RankCloak revision release candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rankcloak.revision_release import RevisionReleaseError, verify_release_candidate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument(
        "--require-doi-null", action="store_true",
        help="Fail unless the offline candidate records doi: null.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_release_candidate(args.candidate_dir)
        if args.require_doi_null and report["doi"] is not None:
            raise RevisionReleaseError("Candidate DOI is not null")
    except RevisionReleaseError as exc:
        raise SystemExit("revision release verification failed: {}".format(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
