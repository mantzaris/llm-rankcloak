#!/usr/bin/env python3
"""Join hash-verified held-out evaluator scores into primary R features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_evaluator_join import (  # noqa: E402
    EvaluatorFeatureJoinError,
    join_primary_heldout_evaluator_features,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify primary preprocessing/evaluator lineage and emit the exact "
            "full-message feature table consumed by the locked R models."
        )
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        required=True,
        help="Primary preprocessing_output_manifest.json",
    )
    parser.add_argument(
        "--evaluator-feature-manifest",
        type=Path,
        action="append",
        required=True,
        help="One primary features_manifest.json; supply exactly three times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = join_primary_heldout_evaluator_features(
            preprocessing_manifest=args.preprocessing_manifest,
            evaluator_feature_manifests=args.evaluator_feature_manifest,
            output_dir=args.output_dir,
        )
    except EvaluatorFeatureJoinError as exc:
        print("held-out evaluator join failed: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(args.output_dir.resolve()),
                "primary_trial_count": manifest["primary_trial_count"],
                "primary_full_message_feature_rows": manifest[
                    "primary_full_message_feature_rows"
                ],
                "source_record_hashes_recomputed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
