#!/usr/bin/env python3
"""Validate saved revision artifacts and produce statistical analysis tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_statistics import (  # noqa: E402
    MixedEffectsUnavailable,
    RevisionStatisticsError,
    run_statistics_analysis,
)


DEFAULT_STATISTICS_CONFIG = (
    PROJECT_ROOT / "configs" / "revision_v1" / "statistics.json"
)


def _flatten(values: Sequence[Sequence[Path]] | None) -> list[Path]:
    return [path for group in (values or []) for path in group]


def _load_mixed_specs(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        value = value.get("models")
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise RevisionStatisticsError(
            "Mixed-effects specification must be a JSON list or {'models': [...]}"
        )
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze immutable revision CSV/JSONL artifacts with payload-level "
            "experimental units and grouped uncertainty."
        )
    )
    parser.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        action="append",
        help="One or more saved trial/recovery CSV or JSONL files.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="+",
        action="append",
        help="One or more saved text/quality-feature CSV or JSONL files.",
    )
    parser.add_argument(
        "--continuous-quality",
        type=Path,
        nargs="+",
        action="append",
        help=(
            "One or more held-out evaluator continuous-quality CSV or JSONL files."
        ),
    )
    parser.add_argument(
        "--detectors",
        type=Path,
        nargs="+",
        action="append",
        help="Saved detector predictions or grouped metric CSV/JSONL files.",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        nargs="+",
        action="append",
        help="One or more saved runtime-profile CSV or JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--statistics-config",
        type=Path,
        default=DEFAULT_STATISTICS_CONFIG,
        help="Frozen statistical-design JSON (defaults to revision_v1/statistics.json).",
    )
    parser.add_argument(
        "--mixed-effects-spec",
        type=Path,
        help=(
            "Optional JSON model specifications. Requested unavailable backends "
            "are reported explicitly and never replaced with fixed-effects fits."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use the deterministic synthetic fixture when no inputs are supplied "
            "and cap bootstrap resamples at 100."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known statistics outputs in the target directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    trial_paths = _flatten(args.trials)
    feature_paths = _flatten(args.features)
    continuous_quality_paths = _flatten(args.continuous_quality)
    detector_paths = _flatten(args.detectors)
    runtime_paths = _flatten(args.runtime)
    if not args.smoke and not any(
        (
            trial_paths,
            feature_paths,
            continuous_quality_paths,
            detector_paths,
            runtime_paths,
        )
    ):
        parser.error(
            "provide at least one saved input, or pass --smoke for the synthetic fixture"
        )
    try:
        artifacts = run_statistics_analysis(
            output_dir=args.output_dir,
            trial_paths=trial_paths,
            feature_paths=feature_paths,
            continuous_quality_paths=continuous_quality_paths,
            detector_paths=detector_paths,
            runtime_paths=runtime_paths,
            statistics_config=args.statistics_config,
            mixed_effects_specs=_load_mixed_specs(args.mixed_effects_spec),
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
    except (RevisionStatisticsError, MixedEffectsUnavailable) as exc:
        parser.exit(2, f"revision statistics failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": artifacts.output_dir,
                "files": artifacts.files,
                "analysis_unit": "payload",
                "segments_as_independent_observations": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
