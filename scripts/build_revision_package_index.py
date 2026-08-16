#!/usr/bin/env python3
"""Build the non-duplicative final computational evidence-package index."""

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

from rankcloak.revision_package_index import (  # noqa: E402
    PackageIndexError,
    build_package_index,
)


def _labeled_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise PackageIndexError(f"{option} values must use LABEL=PATH")
        if label in result:
            raise PackageIndexError(f"Duplicate {option} label: {label}")
        result[label] = Path(raw_path)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--component-manifest", action="append", default=[])
    parser.add_argument("--external-reference", action="append", default=[])
    parser.add_argument("--require-relative-path", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> str:
    values = [
        str(Path(__file__).resolve()),
        "--package-root",
        str(args.package_root.resolve()),
    ]
    for value in args.component_manifest:
        values.extend(["--component-manifest", value])
    for value in args.external_reference:
        values.extend(["--external-reference", value])
    for value in args.require_relative_path:
        values.extend(["--require-relative-path", value])
    if args.output is not None:
        values.extend(["--output", str(args.output.resolve())])
    if args.overwrite:
        values.append("--overwrite")
    return " ".join(shlex.quote(value) for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = build_package_index(
            package_root=args.package_root,
            component_manifests=_labeled_paths(
                args.component_manifest, option="--component-manifest"
            ),
            external_references=_labeled_paths(
                args.external_reference, option="--external-reference"
            ),
            required_relative_paths=args.require_relative_path,
            output_path=args.output,
            command=_command(args),
            overwrite=args.overwrite,
        )
    except PackageIndexError as exc:
        raise SystemExit(f"package index build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": artifacts.manifest_path,
                "manifest_sha256": artifacts.manifest_sha256,
                "package_file_count": artifacts.package_file_count,
                "external_reference_count": artifacts.external_reference_count,
                "validated_declared_output_count": (
                    artifacts.validated_declared_output_count
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
