#!/usr/bin/env python3
"""Materialize and freeze the public revision-v1 payload corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Optional, Sequence

from rankcloak.revision_artifacts import (
    file_sha256,
    write_immutable_json,
    write_immutable_jsonl,
)
from rankcloak.revision_payloads import (
    REVISION_CORPUS_ID,
    REVISION_CORPUS_SHA256,
    REVISION_DERIVATION_VERSION,
    REVISION_PUBLIC_SEED_MATERIAL,
    generate_revision_v1_payloads,
    revision_corpus_sha256,
    revision_payload_class_counts,
    revision_payload_records,
    validate_revision_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write immutable RankCloak revision-v1 public payload artifacts."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-corpus-sha256",
        default=REVISION_CORPUS_SHA256,
    )
    return parser


def cryptography_version() -> Optional[str]:
    try:
        return importlib.metadata.version("cryptography")
    except importlib.metadata.PackageNotFoundError:
        return None


def build_payload_artifacts(
    output_dir: Path,
    expected_corpus_sha256: Optional[str] = None,
) -> dict:
    payloads = generate_revision_v1_payloads()
    validation = validate_revision_corpus(
        payloads, expected_sha256=expected_corpus_sha256
    )
    if validation["status"] != "ok":
        raise RuntimeError(
            "Revision payload validation failed: {}".format(
                "; ".join(validation["errors"])
            )
        )
    output_dir = Path(output_dir)
    records = revision_payload_records(payloads, include_payload_text=True)
    payload_path = output_dir / "revision_payloads.jsonl"
    write_immutable_jsonl(payload_path, records)
    manifest = {
        "schema_version": "1.0",
        "manifest_type": "revision_payload_corpus",
        "corpus_id": REVISION_CORPUS_ID,
        "derivation_version": REVISION_DERIVATION_VERSION,
        "public_seed_material_utf8": REVISION_PUBLIC_SEED_MATERIAL.decode("ascii"),
        "public_seed_material_sha256": hashlib.sha256(
            REVISION_PUBLIC_SEED_MATERIAL
        ).hexdigest(),
        "cryptography_version": cryptography_version(),
        "payload_count": len(payloads),
        "class_counts": revision_payload_class_counts(payloads),
        "corpus_sha256": revision_corpus_sha256(payloads),
        "payload_file": payload_path.name,
        "payload_file_sha256": file_sha256(payload_path),
        "payload_file_size_bytes": payload_path.stat().st_size,
        "validation": validation,
        "scope": "Public deterministic test vectors only; no operational secrets.",
    }
    write_immutable_json(output_dir / "PAYLOAD_MANIFEST.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_payload_artifacts(
        output_dir=args.output_dir,
        expected_corpus_sha256=args.expected_corpus_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
