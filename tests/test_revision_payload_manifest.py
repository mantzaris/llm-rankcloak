import json

import pytest

from rankcloak.revision_artifacts import file_sha256
from rankcloak.revision_payloads import REVISION_CORPUS_SHA256
from scripts.build_revision_payload_manifest import build_payload_artifacts


def test_payload_manifest_builder_is_immutable_and_reproducible(tmp_path):
    first = build_payload_artifacts(tmp_path)
    second = build_payload_artifacts(tmp_path)

    assert first == second
    assert first["payload_count"] == 480
    assert first["corpus_sha256"] == REVISION_CORPUS_SHA256
    assert first["payload_file_sha256"] == file_sha256(
        tmp_path / "revision_payloads.jsonl"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "revision_payloads.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 480
    assert all("payload_text" in row for row in rows)


def test_payload_manifest_builder_rejects_wrong_expected_hash(tmp_path):
    with pytest.raises(RuntimeError, match="corpus SHA-256 mismatch"):
        build_payload_artifacts(tmp_path, expected_corpus_sha256="0" * 64)
