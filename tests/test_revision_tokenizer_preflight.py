import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rankcloak.revision_artifacts import ImmutableArtifactError
from rankcloak.revision_config import load_revision_config_set
from rankcloak.revision_tokenizer_preflight import (
    MANIFEST_HASH_FIELD,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
    TokenizerPreflightError,
    audit_payload_tokenization,
    audit_prompt_context,
    collect_preflight_prompts,
    verify_preflight_output,
    write_preflight_output,
)


class FusionTokenizer:
    """Small tokenizer double exposing the historical first-token traps."""

    def __init__(self, *, adds_bos, split_at=1, prefix=b"", suffix=b""):
        self.adds_bos = bool(adds_bos)
        self.split_at = int(split_at)
        self.prefix = bytes(prefix)
        self.suffix = bytes(suffix)
        self._piece_to_id = {}
        self._id_to_piece = {}

    def token_bos(self):
        # Qwen-like doubles expose a BOS identity but need not auto-insert it.
        return 99

    def _id(self, piece):
        key = bytes(piece)
        if key not in self._piece_to_id:
            token_id = 100 + len(self._piece_to_id)
            self._piece_to_id[key] = token_id
            self._id_to_piece[token_id] = key
        return self._piece_to_id[key]

    def tokenize(self, value, add_bos=False, special=False):
        del special
        raw = bytes(value)
        split = min(max(1, self.split_at), len(raw))
        pieces = [self.prefix + raw[:split]]
        if split < len(raw):
            pieces.append(raw[split:])
        values = [self._id(piece) for piece in pieces]
        if add_bos and self.adds_bos:
            values.insert(0, 99)
        return values

    def detokenize(self, token_ids):
        return b"".join(self._id_to_piece[int(value)] for value in token_ids) + self.suffix


def _payload(text):
    return SimpleNamespace(
        payload_name="fixture_payload",
        payload_class="fixture",
        payload_index=0,
        payload_text=text,
        payload_bytes=text.encode("utf-8"),
    )


def _prompt(text="Write complete prose."):
    return {
        "prompt_id": "fixture_prompt",
        "category_id": "fixture_category",
        "language": "en",
        "text": text,
    }


@pytest.mark.parametrize("split_at", [1, 2])
def test_payload_fusion_tokens_recover_every_original_byte(split_at):
    model = FusionTokenizer(adds_bos=False, split_at=split_at)
    record = audit_payload_tokenization(model, "qwen_fixture", _payload("abcdef"))

    assert record["audit_status"] == "pass"
    assert record["token_ids"]
    assert record["framing"]["exact_original_byte_recovery"] is True
    assert record["framing"]["prefix_byte_length"] == 0
    assert record["failure_codes"] == []


@pytest.mark.parametrize("prefix", [b" ", b"  "])
def test_explicit_ascii_space_prefix_is_reversible_and_recorded(prefix):
    model = FusionTokenizer(adds_bos=True, split_at=2, prefix=prefix)
    record = audit_payload_tokenization(model, "mistral_fixture", _payload("abcdef"))

    assert record["audit_status"] == "pass"
    assert record["framing"]["prefix_permitted"] is True
    assert record["framing"]["prefix_byte_length"] == len(prefix)
    assert record["framing"]["exact_original_byte_recovery"] is True


def test_nonspace_prefix_and_suffix_transformations_fail_closed():
    nonspace = audit_payload_tokenization(
        FusionTokenizer(adds_bos=False, prefix=b"x"),
        "bad_prefix",
        _payload("abcdef"),
    )
    suffix = audit_payload_tokenization(
        FusionTokenizer(adds_bos=False, suffix=b"!"),
        "bad_suffix",
        _payload("abcdef"),
    )

    assert nonspace["audit_status"] == "fail"
    assert "non_space_or_non_prefix_transformation" in nonspace["failure_codes"]
    assert suffix["audit_status"] == "fail"
    assert "payload_not_exact_suffix" in suffix["failure_codes"]
    assert "original_payload_bytes_not_recovered" in suffix["failure_codes"]


def test_no_bos_model_retains_first_real_prompt_token():
    model = FusionTokenizer(adds_bos=False, split_at=2)
    record = audit_prompt_context(model, "qwen_fixture", _prompt())

    assert record["audit_status"] == "pass"
    assert record["leading_actual_bos_removed"] is False
    assert record["removed_token_count"] == 0
    assert record["first_real_token_retained"] is True
    assert record["context_token_ids"] == record["no_special_token_ids"]
    assert record["context_token_ids"][0] == record["special_enabled_token_ids"][0]


def test_actual_bos_is_removed_once_without_deleting_first_real_token():
    model = FusionTokenizer(adds_bos=True, split_at=1, prefix=b" ")
    record = audit_prompt_context(model, "mistral_fixture", _prompt())

    assert record["audit_status"] == "pass"
    assert record["leading_actual_bos_removed"] is True
    assert record["removed_token_count"] == 1
    assert record["special_enabled_token_ids"][0] == 99
    assert record["context_token_ids"] == record["special_enabled_token_ids"][1:]
    assert record["first_context_token_id"] == record["first_no_special_token_id"]
    assert record["framing"]["prefix_byte_length"] == 1


def test_frozen_prompt_registry_contains_exact_18_6_6_panel():
    prompts = collect_preflight_prompts(load_revision_config_set())
    counts = {
        language: sum(row["language"] == language for row in prompts)
        for language in ("en", "es", "zh_hans")
    }

    assert len(prompts) == 30
    assert len({row["prompt_id"] for row in prompts}) == 30
    assert counts == {"en": 18, "es": 6, "zh_hans": 6}


def _publishable_bundle():
    model = FusionTokenizer(adds_bos=False, split_at=2)
    rows = [
        audit_payload_tokenization(model, "fixture", _payload("abcdef")),
        audit_prompt_context(model, "fixture", _prompt()),
    ]
    for index, row in enumerate(rows):
        row["record_index"] = index
    manifest = {
        "schema_version": "2.0",
        "manifest_type": "rankcloak_tokenizer_preflight",
        "preflight_id": "fixture",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "status": "pass",
        "counts": {
            "total_check_count": 2,
            "failure_count": 0,
        },
    }
    return manifest, rows


def test_output_bundle_is_self_hashed_atomically_published_and_no_overwrite(tmp_path):
    output = tmp_path / "tokenizer_preflight_v2"
    manifest, rows = _publishable_bundle()
    published = write_preflight_output(output, manifest, rows)
    report = verify_preflight_output(output)

    assert published[MANIFEST_HASH_FIELD] == report["preflight_manifest_sha256"]
    assert report["status"] == "ok"
    assert report["scientific_status"] == "pass"
    assert report["record_count"] == 2
    assert {path.name for path in output.iterdir()} == {
        "TOKENIZER_PREFLIGHT_MANIFEST.json",
        "records.jsonl",
        "failures.jsonl",
    }
    with pytest.raises(ImmutableArtifactError, match="refusing overwrite"):
        write_preflight_output(output, manifest, rows)


def test_verification_detects_record_and_manifest_tampering(tmp_path):
    output = tmp_path / "tokenizer_preflight_v2"
    manifest, rows = _publishable_bundle()
    write_preflight_output(output, manifest, rows)

    records_path = output / "records.jsonl"
    records_path.write_bytes(records_path.read_bytes() + b" \n")
    with pytest.raises(TokenizerPreflightError, match="size mismatch"):
        verify_preflight_output(output)

    second = tmp_path / "second"
    write_preflight_output(second, manifest, rows)
    manifest_path = second / "TOKENIZER_PREFLIGHT_MANIFEST.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["status"] = "fail"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TokenizerPreflightError, match="self-hash"):
        verify_preflight_output(second)

