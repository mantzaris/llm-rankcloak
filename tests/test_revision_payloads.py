import base64
from collections import Counter
from dataclasses import replace

from rankcloak.revision_payloads import (
    REVISION_CORPUS_SHA256,
    REVISION_HEX_PAYLOAD_CLASSES,
    REVISION_PAYLOAD_CLASSES,
    derive_public_bytes,
    generate_revision_v1_payloads,
    revision_corpus_sha256,
    validate_revision_corpus,
    validate_revision_payload,
)


def test_revision_corpus_is_exactly_eight_by_sixty_and_frozen():
    first = generate_revision_v1_payloads()
    second = generate_revision_v1_payloads()

    assert len(first) == 480
    assert [payload.payload_name for payload in first] == [
        payload.payload_name for payload in second
    ]
    assert [payload.payload_text for payload in first] == [
        payload.payload_text for payload in second
    ]
    assert Counter(payload.payload_class for payload in first) == {
        payload_class: 60 for payload_class in REVISION_PAYLOAD_CLASSES
    }
    assert revision_corpus_sha256(first) == REVISION_CORPUS_SHA256


def test_revision_corpus_vectors_pass_algorithm_validation():
    report = validate_revision_corpus()
    assert report["status"] == "ok"
    assert report["payload_count"] == 480
    assert report["corpus_sha256"] == REVISION_CORPUS_SHA256
    assert report["invalid_payload_names"] == []


def test_revision_payload_formats_and_lengths_are_exact():
    payloads = generate_revision_v1_payloads()
    expected = {
        "sha256_hex": (64, 32),
        "hmac_sha256_hex": (64, 32),
        "nonce_96_bit_hex": (24, 12),
        "token_128_bit_hex": (32, 16),
        "uuid_v4": (36, 16),
        "aes256_gcm_base64": (64, 48),
        "chacha20_poly1305_base64": (64, 48),
        "ed25519_signature_base64": (88, 64),
    }
    for payload in payloads:
        text_length, artifact_length = expected[payload.payload_class]
        assert len(payload.payload_text) == text_length
        assert len(payload.artifact_bytes) == artifact_length
        assert len(payload.payload_bytes) == text_length
        assert payload.artifact_bit_length == artifact_length * 8
        if payload.payload_class in REVISION_HEX_PAYLOAD_CLASSES:
            assert bytes.fromhex(payload.payload_text) == payload.artifact_bytes
        if payload.is_base64_like:
            assert (
                base64.b64decode(payload.payload_text, validate=True)
                == payload.artifact_bytes
            )


def test_public_derivation_is_domain_separated():
    sha_message = derive_public_bytes("sha256_hex", 0, "message", 32)
    sha_other_index = derive_public_bytes("sha256_hex", 1, "message", 32)
    hmac_message = derive_public_bytes("hmac_sha256_hex", 0, "message", 32)
    hmac_key = derive_public_bytes("hmac_sha256_hex", 0, "public-test-key", 32)
    assert len({sha_message, sha_other_index, hmac_message, hmac_key}) == 4


def test_tampered_payload_fails_closed():
    payload = generate_revision_v1_payloads()[0]
    tampered = replace(payload, payload_text="0" + payload.payload_text[1:])
    assert validate_revision_payload(payload) is True
    assert validate_revision_payload(tampered) is False
