from rankcloak.synthetic_payloads import generate_synthetic_payloads


EXPECTED_PAYLOAD_NAMES = {
    "sha256_public_test_string",
    "random_128_bit_hex",
    "random_256_bit_hex",
    "synthetic_uuid_v4_like",
    "synthetic_bearer_token_like",
    "synthetic_jwt_like_invalid_signature",
    "synthetic_hmac_like_tag",
    "synthetic_ciphertext_like_base64",
}


def test_payload_generation_is_deterministic():
    first = generate_synthetic_payloads()
    second = generate_synthetic_payloads()
    assert [(payload.name, payload.text) for payload in first] == [
        (payload.name, payload.text) for payload in second
    ]


def test_expected_payload_names_exist():
    names = {payload.name for payload in generate_synthetic_payloads()}
    assert EXPECTED_PAYLOAD_NAMES.issubset(names)


def test_all_payloads_are_marked_synthetic():
    payloads = generate_synthetic_payloads()
    assert payloads
    assert all(payload.is_synthetic for payload in payloads)
    assert all("synthetic" in payload.description.lower() or payload.kind in {"sha256_hex", "random_hex", "nonce_hex"} for payload in payloads)

