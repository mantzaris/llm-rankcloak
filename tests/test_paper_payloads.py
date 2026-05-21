import base64

from rankcloak.paper_payloads import (
    FULL_PAYLOAD_CLASSES,
    PILOT_PAYLOAD_CLASSES,
    generate_full_paper_payloads,
    generate_pilot_paper_payloads,
)
from rankcloak.rank_codec import is_hex_text


def test_paper_payload_generation_is_deterministic():
    first = generate_pilot_paper_payloads()
    second = generate_pilot_paper_payloads()
    assert [payload.payload_name for payload in first] == [payload.payload_name for payload in second]
    assert [payload.payload_text for payload in first] == [payload.payload_text for payload in second]


def test_expected_paper_payload_classes_exist():
    pilot_payloads = generate_pilot_paper_payloads()
    full_payloads = generate_full_paper_payloads()
    assert {payload.payload_class for payload in pilot_payloads} == set(PILOT_PAYLOAD_CLASSES)
    assert {payload.payload_class for payload in full_payloads} == set(FULL_PAYLOAD_CLASSES)


def test_paper_payload_names_are_unique():
    payloads = generate_full_paper_payloads()
    names = [payload.payload_name for payload in payloads]
    assert len(names) == len(set(names))


def test_hex_like_paper_payloads_are_lowercase_hex():
    for payload in generate_full_paper_payloads():
        if payload.is_hex_like:
            assert payload.payload_text == payload.payload_text.lower()
            assert is_hex_text(payload.payload_text)


def test_base64_like_paper_payloads_are_decodable():
    for payload in generate_full_paper_payloads():
        if payload.is_base64_like:
            assert base64.b64decode(payload.payload_text.encode("ascii"), validate=True)
