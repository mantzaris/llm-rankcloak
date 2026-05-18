import pytest

from rankcloak.rank_codec import (
    decode_hex_nibble_ranks_to_text,
    encode_hex_nibbles_to_ranks,
    is_hex_text,
)


@pytest.mark.parametrize(
    "hex_text",
    [
        "0123456789abcdef",
        "21b723f1138ae8a043de0eb7124f5275",
        "635ae65bf4b30222b00e3ce63e133098",
    ],
)
def test_hex_nibble_codec_roundtrips(hex_text):
    encoded = encode_hex_nibbles_to_ranks(hex_text)
    assert encoded["metadata"]["encoding"] == "raw_hex_nibbles"
    assert encoded["metadata"]["hex_character_length"] == len(hex_text)
    assert all(1 <= rank <= 16 for rank in encoded["ranks"])
    decoded = decode_hex_nibble_ranks_to_text(encoded["ranks"], encoded["metadata"])
    assert decoded == hex_text.lower()


def test_invalid_hex_input_is_rejected():
    assert not is_hex_text("not-hex")
    with pytest.raises(ValueError):
        encode_hex_nibbles_to_ranks("not-hex")

