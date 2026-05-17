import pytest

from rankcloak.rank_codec import (
    SUPPORTED_ALPHABET_SIZES,
    decode_bounded_ranks_to_bytes,
    encode_bytes_to_bounded_ranks,
)


DETERMINISTIC_BYTE_STRINGS = [
    b"",
    b"rankcloak",
    bytes(range(16)),
    bytes([255, 0, 1, 127, 128, 42]),
]


@pytest.mark.parametrize("alphabet_size", SUPPORTED_ALPHABET_SIZES)
@pytest.mark.parametrize("data", DETERMINISTIC_BYTE_STRINGS)
def test_bounded_rank_codec_roundtrip(alphabet_size, data):
    encoded = encode_bytes_to_bounded_ranks(data, alphabet_size)
    ranks = encoded["ranks"]
    metadata = encoded["metadata"]
    assert metadata["alphabet_size"] == alphabet_size
    assert metadata["original_byte_length"] == len(data)
    assert all(1 <= rank <= alphabet_size for rank in ranks)
    decoded = decode_bounded_ranks_to_bytes(ranks, metadata)
    assert decoded == data


def test_invalid_alphabet_size_is_rejected():
    with pytest.raises(ValueError):
        encode_bytes_to_bounded_ranks(b"abc", 3)

