from rankcloak.rank_codec import decode_hex_nibble_ranks_to_text, encode_hex_nibbles_to_ranks
from rankcloak.segmented_protocol import (
    CONTROL_CODEBOOK,
    chunk_rank_sequence,
    flatten_rank_chunks,
)
from rankcloak.synthetic_payloads import generate_synthetic_payloads


def test_chunk_rank_sequence_splits_and_flattens():
    ranks = list(range(1, 18))
    chunks = chunk_rank_sequence(ranks, 8)
    assert chunks == [list(range(1, 9)), list(range(9, 17)), [17]]
    assert flatten_rank_chunks(chunks) == ranks


def test_sha_like_payload_segments_to_eight_chunks():
    payloads = {payload.name: payload for payload in generate_synthetic_payloads()}
    payload = payloads["sha256_public_test_string"]
    encoded = encode_hex_nibbles_to_ranks(payload.text)
    chunks = chunk_rank_sequence(encoded["ranks"], 8)
    assert len(encoded["ranks"]) == 64
    assert len(chunks) == 8
    assert all(len(chunk) == 8 for chunk in chunks)


def test_control_codebook_c1_configuration():
    assert "C1" in CONTROL_CODEBOOK
    entry = CONTROL_CODEBOOK["C1"]
    assert entry["payload_codec"] == "raw_hex_nibbles"
    assert entry["segment_size"] == 8
    assert entry["decode_policy"] == "forced_prefix_only"
    assert entry["topic_schedule_name"] == "mixed_recipe_forum_car_blog"


def test_raw_hex_nibbles_roundtrip_for_protocol_payloads():
    payloads = {payload.name: payload for payload in generate_synthetic_payloads()}
    for payload_name in ["sha256_public_test_string", "random_128_bit_hex"]:
        payload = payloads[payload_name]
        encoded = encode_hex_nibbles_to_ranks(payload.text)
        assert all(1 <= rank <= 16 for rank in encoded["ranks"])
        decoded = decode_hex_nibble_ranks_to_text(encoded["ranks"], encoded["metadata"])
        assert decoded == payload.text.lower()
