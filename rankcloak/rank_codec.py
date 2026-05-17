"""Stable rank logic and bounded-rank codecs."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .model_io import (
    evaluate_context,
    get_last_logits,
    make_context_token_ids,
    safe_detokenize,
    tokenize_payload_text,
)


HEX_ALPHABET = "0123456789abcdef"
SUPPORTED_ALPHABET_SIZES = [2, 4, 8, 16, 32, 64]


def sorted_token_ids_from_logits(logits: Sequence[float]) -> np.ndarray:
    """Return token ids sorted by decreasing logit, tie-broken by token id."""

    scores = np.asarray(logits, dtype=np.float64)
    token_ids = np.arange(scores.shape[0], dtype=np.int64)
    # np.lexsort uses the last key as primary: -score first, token id second.
    order = np.lexsort((token_ids, -scores))
    return token_ids[order]


def rank_of_token(logits: Sequence[float], token_id: int) -> int:
    """Return the stable 1-indexed rank of token_id under the logits."""

    scores = np.asarray(logits, dtype=np.float64)
    token_id = int(token_id)
    if token_id < 0 or token_id >= scores.shape[0]:
        raise ValueError("token_id is outside the logits range")
    target_score = scores[token_id]
    token_ids = np.arange(scores.shape[0], dtype=np.int64)
    # Higher scores outrank the target; equal scores only outrank when id is lower.
    higher_score_count = np.count_nonzero(scores > target_score)
    tied_lower_id_count = np.count_nonzero((scores == target_score) & (token_ids < token_id))
    return int(1 + higher_score_count + tied_lower_id_count)


def token_id_at_rank(logits: Sequence[float], rank: int) -> int:
    """Return the token id at a stable 1-indexed rank."""

    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be 1-indexed and positive")
    sorted_ids = sorted_token_ids_from_logits(logits)
    if rank > len(sorted_ids):
        raise ValueError("rank exceeds the vocabulary size")
    return int(sorted_ids[rank - 1])


def token_log_probability(logits: Sequence[float], token_id: int) -> float:
    """Return the selected token log probability with a stable log-softmax."""

    scores = np.asarray(logits, dtype=np.float64)
    token_id = int(token_id)
    maximum_score = float(np.max(scores))
    log_normalizer = maximum_score + float(np.log(np.sum(np.exp(scores - maximum_score))))
    return float(scores[token_id] - log_normalizer)


def test_stable_rank_ordering() -> Dict[str, object]:
    """Small deterministic tie-breaking test for the required rank order."""

    logits = np.array([0.1, 2.0, 2.0, -1.0, 0.1], dtype=np.float64)
    sorted_ids = sorted_token_ids_from_logits(logits).tolist()
    ranks = {token_id: rank_of_token(logits, token_id) for token_id in range(len(logits))}
    expected_sorted_ids = [1, 2, 0, 4, 3]
    return {
        "passed": sorted_ids == expected_sorted_ids and ranks == {1: 1, 2: 2, 0: 3, 4: 4, 3: 5},
        "sorted_ids": sorted_ids,
        "ranks": ranks,
        "expected_sorted_ids": expected_sorted_ids,
    }


def rank_trace_from_token_ids(
    model: Any,
    context_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
) -> Dict[str, object]:
    """Compute a rank trace for target tokens after a context."""

    context = list(map(int, context_token_ids))
    targets = list(map(int, target_token_ids))
    evaluate_context(model, context)
    ranks: List[int] = []
    token_log_probabilities: List[float] = []
    for token_id in targets:
        logits = get_last_logits(model)
        ranks.append(rank_of_token(logits, token_id))
        token_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    return {
        "context_token_ids": context,
        "target_token_ids": targets,
        "ranks": ranks,
        "token_log_probabilities": token_log_probabilities,
        "success": True,
    }


def direct_subword_ranks_for_text(model: Any, text: str) -> Dict[str, object]:
    """Tokenize a payload and compute its direct subword rank trace."""

    payload_token_ids = tokenize_payload_text(model, text)
    bos_context = make_context_token_ids(model, "")
    trace = rank_trace_from_token_ids(model, bos_context, payload_token_ids)
    trace["payload_token_ids"] = payload_token_ids
    return trace


def generate_token_ids_from_ranks(
    model: Any,
    context_token_ids: Sequence[int],
    ranks: Sequence[int],
) -> Dict[str, object]:
    """Generate token ids by selecting each requested rank under the context."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    generated_ids: List[int] = []
    selected_log_probabilities: List[float] = []
    for rank in ranks:
        logits = get_last_logits(model)
        token_id = token_id_at_rank(logits, int(rank))
        generated_ids.append(token_id)
        selected_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    return {
        "context_token_ids": context,
        "ranks": list(map(int, ranks)),
        "generated_token_ids": generated_ids,
        "generated_text": safe_detokenize(model, generated_ids),
        "token_log_probabilities": selected_log_probabilities,
        "success": True,
    }


def recover_ranks_from_generated_ids(
    model: Any,
    context_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
) -> Dict[str, object]:
    """Recover ranks from generated token ids under the same key context."""

    return rank_trace_from_token_ids(model, context_token_ids, generated_token_ids)


def encode_bytes_to_bounded_ranks(data: bytes, alphabet_size: int) -> Dict[str, object]:
    """Encode bytes as 1-indexed ranks using a power-of-two alphabet size."""

    alphabet_size = int(alphabet_size)
    if alphabet_size not in SUPPORTED_ALPHABET_SIZES:
        raise ValueError("alphabet_size must be one of {}".format(SUPPORTED_ALPHABET_SIZES))
    bits_per_symbol = int(math.log2(alphabet_size))
    bit_string = "".join(format(byte, "08b") for byte in data)
    padding_bits = (-len(bit_string)) % bits_per_symbol
    if padding_bits:
        bit_string += "0" * padding_bits
    digits = [
        int(bit_string[index : index + bits_per_symbol], 2)
        for index in range(0, len(bit_string), bits_per_symbol)
    ]
    ranks = [digit + 1 for digit in digits]
    return {
        "ranks": ranks,
        "metadata": {
            "alphabet_size": alphabet_size,
            "bits_per_symbol": bits_per_symbol,
            "original_byte_length": len(data),
            "padding_bits": padding_bits,
        },
    }


def decode_bounded_ranks_to_bytes(ranks: Sequence[int], metadata: Dict[str, int]) -> bytes:
    """Decode 1-indexed bounded ranks back to the original bytes."""

    alphabet_size = int(metadata["alphabet_size"])
    bits_per_symbol = int(metadata["bits_per_symbol"])
    original_byte_length = int(metadata["original_byte_length"])
    padding_bits = int(metadata.get("padding_bits", 0))
    max_rank = alphabet_size
    digits = []
    for rank in ranks:
        rank = int(rank)
        if rank < 1 or rank > max_rank:
            raise ValueError("bounded rank {} is outside 1..{}".format(rank, max_rank))
        digits.append(rank - 1)
    bit_string = "".join(format(digit, "0{}b".format(bits_per_symbol)) for digit in digits)
    if padding_bits:
        bit_string = bit_string[:-padding_bits]
    byte_values = [
        int(bit_string[index : index + 8], 2)
        for index in range(0, min(len(bit_string), original_byte_length * 8), 8)
    ]
    return bytes(byte_values[:original_byte_length])


def encode_bytes_as_hex_character_ranks(data: bytes) -> Dict[str, object]:
    """Encode bytes as lowercase hex characters mapped to ranks 1..16."""

    hex_text = data.hex()
    ranks = [HEX_ALPHABET.index(character) + 1 for character in hex_text]
    return {
        "ranks": ranks,
        "metadata": {
            "encoding": "hex_character",
            "alphabet_size": 16,
            "original_byte_length": len(data),
            "hex_length": len(hex_text),
        },
    }


def decode_hex_character_ranks_to_bytes(ranks: Sequence[int], metadata: Dict[str, int]) -> bytes:
    """Decode hex-character ranks back to bytes."""

    characters = []
    for rank in ranks:
        rank = int(rank)
        if rank < 1 or rank > 16:
            raise ValueError("hex rank {} is outside 1..16".format(rank))
        characters.append(HEX_ALPHABET[rank - 1])
    data = bytes.fromhex("".join(characters))
    return data[: int(metadata["original_byte_length"])]


def bounded_roundtrip_rows(payloads: Iterable[Any], alphabet_sizes: Sequence[int]) -> List[dict]:
    """Create legacy recovery rows for hex-character and fixed-radix encodings."""

    rows: List[dict] = []
    for payload in payloads:
        hex_encoded = encode_bytes_as_hex_character_ranks(payload.bytes_value)
        hex_decoded = decode_hex_character_ranks_to_bytes(
            hex_encoded["ranks"], hex_encoded["metadata"]
        )
        rows.append(
            {
                "payload_name": payload.name,
                "encoding": "hex_character",
                "alphabet_size": 16,
                "rank_count": len(hex_encoded["ranks"]),
                "original_byte_length": len(payload.bytes_value),
                "exact_recovery": hex_decoded == payload.bytes_value,
                "expansion_ranks_per_byte": len(hex_encoded["ranks"]) / max(len(payload.bytes_value), 1),
            }
        )
        for alphabet_size in alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(payload.bytes_value, alphabet_size)
            decoded = decode_bounded_ranks_to_bytes(encoded["ranks"], encoded["metadata"])
            rows.append(
                {
                    "payload_name": payload.name,
                    "encoding": "fixed_radix_bits",
                    "alphabet_size": int(alphabet_size),
                    "rank_count": len(encoded["ranks"]),
                    "original_byte_length": len(payload.bytes_value),
                    "exact_recovery": decoded == payload.bytes_value,
                    "expansion_ranks_per_byte": len(encoded["ranks"]) / max(len(payload.bytes_value), 1),
                }
            )
    return rows


def codec_roundtrip_rows(payloads: Iterable[Any], alphabet_sizes: Sequence[int]) -> List[dict]:
    """Create explicit codec-level byte/rank roundtrip result rows."""

    rows: List[dict] = []
    for payload in payloads:
        hex_encoded = encode_bytes_as_hex_character_ranks(payload.bytes_value)
        hex_decoded = decode_hex_character_ranks_to_bytes(
            hex_encoded["ranks"], hex_encoded["metadata"]
        )
        rows.append(
            {
                "payload_name": payload.name,
                "payload_kind": payload.kind,
                "payload_byte_length": len(payload.bytes_value),
                "encoding_name": "hex_character",
                "alphabet_size": 16,
                "rank_count": len(hex_encoded["ranks"]),
                "exact_roundtrip": hex_decoded == payload.bytes_value,
                "notes": "lowercase hex characters map directly to ranks 1..16",
            }
        )
        for alphabet_size in alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(payload.bytes_value, alphabet_size)
            decoded = decode_bounded_ranks_to_bytes(encoded["ranks"], encoded["metadata"])
            rows.append(
                {
                    "payload_name": payload.name,
                    "payload_kind": payload.kind,
                    "payload_byte_length": len(payload.bytes_value),
                    "encoding_name": "fixed_radix_bits",
                    "alphabet_size": int(alphabet_size),
                    "rank_count": len(encoded["ranks"]),
                    "exact_roundtrip": decoded == payload.bytes_value,
                    "notes": "metadata stores original byte length and padding bits",
                }
            )
    return rows
