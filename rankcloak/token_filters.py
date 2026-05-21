"""Deterministic token filtering for RankCloak rank experiments."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .model_io import get_vocab_size, safe_detokenize


SAFE_TEXT_FILTER_V1 = "safe_text_filter_v1"
_ALLOWED_TOKEN_MASK_CACHE = {}


def is_safe_text_token_piece(piece: str) -> bool:
    """Return True when a decoded token piece looks safe for ordinary prose."""

    if not piece:
        return False
    if "\ufffd" in piece:
        return False
    if any(ord(character) < 32 and character not in {"\n", "\t"} for character in piece):
        return False

    lowered = piece.lower()
    stripped = piece.strip()
    blocked_substrings = [
        "```",
        "`",
        "\\section",
        "\\frac",
        "{\\",
        "\\rm",
        "http",
        "www.",
        ".com",
        "&lt",
        "&gt",
        "&amp",
        "</",
        "<",
        ">",
        "[",
        "]",
        "___",
        "|||",
    ]
    if any(fragment in lowered for fragment in blocked_substrings):
        return False
    if stripped.startswith("#") or stripped.startswith("##"):
        return False
    if "\\\\" in piece or piece.count("\\") >= 2:
        return False
    return True


def decoded_token_piece(model: Any, token_id: int) -> str:
    """Decode a single token id for filter inspection."""

    return safe_detokenize(model, [int(token_id)])


def build_allowed_token_mask(model: Any, filter_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Build a deterministic allowed-token mask for the requested filter."""

    if filter_name in (None, "", "none"):
        return None
    if filter_name != SAFE_TEXT_FILTER_V1:
        raise ValueError("Unknown token filter: {}".format(filter_name))
    cache_key = (id(model), filter_name)
    if cache_key not in _ALLOWED_TOKEN_MASK_CACHE:
        _ALLOWED_TOKEN_MASK_CACHE[cache_key] = _build_allowed_token_mask(model, filter_name)
    return _ALLOWED_TOKEN_MASK_CACHE[cache_key]


def _build_allowed_token_mask(model: Any, filter_name: str) -> np.ndarray:
    vocab_size = get_vocab_size(model)
    if vocab_size is None:
        raise ValueError("Cannot build token filter mask because vocab size is unavailable.")
    mask = np.zeros(int(vocab_size), dtype=bool)
    for token_id in range(int(vocab_size)):
        try:
            piece = decoded_token_piece(model, token_id)
        except Exception:
            piece = ""
        mask[token_id] = is_safe_text_token_piece(piece)
    if not np.any(mask):
        raise ValueError("Token filter {} rejected the entire vocabulary.".format(filter_name))
    return mask


def _allowed_token_ids(logits: Sequence[float], allowed_token_mask: Optional[Sequence[bool]]) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if allowed_token_mask is None:
        return np.arange(scores.shape[0], dtype=np.int64)
    mask = np.asarray(allowed_token_mask, dtype=bool)
    if mask.shape[0] != scores.shape[0]:
        raise ValueError("allowed_token_mask length does not match logits length")
    token_ids = np.nonzero(mask)[0].astype(np.int64)
    if token_ids.size == 0:
        raise ValueError("allowed_token_mask rejects the entire vocabulary")
    return token_ids


def choose_token_at_rank_with_optional_filter(
    logits: Sequence[float],
    rank: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> int:
    """Choose the token at a 1-indexed rank, optionally within an allowed-token set."""

    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be 1-indexed and positive")
    scores = np.asarray(logits, dtype=np.float64)
    token_ids = _allowed_token_ids(scores, allowed_token_mask)
    if rank > token_ids.size:
        raise ValueError("rank exceeds the allowed vocabulary size")
    order = np.lexsort((token_ids, -scores[token_ids]))
    return int(token_ids[order][rank - 1])


def rank_token_with_optional_filter(
    logits: Sequence[float],
    target_token_id: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> int:
    """Return the stable 1-indexed rank under an optional allowed-token set."""

    scores = np.asarray(logits, dtype=np.float64)
    target_token_id = int(target_token_id)
    if target_token_id < 0 or target_token_id >= scores.shape[0]:
        raise ValueError("target_token_id is outside the logits range")
    if allowed_token_mask is not None and not bool(allowed_token_mask[target_token_id]):
        raise ValueError("target token {} is not allowed by the token filter".format(target_token_id))
    token_ids = _allowed_token_ids(scores, allowed_token_mask)
    target_score = scores[target_token_id]
    higher_score_count = np.count_nonzero(scores[token_ids] > target_score)
    tied_lower_id_count = np.count_nonzero(
        (scores[token_ids] == target_score) & (token_ids < target_token_id)
    )
    return int(1 + higher_score_count + tied_lower_id_count)
