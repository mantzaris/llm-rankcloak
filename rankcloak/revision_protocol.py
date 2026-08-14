"""Protocol primitives for the confirmatory RankCloak revision study.

The submitted paper's pilot code remains untouched.  This module defines the
frozen revision-v1 execution semantics used by the multi-model confirmatory
study: direct rank transcoding, bounded codecs, deterministic filters and
tails, exact token-ID replay, text-retokenization diagnostics, and controlled
transmission transformations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model_io import (
    detokenize_bytes,
    evaluate_context,
    get_last_logits,
    make_context_token_ids,
    safe_detokenize,
    tokenize_bytes,
    tokenize_payload_text,
)
from .rank_codec import (
    decode_bounded_ranks_to_bytes,
    decode_hex_nibble_ranks_to_text,
    encode_bytes_to_bounded_ranks,
    encode_hex_nibbles_to_ranks,
    rank_of_token,
    token_id_at_rank,
    token_log_probability,
)
from .segmented_protocol import should_stop_sentence_tail
from .token_filters import (
    SAFE_TEXT_FILTER_V1,
    build_allowed_token_mask,
    choose_token_at_rank_with_optional_filter,
    rank_token_with_optional_filter,
)


ROUND_TRIP_STABLE_FILTER_V1 = "roundtrip_stable_filter_v1"
TAIL_NONE = "none"
TAIL_FIXED_40 = "fixed_tail40"
TAIL_SENTENCE_20_60 = "sentence_tail_min20_max60"
TAIL_SEMANTIC_V1 = "dynamic_completion_v1"
REPLAY_TOKEN_IDS = "saved_token_ids"
REPLAY_TEXT_RETOKENIZE = "detokenized_text_retokenized"
REPLAY_GREEDY_LEADIN = "greedy_leadin_regeneration"


@dataclass(frozen=True)
class Representation:
    """A payload represented as a deterministic 1-indexed rank sequence."""

    name: str
    ranks: Tuple[int, ...]
    metadata: Mapping[str, object]
    payload_bytes: bytes
    payload_text: str


def canonical_json_sha256(value: object) -> str:
    """Return a stable SHA-256 digest for JSON-compatible protocol state."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_sha256(token_ids: Sequence[int]) -> str:
    return canonical_json_sha256([int(token_id) for token_id in token_ids])


def _strip_optional_bos(model: Any, token_ids: Sequence[int]) -> List[int]:
    """Strip one model BOS token without assuming a model family."""

    values = list(map(int, token_ids))
    bos = getattr(model, "token_bos", None)
    if callable(bos):
        try:
            bos = int(bos())
        except Exception:
            bos = None
    elif bos is not None:
        try:
            bos = int(bos)
        except Exception:
            bos = None
    if values and bos is not None and values[0] == bos:
        return values[1:]
    return values


def text_to_token_ids(model: Any, text: str) -> List[int]:
    """Tokenize transmitted text without retaining an automatically added BOS."""

    return _strip_optional_bos(
        model,
        tokenize_bytes(model, text.encode("utf-8"), add_bos=True),
    )


def build_round_trip_stable_mask(
    model: Any,
    require_safe_text: bool = True,
) -> np.ndarray:
    """Return tokens whose isolated text deterministically retokenizes to itself.

    This deliberately modest criterion is context independent and auditable. It
    does not claim that arbitrary edited strings preserve token boundaries; the
    transmission experiment measures that separate property.
    """

    safe_mask = build_allowed_token_mask(model, SAFE_TEXT_FILTER_V1) if require_safe_text else None
    vocab_size = len(safe_mask) if safe_mask is not None else None
    if vocab_size is None:
        from .model_io import get_vocab_size

        vocab_size = get_vocab_size(model)
    if vocab_size is None:
        raise ValueError("Cannot build round-trip mask: model vocabulary size is unavailable")
    mask = np.zeros(int(vocab_size), dtype=bool)
    for token_id in range(int(vocab_size)):
        if safe_mask is not None and not bool(safe_mask[token_id]):
            continue
        piece = safe_detokenize(model, [token_id])
        if not piece or "\ufffd" in piece:
            continue
        try:
            retokenized = text_to_token_ids(model, piece)
        except Exception:
            continue
        mask[token_id] = retokenized == [token_id]
    if not np.any(mask):
        raise ValueError("Round-trip-stable filter rejected the entire vocabulary")
    return mask


def build_revision_filter_mask(model: Any, filter_name: str) -> Optional[np.ndarray]:
    if filter_name in {"", "none", None}:
        return None
    if filter_name == SAFE_TEXT_FILTER_V1:
        return build_allowed_token_mask(model, SAFE_TEXT_FILTER_V1)
    if filter_name == ROUND_TRIP_STABLE_FILTER_V1:
        return build_round_trip_stable_mask(model, require_safe_text=True)
    raise ValueError("Unknown revision filter: {}".format(filter_name))


_DANGLING_TAIL_WORDS = {
    "a", "an", "and", "as", "at", "because", "but", "by", "for", "from",
    "if", "in", "of", "or", "so", "than", "that", "the", "then", "to",
    "when", "while", "with",
}


def delimiters_balanced(text: str) -> bool:
    """Conservatively check delimiters for the deterministic dynamic tail."""

    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for character in text:
        if character in pairs.values():
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    if stack:
        return False
    if text.count("\u201c") != text.count("\u201d") or text.count("\u2018") != text.count("\u2019"):
        return False
    without_apostrophes = re.sub(r"(?<=\w)'(?=\w)", "", text)
    return without_apostrophes.count('"') % 2 == 0


def dynamic_tail_complete(text: str, token_count: int, minimum_tokens: int = 8) -> bool:
    """Apply the frozen semantic-completeness stopping heuristic."""

    if int(token_count) < int(minimum_tokens):
        return False
    stripped = str(text).rstrip()
    if not stripped or not stripped.endswith((".", "!", "?", "\n\n")):
        return False
    if not delimiters_balanced(stripped):
        return False
    without_ending = re.sub(r"[.!?\"'\u201d\u2019\s]+$", "", stripped)
    if not without_ending or without_ending.endswith((",", ";", ":", "-", "\u2014")):
        return False
    words = re.findall(r"[A-Za-z]+", without_ending.lower())
    return not words or words[-1] not in _DANGLING_TAIL_WORDS


def bounded_representation(
    payload_bytes: bytes,
    payload_text: str,
    codec: str,
) -> Representation:
    """Create an ASCII B=8/B=16 or eligible raw-hex representation."""

    if codec == "ascii_b8":
        encoded = encode_bytes_to_bounded_ranks(payload_bytes, 8)
    elif codec == "ascii_b16":
        encoded = encode_bytes_to_bounded_ranks(payload_bytes, 16)
    elif codec == "hex_nibble":
        encoded = encode_hex_nibbles_to_ranks(payload_text)
    else:
        raise ValueError("Unknown bounded codec: {}".format(codec))
    return Representation(
        name=codec,
        ranks=tuple(map(int, encoded["ranks"])),
        metadata=dict(encoded["metadata"]),
        payload_bytes=bytes(payload_bytes),
        payload_text=str(payload_text),
    )


def direct_representation(model: Any, payload_text: str) -> Representation:
    """Represent literal payload subwords under an empty/BOS context.

    The tokenizer receives the original UTF-8 bytes with special-token insertion
    disabled. Some vocabularies deterministically detokenize a beginning-of-text
    token sequence with space bytes in front. Such bytes are explicit reversible
    framing metadata; deletion of a payload byte or any non-space affix fails
    before generation.
    """

    payload_bytes = payload_text.encode("utf-8")
    target_ids = tokenize_payload_text(model, payload_text)
    serialized_bytes = detokenize_bytes(model, target_ids)
    if not serialized_bytes.endswith(payload_bytes):
        raise ValueError(
            "Direct-subword tokenizer does not preserve the original payload as an exact suffix"
        )
    prefix_bytes = serialized_bytes[: len(serialized_bytes) - len(payload_bytes)]
    if prefix_bytes.strip(b" "):
        raise ValueError(
            "Direct-subword tokenizer introduced a non-space or suffix transformation"
        )
    source_context = make_context_token_ids(model, "")
    evaluate_context(model, source_context)
    ranks: List[int] = []
    log_probabilities: List[float] = []
    for token_id in target_ids:
        logits = get_last_logits(model)
        ranks.append(rank_of_token(logits, token_id))
        log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([int(token_id)])
    return Representation(
        name="direct_subword",
        ranks=tuple(ranks),
        metadata={
            "source_context_token_ids": list(map(int, source_context)),
            "payload_token_ids": list(map(int, target_ids)),
            "payload_token_log_probabilities": log_probabilities,
            "payload_tokenization_contract": (
                "literal_utf8_no_special_tokens_reversible_space_prefix_v2"
            ),
            "original_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "target_detokenized_sha256": hashlib.sha256(serialized_bytes).hexdigest(),
            "detokenized_prefix_bytes_base64": base64.b64encode(prefix_bytes).decode("ascii"),
            "detokenized_prefix_byte_length": len(prefix_bytes),
            "detokenized_prefix_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
            "payload_fidelity_preflight": True,
        },
        payload_bytes=payload_bytes,
        payload_text=payload_text,
    )


def decode_representation(
    model: Any,
    representation: Representation,
    ranks: Sequence[int],
) -> Dict[str, object]:
    """Decode ranks and distinguish representation replay from payload recovery."""

    recovered_ranks = list(map(int, ranks))
    original_sha256 = hashlib.sha256(representation.payload_bytes).hexdigest()
    serialized_bytes = b""
    prefix_exact = True
    try:
        if representation.name in {"ascii_b8", "ascii_b16"}:
            recovered_bytes = decode_bounded_ranks_to_bytes(
                recovered_ranks, dict(representation.metadata)
            )
            recovered_text = recovered_bytes.decode("utf-8", errors="replace")
            serialized_bytes = recovered_bytes
            token_ids: List[int] = []
        elif representation.name == "hex_nibble":
            recovered_text = decode_hex_nibble_ranks_to_text(
                recovered_ranks, dict(representation.metadata)
            )
            recovered_bytes = recovered_text.encode("utf-8")
            serialized_bytes = recovered_bytes
            token_ids = []
        elif representation.name == "direct_subword":
            metadata = dict(representation.metadata)
            if metadata.get("payload_tokenization_contract") != (
                "literal_utf8_no_special_tokens_reversible_space_prefix_v2"
            ):
                raise ValueError("Unknown or missing direct payload-tokenization contract")
            if metadata.get("original_payload_sha256") != original_sha256:
                raise ValueError("Direct representation original-payload hash mismatch")
            prefix_bytes = base64.b64decode(
                str(metadata["detokenized_prefix_bytes_base64"]), validate=True
            )
            if (
                len(prefix_bytes) != int(metadata["detokenized_prefix_byte_length"])
                or hashlib.sha256(prefix_bytes).hexdigest()
                != metadata["detokenized_prefix_sha256"]
                or prefix_bytes.strip(b" ")
            ):
                raise ValueError("Direct representation prefix metadata is invalid")
            source_context = list(map(int, metadata["source_context_token_ids"]))
            evaluate_context(model, source_context)
            token_ids = []
            for rank in recovered_ranks:
                token_id = token_id_at_rank(get_last_logits(model), rank)
                token_ids.append(token_id)
                model.eval([token_id])
            serialized_bytes = detokenize_bytes(model, token_ids)
            prefix_exact = serialized_bytes.startswith(prefix_bytes)
            recovered_bytes = (
                serialized_bytes[len(prefix_bytes) :] if prefix_exact else b""
            )
            recovered_text = recovered_bytes.decode("utf-8", errors="replace")
        else:
            raise ValueError("Unknown representation: {}".format(representation.name))
    except Exception as exc:
        return {
            "success": False,
            "recovery_outcome_semantics": "original_serialized_payload_bytes_sha256_v1",
            "exact_recovery": False,
            "exact_payload_recovery": False,
            "exact_representation_recovery": False,
            "recovered_bytes": b"",
            "recovered_text": "",
            "recovered_serialized_bytes": b"",
            "recovered_token_ids": [],
            "original_payload_sha256": original_sha256,
            "recovered_payload_sha256": hashlib.sha256(b"").hexdigest(),
            "detokenized_prefix_exact": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }

    exact_representation = recovered_ranks == list(map(int, representation.ranks))
    if representation.name == "direct_subword":
        expected_token_ids = list(map(int, representation.metadata["payload_token_ids"]))
        exact_representation = exact_representation and token_ids == expected_token_ids
    exact_payload = (
        prefix_exact
        and recovered_bytes == representation.payload_bytes
        and hashlib.sha256(recovered_bytes).hexdigest() == original_sha256
    )
    return {
        "success": True,
        "recovery_outcome_semantics": "original_serialized_payload_bytes_sha256_v1",
        # exact_recovery is the primary original-artifact outcome.
        "exact_recovery": bool(exact_payload),
        "exact_payload_recovery": bool(exact_payload),
        "exact_representation_recovery": bool(exact_representation),
        "recovered_bytes": recovered_bytes,
        "recovered_text": recovered_text,
        "recovered_serialized_bytes": serialized_bytes,
        "recovered_token_ids": token_ids,
        "original_payload_sha256": original_sha256,
        "recovered_payload_sha256": hashlib.sha256(recovered_bytes).hexdigest(),
        "detokenized_prefix_exact": bool(prefix_exact),
        "error": None,
    }


def generate_rank_span(
    model: Any,
    context_token_ids: Sequence[int],
    ranks: Sequence[int],
    allowed_token_mask: Optional[Sequence[bool]] = None,
    leadin_token_count: int = 0,
    tail_policy: str = TAIL_NONE,
    quality_rank_ceiling: Optional[int] = None,
) -> Dict[str, object]:
    """Generate one lead-in/forced-span/tail message using serial evaluation.

    ``quality_rank_ceiling`` is the fixed bounded alphabet size ``B`` used for
    same-context quality validation. When supplied, every forced step records
    the model log probabilities of the realized token, admissible rank 1, and
    admissible rank ``B`` before the realized token is appended. Direct
    subword transcoding has no single fixed ``B`` and therefore leaves the
    rank-``B`` endpoint unavailable while still recording the greedy endpoint.
    """

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    leadin_ids: List[int] = []
    forced_ids: List[int] = []
    tail_ids: List[int] = []
    leadin_logp: List[float] = []
    forced_logp: List[float] = []
    tail_logp: List[float] = []
    realized_ranks: List[int] = []
    greedy_token_ids: List[int] = []
    greedy_logp: List[float] = []
    rank_b_token_ids: Optional[List[int]] = (
        [] if quality_rank_ceiling is not None else None
    )
    rank_b_logp: Optional[List[float]] = (
        [] if quality_rank_ceiling is not None else None
    )

    if quality_rank_ceiling is not None and int(quality_rank_ceiling) < 1:
        raise ValueError("quality_rank_ceiling must be a positive rank")

    for _ in range(int(leadin_token_count)):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, 1, allowed_token_mask)
        leadin_ids.append(token_id)
        leadin_logp.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    for requested_rank in ranks:
        logits = get_last_logits(model)
        realized_rank = int(requested_rank)
        if (
            quality_rank_ceiling is not None
            and realized_rank > int(quality_rank_ceiling)
        ):
            raise ValueError(
                "realized rank {} exceeds quality rank ceiling {}".format(
                    realized_rank, int(quality_rank_ceiling)
                )
            )
        greedy_token_id = choose_token_at_rank_with_optional_filter(
            logits, 1, allowed_token_mask
        )
        greedy_token_ids.append(greedy_token_id)
        greedy_logp.append(token_log_probability(logits, greedy_token_id))
        if quality_rank_ceiling is not None:
            rank_b_token_id = choose_token_at_rank_with_optional_filter(
                logits, int(quality_rank_ceiling), allowed_token_mask
            )
            assert rank_b_token_ids is not None and rank_b_logp is not None
            rank_b_token_ids.append(rank_b_token_id)
            rank_b_logp.append(token_log_probability(logits, rank_b_token_id))
        token_id = choose_token_at_rank_with_optional_filter(
            logits, realized_rank, allowed_token_mask
        )
        realized_ranks.append(realized_rank)
        forced_ids.append(token_id)
        forced_logp.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    if tail_policy == TAIL_NONE:
        maximum_tail_tokens = 0
    elif tail_policy == TAIL_FIXED_40:
        maximum_tail_tokens = 40
    elif tail_policy == TAIL_SENTENCE_20_60:
        maximum_tail_tokens = 60
    elif tail_policy == TAIL_SEMANTIC_V1:
        maximum_tail_tokens = 256
    else:
        raise ValueError("Unknown tail policy: {}".format(tail_policy))

    for _ in range(maximum_tail_tokens):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, 1, allowed_token_mask)
        tail_ids.append(token_id)
        tail_logp.append(token_log_probability(logits, token_id))
        model.eval([token_id])
        tail_text = safe_detokenize(model, tail_ids)
        if tail_policy == TAIL_SENTENCE_20_60 and should_stop_sentence_tail(
            tail_text, len(tail_ids), 20, 60
        ):
            break
        if tail_policy == TAIL_SEMANTIC_V1 and dynamic_tail_complete(
            tail_text, len(tail_ids), minimum_tokens=8
        ):
            break

    full_ids = leadin_ids + forced_ids + tail_ids
    dynamic_censored = bool(
        tail_policy == TAIL_SEMANTIC_V1
        and len(tail_ids) >= 256
        and not dynamic_tail_complete(safe_detokenize(model, tail_ids), len(tail_ids), 8)
    )
    return {
        "context_token_ids": context,
        "context_sha256": context_sha256(context),
        "leadin_token_ids": leadin_ids,
        "forced_token_ids": forced_ids,
        "tail_token_ids": tail_ids,
        "full_token_ids": full_ids,
        "leadin_text": safe_detokenize(model, leadin_ids),
        "forced_text": safe_detokenize(model, forced_ids),
        "tail_text": safe_detokenize(model, tail_ids),
        "full_text": safe_detokenize(model, full_ids),
        "leadin_log_probabilities": leadin_logp,
        "forced_log_probabilities": forced_logp,
        "realized_ranks": realized_ranks,
        "greedy_token_ids": greedy_token_ids,
        "greedy_log_probabilities": greedy_logp,
        "quality_rank_ceiling": (
            int(quality_rank_ceiling)
            if quality_rank_ceiling is not None
            else None
        ),
        "rank_B_token_ids": rank_b_token_ids,
        "rank_B_log_probabilities": rank_b_logp,
        "tail_log_probabilities": tail_logp,
        "tail_stop_reason": (
            "none"
            if tail_policy == TAIL_NONE
            else "emergency_cap_censored"
            if dynamic_censored
            else "semantic_complete"
            if tail_policy == TAIL_SEMANTIC_V1
            else "sentence_or_cap"
            if tail_policy == TAIL_SENTENCE_20_60
            else "fixed_length"
        ),
        "tail_censored": dynamic_censored,
        "forced_start": len(leadin_ids),
        "forced_stop": len(leadin_ids) + len(forced_ids),
    }


def recover_rank_span(
    model: Any,
    context_token_ids: Sequence[int],
    leadin_token_ids: Sequence[int],
    forced_token_ids: Sequence[int],
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Recover a forced rank span using the same serial schedule as generation."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    for token_id in map(int, leadin_token_ids):
        model.eval([token_id])
    ranks: List[int] = []
    logp: List[float] = []
    for token_id in map(int, forced_token_ids):
        logits = get_last_logits(model)
        ranks.append(
            rank_token_with_optional_filter(logits, token_id, allowed_token_mask)
        )
        logp.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    return {
        "ranks": ranks,
        "token_log_probabilities": logp,
        "context_sha256": context_sha256(context),
    }


def regenerate_greedy_leadin(
    model: Any,
    context_token_ids: Sequence[int],
    count: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> List[int]:
    """Regenerate a lead-in serially; this is a distinct tested replay mode."""

    evaluate_context(model, list(map(int, context_token_ids)))
    result: List[int] = []
    for _ in range(int(count)):
        token_id = choose_token_at_rank_with_optional_filter(
            get_last_logits(model), 1, allowed_token_mask
        )
        result.append(token_id)
        model.eval([token_id])
    return result


def first_divergence(expected: Sequence[int], observed: Sequence[int]) -> Dict[str, object]:
    """Return the first differing index and values, including length divergence."""

    expected_values = list(map(int, expected))
    observed_values = list(map(int, observed))
    common = min(len(expected_values), len(observed_values))
    index = next(
        (i for i in range(common) if expected_values[i] != observed_values[i]),
        None,
    )
    if index is None and len(expected_values) != len(observed_values):
        index = common
    return {
        "diverged": index is not None,
        "position_zero_based": index,
        "expected_value": (
            expected_values[index] if index is not None and index < len(expected_values) else None
        ),
        "observed_value": (
            observed_values[index] if index is not None and index < len(observed_values) else None
        ),
        "expected_length": len(expected_values),
        "observed_length": len(observed_values),
    }


def retokenize_message(model: Any, generated: Mapping[str, object]) -> Dict[str, object]:
    """Retokenize public text and diagnose whether recorded span boundaries survive."""

    expected = list(map(int, generated["full_token_ids"]))
    observed = text_to_token_ids(model, str(generated["full_text"]))
    start = int(generated["forced_start"])
    stop = int(generated["forced_stop"])
    return {
        "retokenized_token_ids": observed,
        "forced_token_ids": observed[start:stop],
        "full_token_ids_match": expected == observed,
        "divergence": first_divergence(expected, observed),
        "boundary_rule": "saved token offsets applied after full-text retokenization",
    }


def apply_transmission_transform(
    text: str,
    transform: str,
    seed: int = 0,
) -> str:
    """Apply one frozen, deterministic text-space channel transformation."""

    value = str(text)
    if transform == "unmodified":
        return value
    if transform in {"line_endings_crlf", "line_endings"}:
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    if transform == "whitespace_trim":
        return "\n".join(line.strip() for line in value.splitlines()).strip()
    if transform == "whitespace_collapse":
        return re.sub(r"\s+", " ", value).strip()
    if transform == "unicode_nfc":
        return unicodedata.normalize("NFC", value)
    if transform in {"unicode_nfkc", "unicode_normalization"}:
        return unicodedata.normalize("NFKC", value)
    if transform == "smart_quote_conversion":
        return value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    if transform == "quote_conversion":
        converted = re.sub(r"(?<=\w)'(?=\w)", "\u2019", value)
        output: List[str] = []
        double_open = True
        single_open = True
        for character in converted:
            if character == '"':
                output.append("\u201c" if double_open else "\u201d")
                double_open = not double_open
            elif character == "'":
                output.append("\u2018" if single_open else "\u2019")
                single_open = not single_open
            else:
                output.append(character)
        return "".join(output)
    if transform in {"markdown_copy", "markdown_copy_paste"}:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join("> " + line.rstrip() for line in normalized.strip().split("\n"))
    if not value:
        return value
    position_digest = int(
        hashlib.sha256((str(seed) + "|" + transform).encode()).hexdigest()[:8], 16
    )
    if transform in {"character_insert", "character_insertion"}:
        position = position_digest % len(value)
        return value[:position] + "x" + value[position:]
    if transform in {"character_delete", "character_deletion"}:
        eligible = [index for index, character in enumerate(value) if not character.isspace()]
        if not eligible:
            return value
        position = eligible[position_digest % len(eligible)]
        return value[:position] + value[position + 1 :]
    if transform in {"character_substitute", "character_substitution"}:
        eligible = [index for index, character in enumerate(value) if character.isalnum()]
        if not eligible:
            return value
        position = eligible[position_digest % len(eligible)]
        replacement = "x" if value[position] != "x" else "y"
        return value[:position] + replacement + value[position + 1 :]
    if transform in {"truncation_10pct", "truncation"}:
        keep = max(0, int(len(value) * 0.9))
        return value[:keep]
    raise ValueError("Unknown text-space transmission transform: {}".format(transform))


def transform_token_ids(
    token_ids: Sequence[int],
    transform: str,
    seed: int = 0,
) -> List[int]:
    """Apply transformations explicitly defined in token rather than text space."""

    values = list(map(int, token_ids))
    if transform == "unmodified":
        return values
    if transform == "token_deletion":
        # The frozen channel condition excludes the first and final token so
        # it does not conflate deletion with explicit boundary truncation.
        if len(values) <= 2:
            return values
        eligible = list(range(1, len(values) - 1))
        position = eligible[
            int(
                hashlib.sha256(
                    (str(seed) + "|token_deletion").encode()
                ).hexdigest()[:8],
                16,
            )
            % len(eligible)
        ]
        return values[:position] + values[position + 1 :]
    if transform == "truncation":
        remove = max(1, int(np.ceil(len(values) * 0.1))) if values else 0
        return values[: len(values) - remove]
    raise ValueError("Transformation is not token-space: {}".format(transform))


def diagnose_rank_failure(
    expected_ranks: Sequence[int],
    recovered_ranks: Sequence[int],
    expected_token_ids: Sequence[int],
    observed_token_ids: Sequence[int],
    context_token_ids: Sequence[int],
    boundary_offsets: Tuple[int, int],
    category: str,
) -> Dict[str, object]:
    """Create the mandatory machine-readable failure record."""

    rank_diff = first_divergence(expected_ranks, recovered_ranks)
    token_diff = first_divergence(expected_token_ids, observed_token_ids)
    return {
        "failure_category": str(category),
        "first_rank_divergence": rank_diff,
        "first_token_divergence": token_diff,
        "expected_token_id": token_diff["expected_value"],
        "recovered_token_id": token_diff["observed_value"],
        "expected_rank": rank_diff["expected_value"],
        "recovered_rank": rank_diff["observed_value"],
        "context_sha256": context_sha256(context_token_ids),
        "boundary_start": int(boundary_offsets[0]),
        "boundary_stop": int(boundary_offsets[1]),
    }
