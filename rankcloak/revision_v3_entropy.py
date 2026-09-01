"""Entropy-gated RankCloak protocol primitives for revision V3.

This is a payload-carrying eligibility gate inspired by entropy-aware test-time
watermarking, not an implementation of the watermarking task in Cai et al.
At each payload-span step the encoder computes Shannon entropy over the same
admissible next-token distribution used for rank selection.  It consumes a
payload rank only when entropy is at least the public threshold; otherwise it
emits admissible rank 1 and leaves the payload index unchanged.  Replay repeats
that decision before observing each saved token, so no gate-position side
channel is required.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .model_io import evaluate_context, get_last_logits, safe_detokenize
from .rank_codec import token_log_probability
from .revision_protocol import (
    TAIL_NONE,
    context_sha256,
    generate_rank_span,
    retokenize_message,
)
from .token_filters import (
    choose_token_at_rank_with_optional_filter,
    rank_token_with_optional_filter,
)


SCHEMA_VERSION = "rankcloak-entropy-gate-v1"
GATE_RULE = "eligible_when_filtered_next_token_shannon_entropy_bits_gte_threshold_v1"


class EntropyGateError(ValueError):
    """Raised when an entropy-gated generation request is malformed."""


def shannon_entropy_bits(
    logits: Sequence[float], allowed_token_mask: Optional[Sequence[bool]] = None
) -> float:
    """Return numerically stable Shannon entropy in bits."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise EntropyGateError("logits must be a non-empty vector")
    if allowed_token_mask is not None:
        mask = np.asarray(allowed_token_mask, dtype=bool)
        if mask.shape != values.shape:
            raise EntropyGateError("allowed-token mask must match logits")
        values = values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise EntropyGateError("no finite admissible logits remain")
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    probabilities = weights / total
    entropy_nats = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy_nats / math.log(2.0))


def entropy_eligible(entropy_bits: float, threshold_bits: Optional[float]) -> bool:
    """Apply the inclusive, predeclared threshold boundary."""

    if threshold_bits is None:
        return True
    threshold = float(threshold_bits)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise EntropyGateError("entropy threshold must be finite and non-negative")
    return float(entropy_bits) >= threshold


def _annotate_ungated(
    generated: Dict[str, object],
    original_rank_count: int,
    generated_rank_count: int,
) -> Dict[str, object]:
    result = dict(generated)
    forced_ids = list(map(int, result["forced_token_ids"]))
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "gate_rule": GATE_RULE,
            "entropy_threshold_bits": None,
            "embedding_token_ids": forced_ids,
            "embedding_text": str(result["forced_text"]),
            "embedding_entropies_bits": [None] * len(forced_ids),
            "embedding_eligible_mask": [True] * len(forced_ids),
            "embedding_token_roles": ["payload"] * len(forced_ids),
            "consumed_payload_rank_indices": list(range(generated_rank_count)),
            "requested_payload_rank_count": int(original_rank_count),
            "consumed_payload_rank_count": int(generated_rank_count),
            "eligible_position_count": int(generated_rank_count),
            "ineligible_position_count": 0,
            "payload_completion": bool(generated_rank_count == original_rank_count),
            "payload_fraction_embedded": (
                1.0
                if original_rank_count == 0
                else float(generated_rank_count / original_rank_count)
            ),
            "capacity_failure": (
                None
                if generated_rank_count == original_rank_count
                else "maximum_generated_tokens_reached_before_payload_completion"
            ),
            "embedding_start": int(result["forced_start"]),
            "embedding_stop": int(result["forced_stop"]),
        }
    )
    return result


def generate_entropy_gated_span(
    model: Any,
    context_token_ids: Sequence[int],
    ranks: Sequence[int],
    *,
    entropy_threshold_bits: Optional[float],
    maximum_generated_tokens: Optional[int] = None,
    allowed_token_mask: Optional[Sequence[bool]] = None,
    leadin_token_count: int = 0,
    tail_policy: str = TAIL_NONE,
    quality_rank_ceiling: Optional[int] = None,
) -> Dict[str, object]:
    """Generate a bounded payload span under the entropy eligibility rule."""

    requested_ranks = list(map(int, ranks))
    if any(rank < 1 for rank in requested_ranks):
        raise EntropyGateError("payload ranks must be positive")
    leadin_count = int(leadin_token_count)
    if leadin_count < 0:
        raise EntropyGateError("leadin_token_count must be non-negative")
    if maximum_generated_tokens is None:
        if entropy_threshold_bits is None:
            budget = len(requested_ranks)
        else:
            raise EntropyGateError(
                "a finite maximum_generated_tokens is required when the gate is enabled"
            )
    else:
        budget = int(maximum_generated_tokens)
    if budget < 0:
        raise EntropyGateError("maximum_generated_tokens must be non-negative")

    if entropy_threshold_bits is None:
        embedded = requested_ranks[:budget]
        generated = generate_rank_span(
            model,
            context_token_ids,
            embedded,
            allowed_token_mask=allowed_token_mask,
            leadin_token_count=leadin_count,
            tail_policy=tail_policy,
            quality_rank_ceiling=quality_rank_ceiling,
        )
        return _annotate_ungated(generated, len(requested_ranks), len(embedded))

    threshold = float(entropy_threshold_bits)
    entropy_eligible(0.0, threshold)
    if quality_rank_ceiling is not None and int(quality_rank_ceiling) < 1:
        raise EntropyGateError("quality_rank_ceiling must be positive")

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    leadin_ids: List[int] = []
    leadin_logp: List[float] = []
    for _ in range(leadin_count):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(
            logits, 1, allowed_token_mask
        )
        leadin_ids.append(int(token_id))
        leadin_logp.append(float(token_log_probability(logits, token_id)))
        model.eval([int(token_id)])

    embedding_ids: List[int] = []
    embedding_logp: List[float] = []
    entropies: List[float] = []
    eligible_mask: List[bool] = []
    roles: List[str] = []
    realized_ranks: List[Optional[int]] = []
    consumed_indices: List[int] = []
    greedy_ids: List[int] = []
    greedy_logp: List[float] = []
    rank_b_ids: Optional[List[int]] = [] if quality_rank_ceiling is not None else None
    rank_b_logp: Optional[List[float]] = [] if quality_rank_ceiling is not None else None
    payload_index = 0
    for _ in range(budget):
        if payload_index >= len(requested_ranks):
            break
        logits = get_last_logits(model)
        entropy = shannon_entropy_bits(logits, allowed_token_mask)
        eligible = entropy_eligible(entropy, threshold)
        greedy_id = choose_token_at_rank_with_optional_filter(
            logits, 1, allowed_token_mask
        )
        greedy_ids.append(int(greedy_id))
        greedy_logp.append(float(token_log_probability(logits, greedy_id)))
        if eligible:
            requested_rank = int(requested_ranks[payload_index])
            if quality_rank_ceiling is not None and requested_rank > int(
                quality_rank_ceiling
            ):
                raise EntropyGateError(
                    "payload rank exceeds quality_rank_ceiling"
                )
            token_id = choose_token_at_rank_with_optional_filter(
                logits, requested_rank, allowed_token_mask
            )
            consumed_indices.append(int(payload_index))
            payload_index += 1
            roles.append("payload")
            realized_ranks.append(requested_rank)
        else:
            token_id = greedy_id
            roles.append("unforced_skip")
            realized_ranks.append(None)
        if quality_rank_ceiling is not None:
            rank_b_id = choose_token_at_rank_with_optional_filter(
                logits, int(quality_rank_ceiling), allowed_token_mask
            )
            assert rank_b_ids is not None and rank_b_logp is not None
            rank_b_ids.append(int(rank_b_id))
            rank_b_logp.append(float(token_log_probability(logits, rank_b_id)))
        embedding_ids.append(int(token_id))
        embedding_logp.append(float(token_log_probability(logits, token_id)))
        entropies.append(float(entropy))
        eligible_mask.append(bool(eligible))
        model.eval([int(token_id)])

    completed = payload_index == len(requested_ranks)
    tail_ids: List[int] = []
    tail_logp: List[float] = []
    tail_text = ""
    tail_stop_reason = "none"
    tail_censored = False
    if completed and tail_policy != TAIL_NONE:
        tail_result = generate_rank_span(
            model,
            context + leadin_ids + embedding_ids,
            [],
            allowed_token_mask=allowed_token_mask,
            leadin_token_count=0,
            tail_policy=tail_policy,
            quality_rank_ceiling=None,
        )
        tail_ids = list(map(int, tail_result["tail_token_ids"]))
        tail_logp = list(map(float, tail_result["tail_log_probabilities"]))
        tail_text = str(tail_result["tail_text"])
        tail_stop_reason = str(tail_result["tail_stop_reason"])
        tail_censored = bool(tail_result["tail_censored"])

    full_ids = leadin_ids + embedding_ids + tail_ids
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_rule": GATE_RULE,
        "entropy_threshold_bits": threshold,
        "context_token_ids": context,
        "context_sha256": context_sha256(context),
        "leadin_token_ids": leadin_ids,
        "leadin_log_probabilities": leadin_logp,
        "leadin_text": safe_detokenize(model, leadin_ids),
        "embedding_token_ids": embedding_ids,
        "embedding_log_probabilities": embedding_logp,
        "embedding_text": safe_detokenize(model, embedding_ids),
        "embedding_entropies_bits": entropies,
        "embedding_eligible_mask": eligible_mask,
        "embedding_token_roles": roles,
        "consumed_payload_rank_indices": consumed_indices,
        "requested_payload_rank_count": int(len(requested_ranks)),
        "consumed_payload_rank_count": int(payload_index),
        "eligible_position_count": int(sum(eligible_mask)),
        "ineligible_position_count": int(len(eligible_mask) - sum(eligible_mask)),
        "payload_completion": bool(completed),
        "payload_fraction_embedded": (
            1.0
            if len(requested_ranks) == 0
            else float(payload_index / len(requested_ranks))
        ),
        "capacity_failure": (
            None
            if completed
            else "maximum_generated_tokens_reached_before_payload_completion"
        ),
        "realized_ranks": realized_ranks,
        "greedy_token_ids": greedy_ids,
        "greedy_log_probabilities": greedy_logp,
        "quality_rank_ceiling": (
            int(quality_rank_ceiling) if quality_rank_ceiling is not None else None
        ),
        "rank_B_token_ids": rank_b_ids,
        "rank_B_log_probabilities": rank_b_logp,
        "tail_token_ids": tail_ids,
        "tail_log_probabilities": tail_logp,
        "tail_text": tail_text,
        "tail_stop_reason": tail_stop_reason,
        "tail_censored": tail_censored,
        "full_token_ids": full_ids,
        "full_text": safe_detokenize(model, full_ids),
        "forced_token_ids": embedding_ids,
        "forced_text": safe_detokenize(model, embedding_ids),
        "forced_start": int(len(leadin_ids)),
        "forced_stop": int(len(leadin_ids) + len(embedding_ids)),
        "embedding_start": int(len(leadin_ids)),
        "embedding_stop": int(len(leadin_ids) + len(embedding_ids)),
    }


def recover_entropy_gated_span(
    model: Any,
    context_token_ids: Sequence[int],
    leadin_token_ids: Sequence[int],
    embedding_token_ids: Sequence[int],
    *,
    entropy_threshold_bits: Optional[float],
    expected_payload_rank_count: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Replay gate decisions and recover ranks from saved embedding token IDs."""

    expected = int(expected_payload_rank_count)
    if expected < 0:
        raise EntropyGateError("expected_payload_rank_count must be non-negative")
    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    for token_id in map(int, leadin_token_ids):
        model.eval([token_id])
    ranks: List[int] = []
    entropies: List[float] = []
    eligible_mask: List[bool] = []
    log_probabilities: List[float] = []
    for token_id in map(int, embedding_token_ids):
        logits = get_last_logits(model)
        entropy = shannon_entropy_bits(logits, allowed_token_mask)
        eligible = entropy_eligible(entropy, entropy_threshold_bits)
        entropies.append(float(entropy))
        eligible_mask.append(bool(eligible))
        if eligible and len(ranks) < expected:
            ranks.append(
                int(
                    rank_token_with_optional_filter(
                        logits, token_id, allowed_token_mask
                    )
                )
            )
        log_probabilities.append(float(token_log_probability(logits, token_id)))
        model.eval([token_id])
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_rule": GATE_RULE,
        "entropy_threshold_bits": entropy_threshold_bits,
        "ranks": ranks,
        "embedding_entropies_bits": entropies,
        "embedding_eligible_mask": eligible_mask,
        "token_log_probabilities": log_probabilities,
        "expected_payload_rank_count": expected,
        "recovered_payload_rank_count": int(len(ranks)),
        "payload_completion": bool(len(ranks) == expected),
        "context_sha256": context_sha256(context),
    }


def calibrate_entropy_gate_thresholds(
    ordinary_development_entropies_bits: Sequence[float],
) -> Dict[str, object]:
    """Freeze median and upper-quartile gates from clean development traces.

    Detector labels and detector outcomes are not inputs. NumPy's linear
    quantile convention is recorded so independent reruns agree at finite
    sample boundaries.
    """

    values = np.asarray(ordinary_development_entropies_bits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise EntropyGateError("development entropies must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise EntropyGateError("development entropies must be finite and non-negative")
    moderate = float(np.quantile(values, 0.50, method="linear"))
    strict = float(np.quantile(values, 0.75, method="linear"))
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "ordinary_generation_development_traces_only",
        "development_position_count": int(values.size),
        "quantile_method": "numpy_linear",
        "moderate_quantile": 0.50,
        "moderate_threshold_bits": moderate,
        "strict_quantile": 0.75,
        "strict_threshold_bits": strict,
        "detector_outcomes_used": False,
    }


def retokenize_entropy_gated_message(
    model: Any, generated: Dict[str, object]
) -> Dict[str, object]:
    """Run the visible-text tokenization diagnostic independently of replay."""

    diagnostic = retokenize_message(model, generated)
    diagnostic["evaluation_scope"] = "visible_text_retokenization_independent_of_saved_id_replay"
    diagnostic["embedding_token_ids"] = list(
        map(int, diagnostic.pop("forced_token_ids"))
    )
    return diagnostic


__all__ = [
    "EntropyGateError",
    "GATE_RULE",
    "SCHEMA_VERSION",
    "calibrate_entropy_gate_thresholds",
    "entropy_eligible",
    "generate_entropy_gated_span",
    "recover_entropy_gated_span",
    "retokenize_entropy_gated_message",
    "shannon_entropy_bits",
]
