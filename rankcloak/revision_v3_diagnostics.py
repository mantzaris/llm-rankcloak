"""Shared next-token diagnostics for revision-V3 model-backed studies."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .model_io import evaluate_context, get_last_logits
from .rank_codec import token_log_probability
from .token_filters import (
    choose_token_at_rank_with_optional_filter,
    rank_token_with_optional_filter,
)


class GenerationDiagnosticError(ValueError):
    """Raised when a next-token diagnostic request is malformed."""


def shannon_entropy_bits(
    logits: Sequence[float], allowed_token_mask: Optional[Sequence[bool]] = None
) -> float:
    """Return numerically stable Shannon entropy in bits."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise GenerationDiagnosticError("logits must be a non-empty vector")
    if allowed_token_mask is not None:
        mask = np.asarray(allowed_token_mask, dtype=bool)
        if mask.shape != values.shape:
            raise GenerationDiagnosticError(
                "allowed-token mask must match logits"
            )
        values = values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise GenerationDiagnosticError("no finite admissible logits remain")
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    probabilities = weights / float(np.sum(weights))
    entropy_nats = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy_nats / math.log(2.0))


def next_token_diagnostic(
    logits: Sequence[float],
    observed_token_id: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Describe one observed token under the preceding model distribution."""

    token_id = int(observed_token_id)
    greedy_token_id = choose_token_at_rank_with_optional_filter(
        logits, 1, allowed_token_mask
    )
    observed_logp = float(token_log_probability(logits, token_id))
    greedy_logp = float(token_log_probability(logits, greedy_token_id))
    return {
        "entropy_bits": shannon_entropy_bits(logits, allowed_token_mask),
        "observed_token_id": token_id,
        "observed_rank": int(
            rank_token_with_optional_filter(
                logits, token_id, allowed_token_mask
            )
        ),
        "observed_log_probability": observed_logp,
        "observed_surprisal_nats": float(-observed_logp),
        "greedy_token_id": int(greedy_token_id),
        "greedy_log_probability": greedy_logp,
        "rank_pressure_log_probability_gap_nats": float(
            greedy_logp - observed_logp
        ),
    }


def trace_observed_tokens(
    model: Any,
    context_token_ids: Sequence[int],
    observed_token_ids: Sequence[int],
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Replay an observed token path and record every preceding distribution."""

    context = list(map(int, context_token_ids))
    observed = list(map(int, observed_token_ids))
    evaluate_context(model, context)
    diagnostics: List[Dict[str, object]] = []
    for token_id in observed:
        diagnostic = next_token_diagnostic(
            get_last_logits(model), token_id, allowed_token_mask
        )
        diagnostics.append(diagnostic)
        model.eval([token_id])
    return {
        "context_token_ids": context,
        "observed_token_ids": observed,
        "position_count": len(observed),
        "entropy_bits": [float(row["entropy_bits"]) for row in diagnostics],
        "observed_ranks": [int(row["observed_rank"]) for row in diagnostics],
        "observed_log_probabilities": [
            float(row["observed_log_probability"]) for row in diagnostics
        ],
        "observed_surprisals_nats": [
            float(row["observed_surprisal_nats"]) for row in diagnostics
        ],
        "greedy_token_ids": [int(row["greedy_token_id"]) for row in diagnostics],
        "greedy_log_probabilities": [
            float(row["greedy_log_probability"]) for row in diagnostics
        ],
        "rank_pressure_log_probability_gaps_nats": [
            float(row["rank_pressure_log_probability_gap_nats"])
            for row in diagnostics
        ],
    }


__all__ = [
    "GenerationDiagnosticError",
    "next_token_diagnostic",
    "shannon_entropy_bits",
    "trace_observed_tokens",
]
