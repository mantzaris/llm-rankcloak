"""Baseline cover generation helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from .model_io import evaluate_context, get_last_logits, safe_detokenize
from .rank_codec import rank_of_token, token_id_at_rank, token_log_probability


def generate_greedy_baseline(
    model: Any,
    context_token_ids: Sequence[int],
    max_tokens: int,
) -> Dict[str, object]:
    """Generate a greedy baseline by repeatedly selecting the rank-1 token."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    generated_token_ids = []
    ranks = []
    token_log_probabilities = []
    started_at = time.perf_counter()
    for _ in range(int(max_tokens)):
        logits = get_last_logits(model)
        token_id = token_id_at_rank(logits, 1)
        generated_token_ids.append(token_id)
        ranks.append(rank_of_token(logits, token_id))
        token_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    generation_seconds = time.perf_counter() - started_at
    generated_text = safe_detokenize(model, generated_token_ids)
    return {
        "baseline_mode": "greedy",
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "generated_token_count": len(generated_token_ids),
        "generated_character_count": len(generated_text),
        "generation_seconds": generation_seconds,
        "ranks": ranks,
        "token_log_probabilities": token_log_probabilities,
    }

