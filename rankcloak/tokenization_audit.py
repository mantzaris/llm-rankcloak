"""Tokenization audit helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable, List

from .model_io import safe_detokenize, tokenize_payload_text
from .synthetic_payloads import SyntheticPayload


def audit_payload_tokenization(
    payloads: Iterable[SyntheticPayload],
    model: Any = None,
    first_token_limit: int = 12,
) -> List[dict]:
    """Audit payload lengths and, when available, LLM tokenizer behavior."""

    rows = []
    for payload in payloads:
        token_ids = []
        token_pieces = []
        tokenizer_available = model is not None
        if tokenizer_available:
            token_ids = tokenize_payload_text(model, payload.text)
            token_pieces = [
                safe_detokenize(model, [token_id]) for token_id in token_ids[:first_token_limit]
            ]

        character_length = len(payload.text)
        byte_length = len(payload.bytes_value)
        token_count = len(token_ids) if tokenizer_available else None
        rows.append(
            {
                "payload_name": payload.name,
                "payload_kind": payload.kind,
                "description": payload.description,
                "character_length": character_length,
                "byte_length": byte_length,
                "llm_tokenizer_available": tokenizer_available,
                "llm_token_count": token_count,
                "tokens_per_character": (
                    token_count / character_length if tokenizer_available and character_length else None
                ),
                "tokens_per_byte": (
                    token_count / byte_length if tokenizer_available and byte_length else None
                ),
                "first_token_ids_json": json.dumps(token_ids[:first_token_limit]),
                "first_token_pieces_json": json.dumps(token_pieces, ensure_ascii=False),
                "payload_text": payload.text,
            }
        )
    return rows

