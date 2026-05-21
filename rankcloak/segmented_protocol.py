"""Two-stage segmented multi-cover RankCloak pilot."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .metrics import (
    artifact_flags,
    extract_text_features,
    summarize_optional_log_probabilities,
    summarize_optional_ranks,
)
from .model_io import evaluate_context, get_last_logits, make_context_token_ids, safe_detokenize
from .plotting import (
    plot_segmented_condition_length,
    plot_segmented_condition_mean_logprob,
    plot_segmented_condition_repetition,
    plot_segmented_recovery_by_condition,
    plot_segmented_single_vs_multi_topic,
    plot_quality_filter_effect_artifacts,
    plot_quality_filter_effect_logprob,
    plot_quality_forced_vs_full_logprob,
    plot_quality_forced_vs_full_repetition,
    plot_quality_recovery_by_condition,
    plot_quality_tail_policy_logprob,
)
from .prompts import prompt_family
from .rank_codec import (
    decode_bounded_ranks_to_bytes,
    decode_hex_nibble_ranks_to_text,
    encode_bytes_to_bounded_ranks,
    encode_hex_nibbles_to_ranks,
    recover_ranks_from_generated_ids,
    rank_of_token,
    token_id_at_rank,
    token_log_probability,
)
from .reproducibility import repo_relative_path, write_manifest
from .schemas import (
    SEGMENTED_PROTOCOL_TRIAL_COLUMNS,
    SEGMENTED_QUALITY_FEATURE_COLUMNS,
    SEGMENTED_QUALITY_TRIAL_COLUMNS,
)
from .synthetic_payloads import SyntheticPayload
from .token_filters import (
    SAFE_TEXT_FILTER_V1,
    build_allowed_token_mask,
    choose_token_at_rank_with_optional_filter,
    rank_token_with_optional_filter,
)


CONTROL_CODEBOOK: Dict[str, Dict[str, object]] = {
    "C1": {
        "description": "Synthetic SHA-256 hex response using segmented raw-hex-nibble covers.",
        "payload_name": "sha256_public_test_string",
        "payload_codec": "raw_hex_nibbles",
        "segment_size": 8,
        "topic_schedule_name": "mixed_recipe_forum_car_blog",
        "natural_tail_tokens": 40,
        "decode_policy": "forced_prefix_only",
    }
}


@dataclass(frozen=True)
class SegmentedCondition:
    name: str
    segmented: bool
    prompt_names: Tuple[str, ...]
    topic_schedule_name: str
    natural_tail_tokens: int
    notes: str


SEGMENTED_PROTOCOL_CONDITIONS: Tuple[SegmentedCondition, ...] = (
    SegmentedCondition(
        name="single_long_recipe_no_tail",
        segmented=False,
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        natural_tail_tokens=0,
        notes="one forced span; no natural tail",
    ),
    SegmentedCondition(
        name="single_long_recipe_tail40",
        segmented=False,
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        natural_tail_tokens=40,
        notes="one forced span; decoder ignores greedy tail",
    ),
    SegmentedCondition(
        name="segmented_single_topic_no_tail",
        segmented=True,
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        natural_tail_tokens=0,
        notes="8-rank segments; same recipe prompt for every message",
    ),
    SegmentedCondition(
        name="segmented_single_topic_tail40",
        segmented=True,
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        natural_tail_tokens=40,
        notes="8-rank segments; same recipe prompt; decoder ignores greedy tails",
    ),
    SegmentedCondition(
        name="segmented_multi_topic_tail40",
        segmented=True,
        prompt_names=(
            "recipe_long_specific",
            "recipe_forum_exchange_specific",
            "car_buying_dialogue_specific",
            "recipe_blog",
        ),
        topic_schedule_name="mixed_recipe_forum_car_blog",
        natural_tail_tokens=40,
        notes="8-rank segments; rotating prompt schedule; decoder ignores greedy tails",
    ),
)


SEGMENTED_FEATURE_COLUMNS = [
    "source_type",
    "trial_id",
    "condition_name",
    "payload_name",
    "prompt_name",
    "prompt_family",
    "segment_index",
    "token_count",
    "character_count",
    "line_count",
    "whitespace_fraction",
    "punctuation_fraction",
    "digit_fraction",
    "alphabetic_fraction",
    "unique_token_fraction",
    "repeated_token_fraction",
    "mean_token_log_probability",
    "median_token_log_probability",
    "mean_generated_rank",
    "p95_generated_rank",
]


@dataclass(frozen=True)
class SegmentedQualityCondition:
    name: str
    prompt_names: Tuple[str, ...]
    topic_schedule_name: str
    tail_policy: str
    requested_tail_token_count: int
    token_filter_name: str
    notes: str


SEGMENTED_QUALITY_CONDITIONS: Tuple[SegmentedQualityCondition, ...] = (
    SegmentedQualityCondition(
        name="segmented_single_topic_fixed_tail40_unfiltered",
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        tail_policy="fixed_tail40",
        requested_tail_token_count=40,
        token_filter_name="none",
        notes="single recipe prompt; fixed 40-token greedy tail; no token filter",
    ),
    SegmentedQualityCondition(
        name="segmented_single_topic_sentence_tail_unfiltered",
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        tail_policy="sentence_tail_min20_max60",
        requested_tail_token_count=60,
        token_filter_name="none",
        notes="single recipe prompt; sentence-boundary greedy tail; no token filter",
    ),
    SegmentedQualityCondition(
        name="segmented_multi_topic_sentence_tail_unfiltered",
        prompt_names=(
            "recipe_long_specific",
            "recipe_forum_exchange_specific",
            "recipe_blog",
            "grocery_planning_note_specific",
            "plant_care_note_specific",
        ),
        topic_schedule_name="mixed_recipe_forum_blog_grocery_plant",
        tail_policy="sentence_tail_min20_max60",
        requested_tail_token_count=60,
        token_filter_name="none",
        notes="rotating ordinary prose prompts; sentence-boundary greedy tail; no token filter",
    ),
    SegmentedQualityCondition(
        name="segmented_single_topic_sentence_tail_filtered",
        prompt_names=("recipe_long_specific",),
        topic_schedule_name="single_recipe_long",
        tail_policy="sentence_tail_min20_max60",
        requested_tail_token_count=60,
        token_filter_name=SAFE_TEXT_FILTER_V1,
        notes="single recipe prompt; sentence-boundary greedy tail; safe-text token filter",
    ),
    SegmentedQualityCondition(
        name="segmented_multi_topic_sentence_tail_filtered",
        prompt_names=(
            "recipe_long_specific",
            "recipe_forum_exchange_specific",
            "recipe_blog",
            "grocery_planning_note_specific",
            "plant_care_note_specific",
        ),
        topic_schedule_name="mixed_recipe_forum_blog_grocery_plant",
        tail_policy="sentence_tail_min20_max60",
        requested_tail_token_count=60,
        token_filter_name=SAFE_TEXT_FILTER_V1,
        notes="rotating ordinary prose prompts; sentence-boundary greedy tail; safe-text token filter",
    ),
)


def likely_sentence_boundary(text: str) -> bool:
    """Return True when text ends at a simple sentence boundary."""

    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if stripped.endswith((".", "?", "!")):
        return True
    return stripped.endswith(('."', '?"', '!"', ".'", "?'", "!'", ".”", "?”", "!”"))


def should_stop_sentence_tail(
    tail_text: str,
    tail_token_count: int,
    minimum_tail_tokens: int = 20,
    maximum_tail_tokens: int = 60,
) -> bool:
    """Decide whether a sentence-boundary tail should stop."""

    tail_token_count = int(tail_token_count)
    if tail_token_count >= int(maximum_tail_tokens):
        return True
    if tail_token_count < int(minimum_tail_tokens):
        return False
    return likely_sentence_boundary(tail_text)


def prompt_for_quality_segment(condition: SegmentedQualityCondition, segment_index: int) -> str:
    return condition.prompt_names[segment_index % len(condition.prompt_names)]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def ordered_frame(rows: Sequence[dict], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    extra_columns = [column for column in frame.columns if column not in columns]
    return frame.reindex(columns=list(columns) + extra_columns)


def chunk_rank_sequence(ranks: Sequence[int], segment_size: int) -> List[List[int]]:
    """Split a rank sequence into non-empty chunks of at most segment_size."""

    segment_size = int(segment_size)
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    return [
        list(map(int, ranks[index : index + segment_size]))
        for index in range(0, len(ranks), segment_size)
    ]


def flatten_rank_chunks(chunks: Sequence[Sequence[int]]) -> List[int]:
    """Reconstruct a rank sequence from chunks."""

    flattened: List[int] = []
    for chunk in chunks:
        flattened.extend(map(int, chunk))
    return flattened


def prompt_for_segment(condition: SegmentedCondition, segment_index: int) -> str:
    prompt_names = condition.prompt_names
    return prompt_names[segment_index % len(prompt_names)]


def generate_forced_prefix_with_natural_tail(
    model: Any,
    context_token_ids: Sequence[int],
    forced_ranks: Sequence[int],
    natural_tail_tokens: int,
) -> Dict[str, object]:
    """Generate forced-rank prefix tokens followed by optional greedy tail tokens."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    forced_token_ids: List[int] = []
    generated_token_ids: List[int] = []
    generated_ranks: List[int] = []
    token_log_probabilities: List[float] = []

    for rank in forced_ranks:
        logits = get_last_logits(model)
        token_id = token_id_at_rank(logits, int(rank))
        forced_token_ids.append(token_id)
        generated_token_ids.append(token_id)
        generated_ranks.append(rank_of_token(logits, token_id))
        token_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    for _ in range(int(natural_tail_tokens)):
        logits = get_last_logits(model)
        token_id = token_id_at_rank(logits, 1)
        generated_token_ids.append(token_id)
        generated_ranks.append(rank_of_token(logits, token_id))
        token_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    return {
        "forced_token_ids": forced_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_ranks": generated_ranks,
        "generated_text": safe_detokenize(model, generated_token_ids),
        "forced_text": safe_detokenize(model, forced_token_ids),
        "token_log_probabilities": token_log_probabilities,
    }


def encode_control_code(control_code: str, alphabet_size: int = 16) -> Dict[str, object]:
    """Encode a compact synthetic control code with fixed-radix byte ranks."""

    return encode_bytes_to_bounded_ranks(control_code.encode("utf-8"), alphabet_size)


def decode_control_code(ranks: Sequence[int], metadata: Dict[str, int]) -> str:
    """Decode a recovered compact synthetic control code."""

    return decode_bounded_ranks_to_bytes(ranks, metadata).decode("utf-8")


def recover_ranks_from_generated_ids_with_optional_filter(
    model: Any,
    context_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Recover ranks with the same optional token filter used by generation."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    recovered_ranks: List[int] = []
    token_log_probabilities: List[float] = []
    for token_id in map(int, generated_token_ids):
        logits = get_last_logits(model)
        recovered_ranks.append(
            rank_token_with_optional_filter(logits, token_id, allowed_token_mask)
        )
        token_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    return {
        "ranks": recovered_ranks,
        "token_log_probabilities": token_log_probabilities,
        "success": True,
    }


def generate_quality_message(
    model: Any,
    context_token_ids: Sequence[int],
    forced_ranks: Sequence[int],
    tail_policy: str,
    requested_tail_token_count: int,
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Generate a forced prefix plus deterministic natural tail for quality-control runs."""

    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    forced_token_ids: List[int] = []
    forced_rank_values: List[int] = []
    forced_log_probabilities: List[float] = []
    tail_token_ids: List[int] = []
    tail_rank_values: List[int] = []
    tail_log_probabilities: List[float] = []

    for rank in forced_ranks:
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, int(rank), allowed_token_mask)
        forced_token_ids.append(token_id)
        forced_rank_values.append(
            rank_token_with_optional_filter(logits, token_id, allowed_token_mask)
        )
        forced_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    if tail_policy == "fixed_tail40":
        maximum_tail_tokens = int(requested_tail_token_count)
        minimum_tail_tokens = maximum_tail_tokens
    elif tail_policy == "sentence_tail_min20_max60":
        maximum_tail_tokens = 60
        minimum_tail_tokens = 20
    else:
        raise ValueError("Unknown tail policy: {}".format(tail_policy))

    for _ in range(maximum_tail_tokens):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, 1, allowed_token_mask)
        tail_token_ids.append(token_id)
        tail_rank_values.append(
            rank_token_with_optional_filter(logits, token_id, allowed_token_mask)
        )
        tail_log_probabilities.append(token_log_probability(logits, token_id))
        model.eval([token_id])
        if tail_policy == "sentence_tail_min20_max60":
            tail_text = safe_detokenize(model, tail_token_ids)
            if should_stop_sentence_tail(
                tail_text,
                len(tail_token_ids),
                minimum_tail_tokens=minimum_tail_tokens,
                maximum_tail_tokens=maximum_tail_tokens,
            ):
                break

    full_token_ids = forced_token_ids + tail_token_ids
    full_rank_values = forced_rank_values + tail_rank_values
    full_log_probabilities = forced_log_probabilities + tail_log_probabilities
    return {
        "forced_token_ids": forced_token_ids,
        "tail_token_ids": tail_token_ids,
        "full_token_ids": full_token_ids,
        "forced_rank_values": forced_rank_values,
        "tail_rank_values": tail_rank_values,
        "full_rank_values": full_rank_values,
        "forced_log_probabilities": forced_log_probabilities,
        "tail_log_probabilities": tail_log_probabilities,
        "full_log_probabilities": full_log_probabilities,
        "forced_prefix_text": safe_detokenize(model, forced_token_ids),
        "tail_text": safe_detokenize(model, tail_token_ids),
        "full_message_text": safe_detokenize(model, full_token_ids),
    }


def metrics_for_text_part(
    text: str,
    token_ids: Sequence[int],
    ranks: Sequence[int],
    token_log_probabilities: Sequence[float],
) -> dict:
    """Compute metric row for one text part."""

    return extract_text_features(
        text,
        token_ids=token_ids,
        ranks=ranks,
        token_log_probabilities=token_log_probabilities,
    )


def run_control_request_trial(
    model: Any,
    cover_prompts: Dict[str, str],
) -> Tuple[dict, Dict[str, object]]:
    control_code = "C1"
    control_prompt_name = "recipe_forum_exchange_specific"
    control_codec_name = "ascii_bytes_fixed_radix_b16"
    encoded = encode_control_code(control_code, alphabet_size=16)
    context_token_ids = make_context_token_ids(model, cover_prompts[control_prompt_name])

    generation_started_at = time.perf_counter()
    generated = generate_forced_prefix_with_natural_tail(
        model=model,
        context_token_ids=context_token_ids,
        forced_ranks=encoded["ranks"],
        natural_tail_tokens=0,
    )
    generation_seconds = time.perf_counter() - generation_started_at

    recovery_started_at = time.perf_counter()
    forced_token_count = len(encoded["ranks"])
    recovered = recover_ranks_from_generated_ids(
        model,
        context_token_ids,
        generated["generated_token_ids"][:forced_token_count],
    )
    recovered_control_code = decode_control_code(recovered["ranks"], encoded["metadata"])
    recovery_seconds = time.perf_counter() - recovery_started_at
    exact_recovery = recovered_control_code == control_code

    row = {
        "control_code": control_code,
        "control_prompt_name": control_prompt_name,
        "control_codec_name": control_codec_name,
        "control_rank_count": len(encoded["ranks"]),
        "control_generated_token_count": len(generated["generated_token_ids"]),
        "control_generated_character_count": len(generated["generated_text"]),
        "control_exact_recovery": exact_recovery,
        "control_generation_seconds": generation_seconds,
        "control_recovery_seconds": recovery_seconds,
        "generated_control_text": generated["generated_text"],
        "recovered_control_code": recovered_control_code,
        "notes": "synthetic compact control code; not a key exchange or operational command",
    }
    return row, CONTROL_CODEBOOK[control_code]


def encode_payload_as_raw_hex_nibbles(payload: SyntheticPayload, payload_text_limit: Optional[int]) -> Dict[str, object]:
    payload_text = payload.text
    if payload_text_limit is not None:
        payload_text = payload_text[: int(payload_text_limit)]
    encoded = encode_hex_nibbles_to_ranks(payload_text)
    return {
        "payload_text": payload_text.lower(),
        "ranks": encoded["ranks"],
        "metadata": encoded["metadata"],
    }


def run_response_trial(
    trial_id: str,
    payload: SyntheticPayload,
    condition: SegmentedCondition,
    cover_prompts: Dict[str, str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    segment_size: int,
    payload_text_limit: Optional[int],
) -> Tuple[dict, List[dict], List[dict]]:
    encoded_payload = encode_payload_as_raw_hex_nibbles(payload, payload_text_limit)
    ranks = encoded_payload["ranks"]
    chunks = (
        chunk_rank_sequence(ranks, segment_size)
        if condition.segmented
        else [list(map(int, ranks))]
    )
    recovered_chunks: List[List[int]] = []
    message_rows: List[dict] = []
    feature_rows: List[dict] = []
    all_token_log_probabilities: List[float] = []
    all_generated_ranks: List[int] = []
    generation_seconds_total = 0.0
    recovery_seconds_total = 0.0

    for segment_index_zero, chunk in enumerate(chunks):
        prompt_name = prompt_for_segment(condition, segment_index_zero)
        context_token_ids = make_context_token_ids(model, cover_prompts[prompt_name])
        generation_started_at = time.perf_counter()
        generated = generate_forced_prefix_with_natural_tail(
            model=model,
            context_token_ids=context_token_ids,
            forced_ranks=chunk,
            natural_tail_tokens=condition.natural_tail_tokens,
        )
        generation_seconds = time.perf_counter() - generation_started_at
        generation_seconds_total += generation_seconds

        forced_token_count = len(chunk)
        recovery_started_at = time.perf_counter()
        recovered = recover_ranks_from_generated_ids(
            model,
            context_token_ids,
            generated["generated_token_ids"][:forced_token_count],
        )
        recovery_seconds = time.perf_counter() - recovery_started_at
        recovery_seconds_total += recovery_seconds

        recovered_chunk = list(map(int, recovered["ranks"]))
        recovered_chunks.append(recovered_chunk)
        exact_segment_recovery = recovered_chunk == list(map(int, chunk))
        token_log_probabilities = list(map(float, generated["token_log_probabilities"]))
        generated_ranks = list(map(int, generated["generated_ranks"]))
        all_token_log_probabilities.extend(token_log_probabilities)
        all_generated_ranks.extend(generated_ranks)
        feature_row = extract_text_features(
            generated["generated_text"],
            token_ids=generated["generated_token_ids"],
            ranks=generated_ranks,
            token_log_probabilities=token_log_probabilities,
        )
        feature_row.update(
            {
                "source_type": "segmented_protocol_message",
                "trial_id": trial_id,
                "condition_name": condition.name,
                "payload_name": payload.name,
                "prompt_name": prompt_name,
                "prompt_family": prompt_family(prompt_name),
                "segment_index": segment_index_zero + 1,
            }
        )
        feature_rows.append(feature_row)

        message_rows.append(
            {
                "trial_id": trial_id,
                "payload_name": payload.name,
                "condition_name": condition.name,
                "segment_index": segment_index_zero + 1,
                "segment_count": len(chunks),
                "prompt_name": prompt_name,
                "prompt_family": prompt_family(prompt_name),
                "forced_rank_count": len(chunk),
                "natural_tail_tokens": condition.natural_tail_tokens,
                "generated_token_count": len(generated["generated_token_ids"]),
                "generated_character_count": len(generated["generated_text"]),
                "exact_segment_recovery": exact_segment_recovery,
                "mean_token_log_probability": feature_row["mean_token_log_probability"],
                "median_token_log_probability": feature_row["median_token_log_probability"],
                "mean_generated_rank": feature_row["mean_generated_rank"],
                "p95_generated_rank": feature_row["p95_generated_rank"],
                "repeated_token_fraction": feature_row["repeated_token_fraction"],
                "punctuation_fraction": feature_row["punctuation_fraction"],
                "alphabetic_fraction": feature_row["alphabetic_fraction"],
                "generated_text": generated["generated_text"],
                "notes": "decode policy: forced prefix only; natural tail ignored",
            }
        )

    recovered_ranks = flatten_rank_chunks(recovered_chunks)
    recovered_payload_text = decode_hex_nibble_ranks_to_text(
        recovered_ranks,
        encoded_payload["metadata"],
    )
    exact_recovery = recovered_payload_text == encoded_payload["payload_text"]
    log_probability_summary = summarize_optional_log_probabilities(all_token_log_probabilities)
    rank_summary = summarize_optional_ranks(all_generated_ranks)
    repeated_values = [
        row["repeated_token_fraction"]
        for row in feature_rows
        if row.get("repeated_token_fraction") is not None
    ]
    punctuation_values = [
        row["punctuation_fraction"]
        for row in feature_rows
        if row.get("punctuation_fraction") is not None
    ]
    alphabetic_values = [
        row["alphabetic_fraction"]
        for row in feature_rows
        if row.get("alphabetic_fraction") is not None
    ]
    trial_row = {
        "trial_id": trial_id,
        "payload_name": payload.name,
        "payload_kind": payload.kind,
        "payload_text_character_length": len(encoded_payload["payload_text"]),
        "payload_text_byte_length": len(encoded_payload["payload_text"].encode("utf-8")),
        "payload_codec_name": "raw_hex_nibbles",
        "condition_name": condition.name,
        "segment_size": int(segment_size),
        "segment_count": len(chunks),
        "message_count": len(chunks),
        "topic_schedule_name": condition.topic_schedule_name,
        "natural_tail_tokens_per_message": condition.natural_tail_tokens,
        "total_forced_rank_count": len(ranks),
        "total_generated_token_count": sum(row["generated_token_count"] for row in message_rows),
        "total_generated_character_count": sum(row["generated_character_count"] for row in message_rows),
        "exact_recovery": exact_recovery,
        "generation_seconds": generation_seconds_total,
        "recovery_seconds": recovery_seconds_total,
        "mean_token_log_probability": log_probability_summary["mean_token_log_probability"],
        "median_token_log_probability": log_probability_summary["median_token_log_probability"],
        "mean_generated_rank": rank_summary["mean_generated_rank"],
        "p95_generated_rank": rank_summary["p95_generated_rank"],
        "repeated_token_fraction_mean": (
            sum(repeated_values) / len(repeated_values) if repeated_values else None
        ),
        "punctuation_fraction_mean": (
            sum(punctuation_values) / len(punctuation_values) if punctuation_values else None
        ),
        "alphabetic_fraction_mean": (
            sum(alphabetic_values) / len(alphabetic_values) if alphabetic_values else None
        ),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": model_path_relative,
        "notes": condition.notes,
    }
    return trial_row, message_rows, feature_rows


def segmented_quality_notes(message_row: dict) -> str:
    text = message_row.get("generated_text") or ""
    notes = []
    if not text.strip():
        notes.append("empty output")
    if "http" in text or "```" in text or text.count("...") >= 2:
        notes.append("formatting, link-like, or ellipsis artifact")
    if "[" in text or "]" in text or "\\" in text or "{" in text or "}" in text:
        notes.append("placeholder-like or markup artifact")
    repeated = message_row.get("repeated_token_fraction")
    punctuation = message_row.get("punctuation_fraction")
    alphabetic = message_row.get("alphabetic_fraction")
    if repeated is not None and repeated > 0.35:
        notes.append("repetitive")
    if punctuation is not None and punctuation > 0.12:
        notes.append("punctuation-heavy")
    if alphabetic is not None and alphabetic > 0.68 and not notes:
        notes.append("mostly prose-like by lightweight features")
    if not notes:
        notes.append("mixed; inspect manually")
    return "; ".join(notes)


def markdown_text_block(text: str, limit: int = 900) -> str:
    return (text or "").strip()[:limit].replace("```", "` ` `")


def write_segmented_protocol_comparison(
    output_dir: Path,
    control_row: dict,
    trial_frame: pd.DataFrame,
    message_rows: Sequence[dict],
) -> Path:
    path = output_dir / "SEGMENTED_PROTOCOL_COMPARISON.md"
    lines = [
        "# Two-Stage Segmented Multi-Cover RankCloak Pilot",
        "",
        "This pilot tests whether splitting a synthetic payload across several short cover "
        "messages reduces cover-text drift compared with one longer forced-rank cover message.",
        "",
        "The simulated parties already share `K_common`: model file, tokenizer, quantization, "
        "rank ordering, payload codec, prompt templates, control prompt, segment-size rule, "
        "topic schedule, and forced-prefix decode policy. This is not key exchange.",
        "",
        "## Control Request",
        "",
        "- control_code: `{}`".format(control_row.get("control_code")),
        "- control_prompt_name: `{}`".format(control_row.get("control_prompt_name")),
        "- control_codec_name: `{}`".format(control_row.get("control_codec_name")),
        "- exact_recovery: `{}`".format(control_row.get("control_exact_recovery")),
        "- generated_token_count: `{}`".format(control_row.get("control_generated_token_count")),
        "",
        "## Conditions",
        "",
        "| Condition | Messages | Tail tokens/message | Recovery rate | Mean logprob | Mean repetition | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not trial_frame.empty:
        grouped = trial_frame.groupby("condition_name", as_index=False).agg(
            {
                "message_count": "mean",
                "natural_tail_tokens_per_message": "mean",
                "exact_recovery": "mean",
                "mean_token_log_probability": "mean",
                "repeated_token_fraction_mean": "mean",
                "notes": "first",
            }
        )
        for _, row in grouped.iterrows():
            lines.append(
                "| `{}` | {:.1f} | {:.0f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                    row["condition_name"],
                    row["message_count"],
                    row["natural_tail_tokens_per_message"],
                    row["exact_recovery"],
                    row["mean_token_log_probability"],
                    row["repeated_token_fraction_mean"],
                    row["notes"],
                )
            )
    lines.extend(
        [
            "",
            "## Recovery Summary",
            "",
            "- response_trial_count: `{}`".format(len(trial_frame)),
            "- response_recovery_pass_count: `{}`".format(
                int(trial_frame["exact_recovery"].astype(bool).sum()) if not trial_frame.empty else 0
            ),
            "- response_recovery_fail_count: `{}`".format(
                int((~trial_frame["exact_recovery"].astype(bool)).sum()) if not trial_frame.empty else 0
            ),
            "",
            "## Comparison Notes",
            "",
            "- Single-long conditions keep each payload in one forced span.",
            "- Segmented conditions restart from a clean prompt context for each short rank chunk.",
            "- Tail40 conditions append greedy natural text that is not decoded.",
            "- Multi-topic segmentation rotates recipe, forum, car-buying, and recipe-blog prompts.",
            "",
            "## Generated Examples",
            "",
        ]
    )
    examples_by_condition: Dict[str, List[dict]] = {}
    for message_row in message_rows:
        condition_name = str(message_row.get("condition_name"))
        examples_by_condition.setdefault(condition_name, [])
        if len(examples_by_condition[condition_name]) < 2:
            examples_by_condition[condition_name].append(message_row)

    for condition in SEGMENTED_PROTOCOL_CONDITIONS:
        lines.extend(["### `{}`".format(condition.name), ""])
        for message_row in examples_by_condition.get(condition.name, []):
            lines.extend(
                [
                    "- trial_id: `{}`".format(message_row.get("trial_id")),
                    "- payload_name: `{}`".format(message_row.get("payload_name")),
                    "- segment_index: `{}`".format(message_row.get("segment_index")),
                    "- prompt_name: `{}`".format(message_row.get("prompt_name")),
                    "- exact_segment_recovery: `{}`".format(
                        message_row.get("exact_segment_recovery")
                    ),
                    "- generated_token_count: `{}`".format(
                        message_row.get("generated_token_count")
                    ),
                    "- mean_token_log_probability: `{}`".format(
                        message_row.get("mean_token_log_probability")
                    ),
                    "- notes: {}".format(segmented_quality_notes(message_row)),
                    "",
                    "```text",
                    markdown_text_block(message_row.get("generated_text") or ""),
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Limitations",
            "",
            "- All payloads are deterministic synthetic examples.",
            "- This is not encryption, key exchange, authentication, signing, or cryptographic security.",
            "- Exact recovery requires the same model, tokenizer, quantization, rank ordering, prompts, and unmodified text.",
            "- Natural tails are ignored by the decoder and do not carry payload ranks in this pilot.",
            "- The quality notes are lightweight heuristics and manual-inspection aids, not detector AUC.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_segmented_summary_markdown(
    output_dir: Path,
    summary: dict,
) -> Path:
    path = output_dir / "SUMMARY.md"
    text = """# RankCloak Segmented Protocol Pilot Summary

- Profile: {profile}
- Model status: {model_status}
- Model: {model_repo_id} / {model_filename}
- Control request exact recovery: {control_recovery}
- Response recovery: {response_passes} pass, {response_failures} fail
- Response trial count: {response_trial_count}

## Main Quality Observation

Segmentation restarts each cover message from a clean prompt context and allows optional greedy tails, so this pilot compares cover drift against one longer forced span. The first-pass quality result should be read from `SEGMENTED_PROTOCOL_COMPARISON.md` and `cover_text_features.csv`; exact recovery alone is not a cover-quality score.

## Important Limitations

- This is not encryption, key exchange, authentication, signing, or cryptographic security.
- The control code is synthetic and maps to a pre-agreed local experiment configuration.
- The decoder recovers only the forced prefix and ignores natural tails.
- All examples are deterministic synthetic payloads.
- Exact-copy conditions are required.

## Next Recommended Experiment

If this pilot suggests lower drift, run a follow-up using raw-hex-nibble payload coding for more hex-like payloads and a small manual quality rubric.
""".format(
        profile=summary["profile"],
        model_status="loaded" if summary["model_loaded"] else "not loaded",
        model_repo_id=summary["model_repo_id"],
        model_filename=summary["model_filename"],
        control_recovery=summary["control_request_exact_recovery"],
        response_passes=summary["response_recovery_pass_count"],
        response_failures=summary["response_recovery_fail_count"],
        response_trial_count=summary["response_trial_count"],
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_segmented_protocol_pilot(
    output_dir: Path,
    project_root: Path,
    model: Any,
    model_path: Optional[Path],
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    payloads: Sequence[SyntheticPayload],
    cover_prompts: Dict[str, str],
    command_line_args: Sequence[str],
    model_loaded: bool,
    model_status: str,
    model_error: Optional[str],
    payload_text_limit: Optional[int] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    segment_size = int(CONTROL_CODEBOOK["C1"]["segment_size"])

    if model is None:
        control_rows: List[dict] = []
        trial_frame = ordered_frame([], SEGMENTED_PROTOCOL_TRIAL_COLUMNS)
        message_rows: List[dict] = []
        feature_frame = ordered_frame([], SEGMENTED_FEATURE_COLUMNS)
        control_exact_recovery = False
    else:
        control_row, _control_config = run_control_request_trial(model, cover_prompts)
        control_rows = [control_row]
        control_exact_recovery = bool(control_row["control_exact_recovery"])
        trial_rows = []
        message_rows = []
        feature_rows = []
        trial_counter = 1
        for payload in payloads:
            for condition in SEGMENTED_PROTOCOL_CONDITIONS:
                trial_id = "segmented_{:03d}".format(trial_counter)
                trial_counter += 1
                trial_row, trial_message_rows, trial_feature_rows = run_response_trial(
                    trial_id=trial_id,
                    payload=payload,
                    condition=condition,
                    cover_prompts=cover_prompts,
                    model=model,
                    model_repo_id=model_repo_id,
                    model_filename=model_filename,
                    model_path_relative=model_path_relative,
                    segment_size=segment_size,
                    payload_text_limit=payload_text_limit,
                )
                trial_rows.append(trial_row)
                message_rows.extend(trial_message_rows)
                feature_rows.extend(trial_feature_rows)
        trial_frame = ordered_frame(trial_rows, SEGMENTED_PROTOCOL_TRIAL_COLUMNS)
        feature_frame = ordered_frame(feature_rows, SEGMENTED_FEATURE_COLUMNS)

    control_path = output_dir / "control_request_trial.jsonl"
    write_jsonl(control_path, control_rows)
    trial_path = output_dir / "segmented_protocol_trials.csv"
    trial_frame.to_csv(trial_path, index=False)
    messages_path = output_dir / "segmented_protocol_messages.jsonl"
    write_jsonl(messages_path, message_rows)
    feature_path = output_dir / "cover_text_features.csv"
    feature_frame.to_csv(feature_path, index=False)

    comparison_path = write_segmented_protocol_comparison(
        output_dir=output_dir,
        control_row=control_rows[0] if control_rows else {},
        trial_frame=trial_frame,
        message_rows=message_rows,
    )
    figure_paths = [
        plot_segmented_condition_mean_logprob(
            trial_frame, figures_dir / "segmented_condition_mean_logprob.png"
        ),
        plot_segmented_condition_repetition(
            trial_frame, figures_dir / "segmented_condition_repetition.png"
        ),
        plot_segmented_condition_length(
            trial_frame, figures_dir / "segmented_condition_length.png"
        ),
        plot_segmented_recovery_by_condition(
            trial_frame, figures_dir / "segmented_recovery_by_condition.png"
        ),
        plot_segmented_single_vs_multi_topic(
            trial_frame, figures_dir / "segmented_single_vs_multi_topic.png"
        ),
    ]
    manifest_path = output_dir / "MANIFEST.json"
    write_manifest(
        output_path=manifest_path,
        project_root=project_root,
        profile="segmented-protocol-pilot",
        output_dir=output_dir,
        command_line_args=command_line_args,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
    )

    if not trial_frame.empty and "exact_recovery" in trial_frame:
        exact_values = trial_frame["exact_recovery"].astype(bool)
        response_pass_count = int(exact_values.sum())
        response_fail_count = int((~exact_values).sum())
    else:
        response_pass_count = 0
        response_fail_count = 0
    generated_files = [
        control_path,
        trial_path,
        messages_path,
        comparison_path,
        feature_path,
        manifest_path,
        *figure_paths,
        output_dir / "summary.json",
        output_dir / "SUMMARY.md",
    ]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": "segmented-protocol-pilot",
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_loaded": bool(model_loaded),
        "control_request_exact_recovery": bool(control_exact_recovery),
        "response_trial_count": int(len(trial_frame)),
        "response_recovery_pass_count": response_pass_count,
        "response_recovery_fail_count": response_fail_count,
        "payloads": [payload.name for payload in payloads],
        "conditions": [condition.name for condition in SEGMENTED_PROTOCOL_CONDITIONS],
        "segment_size": segment_size,
        "natural_tail_settings": sorted(
            set(condition.natural_tail_tokens for condition in SEGMENTED_PROTOCOL_CONDITIONS)
        ),
        "generated_result_files": [
            repo_relative_path(path, project_root) for path in generated_files
        ],
        "important_notes": [
            "All payloads are deterministic synthetic examples.",
            "Control code C1 is a compact synthetic codebook entry, not a key exchange.",
            "Decoder recovers only forced prefixes and ignores natural greedy tails.",
            "Model status: {}".format(model_status),
            "Model error: {}".format(model_error) if model_error else "Model loaded or was not required.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_segmented_summary_markdown(output_dir, summary)
    return summary


def mean_of_present(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def feature_row_for_quality_part(
    source_type: str,
    trial_id: str,
    condition_name: str,
    payload_name: Optional[str],
    prompt_name: str,
    segment_index: int,
    token_filter_name: str,
    tail_policy: str,
    text: str,
    token_ids: Sequence[int],
    ranks: Sequence[int],
    token_log_probabilities: Sequence[float],
) -> dict:
    row = metrics_for_text_part(text, token_ids, ranks, token_log_probabilities)
    row.update(
        {
            "source_type": source_type,
            "trial_id": trial_id,
            "condition_name": condition_name,
            "payload_name": payload_name,
            "prompt_name": prompt_name,
            "prompt_family": prompt_family(prompt_name),
            "segment_index": segment_index,
            "token_filter_name": token_filter_name,
            "tail_policy": tail_policy,
        }
    )
    return row


def run_quality_control_request_trial(
    condition: SegmentedQualityCondition,
    model: Any,
    cover_prompts: Dict[str, str],
    allowed_token_mask: Optional[Sequence[bool]],
) -> Tuple[dict, List[dict]]:
    control_code = "C1"
    control_prompt_name = "recipe_forum_exchange_specific"
    control_codec_name = "ascii_bytes_fixed_radix_b16"
    encoded = encode_control_code(control_code, alphabet_size=16)
    context_token_ids = make_context_token_ids(model, cover_prompts[control_prompt_name])

    generation_started_at = time.perf_counter()
    generated = generate_quality_message(
        model=model,
        context_token_ids=context_token_ids,
        forced_ranks=encoded["ranks"],
        tail_policy=condition.tail_policy,
        requested_tail_token_count=condition.requested_tail_token_count,
        allowed_token_mask=allowed_token_mask,
    )
    generation_seconds = time.perf_counter() - generation_started_at

    recovery_started_at = time.perf_counter()
    recovered = recover_ranks_from_generated_ids_with_optional_filter(
        model,
        context_token_ids,
        generated["forced_token_ids"],
        allowed_token_mask=allowed_token_mask,
    )
    recovered_control_code = decode_control_code(recovered["ranks"], encoded["metadata"])
    recovery_seconds = time.perf_counter() - recovery_started_at
    exact_recovery = recovered_control_code == control_code

    forced_features = feature_row_for_quality_part(
        source_type="control_forced_prefix",
        trial_id="control_{}".format(condition.name),
        condition_name=condition.name,
        payload_name=None,
        prompt_name=control_prompt_name,
        segment_index=0,
        token_filter_name=condition.token_filter_name,
        tail_policy=condition.tail_policy,
        text=generated["forced_prefix_text"],
        token_ids=generated["forced_token_ids"],
        ranks=generated["forced_rank_values"],
        token_log_probabilities=generated["forced_log_probabilities"],
    )
    full_features = feature_row_for_quality_part(
        source_type="control_full_message",
        trial_id="control_{}".format(condition.name),
        condition_name=condition.name,
        payload_name=None,
        prompt_name=control_prompt_name,
        segment_index=0,
        token_filter_name=condition.token_filter_name,
        tail_policy=condition.tail_policy,
        text=generated["full_message_text"],
        token_ids=generated["full_token_ids"],
        ranks=generated["full_rank_values"],
        token_log_probabilities=generated["full_log_probabilities"],
    )

    row = {
        "condition_name": condition.name,
        "control_code": control_code,
        "control_prompt_name": control_prompt_name,
        "control_codec_name": control_codec_name,
        "token_filter_name": condition.token_filter_name,
        "tail_policy": condition.tail_policy,
        "requested_tail_token_count": condition.requested_tail_token_count,
        "actual_tail_token_count": len(generated["tail_token_ids"]),
        "control_rank_count": len(encoded["ranks"]),
        "forced_prefix_token_count": len(generated["forced_token_ids"]),
        "full_message_token_count": len(generated["full_token_ids"]),
        "full_message_character_count": len(generated["full_message_text"]),
        "control_exact_recovery": exact_recovery,
        "control_generation_seconds": generation_seconds,
        "control_recovery_seconds": recovery_seconds,
        "forced_prefix_text": generated["forced_prefix_text"],
        "tail_text": generated["tail_text"],
        "full_message_text": generated["full_message_text"],
        "recovered_control_code": recovered_control_code,
        "notes": "synthetic compact control label; forced prefix decoded and natural tail ignored",
    }
    return row, [forced_features, full_features]


def run_quality_response_trial(
    trial_id: str,
    payload: SyntheticPayload,
    condition: SegmentedQualityCondition,
    cover_prompts: Dict[str, str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    segment_size: int,
    payload_text_limit: Optional[int],
    allowed_token_mask: Optional[Sequence[bool]],
) -> Tuple[dict, List[dict], List[dict]]:
    encoded_payload = encode_payload_as_raw_hex_nibbles(payload, payload_text_limit)
    chunks = chunk_rank_sequence(encoded_payload["ranks"], segment_size)
    recovered_chunks: List[List[int]] = []
    message_rows: List[dict] = []
    feature_rows: List[dict] = []
    generation_seconds_total = 0.0
    recovery_seconds_total = 0.0

    for segment_index_zero, chunk in enumerate(chunks):
        prompt_name = prompt_for_quality_segment(condition, segment_index_zero)
        context_token_ids = make_context_token_ids(model, cover_prompts[prompt_name])
        generation_started_at = time.perf_counter()
        generated = generate_quality_message(
            model=model,
            context_token_ids=context_token_ids,
            forced_ranks=chunk,
            tail_policy=condition.tail_policy,
            requested_tail_token_count=condition.requested_tail_token_count,
            allowed_token_mask=allowed_token_mask,
        )
        generation_seconds_total += time.perf_counter() - generation_started_at

        recovery_error = None
        recovery_started_at = time.perf_counter()
        try:
            recovered = recover_ranks_from_generated_ids_with_optional_filter(
                model,
                context_token_ids,
                generated["forced_token_ids"],
                allowed_token_mask=allowed_token_mask,
            )
            recovered_chunk = list(map(int, recovered["ranks"]))
        except Exception as exc:
            recovered_chunk = []
            recovery_error = str(exc)
        recovery_seconds_total += time.perf_counter() - recovery_started_at
        recovered_chunks.append(recovered_chunk)
        exact_segment_recovery = recovered_chunk == list(map(int, chunk))

        forced_features = feature_row_for_quality_part(
            source_type="segmented_forced_prefix",
            trial_id=trial_id,
            condition_name=condition.name,
            payload_name=payload.name,
            prompt_name=prompt_name,
            segment_index=segment_index_zero + 1,
            token_filter_name=condition.token_filter_name,
            tail_policy=condition.tail_policy,
            text=generated["forced_prefix_text"],
            token_ids=generated["forced_token_ids"],
            ranks=generated["forced_rank_values"],
            token_log_probabilities=generated["forced_log_probabilities"],
        )
        full_features = feature_row_for_quality_part(
            source_type="segmented_full_message",
            trial_id=trial_id,
            condition_name=condition.name,
            payload_name=payload.name,
            prompt_name=prompt_name,
            segment_index=segment_index_zero + 1,
            token_filter_name=condition.token_filter_name,
            tail_policy=condition.tail_policy,
            text=generated["full_message_text"],
            token_ids=generated["full_token_ids"],
            ranks=generated["full_rank_values"],
            token_log_probabilities=generated["full_log_probabilities"],
        )
        feature_rows.extend([forced_features, full_features])

        message_rows.append(
            {
                "trial_id": trial_id,
                "payload_name": payload.name,
                "condition_name": condition.name,
                "segment_index": segment_index_zero + 1,
                "segment_count": len(chunks),
                "prompt_name": prompt_name,
                "prompt_family": prompt_family(prompt_name),
                "token_filter_name": condition.token_filter_name,
                "forced_rank_count": len(chunk),
                "forced_prefix_token_count": len(generated["forced_token_ids"]),
                "forced_prefix_character_count": len(generated["forced_prefix_text"]),
                "forced_prefix_text": generated["forced_prefix_text"],
                "forced_prefix_mean_token_log_probability": forced_features["mean_token_log_probability"],
                "forced_prefix_median_token_log_probability": forced_features["median_token_log_probability"],
                "forced_prefix_mean_generated_rank": forced_features["mean_generated_rank"],
                "forced_prefix_p95_generated_rank": forced_features["p95_generated_rank"],
                "forced_prefix_repeated_token_fraction": forced_features["repeated_token_fraction"],
                "forced_prefix_punctuation_fraction": forced_features["punctuation_fraction"],
                "forced_prefix_alphabetic_fraction": forced_features["alphabetic_fraction"],
                "tail_policy": condition.tail_policy,
                "requested_tail_token_count": condition.requested_tail_token_count,
                "actual_tail_token_count": len(generated["tail_token_ids"]),
                "tail_text": generated["tail_text"],
                "full_message_token_count": len(generated["full_token_ids"]),
                "full_message_character_count": len(generated["full_message_text"]),
                "full_message_text": generated["full_message_text"],
                "full_message_mean_token_log_probability": full_features["mean_token_log_probability"],
                "full_message_median_token_log_probability": full_features["median_token_log_probability"],
                "full_message_mean_generated_rank": full_features["mean_generated_rank"],
                "full_message_p95_generated_rank": full_features["p95_generated_rank"],
                "full_message_repeated_token_fraction": full_features["repeated_token_fraction"],
                "full_message_punctuation_fraction": full_features["punctuation_fraction"],
                "full_message_alphabetic_fraction": full_features["alphabetic_fraction"],
                "generated_text": generated["full_message_text"],
                "exact_segment_recovery": exact_segment_recovery,
                "notes": (
                    "decode policy: forced prefix only; natural tail ignored"
                    if recovery_error is None
                    else "recovery error: {}".format(recovery_error)
                ),
            }
        )

    recovered_ranks = flatten_rank_chunks(recovered_chunks)
    try:
        recovered_payload_text = decode_hex_nibble_ranks_to_text(
            recovered_ranks,
            encoded_payload["metadata"],
        )
        exact_recovery = recovered_payload_text == encoded_payload["payload_text"]
    except Exception:
        exact_recovery = False

    forced_feature_rows = [
        row for row in feature_rows if row["source_type"] == "segmented_forced_prefix"
    ]
    full_feature_rows = [
        row for row in feature_rows if row["source_type"] == "segmented_full_message"
    ]
    trial_row = {
        "trial_id": trial_id,
        "payload_name": payload.name,
        "payload_kind": payload.kind,
        "payload_text_character_length": len(encoded_payload["payload_text"]),
        "payload_text_byte_length": len(encoded_payload["payload_text"].encode("utf-8")),
        "payload_codec_name": "raw_hex_nibbles",
        "condition_name": condition.name,
        "segment_size": int(segment_size),
        "segment_count": len(chunks),
        "message_count": len(chunks),
        "topic_schedule_name": condition.topic_schedule_name,
        "tail_policy": condition.tail_policy,
        "requested_tail_token_count": condition.requested_tail_token_count,
        "actual_tail_token_count_mean": mean_of_present(
            [row["actual_tail_token_count"] for row in message_rows]
        ),
        "token_filter_name": condition.token_filter_name,
        "total_forced_rank_count": len(encoded_payload["ranks"]),
        "total_forced_prefix_token_count": sum(
            row["forced_prefix_token_count"] for row in message_rows
        ),
        "total_full_message_token_count": sum(
            row["full_message_token_count"] for row in message_rows
        ),
        "total_full_message_character_count": sum(
            row["full_message_character_count"] for row in message_rows
        ),
        "exact_recovery": exact_recovery,
        "generation_seconds": generation_seconds_total,
        "recovery_seconds": recovery_seconds_total,
        "forced_prefix_mean_log_probability_mean": mean_of_present(
            [row["mean_token_log_probability"] for row in forced_feature_rows]
        ),
        "forced_prefix_repetition_mean": mean_of_present(
            [row["repeated_token_fraction"] for row in forced_feature_rows]
        ),
        "forced_prefix_punctuation_fraction_mean": mean_of_present(
            [row["punctuation_fraction"] for row in forced_feature_rows]
        ),
        "forced_prefix_alphabetic_fraction_mean": mean_of_present(
            [row["alphabetic_fraction"] for row in forced_feature_rows]
        ),
        "full_message_mean_log_probability_mean": mean_of_present(
            [row["mean_token_log_probability"] for row in full_feature_rows]
        ),
        "full_message_repetition_mean": mean_of_present(
            [row["repeated_token_fraction"] for row in full_feature_rows]
        ),
        "full_message_punctuation_fraction_mean": mean_of_present(
            [row["punctuation_fraction"] for row in full_feature_rows]
        ),
        "full_message_alphabetic_fraction_mean": mean_of_present(
            [row["alphabetic_fraction"] for row in full_feature_rows]
        ),
        "mean_generated_rank": mean_of_present(
            [row["mean_generated_rank"] for row in full_feature_rows]
        ),
        "p95_generated_rank": mean_of_present(
            [row["p95_generated_rank"] for row in full_feature_rows]
        ),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": model_path_relative,
        "notes": condition.notes + "; legacy metric columns refer to full-message metrics",
        "mean_token_log_probability": mean_of_present(
            [row["mean_token_log_probability"] for row in full_feature_rows]
        ),
        "repeated_token_fraction_mean": mean_of_present(
            [row["repeated_token_fraction"] for row in full_feature_rows]
        ),
        "punctuation_fraction_mean": mean_of_present(
            [row["punctuation_fraction"] for row in full_feature_rows]
        ),
        "alphabetic_fraction_mean": mean_of_present(
            [row["alphabetic_fraction"] for row in full_feature_rows]
        ),
        "forced_prefix_artifact_count_mean": mean_of_present(
            [row.get("artifact_count_total") for row in forced_feature_rows]
        ),
        "full_message_artifact_count_mean": mean_of_present(
            [row.get("artifact_count_total") for row in full_feature_rows]
        ),
    }
    return trial_row, message_rows, feature_rows


def quality_message_notes(message_row: dict) -> str:
    text = message_row.get("full_message_text") or ""
    flags = artifact_flags(text)
    notes = []
    if flags["artifact_count_total"]:
        artifact_names = [
            name.replace("contains_", "")
            for name, value in flags.items()
            if name.startswith("contains_") and value
        ]
        notes.append("artifact flags: {}".format(", ".join(artifact_names)))
    forced_text = message_row.get("forced_prefix_text") or ""
    if len(forced_text.strip()) < 20:
        notes.append("forced prefix is very short")
    if message_row.get("actual_tail_token_count", 0) >= 60:
        notes.append("sentence tail reached max token cap")
    full_repetition = message_row.get("full_message_repeated_token_fraction")
    if full_repetition is not None and full_repetition > 0.35:
        notes.append("full message is repetitive")
    if not notes:
        notes.append("mixed or ordinary by lightweight heuristics; inspect manually")
    return "; ".join(notes)


def write_segmented_quality_comparison(
    output_dir: Path,
    control_rows: Sequence[dict],
    trial_frame: pd.DataFrame,
    message_rows: Sequence[dict],
) -> Path:
    path = output_dir / "SEGMENTED_QUALITY_COMPARISON.md"
    lines = [
        "# Segmented Protocol Quality Controls Pilot",
        "",
        "This pilot extends the segmented multi-cover protocol by separating forced-prefix "
        "metrics from full-message metrics, adding sentence-boundary tails, adding natural "
        "tails to compact control requests, and testing a deterministic safe-text token filter.",
        "",
        "The simulated parties already share `K_common`. This is not encryption, key exchange, "
        "authentication, signing, or cryptographic security.",
        "",
        "## Metric Separation",
        "",
        "Forced-prefix metrics describe the payload-bearing tokens only. Full-message metrics "
        "describe the public message including the natural tail. Tail-heavy conditions can look "
        "better in full-message metrics even when the forced prefix remains awkward.",
        "",
        "## Control Request Recovery",
        "",
        "| Condition | Filter | Tail policy | Tail tokens | Recovery | Artifact count |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in control_rows:
        flags = artifact_flags(row.get("full_message_text") or "")
        lines.append(
            "| `{}` | `{}` | `{}` | {} | `{}` | {} |".format(
                row.get("condition_name"),
                row.get("token_filter_name"),
                row.get("tail_policy"),
                row.get("actual_tail_token_count"),
                row.get("control_exact_recovery"),
                flags["artifact_count_total"],
            )
        )

    lines.extend(
        [
            "",
            "## Response Condition Comparison",
            "",
            "| Condition | Filter | Tail policy | Recovery | Forced logprob | Full logprob | Forced repetition | Full repetition | Full artifacts |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not trial_frame.empty:
        grouped = trial_frame.groupby("condition_name", as_index=False).agg(
            {
                "token_filter_name": "first",
                "tail_policy": "first",
                "exact_recovery": "mean",
                "forced_prefix_mean_log_probability_mean": "mean",
                "full_message_mean_log_probability_mean": "mean",
                "forced_prefix_repetition_mean": "mean",
                "full_message_repetition_mean": "mean",
                "full_message_artifact_count_mean": "mean",
            }
        )
        for _, row in grouped.iterrows():
            lines.append(
                "| `{}` | `{}` | `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    row["condition_name"],
                    row["token_filter_name"],
                    row["tail_policy"],
                    row["exact_recovery"],
                    row["forced_prefix_mean_log_probability_mean"],
                    row["full_message_mean_log_probability_mean"],
                    row["forced_prefix_repetition_mean"],
                    row["full_message_repetition_mean"],
                    row["full_message_artifact_count_mean"],
                )
            )

    lines.extend(
        [
            "",
            "## Focused Comparisons",
            "",
            "- `fixed_tail40` is the prior comparable baseline.",
            "- `sentence_tail_min20_max60` tries to avoid abrupt endings while staying deterministic.",
            "- `safe_text_filter_v1` ranks and recovers only over a deterministic allowed-token set.",
            "- Multi-topic schedules rotate ordinary prose prompts and avoid the prior biology/car dialogue failure modes.",
            "",
            "## Examples",
            "",
        ]
    )
    examples_by_condition: Dict[str, List[dict]] = {}
    for message_row in message_rows:
        condition_name = str(message_row.get("condition_name"))
        examples_by_condition.setdefault(condition_name, [])
        if len(examples_by_condition[condition_name]) < 1:
            examples_by_condition[condition_name].append(message_row)

    for condition in SEGMENTED_QUALITY_CONDITIONS:
        lines.extend(["### `{}`".format(condition.name), ""])
        for message_row in examples_by_condition.get(condition.name, []):
            lines.extend(
                [
                    "- trial_id: `{}`".format(message_row.get("trial_id")),
                    "- payload_name: `{}`".format(message_row.get("payload_name")),
                    "- segment_index: `{}`".format(message_row.get("segment_index")),
                    "- prompt_name: `{}`".format(message_row.get("prompt_name")),
                    "- token_filter_name: `{}`".format(message_row.get("token_filter_name")),
                    "- tail_policy: `{}`".format(message_row.get("tail_policy")),
                    "- exact_segment_recovery: `{}`".format(
                        message_row.get("exact_segment_recovery")
                    ),
                    "- actual_tail_token_count: `{}`".format(
                        message_row.get("actual_tail_token_count")
                    ),
                    "- notes: {}".format(quality_message_notes(message_row)),
                    "",
                    "Forced prefix:",
                    "",
                    "```text",
                    markdown_text_block(message_row.get("forced_prefix_text") or "", limit=500),
                    "```",
                    "",
                    "Tail:",
                    "",
                    "```text",
                    markdown_text_block(message_row.get("tail_text") or "", limit=700),
                    "```",
                    "",
                    "Full message:",
                    "",
                    "```text",
                    markdown_text_block(message_row.get("full_message_text") or "", limit=1000),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Limitations",
            "",
            "- All payloads are deterministic synthetic examples.",
            "- The control code is a synthetic codebook label, not a secret key or operational command.",
            "- This is not encryption, key exchange, authentication, signing, or cryptographic security.",
            "- Exact-copy conditions are required for recovery.",
            "- The safe-text filter is heuristic and may change capacity, rank pressure, and style.",
            "- Full-message quality can improve because of tails even if forced prefixes remain unnatural.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_segmented_quality_summary_markdown(output_dir: Path, summary: dict) -> Path:
    path = output_dir / "SUMMARY.md"
    text = """# RankCloak Segmented Quality Controls Summary

- Profile: {profile}
- Model status: {model_status}
- Model: {model_repo_id} / {model_filename}
- Control recovery: {control_passes} pass, {control_failures} fail
- Response recovery: {response_passes} pass, {response_failures} fail
- Response trial count: {response_trial_count}

## Main Quality Observation

This profile separates payload-bearing forced-prefix metrics from full-message metrics. Sentence tails and safe-token filtering should be interpreted through both views: a full message may look natural because of its tail even when the forced prefix remains visibly constrained.

## Important Limitations

- Synthetic payloads only.
- No encryption, key exchange, authentication, signing, or cryptographic security is claimed.
- The control code is a compact pre-agreed label.
- The decoder ignores tails and recovers only forced prefixes.
- Exact-copy conditions are required.

## Next Recommended Experiment

Inspect `SEGMENTED_QUALITY_COMPARISON.md`, then decide whether to keep `safe_text_filter_v1` and whether forced-prefix quality justifies a smaller segment size or a distribution-matched rank code.
""".format(
        profile=summary["profile"],
        model_status="loaded" if summary["model_loaded"] else "not loaded",
        model_repo_id=summary["model_repo_id"],
        model_filename=summary["model_filename"],
        control_passes=summary["control_recovery_pass_count"],
        control_failures=summary["control_recovery_fail_count"],
        response_passes=summary["response_recovery_pass_count"],
        response_failures=summary["response_recovery_fail_count"],
        response_trial_count=summary["response_trial_count"],
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_segmented_quality_controls(
    output_dir: Path,
    project_root: Path,
    model: Any,
    model_path: Optional[Path],
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    payloads: Sequence[SyntheticPayload],
    cover_prompts: Dict[str, str],
    command_line_args: Sequence[str],
    model_loaded: bool,
    model_status: str,
    model_error: Optional[str],
    payload_text_limit: Optional[int] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    segment_size = int(CONTROL_CODEBOOK["C1"]["segment_size"])

    control_rows: List[dict] = []
    trial_rows: List[dict] = []
    message_rows: List[dict] = []
    feature_rows: List[dict] = []
    allowed_masks: Dict[str, Optional[Sequence[bool]]] = {"none": None}

    if model is not None:
        if any(condition.token_filter_name == SAFE_TEXT_FILTER_V1 for condition in SEGMENTED_QUALITY_CONDITIONS):
            allowed_masks[SAFE_TEXT_FILTER_V1] = build_allowed_token_mask(model, SAFE_TEXT_FILTER_V1)
        for condition in SEGMENTED_QUALITY_CONDITIONS:
            allowed_token_mask = allowed_masks.get(condition.token_filter_name)
            control_row, control_feature_rows = run_quality_control_request_trial(
                condition=condition,
                model=model,
                cover_prompts=cover_prompts,
                allowed_token_mask=allowed_token_mask,
            )
            control_rows.append(control_row)
            feature_rows.extend(control_feature_rows)

        trial_counter = 1
        for payload in payloads:
            for condition in SEGMENTED_QUALITY_CONDITIONS:
                trial_id = "quality_{:03d}".format(trial_counter)
                trial_counter += 1
                trial_row, trial_message_rows, trial_feature_rows = run_quality_response_trial(
                    trial_id=trial_id,
                    payload=payload,
                    condition=condition,
                    cover_prompts=cover_prompts,
                    model=model,
                    model_repo_id=model_repo_id,
                    model_filename=model_filename,
                    model_path_relative=model_path_relative,
                    segment_size=segment_size,
                    payload_text_limit=payload_text_limit,
                    allowed_token_mask=allowed_masks.get(condition.token_filter_name),
                )
                trial_rows.append(trial_row)
                message_rows.extend(trial_message_rows)
                feature_rows.extend(trial_feature_rows)

    trial_frame = ordered_frame(trial_rows, SEGMENTED_QUALITY_TRIAL_COLUMNS)
    feature_frame = ordered_frame(feature_rows, SEGMENTED_QUALITY_FEATURE_COLUMNS)

    control_path = output_dir / "control_request_trials.jsonl"
    write_jsonl(control_path, control_rows)
    trial_path = output_dir / "segmented_quality_trials.csv"
    trial_frame.to_csv(trial_path, index=False)
    messages_path = output_dir / "segmented_quality_messages.jsonl"
    write_jsonl(messages_path, message_rows)
    feature_path = output_dir / "cover_text_features.csv"
    feature_frame.to_csv(feature_path, index=False)
    comparison_path = write_segmented_quality_comparison(
        output_dir=output_dir,
        control_rows=control_rows,
        trial_frame=trial_frame,
        message_rows=message_rows,
    )

    figure_paths = [
        plot_quality_forced_vs_full_logprob(
            trial_frame, figures_dir / "quality_forced_vs_full_logprob.png"
        ),
        plot_quality_forced_vs_full_repetition(
            trial_frame, figures_dir / "quality_forced_vs_full_repetition.png"
        ),
        plot_quality_tail_policy_logprob(
            trial_frame, figures_dir / "quality_tail_policy_logprob.png"
        ),
        plot_quality_filter_effect_logprob(
            trial_frame, figures_dir / "quality_filter_effect_logprob.png"
        ),
        plot_quality_filter_effect_artifacts(
            feature_frame, figures_dir / "quality_filter_effect_artifacts.png"
        ),
        plot_quality_recovery_by_condition(
            trial_frame, figures_dir / "quality_recovery_by_condition.png"
        ),
    ]
    manifest_path = output_dir / "MANIFEST.json"
    write_manifest(
        output_path=manifest_path,
        project_root=project_root,
        profile="segmented-quality-controls",
        output_dir=output_dir,
        command_line_args=command_line_args,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
    )

    control_values = [bool(row.get("control_exact_recovery")) for row in control_rows]
    if not trial_frame.empty and "exact_recovery" in trial_frame:
        response_values = list(trial_frame["exact_recovery"].astype(bool))
    else:
        response_values = []
    generated_files = [
        control_path,
        trial_path,
        messages_path,
        feature_path,
        comparison_path,
        manifest_path,
        *figure_paths,
        output_dir / "summary.json",
        output_dir / "SUMMARY.md",
    ]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": "segmented-quality-controls",
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_loaded": bool(model_loaded),
        "control_trial_count": len(control_rows),
        "control_recovery_pass_count": int(sum(control_values)),
        "control_recovery_fail_count": int(len(control_values) - sum(control_values)),
        "response_trial_count": len(response_values),
        "response_recovery_pass_count": int(sum(response_values)),
        "response_recovery_fail_count": int(len(response_values) - sum(response_values)),
        "payloads": [payload.name for payload in payloads],
        "conditions": [condition.name for condition in SEGMENTED_QUALITY_CONDITIONS],
        "segment_size": segment_size,
        "tail_policies": sorted(set(condition.tail_policy for condition in SEGMENTED_QUALITY_CONDITIONS)),
        "token_filter_names": sorted(set(condition.token_filter_name for condition in SEGMENTED_QUALITY_CONDITIONS)),
        "generated_result_files": [
            repo_relative_path(path, project_root) for path in generated_files
        ],
        "important_notes": [
            "All payloads are deterministic synthetic examples.",
            "Forced-prefix and full-message metrics are separated.",
            "Control code C1 is a compact synthetic codebook label, not a key exchange.",
            "Decoder recovers only forced prefixes and ignores natural tails.",
            "safe_text_filter_v1 is deterministic and heuristic.",
            "Model status: {}".format(model_status),
            "Model error: {}".format(model_error) if model_error else "Model loaded or was not required.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_segmented_quality_summary_markdown(output_dir, summary)
    return summary
