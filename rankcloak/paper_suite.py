"""Paper-oriented RankCloak experiment suite."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import generate_greedy_baseline
from .bootstrap_statistics import bootstrap_difference_ci, bootstrap_mean_ci
from .detection import prepare_detector_dataset, run_detector_baselines
from .metrics import extract_text_features, summarize_optional_log_probabilities, summarize_optional_ranks
from .model_io import evaluate_context, get_last_logits, make_context_token_ids, safe_detokenize
from .paper_payloads import (
    PaperPayload,
    generate_full_paper_payloads,
    generate_pilot_paper_payloads,
    paper_payload_rows,
    payload_class_counts,
)
from .prompts import cover_prompt_dictionary, prompt_family
from .rank_codec import (
    decode_bounded_ranks_to_bytes,
    decode_hex_nibble_ranks_to_text,
    direct_subword_ranks_for_text,
    encode_bytes_to_bounded_ranks,
    encode_hex_nibbles_to_ranks,
    generate_token_ids_from_ranks,
    recover_ranks_from_generated_ids,
    token_log_probability,
)
from .reproducibility import repo_relative_path, write_manifest
from .schemas import (
    DETECTOR_BASELINE_COLUMNS,
    DETECTOR_DATASET_COLUMNS,
    EFFECT_SIZE_SUMMARY_COLUMNS,
    PAPER_CODEC_COMPARISON_COLUMNS,
    PAPER_COVER_TEXT_FEATURE_COLUMNS,
    PAPER_PAYLOAD_COLUMNS,
    PAPER_RANK_PRESSURE_COLUMNS,
    PAPER_SEGMENTED_TRIAL_COLUMNS,
    PAPER_STEGOTEXT_TRIAL_COLUMNS,
    STATISTICAL_SUMMARY_COLUMNS,
)
from .segmented_protocol import (
    chunk_rank_sequence,
    flatten_rank_chunks,
    mean_of_present,
    should_stop_sentence_tail,
)
from .token_filters import (
    SAFE_TEXT_FILTER_V1,
    build_allowed_token_mask,
    choose_token_at_rank_with_optional_filter,
    rank_token_with_optional_filter,
)


PAPER_PILOT_PROMPTS = [
    "recipe_long_specific",
    "recipe_forum_exchange_specific",
    "grocery_planning_note_specific",
]
PAPER_MAIN_PROMPTS = [
    "recipe_long_specific",
    "recipe_forum_exchange_specific",
    "recipe_blog",
    "grocery_planning_note_specific",
    "plant_care_note_specific",
]
PAPER_PROTOCOL_VARIANTS = [
    "raw_subword_rank_pressure",
    "nonseg_ascii_b8",
    "nonseg_ascii_b16",
    "nonseg_hex_nibble_b16",
    "segmented_hex_single_topic_sentence_tail_filtered",
    "segmented_hex_multi_topic_sentence_tail_filtered",
    "segmented_hex_multi_topic_leadin8_sentence_tail_filtered",
]
PAPER_SEGMENT_SIZE = 8
PAPER_TAIL_POLICY = "sentence_tail_min20_max60"
PAPER_BOOTSTRAP_SEED = 20260521


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


def write_frame(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> pd.DataFrame:
    frame = ordered_frame(rows, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def prompt_names_for_profile(profile: str) -> List[str]:
    if profile == "paper-main":
        return list(PAPER_MAIN_PROMPTS)
    return list(PAPER_PILOT_PROMPTS)


def payloads_for_profile(profile: str) -> List[PaperPayload]:
    if profile == "paper-main":
        return generate_full_paper_payloads()
    return generate_pilot_paper_payloads()


def paper_rank_summary(payload_name: str, ranks: Sequence[int]) -> dict:
    if not ranks:
        return {
            "payload_name": payload_name,
            "rank_count": 0,
            "mean_rank": None,
            "median_rank": None,
            "p95_rank": None,
            "max_rank": None,
            "fraction_rank_le_8": None,
            "fraction_rank_le_16": None,
            "fraction_rank_le_64": None,
        }
    values = np.asarray(list(map(int, ranks)), dtype=np.float64)
    return {
        "payload_name": payload_name,
        "rank_count": int(values.size),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
        "p95_rank": float(np.percentile(values, 95)),
        "max_rank": int(np.max(values)),
        "fraction_rank_le_8": float(np.mean(values <= 8)),
        "fraction_rank_le_16": float(np.mean(values <= 16)),
        "fraction_rank_le_64": float(np.mean(values <= 64)),
    }


def run_paper_rank_pressure(
    payloads: Sequence[PaperPayload],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
) -> List[dict]:
    rows = []
    for payload in payloads:
        if model is None:
            row = paper_rank_summary(payload.payload_name, [])
            row.update(
                {
                    "payload_class": payload.payload_class,
                    "representation_name": "raw_subword_rank_pressure",
                    "model_repo_id": model_repo_id,
                    "model_filename": model_filename,
                    "model_path_relative": model_path_relative,
                    "notes": "model unavailable; direct subword rank pressure skipped",
                }
            )
            rows.append(row)
            continue
        trace = direct_subword_ranks_for_text(model, payload.payload_text)
        row = paper_rank_summary(payload.payload_name, trace["ranks"])
        row.update(
            {
                "payload_class": payload.payload_class,
                "representation_name": "raw_subword_rank_pressure",
                "model_repo_id": model_repo_id,
                "model_filename": model_filename,
                "model_path_relative": model_path_relative,
                "notes": "diagnostic baseline only; no cover generation",
            }
        )
        rows.append(row)
    return rows


def encode_payload_representation(
    payload: PaperPayload,
    representation_name: str,
    alphabet_size: Optional[int],
) -> Optional[dict]:
    if representation_name == "ascii_bytes_fixed_radix":
        if alphabet_size is None:
            raise ValueError("alphabet_size is required for ascii fixed-radix encoding")
        encoded = encode_bytes_to_bounded_ranks(payload.payload_bytes, int(alphabet_size))
        encoded["representation_name"] = representation_name
        return encoded
    if representation_name == "raw_hex_nibbles":
        if not payload.is_hex_like:
            return None
        encoded = encode_hex_nibbles_to_ranks(payload.payload_text)
        encoded["representation_name"] = representation_name
        return encoded
    raise ValueError("Unknown representation: {}".format(representation_name))


def decode_payload_representation(
    ranks: Sequence[int],
    metadata: Dict[str, Any],
    representation_name: str,
) -> bytes:
    if representation_name == "ascii_bytes_fixed_radix":
        return decode_bounded_ranks_to_bytes(ranks, metadata)
    if representation_name == "raw_hex_nibbles":
        return decode_hex_nibble_ranks_to_text(ranks, metadata).encode("utf-8")
    raise ValueError("Unknown representation: {}".format(representation_name))


def build_paper_codec_comparison(payloads: Sequence[PaperPayload]) -> List[dict]:
    rows = []
    for payload in payloads:
        for alphabet_size in (8, 16):
            encoded = encode_payload_representation(payload, "ascii_bytes_fixed_radix", alphabet_size)
            decoded = decode_payload_representation(
                encoded["ranks"], encoded["metadata"], "ascii_bytes_fixed_radix"
            )
            rows.append(
                {
                    "payload_name": payload.payload_name,
                    "payload_class": payload.payload_class,
                    "representation_name": "ascii_bytes_fixed_radix",
                    "alphabet_size": alphabet_size,
                    "rank_count": len(encoded["ranks"]),
                    "max_possible_rank": alphabet_size,
                    "bits_per_rank_estimate": float(math.log2(alphabet_size)),
                    "applies_to_payload": True,
                    "exact_codec_roundtrip": decoded == payload.payload_bytes,
                    "notes": "ASCII artifact bytes encoded as fixed-radix bounded ranks",
                }
            )
        if payload.is_hex_like:
            encoded_hex = encode_payload_representation(payload, "raw_hex_nibbles", 16)
            decoded_hex = decode_payload_representation(
                encoded_hex["ranks"], encoded_hex["metadata"], "raw_hex_nibbles"
            )
            rows.append(
                {
                    "payload_name": payload.payload_name,
                    "payload_class": payload.payload_class,
                    "representation_name": "raw_hex_nibbles",
                    "alphabet_size": 16,
                    "rank_count": len(encoded_hex["ranks"]),
                    "max_possible_rank": 16,
                    "bits_per_rank_estimate": 4.0,
                    "applies_to_payload": True,
                    "exact_codec_roundtrip": decoded_hex == payload.payload_text.encode("utf-8"),
                    "notes": "one lowercase hex character per rank",
                }
            )
        else:
            rows.append(
                {
                    "payload_name": payload.payload_name,
                    "payload_class": payload.payload_class,
                    "representation_name": "raw_hex_nibbles",
                    "alphabet_size": 16,
                    "rank_count": None,
                    "max_possible_rank": 16,
                    "bits_per_rank_estimate": 4.0,
                    "applies_to_payload": False,
                    "exact_codec_roundtrip": None,
                    "notes": "not applicable because payload text is not lowercase hex",
                }
            )
    return rows


def feature_row(
    source_type: str,
    trial_id: str,
    payload: Optional[PaperPayload],
    protocol_variant: Optional[str],
    representation_name: Optional[str],
    prompt_name: str,
    segment_index: Optional[int],
    text: str,
    token_ids: Sequence[int],
    ranks: Sequence[int],
    token_log_probabilities: Sequence[float],
    notes: str,
) -> dict:
    row = extract_text_features(
        text,
        token_ids=token_ids,
        ranks=ranks,
        token_log_probabilities=token_log_probabilities,
    )
    row.update(
        {
            "source_type": source_type,
            "trial_id": trial_id,
            "payload_name": payload.payload_name if payload else None,
            "payload_class": payload.payload_class if payload else None,
            "protocol_variant": protocol_variant,
            "representation_name": representation_name,
            "prompt_name": prompt_name,
            "prompt_family": prompt_family(prompt_name),
            "segment_index": segment_index,
            "notes": notes,
        }
    )
    return row


def nonseg_variant_specs() -> List[dict]:
    return [
        {
            "protocol_variant": "nonseg_ascii_b8",
            "representation_name": "ascii_bytes_fixed_radix",
            "alphabet_size": 8,
        },
        {
            "protocol_variant": "nonseg_ascii_b16",
            "representation_name": "ascii_bytes_fixed_radix",
            "alphabet_size": 16,
        },
        {
            "protocol_variant": "nonseg_hex_nibble_b16",
            "representation_name": "raw_hex_nibbles",
            "alphabet_size": 16,
        },
    ]


def run_paper_nonseg_trials(
    payloads: Sequence[PaperPayload],
    prompt_names: Sequence[str],
    cover_prompts: Dict[str, str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
) -> Tuple[List[dict], List[dict], List[dict]]:
    trial_rows: List[dict] = []
    example_rows: List[dict] = []
    feature_rows: List[dict] = []
    if model is None:
        return trial_rows, example_rows, feature_rows
    trial_counter = 1
    for payload in payloads:
        for spec in nonseg_variant_specs():
            encoded = encode_payload_representation(
                payload, spec["representation_name"], spec["alphabet_size"]
            )
            if encoded is None:
                continue
            for prompt_name in prompt_names:
                trial_id = "paper_nonseg_{:04d}".format(trial_counter)
                trial_counter += 1
                context_token_ids = make_context_token_ids(model, cover_prompts[prompt_name])
                generation_started = time.perf_counter()
                generated = generate_token_ids_from_ranks(
                    model, context_token_ids, encoded["ranks"]
                )
                generation_seconds = time.perf_counter() - generation_started
                recovery_started = time.perf_counter()
                recovered = recover_ranks_from_generated_ids(
                    model, context_token_ids, generated["generated_token_ids"]
                )
                recovery_seconds = time.perf_counter() - recovery_started
                decoded = decode_payload_representation(
                    recovered["ranks"], encoded["metadata"], spec["representation_name"]
                )
                expected = (
                    payload.payload_text.encode("utf-8")
                    if spec["representation_name"] == "raw_hex_nibbles"
                    else payload.payload_bytes
                )
                exact_recovery = decoded == expected
                features = feature_row(
                    source_type="nonseg_rankcloak",
                    trial_id=trial_id,
                    payload=payload,
                    protocol_variant=spec["protocol_variant"],
                    representation_name=spec["representation_name"],
                    prompt_name=prompt_name,
                    segment_index=None,
                    text=generated["generated_text"],
                    token_ids=generated["generated_token_ids"],
                    ranks=recovered["ranks"],
                    token_log_probabilities=generated.get("token_log_probabilities", []),
                    notes="non-segmented exact-copy RankCloak cover",
                )
                row = {
                    "trial_id": trial_id,
                    "protocol_variant": spec["protocol_variant"],
                    "payload_name": payload.payload_name,
                    "payload_class": payload.payload_class,
                    "payload_kind": payload.payload_kind,
                    "representation_name": spec["representation_name"],
                    "alphabet_size": spec["alphabet_size"],
                    "prompt_name": prompt_name,
                    "prompt_family": prompt_family(prompt_name),
                    "rank_count": len(encoded["ranks"]),
                    "generated_token_count": len(generated["generated_token_ids"]),
                    "generated_character_count": len(generated["generated_text"]),
                    "exact_recovery": exact_recovery,
                    "generation_seconds": generation_seconds,
                    "recovery_seconds": recovery_seconds,
                    "mean_token_log_probability": features["mean_token_log_probability"],
                    "median_token_log_probability": features["median_token_log_probability"],
                    "mean_generated_rank": features["mean_generated_rank"],
                    "p95_generated_rank": features["p95_generated_rank"],
                    "repeated_token_fraction": features["repeated_token_fraction"],
                    "punctuation_fraction": features["punctuation_fraction"],
                    "alphabetic_fraction": features["alphabetic_fraction"],
                    "artifact_count_total": features["artifact_count_total"],
                    "model_repo_id": model_repo_id,
                    "model_filename": model_filename,
                    "model_path_relative": model_path_relative,
                    "notes": "synthetic payload; not encryption or cryptographic security",
                }
                trial_rows.append(row)
                feature_rows.append(features)
                example_rows.append(
                    {
                        **row,
                        "generated_text": generated["generated_text"],
                        "generated_token_ids": generated["generated_token_ids"],
                        "recovered_ranks": recovered["ranks"],
                    }
                )
    return trial_rows, example_rows, feature_rows


def segmented_variant_specs() -> List[dict]:
    return [
        {
            "protocol_variant": "segmented_hex_single_topic_sentence_tail_filtered",
            "prompt_names": ("recipe_long_specific",),
            "topic_schedule_name": "single_recipe_long",
            "leadin_token_count": 0,
            "leadin_policy": "none",
        },
        {
            "protocol_variant": "segmented_hex_multi_topic_sentence_tail_filtered",
            "prompt_names": (
                "recipe_long_specific",
                "recipe_forum_exchange_specific",
                "recipe_blog",
                "grocery_planning_note_specific",
                "plant_care_note_specific",
            ),
            "topic_schedule_name": "mixed_recipe_forum_blog_grocery_plant",
            "leadin_token_count": 0,
            "leadin_policy": "none",
        },
        {
            "protocol_variant": "segmented_hex_multi_topic_leadin8_sentence_tail_filtered",
            "prompt_names": (
                "recipe_long_specific",
                "recipe_forum_exchange_specific",
                "recipe_blog",
                "grocery_planning_note_specific",
                "plant_care_note_specific",
            ),
            "topic_schedule_name": "mixed_recipe_forum_blog_grocery_plant",
            "leadin_token_count": 8,
            "leadin_policy": "greedy_leadin8",
        },
    ]


def decode_forced_span_token_ids(
    full_token_ids: Sequence[int],
    leadin_token_count: int,
    forced_prefix_token_count: int,
) -> List[int]:
    start = int(leadin_token_count)
    stop = start + int(forced_prefix_token_count)
    return list(map(int, full_token_ids[start:stop]))


def generate_segmented_message_with_leadin(
    model: Any,
    context_token_ids: Sequence[int],
    forced_ranks: Sequence[int],
    leadin_token_count: int,
    tail_policy: str,
    allowed_token_mask: Optional[Sequence[bool]],
) -> dict:
    context = list(map(int, context_token_ids))
    evaluate_context(model, context)
    leadin_ids: List[int] = []
    leadin_ranks: List[int] = []
    leadin_log_probs: List[float] = []
    forced_ids: List[int] = []
    forced_ranks_actual: List[int] = []
    forced_log_probs: List[float] = []
    tail_ids: List[int] = []
    tail_ranks: List[int] = []
    tail_log_probs: List[float] = []

    for _ in range(int(leadin_token_count)):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, 1, allowed_token_mask)
        leadin_ids.append(token_id)
        leadin_ranks.append(rank_token_with_optional_filter(logits, token_id, allowed_token_mask))
        leadin_log_probs.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    for rank in forced_ranks:
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, int(rank), allowed_token_mask)
        forced_ids.append(token_id)
        forced_ranks_actual.append(rank_token_with_optional_filter(logits, token_id, allowed_token_mask))
        forced_log_probs.append(token_log_probability(logits, token_id))
        model.eval([token_id])

    if tail_policy != PAPER_TAIL_POLICY:
        raise ValueError("Unsupported paper tail policy: {}".format(tail_policy))
    for _ in range(60):
        logits = get_last_logits(model)
        token_id = choose_token_at_rank_with_optional_filter(logits, 1, allowed_token_mask)
        tail_ids.append(token_id)
        tail_ranks.append(rank_token_with_optional_filter(logits, token_id, allowed_token_mask))
        tail_log_probs.append(token_log_probability(logits, token_id))
        model.eval([token_id])
        tail_text = safe_detokenize(model, tail_ids)
        if should_stop_sentence_tail(tail_text, len(tail_ids), 20, 60):
            break

    full_ids = leadin_ids + forced_ids + tail_ids
    full_ranks = leadin_ranks + forced_ranks_actual + tail_ranks
    full_log_probs = leadin_log_probs + forced_log_probs + tail_log_probs
    return {
        "leadin_token_ids": leadin_ids,
        "forced_token_ids": forced_ids,
        "tail_token_ids": tail_ids,
        "full_token_ids": full_ids,
        "leadin_ranks": leadin_ranks,
        "forced_ranks": forced_ranks_actual,
        "tail_ranks": tail_ranks,
        "full_ranks": full_ranks,
        "leadin_log_probabilities": leadin_log_probs,
        "forced_log_probabilities": forced_log_probs,
        "tail_log_probabilities": tail_log_probs,
        "full_log_probabilities": full_log_probs,
        "leadin_text": safe_detokenize(model, leadin_ids),
        "forced_prefix_text": safe_detokenize(model, forced_ids),
        "tail_text": safe_detokenize(model, tail_ids),
        "full_message_text": safe_detokenize(model, full_ids),
    }


def recover_forced_ranks_after_leadin(
    model: Any,
    context_token_ids: Sequence[int],
    leadin_token_ids: Sequence[int],
    forced_token_ids: Sequence[int],
    allowed_token_mask: Optional[Sequence[bool]],
) -> List[int]:
    evaluate_context(model, list(map(int, context_token_ids)))
    if leadin_token_ids:
        model.eval(list(map(int, leadin_token_ids)))
    recovered: List[int] = []
    for token_id in map(int, forced_token_ids):
        logits = get_last_logits(model)
        recovered.append(rank_token_with_optional_filter(logits, token_id, allowed_token_mask))
        model.eval([token_id])
    return recovered


def run_paper_segmented_trials(
    payloads: Sequence[PaperPayload],
    cover_prompts: Dict[str, str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
) -> Tuple[List[dict], List[dict], List[dict]]:
    trial_rows: List[dict] = []
    message_rows: List[dict] = []
    feature_rows: List[dict] = []
    if model is None:
        return trial_rows, message_rows, feature_rows
    allowed_token_mask = build_allowed_token_mask(model, SAFE_TEXT_FILTER_V1)
    trial_counter = 1
    for payload in payloads:
        if not payload.is_hex_like:
            continue
        encoded = encode_payload_representation(payload, "raw_hex_nibbles", 16)
        chunks = chunk_rank_sequence(encoded["ranks"], PAPER_SEGMENT_SIZE)
        for spec in segmented_variant_specs():
            trial_id = "paper_segmented_{:04d}".format(trial_counter)
            trial_counter += 1
            recovered_chunks: List[List[int]] = []
            trial_message_rows: List[dict] = []
            trial_feature_rows: List[dict] = []
            generation_seconds_total = 0.0
            recovery_seconds_total = 0.0
            for segment_index_zero, chunk in enumerate(chunks):
                prompt_name = spec["prompt_names"][segment_index_zero % len(spec["prompt_names"])]
                context_token_ids = make_context_token_ids(model, cover_prompts[prompt_name])
                generation_started = time.perf_counter()
                generated = generate_segmented_message_with_leadin(
                    model=model,
                    context_token_ids=context_token_ids,
                    forced_ranks=chunk,
                    leadin_token_count=spec["leadin_token_count"],
                    tail_policy=PAPER_TAIL_POLICY,
                    allowed_token_mask=allowed_token_mask,
                )
                generation_seconds_total += time.perf_counter() - generation_started

                recovery_started = time.perf_counter()
                recovered_chunk = recover_forced_ranks_after_leadin(
                    model=model,
                    context_token_ids=context_token_ids,
                    leadin_token_ids=generated["leadin_token_ids"],
                    forced_token_ids=generated["forced_token_ids"],
                    allowed_token_mask=allowed_token_mask,
                )
                recovery_seconds_total += time.perf_counter() - recovery_started
                recovered_chunks.append(recovered_chunk)
                exact_segment_recovery = recovered_chunk == list(map(int, chunk))
                segment_index = segment_index_zero + 1

                leadin_features = feature_row(
                    "segmented_leadin",
                    trial_id,
                    payload,
                    spec["protocol_variant"],
                    "raw_hex_nibbles",
                    prompt_name,
                    segment_index,
                    generated["leadin_text"],
                    generated["leadin_token_ids"],
                    generated["leadin_ranks"],
                    generated["leadin_log_probabilities"],
                    "greedy lead-in ignored by decoder",
                )
                forced_features = feature_row(
                    "segmented_forced_prefix",
                    trial_id,
                    payload,
                    spec["protocol_variant"],
                    "raw_hex_nibbles",
                    prompt_name,
                    segment_index,
                    generated["forced_prefix_text"],
                    generated["forced_token_ids"],
                    generated["forced_ranks"],
                    generated["forced_log_probabilities"],
                    "payload-bearing forced prefix",
                )
                tail_features = feature_row(
                    "segmented_tail",
                    trial_id,
                    payload,
                    spec["protocol_variant"],
                    "raw_hex_nibbles",
                    prompt_name,
                    segment_index,
                    generated["tail_text"],
                    generated["tail_token_ids"],
                    generated["tail_ranks"],
                    generated["tail_log_probabilities"],
                    "greedy natural tail ignored by decoder",
                )
                full_features = feature_row(
                    "segmented_full_message",
                    trial_id,
                    payload,
                    spec["protocol_variant"],
                    "raw_hex_nibbles",
                    prompt_name,
                    segment_index,
                    generated["full_message_text"],
                    generated["full_token_ids"],
                    generated["full_ranks"],
                    generated["full_log_probabilities"],
                    "full public message including optional lead-in and tail",
                )
                trial_feature_rows.extend([leadin_features, forced_features, tail_features, full_features])
                trial_message_rows.append(
                    {
                        "trial_id": trial_id,
                        "protocol_variant": spec["protocol_variant"],
                        "payload_name": payload.payload_name,
                        "payload_class": payload.payload_class,
                        "segment_index": segment_index,
                        "segment_count": len(chunks),
                        "prompt_name": prompt_name,
                        "prompt_family": prompt_family(prompt_name),
                        "leadin_text": generated["leadin_text"],
                        "forced_prefix_text": generated["forced_prefix_text"],
                        "tail_text": generated["tail_text"],
                        "full_message_text": generated["full_message_text"],
                        "leadin_token_count": len(generated["leadin_token_ids"]),
                        "forced_prefix_token_count": len(generated["forced_token_ids"]),
                        "actual_tail_token_count": len(generated["tail_token_ids"]),
                        "full_message_token_count": len(generated["full_token_ids"]),
                        "exact_segment_recovery": exact_segment_recovery,
                        "forced_prefix_mean_token_log_probability": forced_features["mean_token_log_probability"],
                        "full_message_mean_token_log_probability": full_features["mean_token_log_probability"],
                        "forced_prefix_repeated_token_fraction": forced_features["repeated_token_fraction"],
                        "full_message_repeated_token_fraction": full_features["repeated_token_fraction"],
                        "forced_prefix_artifact_count_total": forced_features["artifact_count_total"],
                        "full_message_artifact_count_total": full_features["artifact_count_total"],
                        "notes": "decode policy: ignore lead-in and tail, recover forced span only",
                    }
                )
            recovered_ranks = flatten_rank_chunks(recovered_chunks)
            recovered_text = decode_hex_nibble_ranks_to_text(recovered_ranks, encoded["metadata"])
            exact_recovery = recovered_text == payload.payload_text.lower()
            forced_rows = [row for row in trial_feature_rows if row["source_type"] == "segmented_forced_prefix"]
            full_rows = [row for row in trial_feature_rows if row["source_type"] == "segmented_full_message"]
            trial_row = {
                "trial_id": trial_id,
                "protocol_variant": spec["protocol_variant"],
                "payload_name": payload.payload_name,
                "payload_class": payload.payload_class,
                "payload_kind": payload.payload_kind,
                "representation_name": "raw_hex_nibbles",
                "segment_size": PAPER_SEGMENT_SIZE,
                "segment_count": len(chunks),
                "message_count": len(chunks),
                "topic_schedule_name": spec["topic_schedule_name"],
                "leadin_policy": spec["leadin_policy"],
                "leadin_token_count": spec["leadin_token_count"],
                "tail_policy": PAPER_TAIL_POLICY,
                "actual_tail_token_count_mean": mean_of_present(
                    [row["actual_tail_token_count"] for row in trial_message_rows]
                ),
                "token_filter_name": SAFE_TEXT_FILTER_V1,
                "total_forced_rank_count": len(encoded["ranks"]),
                "total_forced_prefix_token_count": sum(
                    row["forced_prefix_token_count"] for row in trial_message_rows
                ),
                "total_full_message_token_count": sum(
                    row["full_message_token_count"] for row in trial_message_rows
                ),
                "total_full_message_character_count": sum(
                    len(row["full_message_text"]) for row in trial_message_rows
                ),
                "exact_recovery": exact_recovery,
                "generation_seconds": generation_seconds_total,
                "recovery_seconds": recovery_seconds_total,
                "forced_prefix_mean_log_probability_mean": mean_of_present(
                    [row["mean_token_log_probability"] for row in forced_rows]
                ),
                "forced_prefix_repetition_mean": mean_of_present(
                    [row["repeated_token_fraction"] for row in forced_rows]
                ),
                "forced_prefix_punctuation_fraction_mean": mean_of_present(
                    [row["punctuation_fraction"] for row in forced_rows]
                ),
                "forced_prefix_artifact_count_mean": mean_of_present(
                    [row["artifact_count_total"] for row in forced_rows]
                ),
                "full_message_mean_log_probability_mean": mean_of_present(
                    [row["mean_token_log_probability"] for row in full_rows]
                ),
                "full_message_repetition_mean": mean_of_present(
                    [row["repeated_token_fraction"] for row in full_rows]
                ),
                "full_message_punctuation_fraction_mean": mean_of_present(
                    [row["punctuation_fraction"] for row in full_rows]
                ),
                "full_message_artifact_count_mean": mean_of_present(
                    [row["artifact_count_total"] for row in full_rows]
                ),
                "model_repo_id": model_repo_id,
                "model_filename": model_filename,
                "model_path_relative": model_path_relative,
                "notes": "safe-text filtered segmented raw-hex-nibble protocol; exact-copy only",
            }
            trial_rows.append(trial_row)
            message_rows.extend(trial_message_rows)
            feature_rows.extend(trial_feature_rows)
    return trial_rows, message_rows, feature_rows


def build_baseline_targets(feature_rows: Sequence[dict]) -> Dict[str, List[int]]:
    targets: Dict[str, List[int]] = {}
    frame = pd.DataFrame(list(feature_rows))
    if frame.empty or "prompt_name" not in frame:
        return targets
    source_frame = frame[frame["source_type"].isin(["nonseg_rankcloak", "segmented_full_message"])]
    for (prompt_name, protocol_variant), group in source_frame.groupby(["prompt_name", "protocol_variant"]):
        token_counts = pd.to_numeric(group["token_count"], errors="coerce").dropna()
        if token_counts.empty:
            continue
        target = int(max(1, round(float(token_counts.median()))))
        targets.setdefault(prompt_name, [])
        if target not in targets[prompt_name]:
            targets[prompt_name].append(target)
    return {prompt: sorted(values)[:8] for prompt, values in targets.items()}


def run_paper_baselines(
    baseline_targets: Dict[str, List[int]],
    cover_prompts: Dict[str, str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
) -> Tuple[List[dict], List[dict]]:
    baseline_rows: List[dict] = []
    feature_rows: List[dict] = []
    if model is None:
        return baseline_rows, feature_rows
    baseline_counter = 1
    for prompt_name, target_counts in sorted(baseline_targets.items()):
        for target_count in target_counts:
            baseline_id = "paper_baseline_{:04d}".format(baseline_counter)
            baseline_counter += 1
            context_ids = make_context_token_ids(model, cover_prompts[prompt_name])
            generated = generate_greedy_baseline(model, context_ids, int(target_count))
            features = feature_row(
                source_type="baseline",
                trial_id=baseline_id,
                payload=None,
                protocol_variant="baseline_greedy",
                representation_name=None,
                prompt_name=prompt_name,
                segment_index=None,
                text=generated["generated_text"],
                token_ids=generated["generated_token_ids"],
                ranks=generated["ranks"],
                token_log_probabilities=generated["token_log_probabilities"],
                notes="ordinary greedy baseline matched approximately by token count",
            )
            row = {
                "baseline_id": baseline_id,
                "prompt_name": prompt_name,
                "prompt_family": prompt_family(prompt_name),
                "target_token_count": int(target_count),
                "generated_text": generated["generated_text"],
                "generated_token_count": generated["generated_token_count"],
                "generated_character_count": generated["generated_character_count"],
                "mean_token_log_probability": features["mean_token_log_probability"],
                "repeated_token_fraction": features["repeated_token_fraction"],
                "punctuation_fraction": features["punctuation_fraction"],
                "artifact_count_total": features["artifact_count_total"],
                "baseline_mode": "greedy",
                "model_repo_id": model_repo_id,
                "model_filename": model_filename,
                "notes": "baseline text contains no payload ranks",
            }
            baseline_rows.append(row)
            feature_rows.append(features)
    return baseline_rows, feature_rows


def build_statistical_summary(
    stego_frame: pd.DataFrame,
    segmented_frame: pd.DataFrame,
    n_resamples: int,
) -> pd.DataFrame:
    rows = []
    metric_sources = []
    if not stego_frame.empty:
        frame = stego_frame.copy()
        frame["exact_recovery_numeric"] = frame["exact_recovery"].astype(float)
        metric_sources.append(
            (
                frame,
                [
                    "exact_recovery_numeric",
                    "rank_count",
                    "generated_token_count",
                    "mean_token_log_probability",
                    "p95_generated_rank",
                    "repeated_token_fraction",
                    "artifact_count_total",
                ],
            )
        )
    if not segmented_frame.empty:
        frame = segmented_frame.copy()
        frame["exact_recovery_numeric"] = frame["exact_recovery"].astype(float)
        metric_sources.append(
            (
                frame,
                [
                    "exact_recovery_numeric",
                    "total_full_message_token_count",
                    "forced_prefix_mean_log_probability_mean",
                    "full_message_mean_log_probability_mean",
                    "full_message_repetition_mean",
                    "full_message_artifact_count_mean",
                ],
            )
        )
    for frame, metrics in metric_sources:
        for metric_name in metrics:
            if metric_name not in frame.columns:
                continue
            for protocol_variant, group in frame.groupby("protocol_variant"):
                stats = bootstrap_mean_ci(group[metric_name], n_resamples, PAPER_BOOTSTRAP_SEED)
                rows.append(
                    {
                        "metric_name": metric_name.replace("_numeric", ""),
                        "group_name": "protocol_variant",
                        "protocol_variant": protocol_variant,
                        "payload_class": None,
                        "prompt_family": None,
                        **stats,
                        "notes": "deterministic bootstrap over recorded trial rows",
                    }
                )
            if "payload_class" in frame.columns:
                for (protocol_variant, payload_class), group in frame.groupby(["protocol_variant", "payload_class"]):
                    stats = bootstrap_mean_ci(group[metric_name], n_resamples, PAPER_BOOTSTRAP_SEED)
                    rows.append(
                        {
                            "metric_name": metric_name.replace("_numeric", ""),
                            "group_name": "protocol_variant_by_payload_class",
                            "protocol_variant": protocol_variant,
                            "payload_class": payload_class,
                            "prompt_family": None,
                            **stats,
                            "notes": "deterministic bootstrap over recorded trial rows",
                        }
                    )
    return ordered_frame(rows, STATISTICAL_SUMMARY_COLUMNS)


def effect_row(
    comparison_name: str,
    metric_name: str,
    group_a: str,
    values_a: Sequence[object],
    group_b: str,
    values_b: Sequence[object],
    n_resamples: int,
    notes: str,
) -> dict:
    stats = bootstrap_difference_ci(values_a, values_b, n_resamples, PAPER_BOOTSTRAP_SEED)
    return {
        "comparison_name": comparison_name,
        "metric_name": metric_name,
        "group_a": group_a,
        "group_b": group_b,
        **stats,
        "notes": notes,
    }


def build_effect_size_summary(
    stego_frame: pd.DataFrame,
    segmented_frame: pd.DataFrame,
    project_root: Path,
    n_resamples: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    if not stego_frame.empty:
        for metric in ["mean_token_log_probability", "generated_token_count", "artifact_count_total"]:
            group_a = stego_frame[stego_frame["protocol_variant"] == "nonseg_ascii_b8"]
            group_b = stego_frame[stego_frame["protocol_variant"] == "nonseg_ascii_b16"]
            if not group_a.empty and not group_b.empty:
                rows.append(
                    effect_row(
                        "nonseg_ascii_b8_vs_nonseg_ascii_b16",
                        metric,
                        "nonseg_ascii_b8",
                        group_a[metric],
                        "nonseg_ascii_b16",
                        group_b[metric],
                        n_resamples,
                        "positive difference means B=16 has a larger metric value",
                    )
                )
            hex_payloads = set(stego_frame[stego_frame["protocol_variant"] == "nonseg_hex_nibble_b16"]["payload_name"])
            group_a = stego_frame[
                (stego_frame["protocol_variant"] == "nonseg_ascii_b16")
                & (stego_frame["payload_name"].isin(hex_payloads))
            ]
            group_b = stego_frame[stego_frame["protocol_variant"] == "nonseg_hex_nibble_b16"]
            if not group_a.empty and not group_b.empty:
                rows.append(
                    effect_row(
                        "nonseg_ascii_b16_vs_nonseg_hex_nibble_b16",
                        metric,
                        "nonseg_ascii_b16_hex_payloads",
                        group_a[metric],
                        "nonseg_hex_nibble_b16",
                        group_b[metric],
                        n_resamples,
                        "hex payloads only",
                    )
                )
    if not segmented_frame.empty:
        for metric in ["full_message_mean_log_probability_mean", "full_message_artifact_count_mean", "total_full_message_token_count"]:
            single = segmented_frame[
                segmented_frame["protocol_variant"] == "segmented_hex_single_topic_sentence_tail_filtered"
            ]
            multi = segmented_frame[
                segmented_frame["protocol_variant"] == "segmented_hex_multi_topic_sentence_tail_filtered"
            ]
            if not single.empty and not multi.empty:
                rows.append(
                    effect_row(
                        "segmented_single_topic_vs_multi_topic",
                        metric,
                        "segmented_single_topic",
                        single[metric],
                        "segmented_multi_topic",
                        multi[metric],
                        n_resamples,
                        "segmented hex payloads only",
                    )
                )
            forced_values = segmented_frame["forced_prefix_mean_log_probability_mean"]
            full_values = segmented_frame["full_message_mean_log_probability_mean"]
            rows.append(
                effect_row(
                    "forced_prefix_vs_full_message",
                    "mean_log_probability",
                    "forced_prefix",
                    forced_values,
                    "full_message",
                    full_values,
                    n_resamples,
                    "paired structure is summarized as an independent bootstrap for this pilot",
                )
            )
    legacy_path = project_root / "results" / "rankcloak_segmented_quality_controls" / "segmented_quality_trials.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        if "token_filter_name" in legacy.columns:
            unfiltered = legacy[legacy["token_filter_name"] == "none"]
            filtered = legacy[legacy["token_filter_name"] == SAFE_TEXT_FILTER_V1]
            for metric in ["full_message_mean_log_probability_mean", "full_message_artifact_count_mean"]:
                if metric in legacy.columns and not unfiltered.empty and not filtered.empty:
                    rows.append(
                        effect_row(
                            "legacy_unfiltered_vs_filtered_pilot",
                            metric,
                            "unfiltered",
                            unfiltered[metric],
                            "safe_text_filter_v1",
                            filtered[metric],
                            n_resamples,
                            "imported from results/rankcloak_segmented_quality_controls",
                        )
                    )
    return ordered_frame(rows, EFFECT_SIZE_SUMMARY_COLUMNS)


def save_bar_figure(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path, title: str, ylabel: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        plot_frame = frame[[x_column, y_column]].dropna()
        grouped = plot_frame.groupby(x_column, as_index=False)[y_column].mean()
        ax.bar(grouped[x_column].astype(str), grouped[y_column], color="#476a6f")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_paper_figures(
    output_dir: Path,
    payload_frame: pd.DataFrame,
    rank_frame: pd.DataFrame,
    codec_frame: pd.DataFrame,
    stego_frame: pd.DataFrame,
    segmented_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    detector_frame: pd.DataFrame,
    effect_frame: pd.DataFrame,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    paths.append(
        save_bar_figure(
            codec_frame[codec_frame["applies_to_payload"] == True] if not codec_frame.empty else codec_frame,
            "representation_name",
            "rank_count",
            figures_dir / "paper_payload_representation_rank_counts.png",
            "Payload Representation Rank Counts",
            "Mean rank count",
        )
    )
    if not rank_frame.empty:
        rank_group = rank_frame.groupby("payload_class", as_index=False).agg({"p95_rank": "mean", "max_rank": "max"})
    else:
        rank_group = rank_frame
    path = figures_dir / "paper_direct_subword_rank_pressure.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    if rank_group.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No direct subword rank data", ha="center", va="center")
    else:
        ax.bar(rank_group["payload_class"], rank_group["p95_rank"], label="mean p95 rank", color="#8a5a28")
        ax.scatter(rank_group["payload_class"], rank_group["max_rank"], label="max rank", color="#1f1f1f")
        ax.set_yscale("log")
        ax.set_ylabel("Rank (log scale)")
        ax.set_title("Direct Subword Rank Pressure")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    ascii_frame = stego_frame[stego_frame["protocol_variant"].isin(["nonseg_ascii_b8", "nonseg_ascii_b16"])] if not stego_frame.empty else stego_frame
    path = figures_dir / "paper_alphabet_capacity_quality.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    if ascii_frame.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No ASCII fixed-radix rows", ha="center", va="center")
    else:
        for variant, group in ascii_frame.groupby("protocol_variant"):
            ax.scatter(group["generated_token_count"], group["mean_token_log_probability"], label=variant, alpha=0.75)
        ax.set_xlabel("Generated token count")
        ax.set_ylabel("Mean token log probability")
        ax.set_title("Alphabet Capacity Vs Cover Quality")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    paths.append(
        save_bar_figure(
            stego_frame,
            "protocol_variant",
            "mean_token_log_probability",
            figures_dir / "paper_nonseg_protocol_variant_quality.png",
            "Non-Segmented Protocol Variant Quality",
            "Mean token log probability",
        )
    )
    paths.append(
        save_bar_figure(
            segmented_frame,
            "protocol_variant",
            "full_message_mean_log_probability_mean",
            figures_dir / "paper_segmented_protocol_variant_quality.png",
            "Segmented Protocol Variant Quality",
            "Full-message mean log probability",
        )
    )
    path = figures_dir / "paper_forced_prefix_vs_full_message.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    if segmented_frame.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No segmented rows", ha="center", va="center")
    else:
        grouped = segmented_frame.groupby("protocol_variant", as_index=False).agg(
            {
                "forced_prefix_mean_log_probability_mean": "mean",
                "full_message_mean_log_probability_mean": "mean",
            }
        )
        x = np.arange(len(grouped))
        ax.bar(x - 0.2, grouped["forced_prefix_mean_log_probability_mean"], width=0.4, label="forced prefix")
        ax.bar(x + 0.2, grouped["full_message_mean_log_probability_mean"], width=0.4, label="full message")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["protocol_variant"], rotation=35, ha="right")
        ax.set_ylabel("Mean token log probability")
        ax.set_title("Forced Prefix Vs Full Message")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    combined_artifacts = pd.concat(
        [
            stego_frame[["protocol_variant", "artifact_count_total"]] if not stego_frame.empty else pd.DataFrame(),
            segmented_frame.rename(columns={"full_message_artifact_count_mean": "artifact_count_total"})[
                ["protocol_variant", "artifact_count_total"]
            ]
            if not segmented_frame.empty
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    paths.append(
        save_bar_figure(
            combined_artifacts,
            "protocol_variant",
            "artifact_count_total",
            figures_dir / "paper_artifact_counts_by_variant.png",
            "Artifact Counts By Variant",
            "Mean artifact count",
        )
    )
    recovery_rows = []
    if not stego_frame.empty:
        recovery_rows.extend(stego_frame[["protocol_variant", "exact_recovery"]].to_dict("records"))
    if not segmented_frame.empty:
        recovery_rows.extend(segmented_frame[["protocol_variant", "exact_recovery"]].to_dict("records"))
    recovery_frame = pd.DataFrame(recovery_rows)
    paths.append(
        save_bar_figure(
            recovery_frame,
            "protocol_variant",
            "exact_recovery",
            figures_dir / "paper_recovery_by_variant.png",
            "Exact Recovery By Variant",
            "Recovery rate",
        )
    )
    paths.append(
        save_bar_figure(
            detector_frame,
            "detector_name",
            "auc",
            figures_dir / "paper_detector_auc.png",
            "Detector Baseline AUC",
            "AUC",
        )
    )
    paths.append(
        save_bar_figure(
            effect_frame,
            "comparison_name",
            "difference_b_minus_a",
            figures_dir / "paper_effect_sizes_summary.png",
            "Selected Effect Sizes",
            "Difference B minus A",
        )
    )
    return paths


def write_paper_markdown_outputs(
    output_dir: Path,
    profile: str,
    summary: dict,
    payload_frame: pd.DataFrame,
    stego_frame: pd.DataFrame,
    segmented_frame: pd.DataFrame,
    detector_frame: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    effect_frame: pd.DataFrame,
    figure_paths: Sequence[Path],
    project_root: Path,
) -> List[Path]:
    result_summary_path = output_dir / "PAPER_RESULTS_SUMMARY.md"
    nonseg_passes = int(stego_frame["exact_recovery"].astype(bool).sum()) if not stego_frame.empty else 0
    nonseg_failures = int((~stego_frame["exact_recovery"].astype(bool)).sum()) if not stego_frame.empty else 0
    segmented_passes = int(segmented_frame["exact_recovery"].astype(bool).sum()) if not segmented_frame.empty else 0
    segmented_failures = int((~segmented_frame["exact_recovery"].astype(bool)).sum()) if not segmented_frame.empty else 0
    result_summary = """# RankCloak Paper Results Summary

## Overview

Profile: `{profile}`

This run evaluates deterministic synthetic payloads under exact-copy conditions. It does not claim encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or undetectability.

## Exact Recovery Summary

- Non-segmented recovery: {nonseg_passes} pass, {nonseg_failures} fail.
- Segmented recovery: {segmented_passes} pass, {segmented_failures} fail.

## Payload Representation Results

See `paper_codec_comparison.csv` and `paper_payloads.csv`.

## Direct Subword Rank Pressure

See `paper_rank_pressure.csv`. Direct subword rank pressure is a diagnostic baseline and does not generate cover text in this suite.

## Non-Segmented Bounded-Rank Results

See `paper_stegotext_trials.csv`. The main non-segmented variants are `nonseg_ascii_b8`, `nonseg_ascii_b16`, and `nonseg_hex_nibble_b16`.

## Hex-Nibble Results

Hex-nibble rows apply only to payloads marked `is_hex_like`.

## Prompt And Alphabet-Size Results

See `paper_cover_text_features.csv` and the figure index for prompt-family and alphabet-size comparisons.

## Segmented Protocol Results

See `paper_segmented_trials.csv` and `paper_segmented_messages.jsonl`.

## Lead-In Segmented Variant Results

The lead-in variant is implemented as `segmented_hex_multi_topic_leadin8_sentence_tail_filtered`. The decoder ignores the greedy lead-in, decodes the forced span, and ignores the tail.

## Forced-Prefix Versus Full-Message Results

Segmented rows separate forced-prefix metrics from full-message metrics. Full-message quality can be tail-driven and should not be treated as payload-bearing-token quality.

## Safe-Text Filter And Artifact Results

Segmented paper variants use `safe_text_filter_v1`, a deterministic heuristic token filter. See artifact columns in `paper_cover_text_features.csv`.

## Detector Baseline Results

Detector rows: {detector_rows}. These are lightweight feature baselines only and do not establish strong steganalysis.

## Statistical Uncertainty

Bootstrap summary rows: {stat_rows}. Effect-size rows: {effect_rows}.

## Recommended Main Paper Claims

- The suite measures exact-copy recovery and cover-quality proxies for deterministic synthetic artifacts.
- Bounded-rank and hex-nibble encodings provide controlled rank pressure compared with direct subword ranks.
- Segmented variants require separate forced-prefix and full-message metrics.

## Claims Not Supported

- No cryptographic security claim.
- No undetectability claim.
- No edit robustness claim.
- No cross-model portability claim.

## Limitations

The current run is `{profile}`. If this is `paper-main-pilot`, treat it as a validation run before the larger frozen `paper-main` matrix.
""".format(
        profile=profile,
        nonseg_passes=nonseg_passes,
        nonseg_failures=nonseg_failures,
        segmented_passes=segmented_passes,
        segmented_failures=segmented_failures,
        detector_rows=len(detector_frame),
        stat_rows=len(statistics_frame),
        effect_rows=len(effect_frame),
    )
    result_summary_path.write_text(result_summary, encoding="utf-8")

    comparison_path = output_dir / "PAPER_COMPARISON_TABLES.md"
    payload_counts = payload_frame["payload_class"].value_counts().to_dict() if not payload_frame.empty else {}
    recovery_summary_rows = []
    if not stego_frame.empty:
        recovery_summary_rows.append(stego_frame.groupby("protocol_variant")["exact_recovery"].agg(["count", "sum"]).reset_index())
    if not segmented_frame.empty:
        recovery_summary_rows.append(segmented_frame.groupby("protocol_variant")["exact_recovery"].agg(["count", "sum"]).reset_index())
    recovery_table = pd.concat(recovery_summary_rows, ignore_index=True) if recovery_summary_rows else pd.DataFrame()
    lines = [
        "# Paper Comparison Tables",
        "",
        "## Table A: Payload Classes And Counts",
        "",
        "| Payload class | Count |",
        "| --- | ---: |",
    ]
    for payload_class, count in sorted(payload_counts.items()):
        lines.append("| `{}` | {} |".format(payload_class, count))
    lines.extend(
        [
            "",
            "## Table B: Protocol Variants",
            "",
            "| Protocol variant | Role |",
            "| --- | --- |",
        ]
    )
    for variant in PAPER_PROTOCOL_VARIANTS:
        lines.append("| `{}` | implemented paper-suite variant |".format(variant))
    lines.extend(["", "## Table C: Recovery Summary By Protocol Variant", "", "| Variant | Trials | Passes |", "| --- | ---: | ---: |"])
    if not recovery_table.empty:
        for _, row in recovery_table.iterrows():
            lines.append("| `{}` | {} | {} |".format(row["protocol_variant"], int(row["count"]), int(row["sum"])))
    lines.extend(
        [
            "",
            "## Table D: Payload Representation Rank Counts",
            "",
            "See `paper_codec_comparison.csv`.",
            "",
            "## Table E: Cover-Quality Metrics By Protocol Variant",
            "",
            "See `statistical_summary.csv`.",
            "",
            "## Table F: Forced-Prefix Versus Full-Message Metrics",
            "",
            "See `paper_segmented_trials.csv`.",
            "",
            "## Table G: Detector Results",
            "",
            "See `detector_baseline.csv`.",
            "",
            "## Table H: Limitations And Unsupported Claims",
            "",
            "- Synthetic payloads only.",
            "- Exact-copy conditions only.",
            "- No encryption, key exchange, authentication, signing, or cryptographic security claim.",
            "- No undetectability claim.",
        ]
    )
    comparison_path.write_text("\n".join(lines), encoding="utf-8")

    figure_index_path = output_dir / "PAPER_FIGURE_INDEX.md"
    figure_lines = ["# Paper Figure Index", ""]
    for path in figure_paths:
        relative = repo_relative_path(path, project_root)
        figure_lines.extend(
            [
                "## `{}`".format(relative),
                "",
                "- caption draft: RankCloak paper-suite figure generated from the current run.",
                "- source CSV/JSONL file: see filename-specific plotting code in `rankcloak/paper_suite.py`.",
                "- interpretation: use as pilot evidence unless the profile is `paper-main`.",
                "- status: generated.",
                "",
            ]
        )
    figure_index_path.write_text("\n".join(figure_lines), encoding="utf-8")
    return [result_summary_path, comparison_path, figure_index_path]


def write_summary_markdown(output_dir: Path, summary: dict) -> Path:
    path = output_dir / "SUMMARY.md"
    text = """# RankCloak Paper Suite Summary

- Profile: {profile}
- Model status: {model_status}
- Payload count: {payload_count}
- Non-segmented trials: {nonseg_trial_count}
- Segmented trials: {segmented_trial_count}
- Baseline rows: {baseline_count}
- Recovery: {passes} pass, {failures} fail
- Detector rows: {detector_rows}
- Statistical rows: {stat_rows}
- Effect-size rows: {effect_rows}

## Scope

This is an empirical exact-copy measurement study over deterministic synthetic payloads. It is not encryption, key exchange, authentication, signing, digital signatures, credential handling, cryptographic security, or an undetectability claim.
""".format(
        profile=summary["profile"],
        model_status="loaded" if summary["model_loaded"] else "not loaded",
        payload_count=summary["payload_count"],
        nonseg_trial_count=summary["nonseg_trial_count"],
        segmented_trial_count=summary["segmented_trial_count"],
        baseline_count=summary["baseline_count"],
        passes=summary["recovery_pass_count"],
        failures=summary["recovery_fail_count"],
        detector_rows=summary["detector_rows"],
        stat_rows=summary["statistical_summary_rows"],
        effect_rows=summary["effect_size_rows"],
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_paper_suite(
    profile: str,
    output_dir: Path,
    project_root: Path,
    model: Any,
    model_path: Optional[Path],
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    command_line_args: Sequence[str],
    model_loaded: bool,
    model_status: str,
    model_error: Optional[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = payloads_for_profile(profile)
    prompt_names = prompt_names_for_profile(profile)
    cover_prompts = cover_prompt_dictionary()
    n_resamples = 1000 if profile == "paper-main-pilot" else 2000

    payload_frame = write_frame(output_dir / "paper_payloads.csv", paper_payload_rows(payloads), PAPER_PAYLOAD_COLUMNS)
    rank_frame = write_frame(
        output_dir / "paper_rank_pressure.csv",
        run_paper_rank_pressure(payloads, model, model_repo_id, model_filename, model_path_relative),
        PAPER_RANK_PRESSURE_COLUMNS,
    )
    codec_frame = write_frame(
        output_dir / "paper_codec_comparison.csv",
        build_paper_codec_comparison(payloads),
        PAPER_CODEC_COMPARISON_COLUMNS,
    )
    stego_rows, nonseg_examples, nonseg_feature_rows = run_paper_nonseg_trials(
        payloads,
        prompt_names,
        cover_prompts,
        model,
        model_repo_id,
        model_filename,
        model_path_relative,
    )
    stego_frame = write_frame(output_dir / "paper_stegotext_trials.csv", stego_rows, PAPER_STEGOTEXT_TRIAL_COLUMNS)
    write_jsonl(output_dir / "paper_nonseg_examples.jsonl", nonseg_examples)
    segmented_rows, segmented_message_rows, segmented_feature_rows = run_paper_segmented_trials(
        payloads,
        cover_prompts,
        model,
        model_repo_id,
        model_filename,
        model_path_relative,
    )
    segmented_frame = write_frame(output_dir / "paper_segmented_trials.csv", segmented_rows, PAPER_SEGMENTED_TRIAL_COLUMNS)
    write_jsonl(output_dir / "paper_segmented_messages.jsonl", segmented_message_rows)

    baseline_targets = build_baseline_targets(nonseg_feature_rows + segmented_feature_rows)
    baseline_rows, baseline_feature_rows = run_paper_baselines(
        baseline_targets,
        cover_prompts,
        model,
        model_repo_id,
        model_filename,
    )
    write_jsonl(output_dir / "paper_baseline_examples.jsonl", baseline_rows)
    feature_rows = nonseg_feature_rows + segmented_feature_rows + baseline_feature_rows
    feature_frame = write_frame(output_dir / "paper_cover_text_features.csv", feature_rows, PAPER_COVER_TEXT_FEATURE_COLUMNS)

    detector_dataset = prepare_detector_dataset(feature_frame)
    detector_dataset_frame = ordered_frame(detector_dataset.to_dict("records"), DETECTOR_DATASET_COLUMNS)
    detector_dataset_frame.to_csv(output_dir / "detector_dataset.csv", index=False)
    detector_frame = ordered_frame(
        run_detector_baselines(detector_dataset_frame).to_dict("records"),
        DETECTOR_BASELINE_COLUMNS,
    )
    detector_frame.to_csv(output_dir / "detector_baseline.csv", index=False)

    statistics_frame = build_statistical_summary(stego_frame, segmented_frame, n_resamples)
    statistics_frame.to_csv(output_dir / "statistical_summary.csv", index=False)
    effect_frame = build_effect_size_summary(stego_frame, segmented_frame, project_root, n_resamples)
    effect_frame.to_csv(output_dir / "effect_size_summary.csv", index=False)

    figure_paths = write_paper_figures(
        output_dir,
        payload_frame,
        rank_frame,
        codec_frame,
        stego_frame,
        segmented_frame,
        feature_frame,
        detector_frame,
        effect_frame,
    )
    markdown_paths = write_paper_markdown_outputs(
        output_dir,
        profile,
        {},
        payload_frame,
        stego_frame,
        segmented_frame,
        detector_frame,
        statistics_frame,
        effect_frame,
        figure_paths,
        project_root,
    )
    manifest_path = output_dir / "MANIFEST.json"
    write_manifest(
        output_path=manifest_path,
        project_root=project_root,
        profile=profile,
        output_dir=output_dir,
        command_line_args=command_line_args,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
    )

    recovery_values = []
    if not stego_frame.empty:
        recovery_values.extend(list(stego_frame["exact_recovery"].astype(bool)))
    if not segmented_frame.empty:
        recovery_values.extend(list(segmented_frame["exact_recovery"].astype(bool)))
    generated_files = [
        output_dir / "paper_payloads.csv",
        output_dir / "paper_rank_pressure.csv",
        output_dir / "paper_codec_comparison.csv",
        output_dir / "paper_stegotext_trials.csv",
        output_dir / "paper_nonseg_examples.jsonl",
        output_dir / "paper_segmented_trials.csv",
        output_dir / "paper_segmented_messages.jsonl",
        output_dir / "paper_baseline_examples.jsonl",
        output_dir / "paper_cover_text_features.csv",
        output_dir / "detector_dataset.csv",
        output_dir / "detector_baseline.csv",
        output_dir / "statistical_summary.csv",
        output_dir / "effect_size_summary.csv",
        *figure_paths,
        *markdown_paths,
        manifest_path,
        output_dir / "summary.json",
        output_dir / "SUMMARY.md",
    ]
    summary = {
        "profile": profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_loaded": bool(model_loaded),
        "payload_count": len(payloads),
        "payload_class_counts": payload_class_counts(payloads),
        "protocol_variants": list(PAPER_PROTOCOL_VARIANTS),
        "nonseg_trial_count": int(len(stego_frame)),
        "segmented_trial_count": int(len(segmented_frame)),
        "baseline_count": int(len(baseline_rows)),
        "recovery_pass_count": int(sum(recovery_values)),
        "recovery_fail_count": int(len(recovery_values) - sum(recovery_values)),
        "detector_rows": int(len(detector_dataset_frame)),
        "detector_results_available": not detector_frame.empty,
        "statistical_summary_rows": int(len(statistics_frame)),
        "effect_size_rows": int(len(effect_frame)),
        "generated_result_files": [repo_relative_path(path, project_root) for path in generated_files],
        "important_notes": [
            "All payloads are deterministic synthetic examples.",
            "Exact-copy conditions are required.",
            "No encryption, key exchange, authentication, signing, credential handling, cryptographic security, or undetectability is claimed.",
            "K_common is assumed for protocol variants and includes model, tokenizer, quantization, prompt templates, codec, segmentation, tail, filter, and decode rules.",
            "Model status: {}".format(model_status),
            "Model error: {}".format(model_error) if model_error else "Model loaded or was not required.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir, summary)
    return summary


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def run_paper_analysis(
    output_dir: Path,
    project_root: Path,
    command_line_args: Sequence[str],
    model_repo_id: str,
    model_filename: str,
    model_path: Optional[Path],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        project_root / "results" / "rankcloak_small_full",
        project_root / "results" / "rankcloak_strong_prompt_sweep",
        project_root / "results" / "rankcloak_dialogue_key_pilot",
        project_root / "results" / "rankcloak_payload_granularity_pilot",
        project_root / "results" / "rankcloak_segmented_protocol_pilot",
        project_root / "results" / "rankcloak_segmented_quality_controls",
        project_root / "results" / "rankcloak_paper_main_pilot",
        project_root / "results" / "rankcloak_paper_main",
    ]
    recovery_rows = []
    payload_representation_rows = []
    prompt_quality_rows = []
    segmented_rows = []
    detector_rows = []
    for directory in source_dirs:
        if not directory.exists():
            continue
        summary = read_json_if_exists(directory / "summary.json")
        recovery_rows.append(
            {
                "result_directory": repo_relative_path(directory, project_root),
                "profile": summary.get("profile"),
                "model_loaded": summary.get("model_loaded"),
                "codec_roundtrip_pass_count": summary.get("codec_roundtrip_pass_count"),
                "codec_roundtrip_fail_count": summary.get("codec_roundtrip_fail_count"),
                "stegotext_recovery_pass_count": summary.get("stegotext_recovery_pass_count"),
                "stegotext_recovery_fail_count": summary.get("stegotext_recovery_fail_count"),
                "response_recovery_pass_count": summary.get("response_recovery_pass_count"),
                "response_recovery_fail_count": summary.get("response_recovery_fail_count"),
                "recovery_pass_count": summary.get("recovery_pass_count"),
                "recovery_fail_count": summary.get("recovery_fail_count"),
            }
        )
        for filename in ("payload_granularity_comparison.csv", "paper_codec_comparison.csv"):
            path = directory / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame["result_directory"] = repo_relative_path(directory, project_root)
                payload_representation_rows.extend(frame.to_dict("records"))
        for filename in ("cover_text_features.csv", "paper_cover_text_features.csv"):
            path = directory / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame["result_directory"] = repo_relative_path(directory, project_root)
                prompt_quality_rows.extend(frame.to_dict("records"))
        for filename in ("segmented_protocol_trials.csv", "segmented_quality_trials.csv", "paper_segmented_trials.csv"):
            path = directory / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame["result_directory"] = repo_relative_path(directory, project_root)
                segmented_rows.extend(frame.to_dict("records"))
        path = directory / "detector_baseline.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["result_directory"] = repo_relative_path(directory, project_root)
            detector_rows.extend(frame.to_dict("records"))

    recovery_frame = pd.DataFrame(recovery_rows)
    payload_frame = pd.DataFrame(payload_representation_rows)
    prompt_frame = pd.DataFrame(prompt_quality_rows)
    segmented_frame = pd.DataFrame(segmented_rows)
    detector_frame = pd.DataFrame(detector_rows)
    recovery_frame.to_csv(output_dir / "all_recovery_summary.csv", index=False)
    payload_frame.to_csv(output_dir / "all_payload_representation_summary.csv", index=False)
    prompt_frame.to_csv(output_dir / "all_prompt_quality_summary.csv", index=False)
    segmented_frame.to_csv(output_dir / "all_segmented_protocol_summary.csv", index=False)
    detector_frame.to_csv(output_dir / "all_detector_summary.csv", index=False)

    figure_paths = [
        save_bar_figure(
            recovery_frame.fillna(0),
            "profile",
            "recovery_pass_count" if "recovery_pass_count" in recovery_frame.columns else "stegotext_recovery_pass_count",
            figures_dir / "analysis_recovery_by_profile.png",
            "Recovery Summary By Profile",
            "Recorded pass count",
        ),
        save_bar_figure(
            payload_frame,
            "representation_name",
            "rank_count",
            figures_dir / "analysis_payload_representation_rank_count.png",
            "Payload Representation Rank Count",
            "Mean rank count",
        ),
        save_bar_figure(
            prompt_frame[prompt_frame.get("source_type", "") == "rankcloak"] if "source_type" in prompt_frame else prompt_frame,
            "prompt_family",
            "mean_token_log_probability",
            figures_dir / "analysis_prompt_quality.png",
            "Prompt Quality Summary",
            "Mean token log probability",
        ),
        save_bar_figure(
            segmented_frame,
            "condition_name" if "condition_name" in segmented_frame.columns else "protocol_variant",
            "full_message_mean_log_probability_mean" if "full_message_mean_log_probability_mean" in segmented_frame.columns else "mean_token_log_probability",
            figures_dir / "analysis_segmented_summary.png",
            "Segmented Protocol Summary",
            "Mean log probability",
        ),
    ]
    summary_md = output_dir / "PAPER_ANALYSIS_SUMMARY.md"
    summary_md.write_text(
        """# RankCloak Paper Analysis Summary

This profile aggregates existing pilot and paper-suite result directories without running model generation.

- Source directories inspected: {source_count}
- Recovery rows: {recovery_rows}
- Payload representation rows: {payload_rows}
- Prompt quality rows: {prompt_rows}
- Segmented rows: {segmented_rows}
- Detector rows: {detector_rows}

Use this output to decide which pilot results belong in the main manuscript versus supplemental material. All claims remain limited to deterministic synthetic payloads and exact-copy conditions.
""".format(
            source_count=sum(1 for directory in source_dirs if directory.exists()),
            recovery_rows=len(recovery_frame),
            payload_rows=len(payload_frame),
            prompt_rows=len(prompt_frame),
            segmented_rows=len(segmented_frame),
            detector_rows=len(detector_frame),
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "MANIFEST.json"
    write_manifest(
        output_path=manifest_path,
        project_root=project_root,
        profile="paper-analysis",
        output_dir=output_dir,
        command_line_args=command_line_args,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
    )
    generated_files = [
        output_dir / "all_recovery_summary.csv",
        output_dir / "all_payload_representation_summary.csv",
        output_dir / "all_prompt_quality_summary.csv",
        output_dir / "all_segmented_protocol_summary.csv",
        output_dir / "all_detector_summary.csv",
        summary_md,
        manifest_path,
        *figure_paths,
        output_dir / "summary.json",
    ]
    summary = {
        "profile": "paper-analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_loaded": False,
        "source_directories_found": [repo_relative_path(directory, project_root) for directory in source_dirs if directory.exists()],
        "recovery_rows": int(len(recovery_frame)),
        "payload_representation_rows": int(len(payload_frame)),
        "prompt_quality_rows": int(len(prompt_frame)),
        "segmented_rows": int(len(segmented_frame)),
        "detector_rows": int(len(detector_frame)),
        "generated_result_files": [repo_relative_path(path, project_root) for path in generated_files],
        "important_notes": [
            "Aggregation profile only; no model generation was run.",
            "Existing pilot and paper-suite artifacts are not overwritten.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
