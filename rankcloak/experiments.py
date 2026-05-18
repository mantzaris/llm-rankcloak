"""Experiment runner for RankCloak research profiles."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .baselines import generate_greedy_baseline
from .metrics import (
    extract_text_features,
    summarize_optional_ranks,
    summarize_rank_sequence,
)
from .model_io import (
    FALLBACK_LLAMA3_SPEC,
    PREFERRED_LLAMA3_SPEC,
    download_llama3_gguf,
    existing_llama3_model_path,
    load_llama_cpp_model,
    make_context_token_ids,
)
from .plotting import (
    plot_dialogue_prompt_quality_scatter,
    plot_dialogue_prompt_repetition,
    plot_cover_length_vs_rank_alphabet,
    plot_cover_text_feature_comparison,
    plot_payload_representation_rank_count,
    plot_rank_summary_direct_subword,
    plot_recovery_by_cover_prompt_and_alphabet,
    plot_strong_prompt_length,
    plot_strong_prompt_mean_logprob,
    plot_strong_prompt_rank_pressure,
    plot_strong_prompt_recovery,
    plot_token_count_by_payload,
)
from .prompts import cover_prompt_dictionary, prompt_family
from .rank_codec import (
    SUPPORTED_ALPHABET_SIZES,
    codec_roundtrip_rows,
    decode_bounded_ranks_to_bytes,
    decode_hex_nibble_ranks_to_text,
    direct_subword_ranks_for_text,
    encode_bytes_to_bounded_ranks,
    encode_hex_nibbles_to_ranks,
    generate_token_ids_from_ranks,
    is_hex_text,
    recover_ranks_from_generated_ids,
    test_stable_rank_ordering,
)
from .reproducibility import repo_relative_path, write_manifest
from .schemas import (
    BASELINE_COVER_COLUMNS,
    CODEC_ROUNDTRIP_COLUMNS,
    COVER_TEXT_FEATURE_COLUMNS,
    STEGOTEXT_RECOVERY_COLUMNS,
)
from .synthetic_payloads import SyntheticPayload, generate_synthetic_payloads
from .tokenization_audit import audit_payload_tokenization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rankcloak_crypto_artifact_exploration"


PROFILE_CONFIGS = {
    "smoke": {
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "payload_names": ["sha256_public_test_string"],
        "cover_prompt_names": ["play_dialogue", "recipe_blog"],
        "alphabet_sizes": [16, 32],
        "default_max_payload_bytes": 8,
        "requires_stegotext": True,
        "baseline_token_cap": 32,
    },
    "small": {
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "payload_names": [
            "sha256_public_test_string",
            "random_128_bit_hex",
            "random_256_bit_hex",
            "synthetic_uuid_v4_like",
        ],
        "cover_prompt_names": [
            "play_dialogue",
            "recipe_blog",
            "forum_reply",
            "technical_documentation",
        ],
        "alphabet_sizes": [8, 16, 32, 64],
        "default_max_payload_bytes": None,
        "requires_stegotext": True,
        "baseline_token_cap": 96,
    },
    "codec-only": {
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "payload_names": None,
        "cover_prompt_names": [],
        "alphabet_sizes": SUPPORTED_ALPHABET_SIZES,
        "default_max_payload_bytes": None,
        "requires_stegotext": False,
        "baseline_token_cap": 0,
    },
    "audit-only": {
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "payload_names": None,
        "cover_prompt_names": [],
        "alphabet_sizes": SUPPORTED_ALPHABET_SIZES,
        "default_max_payload_bytes": None,
        "requires_stegotext": False,
        "baseline_token_cap": 0,
    },
    "strong-prompts-pilot": {
        "default_output_dir": PROJECT_ROOT / "results" / "rankcloak_strong_prompt_pilot",
        "payload_names": [
            "sha256_public_test_string",
            "random_128_bit_hex",
        ],
        "cover_prompt_names": [
            "recipe_blog",
            "recipe_long_specific",
            "biology_long_specific",
            "car_buying_long_specific",
        ],
        "alphabet_sizes": [16, 32],
        "default_max_payload_bytes": None,
        "requires_stegotext": True,
        "baseline_token_cap": 96,
        "write_prompt_comparison": True,
    },
    "strong-prompts": {
        "default_output_dir": PROJECT_ROOT / "results" / "rankcloak_strong_prompt_sweep",
        "payload_names": [
            "sha256_public_test_string",
            "random_128_bit_hex",
            "synthetic_uuid_v4_like",
        ],
        "cover_prompt_names": [
            "recipe_blog",
            "recipe_long_specific",
            "biology_long_specific",
            "car_buying_long_specific",
            "forum_reply",
        ],
        "alphabet_sizes": [8, 16, 32, 64],
        "default_max_payload_bytes": None,
        "requires_stegotext": True,
        "baseline_token_cap": 96,
        "write_prompt_comparison": True,
    },
    "dialogue-key-pilot": {
        "default_output_dir": PROJECT_ROOT / "results" / "rankcloak_dialogue_key_pilot",
        "payload_names": [
            "sha256_public_test_string",
            "random_128_bit_hex",
        ],
        "cover_prompt_names": [
            "recipe_blog",
            "recipe_long_specific",
            "recipe_dialogue_specific",
            "recipe_forum_exchange_specific",
            "car_buying_dialogue_specific",
            "biology_tutor_dialogue_specific",
        ],
        "alphabet_sizes": [8, 16],
        "default_max_payload_bytes": None,
        "requires_stegotext": True,
        "baseline_token_cap": 96,
        "write_dialogue_comparison": True,
    },
    "payload-granularity-pilot": {
        "default_output_dir": PROJECT_ROOT / "results" / "rankcloak_payload_granularity_pilot",
        "payload_names": [
            "sha256_public_test_string",
            "random_128_bit_hex",
        ],
        "cover_prompt_names": [],
        "alphabet_sizes": [8, 16],
        "default_max_payload_bytes": None,
        "requires_stegotext": False,
        "baseline_token_cap": 0,
        "write_payload_granularity": True,
    },
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def ordered_frame(rows: List[dict], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    leading_columns = list(columns)
    extra_columns = [column for column in frame.columns if column not in leading_columns]
    return frame.reindex(columns=leading_columns + extra_columns)


def resolve_model_identity(model_path: Optional[Path]) -> tuple:
    if model_path is not None and str(model_path).endswith(PREFERRED_LLAMA3_SPEC.filename):
        return PREFERRED_LLAMA3_SPEC.repo_id, PREFERRED_LLAMA3_SPEC.filename
    if model_path is not None and str(model_path).endswith(FALLBACK_LLAMA3_SPEC.filename):
        return FALLBACK_LLAMA3_SPEC.repo_id, FALLBACK_LLAMA3_SPEC.filename
    return PREFERRED_LLAMA3_SPEC.repo_id, PREFERRED_LLAMA3_SPEC.filename


def selected_payloads(payloads: Sequence[SyntheticPayload], payload_names: Optional[Sequence[str]]) -> List[SyntheticPayload]:
    if payload_names is None:
        return list(payloads)
    by_name = {payload.name: payload for payload in payloads}
    missing = [name for name in payload_names if name not in by_name]
    if missing:
        raise KeyError("Unknown payload names: {}".format(", ".join(missing)))
    return [by_name[name] for name in payload_names]


def payload_bytes_for_profile(payload: SyntheticPayload, byte_limit: Optional[int]) -> tuple:
    if byte_limit is None or byte_limit >= len(payload.bytes_value):
        return payload.bytes_value, "full payload"
    return payload.bytes_value[:byte_limit], "first {} bytes".format(byte_limit)


def build_prompt_metadata(
    cover_prompts: Dict[str, str],
    cover_prompt_names: Sequence[str],
    model: Any,
) -> Dict[str, dict]:
    metadata = {}
    for prompt_name in cover_prompt_names:
        prompt_text = cover_prompts[prompt_name]
        prompt_length_tokens = None
        if model is not None:
            try:
                prompt_length_tokens = len(make_context_token_ids(model, prompt_text))
            except Exception:
                prompt_length_tokens = None
        metadata[prompt_name] = {
            "prompt_family": prompt_family(prompt_name),
            "prompt_length_characters": len(prompt_text),
            "prompt_length_tokens": prompt_length_tokens,
        }
    return metadata


def load_model_for_profile(
    profile: str,
    model_path: Optional[str],
    skip_model_download: bool,
) -> tuple:
    if profile == "codec-only":
        return None, existing_llama3_model_path(), "not_requested", None, 0.0

    resolved_model_path = Path(model_path) if model_path else existing_llama3_model_path()
    if resolved_model_path is None and not skip_model_download:
        try:
            resolved_model_path = download_llama3_gguf()
        except Exception as exc:
            return None, None, "unavailable", str(exc), 0.0

    if resolved_model_path is None:
        return None, None, "unavailable", "No local GGUF model found.", 0.0

    started_at = time.perf_counter()
    try:
        model = load_llama_cpp_model(model_path=resolved_model_path, n_ctx=2048, logits_all=True)
        return model, resolved_model_path, "loaded", None, time.perf_counter() - started_at
    except Exception as exc:
        return None, resolved_model_path, "unavailable", str(exc), time.perf_counter() - started_at


def run_tokenization_audit(payloads: Sequence[SyntheticPayload], model: Any, output_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(audit_payload_tokenization(payloads, model=model))
    frame.to_csv(output_dir / "tokenization_audit.csv", index=False)
    return frame


def run_rank_statistics(
    payloads: Sequence[SyntheticPayload],
    model: Any,
    model_error: Optional[str],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    if model is not None:
        for payload in payloads:
            trace = direct_subword_ranks_for_text(model, payload.text)
            row = summarize_rank_sequence(payload.name, trace["ranks"])
            row.update({"payload_kind": payload.kind, "model_loaded": True, "skip_reason": None})
            rows.append(row)
    else:
        for payload in payloads:
            row = summarize_rank_sequence(payload.name, [])
            row.update(
                {
                    "payload_kind": payload.kind,
                    "model_loaded": False,
                    "skip_reason": model_error or "model unavailable",
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "rank_statistics.csv", index=False)
    return frame


def run_codec_roundtrips(
    payloads: Sequence[SyntheticPayload],
    alphabet_sizes: Sequence[int],
    output_dir: Path,
) -> pd.DataFrame:
    rows = codec_roundtrip_rows(payloads, alphabet_sizes)
    frame = ordered_frame(rows, CODEC_ROUNDTRIP_COLUMNS)
    frame.to_csv(output_dir / "codec_roundtrip_trials.csv", index=False)
    return frame


def run_stegotext_trials(
    payloads: Sequence[SyntheticPayload],
    cover_prompts: Dict[str, str],
    cover_prompt_names: Sequence[str],
    alphabet_sizes: Sequence[int],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    model_path_relative: Optional[str],
    payload_byte_limit: Optional[int],
    prompt_metadata: Dict[str, dict],
    output_dir: Path,
) -> tuple:
    trial_rows = []
    cover_examples = []
    if model is None:
        frame = ordered_frame([], STEGOTEXT_RECOVERY_COLUMNS)
        frame.to_csv(output_dir / "stegotext_recovery_trials.csv", index=False)
        write_jsonl(output_dir / "cover_examples.jsonl", [])
        return frame, cover_examples

    for payload in payloads:
        payload_bytes, slice_description = payload_bytes_for_profile(payload, payload_byte_limit)
        for alphabet_size in alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(payload_bytes, alphabet_size)
            for prompt_name in cover_prompt_names:
                prompt = cover_prompts[prompt_name]
                context_ids = make_context_token_ids(model, prompt)
                generation_started_at = time.perf_counter()
                generated = generate_token_ids_from_ranks(model, context_ids, encoded["ranks"])
                generation_seconds = time.perf_counter() - generation_started_at

                recovery_started_at = time.perf_counter()
                recovered = recover_ranks_from_generated_ids(
                    model, context_ids, generated["generated_token_ids"]
                )
                recovery_seconds = time.perf_counter() - recovery_started_at
                decoded_bytes = decode_bounded_ranks_to_bytes(recovered["ranks"], encoded["metadata"])
                exact_recovery = decoded_bytes == payload_bytes
                rank_summary = summarize_optional_ranks(recovered["ranks"])
                generated_text = generated["generated_text"]
                row = {
                    "payload_name": payload.name,
                    "payload_kind": payload.kind,
                    "payload_byte_length": len(payload_bytes),
                    "payload_slice_description": slice_description,
                    "encoding_name": "fixed_radix_bits",
                    "alphabet_size": int(alphabet_size),
                    "cover_prompt_name": prompt_name,
                    "prompt_family": prompt_metadata[prompt_name]["prompt_family"],
                    "prompt_length_characters": prompt_metadata[prompt_name]["prompt_length_characters"],
                    "prompt_length_tokens": prompt_metadata[prompt_name]["prompt_length_tokens"],
                    "rank_count": len(encoded["ranks"]),
                    "generated_token_count": len(generated["generated_token_ids"]),
                    "generated_character_count": len(generated_text),
                    "exact_recovery": exact_recovery,
                    "generation_seconds": generation_seconds,
                    "recovery_seconds": recovery_seconds,
                    "mean_generated_rank": rank_summary["mean_generated_rank"],
                    "median_generated_rank": rank_summary["median_generated_rank"],
                    "p95_generated_rank": rank_summary["p95_generated_rank"],
                    "max_generated_rank": rank_summary["max_generated_rank"],
                    "fraction_generated_rank_le_16": rank_summary["fraction_generated_rank_le_16"],
                    "fraction_generated_rank_le_64": rank_summary["fraction_generated_rank_le_64"],
                    "model_repo_id": model_repo_id,
                    "model_filename": model_filename,
                    "model_path_relative": model_path_relative,
                    "notes": "exact-copy bounded-rank recovery",
                }
                trial_rows.append(row)
                cover_examples.append(
                    {
                        "source_type": "rankcloak",
                        "payload_name": payload.name,
                        "payload_kind": payload.kind,
                        "payload_byte_length": len(payload_bytes),
                        "payload_slice_description": slice_description,
                        "encoding_name": "fixed_radix_bits",
                        "alphabet_size": int(alphabet_size),
                        "cover_prompt_name": prompt_name,
                        "prompt_family": prompt_metadata[prompt_name]["prompt_family"],
                        "prompt_length_characters": prompt_metadata[prompt_name]["prompt_length_characters"],
                        "prompt_length_tokens": prompt_metadata[prompt_name]["prompt_length_tokens"],
                        "rank_count": len(encoded["ranks"]),
                        "generated_text": generated_text,
                        "generated_token_ids": generated["generated_token_ids"],
                        "generated_token_count": len(generated["generated_token_ids"]),
                        "generated_character_count": len(generated_text),
                        "recovered_ranks": recovered["ranks"],
                        "token_log_probabilities": generated.get("token_log_probabilities", []),
                        "exact_recovery": exact_recovery,
                        "generation_seconds": generation_seconds,
                        "recovery_seconds": recovery_seconds,
                        "model_repo_id": model_repo_id,
                        "model_filename": model_filename,
                        "model_path_relative": model_path_relative,
                        "notes": "synthetic payload; not encryption",
                    }
                )

    frame = ordered_frame(trial_rows, STEGOTEXT_RECOVERY_COLUMNS)
    frame.to_csv(output_dir / "stegotext_recovery_trials.csv", index=False)
    write_jsonl(output_dir / "cover_examples.jsonl", cover_examples)
    return frame, cover_examples


def baseline_token_target(stegotext_frame: pd.DataFrame, cap: int) -> int:
    if cap <= 0 or stegotext_frame.empty or "rank_count" not in stegotext_frame:
        return 0
    rank_counts = stegotext_frame["rank_count"].dropna()
    if rank_counts.empty:
        return 0
    return max(1, min(int(rank_counts.median()), cap))


def run_baselines(
    cover_prompts: Dict[str, str],
    cover_prompt_names: Sequence[str],
    model: Any,
    model_repo_id: str,
    model_filename: str,
    target_token_count: int,
    prompt_metadata: Dict[str, dict],
    output_dir: Path,
) -> tuple:
    baseline_rows = []
    baseline_examples = []
    if model is None or target_token_count <= 0:
        frame = ordered_frame([], BASELINE_COVER_COLUMNS)
        write_jsonl(output_dir / "baseline_cover_examples.jsonl", [])
        return frame, baseline_examples

    for prompt_name in cover_prompt_names:
        context_ids = make_context_token_ids(model, cover_prompts[prompt_name])
        generated = generate_greedy_baseline(model, context_ids, target_token_count)
        row = {
            "cover_prompt_name": prompt_name,
            "prompt_family": prompt_metadata[prompt_name]["prompt_family"],
            "prompt_length_characters": prompt_metadata[prompt_name]["prompt_length_characters"],
            "prompt_length_tokens": prompt_metadata[prompt_name]["prompt_length_tokens"],
            "baseline_mode": "greedy",
            "generated_text": generated["generated_text"],
            "generated_token_count": generated["generated_token_count"],
            "generated_character_count": generated["generated_character_count"],
            "generation_seconds": generated["generation_seconds"],
            "model_repo_id": model_repo_id,
            "model_filename": model_filename,
            "notes": "greedy baseline at approximate RankCloak token length",
        }
        baseline_rows.append(row)
        baseline_examples.append(
            dict(
                row,
                source_type="baseline",
                generated_token_ids=generated["generated_token_ids"],
                ranks=generated["ranks"],
                token_log_probabilities=generated["token_log_probabilities"],
            )
        )

    frame = ordered_frame(baseline_rows, BASELINE_COVER_COLUMNS)
    write_jsonl(output_dir / "baseline_cover_examples.jsonl", baseline_examples)
    return frame, baseline_examples


def build_feature_rows(cover_examples: Sequence[dict], baseline_examples: Sequence[dict]) -> List[dict]:
    rows = []
    for index, example in enumerate(cover_examples):
        feature_row = extract_text_features(
            example.get("generated_text", ""),
            token_ids=example.get("generated_token_ids"),
            ranks=example.get("recovered_ranks"),
            token_log_probabilities=example.get("token_log_probabilities"),
        )
        feature_row.update(
            {
                "source_type": "rankcloak",
                "source_id": "rankcloak_{}".format(index),
                "payload_name": example.get("payload_name"),
                "cover_prompt_name": example.get("cover_prompt_name"),
                "prompt_family": example.get("prompt_family"),
                "prompt_length_characters": example.get("prompt_length_characters"),
                "prompt_length_tokens": example.get("prompt_length_tokens"),
                "alphabet_size": example.get("alphabet_size"),
                "baseline_mode": None,
            }
        )
        rows.append(feature_row)

    for index, example in enumerate(baseline_examples):
        feature_row = extract_text_features(
            example.get("generated_text", ""),
            token_ids=example.get("generated_token_ids"),
            ranks=example.get("ranks"),
            token_log_probabilities=example.get("token_log_probabilities"),
        )
        feature_row.update(
            {
                "source_type": "baseline",
                "source_id": "baseline_{}".format(index),
                "payload_name": None,
                "cover_prompt_name": example.get("cover_prompt_name"),
                "prompt_family": example.get("prompt_family"),
                "prompt_length_characters": example.get("prompt_length_characters"),
                "prompt_length_tokens": example.get("prompt_length_tokens"),
                "alphabet_size": None,
                "baseline_mode": example.get("baseline_mode"),
            }
        )
        rows.append(feature_row)
    return rows


def write_cover_text_features(
    cover_examples: Sequence[dict],
    baseline_examples: Sequence[dict],
    output_dir: Path,
) -> pd.DataFrame:
    rows = build_feature_rows(cover_examples, baseline_examples)
    frame = ordered_frame(rows, COVER_TEXT_FEATURE_COLUMNS)
    frame.to_csv(output_dir / "cover_text_features.csv", index=False)
    return frame


def neutral_quality_notes(example: dict, feature_row: Optional[dict]) -> str:
    text = example.get("generated_text") or ""
    alphabetic_characters = [character for character in text if character.isalpha()]
    uppercase_fraction = (
        sum(1 for character in alphabetic_characters if character.isupper())
        / float(len(alphabetic_characters))
        if alphabetic_characters
        else 0.0
    )
    notes = []
    if not text.strip():
        notes.append("empty output")
    if "```" in text or "http" in text or text.count("...") >= 2:
        notes.append("contains formatting, link-like, or ellipsis artifacts")
    if "\\" in text or "{" in text or "}" in text or "[" in text or "]" in text:
        notes.append("contains markup-like or placeholder characters")
    if uppercase_fraction > 0.18:
        notes.append("unusually high uppercase-letter fraction")
    if feature_row:
        punctuation_fraction = feature_row.get("punctuation_fraction")
        repeated_token_fraction = feature_row.get("repeated_token_fraction")
        alphabetic_fraction = feature_row.get("alphabetic_fraction")
        line_count = feature_row.get("line_count")
        if punctuation_fraction is not None and punctuation_fraction > 0.16:
            notes.append("high punctuation or formatting fraction")
        if repeated_token_fraction is not None and repeated_token_fraction > 0.35:
            notes.append("higher repeated-token fraction")
        if (
            alphabetic_fraction is not None
            and alphabetic_fraction > 0.68
            and line_count is not None
            and line_count <= 8
        ):
            notes.append("mostly prose-like by lightweight heuristics")
    if not notes:
        notes.append("mixed or neutral by lightweight heuristics; inspect manually")
    return "; ".join(notes)


def markdown_safe_text_block(text: str, limit: int = 1200) -> str:
    """Return generated text clipped and neutralized for fenced Markdown display."""

    clipped = (text or "").strip()[:limit]
    return clipped.replace("```", "` ` `")


def write_prompt_comparison(
    cover_examples: Sequence[dict],
    feature_frame: pd.DataFrame,
    cover_prompt_names: Sequence[str],
    output_dir: Path,
    examples_per_prompt: int = 3,
    filename: str = "PROMPT_COMPARISON.md",
    title: str = "RankCloak Strong Prompt Comparison",
) -> Path:
    """Write a compact manual-inspection report for prompt quality comparison."""

    path = output_dir / filename
    feature_by_source_id = {}
    if not feature_frame.empty and "source_id" in feature_frame:
        for _, row in feature_frame.iterrows():
            feature_by_source_id[row["source_id"]] = row.to_dict()

    lines = [
        "# {}".format(title),
        "",
        "This file samples generated RankCloak cover text for manual inspection. "
        "The notes are lightweight heuristics, not human quality judgments.",
        "",
        "## Prompt Names Tested",
        "",
    ]
    for prompt_name in cover_prompt_names:
        lines.append("- `{}`".format(prompt_name))
    lines.append("")

    by_prompt: Dict[str, List[dict]] = {prompt_name: [] for prompt_name in cover_prompt_names}
    for example in cover_examples:
        prompt_name = example.get("cover_prompt_name")
        if prompt_name in by_prompt and len(by_prompt[prompt_name]) < examples_per_prompt:
            by_prompt[prompt_name].append(example)

    for prompt_name in cover_prompt_names:
        lines.extend(["## `{}`".format(prompt_name), ""])
        examples = by_prompt.get(prompt_name, [])
        if not examples:
            lines.extend(["No examples were generated for this prompt.", ""])
            continue
        for index, example in enumerate(examples, start=1):
            source_id = "rankcloak_{}".format(cover_examples.index(example))
            feature_row = feature_by_source_id.get(source_id)
            mean_log_probability = (
                feature_row.get("mean_token_log_probability") if feature_row else None
            )
            lines.extend(
                [
                    "### Example {}".format(index),
                    "",
                    "- payload_name: `{}`".format(example.get("payload_name")),
                    "- alphabet_size: `{}`".format(example.get("alphabet_size")),
                    "- exact_recovery: `{}`".format(example.get("exact_recovery")),
                    "- generated_token_count: `{}`".format(example.get("generated_token_count")),
                    "- mean_token_log_probability: `{}`".format(mean_log_probability),
                    "- notes: {}".format(neutral_quality_notes(example, feature_row)),
                    "",
                    "```text",
                    markdown_safe_text_block(example.get("generated_text") or ""),
                    "```",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def artifact_bit_length_if_known(payload: SyntheticPayload) -> Optional[int]:
    if payload.kind in {"sha256_hex", "random_hex", "nonce_hex", "hmac_like"} and is_hex_text(payload.text):
        return len(payload.text) * 4
    return None


def run_payload_granularity_comparison(
    payloads: Sequence[SyntheticPayload],
    alphabet_sizes: Sequence[int],
    model: Any,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for alphabet_size in alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(payload.bytes_value, alphabet_size)
            rows.append(
                {
                    "payload_name": payload.name,
                    "representation_name": "ascii_bytes_fixed_radix",
                    "alphabet_size": int(alphabet_size),
                    "artifact_text_character_length": len(payload.text),
                    "artifact_text_byte_length": len(payload.bytes_value),
                    "artifact_bit_length_if_known": artifact_bit_length_if_known(payload),
                    "rank_count": len(encoded["ranks"]),
                    "max_possible_rank": int(alphabet_size),
                    "bits_per_rank_estimate": float(math.log2(alphabet_size)),
                    "notes": "ASCII artifact string bytes encoded as fixed-radix bounded ranks",
                }
            )

        if is_hex_text(payload.text):
            encoded_hex = encode_hex_nibbles_to_ranks(payload.text)
            decoded_hex = decode_hex_nibble_ranks_to_text(
                encoded_hex["ranks"], encoded_hex["metadata"]
            )
            rows.append(
                {
                    "payload_name": payload.name,
                    "representation_name": "raw_hex_nibbles",
                    "alphabet_size": 16,
                    "artifact_text_character_length": len(payload.text),
                    "artifact_text_byte_length": len(payload.bytes_value),
                    "artifact_bit_length_if_known": len(payload.text) * 4,
                    "rank_count": len(encoded_hex["ranks"]),
                    "max_possible_rank": 16,
                    "bits_per_rank_estimate": 4.0,
                    "notes": (
                        "one hex character per rank; exact roundtrip {}".format(
                            decoded_hex == payload.text.lower()
                        )
                    ),
                }
            )

        if model is not None:
            trace = direct_subword_ranks_for_text(model, payload.text)
            ranks = trace["ranks"]
            rows.append(
                {
                    "payload_name": payload.name,
                    "representation_name": "raw_subword_direct",
                    "alphabet_size": None,
                    "artifact_text_character_length": len(payload.text),
                    "artifact_text_byte_length": len(payload.bytes_value),
                    "artifact_bit_length_if_known": artifact_bit_length_if_known(payload),
                    "rank_count": len(trace.get("payload_token_ids", [])),
                    "max_possible_rank": max(ranks) if ranks else None,
                    "bits_per_rank_estimate": None,
                    "notes": "direct model-token payload baseline; not bounded-rank cover generation",
                }
            )
        else:
            rows.append(
                {
                    "payload_name": payload.name,
                    "representation_name": "raw_subword_direct",
                    "alphabet_size": None,
                    "artifact_text_character_length": len(payload.text),
                    "artifact_text_byte_length": len(payload.bytes_value),
                    "artifact_bit_length_if_known": artifact_bit_length_if_known(payload),
                    "rank_count": None,
                    "max_possible_rank": None,
                    "bits_per_rank_estimate": None,
                    "notes": "model unavailable; direct subword rank pressure skipped",
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "payload_granularity_comparison.csv", index=False)
    return frame


def write_summary(
    output_dir: Path,
    project_root: Path,
    profile: str,
    model_repo_id: str,
    model_filename: str,
    model_path: Optional[Path],
    model_loaded: bool,
    payloads: Sequence[SyntheticPayload],
    cover_prompt_names: Sequence[str],
    alphabet_sizes: Sequence[int],
    codec_frame: pd.DataFrame,
    stegotext_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    generated_files: Sequence[Path],
    important_notes: Sequence[str],
) -> dict:
    codec_pass_count = int(codec_frame["exact_roundtrip"].sum()) if not codec_frame.empty else 0
    codec_fail_count = int((~codec_frame["exact_roundtrip"].astype(bool)).sum()) if not codec_frame.empty else 0
    if not stegotext_frame.empty and "exact_recovery" in stegotext_frame:
        stego_values = stegotext_frame["exact_recovery"].dropna().astype(bool)
        stego_pass_count = int(stego_values.sum())
        stego_fail_count = int((~stego_values).sum())
    else:
        stego_pass_count = 0
        stego_fail_count = 0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_loaded": bool(model_loaded),
        "number_of_payloads": len(payloads),
        "number_of_cover_prompts": len(cover_prompt_names),
        "alphabet_sizes": list(map(int, alphabet_sizes)),
        "codec_roundtrip_pass_count": codec_pass_count,
        "codec_roundtrip_fail_count": codec_fail_count,
        "stegotext_recovery_pass_count": stego_pass_count,
        "stegotext_recovery_fail_count": stego_fail_count,
        "baseline_count": int(len(baseline_frame)),
        "feature_row_count": int(len(feature_frame)),
        "generated_result_files": [
            repo_relative_path(path, project_root) for path in generated_files
        ],
        "important_notes": list(important_notes),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = """# RankCloak Crypto Artifact Exploration Summary

- Run profile: {profile}
- Model status: {model_status}
- Model: {model_repo_id} / {model_filename}
- Codec roundtrip result: {codec_passes} pass, {codec_failures} fail
- Stegotext recovery result: {stego_passes} pass, {stego_failures} fail
- Baseline generation result: {baseline_count} rows
- Feature extraction result: {feature_count} rows

## Important Limitations

- This is not encryption, key exchange, authentication, or cryptographic security.
- All payloads are deterministic synthetic examples.
- Exact recovery requires the same model, tokenizer, quantization, rank ordering, and unmodified generated text.
- The current detector work is feature extraction only; detector AUC remains a TODO.

## Next Recommended Experiment

Run `python3 scripts/run_experiment.py --profile small --overwrite` when CPU time is available.

## Generated Files

{files}
""".format(
        profile=profile,
        model_status="loaded" if model_loaded else "not loaded",
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        codec_passes=codec_pass_count,
        codec_failures=codec_fail_count,
        stego_passes=stego_pass_count,
        stego_failures=stego_fail_count,
        baseline_count=len(baseline_frame),
        feature_count=len(feature_frame),
        files="\n".join("- `{}`".format(repo_relative_path(path, project_root)) for path in generated_files),
    )
    (output_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    return summary


def run_experiment(args: argparse.Namespace) -> dict:
    profile = args.profile
    config = PROFILE_CONFIGS[profile]
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["default_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rank_order_test = test_stable_rank_ordering()
    if not rank_order_test["passed"]:
        raise AssertionError("Stable rank ordering test failed: {}".format(rank_order_test))

    all_payloads = generate_synthetic_payloads()
    experiment_payloads = selected_payloads(all_payloads, config["payload_names"])
    alphabet_sizes = list(map(int, config["alphabet_sizes"]))
    cover_prompts = cover_prompt_dictionary()
    cover_prompt_names = list(config["cover_prompt_names"])
    payload_byte_limit = (
        args.max_payload_bytes
        if args.max_payload_bytes is not None
        else config["default_max_payload_bytes"]
    )

    model, model_path, model_status, model_error, model_load_seconds = load_model_for_profile(
        profile=profile,
        model_path=args.model_path,
        skip_model_download=args.skip_model_download,
    )
    model_repo_id, model_filename = resolve_model_identity(model_path)
    model_path_relative = repo_relative_path(model_path, PROJECT_ROOT)
    model_loaded = model is not None
    prompt_metadata = build_prompt_metadata(cover_prompts, cover_prompt_names, model)

    audit_payloads = all_payloads
    rank_payloads = all_payloads if profile == "audit-only" else experiment_payloads
    tokenization_frame = run_tokenization_audit(audit_payloads, model, output_dir)
    rank_frame = run_rank_statistics(rank_payloads, model, model_error, output_dir)
    codec_frame = run_codec_roundtrips(all_payloads, SUPPORTED_ALPHABET_SIZES, output_dir)

    if config["requires_stegotext"]:
        stegotext_frame, cover_examples = run_stegotext_trials(
            payloads=experiment_payloads,
            cover_prompts=cover_prompts,
            cover_prompt_names=cover_prompt_names,
            alphabet_sizes=alphabet_sizes,
            model=model,
            model_repo_id=model_repo_id,
            model_filename=model_filename,
            model_path_relative=model_path_relative,
            payload_byte_limit=payload_byte_limit,
            prompt_metadata=prompt_metadata,
            output_dir=output_dir,
        )
    else:
        stegotext_frame = ordered_frame([], STEGOTEXT_RECOVERY_COLUMNS)
        stegotext_frame.to_csv(output_dir / "stegotext_recovery_trials.csv", index=False)
        cover_examples = []
        write_jsonl(output_dir / "cover_examples.jsonl", [])

    target_token_count = baseline_token_target(stegotext_frame, int(config["baseline_token_cap"]))
    baseline_frame, baseline_examples = run_baselines(
        cover_prompts=cover_prompts,
        cover_prompt_names=cover_prompt_names,
        model=model,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        target_token_count=target_token_count,
        prompt_metadata=prompt_metadata,
        output_dir=output_dir,
    )
    feature_frame = write_cover_text_features(cover_examples, baseline_examples, output_dir)

    token_count_figure = plot_token_count_by_payload(
        tokenization_frame, figures_dir / "token_count_by_payload.png"
    )
    rank_summary_figure = plot_rank_summary_direct_subword(
        rank_frame, figures_dir / "rank_summary_direct_subword.png"
    )
    cover_length_figure = plot_cover_length_vs_rank_alphabet(
        codec_frame, figures_dir / "cover_length_vs_rank_alphabet.png"
    )
    recovery_figure = plot_recovery_by_cover_prompt_and_alphabet(
        stegotext_frame, figures_dir / "recovery_by_cover_prompt_and_alphabet.png"
    )
    feature_figure = plot_cover_text_feature_comparison(
        feature_frame, figures_dir / "cover_text_feature_comparison.png"
    )
    extra_generated_files = []
    if config.get("write_prompt_comparison"):
        prompt_logprob_figure = plot_strong_prompt_mean_logprob(
            feature_frame, figures_dir / "strong_prompt_mean_logprob_by_prompt.png"
        )
        prompt_recovery_figure = plot_strong_prompt_recovery(
            stegotext_frame, figures_dir / "strong_prompt_recovery_by_prompt.png"
        )
        prompt_length_figure = plot_strong_prompt_length(
            stegotext_frame, figures_dir / "strong_prompt_length_by_prompt.png"
        )
        prompt_rank_figure = plot_strong_prompt_rank_pressure(
            stegotext_frame, figures_dir / "strong_prompt_rank_pressure.png"
        )
        prompt_comparison_path = write_prompt_comparison(
            cover_examples=cover_examples,
            feature_frame=feature_frame,
            cover_prompt_names=cover_prompt_names,
            output_dir=output_dir,
        )
        extra_generated_files.extend(
            [
                prompt_logprob_figure,
                prompt_recovery_figure,
                prompt_length_figure,
                prompt_rank_figure,
                prompt_comparison_path,
            ]
        )
    if config.get("write_dialogue_comparison"):
        dialogue_logprob_figure = plot_strong_prompt_mean_logprob(
            feature_frame, figures_dir / "dialogue_prompt_mean_logprob.png"
        )
        dialogue_repetition_figure = plot_dialogue_prompt_repetition(
            feature_frame, figures_dir / "dialogue_prompt_repetition.png"
        )
        dialogue_length_figure = plot_strong_prompt_length(
            stegotext_frame, figures_dir / "dialogue_prompt_length.png"
        )
        dialogue_scatter_figure = plot_dialogue_prompt_quality_scatter(
            feature_frame, figures_dir / "dialogue_prompt_quality_scatter.png"
        )
        dialogue_comparison_path = write_prompt_comparison(
            cover_examples=cover_examples,
            feature_frame=feature_frame,
            cover_prompt_names=cover_prompt_names,
            output_dir=output_dir,
            examples_per_prompt=2,
            filename="DIALOGUE_PROMPT_COMPARISON.md",
            title="RankCloak Dialogue Key Prompt Comparison",
        )
        extra_generated_files.extend(
            [
                dialogue_logprob_figure,
                dialogue_repetition_figure,
                dialogue_length_figure,
                dialogue_scatter_figure,
                dialogue_comparison_path,
            ]
        )
    if config.get("write_payload_granularity"):
        payload_granularity_frame = run_payload_granularity_comparison(
            payloads=experiment_payloads,
            alphabet_sizes=alphabet_sizes,
            model=model,
            output_dir=output_dir,
        )
        payload_granularity_figure = plot_payload_representation_rank_count(
            payload_granularity_frame,
            figures_dir / "payload_representation_rank_count.png",
        )
        extra_generated_files.extend(
            [
                output_dir / "payload_granularity_comparison.csv",
                payload_granularity_figure,
            ]
        )

    manifest_path = output_dir / "MANIFEST.json"
    write_manifest(
        output_path=manifest_path,
        project_root=PROJECT_ROOT,
        profile=profile,
        output_dir=output_dir,
        command_line_args=sys.argv[1:],
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
    )

    generated_files = [
        output_dir / "tokenization_audit.csv",
        output_dir / "rank_statistics.csv",
        output_dir / "codec_roundtrip_trials.csv",
        output_dir / "stegotext_recovery_trials.csv",
        output_dir / "cover_examples.jsonl",
        output_dir / "baseline_cover_examples.jsonl",
        output_dir / "cover_text_features.csv",
        manifest_path,
        token_count_figure,
        rank_summary_figure,
        cover_length_figure,
        recovery_figure,
        feature_figure,
        *extra_generated_files,
        output_dir / "summary.json",
        output_dir / "SUMMARY.md",
    ]
    notes = [
        "All payloads are deterministic synthetic examples.",
        "Model status: {}; load seconds: {:.3f}".format(model_status, model_load_seconds),
        "Model error: {}".format(model_error) if model_error else "Model loaded or was not required.",
        "Detector AUC is not implemented; cover_text_features.csv is feature extraction only.",
    ]
    summary = write_summary(
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        profile=profile,
        model_repo_id=model_repo_id,
        model_filename=model_filename,
        model_path=model_path,
        model_loaded=model_loaded,
        payloads=experiment_payloads,
        cover_prompt_names=cover_prompt_names,
        alphabet_sizes=alphabet_sizes,
        codec_frame=codec_frame,
        stegotext_frame=stegotext_frame,
        baseline_frame=baseline_frame,
        feature_frame=feature_frame,
        generated_files=generated_files,
        important_notes=notes,
    )
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RankCloak experiment profiles.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIGS),
        default="smoke",
        help="Experiment profile to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result outputs.",
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit GGUF model path.")
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=None,
        help="Optional payload byte limit for debugging stegotext trials.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Do not attempt Hugging Face model download if no local model exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deterministic result files in the output directory to be overwritten.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)
