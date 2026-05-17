"""Run a lightweight RankCloak smoke experiment and write result files."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.metrics import summarize_rank_sequence
from rankcloak.model_io import (
    FALLBACK_LLAMA3_SPEC,
    PREFERRED_LLAMA3_SPEC,
    default_thread_count,
    existing_llama3_model_path,
    load_llama_cpp_model,
    make_context_token_ids,
)
from rankcloak.plotting import (
    plot_cover_length_vs_rank_alphabet,
    plot_rank_summary_direct_subword,
    plot_token_count_by_payload,
)
from rankcloak.rank_codec import (
    SUPPORTED_ALPHABET_SIZES,
    bounded_roundtrip_rows,
    decode_bounded_ranks_to_bytes,
    direct_subword_ranks_for_text,
    encode_bytes_to_bounded_ranks,
    generate_token_ids_from_ranks,
    recover_ranks_from_generated_ids,
    test_stable_rank_ordering,
)
from rankcloak.synthetic_payloads import generate_synthetic_payloads
from rankcloak.tokenization_audit import audit_payload_tokenization


RESULTS_DIR = PROJECT_ROOT / "results" / "rankcloak_crypto_artifact_exploration"
FIGURES_DIR = RESULTS_DIR / "figures"


def cover_prompt_dictionary() -> dict:
    return {
        "play_dialogue": (
            "Write a short original stage dialogue between two lighthouse keepers during a calm "
            "maintenance shift. Keep the dialogue grounded, mundane, and naturally paced.\n"
        ),
        "original_fantasy_fiction": (
            "Continue an original low-fantasy travel scene about a cartographer crossing a foggy "
            "salt marsh. Use quiet sensory detail and no named copyrighted worlds.\n"
        ),
        "recipe_blog": (
            "Write a practical recipe blog paragraph about preparing lentil stew for a rainy "
            "weekday dinner. Use warm but ordinary cooking language.\n"
        ),
        "forum_reply": (
            "Write a helpful forum reply to someone asking how to organize a small community "
            "workshop. Be specific, friendly, and concise.\n"
        ),
        "technical_documentation": (
            "Write a technical documentation paragraph explaining how to rotate application logs "
            "on a local server. Use clear operational prose without commands that affect real systems.\n"
        ),
        "code_review_comment": (
            "Write a code review comment about simplifying a validation helper and improving test "
            "coverage. Be constructive and precise.\n"
        ),
        "meeting_minutes": (
            "Write concise meeting minutes for a neighborhood garden planning meeting. Include "
            "neutral agenda-style phrasing.\n"
        ),
    }


def resolve_model_identity(model_path: Path) -> tuple:
    if str(model_path).endswith(PREFERRED_LLAMA3_SPEC.filename):
        return PREFERRED_LLAMA3_SPEC.repo_id, PREFERRED_LLAMA3_SPEC.filename
    if str(model_path).endswith(FALLBACK_LLAMA3_SPEC.filename):
        return FALLBACK_LLAMA3_SPEC.repo_id, FALLBACK_LLAMA3_SPEC.filename
    return PREFERRED_LLAMA3_SPEC.repo_id, PREFERRED_LLAMA3_SPEC.filename


def main() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    environment_info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "n_threads": default_thread_count(),
    }
    print(json.dumps(environment_info, indent=2))

    rank_order_test = test_stable_rank_ordering()
    if not rank_order_test["passed"]:
        raise AssertionError("Stable rank ordering test failed")

    model_path = existing_llama3_model_path()
    model = None
    model_status = "not_loaded"
    model_error = None
    model_load_start = time.perf_counter()
    try:
        if model_path is not None:
            model = load_llama_cpp_model(
                model_path=model_path,
                n_ctx=2048,
                n_threads=default_thread_count(),
                logits_all=True,
            )
            model_status = "loaded"
        else:
            model_status = "unavailable"
            model_error = "No supported local GGUF model found."
    except Exception as exc:
        model = None
        model_status = "unavailable"
        model_error = str(exc)
    model_load_seconds = time.perf_counter() - model_load_start

    payloads = generate_synthetic_payloads()
    cover_prompts = cover_prompt_dictionary()

    tokenization_frame = pd.DataFrame(audit_payload_tokenization(payloads, model=model))
    tokenization_path = RESULTS_DIR / "tokenization_audit.csv"
    tokenization_frame.to_csv(tokenization_path, index=False)

    rank_start = time.perf_counter()
    rank_rows = []
    if model is not None:
        for payload in payloads:
            trace = direct_subword_ranks_for_text(model, payload.text)
            row = summarize_rank_sequence(payload.name, trace["ranks"])
            row.update({"payload_kind": payload.kind, "model_loaded": True, "skip_reason": None})
            rank_rows.append(row)
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
            rank_rows.append(row)
    rank_seconds = time.perf_counter() - rank_start
    rank_frame = pd.DataFrame(rank_rows)
    rank_statistics_path = RESULTS_DIR / "rank_statistics.csv"
    rank_frame.to_csv(rank_statistics_path, index=False)

    recovery_frame = pd.DataFrame(bounded_roundtrip_rows(payloads, SUPPORTED_ALPHABET_SIZES))
    if not bool(recovery_frame["exact_recovery"].all()):
        raise AssertionError("A bounded-rank recovery trial failed")
    recovery_trials_path = RESULTS_DIR / "recovery_trials.csv"
    recovery_frame.to_csv(recovery_trials_path, index=False)

    cover_examples_path = RESULTS_DIR / "cover_examples.jsonl"
    smoke_payload = next(payload for payload in payloads if payload.name == "sha256_public_test_string")
    smoke_payload_byte_limit = int(os.environ.get("RANKCLOAK_SMOKE_BYTES", "8"))
    smoke_payload_bytes = smoke_payload.bytes_value[:smoke_payload_byte_limit]
    smoke_cover_prompt_names = ["play_dialogue", "recipe_blog"]
    smoke_alphabet_sizes = [16, 32]
    cover_examples = []
    smoke_start = time.perf_counter()
    if model is not None:
        for alphabet_size in smoke_alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(smoke_payload_bytes, alphabet_size)
            for prompt_name in smoke_cover_prompt_names:
                context_ids = make_context_token_ids(model, cover_prompts[prompt_name])
                generated = generate_token_ids_from_ranks(model, context_ids, encoded["ranks"])
                recovered = recover_ranks_from_generated_ids(
                    model, context_ids, generated["generated_token_ids"]
                )
                decoded_bytes = decode_bounded_ranks_to_bytes(recovered["ranks"], encoded["metadata"])
                cover_examples.append(
                    {
                        "status": "generated",
                        "payload_name": smoke_payload.name,
                        "payload_bytes_used": len(smoke_payload_bytes),
                        "payload_text_prefix": smoke_payload_bytes.decode("utf-8", errors="replace"),
                        "truncated_for_smoke": len(smoke_payload_bytes) < len(smoke_payload.bytes_value),
                        "alphabet_size": alphabet_size,
                        "cover_prompt_name": prompt_name,
                        "rank_count": len(encoded["ranks"]),
                        "generated_token_count": len(generated["generated_token_ids"]),
                        "generated_text": generated["generated_text"],
                        "generated_token_ids": generated["generated_token_ids"],
                        "recovered_ranks": recovered["ranks"],
                        "exact_recovery": decoded_bytes == smoke_payload_bytes,
                    }
                )
    else:
        for alphabet_size in smoke_alphabet_sizes:
            encoded = encode_bytes_to_bounded_ranks(smoke_payload_bytes, alphabet_size)
            for prompt_name in smoke_cover_prompt_names:
                cover_examples.append(
                    {
                        "status": "skipped_model_unavailable",
                        "payload_name": smoke_payload.name,
                        "payload_bytes_used": len(smoke_payload_bytes),
                        "payload_text_prefix": smoke_payload_bytes.decode("utf-8", errors="replace"),
                        "truncated_for_smoke": len(smoke_payload_bytes) < len(smoke_payload.bytes_value),
                        "alphabet_size": alphabet_size,
                        "cover_prompt_name": prompt_name,
                        "rank_count": len(encoded["ranks"]),
                        "generated_token_count": None,
                        "generated_text": None,
                        "generated_token_ids": [],
                        "recovered_ranks": [],
                        "exact_recovery": None,
                        "skip_reason": model_error or "model unavailable",
                    }
                )
    smoke_seconds = time.perf_counter() - smoke_start
    with cover_examples_path.open("w", encoding="utf-8") as handle:
        for example in cover_examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    token_count_figure = plot_token_count_by_payload(
        tokenization_frame, FIGURES_DIR / "token_count_by_payload.png"
    )
    rank_summary_figure = plot_rank_summary_direct_subword(
        rank_frame, FIGURES_DIR / "rank_summary_direct_subword.png"
    )
    cover_length_figure = plot_cover_length_vs_rank_alphabet(
        recovery_frame, FIGURES_DIR / "cover_length_vs_rank_alphabet.png"
    )

    generated_result_files = [
        tokenization_path,
        rank_statistics_path,
        recovery_trials_path,
        cover_examples_path,
        token_count_figure,
        rank_summary_figure,
        cover_length_figure,
    ]
    exact_recovery_values = [
        example.get("exact_recovery")
        for example in cover_examples
        if example.get("exact_recovery") is not None
    ]
    exact_recovery_pass_count = sum(1 for value in exact_recovery_values if value is True)
    exact_recovery_fail_count = sum(1 for value in exact_recovery_values if value is False)
    smoke_test_status = (
        "generated"
        if any(example.get("status") == "generated" for example in cover_examples)
        else "skipped_model_unavailable"
    )
    summary_model_path = model_path if model_path is not None else PREFERRED_LLAMA3_SPEC.destination
    model_repo_id, model_filename = resolve_model_identity(summary_model_path)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(summary_model_path),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_status": model_status,
        "model_load_seconds": model_load_seconds,
        "number_of_payloads": len(payloads),
        "number_of_cover_prompts": len(cover_prompts),
        "smoke_test_status": smoke_test_status,
        "smoke_seconds": smoke_seconds,
        "rank_seconds": rank_seconds,
        "exact_recovery_pass_count": exact_recovery_pass_count,
        "exact_recovery_fail_count": exact_recovery_fail_count,
        "paths_of_generated_result_files": [str(path) for path in generated_result_files],
        "short_notes": [
            "All payloads are deterministic synthetic examples.",
            "Direct subword rank statistics require the local GGUF model and llama-cpp-python.",
            "Bounded-rank codec round trips are tested without a model.",
            "Smoke stegotext generation is truncated for CPU safety.",
        ],
    }

    summary_json_path = RESULTS_DIR / "summary.json"
    summary_md_path = RESULTS_DIR / "SUMMARY.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md = """# RankCloak Crypto Artifact Exploration Summary

- Timestamp: {timestamp}
- Model status: {model_status}
- Model path: {model_path}
- Payloads: {payloads}
- Cover prompts: {cover_prompts}
- Smoke test status: {smoke_status}
- Exact recovery passes: {passes}
- Exact recovery failures: {failures}

## Notes

All examples are deterministic and synthetic. This notebook studies concealment of cryptographic-artifact-like strings in generated cover text under exact-copy conditions. It does not provide encryption, key exchange, or a proof of steganographic security.

## Generated Files

{files}
""".format(
        timestamp=summary["timestamp"],
        model_status=model_status,
        model_path=summary["model_path"],
        payloads=len(payloads),
        cover_prompts=len(cover_prompts),
        smoke_status=smoke_test_status,
        passes=exact_recovery_pass_count,
        failures=exact_recovery_fail_count,
        files="\n".join("- `{}`".format(path) for path in generated_result_files),
    )
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
