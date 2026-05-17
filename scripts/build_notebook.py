"""Build the first RankCloak research notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_rankcloak_crypto_artifact_exploration.ipynb"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code_cell(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    cells = [
        markdown_cell(
            """
# RankCloak: Character-Level and Subword LLM Steganography for Cryptographic Artifacts

This notebook is the first RankCloak research pass. It studies deterministic synthetic cryptographic-artifact-like payloads under rank-transcoding steganography, with special attention to how raw high-entropy strings behave under direct subword-token encoding versus bounded-rank alternatives.
"""
        ),
        markdown_cell(
            """
## Research Motivation

Raw cryptographic artifacts are high-entropy strings. SHA-256 digests, random key-like values, nonces, UUID-like values, bearer-token-like strings, HMAC-like tags, JWT-like strings, and ciphertext-like blocks have weak natural-language predictability. That makes them difficult payloads for direct token-rank steganography: if the model assigns the next artifact token a low probability, the corresponding rank can be large, forcing cover generation to choose an unlikely token.

All examples in this notebook are deterministic and synthetic. No real credentials, private keys, tokens, accounts, services, or operational secrets are used.
"""
        ),
        markdown_cell(
            """
## Method Background

The Calgacus paper, "LLMs can hide text in other text of the same length", presents a token-rank protocol:

- tokenize the secret text with the same LLM tokenizer used for generation
- compute each secret token's rank under the LLM next-token distribution
- generate cover text from a key prompt by selecting the token at the same rank at each step
- recover by recomputing the cover token ranks with the same LLM and key prompt, then reversing the rank process

The key limitation for RankCloak is already visible in the paper: difficult-to-predict strings such as hash-like text produce high ranks and can yield broken stegotext. RankCloak narrows the payload ranks with character-level and fixed-radix encodings, then measures the length and recovery tradeoff.
"""
        ),
        markdown_cell(
            """
## Research Questions

RQ1: How does direct subword tokenization behave for cryptographic artifacts?

RQ2: Does character-level or bounded-rank encoding improve plausibility and recovery?

RQ3: Which cover disguises tolerate larger rank alphabets?

RQ4: What is the CPU cost of a small local experiment?
"""
        ),
        code_cell(
            """
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rankcloak.metrics import summarize_rank_sequence
from rankcloak.model_io import (
    FALLBACK_LLAMA3_SPEC,
    PREFERRED_LLAMA3_SPEC,
    default_thread_count,
    download_llama3_gguf,
    existing_llama3_model_path,
    load_llama_cpp_model,
    make_context_token_ids,
    safe_detokenize,
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

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results" / "rankcloak_crypto_artifact_exploration"
FIGURES_DIR = RESULTS_DIR / "figures"
REFERENCES_DIR = PROJECT_ROOT / "references"
MODELS_DIR = PROJECT_ROOT / "models" / "llama3_8b"

for directory in [RESULTS_DIR, FIGURES_DIR, REFERENCES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Results dir:", RESULTS_DIR)
"""
        ),
        markdown_cell(
            """
## Environment and Model Setup

This notebook is CPU-first. It detects the local CPU count, locates the preferred or fallback Llama 3 8B Instruct GGUF, and tries to load it with `llama-cpp-python` using logits access. To avoid accidental multi-gigabyte downloads during notebook execution, model download only runs when `RANKCLOAK_DOWNLOAD_MODEL=1` is set in the environment. The agent build step may also download the model after explicit approval.
"""
        ),
        code_cell(
            """
environment_info = {
    "python": sys.version.replace("\\n", " "),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "n_threads": default_thread_count(),
}
print(json.dumps(environment_info, indent=2))

model_path = existing_llama3_model_path()
download_requested = os.environ.get("RANKCLOAK_DOWNLOAD_MODEL", "0") == "1"
if model_path is None and download_requested:
    print("RANKCLOAK_DOWNLOAD_MODEL=1, attempting GGUF download with huggingface_hub.")
    model_path = download_llama3_gguf()
elif model_path is None:
    print("No local GGUF model found. Set RANKCLOAK_DOWNLOAD_MODEL=1 or manually place a supported file in models/llama3_8b/.")

model = None
model_status = "not_loaded"
model_error = None
load_start = time.perf_counter()
try:
    if model_path is not None:
        model = load_llama_cpp_model(model_path=model_path, n_ctx=2048, n_threads=default_thread_count(), logits_all=True)
        model_status = "loaded"
except Exception as exc:
    model_error = str(exc)
    model_status = "unavailable"

model_load_seconds = time.perf_counter() - load_start
print("model_status:", model_status)
print("model_path:", model_path)
if model_error:
    print("model_error:", model_error)
"""
        ),
        markdown_cell(
            """
## Source Notes

Paper source: `references/2510.20075.pdf` should contain the arXiv PDF when downloaded locally. The paper's method section defines rank-preserving encoding and decoding. The paper also notes that a hash-like string can produce high ranks and broken recipe-style stegotext, which motivates bounded-rank encodings here.

Previous repository source: `https://github.com/mantzaris/LlmStenoExplore` was inspected through GitHub rather than cloned into this repository. The useful implementation patterns were token-id-level correctness checks, stable rank ordering, `llama-cpp-python` loading with `logits_all=True`, CPU smoke profiles, and manifest-style outputs.
"""
        ),
        markdown_cell(
            """
## Synthetic Payload Generation

The payloads below are deterministic synthetic examples only. Strings that resemble bearer tokens, JWTs, HMAC tags, or ciphertext are deliberately fake and have no operational value.
"""
        ),
        code_cell(
            """
payloads = generate_synthetic_payloads()
payload_frame = pd.DataFrame(
    [
        {
            "payload_name": payload.name,
            "kind": payload.kind,
            "character_length": len(payload.text),
            "byte_length": len(payload.bytes_value),
            "description": payload.description,
            "text": payload.text,
        }
        for payload in payloads
    ]
)
display(payload_frame)
"""
        ),
        markdown_cell(
            """
## Tokenization Audit

For each payload, the audit records character length, byte length, LLM token count, token ratios, and the first several token ids and decoded token pieces. If the model is unavailable, the length fields are still written and tokenizer-dependent fields are left empty.
"""
        ),
        code_cell(
            """
tokenization_rows = audit_payload_tokenization(payloads, model=model)
tokenization_frame = pd.DataFrame(tokenization_rows)
tokenization_path = RESULTS_DIR / "tokenization_audit.csv"
tokenization_frame.to_csv(tokenization_path, index=False)
display(tokenization_frame)
print("wrote", tokenization_path)
"""
        ),
        markdown_cell(
            """
## Stable Rank Ordering Test

Ranks are 1-indexed. The deterministic order is decreasing logit, with ties broken by increasing token id. The synthetic test below verifies that tie behavior before any model-dependent experiment runs.
"""
        ),
        code_cell(
            """
rank_order_test = test_stable_rank_ordering()
print(json.dumps(rank_order_test, indent=2))
assert rank_order_test["passed"], "Stable rank ordering test failed"
"""
        ),
        markdown_cell(
            """
## Rank Statistics for Direct Subword Encoding

This section computes direct raw subword-token ranks for each payload when model logits are available. These are the ranks Calgacus-style direct encoding would carry into cover generation.
"""
        ),
        code_cell(
            """
rank_rows = []
direct_rank_traces = {}
rank_start = time.perf_counter()

if model is not None:
    for payload in payloads:
        trace = direct_subword_ranks_for_text(model, payload.text)
        direct_rank_traces[payload.name] = trace
        row = summarize_rank_sequence(payload.name, trace["ranks"])
        row.update(
            {
                "payload_kind": payload.kind,
                "model_loaded": True,
                "skip_reason": None,
            }
        )
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
display(rank_frame)
print("wrote", rank_statistics_path)
print("rank_seconds:", round(rank_seconds, 3))
"""
        ),
        markdown_cell(
            """
## Bounded-Rank Encodings

This section tests exact byte recovery for two alternatives:

- hex-character encoding: bytes are represented as lowercase hex and each hex character maps to rank 1..16
- fixed-radix bit encoding: bytes are grouped into base `B` symbols for `B in [2, 4, 8, 16, 32, 64]`, then each symbol maps to rank `digit + 1`

The metadata stores the original byte length and padding bits. In this first notebook, metadata is not hidden in cover text.
"""
        ),
        code_cell(
            """
recovery_rows = bounded_roundtrip_rows(payloads, SUPPORTED_ALPHABET_SIZES)
recovery_frame = pd.DataFrame(recovery_rows)
recovery_trials_path = RESULTS_DIR / "recovery_trials.csv"
recovery_frame.to_csv(recovery_trials_path, index=False)
display(recovery_frame)
assert bool(recovery_frame["exact_recovery"].all()), "A bounded-rank recovery trial failed"
print("wrote", recovery_trials_path)
"""
        ),
        markdown_cell(
            """
## Cover Prompts

The cover prompts below are original and non-copyrighted. They are stored as a dictionary so the same prompts can be reused by smoke tests and future sweeps.
"""
        ),
        code_cell(
            """
cover_prompts = {
    "play_dialogue": (
        "Write a short original stage dialogue between two lighthouse keepers during a calm maintenance shift. "
        "Keep the dialogue grounded, mundane, and naturally paced.\\n"
    ),
    "original_fantasy_fiction": (
        "Continue an original low-fantasy travel scene about a cartographer crossing a foggy salt marsh. "
        "Use quiet sensory detail and no named copyrighted worlds.\\n"
    ),
    "recipe_blog": (
        "Write a practical recipe blog paragraph about preparing lentil stew for a rainy weekday dinner. "
        "Use warm but ordinary cooking language.\\n"
    ),
    "forum_reply": (
        "Write a helpful forum reply to someone asking how to organize a small community workshop. "
        "Be specific, friendly, and concise.\\n"
    ),
    "technical_documentation": (
        "Write a technical documentation paragraph explaining how to rotate application logs on a local server. "
        "Use clear operational prose without commands that affect real systems.\\n"
    ),
    "code_review_comment": (
        "Write a code review comment about simplifying a validation helper and improving test coverage. "
        "Be constructive and precise.\\n"
    ),
    "meeting_minutes": (
        "Write concise meeting minutes for a neighborhood garden planning meeting. "
        "Include neutral agenda-style phrasing.\\n"
    ),
}

print(json.dumps(cover_prompts, indent=2))
"""
        ),
        markdown_cell(
            """
## Stegotext Generation Smoke Test

The smoke test uses the SHA-256 digest payload, bounded alphabets `B=16` and `B=32`, and two cover prompts: play dialogue and recipe blog. To keep CPU runtime bounded, it encodes only the first few bytes of the SHA-256 digest string. Full-payload runs should be explicit future experiments.
"""
        ),
        code_cell(
            """
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
            prompt = cover_prompts[prompt_name]
            context_ids = make_context_token_ids(model, prompt)
            generated = generate_token_ids_from_ranks(model, context_ids, encoded["ranks"])
            recovered = recover_ranks_from_generated_ids(model, context_ids, generated["generated_token_ids"])
            decoded_bytes = decode_bounded_ranks_to_bytes(recovered["ranks"], encoded["metadata"])
            exact_recovery = decoded_bytes == smoke_payload_bytes
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
                    "exact_recovery": exact_recovery,
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

with cover_examples_path.open("w", encoding="utf-8") as handle:
    for example in cover_examples:
        handle.write(json.dumps(example, ensure_ascii=False) + "\\n")

smoke_seconds = time.perf_counter() - smoke_start
cover_examples_frame = pd.DataFrame(cover_examples)
display(cover_examples_frame)
print("wrote", cover_examples_path)
print("smoke_seconds:", round(smoke_seconds, 3))
"""
        ),
        markdown_cell(
            """
## Plots

The figures are saved with matplotlib only. Placeholder figures are written if the model-dependent data is unavailable.
"""
        ),
        code_cell(
            """
token_count_figure = plot_token_count_by_payload(
    tokenization_frame, FIGURES_DIR / "token_count_by_payload.png"
)
rank_summary_figure = plot_rank_summary_direct_subword(
    rank_frame, FIGURES_DIR / "rank_summary_direct_subword.png"
)
cover_length_figure = plot_cover_length_vs_rank_alphabet(
    recovery_frame, FIGURES_DIR / "cover_length_vs_rank_alphabet.png"
)
print(token_count_figure)
print(rank_summary_figure)
print(cover_length_figure)
"""
        ),
        markdown_cell(
            """
## Limitations

- This is not encryption.
- This is not key exchange.
- Exact recovery requires the same model, tokenizer, quantization, and deterministic rank ordering.
- Public text channels must preserve the generated text exactly.
- All examples are synthetic.
- High-entropy payloads are harder than natural-language payloads because their direct next-token ranks tend to be larger.
- The smoke test is intentionally small and does not establish security, indistinguishability, or robustness.
"""
        ),
        markdown_cell(
            """
## Next Experiments

- larger cover genre sweep
- model comparison across Phi, Llama, Mistral, Qwen, and Gemma
- detector AUC experiments
- distribution-matched rank coding
- edit robustness tests
- human or LLM plausibility study
"""
        ),
        markdown_cell(
            """
## Final Summary Outputs

The cell below writes both machine-readable and human-readable summaries.
"""
        ),
        code_cell(
            """
generated_result_files = [
    tokenization_path,
    rank_statistics_path,
    recovery_trials_path,
    cover_examples_path,
    token_count_figure,
    rank_summary_figure,
    cover_length_figure,
]

generated_statuses = [example.get("status") for example in cover_examples]
if any(status == "generated" for status in generated_statuses):
    smoke_test_status = "generated"
else:
    smoke_test_status = "skipped_model_unavailable"

exact_recovery_values = [
    example.get("exact_recovery")
    for example in cover_examples
    if example.get("exact_recovery") is not None
]
exact_recovery_pass_count = sum(1 for value in exact_recovery_values if value is True)
exact_recovery_fail_count = sum(1 for value in exact_recovery_values if value is False)

if model_path is not None and str(model_path).endswith(PREFERRED_LLAMA3_SPEC.filename):
    model_repo_id = PREFERRED_LLAMA3_SPEC.repo_id
    model_filename = PREFERRED_LLAMA3_SPEC.filename
elif model_path is not None and str(model_path).endswith(FALLBACK_LLAMA3_SPEC.filename):
    model_repo_id = FALLBACK_LLAMA3_SPEC.repo_id
    model_filename = FALLBACK_LLAMA3_SPEC.filename
else:
    model_repo_id = PREFERRED_LLAMA3_SPEC.repo_id
    model_filename = PREFERRED_LLAMA3_SPEC.filename

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_path": str(model_path) if model_path is not None else str(PREFERRED_LLAMA3_SPEC.destination),
    "model_repo_id": model_repo_id,
    "model_filename": model_filename,
    "model_status": model_status,
    "model_load_seconds": model_load_seconds,
    "number_of_payloads": len(payloads),
    "number_of_cover_prompts": len(cover_prompts),
    "smoke_test_status": smoke_test_status,
    "smoke_seconds": smoke_seconds,
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

summary_md = f\"\"\"# RankCloak Crypto Artifact Exploration Summary

- Timestamp: {summary["timestamp"]}
- Model status: {model_status}
- Model path: {summary["model_path"]}
- Payloads: {len(payloads)}
- Cover prompts: {len(cover_prompts)}
- Smoke test status: {smoke_test_status}
- Exact recovery passes: {exact_recovery_pass_count}
- Exact recovery failures: {exact_recovery_fail_count}

## Notes

All examples are deterministic and synthetic. This notebook studies concealment of cryptographic-artifact-like strings in generated cover text under exact-copy conditions. It does not provide encryption, key exchange, or a proof of steganographic security.

## Generated Files

\"\"\"
summary_md += "\\n".join(f"- `{path}`" for path in generated_result_files)
summary_md += "\\n"
summary_md_path.write_text(summary_md, encoding="utf-8")

print("wrote", summary_json_path)
print("wrote", summary_md_path)
print(json.dumps(summary, indent=2))
"""
        ),
    ]

    notebook["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(NOTEBOOK_PATH))


if __name__ == "__main__":
    build_notebook()
    print(NOTEBOOK_PATH)
