"""Deterministic tokenizer-only preflight for the revision-v1 study.

The preflight loads each pinned GGUF with ``vocab_only=True``.  It never
evaluates a context or generates a token.  Its purpose is to fail closed before
GPU work if direct-subword payload bytes or prompt token boundaries do not obey
the payload-fidelity-v2 contract.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model_io import (
    detokenize_bytes,
    get_bos_token_id,
    make_context_token_ids,
    preload_pip_cuda_libraries,
    tokenize_bytes,
    tokenize_payload_text,
)
from .revision_artifacts import (
    ImmutableArtifactError,
    build_directory_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    verify_directory_manifest,
)
from .revision_config import (
    DEFAULT_REVISION_CONFIG_DIR,
    flatten_prompt_templates,
    load_json_object,
    load_revision_config_set,
    verify_model_artifact_pins,
)
from .revision_payloads import (
    REVISION_CORPUS_ID,
    REVISION_CORPUS_SHA256,
    generate_revision_v1_payloads,
    revision_corpus_sha256,
    revision_payload_records,
    validate_revision_corpus,
)


PREFLIGHT_ID = "rankcloak_scientific_reports_revision_v1_tokenizer_preflight_v2"
PREFLIGHT_SCHEMA_VERSION = "2.0"
PROTOCOL_CONTRACT_REVISION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
PAYLOAD_TOKENIZATION_CONTRACT = (
    "literal_utf8_no_special_tokens_reversible_space_prefix_v2"
)
PROMPT_CONTEXT_CONTRACT = "actual_bos_only_removal_first_real_token_retention_v2"
PREFIX_POLICY = "zero_or_more_ascii_space_0x20_bytes_only"
EXECUTION_MODE = "llama_cpp_vocab_only_no_generation"
MANIFEST_NAME = "TOKENIZER_PREFLIGHT_MANIFEST.json"
RECORDS_NAME = "records.jsonl"
FAILURES_NAME = "failures.jsonl"
MANIFEST_HASH_FIELD = "preflight_manifest_sha256"

RUNTIME_SOURCE_PATHS: Tuple[str, ...] = (
    "rankcloak/model_io.py",
    "rankcloak/revision_artifacts.py",
    "rankcloak/revision_config.py",
    "rankcloak/revision_payloads.py",
    "rankcloak/revision_tokenizer_preflight.py",
    "scripts/run_revision_tokenizer_preflight.py",
)


class TokenizerPreflightError(RuntimeError):
    """Raised when the preflight inputs or frozen output are invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _prefix_metadata(serialized: bytes, expected: bytes) -> Dict[str, object]:
    """Describe reversible leading-space framing and exact suffix recovery."""

    suffix_exact = bool(expected) and serialized.endswith(expected)
    prefix = serialized[: len(serialized) - len(expected)] if suffix_exact else b""
    prefix_permitted = suffix_exact and all(value == 0x20 for value in prefix)
    recovered = serialized[len(prefix) :] if suffix_exact else b""
    return {
        "serialized_utf8_sha256": _sha256_bytes(serialized),
        "serialized_byte_length": len(serialized),
        "expected_utf8_sha256": _sha256_bytes(expected),
        "expected_byte_length": len(expected),
        "suffix_exact": suffix_exact,
        "prefix_policy": PREFIX_POLICY,
        "prefix_permitted": prefix_permitted,
        "prefix_bytes_base64": base64.b64encode(prefix).decode("ascii"),
        "prefix_byte_length": len(prefix),
        "prefix_sha256": _sha256_bytes(prefix),
        "recovered_utf8_sha256": _sha256_bytes(recovered),
        "recovered_byte_length": len(recovered),
        "exact_original_byte_recovery": recovered == expected,
    }


def _audit_exception_record(
    record_type: str,
    model_id: str,
    item_id: str,
    exc: Exception,
    base: Mapping[str, object],
) -> Dict[str, object]:
    record = dict(base)
    record.update(
        {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "preflight_id": PREFLIGHT_ID,
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "execution_mode": EXECUTION_MODE,
            "record_type": record_type,
            "model_id": str(model_id),
            "item_id": str(item_id),
            "audit_status": "fail",
            "failure_codes": ["tokenizer_exception"],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    )
    return record


def audit_payload_tokenization(
    model: Any,
    model_id: str,
    payload: Any,
) -> Dict[str, object]:
    """Audit one literal cryptographic payload without model evaluation."""

    payload_bytes = bytes(payload.payload_bytes)
    base = {
        "payload_name": str(payload.payload_name),
        "payload_class": str(payload.payload_class),
        "payload_index": int(payload.payload_index),
        "input_utf8_sha256": _sha256_bytes(payload_bytes),
        "input_byte_length": len(payload_bytes),
        "payload_tokenization_contract": PAYLOAD_TOKENIZATION_CONTRACT,
    }
    try:
        token_ids = tokenize_payload_text(model, str(payload.payload_text))
        serialized = detokenize_bytes(model, token_ids)
        framing = _prefix_metadata(serialized, payload_bytes)
        failure_codes: List[str] = []
        if not token_ids:
            failure_codes.append("empty_payload_tokenization")
        if not framing["suffix_exact"]:
            failure_codes.append("payload_not_exact_suffix")
        if not framing["prefix_permitted"]:
            failure_codes.append("non_space_or_non_prefix_transformation")
        if not framing["exact_original_byte_recovery"]:
            failure_codes.append("original_payload_bytes_not_recovered")
        record = dict(base)
        record.update(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "preflight_id": PREFLIGHT_ID,
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "execution_mode": EXECUTION_MODE,
                "record_type": "payload_tokenization",
                "model_id": str(model_id),
                "item_id": str(payload.payload_name),
                "token_ids": list(map(int, token_ids)),
                "token_count": len(token_ids),
                "framing": framing,
                "audit_status": "pass" if not failure_codes else "fail",
                "failure_codes": failure_codes,
            }
        )
        return record
    except Exception as exc:
        return _audit_exception_record(
            "payload_tokenization",
            model_id,
            str(payload.payload_name),
            exc,
            base,
        )


def _tokenize_no_special(model: Any, text_bytes: bytes) -> List[int]:
    try:
        values = model.tokenize(text_bytes, add_bos=False, special=False)
    except TypeError:
        try:
            values = model.tokenize(text_bytes, add_bos=False)
        except TypeError as exc:
            raise TypeError(
                "Tokenizer cannot guarantee explicit add_bos=False prompt semantics"
            ) from exc
    return list(map(int, values))


def audit_prompt_context(
    model: Any,
    model_id: str,
    prompt: Mapping[str, object],
) -> Dict[str, object]:
    """Audit one prompt for actual-BOS-only removal and first-token retention."""

    prompt_id = str(prompt["prompt_id"])
    text = str(prompt["text"])
    prompt_bytes = text.encode("utf-8")
    base = {
        "prompt_id": prompt_id,
        "prompt_category": str(prompt["category_id"]),
        "language": str(prompt["language"]),
        "input_utf8_sha256": _sha256_bytes(prompt_bytes),
        "input_byte_length": len(prompt_bytes),
        "prompt_context_contract": PROMPT_CONTEXT_CONTRACT,
    }
    try:
        special_ids = list(
            map(int, tokenize_bytes(model, prompt_bytes, add_bos=True))
        )
        no_special_ids = _tokenize_no_special(model, prompt_bytes)
        bos_token_id = get_bos_token_id(model)
        leading_actual_bos = bool(
            special_ids
            and bos_token_id is not None
            and int(special_ids[0]) == int(bos_token_id)
        )
        expected_context_ids = special_ids[1:] if leading_actual_bos else special_ids
        context_ids = make_context_token_ids(model, text)
        serialized = detokenize_bytes(model, context_ids)
        framing = _prefix_metadata(serialized, prompt_bytes)
        first_real_id = no_special_ids[0] if no_special_ids else None
        first_context_id = context_ids[0] if context_ids else None
        failure_codes: List[str] = []
        if not no_special_ids:
            failure_codes.append("empty_no_special_prompt_tokenization")
        if not context_ids:
            failure_codes.append("empty_prompt_context")
        if context_ids != expected_context_ids:
            failure_codes.append("actual_bos_removal_mismatch")
        if context_ids != no_special_ids:
            failure_codes.append("prompt_context_differs_from_no_special_tokens")
        if first_real_id is None or first_context_id != first_real_id:
            failure_codes.append("first_real_prompt_token_not_retained")
        if not framing["suffix_exact"]:
            failure_codes.append("prompt_not_exact_suffix")
        if not framing["prefix_permitted"]:
            failure_codes.append("prompt_non_space_or_non_prefix_transformation")
        if not framing["exact_original_byte_recovery"]:
            failure_codes.append("original_prompt_bytes_not_recovered")
        record = dict(base)
        record.update(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "preflight_id": PREFLIGHT_ID,
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "execution_mode": EXECUTION_MODE,
                "record_type": "prompt_context",
                "model_id": str(model_id),
                "bos_token_id": bos_token_id,
                "leading_actual_bos_removed": leading_actual_bos,
                "removed_token_count": 1 if leading_actual_bos else 0,
                "special_enabled_token_ids": special_ids,
                "no_special_token_ids": no_special_ids,
                "expected_context_token_ids": expected_context_ids,
                "context_token_ids": list(map(int, context_ids)),
                "special_enabled_token_count": len(special_ids),
                "no_special_token_count": len(no_special_ids),
                "context_token_count": len(context_ids),
                "first_no_special_token_id": first_real_id,
                "first_context_token_id": first_context_id,
                "first_real_token_retained": first_context_id == first_real_id,
                "context_matches_no_special_tokenization": context_ids == no_special_ids,
                "context_matches_actual_bos_only_removal": context_ids
                == expected_context_ids,
                "framing": framing,
                "audit_status": "pass" if not failure_codes else "fail",
                "failure_codes": failure_codes,
            }
        )
        return record
    except Exception as exc:
        return _audit_exception_record(
            "prompt_context", model_id, prompt_id, exc, base
        )


def collect_preflight_prompts(
    configs: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Return the frozen 18 English, six Spanish, and six Mandarin prompts."""

    prompts: List[Dict[str, object]] = []
    for row in flatten_prompt_templates(configs["prompts"]):
        prompts.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "category_id": str(row["category_id"]),
                "language": "en",
                "text": str(row["text"]),
            }
        )
    for language in configs["multilingual"].get("languages", []):
        if not isinstance(language, dict):
            raise TokenizerPreflightError("multilingual language entry is not an object")
        language_id = str(language.get("language_id"))
        by_category = language.get("prompts_by_category", {})
        if not isinstance(by_category, dict):
            raise TokenizerPreflightError("multilingual prompts_by_category is not an object")
        for category_id, text in by_category.items():
            prompts.append(
                {
                    "prompt_id": "multilingual_{}_{}".format(
                        language_id, category_id
                    ),
                    "category_id": str(category_id),
                    "language": language_id,
                    "text": str(text),
                }
            )
    language_counts: Dict[str, int] = {}
    for row in prompts:
        language_counts[row["language"]] = language_counts.get(row["language"], 0) + 1
    if language_counts != {"en": 18, "es": 6, "zh_hans": 6}:
        raise TokenizerPreflightError(
            "Expected prompt counts en=18, es=6, zh_hans=6; got {}".format(
                language_counts
            )
        )
    identifiers = [row["prompt_id"] for row in prompts]
    if len(set(identifiers)) != 30:
        raise TokenizerPreflightError("Preflight prompt identifiers are not unique")
    return prompts


def load_vocab_only_tokenizer(model_path: Path) -> Any:
    """Load only a GGUF vocabulary; no model tensors, context, or GPU work."""

    preload_pip_cuda_libraries()
    try:
        from llama_cpp import Llama
    except (ImportError, RuntimeError) as exc:
        raise TokenizerPreflightError(
            "llama-cpp-python is required for the tokenizer preflight"
        ) from exc
    return Llama(
        model_path=str(Path(model_path)),
        vocab_only=True,
        n_gpu_layers=0,
        n_ctx=16,
        logits_all=False,
        verbose=False,
    )


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_verified_inputs(
    project_root: Path,
    config_dir: Path,
) -> Tuple[Dict[str, Dict[str, object]], List[Any], Dict[str, object]]:
    configs = load_revision_config_set(config_dir)
    config_manifest_path = config_dir / "config_manifest.json"
    config_manifest = load_json_object(config_manifest_path)
    config_verification = verify_directory_manifest(
        config_dir,
        config_manifest,
        require_no_extra_files=True,
        ignored_extra_paths=("config_manifest.json",),
    )
    if config_verification["status"] != "ok":
        raise TokenizerPreflightError(
            "Frozen config manifest verification failed: {}".format(
                "; ".join(config_verification["errors"])
            )
        )
    model_verification = verify_model_artifact_pins(
        configs["models"], project_root=project_root, verify_sha256=True
    )
    if model_verification["status"] != "ok":
        raise TokenizerPreflightError(
            "Pinned model verification failed: {}".format(
                "; ".join(model_verification["errors"])
            )
        )
    payloads = generate_revision_v1_payloads()
    corpus_validation = validate_revision_corpus(
        payloads, expected_sha256=REVISION_CORPUS_SHA256
    )
    if corpus_validation["status"] != "ok":
        raise TokenizerPreflightError(
            "Frozen payload corpus verification failed: {}".format(
                "; ".join(corpus_validation["errors"])
            )
        )
    source_manifest = build_directory_manifest(
        project_root,
        relative_paths=RUNTIME_SOURCE_PATHS,
        exclude_paths=(),
    )
    selected_config_files = ("models.json", "prompts.json", "multilingual.json")
    inputs = {
        "config": {
            "config_manifest_relative_path": str(
                config_manifest_path.relative_to(project_root).as_posix()
            ),
            "config_manifest_file_sha256": file_sha256(config_manifest_path),
            "config_files_sha256": str(config_manifest["files_sha256"]),
            "selected_files": [
                {
                    "relative_path": str((config_dir / name).relative_to(project_root).as_posix()),
                    "sha256": file_sha256(config_dir / name),
                    "size_bytes": (config_dir / name).stat().st_size,
                }
                for name in selected_config_files
            ],
            "verification": config_verification,
        },
        "corpus": {
            "corpus_id": REVISION_CORPUS_ID,
            "expected_corpus_sha256": REVISION_CORPUS_SHA256,
            "actual_corpus_sha256": revision_corpus_sha256(payloads),
            "payload_count": len(payloads),
            "payload_records_sha256": canonical_json_sha256(
                revision_payload_records(payloads, include_payload_text=True)
            ),
            "validation": corpus_validation,
        },
        "models": model_verification,
        "source": source_manifest,
    }
    return configs, payloads, inputs


def run_tokenizer_preflight(
    project_root: Path,
    config_dir: Optional[Path] = None,
    model_loader: Callable[[Path], Any] = load_vocab_only_tokenizer,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Run all 1,530 tokenizer-only checks and return an unpublished bundle."""

    project_root = Path(project_root).resolve()
    config_dir = (
        Path(config_dir).resolve()
        if config_dir is not None
        else (project_root / "configs" / "revision_v1").resolve()
    )
    configs, payloads, inputs = _load_verified_inputs(project_root, config_dir)
    prompts = collect_preflight_prompts(configs)
    model_entries = configs["models"].get("models", [])
    if not isinstance(model_entries, list) or len(model_entries) != 3:
        raise TokenizerPreflightError("Tokenizer preflight requires exactly three models")
    records: List[Dict[str, object]] = []
    model_summaries: List[Dict[str, object]] = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            raise TokenizerPreflightError("Model entry is not an object")
        model_id = str(entry["model_id"])
        model_path = project_root / str(entry["relative_path"])
        model = model_loader(model_path)
        try:
            first_index = len(records)
            records.extend(
                audit_payload_tokenization(model, model_id, payload)
                for payload in payloads
            )
            records.extend(
                audit_prompt_context(model, model_id, prompt) for prompt in prompts
            )
            selected = records[first_index:]
            failures = [row for row in selected if row["audit_status"] != "pass"]
            prefix_counts: Dict[str, int] = {}
            for row in selected:
                framing = row.get("framing")
                if isinstance(framing, dict):
                    key = str(framing.get("prefix_byte_length"))
                    prefix_counts[key] = prefix_counts.get(key, 0) + 1
            model_summaries.append(
                {
                    "model_id": model_id,
                    "payload_check_count": len(payloads),
                    "prompt_check_count": len(prompts),
                    "check_count": len(selected),
                    "pass_count": len(selected) - len(failures),
                    "failure_count": len(failures),
                    "permitted_prefix_byte_length_counts": prefix_counts,
                }
            )
        finally:
            del model
            gc.collect()
    for index, record in enumerate(records):
        record["record_index"] = index
    failures = [row for row in records if row["audit_status"] != "pass"]
    record_type_counts = {
        "payload_tokenization": sum(
            row["record_type"] == "payload_tokenization" for row in records
        ),
        "prompt_context": sum(row["record_type"] == "prompt_context" for row in records),
    }
    if record_type_counts != {"payload_tokenization": 1440, "prompt_context": 90}:
        raise TokenizerPreflightError(
            "Preflight check arithmetic mismatch: {}".format(record_type_counts)
        )
    manifest: Dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "manifest_type": "rankcloak_tokenizer_preflight",
        "preflight_id": PREFLIGHT_ID,
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "payload_tokenization_contract": PAYLOAD_TOKENIZATION_CONTRACT,
        "prompt_context_contract": PROMPT_CONTEXT_CONTRACT,
        "permitted_prefix_policy": PREFIX_POLICY,
        "execution_mode": EXECUTION_MODE,
        "generation_or_context_evaluation_performed": False,
        "deterministic_no_timestamp": True,
        "status": "pass" if not failures else "fail",
        "counts": {
            "model_count": len(model_entries),
            "payloads_per_model": len(payloads),
            "prompts_per_model": len(prompts),
            "prompt_language_counts_per_model": {"en": 18, "es": 6, "zh_hans": 6},
            "payload_check_count": record_type_counts["payload_tokenization"],
            "prompt_check_count": record_type_counts["prompt_context"],
            "total_check_count": len(records),
            "pass_count": len(records) - len(failures),
            "failure_count": len(failures),
        },
        "model_summaries": model_summaries,
        "inputs": inputs,
        "runtime": {
            "llama_cpp_python_version": _package_version("llama-cpp-python"),
            "python_implementation_contract": "CPython-compatible deterministic tokenization",
        },
    }
    return manifest, records


def _finalize_manifest(
    manifest: Mapping[str, object],
    records_bytes: bytes,
    failures_bytes: bytes,
) -> Dict[str, object]:
    value = dict(manifest)
    value["output_files"] = [
        {
            "path": RECORDS_NAME,
            "size_bytes": len(records_bytes),
            "sha256": _sha256_bytes(records_bytes),
            "record_count": records_bytes.count(b"\n"),
        },
        {
            "path": FAILURES_NAME,
            "size_bytes": len(failures_bytes),
            "sha256": _sha256_bytes(failures_bytes),
            "record_count": failures_bytes.count(b"\n"),
        },
    ]
    value.pop(MANIFEST_HASH_FIELD, None)
    value[MANIFEST_HASH_FIELD] = canonical_json_sha256(value)
    return value


def write_preflight_output(
    output_dir: Path,
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Publish a complete immutable directory by one same-filesystem rename."""

    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ImmutableArtifactError(
            "Tokenizer preflight output already exists; refusing overwrite: {}".format(
                output_dir
            )
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ordered = [dict(row) for row in records]
    failures = [row for row in ordered if row.get("audit_status") != "pass"]
    records_bytes = _jsonl_bytes(ordered)
    failures_bytes = _jsonl_bytes(failures)
    final_manifest = _finalize_manifest(manifest, records_bytes, failures_bytes)
    manifest_bytes = json.dumps(
        final_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    temporary = Path(
        tempfile.mkdtemp(prefix=".{}-".format(output_dir.name), dir=str(output_dir.parent))
    )
    published = False
    try:
        (temporary / RECORDS_NAME).write_bytes(records_bytes)
        (temporary / FAILURES_NAME).write_bytes(failures_bytes)
        (temporary / MANIFEST_NAME).write_bytes(manifest_bytes)
        for path in (temporary / RECORDS_NAME, temporary / FAILURES_NAME, temporary / MANIFEST_NAME):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(str(temporary), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_dir.exists() or output_dir.is_symlink():
            raise ImmutableArtifactError(
                "Tokenizer preflight output appeared concurrently; refusing overwrite: {}".format(
                    output_dir
                )
            )
        os.rename(str(temporary), str(output_dir))
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    verify_preflight_output(output_dir)
    return final_manifest


def verify_preflight_output(output_dir: Path) -> Dict[str, object]:
    """Verify self-hash, file hashes, counts, and v2 record contracts."""

    output_dir = Path(output_dir)
    expected_names = {MANIFEST_NAME, RECORDS_NAME, FAILURES_NAME}
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise TokenizerPreflightError("Preflight output is missing or invalid")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise TokenizerPreflightError(
            "Preflight output file set mismatch: {}".format(sorted(actual_names))
        )
    manifest = load_json_object(output_dir / MANIFEST_NAME)
    stored_hash = manifest.get(MANIFEST_HASH_FIELD)
    unhashed = dict(manifest)
    unhashed.pop(MANIFEST_HASH_FIELD, None)
    if stored_hash != canonical_json_sha256(unhashed):
        raise TokenizerPreflightError("Preflight manifest self-hash mismatch")
    file_records = manifest.get("output_files", [])
    if not isinstance(file_records, list) or len(file_records) != 2:
        raise TokenizerPreflightError("Preflight output_files must contain two entries")
    for record in file_records:
        path = output_dir / str(record["path"])
        if path.stat().st_size != int(record["size_bytes"]):
            raise TokenizerPreflightError("Preflight output size mismatch: {}".format(path.name))
        if file_sha256(path) != record["sha256"]:
            raise TokenizerPreflightError("Preflight output SHA-256 mismatch: {}".format(path.name))
    records = [
        json.loads(line)
        for line in (output_dir / RECORDS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    failures = [
        json.loads(line)
        for line in (output_dir / FAILURES_NAME).read_text(encoding="utf-8").splitlines()
    ]
    counts = manifest.get("counts", {})
    if len(records) != int(counts.get("total_check_count", -1)):
        raise TokenizerPreflightError("Preflight total record count mismatch")
    expected_failures = [row for row in records if row.get("audit_status") != "pass"]
    if failures != expected_failures or len(failures) != int(counts.get("failure_count", -1)):
        raise TokenizerPreflightError("Preflight failure records mismatch")
    for index, record in enumerate(records):
        if record.get("record_index") != index:
            raise TokenizerPreflightError("Preflight record_index sequence mismatch")
        if record.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION:
            raise TokenizerPreflightError("Preflight record protocol revision mismatch")
        if record.get("result_schema_revision") != RESULT_SCHEMA_REVISION:
            raise TokenizerPreflightError("Preflight record result schema mismatch")
    status = "pass" if not failures else "fail"
    if manifest.get("status") != status:
        raise TokenizerPreflightError("Preflight manifest status mismatch")
    return {
        "status": "ok",
        "scientific_status": status,
        "preflight_manifest_sha256": stored_hash,
        "record_count": len(records),
        "failure_count": len(failures),
        "records_sha256": file_sha256(output_dir / RECORDS_NAME),
        "failures_sha256": file_sha256(output_dir / FAILURES_NAME),
    }

