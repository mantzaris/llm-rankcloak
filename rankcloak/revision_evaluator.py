"""Deterministic cross-family held-out log-probability evaluation.

The evaluator consumes only completed, immutable revision-runner shards.  Each
invocation loads exactly one content-pinned llama.cpp evaluator and accepts
source text from exactly one *different* generator family according to the
predeclared cyclic map below.  Text is scored serially under its recorded overt
prompt; segmented RankCloak messages are scored segment-by-segment and then
aggregated by evaluator-token count.

No value produced by this module is a human-rating substitute.  Limited runs
are relabelled exploratory and cannot be pooled with confirmatory evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model_io import (
    evaluate_context,
    get_last_logits,
    load_llama_cpp_model,
    make_context_token_ids,
    preload_pip_cuda_libraries,
    tokenize_bytes,
)
from .rank_codec import token_log_probability
from .revision_artifacts import (
    ArtifactIntegrityError,
    build_run_identity_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    initialize_checkpoint,
    load_checkpoint,
    pending_trial_ids,
    record_checkpoint_result,
    save_checkpoint,
    trial_ids_sha256,
    verify_directory_manifest,
    write_immutable_json,
    write_immutable_jsonl,
)
from .revision_config import (
    DEFAULT_REVISION_CONFIG_DIR,
    load_revision_config_set,
    verify_model_artifact_pins,
)
from .revision_runner import (
    EVIDENCE_ABLATION_V2,
    EVIDENCE_MULTILINGUAL_V2,
    EVIDENCE_PRIMARY_V2,
    EVIDENCE_SMOKE_V3,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCHEMA_VERSION = "2.0"
STANDARD_SOURCE_STAGES = ("primary_v2", "ablation_v2", "multilingual_v2")
SMOKE_SOURCE_STAGE = "smoke_v3"
SOURCE_STAGES = STANDARD_SOURCE_STAGES + (SMOKE_SOURCE_STAGE,)
SOURCE_RECORD_TYPES = ("rankcloak_trial", "ordinary_control")

LLAMA_MODEL_ID = "llama3_8b_instruct_q4_k_m"
QWEN_MODEL_ID = "qwen2_5_7b_instruct_q4_k_m"
MISTRAL_MODEL_ID = "mistral_7b_instruct_v0_3_q4_k_m"

# This map is part of the hashed source contract, not a runtime choice.
EVALUATOR_BY_GENERATOR = {
    LLAMA_MODEL_ID: QWEN_MODEL_ID,
    QWEN_MODEL_ID: MISTRAL_MODEL_ID,
    MISTRAL_MODEL_ID: LLAMA_MODEL_ID,
}
GENERATOR_BY_EVALUATOR = {
    evaluator: generator for generator, evaluator in EVALUATOR_BY_GENERATOR.items()
}

SOURCE_EVIDENCE_BY_STAGE = {
    "primary_v2": EVIDENCE_PRIMARY_V2,
    "ablation_v2": EVIDENCE_ABLATION_V2,
    "multilingual_v2": EVIDENCE_MULTILINGUAL_V2,
    SMOKE_SOURCE_STAGE: EVIDENCE_SMOKE_V3,
}
EVIDENCE_BY_STAGE = {
    "primary_v2": (
        "confirmatory_heldout_evaluator_primary_v2_payload_fidelity_after_source_manifest_freeze"
    ),
    "ablation_v2": (
        "confirmatory_supporting_heldout_evaluator_ablation_v2_payload_fidelity_after_source_manifest_freeze"
    ),
    "multilingual_v2": (
        "secondary_supplementary_heldout_evaluator_multilingual_v2_payload_fidelity_after_source_manifest_freeze"
    ),
    SMOKE_SOURCE_STAGE: EVIDENCE_SMOKE_V3,
}
EVIDENCE_PARTITION_BY_STAGE = {
    "primary_v2": "confirmatory_primary_v2_payload_fidelity_v2",
    "ablation_v2": "confirmatory_supporting_ablation_v2_payload_fidelity_v2",
    "multilingual_v2": "secondary_supplementary_multilingual_v2_payload_fidelity_v2",
    SMOKE_SOURCE_STAGE: (
        "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling"
    ),
}
EVIDENCE_LIMITED = "exploratory_limited_not_for_confirmatory_pooling"


class RevisionEvaluatorError(RuntimeError):
    """Raised when evaluator inputs or execution violate the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, value: object) -> str:
    digest = canonical_json_sha256(value)[:24]
    return "{}__{}".format(prefix, digest)


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise RevisionEvaluatorError("Non-finite evaluator value cannot be serialized")
    return value


def _load_json_object(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("Cannot load JSON artifact {}".format(path)) from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError("JSON artifact is not an object: {}".format(path))
    return value


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("Missing JSONL artifact: {}".format(path)) from exc
    rows: List[Dict[str, object]] = []
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(
                    "Invalid JSONL at {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise ArtifactIntegrityError(
                    "JSONL row at {}:{} is not an object".format(path, line_number)
                )
            rows.append(row)
    return rows


def _append_jsonl_fsync(path: Path, row: Mapping[str, object]) -> None:
    """Append one canonical JSON record under a process lock and fsync it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactIntegrityError("Refusing append through symlink: {}".format(path))
    content = canonical_json_bytes(_json_safe(row)) + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Linux is the execution target
            pass
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("Short append to {}".format(path))
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactIntegrityError("Refusing write through symlink: {}".format(path))
    content = b"".join(canonical_json_bytes(_json_safe(row)) + b"\n" for row in rows)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".{}.".format(path.name), suffix=".tmp",
            dir=str(path.parent), delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def verify_config_manifest(config_dir: Path) -> Dict[str, object]:
    manifest_path = Path(config_dir) / "config_manifest.json"
    manifest = _load_json_object(manifest_path)
    report = verify_directory_manifest(
        Path(config_dir), manifest, require_no_extra_files=True
    )
    if report["status"] != "ok":
        raise ArtifactIntegrityError(
            "Frozen config manifest failed: {}".format("; ".join(report["errors"]))
        )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "files_sha256": manifest.get("files_sha256"),
        "verified_file_count": report["verified_file_count"],
    }


def _verify_internal_file_list(manifest: Mapping[str, object], label: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ArtifactIntegrityError("{}.files is not a list".format(label))
    expected = canonical_json_sha256(files)
    if expected != manifest.get("files_sha256"):
        raise ArtifactIntegrityError("{} file-list hash mismatch".format(label))


def _verify_run_identity(identity: Mapping[str, object]) -> None:
    claimed = identity.get("run_identity_sha256")
    unsigned = dict(identity)
    unsigned.pop("run_identity_sha256", None)
    if canonical_json_sha256(unsigned) != claimed:
        raise ArtifactIntegrityError("Runner run_identity_sha256 mismatch")


def _identity_argument(identity: Mapping[str, object], name: str) -> str:
    prefix = "{}=".format(name)
    matches = [
        str(value)[len(prefix) :]
        for value in identity.get("command_line_args", [])
        if str(value).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ArtifactIntegrityError(
            "Runner identity does not bind exactly one {}".format(name)
        )
    return matches[0]


def _model_entry(
    configs: Mapping[str, Mapping[str, object]], model_id: str
) -> Dict[str, object]:
    matches = [
        dict(row)
        for row in configs["models"].get("models", [])
        if isinstance(row, dict) and row.get("model_id") == str(model_id)
    ]
    if len(matches) != 1:
        raise RevisionEvaluatorError(
            "Expected one pinned model entry for {}".format(model_id)
        )
    return matches[0]


def verify_evaluator_model(
    configs: Mapping[str, Mapping[str, object]],
    evaluator_model_id: str,
    project_root: Path,
) -> Dict[str, object]:
    entry = _model_entry(configs, evaluator_model_id)
    verification = verify_model_artifact_pins(
        {"models": [entry]}, project_root=project_root, verify_sha256=True
    )
    if verification["status"] != "ok":
        raise ArtifactIntegrityError(
            "Pinned evaluator artifact failed: {}".format(
                "; ".join(verification["errors"])
            )
        )
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "configured_model": entry,
        "verification": verification["records"][0],
        "role": "heldout_evaluator_only",
    }


def evaluator_for_generator(generator_model_id: str) -> str:
    try:
        evaluator = EVALUATOR_BY_GENERATOR[str(generator_model_id)]
    except KeyError as exc:
        raise RevisionEvaluatorError(
            "Generator has no predeclared held-out evaluator: {}".format(
                generator_model_id
            )
        ) from exc
    if evaluator == str(generator_model_id):  # defensive invariant
        raise RevisionEvaluatorError("Held-out evaluator cannot equal generator")
    return evaluator


def generator_for_evaluator(evaluator_model_id: str) -> str:
    try:
        generator = GENERATOR_BY_EVALUATOR[str(evaluator_model_id)]
    except KeyError as exc:
        raise RevisionEvaluatorError(
            "Evaluator is outside the predeclared cyclic map: {}".format(
                evaluator_model_id
            )
        ) from exc
    if generator == str(evaluator_model_id):
        raise RevisionEvaluatorError("Held-out evaluator cannot equal generator")
    return generator


def _required_runner_paths(run_dir: Path) -> Dict[str, Path]:
    names = (
        "records.jsonl",
        "plan.jsonl",
        "checkpoint.json",
        "run_identity.json",
        "model_manifest.json",
        "payload_manifest.json",
        "source_manifest.json",
        "runtime_manifest.json",
        "hardware_manifest.json",
    )
    paths = {name: Path(run_dir) / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ArtifactIntegrityError(
            "Runner shard missing required regular files: {}".format(", ".join(missing))
        )
    return paths


def verify_completed_runner_shard(
    run_dir: Path,
    stage: str,
    generator_model_id: str,
    config_manifest_sha256: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Verify one completed runner shard without opening its generator GGUF."""

    if stage not in SOURCE_STAGES:
        raise RevisionEvaluatorError("Unsupported evaluator source stage: {}".format(stage))
    paths = _required_runner_paths(run_dir)
    identity = _load_json_object(paths["run_identity.json"])
    _verify_run_identity(identity)
    if (
        identity.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or identity.get("result_schema_revision") != RESULT_SCHEMA_REVISION
    ):
        raise ArtifactIntegrityError(
            "Runner identity lacks the payload-fidelity-v2 result contract"
        )
    if (
        _identity_argument(identity, "stage") != stage
        or _identity_argument(identity, "model_id") != generator_model_id
        or _identity_argument(identity, "protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or _identity_argument(identity, "result_schema_revision")
        != RESULT_SCHEMA_REVISION
    ):
        raise ArtifactIntegrityError("Runner scientific identity contract mismatch")
    if identity.get("config_manifest_sha256") != str(config_manifest_sha256):
        raise ArtifactIntegrityError("Runner shard config manifest does not match current freeze")
    if identity.get("payload_manifest_sha256") != file_sha256(paths["payload_manifest.json"]):
        raise ArtifactIntegrityError("Runner payload manifest hash mismatch")

    model_manifest = _load_json_object(paths["model_manifest.json"])
    embedded_models = identity.get("model_artifacts")
    if embedded_models != [model_manifest]:
        raise ArtifactIntegrityError("Runner model manifest is not identity-bound")
    configured = model_manifest.get("configured_model", {})
    if not isinstance(configured, dict) or configured.get("model_id") != generator_model_id:
        raise ArtifactIntegrityError("Runner shard generator model mismatch")
    model_verification = model_manifest.get("verification", {})
    expected_generator_hash = configured.get("artifact_sha256")
    if (
        not isinstance(model_verification, dict)
        or model_verification.get("status") != "ok"
        or model_verification.get("actual_sha256") != expected_generator_hash
        or model_verification.get("expected_sha256") != expected_generator_hash
    ):
        raise ArtifactIntegrityError(
            "Runner generator artifact verification is incomplete or inconsistent"
        )

    source_manifest = _load_json_object(paths["source_manifest.json"])
    _verify_internal_file_list(source_manifest, "runner source manifest")
    for argument, filename in (
        ("source_manifest_sha256", "source_manifest.json"),
        ("runtime_manifest_sha256", "runtime_manifest.json"),
        ("hardware_manifest_sha256", "hardware_manifest.json"),
    ):
        if _identity_argument(identity, argument) != file_sha256(paths[filename]):
            raise ArtifactIntegrityError(
                "Runner identity {} mismatch".format(argument)
            )

    plan = load_jsonl(paths["plan.jsonl"])
    expected_source_evidence = SOURCE_EVIDENCE_BY_STAGE[stage]
    if any(
        row.get("evidence_status") != expected_source_evidence
        or row.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        for row in plan
    ):
        raise ArtifactIntegrityError(
            "Runner plan violates its v2 source evidence/result contract"
        )
    planned_ids = [str(row.get("work_id")) for row in plan]
    if any(value in {"", "None"} for value in planned_ids):
        raise ArtifactIntegrityError("Runner plan has a missing work_id")
    if len(set(planned_ids)) != len(planned_ids):
        raise ArtifactIntegrityError("Runner plan has duplicate work IDs")
    if identity.get("planned_trial_count") != len(planned_ids):
        raise ArtifactIntegrityError("Runner identity planned count mismatch")
    if identity.get("planned_trial_ids_sha256") != trial_ids_sha256(planned_ids):
        raise ArtifactIntegrityError("Runner ordered plan hash mismatch")
    expected_study_id = str(identity.get("study_id", ""))
    if "/{}/{}".format(stage, generator_model_id) not in expected_study_id:
        raise ArtifactIntegrityError("Runner study identity does not match stage/model")

    checkpoint = load_checkpoint(paths["checkpoint.json"])
    if checkpoint.get("planned_trial_ids_sha256") != trial_ids_sha256(planned_ids):
        raise ArtifactIntegrityError("Runner checkpoint ordered plan hash mismatch")
    if checkpoint.get("planned_trial_count") != len(planned_ids):
        raise ArtifactIntegrityError("Runner checkpoint planned count mismatch")
    if set(map(str, checkpoint.get("completed_trial_ids", []))) != set(planned_ids):
        raise ArtifactIntegrityError("Runner shard is not complete")
    if checkpoint.get("failed_trial_ids"):
        raise ArtifactIntegrityError("Runner shard has unresolved failed work")

    records = load_jsonl(paths["records.jsonl"])
    completed: Dict[str, Dict[str, object]] = {}
    for record in records:
        work_id = str(record.get("work_id"))
        if work_id not in set(planned_ids):
            raise ArtifactIntegrityError("Runner record has unplanned work ID")
        if record.get("execution_status") == "completed":
            if work_id in completed:
                raise ArtifactIntegrityError("Runner record has duplicate completion")
            completed[work_id] = record
    if set(completed) != set(planned_ids):
        raise ArtifactIntegrityError("Runner checkpoint has no one-to-one durable completions")

    selected: List[Dict[str, object]] = []
    plan_by_id = {str(row["work_id"]): row for row in plan}
    for work_id in planned_ids:
        record = completed[work_id]
        task = plan_by_id[work_id]
        if record.get("record_type") not in SOURCE_RECORD_TYPES:
            continue
        if str(record.get("model_id")) != generator_model_id:
            raise ArtifactIntegrityError("Source record generator-model mismatch")
        if (
            record.get("evidence_status") != expected_source_evidence
            or record.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or record.get("result_schema_revision") != RESULT_SCHEMA_REVISION
            or record.get("study_phase") != task.get("study_phase")
        ):
            raise ArtifactIntegrityError(
                "Source record violates its v2 evidence/phase/result contract"
            )
        selected.append(record)

    files = [
        {
            "role": name.rsplit(".", 1)[0],
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in sorted(paths.items())
    ]
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_type": "heldout_evaluator_verified_runner_input",
        "stage": stage,
        "generator_model_id": generator_model_id,
        "runner_planned_count": len(planned_ids),
        "selected_score_record_count": len(selected),
        "files": files,
        "files_sha256": canonical_json_sha256(files),
        "generator_artifact_sha256": configured.get("artifact_sha256"),
        "generator_artifact_verification": model_manifest.get("verification"),
        "generator_artifact_opened_by_evaluator": False,
        "evidence_partition": EVIDENCE_PARTITION_BY_STAGE[stage],
        "confirmatory_pooling_eligible": stage != SMOKE_SOURCE_STAGE,
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
    }
    return selected, manifest


def build_prompt_lookup(
    configs: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Build the canonical content-hashed English and multilingual prompt lookup."""

    entries: List[Dict[str, object]] = []

    def add_entry(
        language: object,
        prompt_id: object,
        prompt_category: object,
        prompt_text: object,
        source_config: str,
    ) -> None:
        values = tuple(
            str(value) for value in (language, prompt_id, prompt_category, prompt_text)
        )
        if any(not value for value in values):
            raise ArtifactIntegrityError(
                "Frozen prompt lookup contains an empty identity or text field"
            )
        text_value = values[3]
        entries.append(
            {
                "language": values[0],
                "prompt_id": values[1],
                "prompt_category": values[2],
                "prompt_text": text_value,
                "prompt_text_sha256": hashlib.sha256(
                    text_value.encode("utf-8")
                ).hexdigest(),
                "source_config": source_config,
            }
        )

    prompt_config = configs.get("prompts")
    if not isinstance(prompt_config, Mapping):
        raise ArtifactIntegrityError("Frozen configs lack prompts.json content")
    english_language = prompt_config.get("language", "en")
    categories = prompt_config.get("categories")
    if not isinstance(categories, list):
        raise ArtifactIntegrityError("Frozen prompts.categories is not a list")
    for category in categories:
        if not isinstance(category, Mapping) or not isinstance(
            category.get("templates"), list
        ):
            raise ArtifactIntegrityError("Malformed frozen English prompt category")
        for template in category["templates"]:
            if not isinstance(template, Mapping):
                raise ArtifactIntegrityError("Malformed frozen English prompt template")
            add_entry(
                english_language,
                template.get("prompt_id"),
                category.get("category_id"),
                template.get("text"),
                "prompts.json",
            )

    multilingual_config = configs.get("multilingual")
    if not isinstance(multilingual_config, Mapping):
        raise ArtifactIntegrityError("Frozen configs lack multilingual.json content")
    languages = multilingual_config.get("languages")
    if not isinstance(languages, list):
        raise ArtifactIntegrityError("Frozen multilingual.languages is not a list")
    for language in languages:
        if not isinstance(language, Mapping) or not isinstance(
            language.get("prompts_by_category"), Mapping
        ):
            raise ArtifactIntegrityError("Malformed frozen multilingual prompt entry")
        language_id = str(language.get("language_id"))
        for category_id, prompt_text in language["prompts_by_category"].items():
            add_entry(
                language_id,
                "{}_{}_01".format(language_id, category_id),
                category_id,
                prompt_text,
                "multilingual.json",
            )

    entries.sort(
        key=lambda row: (
            str(row["language"]), str(row["prompt_id"]),
            str(row["prompt_category"]),
        )
    )
    keys = [
        (str(row["language"]), str(row["prompt_id"]), str(row["prompt_category"]))
        for row in entries
    ]
    if len(keys) != len(set(keys)):
        raise ArtifactIntegrityError("Frozen prompt lookup has duplicate identities")
    lookup: Dict[str, object] = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_type": "heldout_evaluator_frozen_prompt_lookup",
        "entry_count": len(entries),
        "entries": entries,
    }
    lookup["prompt_lookup_sha256"] = canonical_json_sha256(lookup)
    return lookup


def _validated_prompt_lookup(
    prompt_lookup: Optional[Mapping[str, object]],
) -> Tuple[Dict[Tuple[str, str, str], Mapping[str, object]], str]:
    if not isinstance(prompt_lookup, Mapping):
        raise ArtifactIntegrityError(
            "Ordinary-control prompt text requires a verified frozen prompt lookup"
        )
    unhashed = dict(prompt_lookup)
    claimed = unhashed.pop("prompt_lookup_sha256", None)
    if not isinstance(claimed, str) or canonical_json_sha256(unhashed) != claimed:
        raise ArtifactIntegrityError("Frozen prompt lookup self-hash mismatch")
    entries = prompt_lookup.get("entries")
    if not isinstance(entries, list) or prompt_lookup.get("entry_count") != len(entries):
        raise ArtifactIntegrityError("Frozen prompt lookup entry count mismatch")
    indexed: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArtifactIntegrityError("Frozen prompt lookup entry is not an object")
        key = (
            str(entry.get("language")), str(entry.get("prompt_id")),
            str(entry.get("prompt_category")),
        )
        text_value = entry.get("prompt_text")
        if not isinstance(text_value, str) or hashlib.sha256(
            text_value.encode("utf-8")
        ).hexdigest() != entry.get("prompt_text_sha256"):
            raise ArtifactIntegrityError("Frozen prompt lookup text hash mismatch")
        if key in indexed:
            raise ArtifactIntegrityError("Frozen prompt lookup has duplicate identities")
        indexed[key] = entry
    return indexed, claimed


def _source_segments(
    record: Mapping[str, object],
    prompt_lookup: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, object]]:
    record_type = str(record.get("record_type"))
    if record_type == "rankcloak_trial":
        raw_segments = record.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ArtifactIntegrityError("RankCloak source has no recorded segments")
        segments: List[Dict[str, object]] = []
        for position, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise ArtifactIntegrityError("RankCloak segment is not an object")
            prompt = raw.get("prompt")
            if not isinstance(prompt, dict):
                raise ArtifactIntegrityError("RankCloak segment lacks its recorded prompt")
            prompt_text = prompt.get("prompt_text")
            text = raw.get("full_text")
            if not isinstance(prompt_text, str) or not isinstance(text, str):
                raise ArtifactIntegrityError("RankCloak segment prompt/text is not a string")
            segments.append(
                {
                    "segment_index": int(raw.get("segment_index", position)),
                    "prompt_id": prompt.get("prompt_id"),
                    "prompt_category": prompt.get("prompt_category"),
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
        if [row["segment_index"] for row in segments] != list(range(len(segments))):
            raise ArtifactIntegrityError("RankCloak segment indices are not contiguous and ordered")
        expected_full = "\n\n".join(str(row["text"]) for row in segments)
        if record.get("full_text") != expected_full:
            raise ArtifactIntegrityError("RankCloak full text disagrees with recorded segments")
        return segments

    if record_type == "ordinary_control":
        generation = record.get("generation")
        if not isinstance(generation, dict):
            raise ArtifactIntegrityError("Control source lacks generation metadata")
        indexed_prompts, lookup_sha256 = _validated_prompt_lookup(prompt_lookup)
        language = str(record.get("language") or "en")
        prompt_id = str(record.get("prompt_id"))
        prompt_category = str(record.get("prompt_category"))
        key = (language, prompt_id, prompt_category)
        if key not in indexed_prompts:
            raise ArtifactIntegrityError(
                "Control prompt identity is absent from frozen configs: {}".format(key)
            )
        prompt_entry = indexed_prompts[key]
        prompt_text = str(prompt_entry["prompt_text"])
        recorded_prompt = generation.get("prompt")
        if recorded_prompt is not None and recorded_prompt != prompt_text:
            raise ArtifactIntegrityError(
                "Control generation prompt disagrees with frozen config lookup"
            )
        text = record.get("full_text")
        if not isinstance(text, str):
            raise ArtifactIntegrityError("Control text is not a string")
        if generation.get("text") != text:
            raise ArtifactIntegrityError("Control full text disagrees with generation text")
        return [
            {
                "segment_index": 0,
                "prompt_id": prompt_id,
                "prompt_category": prompt_category,
                "prompt_text": prompt_text,
                "prompt_text_sha256": prompt_entry["prompt_text_sha256"],
                "prompt_source": "verified_frozen_config_lookup",
                "prompt_lookup_sha256": lookup_sha256,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        ]
    raise ArtifactIntegrityError("Unsupported scoring source record type")


def _protocol_variant(record: Mapping[str, object]) -> str:
    if record.get("record_type") == "rankcloak_trial":
        return str(record.get("protocol_variant"))
    return "ordinary_llm_control_{}".format(record.get("control_view", "full_message"))


def build_scoring_plan(
    stage_records: Mapping[str, Sequence[Mapping[str, object]]],
    evaluator_model_id: str,
    generator_model_id: Optional[str] = None,
    generator_artifact_sha256: Optional[str] = None,
    prompt_lookup: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, object]]:
    # Build the content-hashed ordered evaluation plan.
    expected_generator = generator_for_evaluator(evaluator_model_id)
    generator = str(generator_model_id or expected_generator)
    if generator != expected_generator:
        raise RevisionEvaluatorError("Evaluator/generator pair violates cyclic mapping")
    if generator == evaluator_model_id:
        raise RevisionEvaluatorError("Same-model evaluation is forbidden")
    unknown_stages = set(stage_records) - set(SOURCE_STAGES)
    if unknown_stages:
        raise RevisionEvaluatorError(
            "Unknown source stages: {}".format(", ".join(sorted(unknown_stages)))
        )
    if SMOKE_SOURCE_STAGE in stage_records and len(stage_records) != 1:
        raise RevisionEvaluatorError(
            "Smoke-v3 evaluator sources must be isolated from every non-smoke stage"
        )
    prompt_lookup_sha256 = None
    if prompt_lookup is not None:
        _, prompt_lookup_sha256 = _validated_prompt_lookup(prompt_lookup)
    plan: List[Dict[str, object]] = []
    for stage in SOURCE_STAGES:
        for source_position, source in enumerate(stage_records.get(stage, [])):
            if str(source.get("model_id")) != generator:
                raise RevisionEvaluatorError("Scoring plan contains a different generator")
            if str(source.get("record_type")) not in SOURCE_RECORD_TYPES:
                raise RevisionEvaluatorError("Scoring plan contains an unsupported record")
            if (
                source.get("evidence_status") != SOURCE_EVIDENCE_BY_STAGE[stage]
                or source.get("protocol_contract_revision")
                != PROTOCOL_CONTRACT_REVISION
                or source.get("result_schema_revision") != RESULT_SCHEMA_REVISION
            ):
                raise RevisionEvaluatorError(
                    "Scoring source violates its v2 evidence/result contract"
                )
            source_hash = canonical_json_sha256(source)
            segments = _source_segments(source, prompt_lookup=prompt_lookup)
            evidence_partition = EVIDENCE_PARTITION_BY_STAGE[stage]
            pooling_eligible = stage != SMOKE_SOURCE_STAGE
            core = {
                "source_stage": stage,
                "source_position": source_position,
                "source_work_id": str(source["work_id"]),
                "source_record_sha256": source_hash,
                "generator_model_id": generator,
                "evaluator_model_id": evaluator_model_id,
                "generator_artifact_sha256": generator_artifact_sha256,
                "prompt_lookup_sha256": prompt_lookup_sha256,
                "evidence_partition": evidence_partition,
                "confirmatory_pooling_eligible": pooling_eligible,
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "source_protocol_contract_revision": source.get(
                    "protocol_contract_revision"
                ),
                "source_result_schema_revision": source.get(
                    "result_schema_revision"
                ),
                "segments": segments,
            }
            evaluation_prefix = (
                "heldout_eval_smoke_v3_payload_fidelity_v2"
                if stage == SMOKE_SOURCE_STAGE
                else "heldout_eval_payload_fidelity_v2"
            )
            evaluation_id = _stable_id(evaluation_prefix, core)
            raw_trial_id = str(
                source.get("trial_id")
                or source.get("control_id")
                or source["work_id"]
            )
            trial_id = (
                _stable_id(
                    "smoke_v3_source_trial_payload_fidelity_v2",
                    {
                        "source_stage": stage,
                        "source_work_id": source["work_id"],
                        "raw_trial_id": raw_trial_id,
                    },
                )
                if stage == SMOKE_SOURCE_STAGE
                else raw_trial_id
            )
            prompt_texts = [str(segment["prompt_text"]) for segment in segments]
            text_parts = [str(segment["text"]) for segment in segments]
            plan.append(
                {
                    "schema_version": EVALUATOR_SCHEMA_VERSION,
                    "evaluation_id": evaluation_id,
                    **core,
                    "source_record_type": source["record_type"],
                    "source_evidence_status": source.get("evidence_status"),
                    "evidence_status": EVIDENCE_BY_STAGE[stage],
                    "trial_id": trial_id,
                    "source_trial_id_raw": raw_trial_id,
                    "payload_name": source.get("payload_name"),
                    "payload_class": source.get("payload_class"),
                    "payload_split": source.get("payload_split"),
                    "study_phase": source.get("study_phase"),
                    "protocol_variant": _protocol_variant(source),
                    "prompt_id": source.get("prompt_id"),
                    "prompt_category": source.get("prompt_category"),
                    "language": source.get("language"),
                    "text_view": (
                        source.get("control_view", "full_message")
                        if source.get("record_type") == "ordinary_control"
                        else "full_message"
                    ),
                    "text": "\n\n".join(text_parts),
                    "text_sha256": hashlib.sha256(
                        "\n\n".join(text_parts).encode("utf-8")
                    ).hexdigest(),
                    "prompt_text": "\n\n".join(prompt_texts),
                    "prompt_count": len(prompt_texts),
                    "segment_count": len(segments),
                    "prompt_conditioning": (
                        "per_segment_recorded_overt_prompt_token_weighted"
                        if len(segments) > 1
                        else "recorded_overt_prompt"
                    ),
                }
            )
    identifiers = [str(row["evaluation_id"]) for row in plan]
    if len(set(identifiers)) != len(identifiers):
        raise RevisionEvaluatorError("Content-hashed scoring plan has duplicate IDs")
    return plan


def relabel_limited_plan(
    plan: Sequence[Mapping[str, object]], limit: int
) -> List[Dict[str, object]]:
    if int(limit) <= 0:
        raise RevisionEvaluatorError("Limited evaluation count must be positive")
    if any(row.get("source_stage") == SMOKE_SOURCE_STAGE for row in plan):
        raise RevisionEvaluatorError(
            "Smoke-v3 already has a frozen exploratory plan; --limit is forbidden"
        )
    selected: List[Dict[str, object]] = []
    for row in list(plan)[: int(limit)]:
        item = dict(row)
        original_id = str(item["evaluation_id"])
        item["original_frozen_evaluation_id"] = original_id
        item["evaluation_id"] = _stable_id(
            "heldout_eval_limited_v1", {"original": original_id, "limit": int(limit)}
        )
        item["evidence_status"] = EVIDENCE_LIMITED
        selected.append(item)
    return selected


def scoring_plan_summary(plan: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    stage_counts = {
        stage: sum(str(row["source_stage"]) == stage for row in plan)
        for stage in SOURCE_STAGES
    }
    type_counts = {
        kind: sum(str(row["source_record_type"]) == kind for row in plan)
        for kind in SOURCE_RECORD_TYPES
    }
    return {
        "evaluation_task_count": len(plan),
        "ordered_evaluation_ids_sha256": trial_ids_sha256(
            [str(row["evaluation_id"]) for row in plan]
        ),
        "scoring_plan_content_sha256": canonical_json_sha256(list(plan)),
        "source_stage_counts": stage_counts,
        "source_record_type_counts": type_counts,
        "generator_model_ids": sorted({str(row["generator_model_id"]) for row in plan}),
        "evaluator_model_ids": sorted({str(row["evaluator_model_id"]) for row in plan}),
        "evidence_statuses": sorted({str(row["evidence_status"]) for row in plan}),
        "evidence_partitions": sorted(
            {str(row["evidence_partition"]) for row in plan}
        ),
        "confirmatory_pooling_eligible": bool(plan)
        and all(bool(row["confirmatory_pooling_eligible"]) for row in plan),
        "smoke_v3_isolated": bool(plan)
        and all(row["source_stage"] == SMOKE_SOURCE_STAGE for row in plan),
    }


def score_conditioned_text(
    model: Any,
    prompt_text: str,
    text: str,
    context_limit: int = 4096,
) -> Dict[str, object]:
    """Score evaluator-tokenized text serially after its overt prompt."""

    context = make_context_token_ids(model, str(prompt_text))
    targets = list(map(int, tokenize_bytes(model, str(text).encode("utf-8"), add_bos=False)))
    if not targets:
        raise RevisionEvaluatorError("Evaluator tokenization produced no text tokens")
    if len(context) + len(targets) > int(context_limit):
        raise RevisionEvaluatorError(
            "Evaluator prompt plus text exceeds context limit ({} + {} > {})".format(
                len(context), len(targets), context_limit
            )
        )
    started = time.perf_counter()
    evaluate_context(model, context)
    token_logps: List[float] = []
    for token_id in targets:
        logits = get_last_logits(model)
        if token_id < 0 or token_id >= logits.size:
            raise RevisionEvaluatorError("Evaluator tokenizer returned out-of-vocabulary ID")
        token_logps.append(token_log_probability(logits, token_id))
        model.eval([token_id])
    elapsed = time.perf_counter() - started
    total_logp = float(np.sum(np.asarray(token_logps, dtype=np.float64)))
    count = len(targets)
    return {
        "evaluator_token_count": count,
        "context_token_count": len(context),
        "total_log_probability": total_logp,
        "mean_log_probability": total_logp / count,
        "total_nll": -total_logp,
        "mean_nll": -total_logp / count,
        "scoring_seconds": elapsed,
        "evaluator_tokens_per_second": count / elapsed if elapsed > 0 else None,
        "tokenization_and_scoring_mode": "serial_llama_cpp_recorded_prompt_v1",
    }


def aggregate_segment_scores(
    segment_scores: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    if not segment_scores:
        raise RevisionEvaluatorError("Cannot aggregate zero segment scores")
    token_count = sum(int(row["evaluator_token_count"]) for row in segment_scores)
    if token_count <= 0:
        raise RevisionEvaluatorError("Segment aggregation has zero evaluated tokens")
    total_logp = sum(float(row["total_log_probability"]) for row in segment_scores)
    seconds = sum(float(row["scoring_seconds"]) for row in segment_scores)
    return {
        "evaluator_token_count": token_count,
        "heldout_evaluator_total_log_probability": total_logp,
        "heldout_evaluator_mean_log_probability": total_logp / token_count,
        "heldout_evaluator_log_probability": total_logp / token_count,
        "heldout_evaluator_total_nll": -total_logp,
        "heldout_evaluator_mean_nll": -total_logp / token_count,
        "scoring_seconds": seconds,
        "evaluator_tokens_per_second": token_count / seconds if seconds > 0 else None,
        "segment_aggregation": "sum_log_probability_then_divide_by_total_evaluator_tokens",
    }


def evaluate_task(
    model: Any,
    task: Mapping[str, object],
    evaluator_model_manifest: Mapping[str, object],
    input_manifest_sha256: str,
    config_manifest_sha256: str,
    context_limit: int = 4096,
) -> Dict[str, object]:
    evaluator_id = str(task["evaluator_model_id"])
    generator_id = str(task["generator_model_id"])
    if evaluator_for_generator(generator_id) != evaluator_id or evaluator_id == generator_id:
        raise RevisionEvaluatorError("Task violates held-out model mapping")
    model_identity = getattr(model, "rankcloak_revision_model_id", evaluator_id)
    if str(model_identity) != evaluator_id:
        raise RevisionEvaluatorError("Loaded evaluator identity does not match task")
    started = time.perf_counter()
    scores: List[Dict[str, object]] = []
    for expected_index, segment in enumerate(task["segments"]):
        if int(segment["segment_index"]) != expected_index:
            raise ArtifactIntegrityError("Evaluation task segment order changed")
        prompt_text = str(segment["prompt_text"])
        text = str(segment["text"])
        if hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() != segment["prompt_text_sha256"]:
            raise ArtifactIntegrityError("Evaluation prompt hash mismatch")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != segment["text_sha256"]:
            raise ArtifactIntegrityError("Evaluation text hash mismatch")
        score = score_conditioned_text(
            model, prompt_text, text, context_limit=context_limit
        )
        scores.append(
            {
                "segment_index": expected_index,
                "prompt_id": segment.get("prompt_id"),
                "prompt_category": segment.get("prompt_category"),
                "prompt_text_sha256": segment["prompt_text_sha256"],
                "text_sha256": segment["text_sha256"],
                **score,
            }
        )
    aggregate = aggregate_segment_scores(scores)
    configured = evaluator_model_manifest["configured_model"]
    verification = evaluator_model_manifest["verification"]
    return _json_safe(
        {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "record_type": "heldout_evaluator_feature",
            "evaluation_id": task["evaluation_id"],
            "row_id": task["evaluation_id"],
            "trial_id": task["trial_id"],
            "source_trial_id_raw": task.get("source_trial_id_raw"),
            "source_stage": task["source_stage"],
            "evidence_partition": task["evidence_partition"],
            "confirmatory_pooling_eligible": task["confirmatory_pooling_eligible"],
            "source_work_id": task["source_work_id"],
            "source_record_type": task["source_record_type"],
            "source_record_sha256": task["source_record_sha256"],
            "source_evidence_status": task["source_evidence_status"],
            "evidence_status": task["evidence_status"],
            "study_phase": task.get("study_phase"),
            "protocol_contract_revision": task["protocol_contract_revision"],
            "result_schema_revision": task["result_schema_revision"],
            "source_protocol_contract_revision": task[
                "source_protocol_contract_revision"
            ],
            "source_result_schema_revision": task[
                "source_result_schema_revision"
            ],
            "payload_name": task.get("payload_name"),
            "payload_class": task.get("payload_class"),
            "payload_split": task.get("payload_split"),
            # model_id remains the generator for downstream condition grouping.
            "model_id": generator_id,
            "generator_model_id": generator_id,
            "evaluator_model_id": evaluator_id,
            "same_model_evaluation": False,
            "protocol_variant": task["protocol_variant"],
            "prompt_id": task.get("prompt_id"),
            "prompt_category": task.get("prompt_category"),
            "prompt_text": task["prompt_text"],
            "prompt_count": task["prompt_count"],
            "prompt_conditioning": task["prompt_conditioning"],
            "language": task.get("language"),
            "text_view": task["text_view"],
            "text": task["text"],
            "text_sha256": task["text_sha256"],
            "segment_count": task["segment_count"],
            "segment_scores": scores,
            **aggregate,
            "wall_seconds": time.perf_counter() - started,
            "evaluator_artifact_sha256": configured.get("artifact_sha256"),
            "evaluator_artifact_actual_sha256": verification.get("actual_sha256"),
            "generator_artifact_sha256": task.get("generator_artifact_sha256"),
            "input_results_manifest_sha256": input_manifest_sha256,
            "config_manifest_sha256": config_manifest_sha256,
            "human_rating_substitute": False,
            "quality_estimand": "mean_evaluator_token_log_probability_conditioned_on_recorded_overt_prompt",
        }
    )  # type: ignore[return-value]


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _llama_cpp_manifest() -> Dict[str, object]:
    preload_pip_cuda_libraries()
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as api

        raw = api.llama_print_system_info()
        return {
            "available": True,
            "python_package_version": getattr(llama_cpp, "__version__", None),
            "system_info": raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw),
            "gpu_offload_supported": bool(api.llama_supports_gpu_offload()),
        }
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, exc)}


def configure_deterministic_backend_environment(n_gpu_layers: int) -> None:
    if int(n_gpu_layers) == 0:
        return
    for name, value in {
        "CUDA_LAUNCH_BLOCKING": "1",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_CUDA_DISABLE_FUSION": "1",
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ.setdefault(name, value)


def query_gpu_inventory() -> List[Dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows: List[Dict[str, object]] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append(
                {
                    "uuid": values[0], "name": values[1], "pci_bus_id": values[2],
                    "memory_total_mib": int(values[3]), "driver_version": values[4],
                }
            )
    return rows


def pin_gpu_by_uuid(gpu_uuid: str) -> Dict[str, object]:
    matches = [row for row in query_gpu_inventory() if row["uuid"] == str(gpu_uuid)]
    if len(matches) != 1:
        raise RevisionEvaluatorError(
            "GPU UUID {!r} did not resolve exactly once".format(gpu_uuid)
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    return matches[0]


def _source_manifest(project_root: Path) -> Dict[str, object]:
    relative_paths = (
        "rankcloak/revision_evaluator.py",
        "rankcloak/revision_artifacts.py",
        "rankcloak/revision_config.py",
        "rankcloak/model_io.py",
        "rankcloak/rank_codec.py",
        "scripts/run_revision_evaluator.py",
        "pyproject.toml",
    )
    files: List[Dict[str, object]] = []
    for relative in relative_paths:
        path = Path(project_root) / relative
        if not path.is_file() or path.is_symlink():
            raise ArtifactIntegrityError("Evaluator source missing: {}".format(relative))
        files.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_type": "heldout_evaluator_source",
        "files": files,
        "files_sha256": canonical_json_sha256(files),
        "cyclic_mapping": EVALUATOR_BY_GENERATOR,
    }


def _runtime_manifest() -> Dict[str, object]:
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("rankcloak", "llama-cpp-python", "numpy")
        },
        "llama_cpp_backend": _llama_cpp_manifest(),
        "deterministic_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
                "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION",
                "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "scoring_semantics": "serial_llama_cpp_recorded_prompt_v1",
    }


def build_input_results_manifest(
    manifests: Sequence[Mapping[str, object]],
    evaluator_model_id: str,
    generator_model_id: str,
) -> Dict[str, object]:
    value: Dict[str, object] = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_type": "heldout_evaluator_inputs",
        "generator_model_id": generator_model_id,
        "evaluator_model_id": evaluator_model_id,
        "same_model_evaluation": False,
        "generator_artifact_opened_by_evaluator": False,
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "runner_shards": list(manifests),
    }
    value["inputs_sha256"] = canonical_json_sha256(value)
    return value


def prepare_evaluator_run(
    plan: Sequence[Mapping[str, object]],
    input_manifest: Mapping[str, object],
    evaluator_model_manifest: Mapping[str, object],
    config_verification: Mapping[str, object],
    evaluator_model_id: str,
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
    context_limit: int = 4096,
    gpu_uuid: Optional[str] = None,
    n_gpu_layers: int = -1,
    resume: bool = False,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists() and not resume:
        raise RevisionEvaluatorError(
            "Output already has a checkpoint; pass --resume to verify and continue"
        )
    if any(
        row.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        for row in plan
    ):
        raise RevisionEvaluatorError(
            "Evaluator plan lacks the payload-fidelity-v2 result contract"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _source_manifest(Path(project_root))
    runtime_manifest = _runtime_manifest()
    hardware_manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "selected_gpu_uuid": gpu_uuid,
        "gpu_inventory": query_gpu_inventory(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    write_immutable_jsonl(output_dir / "plan.jsonl", plan)
    write_immutable_json(output_dir / "input_results_manifest.json", input_manifest)
    write_immutable_json(output_dir / "evaluator_model_manifest.json", evaluator_model_manifest)
    write_immutable_json(output_dir / "source_manifest.json", source_manifest)
    write_immutable_json(output_dir / "runtime_manifest.json", runtime_manifest)
    write_immutable_json(output_dir / "hardware_manifest.json", hardware_manifest)
    ids = [str(row["evaluation_id"]) for row in plan]
    partitions = sorted({str(row.get("evidence_partition")) for row in plan})
    if not partitions:
        raise RevisionEvaluatorError("Evaluator run has no evidence partition")
    if (
        "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling" in partitions
        and len(partitions) != 1
    ):
        raise RevisionEvaluatorError(
            "Smoke-v3 evidence cannot share an evaluator run with another partition"
        )
    partition_label = "+".join(partitions)
    identity = build_run_identity_manifest(
        study_id="revision_v1/heldout_evaluator/{}/{}".format(
            partition_label, evaluator_model_id
        ),
        config_manifest_sha256=str(config_verification["sha256"]),
        payload_manifest_sha256=file_sha256(output_dir / "input_results_manifest.json"),
        planned_trial_ids=ids,
        model_artifacts=[evaluator_model_manifest],
        command_line_args=[
            "evaluator_model_id={}".format(evaluator_model_id),
            "generator_model_id={}".format(generator_for_evaluator(evaluator_model_id)),
            "context_limit={}".format(int(context_limit)),
            "gpu_uuid={}".format(gpu_uuid or "cpu"),
            "n_gpu_layers={}".format(int(n_gpu_layers)),
            "evidence_partitions={}".format(partition_label),
            "source_stages={}".format(
                ",".join(sorted({str(row["source_stage"]) for row in plan}))
            ),
            "protocol_contract_revision={}".format(
                PROTOCOL_CONTRACT_REVISION
            ),
            "result_schema_revision={}".format(RESULT_SCHEMA_REVISION),
            "source_manifest_sha256={}".format(file_sha256(output_dir / "source_manifest.json")),
            "runtime_manifest_sha256={}".format(file_sha256(output_dir / "runtime_manifest.json")),
            "hardware_manifest_sha256={}".format(file_sha256(output_dir / "hardware_manifest.json")),
        ],
    )
    identity["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
    identity["result_schema_revision"] = RESULT_SCHEMA_REVISION
    identity.pop("run_identity_sha256", None)
    identity["run_identity_sha256"] = canonical_json_sha256(identity)
    write_immutable_json(output_dir / "run_identity.json", identity)
    checkpoint = initialize_checkpoint(
        checkpoint_path,
        study_id=str(identity["study_id"]),
        config_manifest_sha256=str(config_verification["sha256"]),
        planned_trial_ids=ids,
    )
    return {"run_identity": identity, "checkpoint": checkpoint}


def reconcile_checkpoint(
    checkpoint_path: Path,
    planned_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    planned = set(map(str, planned_ids))
    completions: Dict[str, Mapping[str, object]] = {}
    failures: Dict[str, Mapping[str, object]] = {}
    attempts: Dict[str, int] = {}
    seen_attempts = set()
    for row in records:
        evaluation_id = str(row.get("evaluation_id"))
        if evaluation_id not in planned:
            raise ArtifactIntegrityError("Evaluator result has an unplanned ID")
        attempt = int(row.get("attempt_index", 1))
        key = (evaluation_id, attempt)
        if key in seen_attempts:
            raise ArtifactIntegrityError("Duplicate evaluator durable attempt")
        seen_attempts.add(key)
        attempts[evaluation_id] = max(attempts.get(evaluation_id, 0), attempt)
        status = row.get("execution_status")
        if status == "completed":
            if evaluation_id in completions:
                raise ArtifactIntegrityError("Multiple evaluator completions for one task")
            completions[evaluation_id] = row
        elif status == "failed":
            failures[evaluation_id] = row
        else:
            raise ArtifactIntegrityError("Evaluator record has invalid execution status")
    checkpoint_completed = set(map(str, checkpoint.get("completed_trial_ids", [])))
    if checkpoint_completed - set(completions):
        raise ArtifactIntegrityError("Checkpoint completion lacks a durable evaluator row")
    checkpoint["completed_trial_ids"] = sorted(completions)
    current_failures = set(failures) - set(completions)
    checkpoint["failed_trial_ids"] = sorted(current_failures)
    checkpoint["failure_details"] = {
        key: failures[key].get("error", {}) for key in sorted(current_failures)
    }
    checkpoint["attempt_counts"] = dict(sorted(attempts.items()))
    checkpoint["updated_at"] = utc_now()
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def export_feature_table(output_dir: Path) -> Dict[str, object]:
    # Atomically materialize flat feature and continuous-quality tables.
    records_path = Path(output_dir) / "records.jsonl"
    records = load_jsonl(records_path) if records_path.exists() else []
    features = [
        {
            key: value
            for key, value in row.items()
            if key not in {"attempt_index", "completed_at", "segment_scores"}
        }
        for row in records
        if row.get("execution_status") == "completed"
        and row.get("record_type") == "heldout_evaluator_feature"
    ]
    if any(
        row.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        for row in features
    ):
        raise ArtifactIntegrityError(
            "Evaluator feature table contains a legacy or missing result contract"
        )
    evidence_statuses = sorted({str(row.get("evidence_status")) for row in features})
    smoke_only = bool(features) and evidence_statuses == [EVIDENCE_SMOKE_V3]
    if EVIDENCE_SMOKE_V3 in evidence_statuses and not smoke_only:
        raise ArtifactIntegrityError(
            "Exploratory smoke evaluator rows cannot share an analysis table with other evidence"
        )
    if smoke_only and any(
        row.get("confirmatory_pooling_eligible") is not False for row in features
    ):
        raise ArtifactIntegrityError(
            "Smoke evaluator row is missing its no-confirmatory-pooling marker"
        )
    feature_path = Path(output_dir) / "features.jsonl"
    _atomic_write_jsonl(feature_path, features)
    continuous = []
    for row in features:
        continuous.append(
            {
                key: value
                for key, value in {
                    **row,
                    "source_trial_id": row.get("trial_id"),
                    "trial_id": row.get("evaluation_id"),
                    "analysis_unit": "source_payload_trial_or_control_view",
                }.items()
                if key not in {"text", "prompt_text"}
            }
        )
    continuous_path = Path(output_dir) / "continuous_quality.jsonl"
    _atomic_write_jsonl(continuous_path, continuous)
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "manifest_type": "heldout_evaluator_feature_table",
        "row_count": len(features),
        "path": str(feature_path.resolve()),
        "sha256": file_sha256(feature_path),
        "continuous_quality_path": str(continuous_path.resolve()),
        "continuous_quality_sha256": file_sha256(continuous_path),
        "analysis_unit": "source_payload_trial_or_control_view",
        "nested_segments_are_not_independent": True,
        "evidence_statuses": evidence_statuses,
        "evidence_partition": (
            "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling"
            if smoke_only
            else "non_smoke_evaluator_outputs"
        ),
        "confirmatory_pooling_eligible": not smoke_only,
        "statistics_ingestion": (
            {
                "policy": "exploratory_only_do_not_pool_with_confirmatory",
                "features": "--features features.jsonl",
                "continuous_quality": "--trials continuous_quality.jsonl",
            }
            if smoke_only
            else {
                "features": "--features features.jsonl",
                "continuous_quality": "--trials continuous_quality.jsonl",
            }
        ),
    }
    write_immutable_json(Path(output_dir) / "features_manifest.json", manifest)
    return manifest


def export_auxiliary_timing_summary(
    output_dir: Path,
    plan: Sequence[Mapping[str, object]],
    evaluator_model_id: str,
    gpu_count: int,
) -> Dict[str, object]:
    # Emit the exact self-hashed auxiliary schema consumed by revision_compute.
    if not plan or any(
        row.get("source_stage") != SMOKE_SOURCE_STAGE
        or row.get("evidence_status") != EVIDENCE_SMOKE_V3
        or row.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        or row.get("confirmatory_pooling_eligible") is not False
        for row in plan
    ):
        raise RevisionEvaluatorError(
            "Auxiliary timing may be exported only for an isolated complete smoke-v3 plan"
        )
    if isinstance(gpu_count, bool) or int(gpu_count) != 1:
        raise RevisionEvaluatorError(
            "Smoke-v3 evaluator timing must charge exactly one pinned GPU"
        )
    records_path = Path(output_dir) / "records.jsonl"
    records = load_jsonl(records_path)
    completed: Dict[str, Mapping[str, object]] = {}
    planned_ids = [str(row["evaluation_id"]) for row in plan]
    for row in records:
        if (
            row.get("execution_status") == "completed"
            and row.get("record_type") == "heldout_evaluator_feature"
        ):
            evaluation_id = str(row.get("evaluation_id"))
            if evaluation_id in completed:
                raise ArtifactIntegrityError(
                    "Auxiliary timing found duplicate evaluator completion"
                )
            completed[evaluation_id] = row
    if set(completed) != set(planned_ids):
        raise ArtifactIntegrityError(
            "Auxiliary timing requires one completion for every smoke evaluation"
        )
    ordered = [completed[evaluation_id] for evaluation_id in planned_ids]
    if any(
        row.get("evidence_status") != EVIDENCE_SMOKE_V3
        or row.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        or row.get("confirmatory_pooling_eligible") is not False
        or row.get("evaluator_model_id") != evaluator_model_id
        for row in ordered
    ):
        raise ArtifactIntegrityError(
            "Auxiliary timing completion violates smoke evidence or evaluator identity"
        )
    generator_ids = sorted({str(row.get("generator_model_id")) for row in ordered})
    if len(generator_ids) != 1:
        raise ArtifactIntegrityError("Auxiliary timing spans multiple generator models")
    wall_values = [float(row.get("wall_seconds")) for row in ordered]
    scoring_values = [float(row.get("scoring_seconds")) for row in ordered]
    if (
        any(not np.isfinite(value) or value < 0 for value in wall_values + scoring_values)
        or sum(wall_values) <= 0
    ):
        raise ArtifactIntegrityError("Auxiliary timing has invalid measured durations")
    elapsed_seconds = float(sum(wall_values))
    scoring_seconds = float(sum(scoring_values))
    total_tokens = sum(int(row.get("evaluator_token_count")) for row in ordered)
    event_rows = load_jsonl(Path(output_dir) / "events.jsonl")
    load_events = [
        row for row in event_rows
        if row.get("event") == "evaluator_model_loaded"
    ]
    if len(load_events) != 1:
        raise ArtifactIntegrityError(
            "Smoke-v3 evaluator timing requires one durable model-load event"
        )
    load_event = load_events[0]
    model_load_seconds = float(load_event.get("model_load_seconds"))
    if (
        not np.isfinite(model_load_seconds)
        or model_load_seconds <= 0
        or load_event.get("evaluator_model_id") != evaluator_model_id
        or not str(load_event.get("gpu_uuid", "")).startswith("GPU-")
    ):
        raise ArtifactIntegrityError(
            "Smoke-v3 evaluator model-load timing or GPU binding is invalid"
        )
    from .revision_compute import build_auxiliary_timing_record

    timing = build_auxiliary_timing_record(
        component="evaluator",
        component_id="heldout_evaluator_smoke_v3_{}_from_{}".format(
            evaluator_model_id, generator_ids[0]
        ),
        completed_units=len(ordered),
        elapsed_seconds=elapsed_seconds,
        gpu_count=int(gpu_count),
        model_id=evaluator_model_id,
        model_load_seconds=model_load_seconds,
    )
    timing.pop("timing_manifest_sha256", None)
    timing.update(
        {
            "generator_model_id": generator_ids[0],
            "source_stage": SMOKE_SOURCE_STAGE,
            "evidence_partition": "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling",
            "confirmatory_pooling_eligible": False,
            "unit_definition": "one_saved_source_payload_trial_or_control_view",
            "elapsed_seconds_definition": "sum_of_durable_per_task_wall_seconds_excluding_model_load",
            "summed_scoring_seconds": scoring_seconds,
            "total_evaluator_tokens": total_tokens,
            "evaluator_tokens_per_second": (
                total_tokens / elapsed_seconds if elapsed_seconds > 0 else None
            ),
            "ordered_evaluation_ids_sha256": trial_ids_sha256(planned_ids),
            "scoring_plan_content_sha256": canonical_json_sha256(list(plan)),
            "records_jsonl_sha256": file_sha256(records_path),
            "input_results_manifest_sha256": ordered[0].get(
                "input_results_manifest_sha256"
            ),
            "config_manifest_sha256": ordered[0].get("config_manifest_sha256"),
        }
    )
    timing["timing_manifest_sha256"] = canonical_json_sha256(timing)
    path = Path(output_dir) / "auxiliary_timing.json"
    write_immutable_json(path, timing)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "record": timing,
        "revision_compute_argument": "--auxiliary-timing {}".format(path),
    }


def run_evaluation_plan(
    model: Any,
    plan: Sequence[Mapping[str, object]],
    evaluator_model_manifest: Mapping[str, object],
    input_manifest_sha256: str,
    config_manifest_sha256: str,
    output_dir: Path,
    context_limit: int = 4096,
    max_pending: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    planned_ids = [str(row["evaluation_id"]) for row in plan]
    existing = load_jsonl(records_path) if records_path.exists() else []
    checkpoint = reconcile_checkpoint(checkpoint_path, planned_ids, existing)
    pending = pending_trial_ids(planned_ids, checkpoint)
    if max_pending is not None:
        pending = pending[: max(0, int(max_pending))]
    tasks = {str(row["evaluation_id"]): row for row in plan}
    consecutive_errors = 0
    for evaluation_id in pending:
        checkpoint = load_checkpoint(checkpoint_path)
        attempt = int(checkpoint["attempt_counts"].get(evaluation_id, 0)) + 1
        task = tasks[evaluation_id]
        try:
            row = evaluate_task(
                model, task, evaluator_model_manifest,
                input_manifest_sha256=input_manifest_sha256,
                config_manifest_sha256=config_manifest_sha256,
                context_limit=context_limit,
            )
            row.update(
                {
                    "execution_status": "completed",
                    "attempt_index": attempt,
                    "completed_at": utc_now(),
                }
            )
            _append_jsonl_fsync(records_path, row)
            record_checkpoint_result(checkpoint_path, evaluation_id, "completed")
            consecutive_errors = 0
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure = {
                "schema_version": EVALUATOR_SCHEMA_VERSION,
                "record_type": "heldout_evaluator_failure",
                "evaluation_id": evaluation_id,
                "source_work_id": task.get("source_work_id"),
                "generator_model_id": task.get("generator_model_id"),
                "evaluator_model_id": task.get("evaluator_model_id"),
                "evidence_status": task.get("evidence_status"),
                "evidence_partition": task.get("evidence_partition"),
                "study_phase": task.get("study_phase"),
                "protocol_contract_revision": task.get(
                    "protocol_contract_revision"
                ),
                "result_schema_revision": task.get("result_schema_revision"),
                "confirmatory_pooling_eligible": task.get(
                    "confirmatory_pooling_eligible"
                ),
                "execution_status": "failed",
                "attempt_index": attempt,
                "failed_at": utc_now(),
                "error": error,
            }
            _append_jsonl_fsync(records_path, failure)
            record_checkpoint_result(checkpoint_path, evaluation_id, "failed", error)
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise RevisionEvaluatorError(
                    "Stopped after three consecutive evaluator failures"
                ) from exc
    final = load_checkpoint(checkpoint_path)
    completed_count = len(final["completed_trial_ids"])
    feature_manifest = None
    if completed_count == len(plan) and not final["failed_trial_ids"]:
        feature_manifest = export_feature_table(output_dir)
    return {
        "planned": len(plan),
        "completed": completed_count,
        "failed_current": len(final["failed_trial_ids"]),
        "remaining": len(plan) - completed_count,
        "records_path": str(records_path),
        "features_manifest": feature_manifest,
    }


def load_one_evaluator_model(
    configs: Mapping[str, Mapping[str, object]],
    evaluator_model_id: str,
    project_root: Path = PROJECT_ROOT,
    context_limit: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Any, float]:
    entry = _model_entry(configs, evaluator_model_id)
    started = time.perf_counter()
    model = load_llama_cpp_model(
        model_path=Path(project_root) / str(entry["relative_path"]),
        n_ctx=int(context_limit), n_threads=n_threads,
        n_gpu_layers=int(n_gpu_layers), logits_all=True, verbose=verbose,
    )
    try:
        setattr(model, "rankcloak_revision_model_id", evaluator_model_id)
    except Exception:
        pass
    return model, time.perf_counter() - started


def _default_output_dir(
    evaluator_model_id: str, stages: Sequence[str], limited: bool
) -> Path:
    if stages == [SMOKE_SOURCE_STAGE] or tuple(stages) == (SMOKE_SOURCE_STAGE,):
        label = SMOKE_SOURCE_STAGE
    else:
        label = "limited" if limited else "_".join(stages)
    return PROJECT_ROOT / "results" / "revision_v1" / "heldout_evaluator" / label / evaluator_model_id


def select_source_stages(requested: Optional[Sequence[str]]) -> List[str]:
    requested_values = list(requested) if requested is not None else ["primary_v2"]
    unknown = set(requested_values) - set(SOURCE_STAGES)
    if unknown:
        raise RevisionEvaluatorError(
            "Unsupported evaluator source stage: {}".format(
                ", ".join(sorted(unknown))
            )
        )
    stages = [stage for stage in SOURCE_STAGES if stage in set(requested_values)]
    if not stages:
        raise RevisionEvaluatorError("Evaluator source-stage selection is empty")
    return stages



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-model", required=True, choices=tuple(GENERATOR_BY_EVALUATOR))
    parser.add_argument(
        "--source-stage", action="append", choices=SOURCE_STAGES,
        help="Completed runner stage to score; repeat as needed (default: primary_v2).",
    )
    parser.add_argument("--source-results-root", type=Path, default=PROJECT_ROOT / "results" / "revision_v1")
    parser.add_argument("--limit", type=int, help="Exploratory-only content-addressed subset.")
    parser.add_argument("--max-pending", type=int, help="Operational checkpoint chunk; does not alter the frozen plan.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--context", dest="context_limit", type=int, default=4096)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_REVISION_CONFIG_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--verbose-model", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise RevisionEvaluatorError("--limit must be positive")
    if args.max_pending is not None and args.max_pending <= 0:
        raise RevisionEvaluatorError("--max-pending must be positive")
    if args.context_limit <= 0:
        raise RevisionEvaluatorError("--context must be positive")
    stages = select_source_stages(args.source_stage)
    if SMOKE_SOURCE_STAGE in stages and len(stages) != 1:
        raise RevisionEvaluatorError(
            "--source-stage smoke_v3 cannot be combined with another source stage"
        )
    if SMOKE_SOURCE_STAGE in stages and args.limit is not None:
        raise RevisionEvaluatorError(
            "--limit is forbidden for the frozen exploratory smoke-v3 plan"
        )
    evaluator_id = str(args.evaluator_model)
    generator_id = generator_for_evaluator(evaluator_id)
    configs = load_revision_config_set(args.config_dir)
    config_verification = verify_config_manifest(args.config_dir)
    prompt_lookup = build_prompt_lookup(configs)

    stage_records: Dict[str, Sequence[Mapping[str, object]]] = {}
    shard_manifests: List[Mapping[str, object]] = []
    for stage in stages:
        run_dir = Path(args.source_results_root) / stage / generator_id
        records, manifest = verify_completed_runner_shard(
            run_dir, stage, generator_id, str(config_verification["sha256"])
        )
        stage_records[stage] = records
        shard_manifests.append(manifest)
    generator_hashes = {
        str(manifest.get("generator_artifact_sha256"))
        for manifest in shard_manifests
    }
    if len(generator_hashes) != 1 or "None" in generator_hashes:
        raise ArtifactIntegrityError(
            "Runner shards disagree on the pinned generator artifact"
        )
    full_plan = build_scoring_plan(
        stage_records, evaluator_id, generator_id,
        generator_artifact_sha256=next(iter(generator_hashes)),
        prompt_lookup=prompt_lookup,
    )
    plan = relabel_limited_plan(full_plan, args.limit) if args.limit is not None else full_plan
    input_manifest = build_input_results_manifest(shard_manifests, evaluator_id, generator_id)
    evaluator_manifest = verify_evaluator_model(configs, evaluator_id, args.project_root)
    summary = scoring_plan_summary(plan)
    summary.update(
        {
            "dry_run": bool(args.dry_run),
            "full_unlimited_evaluation_tasks": len(full_plan),
            "source_stages": stages,
            "generator_model_id": generator_id,
            "evaluator_model_id": evaluator_id,
            "same_model_evaluation": False,
            "generator_artifact_opened": False,
        }
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.n_gpu_layers != 0 and not args.gpu_uuid:
        raise RevisionEvaluatorError("GPU execution requires an exact --gpu-uuid")
    if args.gpu_uuid:
        pin_gpu_by_uuid(args.gpu_uuid)
    configure_deterministic_backend_environment(args.n_gpu_layers)
    output_dir = args.output_dir or _default_output_dir(evaluator_id, stages, args.limit is not None)
    preparation = prepare_evaluator_run(
        plan, input_manifest, evaluator_manifest, config_verification,
        evaluator_id, output_dir, project_root=args.project_root,
        context_limit=args.context_limit, gpu_uuid=args.gpu_uuid,
        n_gpu_layers=args.n_gpu_layers, resume=args.resume,
    )
    checkpoint = preparation["checkpoint"]
    ids = [str(row["evaluation_id"]) for row in plan]
    if not pending_trial_ids(ids, checkpoint):
        feature_manifest = export_feature_table(output_dir)
        auxiliary_timing = (
            export_auxiliary_timing_summary(
                output_dir, plan, evaluator_id,
                gpu_count=1 if args.n_gpu_layers != 0 else 0,
            )
            if stages == [SMOKE_SOURCE_STAGE]
            else None
        )
        summary.update(
            {
                "output_dir": str(output_dir),
                "execution": "already_complete",
                "features_manifest": feature_manifest,
                "auxiliary_timing": auxiliary_timing,
            }
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    model = None
    try:
        model, load_seconds = load_one_evaluator_model(
            configs, evaluator_id, project_root=args.project_root,
            context_limit=args.context_limit, n_gpu_layers=args.n_gpu_layers,
            n_threads=args.threads, verbose=args.verbose_model,
        )
        _append_jsonl_fsync(
            output_dir / "events.jsonl",
            {"event": "evaluator_model_loaded", "at": utc_now(), "evaluator_model_id": evaluator_id,
             "generator_model_id": generator_id, "model_load_seconds": load_seconds,
             "gpu_uuid": args.gpu_uuid, "n_gpu_layers": args.n_gpu_layers},
        )
        execution = run_evaluation_plan(
            model, plan, evaluator_manifest,
            input_manifest_sha256=file_sha256(output_dir / "input_results_manifest.json"),
            config_manifest_sha256=str(config_verification["sha256"]),
            output_dir=output_dir, context_limit=args.context_limit,
            max_pending=args.max_pending,
        )
        _append_jsonl_fsync(
            output_dir / "events.jsonl",
            {"event": "evaluator_session_finished", "at": utc_now(), **execution},
        )
        auxiliary_timing = None
        if (
            stages == [SMOKE_SOURCE_STAGE]
            and execution["completed"] == execution["planned"]
            and execution["failed_current"] == 0
        ):
            auxiliary_timing = export_auxiliary_timing_summary(
                output_dir, plan, evaluator_id,
                gpu_count=1 if args.n_gpu_layers != 0 else 0,
            )
        summary.update(
            {
                "output_dir": str(output_dir),
                "model_load_seconds": load_seconds,
                "execution": execution,
                "auxiliary_timing": auxiliary_timing,
            }
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if execution["failed_current"] == 0 else 2
    finally:
        close = getattr(model, "close", None) if model is not None else None
        if callable(close):
            close()


__all__ = [
    "EVALUATOR_BY_GENERATOR", "EVIDENCE_LIMITED", "GENERATOR_BY_EVALUATOR",
    "RevisionEvaluatorError", "aggregate_segment_scores", "build_argument_parser",
    "build_input_results_manifest", "build_prompt_lookup", "build_scoring_plan",
    "evaluate_task",
    "evaluator_for_generator", "export_auxiliary_timing_summary",
    "select_source_stages",
    "export_feature_table", "generator_for_evaluator",
    "main", "prepare_evaluator_run", "reconcile_checkpoint", "relabel_limited_plan",
    "run_evaluation_plan", "score_conditioned_text", "scoring_plan_summary",
    "verify_completed_runner_shard", "verify_config_manifest",
    "EVIDENCE_SMOKE_V3", "SMOKE_SOURCE_STAGE",
]
