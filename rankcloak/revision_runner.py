"""Model-backed execution for the frozen Scientific Reports revision-v1 study.

The module deliberately separates immutable planning from mutable execution.
Every real invocation loads one and only one content-pinned GGUF model.  Smoke
and limited diagnostic runs carry evidence labels distinct from confirmatory
outputs and therefore cannot be pooled accidentally by downstream analyses.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .model_io import (
    evaluate_context,
    get_bos_token_id,
    get_last_logits,
    load_llama_cpp_model,
    make_context_token_ids,
    preload_pip_cuda_libraries,
    safe_detokenize,
)
from .revision_artifacts import (
    ArtifactIntegrityError,
    build_run_identity_manifest,
    canonical_json_bytes,
    file_sha256,
    initialize_checkpoint,
    load_checkpoint,
    pending_trial_ids,
    record_checkpoint_result,
    save_checkpoint,
    verify_directory_manifest,
    write_immutable_json,
    write_immutable_jsonl,
)
from .revision_config import (
    DEFAULT_REVISION_CONFIG_DIR,
    RevisionConfigError,
    build_ablation_trial_plan,
    build_multilingual_trial_plan,
    build_primary_control_plan,
    build_primary_trial_plan,
    flatten_prompt_templates,
    load_revision_config_set,
    model_ids,
    prompt_by_id,
    verify_model_artifact_pins,
)
from .revision_payloads import (
    REVISION_CORPUS_ID,
    REVISION_CORPUS_SHA256,
    RevisionPayload,
    generate_revision_v1_payloads,
    revision_payload_records,
    validate_revision_corpus,
)
from .revision_protocol import (
    TAIL_NONE,
    Representation,
    apply_transmission_transform,
    bounded_representation,
    build_revision_filter_mask,
    decode_representation,
    diagnose_rank_failure,
    direct_representation,
    first_divergence,
    generate_rank_span,
    regenerate_greedy_leadin,
    recover_rank_span,
    retokenize_message,
    text_to_token_ids,
    transform_token_ids,
)
from .rank_codec import token_log_probability
from .revision_v3_diagnostics import next_token_diagnostic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCHEMA_VERSION = "1.0"
PROTOCOL_CONTRACT_REVISION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
STAGES = (
    "smoke",
    "smoke_v2",
    "smoke_v3",
    "primary",
    "primary_v2",
    "ablation",
    "ablation_v2",
    "multilingual",
    "multilingual_v2",
    "robustness",
    "robustness_v2",
)
EVIDENCE_SMOKE = "exploratory_smoke_not_for_confirmatory_pooling"
EVIDENCE_LIMITED = "exploratory_limited_not_for_confirmatory_pooling"
EVIDENCE_PRIMARY = "confirmatory_after_manifest_freeze"
EVIDENCE_ABLATION = "confirmatory_ablation_after_manifest_freeze"
EVIDENCE_MULTILINGUAL = "secondary_supplementary_after_manifest_freeze"
EVIDENCE_ROBUSTNESS = (
    "confirmatory_supporting_robustness_after_manifest_freeze"
)
EVIDENCE_SMOKE_V3 = (
    "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling"
)
EVIDENCE_PRIMARY_V2 = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_ABLATION_V2 = (
    "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_MULTILINGUAL_V2 = (
    "secondary_supplementary_multilingual_v2_payload_fidelity_after_manifest_freeze"
)
EVIDENCE_ROBUSTNESS_V2 = (
    "confirmatory_supporting_robustness_v2_payload_fidelity_after_manifest_freeze"
)
CONTROL_TEMPERATURE = 0.8
CONTROL_TOP_P = 0.95
_PERSISTED_FILTER_MASK_ARTIFACTS: set = set()


class RevisionRunnerError(RuntimeError):
    """Raised when execution would violate a frozen scientific contract."""


class ConditionUnavailable(RevisionRunnerError):
    """A frozen condition is scientifically undefined for the pinned tokenizer."""

    def __init__(self, details: Mapping[str, object]):
        self.details = dict(details)
        super().__init__(str(self.details.get("reason_code", "condition_unavailable")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def _cached_default_payloads() -> Tuple[RevisionPayload, ...]:
    return tuple(generate_revision_v1_payloads())


def _default_payloads() -> List[RevisionPayload]:
    return list(_cached_default_payloads())


def _stable_id(prefix: str, *parts: object) -> str:
    source = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    return "{}__{}".format(prefix, digest)


def _stage_evidence_status(stage: str) -> str:
    return {
        "smoke": EVIDENCE_SMOKE,
        "smoke_v2": EVIDENCE_SMOKE,
        "smoke_v3": EVIDENCE_SMOKE_V3,
        "primary": EVIDENCE_PRIMARY,
        "primary_v2": EVIDENCE_PRIMARY_V2,
        "ablation": EVIDENCE_ABLATION,
        "ablation_v2": EVIDENCE_ABLATION_V2,
        "multilingual": EVIDENCE_MULTILINGUAL,
        "multilingual_v2": EVIDENCE_MULTILINGUAL_V2,
        "robustness": EVIDENCE_ROBUSTNESS,
        "robustness_v2": EVIDENCE_ROBUSTNESS_V2,
    }[stage]


def _rankcloak_work(row: Mapping[str, object], evidence_status: str) -> Dict[str, object]:
    result = dict(row)
    result.update(
        {
            "work_id": str(row["trial_id"]),
            "work_kind": (
                "rankcloak" if bool(row.get("generation_required", True)) else "reference"
            ),
            "evidence_status": evidence_status,
            "replay_modes": (
                [
                    "saved_token_ids",
                    "detokenized_text_retokenized",
                    "greedy_leadin_regeneration",
                ]
                if evidence_status == EVIDENCE_SMOKE
                else ["saved_token_ids"]
            ),
        }
    )
    return result


def _control_work(
    control: Mapping[str, object],
    evidence_status: str,
    language: str = "en",
    prompt_text: Optional[str] = None,
) -> Dict[str, object]:
    result = dict(control)
    result.update(
        {
            "work_id": str(control["control_id"]),
            "work_kind": "control",
            "study_phase": "ordinary_llm_control",
            "evidence_status": evidence_status,
            "language": language,
            "generation_required": True,
        }
    )
    if prompt_text is not None:
        result["prompt_text"] = prompt_text
    return result


def _multilingual_control(row: Mapping[str, object]) -> Dict[str, object]:
    source_id = str(row["trial_id"])
    control_id = _stable_id("revision_v1_multilingual_control", source_id, "full_message")
    return {
        "control_id": control_id,
        "source_trial_id": source_id,
        "model_id": row["model_id"],
        "payload_name": row["payload_name"],
        "payload_class": row["payload_class"],
        "payload_split": row["payload_split"],
        "prompt_id": row["prompt_id"],
        "prompt_text": row["prompt_text"],
        "prompt_category": row["prompt_category"],
        "control_view": "full_message",
        "control_mode": "seeded_prompt_and_length_matched_sampling",
        "target_token_count": None,
        "sampling_seed": None,
        "temperature": CONTROL_TEMPERATURE,
        "top_p": CONTROL_TOP_P,
        "generation_required": True,
    }


def _build_primary_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    rankcloak_rows = build_primary_trial_plan(payloads=payloads, configs=configs)
    controls = build_primary_control_plan(rankcloak_rows, configs=configs)
    by_source: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for control in controls:
        by_source[str(control["source_trial_id"])].append(control)
    work: List[Dict[str, object]] = []
    for row in rankcloak_rows:
        work.append(_rankcloak_work(row, EVIDENCE_PRIMARY))
        ordered_controls = sorted(
            by_source[str(row["trial_id"])],
            key=lambda value: 0 if value["control_view"] == "full_message" else 1,
        )
        work.extend(_control_work(control, EVIDENCE_PRIMARY) for control in ordered_controls)
    return work


def _build_ablation_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    return [
        _rankcloak_work(row, EVIDENCE_ABLATION)
        for row in build_ablation_trial_plan(payloads=payloads, configs=configs)
    ]


def _build_multilingual_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    work: List[Dict[str, object]] = []
    for row in build_multilingual_trial_plan(payloads=payloads, configs=configs):
        work.append(_rankcloak_work(row, EVIDENCE_MULTILINGUAL))
        work.append(
            _control_work(
                _multilingual_control(row),
                EVIDENCE_MULTILINGUAL,
                language=str(row["language"]),
                prompt_text=str(row["prompt_text"]),
            )
        )
    return work


def _smoke_trial_id(model_id: str, position: int, original_id: str) -> str:
    return _stable_id("revision_v1_smoke", model_id, position, original_id)


def _smoke_control(row: Mapping[str, object], view: str) -> Dict[str, object]:
    control_id = _stable_id("revision_v1_smoke_control", row["trial_id"], view)
    return {
        "control_id": control_id,
        "source_trial_id": row["trial_id"],
        "model_id": row["model_id"],
        "payload_name": row["payload_name"],
        "payload_class": row["payload_class"],
        "payload_split": row["payload_split"],
        "prompt_id": row["prompt_id"],
        "prompt_category": row["prompt_category"],
        "control_view": view,
        "control_mode": "seeded_prompt_and_length_matched_sampling",
        "target_token_count": None,
        "sampling_seed": None,
        "temperature": CONTROL_TEMPERATURE,
        "top_p": CONTROL_TOP_P,
        "generation_required": True,
    }


def _select_smoke_rankcloak_rows(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    """Select twelve balanced and length-representative checks per model.

    Each model uses one template index across the six prompt categories, so the
    three shards collectively exercise all 18 English templates. Direct/ASCII
    paths deliberately include UUID, AEAD Base64, and the longest Ed25519
    payload; hex and segmented paths rotate all four eligible hex classes.
    """

    primary = build_primary_trial_plan(payloads=payloads, configs=configs)
    ablations = build_ablation_trial_plan(payloads=payloads, configs=configs)
    categories = list(configs["prompts"]["categories"])
    protocols = [
        "direct_subword_calgacus",
        "nonseg_ascii_b8",
        "nonseg_ascii_b16",
        "nonseg_hex_nibble_b16",
        "segmented_hex_single_topic",
        "segmented_hex_multi_topic",
    ]
    extra_conditions: Sequence[Tuple[str, object]] = (
        ("token_filter", "none"),
        ("token_filter", "roundtrip_stable_filter_v1"),
        ("leadin_tokens", 8),
        ("tail_policy", "none"),
        ("tail_policy", "sentence_tail_min20_max60"),
        ("segment_size_ranks", 16),
    )
    hex_classes = [
        "sha256_hex",
        "hmac_sha256_hex",
        "nonce_96_bit_hex",
        "token_128_bit_hex",
    ]
    ascii_long_classes = [
        "aes256_gcm_base64",
        "chacha20_poly1305_base64",
        "aes256_gcm_base64",
    ]
    selected: List[Dict[str, object]] = []
    known_models = model_ids(configs["models"])
    for model_index, model_id in enumerate(known_models):
        model_primary = [row for row in primary if row["model_id"] == model_id]
        model_ablation = [row for row in ablations if row["model_id"] == model_id]
        model_rows: List[Dict[str, object]] = []
        used_payloads: set = set()
        primary_classes = [
            "ed25519_signature_base64",
            "uuid_v4",
            ascii_long_classes[model_index],
            *[hex_classes[(model_index * 3 + offset) % 4] for offset in range(3)],
        ]
        for protocol, category, payload_class in zip(
            protocols, categories, primary_classes
        ):
            prompt_id = str(category["templates"][model_index]["prompt_id"])
            matches = [
                row
                for row in model_primary
                if row["protocol_variant"] == protocol
                and row["prompt_category"] == category["category_id"]
                and row["prompt_id"] == prompt_id
                and row["payload_class"] == payload_class
                and row["payload_name"] not in used_payloads
            ]
            if not matches:
                raise RevisionRunnerError(
                    "Could not balance smoke primary path {} / {} / {} / {}".format(
                        model_id, protocol, prompt_id, payload_class
                    )
                )
            chosen = dict(matches[0])
            used_payloads.add(chosen["payload_name"])
            model_rows.append(chosen)
        for extra_index, ((factor, level), category) in enumerate(
            zip(extra_conditions, categories)
        ):
            prompt_id = str(category["templates"][model_index]["prompt_id"])
            rotation = (model_index + extra_index) % len(hex_classes)
            class_preferences = hex_classes[rotation:] + hex_classes[:rotation]
            matches: List[Mapping[str, object]] = []
            for payload_class in class_preferences:
                matches = [
                    row
                    for row in model_ablation
                    if row["ablation_factor"] == factor
                    and row["ablation_level"] == level
                    and row["prompt_category"] == category["category_id"]
                    and row["prompt_id"] == prompt_id
                    and row["payload_class"] == payload_class
                    and row["payload_name"] not in used_payloads
                ]
                if matches:
                    break
            if not matches:
                raise RevisionRunnerError(
                    "Could not balance smoke ablation path {} / {}={} / {}".format(
                        model_id, factor, level, prompt_id
                    )
                )
            chosen = dict(matches[0])
            smoke_overrides: List[str] = []
            if factor == "token_filter" and level == "none":
                chosen["tail_policy"] = "fixed_tail40"
                smoke_overrides.append("tail_policy=fixed_tail40")
            if factor == "segment_size_ranks" and level == 16:
                chosen["leadin_tokens"] = 32
                smoke_overrides.append("leadin_tokens=32")
            chosen["smoke_overrides"] = smoke_overrides
            used_payloads.add(chosen["payload_name"])
            model_rows.append(chosen)
        if len(model_rows) != 12:
            raise RevisionRunnerError("Smoke shard must contain twelve RankCloak rows")
        model_classes = {str(row["payload_class"]) for row in model_rows}
        if "ed25519_signature_base64" not in model_classes:
            raise RevisionRunnerError("Every smoke model must include the longest Base64 class")
        if not any(payload_class not in hex_classes for payload_class in model_classes):
            raise RevisionRunnerError("Every smoke model must include a non-hex class")
        for position, row in enumerate(model_rows):
            original_id = str(row["trial_id"])
            row.update(
                {
                    "trial_id": _smoke_trial_id(model_id, position, original_id),
                    "original_confirmatory_trial_id": original_id,
                    "study_phase": "smoke_exploratory",
                    "generation_required": True,
                    "primary_overlap": False,
                    "source_trial_id": None,
                    "notes": "Balanced exploratory smoke path; never pool with confirmatory data.",
                }
            )
            selected.append(row)
    expected_classes = {payload.payload_class for payload in payloads}
    observed_classes = {str(row["payload_class"]) for row in selected}
    if observed_classes != expected_classes:
        raise RevisionRunnerError(
            "Smoke payload-class coverage drifted: {}".format(
                sorted(expected_classes - observed_classes)
            )
        )
    if len({str(row["prompt_id"]) for row in selected}) != 18:
        raise RevisionRunnerError("Smoke plan must exercise all 18 prompt templates")
    return selected


def _build_smoke_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    work: List[Dict[str, object]] = []
    for row in _select_smoke_rankcloak_rows(configs, payloads):
        work.append(_rankcloak_work(row, EVIDENCE_SMOKE))
        work.append(_control_work(_smoke_control(row, "full_message"), EVIDENCE_SMOKE))
        if bool(row.get("segmented")):
            work.append(_control_work(_smoke_control(row, "forced_span"), EVIDENCE_SMOKE))
    return work


def _build_smoke_v2_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    """Reissue the balanced smoke under disjoint immutable work identities.

    Smoke v1 remains readable and unchanged.  V2 exists because the frozen
    feasibility rule and source manifest changed after the first model audit.
    """

    original = _build_smoke_work_plan(configs, payloads)
    identity_map = {
        str(row["work_id"]): _stable_id("revision_v1_smoke_v2", row["work_id"])
        for row in original
    }
    result: List[Dict[str, object]] = []
    for source in original:
        row = dict(source)
        old_work_id = str(row["work_id"])
        row["original_smoke_v1_work_id"] = old_work_id
        row["work_id"] = identity_map[old_work_id]
        row["study_phase"] = "smoke_v2_exploratory"
        if row["work_kind"] == "rankcloak":
            row["trial_id"] = row["work_id"]
        else:
            row["control_id"] = row["work_id"]
            source_id = str(row["source_trial_id"])
            row["source_trial_id"] = identity_map[source_id]
        result.append(row)
    return result



SUPERSEDING_BASE_STAGE = {
    "smoke_v3": "smoke_v2",
    "primary_v2": "primary",
    "ablation_v2": "ablation",
    "multilingual_v2": "multilingual",
    "robustness_v2": "robustness",
}
SUPERSEDING_SOURCE_STAGE = {
    "primary": "primary_v2",
    "ablation": "ablation_v2",
}


def _superseding_work_id(stage: str, superseded_work_id: str) -> str:
    if stage not in SUPERSEDING_BASE_STAGE:
        raise RevisionRunnerError("Unknown superseding stage: {}".format(stage))
    return _stable_id(
        "revision_v1_{}_{}".format(stage, PROTOCOL_CONTRACT_REVISION),
        superseded_work_id,
    )


def _superseding_study_phase(
    stage: str, row: Mapping[str, object]
) -> str:
    if stage == "smoke_v3":
        return (
            "ordinary_llm_control_smoke_v3"
            if row.get("work_kind") == "control"
            else "smoke_v3_exploratory"
        )
    if stage == "primary_v2":
        return (
            "ordinary_llm_control_primary_v2"
            if row.get("work_kind") == "control"
            else "primary_v2_confirmatory"
        )
    if stage == "ablation_v2":
        return "ablation_v2_confirmatory"
    if stage == "multilingual_v2":
        return (
            "ordinary_llm_control_multilingual_v2"
            if row.get("work_kind") == "control"
            else "multilingual_v2_secondary"
        )
    if stage == "robustness_v2":
        return (
            "robustness_v2_transformation_generation"
            if row.get("work_kind") == "robustness_transform"
            else "robustness_v2_confirmatory_supporting"
        )
    raise RevisionRunnerError("Unknown superseding stage: {}".format(stage))



def _contract_record_fields(task: Mapping[str, object]) -> Dict[str, str]:
    protocol_revision = task.get("protocol_contract_revision")
    result_revision = task.get("result_schema_revision")
    if protocol_revision is None and result_revision is None:
        return {}
    if protocol_revision != PROTOCOL_CONTRACT_REVISION:
        raise RevisionRunnerError(
            "Superseding task has an invalid protocol_contract_revision"
        )
    if result_revision != RESULT_SCHEMA_REVISION:
        raise RevisionRunnerError(
            "Superseding task has an invalid result_schema_revision"
        )
    return {
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
    }


def _supersede_work_plan(
    original: Sequence[Mapping[str, object]],
    stage: str,
    *,
    original_confirmatory_stage_by_id: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    # Create a disjoint payload-fidelity-v2 plan without changing conditions.
    if stage not in SUPERSEDING_BASE_STAGE:
        raise RevisionRunnerError("Unknown superseding stage: {}".format(stage))
    identity_map = {
        str(row["work_id"]): _superseding_work_id(stage, str(row["work_id"]))
        for row in original
    }
    if len(identity_map) != len(original):
        raise RevisionRunnerError("Superseded plan contains duplicate work IDs")
    result: List[Dict[str, object]] = []
    evidence_status = _stage_evidence_status(stage)
    for source in original:
        row = dict(source)
        old_work_id = str(source["work_id"])
        new_work_id = identity_map[old_work_id]
        row.update(
            {
                "work_id": new_work_id,
                "supersedes_work_id": old_work_id,
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "evidence_status": evidence_status,
                "study_phase": _superseding_study_phase(stage, source),
            }
        )
        if source.get("work_kind") == "control":
            if str(source.get("control_id")) != old_work_id:
                raise RevisionRunnerError("Control identity differs from work identity")
            row["control_id"] = new_work_id
        elif source.get("trial_id") is not None:
            if str(source.get("trial_id")) != old_work_id:
                raise RevisionRunnerError("Trial identity differs from work identity")
            row["trial_id"] = new_work_id

        old_source = source.get("source_trial_id")
        if old_source is not None:
            old_source_id = str(old_source)
            row["supersedes_source_trial_id"] = old_source_id
            if old_source_id in identity_map:
                row["source_trial_id"] = identity_map[old_source_id]
            elif stage == "ablation_v2":
                row["source_trial_id"] = _superseding_work_id(
                    "primary_v2", old_source_id
                )
            elif stage == "robustness_v2":
                old_source_stage = str(source.get("source_stage"))
                new_source_stage = SUPERSEDING_SOURCE_STAGE.get(old_source_stage)
                if new_source_stage is None:
                    raise RevisionRunnerError(
                        "Robustness source stage cannot be superseded: {}".format(
                            old_source_stage
                        )
                    )
                row["source_stage"] = new_source_stage
                row["source_trial_id"] = _superseding_work_id(
                    new_source_stage, old_source_id
                )
            else:
                raise RevisionRunnerError(
                    "Unmapped external source identity in {}: {}".format(
                        stage, old_source_id
                    )
                )

        transform_work_id = source.get("transform_work_id")
        if transform_work_id is not None:
            old_transform_id = str(transform_work_id)
            if old_transform_id not in identity_map:
                raise RevisionRunnerError(
                    "Transform dependency is absent from the robustness plan"
                )
            row["supersedes_transform_work_id"] = old_transform_id
            row["transform_work_id"] = identity_map[old_transform_id]

        original_primary_id = source.get("original_confirmatory_trial_id")
        if original_primary_id is not None:
            old_primary_id = str(original_primary_id)
            if original_confirmatory_stage_by_id is None:
                raise RevisionRunnerError(
                    "Original confirmatory identity has no frozen-plan membership map"
                )
            original_stage = original_confirmatory_stage_by_id.get(old_primary_id)
            new_original_stage = SUPERSEDING_SOURCE_STAGE.get(str(original_stage))
            if new_original_stage not in {"primary_v2", "ablation_v2"}:
                raise RevisionRunnerError(
                    "Original confirmatory identity is absent or ambiguous across "
                    "the frozen primary/ablation plans: {}".format(old_primary_id)
                )
            row["supersedes_original_confirmatory_trial_id"] = old_primary_id
            row["original_confirmatory_trial_id"] = _superseding_work_id(
                new_original_stage, old_primary_id
            )
        result.append(row)

    if {str(row["work_id"]) for row in result} & set(identity_map):
        raise RevisionRunnerError("Superseding work identities overlap prior identities")
    return result


def _build_smoke_v3_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    primary_ids = {
        str(row["work_id"])
        for row in _build_primary_work_plan(configs, payloads)
    }
    ablation_ids = {
        str(row["work_id"])
        for row in _build_ablation_work_plan(configs, payloads)
    }
    if primary_ids & ablation_ids:
        raise RevisionRunnerError(
            "Frozen primary and ablation plans have ambiguous work identities"
        )
    original_stage_by_id = {
        **{work_id: "primary" for work_id in primary_ids},
        **{work_id: "ablation" for work_id in ablation_ids},
    }
    return _supersede_work_plan(
        _build_smoke_v2_work_plan(configs, payloads),
        "smoke_v3",
        original_confirmatory_stage_by_id=original_stage_by_id,
    )


def _build_primary_v2_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    return _supersede_work_plan(_build_primary_work_plan(configs, payloads), "primary_v2")


def _build_ablation_v2_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    return _supersede_work_plan(_build_ablation_work_plan(configs, payloads), "ablation_v2")


def _build_multilingual_v2_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    return _supersede_work_plan(
        _build_multilingual_work_plan(configs, payloads), "multilingual_v2"
    )

def _robustness_work(
    source: Mapping[str, object],
    family: str,
    replay_mode: str,
    transformation_id: str,
    source_stage: str,
    execution_model_id: Optional[str] = None,
    reference: bool = False,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    source_trial_id = str(source["trial_id"])
    decoder_model_id = str(execution_model_id or source["model_id"])
    work_id = _stable_id(
        "revision_v1_robustness",
        family,
        source_trial_id,
        replay_mode,
        transformation_id,
        decoder_model_id,
    )
    row: Dict[str, object] = {
        "work_id": work_id,
        "trial_id": work_id,
        "work_kind": "reference" if reference else "robustness_decode",
        "generation_required": False,
        "decode_required": not reference,
        "study_id": source["study_id"],
        "study_phase": "robustness_confirmatory_supporting",
        "evidence_status": EVIDENCE_ROBUSTNESS,
        "robustness_family": family,
        "model_id": decoder_model_id,
        "source_model_id": source["model_id"],
        "source_stage": source_stage,
        "source_trial_id": source_trial_id,
        "payload_name": source["payload_name"],
        "payload_class": source["payload_class"],
        "payload_index": source["payload_index"],
        "payload_split": source["payload_split"],
        "prompt_id": source["prompt_id"],
        "prompt_category": source["prompt_category"],
        "language": source["language"],
        "protocol_variant": source["protocol_variant"],
        "replay_mode": replay_mode,
        "transformation_id": transformation_id,
    }
    if extra:
        row.update(dict(extra))
    return row


def _robustness_transform_work(
    source: Mapping[str, object],
    transformation: Mapping[str, object],
) -> Dict[str, object]:
    model_id = str(transformation["model_id"])
    work_id = _stable_id(
        "revision_v1_robustness_transform",
        source["trial_id"],
        transformation["transformation_id"],
        model_id,
    )
    return {
        "work_id": work_id,
        "trial_id": work_id,
        "work_kind": "robustness_transform",
        "generation_required": True,
        "decode_required": False,
        "study_id": source["study_id"],
        "study_phase": "robustness_transformation_generation",
        "evidence_status": EVIDENCE_ROBUSTNESS,
        "robustness_family": "raw_transmission_transform",
        "model_id": model_id,
        "source_model_id": source["model_id"],
        "source_stage": "primary",
        "source_trial_id": source["trial_id"],
        "payload_name": source["payload_name"],
        "payload_class": source["payload_class"],
        "payload_index": source["payload_index"],
        "payload_split": source["payload_split"],
        "prompt_id": source["prompt_id"],
        "prompt_category": source["prompt_category"],
        "language": source["language"],
        "protocol_variant": source["protocol_variant"],
        "replay_mode": "not_applicable_transform_generation",
        "transformation_id": "paraphrase",
        "transformation": dict(transformation),
    }


def _build_robustness_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    """Materialize 3,600 outcomes plus 144 Qwen transform artifacts."""

    primary = build_primary_trial_plan(payloads=payloads, configs=configs)
    ablation = build_ablation_trial_plan(payloads=payloads, configs=configs)
    canonical_sources = [
        row
        for row in primary
        if row["protocol_variant"] == "segmented_hex_multi_topic"
        and int(row["payload_index"]) <= 11
    ]
    leadin_sources = [
        row
        for row in ablation
        if row.get("ablation_factor") == "leadin_tokens"
        and row.get("ablation_level") == 8
    ]
    roundtrip_sources = [
        row
        for row in ablation
        if row.get("ablation_factor") == "token_filter"
        and row.get("ablation_level") == "roundtrip_stable_filter_v1"
    ]
    if not (
        len(canonical_sources) == len(leadin_sources) == len(roundtrip_sources) == 144
    ):
        raise RevisionRunnerError(
            "Robustness source selections must each contain 144 paired blocks"
        )
    robustness = configs["robustness"]
    transformations = list(robustness["transformations"])
    paraphrase = next(
        dict(row)
        for row in transformations
        if row["transformation_id"] == "paraphrase"
    )

    # Transform artifacts precede every outcome so the Qwen shard can produce
    # them first. Source-model decoder shards consume these immutable records.
    work: List[Dict[str, object]] = [
        _robustness_transform_work(source, paraphrase)
        for source in canonical_sources
    ]
    transform_id_by_source = {
        str(row["source_trial_id"]): str(row["work_id"])
        for row in work
    }

    replay_levels = list(robustness["replay_modes"]["levels"])
    for source in leadin_sources:
        for replay_mode in replay_levels:
            work.append(
                _robustness_work(
                    source,
                    "replay_modes",
                    str(replay_mode),
                    "unmodified",
                    "ablation",
                    reference=replay_mode == "saved_token_ids",
                )
            )

    for source in canonical_sources:
        for transformation in transformations:
            transformation_id = str(transformation["transformation_id"])
            extra: Dict[str, object] = {"transformation": dict(transformation)}
            if transformation_id == "paraphrase":
                extra["transform_work_id"] = transform_id_by_source[
                    str(source["trial_id"])
                ]
                extra["transformation_model_id"] = paraphrase["model_id"]
            work.append(
                _robustness_work(
                    source,
                    "raw_transmission",
                    "transformed_text_retokenized",
                    transformation_id,
                    "primary",
                    execution_model_id=str(source["model_id"]),
                    reference=transformation_id == "unmodified",
                    extra=extra,
                )
            )

    mitigation_ids = list(
        robustness["limited_mitigation"]["transformation_ids"]
    )
    mitigation_pipeline = list(robustness["limited_mitigation"]["pipeline"])
    for source in roundtrip_sources:
        for transformation_id in mitigation_ids:
            work.append(
                _robustness_work(
                    source,
                    "limited_mitigation",
                    "canonicalized_text_retokenized",
                    str(transformation_id),
                    "ablation",
                    reference=transformation_id == "unmodified",
                    extra={"mitigation_pipeline": mitigation_pipeline},
                )
            )

    models = model_ids(configs["models"])
    for source in canonical_sources:
        for decoder_model_id in models:
            if decoder_model_id == source["model_id"]:
                continue
            work.append(
                _robustness_work(
                    source,
                    "cross_model_mismatch",
                    "cross_model_text_retokenized",
                    "unmodified",
                    "primary",
                    execution_model_id=decoder_model_id,
                )
            )

    expected = robustness["expected_counts"]
    outcomes = [row for row in work if row["work_kind"] != "robustness_transform"]
    decode_count = sum(row["work_kind"] == "robustness_decode" for row in work)
    transform_count = sum(row["work_kind"] == "robustness_transform" for row in work)
    if len(outcomes) != int(expected["robustness_outcome_rows"]):
        raise RevisionRunnerError("Robustness outcome-row arithmetic drifted")
    if decode_count != int(expected["additional_decode_only_runs"]):
        raise RevisionRunnerError("Robustness decode-only arithmetic drifted")
    if transform_count != 144:
        raise RevisionRunnerError("Robustness paraphrase-transform arithmetic drifted")
    return work


def _build_robustness_v2_work_plan(
    configs: Mapping[str, Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
) -> List[Dict[str, object]]:
    return _supersede_work_plan(
        _build_robustness_work_plan(configs, payloads), "robustness_v2"
    )


def build_stage_plan(
    stage: str,
    model_id: Optional[str] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    payloads: Optional[Sequence[RevisionPayload]] = None,
) -> List[Dict[str, object]]:
    """Materialize ordered generation/reference work for one frozen stage."""

    if stage not in STAGES:
        raise RevisionConfigError("Unknown revision stage: {}".format(stage))
    loaded = dict(configs) if configs is not None else load_revision_config_set()
    selected_payloads = list(payloads) if payloads is not None else _default_payloads()
    builders = {
        "smoke": _build_smoke_work_plan,
        "smoke_v2": _build_smoke_v2_work_plan,
        "smoke_v3": _build_smoke_v3_work_plan,
        "primary": _build_primary_work_plan,
        "primary_v2": _build_primary_v2_work_plan,
        "ablation": _build_ablation_work_plan,
        "ablation_v2": _build_ablation_v2_work_plan,
        "multilingual": _build_multilingual_work_plan,
        "multilingual_v2": _build_multilingual_v2_work_plan,
        "robustness": _build_robustness_work_plan,
        "robustness_v2": _build_robustness_v2_work_plan,
    }
    work = builders[stage](loaded, selected_payloads)
    known_models = model_ids(loaded["models"])
    if model_id is not None:
        if model_id not in known_models:
            raise RevisionConfigError("Unknown model_id: {}".format(model_id))
        work = [row for row in work if row["model_id"] == model_id]
    work_ids = [str(row["work_id"]) for row in work]
    if len(work_ids) != len(set(work_ids)):
        raise RevisionRunnerError("Stage plan has duplicate work IDs")
    return work


def plan_summary(plan: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "planned_work_units": len(plan),
        "work_kind_counts": dict(sorted(Counter(str(row["work_kind"]) for row in plan).items())),
        "model_counts": dict(sorted(Counter(str(row["model_id"]) for row in plan).items())),
        "protocol_counts": dict(
            sorted(
                Counter(
                    str(row.get("protocol_variant", "ordinary_control"))
                    for row in plan
                ).items()
            )
        ),
        "evidence_status_counts": dict(
            sorted(Counter(str(row["evidence_status"]) for row in plan).items())
        ),
    }


def relabel_limited_plan(plan: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Give diagnostic subsets disjoint identities and evidence labels."""

    identity_map = {
        str(row["work_id"]): _stable_id("revision_v1_limited", row["work_id"])
        for row in plan
    }
    relabeled: List[Dict[str, object]] = []
    for source in plan:
        row = dict(source)
        old_work_id = str(row["work_id"])
        row["original_frozen_work_id"] = old_work_id
        row["work_id"] = identity_map[old_work_id]
        row["evidence_status"] = EVIDENCE_LIMITED
        if row["work_kind"] != "control":
            row["trial_id"] = row["work_id"]
        else:
            row["control_id"] = row["work_id"]
            source_id = str(row["source_trial_id"])
            if source_id in identity_map:
                row["source_trial_id"] = identity_map[source_id]
        transform_id = row.get("transform_work_id")
        if transform_id is not None and str(transform_id) in identity_map:
            row["transform_work_id"] = identity_map[str(transform_id)]
        relabeled.append(row)
    return relabeled


def _model_entry(configs: Mapping[str, Mapping[str, object]], model_id: str) -> Dict[str, object]:
    matches = [
        dict(row)
        for row in configs["models"].get("models", [])
        if isinstance(row, dict) and row.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise RevisionRunnerError("Expected one pinned model entry for {}".format(model_id))
    return matches[0]


def _prompt_text(task: Mapping[str, object], configs: Mapping[str, Mapping[str, object]]) -> str:
    if task.get("prompt_text") is not None:
        return str(task["prompt_text"])
    return str(prompt_by_id(str(task["prompt_id"]), configs=configs)["text"])


def _segment_prompts(
    task: Mapping[str, object],
    segment_count: int,
    configs: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, str]]:
    base_text = _prompt_text(task, configs)
    base = {
        "prompt_id": str(task["prompt_id"]),
        "prompt_category": str(task["prompt_category"]),
        "prompt_text": base_text,
    }
    if task.get("topic_schedule") != "deterministic_six_category_rotation":
        return [dict(base) for _ in range(segment_count)]
    prompts = flatten_prompt_templates(configs["prompts"])
    matches = [row for row in prompts if row["prompt_id"] == task["prompt_id"]]
    if len(matches) != 1:
        raise RevisionRunnerError("Cannot resolve base prompt for topic rotation")
    base_row = matches[0]
    categories = configs["prompts"]["categories"]
    base_category_index = int(base_row["category_index"])
    template_index = int(base_row["template_index"])
    result: List[Dict[str, str]] = []
    for index in range(segment_count):
        category = categories[(base_category_index + index) % len(categories)]
        template = category["templates"][template_index]
        result.append(
            {
                "prompt_id": str(template["prompt_id"]),
                "prompt_category": str(category["category_id"]),
                "prompt_text": str(template["text"]),
            }
        )
    return result


def build_representation(model: Any, task: Mapping[str, object], payload: RevisionPayload) -> Representation:
    variant = str(task["protocol_variant"])
    if variant == "direct_subword_calgacus":
        return direct_representation(model, payload.payload_text)
    if variant == "nonseg_ascii_b8":
        return bounded_representation(payload.payload_bytes, payload.payload_text, "ascii_b8")
    if variant in {"nonseg_ascii_b16"}:
        return bounded_representation(payload.payload_bytes, payload.payload_text, "ascii_b16")
    if variant in {
        "nonseg_hex_nibble_b16",
        "segmented_hex_single_topic",
        "segmented_hex_multi_topic",
    }:
        return bounded_representation(payload.payload_bytes, payload.payload_text, "hex_nibble")
    raise RevisionRunnerError("Unsupported protocol variant: {}".format(variant))



def representation_capacity_bits(
    representation: Representation,
) -> Tuple[Optional[int], str]:
    """Return the information estimand actually encoded by a bounded codec."""

    metadata = dict(representation.metadata)
    if representation.name == "hex_nibble":
        length = int(metadata["hex_character_length"])
        return 4 * length, "raw_hex_nibbles_4_bits_per_character"
    if representation.name in {"ascii_b8", "ascii_b16"}:
        length = int(metadata["original_byte_length"])
        return 8 * length, "display_bytes_8_bits_per_original_byte"
    if representation.name == "direct_subword":
        return None, "not_defined_for_variable_unbounded_direct_subword_ranks"
    raise RevisionRunnerError(
        "No capacity-bit estimand for representation {}".format(
            representation.name
        )
    )

def derive_control_seed(source_trial_id: str, control_view: str) -> int:
    suffix = "/full-control" if control_view == "full_message" else "/forced-control"
    digest = hashlib.sha256((str(source_trial_id) + suffix).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _call_token_id(model: Any, names: Sequence[str]) -> Optional[int]:
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        try:
            token_id = int(value() if callable(value) else value)
        except Exception:
            continue
        if token_id >= 0:
            return token_id
    return None


def excluded_control_token_ids(model: Any) -> List[int]:
    """Return BOS/EOS/EOT IDs excluded by the historical control sampler."""

    values = {
        value
        for value in (
            get_bos_token_id(model),
            _call_token_id(model, ("token_eos", "eos_token_id")),
            _call_token_id(model, ("token_eot", "eot_token_id")),
        )
        if value is not None
    }
    return sorted(map(int, values))


def sample_top_p_token(
    logits: Sequence[float],
    rng: np.random.Generator,
    temperature: float,
    top_p: float,
    excluded_token_ids: Sequence[int] = (),
    allowed_token_mask: Optional[Sequence[bool]] = None,
) -> int:
    """Sample one token under the historical serial PCG64 top-p contract."""

    values = np.asarray(logits, dtype=np.float64).copy()
    if values.ndim != 1 or not values.size:
        raise RevisionRunnerError("Sampling logits must be a non-empty vector")
    if allowed_token_mask is not None:
        mask = np.asarray(allowed_token_mask, dtype=bool)
        if mask.shape != values.shape:
            raise RevisionRunnerError("Allowed-token mask must match sampling logits")
        values[~mask] = -np.inf
    for token_id in excluded_token_ids:
        if 0 <= int(token_id) < values.size:
            values[int(token_id)] = -np.inf
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RevisionRunnerError("No finite token remains for ordinary control sampling")
    if not (float(temperature) > 0.0):
        raise RevisionRunnerError("Control temperature must be positive")
    if not (0.0 < float(top_p) <= 1.0):
        raise RevisionRunnerError("Control top_p must be in (0, 1]")
    token_ids = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((token_ids, -values))
    order = order[np.isfinite(values[order])]
    scaled = values[order] / float(temperature)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    keep = int(np.searchsorted(cumulative, float(top_p), side="left")) + 1
    order = order[:keep]
    probabilities = probabilities[:keep]
    probabilities /= probabilities.sum()
    draw = float(rng.random())
    selected = min(
        int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
        len(order) - 1,
    )
    return int(order[selected])


def generate_length_matched_control(
    model: Any,
    prompt_text: str,
    target_token_count: int,
    seed: int,
    temperature: float = CONTROL_TEMPERATURE,
    top_p: float = CONTROL_TOP_P,
    context_limit: int = 4096,
) -> Dict[str, object]:
    """Generate an exact-length ordinary control with a seeded serial sampler."""

    target = int(target_token_count)
    if target < 0:
        raise RevisionRunnerError("Control target length cannot be negative")
    context = make_context_token_ids(model, str(prompt_text))
    if len(context) + target > int(context_limit):
        raise RevisionRunnerError(
            "Control target {} plus prompt {} exceeds context {}".format(
                target, len(context), context_limit
            )
        )
    evaluate_context(model, context)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    excluded = excluded_control_token_ids(model)
    token_ids: List[int] = []
    log_probabilities: List[float] = []
    entropies: List[float] = []
    sampled_ranks: List[int] = []
    greedy_token_ids: List[int] = []
    greedy_log_probabilities: List[float] = []
    rank_pressure_gaps: List[float] = []
    for _ in range(target):
        logits = get_last_logits(model)
        token_id = sample_top_p_token(
            logits, rng, float(temperature), float(top_p), excluded
        )
        token_ids.append(token_id)
        diagnostic = next_token_diagnostic(logits, token_id)
        log_probabilities.append(
            float(diagnostic["observed_log_probability"])
        )
        entropies.append(float(diagnostic["entropy_bits"]))
        sampled_ranks.append(int(diagnostic["observed_rank"]))
        greedy_token_ids.append(int(diagnostic["greedy_token_id"]))
        greedy_log_probabilities.append(
            float(diagnostic["greedy_log_probability"])
        )
        rank_pressure_gaps.append(
            float(diagnostic["rank_pressure_log_probability_gap_nats"])
        )
        model.eval([token_id])
    return {
        "context_token_ids": list(map(int, context)),
        "token_ids": token_ids,
        "token_role_mask": ["ordinary_control"] * len(token_ids),
        "text": safe_detokenize(model, token_ids),
        "token_log_probabilities": log_probabilities,
        "next_token_entropies_bits": entropies,
        "sampled_token_ranks": sampled_ranks,
        "greedy_token_ids": greedy_token_ids,
        "greedy_log_probabilities": greedy_log_probabilities,
        "rank_pressure_log_probability_gaps_nats": rank_pressure_gaps,
        "target_token_count": target,
        "sampling_seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "excluded_special_token_ids": excluded,
        "sampler": "numpy_pcg64_serial_top_p_v1_token_id_tiebreak",
    }


def _json_safe(value: object) -> object:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _decode_record(decoded: Mapping[str, object]) -> Dict[str, object]:
    recovered_bytes = decoded.get("recovered_bytes", b"")
    if not isinstance(recovered_bytes, bytes):
        recovered_bytes = bytes(recovered_bytes)
    serialized_bytes = decoded.get("recovered_serialized_bytes", recovered_bytes)
    if not isinstance(serialized_bytes, bytes):
        serialized_bytes = bytes(serialized_bytes)
    exact_payload = bool(decoded.get("exact_payload_recovery", False))
    return {
        "success": bool(decoded.get("success")),
        "recovery_outcome_semantics": decoded.get("recovery_outcome_semantics"),
        # exact_recovery is an alias for literal original-payload recovery.
        "exact_recovery": exact_payload,
        "exact_payload_recovery": exact_payload,
        "exact_representation_recovery": bool(
            decoded.get("exact_representation_recovery", False)
        ),
        "recovered_text": str(decoded.get("recovered_text", "")),
        "recovered_bytes_base64": base64.b64encode(recovered_bytes).decode("ascii"),
        "recovered_serialized_bytes_base64": base64.b64encode(
            serialized_bytes
        ).decode("ascii"),
        "recovered_token_ids": list(map(int, decoded.get("recovered_token_ids", []))),
        "original_payload_sha256": decoded.get("original_payload_sha256"),
        "recovered_payload_sha256": decoded.get("recovered_payload_sha256"),
        "detokenized_prefix_exact": decoded.get("detokenized_prefix_exact"),
        "error": decoded.get("error"),
    }


def _mask_metadata(mask: Optional[np.ndarray], filter_name: Optional[str]) -> Dict[str, object]:
    if mask is None:
        return {
            "filter_name": filter_name or "none",
            "vocabulary_size": None,
            "allowed_token_count": None,
            "packed_mask_sha256": None,
        }
    boolean_mask = np.asarray(mask, dtype=bool)
    packed = np.packbits(boolean_mask.astype(np.uint8), bitorder="little").tobytes()
    return {
        "filter_name": str(filter_name),
        "vocabulary_size": int(boolean_mask.size),
        "allowed_token_count": int(np.count_nonzero(boolean_mask)),
        "packed_mask_sha256": hashlib.sha256(packed).hexdigest(),
    }


def _get_filter_mask(
    model: Any,
    model_id: str,
    filter_name: Optional[str],
    cache: MutableMapping[
        str, Tuple[Optional[np.ndarray], Dict[str, object]]
    ],
    output_dir: Optional[Path],
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    key = str(filter_name or "none")
    if key not in cache:
        try:
            mask = build_revision_filter_mask(model, key)
            cache[key] = (mask, _mask_metadata(mask, key))
        except ValueError as exc:
            if not (
                key == "roundtrip_stable_filter_v1"
                and str(exc)
                == "Round-trip-stable filter rejected the entire vocabulary"
            ):
                raise
            safe_mask = build_revision_filter_mask(model, "safe_text_filter_v1")
            safe_array = np.asarray(safe_mask, dtype=bool)
            unavailable = {
                "filter_name": key,
                "condition_unavailable": True,
                "reason_code": "empty_isolated_roundtrip_vocabulary",
                "reason": (
                    "No token satisfies safe_text=true and isolated "
                    "detokenize-to-retokenize exactness for the pinned tokenizer."
                ),
                "feasibility_criterion": (
                    "safe_text_true_and_isolated_detokenize_retokenize_exact"
                ),
                "minimum_allowed_token_count": 1,
                "safe_count": int(np.count_nonzero(safe_array)),
                "stable_count": 0,
                "vocabulary_size": int(safe_array.size),
                "prefix_conditioned_substitution_permitted": False,
            }
            cache[key] = (None, unavailable)
    mask, metadata = cache[key]
    if bool(metadata.get("condition_unavailable")):
        raise ConditionUnavailable(metadata)
    if output_dir is not None and mask is not None:
        artifact_path = (
            Path(output_dir)
            / "filter_masks"
            / "{}__{}.json".format(model_id, key)
        )
        persistence_key = (
            str(artifact_path.resolve()),
            str(metadata["packed_mask_sha256"]),
        )
        if persistence_key not in _PERSISTED_FILTER_MASK_ARTIFACTS:
            allowed = (
                np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()
            )
            artifact = dict(metadata)
            artifact.update(
                {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "model_id": model_id,
                    "allowed_token_ids": allowed,
                    "encoding": "sorted allowed token IDs; complement is rejected",
                }
            )
            write_immutable_json(artifact_path, artifact)
            _PERSISTED_FILTER_MASK_ARTIFACTS.add(persistence_key)
    return mask, dict(metadata)


def _role_mask(generated: Mapping[str, object]) -> List[str]:
    return (
        ["leadin"] * len(generated["leadin_token_ids"])
        + ["forced"] * len(generated["forced_token_ids"])
        + ["tail"] * len(generated["tail_token_ids"])
    )


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(np.asarray(values, dtype=float))) if values else None


def _complete_failure_record(
    failure: Mapping[str, object],
    task: Mapping[str, object],
    replay_mode: str,
) -> Dict[str, object]:
    """Add the exact frozen robustness failure-field aliases."""

    record = dict(failure)
    rank_position = dict(record.get("first_rank_divergence", {})).get(
        "position_zero_based"
    )
    token_position = dict(record.get("first_token_divergence", {})).get(
        "position_zero_based"
    )
    record.update(
        {
            "trial_id": task["trial_id"],
            "transformation_id": "unmodified",
            "replay_mode": replay_mode,
            "first_differing_position": (
                rank_position if rank_position is not None else token_position
            ),
            "boundary_start_offset": record.get("boundary_start"),
            "boundary_end_offset": record.get("boundary_stop"),
        }
    )
    return record


def _payload_mismatch_failure(
    decoded: Mapping[str, object],
    task: Mapping[str, object],
    replay_mode: str,
    segment: Mapping[str, object],
) -> Dict[str, object]:
    """Classify literal-payload failure after exact rank representation replay."""

    no_rank_divergence = first_divergence([], [])
    failure = {
        "failure_category": "inverse_payload_fidelity_mismatch",
        "failure_layer": "original_serialized_payload_bytes",
        "first_rank_divergence": no_rank_divergence,
        "first_token_divergence": first_divergence([], []),
        "expected_token_id": None,
        "recovered_token_id": None,
        "expected_rank": None,
        "recovered_rank": None,
        "context_sha256": segment.get("context_sha256"),
        "boundary_start": segment.get("forced_start"),
        "boundary_stop": segment.get("forced_stop"),
        "expected_payload_sha256": decoded.get("original_payload_sha256"),
        "recovered_payload_sha256": decoded.get("recovered_payload_sha256"),
        "recovery_outcome_semantics": decoded.get("recovery_outcome_semantics"),
        "exact_representation_recovery": decoded.get(
            "exact_representation_recovery"
        ),
        "exact_payload_recovery": decoded.get("exact_payload_recovery"),
        "segment_index": 0,
    }
    return _complete_failure_record(failure, task, replay_mode)


def execute_rankcloak_trial(
    model: Any,
    task: Mapping[str, object],
    payload: RevisionPayload,
    configs: Mapping[str, Mapping[str, object]],
    mask_cache: Optional[MutableMapping[
        str, Tuple[Optional[np.ndarray], Dict[str, object]]
    ]] = None,
    output_dir: Optional[Path] = None,
    context_limit: int = 4096,
) -> Dict[str, object]:
    """Execute generation, exact replay, inverse decoding, and diagnostics."""

    started = time.perf_counter()
    process_peak_rss_start = process_peak_rss_bytes()
    mask_cache = mask_cache if mask_cache is not None else {}
    representation_started = time.perf_counter()
    representation = build_representation(model, task, payload)
    representation_seconds = time.perf_counter() - representation_started
    ranks = list(map(int, representation.ranks))
    segmented = bool(task.get("segmented"))
    segment_size = int(task.get("segment_size_ranks") or len(ranks) or 1)
    rank_chunks = [ranks[index : index + segment_size] for index in range(0, len(ranks), segment_size)]
    if not segmented:
        rank_chunks = [ranks]
    prompts = _segment_prompts(task, len(rank_chunks), configs)
    filter_name = str(task.get("token_filter") or "none")
    filter_started = time.perf_counter()
    allowed_mask, allowed_mask_metadata = _get_filter_mask(
        model,
        str(task["model_id"]),
        filter_name,
        mask_cache,
        output_dir,
    )
    filter_setup_seconds = time.perf_counter() - filter_started
    leadin_tokens = int(task.get("leadin_tokens") or 0)
    tail_policy = str(task.get("tail_policy") or TAIL_NONE)
    replay_modes = tuple(
        map(str, task.get("replay_modes", ["saved_token_ids"]))
    )
    allowed_replay_modes = {
        "saved_token_ids",
        "detokenized_text_retokenized",
        "greedy_leadin_regeneration",
    }
    if "saved_token_ids" not in replay_modes:
        raise RevisionRunnerError("Every trial must include saved_token_ids replay")
    unknown_replay_modes = set(replay_modes) - allowed_replay_modes
    if unknown_replay_modes:
        raise RevisionRunnerError(
            "Unknown replay modes: {}".format(
                ", ".join(sorted(unknown_replay_modes))
            )
        )
    run_text_replay = "detokenized_text_retokenized" in replay_modes
    run_greedy_replay = "greedy_leadin_regeneration" in replay_modes

    segments: List[Dict[str, object]] = []
    saved_ranks: List[int] = []
    text_ranks: List[int] = []
    greedy_ranks: List[int] = []
    saved_exact_by_segment: List[bool] = []
    text_exact_by_segment: List[bool] = []
    greedy_exact_by_segment: List[bool] = []
    forced_logp: List[float] = []
    tail_logp: List[float] = []
    leadin_logp: List[float] = []
    generation_seconds = 0.0
    saved_replay_seconds = 0.0
    greedy_replay_seconds = 0.0
    text_replay_seconds = 0.0
    for segment_index, (rank_chunk, prompt) in enumerate(zip(rank_chunks, prompts)):
        context = make_context_token_ids(model, prompt["prompt_text"])
        maximum_tail = 256 if tail_policy == "dynamic_completion_v1" else 60 if tail_policy == "sentence_tail_min20_max60" else 40 if tail_policy == "fixed_tail40" else 0
        if len(context) + leadin_tokens + len(rank_chunk) + maximum_tail > int(context_limit):
            raise RevisionRunnerError(
                "Segment {} can exceed configured context {}".format(segment_index, context_limit)
            )
        operation_started = time.perf_counter()
        generated = generate_rank_span(
            model,
            context,
            rank_chunk,
            allowed_token_mask=allowed_mask,
            leadin_token_count=leadin_tokens,
            tail_policy=tail_policy,
            quality_rank_ceiling=(
                int(task["alphabet_size"])
                if task.get("alphabet_size") is not None
                else None
            ),
        )
        segment_generation_seconds = time.perf_counter() - operation_started
        generation_seconds += segment_generation_seconds
        operation_started = time.perf_counter()
        recovered = recover_rank_span(
            model,
            context,
            generated["leadin_token_ids"],
            generated["forced_token_ids"],
            allowed_token_mask=allowed_mask,
        )
        segment_saved_replay_seconds = time.perf_counter() - operation_started
        saved_replay_seconds += segment_saved_replay_seconds
        recovered_ranks = list(map(int, recovered["ranks"]))
        saved_ranks.extend(recovered_ranks)
        saved_exact_by_segment.append(recovered_ranks == rank_chunk)

        segment_greedy_replay_seconds = 0.0
        regenerated_leadin: List[int] = []
        greedy_segment_ranks: List[int] = []
        greedy_segment_exact: Optional[bool] = None
        if run_greedy_replay:
            operation_started = time.perf_counter()
            regenerated_leadin = regenerate_greedy_leadin(
                model,
                context,
                leadin_tokens,
                allowed_token_mask=allowed_mask,
            )
            greedy_recovered = recover_rank_span(
                model,
                context,
                regenerated_leadin,
                generated["forced_token_ids"],
                allowed_token_mask=allowed_mask,
            )
            greedy_segment_ranks = list(map(int, greedy_recovered["ranks"]))
            greedy_ranks.extend(greedy_segment_ranks)
            greedy_segment_exact = greedy_segment_ranks == rank_chunk
            greedy_exact_by_segment.append(greedy_segment_exact)
            segment_greedy_replay_seconds = (
                time.perf_counter() - operation_started
            )
            greedy_replay_seconds += segment_greedy_replay_seconds

        start = int(generated["forced_start"])
        stop = int(generated["forced_stop"])
        segment_text_replay_seconds = 0.0
        observed: List[int] = []
        observed_leadin: List[int] = []
        observed_forced: List[int] = []
        text_segment_ranks: List[int] = []
        text_segment_exact: Optional[bool] = None
        text_error: Optional[str] = None
        retokenized: Dict[str, object] = {
            "full_token_ids_match": None,
            "divergence": None,
            "boundary_rule": None,
        }
        if run_text_replay:
            operation_started = time.perf_counter()
            retokenized = retokenize_message(model, generated)
            observed = list(map(int, retokenized["retokenized_token_ids"]))
            observed_leadin = observed[:start]
            observed_forced = observed[start:stop]
            try:
                text_recovered = recover_rank_span(
                    model,
                    context,
                    observed_leadin,
                    observed_forced,
                    allowed_token_mask=allowed_mask,
                )
                text_segment_ranks = list(map(int, text_recovered["ranks"]))
            except Exception as exc:
                text_segment_ranks = []
                text_error = "{}: {}".format(type(exc).__name__, exc)
            text_ranks.extend(text_segment_ranks)
            text_segment_exact = text_segment_ranks == rank_chunk
            text_exact_by_segment.append(text_segment_exact)
            segment_text_replay_seconds = time.perf_counter() - operation_started
            text_replay_seconds += segment_text_replay_seconds

        role_mask = _role_mask(generated)
        if len(role_mask) != len(generated["full_token_ids"]):
            raise RevisionRunnerError("Token role mask length mismatch")
        segment_record = {
            "segment_index": segment_index,
            "expected_ranks": rank_chunk,
            "prompt": prompt,
            "context_token_ids": list(map(int, context)),
            "context_sha256": generated["context_sha256"],
            "leadin_token_ids": generated["leadin_token_ids"],
            "forced_token_ids": generated["forced_token_ids"],
            "tail_token_ids": generated["tail_token_ids"],
            "full_token_ids": generated["full_token_ids"],
            "token_role_mask": role_mask,
            "leadin_text": generated["leadin_text"],
            "forced_text": generated["forced_text"],
            "tail_text": generated["tail_text"],
            "full_text": generated["full_text"],
            "leadin_log_probabilities": generated["leadin_log_probabilities"],
            "forced_log_probabilities": generated["forced_log_probabilities"],
            "realized_ranks": generated["realized_ranks"],
            "greedy_token_ids": generated["greedy_token_ids"],
            "greedy_log_probabilities": generated[
                "greedy_log_probabilities"
            ],
            "quality_rank_ceiling": generated["quality_rank_ceiling"],
            "rank_B_token_ids": generated["rank_B_token_ids"],
            "rank_B_log_probabilities": generated[
                "rank_B_log_probabilities"
            ],
            "tail_log_probabilities": generated["tail_log_probabilities"],
            "tail_stop_reason": generated["tail_stop_reason"],
            "tail_censored": generated["tail_censored"],
            "forced_start": start,
            "forced_stop": stop,
            "timing": {
                "generation_seconds": segment_generation_seconds,
                "saved_token_id_replay_seconds": segment_saved_replay_seconds,
                "detokenized_text_retokenized_seconds": segment_text_replay_seconds,
                "greedy_leadin_regeneration_seconds": segment_greedy_replay_seconds,
            },
            "saved_token_id_replay": {
                "replay_mode": "saved_token_ids",
                "recovered_ranks": recovered_ranks,
                "exact_rank_replay": recovered_ranks == rank_chunk,
                "token_log_probabilities": recovered["token_log_probabilities"],
            },
            "text_retokenization_replay": {
                "replay_mode": "detokenized_text_retokenized",
                "executed": run_text_replay,
                "retokenized_token_ids": observed,
                "full_token_ids_match": retokenized["full_token_ids_match"],
                "token_divergence": retokenized["divergence"],
                "boundary_rule": retokenized["boundary_rule"],
                "observed_leadin_token_ids": observed_leadin,
                "observed_forced_token_ids": observed_forced,
                "recovered_ranks": text_segment_ranks,
                "exact_rank_replay": text_segment_exact,
                "error": text_error,
            },
            "greedy_leadin_replay": {
                "replay_mode": "greedy_leadin_regeneration",
                "executed": run_greedy_replay,
                "regenerated_leadin_token_ids": regenerated_leadin,
                "generated_leadin_token_ids": generated["leadin_token_ids"],
                "leadin_divergence": (
                    first_divergence(
                        generated["leadin_token_ids"], regenerated_leadin
                    )
                    if run_greedy_replay
                    else None
                ),
                "recovered_ranks": greedy_segment_ranks,
                "exact_rank_replay": greedy_segment_exact,
            },
        }
        segments.append(segment_record)
        leadin_logp.extend(map(float, generated["leadin_log_probabilities"]))
        forced_logp.extend(map(float, generated["forced_log_probabilities"]))
        tail_logp.extend(map(float, generated["tail_log_probabilities"]))

    decode_started = time.perf_counter()
    saved_decoded = decode_representation(model, representation, saved_ranks)
    saved_inverse_seconds = time.perf_counter() - decode_started
    text_decoded: Optional[Mapping[str, object]] = None
    text_inverse_seconds = 0.0
    if run_text_replay:
        decode_started = time.perf_counter()
        text_decoded = decode_representation(model, representation, text_ranks)
        text_inverse_seconds = time.perf_counter() - decode_started
    greedy_decoded: Optional[Mapping[str, object]] = None
    greedy_inverse_seconds = 0.0
    if run_greedy_replay:
        decode_started = time.perf_counter()
        greedy_decoded = decode_representation(model, representation, greedy_ranks)
        greedy_inverse_seconds = time.perf_counter() - decode_started
    saved_representation_exact = bool(
        all(saved_exact_by_segment)
        and saved_decoded["exact_representation_recovery"]
    )
    saved_exact = bool(
        all(saved_exact_by_segment) and saved_decoded["exact_payload_recovery"]
    )
    text_representation_exact: Optional[bool] = (
        bool(
            all(text_exact_by_segment)
            and text_decoded["exact_representation_recovery"]
        )
        if text_decoded is not None
        else None
    )
    text_exact: Optional[bool] = (
        bool(all(text_exact_by_segment) and text_decoded["exact_recovery"])
        if text_decoded is not None
        else None
    )
    greedy_representation_exact: Optional[bool] = (
        bool(
            all(greedy_exact_by_segment)
            and greedy_decoded["exact_representation_recovery"]
        )
        if greedy_decoded is not None
        else None
    )
    greedy_exact: Optional[bool] = (
        bool(all(greedy_exact_by_segment) and greedy_decoded["exact_payload_recovery"])
        if greedy_decoded is not None
        else None
    )
    failure = None
    if not saved_exact:
        if all(saved_exact_by_segment) and saved_representation_exact:
            failure = _payload_mismatch_failure(
                saved_decoded, task, "saved_token_ids", segments[0]
            )
        else:
            bad_index = next(
                (index for index, exact in enumerate(saved_exact_by_segment) if not exact),
                0,
            )
            bad = segments[bad_index]
            failure = diagnose_rank_failure(
                bad["expected_ranks"],
                bad["saved_token_id_replay"]["recovered_ranks"],
                bad["forced_token_ids"],
                bad["forced_token_ids"],
                bad["context_token_ids"],
                (bad["forced_start"], bad["forced_stop"]),
                "saved_token_id_rank_replay",
            )
            failure["segment_index"] = bad_index
            failure = _complete_failure_record(failure, task, "saved_token_ids")

    text_failure = None
    if run_text_replay and not text_exact:
        if all(text_exact_by_segment) and text_representation_exact:
            text_failure = _payload_mismatch_failure(
                text_decoded, task, "detokenized_text_retokenized", segments[0]
            )
        else:
            bad_index = next(
                (index for index, exact in enumerate(text_exact_by_segment) if not exact),
                0,
            )
            bad = segments[bad_index]
            text_failure = diagnose_rank_failure(
                bad["expected_ranks"],
                bad["text_retokenization_replay"]["recovered_ranks"],
                bad["forced_token_ids"],
                bad["text_retokenization_replay"]["observed_forced_token_ids"],
                bad["context_token_ids"],
                (bad["forced_start"], bad["forced_stop"]),
                "detokenized_text_retokenization",
            )
            text_failure["segment_index"] = bad_index
            text_failure = _complete_failure_record(
                text_failure, task, "detokenized_text_retokenized"
            )

    greedy_failure = None
    if run_greedy_replay and not greedy_exact:
        if all(greedy_exact_by_segment) and greedy_representation_exact:
            greedy_failure = _payload_mismatch_failure(
                greedy_decoded, task, "greedy_leadin_regeneration", segments[0]
            )
        else:
            bad_index = next(
                (index for index, exact in enumerate(greedy_exact_by_segment) if not exact),
                0,
            )
            bad = segments[bad_index]
            greedy_failure = diagnose_rank_failure(
                bad["expected_ranks"],
                bad["greedy_leadin_replay"]["recovered_ranks"],
                bad["forced_token_ids"],
                bad["forced_token_ids"],
                bad["context_token_ids"],
                (bad["forced_start"], bad["forced_stop"]),
                "greedy_leadin_regeneration",
            )
            greedy_failure["segment_index"] = bad_index
            greedy_failure = _complete_failure_record(
                greedy_failure, task, "greedy_leadin_regeneration"
            )

    forced_count = sum(len(segment["forced_token_ids"]) for segment in segments)
    full_count = sum(len(segment["full_token_ids"]) for segment in segments)
    source_bits, source_bits_estimand = representation_capacity_bits(
        representation
    )
    serialized_payload_bits = len(payload.payload_bytes) * 8
    encoding_seconds = (
        representation_seconds + filter_setup_seconds + generation_seconds
    )
    decoding_seconds = saved_replay_seconds + saved_inverse_seconds
    total_seconds = time.perf_counter() - started
    record = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_type": "rankcloak_trial",
        "work_id": task["work_id"],
        "trial_id": task["trial_id"],
        "evidence_status": task["evidence_status"],
        "study_phase": task["study_phase"],
        **_contract_record_fields(task),
        "model_id": task["model_id"],
        "payload_name": payload.payload_name,
        "payload_class": payload.payload_class,
        "payload_index": payload.payload_index,
        "payload_split": task["payload_split"],
        "payload_text": payload.payload_text,
        "payload_text_sha256": hashlib.sha256(payload.payload_bytes).hexdigest(),
        "original_payload_sha256": hashlib.sha256(payload.payload_bytes).hexdigest(),
        "artifact_bit_length": payload.artifact_bit_length,
        "serialized_payload_bits": serialized_payload_bits,
        "protocol_variant": task["protocol_variant"],
        "representation": {
            "name": representation.name,
            "expected_ranks": ranks,
            "metadata": representation.metadata,
            "representation_source_bits": source_bits,
            "representation_bit_estimand": source_bits_estimand,
        },
        "H_bits": source_bits,
        "H_source": source_bits_estimand,
        "representation_source_bits": source_bits,
        "alphabet_size_B": task.get("alphabet_size"),
        "prompt_id": task["prompt_id"],
        "prompt_category": task["prompt_category"],
        "language": task["language"],
        "segmented": segmented,
        "topic_schedule": task.get("topic_schedule"),
        "topic_rotation_rule": (
            "anchored_at_assigned_category; increment_category_modulo_6_per_segment; fixed_assigned_template_index"
            if task.get("topic_schedule") == "deterministic_six_category_rotation"
            else "single_assigned_prompt"
            if task.get("topic_schedule") == "single_assigned_prompt"
            else None
        ),
        "segment_size_ranks": segment_size if segmented else None,
        "leadin_tokens": leadin_tokens,
        "tail_policy": tail_policy,
        "token_filter": filter_name,
        "replay_modes": list(replay_modes),
        "allowed_token_mask": allowed_mask_metadata,
        "segments": segments,
        "token_role_masks": [segment["token_role_mask"] for segment in segments],
        "full_text": "\n\n".join(str(segment["full_text"]) for segment in segments),
        "forced_text": "\n\n".join(str(segment["forced_text"]) for segment in segments),
        "segment_count": len(segments),
        "forced_token_count": forced_count,
        "tail_token_count": sum(len(segment["tail_token_ids"]) for segment in segments),
        "leadin_token_count": sum(len(segment["leadin_token_ids"]) for segment in segments),
        "full_token_count": full_count,
        "effective_bits_per_full_token": (
            source_bits / full_count
            if source_bits is not None and full_count
            else None
        ),
        "effective_artifact_bits_per_full_token": (
            payload.artifact_bit_length / full_count if full_count else None
        ),
        "effective_serialized_bits_per_full_token": (
            serialized_payload_bits / full_count if full_count else None
        ),
        "cover_tokens_per_payload_display_byte": (
            full_count / len(representation.payload_bytes)
            if representation.payload_bytes
            else None
        ),
        "saved_token_id_replay": {
            "replay_mode": "saved_token_ids",
            "recovered_ranks": saved_ranks,
            "all_segment_ranks_exact": all(saved_exact_by_segment),
            "decoded": _decode_record(saved_decoded),
            "exact_representation_recovery": saved_representation_exact,
            "exact_payload_recovery": saved_exact,
            "exact_recovery": saved_exact,
            "failure": failure,
        },
        "text_retokenization_replay": {
            "replay_mode": "detokenized_text_retokenized",
            "executed": run_text_replay,
            "recovered_ranks": text_ranks,
            "all_segment_ranks_exact": (
                all(text_exact_by_segment) if run_text_replay else None
            ),
            "decoded": (
                _decode_record(text_decoded) if text_decoded is not None else None
            ),
            "exact_representation_recovery": text_representation_exact,
            "exact_payload_recovery": text_exact,
            "exact_recovery": text_exact,
            "failure": text_failure,
        },
        "greedy_leadin_replay": {
            "replay_mode": "greedy_leadin_regeneration",
            "executed": run_greedy_replay,
            "recovered_ranks": greedy_ranks,
            "all_segment_ranks_exact": (
                all(greedy_exact_by_segment) if run_greedy_replay else None
            ),
            "decoded": (
                _decode_record(greedy_decoded)
                if greedy_decoded is not None
                else None
            ),
            "exact_representation_recovery": greedy_representation_exact,
            "exact_payload_recovery": greedy_exact,
            "exact_recovery": greedy_exact,
            "failure": greedy_failure,
        },
        "quality": {
            "mean_forced_token_log_probability": _mean_or_none(forced_logp),
            "mean_tail_token_log_probability": _mean_or_none(tail_logp),
            "mean_leadin_token_log_probability": _mean_or_none(leadin_logp),
        },
        "timing": {
            "representation_seconds": representation_seconds,
            "filter_setup_seconds": filter_setup_seconds,
            "generation_seconds": generation_seconds,
            "saved_token_id_replay_seconds": saved_replay_seconds,
            "detokenized_text_retokenized_seconds": text_replay_seconds,
            "greedy_leadin_regeneration_seconds": greedy_replay_seconds,
            "saved_inverse_transcode_seconds": saved_inverse_seconds,
            "text_inverse_transcode_seconds": text_inverse_seconds,
            "greedy_inverse_transcode_seconds": greedy_inverse_seconds,
            "encoding_seconds": encoding_seconds,
            "supported_decoding_seconds": decoding_seconds,
            "total_seconds": total_seconds,
            "cover_tokens_per_generation_second": (
                full_count / generation_seconds if generation_seconds > 0 else None
            ),
            "representation_bits_per_encoding_second": (
                source_bits / encoding_seconds
                if source_bits is not None and encoding_seconds > 0
                else None
            ),
            "payload_bits_per_encoding_second": (
                payload.artifact_bit_length / encoding_seconds
                if encoding_seconds > 0
                else None
            ),
            "serialized_bits_per_encoding_second": (
                serialized_payload_bits / encoding_seconds
                if encoding_seconds > 0
                else None
            ),
            "forced_tokens_per_supported_decoding_second": (
                forced_count / decoding_seconds if decoding_seconds > 0 else None
            ),
            "process_peak_rss_bytes_at_trial_start": process_peak_rss_start,
            "process_peak_rss_bytes_at_trial_end": process_peak_rss_bytes(),
            "segmentation_tail_overhead_estimand": (
                "paired total-encoding-time contrasts across frozen segment/tail ablations"
            ),
        },
        "execution_seconds": total_seconds,
    }
    return _json_safe(record)  # type: ignore[return-value]



def _greedy_rewrite(
    model: Any,
    prompt_template: str,
    source_text: str,
    target_token_count: int,
    context_limit: int,
) -> Dict[str, object]:
    """Generate the frozen deterministic severe paraphrase condition."""

    prompt = str(prompt_template).format(text=str(source_text))
    context = make_context_token_ids(model, prompt)
    target = int(target_token_count)
    if len(context) + target > int(context_limit):
        raise RevisionRunnerError("Paraphrase prompt and output exceed context limit")
    evaluate_context(model, context)
    excluded = set(excluded_control_token_ids(model))
    token_ids: List[int] = []
    for _ in range(target):
        logits = np.asarray(get_last_logits(model), dtype=np.float64).copy()
        for token_id in excluded:
            if 0 <= token_id < logits.size:
                logits[token_id] = -np.inf
        finite = np.isfinite(logits)
        if not np.any(finite):
            raise RevisionRunnerError("No finite token remains for greedy paraphrase")
        candidates = np.flatnonzero(finite)
        best_value = np.max(logits[candidates])
        token_id = int(np.min(candidates[logits[candidates] == best_value]))
        token_ids.append(token_id)
        model.eval([token_id])
    return {
        "prompt": prompt,
        "context_token_ids": context,
        "token_ids": token_ids,
        "text": safe_detokenize(model, token_ids),
        "target_token_count": target,
        "generation_mode": "greedy_token_id_tiebreak",
    }


def _robustness_position_digest(
    source_trial_id: str, transformation_id: str
) -> Tuple[int, str]:
    seed_material = (str(source_trial_id) + str(transformation_id)).encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    return int.from_bytes(digest, "big", signed=False), digest.hex()


def _frozen_text_transform(
    text: str, source_trial_id: str, transformation_id: str
) -> Tuple[str, Optional[int], Optional[int], str]:
    """Apply the exact sha256-modulo eligible-position contract."""

    value = str(text)
    position_digest, digest_hex = _robustness_position_digest(
        source_trial_id, transformation_id
    )
    if transformation_id == "character_insertion":
        eligible = list(range(len(value)))
        if not eligible:
            return value, None, 0, digest_hex
        position = eligible[position_digest % len(eligible)]
        return value[:position] + "x" + value[position:], position, len(eligible), digest_hex
    if transformation_id == "character_deletion":
        eligible = [i for i, character in enumerate(value) if not character.isspace()]
        if not eligible:
            return value, None, 0, digest_hex
        position = eligible[position_digest % len(eligible)]
        return value[:position] + value[position + 1 :], position, len(eligible), digest_hex
    if transformation_id == "character_substitution":
        eligible = [i for i, character in enumerate(value) if character.isalnum()]
        if not eligible:
            return value, None, 0, digest_hex
        position = eligible[position_digest % len(eligible)]
        replacement = "x" if value[position] != "x" else "y"
        return (
            value[:position] + replacement + value[position + 1 :],
            position,
            len(eligible),
            digest_hex,
        )
    transformed = apply_transmission_transform(value, transformation_id)
    return transformed, None, None, digest_hex


def _frozen_token_transform(
    token_ids: Sequence[int], source_trial_id: str, transformation_id: str
) -> Tuple[List[int], Optional[int], Optional[int], str]:
    values = list(map(int, token_ids))
    position_digest, digest_hex = _robustness_position_digest(
        source_trial_id, transformation_id
    )
    if transformation_id == "token_deletion":
        eligible = list(range(1, len(values) - 1)) if len(values) > 2 else []
        if not eligible:
            return values, None, 0, digest_hex
        position = eligible[position_digest % len(eligible)]
        return values[:position] + values[position + 1 :], position, len(eligible), digest_hex
    if transformation_id == "truncation":
        remove = max(1, int(math.ceil(len(values) * 0.1))) if values else 0
        position = len(values) - remove if remove else None
        return values[:position] if position is not None else values, position, remove, digest_hex
    raise RevisionRunnerError(
        "Unsupported frozen token-space transformation: {}".format(
            transformation_id
        )
    )


def _canonicalize_mitigation_text(text: str) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def _source_record_sha256(source_record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(_json_safe(source_record))).hexdigest()


def execute_robustness_transform(
    model: Any,
    task: Mapping[str, object],
    source_record: Mapping[str, object],
    context_limit: int = 4096,
) -> Dict[str, object]:
    """Create one immutable Qwen paraphrase artifact without decoding it."""

    started = time.perf_counter()
    transformation = dict(task["transformation"])
    if str(task["model_id"]) != str(transformation["model_id"]):
        raise RevisionRunnerError("Paraphrase task is not assigned to its pinned model")
    if str(task["source_trial_id"]) != str(source_record.get("trial_id")):
        raise RevisionRunnerError("Paraphrase source record does not match its work unit")
    outputs: List[Dict[str, object]] = []
    for segment_index, source_segment in enumerate(source_record.get("segments", [])):
        source_text = str(source_segment["full_text"])
        source_token_count = len(source_segment["full_token_ids"])
        rewrite = _greedy_rewrite(
            model,
            str(transformation["prompt"]),
            source_text,
            source_token_count,
            context_limit,
        )
        outputs.append(
            {
                "segment_index": segment_index,
                "source_text_sha256": hashlib.sha256(
                    source_text.encode("utf-8")
                ).hexdigest(),
                "source_model_token_count_target": source_token_count,
                **rewrite,
            }
        )
    if not outputs:
        raise RevisionRunnerError("Paraphrase source contains no segments")
    return _json_safe(
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "record_type": "robustness_transform",
            "work_id": task["work_id"],
            "trial_id": task["trial_id"],
            "evidence_status": task["evidence_status"],
            "study_phase": task["study_phase"],
        **_contract_record_fields(task),
            "robustness_family": task["robustness_family"],
            "transformation_id": task["transformation_id"],
            "transformation_model_id": task["model_id"],
            "source_model_id": task["source_model_id"],
            "source_stage": task["source_stage"],
            "source_trial_id": task["source_trial_id"],
            "source_record_sha256": _source_record_sha256(source_record),
            "segment_outputs": outputs,
            "segment_count": len(outputs),
            "decode_performed": False,
            "execution_seconds": time.perf_counter() - started,
        }
    )  # type: ignore[return-value]


def execute_robustness_reference(
    task: Mapping[str, object],
    source_record: Mapping[str, object],
) -> Dict[str, object]:
    supported = dict(source_record.get("saved_token_id_replay", {}))
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_type": "robustness_reference",
        "work_id": task["work_id"],
        "trial_id": task["trial_id"],
        "evidence_status": task["evidence_status"],
        "study_phase": task["study_phase"],
        **_contract_record_fields(task),
        "robustness_family": task["robustness_family"],
        "model_id": task["model_id"],
        "source_model_id": task["source_model_id"],
        "source_stage": task["source_stage"],
        "source_trial_id": task["source_trial_id"],
        "source_record_sha256": _source_record_sha256(source_record),
        "payload_name": task["payload_name"],
        "replay_mode": task["replay_mode"],
        "transformation_id": task["transformation_id"],
        "decode_performed": False,
        "reference_field": "saved_token_id_replay.exact_recovery",
        "exact_recovery": supported.get("exact_recovery"),
        "failure": supported.get("failure"),
    }


def execute_robustness_decode(
    model: Any,
    task: Mapping[str, object],
    source_record: Mapping[str, object],
    payload: RevisionPayload,
    transformation_record: Optional[Mapping[str, object]] = None,
    mask_cache: Optional[MutableMapping[
        str, Tuple[Optional[np.ndarray], Dict[str, object]]
    ]] = None,
    output_dir: Optional[Path] = None,
    context_limit: int = 4096,
) -> Dict[str, object]:
    """Execute one frozen replay, transmission, mitigation, or mismatch decode."""

    started = time.perf_counter()
    mask_cache = mask_cache if mask_cache is not None else {}
    representation = bounded_representation(
        payload.payload_bytes, payload.payload_text, "hex_nibble"
    )
    expected_ranks = list(map(int, representation.ranks))
    filter_name = str(source_record.get("token_filter") or "none")
    allowed_mask, mask_metadata = _get_filter_mask(
        model,
        str(task["model_id"]),
        filter_name,
        mask_cache,
        output_dir,
    )
    same_model = str(task["model_id"]) == str(task["source_model_id"])
    family = str(task["robustness_family"])
    replay_mode = str(task["replay_mode"])
    transformation_id = str(task["transformation_id"])
    recovered_ranks: List[int] = []
    segment_outcomes: List[Dict[str, object]] = []
    failure: Optional[Dict[str, object]] = None
    transformation_record_sha256: Optional[str] = None
    paraphrase_by_segment: Dict[int, Mapping[str, object]] = {}
    if transformation_id == "paraphrase":
        if transformation_record is None:
            raise RevisionRunnerError(
                "Paraphrase decode requires its immutable transform artifact"
            )
        if str(transformation_record.get("work_id")) != str(
            task.get("transform_work_id")
        ):
            raise ArtifactIntegrityError("Paraphrase transform work ID mismatch")
        if str(transformation_record.get("source_trial_id")) != str(
            task["source_trial_id"]
        ):
            raise ArtifactIntegrityError("Paraphrase transform source mismatch")
        if str(transformation_record.get("transformation_model_id")) != str(
            task.get("transformation_model_id")
        ):
            raise ArtifactIntegrityError("Paraphrase transformation-model mismatch")
        transformation_record_sha256 = _source_record_sha256(transformation_record)
        for output in transformation_record.get("segment_outputs", []):
            segment_key = int(output["segment_index"])
            if segment_key in paraphrase_by_segment:
                raise ArtifactIntegrityError("Duplicate paraphrase segment output")
            paraphrase_by_segment[segment_key] = output

    source_segments = list(source_record.get("segments", []))
    for segment_index, source_segment in enumerate(source_segments):
        expected_segment_ranks = list(map(int, source_segment["expected_ranks"]))
        source_full_ids = list(map(int, source_segment["full_token_ids"]))
        source_full_text = str(source_segment["full_text"])
        start = int(source_segment["forced_start"])
        stop = int(source_segment["forced_stop"])
        prompt = dict(source_segment["prompt"])
        context = (
            list(map(int, source_segment["context_token_ids"]))
            if same_model
            else make_context_token_ids(model, str(prompt["prompt_text"]))
        )
        observed_full_ids: List[int]
        observed_text = source_full_text
        regenerated_leadin: List[int] = []
        resolved_edit_position: Optional[int] = None
        eligible_position_count: Optional[int] = None
        _, transformation_seed_sha256 = _robustness_position_digest(
            str(task["source_trial_id"]), transformation_id
        )

        if replay_mode == "greedy_leadin_regeneration":
            if not same_model:
                raise RevisionRunnerError("Greedy lead-in replay requires the source model")
            regenerated_leadin = regenerate_greedy_leadin(
                model,
                context,
                start,
                allowed_token_mask=allowed_mask,
            )
            observed_full_ids = (
                regenerated_leadin
                + list(map(int, source_segment["forced_token_ids"]))
                + list(map(int, source_segment["tail_token_ids"]))
            )
        elif transformation_id in {"token_deletion", "truncation"}:
            (
                observed_full_ids,
                resolved_edit_position,
                eligible_position_count,
                transformation_seed_sha256,
            ) = _frozen_token_transform(
                source_full_ids,
                str(task["source_trial_id"]),
                transformation_id,
            )
            observed_text = safe_detokenize(model, observed_full_ids)
        elif transformation_id == "paraphrase":
            if segment_index not in paraphrase_by_segment:
                raise ArtifactIntegrityError("Missing paraphrase segment output")
            rewrite = paraphrase_by_segment[segment_index]
            if str(rewrite.get("source_text_sha256")) != hashlib.sha256(
                source_full_text.encode("utf-8")
            ).hexdigest():
                raise ArtifactIntegrityError("Paraphrase segment source hash mismatch")
            observed_text = str(rewrite["text"])
            # Qwen token IDs are never interpreted by the source decoder.
            observed_full_ids = text_to_token_ids(model, observed_text)
        else:
            if replay_mode == "detokenized_text_retokenized" or family == "cross_model_mismatch":
                transformed_text = source_full_text
            else:
                (
                    transformed_text,
                    resolved_edit_position,
                    eligible_position_count,
                    transformation_seed_sha256,
                ) = _frozen_text_transform(
                    source_full_text,
                    str(task["source_trial_id"]),
                    transformation_id,
                )
            if family == "limited_mitigation":
                transformed_text = _canonicalize_mitigation_text(transformed_text)
            observed_text = transformed_text
            observed_full_ids = text_to_token_ids(model, transformed_text)

        observed_leadin = observed_full_ids[:start]
        observed_forced = observed_full_ids[start:stop]
        recovery_error = None
        try:
            recovered = recover_rank_span(
                model,
                context,
                observed_leadin,
                observed_forced,
                allowed_token_mask=allowed_mask,
            )
            segment_ranks = list(map(int, recovered["ranks"]))
        except Exception as exc:
            segment_ranks = []
            recovery_error = "{}: {}".format(type(exc).__name__, exc)
        recovered_ranks.extend(segment_ranks)
        segment_exact = segment_ranks == expected_segment_ranks
        source_checksum = hashlib.sha256(source_full_text.encode("utf-8")).hexdigest()
        observed_checksum = hashlib.sha256(observed_text.encode("utf-8")).hexdigest()
        segment_outcome = {
            "segment_index": segment_index,
            "prompt": prompt,
            "expected_ranks": expected_segment_ranks,
            "recovered_ranks": segment_ranks,
            "exact_rank_replay": segment_exact,
            "source_full_token_ids": source_full_ids,
            "observed_full_token_ids": observed_full_ids,
            "observed_leadin_token_ids": observed_leadin,
            "observed_forced_token_ids": observed_forced,
            "token_divergence": first_divergence(source_full_ids, observed_full_ids),
            "source_text": source_full_text,
            "observed_text": observed_text,
            "source_checksum_sha256": source_checksum,
            "observed_checksum_sha256": observed_checksum,
            "checksum_detected_change": source_checksum != observed_checksum,
            "boundary_start_offset": start,
            "boundary_end_offset": stop,
            "context_token_ids": context,
            "regenerated_leadin_token_ids": regenerated_leadin,
            "resolved_edit_position_zero_based": resolved_edit_position,
            "eligible_position_count": eligible_position_count,
            "transformation_seed_sha256": transformation_seed_sha256,
            "transformation_position_rule": (
                "sha256((source_trial_id || transformation_id).utf8)_modulo_eligible_positions_v1"
            ),
            "recovery_error": recovery_error,
        }
        segment_outcomes.append(segment_outcome)
        if not segment_exact and failure is None:
            raw_failure = diagnose_rank_failure(
                expected_segment_ranks,
                segment_ranks,
                list(map(int, source_segment["forced_token_ids"])),
                observed_forced,
                context,
                (start, stop),
                family,
            )
            raw_failure["segment_index"] = segment_index
            raw_failure["transformation_id"] = transformation_id
            failure = _complete_failure_record(raw_failure, task, replay_mode)

    decoded = decode_representation(model, representation, recovered_ranks)
    exact_recovery = bool(
        len(segment_outcomes) > 0
        and all(bool(row["exact_rank_replay"]) for row in segment_outcomes)
        and decoded["exact_recovery"]
    )
    return _json_safe(
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "record_type": "robustness_decode",
            "work_id": task["work_id"],
            "trial_id": task["trial_id"],
            "evidence_status": task["evidence_status"],
            "study_phase": task["study_phase"],
        **_contract_record_fields(task),
            "robustness_family": family,
            "model_id": task["model_id"],
            "source_model_id": task["source_model_id"],
            "source_stage": task["source_stage"],
            "source_trial_id": task["source_trial_id"],
            "source_record_sha256": _source_record_sha256(source_record),
            "payload_name": payload.payload_name,
            "payload_class": payload.payload_class,
            "payload_split": task["payload_split"],
            "replay_mode": replay_mode,
            "transformation_id": transformation_id,
            "same_model_decoder": same_model,
            "token_filter": filter_name,
            "allowed_token_mask": mask_metadata,
            "expected_ranks": expected_ranks,
            "recovered_ranks": recovered_ranks,
            "segment_outcomes": segment_outcomes,
            "decoded": _decode_record(decoded),
            "exact_recovery": exact_recovery,
            "failure": failure,
            "mitigation_pipeline": task.get("mitigation_pipeline"),
            "transform_work_id": task.get("transform_work_id"),
            "transformation_model_id": task.get("transformation_model_id"),
            "transformation_record_sha256": transformation_record_sha256,
            "execution_seconds": time.perf_counter() - started,
        }
    )  # type: ignore[return-value]

def execute_control_trial(
    model: Any,
    task: Mapping[str, object],
    source_record: Mapping[str, object],
    configs: Mapping[str, Mapping[str, object]],
    context_limit: int = 4096,
) -> Dict[str, object]:
    started = time.perf_counter()
    view = str(task["control_view"])
    target_field = "full_token_count" if view == "full_message" else "forced_token_count"
    target = int(source_record[target_field])
    seed = derive_control_seed(str(task["source_trial_id"]), view)
    generated = generate_length_matched_control(
        model,
        _prompt_text(task, configs),
        target,
        seed,
        temperature=float(task.get("temperature", CONTROL_TEMPERATURE)),
        top_p=float(task.get("top_p", CONTROL_TOP_P)),
        context_limit=context_limit,
    )
    return _json_safe(
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "record_type": "ordinary_control",
            "work_id": task["work_id"],
            "control_id": task["control_id"],
            "source_trial_id": task["source_trial_id"],
            "evidence_status": task["evidence_status"],
            "study_phase": task["study_phase"],
        **_contract_record_fields(task),
            "model_id": task["model_id"],
            "payload_name": task["payload_name"],
            "payload_class": task["payload_class"],
            "payload_split": task["payload_split"],
            "prompt_id": task["prompt_id"],
            "prompt_category": task["prompt_category"],
            "language": task["language"],
            "control_view": view,
            "source_length_field": target_field,
            "generation": generated,
            "full_text": generated["text"],
            "full_token_count": len(generated["token_ids"]),
            "execution_seconds": time.perf_counter() - started,
        }
    )  # type: ignore[return-value]


def execute_reference(task: Mapping[str, object]) -> Dict[str, object]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_type": "canonical_primary_reference",
        "work_id": task["work_id"],
        "trial_id": task["trial_id"],
        "source_trial_id": task["source_trial_id"],
        "evidence_status": task["evidence_status"],
        "study_phase": task["study_phase"],
        **_contract_record_fields(task),
        "model_id": task["model_id"],
        "payload_name": task["payload_name"],
        "ablation_factor": task.get("ablation_factor"),
        "ablation_level": task.get("ablation_level"),
        "generation_performed": False,
        "resolution_rule": "Join source_trial_id to immutable primary-stage output during analysis.",
    }


def _configured_tokenizer_identity(
    configs: Mapping[str, Mapping[str, object]], model_id: str
) -> Dict[str, object]:
    entry = _model_entry(configs, model_id)
    source = str(
        configs["models"].get("execution_policy", {}).get(
            "tokenizer_source", "embedded_gguf"
        )
    )
    return {
        "model_id": model_id,
        "model_family": entry.get("family"),
        "model_artifact_sha256": entry.get("artifact_sha256"),
        "tokenizer_id": "{}::{}::{}".format(
            entry.get("repo_id"), entry.get("filename"), source
        ),
        "tokenizer_revision": entry.get("revision"),
        "tokenizer_artifact_sha256": entry.get("artifact_sha256"),
        "tokenizer_source": source,
    }


def execute_condition_unavailable(
    task: Mapping[str, object],
    configs: Mapping[str, Mapping[str, object]],
    details: Mapping[str, object],
) -> Dict[str, object]:
    """Persist a completed, non-outcome row for a failed frozen feasibility gate."""

    record: Dict[str, object] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_type": "condition_unavailable",
        "work_id": task["work_id"],
        "trial_id": task.get("trial_id", task.get("control_id")),
        "evidence_status": task["evidence_status"],
        "study_phase": task["study_phase"],
        **_contract_record_fields(task),
        "work_kind": task["work_kind"],
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        **_configured_tokenizer_identity(configs, str(task["model_id"])),
        **dict(details),
    }
    for name in (
        "payload_name",
        "payload_class",
        "payload_index",
        "payload_split",
        "protocol_variant",
        "representation_id",
        "alphabet_size",
        "prompt_id",
        "prompt_category",
        "language",
        "segmented",
        "topic_schedule",
        "segment_size_ranks",
        "leadin_tokens",
        "tail_policy",
        "token_filter",
        "ablation_factor",
        "ablation_level",
    ):
        if name in task:
            record[name] = task[name]
    return _json_safe(record)  # type: ignore[return-value]


def _unavailability_root(source_record: Mapping[str, object]) -> Dict[str, object]:
    existing = source_record.get("dependency_root")
    if isinstance(existing, Mapping):
        return dict(existing)
    return {
        "work_id": source_record.get("work_id"),
        "trial_id": source_record.get("trial_id"),
        "record_type": source_record.get("record_type"),
        "reason_code": source_record.get("reason_code"),
        "reason": source_record.get("reason"),
        "model_id": source_record.get("model_id"),
        "tokenizer_id": source_record.get("tokenizer_id"),
        "tokenizer_revision": source_record.get("tokenizer_revision"),
        "tokenizer_artifact_sha256": source_record.get(
            "tokenizer_artifact_sha256"
        ),
        "safe_count": source_record.get("safe_count"),
        "stable_count": source_record.get("stable_count"),
    }


def execute_dependent_unavailable(
    task: Mapping[str, object],
    configs: Mapping[str, Mapping[str, object]],
    source_record: Mapping[str, object],
    dependency_role: str,
) -> Dict[str, object]:
    """Propagate an unavailable source without generating or decoding."""

    root = _unavailability_root(source_record)
    record: Dict[str, object] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "record_type": "dependent_unavailable",
        "work_id": task["work_id"],
        "trial_id": task.get("trial_id", task.get("control_id")),
        "control_id": task.get("control_id"),
        "source_trial_id": task.get("source_trial_id"),
        "source_record_sha256": _source_record_sha256(source_record),
        "evidence_status": task["evidence_status"],
        "study_phase": task["study_phase"],
        **_contract_record_fields(task),
        "work_kind": task["work_kind"],
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        "reason_code": "source_condition_unavailable",
        "reason": "Required {} is unavailable under the frozen feasibility gate.".format(
            dependency_role
        ),
        "dependency_role": dependency_role,
        "dependency_record_type": source_record.get("record_type"),
        "dependency_root": root,
        **_configured_tokenizer_identity(configs, str(task["model_id"])),
    }
    for name in (
        "payload_name",
        "payload_class",
        "payload_split",
        "protocol_variant",
        "prompt_id",
        "prompt_category",
        "language",
        "control_view",
        "ablation_factor",
        "ablation_level",
        "robustness_family",
        "source_model_id",
        "source_stage",
        "replay_mode",
        "transformation_id",
        "mitigation_pipeline",
        "transform_work_id",
        "transformation_model_id",
    ):
        if name in task:
            record[name] = task[name]
    return _json_safe(record)  # type: ignore[return-value]


def _record_is_unavailable(record: Mapping[str, object]) -> bool:
    return str(record.get("record_type")) in {
        "condition_unavailable",
        "dependent_unavailable",
    }


def append_jsonl_fsync(path: Path, row: Mapping[str, object]) -> None:
    """Append one canonical JSON row and make it durable before checkpointing."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactIntegrityError("Refusing append through symlink: {}".format(path))
    content = canonical_json_bytes(_json_safe(row)) + b"\n"
    with path.open("ab", buffering=0) as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows is not an execution target
            pass
        remaining = memoryview(content)
        while remaining:
            written = handle.write(remaining)
            if written is None or written <= 0:
                raise OSError("Short write while appending {}".format(path))
            remaining = remaining[written:]
        os.fsync(handle.fileno())


def load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(
                    "Invalid JSONL at {}:{}: {}".format(path, line_number, exc)
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactIntegrityError(
                    "JSONL row at {}:{} is not an object".format(path, line_number)
                )
            records.append(value)
    return records



def load_external_source_records(
    plan: Sequence[Mapping[str, object]],
    primary_results_root: Path,
    ablation_results_root: Path,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Load and content-address every immutable source result required by a plan."""

    roots = {
        "primary": Path(primary_results_root),
        "primary_v2": Path(primary_results_root),
        "ablation": Path(ablation_results_root),
        "ablation_v2": Path(ablation_results_root),
    }
    requirements = sorted(
        {
            (str(row["source_stage"]), str(row["source_model_id"]))
            for row in plan
            if row.get("robustness_family") is not None
        }
    )
    records_by_id: Dict[str, Dict[str, object]] = {}
    files: List[Dict[str, object]] = []
    for stage, source_model_id in requirements:
        if stage not in roots:
            raise RevisionRunnerError("No source-result root for stage {}".format(stage))
        run_dir = roots[stage] / source_model_id
        records_path = run_dir / "records.jsonl"
        identity_path = run_dir / "run_identity.json"
        if not records_path.is_file() or not identity_path.is_file():
            raise RevisionRunnerError(
                "Missing immutable {} source outputs for {} under {}".format(
                    stage, source_model_id, run_dir
                )
            )
        identity_value = json.loads(identity_path.read_text(encoding="utf-8"))
        if stage in {"primary_v2", "ablation_v2"}:
            if (
                identity_value.get("protocol_contract_revision")
                != PROTOCOL_CONTRACT_REVISION
                or identity_value.get("result_schema_revision")
                != RESULT_SCHEMA_REVISION
            ):
                raise ArtifactIntegrityError(
                    "Superseding source run identity lacks payload-aware contract"
                )
        for source_record in load_jsonl_records(records_path):
            if stage in {"primary_v2", "ablation_v2"} and (
                source_record.get("protocol_contract_revision")
                != PROTOCOL_CONTRACT_REVISION
                or source_record.get("result_schema_revision")
                != RESULT_SCHEMA_REVISION
            ):
                raise ArtifactIntegrityError(
                    "Superseding source record lacks payload-aware contract"
                )
            if (
                source_record.get("execution_status") == "completed"
                and source_record.get("record_type")
                in {
                    "rankcloak_trial",
                    "condition_unavailable",
                    "dependent_unavailable",
                }
            ):
                source_id = str(source_record["trial_id"])
                if source_id in records_by_id:
                    raise ArtifactIntegrityError(
                        "Duplicate external source trial ID: {}".format(source_id)
                    )
                records_by_id[source_id] = source_record
        files.extend(
            [
                {
                    "stage": stage,
                    "source_model_id": source_model_id,
                    "role": "records",
                    "path": str(records_path.resolve()),
                    "size_bytes": records_path.stat().st_size,
                    "sha256": file_sha256(records_path),
                },
                {
                    "stage": stage,
                    "source_model_id": source_model_id,
                    "role": "run_identity",
                    "path": str(identity_path.resolve()),
                    "size_bytes": identity_path.stat().st_size,
                    "sha256": file_sha256(identity_path),
                },
            ]
        )
    required_ids = {
        str(row["source_trial_id"])
        for row in plan
        if row.get("robustness_family") is not None
    }
    missing = sorted(required_ids - set(records_by_id))
    if missing:
        raise RevisionRunnerError(
            "Missing {} required robustness source records; first: {}".format(
                len(missing), ", ".join(missing[:3])
            )
        )
    selected = {source_id: records_by_id[source_id] for source_id in required_ids}
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "manifest_type": "robustness_external_source_results",
        "required_source_record_count": len(selected),
        "required_source_trial_ids_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(selected))
        ).hexdigest(),
        "files": files,
        "files_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }
    return selected, manifest

def load_external_transformation_records(
    plan: Sequence[Mapping[str, object]],
    robustness_results_root: Path,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Load content-addressed Qwen paraphrases needed by decoder shards."""

    required_ids = {
        str(row["transform_work_id"])
        for row in plan
        if row.get("transform_work_id") is not None
    }
    local_ids = {
        str(row["work_id"])
        for row in plan
        if row.get("work_kind") == "robustness_transform"
    }
    required_external = required_ids - local_ids
    if not required_external:
        return {}, {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "manifest_type": "robustness_external_transform_results",
            "required_transform_record_count": 0,
            "files": [],
            "files_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        }
    transform_model_ids = {
        str(row["transformation_model_id"])
        for row in plan
        if str(row.get("transform_work_id")) in required_external
    }
    if len(transform_model_ids) != 1:
        raise RevisionRunnerError(
            "External paraphrase dependencies must use exactly one pinned model"
        )
    transformation_model_id = next(iter(transform_model_ids))
    run_dir = Path(robustness_results_root) / transformation_model_id
    records_path = run_dir / "records.jsonl"
    identity_path = run_dir / "run_identity.json"
    if not records_path.is_file() or not identity_path.is_file():
        raise RevisionRunnerError(
            "Run the {} robustness shard first; its paraphrase artifacts are missing".format(
                transformation_model_id
            )
        )
    requires_payload_fidelity_v2 = any(
        row.get("protocol_contract_revision") == PROTOCOL_CONTRACT_REVISION
        for row in plan
    )
    identity_value = json.loads(identity_path.read_text(encoding="utf-8"))
    if requires_payload_fidelity_v2 and (
        identity_value.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or identity_value.get("result_schema_revision")
        != RESULT_SCHEMA_REVISION
    ):
        raise ArtifactIntegrityError(
            "Superseding transformation identity lacks payload-aware contract"
        )
    available: Dict[str, Dict[str, object]] = {}
    for record in load_jsonl_records(records_path):
        if requires_payload_fidelity_v2 and (
            record.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or record.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        ):
            raise ArtifactIntegrityError(
                "Superseding transformation record lacks payload-aware contract"
            )
        if (
            record.get("execution_status") == "completed"
            and record.get("record_type") == "robustness_transform"
        ):
            work_id = str(record["work_id"])
            if work_id in available:
                raise ArtifactIntegrityError(
                    "Duplicate external transform work ID: {}".format(work_id)
                )
            available[work_id] = record
    missing = sorted(required_external - set(available))
    if missing:
        raise RevisionRunnerError(
            "Missing {} paraphrase artifacts; first: {}".format(
                len(missing), ", ".join(missing[:3])
            )
        )
    selected = {work_id: available[work_id] for work_id in required_external}
    files = [
        {
            "transformation_model_id": transformation_model_id,
            "role": "records",
            "path": str(records_path.resolve()),
            "size_bytes": records_path.stat().st_size,
            "sha256": file_sha256(records_path),
        },
        {
            "transformation_model_id": transformation_model_id,
            "role": "run_identity",
            "path": str(identity_path.resolve()),
            "size_bytes": identity_path.stat().st_size,
            "sha256": file_sha256(identity_path),
        },
    ]
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "manifest_type": "robustness_external_transform_results",
        "required_transform_record_count": len(selected),
        "required_transform_work_ids_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(selected))
        ).hexdigest(),
        "files": files,
        "files_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }
    return selected, manifest


def reconcile_checkpoint_from_records(
    checkpoint_path: Path,
    planned_work_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Repair only the safe append-before-checkpoint crash window."""

    checkpoint = load_checkpoint(checkpoint_path)
    planned = set(map(str, planned_work_ids))
    completed_rows: Dict[str, Mapping[str, object]] = {}
    failed_rows: Dict[str, Mapping[str, object]] = {}
    attempts: Counter = Counter()
    seen_attempts: set = set()
    for record in records:
        work_id = str(record.get("work_id"))
        if work_id not in planned:
            raise ArtifactIntegrityError("Result contains unplanned work ID: {}".format(work_id))
        attempt_index = int(record.get("attempt_index", 1))
        attempt_key = (work_id, attempt_index)
        if attempt_key in seen_attempts:
            raise ArtifactIntegrityError(
                "Duplicate durable attempt {} for {}".format(attempt_index, work_id)
            )
        seen_attempts.add(attempt_key)
        attempts[work_id] = max(attempts[work_id], attempt_index)
        if record.get("execution_status") == "completed":
            if work_id in completed_rows:
                raise ArtifactIntegrityError(
                    "Multiple durable completions for {}".format(work_id)
                )
            completed_rows[work_id] = record
        elif record.get("execution_status") == "failed":
            failed_rows[work_id] = record
        else:
            raise ArtifactIntegrityError("Result row has invalid execution_status")
    checkpoint_completed = set(map(str, checkpoint["completed_trial_ids"]))
    missing_rows = checkpoint_completed - set(completed_rows)
    if missing_rows:
        raise ArtifactIntegrityError(
            "Checkpoint marks completed work without durable rows: {}".format(
                ", ".join(sorted(missing_rows))
            )
        )
    completed = set(completed_rows)
    failed = (set(failed_rows) - completed)
    checkpoint["completed_trial_ids"] = sorted(completed)
    checkpoint["failed_trial_ids"] = sorted(failed)
    checkpoint["failure_details"] = {
        work_id: dict(failed_rows[work_id].get("error", {})) for work_id in sorted(failed)
    }
    checkpoint["attempt_counts"] = {
        work_id: int(count) for work_id, count in sorted(attempts.items())
    }
    checkpoint["updated_at"] = utc_now()
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def run_work_plan(
    model: Any,
    plan: Sequence[Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
    configs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    context_limit: int = 4096,
    max_pending: Optional[int] = None,
    external_source_records: Optional[Mapping[str, Mapping[str, object]]] = None,
    external_transformation_records: Optional[
        Mapping[str, Mapping[str, object]]
    ] = None,
) -> Dict[str, object]:
    """Run pending ordered work, durably appending each attempt before checkpoint."""

    output_dir = Path(output_dir)
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    planned_ids = [str(row["work_id"]) for row in plan]
    records = load_jsonl_records(records_path)
    checkpoint = reconcile_checkpoint_from_records(checkpoint_path, planned_ids, records)
    pending = pending_trial_ids(planned_ids, checkpoint)
    if max_pending is not None:
        pending = pending[: max(0, int(max_pending))]
    plan_by_id = {str(row["work_id"]): row for row in plan}
    payload_by_name = {payload.payload_name: payload for payload in payloads}
    source_records: Dict[str, Mapping[str, object]] = dict(
        external_source_records or {}
    )
    source_records.update(
        {
            str(record["work_id"]): record
            for record in records
            if record.get("execution_status") == "completed"
            and record.get("record_type")
            in {
                "rankcloak_trial",
                "condition_unavailable",
                "dependent_unavailable",
            }
        }
    )
    transformation_records: Dict[str, Mapping[str, object]] = dict(
        external_transformation_records or {}
    )
    transformation_records.update(
        {
            str(record["work_id"]): record
            for record in records
            if record.get("execution_status") == "completed"
            and record.get("record_type") == "robustness_transform"
        }
    )
    mask_cache: Dict[
        str, Tuple[Optional[np.ndarray], Dict[str, object]]
    ] = {}
    consecutive_errors = 0
    for work_id in pending:
        task = plan_by_id[work_id]
        checkpoint = load_checkpoint(checkpoint_path)
        attempt_index = int(checkpoint["attempt_counts"].get(work_id, 0)) + 1
        try:
            if task["work_kind"] == "rankcloak":
                payload = payload_by_name[str(task["payload_name"])]
                try:
                    record = execute_rankcloak_trial(
                        model,
                        task,
                        payload,
                        configs,
                        mask_cache=mask_cache,
                        output_dir=output_dir,
                        context_limit=context_limit,
                    )
                except ConditionUnavailable as unavailable:
                    record = execute_condition_unavailable(
                        task, configs, unavailable.details
                    )
            elif task["work_kind"] == "control":
                source_id = str(task["source_trial_id"])
                if source_id not in source_records:
                    raise RevisionRunnerError(
                        "Control source result is unavailable: {}".format(source_id)
                    )
                source_record = source_records[source_id]
                if _record_is_unavailable(source_record):
                    record = execute_dependent_unavailable(
                        task, configs, source_record, "rankcloak control source"
                    )
                else:
                    record = execute_control_trial(
                        model,
                        task,
                        source_record,
                        configs,
                        context_limit=context_limit,
                    )
            elif task["work_kind"] == "robustness_transform":
                source_id = str(task["source_trial_id"])
                if source_id not in source_records:
                    raise RevisionRunnerError(
                        "Robustness transform source is unavailable: {}".format(
                            source_id
                        )
                    )
                source_record = source_records[source_id]
                if _record_is_unavailable(source_record):
                    record = execute_dependent_unavailable(
                        task, configs, source_record, "paraphrase source"
                    )
                else:
                    record = execute_robustness_transform(
                        model,
                        task,
                        source_record,
                        context_limit=context_limit,
                    )
            elif task["work_kind"] == "robustness_decode":
                source_id = str(task["source_trial_id"])
                if source_id not in source_records:
                    raise RevisionRunnerError(
                        "Robustness source result is unavailable: {}".format(source_id)
                    )
                payload = payload_by_name[str(task["payload_name"])]
                transform_id = task.get("transform_work_id")
                transformation_record = (
                    transformation_records.get(str(transform_id))
                    if transform_id is not None
                    else None
                )
                source_record = source_records[source_id]
                dependency = (
                    source_record
                    if _record_is_unavailable(source_record)
                    else transformation_record
                    if transformation_record is not None
                    and _record_is_unavailable(transformation_record)
                    else None
                )
                if dependency is not None:
                    record = execute_dependent_unavailable(
                        task, configs, dependency, "robustness decode input"
                    )
                else:
                    record = execute_robustness_decode(
                        model,
                        task,
                        source_record,
                        payload,
                        transformation_record=transformation_record,
                        mask_cache=mask_cache,
                        output_dir=output_dir,
                        context_limit=context_limit,
                    )
            elif task["work_kind"] == "reference":
                if task.get("robustness_family") is not None:
                    source_id = str(task["source_trial_id"])
                    if source_id not in source_records:
                        raise RevisionRunnerError(
                            "Robustness reference source is unavailable: {}".format(
                                source_id
                            )
                        )
                    source_record = source_records[source_id]
                    if _record_is_unavailable(source_record):
                        record = execute_dependent_unavailable(
                            task, configs, source_record, "robustness reference source"
                        )
                    else:
                        record = execute_robustness_reference(task, source_record)
                else:
                    record = execute_reference(task)
            else:
                raise RevisionRunnerError("Unknown work kind: {}".format(task["work_kind"]))
            record.update(
                {
                    "execution_status": "completed",
                    "attempt_index": attempt_index,
                    "completed_at": utc_now(),
                }
            )
            append_jsonl_fsync(records_path, record)
            record_checkpoint_result(checkpoint_path, work_id, "completed")
            if record.get("record_type") in {
                "rankcloak_trial",
                "condition_unavailable",
                "dependent_unavailable",
            } and task["work_kind"] == "rankcloak":
                source_records[work_id] = record
            elif record.get("record_type") in {
                "robustness_transform",
                "dependent_unavailable",
            } and task["work_kind"] == "robustness_transform":
                transformation_records[work_id] = record
            consecutive_errors = 0
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_record = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "record_type": "execution_failure",
                "work_id": work_id,
                "work_kind": task["work_kind"],
                "model_id": task["model_id"],
                "evidence_status": task["evidence_status"],
                "study_phase": task["study_phase"],
                **_contract_record_fields(task),
                "execution_status": "failed",
                "attempt_index": attempt_index,
                "failed_at": utc_now(),
                "error": error,
            }
            append_jsonl_fsync(records_path, failure_record)
            record_checkpoint_result(checkpoint_path, work_id, "failed", error)
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise RevisionRunnerError(
                    "Stopped after three consecutive execution errors; inspect records.jsonl"
                ) from exc
    final_checkpoint = load_checkpoint(checkpoint_path)
    return {
        "planned": len(plan),
        "completed": len(final_checkpoint["completed_trial_ids"]),
        "failed_current": len(final_checkpoint["failed_trial_ids"]),
        "remaining": len(plan) - len(final_checkpoint["completed_trial_ids"]),
        "records_path": str(records_path),
        "checkpoint_path": str(checkpoint_path),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def process_peak_rss_bytes() -> Optional[int]:
    """Return the auditable OS high-water RSS for this process."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return value * multiplier


def _process_current_rss_bytes() -> Optional[int]:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def query_selected_gpu_memory_mib(gpu_uuid: str) -> Optional[int]:
    command = [
        "nvidia-smi",
        "--id={}".format(gpu_uuid),
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        return int(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


class RuntimeMemoryProfiler:
    """Poll process RSS and selected-device total VRAM during a model shard."""

    def __init__(self, gpu_uuid: Optional[str], interval_seconds: float = 1.0):
        self.gpu_uuid = str(gpu_uuid) if gpu_uuid else None
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[str] = None
        self._samples = 0
        self._rss_samples = 0
        self._gpu_samples = 0
        self._peak_rss = 0
        self._peak_gpu_mib = 0
        self._initial_gpu_mib: Optional[int] = None
        self._final_gpu_mib: Optional[int] = None

    def _sample(self) -> None:
        rss = _process_current_rss_bytes()
        gpu_mib = (
            query_selected_gpu_memory_mib(self.gpu_uuid)
            if self.gpu_uuid is not None
            else None
        )
        self._samples += 1
        if rss is not None:
            self._rss_samples += 1
            self._peak_rss = max(self._peak_rss, rss)
        if gpu_mib is not None:
            self._gpu_samples += 1
            if self._initial_gpu_mib is None:
                self._initial_gpu_mib = gpu_mib
            self._final_gpu_mib = gpu_mib
            self._peak_gpu_mib = max(self._peak_gpu_mib, gpu_mib)

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RevisionRunnerError("Memory profiler has already started")
        self._started_at = utc_now()
        self._sample()
        self._thread = threading.Thread(
            target=self._loop, name="rankcloak-memory-profiler", daemon=True
        )
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
            self._thread = None
        self._sample()
        return {
            "event": "memory_profile",
            "at": utc_now(),
            "started_at": self._started_at,
            "poll_interval_seconds": self.interval_seconds,
            "sample_count": self._samples,
            "process_rss_sample_count": self._rss_samples,
            "process_peak_rss_bytes_sampled": (
                self._peak_rss if self._rss_samples else None
            ),
            "process_peak_rss_bytes_os_high_water": process_peak_rss_bytes(),
            "selected_gpu_uuid": self.gpu_uuid,
            "selected_gpu_sample_count": self._gpu_samples,
            "selected_gpu_initial_used_memory_mib": self._initial_gpu_mib,
            "selected_gpu_final_used_memory_mib": self._final_gpu_mib,
            "selected_gpu_peak_used_memory_mib_sampled": (
                self._peak_gpu_mib if self._gpu_samples else None
            ),
            "gpu_measurement_scope": (
                "total selected-device memory.used; includes any co-tenant processes; one-second polling is not a kernel-exact peak"
                if self.gpu_uuid is not None
                else "cpu_execution_no_gpu_sampling"
            ),
        }


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
        if len(values) != 5:
            continue
        rows.append(
            {
                "uuid": values[0],
                "name": values[1],
                "pci_bus_id": values[2],
                "memory_total_mib": int(values[3]),
                "driver_version": values[4],
            }
        )
    return rows


def pin_gpu_by_uuid(gpu_uuid: str) -> Dict[str, object]:
    inventory = query_gpu_inventory()
    matches = [row for row in inventory if row["uuid"] == str(gpu_uuid)]
    if len(matches) != 1:
        raise RevisionRunnerError(
            "GPU UUID {!r} did not resolve exactly once; available: {}".format(
                gpu_uuid, ", ".join(str(row["uuid"]) for row in inventory) or "none"
            )
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    return matches[0]


def verify_config_manifest(config_dir: Path) -> Dict[str, object]:
    manifest_path = Path(config_dir) / "config_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("Cannot load frozen config manifest") from exc
    report = verify_directory_manifest(
        Path(config_dir), manifest, require_no_extra_files=True
    )
    if report["status"] != "ok":
        raise ArtifactIntegrityError(
            "Frozen config manifest failed: {}".format("; ".join(report["errors"]))
        )
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "files_sha256": manifest["files_sha256"],
        "verified_file_count": report["verified_file_count"],
    }


def verify_payload_corpus(payloads: Sequence[RevisionPayload]) -> Dict[str, object]:
    report = validate_revision_corpus(payloads, expected_sha256=REVISION_CORPUS_SHA256)
    if report["status"] != "ok":
        raise ArtifactIntegrityError(
            "Revision payload corpus failed: {}".format("; ".join(report["errors"]))
        )
    return report


def _source_manifest(project_root: Path) -> Dict[str, object]:
    relative_paths = (
        "rankcloak/revision_runner.py",
        "rankcloak/revision_config.py",
        "rankcloak/revision_payloads.py",
        "rankcloak/revision_artifacts.py",
        "rankcloak/revision_protocol.py",
        "rankcloak/model_io.py",
        "rankcloak/rank_codec.py",
        "rankcloak/token_filters.py",
        "rankcloak/segmented_protocol.py",
        "scripts/run_revision_matrix.py",
        "pyproject.toml",
    )
    files = []
    for relative_path in relative_paths:
        path = Path(project_root) / relative_path
        if not path.is_file():
            raise ArtifactIntegrityError("Required source file missing: {}".format(relative_path))
        files.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *relative_paths],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        head = None
        status = []
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "manifest_type": "revision_runner_source",
        "git_head": head,
        "selected_source_status": status,
        "files": files,
        "files_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _runtime_manifest() -> Dict[str, object]:
    deterministic_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_LAUNCH_BLOCKING",
            "GGML_CUDA_DISABLE_GRAPHS",
            "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F",
            "CUBLAS_WORKSPACE_CONFIG",
        )
    }
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "rankcloak",
                "llama-cpp-python",
                "numpy",
                "cryptography",
            )
        },
        "llama_cpp_backend": _llama_cpp_backend_manifest(),
        "deterministic_environment": deterministic_environment,
        "control_sampler": "numpy_pcg64_serial_top_p_v1_token_id_tiebreak",
    }


def _llama_cpp_backend_manifest() -> Dict[str, object]:
    """Record the exact installed backend build without loading a GGUF."""

    preload_pip_cuda_libraries()
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as llama_cpp_api

        raw = llama_cpp_api.llama_print_system_info()
        system_info = (
            raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes)
            else str(raw)
        )
        return {
            "available": True,
            "python_package_version": getattr(llama_cpp, "__version__", None),
            "system_info": system_info,
            "gpu_offload_supported": bool(
                llama_cpp_api.llama_supports_gpu_offload()
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }


def _hardware_manifest(selected_gpu_uuid: Optional[str]) -> Dict[str, object]:
    inventory = query_gpu_inventory()
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "selected_gpu_uuid": selected_gpu_uuid,
        "gpu_inventory": inventory,
    }


def _model_artifact_manifest(
    configs: Mapping[str, Mapping[str, object]],
    model_id: str,
    project_root: Path,
) -> Dict[str, object]:
    entry = _model_entry(configs, model_id)
    verification = verify_model_artifact_pins(
        {"models": [entry]}, project_root=project_root, verify_sha256=True
    )
    if verification["status"] != "ok":
        raise ArtifactIntegrityError(
            "Pinned model artifact failed: {}".format("; ".join(verification["errors"]))
        )
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "configured_model": entry,
        "execution_policy": dict(
            configs["models"].get("execution_policy", {})
        ),
        "verification": verification["records"][0],
    }


def configure_deterministic_backend_environment(n_gpu_layers: int) -> None:
    """Freeze the backend environment before importing llama.cpp."""

    if int(n_gpu_layers) == 0:
        return
    values = {
        "CUDA_LAUNCH_BLOCKING": "1",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_CUDA_DISABLE_FUSION": "1",
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in values.items():
        os.environ.setdefault(name, value)


def prepare_run_artifacts(
    plan: Sequence[Mapping[str, object]],
    payloads: Sequence[RevisionPayload],
    configs: Mapping[str, Mapping[str, object]],
    stage: str,
    model_id: str,
    output_dir: Path,
    config_dir: Path = DEFAULT_REVISION_CONFIG_DIR,
    project_root: Path = PROJECT_ROOT,
    context_limit: int = 4096,
    gpu_uuid: Optional[str] = None,
    n_gpu_layers: int = -1,
    resume: bool = False,
    external_source_manifest: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Verify inputs and freeze all scientific identities before model loading."""

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists() and not resume:
        raise RevisionRunnerError(
            "Output already has a checkpoint; pass --resume after verifying its identity"
        )
    config_verification = verify_config_manifest(Path(config_dir))
    corpus_verification = verify_payload_corpus(payloads)
    model_manifest = _model_artifact_manifest(configs, model_id, Path(project_root))
    if stage in SUPERSEDING_BASE_STAGE:
        invalid_contract_rows = [
            str(row.get("work_id"))
            for row in plan
            if row.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        ]
        if invalid_contract_rows:
            raise RevisionRunnerError(
                "Superseding plan lacks its payload-aware contract; first: {}".format(
                    invalid_contract_rows[0]
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "manifest_type": "revision_v1_public_payload_corpus",
        "corpus_id": REVISION_CORPUS_ID,
        "corpus_sha256": corpus_verification["corpus_sha256"],
        "payload_count": len(payloads),
        "records": revision_payload_records(payloads, include_payload_text=True),
    }
    write_immutable_json(output_dir / "payload_manifest.json", payload_manifest)
    write_immutable_jsonl(output_dir / "plan.jsonl", plan)
    write_immutable_json(output_dir / "model_manifest.json", model_manifest)
    write_immutable_json(output_dir / "source_manifest.json", _source_manifest(Path(project_root)))
    write_immutable_json(output_dir / "runtime_manifest.json", _runtime_manifest())
    write_immutable_json(output_dir / "hardware_manifest.json", _hardware_manifest(gpu_uuid))
    if external_source_manifest is not None:
        write_immutable_json(
            output_dir / "input_results_manifest.json",
            external_source_manifest,
        )

    planned_ids = [str(row["work_id"]) for row in plan]
    scientific_args = [
        "stage={}".format(stage),
        "model_id={}".format(model_id),
        "evidence_status={}".format(
            next(iter({str(row["evidence_status"]) for row in plan}), _stage_evidence_status(stage))
        ),
        "context_limit={}".format(int(context_limit)),
        "gpu_uuid={}".format(gpu_uuid or "cpu"),
        "n_gpu_layers={}".format(int(n_gpu_layers)),
        "source_manifest_sha256={}".format(
            file_sha256(output_dir / "source_manifest.json")
        ),
        "runtime_manifest_sha256={}".format(
            file_sha256(output_dir / "runtime_manifest.json")
        ),
        "hardware_manifest_sha256={}".format(
            file_sha256(output_dir / "hardware_manifest.json")
        ),
        "input_results_manifest_sha256={}".format(
            file_sha256(output_dir / "input_results_manifest.json")
            if external_source_manifest is not None
            else "none"
        ),
    ]
    if stage in SUPERSEDING_BASE_STAGE:
        scientific_args.extend(
            [
                "protocol_contract_revision={}".format(
                    PROTOCOL_CONTRACT_REVISION
                ),
                "result_schema_revision={}".format(RESULT_SCHEMA_REVISION),
            ]
        )
    identity = build_run_identity_manifest(
        study_id="{}/{}/{}".format(REVISION_CORPUS_ID, stage, model_id),
        config_manifest_sha256=str(config_verification["sha256"]),
        payload_manifest_sha256=file_sha256(output_dir / "payload_manifest.json"),
        planned_trial_ids=planned_ids,
        model_artifacts=[model_manifest],
        command_line_args=scientific_args,
    )
    if stage in SUPERSEDING_BASE_STAGE:
        identity["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
        identity["result_schema_revision"] = RESULT_SCHEMA_REVISION
        identity.pop("run_identity_sha256", None)
        identity["run_identity_sha256"] = hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest()
    write_immutable_json(output_dir / "run_identity.json", identity)
    checkpoint = initialize_checkpoint(
        checkpoint_path,
        study_id=str(identity["study_id"]),
        config_manifest_sha256=str(config_verification["sha256"]),
        planned_trial_ids=planned_ids,
    )
    return {
        "config_verification": config_verification,
        "corpus_verification": corpus_verification,
        "model_manifest": model_manifest,
        "run_identity": identity,
        "checkpoint": checkpoint,
    }


def load_one_pinned_model(
    configs: Mapping[str, Mapping[str, object]],
    model_id: str,
    project_root: Path = PROJECT_ROOT,
    context_limit: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Any, float]:
    """Load exactly the selected model; callers must release it before another."""

    entry = _model_entry(configs, model_id)
    model_path = Path(project_root) / str(entry["relative_path"])
    started = time.perf_counter()
    model = load_llama_cpp_model(
        model_path=model_path,
        n_ctx=int(context_limit),
        n_threads=n_threads,
        n_gpu_layers=int(n_gpu_layers),
        logits_all=True,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    try:
        setattr(model, "rankcloak_revision_model_id", model_id)
    except Exception:
        pass
    return model, elapsed


def _default_output_dir(stage: str, model_id: str, evidence_status: str) -> Path:
    label = (
        stage
        if stage in {"smoke", "smoke_v2"}
        else "limited"
        if evidence_status == EVIDENCE_LIMITED
        else stage
    )
    return PROJECT_ROOT / "results" / "revision_v1" / label / model_id


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--model", dest="model_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--context", dest="context_limit", type=int, default=4096)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_REVISION_CONFIG_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--primary-results-root",
        type=Path,
        help="Root containing one immutable primary result directory per model.",
    )
    parser.add_argument(
        "--ablation-results-root",
        type=Path,
        help="Root containing one immutable ablation result directory per model.",
    )
    parser.add_argument(
        "--robustness-results-root",
        type=Path,
        help="Root containing robustness shards; Qwen paraphrases must exist first.",
    )
    parser.add_argument("--verbose-model", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise RevisionRunnerError("--limit must be positive")
    if args.context_limit <= 0:
        raise RevisionRunnerError("--context must be positive")
    configs = load_revision_config_set(args.config_dir)
    payloads = _default_payloads()
    verify_config_manifest(args.config_dir)
    verify_payload_corpus(payloads)
    full_plan = build_stage_plan(args.stage, model_id=args.model_id, configs=configs, payloads=payloads)
    plan = full_plan
    if args.limit is not None and args.limit < len(plan):
        plan = relabel_limited_plan(plan[: args.limit])
    summary = plan_summary(plan)
    summary.update(
        {
            "stage": args.stage,
            "model_id": args.model_id,
            "dry_run": bool(args.dry_run),
            "full_unlimited_work_units": len(full_plan),
        }
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.model_id is None:
        raise RevisionRunnerError("Execution requires --model so only one GGUF is loaded")
    if args.n_gpu_layers != 0 and not args.gpu_uuid:
        raise RevisionRunnerError("GPU execution requires an exact --gpu-uuid")
    if args.gpu_uuid:
        pin_gpu_by_uuid(args.gpu_uuid)
    configure_deterministic_backend_environment(args.n_gpu_layers)
    evidence_status = str(plan[0]["evidence_status"]) if plan else _stage_evidence_status(args.stage)
    output_dir = args.output_dir or _default_output_dir(args.stage, args.model_id, evidence_status)
    external_source_records: Dict[str, Dict[str, object]] = {}
    external_transformation_records: Dict[str, Dict[str, object]] = {}
    external_source_manifest: Optional[Dict[str, object]] = None
    if args.stage in {"robustness", "robustness_v2"}:
        source_primary_stage = (
            "primary_v2" if args.stage == "robustness_v2" else "primary"
        )
        source_ablation_stage = (
            "ablation_v2" if args.stage == "robustness_v2" else "ablation"
        )
        primary_root = (
            args.primary_results_root
            or args.project_root
            / "results"
            / "revision_v1"
            / source_primary_stage
        )
        ablation_root = (
            args.ablation_results_root
            or args.project_root
            / "results"
            / "revision_v1"
            / source_ablation_stage
        )
        external_source_records, source_results_manifest = (
            load_external_source_records(plan, primary_root, ablation_root)
        )
        robustness_root = (
            args.robustness_results_root
            or args.project_root / "results" / "revision_v1" / args.stage
        )
        (
            external_transformation_records,
            transformation_results_manifest,
        ) = load_external_transformation_records(plan, robustness_root)
        external_source_manifest = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "manifest_type": "robustness_execution_inputs",
            "source_results": source_results_manifest,
            "transformation_results": transformation_results_manifest,
        }
        external_source_manifest["inputs_sha256"] = hashlib.sha256(
            canonical_json_bytes(external_source_manifest)
        ).hexdigest()
    preparation = prepare_run_artifacts(
        plan,
        payloads,
        configs,
        args.stage,
        args.model_id,
        output_dir,
        config_dir=args.config_dir,
        project_root=args.project_root,
        context_limit=args.context_limit,
        gpu_uuid=args.gpu_uuid,
        n_gpu_layers=args.n_gpu_layers,
        resume=args.resume,
        external_source_manifest=external_source_manifest,
    )
    checkpoint = preparation["checkpoint"]
    if not pending_trial_ids([str(row["work_id"]) for row in plan], checkpoint):
        summary.update({"output_dir": str(output_dir), "execution": "already_complete"})
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    model = None
    memory_profiler: Optional[RuntimeMemoryProfiler] = RuntimeMemoryProfiler(
        args.gpu_uuid
    )
    memory_profiler.start()
    try:
        model, load_seconds = load_one_pinned_model(
            configs,
            args.model_id,
            project_root=args.project_root,
            context_limit=args.context_limit,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.threads,
            verbose=args.verbose_model,
        )
        append_jsonl_fsync(
            Path(output_dir) / "events.jsonl",
            {
                "event": "model_loaded",
                "at": utc_now(),
                "model_id": args.model_id,
                "model_load_seconds": load_seconds,
                "context_limit": args.context_limit,
                "n_gpu_layers": args.n_gpu_layers,
                "gpu_uuid": args.gpu_uuid,
                "n_batch": getattr(model, "rankcloak_n_batch", None),
                "n_ubatch": getattr(model, "rankcloak_n_ubatch", None),
            },
        )
        execution = run_work_plan(
            model,
            plan,
            payloads,
            configs,
            output_dir,
            context_limit=args.context_limit,
            external_source_records=external_source_records,
            external_transformation_records=external_transformation_records,
        )
        append_jsonl_fsync(
            Path(output_dir) / "events.jsonl", memory_profiler.stop()
        )
        memory_profiler = None
        append_jsonl_fsync(
            Path(output_dir) / "events.jsonl",
            {"event": "session_finished", "at": utc_now(), **execution},
        )
        summary.update(
            {
                "output_dir": str(output_dir),
                "model_load_seconds": load_seconds,
                "execution": execution,
            }
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if execution["failed_current"] == 0 else 2
    finally:
        if memory_profiler is not None:
            append_jsonl_fsync(
                Path(output_dir) / "events.jsonl", memory_profiler.stop()
            )
        close = getattr(model, "close", None) if model is not None else None
        if callable(close):
            close()


__all__ = [
    "EVIDENCE_ABLATION",
    "EVIDENCE_ABLATION_V2",
    "EVIDENCE_LIMITED",
    "EVIDENCE_MULTILINGUAL",
    "EVIDENCE_MULTILINGUAL_V2",
    "EVIDENCE_PRIMARY",
    "EVIDENCE_PRIMARY_V2",
    "EVIDENCE_ROBUSTNESS",
    "EVIDENCE_ROBUSTNESS_V2",
    "EVIDENCE_SMOKE",
    "EVIDENCE_SMOKE_V3",
    "PROTOCOL_CONTRACT_REVISION",
    "RESULT_SCHEMA_REVISION",
    "RevisionRunnerError",
    "append_jsonl_fsync",
    "build_argument_parser",
    "build_representation",
    "build_stage_plan",
    "configure_deterministic_backend_environment",
    "derive_control_seed",
    "execute_control_trial",
    "execute_rankcloak_trial",
    "execute_robustness_decode",
    "execute_robustness_reference",
    "execute_robustness_transform",
    "excluded_control_token_ids",
    "generate_length_matched_control",
    "load_jsonl_records",
    "load_external_source_records",
    "load_external_transformation_records",
    "main",
    "plan_summary",
    "prepare_run_artifacts",
    "representation_capacity_bits",
    "RuntimeMemoryProfiler",
    "reconcile_checkpoint_from_records",
    "relabel_limited_plan",
    "run_work_plan",
    "sample_top_p_token",
]
