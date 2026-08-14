"""Load, validate, and materialize metadata plans for revision-v1 configs.

This module is intentionally model-free.  It freezes experimental identities
and allocations but leaves model execution to the revision runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_payloads import (
    REVISION_HEX_PAYLOAD_CLASSES,
    REVISION_INSTANCES_PER_CLASS,
    REVISION_PAYLOAD_CLASSES,
    RevisionPayload,
    generate_revision_v1_payloads,
    revision_corpus_sha256,
)


DEFAULT_REVISION_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "revision_v1"
)
REQUIRED_REVISION_CONFIG_FILES: Tuple[str, ...] = (
    "models.json",
    "prompts.json",
    "primary.json",
    "ablations.json",
    "robustness.json",
    "multilingual.json",
    "statistics.json",
)


class RevisionConfigError(ValueError):
    """Raised when a frozen revision configuration is missing or inconsistent."""


def load_json_object(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RevisionConfigError("Missing revision config: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise RevisionConfigError(
            "Invalid JSON in revision config {}: {}".format(path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise RevisionConfigError(
            "Revision config must contain a JSON object: {}".format(path)
        )
    return value


def load_revision_config_set(
    config_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    root = Path(config_dir) if config_dir is not None else DEFAULT_REVISION_CONFIG_DIR
    return {
        Path(filename).stem: load_json_object(root / filename)
        for filename in REQUIRED_REVISION_CONFIG_FILES
    }


def flatten_prompt_templates(
    prompt_config: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    categories = prompt_config.get("categories", [])
    if not isinstance(categories, list):
        raise RevisionConfigError("prompts.categories must be a list")
    for category_index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise RevisionConfigError("each prompt category must be an object")
        category_id = category.get("category_id")
        templates = category.get("templates", [])
        if not isinstance(templates, list):
            raise RevisionConfigError("prompt templates must be a list")
        for template_index, template in enumerate(templates):
            if not isinstance(template, dict):
                raise RevisionConfigError("each prompt template must be an object")
            rows.append(
                {
                    "category_id": category_id,
                    "category_index": category_index,
                    "template_index": template_index,
                    "prompt_id": template.get("prompt_id"),
                    "text": template.get("text"),
                }
            )
    return rows


def prompt_by_id(
    prompt_id: str,
    prompt_config: Optional[Mapping[str, object]] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return one exact English prompt template by its frozen identifier."""

    if prompt_config is None:
        loaded = (
            dict(configs)
            if configs is not None
            else load_revision_config_set(config_dir)
        )
        prompt_config = loaded["prompts"]
    matches = [
        row
        for row in flatten_prompt_templates(prompt_config)
        if row["prompt_id"] == str(prompt_id)
    ]
    if len(matches) != 1:
        raise RevisionConfigError(
            "Expected one prompt named {!r}, found {}".format(
                prompt_id, len(matches)
            )
        )
    return matches[0]


def model_ids(model_config: Mapping[str, object]) -> List[str]:
    models = model_config.get("models", [])
    if not isinstance(models, list):
        raise RevisionConfigError("models.models must be a list")
    return [str(model.get("model_id")) for model in models if isinstance(model, dict)]


def unresolved_model_ids(model_config: Mapping[str, object]) -> List[str]:
    unresolved = []
    for model in model_config.get("models", []):
        if not isinstance(model, dict):
            continue
        if (
            model.get("pin_status") != "pinned"
            or not model.get("artifact_sha256")
            or not model.get("filename")
            or not model.get("relative_path")
        ):
            unresolved.append(str(model.get("model_id")))
    return unresolved


def missing_model_artifact_ids(
    model_config: Mapping[str, object],
    project_root: Optional[Path] = None,
) -> List[str]:
    """Return pinned model IDs whose exact local artifact is absent or wrong-sized."""

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    missing = []
    for model in model_config.get("models", []):
        if not isinstance(model, dict):
            continue
        relative_path = model.get("relative_path")
        expected_size = model.get("artifact_size_bytes")
        path = root / str(relative_path) if relative_path else None
        if (
            path is None
            or not path.is_file()
            or (
                expected_size is not None
                and path.stat().st_size != int(expected_size)
            )
        ):
            missing.append(str(model.get("model_id")))
    return missing


def verify_model_artifact_pins(
    model_config: Mapping[str, object],
    project_root: Optional[Path] = None,
    verify_sha256: bool = True,
) -> Dict[str, object]:
    """Verify configured path, byte size, and optionally the full content hash."""

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    records: List[Dict[str, object]] = []
    errors: List[str] = []
    for model in model_config.get("models", []):
        if not isinstance(model, dict):
            errors.append("model entry is not an object")
            continue
        model_id = str(model.get("model_id"))
        relative_path = model.get("relative_path")
        expected_size = model.get("artifact_size_bytes")
        expected_sha256 = model.get("artifact_sha256")
        path = root / str(relative_path) if relative_path else None
        status = "ok"
        actual_size = None
        actual_sha256 = None
        if path is None or not path.is_file():
            status = "missing"
        else:
            actual_size = path.stat().st_size
            if expected_size is None or actual_size != int(expected_size):
                status = "size_mismatch"
            elif verify_sha256:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_sha256 = digest.hexdigest()
                if not expected_sha256 or actual_sha256 != expected_sha256:
                    status = "sha256_mismatch"
        if status != "ok":
            errors.append("{}: {}".format(model_id, status))
        records.append(
            {
                "model_id": model_id,
                "relative_path": relative_path,
                "status": status,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "sha256_checked": actual_sha256 is not None,
            }
        )
    return {
        "status": "ok" if not errors else "error",
        "sha256_checked": bool(verify_sha256),
        "records": records,
        "errors": errors,
    }


def payload_split(payload_index: int) -> str:
    index = int(payload_index)
    if 0 <= index <= 35:
        return "train"
    if 36 <= index <= 47:
        return "validation"
    if 48 <= index <= 59:
        return "test"
    raise RevisionConfigError(
        "payload index {} is outside the frozen 0..59 corpus".format(index)
    )


def assign_english_prompt(
    payload: RevisionPayload,
    model_id: str,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Apply the frozen balanced-cyclic prompt allocation."""

    loaded = dict(configs) if configs is not None else load_revision_config_set(config_dir)
    models = model_ids(loaded["models"])
    if model_id not in models:
        raise RevisionConfigError("Unknown model_id: {}".format(model_id))
    try:
        class_index = list(REVISION_PAYLOAD_CLASSES).index(payload.payload_class)
    except ValueError as exc:
        raise RevisionConfigError(
            "Unknown revision payload class: {}".format(payload.payload_class)
        ) from exc
    model_index = models.index(model_id)
    categories = loaded["prompts"].get("categories", [])
    if len(categories) != 6:
        raise RevisionConfigError("balanced_cyclic_v1 requires six prompt categories")
    category_index = (
        payload.payload_index + model_index + class_index
    ) % len(categories)
    category = categories[category_index]
    templates = category.get("templates", [])
    if len(templates) != 3:
        raise RevisionConfigError(
            "balanced_cyclic_v1 requires three templates per category"
        )
    template_index = (
        payload.payload_index // 6 + model_index + class_index
    ) % len(templates)
    template = templates[template_index]
    return {
        "prompt_id": str(template["prompt_id"]),
        "prompt_text": str(template["text"]),
        "prompt_category": str(category["category_id"]),
        "prompt_category_index": category_index,
        "prompt_template_index": template_index,
    }


def _stable_identifier(prefix: str, *parts: object) -> str:
    readable = "__".join(
        str(part).strip().replace("/", "_").replace(" ", "_").lower()
        for part in parts
    )
    value = "{}__{}".format(prefix, readable)
    if len(value) <= 220:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return "{}__{}__{}".format(prefix, readable[:180].rstrip("_"), digest)


def build_primary_trial_plan(
    payloads: Optional[Iterable[RevisionPayload]] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Materialize the 6,480 model-free primary trial identities."""

    selected = list(payloads) if payloads is not None else generate_revision_v1_payloads()
    loaded = dict(configs) if configs is not None else load_revision_config_set(config_dir)
    models = model_ids(loaded["models"])
    protocols = loaded["primary"].get("protocols", [])
    canonical = loaded["primary"].get("segmented_canonical", {})
    tail = canonical.get("tail_policy", {})
    rows: List[Dict[str, object]] = []
    for payload in selected:
        for model_id in models:
            prompt = assign_english_prompt(payload, model_id, configs=loaded)
            for protocol in protocols:
                applies_to = protocol.get("applies_to")
                if (
                    applies_to == "hex_eligible"
                    and payload.payload_class not in REVISION_HEX_PAYLOAD_CLASSES
                ):
                    continue
                variant = str(protocol["protocol_variant"])
                segmented = bool(protocol.get("segmented", False))
                trial_id = _stable_identifier(
                    "revision_v1",
                    model_id,
                    payload.payload_name,
                    variant,
                    prompt["prompt_id"],
                )
                rows.append(
                    {
                        "trial_id": trial_id,
                        "study_id": loaded["primary"]["study_id"],
                        "study_phase": "primary_confirmatory",
                        "model_id": model_id,
                        "payload_name": payload.payload_name,
                        "payload_class": payload.payload_class,
                        "payload_index": payload.payload_index,
                        "payload_split": payload_split(payload.payload_index),
                        "protocol_variant": variant,
                        "representation_name": protocol["representation_name"],
                        "alphabet_size": protocol.get("alphabet_size"),
                        "prompt_id": prompt["prompt_id"],
                        "prompt_category": prompt["prompt_category"],
                        "language": "en",
                        "segmented": segmented,
                        "topic_schedule": protocol.get("topic_schedule"),
                        "token_filter": (
                            canonical.get("token_filter") if segmented else None
                        ),
                        "leadin_tokens": (
                            canonical.get("leadin_tokens") if segmented else None
                        ),
                        "tail_policy": (
                            tail.get("tail_policy_id") if segmented else None
                        ),
                        "segment_size_ranks": (
                            canonical.get("segment_size_ranks")
                            if segmented
                            else None
                        ),
                        "generation_required": True,
                        "source_trial_id": None,
                        "notes": "Frozen revision-v1 primary trial identity.",
                    }
                )
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RevisionConfigError("primary trial IDs are not unique")
    return rows


def build_primary_control_plan(
    primary_plan: Optional[Sequence[Mapping[str, object]]] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Build 6,480 full-length and 1,440 forced-span control identities."""

    loaded = dict(configs) if configs is not None else load_revision_config_set(config_dir)
    trials = (
        list(primary_plan)
        if primary_plan is not None
        else build_primary_trial_plan(configs=loaded)
    )
    rows: List[Dict[str, object]] = []
    for trial in trials:
        views = ["full_message"]
        if trial.get("segmented"):
            views.append("forced_span")
        for view in views:
            control_id = _stable_identifier(
                "control", trial["trial_id"], view
            )
            rows.append(
                {
                    "control_id": control_id,
                    "source_trial_id": trial["trial_id"],
                    "model_id": trial["model_id"],
                    "payload_name": trial["payload_name"],
                    "payload_class": trial["payload_class"],
                    "payload_split": trial["payload_split"],
                    "prompt_id": trial["prompt_id"],
                    "prompt_category": trial["prompt_category"],
                    "control_view": view,
                    "control_mode": "seeded_prompt_and_length_matched_sampling",
                    "target_token_count": None,
                    "sampling_seed": None,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "generation_required": True,
                    "notes": "Target length and deterministic seed resolve from source output.",
                }
            )
    control_ids = [row["control_id"] for row in rows]
    if len(control_ids) != len(set(control_ids)):
        raise RevisionConfigError("control IDs are not unique")
    return rows


def _ablation_conditions(config: Mapping[str, object]) -> List[Dict[str, object]]:
    canonical = dict(config["canonical"])
    conditions = [
        {
            "condition_id": "canonical",
            "factor_id": "canonical",
            "factor_level": "canonical",
            **canonical,
        }
    ]
    for factor in config.get("factors", []):
        factor_id = str(factor["factor_id"])
        canonical_level = factor["canonical_level"]
        for level in factor["levels"]:
            if level == canonical_level:
                continue
            condition = dict(canonical)
            condition[factor_id] = level
            condition.update(
                {
                    "condition_id": "{}={}".format(factor_id, level),
                    "factor_id": factor_id,
                    "factor_level": level,
                }
            )
            conditions.append(condition)
    return conditions


def build_ablation_trial_plan(
    payloads: Optional[Iterable[RevisionPayload]] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Materialize 1,872 unique OFAT rows; canonical rows reuse primary data."""

    selected = list(payloads) if payloads is not None else generate_revision_v1_payloads()
    loaded = dict(configs) if configs is not None else load_revision_config_set(config_dir)
    subset = [
        payload
        for payload in selected
        if payload.payload_class in REVISION_HEX_PAYLOAD_CLASSES
        and 0 <= payload.payload_index <= 11
    ]
    conditions = _ablation_conditions(loaded["ablations"])
    rows: List[Dict[str, object]] = []
    for payload in subset:
        for model_id in model_ids(loaded["models"]):
            prompt = assign_english_prompt(payload, model_id, configs=loaded)
            for condition in conditions:
                primary_overlap = condition["condition_id"] == "canonical"
                trial_id = _stable_identifier(
                    "revision_v1_ablation",
                    model_id,
                    payload.payload_name,
                    condition["condition_id"],
                    prompt["prompt_id"],
                )
                rows.append(
                    {
                        "trial_id": trial_id,
                        "study_id": loaded["primary"]["study_id"],
                        "study_phase": "ablation_confirmatory",
                        "model_id": model_id,
                        "payload_name": payload.payload_name,
                        "payload_class": payload.payload_class,
                        "payload_index": payload.payload_index,
                        "payload_split": payload_split(payload.payload_index),
                        "protocol_variant": "segmented_hex_multi_topic",
                        "representation_name": "raw_hex_nibbles",
                        "alphabet_size": 16,
                        "prompt_id": prompt["prompt_id"],
                        "prompt_category": prompt["prompt_category"],
                        "language": "en",
                        "segmented": True,
                        "topic_schedule": "deterministic_six_category_rotation",
                        "token_filter": condition["token_filter"],
                        "leadin_tokens": condition["leadin_tokens"],
                        "tail_policy": condition["tail_policy"],
                        "segment_size_ranks": condition["segment_size_ranks"],
                        "ablation_factor": condition["factor_id"],
                        "ablation_level": condition["factor_level"],
                        "primary_overlap": primary_overlap,
                        "generation_required": not primary_overlap,
                        "source_trial_id": (
                            _stable_identifier(
                                "revision_v1",
                                model_id,
                                payload.payload_name,
                                "segmented_hex_multi_topic",
                                prompt["prompt_id"],
                            )
                            if primary_overlap
                            else None
                        ),
                        "notes": "Frozen paired one-factor-at-a-time ablation.",
                    }
                )
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RevisionConfigError("ablation trial IDs are not unique")
    return rows


def build_multilingual_trial_plan(
    payloads: Optional[Iterable[RevisionPayload]] = None,
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    config_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Materialize the 576 secondary multilingual trial identities."""

    selected = list(payloads) if payloads is not None else generate_revision_v1_payloads()
    loaded = dict(configs) if configs is not None else load_revision_config_set(config_dir)
    selected = [payload for payload in selected if 0 <= payload.payload_index <= 5]
    categories = [
        category["category_id"] for category in loaded["prompts"]["categories"]
    ]
    models = model_ids(loaded["models"])
    rows: List[Dict[str, object]] = []
    for payload in selected:
        class_index = list(REVISION_PAYLOAD_CLASSES).index(payload.payload_class)
        for model_index, model_id in enumerate(models):
            category_index = (
                payload.payload_index + model_index + class_index
            ) % len(categories)
            category_id = categories[category_index]
            for language in loaded["multilingual"]["languages"]:
                language_id = language["language_id"]
                prompt_text = language["prompts_by_category"][category_id]
                prompt_id = "{}_{}_01".format(language_id, category_id)
                for protocol_variant in loaded["multilingual"]["protocols"]:
                    trial_id = _stable_identifier(
                        "revision_v1_multilingual",
                        model_id,
                        payload.payload_name,
                        language_id,
                        protocol_variant,
                        prompt_id,
                    )
                    rows.append(
                        {
                            "trial_id": trial_id,
                            "study_id": loaded["primary"]["study_id"],
                            "study_phase": "multilingual_secondary",
                            "model_id": model_id,
                            "payload_name": payload.payload_name,
                            "payload_class": payload.payload_class,
                            "payload_index": payload.payload_index,
                            "payload_split": payload_split(payload.payload_index),
                            "protocol_variant": protocol_variant,
                            "representation_name": (
                                "raw_subword_direct"
                                if protocol_variant == "direct_subword_calgacus"
                                else "ascii_bytes_fixed_radix"
                            ),
                            "alphabet_size": (
                                None
                                if protocol_variant == "direct_subword_calgacus"
                                else 16
                            ),
                            "prompt_id": prompt_id,
                            "prompt_text": prompt_text,
                            "prompt_category": category_id,
                            "language": language_id,
                            "segmented": False,
                            "generation_required": True,
                            "source_trial_id": None,
                            "notes": "Frozen revision-v1 multilingual secondary trial.",
                        }
                    )
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RevisionConfigError("multilingual trial IDs are not unique")
    return rows


def validate_revision_config_set(
    config_dir: Optional[Path] = None,
    require_execution_ready: bool = False,
) -> Dict[str, object]:
    """Validate frozen structure, arithmetic, allocations, and model pins."""

    errors: List[str] = []
    warnings: List[str] = []
    try:
        loaded = load_revision_config_set(config_dir)
    except RevisionConfigError as exc:
        return {"status": "error", "errors": [str(exc)], "warnings": []}

    for name, config in loaded.items():
        if config.get("schema_version") != "1.0":
            errors.append("{} has unsupported schema_version".format(name))
        if not config.get("config_id"):
            errors.append("{} is missing config_id".format(name))

    models = model_ids(loaded["models"])
    if len(models) != 3 or len(set(models)) != 3:
        errors.append("models.json must define three unique model IDs")
    unresolved = unresolved_model_ids(loaded["models"])
    if unresolved:
        message = "unresolved model artifact pins: {}".format(", ".join(unresolved))
        if require_execution_ready:
            errors.append(message)
        else:
            warnings.append(message)
    missing_artifacts = missing_model_artifact_ids(loaded["models"])
    if missing_artifacts:
        message = "missing local pinned model artifacts: {}".format(
            ", ".join(missing_artifacts)
        )
        if require_execution_ready:
            errors.append(message)
        else:
            warnings.append(message)
    model_artifact_validation = verify_model_artifact_pins(
        loaded["models"], verify_sha256=require_execution_ready
    )
    if require_execution_ready and model_artifact_validation["status"] != "ok":
        errors.append(
            "model artifact pin validation failed: {}".format(
                "; ".join(model_artifact_validation["errors"])
            )
        )

    prompts = flatten_prompt_templates(loaded["prompts"])
    categories = Counter(row["category_id"] for row in prompts)
    if len(categories) != 6 or set(categories.values()) != {3}:
        errors.append("prompts.json must define six categories with three templates each")
    prompt_ids = [row["prompt_id"] for row in prompts]
    if len(prompt_ids) != 18 or len(set(prompt_ids)) != 18:
        errors.append("prompts.json must define 18 unique prompt IDs")

    corpus = loaded["primary"].get("corpus", {})
    if tuple(corpus.get("payload_classes", [])) != REVISION_PAYLOAD_CLASSES:
        errors.append("primary corpus class order differs from revision payload generator")
    if corpus.get("instances_per_class") != REVISION_INSTANCES_PER_CLASS:
        errors.append("primary corpus instances_per_class must be 60")
    actual_corpus_sha256 = revision_corpus_sha256()
    configured_corpus_sha256 = corpus.get("corpus_sha256")
    if configured_corpus_sha256 is None:
        warnings.append("primary corpus_sha256 has not yet been frozen")
    elif configured_corpus_sha256 != actual_corpus_sha256:
        errors.append("primary corpus_sha256 does not match the generator")

    primary_plan = build_primary_trial_plan(configs=loaded)
    primary_controls = build_primary_control_plan(primary_plan, configs=loaded)
    ablation_plan = build_ablation_trial_plan(configs=loaded)
    multilingual_plan = build_multilingual_trial_plan(configs=loaded)
    expected_primary = loaded["primary"]["expected_counts"]
    if len(primary_plan) != expected_primary["rankcloak_trials"]:
        errors.append("primary trial count does not match config")
    if len(primary_controls) != expected_primary["ordinary_control_texts"]:
        errors.append("primary control count does not match config")
    prompt_counts = Counter(row["prompt_id"] for row in primary_plan)
    if set(prompt_counts.values()) != {
        expected_primary["rankcloak_trials_per_prompt_template"]
    }:
        errors.append("primary prompt allocation is not exactly balanced")
    model_counts = Counter(row["model_id"] for row in primary_plan)
    if set(model_counts.values()) != {
        expected_primary["rankcloak_trials_per_model"]
    }:
        errors.append("primary model allocation is not exactly balanced")

    expected_ablation = loaded["ablations"]["expected_counts"]
    if len(ablation_plan) != expected_ablation["unique_condition_rows"]:
        errors.append("ablation unique-row count does not match config")
    generated_key = (
        "new_generated_rankcloak_texts_planned"
        if "new_generated_rankcloak_texts_planned" in expected_ablation
        else "new_generated_rankcloak_texts"
    )
    if sum(bool(row["generation_required"]) for row in ablation_plan) != expected_ablation[
        generated_key
    ]:
        errors.append("ablation new-generation count does not match config")

    robustness = loaded["robustness"]
    if len(robustness.get("transformations", [])) != 13:
        errors.append("robustness config must define 13 transformations")
    additional_decodes = (
        robustness["replay_modes"]["additional_decode_only_runs"]
        + robustness["raw_transmission"]["additional_decode_only_runs"]
        + robustness["limited_mitigation"]["additional_decode_only_runs"]
        + robustness["cross_model_mismatch"]["additional_decode_only_runs"]
    )
    if additional_decodes != robustness["expected_counts"][
        "additional_decode_only_runs"
    ]:
        errors.append("robustness decode-only arithmetic is inconsistent")

    expected_multilingual = loaded["multilingual"]["expected_counts"]
    if len(multilingual_plan) != expected_multilingual["rankcloak_trials"]:
        errors.append("multilingual trial count does not match config")

    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "warnings": warnings,
        "execution_ready": (
            not unresolved
            and not missing_artifacts
            and not errors
            and configured_corpus_sha256 is not None
        ),
        "unresolved_model_ids": unresolved,
        "missing_model_artifact_ids": missing_artifacts,
        "model_artifact_validation": model_artifact_validation,
        "corpus_sha256": actual_corpus_sha256,
        "counts": {
            "payloads": 480,
            "prompt_templates": len(prompts),
            "primary_rankcloak_trials": len(primary_plan),
            "primary_controls": len(primary_controls),
            "ablation_unique_rows": len(ablation_plan),
            "ablation_new_generation_rows": sum(
                bool(row["generation_required"]) for row in ablation_plan
            ),
            "robustness_additional_decode_only_runs": additional_decodes,
            "multilingual_rankcloak_trials": len(multilingual_plan),
        },
    }
