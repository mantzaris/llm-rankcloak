"""Participant-free readability and blinded-stimulus evidence.

This module consumes immutable primary JSONL records.  It prepares a balanced
computational stimulus inventory and deterministic surface diagnostics, but it
does not create participant schedules, human ratings, ethical approval, or a
completed human-written control condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from rankcloak.revision_statistics import automated_text_quality_metrics


OUTPUT_FILENAMES = {
    "candidate_pool": "computational_candidate_pool.csv",
    "inventory": "planned_stimulus_inventory.csv",
    "provenance": "selected_stimulus_provenance.csv",
    "unavailable": "unavailable_human_control_strata.csv",
    "metrics": "selected_stimulus_readability_metrics.csv",
    "summary": "selected_stimulus_readability_summary.csv",
    "randomization": "randomization_metadata.json",
    "status": "human_evaluation_status.json",
    "manifest": "readability_and_stimulus_manifest.json",
}

HUMAN_CONTROL_CONDITION = "human_written_control"
AVAILABLE_CONDITIONS = (
    "ordinary_llm_control",
    "direct_subword_calgacus",
    "rankcloak_ascii_b8",
    "rankcloak_ascii_b16",
    "rankcloak_hex_nibble",
    "rankcloak_segmented_forced_span",
    "rankcloak_segmented_full_message",
)
PROTOCOL_CONDITIONS = {
    "direct_subword_calgacus": ("direct_subword_calgacus", "full_text"),
    "nonseg_ascii_b8": ("rankcloak_ascii_b8", "full_text"),
    "nonseg_ascii_b16": ("rankcloak_ascii_b16", "full_text"),
    "nonseg_hex_nibble_b16": ("rankcloak_hex_nibble", "full_text"),
}
SEGMENTED_PROTOCOLS = {
    "segmented_hex_single_topic",
    "segmented_hex_multi_topic",
}
EXPERIMENT_TO_DESIGN_CATEGORY = {
    "casual_conversation": "casual_conversation",
    "professional_communication": "professional_communication",
    "forum_question_answer": "forum_question_answer",
    "recipe_procedure": "procedural_instructions",
    "personal_narrative_blog": "personal_narrative_blog",
    "factual_explanatory": "factual_explanatory_prose",
}
SUMMARY_METRICS = (
    "word_count",
    "sentence_count",
    "character_count",
    "flesch_reading_ease_heuristic",
    "flesch_kincaid_grade_heuristic",
    "coleman_liau_index",
    "unique_word_fraction",
    "repeated_bigram_fraction",
    "repeated_trigram_fraction",
    "maximum_identical_word_run",
    "duplicate_sentence_count",
    "artifact_like_fragment_count",
    "surface_flag_total",
    "tfidf_prompt_similarity",
)


class ReadabilityEvidenceError(ValueError):
    """Raised when saved evidence cannot support the frozen stimulus strata."""


@dataclass(frozen=True)
class ReadabilityArtifacts:
    output_dir: str
    files: dict[str, str]
    summary: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise ReadabilityEvidenceError(f"Missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReadabilityEvidenceError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadabilityEvidenceError(f"{label} must contain a JSON object")
    return value


def _stable_digest(seed: int, stage: str, *parts: object) -> str:
    value = "\x1f".join([str(seed), stage, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_contract(
    prompt_config: Mapping[str, Any], design: Mapping[str, Any]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    categories = prompt_config.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ReadabilityEvidenceError("Prompt config lacks categories")
    design_categories = [str(value) for value in design.get("prompt_categories", [])]
    expected_templates = int(design.get("templates_per_category", 0))
    mapping: dict[str, dict[str, str]] = {}
    observed_design_categories: list[str] = []
    for category in categories:
        if not isinstance(category, Mapping):
            raise ReadabilityEvidenceError("Prompt category entries must be objects")
        experiment_category = str(category.get("category_id", ""))
        design_category = EXPERIMENT_TO_DESIGN_CATEGORY.get(
            experiment_category, experiment_category
        )
        templates = category.get("templates")
        if not isinstance(templates, list) or len(templates) != expected_templates:
            raise ReadabilityEvidenceError(
                f"Prompt category {experiment_category} has the wrong template count"
            )
        observed_design_categories.append(design_category)
        for number, template in enumerate(templates, start=1):
            if not isinstance(template, Mapping) or not template.get("prompt_id"):
                raise ReadabilityEvidenceError("Prompt templates require prompt_id")
            prompt_id = str(template["prompt_id"])
            if prompt_id in mapping:
                raise ReadabilityEvidenceError(f"Duplicate prompt_id {prompt_id}")
            mapping[prompt_id] = {
                "experiment_category": experiment_category,
                "design_category": design_category,
                "template_id": f"{design_category}_template_{number}",
                "prompt_text": str(template.get("text", "")),
                "template_number": str(number),
            }
    if observed_design_categories != design_categories:
        raise ReadabilityEvidenceError(
            "Experiment-to-design prompt category mapping does not match design order"
        )
    return mapping, design_categories


def _record_variants(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    if record.get("record_type") == "ordinary_control":
        return [("ordinary_llm_control", "full_text")]
    if record.get("record_type") != "rankcloak_trial":
        return []
    protocol = str(record.get("protocol_variant", ""))
    if protocol in PROTOCOL_CONDITIONS:
        return [PROTOCOL_CONDITIONS[protocol]]
    if protocol in SEGMENTED_PROTOCOLS:
        return [
            ("rankcloak_segmented_forced_span", "forced_text"),
            ("rankcloak_segmented_full_message", "full_text"),
        ]
    return []


def _load_candidates(
    records_paths: Sequence[str | Path],
    prompt_map: Mapping[str, Mapping[str, str]],
    eligible_payload_classes: set[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    model_order: list[str] = []
    work_ids: set[tuple[str, str]] = set()
    input_counts: dict[str, int] = {}
    for raw_path in records_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ReadabilityEvidenceError(f"Missing primary record file: {path}")
        record_count = 0
        path_models: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for source_row, line in enumerate(handle, start=1):
                record_count += 1
                try:
                    record = json.loads(line)
                except Exception as exc:
                    raise ReadabilityEvidenceError(
                        f"Malformed JSON at {path}:{source_row}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ReadabilityEvidenceError(
                        f"Record at {path}:{source_row} is not an object"
                    )
                model_id = str(record.get("model_id", ""))
                if not model_id:
                    raise ReadabilityEvidenceError(
                        f"Record at {path}:{source_row} lacks model_id"
                    )
                path_models.add(model_id)
                if model_id not in model_order:
                    model_order.append(model_id)
                work_id = str(record.get("work_id", ""))
                if not work_id:
                    raise ReadabilityEvidenceError(
                        f"Record at {path}:{source_row} lacks work_id"
                    )
                work_key = (model_id, work_id)
                if work_key in work_ids:
                    raise ReadabilityEvidenceError(
                        f"Duplicate model/work identity {model_id}/{work_id}"
                    )
                work_ids.add(work_key)
                if str(record.get("execution_status", "")) != "completed":
                    raise ReadabilityEvidenceError(
                        f"Noncompleted primary record at {path}:{source_row}"
                    )
                if str(record.get("payload_class", "")) not in eligible_payload_classes:
                    continue
                prompt_id = str(record.get("prompt_id", ""))
                prompt = prompt_map.get(prompt_id)
                if prompt is None:
                    raise ReadabilityEvidenceError(
                        f"Unknown prompt_id {prompt_id} at {path}:{source_row}"
                    )
                if str(record.get("prompt_category", "")) != prompt["experiment_category"]:
                    raise ReadabilityEvidenceError(
                        f"Prompt category mismatch for {prompt_id} at {path}:{source_row}"
                    )
                raw_record_sha256 = hashlib.sha256(line.encode("utf-8")).hexdigest()
                for condition, text_field in _record_variants(record):
                    text = record.get(text_field)
                    if not isinstance(text, str) or not text.strip():
                        raise ReadabilityEvidenceError(
                            f"Empty {text_field} for {condition} at {path}:{source_row}"
                        )
                    candidate_id = "C" + hashlib.sha256(
                        f"{raw_record_sha256}\x1f{condition}\x1f{text_field}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:24]
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "condition": condition,
                            "prompt_category": prompt["design_category"],
                            "experiment_prompt_category": prompt[
                                "experiment_category"
                            ],
                            "template_id": prompt["template_id"],
                            "template_number": int(prompt["template_number"]),
                            "prompt_id": prompt_id,
                            "prompt_text": prompt["prompt_text"],
                            "payload_id": str(record.get("payload_name", "")),
                            "payload_class": str(record.get("payload_class", "")),
                            "model_id": model_id,
                            "presentation_scope": (
                                "forced_span"
                                if text_field == "forced_text"
                                else "full_message"
                            ),
                            "protocol_variant": str(
                                record.get("protocol_variant", "ordinary_llm_control")
                            ),
                            "topic_schedule": str(record.get("topic_schedule", "")),
                            "source_work_id": work_id,
                            "source_trial_id": str(
                                record.get("trial_id")
                                or record.get("source_trial_id")
                                or work_id
                            ),
                            "source_file": str(path.resolve()),
                            "source_row": source_row,
                            "source_record_sha256": raw_record_sha256,
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "message_text": text,
                            "language": str(record.get("language", "en")),
                            "license_status": "experiment_generated_internal",
                            "safety_screen_status": "not_manually_reviewed",
                        }
                    )
        if len(path_models) != 1:
            raise ReadabilityEvidenceError(
                f"Primary record file must contain one model identity: {path}"
            )
        input_counts[str(path.resolve())] = record_count
    if len(model_order) < 1:
        raise ReadabilityEvidenceError("No primary model identities were observed")
    identifiers = [row["candidate_id"] for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ReadabilityEvidenceError("Candidate IDs are not unique")
    return candidates, model_order, input_counts


def _ordered_strata(design: Mapping[str, Any]) -> Iterable[tuple[str, int, str, int, str]]:
    categories = [str(value) for value in design["prompt_categories"]]
    templates = int(design["templates_per_category"])
    payload_classes = [str(value) for value in design["eligible_payload_classes"]]
    for category_index, category in enumerate(categories):
        for template_number in range(1, templates + 1):
            template_id = f"{category}_template_{template_number}"
            for payload_index, payload_class in enumerate(payload_classes):
                yield category, category_index, template_id, payload_index, payload_class


def _prepare_pool_and_selection(
    candidates: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    model_order: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed = int(design["random_seed"])
    replicates = int(design.get("candidate_replicates_per_stratum", 0))
    if replicates < 2:
        raise ReadabilityEvidenceError(
            "At least two computational candidates per stratum are required"
        )
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        key = (
            str(row["condition"]),
            str(row["prompt_category"]),
            str(row["template_id"]),
            str(row["payload_class"]),
            str(row["model_id"]),
        )
        groups[key].append(row)

    design_conditions = [str(value) for value in design.get("conditions", [])]
    if set(design_conditions) != set(AVAILABLE_CONDITIONS) | {HUMAN_CONTROL_CONDITION}:
        raise ReadabilityEvidenceError("Human-study condition set differs from contract")
    available_in_order = [
        condition for condition in design_conditions if condition != HUMAN_CONTROL_CONDITION
    ]
    used_records: set[str] = set()
    used_texts: set[str] = set()
    pool: list[dict[str, Any]] = []
    candidate_availability: dict[str, list[int]] = defaultdict(list)
    for condition in available_in_order:
        for category, category_index, template_id, payload_index, payload_class in _ordered_strata(design):
            target_model = model_order[
                (category_index + int(template_id.rsplit("_", 1)[1]) - 1 + payload_index)
                % len(model_order)
            ]
            key = (condition, category, template_id, payload_class, target_model)
            options = sorted(groups.get(key, []), key=lambda row: row["candidate_id"])
            candidate_availability[condition].append(len(options))
            options.sort(
                key=lambda row: _stable_digest(
                    seed, "candidate_pool", condition, template_id,
                    payload_class, row["candidate_id"]
                )
            )
            selected_options: list[dict[str, Any]] = []
            for row in options:
                if row["source_record_sha256"] in used_records:
                    continue
                if row["text_sha256"] in used_texts:
                    continue
                selected_options.append(row)
                used_records.add(str(row["source_record_sha256"]))
                used_texts.add(str(row["text_sha256"]))
                if len(selected_options) == replicates:
                    break
            if len(selected_options) != replicates:
                raise ReadabilityEvidenceError(
                    "Insufficient unique candidates for stratum "
                    f"{condition}/{template_id}/{payload_class}/{target_model}: "
                    f"needed {replicates}, found {len(selected_options)}"
                )
            for replicate, row in enumerate(selected_options, start=1):
                item = dict(row)
                item["candidate_replicate"] = replicate
                item["target_model_assignment"] = target_model
                pool.append(item)

    selected_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        selected_by_key[
            (
                str(row["condition"]),
                str(row["prompt_category"]),
                str(row["template_id"]),
                str(row["payload_class"]),
            )
        ].append(row)
    selected: list[dict[str, Any]] = []
    for condition in available_in_order:
        for category, _, template_id, _, payload_class in _ordered_strata(design):
            options = selected_by_key[(condition, category, template_id, payload_class)]
            options.sort(
                key=lambda row: _stable_digest(
                    seed, "selected_stimulus", row["candidate_id"]
                )
            )
            selected.append(dict(options[0]))

    expected_strata = (
        len(design["prompt_categories"])
        * int(design["templates_per_category"])
        * len(design["eligible_payload_classes"])
    )
    expected_selected = len(available_in_order) * expected_strata
    if len(selected) != expected_selected:
        raise ReadabilityEvidenceError("Selected computational stimulus count mismatch")
    if len({row["text_sha256"] for row in selected}) != len(selected):
        raise ReadabilityEvidenceError("Selected stimulus texts are not unique")
    if len({row["source_record_sha256"] for row in selected}) != len(selected):
        raise ReadabilityEvidenceError("Selected stimuli reuse a raw source record")
    model_balance = {
        condition: dict(
            sorted(Counter(
                row["model_id"] for row in selected if row["condition"] == condition
            ).items())
        )
        for condition in available_in_order
    }
    topic_balance = {
        condition: dict(
            sorted(Counter(
                row["protocol_variant"]
                for row in selected
                if row["condition"] == condition
            ).items())
        )
        for condition in (
            "rankcloak_segmented_forced_span",
            "rankcloak_segmented_full_message",
        )
    }
    audit = {
        "candidate_availability": {
            condition: {
                "strata": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for condition, values in sorted(candidate_availability.items())
        },
        "model_balance": model_balance,
        "segmented_protocol_distribution": topic_balance,
    }
    return pool, selected, audit


def _unavailable_human_rows(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, _, template_id, _, payload_class in _ordered_strata(design):
        rows.append(
            {
                "condition": HUMAN_CONTROL_CONDITION,
                "prompt_category": category,
                "template_id": template_id,
                "payload_class": payload_class,
                "availability_status": "unavailable",
                "reason_code": "insufficient_licensed_coverage_and_manual_review_pending",
                "message_text_available": False,
                "counted_as_human_outcome": False,
            }
        )
    return rows


def _blind_inventory(
    selected: Sequence[Mapping[str, Any]],
    unavailable: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str, str], str]]:
    identities: list[tuple[str, str]] = [
        ("candidate", str(row["candidate_id"])) for row in selected
    ]
    unavailable_keys: list[tuple[str, str, str, str]] = []
    for row in unavailable:
        key = (
            str(row["condition"]), str(row["prompt_category"]),
            str(row["template_id"]), str(row["payload_class"]),
        )
        unavailable_keys.append(key)
        identities.append(("unavailable", "\x1f".join(key)))
    identities.sort(key=lambda item: _stable_digest(seed, "blind", *item))
    blind = {identity: f"B{index:04d}" for index, identity in enumerate(identities, 1)}
    inventory_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for row in selected:
        blind_id = blind[("candidate", str(row["candidate_id"]))]
        inventory_rows.append(
            {
                "stimulus_blind_id": blind_id,
                "topic_label": str(row["prompt_category"]).replace("_", " "),
                "message_text": row["message_text"],
                "availability_status": "computational_text_available_not_safety_reviewed",
            }
        )
        provenance_rows.append(
            {
                "stimulus_blind_id": blind_id,
                **{
                    key: row[key]
                    for key in (
                        "candidate_id", "condition", "prompt_category", "template_id",
                        "prompt_id", "payload_id", "payload_class", "model_id",
                        "presentation_scope", "protocol_variant", "topic_schedule",
                        "source_work_id", "source_trial_id", "source_file", "source_row",
                        "source_record_sha256", "text_sha256", "language",
                        "license_status", "safety_screen_status",
                    )
                },
            }
        )
    unavailable_blind: dict[tuple[str, str, str, str], str] = {}
    for row, key in zip(unavailable, unavailable_keys):
        blind_id = blind[("unavailable", "\x1f".join(key))]
        unavailable_blind[key] = blind_id
        inventory_rows.append(
            {
                "stimulus_blind_id": blind_id,
                "topic_label": str(row["prompt_category"]).replace("_", " "),
                "message_text": "",
                "availability_status": "unavailable_human_written_control",
            }
        )
    inventory = pd.DataFrame(inventory_rows).sort_values("stimulus_blind_id")
    provenance = pd.DataFrame(provenance_rows).sort_values("stimulus_blind_id")
    return inventory.reset_index(drop=True), provenance.reset_index(drop=True), unavailable_blind


def _readability_tables(
    selected: Sequence[Mapping[str, Any]],
    provenance: pd.DataFrame,
    *,
    seed: int,
    confidence_level: float,
    bootstrap_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blind_by_candidate = dict(
        zip(provenance["candidate_id"], provenance["stimulus_blind_id"])
    )
    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        metrics = automated_text_quality_metrics(
            str(selected_row["message_text"]),
            str(selected_row["prompt_text"]),
            language=str(selected_row["language"]),
        )
        rows.append(
            {
                "stimulus_blind_id": blind_by_candidate[selected_row["candidate_id"]],
                "candidate_id": selected_row["candidate_id"],
                "condition": selected_row["condition"],
                "prompt_category": selected_row["prompt_category"],
                "template_id": selected_row["template_id"],
                "payload_class": selected_row["payload_class"],
                "model_id": selected_row["model_id"],
                "presentation_scope": selected_row["presentation_scope"],
                "text_sha256": selected_row["text_sha256"],
                **metrics,
            }
        )
    metric_frame = pd.DataFrame(rows).sort_values("stimulus_blind_id").reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    alpha = (1.0 - confidence_level) / 2.0
    for condition, cell in metric_frame.groupby("condition", sort=True):
        template_labels = sorted(cell["template_id"].unique())
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(cell[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            estimates: list[float] = []
            if len(template_labels) > 1 and bootstrap_resamples > 0:
                rng = np.random.default_rng(
                    int(_stable_digest(seed, "readability_ci", condition, metric)[:16], 16)
                )
                by_template = {
                    label: pd.to_numeric(
                        cell.loc[cell["template_id"] == label, metric], errors="coerce"
                    ).to_numpy(dtype=float)
                    for label in template_labels
                }
                for _ in range(bootstrap_resamples):
                    sampled = rng.choice(template_labels, len(template_labels), replace=True)
                    draw = np.concatenate([by_template[label] for label in sampled])
                    draw = draw[np.isfinite(draw)]
                    if draw.size:
                        estimates.append(float(np.mean(draw)))
            summary_rows.append(
                {
                    "condition": condition,
                    "outcome": metric,
                    "n": int(len(finite)),
                    "missing_n": int(len(values) - len(finite)),
                    "prompt_template_units": len(template_labels),
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "standard_deviation": (
                        float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
                    ),
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "minimum": float(finite.min()) if len(finite) else np.nan,
                    "maximum": float(finite.max()) if len(finite) else np.nan,
                    "ci_low": (
                        float(np.quantile(estimates, alpha)) if estimates else np.nan
                    ),
                    "ci_high": (
                        float(np.quantile(estimates, 1.0 - alpha))
                        if estimates else np.nan
                    ),
                    "interval_method": "prompt_template_cluster_percentile_bootstrap",
                    "confidence_level": confidence_level,
                    "bootstrap_resamples": bootstrap_resamples,
                    "estimand_scope": "selected_computational_stimulus_surface_diagnostic",
                    "human_rating_substitute": False,
                }
            )
    return metric_frame, pd.DataFrame(summary_rows)


def _atomic_write_csv(frame: pd.DataFrame, target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)
    return target


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def _input_entry(path: str | Path, *, record_count: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    entry: dict[str, Any] = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if record_count is not None:
        entry["record_count"] = record_count
    return entry


def build_readability_artifacts(
    *,
    records_paths: Sequence[str | Path],
    design_path: str | Path,
    prompt_config_path: str | Path,
    control_audit_path: str | Path,
    instrument_path: str | Path,
    power_paths: Sequence[str | Path],
    output_dir: str | Path,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    overwrite: bool = False,
) -> ReadabilityArtifacts:
    """Build a balanced, blinded, explicitly uncollected stimulus package."""

    if not 0.0 < confidence_level < 1.0 or bootstrap_resamples <= 0:
        raise ReadabilityEvidenceError("Invalid interval configuration")
    design = _read_json(design_path, label="human-study design")
    prompts = _read_json(prompt_config_path, label="prompt config")
    control_audit = _read_json(control_audit_path, label="human-control audit")
    _read_json(instrument_path, label="rating instrument")
    for path in power_paths:
        if not Path(path).is_file():
            raise ReadabilityEvidenceError(f"Missing power-planning input: {path}")
    if "DRAFT" not in str(design.get("status", "")):
        raise ReadabilityEvidenceError("Human-study design is not marked as draft")
    if str(design.get("language", "")).lower() != "english":
        raise ReadabilityEvidenceError("Stimulus preparation is English-only")
    prompt_map, _ = _prompt_contract(prompts, design)
    eligible_payloads = {
        str(value) for value in design.get("eligible_payload_classes", [])
    }
    candidates, model_order, input_counts = _load_candidates(
        records_paths, prompt_map, eligible_payloads
    )
    pool, selected, selection_audit = _prepare_pool_and_selection(
        candidates, design, model_order
    )
    unavailable = _unavailable_human_rows(design)
    seed = int(design["random_seed"])
    inventory, provenance, unavailable_blind = _blind_inventory(
        selected, unavailable, seed
    )
    unavailable_frame = pd.DataFrame(
        [
            {
                "stimulus_blind_id": unavailable_blind[
                    (
                        str(row["condition"]), str(row["prompt_category"]),
                        str(row["template_id"]), str(row["payload_class"]),
                    )
                ],
                **row,
            }
            for row in unavailable
        ]
    ).sort_values("stimulus_blind_id")
    metrics, metric_summary = _readability_tables(
        selected,
        provenance,
        seed=seed,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
    )

    pool_frame = pd.DataFrame(pool).drop(columns=["message_text", "prompt_text"])
    pool_frame = pool_frame.sort_values("candidate_id").reset_index(drop=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise ReadabilityEvidenceError(
            "Refusing to overwrite readability outputs: "
            + ", ".join(str(path) for path in existing)
        )
    frames = {
        "candidate_pool": pool_frame,
        "inventory": inventory,
        "provenance": provenance,
        "unavailable": unavailable_frame,
        "metrics": metrics,
        "summary": metric_summary,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, targets[key])

    design_model_labels = [str(value) for value in design.get("model_families", [])]
    llama_label_mismatch = any("llama_3_1" in value for value in design_model_labels) and any(
        "llama3_8b" in value and "3_1" not in value for value in model_order
    )
    randomization = {
        "schema_version": "rankcloak-participant-free-stimulus-selection-v1",
        "status": "INCOMPLETE_NO_PARTICIPANT_RANDOMIZATION",
        "seed": seed,
        "selection_algorithm": "sha256_seeded_stratum_selection_v1",
        "blinding_algorithm": "sha256_seeded_global_order_v1",
        "candidate_replicates_per_available_stratum": int(
            design["candidate_replicates_per_stratum"]
        ),
        "available_condition_count": len(AVAILABLE_CONDITIONS),
        "planned_condition_count": len(design["conditions"]),
        "selected_computational_stimuli": len(selected),
        "unavailable_human_control_strata": len(unavailable),
        "planned_inventory_rows": len(inventory),
        "participant_schedule_emitted": False,
        "attention_checks_scheduled": False,
        "manual_safety_review_completed": False,
        "selection_audit": selection_audit,
    }
    _atomic_write_json(randomization, targets["randomization"])
    status = {
        "schema_version": "rankcloak-human-evaluation-status-v1",
        "status": "UNCOLLECTED_BLOCKED_NO_HUMAN_PARTICIPANT_DATA",
        "human_participant_rows": 0,
        "human_rating_rows": 0,
        "human_outcomes_estimated": False,
        "recruitment_authorized": False,
        "survey_deployed": False,
        "participant_schedule_emitted": False,
        "computational_stimulus_rows_available": len(selected),
        "human_control_rows_available": 0,
        "human_control_rows_unavailable": len(unavailable),
        "human_control_gate": control_audit.get("pre_recruitment_gate"),
        "human_control_manual_review_status": "not_completed",
        "automated_metrics_are_human_rating_substitutes": False,
        "power_results_scope": "planning_only_not_an_observed_result",
        "design_model_family_labels": design_model_labels,
        "observed_computational_model_ids": model_order,
        "model_identity_alignment_status": (
            "draft_design_llama_version_label_mismatch"
            if llama_label_mismatch
            else "design_labels_not_used_as_scientific_model_identity"
        ),
    }
    _atomic_write_json(status, targets["status"])

    inputs: dict[str, Any] = {
        "records": [
            _input_entry(path, record_count=input_counts[str(Path(path).resolve())])
            for path in records_paths
        ],
        "design": _input_entry(design_path),
        "prompt_config": _input_entry(prompt_config_path),
        "human_control_audit": _input_entry(control_audit_path),
        "rating_instrument": _input_entry(instrument_path),
        "power_planning": [_input_entry(path) for path in power_paths],
    }
    summary = {
        "raw_input_records": sum(input_counts.values()),
        "eligible_candidate_records_before_balancing": len(candidates),
        "candidate_pool_rows": len(pool_frame),
        "selected_computational_stimuli": len(selected),
        "unavailable_human_control_strata": len(unavailable),
        "planned_inventory_rows": len(inventory),
        "readability_metric_rows": len(metrics),
        "readability_summary_rows": len(metric_summary),
        "human_participant_rows": 0,
        "human_rating_rows": 0,
    }
    manifest: dict[str, Any] = {
        "schema_version": "rankcloak-readability-and-stimulus-evidence-v1",
        "status": "computational_complete_human_evaluation_uncollected",
        "inputs": inputs,
        "outputs": {},
        "summary": summary,
        "selection_unit": "one_complete_message_not_individual_segment",
        "readability_interval_unit": "prompt_template",
        "confidence_level": confidence_level,
        "bootstrap_resamples": bootstrap_resamples,
        "automated_metrics_scope": "surface_diagnostics_not_human_judgements",
        "large_raw_artifacts_copied": False,
    }
    for key, target in targets.items():
        if key == "manifest":
            continue
        entry = {
            "path": str(target.resolve()),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        if key in frames:
            entry["row_count"] = int(len(frames[key]))
        manifest["outputs"][key] = entry
    _atomic_write_json(manifest, targets["manifest"])
    return ReadabilityArtifacts(
        output_dir=str(output_path.resolve()),
        files={key: str(path.resolve()) for key, path in targets.items()},
        summary=summary,
    )
