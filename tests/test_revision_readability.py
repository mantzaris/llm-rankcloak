from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_readability import (
    OUTPUT_FILENAMES,
    ReadabilityEvidenceError,
    build_readability_artifacts,
    file_sha256,
)


MODELS = ("llama3_8b", "qwen2_5_7b", "mistral_7b")


def _record(
    *,
    model: str,
    index: int,
    record_type: str,
    protocol: str,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "execution_status": "completed",
        "model_id": model,
        "work_id": f"{model}_work_{index}",
        "trial_id": f"{model}_trial_{index}",
        "payload_name": f"payload_{index}",
        "payload_class": "sha256_hex",
        "prompt_id": "casual_prompt",
        "prompt_category": "casual_conversation",
        "language": "en",
        "protocol_variant": protocol,
        "topic_schedule": "single_topic",
        "forced_text": f"Forced message {model} {protocol} {index}.",
        "full_text": f"Full message {model} {protocol} {index}.",
    }


def _fixture(tmp_path: Path, *, segmented_records: int = 4) -> dict[str, object]:
    design = {
        "schema_version": "test",
        "status": "DRAFT_NOT_APPROVED_NO_HUMAN_ACTIVITY",
        "language": "English",
        "random_seed": 20260808,
        "conditions": [
            "human_written_control",
            "ordinary_llm_control",
            "direct_subword_calgacus",
            "rankcloak_ascii_b8",
            "rankcloak_ascii_b16",
            "rankcloak_hex_nibble",
            "rankcloak_segmented_forced_span",
            "rankcloak_segmented_full_message",
        ],
        "prompt_categories": ["casual_conversation"],
        "templates_per_category": 1,
        "eligible_payload_classes": ["sha256_hex"],
        "model_families": [
            "llama_3_1_8b_instruct",
            "qwen_2_5_7b_instruct",
            "mistral_7b_instruct_v0_3",
        ],
        "candidate_replicates_per_stratum": 2,
    }
    prompts = {
        "categories": [
            {
                "category_id": "casual_conversation",
                "templates": [
                    {"prompt_id": "casual_prompt", "text": "Write a casual note."}
                ],
            }
        ]
    }
    control_audit = {
        "pre_recruitment_gate": (
            "BLOCKED_INSUFFICIENT_AUTOMATED_COVERAGE_AND_PENDING_MANUAL_REVIEW"
        )
    }
    instrument = {"status": "DRAFT_NOT_APPROVED", "scales": []}
    paths: dict[str, object] = {}
    for name, value in (
        ("design", design),
        ("prompts", prompts),
        ("control_audit", control_audit),
        ("instrument", instrument),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    power = tmp_path / "power.csv"
    power.write_text("planning_only\ntrue\n", encoding="utf-8")
    paths["power"] = power

    record_paths: list[Path] = []
    for model_index, model in enumerate(MODELS):
        path = tmp_path / f"{model}.jsonl"
        rows: list[dict[str, object]] = []
        if model_index == 0:
            counter = 0
            for record_type, protocol, count in (
                ("ordinary_control", "ordinary_llm_control", 2),
                ("rankcloak_trial", "direct_subword_calgacus", 2),
                ("rankcloak_trial", "nonseg_ascii_b8", 2),
                ("rankcloak_trial", "nonseg_ascii_b16", 2),
                ("rankcloak_trial", "nonseg_hex_nibble_b16", 2),
                ("rankcloak_trial", "segmented_hex_single_topic", segmented_records),
            ):
                for _ in range(count):
                    counter += 1
                    rows.append(
                        _record(
                            model=model,
                            index=counter,
                            record_type=record_type,
                            protocol=protocol,
                        )
                    )
        else:
            rows.append(
                _record(
                    model=model,
                    index=1,
                    record_type="rankcloak_trial",
                    protocol="not_a_stimulus_protocol",
                )
            )
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        record_paths.append(path)
    paths["records"] = record_paths
    return paths


def _build(paths: dict[str, object], output: Path):
    return build_readability_artifacts(
        records_paths=paths["records"],
        design_path=paths["design"],
        prompt_config_path=paths["prompts"],
        control_audit_path=paths["control_audit"],
        instrument_path=paths["instrument"],
        power_paths=[paths["power"]],
        output_dir=output,
        bootstrap_resamples=20,
    )


def test_participant_free_package_is_balanced_hashed_and_explicit(tmp_path: Path):
    paths = _fixture(tmp_path)
    output = tmp_path / "output"
    artifacts = _build(paths, output)
    assert artifacts.summary["candidate_pool_rows"] == 14
    assert artifacts.summary["selected_computational_stimuli"] == 7
    assert artifacts.summary["unavailable_human_control_strata"] == 1
    assert artifacts.summary["planned_inventory_rows"] == 8
    assert artifacts.summary["human_rating_rows"] == 0

    pool = pd.read_csv(output / OUTPUT_FILENAMES["candidate_pool"])
    inventory = pd.read_csv(output / OUTPUT_FILENAMES["inventory"])
    provenance = pd.read_csv(output / OUTPUT_FILENAMES["provenance"])
    metrics = pd.read_csv(output / OUTPUT_FILENAMES["metrics"])
    unavailable = pd.read_csv(output / OUTPUT_FILENAMES["unavailable"])
    assert "message_text" not in pool.columns
    assert inventory["stimulus_blind_id"].nunique() == 8
    assert inventory["message_text"].notna().sum() == 7
    assert provenance["condition"].nunique() == 7
    assert provenance["source_record_sha256"].nunique() == 7
    assert metrics["human_rating_substitute"].eq(False).all()
    assert unavailable["counted_as_human_outcome"].eq(False).all()

    status = json.loads((output / OUTPUT_FILENAMES["status"]).read_text())
    assert status["human_participant_rows"] == 0
    assert status["human_rating_rows"] == 0
    assert status["participant_schedule_emitted"] is False
    assert status["model_identity_alignment_status"] == (
        "draft_design_llama_version_label_mismatch"
    )
    randomization = json.loads(
        (output / OUTPUT_FILENAMES["randomization"]).read_text()
    )
    assert randomization["participant_schedule_emitted"] is False
    assert randomization["attention_checks_scheduled"] is False

    manifest = json.loads((output / OUTPUT_FILENAMES["manifest"]).read_text())
    assert manifest["large_raw_artifacts_copied"] is False
    for key, entry in manifest["outputs"].items():
        assert file_sha256(entry["path"]) == entry["sha256"], key


def test_selection_is_deterministic_for_same_sources(tmp_path: Path):
    paths = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(paths, first)
    _build(paths, second)
    for key in (
        "candidate_pool",
        "inventory",
        "provenance",
        "unavailable",
        "metrics",
        "summary",
        "randomization",
        "status",
    ):
        assert (first / OUTPUT_FILENAMES[key]).read_bytes() == (
            second / OUTPUT_FILENAMES[key]
        ).read_bytes()


def test_insufficient_unique_segmented_candidates_fails_closed(tmp_path: Path):
    paths = _fixture(tmp_path, segmented_records=3)
    with pytest.raises(ReadabilityEvidenceError, match="Insufficient unique candidates"):
        _build(paths, tmp_path / "output")
