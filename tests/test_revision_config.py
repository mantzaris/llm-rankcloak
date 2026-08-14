from collections import Counter
import json

from rankcloak.revision_config import (
    DEFAULT_REVISION_CONFIG_DIR,
    assign_english_prompt,
    build_ablation_trial_plan,
    build_multilingual_trial_plan,
    build_primary_control_plan,
    build_primary_trial_plan,
    flatten_prompt_templates,
    load_revision_config_set,
    missing_model_artifact_ids,
    model_ids,
    prompt_by_id,
    validate_revision_config_set,
    verify_model_artifact_pins,
)
from rankcloak.revision_artifacts import verify_directory_manifest
from rankcloak.revision_payloads import (
    REVISION_HEX_PAYLOAD_CLASSES,
    generate_revision_v1_payloads,
)


def test_revision_config_set_validates_and_reports_exact_counts():
    report = validate_revision_config_set()
    assert report["status"] == "ok"
    assert report["errors"] == []
    assert report["counts"] == {
        "payloads": 480,
        "prompt_templates": 18,
        "primary_rankcloak_trials": 6480,
        "primary_controls": 7920,
        "ablation_unique_rows": 1872,
        "ablation_new_generation_rows": 1728,
        "robustness_additional_decode_only_runs": 3168,
        "multilingual_rankcloak_trials": 576,
    }
    assert report["execution_ready"] is (
        len(report["missing_model_artifact_ids"]) == 0
    )


def test_frozen_config_manifest_verifies_without_unlisted_files():
    manifest = json.loads(
        (DEFAULT_REVISION_CONFIG_DIR / "config_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = verify_directory_manifest(
        DEFAULT_REVISION_CONFIG_DIR,
        manifest,
        require_no_extra_files=True,
    )
    assert report == {
        "status": "ok",
        "verified_file_count": 10,
        "errors": [],
    }
    assert (
        manifest["files_sha256"]
        == "63db256f44c8f94a3d5e81fb69d1f3cdd3af0b88e2dcf77bf19fd02c974eb4df"
    )


def test_roundtrip_filter_has_frozen_preconfirmatory_feasibility_gate():
    config = load_revision_config_set()["ablations"]
    gate = config["feasibility_gates"]["roundtrip_stable_filter_v1"]
    assert gate["minimum_allowed_token_count"] == 1
    assert gate["criterion"] == (
        "safe_text_true_and_isolated_detokenize_retokenize_exact"
    )
    assert gate["on_empty"] == (
        "emit_completed_condition_unavailable_and_propagate_dependent_unavailable"
    )
    assert gate["prefix_conditioned_substitution_permitted"] is False
    expected = config["expected_counts"]
    assert expected["unique_condition_rows"] == 1872
    assert expected["new_generated_rankcloak_texts_planned"] == 1728
    assert expected["new_generated_rankcloak_texts_executable"]["status"] == (
        "model_dependent_observed_count"
    )


def test_model_specs_are_content_pinned():
    configs = load_revision_config_set()
    assert len(model_ids(configs["models"])) == 3
    for model in configs["models"]["models"]:
        assert len(model["artifact_sha256"]) == 64
        assert model["artifact_size_bytes"] > 4_000_000_000
        assert model["filename"].endswith(".gguf")
        assert model["relative_path"].startswith("models/")
    missing = missing_model_artifact_ids(configs["models"])
    assert set(missing).issubset(set(model_ids(configs["models"])))


def test_model_pin_verifier_detects_content_mismatch(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    config = {
        "models": [
            {
                "model_id": "test",
                "relative_path": "model.gguf",
                "artifact_size_bytes": 5,
                "artifact_sha256": "0" * 64,
            }
        ]
    }
    report = verify_model_artifact_pins(
        config, project_root=tmp_path, verify_sha256=True
    )
    assert report["status"] == "error"
    assert report["records"][0]["status"] == "sha256_mismatch"


def test_prompt_registry_and_lookup_are_exact():
    configs = load_revision_config_set()
    prompts = flatten_prompt_templates(configs["prompts"])
    assert len(prompts) == 18
    assert len({row["prompt_id"] for row in prompts}) == 18
    assert set(Counter(row["category_id"] for row in prompts).values()) == {3}
    selected = prompt_by_id("professional_project_update", configs=configs)
    assert selected["category_id"] == "professional_communication"
    assert selected["text"].startswith("Write a concise professional")


def test_balanced_prompt_assignment_per_model_class_and_template():
    configs = load_revision_config_set()
    payloads = generate_revision_v1_payloads()
    blocks = []
    for payload in payloads:
        for model_id in model_ids(configs["models"]):
            prompt = assign_english_prompt(payload, model_id, configs=configs)
            blocks.append((payload, model_id, prompt))

    assert len(blocks) == 1440
    template_counts = Counter(prompt["prompt_id"] for _, _, prompt in blocks)
    assert set(template_counts.values()) == {80}
    hex_counts = Counter(
        prompt["prompt_id"]
        for payload, _, prompt in blocks
        if payload.payload_class in REVISION_HEX_PAYLOAD_CLASSES
    )
    assert set(hex_counts.values()) == {40}
    model_class_category = Counter(
        (model_id, payload.payload_class, prompt["prompt_category"])
        for payload, model_id, prompt in blocks
    )
    assert set(model_class_category.values()) == {10}


def test_primary_plan_protocol_prompt_model_and_split_counts():
    plan = build_primary_trial_plan()
    assert len(plan) == 6480
    assert len({row["trial_id"] for row in plan}) == 6480
    assert Counter(row["protocol_variant"] for row in plan) == {
        "direct_subword_calgacus": 1440,
        "nonseg_ascii_b8": 1440,
        "nonseg_ascii_b16": 1440,
        "nonseg_hex_nibble_b16": 720,
        "segmented_hex_single_topic": 720,
        "segmented_hex_multi_topic": 720,
    }
    assert set(Counter(row["model_id"] for row in plan).values()) == {2160}
    assert set(Counter(row["prompt_id"] for row in plan).values()) == {360}
    assert Counter(row["payload_split"] for row in plan) == {
        "train": 3888,
        "validation": 1296,
        "test": 1296,
    }


def test_primary_controls_include_separate_forced_span_matches():
    primary = build_primary_trial_plan()
    controls = build_primary_control_plan(primary)
    assert len(controls) == 7920
    assert len({row["control_id"] for row in controls}) == 7920
    assert Counter(row["control_view"] for row in controls) == {
        "full_message": 6480,
        "forced_span": 1440,
    }


def test_ablation_plan_deduplicates_canonical_rows():
    plan = build_ablation_trial_plan()
    assert len(plan) == 1872
    assert len({row["trial_id"] for row in plan}) == 1872
    assert sum(bool(row["primary_overlap"]) for row in plan) == 144
    assert sum(bool(row["generation_required"]) for row in plan) == 1728
    assert len(
        {
            (
                row["token_filter"],
                row["leadin_tokens"],
                row["tail_policy"],
                row["segment_size_ranks"],
            )
            for row in plan
        }
    ) == 13


def test_multilingual_plan_has_two_languages_and_two_protocols():
    plan = build_multilingual_trial_plan()
    assert len(plan) == 576
    assert len({row["trial_id"] for row in plan}) == 576
    assert Counter(row["language"] for row in plan) == {
        "es": 288,
        "zh_hans": 288,
    }
    assert Counter(row["protocol_variant"] for row in plan) == {
        "direct_subword_calgacus": 288,
        "nonseg_ascii_b16": 288,
    }
