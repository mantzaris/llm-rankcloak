import hashlib
from collections import Counter

import numpy as np

import rankcloak.revision_runner as revision_runner
from rankcloak.revision_artifacts import initialize_checkpoint
from rankcloak.revision_config import load_revision_config_set
from rankcloak.revision_payloads import make_revision_payload
from rankcloak.revision_runner import (
    EVIDENCE_ABLATION_V2,
    EVIDENCE_LIMITED,
    EVIDENCE_MULTILINGUAL_V2,
    EVIDENCE_PRIMARY_V2,
    EVIDENCE_ROBUSTNESS_V2,
    EVIDENCE_SMOKE,
    EVIDENCE_SMOKE_V3,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
    build_stage_plan,
    derive_control_seed,
    execute_rankcloak_trial,
    execute_robustness_decode,
    execute_robustness_transform,
    generate_length_matched_control,
    load_jsonl_records,
    relabel_limited_plan,
    run_work_plan,
    _frozen_text_transform,
    _frozen_token_transform,
)


class FakeByteModel:
    """Deterministic llama.cpp-shaped byte model; never loads model artifacts."""

    def __init__(self):
        self.n_tokens = 0
        self.scores = np.zeros((8192, 258), dtype=float)

    def token_bos(self):
        return 0

    def token_eos(self):
        return 257

    def n_vocab(self):
        return 258

    def reset(self):
        self.n_tokens = 0

    def eval(self, token_ids):
        for token_id in token_ids:
            row = -np.arange(258, dtype=float) / 96.0
            # The rotation makes the sampler context sensitive while remaining
            # exactly replayable under the same serial schedule.
            row = np.roll(row, (self.n_tokens + int(token_id)) % 31)
            self.scores[self.n_tokens] = row
            self.n_tokens += 1

    def tokenize(self, value, add_bos=True):
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        ids = [int(byte) + 1 for byte in raw]
        return ([0] if add_bos else []) + ids

    def detokenize(self, token_ids):
        return bytes(
            int(token_id) - 1
            for token_id in token_ids
            if 1 <= int(token_id) <= 256
        )


class EmptyIsolatedRoundTripModel(FakeByteModel):
    """Has safe decoded pieces but deliberately no isolated exact retokenization."""

    def tokenize(self, value, add_bos=True):
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        ids = [((int(byte) + 1) % 256) + 1 for byte in raw]
        return ([0] if add_bos else []) + ids


def _task(payload, variant, *, segmented=False, topic_schedule=None):
    codec = {
        "direct_subword_calgacus": ("raw_subword_direct", None),
        "nonseg_ascii_b8": ("ascii_bytes_fixed_radix", 8),
        "nonseg_ascii_b16": ("ascii_bytes_fixed_radix", 16),
        "nonseg_hex_nibble_b16": ("raw_hex_nibbles", 16),
        "segmented_hex_single_topic": ("raw_hex_nibbles", 16),
        "segmented_hex_multi_topic": ("raw_hex_nibbles", 16),
    }[variant]
    return {
        "work_id": "work-{}".format(variant),
        "trial_id": "work-{}".format(variant),
        "work_kind": "rankcloak",
        "evidence_status": "unit_test",
        "study_phase": "unit_test",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "model_id": "fake",
        "payload_name": payload.payload_name,
        "payload_class": payload.payload_class,
        "payload_index": payload.payload_index,
        "payload_split": "train",
        "protocol_variant": variant,
        "representation_name": codec[0],
        "alphabet_size": codec[1],
        "prompt_id": "casual_weekend_chat",
        "prompt_category": "casual_conversation",
        "language": "en",
        "segmented": segmented,
        "topic_schedule": topic_schedule,
        "token_filter": "none",
        "leadin_tokens": 0,
        "tail_policy": "none",
        "segment_size_ranks": 8 if segmented else None,
        "generation_required": True,
        "source_trial_id": None,
        "replay_modes": [
            "saved_token_ids",
            "detokenized_text_retokenized",
            "greedy_leadin_regeneration",
        ],
    }


def _assert_payload_aware_contract(record):
    assert record["protocol_contract_revision"] == PROTOCOL_CONTRACT_REVISION
    assert record["result_schema_revision"] == RESULT_SCHEMA_REVISION


def test_payload_aware_contract_is_emitted_by_fake_execution_paths():
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 0)
    source_task = _task(payload, "nonseg_hex_nibble_b16")
    source_task["replay_modes"] = ["saved_token_ids"]
    source = execute_rankcloak_trial(
        FakeByteModel(), source_task, payload, configs, context_limit=512
    )
    _assert_payload_aware_contract(source)

    control_task = {
        "work_id": "payload-aware-control",
        "control_id": "payload-aware-control",
        "work_kind": "control",
        "source_trial_id": source["trial_id"],
        "evidence_status": "unit_test",
        "study_phase": "ordinary_llm_control_primary_v2",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "model_id": "fake",
        "payload_name": payload.payload_name,
        "payload_class": payload.payload_class,
        "payload_split": "test",
        "prompt_id": source_task["prompt_id"],
        "prompt_category": source_task["prompt_category"],
        "language": "en",
        "control_view": "full_message",
    }
    control = revision_runner.execute_control_trial(
        FakeByteModel(), control_task, source, configs, context_limit=512
    )
    _assert_payload_aware_contract(control)

    reference_task = next(
        row for row in build_stage_plan("ablation_v2")
        if row["work_kind"] == "reference"
    )
    _assert_payload_aware_contract(revision_runner.execute_reference(reference_task))

    robustness_task = {
        "work_id": "payload-aware-robustness-reference",
        "trial_id": "payload-aware-robustness-reference",
        "evidence_status": EVIDENCE_ROBUSTNESS_V2,
        "study_phase": "robustness_v2_confirmatory_supporting",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "robustness_family": "replay_modes",
        "model_id": "fake",
        "source_model_id": "fake",
        "source_stage": "primary_v2",
        "source_trial_id": source["trial_id"],
        "payload_name": payload.payload_name,
        "replay_mode": "saved_token_ids",
        "transformation_id": "unmodified",
    }
    _assert_payload_aware_contract(
        revision_runner.execute_robustness_reference(robustness_task, source)
    )


def test_stage_plans_have_frozen_generation_and_reference_counts():
    primary = build_stage_plan("primary")
    assert len(primary) == 14_400
    assert Counter(row["work_kind"] for row in primary) == {
        "rankcloak": 6_480,
        "control": 7_920,
    }
    assert set(Counter(row["model_id"] for row in primary).values()) == {4_800}

    ablation = build_stage_plan("ablation")
    assert len(ablation) == 1_872
    assert Counter(row["work_kind"] for row in ablation) == {
        "rankcloak": 1_728,
        "reference": 144,
    }

    multilingual = build_stage_plan("multilingual")
    assert len(multilingual) == 1_152
    assert Counter(row["work_kind"] for row in multilingual) == {
        "rankcloak": 576,
        "control": 576,
    }

    robustness = build_stage_plan("robustness")
    assert len(robustness) == 3_744
    assert Counter(row["work_kind"] for row in robustness) == {
        "robustness_decode": 3_168,
        "reference": 432,
        "robustness_transform": 144,
    }
    transforms = [
        row for row in robustness if row["work_kind"] == "robustness_transform"
    ]
    assert {row["model_id"] for row in transforms} == {
        "qwen2_5_7b_instruct_q4_k_m"
    }
    paraphrase_decodes = [
        row
        for row in robustness
        if row.get("transformation_id") == "paraphrase"
        and row["work_kind"] == "robustness_decode"
    ]
    assert len(paraphrase_decodes) == 144
    assert {row["model_id"] for row in paraphrase_decodes} == {
        row["source_model_id"] for row in paraphrase_decodes
    }
    assert {row["transform_work_id"] for row in paraphrase_decodes} == {
        row["work_id"] for row in transforms
    }


def test_smoke_plan_is_balanced_and_disjoint_from_confirmatory_evidence():
    smoke = build_stage_plan("smoke")
    assert len(smoke) == 96
    assert Counter(row["work_kind"] for row in smoke) == {
        "rankcloak": 36,
        "control": 60,
    }
    assert set(Counter(row["model_id"] for row in smoke).values()) == {32}
    rankcloak = [row for row in smoke if row["work_kind"] == "rankcloak"]
    assert set(Counter(row["prompt_category"] for row in rankcloak).values()) == {6}
    assert len({row["prompt_id"] for row in rankcloak}) == 18
    assert {row["payload_class"] for row in rankcloak} == {
        "sha256_hex",
        "hmac_sha256_hex",
        "nonce_96_bit_hex",
        "token_128_bit_hex",
        "uuid_v4",
        "aes256_gcm_base64",
        "chacha20_poly1305_base64",
        "ed25519_signature_base64",
    }
    for model_id in {row["model_id"] for row in rankcloak}:
        model_classes = {
            row["payload_class"]
            for row in rankcloak
            if row["model_id"] == model_id
        }
        assert "ed25519_signature_base64" in model_classes
        assert model_classes - {
            "sha256_hex",
            "hmac_sha256_hex",
            "nonce_96_bit_hex",
            "token_128_bit_hex",
        }
    assert {row["protocol_variant"] for row in rankcloak} == {
        "direct_subword_calgacus",
        "nonseg_ascii_b8",
        "nonseg_ascii_b16",
        "nonseg_hex_nibble_b16",
        "segmented_hex_single_topic",
        "segmented_hex_multi_topic",
    }
    assert {row["evidence_status"] for row in smoke} == {EVIDENCE_SMOKE}
    assert {tuple(row["replay_modes"]) for row in rankcloak} == {
        (
            "saved_token_ids",
            "detokenized_text_retokenized",
            "greedy_leadin_regeneration",
        )
    }
    assert {row.get("token_filter") for row in rankcloak if row["segmented"]} >= {
        "none",
        "safe_text_filter_v1",
        "roundtrip_stable_filter_v1",
    }
    segmented = [row for row in rankcloak if row["segmented"]]
    assert {row.get("tail_policy") for row in segmented} >= {
        "none",
        "fixed_tail40",
        "sentence_tail_min20_max60",
        "dynamic_completion_v1",
    }
    assert {row.get("leadin_tokens") for row in segmented} >= {0, 8, 32}

    smoke_v2 = build_stage_plan("smoke_v2")
    assert len(smoke_v2) == len(smoke)
    assert {row["work_id"] for row in smoke_v2}.isdisjoint(
        {row["work_id"] for row in smoke}
    )
    assert {row["original_smoke_v1_work_id"] for row in smoke_v2} == {
        row["work_id"] for row in smoke
    }
    assert {row["study_phase"] for row in smoke_v2} == {
        "smoke_v2_exploratory"
    }
    v2_rank_ids = {
        row["work_id"] for row in smoke_v2 if row["work_kind"] == "rankcloak"
    }
    assert all(
        row["source_trial_id"] in v2_rank_ids
        for row in smoke_v2
        if row["work_kind"] == "control"
    )


def test_payload_fidelity_superseding_plans_preserve_conditions_and_remap_links():
    pairs = (
        ("smoke_v2", "smoke_v3", EVIDENCE_SMOKE_V3),
        ("primary", "primary_v2", EVIDENCE_PRIMARY_V2),
        ("ablation", "ablation_v2", EVIDENCE_ABLATION_V2),
        ("multilingual", "multilingual_v2", EVIDENCE_MULTILINGUAL_V2),
        ("robustness", "robustness_v2", EVIDENCE_ROBUSTNESS_V2),
    )
    plans = {}
    excluded = {
        "work_id", "trial_id", "control_id", "source_trial_id",
        "source_stage", "transform_work_id",
        "original_confirmatory_trial_id", "evidence_status", "study_phase",
        "protocol_contract_revision", "result_schema_revision",
        "supersedes_work_id", "supersedes_source_trial_id",
        "supersedes_transform_work_id",
        "supersedes_original_confirmatory_trial_id",
    }
    for base_stage, new_stage, evidence_status in pairs:
        base = build_stage_plan(base_stage)
        new = build_stage_plan(new_stage)
        plans[base_stage] = base
        plans[new_stage] = new
        assert len(new) == len(base)
        assert [row["supersedes_work_id"] for row in new] == [
            row["work_id"] for row in base
        ]
        assert {row["work_id"] for row in new}.isdisjoint(
            {row["work_id"] for row in base}
        )
        assert {row["evidence_status"] for row in new} == {evidence_status}
        assert {row["protocol_contract_revision"] for row in new} == {
            PROTOCOL_CONTRACT_REVISION
        }
        assert {row["result_schema_revision"] for row in new} == {
            RESULT_SCHEMA_REVISION
        }
        for old, replacement in zip(base, new):
            assert {key: value for key, value in old.items() if key not in excluded} == {
                key: value for key, value in replacement.items() if key not in excluded
            }

    new_primary_by_old = {
        row["supersedes_work_id"]: row["work_id"]
        for row in plans["primary_v2"]
    }
    new_ablation_by_old = {
        row["supersedes_work_id"]: row["work_id"]
        for row in plans["ablation_v2"]
    }
    new_primary_ids = set(new_primary_by_old.values())
    new_ablation_ids = set(new_ablation_by_old.values())
    smoke_confirmatory_links = [
        row["original_confirmatory_trial_id"]
        for row in plans["smoke_v3"]
        if row["work_kind"] == "rankcloak"
    ]
    assert len(smoke_confirmatory_links) == 36
    assert all(
        trial_id in new_primary_ids | new_ablation_ids
        for trial_id in smoke_confirmatory_links
    )
    assert sum(trial_id in new_primary_ids for trial_id in smoke_confirmatory_links) == 18
    assert sum(trial_id in new_ablation_ids for trial_id in smoke_confirmatory_links) == 18
    for stage in ("smoke_v3", "primary_v2", "multilingual_v2"):
        internal = {row["work_id"] for row in plans[stage]}
        assert all(
            row["source_trial_id"] in internal
            for row in plans[stage]
            if row.get("source_trial_id") is not None
        )
    for row in plans["ablation_v2"]:
        if row.get("source_trial_id") is not None:
            assert row["source_trial_id"] == new_primary_by_old[
                row["supersedes_source_trial_id"]
            ]
    robustness_ids = {row["work_id"] for row in plans["robustness_v2"]}
    for row in plans["robustness_v2"]:
        source_map = (
            new_primary_by_old
            if row["source_stage"] == "primary_v2"
            else new_ablation_by_old
        )
        assert row["source_trial_id"] == source_map[
            row["supersedes_source_trial_id"]
        ]
        if row.get("transform_work_id") is not None:
            assert row["transform_work_id"] in robustness_ids


def test_limited_plan_gets_disjoint_ids_and_nonconfirmatory_label():
    source = build_stage_plan("primary", model_id="llama3_8b_instruct_q4_k_m")[:5]
    limited = relabel_limited_plan(source)
    assert {row["evidence_status"] for row in limited} == {EVIDENCE_LIMITED}
    assert {row["work_id"] for row in limited}.isdisjoint(
        {row["work_id"] for row in source}
    )
    sources = {row["work_id"] for row in limited if row["work_kind"] == "rankcloak"}
    for control in (row for row in limited if row["work_kind"] == "control"):
        assert control["source_trial_id"] in sources


def test_control_seed_and_serial_top_p_generation_are_deterministic():
    seed = derive_control_seed("source-trial", "full_message")
    assert seed == derive_control_seed("source-trial", "full_message")
    assert seed != derive_control_seed("source-trial", "forced_span")
    first = generate_length_matched_control(
        FakeByteModel(), "short prompt", 30, seed, context_limit=256
    )
    second = generate_length_matched_control(
        FakeByteModel(), "short prompt", 30, seed, context_limit=256
    )
    different = generate_length_matched_control(
        FakeByteModel(), "short prompt", 30, seed + 1, context_limit=256
    )
    assert first["token_ids"] == second["token_ids"]
    assert first["text"] == second["text"]
    assert first["token_ids"] != different["token_ids"]
    assert len(first["token_role_mask"]) == 30
    assert set(first["token_role_mask"]) == {"ordinary_control"}


def test_direct_ascii_and_hex_trials_inverse_decode_under_exact_replay():
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 0)
    for variant in (
        "direct_subword_calgacus",
        "nonseg_ascii_b8",
        "nonseg_ascii_b16",
        "nonseg_hex_nibble_b16",
    ):
        record = execute_rankcloak_trial(
            FakeByteModel(),
            _task(payload, variant),
            payload,
            configs,
            context_limit=1024,
        )
        assert record["saved_token_id_replay"]["exact_recovery"] is True
        assert record["saved_token_id_replay"]["decoded"]["exact_recovery"] is True
        assert record["forced_token_count"] == len(
            record["representation"]["expected_ranks"]
        )
        assert all(
            len(segment["token_role_mask"]) == len(segment["full_token_ids"])
            for segment in record["segments"]
        )


def test_confirmatory_task_can_skip_diagnostic_replays():
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 2)
    task = _task(payload, "nonseg_hex_nibble_b16")
    task["replay_modes"] = ["saved_token_ids"]
    record = execute_rankcloak_trial(
        FakeByteModel(), task, payload, configs, context_limit=512
    )
    assert record["saved_token_id_replay"]["exact_recovery"] is True
    assert record["text_retokenization_replay"]["executed"] is False
    assert record["text_retokenization_replay"]["exact_recovery"] is None
    assert record["greedy_leadin_replay"]["executed"] is False
    assert record["greedy_leadin_replay"]["exact_recovery"] is None


def test_segmented_prompt_rotation_and_token_masks_are_persisted():
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 1)
    task = _task(
        payload,
        "segmented_hex_multi_topic",
        segmented=True,
        topic_schedule="deterministic_six_category_rotation",
    )
    task["leadin_tokens"] = 8
    record = execute_rankcloak_trial(
        FakeByteModel(), task, payload, configs, context_limit=512
    )
    assert record["saved_token_id_replay"]["exact_recovery"] is True
    assert record["saved_token_id_replay"]["replay_mode"] == "saved_token_ids"
    assert (
        record["text_retokenization_replay"]["replay_mode"]
        == "detokenized_text_retokenized"
    )
    assert (
        record["greedy_leadin_replay"]["replay_mode"]
        == "greedy_leadin_regeneration"
    )
    assert record["greedy_leadin_replay"]["exact_recovery"] is True
    assert record["segment_count"] == 3
    assert len({segment["prompt"]["prompt_category"] for segment in record["segments"]}) == 3
    assert len(record["token_role_masks"]) == record["segment_count"]
    assert all(
        set(mask) == {"leadin", "forced"}
        for mask in record["token_role_masks"]
    )
    assert record["topic_rotation_rule"].startswith("anchored_at_assigned_category")


def test_append_checkpoint_resume_does_not_duplicate_completed_rows(tmp_path):
    configs = load_revision_config_set()
    payloads = [
        make_revision_payload("nonce_96_bit_hex", 0),
        make_revision_payload("nonce_96_bit_hex", 1),
    ]
    plan = [
        _task(payloads[0], "nonseg_hex_nibble_b16"),
        {
            **_task(payloads[1], "nonseg_hex_nibble_b16"),
            "work_id": "work-second",
            "trial_id": "work-second",
        },
    ]
    work_ids = [row["work_id"] for row in plan]
    initialize_checkpoint(
        tmp_path / "checkpoint.json",
        study_id="unit-test",
        config_manifest_sha256="a" * 64,
        planned_trial_ids=work_ids,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    partial = run_work_plan(
        FakeByteModel(),
        plan,
        payloads,
        configs,
        tmp_path,
        context_limit=512,
        max_pending=1,
    )
    assert partial["completed"] == 1
    completed = run_work_plan(
        FakeByteModel(), plan, payloads, configs, tmp_path, context_limit=512
    )
    assert completed["completed"] == 2
    records = load_jsonl_records(tmp_path / "records.jsonl")
    completed_ids = [
        row["work_id"] for row in records if row["execution_status"] == "completed"
    ]
    assert completed_ids == work_ids
    assert len(completed_ids) == len(set(completed_ids))


def test_empty_roundtrip_filter_is_completed_unavailable_and_control_propagates(
    tmp_path,
):
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 0)
    source = _task(payload, "nonseg_hex_nibble_b16")
    source.update(
        {
            "work_id": "unavailable-source",
            "trial_id": "unavailable-source",
            "model_id": "mistral_7b_instruct_v0_3_q4_k_m",
            "token_filter": "roundtrip_stable_filter_v1",
            "replay_modes": ["saved_token_ids"],
        }
    )
    control = {
        "work_id": "dependent-control",
        "control_id": "dependent-control",
        "work_kind": "control",
        "source_trial_id": source["trial_id"],
        "evidence_status": "unit_test",
        "study_phase": "ordinary_llm_control",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "model_id": source["model_id"],
        "payload_name": payload.payload_name,
        "payload_class": payload.payload_class,
        "payload_split": "train",
        "prompt_id": source["prompt_id"],
        "prompt_category": source["prompt_category"],
        "language": "en",
        "control_view": "full_message",
    }
    initialize_checkpoint(
        tmp_path / "checkpoint.json",
        study_id="unit-test-unavailable",
        config_manifest_sha256="b" * 64,
        planned_trial_ids=[source["work_id"], control["work_id"]],
        timestamp="2026-01-01T00:00:00+00:00",
    )
    result = run_work_plan(
        EmptyIsolatedRoundTripModel(),
        [source, control],
        [payload],
        configs,
        tmp_path,
        context_limit=512,
    )
    assert result["completed"] == 2
    assert result["failed_current"] == 0
    source_record, control_record = load_jsonl_records(tmp_path / "records.jsonl")
    assert source_record["execution_status"] == "completed"
    assert source_record["record_type"] == "condition_unavailable"
    assert source_record["reason_code"] == "empty_isolated_roundtrip_vocabulary"
    assert source_record["safe_count"] > 0
    assert source_record["stable_count"] == 0
    assert source_record["exact_recovery"] is None
    _assert_payload_aware_contract(source_record)
    assert source_record["tokenizer_id"].endswith("::embedded_gguf")
    assert control_record["execution_status"] == "completed"
    assert control_record["record_type"] == "dependent_unavailable"
    assert control_record["reason_code"] == "source_condition_unavailable"
    assert control_record["dependency_root"]["work_id"] == source["work_id"]
    assert control_record["generation_performed"] is False
    _assert_payload_aware_contract(control_record)




def test_representation_rate_estimands_and_quality_endpoint_traces():
    configs = load_revision_config_set()
    sha = make_revision_payload("sha256_hex", 0)

    hex_task = _task(sha, "nonseg_hex_nibble_b16")
    hex_task["replay_modes"] = ["saved_token_ids"]
    hex_record = execute_rankcloak_trial(
        FakeByteModel(), hex_task, sha, configs, context_limit=1024
    )
    assert hex_record["artifact_bit_length"] == 256
    assert hex_record["serialized_payload_bits"] == len(sha.payload_bytes) * 8 == 512
    assert hex_record["representation_source_bits"] == 4 * len(sha.payload_text) == 256
    segment = hex_record["segments"][0]
    assert segment["realized_ranks"] == segment["expected_ranks"]
    assert segment["quality_rank_ceiling"] == 16
    assert len(segment["greedy_log_probabilities"]) == len(segment["expected_ranks"])
    assert len(segment["rank_B_log_probabilities"]) == len(segment["expected_ranks"])

    ascii_task = _task(sha, "nonseg_ascii_b16")
    ascii_task["replay_modes"] = ["saved_token_ids"]
    ascii_record = execute_rankcloak_trial(
        FakeByteModel(), ascii_task, sha, configs, context_limit=1024
    )
    assert ascii_record["representation_source_bits"] == len(sha.payload_bytes) * 8

    direct_task = _task(sha, "direct_subword_calgacus")
    direct_task["replay_modes"] = ["saved_token_ids"]
    direct_record = execute_rankcloak_trial(
        FakeByteModel(), direct_task, sha, configs, context_limit=1024
    )
    assert direct_record["H_bits"] is None
    assert direct_record["representation_source_bits"] is None
    direct_saved = direct_record["saved_token_id_replay"]
    assert direct_saved["exact_recovery"] is direct_saved["exact_payload_recovery"]
    assert direct_saved["exact_recovery"] is True
    assert direct_saved["exact_representation_recovery"] is True
    assert direct_saved["decoded"]["exact_recovery"] is (
        direct_saved["decoded"]["exact_payload_recovery"]
    )
    assert direct_saved["decoded"]["recovery_outcome_semantics"] == (
        "original_serialized_payload_bytes_sha256_v1"
    )
    assert direct_saved["decoded"]["original_payload_sha256"] == (
        direct_record["original_payload_sha256"]
    )
    assert direct_saved["decoded"]["recovered_payload_sha256"] == (
        direct_record["original_payload_sha256"]
    )
    assert direct_record["effective_bits_per_full_token"] is None
    assert direct_record["effective_artifact_bits_per_full_token"] > 0
    assert direct_record["effective_serialized_bits_per_full_token"] > 0
    assert direct_record["segments"][0]["rank_B_log_probabilities"] is None
    assert len(direct_record["segments"][0]["greedy_log_probabilities"]) == len(
        direct_record["segments"][0]["expected_ranks"]
    )

    base64_payload = make_revision_payload("ed25519_signature_base64", 0)
    base64_task = _task(base64_payload, "nonseg_ascii_b8")
    base64_task["replay_modes"] = ["saved_token_ids"]
    base64_record = execute_rankcloak_trial(
        FakeByteModel(), base64_task, base64_payload, configs, context_limit=2048
    )
    assert base64_record["representation_source_bits"] == len(
        base64_payload.payload_bytes
    ) * 8
    assert base64_record["serialized_payload_bits"] != base64_record[
        "artifact_bit_length"
    ]


def test_payload_mismatch_is_not_classified_as_rank_mismatch(monkeypatch):
    configs = load_revision_config_set()
    payload = make_revision_payload("sha256_hex", 0)
    task = _task(payload, "direct_subword_calgacus")
    task["replay_modes"] = ["saved_token_ids"]
    real_decode = revision_runner.decode_representation

    def payload_mismatch(model, representation, ranks):
        decoded = dict(real_decode(model, representation, ranks))
        decoded.update(
            {
                "exact_recovery": False,
                "exact_payload_recovery": False,
                "exact_representation_recovery": True,
                "recovered_payload_sha256": "0" * 64,
            }
        )
        return decoded

    monkeypatch.setattr(revision_runner, "decode_representation", payload_mismatch)
    record = execute_rankcloak_trial(
        FakeByteModel(), task, payload, configs, context_limit=1024
    )
    saved = record["saved_token_id_replay"]
    assert saved["all_segment_ranks_exact"] is True
    assert saved["exact_representation_recovery"] is True
    assert saved["exact_payload_recovery"] is False
    assert saved["exact_recovery"] is saved["exact_payload_recovery"]
    assert saved["failure"]["failure_category"] == (
        "inverse_payload_fidelity_mismatch"
    )
    assert saved["failure"]["first_rank_divergence"]["diverged"] is False
    assert saved["failure"]["recovery_outcome_semantics"] == (
        "original_serialized_payload_bytes_sha256_v1"
    )


def test_frozen_robustness_positions_use_declared_eligible_sets_and_seed():
    trial_id = "source-trial-17"
    value = " A-\tB9 "
    digest = int.from_bytes(
        hashlib.sha256((trial_id + "character_deletion").encode("utf-8")).digest(),
        "big",
    )
    eligible = [index for index, character in enumerate(value) if not character.isspace()]
    deleted, position, count, digest_hex = _frozen_text_transform(
        value, trial_id, "character_deletion"
    )
    assert position == eligible[digest % len(eligible)]
    assert not value[position].isspace()
    assert deleted == value[:position] + value[position + 1 :]
    assert count == len(eligible)
    assert digest_hex == hashlib.sha256(
        (trial_id + "character_deletion").encode("utf-8")
    ).hexdigest()

    substituted, sub_position, _, _ = _frozen_text_transform(
        value, trial_id, "character_substitution"
    )
    assert value[sub_position].isalnum()
    assert substituted != value

    token_ids = [10, 11, 12, 13, 14]
    transformed, token_position, eligible_count, _ = _frozen_token_transform(
        token_ids, trial_id, "token_deletion"
    )
    assert token_position in {1, 2, 3}
    assert eligible_count == 3
    assert transformed == token_ids[:token_position] + token_ids[token_position + 1 :]


def test_paraphrase_is_generated_as_artifact_then_retokenized_by_source_model():
    configs = load_revision_config_set()
    payload = make_revision_payload("nonce_96_bit_hex", 3)
    source_task = _task(
        payload,
        "segmented_hex_multi_topic",
        segmented=True,
        topic_schedule="deterministic_six_category_rotation",
    )
    source_task["replay_modes"] = ["saved_token_ids"]
    source = execute_rankcloak_trial(
        FakeByteModel(), source_task, payload, configs, context_limit=1024
    )
    transform_work_id = "transform-work"
    transformation = {
        "transformation_id": "paraphrase",
        "operation": "greedy_deterministic_rewrite",
        "model_id": "fake",
        "prompt": "Rewrite and return only the rewrite.\n\n{text}",
    }
    transform_task = {
        "work_id": transform_work_id,
        "trial_id": transform_work_id,
        "evidence_status": "unit_test",
        "study_phase": "robustness_transformation_generation",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "robustness_family": "raw_transmission_transform",
        "transformation_id": "paraphrase",
        "transformation": transformation,
        "model_id": "fake",
        "source_model_id": "fake",
        "source_stage": "primary",
        "source_trial_id": source["trial_id"],
    }
    transform_record = execute_robustness_transform(
        FakeByteModel(), transform_task, source, context_limit=1024
    )
    assert transform_record["record_type"] == "robustness_transform"
    assert transform_record["decode_performed"] is False
    _assert_payload_aware_contract(transform_record)
    # Foreign-model token IDs are intentionally unusable; decode must consume text.
    for output in transform_record["segment_outputs"]:
        output["token_ids"] = [999_999]
    decode_task = {
        "work_id": "decode-work",
        "trial_id": "decode-work",
        "evidence_status": "unit_test",
        "study_phase": "robustness_confirmatory_supporting",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "robustness_family": "raw_transmission",
        "model_id": "fake",
        "source_model_id": "fake",
        "source_stage": "primary",
        "source_trial_id": source["trial_id"],
        "payload_name": payload.payload_name,
        "payload_class": payload.payload_class,
        "payload_split": "test",
        "replay_mode": "transformed_text_retokenized",
        "transformation_id": "paraphrase",
        "transform_work_id": transform_work_id,
        "transformation_model_id": "fake",
    }
    decoded = execute_robustness_decode(
        FakeByteModel(),
        decode_task,
        source,
        payload,
        transformation_record=transform_record,
        context_limit=1024,
    )
    assert decoded["record_type"] == "robustness_decode"
    _assert_payload_aware_contract(decoded)
    assert decoded["transformation_record_sha256"]
    assert all(
        999_999 not in segment["observed_full_token_ids"]
        for segment in decoded["segment_outcomes"]
    )
