import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_artifacts import (
    ArtifactIntegrityError,
    build_run_identity_manifest,
    canonical_json_sha256,
    file_sha256,
    initialize_checkpoint,
    record_checkpoint_result,
    write_immutable_json,
    write_immutable_jsonl,
)
from rankcloak.revision_evaluator import (
    EVALUATOR_BY_GENERATOR,
    EVIDENCE_LIMITED,
    EVIDENCE_SMOKE_V3,
    LLAMA_MODEL_ID,
    MISTRAL_MODEL_ID,
    QWEN_MODEL_ID,
    SMOKE_SOURCE_STAGE,
    RevisionEvaluatorError,
    aggregate_segment_scores,
    build_prompt_lookup,
    build_scoring_plan,
    evaluator_for_generator,
    export_auxiliary_timing_summary,
    generator_for_evaluator,
    reconcile_checkpoint,
    relabel_limited_plan,
    run_evaluation_plan,
    score_conditioned_text,
    select_source_stages,
    verify_completed_runner_shard,
)
from rankcloak.revision_compute import load_auxiliary_timings
from rankcloak.revision_config import load_revision_config_set
from rankcloak.revision_runner import (
    EVIDENCE_PRIMARY_V2,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
)
from rankcloak.revision_statistics import validate_feature_rows, validate_trial_results


class FakeEvaluatorModel:
    """Tiny stateful model implementing the llama.cpp methods used by scoring."""

    def __init__(self, model_id=QWEN_MODEL_ID, vocab_size=11):
        self.rankcloak_revision_model_id = model_id
        self.vocab_size = vocab_size
        self.n_tokens = 0
        self.scores = np.zeros((8192, vocab_size), dtype=np.float64)
        self._last = 0

    def token_bos(self):
        return 0

    def tokenize(self, data, add_bos=True):
        values = [1 + (byte % (self.vocab_size - 1)) for byte in bytes(data)]
        return ([0] if add_bos else []) + values

    def reset(self):
        self.n_tokens = 0
        self._last = 0

    def eval(self, token_ids):
        for token_id in token_ids:
            self._last = int(token_id)
            preferred = (self._last + 3) % self.vocab_size
            logits = -np.abs(np.arange(self.vocab_size) - preferred).astype(float)
            self.scores[self.n_tokens] = logits
            self.n_tokens += 1


def rankcloak_record(model_id=LLAMA_MODEL_ID, work_id="trial-a", segmented=True):
    raw_segments = [
        ("prompt alpha", " first text"),
        ("prompt beta", " second text"),
    ] if segmented else [("prompt alpha", " first text")]
    segments = []
    for index, (prompt, text) in enumerate(raw_segments):
        segments.append(
            {
                "segment_index": index,
                "prompt": {
                    "prompt_id": "prompt-{}".format(index),
                    "prompt_category": "category-{}".format(index),
                    "prompt_text": prompt,
                },
                "full_text": text,
            }
        )
    return {
        "record_type": "rankcloak_trial",
        "execution_status": "completed",
        "work_id": work_id,
        "trial_id": work_id,
        "evidence_status": EVIDENCE_PRIMARY_V2,
        "study_phase": "primary_v2_confirmatory",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "model_id": model_id,
        "payload_name": "sha256_000",
        "payload_class": "sha256",
        "payload_split": "test",
        "protocol_variant": "ascii_b8",
        "prompt_id": "prompt-0",
        "prompt_category": "category-0",
        "language": "en",
        "segments": segments,
        "full_text": "\n\n".join(text for _, text in raw_segments),
    }


def control_record(model_id=LLAMA_MODEL_ID, work_id="control-a"):
    prompt_text = (
        "Write a friendly message to a friend about making relaxed plans for the "
        "weekend. Use ordinary conversational English and complete thoughts."
    )
    return {
        "record_type": "ordinary_control",
        "execution_status": "completed",
        "work_id": work_id,
        "control_id": work_id,
        "source_trial_id": "trial-a",
        "evidence_status": EVIDENCE_PRIMARY_V2,
        "study_phase": "ordinary_llm_control_primary_v2",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "model_id": model_id,
        "payload_name": "sha256_000",
        "payload_class": "sha256",
        "payload_split": "test",
        "prompt_id": "casual_weekend_chat",
        "prompt_category": "casual_conversation",
        "language": "en",
        "control_view": "full_message",
        "generation": {"prompt": prompt_text, "text": " ordinary text"},
        "full_text": " ordinary text",
    }


def evaluator_manifest(model_id=QWEN_MODEL_ID):
    digest = "a" * 64
    return {
        "configured_model": {"model_id": model_id, "artifact_sha256": digest},
        "verification": {"actual_sha256": digest},
    }


def test_cli_source_stage_default_is_primary_v2_and_legacy_names_fail_closed():
    assert select_source_stages(None) == ["primary_v2"]
    assert select_source_stages(["smoke_v3"]) == ["smoke_v3"]
    with pytest.raises(RevisionEvaluatorError, match="Unsupported"):
        select_source_stages(["primary"])
    with pytest.raises(RevisionEvaluatorError, match="Unsupported"):
        select_source_stages(["smoke_v2"])



def test_cyclic_mapping_is_total_and_never_same_model():
    assert EVALUATOR_BY_GENERATOR == {
        LLAMA_MODEL_ID: QWEN_MODEL_ID,
        QWEN_MODEL_ID: MISTRAL_MODEL_ID,
        MISTRAL_MODEL_ID: LLAMA_MODEL_ID,
    }
    for generator, evaluator in EVALUATOR_BY_GENERATOR.items():
        assert evaluator_for_generator(generator) == evaluator
        assert generator_for_evaluator(evaluator) == generator
        assert evaluator != generator
    with pytest.raises(RevisionEvaluatorError, match="cyclic mapping"):
        build_scoring_plan(
            {"primary_v2": [rankcloak_record()]},
            evaluator_model_id=QWEN_MODEL_ID,
            generator_model_id=MISTRAL_MODEL_ID,
        )


def test_conditioned_scoring_is_serial_and_prompt_sensitive():
    model = FakeEvaluatorModel()
    first = score_conditioned_text(model, "alpha", "same text", context_limit=100)
    second = score_conditioned_text(model, "omegz", "same text", context_limit=100)
    assert first["evaluator_token_count"] == len("same text")
    assert first["total_nll"] == pytest.approx(-first["total_log_probability"])
    assert first["mean_nll"] == pytest.approx(-first["mean_log_probability"])
    assert first["total_log_probability"] != second["total_log_probability"]
    assert first["tokenization_and_scoring_mode"].startswith("serial_llama_cpp")
    with pytest.raises(RevisionEvaluatorError, match="context limit"):
        score_conditioned_text(model, "too long", "also long", context_limit=3)


def test_segment_aggregation_is_token_weighted_not_mean_of_means():
    scores = [
        {"evaluator_token_count": 2, "total_log_probability": -2.0, "scoring_seconds": 1.0},
        {"evaluator_token_count": 8, "total_log_probability": -24.0, "scoring_seconds": 3.0},
    ]
    result = aggregate_segment_scores(scores)
    assert result["evaluator_token_count"] == 10
    assert result["heldout_evaluator_log_probability"] == pytest.approx(-2.6)
    assert result["heldout_evaluator_mean_nll"] == pytest.approx(2.6)
    assert result["evaluator_tokens_per_second"] == pytest.approx(2.5)


def test_scoring_plan_records_per_segment_prompts_and_limited_evidence():
    source = rankcloak_record()
    plan = build_scoring_plan({"primary_v2": [source]}, QWEN_MODEL_ID)
    assert len(plan) == 1
    assert plan[0]["segment_count"] == 2
    assert [row["prompt_text"] for row in plan[0]["segments"]] == [
        "prompt alpha", "prompt beta"
    ]
    assert plan[0]["prompt_conditioning"].startswith("per_segment")
    limited = relabel_limited_plan(plan, 1)
    assert limited[0]["evidence_status"] == EVIDENCE_LIMITED
    assert limited[0]["original_frozen_evaluation_id"] == plan[0]["evaluation_id"]
    assert limited[0]["evaluation_id"] != plan[0]["evaluation_id"]


def test_smoke_plan_is_disjoint_exploratory_and_cannot_be_pooled_or_limited():
    source = rankcloak_record()
    source["evidence_status"] = EVIDENCE_SMOKE_V3
    smoke = build_scoring_plan({SMOKE_SOURCE_STAGE: [source]}, QWEN_MODEL_ID)
    primary_source = rankcloak_record()
    primary = build_scoring_plan({"primary_v2": [primary_source]}, QWEN_MODEL_ID)

    assert smoke[0]["evidence_status"] == EVIDENCE_SMOKE_V3
    assert smoke[0]["evidence_partition"] == (
        "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling"
    )
    assert smoke[0]["confirmatory_pooling_eligible"] is False
    assert smoke[0]["source_trial_id_raw"] == primary[0]["trial_id"]
    assert smoke[0]["trial_id"] != primary[0]["trial_id"]
    assert smoke[0]["evaluation_id"] != primary[0]["evaluation_id"]
    with pytest.raises(RevisionEvaluatorError, match="isolated"):
        build_scoring_plan(
            {SMOKE_SOURCE_STAGE: [source], "primary_v2": [primary_source]},
            QWEN_MODEL_ID,
        )
    with pytest.raises(RevisionEvaluatorError, match="limit is forbidden"):
        relabel_limited_plan(smoke, 1)


def test_smoke_timing_is_self_hashed_and_revision_compute_compatible(tmp_path):
    source = rankcloak_record()
    source["evidence_status"] = EVIDENCE_SMOKE_V3
    plan = build_scoring_plan({SMOKE_SOURCE_STAGE: [source]}, QWEN_MODEL_ID)
    identifiers = [str(row["evaluation_id"]) for row in plan]
    initialize_checkpoint(
        tmp_path / "checkpoint.json", "test/evaluator/smoke", "c" * 64,
        identifiers,
    )
    result = run_evaluation_plan(
        FakeEvaluatorModel(), plan, evaluator_manifest(), "i" * 64, "c" * 64,
        tmp_path, context_limit=512,
    )
    assert result["completed"] == len(plan)
    write_immutable_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "event": "evaluator_model_loaded",
                "evaluator_model_id": QWEN_MODEL_ID,
                "generator_model_id": LLAMA_MODEL_ID,
                "model_load_seconds": 2.5,
                "gpu_uuid": "GPU-fixture",
            }
        ],
    )
    exported = export_auxiliary_timing_summary(
        tmp_path, plan, QWEN_MODEL_ID, gpu_count=1
    )
    timing_path = Path(exported["path"])
    loaded = load_auxiliary_timings([timing_path])
    assert len(loaded) == 1
    assert loaded[0]["component"] == "evaluator"
    assert loaded[0]["completed_units"] == len(plan)
    assert loaded[0]["confirmatory_pooling_eligible"] is False
    assert loaded[0]["evidence_partition"] == (
        "exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling"
    )
    feature_manifest = json.loads(
        (tmp_path / "features_manifest.json").read_text(encoding="utf-8")
    )
    assert feature_manifest["confirmatory_pooling_eligible"] is False


@pytest.mark.parametrize(
    "language,prompt_id,prompt_category,expected_text",
    [
        (
            "en", "casual_weekend_chat", "casual_conversation",
            "Write a friendly message to a friend about making relaxed plans for the weekend. Use ordinary conversational English and complete thoughts.",
        ),
        (
            "es", "es_casual_conversation_01", "casual_conversation",
            "Escribe un mensaje amistoso a una persona conocida sobre planes tranquilos para el fin de semana. Usa un español cotidiano y termina las ideas de forma natural.",
        ),
        (
            "zh_hans", "zh_hans_casual_conversation_01", "casual_conversation",
            "写一段给朋友的自然消息，谈谈轻松的周末计划。使用日常语言，并把意思表达完整。",
        ),
    ],
)
def test_control_prompt_is_resolved_exactly_from_hashed_frozen_configs(
    language, prompt_id, prompt_category, expected_text
):
    lookup = build_prompt_lookup(load_revision_config_set())
    source = control_record()
    source.update(
        {"language": language, "prompt_id": prompt_id,
         "prompt_category": prompt_category}
    )
    source["generation"].pop("prompt")
    plan = build_scoring_plan(
        {"primary_v2": [source]}, QWEN_MODEL_ID, prompt_lookup=lookup
    )
    assert plan[0]["segments"][0]["prompt_text"] == expected_text
    assert plan[0]["segments"][0]["prompt_source"] == (
        "verified_frozen_config_lookup"
    )
    assert plan[0]["prompt_lookup_sha256"] == lookup["prompt_lookup_sha256"]


def test_run_resume_exports_statistics_ingestible_flat_features(tmp_path):
    plan = build_scoring_plan(
        {"primary_v2": [rankcloak_record(), control_record()]}, QWEN_MODEL_ID,
        prompt_lookup=build_prompt_lookup(load_revision_config_set()),
    )
    identifiers = [row["evaluation_id"] for row in plan]
    initialize_checkpoint(
        tmp_path / "checkpoint.json", "test/evaluator", "c" * 64, identifiers
    )
    model = FakeEvaluatorModel()
    result = run_evaluation_plan(
        model, plan, evaluator_manifest(), "i" * 64, "c" * 64, tmp_path,
        context_limit=512, max_pending=1,
    )
    assert result["completed"] == 1
    assert result["remaining"] == 1
    assert not (tmp_path / "features.jsonl").exists()
    result = run_evaluation_plan(
        model, plan, evaluator_manifest(), "i" * 64, "c" * 64, tmp_path,
        context_limit=512,
    )
    assert result["completed"] == 2
    rows = [json.loads(line) for line in (tmp_path / "features.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["same_model_evaluation"] is False for row in rows)
    assert all(np.isfinite(row["heldout_evaluator_log_probability"]) for row in rows)
    validated, payload_column = validate_feature_rows(pd.DataFrame(rows))
    assert payload_column == "payload_name"
    assert len(validated) == 2
    continuous = [
        json.loads(line)
        for line in (tmp_path / "continuous_quality.jsonl").read_text().splitlines()
    ]
    continuous_validated, continuous_payload = validate_trial_results(
        pd.DataFrame(continuous)
    )
    assert continuous_payload == "payload_name"
    assert len(continuous_validated) == 2


def test_checkpoint_reconciliation_rejects_phantom_completion(tmp_path):
    identifiers = ["eval-a"]
    checkpoint_path = tmp_path / "checkpoint.json"
    initialize_checkpoint(checkpoint_path, "test/evaluator", "c" * 64, identifiers)
    record_checkpoint_result(checkpoint_path, "eval-a", "completed")
    with pytest.raises(ArtifactIntegrityError, match="lacks a durable"):
        reconcile_checkpoint(checkpoint_path, identifiers, [])


def make_completed_runner_shard(tmp_path: Path, config_hash: str) -> Path:
    run_dir = tmp_path / "primary_v2" / LLAMA_MODEL_ID
    run_dir.mkdir(parents=True)
    record = rankcloak_record()
    plan = [
        {
            "work_id": record["work_id"],
            "model_id": LLAMA_MODEL_ID,
            "work_kind": "rankcloak",
            "evidence_status": EVIDENCE_PRIMARY_V2,
            "study_phase": "primary_v2_confirmatory",
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
        }
    ]
    payload_manifest = {"corpus": "fixture"}
    model_manifest = {
        "configured_model": {
            "model_id": LLAMA_MODEL_ID,
            "artifact_sha256": "1" * 64,
        },
        "verification": {
            "status": "ok",
            "expected_sha256": "1" * 64,
            "actual_sha256": "1" * 64,
        },
    }
    source_manifest = {
        "files": [],
        "files_sha256": canonical_json_sha256([]),
    }
    write_immutable_jsonl(run_dir / "plan.jsonl", plan)
    write_immutable_jsonl(run_dir / "records.jsonl", [record])
    write_immutable_json(run_dir / "payload_manifest.json", payload_manifest)
    write_immutable_json(run_dir / "model_manifest.json", model_manifest)
    write_immutable_json(run_dir / "source_manifest.json", source_manifest)
    write_immutable_json(run_dir / "runtime_manifest.json", {"fixture": True})
    write_immutable_json(run_dir / "hardware_manifest.json", {"fixture": True})
    identity = build_run_identity_manifest(
        study_id="revision_v1/primary_v2/{}".format(LLAMA_MODEL_ID),
        config_manifest_sha256=config_hash,
        payload_manifest_sha256=file_sha256(run_dir / "payload_manifest.json"),
        planned_trial_ids=[record["work_id"]],
        model_artifacts=[model_manifest],
        command_line_args=[
            "stage=primary_v2",
            "model_id={}".format(LLAMA_MODEL_ID),
            "protocol_contract_revision={}".format(PROTOCOL_CONTRACT_REVISION),
            "result_schema_revision={}".format(RESULT_SCHEMA_REVISION),
            "source_manifest_sha256={}".format(
                file_sha256(run_dir / "source_manifest.json")
            ),
            "runtime_manifest_sha256={}".format(
                file_sha256(run_dir / "runtime_manifest.json")
            ),
            "hardware_manifest_sha256={}".format(
                file_sha256(run_dir / "hardware_manifest.json")
            ),
        ],
    )
    identity["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
    identity["result_schema_revision"] = RESULT_SCHEMA_REVISION
    identity.pop("run_identity_sha256", None)
    identity["run_identity_sha256"] = canonical_json_sha256(identity)
    write_immutable_json(run_dir / "run_identity.json", identity)
    initialize_checkpoint(
        run_dir / "checkpoint.json", identity["study_id"], config_hash,
        [record["work_id"]],
    )
    record_checkpoint_result(run_dir / "checkpoint.json", record["work_id"], "completed")
    return run_dir


def test_completed_runner_verification_is_content_bound_and_does_not_open_weights(tmp_path):
    config_hash = "c" * 64
    run_dir = make_completed_runner_shard(tmp_path, config_hash)
    records, manifest = verify_completed_runner_shard(
        run_dir, "primary_v2", LLAMA_MODEL_ID, config_hash
    )
    assert len(records) == 1
    assert manifest["generator_artifact_opened_by_evaluator"] is False
    assert manifest["generator_artifact_sha256"] == "1" * 64
    with (run_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"work_id": "intruder", "execution_status": "completed"}) + "\n")
    with pytest.raises(ArtifactIntegrityError, match="unplanned"):
        verify_completed_runner_shard(run_dir, "primary_v2", LLAMA_MODEL_ID, config_hash)


def test_source_record_hash_and_model_identity_are_enforced():
    source = rankcloak_record()
    plan = build_scoring_plan({"primary_v2": [source]}, QWEN_MODEL_ID)
    task = dict(plan[0])
    task["generator_model_id"] = QWEN_MODEL_ID
    model = FakeEvaluatorModel()
    from rankcloak.revision_evaluator import evaluate_task

    with pytest.raises(RevisionEvaluatorError, match="mapping"):
        evaluate_task(model, task, evaluator_manifest(), "i" * 64, "c" * 64)
    wrong_model = FakeEvaluatorModel(model_id=MISTRAL_MODEL_ID)
    with pytest.raises(RevisionEvaluatorError, match="Loaded evaluator identity"):
        evaluate_task(wrong_model, plan[0], evaluator_manifest(), "i" * 64, "c" * 64)
