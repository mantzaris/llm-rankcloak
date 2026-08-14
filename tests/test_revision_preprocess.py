import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_artifacts import (
    build_run_identity_manifest,
    canonical_json_bytes,
    file_sha256,
)
from rankcloak.revision_detection import normalize_detector_frame
from rankcloak.revision_preprocess import (
    EVIDENCE_ABLATION,
    EVIDENCE_PRIMARY,
    EVIDENCE_ROBUSTNESS,
    EVIDENCE_SMOKE,
    RevisionPreprocessError,
    preprocess_revision_results,
)
from rankcloak.revision_statistics import (
    validate_feature_rows,
    validate_runtime_results,
    validate_trial_results,
)


MODEL_ID = "fixture_model"
MODEL_HASH = "1" * 64
CONFIG_HASH = "2" * 64
CORPUS_HASH = "3" * 64
PROTOCOL_CONTRACT_REVISION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
RECOVERY_SEMANTICS = "original_serialized_payload_bytes_sha256_v1"
STAGE_PHASES = {
    "smoke_v3": {
        "rankcloak": "smoke_v3_exploratory",
        "control": "ordinary_llm_control_smoke_v3",
    },
    "primary_v2": {
        "rankcloak": "primary_v2_confirmatory",
        "control": "ordinary_llm_control_primary_v2",
    },
    "ablation_v2": {
        "rankcloak": "ablation_v2_confirmatory",
        "reference": "ablation_v2_confirmatory",
    },
    "multilingual_v2": {
        "rankcloak": "multilingual_v2_secondary",
        "control": "ordinary_llm_control_multilingual_v2",
    },
    "robustness_v2": {
        "robustness_decode": "robustness_v2_confirmatory_supporting",
        "reference": "robustness_v2_confirmatory_supporting",
        "robustness_transform": "robustness_v2_transformation_generation",
    },
}


def _json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonl(path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _source_manifest():
    files = [
        {
            "path": "rankcloak/revision_runner.py",
            "size_bytes": 123,
            "sha256": "4" * 64,
        }
    ]
    return {
        "schema_version": "1.0",
        "manifest_type": "revision_runner_source",
        "files": files,
        "files_sha256": __import__("hashlib").sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _model_manifest(model_id=MODEL_ID):
    return {
        "schema_version": "1.0",
        "configured_model": {
            "model_id": model_id,
            "artifact_sha256": MODEL_HASH,
            "artifact_size_bytes": 10,
            "relative_path": "models/fixture.gguf",
        },
        "execution_policy": {"backend": "fixture"},
        "verification": {
            "model_id": model_id,
            "status": "ok",
            "expected_size_bytes": 10,
            "actual_size_bytes": 10,
            "expected_sha256": MODEL_HASH,
            "actual_sha256": MODEL_HASH,
            "sha256_checked": True,
        },
    }


def _payload_manifest():
    return {
        "schema_version": "1.0",
        "manifest_type": "revision_v1_public_payload_corpus",
        "corpus_id": "fixture_corpus",
        "corpus_sha256": CORPUS_HASH,
        "payload_count": 1,
        "records": [
            {
                "payload_name": "payload-001",
                "payload_class": "sha256_hex",
                "payload_text": "00aa",
            }
        ],
    }


def _rank_task(work_id="rank-1", evidence=EVIDENCE_PRIMARY):
    return {
        "work_id": work_id,
        "trial_id": work_id,
        "work_kind": "rankcloak",
        "evidence_status": evidence,
        "study_phase": "primary_confirmatory",
        "model_id": MODEL_ID,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "protocol_variant": "nonseg_ascii_b8",
        "representation_name": "ascii_bytes_fixed_radix",
        "alphabet_size": 8,
        "token_filter": "safe_text_filter_v1",
        "tail_policy": "fixed_tail40",
        "leadin_tokens": 0,
        "segmented": False,
    }


def _failure(trial_id="rank-1"):
    return {
        "failure_category": "detokenized_text_retokenization",
        "first_rank_divergence": {
            "diverged": True,
            "position_zero_based": 1,
            "expected_value": 2,
            "observed_value": 3,
            "expected_length": 2,
            "observed_length": 2,
        },
        "first_token_divergence": {
            "diverged": True,
            "position_zero_based": 1,
            "expected_value": 11,
            "observed_value": 12,
            "expected_length": 4,
            "observed_length": 4,
        },
        "expected_token_id": 11,
        "recovered_token_id": 12,
        "expected_rank": 2,
        "recovered_rank": 3,
        "context_sha256": "5" * 64,
        "boundary_start": 0,
        "boundary_stop": 2,
        "boundary_start_offset": 0,
        "boundary_end_offset": 2,
        "first_differing_position": 1,
        "trial_id": trial_id,
        "transformation_id": "unmodified",
        "replay_mode": "detokenized_text_retokenized",
        "segment_index": 0,
    }


def _rank_record(work_id="rank-1", evidence=EVIDENCE_PRIMARY, attempt=2):
    segment = {
        "segment_index": 0,
        "prompt": {
            "prompt_id": "prompt-1",
            "prompt_category": "casual",
            "prompt_text": "Write a short casual note.",
        },
        "forced_token_ids": [10, 11],
        "tail_token_ids": [12, 13],
        "leadin_token_ids": [],
        "full_token_ids": [10, 11, 12, 13],
        "forced_text": "Forced sentence.",
        "tail_text": " Natural tail.",
        "full_text": "Forced sentence. Natural tail.",
        "forced_log_probabilities": [-2.0, -2.5],
        "tail_log_probabilities": [-0.2, -0.3],
        "leadin_log_probabilities": [],
    }
    return {
        "schema_version": "1.0",
        "record_type": "rankcloak_trial",
        "work_id": work_id,
        "trial_id": work_id,
        "evidence_status": evidence,
        "study_phase": "primary_confirmatory",
        "model_id": MODEL_ID,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "protocol_variant": "nonseg_ascii_b8",
        "representation": {"name": "ascii_b8", "expected_ranks": [1, 2]},
        "H_bits": 32,
        "alphabet_size_B": 8,
        "segmented": False,
        "segment_count": 1,
        "segment_size_ranks": None,
        "token_filter": "safe_text_filter_v1",
        "tail_policy": "fixed_tail40",
        "leadin_tokens": 0,
        "segments": [segment],
        "full_text": segment["full_text"],
        "forced_text": segment["forced_text"],
        "forced_token_count": 2,
        "tail_token_count": 2,
        "full_token_count": 4,
        "effective_bits_per_full_token": 8.0,
        "cover_tokens_per_payload_display_byte": 1.0,
        "saved_token_id_replay": {
            "replay_mode": "saved_token_ids",
            "all_segment_ranks_exact": True,
            "exact_payload_recovery": True,
            "exact_recovery": True,
            "decoded": {
                "recovery_outcome_semantics": RECOVERY_SEMANTICS,
                "exact_payload_recovery": True,
                "exact_recovery": True,
                "original_payload_sha256": "a" * 64,
                "recovered_payload_sha256": "a" * 64,
            },
            "failure": None,
        },
        "text_retokenization_replay": {
            "replay_mode": "detokenized_text_retokenized",
            "executed": True,
            "all_segment_ranks_exact": False,
            "exact_payload_recovery": False,
            "exact_recovery": False,
            "decoded": {
                "recovery_outcome_semantics": RECOVERY_SEMANTICS,
                "exact_payload_recovery": False,
                "exact_recovery": False,
                "original_payload_sha256": "a" * 64,
                "recovered_payload_sha256": "b" * 64,
            },
            "failure": _failure(work_id),
        },
        "greedy_leadin_replay": {
            "replay_mode": "greedy_leadin_regeneration",
            "executed": False,
            "exact_recovery": None,
            "failure": None,
        },
        "quality": {
            "mean_forced_token_log_probability": -2.25,
            "mean_tail_token_log_probability": -0.25,
        },
        "timing": {
            "representation_seconds": 0.1,
            "filter_setup_seconds": 0.2,
            "generation_seconds": 1.0,
            "encoding_seconds": 1.3,
            "saved_token_id_replay_seconds": 0.4,
            "supported_decoding_seconds": 0.5,
            "cover_tokens_per_generation_second": 4.0,
            "payload_bits_per_encoding_second": 32 / 1.3,
            "forced_tokens_per_supported_decoding_second": 4.0,
        },
        "execution_seconds": 1.8,
        "execution_status": "completed",
        "attempt_index": attempt,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }


def _control_task(evidence=EVIDENCE_PRIMARY):
    return {
        "work_id": "control-1",
        "control_id": "control-1",
        "source_trial_id": "rank-1",
        "work_kind": "control",
        "evidence_status": evidence,
        "study_phase": "ordinary_llm_control",
        "model_id": MODEL_ID,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "control_view": "full_message",
    }


def _control_record(evidence=EVIDENCE_PRIMARY):
    return {
        "schema_version": "1.0",
        "record_type": "ordinary_control",
        "work_id": "control-1",
        "control_id": "control-1",
        "source_trial_id": "rank-1",
        "evidence_status": evidence,
        "study_phase": "ordinary_llm_control",
        "model_id": MODEL_ID,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "control_view": "full_message",
        "generation": {
            "token_ids": [20, 21, 22, 23],
            "token_log_probabilities": [-0.5, -0.4, -0.6, -0.5],
            "target_token_count": 4,
            "text": "Ordinary control sentence.",
        },
        "full_text": "Ordinary control sentence.",
        "full_token_count": 4,
        "execution_seconds": 0.8,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }


def _execution_failure(evidence=EVIDENCE_PRIMARY):
    return {
        "schema_version": "1.0",
        "record_type": "execution_failure",
        "work_id": "rank-1",
        "work_kind": "rankcloak",
        "model_id": MODEL_ID,
        "evidence_status": evidence,
        "execution_status": "failed",
        "attempt_index": 1,
        "failed_at": "2026-08-08T00:00:00+00:00",
        "error": {"type": "FixtureError", "message": "first attempt failed"},
    }


def _write_shard(path, *, stage, evidence, plan, records):
    path.mkdir(parents=True)
    payload = _payload_manifest()
    model = _model_manifest()
    phase_by_kind = STAGE_PHASES[stage]
    plan = list(plan)
    for row in plan:
        row["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
        row["result_schema_revision"] = RESULT_SCHEMA_REVISION
        row["study_phase"] = phase_by_kind[row["work_kind"]]
    planned = {row["work_id"]: row for row in plan}
    records = list(records)
    for row in records:
        row["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
        row["result_schema_revision"] = RESULT_SCHEMA_REVISION
        row["study_phase"] = planned[row["work_id"]]["study_phase"]
    records_by_trial = {
        row["trial_id"]: row for row in records if row.get("trial_id")
    }
    records_by_work = {row["work_id"]: row for row in records}
    for row in records:
        source = records_by_trial.get(row.get("source_trial_id"))
        if source is not None:
            row["source_record_sha256"] = hashlib.sha256(
                canonical_json_bytes(source)
            ).hexdigest()
        transform = records_by_work.get(row.get("transform_work_id"))
        if transform is not None:
            row["transformation_record_sha256"] = hashlib.sha256(
                canonical_json_bytes(transform)
            ).hexdigest()
    _jsonl(path / "plan.jsonl", plan)
    _jsonl(path / "records.jsonl", records)
    _json(path / "payload_manifest.json", payload)
    _json(path / "model_manifest.json", model)
    _json(path / "source_manifest.json", _source_manifest())
    _json(
        path / "hardware_manifest.json",
        {
            "schema_version": "1.0",
            "machine": "fixture",
            "selected_gpu_uuid": "GPU-fixture",
            "gpu_inventory": [],
        },
    )
    _jsonl(
        path / "events.jsonl",
        [
            {
                "event": "model_loaded",
                "model_id": MODEL_ID,
                "model_load_seconds": 2.0,
            }
        ],
    )
    identity = build_run_identity_manifest(
        study_id=f"fixture_corpus/{stage}/{MODEL_ID}",
        config_manifest_sha256=CONFIG_HASH,
        payload_manifest_sha256=file_sha256(path / "payload_manifest.json"),
        planned_trial_ids=[row["work_id"] for row in plan],
        model_artifacts=[model],
        command_line_args=[
            f"stage={stage}",
            f"model_id={MODEL_ID}",
            f"evidence_status={evidence}",
            f"protocol_contract_revision={PROTOCOL_CONTRACT_REVISION}",
            f"result_schema_revision={RESULT_SCHEMA_REVISION}",
        ],
    )
    identity["protocol_contract_revision"] = PROTOCOL_CONTRACT_REVISION
    identity["result_schema_revision"] = RESULT_SCHEMA_REVISION
    identity_body = dict(identity)
    identity_body.pop("run_identity_sha256")
    identity["run_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(identity_body)
    ).hexdigest()
    _json(path / "run_identity.json", identity)
    return path


def _primary_shard(tmp_path, name="primary", evidence=EVIDENCE_PRIMARY):
    rank_task = _rank_task(evidence=evidence)
    control_task = _control_task(evidence=evidence)
    records = [
        _execution_failure(evidence=evidence),
        _rank_record(evidence=evidence),
        _control_record(evidence=evidence),
    ]
    return _write_shard(
        tmp_path / name,
        stage="smoke_v3" if evidence == EVIDENCE_SMOKE else "primary_v2",
        evidence=evidence,
        plan=[rank_task, control_task],
        records=records,
    )


def _direct_payload_mismatch_shard(tmp_path, name, mutate=None):
    task = _rank_task(work_id=name)
    task.update(
        {
            "protocol_variant": "direct_subword_calgacus",
            "representation_name": "raw_subword_direct",
        }
    )
    record = _rank_record(work_id=name, attempt=1)
    record["protocol_variant"] = "direct_subword_calgacus"
    record["representation"] = {
        "name": "direct_subword",
        "expected_ranks": [1, 2],
    }
    saved = record["saved_token_id_replay"]
    saved.update(
        {
            "all_segment_ranks_exact": True,
            "exact_payload_recovery": False,
            "exact_recovery": False,
            "failure": {
                **_failure(name),
                "failure_category": "payload_fidelity_mismatch",
                "replay_mode": "saved_token_ids",
            },
        }
    )
    saved["decoded"].update(
        {
            "exact_payload_recovery": False,
            "exact_recovery": False,
            "recovered_payload_sha256": "b" * 64,
        }
    )
    if mutate is not None:
        mutate(task, record)
    return _write_shard(
        tmp_path / name,
        stage="primary_v2",
        evidence=EVIDENCE_PRIMARY,
        plan=[task],
        records=[record],
    )


def test_direct_payload_mismatch_is_not_misreported_as_rank_recovery(tmp_path):
    shard = _direct_payload_mismatch_shard(tmp_path, "direct-mismatch")
    output = tmp_path / "direct-flat"
    preprocess_revision_results(run_dirs=[shard], output_dir=output)
    trials = pd.read_csv(output / "trials.csv")
    saved = trials[trials["replay_mode"].eq("saved_token_ids")].iloc[0]
    assert saved["protocol_contract_revision"] == PROTOCOL_CONTRACT_REVISION
    assert saved["result_schema_revision"] == RESULT_SCHEMA_REVISION
    assert saved["exact_rank_replay"] == 1
    assert saved["exact_payload_recovery"] == 0
    assert saved["exact_recovery"] == 0
    assert saved["recovery_outcome_semantics"] == RECOVERY_SEMANTICS
    manifest = json.loads(
        (output / "preprocessing_output_manifest.json").read_text()
    )
    contract = manifest["invariants"]["payload_fidelity_contract"]
    assert contract["direct_rows"] == 2
    assert contract["direct_rows_contract_verified"] == 2
    assert contract["exact_rank_replay_role"] == "diagnostic_only"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task, record: record["saved_token_id_replay"].update(
                {"exact_recovery": True}
            ),
            "compatibility alias differs",
        ),
        (
            lambda task, record: record["saved_token_id_replay"]["decoded"].pop(
                "recovery_outcome_semantics"
            ),
            "lacks recovery_outcome_semantics",
        ),
        (
            lambda task, record: record["saved_token_id_replay"].pop(
                "all_segment_ranks_exact"
            ),
            "lacks exact_rank_replay",
        ),
    ],
)
def test_direct_payload_contract_failures_are_rejected_before_output(
    tmp_path, mutation, message
):
    shard = _direct_payload_mismatch_shard(tmp_path, "invalid-direct", mutation)
    output = tmp_path / "invalid-flat"
    with pytest.raises(RevisionPreprocessError, match=message):
        preprocess_revision_results(run_dirs=[shard], output_dir=output)
    assert not output.exists()


def test_missing_protocol_revision_and_legacy_stage_are_ineligible(tmp_path):
    missing = _direct_payload_mismatch_shard(tmp_path, "missing-contract")
    records_path = missing / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0].pop("protocol_contract_revision")
    _jsonl(records_path, records)
    with pytest.raises(RevisionPreprocessError, match="lacks the frozen protocol"):
        preprocess_revision_results(
            run_dirs=[missing], output_dir=tmp_path / "missing-contract-flat"
        )

    missing_schema = _direct_payload_mismatch_shard(tmp_path, "missing-result-schema")
    schema_records_path = missing_schema / "records.jsonl"
    schema_records = [
        json.loads(line) for line in schema_records_path.read_text().splitlines()
    ]
    schema_records[0].pop("result_schema_revision")
    _jsonl(schema_records_path, schema_records)
    with pytest.raises(RevisionPreprocessError, match="lacks the frozen result schema"):
        preprocess_revision_results(
            run_dirs=[missing_schema],
            output_dir=tmp_path / "missing-result-schema-flat",
        )

    legacy = _direct_payload_mismatch_shard(tmp_path, "legacy-stage")
    identity_path = legacy / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["study_id"] = identity["study_id"].replace(
        "/primary_v2/", "/primary/"
    )
    identity["command_line_args"] = [
        "stage=primary" if value == "stage=primary_v2" else value
        for value in identity["command_line_args"]
    ]
    body = dict(identity)
    body.pop("run_identity_sha256")
    identity["run_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    _json(identity_path, identity)
    with pytest.raises(RevisionPreprocessError, match="invalid for stage"):
        preprocess_revision_results(
            run_dirs=[legacy], output_dir=tmp_path / "legacy-stage-flat"
        )


def test_preprocess_flattens_runner_records_and_matches_downstream_contracts(tmp_path):
    shard = _primary_shard(tmp_path)
    output = tmp_path / "flat"
    artifacts = preprocess_revision_results(run_dirs=[shard], output_dir=output)

    assert artifacts.row_counts == {
        "trials": 2,
        "features": 3,
        "runtime": 3,
        "failures": 2,
        "detector": 2,
        "unavailable": 0,
    }
    trials = pd.read_csv(output / "trials.csv")
    features = pd.read_csv(output / "features.csv")
    runtime = pd.read_csv(output / "runtime.csv")
    failures = pd.read_csv(output / "failures.csv")
    detector = pd.read_json(output / "detector_corpus.jsonl", lines=True)

    validate_trial_results(trials)
    validate_feature_rows(features)
    validate_runtime_results(runtime)
    normalized = normalize_detector_frame(detector)
    assert set(trials["replay_mode"]) == {
        "saved_token_ids",
        "detokenized_text_retokenized",
    }
    assert set(trials["protocol_contract_revision"]) == {
        PROTOCOL_CONTRACT_REVISION
    }
    assert set(trials["result_schema_revision"]) == {RESULT_SCHEMA_REVISION}
    assert set(features["protocol_contract_revision"]) == {
        PROTOCOL_CONTRACT_REVISION
    }
    assert set(features["result_schema_revision"]) == {RESULT_SCHEMA_REVISION}
    assert set(trials["recovery_outcome_semantics"]) == {RECOVERY_SEMANTICS}
    assert trials["exact_recovery"].eq(trials["exact_payload_recovery"]).all()
    saved = trials[trials["replay_mode"].eq("saved_token_ids")].iloc[0]
    retokenized = trials[
        trials["replay_mode"].eq("detokenized_text_retokenized")
    ].iloc[0]
    assert saved["exact_rank_replay"] == 1
    assert saved["exact_payload_recovery"] == 1
    assert retokenized["exact_rank_replay"] == 0
    assert retokenized["exact_payload_recovery"] == 0
    assert set(features["source_type"]) == {"rankcloak", "ordinary_llm_control"}
    assert set(features["view"]) == {"forced_span", "full_message"}
    assert set(normalized["label"]) == {0, 1}
    assert set(detector["codec_id"]) == set(detector["protocol_variant"])
    assert "representation_name" in detector.columns
    assert normalized["payload_group_id"].nunique() == 1
    assert set(failures["divergence_fields_availability"]) == {
        "recorded",
        "not_applicable",
    }
    assert failures.loc[
        failures["divergence_fields_availability"] == "recorded", "context_sha256"
    ].str.len().eq(64).all()
    assert runtime["peak_gpu_memory_mib"].isna().all()
    assert runtime["peak_gpu_memory_availability"].eq(
        "unavailable_not_recorded_by_runner_v1"
    ).all()
    output_manifest = json.loads((output / "preprocessing_output_manifest.json").read_text())
    assert output_manifest["invariants"]["missing_values_imputed"] is False
    assert output_manifest["invariants"]["detector_pair_count"] == 1
    assert output_manifest["invariants"]["payload_fidelity_contract"] == {
        "contract_version": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "semantics": RECOVERY_SEMANTICS,
        "primary_outcome": "exact_payload_recovery",
        "compatibility_alias": "exact_recovery",
        "alias_equality_validated": True,
        "exact_rank_replay_role": "diagnostic_only",
        "direct_rows": 0,
        "direct_rows_contract_verified": 0,
    }


def test_unavailable_conditions_are_counted_but_excluded_from_estimands(tmp_path):
    rank_task = _rank_task(work_id="unavailable-rank", evidence=EVIDENCE_SMOKE)
    rank_task.update(
        {
            "study_phase": "smoke_v2_exploratory",
            "token_filter": "roundtrip_stable_filter_v1",
        }
    )
    control_task = _control_task(evidence=EVIDENCE_SMOKE)
    control_task.update(
        {
            "work_id": "unavailable-control",
            "control_id": "unavailable-control",
            "source_trial_id": "unavailable-rank",
            "study_phase": "ordinary_llm_control",
        }
    )
    tokenizer_id = "fixture/repo::fixture.gguf::embedded_gguf"
    root = {
        "work_id": "unavailable-rank",
        "trial_id": "unavailable-rank",
        "record_type": "condition_unavailable",
        "reason_code": "empty_isolated_roundtrip_vocabulary",
        "reason": "No isolated stable safe token.",
        "model_id": MODEL_ID,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact_sha256": MODEL_HASH,
        "safe_count": 31_464,
        "stable_count": 0,
    }
    rank_record = {
        "schema_version": "1.0",
        "record_type": "condition_unavailable",
        "work_id": "unavailable-rank",
        "trial_id": "unavailable-rank",
        "work_kind": "rankcloak",
        "evidence_status": EVIDENCE_SMOKE,
        "study_phase": "smoke_v2_exploratory",
        "model_id": MODEL_ID,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact_sha256": MODEL_HASH,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "protocol_variant": "nonseg_ascii_b8",
        "token_filter": "roundtrip_stable_filter_v1",
        "reason_code": "empty_isolated_roundtrip_vocabulary",
        "reason": "No isolated stable safe token.",
        "safe_count": 31_464,
        "stable_count": 0,
        "vocabulary_size": 32_768,
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    control_record = {
        "schema_version": "1.0",
        "record_type": "dependent_unavailable",
        "work_id": "unavailable-control",
        "trial_id": "unavailable-control",
        "control_id": "unavailable-control",
        "source_trial_id": "unavailable-rank",
        "source_record_sha256": hashlib.sha256(
            canonical_json_bytes(rank_record)
        ).hexdigest(),
        "work_kind": "control",
        "evidence_status": EVIDENCE_SMOKE,
        "study_phase": "ordinary_llm_control",
        "model_id": MODEL_ID,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact_sha256": MODEL_HASH,
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "prompt_id": "prompt-1",
        "prompt_category": "casual",
        "language": "en",
        "reason_code": "source_condition_unavailable",
        "reason": "Required source is unavailable.",
        "dependency_role": "rankcloak control source",
        "dependency_record_type": "condition_unavailable",
        "dependency_root": root,
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    shard = _write_shard(
        tmp_path / "smoke-v2-unavailable",
        stage="smoke_v3",
        evidence=EVIDENCE_SMOKE,
        plan=[rank_task, control_task],
        records=[rank_record, control_record],
    )
    output = tmp_path / "unavailable-flat"
    artifacts = preprocess_revision_results(run_dirs=[shard], output_dir=output)
    assert artifacts.row_counts == {
        "trials": 0,
        "features": 0,
        "runtime": 3,
        "failures": 0,
        "detector": 0,
        "unavailable": 2,
    }
    rows = list(csv.DictReader((output / "unavailable.csv").open()))
    assert {row["record_type"] for row in rows} == {
        "condition_unavailable",
        "dependent_unavailable",
    }
    assert {row["root_condition_work_id"] for row in rows} == {
        "unavailable-rank"
    }
    assert all(row["excluded_from_estimands"] == "True" for row in rows)
    runtime = pd.read_csv(output / "runtime.csv")
    assert {
        "condition_unavailable_no_execution",
        "dependent_unavailable_no_execution",
    }.issubset(set(runtime["runtime_scope"]))
    output_manifest = json.loads(
        (output / "preprocessing_output_manifest.json").read_text()
    )
    assert output_manifest["invariants"][
        "unavailable_rows_are_not_recovery_failures"
    ] is True


def test_preprocess_cli_and_no_overwrite_policy(tmp_path):
    shard = _primary_shard(tmp_path)
    output = tmp_path / "cli-flat"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_revision_results.py",
            "--run-dir",
            str(shard),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["row_counts"]["detector"] == 2
    with pytest.raises(RevisionPreprocessError, match="overwrite"):
        preprocess_revision_results(run_dirs=[shard], output_dir=output)


def test_rejects_smoke_and_confirmatory_pooling(tmp_path):
    primary = _primary_shard(tmp_path, name="primary")
    rank_task = _rank_task(work_id="smoke-rank", evidence=EVIDENCE_SMOKE)
    control_task = _control_task(evidence=EVIDENCE_SMOKE)
    control_task.update(
        {
            "work_id": "smoke-control",
            "control_id": "smoke-control",
            "source_trial_id": "smoke-rank",
        }
    )
    rank_record = _rank_record(
        work_id="smoke-rank", evidence=EVIDENCE_SMOKE, attempt=1
    )
    control_record = _control_record(evidence=EVIDENCE_SMOKE)
    control_record.update(
        {
            "work_id": "smoke-control",
            "control_id": "smoke-control",
            "source_trial_id": "smoke-rank",
        }
    )
    smoke = _write_shard(
        tmp_path / "smoke",
        stage="smoke_v3",
        evidence=EVIDENCE_SMOKE,
        plan=[rank_task, control_task],
        records=[rank_record, control_record],
    )
    with pytest.raises(RevisionPreprocessError, match="cannot be mixed"):
        preprocess_revision_results(
            run_dirs=[primary, smoke], output_dir=tmp_path / "mixed"
        )


def test_rejects_identity_tamper_and_duplicate_attempt(tmp_path):
    shard = _primary_shard(tmp_path, name="tampered")
    identity_path = shard / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["planned_trial_count"] = 99
    _json(identity_path, identity)
    with pytest.raises(RevisionPreprocessError, match="run_identity_sha256"):
        preprocess_revision_results(run_dirs=[shard], output_dir=tmp_path / "bad-identity")

    duplicate = _primary_shard(tmp_path, name="duplicate")
    rows = [json.loads(line) for line in (duplicate / "records.jsonl").read_text().splitlines()]
    rows.append(deepcopy(rows[-1]))
    _jsonl(duplicate / "records.jsonl", rows)
    with pytest.raises(RevisionPreprocessError, match="Duplicate durable attempt"):
        preprocess_revision_results(run_dirs=[duplicate], output_dir=tmp_path / "bad-attempt")


def test_reference_join_resolves_primary_without_emitting_reference_source(tmp_path):
    primary = _write_shard(
        tmp_path / "source-primary",
        stage="primary_v2",
        evidence=EVIDENCE_PRIMARY,
        plan=[_rank_task()],
        records=[_rank_record(attempt=1)],
    )
    task = {
        **_rank_task(work_id="ablation-ref", evidence=EVIDENCE_ABLATION),
        "work_kind": "reference",
        "source_trial_id": "rank-1",
        "ablation_factor": "token_filter",
        "ablation_level": "safe_text_filter_v1",
        "study_phase": "ablation_confirmatory",
    }
    record = {
        "schema_version": "1.0",
        "record_type": "canonical_primary_reference",
        "work_id": "ablation-ref",
        "trial_id": "ablation-ref",
        "source_trial_id": "rank-1",
        "evidence_status": EVIDENCE_ABLATION,
        "study_phase": "ablation_confirmatory",
        "model_id": MODEL_ID,
        "payload_name": "payload-001",
        "ablation_factor": "token_filter",
        "ablation_level": "safe_text_filter_v1",
        "generation_performed": False,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    ablation = _write_shard(
        tmp_path / "ablation",
        stage="ablation_v2",
        evidence=EVIDENCE_ABLATION,
        plan=[task],
        records=[record],
    )
    output = tmp_path / "reference-flat"
    artifacts = preprocess_revision_results(
        run_dirs=[ablation], reference_run_dirs=[primary], output_dir=output
    )
    assert artifacts.row_counts["trials"] == 1
    row = next(csv.DictReader((output / "trials.csv").open()))
    assert row["trial_id"] == "ablation-ref"
    assert row["source_trial_id"] == "rank-1"
    assert row["exact_recovery"] == "1"
    assert artifacts.row_counts["detector"] == 0


def test_robustness_adapter_checks_source_hash_and_preserves_failure_fields(tmp_path):
    source_record = _rank_record(attempt=1)
    primary = _write_shard(
        tmp_path / "robust-source",
        stage="primary_v2",
        evidence=EVIDENCE_PRIMARY,
        plan=[_rank_task()],
        records=[source_record],
    )
    task = {
        **_rank_task(work_id="robust-1", evidence=EVIDENCE_ROBUSTNESS),
        "work_kind": "robustness_decode",
        "study_phase": "robustness_confirmatory_supporting",
        "robustness_family": "raw_transmission",
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "replay_mode": "transformed_text_retokenized",
        "transformation_id": "whitespace_collapse",
    }
    failure = _failure("robust-1")
    failure.update(
        {
            "failure_category": "raw_transmission",
            "replay_mode": "transformed_text_retokenized",
            "transformation_id": "whitespace_collapse",
        }
    )
    robustness_record = {
        "schema_version": "1.0",
        "record_type": "robustness_decode",
        "work_id": "robust-1",
        "trial_id": "robust-1",
        "evidence_status": EVIDENCE_ROBUSTNESS,
        "study_phase": "robustness_confirmatory_supporting",
        "robustness_family": "raw_transmission",
        "model_id": MODEL_ID,
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "source_record_sha256": hashlib.sha256(
            canonical_json_bytes(source_record)
        ).hexdigest(),
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "replay_mode": "transformed_text_retokenized",
        "transformation_id": "whitespace_collapse",
        "segment_outcomes": [
            {
                "segment_index": 0,
                "prompt": {
                    "prompt_id": "prompt-1",
                    "prompt_category": "casual",
                    "prompt_text": "Write a short casual note.",
                },
                "observed_text": "Forced sentence. Natural tail.",
                "observed_full_token_ids": [10, 12, 12, 13],
                "exact_rank_replay": False,
            }
        ],
        "decoded": {
            "recovery_outcome_semantics": RECOVERY_SEMANTICS,
            "exact_payload_recovery": False,
            "exact_recovery": False,
            "original_payload_sha256": "a" * 64,
            "recovered_payload_sha256": "b" * 64,
        },
        "exact_payload_recovery": False,
        "exact_recovery": False,
        "failure": failure,
        "execution_seconds": 0.7,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    robustness = _write_shard(
        tmp_path / "robustness",
        stage="robustness_v2",
        evidence=EVIDENCE_ROBUSTNESS,
        plan=[task],
        records=[robustness_record],
    )
    output = tmp_path / "robust-flat"
    artifacts = preprocess_revision_results(
        run_dirs=[robustness], reference_run_dirs=[primary], output_dir=output
    )
    assert artifacts.row_counts["trials"] == 1
    assert artifacts.row_counts["features"] == 1
    failure_row = next(csv.DictReader((output / "failures.csv").open()))
    assert failure_row["transformation_id"] == "whitespace_collapse"
    assert failure_row["first_differing_position"] == "1"
    assert failure_row["context_sha256"] == "5" * 64

    bad_record = deepcopy(robustness_record)
    bad_record["source_record_sha256"] = "0" * 64
    bad_shard = _write_shard(
        tmp_path / "bad-robustness",
        stage="robustness_v2",
        evidence=EVIDENCE_ROBUSTNESS,
        plan=[task],
        records=[bad_record],
    )
    with pytest.raises(RevisionPreprocessError, match="source hash mismatch"):
        preprocess_revision_results(
            run_dirs=[bad_shard],
            reference_run_dirs=[primary],
            output_dir=tmp_path / "bad-robust-flat",
        )


def test_paraphrase_transform_is_provenance_only_and_hash_linked(tmp_path):
    source_record = _rank_record(attempt=1)
    primary = _write_shard(
        tmp_path / "paraphrase-source",
        stage="primary_v2",
        evidence=EVIDENCE_PRIMARY,
        plan=[_rank_task()],
        records=[source_record],
    )
    transform_task = {
        **_rank_task(work_id="transform-1", evidence=EVIDENCE_ROBUSTNESS),
        "work_kind": "robustness_transform",
        "study_phase": "robustness_transformation_generation",
        "robustness_family": "raw_transmission_transform",
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "replay_mode": "not_applicable_transform_generation",
        "transformation_id": "paraphrase",
    }
    transform_record = {
        "schema_version": "1.0",
        "record_type": "robustness_transform",
        "work_id": "transform-1",
        "trial_id": "transform-1",
        "evidence_status": EVIDENCE_ROBUSTNESS,
        "study_phase": "robustness_transformation_generation",
        "robustness_family": "raw_transmission_transform",
        "transformation_id": "paraphrase",
        "transformation_model_id": MODEL_ID,
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "source_record_sha256": hashlib.sha256(
            canonical_json_bytes(source_record)
        ).hexdigest(),
        "segment_outputs": [
            {
                "segment_index": 0,
                "source_text_sha256": hashlib.sha256(
                    b"Forced sentence. Natural tail."
                ).hexdigest(),
                "token_ids": [30, 31, 32, 33],
                "text": "Paraphrased natural sentence.",
            }
        ],
        "segment_count": 1,
        "decode_performed": False,
        "execution_seconds": 0.6,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    decode_task = {
        **_rank_task(work_id="decode-1", evidence=EVIDENCE_ROBUSTNESS),
        "work_kind": "robustness_decode",
        "study_phase": "robustness_confirmatory_supporting",
        "robustness_family": "raw_transmission",
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "replay_mode": "transformed_text_retokenized",
        "transformation_id": "paraphrase",
        "transform_work_id": "transform-1",
        "transformation_model_id": MODEL_ID,
    }
    decode_record = {
        "schema_version": "1.0",
        "record_type": "robustness_decode",
        "work_id": "decode-1",
        "trial_id": "decode-1",
        "evidence_status": EVIDENCE_ROBUSTNESS,
        "study_phase": "robustness_confirmatory_supporting",
        "robustness_family": "raw_transmission",
        "model_id": MODEL_ID,
        "source_model_id": MODEL_ID,
        "source_stage": "primary",
        "source_trial_id": "rank-1",
        "source_record_sha256": hashlib.sha256(
            canonical_json_bytes(source_record)
        ).hexdigest(),
        "payload_name": "payload-001",
        "payload_class": "sha256_hex",
        "payload_split": "train",
        "replay_mode": "transformed_text_retokenized",
        "transformation_id": "paraphrase",
        "transform_work_id": "transform-1",
        "transformation_model_id": MODEL_ID,
        "transformation_record_sha256": hashlib.sha256(
            canonical_json_bytes(transform_record)
        ).hexdigest(),
        "segment_outcomes": [
            {
                "segment_index": 0,
                "prompt": {
                    "prompt_id": "prompt-1",
                    "prompt_category": "casual",
                    "prompt_text": "Write a short casual note.",
                },
                "observed_text": "Paraphrased natural sentence.",
                "observed_full_token_ids": [40, 41, 42, 43],
                "exact_rank_replay": True,
            }
        ],
        "decoded": {
            "recovery_outcome_semantics": RECOVERY_SEMANTICS,
            "exact_payload_recovery": True,
            "exact_recovery": True,
            "original_payload_sha256": "a" * 64,
            "recovered_payload_sha256": "a" * 64,
        },
        "exact_payload_recovery": True,
        "exact_recovery": True,
        "failure": None,
        "execution_seconds": 0.7,
        "execution_status": "completed",
        "attempt_index": 1,
        "completed_at": "2026-08-08T00:00:00+00:00",
    }
    robustness = _write_shard(
        tmp_path / "paraphrase-robustness",
        stage="robustness_v2",
        evidence=EVIDENCE_ROBUSTNESS,
        plan=[transform_task, decode_task],
        records=[transform_record, decode_record],
    )
    output = tmp_path / "paraphrase-flat"
    artifacts = preprocess_revision_results(
        run_dirs=[robustness], reference_run_dirs=[primary], output_dir=output
    )
    assert artifacts.row_counts["trials"] == 1
    assert artifacts.row_counts["runtime"] == 3
    trial = next(csv.DictReader((output / "trials.csv").open()))
    assert trial["work_id"] == "decode-1"
    assert trial["transform_work_id"] == "transform-1"
    assert len(trial["transformation_record_sha256"]) == 64
    inputs = json.loads((output / "preprocessing_input_manifest.json").read_text())
    assert inputs["reference_join_counts"]["robustness_transform_provenance"] == 1
    assert inputs["reference_join_counts"]["robustness_decode_transform_link"] == 1


def test_strict_sample_size_rejects_missing_completion(tmp_path):
    shard = _write_shard(
        tmp_path / "incomplete",
        stage="primary_v2",
        evidence=EVIDENCE_PRIMARY,
        plan=[_rank_task()],
        records=[_execution_failure()],
    )
    with pytest.raises(RevisionPreprocessError, match="Incomplete shard"):
        preprocess_revision_results(run_dirs=[shard], output_dir=tmp_path / "strict")
    artifacts = preprocess_revision_results(
        run_dirs=[shard], output_dir=tmp_path / "diagnostic", strict_complete=False
    )
    assert artifacts.row_counts["trials"] == 0
    assert artifacts.row_counts["failures"] == 1
