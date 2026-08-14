"""Fail-closed GPU-budget projection for the frozen revision-v1 study.

The projector deliberately consumes only completed *exploratory smoke* shards.
It never relabels those observations as confirmatory evidence.  Frozen runner
plans provide the work counts; smoke timings provide model/operation rates.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_artifacts import (
    build_directory_manifest,
    canonical_json_sha256,
    file_sha256,
    load_checkpoint,
    trial_ids_sha256,
    verify_directory_manifest,
)
from .revision_config import DEFAULT_REVISION_CONFIG_DIR, load_revision_config_set, model_ids
from .revision_payloads import REVISION_CORPUS_ID
from .revision_invalidation import verify_invalidation_entry
from .revision_runner import (
    EVIDENCE_SMOKE_V3,
    PROJECT_ROOT,
    PROTOCOL_CONTRACT_REVISION,
    RESULT_SCHEMA_REVISION,
    build_stage_plan,
    verify_config_manifest,
)


COMPUTE_SCHEMA_VERSION = "rankcloak-revision-compute-projection-v1"
AUXILIARY_TIMING_SCHEMA_VERSION = "rankcloak-revision-compute-timing-v2"
LEGACY_INCURRED_LEDGER_SCHEMA_VERSION = (
    "rankcloak-revision-legacy-incurred-gpu-ledger-v1"
)
LEGACY_INCURRED_LEDGER_HASH_FIELD = "legacy_incurred_ledger_sha256"
LEGACY_SMOKE_STAGE = "smoke_v2"
LEGACY_EVIDENCE_STATUS = "exploratory_smoke_not_for_confirmatory_pooling"
LEGACY_CONFIG_MANIFEST_SHA256 = (
    "dc0e7e022036e2681c87ad06446cbebd56d676faf81a0544a55d56375d4eadcd"
)
LEGACY_RUNNER_SOURCE_FILES_SHA256 = (
    "f7ae057ba165cbbc091932f04ad35aa3c15643f6077a4ae48cdda399cb59bb9a"
)
LEGACY_RUNNER_SOURCE_MANIFEST_SHA256 = (
    "94d9d7c0b06733e9dfdce8f39141066002374b73dad454a0a00c0e92b50c8322"
)
LEGACY_RUNNER_RUNTIME_SHA256 = (
    "c206a1cbfbbb4781d87fd120a3fe45400a505dd2fa2bfb346278c84e56776664"
)
LEGACY_RUNNER_HARDWARE_SHA256 = (
    "3f59f33f69daa2d5d71531968a409162229266db785361637834e166a903522a"
)
LEGACY_RUNNER_PAYLOAD_SHA256 = (
    "852f4c15ca2db22c9687f205df6f373e0455f4e63f2956fa6a22aba66bde5691"
)
LEGACY_EVALUATOR_SOURCE_FILES_SHA256 = (
    "689b7f850d699966ff9498a94dc7f9ee518492044ee8054b5f71ec95670aa1c7"
)
LEGACY_EVALUATOR_SOURCE_MANIFEST_SHA256 = (
    "db3c67ff7afefd5dc5ff670c6b9a841d963ac40d205cd8d1977aff4b32eb2f05"
)
LEGACY_EVALUATOR_RUNTIME_SHA256 = (
    "73bb2bc1933671429c4264420e5f52a5ce7a3433556607c464316a7e8abb2d4d"
)
LEGACY_EVALUATOR_HARDWARE_SHA256 = (
    "d4660fe2f99c09c0e02354d78bb3acc62145ac073c662b21be1f64ad278fa3df"
)
LEGACY_EVALUATOR_TIMING_SHA256 = {
    "llama3_8b_instruct_q4_k_m": "3f0d925f368d3883a722d3e77a0125b638f7f7044a5d11a9abd362e70fd984ea",
    "qwen2_5_7b_instruct_q4_k_m": "0783e1bab9a6f8a8887a49e52fd4677c37348c38d27a323f7fdf3116b4cb345f",
    "mistral_7b_instruct_v0_3_q4_k_m": "f8d00bfe74ddb84a2ac52d19c6ef0b6269e3698128cb744cf45de221e6ded800",
}
EXPECTED_INVALIDATION_MANIFEST_SHA256 = (
    "a9836f60344c38568f4dbc014deb6c428b1bfad216f9a55da683edd978f9168c"
)
EXPECTED_INVALIDATED_RUN_IDENTITY_SHA256 = (
    "fda1f6aba51df4f18606f017f66eb64c8efe7700ef18b23263157772de772c76"
)
EXPECTED_INVALIDATED_SHARD_TREE_SHA256 = (
    "97af0aeadc76127c1d7eaa426869f084ec6f16769ffd2b58ab154a774aa0f108"
)
EXPECTED_INVALIDATED_GPU_SECONDS = 2189.687278
DEFAULT_BUDGET_GPU_HOURS = 150.0
EXPECTED_SMOKE_STAGE = "smoke_v3"
EXPECTED_MODELS = (
    "llama3_8b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m",
)
MISTRAL_MODEL_ID = "mistral_7b_instruct_v0_3_q4_k_m"
PROJECTED_RUNNER_STAGES = (
    "primary_v2", "ablation_v2", "multilingual_v2", "robustness_v2"
)
EXPECTED_PLAN_COUNTS = {
    "smoke_v3": {
        "total": 96,
        "work_kinds": {"rankcloak": 36, "control": 60},
        "models": {model: 32 for model in EXPECTED_MODELS},
    },
    "primary_v2": {
        "total": 14400,
        "work_kinds": {"rankcloak": 6480, "control": 7920},
        "models": {model: 4800 for model in EXPECTED_MODELS},
    },
    "ablation_v2": {
        "total": 1872,
        "work_kinds": {"rankcloak": 1728, "reference": 144},
        "models": {model: 624 for model in EXPECTED_MODELS},
    },
    "multilingual_v2": {
        "total": 1152,
        "work_kinds": {"rankcloak": 576, "control": 576},
        "models": {model: 384 for model in EXPECTED_MODELS},
    },
    "robustness_v2": {
        "total": 3744,
        "work_kinds": {
            "robustness_decode": 3168,
            "reference": 432,
            "robustness_transform": 144,
        },
        "models": {
            EXPECTED_MODELS[0]: 1200,
            EXPECTED_MODELS[1]: 1344,
            EXPECTED_MODELS[2]: 1200,
        },
    },
}

# Smoke-v3 verifies that the pinned Mistral tokenizer has no token satisfying
# the frozen isolated round-trip-stable filter. These are completed
# non-outcomes, not failed work. The same structural infeasibility propagates
# to the 48 matched ablation sources and every one of their 336 robustness work
# units (48 zero-compute references plus 288 dependent decodes).
EXPECTED_SMOKE_UNAVAILABLE = {
    MISTRAL_MODEL_ID: {
        "condition_unavailable": 1,
        "dependent_unavailable": 2,
    }
}
EXPECTED_UNAVAILABLE_PROPAGATION = {
    "ablation_v2": {MISTRAL_MODEL_ID: 48},
    "robustness_v2": {MISTRAL_MODEL_ID: 336},
}
EXPECTED_UNAVAILABLE_WORK_KINDS = {
    "ablation_v2": {"rankcloak": 48},
    "robustness_v2": {"reference": 48, "robustness_decode": 288},
}
UNAVAILABLE_RECORD_TYPES = {"condition_unavailable", "dependent_unavailable"}

# Each generator contributes 4,800 primary, 576 generated ablation, and 384
# multilingual records.  The cyclic evaluator mapping therefore assigns 5,760
# records to each evaluator model.
EVALUATOR_UNITS_PER_MODEL = 5760

# matched + 18 held-out templates + 3 held-out models + 6 held-out codecs.
DETECTOR_SPLITS = 28
DETECTOR_FITS = {
    "published_textcnn_equivalent": DETECTOR_SPLITS,
    "deberta_v3_base_classifier": DETECTOR_SPLITS,
}

REQUIRED_SHARD_FILES = (
    "plan.jsonl",
    "run_identity.json",
    "payload_manifest.json",
    "model_manifest.json",
    "source_manifest.json",
    "runtime_manifest.json",
    "hardware_manifest.json",
    "checkpoint.json",
    "records.jsonl",
    "events.jsonl",
)


class RevisionComputeError(RuntimeError):
    """Raised when a projection input cannot support a safe go decision."""


def _read_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionComputeError("Cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise RevisionComputeError("Expected a JSON object: {}".format(path))
    return value


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RevisionComputeError("Cannot read JSONL {}: {}".format(path, exc)) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RevisionComputeError(
                "Invalid JSONL at {}:{}: {}".format(path, line_number, exc)
            ) from exc
        if not isinstance(value, dict):
            raise RevisionComputeError(
                "Expected an object at {}:{}".format(path, line_number)
            )
        rows.append(value)
    return rows


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RevisionComputeError("{} must be numeric".format(label))
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RevisionComputeError("{} must be finite and non-negative".format(label))
    return number


def _scientific_args(identity: Mapping[str, object]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    values = identity.get("command_line_args")
    if not isinstance(values, list):
        raise RevisionComputeError("run_identity command_line_args must be a list")
    for raw in values:
        text = str(raw)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in parsed:
            raise RevisionComputeError("Duplicate run-identity argument: {}".format(key))
        parsed[key] = value
    return parsed


def _verify_bound_manifest(
    shard: Path,
    identity_args: Mapping[str, str],
    filename: str,
    argument_name: str,
) -> None:
    expected = identity_args.get(argument_name)
    actual = file_sha256(shard / filename)
    if expected != actual:
        raise RevisionComputeError(
            "{} is not bound to run_identity ({})".format(filename, argument_name)
        )


def _identity_without_digest(identity: Mapping[str, object]) -> Dict[str, object]:
    value = dict(identity)
    value.pop("run_identity_sha256", None)
    return value


def _model_from_plan(plan: Sequence[Mapping[str, object]]) -> str:
    observed = {str(row.get("model_id")) for row in plan}
    if len(observed) != 1:
        raise RevisionComputeError("Smoke shard must contain exactly one model")
    model_id = next(iter(observed))
    if model_id not in EXPECTED_MODELS:
        raise RevisionComputeError("Unexpected smoke model: {}".format(model_id))
    return model_id


def _record_elapsed(record: Mapping[str, object]) -> float:
    if record.get("record_type") in UNAVAILABLE_RECORD_TYPES:
        return 0.0
    if record.get("record_type") == "rankcloak_trial":
        timing = record.get("timing")
        if not isinstance(timing, dict):
            raise RevisionComputeError("RankCloak record is missing timing")
        return _finite_nonnegative(timing.get("total_seconds"), "timing.total_seconds")
    return _finite_nonnegative(record.get("execution_seconds"), "execution_seconds")


def _validate_unavailable_flags(record: Mapping[str, object]) -> None:
    """Validate fields that distinguish a completed non-outcome from a failure."""

    work_id = str(record.get("work_id"))
    required = {
        "condition_available": False,
        "excluded_from_estimands": True,
        "generation_performed": False,
        "decode_performed": False,
        "exact_recovery": None,
    }
    for field, expected in required.items():
        if record.get(field) is not expected:
            raise RevisionComputeError(
                "Unavailable record {} has invalid {}".format(work_id, field)
            )
    for field in ("execution_seconds",):
        if field in record and _finite_nonnegative(record[field], field) != 0.0:
            raise RevisionComputeError(
                "Unavailable record {} reports nonzero execution".format(work_id)
            )


def _validate_unavailable_record(
    record: Mapping[str, object],
    task: Mapping[str, object],
    completed_records: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Mirror the frozen preprocessor's unavailable-source validation contract."""

    _validate_unavailable_flags(record)
    work_id = str(record.get("work_id"))
    record_type = str(record.get("record_type"))
    if str(record.get("work_kind")) != str(task.get("work_kind")):
        raise RevisionComputeError(
            "Unavailable record work_kind differs from plan for {}".format(work_id)
        )

    if record_type == "condition_unavailable":
        if str(task.get("work_kind")) != "rankcloak":
            raise RevisionComputeError("Only RankCloak work may be condition-unavailable")
        if record.get("reason_code") != "empty_isolated_roundtrip_vocabulary":
            raise RevisionComputeError("Unknown frozen condition-unavailable reason")
        stable_count = record.get("stable_count")
        safe_count = record.get("safe_count")
        if isinstance(stable_count, bool) or not isinstance(stable_count, int) or stable_count != 0:
            raise RevisionComputeError("Empty round-trip vocabulary must record stable_count=0")
        if isinstance(safe_count, bool) or not isinstance(safe_count, int) or safe_count <= 0:
            raise RevisionComputeError("Round-trip audit must record a nonempty safe mask")
        if str(record.get("token_filter")) != "roundtrip_stable_filter_v1":
            raise RevisionComputeError("Unavailable condition has the wrong frozen filter")
        if str(record.get("token_filter")) != str(task.get("token_filter")):
            raise RevisionComputeError("Unavailable condition filter differs from plan")
        if str(record.get("trial_id")) != str(task.get("trial_id")):
            raise RevisionComputeError("Unavailable condition trial differs from plan")
        return {
            "model_id": str(record.get("model_id")),
            "token_filter": str(record.get("token_filter")),
            "ablation_factor": str(task.get("ablation_factor")),
            "ablation_level": str(task.get("ablation_level")),
            "protocol_variant": str(task.get("protocol_variant")),
            "reason_code": str(record.get("reason_code")),
            "root_smoke_work_id": work_id,
            "safe_count": safe_count,
            "stable_count": stable_count,
        }

    if record_type != "dependent_unavailable":
        raise RevisionComputeError("Unknown unavailable record type {}".format(record_type))
    if record.get("reason_code") != "source_condition_unavailable":
        raise RevisionComputeError("Unknown dependent-unavailable reason")
    source_id = str(record.get("source_trial_id"))
    if source_id != str(task.get("source_trial_id")):
        raise RevisionComputeError("Dependent-unavailable source differs from plan")
    candidates = [
        source
        for source in completed_records.values()
        if str(source.get("trial_id")) == source_id
    ]
    if len(candidates) != 1:
        raise RevisionComputeError(
            "Dependent-unavailable source does not resolve uniquely: {}".format(source_id)
        )
    source = candidates[0]
    if str(source.get("record_type")) not in UNAVAILABLE_RECORD_TYPES:
        raise RevisionComputeError("Dependent-unavailable row resolves to an outcome source")
    if str(record.get("source_record_sha256")) != canonical_json_sha256(source):
        raise RevisionComputeError("Dependent-unavailable source hash mismatch")
    if str(record.get("dependency_record_type")) != str(source.get("record_type")):
        raise RevisionComputeError("Dependent-unavailable source record type mismatch")
    embedded_root = record.get("dependency_root")
    if not isinstance(embedded_root, dict):
        raise RevisionComputeError("Dependent-unavailable row lacks dependency_root")
    source_root = (
        source.get("dependency_root")
        if isinstance(source.get("dependency_root"), dict)
        else source
    )
    for field in (
        "work_id",
        "trial_id",
        "record_type",
        "reason_code",
        "model_id",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_artifact_sha256",
        "safe_count",
        "stable_count",
    ):
        if str(embedded_root.get(field)) != str(source_root.get(field)):
            raise RevisionComputeError(
                "Dependent-unavailable root {} mismatch for {}".format(field, work_id)
            )
    return {}


def _supported_rankcloak_seconds(record: Mapping[str, object]) -> float:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        raise RevisionComputeError("RankCloak record is missing timing")
    # The confirmatory runner executes saved-token replay only.  Using the
    # explicitly reported encoding and supported-decoding timers avoids charging
    # the two smoke-only diagnostic replay paths to every confirmatory row.
    encoding = _finite_nonnegative(timing.get("encoding_seconds"), "encoding_seconds")
    decoding = _finite_nonnegative(
        timing.get("supported_decoding_seconds"), "supported_decoding_seconds"
    )
    return encoding + decoding


def _diagnostic_replay_seconds(record: Mapping[str, object], replay_mode: str) -> float:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        raise RevisionComputeError("RankCloak record is missing timing")
    if replay_mode == "greedy_leadin_regeneration":
        keys = ("greedy_leadin_regeneration_seconds", "greedy_inverse_transcode_seconds")
    elif replay_mode == "saved_token_ids":
        keys = ("saved_token_id_replay_seconds", "saved_inverse_transcode_seconds")
    else:
        keys = ("detokenized_text_retokenized_seconds", "text_inverse_transcode_seconds")
    return sum(_finite_nonnegative(timing.get(key), key) for key in keys)


def _validate_manifest_shapes(shard: Path, identity: Mapping[str, object]) -> None:
    payload = _read_json(shard / "payload_manifest.json")
    model = _read_json(shard / "model_manifest.json")
    source = _read_json(shard / "source_manifest.json")
    runtime = _read_json(shard / "runtime_manifest.json")
    hardware = _read_json(shard / "hardware_manifest.json")
    if identity.get("payload_manifest_sha256") != file_sha256(shard / "payload_manifest.json"):
        raise RevisionComputeError("Payload manifest hash mismatch")
    artifacts = identity.get("model_artifacts")
    if artifacts != [model]:
        raise RevisionComputeError("Embedded model manifest mismatch")
    configured = model.get("configured_model")
    verification = model.get("verification")
    if not isinstance(configured, dict) or not isinstance(verification, dict):
        raise RevisionComputeError("Model manifest is incomplete")
    expected_digest = configured.get("artifact_sha256")
    actual_digest = verification.get("actual_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise RevisionComputeError("Model manifest lacks a pinned SHA-256")
    if actual_digest != expected_digest or verification.get("status") != "ok":
        raise RevisionComputeError("Model artifact verification did not succeed")
    if payload.get("manifest_type") != "revision_v1_public_payload_corpus":
        raise RevisionComputeError("Unexpected payload manifest type")
    if int(payload.get("payload_count", -1)) != 480:
        raise RevisionComputeError("Smoke shard does not bind the 480-payload corpus")
    if source.get("manifest_type") != "revision_runner_source":
        raise RevisionComputeError("Unexpected source manifest type")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise RevisionComputeError("Source manifest has no files")
    if source.get("files_sha256") != hashlib.sha256(
        json.dumps(files, ensure_ascii=False, allow_nan=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise RevisionComputeError("Source manifest file-list hash mismatch")
    if runtime.get("schema_version") != "1.0" or hardware.get("schema_version") != "1.0":
        raise RevisionComputeError("Runtime or hardware manifest schema mismatch")
    selected_gpu = hardware.get("selected_gpu_uuid")
    if not isinstance(selected_gpu, str) or not selected_gpu.startswith("GPU-"):
        raise RevisionComputeError(
            "Smoke-v3 timing must come from an explicitly selected GPU UUID"
        )


def verify_smoke_shard(
    shard: Path,
    expected_plan: Sequence[Mapping[str, object]],
    expected_config_sha256: str,
) -> Dict[str, object]:
    """Verify one complete runner smoke shard and return timing observations."""

    shard = Path(shard).resolve()
    missing = [name for name in REQUIRED_SHARD_FILES if not (shard / name).is_file()]
    if missing:
        raise RevisionComputeError(
            "Smoke shard {} is missing {}".format(shard, ", ".join(missing))
        )
    plan = _read_jsonl(shard / "plan.jsonl")
    model_id = _model_from_plan(plan)
    if canonical_json_sha256(plan) != canonical_json_sha256(list(expected_plan)):
        raise RevisionComputeError(
            "Smoke plan does not equal the frozen {} plan".format(model_id)
        )
    if any(str(row.get("evidence_status")) != EVIDENCE_SMOKE_V3 for row in plan):
        raise RevisionComputeError("Smoke plan has a non-smoke evidence label")
    if any(
        row.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION
        or row.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        for row in plan
    ):
        raise RevisionComputeError(
            "Smoke plan lacks the payload-fidelity-v2 result contract"
        )
    planned_ids = [str(row["work_id"]) for row in plan]
    identity = _read_json(shard / "run_identity.json")
    if identity.get("run_identity_sha256") != canonical_json_sha256(
        _identity_without_digest(identity)
    ):
        raise RevisionComputeError("Run identity self-hash mismatch")
    if (
        identity.get("protocol_contract_revision")
        != PROTOCOL_CONTRACT_REVISION
        or identity.get("result_schema_revision") != RESULT_SCHEMA_REVISION
    ):
        raise RevisionComputeError(
            "Run identity lacks the payload-fidelity-v2 result contract"
        )
    if identity.get("study_id") != "{}/{}/{}".format(
        REVISION_CORPUS_ID, EXPECTED_SMOKE_STAGE, model_id
    ):
        raise RevisionComputeError("Run identity is not the immutable smoke-v3 study")
    if identity.get("config_manifest_sha256") != expected_config_sha256:
        raise RevisionComputeError("Smoke shard uses a different frozen config manifest")
    if identity.get("planned_trial_count") != len(planned_ids):
        raise RevisionComputeError("Run identity planned count mismatch")
    if identity.get("planned_trial_ids_sha256") != trial_ids_sha256(planned_ids):
        raise RevisionComputeError("Run identity ordered-plan hash mismatch")
    args = _scientific_args(identity)
    expected_args = {
        "stage": EXPECTED_SMOKE_STAGE,
        "model_id": model_id,
        "evidence_status": EVIDENCE_SMOKE_V3,
        "context_limit": "4096",
        "n_gpu_layers": "-1",
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise RevisionComputeError("Run identity argument {} mismatch".format(key))
    _verify_bound_manifest(shard, args, "source_manifest.json", "source_manifest_sha256")
    _verify_bound_manifest(shard, args, "runtime_manifest.json", "runtime_manifest_sha256")
    _verify_bound_manifest(shard, args, "hardware_manifest.json", "hardware_manifest_sha256")
    _validate_manifest_shapes(shard, identity)

    checkpoint = load_checkpoint(shard / "checkpoint.json")
    if checkpoint.get("study_id") != identity.get("study_id"):
        raise RevisionComputeError("Checkpoint study identity mismatch")
    if checkpoint.get("config_manifest_sha256") != expected_config_sha256:
        raise RevisionComputeError("Checkpoint config identity mismatch")
    if checkpoint.get("planned_trial_count") != len(planned_ids):
        raise RevisionComputeError("Checkpoint planned count mismatch")
    if checkpoint.get("planned_trial_ids_sha256") != trial_ids_sha256(planned_ids):
        raise RevisionComputeError("Checkpoint ordered-plan hash mismatch")
    completed = set(map(str, checkpoint.get("completed_trial_ids", [])))
    if completed != set(planned_ids) or checkpoint.get("failed_trial_ids"):
        raise RevisionComputeError("Smoke checkpoint is incomplete or has current failures")

    records = _read_jsonl(shard / "records.jsonl")
    plan_by_id = {str(row["work_id"]): row for row in plan}
    completed_records: Dict[str, Mapping[str, object]] = {}
    for record in records:
        if record.get("execution_status") != "completed":
            continue
        work_id = str(record.get("work_id"))
        if work_id not in plan_by_id:
            raise RevisionComputeError("Completed record has an unknown work ID")
        if work_id in completed_records:
            raise RevisionComputeError("Duplicate durable completion: {}".format(work_id))
        task = plan_by_id[work_id]
        if str(record.get("model_id")) != model_id:
            raise RevisionComputeError("Completed record model mismatch")
        if str(record.get("evidence_status")) != EVIDENCE_SMOKE_V3:
            raise RevisionComputeError("Completed record has a non-smoke evidence label")
        if (
            record.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or record.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        ):
            raise RevisionComputeError(
                "Completed record lacks the payload-fidelity-v2 result contract"
            )
        if record.get("study_phase") != task.get("study_phase"):
            raise RevisionComputeError("Completed record study_phase differs from plan")
        expected_types = {
            "rankcloak": {"rankcloak_trial", "condition_unavailable"},
            "control": {"ordinary_control", "dependent_unavailable"},
        }
        if record.get("record_type") not in expected_types[str(task["work_kind"])]:
            raise RevisionComputeError("Completed record type does not match its plan task")
        _record_elapsed(record)
        completed_records[work_id] = record
    if set(completed_records) != set(planned_ids):
        raise RevisionComputeError("Smoke records lack one-to-one durable completions")

    unavailability_contracts: List[Dict[str, object]] = []
    unavailable_record_counts: Counter = Counter()
    for work_id in planned_ids:
        task = plan_by_id[work_id]
        record = completed_records[work_id]
        record_type = str(record.get("record_type"))
        if record_type in UNAVAILABLE_RECORD_TYPES:
            contract = _validate_unavailable_record(record, task, completed_records)
            unavailable_record_counts[record_type] += 1
            if contract:
                unavailability_contracts.append(contract)

    events = _read_jsonl(shard / "events.jsonl")
    hardware = _read_json(shard / "hardware_manifest.json")
    selected_gpu_uuid = str(hardware["selected_gpu_uuid"])
    if args.get("gpu_uuid") != selected_gpu_uuid:
        raise RevisionComputeError("Run identity and hardware GPU UUID mismatch")
    load_events = [row for row in events if row.get("event") == "model_loaded"]
    if not load_events:
        raise RevisionComputeError("Smoke shard has no model-load timing")
    load_seconds = []
    for event in load_events:
        if str(event.get("model_id")) != model_id:
            raise RevisionComputeError("Model-load event model mismatch")
        if str(event.get("gpu_uuid")) != selected_gpu_uuid:
            raise RevisionComputeError("Model-load event GPU UUID mismatch")
        load_seconds.append(
            _finite_nonnegative(event.get("model_load_seconds"), "model_load_seconds")
        )
    profiles = [row for row in events if row.get("event") == "memory_profile"]
    if len(profiles) != 1:
        raise RevisionComputeError(
            "Smoke-v3 shard requires exactly one final memory_profile event"
        )
    profile = profiles[0]
    profile_started = _timestamp(
        profile.get("started_at"), "smoke-v3 memory_profile.started_at"
    )
    profile_ended = _timestamp(
        profile.get("at"), "smoke-v3 memory_profile.at"
    )
    profile_seconds = (profile_ended - profile_started).total_seconds()
    if (
        profile_seconds <= 0
        or str(profile.get("selected_gpu_uuid")) != selected_gpu_uuid
    ):
        raise RevisionComputeError("Smoke-v3 memory-profile GPU span is invalid")
    for field in (
        "sample_count",
        "selected_gpu_sample_count",
        "process_rss_sample_count",
    ):
        if not isinstance(profile.get(field), int) or int(profile[field]) <= 0:
            raise RevisionComputeError("Smoke-v3 memory profile lacks positive samples")
    finished = [row for row in events if row.get("event") == "session_finished"]
    if not finished or int(finished[-1].get("remaining", -1)) != 0:
        raise RevisionComputeError("Smoke shard lacks a final completed session event")

    observations: List[Dict[str, object]] = []
    for work_id in planned_ids:
        task = plan_by_id[work_id]
        record = completed_records[work_id]
        row: Dict[str, object] = {
            "work_id": work_id,
            "model_id": model_id,
            "work_kind": task["work_kind"],
            "protocol_variant": task.get("protocol_variant"),
            "control_view": task.get("control_view"),
            "record_type": record.get("record_type"),
            "reason_code": record.get("reason_code"),
            "actual_seconds": _record_elapsed(record),
            "condition_available": record.get("record_type")
            not in UNAVAILABLE_RECORD_TYPES,
        }
        if task["work_kind"] == "rankcloak" and row["condition_available"]:
            row["supported_seconds"] = _supported_rankcloak_seconds(record)
            row["saved_replay_seconds"] = _diagnostic_replay_seconds(
                record, "saved_token_ids"
            )
            row["text_replay_seconds"] = _diagnostic_replay_seconds(
                record, "detokenized_text_retokenized"
            )
            row["greedy_replay_seconds"] = _diagnostic_replay_seconds(
                record, "greedy_leadin_regeneration"
            )
        observations.append(row)
    return {
        "path": str(shard),
        "model_id": model_id,
        "plan_sha256": canonical_json_sha256(plan),
        "run_identity_sha256": identity["run_identity_sha256"],
        "records_sha256": file_sha256(shard / "records.jsonl"),
        "hardware_manifest_sha256": file_sha256(shard / "hardware_manifest.json"),
        "work_units": len(plan),
        "available_work_units": len(plan) - sum(unavailable_record_counts.values()),
        "unavailable_work_units": sum(unavailable_record_counts.values()),
        "unavailable_record_counts": dict(sorted(unavailable_record_counts.items())),
        "unavailability_contracts": unavailability_contracts,
        "load_seconds": load_seconds,
        "observations": observations,
        "actual_gpu_seconds": profile_seconds,
        "actual_gpu_charge_policy": "memory_profile_wall_span_v1",
        "diagnostic_model_load_plus_record_seconds": sum(load_seconds)
        + sum(float(row["actual_seconds"]) for row in observations),
    }


def _validate_frozen_counts(plans: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    for stage, expected in EXPECTED_PLAN_COUNTS.items():
        plan = plans[stage]
        kinds = dict(Counter(str(row["work_kind"]) for row in plan))
        models = dict(Counter(str(row["model_id"]) for row in plan))
        if len(plan) != expected["total"] or kinds != expected["work_kinds"] or models != expected["models"]:
            raise RevisionComputeError(
                "Frozen {} plan counts drifted: total={}, kinds={}, models={}".format(
                    stage, len(plan), kinds, models
                )
            )


def load_frozen_plans(
    configs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    loaded = dict(configs) if configs is not None else load_revision_config_set()
    plans = {
        stage: build_stage_plan(stage, configs=loaded)
        for stage in (EXPECTED_SMOKE_STAGE,) + PROJECTED_RUNNER_STAGES
    }
    _validate_frozen_counts(plans)
    return plans


def discover_smoke_shards(root: Path) -> List[Path]:
    """Find candidate runner shards below a root without accepting other stages."""

    root = Path(root)
    if (root / "plan.jsonl").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("plan.jsonl")})


def _upper_multiplier(sample_size: int) -> float:
    # With one observation for several smoke protocol strata, an empirical CI
    # would be falsely precise.  The fixed factors are deliberately conservative.
    if sample_size <= 1:
        return 2.0
    if sample_size == 2:
        return 1.75
    if sample_size < 5:
        return 1.5
    return 1.35


def _rate_summary(values: Sequence[float]) -> Dict[str, object]:
    clean = [float(value) for value in values]
    if not clean:
        raise RevisionComputeError("A required smoke timing stratum is empty")
    point = statistics.median(clean)
    upper = max(clean) * _upper_multiplier(len(clean))
    return {
        "sample_size": len(clean),
        "point_seconds_per_unit": point,
        "upper_seconds_per_unit": upper,
        "observed_min_seconds": min(clean),
        "observed_max_seconds": max(clean),
        "upper_method": "observed_max_times_fixed_small_n_multiplier",
        "upper_multiplier": _upper_multiplier(len(clean)),
    }


def _smoke_rates(shards: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    rates: Dict[str, object] = {}
    for shard in shards:
        model_id = str(shard["model_id"])
        grouped: Dict[str, List[float]] = defaultdict(list)
        replay: Dict[str, List[float]] = defaultdict(list)
        full_control: List[float] = []
        all_supported: List[float] = []
        for row in shard["observations"]:  # type: ignore[index]
            if not bool(row.get("condition_available", True)):
                continue
            kind = str(row["work_kind"])
            if kind == "rankcloak":
                protocol = str(row["protocol_variant"])
                seconds = float(row["supported_seconds"])
                grouped["rankcloak:{}".format(protocol)].append(seconds)
                all_supported.append(seconds)
                replay["saved_token_ids:{}".format(protocol)].append(
                    float(row["saved_replay_seconds"])
                )
                replay["text_retokenized:{}".format(protocol)].append(
                    float(row["text_replay_seconds"])
                )
                replay["greedy_leadin_regeneration:{}".format(protocol)].append(
                    float(row["greedy_replay_seconds"])
                )
            else:
                view = str(row["control_view"])
                seconds = float(row["actual_seconds"])
                grouped["control:{}".format(view)].append(seconds)
                if view == "full_message":
                    full_control.append(seconds)
        if not all_supported or not full_control:
            raise RevisionComputeError("Smoke shard lacks generation timing strata")
        rates[model_id] = {
            "strata": {key: _rate_summary(values) for key, values in sorted(grouped.items())},
            "replay": {key: _rate_summary(values) for key, values in sorted(replay.items())},
            "all_supported": _rate_summary(all_supported),
            "full_control": _rate_summary(full_control),
            "model_load": _rate_summary([float(value) for value in shard["load_seconds"]]),
        }
    return rates


def _task_rate_key(task: Mapping[str, object]) -> Optional[str]:
    kind = str(task["work_kind"])
    if kind == "rankcloak":
        return "rankcloak:{}".format(task["protocol_variant"])
    if kind == "control":
        return "control:{}".format(task["control_view"])
    return None


def _derive_projected_unavailability(
    plans: Mapping[str, Sequence[Mapping[str, object]]],
    smoke_shards: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, object]]:
    """Join smoke feasibility roots to the frozen downstream work graph."""

    smoke_counts: Dict[str, Dict[str, int]] = {}
    contracts: List[Mapping[str, object]] = []
    for shard in smoke_shards:
        model_id = str(shard["model_id"])
        counts = {
            str(key): int(value)
            for key, value in dict(shard.get("unavailable_record_counts", {})).items()
        }
        if counts:
            smoke_counts[model_id] = counts
        for contract in shard.get("unavailability_contracts", []):  # type: ignore[union-attr]
            if not isinstance(contract, dict):
                raise RevisionComputeError("Malformed smoke unavailability contract")
            contracts.append(contract)
    if smoke_counts != EXPECTED_SMOKE_UNAVAILABLE:
        raise RevisionComputeError(
            "Frozen smoke-v3 unavailable counts drifted: {}".format(smoke_counts)
        )

    projected: Dict[str, Dict[str, Dict[str, object]]] = {
        stage: {} for stage in PROJECTED_RUNNER_STAGES
    }
    unavailable_ablation_ids: set[str] = set()
    roots_by_ablation_id: Dict[str, str] = {}
    for contract in contracts:
        matches = [
            task
            for task in plans["ablation_v2"]
            if str(task.get("work_kind")) == "rankcloak"
            and str(task.get("model_id")) == str(contract.get("model_id"))
            and str(task.get("token_filter")) == str(contract.get("token_filter"))
            and str(task.get("ablation_factor")) == str(contract.get("ablation_factor"))
            and str(task.get("ablation_level")) == str(contract.get("ablation_level"))
            and str(task.get("protocol_variant")) == str(contract.get("protocol_variant"))
        ]
        if not matches:
            raise RevisionComputeError(
                "Smoke unavailable condition has no frozen ablation descendants"
            )
        for task in matches:
            work_id = str(task["work_id"])
            root = str(contract["root_smoke_work_id"])
            previous = roots_by_ablation_id.get(work_id)
            if previous is not None and previous != root:
                raise RevisionComputeError(
                    "Ablation work has conflicting unavailability roots: {}".format(work_id)
                )
            unavailable_ablation_ids.add(work_id)
            roots_by_ablation_id[work_id] = root
            projected["ablation_v2"][work_id] = {
                "record_type": "condition_unavailable",
                "reason_code": contract["reason_code"],
                "root_smoke_work_id": root,
            }

    for task in plans["robustness_v2"]:
        source_id = str(task.get("source_trial_id"))
        if str(task.get("source_stage")) != "ablation_v2" or source_id not in unavailable_ablation_ids:
            continue
        work_id = str(task["work_id"])
        projected["robustness_v2"][work_id] = {
            "record_type": "dependent_unavailable",
            "reason_code": "source_condition_unavailable",
            "source_trial_id": source_id,
            "root_smoke_work_id": roots_by_ablation_id[source_id],
        }

    propagated_counts: Dict[str, Dict[str, int]] = {}
    kind_counts: Dict[str, Dict[str, int]] = {}
    for stage in PROJECTED_RUNNER_STAGES:
        ids = set(projected[stage])
        selected = [task for task in plans[stage] if str(task["work_id"]) in ids]
        by_model = dict(sorted(Counter(str(task["model_id"]) for task in selected).items()))
        by_kind = dict(sorted(Counter(str(task["work_kind"]) for task in selected).items()))
        if by_model:
            propagated_counts[stage] = by_model
        if by_kind:
            kind_counts[stage] = by_kind
    if propagated_counts != EXPECTED_UNAVAILABLE_PROPAGATION:
        raise RevisionComputeError(
            "Frozen unavailable propagation counts drifted: {}".format(propagated_counts)
        )
    if kind_counts != EXPECTED_UNAVAILABLE_WORK_KINDS:
        raise RevisionComputeError(
            "Frozen unavailable work-kind propagation drifted: {}".format(kind_counts)
        )

    summary = {
        "method": "smoke_condition_contract_to_ablation_factor_then_exact_source_trial_id_join",
        "smoke_unavailable_counts": smoke_counts,
        "projected_counts_by_stage_model": propagated_counts,
        "projected_counts_by_stage_work_kind": kind_counts,
        "unavailable_ablation_source_count": len(unavailable_ablation_ids),
        "unavailable_ablation_source_ids_sha256": canonical_json_sha256(
            sorted(unavailable_ablation_ids)
        ),
        "timing_and_recovery_policy": (
            "Completed unavailable rows are zero-execution non-outcomes and are excluded "
            "from timing-rate and recovery estimands."
        ),
    }
    return projected, summary


def _runner_stage_projection(
    stage: str,
    model_id: str,
    plan: Sequence[Mapping[str, object]],
    model_rates: Mapping[str, object],
    projected_unavailable: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    selected = [row for row in plan if str(row["model_id"]) == model_id]
    counts: Counter = Counter()
    for task in selected:
        if str(task["work_id"]) in projected_unavailable:
            unavailable = projected_unavailable[str(task["work_id"])]
            key = "planned_unavailable:{}".format(unavailable["record_type"])
            counts[key] += 1
            continue
        kind = str(task["work_kind"])
        if kind in {"rankcloak", "control"}:
            key = _task_rate_key(task)
        elif kind == "reference":
            key = "reference"
        elif kind == "robustness_transform":
            key = "robustness_transform"
        elif kind == "robustness_decode":
            mode = str(task.get("replay_mode"))
            protocol = str(task.get("protocol_variant"))
            key = (
                "robustness_decode:greedy_leadin_regeneration:{}".format(protocol)
                if mode == "greedy_leadin_regeneration"
                else "robustness_decode:text_retokenized:{}".format(protocol)
            )
        else:
            raise RevisionComputeError("Unknown frozen work kind: {}".format(kind))
        counts[str(key)] += 1

    details: List[Dict[str, object]] = []
    point_seconds = 0.0
    upper_seconds = 0.0
    strata = model_rates["strata"]  # type: ignore[index]
    replay = model_rates["replay"]  # type: ignore[index]
    for key, count in sorted(counts.items()):
        if key.startswith("planned_unavailable:"):
            rate = {
                "sample_size": 0,
                "point_seconds_per_unit": 0.0,
                "upper_seconds_per_unit": 0.0,
                "upper_method": "structural_unavailability_propagated_from_smoke_v2",
                "excluded_from_timing_and_recovery_estimands": True,
            }
        elif key == "reference":
            rate = {
                "sample_size": 0,
                "point_seconds_per_unit": 0.0,
                "upper_seconds_per_unit": 0.0,
                "upper_method": "noncomputational_reference",
            }
        elif key == "robustness_transform":
            # Each severe paraphrase may contain several segments.  A matched
            # full-message control is the nearest smoke operation; multipliers
            # cover the rewrite prompt and multiple segment calls.
            base = model_rates["full_control"]  # type: ignore[index]
            rate = dict(base)
            rate["point_seconds_per_unit"] = float(base["point_seconds_per_unit"]) * 1.5
            rate["upper_seconds_per_unit"] = float(base["upper_seconds_per_unit"]) * 2.0
            rate["upper_method"] = "full_control_proxy_with_twofold_paraphrase_upper"
        elif key.startswith("robustness_decode:"):
            replay_key = key.split(":", 1)[1]
            rate = replay.get(replay_key)
            if rate is None:
                raise RevisionComputeError("Missing smoke replay stratum {}".format(replay_key))
            rate = dict(rate)
            rate["upper_seconds_per_unit"] = float(rate["upper_seconds_per_unit"]) * 1.5
            rate["upper_method"] = "diagnostic_replay_proxy_with_transform_overhead"
        else:
            rate = strata.get(key)
            if rate is None:
                raise RevisionComputeError(
                    "Missing smoke timing stratum {} for {}".format(key, model_id)
                )
            rate = dict(rate)
        point = count * float(rate["point_seconds_per_unit"])
        upper = count * float(rate["upper_seconds_per_unit"])
        point_seconds += point
        upper_seconds += upper
        details.append(
            {
                "stratum": key,
                "target_units": count,
                **rate,
                "point_seconds": point,
                "upper_seconds": upper,
            }
        )
    load = model_rates["model_load"]  # type: ignore[index]
    point_seconds += float(load["point_seconds_per_unit"])
    upper_seconds += float(load["upper_seconds_per_unit"])
    return {
        "stage": stage,
        "model_id": model_id,
        "resource_class": "gpu",
        "target_work_units": len(selected),
        "projected_available_work_units": len(selected)
        - sum(
            count
            for key, count in counts.items()
            if str(key).startswith("planned_unavailable:")
        ),
        "projected_unavailable_work_units": sum(
            count
            for key, count in counts.items()
            if str(key).startswith("planned_unavailable:")
        ),
        "work_kind_counts": dict(sorted(Counter(str(row["work_kind"]) for row in selected).items())),
        "point_seconds": point_seconds,
        "upper_seconds": upper_seconds,
        "point_gpu_hours": point_seconds / 3600.0,
        "upper_gpu_hours": upper_seconds / 3600.0,
        "model_load": load,
        "strata": details,
        "timing_source": "completed_exploratory_smoke_runner_shard",
    }


def _auxiliary_digest(record: Mapping[str, object]) -> str:
    value = dict(record)
    value.pop("timing_manifest_sha256", None)
    return canonical_json_sha256(value)


def load_auxiliary_timings(paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Load optional, self-hashed evaluator or detector smoke timing records."""

    records: List[Dict[str, object]] = []
    for path in paths:
        record = _read_json(Path(path))
        if record.get("schema_version") != AUXILIARY_TIMING_SCHEMA_VERSION:
            raise RevisionComputeError("Unsupported auxiliary timing schema: {}".format(path))
        if record.get("evidence_status") != EVIDENCE_SMOKE_V3:
            raise RevisionComputeError("Auxiliary timing is not labelled exploratory smoke-v3")
        if record.get("source_stage") != EXPECTED_SMOKE_STAGE:
            raise RevisionComputeError("Auxiliary timing is not sourced from smoke-v3")
        if (
            record.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or record.get("result_schema_revision") != RESULT_SCHEMA_REVISION
        ):
            raise RevisionComputeError(
                "Auxiliary timing lacks the payload-fidelity-v2 result contract"
            )
        if record.get("complete") is not True:
            raise RevisionComputeError("Auxiliary timing is incomplete: {}".format(path))
        if record.get("timing_manifest_sha256") != _auxiliary_digest(record):
            raise RevisionComputeError("Auxiliary timing self-hash mismatch: {}".format(path))
        component = str(record.get("component"))
        if component not in {"evaluator", "detector"}:
            raise RevisionComputeError("Unknown auxiliary timing component: {}".format(component))
        units = record.get("completed_units")
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise RevisionComputeError("Auxiliary completed_units must be positive")
        elapsed = _finite_nonnegative(
            record.get("elapsed_seconds"), "auxiliary elapsed_seconds"
        )
        if elapsed <= 0:
            raise RevisionComputeError("Auxiliary elapsed_seconds must be positive")
        gpu_count = record.get("gpu_count")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise RevisionComputeError("Auxiliary gpu_count must be a non-negative integer")
        if component == "evaluator":
            if (
                _finite_nonnegative(
                    record.get("model_load_seconds"),
                    "evaluator model_load_seconds",
                )
                <= 0
                or record.get("model_load_seconds_excluded_from_rate") is not True
                or record.get("incurred_charge_definition")
                != "elapsed_seconds_plus_model_load_seconds"
            ):
                raise RevisionComputeError(
                    "Evaluator timing lacks its measured model-load charge"
                )
        records.append({**record, "path": str(Path(path).resolve())})
    return records


def build_auxiliary_timing_record(
    component: str,
    component_id: str,
    completed_units: int,
    elapsed_seconds: float,
    gpu_count: int,
    model_id: Optional[str] = None,
    model_load_seconds: Optional[float] = None,
) -> Dict[str, object]:
    """Build the canonical self-hashed schema used by optional timing inputs."""

    value: Dict[str, object] = {
        "schema_version": AUXILIARY_TIMING_SCHEMA_VERSION,
        "component": str(component),
        "component_id": str(component_id),
        "model_id": model_id,
        "evidence_status": EVIDENCE_SMOKE_V3,
        "source_stage": EXPECTED_SMOKE_STAGE,
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "complete": True,
        "completed_units": int(completed_units),
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_count": int(gpu_count),
    }
    if component == "evaluator":
        if model_load_seconds is None or _finite_nonnegative(
            model_load_seconds, "evaluator model_load_seconds"
        ) <= 0:
            raise RevisionComputeError(
                "Evaluator timing requires a positive measured model_load_seconds"
            )
        value.update(
            {
                "model_load_seconds": float(model_load_seconds),
                "model_load_seconds_excluded_from_rate": True,
                "incurred_charge_definition": (
                    "elapsed_seconds_plus_model_load_seconds"
                ),
            }
        )
    value["timing_manifest_sha256"] = _auxiliary_digest(value)
    return value


def _auxiliary_rate(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    seconds_per_unit = [
        float(row["elapsed_seconds"]) / int(row["completed_units"]) for row in records
    ]
    summary = _rate_summary(seconds_per_unit)
    summary["gpu_count"] = max(int(row["gpu_count"]) for row in records)
    summary["input_count"] = len(records)
    return summary


def _evaluator_projections(
    rates: Mapping[str, object], auxiliary: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    by_model: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in auxiliary:
        if row["component"] == "evaluator":
            model_id = str(row.get("model_id"))
            if model_id not in EXPECTED_MODELS:
                raise RevisionComputeError("Evaluator timing has an unknown model_id")
            by_model[model_id].append(row)
    projections = []
    for model_id in EXPECTED_MODELS:
        if len(by_model.get(model_id, [])) != 1:
            raise RevisionComputeError(
                "Exactly one measured smoke-v3 evaluator timing is required for {}".format(
                    model_id
                )
            )
        timing_record = by_model[model_id][0]
        expected_prefix = "heldout_evaluator_smoke_v3_{}".format(model_id)
        if not str(timing_record.get("component_id")).startswith(expected_prefix):
            raise RevisionComputeError(
                "Evaluator timing component_id is not the frozen smoke-v3 identity"
            )
        if int(timing_record.get("gpu_count", -1)) != 1:
            raise RevisionComputeError(
                "Evaluator timing must charge exactly one GPU"
            )
        rate = _auxiliary_rate([timing_record])
        timing_source = "required_completed_evaluator_smoke_v3_timing"
        load = rates[model_id]["model_load"]  # type: ignore[index]
        point_seconds = EVALUATOR_UNITS_PER_MODEL * float(rate["point_seconds_per_unit"])
        upper_seconds = EVALUATOR_UNITS_PER_MODEL * float(rate["upper_seconds_per_unit"])
        point_seconds += float(load["point_seconds_per_unit"])
        upper_seconds += float(load["upper_seconds_per_unit"])
        gpu_count = int(rate.get("gpu_count", 1))
        projections.append(
            {
                "stage": "heldout_evaluator",
                "model_id": model_id,
                "resource_class": "gpu" if gpu_count else "cpu",
                "target_work_units": EVALUATOR_UNITS_PER_MODEL,
                "point_seconds": point_seconds,
                "upper_seconds": upper_seconds,
                "point_gpu_hours": point_seconds * gpu_count / 3600.0,
                "upper_gpu_hours": upper_seconds * gpu_count / 3600.0,
                "model_load": load,
                "rate": rate,
                "timing_source": timing_source,
            }
        )
    return projections


def _detector_projections(
    configs: Mapping[str, Mapping[str, object]],
    auxiliary: Sequence[Mapping[str, object]],
    config_dir: Path,
) -> List[Dict[str, object]]:
    detector_config = configs.get("detectors")
    # Detectors live in a nested file and are not part of load_revision_config_set
    # in older checkouts; read the frozen file explicitly in that case.
    if not isinstance(detector_config, dict):
        path = Path(config_dir) / "detectors" / "default.json"
        detector_config = _read_json(path)
    configured = {
        str(row["name"]): row
        for row in detector_config.get("detectors", [])  # type: ignore[union-attr]
        if isinstance(row, dict) and row.get("enabled", True)
    }
    if set(configured) != set(DETECTOR_FITS):
        raise RevisionComputeError("Frozen detector configuration/counts drifted")
    by_id: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in auxiliary:
        if row["component"] == "detector":
            by_id[str(row["component_id"])].append(row)
    projections = []
    for detector_id, target_fits in DETECTOR_FITS.items():
        config = configured[detector_id]
        device = str(config.get("device", "cpu")).lower()
        if by_id.get(detector_id):
            rate = _auxiliary_rate(by_id[detector_id])
            point_seconds = target_fits * float(rate["point_seconds_per_unit"])
            upper_seconds = target_fits * float(rate["upper_seconds_per_unit"])
            gpu_count = int(rate["gpu_count"])
            if (device == "cpu" and gpu_count != 0) or (
                device != "cpu" and gpu_count == 0
            ):
                raise RevisionComputeError(
                    "Detector timing resource disagrees with frozen device for {}".format(
                        detector_id
                    )
                )
            source = "optional_completed_detector_smoke_timing"
        elif device == "cpu":
            rate = None
            point_seconds = None
            upper_seconds = None
            gpu_count = 0
            source = "configured_cpu_optional_wall_timing_not_supplied"
        else:
            raise RevisionComputeError(
                "GPU detector {} requires an auxiliary timing input".format(detector_id)
            )
        projections.append(
            {
                "stage": "neural_detector",
                "model_id": detector_id,
                "resource_class": "gpu" if gpu_count else "cpu",
                "target_work_units": target_fits,
                "epochs_per_fit": int(config.get("epochs", 1)),
                "point_seconds": point_seconds,
                "upper_seconds": upper_seconds,
                "point_gpu_hours": (
                    float(point_seconds) * gpu_count / 3600.0
                    if point_seconds is not None
                    else 0.0
                ),
                "upper_gpu_hours": (
                    float(upper_seconds) * gpu_count / 3600.0
                    if upper_seconds is not None
                    else 0.0
                ),
                "rate": rate,
                "timing_source": source,
                "gpu_budget_note": (
                    "CPU-configured detector contributes zero GPU-hours; wall time is unknown "
                    "until optional timing is supplied."
                    if point_seconds is None
                    else None
                ),
            }
        )
    return projections


def _base_report(budget_gpu_hours: float) -> Dict[str, object]:
    return {
        "schema_version": COMPUTE_SCHEMA_VERSION,
        "budget_gpu_hours": float(budget_gpu_hours),
        "evidence_policy": {
            "accepted_runner_label": EVIDENCE_SMOKE_V3,
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "accepted_runner_stages": [
                EXPECTED_SMOKE_STAGE, *PROJECTED_RUNNER_STAGES
            ],
            "legacy_artifacts_are_incurred_charge_only": True,
            "smoke_is_exploratory_only": True,
            "confirmatory_pooling_permitted": False,
            "gate_is_compute_only": True,
        },
    }


def _legacy_ledger_digest(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop(LEGACY_INCURRED_LEDGER_HASH_FIELD, None)
    return canonical_json_sha256(unsigned)


def _reject_symlink_path(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RevisionComputeError("{} traverses a symlink".format(label))
    return candidate


def _verified_directory_binding(root: Path) -> Dict[str, object]:
    candidate = _reject_symlink_path(Path(root), "Legacy charge source")
    if not candidate.is_dir():
        raise RevisionComputeError("Legacy charge source is not a regular directory")
    if any(path.is_symlink() for path in candidate.rglob("*")):
        raise RevisionComputeError("Legacy charge source contains a symlink")
    resolved = candidate.resolve(strict=True)
    manifest = build_directory_manifest(resolved, exclude_paths=())
    report = verify_directory_manifest(
        resolved, manifest, require_no_extra_files=True, ignored_extra_paths=()
    )
    if report.get("status") != "ok":
        raise RevisionComputeError(
            "Legacy charge directory manifest failed: {}".format(
                report.get("errors")
            )
        )
    return {"absolute_path": str(resolved), "directory_manifest": manifest}


def _timestamp(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RevisionComputeError("{} must be ISO-8601".format(label)) from exc
    if parsed.tzinfo is None:
        raise RevisionComputeError("{} must include a timezone".format(label))
    return parsed


def _legacy_smoke_charge(root: Path, model_id: str) -> Dict[str, object]:
    binding = _verified_directory_binding(root)
    resolved = Path(binding["absolute_path"])
    required = set(REQUIRED_SHARD_FILES)
    listed = {
        str(row["path"])
        for row in binding["directory_manifest"]["files"]
    }
    if not required.issubset(listed):
        raise RevisionComputeError("Legacy smoke shard lacks required files")
    plan = _read_jsonl(resolved / "plan.jsonl")
    expected_plan = build_stage_plan(LEGACY_SMOKE_STAGE, model_id=model_id)
    if canonical_json_sha256(plan) != canonical_json_sha256(expected_plan):
        raise RevisionComputeError("Legacy smoke plan is not the frozen smoke-v2 plan")
    if any(row.get("evidence_status") != LEGACY_EVIDENCE_STATUS for row in plan):
        raise RevisionComputeError("Legacy smoke plan has an unexpected evidence label")
    identity = _read_json(resolved / "run_identity.json")
    if identity.get("run_identity_sha256") != canonical_json_sha256(
        _identity_without_digest(identity)
    ):
        raise RevisionComputeError("Legacy smoke run identity self-hash mismatch")
    if identity.get("study_id") != "{}/{}/{}".format(
        REVISION_CORPUS_ID, LEGACY_SMOKE_STAGE, model_id
    ):
        raise RevisionComputeError("Legacy smoke run identity mismatch")
    args = _scientific_args(identity)
    if (
        args.get("stage") != LEGACY_SMOKE_STAGE
        or args.get("model_id") != model_id
        or args.get("evidence_status") != LEGACY_EVIDENCE_STATUS
        or args.get("source_manifest_sha256")
        != LEGACY_RUNNER_SOURCE_MANIFEST_SHA256
        or args.get("runtime_manifest_sha256") != LEGACY_RUNNER_RUNTIME_SHA256
        or args.get("hardware_manifest_sha256") != LEGACY_RUNNER_HARDWARE_SHA256
        or args.get("input_results_manifest_sha256") != "none"
    ):
        raise RevisionComputeError("Legacy smoke scientific arguments mismatch")
    source_manifest = _read_json(resolved / "source_manifest.json")
    if (
        identity.get("config_manifest_sha256")
        != LEGACY_CONFIG_MANIFEST_SHA256
        or file_sha256(resolved / "payload_manifest.json")
        != LEGACY_RUNNER_PAYLOAD_SHA256
        or source_manifest.get("files_sha256")
        != LEGACY_RUNNER_SOURCE_FILES_SHA256
        or file_sha256(resolved / "source_manifest.json")
        != LEGACY_RUNNER_SOURCE_MANIFEST_SHA256
        or file_sha256(resolved / "runtime_manifest.json")
        != LEGACY_RUNNER_RUNTIME_SHA256
        or file_sha256(resolved / "hardware_manifest.json")
        != LEGACY_RUNNER_HARDWARE_SHA256
    ):
        raise RevisionComputeError("Legacy smoke frozen artifact hash mismatch")
    configured_models = load_revision_config_set()["models"]["models"]
    expected_model = next(
        row for row in configured_models if row["model_id"] == model_id
    )
    model_manifest = _read_json(resolved / "model_manifest.json")
    if model_manifest.get("configured_model") != expected_model:
        raise RevisionComputeError("Legacy smoke model pin differs from frozen config")
    _validate_manifest_shapes(resolved, identity)
    checkpoint = load_checkpoint(resolved / "checkpoint.json")
    ids = [str(row["work_id"]) for row in plan]
    if (
        identity.get("planned_trial_count") != len(ids)
        or identity.get("planned_trial_ids_sha256") != trial_ids_sha256(ids)
        or checkpoint.get("study_id") != identity.get("study_id")
        or checkpoint.get("config_manifest_sha256")
        != identity.get("config_manifest_sha256")
        or checkpoint.get("planned_trial_count") != len(ids)
        or checkpoint.get("planned_trial_ids_sha256") != trial_ids_sha256(ids)
        or set(map(str, checkpoint.get("completed_trial_ids", []))) != set(ids)
        or checkpoint.get("failed_trial_ids")
    ):
        raise RevisionComputeError("Legacy smoke checkpoint/identity is not complete")
    records = _read_jsonl(resolved / "records.jsonl")
    completed: Dict[str, Mapping[str, object]] = {}
    for record in records:
        work_id = str(record.get("work_id"))
        if work_id not in set(ids):
            raise RevisionComputeError("Legacy smoke contains an unplanned record")
        if record.get("execution_status") != "completed":
            raise RevisionComputeError("Legacy smoke contains a non-completion record")
        if work_id in completed:
            raise RevisionComputeError("Legacy smoke contains duplicate completions")
        if (
            record.get("model_id") != model_id
            or record.get("evidence_status") != LEGACY_EVIDENCE_STATUS
        ):
            raise RevisionComputeError("Legacy smoke record identity mismatch")
        completed[work_id] = record
    if set(completed) != set(ids):
        raise RevisionComputeError("Legacy smoke lacks one-to-one completions")
    hardware = _read_json(resolved / "hardware_manifest.json")
    gpu_uuid = str(hardware.get("selected_gpu_uuid"))
    if not gpu_uuid.startswith("GPU-") or args.get("gpu_uuid") != gpu_uuid:
        raise RevisionComputeError("Legacy smoke GPU binding mismatch")
    events = _read_jsonl(resolved / "events.jsonl")
    load_events = [row for row in events if row.get("event") == "model_loaded"]
    if (
        len(load_events) != 1
        or load_events[0].get("model_id") != model_id
        or str(load_events[0].get("gpu_uuid")) != gpu_uuid
        or _finite_nonnegative(
            load_events[0].get("model_load_seconds"), "legacy model load"
        )
        <= 0
    ):
        raise RevisionComputeError("Legacy smoke model-load event mismatch")
    profiles = [row for row in events if row.get("event") == "memory_profile"]
    if len(profiles) != 1:
        raise RevisionComputeError("Legacy smoke requires exactly one memory profile")
    profile = profiles[0]
    started = _timestamp(profile.get("started_at"), "legacy smoke started_at")
    ended = _timestamp(profile.get("at"), "legacy smoke at")
    elapsed = (ended - started).total_seconds()
    if elapsed <= 0 or str(profile.get("selected_gpu_uuid")) != gpu_uuid:
        raise RevisionComputeError("Legacy smoke memory profile is invalid")
    for field in ("sample_count", "selected_gpu_sample_count", "process_rss_sample_count"):
        if not isinstance(profile.get(field), int) or int(profile[field]) <= 0:
            raise RevisionComputeError("Legacy smoke profile lacks positive samples")
    finished = [row for row in events if row.get("event") == "session_finished"]
    event_times = [_timestamp(row.get("at"), "legacy smoke event time") for row in events]
    if (
        len(finished) != 1
        or event_times != sorted(event_times)
        or int(finished[0].get("planned", -1)) != len(ids)
        or int(finished[0].get("completed", -1)) != len(ids)
        or int(finished[0].get("failed_current", -1)) != 0
        or int(finished[0].get("remaining", -1)) != 0
        or ended > _timestamp(finished[0].get("at"), "legacy smoke finished at")
    ):
        raise RevisionComputeError("Legacy smoke lacks one ordered completed session event")
    return {
        "component": "legacy_smoke_v2",
        "model_id": model_id,
        "scientific_use": "incurred_gpu_charge_only_never_rate_evidence",
        "charge_only_not_rate_evidence": True,
        "scientific_evidence_allowed": False,
        "rate_evidence_allowed": False,
        "charge_policy": "memory_profile_wall_span_v1",
        "incurred_gpu_seconds": elapsed,
        "occupancy_started_at": profile["started_at"],
        "occupancy_ended_at": profile["at"],
        "gpu_uuid": gpu_uuid,
        "run_identity_sha256": identity["run_identity_sha256"],
        "plan_sha256": canonical_json_sha256(plan),
        **binding,
    }


def _legacy_evaluator_charge(root: Path, model_id: str) -> Dict[str, object]:
    binding = _verified_directory_binding(root)
    resolved = Path(binding["absolute_path"])
    timing = _read_json(resolved / "auxiliary_timing.json")
    if timing.get("schema_version") != "rankcloak-revision-compute-timing-v1":
        raise RevisionComputeError("Legacy evaluator timing schema mismatch")
    if (
        timing.get("timing_manifest_sha256") != _auxiliary_digest(timing)
        or timing.get("timing_manifest_sha256")
        != LEGACY_EVALUATOR_TIMING_SHA256[model_id]
    ):
        raise RevisionComputeError("Legacy evaluator timing self-hash mismatch")
    if (
        timing.get("source_stage") != LEGACY_SMOKE_STAGE
        or timing.get("evidence_status") != LEGACY_EVIDENCE_STATUS
        or timing.get("component") != "evaluator"
        or timing.get("complete") is not True
        or timing.get("confirmatory_pooling_eligible") is not False
        or timing.get("model_id") != model_id
        or int(timing.get("gpu_count", -1)) != 1
    ):
        raise RevisionComputeError("Legacy evaluator timing contract mismatch")
    if not str(timing.get("component_id")).startswith(
        "heldout_evaluator_smoke_v2_{}".format(model_id)
    ):
        raise RevisionComputeError("Legacy evaluator component identity mismatch")
    if timing.get("records_jsonl_sha256") != file_sha256(
        resolved / "records.jsonl"
    ):
        raise RevisionComputeError("Legacy evaluator records hash mismatch")
    elapsed = _finite_nonnegative(
        timing.get("elapsed_seconds"), "legacy evaluator elapsed_seconds"
    )
    if elapsed <= 0 or timing.get("model_load_seconds_excluded") is not True:
        raise RevisionComputeError("Legacy evaluator elapsed-time contract mismatch")
    identity = _read_json(resolved / "run_identity.json")
    if identity.get("run_identity_sha256") != canonical_json_sha256(
        _identity_without_digest(identity)
    ):
        raise RevisionComputeError("Legacy evaluator run identity self-hash mismatch")
    args = _scientific_args(identity)
    if (
        args.get("evaluator_model_id") != model_id
        or args.get("source_stages") != LEGACY_SMOKE_STAGE
    ):
        raise RevisionComputeError("Legacy evaluator run identity mismatch")
    if (
        identity.get("config_manifest_sha256")
        != LEGACY_CONFIG_MANIFEST_SHA256
        or file_sha256(resolved / "source_manifest.json")
        != LEGACY_EVALUATOR_SOURCE_MANIFEST_SHA256
        or file_sha256(resolved / "runtime_manifest.json")
        != LEGACY_EVALUATOR_RUNTIME_SHA256
        or file_sha256(resolved / "hardware_manifest.json")
        != LEGACY_EVALUATOR_HARDWARE_SHA256
    ):
        raise RevisionComputeError("Legacy evaluator frozen artifact hash mismatch")
    expected_generator_by_evaluator = {
        EXPECTED_MODELS[0]: EXPECTED_MODELS[2],
        EXPECTED_MODELS[1]: EXPECTED_MODELS[0],
        EXPECTED_MODELS[2]: EXPECTED_MODELS[1],
    }
    if (
        args.get("generator_model_id") != expected_generator_by_evaluator[model_id]
        or timing.get("generator_model_id")
        != expected_generator_by_evaluator[model_id]
    ):
        raise RevisionComputeError("Legacy evaluator generator mapping mismatch")
    plan = _read_jsonl(resolved / "plan.jsonl")
    ids = [str(row.get("evaluation_id")) for row in plan]
    if (
        len(ids) != len(set(ids))
        or identity.get("planned_trial_count") != len(ids)
        or identity.get("planned_trial_ids_sha256") != trial_ids_sha256(ids)
        or timing.get("ordered_evaluation_ids_sha256") != trial_ids_sha256(ids)
        or timing.get("scoring_plan_content_sha256") != canonical_json_sha256(plan)
    ):
        raise RevisionComputeError("Legacy evaluator ordered plan mismatch")
    checkpoint = load_checkpoint(resolved / "checkpoint.json")
    if (
        checkpoint.get("study_id") != identity.get("study_id")
        or checkpoint.get("config_manifest_sha256")
        != identity.get("config_manifest_sha256")
        or checkpoint.get("planned_trial_count") != len(ids)
        or checkpoint.get("planned_trial_ids_sha256") != trial_ids_sha256(ids)
        or set(map(str, checkpoint.get("completed_trial_ids", []))) != set(ids)
        or checkpoint.get("failed_trial_ids")
    ):
        raise RevisionComputeError("Legacy evaluator checkpoint/identity is incomplete")
    records = _read_jsonl(resolved / "records.jsonl")
    completions: Dict[str, Mapping[str, object]] = {}
    for record in records:
        evaluation_id = str(record.get("evaluation_id"))
        if evaluation_id not in set(ids):
            raise RevisionComputeError("Legacy evaluator contains an unplanned record")
        if (
            record.get("execution_status") != "completed"
            or record.get("record_type") != "heldout_evaluator_feature"
        ):
            raise RevisionComputeError("Legacy evaluator contains a non-feature record")
        if evaluation_id in completions:
            raise RevisionComputeError("Legacy evaluator duplicate completion")
        if (
            record.get("evaluator_model_id") != model_id
            or record.get("generator_model_id")
            != expected_generator_by_evaluator[model_id]
            or record.get("evidence_status") != LEGACY_EVIDENCE_STATUS
            or record.get("confirmatory_pooling_eligible") is not False
        ):
            raise RevisionComputeError("Legacy evaluator record identity mismatch")
        completions[evaluation_id] = record
    if set(completions) != set(ids):
        raise RevisionComputeError("Legacy evaluator lacks one-to-one completions")
    durable_elapsed = sum(float(completions[value]["wall_seconds"]) for value in ids)
    if durable_elapsed != elapsed:
        raise RevisionComputeError("Legacy evaluator elapsed time is not record-derived")
    evaluator_manifest = _read_json(resolved / "evaluator_model_manifest.json")
    configured_models = load_revision_config_set()["models"]["models"]
    expected_model = next(
        row for row in configured_models if row["model_id"] == model_id
    )
    configured = evaluator_manifest.get("configured_model")
    verification = evaluator_manifest.get("verification")
    if (
        identity.get("model_artifacts") != [evaluator_manifest]
        or not isinstance(configured, dict)
        or configured != expected_model
        or configured.get("model_id") != model_id
        or not isinstance(verification, dict)
        or verification.get("status") != "ok"
        or verification.get("actual_sha256") != configured.get("artifact_sha256")
        or verification.get("expected_sha256") != configured.get("artifact_sha256")
    ):
        raise RevisionComputeError("Legacy evaluator model pin mismatch")
    input_path = resolved / "input_results_manifest.json"
    if (
        identity.get("payload_manifest_sha256") != file_sha256(input_path)
        or timing.get("input_results_manifest_sha256") != file_sha256(input_path)
        or timing.get("config_manifest_sha256")
        != identity.get("config_manifest_sha256")
    ):
        raise RevisionComputeError("Legacy evaluator input/config binding mismatch")
    source_manifest = _read_json(resolved / "source_manifest.json")
    files = source_manifest.get("files")
    if (
        not isinstance(files, list)
        or source_manifest.get("files_sha256") != canonical_json_sha256(files)
        or source_manifest.get("files_sha256")
        != LEGACY_EVALUATOR_SOURCE_FILES_SHA256
    ):
        raise RevisionComputeError("Legacy evaluator source manifest mismatch")
    for argument, filename in (
        ("source_manifest_sha256", "source_manifest.json"),
        ("runtime_manifest_sha256", "runtime_manifest.json"),
        ("hardware_manifest_sha256", "hardware_manifest.json"),
    ):
        if args.get(argument) != file_sha256(resolved / filename):
            raise RevisionComputeError("Legacy evaluator bound manifest mismatch")
    hardware = _read_json(resolved / "hardware_manifest.json")
    gpu_uuid = str(hardware.get("selected_gpu_uuid"))
    if not gpu_uuid.startswith("GPU-") or args.get("gpu_uuid") != gpu_uuid:
        raise RevisionComputeError("Legacy evaluator GPU binding mismatch")
    events = _read_jsonl(resolved / "events.jsonl")
    loads = [row for row in events if row.get("event") == "evaluator_model_loaded"]
    if len(loads) != 1:
        raise RevisionComputeError("Legacy evaluator requires one model-load event")
    load = _finite_nonnegative(loads[0].get("model_load_seconds"), "model load")
    if (
        load <= 0
        or loads[0].get("evaluator_model_id") != model_id
        or str(loads[0].get("gpu_uuid")) != gpu_uuid
    ):
        raise RevisionComputeError("Legacy evaluator model-load binding mismatch")
    finished = [
        row for row in events if row.get("event") == "evaluator_session_finished"
    ]
    event_times = [_timestamp(row.get("at"), "legacy evaluator event time") for row in events]
    if (
        len(finished) != 1
        or event_times != sorted(event_times)
        or int(finished[0].get("planned", -1)) != len(ids)
        or int(finished[0].get("completed", -1)) != len(ids)
        or int(finished[0].get("failed_current", -1)) != 0
        or int(finished[0].get("remaining", -1)) != 0
        or _timestamp(loads[0].get("at"), "legacy evaluator load at")
        > _timestamp(finished[0].get("at"), "legacy evaluator finished at")
    ):
        raise RevisionComputeError("Legacy evaluator lacks one ordered completed session event")
    return {
        "component": "legacy_heldout_evaluator_smoke_v2",
        "model_id": model_id,
        "scientific_use": "incurred_gpu_charge_only_never_rate_evidence",
        "charge_only_not_rate_evidence": True,
        "scientific_evidence_allowed": False,
        "rate_evidence_allowed": False,
        "charge_policy": "durable_task_elapsed_plus_model_load_v1",
        "durable_task_elapsed_seconds": elapsed,
        "model_load_seconds": load,
        "incurred_gpu_seconds": elapsed + load,
        "occupancy_started_at": (
            _timestamp(loads[0].get("at"), "legacy evaluator load at")
            - timedelta(seconds=load)
        ).isoformat(),
        "occupancy_ended_at": finished[0]["at"],
        "gpu_uuid": gpu_uuid,
        "timing_manifest_sha256": timing["timing_manifest_sha256"],
        "run_identity_sha256": identity["run_identity_sha256"],
        **binding,
    }


def build_legacy_incurred_charge_ledger(
    smoke_v2_shards: Sequence[Path],
    evaluator_smoke_v2_dirs: Sequence[Path],
    created_at: str,
) -> Dict[str, object]:
    if len(smoke_v2_shards) != len(EXPECTED_MODELS):
        raise RevisionComputeError("Legacy ledger requires three smoke-v2 shards")
    if len(evaluator_smoke_v2_dirs) != len(EXPECTED_MODELS):
        raise RevisionComputeError("Legacy ledger requires three evaluator directories")
    smoke_by_model = {Path(path).name: Path(path) for path in smoke_v2_shards}
    evaluator_by_model = {
        Path(path).name: Path(path) for path in evaluator_smoke_v2_dirs
    }
    if set(smoke_by_model) != set(EXPECTED_MODELS):
        raise RevisionComputeError("Legacy smoke directories do not cover all models")
    if set(evaluator_by_model) != set(EXPECTED_MODELS):
        raise RevisionComputeError("Legacy evaluator directories do not cover all models")
    entries = [
        _legacy_smoke_charge(smoke_by_model[model], model)
        for model in EXPECTED_MODELS
    ] + [
        _legacy_evaluator_charge(evaluator_by_model[model], model)
        for model in EXPECTED_MODELS
    ]
    paths = [str(row["absolute_path"]) for row in entries]
    identities = [str(row["run_identity_sha256"]) for row in entries]
    if len(paths) != len(set(paths)) or len(identities) != len(set(identities)):
        raise RevisionComputeError("Legacy incurred ledger contains duplicate sources")
    intervals_by_gpu: Dict[str, List[Tuple[datetime, datetime, str]]] = defaultdict(list)
    for row in entries:
        started = _timestamp(row["occupancy_started_at"], "legacy occupancy start")
        ended = _timestamp(row["occupancy_ended_at"], "legacy occupancy end")
        if ended <= started:
            raise RevisionComputeError("Legacy incurred occupancy interval is invalid")
        if float(row["incurred_gpu_seconds"]) > (ended - started).total_seconds():
            raise RevisionComputeError("Legacy incurred charge exceeds occupancy span")
        intervals_by_gpu[str(row["gpu_uuid"])].append(
            (started, ended, str(row["absolute_path"]))
        )
    for intervals in intervals_by_gpu.values():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise RevisionComputeError("Legacy incurred GPU intervals overlap")
    total_seconds = sum(float(row["incurred_gpu_seconds"]) for row in entries)
    value: Dict[str, object] = {
        "schema_version": LEGACY_INCURRED_LEDGER_SCHEMA_VERSION,
        "manifest_type": "legacy_incurred_gpu_charge_ledger",
        "artifact_role": "charge_only_not_rate_evidence",
        "charge_only_not_rate_evidence": True,
        "created_at": str(created_at),
        "scientific_use": "incurred_charge_only_never_timing_rate_or_evidence",
        "scientific_evidence_allowed": False,
        "rate_evidence_allowed": False,
        "entry_count": len(entries),
        "entries": entries,
        "total_incurred_gpu_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
    }
    _timestamp(value["created_at"], "legacy ledger created_at")
    value[LEGACY_INCURRED_LEDGER_HASH_FIELD] = _legacy_ledger_digest(value)
    return value


def verify_legacy_incurred_charge_ledger(path: Path) -> Dict[str, object]:
    ledger_path = _reject_symlink_path(Path(path), "Legacy incurred ledger")
    if not ledger_path.is_file():
        raise RevisionComputeError("Legacy incurred ledger is not a regular file")
    value = _read_json(ledger_path)
    if value.get("schema_version") != LEGACY_INCURRED_LEDGER_SCHEMA_VERSION:
        raise RevisionComputeError("Legacy incurred ledger schema mismatch")
    if value.get("manifest_type") != "legacy_incurred_gpu_charge_ledger":
        raise RevisionComputeError("Legacy incurred ledger type mismatch")
    if (
        value.get("scientific_use")
        != "incurred_charge_only_never_timing_rate_or_evidence"
        or value.get("artifact_role") != "charge_only_not_rate_evidence"
        or value.get("charge_only_not_rate_evidence") is not True
        or value.get("scientific_evidence_allowed") is not False
        or value.get("rate_evidence_allowed") is not False
    ):
        raise RevisionComputeError("Legacy incurred ledger use policy mismatch")
    if value.get(LEGACY_INCURRED_LEDGER_HASH_FIELD) != _legacy_ledger_digest(value):
        raise RevisionComputeError("Legacy incurred ledger self-hash mismatch")
    _timestamp(value.get("created_at"), "legacy ledger created_at")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 2 * len(EXPECTED_MODELS):
        raise RevisionComputeError("Legacy incurred ledger entry count mismatch")
    recomputed: List[Dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RevisionComputeError("Legacy incurred ledger entry is malformed")
        root = Path(str(entry.get("absolute_path")))
        model_id = str(entry.get("model_id"))
        if entry.get("component") == "legacy_smoke_v2":
            actual = _legacy_smoke_charge(root, model_id)
        elif entry.get("component") == "legacy_heldout_evaluator_smoke_v2":
            actual = _legacy_evaluator_charge(root, model_id)
        else:
            raise RevisionComputeError("Unknown legacy incurred ledger component")
        if actual != entry:
            raise RevisionComputeError("Legacy incurred artifact changed after ledger creation")
        if (
            entry.get("charge_only_not_rate_evidence") is not True
            or entry.get("scientific_evidence_allowed") is not False
            or entry.get("rate_evidence_allowed") is not False
        ):
            raise RevisionComputeError("Legacy incurred entry permits scientific use")
        recomputed.append(actual)
    if int(value.get("entry_count", -1)) != len(recomputed):
        raise RevisionComputeError("Legacy incurred ledger entry_count mismatch")
    expected_components = {
        (component, model_id)
        for component in (
            "legacy_smoke_v2",
            "legacy_heldout_evaluator_smoke_v2",
        )
        for model_id in EXPECTED_MODELS
    }
    observed_components = {
        (str(row["component"]), str(row["model_id"])) for row in recomputed
    }
    paths = [str(row["absolute_path"]) for row in recomputed]
    identities = [str(row["run_identity_sha256"]) for row in recomputed]
    if (
        observed_components != expected_components
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
    ):
        raise RevisionComputeError("Legacy incurred ledger coverage/uniqueness mismatch")
    intervals_by_gpu: Dict[str, List[Tuple[datetime, datetime, str]]] = defaultdict(list)
    for row in recomputed:
        started = _timestamp(row["occupancy_started_at"], "legacy occupancy start")
        ended = _timestamp(row["occupancy_ended_at"], "legacy occupancy end")
        intervals_by_gpu[str(row["gpu_uuid"])].append(
            (started, ended, str(row["absolute_path"]))
        )
    for intervals in intervals_by_gpu.values():
        ordered = sorted(intervals)
        if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise RevisionComputeError("Legacy incurred GPU intervals overlap")
    total = sum(float(row["incurred_gpu_seconds"]) for row in recomputed)
    if (
        value.get("total_incurred_gpu_seconds") != total
        or value.get("total_hours") != total / 3600.0
    ):
        raise RevisionComputeError("Legacy incurred ledger total mismatch")
    return {
        "status": "ok",
        "path": str(ledger_path.resolve(strict=True)),
        LEGACY_INCURRED_LEDGER_HASH_FIELD: value[LEGACY_INCURRED_LEDGER_HASH_FIELD],
        "ledger_sha256": value[LEGACY_INCURRED_LEDGER_HASH_FIELD],
        "created_at": value["created_at"],
        "entries": recomputed,
        "total_incurred_gpu_seconds": total,
        "total_seconds": total,
        "total_hours": total / 3600.0,
        "scientific_use": value["scientific_use"],
        "charge_only_not_rate_evidence": True,
        "scientific_evidence_allowed": False,
        "rate_evidence_allowed": False,
    }


def verify_legacy_gpu_ledger(path: Path) -> Dict[str, object]:
    return verify_legacy_incurred_charge_ledger(path)


def _verified_invalidation_charges(paths: Sequence[Path]) -> List[Dict[str, object]]:
    if len(paths) != 1:
        raise RevisionComputeError(
            "Exactly one verified invalidation registry entry is required"
        )
    entry_path = _reject_symlink_path(
        Path(paths[0]), "Invalidation registry entry"
    )
    value = verify_invalidation_entry(entry_path)
    if value.get("status") != "ok":
        raise RevisionComputeError("Invalidation verifier did not return status=ok")
    if value.get("scientific_status") != "invalidated_not_for_pooling":
        raise RevisionComputeError("Invalidation is not excluded from pooling")
    if value.get("charge_policy") != "memory_profile_wall_span_v1":
        raise RevisionComputeError("Invalidation uses an unapproved charge policy")
    if (
        value.get("invalidation_manifest_sha256")
        != EXPECTED_INVALIDATION_MANIFEST_SHA256
        or value.get("run_identity_sha256")
        != EXPECTED_INVALIDATED_RUN_IDENTITY_SHA256
        or value.get("shard_tree_sha256")
        != EXPECTED_INVALIDATED_SHARD_TREE_SHA256
    ):
        raise RevisionComputeError("Invalidation identity is not the frozen Qwen shard")
    stages = set(map(str, value.get("superseding_stages", [])))
    if stages != {"smoke_v3", "primary_v2"}:
        raise RevisionComputeError(
            "Invalidation does not bind exactly the smoke-v3/primary-v2 supersession"
        )
    execution_state = value.get("execution_state")
    if (
        not isinstance(execution_state, dict)
        or execution_state.get("terminal_state") != "stopped_incomplete"
        or execution_state.get("incomplete") is not True
        or int(execution_state.get("planned_work_units", -1)) != 4800
        or int(execution_state.get("completed_work_units", -1)) != 234
        or int(execution_state.get("remaining_work_units", -1)) != 4566
    ):
        raise RevisionComputeError("Invalidation is not the frozen stopped shard state")
    seconds = _finite_nonnegative(
        value.get("incurred_gpu_seconds"), "invalidation incurred_gpu_seconds"
    )
    if seconds != EXPECTED_INVALIDATED_GPU_SECONDS:
        raise RevisionComputeError("Invalidation GPU charge differs from the frozen value")

    # The public verifier authenticates the full manifest and in-place shard.
    # Read its already-verified timing fields only to enforce cross-artifact
    # duplicate/overlap checks; they are never used as rate evidence.
    supplied_intervals = value.get("occupancy_intervals")
    intervals: List[Dict[str, str]] = []
    span_total = 0.0
    if supplied_intervals is not None:
        # Test doubles may expose the verifier-authenticated compact interval
        # directly. The production verifier deliberately does not, so the
        # production path below reads only the already-authenticated manifest.
        if not isinstance(supplied_intervals, list) or not supplied_intervals:
            raise RevisionComputeError("Invalidation lacks verified GPU occupancy spans")
        for span in supplied_intervals:
            if not isinstance(span, dict):
                raise RevisionComputeError("Invalidation occupancy span is malformed")
            gpu_uuid = str(span.get("gpu_uuid"))
            started = _timestamp(span.get("started_at"), "invalidation span start")
            ended = _timestamp(span.get("ended_at"), "invalidation span end")
            if not gpu_uuid.startswith("GPU-") or ended <= started:
                raise RevisionComputeError("Invalidation occupancy span is invalid")
            elapsed = (ended - started).total_seconds()
            span_total += elapsed
            intervals.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                }
            )
    else:
        raw = _read_json(entry_path)
        incurred = raw.get("incurred_compute")
        if not isinstance(incurred, dict):
            raise RevisionComputeError("Invalidation lacks incurred-compute details")
        spans = incurred.get("memory_profile_spans")
        gpu_uuid = str(incurred.get("selected_gpu_uuid"))
        if not isinstance(spans, list) or not spans or not gpu_uuid.startswith("GPU-"):
            raise RevisionComputeError("Invalidation lacks verified GPU occupancy spans")
        for span in spans:
            if not isinstance(span, dict) or str(span.get("selected_gpu_uuid")) != gpu_uuid:
                raise RevisionComputeError("Invalidation occupancy GPU binding mismatch")
            started = _timestamp(span.get("started_at"), "invalidation span start")
            ended = _timestamp(span.get("ended_at"), "invalidation span end")
            if ended <= started:
                raise RevisionComputeError("Invalidation occupancy span is invalid")
            elapsed = _finite_nonnegative(
                span.get("elapsed_seconds"), "invalidation span elapsed"
            )
            if elapsed != (ended - started).total_seconds():
                raise RevisionComputeError("Invalidation occupancy elapsed mismatch")
            span_total += elapsed
            intervals.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                }
            )
    if span_total != seconds:
        raise RevisionComputeError("Invalidation occupancy total mismatch")
    result = dict(value)
    if supplied_intervals is not None:
        registry_entry_path = str(value.get("registry_entry_path", ""))
        if not registry_entry_path:
            raise RevisionComputeError("Invalidation verifier omitted registry path")
    else:
        registry_entry_path = str(entry_path.resolve(strict=True))
    result["registry_entry_path"] = registry_entry_path
    result["charge_only_not_rate_evidence"] = True
    result["scientific_evidence_allowed"] = False
    result["rate_evidence_allowed"] = False
    result["occupancy_intervals"] = intervals
    return [result]


def _validate_combined_incurred_charges(
    legacy: Mapping[str, object],
    invalidations: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    legacy_entries = legacy.get("entries")
    if not isinstance(legacy_entries, list) or len(legacy_entries) != 6:
        raise RevisionComputeError("Legacy incurred ledger does not contain six entries")
    if len(invalidations) != 1:
        raise RevisionComputeError("Combined incurred audit requires one invalidation")
    invalidation = invalidations[0]
    legacy_paths = {str(row.get("absolute_path")) for row in legacy_entries}
    legacy_identities = {
        str(row.get("run_identity_sha256")) for row in legacy_entries
    }
    invalidated_path = str(invalidation.get("shard_path"))
    invalidated_identity = str(invalidation.get("run_identity_sha256"))
    if invalidated_path in legacy_paths or invalidated_identity in legacy_identities:
        raise RevisionComputeError(
            "Legacy ledger and invalidation double-count the same source identity"
        )
    registry_path = str(invalidation.get("registry_entry_path"))
    if registry_path in legacy_paths:
        raise RevisionComputeError("Invalidation registry is inside a charged source")

    intervals_by_gpu: Dict[str, List[Tuple[datetime, datetime, str]]] = defaultdict(list)
    for row in legacy_entries:
        started = _timestamp(row.get("occupancy_started_at"), "legacy occupancy start")
        ended = _timestamp(row.get("occupancy_ended_at"), "legacy occupancy end")
        intervals_by_gpu[str(row.get("gpu_uuid"))].append(
            (started, ended, str(row.get("absolute_path")))
        )
    for row in invalidation.get("occupancy_intervals", []):
        if not isinstance(row, dict):
            raise RevisionComputeError("Invalidation occupancy interval is malformed")
        started = _timestamp(row.get("started_at"), "invalidation occupancy start")
        ended = _timestamp(row.get("ended_at"), "invalidation occupancy end")
        intervals_by_gpu[str(row.get("gpu_uuid"))].append(
            (started, ended, invalidated_path)
        )
    for intervals in intervals_by_gpu.values():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise RevisionComputeError(
                    "Legacy and invalidated incurred GPU intervals overlap"
                )
    legacy_seconds = float(legacy["total_incurred_gpu_seconds"])
    invalidated_seconds = sum(
        float(row["incurred_gpu_seconds"]) for row in invalidations
    )
    return {
        "status": "ok_no_duplicate_paths_identities_or_gpu_intervals",
        "legacy_seconds": legacy_seconds,
        "invalidated_seconds": invalidated_seconds,
        "combined_seconds": legacy_seconds + invalidated_seconds,
        "combined_gpu_hours": (legacy_seconds + invalidated_seconds) / 3600.0,
    }


def project_revision_compute(
    smoke_shards: Sequence[Path],
    auxiliary_timing_paths: Sequence[Path] = (),
    invalidation_manifest_paths: Sequence[Path] = (),
    legacy_incurred_ledger_path: Optional[Path] = None,
    budget_gpu_hours: float = DEFAULT_BUDGET_GPU_HOURS,
    config_dir: Path = DEFAULT_REVISION_CONFIG_DIR,
) -> Dict[str, object]:
    """Return a complete point/upper projection and a fail-closed gate decision."""

    report = _base_report(budget_gpu_hours)
    errors: List[str] = []
    try:
        budget = _finite_nonnegative(budget_gpu_hours, "budget_gpu_hours")
        if budget <= 0:
            raise RevisionComputeError("budget_gpu_hours must be positive")
        configs = load_revision_config_set(Path(config_dir))
        configured_models = tuple(model_ids(configs["models"]))
        if configured_models != EXPECTED_MODELS:
            raise RevisionComputeError("Frozen model order or identities drifted")
        config_verification = verify_config_manifest(Path(config_dir))
        plans = load_frozen_plans(configs)
        candidates = [Path(path) for path in smoke_shards]
        if len(candidates) != len(EXPECTED_MODELS):
            raise RevisionComputeError(
                "Exactly three completed smoke shards are required; received {}".format(
                    len(candidates)
                )
            )
        verified = []
        seen = set()
        smoke_by_model = {
            model: [
                row for row in plans[EXPECTED_SMOKE_STAGE] if row["model_id"] == model
            ]
            for model in EXPECTED_MODELS
        }
        for path in candidates:
            raw_plan = _read_jsonl(path / "plan.jsonl")
            model_id = _model_from_plan(raw_plan)
            if model_id in seen:
                raise RevisionComputeError("Duplicate smoke shard for {}".format(model_id))
            seen.add(model_id)
            verified.append(
                verify_smoke_shard(
                    path,
                    smoke_by_model[model_id],
                    str(config_verification["sha256"]),
                )
            )
        if seen != set(EXPECTED_MODELS):
            raise RevisionComputeError("Smoke shards do not cover all three models")
        verified.sort(key=lambda row: EXPECTED_MODELS.index(str(row["model_id"])))
        projected_unavailability, unavailability_summary = (
            _derive_projected_unavailability(plans, verified)
        )
        rates = _smoke_rates(verified)
        auxiliary = load_auxiliary_timings(auxiliary_timing_paths)
        if legacy_incurred_ledger_path is None:
            raise RevisionComputeError(
                "A verified legacy incurred-only charge ledger is required"
            )
        legacy_incurred = verify_legacy_incurred_charge_ledger(
            Path(legacy_incurred_ledger_path)
        )
        invalidations = _verified_invalidation_charges(
            invalidation_manifest_paths
        )
        incurred_charge_audit = _validate_combined_incurred_charges(
            legacy_incurred, invalidations
        )

        projections: List[Dict[str, object]] = []
        for entry in legacy_incurred["entries"]:
            seconds = float(entry["incurred_gpu_seconds"])
            projections.append(
                {
                    "stage": "legacy_incurred_observed",
                    "model_id": entry["model_id"],
                    "component": entry["component"],
                    "resource_class": "gpu",
                    "target_work_units": 1,
                    "point_seconds": seconds,
                    "upper_seconds": seconds,
                    "point_gpu_hours": seconds / 3600.0,
                    "upper_gpu_hours": seconds / 3600.0,
                    "timing_source": entry["charge_policy"],
                    "scientific_use": entry["scientific_use"],
                }
            )
        for entry in invalidations:
            seconds = float(entry["incurred_gpu_seconds"])
            state = entry["execution_state"]
            projections.append(
                {
                    "stage": "invalidated_shard_observed",
                    "model_id": entry["run_identity_sha256"],
                    "resource_class": "gpu",
                    "target_work_units": int(state["completed_work_units"]),
                    "point_seconds": seconds,
                    "upper_seconds": seconds,
                    "point_gpu_hours": seconds / 3600.0,
                    "upper_gpu_hours": seconds / 3600.0,
                    "timing_source": entry["charge_policy"],
                    "scientific_use": "invalidated_not_for_pooling_incurred_charge_only",
                    "invalidation_manifest_sha256": entry["invalidation_manifest_sha256"],
                }
            )
        for timing in auxiliary:
            if int(timing["gpu_count"]) <= 0:
                continue
            seconds = float(timing["elapsed_seconds"]) * int(timing["gpu_count"])
            if timing["component"] == "evaluator":
                seconds += float(timing["model_load_seconds"])
            projections.append(
                {
                    "stage": "auxiliary_smoke_v3_observed",
                    "model_id": timing.get("model_id") or timing["component_id"],
                    "component": timing["component"],
                    "resource_class": "gpu",
                    "target_work_units": int(timing["completed_units"]),
                    "point_seconds": seconds,
                    "upper_seconds": seconds,
                    "point_gpu_hours": seconds / 3600.0,
                    "upper_gpu_hours": seconds / 3600.0,
                    "timing_source": "observed_smoke_v3_auxiliary_incurred_charge",
                    "scientific_use": "incurred_charge_and_v3_rate_source",
                }
            )
        # Already-incurred smoke GPU time is included so the ceiling applies to
        # the complete computational study, not only the work remaining.
        for shard in verified:
            seconds = float(shard["actual_gpu_seconds"])
            projections.append(
                {
                    "stage": "smoke_observed",
                    "model_id": shard["model_id"],
                    "resource_class": "gpu",
                    "target_work_units": shard["work_units"],
                    "projected_available_work_units": shard["available_work_units"],
                    "projected_unavailable_work_units": shard["unavailable_work_units"],
                    "point_seconds": seconds,
                    "upper_seconds": seconds,
                    "point_gpu_hours": seconds / 3600.0,
                    "upper_gpu_hours": seconds / 3600.0,
                    "timing_source": "observed_completed_smoke_including_all_replays_and_loads",
                }
            )
        for stage in PROJECTED_RUNNER_STAGES:
            for model_id in EXPECTED_MODELS:
                projections.append(
                    _runner_stage_projection(
                        stage,
                        model_id,
                        plans[stage],
                        rates[model_id],
                        projected_unavailability[stage],
                    )
                )
        projections.extend(_evaluator_projections(rates, auxiliary))
        projections.extend(_detector_projections(configs, auxiliary, Path(config_dir)))
        point_total = sum(float(row["point_gpu_hours"]) for row in projections)
        upper_total = sum(float(row["upper_gpu_hours"]) for row in projections)
        by_stage = []
        for stage in sorted({str(row["stage"]) for row in projections}):
            selected = [row for row in projections if row["stage"] == stage]
            by_stage.append(
                {
                    "stage": stage,
                    "target_work_units": sum(int(row["target_work_units"]) for row in selected),
                    "projected_available_work_units": sum(
                        int(row.get("projected_available_work_units", row["target_work_units"]))
                        for row in selected
                    ),
                    "projected_unavailable_work_units": sum(
                        int(row.get("projected_unavailable_work_units", 0))
                        for row in selected
                    ),
                    "point_gpu_hours": sum(float(row["point_gpu_hours"]) for row in selected),
                    "upper_gpu_hours": sum(float(row["upper_gpu_hours"]) for row in selected),
                }
            )
        go = upper_total <= budget
        report.update(
            {
                "input_status": "complete",
                "incomplete_reasons": [],
                "config_manifest_sha256": config_verification["sha256"],
                "frozen_plan": {
                    stage: {
                        "work_units": len(plan),
                        "plan_sha256": canonical_json_sha256(plan),
                        "work_kind_counts": dict(
                            sorted(Counter(str(row["work_kind"]) for row in plan).items())
                        ),
                        "model_counts": dict(
                            sorted(Counter(str(row["model_id"]) for row in plan).items())
                        ),
                    }
                    for stage, plan in plans.items()
                },
                "verified_smoke_shards": [
                    {key: value for key, value in row.items() if key != "observations"}
                    for row in verified
                ],
                "unavailability_propagation": unavailability_summary,
                "auxiliary_timing_inputs": auxiliary,
                "legacy_incurred_charge_ledger": legacy_incurred,
                "verified_invalidation_manifests": invalidations,
                "combined_incurred_charge_audit": incurred_charge_audit,
                "projection_rows": projections,
                "stage_totals": by_stage,
                "totals": {
                    "point_gpu_hours": point_total,
                    "upper_gpu_hours": upper_total,
                    "budget_gpu_hours": budget,
                    "upper_headroom_gpu_hours": budget - upper_total,
                },
                "decision": {
                    "go": go,
                    "status": "go_within_budget" if go else "no_go_over_budget",
                    "reason": (
                        "All required smoke evidence is complete and the conservative upper "
                        "projection is within the approved ceiling."
                        if go
                        else "The conservative upper projection exceeds the approved ceiling."
                    ),
                },
            }
        )
    except Exception as exc:
        errors.append("{}: {}".format(type(exc).__name__, exc))
        report.update(
            {
                "input_status": "incomplete_or_invalid",
                "incomplete_reasons": errors,
                "projection_rows": [],
                "stage_totals": [],
                "totals": None,
                "decision": {
                    "go": False,
                    "status": "no_go_incomplete_inputs",
                    "reason": "A go decision is forbidden until every required input verifies.",
                },
            }
        )
    report["projection_sha256"] = canonical_json_sha256(report)
    return report


__all__ = [
    "AUXILIARY_TIMING_SCHEMA_VERSION",
    "COMPUTE_SCHEMA_VERSION",
    "DEFAULT_BUDGET_GPU_HOURS",
    "DETECTOR_FITS",
    "EVALUATOR_UNITS_PER_MODEL",
    "EXPECTED_MODELS",
    "EXPECTED_PLAN_COUNTS",
    "EXPECTED_SMOKE_STAGE",
    "EXPECTED_SMOKE_UNAVAILABLE",
    "EXPECTED_UNAVAILABLE_PROPAGATION",
    "RevisionComputeError",
    "build_auxiliary_timing_record",
    "build_legacy_incurred_charge_ledger",
    "discover_smoke_shards",
    "load_auxiliary_timings",
    "load_frozen_plans",
    "project_revision_compute",
    "verify_legacy_gpu_ledger",
    "verify_legacy_incurred_charge_ledger",
    "verify_smoke_shard",
]
