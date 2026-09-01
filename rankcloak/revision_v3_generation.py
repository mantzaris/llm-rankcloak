"""Resumable model-backed execution for the revision-V3 generation studies.

The executable CSV ledgers are authoritative. Every completed row is written
atomically to an immutable per-trial JSON record and is validated against its
plan row, source records, configuration hashes, model artifact, and protocol
amendment before it is accepted on resume.
"""

from __future__ import annotations

import argparse
import base64
import csv
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .model_io import (
    get_bos_token_id,
    get_vocab_size,
    llama_cpp_gpu_offload_supported,
    load_llama_cpp_model,
    make_context_token_ids,
)
from .revision_payloads import generate_revision_v1_payloads, validate_revision_corpus
from .revision_protocol import (
    Representation,
    decode_representation,
    first_divergence,
    generate_rank_span,
    recover_rank_span,
    retokenize_message,
)
from .revision_runner import generate_length_matched_control
from .revision_v3_diagnostics import trace_observed_tokens
from .revision_v3_entropy import (
    calibrate_entropy_gate_thresholds,
    generate_entropy_gated_span,
    generate_ordinary_entropy_trace,
    recover_entropy_gated_span,
    retokenize_entropy_gated_message,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results/revision_v3/generation"
PROVENANCE_ROOT = PROJECT_ROOT / "results/revision_v3/provenance"
ENTROPY_PLAN = PROVENANCE_ROOT / "entropy_generation_plan.csv"
CALIBRATION_PLAN = PROVENANCE_ROOT / "entropy_calibration_plan.csv"
QUANTIZATION_PLAN = PROVENANCE_ROOT / "quantization_generation_plan.csv"
REQUIREMENTS_CONFIG = PROJECT_ROOT / "configs/revision_v3/generation_requirements.json"
ENTROPY_CONFIG = PROJECT_ROOT / "configs/revision_v3/entropy_gate.json"
QUANTIZATION_CONFIG = PROJECT_ROOT / "configs/revision_v3/quantization.json"
PROMPT_CONFIG = PROJECT_ROOT / "configs/revision_v1/prompts.json"
PROTOCOL_AMENDMENT = PROJECT_ROOT / "revision_docs/REVISION_V3_GENERATION_PROTOCOL_AMENDMENT.md"
QWEN_Q4_ROOT = PROJECT_ROOT / "results/revision_v1/primary_v2/qwen2_5_7b_instruct_q4_k_m"
QWEN_Q4_RECORDS = QWEN_Q4_ROOT / "records.jsonl"
QWEN_Q4_PLAN = QWEN_Q4_ROOT / "plan.jsonl"
SCHEMA_VERSION = "rankcloak-revision-v3-model-generation-v1"
CONTEXT_LIMIT = 4096
DEFAULT_GPU_UUID = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"


class GenerationExecutionError(RuntimeError):
    """Raised when execution would violate a frozen generation contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


@functools.lru_cache(maxsize=None)
def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def prompt_index() -> Mapping[str, Mapping[str, str]]:
    document = load_json(PROMPT_CONFIG)
    return {
        str(template["prompt_id"]): {
            "category": str(category["category_id"]),
            "text": str(template["text"]),
        }
        for category in document["categories"]
        for template in category["templates"]
    }


@functools.lru_cache(maxsize=None)
def source_hashes(plan_path: Path) -> Mapping[str, str]:
    paths = (
        plan_path,
        REQUIREMENTS_CONFIG,
        ENTROPY_CONFIG,
        QUANTIZATION_CONFIG,
        PROMPT_CONFIG,
        PROTOCOL_AMENDMENT,
        Path(__file__),
        PROJECT_ROOT / "rankcloak/revision_v3_entropy.py",
        PROJECT_ROOT / "rankcloak/revision_v3_diagnostics.py",
        PROJECT_ROOT / "rankcloak/revision_runner.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in paths}


class GpuMemoryMonitor:
    """Poll physical GPU memory and expose per-operation maxima."""

    def __init__(self, gpu_uuid: str, interval_seconds: float = 0.5):
        self.gpu_uuid = str(gpu_uuid)
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._maximum_mib: Optional[int] = None
        self._latest_mib: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._sample()

    def reset_peak(self) -> Optional[int]:
        self._sample()
        with self._lock:
            self._maximum_mib = self._latest_mib
            return self._latest_mib

    def snapshot(self) -> Mapping[str, Optional[int]]:
        self._sample()
        with self._lock:
            return {
                "gpu_memory_latest_mib": self._latest_mib,
                "gpu_memory_peak_mib": self._maximum_mib,
            }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            observed = None
            for line in completed.stdout.splitlines():
                uuid, memory = [part.strip() for part in line.split(",", 1)]
                if uuid == self.gpu_uuid:
                    observed = int(memory)
                    break
            with self._lock:
                self._latest_mib = observed
                if observed is not None and (
                    self._maximum_mib is None or observed > self._maximum_mib
                ):
                    self._maximum_mib = observed
        except Exception:
            return


def gpu_inventory(gpu_uuid: str) -> Mapping[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, name, uuid, driver, memory = [part.strip() for part in line.split(",", 4)]
        rows.append(
            {
                "nvidia_smi_index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total_mib": int(memory),
            }
        )
    selected = next((row for row in rows if row["uuid"] == gpu_uuid), None)
    if selected is None:
        raise GenerationExecutionError("requested GPU UUID is absent")
    if selected["name"] != "NVIDIA RTX 5000 Ada Generation":
        raise GenerationExecutionError("requested GPU is not the authorized RTX 5000 Ada")
    return {"selected": selected, "all_devices": rows}


def configure_deterministic_gpu(gpu_uuid: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    os.environ["GGML_CUDA_DISABLE_FUSION"] = "1"
    os.environ["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def requirement_artifacts() -> Mapping[str, Mapping[str, object]]:
    requirements = load_json(REQUIREMENTS_CONFIG)
    return {str(row["model_id"]): dict(row) for row in requirements["artifacts"]}


def verify_model_artifact(model_id: str) -> Mapping[str, object]:
    artifacts = requirement_artifacts()
    if model_id not in artifacts:
        raise GenerationExecutionError("model is absent from pinned requirements")
    expected = artifacts[model_id]
    path = PROJECT_ROOT / str(expected["expected_path"])
    if not path.is_file():
        raise GenerationExecutionError("pinned model file is unavailable: {}".format(path))
    size = int(path.stat().st_size)
    if size != int(expected["size_bytes"]):
        raise GenerationExecutionError("pinned model byte count mismatch")
    digest = file_sha256(path)
    if digest != str(expected["sha256"]):
        raise GenerationExecutionError("pinned model SHA-256 mismatch")
    return {
        "model_id": model_id,
        "path": str(path),
        "repo_id": expected["repo_id"],
        "revision": expected["revision"],
        "filename": expected["filename"],
        "quantization": expected["quantization"],
        "size_bytes": size,
        "sha256": digest,
        "verified_at": utc_now(),
    }


def backend_manifest(gpu_uuid: str) -> Mapping[str, object]:
    from llama_cpp import llama_cpp as llama_cpp_api

    package_version = importlib.metadata.version("llama-cpp-python")
    if package_version != "0.3.23":
        raise GenerationExecutionError("llama-cpp-python version is not 0.3.23")
    system_info = llama_cpp_api.llama_print_system_info().decode("utf-8", "replace")
    if "CUDA" not in system_info.upper() or not llama_cpp_gpu_offload_supported():
        raise GenerationExecutionError("llama.cpp backend does not support CUDA offload")
    packages = {}
    for name in (
        "llama-cpp-python",
        "numpy",
        "pandas",
        "cryptography",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "llama_cpp_system_info": system_info,
        "gpu_offload_supported": True,
        "deterministic_environment": {
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
        },
        "gpu_inventory": gpu_inventory(gpu_uuid),
    }


def load_verified_model(
    model_id: str, gpu_uuid: str, monitor: GpuMemoryMonitor
) -> Tuple[Any, Mapping[str, object]]:
    configure_deterministic_gpu(gpu_uuid)
    verification = verify_model_artifact(model_id)
    backend = backend_manifest(gpu_uuid)
    load_started = time.perf_counter()
    monitor.reset_peak()
    model = load_llama_cpp_model(
        Path(str(verification["path"])),
        n_ctx=CONTEXT_LIMIT,
        n_threads=max(1, (os.cpu_count() or 2) - 1),
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_started
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    manifest = {
        "artifact": verification,
        "backend": backend,
        "model_load_seconds": float(load_seconds),
        "model_load_memory": monitor.snapshot(),
        "n_ctx": CONTEXT_LIMIT,
        "n_threads": max(1, (os.cpu_count() or 2) - 1),
        "n_gpu_layers": -1,
        "n_batch": 1,
        "n_ubatch": 1,
        "logits_all": True,
        "vocabulary_size": get_vocab_size(model),
        "bos_token_id": get_bos_token_id(model),
        "gguf_general_name": metadata.get("general.name"),
        "loaded_at": utc_now(),
    }
    return model, manifest


def historical_primary_records(model_id: str) -> Mapping[str, Mapping[str, object]]:
    path = PROJECT_ROOT / "results/revision_v1/primary_v2" / model_id / "records.jsonl"
    records = load_jsonl(path)
    return {
        str(row["trial_id"]): row
        for row in records
        if row.get("record_type") == "rankcloak_trial"
    }


def qwen_historical_indexes() -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    records = load_jsonl(QWEN_Q4_RECORDS)
    tasks = load_jsonl(QWEN_Q4_PLAN)
    rank = {
        str(row["trial_id"]): row
        for row in records
        if row.get("record_type") == "rankcloak_trial"
    }
    controls = {
        str(row["source_trial_id"]): row
        for row in records
        if row.get("record_type") == "ordinary_control"
        and row.get("control_view") == "full_message"
    }
    task = {
        str(row["trial_id"]): row
        for row in tasks
        if row.get("work_kind") == "rankcloak"
    }
    if not (rank.keys() == controls.keys() == task.keys()):
        raise GenerationExecutionError("historical Q4 rank/control/task indexes differ")
    return {"rank": rank, "control": controls, "task": task}


def payload_index() -> Mapping[str, object]:
    payloads = generate_revision_v1_payloads()
    validation = validate_revision_corpus(payloads)
    if validation["status"] != "ok":
        raise GenerationExecutionError("authoritative payload corpus validation failed")
    return {str(payload.payload_name): payload for payload in payloads}


def representation_from_source(
    source: Mapping[str, object], payloads: Mapping[str, object]
) -> Representation:
    payload_name = str(source["payload_name"])
    if payload_name not in payloads:
        raise GenerationExecutionError("payload is absent from authoritative corpus")
    payload = payloads[payload_name]
    document = source["representation"]
    return Representation(
        name=str(document["name"]),
        ranks=tuple(map(int, document["expected_ranks"])),
        metadata=dict(document["metadata"]),
        payload_bytes=bytes(payload.payload_bytes),
        payload_text=str(payload.payload_text),
    )


def model_record_path(
    result_root: Path,
    phase: str,
    model_id: str,
    plan_id: str,
    smoke: bool,
) -> Path:
    namespace = "smoke" if smoke else "raw"
    return result_root / namespace / phase / model_id / (str(plan_id) + ".json")


def failure_record_path(
    result_root: Path, phase: str, model_id: str, plan_id: str, smoke: bool
) -> Path:
    namespace = "smoke_failures" if smoke else "failures"
    return result_root / namespace / phase / model_id / (str(plan_id) + ".json")


def validated_existing(path: Path, plan_row_sha256: str, model_sha256: str) -> bool:
    if not path.is_file():
        return False
    record = load_json(path)
    if not (
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("execution_status") == "completed"
        and record.get("plan_row_sha256") == plan_row_sha256
        and record.get("model_artifact_sha256") == model_sha256
    ):
        raise GenerationExecutionError(
            "existing result does not match the active plan/model: {}".format(path)
        )
    return True


def immutable_result(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise GenerationExecutionError("refusing to overwrite completed result")
    atomic_json(path, value)


def operation_provenance(
    plan_path: Path,
    plan_row: Mapping[str, object],
    model_manifest: Mapping[str, object],
    started_at: str,
    started_perf: float,
    rss_start: int,
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_status": "completed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "execution_seconds": float(time.perf_counter() - started_perf),
        "process_peak_rss_start_bytes": int(rss_start),
        "process_peak_rss_end_bytes": int(process_peak_rss_bytes()),
        "gpu_memory": monitor.snapshot(),
        "plan_id": str(plan_row["plan_id"]),
        "plan_row": dict(plan_row),
        "plan_row_sha256": canonical_sha256(plan_row),
        "plan_file": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_file_sha256": file_sha256(plan_path),
        "source_hashes": source_hashes(plan_path),
        "protocol_amendment_commit": git_output(
            "log", "-1", "--format=%H", "--", str(PROTOCOL_AMENDMENT.relative_to(PROJECT_ROOT))
        ),
        "execution_git_commit": git_output("rev-parse", "HEAD"),
        "model_artifact_sha256": model_manifest["artifact"]["sha256"],
        "model_manifest": model_manifest,
    }


def execute_calibration_row(
    model: Any,
    row: Mapping[str, str],
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    prompts = prompt_index()
    prompt_id = str(row["prompt_template_id"])
    if prompt_id not in prompts:
        raise GenerationExecutionError("unknown calibration prompt")
    prompt_text = str(prompts[prompt_id]["text"])
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    if prompt_sha != str(row["rendered_prompt_sha256"]):
        raise GenerationExecutionError("calibration rendered-prompt hash mismatch")
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    context = make_context_token_ids(model, prompt_text)
    target = int(row["target_token_count"])
    if len(context) + target > CONTEXT_LIMIT:
        raise GenerationExecutionError("calibration trace exceeds context limit")
    trace = generate_ordinary_entropy_trace(
        model,
        context,
        target,
        sampling_seed=int(row["random_seed"]),
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
    )
    if not (
        len(trace["token_ids"])
        == len(trace["next_token_entropies_bits"])
        == len(trace["sampled_token_ranks"])
        == target
    ):
        raise GenerationExecutionError("calibration trace length mismatch")
    if not np.isfinite(
        np.asarray(trace["next_token_entropies_bits"], dtype=float)
    ).all():
        raise GenerationExecutionError("calibration trace has non-finite entropy")
    record = dict(
        operation_provenance(
            CALIBRATION_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record.update(
        {
            "record_type": "entropy_calibration_trace",
            "model_id": row["model_id"],
            "prompt_template_id": prompt_id,
            "prompt_category": prompts[prompt_id]["category"],
            "rendered_prompt": prompt_text,
            "rendered_prompt_sha256": prompt_sha,
            "generation": trace,
            "validation": {
                "target_token_count_exact": True,
                "finite_entropy_at_every_position": True,
                "detector_outcomes_used": False,
            },
        }
    )
    return record


def freeze_model_thresholds(result_root: Path, model_id: str) -> Mapping[str, object]:
    rows = [
        row
        for row in load_csv(CALIBRATION_PLAN)
        if str(row["model_id"]) == str(model_id)
    ]
    if len(rows) != 6:
        raise GenerationExecutionError("model calibration plan does not contain six traces")
    records = []
    entropies: List[float] = []
    for row in rows:
        path = model_record_path(
            result_root, "entropy_calibration", model_id, row["plan_id"], False
        )
        if not path.is_file():
            raise GenerationExecutionError("cannot freeze incomplete calibration")
        record = load_json(path)
        if record.get("plan_row_sha256") != canonical_sha256(row):
            raise GenerationExecutionError("calibration result plan hash mismatch")
        values = list(map(float, record["generation"]["next_token_entropies_bits"]))
        if len(values) != int(row["target_token_count"]):
            raise GenerationExecutionError("calibration result length mismatch")
        entropies.extend(values)
        records.append(
            {
                "plan_id": row["plan_id"],
                "result_path": str(path.relative_to(PROJECT_ROOT)),
                "result_sha256": file_sha256(path),
                "position_count": len(values),
            }
        )
    thresholds = dict(calibrate_entropy_gate_thresholds(entropies))
    thresholds.update(
        {
            "model_id": model_id,
            "trace_count": len(records),
            "trace_records": records,
            "calibration_plan_sha256": file_sha256(CALIBRATION_PLAN),
            "entropy_config_sha256": file_sha256(ENTROPY_CONFIG),
            "protocol_amendment_sha256": file_sha256(PROTOCOL_AMENDMENT),
            "frozen_at": utc_now(),
        }
    )
    output = result_root / "calibration/thresholds" / (model_id + ".json")
    if output.exists():
        existing = load_json(output)
        comparable = dict(existing)
        comparable.pop("frozen_at", None)
        candidate = dict(thresholds)
        candidate.pop("frozen_at", None)
        if canonical_sha256(comparable) != canonical_sha256(candidate):
            raise GenerationExecutionError("frozen threshold file would change")
        return existing
    atomic_json(output, thresholds)
    return thresholds


def load_model_thresholds(result_root: Path, model_id: str) -> Mapping[str, object]:
    path = result_root / "calibration/thresholds" / (model_id + ".json")
    if not path.is_file():
        raise GenerationExecutionError("entropy thresholds are not frozen for model")
    thresholds = load_json(path)
    if not (
        thresholds.get("model_id") == model_id
        and thresholds.get("detector_outcomes_used") is False
        and int(thresholds.get("development_position_count", 0)) == 768
    ):
        raise GenerationExecutionError("frozen entropy threshold contract is invalid")
    return thresholds


def entropy_threshold_for_level(
    thresholds: Mapping[str, object], gate_level: str
) -> Optional[float]:
    if gate_level == "ungated":
        return None
    if gate_level == "moderate":
        return float(thresholds["moderate_threshold_bits"])
    if gate_level == "strict":
        return float(thresholds["strict_threshold_bits"])
    raise GenerationExecutionError("unknown entropy gate level")


def entropy_rankcloak_record(
    model: Any,
    row: Mapping[str, str],
    source: Mapping[str, object],
    representation: Representation,
    thresholds: Mapping[str, object],
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    if not (
        str(row["population"]) == "rankcloak"
        and str(source["model_id"]) == str(row["model_id"])
        and str(source["payload_name"]) == str(row["payload_name"])
        and str(source["representation"]["name"])
        == str(row["representation_name"])
        and str(source["token_filter"]) == "none"
        and str(source["tail_policy"]) == "none"
        and int(source["leadin_token_count"]) == 0
        and int(source["segment_count"]) == 1
    ):
        raise GenerationExecutionError("entropy source protocol does not match plan")
    prompts = prompt_index()
    prompt_id = str(row["prompt_template_id"])
    prompt_text = str(prompts[prompt_id]["text"])
    context = make_context_token_ids(model, prompt_text)
    ranks = list(map(int, representation.ranks))
    gate_level = str(row["gate_level"])
    threshold = entropy_threshold_for_level(thresholds, gate_level)
    maximum = len(ranks)
    if threshold is not None:
        maximum = min(
            CONTEXT_LIMIT - len(context),
            int(load_json(ENTROPY_CONFIG)["fixed_payload_maximum_length_multiplier"])
            * len(ranks),
        )
    if maximum < 0 or len(context) + maximum > CONTEXT_LIMIT:
        raise GenerationExecutionError("entropy trial exceeds context allowance")
    quality_rank_ceiling = (
        16 if representation.name in {"ascii_b16", "hex_nibble"} else None
    )
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    generated = generate_entropy_gated_span(
        model,
        context,
        ranks,
        entropy_threshold_bits=threshold,
        maximum_generated_tokens=maximum,
        allowed_token_mask=None,
        leadin_token_count=0,
        tail_policy="none",
        quality_rank_ceiling=quality_rank_ceiling,
        sampling_seed=(
            int(row["random_seed"]) if threshold is not None else None
        ),
        temperature=float(row["ordinary_sampling_temperature"]),
        top_p=float(row["ordinary_sampling_top_p"]),
    )
    saved = recover_entropy_gated_span(
        model,
        context,
        generated["leadin_token_ids"],
        generated["embedding_token_ids"],
        entropy_threshold_bits=threshold,
        expected_payload_rank_count=len(ranks),
        allowed_token_mask=None,
    )
    consumed = int(generated["consumed_payload_rank_count"])
    expected_prefix = ranks[:consumed]
    if list(map(int, saved["ranks"])) != expected_prefix:
        raise GenerationExecutionError("saved-ID entropy replay rank mismatch")
    if list(map(bool, saved["embedding_eligible_mask"])) != list(
        map(bool, generated["embedding_eligible_mask"])
    ):
        raise GenerationExecutionError("encoder/decoder entropy eligibility mismatch")
    if list(map(str, saved["embedding_token_roles"])) != list(
        map(str, generated["embedding_token_roles"])
    ):
        raise GenerationExecutionError("encoder/decoder token-role mismatch")
    completed = bool(generated["payload_completion"])
    decoded = decode_representation(model, representation, saved["ranks"])
    if completed and not bool(decoded["exact_payload_recovery"]):
        raise GenerationExecutionError("completed saved-ID replay did not recover payload")

    visible_diagnostic = retokenize_entropy_gated_message(model, dict(generated))
    visible_error = None
    try:
        visible_saved = recover_entropy_gated_span(
            model,
            context,
            generated["leadin_token_ids"],
            visible_diagnostic["embedding_token_ids"],
            entropy_threshold_bits=threshold,
            expected_payload_rank_count=len(ranks),
            allowed_token_mask=None,
        )
        visible_decoded = decode_representation(
            model, representation, visible_saved["ranks"]
        )
    except Exception as exc:
        visible_saved = None
        visible_decoded = None
        visible_error = "{}: {}".format(type(exc).__name__, exc)

    fixed_budget = len(ranks)
    roles_in_budget = list(map(str, generated["embedding_token_roles"][:fixed_budget]))
    ranks_in_budget = sum(role == "payload" for role in roles_in_budget)
    generated_count = len(generated["embedding_token_ids"])
    serialized_bits = len(representation.payload_bytes) * 8
    record = dict(
        operation_provenance(
            ENTROPY_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record.update(
        {
            "record_type": "entropy_rankcloak_trial",
            "model_id": row["model_id"],
            "payload_name": row["payload_name"],
            "payload_class": row["payload_class"],
            "representation_name": representation.name,
            "expected_ranks": ranks,
            "expected_ranks_sha256": canonical_sha256(ranks),
            "rendered_prompt": prompt_text,
            "rendered_prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
            "thresholds": dict(thresholds),
            "threshold_bits": threshold,
            "generation": generated,
            "saved_token_id_replay": {
                "replay": saved,
                "decoded": decoded,
                "exact_rank_prefix_recovery": list(map(int, saved["ranks"]))
                == expected_prefix,
                "exact_payload_recovery": bool(decoded["exact_payload_recovery"]),
            },
            "visible_text_retokenization": {
                "diagnostic": visible_diagnostic,
                "replay": visible_saved,
                "decoded": visible_decoded,
                "exact_payload_recovery": bool(
                    visible_decoded
                    and visible_decoded.get("exact_payload_recovery")
                ),
                "error": visible_error,
            },
            "fixed_payload": {
                "maximum_embedding_token_count": maximum,
                "payload_completion": completed,
                "embedding_token_count": generated_count,
                "eligible_position_count": int(generated["eligible_position_count"]),
                "eligible_position_fraction": (
                    float(generated["eligible_position_count"] / generated_count)
                    if generated_count
                    else None
                ),
                "serialized_payload_bits": serialized_bits,
                "bits_per_generated_token": (
                    float(serialized_bits / generated_count)
                    if completed and generated_count
                    else None
                ),
            },
            "fixed_token_budget": {
                "generated_token_budget": fixed_budget,
                "observed_embedding_positions": min(fixed_budget, generated_count),
                "payload_ranks_embedded": int(ranks_in_budget),
                "payload_rank_count": len(ranks),
                "payload_fraction_embedded": (
                    float(ranks_in_budget / len(ranks)) if ranks else 1.0
                ),
                "serialized_bits_embedded": (
                    float(serialized_bits * ranks_in_budget / len(ranks))
                    if ranks
                    else float(serialized_bits)
                ),
            },
            "source_lineage": {
                "reference_trial_id": source["trial_id"],
                "reference_trial_sha256": canonical_sha256(source),
                "source_payload_text_sha256": source["payload_text_sha256"],
                "source_token_filter": source["token_filter"],
            },
            "validation": {
                "encoder_decoder_gate_positions_exact": True,
                "saved_rank_prefix_exact": True,
                "all_embedding_entropies_finite": bool(
                    np.isfinite(
                        np.asarray(generated["embedding_entropies_bits"], dtype=float)
                    ).all()
                ),
                "all_position_arrays_same_length": len(
                    generated["embedding_token_ids"]
                )
                == len(generated["embedding_entropies_bits"])
                == len(generated["embedding_eligible_mask"])
                == len(generated["embedding_token_roles"])
                == len(generated["embedding_observed_ranks"]),
            },
        }
    )
    if not all(record["validation"].values()):
        raise GenerationExecutionError("entropy RankCloak validation failed")
    return record


def entropy_control_record(
    model: Any,
    row: Mapping[str, str],
    paired_rankcloak: Mapping[str, object],
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    if str(row["population"]) != "ordinary_control":
        raise GenerationExecutionError("entropy control row has wrong population")
    if str(paired_rankcloak["plan_row"]["pairing_unit_id"]) != str(
        row["pairing_unit_id"]
    ):
        raise GenerationExecutionError("entropy control pairing mismatch")
    prompts = prompt_index()
    prompt_text = str(prompts[str(row["prompt_template_id"])]["text"])
    target = len(paired_rankcloak["generation"]["full_token_ids"])
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    generated = generate_length_matched_control(
        model,
        prompt_text,
        target,
        int(row["random_seed"]),
        temperature=float(row["ordinary_sampling_temperature"]),
        top_p=float(row["ordinary_sampling_top_p"]),
        context_limit=CONTEXT_LIMIT,
    )
    if not (
        len(generated["token_ids"])
        == len(generated["next_token_entropies_bits"])
        == len(generated["sampled_token_ranks"])
        == target
    ):
        raise GenerationExecutionError("entropy control target length mismatch")
    record = dict(
        operation_provenance(
            ENTROPY_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record.update(
        {
            "record_type": "entropy_ordinary_control",
            "model_id": row["model_id"],
            "payload_name": row["payload_name"],
            "payload_class": row["payload_class"],
            "representation_name": row["representation_name"],
            "rendered_prompt": prompt_text,
            "rendered_prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
            "generation": generated,
            "paired_rankcloak_plan_id": paired_rankcloak["plan_id"],
            "paired_rankcloak_result_sha256": canonical_sha256(paired_rankcloak),
            "paired_gate_level": row["gate_level"],
            "validation": {
                "target_length_matches_paired_rankcloak": True,
                "sampling_seed_is_predeclared": int(generated["sampling_seed"])
                == int(row["paired_ordinary_control_seed"]),
                "sampler_is_predeclared": generated["sampler"]
                == row["ordinary_sampler"],
                "all_entropies_finite": bool(
                    np.isfinite(
                        np.asarray(generated["next_token_entropies_bits"], dtype=float)
                    ).all()
                ),
            },
        }
    )
    if not all(record["validation"].values()):
        raise GenerationExecutionError("entropy ordinary-control validation failed")
    return record


def quantization_source_bundle(
    row: Mapping[str, str],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    trial_id = str(row["reference_q4_trial_id"])
    try:
        rank = indexes["rank"][trial_id]
        control = indexes["control"][trial_id]
        task = indexes["task"][trial_id]
    except KeyError as exc:
        raise GenerationExecutionError("historical Q4 source bundle is incomplete") from exc
    if not (
        canonical_sha256(rank) == row["historical_rank_record_sha256"]
        and canonical_sha256(control) == row["historical_control_record_sha256"]
        and canonical_sha256(task) == row["historical_task_sha256"]
    ):
        raise GenerationExecutionError("historical Q4 source hash mismatch")
    generation = control["generation"]
    if not (
        int(generation["sampling_seed"])
        == int(row["historical_control_sampling_seed"])
        == int(row["random_seed"])
        and float(generation["temperature"]) == float(row["temperature"])
        and float(generation["top_p"]) == float(row["top_p"])
        and generation["sampler"] == row["sampler"]
        and int(generation["target_token_count"])
        == int(row["target_token_count"])
    ):
        raise GenerationExecutionError("historical Q4 sampling contract mismatch")
    return rank, control, task


def historical_output_ids(
    population: str,
    rank: Mapping[str, object],
    control: Mapping[str, object],
) -> List[int]:
    if population == "rankcloak":
        if int(rank["segment_count"]) != 1:
            raise GenerationExecutionError("quantization source is unexpectedly segmented")
        return list(map(int, rank["segments"][0]["full_token_ids"]))
    if population == "ordinary_control":
        return list(map(int, control["generation"]["token_ids"]))
    raise GenerationExecutionError("unknown quantization population")


def validate_quantization_context(
    model: Any,
    row: Mapping[str, str],
    rank: Mapping[str, object],
    control: Mapping[str, object],
) -> Tuple[str, List[int]]:
    prompts = prompt_index()
    prompt_id = str(row["prompt_template_id"])
    prompt_text = str(prompts[prompt_id]["text"])
    if hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() != row[
        "rendered_prompt_sha256"
    ]:
        raise GenerationExecutionError("quantization prompt hash mismatch")
    context = make_context_token_ids(model, prompt_text)
    historical_context = list(map(int, control["generation"]["context_token_ids"]))
    rank_context = list(map(int, rank["segments"][0]["context_token_ids"]))
    if context != historical_context or context != rank_context:
        raise GenerationExecutionError("embedded tokenizer or rendered context diverged")
    if canonical_sha256(context) != row[
        "historical_prompt_context_token_ids_sha256"
    ]:
        raise GenerationExecutionError("quantization context hash mismatch")
    return prompt_text, context


def trace_summary(trace: Mapping[str, object]) -> Mapping[str, object]:
    def mean(values: Sequence[object]) -> Optional[float]:
        return float(np.mean(np.asarray(values, dtype=float))) if values else None

    return {
        "position_count": int(trace["position_count"]),
        "mean_entropy_bits": mean(trace["entropy_bits"]),
        "mean_observed_rank": mean(trace["observed_ranks"]),
        "mean_observed_surprisal_nats": mean(trace["observed_surprisals_nats"]),
        "mean_rank_pressure_log_probability_gap_nats": mean(
            trace["rank_pressure_log_probability_gaps_nats"]
        ),
        "tail_rank_frequency_gt_100": (
            float(
                np.mean(np.asarray(trace["observed_ranks"], dtype=int) > 100)
            )
            if trace["observed_ranks"]
            else None
        ),
    }


def quantization_q4_replay_record(
    model: Any,
    row: Mapping[str, str],
    rank: Mapping[str, object],
    control: Mapping[str, object],
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    if str(row["quantization"]) != "Q4_K_M" or parse_bool(
        row["generation_required"]
    ):
        raise GenerationExecutionError("Q4 replay row has invalid status")
    prompt_text, context = validate_quantization_context(
        model, row, rank, control
    )
    population = str(row["population"])
    observed = historical_output_ids(population, rank, control)
    if len(observed) != int(row["target_token_count"]):
        raise GenerationExecutionError("historical Q4 output length mismatch")
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    trace = trace_observed_tokens(model, context, observed)
    expected_ranks = list(map(int, rank["representation"]["expected_ranks"]))
    rank_replay_exact = None
    if population == "rankcloak":
        rank_replay_exact = list(map(int, trace["observed_ranks"])) == expected_ranks
        if not rank_replay_exact:
            raise GenerationExecutionError(
                "historical Q4 ranks do not replay under the pinned backend"
            )
    record = dict(
        operation_provenance(
            QUANTIZATION_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record.update(
        {
            "record_type": "quantization_q4_model_backed_replay",
            "model_id": row["model_id"],
            "population": population,
            "new_generation_performed": False,
            "rendered_prompt": prompt_text,
            "context_token_ids": context,
            "historical_output_token_ids": observed,
            "historical_output_text": (
                rank["full_text"]
                if population == "rankcloak"
                else control["generation"]["text"]
            ),
            "distribution_trace": trace,
            "distribution_summary": trace_summary(trace),
            "expected_ranks": expected_ranks if population == "rankcloak" else None,
            "rank_replay_exact": rank_replay_exact,
            "source_lineage": {
                "rank_trial_id": rank["trial_id"],
                "rank_record_sha256": canonical_sha256(rank),
                "control_id": control["control_id"],
                "control_record_sha256": canonical_sha256(control),
            },
            "validation": {
                "embedded_tokenizer_context_exact": True,
                "historical_output_length_exact": True,
                "historical_control_seed_exact": int(
                    control["generation"]["sampling_seed"]
                )
                == int(row["historical_control_sampling_seed"]),
                "rank_replay_exact_when_applicable": rank_replay_exact is not False,
            },
        }
    )
    if not all(record["validation"].values()):
        raise GenerationExecutionError("Q4 replay validation failed")
    return record


def paired_distribution_comparison(
    q4_trace: Mapping[str, object], q8_trace: Mapping[str, object]
) -> Mapping[str, object]:
    q4_entropy = np.asarray(q4_trace["entropy_bits"], dtype=float)
    q8_entropy = np.asarray(q8_trace["entropy_bits"], dtype=float)
    q4_ranks = np.asarray(q4_trace["observed_ranks"], dtype=int)
    q8_ranks = np.asarray(q8_trace["observed_ranks"], dtype=int)
    q4_top = np.asarray(q4_trace["greedy_token_ids"], dtype=int)
    q8_top = np.asarray(q8_trace["greedy_token_ids"], dtype=int)
    if not (
        q4_entropy.shape == q8_entropy.shape == q4_ranks.shape == q8_ranks.shape
        and q4_top.shape == q8_top.shape == q4_ranks.shape
    ):
        raise GenerationExecutionError("paired quantization trace lengths differ")
    return {
        "position_count": int(q4_entropy.size),
        "mean_entropy_q8_minus_q4_bits": (
            float(np.mean(q8_entropy - q4_entropy)) if q4_entropy.size else None
        ),
        "median_entropy_q8_minus_q4_bits": (
            float(np.median(q8_entropy - q4_entropy)) if q4_entropy.size else None
        ),
        "observed_token_rank_changed_count": int(np.sum(q4_ranks != q8_ranks)),
        "observed_token_rank_changed_fraction": (
            float(np.mean(q4_ranks != q8_ranks)) if q4_ranks.size else None
        ),
        "greedy_token_changed_count": int(np.sum(q4_top != q8_top)),
        "greedy_token_changed_fraction": (
            float(np.mean(q4_top != q8_top)) if q4_top.size else None
        ),
        "mean_absolute_observed_rank_change": (
            float(np.mean(np.abs(q8_ranks - q4_ranks))) if q4_ranks.size else None
        ),
    }


def quantization_q8_generation_record(
    model: Any,
    row: Mapping[str, str],
    q4_row: Mapping[str, str],
    q4_replay: Mapping[str, object],
    rank: Mapping[str, object],
    control: Mapping[str, object],
    representation: Representation,
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
) -> Mapping[str, object]:
    if not (
        str(row["quantization"]) == "Q8_0"
        and parse_bool(row["generation_required"])
        and str(row["pairing_unit_id"]) == str(q4_row["pairing_unit_id"])
        and str(row["non_quantization_contract_sha256"])
        == str(q4_row["non_quantization_contract_sha256"])
        and int(row["historical_control_sampling_seed"])
        == int(q4_row["historical_control_sampling_seed"])
    ):
        raise GenerationExecutionError("Q4/Q8 pairing contract mismatch")
    prompt_text, context = validate_quantization_context(
        model, row, rank, control
    )
    population = str(row["population"])
    historical_ids = historical_output_ids(population, rank, control)
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    expected_ranks = list(map(int, representation.ranks))
    visible = None
    decoded = None
    if population == "rankcloak":
        quality_rank_ceiling = 8 if representation.name == "ascii_b8" else 16
        generated = generate_rank_span(
            model,
            context,
            expected_ranks,
            allowed_token_mask=None,
            leadin_token_count=0,
            tail_policy="none",
            quality_rank_ceiling=quality_rank_ceiling,
        )
        output_ids = list(map(int, generated["full_token_ids"]))
        own_trace = trace_observed_tokens(model, context, output_ids)
        if list(map(int, own_trace["observed_ranks"])) != expected_ranks:
            raise GenerationExecutionError("Q8 generated ranks did not replay exactly")
        decoded = decode_representation(model, representation, own_trace["observed_ranks"])
        if not bool(decoded["exact_payload_recovery"]):
            raise GenerationExecutionError("Q8 saved-ID replay did not recover payload")
        visible_diagnostic = retokenize_message(model, generated)
        visible_error = None
        try:
            visible_replay = recover_rank_span(
                model,
                context,
                [],
                visible_diagnostic["forced_token_ids"],
                allowed_token_mask=None,
            )
            visible_decoded = decode_representation(
                model, representation, visible_replay["ranks"]
            )
        except Exception as exc:
            visible_replay = None
            visible_decoded = None
            visible_error = "{}: {}".format(type(exc).__name__, exc)
        visible = {
            "diagnostic": visible_diagnostic,
            "replay": visible_replay,
            "decoded": visible_decoded,
            "exact_payload_recovery": bool(
                visible_decoded and visible_decoded.get("exact_payload_recovery")
            ),
            "error": visible_error,
        }
    else:
        generated = generate_length_matched_control(
            model,
            prompt_text,
            int(row["target_token_count"]),
            int(row["historical_control_sampling_seed"]),
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            context_limit=CONTEXT_LIMIT,
        )
        output_ids = list(map(int, generated["token_ids"]))
        own_trace = {
            "context_token_ids": list(map(int, generated["context_token_ids"])),
            "observed_token_ids": output_ids,
            "position_count": len(output_ids),
            "entropy_bits": list(map(float, generated["next_token_entropies_bits"])),
            "observed_ranks": list(map(int, generated["sampled_token_ranks"])),
            "observed_log_probabilities": list(
                map(float, generated["token_log_probabilities"])
            ),
            "observed_surprisals_nats": [
                float(-value) for value in generated["token_log_probabilities"]
            ],
            "greedy_token_ids": list(map(int, generated["greedy_token_ids"])),
            "greedy_log_probabilities": list(
                map(float, generated["greedy_log_probabilities"])
            ),
            "rank_pressure_log_probability_gaps_nats": list(
                map(float, generated["rank_pressure_log_probability_gaps_nats"])
            ),
        }
    if len(output_ids) != int(row["target_token_count"]):
        raise GenerationExecutionError("Q8 output target length mismatch")
    historical_path_trace = trace_observed_tokens(model, context, historical_ids)
    comparison = paired_distribution_comparison(
        q4_replay["distribution_trace"], historical_path_trace
    )
    divergence = first_divergence(historical_ids, output_ids)
    record = dict(
        operation_provenance(
            QUANTIZATION_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record.update(
        {
            "record_type": "quantization_q8_generation",
            "model_id": row["model_id"],
            "population": population,
            "new_generation_performed": True,
            "rendered_prompt": prompt_text,
            "context_token_ids": context,
            "generation": generated,
            "saved_token_id_decoded": decoded,
            "visible_text_retokenization": visible,
            "q8_own_path_distribution_trace": own_trace,
            "q8_own_path_distribution_summary": trace_summary(own_trace),
            "q8_replay_of_historical_q4_path": historical_path_trace,
            "q4_q8_same_path_distribution_comparison": comparison,
            "q4_q8_generated_output_comparison": {
                "q4_token_count": len(historical_ids),
                "q8_token_count": len(output_ids),
                "exact_token_sequence_match": historical_ids == output_ids,
                "first_divergence": divergence,
                "positionwise_token_match_fraction": (
                    float(
                        np.mean(
                            np.asarray(historical_ids, dtype=int)
                            == np.asarray(output_ids, dtype=int)
                        )
                    )
                    if len(historical_ids) == len(output_ids) and output_ids
                    else None
                ),
            },
            "paired_q4_replay_plan_id": q4_replay["plan_id"],
            "paired_q4_replay_sha256": canonical_sha256(q4_replay),
            "source_lineage": {
                "rank_trial_id": rank["trial_id"],
                "rank_record_sha256": canonical_sha256(rank),
                "control_id": control["control_id"],
                "control_record_sha256": canonical_sha256(control),
            },
            "validation": {
                "embedded_tokenizer_context_exact": True,
                "non_quantization_contract_matches_q4": True,
                "historical_seed_matches_q4": True,
                "target_length_exact": True,
                "q8_saved_rank_replay_exact_when_applicable": (
                    list(map(int, own_trace["observed_ranks"])) == expected_ranks
                    if population == "rankcloak"
                    else True
                ),
                "q8_payload_recovery_exact_when_applicable": (
                    bool(decoded and decoded.get("exact_payload_recovery"))
                    if population == "rankcloak"
                    else True
                ),
            },
        }
    )
    if not all(record["validation"].values()):
        raise GenerationExecutionError("Q8 generation validation failed")
    return record


def plan_for_phase(phase: str) -> Tuple[Path, List[Dict[str, str]]]:
    if phase == "entropy_calibration":
        return CALIBRATION_PLAN, load_csv(CALIBRATION_PLAN)
    if phase == "entropy":
        return ENTROPY_PLAN, load_csv(ENTROPY_PLAN)
    if phase == "quantization":
        return QUANTIZATION_PLAN, load_csv(QUANTIZATION_PLAN)
    raise GenerationExecutionError("unknown execution phase")


def sort_phase_rows(phase: str, rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    gate_order = {"ungated": 0, "moderate": 1, "strict": 2}
    population_order = {"rankcloak": 0, "ordinary_control": 1}
    if phase == "entropy":
        return sorted(
            rows,
            key=lambda row: (
                row["experimental_cell_id"],
                gate_order[row["gate_level"]],
                population_order[row["population"]],
                row["plan_id"],
            ),
        )
    return sorted(rows, key=lambda row: row["plan_id"])


def phase_model_rows(phase: str, model_id: str) -> Tuple[Path, List[Dict[str, str]]]:
    plan_path, rows = plan_for_phase(phase)
    selected = [row for row in rows if str(row["model_id"]) == str(model_id)]
    if phase == "quantization":
        if model_id == "qwen2_5_7b_instruct_q4_k_m":
            selected = [row for row in selected if row["quantization"] == "Q4_K_M"]
        elif model_id == "qwen2_5_7b_instruct_q8_0":
            selected = [row for row in selected if row["quantization"] == "Q8_0"]
    if not selected:
        raise GenerationExecutionError("phase/model selection has no planned rows")
    return plan_path, sort_phase_rows(phase, selected)


def write_failure(
    path: Path,
    phase: str,
    model_id: str,
    row: Mapping[str, str],
    exc: BaseException,
    smoke: bool,
) -> Path:
    target = path
    if target.exists():
        target = target.with_name(
            target.stem
            + ".retry-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + target.suffix
        )
    atomic_json(
        target,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_status": "failed",
            "phase": phase,
            "model_id": model_id,
            "smoke": bool(smoke),
            "plan_id": row["plan_id"],
            "plan_row": dict(row),
            "plan_row_sha256": canonical_sha256(row),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "execution_git_commit": git_output("rev-parse", "HEAD"),
        },
    )
    return target


def phase_status(
    result_root: Path, phase: str, model_id: str, smoke: bool = False
) -> Mapping[str, object]:
    _, rows = phase_model_rows(phase, model_id)
    completed = []
    failed = []
    pending = []
    for row in rows:
        result_path = model_record_path(
            result_root, phase, model_id, row["plan_id"], smoke
        )
        failure_path = failure_record_path(
            result_root, phase, model_id, row["plan_id"], smoke
        )
        if result_path.is_file():
            completed.append(row["plan_id"])
        elif failure_path.is_file():
            failed.append(row["plan_id"])
        else:
            pending.append(row["plan_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "model_id": model_id,
        "smoke": bool(smoke),
        "planned": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "pending": len(pending),
        "completed_plan_ids": completed,
        "failed_plan_ids": failed,
        "pending_plan_ids": pending,
        "checked_at": utc_now(),
    }


def run_phase(
    *,
    phase: str,
    model_id: str,
    result_root: Path,
    gpu_uuid: str,
    smoke: bool,
    limit: Optional[int],
    retry_failed: bool,
    continue_on_error: bool,
) -> Mapping[str, object]:
    plan_path, rows = phase_model_rows(phase, model_id)
    if smoke:
        rows = rows[: int(limit or 1)]
    elif limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise GenerationExecutionError("no rows selected")
    configure_deterministic_gpu(gpu_uuid)
    monitor = GpuMemoryMonitor(gpu_uuid)
    monitor.start()
    model = None
    run_started = utc_now()
    run_perf = time.perf_counter()
    completed_count = 0
    resumed_count = 0
    failure_count = 0
    try:
        model, model_manifest = load_verified_model(model_id, gpu_uuid, monitor)
        run_manifest_path = (
            result_root
            / "provenance/model_runs"
            / phase
            / model_id
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        )
        atomic_json(
            run_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "model_id": model_id,
                "smoke": bool(smoke),
                "selected_plan_row_count": len(rows),
                "gpu_uuid": gpu_uuid,
                "started_at": run_started,
                "model_manifest": model_manifest,
                "plan_file": str(plan_path.relative_to(PROJECT_ROOT)),
                "plan_file_sha256": file_sha256(plan_path),
                "command": [sys.executable, *sys.argv],
            },
        )
        model_sha = str(model_manifest["artifact"]["sha256"])
        entropy_sources = (
            historical_primary_records(model_id) if phase == "entropy" else {}
        )
        payloads = payload_index() if phase in {"entropy", "quantization"} else {}
        thresholds = (
            load_model_thresholds(result_root, model_id)
            if phase == "entropy"
            else None
        )
        quant_indexes = qwen_historical_indexes() if phase == "quantization" else None
        quant_rows = load_csv(QUANTIZATION_PLAN) if phase == "quantization" else []
        quant_by_pair = {
            (row["pairing_unit_id"], row["quantization"]): row for row in quant_rows
        }
        entropy_rank_by_pair = {
            row["pairing_unit_id"]: row
            for row in load_csv(ENTROPY_PLAN)
            if row["population"] == "rankcloak"
        }
        for index, row in enumerate(rows, 1):
            result_path = model_record_path(
                result_root, phase, model_id, row["plan_id"], smoke
            )
            failure_path = failure_record_path(
                result_root, phase, model_id, row["plan_id"], smoke
            )
            row_sha = canonical_sha256(row)
            if validated_existing(result_path, row_sha, model_sha):
                resumed_count += 1
                print(
                    json.dumps(
                        {
                            "event": "resume_skip",
                            "phase": phase,
                            "model_id": model_id,
                            "plan_id": row["plan_id"],
                            "index": index,
                            "selected": len(rows),
                        }
                    ),
                    flush=True,
                )
                continue
            if failure_path.exists() and not retry_failed:
                raise GenerationExecutionError(
                    "prior failure exists; pass --retry-failed after inspection"
                )
            try:
                if phase == "entropy_calibration":
                    record = execute_calibration_row(
                        model, row, model_manifest, monitor
                    )
                elif phase == "entropy":
                    if row["population"] == "rankcloak":
                        source_id = row["reference_same_payload_trial_id"]
                        if source_id not in entropy_sources:
                            raise GenerationExecutionError(
                                "entropy source trial is unavailable"
                            )
                        source = entropy_sources[source_id]
                        representation = representation_from_source(source, payloads)
                        record = entropy_rankcloak_record(
                            model,
                            row,
                            source,
                            representation,
                            thresholds,
                            model_manifest,
                            monitor,
                        )
                    else:
                        rank_row = entropy_rank_by_pair[row["pairing_unit_id"]]
                        rank_path = model_record_path(
                            result_root,
                            phase,
                            model_id,
                            rank_row["plan_id"],
                            smoke,
                        )
                        if not rank_path.is_file():
                            raise GenerationExecutionError(
                                "paired RankCloak result must precede control"
                            )
                        record = entropy_control_record(
                            model,
                            row,
                            load_json(rank_path),
                            model_manifest,
                            monitor,
                        )
                else:
                    assert quant_indexes is not None
                    rank, control, _task = quantization_source_bundle(
                        row, quant_indexes
                    )
                    if row["quantization"] == "Q4_K_M":
                        record = quantization_q4_replay_record(
                            model,
                            row,
                            rank,
                            control,
                            model_manifest,
                            monitor,
                        )
                    else:
                        q4_row = quant_by_pair[(row["pairing_unit_id"], "Q4_K_M")]
                        q4_path = model_record_path(
                            result_root,
                            phase,
                            "qwen2_5_7b_instruct_q4_k_m",
                            q4_row["plan_id"],
                            smoke,
                        )
                        if not q4_path.is_file():
                            raise GenerationExecutionError(
                                "paired Q4 model-backed replay must precede Q8"
                            )
                        representation = representation_from_source(rank, payloads)
                        record = quantization_q8_generation_record(
                            model,
                            row,
                            q4_row,
                            load_json(q4_path),
                            rank,
                            control,
                            representation,
                            model_manifest,
                            monitor,
                        )
                immutable_result(result_path, record)
                completed_count += 1
                print(
                    json.dumps(
                        {
                            "event": "completed",
                            "phase": phase,
                            "model_id": model_id,
                            "plan_id": row["plan_id"],
                            "index": index,
                            "selected": len(rows),
                            "seconds": record["execution_seconds"],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                failure_count += 1
                written = write_failure(
                    failure_path, phase, model_id, row, exc, smoke
                )
                print(
                    json.dumps(
                        {
                            "event": "failed",
                            "phase": phase,
                            "model_id": model_id,
                            "plan_id": row["plan_id"],
                            "failure_path": str(written),
                            "error": "{}: {}".format(type(exc).__name__, exc),
                        }
                    ),
                    flush=True,
                )
                if not continue_on_error:
                    raise
        if phase == "entropy_calibration" and not smoke and limit is None:
            freeze_model_thresholds(result_root, model_id)
    finally:
        model = None
        monitor.stop()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "model_id": model_id,
        "smoke": bool(smoke),
        "selected": len(rows),
        "completed_this_run": completed_count,
        "resumed_completed": resumed_count,
        "failed_this_run": failure_count,
        "started_at": run_started,
        "completed_at": utc_now(),
        "execution_seconds": float(time.perf_counter() - run_perf),
    }
    status_path = (
        result_root
        / "status"
        / ("smoke" if smoke else "full")
        / (phase + "__" + model_id + ".json")
    )
    atomic_json(status_path, summary)
    return summary


def dry_run_summary(
    phase: str, model_id: str, result_root: Path, smoke: bool, limit: Optional[int]
) -> Mapping[str, object]:
    plan_path, rows = phase_model_rows(phase, model_id)
    if smoke:
        rows = rows[: int(limit or 1)]
    elif limit is not None:
        rows = rows[: int(limit)]
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "model_loaded": False,
        "phase": phase,
        "model_id": model_id,
        "smoke": bool(smoke),
        "selected_row_count": len(rows),
        "plan_file": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_file_sha256": file_sha256(plan_path),
        "first_plan_id": rows[0]["plan_id"] if rows else None,
        "last_plan_id": rows[-1]["plan_id"] if rows else None,
        "result_root": str(result_root),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("entropy_calibration", "entropy", "quantization"),
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    result_root = args.output_dir.resolve()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.status:
        print(
            json.dumps(
                phase_status(
                    result_root, args.phase, args.model_id, smoke=args.smoke
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                dry_run_summary(
                    args.phase,
                    args.model_id,
                    result_root,
                    args.smoke,
                    args.limit,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.gpu_uuid:
        parser.error("--gpu-uuid is required for model-backed execution")
    if args.gpu_uuid != DEFAULT_GPU_UUID:
        parser.error("only the authorized RTX 5000 Ada UUID is accepted")
    summary = run_phase(
        phase=args.phase,
        model_id=args.model_id,
        result_root=result_root,
        gpu_uuid=args.gpu_uuid,
        smoke=args.smoke,
        limit=args.limit,
        retry_failed=args.retry_failed,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "GenerationExecutionError",
    "GpuMemoryMonitor",
    "atomic_json",
    "canonical_sha256",
    "dry_run_summary",
    "entropy_control_record",
    "entropy_rankcloak_record",
    "freeze_model_thresholds",
    "main",
    "paired_distribution_comparison",
    "phase_status",
    "quantization_q4_replay_record",
    "quantization_q8_generation_record",
    "run_phase",
    "verify_model_artifact",
]
