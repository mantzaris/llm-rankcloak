"""Run leakage-resistant raw-text detectors for the RankCloak revision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_detection import (  # noqa: E402
    CONFIRMATORY_TRANSFORMER_ARTIFACTS,
    CONFIRMATORY_TRANSFORMER_MODEL_ID,
    CONFIRMATORY_TRANSFORMER_RELATIVE_PATH,
    CONFIRMATORY_TRANSFORMER_REVISION,
    RevisionDetectionError,
    load_detector_config,
    read_detector_frame,
    prepare_revision_detector_suite,
    split_manifest_rows,
)
from scripts.build_detector_cuda_budget_gate import read_gate  # noqa: E402
from rankcloak.revision_artifacts import canonical_json_sha256  # noqa: E402
from rankcloak.revision_detector_execution import (  # noqa: E402
    DetectorExecutionContext,
    EXPECTED_GPU_LEDGER_SOURCES,
    atomic_write_json,
    build_validated_checkpoint_equivalence_payload,
    detector_finalization_paths,
    execute_checkpointed_detector_suite,
    mark_detector_awaiting_supervisor_finalization,
    read_detector_gpu_accounting_ledger,
    read_detector_cuda_reproducibility_report,
    verify_status_file,
    write_detector_cuda_reproducibility_report,
    write_detector_finalization_candidate,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "detectors" / "default.json"
DEFAULT_CONFIRMATORY_PLAN = (
    PROJECT_ROOT / "analysis" / "revision_v1" / "detector_confirmatory_plan.json"
)
DEFAULT_EXECUTION_POLICY = (
    PROJECT_ROOT
    / "operations"
    / "confirmatory_v2"
    / "detector_cuda_policy_v2.json"
)
ENVIRONMENT_ROOT = PROJECT_ROOT / "environment" / "revision_v1"
DEFAULT_EQUIVALENCE_ROOT = (
    PROJECT_ROOT / "results" / "revision_v1" / "detector_cuda_reproducibility_v2"
)
DEFAULT_EQUIVALENCE_REPORTS = tuple(
    DEFAULT_EQUIVALENCE_ROOT
    / "task_{}".format(index)
    / "cuda_reproducibility_report.json"
    for index in (0, 1)
)
DEFAULT_GPU_ACCOUNTING_LEDGER = (
    DEFAULT_EQUIVALENCE_ROOT / "gpu_accounting_ledger.json"
)
DEFAULT_CUDA_BUDGET_GATE = DEFAULT_EQUIVALENCE_ROOT / "cuda_budget_gate.json"
PRIMARY_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "primary.json"
PROMPT_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "prompts.json"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "models.json"
PRIMARY_EVIDENCE_STATUS = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
PREPROCESSING_BINDING_SCHEMA = (
    "rankcloak-revision-primary-detector-preprocessing-binding-v1"
)
OUTPUT_FILENAMES = (
    "detector_metrics.csv",
    "detector_predictions.csv",
    "detector_dataset_manifest.csv",
    "detector_split_manifest.json",
    "detector_failures.json",
    "detector_run_manifest.json",
)


def _verify_execution_environment() -> dict:
    """Verify and bind the complete frozen environment bundle file set."""

    declared_root = ENVIRONMENT_ROOT
    if declared_root.is_symlink() or not declared_root.is_dir():
        raise RevisionDetectionError("Frozen environment root is missing or unsafe.")
    root = declared_root.resolve()
    manifest_path = root / "environment_manifest.json"
    manifest = _read_json_object(manifest_path, "environment manifest")
    records = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_type")
        != "rankcloak_revision_environment_file_set"
        or manifest.get("snapshot_status") != "complete"
        or not isinstance(records, list)
        or int(manifest.get("file_count", -1)) != len(records)
        or manifest.get("files_sha256") != canonical_json_sha256(records)
    ):
        raise RevisionDetectionError("Frozen environment manifest is invalid.")
    listed = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RevisionDetectionError("Frozen environment manifest row is invalid.")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RevisionDetectionError("Frozen environment path is unsafe.")
        path = root / relative
        if (
            relative.as_posix() in listed
            or not path.is_file()
            or path.is_symlink()
            or int(path.stat().st_size) != int(record["size_bytes"])
            or _sha256_file(path) != record["sha256"]
        ):
            raise RevisionDetectionError(
                "Frozen environment file identity differs: {}".format(relative)
            )
        listed.add(relative.as_posix())
    all_entries = list(root.rglob("*"))
    unsafe_entries = [
        path
        for path in all_entries
        if path.is_symlink() or not (path.is_file() or path.is_dir())
    ]
    if unsafe_entries:
        raise RevisionDetectionError(
            "Frozen environment contains symlink/special entries: {}".format(
                ", ".join(
                    path.relative_to(root).as_posix()
                    for path in unsafe_entries
                )
            )
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in all_entries
        if path.is_file()
    }
    if actual != listed | {"environment_manifest.json"}:
        raise RevisionDetectionError("Frozen environment file set differs.")
    expected_directories = {
        parent.as_posix()
        for name in actual
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in all_entries
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise RevisionDetectionError("Frozen environment directory set differs.")
    checksum_path = root / "CHECKSUMS.sha256"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise RevisionDetectionError("Frozen environment CHECKSUMS is unsafe.")
    checksum_rows = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match.group(2) in checksum_rows:
            raise RevisionDetectionError("Frozen environment CHECKSUMS is invalid.")
        checksum_rows[match.group(2)] = match.group(1)
    expected_checksum_paths = listed - {"CHECKSUMS.sha256"}
    if set(checksum_rows) != expected_checksum_paths or any(
        _sha256_file(root / relative) != digest
        for relative, digest in checksum_rows.items()
    ):
        raise RevisionDetectionError("Frozen environment CHECKSUMS differs.")
    required = {}
    for name in (
        "environment_manifest.json",
        "scientific_pins.json",
        "CHECKSUMS.sha256",
    ):
        path = root / name
        required[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "schema_version": "rankcloak-revision-detector-environment-binding-v1",
        "root": str(root),
        "required_files": required,
        "environment_files_sha256": manifest["files_sha256"],
        "verified_file_count": len(records),
        "verification_status": "ok",
    }


def _verify_execution_policy(path: Path) -> dict:
    """Validate the closed CUDA-only pre-benchmark policy exactly."""

    declared_path = Path(path)
    if declared_path.is_symlink() or not declared_path.is_file():
        raise RevisionDetectionError(
            "Detector acceleration policy is missing or unsafe."
        )
    policy_path = declared_path.resolve()
    policy = _read_json_object(policy_path, "detector acceleration policy")
    unsigned = dict(policy)
    claimed = unsigned.pop("policy_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise RevisionDetectionError("Detector acceleration policy self-hash differs.")
    expected_keys = {
        "schema_version",
        "policy_status",
        "authorized_ceiling",
        "audit",
        "execution",
        "benchmark",
        "equivalence",
        "ceiling",
        "policy_sha256",
    }
    if set(policy) != expected_keys or (
        policy.get("schema_version")
        != "rankcloak-revision-detector-cuda-policy-v2"
        or policy.get("policy_status")
        != "cuda_only_predeclared_before_new_benchmarks"
    ):
        raise RevisionDetectionError("Detector acceleration policy schema differs.")
    projection = policy.get("authorized_ceiling")
    execution = policy.get("execution")
    benchmark = policy.get("benchmark")
    equivalence = policy.get("equivalence")
    ceiling = policy.get("ceiling")
    audit = policy.get("audit")
    if projection != {
        "gpu_hours": 165.0,
        "historical_actual_gpu_hours_floor": 62.4783840698,
        "projection_path": "results/revision_v1/compute_projection_165h_v2.json",
        "projection_sha256": (
            "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
        ),
    }:
        raise RevisionDetectionError("Detector policy ceiling authorization differs.")
    projection_path = PROJECT_ROOT / str(projection["projection_path"])
    projection_document = _read_json_object(projection_path, "compute projection")
    projection_unsigned = dict(projection_document)
    projection_claimed = projection_unsigned.pop("projection_sha256", None)
    if (
        projection_claimed != projection["projection_sha256"]
        or canonical_json_sha256(projection_unsigned) != projection_claimed
    ):
        raise RevisionDetectionError("Detector policy projection identity differs.")
    if execution != {
        "device": "cuda:0",
        "gpu_uuid": "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf",
        "workers": 1,
        "torch_num_threads": 1,
    }:
        raise RevisionDetectionError("Detector policy execution contract differs.")
    if benchmark != {
        "task_indices": [0, 1],
        "checkpoint_reuse": True,
        "cuda_reproducibility_fit_count_per_architecture": 2,
        "allowed_failed_fit_retry_count_per_architecture": 1,
        "projection_safety_multiplier": 1.5,
        "full_matrix_budget_gate_required": True,
    }:
        raise RevisionDetectionError("Detector policy benchmark contract differs.")
    expected_equivalence = {
        "same_device_cuda": {
            "task_design_exact": True,
            "row_identity_order_labels_exact": True,
            "model_state_sha256_exact": True,
            "scores_exact": True,
            "metrics_exact": True,
            "predictions_exact": True,
        }
    }
    if equivalence != expected_equivalence:
        raise RevisionDetectionError("Detector policy equivalence contract differs.")
    if ceiling != {
        "next_fit_upper_seconds_by_detector": {
            "published_textcnn_equivalent": 900.0,
            "deberta_v3_base_classifier": 7200.0,
        },
        "post_benchmark_tighter_gate_required": True,
    }:
        raise RevisionDetectionError("Detector policy next-fit bounds differ.")
    if not isinstance(audit, dict) or set(audit) != {
        "diagnostic_path",
        "diagnostic_sha256",
        "cpu_diagnostics_status",
        "cpu_neural_training_authorized",
        "derivation",
    } or audit.get("cpu_diagnostics_status") != (
        "preserved_feasibility_evidence_only"
    ) or audit.get("cpu_neural_training_authorized") is not False or audit.get(
        "derivation"
    ) != "revision_takeover_2026-08-15_cuda_only_v2":
        raise RevisionDetectionError("Detector policy audit provenance differs.")
    audit_path = PROJECT_ROOT / str(audit["diagnostic_path"])
    if (
        not audit_path.is_file()
        or audit_path.is_symlink()
        or _sha256_file(audit_path) != audit.get("diagnostic_sha256")
    ):
        raise RevisionDetectionError("Detector policy diagnostic identity differs.")
    return {
        "path": str(policy_path),
        "sha256": _sha256_file(policy_path),
        "size_bytes": int(policy_path.stat().st_size),
        "policy_sha256": claimed,
        "policy": policy,
    }


def _equivalence_policy_identity(
    execution_policy_binding: dict, environment_binding: dict
) -> dict:
    return {
        "schema_version": "rankcloak-revision-detector-equivalence-policy-identity-v1",
        "execution_policy_path": execution_policy_binding["path"],
        "execution_policy_sha256": execution_policy_binding["sha256"],
        "execution_policy_content_sha256": execution_policy_binding[
            "policy_sha256"
        ],
        "environment_binding": environment_binding,
        "environment_binding_sha256": canonical_json_sha256(
            environment_binding
        ),
        "equivalence_policy": execution_policy_binding["policy"][
            "equivalence"
        ],
        "equivalence_policy_sha256": canonical_json_sha256(
            execution_policy_binding["policy"]["equivalence"]
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict:
    def command(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty_worktree": None if status is None else bool(status),
    }


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _write_csv(path: Path, frame: object) -> None:
    """Write a final tabular product atomically before the run manifest seal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-{}-".format(path.name), dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise RevisionDetectionError("Output path is not a directory: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)
    entries = sorted(item.name for item in path.iterdir())
    conflicts = sorted(set(entries) & set(OUTPUT_FILENAMES))
    unknown = sorted(set(entries) - set(OUTPUT_FILENAMES))
    if entries and not overwrite:
        raise RevisionDetectionError(
            "Output directory is not empty; choose a new directory or pass --overwrite: "
            "{}".format(", ".join(entries))
        )
    if overwrite and unknown:
        raise RevisionDetectionError(
            "Refusing --overwrite because the output directory contains unknown entries: "
            "{}".format(", ".join(unknown))
        )
    unsafe = [
        name
        for name in conflicts
        if (path / name).is_symlink() or not (path / name).is_file()
    ]
    if unsafe:
        raise RevisionDetectionError(
            "Refusing to overwrite non-regular or symlink detector products: {}".format(
                ", ".join(unsafe)
            )
        )


def _runtime_value(argument: object, environment_name: str, default: object) -> object:
    """Resolve an explicit CLI runtime override, then a supervisor environment value."""

    if argument is not None:
        return argument
    value = os.environ.get(environment_name)
    return default if value in (None, "") else value


def _bind_runtime_before_torch_import(
    *, device: str, gpu_uuid: Optional[str], workers: int
) -> None:
    if workers != 1:
        raise RevisionDetectionError(
            "Only --workers 1 is implemented safely; refusing an unsafe worker count."
        )
    if device not in {"cpu", "cuda:0"}:
        raise RevisionDetectionError("--device must be cpu or cuda:0.")
    if device == "cuda:0":
        if not gpu_uuid or not str(gpu_uuid).startswith("GPU-"):
            raise RevisionDetectionError(
                "--device cuda:0 requires an exact --gpu-uuid beginning GPU-."
            )
        observed = os.environ.get("CUDA_VISIBLE_DEVICES")
        if observed not in (None, str(gpu_uuid)):
            raise RevisionDetectionError(
                "CUDA_VISIBLE_DEVICES is already bound to a different device."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    elif gpu_uuid is not None:
        raise RevisionDetectionError("--gpu-uuid is valid only with --device cuda:0.")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _runtime_config(config: dict, device: str) -> dict:
    """Copy the frozen config and override execution-only device/thread fields."""

    value = json.loads(json.dumps(config))
    for detector in value.get("detectors", []):
        detector["device"] = device
        detector["torch_num_threads"] = 1
    return value


def _load_confirmatory_plan(path: Path, config_path: Path) -> dict:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RevisionDetectionError(
            "Confirmatory detector plan does not exist: {}".format(path)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RevisionDetectionError(
            "Confirmatory detector plan is invalid JSON: {}".format(exc)
        ) from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != (
        "rankcloak-revision-detector-confirmatory-plan-v1"
    ):
        raise RevisionDetectionError("Unexpected confirmatory detector plan schema.")
    if str(plan.get("base_config_sha256", "")) != _sha256_file(config_path):
        raise RevisionDetectionError(
            "Confirmatory plan does not pin the selected detector config bytes."
        )
    transformer = plan.get("transformer_pin")
    if not isinstance(transformer, dict):
        raise RevisionDetectionError("Confirmatory plan is missing transformer_pin.")
    expected_artifacts = {
        name: {
            "sha256": str(specification["sha256"]),
            "size_bytes": int(specification["size_bytes"]),
        }
        for name, specification in CONFIRMATORY_TRANSFORMER_ARTIFACTS.items()
    }
    expected_transformer = {
        "upstream_model_id": CONFIRMATORY_TRANSFORMER_MODEL_ID,
        "upstream_revision": CONFIRMATORY_TRANSFORMER_REVISION,
        "local_path": CONFIRMATORY_TRANSFORMER_RELATIVE_PATH,
        "artifact_policy": "exact_regular_files_no_symlinks_cache_metadata_ignored",
        "artifacts": expected_artifacts,
    }
    if transformer != expected_transformer:
        raise RevisionDetectionError(
            "Confirmatory plan transformer pin differs from the code-enforced frozen pin."
        )
    policy = plan.get("implementation_policy")
    if policy != {
        "allow_model_downloads": False,
        "allow_smoke_fallback": False,
        "require_all_detector_split_executions_complete": True,
        "required_detector_identities": [
            {
                "kind": "text_cnn",
                "name": "published_textcnn_equivalent",
            },
            {
                "kind": "pretrained_transformer",
                "name": "deberta_v3_base_classifier",
            },
        ],
    }:
        raise RevisionDetectionError(
            "Confirmatory plan implementation policy is not the frozen policy."
        )
    return plan


def _read_json_object(path: Path, label: str) -> dict:
    if Path(path).is_symlink() or not Path(path).is_file():
        raise RevisionDetectionError(
            "{} is missing or unsafe: {}".format(label, path)
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RevisionDetectionError(
            "Cannot load {} at {}: {}".format(label, path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise RevisionDetectionError("{} must be a JSON object.".format(label))
    return value


def _read_single_checkpoint_metric(path: Path) -> dict:
    """Read one validated columnar metric row from an atomic fit checkpoint."""

    payload = _read_json_object(path, "detector fit metric checkpoint")
    columns = payload.get("columns")
    rows = payload.get("rows")
    if (
        set(payload) != {"schema_version", "columns", "rows"}
        or payload.get("schema_version")
        != "rankcloak-revision-detector-fit-rows-v1"
        or not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], list)
        or len(rows[0]) != len(columns)
    ):
        raise RevisionDetectionError(
            "Detector fit metric checkpoint row payload is malformed."
        )
    metric = dict(zip(columns, rows[0]))
    metadata = metric.get("implementation_metadata_json")
    if not isinstance(metadata, str) or not metadata:
        raise RevisionDetectionError(
            "Detector fit metric checkpoint lacks implementation metadata."
        )
    return metric


def _resolve_preprocessing_output(
    manifest_path: Path, declaration: dict, label: str
) -> Path:
    raw = Path(str(declaration.get("path", "")))
    declared_candidate = raw if raw.is_absolute() else manifest_path.parent / raw
    if not declared_candidate.is_file() or declared_candidate.is_symlink():
        raise RevisionDetectionError(
            "Primary preprocessing {} is missing or unsafe: {}".format(
                label, declared_candidate
            )
        )
    return declared_candidate.resolve()


def _verify_primary_preprocessing_detector_input(
    input_path: Path,
    preprocessing_manifest_path: Path,
    contract: dict,
) -> dict:
    """Bind confirmatory detector bytes to the strict primary preprocessing run."""

    raw_manifest_path = Path(preprocessing_manifest_path)
    raw_input_path = Path(input_path)
    if not raw_manifest_path.is_file() or raw_manifest_path.is_symlink():
        raise RevisionDetectionError(
            "Confirmatory preprocessing manifest is missing or unsafe: {}".format(
                raw_manifest_path
            )
        )
    if not raw_input_path.is_file() or raw_input_path.is_symlink():
        raise RevisionDetectionError(
            "Confirmatory detector input is missing or unsafe: {}".format(
                raw_input_path
            )
        )
    manifest_path = raw_manifest_path.resolve()
    input_path = raw_input_path.resolve()
    manifest = _read_json_object(manifest_path, "primary preprocessing manifest")
    if (
        manifest.get("schema_version") != "2.0"
        or manifest.get("manifest_type") != "revision_preprocessing_outputs"
    ):
        raise RevisionDetectionError(
            "Confirmatory detector input requires preprocessing output schema 2.0."
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or manifest.get(
        "outputs_sha256"
    ) != canonical_json_sha256(outputs):
        raise RevisionDetectionError("Primary preprocessing output list hash differs.")
    by_role = {
        str(row.get("role", "")): row for row in outputs if isinstance(row, dict)
    }
    if len(by_role) != len(outputs) or not {"detector", "input_manifest"}.issubset(
        by_role
    ):
        raise RevisionDetectionError(
            "Primary preprocessing manifest lacks unique detector/input-manifest outputs."
        )
    detector_declaration = by_role["detector"]
    detector_path = _resolve_preprocessing_output(
        manifest_path, detector_declaration, "detector output"
    )
    detector_hash = _sha256_file(detector_path)
    expected_rows = int(contract["rows"])
    if (
        detector_path != input_path
        or detector_hash != str(detector_declaration.get("sha256", ""))
        or detector_path.stat().st_size
        != int(detector_declaration.get("size_bytes", -1))
        or int(detector_declaration.get("row_count", -1)) != expected_rows
        or int(manifest.get("row_counts", {}).get("detector", -1))
        != expected_rows
    ):
        raise RevisionDetectionError(
            "--input is not the exact detector artifact declared by primary preprocessing."
        )
    invariants = manifest.get("invariants")
    if not isinstance(invariants, dict) or (
        int(invariants.get("detector_pair_count", -1))
        != int(contract["positive_rows"])
        or invariants.get("detector_grouping_unit") != "payload_name"
    ):
        raise RevisionDetectionError(
            "Primary preprocessing detector-pair invariants differ from the contract."
        )
    input_declaration = by_role["input_manifest"]
    input_manifest_path = _resolve_preprocessing_output(
        manifest_path, input_declaration, "input manifest"
    )
    input_manifest_hash = _sha256_file(input_manifest_path)
    if (
        input_manifest_hash != str(input_declaration.get("sha256", ""))
        or input_manifest_hash != manifest.get("input_manifest_sha256")
        or input_manifest_path.stat().st_size
        != int(input_declaration.get("size_bytes", -1))
    ):
        raise RevisionDetectionError("Primary preprocessing input-manifest identity differs.")
    input_manifest = _read_json_object(
        input_manifest_path, "primary preprocessing input manifest"
    )
    input_files = input_manifest.get("input_files")
    if (
        input_manifest.get("schema_version") != "2.0"
        or input_manifest.get("manifest_type") != "revision_preprocessing_inputs"
        or input_manifest.get("strict_complete") is not True
        or int(input_manifest.get("emitted_run_count", -1)) != 3
        or int(input_manifest.get("reference_run_count", -1)) != 0
        or not isinstance(input_files, list)
        or input_manifest.get("input_files_sha256")
        != canonical_json_sha256(input_files)
    ):
        raise RevisionDetectionError(
            "Primary preprocessing input lineage is not strict and hash-complete."
        )
    verified_input_paths = set()
    verified_input_records = []
    for record in input_files:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
        ):
            raise RevisionDetectionError(
                "Primary preprocessing input-file declaration is malformed."
            )
        raw_declared_path = Path(record["path"])
        declared_path = (
            raw_declared_path
            if raw_declared_path.is_absolute()
            else input_manifest_path.parent / raw_declared_path
        )
        if (
            declared_path.is_symlink()
            or not declared_path.is_file()
            or declared_path.resolve() in verified_input_paths
            or int(declared_path.stat().st_size)
            != int(record.get("size_bytes", -1))
            or _sha256_file(declared_path) != record["sha256"]
        ):
            raise RevisionDetectionError(
                "Primary preprocessing input-file bytes differ or are unsafe: {}".format(
                    declared_path
                )
            )
        verified_input_paths.add(declared_path.resolve())
        verified_input_records.append(
            {
                "path": str(declared_path.resolve()),
                "sha256": record["sha256"],
                "size_bytes": int(record["size_bytes"]),
                "role": record.get("role"),
                "run_identity_sha256": record.get("run_identity_sha256"),
            }
        )
    shards = input_manifest.get("run_shards")
    if not isinstance(shards, list):
        raise RevisionDetectionError("Primary preprocessing run_shards is malformed.")
    emitted = [row for row in shards if isinstance(row, dict) and row.get("role") == "input"]
    if (
        len(emitted) != 3
        or len(shards) != 3
        or {str(row.get("model_id", "")) for row in emitted}
        != set(map(str, contract["model_ids"]))
        or any(
            row.get("stage") != "primary_v2"
            or row.get("evidence_status") != PRIMARY_EVIDENCE_STATUS
            or int(row.get("planned_work_units", -1)) <= 0
            or int(row.get("completed_work_units", -2))
            != int(row.get("planned_work_units", -1))
            for row in emitted
        )
    ):
        raise RevisionDetectionError(
            "Primary preprocessing lineage is not exactly three complete primary model shards."
        )
    return {
        "schema_version": PREPROCESSING_BINDING_SCHEMA,
        "preprocessing_manifest_path": str(manifest_path),
        "preprocessing_manifest_sha256": _sha256_file(manifest_path),
        "preprocessing_input_manifest_path": str(input_manifest_path),
        "preprocessing_input_manifest_sha256": input_manifest_hash,
        "detector_path": str(detector_path),
        "detector_sha256": detector_hash,
        "detector_size_bytes": detector_path.stat().st_size,
        "detector_row_count": expected_rows,
        "strict_complete": True,
        "primary_shard_count": 3,
        "primary_model_ids": sorted(map(str, contract["model_ids"])),
        "verified_input_file_count": len(verified_input_records),
        "verified_input_files_sha256": canonical_json_sha256(
            verified_input_records
        ),
    }


def _primary_detector_contract() -> dict:
    """Derive the exact 28-split corpus contract from frozen study configs."""

    primary = _read_json_object(PRIMARY_CONFIG, "primary config")
    prompts = _read_json_object(PROMPT_CONFIG, "prompt config")
    models = _read_json_object(MODEL_CONFIG, "model config")
    prompt_ids = [
        str(template["prompt_id"])
        for category in prompts.get("categories", [])
        for template in category.get("templates", [])
    ]
    model_ids = [str(model["model_id"]) for model in models.get("models", [])]
    codec_ids = [
        str(protocol["protocol_variant"])
        for protocol in primary.get("protocols", [])
    ]
    control_count = int(
        primary.get("expected_counts", {}).get("ordinary_control_texts", -1)
    )
    contract = {
        "schema_version": "rankcloak-revision-primary-detector-contract-v1",
        "input_scope": "primary_v2_complete_detector_corpus_only",
        "rows": int(2 * control_count),
        "payload_groups": int(primary.get("corpus", {}).get("payload_count", -1)),
        "positive_rows": control_count,
        "negative_rows": control_count,
        "prompt_template_ids": sorted(prompt_ids),
        "model_ids": sorted(model_ids),
        "codec_ids": sorted(codec_ids),
        "split_count": 28,
        "source_configs": {
            "primary.json": _sha256_file(PRIMARY_CONFIG),
            "prompts.json": _sha256_file(PROMPT_CONFIG),
            "models.json": _sha256_file(MODEL_CONFIG),
        },
    }
    if (
        len(set(prompt_ids)) != 18
        or len(set(model_ids)) != 3
        or len(set(codec_ids)) != 6
        or control_count != 7920
        or contract["payload_groups"] != 480
    ):
        raise RevisionDetectionError(
            "Frozen primary configs no longer imply the prespecified "
            "18-template/3-model/6-codec/480-payload detector design."
        )
    return contract


def _execution_identity_rows(metrics: object) -> list:
    rows = []
    if not isinstance(metrics, object) or getattr(metrics, "empty", True):
        return rows
    for record in metrics.to_dict(orient="records"):
        try:
            metadata = json.loads(str(record.get("implementation_metadata_json", "{}")))
        except json.JSONDecodeError:
            metadata = {}
        rows.append(
            {
                "split_id": str(record.get("split_id", "")),
                "detector_name": str(record.get("detector_name", "")),
                "requested_kind": str(record.get("requested_kind", "")),
                "implementation_kind": str(record.get("implementation_kind", "")),
                "implementation_status": str(record.get("implementation_status", "")),
                "seed": int(record.get("seed", 0)),
                "model_state_hash_algorithm": metadata.get(
                    "model_state_hash_algorithm"
                ),
                "model_state_sha256": metadata.get("model_state_sha256"),
                "model_artifact_set_sha256": metadata.get(
                    "model_artifact_set_sha256"
                ),
            }
        )
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate revision raw-text detectors with payload-grouped, "
            "leakage-checked splits."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV or JSONL detector data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        help=(
            "Primary preprocessing_output_manifest.json that declares --input; "
            "required outside --smoke."
        ),
    )
    parser.add_argument(
        "--confirmatory-plan",
        type=Path,
        default=DEFAULT_CONFIRMATORY_PLAN,
        help="Frozen confirmatory identity plan (validated and used outside --smoke).",
    )
    parser.add_argument(
        "--execution-policy",
        type=Path,
        default=DEFAULT_EXECUTION_POLICY,
        help=(
            "Pre-benchmark acceleration, equivalence, and next-fit ceiling policy; "
            "strictly required for confirmatory execution."
        ),
    )
    parser.add_argument(
        "--cuda-budget-gate",
        type=Path,
        default=DEFAULT_CUDA_BUDGET_GATE,
        help=(
            "Signed benchmark-derived post-reproducibility budget gate; "
            "required before the full confirmatory matrix."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Force neural requests to the labelled non-neural smoke fallback and cap "
            "bootstrap repetitions."
        ),
    )
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help=(
            "Legacy option retained for CLI compatibility; confirmatory execution rejects it."
        ),
    )
    parser.add_argument(
        "--accept-smoke-fallback",
        action="store_true",
        help="Legacy option retained for compatibility; confirmatory execution rejects it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known detector products in the selected output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse only identity/hash-valid per-fit checkpoints. Confirmatory "
            "--overwrite retries also resume valid sibling checkpoints automatically."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda:0"),
        default=None,
        help=(
            "Execution-only device override (or RANKCLOAK_DETECTOR_DEVICE); does "
            "not mutate the frozen config bytes."
        ),
    )
    parser.add_argument(
        "--gpu-uuid",
        default=None,
        help=(
            "Exact GPU UUID required with cuda:0 (or "
            "RANKCLOAK_DETECTOR_GPU_UUID)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Bounded fit workers (or RANKCLOAK_DETECTOR_WORKERS); only 1 is "
            "implemented safely."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Per-fit checkpoint root; defaults to the sibling <output-dir>.checkpoints.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        help="Atomic live status path; defaults to sibling <output-dir>.status.json.",
    )
    parser.add_argument(
        "--fit-permit-file",
        type=Path,
        help=(
            "One-use signed next-fit ceiling permit; defaults to sibling "
            "<output-dir>.fit_permit.json for confirmatory execution."
        ),
    )
    parser.add_argument(
        "--fit-permit-receipt-dir",
        type=Path,
        help=(
            "Durable signed consumed-permit receipts; defaults to "
            "<checkpoint-dir>/fit_permit_receipts."
        ),
    )
    parser.add_argument(
        "--benchmark-one-fit",
        action="store_true",
        help="Run only the first unfinished fit, checkpoint it, and stop at its boundary.",
    )
    parser.add_argument(
        "--benchmark-task-index",
        type=int,
        help=(
            "Zero-based frozen task index to benchmark and checkpoint; requires "
            "--benchmark-one-fit. Task 0 is TextCNN and task 1 is DeBERTa."
        ),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help="Optional atomic JSON record for --benchmark-one-fit outside final products.",
    )
    parser.add_argument(
        "--equivalence-role",
        choices=("cuda", "cuda_repeat"),
        help=(
            "Run/export one full frozen representative fit as a signed "
            "CUDA reproducibility artifact; never publishes final detector products."
        ),
    )
    parser.add_argument(
        "--equivalence-task-index",
        type=int,
        choices=(0, 1),
        help="Frozen representative task index (0 TextCNN, 1 DeBERTa).",
    )
    parser.add_argument(
        "--equivalence-artifact",
        type=Path,
        help="Atomic signed artifact for --equivalence-role.",
    )
    parser.add_argument("--equivalence-report-output", type=Path)
    parser.add_argument("--equivalence-cpu-artifact", type=Path)
    parser.add_argument("--equivalence-cuda-artifact", type=Path)
    parser.add_argument("--equivalence-cuda-repeat-artifact", type=Path)
    parser.add_argument(
        "--equivalence-required-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "Signed passing task0/task1 report required before normal production; "
            "repeat exactly twice for the full gate."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        equivalence_fit_mode = args.equivalence_role is not None
        equivalence_report_mode = args.equivalence_report_output is not None
        if equivalence_fit_mode and (
            args.equivalence_task_index is None
            or args.equivalence_artifact is None
        ):
            raise RevisionDetectionError(
                "--equivalence-role requires --equivalence-task-index and "
                "--equivalence-artifact."
            )
        if equivalence_report_mode and any(
            value is None
            for value in (
                args.equivalence_task_index,
                args.equivalence_cuda_artifact,
                args.equivalence_cuda_repeat_artifact,
            )
        ):
            raise RevisionDetectionError(
                "CUDA reproducibility report mode requires task index and both CUDA artifacts."
            )
        if equivalence_report_mode and args.equivalence_cpu_artifact is not None:
            raise RevisionDetectionError(
                "CPU artifacts are prohibited by the CUDA-only reproducibility gate."
            )
        if equivalence_fit_mode and equivalence_report_mode:
            raise RevisionDetectionError(
                "Equivalence fit and report modes are mutually exclusive."
            )
        if args.smoke and (equivalence_fit_mode or equivalence_report_mode):
            raise RevisionDetectionError(
                "Production equivalence modes are unavailable under --smoke."
            )
        if args.benchmark_task_index is not None and not args.benchmark_one_fit:
            raise RevisionDetectionError(
                "--benchmark-task-index requires --benchmark-one-fit."
            )
        config_path = args.config.resolve()
        input_path = args.input.resolve()
        output_dir = args.output_dir.resolve()
        checkpoint_dir = (
            args.checkpoint_dir.resolve()
            if args.checkpoint_dir is not None
            else output_dir.with_name(output_dir.name + ".checkpoints")
        )
        status_file = (
            args.status_file.resolve()
            if args.status_file is not None
            else output_dir.with_name(output_dir.name + ".status.json")
        )
        fit_permit_file = (
            args.fit_permit_file.resolve()
            if args.fit_permit_file is not None
            else output_dir.with_name(output_dir.name + ".fit_permit.json")
        )
        fit_permit_receipt_dir = (
            args.fit_permit_receipt_dir.resolve()
            if args.fit_permit_receipt_dir is not None
            else checkpoint_dir / "fit_permit_receipts"
        )
        execution_policy_binding = None
        if not args.smoke:
            execution_policy_binding = _verify_execution_policy(
                args.execution_policy.resolve()
            )
        policy_execution = (
            None
            if execution_policy_binding is None
            else execution_policy_binding["policy"]["execution"]
        )
        requested_device = _runtime_value(
            args.device, "RANKCLOAK_DETECTOR_DEVICE", None
        )
        if requested_device in (None, ""):
            device = (
                "cpu"
                if args.smoke or equivalence_report_mode
                else str(policy_execution["device"])
            )
        else:
            device = str(requested_device)
        default_gpu_uuid = (
            policy_execution["gpu_uuid"]
            if policy_execution is not None and device == "cuda:0"
            else None
        )
        raw_gpu_uuid = _runtime_value(
            args.gpu_uuid, "RANKCLOAK_DETECTOR_GPU_UUID", default_gpu_uuid
        )
        gpu_uuid = None if raw_gpu_uuid in (None, "") else str(raw_gpu_uuid)
        default_workers = (
            1
            if policy_execution is None
            else int(policy_execution["workers"])
        )
        try:
            workers = int(
                _runtime_value(
                    args.workers,
                    "RANKCLOAK_DETECTOR_WORKERS",
                    default_workers,
                )
            )
        except (TypeError, ValueError) as exc:
            raise RevisionDetectionError("--workers must be an integer.") from exc
        _bind_runtime_before_torch_import(
            device=device, gpu_uuid=gpu_uuid, workers=workers
        )
        config = load_detector_config(config_path)
        runtime_config = _runtime_config(config, device)
        confirmatory_plan = None
        detector_dataset_contract = None
        environment_binding = None
        equivalence_policy_identity = None
        required_equivalence_reports = []
        required_gpu_accounting_ledger = None
        required_cuda_budget_gate = None
        confirmatory_plan_path = args.confirmatory_plan.resolve()
        if not args.smoke:
            if args.allow_model_downloads:
                raise RevisionDetectionError(
                    "--allow-model-downloads is prohibited for confirmatory execution."
                )
            if args.accept_smoke_fallback:
                raise RevisionDetectionError(
                    "--accept-smoke-fallback is prohibited for confirmatory execution."
                )
            confirmatory_plan = _load_confirmatory_plan(
                confirmatory_plan_path, config_path
            )
            environment_binding = _verify_execution_environment()
            equivalence_policy_identity = _equivalence_policy_identity(
                execution_policy_binding, environment_binding
            )
            runtime_matches_policy = (
                device == policy_execution["device"]
                and gpu_uuid == policy_execution["gpu_uuid"]
                and workers == int(policy_execution["workers"])
            )
            if not equivalence_report_mode and not runtime_matches_policy:
                raise RevisionDetectionError(
                    "Confirmatory neural training is CUDA-only and must match the pinned policy."
                )
            if equivalence_report_mode:
                report = write_detector_cuda_reproducibility_report(
                    args.equivalence_report_output.resolve(),
                    cuda_artifact_path=args.equivalence_cuda_artifact.resolve(),
                    cuda_repeat_artifact_path=(
                        args.equivalence_cuda_repeat_artifact.resolve()
                    ),
                    equivalence_policy=execution_policy_binding["policy"][
                        "equivalence"
                    ],
                    policy_identity=equivalence_policy_identity,
                )
                if report["decision"].get("reproducible") is not True:
                    print(
                        "methodological halt: detector CUDA reproducibility failed; "
                        "signed report preserved at {}".format(
                            args.equivalence_report_output.resolve()
                        ),
                        file=sys.stderr,
                    )
                    return 4
                read_detector_cuda_reproducibility_report(
                    args.equivalence_report_output.resolve(),
                    expected_task_index=args.equivalence_task_index,
                    expected_policy_identity=equivalence_policy_identity,
                    expected_equivalence_policy=execution_policy_binding[
                        "policy"
                    ]["equivalence"],
                )
                print(json.dumps(report, sort_keys=True))
                return 0
            if not equivalence_fit_mode and not args.benchmark_one_fit:
                report_paths = (
                    [path.resolve() for path in args.equivalence_required_report]
                    if args.equivalence_required_report
                    else [path.resolve() for path in DEFAULT_EQUIVALENCE_REPORTS]
                )
                if len(report_paths) != 2:
                    raise RevisionDetectionError(
                        "Normal CUDA production requires exactly task0 and task1 "
                        "CUDA reproducibility reports."
                    )
                for task_index, report_path in enumerate(report_paths):
                    required_equivalence_reports.append(
                        read_detector_cuda_reproducibility_report(
                            report_path,
                            expected_task_index=task_index,
                            expected_policy_identity=equivalence_policy_identity,
                            expected_equivalence_policy=execution_policy_binding[
                                "policy"
                            ]["equivalence"],
                        )
                    )
                ledger = read_detector_gpu_accounting_ledger(
                    DEFAULT_GPU_ACCOUNTING_LEDGER
                )
                observed_sources = {
                    str(source["source_id"]): str(source["component"])
                    for source in ledger["sources"]
                }
                if observed_sources != EXPECTED_GPU_LEDGER_SOURCES:
                    raise RevisionDetectionError(
                        "Normal CUDA production requires the exact six finalized "
                        "benchmark/equivalence GPU ledger sources."
                    )
                required_gpu_accounting_ledger = {
                    "path": str(DEFAULT_GPU_ACCOUNTING_LEDGER.resolve()),
                    "sha256": _sha256_file(DEFAULT_GPU_ACCOUNTING_LEDGER),
                    "size_bytes": int(
                        DEFAULT_GPU_ACCOUNTING_LEDGER.stat().st_size
                    ),
                    "ledger_sha256": ledger["ledger_sha256"],
                    "sources_sha256": ledger["sources_sha256"],
                    "intervals_sha256": ledger["intervals_sha256"],
                    "cumulative_elapsed_seconds": ledger[
                        "cumulative_elapsed_seconds"
                    ],
                }
                budget_gate_path = args.cuda_budget_gate.resolve()
                budget_gate = read_gate(
                    budget_gate_path,
                    expected_stage="post_reproducibility_preproduction",
                )
                gate_inputs = budget_gate.get("inputs", {})
                gate_policy = gate_inputs.get("policy", {})
                gate_ledger = gate_inputs.get("gpu_ledger", {})
                if (
                    Path(str(gate_policy.get("path", ""))).resolve()
                    != Path(execution_policy_binding["path"]).resolve()
                    or gate_policy.get("sha256")
                    != execution_policy_binding["sha256"]
                    or gate_policy.get("policy_sha256")
                    != execution_policy_binding["policy_sha256"]
                    or Path(str(gate_ledger.get("path", ""))).resolve()
                    != DEFAULT_GPU_ACCOUNTING_LEDGER.resolve()
                    or gate_ledger.get("sha256")
                    != required_gpu_accounting_ledger["sha256"]
                    or gate_ledger.get("ledger_sha256")
                    != required_gpu_accounting_ledger["ledger_sha256"]
                ):
                    raise RevisionDetectionError(
                        "CUDA budget gate does not bind the active policy and ledger."
                    )
                required_cuda_budget_gate = {
                    "path": str(budget_gate_path),
                    "sha256": _sha256_file(budget_gate_path),
                    "size_bytes": int(budget_gate_path.stat().st_size),
                    "gate_sha256": budget_gate["gate_sha256"],
                    "gate_stage": budget_gate["gate_stage"],
                    "projection_sha256": budget_gate["projection_sha256"],
                    "projected_cumulative_gpu_hours": budget_gate[
                        "projection"
                    ]["projected_cumulative_gpu_hours"],
                }
            detector_dataset_contract = _primary_detector_contract()
            if args.preprocessing_manifest is None:
                raise RevisionDetectionError(
                    "Confirmatory execution requires --preprocessing-manifest."
                )
            detector_dataset_contract["preprocessing_binding"] = (
                _verify_primary_preprocessing_detector_input(
                    args.input,
                    args.preprocessing_manifest,
                    detector_dataset_contract,
                )
            )
        raw_frame = read_detector_frame(input_path)
        _prepare_output_directory(output_dir, overwrite=bool(args.overwrite))
        prepared = prepare_revision_detector_suite(
            raw_frame,
            runtime_config,
            smoke=bool(args.smoke),
            allow_model_downloads=bool(args.allow_model_downloads),
            confirmatory_dataset_contract=detector_dataset_contract,
        )
        lineage = {
            "input_path": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "preprocessing_manifest_path": (
                None
                if args.preprocessing_manifest is None
                else str(args.preprocessing_manifest.resolve())
            ),
            "preprocessing_manifest_sha256": (
                None
                if args.preprocessing_manifest is None
                else _sha256_file(args.preprocessing_manifest.resolve())
            ),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "runtime_config_sha256": canonical_json_sha256(runtime_config),
            "confirmatory_plan_path": (
                None if args.smoke else str(confirmatory_plan_path)
            ),
            "confirmatory_plan_sha256": (
                None if args.smoke else _sha256_file(confirmatory_plan_path)
            ),
            "preprocessing_binding": (
                None
                if detector_dataset_contract is None
                else detector_dataset_contract.get("preprocessing_binding")
            ),
            "execution_policy": execution_policy_binding,
            "environment_binding": environment_binding,
            "required_equivalence_reports": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "report_sha256": report["report_sha256"],
                }
                for path, report in zip(
                    (
                        [path.resolve() for path in args.equivalence_required_report]
                        if args.equivalence_required_report
                        else [path.resolve() for path in DEFAULT_EQUIVALENCE_REPORTS]
                    ),
                    required_equivalence_reports,
                )
            ],
            "pre_final_gpu_accounting_ledger_path": (
                None
                if args.smoke or equivalence_fit_mode
                else str(DEFAULT_GPU_ACCOUNTING_LEDGER.resolve())
            ),
            "pre_final_gpu_accounting_ledger": required_gpu_accounting_ledger,
            "cuda_budget_gate": required_cuda_budget_gate,
        }
        source_paths = [
            PROJECT_ROOT / "rankcloak" / "revision_detection.py",
            PROJECT_ROOT / "rankcloak" / "revision_detector_execution.py",
            Path(__file__).resolve(),
        ]
        source_files = [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in source_paths
        ]
        execution_context = DetectorExecutionContext(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            status_file=status_file,
            device=device,
            gpu_uuid=gpu_uuid,
            workers=workers,
            lineage=lineage,
            source={
                "files": source_files,
                "files_sha256": canonical_json_sha256(source_files),
            },
            # A loaded confirmatory supervisor retries the exact legacy argv with
            # --overwrite. Such retries must preserve valid sibling checkpoints.
            resume=bool(args.resume or (args.overwrite and not args.smoke)),
            execution_policy=(
                None
                if execution_policy_binding is None
                else execution_policy_binding["policy"]
            ),
            execution_policy_path=(
                None
                if execution_policy_binding is None
                else Path(execution_policy_binding["path"])
            ),
            execution_policy_sha256=(
                None
                if execution_policy_binding is None
                else str(execution_policy_binding["sha256"])
            ),
            fit_permit_file=fit_permit_file,
            fit_permit_receipt_dir=fit_permit_receipt_dir,
            require_fit_permit=(
                not args.smoke
            ),
        )
        benchmark_started = time.monotonic()
        selected_task_index = (
            args.equivalence_task_index
            if equivalence_fit_mode
            else args.benchmark_task_index
        )
        outcome = execute_checkpointed_detector_suite(
            prepared,
            execution_context,
            stop_after_new_fits=(
                1 if (args.benchmark_one_fit or equivalence_fit_mode) else None
            ),
            benchmark_task_index=selected_task_index,
        )
        if equivalence_fit_mode:
            status = verify_status_file(status_file)
            if (
                outcome.last_completed_checkpoint is None
                or int(outcome.last_completed_checkpoint.get("task_ordinal", -1))
                != int(args.equivalence_task_index)
            ):
                raise RevisionDetectionError(
                    "Equivalence fit did not commit the selected frozen task."
                )
            provenance = {
                    "device": device,
                    "gpu_uuid": gpu_uuid,
                    "workers": int(workers),
                    "peak_rss_bytes": int(status["peak_rss_bytes"]),
                    "peak_vram_bytes": int(status["peak_vram_bytes"]),
                    "gpu_accounting": "supervisor_closes_after_exit",
                    "status_file": str(status_file),
                    "status_sha256": status["status_sha256"],
                    "execution_policy_path": execution_policy_binding["path"],
                    "execution_policy_sha256": execution_policy_binding["sha256"],
                    "policy_identity": equivalence_policy_identity,
                    "equivalence_role": args.equivalence_role,
            }
            payload = build_validated_checkpoint_equivalence_payload(
                prepared,
                execution_context,
                task_index=args.equivalence_task_index,
                role=args.equivalence_role,
                provenance=provenance,
            )
            fit_directory = (
                checkpoint_dir
                / "fits"
                / "{:04d}".format(int(args.equivalence_task_index))
            )
            candidate_files = {
                name: {
                    "path": str((fit_directory / name).resolve()),
                    "sha256": _sha256_file(fit_directory / name),
                    "size_bytes": int((fit_directory / name).stat().st_size),
                }
                for name in ("metric.json", "predictions.json", "manifest.json")
            }
            candidate_path, _terminal_receipt_path = detector_finalization_paths(
                checkpoint_dir,
                kind="equivalence_artifact",
                requested_output_path=args.equivalence_artifact.resolve(),
                task_index=args.equivalence_task_index,
                role=args.equivalence_role,
            )
            candidate = write_detector_finalization_candidate(
                candidate_path,
                kind="equivalence_artifact",
                run_identity_sha256=outcome.run_identity_sha256,
                payload=payload,
                output_files=candidate_files,
                requested_output_path=args.equivalence_artifact.resolve(),
            )
            mark_detector_awaiting_supervisor_finalization(
                status_file,
                run_identity_sha256=outcome.run_identity_sha256,
                candidate_path=candidate_path,
            )
            print(json.dumps(candidate, sort_keys=True))
            return 0
        if args.benchmark_one_fit:
            benchmark_seconds = max(0.0, time.monotonic() - benchmark_started)
            benchmark_status = verify_status_file(status_file)
            benchmark_index = (
                args.benchmark_task_index
                if args.benchmark_task_index is not None
                else (
                    None
                    if outcome.last_completed_checkpoint is None
                    else int(outcome.last_completed_checkpoint["task_ordinal"])
                )
            )
            benchmark_task = (
                None
                if benchmark_index is None
                else json.loads(
                    (checkpoint_dir / "execution_plan.json").read_text(
                        encoding="utf-8"
                    )
                )["tasks"][int(benchmark_index)]
            )
            actual_fit_seconds = (
                None
                if benchmark_index is None
                else float(
                    json.loads(
                        (
                            checkpoint_dir
                            / "fits"
                            / "{:04d}".format(int(benchmark_index))
                            / "manifest.json"
                        ).read_text(encoding="utf-8")
                    )["elapsed_seconds"]
                )
            )
            metric_record = (
                None
                if benchmark_index is None
                else _read_single_checkpoint_metric(
                    checkpoint_dir
                    / "fits"
                    / "{:04d}".format(int(benchmark_index))
                    / "metric.json"
                )
            )
            try:
                implementation_metadata = (
                    {}
                    if metric_record is None
                    else json.loads(metric_record["implementation_metadata_json"])
                )
            except json.JSONDecodeError as exc:
                raise RevisionDetectionError(
                    "Detector fit metric implementation metadata is invalid JSON."
                ) from exc
            if not isinstance(implementation_metadata, dict):
                raise RevisionDetectionError(
                    "Detector fit metric implementation metadata is not an object."
                )
            phase_timings = implementation_metadata.get(
                "phase_timings_seconds", {}
            )
            checkpoint_seconds = float(sum(outcome.checkpoint_durations_seconds))
            model_total_seconds = float(phase_timings.get("total", 0.0))
            benchmark_record = {
                "schema_version": "rankcloak-revision-detector-benchmark-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": device,
                "gpu_uuid": gpu_uuid,
                "workers": workers,
                "completed_fit_count": outcome.completed_fit_count,
                "total_fit_count": outcome.total_fit_count,
                "resumed_fit_count": outcome.resumed_fit_count,
                "new_fit_count": int(
                    outcome.completed_fit_count - outcome.resumed_fit_count
                ),
                "benchmark_task_index": args.benchmark_task_index,
                "benchmark_task_identity": benchmark_task,
                "wall_seconds": benchmark_seconds,
                "fit_durations_seconds": outcome.fit_durations_seconds,
                "fit_elapsed_seconds": actual_fit_seconds,
                "phase_timings_seconds": phase_timings,
                "fit_non_model_analysis_seconds": (
                    None
                    if actual_fit_seconds is None
                    else max(0.0, actual_fit_seconds - model_total_seconds)
                ),
                "checkpoint_durations_seconds": outcome.checkpoint_durations_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "invocation_overhead_seconds": (
                    None
                    if actual_fit_seconds is None
                    else max(
                        0.0,
                        benchmark_seconds - actual_fit_seconds - checkpoint_seconds,
                    )
                ),
                "peak_rss_bytes": benchmark_status["peak_rss_bytes"],
                "peak_vram_bytes": benchmark_status["peak_vram_bytes"],
                "gpu_accounting": (
                    benchmark_status["gpu_accounting"]
                    if device == "cpu"
                    else "supervisor_closes_after_exit"
                ),
                "run_identity_sha256": outcome.run_identity_sha256,
                "execution_plan_sha256": outcome.plan_sha256,
                "last_completed_checkpoint": outcome.last_completed_checkpoint,
                "status_file": str(status_file),
                "checkpoint_dir": str(checkpoint_dir),
            }
            benchmark_output = (
                args.benchmark_output.resolve()
                if args.benchmark_output is not None
                else checkpoint_dir / "benchmark.json"
            )
            if device == "cuda:0":
                fit_directory = (
                    checkpoint_dir / "fits" / "{:04d}".format(int(benchmark_index))
                )
                candidate_files = {
                    name: {
                        "path": str((fit_directory / name).resolve()),
                        "sha256": _sha256_file(fit_directory / name),
                        "size_bytes": int((fit_directory / name).stat().st_size),
                    }
                    for name in ("metric.json", "predictions.json", "manifest.json")
                }
                candidate_path, _terminal_receipt_path = detector_finalization_paths(
                    checkpoint_dir,
                    kind="benchmark_artifact",
                    requested_output_path=benchmark_output,
                    task_index=benchmark_index,
                    role="benchmark",
                )
                candidate = write_detector_finalization_candidate(
                    candidate_path,
                    kind="benchmark_artifact",
                    run_identity_sha256=outcome.run_identity_sha256,
                    payload=benchmark_record,
                    output_files=candidate_files,
                    requested_output_path=benchmark_output,
                )
                mark_detector_awaiting_supervisor_finalization(
                    status_file,
                    run_identity_sha256=outcome.run_identity_sha256,
                    candidate_path=candidate_path,
                )
                print(json.dumps(candidate, sort_keys=True))
                return 0
            benchmark_record["benchmark_sha256"] = canonical_json_sha256(
                benchmark_record
            )
            _write_json(benchmark_output, benchmark_record)
            print(json.dumps(benchmark_record, sort_keys=True))
            return 0
        if outcome.stopped_at_fit_boundary:
            print(
                "detector execution stopped safely at a fit boundary after "
                "{}/{} fits".format(
                    outcome.completed_fit_count, outcome.total_fit_count
                ),
                file=sys.stderr,
            )
            return 75
        if outcome.result is None:
            raise RevisionDetectionError(
                "Detector executor completed without an aggregate result."
            )
        result = outcome.result
    except RevisionDetectionError as exc:
        print("revision detector error: {}".format(exc), file=sys.stderr)
        return 2

    _write_csv(output_dir / "detector_metrics.csv", result.metrics)
    _write_csv(output_dir / "detector_predictions.csv", result.predictions)
    dataset_manifest_columns = [
        "row_id",
        "text_sha256",
        "label",
        "payload_group_id",
        "prompt_template_id",
        "model_id",
        "codec_id",
    ]
    _write_csv(
        output_dir / "detector_dataset_manifest.csv",
        result.normalized_frame[dataset_manifest_columns],
    )
    _write_json(
        output_dir / "detector_split_manifest.json",
        {
            "schema_version": "rankcloak-revision-detector-splits-v2",
            "splits": split_manifest_rows(result.normalized_frame, result.splits),
            "skipped_splits": [item.__dict__ for item in result.skipped_splits],
            "split_contract": result.run_metadata.get("split_contract"),
        },
    )
    _write_json(output_dir / "detector_failures.json", result.failures)
    fallback_rows = 0
    if not result.metrics.empty and "implementation_status" in result.metrics:
        fallback_rows = int(result.metrics["implementation_status"].eq("smoke_fallback").sum())
    product_files = {
        name: {
            "sha256": _sha256_file(output_dir / name),
            "size_bytes": int((output_dir / name).stat().st_size),
        }
        for name in OUTPUT_FILENAMES
        if name != "detector_run_manifest.json"
    }
    execution_identities = _execution_identity_rows(result.metrics)
    final_gpu_accounting = None if device == "cpu" else "supervisor_closes_after_exit"
    final_execution_completed_at = (
        outcome.completed_at_utc
        if device == "cpu"
        else "supervisor_closes_after_exit"
    )
    manifest = {
        "schema_version": "rankcloak-revision-detector-run-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "smoke" if args.smoke else "confirmatory",
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "preprocessing_manifest_path": (
            None
            if args.preprocessing_manifest is None
            else str(args.preprocessing_manifest.resolve())
        ),
        "preprocessing_manifest_sha256": (
            None
            if args.preprocessing_manifest is None
            else _sha256_file(args.preprocessing_manifest.resolve())
        ),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "confirmatory_plan_path": (
            None if args.smoke else str(confirmatory_plan_path)
        ),
        "confirmatory_plan_sha256": (
            None if args.smoke else _sha256_file(confirmatory_plan_path)
        ),
        "confirmatory_plan_schema_version": (
            None if confirmatory_plan is None else confirmatory_plan["schema_version"]
        ),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "status_file": str(status_file),
        "fit_permit_file": str(fit_permit_file),
        "fit_permit_receipt_dir": str(fit_permit_receipt_dir),
        "execution_policy_path": (
            None
            if execution_policy_binding is None
            else execution_policy_binding["path"]
        ),
        "execution_policy_sha256": (
            None
            if execution_policy_binding is None
            else execution_policy_binding["sha256"]
        ),
        "execution_policy_content_sha256": (
            None
            if execution_policy_binding is None
            else execution_policy_binding["policy_sha256"]
        ),
        "environment_binding": environment_binding,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "workers": int(workers),
        "completed_fit_count": int(outcome.completed_fit_count),
        "total_fit_count": int(outcome.total_fit_count),
        "resumed_fit_count": int(outcome.resumed_fit_count),
        "recovered_errors": outcome.recovered_errors,
        "execution_started_at_utc": outcome.started_at_utc,
        "execution_completed_at_utc": final_execution_completed_at,
        "fit_durations_seconds": outcome.fit_durations_seconds,
        "checkpoint_cumulative_fit_seconds": float(
            sum(outcome.fit_durations_seconds)
        ),
        "run_identity": outcome.run_identity,
        "run_identity_sha256": outcome.run_identity_sha256,
        "execution_plan_sha256": outcome.plan_sha256,
        "last_completed_checkpoint": outcome.last_completed_checkpoint,
        "gpu_accounting": final_gpu_accounting,
        "required_equivalence_reports": lineage[
            "required_equivalence_reports"
        ],
        "pre_final_gpu_accounting_ledger_path": lineage[
            "pre_final_gpu_accounting_ledger_path"
        ],
        "pre_final_gpu_accounting_ledger": lineage[
            "pre_final_gpu_accounting_ledger"
        ],
        "cuda_budget_gate": lineage["cuda_budget_gate"],
        "smoke": bool(args.smoke),
        "allow_model_downloads_cli": bool(args.allow_model_downloads),
        "accept_smoke_fallback": bool(args.accept_smoke_fallback),
        "raw_rows": int(len(raw_frame)),
        "normalized_rows": int(len(result.normalized_frame)),
        "split_count": int(len(result.splits)),
        "skipped_split_count": int(len(result.skipped_splits)),
        "metric_rows": int(len(result.metrics)),
        "prediction_rows": int(len(result.predictions)),
        "failure_count": int(len(result.failures)),
        "smoke_fallback_metric_rows": fallback_rows,
        "confirmatory_complete": result.run_metadata.get("confirmatory_complete"),
        "detector_run_metadata": result.run_metadata,
        "detector_execution_identities": execution_identities,
        "output_files": product_files,
        "git": _git_metadata(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }
    final_manifest_path = output_dir / "detector_run_manifest.json"
    candidate_files = {
        name: {
            "path": str((output_dir / name).resolve()),
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }
        for name, identity in product_files.items()
    }
    if device == "cuda:0":
        candidate_path, _terminal_receipt_path = detector_finalization_paths(
            checkpoint_dir,
            kind="detector_run_manifest",
            requested_output_path=final_manifest_path,
            role="suite",
        )
        candidate = write_detector_finalization_candidate(
            candidate_path,
            kind="detector_run_manifest",
            run_identity_sha256=outcome.run_identity_sha256,
            payload=manifest,
            output_files=candidate_files,
            requested_output_path=final_manifest_path,
        )
        mark_detector_awaiting_supervisor_finalization(
            status_file,
            run_identity_sha256=outcome.run_identity_sha256,
            candidate_path=candidate_path,
        )
        print(json.dumps(candidate, sort_keys=True))
    else:
        _write_json(final_manifest_path, manifest)
    print(
        "wrote {} metric rows and {} predictions across {} splits to {}".format(
            len(result.metrics), len(result.predictions), len(result.splits), output_dir
        )
    )
    if result.failures:
        print(
            "{} detector/split executions failed; see detector_failures.json".format(
                len(result.failures)
            ),
            file=sys.stderr,
        )
        return 3
    if fallback_rows and not (args.smoke or args.accept_smoke_fallback):
        print(
            "{} metric rows are non-neural smoke fallbacks; rerun with neural dependencies "
            "or explicitly pass --accept-smoke-fallback.".format(fallback_rows),
            file=sys.stderr,
        )
        return 3
    if not args.smoke and not bool(result.run_metadata.get("confirmatory_complete")):
        print(
            "confirmatory detector execution is incomplete; the run is invalid.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
