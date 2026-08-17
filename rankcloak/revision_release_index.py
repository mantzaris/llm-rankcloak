"""Legacy compatibility validator for pre-final confirmatory indexes.

This module preserves fail-closed validation for the earlier confirmatory-index
schema.  It is not the active code-and-data release recipe; current archives use
``rankcloak.revision_release`` and the sealed final experiment package.  The
legacy validator performs no experiment, model load, network request, DOI
action, or publication action, and manuscript preparation is not part of its
current computational action map.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .revision_artifacts import canonical_json_sha256, file_sha256
from .revision_progress import RevisionProgressError, verify_progress_snapshot


INDEX_SCHEMA = "rankcloak-confirmatory-release-index-v1"
INDEX_TYPE = "confirmatory_release_artifact_index"
PROTOCOL_REVISION = "payload_fidelity_v2"
RESULT_REVISION = "payload_aware_result_v2"
AUTHORIZED_PROJECTION_SHA256 = (
    "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
)
DEFAULT_INDEX = Path("results/revision_v1/confirmatory_release_index_v1.json")
MODEL_WEIGHT_SUFFIXES = {
    ".gguf", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".h5"
}
CACHE_COMPONENTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
CHECKPOINT_EXCLUDED_COMPONENTS = {
    "recovered_orphan_fit_files",
    "recovered_fit_permits",
}
CHECKPOINT_EXCLUDED_NAMES = {".execution.lock"}
FORBIDDEN_RESULT_PATH_TOKENS = {"smoke", "smoke_v2", "smoke_v3", "invalidated", "superseded", "fixture"}
VALIDATOR_SOURCE_PATHS = (
    "rankcloak/revision_release_index.py",
    "rankcloak/revision_artifacts.py",
    "rankcloak/revision_progress.py",
    "rankcloak/revision_detection.py",
    "rankcloak/revision_detector_execution.py",
    "scripts/run_revision_detectors.py",
    "scripts/supervise_confirmatory_v2.py",
    "operations/confirmatory_v2/downstream_commands.json",
    "operations/confirmatory_v2/detector_acceleration_policy_v1.json",
)
VALIDATOR_CANDIDATE_PATHS = {
    "rankcloak/revision_release_index.py": "source/rankcloak/revision_release_index.py",
    "rankcloak/revision_artifacts.py": "source/rankcloak/revision_artifacts.py",
    "rankcloak/revision_progress.py": "source/rankcloak/revision_progress.py",
    "rankcloak/revision_detection.py": "source/rankcloak/revision_detection.py",
    "rankcloak/revision_detector_execution.py": "source/rankcloak/revision_detector_execution.py",
    "scripts/run_revision_detectors.py": "source/scripts/run_revision_detectors.py",
    "scripts/supervise_confirmatory_v2.py": "source/scripts/supervise_confirmatory_v2.py",
    "operations/confirmatory_v2/downstream_commands.json": "source/operations/confirmatory_v2/downstream_commands.json",
    "operations/confirmatory_v2/detector_acceleration_policy_v1.json": "source/operations/confirmatory_v2/detector_acceleration_policy_v1.json",
}
INDEX_FIELDS = {
    "schema_version", "manifest_type", "status",
    "source_project_root",
    "protocol_contract_revision", "result_schema_revision",
    "authorized_projection_sha256", "network_access_used",
    "external_publication_performed", "doi_minted_or_reserved",
    "model_weights_included", "invalidated_or_superseded_outputs_included",
    "verification", "validator_sources", "validator_sources_sha256",
    "artifacts", "artifacts_sha256", "manifest_sha256",
}
ARTIFACT_FIELDS = {
    "group", "source", "destination", "evidence_role", "completion_kind",
    "file_count", "files", "files_sha256",
}
LEGACY_EXPECTED_ACTION_IDS = (
    "preprocess_primary_v2",
    "preprocess_ablation_v2",
    "preprocess_multilingual_v2",
    "preprocess_robustness_v2",
    "primary_evaluator_join",
    "detector",
    "statistics",
    "mixed_models_r",
    "theory",
    "reports",
    "figures",
)
LEGACY_EXPECTED_ACTION_KINDS = (
    "preprocess_v2", "preprocess_v2", "preprocess_v2", "preprocess_v2",
    "evaluator_join_v1", "detector_v2", "statistics_v1", "mixed_models_v1",
    "theory_v1", "report_v1", "figures_v1",
)

# (group, source, destination, evidence role, completion kind)
LEGACY_CONFIRMATORY_ARTIFACT_SPECS = (
    ("configs", "configs/revision_v1", "configs/revision_v1", "supporting_methodological_material", "detector_v2"),
    ("configs", "analysis/revision_v1/detector_confirmatory_plan.json", "configs/revision_v1/detector_confirmatory_plan.json", "supporting_methodological_material", "detector_v2"),
    ("raw_results", "results/revision_v1/compute_projection_165h_v2.json", "results/validation/compute_projection_165h_v2.json", "exploratory_compute_gate_not_for_confirmatory_pooling", "compute_gate_v2"),
    ("raw_results", "results/revision_v1/supervisor/detector_live_readonly_audit_20260813T0145Z.json", "results/operations/detector_live_readonly_audit_20260813T0145Z.json", "supporting_methodological_material", "detector_audit_v1"),
    ("raw_results", "results/revision_v1/primary_v2", "results/confirmatory/primary_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/ablation_v2", "results/confirmatory/ablation_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/multilingual_v2", "results/confirmatory/multilingual_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/robustness_v2", "results/confirmatory/robustness_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/heldout_evaluator/primary_v2", "results/confirmatory/heldout_evaluator/primary_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/heldout_evaluator/ablation_v2", "results/confirmatory/heldout_evaluator/ablation_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/heldout_evaluator/multilingual_v2", "results/confirmatory/heldout_evaluator/multilingual_v2", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/heldout_evaluator/upstream_dependent_unavailability_v1.json", "results/confirmatory/provenance/upstream_dependent_unavailability_v1.json", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/final_progress_snapshot_v1.json", "results/confirmatory/provenance/final_progress_snapshot_v1.json", "confirmatory_scientific_evidence", "final_progress_v1"),
    ("raw_results", "results/revision_v1/neural_detector/confirmatory_v2", "results/confirmatory/neural_detector", "confirmatory_scientific_evidence", "detector_v2"),
    ("raw_results", "results/revision_v1/neural_detector/confirmatory_v2.checkpoints", "results/confirmatory/neural_detector/checkpoints", "confirmatory_scientific_evidence", "detector_checkpoint_v1"),
    ("raw_results", "results/revision_v1/neural_detector/confirmatory_v2.status.json", "results/confirmatory/neural_detector/status.json", "confirmatory_scientific_evidence", "detector_checkpoint_v1"),
    ("raw_results", "results/revision_v1/detector_equivalence_v1", "results/confirmatory/neural_detector/equivalence_v1", "confirmatory_scientific_evidence", "detector_equivalence_v1"),
    ("raw_results", "results/revision_v1/supervisor/detector_benchmark_task_0_cuda.json", "results/confirmatory/neural_detector/benchmarks/task_0_cuda.json", "confirmatory_scientific_evidence", "detector_benchmark_v1"),
    ("raw_results", "results/revision_v1/supervisor/detector_benchmark_task_1_cuda.json", "results/confirmatory/neural_detector/benchmarks/task_1_cuda.json", "confirmatory_scientific_evidence", "detector_benchmark_v1"),
    ("raw_results", "results/revision_v1/supervisor/confirmatory_v2_events.jsonl", "results/operations/confirmatory_v2_events.jsonl", "supporting_methodological_material", "confirmatory_event_log_v1"),
    ("processed_results", "results/revision_v1/analysis_inputs/primary_v2", "results/processed/primary_v2", "confirmatory_scientific_evidence", "preprocess_v2"),
    ("processed_results", "results/revision_v1/analysis_inputs/ablation_v2", "results/processed/ablation_v2", "confirmatory_scientific_evidence", "preprocess_v2"),
    ("processed_results", "results/revision_v1/analysis_inputs/multilingual_v2", "results/processed/multilingual_v2", "confirmatory_scientific_evidence", "preprocess_v2"),
    ("processed_results", "results/revision_v1/analysis_inputs/robustness_v2", "results/processed/robustness_v2", "confirmatory_scientific_evidence", "preprocess_v2"),
    ("processed_results", "results/revision_v1/analysis_inputs/primary_heldout_join_v2", "results/processed/primary_heldout_join_v2", "confirmatory_scientific_evidence", "evaluator_join_v1"),
    ("statistics_outputs", "results/revision_v1/analysis/statistics_v2", "analysis/statistics_v2", "confirmatory_scientific_evidence", "statistics_v1"),
    ("statistics_outputs", "results/revision_v1/analysis/mixed_models_v2", "analysis/mixed_models_v2", "confirmatory_scientific_evidence", "mixed_models_v1"),
    ("statistics_outputs", "results/revision_v1/theory/confirmatory_v2", "analysis/theory_confirmatory_v2", "confirmatory_scientific_evidence", "theory_v1"),
    ("figure_table_outputs", "results/revision_v1/reports/confirmatory_v2", "reporting/confirmatory_v2", "confirmatory_scientific_evidence", "report_v1"),
    ("figure_table_outputs", "results/revision_v1/reports/confirmatory_v2_figures", "reporting/confirmatory_v2_figures", "confirmatory_scientific_evidence", "figures_v1"),
    ("environment_inputs", "environment/revision_v1", "environment/revision_v1", "environment_reproduction_input", "detector_environment_v1"),
)
CONFIRMATORY_FILE_SOURCES = frozenset({
    "analysis/revision_v1/detector_confirmatory_plan.json",
    "results/revision_v1/compute_projection_165h_v2.json",
    "results/revision_v1/heldout_evaluator/upstream_dependent_unavailability_v1.json",
    "results/revision_v1/final_progress_snapshot_v1.json",
    "results/revision_v1/neural_detector/confirmatory_v2.status.json",
    "results/revision_v1/supervisor/detector_benchmark_task_0_cuda.json",
    "results/revision_v1/supervisor/detector_benchmark_task_1_cuda.json",
    "results/revision_v1/supervisor/detector_live_readonly_audit_20260813T0145Z.json",
    "results/revision_v1/supervisor/confirmatory_v2_events.jsonl",
})

DETECTOR_SEMANTIC_SOURCES = frozenset({
    "results/revision_v1/neural_detector/confirmatory_v2",
    "results/revision_v1/neural_detector/confirmatory_v2.checkpoints",
    "results/revision_v1/neural_detector/confirmatory_v2.status.json",
    "results/revision_v1/detector_equivalence_v1",
    "results/revision_v1/supervisor/detector_benchmark_task_0_cuda.json",
    "results/revision_v1/supervisor/detector_benchmark_task_1_cuda.json",
})
DETECTOR_SIGNED_SCHEMA_FIELDS = {
    "rankcloak-revision-detector-acceleration-policy-v1": "policy_sha256",
    "rankcloak-revision-detector-fit-checkpoint-v1": "manifest_sha256",
    "rankcloak-revision-detector-status-v1": "status_sha256",
    "rankcloak-revision-detector-fit-permit-receipt-v1": "receipt_sha256",
    "rankcloak-revision-detector-equivalence-fit-artifact-v1": "artifact_sha256",
    "rankcloak-revision-detector-device-equivalence-v1": "report_sha256",
    "rankcloak-revision-detector-finalization-candidate-v1": "candidate_sha256",
    "rankcloak-revision-detector-terminal-receipt-v1": "terminal_receipt_sha256",
    "rankcloak-revision-detector-gpu-accounting-ledger-v1": "ledger_sha256",
    "rankcloak-revision-detector-gpu-ledger-incorporation-v1": "incorporation_sha256",
    "rankcloak-revision-detector-benchmark-v1": "benchmark_sha256",
    "rankcloak-revision-detector-run-v2": "manifest_sha256",
}


class ConfirmatoryReleaseIndexError(RuntimeError):
    """Raised when confirmatory release bytes are incomplete or unverified."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"


def _safe_regular(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ConfirmatoryReleaseIndexError("{} must be a regular non-symlink file".format(label))
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ConfirmatoryReleaseIndexError("{} escapes the project root".format(label)) from exc
    return resolved


def _file_record(path: Path, root: Path) -> dict[str, object]:
    resolved = _safe_regular(path, root, "release artifact")
    if resolved.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
        raise ConfirmatoryReleaseIndexError("model weights are forbidden: {}".format(resolved))
    relative = resolved.relative_to(root.resolve())
    if set(part.lower() for part in relative.parts) & FORBIDDEN_RESULT_PATH_TOKENS:
        raise ConfirmatoryReleaseIndexError(
            "superseded, invalid, smoke, or fixture bytes are forbidden: {}".format(relative)
        )
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ConfirmatoryReleaseIndexError(
            "release artifact changed while being hashed: {}".format(relative)
        )
    return {
        "path": relative.as_posix(),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def confirmatory_artifact_path_policy(source_text: str, relative: Path) -> str:
    """Classify safe final bytes and reject active/temporary checkpoint bytes."""

    lowered = tuple(part.lower() for part in relative.parts)
    name = lowered[-1] if lowered else ""
    if (
        name in CHECKPOINT_EXCLUDED_NAMES
        or set(lowered) & (CACHE_COMPONENTS | CHECKPOINT_EXCLUDED_COMPONENTS)
    ):
        return "exclude_checkpoint_transient"
    if name.startswith(".tmp-") or name.endswith(".fit_permit.json"):
        raise ConfirmatoryReleaseIndexError(
            "active or temporary checkpoint byte is forbidden: {}".format(relative)
        )
    return "include"


def _verify_terminal_event_log(path: Path) -> None:
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ConfirmatoryReleaseIndexError(
            "confirmatory event log changed while being read"
        )
    lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
    try:
        events = [json.loads(line) for line in lines]
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryReleaseIndexError(
            "confirmatory event log is missing a valid terminal event"
        ) from exc
    if not events or not all(isinstance(event, dict) for event in events):
        raise ConfirmatoryReleaseIndexError(
            "confirmatory event log contains a malformed event"
        )
    terminal = events[-1]
    if terminal.get("event") != (
        "confirmatory_post_primary_complete"
    ) or any(
        event.get("event") == "confirmatory_post_primary_complete"
        for event in events[:-1]
    ):
        raise ConfirmatoryReleaseIndexError(
            "confirmatory event log lacks the terminal completion event"
        )


def _artifact_files(root: Path, source_text: str) -> list[dict[str, object]]:
    source = root / source_text
    if source.is_symlink() or not source.exists():
        raise ConfirmatoryReleaseIndexError("confirmatory source is missing or unsafe: {}".format(source_text))
    expected_file = source_text in CONFIRMATORY_FILE_SOURCES
    if source.is_file() is not expected_file or source.is_dir() is expected_file:
        raise ConfirmatoryReleaseIndexError(
            "confirmatory source kind differs from frozen map: {}".format(source_text)
        )
    candidates = [source] if source.is_file() else sorted(source.rglob("*"), key=lambda p: p.as_posix())
    rows: list[dict[str, object]] = []
    for path in candidates:
        relative = path.relative_to(root)
        nested = Path(path.name) if source.is_file() else path.relative_to(source)
        if path.is_symlink():
            raise ConfirmatoryReleaseIndexError("symlink in confirmatory source: {}".format(relative))
        policy = confirmatory_artifact_path_policy(source_text, nested)
        if policy.startswith("exclude_"):
            continue
        if path.is_dir():
            continue
        rows.append(_file_record(path, root))
    if not rows:
        raise ConfirmatoryReleaseIndexError("confirmatory source is empty: {}".format(source_text))
    if source_text == "results/revision_v1/supervisor/confirmatory_v2_events.jsonl":
        _verify_terminal_event_log(source)
    return rows


def _load_supervisor(project_root: Path):
    path = project_root / "scripts/supervise_confirmatory_v2.py"
    _safe_regular(path, project_root, "confirmatory supervisor")
    name = "_rankcloak_release_confirmatory_supervisor"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConfirmatoryReleaseIndexError("cannot load confirmatory supervisor")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    sys.path.insert(0, str(project_root))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def verify_legacy_confirmatory_pipeline(project_root: Path) -> dict[str, object]:
    """Validate the retained pre-final index contract against local artifacts.

    The current code-and-data release does not call this compatibility path.
    """
    root = Path(project_root).resolve(strict=True)
    progress_path = root / "results/revision_v1/final_progress_snapshot_v1.json"
    try:
        progress = verify_progress_snapshot(progress_path)
    except RevisionProgressError as exc:
        raise ConfirmatoryReleaseIndexError("sealed final progress is invalid: {}".format(exc)) from exc
    counts = progress.get("counts")
    if (
        progress.get("execution_status") != "complete"
        or not isinstance(counts, dict)
        or int(counts.get("completed", -1)) != int(counts.get("total", -2))
        or int(counts.get("failures", -1)) != 0
        or int(counts.get("remaining", -1)) != 0
    ):
        raise ConfirmatoryReleaseIndexError("sealed final progress is not complete and failure-free")

    supervisor = _load_supervisor(root)
    substitutions = supervisor._format_values()
    actions: list[dict[str, object]] = []
    preprocess = supervisor.preprocess_specs()
    contract = supervisor.load_command_contract(
        root / "operations/confirmatory_v2/downstream_commands.json"
    )
    specifications = list(preprocess) + list(contract["operations"])
    observed_ids = tuple(str(row.get("operation_id")) for row in specifications)
    if observed_ids != LEGACY_EXPECTED_ACTION_IDS:
        raise ConfirmatoryReleaseIndexError("confirmatory completion action order changed")
    for row in specifications:
        if not supervisor.verify_completion(row, substitutions):
            raise ConfirmatoryReleaseIndexError(
                "confirmatory completion is absent: {}".format(row.get("operation_id"))
            )
        completion = row["completion"]
        manifest_path = Path(str(completion["path"]).format_map(substitutions))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        actions.append(
            {
                "operation_id": row["operation_id"],
                "completion_kind": completion["kind"],
                "manifest": _file_record(manifest_path, root),
            }
        )
    return {
        "status": "verified_complete",
        "final_progress": {
            **_file_record(progress_path, root),
            "progress_sha256": progress["progress_sha256"],
            "execution_status": progress["execution_status"],
            "counts": counts,
            "cumulative_actual_gpu_hours": progress["cumulative_actual_gpu_hours"],
        },
        "actions": actions,
        "actions_sha256": canonical_json_sha256(actions),
    }


def render_confirmatory_release_index(project_root: Path) -> dict[str, object]:
    """Render the legacy index schema; not used by the active release recipe."""
    root = Path(project_root).resolve(strict=True)
    verification = verify_legacy_confirmatory_pipeline(root)
    validators = [_file_record(root / relative, root) for relative in VALIDATOR_SOURCE_PATHS]
    artifacts = []
    for group, source, destination, role, kind in LEGACY_CONFIRMATORY_ARTIFACT_SPECS:
        files = _artifact_files(root, source)
        artifacts.append(
            {
                "group": group,
                "source": source,
                "destination": destination,
                "evidence_role": role,
                "completion_kind": kind,
                "file_count": len(files),
                "files": files,
                "files_sha256": canonical_json_sha256(files),
            }
        )
    value: dict[str, object] = {
        "schema_version": INDEX_SCHEMA,
        "manifest_type": INDEX_TYPE,
        "status": "verified_complete",
        "source_project_root": str(root),
        "protocol_contract_revision": PROTOCOL_REVISION,
        "result_schema_revision": RESULT_REVISION,
        "authorized_projection_sha256": AUTHORIZED_PROJECTION_SHA256,
        "network_access_used": False,
        "external_publication_performed": False,
        "doi_minted_or_reserved": False,
        "model_weights_included": False,
        "invalidated_or_superseded_outputs_included": False,
        "verification": verification,
        "validator_sources": validators,
        "validator_sources_sha256": canonical_json_sha256(validators),
        "artifacts": artifacts,
        "artifacts_sha256": canonical_json_sha256(artifacts),
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    return value


def _verify_index_document(
    value: Mapping[str, object], project_root: Path, *, verify_live: bool
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    if set(value) != INDEX_FIELDS:
        raise ConfirmatoryReleaseIndexError("confirmatory release index fields mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ConfirmatoryReleaseIndexError("confirmatory release index self-hash mismatch")
    expected_scalars = {
        "schema_version": INDEX_SCHEMA,
        "manifest_type": INDEX_TYPE,
        "status": "verified_complete",
        "protocol_contract_revision": PROTOCOL_REVISION,
        "result_schema_revision": RESULT_REVISION,
        "authorized_projection_sha256": AUTHORIZED_PROJECTION_SHA256,
        "network_access_used": False,
        "external_publication_performed": False,
        "doi_minted_or_reserved": False,
        "model_weights_included": False,
        "invalidated_or_superseded_outputs_included": False,
    }
    if any(value.get(key) != expected for key, expected in expected_scalars.items()):
        raise ConfirmatoryReleaseIndexError("confirmatory release index contract mismatch")
    if value.get("source_project_root") != str(root):
        raise ConfirmatoryReleaseIndexError("confirmatory source-project root differs")
    validators = value.get("validator_sources")
    if (
        not isinstance(validators, list)
        or value.get("validator_sources_sha256") != canonical_json_sha256(validators)
        or [row.get("path") for row in validators if isinstance(row, dict)]
        != list(VALIDATOR_SOURCE_PATHS)
    ):
        raise ConfirmatoryReleaseIndexError("confirmatory validator-source declaration mismatch")
    for row in validators:
        if row != _file_record(root / str(row["path"]), root):
            raise ConfirmatoryReleaseIndexError("confirmatory validator source changed")
    if verify_live and value.get("verification") != verify_legacy_confirmatory_pipeline(root):
        raise ConfirmatoryReleaseIndexError("live confirmatory verification differs from release index")
    verification = value.get("verification")
    actions = verification.get("actions") if isinstance(verification, dict) else None
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "verified_complete"
        or not isinstance(actions, list)
        or any(not isinstance(row, dict) for row in actions)
        or tuple(str(row.get("operation_id")) for row in actions) != LEGACY_EXPECTED_ACTION_IDS
        or verification.get("actions_sha256") != canonical_json_sha256(actions)
    ):
        raise ConfirmatoryReleaseIndexError("confirmatory pipeline-verification declaration mismatch")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or value.get("artifacts_sha256") != canonical_json_sha256(artifacts):
        raise ConfirmatoryReleaseIndexError("confirmatory artifact-list hash mismatch")
    expected_specs = [tuple(row) for row in LEGACY_CONFIRMATORY_ARTIFACT_SPECS]
    observed_specs = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ConfirmatoryReleaseIndexError("confirmatory artifact row is malformed")
        if set(artifact) != ARTIFACT_FIELDS:
            raise ConfirmatoryReleaseIndexError("confirmatory artifact fields mismatch")
        observed_specs.append(tuple(artifact.get(key) for key in (
            "group", "source", "destination", "evidence_role", "completion_kind"
        )))
        files = artifact.get("files")
        actual = _artifact_files(root, str(artifact.get("source")))
        if (
            not isinstance(files, list)
            or artifact.get("file_count") != len(files)
            or artifact.get("files_sha256") != canonical_json_sha256(files)
            or files != actual
        ):
            raise ConfirmatoryReleaseIndexError(
                "confirmatory artifact bytes differ: {}".format(artifact.get("source"))
            )
    if observed_specs != expected_specs:
        raise ConfirmatoryReleaseIndexError("confirmatory artifact map differs from the frozen release map")
    return {
        "status": "verified_complete",
        "manifest_sha256": claimed,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "verification": verification,
    }


def verify_confirmatory_release_index(path: Path, project_root: Path) -> dict[str, object]:
    index_path = _safe_regular(Path(path), Path(project_root), "confirmatory release index")
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfirmatoryReleaseIndexError("confirmatory release index is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ConfirmatoryReleaseIndexError("confirmatory release index must be an object")
    return _verify_index_document(value, Path(project_root), verify_live=True)


_REPOSITORY_TOP_LEVELS = {
    ".paper",
    "analysis",
    "configs",
    "environment",
    "operations",
    "rankcloak",
    "release",
    "results",
    "revision_docs",
    "scripts",
    "tests",
}
_INTENTIONALLY_UNPACKAGED_PREFIXES = {
    ".venv",
    "models",
}
_INTENTIONALLY_EXCLUDED_PATH_COMPONENTS = {
    "recovered_orphan_fit_files",
    "recovered_fit_permits",
}
_EMPTY_OPERATIONAL_DIRECTORY_NAMES = {
    "cpu_run",
    "cuda_repeat_run",
    "report_work",
}
_KNOWN_ABSENT_CPU_RECEIPT_DIRS = {
    Path(
        "results/revision_v1/detector_equivalence_v1/task_{}/"
        "cpu_run.checkpoints/fit_permit_receipts".format(index)
    )
    for index in (0, 1)
}
_KNOWN_ABSENT_PRODUCTION_PERMIT = Path(
    "results/revision_v1/neural_detector/confirmatory_v2.fit_permit.json"
)


def _lexical_staged_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ConfirmatoryReleaseIndexError("unsafe staged path: {}".format(relative))
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ConfirmatoryReleaseIndexError(
                "symlink in staged path: {}".format(relative)
            )
    return candidate


def _staged_index_maps(
    value: Mapping[str, object], root: Path
) -> tuple[dict[str, Path], list[tuple[Path, Path]]]:
    """Map original repository-relative bytes/directories to staged paths."""

    file_map: dict[str, Path] = {}
    directory_map: list[tuple[Path, Path]] = []
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfirmatoryReleaseIndexError("staged artifact map is malformed")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ConfirmatoryReleaseIndexError("staged artifact row is malformed")
        source = Path(str(artifact.get("source", "")))
        destination = Path(str(artifact.get("destination", "")))
        files = artifact.get("files")
        if not isinstance(files, list):
            raise ConfirmatoryReleaseIndexError("staged artifact files are malformed")
        source_is_file = source.as_posix() in CONFIRMATORY_FILE_SOURCES
        if source_is_file and (
            len(files) != 1
            or not isinstance(files[0], dict)
            or files[0].get("path") != source.as_posix()
        ):
            raise ConfirmatoryReleaseIndexError(
                "staged file source declaration differs from frozen map"
            )
        if not source_is_file:
            directory_map.append((source, _lexical_staged_path(root, destination)))
        for row in files:
            if not isinstance(row, dict):
                raise ConfirmatoryReleaseIndexError(
                    "staged artifact file row is malformed"
                )
            original = Path(str(row.get("path", "")))
            try:
                nested = Path(original.name) if source_is_file else original.relative_to(source)
            except ValueError as exc:
                raise ConfirmatoryReleaseIndexError(
                    "indexed file escapes its source"
                ) from exc
            target_relative = destination if source_is_file else destination / nested
            target = _lexical_staged_path(root, target_relative)
            key = original.as_posix()
            prior = file_map.get(key)
            if prior is not None and prior != target:
                raise ConfirmatoryReleaseIndexError(
                    "one original byte maps to multiple staged paths: {}".format(key)
                )
            file_map[key] = target
    validators = value.get("validator_sources")
    if not isinstance(validators, list):
        raise ConfirmatoryReleaseIndexError("staged validator map is malformed")
    for row in validators:
        if not isinstance(row, dict):
            raise ConfirmatoryReleaseIndexError("staged validator row is malformed")
        original = str(row.get("path", ""))
        target_text = VALIDATOR_CANDIDATE_PATHS.get(original)
        if target_text is None:
            raise ConfirmatoryReleaseIndexError("staged validator mapping is absent")
        target = _lexical_staged_path(root, Path(target_text))
        prior = file_map.get(original)
        if prior is not None and prior != target:
            raise ConfirmatoryReleaseIndexError(
                "validator mapping is ambiguous: {}".format(original)
            )
        file_map[original] = target
    directory_map.sort(key=lambda row: len(row[0].parts), reverse=True)
    return file_map, directory_map


def _staged_repository_relative(path_text: str, source_root: Path) -> Path | None:
    path = Path(path_text)
    if path.is_absolute():
        try:
            relative = path.relative_to(source_root)
        except ValueError as exc:
            raise ConfirmatoryReleaseIndexError(
                "signed artifact references an external host path: {}".format(path)
            ) from exc
    else:
        relative = path
    if ".." in relative.parts or "." in relative.parts:
        raise ConfirmatoryReleaseIndexError(
            "signed artifact contains an unsafe repository path: {}".format(path)
        )
    if not relative.parts or relative.parts[0] not in _REPOSITORY_TOP_LEVELS:
        return None
    return relative


def _resolve_staged_repository_path(
    path_text: str,
    *,
    source_root: Path,
    root: Path,
    file_map: Mapping[str, Path],
    directory_map: Sequence[tuple[Path, Path]],
) -> Path | None:
    """Resolve only through the signed index; never touch the original host path."""

    relative = _staged_repository_relative(path_text, source_root)
    if relative is None:
        return None
    if relative.parts and relative.parts[0] in _INTENTIONALLY_UNPACKAGED_PREFIXES:
        return None
    if set(relative.parts) & _INTENTIONALLY_EXCLUDED_PATH_COMPONENTS:
        return None
    if relative == _KNOWN_ABSENT_PRODUCTION_PERMIT:
        return _lexical_staged_path(
            root, Path("results/confirmatory/neural_detector/confirmatory_v2.fit_permit.json")
        )
    direct = file_map.get(relative.as_posix())
    if direct is not None:
        return direct
    for original_root, staged_root in directory_map:
        try:
            nested = relative.relative_to(original_root)
        except ValueError:
            continue
        return _lexical_staged_path(root, staged_root.relative_to(root) / nested)
    raise ConfirmatoryReleaseIndexError(
        "signed artifact path is not indexed for staging: {}".format(relative)
    )


def _verify_staged_identity_path(
    path_text: str,
    declaration: Mapping[str, object],
    *,
    source_root: Path,
    root: Path,
    file_map: Mapping[str, Path],
    directory_map: Sequence[tuple[Path, Path]],
) -> None:
    candidate = _resolve_staged_repository_path(
        path_text,
        source_root=source_root,
        root=root,
        file_map=file_map,
        directory_map=directory_map,
    )
    relative = _staged_repository_relative(path_text, source_root)
    if candidate is None:
        return
    assert relative is not None
    if str(relative).endswith(".fit_permit.json"):
        if candidate.exists() or candidate.is_symlink():
            raise ConfirmatoryReleaseIndexError(
                "active fit permit was staged: {}".format(relative)
            )
        return
    resolved = _safe_regular(candidate, root, "staged signed dependency")
    expected_hash = declaration.get("sha256")
    expected_size = declaration.get("size_bytes")
    if expected_hash is not None:
        observed_hash = file_sha256(resolved)
        if relative == Path("results/revision_v1/compute_projection_165h_v2.json"):
            try:
                projection = json.loads(resolved.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfirmatoryReleaseIndexError(
                    "staged authorized projection is invalid JSON"
                ) from exc
            unsigned_projection = dict(projection)
            recorded_projection_hash = unsigned_projection.pop(
                "projection_sha256", None
            )
            if (
                recorded_projection_hash != expected_hash
                or recorded_projection_hash
                != canonical_json_sha256(unsigned_projection)
            ):
                raise ConfirmatoryReleaseIndexError(
                    "staged authorized projection semantic hash mismatch"
                )
        elif observed_hash != expected_hash:
            raise ConfirmatoryReleaseIndexError(
                "staged signed dependency hash mismatch: {}".format(relative)
            )
    if expected_size is not None and resolved.stat().st_size != int(expected_size):
        raise ConfirmatoryReleaseIndexError(
            "staged signed dependency size mismatch: {}".format(relative)
        )


def _verify_staged_json_paths(
    value: object,
    *,
    source_root: Path,
    root: Path,
    file_map: Mapping[str, Path],
    directory_map: Sequence[tuple[Path, Path]],
) -> None:
    """Verify repository-bound identities in a staged signed JSON document."""

    if isinstance(value, list):
        for item in value:
            _verify_staged_json_paths(
                item,
                source_root=source_root,
                root=root,
                file_map=file_map,
                directory_map=directory_map,
            )
        return
    if not isinstance(value, dict):
        return
    schema = value.get("schema_version")
    signature_field = DETECTOR_SIGNED_SCHEMA_FIELDS.get(str(schema))
    if signature_field is not None:
        unsigned = dict(value)
        claimed = unsigned.pop(signature_field, None)
        if claimed != canonical_json_sha256(unsigned):
            raise ConfirmatoryReleaseIndexError(
                "staged detector signed-document self-hash mismatch: {}".format(
                    schema
                )
            )
    path_value = value.get("path")
    if isinstance(path_value, str) and (
        "sha256" in value or "size_bytes" in value
    ):
        _verify_staged_identity_path(
            path_value,
            value,
            source_root=source_root,
            root=root,
            file_map=file_map,
            directory_map=directory_map,
        )
    for key, path_value in value.items():
        if not isinstance(key, str) or not isinstance(path_value, str):
            continue
        handled_identity = False
        if key.endswith("_path"):
            hash_key = key[:-5] + "_sha256"
            if hash_key in value:
                _verify_staged_identity_path(
                    path_value,
                    {"sha256": value.get(hash_key)},
                    source_root=source_root,
                    root=root,
                    file_map=file_map,
                    directory_map=directory_map,
                )
                handled_identity = True
        if not handled_identity and key in {
            "checkpoint_dir",
            "fit_permit_file",
            "fit_permit_receipt_dir",
            "model_directory",
            "output_dir",
            "pre_final_gpu_accounting_ledger_path",
            "requested_output_path",
            "root",
            "status_file",
        }:
            candidate = _resolve_staged_repository_path(
                path_value,
                source_root=source_root,
                root=root,
                file_map=file_map,
                directory_map=directory_map,
            )
            relative = _staged_repository_relative(path_value, source_root)
            if candidate is not None and relative is not None:
                if str(relative).endswith(".fit_permit.json"):
                    if candidate.exists() or candidate.is_symlink():
                        raise ConfirmatoryReleaseIndexError(
                            "active fit permit was staged: {}".format(relative)
                        )
                elif not candidate.exists() and (
                    relative.name not in _EMPTY_OPERATIONAL_DIRECTORY_NAMES
                    and relative not in _KNOWN_ABSENT_CPU_RECEIPT_DIRS
                ):
                    raise ConfirmatoryReleaseIndexError(
                        "staged signed path is absent: {}".format(relative)
                    )
    for item in value.values():
        _verify_staged_json_paths(
            item,
            source_root=source_root,
            root=root,
            file_map=file_map,
            directory_map=directory_map,
        )


def verify_staged_confirmatory_release_index(
    path: Path, candidate_root: Path
) -> dict[str, object]:
    """Verify indexed artifact bytes after paths are remapped into a candidate."""

    root = Path(candidate_root).resolve(strict=True)
    index_path = _safe_regular(Path(path), root, "staged confirmatory release index")
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfirmatoryReleaseIndexError("staged confirmatory index is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ConfirmatoryReleaseIndexError("staged confirmatory index must be an object")
    if set(value) != INDEX_FIELDS:
        raise ConfirmatoryReleaseIndexError("staged confirmatory index fields mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ConfirmatoryReleaseIndexError("staged confirmatory index self-hash mismatch")
    if any(value.get(key) != expected for key, expected in {
        "schema_version": INDEX_SCHEMA,
        "manifest_type": INDEX_TYPE,
        "status": "verified_complete",
        "protocol_contract_revision": PROTOCOL_REVISION,
        "result_schema_revision": RESULT_REVISION,
        "authorized_projection_sha256": AUTHORIZED_PROJECTION_SHA256,
        "network_access_used": False,
        "external_publication_performed": False,
        "doi_minted_or_reserved": False,
        "model_weights_included": False,
        "invalidated_or_superseded_outputs_included": False,
    }.items()):
        raise ConfirmatoryReleaseIndexError("staged confirmatory index contract mismatch")
    source_root_text = value.get("source_project_root")
    source_root = Path(str(source_root_text))
    if (
        not isinstance(source_root_text, str)
        or not source_root_text
        or not source_root.is_absolute()
        or any(part in {".", ".."} for part in source_root.parts)
        or source_root.as_posix() != source_root_text
        or source_root == root
    ):
        raise ConfirmatoryReleaseIndexError(
            "staged source-project root declaration is unsafe"
        )
    validators = value.get("validator_sources")
    if (
        not isinstance(validators, list)
        or value.get("validator_sources_sha256") != canonical_json_sha256(validators)
        or [row.get("path") for row in validators if isinstance(row, dict)]
        != list(VALIDATOR_SOURCE_PATHS)
    ):
        raise ConfirmatoryReleaseIndexError("staged validator-source declaration mismatch")
    for row in validators:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ConfirmatoryReleaseIndexError("staged validator-source row is malformed")
        candidate = _lexical_staged_path(
            root, Path(VALIDATOR_CANDIDATE_PATHS[str(row["path"])])
        )
        resolved = _safe_regular(candidate, root, "staged validator source")
        if resolved.stat().st_size != row["size_bytes"] or file_sha256(resolved) != row["sha256"]:
            raise ConfirmatoryReleaseIndexError("staged validator source hash mismatch")
    verification = value.get("verification")
    actions = verification.get("actions") if isinstance(verification, dict) else None
    final_progress = verification.get("final_progress") if isinstance(verification, dict) else None
    counts = final_progress.get("counts") if isinstance(final_progress, dict) else None
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "verified_complete"
        or not isinstance(actions, list)
        or any(not isinstance(row, dict) for row in actions)
        or tuple(str(row.get("operation_id")) for row in actions) != LEGACY_EXPECTED_ACTION_IDS
        or verification.get("actions_sha256") != canonical_json_sha256(actions)
        or not isinstance(counts, dict)
        or final_progress.get("execution_status") != "complete"
        or int(counts.get("completed", -1)) != int(counts.get("total", -2))
        or int(counts.get("failures", -1)) != 0
        or int(counts.get("remaining", -1)) != 0
    ):
        raise ConfirmatoryReleaseIndexError("staged pipeline verification is incomplete")
    for action, operation_id, completion_kind in zip(
        actions, LEGACY_EXPECTED_ACTION_IDS, LEGACY_EXPECTED_ACTION_KINDS
    ):
        manifest = action.get("manifest")
        if (
            set(action) != {"operation_id", "completion_kind", "manifest"}
            or action.get("operation_id") != operation_id
            or action.get("completion_kind") != completion_kind
            or not isinstance(manifest, dict)
            or set(manifest) != {"path", "size_bytes", "sha256"}
        ):
            raise ConfirmatoryReleaseIndexError(
                "staged completion-action identity is malformed"
            )
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or value.get("artifacts_sha256") != canonical_json_sha256(artifacts):
        raise ConfirmatoryReleaseIndexError("staged confirmatory artifact-list hash mismatch")
    observed_specs = [
        tuple(row.get(key) for key in (
            "group", "source", "destination", "evidence_role", "completion_kind"
        ))
        for row in artifacts if isinstance(row, dict)
    ]
    if observed_specs != [tuple(row) for row in LEGACY_CONFIRMATORY_ARTIFACT_SPECS]:
        raise ConfirmatoryReleaseIndexError("staged confirmatory artifact map mismatch")
    expected_staged_paths: set[str] = set()
    directory_destinations: list[Path] = []
    semantic_json_paths: set[Path] = set()
    seen_original_paths: set[str] = set()
    seen_staged_targets: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS:
            raise ConfirmatoryReleaseIndexError("staged confirmatory artifact fields mismatch")
        files = artifact.get("files")
        if (
            not isinstance(files, list)
            or not files
            or artifact.get("file_count") != len(files)
            or artifact.get("files_sha256") != canonical_json_sha256(files)
        ):
            raise ConfirmatoryReleaseIndexError("staged confirmatory file declaration mismatch")
        source_text = str(artifact["source"])
        source = Path(source_text)
        destination = Path(str(artifact["destination"]))
        expected_paths = set()
        source_is_file = source_text in CONFIRMATORY_FILE_SOURCES
        if source_is_file and (
            len(files) != 1
            or not isinstance(files[0], dict)
            or files[0].get("path") != source_text
        ):
            raise ConfirmatoryReleaseIndexError(
                "staged file source declaration differs from frozen map"
            )
        if not source_is_file:
            directory_destinations.append(destination)
        for row in files:
            if not isinstance(row, dict) or set(row) != {
                "path", "size_bytes", "sha256"
            }:
                raise ConfirmatoryReleaseIndexError(
                    "staged confirmatory file row is malformed"
                )
            original = Path(str(row.get("path")))
            if original.as_posix() in seen_original_paths:
                raise ConfirmatoryReleaseIndexError(
                    "duplicate indexed confirmatory file path: {}".format(original)
                )
            seen_original_paths.add(original.as_posix())
            try:
                nested = Path(original.name) if source_is_file else original.relative_to(source)
            except ValueError as exc:
                raise ConfirmatoryReleaseIndexError("indexed file escapes its source") from exc
            candidate_relative = destination if source_is_file else destination / nested
            if candidate_relative.as_posix() in seen_staged_targets:
                raise ConfirmatoryReleaseIndexError(
                    "duplicate staged confirmatory target: {}".format(
                        candidate_relative
                    )
                )
            seen_staged_targets.add(candidate_relative.as_posix())
            candidate = _lexical_staged_path(root, candidate_relative)
            resolved = _safe_regular(candidate, root, "staged confirmatory artifact")
            if (
                resolved.stat().st_size != row.get("size_bytes")
                or file_sha256(resolved) != row.get("sha256")
            ):
                raise ConfirmatoryReleaseIndexError(
                    "staged confirmatory artifact hash mismatch: {}".format(candidate)
                )
            staged_relative = resolved.relative_to(root).as_posix()
            expected_paths.add(staged_relative)
            expected_staged_paths.add(staged_relative)
            if source_text in DETECTOR_SEMANTIC_SOURCES and candidate.suffix == ".json":
                semantic_json_paths.add(candidate)

    minimal_destinations = [
        destination
        for destination in directory_destinations
        if not any(
            destination != other and destination.is_relative_to(other)
            for other in directory_destinations
        )
    ]
    for destination in minimal_destinations:
        destination_root = _lexical_staged_path(root, destination)
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ConfirmatoryReleaseIndexError(
                "staged confirmatory directory is missing or unsafe: {}".format(
                    destination
                )
            )
        actual_paths: set[str] = set()
        for item in sorted(destination_root.rglob("*"), key=lambda row: row.as_posix()):
            relative = item.relative_to(root)
            if item.is_symlink():
                raise ConfirmatoryReleaseIndexError(
                    "symlink in staged confirmatory directory: {}".format(relative)
                )
            if item.is_dir():
                continue
            if not item.is_file() or not stat.S_ISREG(item.stat().st_mode):
                raise ConfirmatoryReleaseIndexError(
                    "special file in staged confirmatory directory: {}".format(relative)
                )
            actual_paths.add(relative.as_posix())
        expected_under_destination = {
            item for item in expected_staged_paths
            if Path(item).is_relative_to(destination)
        }
        if actual_paths != expected_under_destination:
            raise ConfirmatoryReleaseIndexError(
                "staged confirmatory directory file-set mismatch: {}".format(
                    destination
                )
            )

    for path_item in root.rglob("*.fit_permit.json"):
        raise ConfirmatoryReleaseIndexError(
            "active fit permit was staged: {}".format(path_item.relative_to(root))
        )

    file_map, directory_map = _staged_index_maps(value, root)
    semantic_json_paths.add(
        _lexical_staged_path(
            root,
            Path(
                VALIDATOR_CANDIDATE_PATHS[
                    "operations/confirmatory_v2/detector_acceleration_policy_v1.json"
                ]
            ),
        )
    )
    _verify_staged_json_paths(
        verification,
        source_root=source_root,
        root=root,
        file_map=file_map,
        directory_map=directory_map,
    )
    for semantic_path in sorted(semantic_json_paths, key=lambda row: row.as_posix()):
        try:
            document = json.loads(semantic_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfirmatoryReleaseIndexError(
                "staged detector semantic artifact is invalid JSON: {}".format(
                    semantic_path.relative_to(root)
                )
            ) from exc
        _verify_staged_json_paths(
            document,
            source_root=source_root,
            root=root,
            file_map=file_map,
            directory_map=directory_map,
        )
    return {
        "status": "verified_complete",
        "manifest_sha256": claimed,
        "artifact_count": len(artifacts),
    }


def write_confirmatory_release_index(path: Path, value: Mapping[str, object]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise ConfirmatoryReleaseIndexError("confirmatory release index already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_text = tempfile.mkstemp(prefix=".{}-".format(target.name), dir=str(target.parent))
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ConfirmatoryReleaseIndexError("confirmatory release index already exists") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = args.project_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        if args.check:
            report = verify_confirmatory_release_index(output, root)
        else:
            value = render_confirmatory_release_index(root)
            report = {
                "status": "verified_complete",
                "manifest_sha256": value["manifest_sha256"],
                "artifact_count": len(value["artifacts"]),
                "output": str(output),
                "output_created": False,
            }
            if not args.dry_run:
                write_confirmatory_release_index(output, value)
                report["output_created"] = True
    except ConfirmatoryReleaseIndexError as exc:
        raise SystemExit(
            "confirmatory release index verification failed: {}".format(exc)
        ) from exc
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "LEGACY_CONFIRMATORY_ARTIFACT_SPECS",
    "ConfirmatoryReleaseIndexError",
    "INDEX_SCHEMA",
    "render_confirmatory_release_index",
    "verify_confirmatory_release_index",
    "verify_staged_confirmatory_release_index",
    "verify_legacy_confirmatory_pipeline",
    "write_confirmatory_release_index",
]
