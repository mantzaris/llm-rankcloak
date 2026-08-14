#!/usr/bin/env python3
"""Build and verify the offline RankCloak revision environment snapshot.

This command is deliberately observational: it installs nothing, downloads
nothing, and never copies model weights.  It records installed distributions,
native/backend identities, hardware, and the already-frozen scientific inputs
as content-addressed metadata suitable for an offline DOI release candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT = Path("environment/revision_v1")
MODEL_WEIGHT_SUFFIXES = {
    ".gguf", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".h5"
}
REQUIRED_CUDA_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_LAUNCH_BLOCKING": "1",
    "GGML_CUDA_DISABLE_GRAPHS": "1",
    "GGML_CUDA_DISABLE_FUSION": "1",
    "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
R_REQUIRED_ENVIRONMENT = {
    "R_ENVIRON_USER": "/dev/null",
    "R_PROFILE_USER": "/dev/null",
    "R_LIBS_USER": ".r_libs/revision_v1:$HOME/R/x86_64-pc-linux-gnu-library/4.4",
}
OBSERVED_ENVIRONMENT_NAMES = tuple(
    sorted(
        set(REQUIRED_CUDA_ENVIRONMENT)
        | {"CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"}
    )
)

# These paths define the source contract used to interpret revision-v1 results.
# They are content-pinned in the environment snapshot; model files are recorded
# separately by identifier, size, and SHA-256 and are never copied.
SCIENTIFIC_SOURCE_PATHS = (
    "CITATION.cff",
    "pyproject.toml",
    "rankcloak/model_io.py",
    "rankcloak/rank_codec.py",
    "rankcloak/token_filters.py",
    "rankcloak/segmented_protocol.py",
    "rankcloak/revision_artifacts.py",
    "rankcloak/revision_config.py",
    "rankcloak/revision_payloads.py",
    "rankcloak/revision_protocol.py",
    "rankcloak/revision_runner.py",
    "rankcloak/revision_evaluator.py",
    "rankcloak/revision_evaluator_join.py",
    "rankcloak/revision_compute.py",
    "rankcloak/revision_tokenizer_preflight.py",
    "rankcloak/revision_invalidation.py",
    "rankcloak/revision_preprocess.py",
    "rankcloak/revision_statistics.py",
    "rankcloak/revision_detection.py",
    "rankcloak/revision_detector_execution.py",
    "rankcloak/revision_theory.py",
    "rankcloak/revision_reporting.py",
    "rankcloak/revision_progress.py",
    "rankcloak/revision_release.py",
    "rankcloak/revision_release_index.py",
    "scripts/build_revision_payload_manifest.py",
    "scripts/run_revision_matrix.py",
    "scripts/run_revision_evaluator.py",
    "scripts/join_revision_evaluator_features.py",
    "scripts/run_revision_tokenizer_preflight.py",
    "scripts/invalidate_revision_shard.py",
    "scripts/manage_legacy_gpu_ledger.py",
    "scripts/project_revision_compute.py",
    "scripts/preprocess_revision_results.py",
    "scripts/run_revision_statistics.py",
    "scripts/run_revision_mixed_models.R",
    "scripts/run_revision_detectors.py",
    "scripts/build_revision_theory.py",
    "scripts/build_revision_reports.py",
    "scripts/update_revision_progress.py",
    "scripts/supervise_primary_v2.py",
    "scripts/supervise_confirmatory_v2.py",
    "scripts/revise_revision_manuscripts.py",
    "scripts/build_revision_environment_lock.py",
    "scripts/build_revision_release.py",
    "scripts/build_revision_confirmatory_release_index.py",
    "scripts/verify_revision_release.py",
    "operations/confirmatory_v2/downstream_commands.json",
    "operations/confirmatory_v2/detector_acceleration_policy_v1.json",
    "release/revision_v1_template/release_spec.json",
    "release/revision_v1_template/README.md",
    "revision_docs/DOI_RELEASE_PLAN.md",
    "analysis/revision_v1/r_environment.lock.json",
    "analysis/revision_v1/run_with_locked_r.sh",
    "analysis/revision_v1/confirmatory_model_plan.json",
    "analysis/revision_v1/detector_confirmatory_plan.json",
    "tests/test_revision_detection.py",
    "tests/test_revision_detector_execution.py",
    "tests/test_revision_evaluator_join.py",
    "tests/test_revision_progress.py",
    "tests/test_revision_reporting.py",
    "tests/test_primary_v2_supervisor.py",
    "tests/test_confirmatory_v2_orchestrator.py",
    "tests/test_revision_manuscripts.py",
    "tests/test_revision_release.py",
    "tests/test_revision_release_index.py",
    "tests/test_revision_environment_lock.py",
    "revision_docs/NEURAL_DETECTOR_EXECUTION.md",
    "human_study/config/power_design_grid.json",
    "human_study/power/planning_power_design_grid.csv",
    "human_study/power/simulate_power.py",
    "human_study/tests/test_power_design_grid.py",
    "human_study/power/PLANNING_RESULTS.md",
    "human_study/power/ASSUMPTIONS_AND_SENSITIVITY.md",
    "human_study/README.md",
)

# This closed subset makes the detector checkpoint/equivalence release closure
# explicit inside scientific_pins.json.  Every byte is already hashed through
# SCIENTIFIC_SOURCE_PATHS; the separate declaration prevents a future source,
# validator, specification, test, or explanatory document from silently
# falling out of the reproducibility contract.
DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS = (
    "rankcloak/revision_artifacts.py",
    "rankcloak/revision_detection.py",
    "rankcloak/revision_detector_execution.py",
    "rankcloak/revision_progress.py",
    "rankcloak/revision_release.py",
    "rankcloak/revision_release_index.py",
    "scripts/run_revision_detectors.py",
    "scripts/supervise_confirmatory_v2.py",
    "scripts/build_revision_environment_lock.py",
    "scripts/build_revision_release.py",
    "scripts/build_revision_confirmatory_release_index.py",
    "scripts/verify_revision_release.py",
    "operations/confirmatory_v2/downstream_commands.json",
    "operations/confirmatory_v2/detector_acceleration_policy_v1.json",
    "release/revision_v1_template/release_spec.json",
    "release/revision_v1_template/README.md",
    "revision_docs/DOI_RELEASE_PLAN.md",
    "revision_docs/NEURAL_DETECTOR_EXECUTION.md",
    "tests/test_revision_detection.py",
    "tests/test_revision_detector_execution.py",
    "tests/test_revision_progress.py",
    "tests/test_confirmatory_v2_orchestrator.py",
    "tests/test_revision_release.py",
    "tests/test_revision_release_index.py",
    "tests/test_revision_environment_lock.py",
)
SCIENTIFIC_SOURCE_DIRECTORIES = ("human_study",)
SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_COMPONENTS = {"__pycache__"}
SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
SCIENTIFIC_SOURCE_DIRECTORY_FORBIDDEN_NAMES = {
    "raw_responses.csv", "raw_responses.jsonl", "participant_ids.csv",
    "participant_data.csv", "survey_export.csv", "prolific_export.csv",
    "mturk_export.csv", "signed_consent.pdf",
}

_REVISION_MODEL_IDS = (
    "llama3_8b_instruct_q4_k_m",
    "qwen2_5_7b_instruct_q4_k_m",
    "mistral_7b_instruct_v0_3_q4_k_m",
)
VALIDATION_RESULT_PATHS = (
    (
        "results/revision_v1/tokenizer_preflight_v2/TOKENIZER_PREFLIGHT_MANIFEST.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json",
        "forensic_invalidated_not_for_pooling",
    ),
    (
        "results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json",
        "forensic_charge_only_not_rate_evidence",
    ),
    (
        "results/revision_v1/compute_projection_v2.json",
        "exploratory_compute_gate_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/compute_projection_165h_v2.json",
        "authorized_compute_gate_not_scientific_outcome",
    ),
    (
        "results/revision_v1/supervisor/detector_live_readonly_audit_20260813T0145Z.json",
        "operational_diagnostic_not_scientific_evidence",
    ),
    (
        "results/revision_v1/supervisor/detector_live_readonly_audit_addendum_20260813T0151Z.json",
        "operational_diagnostic_not_scientific_evidence",
    ),
    (
        "results/revision_v1/smoke_v3_preprocessed_v1/preprocessing_output_manifest.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/smoke_v3_integrity_v1/output_manifest.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/smoke_v3_statistics_v3/statistics_run_manifest.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/smoke_v3_reports_v3/report_output_manifest.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
    (
        "results/revision_v1/smoke_v3_theory_v2/theory_validation_manifest.json",
        "exploratory_validation_not_for_confirmatory_pooling",
    ),
) + tuple(
    (
        "results/revision_v1/smoke_v3/{}/run_identity.json".format(model_id),
        "exploratory_validation_not_for_confirmatory_pooling",
    )
    for model_id in _REVISION_MODEL_IDS
) + tuple(
    (
        "results/revision_v1/heldout_evaluator/smoke_v3/{}/run_identity.json".format(
            model_id
        ),
        "exploratory_validation_not_for_confirmatory_pooling",
    )
    for model_id in _REVISION_MODEL_IDS
)


class EnvironmentLockError(RuntimeError):
    """Raised when an environment snapshot cannot be made or verified."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], env: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    try:
        completed = subprocess.run(
            [str(value) for value in command],
            check=False,
            capture_output=True,
            text=True,
            env=None if env is None else dict(env),
        )
    except FileNotFoundError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "not found"}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _safe_version_text(text: str) -> str:
    """Keep version output useful without retaining absolute installation paths."""

    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first = re.sub(r"\s+from\s+/\S+", "", first)
    first = re.sub(r"\s+\(python\s+/\S+\)", "", first)
    return first[:500]


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_file_hash(distribution: importlib.metadata.Distribution, name: str) -> Optional[str]:
    path = getattr(distribution, "_path", None)
    if path is None:
        return None
    candidate = Path(path) / name
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return sha256_file(candidate)


def collect_python_environment(project_root: Path) -> Tuple[Dict[str, object], bytes]:
    records: List[Dict[str, object]] = []
    requirements: List[str] = []
    seen: Dict[str, List[str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name") or "unknown"
        name = _normalized_name(str(raw_name))
        version = str(distribution.version)
        direct_text = distribution.read_text("direct_url.json")
        direct_hash = None
        source_kind = "index_or_unspecified"
        vcs_commit = None
        editable = False
        if direct_text:
            direct_hash = sha256_bytes(direct_text.encode("utf-8"))
            try:
                direct = json.loads(direct_text)
            except json.JSONDecodeError:
                direct = {}
            if isinstance(direct, dict):
                directory = direct.get("dir_info")
                vcs = direct.get("vcs_info")
                if isinstance(directory, dict):
                    editable = bool(directory.get("editable"))
                    source_kind = "editable_local_directory" if editable else "local_directory"
                elif isinstance(vcs, dict):
                    source_kind = "version_control"
                    commit = vcs.get("commit_id")
                    vcs_commit = str(commit) if commit else None
                elif str(direct.get("url", "")).startswith("file:"):
                    source_kind = "local_artifact"
        record = {
            "name": name,
            "version": version,
            "source_kind": source_kind,
            "editable": editable,
            "vcs_commit": vcs_commit,
            "direct_url_record_sha256": direct_hash,
            "metadata_sha256": _distribution_file_hash(distribution, "METADATA"),
            "record_sha256": _distribution_file_hash(distribution, "RECORD"),
        }
        records.append(record)
        seen.setdefault(name, []).append(version)

    records.sort(key=lambda row: (str(row["name"]), str(row["version"]), str(row["source_kind"])))
    for name in sorted(seen):
        versions = sorted(set(seen[name]))
        if name == "rankcloak":
            requirements.append("-e .")
        elif len(versions) == 1:
            requirements.append("{}=={}".format(name, versions[0]))
        else:
            requirements.append("# duplicate-installed-distribution {} versions={}".format(name, ",".join(versions)))
            requirements.extend("{}=={}".format(name, version) for version in versions)

    requirements_text = (
        "# Observed exact distribution versions; generated offline.\n"
        "# This is not a cross-platform wheel-hash lock and performs no installation itself.\n"
        + "\n".join(requirements)
        + "\n"
    ).encode("utf-8")
    executable = Path(sys.executable)
    runtime = {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_executable_role": "environment_used_to_build_snapshot",
        "virtual_environment_active": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "pip_version": importlib.metadata.version("pip") if _package_exists("pip") else None,
        "package_count": len(records),
        "packages_sha256": sha256_bytes(canonical_json_bytes(records)),
        "requirements_lock_sha256": sha256_bytes(requirements_text),
        "lock_strength": "installed_versions_plus_distribution_metadata_fingerprints",
        "wheel_artifacts_vendored": False,
        "automatic_installation": False,
        "absolute_executable_path_recorded": False,
        "executable_name": executable.name,
        "packages": records,
    }
    return runtime, requirements_text


def _package_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


R_PROBE = r'''
cat("RUNTIME\t", paste(R.version$major, R.version$minor, sep="."), "\t", R.version$platform, "\n", sep="")
libs <- .libPaths()
for (i in seq_along(libs)) {
  cat("LIBRARY\t", i, "\t", normalizePath(libs[[i]], mustWork=FALSE), "\n", sep="")
  ip <- tryCatch(installed.packages(lib.loc=libs[[i]], fields=c("Priority")), error=function(e) NULL)
  if (is.null(ip) || nrow(ip) == 0) next
  for (j in seq_len(nrow(ip))) {
    priority <- if ("Priority" %in% colnames(ip) && !is.na(ip[j,"Priority"])) ip[j,"Priority"] else ""
    cat("PACKAGE\t", i, "\t", ip[j,"Package"], "\t", ip[j,"Version"], "\t", priority, "\n", sep="")
  }
}
'''


def _parse_description(path: Path) -> Mapping[str, str]:
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return fields
    for line in lines:
        if line.startswith((" ", "\t")) and current:
            fields[current] = fields[current] + " " + line.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            current = key.strip()
            fields[current] = value.strip()
    return fields


def _portable_library_path(raw_path: str, project_root: Path) -> Tuple[str, str]:
    path = Path(raw_path)
    try:
        relative = path.resolve().relative_to(project_root.resolve())
        return relative.as_posix(), "repository_relative"
    except (OSError, ValueError):
        pass
    try:
        relative_home = path.resolve().relative_to(Path.home().resolve())
        return "$HOME/{}".format(relative_home.as_posix()), "home_relative"
    except (OSError, ValueError):
        pass
    return str(path), "system_absolute"


def _r_versions_equivalent(left: object, right: object) -> bool:
    return re.split(r"[.-]", str(left)) == re.split(r"[.-]", str(right))


def inspect_r_project_library(project_root: Path, lock: Mapping[str, object]) -> Dict[str, object]:
    library = lock.get("library", {}) if isinstance(lock.get("library"), dict) else {}
    relative = str(library.get("relative_path", ""))
    if not relative:
        order = library.get("resolution_order", [])
        for entry in order if isinstance(order, list) else []:
            if isinstance(entry, dict) and entry.get("role") == "project_revision_library":
                relative = str(entry.get("path", ""))
                break
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EnvironmentLockError("R project library path must be a safe repository-relative path")
    library_path = project_root / relative
    records: List[Dict[str, object]] = []
    if library_path.is_dir():
        for description in sorted(library_path.glob("*/DESCRIPTION")):
            if not description.is_file() or description.is_symlink():
                continue
            metadata = _parse_description(description)
            records.append({
                "package": metadata.get("Package", description.parent.name),
                "version": metadata.get("Version"),
                "description_sha256": sha256_file(description),
            })
    return {
        "relative_path": relative,
        "directory_present": library_path.is_dir(),
        "observed_package_count": len(records),
        "records": records,
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def resolve_required_r_packages(
    lock: Mapping[str, object],
    observed_packages: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    # Resolve each required package in the launchers declared library order.
    expected = lock.get("packages", {}) if isinstance(lock.get("packages"), dict) else {}
    policy = lock.get("policy", {}) if isinstance(lock.get("policy"), dict) else {}
    require_project = bool(policy.get("system_or_user_copy_does_not_satisfy_locked_package"))
    records: List[Dict[str, object]] = []
    for name in sorted(expected):
        specification = expected[name]
        expected_version = str(specification.get("version")) if isinstance(specification, dict) else ""
        candidates = sorted(
            (row for row in observed_packages if row.get("name") == name),
            key=lambda row: int(row.get("library_index", 999999)),
        )
        chosen = candidates[0] if candidates else None
        status = "missing"
        if chosen is not None:
            if not _r_versions_equivalent(chosen.get("version"), expected_version):
                status = "version_mismatch"
            elif isinstance(specification, dict) and specification.get("required_library_role") and chosen.get("library_role") != specification.get("required_library_role"):
                status = "library_role_mismatch"
            elif require_project and chosen.get("library_path_kind") != "repository_relative":
                status = "scope_mismatch"
            else:
                status = "ok"
        records.append({
            "package": str(name),
            "expected_version": expected_version,
            "found_version": chosen.get("version") if chosen else None,
            "resolved_library_path": chosen.get("library_path") if chosen else None,
            "resolved_library_path_kind": chosen.get("library_path_kind") if chosen else None,
            "resolved_library_role": chosen.get("library_role") if chosen else None,
            "resolved_library_index": chosen.get("library_index") if chosen else None,
            "status": status,
        })
    errors = ["{}:{}".format(row["package"], row["status"]) for row in records if row["status"] != "ok"]
    return {"complete": not errors, "records": records, "errors": errors}


def _declared_r_libraries(lock: Mapping[str, object], project_root: Path) -> List[Dict[str, object]]:
    library = lock.get("library", {}) if isinstance(lock.get("library"), dict) else {}
    order = library.get("resolution_order", [])
    declared: List[Dict[str, object]] = []
    for entry in order if isinstance(order, list) else []:
        if not isinstance(entry, dict):
            continue
        path_kind = str(entry.get("path_kind", ""))
        raw = str(entry.get("path", ""))
        if path_kind == "repository_relative":
            portable = raw
        elif path_kind == "absolute_existing":
            portable, _ = _portable_library_path(raw, project_root)
        else:
            portable = raw
        declared.append({"role": entry.get("role"), "path": portable, "path_kind": path_kind})
    return declared


def _portable_r_lock(lock: Mapping[str, object], project_root: Path) -> Dict[str, object]:
    copied = json.loads(json.dumps(lock))
    library = copied.get("library", {})
    if isinstance(library, dict):
        order = library.get("resolution_order", [])
        for entry in order if isinstance(order, list) else []:
            if isinstance(entry, dict) and entry.get("path_kind") == "absolute_existing":
                entry["path"], entry["path_kind"] = _portable_library_path(str(entry.get("path", "")), project_root)
    return copied


def collect_r_environment(project_root: Path, rscript: str) -> Dict[str, object]:
    lock_path = project_root / "analysis/revision_v1/r_environment.lock.json"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise EnvironmentLockError("missing R project lock: analysis/revision_v1/r_environment.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    launcher_path = project_root / "analysis/revision_v1/run_with_locked_r.sh"
    command_env = dict(os.environ)
    command_env.update({"R_ENVIRON_USER": "/dev/null", "R_PROFILE_USER": "/dev/null"})
    if rscript != "Rscript":
        command_env["RANKCLOAK_RSCRIPT"] = rscript
    command = (
        ["/bin/bash", str(launcher_path), "-e", R_PROBE]
        if launcher_path.is_file()
        else [rscript, "--vanilla", "-e", R_PROBE]
    )
    probe = _run(command, env=command_env)
    runtime_version = None
    r_platform = None
    packages: List[Dict[str, object]] = []
    libraries: Dict[int, Dict[str, object]] = {}
    if probe["available"] and probe["returncode"] == 0:
        for line in str(probe["stdout"]).splitlines():
            fields = line.split("	")
            if fields[0] == "RUNTIME" and len(fields) >= 3:
                runtime_version, r_platform = fields[1], fields[2]
            elif fields[0] == "LIBRARY" and len(fields) >= 3:
                library_index = int(fields[1])
                portable, kind = _portable_library_path(fields[2], project_root)
                libraries[library_index] = {
                    "library_index": library_index,
                    "path": portable,
                    "path_kind": kind,
                }
            elif fields[0] == "PACKAGE" and len(fields) >= 5:
                priority = fields[4] or None
                library_index = int(fields[1])
                library = libraries.get(library_index, {})
                path_kind = library.get("path_kind")
                scope = (
                    "project_local" if path_kind == "repository_relative"
                    else "system_or_recommended" if priority in {"base", "recommended"}
                    else "user_or_site"
                )
                packages.append({
                    "name": fields[2],
                    "version": fields[3],
                    "priority": priority,
                    "library_scope": scope,
                    "library_index": library_index,
                    "library_path": library.get("path"),
                    "library_path_kind": path_kind,
                })
    packages.sort(key=lambda row: (str(row["name"]), int(row["library_index"]), str(row["version"])))
    declared_libraries = _declared_r_libraries(lock, project_root)
    roles_by_path = {str(entry["path"]): entry.get("role") for entry in declared_libraries}
    for library in libraries.values():
        library["role"] = roles_by_path.get(str(library.get("path")), "r_runtime_default")
    for package in packages:
        package["library_role"] = libraries.get(int(package["library_index"]), {}).get("role")
    project_status = inspect_r_project_library(project_root, lock)
    required_resolution = resolve_required_r_packages(lock, packages)
    expected_version = str(lock.get("r", {}).get("version")) if isinstance(lock.get("r"), dict) else None
    errors: List[str] = []
    if not probe["available"]:
        errors.append("Rscript unavailable")
    elif probe["returncode"] != 0:
        errors.append("locked R launcher probe failed with return code {}".format(probe["returncode"]))
    if runtime_version != expected_version:
        errors.append("R runtime mismatch: expected {}, found {}".format(expected_version, runtime_version))
    errors.extend("required_package_{}".format(value) for value in required_resolution["errors"])
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_available": bool(probe["available"] and probe["returncode"] == 0),
        "runtime_version": runtime_version,
        "runtime_platform": r_platform,
        "required_runtime_version": expected_version,
        "runtime_exact": runtime_version == expected_version,
        "startup_files_disabled_during_probe": True,
        "declared_lock_relative_path": "analysis/revision_v1/r_environment.lock.json",
        "declared_lock_sha256": sha256_file(lock_path),
        "declared_lock": _portable_r_lock(lock, project_root),
        "declared_library_resolution": declared_libraries,
        "launcher_relative_path": "analysis/revision_v1/run_with_locked_r.sh",
        "launcher_sha256": sha256_file(launcher_path) if launcher_path.is_file() else None,
        "launcher_probe_returncode": probe.get("returncode"),
        "library_paths": [libraries[index] for index in sorted(libraries)],
        "project_library_inventory": project_status,
        "required_package_resolution": required_resolution,
        "observed_package_count": len(packages),
        "observed_packages_sha256": sha256_bytes(canonical_json_bytes(packages)),
        "observed_packages": packages,
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "automatic_installation": False,
    }

def _parse_lscpu() -> Dict[str, object]:
    probe = _run(["lscpu", "-J"])
    if not probe["available"] or probe["returncode"] != 0:
        return {"available": False}
    try:
        value = json.loads(str(probe["stdout"]))
    except json.JSONDecodeError:
        return {"available": False, "error": "invalid lscpu JSON"}
    raw = {
        str(item.get("field", "")).rstrip(":"): item.get("data")
        for item in value.get("lscpu", []) if isinstance(item, dict)
    }
    selected = (
        "Architecture", "CPU(s)", "On-line CPU(s) list", "Vendor ID", "Model name",
        "Thread(s) per core", "Core(s) per socket", "Socket(s)", "L1d cache",
        "L1i cache", "L2 cache", "L3 cache", "NUMA node(s)",
    )
    return {"available": True, "fields": {key: raw.get(key) for key in selected}}


def _memory_inventory() -> Dict[str, object]:
    probe = _run(["free", "-b"])
    if not probe["available"] or probe["returncode"] != 0:
        return {"available": False}
    for line in str(probe["stdout"]).splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":") == "Mem" and len(fields) >= 2:
            return {"available": True, "total_bytes": int(fields[1])}
    return {"available": False, "error": "Mem row missing"}


def _gpu_inventory() -> Tuple[List[Dict[str, object]], Optional[str]]:
    query = _run([
        "nvidia-smi", "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows: List[Dict[str, object]] = []
    if query["available"] and query["returncode"] == 0:
        for parsed in csv.reader(str(query["stdout"]).splitlines()):
            values = [value.strip() for value in parsed]
            if len(values) == 5:
                rows.append({
                    "uuid": values[0], "name": values[1], "pci_bus_id": values[2],
                    "memory_total_mib": int(values[3]), "driver_version": values[4],
                })
    rows.sort(key=lambda row: str(row["pci_bus_id"]))
    banner = _run(["nvidia-smi"])
    host_cuda = None
    if banner["available"] and banner["returncode"] == 0:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", str(banner["stdout"]))
        host_cuda = match.group(1) if match else None
    return rows, host_cuda


def _torch_cuda_inventory() -> Dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__}
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "runtime_index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": "{}.{}".format(props.major, props.minor),
            })
    return {
        "available": True,
        "torch_version": str(torch.__version__),
        "torch_compiled_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_runtime_available": bool(torch.cuda.is_available()),
        "runtime_devices": devices,
    }


def _llama_cpp_inventory() -> Dict[str, object]:
    try:
        from rankcloak.model_io import preload_pip_cuda_libraries

        loaded_paths = [Path(value) for value in preload_pip_cuda_libraries()]
        import llama_cpp
        from llama_cpp import llama_cpp as api
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__}
    raw = api.llama_print_system_info()
    package_root = Path(llama_cpp.__file__).resolve().parent
    native = []
    for path in sorted(package_root.rglob("*.so")):
        if path.is_file() and not path.is_symlink():
            native.append({
                "path_within_python_package": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    loaded = []
    for path in loaded_paths:
        if path.is_file() and not path.is_symlink():
            loaded.append({"basename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    loaded.sort(key=lambda row: str(row["basename"]))
    return {
        "available": True,
        "python_package_version": getattr(llama_cpp, "__version__", None),
        "system_info": raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw),
        "gpu_offload_supported": bool(api.llama_supports_gpu_offload()),
        "native_libraries": native,
        "preloaded_pip_cuda_libraries": loaded,
    }


def _tool_versions() -> List[Dict[str, object]]:
    commands = {
        "git": ["git", "--version"], "gcc": ["gcc", "--version"],
        "g++": ["g++", "--version"], "cmake": ["cmake", "--version"],
        "nvidia-smi": ["nvidia-smi", "--version"], "nvcc": ["nvcc", "--version"],
        "Rscript": ["Rscript", "--version"],
    }
    rows = []
    for name in sorted(commands):
        probe = _run(commands[name])
        combined = str(probe["stdout"]) + "\n" + str(probe["stderr"])
        rows.append({
            "name": name,
            "available": bool(probe["available"] and probe["returncode"] == 0),
            "version_line": _safe_version_text(combined) if probe["available"] else None,
        })
    return rows


def collect_backend_system_environment(selected_gpu_uuid: Optional[str]) -> Dict[str, object]:
    gpu_rows, host_cuda = _gpu_inventory()
    selected_matches = [row for row in gpu_rows if row["uuid"] == selected_gpu_uuid] if selected_gpu_uuid else []
    os_release: Dict[str, str] = {}
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        for line in release_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                    os_release[key] = value.strip().strip('"')
    return {
        "schema_version": SCHEMA_VERSION,
        "operating_system": {
            "release": os_release,
            "kernel_release": platform.release(),
            "kernel_version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": _parse_lscpu(),
        "memory": _memory_inventory(),
        "gpu_inventory": gpu_rows,
        "selected_execution_gpu_uuid": selected_gpu_uuid,
        "selected_gpu_resolves_exactly_once": len(selected_matches) == 1,
        "host_driver_advertised_cuda_version": host_cuda,
        "torch_cuda": _torch_cuda_inventory(),
        "llama_cpp_backend": _llama_cpp_inventory(),
        "tool_versions": _tool_versions(),
    }


def build_determinism_record(selected_gpu_uuid: Optional[str]) -> Dict[str, object]:
    required = dict(REQUIRED_CUDA_ENVIRONMENT)
    required["CUDA_VISIBLE_DEVICES"] = selected_gpu_uuid
    observed = {name: os.environ.get(name) for name in OBSERVED_ENVIRONMENT_NAMES}
    return {
        "schema_version": SCHEMA_VERSION,
        "backend_semantics": "single_model_serial_llama_cpp_exact_token_replay",
        "required_for_gpu_runs": required,
        "required_for_locked_r_runs": dict(R_REQUIRED_ENVIRONMENT),
        "observed_while_snapshot_was_built": observed,
        "observed_values_are_not_substitutes_for_launch_requirements": True,
        "model_loading_policy": {
            "one_model_loaded_at_a_time": True,
            "rank_replay_n_batch": 1,
            "rank_replay_n_ubatch": 1,
            "tokenizer_source": "embedded_gguf",
            "gpu_selection_by_exact_uuid": True,
        },
        "seed_policy": "seeds_and_trial identities are frozen in configs/revision_v1",
    }


def _safe_model_pin(model: Mapping[str, object], verification: Optional[Mapping[str, object]]) -> Dict[str, object]:
    kept = (
        "model_id", "family", "architecture_scale", "repo_id", "revision",
        "upstream_repo_id", "upstream_revision", "license", "filename",
        "relative_path", "quantization", "artifact_size_bytes", "artifact_sha256", "pin_status",
    )
    result = {key: model.get(key) for key in kept if key in model}
    result["model_weight_included"] = False
    if verification is not None:
        result["local_verification"] = dict(verification)
    return result


def _load_pinned_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise EnvironmentLockError(
            "required {} is missing, non-regular, or a symlink: {}".format(label, path)
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentLockError("could not parse {}: {}".format(label, exc)) from exc
    if not isinstance(value, dict):
        raise EnvironmentLockError("{} must contain a JSON object".format(label))
    return value


def _verify_self_hash(value: Mapping[str, object], field: str, label: str) -> None:
    recorded = value.get(field)
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise EnvironmentLockError("{} has invalid {}".format(label, field))
    unhashed = dict(value)
    unhashed.pop(field, None)
    observed = sha256_bytes(canonical_json_bytes(unhashed))
    if observed != recorded:
        raise EnvironmentLockError("{} self-hash mismatch".format(label))


def _verify_manifest_file_rows(
    project_root: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    label: str,
) -> None:
    root = project_root.resolve()
    for row in rows:
        if not isinstance(row, Mapping):
            raise EnvironmentLockError("{} has a non-object file row".format(label))
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise EnvironmentLockError("{} has an invalid file path".format(label))
        candidate = Path(raw_path)
        candidate = candidate if candidate.is_absolute() else manifest_path.parent / candidate
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise EnvironmentLockError("{} file escapes project root".format(label)) from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise EnvironmentLockError("{} listed file is missing or unsafe: {}".format(label, raw_path))
        expected_size = row.get("size_bytes", row.get("bytes"))
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise EnvironmentLockError("{} listed file size mismatch: {}".format(label, raw_path))
        if sha256_file(candidate) != row.get("sha256"):
            raise EnvironmentLockError("{} listed file SHA-256 mismatch: {}".format(label, raw_path))


def _verify_derived_validation_manifest(
    project_root: Path, manifest_path: Path, value: Mapping[str, object], relative: str
) -> Dict[str, object]:
    if "smoke_v3_preprocessed_v1" in relative:
        if value.get("manifest_type") != "revision_preprocessing_outputs":
            raise EnvironmentLockError("unexpected smoke-v3 preprocessing manifest")
        invariants = value.get("invariants")
        fidelity = invariants.get("payload_fidelity_contract") if isinstance(invariants, dict) else None
        if not isinstance(fidelity, dict) or fidelity.get("contract_version") != "payload_fidelity_v2":
            raise EnvironmentLockError("preprocessing payload-fidelity contract mismatch")
        rows = value.get("outputs")
        if not isinstance(rows, list) or sha256_bytes(canonical_json_bytes(rows)) != value.get("outputs_sha256"):
            raise EnvironmentLockError("preprocessing output manifest array hash mismatch")
    elif "smoke_v3_integrity_v1" in relative:
        if value.get("artifact_type") != "smoke_v3_exploratory_integrity_output_manifest":
            raise EnvironmentLockError("unexpected smoke-v3 integrity manifest")
        if value.get("confirmatory_pooling_eligible") is not False:
            raise EnvironmentLockError("smoke-v3 integrity output permits confirmatory pooling")
        rows = value.get("files")
    elif "smoke_v3_statistics_v3" in relative:
        if value.get("schema_version") != "1.0":
            raise EnvironmentLockError("unexpected smoke-v3 statistics manifest")
        outputs = value.get("outputs")
        rows = list(outputs.values()) if isinstance(outputs, dict) else None
        config = value.get("statistics_config")
        if not isinstance(config, dict):
            raise EnvironmentLockError("statistics configuration pin is missing")
        _verify_manifest_file_rows(project_root, manifest_path, [config], relative)
    elif "smoke_v3_reports_v3" in relative:
        if value.get("artifact_type") != "report_output_manifest":
            raise EnvironmentLockError("unexpected smoke-v3 report manifest")
        if "do not hand-edit" not in str(value.get("generator_notice", "")).lower():
            raise EnvironmentLockError("smoke-v3 report generator notice is missing")
        rows = value.get("files")
    elif "smoke_v3_theory_v2" in relative:
        if value.get("artifact_type") != "rankcloak_capacity_quality_theory_validation":
            raise EnvironmentLockError("unexpected smoke-v3 theory manifest")
        rows = value.get("tables")
    else:
        raise EnvironmentLockError("unknown derived validation manifest: {}".format(relative))
    if not isinstance(rows, list) or not rows:
        raise EnvironmentLockError("{} has no output file rows".format(relative))
    _verify_manifest_file_rows(project_root, manifest_path, rows, relative)

    directory_rows: List[Dict[str, object]] = []
    for path in sorted(manifest_path.parent.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise EnvironmentLockError("symlink in derived validation directory: {}".format(path))
        if not path.is_file():
            continue
        directory_rows.append(
            {
                "path": path.relative_to(manifest_path.parent).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "directory_file_count": len(directory_rows),
        "directory_files_sha256": sha256_bytes(canonical_json_bytes(directory_rows)),
    }


def collect_validation_result_pins(project_root: Path) -> List[Dict[str, object]]:
    # Pin validation/forensic artifacts without promoting them to scientific evidence.
    pins: List[Dict[str, object]] = []
    for relative, role in VALIDATION_RESULT_PATHS:
        path = project_root / relative
        value = _load_pinned_json(path, relative)
        derived_pin: Optional[Dict[str, object]] = None
        if "TOKENIZER_PREFLIGHT_MANIFEST" in relative:
            if value.get("status") != "pass":
                raise EnvironmentLockError("tokenizer preflight is not passing")
            if value.get("protocol_contract_revision") != "payload_fidelity_v2":
                raise EnvironmentLockError("tokenizer preflight protocol revision mismatch")
            if value.get("result_schema_revision") != "payload_aware_result_v2":
                raise EnvironmentLockError("tokenizer preflight result schema mismatch")
            counts = value.get("counts")
            summaries = value.get("model_summaries")
            if not isinstance(counts, dict) or counts.get("failure_count") != 0:
                raise EnvironmentLockError("tokenizer preflight records failures")
            if counts.get("payloads_per_model") != 480:
                raise EnvironmentLockError("tokenizer preflight payload count mismatch")
            if not isinstance(summaries, list) or {
                row.get("model_id") for row in summaries if isinstance(row, dict)
            } != set(_REVISION_MODEL_IDS):
                raise EnvironmentLockError("tokenizer preflight model set mismatch")
            _verify_self_hash(value, "preflight_manifest_sha256", relative)
        elif "/invalidations/" in relative:
            if value.get("scientific_status") != "invalidated_not_for_pooling":
                raise EnvironmentLockError("invalidation registry entry is not fail-closed")
            _verify_self_hash(value, "invalidation_manifest_sha256", relative)
        elif "/incurred_charges/" in relative:
            if value.get("artifact_role") != "charge_only_not_rate_evidence":
                raise EnvironmentLockError("legacy ledger role is not charge-only")
            if value.get("charge_only_not_rate_evidence") is not True:
                raise EnvironmentLockError("legacy ledger is not marked charge-only")
            if value.get("scientific_evidence_allowed") is not False:
                raise EnvironmentLockError("legacy ledger permits scientific evidence")
            if value.get("rate_evidence_allowed") is not False:
                raise EnvironmentLockError("legacy ledger permits rate evidence")
            _verify_self_hash(value, "legacy_incurred_ledger_sha256", relative)
        elif relative.endswith(("compute_projection_v2.json", "compute_projection_165h_v2.json")):
            if value.get("input_status") != "complete":
                raise EnvironmentLockError("compute projection inputs are incomplete")
            policy = value.get("evidence_policy")
            if not isinstance(policy, dict) or policy.get("protocol_contract_revision") != "payload_fidelity_v2":
                raise EnvironmentLockError("compute projection protocol policy mismatch")
            audit = value.get("combined_incurred_charge_audit")
            if not isinstance(audit, dict) or not str(audit.get("status", "")).startswith("ok"):
                raise EnvironmentLockError("compute projection incurred-charge audit failed")
            _verify_self_hash(value, "projection_sha256", relative)
            if relative.endswith("compute_projection_165h_v2.json") and (
                value.get("projection_sha256")
                != "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
                or value.get("budget_gpu_hours") != 165.0
                or value.get("decision")
                != {
                    "go": True,
                    "reason": "All required smoke evidence is complete and the conservative upper projection is within the approved ceiling.",
                    "status": "go_within_budget",
                }
            ):
                raise EnvironmentLockError(
                    "authorized 165-hour compute projection contract mismatch"
                )
        elif relative.endswith("detector_live_readonly_audit_20260813T0145Z.json"):
            decision = value.get("decision")
            durable = value.get("durable_preceding_state")
            contract = value.get("execution_contract")
            if (
                value.get("schema_version")
                != "rankcloak-detector-live-readonly-audit-v1"
                or not isinstance(decision, dict)
                or decision.get("credible_completion_within_four_additional_hours")
                is not False
                or decision.get("disposition")
                != "stop_only_uncheckpointed_detector_after_disarming_blind_supervisor_retry"
                or not isinstance(durable, dict)
                or int(durable.get("completed", -1)) != 38448
                or int(durable.get("failures", -1)) != 0
                or float(durable.get("cumulative_actual_gpu_hours", -1.0))
                != 61.549997679508685
                or durable.get("detector_output_directory_was_empty") is not True
                or not isinstance(contract, dict)
                or int(contract.get("total_executions", -1)) != 56
                or contract.get("execution_order")
                != "split_outer_detector_inner_serial"
            ):
                raise EnvironmentLockError(
                    "detector replacement audit contract mismatch"
                )
        elif relative.endswith(
            "detector_live_readonly_audit_addendum_20260813T0151Z.json"
        ):
            evidence = value.get("additional_read_only_evidence")
            projection = value.get("runtime_projection")
            stop = value.get("safe_stop")
            base = value.get("base_audit")
            if (
                value.get("schema_version")
                != "rankcloak-detector-live-readonly-audit-addendum-v1"
                or value.get("evidence_role")
                != "operational_diagnostic_not_scientific_evidence"
                or value.get("confirmatory_pooling_permitted") is not False
                or not isinstance(base, dict)
                or base.get("sha256")
                != "1f4bd079e7133805efb729c9fd4ed55838eeb7d396e0daabc1de20405addfcf7"
                or not isinstance(evidence, dict)
                or int(evidence.get("process_rchar_bytes", -1)) != 843674074
                or evidence.get("second_transformer_preload_verification_completed")
                is not False
                or evidence.get("detector_output_directory_remained_empty")
                is not True
                or not isinstance(projection, dict)
                or projection.get("credible_completion_within_four_additional_hours")
                is not False
                or float(
                    projection.get("remaining_deberta_hours_row_scaled_approximate", -1)
                )
                != 623.0
                or not isinstance(stop, dict)
                or stop.get("event")
                != "detector_atomic_run_stopped_for_checkpointed_replacement"
                or int(stop.get("preceding_completed_units", -1)) != 38448
                or float(stop.get("cumulative_actual_gpu_hours", -1.0))
                != 61.549997679508685
                or stop.get("scientific_retry_charged") is not False
                or stop.get("preceding_results_preserved") is not True
            ):
                raise EnvironmentLockError(
                    "detector replacement audit addendum contract mismatch"
                )
        elif "/smoke_v3_" in relative:
            derived_pin = _verify_derived_validation_manifest(
                project_root, path, value, relative
            )
        elif relative.endswith("run_identity.json"):
            if value.get("protocol_contract_revision") != "payload_fidelity_v2":
                raise EnvironmentLockError("{} protocol revision mismatch".format(relative))
            if value.get("result_schema_revision") != "payload_aware_result_v2":
                raise EnvironmentLockError("{} result schema mismatch".format(relative))
            expected_model_id = Path(relative).parent.name
            artifacts = value.get("model_artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 1:
                raise EnvironmentLockError("{} model artifact cardinality mismatch".format(relative))
            artifact = artifacts[0]
            configured = artifact.get("configured_model") if isinstance(artifact, dict) else None
            verification = artifact.get("verification") if isinstance(artifact, dict) else None
            if not isinstance(configured, dict) or configured.get("model_id") != expected_model_id:
                raise EnvironmentLockError("{} configured model identity mismatch".format(relative))
            if not isinstance(verification, dict) or verification.get("model_id") != expected_model_id:
                raise EnvironmentLockError("{} verified model identity mismatch".format(relative))
            if verification.get("status") != "ok" or verification.get("sha256_checked") is not True:
                raise EnvironmentLockError("{} model verification is not complete".format(relative))
            if configured.get("artifact_sha256") != verification.get("actual_sha256"):
                raise EnvironmentLockError("{} model SHA-256 mismatch".format(relative))
            study_id = str(value.get("study_id", ""))
            if "smoke_v3" not in study_id or expected_model_id not in study_id:
                raise EnvironmentLockError("{} study identity mismatch".format(relative))
            if "/heldout_evaluator/" in relative and "no_confirmatory_pooling" not in study_id:
                raise EnvironmentLockError("{} held-out smoke role mismatch".format(relative))
            _verify_self_hash(value, "run_identity_sha256", relative)
        pin = {
            "path": relative,
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "semantic_status": "verified",
        }
        if derived_pin is not None:
            pin.update(derived_pin)
        pins.append(pin)
    return pins


def collect_scientific_pins(project_root: Path, verify_model_files: bool) -> Dict[str, object]:
    sys.path.insert(0, str(project_root))
    try:
        from rankcloak.revision_artifacts import file_sha256, verify_directory_manifest
        from rankcloak.revision_config import verify_model_artifact_pins
        from rankcloak.revision_payloads import (
            REVISION_CORPUS_ID, REVISION_CORPUS_SHA256, REVISION_DERIVATION_VERSION,
            REVISION_PAYLOAD_CLASSES, REVISION_PUBLIC_SEED_MATERIAL,
            generate_revision_v1_payloads, revision_payload_records, validate_revision_corpus,
        )
        from rankcloak.revision_runner import PROTOCOL_CONTRACT_REVISION, RESULT_SCHEMA_REVISION
    finally:
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)

    config_dir = project_root / "configs/revision_v1"
    config_manifest_path = config_dir / "config_manifest.json"
    config_manifest = json.loads(config_manifest_path.read_text(encoding="utf-8"))
    config_report = verify_directory_manifest(config_dir, config_manifest, require_no_extra_files=True)
    if config_report["status"] != "ok":
        raise EnvironmentLockError("frozen config manifest failed: {}".format("; ".join(config_report["errors"])))

    payloads = generate_revision_v1_payloads()
    corpus_report = validate_revision_corpus(payloads, expected_sha256=REVISION_CORPUS_SHA256)
    if corpus_report["status"] != "ok":
        raise EnvironmentLockError("frozen corpus failed: {}".format("; ".join(corpus_report["errors"])))
    records = revision_payload_records(payloads, include_payload_text=True)
    jsonl = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    release_payload_path = project_root / "release_inputs/revision_v1/revision_payloads.jsonl"
    release_payload_manifest_path = project_root / "release_inputs/revision_v1/PAYLOAD_MANIFEST.json"
    release_payload_manifest = _load_pinned_json(
        release_payload_manifest_path, "public release payload manifest"
    )
    expected_jsonl_sha256 = sha256_bytes(jsonl)
    if not release_payload_path.is_file() or release_payload_path.is_symlink():
        raise EnvironmentLockError("public release payload file is missing or unsafe")
    if sha256_file(release_payload_path) != expected_jsonl_sha256:
        raise EnvironmentLockError("public release payload bytes differ from frozen corpus")
    if release_payload_manifest.get("manifest_type") != "revision_payload_corpus":
        raise EnvironmentLockError("unexpected public release payload manifest")
    if release_payload_manifest.get("corpus_sha256") != REVISION_CORPUS_SHA256:
        raise EnvironmentLockError("public release payload corpus hash mismatch")
    if release_payload_manifest.get("payload_file_sha256") != expected_jsonl_sha256:
        raise EnvironmentLockError("public release payload file pin mismatch")
    if release_payload_manifest.get("payload_count") != len(payloads):
        raise EnvironmentLockError("public release payload count mismatch")
    release_validation = release_payload_manifest.get("validation")
    if not isinstance(release_validation, dict) or release_validation.get("status") != "ok":
        raise EnvironmentLockError("public release payload validation is not passing")

    model_config_path = config_dir / "models.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    verification = verify_model_artifact_pins(
        model_config, project_root=project_root, verify_sha256=verify_model_files
    )
    verification_by_id = {row["model_id"]: row for row in verification["records"]}
    if verify_model_files and verification["status"] != "ok":
        raise EnvironmentLockError("model artifact verification failed: {}".format("; ".join(verification["errors"])))
    models = [
        _safe_model_pin(model, verification_by_id.get(model.get("model_id")))
        for model in model_config.get("models", []) if isinstance(model, dict)
    ]
    if PROTOCOL_CONTRACT_REVISION != "payload_fidelity_v2":
        raise EnvironmentLockError("unexpected protocol contract revision")
    if RESULT_SCHEMA_REVISION != "payload_aware_result_v2":
        raise EnvironmentLockError("unexpected result schema revision")

    source_relatives = set(SCIENTIFIC_SOURCE_PATHS)
    closure_paths = set(DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS)
    if (
        len(closure_paths) != len(DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS)
        or not closure_paths.issubset(source_relatives)
    ):
        raise EnvironmentLockError(
            "detector release-closure source contract is duplicated or unpinned"
        )
    for directory_relative in SCIENTIFIC_SOURCE_DIRECTORIES:
        directory = project_root / directory_relative
        if not directory.is_dir() or directory.is_symlink():
            raise EnvironmentLockError(
                "required scientific source directory missing or unsafe: {}".format(
                    directory_relative
                )
            )
        for path in sorted(directory.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink():
                raise EnvironmentLockError(
                    "symlink in scientific source directory: {}".format(
                        path.relative_to(project_root).as_posix()
                    )
                )
            if not path.is_file():
                continue
            relative_path = path.relative_to(project_root)
            if set(relative_path.parts) & SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_COMPONENTS:
                continue
            if path.suffix.lower() in SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_SUFFIXES:
                continue
            if path.name.lower() in SCIENTIFIC_SOURCE_DIRECTORY_FORBIDDEN_NAMES:
                raise EnvironmentLockError(
                    "raw or identifying human data may not be environment-pinned: {}".format(
                        relative_path.as_posix()
                    )
                )
            source_relatives.add(relative_path.as_posix())

    sources = []
    for relative in sorted(source_relatives):
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise EnvironmentLockError("required environment/source pin missing: {}".format(relative))
        sources.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    validation_results = collect_validation_result_pins(project_root)
    projection = _load_pinned_json(
        project_root / "results/revision_v1/compute_projection_165h_v2.json",
        "authorized compute projection",
    )
    projection_decision = {
        "path": "results/revision_v1/compute_projection_165h_v2.json",
        "projection_sha256": projection.get("projection_sha256"),
        "budget_gpu_hours": projection.get("budget_gpu_hours"),
        "decision": projection.get("decision"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "manifest_path": "configs/revision_v1/config_manifest.json",
            "manifest_sha256": file_sha256(config_manifest_path),
            "files_sha256": config_manifest.get("files_sha256"),
            "verified_file_count": config_report.get("verified_file_count"),
            "verification_status": config_report.get("status"),
        },
        "corpus": {
            "corpus_id": REVISION_CORPUS_ID,
            "derivation_version": REVISION_DERIVATION_VERSION,
            "class_order": list(REVISION_PAYLOAD_CLASSES),
            "payload_count": len(payloads),
            "class_counts": corpus_report["class_counts"],
            "canonical_record_array_sha256": corpus_report["corpus_sha256"],
            "expected_canonical_record_array_sha256": REVISION_CORPUS_SHA256,
            "canonical_jsonl_sha256": expected_jsonl_sha256,
            "public_seed_material_sha256": sha256_bytes(REVISION_PUBLIC_SEED_MATERIAL),
            "payload_values_included_in_environment_bundle": False,
            "release_export": {
                "payload_path": "release_inputs/revision_v1/revision_payloads.jsonl",
                "payload_sha256": sha256_file(release_payload_path),
                "manifest_path": "release_inputs/revision_v1/PAYLOAD_MANIFEST.json",
                "manifest_sha256": sha256_file(release_payload_manifest_path),
                "validation_status": release_validation.get("status"),
            },
        },
        "models": models,
        "model_file_sha256_verification_requested": bool(verify_model_files),
        "model_verification_status": verification["status"] if verify_model_files else "size_checked_only",
        "model_weights_included": False,
        "protocol_contract": {
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "payload_tokenization_contract": "literal_utf8_no_special_tokens_reversible_space_prefix_v2",
            "prompt_context_contract": "actual_bos_only_removal_first_real_token_retention_v2",
        },
        "source_files": sources,
        "source_files_sha256": sha256_bytes(canonical_json_bytes(sources)),
        "detector_release_closure_source_contract": {
            "paths": list(DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS),
            "paths_sha256": sha256_bytes(
                canonical_json_bytes(list(DETECTOR_RELEASE_CLOSURE_SOURCE_PATHS))
            ),
            "all_paths_content_pinned_in_source_files": True,
            "generated_detector_results_copied_into_environment_bundle": False,
            "model_weights_included": False,
        },
        "source_directory_contract": {
            "included_directories": list(SCIENTIFIC_SOURCE_DIRECTORIES),
            "cache_components_excluded": sorted(SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_COMPONENTS),
            "bytecode_suffixes_excluded": sorted(SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_SUFFIXES),
            "raw_or_identifying_human_data_forbidden": True,
            "human_results_included": False,
            "human_material_role": "pre_recruitment_planning_not_empirical_results",
        },
        "validation_result_artifacts": validation_results,
        "validation_result_artifacts_sha256": sha256_bytes(canonical_json_bytes(validation_results)),
        "validation_result_policy": {
            "confirmatory_pooling_permitted": False,
            "invalidated_shard_outputs_included": False,
            "legacy_artifacts_rate_evidence_allowed": False,
        },
        "current_compute_gate_decision": projection_decision,
    }


def collect_snapshot(
    project_root: Path,
    selected_gpu_uuid: Optional[str],
    rscript: str = "Rscript",
    verify_model_files: bool = False,
) -> Dict[str, object]:
    python, requirements = collect_python_environment(project_root)
    r_environment = collect_r_environment(project_root, rscript)
    backend = collect_backend_system_environment(selected_gpu_uuid)
    scientific = collect_scientific_pins(project_root, verify_model_files)
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": "rankcloak_scientific_reports_revision_v1_environment",
        "status": "complete" if r_environment["status"] == "complete" else "incomplete_r_project_library",
        "offline_observation_only": True,
        "network_access_used": False,
        "installation_performed": False,
        "model_weights_copied": False,
        "python": python,
        "requirements_lock_bytes": requirements,
        "r": r_environment,
        "backend_system": backend,
        "determinism": build_determinism_record(selected_gpu_uuid),
        "scientific_pins": scientific,
    }


def _readme(snapshot: Mapping[str, object]) -> bytes:
    status = snapshot.get("status")
    return (
        "# RankCloak revision-v1 environment snapshot\n\n"
        "This directory is a content-hashed, offline observation of the software, "
        "R lock, llama.cpp/CUDA backend, hardware, deterministic launch variables, "
        "payload-fidelity-v2 scientific inputs, the authorized 165-GPU-hour gate, "
        "final supervisor/join/detector/R/report/manuscript/release tooling, "
        "the detector checkpoint/equivalence release specification, validators, "
        "tests, and documentation, validation/forensic audit artifacts, "
        "and participant-free human-study planning identities used for the Scientific "
        "Reports revision.\n\n"
        "Snapshot status: `{}`. No dependency was installed, no network service was "
        "contacted, and no model weight is copied here. `requirements-lock.txt` records "
        "exact installed versions and distribution metadata fingerprints are in "
        "`python_environment.json`; it is not a cross-platform wheel-hash lock.\n\n"
        "The declared project-local R library is checked separately from user/system "
        "libraries. An incomplete status must not be represented as a completed mixed-effects "
        "analysis environment. Obtain dependencies and model files only under their applicable "
        "licenses, then verify them locally with the commands in `REPRODUCE.md`. "
        "Exploratory smoke artifacts cannot be pooled as confirmatory evidence; the "
        "legacy incurred-charge ledger is neither scientific nor runtime-rate evidence. "
        "The detector release closure retains self-hashed final manifest/status/receipt/ledger "
        "identities plus checkpoint, equivalence, benchmark, incorporation, and event-log "
        "provenance through the confirmatory release index; execution locks, active permits, "
        "quarantined stale permits, caches, and model weights remain excluded. "
        "Confirmatory result bytes are releasable only through the separately generated, "
        "self-hashed confirmatory release index after all completion gates pass.\n"
    ).format(status).encode("utf-8")


def _reproduce(selected_gpu_uuid: Optional[str]) -> bytes:
    gpu = selected_gpu_uuid or "<EXACT_GPU_UUID>"
    return (
        "# Offline environment verification\n\n"
        "These commands do not download, install, run experiments, or publish a release.\n\n"
        "```bash\n"
        ".venv/bin/python scripts/build_revision_environment_lock.py \\\n  --output-dir environment/revision_v1 --check\n"
        ".venv/bin/python scripts/build_revision_environment_lock.py \\\n  --output-dir environment/revision_v1 --check --verify-model-files\n"
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={} \\\nCUDA_LAUNCH_BLOCKING=1 GGML_CUDA_DISABLE_GRAPHS=1 \\\nGGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \\\nCUBLAS_WORKSPACE_CONFIG=:4096:8 \\\n.venv/bin/python scripts/run_revision_matrix.py --help\n"
        "analysis/revision_v1/run_with_locked_r.sh --version\n"
        ".venv/bin/python scripts/build_revision_confirmatory_release_index.py --dry-run\n"
        "```\n\n"
        "The final three commands are preflight demonstrations only. Running experiments or "
        "installing the missing project-local R packages is a separate controlled action.\n"
    ).format(gpu).encode("utf-8")


def render_bundle(snapshot: Mapping[str, object]) -> Dict[str, bytes]:
    requirements = snapshot.get("requirements_lock_bytes")
    if not isinstance(requirements, (bytes, bytearray)):
        raise EnvironmentLockError("snapshot requirements_lock_bytes must be bytes")
    files: Dict[str, bytes] = {
        "README.md": _readme(snapshot),
        "REPRODUCE.md": _reproduce(snapshot.get("backend_system", {}).get("selected_execution_gpu_uuid")),
        "requirements-lock.txt": bytes(requirements),
        "python_environment.json": json_bytes(snapshot["python"]),
        "r_environment.json": json_bytes(snapshot["r"]),
        "backend_cuda_hardware.json": json_bytes(snapshot["backend_system"]),
        "determinism.json": json_bytes(snapshot["determinism"]),
        "scientific_pins.json": json_bytes(snapshot["scientific_pins"]),
        "bundle_status.json": json_bytes({
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot["snapshot_id"],
            "status": snapshot["status"],
            "offline_observation_only": True,
            "network_access_used": False,
            "installation_performed": False,
            "model_weights_copied": False,
            "external_publication_performed": False,
        }),
    }
    checksums = "".join(
        "{}  {}\n".format(sha256_bytes(files[path]), path) for path in sorted(files)
    ).encode("utf-8")
    files["CHECKSUMS.sha256"] = checksums
    records = [
        {"path": path, "size_bytes": len(files[path]), "sha256": sha256_bytes(files[path])}
        for path in sorted(files)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "rankcloak_revision_environment_file_set",
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_status": snapshot["status"],
        "file_count": len(records),
        "files_sha256": sha256_bytes(canonical_json_bytes(records)),
        "files": records,
    }
    files["environment_manifest.json"] = json_bytes(manifest)
    return files


def _scan_for_forbidden_content(files: Mapping[str, bytes]) -> None:
    for relative, content in files.items():
        path = Path(relative)
        if path.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
            raise EnvironmentLockError("model-weight file forbidden in environment bundle: {}".format(relative))
        text = content.decode("utf-8", errors="ignore")
        if re.search(r"(?:file://)?/(?:home|Users)/[^\s\"']+", text):
            raise EnvironmentLockError("absolute user path leaked into {}".format(relative))
        if "-----BEGIN PRIVATE KEY-----" in text:
            raise EnvironmentLockError("private-key signature forbidden in {}".format(relative))


def write_bundle(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise EnvironmentLockError("output directory already exists; use --check or choose a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".{}.".format(output_dir.name), dir=str(output_dir.parent)))
    try:
        for relative, content in sorted(files.items()):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        _scan_for_forbidden_content(files)
        if output_dir.exists() or output_dir.is_symlink():
            raise EnvironmentLockError("output appeared during staging; refusing overwrite")
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_bundle(output_dir: Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    errors: List[str] = []
    manifest_path = output_dir / "environment_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "errors": ["cannot load environment manifest: {}".format(exc)]}
    listed = set()
    records = manifest.get("files", [])
    if not isinstance(records, list):
        return {"status": "error", "errors": ["manifest files must be a list"]}
    for row in records:
        if not isinstance(row, dict):
            errors.append("invalid manifest row")
            continue
        relative = str(row.get("path"))
        listed.add(relative)
        path = output_dir / relative
        if not path.is_file() or path.is_symlink():
            errors.append("missing or invalid file: {}".format(relative))
            continue
        if path.stat().st_size != row.get("size_bytes"):
            errors.append("size mismatch: {}".format(relative))
        if sha256_file(path) != row.get("sha256"):
            errors.append("SHA-256 mismatch: {}".format(relative))
    expected_records_hash = sha256_bytes(canonical_json_bytes(records))
    if expected_records_hash != manifest.get("files_sha256"):
        errors.append("manifest file-set hash mismatch")
    if len(records) != manifest.get("file_count"):
        errors.append("manifest file_count mismatch")
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file() and not path.is_symlink()
    }
    expected = listed | {"environment_manifest.json"}
    if actual != expected:
        errors.append("bundle file-set mismatch: extra={} missing={}".format(sorted(actual - expected), sorted(expected - actual)))
    content = {relative: (output_dir / relative).read_bytes() for relative in actual if (output_dir / relative).is_file()}
    try:
        _scan_for_forbidden_content(content)
    except EnvironmentLockError as exc:
        errors.append(str(exc))
    checksum_path = output_dir / "CHECKSUMS.sha256"
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match:
                errors.append("invalid CHECKSUMS row")
                continue
            digest, relative = match.groups()
            path = output_dir / relative
            if not path.is_file() or sha256_file(path) != digest:
                errors.append("CHECKSUMS mismatch: {}".format(relative))
    else:
        errors.append("CHECKSUMS.sha256 missing")
    return {"status": "ok" if not errors else "error", "verified_file_count": len(records), "errors": errors}


def _without_local_model_verification(value: Mapping[str, object]) -> Dict[str, object]:
    copied = json.loads(json.dumps(value))
    copied.pop("model_file_sha256_verification_requested", None)
    copied.pop("model_verification_status", None)
    for model in copied.get("models", []):
        if isinstance(model, dict):
            model.pop("local_verification", None)
    return copied


def verify_live_scientific_pins(output_dir: Path, project_root: Path, verify_model_files: bool) -> Dict[str, object]:
    expected = json.loads((Path(output_dir) / "scientific_pins.json").read_text(encoding="utf-8"))
    actual = collect_scientific_pins(project_root, verify_model_files)
    left = _without_local_model_verification(expected)
    right = _without_local_model_verification(actual)
    errors = [] if left == right else ["live scientific inputs differ from scientific_pins.json"]
    if verify_model_files and actual.get("model_verification_status") != "ok":
        errors.append("live model artifact SHA-256 verification failed")
    return {
        "status": "ok" if not errors else "error",
        "model_file_sha256_verification_requested": bool(verify_model_files),
        "model_verification_status": actual.get("model_verification_status"),
        "errors": errors,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-uuid", default=None, help="exact predeclared execution GPU UUID")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--verify-model-files", action="store_true", help="stream and verify local model hashes; weights are never copied")
    parser.add_argument("--check", action="store_true", help="verify bundle plus live scientific pins without writing")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if args.check:
        bundle_report = verify_bundle(output_dir)
        live_report = verify_live_scientific_pins(output_dir, project_root, args.verify_model_files)
        report = {"bundle": bundle_report, "live_scientific_pins": live_report}
        report["status"] = "ok" if bundle_report["status"] == live_report["status"] == "ok" else "error"
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["status"] == "ok" else 2
    snapshot = collect_snapshot(
        project_root=project_root,
        selected_gpu_uuid=args.gpu_uuid,
        rscript=args.rscript,
        verify_model_files=args.verify_model_files,
    )
    files = render_bundle(snapshot)
    write_bundle(output_dir, files)
    report = verify_bundle(output_dir)
    print(json.dumps({
        "status": report["status"],
        "output_dir": output_dir.relative_to(project_root).as_posix() if output_dir.is_relative_to(project_root) else str(output_dir),
        "snapshot_status": snapshot["status"],
        "model_weights_copied": False,
        "network_access_used": False,
        "installation_performed": False,
        "verified_file_count": report.get("verified_file_count"),
    }, sort_keys=True, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
