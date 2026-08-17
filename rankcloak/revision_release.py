"""Offline, allowlist-driven assembly of the RankCloak code-and-data archive.

This module only stages local files.  It contains no HTTP client, repository
upload, Zenodo integration, DOI minting, or publication operation.  Every
assembled package remains externally unpublished even when its local content
passes the final-readiness checks.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_artifacts import canonical_json_sha256, file_sha256
from .revision_release_index import (
    CONFIRMATORY_ARTIFACT_SPECS,
    ConfirmatoryReleaseIndexError,
    confirmatory_artifact_path_policy,
    verify_confirmatory_release_index,
    verify_staged_confirmatory_release_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "release" / "revision_v1_template" / "release_spec.json"
RELEASE_SCHEMA_VERSION = "1.0"
EXTERNAL_STATUS = "draft_external_action_prohibited"
FINAL_CONTENT_STATUS = "final_ready_offline_candidate"
DRAFT_CONTENT_STATUS = "draft_incomplete_offline_candidate"
PACKAGE_MANIFEST_TYPE = "rankcloak_code_data_offline_release_candidate"
PORTABLE_EVIDENCE_SCHEMA = "rankcloak-portable-final-evidence-package-v1"
HOME_PREFIX = "/" + "home/"
CONFIRMATORY_INDEX_DESTINATION = "CONFIRMATORY_ARTIFACT_INDEX.json"
CONFIRMATORY_SOURCES = {row[1] for row in CONFIRMATORY_ARTIFACT_SPECS}

ARTIFACT_GROUPS = (
    "source_code",
    "configs",
    "public_payload_corpus",
    "raw_results",
    "processed_results",
    "statistics_outputs",
    "figure_table_outputs",
    "human_materials",
    "environment_inputs",
    "documentation",
)
DEFAULT_REQUIRED_GROUPS = ARTIFACT_GROUPS[:-1]

EVIDENCE_ROLES = {
    "confirmatory_scientific_evidence",
    "authoritative_final_evidence",
    "exploratory_validation_not_for_confirmatory_pooling",
    "exploratory_compute_gate_not_for_confirmatory_pooling",
    "forensic_invalidated_not_for_pooling",
    "forensic_charge_only_not_rate_evidence",
    "supporting_methodological_material",
    "human_study_preregistration_material_no_responses",
    "environment_reproduction_input",
    "documentation_not_scientific_result",
}

FORBIDDEN_COMPONENTS = {
    ".git",
    ".venv",
    "venv",
    "models",
    "external_sources",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
}
CACHE_COMPONENTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
}
MODEL_WEIGHT_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".ckpt",
    ".pth",
    ".pt",
    ".onnx",
    ".h5",
}
MODEL_WEIGHT_NAMES = {
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks"}
SECRET_NAMES = {
    ".env",
    ".env.local",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    ".netrc",
}
RAW_HUMAN_PATTERNS = (
    "raw_human",
    "raw_responses",
    "responses.csv",
    "responses.jsonl",
    "participant_ids",
    "participant_identifiers",
    "participant_data",
    "survey_export",
    "prolific_export",
    "mturk_export",
    "signed_consent",
    "email_addresses",
    "contact_information",
)
TEXT_SUFFIXES = {
    "",
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".r",
    ".jl",
    ".sh",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".cff",
    ".csv",
    ".tsv",
    ".tex",
    ".bib",
}
PLACEHOLDER_PATTERNS = (
    re.compile(r"\{\{[A-Z][A-Z0-9_ -]*\}\}"),
    re.compile(r"<<[A-Z][A-Z0-9_ -]*>>"),
    re.compile(r"\[INSERT(?: [^\]]+)?\]", re.IGNORECASE),
    re.compile(r"\b(?:REPLACE_ME|CHANGEME|YOUR_[A-Z0-9_]+)\b"),
    re.compile(r"\b(?:TBD|TO_BE_DETERMINED)\b"),
)
SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN ([A-Z ]*PRIVATE KEY)-----[\s\S]{80,}"
        r"-----END \1-----"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
FORBIDDEN_ACTION_KEYS = {
    "zenodo",
    "deposit",
    "publish",
    "publication_action",
    "upload",
    "network_action",
    "api_token",
    "access_token",
}


class RevisionReleaseError(RuntimeError):
    """Raised when a release candidate would violate the offline contract."""


def _load_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RevisionReleaseError("Cannot load release specification: {}".format(path)) from exc
    if not isinstance(value, dict):
        raise RevisionReleaseError("Release specification must be a JSON object")
    return value


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _validate_no_external_action(spec: Mapping[str, object]) -> None:
    forbidden = sorted(set(_walk_keys(spec)) & FORBIDDEN_ACTION_KEYS)
    if forbidden:
        raise RevisionReleaseError(
            "Release specification requests a prohibited external action: {}".format(
                ", ".join(forbidden)
            )
        )
    status = str(spec.get("release_status", EXTERNAL_STATUS))
    if status != EXTERNAL_STATUS:
        raise RevisionReleaseError(
            "release_status must remain {!r}; this assembler cannot publish".format(
                EXTERNAL_STATUS
            )
        )


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RevisionReleaseError("{} must be a non-empty safe relative path".format(label))
    return path


def _within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _path_exclusion(relative: PurePosixPath, group: str) -> Optional[Tuple[str, str]]:
    lower_parts = tuple(part.lower() for part in relative.parts)
    name = lower_parts[-1] if lower_parts else ""
    suffix = PurePosixPath(name).suffix.lower()
    component_hits = set(lower_parts) & FORBIDDEN_COMPONENTS
    if component_hits:
        hit = sorted(component_hits)[0]
        severity = "informational" if hit in CACHE_COMPONENTS else "blocking"
        return "forbidden_path_component:{}".format(hit), severity
    if suffix in MODEL_WEIGHT_SUFFIXES or name in MODEL_WEIGHT_NAMES:
        return "model_weight", "blocking"
    if suffix in SECRET_SUFFIXES or name in SECRET_NAMES:
        return "secret_or_private_key_file", "blocking"
    joined = "/".join(lower_parts)
    for pattern in RAW_HUMAN_PATTERNS:
        if pattern in joined:
            return "participant_identifier_or_raw_human_data", "blocking"
    if group == "human_materials" and suffix in {".parquet", ".xlsx", ".sav"}:
        return "unsupported_human_data_binary", "blocking"
    if name in {".ds_store", "thumbs.db"}:
        return "cache_or_os_metadata", "informational"
    return None


def _read_text_if_applicable(path: Path) -> Optional[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RevisionReleaseError("Cannot read release input {}".format(path)) from exc
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _content_findings(
    path: Path, scan_placeholders: bool = True
) -> Tuple[List[str], List[str]]:
    text = _read_text_if_applicable(path)
    if text is None:
        return [], []
    placeholder_patterns = (
        PLACEHOLDER_PATTERNS[1:]
        if path.suffix.lower() in {".bib", ".tex"}
        else PLACEHOLDER_PATTERNS
    )
    placeholders = (
        sorted(
            {
                match.group(0)
                for pattern in placeholder_patterns
                for match in pattern.finditer(text)
            }
        )
        if scan_placeholders
        else []
    )
    secrets = sorted(
        {pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)}
    )
    return placeholders[:20], secrets


def validate_release_spec(spec: Mapping[str, object]) -> Dict[str, object]:
    _validate_no_external_action(spec)
    if spec.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise RevisionReleaseError("Unsupported release specification schema_version")
    metadata = spec.get("metadata")
    if not isinstance(metadata, dict):
        raise RevisionReleaseError("metadata must be an object")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RevisionReleaseError("artifacts must be an object")
    unknown_groups = sorted(set(map(str, artifacts)) - set(ARTIFACT_GROUPS))
    if unknown_groups:
        raise RevisionReleaseError(
            "Unknown artifact groups: {}".format(", ".join(unknown_groups))
        )
    for group, entries in artifacts.items():
        if not isinstance(entries, list):
            raise RevisionReleaseError("artifacts.{} must be a list".format(group))
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RevisionReleaseError("Artifact entry must be an object")
            _safe_relative_path(entry.get("source"), "artifact source")
            _safe_relative_path(entry.get("destination"), "artifact destination")
            if "sha256" in entry and not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
                raise RevisionReleaseError("Artifact sha256 must be 64 lowercase hex characters")
            if set(entry) - {
                "source", "destination", "required", "sha256", "note",
                "evidence_role", "portable_paths", "exclude_paths",
            }:
                raise RevisionReleaseError(
                    "Unsupported keys in artifacts.{}[{}]".format(group, index)
                )
            if not isinstance(entry.get("portable_paths", False), bool):
                raise RevisionReleaseError(
                    "portable_paths must be true or false in artifacts.{}[{}]".format(
                        group, index
                    )
                )
            exclude_paths = entry.get("exclude_paths", [])
            if not isinstance(exclude_paths, list):
                raise RevisionReleaseError(
                    "exclude_paths must be a list in artifacts.{}[{}]".format(
                        group, index
                    )
                )
            for excluded in exclude_paths:
                _safe_relative_path(excluded, "artifact exclude path")
            role = entry.get("evidence_role")
            if role is not None and str(role) not in EVIDENCE_ROLES:
                raise RevisionReleaseError(
                    "Unsupported evidence_role in artifacts.{}[{}]".format(group, index)
                )
            source_text = str(entry.get("source", ""))
            if group == "raw_results" and role is None:
                raise RevisionReleaseError(
                    "Every raw_results entry requires an explicit evidence_role"
                )
            if source_text == "results/revision_v1/primary" or source_text.startswith("results/revision_v1/primary/"):
                raise RevisionReleaseError(
                    "The invalidated legacy primary tree may not be allowlisted"
                )
            if "/smoke_v2" in source_text:
                raise RevisionReleaseError(
                    "Legacy smoke_v2 artifacts may not be allowlisted; use the charge ledger"
                )
            required_role = None
            if "/invalidations/" in source_text:
                required_role = "forensic_invalidated_not_for_pooling"
            elif "/incurred_charges/" in source_text:
                required_role = "forensic_charge_only_not_rate_evidence"
            elif "/smoke_v3" in source_text:
                required_role = "exploratory_validation_not_for_confirmatory_pooling"
            elif source_text.endswith(("compute_projection_v2.json", "compute_projection_165h_v2.json")):
                required_role = "exploratory_compute_gate_not_for_confirmatory_pooling"
            if required_role is not None and role != required_role:
                raise RevisionReleaseError(
                    "{} must have evidence_role {}".format(source_text, required_role)
                )
    gated_entries = [
        (group, entry)
        for group in ARTIFACT_GROUPS
        for entry in artifacts.get(group, [])
        if entry.get("evidence_role") == "confirmatory_scientific_evidence"
    ]
    confirmatory_index = spec.get("confirmatory_artifact_index")
    if gated_entries:
        if confirmatory_index in (None, ""):
            raise RevisionReleaseError(
                "confirmatory_artifact_index is required for confirmatory artifacts"
            )
        index_relative = _safe_relative_path(
            confirmatory_index, "confirmatory_artifact_index"
        ).as_posix()
        observed = [
            (
                group,
                str(entry.get("source")),
                str(entry.get("destination")),
                str(entry.get("evidence_role")),
            )
            for group, entry in gated_entries
        ]
        expected = [tuple(row[:4]) for row in CONFIRMATORY_ARTIFACT_SPECS]
        if observed != expected:
            raise RevisionReleaseError(
                "Confirmatory allowlist must exactly match the verified release map"
            )
        index_entries = [
            (group, entry)
            for group in ARTIFACT_GROUPS
            for entry in artifacts.get(group, [])
            if str(entry.get("source", "")) == index_relative
        ]
        if len(index_entries) != 1:
            raise RevisionReleaseError(
                "The confirmatory artifact index must be allowlisted exactly once"
            )
        index_group, index_entry = index_entries[0]
        if (
            index_group != "raw_results"
            or index_entry.get("destination") != CONFIRMATORY_INDEX_DESTINATION
            or index_entry.get("evidence_role") != "supporting_methodological_material"
        ):
            raise RevisionReleaseError(
                "The confirmatory artifact index allowlist identity is invalid"
            )
        policy = spec.get("evidence_partition_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("confirmatory_artifact_index_required") is not True
            or policy.get("partial_or_unverified_confirmatory_outputs_included") is not False
            or policy.get("invalidated_shard_bytes_included") is not False
            or policy.get("protocol_contract_revision") != "payload_fidelity_v2"
            or policy.get("result_schema_revision") != "payload_aware_result_v2"
        ):
            raise RevisionReleaseError(
                "Confirmatory evidence partition policy is missing or unsafe"
            )
    elif confirmatory_index not in (None, ""):
        raise RevisionReleaseError(
            "confirmatory_artifact_index is declared without confirmatory artifacts"
        )
    authoritative = spec.get("authoritative_evidence_package")
    if authoritative is not None:
        if not isinstance(authoritative, dict):
            raise RevisionReleaseError(
                "authoritative_evidence_package must be an object"
            )
        allowed = {
            "source", "manifest", "manifest_sha256", "status",
            "required_external_labels",
        }
        if set(authoritative) - allowed:
            raise RevisionReleaseError(
                "Unsupported authoritative_evidence_package keys"
            )
        source = _safe_relative_path(
            authoritative.get("source"), "authoritative evidence source"
        )
        manifest = _safe_relative_path(
            authoritative.get("manifest"), "authoritative evidence manifest"
        )
        if manifest.parent != source:
            raise RevisionReleaseError(
                "Authoritative evidence manifest must be directly under its source"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(authoritative.get("manifest_sha256", ""))
        ):
            raise RevisionReleaseError(
                "Authoritative evidence manifest requires a pinned SHA-256"
            )
        if authoritative.get("status") != "passed":
            raise RevisionReleaseError(
                "Authoritative evidence status must be passed"
            )
        labels = authoritative.get("required_external_labels", [])
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise RevisionReleaseError(
                "required_external_labels must be a non-empty unique string list"
            )
    if not isinstance(spec.get("tracked_sources_only", False), bool):
        raise RevisionReleaseError("tracked_sources_only must be true or false")
    required_groups = spec.get("required_groups", list(DEFAULT_REQUIRED_GROUPS))
    if not isinstance(required_groups, list):
        raise RevisionReleaseError("required_groups must be a list")
    invalid_required = sorted(set(map(str, required_groups)) - set(ARTIFACT_GROUPS))
    if invalid_required:
        raise RevisionReleaseError("Unknown required_groups entries")
    if metadata.get("direct_participant_identifiers_included") is True:
        raise RevisionReleaseError("Participant identifiers may never be staged")
    if metadata.get("raw_human_response_data_included") is True:
        raise RevisionReleaseError("Raw human response data may never be staged")
    doi = metadata.get("doi")
    if doi not in (None, "") and not DOI_PATTERN.fullmatch(str(doi)):
        raise RevisionReleaseError("A supplied DOI is not syntactically valid")
    commands = spec.get("reproduction_commands", [])
    if not isinstance(commands, list) or not all(isinstance(row, dict) for row in commands):
        raise RevisionReleaseError("reproduction_commands must be a list of objects")
    third_party = spec.get("third_party", [])
    if not isinstance(third_party, list) or not all(isinstance(row, dict) for row in third_party):
        raise RevisionReleaseError("third_party must be a list of objects")
    return {
        "required_groups": [str(value) for value in required_groups],
        "artifact_entry_count": sum(len(value) for value in artifacts.values()),
        "confirmatory_gate_required": bool(gated_entries),
        "confirmatory_index_path": (
            str(confirmatory_index) if gated_entries else None
        ),
        "authoritative_evidence_required": authoritative is not None,
    }


def _entry_files(
    project_root: Path,
    group: str,
    entry: Mapping[str, object],
    tracked_files: Optional[set[str]] = None,
) -> Dict[str, object]:
    source_relative = _safe_relative_path(entry["source"], "artifact source")
    destination_relative = _safe_relative_path(entry["destination"], "artifact destination")
    source = (project_root / Path(*source_relative.parts)).resolve(strict=False)
    if not _within_root(project_root.resolve(), source):
        raise RevisionReleaseError("Artifact source escapes project root")
    required = bool(entry.get("required", True))
    result: Dict[str, object] = {
        "group": group,
        "source": source_relative.as_posix(),
        "destination": destination_relative.as_posix(),
        "required": required,
        "files": [],
        "missing": False,
        "exclusions": [],
    }
    if entry.get("evidence_role") is not None:
        result["evidence_role"] = str(entry["evidence_role"])
    if entry.get("note") is not None:
        result["note"] = str(entry["note"])
    excluded_relative_paths = tuple(
        _safe_relative_path(value, "artifact exclude path").as_posix()
        for value in entry.get("exclude_paths", [])
    )
    if not source.exists() and not source.is_symlink():
        result["missing"] = True
        return result
    candidates: List[Tuple[Path, PurePosixPath]] = []
    if source.is_symlink():
        result["exclusions"].append(
            {"path": source_relative.as_posix(), "reason": "symlink", "severity": "blocking"}
        )
        return result
    if source.is_file():
        candidates.append((source, destination_relative))
    elif source.is_dir():
        for path in sorted(source.rglob("*"), key=lambda value: value.as_posix()):
            nested = PurePosixPath(path.relative_to(source).as_posix())
            candidates.append((path, destination_relative / nested))
    else:
        result["exclusions"].append(
            {"path": source_relative.as_posix(), "reason": "special_file", "severity": "blocking"}
        )
        return result
    for path, destination in candidates:
        project_relative = PurePosixPath(path.relative_to(project_root).as_posix())
        nested_relative = (
            PurePosixPath(path.name)
            if source.is_file()
            else PurePosixPath(path.relative_to(source).as_posix())
        )
        nested_text = nested_relative.as_posix()
        if any(
            nested_text == excluded
            or nested_text.startswith(excluded.rstrip("/") + "/")
            for excluded in excluded_relative_paths
        ):
            result["exclusions"].append(
                {
                    "path": project_relative.as_posix(),
                    "reason": "archive_policy_omission",
                    "severity": "informational",
                }
            )
            continue
        if tracked_files is not None and project_relative.as_posix() not in tracked_files:
            result["exclusions"].append(
                {
                    "path": project_relative.as_posix(),
                    "reason": "untracked_source_not_included",
                    "severity": "informational",
                }
            )
            continue
        if path.is_symlink():
            result["exclusions"].append(
                {"path": project_relative.as_posix(), "reason": "symlink", "severity": "blocking"}
            )
            continue
        if (
            source_relative.as_posix() in CONFIRMATORY_SOURCES
            and entry.get("evidence_role") == "confirmatory_scientific_evidence"
        ):
            nested = (
                PurePosixPath(path.name)
                if source.is_file()
                else PurePosixPath(path.relative_to(source).as_posix())
            )
            try:
                decision = confirmatory_artifact_path_policy(
                    source_relative.as_posix(), Path(*nested.parts)
                )
            except ConfirmatoryReleaseIndexError as exc:
                raise RevisionReleaseError(str(exc)) from exc
            if decision.startswith("exclude_"):
                result["exclusions"].append(
                    {
                        "path": project_relative.as_posix(),
                        "reason": decision,
                        "severity": "informational",
                    }
                )
                continue
        if path.is_dir():
            continue
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            result["exclusions"].append(
                {"path": project_relative.as_posix(), "reason": "special_file", "severity": "blocking"}
            )
            continue
        exclusion = _path_exclusion(project_relative, group)
        if exclusion:
            reason, severity = exclusion
            result["exclusions"].append(
                {"path": project_relative.as_posix(), "reason": reason, "severity": severity}
            )
            continue
        placeholders, secrets = _content_findings(
            path, scan_placeholders=group in {"documentation", "human_materials"}
        )
        if secrets:
            result["exclusions"].append(
                {
                    "path": project_relative.as_posix(),
                    "reason": "high_confidence_secret_signature",
                    "severity": "blocking",
                    "signatures": secrets,
                }
            )
            continue
        actual_hash = file_sha256(path)
        expected_hash = entry.get("sha256")
        if expected_hash is not None and actual_hash != expected_hash:
            raise RevisionReleaseError(
                "Pinned artifact hash mismatch: {}".format(project_relative)
            )
        file_record = {
            "group": group,
            "source": project_relative.as_posix(),
            "destination": destination.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
            "placeholders": placeholders,
        }
        if entry.get("evidence_role") is not None:
            file_record["evidence_role"] = str(entry["evidence_role"])
        if entry.get("portable_paths") is True:
            file_record["portable_paths"] = True
        result["files"].append(file_record)
    return result



def _tracked_repository_files(project_root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RevisionReleaseError(
            "tracked_sources_only requires a readable Git index"
        ) from exc
    return {
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    }


def _portable_manifest_reference(
    raw: object, project_root: Path
) -> tuple[PurePosixPath, Path]:
    text = str(raw).replace("\\", "/")
    candidate = Path(text)
    root = project_root.resolve()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if _within_root(root, resolved):
            relative = PurePosixPath(resolved.relative_to(root).as_posix())
        else:
            parts = PurePosixPath(text).parts
            try:
                marker = len(parts) - 1 - tuple(reversed(parts)).index(root.name)
            except ValueError as exc:
                raise RevisionReleaseError(
                    "Authoritative manifest path is not repository-bound: {}".format(
                        text
                    )
                ) from exc
            relative = _safe_relative_path(
                PurePosixPath(*parts[marker + 1 :]).as_posix(),
                "authoritative manifest reference",
            )
    else:
        relative = _safe_relative_path(text, "authoritative manifest reference")
    return relative, root / Path(*relative.parts)


def verify_authoritative_evidence_package(
    spec: Mapping[str, object],
    project_root: Path,
    included_files: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    declaration = spec.get("authoritative_evidence_package")
    if declaration is None:
        return {
            "required": False,
            "status": "not_required",
            "manifest_path": None,
            "manifest_sha256": None,
            "included_package_file_count": 0,
            "verified_external_reference_count": 0,
        }
    if not isinstance(declaration, Mapping):
        raise RevisionReleaseError("authoritative_evidence_package is malformed")
    root = Path(project_root).resolve()
    source_relative = _safe_relative_path(
        declaration["source"], "authoritative evidence source"
    )
    manifest_relative = _safe_relative_path(
        declaration["manifest"], "authoritative evidence manifest"
    )
    manifest_path = root / Path(*manifest_relative.parts)
    report: Dict[str, object] = {
        "required": True,
        "status": "missing",
        "manifest_path": manifest_relative.as_posix(),
        "manifest_sha256": None,
        "included_package_file_count": 0,
        "verified_external_reference_count": 0,
    }
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return report
    observed_manifest_sha = file_sha256(manifest_path)
    if observed_manifest_sha != declaration["manifest_sha256"]:
        raise RevisionReleaseError(
            "Authoritative final-package manifest hash mismatch"
        )
    manifest = _load_json(manifest_path)
    unsigned = dict(manifest)
    signature = unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version")
        != "rankcloak-final-experiment-package-index-v1"
        or manifest.get("status") != declaration["status"]
        or signature != canonical_json_sha256(unsigned)
    ):
        raise RevisionReleaseError(
            "Authoritative final-package manifest failed its signed status check"
        )
    package_rows = manifest.get("package_files")
    if not isinstance(package_rows, list):
        raise RevisionReleaseError(
            "Authoritative final-package manifest lacks package_files"
        )
    declared_package: Dict[str, Mapping[str, object]] = {}
    for row in package_rows:
        if not isinstance(row, Mapping):
            raise RevisionReleaseError("Malformed final-package file identity")
        relative = _safe_relative_path(
            row.get("path"), "authoritative package file path"
        ).as_posix()
        if relative in declared_package:
            raise RevisionReleaseError(
                "Duplicate authoritative package file path: {}".format(relative)
            )
        declared_package[relative] = row
    source_prefix = source_relative.as_posix().rstrip("/") + "/"
    source_entries = [
        entry
        for group in ARTIFACT_GROUPS
        for entry in spec.get("artifacts", {}).get(group, [])
        if str(entry.get("source", "")).rstrip("/")
        == source_relative.as_posix().rstrip("/")
    ]
    if len(source_entries) != 1:
        raise RevisionReleaseError(
            "Authoritative evidence source must have exactly one artifact entry"
        )
    excluded = tuple(
        _safe_relative_path(value, "authoritative package exclusion").as_posix()
        for value in source_entries[0].get("exclude_paths", [])
    )
    expected_package = {
        path
        for path in declared_package
        if not any(
            path == omitted or path.startswith(omitted.rstrip("/") + "/")
            for omitted in excluded
        )
    }
    included_nested_paths: set[str] = set()
    for row in included_files:
        source_path = str(row.get("source", ""))
        if not source_path.startswith(source_prefix):
            continue
        nested = source_path[len(source_prefix) :]
        declared = declared_package.get(nested)
        if declared is None:
            raise RevisionReleaseError(
                "Included final-package file is absent from the sealed index: {}".format(
                    source_path
                )
            )
        if (
            row.get("sha256") != declared.get("sha256")
            or row.get("size_bytes") != declared.get("size_bytes")
        ):
            raise RevisionReleaseError(
                "Included final-package file differs from the sealed index: {}".format(
                    source_path
                )
            )
        included_nested_paths.add(nested)
    if included_nested_paths != expected_package:
        raise RevisionReleaseError(
            "Authoritative final-package coverage differs: missing={} extra={}".format(
                sorted(expected_package - included_nested_paths),
                sorted(included_nested_paths - expected_package),
            )
        )
    external_rows = manifest.get("external_references")
    if not isinstance(external_rows, list):
        raise RevisionReleaseError(
            "Authoritative final-package manifest lacks external_references"
        )
    by_label = {
        str(row.get("label")): row
        for row in external_rows
        if isinstance(row, Mapping)
    }
    verified_external = 0
    for label in declaration["required_external_labels"]:
        row = by_label.get(str(label))
        if row is None:
            raise RevisionReleaseError(
                "Required final-package external reference is absent: {}".format(
                    label
                )
            )
        relative, referenced = _portable_manifest_reference(
            row.get("path"), project_root
        )
        if (
            not referenced.is_file()
            or referenced.is_symlink()
            or referenced.stat().st_size != row.get("size_bytes")
            or file_sha256(referenced) != row.get("sha256")
        ):
            raise RevisionReleaseError(
                "Required final-package external reference differs: {}".format(
                    relative
                )
            )
        verified_external += 1
    report.update(
        {
            "status": "verified_complete",
            "manifest_sha256": observed_manifest_sha,
            "included_package_file_count": len(included_nested_paths),
            "verified_external_reference_count": verified_external,
            "source_package_file_count": manifest.get("summary", {}).get(
                "package_file_count"
            ),
            "source_external_reference_count": manifest.get("summary", {}).get(
                "external_reference_count"
            ),
        }
    )
    return report


def resolve_release_inputs(
    spec: Mapping[str, object], project_root: Path = PROJECT_ROOT
) -> Dict[str, object]:
    validation = validate_release_spec(spec)
    root = Path(project_root).resolve()
    tracked_files = (
        _tracked_repository_files(root)
        if spec.get("tracked_sources_only") is True
        else None
    )
    gate_report: Dict[str, object] = {
        "required": bool(validation["confirmatory_gate_required"]),
        "status": "not_required",
        "index_path": validation.get("confirmatory_index_path"),
        "manifest_sha256": None,
        "artifact_count": 0,
    }
    verified_by_source: Dict[str, Mapping[str, object]] = {}
    if validation["confirmatory_gate_required"]:
        index_path = root / str(validation["confirmatory_index_path"])
        if not index_path.exists() and not index_path.is_symlink():
            gate_report["status"] = "missing"
        else:
            try:
                verified = verify_confirmatory_release_index(index_path, root)
            except ConfirmatoryReleaseIndexError as exc:
                raise RevisionReleaseError(
                    "Confirmatory artifact index verification failed: {}".format(exc)
                ) from exc
            gate_report.update({
                "status": verified["status"],
                "manifest_sha256": verified["manifest_sha256"],
                "artifact_count": verified["artifact_count"],
            })
            verified_by_source = {
                str(row["source"]): row for row in verified["artifacts"]
            }
    resolutions: List[Dict[str, object]] = []
    destinations: Dict[str, str] = {}
    for group in ARTIFACT_GROUPS:
        for entry in spec.get("artifacts", {}).get(group, []):
            source_text = str(entry.get("source", ""))
            if (
                source_text in CONFIRMATORY_SOURCES
                and entry.get("evidence_role") == "confirmatory_scientific_evidence"
                and not verified_by_source
            ):
                resolution = {
                    "group": group,
                    "source": source_text,
                    "destination": str(entry.get("destination")),
                    "required": bool(entry.get("required", True)),
                    "evidence_role": entry.get("evidence_role"),
                    "note": entry.get("note"),
                    "files": [],
                    "missing": True,
                    "exclusions": [],
                    "verification_status": "confirmatory_index_missing",
                }
            else:
                resolution = _entry_files(
                    root, group, entry, tracked_files=tracked_files
                )
            if source_text in verified_by_source:
                indexed = verified_by_source[source_text]
                observed_files = [
                    {
                        "path": row["source"],
                        "size_bytes": row["size_bytes"],
                        "sha256": row["sha256"],
                    }
                    for row in resolution["files"]
                ]
                if (
                    group != indexed.get("group")
                    or entry.get("destination") != indexed.get("destination")
                    or entry.get("evidence_role") != indexed.get("evidence_role")
                    or observed_files != indexed.get("files")
                ):
                    raise RevisionReleaseError(
                        "Allowlisted confirmatory bytes differ from verified index: {}".format(
                            source_text
                        )
                    )
                resolution["verification_status"] = "verified_complete"
            resolutions.append(resolution)
            for file_record in resolution["files"]:
                destination = str(file_record["destination"])
                if destination in destinations:
                    raise RevisionReleaseError(
                        "Two allowlist inputs map to {}: {} and {}".format(
                            destination, destinations[destination], file_record["source"]
                        )
                    )
                destinations[destination] = str(file_record["source"])
    missing = [
        {
            "group": row["group"],
            "source": row["source"],
            "required": row["required"],
        }
        for row in resolutions
        if row["missing"]
    ]
    exclusions = [
        {"group": row["group"], "entry_source": row["source"], **excluded}
        for row in resolutions
        for excluded in row["exclusions"]
    ]
    files = [file_record for row in resolutions for file_record in row["files"]]
    group_counts = {
        group: sum(file_record["group"] == group for file_record in files)
        for group in ARTIFACT_GROUPS
    }
    authoritative = verify_authoritative_evidence_package(
        spec, root, files
    )
    return {
        "validation": validation,
        "resolutions": resolutions,
        "files": sorted(files, key=lambda row: str(row["destination"])),
        "missing_artifacts": missing,
        "exclusions": exclusions,
        "group_file_counts": group_counts,
        "total_file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "confirmatory_artifact_verification": gate_report,
        "authoritative_evidence_verification": authoritative,
    }


def _metadata_missing(spec: Mapping[str, object]) -> List[str]:
    metadata = spec["metadata"]
    missing: List[str] = []
    for field in ("title", "version", "description"):
        if not str(metadata.get(field, "")).strip():
            missing.append("metadata.{}".format(field))
    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        missing.append("metadata.creators")
    else:
        for index, creator in enumerate(creators):
            if not isinstance(creator, dict) or not str(creator.get("name", "")).strip():
                missing.append("metadata.creators[{}].name".format(index))
    for field in (
        "direct_participant_identifiers_included",
        "raw_human_response_data_included",
        "human_participant_outcomes_collected",
    ):
        if metadata.get(field) is not False:
            missing.append("metadata.{} must be false".format(field))
    return missing


def _explicit_third_party(spec: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in spec.get("third_party", []):
        row = {str(key): value for key, value in raw.items()}
        if not row.get("id"):
            row["id"] = _stable_inventory_id(row)
        row.setdefault("artifact_included", False)
        row["inventory_source"] = "release_spec"
        rows.append(row)
    return rows


def _stable_inventory_id(row: Mapping[str, object]) -> str:
    return "third_party_{}".format(canonical_json_sha256(row)[:16])


def _configured_model_inventory(project_root: Path) -> List[Dict[str, object]]:
    path = Path(project_root) / "configs" / "revision_v1" / "models.json"
    if not path.is_file():
        return []
    value = _load_json(path)
    rows: List[Dict[str, object]] = []
    for model in value.get("models", []):
        if not isinstance(model, dict):
            continue
        license_name = model.get("license")
        rows.append(
            {
                "id": model.get("model_id"),
                "name": model.get("repo_id"),
                "kind": "model_identifier_only",
                "revision": model.get("revision"),
                "upstream_repo_id": model.get("upstream_repo_id"),
                "upstream_revision": model.get("upstream_revision"),
                "quantization": model.get("quantization"),
                "artifact_sha256": model.get("artifact_sha256"),
                "license": license_name or "license_not_recorded",
                "license_status": "recorded" if license_name else "missing",
                "artifact_included": False,
                "inventory_source": "configs/revision_v1/models.json",
            }
        )
    return rows


def build_third_party_inventory(
    spec: Mapping[str, object], project_root: Path = PROJECT_ROOT
) -> List[Dict[str, object]]:
    automatic = {str(row["id"]): row for row in _configured_model_inventory(project_root)}
    for row in _explicit_third_party(spec):
        identifier = str(row["id"])
        if identifier in automatic:
            merged = dict(automatic[identifier])
            merged.update(row)
            if row.get("license"):
                merged["license_status"] = "recorded_release_spec_override"
            merged["inventory_source"] = "model_config_plus_release_spec"
            automatic[identifier] = merged
        else:
            automatic[identifier] = row
    return [automatic[key] for key in sorted(automatic)]


def _is_exact_lock(file_record: Mapping[str, object], project_root: Path) -> bool:
    name = PurePosixPath(str(file_record["source"])).name.lower()
    if name == "environment_manifest.json":
        return _is_verified_environment_snapshot(file_record, project_root)
    if name.startswith("requirements") and name.endswith(".txt"):
        path = project_root / str(file_record["source"])
        text = _read_text_if_applicable(path) or ""
        requirements = [
            line.strip() for line in text.splitlines()
            if line.strip()
            and not line.lstrip().startswith(("#", "--"))
            and line.strip() not in {"-e .", "--editable ."}
        ]
        return bool(requirements) and all("==" in line for line in requirements)
    if "lock" in name or name in {"poetry.lock", "uv.lock", "pdm.lock"}:
        return Path(project_root, str(file_record["source"])).stat().st_size > 0
    return False


def _environment_lock_source_contract() -> Tuple[
    Tuple[str, ...], Tuple[str, ...], frozenset[str], frozenset[str], frozenset[str]
]:
    """Load the release-side source contract from the pinned environment builder."""

    builder_path = PROJECT_ROOT / "scripts/build_revision_environment_lock.py"
    spec = importlib.util.spec_from_file_location(
        "_rankcloak_release_environment_contract", builder_path
    )
    if spec is None or spec.loader is None:
        raise RevisionReleaseError("Cannot load the environment source contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        tuple(module.SCIENTIFIC_SOURCE_PATHS),
        tuple(module.SCIENTIFIC_SOURCE_DIRECTORIES),
        frozenset(module.SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_COMPONENTS),
        frozenset(module.SCIENTIFIC_SOURCE_DIRECTORY_EXCLUDED_SUFFIXES),
        frozenset(module.SCIENTIFIC_SOURCE_DIRECTORY_FORBIDDEN_NAMES),
    )


def _expected_environment_scientific_source_paths(
    project_root: Path,
) -> Optional[set[str]]:
    try:
        (
            static_paths,
            directories,
            excluded_components,
            excluded_suffixes,
            forbidden_names,
        ) = _environment_lock_source_contract()
    except (OSError, AttributeError, ImportError, RevisionReleaseError, SyntaxError):
        return None
    root = Path(project_root).resolve()
    expected = set(static_paths)
    for directory_relative in directories:
        directory = root / directory_relative
        if not directory.is_dir() or directory.is_symlink():
            return None
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                return None
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if set(relative.parts) & excluded_components:
                continue
            if path.suffix.lower() in excluded_suffixes:
                continue
            if path.name.lower() in forbidden_names:
                return None
            expected.add(relative.as_posix())
    for relative in expected:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            return None
    return expected


def _is_verified_environment_snapshot(
    file_record: Mapping[str, object], project_root: Path
) -> bool:
    # Verify the complete local environment file set without trusting its name.
    manifest_path = Path(project_root) / str(file_record["source"])
    try:
        manifest = _load_json(manifest_path)
    except RevisionReleaseError:
        return False
    if (
        manifest.get("manifest_type")
        != "rankcloak_revision_environment_file_set"
        or manifest.get("snapshot_status") != "complete"
    ):
        return False
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        return False
    if len(rows) != manifest.get("file_count"):
        return False
    if canonical_json_sha256(rows) != manifest.get("files_sha256"):
        return False
    base = manifest_path.parent.resolve()
    required = {
        "CHECKSUMS.sha256",
        "README.md",
        "REPRODUCE.md",
        "backend_cuda_hardware.json",
        "bundle_status.json",
        "determinism.json",
        "python_environment.json",
        "r_environment.json",
        "requirements-lock.txt",
        "scientific_pins.json",
    }
    listed = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        try:
            relative = _safe_relative_path(row.get("path"), "environment manifest path")
        except RevisionReleaseError:
            return False
        relative_text = relative.as_posix()
        if relative_text in listed:
            return False
        listed.add(relative_text)
        path = (base / Path(*relative.parts)).resolve(strict=False)
        if (
            not _within_root(base, path)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            return False
    actual = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != listed | {manifest_path.name} or not required.issubset(listed):
        return False
    try:
        status = _load_json(base / "bundle_status.json")
        python_environment = _load_json(base / "python_environment.json")
        r_environment = _load_json(base / "r_environment.json")
        scientific_pins = _load_json(base / "scientific_pins.json")
    except RevisionReleaseError:
        return False
    if (
        status.get("status") != "complete"
        or status.get("network_access_used") is not False
        or status.get("installation_performed") is not False
        or status.get("model_weights_copied") is not False
        or status.get("external_publication_performed") is not False
    ):
        return False
    packages = python_environment.get("packages")
    if not isinstance(packages, list) or not packages:
        return False
    if any(
        not isinstance(row, dict) or not row.get("name") or not row.get("version")
        for row in packages
    ):
        return False
    if canonical_json_sha256(packages) != python_environment.get("packages_sha256"):
        return False
    requirements_path = base / "requirements-lock.txt"
    if file_sha256(requirements_path) != python_environment.get("requirements_lock_sha256"):
        return False
    if r_environment.get("status") != "complete":
        return False
    if scientific_pins.get("model_weights_included") is not False:
        return False
    source_files = scientific_pins.get("source_files")
    if (
        not isinstance(source_files, list)
        or not source_files
        or scientific_pins.get("source_files_sha256")
        != canonical_json_sha256(source_files)
    ):
        return False
    seen_sources = set()
    for row in source_files:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            return False
        try:
            relative = _safe_relative_path(row.get("path"), "scientific source path")
        except RevisionReleaseError:
            return False
        if relative.as_posix() in seen_sources:
            return False
        seen_sources.add(relative.as_posix())
        source = (Path(project_root) / Path(*relative.parts)).resolve(strict=False)
        if (
            not _within_root(Path(project_root).resolve(), source)
            or not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != row.get("size_bytes")
            or file_sha256(source) != row.get("sha256")
        ):
            return False
    expected_sources = _expected_environment_scientific_source_paths(project_root)
    if expected_sources is None or seen_sources != expected_sources:
        return False
    gate = scientific_pins.get("current_compute_gate_decision")
    expected_decision = {
        "go": True,
        "reason": (
            "All required smoke evidence is complete and the conservative upper "
            "projection is within the approved ceiling."
        ),
        "status": "go_within_budget",
    }
    if (
        not isinstance(gate, dict)
        or gate.get("path") != "results/revision_v1/compute_projection_165h_v2.json"
        or gate.get("projection_sha256")
        != "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
        or gate.get("budget_gpu_hours") != 165.0
        or gate.get("decision") != expected_decision
    ):
        return False
    gate_path = Path(project_root) / str(gate["path"])
    try:
        gate_document = _load_json(gate_path)
    except RevisionReleaseError:
        return False
    unsigned_gate = dict(gate_document)
    recorded_gate_hash = unsigned_gate.pop("projection_sha256", None)
    validation_rows = scientific_pins.get("validation_result_artifacts")
    if (
        recorded_gate_hash != canonical_json_sha256(unsigned_gate)
        or not isinstance(validation_rows, list)
        or scientific_pins.get("validation_result_artifacts_sha256")
        != canonical_json_sha256(validation_rows)
    ):
        return False
    gate_rows = [
        row for row in validation_rows
        if isinstance(row, dict) and row.get("path") == gate["path"]
    ]
    if len(gate_rows) != 1:
        return False
    gate_row = gate_rows[0]
    if (
        gate_path.is_symlink()
        or gate_document.get("projection_sha256") != gate["projection_sha256"]
        or gate_document.get("budget_gpu_hours") != gate["budget_gpu_hours"]
        or gate_document.get("decision") != gate["decision"]
        or gate_document.get("input_status") != "complete"
        or not isinstance(gate_document.get("evidence_policy"), dict)
        or gate_document["evidence_policy"].get("protocol_contract_revision")
        != "payload_fidelity_v2"
        or gate_document["evidence_policy"].get("result_schema_revision")
        != "payload_aware_result_v2"
        or not isinstance(gate_document.get("combined_incurred_charge_audit"), dict)
        or not str(
            gate_document["combined_incurred_charge_audit"].get("status", "")
        ).startswith("ok")
        or gate_row.get("role")
        != "authorized_compute_gate_not_scientific_outcome"
        or gate_row.get("semantic_status") != "verified"
        or gate_row.get("size_bytes") != gate_path.stat().st_size
        or gate_row.get("sha256") != file_sha256(gate_path)
    ):
        return False
    return True


def build_environment_inventory(
    resolution: Mapping[str, object], project_root: Path = PROJECT_ROOT
) -> Dict[str, object]:
    inputs = [
        dict(row) for row in resolution["files"] if row["group"] == "environment_inputs"
    ]
    verified_snapshots = [
        row
        for row in inputs
        if PurePosixPath(str(row["source"])).name.lower()
        == "environment_manifest.json"
        and _is_verified_environment_snapshot(row, Path(project_root))
    ]
    exact = bool(verified_snapshots) or any(
        _is_exact_lock(row, Path(project_root)) for row in inputs
    )
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "python_version_used_by_assembler": platform.python_version(),
        "platform_used_by_assembler": platform.platform(),
        "environment_inputs": inputs,
        "exact_lock_input_present": exact,
        "verified_environment_snapshot_present": bool(verified_snapshots),
        "verified_environment_snapshot_sources": [
            str(row["source"]) for row in verified_snapshots
        ],
        "lock_status": (
            "verified_observed_environment_snapshot_supplied"
            if verified_snapshots
            else "exact_lock_input_supplied"
            if exact
            else "declarative_constraints_only"
        ),
        "notice": "The assembler records supplied inputs but does not install or resolve packages.",
    }


def assess_content_readiness(
    spec: Mapping[str, object],
    resolution: Mapping[str, object],
    third_party: Sequence[Mapping[str, object]],
    environment: Mapping[str, object],
) -> Dict[str, object]:
    blockers: List[Dict[str, object]] = []
    notices: List[Dict[str, object]] = []
    for field in _metadata_missing(spec):
        blockers.append({"code": "missing_metadata", "detail": field})
    required_groups = set(resolution["validation"]["required_groups"])
    for group in sorted(required_groups):
        if int(resolution["group_file_counts"].get(group, 0)) == 0:
            blockers.append({"code": "empty_required_group", "detail": group})
    for missing in resolution["missing_artifacts"]:
        if missing["required"]:
            blockers.append({"code": "missing_required_artifact", "detail": missing})
    for excluded in resolution["exclusions"]:
        if excluded["severity"] == "blocking":
            blockers.append({"code": "prohibited_or_sensitive_artifact", "detail": excluded})
    confirmatory = resolution["confirmatory_artifact_verification"]
    if confirmatory.get("required") and confirmatory.get("status") != "verified_complete":
        blockers.append({
            "code": "missing_verified_confirmatory_artifact_index",
            "detail": {
                "path": confirmatory.get("index_path"),
                "status": confirmatory.get("status"),
            },
        })
    authoritative = resolution["authoritative_evidence_verification"]
    if authoritative.get("required") and authoritative.get("status") != "verified_complete":
        blockers.append(
            {
                "code": "missing_verified_authoritative_evidence_package",
                "detail": {
                    "path": authoritative.get("manifest_path"),
                    "status": authoritative.get("status"),
                },
            }
        )
    for file_record in resolution["files"]:
        if file_record.get("placeholders"):
            blockers.append(
                {
                    "code": "unresolved_placeholder",
                    "detail": {
                        "group": file_record["group"],
                        "path": file_record["source"],
                        "markers": file_record["placeholders"],
                    },
                }
            )
    for row in third_party:
        license_name = str(row.get("license", "")).strip().lower()
        if not license_name or license_name in {
            "unknown", "missing", "license_not_recorded", "not_resolved_offline"
        }:
            finding = {"detail": row.get("id")}
            if row.get("artifact_included") is True:
                blockers.append({"code": "third_party_license_unresolved", **finding})
            else:
                notices.append(
                    {
                        "code": "third_party_license_unresolved_reference_only_bytes_excluded",
                        **finding,
                    }
                )
        if row.get("artifact_included") is True and row.get("kind") == "model_identifier_only":
            blockers.append({"code": "model_artifact_must_not_be_included", "detail": row.get("id")})
    if not environment.get("exact_lock_input_present"):
        blockers.append({"code": "missing_exact_environment_lock", "detail": environment["lock_status"]})
    commands = spec.get("reproduction_commands", [])
    if not commands:
        blockers.append({"code": "missing_reproduction_commands", "detail": None})
    for index, command in enumerate(commands):
        if not str(command.get("command", "")).strip():
            blockers.append({"code": "invalid_reproduction_command", "detail": index})
    ready = not blockers
    publication_blockers: List[Dict[str, object]] = [
        {
            "code": "external_publication_authorization_absent",
            "detail": "This offline assembler cannot upload, deposit, reserve a DOI, or publish.",
        }
    ]
    if spec["metadata"].get("doi") in (None, ""):
        publication_blockers.append({"code": "doi_not_assigned", "detail": None})
    publication_blockers.extend(
        {
            "code": "human_material_placeholder_unresolved",
            "detail": blocker["detail"],
        }
        for blocker in blockers
        if blocker.get("code") == "unresolved_placeholder"
        and isinstance(blocker.get("detail"), dict)
        and blocker["detail"].get("group") == "human_materials"
    )
    return {
        "content_readiness": FINAL_CONTENT_STATUS if ready else DRAFT_CONTENT_STATUS,
        "final_ready": ready,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "notice_count": len(notices),
        "notices": notices,
        "release_status": EXTERNAL_STATUS,
        "external_action_performed": False,
        "doi_minted_or_reserved_by_assembler": False,
        "publication_ready": False,
        "publication_blocker_count": len(publication_blockers),
        "publication_blockers": publication_blockers,
    }


def _artifact_evidence_partitions(resolution: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for entry in resolution["resolutions"]:
        role = entry.get("evidence_role")
        if role is None:
            continue
        rows.append(
            {
                "group": entry["group"],
                "source": entry["source"],
                "destination": entry["destination"],
                "evidence_role": role,
                "required": entry["required"],
                "missing": entry["missing"],
                "included_file_count": len(entry["files"]),
                "note": entry.get("note"),
                "verification_status": entry.get("verification_status"),
            }
        )
    return rows


def plan_release(
    spec: Mapping[str, object], project_root: Path = PROJECT_ROOT
) -> Dict[str, object]:
    resolution = resolve_release_inputs(spec, project_root=project_root)
    third_party = build_third_party_inventory(spec, project_root=project_root)
    environment = build_environment_inventory(resolution, project_root=project_root)
    readiness = assess_content_readiness(spec, resolution, third_party, environment)
    partitions = _artifact_evidence_partitions(resolution)
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "operation": "offline_release_candidate_plan",
        "release_status": EXTERNAL_STATUS,
        "external_action_performed": False,
        "network_access_used": False,
        "resolution": resolution,
        "third_party_inventory": third_party,
        "environment_inventory": environment,
        "artifact_evidence_partitions": partitions,
        "confirmatory_artifact_verification": resolution[
            "confirmatory_artifact_verification"
        ],
        "authoritative_evidence_verification": resolution[
            "authoritative_evidence_verification"
        ],
        "readiness": readiness,
    }


def _write_bytes_no_overwrite(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("Short write to {}".format(path))
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_no_overwrite(path: Path, value: object) -> None:
    _write_bytes_no_overwrite(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )


def _portable_text_bytes(path: Path, project_root: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES or b"\0" in data[:8192]:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    root_text = project_root.as_posix()
    text = text.replace(root_text + "/", "")
    text = text.replace(root_text, ".")
    home_pattern = re.escape(HOME_PREFIX) + r"[^/]+/"
    text = re.sub(
        home_pattern + r"Documents/repos/llm-rankcloak/",
        "",
        text,
    )
    text = re.sub(
        home_pattern + r"Documents/repos/llm-rankcloak",
        ".",
        text,
    )
    text = re.sub(
        home_pattern + r"R/x86_64-pc-linux-gnu-library/4[.]4/",
        "external_environment/R_user_library/4.4/",
        text,
    )
    text = re.sub(
        home_pattern + r"R/x86_64-pc-linux-gnu-library/4[.]4",
        "external_environment/R_user_library/4.4",
        text,
    )
    return text.encode("utf-8")


def _copy_allowlisted_files(
    project_root: Path, staging: Path, files: Sequence[Mapping[str, object]]
) -> None:
    for row in files:
        source = Path(project_root) / str(row["source"])
        destination = staging / Path(*PurePosixPath(str(row["destination"])).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise RevisionReleaseError("Staging destination collision: {}".format(destination))
        if file_sha256(source) != row["sha256"]:
            raise RevisionReleaseError(
                "Allowlisted source changed during assembly: {}".format(row["source"])
            )
        if row.get("portable_paths") is True:
            _write_bytes_no_overwrite(
                destination, _portable_text_bytes(source, project_root)
            )
        else:
            shutil.copyfile(source, destination, follow_symlinks=False)
        os.chmod(destination, 0o644)
        if (
            row.get("portable_paths") is not True
            and file_sha256(destination) != row["sha256"]
        ):
            raise RevisionReleaseError("Copied artifact hash mismatch: {}".format(row["source"]))


def _candidate_reference_path(staging: Path, raw: object) -> Optional[Path]:
    text = str(raw).replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    try:
        relative = _safe_relative_path(text, "portable manifest reference")
    except RevisionReleaseError:
        return None
    candidate = staging / Path(*relative.parts)
    if not _within_root(staging.resolve(), candidate.resolve(strict=False)):
        return None
    return candidate


def _reference_is_omitted(raw: object, omitted: Sequence[str]) -> bool:
    text = str(raw).replace("\\", "/").lstrip("./")
    return any(
        text == value or text.startswith(value.rstrip("/") + "/")
        for value in omitted
    )


def _refresh_portable_json(
    value: object,
    staging: Path,
    omitted: Sequence[str],
) -> object:
    if isinstance(value, list):
        return [
            _refresh_portable_json(item, staging, omitted)
            for item in value
            if not (
                isinstance(item, dict)
                and _reference_is_omitted(item.get("path", ""), omitted)
            )
        ]
    if not isinstance(value, dict):
        return value
    result: Dict[str, object] = {}
    for key, item in value.items():
        if (
            key == "inspection_png_300dpi"
            and isinstance(item, str)
            and _reference_is_omitted(item, omitted)
        ):
            continue
        if key == "outputs" and isinstance(item, dict):
            outputs: Dict[str, object] = {}
            for label, declaration in item.items():
                if (
                    isinstance(declaration, dict)
                    and _reference_is_omitted(declaration.get("path", ""), omitted)
                ):
                    continue
                outputs[str(label)] = _refresh_portable_json(
                    declaration, staging, omitted
                )
            result[key] = outputs
            continue
        result[key] = _refresh_portable_json(item, staging, omitted)
    path_value = result.get("path")
    candidate = (
        _candidate_reference_path(staging, path_value)
        if isinstance(path_value, str)
        else None
    )
    if candidate is not None and candidate.is_file() and not candidate.is_symlink():
        if "sha256" in result:
            result["sha256"] = file_sha256(candidate)
        if "size_bytes" in result:
            result["size_bytes"] = candidate.stat().st_size
        if "bytes" in result:
            result["bytes"] = candidate.stat().st_size
    if result.get("schema_version") == "rankcloak-computational-figure-evidence-v2":
        result["inspection_output_format"] = "inspection_pngs_omitted_from_archive"
        summary = result.get("summary")
        if isinstance(summary, dict):
            summary["rendered_file_count"] = int(summary.get("figure_count", 0))
    return result


def _refresh_portable_manifests(
    staging: Path,
    files: Sequence[Mapping[str, object]],
    omitted: Sequence[str],
) -> None:
    destinations = sorted(
        {
            str(row["destination"])
            for row in files
            if row.get("portable_paths") is True
            and PurePosixPath(str(row["destination"])).suffix.lower() == ".json"
        }
    )
    for _ in range(12):
        changed = False
        for destination in destinations:
            path = staging / Path(*PurePosixPath(destination).parts)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            refreshed = _refresh_portable_json(value, staging, omitted)
            if not isinstance(refreshed, dict):
                continue
            self_hash = None
            for field in ("manifest_sha256", "progress_sha256"):
                if field in refreshed:
                    self_hash = field
                    break
            if self_hash is not None:
                unsigned = dict(refreshed)
                unsigned.pop(self_hash, None)
                refreshed[self_hash] = canonical_json_sha256(unsigned)
            encoded = (
                json.dumps(
                    refreshed,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            if path.read_bytes() != encoded:
                path.write_bytes(encoded)
                os.chmod(path, 0o644)
                changed = True
        if not changed:
            return
    raise RevisionReleaseError(
        "Portable manifest references did not converge after path rebasing"
    )


def _portable_rewrite_records(
    staging: Path, files: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in files:
        if row.get("portable_paths") is not True:
            continue
        destination = staging / Path(*PurePosixPath(str(row["destination"])).parts)
        staged_hash = file_sha256(destination)
        if staged_hash == row["sha256"]:
            continue
        records.append(
            {
                "path": str(row["destination"]),
                "source_path": str(row["source"]),
                "source_sha256": str(row["sha256"]),
                "source_size_bytes": int(row["size_bytes"]),
                "archive_sha256": staged_hash,
                "archive_size_bytes": destination.stat().st_size,
                "transformation": "machine_local_paths_rebased_to_portable_relative_paths",
            }
        )
    return records


def _write_portable_evidence_manifest(
    spec: Mapping[str, object],
    plan: Mapping[str, object],
    staging: Path,
    rewrite_records: Sequence[Mapping[str, object]],
) -> None:
    declaration = spec.get("authoritative_evidence_package")
    if not isinstance(declaration, Mapping):
        return
    source = str(declaration["source"]).rstrip("/")
    entries = [
        entry
        for entry in spec.get("artifacts", {}).get("figure_table_outputs", [])
        if str(entry.get("source", "")).rstrip("/") == source
    ]
    if len(entries) != 1:
        raise RevisionReleaseError(
            "Authoritative evidence source must have one figure_table_outputs entry"
        )
    entry = entries[0]
    destination_root = PurePosixPath(str(entry["destination"]))
    target_root = staging / Path(*destination_root.parts)
    target = target_root / "manifest.json"
    if target.exists() or target.is_symlink():
        raise RevisionReleaseError(
            "Portable evidence manifest destination was not omitted from source copy"
        )
    records = []
    source_prefix = source + "/"
    for row in plan["resolution"]["files"]:
        source_path = str(row["source"])
        if not source_path.startswith(source_prefix):
            continue
        destination = staging / Path(*PurePosixPath(str(row["destination"])).parts)
        records.append(
            {
                "path": str(PurePosixPath(str(row["destination"])).relative_to(destination_root)),
                "source_sha256": str(row["sha256"]),
                "source_size_bytes": int(row["size_bytes"]),
                "archive_sha256": file_sha256(destination),
                "archive_size_bytes": destination.stat().st_size,
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    omitted = [str(value) for value in entry.get("exclude_paths", [])]
    evidence = {
        "schema_version": PORTABLE_EVIDENCE_SCHEMA,
        "status": "passed",
        "source_package_manifest": {
            "repository_relative_path": str(declaration["manifest"]),
            "source_sha256": str(declaration["manifest_sha256"]),
            "source_status": str(declaration["status"]),
            "verification": plan["authoritative_evidence_verification"],
        },
        "verified_external_reference_labels": sorted(
            str(value)
            for value in declaration.get("required_external_labels", [])
        ),
        "external_reference_policy": (
            "Only the listed source-manifest external references are archive "
            "dependencies; abandoned CPU-feasibility, GPU-ledger, and pre-final "
            "packaging records are not dependencies of this candidate"
        ),
        "archive_files": records,
        "archive_files_sha256": canonical_json_sha256(records),
        "archive_file_count": len(records),
        "omitted_source_paths": sorted(omitted),
        "omission_policy": (
            "duplicate raster previews, pre-final diagnostics, and internal "
            "process inventories are omitted; canonical vector figures, source "
            "tables, final results, and scientific provenance are retained"
        ),
        "portable_path_rewrites": [
            dict(row)
            for row in rewrite_records
            if str(row["source_path"]).startswith(source_prefix)
        ],
        "human_participant_outcomes_collected": False,
        "raw_human_response_data_included": False,
        "cross_device_equivalence_evidence_included": False,
        "model_weights_included": False,
    }
    evidence["manifest_sha256"] = canonical_json_sha256(evidence)
    _write_json_no_overwrite(target, evidence)


def _metadata_file(spec: Mapping[str, object], name: str, project_root: Path) -> Path:
    value = spec.get(name)
    if not value:
        raise RevisionReleaseError("Release specification is missing {}".format(name))
    relative = _safe_relative_path(value, name)
    path = (Path(project_root) / Path(*relative.parts)).resolve(strict=False)
    if not _within_root(Path(project_root).resolve(), path) or not path.is_file() or path.is_symlink():
        raise RevisionReleaseError("Invalid {} input".format(name))
    exclusion = _path_exclusion(PurePosixPath(path.relative_to(project_root).as_posix()), "documentation")
    if exclusion:
        raise RevisionReleaseError("{} is prohibited: {}".format(name, exclusion[0]))
    return path


def _render_readme(spec: Mapping[str, object], readiness: Mapping[str, object]) -> str:
    metadata = spec["metadata"]
    doi = metadata.get("doi")
    doi_line = (
        "User-supplied DOI metadata: `{}` (not minted or verified by this assembler).".format(doi)
        if doi
        else "No DOI is assigned in this offline candidate; no deposit was performed."
    )
    return "\n".join(
        [
            "# {}".format(metadata.get("title")),
            "",
            str(metadata.get("description")),
            "",
            "Version: `{}`".format(metadata.get("version")),
            "",
            "Release status: `{}`".format(EXTERNAL_STATUS),
            "",
            "Content readiness: `{}`".format(readiness["content_readiness"]),
            "",
            "No human participant outcomes were collected or included.",
            "",
            doi_line,
            "",
            "This directory is an offline release candidate. Building it did not contact Zenodo,",
            "reserve a DOI, upload data, publish a release, or perform any other external action.",
            "",
            "See `REPRODUCE.md`, `THIRD_PARTY_INVENTORY.json`, `ENVIRONMENT_INPUTS.json`,",
            "`ARTIFACT_EVIDENCE_ROLES.json`, `ASSEMBLY_REPORT.json`, and",
            "`PACKAGE_MANIFEST.json` for the auditable package contract. The portable final-evidence index is",
            "results/revision_v1/final_experiment_package/manifest.json.",
            "",
        ]
    )


def _render_reproduction(spec: Mapping[str, object]) -> str:
    lines = [
        "# Reproduction commands",
        "",
        "These commands are documentation only. The package assembler did not execute them",
        "and did not access the network.",
        "",
    ]
    for index, row in enumerate(spec.get("reproduction_commands", []), start=1):
        lines.extend(
            [
                "## {}. {}".format(index, row.get("label", "Command")),
                "",
                "```bash",
                str(row.get("command", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _manifest_files(staging: Path) -> List[Dict[str, object]]:
    excluded = {"SHA256SUMS", "PACKAGE_MANIFEST.json", "PACKAGE_MANIFEST.sha256"}
    rows: List[Dict[str, object]] = []
    for path in sorted(staging.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(staging).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return rows


def _verify_staged_placeholders(
    staging: Path, ignored_paths: Iterable[str] = ()
) -> List[Dict[str, object]]:
    ignored = set(map(str, ignored_paths))
    findings: List[Dict[str, object]] = []
    for path in sorted(staging.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.relative_to(staging).as_posix() in ignored:
            continue
        placeholders, secrets = _content_findings(path)
        if placeholders:
            findings.append({"path": path.relative_to(staging).as_posix(), "markers": placeholders})
        if secrets:
            raise RevisionReleaseError(
                "Secret signature appeared in staged file {}".format(path.relative_to(staging))
            )
    return findings


def _contains_byte_sequence(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    previous = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            combined = previous + block
            if needle in combined:
                return True
            previous = combined[-overlap:] if overlap else b""
    return False


def _load_candidate_json(path: Path, label: str) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionReleaseError("Cannot load {} from staged candidate".format(label)) from exc
    if not isinstance(value, dict):
        raise RevisionReleaseError("{} must be a JSON object".format(label))
    return value


def verify_release_candidate(candidate_dir: Path) -> Dict[str, object]:
    # Independently re-read every staged byte; never trust the build return value.
    root = Path(candidate_dir)
    if not root.is_dir() or root.is_symlink():
        raise RevisionReleaseError("Release candidate is missing, non-directory, or a symlink")
    root = root.resolve()
    actual_files: Dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise RevisionReleaseError("Symlink in staged candidate: {}".format(relative))
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise RevisionReleaseError("Special file in staged candidate: {}".format(relative))
        exclusion = _path_exclusion(relative, "documentation")
        if exclusion is not None:
            raise RevisionReleaseError(
                "Prohibited staged path {}: {}".format(relative, exclusion[0])
            )
        if _contains_byte_sequence(path, HOME_PREFIX.encode("ascii")):
            raise RevisionReleaseError(
                "Machine-local absolute path in staged candidate: {}".format(relative)
            )
        _, secrets = _content_findings(path, scan_placeholders=False)
        if secrets:
            raise RevisionReleaseError("Secret signature in staged candidate: {}".format(relative))
        actual_files[relative.as_posix()] = path

    required_metadata = {
        "PACKAGE_MANIFEST.json", "PACKAGE_MANIFEST.sha256", "SHA256SUMS",
        "ASSEMBLY_REPORT.json", "ARTIFACT_EVIDENCE_ROLES.json",
        "ENVIRONMENT_INPUTS.json", "THIRD_PARTY_INVENTORY.json",
    }
    missing_metadata = sorted(required_metadata - set(actual_files))
    if missing_metadata:
        raise RevisionReleaseError(
            "Staged candidate is missing audit metadata: {}".format(", ".join(missing_metadata))
        )
    manifest = _load_candidate_json(actual_files["PACKAGE_MANIFEST.json"], "package manifest")
    if manifest.get("manifest_type") != PACKAGE_MANIFEST_TYPE:
        raise RevisionReleaseError("Unexpected staged package manifest type")
    if manifest.get("release_status") != EXTERNAL_STATUS:
        raise RevisionReleaseError("Staged release status is not offline-draft")
    if manifest.get("external_action_performed") is not False:
        raise RevisionReleaseError("Staged manifest claims an external action")
    if manifest.get("network_access_used") is not False:
        raise RevisionReleaseError("Staged manifest claims network access")
    if manifest.get("doi") in (None, ""):
        if manifest.get("doi_provenance") != "not_assigned":
            raise RevisionReleaseError("Null DOI has inconsistent provenance")
    elif manifest.get("doi_provenance") != "user_supplied_unverified":
        raise RevisionReleaseError("Supplied DOI has inconsistent provenance")
    if manifest.get("human_participant_outcomes_collected") is not False:
        raise RevisionReleaseError(
            "Candidate must state that no human participant outcomes were collected"
        )

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RevisionReleaseError("Package manifest files must be a non-empty array")
    if manifest.get("file_count_excluding_manifest_and_checksum") != len(rows):
        raise RevisionReleaseError("Package manifest file count mismatch")
    if canonical_json_sha256(rows) != manifest.get("files_sha256"):
        raise RevisionReleaseError("Package manifest file-array hash mismatch")
    listed: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RevisionReleaseError("Invalid package manifest file row")
        relative = _safe_relative_path(row.get("path"), "package manifest path").as_posix()
        if relative in listed:
            raise RevisionReleaseError("Duplicate package manifest path: {}".format(relative))
        if relative in {"PACKAGE_MANIFEST.json", "PACKAGE_MANIFEST.sha256", "SHA256SUMS"}:
            raise RevisionReleaseError("Package metadata recursively listed in content manifest")
        path = actual_files.get(relative)
        if path is None:
            raise RevisionReleaseError("Manifest-listed file is missing: {}".format(relative))
        if path.stat().st_size != row.get("size_bytes") or file_sha256(path) != row.get("sha256"):
            raise RevisionReleaseError("Manifest-listed file hash or size mismatch: {}".format(relative))
        listed[relative] = row
    expected_files = set(listed) | {
        "PACKAGE_MANIFEST.json", "PACKAGE_MANIFEST.sha256", "SHA256SUMS"
    }
    if set(actual_files) != expected_files:
        raise RevisionReleaseError(
            "Staged candidate file-set mismatch: extra={} missing={}".format(
                sorted(set(actual_files) - expected_files),
                sorted(expected_files - set(actual_files)),
            )
        )
    expected_sums = "".join(
        "{}  {}\n".format(row["sha256"], row["path"]) for row in rows
    )
    if actual_files["SHA256SUMS"].read_text(encoding="utf-8") != expected_sums:
        raise RevisionReleaseError("SHA256SUMS does not exactly match package manifest rows")
    manifest_sha256 = file_sha256(actual_files["PACKAGE_MANIFEST.json"])
    expected_manifest_hash_line = "{}  PACKAGE_MANIFEST.json\n".format(manifest_sha256)
    if actual_files["PACKAGE_MANIFEST.sha256"].read_text(encoding="utf-8") != expected_manifest_hash_line:
        raise RevisionReleaseError("PACKAGE_MANIFEST.sha256 mismatch")

    assembly = _load_candidate_json(actual_files["ASSEMBLY_REPORT.json"], "assembly report")
    if assembly.get("network_access_used") is not False:
        raise RevisionReleaseError("Assembly report claims network access")
    if assembly.get("external_action_performed") is not False:
        raise RevisionReleaseError("Assembly report claims external action")
    if assembly.get("doi_minted_or_reserved_by_assembler") is not False:
        raise RevisionReleaseError("Assembly report claims DOI action")
    partitions = json.loads(actual_files["ARTIFACT_EVIDENCE_ROLES.json"].read_text(encoding="utf-8"))
    if not isinstance(partitions, list) or partitions != assembly.get("artifact_evidence_partitions"):
        raise RevisionReleaseError("Evidence-role registry and assembly report differ")
    for row in partitions:
        if not isinstance(row, dict) or row.get("evidence_role") not in EVIDENCE_ROLES:
            raise RevisionReleaseError("Invalid staged evidence-role row")
        source = str(row.get("source", ""))
        role = row.get("evidence_role")
        if source == "results/revision_v1/primary" or source.startswith("results/revision_v1/primary/"):
            raise RevisionReleaseError("Invalidated legacy primary tree was staged")
        if "/smoke_v2" in source:
            raise RevisionReleaseError("Legacy smoke_v2 shard was staged")
        if "/invalidations/" in source and role != "forensic_invalidated_not_for_pooling":
            raise RevisionReleaseError("Invalidation registry has incorrect staged role")
        if "/incurred_charges/" in source and role != "forensic_charge_only_not_rate_evidence":
            raise RevisionReleaseError("Prior-charge ledger has incorrect staged role")
    confirmatory_partitions = [
        row for row in partitions
        if row.get("verification_status") == "verified_complete"
    ]
    if confirmatory_partitions:
        index_path = actual_files.get(CONFIRMATORY_INDEX_DESTINATION)
        if index_path is None:
            raise RevisionReleaseError("Verified confirmatory index is missing from candidate")
        try:
            staged_index = verify_staged_confirmatory_release_index(index_path, root)
        except ConfirmatoryReleaseIndexError as exc:
            raise RevisionReleaseError(
                "Staged confirmatory artifact verification failed: {}".format(exc)
            ) from exc
        if staged_index["artifact_count"] != len(confirmatory_partitions):
            raise RevisionReleaseError(
                "Staged confirmatory partition/index cardinality mismatch"
            )

    portable_path = actual_files.get(
        "results/revision_v1/final_experiment_package/manifest.json"
    )
    if portable_path is not None:
        portable = _load_candidate_json(
            portable_path, "portable final-evidence manifest"
        )
        signature = portable.pop("manifest_sha256", None)
        if (
            portable.get("schema_version") != PORTABLE_EVIDENCE_SCHEMA
            or portable.get("status") != "passed"
            or signature != canonical_json_sha256(portable)
            or portable.get("human_participant_outcomes_collected") is not False
            or portable.get("cross_device_equivalence_evidence_included") is not False
            or portable.get("model_weights_included") is not False
        ):
            raise RevisionReleaseError(
                "Portable final-evidence manifest failed its integrity or scope checks"
            )
        evidence_rows = portable.get("archive_files")
        if (
            not isinstance(evidence_rows, list)
            or portable.get("archive_file_count") != len(evidence_rows)
            or portable.get("archive_files_sha256")
            != canonical_json_sha256(evidence_rows)
        ):
            raise RevisionReleaseError(
                "Portable final-evidence manifest file inventory is inconsistent"
            )
        evidence_root = portable_path.parent
        for row in evidence_rows:
            if not isinstance(row, dict):
                raise RevisionReleaseError(
                    "Portable final-evidence manifest has a malformed row"
                )
            relative = _safe_relative_path(
                row.get("path"), "portable evidence file path"
            )
            path = evidence_root / Path(*relative.parts)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != row.get("archive_size_bytes")
                or file_sha256(path) != row.get("archive_sha256")
            ):
                raise RevisionReleaseError(
                    "Portable final-evidence file hash or size mismatch: {}".format(
                        relative
                    )
                )

    return {
        "status": "ok",
        "candidate_dir": str(root),
        "package_manifest_sha256": manifest_sha256,
        "file_count": len(actual_files),
        "manifested_content_file_count": len(rows),
        "total_size_bytes": sum(path.stat().st_size for path in actual_files.values()),
        "release_status": manifest.get("release_status"),
        "content_readiness": manifest.get("content_readiness"),
        "doi": manifest.get("doi") or None,
        "network_access_used": False,
        "external_action_performed": False,
        "model_weights_included": False,
        "evidence_partition_count": len(partitions),
    }


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically install a directory without replacing an existing target."""

    if destination.exists() or destination.is_symlink():
        raise RevisionReleaseError("Release output already exists: {}".format(destination))
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise RevisionReleaseError("Release output already exists: {}".format(destination))
        if error not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(error, os.strerror(error), str(destination))
    raise RevisionReleaseError(
        "Atomic no-replace directory installation is unavailable on this host; "
        "the assembler refuses a racy fallback"
    )


def build_release_candidate(
    spec: Mapping[str, object],
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
    dry_run: bool = False,
    require_final_ready: bool = False,
) -> Dict[str, object]:
    """Plan or atomically stage one offline, non-published release candidate."""

    root = Path(project_root).resolve()
    target = Path(output_dir).resolve(strict=False)
    if target == root or _within_root(target, root):
        raise RevisionReleaseError("Release output may not be the project root or its parent")
    plan = plan_release(spec, project_root=root)
    if require_final_ready and not plan["readiness"]["final_ready"]:
        raise RevisionReleaseError(
            "Release candidate is not final-ready; blockers: {}".format(
                plan["readiness"]["blocker_count"]
            )
        )
    if dry_run:
        return {**plan, "dry_run": True, "output_created": False, "output_dir": str(target)}
    if target.exists() or target.is_symlink():
        raise RevisionReleaseError("Release output already exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}.".format(target.name), suffix=".staging", dir=str(target.parent)))
    installed = False
    try:
        _copy_allowlisted_files(root, temporary, plan["resolution"]["files"])
        omitted_destinations = [
            str(row["path"])
            for row in plan["resolution"]["exclusions"]
            if row.get("reason") == "archive_policy_omission"
        ]
        _refresh_portable_manifests(
            temporary, plan["resolution"]["files"], omitted_destinations
        )
        portable_rewrites = _portable_rewrite_records(
            temporary, plan["resolution"]["files"]
        )
        _write_portable_evidence_manifest(
            spec, plan, temporary, portable_rewrites
        )
        license_path = _metadata_file(spec, "license_file", root)
        citation_path = _metadata_file(spec, "citation_file", root)
        _write_bytes_no_overwrite(temporary / "LICENSE", license_path.read_bytes())
        _write_bytes_no_overwrite(temporary / "CITATION.cff", citation_path.read_bytes())
        _write_bytes_no_overwrite(
            temporary / "README.md", _render_readme(spec, plan["readiness"]).encode("utf-8")
        )
        _write_bytes_no_overwrite(
            temporary / "REPRODUCE.md", _render_reproduction(spec).encode("utf-8")
        )
        _write_json_no_overwrite(temporary / "THIRD_PARTY_INVENTORY.json", plan["third_party_inventory"])
        _write_json_no_overwrite(temporary / "ENVIRONMENT_INPUTS.json", plan["environment_inventory"])
        _write_json_no_overwrite(
            temporary / "ARTIFACT_EVIDENCE_ROLES.json",
            plan["artifact_evidence_partitions"],
        )
        assembly_report = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_status": EXTERNAL_STATUS,
            "content_readiness": plan["readiness"],
            "missing_artifacts": plan["resolution"]["missing_artifacts"],
            "excluded_artifacts": plan["resolution"]["exclusions"],
            "group_file_counts": plan["resolution"]["group_file_counts"],
            "artifact_evidence_partitions": plan["artifact_evidence_partitions"],
            "confirmatory_artifact_verification": plan[
                "confirmatory_artifact_verification"
            ],
            "authoritative_evidence_verification": plan[
                "authoritative_evidence_verification"
            ],
            "portable_path_rewrite_count": len(portable_rewrites),
            "copied_allowlist_file_count": plan["resolution"]["total_file_count"],
            "copied_allowlist_size_bytes": plan["resolution"]["total_size_bytes"],
            "network_access_used": False,
            "external_action_performed": False,
            "doi_minted_or_reserved_by_assembler": False,
        }
        _write_json_no_overwrite(temporary / "ASSEMBLY_REPORT.json", assembly_report)
        non_template_destinations = {
            str(row["destination"])
            for row in plan["resolution"]["files"]
            if row["group"] not in {"documentation", "human_materials"}
        }
        staged_placeholders = _verify_staged_placeholders(
            temporary, ignored_paths=non_template_destinations
        )
        source_placeholder_paths = {
            str(blocker.get("detail", {}).get("path"))
            for blocker in plan["readiness"]["blockers"]
            if blocker.get("code") == "unresolved_placeholder"
            and isinstance(blocker.get("detail"), dict)
        }
        unexpected = [
            finding for finding in staged_placeholders
            if finding["path"] != "ASSEMBLY_REPORT.json"
            if finding["path"] not in source_placeholder_paths
            and not any(finding["path"] == row["destination"] for row in plan["resolution"]["files"] if row["source"] in source_placeholder_paths)
        ]
        if unexpected:
            raise RevisionReleaseError("Generated metadata contains unresolved placeholders")
        if require_final_ready and staged_placeholders:
            raise RevisionReleaseError("Final-ready staging still contains placeholders")
        files = _manifest_files(temporary)
        checksums = "".join("{}  {}\n".format(row["sha256"], row["path"]) for row in files)
        _write_bytes_no_overwrite(temporary / "SHA256SUMS", checksums.encode("utf-8"))
        package_manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "manifest_type": PACKAGE_MANIFEST_TYPE,
            "package_id": spec.get("package_id"),
            "version": spec["metadata"].get("version"),
            "release_status": EXTERNAL_STATUS,
            "content_readiness": plan["readiness"]["content_readiness"],
            "external_action_performed": False,
            "network_access_used": False,
            "doi": spec["metadata"].get("doi") or None,
            "doi_provenance": "user_supplied_unverified" if spec["metadata"].get("doi") else "not_assigned",
            "human_participant_outcomes_collected": False,
            "specification_sha256": canonical_json_sha256(spec),
            "file_count_excluding_manifest_and_checksum": len(files),
            "files": files,
            "files_sha256": canonical_json_sha256(files),
        }
        _write_json_no_overwrite(temporary / "PACKAGE_MANIFEST.json", package_manifest)
        manifest_hash = file_sha256(temporary / "PACKAGE_MANIFEST.json")
        _write_bytes_no_overwrite(
            temporary / "PACKAGE_MANIFEST.sha256",
            "{}  PACKAGE_MANIFEST.json\n".format(manifest_hash).encode("utf-8"),
        )
        for row in files:
            if file_sha256(temporary / row["path"]) != row["sha256"]:
                raise RevisionReleaseError("Final staged checksum verification failed")
        _rename_no_replace(temporary, target)
        installed = True
        return {
            **plan,
            "dry_run": False,
            "output_created": True,
            "output_dir": str(target),
            "package_manifest_sha256": manifest_hash,
        }
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def publish_release(*args: object, **kwargs: object) -> None:
    """Explicitly refuse publication, upload, deposit, and DOI operations."""

    del args, kwargs
    raise RevisionReleaseError(
        "External release actions are prohibited; obtain explicit approval and use a separate audited workflow"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-final-ready",
        action="store_true",
        help="Refuse staging unless all local content-readiness checks pass.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        spec = _load_json(args.spec)
        report = build_release_candidate(
            spec,
            output_dir=args.output_dir,
            project_root=args.project_root,
            dry_run=args.dry_run,
            require_final_ready=args.require_final_ready,
        )
    except RevisionReleaseError as exc:
        raise SystemExit("revision release assembly failed: {}".format(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "ARTIFACT_GROUPS",
    "DEFAULT_SPEC_PATH",
    "DRAFT_CONTENT_STATUS",
    "EVIDENCE_ROLES",
    "EXTERNAL_STATUS",
    "FINAL_CONTENT_STATUS",
    "RevisionReleaseError",
    "assess_content_readiness",
    "build_argument_parser",
    "build_environment_inventory",
    "build_release_candidate",
    "build_third_party_inventory",
    "main",
    "plan_release",
    "publish_release",
    "verify_authoritative_evidence_package",
    "resolve_release_inputs",
    "validate_release_spec",
    "verify_release_candidate",
]
