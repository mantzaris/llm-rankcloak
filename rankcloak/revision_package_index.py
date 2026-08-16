"""Atomic, hash-validated index for the computational revision evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rankcloak-final-experiment-package-index-v1"


class PackageIndexError(ValueError):
    """Raised when package evidence or a declared reference is inconsistent."""


@dataclass(frozen=True)
class PackageIndexArtifacts:
    manifest_path: str
    manifest_sha256: str
    package_file_count: int
    external_reference_count: int
    validated_declared_output_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PackageIndexError(f"Missing or unsafe {label}: {path}")
    return path.resolve()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackageIndexError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageIndexError(f"{label} must contain a JSON object")
    return value


def _observed_row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = sum(1 for _ in reader)
        return max(rows - 1, 0)
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


def _file_identity(
    path: Path,
    *,
    relative_to: Path | None = None,
    include_row_count: bool = False,
) -> dict[str, Any]:
    resolved = _regular_file(path, label="artifact")
    displayed_path = str(resolved)
    if relative_to is not None:
        try:
            displayed_path = str(resolved.relative_to(relative_to))
        except ValueError:
            pass
    result: dict[str, Any] = {
        "path": displayed_path,
        "sha256": file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if include_row_count:
        row_count = _observed_row_count(resolved)
        if row_count is not None:
            result["row_count"] = row_count
    return result


def _declared_output_path(
    manifest_path: Path,
    declaration: Mapping[str, Any],
    *,
    label: str,
    repository_root: Path | None,
    repository_relative: bool,
) -> Path:
    raw = declaration.get("path")
    if not isinstance(raw, str) or not raw:
        raise PackageIndexError(f"{label} lacks a declared path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        if repository_relative:
            if repository_root is None:
                raise PackageIndexError(
                    f"{label} declares repository-relative paths but no repository root was found"
                )
            candidate = repository_root / candidate
        else:
            candidate = manifest_path.parent / candidate
    return _regular_file(candidate, label=label)


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    return None


def _validate_component_manifest(
    label: str,
    manifest_path: Path,
    repository_root: Path | None,
) -> tuple[dict[str, Any], int]:
    resolved = _regular_file(manifest_path, label=f"component manifest {label}")
    manifest = _read_json(resolved, label=f"component manifest {label}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise PackageIndexError(
            f"Component manifest {label} has no declared outputs object"
        )
    output_entries: list[dict[str, Any]] = []
    repository_relative = (
        manifest.get("portable_repository_relative_paths") is True
    )
    for output_label, declaration in sorted(
        outputs.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(declaration, Mapping):
            raise PackageIndexError(
                f"Component manifest {label} output {output_label} is malformed"
            )
        path = _declared_output_path(
            resolved,
            declaration,
            label=f"{label} output {output_label}",
            repository_root=repository_root,
            repository_relative=repository_relative,
        )
        observed_sha = file_sha256(path)
        observed_size = int(path.stat().st_size)
        declared_size = declaration.get("size_bytes", declaration.get("bytes"))
        if declaration.get("sha256") != observed_sha:
            raise PackageIndexError(
                f"Component manifest {label} output {output_label} hash mismatch"
            )
        if declared_size is not None and int(declared_size) != observed_size:
            raise PackageIndexError(
                f"Component manifest {label} output {output_label} size mismatch"
            )
        declared_rows = declaration.get("row_count")
        observed_rows = _observed_row_count(path)
        if (
            declared_rows is not None
            and observed_rows is not None
            and int(declared_rows) != observed_rows
        ):
            raise PackageIndexError(
                f"Component manifest {label} output {output_label} row-count mismatch"
            )
        output_entries.append(
            {
                "label": str(output_label),
                "path": str(path),
                "sha256": observed_sha,
                "size_bytes": observed_size,
                "row_count": observed_rows,
            }
        )
    entry = _file_identity(resolved)
    entry.update(
        {
            "label": label,
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "validated_outputs": output_entries,
        }
    )
    return entry, len(output_entries)


def _package_files(package_root: Path, output_path: Path) -> list[Path]:
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(package_root):
        directory_path = Path(directory)
        for name in list(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise PackageIndexError(
                    f"Package tree contains an unsafe symlink: {candidate}"
                )
        for name in file_names:
            candidate = directory_path / name
            if candidate == output_path:
                continue
            if candidate.is_symlink():
                raise PackageIndexError(
                    f"Package tree contains an unsafe symlink: {candidate}"
                )
            if not candidate.is_file():
                raise PackageIndexError(
                    f"Package tree contains an unsafe file: {candidate}"
                )
            if name.startswith(f".{output_path.name}.tmp-"):
                raise PackageIndexError(
                    f"Package tree contains a stale manifest temporary: {candidate}"
                )
            files.append(candidate.resolve())
    return sorted(files, key=lambda path: str(path.relative_to(package_root)))


def _signed_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def _atomic_json(value: Mapping[str, Any], target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def build_package_index(
    *,
    package_root: str | Path,
    component_manifests: Mapping[str, str | Path],
    external_references: Mapping[str, str | Path],
    required_relative_paths: Sequence[str | Path] = (),
    output_path: str | Path | None = None,
    command: str | None = None,
    overwrite: bool = False,
) -> PackageIndexArtifacts:
    """Validate package components and atomically publish a non-copying index."""

    raw_root = Path(package_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise PackageIndexError(f"Missing or unsafe package root: {raw_root}")
    root = raw_root.resolve()
    repository_root = _repository_root(root)
    target = (
        root / "manifest.json"
        if output_path is None
        else Path(output_path).resolve()
    )
    if target.parent != root:
        raise PackageIndexError(
            "Package index must be written directly under package root"
        )
    if target.exists() and not overwrite:
        raise PackageIndexError(f"Refusing to overwrite package index: {target}")

    required: list[str] = []
    for relative in required_relative_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PackageIndexError(f"Required package path is unsafe: {relative}")
        resolved = _regular_file(
            root / candidate, label=f"required package file {relative}"
        )
        required.append(str(resolved.relative_to(root)))

    components: list[dict[str, Any]] = []
    declared_output_count = 0
    for label, path in sorted(component_manifests.items()):
        if not label:
            raise PackageIndexError("Component manifest label is empty")
        entry, validated_count = _validate_component_manifest(
            label, Path(path), repository_root
        )
        components.append(entry)
        declared_output_count += validated_count

    references: list[dict[str, Any]] = []
    for label, path in sorted(external_references.items()):
        if not label:
            raise PackageIndexError("External reference label is empty")
        entry = _file_identity(Path(path), include_row_count=True)
        entry["label"] = label
        references.append(entry)

    package_files = _package_files(root, target)
    package_entries = [
        _file_identity(path, relative_to=root, include_row_count=True)
        for path in package_files
    ]
    manifest = _signed_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "package_root": str(root),
            "large_external_artifacts_copied": False,
            "required_relative_paths": sorted(required),
            "component_manifests": components,
            "external_references": references,
            "package_files": package_entries,
            "summary": {
                "package_file_count": len(package_entries),
                "package_bytes": sum(
                    int(entry["size_bytes"]) for entry in package_entries
                ),
                "component_manifest_count": len(components),
                "validated_declared_output_count": declared_output_count,
                "external_reference_count": len(references),
            },
            "generation_command": command,
        }
    )
    _atomic_json(manifest, target)
    reread = _read_json(target, label="published package index")
    observed_signature = reread.pop("manifest_sha256", None)
    if observed_signature != canonical_json_sha256(reread):
        raise PackageIndexError("Published package index self-hash differs")
    return PackageIndexArtifacts(
        manifest_path=str(target),
        manifest_sha256=str(observed_signature),
        package_file_count=len(package_entries),
        external_reference_count=len(references),
        validated_declared_output_count=declared_output_count,
    )
