"""Read-only preservation inventory for local revision changes and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "rankcloak-revision-change-inventory-v1"
INVENTORY_COLUMNS = (
    "local_status",
    "preservation_class",
    "path",
    "entry_type",
    "size_bytes",
    "sha256",
    "notes",
)


class ChangeInventoryError(ValueError):
    """Raised when a safe, complete local-change inventory cannot be created."""


@dataclass(frozen=True)
class ChangeInventoryArtifacts:
    inventory_path: str
    manifest_path: str
    repository_entry_count: int
    extra_entry_count: int
    total_entry_count: int


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_z(project_root: Path, arguments: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ChangeInventoryError(
            f"Read-only Git inventory command failed: git {' '.join(arguments)}"
        ) from exc
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    ]


def _classify(path: str, *, status: str, preexisting: set[str]) -> tuple[str, str]:
    if path in preexisting:
        return "preexisting_user_change", "Preserve unchanged; predates this computation session."
    if path.startswith("results/") or path.startswith("release_inputs/"):
        return (
            "generated_or_modified_scientific_evidence",
            "Preserve as scientific evidence, provenance, checkpoint, or diagnostic.",
        )
    if path.startswith(("rankcloak/", "scripts/", "tests/", "configs/", "analysis/", "operations/", "environment/")):
        return (
            "source_configuration_environment_or_test",
            "Preserve as reviewable implementation or validation material.",
        )
    if status == "untracked":
        return "untracked_unclassified_preserve", "Preserve pending manual review."
    return "tracked_local_change_preserve", "Preserve pending manual review."


def _file_row(
    path: Path,
    *,
    display_path: str,
    local_status: str,
    preservation_class: str,
    notes: str,
) -> dict[str, str]:
    if path.is_symlink():
        return {
            "local_status": local_status,
            "preservation_class": preservation_class,
            "path": display_path,
            "entry_type": "symlink_not_followed",
            "size_bytes": "",
            "sha256": "",
            "notes": f"{notes} Symlink target: {os.readlink(path)}",
        }
    if not path.is_file():
        raise ChangeInventoryError(f"Inventory path is missing or unsafe: {path}")
    return {
        "local_status": local_status,
        "preservation_class": preservation_class,
        "path": display_path,
        "entry_type": "regular_file",
        "size_bytes": str(int(path.stat().st_size)),
        "sha256": file_sha256(path),
        "notes": notes,
    }


def _repository_rows(
    project_root: Path, *, preexisting_paths: Iterable[str]
) -> list[dict[str, str]]:
    staged = _git_z(project_root, ("diff", "--cached", "--name-only", "-z"))
    if staged:
        raise ChangeInventoryError(
            "Refusing to inventory a staged worktree: " + ", ".join(sorted(staged))
        )
    modified = _git_z(project_root, ("diff", "--name-only", "-z"))
    untracked = _git_z(
        project_root, ("ls-files", "--others", "--exclude-standard", "-z")
    )
    overlap = set(modified) & set(untracked)
    if overlap:
        raise ChangeInventoryError(
            "Git classified paths as both tracked and untracked: "
            + ", ".join(sorted(overlap))
        )
    preexisting = {str(Path(value)) for value in preexisting_paths}
    rows: list[dict[str, str]] = []
    for status, paths in (("tracked_modified", modified), ("untracked", untracked)):
        for relative in sorted(paths):
            normalized = str(Path(relative))
            candidate = project_root / normalized
            preservation_class, notes = _classify(
                normalized, status=status, preexisting=preexisting
            )
            rows.append(
                _file_row(
                    candidate,
                    display_path=normalized,
                    local_status=status,
                    preservation_class=preservation_class,
                    notes=notes,
                )
            )
    missing_preexisting = preexisting - {row["path"] for row in rows}
    if missing_preexisting:
        raise ChangeInventoryError(
            "Declared preexisting paths are not local changes: "
            + ", ".join(sorted(missing_preexisting))
        )
    return rows


def _extra_rows(
    project_root: Path, extra_paths: Sequence[str | Path]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    temporary_root = Path("/tmp").resolve()
    for raw in extra_paths:
        root = Path(raw).resolve()
        try:
            relative = root.relative_to(project_root)
            display_root = str(relative)
        except ValueError:
            try:
                root.relative_to(temporary_root)
            except ValueError as exc:
                raise ChangeInventoryError(
                    f"Extra inventory path must be in the repository or /tmp: {root}"
                ) from exc
            display_root = str(root)
        if root.is_symlink():
            candidates = [root]
        elif root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [path for path in sorted(root.rglob("*")) if path.is_file() or path.is_symlink()]
            rows.append(
                {
                    "local_status": "preserved_extra_root",
                    "preservation_class": "failed_or_temporary_diagnostic_preserve",
                    "path": display_root,
                    "entry_type": "directory_root",
                    "size_bytes": str(
                        sum(
                            int(path.stat().st_size)
                            for path in candidates
                            if path.is_file() and not path.is_symlink()
                        )
                    ),
                    "sha256": "",
                    "notes": "Read-only preserved diagnostic root; directory hash not synthesized.",
                }
            )
        else:
            raise ChangeInventoryError(f"Extra inventory path is missing: {root}")
        for path in candidates:
            try:
                relative = path.relative_to(project_root)
                display = str(relative)
            except ValueError:
                display = str(path)
            if display in seen:
                continue
            seen.add(display)
            rows.append(
                _file_row(
                    path,
                    display_path=display,
                    local_status="preserved_extra_file",
                    preservation_class="failed_or_temporary_diagnostic_preserve",
                    notes="Preserved in place; no deletion, move, or quarantine performed.",
                )
            )
    return rows


def _serialize_tsv(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = ["\t".join(INVENTORY_COLUMNS)]
    for row in rows:
        values = []
        for column in INVENTORY_COLUMNS:
            value = str(row.get(column, ""))
            if "\t" in value or "\n" in value or "\r" in value:
                raise ChangeInventoryError(
                    f"Inventory field contains a forbidden line separator: {column}"
                )
            values.append(value)
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_bytes(target: Path, content: bytes) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(content)
    os.replace(temporary, target)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def build_change_inventory(
    *,
    project_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    preexisting_paths: Sequence[str] = (),
    extra_paths: Sequence[str | Path] = (),
    planned_output_paths: Sequence[str | Path] = (),
    command: str | None = None,
    overwrite: bool = False,
) -> ChangeInventoryArtifacts:
    """Inventory local files without staging, deleting, moving, or following links."""

    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    manifest = Path(manifest_path).resolve()
    for target in (output, manifest):
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ChangeInventoryError(
                "Inventory outputs must remain inside the repository"
            ) from exc
        if target.exists() and not overwrite:
            raise ChangeInventoryError(f"Refusing to overwrite inventory output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    output_rel = str(output.relative_to(root))
    manifest_rel = str(manifest.relative_to(root))
    planned_relatives: list[str] = []
    for raw in planned_output_paths:
        path = Path(raw).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise ChangeInventoryError(
                f"Planned inventory output must remain inside the repository: {path}"
            ) from exc
        planned_relatives.append(relative)
    self_paths = {output_rel, manifest_rel, *planned_relatives}

    repository_rows = [
        row
        for row in _repository_rows(root, preexisting_paths=preexisting_paths)
        if row["path"] not in self_paths
    ]
    extra_rows = _extra_rows(root, extra_paths)
    indexed_paths = {row["path"] for row in repository_rows}
    deduplicated_extra = [row for row in extra_rows if row["path"] not in indexed_paths]
    self_rows = [
        {
            "local_status": "generated_inventory_output",
            "preservation_class": "source_configuration_environment_or_test",
            "path": relative,
            "entry_type": "regular_file_self_identity_in_package_manifest",
            "size_bytes": "",
            "sha256": "",
            "notes": "Self-referential identity intentionally delegated to the signed package manifest.",
        }
        for relative in sorted(self_paths)
    ]
    rows = [*repository_rows, *deduplicated_extra, *self_rows]
    rows.sort(key=lambda row: (row["path"], row["local_status"]))
    _atomic_bytes(output, _serialize_tsv(rows))

    status_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["local_status"]] = status_counts.get(row["local_status"], 0) + 1
        class_counts[row["preservation_class"]] = class_counts.get(row["preservation_class"], 0) + 1
    manifest_value = _signed(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "read_only_inventory": True,
            "staged_path_count": 0,
            "deletion_move_or_quarantine_performed": False,
            "preexisting_paths": sorted(str(Path(value)) for value in preexisting_paths),
            "extra_roots": sorted(str(Path(value).resolve()) for value in extra_paths),
            "planned_output_paths": sorted(planned_relatives),
            "outputs": {
                "inventory": {
                    "path": str(output),
                    "sha256": file_sha256(output),
                    "size_bytes": int(output.stat().st_size),
                    "row_count": len(rows),
                }
            },
            "summary": {
                "repository_entry_count": len(repository_rows),
                "extra_entry_count": len(deduplicated_extra),
                "total_entry_count": len(rows),
                "local_status_counts": status_counts,
                "preservation_class_counts": class_counts,
            },
            "generation_command": command,
        }
    )
    _atomic_bytes(
        manifest,
        (json.dumps(manifest_value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return ChangeInventoryArtifacts(
        inventory_path=str(output),
        manifest_path=str(manifest),
        repository_entry_count=len(repository_rows),
        extra_entry_count=len(deduplicated_extra),
        total_entry_count=len(rows),
    )
