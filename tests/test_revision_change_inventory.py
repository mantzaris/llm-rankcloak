from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rankcloak.revision_change_inventory import (
    ChangeInventoryError,
    build_change_inventory,
    canonical_json_sha256,
    file_sha256,
)


class _Result:
    def __init__(self, stdout: bytes):
        self.stdout = stdout


def _mock_git(monkeypatch, outputs):
    def run(command, **_kwargs):
        key = tuple(command[1:])
        return _Result(outputs.get(key, b""))

    monkeypatch.setattr(subprocess, "run", run)


def test_change_inventory_hashes_dirty_files_and_preserved_temp(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package = root / "results" / "revision_v1" / "final_experiment_package"
    package.mkdir(parents=True)
    tracked = root / "rankcloak" / "module.py"
    tracked.parent.mkdir()
    tracked.write_text("changed\n", encoding="utf-8")
    preexisting = root / ".gitignore"
    preexisting.write_text("existing\n", encoding="utf-8")
    generated = root / "results" / "generated.csv"
    generated.write_text("value\n1\n", encoding="utf-8")
    extra = tmp_path / "diagnostics"
    extra.mkdir()
    extra_file = extra / "failed_cpu.json"
    extra_file.write_text('{"status":"timeout"}\n', encoding="utf-8")
    _mock_git(
        monkeypatch,
        {
            ("diff", "--cached", "--name-only", "-z"): b"",
            ("diff", "--name-only", "-z"): b".gitignore\0rankcloak/module.py\0",
            ("ls-files", "--others", "--exclude-standard", "-z"): b"results/generated.csv\0",
        },
    )
    output = package / "changed_and_generated_files.txt"
    manifest_path = package / "change_inventory_manifest.json"
    artifacts = build_change_inventory(
        project_root=root,
        output_path=output,
        manifest_path=manifest_path,
        preexisting_paths=(".gitignore",),
        extra_paths=(extra,),
        planned_output_paths=(package / "manifest.json",),
        command="fixture",
    )
    assert artifacts.repository_entry_count == 3
    text = output.read_text(encoding="utf-8")
    assert "preexisting_user_change\t.gitignore" in text
    assert "generated_or_modified_scientific_evidence\tresults/generated.csv" in text
    assert str(extra_file) in text
    assert "results/revision_v1/final_experiment_package/manifest.json" in text
    assert "Preserved in place" in text
    manifest = json.loads(manifest_path.read_text())
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    declaration = manifest["outputs"]["inventory"]
    assert declaration["sha256"] == file_sha256(output)
    assert manifest["staged_path_count"] == 0
    assert manifest["deletion_move_or_quarantine_performed"] is False


def test_change_inventory_rejects_staged_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _mock_git(
        monkeypatch,
        {("diff", "--cached", "--name-only", "-z"): b"staged.txt\0"},
    )
    with pytest.raises(ChangeInventoryError, match="staged worktree"):
        build_change_inventory(
            project_root=root,
            output_path=root / "inventory.txt",
            manifest_path=root / "manifest.json",
        )


def test_change_inventory_rejects_extra_path_outside_repo_or_tmp(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    outside = Path("/var") / "rankcloak-forbidden-fixture"
    _mock_git(
        monkeypatch,
        {
            ("diff", "--cached", "--name-only", "-z"): b"",
            ("diff", "--name-only", "-z"): b"",
            ("ls-files", "--others", "--exclude-standard", "-z"): b"",
        },
    )
    with pytest.raises(ChangeInventoryError, match="repository or /tmp"):
        build_change_inventory(
            project_root=root,
            output_path=root / "inventory.txt",
            manifest_path=root / "manifest.json",
            extra_paths=(outside,),
        )


def test_change_inventory_refuses_unrequested_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    output = root / "inventory.txt"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ChangeInventoryError, match="Refusing to overwrite"):
        build_change_inventory(
            project_root=root,
            output_path=output,
            manifest_path=root / "manifest.json",
        )
