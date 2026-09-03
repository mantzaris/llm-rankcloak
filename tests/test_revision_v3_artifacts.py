import subprocess
from pathlib import Path

from rankcloak.revision_v3_artifacts import portable_artifact_files


def test_portable_artifact_inventory_excludes_ignored_local_files(tmp_path):
    repository = tmp_path / "repo"
    output = repository / "results" / "revision_v3"
    output.mkdir(parents=True)
    (repository / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (output / "tracked.csv").write_text("value\n1\n", encoding="utf-8")
    (output / "new.json").write_text("{}\n", encoding="utf-8")
    (output / "local.log").write_text("local only\n", encoding="utf-8")
    (output / "artifact_manifest.csv").write_text("self\n", encoding="utf-8")
    validation = output / "provenance" / "validation_report.json"
    validation.parent.mkdir()
    validation.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "add",
            ".gitignore",
            "results/revision_v3/tracked.csv",
        ],
        check=True,
    )

    observed = {
        path.relative_to(output).as_posix()
        for path in portable_artifact_files(output, repository)
    }
    assert observed == {"new.json", "tracked.csv"}


def test_external_artifact_inventory_uses_regular_files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "external"
    output.mkdir()
    (output / "result.txt").write_text("result\n", encoding="utf-8")
    assert [
        path.relative_to(output).as_posix()
        for path in portable_artifact_files(output, project)
    ] == ["result.txt"]
