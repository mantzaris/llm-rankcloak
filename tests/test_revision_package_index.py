from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_package_index import (
    PackageIndexError,
    build_package_index,
    canonical_json_sha256,
    file_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "package"
    (package / "tables").mkdir(parents=True)
    (package / "STATUS.md").write_text("computational status\n", encoding="utf-8")
    local_table = package / "tables" / "stage_counts.csv"
    pd.DataFrame([{"stage": "primary", "completed": 12}]).to_csv(
        local_table, index=False
    )
    component_output = package / "tables" / "component.csv"
    pd.DataFrame([{"value": 1}, {"value": 2}]).to_csv(
        component_output, index=False
    )
    component_manifest = package / "component_manifest.json"
    component_manifest.write_text(
        json.dumps(
            {
                "schema_version": "fixture-v1",
                "status": "passed",
                "outputs": {
                    "table": {
                        "path": str(component_output.resolve()),
                        "sha256": file_sha256(component_output),
                        "size_bytes": component_output.stat().st_size,
                        "row_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "raw.jsonl"
    external.write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")
    return package, component_manifest, external


def test_package_index_hashes_local_files_and_external_references(tmp_path: Path):
    package, component, external = _fixture(tmp_path)
    artifacts = build_package_index(
        package_root=package,
        component_manifests={"component": component},
        external_references={"large_raw_reference": external},
        required_relative_paths=("STATUS.md", "tables/stage_counts.csv"),
        command="fixture command",
    )
    manifest_path = Path(artifacts.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest.pop("manifest_sha256")
    assert signature == canonical_json_sha256(manifest)
    assert manifest["large_external_artifacts_copied"] is False
    assert manifest["summary"] == {
        "package_file_count": 4,
        "package_bytes": sum(
            path.stat().st_size
            for path in (
                package / "STATUS.md",
                package / "component_manifest.json",
                package / "tables" / "component.csv",
                package / "tables" / "stage_counts.csv",
            )
        ),
        "component_manifest_count": 1,
        "validated_declared_output_count": 1,
        "external_reference_count": 1,
    }
    assert manifest["external_references"][0]["row_count"] == 2
    assert external.read_text(encoding="utf-8") == '{"row": 1}\n{"row": 2}\n'


def test_package_index_rejects_tampered_component_output(tmp_path: Path):
    package, component, external = _fixture(tmp_path)
    (package / "tables" / "component.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(PackageIndexError, match="hash mismatch"):
        build_package_index(
            package_root=package,
            component_manifests={"component": component},
            external_references={"raw": external},
        )


def test_package_index_accepts_opt_in_repository_relative_figure_paths(
    tmp_path: Path,
):
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    package = repository / "results" / "package"
    figures = package / "figures"
    figures.mkdir(parents=True)
    output = figures / "figure.pdf"
    output.write_bytes(b"%PDF-fixture")
    component = figures / "figure_manifest.json"
    component.write_text(
        json.dumps(
            {
                "schema_version": "figure-fixture-v2",
                "status": "passed",
                "portable_repository_relative_paths": True,
                "outputs": {
                    "figure": {
                        "path": "results/package/figures/figure.pdf",
                        "sha256": file_sha256(output),
                        "size_bytes": output.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = build_package_index(
        package_root=package,
        component_manifests={"figures": component},
        external_references={},
    )

    manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
    validated = manifest["component_manifests"][0]["validated_outputs"][0]
    assert validated["sha256"] == file_sha256(output)


def test_package_index_rejects_symlink_in_package_tree(tmp_path: Path):
    package, component, external = _fixture(tmp_path)
    (package / "unsafe_link").symlink_to(external)
    with pytest.raises(PackageIndexError, match="symlink"):
        build_package_index(
            package_root=package,
            component_manifests={"component": component},
            external_references={"raw": external},
        )


def test_package_index_refuses_unrequested_overwrite(tmp_path: Path):
    package, component, external = _fixture(tmp_path)
    build_package_index(
        package_root=package,
        component_manifests={"component": component},
        external_references={"raw": external},
    )
    with pytest.raises(PackageIndexError, match="Refusing to overwrite"):
        build_package_index(
            package_root=package,
            component_manifests={"component": component},
            external_references={"raw": external},
        )
