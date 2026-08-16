from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_revision_detectors.py"
SPEC = importlib.util.spec_from_file_location("run_revision_detectors_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_single_checkpoint_metric_decodes_columnar_row(tmp_path: Path) -> None:
    path = tmp_path / "metric.json"
    payload = {
        "schema_version": "rankcloak-revision-detector-fit-rows-v1",
        "columns": [
            "detector_name",
            "implementation_metadata_json",
            "roc_auc",
        ],
        "rows": [
            [
                "published_textcnn_equivalent",
                json.dumps({"phase_timings_seconds": {"total": 12.5}}),
                0.91,
            ]
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    metric = runner._read_single_checkpoint_metric(path)

    assert metric["detector_name"] == "published_textcnn_equivalent"
    assert metric["roc_auc"] == 0.91
    assert json.loads(metric["implementation_metadata_json"])[
        "phase_timings_seconds"
    ]["total"] == 12.5


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"implementation_metadata_json": "{}"},
            "row payload is malformed",
        ),
        (
            {
                "schema_version": "rankcloak-revision-detector-fit-rows-v1",
                "columns": ["detector_name"],
                "rows": [["detector"]],
            },
            "lacks implementation metadata",
        ),
    ],
)
def test_single_checkpoint_metric_rejects_wrong_shape(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(runner.RevisionDetectionError, match=message):
        runner._read_single_checkpoint_metric(path)
