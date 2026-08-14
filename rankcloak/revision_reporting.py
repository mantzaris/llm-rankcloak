"""Deterministic report products for the Scientific Reports revision.

The reporting layer accepts paths to machine-produced manifests, never result
values.  It verifies every declared digest before reading an artifact, seals
legacy detector outputs that did not originally list output digests, checks
sample-size relationships, and writes content-addressed LaTeX/CSV/plot
products.  Missing evidence is rendered as explicitly unavailable; this module
does not impute, simulate, or hand-enter scientific outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .revision_artifacts import canonical_json_sha256 as compact_json_sha256


REPORT_SCHEMA_VERSION = "rankcloak-revision-report-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_MODEL_PLAN = (
    PROJECT_ROOT / "analysis" / "revision_v1" / "confirmatory_model_plan.json"
)
R_ENVIRONMENT_LOCK = (
    PROJECT_ROOT / "analysis" / "revision_v1" / "r_environment.lock.json"
)
R_MIXED_MODEL_DRIVER = PROJECT_ROOT / "scripts" / "run_revision_mixed_models.R"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "revision_v1" / "models.json"
EVALUATOR_UNAVAILABILITY_SCHEMA = (
    "rankcloak-heldout-evaluator-upstream-unavailability-v1"
)
EVALUATOR_UNAVAILABILITY_MANIFEST_TYPE = (
    "heldout_evaluator_upstream_dependent_unavailability"
)
AUTHORIZED_PROJECTION_SHA256 = (
    "35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3"
)
FROZEN_EVALUATOR_TARGET_UNITS = 17280
SCOREABLE_EVALUATOR_UNITS = 17232
UPSTREAM_UNAVAILABLE_EVALUATOR_UNITS = 48
GENERATOR_NOTICE = (
    "Generated from hash-verified machine outputs; do not hand-edit numeric results."
)
UNAVAILABLE = "unavailable"
MAX_MAIN_DISPLAY_ITEMS = 7
PROTOCOL_CONTRACT_REVISION = "payload_fidelity_v2"
RESULT_SCHEMA_REVISION = "payload_aware_result_v2"
PAYLOAD_RECOVERY_SEMANTICS = "original_serialized_payload_bytes_sha256_v1"
PAYLOAD_RECOVERY_OUTCOME = "exact_payload_recovery"
PRIMARY_EVIDENCE_STATUS = (
    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
)
PRIMARY_STUDY_PHASE = "primary_v2_confirmatory"
SUPERSEDING_EVIDENCE_PHASES = {
    "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling": {
        "smoke_v3_exploratory",
        "ordinary_llm_control_smoke_v3",
    },
    PRIMARY_EVIDENCE_STATUS: {PRIMARY_STUDY_PHASE},
    "confirmatory_ablation_v2_payload_fidelity_after_manifest_freeze": {
        "ablation_v2_confirmatory",
    },
    "secondary_supplementary_multilingual_v2_payload_fidelity_after_manifest_freeze": {
        "multilingual_v2_secondary",
    },
    "confirmatory_supporting_robustness_v2_payload_fidelity_after_manifest_freeze": {
        "robustness_v2_confirmatory_supporting",
    },
}
EVALUATOR_EVIDENCE_PHASES = {
    "confirmatory_heldout_evaluator_primary_v2_payload_fidelity_after_source_manifest_freeze": {
        "primary_v2_confirmatory",
        "ordinary_llm_control_primary_v2",
    },
    "confirmatory_supporting_heldout_evaluator_ablation_v2_payload_fidelity_after_source_manifest_freeze": {
        "ablation_v2_confirmatory",
    },
    "secondary_supplementary_heldout_evaluator_multilingual_v2_payload_fidelity_after_source_manifest_freeze": {
        "multilingual_v2_secondary",
        "ordinary_llm_control_multilingual_v2",
    },
    "exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling": {
        "smoke_v3_exploratory",
        "ordinary_llm_control_smoke_v3",
    },
}


class RevisionReportingError(ValueError):
    """Raised when evidence or a report product violates the reporting contract."""


class ReportArtifactConflict(RevisionReportingError):
    """Raised when a report build would replace an existing file with new bytes."""


@dataclass(frozen=True)
class VerifiedArtifact:
    source_kind: str
    logical_name: str
    path: Path
    sha256: str
    size_bytes: int
    row_count: Optional[int]
    manifest_declared_sha256: bool


@dataclass
class VerifiedSources:
    manifests: Dict[str, Dict[str, Any]]
    manifest_paths: Dict[str, Path]
    artifacts: Dict[str, VerifiedArtifact]
    tables: Dict[str, List[Dict[str, str]]]
    json_values: Dict[str, Any]
    fixture_mode: bool = False


@dataclass(frozen=True)
class ReportBuild:
    output_dir: Path
    files: Mapping[str, Path]
    source_manifest: Mapping[str, Any]
    integrity_report: Mapping[str, Any]


MAIN_DISPLAYS: Tuple[Mapping[str, str], ...] = (
    {
        "id": "main_figure_1",
        "type": "figure",
        "number": "1",
        "label": "fig:capacity-quality-framework",
        "title": "Payload representation and theoretical capacity--quality framework",
    },
    {
        "id": "main_figure_2",
        "type": "figure",
        "number": "2",
        "label": "fig:primary-multimodel",
        "title": "Primary multi-model RankCloak results",
    },
    {
        "id": "main_figure_3",
        "type": "figure",
        "number": "3",
        "label": "fig:human-quality",
        "title": "Cover quality and blinded human assessment",
    },
    {
        "id": "main_figure_4",
        "type": "figure",
        "number": "4",
        "label": "fig:replay-fragility",
        "title": "Replay fragility and lead-in analysis",
    },
    {
        "id": "main_figure_5",
        "type": "figure",
        "number": "5",
        "label": "fig:neural-steganalysis",
        "title": "Neural steganalysis and generalization",
    },
    {
        "id": "main_table_1",
        "type": "table",
        "number": "1",
        "label": "tab:study-recovery",
        "title": "Study design and primary recovery summary",
    },
    {
        "id": "main_table_2",
        "type": "table",
        "number": "2",
        "label": "tab:effects-runtime",
        "title": "Primary effect sizes and computational performance",
    },
)


SUPPLEMENTARY_TABLE_TITLES: Tuple[str, ...] = (
    "Protocol and variant definitions",
    "Models, tokenizers, revisions, quantization, and hardware",
    "Payload classes, lengths, algorithms, and corpus counts",
    "Complete recovery matrix",
    "Full mixed-effects results",
    "Prompt-category and multilingual results",
    "Filter and tail ablations",
    "Lead-in and replay-boundary failures",
    "Transmission robustness",
    "Human evaluation and inter-rater reliability",
    "Steganalysis metrics",
    "Runtime and memory measurements",
    "Failure taxonomy",
)


SUPPLEMENTARY_FIGURE_TITLES: Tuple[str, ...] = (
    "Experiment flow and completed matrix",
    "Full direct-rank distributions by payload class and model",
    "Per-model capacity--quality frontiers",
    "Filter ablation",
    "Lead-in-length sweep and first-divergence positions",
    "Tail-policy and segment-size comparison",
    "Transmission-transformation results",
    "Human-rating distributions",
    "Readability and held-out evaluator metrics",
    "Detector ROC and precision--recall curves",
    "Held-out model, prompt, and codec detection",
    "Computational overhead",
    "Multilingual secondary experiment",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionReportingError("Cannot read JSON manifest {}: {}".format(path, exc))


def _safe_int(value: Any, label: str, allow_zero: bool = True) -> int:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        raise RevisionReportingError("{} must be an integer, got {!r}".format(label, value))
    integer = int(numeric)
    if not math.isfinite(numeric) or numeric != integer:
        raise RevisionReportingError("{} must be an integer, got {!r}".format(label, value))
    if integer < 0 or (not allow_zero and integer == 0):
        raise RevisionReportingError("{} is outside the permitted range".format(label))
    return integer


def _safe_float(value: Any, label: str) -> float:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        raise RevisionReportingError("{} must be numeric, got {!r}".format(label, value))
    if not math.isfinite(numeric):
        raise RevisionReportingError("{} must be finite".format(label))
    return numeric



def _safe_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise RevisionReportingError("{} must be a Boolean, got {!r}".format(label, value))


def _validate_payload_fidelity_contract(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RevisionReportingError("{} lacks payload_fidelity_contract".format(label))
    expected = {
        "contract_version": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "semantics": PAYLOAD_RECOVERY_SEMANTICS,
        "primary_outcome": PAYLOAD_RECOVERY_OUTCOME,
        "compatibility_alias": "exact_recovery",
        "exact_rank_replay_role": "diagnostic_only",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise RevisionReportingError(
                "{} payload_fidelity_contract {} must equal {!r}".format(
                    label, field, expected_value
                )
            )
    if value.get("alias_equality_validated") is not True:
        raise RevisionReportingError(
            "{} payload_fidelity_contract did not validate alias equality".format(label)
        )
    direct_rows = _safe_int(
        value.get("direct_rows"), "{} payload_fidelity_contract direct_rows".format(label)
    )
    verified = _safe_int(
        value.get("direct_rows_contract_verified"),
        "{} payload_fidelity_contract direct_rows_contract_verified".format(label),
    )
    if direct_rows != verified:
        raise RevisionReportingError(
            "{} payload_fidelity_contract did not verify every direct row".format(label)
        )
    return value


def _validate_superseding_evidence_phase(
    row: Mapping[str, Any],
    *,
    label: str,
    scoped: bool = False,
    allow_evaluator: bool = False,
) -> None:
    suffix = "_scope" if scoped else ""
    evidence = str(row.get("evidence_status" + suffix, "")).strip()
    phase = str(row.get("study_phase" + suffix, "")).strip()
    allowed = set(SUPERSEDING_EVIDENCE_PHASES.get(evidence, set()))
    if allow_evaluator:
        allowed.update(EVALUATOR_EVIDENCE_PHASES.get(evidence, set()))
    if phase not in allowed:
        raise RevisionReportingError(
            "{} has a legacy or mismatched evidence/study-phase label".format(label)
        )


def _is_direct_subword_row(row: Mapping[str, Any]) -> bool:
    values = {
        str(row.get("protocol_variant", "")).strip().lower(),
        str(row.get("representation_name", "")).strip().lower(),
        str(row.get("codec_id", "")).strip().lower(),
    }
    return bool(
        values
        & {
            "direct_subword_calgacus",
            "direct_subword",
            "raw_subword_direct",
        }
    )

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [
                {str(key): "" if value is None else str(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error) as exc:
        raise RevisionReportingError("Cannot read CSV artifact {}: {}".format(path, exc))


def _read_jsonl_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RevisionReportingError(
                        "JSONL row {}:{} is not an object".format(path, line_number)
                    )
                rows.append(
                    {
                        str(key): (
                            json.dumps(item, sort_keys=True, ensure_ascii=False)
                            if isinstance(item, (dict, list))
                            else "" if item is None else str(item)
                        )
                        for key, item in value.items()
                    }
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionReportingError("Cannot read JSONL artifact {}: {}".format(path, exc))
    return rows


def _load_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_rows(path)
    return None


def _resolve_declared_path(manifest_path: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise RevisionReportingError("Artifact path in {} is missing".format(manifest_path))
    raw = Path(declared)
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()
    candidate = (manifest_path.parent / raw).resolve()
    try:
        candidate.relative_to(manifest_path.parent.resolve())
    except ValueError:
        # Absolute manifests are portable by basename when copied as a directory.
        candidate = (manifest_path.parent / raw.name).resolve()
    if candidate.is_file():
        return candidate
    sibling = (manifest_path.parent / raw.name).resolve()
    if sibling.is_file():
        return sibling
    raise RevisionReportingError(
        "Manifest {} declares missing artifact {}".format(manifest_path, declared)
    )


def _verify_file(
    *,
    source_kind: str,
    logical_name: str,
    manifest_path: Path,
    declared_path: Any,
    declared_sha256: Optional[Any],
    declared_bytes: Optional[Any],
    declared_rows: Optional[Any],
) -> Tuple[VerifiedArtifact, Optional[List[Dict[str, str]]], Optional[Any]]:
    path = _resolve_declared_path(manifest_path, declared_path)
    if path.is_symlink():
        raise RevisionReportingError("Report inputs may not be symlinks: {}".format(path))
    actual_hash = file_sha256(path)
    actual_size = path.stat().st_size
    if declared_sha256 is not None and actual_hash != str(declared_sha256):
        raise RevisionReportingError(
            "SHA-256 mismatch for {}.{}: expected {}, observed {}".format(
                source_kind, logical_name, declared_sha256, actual_hash
            )
        )
    if declared_bytes is not None and actual_size != _safe_int(
        declared_bytes, "{}.{}.bytes".format(source_kind, logical_name)
    ):
        raise RevisionReportingError(
            "Byte-count mismatch for {}.{}".format(source_kind, logical_name)
        )
    rows = _load_rows(path)
    json_value = None
    if rows is None and path.suffix.lower() == ".json":
        json_value = _read_json(path)
    row_count = (
        len(rows)
        if rows is not None
        else len(json_value)
        if isinstance(json_value, list)
        else None
    )
    if declared_rows is not None and row_count is not None:
        if row_count != _safe_int(
            declared_rows, "{}.{}.row_count".format(source_kind, logical_name)
        ):
            raise RevisionReportingError(
                "Row-count mismatch for {}.{}".format(source_kind, logical_name)
            )
    return (
        VerifiedArtifact(
            source_kind=source_kind,
            logical_name=logical_name,
            path=path,
            sha256=actual_hash,
            size_bytes=actual_size,
            row_count=row_count,
            manifest_declared_sha256=declared_sha256 is not None,
        ),
        rows,
        json_value,
    )


def _statistics_entries(manifest: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    if str(manifest.get("schema_version", "")) != "1.0":
        raise RevisionReportingError("Unsupported statistics manifest schema_version")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise RevisionReportingError("Statistics manifest outputs must be a non-empty object")
    required = {"recovery", "continuous", "effects", "integrity"}
    missing = sorted(required - set(outputs))
    if missing:
        raise RevisionReportingError(
            "Statistics manifest lacks required outputs: {}".format(", ".join(missing))
        )
    malformed = sorted(name for name, value in outputs.items() if not isinstance(value, dict))
    if malformed:
        raise RevisionReportingError(
            "Malformed statistics output declarations: {}".format(", ".join(malformed))
        )
    return [(str(name), value) for name, value in sorted(outputs.items())]


def _theory_entries(manifest: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    if str(manifest.get("schema_version", "")) != "1.0":
        raise RevisionReportingError("Unsupported theory manifest schema_version")
    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise RevisionReportingError("Theory manifest tables must be a list")
    entries: List[Tuple[str, Mapping[str, Any]]] = []
    for value in tables:
        if not isinstance(value, dict) or not value.get("name"):
            raise RevisionReportingError("Malformed theory table declaration")
        entries.append((str(value["name"]), value))
    required = {
        "capacity_validation",
        "capacity_plot",
        "quality_validation",
        "quality_plot",
        "exact_recovery",
        "cascade",
    }
    missing = sorted(required - {name for name, _ in entries})
    if missing:
        raise RevisionReportingError(
            "Theory manifest lacks required tables: {}".format(", ".join(missing))
        )
    return sorted(entries)


DETECTOR_OUTPUTS: Tuple[Tuple[str, str], ...] = (
    ("metrics", "detector_metrics.csv"),
    ("predictions", "detector_predictions.csv"),
    ("dataset", "detector_dataset_manifest.csv"),
    ("splits", "detector_split_manifest.json"),
    ("failures", "detector_failures.json"),
)


def _detector_entries(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> List[Tuple[str, Mapping[str, Any]]]:
    schema = str(manifest.get("schema_version", ""))
    if schema not in {
        "rankcloak-revision-detector-run-v1",
        "rankcloak-revision-detector-run-v2",
    }:
        raise RevisionReportingError("Unsupported detector manifest schema_version")
    if bool(manifest.get("smoke")):
        # The caller decides whether fixtures may be used; this flag is checked later.
        pass
    count_fields = {
        "metrics": "metric_rows",
        "predictions": "prediction_rows",
        "dataset": "normalized_rows",
        "failures": "failure_count",
    }
    entries: List[Tuple[str, Mapping[str, Any]]] = []
    declared_outputs = manifest.get("output_files")
    if schema == "rankcloak-revision-detector-run-v2":
        expected_filenames = {filename for _, filename in DETECTOR_OUTPUTS}
        if not isinstance(declared_outputs, dict) or set(declared_outputs) != expected_filenames:
            raise RevisionReportingError(
                "Detector v2 output_files must declare exactly {}".format(
                    ", ".join(sorted(expected_filenames))
                )
            )
    for logical_name, filename in DETECTOR_OUTPUTS:
        path = manifest_path.parent / filename
        if not path.is_file():
            raise RevisionReportingError("Detector output is missing: {}".format(path))
        identity: Dict[str, Any] = {}
        if schema == "rankcloak-revision-detector-run-v2":
            value = declared_outputs.get(filename)  # type: ignore[union-attr]
            if not isinstance(value, dict) or value.get("sha256") is None or value.get("size_bytes") is None:
                raise RevisionReportingError(
                    "Detector v2 output {} lacks SHA-256/size identity".format(filename)
                )
            identity = dict(value)
        entries.append(
            (
                logical_name,
                {
                    **identity,
                    "path": filename,
                    "row_count": manifest.get(count_fields.get(logical_name, "")),
                },
            )
        )
    return entries


MIXED_MODEL_OUTPUTS = {
    "coefficients",
    "contrasts",
    "diagnostics",
    "wilson",
    "dispersion",
    "status",
}


def _frozen_model_pins() -> Dict[str, str]:
    value = _read_json(MODELS_CONFIG)
    rows = value.get("models") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise RevisionReportingError("Frozen models.json is malformed")
    pins = {
        str(row.get("model_id", "")): str(row.get("artifact_sha256", ""))
        for row in rows
        if isinstance(row, dict)
    }
    if (
        len(pins) != 3
        or "" in pins
        or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in pins.values())
    ):
        raise RevisionReportingError("Frozen evaluator artifact pins are malformed")
    return pins


def _mixed_model_entries(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> List[Tuple[str, Mapping[str, Any]]]:
    if str(manifest.get("schema_version", "")) != "1.0" or manifest.get(
        "manifest_type"
    ) != "rankcloak_revision_v1_mixed_model_run":
        raise RevisionReportingError("Unsupported mixed-model run manifest")
    if manifest.get("plan_id") != (
        "rankcloak_revision_primary_v2_prespecified_confirmatory_models"
    ):
        raise RevisionReportingError("Mixed-model manifest does not use the frozen primary plan")
    if not CONFIRMATORY_MODEL_PLAN.is_file() or manifest.get("plan_sha256") != file_sha256(
        CONFIRMATORY_MODEL_PLAN
    ):
        raise RevisionReportingError("Mixed-model manifest plan hash is not the frozen plan")
    if not R_ENVIRONMENT_LOCK.is_file() or manifest.get(
        "environment_lock_sha256"
    ) != file_sha256(R_ENVIRONMENT_LOCK):
        raise RevisionReportingError("Mixed-model manifest environment lock hash differs")
    if manifest.get("validation_only") is not False:
        raise RevisionReportingError("Validation-only R output cannot be primary inference")
    if manifest.get("analysis_unit") != "payload_trial":
        raise RevisionReportingError("Mixed-model analysis unit must be payload_trial")
    if manifest.get("segments_as_independent_observations") is not False:
        raise RevisionReportingError("Mixed-model output permits segment pseudoreplication")
    if manifest.get("fixed_effects_fallback") is not False:
        raise RevisionReportingError("Mixed-model output used or permits fixed-effects fallback")
    _validate_payload_fidelity_contract(
        manifest.get("payload_fidelity_contract"), label="Mixed-model manifest"
    )
    inputs = manifest.get("input_files")
    if not isinstance(inputs, list) or not all(isinstance(row, dict) for row in inputs):
        raise RevisionReportingError("Mixed-model input_files must be a list")
    by_role = {str(row.get("role", "")): row for row in inputs}
    if len(by_role) != len(inputs):
        raise RevisionReportingError("Mixed-model input roles must be unique")
    required_inputs = {
        "driver_source",
        "plan",
        "environment_lock",
        "trials",
        "features",
        "feature_join_manifest",
        "runtime",
    }
    if not required_inputs.issubset(by_role):
        raise RevisionReportingError(
            "Mixed-model manifest lacks required inputs: {}".format(
                ", ".join(sorted(required_inputs - set(by_role)))
            )
        )
    verified_input_paths: Dict[str, Path] = {}
    for role, declaration in by_role.items():
        path = _resolve_declared_path(manifest_path, declaration.get("path"))
        if declaration.get("sha256") != file_sha256(path):
            raise RevisionReportingError(
                "Mixed-model input {} SHA-256 differs".format(role)
            )
        if declaration.get("size_bytes") is not None and path.stat().st_size != _safe_int(
            declaration.get("size_bytes"), "mixed-model input {} bytes".format(role)
        ):
            raise RevisionReportingError(
                "Mixed-model input {} byte count differs".format(role)
            )
        verified_input_paths[role] = path
    if (
        verified_input_paths["driver_source"] != R_MIXED_MODEL_DRIVER.resolve()
        or by_role["driver_source"].get("sha256")
        != file_sha256(R_MIXED_MODEL_DRIVER)
    ):
        raise RevisionReportingError(
            "Mixed-model driver is not the locked repository R source"
        )
    join_manifest = _read_json(verified_input_paths["feature_join_manifest"])
    if not isinstance(join_manifest, dict) or (
        join_manifest.get("schema_version")
        != "rankcloak-revision-heldout-feature-join-v1"
        or join_manifest.get("manifest_type")
        != "rankcloak_revision_primary_heldout_feature_join"
        or join_manifest.get("primary_trial_count") != 6480
        or join_manifest.get("unmatched_primary_trials") != 0
        or join_manifest.get("source_record_hashes_recomputed") is not True
        or join_manifest.get("evaluator_source_records_byte_identical_to_preprocessing")
        is not True
        or join_manifest.get("evaluator_artifact_pins_verified") is not True
        or join_manifest.get("models_config_sha256") != file_sha256(MODELS_CONFIG)
        or join_manifest.get("evaluator_artifact_pins") != _frozen_model_pins()
    ):
        raise RevisionReportingError("R input feature join failed its primary contract")
    joined_output = join_manifest.get("outputs", {}).get("features")
    if not isinstance(joined_output, dict) or (
        joined_output.get("sha256") != by_role["features"].get("sha256")
        or file_sha256(verified_input_paths["features"])
        != joined_output.get("sha256")
    ):
        raise RevisionReportingError("R feature input is not the joined evaluator table")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != MIXED_MODEL_OUTPUTS:
        raise RevisionReportingError(
            "Mixed-model outputs must declare exactly {}".format(
                ", ".join(sorted(MIXED_MODEL_OUTPUTS))
            )
        )
    entries: List[Tuple[str, Mapping[str, Any]]] = []
    for name in sorted(outputs):
        declaration = outputs[name]
        if not isinstance(declaration, dict) or declaration.get("sha256") is None:
            raise RevisionReportingError(
                "Mixed-model output {} lacks a SHA-256 declaration".format(name)
            )
        entries.append((str(name), declaration))
    return entries


def _evaluator_unavailability_entries(
    manifest: Mapping[str, Any],
) -> List[Tuple[str, Mapping[str, Any]]]:
    unsigned = dict(manifest)
    claimed_manifest_hash = unsigned.pop("manifest_sha256", None)
    required = {
        "schema_version": EVALUATOR_UNAVAILABILITY_SCHEMA,
        "manifest_type": EVALUATOR_UNAVAILABILITY_MANIFEST_TYPE,
        "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
        "result_schema_revision": RESULT_SCHEMA_REVISION,
        "authorized_projection_sha256": AUTHORIZED_PROJECTION_SHA256,
        "frozen_evaluator_target_units": FROZEN_EVALUATOR_TARGET_UNITS,
        "scoreable_evaluator_units": SCOREABLE_EVALUATOR_UNITS,
        "upstream_dependent_unavailable_units": UPSTREAM_UNAVAILABLE_EVALUATOR_UNITS,
        "terminal_accounted_units": FROZEN_EVALUATOR_TARGET_UNITS,
        "scoring_attempted_for_unavailable_units": False,
        "scores_imputed_or_fabricated": False,
        "analysis_policy": (
            "terminal_design_units_excluded_from_quality_estimands_and_not_scored"
        ),
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RevisionReportingError(
            "Evaluator-unavailability manifest violates the frozen accounting contract"
        )
    if claimed_manifest_hash != compact_json_sha256(unsigned):
        raise RevisionReportingError(
            "Evaluator-unavailability manifest self-hash differs"
        )
    files = manifest.get("source_files")
    if (
        not isinstance(files, list)
        or len(files) != 4
        or manifest.get("source_files_sha256") != compact_json_sha256(files)
    ):
        raise RevisionReportingError(
            "Evaluator-unavailability source-file list hash differs"
        )
    by_role = {
        str(row.get("role", "")): row for row in files if isinstance(row, dict)
    }
    if set(by_role) != {"plan", "checkpoint", "records", "run_identity"} or len(
        by_role
    ) != len(files):
        raise RevisionReportingError(
            "Evaluator-unavailability source roles are incomplete or duplicated"
        )
    for role, row in by_role.items():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
            or _safe_int(
                row.get("size_bytes"),
                "evaluator-unavailability {} bytes".format(role),
                allow_zero=False,
            )
            <= 0
        ):
            raise RevisionReportingError(
                "Evaluator-unavailability source identity is malformed"
            )
    units = manifest.get("units")
    if (
        not isinstance(units, list)
        or len(units) != UPSTREAM_UNAVAILABLE_EVALUATOR_UNITS
        or manifest.get("units_sha256") != compact_json_sha256(units)
    ):
        raise RevisionReportingError("Evaluator-unavailability unit-list hash differs")
    identifiers = []
    for unit in units:
        if not isinstance(unit, dict) or any(
            unit.get(key) != value
            for key, value in {
                "terminal_status": "upstream_dependent_unavailable_not_scored",
                "source_stage": "ablation_v2",
                "source_record_type": "condition_unavailable",
                "reason_code": "empty_isolated_roundtrip_vocabulary",
                "generator_model_id": "mistral_7b_instruct_v0_3_q4_k_m",
                "evaluator_model_id": "llama3_8b_instruct_q4_k_m",
                "scoring_attempted": False,
                "score_imputed": False,
            }.items()
        ):
            raise RevisionReportingError(
                "Evaluator-unavailability unit violates terminal semantics"
            )
        identifier = str(unit.get("source_work_id", ""))
        if (
            not identifier
            or not str(unit.get("protocol_variant", ""))
            or not str(unit.get("payload_name", ""))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(unit.get("source_record_sha256", ""))
            )
        ):
            raise RevisionReportingError(
                "Evaluator-unavailability unit identity is malformed"
            )
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise RevisionReportingError(
            "Evaluator-unavailability source work IDs are duplicated"
        )
    return sorted(by_role.items())


def _generic_entries(manifest: Mapping[str, Any], kind: str) -> List[Tuple[str, Mapping[str, Any]]]:
    if not str(manifest.get("schema_version", "")).strip():
        raise RevisionReportingError("{} manifest lacks schema_version".format(kind.capitalize()))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        # A runner environment manifest is metadata-only, not numeric evidence.
        if kind == "runtime" and "packages" in manifest:
            return []
        raise RevisionReportingError(
            "{} manifest must declare a non-empty outputs object".format(kind.capitalize())
        )
    entries = []
    for name, value in sorted(outputs.items()):
        if not isinstance(value, dict):
            raise RevisionReportingError("Malformed {} output {}".format(kind, name))
        if value.get("sha256") is None:
            raise RevisionReportingError(
                "Generic {} output {} lacks a declared SHA-256".format(kind, name)
            )
        entries.append((str(name), value))
    return entries


def _preprocessing_entries(
    manifest: Mapping[str, Any],
) -> List[Tuple[str, Mapping[str, Any]]]:
    if str(manifest.get("schema_version", "")) != "2.0":
        raise RevisionReportingError(
            "Unsupported preprocessing manifest schema_version; payload_fidelity_v2 "
            "requires preprocessing schema 2.0"
        )
    if manifest.get("manifest_type") != "revision_preprocessing_outputs":
        raise RevisionReportingError("Expected a preprocessing output manifest")
    invariants = manifest.get("invariants")
    if not isinstance(invariants, dict) or not (
        invariants.get("unavailable_rows_excluded_from_estimands") is True
        and invariants.get("unavailable_rows_are_not_recovery_failures") is True
    ):
        raise RevisionReportingError(
            "Preprocessing manifest lacks unavailable-row exclusion invariants"
        )
    _validate_payload_fidelity_contract(
        invariants.get("payload_fidelity_contract"),
        label="Preprocessing manifest",
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RevisionReportingError("Preprocessing manifest outputs must be a non-empty list")
    entries: List[Tuple[str, Mapping[str, Any]]] = []
    for value in outputs:
        if not isinstance(value, dict) or not value.get("role"):
            raise RevisionReportingError("Malformed preprocessing output declaration")
        if value.get("sha256") is None:
            raise RevisionReportingError(
                "Preprocessing output {} lacks SHA-256".format(value.get("role"))
            )
        entries.append((str(value["role"]), value))
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise RevisionReportingError("Duplicate preprocessing output role")
    if "unavailable" not in names:
        raise RevisionReportingError(
            "Preprocessing manifest lacks the unavailable-condition artifact"
        )
    row_counts = manifest.get("row_counts")
    unavailable_declaration = dict(entries[names.index("unavailable")][1])
    if not isinstance(row_counts, dict) or _safe_int(
        row_counts.get("unavailable"), "preprocessing unavailable row count"
    ) != _safe_int(
        unavailable_declaration.get("row_count"),
        "preprocessing unavailable declaration row count",
    ):
        raise RevisionReportingError("Preprocessing unavailable row-count mismatch")
    return sorted(entries)


def load_verified_sources(
    *,
    statistics_manifest: Optional[Path] = None,
    theory_manifest: Optional[Path] = None,
    detector_manifest: Optional[Path] = None,
    mixed_model_manifest: Optional[Path] = None,
    evaluator_unavailability_manifest: Optional[Path] = None,
    runtime_manifests: Sequence[Path] = (),
    preprocessing_manifests: Sequence[Path] = (),
    fixture_mode: bool = False,
) -> VerifiedSources:
    """Load only manifest-addressed sources and verify all available digests.

    Detector v1 run manifests predate an output-digest field. Those fixed-name
    outputs are verified against manifest row counts and sealed by digest in the
    report source manifest. Detector v2, R mixed models, statistics, theory, and
    generic runtime artifacts must match their manifest-declared digests.
    """

    requested: List[Tuple[str, Path]] = []
    for kind, value in (
        ("statistics", statistics_manifest),
        ("theory", theory_manifest),
        ("detector", detector_manifest),
        ("mixed_model", mixed_model_manifest),
        ("evaluator_unavailability", evaluator_unavailability_manifest),
    ):
        if value is not None:
            requested.append((kind, Path(value)))
    requested.extend(("runtime_{}".format(index + 1), Path(path)) for index, path in enumerate(runtime_manifests))
    requested.extend(
        ("preprocess_{}".format(index + 1), Path(path))
        for index, path in enumerate(preprocessing_manifests)
    )
    if not requested:
        raise RevisionReportingError("At least one machine-output manifest is required")

    result = VerifiedSources({}, {}, {}, {}, {}, fixture_mode=fixture_mode)
    manifest_hashes: Dict[str, Path] = {}
    for source_key, raw_manifest_path in requested:
        manifest_path = raw_manifest_path.resolve()
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RevisionReportingError("Invalid manifest path: {}".format(manifest_path))
        manifest_hash = file_sha256(manifest_path)
        if manifest_hash in manifest_hashes:
            raise RevisionReportingError(
                "Byte-identical manifests were supplied twice: {} and {}".format(
                    manifest_hashes[manifest_hash], manifest_path
                )
            )
        manifest_hashes[manifest_hash] = manifest_path
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RevisionReportingError("Manifest is not a JSON object: {}".format(manifest_path))
        base_kind = source_key.split("_", 1)[0]
        if source_key == "mixed_model":
            entries = _mixed_model_entries(manifest_path, manifest)
        elif source_key == "evaluator_unavailability":
            entries = _evaluator_unavailability_entries(manifest)
        elif base_kind == "statistics":
            if bool(manifest.get("smoke_fixture")):
                # Older manifests place this in the integrity artifact, checked below.
                if not fixture_mode:
                    raise RevisionReportingError("Smoke fixtures require fixture_mode=True")
            entries = _statistics_entries(manifest)
        elif base_kind == "theory":
            entries = _theory_entries(manifest)
        elif base_kind == "detector":
            if bool(manifest.get("smoke")) and not fixture_mode:
                raise RevisionReportingError("Detector smoke output cannot enter a scientific report")
            entries = _detector_entries(manifest_path, manifest)
        elif base_kind == "preprocess":
            entries = _preprocessing_entries(manifest)
        else:
            entries = _generic_entries(manifest, "runtime")
        result.manifests[source_key] = dict(manifest)
        result.manifest_paths[source_key] = manifest_path
        for logical_name, declaration in entries:
            artifact_key = "{}.{}".format(source_key, logical_name)
            artifact, rows, json_value = _verify_file(
                source_kind=source_key,
                logical_name=logical_name,
                manifest_path=manifest_path,
                declared_path=declaration.get("path"),
                declared_sha256=declaration.get("sha256"),
                declared_bytes=declaration.get("bytes", declaration.get("size_bytes")),
                declared_rows=declaration.get("row_count"),
            )
            result.artifacts[artifact_key] = artifact
            if rows is not None:
                result.tables[artifact_key] = rows
            if json_value is not None:
                result.json_values[artifact_key] = json_value
        if not entries and base_kind == "runtime":
            result.json_values["{}.manifest".format(source_key)] = dict(manifest)

    _validate_source_integrity(result)
    return result


def _artifact_rows(sources: VerifiedSources, suffix: str) -> Tuple[List[Dict[str, str]], Optional[VerifiedArtifact]]:
    matches = [(key, rows) for key, rows in sources.tables.items() if key.endswith("." + suffix)]
    if not matches:
        return [], None
    if len(matches) > 1:
        # Runtime shards may intentionally share a logical table; concatenate with lineage.
        combined: List[Dict[str, str]] = []
        for key, rows in sorted(matches):
            artifact = sources.artifacts[key]
            for row in rows:
                value = dict(row)
                value["_source_key"] = key
                value["_source_sha256"] = artifact.sha256
                combined.append(value)
        return combined, None
    key, rows = matches[0]
    return [dict(row) for row in rows], sources.artifacts[key]


def _exact_artifact_rows(
    sources: VerifiedSources, key: str
) -> Tuple[List[Dict[str, str]], Optional[VerifiedArtifact]]:
    """Return one named artifact without suffix matching across source classes."""

    rows = sources.tables.get(key)
    artifact = sources.artifacts.get(key)
    if rows is None:
        return [], artifact
    return [dict(row) for row in rows], artifact


def _runtime_summary_rows(sources: VerifiedSources) -> List[Dict[str, str]]:
    """Return optional generic runtime summaries, excluding statistics products."""

    rows: List[Dict[str, str]] = []
    for key, values in sorted(sources.tables.items()):
        if not key.startswith("runtime_"):
            continue
        logical_name = key.split(".", 1)[1].lower()
        if not any(token in logical_name for token in ("runtime", "profile", "summary", "performance")):
            continue
        artifact = sources.artifacts[key]
        for row in values:
            value = dict(row)
            value["_source_key"] = key
            value["_source_sha256"] = artifact.sha256
            rows.append(value)
    return rows



def _require_revision_fields(
    row: Mapping[str, Any], *, label: str, scoped: bool = False
) -> None:
    suffix = "_scope" if scoped else ""
    expected = (
        ("protocol_contract_revision" + suffix, PROTOCOL_CONTRACT_REVISION),
        ("result_schema_revision" + suffix, RESULT_SCHEMA_REVISION),
    )
    for field, expected_value in expected:
        if str(row.get(field, "")).strip() != expected_value:
            raise RevisionReportingError(
                "{} {} must equal {!r}".format(label, field, expected_value)
            )


def _validate_statistics_recovery_contract(
    sources: VerifiedSources, integrity: Mapping[str, Any]
) -> None:
    if "statistics" not in sources.manifests:
        return
    contract = _validate_payload_fidelity_contract(
        integrity.get("payload_fidelity_contract"),
        label="Statistics integrity report",
    )
    primary_scope = integrity.get("primary_effect_scope")
    if not isinstance(primary_scope, dict):
        raise RevisionReportingError(
            "Statistics integrity report lacks primary_effect_scope"
        )
    for field, expected_value in (
        ("replay_mode", "saved_token_ids"),
        ("transformation_id", "unmodified"),
        ("mitigation_id", "none"),
        ("evidence_status", PRIMARY_EVIDENCE_STATUS),
        ("study_phase", PRIMARY_STUDY_PHASE),
        ("protocol_contract_revision", PROTOCOL_CONTRACT_REVISION),
        ("result_schema_revision", RESULT_SCHEMA_REVISION),
    ):
        if primary_scope.get(field) != expected_value:
            raise RevisionReportingError(
                "Statistics primary_effect_scope {} must equal {!r}".format(
                    field, expected_value
                )
            )
    if primary_scope.get("diagnostic_replay_fallback") is not False:
        raise RevisionReportingError(
            "Statistics primary effects permit a diagnostic replay fallback"
        )
    if primary_scope.get("pairwise_effects_are_primary_inference") is not False:
        raise RevisionReportingError(
            "Generic pairwise effects cannot be primary inference"
        )

    recovery, _ = _artifact_rows(sources, "recovery")
    for index, row in enumerate(recovery, start=1):
        label = "Statistics recovery row {}".format(index)
        _require_revision_fields(row, label=label)
        _validate_superseding_evidence_phase(row, label=label)
        if row.get("recovery_outcome") != PAYLOAD_RECOVERY_OUTCOME:
            raise RevisionReportingError(
                "{} does not identify exact payload recovery as its outcome".format(label)
            )
        if row.get("recovery_outcome_semantics") != PAYLOAD_RECOVERY_SEMANTICS:
            raise RevisionReportingError(
                "{} has ambiguous recovery semantics".format(label)
            )
        if not _safe_bool(
            row.get("exact_recovery_compatibility_alias"),
            label + " exact_recovery_compatibility_alias",
        ):
            raise RevisionReportingError(
                "{} did not validate the compatibility alias".format(label)
            )
        if not _safe_bool(
            row.get("rank_replay_diagnostic_only"),
            label + " rank_replay_diagnostic_only",
        ):
            raise RevisionReportingError(
                "{} treats rank replay as a scientific recovery outcome".format(label)
            )
        payload_successes = _safe_int(
            row.get("payload_recovery_successes"),
            label + " payload_recovery_successes",
        )
        alias_successes = _safe_int(row.get("successes"), label + " successes")
        if payload_successes != alias_successes:
            raise RevisionReportingError(
                "{} payload successes differ from the compatibility alias".format(label)
            )
        payload_rate = _safe_float(
            row.get("exact_payload_recovery_rate"),
            label + " exact_payload_recovery_rate",
        )
        alias_rate = _safe_float(
            row.get("exact_recovery_rate"), label + " exact_recovery_rate"
        )
        if not math.isclose(payload_rate, alias_rate, rel_tol=1e-12, abs_tol=1e-12):
            raise RevisionReportingError(
                "{} payload rate differs from the compatibility alias".format(label)
            )
        rank_n = _safe_int(row.get("rank_replay_n"), label + " rank_replay_n")
        rank_successes = _safe_int(
            row.get("rank_replay_successes"), label + " rank_replay_successes"
        )
        if rank_successes > rank_n:
            raise RevisionReportingError(
                "{} rank replay successes exceed its diagnostic denominator".format(label)
            )

    effects, _ = _artifact_rows(sources, "effects")
    quality_outcomes = {
        "human_naturalness",
        "human_suspiciousness",
        "heldout_evaluator_log_probability",
        "mean_log_probability",
        "flesch_reading_ease_heuristic",
        "flesch_kincaid_grade_heuristic",
        "coleman_liau_index",
        "unique_word_fraction",
        "repeated_bigram_fraction",
        "surface_flag_total",
        "tfidf_prompt_similarity",
    }
    for index, row in enumerate(effects, start=1):
        label = "Statistics effect row {}".format(index)
        primary_inference = _safe_bool(
            row.get("primary_inference", False), label + " primary_inference"
        )
        inference_role = row.get("inference_role")
        if primary_inference:
            raise RevisionReportingError(
                "{} designates Python pooled/pairwise output as primary; primary "
                "inference must come from the locked R mixed-model manifest".format(label)
            )
        if not primary_inference and inference_role not in {
            "descriptive_exploratory_pairwise",
            "",
            None,
        }:
            raise RevisionReportingError(
                "{} has an unsupported generic-effect inference role".format(label)
            )
        if row.get("comparison_design") == "partially_overlapping_payload":
            if _safe_bool(
                row.get("inferential_p_value_supported", False),
                label + " inferential_p_value_supported",
            ):
                raise RevisionReportingError(
                    "{} claims a p-value for partially overlapping payload sets".format(label)
                )
            if any(str(row.get(field, "")).strip() for field in (
                "p_value_raw", "p_value_holm", "p_value_bh"
            )):
                raise RevisionReportingError(
                    "{} exposes unsupported pairwise p-values".format(label)
                )
        outcome = str(row.get("outcome", ""))
        if outcome in {"exact_recovery", PAYLOAD_RECOVERY_OUTCOME}:
            _require_revision_fields(row, label=label)
            _require_revision_fields(row, label=label, scoped=True)
            _validate_superseding_evidence_phase(row, label=label, scoped=True)
            if row.get("recovery_outcome") != PAYLOAD_RECOVERY_OUTCOME:
                raise RevisionReportingError(
                    "{} does not identify exact payload recovery".format(label)
                )
            if row.get("recovery_outcome_semantics") != PAYLOAD_RECOVERY_SEMANTICS:
                raise RevisionReportingError(
                    "{} has ambiguous recovery semantics".format(label)
                )
            if not _safe_bool(
                row.get("exact_recovery_compatibility_alias"),
                label + " exact_recovery_compatibility_alias",
            ):
                raise RevisionReportingError(
                    "{} did not validate the recovery alias".format(label)
                )
            if not _safe_bool(
                row.get("exact_rank_replay_diagnostic_only"),
                label + " exact_rank_replay_diagnostic_only",
            ):
                raise RevisionReportingError(
                    "{} promotes rank replay beyond a diagnostic".format(label)
                )
        if outcome in quality_outcomes:
            _require_revision_fields(row, label=label, scoped=True)
            _validate_superseding_evidence_phase(
                row, label=label, scoped=True, allow_evaluator=True
            )
            views = [
                str(row.get(field, "")).strip()
                for field in ("view_scope", "text_view_scope", "span_type_scope")
                if str(row.get(field, "")).strip()
            ]
            if len(set(views)) != 1:
                raise RevisionReportingError(
                    "{} lacks one explicit forced-span/full-message view scope".format(label)
                )


def _validate_preprocessing_recovery_contract(sources: VerifiedSources) -> None:
    for source_key, manifest in sorted(sources.manifests.items()):
        if not source_key.startswith("preprocess_"):
            continue
        contract = _validate_payload_fidelity_contract(
            manifest.get("invariants", {}).get("payload_fidelity_contract"),
            label="{} manifest".format(source_key),
        )
        rows = sources.tables.get(source_key + ".trials", [])
        direct_rows = 0
        for index, row in enumerate(rows, start=1):
            label = "{} trial row {}".format(source_key, index)
            if row.get("preprocess_schema_version") != "2.0":
                raise RevisionReportingError(
                    "{} is not preprocessing schema 2.0".format(label)
                )
            _require_revision_fields(row, label=label)
            _validate_superseding_evidence_phase(row, label=label)
            if row.get("recovery_outcome_semantics") != PAYLOAD_RECOVERY_SEMANTICS:
                raise RevisionReportingError(
                    "{} has ambiguous recovery semantics".format(label)
                )
            payload = _safe_bool(
                row.get(PAYLOAD_RECOVERY_OUTCOME),
                label + " exact_payload_recovery",
            )
            alias = _safe_bool(row.get("exact_recovery"), label + " exact_recovery")
            if payload != alias:
                raise RevisionReportingError(
                    "{} recovery alias differs from payload recovery".format(label)
                )
            if _is_direct_subword_row(row):
                direct_rows += 1
                _safe_bool(
                    row.get("exact_rank_replay"), label + " exact_rank_replay"
                )
        expected_direct = _safe_int(
            contract.get("direct_rows"),
            "{} payload_fidelity_contract direct_rows".format(source_key),
        )
        if direct_rows != expected_direct:
            raise RevisionReportingError(
                "{} direct-row count disagrees with its manifest contract".format(
                    source_key
                )
            )

def _validate_source_integrity(sources: VerifiedSources) -> None:
    integrity = sources.json_values.get("statistics.integrity")
    if isinstance(integrity, dict):
        if integrity.get("status") != "passed":
            raise RevisionReportingError("Statistics integrity report did not pass")
        if integrity.get("analysis_unit") != "payload":
            raise RevisionReportingError("Statistics analysis unit must be payload")
        if bool(integrity.get("segments_as_independent_observations")):
            raise RevisionReportingError("Segment pseudoreplication is forbidden")
        if bool(integrity.get("smoke_fixture")) and not sources.fixture_mode:
            raise RevisionReportingError("Statistics smoke fixture cannot enter a scientific report")
    statistics_integrity = integrity if isinstance(integrity, dict) else {}
    _validate_statistics_recovery_contract(sources, statistics_integrity)
    _validate_preprocessing_recovery_contract(sources)
    _validate_statistics_sample_sizes(sources, statistics_integrity)
    _validate_theory_consistency(sources)
    _validate_detector_consistency(sources)
    _validate_mixed_model_consistency(sources)
    _validate_evaluator_unavailability_consistency(sources)
    _validate_generic_runtime(sources)


def _validate_statistics_sample_sizes(
    sources: VerifiedSources, integrity: Mapping[str, Any]
) -> None:
    recovery, _ = _artifact_rows(sources, "recovery")
    continuous, _ = _artifact_rows(sources, "continuous")
    effects, _ = _artifact_rows(sources, "effects")
    maxima = integrity.get("independent_payloads", {}) if isinstance(integrity, dict) else {}
    maximum_trials = int(maxima.get("trials", 0) or 0) if isinstance(maxima, dict) else 0
    maximum_runtime = int(maxima.get("runtime", 0) or 0) if isinstance(maxima, dict) else 0

    for index, row in enumerate(recovery, start=1):
        n = _safe_int(row.get("n_payloads"), "recovery row {} n_payloads".format(index), False)
        successes = _safe_int(
            row.get("payload_recovery_successes"),
            "recovery row {} payload_recovery_successes".format(index),
        )
        if successes > n:
            raise RevisionReportingError("Recovery successes exceed n_payloads at row {}".format(index))
        rate = _safe_float(
            row.get("exact_payload_recovery_rate"),
            "recovery row {} exact_payload_recovery_rate".format(index),
        )
        if not math.isclose(rate, successes / float(n), rel_tol=1e-9, abs_tol=1e-12):
            raise RevisionReportingError("Recovery rate is inconsistent with successes/n at row {}".format(index))
        low = _safe_float(row.get("wilson_ci_low"), "recovery row {} CI low".format(index))
        high = _safe_float(row.get("wilson_ci_high"), "recovery row {} CI high".format(index))
        if not (0.0 <= low <= rate <= high <= 1.0):
            raise RevisionReportingError("Recovery confidence interval is inconsistent at row {}".format(index))
        if row.get("analysis_unit") and row.get("analysis_unit") != "payload":
            raise RevisionReportingError("Recovery row does not use payload analysis units")
        if maximum_trials and n > maximum_trials:
            raise RevisionReportingError("Recovery n_payloads exceeds the integrity payload count")

    recovery_lookup: Dict[Tuple[Tuple[str, str], ...], int] = {}
    grouping = (
        "phase",
        "model_id",
        "protocol_variant",
        "prompt_category",
        "payload_class",
        "replay_mode",
        "transformation_id",
        "mitigation_id",
    )
    for row in recovery:
        key = tuple((column, row.get(column, "")) for column in grouping if row.get(column, "") != "")
        recovery_lookup[key] = _safe_int(row["n_payloads"], "recovery n_payloads", False)

    for index, row in enumerate(continuous, start=1):
        for column in ("mean", "standard_deviation", "median", "ci_low", "ci_high"):
            if row.get(column, "").strip():
                _safe_float(row[column], "continuous row {} {}".format(index, column))
        if all(row.get(column, "").strip() for column in ("mean", "ci_low", "ci_high")):
            mean = _safe_float(row["mean"], "continuous mean")
            low = _safe_float(row["ci_low"], "continuous CI low")
            high = _safe_float(row["ci_high"], "continuous CI high")
            if not low <= mean <= high:
                raise RevisionReportingError("Continuous confidence interval does not contain its mean")
        if not row.get("n_payloads", "").strip():
            continue
        n = _safe_int(row["n_payloads"], "continuous row {} n_payloads".format(index), False)
        if row.get("analysis_unit") and row.get("analysis_unit") != "payload":
            raise RevisionReportingError("Continuous row does not use payload analysis units")
        outcome = row.get("outcome", "")
        ceiling = maximum_runtime if outcome in {
            "payload_bits_per_second",
            "encoding_tokens_per_second",
            "decoding_tokens_per_second",
            "encoding_seconds",
            "decoding_seconds",
            "peak_ram_mib",
            "peak_gpu_memory_mib",
            "cover_tokens_per_payload_byte",
        } else maximum_trials
        if ceiling and n > ceiling:
            raise RevisionReportingError("Continuous n_payloads exceeds its integrity payload count")
        key = tuple((column, row.get(column, "")) for column in grouping if row.get(column, "") != "")
        if key in recovery_lookup and outcome == "effective_payload_rate" and n != recovery_lookup[key]:
            raise RevisionReportingError(
                "Recovery and effective-rate sample sizes disagree for {}".format(dict(key))
            )

    factor_to_column = {
        "protocol_variant": "protocol_variant",
        "model_id": "model_id",
        "prompt_category": "prompt_category",
    }
    for index, row in enumerate(effects, start=1):
        for column in (
            "mean_difference", "mean_difference_ci_low", "mean_difference_ci_high",
            "risk_difference", "risk_ratio", "odds_ratio_haldane_anscombe",
            "p_value_raw", "p_value_holm", "p_value_bh", "hedges_g",
        ):
            if row.get(column, "").strip():
                value = _safe_float(row[column], "effect row {} {}".format(index, column))
                if column.startswith("p_value") and not 0.0 <= value <= 1.0:
                    raise RevisionReportingError("Adjusted and raw p-values must lie in [0, 1]")
        if all(row.get(column, "").strip() for column in ("mean_difference", "mean_difference_ci_low", "mean_difference_ci_high")):
            estimate = _safe_float(row["mean_difference"], "effect estimate")
            low = _safe_float(row["mean_difference_ci_low"], "effect CI low")
            high = _safe_float(row["mean_difference_ci_high"], "effect CI high")
            if not low <= estimate <= high:
                raise RevisionReportingError("Effect confidence interval does not contain its estimate")
        for column in ("n_payloads_first", "n_payloads_second", "n_payloads_paired"):
            if row.get(column, "").strip():
                n = _safe_int(row[column], "effect row {} {}".format(index, column))
                if maximum_trials and n > maximum_trials:
                    raise RevisionReportingError("Effect sample size exceeds integrity payload count")
        paired = row.get("n_payloads_paired", "").strip()
        first = row.get("n_payloads_first", "").strip()
        second = row.get("n_payloads_second", "").strip()
        if paired and first and second:
            if _safe_int(paired, "paired n") > min(_safe_int(first, "first n"), _safe_int(second, "second n")):
                raise RevisionReportingError("Paired sample size exceeds a comparison arm")
        if row.get("outcome") in {"exact_recovery", PAYLOAD_RECOVERY_OUTCOME} and recovery:
            factor = row.get("factor", "")
            factor_column = factor_to_column.get(factor)
            if factor_column:
                for level_field, n_field in (
                    ("level_first", "n_payloads_first"),
                    ("level_second", "n_payloads_second"),
                ):
                    level = row.get(level_field, "")
                    if not level or not row.get(n_field, ""):
                        continue
                    available = sum(
                        _safe_int(item["n_payloads"], "recovery n")
                        for item in recovery
                        if item.get(factor_column) == level
                    )
                    if available and _safe_int(row[n_field], "effect n") > available:
                        raise RevisionReportingError(
                            "Effect sample size exceeds recovery evidence for {}={}".format(factor, level)
                        )


def _validate_theory_consistency(sources: VerifiedSources) -> None:
    manifest = sources.manifests.get("theory")
    if not isinstance(manifest, dict):
        return
    summary = manifest.get("summary", {})
    if not isinstance(summary, dict):
        raise RevisionReportingError("Theory summary must be an object")
    capacity_plot, _ = _artifact_rows(sources, "capacity_plot")
    quality_plot, _ = _artifact_rows(sources, "quality_plot")
    exact, _ = _artifact_rows(sources, "exact_recovery")
    cascade, _ = _artifact_rows(sources, "cascade")
    expected_pairs = (
        ("capacity_evaluable_count", len(capacity_plot)),
        ("quality_evaluable_count", len(quality_plot)),
        ("cascade_evaluable_count", len(cascade)),
    )
    for key, observed in expected_pairs:
        if key in summary and _safe_int(summary[key], "theory summary {}".format(key)) != observed:
            raise RevisionReportingError("Theory summary {} disagrees with table rows".format(key))
    input_count = _safe_int(summary.get("input_record_count", 0), "theory input_record_count")
    for key in (
        "quality_fully_bound_validated_count",
        "exact_proposition_confirmed_count",
        "exact_observed_only_count",
    ):
        if key in summary and _safe_int(summary[key], "theory {}".format(key)) > input_count:
            raise RevisionReportingError("Theory {} exceeds input count".format(key))
    if exact and len(exact) > input_count:
        raise RevisionReportingError("Theory exact-recovery rows exceed input records")

    capacity_validation, _ = _artifact_rows(sources, "capacity_validation")
    quality_validation, _ = _artifact_rows(sources, "quality_validation")
    metadata_columns = ("model_id", "protocol_variant", "payload_name")
    quality_by_trial = {row.get("trial_id"): row for row in quality_validation if row.get("trial_id")}
    for row in capacity_validation:
        trial_id = row.get("trial_id")
        other = quality_by_trial.get(trial_id)
        if other is None:
            continue
        for column in metadata_columns:
            if row.get(column) and other.get(column) and row[column] != other[column]:
                raise RevisionReportingError(
                    "Theory metadata differs across capacity/quality tables for trial {}".format(trial_id)
                )


def _validate_detector_preprocessing_binding(
    manifest: Mapping[str, Any], dataset_contract: Mapping[str, Any]
) -> None:
    binding = dataset_contract.get("preprocessing_binding")
    if not isinstance(binding, dict) or any(
        binding.get(key) != expected
        for key, expected in {
            "schema_version": (
                "rankcloak-revision-primary-detector-preprocessing-binding-v1"
            ),
            "strict_complete": True,
            "primary_shard_count": 3,
            "detector_row_count": 15840,
        }.items()
    ):
        raise RevisionReportingError(
            "Detector input lacks the strict primary preprocessing binding"
        )
    model_ids = sorted(map(str, dataset_contract.get("model_ids", [])))
    if binding.get("primary_model_ids") != model_ids or len(model_ids) != 3:
        raise RevisionReportingError(
            "Detector preprocessing binding has the wrong primary models"
        )
    preprocessing_path = Path(str(binding.get("preprocessing_manifest_path", "")))
    detector_path = Path(str(binding.get("detector_path", "")))
    input_manifest_path = Path(
        str(binding.get("preprocessing_input_manifest_path", ""))
    )
    for path, expected_hash, label in (
        (
            preprocessing_path,
            binding.get("preprocessing_manifest_sha256"),
            "preprocessing manifest",
        ),
        (detector_path, binding.get("detector_sha256"), "detector corpus"),
        (
            input_manifest_path,
            binding.get("preprocessing_input_manifest_sha256"),
            "preprocessing input manifest",
        ),
    ):
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != expected_hash
        ):
            raise RevisionReportingError(
                "Detector-bound {} is missing, unsafe, or hash-mismatched".format(
                    label
                )
            )
    if (
        str(Path(str(manifest.get("input_path", ""))).resolve())
        != str(detector_path.resolve())
        or manifest.get("input_sha256") != binding.get("detector_sha256")
        or manifest.get("preprocessing_manifest_sha256")
        != binding.get("preprocessing_manifest_sha256")
        or str(Path(str(manifest.get("preprocessing_manifest_path", ""))).resolve())
        != str(preprocessing_path.resolve())
        or detector_path.stat().st_size != binding.get("detector_size_bytes")
    ):
        raise RevisionReportingError(
            "Detector run identity differs from its preprocessing binding"
        )
    preprocessing = _read_json(preprocessing_path)
    outputs = preprocessing.get("outputs") if isinstance(preprocessing, dict) else None
    invariants = (
        preprocessing.get("invariants") if isinstance(preprocessing, dict) else None
    )
    if (
        not isinstance(outputs, list)
        or preprocessing.get("schema_version") != "2.0"
        or preprocessing.get("manifest_type") != "revision_preprocessing_outputs"
        or preprocessing.get("outputs_sha256") != compact_json_sha256(outputs)
        or preprocessing.get("input_manifest_sha256")
        != binding.get("preprocessing_input_manifest_sha256")
        or preprocessing.get("row_counts", {}).get("detector") != 15840
        or not isinstance(invariants, dict)
        or invariants.get("detector_pair_count") != 7920
        or invariants.get("detector_grouping_unit") != "payload_name"
    ):
        raise RevisionReportingError(
            "Detector-bound preprocessing output manifest is malformed"
        )
    by_role = {
        str(row.get("role", "")): row for row in outputs if isinstance(row, dict)
    }
    if len(by_role) != len(outputs) or not {"detector", "input_manifest"}.issubset(
        by_role
    ):
        raise RevisionReportingError(
            "Detector-bound preprocessing roles are incomplete or duplicated"
        )
    detector_declaration = by_role["detector"]
    declared_detector_path = _resolve_declared_path(
        preprocessing_path, detector_declaration.get("path")
    )
    if (
        declared_detector_path != detector_path.resolve()
        or detector_declaration.get("sha256") != binding.get("detector_sha256")
        or detector_declaration.get("size_bytes")
        != binding.get("detector_size_bytes")
        or detector_declaration.get("row_count") != 15840
    ):
        raise RevisionReportingError(
            "Detector bytes are not the preprocessing-declared detector output"
        )
    input_declaration = by_role["input_manifest"]
    if (
        _resolve_declared_path(preprocessing_path, input_declaration.get("path"))
        != input_manifest_path.resolve()
        or input_declaration.get("sha256")
        != binding.get("preprocessing_input_manifest_sha256")
    ):
        raise RevisionReportingError(
            "Detector preprocessing input-manifest declaration differs"
        )
    preprocessing_inputs = _read_json(input_manifest_path)
    if not isinstance(preprocessing_inputs, dict):
        raise RevisionReportingError(
            "Detector preprocessing input manifest must be a JSON object"
        )
    input_files = (
        preprocessing_inputs.get("input_files")
        if isinstance(preprocessing_inputs, dict)
        else None
    )
    shards = (
        preprocessing_inputs.get("run_shards")
        if isinstance(preprocessing_inputs, dict)
        else None
    )
    emitted = (
        [row for row in shards if isinstance(row, dict) and row.get("role") == "input"]
        if isinstance(shards, list)
        else []
    )
    if (
        preprocessing_inputs.get("schema_version") != "2.0"
        or preprocessing_inputs.get("manifest_type")
        != "revision_preprocessing_inputs"
        or preprocessing_inputs.get("strict_complete") is not True
        or preprocessing_inputs.get("emitted_run_count") != 3
        or preprocessing_inputs.get("reference_run_count") != 0
        or not isinstance(input_files, list)
        or preprocessing_inputs.get("input_files_sha256")
        != compact_json_sha256(input_files)
        or len(emitted) != 3
        or len(shards) != 3
        or sorted(str(row.get("model_id", "")) for row in emitted) != model_ids
        or any(
            row.get("stage") != "primary_v2"
            or row.get("evidence_status") != PRIMARY_EVIDENCE_STATUS
            or _safe_int(row.get("completed_work_units"), "completed work units")
            != _safe_int(row.get("planned_work_units"), "planned work units")
            for row in emitted
        )
    ):
        raise RevisionReportingError(
            "Detector preprocessing input lineage is not three complete primary shards"
        )


def _validate_detector_consistency(sources: VerifiedSources) -> None:
    manifest = sources.manifests.get("detector")
    if not isinstance(manifest, dict):
        return
    metrics, _ = _artifact_rows(sources, "metrics")
    predictions, _ = _artifact_rows(sources, "predictions")
    dataset, _ = _artifact_rows(sources, "dataset")
    schema = str(manifest.get("schema_version", ""))
    for field, observed in (
        ("metric_rows", len(metrics)),
        ("prediction_rows", len(predictions)),
        ("normalized_rows", len(dataset)),
    ):
        if field in manifest and _safe_int(manifest[field], "detector {}".format(field)) != observed:
            raise RevisionReportingError("Detector {} disagrees with output rows".format(field))
    counts: Dict[Tuple[str, str], int] = {}
    groups: Dict[Tuple[str, str], set] = {}
    prediction_labels: Dict[Tuple[str, str], Dict[int, int]] = {}
    for row in predictions:
        key = (row.get("split_id", ""), row.get("detector_name", ""))
        counts[key] = counts.get(key, 0) + 1
        groups.setdefault(key, set()).add(row.get("payload_group_id", ""))
        label = _safe_int(row.get("label"), "detector prediction label")
        if label not in {0, 1}:
            raise RevisionReportingError("Detector prediction label must be binary")
        prediction_labels.setdefault(key, {0: 0, 1: 0})[label] += 1
    if any(value[0] != value[1] or value[0] == 0 for value in prediction_labels.values()):
        raise RevisionReportingError("Detector prediction splits are not exactly balanced")
    for row in metrics:
        key = (row.get("split_id", ""), row.get("detector_name", ""))
        for column in (
            "roc_auc", "pr_auc", "balanced_accuracy", "f1", "sensitivity", "specificity",
            "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc_ci_low", "pr_auc_ci_high",
        ):
            if row.get(column, "").strip():
                value = _safe_float(row[column], "detector {}".format(column))
                if not 0.0 <= value <= 1.0:
                    raise RevisionReportingError("Detector metrics and intervals must lie in [0, 1]")
        if row.get("test_rows", "") and counts.get(key, 0) != _safe_int(row["test_rows"], "detector test_rows"):
            raise RevisionReportingError("Detector prediction count disagrees with metric row {}".format(key))
        for group_field in ("test_payload_groups", "n_payload_groups"):
            if row.get(group_field, "") and len(groups.get(key, set())) != _safe_int(
                row[group_field], "detector {}".format(group_field)
            ):
                raise RevisionReportingError("Detector payload-group count disagrees for {}".format(key))
    if schema == "rankcloak-revision-detector-run-v2" and not bool(manifest.get("smoke")):
        required = {
            "execution_mode": "confirmatory",
            "confirmatory_complete": True,
            "split_count": 28,
            "skipped_split_count": 0,
            "failure_count": 0,
            "smoke_fallback_metric_rows": 0,
            "metric_rows": 56,
        }
        for field, expected in required.items():
            if manifest.get(field) != expected:
                raise RevisionReportingError(
                    "Confirmatory detector v2 {} must equal {!r}".format(
                        field, expected
                    )
                )
        metadata = manifest.get("detector_run_metadata")
        split_contract = metadata.get("split_contract") if isinstance(metadata, dict) else None
        if not isinstance(split_contract, dict) or (
            split_contract.get("schema_version")
            != "rankcloak-revision-detector-splits-v2"
            or split_contract.get("input_scope")
            != "primary_full_detector_corpus_only"
            or split_contract.get("expected_split_count") != 28
            or split_contract.get("missing_split_ids") != []
            or split_contract.get("unexpected_split_ids") != []
        ):
            raise RevisionReportingError(
                "Confirmatory detector v2 lacks the complete primary 28-split contract"
            )
        expected_ids = split_contract.get("expected_split_ids")
        if not isinstance(expected_ids, list) or len(set(map(str, expected_ids))) != 28:
            raise RevisionReportingError("Detector split contract does not contain 28 unique IDs")
        expected_ids = list(map(str, expected_ids))
        if (
            expected_ids.count("matched") != 1
            or sum(value.startswith("held_out_template:") for value in expected_ids) != 18
            or sum(value.startswith("leave_one_model:") for value in expected_ids) != 3
            or sum(value.startswith("leave_one_codec:") for value in expected_ids) != 6
            or split_contract.get("expected_split_ids_sha256")
            != hashlib.sha256("\n".join(sorted(expected_ids)).encode("utf-8")).hexdigest()
        ):
            raise RevisionReportingError("Detector split identities violate the 1+18+3+6 contract")
        dataset_contract = split_contract.get("dataset_contract")
        if not isinstance(dataset_contract, dict) or any(
            dataset_contract.get(field) != expected
            for field, expected in {
                "schema_version": "rankcloak-revision-primary-detector-contract-v1",
                "input_scope": "primary_v2_complete_detector_corpus_only",
                "rows": 15840,
                "payload_groups": 480,
                "positive_rows": 7920,
                "negative_rows": 7920,
                "split_count": 28,
            }.items()
        ):
            raise RevisionReportingError("Detector v2 input is not the complete primary corpus")
        _validate_detector_preprocessing_binding(manifest, dataset_contract)
        dataset_group_counts: Dict[str, Dict[int, int]] = {}
        for row in dataset:
            group = str(row.get("payload_group_id", ""))
            label = _safe_int(row.get("label"), "detector dataset label")
            if not group or label not in {0, 1}:
                raise RevisionReportingError(
                    "Detector dataset contains an invalid group/label identity"
                )
            dataset_group_counts.setdefault(group, {0: 0, 1: 0})[label] += 1
        if len(dataset_group_counts) != 480 or any(
            counts[0] != counts[1] or counts[0] == 0
            for counts in dataset_group_counts.values()
        ):
            raise RevisionReportingError(
                "Detector dataset payload groups are not exactly class-balanced"
            )
        for column, contract_field, expected_count in (
            ("prompt_template_id", "prompt_template_ids", 18),
            ("model_id", "model_ids", 3),
            ("codec_id", "codec_ids", 6),
        ):
            expected_levels = set(map(str, dataset_contract.get(contract_field, [])))
            observed_levels = {str(row.get(column, "")) for row in dataset}
            if (
                len(expected_levels) != expected_count
                or observed_levels != expected_levels
            ):
                raise RevisionReportingError(
                    "Detector dataset {} levels differ from its primary contract".format(
                        column
                    )
                )
        source_configs = dataset_contract.get("source_configs")
        expected_config_paths = {
            "primary.json": PROJECT_ROOT / "configs" / "revision_v1" / "primary.json",
            "prompts.json": PROJECT_ROOT / "configs" / "revision_v1" / "prompts.json",
            "models.json": PROJECT_ROOT / "configs" / "revision_v1" / "models.json",
        }
        if not isinstance(source_configs, dict) or any(
            source_configs.get(name) != file_sha256(path)
            for name, path in expected_config_paths.items()
        ):
            raise RevisionReportingError("Detector primary source-config hashes differ")
        observed_ids = {str(row.get("split_id", "")) for row in metrics}
        if observed_ids != set(expected_ids):
            raise RevisionReportingError("Detector metric rows do not realize all 28 split IDs")
        if (
            metadata.get("expected_detector_split_executions") != 56
            or metadata.get("complete_detector_split_executions") != 56
            or metadata.get("confirmatory_complete") is not True
        ):
            raise RevisionReportingError("Detector run metadata does not complete all 56 executions")
        identities = manifest.get("detector_execution_identities")
        if not isinstance(identities, list) or len(identities) != 56:
            raise RevisionReportingError("Detector v2 lacks 56 execution identities")
        identity_pairs = set()
        required_detectors = {
            ("published_textcnn_equivalent", "text_cnn"),
            ("deberta_v3_base_classifier", "pretrained_transformer"),
        }
        for identity in identities:
            if not isinstance(identity, dict) or (
                identity.get("implementation_status") != "complete"
                or identity.get("implementation_kind") != identity.get("requested_kind")
                or (identity.get("detector_name"), identity.get("requested_kind"))
                not in required_detectors
                or identity.get("split_id") not in expected_ids
            ):
                raise RevisionReportingError("Detector v2 contains an invalid execution identity")
            identity_pairs.add((identity.get("split_id"), identity.get("detector_name")))
        if len(identity_pairs) != 56:
            raise RevisionReportingError("Detector v2 execution identities are not one per split/model")
        split_value = sources.json_values.get("detector.splits")
        if not isinstance(split_value, dict) or split_value.get(
            "schema_version"
        ) != "rankcloak-revision-detector-splits-v2":
            raise RevisionReportingError("Detector split artifact is not schema v2")
        if split_value.get("skipped_splits") != []:
            raise RevisionReportingError("Detector split artifact records skipped splits")
        split_rows = split_value.get("splits")
        if not isinstance(split_rows, list) or len(split_rows) != 28 or {
            str(row.get("split_id", "")) for row in split_rows if isinstance(row, dict)
        } != set(expected_ids):
            raise RevisionReportingError("Detector split artifact is missing prespecified IDs")
        for row in split_rows:
            if not isinstance(row, dict):
                raise RevisionReportingError("Detector split row is malformed")
            train_rows = _safe_int(row.get("train_rows"), "detector split train rows", False)
            test_rows = _safe_int(row.get("test_rows"), "detector split test rows", False)
            train_positive = _safe_int(
                row.get("train_positive_rows"), "detector split train positives", False
            )
            test_positive = _safe_int(
                row.get("test_positive_rows"), "detector split test positives", False
            )
            if train_rows != 2 * train_positive or test_rows != 2 * test_positive:
                raise RevisionReportingError(
                    "Detector split manifest is not exactly class-balanced"
                )


MIXED_MODEL_IDS = {
    "primary_exact_recovery",
    "primary_artifact_counts",
    "primary_effective_artifact_rate",
    "primary_cover_log_probability",
    "primary_heldout_evaluator_log_probability",
    "primary_payload_throughput",
    "human_naturalness_and_suspiciousness_clmm",
}


def _validate_mixed_model_consistency(sources: VerifiedSources) -> None:
    manifest = sources.manifests.get("mixed_model")
    if not isinstance(manifest, dict):
        return
    statuses = sources.json_values.get("mixed_model.status")
    if not isinstance(statuses, list) or not all(isinstance(row, dict) for row in statuses):
        raise RevisionReportingError("Mixed-model status artifact must be a JSON list")
    status_ids = [str(row.get("model_id", "")) for row in statuses]
    if set(status_ids) != MIXED_MODEL_IDS or len(status_ids) != len(set(status_ids)):
        raise RevisionReportingError("Mixed-model status artifact does not cover the frozen model set")
    by_id = {str(row["model_id"]): row for row in statuses}
    coefficients, _ = _exact_artifact_rows(sources, "mixed_model.coefficients")
    contrasts, _ = _exact_artifact_rows(sources, "mixed_model.contrasts")
    coefficients_by_model: Dict[str, int] = {}
    for row in coefficients:
        model_id = str(row.get("model_id", ""))
        coefficients_by_model[model_id] = coefficients_by_model.get(model_id, 0) + 1
    for model_id, row in by_id.items():
        if row.get("fixed_effects_fallback") is not False:
            raise RevisionReportingError(
                "Mixed model {} used or permits fixed-effects fallback".format(model_id)
            )
        status = str(row.get("status", ""))
        if model_id == "human_naturalness_and_suspiciousness_clmm":
            if status != "external_until_irb_approved_ratings_exist":
                raise RevisionReportingError("Human model status differs from the frozen plan")
        elif model_id == "primary_exact_recovery":
            if status not in {
                "completed",
                "completed_with_diagnostic_warning",
                "not_fitted_complete_outcome_separation_all_success",
                "not_fitted_complete_outcome_separation_all_failure",
            }:
                raise RevisionReportingError("Unexpected exact-recovery mixed-model status")
        elif model_id == "primary_artifact_counts":
            if status not in {
                "completed",
                "completed_with_diagnostic_warning",
                "not_fitted_all_zero_counts",
            }:
                raise RevisionReportingError("Unexpected artifact mixed-model status")
        elif status not in {"completed", "completed_with_diagnostic_warning"}:
            raise RevisionReportingError(
                "Required mixed model {} did not run to completion".format(model_id)
            )
        if status in {"completed", "completed_with_diagnostic_warning"}:
            declared_coefficients = _safe_int(
                row.get("coefficient_rows"),
                "mixed-model {} coefficient_rows".format(model_id),
                allow_zero=False,
            )
            if coefficients_by_model.get(model_id, 0) != declared_coefficients:
                raise RevisionReportingError(
                    "Mixed model {} coefficient rows disagree with status".format(
                        model_id
                    )
                )
    for suffix in ("coefficients", "contrasts"):
        rows, _ = _exact_artifact_rows(sources, "mixed_model." + suffix)
        for index, row in enumerate(rows, start=1):
            model_id = str(row.get("model_id", ""))
            if model_id not in MIXED_MODEL_IDS - {
                "human_naturalness_and_suspiciousness_clmm"
            }:
                raise RevisionReportingError(
                    "Mixed-model {} row {} has an unknown model ID".format(suffix, index)
                )
            if "fixed_effects_fallback" not in row or _safe_bool(
                row.get("fixed_effects_fallback"),
                "mixed-model {} row {} fallback".format(suffix, index),
            ):
                raise RevisionReportingError("Mixed-model row records fixed-effects fallback")
            if suffix == "coefficients" and row.get("backend") != "R_lme4":
                raise RevisionReportingError("Primary coefficients are not from R lme4")
            for field in ("estimate", "standard_error", "p_value_raw"):
                if str(row.get(field, "")).strip():
                    _safe_float(row[field], "mixed-model {} {}".format(suffix, field))
            if suffix == "contrasts":
                if row.get("adjustment") != "holm":
                    raise RevisionReportingError("Primary R contrasts must use Holm adjustment")
                adjusted = _safe_float(
                    row.get("p_value_holm"), "mixed-model contrast p_value_holm"
                )
                if not 0.0 <= adjusted <= 1.0:
                    raise RevisionReportingError("Mixed-model adjusted p-value lies outside [0, 1]")
                estimate = _safe_float(row.get("estimate"), "mixed-model contrast estimate")
                low = _safe_float(row.get("ci_low"), "mixed-model contrast ci_low")
                high = _safe_float(row.get("ci_high"), "mixed-model contrast ci_high")
                if not low <= estimate <= high:
                    raise RevisionReportingError("Mixed-model contrast interval is inconsistent")

    required_contrast_families = {
        "primary_exact_recovery": {
            "recovery_protocol_within_model",
            "recovery_model_within_protocol",
            "recovery_prompt_category",
        },
        "primary_artifact_counts": {"artifact_protocol_within_model"},
        "primary_effective_artifact_rate": {"continuous_protocol_within_model"},
        "primary_cover_log_probability": {"continuous_protocol_within_model"},
        "primary_heldout_evaluator_log_probability": {
            "continuous_protocol_within_model"
        },
        "primary_payload_throughput": {"continuous_protocol_within_model"},
    }
    observed_families: Dict[str, set] = {}
    for row in contrasts:
        observed_families.setdefault(str(row.get("model_id", "")), set()).add(
            str(row.get("multiplicity_family", ""))
        )
    for model_id, families in required_contrast_families.items():
        if by_id[model_id].get("status") in {
            "completed",
            "completed_with_diagnostic_warning",
        } and not families.issubset(observed_families.get(model_id, set())):
            raise RevisionReportingError(
                "Mixed model {} lacks prespecified contrast families".format(model_id)
            )

    def assert_no_fallback(value: Any, label: str) -> None:
        if isinstance(value, dict):
            if "fixed_effects_fallback" in value and value[
                "fixed_effects_fallback"
            ] is not False:
                raise RevisionReportingError("{} records fixed-effects fallback".format(label))
            for key, nested in value.items():
                assert_no_fallback(nested, "{}.{}".format(label, key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                assert_no_fallback(nested, "{}[{}]".format(label, index))

    assert_no_fallback(
        sources.json_values.get("mixed_model.diagnostics"),
        "Mixed-model diagnostics",
    )


def _raw_jsonl_objects(path: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RevisionReportingError(
                        "{} row {} is not an object".format(label, line_number)
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionReportingError("Cannot read {}: {}".format(label, exc))
    return rows


def _validate_evaluator_unavailability_consistency(
    sources: VerifiedSources,
) -> None:
    manifest = sources.manifests.get("evaluator_unavailability")
    if not isinstance(manifest, dict):
        return
    required_artifacts = {
        role: sources.artifacts.get("evaluator_unavailability." + role)
        for role in ("plan", "checkpoint", "records", "run_identity")
    }
    if any(value is None for value in required_artifacts.values()):
        raise RevisionReportingError(
            "Evaluator-unavailability lineage artifacts are incomplete"
        )
    plan_rows = _raw_jsonl_objects(required_artifacts["plan"].path, "unavailability plan")
    record_rows = _raw_jsonl_objects(
        required_artifacts["records"].path, "unavailability records"
    )
    plan_by_id: Dict[str, Dict[str, Any]] = {}
    for row in plan_rows:
        identifier = str(row.get("work_id", ""))
        if not identifier or identifier in plan_by_id:
            raise RevisionReportingError(
                "Evaluator-unavailability plan work IDs are empty or duplicated"
            )
        plan_by_id[identifier] = row
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for row in record_rows:
        identifier = str(row.get("work_id", ""))
        if not identifier or identifier in records_by_id:
            raise RevisionReportingError(
                "Evaluator-unavailability record work IDs are empty or duplicated"
            )
        records_by_id[identifier] = row
    checkpoint = sources.json_values.get("evaluator_unavailability.checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("completed_trial_ids"), list
    ):
        raise RevisionReportingError(
            "Evaluator-unavailability checkpoint is malformed"
        )
    completed_ids = set(map(str, checkpoint["completed_trial_ids"]))
    units = manifest["units"]
    unit_ids = {str(row["source_work_id"]) for row in units}
    for unit in units:
        identifier = str(unit["source_work_id"])
        record = records_by_id.get(identifier)
        task = plan_by_id.get(identifier)
        if record is None or task is None or identifier not in completed_ids:
            raise RevisionReportingError(
                "Evaluator-unavailability unit is absent from source lineage"
            )
        if compact_json_sha256(record) != unit.get("source_record_sha256"):
            raise RevisionReportingError(
                "Evaluator-unavailability source record hash differs"
            )
        if (
            record.get("execution_status") != "completed"
            or record.get("record_type") != "condition_unavailable"
            or record.get("reason_code") != unit.get("reason_code")
            or record.get("protocol_contract_revision")
            != PROTOCOL_CONTRACT_REVISION
            or record.get("result_schema_revision") != RESULT_SCHEMA_REVISION
            or task.get("work_kind") != "rankcloak"
            or task.get("protocol_variant") != unit.get("protocol_variant")
            or task.get("payload_name") != unit.get("payload_name")
        ):
            raise RevisionReportingError(
                "Evaluator-unavailability unit metadata differs from source lineage"
            )
    eligible_ids = {
        identifier
        for identifier, record in records_by_id.items()
        if identifier in completed_ids
        and record.get("execution_status") == "completed"
        and record.get("record_type") == "condition_unavailable"
        and isinstance(plan_by_id.get(identifier), dict)
        and plan_by_id[identifier].get("work_kind") == "rankcloak"
    }
    if eligible_ids != unit_ids:
        raise RevisionReportingError(
            "Evaluator-unavailability units do not exhaust terminal source records"
        )


def _validate_generic_runtime(sources: VerifiedSources) -> None:
    for index, row in enumerate(_runtime_summary_rows(sources), start=1):
        if row.get("n_payloads", "").strip():
            _safe_int(row["n_payloads"], "generic runtime row {} n_payloads".format(index), False)
        if row.get("mean", "").strip():
            mean = _safe_float(row["mean"], "generic runtime row {} mean".format(index))
            if row.get("ci_low", "").strip() and row.get("ci_high", "").strip():
                low = _safe_float(row["ci_low"], "generic runtime row {} CI low".format(index))
                high = _safe_float(row["ci_high"], "generic runtime row {} CI high".format(index))
                if not low <= mean <= high:
                    raise RevisionReportingError("Generic runtime confidence interval does not contain its mean")


def display_registry() -> Dict[str, Any]:
    supplementary: List[Dict[str, str]] = []
    for index, title in enumerate(SUPPLEMENTARY_TABLE_TITLES, start=1):
        supplementary.append(
            {
                "id": "supplementary_table_s{}".format(index),
                "type": "table",
                "number": "S{}".format(index),
                "label": "tab:s{}".format(index),
                "title": title,
            }
        )
    for index, title in enumerate(SUPPLEMENTARY_FIGURE_TITLES, start=1):
        supplementary.append(
            {
                "id": "supplementary_figure_s{}".format(index),
                "type": "figure",
                "number": "S{}".format(index),
                "label": "fig:s{}".format(index),
                "title": title,
            }
        )
    registry = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "main_display_limit": MAX_MAIN_DISPLAY_ITEMS,
        "main_display_count": len(MAIN_DISPLAYS),
        "main": [dict(value) for value in MAIN_DISPLAYS],
        "supplementary": supplementary,
    }
    validate_display_registry(registry)
    return registry


def validate_display_registry(registry: Mapping[str, Any]) -> None:
    main = registry.get("main")
    supplementary = registry.get("supplementary")
    if not isinstance(main, list) or not isinstance(supplementary, list):
        raise RevisionReportingError("Display registry lists are malformed")
    if len(main) > MAX_MAIN_DISPLAY_ITEMS:
        raise RevisionReportingError("Main figure-plus-table count exceeds seven")
    if registry.get("main_display_count") != len(main):
        raise RevisionReportingError("Main display count does not match registry")
    all_items = main + supplementary
    ids = [str(value.get("id")) for value in all_items]
    labels = [str(value.get("label")) for value in all_items]
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise RevisionReportingError("Display IDs and LaTeX labels must be unique")
    for group in (main, supplementary):
        by_type: Dict[str, List[str]] = {}
        for item in group:
            by_type.setdefault(str(item.get("type")), []).append(str(item.get("number")))
        for item_type, numbers in by_type.items():
            if len(numbers) != len(set(numbers)):
                raise RevisionReportingError("Duplicate {} number in display registry".format(item_type))


def _source_hash_for(sources: VerifiedSources, suffix: str) -> str:
    matches = [artifact.sha256 for key, artifact in sources.artifacts.items() if key.endswith("." + suffix)]
    return ";".join(sorted(matches))


def _exact_source_hash(sources: VerifiedSources, key: str) -> str:
    artifact = sources.artifacts.get(key)
    return artifact.sha256 if artifact is not None else ""


def _with_lineage(
    rows: Iterable[Mapping[str, Any]], source_key: str, source_hash: str, status: str = "available"
) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        value = dict(row)
        value["report_status"] = status
        value["source_artifact"] = value.pop("_source_key", source_key)
        value["source_sha256"] = value.pop("_source_sha256", source_hash)
        result.append(value)
    return result


def _unavailable_rows(reason: str, source_key: str = "none") -> List[Dict[str, str]]:
    return [
        {
            "report_status": UNAVAILABLE,
            "reason": reason,
            "source_artifact": source_key,
            "source_sha256": "",
        }
    ]


_EXPLORATORY_RE = re.compile(r"smoke|pilot|exploratory|limited", re.IGNORECASE)


def _eligible_rows(rows: Sequence[Mapping[str, str]], fixture_mode: bool) -> List[Dict[str, str]]:
    if fixture_mode:
        return [dict(row) for row in rows]
    result = []
    for row in rows:
        labels = " ".join(
            row.get(column, "")
            for column in ("phase", "study_phase", "evidence_status")
        )
        if labels and _EXPLORATORY_RE.search(labels):
            continue
        result.append(dict(row))
    return result


def _primary_recovery_rows(
    rows: Sequence[Mapping[str, str]], fixture_mode: bool
) -> List[Dict[str, str]]:
    eligible = _eligible_rows(rows, fixture_mode)
    result = []
    for row in eligible:
        if row.get("evidence_status") != PRIMARY_EVIDENCE_STATUS:
            continue
        if row.get("study_phase") != PRIMARY_STUDY_PHASE:
            continue
        if row.get("protocol_contract_revision") != PROTOCOL_CONTRACT_REVISION:
            continue
        if row.get("result_schema_revision") != RESULT_SCHEMA_REVISION:
            continue
        replay = row.get("replay_mode", "")
        transformation = row.get("transformation_id", "")
        mitigation = row.get("mitigation_id", "")
        if replay not in ("", "saved_token_ids"):
            continue
        if transformation not in ("", "unmodified"):
            continue
        if mitigation not in ("", "none"):
            continue
        result.append(dict(row))
    return result


def _multilingual_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    result = []
    for row in rows:
        language = row.get("language", "").strip().lower().replace("_", "-")
        labels = " ".join(
            row.get(column, "").lower() for column in ("phase", "study_phase")
        )
        if language in {"es", "es-es", "zh", "zh-cn", "cmn"} or any(
            token in labels for token in ("multilingual", "spanish", "mandarin")
        ):
            result.append(dict(row))
    return result


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        raise RevisionReportingError("Cannot calculate an interval with n=0")
    proportion = successes / float(n)
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _main_table_1_rows(sources: VerifiedSources) -> List[Dict[str, Any]]:
    raw, _ = _artifact_rows(sources, "recovery")
    rows = _primary_recovery_rows(raw, sources.fixture_mode)
    if not rows:
        return _unavailable_rows("Primary recovery statistics have not been supplied.")
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        model = row.get("model_id", UNAVAILABLE) or UNAVAILABLE
        protocol = row.get("protocol_variant", UNAVAILABLE) or UNAVAILABLE
        key = (model, protocol)
        group = groups.setdefault(
            key,
            {
                "model_id": model,
                "protocol_variant": protocol,
                "payload_recovery_successes": 0,
                "n_payloads": 0,
                "payload_classes": set(),
            },
        )
        group["payload_recovery_successes"] += _safe_int(
            row["payload_recovery_successes"], "payload recovery successes"
        )
        group["n_payloads"] += _safe_int(row["n_payloads"], "recovery n", False)
        if row.get("payload_class"):
            group["payload_classes"].add(row["payload_class"])
    output = []
    for key in sorted(groups):
        group = groups[key]
        low, high = _wilson(
            group["payload_recovery_successes"], group["n_payloads"]
        )
        output.append(
            {
                "model_id": group["model_id"],
                "protocol_variant": group["protocol_variant"],
                "payload_class_count": len(group["payload_classes"]),
                "n_payloads": group["n_payloads"],
                "payload_recovery_successes": group["payload_recovery_successes"],
                "exact_payload_recovery_rate": (
                    group["payload_recovery_successes"]
                    / float(group["n_payloads"])
                ),
                "recovery_outcome": PAYLOAD_RECOVERY_OUTCOME,
                "recovery_outcome_semantics": PAYLOAD_RECOVERY_SEMANTICS,
                "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
                "result_schema_revision": RESULT_SCHEMA_REVISION,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return _with_lineage(output, "statistics.recovery", _source_hash_for(sources, "recovery"))


def _main_table_2_rows(sources: VerifiedSources) -> List[Dict[str, Any]]:
    contrasts, _ = _exact_artifact_rows(sources, "mixed_model.contrasts")
    continuous, _ = _artifact_rows(sources, "continuous")
    outcome_by_model = {
        "primary_exact_recovery": PAYLOAD_RECOVERY_OUTCOME,
        "primary_artifact_counts": "artifact_count",
        "primary_effective_artifact_rate": "effective_artifact_bits_per_full_token",
        "primary_cover_log_probability": "mean_log_probability",
        "primary_heldout_evaluator_log_probability": "heldout_evaluator_log_probability",
        "primary_payload_throughput": "payload_bits_per_second",
    }
    eligible_effects = [
        row for row in contrasts if row.get("model_id") in outcome_by_model
    ]
    chosen_effects = []
    for model_id in outcome_by_model:
        candidates = sorted(
            (row for row in eligible_effects if row.get("model_id") == model_id),
            key=lambda row: (
                row.get("multiplicity_family", ""),
                row.get("contrast", ""),
            ),
        )
        if candidates:
            chosen_effects.append(candidates[0])
    output: List[Dict[str, Any]] = []
    for row in chosen_effects[:8]:
        by_values = [
            "{}={}".format(field, row[field])
            for field in ("model_id", "protocol_variant", "prompt_category")
            if str(row.get(field, "")).strip() and field != "model_id"
        ]
        output.append(
            {
                "section": "effect",
                "condition": "{}: {}{}".format(
                    row.get("multiplicity_family", ""),
                    row.get("contrast", ""),
                    " (" + ", ".join(by_values) + ")" if by_values else "",
                ),
                "outcome": outcome_by_model[str(row.get("model_id"))],
                "estimate": row.get("estimate", ""),
                "ci_low": row.get("ci_low", ""),
                "ci_high": row.get("ci_high", ""),
                "n_payloads": "payload-trial mixed model",
                "adjusted_p": row.get("p_value_holm", ""),
                "source_artifact": "mixed_model.contrasts",
                "source_sha256": _exact_source_hash(
                    sources, "mixed_model.contrasts"
                ),
                "report_status": "available",
            }
        )
    if not output:
        output.extend(
            _unavailable_rows(
                "Verified prespecified R mixed-model contrasts are unavailable; "
                "Python pooled/pairwise effects are not a confirmatory substitute.",
                "mixed_model.contrasts",
            )
        )
    runtime_outcomes = {
        "encoding_seconds": 0,
        "decoding_seconds": 1,
        "encoding_tokens_per_second": 2,
        "decoding_tokens_per_second": 3,
        "payload_bits_per_second": 4,
        "peak_gpu_memory_mib": 5,
        "peak_ram_mib": 6,
        "cover_tokens_per_payload_byte": 7,
    }
    runtime_rows = [
        row
        for row in _eligible_rows(continuous, sources.fixture_mode)
        + _eligible_rows(_runtime_summary_rows(sources), sources.fixture_mode)
        if row.get("outcome") in runtime_outcomes
    ]
    protocol_order = ("nonseg_ascii_b16", "nonseg_hex_nibble_b16", "direct_subword_calgacus")
    models = sorted({row.get("model_id", "") for row in runtime_rows})
    for model in models:
        candidates = [row for row in runtime_rows if row.get("model_id", "") == model]
        representative = []
        for protocol in protocol_order:
            representative = [row for row in candidates if row.get("protocol_variant", "") == protocol]
            if representative:
                break
        if not representative:
            representative = candidates
        representative.sort(key=lambda row: runtime_outcomes[row.get("outcome", "")])
        for row in representative[:4]:
            output.append(
                {
                    "section": "runtime",
                    "condition": "{} / {}".format(model, row.get("protocol_variant", "")),
                    "outcome": row.get("outcome", ""),
                    "estimate": row.get("mean", ""),
                    "ci_low": row.get("ci_low", ""),
                    "ci_high": row.get("ci_high", ""),
                    "n_payloads": row.get("n_payloads", ""),
                    "adjusted_p": "",
                    "source_artifact": row.get("_source_key", "statistics.continuous"),
                    "source_sha256": row.get("_source_sha256", _source_hash_for(sources, "continuous")),
                    "report_status": "available",
                }
            )
    return output or _unavailable_rows("Primary effect and runtime outputs have not been supplied.")


def _rows_matching(
    rows: Sequence[Mapping[str, str]],
    *,
    columns: Sequence[str] = (),
    patterns: Sequence[str] = (),
) -> List[Dict[str, str]]:
    result = []
    lowered = [pattern.lower() for pattern in patterns]
    for row in rows:
        text = " ".join(row.get(column, "") for column in columns).lower()
        if not lowered or any(pattern in text for pattern in lowered):
            result.append(dict(row))
    return result


def _unavailable_summary_rows(sources: VerifiedSources) -> List[Dict[str, Any]]:
    raw, artifact = _artifact_rows(sources, "unavailable")
    raw = _eligible_rows(raw, sources.fixture_mode)
    if not raw:
        return []
    if artifact is not None:
        source_key = "{}.{}".format(artifact.source_kind, artifact.logical_name)
        for row in raw:
            row["_source_key"] = source_key
            row["_source_sha256"] = artifact.sha256
    grouping = (
        "record_type",
        "model_id",
        "token_filter",
        "ablation_factor",
        "ablation_level",
        "robustness_family",
        "reason_code",
        "root_condition_model_id",
        "_source_key",
        "_source_sha256",
    )
    counts: Dict[Tuple[str, ...], int] = {}
    for row in raw:
        key = tuple(row.get(column, "") for column in grouping)
        counts[key] = counts.get(key, 0) + 1
    output: List[Dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        value = dict(zip(grouping, key))
        value.update(
            {
                "taxonomy_class": "condition_unavailable_not_recovery_failure",
                "unavailable_work_units": count,
                "excluded_from_estimands": True,
            }
        )
        output.append(value)
    return output


def _evaluator_unavailability_summary_rows(
    sources: VerifiedSources,
) -> List[Dict[str, Any]]:
    manifest = sources.manifests.get("evaluator_unavailability")
    manifest_path = sources.manifest_paths.get("evaluator_unavailability")
    if not isinstance(manifest, dict) or manifest_path is None:
        return []
    return [
        {
            "record_type": "heldout_evaluator_terminal_non_outcome",
            "source_stage": "ablation_v2",
            "model_id": "mistral_7b_instruct_v0_3_q4_k_m",
            "evaluator_model_id": "llama3_8b_instruct_q4_k_m",
            "reason_code": "empty_isolated_roundtrip_vocabulary",
            "terminal_status": "upstream_dependent_unavailable_not_scored",
            "taxonomy_class": (
                "heldout_evaluator_upstream_dependent_unavailable_not_scored"
            ),
            "unavailable_work_units": manifest[
                "upstream_dependent_unavailable_units"
            ],
            "scoreable_evaluator_units": manifest["scoreable_evaluator_units"],
            "terminal_accounted_units": manifest["terminal_accounted_units"],
            "frozen_evaluator_target_units": manifest[
                "frozen_evaluator_target_units"
            ],
            "scoring_attempted": False,
            "score_imputed": False,
            "excluded_from_estimands": True,
            "_source_key": "evaluator_unavailability.manifest",
            "_source_sha256": file_sha256(manifest_path),
        }
    ]


def supplementary_table_rows(sources: VerifiedSources, index: int) -> List[Dict[str, Any]]:
    recovery, _ = _artifact_rows(sources, "recovery")
    continuous, _ = _artifact_rows(sources, "continuous")
    mixed_coefficients, _ = _exact_artifact_rows(
        sources, "mixed_model.coefficients"
    )
    mixed_contrasts, _ = _exact_artifact_rows(
        sources, "mixed_model.contrasts"
    )
    detector, _ = _artifact_rows(sources, "metrics")
    runtime = _runtime_summary_rows(sources)
    unavailable = _unavailable_summary_rows(sources)
    evaluator_unavailable = _evaluator_unavailability_summary_rows(sources)
    recovery = _eligible_rows(recovery, sources.fixture_mode)
    continuous = _eligible_rows(continuous, sources.fixture_mode)
    if index == 1:
        protocols = sorted({row.get("protocol_variant", "") for row in recovery if row.get("protocol_variant")})
        rows = [{"protocol_variant": value, "definition": UNAVAILABLE} for value in protocols]
        suffix, source = "recovery", "statistics.recovery"
    elif index == 2:
        pairs = sorted({(row.get("model_id", ""), row.get("hardware_id", "")) for row in continuous if row.get("model_id")})
        rows = [{"model_id": model, "hardware_id": hardware or UNAVAILABLE, "revision_and_hash": UNAVAILABLE} for model, hardware in pairs]
        suffix, source = "continuous", "statistics.continuous"
    elif index == 3:
        grouped: Dict[str, int] = {}
        for row in recovery:
            payload_class = row.get("payload_class", "")
            if payload_class:
                grouped[payload_class] = max(grouped.get(payload_class, 0), _safe_int(row["n_payloads"], "payload class n"))
        rows = [{"payload_class": key, "maximum_independent_n_per_stratum": value} for key, value in sorted(grouped.items())]
        suffix, source = "recovery", "statistics.recovery"
    elif index == 4:
        rows, suffix, source = recovery, "recovery", "statistics.recovery"
    elif index == 5:
        rows = [dict(row, result_type="coefficient") for row in mixed_coefficients]
        rows.extend(dict(row, result_type="prespecified_contrast") for row in mixed_contrasts)
        suffix, source = "coefficients", "mixed_model.coefficients/contrasts"
    elif index == 6:
        rows = _multilingual_rows(recovery + continuous)
        suffix, source = "recovery", "statistics.recovery/continuous"
    elif index == 7:
        rows = _rows_matching(recovery + continuous, columns=("phase", "study_phase", "ablation_factor", "ablation_level", "token_filter", "tail_policy"), patterns=("ablation", "filter", "tail", "roundtrip", "safe", "fixed", "dynamic"))
        rows.extend(
            _rows_matching(
                unavailable,
                columns=("token_filter", "ablation_factor", "reason_code"),
                patterns=("filter", "roundtrip", "empty_isolated"),
            )
        )
        suffix, source = "continuous", "statistics.recovery/continuous"
    elif index == 8:
        rows = _rows_matching(recovery + continuous, columns=("replay_mode", "leadin_tokens", "transformation_id"), patterns=("replay", "retoken", "leadin", "greedy"))
        suffix, source = "recovery", "statistics.recovery/continuous"
    elif index == 9:
        rows = _rows_matching(recovery, columns=("transformation_id", "mitigation_id"), patterns=("trim", "whitespace", "unicode", "quote", "markdown", "insert", "delet", "substitut", "truncate", "paraphrase", "canonical"))
        suffix, source = "recovery", "statistics.recovery"
    elif index == 10:
        rows = _rows_matching(continuous, columns=("outcome",), patterns=("human", "naturalness", "suspiciousness", "fluency", "coherence", "grammar", "reliability"))
        suffix, source = "continuous", "statistics.continuous"
    elif index == 11:
        rows, suffix, source = detector, "metrics", "detector.metrics"
        if not rows:
            rows, _ = _artifact_rows(sources, "detectors")
            suffix, source = "detectors", "statistics.detectors"
    elif index == 12:
        rows = [row for row in continuous + runtime if row.get("outcome") in {
            "encoding_seconds", "decoding_seconds", "encoding_tokens_per_second", "decoding_tokens_per_second", "payload_bits_per_second", "peak_ram_mib", "peak_gpu_memory_mib", "cover_tokens_per_payload_byte"
        }]
        suffix, source = "continuous", "statistics.continuous"
    elif index == 13:
        rows = _artifact_rows(sources, "failures")[0]
        rows.extend(unavailable)
        rows.extend(evaluator_unavailable)
        suffix, source = (
            "failures",
            "runtime.failures/preprocess.unavailable/evaluator_unavailability.manifest",
        )
    else:
        raise RevisionReportingError("Supplementary table index must be 1--13")
    if not rows:
        return _unavailable_rows("Required machine output for Supplementary Table S{} is unavailable.".format(index), source)
    source_hash = _source_hash_for(sources, suffix)
    if index == 5:
        source_hash = ";".join(
            filter(
                None,
                (
                    _exact_source_hash(sources, "mixed_model.coefficients"),
                    _exact_source_hash(sources, "mixed_model.contrasts"),
                ),
            )
        )
    return _with_lineage(rows, source, source_hash)


def _detector_metric_rows(sources: VerifiedSources) -> List[Dict[str, str]]:
    rows, _ = _artifact_rows(sources, "metrics")
    if rows:
        return rows
    rows, _ = _artifact_rows(sources, "detectors")
    return rows


def plot_source_rows(sources: VerifiedSources, plot_id: str) -> List[Dict[str, Any]]:
    recovery, _ = _artifact_rows(sources, "recovery")
    continuous, _ = _artifact_rows(sources, "continuous")
    capacity, _ = _artifact_rows(sources, "capacity_plot")
    quality, _ = _artifact_rows(sources, "quality_plot")
    detector = _detector_metric_rows(sources)
    runtime = _runtime_summary_rows(sources)
    recovery = _eligible_rows(recovery, sources.fixture_mode)
    continuous = _eligible_rows(continuous, sources.fixture_mode)
    rows: List[Dict[str, Any]] = []
    sources_used: List[Tuple[str, str]] = []
    if plot_id == "main_figure_1":
        rows = [dict(row, panel="capacity") for row in capacity] + [dict(row, panel="quality") for row in quality]
        sources_used = [("theory.capacity_plot", "capacity_plot"), ("theory.quality_plot", "quality_plot")]
    elif plot_id == "main_figure_2":
        rates = [
            row
            for row in _primary_recovery_rows(continuous, sources.fixture_mode)
            if row.get("outcome") == "effective_payload_rate"
        ]
        primary_recovery = _primary_recovery_rows(recovery, sources.fixture_mode)
        rows = [dict(row, panel="recovery") for row in primary_recovery] + [dict(row, panel="effective_rate") for row in rates]
        sources_used = [("statistics.recovery", "recovery"), ("statistics.continuous", "continuous")]
    elif plot_id == "main_figure_3":
        rows = _rows_matching(continuous, columns=("outcome", "text_view"), patterns=("human", "naturalness", "suspiciousness"))
        sources_used = [("statistics.continuous", "continuous")]
    elif plot_id == "main_figure_4":
        rows = _rows_matching(recovery + continuous, columns=("replay_mode", "transformation_id", "mitigation_id", "leadin_tokens"), patterns=("retoken", "greedy", "canonical", "leadin", "insert", "delet", "paraphrase", "truncate"))
        sources_used = [("statistics.recovery/continuous", "recovery")]
    elif plot_id == "main_figure_5":
        rows = detector
        sources_used = [("detector.metrics", "metrics")]
    elif plot_id.startswith("supplementary_figure_s"):
        index = int(plot_id.rsplit("s", 1)[1])
        if index == 1:
            rows = recovery
            sources_used = [("statistics.recovery", "recovery")]
        elif index == 2:
            capacity_validation, _ = _artifact_rows(sources, "capacity_validation")
            rows = _rows_matching(capacity_validation, columns=("representation_name", "protocol_variant"), patterns=("direct", "subword"))
            sources_used = [("theory.capacity_validation", "capacity_validation")]
        elif index == 3:
            rows = capacity + quality
            sources_used = [("theory.capacity/quality", "capacity_plot")]
        elif index == 4:
            rows = _rows_matching(recovery + continuous, columns=("token_filter", "protocol_variant", "mitigation_id"), patterns=("filter", "roundtrip", "safe", "none"))
            sources_used = [("statistics.recovery/continuous", "recovery")]
        elif index == 5:
            rows = _rows_matching(recovery + continuous, columns=("leadin_tokens", "replay_mode"), patterns=("leadin", "retoken", "greedy"))
            sources_used = [("statistics.recovery/continuous", "recovery")]
        elif index == 6:
            rows = _rows_matching(recovery + continuous, columns=("tail_policy", "segment_size_ranks", "protocol_variant"), patterns=("tail", "segment", "fixed", "dynamic"))
            sources_used = [("statistics.recovery/continuous", "continuous")]
        elif index == 7:
            rows = _rows_matching(recovery, columns=("transformation_id",), patterns=("trim", "whitespace", "unicode", "quote", "markdown", "insert", "delet", "substitut", "truncate", "paraphrase"))
            sources_used = [("statistics.recovery", "recovery")]
        elif index == 8:
            rows = _rows_matching(continuous, columns=("outcome",), patterns=("human", "rating", "naturalness", "suspiciousness"))
            sources_used = [("statistics.continuous", "continuous")]
        elif index == 9:
            rows = _rows_matching(continuous, columns=("outcome",), patterns=("flesch", "readability", "grammar", "repetition", "heldout", "tfidf"))
            sources_used = [("statistics.continuous", "continuous")]
        elif index == 10:
            predictions, _ = _artifact_rows(sources, "predictions")
            rows = _roc_pr_plot_rows(predictions)
            sources_used = [("detector.predictions", "predictions")]
        elif index == 11:
            rows = _rows_matching(detector, columns=("regime", "split_id"), patterns=("held", "leave_one", "cross"))
            sources_used = [("detector.metrics", "metrics")]
        elif index == 12:
            rows = [row for row in continuous + runtime if row.get("outcome") in {"encoding_seconds", "decoding_seconds", "payload_bits_per_second", "encoding_tokens_per_second", "decoding_tokens_per_second", "peak_ram_mib", "peak_gpu_memory_mib", "cover_tokens_per_payload_byte"}]
            sources_used = [("statistics.continuous", "continuous")]
        elif index == 13:
            rows = _multilingual_rows(recovery + continuous)
            sources_used = [("statistics.recovery/continuous", "recovery")]
    else:
        raise RevisionReportingError("Unknown plot ID: {}".format(plot_id))
    if not rows:
        return _unavailable_rows("Machine evidence for {} is unavailable.".format(plot_id), ",".join(key for key, _ in sources_used) or "none")
    lineage_hash = ";".join(filter(None, (_source_hash_for(sources, suffix) for _, suffix in sources_used)))
    return _with_lineage(rows, ",".join(key for key, _ in sources_used), lineage_hash)


def _roc_pr_plot_rows(predictions: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for row in predictions:
        if not row.get("label", "") or not row.get("score", ""):
            continue
        label = _safe_int(row["label"], "detector label")
        if label not in (0, 1):
            raise RevisionReportingError("Detector labels must be binary")
        score = _safe_float(row["score"], "detector score")
        grouped.setdefault((row.get("split_id", ""), row.get("detector_name", "")), []).append((label, score))
    result: List[Dict[str, Any]] = []
    for (split_id, detector_name), values in sorted(grouped.items()):
        positives = sum(label for label, _ in values)
        negatives = len(values) - positives
        if not positives or not negatives:
            continue
        thresholds = [float("inf")] + sorted({score for _, score in values}, reverse=True) + [float("-inf")]
        for order, threshold in enumerate(thresholds):
            predicted = [score >= threshold for _, score in values]
            tp = sum(bool(value) and label == 1 for value, (label, _) in zip(predicted, values))
            fp = sum(bool(value) and label == 0 for value, (label, _) in zip(predicted, values))
            tpr = tp / float(positives)
            fpr = fp / float(negatives)
            precision = tp / float(tp + fp) if tp + fp else 1.0
            result.append(
                {
                    "split_id": split_id,
                    "detector_name": detector_name,
                    "threshold_order": order,
                    "threshold": "inf" if threshold == float("inf") else "-inf" if threshold == float("-inf") else threshold,
                    "false_positive_rate": fpr,
                    "true_positive_rate": tpr,
                    "recall": tpr,
                    "precision": precision,
                }
            )
    return result


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        rows = _unavailable_rows("No rows were supplied.")
    preferred = ("report_status", "reason", "source_artifact", "source_sha256")
    keys = {str(key) for row in rows for key in row.keys()}
    columns = [column for column in preferred if column in keys] + sorted(keys - set(preferred))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        normalized = {}
        for column in columns:
            value = row.get(column, "")
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value if not isinstance(value, set) else sorted(value), sort_keys=True, ensure_ascii=False)
            normalized[column] = value
        writer.writerow(normalized)
    return buffer.getvalue().encode("utf-8")


_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(_LATEX_REPLACEMENTS.get(character, character) for character in text)


def _format_report_value(column: str, value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "--"
    if str(value).lower() == UNAVAILABLE:
        return "unavailable"
    if any(token in column.lower() for token in ("rate", "ci_", "estimate", "mean", "adjusted_p", "auc", "accuracy", "sensitivity", "specificity", "f1")):
        try:
            numeric = float(str(value))
            if math.isfinite(numeric):
                return "{:.3g}".format(numeric)
        except ValueError:
            pass
    return str(value)


def _select_latex_columns(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> List[str]:
    keys = {str(key) for row in rows for key in row.keys()}
    selected = [column for column in preferred if column in keys]
    if selected:
        return selected
    excluded = {"report_status", "source_artifact", "source_sha256", "reason", "_source_key", "_source_sha256"}
    return sorted(keys - excluded)[:8] or ["report_status", "reason"]


def latex_table_bytes(
    *,
    rows: Sequence[Mapping[str, Any]],
    title: str,
    label: str,
    number: str,
    supplementary: bool,
    preferred_columns: Sequence[str] = (),
) -> bytes:
    columns = _select_latex_columns(rows, preferred_columns)
    unavailable = all(row.get("report_status") == UNAVAILABLE for row in rows)
    environment = "table" if not supplementary else "table"
    lines = [
        "% " + GENERATOR_NOTICE,
        "\\begin{{{}}}[htbp]".format(environment),
        "\\centering",
        "\\caption{{{}}}".format(latex_escape(title)),
        "\\label{{{}}}".format(label),
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(latex_escape(column.replace("_", " ")) for column in columns) + r" \\",
        "\\hline",
    ]
    if unavailable:
        reason = rows[0].get("reason", "Required machine evidence is unavailable.")
        lines.append(
            "\\multicolumn{{{}}}{{l}}{{{}}} \\\\".format(len(columns), latex_escape(reason))
        )
    else:
        for row in rows:
            lines.append(
                " & ".join(
                    latex_escape(_format_report_value(column, row.get(column, "")))
                    for column in columns
                )
                + r" \\"
            )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{{{}}}".format(environment),
            "% Stable display number: {}".format(number),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _plot_script_bytes() -> bytes:
    script = r'''#!/usr/bin/env python3
"""Render hash-sealed RankCloak report plot sources; generated, do not edit."""
import argparse
import csv
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render(source, destination, title):
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    available = [row for row in rows if row.get("report_status") == "available"]
    if not available:
        reason = rows[0].get("reason", "Machine evidence unavailable") if rows else "Machine evidence unavailable"
        axis.axis("off")
        axis.text(0.5, 0.5, "Unavailable\n" + reason, ha="center", va="center", wrap=True)
    else:
        candidates = (
            "exact_payload_recovery_rate", "exact_recovery_rate", "mean", "roc_auc", "pr_auc", "balanced_accuracy",
            "R_effective_bits_per_forced_plus_tail_token", "R_B_bits_per_forced_token",
            "Q_B_nats_per_forced_token", "true_positive_rate", "precision",
        )
        y_column = next((name for name in candidates if any(_float(row.get(name)) is not None for row in available)), None)
        if y_column is None:
            axis.axis("off")
            axis.text(0.5, 0.5, "Verified rows available; no numeric plotting field in this source.", ha="center", va="center", wrap=True)
        else:
            values = [_float(row.get(y_column)) for row in available]
            positions = [index for index, value in enumerate(values) if value is not None]
            numeric = [value for value in values if value is not None]
            axis.plot(positions, numeric, marker="o", linewidth=1.2)
            axis.set_xlabel("Verified source-row order")
            axis.set_ylabel(y_column.replace("_", " "))
            axis.grid(alpha=0.25)
    axis.set_title(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        plt.close(fig)
        raise FileExistsError("Refusing to overwrite rendered figure: {}".format(destination))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + destination.stem + ".",
            suffix=destination.suffix,
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, dpi=180)
        plt.close(fig)
        os.replace(str(temporary), str(destination))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("plot_registry.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="pdf")
    parser.add_argument("--only")
    args = parser.parse_args()
    with args.registry.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        if args.only and record["plot_id"] != args.only:
            continue
        source = args.registry.parent / record["source_csv"]
        destination = args.output_dir / (record["plot_id"] + "." + args.format)
        render(source, destination, record["title"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return script.encode("utf-8")


def _source_manifest(sources: VerifiedSources) -> Dict[str, Any]:
    manifest_rows = []
    for source_key, path in sorted(sources.manifest_paths.items()):
        manifest_rows.append(
            {
                "source_key": source_key,
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts = []
    for key, artifact in sorted(sources.artifacts.items()):
        artifacts.append(
            {
                "artifact_key": key,
                "path": artifact.path.name,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "row_count": artifact.row_count,
                "manifest_declared_sha256": artifact.manifest_declared_sha256,
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "report_source_seal",
        "fixture_mode": sources.fixture_mode,
        "numeric_input_policy": "manifest-addressed machine artifacts only; no numeric overrides",
        "missing_result_policy": "explicitly unavailable; no imputation or synthetic substitution",
        "recovery_reporting_contract": {
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "primary_outcome": PAYLOAD_RECOVERY_OUTCOME,
            "semantics": PAYLOAD_RECOVERY_SEMANTICS,
            "compatibility_alias": "exact_recovery",
            "exact_rank_replay_role": "diagnostic_only",
        },
        "manifests": manifest_rows,
        "artifacts": artifacts,
    }


def _preflight_and_write(output_dir: Path, products: Mapping[str, bytes]) -> Dict[str, Path]:
    paths = {name: output_dir / name for name in products}
    for name, content in products.items():
        path = paths[name]
        if path.is_symlink():
            raise ReportArtifactConflict("Refusing to write through symlink: {}".format(path))
        if path.exists() and path.read_bytes() != content:
            raise ReportArtifactConflict(
                "Refusing to overwrite report artifact with different bytes: {}".format(path)
            )
    for name, content in products.items():
        path = paths[name]
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent), delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    return paths


def _table_preferences(table_id: str) -> Tuple[str, ...]:
    if table_id == "main_table_1":
        return (
            "model_id",
            "protocol_variant",
            "payload_class_count",
            "n_payloads",
            "payload_recovery_successes",
            "exact_payload_recovery_rate",
            "recovery_outcome_semantics",
            "ci_low",
            "ci_high",
        )
    if table_id == "main_table_2":
        return ("section", "condition", "outcome", "estimate", "ci_low", "ci_high", "n_payloads", "adjusted_p")
    return ()


def build_revision_reports(
    *,
    output_dir: Path,
    statistics_manifest: Optional[Path] = None,
    theory_manifest: Optional[Path] = None,
    detector_manifest: Optional[Path] = None,
    mixed_model_manifest: Optional[Path] = None,
    evaluator_unavailability_manifest: Optional[Path] = None,
    runtime_manifests: Sequence[Path] = (),
    preprocessing_manifests: Sequence[Path] = (),
    fixture_mode: bool = False,
) -> ReportBuild:
    """Build all planned tables and plot sources from verified manifests.

    There are intentionally no parameters for estimates, sample sizes, or other
    result values.  Scientific numbers can enter only through verified source
    artifacts.
    """

    sources = load_verified_sources(
        statistics_manifest=statistics_manifest,
        theory_manifest=theory_manifest,
        detector_manifest=detector_manifest,
        mixed_model_manifest=mixed_model_manifest,
        evaluator_unavailability_manifest=evaluator_unavailability_manifest,
        runtime_manifests=runtime_manifests,
        preprocessing_manifests=preprocessing_manifests,
        fixture_mode=fixture_mode,
    )
    registry = display_registry()
    products: Dict[str, bytes] = {}
    source_manifest = _source_manifest(sources)
    products["report_source_manifest.json"] = canonical_json_bytes(source_manifest)
    products["display_registry.json"] = canonical_json_bytes(registry)

    table_status: Dict[str, str] = {}
    main_table_rows = {
        "main_table_1": _main_table_1_rows(sources),
        "main_table_2": _main_table_2_rows(sources),
    }
    for item in registry["main"]:
        if item["type"] != "table":
            continue
        table_id = item["id"]
        rows = main_table_rows[table_id]
        table_status[table_id] = UNAVAILABLE if all(row.get("report_status") == UNAVAILABLE for row in rows) else "available"
        products["tables/{}.csv".format(table_id)] = _csv_bytes(rows)
        products["tables/{}.tex".format(table_id)] = latex_table_bytes(
            rows=rows,
            title=item["title"],
            label=item["label"],
            number=item["number"],
            supplementary=False,
            preferred_columns=_table_preferences(table_id),
        )
    for index, title in enumerate(SUPPLEMENTARY_TABLE_TITLES, start=1):
        table_id = "supplementary_table_s{}".format(index)
        item = next(value for value in registry["supplementary"] if value["id"] == table_id)
        rows = supplementary_table_rows(sources, index)
        table_status[table_id] = UNAVAILABLE if all(row.get("report_status") == UNAVAILABLE for row in rows) else "available"
        products["tables/{}.csv".format(table_id)] = _csv_bytes(rows)
        products["tables/{}.tex".format(table_id)] = latex_table_bytes(
            rows=rows,
            title=title,
            label=item["label"],
            number=item["number"],
            supplementary=True,
        )

    plot_items = [value for value in registry["main"] + registry["supplementary"] if value["type"] == "figure"]
    plot_registry_rows: List[Dict[str, str]] = []
    plot_status: Dict[str, str] = {}
    for item in plot_items:
        rows = plot_source_rows(sources, item["id"])
        plot_status[item["id"]] = UNAVAILABLE if all(row.get("report_status") == UNAVAILABLE for row in rows) else "available"
        filename = "sources/{}.csv".format(item["id"])
        products["plots/" + filename] = _csv_bytes(rows)
        plot_registry_rows.append(
            {
                "plot_id": item["id"],
                "number": item["number"],
                "label": item["label"],
                "title": item["title"],
                "source_csv": filename,
                "report_status": plot_status[item["id"]],
            }
        )
    products["plots/plot_registry.csv"] = _csv_bytes(plot_registry_rows)
    products["plots/plot_revision_figures.py"] = _plot_script_bytes()

    detector_manifest = sources.manifests.get("detector", {})
    evaluator_unavailability = sources.manifests.get("evaluator_unavailability")
    integrity = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "main_display_count": len(registry["main"]),
        "main_display_limit": MAX_MAIN_DISPLAY_ITEMS,
        "sample_size_consistency": "passed",
        "source_hash_validation": "passed",
        "numeric_override_interface": False,
        "primary_inference_source": (
            "locked_r_mixed_model_manifest"
            if "mixed_model" in sources.manifests
            else "unavailable_no_python_fallback"
        ),
        "python_pooled_effects_are_primary_inference": False,
        "evaluator_unavailability_accounting": (
            None
            if not isinstance(evaluator_unavailability, dict)
            else {
                "source_manifest_sha256": file_sha256(
                    sources.manifest_paths["evaluator_unavailability"]
                ),
                "authorized_projection_sha256": evaluator_unavailability[
                    "authorized_projection_sha256"
                ],
                "frozen_evaluator_target_units": evaluator_unavailability[
                    "frozen_evaluator_target_units"
                ],
                "scoreable_quality_estimand_units": evaluator_unavailability[
                    "scoreable_evaluator_units"
                ],
                "terminal_excluded_non_outcomes": evaluator_unavailability[
                    "upstream_dependent_unavailable_units"
                ],
                "terminal_accounted_units": evaluator_unavailability[
                    "terminal_accounted_units"
                ],
                "scoring_attempted_for_unavailable_units": False,
                "scores_imputed_or_fabricated": False,
                "analysis_policy": evaluator_unavailability["analysis_policy"],
            }
        ),
        "missing_results_are_explicit": True,
        "recovery_reporting_contract": {
            "protocol_contract_revision": PROTOCOL_CONTRACT_REVISION,
            "result_schema_revision": RESULT_SCHEMA_REVISION,
            "primary_outcome": PAYLOAD_RECOVERY_OUTCOME,
            "semantics": PAYLOAD_RECOVERY_SEMANTICS,
            "compatibility_alias": "exact_recovery",
            "exact_rank_replay_role": "diagnostic_only",
        },
        "fixture_mode": fixture_mode,
        "table_status": table_status,
        "plot_status": plot_status,
        "detector_output_hash_provenance": (
            "report_time_source_seal_plus_manifest_row_counts"
            if detector_manifest and not any(
                artifact.manifest_declared_sha256
                for key, artifact in sources.artifacts.items()
                if key.startswith("detector.")
            )
            else "manifest_declared"
        ),
    }
    products["report_integrity.json"] = canonical_json_bytes(integrity)

    output_records = []
    for relative_path, content in sorted(products.items()):
        output_records.append(
            {
                "path": relative_path,
                "sha256": bytes_sha256(content),
                "size_bytes": len(content),
            }
        )
    output_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "report_output_manifest",
        "generator_notice": GENERATOR_NOTICE,
        "files": output_records,
        "files_sha256": bytes_sha256(canonical_json_bytes(output_records)),
    }
    products["report_output_manifest.json"] = canonical_json_bytes(output_manifest)
    paths = _preflight_and_write(Path(output_dir), products)
    return ReportBuild(
        output_dir=Path(output_dir).resolve(),
        files={key: path.resolve() for key, path in paths.items()},
        source_manifest=source_manifest,
        integrity_report=integrity,
    )


def verify_report_output_manifest(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "report_output_manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise RevisionReportingError("Malformed report output manifest")
    errors = []
    records = manifest["files"]
    if bytes_sha256(canonical_json_bytes(records)) != manifest.get("files_sha256"):
        errors.append("file-list digest mismatch")
    for record in records:
        path = output_dir / str(record.get("path"))
        if not path.is_file() or path.is_symlink():
            errors.append("missing or invalid output: {}".format(record.get("path")))
            continue
        if path.stat().st_size != record.get("size_bytes"):
            errors.append("size mismatch: {}".format(record.get("path")))
        if file_sha256(path) != record.get("sha256"):
            errors.append("SHA-256 mismatch: {}".format(record.get("path")))
    return {"status": "ok" if not errors else "error", "verified_file_count": len(records), "errors": errors}


__all__ = [
    "GENERATOR_NOTICE",
    "MAIN_DISPLAYS",
    "MAX_MAIN_DISPLAY_ITEMS",
    "REPORT_SCHEMA_VERSION",
    "ReportArtifactConflict",
    "ReportBuild",
    "RevisionReportingError",
    "VerifiedArtifact",
    "VerifiedSources",
    "build_revision_reports",
    "display_registry",
    "latex_escape",
    "latex_table_bytes",
    "load_verified_sources",
    "plot_source_rows",
    "supplementary_table_rows",
    "validate_display_registry",
    "verify_report_output_manifest",
]
