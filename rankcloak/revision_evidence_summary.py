"""Hash-validated, evidence-only summaries for the revision experiment package.

This module renders computational audit records.  It does not generate manuscript
text, publication tables, response-letter language, or human-participant results.
Curated mappings remain visibly classified as confirmatory, secondary,
exploratory, unavailable, or unresolved, while every referenced artifact is
resolved and hashed before any output is published.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEC_SCHEMA = "rankcloak-revision-evidence-records-v1"
MANIFEST_SCHEMA = "rankcloak-revision-evidence-summary-v1"
GPU_BUDGET_SCHEMA = "rankcloak-revision-gpu-budget-v1"
HISTORICAL_GPU_HOURS_FLOOR = 62.4783840698
GPU_HOURS_CEILING = 165.0

EVIDENCE_STATUSES = {
    "confirmatory",
    "secondary",
    "supporting",
    "exploratory",
    "diagnostic",
    "unavailable",
    "unresolved",
    "external_gate",
}
PRIORITY_CLASSES = {
    "core evidence",
    "supporting evidence",
    "diagnostic evidence",
    "unresolved",
}
FORBIDDEN_REFERENCE_PARTS = {".paper", "paper_artifacts"}
FORBIDDEN_REFERENCE_SUFFIXES = {".tex", ".doc", ".docx"}


class EvidenceSummaryError(ValueError):
    """Raised when evidence records or referenced artifacts are inconsistent."""


@dataclass(frozen=True)
class EvidenceSummaryArtifacts:
    output_paths: tuple[str, ...]
    manifest_path: str
    finding_count: int
    reviewer_concern_count: int
    stage_count: int
    total_gpu_hours: float


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceSummaryError(f"Missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceSummaryError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceSummaryError(f"{label} must contain a JSON object")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EvidenceSummaryError(f"Missing or unsafe {label}: {path}")
    return path.resolve()


def _resolve_reference(project_root: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise EvidenceSummaryError(f"{label} must be a nonempty path")
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else project_root / candidate
    resolved = _regular_file(path, label=label)
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as exc:
        raise EvidenceSummaryError(
            f"{label} must remain inside the repository: {resolved}"
        ) from exc
    if (
        any(part in FORBIDDEN_REFERENCE_PARTS for part in relative.parts)
        or relative.suffix.lower() in FORBIDDEN_REFERENCE_SUFFIXES
    ):
        raise EvidenceSummaryError(
            f"Publication/manuscript artifact is out of scope: {relative}"
        )
    return resolved


def _row_count(path: Path) -> int | None:
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle, delimiter=delimiter)) - 1, 0)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


def _identity(path: Path, project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(project_root)),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    rows = _row_count(path)
    if rows is not None:
        result["row_count"] = rows
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceSummaryError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceSummaryError(f"{label} must be finite and nonnegative")
    return result


def _required_text(record: Mapping[str, Any], field: str, *, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSummaryError(f"{label}.{field} must be nonempty text")
    return value.strip()


def _validate_record_set(
    values: object,
    *,
    label: str,
    fields: Sequence[str],
    require_status: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise EvidenceSummaryError(f"{label} must be a nonempty list")
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise EvidenceSummaryError(f"{label}[{index}] must be an object")
        record = dict(raw)
        identifier = _required_text(record, fields[0], label=f"{label}[{index}]")
        if identifier in identifiers:
            raise EvidenceSummaryError(f"Duplicate {label} identifier: {identifier}")
        identifiers.add(identifier)
        for field in fields[1:]:
            _required_text(record, field, label=f"{label}[{index}]")
        if require_status:
            status = _required_text(record, "status", label=f"{label}[{index}]")
            if status not in EVIDENCE_STATUSES:
                raise EvidenceSummaryError(f"Unknown evidence status: {status}")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise EvidenceSummaryError(f"{label}[{index}].artifacts must be nonempty")
        records.append(record)
    return records


def _validate_progress(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = value.get("counts")
    stages = value.get("stage_progress")
    if not isinstance(counts, Mapping) or not isinstance(stages, list) or not stages:
        raise EvidenceSummaryError("Progress ledger lacks counts or stage_progress")
    normalized: list[dict[str, Any]] = []
    for raw in stages:
        if not isinstance(raw, Mapping):
            raise EvidenceSummaryError("Progress stage row is malformed")
        row = {
            key: int(raw.get(key, -1))
            for key in ("total", "completed", "successes", "unavailable", "failures", "remaining")
        }
        row["stage"] = str(raw.get("stage", ""))
        if not row["stage"] or any(row[key] < 0 for key in row if key != "stage"):
            raise EvidenceSummaryError("Progress stage row has invalid values")
        if (
            row["completed"] + row["remaining"] != row["total"]
            or row["successes"] + row["unavailable"] + row["failures"]
            != row["completed"]
        ):
            raise EvidenceSummaryError(f"Progress arithmetic differs for {row['stage']}")
        normalized.append(row)
    for key in ("total", "completed", "successes", "unavailable", "failures", "remaining"):
        if sum(row[key] for row in normalized) != int(counts.get(key, -1)):
            raise EvidenceSummaryError(f"Progress aggregate {key} differs")
    return normalized


def _validate_payloads(path: Path, expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceSummaryError(
                    f"Payload corpus line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise EvidenceSummaryError("Payload corpus rows must be objects")
            rows.append(value)
    counts = Counter(str(row.get("payload_class", "")) for row in rows)
    if (
        len(rows) != int(expected.get("payload_count", -1))
        or len(counts) != int(expected.get("class_count", -1))
        or any(count != int(expected.get("per_class", -1)) for count in counts.values())
        or "" in counts
    ):
        raise EvidenceSummaryError("Payload corpus counts differ from evidence spec")
    result = []
    for payload_class, count in sorted(counts.items()):
        cell = [row for row in rows if str(row["payload_class"]) == payload_class]
        algorithms = sorted({str(row.get("algorithm", "")) for row in cell})
        result.append(
            {
                "payload_class": payload_class,
                "algorithm": "; ".join(algorithms),
                "payload_count": count,
                "minimum_artifact_bits": min(int(row["artifact_bit_length"]) for row in cell),
                "maximum_artifact_bits": max(int(row["artifact_bit_length"]) for row in cell),
            }
        )
    return result


def _validate_payload_manifest(
    value: Mapping[str, Any],
    *,
    payload_path: Path,
    payload_rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> None:
    class_counts = {
        str(row["payload_class"]): int(row["payload_count"])
        for row in payload_rows
    }
    validation = value.get("validation")
    expected_corpus_sha = str(expected.get("payload_corpus_sha256", ""))
    if (
        value.get("schema_version") != "1.0"
        or value.get("manifest_type") != "revision_payload_corpus"
        or int(value.get("payload_count", -1)) != sum(class_counts.values())
        or value.get("class_counts") != class_counts
        or value.get("payload_file_sha256") != file_sha256(payload_path)
        or int(value.get("payload_file_size_bytes", -1))
        != int(payload_path.stat().st_size)
        or len(expected_corpus_sha) != 64
        or value.get("corpus_sha256") != expected_corpus_sha
        or not isinstance(value.get("cryptography_version"), str)
        or not value["cryptography_version"].strip()
        or not isinstance(validation, Mapping)
        or validation.get("status") != "ok"
        or validation.get("errors") != []
        or validation.get("invalid_payload_names") != []
        or int(validation.get("payload_count", -1)) != sum(class_counts.values())
        or validation.get("class_counts") != class_counts
        or validation.get("corpus_sha256") != expected_corpus_sha
    ):
        raise EvidenceSummaryError(
            "Payload manifest does not establish complete cryptographic validation"
        )


def _validate_design_counts(
    prompts: Mapping[str, Any], models: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[dict[str, Any]]:
    category_count = int(prompts.get("category_count", -1))
    template_count = int(prompts.get("template_count", -1))
    model_count = int(models.get("planned_model_count", -1))
    if (
        category_count != int(expected.get("english_prompt_categories", -1))
        or template_count != int(expected.get("english_prompt_templates", -1))
        or model_count != int(expected.get("model_families", -1))
        or not isinstance(models.get("models"), list)
        or len(models["models"]) != model_count
    ):
        raise EvidenceSummaryError("Frozen model/prompt design counts differ")
    return [
        {"design_dimension": "payload_classes", "count": int(expected["class_count"])},
        {"design_dimension": "payloads", "count": int(expected["payload_count"])},
        {"design_dimension": "model_families", "count": model_count},
        {"design_dimension": "english_prompt_categories", "count": category_count},
        {"design_dimension": "english_prompt_templates", "count": template_count},
        {"design_dimension": "secondary_languages", "count": int(expected["secondary_languages"])},
    ]


def _gpu_interval_identity(
    interval: Mapping[str, Any], *, source: str, index: int
) -> tuple[object, ...]:
    """Return a stable process-interval identity, with a fixture-safe fallback."""

    fields = ("pid", "process_start_ticks", "started_at_utc")
    if all(interval.get(field) is not None for field in fields):
        return tuple(interval[field] for field in fields)
    return (source, index)


def _validate_gpu_accounting(
    value: Mapping[str, Any], *, label: str
) -> tuple[float, list[dict[str, Any]]]:
    seconds = _finite_nonnegative(
        value.get("cumulative_elapsed_seconds"),
        label=f"{label} cumulative GPU seconds",
    )
    intervals = value.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise EvidenceSummaryError(f"{label} GPU accounting lacks intervals")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(intervals):
        if not isinstance(raw, Mapping):
            raise EvidenceSummaryError(f"{label} GPU interval {index} is malformed")
        interval = dict(raw)
        interval["elapsed_seconds"] = _finite_nonnegative(
            interval.get("elapsed_seconds"),
            label=f"{label} GPU interval seconds",
        )
        normalized.append(interval)
    interval_sum = sum(float(row["elapsed_seconds"]) for row in normalized)
    if not math.isclose(seconds, interval_sum, rel_tol=0.0, abs_tol=1e-6):
        raise EvidenceSummaryError(f"{label} GPU interval accounting differs")
    return seconds, normalized


def _gpu_interval_bounds(
    interval: Mapping[str, Any], *, label: str
) -> tuple[datetime, datetime] | None:
    started_raw = interval.get("started_at_utc")
    completed_raw = interval.get("completed_at_utc")
    if started_raw is None and completed_raw is None:
        return None
    if not isinstance(started_raw, str) or not isinstance(completed_raw, str):
        raise EvidenceSummaryError(f"{label} GPU interval is not terminal")
    try:
        started = datetime.fromisoformat(started_raw)
        completed = datetime.fromisoformat(completed_raw)
    except ValueError as exc:
        raise EvidenceSummaryError(f"{label} GPU interval timestamp is invalid") from exc
    if completed < started:
        raise EvidenceSummaryError(f"{label} GPU interval ends before it starts")
    elapsed = float(interval["elapsed_seconds"])
    observed = (completed - started).total_seconds()
    if not math.isclose(elapsed, observed, rel_tol=0.0, abs_tol=1e-3):
        raise EvidenceSummaryError(f"{label} GPU interval duration differs")
    return started, completed


def _read_pre_final_gpu_ledger_identity(
    identity: Mapping[str, Any]
) -> dict[str, Any]:
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceSummaryError("Pre-final GPU ledger identity lacks a path")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise EvidenceSummaryError("Pre-final GPU ledger identity path is unsafe")
    if (
        file_sha256(path) != identity.get("sha256")
        or int(path.stat().st_size) != int(identity.get("size_bytes", -1))
    ):
        raise EvidenceSummaryError("Pre-final GPU ledger file identity differs")
    ledger = _read_json(path, label="pre-final GPU accounting ledger")
    unsigned = dict(ledger)
    claimed = unsigned.pop("ledger_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise EvidenceSummaryError("Pre-final GPU ledger self-hash differs")
    if (
        claimed != identity.get("ledger_sha256")
        or ledger.get("sources_sha256") != identity.get("sources_sha256")
        or ledger.get("intervals_sha256") != identity.get("intervals_sha256")
        or not math.isclose(
            float(ledger.get("cumulative_elapsed_seconds", -1.0)),
            float(identity.get("cumulative_elapsed_seconds", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise EvidenceSummaryError("Pre-final GPU ledger signed identity differs")
    return ledger


def _validated_detector_gpu_union(value: Mapping[str, Any]) -> dict[str, Any]:
    terminal = value.get("gpu_accounting")
    if not isinstance(terminal, Mapping):
        raise EvidenceSummaryError("Detector run lacks terminal GPU accounting")
    terminal_seconds, terminal_intervals = _validate_gpu_accounting(
        terminal, label="terminal detector"
    )

    pre_final = value.get("pre_final_gpu_accounting_ledger")
    sources: list[tuple[str, list[dict[str, Any]]]] = [
        ("terminal_detector", terminal_intervals)
    ]
    pre_final_seconds = 0.0
    if pre_final is not None:
        if not isinstance(pre_final, Mapping):
            raise EvidenceSummaryError("Pre-final GPU accounting ledger is malformed")
        if "intervals" not in pre_final:
            pre_final = _read_pre_final_gpu_ledger_identity(pre_final)
        pre_final_seconds, pre_final_intervals = _validate_gpu_accounting(
            pre_final, label="pre-final detector"
        )
        terminal_uuid = terminal.get("gpu_uuid")
        if (
            pre_final.get("gpu_uuid") != terminal_uuid
            or pre_final.get("device") != terminal.get("device")
        ):
            raise EvidenceSummaryError("Detector GPU accounting device identity differs")
        sources.insert(0, ("pre_final_detector", pre_final_intervals))

    union: dict[tuple[object, ...], dict[str, Any]] = {}
    duplicate_count = 0
    for source, intervals in sources:
        for index, interval in enumerate(intervals):
            identity = _gpu_interval_identity(interval, source=source, index=index)
            existing = union.get(identity)
            if existing is None:
                union[identity] = interval
            elif existing != interval:
                raise EvidenceSummaryError(
                    "Duplicate detector GPU process interval differs across ledgers"
                )
            else:
                duplicate_count += 1

    bounded: list[tuple[datetime, datetime, tuple[object, ...]]] = []
    for identity, interval in union.items():
        bounds = _gpu_interval_bounds(interval, label="detector")
        if bounds is not None:
            bounded.append((*bounds, identity))
    bounded.sort(key=lambda row: row[0])
    for previous, current in zip(bounded, bounded[1:]):
        if current[0] < previous[1]:
            raise EvidenceSummaryError(
                "Distinct detector GPU process intervals overlap in wall time"
            )

    union_seconds = float(
        sum(float(interval["elapsed_seconds"]) for interval in union.values())
    )
    return {
        "union_seconds": union_seconds,
        "union_interval_count": len(union),
        "duplicate_interval_count": duplicate_count,
        "terminal_seconds": terminal_seconds,
        "terminal_interval_count": len(terminal_intervals),
        "pre_final_seconds": pre_final_seconds,
        "pre_final_interval_count": (
            0 if pre_final is None else len(pre_final.get("intervals", []))
        ),
        "gpu_uuid": terminal.get("gpu_uuid"),
        "device": terminal.get("device"),
        "accounting_policy": (
            "nonoverlapping_union_of_pre_final_and_terminal_detector_intervals_v1"
        ),
    }


def _validate_detector_run(
    value: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    detector_stage = next((row for row in stages if row["stage"] == "neural_detector"), None)
    if detector_stage is None:
        raise EvidenceSummaryError("Progress ledger lacks neural_detector stage")
    if (
        value.get("schema_version") != "rankcloak-revision-detector-run-v2"
        or value.get("execution_mode") != "confirmatory"
        or value.get("confirmatory_complete") is not True
        or value.get("device") != "cuda:0"
        or int(value.get("failure_count", -1)) != 0
        or int(value.get("completed_fit_count", -1)) != detector_stage["completed"]
        or int(value.get("total_fit_count", -1)) != detector_stage["total"]
    ):
        raise EvidenceSummaryError("Detector run is not a complete frozen CUDA matrix")
    return _validated_detector_gpu_union(value)


def _validate_human_status(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != "rankcloak-human-evaluation-status-v1"
        or value.get("status")
        != "UNCOLLECTED_BLOCKED_NO_HUMAN_PARTICIPANT_DATA"
        or int(value.get("human_participant_rows", -1)) != 0
        or int(value.get("human_rating_rows", -1)) != 0
        or value.get("human_outcomes_estimated") is not False
        or value.get("automated_metrics_are_human_rating_substitutes") is not False
        or value.get("recruitment_authorized") is not False
        or value.get("survey_deployed") is not False
    ):
        raise EvidenceSummaryError(
            "Human-evaluation status does not establish zero uncollected ratings"
        )


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _artifact_text(record: Mapping[str, Any]) -> str:
    return "; ".join(str(value) for value in record["artifacts"])


def _atomic_bytes(target: Path, content: bytes) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(content)
    os.replace(temporary, target)


def _atomic_json(target: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        target,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _atomic_csv(target: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceSummaryError(f"Cannot write empty source table: {target.name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise EvidenceSummaryError(f"Source-table columns differ: {target.name}")
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def _signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = canonical_json_sha256(result)
    return result


def build_evidence_summary(
    *,
    project_root: str | Path,
    package_root: str | Path,
    evidence_spec: str | Path,
    progress_ledger: str | Path,
    payload_corpus: str | Path,
    payload_manifest: str | Path,
    prompts_config: str | Path,
    models_config: str | Path,
    detector_run_manifest: str | Path,
    human_evaluation_status: str | Path,
    observed_head: str,
    command: str | None = None,
    overwrite: bool = False,
) -> EvidenceSummaryArtifacts:
    """Render the evidence-only package summaries after fail-closed validation."""

    root = Path(project_root).resolve()
    package = Path(package_root).resolve()
    if package.is_symlink() or not package.is_dir():
        raise EvidenceSummaryError(f"Missing or unsafe package root: {package}")
    try:
        package.relative_to(root)
    except ValueError as exc:
        raise EvidenceSummaryError("Package root must remain inside the repository") from exc

    input_paths = {
        "evidence_spec": _regular_file(Path(evidence_spec), label="evidence spec"),
        "progress_ledger": _regular_file(Path(progress_ledger), label="progress ledger"),
        "payload_corpus": _regular_file(Path(payload_corpus), label="payload corpus"),
        "payload_manifest": _regular_file(
            Path(payload_manifest), label="payload manifest"
        ),
        "prompts_config": _regular_file(Path(prompts_config), label="prompts config"),
        "models_config": _regular_file(Path(models_config), label="models config"),
        "detector_run_manifest": _regular_file(
            Path(detector_run_manifest), label="detector run manifest"
        ),
        "human_evaluation_status": _regular_file(
            Path(human_evaluation_status), label="human-evaluation status"
        ),
    }
    spec = _read_json(input_paths["evidence_spec"], label="evidence spec")
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise EvidenceSummaryError("Evidence spec schema differs")
    expected_head = str(spec.get("expected_head", ""))
    if len(expected_head) != 40 or observed_head != expected_head:
        raise EvidenceSummaryError("Git HEAD differs from the evidence spec")

    findings = _validate_record_set(
        spec.get("findings"),
        label="findings",
        fields=("finding_id", "topic", "configuration", "sample_size", "estimate", "uncertainty", "effect_size", "limitation"),
    )
    concerns = _validate_record_set(
        spec.get("reviewer_concerns"),
        label="reviewer_concerns",
        fields=("concern_id", "request", "configuration", "sample_size", "principal_numeric_result", "uncertainty_or_test", "limitation"),
    )
    limitations = _validate_record_set(
        spec.get("limitations"),
        label="limitations",
        fields=("limitation_id", "item", "evidence"),
    )
    priorities = _validate_record_set(
        spec.get("artifact_priorities"),
        label="artifact_priorities",
        fields=("artifact_id", "artifact", "classification", "rationale", "source_data", "generation_command"),
        require_status=False,
    )
    for record in priorities:
        if record["classification"] not in PRIORITY_CLASSES:
            raise EvidenceSummaryError(
                f"Unknown artifact priority classification: {record['classification']}"
            )
    audits = _validate_record_set(
        spec.get("audit_items"),
        label="audit_items",
        fields=("audit_id", "item", "result", "notes"),
    )

    reference_cache: dict[str, dict[str, Any]] = {}
    for record in [*findings, *concerns, *limitations, *priorities, *audits]:
        for raw in record["artifacts"]:
            path = _resolve_reference(root, raw, label="evidence artifact")
            key = str(path.relative_to(root))
            reference_cache.setdefault(key, _identity(path, root))

    progress = _read_json(input_paths["progress_ledger"], label="progress ledger")
    stages = _validate_progress(progress)
    expected = spec.get("design_expectations")
    if not isinstance(expected, Mapping):
        raise EvidenceSummaryError("Evidence spec lacks design_expectations")
    payload_rows = _validate_payloads(input_paths["payload_corpus"], expected)
    _validate_payload_manifest(
        _read_json(input_paths["payload_manifest"], label="payload manifest"),
        payload_path=input_paths["payload_corpus"],
        payload_rows=payload_rows,
        expected=expected,
    )
    design_rows = _validate_design_counts(
        _read_json(input_paths["prompts_config"], label="prompts config"),
        _read_json(input_paths["models_config"], label="models config"),
        expected,
    )
    detector = _read_json(input_paths["detector_run_manifest"], label="detector run manifest")
    detector_gpu_accounting = _validate_detector_run(detector, stages)
    detector_gpu_seconds = float(detector_gpu_accounting["union_seconds"])
    _validate_human_status(
        _read_json(
            input_paths["human_evaluation_status"], label="human-evaluation status"
        )
    )

    historical = _finite_nonnegative(
        spec.get("historical_gpu_hours_floor"), label="historical GPU-hours floor"
    )
    ceiling = _finite_nonnegative(spec.get("gpu_hours_ceiling"), label="GPU-hours ceiling")
    if historical != HISTORICAL_GPU_HOURS_FLOOR or ceiling != GPU_HOURS_CEILING:
        raise EvidenceSummaryError("GPU floor or hard ceiling differs from the authorized values")
    new_hours = detector_gpu_seconds / 3600.0
    total_hours = historical + new_hours
    remaining_hours = ceiling - total_hours
    if remaining_hours < 0.0:
        raise EvidenceSummaryError("Observed cumulative GPU usage exceeds the hard ceiling")
    budget = _signed(
        {
            "schema_version": GPU_BUDGET_SCHEMA,
            "status": "within_hard_ceiling",
            "historical_external_gpu_hours_floor": historical,
            "new_detector_accounting_seconds": detector_gpu_seconds,
            "new_detector_accounting_gpu_hours": new_hours,
            "cumulative_gpu_hours": total_hours,
            "hard_ceiling_gpu_hours": ceiling,
            "remaining_headroom_gpu_hours": remaining_hours,
            "historical_floor_was_not_reset": True,
            "accounting_policy": detector_gpu_accounting["accounting_policy"],
            "accounting_components": {
                "pre_final_seconds": detector_gpu_accounting["pre_final_seconds"],
                "pre_final_interval_count": detector_gpu_accounting[
                    "pre_final_interval_count"
                ],
                "terminal_seconds": detector_gpu_accounting["terminal_seconds"],
                "terminal_interval_count": detector_gpu_accounting[
                    "terminal_interval_count"
                ],
                "duplicate_interval_count": detector_gpu_accounting[
                    "duplicate_interval_count"
                ],
                "union_interval_count": detector_gpu_accounting[
                    "union_interval_count"
                ],
            },
            "gpu_uuid": detector_gpu_accounting["gpu_uuid"],
            "detector_run_manifest": _identity(input_paths["detector_run_manifest"], root),
        },
        "budget_sha256",
    )

    output_relatives = (
        "STATUS.md",
        "gpu_budget.json",
        "reviewer_evidence_matrix.md",
        "verified_findings.md",
        "limitations_and_unresolved_items.md",
        "artifact_priority_index.md",
        "tables/stage_counts.csv",
        "tables/corpus_counts.csv",
        "tables/design_counts.csv",
        "tables/verified_findings.csv",
        "tables/reviewer_evidence_matrix.csv",
        "tables/artifact_priority_index.csv",
        "tables/evidence_artifact_references.csv",
        "evidence_summary_manifest.json",
    )
    targets = {relative: package / relative for relative in output_relatives}
    for target in targets.values():
        if target.exists() and not overwrite:
            raise EvidenceSummaryError(f"Refusing to overwrite evidence summary: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    stage_rows = [dict(row) for row in stages]
    finding_rows = [
        {
            "finding_id": row["finding_id"],
            "topic": row["topic"],
            "configuration": row["configuration"],
            "sample_size": row["sample_size"],
            "estimate": row["estimate"],
            "uncertainty": row["uncertainty"],
            "effect_size": row["effect_size"],
            "status": row["status"],
            "artifacts": _artifact_text(row),
            "limitation": row["limitation"],
        }
        for row in findings
    ]
    concern_rows = [
        {
            "concern_id": row["concern_id"],
            "request": row["request"],
            "configuration": row["configuration"],
            "sample_size": row["sample_size"],
            "principal_numeric_result": row["principal_numeric_result"],
            "uncertainty_or_test": row["uncertainty_or_test"],
            "artifacts": _artifact_text(row),
            "status": row["status"],
            "limitation": row["limitation"],
        }
        for row in concerns
    ]
    priority_rows = [
        {
            "artifact_id": row["artifact_id"],
            "artifact": row["artifact"],
            "classification": row["classification"],
            "rationale": row["rationale"],
            "source_data": row["source_data"],
            "generation_command": row["generation_command"],
        }
        for row in priorities
    ]
    reference_rows = [
        {
            "path": reference_cache[key]["path"],
            "sha256": reference_cache[key]["sha256"],
            "size_bytes": reference_cache[key]["size_bytes"],
            "row_count": reference_cache[key].get("row_count"),
        }
        for key in sorted(reference_cache)
    ]

    _atomic_csv(targets["tables/stage_counts.csv"], stage_rows)
    _atomic_csv(targets["tables/corpus_counts.csv"], payload_rows)
    _atomic_csv(targets["tables/design_counts.csv"], design_rows)
    _atomic_csv(targets["tables/verified_findings.csv"], finding_rows)
    _atomic_csv(targets["tables/reviewer_evidence_matrix.csv"], concern_rows)
    _atomic_csv(targets["tables/artifact_priority_index.csv"], priority_rows)
    _atomic_csv(targets["tables/evidence_artifact_references.csv"], reference_rows)
    _atomic_json(targets["gpu_budget.json"], budget)

    status_text = "\n".join(
        [
            "# Computational evidence status",
            "",
            "Evidence artifact only. This file does not revise manuscript or response-letter text.",
            "",
            "## Verified work-unit ledger",
            "",
            _markdown_table(
                ("Stage", "Completed", "Total", "Successful", "Unavailable", "Failures", "Remaining"),
                (
                    (row["stage"], row["completed"], row["total"], row["successes"], row["unavailable"], row["failures"], row["remaining"])
                    for row in stages
                ),
            ),
            "",
            "## Audit state",
            "",
            _markdown_table(
                ("Item", "Result", "Status", "Evidence", "Notes"),
                ((row["item"], row["result"], row["status"], _artifact_text(row), row["notes"]) for row in audits),
            ),
            "",
            "## GPU accounting",
            "",
            f"Historical floor: {historical:.10f} GPU-hours; new non-overlapping detector interval union: {new_hours:.10f}; cumulative: {total_hours:.10f}; remaining below the 165-hour ceiling: {remaining_hours:.10f}.",
            "",
            "## Unresolved or external items",
            "",
            _markdown_table(
                ("Item", "Status", "Evidence"),
                ((row["item"], row["status"], row["evidence"]) for row in limitations if row["status"] in {"unresolved", "unavailable", "external_gate"}),
            ),
            "",
        ]
    )
    _atomic_bytes(targets["STATUS.md"], status_text.encode("utf-8"))

    findings_text = "\n".join(
        [
            "# Verified computational findings",
            "",
            "Neutral evidence inventory; automated quality measures are not human ratings.",
            "",
            _markdown_table(
                ("ID", "Topic", "Configuration", "Sample size", "Estimate", "95% interval/test", "Effect size", "Status", "Artifacts", "Limitation"),
                ((row["finding_id"], row["topic"], row["configuration"], row["sample_size"], row["estimate"], row["uncertainty"], row["effect_size"], row["status"], _artifact_text(row), row["limitation"]) for row in findings),
            ),
            "",
        ]
    )
    _atomic_bytes(targets["verified_findings.md"], findings_text.encode("utf-8"))

    concern_text = "\n".join(
        [
            "# Reviewer concern to computational evidence map",
            "",
            "Evidence artifact only; no response-letter wording is supplied.",
            "",
            _markdown_table(
                ("Concern", "Request", "Configuration", "Sample size", "Principal numeric result", "Uncertainty/test", "Artifacts", "Status", "Required limitation"),
                ((row["concern_id"], row["request"], row["configuration"], row["sample_size"], row["principal_numeric_result"], row["uncertainty_or_test"], _artifact_text(row), row["status"], row["limitation"]) for row in concerns),
            ),
            "",
        ]
    )
    _atomic_bytes(targets["reviewer_evidence_matrix.md"], concern_text.encode("utf-8"))

    limitation_text = "\n".join(
        [
            "# Limitations and unresolved computational items",
            "",
            _markdown_table(
                ("ID", "Item", "Status", "Evidence", "Artifacts"),
                ((row["limitation_id"], row["item"], row["status"], row["evidence"], _artifact_text(row)) for row in limitations),
            ),
            "",
        ]
    )
    _atomic_bytes(
        targets["limitations_and_unresolved_items.md"], limitation_text.encode("utf-8")
    )

    priority_text = "\n".join(
        [
            "# Computational artifact priority index",
            "",
            "Priority describes evidentiary role only; it does not allocate manuscript pages.",
            "",
            _markdown_table(
                ("Artifact", "Classification", "Rationale", "Source data", "Generation command"),
                ((row["artifact"], row["classification"], row["rationale"], row["source_data"], row["generation_command"]) for row in priorities),
            ),
            "",
        ]
    )
    _atomic_bytes(targets["artifact_priority_index.md"], priority_text.encode("utf-8"))

    output_identities = {
        relative: _identity(path, root)
        for relative, path in targets.items()
        if relative != "evidence_summary_manifest.json"
    }
    manifest = _signed(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "passed",
            "scope": "computational_evidence_only_no_manuscript_or_response_text",
            "git_head": observed_head,
            "inputs": {name: _identity(path, root) for name, path in input_paths.items()},
            "referenced_artifacts": reference_rows,
            "outputs": output_identities,
            "summary": {
                "audit_item_count": len(audits),
                "finding_count": len(findings),
                "reviewer_concern_count": len(concerns),
                "limitation_count": len(limitations),
                "artifact_priority_count": len(priorities),
                "stage_count": len(stages),
                "payload_count": sum(row["payload_count"] for row in payload_rows),
                "cumulative_gpu_hours": total_hours,
            },
            "generation_command": command,
        },
        "manifest_sha256",
    )
    _atomic_json(targets["evidence_summary_manifest.json"], manifest)
    return EvidenceSummaryArtifacts(
        output_paths=tuple(str(targets[name]) for name in output_relatives),
        manifest_path=str(targets["evidence_summary_manifest.json"]),
        finding_count=len(findings),
        reviewer_concern_count=len(concerns),
        stage_count=len(stages),
        total_gpu_hours=total_hours,
    )
