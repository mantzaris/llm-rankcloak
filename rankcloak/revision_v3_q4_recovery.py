"""Replay-only visible-text recovery for historical Q4 RankCloak covers.

This module never generates a cover. It retokenizes the already recorded
historical Q4 text with the exact embedded tokenizer and recovers ranks from
that visible token stream. When retokenization reproduces the saved token IDs
exactly, the previously validated Q4 saved-ID rank trace is reused as a
logically equivalent shortcut; divergent tokenizations are replayed through
the pinned Q4 model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .revision_protocol import (
    Representation,
    context_sha256,
    decode_representation,
    recover_rank_span,
    retokenize_message,
)
from .revision_v3_generation import (
    DEFAULT_GPU_UUID,
    DEFAULT_RESULT_ROOT,
    PROJECT_ROOT,
    QUANTIZATION_PLAN,
    GenerationExecutionError,
    GpuMemoryMonitor,
    atomic_json,
    canonical_sha256,
    configure_deterministic_gpu,
    file_sha256,
    git_output,
    immutable_result,
    load_csv,
    load_json,
    load_verified_model,
    operation_provenance,
    payload_index,
    process_peak_rss_bytes,
    qwen_historical_indexes,
    quantization_source_bundle,
    representation_from_source,
    utc_now,
)


MODEL_ID = "qwen2_5_7b_instruct_q4_k_m"
SCHEMA_VERSION = "rankcloak-revision-v3-q4-visible-recovery-v1"
PHASE = "quantization_q4_visible_recovery"


def q4_visible_recovery_outcome(
    model: Any,
    q4_record: Mapping[str, object],
    representation: Representation,
) -> Mapping[str, object]:
    """Recover one existing Q4 cover after visible-text retokenization."""

    plan = q4_record["plan_row"]
    if not (
        q4_record.get("record_type") == "quantization_q4_model_backed_replay"
        and q4_record.get("population") == "rankcloak"
        and q4_record.get("new_generation_performed") is False
        and plan["quantization"] == "Q4_K_M"
        and plan["population"] == "rankcloak"
        and q4_record.get("rank_replay_exact") is True
    ):
        raise GenerationExecutionError(
            "Q4 visible recovery requires a validated historical RankCloak replay"
        )
    expected_ranks = list(map(int, q4_record["expected_ranks"]))
    if expected_ranks != list(map(int, representation.ranks)):
        raise GenerationExecutionError(
            "Q4 replay ranks differ from the authoritative representation"
        )
    saved_ids = list(map(int, q4_record["historical_output_token_ids"]))
    generated_view = {
        "full_token_ids": saved_ids,
        "full_text": str(q4_record["historical_output_text"]),
        "forced_start": 0,
        "forced_stop": len(expected_ranks),
    }
    diagnostic = retokenize_message(model, generated_view)
    if diagnostic["full_token_ids_match"]:
        recovered_ranks = list(
            map(int, q4_record["distribution_trace"]["observed_ranks"])
        )
        if recovered_ranks != expected_ranks:
            raise GenerationExecutionError(
                "validated Q4 saved-ID ranks are inconsistent with the record"
            )
        replay = {
            "ranks": recovered_ranks,
            "token_log_probabilities": None,
            "context_sha256": context_sha256(q4_record["context_token_ids"]),
            "execution_mode": "validated_saved_id_trace_for_identical_retokenization",
        }
        model_rank_replay_performed = False
    else:
        replay = recover_rank_span(
            model,
            q4_record["context_token_ids"],
            [],
            diagnostic["forced_token_ids"],
            allowed_token_mask=None,
        )
        replay = {
            **replay,
            "execution_mode": "model_replay_of_retokenized_visible_text",
        }
        recovered_ranks = list(map(int, replay["ranks"]))
        model_rank_replay_performed = True
    decoded = decode_representation(model, representation, recovered_ranks)
    return {
        "diagnostic": diagnostic,
        "replay": replay,
        "decoded": decoded,
        "exact_payload_recovery": bool(decoded["exact_payload_recovery"]),
        "exact_rank_recovery": recovered_ranks == expected_ranks,
        "model_rank_replay_performed": model_rank_replay_performed,
        "saved_token_count": len(saved_ids),
        "retokenized_token_count": len(diagnostic["retokenized_token_ids"]),
        "recovered_rank_count": len(recovered_ranks),
    }


def q4_recovery_record(
    model: Any,
    row: Mapping[str, str],
    q4_record: Mapping[str, object],
    representation: Representation,
    model_manifest: Mapping[str, object],
    monitor: GpuMemoryMonitor,
    source_path: Path,
) -> Mapping[str, object]:
    """Create a provenance-bound recovery record without generating text."""

    if q4_record["plan_row_sha256"] != canonical_sha256(row):
        raise GenerationExecutionError("Q4 source record differs from its frozen plan row")
    if q4_record["model_artifact_sha256"] != model_manifest["artifact"]["sha256"]:
        raise GenerationExecutionError("Q4 source and recovery model artifacts differ")
    started_at = utc_now()
    started_perf = time.perf_counter()
    rss_start = process_peak_rss_bytes()
    monitor.reset_peak()
    outcome = q4_visible_recovery_outcome(model, q4_record, representation)
    record = dict(
        operation_provenance(
            QUANTIZATION_PLAN,
            row,
            model_manifest,
            started_at,
            started_perf,
            rss_start,
            monitor,
        )
    )
    record["source_hashes"] = {
        **record["source_hashes"],
        str(Path(__file__).relative_to(PROJECT_ROOT)): file_sha256(Path(__file__)),
        "scripts/run_revision_v3_q4_visible_recovery.py": file_sha256(
            PROJECT_ROOT / "scripts/run_revision_v3_q4_visible_recovery.py"
        ),
    }
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "quantization_q4_visible_text_recovery",
            "phase": PHASE,
            "model_id": MODEL_ID,
            "population": "rankcloak",
            "new_cover_generation_performed": False,
            "source_q4_record_path": (
                str(source_path.relative_to(PROJECT_ROOT))
                if source_path.is_relative_to(PROJECT_ROOT)
                else str(source_path)
            ),
            "source_q4_record_sha256": file_sha256(source_path),
            "source_q4_record_canonical_sha256": canonical_sha256(q4_record),
            "source_q4_plan_id": q4_record["plan_id"],
            "historical_output_text_sha256": hashlib.sha256(
                str(q4_record["historical_output_text"]).encode("utf-8")
            ).hexdigest(),
            "visible_text_retokenization": outcome,
            "validation": {
                "source_q4_plan_row_exact": True,
                "source_q4_model_artifact_exact": True,
                "source_q4_saved_rank_replay_exact": (
                    q4_record["rank_replay_exact"] is True
                ),
                "authoritative_representation_ranks_exact": (
                    list(map(int, representation.ranks))
                    == list(map(int, q4_record["expected_ranks"]))
                ),
                "visible_text_recovery_outcome_available": True,
                "new_cover_generation_performed_is_false": True,
            },
        }
    )
    if not all(record["validation"].values()):
        raise GenerationExecutionError("Q4 visible-text recovery validation failed")
    return record


def recovery_rows() -> list[dict[str, str]]:
    return sorted(
        [
            row
            for row in load_csv(QUANTIZATION_PLAN)
            if row["model_id"] == MODEL_ID
            and row["quantization"] == "Q4_K_M"
            and row["population"] == "rankcloak"
        ],
        key=lambda row: row["plan_id"],
    )


def recovery_path(result_root: Path, plan_id: str) -> Path:
    return (
        result_root
        / "raw"
        / PHASE
        / MODEL_ID
        / (str(plan_id) + ".json")
    )


def existing_recovery_valid(
    path: Path,
    row: Mapping[str, str],
    q4_source_path: Path,
    model_artifact_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    record = load_json(path)
    if not (
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("record_type") == "quantization_q4_visible_text_recovery"
        and record.get("execution_status") == "completed"
        and record.get("plan_row_sha256") == canonical_sha256(row)
        and record.get("model_artifact_sha256") == model_artifact_sha256
        and record.get("source_q4_record_sha256") == file_sha256(q4_source_path)
        and record.get("new_cover_generation_performed") is False
        and all(record.get("validation", {}).values())
    ):
        raise GenerationExecutionError(
            "existing Q4 visible-text recovery record failed validation"
        )
    return True


def run(
    *,
    result_root: Path,
    gpu_uuid: str,
    limit: Optional[int] = None,
) -> Mapping[str, object]:
    rows = recovery_rows()
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise GenerationExecutionError("no Q4 recovery rows selected")
    configure_deterministic_gpu(gpu_uuid)
    monitor = GpuMemoryMonitor(gpu_uuid)
    monitor.start()
    model = None
    started_at = utc_now()
    started_perf = time.perf_counter()
    completed = 0
    resumed = 0
    try:
        model, model_manifest = load_verified_model(MODEL_ID, gpu_uuid, monitor)
        model_sha = str(model_manifest["artifact"]["sha256"])
        indexes = qwen_historical_indexes()
        payloads = payload_index()
        for index, row in enumerate(rows, start=1):
            source_path = (
                result_root
                / "raw"
                / "quantization"
                / MODEL_ID
                / (row["plan_id"] + ".json")
            )
            if not source_path.is_file():
                raise GenerationExecutionError(
                    "existing Q4 replay record is unavailable: " + str(source_path)
                )
            target = recovery_path(result_root, row["plan_id"])
            if existing_recovery_valid(target, row, source_path, model_sha):
                resumed += 1
                continue
            q4_record = load_json(source_path)
            rank, _control, _task = quantization_source_bundle(row, indexes)
            representation = representation_from_source(rank, payloads)
            record = q4_recovery_record(
                model,
                row,
                q4_record,
                representation,
                model_manifest,
                monitor,
                source_path,
            )
            immutable_result(target, record)
            completed += 1
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(
                    json.dumps(
                        {
                            "event": "q4_visible_recovery_progress",
                            "completed_this_run": completed,
                            "resumed": resumed,
                            "index": index,
                            "selected": len(rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        completed_at = utc_now()
        run_record = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "model_id": MODEL_ID,
            "new_cover_generation_performed": False,
            "selected_plan_row_count": len(rows),
            "completed_this_run": completed,
            "resumed_count": resumed,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": float(time.perf_counter() - started_perf),
            "gpu_uuid": gpu_uuid,
            "model_manifest": model_manifest,
            "execution_git_commit": git_output("rev-parse", "HEAD"),
            "command": [sys.executable, *sys.argv],
        }
        run_path = (
            result_root
            / "provenance"
            / "model_runs"
            / PHASE
            / MODEL_ID
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        )
        atomic_json(run_path, run_record)
        status = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "model_id": MODEL_ID,
            "planned": len(recovery_rows()),
            "completed": sum(
                recovery_path(result_root, row["plan_id"]).is_file()
                for row in recovery_rows()
            ),
            "new_cover_generation_performed": False,
            "checked_at": completed_at,
        }
        atomic_json(
            result_root
            / "status"
            / "full"
            / (PHASE + "__" + MODEL_ID + ".json"),
            status,
        )
        return {**run_record, "status": status}
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()
            del model
            gc.collect()
        monitor.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    rows = recovery_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
    result_root = args.output_dir.resolve()
    if args.dry_run or args.status:
        completed = sum(
            recovery_path(result_root, row["plan_id"]).is_file() for row in rows
        )
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": PHASE,
                    "dry_run": bool(args.dry_run),
                    "model_loaded": False,
                    "selected": len(rows),
                    "completed": completed,
                    "pending": len(rows) - completed,
                    "new_cover_generation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.gpu_uuid != DEFAULT_GPU_UUID:
        parser.error("the authorized RTX 5000 Ada --gpu-uuid is required")
    summary = run(
        result_root=result_root,
        gpu_uuid=args.gpu_uuid,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PHASE",
    "SCHEMA_VERSION",
    "existing_recovery_valid",
    "main",
    "q4_recovery_record",
    "q4_visible_recovery_outcome",
    "recovery_path",
    "recovery_rows",
    "run",
]
