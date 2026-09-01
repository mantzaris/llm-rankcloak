#!/usr/bin/env python3
"""Prepare immutable revision-V3 corpus audits and reusable analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankcloak.revision_v3_analysis import (  # noqa: E402
    build_topic_variability_pairs,
    file_sha256,
    grouped_mean_interval,
    load_generation_feature_frame,
    prepare_human_control_evaluation,
)
from rankcloak.revision_v3_dedup import (  # noqa: E402
    build_strict_deduplicated_corpus,
)


DEFAULT_DETECTOR_CORPUS = (
    PROJECT_ROOT
    / "results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl"
)
DEFAULT_TRIAL_CORPUS = (
    PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/trials.csv"
)
DEFAULT_RECORD_ROOT = PROJECT_ROOT / "results/revision_v1/primary_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/revision_v3/analysis.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, lineterminator="\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def command_output(command: Sequence[str]) -> Mapping[str, object]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": list(command),
            "exit_code": int(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": "{}: {}".format(type(exc).__name__, exc),
        }


def record_paths() -> list[Path]:
    paths = sorted(DEFAULT_RECORD_ROOT.glob("*/records.jsonl"))
    if len(paths) != 3:
        raise SystemExit("Expected exactly three authoritative model record files")
    return paths


def identity(path: Path, row_count: int | None = None) -> Mapping[str, object]:
    result = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if row_count is not None:
        result["row_count"] = int(row_count)
    return result


def build_environment() -> Mapping[str, object]:
    llama_cpp = None
    try:
        import llama_cpp as imported_llama_cpp

        llama_cpp = str(imported_llama_cpp.__version__)
    except Exception:
        llama_cpp = None
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    model_config = json.loads(
        (PROJECT_ROOT / "configs/revision_v1/models.json").read_text(encoding="utf-8")
    )
    models = []
    for specification in model_config["models"]:
        path = PROJECT_ROOT / str(specification["relative_path"])
        models.append(
            {
                "model_id": specification["model_id"],
                "repo_id": specification["repo_id"],
                "revision": specification["revision"],
                "quantization": specification["quantization"],
                "expected_path": str(path),
                "expected_size_bytes": specification["artifact_size_bytes"],
                "expected_sha256": specification["artifact_sha256"],
                "available": path.is_file(),
            }
        )
    return {
        "schema_version": "rankcloak-revision-v3-environment-v1",
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "matplotlib": matplotlib.__version__,
            "llama_cpp_python": llama_cpp,
        },
        "torch_cuda": {
            "available": bool(torch.cuda.is_available()),
            "runtime_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                {
                    "runtime_index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "nvidia_smi": gpu,
        "generation_models": models,
        "generation_backend_available": llama_cpp is not None,
    }


def write_design_docs(output: Path) -> None:
    atomic_text(
        output / "experiment_design.md",
        """# Revision V3 computational experiment design

The predeclared design is tracked in revision_docs/REVISION_V3_COMPUTATIONAL_PLAN.md and machine-readable configurations under configs/revision_v3/. This result namespace never modifies V2 source data.

The detector corpus is normalized with Unicode NFKC, case folding, and whitespace collapse before SHA-256 exact matching. Complete matched pairs implicated by an exact duplicate are removed. Character-boundary TF-IDF cosine similarity over 3--5 character n-grams defines near duplicates at the predeclared threshold 0.95. Payload groups linked by any near-duplicate edge form an immutable connected component. These components are assigned to train, validation, and test before feature extraction or detector fitting.

Component assignment targeted 60/20/20 train/validation/test allocation; indivisible near-duplicate components and factor balancing produced the exact realized counts recorded in the deduplication summary. Leave-one-model-family-out tests use non-target families from train and validation components and the target family only from test components. Low-FPR thresholds are selected from validation labels and then frozen. The 0.1% operating point is reported only when both validation and test contain at least 1,000 negative observations.

The human-authored secondary control uses the pinned Databricks Dolly 15k v1.0 source, automatically screened by the repository pipeline, deduplicated, checked against the detector corpus, and matched by available prompt topic and text length. It is a computational control, not a human evaluation. RankCloak-versus-human discrimination can include generic machine-versus-human cues.

The model-aware detector uses only saved token-log-probability summaries available for both RankCloak and matched ordinary controls. Rank-only fields are excluded from classifier inputs because corresponding ordinary-control ranks were not stored. Topic variability is a paired comparison of the fixed-codec segmented single-topic and multi-topic conditions; it is reported separately from recovery mode.

The bounded entropy-gate matrix is predeclared in configs/revision_v3/entropy_gate.json. It compares ungated, median-clean-development-entropy, and 75th-percentile gates under both fixed-payload and fixed-token-budget estimands. The bounded quantization design is predeclared in configs/revision_v3/quantization.json and holds the Qwen2.5 model revision, embedded tokenizer, prompts, payloads, codecs, and seeds fixed across Q4_K_M and Q8_0. Both model-backed studies remain unavailable because their exact pinned weights and the required CUDA llama.cpp backend are absent; no historical cross-family result is relabeled as a quantization comparison.

The recovery-mode comparison reuses only the 144-trial V1 robustness sample after verifying its source checksum. Saved token IDs, greedy lead-in regeneration, and visible-text detokenization/retokenization are separate outcomes and are not extrapolated to the full 6,480-trial corpus.
""",
    )
    atomic_text(
        output / "data_dictionary.md",
        """# Revision V3 data dictionary

## Corpus and partitions

- `row_id`: immutable detector observation identifier.
- `pair_id`: matched RankCloak/ordinary-control pair identifier.
- `payload_group_id`: payload instance used as the original grouping unit.
- `raw_text_sha256`: SHA-256 of the original UTF-8 visible text.
- `normalized_text_sha256`: SHA-256 after NFKC, case folding, whitespace collapse, and stripping.
- `dedup_cluster_id` / `split_group_id`: connected component of payload groups joined by exact or thresholded near-duplicate links; the indivisible split/bootstrap unit.
- `partition`: `train`, `validation`, or `test`; assigned only at component level.
- `label`: 1 for RankCloak and 0 for a control.
- `model_id`, `codec_id`, `payload_class`, `prompt_template_id`: preserved experimental factors.

## Predictions and detector metrics

- `evaluation_role`: validation or test row in a detector prediction ledger; training rows are not scored into this ledger.
- `evaluation_id`: `matched` or `leave_one_model__<model_id>`.
- `score`: detector positive-class score; larger values indicate RankCloak.
- `roc_auc`: tie-aware empirical ROC area.
- `partial_auc_fpr_0_01`: exact empirical false-positive-budget step area from FPR 0 to 0.01, divided by 0.01.
- `threshold_at_fpr_*`: threshold selected exclusively on validation scores and labels.
- `tpr_at_fpr_*` / `fpr_at_threshold_*`: test rates at the frozen validation threshold.
- `false_positives_at_fpr_*`: exact count of test controls above the frozen threshold.
- `validation_negative_count` / `test_negative_count`: denominators governing low-FPR resolution.
- `*_ci_low_95` / `*_ci_high_95`: 2,000-resample dedup-cluster bootstrap interval unless the source table states Wilson.
- `fpr_*_available` / `fpr_*_unavailable_reason`: empirical-resolution status and fail-closed explanation.

## Human-authored secondary controls

- `candidate_id` / `source_record_id`: stable Dolly candidate and pinned source-record identifiers; committed ledgers omit licensed response text.
- `message_text_sha256` / `canonical_text_sha256`: raw-display and canonical-response hashes.
- `relative_word_difference` / `matching_cost`: deterministic length/topic matching diagnostics.
- `human_fpr_at_llm_validation_threshold_0_01`: human-control false-positive rate at a threshold selected without human labels.

## Cover variability and recovery

- `normalized_character_edit_distance`: Levenshtein distance divided by the longer character length.
- `token_jaccard_similarity`: casefolded word-type intersection divided by union.
- `replay_mode`: saved IDs, greedy lead-in regeneration, or detokenized-text retokenization.
- `success_outcome_rows` / `observed_outcome_rows`: recovery numerator and denominator in the bounded historical robustness sample.
- `recovery_rate`: their ratio; its `ci_low` and `ci_high` are source-trial Wilson limits in the recovery table.

## Availability and provenance

- `unavailable_reason`: explicit reason an estimand or experiment could not be computed.
- `sha256`: content checksum over exact artifact bytes.
- `row_count`: data-row count excluding the CSV header; blank for non-tabular artifacts.
""",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    started = utc_now()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            "Refusing to overwrite non-empty revision-V3 output directory: {}".format(
                output
            )
        )
    for directory in (
        "source_tables",
        "manuscript_tables",
        "figures",
        "metrics",
        "detector_predictions",
        "deduplication",
        "provenance",
        "logs",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    detector_source = pd.read_json(DEFAULT_DETECTOR_CORPUS, lines=True)

    print("[prepare] strict deduplication and component split", flush=True)
    dedup = build_strict_deduplicated_corpus(
        detector_source,
        threshold=float(config["deduplication"]["near_duplicate_threshold"]),
        seed=int(config["seed"]),
        fractions={
            "train": float(config["partitions"]["train_fraction"]),
            "validation": float(config["partitions"]["validation_fraction"]),
            "test": float(config["partitions"]["test_fraction"]),
        },
    )
    row_manifest_columns = [
        "row_id",
        "pair_id",
        "payload_group_id",
        "normalized_text_sha256",
        "dedup_cluster_id",
        "partition",
        "label",
        "model_id",
        "codec_id",
        "payload_class",
        "prompt_template_id",
    ]
    atomic_csv(
        output / "deduplication/deduplicated_row_manifest.csv",
        dedup.frame[row_manifest_columns].sort_values("row_id"),
    )
    removed_columns = [
        column
        for column in dedup.removed_rows.columns
        if column not in {"text", "normalized_text"}
    ]
    atomic_csv(
        output / "deduplication/removed_observations.csv",
        dedup.removed_rows[removed_columns],
    )
    atomic_csv(output / "deduplication/exact_duplicate_groups.csv", dedup.exact_groups)
    atomic_csv(output / "deduplication/near_duplicate_pairs.csv", dedup.near_pairs)
    atomic_csv(output / "deduplication/cluster_manifest.csv", dedup.cluster_manifest)
    atomic_csv(output / "deduplication/partition_manifest.csv", dedup.partition_manifest)
    atomic_csv(
        output / "deduplication/factor_counts.csv",
        pd.DataFrame(dedup.summary["factor_counts"]),
    )
    dedup_summary = dict(dedup.summary)
    dedup_summary.pop("factor_counts", None)
    atomic_json(output / "deduplication/summary.json", dedup_summary)
    atomic_json(output / "deduplication/leakage_audit.json", dedup.leakage_audit)

    paths = record_paths()
    print("[prepare] saved generation-trace features", flush=True)
    features, rank_pressure = load_generation_feature_frame(dedup.frame, paths)
    atomic_csv(
        output / "provenance/generation_surprisal_features.csv",
        features.sort_values("row_id"),
    )
    atomic_csv(
        output / "source_tables/rank_pressure_descriptive.csv",
        rank_pressure.sort_values("source_trial_id"),
    )

    print("[prepare] paired topic-conditioned cover variability", flush=True)
    topic_pairs = build_topic_variability_pairs(paths)
    topic_metrics = topic_pairs.drop(columns=["single_excerpt", "multi_excerpt"])
    examples = topic_pairs.sort_values("deterministic_example_order").head(3).copy()
    atomic_csv(output / "source_tables/topic_variability_pairs.csv", topic_metrics)
    atomic_csv(output / "source_tables/topic_variability_examples.csv", examples)
    topic_summary = {
        "schema_version": "rankcloak-revision-v3-topic-variability-summary-v1",
        "pair_count": int(len(topic_pairs)),
        "model_count": int(topic_pairs["model_id"].nunique()),
        "payload_count": int(topic_pairs["payload_name"].nunique()),
        "payload_class_count": int(topic_pairs["payload_class"].nunique()),
        "exact_output_uniqueness": grouped_mean_interval(
            topic_pairs,
            "exact_outputs_unique",
            "payload_name",
            seed=int(config["seed"]),
        ),
        "normalized_output_uniqueness": grouped_mean_interval(
            topic_pairs,
            "normalized_outputs_unique",
            "payload_name",
            seed=int(config["seed"]) + 1,
        ),
        "normalized_character_edit_distance": grouped_mean_interval(
            topic_pairs,
            "normalized_character_edit_distance",
            "payload_name",
            seed=int(config["seed"]) + 2,
        ),
        "token_jaccard_similarity": grouped_mean_interval(
            topic_pairs,
            "token_jaccard_similarity",
            "payload_name",
            seed=int(config["seed"]) + 3,
        ),
        "recovery_is_separate_from_variability": True,
        "single_saved_id_exact_recovery_count": int(
            topic_pairs["single_saved_id_exact_recovery"].sum()
        ),
        "multi_saved_id_exact_recovery_count": int(
            topic_pairs["multi_saved_id_exact_recovery"].sum()
        ),
        "single_visible_text_available_count": int(
            topic_pairs["single_visible_text_exact_recovery"].notna().sum()
        ),
        "multi_visible_text_available_count": int(
            topic_pairs["multi_visible_text_exact_recovery"].notna().sum()
        ),
    }
    atomic_json(output / "metrics/topic_variability_summary.json", topic_summary)

    print("[prepare] human-authored secondary controls", flush=True)
    candidate_sha256 = file_sha256(args.human_candidates)
    if candidate_sha256 != "23913150fd9d7d46e2cab1c382eebba09454090bc90145187f7394b0a95619c9":
        raise SystemExit("Human candidate manifest SHA-256 differs from the audited import")
    import_audit_path = args.human_candidates.parent / "import_audit.json"
    if not import_audit_path.is_file():
        raise SystemExit("Human candidate import audit is missing")
    import_audit = json.loads(import_audit_path.read_text(encoding="utf-8"))
    expected_import = {
        "candidate_manifest_sha256": candidate_sha256,
        "candidate_record_count": 6606,
        "source_file_sha256": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        "dataset_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "network_access_performed_by_pipeline": False,
    }
    for field, expected_value in expected_import.items():
        if import_audit.get(field) != expected_value:
            raise SystemExit("Human import audit mismatch for {}".format(field))
    if int(import_audit.get("counts", {}).get("eligible_for_manual_review", -1)) != 540:
        raise SystemExit("Human import audit has an unexpected screened-candidate count")
    human = prepare_human_control_evaluation(
        args.human_candidates,
        dedup.frame,
        threshold=float(config["deduplication"]["near_duplicate_threshold"]),
        maximum_relative_word_difference=float(
            config["human_controls"]["maximum_relative_word_difference"]
        ),
    )
    atomic_csv(
        output / "provenance/human_control_selection_manifest.csv",
        human["selection_manifest"],
    )
    atomic_json(
        output / "provenance/human_control_import_audit.json", import_audit
    )
    atomic_csv(output / "source_tables/human_control_length_matches.csv", human["matches"])
    atomic_csv(
        output / "deduplication/human_exact_duplicate_rows.csv",
        human["exact_duplicate_rows"],
    )
    atomic_csv(
        output / "deduplication/human_near_duplicate_pairs.csv",
        human["near_duplicate_pairs"],
    )
    atomic_csv(
        output / "deduplication/human_primary_cross_corpus_near_pairs.csv",
        human["cross_corpus_near_pairs"],
    )
    atomic_json(output / "deduplication/human_control_audit.json", human["summary"])

    environment = build_environment()
    atomic_json(output / "environment.json", environment)
    source_inputs = {
        "schema_version": "rankcloak-revision-v3-source-inputs-v1",
        "authoritative_trials": identity(DEFAULT_TRIAL_CORPUS, row_count=6480),
        "authoritative_detector_corpus": identity(
            DEFAULT_DETECTOR_CORPUS, row_count=len(detector_source)
        ),
        "authoritative_record_files": [identity(path) for path in paths],
        "analysis_config": identity(args.config),
        "detector_config": identity(PROJECT_ROOT / "configs/revision_v3/detectors.json"),
        "entropy_config": identity(PROJECT_ROOT / "configs/revision_v3/entropy_gate.json"),
        "quantization_config": identity(PROJECT_ROOT / "configs/revision_v3/quantization.json"),
        "generation_requirements": identity(PROJECT_ROOT / "configs/revision_v3/generation_requirements.json"),
        "human_candidate_manifest": identity(args.human_candidates, row_count=6606),
        "human_import_audit": identity(import_audit_path),
        "human_source": {
            "dataset": "databricks-dolly-15k",
            "version": "1.0",
            "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
            "source_file_sha256": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
            "license": "CC-BY-SA-3.0",
            "selected_text_redistributed": False,
        },
    }
    atomic_json(output / "provenance/source_inputs.json", source_inputs)
    write_design_docs(output)

    git_head = command_output(["git", "rev-parse", "HEAD"])
    status_rows = [
        {"experiment": "A", "name": "strict_deduplication", "status": "completed"},
        {"experiment": "B", "name": "human_controls", "status": "prepared_detector_pending"},
        {"experiment": "C", "name": "unseen_model_family", "status": "prepared_detector_pending"},
        {"experiment": "D", "name": "matched_quantization", "status": "blocked_missing_models_backend"},
        {"experiment": "E", "name": "adaptive_detectors", "status": "prepared_detector_pending"},
        {"experiment": "F", "name": "low_fpr", "status": "prepared_detector_pending"},
        {"experiment": "G", "name": "entropy_gate", "status": "implemented_generation_blocked"},
        {"experiment": "H", "name": "topic_variability", "status": "completed"},
    ]
    atomic_csv(output / "provenance/experiment_status.csv", pd.DataFrame(status_rows))
    completed = utc_now()
    manifest = {
        "schema_version": "rankcloak-revision-v3-run-manifest-v1",
        "status": "prepared_detector_runs_pending",
        "git_commit_at_start": git_head["stdout"],
        "configuration_files": [
            "configs/revision_v3/analysis.json",
            "configs/revision_v3/detectors.json",
            "configs/revision_v3/entropy_gate.json",
            "configs/revision_v3/quantization.json",
            "configs/revision_v3/generation_requirements.json",
        ],
        "random_seeds": [int(config["seed"])],
        "commands": [
            ".venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import",
            ".venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import/human_control_candidates.jsonl"
        ],
        "start_time": started,
        "preparation_completion_time": completed,
        "trial_counts": {
            "new_model_generation_trials": 0,
            "reused_rankcloak_trials": 6480,
            "reused_matched_ordinary_controls": 7920,
            "detector_observations_original": int(len(detector_source)),
            "detector_observations_after_exact_deduplication": int(len(dedup.frame)),
            "topic_condition_pairs_reused": int(len(topic_pairs)),
            "human_controls_selected": int(human["summary"]["selected_human_controls"]),
        },
        "experiment_status": status_rows,
        "blocked_generation": {
            "entropy_gate": {
                "reason": "all three exact pinned Q4_K_M GGUF files and a GPU-enabled llama-cpp-python backend are absent",
                "planned_analysis_trials": 720,
                "planned_new_generation_trials": 720,
                "planned_reused_baseline_trials": 0,
                "planned_calibration_traces": 18,
            },
            "matched_quantization": {
                "reason": "the exact Qwen Q4_K_M and matched Q8_0 artifacts plus a GPU-enabled llama-cpp-python backend are absent; acquisition requires a large download",
                "planned_new_trials": 1920,
                "planned_reused_trials": 1920,
            },
        },
        "source_inputs_sha256": canonical_sha256(source_inputs),
    }
    atomic_json(output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_dir": str(output),
                "deduplicated_rows": len(dedup.frame),
                "near_duplicate_pairs": len(dedup.near_pairs),
                "topic_pairs": len(topic_pairs),
                "human_controls": human["summary"]["selected_human_controls"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
