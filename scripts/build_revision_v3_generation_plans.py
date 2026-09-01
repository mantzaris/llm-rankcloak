#!/usr/bin/env python3
"""Build exact dry-run ledgers and preflight blocked V3 model generation.

This command never downloads weights, imports a generation backend, or launches
model inference.  It resolves every planned analysis row against immutable V2
baselines and records which rows require new generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_revision_v3 import atomic_csv, atomic_json  # noqa: E402
from rankcloak.revision_v3_analysis import file_sha256  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DETECTOR_CORPUS = (
    PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl"
)
TRIAL_CORPUS = PROJECT_ROOT / "results/revision_v1/analysis_inputs/primary_v2/trials.csv"
PROMPT_CONFIG = PROJECT_ROOT / "configs/revision_v1/prompts.json"
ENTROPY_CONFIG = PROJECT_ROOT / "configs/revision_v3/entropy_gate.json"
QUANTIZATION_CONFIG = PROJECT_ROOT / "configs/revision_v3/quantization.json"
REQUIREMENTS_CONFIG = PROJECT_ROOT / "configs/revision_v3/generation_requirements.json"
PROTOCOL_AMENDMENT = PROJECT_ROOT / "revision_docs/REVISION_V3_GENERATION_PROTOCOL_AMENDMENT.md"
QWEN_Q4_ROOT = PROJECT_ROOT / "results/revision_v1/primary_v2/qwen2_5_7b_instruct_q4_k_m"
QWEN_Q4_RECORDS = QWEN_Q4_ROOT / "records.jsonl"
QWEN_Q4_PLAN = QWEN_Q4_ROOT / "plan.jsonl"


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["source_line_number"] = line_number
                rows.append(row)
    return pd.DataFrame(rows)


def stable_seed(base_seed: int, *parts: object) -> int:
    message = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "big") & (
        2**63 - 1
    )


def stable_id(prefix: str, *parts: object) -> str:
    message = "|".join(str(part) for part in parts)
    return "{}__{}".format(
        prefix, hashlib.sha256(message.encode("utf-8")).hexdigest()[:24]
    )


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_lookup() -> Mapping[str, Mapping[str, str]]:
    document = load_json(PROMPT_CONFIG)
    return {
        str(template["prompt_id"]): {
            "category": str(category["category_id"]),
            "text": str(template["text"]),
        }
        for category in document["categories"]
        for template in category["templates"]
    }


def source_row(
    detector: pd.DataFrame,
    *,
    model_id: str,
    prompt_id: str | None,
    payload_class: str,
    payload_index: int,
    codec_id: str,
    label: int,
) -> Mapping[str, object]:
    suffix = "_{:03d}".format(payload_index)
    selected = detector.loc[
        detector["model_id"].eq(model_id)
        & detector["payload_class"].eq(payload_class)
        & detector["payload_group_id"].str.endswith(suffix)
        & detector["codec_id"].eq(codec_id)
        & detector["label"].eq(label)
    ]
    if prompt_id is not None:
        selected = selected.loc[selected["prompt_template_id"].eq(prompt_id)]
    if len(selected) != 1:
        raise SystemExit(
            "Expected one reusable source row, found {} for {}".format(
                len(selected),
                (model_id, prompt_id, payload_class, payload_index, codec_id, label),
            )
        )
    return selected.iloc[0].to_dict()


def entropy_plan(detector: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    models = list(map(str, config["models"]))
    templates = list(map(str, config["templates"]))
    indices = list(map(int, config["payload_indices"]))
    if len(templates) != 2 or len(indices) != 2:
        raise SystemExit("Entropy plan requires exactly two template/payload-index pairs")
    payload_classes = sorted(detector["payload_class"].astype(str).unique())
    hex_classes = sorted(
        detector.loc[detector["representation_name"].eq("hex_nibble"), "payload_class"]
        .astype(str)
        .unique()
    )
    cells = []
    for payload_class in payload_classes:
        cells.extend(
            [
                (payload_class, "direct_subword", "direct_subword_calgacus"),
                (payload_class, "ascii_b16", "nonseg_ascii_b16"),
            ]
        )
        if payload_class in hex_classes:
            cells.append((payload_class, "hex_nibble", "nonseg_hex_nibble_b16"))
    if len(payload_classes) != 8 or len(hex_classes) != 4 or len(cells) != 20:
        raise SystemExit("Authoritative entropy cell coverage differs from 8/4/20")

    rows = []
    for model_id in models:
        for payload_class, representation, source_codec in cells:
            for prompt_id, payload_index in zip(templates, indices):
                rank_source = source_row(
                    detector,
                    model_id=model_id,
                    prompt_id=None,
                    payload_class=payload_class,
                    payload_index=payload_index,
                    codec_id=source_codec,
                    label=1,
                )
                control_source = source_row(
                    detector,
                    model_id=model_id,
                    prompt_id=None,
                    payload_class=payload_class,
                    payload_index=payload_index,
                    codec_id=source_codec,
                    label=0,
                )
                if rank_source["pair_id"] != control_source["pair_id"]:
                    raise SystemExit("Reusable entropy baseline is not a matched pair")
                exact_prompt_compatible = bool(
                    rank_source["prompt_template_id"] == prompt_id
                    and control_source["prompt_template_id"] == prompt_id
                )
                if exact_prompt_compatible:
                    raise SystemExit("Dry-run expectation changed: an exact V2 entropy baseline exists")
                experimental_cell_id = stable_id(
                    "entropy_cell",
                    model_id,
                    payload_class,
                    representation,
                    prompt_id,
                    payload_index,
                )
                rankcloak_seed = stable_seed(
                    int(config["seed"]), experimental_cell_id, "rankcloak_ordinary_skip"
                )
                control_seed = stable_seed(
                    int(config["seed"]), experimental_cell_id, "matched_ordinary_control"
                )
                for gate_level in ("ungated", "moderate", "strict"):
                    pairing_id = stable_id(
                        "entropy_pair",
                        model_id,
                        payload_class,
                        representation,
                        prompt_id,
                        payload_index,
                        gate_level,
                    )
                    for label, population, baseline in (
                        (1, "rankcloak", rank_source),
                        (0, "ordinary_control", control_source),
                    ):
                        reused = gate_level == "ungated" and exact_prompt_compatible
                        plan_id = stable_id("entropy_trial", pairing_id, population)
                        if population == "rankcloak":
                            if gate_level == "ungated":
                                generation_method = (
                                    "reused_ordinary_rankcloak_gate_disabled"
                                    if reused
                                    else "new_ordinary_rankcloak_gate_disabled"
                                )
                            else:
                                generation_method = "entropy_gated_rankcloak"
                        else:
                            generation_method = (
                                "reused_length_matched_ordinary_control"
                                if reused
                                else "new_length_matched_ordinary_control"
                            )
                        rows.append(
                            {
                                "plan_id": plan_id,
                                "experimental_cell_id": experimental_cell_id,
                                "pairing_unit_id": pairing_id,
                                "experiment": "entropy_gated_rankcloak",
                                "model_id": model_id,
                                "quantization": "Q4_K_M",
                                "payload_class": payload_class,
                                "representation_name": representation,
                                "source_codec_id": source_codec,
                                "prompt_template_id": prompt_id,
                                "payload_index": payload_index,
                                "payload_name": rank_source["payload_group_id"],
                                "gate_level": gate_level,
                                "threshold_source": {
                                    "ungated": "not_applicable",
                                    "moderate": "model_clean_development_entropy_quantile_0.50",
                                    "strict": "model_clean_development_entropy_quantile_0.75",
                                }[gate_level],
                                "label": label,
                                "population": population,
                                "generation_method": generation_method,
                                "trial_status": "reused_exact_v2_baseline" if reused else "planned_new_generation",
                                "generation_required": not reused,
                                "random_seed": (
                                    rankcloak_seed
                                    if population == "rankcloak"
                                    else control_seed
                                ),
                                "paired_rankcloak_seed": rankcloak_seed,
                                "paired_ordinary_control_seed": control_seed,
                                "seed_shared_across_gate_levels": True,
                                "random_seed_consumed": bool(
                                    population == "ordinary_control"
                                    or gate_level != "ungated"
                                ),
                                "ordinary_sampling_temperature": config[
                                    "ineligible_token_policy"
                                ]["temperature"],
                                "ordinary_sampling_top_p": config[
                                    "ineligible_token_policy"
                                ]["top_p"],
                                "ordinary_sampler": config[
                                    "ineligible_token_policy"
                                ]["sampler"],
                                "reference_same_payload_row_id": baseline["row_id"],
                                "reference_same_payload_pair_id": baseline["pair_id"],
                                "reference_same_payload_trial_id": baseline["source_trial_id"],
                                "reference_prompt_template_id": baseline["prompt_template_id"],
                                "exact_v2_baseline_compatible": exact_prompt_compatible,
                                "unavailable_reuse_reason": (
                                    None if reused else "v2_prompt_template_differs_or_gate_is_enabled"
                                ),
                                "source_row_id": baseline["row_id"] if reused else None,
                                "source_trial_id": baseline["source_trial_id"] if reused else None,
                                "fixed_payload_maximum_length_rule": "min_context_allowance_or_6x_ungated_forced_length",
                                "fixed_token_budget_rule": "paired_ungated_payload_span_token_count",
                                "control_length_rule": (
                                    "not_applicable_rankcloak"
                                    if population == "rankcloak"
                                    else "match_realized_rankcloak_full_token_count"
                                ),
                                "control_prefix_contract": (
                                    "not_applicable_rankcloak"
                                    if population == "rankcloak"
                                    else "same_prompt_seed_and_sampler_across_gate_levels"
                                ),
                            }
                        )
    frame = pd.DataFrame(rows).sort_values("plan_id", kind="stable").reset_index(drop=True)
    if len(frame) != 720 or int(frame["generation_required"].sum()) != 720:
        raise SystemExit("Entropy dry-run counts differ from 720 total / 720 new")
    if frame.loc[~frame["generation_required"], "source_row_id"].isna().any():
        raise SystemExit("A reused entropy baseline lacks a source row")
    seed_counts = frame.groupby(["experimental_cell_id", "population"])[
        "random_seed"
    ].nunique()
    if not seed_counts.eq(1).all() or frame["experimental_cell_id"].nunique() != 120:
        raise SystemExit("Entropy rows do not share one seed per cell and population")
    return frame


def calibration_plan(config: Mapping[str, object]) -> pd.DataFrame:
    prompts = prompt_lookup()
    calibration = config["calibration"]
    rows = []
    for model_id in map(str, config["models"]):
        for prompt_id in map(str, config["calibration_prompt_ids"]):
            if prompt_id not in prompts:
                raise SystemExit("Unknown calibration prompt {}".format(prompt_id))
            plan_id = stable_id("entropy_calibration", model_id, prompt_id)
            rows.append(
                {
                    "plan_id": plan_id,
                    "model_id": model_id,
                    "quantization": "Q4_K_M",
                    "prompt_category": prompts[prompt_id]["category"],
                    "prompt_template_id": prompt_id,
                    "rendered_prompt_sha256": hashlib.sha256(
                        prompts[prompt_id]["text"].encode("utf-8")
                    ).hexdigest(),
                    "generation_method": "ordinary_top_p_clean_development_trace",
                    "random_seed": stable_seed(int(config["seed"]), plan_id),
                    "target_token_count": int(calibration["target_token_count_per_trace"]),
                    "temperature": float(calibration["temperature"]),
                    "top_p": float(calibration["top_p"]),
                    "sampler": calibration["sampler"],
                    "detector_outcomes_used": False,
                    "status": "planned_new_calibration_trace",
                }
            )
    frame = pd.DataFrame(rows).sort_values("plan_id", kind="stable").reset_index(drop=True)
    if (
        len(frame) != 18
        or frame["prompt_category"].nunique() != 6
        or set(frame["generation_method"])
        != {"ordinary_top_p_clean_development_trace"}
    ):
        raise SystemExit("Entropy calibration plan differs from 3 models x 6 categories")
    return frame


def quantization_plan(
    detector: pd.DataFrame,
    trial_splits: Mapping[str, str],
    config: Mapping[str, object],
    raw_records: Sequence[Mapping[str, object]],
    raw_tasks: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    model_id = "qwen2_5_7b_instruct_q4_k_m"
    codec_lookup = {"ascii_b8": "nonseg_ascii_b8", "ascii_b16": "nonseg_ascii_b16"}
    rows = []
    source = detector.loc[
        detector["model_id"].eq(model_id)
        & detector["representation_name"].isin(config["codecs"])
    ].copy()
    expected_source_rows = int(config["planned_reused_rankcloak_trials"]) + int(
        config["planned_reused_ordinary_control_trials"]
    )
    if len(source) != expected_source_rows:
        raise SystemExit("Q4 quantization source rows differ from {}".format(expected_source_rows))
    rank_records = {
        str(record["trial_id"]): record
        for record in raw_records
        if record.get("record_type") == "rankcloak_trial"
    }
    control_records = {}
    for record in raw_records:
        if (
            record.get("record_type") == "ordinary_control"
            and record.get("control_view") == "full_message"
        ):
            trial_id = str(record["source_trial_id"])
            if trial_id in control_records:
                raise SystemExit("Multiple full-message Q4 controls for {}".format(trial_id))
            control_records[trial_id] = record
    task_records = {
        str(record["trial_id"]): record
        for record in raw_tasks
        if record.get("work_kind") == "rankcloak"
    }
    prompts = prompt_lookup()
    for record in source.to_dict("records"):
        representation = str(record["representation_name"])
        if str(record["codec_id"]) != codec_lookup[representation]:
            raise SystemExit("Unexpected Q4 source codec mapping")
        payload_name = str(record["payload_group_id"])
        payload_split = trial_splits.get(payload_name)
        if payload_split not in {"train", "validation", "test"}:
            raise SystemExit("Missing frozen payload split for {}".format(payload_name))
        population = "rankcloak" if int(record["label"]) == 1 else "ordinary_control"
        source_trial_id = str(record["source_trial_id"])
        if (
            source_trial_id not in rank_records
            or source_trial_id not in control_records
            or source_trial_id not in task_records
        ):
            raise SystemExit("Missing raw Q4 lineage for {}".format(source_trial_id))
        rank_record = rank_records[source_trial_id]
        control_record = control_records[source_trial_id]
        task_record = task_records[source_trial_id]
        generation = control_record["generation"]
        prompt_id = str(record["prompt_template_id"])
        if prompt_id not in prompts:
            raise SystemExit("Unknown historical prompt {}".format(prompt_id))
        if not (
            rank_record.get("prompt_id") == prompt_id
            and task_record.get("prompt_id") == prompt_id
            and task_record.get("protocol_variant") == record["codec_id"]
            and rank_record.get("protocol_variant") == record["codec_id"]
            and rank_record.get("representation", {}).get("name") == representation
            and rank_record.get("payload_name") == payload_name
            and control_record.get("payload_name") == payload_name
        ):
            raise SystemExit("Historical Q4 non-quantization identity mismatch")
        if not (
            float(generation["temperature"]) == 0.8
            and float(generation["top_p"]) == 0.95
            and generation["sampler"]
            == "numpy_pcg64_serial_top_p_v1_token_id_tiebreak"
            and int(generation["target_token_count"])
            == int(rank_record["full_token_count"])
        ):
            raise SystemExit("Historical Q4 control sampling contract differs")
        historical_seed = int(generation["sampling_seed"])
        expected_ranks = rank_record["representation"]["expected_ranks"]
        contract = {
            "upstream_base_model_revision": config["upstream_base_model_revision"],
            "quantization_package_revision": config["model_revision"],
            "rendered_prompt_sha256": hashlib.sha256(
                prompts[prompt_id]["text"].encode("utf-8")
            ).hexdigest(),
            "historical_prompt_context_token_ids_sha256": canonical_sha256(
                generation["context_token_ids"]
            ),
            "payload_name": payload_name,
            "payload_text_sha256": rank_record["payload_text_sha256"],
            "representation_sha256": canonical_sha256(rank_record["representation"]),
            "expected_ranks_sha256": canonical_sha256(expected_ranks),
            "protocol_variant": rank_record["protocol_variant"],
            "token_filter": rank_record["token_filter"],
            "allowed_token_mask_sha256": canonical_sha256(
                rank_record.get("allowed_token_mask")
            ),
            "tail_policy": rank_record["tail_policy"],
            "leadin_token_count": rank_record["leadin_token_count"],
            "target_token_count": int(generation["target_token_count"]),
            "historical_control_sampling_seed": historical_seed,
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "sampler": generation["sampler"],
            "excluded_special_token_ids": generation["excluded_special_token_ids"],
            "backend": config["required_backend"],
        }
        contract_sha = canonical_sha256(contract)
        pairing_base = stable_id(
            "quantization_pair",
            record["pair_id"],
            population,
        )
        for quantization, target_model, reused in (
            ("Q4_K_M", model_id, True),
            ("Q8_0", "qwen2_5_7b_instruct_q8_0", False),
        ):
            plan_id = stable_id("quantization_trial", pairing_base, quantization)
            rows.append(
                {
                    "plan_id": plan_id,
                    "pairing_unit_id": pairing_base,
                    "experiment": "matched_quantization_generalization",
                    "model_id": target_model,
                    "base_model": "Qwen2.5-7B-Instruct",
                    "model_revision": config["upstream_base_model_revision"],
                    "quantization_package_revision": config["model_revision"],
                    "quantization": quantization,
                    "embedded_tokenizer_fixed": True,
                    "payload_name": payload_name,
                    "payload_class": record["payload_class"],
                    "payload_split": payload_split,
                    "representation_name": representation,
                    "codec_id": record["codec_id"],
                    "prompt_template_id": record["prompt_template_id"],
                    "label": int(record["label"]),
                    "population": population,
                    "trial_status": "reused_exact_v2_q4" if reused else "planned_new_q8_generation",
                    "generation_required": not reused,
                    "random_seed": historical_seed,
                    "historical_control_sampling_seed": historical_seed,
                    "sampling_seed_applied": population == "ordinary_control",
                    "temperature": float(generation["temperature"]),
                    "top_p": float(generation["top_p"]),
                    "sampler": generation["sampler"],
                    "target_token_count": int(generation["target_token_count"]),
                    "rendered_prompt_sha256": contract["rendered_prompt_sha256"],
                    "historical_prompt_context_token_ids_sha256": contract[
                        "historical_prompt_context_token_ids_sha256"
                    ],
                    "payload_text_sha256": rank_record["payload_text_sha256"],
                    "expected_ranks_sha256": contract["expected_ranks_sha256"],
                    "allowed_token_mask_sha256": contract[
                        "allowed_token_mask_sha256"
                    ],
                    "non_quantization_contract_sha256": contract_sha,
                    "historical_task_sha256": canonical_sha256(task_record),
                    "historical_rank_record_sha256": canonical_sha256(rank_record),
                    "historical_control_record_sha256": canonical_sha256(control_record),
                    "reference_q4_control_id": control_record["control_id"],
                    "reference_q4_row_id": record["row_id"],
                    "reference_q4_pair_id": record["pair_id"],
                    "reference_q4_trial_id": record["source_trial_id"],
                    "source_row_id": record["row_id"] if reused else None,
                    "source_trial_id": record["source_trial_id"] if reused else None,
                }
            )
    frame = pd.DataFrame(rows).sort_values("plan_id", kind="stable").reset_index(drop=True)
    if len(frame) != 3840 or int(frame["generation_required"].sum()) != 1920:
        raise SystemExit("Quantization dry-run counts differ from 3,840 total / 1,920 new")
    unique_payload_counts = (
        frame.groupby(["quantization", "payload_class", "payload_split"])["payload_name"]
        .nunique()
        .unstack(fill_value=0)
    )
    if not (
        (unique_payload_counts["train"] == 36).all()
        and (unique_payload_counts["validation"] == 12).all()
        and (unique_payload_counts["test"] == 12).all()
    ):
        raise SystemExit("Quantization payload splits differ from 36/12/12 per class")
    paired_contracts = frame.groupby("pairing_unit_id").agg(
        quantization_count=("quantization", "nunique"),
        seed_count=("historical_control_sampling_seed", "nunique"),
        contract_count=("non_quantization_contract_sha256", "nunique"),
    )
    if not (
        paired_contracts["quantization_count"].eq(2).all()
        and paired_contracts["seed_count"].eq(1).all()
        and paired_contracts["contract_count"].eq(1).all()
        and frame["random_seed"].eq(frame["historical_control_sampling_seed"]).all()
    ):
        raise SystemExit("Q4/Q8 pairs do not preserve seed and non-quantization contract")
    return frame


def preflight(requirements: Mapping[str, object]) -> Mapping[str, object]:
    rows = []
    for artifact in requirements["artifacts"]:
        path = PROJECT_ROOT / str(artifact["expected_path"])
        available = path.is_file()
        observed_size = int(path.stat().st_size) if available else None
        observed_sha = file_sha256(path) if available else None
        valid = bool(
            available
            and observed_size == int(artifact["size_bytes"])
            and observed_sha == str(artifact["sha256"])
        )
        rows.append(
            {
                "model_id": artifact["model_id"],
                "expected_path": str(path),
                "expected_size_bytes": artifact["size_bytes"],
                "expected_sha256": artifact["sha256"],
                "available": available,
                "observed_size_bytes": observed_size,
                "observed_sha256": observed_sha,
                "valid": valid,
            }
        )
    backend = requirements["required_backend"]
    environment_path = PROJECT_ROOT / str(backend["dedicated_environment"])
    environment_python = environment_path / "bin/python"
    probe = None
    if environment_python.is_file():
        probe = subprocess.run(
            [
                str(environment_python),
                "-c",
                "import importlib.metadata as m; from llama_cpp import llama_cpp; print(m.version(\"llama-cpp-python\")); print(llama_cpp.llama_print_system_info().decode())",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    backend_available = bool(probe is not None and probe.returncode == 0)
    backend_output = probe.stdout.strip() if probe is not None else ""
    backend_version_valid = bool(
        backend_available and backend_output.splitlines()[0] == str(backend["version"])
    )
    backend_cuda_valid = bool(backend_available and "CUDA" in backend_output.upper())
    backend_valid = bool(backend_version_valid and backend_cuda_valid)
    entropy_ready = backend_valid and all(row["valid"] for row in rows[:3])
    quantization_ready = backend_valid and all(row["valid"] for row in rows[2:])
    return {
        "schema_version": "rankcloak-revision-v3-generation-preflight-v2",
        "launch_performed": False,
        "downloads_performed": False,
        "backend_available": backend_available,
        "backend_version_valid": backend_version_valid,
        "backend_cuda_valid": backend_cuda_valid,
        "backend_environment": str(environment_path),
        "backend_probe_stdout": backend_output,
        "backend_probe_stderr": probe.stderr.strip() if probe is not None else "",
        "artifacts": rows,
        "entropy_experiment_ready": entropy_ready,
        "matched_quantization_experiment_ready": quantization_ready,
        "status": "ready" if entropy_ready and quantization_ready else "blocked",
        "requirements_config_sha256": file_sha256(REQUIREMENTS_CONFIG),
        "protocol_amendment_sha256": file_sha256(PROTOCOL_AMENDMENT),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    (output / "provenance").mkdir(parents=True, exist_ok=True)
    detector = load_jsonl(DETECTOR_CORPUS)
    if len(detector) != 15840:
        raise SystemExit("Authoritative detector corpus row count changed")
    trial_frame = pd.read_csv(TRIAL_CORPUS, usecols=["payload_name", "payload_split"])
    split_counts = trial_frame.groupby("payload_name")["payload_split"].nunique()
    if not split_counts.eq(1).all():
        raise SystemExit("A payload has multiple frozen split assignments")
    trial_splits = (
        trial_frame.drop_duplicates("payload_name").set_index("payload_name")["payload_split"].to_dict()
    )
    entropy_config = load_json(ENTROPY_CONFIG)
    quant_config = load_json(QUANTIZATION_CONFIG)
    requirements = load_json(REQUIREMENTS_CONFIG)
    qwen_raw_records = [
        json.loads(line)
        for line in QWEN_Q4_RECORDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qwen_raw_tasks = [
        json.loads(line)
        for line in QWEN_Q4_PLAN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entropy = entropy_plan(detector, entropy_config)
    calibration = calibration_plan(entropy_config)
    quantization = quantization_plan(
        detector,
        trial_splits,
        quant_config,
        qwen_raw_records,
        qwen_raw_tasks,
    )
    atomic_csv(output / "provenance/entropy_generation_plan.csv", entropy)
    atomic_csv(output / "provenance/entropy_calibration_plan.csv", calibration)
    atomic_csv(output / "provenance/quantization_generation_plan.csv", quantization)
    preflight_result = preflight(requirements)
    atomic_json(output / "provenance/generation_preflight.json", preflight_result)
    summary = {
        "schema_version": "rankcloak-revision-v3-generation-plan-summary-v2",
        "protocol_amendment_sha256": file_sha256(PROTOCOL_AMENDMENT),
        "entropy": {
            "analysis_rows": int(len(entropy)),
            "reused_rows": int((~entropy["generation_required"]).sum()),
            "planned_new_generation_rows": int(entropy["generation_required"].sum()),
            "calibration_traces": int(len(calibration)),
            "models": int(entropy["model_id"].nunique()),
            "artifact_classes": int(entropy["payload_class"].nunique()),
            "representation_cells": int(
                entropy[["payload_class", "representation_name"]].drop_duplicates().shape[0]
            ),
            "templates": int(entropy["prompt_template_id"].nunique()),
            "gate_levels": int(entropy["gate_level"].nunique()),
            "experimental_cells": int(entropy["experimental_cell_id"].nunique()),
            "rankcloak_seed_count": int(
                entropy.loc[entropy["population"].eq("rankcloak"), "random_seed"].nunique()
            ),
            "ordinary_control_seed_count": int(
                entropy.loc[
                    entropy["population"].eq("ordinary_control"), "random_seed"
                ].nunique()
            ),
        },
        "quantization": {
            "analysis_rows": int(len(quantization)),
            "reused_q4_rows": int((~quantization["generation_required"]).sum()),
            "planned_new_q8_rows": int(quantization["generation_required"].sum()),
            "payloads": int(quantization["payload_name"].nunique()),
            "artifact_classes": int(quantization["payload_class"].nunique()),
            "codecs": int(quantization["representation_name"].nunique()),
            "quantizations": int(quantization["quantization"].nunique()),
            "historical_seed_source": quant_config[
                "historical_control_seed_source"
            ],
        },
        "preflight_status": preflight_result["status"],
    }
    atomic_json(output / "provenance/generation_plan_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
