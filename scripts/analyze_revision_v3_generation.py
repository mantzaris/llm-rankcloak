#!/usr/bin/env python3
"""Validate and summarize completed revision-V3 model-backed generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_revision_v3 import atomic_csv, atomic_json, atomic_text, utc_now  # noqa: E402
from rankcloak.revision_statistics import automated_text_quality_metrics  # noqa: E402
from rankcloak.revision_v3_generation import (  # noqa: E402
    CALIBRATION_PLAN,
    ENTROPY_PLAN,
    QUANTIZATION_PLAN,
    SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
    load_csv,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_v3"
DEFAULT_GENERATION = DEFAULT_OUTPUT / "generation"
SEED = 20260831
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_COUNTS = {
    "entropy_calibration": 18,
    "entropy": 720,
    "quantization_q4": 1920,
    "quantization_q8": 1920,
}


def stable_seed(*parts: object) -> int:
    material = "\x1f".join([str(SEED), *(str(part) for part in parts)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def load_records(path: Path) -> list[Mapping[str, object]]:
    return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*/*.json"))]


def record_validation_passes(record: Mapping[str, object]) -> bool:
    """Interpret validation fields according to their declared assertion semantics."""

    validation = record.get("validation", {})
    if not isinstance(validation, Mapping):
        return False
    if record.get("record_type") == "entropy_calibration_trace":
        return bool(
            validation.get("target_token_count_exact") is True
            and validation.get("finite_entropy_at_every_position") is True
            and validation.get("detector_outcomes_used") is False
        )
    return bool(validation) and all(value is True for value in validation.values())


def safe_mean(values: Sequence[object]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def group_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    *,
    seed_parts: Sequence[object],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, int]:
    clean = frame[[value_column, group_column]].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.loc[np.isfinite(clean[value_column])]
    groups = sorted(clean[group_column].astype(str).unique())
    if not groups:
        return float("nan"), float("nan"), 0
    by_group = {
        group: clean.loc[clean[group_column].astype(str).eq(group), value_column].to_numpy(dtype=float)
        for group in groups
    }
    rng = np.random.default_rng(stable_seed(*seed_parts))
    estimates = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        selected = rng.choice(groups, size=len(groups), replace=True)
        draw = np.concatenate([by_group[str(group)] for group in selected])
        estimates[index] = float(np.mean(draw))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), int(len(groups))


def surface_metrics(text: str, prompt: str) -> Mapping[str, object]:
    metrics = automated_text_quality_metrics(str(text), str(prompt), language="en")
    return {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "word_count",
            "sentence_count",
            "character_count",
            "flesch_reading_ease_heuristic",
            "flesch_kincaid_grade_heuristic",
            "coleman_liau_index",
            "unique_word_fraction",
            "repeated_bigram_fraction",
            "repeated_trigram_fraction",
            "maximum_identical_word_run",
            "surface_flag_total",
            "artifact_like_fragment_count",
            "prompt_word_jaccard",
        }
    }


def entropy_frames(records: Sequence[Mapping[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    position_rows = []
    for record in records:
        plan = record["plan_row"]
        generation = record["generation"]
        population = str(plan["population"])
        if population == "rankcloak":
            token_ids = generation["embedding_token_ids"]
            text = generation["full_text"]
            entropies = generation["embedding_entropies_bits"]
            ranks = generation["embedding_observed_ranks"]
            logp = generation["embedding_log_probabilities"]
            greedy_logp = generation["embedding_greedy_log_probabilities"]
            pressure = generation[
                "embedding_rank_pressure_log_probability_gaps_nats"
            ]
            roles = generation["embedding_token_roles"]
            eligible = generation["embedding_eligible_mask"]
            realized = generation["realized_ranks"]
            completion = bool(generation["payload_completion"])
            saved_recovery = bool(
                record["saved_token_id_replay"]["exact_payload_recovery"]
            )
            visible_recovery = bool(
                record["visible_text_retokenization"]["exact_payload_recovery"]
            )
            eligible_fraction = (
                float(generation["eligible_position_count"] / len(token_ids))
                if token_ids
                else float("nan")
            )
            bits_per_token = record["fixed_payload"]["bits_per_generated_token"]
            fixed_budget_fraction = record["fixed_token_budget"][
                "payload_fraction_embedded"
            ]
            payload_rank_count = int(generation["requested_payload_rank_count"])
        else:
            token_ids = generation["token_ids"]
            text = generation["text"]
            entropies = generation["next_token_entropies_bits"]
            ranks = generation["sampled_token_ranks"]
            logp = generation["token_log_probabilities"]
            greedy_logp = generation["greedy_log_probabilities"]
            pressure = generation["rank_pressure_log_probability_gaps_nats"]
            roles = ["ordinary_control"] * len(token_ids)
            eligible = [None] * len(token_ids)
            realized = [None] * len(token_ids)
            completion = saved_recovery = visible_recovery = None
            eligible_fraction = bits_per_token = fixed_budget_fraction = None
            payload_rank_count = None
        quality = surface_metrics(text, record["rendered_prompt"])
        trial_rows.append(
            {
                "plan_id": record["plan_id"],
                "experimental_cell_id": plan["experimental_cell_id"],
                "pairing_unit_id": plan["pairing_unit_id"],
                "model_id": plan["model_id"],
                "quantization": plan["quantization"],
                "payload_name": plan["payload_name"],
                "payload_class": plan["payload_class"],
                "payload_index": int(plan["payload_index"]),
                "representation_name": plan["representation_name"],
                "codec_id": plan["source_codec_id"],
                "prompt_template_id": plan["prompt_template_id"],
                "gate_level": plan["gate_level"],
                "population": population,
                "label": int(plan["label"]),
                "threshold_bits": record.get("threshold_bits"),
                "random_seed": int(plan["random_seed"]),
                "generated_token_count": len(token_ids),
                "payload_rank_count": payload_rank_count,
                "payload_completion": completion,
                "saved_id_exact_payload_recovery": saved_recovery,
                "visible_text_exact_payload_recovery": visible_recovery,
                "eligible_position_fraction": eligible_fraction,
                "fixed_payload_bits_per_generated_token": bits_per_token,
                "fixed_token_budget_payload_fraction": fixed_budget_fraction,
                "mean_entropy_bits": safe_mean(entropies),
                "mean_observed_rank": safe_mean(ranks),
                "mean_token_log_probability": safe_mean(logp),
                "mean_token_surprisal_nats": -safe_mean(logp),
                "mean_greedy_log_probability": safe_mean(greedy_logp),
                "mean_rank_pressure_log_probability_gap_nats": safe_mean(pressure),
                "text_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                "full_text": text,
                "execution_seconds": float(record["execution_seconds"]),
                "gpu_memory_peak_mib": record["gpu_memory"]["gpu_memory_peak_mib"],
                **quality,
            }
        )
        for position, token_id in enumerate(token_ids):
            position_rows.append(
                {
                    "plan_id": record["plan_id"],
                    "experimental_cell_id": plan["experimental_cell_id"],
                    "pairing_unit_id": plan["pairing_unit_id"],
                    "model_id": plan["model_id"],
                    "payload_name": plan["payload_name"],
                    "payload_class": plan["payload_class"],
                    "representation_name": plan["representation_name"],
                    "gate_level": plan["gate_level"],
                    "population": population,
                    "position_zero_based": position,
                    "token_id": int(token_id),
                    "token_role": roles[position],
                    "entropy_bits": float(entropies[position]),
                    "eligible": eligible[position],
                    "payload_rank": realized[position],
                    "observed_rank": int(ranks[position]),
                    "token_log_probability": float(logp[position]),
                    "token_surprisal_nats": float(-logp[position]),
                    "greedy_log_probability": float(greedy_logp[position]),
                    "rank_pressure_log_probability_gap_nats": float(pressure[position]),
                }
            )
    trials = pd.DataFrame(trial_rows).sort_values("plan_id").reset_index(drop=True)
    positions = pd.DataFrame(position_rows).sort_values(
        ["plan_id", "position_zero_based"]
    ).reset_index(drop=True)
    ungated = (
        trials.loc[
            trials["population"].eq("rankcloak") & trials["gate_level"].eq("ungated"),
            ["experimental_cell_id", "generated_token_count"],
        ]
        .rename(columns={"generated_token_count": "paired_ungated_token_count"})
        .set_index("experimental_cell_id")
    )
    trials = trials.join(ungated, on="experimental_cell_id")
    trials["added_tokens_vs_ungated"] = np.where(
        trials["population"].eq("rankcloak"),
        trials["generated_token_count"] - trials["paired_ungated_token_count"],
        np.nan,
    )
    trials["length_ratio_vs_ungated"] = np.where(
        trials["population"].eq("rankcloak"),
        trials["generated_token_count"] / trials["paired_ungated_token_count"],
        np.nan,
    )
    return trials, positions


def quantization_frames(
    records: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten paired Q4 replays and Q8 generations without losing pairing."""

    by_pair: dict[str, dict[str, Mapping[str, object]]] = {}
    for record in records:
        plan = record["plan_row"]
        pair_id = str(plan["pairing_unit_id"])
        quantization = str(plan["quantization"])
        if quantization in by_pair.setdefault(pair_id, {}):
            raise ValueError(f"duplicate {quantization} record for {pair_id}")
        by_pair[pair_id][quantization] = record
    incomplete = {
        pair_id: sorted(items)
        for pair_id, items in by_pair.items()
        if set(items) != {"Q4_K_M", "Q8_0"}
    }
    if incomplete:
        raise ValueError(f"incomplete quantization pairs: {list(incomplete.items())[:5]}")

    trial_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for pair_id in sorted(by_pair):
        q4 = by_pair[pair_id]["Q4_K_M"]
        q8 = by_pair[pair_id]["Q8_0"]
        q4_plan = q4["plan_row"]
        q8_plan = q8["plan_row"]
        contract_fields = (
            "pairing_unit_id", "base_model", "model_revision", "representation_name",
            "payload_name", "payload_class", "payload_split", "prompt_template_id",
            "population", "label", "target_token_count",
            "historical_control_sampling_seed", "temperature", "top_p", "sampler",
            "non_quantization_contract_sha256",
        )
        differing = [field for field in contract_fields if str(q4_plan.get(field)) != str(q8_plan.get(field))]
        if differing:
            raise ValueError(f"non-quantization contract differs for {pair_id}: {differing}")

        for quantization, record in (("Q4_K_M", q4), ("Q8_0", q8)):
            plan = record["plan_row"]
            population = str(plan["population"])
            if quantization == "Q4_K_M":
                text = str(record["historical_output_text"])
                token_ids = list(map(int, record["historical_output_token_ids"]))
                trace = record["distribution_trace"]
                summary = record["distribution_summary"]
                saved_recovery = bool(record["rank_replay_exact"]) if population == "rankcloak" else None
                visible_recovery = None
            else:
                generation = record["generation"]
                if population == "rankcloak":
                    text = str(generation["full_text"])
                    token_ids = list(map(int, generation["full_token_ids"]))
                    saved_recovery = bool(record["saved_token_id_decoded"]["exact_payload_recovery"])
                    visible_recovery = bool(record["visible_text_retokenization"]["exact_payload_recovery"])
                else:
                    text = str(generation["text"])
                    token_ids = list(map(int, generation["token_ids"]))
                    saved_recovery = visible_recovery = None
                trace = record["q8_own_path_distribution_trace"]
                summary = record["q8_own_path_distribution_summary"]
            quality = surface_metrics(text, record["rendered_prompt"])
            trial_rows.append(
                {
                    "plan_id": record["plan_id"], "pairing_unit_id": pair_id,
                    "model_id": record["model_id"], "base_model": plan["base_model"],
                    "model_revision": plan["model_revision"], "quantization": quantization,
                    "population": population, "label": int(plan["label"]),
                    "payload_name": plan["payload_name"], "payload_class": plan["payload_class"],
                    "payload_split": plan["payload_split"],
                    "representation_name": plan["representation_name"],
                    "codec_id": plan["codec_id"],
                    "reference_q4_pair_id": plan["reference_q4_pair_id"],
                    "reference_q4_row_id": plan["reference_q4_row_id"],
                    "prompt_template_id": plan["prompt_template_id"],
                    "target_token_count": int(plan["target_token_count"]),
                    "generated_token_count": len(token_ids),
                    "sampling_seed": int(plan["historical_control_sampling_seed"]),
                    "saved_id_exact_payload_recovery": saved_recovery,
                    "visible_text_exact_payload_recovery": visible_recovery,
                    "mean_entropy_bits": summary["mean_entropy_bits"],
                    "mean_observed_rank": summary["mean_observed_rank"],
                    "mean_token_surprisal_nats": summary["mean_observed_surprisal_nats"],
                    "mean_rank_pressure_log_probability_gap_nats": summary["mean_rank_pressure_log_probability_gap_nats"],
                    "tail_rank_frequency_gt_100": summary["tail_rank_frequency_gt_100"],
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "full_text": text, "execution_seconds": float(record["execution_seconds"]),
                    "gpu_memory_peak_mib": record["gpu_memory"]["gpu_memory_peak_mib"],
                    "new_generation_performed": bool(record["new_generation_performed"]),
                    **quality,
                }
            )
            for position, token_id in enumerate(trace["observed_token_ids"]):
                position_rows.append(
                    {
                        "plan_id": record["plan_id"], "pairing_unit_id": pair_id,
                        "payload_name": plan["payload_name"], "population": population,
                        "quantization": quantization, "path": "own_generated_path",
                        "position_zero_based": position, "token_id": int(token_id),
                        "entropy_bits": float(trace["entropy_bits"][position]),
                        "observed_rank": int(trace["observed_ranks"][position]),
                        "token_surprisal_nats": float(trace["observed_surprisals_nats"][position]),
                        "rank_pressure_log_probability_gap_nats": float(trace["rank_pressure_log_probability_gaps_nats"][position]),
                        "greedy_token_id": int(trace["greedy_token_ids"][position]),
                    }
                )

        comparison = q8["q4_q8_same_path_distribution_comparison"]
        output_comparison = q8["q4_q8_generated_output_comparison"]
        q4_trace = q4["distribution_trace"]
        q8_same = q8["q8_replay_of_historical_q4_path"]
        if list(map(int, q4_trace["observed_token_ids"])) != list(map(int, q8_same["observed_token_ids"])):
            raise ValueError(f"same-path token sequence mismatch for {pair_id}")
        paired_rows.append(
            {
                "pairing_unit_id": pair_id, "payload_name": q8_plan["payload_name"],
                "payload_class": q8_plan["payload_class"], "payload_split": q8_plan["payload_split"],
                "representation_name": q8_plan["representation_name"],
                "prompt_template_id": q8_plan["prompt_template_id"],
                "population": q8_plan["population"], "label": int(q8_plan["label"]),
                "target_token_count": int(q8_plan["target_token_count"]),
                "sampling_seed": int(q8_plan["historical_control_sampling_seed"]),
                "same_path_position_count": int(comparison["position_count"]),
                "mean_entropy_q8_minus_q4_bits": comparison["mean_entropy_q8_minus_q4_bits"],
                "median_entropy_q8_minus_q4_bits": comparison["median_entropy_q8_minus_q4_bits"],
                "observed_token_rank_changed_count": int(comparison["observed_token_rank_changed_count"]),
                "observed_token_rank_changed_fraction": comparison["observed_token_rank_changed_fraction"],
                "greedy_token_changed_count": int(comparison["greedy_token_changed_count"]),
                "greedy_token_changed_fraction": comparison["greedy_token_changed_fraction"],
                "mean_absolute_observed_rank_change": comparison["mean_absolute_observed_rank_change"],
                "exact_generated_token_sequence_match": bool(output_comparison["exact_token_sequence_match"]),
                "positionwise_generated_token_match_fraction": output_comparison["positionwise_token_match_fraction"],
                "first_generated_token_divergence": output_comparison["first_divergence"][
                    "position_zero_based"
                ],
            }
        )
        for position, token_id in enumerate(q4_trace["observed_token_ids"]):
            position_rows.append(
                {
                    "plan_id": q8["plan_id"], "pairing_unit_id": pair_id,
                    "payload_name": q8_plan["payload_name"], "population": q8_plan["population"],
                    "quantization": "Q8_0", "path": "historical_q4_token_path",
                    "position_zero_based": position, "token_id": int(token_id),
                    "entropy_bits": float(q8_same["entropy_bits"][position]),
                    "observed_rank": int(q8_same["observed_ranks"][position]),
                    "token_surprisal_nats": float(q8_same["observed_surprisals_nats"][position]),
                    "rank_pressure_log_probability_gap_nats": float(q8_same["rank_pressure_log_probability_gaps_nats"][position]),
                    "greedy_token_id": int(q8_same["greedy_token_ids"][position]),
                }
            )

    trials = pd.DataFrame(trial_rows).sort_values(["pairing_unit_id", "quantization"]).reset_index(drop=True)
    pairs = pd.DataFrame(paired_rows).sort_values("pairing_unit_id").reset_index(drop=True)
    positions = pd.DataFrame(position_rows).sort_values(
        ["pairing_unit_id", "path", "quantization", "position_zero_based"]
    ).reset_index(drop=True)
    return trials, pairs, positions


def summarize_metrics(
    frame: pd.DataFrame,
    strata: Sequence[str],
    metrics: Sequence[str],
    *,
    group_column: str,
    analysis_id: str,
) -> pd.DataFrame:
    """Long-form means and payload-grouped bootstrap intervals."""

    rows: list[dict[str, object]] = []
    grouper: object = list(strata) if len(strata) > 1 else strata[0]
    for keys, cell in frame.groupby(grouper, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(strata, keys))
        for metric in metrics:
            numeric = pd.to_numeric(cell[metric], errors="coerce")
            available = cell.loc[np.isfinite(numeric)].copy()
            if available.empty:
                continue
            available[metric] = pd.to_numeric(available[metric], errors="coerce")
            low, high, group_count = group_bootstrap_interval(
                available,
                metric,
                group_column,
                seed_parts=(analysis_id, *keys, metric),
            )
            values = available[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "analysis_id": analysis_id,
                    **identifiers,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low_95": low,
                    "ci_high_95": high,
                    "observation_count": int(len(values)),
                    "group_count": group_count,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_unit": group_column,
                }
            )
    return pd.DataFrame(rows)


def entropy_pair_differences(trials: pd.DataFrame) -> pd.DataFrame:
    rankcloak = trials.loc[trials["population"].eq("rankcloak")].copy()
    baseline = rankcloak.loc[rankcloak["gate_level"].eq("ungated")].set_index(
        "experimental_cell_id"
    )
    metrics = (
        "generated_token_count",
        "mean_entropy_bits",
        "mean_observed_rank",
        "mean_token_surprisal_nats",
        "mean_rank_pressure_log_probability_gap_nats",
        "word_count",
        "unique_word_fraction",
        "repeated_bigram_fraction",
        "surface_flag_total",
    )
    rows: list[dict[str, object]] = []
    for _, row in rankcloak.loc[rankcloak["gate_level"].isin(["moderate", "strict"])].iterrows():
        reference = baseline.loc[row["experimental_cell_id"]]
        result = {
            "experimental_cell_id": row["experimental_cell_id"],
            "model_id": row["model_id"],
            "payload_name": row["payload_name"],
            "payload_class": row["payload_class"],
            "representation_name": row["representation_name"],
            "prompt_template_id": row["prompt_template_id"],
            "gate_level": row["gate_level"],
        }
        for metric in metrics:
            result[f"{metric}_difference_vs_ungated"] = float(row[metric]) - float(reference[metric])
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["gate_level", "experimental_cell_id"]).reset_index(drop=True)


def calibration_tables(
    generation_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_records = load_records(generation_root / "raw/entropy_calibration")
    trace_rows: list[dict[str, object]] = []
    entropy_by_model: dict[str, list[float]] = {}
    for record in trace_records:
        generation = record["generation"]
        entropies = list(map(float, generation["next_token_entropies_bits"]))
        entropy_by_model.setdefault(str(record["model_id"]), []).extend(entropies)
        trace_rows.append(
            {
                "plan_id": record["plan_id"],
                "model_id": record["model_id"],
                "prompt_category": record["prompt_category"],
                "prompt_template_id": record["prompt_template_id"],
                "sampling_seed": generation["sampling_seed"],
                "position_count": len(entropies),
                "mean_entropy_bits": float(np.mean(entropies)),
                "median_entropy_bits": float(np.median(entropies)),
                "execution_seconds": record["execution_seconds"],
                "gpu_memory_peak_mib": record["gpu_memory"]["gpu_memory_peak_mib"],
                "result_sha256": canonical_sha256(record),
            }
        )
    threshold_rows: list[dict[str, object]] = []
    for path in sorted((generation_root / "calibration/thresholds").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(document["model_id"])
        entropies = np.asarray(entropy_by_model[model_id], dtype=float)
        moderate = float(np.quantile(entropies, 0.5, method="linear"))
        strict = float(np.quantile(entropies, 0.75, method="linear"))
        if not (
            math.isclose(moderate, float(document["moderate_threshold_bits"]), abs_tol=1e-12)
            and math.isclose(strict, float(document["strict_threshold_bits"]), abs_tol=1e-12)
        ):
            raise ValueError(f"frozen entropy threshold does not reproduce for {model_id}")
        threshold_rows.append(
            {
                "model_id": model_id,
                "trace_count": int(document["trace_count"]),
                "development_position_count": int(document["development_position_count"]),
                "moderate_quantile": float(document["moderate_quantile"]),
                "moderate_threshold_bits": moderate,
                "strict_quantile": float(document["strict_quantile"]),
                "strict_threshold_bits": strict,
                "quantile_method": document["quantile_method"],
                "source": document["source"],
                "detector_outcomes_used": bool(document["detector_outcomes_used"]),
                "threshold_record_sha256": file_sha256(path),
            }
        )
    return (
        pd.DataFrame(trace_rows).sort_values(["model_id", "plan_id"]).reset_index(drop=True),
        pd.DataFrame(threshold_rows).sort_values("model_id").reset_index(drop=True),
    )


def validate_generation_ledgers(
    generation_root: Path,
    calibration_records: Sequence[Mapping[str, object]],
    entropy_records: Sequence[Mapping[str, object]],
    quantization_records: Sequence[Mapping[str, object]],
    smoke_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Fail closed on completeness, pairing, sampler, replay, and gate invariants."""

    checks: dict[str, object] = {}
    errors: list[str] = []

    plans = {
        "entropy_calibration": load_csv(CALIBRATION_PLAN),
        "entropy": load_csv(ENTROPY_PLAN),
        "quantization": load_csv(QUANTIZATION_PLAN),
    }
    record_sets = {
        "entropy_calibration": calibration_records,
        "entropy": entropy_records,
        "quantization": quantization_records,
    }
    for name, records in record_sets.items():
        expected = {str(row["plan_id"]) for row in plans[name]}
        observed = [str(record["plan_id"]) for record in records]
        checks[f"{name}_expected_count"] = len(expected)
        checks[f"{name}_completed_count"] = len(observed)
        checks[f"{name}_plan_identity_exact"] = len(observed) == len(set(observed)) and set(observed) == expected
        if not checks[f"{name}_plan_identity_exact"]:
            errors.append(f"{name} completed records do not exactly cover the plan")
        invalid = [
            str(record["plan_id"])
            for record in records
            if record.get("schema_version") != SCHEMA_VERSION
            or record.get("execution_status") != "completed"
            or not record_validation_passes(record)
        ]
        checks[f"{name}_record_validation_failures"] = invalid
        if invalid:
            errors.append(f"{name} contains invalid completed records")

    expected_smoke = {
        ("entropy_calibration_trace", "llama3_8b_instruct_q4_k_m"),
        ("entropy_calibration_trace", "mistral_7b_instruct_v0_3_q4_k_m"),
        ("entropy_calibration_trace", "qwen2_5_7b_instruct_q4_k_m"),
        ("entropy_rankcloak_trial", "llama3_8b_instruct_q4_k_m"),
        ("entropy_rankcloak_trial", "mistral_7b_instruct_v0_3_q4_k_m"),
        ("entropy_rankcloak_trial", "qwen2_5_7b_instruct_q4_k_m"),
        ("quantization_q4_model_backed_replay", "qwen2_5_7b_instruct_q4_k_m"),
        ("quantization_q8_generation", "qwen2_5_7b_instruct_q8_0"),
    }
    observed_smoke = [
        (str(record.get("record_type")), str(record.get("model_id")))
        for record in smoke_records
    ]
    checks["real_model_smoke_record_count"] = len(smoke_records)
    checks["real_model_smoke_matrix_exact"] = bool(
        len(observed_smoke) == len(set(observed_smoke))
        and set(observed_smoke) == expected_smoke
        and all(
            record.get("schema_version") == SCHEMA_VERSION
            and record.get("execution_status") == "completed"
            and record_validation_passes(record)
            for record in smoke_records
        )
    )
    if not checks["real_model_smoke_matrix_exact"]:
        errors.append("real-model smoke matrix is missing, duplicated, or invalid")

    failure_files = sorted((generation_root / "failures").glob("**/*.json"))
    checks["failure_record_count"] = len(failure_files)
    checks["failure_record_paths"] = [str(path.relative_to(PROJECT_ROOT)) for path in failure_files]
    if failure_files:
        errors.append("one or more model-backed trials has a failure record")

    entropy_plan = pd.DataFrame(plans["entropy"])
    checks["entropy_paired_seed_policy"] = bool(
        entropy_plan.groupby(["experimental_cell_id", "population"])["random_seed"].nunique().eq(1).all()
        and entropy_plan.groupby("experimental_cell_id")["random_seed"].nunique().eq(2).all()
    )
    if not checks["entropy_paired_seed_policy"]:
        errors.append("entropy paired seeds do not match the amended protocol")

    gate_errors: list[str] = []
    controls_by_cell: dict[str, list[Mapping[str, object]]] = {}
    for record in entropy_records:
        plan = record["plan_row"]
        if str(plan["population"]) == "ordinary_control":
            controls_by_cell.setdefault(str(plan["experimental_cell_id"]), []).append(record)
            generation = record["generation"]
            if not (
                generation["sampler"] == "numpy_pcg64_serial_top_p_v1_token_id_tiebreak"
                and math.isclose(float(generation["temperature"]), 0.8)
                and math.isclose(float(generation["top_p"]), 0.95)
                and set(generation["token_role_mask"]) == {"ordinary_control"}
            ):
                gate_errors.append(str(record["plan_id"]))
            continue
        generation = record["generation"]
        entropies = list(map(float, generation["embedding_entropies_bits"]))
        eligible = list(map(bool, generation["embedding_eligible_mask"]))
        roles = list(map(str, generation["embedding_token_roles"]))
        threshold = record["threshold_bits"]
        expected_eligible = [True] * len(entropies) if threshold is None else [value >= float(threshold) for value in entropies]
        if threshold is None:
            sampling_policy_valid = bool(
                generation["ineligible_token_policy"] == "not_applicable_gate_disabled"
                and generation["ordinary_sampling_seed"] is None
                and generation["ordinary_sampling_temperature"] is None
                and generation["ordinary_sampling_top_p"] is None
                and generation["ordinary_sampler"] is None
                and not generation["ordinary_sampled_skip_positions"]
            )
        else:
            sampling_policy_valid = bool(
                generation["ineligible_token_policy"] == "ordinary_seeded_top_p_sampling"
                and generation["ordinary_sampler"]
                == "numpy_pcg64_serial_top_p_v1_token_id_tiebreak"
                and math.isclose(
                    float(generation["ordinary_sampling_temperature"]), 0.8
                )
                and math.isclose(float(generation["ordinary_sampling_top_p"]), 0.95)
            )
        role_valid = all(
            role == ("payload" if is_eligible else "ordinary_sampled_skip")
            for role, is_eligible in zip(roles, eligible)
        )
        realized = list(generation["realized_ranks"])
        expected = list(map(int, record["expected_ranks"]))
        consumed = int(generation["consumed_payload_rank_count"])
        if not (
            eligible == expected_eligible
            and role_valid
            and realized == expected[:consumed]
            and sum(eligible) == consumed
            and generation["ineligible_position_count"] == len(eligible) - consumed
            and record["saved_token_id_replay"]["exact_rank_prefix_recovery"]
            and record["validation"]["encoder_decoder_gate_positions_exact"]
            and sampling_policy_valid
        ):
            gate_errors.append(str(record["plan_id"]))
    checks["entropy_gate_invariant_failures"] = gate_errors
    if gate_errors:
        errors.append("entropy gate or replay invariants failed")

    prefix_errors: list[str] = []
    for cell_id, records in controls_by_cell.items():
        if len(records) != 3:
            prefix_errors.append(cell_id)
            continue
        seeds = {int(record["generation"]["sampling_seed"]) for record in records}
        contexts = {tuple(record["generation"]["context_token_ids"]) for record in records}
        ordered = sorted((list(map(int, record["generation"]["token_ids"])) for record in records), key=len)
        prefix_ok = all(longer[: len(shorter)] == shorter for shorter, longer in zip(ordered, ordered[1:]))
        if len(seeds) != 1 or len(contexts) != 1 or not prefix_ok:
            prefix_errors.append(cell_id)
    checks["entropy_control_prefix_failures"] = prefix_errors
    if prefix_errors:
        errors.append("length-matched entropy controls are not deterministic prefixes")

    pair_contract_errors: list[str] = []
    q4_by_pair: dict[str, Mapping[str, object]] = {}
    q8_by_pair: dict[str, Mapping[str, object]] = {}
    for record in quantization_records:
        plan = record["plan_row"]
        pair_id = str(plan["pairing_unit_id"])
        target = q4_by_pair if str(plan["quantization"]) == "Q4_K_M" else q8_by_pair
        if pair_id in target:
            pair_contract_errors.append(pair_id)
        target[pair_id] = record
    for pair_id in sorted(set(q4_by_pair) | set(q8_by_pair)):
        if pair_id not in q4_by_pair or pair_id not in q8_by_pair:
            pair_contract_errors.append(pair_id)
            continue
        q4 = q4_by_pair[pair_id]
        q8 = q8_by_pair[pair_id]
        q4_plan = q4["plan_row"]
        q8_plan = q8["plan_row"]
        if not (
            q4_plan["non_quantization_contract_sha256"] == q8_plan["non_quantization_contract_sha256"]
            and int(q4_plan["historical_control_sampling_seed"]) == int(q8_plan["historical_control_sampling_seed"])
            and not q4["new_generation_performed"]
            and q8["new_generation_performed"]
            and q8["paired_q4_replay_sha256"] == canonical_sha256(q4)
        ):
            pair_contract_errors.append(pair_id)
    checks["quantization_pair_contract_failures"] = sorted(set(pair_contract_errors))
    if pair_contract_errors:
        errors.append("matched-quantization pair contract failed")

    checks["status"] = "pass" if not errors else "fail"
    return {
        "schema_version": "rankcloak-revision-v3-generation-analysis-validation-v1",
        "status": checks["status"],
        "analysis_git_commit": subprocess_git_head(),
        "counts": {
            "calibration_traces": len(calibration_records),
            "entropy_evaluations": len(entropy_records),
            "quantization_q4_replays": len(q4_by_pair),
            "quantization_q8_generations": len(q8_by_pair),
            "real_model_smoke_records": len(smoke_records),
        },
        "checks": checks,
        "errors": errors,
    }


def subprocess_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generation_environment(
    records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not records:
        raise ValueError("generation environment requires completed records")
    backends: dict[str, Mapping[str, object]] = {}
    artifacts: dict[str, Mapping[str, object]] = {}
    artifact_paths: dict[str, set[str]] = {}
    artifact_verification_times: dict[str, set[str]] = {}
    tokenizers: dict[str, Mapping[str, object]] = {}
    source_hashes: dict[str, set[str]] = {}
    for record in records:
        manifest = record["model_manifest"]
        backend = manifest["backend"]
        backends.setdefault(canonical_sha256(backend), backend)
        artifact = manifest["artifact"]
        model_id = str(artifact["model_id"])
        artifact_identity = {
            field: artifact[field]
            for field in (
                "model_id",
                "repo_id",
                "revision",
                "filename",
                "quantization",
                "size_bytes",
                "sha256",
            )
        }
        if model_id in artifacts and canonical_sha256(artifacts[model_id]) != canonical_sha256(artifact_identity):
            raise ValueError(f"generation records disagree on artifact for {model_id}")
        artifacts[model_id] = artifact_identity
        artifact_paths.setdefault(model_id, set()).add(str(artifact["path"]))
        artifact_verification_times.setdefault(model_id, set()).add(
            str(artifact["verified_at"])
        )
        tokenizer = {
            "model_id": artifact["model_id"],
            "identifier": (
                f"{artifact['repo_id']}@{artifact['revision']}:embedded_gguf_tokenizer"
            ),
            "source": "embedded_gguf_tokenizer",
            "gguf_artifact_sha256": artifact["sha256"],
            "vocabulary_size": int(manifest["vocabulary_size"]),
            "bos_token_id": int(manifest["bos_token_id"]),
            "gguf_general_name": manifest["gguf_general_name"],
        }
        if model_id in tokenizers and canonical_sha256(tokenizers[model_id]) != canonical_sha256(tokenizer):
            raise ValueError(f"generation records disagree on tokenizer for {model_id}")
        tokenizers[model_id] = tokenizer
        for source, digest in record.get("source_hashes", {}).items():
            source_hashes.setdefault(str(source), set()).add(str(digest))
    if len(backends) != 1:
        raise ValueError("generation records disagree on backend environment")
    backend = next(iter(backends.values()))
    requirements = json.loads(
        (PROJECT_ROOT / "configs/revision_v3/generation_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    cpu_text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    cpu_match = re.search(r"^model name\s*:\s*(.+)$", cpu_text, flags=re.MULTILINE)
    memory_text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    memory_match = re.search(r"^MemTotal:\s*(\d+)\s*kB$", memory_text, flags=re.MULTILINE)
    nvidia_raw = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_raw)
    required_backend = requirements["required_backend"]
    q4_tokenizer = tokenizers.get("qwen2_5_7b_instruct_q4_k_m")
    q8_tokenizer = tokenizers.get("qwen2_5_7b_instruct_q8_0")
    paired_tokenizer_contract = None
    if q4_tokenizer is not None and q8_tokenizer is not None:
        paired_tokenizer_contract = {
            "q4_model_id": q4_tokenizer["model_id"],
            "q8_model_id": q8_tokenizer["model_id"],
            "upstream_identifier_exact": (
                q4_tokenizer["identifier"] == q8_tokenizer["identifier"]
            ),
            "vocabulary_size_exact": (
                q4_tokenizer["vocabulary_size"] == q8_tokenizer["vocabulary_size"]
            ),
            "bos_token_id_exact": (
                q4_tokenizer["bos_token_id"] == q8_tokenizer["bos_token_id"]
            ),
            "rendered_context_ids_checked_per_trial": True,
        }
        if not all(
            value is True
            for key, value in paired_tokenizer_contract.items()
            if key.endswith("_exact")
        ):
            raise ValueError("paired Q4/Q8 embedded tokenizers disagree")
    return {
        "schema_version": "rankcloak-revision-v3-generation-environment-v1",
        "python_version": backend["python_version"],
        "python_implementation": backend["python_implementation"],
        "platform": backend["platform"],
        "cpu_model": cpu_match.group(1).strip() if cpu_match else None,
        "logical_cpu_count": __import__("os").cpu_count(),
        "physical_memory_bytes": int(memory_match.group(1)) * 1024 if memory_match else None,
        "packages": backend["packages"],
        "llama_cpp_system_info": backend["llama_cpp_system_info"],
        "gpu_offload_supported": backend["gpu_offload_supported"],
        "gpu_inventory": backend["gpu_inventory"],
        "driver_reported_cuda_compatibility_version": cuda_match.group(1) if cuda_match else None,
        "deterministic_environment": backend["deterministic_environment"],
        "dedicated_environment": required_backend["dedicated_environment"],
        "environment_creation_command": required_backend["environment_creation_command"],
        "backend_installation_method": required_backend["installation_method"],
        "backend_installation_command": required_backend["installation_command"],
        "cuda_runtime_installation_command": required_backend["runtime_installation_command"],
        "source_build_unavailable_reason": required_backend["source_build_unavailable_reason"],
        "model_download_commands": [item["download_command"] for item in requirements["artifacts"]],
        "model_artifacts": [
            {
                **artifacts[key],
                "local_paths": sorted(artifact_paths[key]),
                "verified_at": sorted(artifact_verification_times[key]),
            }
            for key in sorted(artifacts)
        ],
        "model_artifact_total_bytes": int(sum(int(item["size_bytes"]) for item in artifacts.values())),
        "tokenizer_identifiers": [tokenizers[key] for key in sorted(tokenizers)],
        "matched_quantization_tokenizer_contract": paired_tokenizer_contract,
        "execution_git_commits": sorted(
            {str(record["execution_git_commit"]) for record in records}
        ),
        "protocol_amendment_commits": sorted(
            {str(record["protocol_amendment_commit"]) for record in records}
        ),
        "recorded_source_hashes": {
            source: sorted(digests) for source, digests in sorted(source_hashes.items())
        },
        "remote_paid_compute_used": False,
        "execution_started_at": min(str(record["started_at"]) for record in records),
        "execution_completed_at": max(str(record["completed_at"]) for record in records),
    }


def position_summary(frame: pd.DataFrame, strata: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: object = list(strata) if len(strata) > 1 else strata[0]
    metrics = (
        "entropy_bits",
        "observed_rank",
        "token_surprisal_nats",
        "rank_pressure_log_probability_gap_nats",
    )
    for keys, cell in frame.groupby(grouper, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(strata, keys))
        for metric in metrics:
            values = pd.to_numeric(cell[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append(
                {
                    **identifiers,
                    "metric": metric,
                    "position_count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "minimum": float(np.min(values)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: object, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "RankCloak revision V3", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "RankCloak revision V3"},
    )
    plt.close(fig)


def build_generation_figures(
    output: Path,
    entropy_trials: pd.DataFrame,
    entropy_summary: pd.DataFrame,
    quantization_trials: pd.DataFrame,
    quantization_pairs: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    levels = ["ungated", "moderate", "strict"]
    labels = ["Ungated", "Median gate", "75th-percentile gate"]
    rankcloak = entropy_trials.loc[entropy_trials["population"].eq("rankcloak")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    measures = (
        ("payload_completion", "Payload completion", (0.0, 1.05)),
        ("fixed_payload_bits_per_generated_token", "Bits / generated token", None),
        ("length_ratio_vs_ungated", "Length ratio vs ungated", None),
        ("visible_text_exact_payload_recovery", "Visible-text exact recovery", (0.0, 1.05)),
    )
    for axis, (metric, title, ylim) in zip(axes.flat, measures):
        cell = entropy_summary.loc[
            entropy_summary["analysis_id"].eq("entropy_overall")
            & entropy_summary["population"].eq("rankcloak")
            & entropy_summary["metric"].eq(metric)
        ].set_index("gate_level")
        means = np.asarray([cell.loc[level, "mean"] for level in levels], dtype=float)
        low = np.asarray([cell.loc[level, "ci_low_95"] for level in levels], dtype=float)
        high = np.asarray([cell.loc[level, "ci_high_95"] for level in levels], dtype=float)
        x = np.arange(3)
        axis.errorbar(x, means, yerr=np.vstack([means - low, high - means]), fmt="o-", capsize=3)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(title)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Entropy-gated RankCloak: capacity and recovery")
    fig.tight_layout()
    save_figure(fig, output / "figures/entropy_gate_capacity_recovery")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for axis, metric, title in zip(
        axes,
        ("mean_entropy_q8_minus_q4_bits", "observed_token_rank_changed_fraction", "positionwise_generated_token_match_fraction"),
        ("Entropy change on fixed Q4 path (bits)", "Observed-rank divergence", "Q4/Q8 generated-token match"),
    ):
        control = quantization_pairs.loc[quantization_pairs["population"].eq("ordinary_control"), metric]
        encoded = quantization_pairs.loc[quantization_pairs["population"].eq("rankcloak"), metric]
        axis.boxplot([control.dropna(), encoded.dropna()], labels=["Control", "RankCloak"], showfliers=False)
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched Q4_K_M versus Q8_0 sensitivity")
    fig.tight_layout()
    save_figure(fig, output / "figures/matched_quantization_sensitivity")

    q8_rank = quantization_trials.loc[
        quantization_trials["quantization"].eq("Q8_0")
        & quantization_trials["population"].eq("rankcloak")
    ]
    recovery = [
        float(q8_rank["saved_id_exact_payload_recovery"].astype(float).mean()),
        float(q8_rank["visible_text_exact_payload_recovery"].astype(float).mean()),
    ]
    fig, axis = plt.subplots(figsize=(3.8, 3.0))
    bars = axis.bar(["Saved token IDs", "Visible text"], recovery, color=["#3b6ea8", "#d08c3c"])
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in recovery], padding=2)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Exact payload recovery")
    axis.set_title("Q8_0 recovery mode")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output / "figures/matched_quantization_recovery")


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    return frame.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        na_rep="--",
        caption=caption,
        label=label,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    generation_root = args.generation_dir.resolve()

    calibration_records = load_records(generation_root / "raw/entropy_calibration")
    entropy_records = load_records(generation_root / "raw/entropy")
    quantization_records = load_records(generation_root / "raw/quantization")
    smoke_records = [
        record
        for phase in ("entropy_calibration", "entropy", "quantization")
        for record in load_records(generation_root / f"smoke/{phase}")
    ]
    audit = validate_generation_ledgers(
        generation_root,
        calibration_records,
        entropy_records,
        quantization_records,
        smoke_records,
    )
    if audit["status"] != "pass":
        atomic_json(output / "provenance/generation_analysis_validation.json", audit)
        raise SystemExit("model-backed generation ledger validation failed: " + "; ".join(audit["errors"]))

    calibration_traces, thresholds = calibration_tables(generation_root)
    entropy_trials, entropy_positions = entropy_frames(entropy_records)
    quantization_trials, quantization_pairs, quantization_positions = quantization_frames(
        quantization_records
    )
    entropy_differences = entropy_pair_differences(entropy_trials)

    entropy_metrics = (
        "payload_completion", "saved_id_exact_payload_recovery",
        "visible_text_exact_payload_recovery", "eligible_position_fraction",
        "fixed_payload_bits_per_generated_token", "fixed_token_budget_payload_fraction",
        "generated_token_count", "added_tokens_vs_ungated", "length_ratio_vs_ungated",
        "mean_entropy_bits", "mean_observed_rank", "mean_token_surprisal_nats",
        "mean_rank_pressure_log_probability_gap_nats", "word_count",
        "unique_word_fraction", "repeated_bigram_fraction", "surface_flag_total",
    )
    entropy_summaries = [
        summarize_metrics(
            entropy_trials, ["population", "gate_level"], entropy_metrics,
            group_column="experimental_cell_id", analysis_id="entropy_overall",
        ),
        summarize_metrics(
            entropy_trials, ["population", "gate_level", "model_id"], entropy_metrics,
            group_column="experimental_cell_id", analysis_id="entropy_by_model",
        ),
        summarize_metrics(
            entropy_trials, ["population", "gate_level", "representation_name"], entropy_metrics,
            group_column="experimental_cell_id", analysis_id="entropy_by_representation",
        ),
        summarize_metrics(
            entropy_trials, ["population", "gate_level", "payload_class"], entropy_metrics,
            group_column="experimental_cell_id", analysis_id="entropy_by_artifact_class",
        ),
    ]
    entropy_summary = pd.concat(entropy_summaries, ignore_index=True, sort=False)
    difference_metrics = [column for column in entropy_differences if column.endswith("_difference_vs_ungated")]
    entropy_difference_summary = summarize_metrics(
        entropy_differences, ["gate_level"], difference_metrics,
        group_column="experimental_cell_id", analysis_id="paired_entropy_difference",
    )

    quant_metrics = (
        "saved_id_exact_payload_recovery", "visible_text_exact_payload_recovery",
        "mean_entropy_bits", "mean_observed_rank", "mean_token_surprisal_nats",
        "mean_rank_pressure_log_probability_gap_nats", "tail_rank_frequency_gt_100",
        "word_count", "unique_word_fraction", "repeated_bigram_fraction", "surface_flag_total",
    )
    quantization_summary = pd.concat(
        [
            summarize_metrics(
                quantization_trials, ["quantization", "population"], quant_metrics,
                group_column="payload_name", analysis_id="quantization_overall",
            ),
            summarize_metrics(
                quantization_trials, ["quantization", "population", "representation_name"], quant_metrics,
                group_column="payload_name", analysis_id="quantization_by_representation",
            ),
            summarize_metrics(
                quantization_trials, ["quantization", "population", "payload_class"], quant_metrics,
                group_column="payload_name", analysis_id="quantization_by_artifact_class",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    paired_metrics = (
        "mean_entropy_q8_minus_q4_bits", "median_entropy_q8_minus_q4_bits",
        "observed_token_rank_changed_fraction", "greedy_token_changed_fraction",
        "mean_absolute_observed_rank_change", "exact_generated_token_sequence_match",
        "positionwise_generated_token_match_fraction",
    )
    quantization_pair_summary = pd.concat(
        [
            summarize_metrics(
                quantization_pairs, ["population"], paired_metrics,
                group_column="payload_name", analysis_id="quantization_pair_overall",
            ),
            summarize_metrics(
                quantization_pairs, ["population", "representation_name"], paired_metrics,
                group_column="payload_name", analysis_id="quantization_pair_by_representation",
            ),
            summarize_metrics(
                quantization_pairs, ["population", "payload_class"], paired_metrics,
                group_column="payload_name", analysis_id="quantization_pair_by_artifact_class",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    entropy_position_summary = position_summary(
        entropy_positions, ["population", "gate_level", "model_id", "token_role"]
    )
    quantization_position_summary = position_summary(
        quantization_positions, ["population", "quantization", "path"]
    )
    source = output / "source_tables"
    provenance = output / "provenance"
    analysis = generation_root / "analysis"
    manuscript = output / "manuscript_tables"
    for directory in (source, provenance, analysis, manuscript, output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)

    tables = {
        source / "entropy_calibration_traces.csv": calibration_traces,
        source / "entropy_calibration_thresholds.csv": thresholds,
        source / "entropy_generation_trials.csv": entropy_trials.drop(columns=["full_text"]),
        source / "entropy_generation_summary.csv": entropy_summary,
        source / "entropy_paired_differences.csv": entropy_differences,
        source / "entropy_paired_difference_summary.csv": entropy_difference_summary,
        source / "entropy_position_summary.csv": entropy_position_summary,
        source / "quantization_generation_trials.csv": quantization_trials.drop(columns=["full_text"]),
        source / "quantization_pair_comparison.csv": quantization_pairs,
        source / "quantization_generation_summary.csv": quantization_summary,
        source / "quantization_pair_summary.csv": quantization_pair_summary,
        source / "quantization_position_summary.csv": quantization_position_summary,
        analysis / "entropy_detector_corpus.csv": entropy_trials,
        analysis / "quantization_detector_corpus.csv": quantization_trials,
    }
    for path, frame in tables.items():
        atomic_csv(path, frame)
    atomic_json(provenance / "generation_analysis_validation.json", audit)
    environment = dict(
        generation_environment(
            [
                *smoke_records,
                *calibration_records,
                *entropy_records,
                *quantization_records,
            ]
        )
    )
    environment.update(
        {
            "real_model_smoke_record_count": len(smoke_records),
            "real_model_smoke_new_generation_count": sum(
                record["record_type"] != "quantization_q4_model_backed_replay"
                for record in smoke_records
            ),
            "real_model_smoke_q4_replay_count": sum(
                record["record_type"] == "quantization_q4_model_backed_replay"
                for record in smoke_records
            ),
            "real_model_smoke_execution_seconds_sum": float(
                sum(float(record["execution_seconds"]) for record in smoke_records)
            ),
        }
    )
    atomic_json(provenance / "generation_environment.json", environment)

    entropy_main = entropy_summary.loc[
        entropy_summary["analysis_id"].eq("entropy_overall")
        & entropy_summary["population"].eq("rankcloak")
        & entropy_summary["metric"].isin(
            ["payload_completion", "fixed_payload_bits_per_generated_token", "length_ratio_vs_ungated", "saved_id_exact_payload_recovery", "visible_text_exact_payload_recovery"]
        )
    ][["gate_level", "metric", "mean", "ci_low_95", "ci_high_95", "observation_count", "group_count"]]
    quant_main = pd.concat(
        [
            quantization_summary.loc[
                quantization_summary["analysis_id"].eq("quantization_overall")
                & quantization_summary["population"].eq("rankcloak")
                & quantization_summary["metric"].isin(["saved_id_exact_payload_recovery", "visible_text_exact_payload_recovery"])
            ][["quantization", "metric", "mean", "ci_low_95", "ci_high_95", "observation_count", "group_count"]],
            quantization_pair_summary.loc[
                quantization_pair_summary["analysis_id"].eq("quantization_pair_overall")
            ][["population", "metric", "mean", "ci_low_95", "ci_high_95", "observation_count", "group_count"]],
        ],
        ignore_index=True,
        sort=False,
    )
    atomic_text(
        manuscript / "entropy_gate_generation.tex",
        latex_table(entropy_main, "Entropy-gated RankCloak generation outcomes.", "tab:entropy-gate-generation"),
    )
    atomic_text(
        manuscript / "entropy_calibration_thresholds.tex",
        latex_table(thresholds.drop(columns=["threshold_record_sha256"]), "Frozen model-specific entropy thresholds.", "tab:entropy-calibration-thresholds"),
    )
    atomic_text(
        manuscript / "matched_quantization_generation.tex",
        latex_table(quant_main, "Matched Q4_K_M and Q8_0 generation outcomes.", "tab:matched-quantization-generation"),
    )
    build_generation_figures(
        output, entropy_trials, entropy_summary, quantization_trials, quantization_pairs
    )

    artifact_rows = []
    for path in sorted(tables):
        artifact_rows.append(
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "source": "model-backed generation JSON ledgers",
                "command": ".venv/bin/python scripts/analyze_revision_v3_generation.py",
            }
        )
    atomic_csv(analysis / "derived_artifact_manifest.csv", pd.DataFrame(artifact_rows))
    print(json.dumps({"status": "pass", "artifacts": len(artifact_rows), "validation": audit["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
