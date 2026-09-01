"""Reusable analyses for the RankCloak revision-V3 computational extension."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .revision_v3_dedup import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    _UnionFind,
    find_near_duplicate_pairs,
    normalize_visible_text,
)


SCHEMA_VERSION = "rankcloak-revision-v3-analysis-v1"
SURPRISAL_FEATURE_VERSION = "shared_generation_log_probability_summaries_v1"
HUMAN_CONTROL_VERSION = "dolly_automatically_screened_length_topic_matched_v1"
TOPIC_VARIABILITY_VERSION = "paired_segmented_single_multi_topic_diversity_v1"
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


class RevisionV3AnalysisError(ValueError):
    """Raised when source records cannot support an analysis as declared."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise RevisionV3AnalysisError(
                    "Invalid JSON at {} line {}: {}".format(path, line_number, exc)
                ) from exc
            if not isinstance(value, dict):
                raise RevisionV3AnalysisError("JSONL records must be objects")
            rows.append(value)
    return rows


def _finite_log_probabilities(record: Mapping[str, object]) -> list[float]:
    record_type = str(record.get("record_type", ""))
    values: list[float] = []
    if record_type == "ordinary_control":
        generation = record.get("generation")
        if isinstance(generation, Mapping):
            values.extend(generation.get("token_log_probabilities", []))
    elif record_type == "rankcloak_trial":
        segments = record.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                for field in (
                    "leadin_log_probabilities",
                    "forced_log_probabilities",
                    "tail_log_probabilities",
                ):
                    raw = segment.get(field, [])
                    if isinstance(raw, list):
                        values.extend(raw)
    result = [float(value) for value in values if math.isfinite(float(value))]
    if not result:
        raise RevisionV3AnalysisError(
            "Record {} lacks shared token log probabilities".format(
                record.get("work_id", record.get("trial_id", "unknown"))
            )
        )
    return result


def generation_surprisal_features(record: Mapping[str, object]) -> Mapping[str, float]:
    """Extract only features available for both RankCloak and clean controls."""

    return surprisal_features_from_log_probabilities(_finite_log_probabilities(record))


def entropy_embedding_log_probabilities(
    generation: Mapping[str, object],
) -> Sequence[float]:
    """Return the harmonized entropy-study embedding-span probability trace."""

    if "embedding_log_probabilities" in generation:
        return generation["embedding_log_probabilities"]
    if "forced_log_probabilities" in generation:
        return generation["forced_log_probabilities"]
    raise RevisionV3AnalysisError(
        "Entropy RankCloak record lacks an embedding-span log-probability trace"
    )


def surprisal_features_from_log_probabilities(
    log_probabilities: Sequence[float],
) -> Mapping[str, float]:
    """Summarize an exact-model token trace using the frozen attacker features."""

    logp = np.asarray(list(log_probabilities), dtype=np.float64)
    if logp.ndim != 1 or not len(logp) or not np.isfinite(logp).all():
        raise RevisionV3AnalysisError(
            "Surprisal features require a non-empty finite log-probability trace"
        )
    surprisal = -logp
    first = surprisal[: max(1, len(surprisal) // 2)]
    last = surprisal[len(surprisal) // 2 :]
    result: Dict[str, float] = {
        "trace_token_count": float(len(surprisal)),
        "surprisal_mean": float(np.mean(surprisal)),
        "surprisal_std": float(np.std(surprisal, ddof=0)),
        "surprisal_min": float(np.min(surprisal)),
        "surprisal_max": float(np.max(surprisal)),
        "surprisal_mean_absolute_step": float(
            np.mean(np.abs(np.diff(surprisal))) if len(surprisal) > 1 else 0.0
        ),
        "surprisal_first_half_mean": float(np.mean(first)),
        "surprisal_last_half_mean": float(np.mean(last)),
    }
    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        result["surprisal_q{:02d}".format(int(quantile * 100))] = float(
            np.quantile(surprisal, quantile)
        )
    for boundary in (2, 4, 6, 8, 10):
        result["surprisal_fraction_ge_{}".format(boundary)] = float(
            np.mean(surprisal >= boundary)
        )
    return result


def rank_pressure_summary(record: Mapping[str, object]) -> Mapping[str, object]:
    """Return descriptive stego-only rank pressure; never used as a classifier input."""

    ranks = []
    gaps = []
    for segment in record.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        ranks.extend(
            int(value)
            for value in segment.get("realized_ranks", [])
            if value is not None
        )
        forced = list(map(float, segment.get("forced_log_probabilities", [])))
        greedy = list(map(float, segment.get("greedy_log_probabilities", [])))
        if len(forced) == len(greedy):
            gaps.extend(greedy_value - forced_value for forced_value, greedy_value in zip(forced, greedy))
    if not ranks:
        return {
            "forced_token_count": 0,
            "mean_forced_rank": None,
            "median_forced_rank": None,
            "maximum_forced_rank": None,
            "mean_rank_pressure_nats": None,
        }
    return {
        "forced_token_count": int(len(ranks)),
        "mean_forced_rank": float(np.mean(ranks)),
        "median_forced_rank": float(np.median(ranks)),
        "maximum_forced_rank": int(max(ranks)),
        "mean_rank_pressure_nats": float(np.mean(gaps)) if gaps else None,
    }


def load_generation_feature_frame(
    detector_frame: pd.DataFrame, record_paths: Sequence[Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join detector observations to saved generation traces without fuzzy keys."""

    trial_records: Dict[str, Mapping[str, object]] = {}
    control_records: Dict[str, Mapping[str, object]] = {}
    rank_rows = []
    for path in record_paths:
        for record in read_jsonl(Path(path)):
            record_type = str(record.get("record_type", ""))
            if record_type == "rankcloak_trial":
                identity = str(record.get("trial_id", ""))
                if not identity or identity in trial_records:
                    raise RevisionV3AnalysisError("Missing or repeated trial_id")
                trial_records[identity] = record
                rank_rows.append(
                    {
                        "source_trial_id": identity,
                        "model_id": str(record.get("model_id", "")),
                        "codec_id": str(record.get("protocol_variant", "")),
                        "payload_class": str(record.get("payload_class", "")),
                        **rank_pressure_summary(record),
                    }
                )
            elif record_type == "ordinary_control":
                identity = str(record.get("control_id", ""))
                if not identity or identity in control_records:
                    raise RevisionV3AnalysisError("Missing or repeated control_id")
                control_records[identity] = record

    feature_rows = []
    missing = []
    for row in detector_frame.to_dict("records"):
        if int(row["label"]) == 1:
            identity = str(row.get("source_trial_id", ""))
            record = trial_records.get(identity)
            source_kind = "rankcloak_trial"
        else:
            identity = str(row.get("control_id", ""))
            record = control_records.get(identity)
            source_kind = "ordinary_control"
        if record is None:
            missing.append((str(row["row_id"]), source_kind, identity))
            continue
        features = generation_surprisal_features(record)
        feature_rows.append(
            {
                "row_id": str(row["row_id"]),
                "label": int(row["label"]),
                "source_record_kind": source_kind,
                **features,
            }
        )
    if missing:
        raise RevisionV3AnalysisError(
            "Generation trace join missed {} rows; first {}".format(
                len(missing), missing[0]
            )
        )
    features = pd.DataFrame(feature_rows)
    if len(features) != len(detector_frame) or features["row_id"].duplicated().any():
        raise RevisionV3AnalysisError("Generation feature join is not one-to-one")
    return features, pd.DataFrame(rank_rows)


def fit_surprisal_detector(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 20260831,
) -> Mapping[str, object]:
    """Tune a model-aware logistic detector on validation ROC AUC only."""

    excluded = {"row_id", "label", "source_record_kind"}
    feature_columns = sorted(column for column in train.columns if column not in excluded)
    if not feature_columns:
        raise RevisionV3AnalysisError("No numeric surprisal features were supplied")
    if set(feature_columns) != set(test.columns) - excluded or set(feature_columns) != set(validation.columns) - excluded:
        raise RevisionV3AnalysisError("Surprisal feature schemas differ across partitions")
    x_train = train[feature_columns].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy(dtype=int)
    x_validation = validation[feature_columns].to_numpy(dtype=np.float64)
    y_validation = validation["label"].to_numpy(dtype=int)
    x_test = test[feature_columns].to_numpy(dtype=np.float64)
    candidates = []
    fitted: Dict[float, Pipeline] = {}
    for raw_c in c_grid:
        c_value = float(raw_c)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2000,
                        solver="liblinear",
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_validation)[:, 1]
        auc = float(roc_auc_score(y_validation, scores))
        candidates.append({"C": c_value, "validation_roc_auc": auc})
        fitted[c_value] = model
    selected = min(candidates, key=lambda item: (-item["validation_roc_auc"], item["C"]))
    selected_c = float(selected["C"])
    model = fitted[selected_c]
    validation_scores = model.predict_proba(x_validation)[:, 1].astype(np.float64)
    test_scores = model.predict_proba(x_test)[:, 1].astype(np.float64)
    classifier = model.named_steps["classifier"]
    return {
        "detector_name": "model_aware_surprisal_logistic",
        "information_access": (
            "saved per-token log probabilities from the exact generation model for "
            "both RankCloak and matched ordinary controls"
        ),
        "feature_version": SURPRISAL_FEATURE_VERSION,
        "feature_columns": feature_columns,
        "candidates": candidates,
        "selected_C": selected_c,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "coefficient_by_feature": {
            feature: float(value)
            for feature, value in zip(feature_columns, classifier.coef_[0])
        },
        "intercept": float(classifier.intercept_[0]),
        "test_tuning": False,
    }


def levenshtein_distance(left: str, right: str) -> int:
    """Myers bit-vector edit distance for arbitrary-length Unicode strings."""

    pattern = str(left)
    text = str(right)
    if not pattern:
        return len(text)
    if not text:
        return len(pattern)
    if len(pattern) > len(text):
        pattern, text = text, pattern
    masks: Dict[str, int] = {}
    for index, character in enumerate(pattern):
        masks[character] = masks.get(character, 0) | (1 << index)
    pv = ~0
    mv = 0
    score = len(pattern)
    last = 1 << (len(pattern) - 1)
    for character in text:
        eq = masks.get(character, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << 1) | 1
        mh <<= 1
        pv = mh | ~(xv | ph)
        mv = ph & xv
    return int(score)


def normalized_edit_distance(left: str, right: str) -> float:
    denominator = max(len(str(left)), len(str(right)))
    if denominator == 0:
        return 0.0
    return float(levenshtein_distance(str(left), str(right)) / denominator)


def token_jaccard(left: str, right: str) -> float:
    a = set(token.casefold() for token in WORD_PATTERN.findall(str(left)))
    b = set(token.casefold() for token in WORD_PATTERN.findall(str(right)))
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))


def build_topic_variability_pairs(record_paths: Sequence[Path]) -> pd.DataFrame:
    """Pair fixed-codec single- and multi-topic trials by model and payload."""

    selected: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    protocols = {
        "segmented_hex_single_topic": "single_topic",
        "segmented_hex_multi_topic": "multi_topic",
    }
    for path in record_paths:
        for record in read_jsonl(Path(path)):
            if record.get("record_type") != "rankcloak_trial":
                continue
            protocol = str(record.get("protocol_variant", ""))
            if protocol not in protocols:
                continue
            key = (str(record["model_id"]), str(record["payload_name"]))
            condition = protocols[protocol]
            cell = selected.setdefault(key, {})
            if condition in cell:
                raise RevisionV3AnalysisError("Repeated topic-condition trial")
            cell[condition] = record
    rows = []
    for key in sorted(selected):
        cell = selected[key]
        if set(cell) != {"single_topic", "multi_topic"}:
            raise RevisionV3AnalysisError("Incomplete paired topic condition")
        single = cell["single_topic"]
        multi = cell["multi_topic"]
        single_text = str(single["full_text"])
        multi_text = str(multi["full_text"])
        identity = "{}\x1f{}".format(*key)
        rows.append(
            {
                "pair_id": "topic-pair-{}".format(
                    hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                ),
                "model_id": key[0],
                "payload_name": key[1],
                "payload_class": str(single["payload_class"]),
                "single_trial_id": str(single["trial_id"]),
                "multi_trial_id": str(multi["trial_id"]),
                "exact_outputs_unique": bool(single_text != multi_text),
                "normalized_outputs_unique": bool(
                    normalize_visible_text(single_text)
                    != normalize_visible_text(multi_text)
                ),
                "normalized_character_edit_distance": normalized_edit_distance(
                    single_text, multi_text
                ),
                "token_jaccard_similarity": token_jaccard(single_text, multi_text),
                "single_character_count": int(len(single_text)),
                "multi_character_count": int(len(multi_text)),
                "single_full_token_count": int(single["full_token_count"]),
                "multi_full_token_count": int(multi["full_token_count"]),
                "single_saved_id_exact_recovery": bool(
                    single.get("saved_token_id_replay", {}).get(
                        "exact_payload_recovery", False
                    )
                ),
                "multi_saved_id_exact_recovery": bool(
                    multi.get("saved_token_id_replay", {}).get(
                        "exact_payload_recovery", False
                    )
                ),
                "single_visible_text_exact_recovery": single.get(
                    "text_retokenization_replay", {}
                ).get("exact_payload_recovery"),
                "multi_visible_text_exact_recovery": multi.get(
                    "text_retokenization_replay", {}
                ).get("exact_payload_recovery"),
                "single_text_sha256": hashlib.sha256(single_text.encode("utf-8")).hexdigest(),
                "multi_text_sha256": hashlib.sha256(multi_text.encode("utf-8")).hexdigest(),
                "deterministic_example_order": hashlib.sha256(
                    ("example\x1f" + identity).encode("utf-8")
                ).hexdigest(),
                "single_excerpt": single_text[:400].replace("\n", " "),
                "multi_excerpt": multi_text[:400].replace("\n", " "),
            }
        )
    return pd.DataFrame(rows)


def grouped_mean_interval(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    *,
    resamples: int = 2000,
    seed: int = 20260831,
) -> Mapping[str, object]:
    groups = sorted(frame[group_column].astype(str).unique())
    by_group = {
        group: frame.loc[frame[group_column].astype(str).eq(group), value_column]
        .astype(float)
        .to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(resamples)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        observations = np.concatenate([by_group[group] for group in sampled])
        values.append(float(np.mean(observations)))
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "mean": float(frame[value_column].astype(float).mean()),
        "ci_low_95": float(low),
        "ci_high_95": float(high),
        "n_observations": int(len(frame)),
        "n_groups": int(len(groups)),
        "bootstrap_unit": group_column,
        "bootstrap_resamples": int(resamples),
    }


def _cross_corpus_near_pairs(
    primary: pd.DataFrame,
    human: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    primary_text = primary["text"].map(normalize_visible_text).astype(str).tolist()
    human_text = human["text"].map(normalize_visible_text).astype(str).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )
    combined = vectorizer.fit_transform(primary_text + human_text)
    primary_matrix = combined[: len(primary)]
    human_matrix = combined[len(primary) :]
    neighbors = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
        radius=max(0.0, 1.0 - float(threshold)) + 1e-12,
    ).fit(primary_matrix)
    distances, indices = neighbors.radius_neighbors(
        human_matrix, return_distance=True, sort_results=True
    )
    rows = []
    for human_position, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, primary_position in zip(row_distances, row_indices):
            similarity = 1.0 - float(distance)
            if similarity + 1e-12 < float(threshold):
                continue
            rows.append(
                {
                    "human_row_id": str(human.iloc[human_position]["row_id"]),
                    "primary_row_id": str(primary.iloc[int(primary_position)]["row_id"]),
                    "primary_partition": str(primary.iloc[int(primary_position)]["partition"]),
                    "cosine_similarity": similarity,
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "human_row_id",
            "primary_row_id",
            "primary_partition",
            "cosine_similarity",
            "threshold",
        ],
    )


def prepare_human_control_evaluation(
    candidate_path: Path,
    primary_frame: pd.DataFrame,
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    maximum_relative_word_difference: float = 0.35,
) -> Mapping[str, object]:
    """Build a licensed, topic/length-matched, test-only human control sample.

    Only automatically screened Dolly candidates are used.  This is not a
    human-subject evaluation and does not override the separate manual-review
    gate for stimuli shown to research participants.
    """

    raw = read_jsonl(Path(candidate_path))
    candidates = [row for row in raw if row.get("eligible_for_manual_review") is True]
    if not candidates:
        raise RevisionV3AnalysisError("No automatically screened human candidates")
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "row_id": "human-{}".format(candidate["candidate_id"]),
                "candidate_id": str(candidate["candidate_id"]),
                "text": str(candidate["message_text"]),
                "label": 0,
                "payload_group_id": "human-{}".format(candidate["candidate_id"]),
                "pair_id": "human-{}".format(candidate["candidate_id"]),
                "prompt_template_id": str(candidate["assigned_revision_prompt_id"]),
                "model_id": "human_databricks_dolly_15k",
                "codec_id": "not_applicable_human_control",
                "payload_class": "not_applicable_human_control",
                "word_count": int(candidate["word_count"]),
                "character_count": int(candidate["character_count"]),
                "message_text_sha256": str(candidate["message_text_sha256"]),
                "canonical_text_sha256": str(candidate["canonical_text_sha256"]),
                "source_record_id": str(candidate["source_record_id"]),
                "source_record_sha256": str(candidate["source_record_sha256"]),
                "source_category": str(candidate["source_category"]),
                "source_dataset_revision": str(candidate["source_dataset_revision"]),
                "source_file_sha256": str(candidate["source_file_sha256"]),
                "license_identifier": str(candidate["license_identifier"]),
            }
        )
    human = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    human["normalized_text"] = human["text"].map(normalize_visible_text)
    human["normalized_text_sha256"] = human["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    exact_groups = human.loc[
        human["normalized_text_sha256"].duplicated(False)
    ].copy()
    human = human.drop_duplicates("normalized_text_sha256", keep="first").reset_index(drop=True)
    near_pairs, tfidf_features = find_near_duplicate_pairs(human, threshold=threshold)
    union = _UnionFind(human["row_id"].astype(str))
    for pair in near_pairs.to_dict("records"):
        union.union(str(pair["left_row_id"]), str(pair["right_row_id"]))
    components: Dict[str, list[str]] = {}
    for row_id in human["row_id"].astype(str):
        components.setdefault(union.find(row_id), []).append(row_id)
    cluster_map = {}
    for members in components.values():
        identity = hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:24]
        for row_id in members:
            cluster_map[row_id] = "human-dedup-cluster-{}".format(identity)
    human["dedup_cluster_id"] = human["row_id"].map(cluster_map)
    human["partition"] = "human_test_only"

    cross_pairs = _cross_corpus_near_pairs(primary_frame, human, float(threshold))
    excluded_cross_ids = set(cross_pairs["human_row_id"].astype(str))
    eligible = human.loc[~human["row_id"].astype(str).isin(excluded_cross_ids)].copy()

    targets = primary_frame.loc[
        primary_frame["partition"].eq("test") & primary_frame["label"].eq(1)
    ].copy()
    targets["word_count"] = targets["text"].map(lambda value: len(WORD_PATTERN.findall(str(value))))
    targets["character_count"] = targets["text"].astype(str).str.len()
    match_rows = []
    selected_ids = []
    for template in sorted(eligible["prompt_template_id"].unique()):
        candidate_cell = eligible.loc[
            eligible["prompt_template_id"].eq(template)
        ].sort_values("row_id")
        target_cell = targets.loc[
            targets["prompt_template_id"].eq(template)
        ].sort_values("row_id")
        if candidate_cell.empty or target_cell.empty:
            continue
        candidate_records = candidate_cell.to_dict("records")
        target_records = target_cell.to_dict("records")
        cost = np.zeros((len(candidate_records), len(target_records)), dtype=np.float64)
        for i, candidate in enumerate(candidate_records):
            for j, target in enumerate(target_records):
                word_difference = abs(candidate["word_count"] - target["word_count"]) / max(
                    1, int(target["word_count"])
                )
                character_difference = abs(
                    candidate["character_count"] - target["character_count"]
                ) / max(1, int(target["character_count"]))
                cost[i, j] = word_difference + 0.25 * character_difference
        candidate_indices, target_indices = linear_sum_assignment(cost)
        for candidate_index, target_index in zip(candidate_indices, target_indices):
            candidate = candidate_records[int(candidate_index)]
            target = target_records[int(target_index)]
            relative_word_difference = abs(
                candidate["word_count"] - target["word_count"]
            ) / max(1, int(target["word_count"]))
            if relative_word_difference > float(maximum_relative_word_difference):
                continue
            selected_ids.append(str(candidate["row_id"]))
            match_rows.append(
                {
                    "human_row_id": str(candidate["row_id"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "prompt_template_id": str(template),
                    "matched_rankcloak_test_row_id": str(target["row_id"]),
                    "human_word_count": int(candidate["word_count"]),
                    "rankcloak_word_count": int(target["word_count"]),
                    "relative_word_difference": float(relative_word_difference),
                    "human_character_count": int(candidate["character_count"]),
                    "rankcloak_character_count": int(target["character_count"]),
                    "matching_cost": float(cost[int(candidate_index), int(target_index)]),
                }
            )
    selected = eligible.loc[eligible["row_id"].astype(str).isin(selected_ids)].copy()
    selected = selected.sort_values("row_id").reset_index(drop=True)
    matches = pd.DataFrame(match_rows).sort_values("human_row_id").reset_index(drop=True)
    if selected.empty or len(selected) != len(matches):
        raise RevisionV3AnalysisError("Human matching produced no sample or duplicate matches")
    manifest_columns = [
        "row_id",
        "candidate_id",
        "prompt_template_id",
        "word_count",
        "character_count",
        "message_text_sha256",
        "canonical_text_sha256",
        "source_record_id",
        "source_record_sha256",
        "source_category",
        "source_dataset_revision",
        "source_file_sha256",
        "license_identifier",
        "dedup_cluster_id",
        "partition",
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "human_control_version": HUMAN_CONTROL_VERSION,
        "source_records": int(len(raw)),
        "automatically_screened_candidates": int(len(candidates)),
        "normalized_exact_duplicate_groups": int(
            exact_groups["normalized_text_sha256"].nunique()
        ),
        "normalized_exact_duplicate_rows_removed": int(
            len(candidates) - len(human)
        ),
        "near_duplicate_pairs_within_human": int(len(near_pairs)),
        "near_duplicate_clusters_within_human": int(
            sum(len(members) > 1 for members in components.values())
        ),
        "tfidf_feature_count": int(tfidf_features),
        "cross_corpus_near_duplicate_pairs": int(len(cross_pairs)),
        "human_rows_excluded_for_cross_corpus_similarity": int(len(excluded_cross_ids)),
        "selected_human_controls": int(len(selected)),
        "selected_template_count": int(selected["prompt_template_id"].nunique()),
        "maximum_relative_word_difference": float(maximum_relative_word_difference),
        "near_duplicate_threshold": float(threshold),
        "partition_policy": "all human near-duplicate clusters assigned to secondary human test only",
        "threshold_selection_uses_human_labels": False,
        "manual_review_status": (
            "not reviewed for participant-facing stimulus use; computational secondary control only"
        ),
    }
    return {
        "selected_with_text": selected,
        "selection_manifest": selected[manifest_columns].copy(),
        "matches": matches,
        "exact_duplicate_rows": exact_groups.drop(columns=["text", "normalized_text"], errors="ignore"),
        "near_duplicate_pairs": near_pairs,
        "cross_corpus_near_pairs": cross_pairs,
        "summary": summary,
    }


__all__ = [
    "HUMAN_CONTROL_VERSION",
    "RevisionV3AnalysisError",
    "SCHEMA_VERSION",
    "SURPRISAL_FEATURE_VERSION",
    "TOPIC_VARIABILITY_VERSION",
    "build_topic_variability_pairs",
    "file_sha256",
    "fit_surprisal_detector",
    "generation_surprisal_features",
    "entropy_embedding_log_probabilities",
    "grouped_mean_interval",
    "levenshtein_distance",
    "load_generation_feature_frame",
    "normalized_edit_distance",
    "prepare_human_control_evaluation",
    "rank_pressure_summary",
    "read_jsonl",
    "surprisal_features_from_log_probabilities",
    "token_jaccard",
]
