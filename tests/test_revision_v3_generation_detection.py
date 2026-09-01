from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rankcloak.revision_v3_analysis import (
    RevisionV3AnalysisError,
    generation_surprisal_features,
    surprisal_features_from_log_probabilities,
)
from rankcloak.revision_v3_generation_detection import locked_partition_deduplicate


def _row(
    row_id: str,
    pair_id: str,
    payload: str,
    partition: str,
    label: int,
    text: str,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "pair_id": pair_id,
        "payload_group_id": payload,
        "partition": partition,
        "label": label,
        "text": text,
        "model_id": "model",
        "representation_name": "ascii_b8",
        "payload_class": "sha256_hex",
    }


def test_locked_dedup_preserves_test_and_removes_complete_exact_pair() -> None:
    frame = pd.DataFrame(
        [
            _row("train-control", "train-pair", "train-payload", "train", 0, "The same normalized sentence."),
            _row("train-rank", "train-pair", "train-payload", "train", 1, "A separate encoded train sentence."),
            _row("train2-control", "train2-pair", "train2-payload", "train", 0, "A clean train control with enough words."),
            _row("train2-rank", "train2-pair", "train2-payload", "train", 1, "A clean encoded train example with enough words."),
            _row("val-control", "val-pair", "val-payload", "validation", 0, "A clean validation control with enough words."),
            _row("val-rank", "val-pair", "val-payload", "validation", 1, "A clean encoded validation example with enough words."),
            _row("test-control", "test-pair", "test-payload", "test", 0, "  THE same\t normalized sentence. "),
            _row("test-rank", "test-pair", "test-payload", "test", 1, "A separate encoded test sentence."),
        ]
    )
    result = locked_partition_deduplicate(frame, threshold=1.0)
    assert set(result.frame["row_id"]) == {
        "train2-control", "train2-rank", "val-control", "val-rank",
        "test-control", "test-rank",
    }
    assert set(result.removed_rows["row_id"]) == {"train-control", "train-rank"}
    assert result.audit["status"] == "pass"
    assert result.audit["checks"]["normalized_text_sha256_cross_partition"] == 0


def test_locked_dedup_resolves_cross_partition_near_component_by_payload() -> None:
    base = "A carefully documented weather explanation discusses pressure clouds wind and rainfall over many ordinary words"
    frame = pd.DataFrame(
        [
            _row("train-c", "train-p", "train-g", "train", 0, "Training control is unrelated and sufficiently descriptive."),
            _row("train-r", "train-p", "train-g", "train", 1, "Training encoded text is unrelated and sufficiently descriptive."),
            _row("val-c", "val-p", "val-g", "validation", 0, base + " today"),
            _row("val-r", "val-p", "val-g", "validation", 1, "Validation encoded companion remains a distinct sentence."),
            _row("val2-c", "val2-p", "val2-g", "validation", 0, "Second validation control remains after filtering."),
            _row("val2-r", "val2-p", "val2-g", "validation", 1, "Second validation encoded row remains after filtering."),
            _row("test-c", "test-p", "test-g", "test", 0, base + " tonight"),
            _row("test-r", "test-p", "test-g", "test", 1, "Test encoded companion remains a distinct sentence."),
        ]
    )
    result = locked_partition_deduplicate(frame, threshold=0.90)
    assert not set(result.frame["payload_group_id"]) & {"val-g"}
    assert {"test-c", "test-r"}.issubset(set(result.frame["row_id"]))
    assert set(result.removed_rows.loc[result.removed_rows["payload_group_id"].eq("val-g"), "row_id"]) == {"val-c", "val-r"}
    assert result.audit["cross_partition_near_components_resolved"] >= 1
    assert result.audit["checks"]["near_pair_cross_partition"] == 0


def test_surprisal_array_helper_matches_record_extractor() -> None:
    logp = [-0.1, -0.5, -2.0, -4.0]
    direct = surprisal_features_from_log_probabilities(logp)
    record = {
        "record_type": "ordinary_control",
        "generation": {"token_log_probabilities": logp},
    }
    assert generation_surprisal_features(record) == direct
    assert direct["trace_token_count"] == 4.0
    assert np.isclose(direct["surprisal_mean"], 1.65)


@pytest.mark.parametrize("values", [[], [0.0, float("nan")], [float("inf")]])
def test_surprisal_array_helper_rejects_unusable_trace(values: list[float]) -> None:
    with pytest.raises(RevisionV3AnalysisError):
        surprisal_features_from_log_probabilities(values)
