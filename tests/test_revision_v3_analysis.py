import json

import numpy as np
import pandas as pd

from rankcloak.revision_v3_analysis import (
    fit_surprisal_detector,
    generation_surprisal_features,
    levenshtein_distance,
    normalized_edit_distance,
    token_jaccard,
)


def test_myers_levenshtein_matches_standard_examples_and_symmetry():
    examples = [
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("RankCloak", "rank cloak", 3),
        ("same", "same", 0),
        ("\u00e9", "e", 1),
    ]
    for left, right, expected in examples:
        assert levenshtein_distance(left, right) == expected
        assert levenshtein_distance(right, left) == expected
    assert normalized_edit_distance("same", "same") == 0.0


def test_token_jaccard_is_transparent_casefolded_word_type_overlap():
    assert token_jaccard("A cat sat", "a dog sat") == 0.5
    assert token_jaccard("", "") == 1.0


def test_generation_surprisal_features_share_schema_between_record_types():
    clean = {
        "record_type": "ordinary_control",
        "generation": {"token_log_probabilities": [-0.1, -0.2, -0.3]},
    }
    stego = {
        "record_type": "rankcloak_trial",
        "segments": [
            {
                "leadin_log_probabilities": [-0.1],
                "forced_log_probabilities": [-2.0],
                "tail_log_probabilities": [-0.3],
            }
        ],
    }
    clean_features = generation_surprisal_features(clean)
    stego_features = generation_surprisal_features(stego)
    assert set(clean_features) == set(stego_features)
    assert clean_features["trace_token_count"] == 3.0
    assert stego_features["surprisal_mean"] > clean_features["surprisal_mean"]


def test_surprisal_detector_tunes_only_with_validation_scores():
    columns = {
        "row_id": ["row-{}".format(i) for i in range(40)],
        "label": [0] * 20 + [1] * 20,
        "source_record_kind": ["ordinary_control"] * 20 + ["rankcloak_trial"] * 20,
        "surprisal_mean": list(np.linspace(0.0, 0.3, 20)) + list(np.linspace(0.7, 1.0, 20)),
        "trace_token_count": [20.0] * 40,
    }
    train = pd.DataFrame(columns)
    validation = train.copy()
    validation["row_id"] = ["val-{}".format(i) for i in range(40)]
    test = train.copy()
    test["row_id"] = ["test-{}".format(i) for i in range(40)]
    result = fit_surprisal_detector(train, validation, test, c_grid=(0.1, 1.0), seed=9)
    assert result["selected_C"] in {0.1, 1.0}
    assert len(result["validation_scores"]) == 40
    assert len(result["test_scores"]) == 40
    assert result["test_tuning"] is False
