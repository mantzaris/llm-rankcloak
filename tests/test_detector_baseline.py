import pandas as pd

from rankcloak.detection import prepare_detector_dataset, run_detector_baselines


def tiny_feature_frame():
    rows = []
    for index in range(6):
        rows.append(
            {
                "source_type": "baseline",
                "trial_id": "b{}".format(index),
                "prompt_family": "recipe",
                "payload_class": None,
                "mean_token_log_probability": -1.0 - index * 0.01,
                "token_count": 20 + index,
                "character_count": 100 + index,
                "repeated_token_fraction": 0.1,
                "artifact_count_total": 0,
            }
        )
        rows.append(
            {
                "source_type": "nonseg_rankcloak",
                "trial_id": "r{}".format(index),
                "prompt_family": "recipe",
                "payload_class": "sha256_hex",
                "mean_token_log_probability": -5.0 - index * 0.01,
                "token_count": 30 + index,
                "character_count": 130 + index,
                "repeated_token_fraction": 0.2,
                "artifact_count_total": 1,
            }
        )
    return pd.DataFrame(rows)


def test_detector_dataset_creation_labels_are_binary():
    dataset = prepare_detector_dataset(tiny_feature_frame())
    assert not dataset.empty
    assert set(dataset["label"].unique()) == {0, 1}


def test_detector_baseline_handles_optional_sklearn():
    dataset = prepare_detector_dataset(tiny_feature_frame())
    results = run_detector_baselines(dataset)
    assert not results.empty
    assert "threshold_mean_token_log_probability" in set(results["detector_name"])
