import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

import rankcloak.revision_detection as revision_detection
from rankcloak.revision_artifacts import canonical_json_sha256, file_sha256
from rankcloak.revision_detection import (
    CONFIRMATORY_TRANSFORMER_ARTIFACTS,
    CONFIRMATORY_TRANSFORMER_MODEL_ID,
    CONFIRMATORY_TRANSFORMER_RELATIVE_PATH,
    CONFIRMATORY_TRANSFORMER_REVISION,
    DetectorSplit,
    RevisionDetectionError,
    _run_torch_text_cnn,
    assert_no_split_leakage,
    build_evaluation_splits,
    deterministic_model_state_sha256,
    deterministic_payload_group_split,
    grouped_bootstrap_detector_metrics,
    normalize_detector_frame,
    run_configured_detector,
    run_revision_detector_suite,
    validate_confirmatory_detector_frame,
    verify_pinned_model_artifacts,
)
from scripts.run_revision_detectors import (
    PREPROCESSING_BINDING_SCHEMA,
    _verify_primary_preprocessing_detector_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detector_frame(group_count=12):
    rows = []
    for group_index in range(group_count):
        for label in (0, 1):
            style = "ordinary calm cover prose" if label == 0 else "forced rank stego prose"
            rows.append(
                {
                    "row_id": "row-{}-{}".format(group_index, label),
                    "text": "{} unique message {} {}".format(style, group_index, label),
                    "label": label,
                    "payload_group_id": "payload-{}".format(group_index),
                    "prompt_template_id": "template-{}".format(group_index % 2),
                    "model_id": "model-{}".format(group_index % 3),
                    "codec_id": "codec-{}".format(group_index % 2),
                }
            )
    return pd.DataFrame(rows)


def crossed_primary_detector_frame():
    prompts = ["prompt-{:02d}".format(index) for index in range(18)]
    models = ["model-{}".format(index) for index in range(3)]
    codecs = ["codec-{}".format(index) for index in range(6)]
    rows = []
    pair_index = 0
    for group_index in range(480):
        pair_count = 17 if group_index < 240 else 16
        for condition_index in range(pair_count):
            for label in (0, 1):
                rows.append(
                    {
                        "row_id": "row-{}-{}".format(pair_index, label),
                        "text": "unique primary message {} {}".format(pair_index, label),
                        "label": label,
                        "payload_group_id": "payload-{:03d}".format(group_index),
                        "prompt_template_id": prompts[
                            (group_index + 3 * condition_index) % len(prompts)
                        ],
                        "model_id": models[condition_index % len(models)],
                        "codec_id": codecs[condition_index % len(codecs)],
                    }
                )
            pair_index += 1
    assert pair_index == 7920
    return pd.DataFrame(rows), prompts, models, codecs


def smoke_config(detectors=None, regimes=None):
    return {
        "schema_version": "rankcloak-revision-detectors-v1",
        "seed": 1234,
        "columns": {},
        "splits": {
            "regimes": regimes or ["matched"],
            "matched_test_fraction": 0.25,
            "assert_text_hash_disjoint": True,
            "minimum_train_rows": 4,
            "minimum_test_rows": 2,
            "fail_on_skipped_split": True,
        },
        "bootstrap": {"resamples": 20, "smoke_resamples": 10},
        "detectors": detectors
        or [{"name": "cnn-request", "kind": "text_cnn", "fallback": "hashed_ngram_smoke"}],
    }


def pinned_fixture_directory(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    contents = {
        "config.json": b'{"model_type":"fake"}\n',
        "pytorch_model.bin": b"tiny fake weights\x00\x01",
        "spm.model": b"tiny sentencepiece bytes",
        "tokenizer_config.json": b'{"vocab_type":"spm"}\n',
    }
    expected = {}
    for name, content in contents.items():
        (directory / name).write_bytes(content)
        expected[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    return directory, expected


def primary_preprocessing_binding_fixture(tmp_path):
    detector = tmp_path / "preprocessed" / "detector_corpus.jsonl"
    detector.parent.mkdir(parents=True)
    detector.write_text('{"label":0}\n{"label":1}\n', encoding="utf-8")
    input_manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_inputs",
        "strict_complete": True,
        "emitted_run_count": 3,
        "reference_run_count": 0,
        "run_shards": [
            {
                "role": "input",
                "stage": "primary_v2",
                "model_id": model_id,
                "evidence_status": (
                    "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
                ),
                "planned_work_units": 2,
                "completed_work_units": 2,
            }
            for model_id in ("model-0", "model-1", "model-2")
        ],
        "input_files": [],
        "input_files_sha256": canonical_json_sha256([]),
    }
    input_path = detector.parent / "preprocessing_input_manifest.json"
    input_path.write_text(json.dumps(input_manifest), encoding="utf-8")
    outputs = [
        {
            "role": "detector",
            "path": detector.name,
            "sha256": file_sha256(detector),
            "size_bytes": detector.stat().st_size,
            "row_count": 2,
        },
        {
            "role": "input_manifest",
            "path": input_path.name,
            "sha256": file_sha256(input_path),
            "size_bytes": input_path.stat().st_size,
            "row_count": None,
        },
    ]
    output_manifest = {
        "schema_version": "2.0",
        "manifest_type": "revision_preprocessing_outputs",
        "input_manifest_sha256": file_sha256(input_path),
        "outputs": outputs,
        "outputs_sha256": canonical_json_sha256(outputs),
        "row_counts": {"detector": 2},
        "invariants": {
            "detector_pair_count": 1,
            "detector_grouping_unit": "payload_name",
        },
    }
    output_path = detector.parent / "preprocessing_output_manifest.json"
    output_path.write_text(json.dumps(output_manifest), encoding="utf-8")
    contract = {
        "rows": 2,
        "positive_rows": 1,
        "model_ids": ["model-0", "model-1", "model-2"],
    }
    return detector, output_path, contract


def test_normalization_requires_payload_group():
    frame = detector_frame().drop(columns=["payload_group_id"])
    with pytest.raises(RevisionDetectionError, match="payload_group_id"):
        normalize_detector_frame(frame)


def test_confirmatory_detector_input_is_bound_to_primary_preprocessing(tmp_path):
    detector, manifest, contract = primary_preprocessing_binding_fixture(tmp_path)
    binding = _verify_primary_preprocessing_detector_input(
        detector, manifest, contract
    )
    assert binding["schema_version"] == PREPROCESSING_BINDING_SCHEMA
    assert binding["detector_sha256"] == file_sha256(detector)
    assert binding["detector_row_count"] == 2
    assert binding["strict_complete"] is True
    detector.write_text(detector.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(RevisionDetectionError, match="exact detector artifact"):
        _verify_primary_preprocessing_detector_input(detector, manifest, contract)


def test_confirmatory_detector_rehashes_every_declared_preprocessing_input(tmp_path):
    detector, manifest_path, contract = primary_preprocessing_binding_fixture(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_manifest_path = detector.parent / "preprocessing_input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    source = detector.parent / "sealed-upstream-shard.jsonl"
    source.write_bytes(b'{"sealed":true}\n')
    input_manifest["input_files"] = [
        {
            "path": source.name,
            "sha256": file_sha256(source),
            "size_bytes": int(source.stat().st_size),
            "role": "primary_shard",
            "run_identity_sha256": "f" * 64,
        }
    ]
    input_manifest["input_files_sha256"] = canonical_json_sha256(
        input_manifest["input_files"]
    )
    input_manifest_path.write_text(json.dumps(input_manifest), encoding="utf-8")
    for row in manifest["outputs"]:
        if row["role"] == "input_manifest":
            row["sha256"] = file_sha256(input_manifest_path)
            row["size_bytes"] = int(input_manifest_path.stat().st_size)
    manifest["input_manifest_sha256"] = file_sha256(input_manifest_path)
    manifest["outputs_sha256"] = canonical_json_sha256(manifest["outputs"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binding = _verify_primary_preprocessing_detector_input(
        detector, manifest_path, contract
    )
    assert binding["verified_input_file_count"] == 1
    source.write_bytes(b'{"sealed":false}\n')
    with pytest.raises(RevisionDetectionError, match="input-file bytes differ"):
        _verify_primary_preprocessing_detector_input(
            detector, manifest_path, contract
        )


def test_detector_cli_help_exposes_preprocessing_manifest_binding():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_revision_detectors.py"),
            "--help",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--preprocessing-manifest" in completed.stdout


def test_normalization_rejects_duplicate_row_ids():
    frame = detector_frame()
    frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    with pytest.raises(RevisionDetectionError, match="unique"):
        normalize_detector_frame(frame)


def test_payload_group_split_is_deterministic_and_row_order_invariant():
    original = normalize_detector_frame(detector_frame())
    shuffled = normalize_detector_frame(detector_frame().sample(frac=1.0, random_state=77))
    train_a, test_a = deterministic_payload_group_split(original, seed=99)
    train_b, test_b = deterministic_payload_group_split(shuffled, seed=99)
    test_groups_a = set(original.iloc[test_a]["payload_group_id"])
    test_groups_b = set(shuffled.iloc[test_b]["payload_group_id"])
    assert test_groups_a == test_groups_b
    assert not set(original.iloc[train_a]["payload_group_id"]) & test_groups_a
    assert not set(shuffled.iloc[train_b]["payload_group_id"]) & test_groups_b


def test_split_assertion_rejects_identical_text_across_groups():
    frame = detector_frame(group_count=2)
    frame.loc[frame["payload_group_id"] == "payload-1", "text"] = frame.loc[
        frame["payload_group_id"] == "payload-0", "text"
    ].tolist()
    normalized = normalize_detector_frame(frame)
    train = tuple(normalized.index[normalized["payload_group_id"] == "payload-0"])
    test = tuple(normalized.index[normalized["payload_group_id"] == "payload-1"])
    split = DetectorSplit("manual", "matched", train, test)
    with pytest.raises(RevisionDetectionError, match="identical raw text"):
        assert_no_split_leakage(normalized, split)


def test_all_evaluation_regimes_keep_payload_groups_disjoint():
    normalized = normalize_detector_frame(detector_frame())
    splits, skipped = build_evaluation_splits(
        normalized,
        minimum_train_rows=4,
        minimum_test_rows=2,
        seed=101,
    )
    assert skipped == []
    assert {split.regime for split in splits} == {
        "matched",
        "held_out_template",
        "leave_one_model",
        "leave_one_codec",
    }
    for split in splits:
        train = normalized.iloc[list(split.train_indices)]
        test = normalized.iloc[list(split.test_indices)]
        assert not set(train["payload_group_id"]) & set(test["payload_group_id"])
        if split.held_out_column:
            assert split.held_out_value not in set(train[split.held_out_column])
            assert set(test[split.held_out_column]) == {split.held_out_value}


def test_primary_crossed_design_realizes_exact_28_prespecified_splits():
    raw, prompts, models, codecs = crossed_primary_detector_frame()
    normalized = normalize_detector_frame(raw)
    expected = validate_confirmatory_detector_frame(
        normalized,
        {
            "rows": 15840,
            "payload_groups": 480,
            "positive_rows": 7920,
            "negative_rows": 7920,
            "prompt_template_ids": prompts,
            "model_ids": models,
            "codec_ids": codecs,
            "split_count": 28,
        },
    )
    splits, skipped = build_evaluation_splits(
        normalized,
        minimum_train_rows=40,
        minimum_test_rows=20,
        seed=20260808,
    )
    assert skipped == []
    assert len(splits) == 28
    assert {split.split_id for split in splits} == set(expected)
    assert sum(split.regime == "held_out_template" for split in splits) == 18
    assert sum(split.regime == "leave_one_model" for split in splits) == 3
    assert sum(split.regime == "leave_one_codec" for split in splits) == 6
    for split in splits:
        assert_no_split_leakage(normalized, split)
        train_labels = normalized.iloc[list(split.train_indices)]["label"].value_counts()
        test_labels = normalized.iloc[list(split.test_indices)]["label"].value_counts()
        assert train_labels[0] == train_labels[1]
        assert test_labels[0] == test_labels[1]
        if split.regime in {"leave_one_model", "leave_one_codec"}:
            assert split.partition_policy == (
                "deterministic_disjoint_payload_partition_v1"
            )
            assert split.excluded_held_out_rows > 0


def test_confirmatory_contract_rejects_group_imbalance_hidden_by_global_balance():
    raw, prompts, models, codecs = crossed_primary_detector_frame()
    first_group = raw.index[raw["payload_group_id"] == "payload-000"]
    second_group = raw.index[raw["payload_group_id"] == "payload-001"]
    raw.loc[next(index for index in first_group if raw.loc[index, "label"] == 0), "label"] = 1
    raw.loc[next(index for index in second_group if raw.loc[index, "label"] == 1), "label"] = 0
    normalized = normalize_detector_frame(raw)
    with pytest.raises(RevisionDetectionError, match="exactly balanced labels"):
        validate_confirmatory_detector_frame(
            normalized,
            {
                "rows": 15840,
                "payload_groups": 480,
                "positive_rows": 7920,
                "negative_rows": 7920,
                "prompt_template_ids": prompts,
                "model_ids": models,
                "codec_ids": codecs,
                "split_count": 28,
            },
        )


def test_grouped_bootstrap_metrics_are_deterministic():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.05, 0.95]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    first = grouped_bootstrap_detector_metrics(
        labels, scores, groups, n_resamples=100, seed=55
    )
    second = grouped_bootstrap_detector_metrics(
        labels, scores, groups, n_resamples=100, seed=55
    )
    assert first == second
    assert first["bootstrap_unit"] == "payload_group_id"
    assert first["roc_auc"] == 1.0
    assert first["roc_auc_ci_low_95"] == 1.0
    assert first["roc_auc_ci_high_95"] == 1.0


def test_pinned_artifact_verifier_accepts_exact_files_and_real_cache(tmp_path):
    directory, expected = pinned_fixture_directory(tmp_path)
    cache = directory / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("metadata", encoding="utf-8")
    manifest = verify_pinned_model_artifacts(
        directory,
        expected,
        required_directory=directory,
        model_id="fixture/model",
        upstream_revision="fixture-revision",
    )
    assert manifest["verification_status"] == "verified"
    assert manifest["policy"] == (
        "exact_regular_files_no_symlinks_cache_metadata_ignored"
    )
    assert manifest["ignored_top_level_entries"] == [".cache"]
    assert [row["path"] for row in manifest["artifacts"]] == sorted(expected)
    assert len(manifest["artifact_set_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["missing", "extra", "hash"])
def test_pinned_artifact_verifier_fails_closed_on_file_set_or_hash(tmp_path, mutation):
    directory, expected = pinned_fixture_directory(tmp_path)
    if mutation == "missing":
        (directory / "spm.model").unlink()
        match = "missing required artifacts"
    elif mutation == "extra":
        (directory / "README.md").write_text("not allowed", encoding="utf-8")
        match = "disallowed extra entries"
    else:
        (directory / "spm.model").write_bytes(b"tampered")
        expected["spm.model"].pop("size_bytes")
        match = "SHA-256 mismatch"
    with pytest.raises(RevisionDetectionError, match=match):
        verify_pinned_model_artifacts(directory, expected)


def test_pinned_artifact_verifier_rejects_symlink_artifact_and_cache_symlink(tmp_path):
    directory, expected = pinned_fixture_directory(tmp_path)
    target = tmp_path / "outside.bin"
    target.write_bytes((directory / "spm.model").read_bytes())
    (directory / "spm.model").unlink()
    (directory / "spm.model").symlink_to(target)
    with pytest.raises(RevisionDetectionError, match="must not be symlinks"):
        verify_pinned_model_artifacts(directory, expected)

    (directory / "spm.model").unlink()
    (directory / "spm.model").write_bytes(target.read_bytes())
    cache = directory / ".cache"
    cache.mkdir()
    (cache / "linked").symlink_to(target)
    with pytest.raises(RevisionDetectionError, match="cache metadata.*symlinks"):
        verify_pinned_model_artifacts(directory, expected)


def test_model_state_sha256_is_order_invariant_and_byte_sensitive():
    torch = pytest.importorskip("torch")
    first = OrderedDict(
        [
            ("z.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("a.bias", torch.tensor([3, 4], dtype=torch.int64)),
        ]
    )
    reordered = OrderedDict(reversed(list(first.items())))
    first_hash = deterministic_model_state_sha256(first)
    assert first_hash == deterministic_model_state_sha256(reordered)
    changed = OrderedDict((name, tensor.clone()) for name, tensor in first.items())
    changed["z.weight"][0, 0] += 1.0
    assert first_hash != deterministic_model_state_sha256(changed)
    assert len(first_hash) == 64


def test_textcnn_reports_reproducible_post_training_state_hash():
    pytest.importorskip("torch")
    train_texts = [
        "ordinary calm cover one",
        "forced rank stego one",
        "ordinary calm cover two",
        "forced rank stego two",
    ]
    labels = [0, 1, 0, 1]
    config = {
        "device": "cpu",
        "torch_num_threads": 1,
        "maximum_length": 8,
        "embedding_dimension": 4,
        "filter_sizes": [2],
        "filters_per_width": 3,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 2,
    }
    scores_a, metadata_a = _run_torch_text_cnn(
        train_texts, labels, train_texts[:2], config, seed=991
    )
    scores_b, metadata_b = _run_torch_text_cnn(
        train_texts, labels, train_texts[:2], config, seed=991
    )
    assert metadata_a["model_state_hash_algorithm"] == "rankcloak-torch-state-v1"
    assert metadata_a["model_state_sha256"] == metadata_b["model_state_sha256"]
    assert len(metadata_a["model_state_sha256"]) == 64
    assert scores_a.tolist() == scores_b.tolist()


def test_non_smoke_neural_failure_never_uses_smoke_fallback(monkeypatch):
    frame = normalize_detector_frame(detector_frame(group_count=4))
    fallback_called = False

    def fail_neural(*args, **kwargs):
        raise RevisionDetectionError("neural failure")

    def record_fallback(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        raise AssertionError("fallback must not run")

    monkeypatch.setattr(revision_detection, "_run_torch_text_cnn", fail_neural)
    monkeypatch.setattr(revision_detection, "_run_hashed_ngram_logistic", record_fallback)
    with pytest.raises(RevisionDetectionError, match="neural failure"):
        run_configured_detector(
            frame.iloc[:4],
            frame.iloc[4:6],
            {
                "name": "cnn",
                "kind": "text_cnn",
                "fallback": "hashed_ngram_smoke",
            },
            seed=1,
            smoke=False,
        )
    assert fallback_called is False


def test_confirmatory_suite_preflights_pin_before_detector_execution(monkeypatch):
    detector_called = False

    def fail_pin(*args, **kwargs):
        raise RevisionDetectionError("fixture pin failure")

    def record_detector(*args, **kwargs):
        nonlocal detector_called
        detector_called = True
        raise AssertionError("detector must not run before preflight")

    config = smoke_config(
        detectors=[
            {
                "name": "published_textcnn_equivalent",
                "kind": "text_cnn",
                "fallback": "hashed_ngram_smoke",
            },
            {
                "name": "deberta_v3_base_classifier",
                "kind": "pretrained_transformer",
                "model_name_or_path": CONFIRMATORY_TRANSFORMER_MODEL_ID,
                "model_revision": None,
                "offline_only": True,
                "allow_downloads": False,
                "fallback": "hashed_ngram_smoke",
            },
        ]
    )
    monkeypatch.setattr(revision_detection, "verify_pinned_model_artifacts", fail_pin)
    monkeypatch.setattr(revision_detection, "run_configured_detector", record_detector)
    with pytest.raises(RevisionDetectionError, match="fixture pin failure"):
        run_revision_detector_suite(detector_frame(), config, smoke=False)
    assert detector_called is False


def test_frozen_confirmatory_plan_matches_code_pins():
    plan = json.loads(
        (
            PROJECT_ROOT / "analysis" / "revision_v1" / "detector_confirmatory_plan.json"
        ).read_text(encoding="utf-8")
    )
    pin = plan["transformer_pin"]
    assert pin["upstream_model_id"] == CONFIRMATORY_TRANSFORMER_MODEL_ID
    assert pin["upstream_revision"] == CONFIRMATORY_TRANSFORMER_REVISION
    assert pin["local_path"] == CONFIRMATORY_TRANSFORMER_RELATIVE_PATH
    assert pin["artifacts"] == CONFIRMATORY_TRANSFORMER_ARTIFACTS
    assert plan["execution_record"] == {
        "confirmatory_results_executed": False,
        "model_downloads_performed": False,
        "gpu_execution_performed": False,
    }


def test_neural_requests_use_explicitly_labelled_smoke_fallback():
    config = smoke_config(
        detectors=[
            {"name": "cnn", "kind": "text_cnn", "fallback": "hashed_ngram_smoke"},
            {
                "name": "transformer",
                "kind": "pretrained_transformer",
                "fallback": "hashed_ngram_smoke",
                "offline_only": True,
                "allow_downloads": False,
            },
        ]
    )
    result = run_revision_detector_suite(detector_frame(), config, smoke=True)
    assert result.failures == []
    assert len(result.metrics) == 2
    assert set(result.metrics["requested_kind"]) == {"text_cnn", "pretrained_transformer"}
    assert set(result.metrics["implementation_kind"]) == {"hashed_ngram_smoke"}
    assert set(result.metrics["implementation_status"]) == {"smoke_fallback"}
    assert set(result.predictions["implementation_kind"]) == {"hashed_ngram_smoke"}


def test_detector_cli_writes_auditable_smoke_products(tmp_path):
    input_path = tmp_path / "detector_input.csv"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "outputs"
    detector_frame().to_csv(input_path, index=False)
    config_path.write_text(json.dumps(smoke_config()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_revision_detectors.py"),
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_path),
            "--smoke",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    expected = {
        "detector_metrics.csv",
        "detector_predictions.csv",
        "detector_dataset_manifest.csv",
        "detector_split_manifest.json",
        "detector_failures.json",
        "detector_run_manifest.json",
    }
    assert expected == {path.name for path in output_path.iterdir()}
    manifest = json.loads((output_path / "detector_run_manifest.json").read_text())
    assert manifest["schema_version"] == "rankcloak-revision-detector-run-v2"
    assert manifest["execution_mode"] == "smoke"
    assert manifest["smoke"] is True
    assert manifest["smoke_fallback_metric_rows"] == 1
    assert manifest["confirmatory_complete"] is None
    assert set(manifest["output_files"]) == expected - {"detector_run_manifest.json"}
    for name, identity in manifest["output_files"].items():
        content = (output_path / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    dataset_manifest = pd.read_csv(output_path / "detector_dataset_manifest.csv")
    assert "text" not in dataset_manifest.columns
    assert dataset_manifest["text_sha256"].str.len().eq(64).all()

    manifest_before = (output_path / "detector_run_manifest.json").read_bytes()
    rerun = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_revision_detectors.py"),
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_path),
            "--smoke",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rerun.returncode == 2
    assert "not empty" in rerun.stderr
    assert (output_path / "detector_run_manifest.json").read_bytes() == manifest_before
