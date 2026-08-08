import json
from types import SimpleNamespace

import pandas as pd

from rankcloak.experiments import PROFILE_CONFIGS
from rankcloak import paper_suite
from rankcloak.paper_suite import (
    append_frame_unique,
    existing_id_set,
    insufficient_detector_row,
    insufficient_effect_row,
    insufficient_statistical_row,
    recovery_failure_note,
    stable_trial_id,
    write_run_progress,
)
from rankcloak.schemas import (
    DETECTOR_BASELINE_COLUMNS,
    EFFECT_SIZE_SUMMARY_COLUMNS,
    PAPER_STEGOTEXT_TRIAL_COLUMNS,
    STATISTICAL_SUMMARY_COLUMNS,
)


def test_append_frame_unique_skips_duplicate_trial_ids(tmp_path):
    path = tmp_path / "paper_stegotext_trials.csv"
    first = {
        "trial_id": "trial_a",
        "protocol_variant": "nonseg_ascii_b16",
        "payload_name": "payload",
        "exact_recovery": True,
        "notes": "first",
    }
    second = dict(first, notes="updated")
    append_frame_unique(path, [first], PAPER_STEGOTEXT_TRIAL_COLUMNS, ["trial_id"])
    frame = append_frame_unique(path, [second], PAPER_STEGOTEXT_TRIAL_COLUMNS, ["trial_id"])
    assert len(frame) == 1
    assert frame.iloc[0]["notes"] == "updated"
    assert existing_id_set(path) == {"trial_a"}


def test_stable_trial_id_is_deterministic_and_readable():
    trial_id = stable_trial_id(
        "nonseg_ascii_b16",
        "paper_sha256_hex_000",
        "ascii_bytes_fixed_radix",
        "b16",
        "recipe_long_specific",
    )
    assert trial_id == stable_trial_id(
        "nonseg_ascii_b16",
        "paper_sha256_hex_000",
        "ascii_bytes_fixed_radix",
        "b16",
        "recipe_long_specific",
    )
    assert "nonseg_ascii_b16" in trial_id
    assert "paper_sha256_hex_000" in trial_id


def test_run_progress_schema(tmp_path):
    progress = write_run_progress(
        output_dir=tmp_path,
        project_root=tmp_path,
        profile="paper-smoke",
        stage="paper-nonseg-generation",
        started_at="2026-05-21T00:00:00+00:00",
        planned_trials=3,
        completed_trials=1,
        skipped_existing_trials=1,
        failed_trials=0,
        remaining_trials=1,
        last_completed_trial_id="trial_a",
        notes=["test"],
    )
    assert progress["planned_trials"] == 3
    assert progress["skipped_existing_trials"] == 1
    loaded = json.loads((tmp_path / "RUN_PROGRESS.json").read_text())
    assert loaded["last_completed_trial_id"] == "trial_a"


def test_insufficient_detector_row_schema():
    frame = pd.DataFrame([insufficient_detector_row("need rows")]).reindex(
        columns=DETECTOR_BASELINE_COLUMNS
    )
    assert frame.iloc[0]["status"] == "insufficient_data"
    assert frame.iloc[0]["notes"] == "need rows"


def test_insufficient_statistics_rows_schema():
    stat_frame = pd.DataFrame([insufficient_statistical_row("need trials")]).reindex(
        columns=STATISTICAL_SUMMARY_COLUMNS
    )
    effect_frame = pd.DataFrame([insufficient_effect_row("need groups")]).reindex(
        columns=EFFECT_SIZE_SUMMARY_COLUMNS
    )
    assert stat_frame.iloc[0]["status"] == "insufficient_data"
    assert effect_frame.iloc[0]["status"] == "insufficient_data"


def test_recovery_failure_note_uses_observed_protocol_variants():
    frame = pd.DataFrame(
        [
            {"protocol_variant": "segmented_leadin", "exact_recovery": True},
            {"protocol_variant": "segmented_single", "exact_recovery": False},
            {"protocol_variant": "segmented_multi", "exact_recovery": False},
        ]
    )

    note = recovery_failure_note(frame, "Segmented")

    assert note.startswith("2 Segmented exact-recovery failures")
    assert "`segmented_multi`" in note
    assert "`segmented_single`" in note
    assert "`segmented_leadin`" not in note


def test_leadin_recovery_summary_note_uses_observed_failure_count():
    frame = pd.DataFrame(
        [
            {
                "protocol_variant": "segmented_hex_multi_topic_leadin8_sentence_tail_filtered",
                "exact_recovery": True,
            },
            {
                "protocol_variant": "segmented_hex_multi_topic_leadin8_sentence_tail_filtered",
                "exact_recovery": False,
            },
            {
                "protocol_variant": "segmented_hex_multi_topic_leadin8_sentence_tail_filtered",
                "exact_recovery": False,
            },
            {"protocol_variant": "segmented_single", "exact_recovery": False},
        ]
    )

    note = paper_suite.leadin_recovery_summary_note(frame)

    assert note.startswith("2 lead-in segmented exact-recovery failures")
    assert "this run" in note
    assert "partial pilot" not in note

    singular_note = paper_suite.leadin_recovery_summary_note(frame.iloc[:2])
    assert singular_note.startswith(
        "1 lead-in segmented exact-recovery failure was observed"
    )


def test_paper_profile_scope_note_matches_resolved_profile():
    assert "pilot-scale" in paper_suite.paper_profile_scope_note(
        "paper-main-pilot-resume"
    )
    assert "smoke test" in paper_suite.paper_profile_scope_note("paper-smoke")
    assert "larger frozen" in paper_suite.paper_profile_scope_note("paper-main")


def test_full_paper_summary_recommends_full_profile_resume(tmp_path):
    paper_suite.write_staged_summary(
        output_dir=tmp_path,
        project_root=tmp_path,
        profile="paper-main",
        stage="paper-statistics",
        model_loaded=False,
        model_status="test",
        notes=[],
    )

    summary_text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert ".venv/bin/python scripts/run_experiment.py --profile paper-main --output-dir . --resume" in summary_text
    assert "paper-main-pilot-resume" not in summary_text


def test_full_paper_summary_preserves_recorded_gpu_resume_options(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(
            {
                "inference_backend": {
                    "cuda_device_order": "PCI_BUS_ID",
                    "cuda_visible_devices": "1",
                },
                "command_line_args": [
                    "--profile",
                    "paper-main",
                    "--model-path",
                    "models/model.gguf",
                    "--n-gpu-layers",
                    "-1",
                    "--resume",
                ],
            }
        ),
        encoding="utf-8",
    )

    paper_suite.write_staged_summary(
        output_dir=tmp_path,
        project_root=tmp_path,
        profile="paper-main",
        stage="paper-statistics",
        model_loaded=True,
        model_status="loaded",
        notes=[],
    )

    summary_text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert (
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 "
        ".venv/bin/python scripts/run_experiment.py --profile paper-main "
        "--output-dir . --model-path models/model.gguf --n-gpu-layers -1 --resume"
        in summary_text
    )


def test_nonseg_rank_drift_is_recorded_as_recovery_failure(monkeypatch):
    payload = SimpleNamespace(
        payload_name="payload",
        payload_class="ciphertext_like_base64",
        payload_kind="ciphertext_like_base64",
        payload_text="A",
        payload_bytes=b"A",
    )
    plan = {
        "trial_id": "trial_rank_drift",
        "payload": payload,
        "spec": {
            "protocol_variant": "nonseg_ascii_b16",
            "representation_name": "ascii_bytes_fixed_radix",
            "alphabet_size": 16,
        },
        "encoded": {
            "ranks": [5, 2],
            "metadata": {
                "alphabet_size": 16,
                "bits_per_symbol": 4,
                "original_byte_length": 1,
                "padding_bits": 0,
            },
        },
        "prompt_name": "recipe_long_specific",
    }
    monkeypatch.setattr(paper_suite, "make_context_token_ids", lambda model, prompt: [1])
    monkeypatch.setattr(
        paper_suite,
        "generate_token_ids_from_ranks",
        lambda model, context, ranks: {
            "generated_text": "test",
            "generated_token_ids": [2, 3],
            "token_log_probabilities": [-1.0, -1.5],
        },
    )
    monkeypatch.setattr(
        paper_suite,
        "recover_ranks_from_generated_ids",
        lambda model, context, token_ids: {"ranks": [17, 2]},
    )

    row, example, features = paper_suite.run_single_nonseg_trial(
        plan,
        paper_suite.cover_prompt_dictionary(),
        object(),
        "test/repo",
        "model.gguf",
        "models/model.gguf",
    )

    assert row["exact_recovery"] is False
    assert "recovery decode failed" in row["notes"]
    assert "bounded rank 17 is outside 1..16" in example["recovery_error"]
    assert "recovery decode failed" in features["notes"]


def test_reconcile_baseline_artifacts_removes_obsolete_targets():
    existing_rows = [
        {"baseline_id": "baseline_keep"},
        {"baseline_id": "baseline_old"},
    ]
    feature_frame = pd.DataFrame(
        [
            {"source_type": "nonseg_rankcloak", "trial_id": "trial_a"},
            {"source_type": "baseline", "trial_id": "baseline_keep"},
            {"source_type": "baseline", "trial_id": "baseline_old"},
        ]
    )

    rows, features, obsolete = paper_suite.reconcile_baseline_artifacts(
        existing_rows, feature_frame, ["baseline_keep"]
    )

    assert rows == [{"baseline_id": "baseline_keep"}]
    assert set(features["trial_id"]) == {"trial_a", "baseline_keep"}
    assert obsolete == ["baseline_old"]


def test_staged_paper_profiles_are_registered():
    for profile in [
        "paper-smoke",
        "paper-diagnostics",
        "paper-nonseg-generation",
        "paper-segmented-generation",
        "paper-baselines",
        "paper-detector",
        "paper-statistics",
        "paper-main-pilot-resume",
        "paper-main",
    ]:
        assert profile in PROFILE_CONFIGS
        assert PROFILE_CONFIGS[profile]["write_staged_paper"] is True


def test_paper_main_uses_the_complete_staged_sequence():
    expected = [
        "paper-diagnostics",
        "paper-nonseg-generation",
        "paper-segmented-generation",
        "paper-baselines",
        "paper-detector",
        "paper-statistics",
    ]

    assert paper_suite.staged_paper_stage_names("paper-main") == expected
    assert paper_suite.staged_paper_stage_names("paper-main-pilot-resume") == expected
