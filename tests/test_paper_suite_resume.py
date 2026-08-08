import json

import pandas as pd

from rankcloak.experiments import PROFILE_CONFIGS
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
    ]:
        assert profile in PROFILE_CONFIGS
        assert PROFILE_CONFIGS[profile]["write_staged_paper"] is True
