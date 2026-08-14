from rankcloak.schemas import (
    REVISION_CHECKPOINT_COLUMNS,
    REVISION_CONTROL_PLAN_COLUMNS,
    REVISION_FAILURE_COLUMNS,
    REVISION_PAYLOAD_COLUMNS,
    REVISION_RECOVERY_RESULT_COLUMNS,
    REVISION_RUNTIME_COLUMNS,
    REVISION_TRIAL_PLAN_COLUMNS,
)


def test_revision_schemas_cover_identity_integrity_failure_and_runtime_fields():
    assert {
        "payload_name",
        "payload_class",
        "payload_text",
        "artifact_sha256",
        "algorithm",
    }.issubset(REVISION_PAYLOAD_COLUMNS)
    assert {
        "trial_id",
        "model_id",
        "payload_split",
        "prompt_id",
        "generation_required",
    }.issubset(REVISION_TRIAL_PLAN_COLUMNS)
    assert {"control_id", "source_trial_id", "control_view"}.issubset(
        REVISION_CONTROL_PLAN_COLUMNS
    )
    assert {
        "exact_recovery",
        "context_sha256",
        "config_manifest_sha256",
    }.issubset(REVISION_RECOVERY_RESULT_COLUMNS)
    assert {
        "first_differing_position",
        "expected_token_id",
        "recovered_rank",
        "boundary_start_offset",
    }.issubset(REVISION_FAILURE_COLUMNS)
    assert {
        "encoding_seconds",
        "payload_bits_per_second",
        "peak_gpu_memory_bytes",
    }.issubset(REVISION_RUNTIME_COLUMNS)
    assert {
        "planned_trial_ids_sha256",
        "completed_trial_ids",
        "attempt_counts",
    }.issubset(REVISION_CHECKPOINT_COLUMNS)
