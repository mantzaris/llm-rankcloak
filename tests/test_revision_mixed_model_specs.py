import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import rankcloak.revision_statistics as revision_statistics


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "revision_v1"
PLAN = ANALYSIS / "confirmatory_model_plan.json"
SPECS = ANALYSIS / "mixed_effects_specs.json"
LOCK = ANALYSIS / "r_environment.lock.json"
LAUNCHER = ANALYSIS / "run_with_locked_r.sh"
R_DRIVER = ROOT / "scripts" / "run_revision_mixed_models.R"
TRIAL_FIXTURE = ANALYSIS / "fixtures" / "all_success_trials.csv"
FEATURE_FIXTURE = ANALYSIS / "fixtures" / "all_zero_artifact_features.csv"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_confirmatory_specs_freeze_random_effects_interactions_and_no_fallback():
    plan = _load(PLAN)
    assert plan["frozen_before_confirmatory_results"] is True
    assert plan["experimental_unit"] == "payload_trial"
    assert plan["segments_as_independent_observations_forbidden"] is True
    assert plan["protocol_contract_revision"] == "payload_fidelity_v2"
    assert plan["result_schema_revision"] == "payload_aware_result_v2"
    primary_filter = plan["filters"]["primary_trials"]
    assert primary_filter == {
        "evidence_status": (
            "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
        ),
        "study_phase": "primary_v2_confirmatory",
        "protocol_contract_revision": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "record_type": "rankcloak_trial",
        "replay_mode": "saved_token_ids",
    }
    for model in plan["models"]:
        assert model["fixed_effects_fallback"] is False
        assert {"payload_name", "prompt_id"}.issubset(model["random_intercepts"])
        assert "model_id * protocol_variant" in model["formula"]
        assert "model_id * prompt_category" in model["formula"]
        assert "protocol_variant * prompt_category" in model["formula"]
    assert all(
        family["adjustment"].startswith("holm")
        for family in plan["contrast_families"]
    )
    recovery = next(
        model for model in plan["models"] if model["model_id"] == "primary_exact_recovery"
    )
    assert recovery["engine"] == "lme4::glmer"
    assert recovery["outcome_role"] == (
        "compatibility_alias_for_exact_payload_recovery"
    )
    assert recovery["recovery_outcome_semantics"] == (
        "original_serialized_payload_bytes_sha256_v1"
    )
    assert recovery["rank_replay_role"] == "diagnostic_only"
    assert "wilson" in recovery["all_success_or_zero_policy"]
    artifacts = next(
        model for model in plan["models"] if model["model_id"] == "primary_artifact_counts"
    )
    assert artifacts["engine"] == "lme4::glmer.nb"
    assert "offset(log(token_count))" in artifacts["formula"]
    assert plan["human_model"]["status"] == "external_until_irb_approved_ratings_exist"
    assert plan["human_model"]["engine"] == "ordinal::clmm"


def test_r_driver_requires_hash_checked_evaluator_join_and_emits_contrast_intervals():
    source = R_DRIVER.read_text(encoding="utf-8")
    assert 'Confirmatory execution requires --feature-join-manifest' in source
    assert 'verify_feature_join_manifest(options$feature_join_manifest' in source
    assert 'rankcloak-revision-heldout-feature-join-v1' in source
    assert 'source_full_message_replicated_across_nested_segment_rows_v1' in source
    assert 'evaluator_artifact_pins_verified' in source
    assert 'models_config_sha256' in source
    assert 'ci_low = if ("lower.CL" %in% names(pairs))' in source
    assert 'ci_high = if ("upper.CL" %in% names(pairs))' in source


def test_generic_specs_are_api_compatible_and_use_trial_units(monkeypatch):
    configuration = _load(SPECS)
    specs = configuration["models"]
    assert {spec["data_source"] for spec in specs} == {"trials"}
    assert all(spec["group_column"] == "payload_name" for spec in specs)
    assert all(spec["variance_component_columns"] == ["prompt_id"] for spec in specs)

    trials = pd.concat([pd.read_csv(TRIAL_FIXTURE)] * 2, ignore_index=True)
    trials.loc[1, "trial_id"] = "fixture_trial_002"
    trials.loc[1, "payload_name"] = "fixture_payload_002"

    def fake_fit(frame, *, formula, family, name, rscript):
        assert len(frame) == 2
        assert "(1 | payload_name)" in formula
        assert "(1 | prompt_id)" in formula
        assert family in {"binomial", "gaussian"}
        assert rscript.endswith("run_with_locked_r.sh")
        return pd.DataFrame(
            {
                "model_name": [name],
                "term": ["(Intercept)"],
                "estimate": [0.0],
                "fixed_effects_fallback": [False],
            }
        )

    monkeypatch.setattr(revision_statistics, "fit_r_lme4", fake_fit)
    coefficients, statuses = revision_statistics.run_mixed_effects_specs(
        {"trials": trials}, specs
    )
    assert len(coefficients) == len(specs)
    assert {status["status"] for status in statuses} == {"completed"}
    assert coefficients["fixed_effects_fallback"].eq(False).all()


def test_r_lock_uses_measured_project_first_composite_resolution():
    lock = _load(LOCK)
    roles = {entry["role"]: entry for entry in lock["library"]["resolution_order"]}
    assert roles["project_revision_library"]["path"] == ".r_libs/revision_v1"
    assert roles["existing_user_r_4_4_library"]["path"].endswith(
        "/R/x86_64-pc-linux-gnu-library/4.4"
    )
    assert lock["library"]["automatic_network_install"] is False
    assert lock["packages"]["lme4"]["version"] == "2.0.6"
    assert lock["packages"]["ordinal"]["version"] == "2026.7.26"
    assert lock["packages"]["ordinal"]["description_version"] == "2026.7-26"
    assert lock["packages"]["emmeans"]["version"] == "1.10.5"
    assert lock["packages"]["jsonlite"]["version"] == "1.8.9"
    assert lock["policy"]["fixed_effects_fallback_for_failed_mixed_models"] is False


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_driver_parses_and_locked_launcher_resolves_exact_packages(tmp_path):
    parsed = subprocess.run(
        ["Rscript", "--vanilla", "-e", f"parse(file={str(R_DRIVER)!r})"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    if not (ROOT / ".r_libs" / "revision_v1").is_dir():
        pytest.skip("project revision R library is unavailable")
    existing = Path(_load(LOCK)["library"]["resolution_order"][1]["path"])
    if not existing.is_dir():
        pytest.skip("declared existing R 4.4 user library is unavailable")
    completed = subprocess.run(
        [str(LAUNCHER), "-e", "cat('locked-environment-ok\\n')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "locked-environment-ok" in completed.stdout
    for package, version in (
        ("lme4", "2.0.6"),
        ("ordinal", "2026.7.26"),
        ("emmeans", "1.10.5"),
        ("jsonlite", "1.8.9"),
    ):
        assert f"locked package {package} {version}" in completed.stderr


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_primary_filter_uses_payload_equality_and_rejects_legacy_contracts(
    tmp_path,
):
    base = pd.read_csv(TRIAL_FIXTURE)
    base["protocol_variant"] = "direct_subword_calgacus"
    base["exact_rank_replay"] = 1
    base["exact_payload_recovery"] = 0
    base["exact_recovery"] = 0
    valid = tmp_path / "valid_direct.csv"
    base.to_csv(valid, index=False)

    alias_mismatch = tmp_path / "alias_mismatch.csv"
    base.assign(exact_recovery=1).to_csv(alias_mismatch, index=False)
    old_evidence = tmp_path / "old_evidence.csv"
    base.assign(evidence_status="confirmatory_after_manifest_freeze").to_csv(
        old_evidence, index=False
    )
    missing_revision = tmp_path / "missing_revision.csv"
    base.drop(columns=["result_schema_revision"]).to_csv(
        missing_revision, index=False
    )

    prefix = """
Sys.setenv(RANKCLOAK_MIXED_MODELS_SOURCE_ONLY='1')
source('scripts/run_revision_mixed_models.R')
plan <- read_json('analysis/revision_v1/confirmatory_model_plan.json')
validate_plan(plan)
"""
    valid_code = prefix + f"""
primary <- filter_primary_trials(read.csv({str(valid)!r}), plan)
stopifnot(primary$exact_rank_replay[[1]] == 1)
stopifnot(primary$exact_payload_recovery[[1]] == 0)
stopifnot(primary$exact_recovery[[1]] == 0)
stopifnot(attr(primary, 'payload_fidelity_contract')$exact_rank_replay_role == 'diagnostic_only')
cat('payload-filter-ok\n')
"""
    valid_result = subprocess.run(
        ["Rscript", "--vanilla", "-e", valid_code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert valid_result.returncode == 0, valid_result.stderr
    assert "payload-filter-ok" in valid_result.stdout

    for invalid, message in (
        (alias_mismatch, "compatibility alias differs"),
        (old_evidence, "No rows satisfy"),
        (missing_revision, "missing columns: result_schema_revision"),
    ):
        completed = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                "-e",
                prefix
                + f"filter_primary_trials(read.csv({str(invalid)!r}), plan)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert message in completed.stderr


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_all_success_and_all_zero_paths_do_not_fit_or_fallback():
    if not (ROOT / ".r_libs" / "revision_v1").is_dir():
        pytest.skip("project revision R library is unavailable")
    code = """
Sys.setenv(RANKCLOAK_MIXED_MODELS_SOURCE_ONLY='1')
source('scripts/run_revision_mixed_models.R')
plan <- read_json('analysis/revision_v1/confirmatory_model_plan.json')
primary <- filter_primary_trials(
  read.csv('analysis/revision_v1/fixtures/all_success_trials.csv'), plan
)
recovery <- fit_recovery_model(primary, plan)
stopifnot(recovery$status$status == 'not_fitted_complete_outcome_separation_all_success')
stopifnot(recovery$diagnostics$glmer_attempted == FALSE)
stopifnot(nrow(recovery$wilson) > 0)
features <- read.csv('analysis/revision_v1/fixtures/all_zero_artifact_features.csv')
artifacts <- fit_artifact_model(features, plan)
stopifnot(artifacts$status$status == 'not_fitted_all_zero_counts')
stopifnot(artifacts$diagnostics$glmer_nb_attempted == FALSE)
stopifnot(recovery$status$fixed_effects_fallback == FALSE)
stopifnot(artifacts$status$fixed_effects_fallback == FALSE)
cat('separation-paths-ok\\n')
"""
    environment = dict(os.environ)
    environment["RANKCLOAK_MIXED_MODELS_SOURCE_ONLY"] = "1"
    completed = subprocess.run(
        [str(LAUNCHER), "-e", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "separation-paths-ok" in completed.stdout


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_validation_only_outputs_are_atomic_and_manifested(tmp_path):
    if not (ROOT / ".r_libs" / "revision_v1").is_dir():
        pytest.skip("project revision R library is unavailable")
    output = tmp_path / "mixed-model-validation"
    completed = subprocess.run(
        [
            str(LAUNCHER),
            str(R_DRIVER),
            "--plan",
            str(PLAN),
            "--environment-lock",
            str(LOCK),
            "--trials",
            str(TRIAL_FIXTURE),
            "--output-dir",
            str(output),
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = _load(output / "mixed_model_run_manifest.json")
    assert manifest["validation_only"] is True
    assert manifest["analysis_unit"] == "payload_trial"
    assert manifest["segments_as_independent_observations"] is False
    assert manifest["fixed_effects_fallback"] is False
    assert manifest["payload_fidelity_contract"] == {
        "contract_version": "payload_fidelity_v2",
        "result_schema_revision": "payload_aware_result_v2",
        "semantics": "original_serialized_payload_bytes_sha256_v1",
        "primary_outcome": "exact_payload_recovery",
        "compatibility_alias": "exact_recovery",
        "alias_equality_validated": True,
        "exact_rank_replay_role": "diagnostic_only",
        "direct_rows": 0,
        "direct_rows_contract_verified": 0,
    }
    assert set(manifest["outputs"]) == {
        "coefficients",
        "contrasts",
        "diagnostics",
        "wilson",
        "dispersion",
        "status",
    }
    rerun = subprocess.run(
        [
            str(LAUNCHER),
            str(R_DRIVER),
            "--plan",
            str(PLAN),
            "--trials",
            str(TRIAL_FIXTURE),
            "--output-dir",
            str(output),
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode != 0
    assert "Output already exists" in rerun.stderr
