import json
from pathlib import Path

import pytest

from rankcloak.revision_compute import (
    EXPECTED_MODELS,
    RevisionComputeError,
    build_legacy_incurred_charge_ledger,
    verify_legacy_gpu_ledger,
)
from scripts import manage_legacy_gpu_ledger as cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SMOKE_ROOT = PROJECT_ROOT / "results" / "revision_v1" / "smoke_v2"
REAL_EVALUATOR_ROOT = (
    PROJECT_ROOT / "results" / "revision_v1" / "heldout_evaluator" / "smoke_v2"
)
REAL_INPUTS_AVAILABLE = all(
    (root / model_id).is_dir()
    for root in (REAL_SMOKE_ROOT, REAL_EVALUATOR_ROOT)
    for model_id in EXPECTED_MODELS
)


def test_publish_no_overwrite_is_atomic_and_preserves_existing_bytes(tmp_path):
    path = tmp_path / "incurred_charges" / "ledger.json"
    cli.publish_no_overwrite(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    frozen = path.read_bytes()
    with pytest.raises(cli.LedgerPublicationError, match="already exists"):
        cli.publish_no_overwrite(path, {"value": 2})
    assert path.read_bytes() == frozen


def test_publish_rejects_symlink_destination_and_symlink_parent(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("sentinel\n", encoding="utf-8")
    link = tmp_path / "ledger.json"
    link.symlink_to(target)
    with pytest.raises(cli.LedgerPublicationError, match="already exists"):
        cli.publish_no_overwrite(link, {"value": 1})
    assert target.read_text(encoding="utf-8") == "sentinel\n"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(cli.LedgerPublicationError, match="parent must not be a symlink"):
        cli.publish_no_overwrite(linked_parent / "ledger.json", {"value": 1})


def test_cli_create_calls_builder_once_and_verifies_published_file(tmp_path, monkeypatch, capsys):
    smoke_root = tmp_path / "smoke"
    evaluator_root = tmp_path / "evaluator"
    for root in (smoke_root, evaluator_root):
        for model_id in EXPECTED_MODELS:
            (root / model_id).mkdir(parents=True)
    ledger = tmp_path / "out" / "ledger.json"
    calls = {"build": 0, "verify": 0}

    def fake_build(smoke, evaluators, *, created_at):
        calls["build"] += 1
        assert [path.name for path in smoke] == list(EXPECTED_MODELS)
        assert [path.name for path in evaluators] == list(EXPECTED_MODELS)
        assert created_at == "2026-08-09T03:00:00+00:00"
        return {"ledger": "fixture"}

    def fake_verify(path):
        calls["verify"] += 1
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {"ledger": "fixture"}
        return {"status": "ok", "total_hours": 1.0}

    monkeypatch.setattr(cli, "build_legacy_incurred_charge_ledger", fake_build)
    monkeypatch.setattr(cli, "verify_legacy_gpu_ledger", fake_verify)
    code = cli.main(
        [
            "create",
            "--smoke-root", str(smoke_root),
            "--evaluator-root", str(evaluator_root),
            "--ledger", str(ledger),
            "--created-at", "2026-08-09T03:00:00+00:00",
        ]
    )
    assert code == 0
    assert calls == {"build": 1, "verify": 1}
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


@pytest.mark.skipif(not REAL_INPUTS_AVAILABLE, reason="local frozen legacy artifacts unavailable")
def test_real_legacy_sources_build_verify_and_reject_root_symlink(tmp_path):
    smoke = [REAL_SMOKE_ROOT / model_id for model_id in EXPECTED_MODELS]
    evaluators = [REAL_EVALUATOR_ROOT / model_id for model_id in EXPECTED_MODELS]
    value = build_legacy_incurred_charge_ledger(
        smoke,
        evaluators,
        created_at="2026-08-09T03:00:00+00:00",
    )
    assert value["entry_count"] == 6
    assert value["scientific_evidence_allowed"] is False
    assert value["rate_evidence_allowed"] is False
    assert value["charge_only_not_rate_evidence"] is True
    assert value["total_hours"] == pytest.approx(
        value["total_incurred_gpu_seconds"] / 3600.0
    )
    ledger = tmp_path / "ledger.json"
    cli.publish_no_overwrite(ledger, value)
    report = verify_legacy_gpu_ledger(ledger)
    assert report["status"] == "ok"
    assert report["total_hours"] == value["total_hours"]
    assert report["scientific_evidence_allowed"] is False
    assert report["rate_evidence_allowed"] is False

    symlink_root = tmp_path / EXPECTED_MODELS[0]
    symlink_root.symlink_to(smoke[0], target_is_directory=True)
    with pytest.raises(RevisionComputeError, match="symlink"):
        build_legacy_incurred_charge_ledger(
            [symlink_root, *smoke[1:]],
            evaluators,
            created_at="2026-08-09T03:00:00+00:00",
        )


@pytest.mark.skipif(not REAL_INPUTS_AVAILABLE, reason="local frozen legacy artifacts unavailable")
def test_real_ledger_self_hash_tampering_fails_closed(tmp_path):
    value = build_legacy_incurred_charge_ledger(
        [REAL_SMOKE_ROOT / model_id for model_id in EXPECTED_MODELS],
        [REAL_EVALUATOR_ROOT / model_id for model_id in EXPECTED_MODELS],
        created_at="2026-08-09T03:00:00+00:00",
    )
    ledger = tmp_path / "ledger.json"
    cli.publish_no_overwrite(ledger, value)
    tampered = json.loads(ledger.read_text(encoding="utf-8"))
    tampered["total_hours"] += 1.0
    ledger.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RevisionComputeError, match="self-hash"):
        verify_legacy_gpu_ledger(ledger)
