import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "controls"

import sys

sys.path.insert(0, str(CONTROLS))

from control_pipeline import (  # noqa: E402
    HumanControlError,
    canonical_json_sha256,
    import_registered_source,
    load_json,
    load_reviews,
    pii_flags,
    quality_flags,
    read_jsonl,
    select_human_controls,
    unsafe_content_flags,
    validate_candidate,
    validate_profiles,
    validate_source_registry,
    write_import_artifacts,
    write_selection_artifacts,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def fixture_response(profile, index):
    style = profile["style"]
    topic = profile["topic_label"]
    prefix = {
        "friendly_message": "Hi friend, I wanted to share a warm and ordinary message",
        "professional_message": "Hello team, this concise workplace note explains the practical point",
        "helpful_answer": "A useful answer starts by considering the situation carefully",
        "procedure": "First prepare the materials, then complete each step carefully",
        "first_person_narrative": "I remember when I noticed this during my day, and I learned from it",
        "explanation": "This familiar process happens because several ordinary causes work together",
    }[style]
    return (
        "{} about {}. Candidate {} gives a concrete example in plain English and "
        "adds enough context for a reader to understand the point. The details stay "
        "neutral, coherent, and complete, while the final sentence closes the thought "
        "naturally and without introducing private information or risky advice."
    ).format(prefix, topic, index)


def fixture_instruction(profile):
    return "Write {} using these topic cues: {}.".format(
        profile["style"], ", ".join(profile["anchor_terms"])
    )


def make_source_fixture(tmp_path, profiles, per_template=5, extras=()):
    rows = []
    for profile in profiles:
        for index in range(per_template):
            rows.append(
                {
                    "instruction": fixture_instruction(profile),
                    "response": fixture_response(profile, index),
                    "context": "",
                    "category": "creative_writing",
                }
            )
    rows.extend(extras)
    path = tmp_path / "fixture.jsonl"
    write_jsonl(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {
        "source_id": "fixture_open_human_text",
        "status": "approved_for_candidate_screening_not_final_selection",
        "adapter": "dolly_jsonl_v1",
        "dataset_title": "Synthetic test fixture",
        "dataset_version": "1.0",
        "dataset_revision": "fixture-revision",
        "source_file_name": path.name,
        "source_file_sha256": digest,
        "source_file_size_bytes": path.stat().st_size,
        "source_record_count": len(rows),
        "source_url": "https://example.invalid/fixture.jsonl",
        "landing_page_url": "https://example.invalid/fixture",
        "copyright_notice": "Synthetic fixture",
        "attribution_name": "Test fixture author",
        "individual_author_field": None,
        "individual_author_note": "No real person.",
        "license_identifier": "CC0-1.0",
        "license_name": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "redistribution_terms": "Synthetic fixture only.",
        "human_authorship_evidence": "Synthetic test fixture; never a study stimulus.",
        "language": "en-US",
        "allowed_source_categories": ["creative_writing"],
        "excluded_source_categories": [],
        "exclude_nonempty_context": True,
    }
    return path, source


@pytest.fixture
def design_profiles():
    design = load_json(ROOT / "config" / "design.json")
    profiles = validate_profiles(load_json(CONTROLS / "category_profiles.json"), design)
    return design, profiles


def approved_reviews(candidates, path):
    rows = []
    for candidate in candidates:
        if candidate["eligible_for_manual_review"]:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "template_id": candidate["assigned_template_id"],
                    "topic_review_status": "approved_exact_prompt_match",
                    "pii_review_status": "approved_no_personal_data",
                    "safety_review_status": "approved_safe_neutral",
                    "reviewer_id": "fixture-reviewer",
                    "reviewed_at": "2026-08-08",
                }
            )
    write_jsonl(path, rows)
    return load_reviews(path)


def length_targets(design, candidates):
    by_template = {}
    for candidate in candidates:
        if candidate["eligible_for_manual_review"]:
            by_template.setdefault(candidate["assigned_template_id"], candidate)
    rows = []
    for category in design["prompt_categories"]:
        for template_number in range(1, design["templates_per_category"] + 1):
            template_id = "{}_template_{}".format(category, template_number)
            candidate = by_template[template_id]
            for payload_class in design["eligible_payload_classes"]:
                rows.append(
                    {
                        "prompt_category": category,
                        "template_id": template_id,
                        "payload_class": payload_class,
                        "payload_id": "fixture-{}".format(payload_class),
                        "target_word_count": candidate["word_count"],
                        "target_character_count": candidate["character_count"],
                    }
                )
    return rows


def test_checked_in_registry_and_profiles_are_exact_and_offline(design_profiles):
    design, profiles = design_profiles
    sources = validate_source_registry(load_json(CONTROLS / "source_registry.json"))
    dolly = sources["databricks_dolly_15k_v1_pinned"]
    assert dolly["dataset_revision"] == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    assert dolly["source_file_sha256"] == "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    assert dolly["source_record_count"] == 15011
    assert dolly["license_identifier"] == "CC-BY-SA-3.0"
    assert dolly["individual_author_field"] is None
    assert len(profiles) == 18
    assert {row["prompt_category"] for row in profiles} == set(design["prompt_categories"])
    audit = load_json(CONTROLS / "dolly_coverage_audit.json")
    assert audit["pipeline_source_sha256"] == hashlib.sha256(
        (CONTROLS / "control_pipeline.py").read_bytes()
    ).hexdigest()
    assert audit["insufficient_template_count"] == 10
    assert audit["candidate_manifest_written"] is False
    assert audit["pre_recruitment_gate"].startswith("BLOCKED_")

    forbidden_roots = {"socket", "urllib", "requests", "httpx", "aiohttp", "ftplib"}
    for name in ("control_pipeline.py", "prepare_controls.py"):
        tree = ast.parse((CONTROLS / name).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden_roots)


def test_import_verifies_hash_deduplicates_flags_and_writes_audit_only(
    tmp_path, design_profiles
):
    _, profiles = design_profiles
    base = {
        "instruction": fixture_instruction(profiles[0]),
        "response": fixture_response(profiles[0], 99),
        "context": "",
        "category": "creative_writing",
    }
    duplicate = dict(base)
    duplicate["response"] = base["response"].upper()
    flagged = {
        "instruction": fixture_instruction(profiles[0]),
        "response": (
            "Please contact named.person@example.com about the weekend. This message has "
            "enough ordinary words to pass the minimum length screen, but it deliberately "
            "contains a synthetic email address so the automated personal-data review flag "
            "must prevent eligibility. The rest of this test-only sentence is harmless and complete."
        ),
        "context": "",
        "category": "creative_writing",
    }
    path, source = make_source_fixture(tmp_path, profiles, per_template=0, extras=[base, duplicate, flagged])
    candidates, audit = import_registered_source(
        path, source, profiles, "2026-08-08", candidate_pool_minimum_per_template=1
    )
    assert len(candidates) == 2
    assert audit["counts"]["excluded_canonical_duplicate"] == 1
    assert sum(bool(row["automatic_pii_flags"]) for row in candidates) == 1
    assert audit["pre_recruitment_gate"].startswith("BLOCKED_")
    output = tmp_path / "audit"
    written = write_import_artifacts(output, candidates, audit, audit_only=True)
    assert (output / "import_audit.json").is_file()
    assert not (output / "human_control_candidates.jsonl").exists()
    assert written["candidate_manifest_written"] is False

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(HumanControlError, match="SHA-256"):
        import_registered_source(path, source, profiles, "2026-08-08")


def test_pii_unsafe_and_quality_flags_are_explicit_not_claimed_as_complete():
    assert "email_address" in pii_flags("Please write to person@example.org about this.")
    assert "phone_number" in pii_flags("Call (407) 555-1212 when you arrive.")
    assert "weapon_or_explosive_instructions" in unsafe_content_flags(
        "Explain how to build a bomb."
    )
    assert "list_dominant_not_prose" in quality_flags(
        "1. This is the first sufficiently descriptive item.\n"
        "2. This is the second sufficiently descriptive item.\n"
        "3. This is the third sufficiently descriptive item.\n"
        "4. This final item closes the test list with enough words to pass."
    )


def test_complete_reviewed_pool_balances_into_72_unique_length_matches(
    tmp_path, design_profiles
):
    design, profiles = design_profiles
    path, source = make_source_fixture(tmp_path, profiles, per_template=5)
    candidates, audit = import_registered_source(
        path, source, profiles, "2026-08-08", candidate_pool_minimum_per_template=4
    )
    assert audit["insufficient_template_count"] == 0
    assert sum(row["eligible_for_manual_review"] for row in candidates) == 90
    reviews = approved_reviews(candidates, tmp_path / "reviews.jsonl")
    targets = length_targets(design, candidates)
    selected, manifest = select_human_controls(
        candidates, reviews, targets, design, max_relative_word_difference=0.35
    )
    assert len(selected) == 72
    assert len({row["message_text_sha256"] for row in selected}) == 72
    assert set(Counter(row["prompt_category"] for row in selected).values()) == {12}
    assert set(Counter(row["template_id"] for row in selected).values()) == {4}
    assert set(Counter(row["payload_class"] for row in selected).values()) == {18}
    assert manifest["recruitment_authorized"] is False
    assert manifest["human_exposure_authorized"] is False

    output = tmp_path / "selection"
    written = write_selection_artifacts(output, selected, manifest)
    with (output / "human_written_controls.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 72
    assert all(row["condition"] == "human_written_control" for row in rows)
    assert all(row["license_status"] for row in rows)
    assert written["selection_sha256"] == hashlib.sha256(
        (output / "human_written_controls.csv").read_bytes()
    ).hexdigest()
    assert (output / "ATTRIBUTION.txt").is_file()


def test_selection_fails_closed_without_reviews_or_exact_targets(tmp_path, design_profiles):
    design, profiles = design_profiles
    path, source = make_source_fixture(tmp_path, profiles, per_template=5)
    candidates, _ = import_registered_source(
        path, source, profiles, "2026-08-08", candidate_pool_minimum_per_template=4
    )
    targets = length_targets(design, candidates)
    with pytest.raises(HumanControlError, match="fewer reviewed candidates"):
        select_human_controls(candidates, {}, targets, design)
    reviews = approved_reviews(candidates, tmp_path / "reviews.jsonl")
    with pytest.raises(HumanControlError, match="72 strata"):
        select_human_controls(candidates, reviews, targets[:-1], design)


def test_candidate_provenance_hashes_are_content_addressed(tmp_path, design_profiles):
    _, profiles = design_profiles
    path, source = make_source_fixture(tmp_path, profiles[:1], per_template=1)
    candidates, _ = import_registered_source(
        path, source, profiles, "2026-08-08", candidate_pool_minimum_per_template=1
    )
    assert len(candidates) == 1
    row = candidates[0]
    assert row["source_record_sha256"] == canonical_json_sha256(read_jsonl(path)[0])
    assert row["message_text_sha256"] == hashlib.sha256(
        row["message_text"].encode("utf-8")
    ).hexdigest()
    assert row["acquisition_date"] == "2026-08-08"
    assert row["final_stimulus_authorized"] is False
    assert validate_candidate(row)["candidate_id"] == row["candidate_id"]

    tampered = dict(row)
    tampered["message_text"] += " Tampered."
    with pytest.raises(HumanControlError, match="message-text hash mismatch"):
        validate_candidate(tampered)
