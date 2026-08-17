from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PAPER = ROOT / "paperV1" / "scientific_reports"
REVISED_PAPER = ROOT / "paperV2" / "scientific_reports"
ORIGINAL = ORIGINAL_PAPER / "references.bib"
STAGED = REVISED_PAPER / "references.bib"
AUDIT = ROOT / "revision_docs" / "REFERENCE_AUDIT.md"

PROTECTED_SHA256 = {
    "main.tex": "17e1045c098184b9472ded03e2dc16a26e451d8676e1ec982114ed8a9a545d74",
    "supplementary.tex": "93208ad95613d4fcd9eb60d5307c364fa783505408ca384ed9fd00bd2d75b995",
    "references.bib": "8b63d49225a5419b1ee531147521cfbd30d03c6f6ffcac602cd715a3905553ee",
    "rankcloak_scientific_reports_manuscript.pdf":
        "ac90fd962f48117b8549e5488a543f42b777d58f773a81aeb5fca1038de74703",
    "rankcloak_scientific_reports_supplementary.pdf":
        "d1e9f57ddbfaf4daaaf92caeea33412813796adae8aa64f539d74f3ddf2bf219",
}

REMOVED_OR_RENAMED = {
    "Badar2025StegomalwareSurvey",
    "Bai2024NextGenerationSteganalysis",
    "Bennett2004LinguisticSteganography",
    "ChapmanDavida1997HidingHidden",
    "Dubey2024Llama3",
    "Dyer2013FormatTransformingEncryption",
    "Kirchenbauer2023WatermarkLLM",
    "NorelliBronstein2025Calgacus",
    "RogerGreenblatt2023HidingReasoning",
    "Sadasivan2023AIDetection",
    "Wayner1995StrongTheoreticalSteganography",
    "Weinberg2012StegoTorus",
    "Zander2007CovertChannelsCountermeasures",
    "Zolkowski2025EarlySignsSteganographicCapabilities",
    "llamacpp",
    "llamacpppython",
    "wang2025dynamicallyallocatedintervalbasedgenerative",
}

ADDED_OR_RENAMED = {
    "Bates2015lme4",
    "Brooks2017glmmTMB",
    "Efron1979Bootstrap",
    "Flesch1948Readability",
    "He2021DeBERTa",
    "JosefssonLiusvaara2017RFC8032",
    "Kim2014CNN",
    "Krawczyk1997RFC2104",
    "NIST2007GCM",
    "NirLangley2018RFC8439",
    "NorelliBronstein2026Calgacus",
    "Sadasivan2025AIDetection",
    "Wang2025DAIRStega",
    "Wilson1927Interval",
    "Zolkowski2026EarlySigns",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entries(path: Path) -> dict[str, tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    starts = list(
        re.finditer(r"(?m)^@(?P<kind>[A-Za-z]+)\{(?P<key>[^,\s]+),", text)
    )
    entries: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        key = match.group("key")
        assert key not in entries, f"duplicate bibliography key: {key}"
        entries[key] = (match.group("kind").lower(), text[match.start() : end])
    return entries


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"(?mi)^\s*{re.escape(name)}\s*=\s*\{{", block)
    if match is None:
        return None
    start = match.end()
    depth = 1
    for index in range(start, len(block)):
        if block[index] == "{":
            depth += 1
        elif block[index] == "}":
            depth -= 1
            if depth == 0:
                return block[start:index].strip()
    raise AssertionError(f"unterminated field {name}")


def _json_block(start_marker: str, end_marker: str) -> list[dict[str, object]]:
    text = AUDIT.read_text(encoding="utf-8")
    payload = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    payload = payload.split("```json", 1)[1].split("```", 1)[0]
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    return parsed


def test_protected_submitted_sources_are_byte_identical() -> None:
    for name, expected in PROTECTED_SHA256.items():
        assert _sha256(ORIGINAL_PAPER / name) == expected


def test_mapping_covers_every_submitted_entry_exactly_once() -> None:
    original_keys = set(_entries(ORIGINAL))
    mapping = _json_block(
        "<!-- BEGIN REFERENCE_MAPPING_JSON -->",
        "<!-- END REFERENCE_MAPPING_JSON -->",
    )
    mapped_keys = [str(item["old_key"]) for item in mapping]
    assert len(mapped_keys) == len(set(mapped_keys)) == 43
    assert set(mapped_keys) == original_keys
    assert {str(item["action"]) for item in mapping} <= {"keep", "replace", "remove"}
    for item in mapping:
        assert item["rationale"]
        assert item["claims_affected"]
        assert str(item["evidence"]).startswith("https://")


def test_revised_set_matches_the_completed_v2_bibliography_delta() -> None:
    original_keys = set(_entries(ORIGINAL))
    revised_keys = set(_entries(STAGED))
    assert original_keys - revised_keys == REMOVED_OR_RENAMED
    assert revised_keys - original_keys == ADDED_OR_RENAMED


def test_staged_entries_have_required_fields_and_unique_dois() -> None:
    entries = _entries(STAGED)
    assert len(entries) == 41
    dois: dict[str, str] = {}
    for key, (kind, block) in entries.items():
        for common in ("author", "title", "year"):
            assert _field(block, common), f"{key} lacks {common}"
        if kind == "article":
            assert _field(block, "journal"), f"{key} lacks journal"
        elif kind in {"inproceedings", "incollection"}:
            assert _field(block, "booktitle"), f"{key} lacks booktitle"
        elif kind == "techreport":
            assert _field(block, "institution"), f"{key} lacks institution"
        elif kind == "misc":
            assert _field(block, "eprint") or _field(block, "howpublished")
        doi = _field(block, "doi")
        if doi is None:
            continue
        assert re.fullmatch(r"10\.\d{4,9}/\S+", doi), (key, doi)
        normalized = doi.casefold()
        assert normalized not in dois, f"duplicate DOI in {dois[normalized]} and {key}"
        dois[normalized] = key


def test_reviewer_highlighted_versions_are_exactly_staged() -> None:
    entries = _entries(STAGED)
    calgacus = entries["NorelliBronstein2026Calgacus"]
    assert calgacus[0] == "inproceedings"
    assert _field(calgacus[1], "year") == "2026"
    assert _field(calgacus[1], "booktitle") == (
        "The Fourteenth International Conference on Learning Representations"
    )
    assert _field(calgacus[1], "url").startswith("https://openreview.net/")
    assert _field(entries["Wang2025DAIRStega"][1], "doi") == (
        "10.1016/j.asoc.2025.113101"
    )
    assert _field(entries["Sadasivan2025AIDetection"][1], "journal") == (
        "Transactions on Machine Learning Research"
    )
    assert _field(entries["Sadasivan2025AIDetection"][1], "year") == "2025"
    early_signs = entries["Zolkowski2026EarlySigns"]
    assert _field(early_signs[1], "year") == "2026"
    assert _field(early_signs[1], "booktitle") == (
        "The Fourteenth International Conference on Learning Representations"
    )
    assert _field(early_signs[1], "url").startswith("https://openreview.net/")
    simmons = entries["Simmons1984Prisoners"][1]
    assert _field(simmons, "booktitle") == (
        "Advances in Cryptology: Proceedings of {CRYPTO} '83"
    )
    assert _field(simmons, "publisher") == "Springer"


def test_completed_v2_uses_no_arxiv_eprints() -> None:
    eprint_keys = {
        key for key, (_, block) in _entries(STAGED).items() if _field(block, "eprint")
    }
    assert eprint_keys == set()


def test_patient_huffman_is_present_and_tangential_suggestion_is_not_forced() -> None:
    entries = _entries(STAGED)
    assert _field(entries["DaiCai2019NearImperceptible"][1], "doi") == (
        "10.18653/v1/P19-1422"
    )
    assert "DingWangTao2020CrossLingualPosition" not in entries
    response = (ROOT / "paperV2/response/response_to_reviewers.tex").read_text(
        encoding="utf-8"
    )
    assert "We therefore did not force the citation" in response


def test_all_completed_v2_citations_resolve() -> None:
    entries = set(_entries(STAGED))
    cited: set[str] = set()
    for source_name in ("main2.tex", "supplementary2.tex"):
        source = (REVISED_PAPER / source_name).read_text(encoding="utf-8")
        for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", source):
            cited.update(key.strip() for key in match.group(1).split(","))
    assert cited
    assert cited <= entries
