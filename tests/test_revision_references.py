from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / ".paper" / "scientific_reports"
ORIGINAL = PAPER / "references.bib"
STAGED = PAPER / "references2.bib"
AUDIT = ROOT / "revision_docs" / "REFERENCE_AUDIT.md"

PROTECTED_SHA256 = {
    "main.tex": "17e1045c098184b9472ded03e2dc16a26e451d8676e1ec982114ed8a9a545d74",
    "supplementary.tex": "93208ad95613d4fcd9eb60d5307c364fa783505408ca384ed9fd00bd2d75b995",
    "references.bib": "8b63d49225a5419b1ee531147521cfbd30d03c6f6ffcac602cd715a3905553ee",
}

REMOVED = {
    "Bai2024NextGenerationSteganalysis",
    "Bennett2004LinguisticSteganography",
    "RogerGreenblatt2023HidingReasoning",
}

ADDED = {
    "DingWangTao2020CrossLingualPosition",
    "HeGaoChen2023DeBERTaV3",
    "Jiang2023Mistral7B",
    "JosefssonLiusvaara2017EdDSA",
    "KrawczykBellareCanetti1997HMAC",
    "NIST2007GCM",
    "NirLangley2018ChaCha20Poly1305",
    "Yang2024Qwen25",
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
        assert _sha256(PAPER / name) == expected


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


def test_staged_set_matches_declared_removals_and_additions() -> None:
    original_keys = set(_entries(ORIGINAL))
    staged_keys = set(_entries(STAGED))
    assert original_keys - staged_keys == REMOVED
    assert staged_keys - original_keys == ADDED
    additions = _json_block(
        "<!-- BEGIN REFERENCE_ADDITIONS_JSON -->",
        "<!-- END REFERENCE_ADDITIONS_JSON -->",
    )
    assert {str(item["key"]) for item in additions} == ADDED


def test_staged_entries_have_required_fields_and_unique_dois() -> None:
    entries = _entries(STAGED)
    assert len(entries) == 48
    dois: dict[str, str] = {}
    for key, (kind, block) in entries.items():
        for common in ("author", "title", "year", "url"):
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
    calgacus = entries["NorelliBronstein2025Calgacus"]
    assert calgacus[0] == "inproceedings"
    assert _field(calgacus[1], "year") == "2026"
    assert _field(calgacus[1], "booktitle") == (
        "The Fourteenth International Conference on Learning Representations"
    )
    assert _field(calgacus[1], "url") == (
        "https://openreview.net/forum?id=tmFQWuIheV"
    )
    assert _field(
        entries["wang2025dynamicallyallocatedintervalbasedgenerative"][1], "doi"
    ) == "10.1016/j.asoc.2025.113101"
    assert _field(entries["Sadasivan2023AIDetection"][1], "journal") == (
        "Transactions on Machine Learning Research"
    )
    assert _field(entries["Sadasivan2023AIDetection"][1], "year") == "2025"
    assert _field(
        entries["Zolkowski2025EarlySignsSteganographicCapabilities"][1], "year"
    ) == "2026"
    assert _field(
        entries["Zolkowski2025EarlySignsSteganographicCapabilities"][1],
        "booktitle",
    ) == "The Fourteenth International Conference on Learning Representations"
    assert _field(entries["Motwani2024SecretCollusion"][1], "doi") == (
        "10.52202/079017-2336"
    )
    assert _field(entries["Simmons1984Prisoners"][1], "doi") == (
        "10.1007/978-1-4684-4730-9_5"
    )


def test_only_model_provenance_reports_use_arxiv_eprints() -> None:
    eprint_keys = {
        key for key, (_, block) in _entries(STAGED).items() if _field(block, "eprint")
    }
    assert eprint_keys == {
        "Dubey2024Llama3",
        "Yang2024Qwen25",
        "Jiang2023Mistral7B",
    }


def test_patient_huffman_and_reviewer_suggested_paper_are_present() -> None:
    entries = _entries(STAGED)
    assert _field(entries["DaiCai2019NearImperceptible"][1], "doi") == (
        "10.18653/v1/P19-1422"
    )
    assert _field(entries["DingWangTao2020CrossLingualPosition"][1], "doi") == (
        "10.18653/v1/2020.acl-main.153"
    )
