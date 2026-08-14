"""Offline provenance, screening, and matching for human-written controls.

This module intentionally has no network client. Acquisition is a separate,
logged step; every analysis input must match a registered byte-level digest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
HUMAN_STUDY_ROOT = ROOT.parent
DEFAULT_REGISTRY = ROOT / "source_registry.json"
DEFAULT_PROFILES = ROOT / "category_profiles.json"
DEFAULT_DESIGN = HUMAN_STUDY_ROOT / "config" / "design.json"
PIPELINE_SCHEMA_VERSION = "rankcloak-human-controls-v1"
REQUIRED_TARGET_FIELDS = {
    "prompt_category",
    "template_id",
    "payload_class",
    "target_word_count",
    "target_character_count",
}
REQUIRED_REVIEW_FIELDS = {
    "candidate_id",
    "template_id",
    "topic_review_status",
    "pii_review_status",
    "safety_review_status",
    "reviewer_id",
    "reviewed_at",
}
REQUIRED_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "source_id",
    "source_dataset_title",
    "source_dataset_version",
    "source_dataset_revision",
    "source_record_id",
    "source_record_locator",
    "source_record_sha256",
    "source_url",
    "landing_page_url",
    "source_file_sha256",
    "source_instruction",
    "source_instruction_sha256",
    "message_text",
    "message_text_sha256",
    "canonical_text_sha256",
    "word_count",
    "character_count",
    "attribution_name",
    "individual_author",
    "individual_author_note",
    "license_identifier",
    "license_url",
    "redistribution_terms",
    "changes_made",
    "acquisition_date",
    "automatic_pii_flags",
    "automatic_unsafe_content_flags",
    "automatic_quality_flags",
    "assigned_template_id",
    "assigned_prompt_category",
    "eligible_for_manual_review",
    "final_stimulus_authorized",
}


class HumanControlError(RuntimeError):
    """Raised when a control artifact is unsafe, incomplete, or irreproducible."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanControlError("Cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise HumanControlError("Expected a JSON object: {}".format(path))
    return value


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HumanControlError("Cannot read JSONL {}: {}".format(path, exc)) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanControlError(
                "Invalid JSONL at {}:{}: {}".format(path, line_number, exc)
            ) from exc
        if not isinstance(value, dict):
            raise HumanControlError(
                "Expected a JSON object at {}:{}".format(path, line_number)
            )
        rows.append(value)
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise HumanControlError("Cannot read CSV {}: {}".format(path, exc)) from exc


def _iso_date(value: object, label: str) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise HumanControlError("{} must be an ISO date".format(label)) from exc
    if parsed > date.today():
        raise HumanControlError("{} cannot be in the future".format(label))
    return parsed.isoformat()


def _atomic_write(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def write_json(path: Path, value: object) -> None:
    _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _validate_sha256(value: object, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise HumanControlError("{} must be a lowercase SHA-256".format(label))
    return text


def validate_source_registry(registry: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    if registry.get("schema_version") != "rankcloak-human-control-sources-v1":
        raise HumanControlError("Unsupported source-registry schema")
    policy = registry.get("policy")
    if not isinstance(policy, dict):
        raise HumanControlError("Source registry lacks a policy")
    if policy.get("network_access_during_import_selection_or_validation") is not False:
        raise HumanControlError("Offline-analysis policy must be explicit")
    allowed_licenses = set(map(str, policy.get("allowed_license_identifiers", [])))
    required = {
        "source_id", "status", "adapter", "dataset_title", "dataset_version",
        "dataset_revision", "source_file_name", "source_file_sha256",
        "source_file_size_bytes", "source_record_count", "source_url",
        "landing_page_url", "attribution_name", "license_identifier",
        "license_url", "redistribution_terms", "human_authorship_evidence",
        "allowed_source_categories", "exclude_nonempty_context",
    }
    result: Dict[str, Mapping[str, object]] = {}
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise HumanControlError("Source registry must contain at least one source")
    for source in sources:
        if not isinstance(source, dict):
            raise HumanControlError("Source registry entry is not an object")
        missing = sorted(required - set(source))
        if missing:
            raise HumanControlError("Source entry missing {}".format(", ".join(missing)))
        source_id = str(source["source_id"])
        if not source_id or source_id in result:
            raise HumanControlError("Source identifiers must be nonempty and unique")
        if source["status"] != "approved_for_candidate_screening_not_final_selection":
            raise HumanControlError("Source is not approved for candidate screening")
        if str(source["license_identifier"]) not in allowed_licenses:
            raise HumanControlError("Source license is not allowed by registry policy")
        _validate_sha256(source["source_file_sha256"], "source_file_sha256")
        if int(source["source_file_size_bytes"]) <= 0 or int(source["source_record_count"]) <= 0:
            raise HumanControlError("Source size and record count must be positive")
        for field in (
            "source_url", "landing_page_url", "license_url", "attribution_name",
            "redistribution_terms", "human_authorship_evidence",
        ):
            if not str(source[field]).strip():
                raise HumanControlError("Source field {} cannot be empty".format(field))
        result[source_id] = source
    return result


def validate_profiles(
    profiles: Mapping[str, object], design: Mapping[str, object]
) -> List[Dict[str, object]]:
    if profiles.get("schema_version") != "rankcloak-human-control-topic-profiles-v1":
        raise HumanControlError("Unsupported topic-profile schema")
    values = profiles.get("templates")
    if not isinstance(values, list):
        raise HumanControlError("Topic profiles templates must be a list")
    expected = {
        "{}_template_{}".format(category, number)
        for category in design["prompt_categories"]
        for number in range(1, int(design["templates_per_category"]) + 1)
    }
    identifiers = [str(row.get("template_id")) for row in values if isinstance(row, dict)]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
        raise HumanControlError("Topic profiles do not match the 18 frozen templates")
    result = []
    for row in values:
        if not isinstance(row, dict):
            raise HumanControlError("Topic profile is not an object")
        if str(row.get("prompt_category")) not in set(design["prompt_categories"]):
            raise HumanControlError("Topic profile has an unknown prompt category")
        anchors = row.get("anchor_terms")
        if not isinstance(anchors, list) or not anchors or not all(str(x).strip() for x in anchors):
            raise HumanControlError("Every topic profile needs anchor terms")
        if int(row.get("minimum_score", 0)) <= 0:
            raise HumanControlError("Every topic profile needs a positive threshold")
        result.append(dict(row))
    return result


def display_text(value: object) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.strip().split("\n")]
    return "\n".join(lines).strip()


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", display_text(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+", text)


def _term_present(text: str, term: str) -> bool:
    normalized_term = canonical_text(term)
    if not normalized_term:
        return False
    return re.search(r"(?:^|\s){}(?:$|\s)".format(re.escape(normalized_term)), text) is not None


def pii_flags(text: str) -> List[str]:
    patterns = (
        ("email_address", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        ("web_url", r"(?:https?://|www\.)\S+", re.IGNORECASE),
        ("ipv4_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0),
        ("phone_number", r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\d)", 0),
        ("social_handle", r"(?<!\w)@[A-Za-z0-9_]{2,30}\b", 0),
        ("street_address", r"\b\d{1,6}\s+[A-Za-z0-9.' -]+\s(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b", re.IGNORECASE),
        ("credential_or_secret", r"\b(?:password|passcode|api[_ -]?key|secret[_ -]?key|access[_ -]?token)\b", re.IGNORECASE),
        ("named_signature_or_recipient", r"(?m)^(?:Dear|Hi|Hello|Regards|Sincerely|Best),?\s+[A-Z][a-z]{2,}\b", 0),
        ("self_identified_name", r"\b[Mm]y name is\s+[A-Z][a-z]{2,}\b", 0),
    )
    return sorted(
        label for label, pattern, flags in patterns if re.search(pattern, text, flags=flags)
    )


def unsafe_content_flags(text: str) -> List[str]:
    patterns = (
        ("self_harm", r"\b(?:suicide|kill myself|self[- ]harm|cut myself)\b"),
        ("weapon_or_explosive_instructions", r"\b(?:build|make|assemble|use)\s+(?:a\s+)?(?:bomb|explosive|firearm|weapon)\b"),
        ("illicit_intrusion", r"\b(?:hack|steal|bypass)\s+(?:an?\s+)?(?:account|password|system|security)\b"),
        ("explicit_sexual_content", r"\b(?:pornograph\w*|explicit sexual|sexual assault|rape)\b"),
        ("medical_crisis_or_prescription", r"\b(?:diagnose|dosage|prescription|overdose|medical emergency)\b"),
        ("operational_illegal_drugs", r"\b(?:manufacture|cook|sell)\s+(?:illegal\s+)?(?:meth|cocaine|heroin|drugs)\b"),
        ("real_credential_context", r"\b(?:social security number|credit card number|bank account number)\b"),
    )
    return sorted(
        label for label, pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)
    )


def quality_flags(text: str, minimum_words: int = 35, maximum_words: int = 300) -> List[str]:
    flags = []
    words = word_tokens(text)
    if len(words) < minimum_words:
        flags.append("too_short")
    if len(words) > maximum_words:
        flags.append("too_long")
    if any(ord(character) < 32 and character not in "\n\t" for character in text):
        flags.append("control_character")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(
        bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines
    )
    if lines and list_lines / len(lines) >= 0.4:
        flags.append("list_dominant_not_prose")
    if "```" in text or re.search(r"(?m)^\s*(?:def |class |import |SELECT\s+|#!/)", text):
        flags.append("code_or_command_block")
    if not re.search(r"[.!?][\"')\]]?\s*$", text):
        flags.append("possibly_incomplete_ending")
    return sorted(flags)


def validate_candidate(candidate: Mapping[str, object]) -> Dict[str, object]:
    """Verify self-contained provenance and content claims before selection."""

    missing = REQUIRED_CANDIDATE_FIELDS - set(candidate)
    if missing:
        raise HumanControlError(
            "Candidate record missing {}".format(", ".join(sorted(missing)))
        )
    value = dict(candidate)
    if value.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise HumanControlError("Candidate has an unsupported schema")
    text = str(value["message_text"])
    instruction = str(value["source_instruction"])
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if str(value["message_text_sha256"]) != text_hash:
        raise HumanControlError("Candidate message-text hash mismatch")
    if str(value["source_instruction_sha256"]) != instruction_hash:
        raise HumanControlError("Candidate source-instruction hash mismatch")
    if str(value["canonical_text_sha256"]) != hashlib.sha256(
        canonical_text(text).encode("utf-8")
    ).hexdigest():
        raise HumanControlError("Candidate canonical-text hash mismatch")
    expected_id = "HC-{}".format(text_hash[:24])
    if str(value["candidate_id"]) != expected_id:
        raise HumanControlError("Candidate content-addressed ID mismatch")
    for field in ("source_record_sha256", "source_file_sha256"):
        _validate_sha256(value[field], field)
    for field in (
        "source_id",
        "source_dataset_title",
        "source_dataset_version",
        "source_dataset_revision",
        "source_record_id",
        "source_record_locator",
        "attribution_name",
        "license_identifier",
        "redistribution_terms",
        "changes_made",
    ):
        if not str(value[field]).strip():
            raise HumanControlError("Candidate {} cannot be empty".format(field))
    for field in ("source_url", "landing_page_url", "license_url"):
        if not str(value[field]).startswith("https://"):
            raise HumanControlError("Candidate {} must be an HTTPS URL".format(field))
    if not str(value.get("individual_author") or "").strip() and not str(
        value.get("individual_author_note") or ""
    ).strip():
        raise HumanControlError("Candidate lacks author or author-unavailability provenance")
    _iso_date(value["acquisition_date"], "acquisition_date")
    try:
        recorded_words = int(value["word_count"])
        recorded_characters = int(value["character_count"])
    except (TypeError, ValueError) as exc:
        raise HumanControlError("Candidate length fields must be integers") from exc
    if recorded_words != len(word_tokens(text)) or recorded_characters != len(text):
        raise HumanControlError("Candidate length metadata does not match its text")
    for field in (
        "automatic_pii_flags",
        "automatic_unsafe_content_flags",
        "automatic_quality_flags",
    ):
        if not isinstance(value[field], list):
            raise HumanControlError("Candidate {} must be a list".format(field))
    if value.get("eligible_for_manual_review") is True:
        if any(value[field] for field in (
            "automatic_pii_flags",
            "automatic_unsafe_content_flags",
            "automatic_quality_flags",
        )):
            raise HumanControlError("Eligible candidate retains an automated exclusion flag")
        if not str(value.get("assigned_template_id") or "").strip() or not str(
            value.get("assigned_prompt_category") or ""
        ).strip():
            raise HumanControlError("Eligible candidate lacks a frozen topic assignment")
    if value.get("final_stimulus_authorized") is not False:
        raise HumanControlError("Candidate input improperly claims stimulus authorization")
    return value


def _style_score(style: str, instruction: str, response: str) -> Tuple[int, List[str]]:
    combined = "{}\n{}".format(instruction, response)
    lower = combined.casefold()
    reasons = []
    score = 0
    if style == "friendly_message":
        if re.search(r"\b(?:message|email|letter|note|text)\b", instruction, re.I):
            score += 2
            reasons.append("message-form instruction")
        if re.search(r"\b(?:friend|neighbor|neighbour)\b", lower):
            score += 1
            reasons.append("friendly recipient")
    elif style == "professional_message":
        if re.search(r"\b(?:professional|workplace|colleague|team|email|meeting|project)\b", lower):
            score += 2
            reasons.append("professional-message cue")
        if re.search(r"(?im)^(?:dear|hello|hi|best|regards|thank you)", response):
            score += 1
            reasons.append("message structure")
    elif style == "helpful_answer":
        if "?" in instruction or re.match(r"\s*(?:how|what|which|should|can)\b", instruction, re.I):
            score += 2
            reasons.append("question-answer structure")
        if len(word_tokens(response)) >= 35:
            score += 1
            reasons.append("developed answer")
    elif style == "procedure":
        if re.search(r"\b(?:how to|instructions?|steps?|recipe|prepare|make|repair|install)\b", lower):
            score += 2
            reasons.append("procedural instruction")
        if re.search(r"\b(?:first|next|then|finally|before|after|until)\b", response, re.I):
            score += 1
            reasons.append("ordered prose")
    elif style == "first_person_narrative":
        first_person = len(re.findall(r"\b(?:I|me|my|we|our)\b", response, re.I))
        if first_person >= 3:
            score += 2
            reasons.append("sustained first person")
        if re.search(r"\b(?:was|were|had|went|noticed|learned|realized|found)\b", response, re.I):
            score += 1
            reasons.append("narrative event cue")
    elif style == "explanation":
        if re.match(r"\s*(?:explain|how|why|what)\b", instruction, re.I):
            score += 2
            reasons.append("explanation instruction")
        if re.search(r"\b(?:because|therefore|as a result|causes?|works? by|process)\b", response, re.I):
            score += 1
            reasons.append("cause-process language")
    else:
        raise HumanControlError("Unknown topic-profile style: {}".format(style))
    return score, reasons


def topic_scores(
    instruction: str,
    response: str,
    profiles: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    normalized_instruction = canonical_text(instruction)
    normalized_response = canonical_text(response)
    scored = []
    for profile in profiles:
        instruction_hits = [
            str(term)
            for term in profile["anchor_terms"]
            if _term_present(normalized_instruction, str(term))
        ]
        response_hits = [
            str(term)
            for term in profile["anchor_terms"]
            if _term_present(normalized_response, str(term))
            and str(term) not in instruction_hits
        ]
        style_points, style_reasons = _style_score(
            str(profile["style"]), instruction, response
        )
        score = 2 * len(instruction_hits) + len(response_hits) + style_points
        scored.append(
            {
                "template_id": profile["template_id"],
                "prompt_category": profile["prompt_category"],
                "revision_prompt_id": profile["revision_prompt_id"],
                "score": score,
                "minimum_score": int(profile["minimum_score"]),
                "instruction_anchor_hits": instruction_hits,
                "response_anchor_hits": response_hits,
                "style_reasons": style_reasons,
            }
        )
    return sorted(scored, key=lambda row: (-int(row["score"]), str(row["template_id"])))


def assign_topic(scored: Sequence[Mapping[str, object]]) -> Tuple[Optional[Dict[str, object]], str]:
    if not scored:
        return None, "no_profiles"
    best = dict(scored[0])
    second_score = int(scored[1]["score"]) if len(scored) > 1 else -1
    if int(best["score"]) < int(best["minimum_score"]):
        return None, "below_profile_threshold"
    if int(best["score"]) - second_score < 1:
        return None, "ambiguous_profile_tie"
    return best, "assigned_pending_manual_review"


def _source_record_id(line_number: int, record: Mapping[str, object]) -> str:
    return "jsonl-line-{:05d}-{}".format(
        line_number, canonical_json_sha256(record)[:16]
    )


def import_registered_source(
    input_path: Path,
    source: Mapping[str, object],
    profiles: Sequence[Mapping[str, object]],
    acquisition_date: str,
    candidate_pool_minimum_per_template: int = 8,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Import and screen a locally acquired, content-pinned source file."""

    input_path = Path(input_path)
    acquired = _iso_date(acquisition_date, "acquisition_date")
    actual_hash = file_sha256(input_path)
    actual_size = input_path.stat().st_size
    if actual_hash != source["source_file_sha256"]:
        raise HumanControlError("Source file SHA-256 does not match the registry")
    if actual_size != int(source["source_file_size_bytes"]):
        raise HumanControlError("Source file byte size does not match the registry")
    if source["adapter"] != "dolly_jsonl_v1":
        raise HumanControlError("Unsupported source adapter: {}".format(source["adapter"]))
    if int(candidate_pool_minimum_per_template) <= 0:
        raise HumanControlError("Candidate-pool minimum must be positive")
    rows = read_jsonl(input_path)
    if len(rows) != int(source["source_record_count"]):
        raise HumanControlError("Source record count does not match the registry")

    allowed_categories = set(map(str, source["allowed_source_categories"]))
    candidates: List[Dict[str, object]] = []
    counts: Counter = Counter()
    seen_text: Dict[str, str] = {}
    for line_number, row in enumerate(rows, 1):
        required = {"instruction", "response", "context", "category"}
        if required - set(row):
            raise HumanControlError(
                "Source line {} lacks {}".format(
                    line_number, ", ".join(sorted(required - set(row)))
                )
            )
        counts["source_records"] += 1
        category = str(row["category"])
        if category not in allowed_categories:
            counts["excluded_source_category"] += 1
            continue
        if bool(source["exclude_nonempty_context"]) and str(row.get("context") or "").strip():
            counts["excluded_nonempty_context"] += 1
            continue
        instruction = display_text(row["instruction"])
        response = display_text(row["response"])
        if not instruction or not response:
            counts["excluded_empty_instruction_or_response"] += 1
            continue
        dedup_canonical = canonical_text(response)
        dedup_hash = hashlib.sha256(dedup_canonical.encode("utf-8")).hexdigest()
        if not dedup_canonical or dedup_hash in seen_text:
            counts["excluded_canonical_duplicate"] += 1
            continue
        seen_text[dedup_hash] = _source_record_id(line_number, row)
        text_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
        candidate_id = "HC-{}".format(text_hash[:24])
        p_flags = pii_flags(response)
        u_flags = unsafe_content_flags("{}\n{}".format(instruction, response))
        q_flags = quality_flags(response)
        scores = topic_scores(instruction, response, profiles)
        assigned, assignment_status = assign_topic(scores)
        reasons = []
        if p_flags:
            reasons.append("automated_pii_flags")
        if u_flags:
            reasons.append("automated_unsafe_content_flags")
        if q_flags:
            reasons.append("quality_flags")
        if assigned is None:
            reasons.append(assignment_status)
        eligible = not reasons
        if eligible:
            counts["eligible_for_manual_review"] += 1
            counts["template:{}".format(assigned["template_id"])] += 1
        else:
            counts["not_eligible_for_manual_review"] += 1
            for reason in reasons:
                counts["reason:{}".format(reason)] += 1
        source_record_id = _source_record_id(line_number, row)
        candidates.append(
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "source_id": source["source_id"],
                "source_dataset_title": source["dataset_title"],
                "source_dataset_version": source["dataset_version"],
                "source_dataset_revision": source["dataset_revision"],
                "source_record_id": source_record_id,
                "source_record_locator": "{} line {}".format(
                    source["source_file_name"], line_number
                ),
                "source_record_sha256": canonical_json_sha256(row),
                "source_url": source["source_url"],
                "landing_page_url": source["landing_page_url"],
                "acquisition_date": acquired,
                "source_file_sha256": actual_hash,
                "source_category": category,
                "source_instruction": instruction,
                "source_instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "message_text": response,
                "message_text_sha256": text_hash,
                "canonical_text_sha256": dedup_hash,
                "word_count": len(word_tokens(response)),
                "character_count": len(response),
                "attribution_name": source["attribution_name"],
                "individual_author": None,
                "individual_author_note": source.get("individual_author_note"),
                "copyright_notice": source.get("copyright_notice"),
                "license_identifier": source["license_identifier"],
                "license_name": source.get("license_name"),
                "license_url": source["license_url"],
                "redistribution_terms": source["redistribution_terms"],
                "changes_made": "Line endings and surrounding whitespace normalized; no semantic edits.",
                "automatic_pii_flags": p_flags,
                "automatic_unsafe_content_flags": u_flags,
                "automatic_quality_flags": q_flags,
                "topic_profile_top_scores": scores[:3],
                "topic_assignment_status": assignment_status,
                "assigned_template_id": assigned.get("template_id") if assigned else None,
                "assigned_prompt_category": assigned.get("prompt_category") if assigned else None,
                "assigned_revision_prompt_id": assigned.get("revision_prompt_id") if assigned else None,
                "assigned_topic_score": assigned.get("score") if assigned else None,
                "eligible_for_manual_review": eligible,
                "eligibility_reasons": reasons,
                "manual_review_status": "not_reviewed",
                "final_stimulus_authorized": False,
            }
        )
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise HumanControlError("Candidate identifiers are not unique after deduplication")
    eligible_template_counts = {
        str(profile["template_id"]): counts.get(
            "template:{}".format(profile["template_id"]), 0
        )
        for profile in profiles
    }
    insufficient_templates = sorted(
        template_id
        for template_id, count in eligible_template_counts.items()
        if count < int(candidate_pool_minimum_per_template)
    )
    audit = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "artifact_type": "human_control_import_audit",
        "status": "CANDIDATE_SCREEN_ONLY_MANUAL_REVIEW_REQUIRED",
        "network_access_performed_by_pipeline": False,
        "source_id": source["source_id"],
        "source_contract_sha256": canonical_json_sha256(source),
        "source_url": source["source_url"],
        "dataset_revision": source["dataset_revision"],
        "license_identifier": source["license_identifier"],
        "license_url": source["license_url"],
        "attribution_name": source["attribution_name"],
        "source_file_name": source["source_file_name"],
        "source_file_size_bytes": actual_size,
        "source_file_sha256": actual_hash,
        "source_record_count": len(rows),
        "acquisition_date": acquired,
        "candidate_record_count": len(candidates),
        "counts": dict(sorted(counts.items())),
        "topic_profiles_sha256": canonical_json_sha256(list(profiles)),
        "pipeline_source_sha256": file_sha256(Path(__file__)),
        "eligible_template_counts": eligible_template_counts,
        "candidate_pool_minimum_per_template": int(candidate_pool_minimum_per_template),
        "insufficient_template_count": len(insufficient_templates),
        "insufficient_templates": insufficient_templates,
        "candidate_manifest_written": False,
        "candidate_manifest_sha256": None,
        "pre_recruitment_gate": (
            "BLOCKED_INSUFFICIENT_AUTOMATED_COVERAGE_AND_PENDING_MANUAL_REVIEW"
            if insufficient_templates
            else "BLOCKED_PENDING_MANUAL_TOPIC_PII_SAFETY_REVIEW_AND_LENGTH_TARGETS"
        ),
    }
    return candidates, audit


def write_import_artifacts(
    output_dir: Path,
    candidates: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
    audit_only: bool = False,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updated = dict(audit)
    if not audit_only:
        candidate_path = output_dir / "human_control_candidates.jsonl"
        write_jsonl(candidate_path, candidates)
        updated["candidate_manifest_written"] = True
        updated["candidate_manifest_sha256"] = file_sha256(candidate_path)
    write_json(output_dir / "import_audit.json", updated)
    return updated


def load_reviews(path: Path) -> Dict[str, Dict[str, object]]:
    rows = read_jsonl(path)
    reviews: Dict[str, Dict[str, object]] = {}
    for row in rows:
        missing = REQUIRED_REVIEW_FIELDS - set(row)
        if missing:
            raise HumanControlError("Review row missing {}".format(", ".join(sorted(missing))))
        candidate_id = str(row["candidate_id"])
        if candidate_id in reviews:
            raise HumanControlError("Duplicate review for {}".format(candidate_id))
        if not str(row["reviewer_id"]).strip() or "PLACEHOLDER" in str(row["reviewer_id"]).upper():
            raise HumanControlError("Review needs a non-placeholder reviewer identifier")
        _iso_date(row["reviewed_at"], "reviewed_at")
        reviews[candidate_id] = dict(row)
    return reviews


def _approved_review(candidate: Mapping[str, object], review: Mapping[str, object]) -> bool:
    return bool(
        review.get("template_id") == candidate.get("assigned_template_id")
        and review.get("topic_review_status") == "approved_exact_prompt_match"
        and review.get("pii_review_status") == "approved_no_personal_data"
        and review.get("safety_review_status") == "approved_safe_neutral"
    )


def validate_targets(
    targets: Sequence[Mapping[str, object]], design: Mapping[str, object]
) -> List[Dict[str, object]]:
    if any(REQUIRED_TARGET_FIELDS - set(row) for row in targets):
        raise HumanControlError("Every length target needs {}".format(sorted(REQUIRED_TARGET_FIELDS)))
    expected = {
        (
            category,
            "{}_template_{}".format(category, template_number),
            payload_class,
        )
        for category in design["prompt_categories"]
        for template_number in range(1, int(design["templates_per_category"]) + 1)
        for payload_class in design["eligible_payload_classes"]
    }
    observed = {
        (str(row["prompt_category"]), str(row["template_id"]), str(row["payload_class"]))
        for row in targets
    }
    if len(targets) != len(expected) or observed != expected:
        raise HumanControlError("Length targets do not cover the frozen 72 strata exactly")
    result = []
    for row in targets:
        value = dict(row)
        for field in ("target_word_count", "target_character_count"):
            try:
                number = int(value[field])
            except (TypeError, ValueError) as exc:
                raise HumanControlError("{} must be an integer".format(field)) from exc
            if number <= 0:
                raise HumanControlError("{} must be positive".format(field))
            value[field] = number
        result.append(value)
    return result


def _length_cost(candidate: Mapping[str, object], target: Mapping[str, object]) -> float:
    word = abs(int(candidate["word_count"]) - int(target["target_word_count"])) / int(
        target["target_word_count"]
    )
    character = abs(
        int(candidate["character_count"]) - int(target["target_character_count"])
    ) / int(target["target_character_count"])
    return word + 0.25 * character


def _optimal_unique_match(
    candidates: Sequence[Mapping[str, object]], targets: Sequence[Mapping[str, object]]
) -> List[Tuple[Mapping[str, object], Mapping[str, object]]]:
    ordered_candidates = sorted(
        candidates,
        key=lambda row: str(row["candidate_id"]),
    )
    ordered_targets = sorted(
        targets,
        key=lambda row: (
            int(row["target_word_count"]), int(row["target_character_count"]),
            str(row["payload_class"]),
        ),
    )
    required = len(ordered_targets)
    if len(ordered_candidates) < required:
        raise HumanControlError("A template has fewer reviewed candidates than targets")
    # There are four targets per template. A bit-mask assignment dynamic
    # program gives the global minimum over all unique candidate/target pairs
    # without depending on a heuristic ordering assumption.
    initial = tuple([-1] * required)
    states: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, initial)}
    for candidate_index, candidate in enumerate(ordered_candidates):
        updated = dict(states)
        for mask, current in states.items():
            for target_index, target in enumerate(ordered_targets):
                bit = 1 << target_index
                if mask & bit:
                    continue
                indices = list(current[1])
                indices[target_index] = candidate_index
                proposed = (
                    current[0] + _length_cost(candidate, target),
                    tuple(indices),
                )
                new_mask = mask | bit
                existing = updated.get(new_mask)
                if existing is None or proposed < existing:
                    updated[new_mask] = proposed
        states = updated
    final = states.get((1 << required) - 1)
    if final is None or any(index < 0 for index in final[1]):
        raise HumanControlError("Could not construct a complete length match")
    return [
        (ordered_candidates[candidate_index], ordered_targets[target_index])
        for target_index, candidate_index in enumerate(final[1])
    ]


def select_human_controls(
    candidates: Sequence[Mapping[str, object]],
    reviews: Mapping[str, Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
    design: Mapping[str, object],
    max_relative_word_difference: float = 0.35,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not (0 <= float(max_relative_word_difference) <= 1):
        raise HumanControlError("Length tolerance must be between zero and one")
    validated_targets = validate_targets(targets, design)
    validated_candidates = [validate_candidate(row) for row in candidates]
    candidate_ids = [str(row["candidate_id"]) for row in validated_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise HumanControlError("Candidate input contains duplicate candidate IDs")
    unknown_reviews = sorted(set(reviews) - set(candidate_ids))
    if unknown_reviews:
        raise HumanControlError(
            "Reviews reference unknown candidates; first: {}".format(unknown_reviews[0])
        )
    canonical_hashes = [
        str(row.get("canonical_text_sha256")) for row in validated_candidates
    ]
    if len(canonical_hashes) != len(set(canonical_hashes)):
        raise HumanControlError("Candidate input contains canonical text duplicates")
    approved = []
    for candidate in validated_candidates:
        if candidate.get("eligible_for_manual_review") is not True:
            continue
        candidate_id = str(candidate["candidate_id"])
        review = reviews.get(candidate_id)
        if review is not None and _approved_review(candidate, review):
            approved.append(candidate)
    by_template: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for candidate in approved:
        by_template[str(candidate["assigned_template_id"])].append(candidate)
    target_by_template: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for target in validated_targets:
        target_by_template[str(target["template_id"])].append(target)

    selected: List[Dict[str, object]] = []
    for template_id in sorted(target_by_template):
        matches = _optimal_unique_match(
            by_template.get(template_id, []), target_by_template[template_id]
        )
        for candidate, target in matches:
            relative_word = abs(
                int(candidate["word_count"]) - int(target["target_word_count"])
            ) / int(target["target_word_count"])
            relative_character = abs(
                int(candidate["character_count"])
                - int(target["target_character_count"])
            ) / int(target["target_character_count"])
            if relative_word > float(max_relative_word_difference):
                raise HumanControlError(
                    "No acceptable length match for {} / {}".format(
                        template_id, target["payload_class"]
                    )
                )
            review = reviews[str(candidate["candidate_id"])]
            selected.append(
                {
                    "stimulus_id": "HWC-{}-{}".format(
                        str(candidate["candidate_id"])[3:15], target["payload_class"]
                    ),
                    "condition": "human_written_control",
                    "prompt_category": target["prompt_category"],
                    "template_id": template_id,
                    "payload_id": target.get(
                        "payload_id", "human_length_stratum_{}".format(target["payload_class"])
                    ),
                    "payload_class": target["payload_class"],
                    "model_family": "human",
                    "presentation_scope": "full_message",
                    "message_text": candidate["message_text"],
                    "license_status": "verified_{}_attribution_and_sharealike_required".format(
                        candidate["license_identifier"]
                    ),
                    "safety_screen_status": "manual_approved_safe_neutral",
                    "source_id": candidate["source_id"],
                    "source_record_id": candidate["source_record_id"],
                    "source_url": candidate["source_url"],
                    "source_dataset_revision": candidate["source_dataset_revision"],
                    "source_record_sha256": candidate["source_record_sha256"],
                    "message_text_sha256": candidate["message_text_sha256"],
                    "attribution_name": candidate["attribution_name"],
                    "license_identifier": candidate["license_identifier"],
                    "license_url": candidate["license_url"],
                    "changes_made": candidate["changes_made"],
                    "acquisition_date": candidate["acquisition_date"],
                    "reviewer_id": review["reviewer_id"],
                    "reviewed_at": review["reviewed_at"],
                    "word_count": candidate["word_count"],
                    "character_count": candidate["character_count"],
                    "target_word_count": target["target_word_count"],
                    "target_character_count": target["target_character_count"],
                    "relative_word_difference": relative_word,
                    "relative_character_difference": relative_character,
                    "synthetic_fixture": "false",
                }
            )
    expected = int(design["stimuli_per_condition"])
    if len(selected) != expected:
        raise HumanControlError(
            "Selected {} human controls, expected {}".format(len(selected), expected)
        )
    text_hashes = [str(row["message_text_sha256"]) for row in selected]
    if len(text_hashes) != len(set(text_hashes)):
        raise HumanControlError("Selected human controls reuse source text")
    balance = {
        "prompt_category": dict(sorted(Counter(str(row["prompt_category"]) for row in selected).items())),
        "template_id": dict(sorted(Counter(str(row["template_id"]) for row in selected).items())),
        "payload_class": dict(sorted(Counter(str(row["payload_class"]) for row in selected).items())),
    }
    constraints = design.get("selection_constraints")
    if not isinstance(constraints, dict):
        raise HumanControlError("Frozen design lacks selection constraints")
    expected_balance = {
        "prompt_category": {
            str(category): int(constraints["stimuli_per_category_within_condition"])
            for category in design["prompt_categories"]
        },
        "template_id": {
            "{}_template_{}".format(category, number): int(
                constraints["stimuli_per_template_within_condition"]
            )
            for category in design["prompt_categories"]
            for number in range(1, int(design["templates_per_category"]) + 1)
        },
        "payload_class": {
            str(payload_class): int(
                constraints["stimuli_per_payload_class_within_condition"]
            )
            for payload_class in design["eligible_payload_classes"]
        },
    }
    expected_balance = {
        name: dict(sorted(values.items())) for name, values in expected_balance.items()
    }
    if balance != expected_balance:
        raise HumanControlError("Selected controls violate the frozen balance constraints")
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "artifact_type": "human_control_selection_audit",
        "status": "PRE_RECRUITMENT_CONTROL_SET_READY_FOR_IRB_REVIEW_NOT_RECRUITMENT",
        "selected_count": len(selected),
        "source_texts_unique": True,
        "manual_review_required_and_present": True,
        "max_relative_word_difference_allowed": float(max_relative_word_difference),
        "max_relative_word_difference_observed": max(
            float(row["relative_word_difference"]) for row in selected
        ),
        "balance": balance,
        "license_identifiers": sorted({str(row["license_identifier"]) for row in selected}),
        "recruitment_authorized": False,
        "human_exposure_authorized": False,
    }
    return sorted(
        selected,
        key=lambda row: (
            str(row["prompt_category"]), str(row["template_id"]), str(row["payload_class"])
        ),
    ), manifest


SELECTION_FIELDS = (
    "stimulus_id", "condition", "prompt_category", "template_id", "payload_id",
    "payload_class", "model_family", "presentation_scope", "message_text",
    "license_status", "safety_screen_status", "source_id", "source_record_id",
    "source_url", "source_dataset_revision", "source_record_sha256",
    "message_text_sha256", "attribution_name", "license_identifier", "license_url",
    "changes_made", "acquisition_date", "reviewer_id", "reviewed_at", "word_count",
    "character_count", "target_word_count", "target_character_count",
    "relative_word_difference", "relative_character_difference", "synthetic_fixture",
)


def write_selection_artifacts(
    output_dir: Path,
    selected: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "human_written_controls.csv"
    write_csv(selection_path, selected, SELECTION_FIELDS)
    updated = dict(manifest)
    updated["selection_sha256"] = file_sha256(selection_path)
    updated["selection_rows_sha256"] = canonical_json_sha256(list(selected))
    write_json(output_dir / "selection_audit.json", updated)
    attribution_lines = [
        "RankCloak human-written controls attribution",
        "",
        "The selected texts are excerpts from databricks-dolly-15k, version 1.0,",
        "Copyright (2023) Databricks, Inc., licensed CC BY-SA 3.0 Unported.",
        "Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "License: https://creativecommons.org/licenses/by-sa/3.0/",
        "Changes: line endings and surrounding whitespace were normalized; no semantic edits.",
        "",
        "Every row retains its pinned revision, source-record hash, and text hash.",
        "No endorsement by Databricks or dataset contributors is implied.",
    ]
    _atomic_write(
        output_dir / "ATTRIBUTION.txt", ("\n".join(attribution_lines) + "\n").encode("utf-8")
    )
    updated["attribution_sha256"] = file_sha256(output_dir / "ATTRIBUTION.txt")
    write_json(output_dir / "selection_audit.json", updated)
    return updated


__all__ = [
    "DEFAULT_DESIGN", "DEFAULT_PROFILES", "DEFAULT_REGISTRY", "HumanControlError",
    "PIPELINE_SCHEMA_VERSION", "SELECTION_FIELDS", "assign_topic", "canonical_json_sha256",
    "canonical_text", "file_sha256", "import_registered_source", "load_json",
    "load_reviews", "pii_flags", "quality_flags", "read_csv", "read_jsonl",
    "select_human_controls", "topic_scores", "unsafe_content_flags",
    "validate_candidate", "validate_profiles", "validate_source_registry", "validate_targets",
    "write_import_artifacts", "write_selection_artifacts",
]
