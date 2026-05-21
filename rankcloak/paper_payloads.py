"""Deterministic paper-oriented payload suite for RankCloak."""

from __future__ import annotations

import base64
import hashlib
import random
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


PAPER_BASE_SEED = 2026052101
PILOT_PAYLOAD_CLASSES = [
    "sha256_hex",
    "random_128_bit_hex",
    "random_256_bit_hex",
    "nonce_96_bit_hex",
    "uuid_v4_like",
    "ciphertext_like_base64",
]
FULL_PAYLOAD_CLASSES = [
    "sha256_hex",
    "random_128_bit_hex",
    "random_256_bit_hex",
    "nonce_96_bit_hex",
    "uuid_v4_like",
    "hmac_like_hex",
    "ciphertext_like_base64",
]


@dataclass(frozen=True)
class PaperPayload:
    payload_name: str
    payload_class: str
    payload_kind: str
    payload_text: str
    payload_bytes: bytes
    artifact_bit_length_if_known: int
    representation_hint: str
    is_hex_like: bool
    is_base64_like: bool
    is_structured: bool
    seed: int
    notes: str

    @property
    def name(self) -> str:
        return self.payload_name

    @property
    def kind(self) -> str:
        return self.payload_kind

    @property
    def text(self) -> str:
        return self.payload_text

    @property
    def bytes_value(self) -> bytes:
        return self.payload_bytes


def deterministic_random_bytes(length: int, seed: int) -> bytes:
    rng = random.Random(int(seed))
    return bytes(rng.randrange(0, 256) for _ in range(int(length)))


def deterministic_uuid_v4(seed: int) -> str:
    raw = bytearray(deterministic_random_bytes(16, seed))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def base64_standard(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def class_seed(payload_class: str, index: int) -> int:
    offsets = {
        "sha256_hex": 1000,
        "random_128_bit_hex": 2000,
        "random_256_bit_hex": 3000,
        "nonce_96_bit_hex": 4000,
        "uuid_v4_like": 5000,
        "hmac_like_hex": 6000,
        "ciphertext_like_base64": 7000,
    }
    return PAPER_BASE_SEED + offsets[payload_class] + int(index)


def make_payload(payload_class: str, index: int) -> PaperPayload:
    seed = class_seed(payload_class, index)
    suffix = "{:03d}".format(index)
    if payload_class == "sha256_hex":
        public_text = "rankcloak-paper-public-seed-{}".format(suffix)
        payload_text = hashlib.sha256(public_text.encode("utf-8")).hexdigest()
        return PaperPayload(
            payload_name="paper_sha256_hex_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="sha256_hex",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=256,
            representation_hint="raw_hex_nibbles",
            is_hex_like=True,
            is_base64_like=False,
            is_structured=False,
            seed=seed,
            notes="SHA-256 digest of a public deterministic string.",
        )
    if payload_class == "random_128_bit_hex":
        payload_text = deterministic_random_bytes(16, seed).hex()
        return PaperPayload(
            payload_name="paper_random_128_bit_hex_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="random_hex",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=128,
            representation_hint="raw_hex_nibbles",
            is_hex_like=True,
            is_base64_like=False,
            is_structured=False,
            seed=seed,
            notes="Fixed-seed synthetic 128-bit hex string.",
        )
    if payload_class == "random_256_bit_hex":
        payload_text = deterministic_random_bytes(32, seed).hex()
        return PaperPayload(
            payload_name="paper_random_256_bit_hex_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="random_hex",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=256,
            representation_hint="raw_hex_nibbles",
            is_hex_like=True,
            is_base64_like=False,
            is_structured=False,
            seed=seed,
            notes="Fixed-seed synthetic 256-bit hex string.",
        )
    if payload_class == "nonce_96_bit_hex":
        payload_text = deterministic_random_bytes(12, seed).hex()
        return PaperPayload(
            payload_name="paper_nonce_96_bit_hex_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="nonce_hex",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=96,
            representation_hint="raw_hex_nibbles",
            is_hex_like=True,
            is_base64_like=False,
            is_structured=False,
            seed=seed,
            notes="Fixed-seed synthetic nonce-like hex string.",
        )
    if payload_class == "uuid_v4_like":
        payload_text = deterministic_uuid_v4(seed)
        return PaperPayload(
            payload_name="paper_uuid_v4_like_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="uuid_v4_like",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=128,
            representation_hint="ascii_bytes_fixed_radix",
            is_hex_like=False,
            is_base64_like=False,
            is_structured=True,
            seed=seed,
            notes="Fixed-seed synthetic UUIDv4-like string.",
        )
    if payload_class == "hmac_like_hex":
        payload_text = deterministic_random_bytes(32, seed).hex()
        return PaperPayload(
            payload_name="paper_hmac_like_hex_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="hmac_like_hex",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=256,
            representation_hint="raw_hex_nibbles",
            is_hex_like=True,
            is_base64_like=False,
            is_structured=False,
            seed=seed,
            notes="Fixed-seed synthetic HMAC-like tag; no real key or HMAC is used.",
        )
    if payload_class == "ciphertext_like_base64":
        raw = deterministic_random_bytes(48, seed)
        payload_text = base64_standard(raw)
        return PaperPayload(
            payload_name="paper_ciphertext_like_base64_{}".format(suffix),
            payload_class=payload_class,
            payload_kind="ciphertext_like_base64",
            payload_text=payload_text,
            payload_bytes=payload_text.encode("utf-8"),
            artifact_bit_length_if_known=len(raw) * 8,
            representation_hint="ascii_bytes_fixed_radix",
            is_hex_like=False,
            is_base64_like=True,
            is_structured=False,
            seed=seed,
            notes="Fixed-seed synthetic base64 ciphertext-like text; no real encryption is used.",
        )
    raise ValueError("Unknown paper payload class: {}".format(payload_class))


def generate_paper_payloads(
    instances_per_class: int,
    payload_classes: Sequence[str],
) -> List[PaperPayload]:
    payloads: List[PaperPayload] = []
    for payload_class in payload_classes:
        for index in range(int(instances_per_class)):
            payloads.append(make_payload(payload_class, index))
    names = [payload.payload_name for payload in payloads]
    if len(names) != len(set(names)):
        raise ValueError("Paper payload names must be unique.")
    return payloads


def generate_pilot_paper_payloads() -> List[PaperPayload]:
    return generate_paper_payloads(2, PILOT_PAYLOAD_CLASSES)


def generate_full_paper_payloads() -> List[PaperPayload]:
    return generate_paper_payloads(5, FULL_PAYLOAD_CLASSES)


def payload_class_counts(payloads: Iterable[PaperPayload]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for payload in payloads:
        counts[payload.payload_class] = counts.get(payload.payload_class, 0) + 1
    return counts


def paper_payload_rows(payloads: Iterable[PaperPayload]) -> List[dict]:
    rows = []
    for payload in payloads:
        rows.append(
            {
                "payload_name": payload.payload_name,
                "payload_class": payload.payload_class,
                "payload_kind": payload.payload_kind,
                "artifact_text_character_length": len(payload.payload_text),
                "artifact_text_byte_length": len(payload.payload_bytes),
                "artifact_bit_length_if_known": payload.artifact_bit_length_if_known,
                "representation_hint": payload.representation_hint,
                "is_hex_like": payload.is_hex_like,
                "is_base64_like": payload.is_base64_like,
                "is_structured": payload.is_structured,
                "seed": payload.seed,
                "notes": payload.notes,
            }
        )
    return rows
