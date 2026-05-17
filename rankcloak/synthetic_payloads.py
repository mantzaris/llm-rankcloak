"""Deterministic synthetic cryptographic-artifact-like payloads."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from typing import List


DEFAULT_SEED = 20260516


@dataclass(frozen=True)
class SyntheticPayload:
    name: str
    kind: str
    text: str
    bytes_value: bytes
    description: str
    is_synthetic: bool = True


def deterministic_random_bytes(length: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(0, 256) for _ in range(length))


def base64url_no_padding(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def synthetic_uuid_v4(seed: int) -> str:
    raw = bytearray(deterministic_random_bytes(16, seed))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def synthetic_jwt_like(seed: int) -> str:
    header = {"alg": "none", "typ": "JWT", "kid": "rankcloak-synthetic"}
    payload = {"sub": "synthetic-user", "iat": 0, "scope": "research-only"}
    signature = deterministic_random_bytes(24, seed)
    return ".".join(
        [
            base64url_no_padding(json.dumps(header, sort_keys=True).encode("utf-8")),
            base64url_no_padding(json.dumps(payload, sort_keys=True).encode("utf-8")),
            "invalid-" + base64url_no_padding(signature),
        ]
    )


def generate_synthetic_payloads(seed: int = DEFAULT_SEED) -> List[SyntheticPayload]:
    """Create deterministic fake artifacts; none are real secrets or credentials."""

    public_test_string = "RankCloak public deterministic test string v1"
    sha256_digest = hashlib.sha256(public_test_string.encode("utf-8")).hexdigest()
    random_128_hex = deterministic_random_bytes(16, seed + 1).hex()
    random_256_hex = deterministic_random_bytes(32, seed + 2).hex()
    nonce_hex = deterministic_random_bytes(12, seed + 3).hex()
    uuid_text = synthetic_uuid_v4(seed + 4)
    bearer = "fake_bearer_" + base64url_no_padding(deterministic_random_bytes(32, seed + 5))
    hmac_like = "hmac_sha256_fake_" + deterministic_random_bytes(32, seed + 6).hex()
    jwt_like = synthetic_jwt_like(seed + 7)
    ciphertext_like = base64.b64encode(deterministic_random_bytes(48, seed + 8)).decode("ascii")

    rows = [
        ("sha256_public_test_string", "sha256_hex", sha256_digest, "SHA-256 digest of a public string"),
        ("random_128_bit_hex", "random_hex", random_128_hex, "Fixed-seed 128-bit hex string"),
        ("random_256_bit_hex", "random_hex", random_256_hex, "Fixed-seed 256-bit hex string"),
        ("nonce_96_bit_hex", "nonce_hex", nonce_hex, "Fixed-seed 96-bit nonce-like hex string"),
        ("synthetic_uuid_v4_like", "uuid", uuid_text, "Fixed-seed synthetic UUIDv4-like value"),
        (
            "synthetic_bearer_token_like",
            "bearer_like",
            bearer,
            "Clearly fake synthetic bearer-token-like string",
        ),
        (
            "synthetic_hmac_like_tag",
            "hmac_like",
            hmac_like,
            "Clearly fake synthetic HMAC-tag-like hex string",
        ),
        (
            "synthetic_jwt_like_invalid_signature",
            "jwt_like",
            jwt_like,
            "Invalid synthetic JWT-like string with fake signature material",
        ),
        (
            "synthetic_ciphertext_like_base64",
            "ciphertext_like",
            ciphertext_like,
            "Fixed-seed synthetic base64 ciphertext-like block",
        ),
    ]

    return [
        SyntheticPayload(
            name=name,
            kind=kind,
            text=text,
            bytes_value=text.encode("utf-8"),
            description=description,
        )
        for name, kind, text, description in rows
    ]
