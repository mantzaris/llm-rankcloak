"""Deterministic real-cryptographic payload corpus for the major revision.

The values produced here are public test vectors.  Keys, nonces, messages, and
signing seeds are derived from domain-separated public seed material.  They
must never be used for operational security.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305


REVISION_CORPUS_ID = "rankcloak_scientific_reports_revision_v1"
REVISION_DERIVATION_VERSION = "shake256-domain-separated-v1"
REVISION_PUBLIC_SEED_MATERIAL = (
    b"RankCloak Scientific Reports major revision v1 public deterministic "
    b"test-vector seed; NOT SECRET; DO NOT USE OPERATIONALLY"
)
REVISION_INSTANCES_PER_CLASS = 60
REVISION_CORPUS_SHA256 = (
    "caf0db84c814e02474a3cd2fc5588a8283cbe00fe41ce448764c6cdc67baa8c0"
)
REVISION_PAYLOAD_CLASSES: Tuple[str, ...] = (
    "sha256_hex",
    "hmac_sha256_hex",
    "nonce_96_bit_hex",
    "token_128_bit_hex",
    "uuid_v4",
    "aes256_gcm_base64",
    "chacha20_poly1305_base64",
    "ed25519_signature_base64",
)
REVISION_HEX_PAYLOAD_CLASSES: Tuple[str, ...] = REVISION_PAYLOAD_CLASSES[:4]
REVISION_BASE64_PAYLOAD_CLASSES: Tuple[str, ...] = REVISION_PAYLOAD_CLASSES[5:]

_AES_AAD = b"rankcloak/revision_v1/aes256_gcm/aad"
_CHACHA_AAD = b"rankcloak/revision_v1/chacha20_poly1305/aad"


@dataclass(frozen=True)
class RevisionPayload:
    """One immutable public test payload and its non-secret provenance."""

    payload_name: str
    payload_class: str
    payload_index: int
    payload_kind: str
    payload_text: str
    payload_bytes: bytes
    artifact_bytes: bytes
    artifact_bit_length: int
    representation_hint: str
    is_hex_like: bool
    is_base64_like: bool
    is_structured: bool
    algorithm: str
    algorithm_parameters: Mapping[str, object]
    public_seed_label: str
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
        """Return display bytes, matching the existing ASCII codec contract."""

        return self.payload_bytes

    def manifest_record(self, include_payload_text: bool = True) -> Dict[str, object]:
        record: Dict[str, object] = {
            "payload_name": self.payload_name,
            "payload_class": self.payload_class,
            "payload_index": self.payload_index,
            "payload_kind": self.payload_kind,
            "payload_text_character_length": len(self.payload_text),
            "payload_text_byte_length": len(self.payload_bytes),
            "artifact_byte_length": len(self.artifact_bytes),
            "artifact_bit_length": self.artifact_bit_length,
            "artifact_sha256": hashlib.sha256(self.artifact_bytes).hexdigest(),
            "payload_text_sha256": hashlib.sha256(self.payload_bytes).hexdigest(),
            "representation_hint": self.representation_hint,
            "is_hex_like": self.is_hex_like,
            "is_base64_like": self.is_base64_like,
            "is_structured": self.is_structured,
            "algorithm": self.algorithm,
            "algorithm_parameters": dict(self.algorithm_parameters),
            "corpus_id": REVISION_CORPUS_ID,
            "derivation_version": REVISION_DERIVATION_VERSION,
            "public_seed_label": self.public_seed_label,
            "notes": self.notes,
        }
        if include_payload_text:
            record["payload_text"] = self.payload_text
        return record


def _domain_material(payload_class: str, index: int, purpose: str) -> bytes:
    if payload_class not in REVISION_PAYLOAD_CLASSES:
        raise ValueError("Unknown revision payload class: {}".format(payload_class))
    if int(index) < 0:
        raise ValueError("payload index must be non-negative")
    label = "{}/{}/{:03d}/{}".format(
        REVISION_CORPUS_ID, payload_class, int(index), purpose
    )
    return REVISION_PUBLIC_SEED_MATERIAL + b"\x00" + label.encode("utf-8")


def derive_public_bytes(payload_class: str, index: int, purpose: str, length: int) -> bytes:
    """Derive deterministic, domain-separated public bytes with SHAKE-256."""

    if int(length) < 0:
        raise ValueError("length must be non-negative")
    return hashlib.shake_256(
        _domain_material(payload_class, int(index), purpose)
    ).digest(int(length))


def _seed_label(payload_class: str, index: int) -> str:
    return "{}/{}/{:03d}".format(REVISION_CORPUS_ID, payload_class, int(index))


def _payload(
    payload_class: str,
    index: int,
    payload_kind: str,
    payload_text: str,
    artifact_bytes: bytes,
    representation_hint: str,
    is_hex_like: bool,
    is_base64_like: bool,
    is_structured: bool,
    algorithm: str,
    algorithm_parameters: Mapping[str, object],
    notes: str,
) -> RevisionPayload:
    return RevisionPayload(
        payload_name="revision_v1_{}_{:03d}".format(payload_class, int(index)),
        payload_class=payload_class,
        payload_index=int(index),
        payload_kind=payload_kind,
        payload_text=payload_text,
        payload_bytes=payload_text.encode("ascii"),
        artifact_bytes=bytes(artifact_bytes),
        artifact_bit_length=len(artifact_bytes) * 8,
        representation_hint=representation_hint,
        is_hex_like=is_hex_like,
        is_base64_like=is_base64_like,
        is_structured=is_structured,
        algorithm=algorithm,
        algorithm_parameters=dict(algorithm_parameters),
        public_seed_label=_seed_label(payload_class, index),
        notes=notes,
    )


def make_revision_payload(payload_class: str, index: int) -> RevisionPayload:
    """Construct one standards-based public cryptographic test artifact."""

    index = int(index)
    if payload_class == "sha256_hex":
        message = derive_public_bytes(payload_class, index, "message", 48)
        digest = hashlib.sha256(message).digest()
        return _payload(
            payload_class,
            index,
            "sha256_digest",
            digest.hex(),
            digest,
            "raw_hex_nibbles",
            True,
            False,
            False,
            "SHA-256",
            {
                "digest_length_bytes": 32,
                "message_length_bytes": len(message),
                "message_sha256": hashlib.sha256(message).hexdigest(),
            },
            "SHA-256 digest of domain-separated public test bytes.",
        )

    if payload_class == "hmac_sha256_hex":
        key = derive_public_bytes(payload_class, index, "public-test-key", 32)
        message = derive_public_bytes(payload_class, index, "message", 48)
        tag = hmac.new(key, message, hashlib.sha256).digest()
        return _payload(
            payload_class,
            index,
            "hmac_sha256_tag",
            tag.hex(),
            tag,
            "raw_hex_nibbles",
            True,
            False,
            False,
            "HMAC-SHA-256",
            {
                "tag_length_bytes": 32,
                "message_length_bytes": len(message),
                "message_sha256": hashlib.sha256(message).hexdigest(),
                "key_derivation_purpose": "public-test-key",
            },
            "HMAC-SHA-256 tag made with a publicly derivable non-secret test key.",
        )

    if payload_class == "nonce_96_bit_hex":
        nonce = derive_public_bytes(payload_class, index, "nonce", 12)
        return _payload(
            payload_class,
            index,
            "cryptographic_nonce",
            nonce.hex(),
            nonce,
            "raw_hex_nibbles",
            True,
            False,
            False,
            "SHAKE-256 deterministic test-vector derivation",
            {"nonce_length_bytes": 12},
            "Public deterministic 96-bit nonce test vector.",
        )

    if payload_class == "token_128_bit_hex":
        token = derive_public_bytes(payload_class, index, "token", 16)
        return _payload(
            payload_class,
            index,
            "cryptographic_token",
            token.hex(),
            token,
            "raw_hex_nibbles",
            True,
            False,
            False,
            "SHAKE-256 deterministic test-vector derivation",
            {"token_length_bytes": 16},
            "Public deterministic 128-bit token test vector.",
        )

    if payload_class == "uuid_v4":
        raw = bytearray(derive_public_bytes(payload_class, index, "uuid", 16))
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        value = uuid.UUID(bytes=bytes(raw))
        return _payload(
            payload_class,
            index,
            "uuid_v4",
            str(value),
            value.bytes,
            "ascii_bytes_fixed_radix",
            False,
            False,
            True,
            "UUID version 4 bit layout over SHAKE-256 test bytes",
            {"uuid_version": 4, "uuid_variant": "RFC 4122"},
            "Standards-conformant UUIDv4 with deterministic public random bits.",
        )

    if payload_class == "aes256_gcm_base64":
        key = derive_public_bytes(payload_class, index, "public-test-key", 32)
        nonce = derive_public_bytes(payload_class, index, "nonce", 12)
        plaintext = derive_public_bytes(payload_class, index, "plaintext", 32)
        ciphertext_and_tag = AESGCM(key).encrypt(nonce, plaintext, _AES_AAD)
        return _payload(
            payload_class,
            index,
            "aes256_gcm_ciphertext_and_tag",
            base64.b64encode(ciphertext_and_tag).decode("ascii"),
            ciphertext_and_tag,
            "ascii_bytes_fixed_radix",
            False,
            True,
            False,
            "AES-256-GCM",
            {
                "key_length_bits": 256,
                "nonce_hex": nonce.hex(),
                "nonce_length_bytes": 12,
                "plaintext_length_bytes": len(plaintext),
                "ciphertext_length_bytes": len(plaintext),
                "tag_length_bytes": 16,
                "aad_utf8": _AES_AAD.decode("ascii"),
                "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
                "key_derivation_purpose": "public-test-key",
            },
            "AES-256-GCM ciphertext and tag from public deterministic test material.",
        )

    if payload_class == "chacha20_poly1305_base64":
        key = derive_public_bytes(payload_class, index, "public-test-key", 32)
        nonce = derive_public_bytes(payload_class, index, "nonce", 12)
        plaintext = derive_public_bytes(payload_class, index, "plaintext", 32)
        ciphertext_and_tag = ChaCha20Poly1305(key).encrypt(
            nonce, plaintext, _CHACHA_AAD
        )
        return _payload(
            payload_class,
            index,
            "chacha20_poly1305_ciphertext_and_tag",
            base64.b64encode(ciphertext_and_tag).decode("ascii"),
            ciphertext_and_tag,
            "ascii_bytes_fixed_radix",
            False,
            True,
            False,
            "ChaCha20-Poly1305",
            {
                "key_length_bits": 256,
                "nonce_hex": nonce.hex(),
                "nonce_length_bytes": 12,
                "plaintext_length_bytes": len(plaintext),
                "ciphertext_length_bytes": len(plaintext),
                "tag_length_bytes": 16,
                "aad_utf8": _CHACHA_AAD.decode("ascii"),
                "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
                "key_derivation_purpose": "public-test-key",
            },
            "ChaCha20-Poly1305 ciphertext and tag from public deterministic test material.",
        )

    if payload_class == "ed25519_signature_base64":
        private_seed = derive_public_bytes(payload_class, index, "public-signing-seed", 32)
        message = derive_public_bytes(payload_class, index, "message", 48)
        private_key = Ed25519PrivateKey.from_private_bytes(private_seed)
        signature = private_key.sign(message)
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return _payload(
            payload_class,
            index,
            "ed25519_signature",
            base64.b64encode(signature).decode("ascii"),
            signature,
            "ascii_bytes_fixed_radix",
            False,
            True,
            False,
            "Ed25519",
            {
                "signature_length_bytes": 64,
                "message_length_bytes": len(message),
                "message_sha256": hashlib.sha256(message).hexdigest(),
                "public_key_base64": base64.b64encode(public_key).decode("ascii"),
                "private_seed_derivation_purpose": "public-signing-seed",
            },
            "Ed25519 signature made with a publicly derivable non-secret test key.",
        )

    raise ValueError("Unknown revision payload class: {}".format(payload_class))


def generate_revision_payloads(
    instances_per_class: int = REVISION_INSTANCES_PER_CLASS,
    payload_classes: Sequence[str] = REVISION_PAYLOAD_CLASSES,
) -> List[RevisionPayload]:
    """Generate a deterministic class-major corpus."""

    instances_per_class = int(instances_per_class)
    if instances_per_class <= 0:
        raise ValueError("instances_per_class must be positive")
    if len(set(payload_classes)) != len(payload_classes):
        raise ValueError("payload_classes must be unique")
    payloads = [
        make_revision_payload(payload_class, index)
        for payload_class in payload_classes
        for index in range(instances_per_class)
    ]
    names = [payload.payload_name for payload in payloads]
    if len(names) != len(set(names)):
        raise ValueError("revision payload names must be unique")
    return payloads


def generate_revision_v1_payloads() -> List[RevisionPayload]:
    """Generate the frozen 8-by-60 confirmatory corpus."""

    return generate_revision_payloads(
        instances_per_class=REVISION_INSTANCES_PER_CLASS,
        payload_classes=REVISION_PAYLOAD_CLASSES,
    )


def validate_revision_payload(payload: RevisionPayload) -> bool:
    """Recompute and cryptographically verify one public test vector."""

    expected = make_revision_payload(payload.payload_class, payload.payload_index)
    if payload != expected:
        return False

    payload_class = payload.payload_class
    index = payload.payload_index
    if payload.is_hex_like:
        if payload.payload_text != payload.payload_text.lower():
            return False
        try:
            if bytes.fromhex(payload.payload_text) != payload.artifact_bytes:
                return False
        except ValueError:
            return False
    if payload.is_base64_like:
        try:
            decoded = base64.b64decode(payload.payload_text, validate=True)
        except Exception:
            return False
        if decoded != payload.artifact_bytes:
            return False

    if payload_class == "uuid_v4":
        parsed = uuid.UUID(payload.payload_text)
        return parsed.version == 4 and parsed.variant == uuid.RFC_4122

    if payload_class == "aes256_gcm_base64":
        key = derive_public_bytes(payload_class, index, "public-test-key", 32)
        nonce = derive_public_bytes(payload_class, index, "nonce", 12)
        plaintext = derive_public_bytes(payload_class, index, "plaintext", 32)
        try:
            recovered = AESGCM(key).decrypt(nonce, payload.artifact_bytes, _AES_AAD)
        except Exception:
            return False
        return hmac.compare_digest(recovered, plaintext)

    if payload_class == "chacha20_poly1305_base64":
        key = derive_public_bytes(payload_class, index, "public-test-key", 32)
        nonce = derive_public_bytes(payload_class, index, "nonce", 12)
        plaintext = derive_public_bytes(payload_class, index, "plaintext", 32)
        try:
            recovered = ChaCha20Poly1305(key).decrypt(
                nonce, payload.artifact_bytes, _CHACHA_AAD
            )
        except Exception:
            return False
        return hmac.compare_digest(recovered, plaintext)

    if payload_class == "ed25519_signature_base64":
        private_seed = derive_public_bytes(
            payload_class, index, "public-signing-seed", 32
        )
        message = derive_public_bytes(payload_class, index, "message", 48)
        public_key = Ed25519PrivateKey.from_private_bytes(
            private_seed
        ).public_key()
        try:
            public_key.verify(payload.artifact_bytes, message)
        except InvalidSignature:
            return False
    return True


def revision_payload_records(
    payloads: Optional[Iterable[RevisionPayload]] = None,
    include_payload_text: bool = True,
) -> List[Dict[str, object]]:
    selected = list(payloads) if payloads is not None else generate_revision_v1_payloads()
    return [
        payload.manifest_record(include_payload_text=include_payload_text)
        for payload in selected
    ]


def revision_corpus_sha256(
    payloads: Optional[Iterable[RevisionPayload]] = None,
) -> str:
    """Hash canonical JSON records for the corpus, including payload values."""

    records = revision_payload_records(payloads, include_payload_text=True)
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def revision_payload_class_counts(
    payloads: Iterable[RevisionPayload],
) -> Dict[str, int]:
    counts = {payload_class: 0 for payload_class in REVISION_PAYLOAD_CLASSES}
    for payload in payloads:
        if payload.payload_class not in counts:
            raise ValueError(
                "Unexpected revision payload class: {}".format(payload.payload_class)
            )
        counts[payload.payload_class] += 1
    return counts


def validate_revision_corpus(
    payloads: Optional[Iterable[RevisionPayload]] = None,
    expected_sha256: Optional[str] = REVISION_CORPUS_SHA256,
) -> Dict[str, object]:
    """Validate identity, counts, formats, algorithms, and optional frozen hash."""

    selected = list(payloads) if payloads is not None else generate_revision_v1_payloads()
    expected_count = len(REVISION_PAYLOAD_CLASSES) * REVISION_INSTANCES_PER_CLASS
    names = [payload.payload_name for payload in selected]
    counts = revision_payload_class_counts(selected)
    digest = revision_corpus_sha256(selected)
    errors: List[str] = []
    if len(selected) != expected_count:
        errors.append(
            "expected {} payloads, found {}".format(expected_count, len(selected))
        )
    if len(names) != len(set(names)):
        errors.append("payload names are not unique")
    for payload_class in REVISION_PAYLOAD_CLASSES:
        actual = counts.get(payload_class, 0)
        if actual != REVISION_INSTANCES_PER_CLASS:
            errors.append(
                "{} has {} payloads, expected {}".format(
                    payload_class, actual, REVISION_INSTANCES_PER_CLASS
                )
            )
    invalid = [
        payload.payload_name
        for payload in selected
        if not validate_revision_payload(payload)
    ]
    if invalid:
        errors.append("{} payloads failed cryptographic validation".format(len(invalid)))
    if expected_sha256 is not None and digest != expected_sha256:
        errors.append(
            "corpus SHA-256 mismatch: expected {}, found {}".format(
                expected_sha256, digest
            )
        )
    return {
        "status": "ok" if not errors else "error",
        "corpus_id": REVISION_CORPUS_ID,
        "payload_count": len(selected),
        "class_counts": counts,
        "corpus_sha256": digest,
        "invalid_payload_names": invalid,
        "errors": errors,
    }
