import base64
import hashlib

import numpy as np
import pytest

from rankcloak.model_io import make_context_token_ids
from rankcloak.revision_protocol import (
    TAIL_NONE,
    apply_transmission_transform,
    bounded_representation,
    decode_representation,
    diagnose_rank_failure,
    direct_representation,
    dynamic_tail_complete,
    first_divergence,
    generate_rank_span,
    recover_rank_span,
    retokenize_message,
    transform_token_ids,
)


class TinyRankModel:
    """Deterministic llama.cpp-shaped model used for protocol unit tests."""

    pieces = ["", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    def __init__(self):
        self.n_tokens = 0
        self.scores = np.zeros((256, len(self.pieces)), dtype=float)

    def token_bos(self):
        return 0

    def n_vocab(self):
        return len(self.pieces)

    def reset(self):
        self.n_tokens = 0

    def eval(self, token_ids):
        for _token_id in token_ids:
            # Stable order is token id 0, 1, ...; context scheduling still matters
            # because every eval call advances the exposed score row.
            self.scores[self.n_tokens] = -np.arange(len(self.pieces), dtype=float)
            self.n_tokens += 1

    def tokenize(self, value, add_bos=True):
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        ids = [self.pieces.index(character) for character in text]
        return ([0] if add_bos else []) + ids

    def detokenize(self, token_ids):
        return "".join(self.pieces[int(token_id)] for token_id in token_ids).encode()


class NoBosRankModel(TinyRankModel):
    """Models a GGUF with add_bos_token=false even when requested."""

    def __init__(self):
        super().__init__()
        self.tokenize_calls = []

    def tokenize(self, value, add_bos=True, special=False):
        self.tokenize_calls.append((bool(add_bos), bool(special)))
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return [self.pieces.index(character) for character in text]


class LeadingSpaceFusionTrapModel(NoBosRankModel):
    """Would fuse an artificial space with one or more payload symbols."""

    def __init__(self, fused_payload_symbols):
        super().__init__()
        self.fused_payload_symbols = int(fused_payload_symbols)

    def tokenize(self, value, add_bos=True, special=False):
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        self.tokenize_calls.append((bool(add_bos), bool(special)))
        if text.startswith(" "):
            return [10] + [
                self.pieces.index(character)
                for character in text[1 + self.fused_payload_symbols :]
            ]
        return [self.pieces.index(character) for character in text]


class ImplicitSpaceRankModel(NoBosRankModel):
    """Models deterministic SentencePiece-style detokenization framing."""

    def __init__(self, prefix=b" "):
        super().__init__()
        self.prefix = bytes(prefix)

    def detokenize(self, token_ids):
        return self.prefix + super().detokenize(token_ids)


class NonReversibleRankModel(NoBosRankModel):
    def detokenize(self, token_ids):
        return super().detokenize(token_ids)[:-1]


class NonSpacePrefixRankModel(NoBosRankModel):
    def detokenize(self, token_ids):
        return b"\t" + super().detokenize(token_ids)


class InternalMutationRankModel(NoBosRankModel):
    def detokenize(self, token_ids):
        value = super().detokenize(token_ids)
        return b"x" + value[1:]


class SuffixMutationRankModel(NoBosRankModel):
    def detokenize(self, token_ids):
        return super().detokenize(token_ids) + b"x"


class BosOnlyPromptModel(TinyRankModel):
    def tokenize(self, value, add_bos=True):
        return [self.token_bos()] if add_bos else []


def test_bounded_exact_token_replay_roundtrip():
    model = TinyRankModel()
    representation = bounded_representation(b"a", "a", "ascii_b8")
    generated = generate_rank_span(
        model,
        context_token_ids=[0],
        ranks=representation.ranks,
        tail_policy=TAIL_NONE,
    )
    recovered = recover_rank_span(
        model,
        [0],
        generated["leadin_token_ids"],
        generated["forced_token_ids"],
    )
    decoded = decode_representation(model, representation, recovered["ranks"])
    assert recovered["ranks"] == list(representation.ranks)
    assert decoded["exact_recovery"] is True


def test_bounded_generation_records_same_context_quality_endpoints():
    model = TinyRankModel()
    generated = generate_rank_span(
        model,
        context_token_ids=[0],
        ranks=[1, 3, 2],
        tail_policy=TAIL_NONE,
        quality_rank_ceiling=4,
    )
    assert generated["realized_ranks"] == [1, 3, 2]
    assert generated["quality_rank_ceiling"] == 4
    assert generated["greedy_token_ids"] == [0, 0, 0]
    assert generated["rank_B_token_ids"] == [3, 3, 3]
    assert len(generated["forced_log_probabilities"]) == 3
    assert len(generated["greedy_log_probabilities"]) == 3
    assert len(generated["rank_B_log_probabilities"]) == 3
    for realized, greedy, upper in zip(
        generated["forced_log_probabilities"],
        generated["greedy_log_probabilities"],
        generated["rank_B_log_probabilities"],
    ):
        assert greedy >= realized >= upper


def test_direct_generation_leaves_fixed_rank_ceiling_unavailable():
    model = TinyRankModel()
    generated = generate_rank_span(model, [0], [2, 3])
    assert generated["realized_ranks"] == [2, 3]
    assert generated["quality_rank_ceiling"] is None
    assert generated["rank_B_token_ids"] is None
    assert generated["rank_B_log_probabilities"] is None
    assert len(generated["greedy_log_probabilities"]) == 2


def test_literal_payload_tokenization_disables_specials_and_keeps_first_token():
    model = NoBosRankModel()
    representation = direct_representation(model, "ab")
    assert representation.metadata["payload_token_ids"] == [2, 3]
    assert model.tokenize_calls[0] == (False, False)
    assert representation.metadata["detokenized_prefix_byte_length"] == 0
    assert representation.metadata["original_payload_sha256"] == hashlib.sha256(
        b"ab"
    ).hexdigest()


@pytest.mark.parametrize("fused_payload_symbols", [1, 2])
def test_raw_payload_path_avoids_leading_space_fusion(fused_payload_symbols):
    model = LeadingSpaceFusionTrapModel(fused_payload_symbols)
    assert model.tokenize(b" ab", add_bos=False, special=False) != [2, 3]
    representation = direct_representation(model, "ab")
    assert representation.metadata["payload_token_ids"] == [2, 3]
    assert representation.metadata["detokenized_prefix_byte_length"] == 0


def test_prompt_context_retains_first_real_token_when_bos_is_not_inserted():
    no_bos = NoBosRankModel()
    assert make_context_token_ids(no_bos, "ab") == [2, 3]
    with_bos = TinyRankModel()
    assert make_context_token_ids(with_bos, "ab") == [2, 3]


def test_nonempty_prompt_with_no_non_bos_tokens_fails_closed():
    with pytest.raises(ValueError, match="non-empty prompt"):
        make_context_token_ids(BosOnlyPromptModel(), "ab")


@pytest.mark.parametrize("prefix", [b" ", b"  "])
def test_direct_reversible_space_prefix_is_explicit_and_removed_on_decode(prefix):
    model = ImplicitSpaceRankModel(prefix)
    representation = direct_representation(model, "ab")
    assert representation.metadata["detokenized_prefix_bytes_base64"] == base64.b64encode(
        prefix
    ).decode("ascii")
    assert representation.metadata["detokenized_prefix_byte_length"] == len(prefix)
    decoded = decode_representation(model, representation, representation.ranks)
    assert decoded["exact_representation_recovery"] is True
    assert decoded["exact_payload_recovery"] is True
    assert decoded["exact_recovery"] is decoded["exact_payload_recovery"]
    assert decoded["recovered_bytes"] == b"ab"
    assert decoded["recovered_serialized_bytes"] == prefix + b"ab"
    assert decoded["recovery_outcome_semantics"] == (
        "original_serialized_payload_bytes_sha256_v1"
    )


@pytest.mark.parametrize(
    "model,error",
    [
        (NonReversibleRankModel(), "does not preserve"),
        (InternalMutationRankModel(), "does not preserve"),
        (SuffixMutationRankModel(), "does not preserve"),
        (NonSpacePrefixRankModel(), "non-space"),
    ],
)
def test_direct_nonprefix_tokenizer_transformations_fail_closed(model, error):
    with pytest.raises(ValueError, match=error):
        direct_representation(model, "ab")


def test_direct_inverse_rank_transcoding_recovers_payload_tokens():
    model = TinyRankModel()
    representation = direct_representation(model, "ab")
    generated = generate_rank_span(model, [0], representation.ranks)
    recovered = recover_rank_span(model, [0], [], generated["forced_token_ids"])
    decoded = decode_representation(model, representation, recovered["ranks"])
    assert decoded["exact_recovery"] is True
    assert decoded["exact_recovery"] is decoded["exact_payload_recovery"]
    assert decoded["exact_representation_recovery"] is True
    assert decoded["recovered_bytes"] == b"ab"
    assert decoded["original_payload_sha256"] == hashlib.sha256(b"ab").hexdigest()
    assert decoded["recovered_payload_sha256"] == decoded["original_payload_sha256"]
    assert decoded["recovered_token_ids"] == representation.metadata["payload_token_ids"]


def test_serial_leadin_replay_and_retokenization_are_separate_modes():
    model = TinyRankModel()
    generated = generate_rank_span(model, [0], [2, 3], leadin_token_count=2)
    recovered = recover_rank_span(
        model,
        [0],
        generated["leadin_token_ids"],
        generated["forced_token_ids"],
    )
    text_replay = retokenize_message(model, generated)
    assert recovered["ranks"] == [2, 3]
    assert "full_token_ids_match" in text_replay
    assert text_replay["boundary_rule"].startswith("saved token offsets")


def test_first_divergence_and_failure_record_include_required_fields():
    assert first_divergence([1, 2], [1, 3])["position_zero_based"] == 1
    assert first_divergence([1], [1, 2])["position_zero_based"] == 1
    assert first_divergence([1], [1])["diverged"] is False
    record = diagnose_rank_failure(
        [1, 2], [1, 3], [4, 5], [4, 6], [7, 8], (2, 4), "retokenization"
    )
    assert record["expected_token_id"] == 5
    assert record["recovered_token_id"] == 6
    assert record["expected_rank"] == 2
    assert record["recovered_rank"] == 3
    assert len(record["context_sha256"]) == 64
    assert record["boundary_start"] == 2


def test_transmission_transforms_are_deterministic_and_named():
    text = "  A  line\nwith “quotes”.  "
    assert apply_transmission_transform(text, "whitespace_collapse") == "A line with “quotes”."
    assert '"quotes"' in apply_transmission_transform(text, "smart_quote_conversion")
    assert apply_transmission_transform(text, "character_insert", seed=9) == apply_transmission_transform(
        text, "character_insert", seed=9
    )


def test_character_edits_respect_frozen_eligible_position_rules():
    assert apply_transmission_transform("  \t\n", "character_deletion", seed=3) == "  \t\n"
    assert apply_transmission_transform(" \tA \n", "character_deletion", seed=3) == " \t \n"
    assert apply_transmission_transform("--!--", "character_substitution", seed=3) == "--!--"
    substituted = apply_transmission_transform("--A--", "character_substitution", seed=3)
    assert substituted[:2] == "--" and substituted[-2:] == "--"
    assert substituted[2] != "A"


def test_dynamic_tail_requires_completeness_not_only_length():
    assert dynamic_tail_complete("This is a complete thought.", 8) is True
    assert dynamic_tail_complete("This remains open and", 20) is False
    assert dynamic_tail_complete("This has (an open delimiter.", 20) is False
    assert dynamic_tail_complete("Too short.", 7) is False


def test_config_named_transforms_and_token_space_operations():
    assert apply_transmission_transform('He said "yes".', "quote_conversion") == "He said \u201cyes\u201d."
    assert apply_transmission_transform("a\nb", "markdown_copy_paste") == "> a\n> b"
    deleted = transform_token_ids([1, 2, 3, 4, 5], "token_deletion", seed=7)
    assert deleted[0] == 1 and deleted[-1] == 5 and len(deleted) == 4
    assert transform_token_ids([1, 2], "token_deletion", seed=7) == [1, 2]
    assert transform_token_ids([1, 2, 3], "truncation") == [1, 2]

