import numpy as np
import pytest

from rankcloak.revision_protocol import generate_rank_span
from rankcloak.revision_v3_entropy import (
    EntropyGateError,
    entropy_eligible,
    generate_entropy_gated_span,
    generate_ordinary_entropy_trace,
    recover_entropy_gated_span,
    retokenize_entropy_gated_message,
    shannon_entropy_bits,
)


class ScheduledEntropyModel:
    pieces = ["", "a", "b", "c"]

    def __init__(self, schedule):
        self.schedule = [np.asarray(row, dtype=float) for row in schedule]
        self.n_tokens = 0
        self.scores = np.zeros((256, len(self.pieces)), dtype=float)

    def token_bos(self):
        return 0

    def n_vocab(self):
        return len(self.pieces)

    def reset(self):
        self.n_tokens = 0

    def eval(self, token_ids):
        for _ in token_ids:
            row = self.schedule[min(self.n_tokens, len(self.schedule) - 1)]
            self.scores[self.n_tokens] = row
            self.n_tokens += 1

    def tokenize(self, value, add_bos=True):
        text = value.decode() if isinstance(value, bytes) else str(value)
        result = [self.pieces.index(character) for character in text]
        return ([0] if add_bos else []) + result

    def detokenize(self, token_ids):
        return "".join(self.pieces[int(value)] for value in token_ids).encode()


LOW = [0.0, -20.0, -20.0, -20.0]
HIGH = [0.0, 0.0, 0.0, 0.0]


def test_entropy_calculation_and_inclusive_threshold_boundary():
    assert shannon_entropy_bits(HIGH) == pytest.approx(2.0)
    assert shannon_entropy_bits(LOW) < 0.001
    assert entropy_eligible(2.0, 2.0) is True
    assert entropy_eligible(1.999, 2.0) is False


def test_disabled_gate_reproduces_ordinary_rankcloak_tokens():
    ordinary = generate_rank_span(ScheduledEntropyModel([HIGH]), [0], [2, 3, 1])
    gated = generate_entropy_gated_span(
        ScheduledEntropyModel([HIGH]),
        [0],
        [2, 3, 1],
        entropy_threshold_bits=None,
    )
    assert gated["full_token_ids"] == ordinary["full_token_ids"]
    assert gated["realized_ranks"] == ordinary["realized_ranks"]
    assert gated["payload_completion"] is True


def test_skipped_positions_do_not_consume_or_shift_payload_symbols():
    schedule = [LOW, HIGH, LOW, HIGH]
    generated = generate_entropy_gated_span(
        ScheduledEntropyModel(schedule),
        [0],
        [2, 3],
        entropy_threshold_bits=1.1,
        maximum_generated_tokens=4,
        sampling_seed=19,
    )
    assert generated["embedding_eligible_mask"] == [False, True, False, True]
    assert generated["consumed_payload_rank_indices"] == [0, 1]
    assert generated["realized_ranks"] == [None, 2, None, 3]
    assert generated["embedding_token_roles"] == [
        "ordinary_sampled_skip",
        "payload",
        "ordinary_sampled_skip",
        "payload",
    ]
    assert generated["ordinary_sampled_skip_positions"] == [0, 2]
    assert all(token_id != 0 for token_id in generated["ordinary_sampled_skip_token_ids"])
    recovered = recover_entropy_gated_span(
        ScheduledEntropyModel(schedule),
        [0],
        [],
        generated["embedding_token_ids"],
        entropy_threshold_bits=1.1,
        expected_payload_rank_count=2,
    )
    assert recovered["embedding_eligible_mask"] == generated["embedding_eligible_mask"]
    assert recovered["ranks"] == [2, 3]
    assert recovered["payload_completion"] is True


def test_insufficient_budget_records_capacity_failure_and_correct_fraction():
    generated = generate_entropy_gated_span(
        ScheduledEntropyModel([LOW, HIGH]),
        [0],
        [2, 3],
        entropy_threshold_bits=1.0,
        maximum_generated_tokens=2,
        sampling_seed=23,
    )
    assert generated["payload_completion"] is False
    assert generated["consumed_payload_rank_count"] == 1
    assert generated["payload_fraction_embedded"] == 0.5
    assert generated["capacity_failure"] is not None


def test_empty_payload_and_fixed_seed_determinism():
    first = generate_entropy_gated_span(
        ScheduledEntropyModel([HIGH]),
        [0],
        [],
        entropy_threshold_bits=1.0,
        maximum_generated_tokens=5,
        sampling_seed=29,
    )
    second = generate_entropy_gated_span(
        ScheduledEntropyModel([HIGH]),
        [0],
        [],
        entropy_threshold_bits=1.0,
        maximum_generated_tokens=5,
        sampling_seed=29,
    )
    assert first["payload_completion"] is True
    assert first["embedding_token_ids"] == []
    assert first["full_token_ids"] == second["full_token_ids"]


def test_visible_text_retokenization_is_a_separate_diagnostic():
    generated = generate_entropy_gated_span(
        ScheduledEntropyModel([HIGH]),
        [0],
        [2],
        entropy_threshold_bits=1.0,
        maximum_generated_tokens=1,
        sampling_seed=31,
    )
    diagnostic = retokenize_entropy_gated_message(
        ScheduledEntropyModel([HIGH]), generated
    )
    assert diagnostic["evaluation_scope"].startswith("visible_text")
    assert "embedding_token_ids" in diagnostic


def test_enabled_gate_requires_a_finite_budget():
    with pytest.raises(EntropyGateError, match="maximum_generated_tokens"):
        generate_entropy_gated_span(
            ScheduledEntropyModel([HIGH]),
            [0],
            [2],
            entropy_threshold_bits=1.0,
            sampling_seed=37,
        )


def test_enabled_gate_requires_a_sampling_seed():
    with pytest.raises(EntropyGateError, match="sampling_seed"):
        generate_entropy_gated_span(
            ScheduledEntropyModel([HIGH]),
            [0],
            [2],
            entropy_threshold_bits=1.0,
            maximum_generated_tokens=1,
        )


def test_ordinary_sampled_skips_are_seeded_and_respect_allowed_mask():
    mask = np.asarray([False, True, True, False])
    first = generate_entropy_gated_span(
        ScheduledEntropyModel([LOW]),
        [0],
        [2, 2, 2, 2],
        entropy_threshold_bits=1.1,
        maximum_generated_tokens=4,
        allowed_token_mask=mask,
        sampling_seed=41,
    )
    second = generate_entropy_gated_span(
        ScheduledEntropyModel([LOW]),
        [0],
        [2, 2, 2, 2],
        entropy_threshold_bits=1.1,
        maximum_generated_tokens=4,
        allowed_token_mask=mask,
        sampling_seed=41,
    )
    assert first["embedding_token_ids"] == second["embedding_token_ids"]
    assert set(first["embedding_token_ids"]).issubset({1, 2})
    assert set(first["embedding_token_roles"]) == {"ordinary_sampled_skip"}
    assert first["consumed_payload_rank_count"] == 0


def test_calibration_trace_uses_same_seeded_top_p_sampler():
    first = generate_ordinary_entropy_trace(
        ScheduledEntropyModel([HIGH]),
        [0],
        6,
        sampling_seed=47,
    )
    second = generate_ordinary_entropy_trace(
        ScheduledEntropyModel([HIGH]),
        [0],
        6,
        sampling_seed=47,
    )
    assert first["token_ids"] == second["token_ids"]
    assert first["sampler"] == "numpy_pcg64_serial_top_p_v1_token_id_tiebreak"
    assert first["temperature"] == 0.8
    assert first["top_p"] == 0.95
    assert first["next_token_entropies_bits"] == pytest.approx([2.0] * 6)
