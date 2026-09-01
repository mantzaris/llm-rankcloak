import random

import numpy as np

from rankcloak.revision_v3_analysis import levenshtein_distance
from rankcloak.revision_v3_entropy import (
    generate_entropy_gated_span,
    recover_entropy_gated_span,
    shannon_entropy_bits,
)


def _dynamic_programming_distance(left, right):
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def test_bit_vector_levenshtein_matches_dynamic_programming_random_cases():
    generator = random.Random(20260831)
    alphabet = "abcd "
    for _ in range(250):
        left = "".join(generator.choice(alphabet) for _ in range(generator.randrange(25)))
        right = "".join(generator.choice(alphabet) for _ in range(generator.randrange(25)))
        assert levenshtein_distance(left, right) == _dynamic_programming_distance(left, right)


class LongPayloadModel:
    pieces = ["", "a", "b", "c", "d"]

    def __init__(self):
        self.n_tokens = 0
        self.scores = np.zeros((1024, len(self.pieces)), dtype=float)

    def token_bos(self):
        return 0

    def n_vocab(self):
        return len(self.pieces)

    def reset(self):
        self.n_tokens = 0

    def eval(self, token_ids):
        for _ in token_ids:
            self.scores[self.n_tokens] = np.zeros(len(self.pieces), dtype=float)
            self.n_tokens += 1

    def tokenize(self, value, add_bos=True):
        text = value.decode() if isinstance(value, bytes) else str(value)
        ids = [self.pieces.index(character) for character in text]
        return ([0] if add_bos else []) + ids

    def detokenize(self, token_ids):
        return "".join(self.pieces[int(value)] for value in token_ids).encode()


def test_maximum_length_payload_capacity_and_replay_accounting():
    ranks = [1 + (index % 5) for index in range(256)]
    generated = generate_entropy_gated_span(
        LongPayloadModel(),
        [0],
        ranks,
        entropy_threshold_bits=2.0,
        maximum_generated_tokens=256,
        quality_rank_ceiling=5,
    )
    assert generated["payload_completion"] is True
    assert generated["eligible_position_count"] == 256
    assert generated["ineligible_position_count"] == 0
    assert generated["consumed_payload_rank_count"] == 256
    recovered = recover_entropy_gated_span(
        LongPayloadModel(),
        [0],
        [],
        generated["embedding_token_ids"],
        entropy_threshold_bits=2.0,
        expected_payload_rank_count=256,
    )
    assert recovered["ranks"] == ranks


def test_entropy_uses_only_admissible_tokens_when_masked():
    logits = np.zeros(5, dtype=float)
    mask = np.asarray([True, True, False, False, False])
    assert shannon_entropy_bits(logits, mask) == 1.0
