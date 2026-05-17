import numpy as np

from rankcloak.rank_codec import (
    rank_of_token,
    sorted_token_ids_from_logits,
    test_stable_rank_ordering as stable_rank_ordering_fixture,
    token_id_at_rank,
)


def test_stable_rank_ordering_fixture_passes():
    result = stable_rank_ordering_fixture()
    assert result["passed"] is True
    assert result["sorted_ids"] == [1, 2, 0, 4, 3]


def test_rank_ordering_is_descending_with_token_id_tie_breaks():
    logits = np.array([0.5, 3.0, 3.0, -1.0, 0.5])
    assert sorted_token_ids_from_logits(logits).tolist() == [1, 2, 0, 4, 3]
    assert rank_of_token(logits, 1) == 1
    assert rank_of_token(logits, 2) == 2
    assert rank_of_token(logits, 0) == 3
    assert rank_of_token(logits, 4) == 4
    assert rank_of_token(logits, 3) == 5


def test_token_id_at_rank_is_one_indexed():
    logits = np.array([0.0, 10.0, 5.0])
    assert token_id_at_rank(logits, 1) == 1
    assert token_id_at_rank(logits, 2) == 2
    assert token_id_at_rank(logits, 3) == 0
