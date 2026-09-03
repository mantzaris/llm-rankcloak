import sys
from pathlib import Path

import pandas as pd
import pytest

from rankcloak.revision_protocol import Representation
from rankcloak.revision_v3_q4_recovery import q4_visible_recovery_outcome


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_revision_v3_generation import (  # noqa: E402
    paired_quantization_recovery_summary,
    paired_quantization_recovery_table,
)


def q4_record() -> dict:
    return {
        "record_type": "quantization_q4_model_backed_replay",
        "population": "rankcloak",
        "new_generation_performed": False,
        "rank_replay_exact": True,
        "plan_row": {
            "quantization": "Q4_K_M",
            "population": "rankcloak",
        },
        "expected_ranks": [2, 3],
        "historical_output_token_ids": [10, 11],
        "historical_output_text": "visible",
        "context_token_ids": [1],
        "distribution_trace": {
            "observed_ranks": [2, 3],
            "context_sha256": "context",
        },
    }


def representation() -> Representation:
    return Representation(
        name="ascii_b8",
        ranks=(2, 3),
        metadata={},
        payload_bytes=b"x",
        payload_text="x",
    )


def test_identical_q4_retokenization_reuses_validated_trace_without_model_replay(
    monkeypatch,
):
    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.retokenize_message",
        lambda model, generated: {
            "retokenized_token_ids": [10, 11],
            "forced_token_ids": [10, 11],
            "full_token_ids_match": True,
            "divergence": {"diverged": False},
            "boundary_rule": "saved token offsets applied after full-text retokenization",
        },
    )

    def forbidden_replay(*args, **kwargs):
        raise AssertionError("model rank replay must not run for identical token IDs")

    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.recover_rank_span",
        forbidden_replay,
    )
    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.decode_representation",
        lambda model, value, ranks: {"exact_payload_recovery": ranks == [2, 3]},
    )
    outcome = q4_visible_recovery_outcome(object(), q4_record(), representation())
    assert outcome["exact_payload_recovery"] is True
    assert outcome["model_rank_replay_performed"] is False
    assert outcome["replay"]["execution_mode"].startswith("validated_saved_id")


def test_divergent_q4_retokenization_replays_observed_visible_tokens(monkeypatch):
    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.retokenize_message",
        lambda model, generated: {
            "retokenized_token_ids": [12, 11],
            "forced_token_ids": [12, 11],
            "full_token_ids_match": False,
            "divergence": {"diverged": True, "position_zero_based": 0},
            "boundary_rule": "saved token offsets applied after full-text retokenization",
        },
    )
    observed = {}

    def replay(model, context, leadin, forced, allowed_token_mask):
        observed["arguments"] = (context, leadin, forced, allowed_token_mask)
        return {
            "ranks": [4, 3],
            "token_log_probabilities": [-1.0, -2.0],
            "context_sha256": "context",
        }

    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.recover_rank_span", replay
    )
    monkeypatch.setattr(
        "rankcloak.revision_v3_q4_recovery.decode_representation",
        lambda model, value, ranks: {"exact_payload_recovery": False},
    )
    outcome = q4_visible_recovery_outcome(object(), q4_record(), representation())
    assert observed["arguments"] == ([1], [], [12, 11], None)
    assert outcome["exact_payload_recovery"] is False
    assert outcome["model_rank_replay_performed"] is True
    assert outcome["exact_rank_recovery"] is False


def paired_trial_rows() -> pd.DataFrame:
    rows = []
    outcomes = [(True, True), (True, False), (False, True), (False, False)]
    for index, (q4_visible, q8_visible) in enumerate(outcomes):
        for quantization, visible in (
            ("Q4_K_M", q4_visible),
            ("Q8_0", q8_visible),
        ):
            rows.append(
                {
                    "pairing_unit_id": f"pair-{index}",
                    "quantization": quantization,
                    "population": "rankcloak",
                    "payload_name": f"payload-{index}",
                    "payload_class": "class",
                    "payload_split": "test",
                    "representation_name": "ascii_b8",
                    "codec_id": "nonseg_ascii_b8",
                    "prompt_template_id": "template",
                    "target_token_count": 8,
                    "saved_id_exact_payload_recovery": True,
                    "visible_text_exact_payload_recovery": visible,
                    "visible_text_full_token_ids_match": visible,
                    "visible_text_model_rank_replay_performed": not visible,
                    "visible_text_retokenized_token_count": 8,
                }
            )
    return pd.DataFrame(rows)


def test_paired_q4_q8_recovery_preserves_all_discordant_outcomes():
    pairs = paired_quantization_recovery_table(paired_trial_rows())
    summary = paired_quantization_recovery_summary(pairs)
    overall = summary.loc[
        summary["analysis_id"].eq("quantization_recovery_overall")
    ].iloc[0]
    assert len(pairs) == 4
    assert overall["both_visible_recover_count"] == 1
    assert overall["q4_only_visible_recover_count"] == 1
    assert overall["q8_only_visible_recover_count"] == 1
    assert overall["neither_visible_recovers_count"] == 1
    assert overall["paired_rate_difference_q8_minus_q4"] == pytest.approx(0.0)
    assert overall["q4_saved_id_successes"] == 4
    assert overall["q8_saved_id_successes"] == 4


def test_paired_q4_q8_recovery_rejects_incomplete_matrix():
    rows = paired_trial_rows()
    rows = rows.loc[
        ~(
            rows["pairing_unit_id"].eq("pair-0")
            & rows["quantization"].eq("Q8_0")
        )
    ]
    with pytest.raises(ValueError, match="incomplete"):
        paired_quantization_recovery_table(rows)


def test_entropy_capacity_source_label_is_conditional_on_completion():
    source = (
        Path(__file__).resolve().parents[1]
        / "results/revision_v3/source_tables/entropy_generation_summary.csv"
    )
    summary = pd.read_csv(source)
    metrics = set(summary["metric"].astype(str))
    assert (
        "fixed_payload_bits_per_generated_token_conditional_on_completion"
        in metrics
    )
    assert "fixed_payload_bits_per_generated_token" not in metrics
    strict = summary.loc[
        summary["analysis_id"].eq("entropy_overall")
        & summary["population"].eq("rankcloak")
        & summary["gate_level"].eq("strict")
        & summary["metric"].eq(
            "fixed_payload_bits_per_generated_token_conditional_on_completion"
        )
    ].iloc[0]
    assert int(strict["observation_count"]) == 114
