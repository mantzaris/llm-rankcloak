r"""Capacity--quality theory and exact-replay validation for RankCloak.

The functions in this module are deliberately model-free.  They calculate
information-theoretic quantities from explicit inputs and validate saved
token-level evidence; they never regenerate text, infer unavailable
probabilities, or substitute synthetic observations for missing results.

Log probabilities are natural logarithms, so :math:`Q_B` and
:math:`\Delta_B` are reported in nats per forced token.  Rates are reported in
bits per token.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


THEORY_SCHEMA_VERSION = "1.0"
TOKEN_ID_TIE_BREAK_RULE = "descending_score_then_ascending_token_id"
EXACT_RECOVERY_PROPOSITION = (
    "If encoder and decoder begin with identical token-ID contexts, use the "
    "same deterministic model, tokenizer, inference configuration, admissible "
    "token set, and descending-score/ascending-token-ID order, and receive "
    "identical forced token IDs, inverse ranking recovers every encoded rank. "
    "An injective payload codec then recovers the payload exactly."
)

CAPACITY_VALIDATION_COLUMNS = (
    "source_file",
    "source_row",
    "trial_id",
    "model_id",
    "protocol_variant",
    "payload_name",
    "representation_name",
    "capacity_status",
    "H_bits",
    "H_source",
    "artifact_bit_length",
    "alphabet_size_B",
    "log2_B",
    "theoretical_n_B",
    "observed_n_forced",
    "observed_n_tail",
    "R_B_bits_per_forced_token",
    "observed_rate_bits_per_forced_token",
    "R_effective_bits_per_forced_plus_tail_token",
    "rate_upper_bound_bits_per_forced_token",
    "rate_bound_holds",
    "observed_forced_count_feasible",
    "code_space_slack_bits",
    "literal_padding_bits",
    "code_space_utilization",
    "unused_codeword_fraction",
    "finite_padding_case",
)

CAPACITY_PLOT_COLUMNS = (
    "trial_id",
    "model_id",
    "protocol_variant",
    "payload_name",
    "representation_name",
    "H_bits",
    "alphabet_size_B",
    "theoretical_n_B",
    "observed_n_forced",
    "observed_n_tail",
    "R_B_bits_per_forced_token",
    "observed_rate_bits_per_forced_token",
    "R_effective_bits_per_forced_plus_tail_token",
    "rate_upper_bound_bits_per_forced_token",
    "code_space_slack_bits",
    "literal_padding_bits",
)

QUALITY_VALIDATION_COLUMNS = (
    "source_file",
    "source_row",
    "trial_id",
    "model_id",
    "protocol_variant",
    "payload_name",
    "representation_name",
    "alphabet_size_B",
    "quality_status",
    "quality_evidence_level",
    "forced_context_count",
    "Q_B_nats_per_forced_token",
    "Q_greedy_nats_per_forced_token",
    "Q_rank_B_nats_per_forced_token",
    "Delta_B_nats_per_forced_token",
    "greedy_lower_bound_available",
    "rank_B_upper_bound_available",
    "greedy_lower_bound_holds_per_context",
    "rank_B_upper_bound_holds_per_context",
    "greedy_lower_bound_holds_in_expectation",
    "rank_B_upper_bound_holds_in_expectation",
    "rank_range_holds",
    "all_available_checks_hold",
    "minimum_lower_margin_nats",
    "minimum_upper_margin_nats",
)

QUALITY_PLOT_COLUMNS = (
    "trial_id",
    "model_id",
    "protocol_variant",
    "payload_name",
    "representation_name",
    "alphabet_size_B",
    "forced_context_count",
    "Q_B_nats_per_forced_token",
    "Q_greedy_nats_per_forced_token",
    "Q_rank_B_nats_per_forced_token",
    "Delta_B_nats_per_forced_token",
    "quality_status",
)

EXACT_RECOVERY_COLUMNS = (
    "source_file",
    "source_row",
    "trial_id",
    "model_id",
    "protocol_variant",
    "exact_recovery_status",
    "trace_step_count",
    "configurations_identical",
    "supported_tie_break_identical",
    "context_token_ids_identical",
    "forced_token_ids_identical",
    "ranked_token_orders_identical",
    "encoder_rank_selection_valid",
    "guarantee_conditions_satisfied",
    "recovered_ranks_equal_expected",
    "observed_exact_recovery",
    "proposition_confirmed",
    "proposition_violation",
    "reported_exact_recovery",
    "missing_evidence",
)

CASCADE_DIAGNOSTIC_COLUMNS = (
    "source_file",
    "source_row",
    "trial_id",
    "model_id",
    "protocol_variant",
    "cascade_status",
    "reference_step_count",
    "edited_step_count",
    "first_context_divergence_step",
    "first_context_token_difference",
    "first_rank_order_divergence_step",
    "first_rank_error_step",
    "context_divergence_count",
    "rank_order_divergence_count",
    "rank_error_count",
    "post_initial_context_divergence_count",
    "post_initial_rank_error_count",
    "cascade_observed",
)

OUTPUT_FILENAMES = {
    "capacity_validation": "theory_capacity_validation.csv",
    "capacity_plot": "theory_capacity_plot_source.csv",
    "quality_validation": "theory_quality_validation.csv",
    "quality_plot": "theory_quality_plot_source.csv",
    "exact_recovery": "theory_exact_recovery_validation.csv",
    "cascade": "theory_cascade_diagnostics.csv",
    "manifest": "theory_validation_manifest.json",
}


class TheoryValidationError(ValueError):
    """Raised when explicit theory inputs are malformed or contradictory."""


@dataclass(frozen=True)
class CapacityMetrics:
    """Finite-payload capacity quantities for one bounded alphabet."""

    H_bits: float
    alphabet_size_B: int
    log2_B: float
    theoretical_n_B: int
    observed_n_forced: Optional[int]
    observed_n_tail: Optional[int]
    R_B_bits_per_forced_token: float
    observed_rate_bits_per_forced_token: Optional[float]
    R_effective_bits_per_forced_plus_tail_token: Optional[float]
    rate_upper_bound_bits_per_forced_token: float
    rate_bound_holds: bool
    observed_forced_count_feasible: Optional[bool]
    code_space_slack_bits: float
    literal_padding_bits: Optional[int]
    code_space_utilization: float
    unused_codeword_fraction: float
    finite_padding_case: str


@dataclass(frozen=True)
class QualityMetrics:
    """Empirical surprisal and same-context rank-bound diagnostics."""

    forced_context_count: int
    Q_B_nats_per_forced_token: float
    Q_greedy_nats_per_forced_token: Optional[float]
    Q_rank_B_nats_per_forced_token: Optional[float]
    Delta_B_nats_per_forced_token: Optional[float]
    greedy_lower_bound_available: bool
    rank_B_upper_bound_available: bool
    greedy_lower_bound_holds_per_context: Optional[bool]
    rank_B_upper_bound_holds_per_context: Optional[bool]
    greedy_lower_bound_holds_in_expectation: Optional[bool]
    rank_B_upper_bound_holds_in_expectation: Optional[bool]
    rank_range_holds: Optional[bool]
    all_available_checks_hold: Optional[bool]
    minimum_lower_margin_nats: Optional[float]
    minimum_upper_margin_nats: Optional[float]
    quality_status: str


@dataclass(frozen=True)
class ExactReplayConfiguration:
    """Identity-bearing state required by the exact-replay proposition."""

    model_identity: str
    tokenizer_identity: str
    inference_config_identity: str
    prompt_token_ids: Tuple[int, ...]
    admissible_token_set_identity: str
    tie_break_rule: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExactReplayConfiguration":
        required = (
            "model_identity",
            "tokenizer_identity",
            "inference_config_identity",
            "prompt_token_ids",
            "admissible_token_set_identity",
            "tie_break_rule",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise TheoryValidationError(
                "Replay configuration missing fields: " + ", ".join(missing)
            )
        prompt = _integer_sequence(value["prompt_token_ids"], "prompt_token_ids")
        text_values = {}
        for key in required:
            if key == "prompt_token_ids":
                continue
            item = value[key]
            if item is None or not str(item).strip():
                raise TheoryValidationError(
                    "Replay configuration {} must be non-empty".format(key)
                )
            text_values[key] = str(item)
        return cls(prompt_token_ids=tuple(prompt), **text_values)


@dataclass(frozen=True)
class TheoryArtifacts:
    """Paths and validation counts emitted by :func:`build_theory_artifacts`."""

    output_dir: str
    files: Mapping[str, str]
    summary: Mapping[str, Any]


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TheoryValidationError("{} must be numeric, not boolean".format(label))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TheoryValidationError("{} must be numeric".format(label)) from exc
    if not math.isfinite(number) or number < 0.0:
        raise TheoryValidationError("{} must be finite and non-negative".format(label))
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    number = _finite_nonnegative(value, label)
    rounded = round(number)
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise TheoryValidationError("{} must be an integer".format(label))
    return int(rounded)


def _alphabet_size(value: Any) -> int:
    alphabet = _nonnegative_integer(value, "B")
    if alphabet < 2:
        raise TheoryValidationError("B must be an integer of at least 2")
    return alphabet


def _near_integer(value: float, tolerance: float = 1e-12) -> Optional[int]:
    closest = int(round(value))
    return closest if math.isclose(value, closest, rel_tol=tolerance, abs_tol=tolerance) else None


def n_B(H: Any, B: Any) -> int:
    """Return ``ceil(H / log2(B))``, the minimum bounded-rank length.

    ``H=0`` has the finite empty-codeword value ``n_B=0``.  The small
    near-integer correction prevents floating-point roundoff from adding a
    spurious symbol when the quotient is mathematically integral.
    """

    entropy = _finite_nonnegative(H, "H")
    alphabet = _alphabet_size(B)
    if entropy == 0.0:
        return 0
    quotient = entropy / math.log2(alphabet)
    exact_integer = _near_integer(quotient)
    return exact_integer if exact_integer is not None else int(math.ceil(quotient))


def R_B(H: Any, B: Any) -> float:
    """Return ``H / n_B`` in bits per forced token.

    The empty-payload convention is zero bits per token rather than the
    undefined expression ``0/0``.
    """

    entropy = _finite_nonnegative(H, "H")
    count = n_B(entropy, B)
    return 0.0 if count == 0 else entropy / count


def R_effective(H: Any, n_forced: Any, n_tail: Any) -> float:
    """Return ``H / (n_forced + n_tail)`` in bits per emitted token."""

    entropy = _finite_nonnegative(H, "H")
    forced = _nonnegative_integer(n_forced, "n_forced")
    tail = _nonnegative_integer(n_tail, "n_tail")
    denominator = forced + tail
    if denominator == 0:
        if entropy == 0.0:
            return 0.0
        raise TheoryValidationError("A positive H requires at least one emitted token")
    return entropy / denominator


def capacity_metrics(
    H: Any,
    B: Any,
    *,
    observed_n_forced: Optional[Any] = None,
    observed_n_tail: Optional[Any] = None,
) -> CapacityMetrics:
    """Calculate ideal and observed finite-payload rate quantities.

    ``literal_padding_bits`` is reported only for power-of-two alphabets.  For
    other alphabets the analogous quantity is code-space slack in bits; it is
    not literal binary padding.
    """

    entropy = _finite_nonnegative(H, "H")
    alphabet = _alphabet_size(B)
    log_bound = math.log2(alphabet)
    ideal_count = n_B(entropy, alphabet)
    nominal_rate = R_B(entropy, alphabet)
    forced = (
        None
        if observed_n_forced is None
        else _nonnegative_integer(observed_n_forced, "observed_n_forced")
    )
    tail = (
        None
        if observed_n_tail is None
        else _nonnegative_integer(observed_n_tail, "observed_n_tail")
    )
    if entropy > 0.0 and forced == 0:
        raise TheoryValidationError("Positive H cannot have zero observed forced tokens")
    observed_rate = None if forced is None else (0.0 if forced == 0 else entropy / forced)
    effective = (
        None
        if forced is None or tail is None
        else R_effective(entropy, forced, tail)
    )
    slack = ideal_count * log_bound - entropy
    if abs(slack) <= 1e-12:
        slack = 0.0
    utilization = math.pow(2.0, -slack)
    utilization = min(1.0, max(0.0, utilization))
    bound_holds = nominal_rate <= log_bound + 1e-12
    feasible = None if forced is None else forced >= ideal_count
    bits_per_symbol = _near_integer(log_bound)
    integral_entropy = _near_integer(entropy)
    literal_padding = (
        None
        if bits_per_symbol is None or integral_entropy is None
        else int(ideal_count * bits_per_symbol - integral_entropy)
    )
    if entropy == 0.0:
        padding_case = "empty_payload"
    elif slack == 0.0:
        padding_case = "exact_symbol_fit"
    elif bits_per_symbol is not None:
        padding_case = "finite_binary_padding"
    else:
        padding_case = "non_power_of_two_code_space_slack"
    return CapacityMetrics(
        H_bits=entropy,
        alphabet_size_B=alphabet,
        log2_B=log_bound,
        theoretical_n_B=ideal_count,
        observed_n_forced=forced,
        observed_n_tail=tail,
        R_B_bits_per_forced_token=nominal_rate,
        observed_rate_bits_per_forced_token=observed_rate,
        R_effective_bits_per_forced_plus_tail_token=effective,
        rate_upper_bound_bits_per_forced_token=log_bound,
        rate_bound_holds=bound_holds,
        observed_forced_count_feasible=feasible,
        code_space_slack_bits=slack,
        literal_padding_bits=literal_padding,
        code_space_utilization=utilization,
        unused_codeword_fraction=1.0 - utilization,
        finite_padding_case=padding_case,
    )


def _log_probability_sequence(values: Sequence[Any], label: str) -> List[float]:
    result: List[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TheoryValidationError(
                "{}[{}] must be numeric".format(label, index)
            ) from exc
        if not math.isfinite(number) or number > 1e-12:
            raise TheoryValidationError(
                "{}[{}] must be a finite natural-log probability <= 0".format(
                    label, index
                )
            )
        result.append(number)
    if not result:
        raise TheoryValidationError("{} must contain at least one observation".format(label))
    return result


def Q_B(realized_log_probabilities: Sequence[Any]) -> float:
    """Return empirical ``E[-log p_h(R)]`` in nats per forced token."""

    values = _log_probability_sequence(realized_log_probabilities, "realized_log_probabilities")
    return -sum(values) / len(values)


def Delta_B(
    realized_log_probabilities: Sequence[Any],
    greedy_log_probabilities: Sequence[Any],
) -> float:
    """Return ``E[-log p_h(R) + log p_h(1)]`` in nats/token."""

    realized = _log_probability_sequence(realized_log_probabilities, "realized_log_probabilities")
    greedy = _log_probability_sequence(greedy_log_probabilities, "greedy_log_probabilities")
    if len(realized) != len(greedy):
        raise TheoryValidationError("Realized and greedy evidence lengths differ")
    return sum(greedy_value - realized_value for realized_value, greedy_value in zip(realized, greedy)) / len(realized)


def validate_quality_bounds(
    realized_log_probabilities: Sequence[Any],
    *,
    greedy_log_probabilities: Optional[Sequence[Any]] = None,
    rank_B_log_probabilities: Optional[Sequence[Any]] = None,
    realized_ranks: Optional[Sequence[Any]] = None,
    alphabet_size: Optional[Any] = None,
    tolerance: float = 1e-10,
) -> QualityMetrics:
    """Validate the same-context greedy and rank-``B`` surprisal bounds.

    For each *fixed* history and admissible-token ordering,
    ``-log p_h(1) <= -log p_h(R) <= -log p_h(B)`` for ``1 <= R <= B``.
    Missing endpoint probabilities make that side of the bound unavailable;
    they are never reconstructed from ranks or means.
    """

    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise TheoryValidationError("tolerance must be finite and non-negative")
    realized = _log_probability_sequence(realized_log_probabilities, "realized_log_probabilities")
    greedy = (
        None
        if greedy_log_probabilities is None
        else _log_probability_sequence(greedy_log_probabilities, "greedy_log_probabilities")
    )
    upper = (
        None
        if rank_B_log_probabilities is None
        else _log_probability_sequence(rank_B_log_probabilities, "rank_B_log_probabilities")
    )
    for label, values in (("greedy", greedy), ("rank_B", upper)):
        if values is not None and len(values) != len(realized):
            raise TheoryValidationError(
                "{} and realized evidence lengths differ".format(label)
            )

    ranks: Optional[List[int]] = None
    alphabet: Optional[int] = None
    rank_range_holds: Optional[bool] = None
    if realized_ranks is not None:
        ranks = [_nonnegative_integer(value, "realized_rank") for value in realized_ranks]
        if len(ranks) != len(realized):
            raise TheoryValidationError("Rank and realized evidence lengths differ")
        if alphabet_size is not None:
            alphabet = _alphabet_size(alphabet_size)
            rank_range_holds = all(1 <= rank <= alphabet for rank in ranks)
    elif alphabet_size is not None:
        alphabet = _alphabet_size(alphabet_size)

    realized_surprisal = [-value for value in realized]
    greedy_surprisal = None if greedy is None else [-value for value in greedy]
    upper_surprisal = None if upper is None else [-value for value in upper]
    quality = sum(realized_surprisal) / len(realized_surprisal)
    greedy_quality = (
        None
        if greedy_surprisal is None
        else sum(greedy_surprisal) / len(greedy_surprisal)
    )
    upper_quality = (
        None
        if upper_surprisal is None
        else sum(upper_surprisal) / len(upper_surprisal)
    )
    lower_row = (
        None
        if greedy_surprisal is None
        else all(g <= r + tolerance for g, r in zip(greedy_surprisal, realized_surprisal))
    )
    upper_row = (
        None
        if upper_surprisal is None
        else all(r <= u + tolerance for r, u in zip(realized_surprisal, upper_surprisal))
    )
    lower_expectation = (
        None if greedy_quality is None else greedy_quality <= quality + tolerance
    )
    upper_expectation_holds = (
        None if upper_quality is None else quality <= upper_quality + tolerance
    )
    lower_margin = (
        None
        if greedy_surprisal is None
        else min(r - g for g, r in zip(greedy_surprisal, realized_surprisal))
    )
    upper_margin = (
        None
        if upper_surprisal is None
        else min(u - r for r, u in zip(realized_surprisal, upper_surprisal))
    )
    checks = [
        value
        for value in (lower_row, upper_row, lower_expectation, upper_expectation_holds, rank_range_holds)
        if value is not None
    ]
    all_checks = None if not checks else all(checks)
    bounds_available = greedy is not None and upper is not None
    if all_checks is False:
        status = "failed"
    elif bounds_available:
        status = "validated"
    elif greedy is not None or upper is not None or rank_range_holds is not None:
        status = "partially_evaluable"
    else:
        status = "not_evaluable_missing_endpoint_probabilities"
    return QualityMetrics(
        forced_context_count=len(realized),
        Q_B_nats_per_forced_token=quality,
        Q_greedy_nats_per_forced_token=greedy_quality,
        Q_rank_B_nats_per_forced_token=upper_quality,
        Delta_B_nats_per_forced_token=(
            None if greedy is None else Delta_B(realized, greedy)
        ),
        greedy_lower_bound_available=greedy is not None,
        rank_B_upper_bound_available=upper is not None,
        greedy_lower_bound_holds_per_context=lower_row,
        rank_B_upper_bound_holds_per_context=upper_row,
        greedy_lower_bound_holds_in_expectation=lower_expectation,
        rank_B_upper_bound_holds_in_expectation=upper_expectation_holds,
        rank_range_holds=rank_range_holds,
        all_available_checks_hold=all_checks,
        minimum_lower_margin_nats=lower_margin,
        minimum_upper_margin_nats=upper_margin,
        quality_status=status,
    )


def deterministic_rank_order(
    scores: Sequence[Any], allowed_token_ids: Optional[Sequence[Any]] = None
) -> List[int]:
    """Order token IDs by decreasing score, tie-broken by increasing ID."""

    parsed: List[float] = []
    for index, score in enumerate(scores):
        try:
            number = float(score)
        except (TypeError, ValueError) as exc:
            raise TheoryValidationError("scores[{}] is not numeric".format(index)) from exc
        if not math.isfinite(number):
            raise TheoryValidationError("scores must be finite")
        parsed.append(number)
    if not parsed:
        raise TheoryValidationError("scores must be non-empty")
    if allowed_token_ids is None:
        allowed = list(range(len(parsed)))
    else:
        allowed = [_nonnegative_integer(value, "allowed_token_id") for value in allowed_token_ids]
        if len(set(allowed)) != len(allowed):
            raise TheoryValidationError("allowed_token_ids contains duplicates")
        if not allowed or any(token_id >= len(parsed) for token_id in allowed):
            raise TheoryValidationError("allowed_token_ids must be a non-empty in-range set")
    return sorted(allowed, key=lambda token_id: (-parsed[token_id], token_id))


def _integer_sequence(value: Any, label: str) -> List[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TheoryValidationError("{} must be a JSON integer list".format(label)) from exc
    if not isinstance(value, (list, tuple)):
        raise TheoryValidationError("{} must be an integer sequence".format(label))
    return [_nonnegative_integer(item, label) for item in value]


def _trace_rows(value: Any, label: str) -> List[Mapping[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TheoryValidationError("{} must be JSON trace rows".format(label)) from exc
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise TheoryValidationError("{} must be a list of mappings".format(label))
    return list(value)


def _first_sequence_difference(left: Sequence[Any], right: Sequence[Any]) -> Optional[int]:
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    return None if len(left) == len(right) else common


def verify_exact_recovery_trace(
    encoder_trace: Sequence[Mapping[str, Any]],
    decoder_trace: Sequence[Mapping[str, Any]],
    encoder_configuration: Mapping[str, Any],
    decoder_configuration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate the exact-recovery proposition against explicit rank traces.

    Encoder rows require ``context_token_ids``, ``ranked_token_ids``,
    ``selected_token_id``, and ``expected_rank``.  Decoder rows require
    ``context_token_ids``, ``ranked_token_ids``, and ``observed_token_id``.
    Rank lists are the already-filtered admissible-token order at that history.
    """

    enc_config = ExactReplayConfiguration.from_mapping(encoder_configuration)
    dec_config = ExactReplayConfiguration.from_mapping(decoder_configuration)
    enc_rows = _trace_rows(list(encoder_trace), "encoder_trace")
    dec_rows = _trace_rows(list(decoder_trace), "decoder_trace")
    if len(enc_rows) != len(dec_rows):
        raise TheoryValidationError("Encoder and decoder trace lengths differ")
    if not enc_rows:
        raise TheoryValidationError("Exact-recovery trace must contain at least one step")

    step_reports: List[Dict[str, Any]] = []
    for step, (encoder, decoder) in enumerate(zip(enc_rows, dec_rows)):
        enc_context = _integer_sequence(encoder.get("context_token_ids"), "encoder context")
        dec_context = _integer_sequence(decoder.get("context_token_ids"), "decoder context")
        enc_order = _integer_sequence(encoder.get("ranked_token_ids"), "encoder rank order")
        dec_order = _integer_sequence(decoder.get("ranked_token_ids"), "decoder rank order")
        selected = _nonnegative_integer(encoder.get("selected_token_id"), "selected_token_id")
        observed = _nonnegative_integer(decoder.get("observed_token_id"), "observed_token_id")
        expected_rank = _nonnegative_integer(encoder.get("expected_rank"), "expected_rank")
        if expected_rank < 1:
            raise TheoryValidationError("expected_rank must be 1-indexed")
        encoder_selection_valid = (
            expected_rank <= len(enc_order) and enc_order[expected_rank - 1] == selected
        )
        try:
            recovered_rank = dec_order.index(observed) + 1
        except ValueError:
            recovered_rank = None
        step_reports.append(
            {
                "step": step,
                "contexts_identical": enc_context == dec_context,
                "ranked_token_orders_identical": enc_order == dec_order,
                "forced_token_ids_identical": selected == observed,
                "encoder_rank_selection_valid": encoder_selection_valid,
                "expected_rank": expected_rank,
                "recovered_rank": recovered_rank,
                "rank_recovered_exactly": recovered_rank == expected_rank,
            }
        )

    configurations_identical = enc_config == dec_config
    tie_break_identical = (
        enc_config.tie_break_rule == TOKEN_ID_TIE_BREAK_RULE
        and dec_config.tie_break_rule == TOKEN_ID_TIE_BREAK_RULE
    )
    contexts_identical = all(row["contexts_identical"] for row in step_reports)
    tokens_identical = all(row["forced_token_ids_identical"] for row in step_reports)
    orders_identical = all(row["ranked_token_orders_identical"] for row in step_reports)
    selections_valid = all(row["encoder_rank_selection_valid"] for row in step_reports)
    recovered_exact = all(row["rank_recovered_exactly"] for row in step_reports)
    assumptions = (
        configurations_identical
        and tie_break_identical
        and contexts_identical
        and tokens_identical
        and orders_identical
        and selections_valid
    )
    proposition_confirmed = assumptions and recovered_exact
    return {
        "schema_version": THEORY_SCHEMA_VERSION,
        "proposition": EXACT_RECOVERY_PROPOSITION,
        "trace_step_count": len(step_reports),
        "configurations_identical": configurations_identical,
        "supported_tie_break_identical": tie_break_identical,
        "context_token_ids_identical": contexts_identical,
        "forced_token_ids_identical": tokens_identical,
        "ranked_token_orders_identical": orders_identical,
        "encoder_rank_selection_valid": selections_valid,
        "guarantee_conditions_satisfied": assumptions,
        "recovered_ranks_equal_expected": recovered_exact,
        "observed_exact_recovery": recovered_exact,
        "proposition_confirmed": proposition_confirmed,
        "proposition_violation": assumptions and not recovered_exact,
        "step_diagnostics": step_reports,
    }


def _trace_rank(row: Mapping[str, Any]) -> Optional[int]:
    order = _integer_sequence(row.get("ranked_token_ids"), "ranked_token_ids")
    token_value = row.get("observed_token_id", row.get("selected_token_id"))
    token_id = _nonnegative_integer(token_value, "observed_token_id")
    try:
        return order.index(token_id) + 1
    except ValueError:
        return None


def diagnose_cascading_context_edit(
    reference_trace: Sequence[Mapping[str, Any]],
    edited_trace: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Locate the first replay divergence and quantify downstream propagation.

    This is a diagnostic, not an error-correction claim.  Each row must contain
    ``context_token_ids``, ``ranked_token_ids``, and an observed or selected
    token ID.  A context edit can change the rank order; because the decoder is
    autoregressive, a changed token history may affect every later context.
    """

    reference = _trace_rows(list(reference_trace), "reference_trace")
    edited = _trace_rows(list(edited_trace), "edited_trace")
    if not reference or not edited:
        raise TheoryValidationError("Cascade traces must both be non-empty")
    common = min(len(reference), len(edited))
    per_step: List[Dict[str, Any]] = []
    for step in range(common):
        ref_row = reference[step]
        edit_row = edited[step]
        ref_context = _integer_sequence(ref_row.get("context_token_ids"), "reference context")
        edit_context = _integer_sequence(edit_row.get("context_token_ids"), "edited context")
        ref_order = _integer_sequence(ref_row.get("ranked_token_ids"), "reference rank order")
        edit_order = _integer_sequence(edit_row.get("ranked_token_ids"), "edited rank order")
        reference_rank = _trace_rank(ref_row)
        edited_rank = _trace_rank(edit_row)
        per_step.append(
            {
                "step": step,
                "context_diverged": ref_context != edit_context,
                "context_first_difference": _first_sequence_difference(ref_context, edit_context),
                "rank_order_diverged": ref_order != edit_order,
                "reference_rank": reference_rank,
                "edited_rank": edited_rank,
                "rank_error": reference_rank != edited_rank,
            }
        )

    def first_step(field: str) -> Optional[int]:
        return next((int(row["step"]) for row in per_step if row[field]), None)

    first_context = first_step("context_diverged")
    first_order = first_step("rank_order_diverged")
    first_rank = first_step("rank_error")
    context_count = sum(bool(row["context_diverged"]) for row in per_step)
    order_count = sum(bool(row["rank_order_diverged"]) for row in per_step)
    rank_count = sum(bool(row["rank_error"]) for row in per_step)
    if len(reference) != len(edited):
        length_delta = abs(len(reference) - len(edited))
        context_count += length_delta
        order_count += length_delta
        rank_count += length_delta
    after = -1 if first_context is None else first_context
    post_context = sum(
        bool(row["context_diverged"]) and int(row["step"]) > after for row in per_step
    )
    post_rank = sum(
        bool(row["rank_error"]) and int(row["step"]) > after for row in per_step
    )
    first_token_difference = None
    if first_context is not None:
        first_token_difference = per_step[first_context]["context_first_difference"]
    return {
        "schema_version": THEORY_SCHEMA_VERSION,
        "cascade_status": (
            "context_edit_with_downstream_cascade"
            if first_context is not None and (post_context > 0 or post_rank > 0)
            else "context_edit_without_observed_downstream_cascade"
            if first_context is not None
            else "no_context_edit_observed"
        ),
        "reference_step_count": len(reference),
        "edited_step_count": len(edited),
        "first_context_divergence_step": first_context,
        "first_context_token_difference": first_token_difference,
        "first_rank_order_divergence_step": first_order,
        "first_rank_error_step": first_rank,
        "context_divergence_count": context_count,
        "rank_order_divergence_count": order_count,
        "rank_error_count": rank_count,
        "post_initial_context_divergence_count": post_context,
        "post_initial_rank_error_count": post_rank,
        "cascade_observed": first_context is not None and (post_context > 0 or post_rank > 0),
        "step_diagnostics": per_step,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trial_records(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read CSV/JSONL trial data and attach deterministic source identities."""

    if not paths:
        raise TheoryValidationError("At least one saved CSV or JSONL input is required")
    records: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    seen_paths = set()
    seen_hashes = set()
    for supplied in paths:
        path = Path(supplied)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise TheoryValidationError("Repeated input path: {}".format(path))
        if not path.is_file():
            raise TheoryValidationError("Input does not exist: {}".format(path))
        digest = file_sha256(path)
        if digest in seen_hashes:
            raise TheoryValidationError("Byte-identical inputs would duplicate evidence")
        seen_paths.add(resolved)
        seen_hashes.add(digest)
        suffix = path.suffix.lower()
        loaded: List[Mapping[str, Any]] = []
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TheoryValidationError(
                            "Invalid JSON at {}:{}".format(path, line_number)
                        ) from exc
                    if not isinstance(item, Mapping):
                        raise TheoryValidationError(
                            "JSONL row at {}:{} is not an object".format(path, line_number)
                        )
                    loaded.append(item)
        else:
            raise TheoryValidationError(
                "Unsupported input {}; expected .csv, .jsonl, or .ndjson".format(path)
            )
        for row_index, item in enumerate(loaded, 1):
            row = dict(item)
            row["_theory_source_file"] = str(path)
            row["_theory_source_row"] = row_index
            records.append(row)
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "record_count": len(loaded),
            }
        )
    return records, sources


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number_or_none(value: Any) -> Optional[float]:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> Optional[int]:
    number = _number_or_none(value)
    if number is None:
        return None
    rounded = round(number)
    return int(rounded) if math.isclose(number, rounded, abs_tol=1e-12) else None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and not _missing(record[key]):
            return record[key]
    return None


def _mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _sequence_or_none(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return list(parsed) if isinstance(parsed, list) else None
    return None


def _identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_file": record.get("_theory_source_file"),
        "source_row": record.get("_theory_source_row"),
        "trial_id": _first_value(record, ("trial_id", "work_id", "id")),
        "model_id": _first_value(record, ("model_id", "model_repo_id", "model")),
        "protocol_variant": _first_value(record, ("protocol_variant", "condition_name", "codec")),
        "payload_name": _first_value(record, ("payload_name", "payload_id")),
    }


def _representation_details(record: Mapping[str, Any]) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    representation = _mapping(record.get("representation"))
    if representation is not None:
        name = _first_value(representation, ("name", "representation_name"))
        metadata = _mapping(representation.get("metadata")) or {}
        return (None if name is None else str(name), metadata)
    name = _first_value(record, ("representation_name", "representation", "codec"))
    metadata = _mapping(record.get("representation_metadata")) or {}
    return (None if name is None else str(name), metadata)


def _capacity_inputs(record: Mapping[str, Any]) -> Dict[str, Any]:
    name, metadata = _representation_details(record)
    explicit_H = _first_value(
        record,
        ("H_bits", "entropy_bits", "payload_bits", "representation_source_bits"),
    )
    H = _number_or_none(explicit_H)
    H_source = None if H is None else "explicit_record_field"
    alphabet_value = _first_value(record, ("alphabet_size_B", "alphabet_size", "B"))
    B = _int_or_none(alphabet_value)
    if metadata:
        if B is None:
            B = _int_or_none(metadata.get("alphabet_size"))
        original_bytes = _int_or_none(metadata.get("original_byte_length"))
        encoding = str(metadata.get("encoding", ""))
        hex_length = _int_or_none(metadata.get("hex_character_length"))
        if H is None and original_bytes is not None and B is not None:
            H = float(original_bytes * 8)
            H_source = "codec_metadata_original_byte_length"
        if H is None and encoding == "raw_hex_nibbles" and hex_length is not None:
            H = float(hex_length * 4)
            H_source = "codec_metadata_raw_hex_nibbles"
        if B is None and encoding == "raw_hex_nibbles":
            B = 16
    if name == "hex_nibble" and B is None:
        B = 16
    forced = _int_or_none(
        _first_value(record, ("observed_n_forced", "forced_token_count", "rank_count", "n_forced"))
    )
    tail = _int_or_none(
        _first_value(record, ("observed_n_tail", "tail_token_count", "n_tail"))
    )
    artifact_bits = _number_or_none(record.get("artifact_bit_length"))
    return {
        "representation_name": name,
        "metadata": metadata,
        "H": H,
        "H_source": H_source,
        "B": B,
        "forced": forced,
        "tail": tail,
        "artifact_bit_length": artifact_bits,
    }


def capacity_validation_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        inputs = _capacity_inputs(record)
        row = _identity(record)
        row.update(
            {
                "representation_name": inputs["representation_name"],
                "H_source": inputs["H_source"],
                "artifact_bit_length": inputs["artifact_bit_length"],
            }
        )
        missing = []
        if inputs["H"] is None:
            missing.append("H_bits")
        if inputs["B"] is None:
            missing.append("alphabet_size_B")
        if missing:
            row["capacity_status"] = "not_evaluable_missing_" + "_and_".join(missing)
            row["H_bits"] = inputs["H"]
            row["alphabet_size_B"] = inputs["B"]
        else:
            try:
                metrics = capacity_metrics(
                    inputs["H"],
                    inputs["B"],
                    observed_n_forced=inputs["forced"],
                    observed_n_tail=inputs["tail"],
                )
                row.update(asdict(metrics))
                if inputs["forced"] is None:
                    status = "theory_only_missing_observed_n_forced"
                elif inputs["tail"] is None:
                    status = "nominal_validated_missing_observed_n_tail"
                elif metrics.observed_forced_count_feasible is False:
                    status = "failed_observed_forced_count_below_information_bound"
                else:
                    status = "validated"
                row["capacity_status"] = status
            except TheoryValidationError as exc:
                row["capacity_status"] = "invalid_explicit_input: {}".format(exc)
                row["H_bits"] = inputs["H"]
                row["alphabet_size_B"] = inputs["B"]
        rows.append({column: row.get(column) for column in CAPACITY_VALIDATION_COLUMNS})
    return rows


def _quality_arrays(record: Mapping[str, Any]) -> Dict[str, Any]:
    realized: List[Any] = []
    greedy: List[Any] = []
    upper: List[Any] = []
    ranks: List[Any] = []
    evidence_level = None
    segments = _sequence_or_none(record.get("segments"))
    if segments is not None:
        greedy_complete = True
        upper_complete = True
        ranks_complete = True
        for item in segments:
            segment = _mapping(item)
            if segment is None:
                continue
            current = _sequence_or_none(
                _first_value(segment, ("forced_log_probabilities", "realized_log_probabilities"))
            )
            if current is None:
                continue
            realized.extend(current)
            current_ranks = _sequence_or_none(
                _first_value(segment, ("expected_ranks", "realized_ranks"))
            )
            current_greedy = _sequence_or_none(
                _first_value(segment, ("greedy_log_probabilities", "rank_1_log_probabilities"))
            )
            current_upper = _sequence_or_none(
                _first_value(segment, ("rank_B_log_probabilities", "rank_b_log_probabilities"))
            )
            if current_ranks is None or len(current_ranks) != len(current):
                ranks_complete = False
            else:
                ranks.extend(current_ranks)
            if current_greedy is None or len(current_greedy) != len(current):
                greedy_complete = False
            else:
                greedy.extend(current_greedy)
            if current_upper is None or len(current_upper) != len(current):
                upper_complete = False
            else:
                upper.extend(current_upper)
        if realized:
            evidence_level = "token_context"
            return {
                "realized": realized,
                "greedy": greedy if greedy_complete else None,
                "upper": upper if upper_complete else None,
                "ranks": ranks if ranks_complete else None,
                "evidence_level": evidence_level,
            }

    flat_realized = _sequence_or_none(
        _first_value(record, ("realized_log_probabilities", "forced_log_probabilities", "token_log_probabilities"))
    )
    if flat_realized:
        return {
            "realized": flat_realized,
            "greedy": _sequence_or_none(
                _first_value(record, ("greedy_log_probabilities", "rank_1_log_probabilities"))
            ),
            "upper": _sequence_or_none(
                _first_value(record, ("rank_B_log_probabilities", "rank_b_log_probabilities"))
            ),
            "ranks": _sequence_or_none(
                _first_value(record, ("realized_ranks", "expected_ranks", "ranks"))
            ),
            "evidence_level": "token_context",
        }

    single = _number_or_none(
        _first_value(record, ("realized_log_probability", "forced_log_probability", "token_log_probability"))
    )
    if single is not None:
        greedy_single = _number_or_none(
            _first_value(record, ("greedy_log_probability", "rank_1_log_probability"))
        )
        upper_single = _number_or_none(
            _first_value(record, ("rank_B_log_probability", "rank_b_log_probability"))
        )
        rank_single = _int_or_none(_first_value(record, ("realized_rank", "expected_rank", "rank")))
        return {
            "realized": [single],
            "greedy": None if greedy_single is None else [greedy_single],
            "upper": None if upper_single is None else [upper_single],
            "ranks": None if rank_single is None else [rank_single],
            "evidence_level": "token_context",
        }

    mean = _number_or_none(
        _first_value(record, ("mean_forced_token_log_probability", "mean_token_log_probability"))
    )
    if mean is not None:
        greedy_mean = _number_or_none(record.get("mean_greedy_log_probability"))
        upper_mean = _number_or_none(
            _first_value(record, ("mean_rank_B_log_probability", "mean_rank_b_log_probability"))
        )
        return {
            "realized": [mean],
            "greedy": None if greedy_mean is None else [greedy_mean],
            "upper": None if upper_mean is None else [upper_mean],
            "ranks": None,
            "evidence_level": "trial_mean_only",
        }
    return {
        "realized": None,
        "greedy": None,
        "upper": None,
        "ranks": None,
        "evidence_level": None,
    }


def quality_validation_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        arrays = _quality_arrays(record)
        representation_name, _ = _representation_details(record)
        capacity = _capacity_inputs(record)
        row = _identity(record)
        row.update(
            {
                "representation_name": representation_name,
                "alphabet_size_B": capacity["B"],
                "quality_evidence_level": arrays["evidence_level"],
            }
        )
        if arrays["realized"] is None:
            row["quality_status"] = "not_evaluable_missing_realized_log_probabilities"
        else:
            try:
                metrics = validate_quality_bounds(
                    arrays["realized"],
                    greedy_log_probabilities=arrays["greedy"],
                    rank_B_log_probabilities=arrays["upper"],
                    realized_ranks=arrays["ranks"],
                    alphabet_size=capacity["B"],
                )
                row.update(asdict(metrics))
                if arrays["evidence_level"] == "trial_mean_only":
                    # Bounds between explicitly saved means are aggregate
                    # checks. They cannot establish every per-context
                    # inequality because the underlying contexts are absent.
                    row["forced_context_count"] = capacity["forced"]
                    row["greedy_lower_bound_holds_per_context"] = None
                    row["rank_B_upper_bound_holds_per_context"] = None
                    aggregate_checks = [
                        value
                        for value in (
                            row.get("greedy_lower_bound_holds_in_expectation"),
                            row.get("rank_B_upper_bound_holds_in_expectation"),
                        )
                        if value is not None
                    ]
                    row["all_available_checks_hold"] = (
                        None if not aggregate_checks else all(aggregate_checks)
                    )
                    if any(value is False for value in aggregate_checks):
                        row["quality_status"] = "failed_aggregate_mean_bound"
                    elif arrays["greedy"] is not None or arrays["upper"] is not None:
                        row["quality_status"] = (
                            "aggregate_means_only_per_context_bounds_not_evaluable"
                        )
            except TheoryValidationError as exc:
                row["quality_status"] = "invalid_explicit_input: {}".format(exc)
        rows.append({column: row.get(column) for column in QUALITY_VALIDATION_COLUMNS})
    return rows


def _flatten_runner_ranks(record: Mapping[str, Any], replay_key: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    representation = _mapping(record.get("representation"))
    expected = None
    if representation is not None:
        expected = _sequence_or_none(representation.get("expected_ranks"))
    replay = _mapping(record.get(replay_key))
    recovered = None if replay is None else _sequence_or_none(replay.get("recovered_ranks"))
    if expected is None or recovered is None:
        return None, None
    parsed_expected = [_int_or_none(value) for value in expected]
    parsed_recovered = [_int_or_none(value) for value in recovered]
    if any(value is None for value in parsed_expected + parsed_recovered):
        return None, None
    return [int(value) for value in parsed_expected], [int(value) for value in parsed_recovered]


def exact_recovery_validation_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = _identity(record)
        encoder_trace = _sequence_or_none(record.get("encoder_trace"))
        decoder_trace = _sequence_or_none(record.get("decoder_trace"))
        encoder_config = _mapping(record.get("encoder_configuration"))
        decoder_config = _mapping(record.get("decoder_configuration"))
        if (
            encoder_trace is not None
            and decoder_trace is not None
            and encoder_config is not None
            and decoder_config is not None
        ):
            try:
                report = verify_exact_recovery_trace(
                    encoder_trace, decoder_trace, encoder_config, decoder_config
                )
                row.update({key: report.get(key) for key in EXACT_RECOVERY_COLUMNS})
                row.update(_identity(record))
                row["exact_recovery_status"] = (
                    "proposition_confirmed"
                    if report["proposition_confirmed"]
                    else "proposition_violation"
                    if report["proposition_violation"]
                    else "assumptions_not_satisfied"
                )
                row["missing_evidence"] = None
            except TheoryValidationError as exc:
                row["exact_recovery_status"] = "invalid_explicit_trace"
                row["missing_evidence"] = str(exc)
        else:
            expected, recovered = _flatten_runner_ranks(record, "saved_token_id_replay")
            reported = None
            saved = _mapping(record.get("saved_token_id_replay"))
            if saved is not None and "exact_recovery" in saved:
                reported = _bool_or_none(saved.get("exact_recovery"))
            if expected is not None and recovered is not None:
                observed = expected == recovered
                row.update(
                    {
                        "exact_recovery_status": "observed_rank_replay_only_proposition_not_evaluable",
                        "trace_step_count": len(expected),
                        "recovered_ranks_equal_expected": observed,
                        "observed_exact_recovery": observed,
                        "reported_exact_recovery": reported,
                        "missing_evidence": "encoder/decoder ranked orders and complete replay identities",
                    }
                )
            else:
                row.update(
                    {
                        "exact_recovery_status": "not_evaluable_missing_trace",
                        "reported_exact_recovery": reported,
                        "missing_evidence": "expected/recovered ranks or explicit proposition trace",
                    }
                )
        rows.append({column: row.get(column) for column in EXACT_RECOVERY_COLUMNS})
    return rows


def cascade_diagnostic_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        reference = _sequence_or_none(
            _first_value(record, ("reference_trace", "cascade_reference_trace"))
        )
        edited = _sequence_or_none(
            _first_value(record, ("edited_trace", "cascade_edited_trace"))
        )
        if reference is None or edited is None:
            continue
        row = _identity(record)
        try:
            report = diagnose_cascading_context_edit(reference, edited)
            row.update(report)
        except TheoryValidationError as exc:
            row["cascade_status"] = "invalid_explicit_trace: {}".format(exc)
        rows.append({column: row.get(column) for column in CASCADE_DIAGNOSTIC_COLUMNS})
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {}
        for column in columns:
            value = row.get(column)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            serialized[column] = value
        writer.writerow(serialized)
    return handle.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TheoryValidationError("Refusing to write through symlink: {}".format(path))
    if path.exists() and not overwrite:
        if path.read_bytes() == content:
            return
        raise TheoryValidationError(
            "Theory output exists with different bytes; pass --overwrite: {}".format(path)
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent), delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_theory_artifacts(
    input_paths: Sequence[Path], output_dir: Path, *, overwrite: bool = False
) -> TheoryArtifacts:
    """Build deterministic validation and plot-source tables from saved data."""

    records, sources = read_trial_records([Path(path) for path in input_paths])
    capacity = capacity_validation_rows(records)
    quality = quality_validation_rows(records)
    exact = exact_recovery_validation_rows(records)
    cascade = cascade_diagnostic_rows(records)
    capacity_plot = [
        {column: row.get(column) for column in CAPACITY_PLOT_COLUMNS}
        for row in capacity
        if row.get("theoretical_n_B") is not None
    ]
    quality_plot = [
        {column: row.get(column) for column in QUALITY_PLOT_COLUMNS}
        for row in quality
        if row.get("Q_B_nats_per_forced_token") is not None
    ]
    output = Path(output_dir)
    table_specs = {
        "capacity_validation": (capacity, CAPACITY_VALIDATION_COLUMNS),
        "capacity_plot": (capacity_plot, CAPACITY_PLOT_COLUMNS),
        "quality_validation": (quality, QUALITY_VALIDATION_COLUMNS),
        "quality_plot": (quality_plot, QUALITY_PLOT_COLUMNS),
        "exact_recovery": (exact, EXACT_RECOVERY_COLUMNS),
        "cascade": (cascade, CASCADE_DIAGNOSTIC_COLUMNS),
    }
    files: Dict[str, str] = {}
    table_manifest: List[Dict[str, Any]] = []
    for key, (rows, columns) in table_specs.items():
        path = output / OUTPUT_FILENAMES[key]
        content = _csv_bytes(rows, columns)
        _atomic_write(path, content, overwrite)
        files[key] = str(path)
        table_manifest.append(
            {
                "name": key,
                "path": path.name,
                "row_count": len(rows),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    summary = {
        "input_record_count": len(records),
        "capacity_evaluable_count": len(capacity_plot),
        "quality_evaluable_count": len(quality_plot),
        "quality_fully_bound_validated_count": sum(
            row.get("quality_status") == "validated" for row in quality
        ),
        "exact_proposition_confirmed_count": sum(
            row.get("proposition_confirmed") is True for row in exact
        ),
        "exact_observed_only_count": sum(
            row.get("exact_recovery_status")
            == "observed_rank_replay_only_proposition_not_evaluable"
            for row in exact
        ),
        "cascade_evaluable_count": len(cascade),
    }
    manifest = {
        "schema_version": THEORY_SCHEMA_VERSION,
        "artifact_type": "rankcloak_capacity_quality_theory_validation",
        "proposition": EXACT_RECOVERY_PROPOSITION,
        "tie_break_rule": TOKEN_ID_TIE_BREAK_RULE,
        "probability_log_base": "natural",
        "quality_unit": "nats_per_forced_token",
        "rate_unit": "bits_per_token",
        "missing_result_policy": "no imputation, reconstruction, or synthetic substitution",
        "inputs": sources,
        "tables": table_manifest,
        "summary": summary,
    }
    manifest_path = output / OUTPUT_FILENAMES["manifest"]
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    _atomic_write(manifest_path, manifest_bytes, overwrite)
    files["manifest"] = str(manifest_path)
    return TheoryArtifacts(output_dir=str(output), files=files, summary=summary)

