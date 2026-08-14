# RankCloak capacity, quality, and exact-replay theory

This document fixes the definitions and validation rules used for the
Scientific Reports revision. The executable implementation is
`rankcloak/revision_theory.py`; `scripts/build_revision_theory.py` converts
saved CSV or JSONL trials into machine-readable validation and plot-source
tables. It does not load an LLM or generate replacement observations.

## 1. Bounded-rank capacity

Let `H >= 0` be the number of information bits in the payload representation
being encoded, and let `B >= 2` be the number of admissible rank symbols. A
forced token carries one symbol in `{1, ..., B}`. A fixed-length rank word of
length `n` has `B^n` codewords. Representing all `2^H` values therefore
requires

```text
B^n >= 2^H,
n_B = ceil(H / log2(B)).
```

The nominal finite-payload rate is

```text
R_B = H / n_B <= log2(B) bits per forced token.
```

For the empty payload, the implementation uses `n_B = 0` and `R_B = 0` by
convention. For positive `H`, zero emitted tokens is invalid.

If a cover contains `n_forced` payload-bearing tokens and `n_tail` natural-tail
tokens, its effective rate under the paper's stated denominator is

```text
R_effective = H / (n_forced + n_tail).
```

The denominator does not silently include a lead-in. If a study reports a
whole-channel rate that includes lead-in or framing tokens, those counts must
be added explicitly and named as a different estimand.

### Finite padding

The finite code-space slack is

```text
s_B = n_B log2(B) - H >= 0,
u_B = 2^H / B^n_B = 2^(-s_B),
```

where `u_B` is code-space utilization. For a power-of-two alphabet,
`log2(B)` is an integer and `s_B` is literal binary padding when `H` is an
integer. For a non-power-of-two alphabet it is code-space slack, not a count
of literal padding bits. The implementation labels these cases separately.

For example, eight bits at `B = 8` require three forced tokens. The nominal
rate is `8/3` bits per token, one bit is padded, and only half of the 512
available rank words are used. At `B = 3`, slack is generally non-integral and
must not be described as binary padding.

`H` is representation-specific. For the fixed-radix ASCII codecs it is eight
times the encoded byte length. For raw hexadecimal nibbles it is four times
the hex-character count. The cryptographic artifact's underlying bit length
is retained separately because a formatted hexadecimal or Base64 string can
have a different serialized length. The analysis never infers `H` from a
payload-class label.

These equations assume a fixed-length, injective representation of an
equiprobable `H`-bit value. They are capacity accounting, not a claim of
optimal entropy coding for a nonuniform source. A concrete codec can use more
than `n_B` forced tokens, but an injective fixed-length codec cannot use fewer.

## 2. Same-context quality bound

At a fixed autoregressive history `h`, restrict the vocabulary to the stated
admissible token set and sort it by decreasing model score, breaking exact
score ties by ascending token ID. Write `p_h(r)` for the probability of the
token at 1-indexed rank `r`. For an encoded rank `R` in `{1, ..., B}`,

```text
Q_B = E[-log p_h(R)]
Delta_B = E[-log p_h(R) + log p_h(1)].
```

Natural logarithms are used, so both quantities are in nats per forced token.
At every recorded context,

```text
-log p_h(1) <= -log p_h(R) <= -log p_h(B),
```

and consequently

```text
E[-log p_h(1)] <= Q_B <= E[-log p_h(B)].
```

The validation code checks both the per-context inequalities and their sample
means. It also checks `1 <= R <= B` when ranks and `B` are present. A complete
validation needs the realized, greedy-rank, and rank-`B` log probabilities
evaluated at every same context under the same filter. A realized mean alone
supports an empirical `Q_B`; it cannot validate either endpoint. Missing
endpoint probabilities remain blank and are labeled not evaluable.

`Delta_B` is the realized surprisal penalty relative to greedy choice at those
same contexts. It is not a human-naturalness score and is not a substitute for
the blinded human study.

### Important non-monotonicity limitation

The pointwise ordering above does **not** imply that empirical `Q_B` must be
monotone when comparing separate generations at different values of `B`.
Changing `B`, the payload representation, the token filter, segmentation, or
an earlier selected token changes later autoregressive histories. The sets of
contexts being averaged can therefore differ. Cross-condition quality ordering
is an empirical result requiring paired designs and uncertainty estimates;
the same-context bound alone does not prove it.

## 3. Exact-copy recovery proposition

### Proposition

Suppose encoder and decoder:

1. begin every forced span with identical prompt, lead-in, and prior token IDs;
2. use byte-identical model and tokenizer identities;
3. use identical deterministic inference configuration and serial evaluation
   schedule;
4. use the identical admissible-token set or filter;
5. order tokens by decreasing score with ascending token ID as the deterministic
   tie-break;
6. preserve the same segment boundaries and codec metadata; and
7. replay the identical forced token IDs.

Then inverse ranking recovers every expected rank. If the bounded payload codec
is injective over those ranks, inverse decoding recovers the payload exactly.

### Proof sketch

At the first forced step, the token-ID history and all deterministic
configuration are identical, so encoder and decoder produce the same ordered
admissible-token list. The transmitted token selected at encoded rank `r_1`
therefore has inverse rank `r_1`. Appending the identical token ID preserves
identical histories. Induction gives the same conclusion at every later forced
step. The recovered rank word equals the encoded word, and injectivity of the
payload codec gives exact payload recovery.

The trace validator requires explicit encoder and decoder configurations,
token-ID contexts, ranked token orders, and transmitted token IDs. Matching
recovered ranks without those identities is reported as observed replay only;
it does not validate the proposition's assumptions. Coincidental recovery
under mismatched configurations is likewise not promoted to a guarantee.

The required trace configuration fields are:

```text
model_identity
tokenizer_identity
inference_config_identity
prompt_token_ids
admissible_token_set_identity
tie_break_rule = descending_score_then_ascending_token_id
```

Encoder trace rows contain `context_token_ids`, `ranked_token_ids`,
`selected_token_id`, and `expected_rank`. Decoder rows contain
`context_token_ids`, `ranked_token_ids`, and `observed_token_id`.

## 4. Context edits and cascading errors

Exact-copy replay is a supported condition, not a transmission-error-correcting
channel. A formatting edit can change tokenization or a token ID at one
boundary. The decoder then evaluates a different history. Its rank order can
change at that step; because the altered history is carried forward, later
orders and recovered ranks can also differ.

`diagnose_cascading_context_edit` compares explicit reference and edited traces
and reports:

- the first context-token, rank-order, and recovered-rank divergence;
- the number of divergent contexts, orders, and ranks; and
- whether any divergence persists after the first edited context.

The diagnostic describes propagation. It neither repairs edits nor implies
that every edit must cause a failure. An edit can occasionally leave an order
or recovered rank unchanged.

## 5. Saved-data interface

Run:

```bash
python scripts/build_revision_theory.py \
  --trials results/revision_v1/trials.jsonl \
  --output-dir results/revision_v1/theory
```

Multiple `--trials` groups and both CSV and JSONL inputs are accepted.
Repeated paths and byte-identical duplicate inputs are rejected. Existing
different outputs are not replaced unless `--overwrite` is explicit.

The builder emits:

| File | Purpose |
|---|---|
| `theory_capacity_validation.csv` | Per-record formula inputs, finite-padding diagnostics, and feasibility status |
| `theory_capacity_plot_source.csv` | Evaluable capacity/rate rows for scripted figures |
| `theory_quality_validation.csv` | `Q_B`, endpoint bounds, `Delta_B`, and availability/failure flags |
| `theory_quality_plot_source.csv` | Evaluable quality rows for scripted figures |
| `theory_exact_recovery_validation.csv` | Assumption-by-assumption proposition audit or observed-only status |
| `theory_cascade_diagnostics.csv` | Explicit reference-versus-edited trace diagnostics |
| `theory_validation_manifest.json` | Input hashes, output hashes, row counts, units, and missing-data policy |

Nested revision-runner records are supported. Codec metadata supplies `H` only
when its semantics are explicit (`original_byte_length` for fixed-radix ASCII
or `hex_character_length` with `raw_hex_nibbles`). Flat saved tables may supply
`H_bits`, `alphabet_size`, forced and tail counts, and token-level log
probabilities directly.

No CLI smoke mode creates synthetic evidence. Synthetic examples exist only in
unit tests. If tail counts, endpoint probabilities, replay identities, or
cascade traces are absent, the corresponding cells remain empty and the
validation status states what is missing.

