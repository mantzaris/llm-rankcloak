# Revision V3 generation protocol amendment

Status: prospectively frozen before model acquisition, model loading, or inspection of
any entropy-gated or Q8 model-backed outcome.

Date: 2026-09-01. Repository baseline:
`5c7c36331e039a1f904e5e19fbce2fb70be83049`. The working tree was clean at
the start of this amendment. This document governs computational generation
only; `paperV2/`, any V3 manuscript directory, the response letter, and the
cover letter remain read-only.

## Reason for amendment

The V1 active entropy design emitted filtered rank 1 at entropy-ineligible
positions, its calibration plan described greedy traces, and its entropy and
quantization ledgers derived new row-specific seeds. Those choices do not match
the historical ordinary-control sampler and do not provide the intended paired
seed contract. They are corrected prospectively here. The original V1 schemas
and plan bytes remain recoverable in Git history at commit `5c7c363`.

No model-backed outcome may be generated or inspected until this amendment,
the V2 active configs, the V2 plan builder, and protocol tests have passed and
been committed locally.

## A1. Entropy-ineligible token policy

At every embedding-span position, the encoder computes next-token Shannon
entropy in bits over the same allowed-token mask used by the relevant
RankCloak rank-selection condition. Eligibility is inclusive: a position
carries the next payload rank if and only if `H >= threshold`.

At an ineligible position, the encoder does not emit rank 1. It samples an
ordinary token with the historical matched-control policy:

- temperature: 0.8;
- top-p: 0.95;
- pseudorandom generator: NumPy PCG64 initialized once at the start of the
  embedding span;
- sampler: `numpy_pcg64_serial_top_p_v1_token_id_tiebreak`;
- deterministic logit ordering: descending logit with token ID as the tie
  breaker;
- exclusions: the same BOS/EOS/EOT exclusions as historical controls plus the
  RankCloak condition allowed-token mask.

The role label is `ordinary_sampled_skip`. A skip consumes no payload rank.
It must never be described as greedy, forced rank 1, or `unforced_skip`.

The replay-assisted decoder recomputes entropy before every observed
embedding-span token. It recovers and consumes a rank only at an eligible
position, ignores observed rank at an ineligible position, and appends every
observed token to model context. It receives the public threshold and normal
replay configuration but no saved eligibility mask or gate-position side
metadata.

Gate-disabled generation remains the ordinary deterministic RankCloak
rank-forcing protocol. The ordinary-sampling RNG is not consulted when the
gate is disabled or at eligible positions.

## A2. Entropy calibration

Each model has six clean development prompts, one from each frozen English
prompt category. Each trace contains exactly 128 ordinary generated tokens
under the same temperature-0.8, top-p-0.95 serial PCG64 sampler described
above. Entropy is measured immediately before every sampled token. The six
traces are pooled within model only.

The model-specific moderate threshold is the NumPy-linear median of pooled
development-position entropies. The strict threshold is the corresponding
75th percentile. The ungated condition has no threshold. Calibration prompts,
seeds, generated tokens, and entropy values are retained. Detector labels,
scores, and final evaluation outcomes are not inputs to calibration.

## A3. Paired entropy seeds

An entropy experimental cell is uniquely identified by model, payload class,
representation, prompt template, and payload index. It receives two stable
seeds derived from base seed `20260831` and the stable cell identity:

1. one RankCloak ordinary-skip seed shared by the ungated, moderate, and strict
   RankCloak rows;
2. one separately namespaced ordinary-control seed shared by the three matched
   control rows.

The ungated row records the shared RankCloak seed even though deterministic
rank forcing does not consume it. Moderate and strict runs initialize PCG64
from that same seed at the beginning of their embedding span. Their stochastic
paths may diverge after gate decisions diverge; the required pairing is common
initialization, not identical downstream random draws.

For matched controls, each gate level uses the same ordinary-control seed.
Controls are generated from the same prompt and sampler, with only target
length varying. Consequently, a shorter length-matched control must be a token
prefix of the longer control from the same cell. Six unrelated row seeds are
forbidden.

## A4. Matched quantization contract

The Q4_K_M historical source and new Q8_0 generation use the same
Qwen2.5-7B-Instruct upstream revision
`a09a35458c702b33eeacc393d103063234e8bc28` and GGUF package revision
`8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`. The embedded tokenizer must
produce identical prompt token IDs and payload-side representation ranks
before a pair can run.

For every paired source trial, the runner loads the immutable historical task,
rank record, and full-message ordinary-control record. The Q8 row is required
to match the Q4 source on:

- rendered prompt bytes and prompt token IDs;
- payload identity, bytes, class, index, and split;
- protocol variant, representation metadata, expected ranks, and alphabet;
- token filter, allowed-token mask, lead-in, segmentation, topic schedule, and
  tail policy;
- target token count and control view;
- temperature 0.8, top-p 0.95, and serial sampler identifier;
- inference backend package version 0.3.23 and deterministic backend settings;
- every other generation parameter except the GGUF quantization and its
  content hash.

The ordinary-control seed is read directly from
`generation.sampling_seed` in the raw historical Q4 control record and reused
for Q8. A newly derived quantization seed is not permitted. Deterministic Q8
RankCloak forcing records that matched historical control seed for provenance
but does not use stochastic sampling when the historical protocol contains no
sampling phase. Complete canonical hashes of the historical task, rank record,
and control record bind each executable pair.

## A5. Active schema changes

- Entropy design: `rankcloak-revision-v3-entropy-gate-design-v1` to V2.
- Quantization design: `rankcloak-revision-v3-quantization-design-v1` to V2.
- Generation requirements: V1 to V2, authorizing only the four exact pinned
  local artifacts and a dedicated CUDA generation environment.
- Entropy protocol output: `rankcloak-entropy-gate-v1` to V2.
- Generation plans and preflight outputs: V1 to V2.

The active CSV plans remain deterministic ledgers. Entropy rows now carry the
experimental-cell seed namespace and sampler fields. Calibration rows name the
ordinary sampler and 128-token target. Quantization rows carry the historical
seed and canonical lineage hashes instead of a newly derived seed.

## A6. Frozen experiment sizes and analysis

The substantive matrix is unchanged:

- 18 calibration traces;
- 360 RankCloak entropy rows: 120 ungated, 120 moderate, 120 strict;
- 360 length-matched entropy controls;
- 1,920 new Q8 quantization rows paired with 1,920 immutable Q4 rows.

Fixed-payload entropy generation continues until all payload ranks are consumed
or the declared maximum is reached. Fixed-token-budget outcomes are derived
from the prefix whose length equals the paired ungated embedding span; they are
not additional generations. Quantization directions remain Q4-to-Q8,
Q8-to-Q4, and pooled training with payload-disjoint evaluation.

All planned outcomes, failures, and unavailable cases will be retained. This
amendment changes protocol correctness and pairing; it does not precommit any
direction of effect on recovery, capacity, quality, entropy, rank pressure, or
detectability.
