# Revision-v1 compute-budget gate

## Current decision

`scripts/project_revision_compute.py` is the fail-closed gate between the
payload-fidelity-v2 exploratory smoke run and the expensive revision study. It
verifies all three completed `smoke_v3` runner shards, the three held-out
evaluator smoke timings, the charge-only legacy ledger, the invalidation record
for the stopped Qwen shard, and the frozen superseding plans. The approved
ceiling is 150 GPU-hours.

The current frozen report is
`results/revision_v1/compute_projection_v2.json`, with projection SHA-256
`685da8dfb81f082eda18893f7681462d9f3fec6216f8c978a8cf92ba8b7b90ad`.
Its decision is **`no_go_over_budget`**:

| Quantity | GPU-hours |
|---|---:|
| Point projection | 56.017000 |
| Conservative upper projection | 158.508216 |
| Approved ceiling | 150.000000 |
| Conservative headroom | -8.508216 |

No `primary_v2` shard was launched after this decision. A point estimate below
the ceiling does not override the prespecified rule: the conservative upper
total controls the launch decision. Execution can proceed only after explicit
authorization of a higher ceiling or a separately approved and refrozen design
whose new projection passes the same gate.

## Evidence and recovery contract

The superseding runner contract is
`protocol_contract_revision = payload_fidelity_v2` and
`result_schema_revision = payload_aware_result_v2`. Direct payload tokenization
uses literal UTF-8 bytes, disables special-token insertion, never removes a
token by position, and permits only explicitly recorded reversible ASCII-space
prefix bytes. Cover-prompt tokenization removes a first token only when that
token is the pinned model's actual BOS.

The primary recovery endpoint is equality of recovered serialized payload
bytes with the immutable original payload, checked by SHA-256 under
`recovery_outcome_semantics = original_serialized_payload_bytes_sha256_v1`.
Exact rank/token replay remains a separate diagnostic endpoint. The compatibility
field `exact_recovery` must equal `exact_payload_recovery`; equality to an
already transformed target-token sequence is not sufficient.

The only possible go decision is `go_within_budget`. It requires:

- all three exact `smoke_v3` model shards and complete checkpoints;
- one durable result per frozen smoke work ID and no current execution failures;
- the exploratory-only payload-fidelity-v2 evidence label on plans and records;
- the two contract revisions and original-byte recovery semantics above;
- intact plan, run-identity, configuration, payload, model, source, runtime,
  hardware, records, and events hashes;
- exactly three valid held-out evaluator smoke timings;
- the verified six-entry legacy incurred-charge ledger;
- exactly one verified whole-shard invalidation manifest for the stopped Qwen
  `primary` shard; and
- a conservative total no greater than the approved ceiling.

Missing, duplicated, failed, differently labelled, semantically stale, or
tampered evidence produces `no_go_incomplete_inputs`. A complete verified
projection over the ceiling produces `no_go_over_budget`.

`smoke_v3`, `primary_v2`, `ablation_v2`, `multilingual_v2`, and
`robustness_v2` have identities disjoint from the superseded stages and from
one another. The gate accepts only the `smoke_v3` evidence label
`exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling`.
Passing engineering checks or contributing timing evidence never promotes a
smoke row into a confirmatory analysis.

## Tokenizer preflight and smoke-v3 verification

Before generation, the vocabulary-only preflight checked all 480 frozen
payloads and all 30 prompts (18 English, six Spanish, and six Simplified
Mandarin) under each of the three pinned tokenizers. The resulting 1,530 checks
all passed; no generation or context evaluation occurred. Llama and Qwen
required zero prefix bytes, whereas Mistral used one explicit reversible ASCII
space for every checked sequence. The immutable manifest is
`results/revision_v1/tokenizer_preflight_v2/TOKENIZER_PREFLIGHT_MANIFEST.json`,
whose self-hash is
`b61eaaed4086124774ae4cca37261a91a5c643d575a89f8dc9c33f505bdb1bec`.

Each `smoke_v3` shard contains `plan.jsonl`, `run_identity.json`,
`payload_manifest.json`, `model_manifest.json`, `source_manifest.json`,
`runtime_manifest.json`, `hardware_manifest.json`, `checkpoint.json`,
`records.jsonl`, and `events.jsonl`. Verification includes:

- equality of the current frozen per-model plan and ordered work IDs;
- exact smoke-v3 study, model, exploratory evidence, protocol, and result-schema
  identities;
- self-hashes and all manifest bindings in the run identity;
- the pinned model digest and successful artifact verification;
- a complete checkpoint, zero current execution failures, and a final session
  event with zero work remaining;
- exactly one completed record for each of 32 planned work IDs per model;
- fail-closed validation of original-payload recovery for RankCloak records;
- exact dependency joins and non-outcome semantics for unavailable rows; and
- a finite model-load timing and complete GPU occupancy interval.

All 96 work items completed. Ninety-three were available outcomes: 35
RankCloak trials and 58 ordinary controls. All 35 available RankCloak rows
passed both representation replay and original-payload recovery; the three
direct rows, one per model family, passed the repaired byte-level endpoint.
These are exploratory engineering checks, not estimates for the paper.

The remaining three Mistral rows are one `condition_unavailable` source and two
`dependent_unavailable` controls. Its 31,464 safe-text tokens contain zero
tokens satisfying the additional isolated detokenize--retokenize identity
criterion. The rows are completed non-outcomes, not runtime or recovery
failures. An exact frozen-factor join projects 48 unavailable Mistral ablation
rows; exact source joins project 336 unavailable robustness rows. All remain in
workload counts but contribute zero execution time and remain outside recovery
and timing estimands.

## Invalidated and legacy incurred compute

The halted pre-remediation shard is preserved at
`results/revision_v1/primary/qwen2_5_7b_instruct_q4_k_m/`. Its complete
scientific status is `invalidated_not_for_pooling`; none of its 234 durable
rows may be resumed, selected, relabelled, or pooled. The external
no-overwrite manifest is
`results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json`,
with self-hash
`a9836f60344c38568f4dbc014deb6c428b1bfad216f9a55da683edd978f9168c`.
It binds the preserved shard tree hash
`97af0aeadc76127c1d7eaa426869f084ec6f16769ffd2b58ab154a774aa0f108`
and charges its exact observed GPU occupancy interval: 2,189.687278 seconds
(0.608246 GPU-hours).

The six completed legacy smoke-v2 charges are retained only in
`results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json`. They
cannot provide timing rates or scientific evidence. Together, the invalidated
shard and legacy ledger contribute 3,437.523429 seconds (0.954868 GPU-hours),
with no duplicate paths, run identities, or GPU intervals. These sunk charges
are included in both point and conservative totals.

## Frozen workloads

The projector materializes each plan with the same runner plan builder used by
execution and asserts the frozen arithmetic. Drift is fatal.

| Stage | Frozen work units | Projected available | Projected unavailable |
|---|---:|---:|---:|
| `smoke_v3` (observed) | 96 | 93 | 3 |
| `primary_v2` | 14,400 | 14,400 | 0 |
| `ablation_v2` | 1,872 | 1,824 | 48 |
| `multilingual_v2` | 1,152 | 1,152 | 0 |
| `robustness_v2` | 3,744 | 3,408 | 336 |
| Held-out evaluator | 17,280 | 17,280 | 0 |
| Neural detectors | 56 fits | 56 | 0 |

The primary plan contains 6,480 RankCloak and 7,920 control rows. Ablations
contain 1,728 RankCloak generations and 144 zero-compute references.
Multilingual work contains 576 RankCloak and 576 controls. Robustness contains
3,168 decodes, 432 references, and 144 transformation generations. Detector
configurations remain CPU-only and therefore contribute zero GPU-hours; a
future device change requires new timing evidence and a new gate.

## Estimator

Completed unavailable smoke rows count toward the 96 completed work units with
zero execution seconds but are removed before timing strata are formed. Their
projected descendants receive explicit zero rates and are never treated as
fast observations.

Point rates are within-model medians. RankCloak generation rates are stratified
by protocol; controls are stratified by full-message versus forced-span view.
The projected RankCloak rate is encoding plus supported saved-token decoding;
smoke-only retokenization and greedy-regeneration diagnostics are excluded.
Model-load time is added once to every projected stage/model shard.

Conservative per-unit rates start with the observed maximum in the relevant
stratum and apply this prespecified small-sample factor:

| Smoke observations in stratum | Factor |
|---:|---:|
| 1 | 2.00 |
| 2 | 1.75 |
| 3--4 | 1.50 |
| 5 or more | 1.35 |

This is a planning bound, not a sampling confidence interval. Robustness
decodes receive an additional 1.5 transformation-overhead factor. Severe
paraphrase generation uses the Qwen full-message control rate, multiplied by
1.5 for the point projection and by 2.0 after the ordinary conservative-rate
calculation. References add no execution time.

All three evaluator timing inputs are complete smoke-v3 observations under the
frozen cyclic mapping: Qwen scores Llama outputs, Mistral scores Qwen outputs,
and Llama scores Mistral outputs. The evaluator stage uses their observed
serial scoring rates with a 2.0 upper multiplier. Model load is charged once
per projected evaluator shard. Point and upper GPU-hours are serial sums, not
parallel wall-clock estimates.

The stage totals recorded in the frozen report are:

| Component | Point GPU-hours | Conservative GPU-hours |
|---|---:|---:|
| `primary_v2` | 25.475365 | 82.337546 |
| `ablation_v2` | 5.071071 | 17.354701 |
| `multilingual_v2` | 2.172055 | 6.839401 |
| `robustness_v2` | 2.579539 | 11.870830 |
| Held-out evaluator | 19.386767 | 38.773535 |
| Neural detectors | 0.000000 | 0.000000 |
| Observed smoke-v3 generation | 0.270597 | 0.270597 |
| Observed evaluator smoke-v3 | 0.106739 | 0.106739 |
| Invalidated stopped shard | 0.608246 | 0.608246 |
| Other legacy incurred compute | 0.346621 | 0.346621 |
| **Total** | **56.017000** | **158.508216** |

## Reproduction command

The no-overwrite report was produced with the three smoke-v3 shards discovered
under one root, the three evaluator timing manifests, and both incurred-charge
audits:

```bash
.venv/bin/python scripts/project_revision_compute.py \
  --smoke-root results/revision_v1/smoke_v3 \
  --auxiliary-timing results/revision_v1/heldout_evaluator/smoke_v3/llama3_8b_instruct_q4_k_m/auxiliary_timing.json \
  --auxiliary-timing results/revision_v1/heldout_evaluator/smoke_v3/qwen2_5_7b_instruct_q4_k_m/auxiliary_timing.json \
  --auxiliary-timing results/revision_v1/heldout_evaluator/smoke_v3/mistral_7b_instruct_v0_3_q4_k_m/auxiliary_timing.json \
  --legacy-incurred-ledger results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json \
  --invalidation-manifest results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json \
  --budget-gpu-hours 150 \
  --output results/revision_v1/compute_projection_v2.json
```

Exit status is 0 for `go_within_budget`, 2 for incomplete or invalid inputs,
and 3 for a verified over-budget projection. The current invocation returned
the third state. The output is a forecast, not permission to spend; actual GPU
occupancy must also be monitored against any subsequently approved ceiling.
