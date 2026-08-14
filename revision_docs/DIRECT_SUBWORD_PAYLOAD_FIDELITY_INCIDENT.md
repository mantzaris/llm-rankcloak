# Direct-subword payload-fidelity incident

## Status and scope

**Status:** confirmed methodological defect; the first Qwen primary shard was
halted, externally invalidated in full, and must remain excluded from
confirmatory analysis. The replacement payload-fidelity contract passed its
tokenizer preflight and exploratory `smoke_v3` engineering checks. The
superseding `primary_v2` study has not been launched because its conservative
compute projection exceeds the approved ceiling.

The defect was detected during the pre-specified read-only integrity audit after
the active Qwen primary shard passed 200 durable rows. The process was stopped
externally with exit status 130 after 234 durable rows. This document records the
state of the preserved shard and the required supersession boundary. It does not
repair, relabel, or modify any result artifact.

This incident concerns two related assumptions in the tokenizer compatibility
wrapper:

1. direct-subword payload tokenization assumes that a prepended display space
   can be removed by deleting exactly one token; and
2. prompt tokenization assumes that `add_bos=True` necessarily produces a BOS
   token that can be removed by deleting exactly one token.

Neither assumption is portable across the three pinned tokenizer/model
artifacts. The first assumption makes the direct-subword comparator fail literal
artifact fidelity. For Qwen, the second assumption also changes the intended
prompt context for RankCloak generations and ordinary controls.

No human participants were recruited and no human-study stimuli were deployed.

The immutable incident record is historical evidence about the defect and its
containment. Neither the affected-prefix checks nor the replacement smoke
checks are confirmatory scientific results.

## Preserved Qwen shard

The affected directory is:

`results/revision_v1/primary/qwen2_5_7b_instruct_q4_k_m/`

The halted shard is internally consistent as an append/checkpoint artifact:

- 4,800 work units were planned;
- 234 work units were durably completed and 0 were marked failed;
- `records.jsonl` has exactly 234 rows, all with attempt index 1;
- the records are the exact first 234 work IDs in frozen plan order;
- the checkpoint completed set equals the durable completed-record set;
- the prefix contains 101 RankCloak trials and 133 ordinary controls; and
- no `session_finished` event exists, as expected for the externally interrupted
  process.

Thus, this is not checkpoint corruption, duplicate execution, nondeterministic
rank replay, or a damaged model artifact. It is a representation and outcome-
definition defect in otherwise durable results.

### Stable artifact hashes

These hashes identify the preserved post-interruption state. The directory must
not be resumed or altered after this snapshot.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `records.jsonl` | 4,244,118 | `bdc720371129c84e31c5f6fd4c3d5eb983a4f195c649729e60e52a4e429cb00a` |
| `checkpoint.json` | 65,040 | `390eb32626815afd3dcc38be577c7272601be5481b69ae4817bd2bfa363f6ec0` |
| `events.jsonl` | 921 | `8ecd8f82016ac55028c0ada5fed06420a2b7c1f6913c626c93a6ea494385991b` |
| `plan.jsonl` | 5,200,561 | `9c952719ce941db71bcf9a2cf71373eb4f28398d4304c176986395371c00c563` |
| `payload_manifest.json` | 643,935 | `852f4c15ca2db22c9687f205df6f373e0455f4e63f2956fa6a22aba66bde5691` |
| `model_manifest.json` | 1,545 | `39d6e15f359878ba2e80ba4492bec304741dd93c7cce1bf744238392b3bac8d9` |
| `source_manifest.json` | 2,377 | `94d9d7c0b06733e9dfdce8f39141066002374b73dad454a0a00c0e92b50c8322` |
| `runtime_manifest.json` | 1,084 | `c206a1cbfbbb4781d87fd120a3fe45400a505dd2fa2bfb346278c84e56776664` |
| `hardware_manifest.json` | 634 | `3f59f33f69daa2d5d71531968a409162229266db785361637834e166a903522a` |
| `run_identity.json` | 2,898 | `08b77d41fe8c122e707b0b34ee1a1d21363013dd2da3a09a17205f3c89aa145c` |

The canonical self-hash stored inside `run_identity.json` is
`fda1f6aba51df4f18606f017f66eb64c8efe7700ef18b23263157772de772c76`.
The ordered-plan hash is
`93d6cdeaf12f402d33bca9198d54770c77898d0f272eceb47c6a0436e2cab39d`,
and the frozen configuration-manifest hash is
`dc0e7e022036e2681c87ad06446cbebd56d676faf81a0544a55d56375d4eadcd`.

The Qwen GGUF was independently rehashed after interruption. Its size was
4,683,074,240 bytes and its SHA-256 was
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`,
identical to the pinned and launch-verified value.

## Observed payload-fidelity failure

The halted prefix contains 17 direct-subword trials. All 17 reported exact
saved-token replay because the recovered token IDs equalled the stored target
token IDs. Only 11 of 17 recovered the original payload text literally. In the
remaining 6 trials, Qwen removed an original prefix: one hexadecimal character
in 4 trials and two hexadecimal characters in 2 trials. In contrast, all 84
completed bounded-codec trials in the same prefix recovered the original payload
text (case-insensitively for the explicitly case-insensitive hex-nibble decoder).

These are invariant checks on an invalidated prefix, not confirmatory recovery
estimates.

Privacy-minimized examples are shown below. Full payload strings are omitted;
the payload names and hashes are sufficient to reproduce each record from the
public deterministic corpus.

| Payload | Payload-text SHA-256 | Original length | Recovered length | Failure |
|---|---|---:|---:|---|
| `revision_v1_sha256_hex_002` | `757e1b864dc66d9fdb51d7113a25bac8beb4663458f507e6bbe7feb981be7e36` | 64 | 63 | one original prefix character removed |
| `revision_v1_sha256_hex_007` | `1796c9626505c4759091649dcf14d56c31fe5dd2c6c3f38f7b3a801b4b26cd1d` | 64 | 63 | one original prefix character removed |
| `revision_v1_sha256_hex_009` | `01730e4e384b47a0055b1fa6b1bd21e1c8b76f075bf12eb37bffc6d4dd2e7389` | 64 | 63 | one original prefix character removed |
| `revision_v1_sha256_hex_010` | `26508a7eaff58da60bbf4b16745931b06ee657787e99ad2edb974d60cd5ac073` | 64 | 62 | two original prefix characters removed |
| `revision_v1_sha256_hex_014` | `6a83c87594b435ecf36c9e474d1eef370dd6b39d7e8f365da5b44b953fb5442a` | 64 | 62 | two original prefix characters removed |
| `revision_v1_sha256_hex_016` | `959c56ef95127b284ee0246d2f8ddd87474610ba063f87b2e240ca1f55e73e21` | 64 | 63 | one original prefix character removed |

For every affected example, the recovered ranks, generated forced-token IDs,
and inverse-transcoded token IDs were internally exact. The false-positive
classification occurs one layer earlier: the stored direct-subword target token
sequence did not faithfully represent the original payload text.

## Exact root cause

The frozen source bytes are identified by `source_manifest.json` above. In that
source snapshot:

- `rankcloak/model_io.py:295-304` prepends one ASCII space to the payload,
  requests `add_bos=True`, and then removes the first returned token. Lines
  300-301 remove a verified BOS when present, but lines 302-303 remove the first
  token even when it is **not** the BOS.
- Token boundaries do not correspond to character boundaries. A tokenizer can
  merge the artificial leading space with one or more initial payload symbols.
  Deleting that token therefore deletes payload content. Conversely, when a BOS
  is present, deleting only the BOS can retain one or more artificial leading
  spaces in the reconstructed display text.
- `rankcloak/revision_protocol.py:227-250` stores this already transformed token
  sequence as `payload_token_ids` for the direct representation.
- `rankcloak/revision_protocol.py:298-304` defines direct
  `decoded.exact_recovery` as equality between recovered token IDs and those
  stored token IDs. Unlike the bounded branches, it does not compare recovered
  bytes or text to the original `representation.payload_bytes` or
  `representation.payload_text`.

Therefore exact inverse-rank replay can be true while exact artifact recovery is
false. The result schema currently conflates these two outcomes for the direct
condition.

There is a related prompt-context defect in `rankcloak/model_io.py:307-321`.
For nonempty prompts, lines 313-316 likewise discard the first token whether or
not it is a verified BOS. The observed Qwen payload truncation demonstrates that
this pinned tokenizer can ignore `add_bos=True`; consequently its prompt helper
also discards the first lexical prompt token. The helper is used for ordinary
controls at `rankcloak/revision_runner.py:952-980` and for every RankCloak
segment at `rankcloak/revision_runner.py:1235`. This affects the intended Qwen
prompt condition even though encoder and decoder share the same shortened token
context.

## Why smoke and tests missed the defect

The balanced `smoke_v2` shards did contain a direct-subword warning signal:

| Model | Direct smoke rows | Token-ID exact | Literal payload-text exact | Observed mismatch |
|---|---:|---:|---:|---|
| Llama 3 8B Instruct | 1 | 1 | 0 | one leading space added |
| Mistral 7B Instruct v0.3 | 1 | 1 | 0 | two leading spaces added |
| Qwen 2.5 7B Instruct | 1 | 1 | 0 | one original prefix character removed |

Smoke passed because its recovery gate consumed the recorded direct
`decoded.exact_recovery` value, which checks equality to the already transformed
token sequence. It did not independently compare the recovered bytes/text with
the immutable corpus payload. Thus all three direct smoke rows were accepted as
exact despite failing literal artifact fidelity.

The unit-test doubles also masked the compatibility branch:

- `tests/test_revision_protocol.py:19-50` and
  `tests/test_revision_runner.py:26-64` always honor `add_bos=True` and emit an
  explicit BOS. They do not model a tokenizer that omits BOS or fuses a leading
  space with payload symbols.
- `tests/test_revision_protocol.py:107-114` asserts recovered-token equality,
  not equality to the original payload text or bytes.
- `tests/test_revision_runner.py:286-310` likewise asserts the schema's
  token-based `exact_recovery` fields without a direct payload-fidelity
  assertion.

The incident is therefore both an implementation defect and a validation-gate
defect: the smoke evidence needed to catch it existed, but the primary outcome
was checked at the wrong semantic layer.

## Affected models, conditions, and estimands

### Confirmed affected conditions

- **Direct subword, all three pinned model families.** Every `smoke_v2` direct
  row differed from its original payload text. The direction is tokenizer-
  specific: added leading spaces for the Llama and Mistral smoke rows, and
  deleted payload prefixes for Qwen.
- **Halted Qwen primary direct condition.** Six of the first 17 direct trials
  lost original payload content. The remaining 11 must not be retained as a
  confirmatory subset because the unsafe representation rule was common to all
  17 and success depended on incidental token boundaries.
- **Qwen prompt-conditioned RankCloak and ordinary-control generation.** The
  same unverified first-token deletion is used to construct all nonempty Qwen
  prompt contexts. Encoder/decoder agreement does not restore the omitted
  prompt content.

### Estimands invalidated by this source identity

- direct-subword exact artifact recovery;
- direct-subword forced-token count, rank-pressure distribution, cover length,
  log-probability, and effective-rate quantities, because they describe a
  transformed or truncated payload rather than the declared artifact;
- contrasts and interactions that use direct subword as the comparator,
  including multi-model capacity-quality frontiers;
- Qwen prompt-category, topic-adherence, cover-quality, and ordinary-control
  comparisons under the intended prompt definitions; and
- any detector, evaluator, human-stimulus, or figure/table input derived from
  these invalidated cover records.

The exact rank-replay traces remain useful only as forensic diagnostics. They
must not be counted as confirmatory artifact recovery. Bounded payload decoding
itself showed no payload-fidelity failure in the halted prefix, but bounded Qwen
rows from this shard still cannot be cherry-picked into the confirmatory matrix:
they share the superseded source identity and unintended prompt construction.

No Llama or Mistral primary shard had been launched when this incident was
identified. Their prior smoke outputs are exploratory diagnostics and must not
be promoted to confirmatory evidence.

## Invalidation and supersession requirements

Before any full primary launch resumes, all of the following are required:

1. Preserve the halted Qwen directory byte-for-byte under its existing run
   identity. Do not resume it, overwrite it, rename it into a valid primary
   shard, or pool any of its rows with a replacement run.
2. Record an explicit machine-readable invalidation/supersession relation in
   the replacement study metadata. The reason must identify both direct payload
   fidelity and unverified BOS removal; absence of execution failures is not a
   reason to retain the rows.
3. Replace the tokenizer wrapper with a model-independent procedure whose
   declared input-to-token-to-output normalization is explicit and verified.
   Whitespace must not be removed by assuming it occupies one token, and BOS
   removal must occur only after verifying the returned token ID is the pinned
   model's BOS.
4. Separate at least two outcomes in machine-readable records and analysis:
   exact rank/token replay and exact original artifact recovery. The latter must
   compare recovered bytes or an explicitly predeclared canonical form against
   the immutable corpus payload.
5. Add regression fixtures for (a) `add_bos=True` being ignored, (b) a leading
   space fused with one payload symbol, (c) a leading space fused with multiple
   payload symbols, and (d) model-specific added prefix spaces. Tests must assert
   original payload-byte/text fidelity, not only stored-token equality.
6. Add a tokenizer-only preflight over all 480 frozen payloads, all prompt
   templates, and all three pinned tokenizers. It must fail closed on any
   unexplained prefix/suffix insertion or deletion and save the model-specific
   round-trip manifest.
7. Create new immutable smoke and primary identities/work IDs/output
   directories. Do not rewrite `smoke_v2` or the halted primary shard. The
   replacement smoke must repeat all three models and explicitly audit literal
   payload fidelity and prompt-context fidelity before compute approval.
8. Recompute the GPU-budget projection from the replacement smoke. The prior
   projection is not a valid launch gate because corrected token sequences and
   prompt lengths can change runtime and cover lengths.
9. Regenerate all downstream evaluator, detector, robustness, statistical,
   figure, and table inputs from the superseding primary outputs. Any prior
   timing artifact retained for engineering reference must remain labeled
   exploratory and must not establish scientific equivalence.
10. Add the incident and supersession decision to the revision audit trail and
    response-letter provenance. Do not report the 11/17 or 84/84 prefix checks
    as study results; they are forensic checks from an invalidated partial run.

These requirements preserve the original RankCloak contribution. They correct
the representation boundary and recovery estimand without changing the rank
definition, bounded codecs, segmentation protocols, or exact-copy replay scope.

## Verified remediation and current execution boundary

The following remediation was verified after the incident. It does not alter
the preserved shard described above.

1. The whole shard was registered externally as
   `invalidated_not_for_pooling` in
   `results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json`.
   Its self-hash is
   `a9836f60344c38568f4dbc014deb6c428b1bfad216f9a55da683edd978f9168c`;
   it binds the preserved tree hash
   `97af0aeadc76127c1d7eaa426869f084ec6f16769ffd2b58ab154a774aa0f108`,
   all 234 durable rows, and the 2,189.687278-second observed GPU interval.
   No row from the shard is eligible for pooling.
2. Direct payload tokenization now consumes the original UTF-8 bytes with
   special-token insertion disabled and never deletes a token by position.
   Detokenization must reproduce the original payload as an exact suffix; only
   explicitly recorded ASCII-space prefix bytes are permitted. Prompt
   tokenization removes a first token only when it equals the pinned model's
   actual BOS identifier.
3. Result records bind
   `protocol_contract_revision = payload_fidelity_v2` and
   `result_schema_revision = payload_aware_result_v2`. The primary endpoint is
   `exact_payload_recovery` under
   `original_serialized_payload_bytes_sha256_v1`: recovered bytes and their
   hash must equal the immutable original payload. The separate
   `exact_representation_recovery` endpoint retains rank/token replay as a
   diagnostic, and the compatibility alias `exact_recovery` is required to
   equal the payload endpoint.
4. Regression tests cover tokenizers that omit BOS, one- and multi-symbol
   space fusion, model-added prefix spaces, non-space/internal/suffix
   transformations, prompt first-token retention, empty prompt handling, and
   endpoint/alias/hash invariants. The preprocessing gate rejects records with
   the old recovery semantics.
5. The vocabulary-only tokenizer preflight evaluated 480 payloads and 30
   prompts under each of the three pinned tokenizers: 1,530 checks in total.
   All passed. Llama and Qwen required no prefix bytes; Mistral's 510 sequences
   used one explicit reversible ASCII-space prefix. The preflight performed no
   generation or context evaluation. Its manifest self-hash is
   `b61eaaed4086124774ae4cca37261a91a5c643d575a89f8dc9c33f505bdb1bec`.
6. New `smoke_v3` identities completed 96 of 96 planned work items with zero
   execution failures. Ninety-three rows were available outcomes. The 35
   available RankCloak trials passed representation replay and original-payload
   recovery, including one direct trial for each model family. Three Mistral
   rows were explicit filter-condition/dependency unavailability records (one
   source and two controls), not recovery failures. All smoke-v3 records remain
   labelled exploratory and cannot enter confirmatory estimands.
7. The replacement compute report
   `results/revision_v1/compute_projection_v2.json` has projection SHA-256
   `685da8dfb81f082eda18893f7681462d9f3fec6216f8c978a8cf92ba8b7b90ad`.
   Its point projection is 56.017000 GPU-hours, but its conservative projection
   is 158.508216 GPU-hours, 8.508216 hours above the approved 150-hour ceiling.
   The verified decision is `no_go_over_budget`; consequently no
   `primary_v2` shard has been launched.

Thus the representation and outcome-definition defect is repaired at the
implementation, schema, preflight, and exploratory-smoke layers. Confirmatory
scientific validation remains pending and must use only the disjoint
`primary_v2`, `ablation_v2`, `multilingual_v2`, and `robustness_v2` identities
after a valid compute authorization.
