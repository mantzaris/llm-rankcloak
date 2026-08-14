# Held-out evaluator protocol

## Estimand and model separation

The held-out evaluator reports the mean evaluator-token log probability of a
saved cover or control text conditioned on its recorded overt prompt. It is an
automated model-based quality outcome, not a human-rating substitute.

The evaluator family is predeclared in source and cannot be selected to improve
an observed result:

| generator | evaluator |
|---|---|
| Llama 3 8B Instruct Q4_K_M | Qwen 2.5 7B Instruct Q4_K_M |
| Qwen 2.5 7B Instruct Q4_K_M | Mistral 7B Instruct v0.3 Q4_K_M |
| Mistral 7B Instruct v0.3 Q4_K_M | Llama 3 8B Instruct Q4_K_M |

One invocation loads exactly one evaluator GGUF. It never loads, tokenizes
with, or otherwise opens the generator GGUF. The generator artifact hash is
inherited from the verified runner manifest, while the evaluator artifact is
independently checked by size and full SHA-256 before execution.

## Input integrity

Each source shard must be complete. The evaluator verifies the runner's frozen
configuration hash, run-identity hash, ordered plan hash, payload-manifest hash,
embedded model manifest, source-manifest file-list hash, checkpoint, and a
one-to-one durable completion for every runner work item. Every accepted plan,
run identity, and record must carry `protocol_contract_revision=payload_fidelity_v2`
and `result_schema_revision=payload_aware_result_v2`. The evaluator records the
byte size and SHA-256 of every input artifact. Only payload-fidelity-v2
`primary_v2`, `ablation_v2`, and `multilingual_v2` shards are accepted; non-text
ablation reference rows are verified
but not scored.

The scoring plan embeds the exact prompt and text of every scoring unit and is
ordered by the canonical stage order (`primary_v2`, `ablation_v2`, `multilingual_v2`) and then
by runner-plan order. Every evaluation ID hashes its source-record bytes,
generator/evaluator identities, artifact identity, prompts, texts, and segment
order. Thus checkpoint identity changes if any scientific scoring input changes.

## Scoring

For non-segmented RankCloak texts and ordinary controls, the evaluator tokenizes
the recorded overt prompt as context and the saved full text as its continuation.
It then evaluates one continuation token at a time with deterministic llama.cpp
settings and stable log-softmax arithmetic.

For segmented RankCloak texts, each segment is separately conditioned on that
segment's recorded prompt. Segment log probabilities are summed and divided by
the total number of evaluator tokens. Segments are never treated as independent
observations. Each output records evaluator-token count, total and mean log
probability, total and mean negative log likelihood, scoring wall time,
throughput, hashes, evidence labels, and per-segment diagnostics.

## Execution

After the corresponding runner shards have completed, an exact dry run is:

```bash
.venv/bin/python scripts/run_revision_evaluator.py \
  --evaluator-model qwen2_5_7b_instruct_q4_k_m \
  --source-stage primary_v2 \
  --dry-run
```

The frozen primary matrix contains 4,800 scoreable records per generator shard
(2,160 RankCloak and 2,640 controls). A completed ablation shard contributes 576
scoreable RankCloak records after excluding 48 non-generative references, and a
completed multilingual shard contributes 384 records. The command verifies the
actual completed artifacts before printing its exact task count.

A GPU execution example is:

```bash
.venv/bin/python scripts/run_revision_evaluator.py \
  --evaluator-model qwen2_5_7b_instruct_q4_k_m \
  --source-stage primary_v2 \
  --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf \
  --n-gpu-layers -1
```

Use `--resume` to continue the identical content-hashed plan. `--max-pending N`
is an operational checkpoint chunk and does not change the plan or evidence
label. In contrast, `--limit N` creates different evaluation IDs and applies
the explicit `exploratory_limited_not_for_confirmatory_pooling` label.

### Exploratory smoke-v3 timing

The completed `smoke_v3` runner shards can be evaluated to replace the
compute projection’s unmeasured evaluator proxy. Smoke inputs must be evaluated
alone: the CLI rejects `smoke_v3` combined with another stage and rejects
`--limit`. Source records, scoring-plan rows, feature rows, and the timing
artifact retain the exact
`exploratory_smoke_v3_payload_fidelity_v2_not_for_confirmatory_pooling` source label, a separate
`exploratory_smoke_v3_payload_fidelity_v2_no_confirmatory_pooling` partition, and
`confirmatory_pooling_eligible=false`. Smoke trial and evaluation IDs are
content-addressed in namespaces disjoint from confirmatory IDs.

Run the three predeclared cross-family pairs (one command at a time on the
pinned GPU):

```bash
.venv/bin/python scripts/run_revision_evaluator.py --evaluator-model qwen2_5_7b_instruct_q4_k_m --source-stage smoke_v3 --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --n-gpu-layers -1
.venv/bin/python scripts/run_revision_evaluator.py --evaluator-model mistral_7b_instruct_v0_3_q4_k_m --source-stage smoke_v3 --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --n-gpu-layers -1
.venv/bin/python scripts/run_revision_evaluator.py --evaluator-model llama3_8b_instruct_q4_k_m --source-stage smoke_v3 --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --n-gpu-layers -1
```

Each completed command writes beneath
`results/revision_v1/heldout_evaluator/smoke_v3/<evaluator-model>/`, including
the ordinary evaluator artifacts and `auxiliary_timing.json`. The latter is a
self-hashed `revision_compute` auxiliary timing record. Its elapsed time is the
sum of durable per-task wall times. The independently recorded model-load span
is excluded from the evaluator rate but added to incurred GPU time by the
compute gate. The self-hashed record is accepted through
`project_revision_compute.py --auxiliary-timing` only under the smoke-v3
payload-fidelity contract.
Partial checkpoint chunks do not emit a timing artifact.

## Outputs and statistical ingestion

`records.jsonl` is the durable attempt log; failures include the full exception
type, message, traceback, model identities, source work ID, and evidence label.
`checkpoint.json` is atomically replaced after the corresponding record has
been fsynced. On completion, two analysis tables are written atomically:

- `features.jsonl` is the flat text-feature table. It retains text and prompt
  fields and can be supplied to `run_revision_statistics.py --features`.
- `continuous_quality.jsonl` is the flat continuous-outcome table. It uses the
  content-addressed evaluation ID as its unique trial ID and can be supplied to
  `run_revision_statistics.py --trials`.

Per-segment diagnostic arrays remain only in `records.jsonl`; both analysis
tables contain one row per source payload trial or control view. Their hashes
and row counts are recorded in `features_manifest.json`.

For the prespecified primary R model, do not pass these evaluator rows directly
to the generic Python statistics adapter. After all three primary evaluator
runs and primary preprocessing are complete, build the closed-world join:

```bash
.venv/bin/python scripts/join_revision_evaluator_features.py \
  --preprocessing-manifest results/revision_v1/analysis_inputs/primary_v2/preprocessing_output_manifest.json \
  --evaluator-feature-manifest results/revision_v1/heldout_evaluator/primary_v2/llama3_8b_instruct_q4_k_m/features_manifest.json \
  --evaluator-feature-manifest results/revision_v1/heldout_evaluator/primary_v2/qwen2_5_7b_instruct_q4_k_m/features_manifest.json \
  --evaluator-feature-manifest results/revision_v1/heldout_evaluator/primary_v2/mistral_7b_instruct_v0_3_q4_k_m/features_manifest.json \
  --output-dir results/revision_v1/analysis_inputs/primary_v2_heldout_join_v1
```

The adapter verifies every input digest, proves that evaluator and preprocessing
reference byte-identical runner `records.jsonl` files, recomputes each canonical
source-record SHA-256, checks the cyclic cross-family mapping and text/metadata,
requires both the configured and actually opened evaluator artifact SHA-256 to
equal that evaluator's frozen `models.json` pin,
and requires one evaluator score for every one of the 6,480 primary RankCloak
trials. It emits only primary full-message feature rows. A trial-level evaluator
score is repeated on nested segment rows solely so the R driver can collapse
segments once per payload; segments never become independent observations.
Both the joined CSV and its v1 join manifest are immutable and hash-addressed.
The join manifest records the frozen model-config hash and all three verified
evaluator pins; the locked R driver refuses joins without those assertions.

## Scope

The outcome tests whether a different pinned model family assigns different
conditional probability to RankCloak and matched control texts. It does not
establish human naturalness, semantic correctness, security, or
undetectability. Those questions require the separately specified human study,
automated diagnostics, and steganalysis experiments.
