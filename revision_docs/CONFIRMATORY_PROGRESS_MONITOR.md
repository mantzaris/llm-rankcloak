# Confirmatory progress monitor

`scripts/update_revision_progress.py` provides a read-only view of all durable
revision-v1 runner/evaluator checkpoints and completed detector products.  The
only file it may replace is:

`results/revision_v1/confirmatory_progress_v1.json`

The JSON schema is `rankcloak-revision-confirmatory-progress-v1`.  The file is
self-hashed in `progress_sha256` and binds every checkpoint, plan, record, event,
and detector product used by the snapshot.  Publication uses a same-directory
temporary file, `fsync`, and `os.replace`, so readers see either the prior
complete snapshot or the new complete snapshot.

## Polling commands

Refresh the canonical status and print a one-line polling summary:

```bash
python scripts/update_revision_progress.py --write --compact
```

Print a fresh view without writing:

```bash
python scripts/update_revision_progress.py --compact
```

Verify an existing snapshot, including every source hash:

```bash
python scripts/update_revision_progress.py --check --compact
```

Use the non-compact form when the per-stage, per-shard, condition, interval,
and recovered-error details are needed.  A stale `--check` result while a job
is advancing is expected; run `--write` to publish the next durable snapshot.

## Counting and accounting semantics

`counts` covers every frozen work unit, including controls and non-generative
reference units.  `completed` partitions exactly into `successes` plus
`unavailable`; current execution failures are separate, and `remaining` makes
the overall total identity explicit.

`recovery_counts` is narrower.  It counts only completed payload-bearing work
for which a replay/recovery was actually attempted:

- `payload_bearing_recovery_attempted`
- `successful_payload_recoveries`
- `payload_recovery_failures`
- payload-bearing `unavailable`

This prevents successful controls from being described as payload recoveries.

The frozen evaluator target has one explicit dependency-unavailability
partition.  Once
`results/revision_v1/heldout_evaluator/upstream_dependent_unavailability_v1.json`
exists, the monitor accepts it only when all of the following reconcile:

- the manifest schema, type, protocol/result revisions, authorized projection,
  and self-hash;
- the exact 17,232 scoreable plus 48 upstream-dependent-unavailable equals
  17,280 terminal evaluator units identity;
- the ordered hashes and current bytes of the Mistral ablation plan,
  checkpoint, records, and run identity;
- the exact 48 sorted unit declarations and their hashes against the complete
  upstream records.

Those 48 terminal design units enter `heldout_evaluator` as both `completed`
and `unavailable`.  They enter neither successes, execution failures, nor any
recovery count.  They have no score, imputed value, completion timestamp, or
GPU interval.  The snapshot binds the manifest plus every declared lineage
file in both `heldout_evaluator_upstream_unavailability` and
`source_artifacts`.  Any matching upstream source in an evaluator scoring plan
is rejected as a double count.

The baseline is the exact frozen GO projection
`results/revision_v1/compute_projection_165h_v2.json` (165 GPU-hour budget,
SHA-256 `35f063dc168282b40931fe6b15d534c56fb4b7a300b3161471a3afea27e407d3`).
Its four observed components supply prior actual charges.  Legacy smoke and
the invalidated Qwen shard remain charge-only and are excluded from both
scientific evidence and the confirmatory throughput denominator.  Smoke-v3
rates retain their existing exploratory projection role but are not pooled as
confirmatory results.

Current confirmatory GPU time is the sum of non-overlapping, GPU-UUID-bound
durable wall spans.  A completed memory profile supplies its exact span; an
open session is conservatively bounded from model-load start through the last
durable checkpoint/event.  The scanner rejects duplicate or overlapping
same-GPU intervals and refuses to infer GPU time when durable work lacks a
model-load/GPU identity.

Throughput uses confirmatory terminal work divided only by measured
confirmatory GPU wall time.  The rolling ETA uses up to the latest 50 durable
completion timestamps.  Both are operational estimates, not scientific
effect estimates.  `current.liveness_claim` is always
`none_durable_state_only`: a checkpoint can show what is next without proving
that an operating-system process is presently alive.

Recovered operational errors are reported when a now-completed work unit has
more than one durable attempt (or retained failure metadata).  They do not
alter scientific recovery outcomes.

## Safety properties

The monitor rejects symlinked inputs or output paths, malformed or duplicated
work identities, checkpoint/record disagreement, source changes during a
scan, stale/tampered snapshots, non-confirmatory detector outputs, and GPU
interval overlap.  Evaluator-unavailability manifests additionally fail closed
on extra/missing fields, self/source/unit hash disagreement, non-exact unit
counts, scoring or imputation claims, source-lineage drift, and overlap with a
scoreable evaluator plan.  The monitor never loads a model, starts GPU work,
edits a shard, or changes a frozen configuration, plan, protocol, or result
record.
