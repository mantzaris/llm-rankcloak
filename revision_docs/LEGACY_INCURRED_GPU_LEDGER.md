# Legacy incurred-GPU charge ledger

## Purpose and exclusion policy

The legacy ledger accounts for GPU time already spent before the
`payload_fidelity_v2` correction. It is budget bookkeeping only. Every ledger
and entry is marked `charge_only_not_rate_evidence`; both
`scientific_evidence_allowed` and `rate_evidence_allowed` are `false`.
Consequently, these rows may be added to incurred GPU-hours, but may not be
pooled into manuscript analyses, used to estimate throughput, or used to
project any remaining experiment.

The completed-work ledger contains exactly six entries:

- three completed `smoke_v2` runner shards, one per pinned model; and
- three completed `heldout_evaluator/smoke_v2` shards, one per cyclic evaluator.

The stopped, invalidated partial Qwen primary shard remains represented by its
separate immutable invalidation registry entry. The compute gate must verify
both artifacts, add both charges, and reject duplicate source paths, hashes, or
overlapping GPU intervals. This separation avoids treating an incomplete shard
as a completed legacy run while retaining every incurred second.

## Charge definitions

Runner charges use the single durable `memory_profile` interval:
`event.at - event.started_at`. This includes model loading, all trials, runner
overhead, and the monitored session tail. Evaluator charges use the self-hashed
sum of durable per-task wall times plus the independently recorded model-load
event. The latter definition is explicitly a historical charge and is not a
rate estimator.

The frozen expected arithmetic is:

| Source | Incurred GPU seconds |
|---|---:|
| Llama runner smoke-v2 | 341.925859 |
| Qwen runner smoke-v2 | 313.765465 |
| Mistral runner smoke-v2 | 249.206562 |
| Llama held-out evaluator smoke-v2 | 102.799487737022 |
| Qwen held-out evaluator smoke-v2 | 124.109651692022 |
| Mistral held-out evaluator smoke-v2 | 116.029125648041 |
| Completed legacy ledger | 1247.836151077086 |
| Invalidated partial Qwen primary | 2189.687278000000 |
| Total prior incurred charge | 3437.523429077086 (0.954867619188 GPU-hours) |

Values are derived by code from immutable event and timing records. They are
not hand-entered into a compute projection.

## Integrity contract

Creation and verification fail closed unless all of the following hold:

- all six source roots and the ledger itself are real, non-symlink paths;
- each recursively enumerated file is regular and matches its frozen size and
  SHA-256 digest, with no unlisted files or links;
- runner plans, checkpoints, and durable records form one-to-one complete
  smoke-v2 executions with the exploratory-only evidence label;
- evaluator plans, checkpoints, records, input-result manifests, and auxiliary
  timings form one-to-one complete cyclic evaluations;
- run identities, source manifests, model artifact SHA-256 values, runtime
  manifests, and the selected GPU UUID match the frozen legacy identities;
- model-load, memory-profile, and session-finished events are unique,
  chronologically valid, and bound to the expected model and GPU;
- source paths, source-tree hashes, and charge intervals are unique and do not
  overlap; and
- the ledger self-hash, entry count, per-entry charge, total seconds, and total
  hours all reproduce exactly.

The public verifier is
`rankcloak.revision_compute.verify_legacy_gpu_ledger(path)`. It returns a
compact dictionary containing `status`, the ledger SHA-256 identity, verified
entries, `total_seconds`, `total_hours`, and the three exclusion-policy fields.

## No-overwrite workflow

The ledger is published once, outside every charged shard:

```bash
.venv/bin/python scripts/manage_legacy_gpu_ledger.py create \
  --smoke-root results/revision_v1/smoke_v2 \
  --evaluator-root results/revision_v1/heldout_evaluator/smoke_v2 \
  --ledger results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json \
  --created-at 2026-08-09T03:00:00+00:00
```

The script writes to a temporary regular file, flushes it, and publishes via a
no-replace hard link. An existing destination is never overwritten. Recheck
the ledger and all source bytes with:

```bash
.venv/bin/python scripts/manage_legacy_gpu_ledger.py verify \
  --ledger results/revision_v1/incurred_charges/legacy_completed_smoke_v2.json
```

No command in this workflow opens a model, executes a GPU kernel, or modifies a
legacy result shard.
