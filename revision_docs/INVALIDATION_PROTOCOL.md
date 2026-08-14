# Immutable shard-invalidation protocol

## Purpose

A revision shard can be scientifically invalid even when its files, checkpoint,
and rank-replay traces are internally consistent. Invalidation therefore must
preserve the evidence of what happened without making the invalid shard appear
to be a successful or reusable run.

`rankcloak.revision_invalidation` implements an external invalidation registry.
It never writes, marks, moves, renames, truncates, or deletes the shard. The
registry entry captures the stopped shard in place, identifies the superseded
scientific inputs, records the reason and replacement namespace, and charges the
GPU time already incurred. Verification re-hashes the shard and fails closed if
anything later changes.

This protocol supersedes the earlier idea of moving an invalid shard. A path
move would violate the requirement that the forensic Qwen shard remain
byte-for-byte at its existing path.

## Safety contract

Creating an entry requires all of the following:

1. The caller has independently confirmed that every process writing the shard
   has stopped.
2. The CLI includes `--confirm-stopped`. The library API likewise requires the
   keyword argument `confirm_stopped=True`.
3. The shard is a real directory, not a symlink, and contains the frozen runner
   identity, plan, checkpoint, and manifests.
4. The registry entry is at an explicit path outside the shard.
5. The registry path does not already exist. Identical retries are not silently
   accepted and no existing entry is overwritten.
6. A nonempty machine reason code, human-readable reason, replacement
   namespace, and at least one superseding stage are supplied.
7. A final `memory_profile` event is present and is bound to the same selected
   GPU UUID as the run identity and hardware manifest.

The flag is an explicit attestation, not a process-discovery mechanism. The
utility neither kills a process nor claims that operating-system process scans
are race-free. A future writer is also not physically prevented from reopening
the shard. Instead, any subsequent byte, metadata, inode, directory, or file-set
change makes registry verification fail.

## Create transaction

Creation follows a no-shard-write transaction:

1. Resolve and validate the non-symlink shard path.
2. Recursively enumerate directories and regular files without following links.
3. Hash every file through a no-follow descriptor and verify its inode and
   metadata before and after reading.
4. Validate the run-identity self-hash, ordered plan, checkpoint, durable result
   attempts, payload binding, embedded model manifest, and source/runtime/
   hardware hashes.
5. Derive the stopped execution state and exact observed GPU charge from hashed
   events.
6. Repeat the complete shard snapshot and reject any build-time change.
7. Build and self-hash the registry object.
8. Snapshot the shard again immediately before publication.
9. Write and `fsync` a temporary registry file in the destination directory,
   then publish it with a same-directory hard link. Link creation is atomic and
   fails if the destination exists. There is no overwrite fallback.
10. Re-verify the published entry and re-snapshot the shard.

Only the external registry parent and entry are created. Temporary registry
files are cleaned up; no path within the shard is opened for writing.

## Registry schema

Schema version `1.0` uses
`manifest_type: revision_shard_invalidation`. The top-level structure is:

```json
{
  "schema_version": "1.0",
  "manifest_type": "revision_shard_invalidation",
  "created_at": "timezone-aware ISO-8601",
  "scientific_status": "invalidated_not_for_pooling",
  "scope": "entire_shard",
  "stop_attestation": {
    "confirmed_stopped": true,
    "method": "explicit_caller_confirmation"
  },
  "invalidation": {
    "reason_code": "machine_readable_reason",
    "reason": "Human-readable explanation."
  },
  "shard": {
    "absolute_path": "/absolute/original/path",
    "preservation_policy": "verify_in_place_never_edit_move_rename_or_delete",
    "snapshot": {}
  },
  "superseded_identity": {},
  "execution_state": {},
  "incurred_compute": {},
  "superseding_target": {
    "namespace": "new immutable target namespace",
    "stages": ["smoke_v3", "primary_v2"],
    "materialization_asserted": false
  },
  "invalidation_manifest_sha256": "..."
}
```

The self-hash is canonical JSON SHA-256 over the whole object after removing
`invalidation_manifest_sha256`. Mapping keys are sorted, separators are compact,
UTF-8 is preserved, and non-finite numbers are forbidden.

### In-place directory snapshot

`shard.snapshot` records:

- directory and file counts;
- every relative directory and file path;
- device, inode, permission mode, byte size, modification time, and change time;
- SHA-256 for every regular file;
- canonical hashes of the directory and file record lists; and
- a `shard_tree_sha256` binding the two list hashes.

Symlinks and non-regular filesystem objects are rejected. Capturing metadata in
addition to bytes intentionally makes a touch, replacement by an identical copy,
or move-and-recreate operation fail verification.

### Superseded identity

`superseded_identity` binds the invalidation to:

- study ID;
- canonical run-identity self-hash and run-identity file hash;
- frozen configuration and payload-manifest hashes;
- ordered planned-work-ID hash, plan count, and plan file hash;
- checkpoint and optional records file hashes;
- source-manifest and source-file-set hashes;
- runtime and hardware manifest hashes;
- embedded model-artifact-list hash; and
- completed, failed, attempted, and planned counts at invalidation.

The checkpoint must agree with the run identity and durable append log. Unknown
work IDs, duplicate attempts, multiple completions, or an append/checkpoint
mismatch prevent registry creation.

### Stopped and incomplete state

`execution_state` records:

- `caller_confirmed_stopped: true`;
- `terminal_state`, either `stopped_incomplete` or `stopped_complete`;
- the explicit `incomplete` Boolean;
- planned, completed, currently failed, remaining, and durable-attempt counts;
- whether a `session_finished` event was observed and its last reported
  remaining count;
- event count; and
- every original evidence-status label in the frozen plan.

An invalidation does not rewrite those original labels. The top-level
`scientific_status: invalidated_not_for_pooling` overrides their scientific use.

## Conservative incurred GPU charge

`incurred_compute.charge_policy` is `memory_profile_wall_span_v1`.
`incurred_gpu_seconds` is the exact sum, without rounding, of
`event.at - event.started_at` for every hashed `memory_profile` event. This span
includes model loading, completed trials, runner overhead, and an interrupted
partial trial while the selected GPU was monitored. It is therefore the
conservative observed charge for budget accounting; durable completed-record
sums alone would omit the interrupted work.

For each profile, creation verifies:

- timezone-aware start and end timestamps with nonnegative duration;
- selected GPU UUID equality across event, hardware manifest, and run identity;
- positive overall, GPU, and process-RSS sample counts;
- a positive polling interval;
- finite nonnegative initial, final, and peak GPU-memory measurements;
- finite nonnegative sampled and operating-system RSS peaks; and
- a sampled GPU peak no smaller than either endpoint.

The manifest retains each verified span, sample and memory values, and the
canonical hash of the source memory-profile events. It also records the sum of
all durable-attempt `execution_seconds`, the number missing that optional field,
and model-load seconds as diagnostics. Those supporting sums may not exceed the
profile span.

The halted Qwen event currently spans
`2026-08-09T01:31:52.576383+00:00` through
`2026-08-09T02:08:22.263661+00:00`, or exactly `2189.687278` seconds. This value
is documented here but no real registry entry was created by the fixture-only
implementation task.

A compute-budget consumer must first call `verify_invalidation_entry`. Only a
successful verification may add `incurred_gpu_seconds / 3600` to consumed GPU
hours. Invalidated rows must never be used as timing strata, recovery outcomes,
or scientific evidence.

## CLI

Create an external entry only after independently confirming the writer has
stopped:

```bash
.venv/bin/python scripts/invalidate_revision_shard.py create   --shard results/revision_v1/primary/qwen2_5_7b_instruct_q4_k_m   --registry-entry results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json   --reason-code direct_subword_payload_fidelity   --reason "Direct token replay did not guarantee recovery of the original payload bytes."   --superseding-target-namespace rankcloak_scientific_reports_revision_v1/primary_v2/qwen2_5_7b_instruct_q4_k_m   --superseding-stage smoke_v3   --superseding-stage primary_v2   --confirm-stopped
```

This command is an example only and was not executed against the real shard.
Creation prints the self-hashed manifest. It fails with status 2 on an invalid
shard, missing stop confirmation, invalid GPU profile, internal identity drift,
inside-shard registry destination, or existing destination.

Re-verify before every budget projection, analysis selection, or audit:

```bash
.venv/bin/python scripts/invalidate_revision_shard.py verify   --registry-entry results/revision_v1/invalidations/primary__qwen2_5_7b__direct_payload_fidelity.json
```

Verification prints a compact report containing the registry and shard hashes,
run/config/source/plan identities, stopped execution state, charge policy,
incurred GPU seconds, and replacement namespace/stages. Any entry or shard
mutation returns status 2.

## Consumer rules

- The original path remains discoverable, so selection and compute pipelines
  must consult the external invalidation registry before accepting a shard.
- An invalidated shard is never resumed, pooled, partially salvaged, or promoted
  by changing a label.
- A replacement run uses new source, plan, stage, work-ID, and output identities.
- `materialization_asserted: false` means an invalidation reserves/describes the
  replacement namespace but does not claim that the replacement run exists.
- Registry verification is required even when the invalidated directory looks
  unchanged or its run checkpoint reports no failures.
- Deleting either the shard or registry entry is not part of this protocol.

## Fixture coverage

`tests/test_revision_invalidation.py` uses only temporary synthetic shards. It
checks self-hashing, no-overwrite atomic publication, byte-for-byte shard
preservation, required stopped attestation, external destination enforcement,
post-registration byte and metadata mutation, entry tampering, ordered-plan
identity drift, symlink rejection, required memory-profile/GPU binding, unique
superseding stages, and CLI creation/verification. It never moves or registers
the real Qwen shard.
