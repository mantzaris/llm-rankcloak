# Confirmatory v2 automatic orchestration

The post-primary supervisor is
scripts/supervise_confirmatory_v2.py. It is operational tooling, separate from
the frozen runner, evaluator, protocol, and configuration sources. Mutable CPU
command wiring lives in operations/confirmatory_v2/downstream_commands.json.

## Safe inspection and launch

Inspect the complete action order without writing state or starting a process:

    .venv/bin/python scripts/supervise_confirmatory_v2.py --dry-run

Run the automatic supervisor after inspecting that output:

    .venv/bin/python scripts/supervise_confirmatory_v2.py \
      --poll-seconds 60 \
      --max-retries-per-action 5

This command is also the safe waiting-mode attachment command while primary is
still running. A nonblocking single-instance lock prevents two orchestrators
from racing. On the pinned GPU, the supervisor attaches only to one exclusive,
exactly matching checkpointed action; it waits for an external-only GPU user
and halts on overlapping, duplicate, or scientifically mismatched runners.

The --once option writes one waiting snapshot and returns if the primary gate
is still closed. It is a supervision diagnostic, not a full run.

## Frozen GPU DAG

All three exact primary_v2 shards must be complete before any post-primary
launch. The serial one-GPU order is:

1. ablation_v2: Llama, Qwen, Mistral.
2. multilingual_v2: Llama, Qwen, Mistral.
3. robustness_v2: Qwen, Llama, Mistral.
4. Held-out evaluator, separated by source stage in the order primary_v2,
   ablation_v2, multilingual_v2; within each stage the cyclic
   generator/evaluator mapping is Llama/Qwen, Qwen/Mistral, Mistral/Llama.
5. An identical --resume export pass immediately follows each evaluator
   scoring shard.

Runner launches use the pinned GPU UUID, 4096-token context, all GPU layers,
and no limiting option. Robustness actions carry the exact primary, ablation,
and robustness roots. Orphan attachment checks those roots as well as stage,
model, output directory, GPU, and backend settings.

The evaluator projection gate runs all nine exact dry plans before scoring.
There are 17,232 scoreable evaluator rows. Forty-eight Mistral-ablation
RankCloak rows are structurally unavailable because the frozen isolated
round-trip vocabulary is empty. They are terminal non-outcomes, are never
scored or imputed, and reconcile the frozen target as 17,232 + 48 = 17,280.

The self-hashed accounting artifact is:

    results/revision_v1/heldout_evaluator/upstream_dependent_unavailability_v1.json

It binds the exact 48 source work IDs and canonical record hashes to the
Mistral ablation plan, checkpoint, records, and run identity. Canonical progress
must incorporate this manifest before final completion.

## Checkpointing, retry, and durable status

Incomplete valid GPU shards resume from their existing checkpoint. Complete
valid shards are verified and skipped. A complete checkpoint is accepted only
when the plan count and identity are exact, failures are empty, and every
planned ID has exactly one durable completed record under the payload-fidelity
contract.

Recoverable failures are retried up to --max-retries-per-action. Retry counts
are restored from both the self-hashed state and independent atomic error log.
Nonrecoverable identity, schema, license, artifact, or frozen-design errors stop
immediately.

Operational files are written atomically under
results/revision_v1/supervisor/:

- confirmatory_v2_orchestrator_state.json contains the current action, stage,
  condition, counts, recoveries and failures, cumulative actual GPU-hours,
  measured throughput, rolling ETA, last checkpoint, recovered errors, budget,
  and unavailability accounting.
- confirmatory_v2_recovered_errors.jsonl records bounded retry and
  automatic-resume history.
- confirmatory_v2_events.jsonl records launches, completions, and approximately
  six-hour progress events.
- post_primary_logs/ contains one process log per attempt.
- post_primary_markers/ contains self-hashed evaluator export-pass markers.

Immediately before reports, after statistics, locked-R models, and theory have
all verified, the supervisor publishes a byte-for-byte, atomic, no-overwrite
copy of the final canonical progress document at:

    results/revision_v1/final_progress_snapshot_v1.json

The updater's `--check` mode revalidates this seal and every bound source. The
reporting phase cannot begin before the seal, and the manuscript package
bundles the immutable copy. Canonical progress is not refreshed after the seal
exists, including on a completed-run restart.

The orchestrator calls scripts/update_revision_progress.py --write as the
canonical updater before every budget decision and throughout a running GPU
action until the immutable final snapshot is sealed.

## Hard compute ceiling

The authorized projection
results/revision_v1/compute_projection_165h_v2.json is verified against its
frozen self-hash. Immediately before every GPU launch and at each running poll,
the supervisor replaces conservatively consumed projected work with measured
confirmatory GPU wall time. It stops if cumulative actual use reaches 165
GPU-hours or the revised conservative upper bound exceeds 165 GPU-hours. The 48
structurally unavailable evaluator rows remain conservatively reserved in the
projection even after their terminal lineage is proven.

## Downstream DAG

After all GPU work and evaluator export passes:

1. Strict stage-isolated preprocessing runs for primary, ablation,
   multilingual, and robustness. Each stage has exactly three input model
   shards. Ablation references primary; robustness references both primary and
   ablation. No incomplete-input option is permitted.
2. Three primary held-out evaluator manifests are hash-joined to the 6,480-row
   primary feature table.
3. Confirmatory neural detectors train only from the primary corpus and its
   bound preprocessing manifest; completion requires exactly 28 prespecified
   splits and 56 non-fallback fit rows.
4. Python statistics, locked-R mixed models, and theory validation run after
   complete dependencies. R consumes the held-out feature join and forbids
   fixed-effects fallback.
5. The immutable final progress snapshot is sealed, then reports consume
   statistics, theory, detector, locked-R, preprocessing, and
   evaluator-unavailability manifests.
6. Five main and thirteen Supplementary figures render individually, so retry
   renders only missing products.
7. The manuscript integrator consumes the sealed reports, figures, statistics,
   locked-R output, immutable final progress, and evaluator-unavailability
   manifest.
   It preserves author originals, emits main2.tex, supplementary2.tex, and the
   response letter in a self-hashed package, derives numeric prose only from
   verified machine tables, compiles the package, and enforces the main-text,
   abstract, title, legend, and seven-display limits.

Every downstream CLI is probed before use. A missing option or unexpected
schema fails closed. Existing valid products are hash-verified and skipped;
strict preprocessing uses staging plus atomic directory rename.

Human recruitment, payment, public deposit, DOI release, and other irreversible
external actions are outside this orchestrator. Human outcomes and the DOI
remain explicitly unavailable until separately authorized.
