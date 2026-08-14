# Neural detector execution

The confirmatory detector matrix contains 56 neural fits: two frozen
architectures over 28 frozen data splits. The first production attempt used
the original serial, CPU-only implementation. A read-only audit after roughly
23 hours established that it was genuinely computing, but that it had only
completed the first transformer fit, exposed no durable per-fit result or
progress counter, and could not credibly finish within four additional hours.
That atomic attempt was stopped without changing any scientific input, split,
seed, architecture, hyperparameter, comparison, or metric. The preceding
38,448 terminal operations and 61.5499976795 GPU-hours were independently
hash-verified before the transition.

The production wrapper is now fit-checkpointed and resumable. It uses the same
scientific fitting functions and writes the legacy six-file detector package
only after all 56 fit checkpoints validate. The operational acceleration
policy, preserved audit, checkpoint identities, status stream, ceiling permits,
and benchmark reports are methodological provenance rather than scientific
outcomes or pooling inputs.

## Frozen transformer identity

The only permitted transformer is `microsoft/deberta-v3-base` at upstream
revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, staged locally at
`models/detectors/deberta_v3_base`. The runner never resolves the upstream
name over the network. It passes the absolute local directory to Transformers
with `local_files_only=true`, `trust_remote_code=false`, the slow tokenizer,
and `use_safetensors=false`.

The directory must contain these exact top-level regular files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 579 | `649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b` |
| `pytorch_model.bin` | 371146213 | `691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33` |
| `spm.model` | 2464616 | `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` |
| `tokenizer_config.json` | 52 | `3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d` |

The verifier runs before confirmatory training and again immediately before
each transformer load. It rejects a missing directory or artifact, any hash or
size mismatch, non-regular artifacts, a symlink in the required model path,
and every unlisted top-level entry. The sole exception is a real `.cache/`
directory containing local Hugging Face download metadata; it is excluded
from the artifact identity and is recursively rejected if it contains a
symlink. This exception does not add any loadable model artifact.

The binary weights remain local-only. `.gitignore` excludes
`models/detectors/`, and the offline release assembler rejects the `models`
path component and `pytorch_model.bin` by name. Do not force-add these files or
copy them into a release input.

## Confirmatory gate

Non-smoke CLI use is confirmatory. It validates
`analysis/revision_v1/detector_confirmatory_plan.json` against both the frozen
detector-config bytes and the code-enforced transformer pin. It requires
exactly `published_textcnn_equivalent` (`text_cnn`) and
`deberta_v3_base_classifier` (`pretrained_transformer`). Downloads and
`--accept-smoke-fallback` are prohibited. Any neural exception becomes a
recorded failure and a nonzero exit; it cannot produce a fallback metric.
Success additionally requires every detector/split execution to have
`implementation_status=complete` and its requested implementation kind.

The confirmatory input is closed to the complete primary detector corpus:
15,840 rows (7,920 matched RankCloak/control pairs), 480 payload groups, 18
prompt templates, three generator models, and six protocol-variant codec IDs.
The runner fails before fitting if any dimension or factor level differs. The
preprocessing manifest must declare the exact `--input` bytes, 15,840 rows,
7,920 matched pairs, and strict completion of the three primary model shards;
the resulting hash binding is embedded in the detector dataset contract and
reverified by reporting. Every payload group and every train/test partition
must contain exactly equal positive and negative row counts. The
28 required split IDs are one matched, 18 held-out-template, three
leave-one-model, and six leave-one-codec splits; two detectors therefore require
exactly 56 completed executions and zero skips. Model and codec conditions are
crossed with payloads, so those nine splits use a deterministic balanced subset
of held-out-condition payload groups for testing and only complementary payload
groups/non-held-out conditions for training. This preserves the held-out value
test while making payload sets disjoint. The split manifest records the
partition policy, excluded held-out rows, purged training rows, and row-ID
digests for audit.

The production command is owned by the confirmatory supervisor and uses the
predeclared acceleration policy:

```bash
python scripts/run_revision_detectors.py \
  --input results/revision_v1/analysis_inputs/primary_v2/detector_corpus.jsonl \
  --preprocessing-manifest results/revision_v1/analysis_inputs/primary_v2/preprocessing_output_manifest.json \
  --output-dir results/revision_v1/neural_detector/confirmatory_v2 \
  --checkpoint-dir results/revision_v1/neural_detector/confirmatory_v2.checkpoints \
  --status-file results/revision_v1/neural_detector/confirmatory_v2.status.json \
  --fit-permit-file results/revision_v1/neural_detector/confirmatory_v2.fit_permit.json \
  --fit-permit-receipt-dir results/revision_v1/neural_detector/confirmatory_v2.checkpoints/fit_permit_receipts \
  --execution-policy operations/confirmatory_v2/detector_acceleration_policy_v1.json \
  --equivalence-required-report results/revision_v1/detector_equivalence_v1/task_0/equivalence_report.json \
  --equivalence-required-report results/revision_v1/detector_equivalence_v1/task_1/equivalence_report.json \
  --device cuda:0 \
  --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf \
  --workers 1 \
  --resume
```

The supervisor also supplies the signed per-fit ceiling-permit path. Direct
production launch without that permit handshake is prohibited. Before every
fit, the runner publishes the next task identity and waits; the supervisor
recomputes actual and conservative GPU use and issues a one-use permit only if
the 165-GPU-hour hard ceiling still has the policy's full next-fit reserve.

Do not pass `--allow-model-downloads` or `--accept-smoke-fallback`. `--resume`
validates and skips only complete, identity-bound fit artifacts. A missing,
corrupt, foreign, duplicated, or symlinked checkpoint fails closed. The
compatibility `--overwrite` spelling never authorizes deletion of fit
checkpoints or unknown output bytes.

`--smoke` retains the dependency-light character n-gram fallback and labels it
`smoke_fallback`. Smoke rows are not neural evidence and are not confirmatory.

## Trained-state and run identity

Immediately after training, both TextCNN and the transformer receive a
deterministic SHA-256 over their complete state dictionaries. The
`rankcloak-torch-state-v1` stream begins with an algorithm domain tag, sorts
tensors by name, and length-prefixes each UTF-8 name, dtype, shape, and exact
contiguous CPU byte string. This makes insertion order and the device used for
serialization irrelevant while making any tensor-name, dtype, shape, or byte
change visible. Training on a different device can legitimately produce
different tensor bytes, so the acceleration gate compares state schemas and
predeclared numerical/prediction criteria in addition to requiring exact
same-device repeatability.

The digest and algorithm appear as explicit detector metric columns and inside
`implementation_metadata_json`. The v2 run manifest repeats every split/model
state identity, includes the verified transformer artifact-set identity and
confirmatory-completeness gate, pins the plan and base config hashes, and
records SHA-256 plus byte size for every other output product. The run manifest
is supervisor-published only after process absence and final GPU accounting are
established, and is self-hashed as `manifest_sha256`.

## Checkpoints, progress, and acceleration gate

Each canonical `(split_id, detector_name, seed)` task writes its metric and
ordered predictions atomically, then writes its self-hashed task manifest last.
The manifest binds the corpus, preprocessing manifest, detector config,
confirmatory plan, split row identities, source/environment snapshot,
acceleration policy, implementation identity, and child hashes. A final run
manifest is published only after exactly 56 valid task manifests have been
aggregated in the original split-outer/detector-inner order. The runner first
writes a signed finalization candidate and enters an awaiting-finalization
state. The supervisor confirms the exact PID is absent, closes the final GPU
interval, publishes the self-hashed manifest, writes a self-hashed terminal
receipt, proves pre-final ledger incorporation, and then writes the signed
terminal status. Restarts at any of those boundaries are idempotent.

The signed status file reports completed/total fits, current split, detector,
seed, elapsed time, rolling throughput and ETA, last durable checkpoint,
recovered errors, process identity, peak memory, and non-overlapping GPU
intervals. The supervisor validates that status independently, closes an
interrupted interval at the first confirmed process absence, and retains all
previous valid fits. It never relaunches the sealed GPU experiments or the
38,448 pre-detector terminal units.

The completion seal is deliberately redundant and fail-closed. The final
manifest binds the closed accounting status, finalization candidate, both
passing equivalence reports, and pre-final ledger. The terminal receipt binds
the candidate, closed status, published output, and non-overlapping intervals.
The final `status_sha256` binds the manifest, terminal receipt, and
ledger-incorporation marker; the signed ledger deduplicates the six CUDA
benchmark/equivalence sources, and the incorporation marker proves those
intervals appear in the canonical final receipt.

The predeclared policy at
`operations/confirmatory_v2/detector_acceleration_policy_v1.json` authorizes one
worker on the specified RTX 5000 Ada GPU. One worker is deliberately
conservative: the audited CPU process peaked at about 9.77 GiB and no measured
multi-worker capacity result existed before acceleration. Torch, OpenMP, MKL,
OpenBLAS, and related intra-fit thread counts remain bounded to prevent
oversubscription. The policy fixes representative task indices for TextCNN and
DeBERTa, exact same-CUDA repeatability requirements, CPU/CUDA numerical and
prediction tolerances, and conservative next-fit ceiling reserves. Benchmark
results may tighten an ETA but may not loosen those reserves or alter the
scientific matrix.

Before production resume, run the focused unit tests, verify the local-only
model artifact set, execute both representative architecture benchmarks, and
record both a same-CUDA repeatability report and a CPU/CUDA equivalence report.
The benchmark checkpoints may be reused only when their complete production
identity validates.

The offline confirmatory release retains the canonical output package, all 56
valid fit checkpoints, completed fit-permit receipts, finalization candidate
and terminal receipt, terminal status, CPU/CUDA and same-CUDA equivalence
artifacts and reports, supervisor-finalized benchmark records, signed GPU
ledger and incorporation marker, and the operational event log. It excludes
`.execution.lock`, the supervisor lock, any unconsumed active
`*.fit_permit.json`, `recovered_fit_permits` quarantine material, caches, and
all local transformer or generator model weights. Those exclusions distinguish
durable reproducibility evidence from live coordination state or unsafe model
bytes.

The upstream checkpoint is a pickle-backed `pytorch_model.bin`; artifact hashes
establish byte identity but do not make pickle intrinsically safe. Load it only
with the pinned Transformers/PyTorch environment and the documented trusted
upstream provenance. Smoke or tiny-fixture tests validate mechanics only; they
do not substitute for the two production-scale representative benchmarks or
the final 56-fit confirmatory result.
