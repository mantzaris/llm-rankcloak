# Revision-v1 model runner

`scripts/run_revision_matrix.py` is the model-backed entry point for the frozen Scientific Reports revision matrix. One process loads exactly one pinned GGUF, verifies configuration, corpus, model SHA-256, source results, and transformation dependencies, then binds the ordered plan to an immutable run identity.

## Frozen stage arithmetic

| Stage | RankCloak generation | Ordinary controls | Other generation | Decode/reference outcomes | Total |
|---|---:|---:|---:|---:|---:|
| `primary` | 6,480 | 7,920 | 0 | 0 | 14,400 |
| `ablation` | 1,728 | 0 | 0 | 144 references | 1,872 |
| `multilingual` | 576 | 576 | 0 | 0 | 1,152 |
| `smoke` | 36 | 60 | 0 | 0 | 96 |
| `smoke_v2` | 36 planned | 60 planned | 0 | 0 | 96 planned |
| `robustness` | 0 | 0 | 144 paraphrases | 3,168 decodes + 432 references | 3,744 |

Primary full-message controls match observed full token count. Segmented primary conditions also receive a forced-span-length control, accounting for the extra 1,440 controls. Ablation reference rows point to immutable primary outputs and are never regenerated.

The robustness total must not be described as 3,600 executions. It contains 3,600 outcome rows: 3,168 additional recovery executions and 432 canonical references. Separately, 144 deterministic Qwen paraphrase generations produce immutable transformation artifacts.

The 96-unit smoke plan has 12 RankCloak paths per model and 20 source-length-dependent controls per model. It covers every protocol; all 18 English prompts collectively; all three filter states; lead-ins of 0, 8, and 32; no, fixed, sentence, and dynamic tails; both topic schedules; and saved-ID, text-retokenized, and greedy-lead-in replay. Its RankCloak rows cover all eight payload classes. Every model includes UUID, the longest Ed25519 Base64 artifact, and other non-hex paths; the four hex classes rotate through hex/segmented paths. This avoids a biased SHA-only compute projection.

Smoke evidence is labeled `exploratory_smoke_not_for_confirmatory_pooling`. Every `--limit` subset gets disjoint work IDs and `exploratory_limited_not_for_confirmatory_pooling`. Neither can join confirmatory data by frozen ID.

`smoke_v2` repeats the balanced design under disjoint work IDs and the separate
default path `results/revision_v1/smoke_v2/<model-id>/`. Smoke v1 is preserved
unchanged. V2 freezes the pre-confirmatory filter-feasibility rule learned
during the tokenizer audit; it remains exploratory and is never pooled with
confirmatory evidence.

## Dry-run and execution

Inspect a plan without loading a model:

```bash
.venv/bin/python scripts/run_revision_matrix.py --stage primary --dry-run
.venv/bin/python scripts/run_revision_matrix.py --stage robustness --dry-run
```

Execute one model/GPU shard using the full UUID from `nvidia-smi -L`:

```bash
.venv/bin/python scripts/run_revision_matrix.py \
  --stage smoke_v2 \
  --model qwen2_5_7b_instruct_q4_k_m \
  --gpu-uuid GPU-REPLACE-WITH-EXACT-UUID \
  --context 4096
```

Resume the identical plan with `--resume`. CPU validation requires `--n-gpu-layers 0` and no GPU UUID. Default paths are `results/revision_v1/<stage>/<model-id>/`; smoke and limited runs use separate labels. A custom `--output-dir` does not alter evidence status or identity.

## Execution semantics

- Direct Calgacus ranks are computed in the source/BOS context and inverse-transcoded after recovery. ASCII B=8, ASCII B=16, and raw hex-nibble codecs use deterministic inverse codecs.
- Generation and recovery evaluate serially. Serial lead-ins replace the historical serial-versus-batched replay implementation error.
- Single-topic segments reuse the assigned prompt. Multi-topic segments rotate from the assigned category modulo six while retaining the assigned template index.
- Segments persist lead-in, forced, and tail IDs, equal-length token-role masks, context hashes, and timing. Filter masks are content-addressed and serialized once per model/filter.
- Saved token-ID replay is the supported exact-copy condition. `detokenized_text_retokenized` and `greedy_leadin_regeneration` are diagnostics and are not silently substituted.
- Confirmatory primary, ablation, and multilingual generation executes saved-token replay only; smoke exercises all replay paths.
- Controls use serial NumPy PCG64 sampling at temperature 0.8 and top-p 0.95. SHA-256 seeds bind source trial and full/forced view. BOS/EOS/EOT are excluded, so exact source-length matching has no early termination.
- Before any `roundtrip_stable_filter_v1` generation, at least one token must
  satisfy both `safe_text=true` and isolated detokenize-to-retokenize exactness.
  An empty mask produces a durable completed `condition_unavailable` row with
  safe/stable counts and pinned model/tokenizer identity. No prefix-conditioned
  frame or other post-hoc substitute is permitted. Dependent controls and
  robustness outcomes become completed `dependent_unavailable` rows; neither
  type is an exact-recovery failure or an executable observation.

## Robustness execution order

Robustness consumes completed immutable primary and ablation outputs. Run the Qwen robustness shard to completion first because it creates all 144 severe-paraphrase text artifacts, including transformations of Llama and Mistral source covers:

```bash
.venv/bin/python scripts/run_revision_matrix.py \
  --stage robustness \
  --model qwen2_5_7b_instruct_q4_k_m \
  --gpu-uuid GPU-REPLACE-WITH-EXACT-UUID \
  --primary-results-root results/revision_v1/primary \
  --ablation-results-root results/revision_v1/ablation
```

Then run the Llama and Mistral shards with `--robustness-results-root results/revision_v1/robustness`. The input manifest content-addresses source records, Qwen records, and run identities. A paraphrase artifact stores Qwen IDs for audit, but its text is retokenized by the original source model in a separate recovery unit. Qwen IDs are never interpreted by another tokenizer. Wrong-model inverse ranking occurs only in the explicit cross-model family.

Positioned edits use the full SHA-256 integer modulo eligible positions. Seed material is the UTF-8 source trial ID immediately followed by the UTF-8 transformation ID, matching frozen `trial_id || transformation_id`. Character deletion is restricted to non-whitespace, substitution to alphanumeric characters, and token deletion excludes first/final tokens. Each segment records resolved zero-based position, eligible count, and seed digest.

## Capacity and quality traces

Results distinguish:

- `artifact_bit_length`: cryptographic artifact size;
- `serialized_payload_bits`: displayed payload bytes times eight;
- `representation_source_bits`: the bounded theory estimand, four bits per hex character or eight bits per original ASCII byte.

Direct subword transcoding has no fixed B, so bounded-theory H/rate and rank-B endpoints are unavailable. It still reports observed artifact and serialized bits per cover token. Every bounded forced context stores realized rank/log probability, admissible rank-1 token/log probability, and rank-B token/log probability under identical logits and filter. Direct paths store realized and rank-1 traces with null rank-B arrays.

## Immutable artifacts and resume

Before loading a model, the runner writes immutable `plan.jsonl`, `run_identity.json`, `payload_manifest.json`, `model_manifest.json`, `source_manifest.json`, `runtime_manifest.json`, `hardware_manifest.json`, and—when needed—`input_results_manifest.json`. Results append as canonical JSONL with `fsync` before atomic checkpoint updates. Resume reconciles the safe append-before-checkpoint crash window, rejects duplicate attempts, and rejects checkpoint-only completions. Scientific recovery failures remain completed outcomes with structured first divergence; runtime exceptions are failed attempts, and three consecutive errors stop a shard.

Planned ablation arithmetic remains 1,872 unique rows and 1,728 planned new
generations. The executable-generation count is model-dependent and is
reported from immutable completions as planned generation rows minus
`condition_unavailable` rows. Robustness retains every planned outcome-row
identity; unavailable sources propagate explicitly instead of reducing or
silently redefining the plan.

## Profiling contract

Model-load duration is a separate event. Each RankCloak row stores representation, filter setup, generation, supported recovery, inverse transcoding, diagnostic replay, throughput, cover tokens per display byte, and process RSS high-water values. Segment timings and role counts support paired segmentation/tail overhead contrasts; timer subtraction is not claimed to isolate GPU kernels.

A one-second background sampler records current/peak process RSS and selected-GPU `memory.used` through the model session. VRAM is total selected-device use and may include co-tenants; it is a sampled high-water mark, not a kernel-exact allocator peak. Hardware, driver, backend, polling interval, sample count, and this limitation are persisted.
