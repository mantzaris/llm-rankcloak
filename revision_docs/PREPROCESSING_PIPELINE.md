# Revision-v1 preprocessing pipeline

`rankcloak.revision_preprocess` is the deterministic boundary between the
model-backed runner and all statistical, detector, figure, and table code. It
reads saved artifacts only. It never loads an LLM, changes a runner record, or
fills in a measurement that the runner did not record.

## Inputs and integrity checks

Each `--run-dir` must be one leaf stage/model directory containing:

- `records.jsonl`
- `plan.jsonl`
- `run_identity.json`
- `model_manifest.json`
- `source_manifest.json`
- `payload_manifest.json`

`events.jsonl`, `hardware_manifest.json`, and
`input_results_manifest.json` are consumed and hashed when present. A
robustness input manifest is also checked against the external source files it
names.

Before writing output, preprocessing verifies:

1. The self-hash of `run_identity.json`.
2. Ordered planned work count and ID hash against `plan.jsonl`.
3. The payload-manifest byte hash, payload count, unique names, and every plan
   payload reference.
4. The source-manifest file-list hash.
5. Exact binding of `model_manifest.json` into the run identity, including the
   runner's recorded full model SHA-256 and byte-size verification.
6. Stage/model/evidence identity and the one-model-per-shard rule.
7. Unique durable `(work_id, attempt_index)` values, one completion at most,
   valid record types, and—by default—a completion for every planned unit.
8. Cross-shard config, corpus, and source-snapshot consistency and globally
   unique emitted work IDs.
9. Control/source metadata and exact length joins, canonical-primary reference
   joins, and robustness source-record hashes.
10. The evidence firewall: smoke or limited exploratory data cannot be pooled
    with confirmatory or secondary data.

`--allow-incomplete` exists only for diagnosing interrupted runs. Final
confirmatory preprocessing must use the strict default.

## Outputs

All products are written atomically, and any existing target causes the run to
fail. Re-running requires a new empty output directory. The fixed products
are:

- `trials.csv`: one payload-condition/replay/transformation analysis row.
  RankCloak rows include each executed replay. Robustness and resolved
  reference rows retain their own condition identities. Representation,
  artifact, and serialized bit estimands remain separate. Controls and
  provenance-only paraphrase transforms are not given fictitious recovery outcomes.
- `features.csv`: nested segment/view text rows. Each RankCloak segment has a
  `forced_span` and `full_message` row; controls retain the matched view;
  robustness rows retain transformed observed text. `segment_index` remains
  explicit so downstream code collapses within a payload trial rather than
  treating segments as independent.
- `runtime.csv`: one row per completed work unit plus rows for recorded
  model-load and shard-memory sessions. Encoding, decoding, filter, generation,
  representation/artifact/serialized throughput, cumulative per-trial process
  RSS, sampled shard RSS, and sampled selected-device memory retain their stated
  runner estimands. Unrecorded memory remains null with an explicit availability
  label; the one-second device sample is not represented as a kernel-exact peak.
- `failures.csv`: every execution attempt failure and every failed scientific
  replay. Scientific failures preserve first differing position, expected and
  recovered token IDs and ranks, context SHA-256, boundary offsets, lengths,
  segment, replay mode, transformation, and failure category. Operational
  failures explicitly mark divergence fields as not applicable.
- `detector_corpus.jsonl`: two raw-text rows per matched control pair, labels 1
  (RankCloak) and 0 (ordinary control). Both share `pair_id` and
  `payload_group_id`; the latter is the payload name, preventing variants of
  one payload from crossing detector splits. It directly satisfies
  `revision_detection.py`'s canonical column contract.
- `unavailable.csv`: one row per completed `condition_unavailable` or
  `dependent_unavailable` work unit, including the frozen reason, safe/stable
  vocabulary counts, model/tokenizer identity, source hash, and root-condition
  provenance. These rows count toward design completion and display but are
  excluded from `trials.csv`, `features.csv`, detector pairs, failure outcomes,
  and every scientific estimand.
- `preprocessing_input_manifest.json`: hashes and identities for every emitted
  and reference shard plus join counts.
- `preprocessing_output_manifest.json`: hashes, byte sizes, row counts, and
  non-imputation invariants for all products.

CSV nulls are intentionally empty numeric cells. Companion availability fields
state why measurements are absent. This lets pandas and the statistical code
treat them as missing rather than mistaking a textual sentinel for a number.

## CLI

One completed shard:

```bash
.venv/bin/python scripts/preprocess_revision_results.py \
  --run-dir results/revision_v1/primary/llama3_8b_instruct_q4_k_m \
  --output-dir results/revision_v1/analysis_inputs/primary_llama
```

Multiple stage/model shards can be supplied by repeating `--run-dir`. When an
ablation canonical reference or robustness result depends on a source shard
that should not itself be emitted, repeat `--reference-run-dir`:

```bash
.venv/bin/python scripts/preprocess_revision_results.py \
  --run-dir results/revision_v1/robustness/llama3_8b_instruct_q4_k_m \
  --reference-run-dir results/revision_v1/primary/llama3_8b_instruct_q4_k_m \
  --reference-run-dir results/revision_v1/ablation/llama3_8b_instruct_q4_k_m \
  --reference-run-dir results/revision_v1/robustness/qwen2_5_7b_instruct_q4_k_m \
  --output-dir results/revision_v1/analysis_inputs/robustness_llama
```

The Python API is:

```python
from rankcloak.revision_preprocess import preprocess_revision_results

artifacts = preprocess_revision_results(
    run_dirs=["results/revision_v1/primary/llama3_8b_instruct_q4_k_m"],
    reference_run_dirs=[],
    output_dir="results/revision_v1/analysis_inputs/primary_llama",
)
```

The resulting `trials.csv`, `features.csv`, and `runtime.csv` can be passed
directly to `scripts/run_revision_statistics.py`. The detector JSONL can be
passed directly to `scripts/run_revision_detectors.py`.

## Explicit adapter boundary

The accepted nested shapes are the runner-v1 record types
`rankcloak_trial`, `ordinary_control`, `canonical_primary_reference`,
`robustness_transform`, `robustness_decode`, `robustness_reference`,
`condition_unavailable`, and `dependent_unavailable`.
`robustness_transform` is retained and hash-linked as paraphrase provenance
and runtime only, never counted as a recovery outcome. Unknown record types or
missing mandatory nested fields fail with a specific error. If the robustness
runner schema changes, this adapter and its fixture tests must be updated
explicitly; it does not guess field aliases or silently drop new structures.
