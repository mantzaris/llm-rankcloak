# Revision V3 computational plan

Status date: 2026-08-31. This plan was written before V3 detector fitting or
outcome inspection. It governs computational work only; `paperV2/` and any
manuscript V3 directory are read-only.

## Audit baseline

- Repository baseline: commit `5cdb22d3cb02480fb090946ef176325d3d107752`;
  the working tree was clean at audit start and no repository-scoped
  `AGENTS.md` exists.
- Authoritative primary inputs are
  `results/revision_v1/analysis_inputs/primary_v2/trials.csv` (6,480
  RankCloak trials; SHA-256
  `12b11bad2c7b6468d7f8e4ed12fde2a0ab29996cda8594d0cbcaa6b06797ae7b`)
  and `detector_corpus.jsonl` (15,840 rows; SHA-256
  `0688cc5904128582cdfad20bc1dbd9a6153dca4edf1a2e6ef1c54aeeb8fbfb76`).
  Raw authoritative records are the three
  `results/revision_v1/primary_v2/*/records.jsonl` files.
- The primary detector corpus has 7,920 RankCloak/control pairs, 480 payload
  groups, three Q4_K_M generator families, 18 templates, six protocol/codec
  IDs, and eight artifact classes. The payload representations are direct
  subword, ASCII-byte fixed radix (B=8 or B=16), and raw hex nibble (B=16);
  the six detector codec IDs additionally distinguish nonsegmented and the two
  segmented topic schedules.
- The previous detector design used payload-grouped train/test partitions but
  did not remove or co-partition all normalized near duplicates before fitting.
  Its declared 0.95 character-ngram audit found 53 pairs, including 44
  cross-payload pairs and one normalized exact pair, affecting 16 frozen
  splits. V3 therefore rebuilds partitions before fitting and does not relabel
  the prior sensitivity analysis as strict deduplication.
- The strongest existing text-only detector is the locally pinned
  `microsoft/deberta-v3-base` classifier; the TextCNN-equivalent is retained as
  the comparison architecture. Per-token source-model log probabilities exist
  for both RankCloak and ordinary controls, permitting a fair model-aware
  surprisal-trace classifier without exposing label-only forced-rank metadata.
- Live hardware at audit: NVIDIA RTX 5000 Ada Generation (32,760 MiB, 32,220
  MiB free) plus Quadro T2000 (4,096 MiB); 55 GiB disk free. The current
  project `.venv` has PyTorch 2.5.1+cu124 and Transformers 4.46.3 but lacks
  `llama-cpp-python`. The three exact primary GGUF paths are absent. A nearby
  Llama Q4 file is not a substitute: its SHA-256 is
  `8ba9baf3a7345f705a11878397500fb25174034f0fd784e83aa4a96aaa47735f`,
  not the pinned primary hash
  `86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7`.
  No repository remote-compute launcher was found; no paid compute will be
  started.

## A/F: deduplicated detector design and low-FPR endpoints

1. Normalize visible text with Unicode NFKC, casefolding, and collapse of all
   Unicode whitespace to one ASCII space, then strip. Record both raw-text and
   normalized-text SHA-256 values.
2. Exact-deduplicate before detector feature extraction. If either member of a
   matched pair would be removed, remove the complete pair so labels remain
   paired and balanced. Select canonical pairs deterministically by sorted
   `pair_id`, then `row_id`.
3. Detect near duplicates on the post-exact-deduplication corpus using
   lowercase character-within-word-boundary TF-IDF 3--5-grams, `min_df=2`, L2
   normalization, float32 storage, and exhaustive cosine neighbor search. The
   predeclared threshold is cosine similarity >= 0.95. This is lexical, not a
   claim of semantic equivalence.
4. Form connected components over exact/near-duplicate links, existing payload
   groups, and matched pairs. The connected component (`split_group_id`) is the
   indivisible partition/bootstrap unit, while `payload_group_id` remains in
   all outputs.
5. Build deterministic 60/20/20 train/validation/test partitions from complete
   components, balancing row and factor counts by a seeded greedy assignment.
   Leave-one-model-family-out partitions exclude the held-out family from both
   training and validation and use disjoint components for testing. The seed is
   `20260831`.
6. Fail closed on overlap of row IDs, pair IDs, payload IDs, normalized hashes,
   near-duplicate clusters, or split-group IDs. Emit row-level assignments,
   pair/cluster ledgers, factor counts, exclusions, and a machine-readable
   leakage audit.
7. Fit hyperparameters and all thresholds using training/validation only.
   Low-FPR thresholds are the least conservative validation-negative score
   cutoffs whose exact empirical false-positive counts do not exceed the
   target; ties are moved above with `nextafter`, never interpolated. One
   percent is available only with at least 100 validation and 100 test
   negatives; 0.1% requires at least 1,000 in each. Otherwise the endpoint is
   explicitly unavailable.
8. Report ROC-AUC; empirical step-function partial ROC area on FPR [0, 0.01]
   divided by 0.01 (maximum 1; no optimistic score interpolation); frozen
   validation-threshold TPR and achieved FPR; exact positive/negative counts;
   and 95% connected-component bootstrap intervals (2,000 resamples). The
   unit of resampling and unavailable-resolution reasons accompany every row.

## B: human-authored secondary controls

Use Databricks Dolly 15k version 1.0 at immutable revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. The registered JSONL is 13,085,339
bytes with SHA-256
`2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`,
licensed CC BY-SA 3.0. The official card says Databricks employees authored the
records and were instructed not to use generative AI. The source was selected
because authorship, immutable bytes, and license are clear; it is not presented
as a participant evaluation or a perfectly paired corpus.

Exclude nonempty-context rows and the brainstorming, classification, closed
QA, information-extraction, and summarization categories. Apply the same V3
normalization/deduplication pipeline. Retain automated quality/safety screens
from the existing human-control code, then deterministically match eligible
responses without replacement to the RankCloak test length distribution within
available prompt categories/templates. Preserve source IDs and hashes but do
not commit licensed source text. Human false-positive rates use thresholds
selected only on matched LLM validation data. Report available strata and
missing genre coverage; interpret RankCloak-versus-human discrimination as
potentially including generic machine-versus-human signal. Do not pool human
and LLM controls unless a separately labeled descriptive analysis is justified.

## C/E: unseen families and adaptive threat models

- Text-only/data-adaptive attacker: the existing DeBERTa-v3-base classifier,
  trained specifically on V3-deduplicated RankCloak versus matched clean LLM
  rows. Retain TextCNN-equivalent as the published-comparator architecture.
- Model-aware attacker: regularized logistic regression over features common
  to both classes and available to an attacker with the exact source generator
  and prompt: token count plus distributional, tail-frequency, run-length, and
  positional summaries of per-token surprisal. Standardization, imputation,
  regularization strength, and the decision threshold are fitted on
  train/validation only. Forced-rank or segment-role fields that exist only for
  RankCloak are excluded from this classifier and reported only as descriptive
  process diagnostics.
- Run matched and all three leave-one-family-out evaluations on V3 partitions.
  Report each family separately and a payload-component bootstrap macro summary;
  codec/artifact breakdowns are emitted only where both labels and adequate
  group counts exist. These tests do not isolate quantization because the
  historical families all used Q4_K_M.

## G: entropy-gated RankCloak design

The implementation will add a reusable protocol primitive with no side-channel
gate metadata. At each step, compute Shannon entropy in bits from the same
filtered next-token distribution used for rank selection. If `H >= tau`, emit
the next requested rank and consume one payload symbol. Otherwise emit the
deterministic filtered rank-1 token and do not advance the payload index. The
decoder recomputes entropy before each observed token, consumes a rank only at
eligible positions, and appends every observed token to context. Equality at
the threshold is eligible. Gate-disabled mode must be byte-for-byte equivalent
to ordinary `generate_rank_span`/`recover_rank_span` behavior.

Thresholds are per-model quantiles from a clean, ordinary-generation
development trace fixed before evaluation: ungated, median entropy (moderate),
and 75th-percentile entropy (strict). Evaluation uses the predeclared payload
indices 3 and 41, deterministically mapped one per selected template, and never
uses calibration traces as evaluation payloads. Fixed-payload generation stops
after all symbols or at `min(context allowance, 6 * ungated forced length)`;
fixed-token-budget outcomes are derived from the first ungated-length
gated-span tokens and report embedded fraction.

The planned balanced matrix is three model families x 20 applicable
artifact-class/codec cells (direct and ASCII-B16 for all eight classes; hex
nibble for four eligible classes) x two predeclared templates
(`explain_everyday_system`, `professional_project_update`) x three gate levels:
360 RankCloak analysis trials. One length/prompt-matched ordinary control per
output adds 360 control trials. The dry-run audit found that V2 assigned the
predeclared payload indices 3 and 41 to different prompt templates, so none of
the 120 ungated RankCloak rows or 120 controls is exactly prompt-compatible.
They therefore cannot be reused. The matrix requires 120 new ungated and 240
new gated RankCloak generations plus 360 new length-matched controls, as well as
18 clean calibration traces (three models x six prompt categories).
Fixed-budget rows are derived, not additional generations. Expected new
evaluation generations: 720; expected result size below 250 MiB; rough RTX 5000
Ada estimate 2--6 GPU-hours and under 16 GiB VRAM based on prior Q4 timings,
with the upper range allowing skipped-token overhead. These are planning
estimates, not observed results.

The exact Q4 files and CUDA `llama-cpp-python==0.3.23` environment are currently
absent, so full entropy generation is unavailable until the pinned artifacts
are restored. Code, fake-model tests, dry-run plans, and explicit acquisition
instructions will still be delivered. No differently hashed model will be
substituted in reported evidence.

## D: matched quantization sensitivity design

Use Qwen2.5-7B-Instruct at upstream revision
`a09a35458c702b33eeacc393d103063234e8bc28` and quantization-package revision
`8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, comparing Q4_K_M with Q8_0 under
the same embedded tokenizer and llama.cpp implementation. Select the two fixed
bounded codecs ASCII-B8 and ASCII-B16 for all 480 payloads; this avoids changing
payload-side direct-subword ranks between quantizations. Reuse the 960 exact
Q4 RankCloak trials and 960 matched controls already present; generate 960 Q8
RankCloak trials and 960 Q8 controls. Use the frozen payload train/validation/test
indices (36/12/12 per class) and evaluate Q4->Q8, Q8->Q4, and pooled-quantization
training with payload-disjoint tests. Paired diagnostics include recovery,
rank/entropy/surprisal changes, token-ID and ordering divergence, detector
endpoints, and unavailable cases. Test negatives total 192 per direction, so
1% is resolvable and 0.1% is not.

The Q4 file is 4,683,074,240 bytes (known pinned SHA-256
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`);
the pinned-revision Q8_0 file is 8,098,525,888 bytes with SHA-256
`9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b`.
The exact pair totals 12,781,600,128 bytes. Acquiring these weights is a large
download, so the run is stopped before acquisition under the review protocol.
The handoff records exact `huggingface-cli download` commands and requires
post-download SHA-256 verification before execution.

## H: topic-conditioned cover variability

Reuse the 720 paired same-model/same-payload hex rows for
`segmented_hex_single_topic` versus `segmented_hex_multi_topic`. Report exact
output uniqueness, normalized character edit distance, whitespace-token
Jaccard overlap, and length ratio, with 2,000-resample payload-group intervals.
Recovery mode and exact recovery are separate columns. Illustrative pairs are
selected by the smallest SHA-256 of the stable pair identity, never by quality
or diversity. This supports only topic-conditioned cover variability; it does
not address the saved-token replay limitation or establish key secrecy.

## Outputs and verification

All generated files live under `results/revision_v3/`; code/configs/tests use
new V3 names. A single build command regenerates tables, LaTeX, figures, prose
handoffs, claim-evidence mappings, manifests, and checksums from authoritative
CSV/JSON inputs. Tests cover normalization, clustering, group-safe three-way
and held-out splits, empirical low-FPR resolution, entropy-gate boundaries and
payload accounting, trace features, human selection, topic metrics, manifest
validation, and deterministic regeneration. Run focused tests, the relevant
full pytest suite, smoke/dry runs, leakage and threshold audits, figure visual
inspection, `git diff --check`, and V2 hash verification before committing.
