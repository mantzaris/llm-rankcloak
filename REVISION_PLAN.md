# RankCloak Scientific Reports Major-Revision Plan

## Scope and immutable-source rule

This plan covers the editor decision and the reviewer reports supplied for the
Scientific Reports manuscript. The revision will be a surgical major revision of
RankCloak, not a replacement paper. The submitted sources remain immutable:

- `.paper/scientific_reports/main.tex`
- `.paper/scientific_reports/supplementary.tex`

All revised text will be written to:

- `.paper/scientific_reports/main2.tex`
- `.paper/scientific_reports/supplementary2.tex`
- `.paper/scientific_reports/response_letter2.tex`

No confirmatory experiment may write into an existing result directory. The active
root is `results/revision_v1/`, driven by a frozen, machine-readable plan and public
seed manifest. Superseding stages have disjoint names and work identities:
`smoke_v3`, `primary_v2`, `ablation_v2`, `multilingual_v2`, and `robustness_v2`.
The release candidate will be tagged and archived in a DOI-assigning repository
before the revision is submitted; no public deposit or release has been authorized.

## Evidence classes and current audit status

| Evidence class | Location | Status and permitted use |
| --- | --- | --- |
| Submitted partial pilot | `results/rankcloak_paper_main_pilot/` | Exploratory only: 20 non-segmented and 7 segmented rows; 26/27 exact recoveries. It is the source of the submitted numerical results. |
| Earlier development pilots | Other existing `results/rankcloak_*pilot*` and sweep directories | Exploratory method-development evidence only; never pool into confirmatory inference. |
| Later rank-safe single-model matrix | `results/rankcloak_paper_gpu_main_rank_safe/` | Exploratory/pre-revision validation only: 475 non-segmented and 75 segmented rows; 550/550 exact recoveries. It post-dates the submitted partial pilot and is not the new multi-model confirmatory study. |
| Invalidated pre-remediation primary shard | `results/revision_v1/primary/qwen2_5_7b_instruct_q4_k_m/` | Stopped after 234/4,800 rows and invalidated in full. Preserve in place; never resume, select, relabel, or pool. The external invalidation manifest self-hash is `a9836f60344c38568f4dbc014deb6c428b1bfad216f9a55da683edd978f9168c`. |
| Tokenizer preflight v2 | `results/revision_v1/tokenizer_preflight_v2/` | Vocabulary-only engineering evidence: 1,530/1,530 payload/prompt checks passed under all three pinned tokenizers; no generation was performed. Not a scientific result. |
| Replacement smoke-v3 | `results/revision_v1/smoke_v3/` | Exploratory engineering evidence only: 96/96 work items completed, 93 available outcomes and three declared Mistral unavailability rows, zero execution failures. Never pool with confirmatory inference. |
| Revision confirmatory study | `results/revision_v1/{primary_v2,ablation_v2,multilingual_v2,robustness_v2}/` | **Not started.** The frozen compute gate is `no_go_over_budget` (56.017 point; 158.508 conservative GPU-hours versus the approved 150-hour ceiling). No `primary_v2` shard exists. |
| Human evaluation | `human_study/` | Materials, power analysis, randomization, selection, and analysis scaffolds prepared; no recruitment, usability pilot, exposure, payment, or ratings. UCF determination and separate authorization remain external gates. |

The submitted lead-in failure must not be described as demonstrated channel or
boundary fragility. Audit of the historical implementation shows that generation
evaluated its eight lead-in tokens serially whereas recovery evaluated them as one
batch. The later serial-replay/rank-safe code recovered 25/25 lead-in trials. The old
row will be retained as a legacy implementation-schedule failure, and the scientific
claim will come from a new, designed lead-in/replay/retokenization study.

The direct-subword incident also exposed that token-ID replay alone was an
insufficient recovery endpoint. The replacement contract tokenizes literal UTF-8
payload bytes without special tokens, permits only recorded reversible ASCII-space
prefix bytes, removes a prompt prefix only when it is the actual BOS, and distinguishes
`exact_representation_recovery` from `exact_payload_recovery`. The compatibility
field `exact_recovery` is now the original serialized-payload byte/SHA-256 endpoint
(`original_serialized_payload_bytes_sha256_v1`), not equality to an already
transformed target-token sequence.

## Current execution gate (2026-08-09)

The immutable tokenizer preflight manifest has self-hash
`b61eaaed4086124774ae4cca37261a91a5c643d575a89f8dc9c33f505bdb1bec`.
The completed replacement smoke and held-out-evaluator timings feed
`results/revision_v1/compute_projection_v2.json`, projection SHA-256
`685da8dfb81f082eda18893f7681462d9f3fec6216f8c978a8cf92ba8b7b90ad`.
The point projection is 56.017000 GPU-hours and the conservative projection is
158.508216 GPU-hours, 8.508216 above the approved ceiling. Under the prespecified
rule the decision is `no_go_over_budget`; therefore expensive confirmatory execution
is paused before `primary_v2`, not partially underway. Smoke-v3 values may support
software validation and rate projection only and must not fill manuscript result
placeholders.

## Current manuscript compliance baseline

Counts use `tools/count_scientific_reports.py` and its documented LaTeX counting
policy. They will be regenerated on every manuscript build.

The current Scientific Reports submission page requires a title of at most 20 words,
an Abstract of at most 200 words, at most 4,500 main-text words excluding Abstract,
Methods, References, and figure legends, no more than eight figures plus tables, and
legends of at most 350 words. A revised LaTeX or Word manuscript is single-column;
final figures are supplied separately. Supplementary Information is a separate PDF
below 50 MB, uses Supplementary Figure/Table S numbering, and every item must be cited
from the manuscript ([Scientific Reports submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines)).

| Item | Submitted source | Synchronized revision scaffold | Revision gate |
| --- | ---: | ---: | ---: |
| Title | 11 words | 10 words | at most 20 |
| Abstract | 186 words | 195 words | at most 200 |
| Introduction | 1,000 words | 854 words | 850--925 target |
| Results | 2,713 words | 457 words | 2,450--2,600 final target |
| Discussion plus limitations | 637 words | 702 words | 700--825 target |
| Journal-defined counted main text | 4,350 words | 2,013 words | target 4,000--4,350; hard maximum 4,500 |
| Conservative main text including table text | 4,863 words | 2,091 words | monitor and reduce |
| Methods | 2,785 words | 2,357 words | preferred 3,000--3,600 after adding essential methods |
| Supplementary Information | 4,946 words | 4,245 words | one separate coherent PDF under 50 MB |
| Main displays | 7 (4 figures, 3 tables) | 7 (5 figures, 2 tables) | at most 8; internal plan 7 |
| Isolated source build | 19 manuscript pages; 12 SI pages | 14 manuscript pages; 14 SI pages; 13 response pages | manuscript length remains subject to final consolidation |

The synchronized revision values are working-draft counts, not evidence that the final
article is already short enough: the Results remain generated-result placeholders.
Every insertion must rerun the counter. Complete parameters and secondary results
belong in Supplementary Information rather than being used to expand the main Results.

The five revision figure legends contain 47, 48, 43, 48, and 34 words (220 total),
all below the 350-word per-figure limit. The submitted-source builds succeed, but the archived
submission-copy PDFs are not byte- or extracted-text-identical to fresh builds from the
current TeX sources. The current sources and named archived PDFs must therefore be
retained together as separate audit inputs rather than described as a byte-exact source
reproduction.

## Display and result provenance gaps to close

- Historical aggregate tables recompute exactly from the submitted-pilot CSV/JSONL
  files, and the standard statistics/detector products can be regenerated.
- The four final high-resolution/composite images used by the manuscript have no
  committed generator or 300-dpi assembly path. Their numerical inputs are traceable,
  but the final raster assets are orphaned/manual.
- Main and Supplementary tables are hand-entered rather than emitted by analysis code.
- The historical pilot manifest represents a later staged pass and does not prove the
  exact environment of every accumulated generation row; it also lacks a model hash
  and physical backend identity.
- The later rank-safe manifest hashes the model but records a dirty worktree, so it does
  not identify the exact source bytes used. It also omits GPU name/UUID/driver and a
  separately identified tokenizer revision (the tokenizer is embedded in the GGUF).
- The notes mention a 3,578-check audit, but no executable audit script or
  machine-readable report is present.

The revision must generate every final figure and LaTeX table from immutable data and
record source archive hash, model and tokenizer artifact hashes, prompt/filter hashes,
per-result checksums, hardware/backend identity, and the complete invocation.

## Editor and reviewer comment tracker

Status meanings: **implemented** refers to frozen code, materials, or prose, not to a
completed result; **blocked** identifies an unexecuted computational estimand under the
current no-go decision; **external gate** requires an institutional, recruitment,
payment, or repository action. Exploratory smoke never counts as confirmatory status.

| ID | Comment or required response | Planned change or experiment | Main-paper location | Complete evidence location | Status | Final response-letter location |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | Accurately report results, remove overstatement, and explain limitations. | Replace partial-pilot claims with frozen confirmatory results; correct the lead-in diagnosis; narrow practical, naturalness, detectability, and robustness language. | Abstract; Results; Discussion; Responsible use and limitations | Supplementary Methods; failure taxonomy; legacy-pilot note | Prose scaffold implemented; confirmatory results blocked at compute gate | `response_letter2.tex`, Editor response E1 |
| E2 | Expand corpus and add statistical analysis. | Generate 8 real-artifact classes with 60 independent public-seed instances each; predefine mixed-effects models and grouped bootstrap intervals. | Methods; Fig. 1; Tables 1--2 | Tables S3--S6; full coefficient and sensitivity tables | Corpus and analysis pipeline implemented; estimates blocked | Editor response E2 |
| E3 | Test multiple architectures, prompt categories, and realistic payloads. | Three open model families, six prompt categories with three templates each, balanced assignments, and standard cryptographic algorithms. | Methods; Fig. 2; Table 1 | Tables S2--S6; Figs. S2--S3 and S13 | Models/prompts frozen; preflight and smoke-v3 complete; primary-v2 not launched | Editor response E3 |
| E4 | Add human readability, stronger neural steganalysis, ablations, failure analysis, and overhead. | Blinded powered human study; CNN/graph and transformer detectors; filter/tail/lead-in/replay ablations; structured first-divergence logs; timing and memory instrumentation. | Figs. 3--5; Table 2 | Tables S7--S13; Figs. S4--S12 | Computational pipelines implemented but blocked; human study externally gated | Editor response E4 |
| E5 | Examine restrictive exact-copy replay before claiming practical use. | Formal exact-recovery proposition; saved-token versus retokenized-text replay; channel transformations; bounded canonicalization/round-trip-stability mitigation; restrained applicability claims. | Methods; Fig. 4; Discussion | Tables S8--S9; Figs. S5 and S7 | Methods/execution paths implemented; robustness evidence blocked | Editor response E5 |
| E6 | Deposit bespoke code/pipeline in a DOI-assigning repository. | Prepare versioned code/data/config/environment release, mint DOI, cite exact version in Code Availability. | Code Availability | Reproduction instructions and release manifest | Offline packaging implemented; public deposit remains external gate | Editor response E6 |
| R1.1 | Exact-copy replay is too restrictive; test minor formatting disruption or mitigation. | Transformation matrix plus deterministic canonicalization and round-trip-stable-token mitigation; explain autoregressive error cascades and limits. | Fig. 4; Methods; Discussion | Table S9; Fig. S7; failure records | Implemented; confirmatory execution blocked | Reviewer 1 response 1 |
| R1.2 | Two examples per payload class are insufficient. | Eight classes times 60 payloads (480 total), with payload as the experimental unit. | Table 1; Methods | Table S3 and released corpus manifest | Corpus implemented; primary evidence blocked | Reviewer 1 response 2 |
| R1.3 | Log probability is not human-perceived naturalness. | Blinded ratings of grammaticality, fluency, coherence, topic adherence, completeness, naturalness, and suspiciousness; automated metrics remain secondary. | Fig. 3 | Table S10; Figs. S8--S9; questionnaire | Materials prepared; recruitment and ratings remain external gates | Reviewer 1 response 3 |
| R1.4 | Isolate deterministic-token-filter impact. | No filter versus current safe-text filter versus cumulative round-trip-stable filter, paired on payload/model/prompt. | Brief result in Fig. 3 or Results if material | Table S7; Fig. S4 | Implemented; smoke exposed declared Mistral unavailability; contrasts blocked | Reviewer 1 response 4 |
| R1.5 | Diagnose lead-in failure over multiple lengths. | Lengths 0, 2, 4, 8, 16, and 32; saved-ID, serial replay, and text-retokenization modes; first-divergence records. Correct old batch-schedule diagnosis. | Fig. 4 | Table S8; Fig. S5; legacy failed example | Historical diagnosis corrected; sweep implemented; execution blocked | Reviewer 1 response 5 |
| R1.6 | Fixed tail caps may cause unnatural endings; test dynamic stopping. | No tail, submitted fixed policy, and deterministic completeness-aware stopping with a declared safety cap and threshold fixed on development data. | Concise Methods and Fig. 3 | Table S7; Fig. S6 | Implemented; confirmatory execution blocked | Reviewer 1 response 6 |
| R1.7 | Preliminary detector is insufficient; add state-of-the-art neural steganalysis. | Retain feature-only diagnostic; add a published raw-text CNN/graph approach and a pretrained transformer classifier, with grouped held-out tests. | Fig. 5 | Table S11; Figs. S10--S11 | Detector pipelines implemented; primary detector corpus unavailable | Reviewer 1 response 7 |
| R1.8 | Single-topic versus multi-topic difference lacks statistical validation. | Balanced prompt categories and paired segmented schedule comparison; mixed-effects estimate with multiplicity control. | Fig. 2 or compact Results contrast | Tables S5--S7; Fig. S6 | Assignment/model specification implemented; estimate blocked | Reviewer 1 response 8 |
| R1.9 | Formalize the capacity-quality relation. | Add \(n_B=\lceil H/\log_2 B\rceil\), \(R_B\), effective rate, \(Q_B\), bounds, excess surprisal \(\Delta_B\), and exact-recovery proposition. | Fig. 1; Methods | Supplementary Equations/Methods; empirical validation | Theory and validation pipeline implemented; confirmatory validation blocked | Reviewer 1 response 9 |
| R1.10 | Ciphertext-like Base64 is not real encryption output. | AES-256-GCM and ChaCha20-Poly1305 ciphertext+tag outputs from public deterministic test vectors; also real HMAC and Ed25519 outputs. | Corpus paragraph; Table 1 | Table S3; public test vectors and hashes | Implemented and hash-validated | Reviewer 1 response 10 |
| R1.11 | A single model does not establish generality. | Current Llama 3 8B Instruct plus pinned Qwen 2.5 7B Instruct and Mistral 7B Instruct v0.3. | Fig. 2; Table 1 | Tables S2 and S4; Fig. S3 | Models pinned/license-audited; preflight/smoke complete; primary evidence blocked | Reviewer 1 response 11 |
| R1.12 | Encoding/decoding overhead is not reported. | Separate model-load time from amortized encoding/decoding; record wall time, tokens/s, bits/s, peak RAM/VRAM, filter/tail cost, and cover tokens/payload byte. | Table 2 | Table S12; Fig. S12 | Instrumentation and smoke timing complete; confirmatory profiling blocked | Reviewer 1 response 12 |
| R1.13 | Too many preprints/technical reports. | Replace preprints with peer-reviewed versions where available; retain Calgacus as the source method and explain remaining provenance sources. | Introduction and References | Reference-audit note in response | Reference audit and revision bibliography implemented | Reviewer 1 response 13 |
| R2.1 | Limited corpus, model families, multilingual behavior, and deployment settings. | Expanded corpus and three-model English primary study; computational multilingual secondary study; transformations as deployment stress tests. | Fig. 2; Fig. 4; Discussion | Tables S4, S6, S9; Figs. S7 and S13 | Infrastructure implemented; confirmatory/multilingual execution blocked | Reviewer 2 response 1 |
| R2.2a | Human perception and stronger steganalysis are missing. | Same human and detector studies as R1.3 and R1.7, with forced-span/full-message distinction preserved. | Figs. 3 and 5 | Tables S10--S11; Figs. S8--S11 | Human study externally gated; detector corpus blocked | Reviewer 2 response 2a |
| R2.2b | Compare with existing linguistic steganography; clarify Ding et al. connection. | Direct Calgacus is the required rank baseline; attempt one compatible peer-reviewed comparator; cite Ding et al. only as broader representation learning. | Introduction; primary comparison in Fig. 2 if compatible | Comparator details and complete audit outputs | Compatibility audits complete; no fair numeric comparator available | Reviewer 2 response 2b |

The user confirmed that the supplied Reviewer 2 text is the complete report available
for this revision. This tracker therefore maps every supplied concern plus the explicit
multilingual, prior-method, and DOI requirements. No additional or unseen comments are
assumed.

## Frozen confirmatory experiment matrix

The unit of inference is the independent payload, never an individual segment. Prompt
templates and payload identifiers are grouped/repeated factors; segments are nested
within payload. Assignments were generated before model execution and checked for
balance by payload class, model, codec, prompt category, and template. The runner
materializes exactly `primary_v2`, `ablation_v2`, `multilingual_v2`, and
`robustness_v2`; these plans remain unexecuted while the compute gate is no-go.

### Corpus, prompts, and primary multi-model panel

The frozen corpus uses 60 rather than the minimum 50 payloads per class so that the study balances
exactly over six prompt categories and permits a 36/12/12 payload-level
train/validation/test split per class.

- 480 payloads: 8 classes x 60 independent instances.
- Classes: SHA-256, HMAC-SHA-256, 96-bit nonce, 128-bit token, UUIDv4,
  AES-256-GCM ciphertext+tag, ChaCha20-Poly1305 ciphertext+tag, and Ed25519
  signature. The first four classes (240 payloads) are hex-eligible.
- Three model families: the current Llama artifact plus pinned Qwen and Mistral
  artifacts of comparable scale and quantization.
- Eighteen original English prompts: six categories x three templates. For each
  model and payload class, ten payloads are assigned to each category and the
  templates rotate so every template receives 80 model-payload blocks, evenly split
  between hex and non-hex classes. Codecs remain paired within a
  model-payload-prompt block.
- Direct subword is an actual cover-generation/recovery condition, not only a
  rank-pressure diagnostic. Its literal-UTF-8/no-special-token contract and
  original-byte recovery endpoint are frozen in the payload-fidelity-v2 plan.
- Ordinary controls are independent, seeded, prompt- and length-matched generations;
  repeated greedy controls are avoided because they can make detector classification
  artificially easy.

| Primary condition | Arithmetic | RankCloak covers/recoveries |
| --- | ---: | ---: |
| Direct Calgacus-style subword | 480 payloads x 3 models | 1,440 |
| ASCII \(B=8\) | 480 x 3 | 1,440 |
| ASCII \(B=16\) | 480 x 3 | 1,440 |
| Hex-nibble \(B=16\) | 240 eligible payloads x 3 | 720 |
| Segmented single-topic | 240 x 3 | 720 |
| Segmented multi-topic | 240 x 3 | 720 |
| **Primary RankCloak total** | | **6,480** |

The primary study adds 6,480 full-message ordinary controls plus 1,440
forced-span-length controls for the segmented conditions: 7,920 controls, 14,400
generated texts total, and 1,440 derived forced-span views that are not new
generations.

### Ablation, replay, multilingual, and comparator panels

The ablation plan uses 48 hex payloads (12 per eligible class) across the three models, giving 144
paired blocks for the one-factor-at-a-time ablations. The canonical condition is
multi-topic, segment size 8, lead-in 0, current safe filter, and dynamic tail.

| Factor | Levels | Condition rows |
| --- | --- | ---: |
| Filter | none, current safe, round-trip-stable | 432 |
| Lead-in | 0, 2, 4, 8, 16, 32 tokens | 864 |
| Tail | none, current fixed, dynamic completeness | 432 |
| Segment size | 4, 8, 16, 32 ranks | 576 |

The shared canonical row makes 1,872 unique ablation rows. Of those, 144 already
exist in the primary matrix, so the ablation panel adds 1,728 generated RankCloak
texts/recoveries.

The same 144 blocks support:

- Three replay modes on lead-in-8 covers: saved token IDs, detokenized text followed
  by retokenization, and greedy lead-in regeneration.
- Thirteen raw channel conditions: unmodified, line-ending conversion,
  leading/trailing whitespace trimming, whitespace collapse, Unicode normalization,
  quote conversion, platform/Markdown copy-paste simulation, single-character
  insertion/deletion/substitution, token deletion, truncation, and deterministic
  paraphrase.
- The limited mitigation (round-trip-stable filtering plus deterministic
  canonicalization/framing) on seven formatting-preserving conditions.
- Two wrong-model decodes for every block.

Together these yield 3,600 robustness outcome rows and 3,168 additional decode-only
executions, with no additional source-cover generation. Every failure records first
divergence, expected/recovered token IDs and ranks, context hash, boundary offsets,
transformation, and failure category.

The Supplementary multilingual panel uses 48 payloads x 3 models x Spanish and
Mandarin x direct/ASCII-\(B=16\): 576 RankCloak texts plus 576 controls. No
multilingual naturalness claim is made without native-speaker assessment.

Direct Calgacus is the required primary prior-method comparator. Compatibility audits
of patient-Huffman and DAIRstega did not identify an official implementation that can
be matched fairly on generator, prompt, tokenizer, payload, stopping, and recovery
conditions. Consequently the frozen plan contains no second construction panel and
will document the incompatibility rather than report a misleading numeric baseline.

### Experiment totals

| Quantity | Frozen design |
| --- | ---: |
| Generated texts, including 144 paraphrase transformations | 17,424 |
| RankCloak paired encode/decode trials | 8,784 |
| Additional robustness decodes | 3,168 |
| Total RankCloak decoding executions | 11,952 |
| Neural detector fits | 56 (2 architectures x 28 grouped splits) |

The primary detector corpus is 6,480 full RankCloak messages versus 6,480 matched
ordinary controls, with a separate 1,440-versus-1,440 forced-span task. Payload-level
splits are 36/12/12 per class (288/96/96 payloads and 3,888/1,296/1,296 primary
RankCloak messages). Each detector has 28 frozen splits: one grouped matched split,
18 held-out-template splits, three leave-one-model-out splits, and six
leave-one-codec-out splits.

Implementation clarification (no change to the frozen design): the confirmatory
detector input combines the full-message and forced-span matched pairs, yielding
15,840 rows (7,920 per label). Detector `codec_id` denotes the six prespecified
protocol variants rather than four collapsed representation implementations. Because
model and all-payload codec conditions are crossed with payload, their held-out tests
use deterministic balanced test payload groups and complementary training payload
groups; no payload crosses a boundary, the held-out value is absent from training,
and all 28 split identities must be present before fitting.

### Statistical analysis gates

- Exact recovery: logistic mixed-effects model when estimable; Wilson intervals and a
  prespecified separation-safe sensitivity analysis if a condition has no failures.
- Human ratings: cumulative-link mixed model with message/payload and rater effects.
- Continuous quality: robust linear mixed model plus payload-grouped bootstrap.
- Artifact counts: negative-binomial model unless a prespecified dispersion check
  supports Poisson.
- Detector metrics: payload-grouped bootstrap confidence intervals; splits grouped by
  payload and template, with matched, held-out-payload, held-out-template,
  cross-model, and cross-codec tests.
- Primary effects, contrasts, original-payload endpoint semantics, and multiplicity
  families are frozen before confirmatory output; Holm adjustment applies within
  primary outcome families.
- Primary inference is accepted only from the locked R mixed-model run. The held-out
  evaluator outcome enters through a hash-checked, source-record-verified join covering
  all 6,480 primary RankCloak trials. Generic Python pooled/pairwise effects remain
  descriptive and cannot substitute when the R output is absent or unidentified.

## Main-paper display allocation (7 total)

| Display | Role | Replaces or consolidates |
| --- | --- | --- |
| Figure 1 | Payload representation plus theoretical capacity-quality framework | Revises submitted representation figure and absorbs formal rate bounds. |
| Figure 2 | Primary multi-model recovery and capacity-quality frontier, including direct and bounded conditions | Replaces submitted capacity-quality figure and most variant narration. |
| Figure 3 | Human naturalness/suspiciousness and forced-span versus full-message quality | Replaces submitted forced/full figure; absorbs the essential filter/tail result. |
| Figure 4 | Replay fragility, retokenization, lead-in sweep, and limited mitigation | Replaces the unsupported single-failure narrative; detailed transformations move to SI. |
| Figure 5 | Strongest neural steganalysis results in matched and held-out settings | Replaces the feature-only pilot diagnostic. |
| Table 1 | Compact study design and primary recovery summary | Replaces submitted payload and protocol summary tables. |
| Table 2 | Primary effect sizes and representative computational performance | Consolidates inferential and overhead results. |

The submitted tail-topic figure and worked-example table move to Supplementary
Information. No eighth display will be added without replacing or consolidating one of
the seven above.

## Supplementary Information allocation

### Planned tables

1. Table S1: protocol and variant definitions.
2. Table S2: models, tokenizers, exact revisions/hashes, quantization, software, and hardware.
3. Table S3: payload classes, algorithms, lengths, public seeds/test vectors, and counts.
4. Table S4: complete recovery matrix with confidence intervals.
5. Table S5: full mixed-effects coefficients, diagnostics, and sensitivity analyses.
6. Table S6: prompt-category and multilingual results.
7. Table S7: filter, tail, and segment-size ablations.
8. Table S8: lead-in sweep, replay modes, and first-divergence taxonomy.
9. Table S9: full transmission-transformation matrix.
10. Table S10: human evaluation, exclusions, rating distributions, reliability, and power analysis.
11. Table S11: detector metrics, splits, calibration, and hyperparameters.
12. Table S12: per-model/per-codec timing, throughput, RAM, and VRAM.
13. Table S13: failure taxonomy and representative cases.

### Planned figures

1. Fig. S1: experiment flow and completed matrix.
2. Fig. S2: direct-rank distributions by class and model.
3. Fig. S3: per-model capacity-quality frontiers.
4. Fig. S4: filter ablation.
5. Fig. S5: lead-in length and first-divergence positions.
6. Fig. S6: tail policy, segment size, and topic schedule.
7. Fig. S7: transmission transformations and mitigation.
8. Fig. S8: human rating distributions.
9. Fig. S9: automated readability and held-out evaluator metrics.
10. Fig. S10: detector ROC and precision-recall curves.
11. Fig. S11: held-out model, prompt, payload, and codec detection.
12. Fig. S12: computational overhead.
13. Fig. S13: multilingual secondary study.

The legacy protocol definitions, exact-copy audit, complete segmented example, failed
lead-in row, and pilot diagnostics will be retained only where they explain method
development. They will be explicitly labeled exploratory and renumbered consistently.
Every retained Supplementary item must be cited from `main2.tex`.

## Computational, storage, and human-study estimate

The original 90--150 GPU-hour planning reservation has been superseded by the
manifest-verified smoke-v3 projection. The conservative column, not the point column,
controls authorization.

| Component | Point GPU-hours | Conservative GPU-hours |
| --- | ---: | ---: |
| `primary_v2` | 25.475365 | 82.337546 |
| `ablation_v2` | 5.071071 | 17.354701 |
| `multilingual_v2` | 2.172055 | 6.839401 |
| `robustness_v2` | 2.579539 | 11.870830 |
| Held-out evaluator | 19.386767 | 38.773535 |
| Observed smoke/evaluator, invalidated shard, and legacy charges | 1.332203 | 1.332203 |
| CPU-configured neural detectors | 0.000000 | 0.000000 |
| **Total** | **56.017000** | **158.508216** |
| **Headroom under approved 150 hours** | **93.983000** | **-8.508216** |

The three Q4 artifacts occupy approximately 14 GB. The approved working reserve
remains 50--100 GB; the DOI package excludes third-party model weights. No additional
GPU stage may begin while the gate remains `no_go_over_budget`.

The human-evaluation scaffold retains eight conditions: human control, ordinary LLM,
direct Calgacus, ASCII \(B=8\), ASCII \(B=16\), hex-nibble, segmented forced span,
and the corresponding full dynamic-tail message. Initial simulation did not justify
80\% power for the smallest planning effect under the proposed three-rating design.
The smallest scientifically important effect, multiplicity family, exclusions,
messages per condition, ratings per message, and final sample size must therefore be
refrozen with statistical and UCF review. No cost is authorized: there has been no
usability pilot, recruitment, exposure, payment, or rating collection.

## Frozen decisions and unresolved external gates

The corpus, all three model artifacts and licenses, English prompts and balanced
assignment, direct payload/BOS contract, seeded controls, filters, segment/topic/tail
policies, transformations, Spanish/Mandarin panel, detector specifications,
statistical contrasts, multiplicity, payload-level resampling, Reviewer 2 scope, and
seven-display allocation are approved and frozen. Comparator audits are complete and
support documenting incompatibility rather than forcing an unmatched number.

Only the following decisions remain open:

1. **GPU authorization.** The conservative projection is 158.508216 GPU-hours.
   Resume only after explicit approval of a sufficient ceiling or an explicitly
   approved/refrozen reduced design that passes a new projection.
2. **Human activity.** UCF determination, a refrozen adequately powered sample,
   population/platform/control coverage, recruitment, pilot, and spending require
   separate authorization. None has occurred.
3. **Public archive.** DOI repository credentials and the exact validated package
   must be approved before any public deposit. No DOI has been minted.
4. **Final prose/results.** Do not replace any generated-result hook until disjoint
   confirmatory outputs and the cross-document audit exist. Retain the current title
   unless completed evidence requires narrowing.

## Final quality gates

- Abstract at most 200 words; title at most 20 words.
- Journal-defined main text at most 4,500 words and preferably at most 4,350,
  excluding Abstract, Methods, References, and figure legends.
- Single-column LaTeX manuscript with no more than 7 planned main displays
  (absolute journal maximum 8); each figure legend at most 350 words and final
  figure files supplied separately.
- Exactly one separately submitted coherent Supplementary PDF under 50 MB; use
  Supplementary Figure/Table S numbering and cite every item from the main text.
- Original `main.tex`, `supplementary.tex`, and both archived submitted PDFs remain
  byte-identical to their protected hashes.
- Pilot and confirmatory data remain physically and analytically separate.
- The invalidated Qwen shard remains immutable and wholly excluded; no primary-v2
  or supporting confirmatory stage begins without a passing authorized compute gate.
- Exact recovery means original serialized-payload byte/SHA-256 equality;
  representation replay remains a separate diagnostic and the compatibility alias
  must agree with the payload endpoint.
- Every confirmatory value comes from immutable machine-readable files and generated tables/figures.
- Payload, prompt template, and nested segment structure are respected in inference.
- Human recruitment begins only after the required UCF determination.
- Manuscript, SI, response letter, and DOI release pass a single cross-document value audit.
- The DOI resolves to the exact code/data release named in Code Availability.
