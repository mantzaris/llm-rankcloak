# V4 contextual coherence replacement report

The requested replacement study is complete. It evaluates complete delivered messages under a fixed rubric that accepts understandable text with minor awkwardness. The primary panel estimate is 90.3% acceptance for RankCloak and 96.6% for matched greedy ordinary output. The paired difference is −6.3 percentage points with a 95% payload-cluster interval of [−8.7, −3.9]. Encoded messages are often accepted under this rubric, while ordinary output performs better on average. These are automated model judgments. They do not establish human acceptance, detection rates, semantic equivalence or indistinguishability.

The scientific source commit is `1f894eb40b4eb4d9a397031bd5532bc4f29c345a` on `main`. It contains the code, frozen plans, raw GPU responses, derived analyses, validation and rebuilt documents. This report is added in a subsequent handoff commit. The final pushed handoff SHA is reported separately rather than embedded into its own future commit. The inspected starting commit and remote were `b7198d86a2765ff92fb67f83e1f55f40b49f4528`. The initial worktree was clean, and no applicable AGENTS.md was found. No branch was created. Later remote inspection still found that starting commit before the scientific commit.

All earlier scientific runs, Stage 1–4 reports and evidence, frozen configurations, original review wording and historical release files remain unchanged. The preservation check verifies 8,940 protected tracked files outside the explicit current-document allowlist. Four substantial method/result blocks also remain byte-identical, covering rank/protocol definitions, primary recovery/capacity, ablation/detectors and transport/entropy/quantization. The only existing non-document housekeeping change ignores routine LaTeX `.fls` files. No archive assembly, upload bundle, publication metadata update, journal submission or editor contact was performed.

## Study definition and freeze

This is a new follow-up designed after the earlier fragment results and motivated by the author’s informal reading. That reading is not a participant dataset. The final protocol was frozen before this study’s final judging, not retrospectively preregistered as an original experiment.

The configuration is `configs/revision_v4/coherence_replacement.json`, SHA256 `662ed091076537ba02b1ff46f672546924856976077062b17103dfe2638c0ae9`. The final freeze is `results/revision_v4/coherence_replacement/plans/final_sample/freeze.json`, recorded at `2026-09-23T10:38:30.456988+00:00`. It hashes the exact sample, six scientific source/configuration files and already generated control caches. The selection seed is 2026092304 and bootstrap seed is 2026092305. The frozen inference, selection and analysis files were not changed after scoring.

The source records describe 8,280 separately delivered cover messages in 1,440 segmented trials and 240 payloads. The four hexadecimal classes are SHA-256, HMAC-SHA256, 96-bit nonce and 128-bit token. A segment is scored with its own authentic prompt. Rotating-topic segments are not assembled into a fictitious conversation.

Each class has 37 payloads outside the former 92-payload scored cohort. A seeded identity hash selects 12 per class, giving 48 payloads from that 148-payload stratum. Every selected payload retains three generator families and both schedules. Two message indices per trial are selected by a second hash, giving 288 trials and 576 encoded messages. Every generator, schedule and class cell contains 24 selected messages. No final scores, text quality or author example choices enter selection. Payload inclusion is 12/37 within the outside-cohort stratum and zero within the earlier scored cohort. Conditional message inclusion is 2/N for a trial with N covers. The estimand therefore targets the declared outside-cohort stratum, not an unbiased census of all 240 payloads.

All selected text is available. The sample includes 85 messages that failed the old joint window rules, comprising 5 Llama, 69 Mistral and 11 Qwen covers. There is no minimum word count or evaluator-boundary alignment exclusion. Calibration contexts/messages are disjoint from both final conditions. No claim is made that all underlying historical material was unseen during development.

The primary question is whether meaning can be followed in an ordinary exchange, allowing minor awkwardness. The complete user-specified rubric is pinned in the configuration and printed in Supplementary Note S20. It permits imperfect grammar, mild repetition and interpretable unusual wording, but disallows mentally repairing a material contradiction or breakdown. Levels 0 and 1 are acceptable. Level 2 materially interferes with following meaning, and level 3 is largely unintelligible or severely disconnected. Logical connectedness from 1 to 5 is secondary and supplies no additional acceptance gate. Short single-idea messages are judged for contextual fit and intelligibility.

Judges receive one complete message and authentic prompt as anonymous JSON data, using identical presentation in both conditions. Text, whitespace, Markdown and errors are preserved without repair. Boundaries, payloads, generator identity, condition labels, trial IDs, prior scores and author opinions are not supplied. No extended reasoning is requested. Every raw prompt, rendered chat input, response and parse result is retained.

## Controls, judges and calibration

Each ordinary control uses the exact generator and original prompt-token context, the historical safe_text_filter_v1 mask and greedy continuation policy. Eight ordinary greedy positions replace the eight forced positions, followed by the unchanged dynamic_completion_v1 heuristic with at least eight tail tokens and at most 256. Both conditions have a 264-token assigned cap. Neither adds an explicit EOS stop beyond the mask and heuristic. Early stops, cap stops, weak outputs and awkward wording are retained. Nothing is truncated to match realized length. Historical temperature-0.8/top-p-0.95 controls are not the baseline.

There are 54 distinct controls, one for each of 18 authentic prompts and three generators, reused 3 to 25 times. The 1,152 logical score slots per arm become 1,074 distinct encoded judge requests and 108 ordinary requests. No exact scoring request is shared across conditions. Exact request caching preserves every source/control mapping and inferential multiplicity. The comparison conditions on this fixed control set, rather than treating duplicated greedy outputs as independently generated controls. It compares delivered-message conditions and does not isolate a causal effect of a hidden boundary.

| Source generator | Judge families |
| --- | --- |
| Llama 3 8B Instruct | Mistral and Qwen |
| Mistral 7B Instruct v0.3 | Llama and Qwen |
| Qwen 2.5 7B Instruct | Llama and Mistral |

All use existing pinned Q4_K_M weights and embedded documented chat templates. Judge and generator effects are not fully crossed. The complete acquisition revisions, tokenizer metadata, file hashes and template hashes are in the configuration and `provenance/model_pins.json`.

| Model | Source revision | GGUF SHA256 |
| --- | --- | --- |
| Llama | `a06c33ec89c1e3402009fb47f466a89127c6d223` | `86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7` |
| Mistral | `61fd4167fff3ab01ee1cfe0da183fa27a944db48` | `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6` |
| Qwen | `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031` | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |

The existing llama-cpp-python 0.3.23 generation environment was retained. Judges use embedded chat_template.default formatting, a single user turn, deterministic temperature-zero decoding, top-p 1, top-k 0, repetition penalty 1 and JSON-schema grammar. Context allocation is 2,048 tokens, with batch and microbatch size 128. The initial output cap is 128. One same-prompt retry at 192 is allowed only for malformed or truncated responses. Valid unfavorable scores are never retried. Execution is randomized within seeded model blocks, with one GPU process at a time.

The fixed calibration set contains 18 constructed examples, six clear, six with minor imperfections and six with explicit contradictions or unrelated disruptions. Intended labels were recorded before inference. The declared sanity check requires at least four of six minor cases accepted and four of six disrupted cases rejected, with nonconstant outputs. It is an instrumentation check, not a human-validity threshold.

| Judge | Clear accepted /6 | Minor accepted /6 | Disrupted rejected /6 | Binary confusion matrix |
| --- | --- | --- | --- | --- |
| Llama | 6 | 6 | 5 | [[5, 1], [0, 12]] |
| Mistral | 6 | 6 | 5 | [[5, 1], [0, 12]] |
| Qwen | 6 | 6 | 6 | [[6, 0], [0, 12]] |

Matrix rows are intended disrupted/acceptable and columns observed disrupted/acceptable. All 54 calibration responses were valid and every model passed the stated sanity check. No rubric amendment was made. Llama accepted an explicit contradiction about attendance at the same meeting time. Mistral accepted simultaneously dry and fully soaked soil even while its explanation identified the contradiction. Both mistakes remain visible. Calibration does not guarantee consistent application or correspondence with readers.

Calibration throughput and tokenization of the prospective workload admitted the full target before final outcomes. The conservative forecast was 16,114.4 seconds, against 19,454.5 seconds available after completed calibration/control jobs and a 1,800-second reserve. There was no outcome-based reduction, enlargement or stopping of the sample.

## Observed results

The primary panel gives the two assigned judges equal weight. Selected messages are averaged within trial, generator/schedule observations within payload, and then across payloads. Complete panel estimates require both judges in both conditions. All paired intervals use 2,000 shared percentile payload-cluster draws stratified by artifact class. They condition on the fixed controls and complete paired availability. They do not treat two models as two independent human participants.

| Analysis | Complete message pairs | Encoded acceptance % [95% CI] | Ordinary acceptance % [95% CI] | Paired difference pp [95% CI] |
| --- | --- | --- | --- | --- |
| Panel overall | 537 | 90.3 [88.4, 92.1] | 96.6 [95.0, 98.1] | -6.3 [-8.7, -3.9] |
| Source Llama | 182 | 91.1 [87.0, 94.8] | 100.0 [100.0, 100.0] | -8.9 [-13.0, -5.2] |
| Source Mistral | 191 | 87.8 [84.1, 91.2] | 92.7 [88.8, 96.1] | -4.9 [-9.9, 0.3] |
| Source Qwen | 164 | 92.3 [89.1, 95.1] | 98.1 [95.9, 99.7] | -5.9 [-9.1, -2.7] |
| Rotating topics | 269 | 90.5 [88.0, 92.9] | 96.9 [95.3, 98.1] | -6.3 [-9.3, -3.5] |
| Single topic | 268 | 90.0 [87.2, 92.5] | 96.2 [93.8, 98.3] | -6.2 [-9.4, -2.9] |
| Judge Llama | 382 | 82.6 [78.6, 86.5] | 90.9 [86.7, 94.5] | -8.3 [-13.5, -2.9] |
| Judge Mistral | 352 | 92.5 [89.8, 95.0] | 100.0 [100.0, 100.0] | -7.5 [-10.2, -5.0] |
| Judge Qwen | 375 | 96.1 [93.8, 97.9] | 100.0 [100.0, 100.0] | -3.9 [-6.2, -2.1] |

The panel contains 537 complete pairs in 280 trials across all 48 selected payloads. Generator-specific complete counts are 182/192 Llama, 191/192 Mistral and 164/192 Qwen. The Qwen complete-pair subset has 88 trials and 47 observed payload clusters. The Mistral-source paired interval includes zero, which does not establish equivalence. The physical-judge rows each concern two source families, so their differences are not pure judge effects.

| Panel disruption level | Encoded % | Ordinary % |
| --- | --- | --- |
| 0 | 11.5 | 33.9 |
| 1 | 78.8 | 62.7 |
| 2 | 8.5 | 3.4 |
| 3 | 1.2 | 0.0 |

Most accepted encoded ratings are level 1 rather than level 0, consistent with the deliberately tolerant standard. Mean logical connectedness is 3.60 [3.53, 3.67] out of 5 for encoded text and 4.00 [3.94, 4.06] for ordinary text. Binary disagreement is 14.6% for encoded and 6.7% for ordinary text under the declared hierarchy. Disagreement is available for 562 and 551 messages respectively, with unweighted counts of 83 and 35. These availability denominators differ from the 537 complete panel pairs.

All 1,182 planned distinct final judge requests ran. There were 31 fixed retries and 20 invalid final responses, comprising 19 truncations and one malformed JSON response. They map to 43 unavailable logical slots, 18 encoded and 25 ordinary. Two failed ordinary requests judged by Mistral account for all 25 ordinary missing slots through reuse. Encoded missing slots are 2 for the Llama judge, 7 for Mistral and 9 for Qwen. The 39 incomplete pairs have mean encoded length 128.4 tokens, compared with 50.5 among complete pairs. Missingness is not assumed random.

Extreme assignments of all missing binary scores give encoded acceptance from 88.9% to 90.5%, ordinary acceptance from 94.8% to 97.0%, and a paired difference from −8.1 to −4.3 percentage points. These all-selected-slot sensitivity bounds are not confidence intervals. No invalid response is converted into acceptance.

Across all 576 selected messages in each condition, mean lengths are 55.8 encoded and 56.4 ordinary generator tokens, with median 28 for both and a 16–264 range. Encoded cap stops number 61 and ordinary cap stops 55. Generator-specific means are 65.6 versus 78.8 for Llama, 54.5 versus 38.6 for Mistral and 47.2 versus 51.8 for Qwen. These residual differences are reported rather than removed by cutting sentences.

Of 110 valid disrupted encoded judge slots, 104 include a literal passage and 73 quote across the recorded boundary. This mapping occurs only after scoring. A whole-message quotation may cross the boundary incidentally. It is not causal localization, and a missing quotation is not proof that no defect exists.

The predeclared smallest-identity-hash examples cover joint acceptance, binary disagreement and joint disruption among complete paired cases. Their full unedited text appears in Supplementary Note S20 and `analysis/examples.json`. The accepted Llama example retains “Garding Basic 2”. The Mistral disagreement retains “Send the emails 3+ calendar dates in advance.” The disrupted Llama example retains “Write this paragraph with the aim being at a level of 6.5 CEFR.” Source trial and segment identities are printed beside all three. They were not exchanged for fluent examples.

## GPU execution and technical checks

Total charged GPU-job wall time is 4781.415026 seconds, or 79.69 minutes, across 12 completed jobs. The ceiling was 21,600 seconds. This includes startup, model hashes, loading, inference, retries and a declared control compatibility check. There were no failed jobs. The NVIDIA RTX 5000 Ada Generation, UUID `GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf`, supplied CUDA inference. Per-job stderr records full layer offload. No unrelated process was interrupted, and no inference silently fell back to CPU.

| Phase | Jobs | Charged wall seconds | Peak observed process MiB |
| --- | --- | --- | --- |
| calibration | 3 | 188.576 | 5116 |
| control capacity | 3 | 156.995 | 5262 |
| controls | 3 | 156.919 | 5006 |
| judging | 3 | 4278.926 | 5118 |

Peak memory is sampled nvidia-smi process memory every two seconds, not an exact instantaneous allocator maximum. Job wall time is not CUDA kernel active time. Final judging executed 1,213 attempts with 736,723 API prompt tokens and 94,698 completion tokens. Calibration used 29,441 prompt and 3,402 completion tokens. The initial 54 ordinary generations emitted 2,868 tokens from 1,280 prompt tokens. The persistent ledger, CUDA stderr and detailed execution receipts are retained under the new results namespace.

A runtime audit found historical cover context allocation 4,096 versus 2,048 for new controls. All actual inputs and outputs fit both. A separate plan recorded the discrepancy, fixed all 54 control identities and required retaining every outcome. After final judging, the 54 controls were generated at 4,096. Every token ID, full text, stop reason and filter hash agreed. No study control or score was replaced. The check establishes equivalence for these requests only and is included in the time cap.

The worker also retained an unused diagnostic token count that requests an extra BOS token for Llama and Mistral. Inspection of the pinned chat handler confirms actual inference uses the already rendered template with add_bos=false. Every actual API prompt count agrees with the final freeze. The diagnostic count is conservatively one token larger, not evidence of double-BOS model input. Raw records and the frozen worker remain unchanged, with the explanation in `provenance/token_count_note.json`. The capacity worker’s load_and_hash_seconds label starts after hashing, so only supervisory wall time is used for the budget.

## Manuscript changes and preserved evidence

The new primary design is in main Methods, Intact-message contextual acceptability, pages 12–13. Main Results, Contextual acceptability of complete messages, begins on page 18, with Table 3 and Figure 5 on page 19. The abstract, introduction, discussion, limitations, cover letter and R1.4 answer now describe measured contextual acceptance under this rubric. Supplementary Note S20, pages 41–44, gives the complete rubric, calibration, full score distributions, physical judges, schedules, lengths, missingness and exact examples.

The former semantic pilots are no longer the primary coherence presentation. Their retained cosine effects are 0.0078 [−0.0416, 0.0572] and −0.0106 [−0.0728, 0.0519]. Both failed the earlier positive-lower-interval criterion. Their context-gain effects remain 2.2200 [1.7784, 2.6777] and 2.0612 [1.7378, 2.4109]. Supplementary Note S17, pages 35–39, retains those numerical results, disjointness and the single historical amendment, all exclusions, donor reuse, subgroups and adverse examples. Raw files and frozen decisions remain in `results/revision_v4/stage2/`.

The historical actual-minus-ordinary fixed-path context-gain difference remains 0.1575 [0.1098, 0.2066] nats per evaluator tail token, and actual-minus-shuffled remains 2.3265 [2.2705, 2.3912]. They are complementary local compatibility diagnostics. The original sampled-ordinary versus greedy-tail difference, 3,057 differing standalone tokenizations and generator-specific nulls remain explicit. The strong historical example still shows its internal word boundary and malformed wording. Short-fragment similarity and same-prompt substitutions did not establish a suitable measure of ordinary contextual acceptability. This motivates changing the target, without proving the old test unfair or messages universally incoherent.

Historical saved-ID recovery, visible retokenization, transformation, strict-gate and quantization endpoints remain unchanged. The supporting forced-span versus full-message likelihood figure moved from main display space to Supplementary Figure S17 in Note S21, page 45. This retains five main figures and three main tables. The method, configured inversion, detectability and transmission conclusions are stable. Neither the new rubric nor the old likelihood diagnostics establishes confidentiality or undetectability.

R1.4 now has completed automated contextual evidence, rather than a hard-coded failed-automation status inherited from the cosine assay. The current ledger records `automated_contextual_evidence_supplied_human_perception_unmeasured`. All 16 written responses are complete for author review, all 13 original quote blocks remain exact and editor judgment remains unknown. Updated public coverage of the new evidence is a separate outstanding action. The earlier immutable ZIP predates this study and was not modified.

Primary methodological sources were checked at the ACL Anthology. [G-Eval](https://aclanthology.org/2023.emnlp-main.153/) supplies an example of structured model evaluation, not validation of these Q4 local judges or a reproduction of its GPT-4 results. [van der Lee and colleagues](https://aclanthology.org/W19-8643/) support explicit and reproducible evaluation design and the separation of automated assessment from human evidence. Bibliography details and claim limits are recorded in `provenance/methodological_sources.md`.

## Validation and final files

Focused scientific tests pass, 19 in total, covering faithful extraction, outcome-independent selection, label blinding, matched controls, parsing, missingness, clustered aggregation and request/cache identity. Source audits join all 576 exact texts, token IDs and authentic prompts to retained records, check every control contract and historical filter mask, and verify all 2,304 blinded logical mappings. Primary means and all three intervals were independently recomputed from retained score classifications. Nineteen scientific output files reproduced byte-for-byte offline without model execution or downloads.

Current validation confirms frozen source hashes, exact quotations, 16 evidence-linked answers, actual compiled locations, CUDA offload, one process at a time and the cumulative budget. Numerical checks reconcile the abstract, results, table, cover and response with retained estimates, distributions and missingness. The original tools and Stage 1–4 logs were not rerun into their historical output locations.

All four final PDFs build without warnings or undefined references. The new clean dependency check compiles main4_submission.tex with its local style dependencies in a fresh temporary directory, producing 24 pages. Its compiled bibliography and editable tables are retained. No upload bundle was created. All 80 PDF pages were scanned for blank pages, stale markers and out-of-page text. All 79 changed or reflowed pages were visually inspected, including every main page, supplement pages 2–45, all response pages and the cover letter. The final response was checked again after fixing an obsolete section name and orphaned headings. The unchanged first supplementary page retains the prior visual review. New figures and tables are readable, with no observed clipping.

The initial scientific build caught a TeX shortstack line-break syntax error, which was corrected without changing results. Its failed log remains. Final successful logs are in `validation/latex/scientific_02`, `correspondence_02` and `submission_01`. Ordinary code/prose whitespace checks pass. Frozen CSV CRLF, raw tool-output spaces and final blank lines in generated TeX remain preserved and documented rather than normalized as scientific changes.

The title has 14 words and the abstract 192 whitespace-delimited words. Main prose has 4,720 TeXcount text words under the explicit convention excluding Abstract, Methods, captions, headings, availability and references. Complete source text has 10,428 words, excluding captions and headings. There are 49 main references. Three algorithm floats are reported separately from the five figures and three tables. The main paper remains 25 pages without reducing fonts. The supplement is 45 pages with 17 figures and 39 tables.

| Final document | Pages | SHA256 |
| --- | --- | --- |
| [main4.pdf](../paperV4/scientific_reports/main4.pdf) | 25 | `e56ffcaaf02bb534e1504897e0c87152b6e9e001240edd8e371d89083ca982bc` |
| [supplementary4.pdf](../paperV4/scientific_reports/supplementary4.pdf) | 45 | `8c672d70932b37e7a79732300844fca51ed7edb82e7d20fecaf1df56c096ba5a` |
| [response_to_reviewers_v4.pdf](../paperV4/response/response_to_reviewers_v4.pdf) | 9 | `cf48b8cdd762969b6e1545e4243c4c331ba016b7c87ccfe7f608a3b0469ebca6` |
| [cover_letter_v4.pdf](../paperV4/cover_letter/cover_letter_v4.pdf) | 1 | `5f0435fdcc5ec2da69aabdb902c618baeb1e6d2b558488c1309cad0074d071d1` |

The editable [main4_submission.tex](../paperV4/scientific_reports/main4_submission.tex) has SHA256 `9d2f36be6b83a814a666252fb5ebdcdedbcd46e8b8dbaf192dc2a713d2047c93`. Response page references name this coherence-replacement author-review edition and stable section/table/figure identifiers, rather than claiming identical pagination in a journal portal.

New code separates instrumentation and selection (`rankcloak/revision_v4_contextual_coherence.py`), inference and GPU supervision (`run_revision_v4_contextual_coherence.py`, `run_revision_v4_contextual_job.py`), prospective freeze, offline analysis, source/document audits, result rendering, response rendering and current builds. The full 190-file scientific commit gives the exact changed-file list through `git show --stat 1f894eb40b4eb4d9a397031bd5532bc4f29c345a`. Most new files are retained plans, raw responses, execution records and validation rather than manuscript prose.

Offline reproduction uses the existing analysis environment

```bash
.venv/bin/python -m scripts.analyze_revision_v4_contextual_coherence --calibration
.venv/bin/python -m scripts.analyze_revision_v4_contextual_coherence --plan final_sample
.venv/bin/python -m scripts.audit_revision_v4_contextual_coherence
.venv/bin/python -m scripts.render_revision_v4_contextual_results
.venv/bin/python -m scripts.audit_revision_v4_contextual_documents
.venv/bin/python -m scripts.validate_revision_v4_contextual_coherence
```

Do not rerun the prospective freezer after outcomes exist. Raw model requests and scores already suffice for table reconstruction. The full new results are under `results/revision_v4/coherence_replacement/`, including `analysis/summary.json`, `manuscript/claim_evidence.json`, `response_status.json`, `gpu/ledger.json` and `validation/` receipts.

The bounded research and document task is complete. The remaining author actions are review of these new claims and responses, then a separately scoped current archive/deposit update and final journal upload preparation. The old manual publication handoff must not be used as coverage of the new study. Human perception remains unmeasured and the editor decides whether the supplied automated evidence addresses R1.4. These are submission decisions, not an unfinished open-ended GPU study.
