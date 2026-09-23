# Scientific Reports V4 Stage 2 report

Stage 2 execution and integration are complete. The frozen exploratory sample and bounded transport checks finished within the compute ceiling. Semantic validation and final V4 archive coverage remain unresolved, so the manuscript is not submission-ready.

## Scope and preserved state

The inspected starting commit was `b964b938a0b2578501edf60555da8eff5ed4eddd` on `main`. The only initial local item was the author's untracked `paperV4/RankCloak_V4_Stage2_Plan.md`. It was read and preserved unchanged. No applicable AGENTS.md was present. No branch was created. The prior submission, historical configurations and results, original review file, and archived Stage 1 evidence were preserved. The initial state records 8,416 protected Git blobs and hashes the author plan and requests file. It is retained in `results/revision_v4/stage2/provenance/initial_state.json`.

The requested Stage 2 work uses `configs/revision_v4/stage2*` and `results/revision_v4/stage2`. The create-only Stage 1 scaffolder was not rerun. No dependency environment was replaced, paid compute acquired, public release created, Zenodo version published or journal submission made. Current working V4 documents remain conspicuous drafts with genuine unresolved items.

## Boundary design, pilot decisions and sample

The retained source reconciles 8,280 structural boundaries, 1,440 segmented trials and 240 payload identities. The segmented population covers four hexadecimal classes, each with 60 payloads. The broader ordinary-control corpus spans eight classes. Every structural boundary remains recorded, including unavailable cases.

The initial prospective configuration freezes exact byte alignment, tokenization, left/right windows, short cases, three-word eligibility, pseudo-boundary offsets, right-length matching, exact-prompt donors, reuse, identical strings and joint three-arm eligibility. Full-prefix detokenization supplies UTF-8 offsets. Each evaluator tokenizes its own joint visible window. Boundary-straddling tokens are excluded. A tokenization-only preflight identified Mistral's single-space rendering prefix. Its unscored inventory was retained, and the explicit offset adjustment was frozen before any scores.

The semantic model is the locally cached purpose-trained `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Eleven required files total 91,578,367 bytes. Exact weight, config, tokenizer and license hashes, Apache 2.0 licensing, runtime dependencies, attention-mask mean pooling, inclusion of nonpadding special tokens, L2 normalization, float32 inference, 256-token truncation and batch size 64 are pinned. Downloaded bytes were zero. No fitted detector checkpoint was used.

The second primary metric is mean conditional context gain on the tail. It subtracts the prompt-only reference from the prompt-plus-left likelihood using the same evaluator tail IDs and denominator. The Llama-to-Qwen, Qwen-to-Mistral and Mistral-to-Llama evaluator map is unchanged. Raw conditional likelihood is secondary. Standalone-right tokenization differences are recorded rather than silently changing the reference path.

Both pilots used at most 36 hash-selected identities, covering all 36 generator, schedule and prompt-category cells. Only ordinary and same-prompt shuffled ordinary windows were scored. No RankCloak metric scores were computed or inspected for metric selection. Intervals use 2,000 payload-cluster bootstrap draws stratified by artifact class. The prospective requirement was a positive lower 95% interval for ordinary-minus-shuffled separation on both primary metrics.

| Pilot | Recipient clusters | Semantic cosine effect and 95% interval | Context gain in nats/token and 95% interval |
| --- | --- | --- | --- |
| Initial, right cap 32 | 36 | 0.0078 [−0.0416, 0.0572] | 2.2200 [1.7784, 2.6777] |
| Fresh amended, right cap 16 | 35 | −0.0106 [−0.0728, 0.0519] | 2.0612 [1.7378, 2.4109] |

The initial pilot charged 222.04 GPU-job seconds, including its failed invocation, and evaluated 6,770 LM tokens plus 1,852 semantic tokens. The fresh pilot charged 175.58 seconds and evaluated 5,930 LM tokens plus 1,454 semantic tokens. Each pilot made 144 new LM requests. Maximum sampled process memory was 4,876 MiB initially and 4,814 MiB for the amendment. The LM batch size remained one after the frozen numerical fixture failed for all evaluators. Embedding batches of up to 64 passed their numerical fixture. Costs and raw timing records are retained in the GPU ledger and per-pilot execution files.

The sole predeclared amendment shortened the right cap to 16, preserving the encoder, pooling, threshold and likelihood definition. Fresh controls exclude all 69 initial recipient-or-donor payloads and are disjoint in both roles. The amended ordinary-donor pool retains that exclusion in the full study. Shuffled RankCloak donors follow their separate frozen same-prompt rule. All pilot left windows contain eight generator tokens, no MiniLM input was truncated, and no exact-identical right pair remained. Short fragments and same-prompt topic overlap limit interpretation, but their causal contribution to failed separation is not established. No further model, layer, window or threshold search followed. The two-metric coherence requirement was not met. R1.4 remains partly unresolved.

The amended inventory has 7,133 eligible boundaries and 1,147 unique exclusions. Overlapping reasons comprise 384 actual, 539 ordinary and 238 shuffled evaluator-boundary straddles, plus 124 short actual windows. Four exact-identical actual donor candidates and 38 pilot donor candidates were skipped during construction. No cross-topic replacement was used. The detailed exclusion table reports generator, schedule, artifact class and selection status.

A conservative forecast for all eligible payloads was about 55,512 GPU-job seconds. The frozen feasible sample instead selects 23 payloads in each of four classes by SHA256 of `seed|payload_name`, using seed 20260922. It retains all three models and both schedules for every selected payload. The 92-payload structural sample contains 3,174 boundaries and 552 planned trials. Joint eligibility leaves 2,750 boundaries and 550 scorable trials, with two trials unavailable in full. These create 8,250 arm-specific units and 14,775 unique conditional/reference requests, of which 14,735 required new execution after pilot reuse. Exact caching preserves every inferential unit mapping. All selected arm windows contain eight generator tokens on the left and eight to sixteen on the right. The 32-token left cap does not imply that longer context was available.

The selected forecast was 23,843.21 seconds, based on the slower measured per-token rate from either pilot for each evaluator, a 1.75 multiplier and per-model loading allowance. The freeze reserved 1,800 seconds for small jobs and 1,800 for safety. Forecasts are admission estimates, not execution measurements or guaranteed precision. The final plan was frozen before RankCloak scoring. The study uses only the validated context-gain metric and is explicitly exploratory. It does not silently substitute a one-metric result for completed two-metric validation.

Within the selected sample, Llama contributes 1,015 eligible boundaries and 43 exclusions, Mistral 717 and 341, and Qwen 1,018 and 40. Each generator has 1,058 structural boundaries split equally between schedules. This variation limits extrapolation to excluded boundaries. The compact six-cell table and the full class/reason table retain that distinction.

Every planned scoring request completed. The final join contains 8,250 units, 2,750 eligible boundaries, 550 trials and 92 payload clusters. No convenience subset of completed rows was used. The primary paired effects and the secondary raw conditional-likelihood effects are below. Units are nats per evaluator tail token. Intervals use 2,000 recipient-payload draws within artifact class and condition on the fixed control corpus.

| Contrast | Context gain and 95% interval | Conditional likelihood and 95% interval |
| --- | --- | --- |
| actual minus ordinary | 0.1575 [0.1098, 0.2066] | 0.0619 [0.0277, 0.0981] |
| actual minus shuffled | 2.3265 [2.2705, 2.3912] | 2.3257 [2.2619, 2.3905] |
| ordinary minus shuffled | 2.1690 [2.1185, 2.2215] | 2.2638 [2.2024, 2.3295] |

These results support local conditional compatibility under the selected evaluator map. They do not establish sentence-level meaning or reader-perceived naturalness. The ordinary-control comparison includes a policy difference. All 2,398 selected ordinary source identities use temperature 0.8 and top-p 0.95 sampling, while RankCloak uses greedy tails. The comparison does not isolate a causal effect of the forced boundary. Generator-specific raw conditional-likelihood differences against ordinary controls include zero for Mistral at −0.0089 [−0.0840, 0.0682] and Qwen at 0.0322 [−0.0184, 0.0863]. These nulls remain in the main text and supplementary model table.

Ordinary donors comprise 411 payloads and 2,398 source identities, with maximum payload reuse 18 and source reuse 2. Shuffled donors comprise 240 payloads and 2,354 source identities, with maxima 32 and 4. Removing each donor payload in turn gives context-gain point estimates from 0.1504 to 0.1643 against ordinary controls and from 2.3210 to 2.3363 against shuffled tails. These are influence ranges, not confidence intervals for new donor populations.

Standalone-right tokenization differs from the retained joint-tail path in 3,057 of 8,250 units. Each conditional/reference pair still scores identical evaluator IDs and uses the same denominator. Evaluator tail lengths span 4 to 34 tokens. The predeclared weak, typical and strong examples are the minimum, upper-middle and maximum actual context gains, with identity-hash tie breaks. Their values are −0.2986, 1.8772 and 7.7527. Exact source IDs, byte ranges and boundary markers are preserved. The strongest example includes an internal `over|much` word boundary and malformed wording, illustrating why its metric label is not a reader rating.

The complete joined records, effects, donor support, donor influence, examples and figure are in `results/revision_v4/stage2/analysis/coherence_study`. The continuation-policy source audit is `analysis/continuation_policy_audit.json`.

The complete frozen-plan hash list is in `results/revision_v4/stage2/provenance/frozen_plan_hashes.json`. The executed study freeze SHA256 is `78df3c0d72d28b9105806725d653435bd8875b708be08e9dec70185b9025f4ad`.

## Transport mechanisms and configuration compatibility

Five identity-hash-selected sources cover one executed unchanged-visible-text success and failure for Llama and Qwen, and one failure for Mistral. The Mistral success stratum is unavailable. The historical raw and limited unmodified rows both reference `saved_token_id_replay.exact_recovery` with no new decode. They are retained as references, not counted as executed visible-text or mitigation successes. Actual unchanged-byte historical transformation replays supply the diagnostic baseline labels.

Twenty-five logical cover arms contain 135 constituent segment instances. Exact input matching reuses 53 historical segment requests and leaves 26 missing requests. The latter comprise nine Llama, four Mistral and thirteen Qwen requests, with a preflight upper count of 805 context and received tokens. Original unmodified references and composite-Markdown outcomes are unchanged.

| Decomposed arm | Exact payload recoveries among five illustrative sources |
| --- | --- |
| Per-line prefix only | 0 |
| Outer trim only | 1 |
| Trailing trim only | 1 |
| Normalize line endings to LF only | 2 |
| Remove declared wrapper after composite operation | 1 |

All 27 prefix-only segment instances change token zero and present a prohibited first forced token. The safe-text mask rejects it before a rank sequence is returned. These cases therefore show immediate span/token incompatibility, not a measured sequence of later contextual rank changes. Outer trimming and wrapper removal recover the same originally failing Mistral case and restore its saved forced-token IDs. The wrapper rule removes only the declared `> ` prefix from every line. It does not consult the payload, search offsets or restore whitespace lost earlier. The isolated line-ending direction matches the historical composite normalization, not the separate LF-to-CRLF population test. These outcomes are mechanism illustrations, not new population recovery rates. Historical composite recovery remains 0/144.

A runtime audit found 4,096 context positions allocated in the historical replay versus 512 in the new bounded worker. All 53 reused segment inputs were checked at the new capacity and reproduced their original rank and error endpoints exactly. These checks evaluated 1,626 tokens. Stage 2 therefore executed 79 distinct segment inputs across the same 25 logical arms, comprising 26 missing requests plus 53 capacity checks. This verifies the selected short inputs and does not establish arbitrary context-capacity equivalence. Per-request and per-model validation records remain under `transport`.

The canonical configuration API requires eleven components covering model, tokenizer, quantization, backend, prompt rendering, BOS/EOS, ordering, filter, codec/framing, gate and boundary/reset policy. Its example records actual pinned model and backend-library identities. Tests verify deterministic serialization, matching acceptance, corrupt-header rejection and mismatch rejection for each component. The prototype was not retrofitted into historical messages. It checks declared compatibility and provides no authentication, confidentiality or repair guarantee.

## Quantization and entropy evidence

The existing shared-history comparison reconciles 1,920 pairs, 244,440 positions and 69,528 observed-rank changes. Encoded messages contribute 960 pairs and 56,321 changes at 122,220 positions. Ordinary controls contribute 960 pairs and 13,207 changes at 122,220 positions. These denominators are distinct from the historical mean of per-message fractions.

The separate new offline endpoint decodes saved Q8 ranks on the fixed historical Q4 token paths using original codec metadata. Source ranks and payload-byte hashes are reconciled first. All 960 encoded rows are supported. Exact byte recovery is 0/960. There are 766 trials with invalid bounded ranks and 194 with valid but incorrect reconstructed bytes. B8 contributes 471 invalid trials and 3,587 invalid positions, while B16 contributes 295 and 904. Ordinary controls have no payload endpoint. No Q8-to-Q4 decoding claim follows. Neighboring-logit replay was unnecessary because no empirical margin distribution is claimed.

The six strict entropy failures retain their original budgets and denominators. Five are Qwen and one Mistral, all ASCII B16. In the declared identity order, consumed/requested ranks are 110/128, 116/128, 54/64, 106/128, 59/128 and 116/128. Longest below-threshold runs are 50, 163, 39, 35, 267 and 51 positions. Strict completion remains 114/120. The exact pinned-tokenizer windows, 12 matched ungated/moderate trajectories and three hash-selected strict successes are integrated without regenerating these trials. The 605 positions across the longest low-entropy failure runs contain 30 to 157 distinct token IDs per run, at most three identical IDs consecutively, and rank-one frequencies from 85.7% to 98.8%. Every position remains an ordinary sampled skip with zero payload progress. These are descriptive token-path characteristics, not identified linguistic causes. Successful strict trajectories also contain low-entropy stretches. Their contexts do not identify a causal linguistic trigger. Historical gating used no token filter and sampled ordinary skips at temperature 0.8 and top-p 0.95. It was not a filtered or greedy-skip experiment. V4 also clarifies that the inherited entropy rate counts eight bits per serialized payload byte per generated token, conditional on completion. The values are unchanged and are not described as Shannon channel capacity.

## Theory, bibliography and filter integration

The new main Methods includes the complete unchanged `safe_text_filter_v1` rule export. Selection performs a full stable sort, whereas known-token rank recovery uses counting. The complexity comparison inspects primary arithmetic-coding and SAAC implementations and the ADG grouping algorithm. It includes probability passes, repeated diagnostic sorts, prefills, KV state, `logits_all` storage, tails, skips, segmentation, finite-precision arithmetic, grouping data structures and direct-subword payload-side passes. It makes no cross-backend timing-superiority claim.

The configuration argument states conditional rank-agreement accounting rather than inventing a recovery rate for a minor version change. The detector is explicitly trace informed. Reproducing saved likelihood features requires the original model/backend, token path, prompts, boundaries, resets and probability convention. Untempered full-vocabulary scores differ from temperature/top-p sampling. AUC is not converted to KL. A fully configured observer may decode, so surface syntax concealment is not cryptographic secrecy.

The sufficient rank-margin bound assumes identical history, allowed set and tie rule and a uniform absolute logit perturbation bound. Gaps above twice that bound preserve order. Independent free-generation divergence changes future histories. Observed-token replay feeds received cover tokens back into the model, not recovered payload symbols. The proposed independent-block, integrity, bounded outer ECC and retransmission fallback is unimplemented. ECC cannot repair an arbitrary wrong codebook or persistent synchronization loss.

The former reference 16 is `Cai2025EntropyGuidedWatermarking`, arXiv:2504.12108. The inspected primary record has no journal reference, which does not prove the absence of another publication. V4 cites the peer-reviewed Lee et al. ACL 2024 SWEET paper for the limited precedent of entropy-threshold selective watermarking. Mark detection and chosen-payload recovery remain separate claims. Primary URLs and inspected code revisions/hashes are retained in the theory audit and source manifest.

## Compute, validation and changed files

Cumulative supervised GPU-job wall time was **14,088.189737 seconds**, or **3 hours 54 minutes 48.19 seconds**, below the 28,800-second ceiling. The ledger includes model loading, fixtures, successful runs, the failed model-ID invocation and a conservatively charged 0.086244-second concurrent-launch refusal that started no second GPU process. Seventeen jobs completed, one invocation failed before model execution, and the refused invocation is separately retained. No planned scoring unit was interrupted or left incomplete. No unrelated GPU job was terminated.

The dedicated device was the RTX 5000 Ada, UUID `GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf`. CUDA offload evidence remains in each model job's stderr. Maximum sampled process memory was 4,876 MiB, including pilot fixtures. Per-model recorded dependency versions agree, and all six pinned backend shared-library hashes were verified again. The completed Stage 1 CUDA readiness evidence was reused without another smoke run.

The three full-study jobs used 4,877.28, 5,356.26 and 3,316.84 seconds for the Qwen, Mistral and Llama evaluators. Across both pilots and the study, 15,023 new unique LM scoring requests evaluated 641,308 context and target tokens. Of these, 14,735 requests and 628,608 tokens belong to the study. Semantic inference computed 215 unique windows and 3,306 nonpadding tokens, in GPU batches of up to 64. These counters exclude numerical fixture calls, whose time remains charged. The 26 missing transport requests retain the 805-token preflight upper count rather than an invented actual count. The 53 capacity checks retain actual evaluated-token counts. Kernel-only GPU time was not separately measured.

Exact accounting and token work are in `gpu/ledger.json`, `gpu/resource_summary.json`, `gpu/job_summary.csv` and `gpu/model_work.csv`. The raw ledger hash is `ea3feeeefdf7787179cf65080574511873a822e5b9c190d7e12f335b79eb7286`. CPU analysis, writing and document compilation are outside the GPU-job ceiling.

The focused scientific tests and relevant existing regressions passed **134 tests in 5.95 seconds**. They cover alignment and boundary straddles, complete score joins, frozen score-blind payload selection, disjoint fresh controls, shared stratified bootstrap draws, checkpoint identity and corruption, configuration mismatch, codec and quantization denominators, exact review wording and GPU accounting. The full study join also reconciles every planned request and validates retained per-token scores and denominators. Separate retained-data audits check entropy runs, original quantization payload bytes and continuation policies.

Final document builds, numerical checks, review preservation, protected-history checks and rendered-page inspection are recorded separately under `results/revision_v4/stage2/validation`. The Stage 2 validator requires the complete frozen sample, all 53 replay checks, evidence-backed answer sets, unchanged source contracts and the resource ledger. It permits supported completed answers while preserving the strict thirteen-block quotation invariant. Archived Stage 1 manifests and build logs were not overwritten. The final validator passed with 8,416 protected files unchanged, all thirteen exact review blocks preserved, sixteen supported or explicitly unresolved answer sets, complete score joins and all 53 capacity checks. All four PDFs compile with no unresolved references or overfull boxes. The manuscript has 25 pages, supplement 38, response 8 and cover letter 1. Eight selected pages across the four documents were visually inspected. One inherited underfull supplement-table warning remains. Build logs and PDF hashes are retained alongside the rendered-page review record.

The main deliverables are grouped below. Machine-readable source linkage is under the Stage 2 results namespace.

- `configs/revision_v4/stage2_coherence.json`, its single amendment and `stage2_semantic_encoder.json` freeze measurement and encoder identity.
- `rankcloak/revision_v4_coherence.py`, `revision_v4_stage2_scoring.py`, `revision_v4_stage2_analysis.py` and shared utilities implement instrumentation, exact request caching and paired inference.
- `rankcloak/revision_v4_transport.py`, `revision_v4_quantization.py` and `revision_v4_compatibility.py` implement the bounded endpoints and compatibility check.
- Stage 2 scripts prepare and freeze plans, supervise and summarize GPU work, check transport reuse, render evidence-backed text, build documents, validate review preservation and prepare the local inventory.
- `tests/test_revision_v4_stage2.py` checks scientific joins, alignment, selection, cluster draws, cache identity, compatibility and accounting invariants.
- `results/revision_v4/stage2` contains all frozen inventories, pilot and study mappings, raw caches, derived tables and figures, transport and quantization outputs, provenance, resource accounting and new validation logs.
- `paperV4/scientific_reports/main4.tex` and `supplementary4.tex`, their V4 inserts, bibliography, figures and tables integrate the supported findings. The reviewer response, cover letter and response matrix are updated conservatively. All four PDFs are rebuilt.
- `revision_docs/REVISION_V4_STAGE2_THEORY.md`, `REVISION_V4_STAGE2_EXECUTION.md` and `REVISION_V4_RELEASE_SPEC.md` supply source-based arguments, reproducible execution and the remaining release contract.

## Response status and remaining submission actions

Thirteen of sixteen response sets have supported answers. The editor's technical synthesis and R1.4 are partial because semantic validation failed. Archive coverage is pending. A completed answer acknowledges limitations or supplies requested evidence. It does not mean the underlying transport or security limitation has been solved.

The published version 2.0.0 DOI remains `10.5281/zenodo.22555497`, with concept DOI `10.5281/zenodo.21987449`. The archived audit verifies V3 coverage at source commit `ce853d42d6ba64065cb63c6bdfc0d825c62734cd`, including 65 paperV3 files, and no V4 coverage. Its manuscript-exclusion description conflicts with the inventory. The local V4 specification corrects that description and inventories the prospective files. No new DOI is invented.

The remaining actions are concrete and separate from this bounded stage.

1. Make an editorial decision on the still unsupported semantic/perceptual portion of R1.4. Retain the unsuccessful pilots and explicitly limited likelihood results. If further evidence is required, freeze a new independently justified design or appropriately governed reader study before obtaining outcomes. Do not search additional metrics until significance appears.
2. Review the final scientific text, response and journal formatting. Keep adverse transport, quantization and entropy outcomes and the trace-informed attacker definition. Resolve genuine pending items before removing the working-draft banners.
3. Finalize and commit source, evidence and document bytes. Assemble a candidate archive from that exact commit, verify its complete inventory and checksum, and exclude model weights, virtual environments, credentials and incidental temporary files. The current prospective inventory is preparation, not a final archive.
4. In an authorized later release stage, publish a real new version under the existing Zenodo concept, inspect its public files, correct the manuscript-inclusion description and update Code Availability with the verified version DOI. Rebuild and verify the corresponding submission documents.
5. Only after those requirements are satisfied, perform the external journal submission. This stage does not claim submission readiness.

The commit containing this report is the Stage 2 handoff commit. Its pushed SHA and any remaining working-tree state are reported in the completion message. Future final archive assembly must use its eventual finalized successor if editorial or evidence files change.
