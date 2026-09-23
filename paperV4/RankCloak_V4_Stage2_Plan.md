# RankCloak V4 Stage 2 plan

Based on the repository at b964b938a0b2578501edf60555da8eff5ed4eddd, including the full Stage 1 report, technical audit, prospective coherence configuration, response matrix, and validation code.

## Objective

Complete the missing scientific evidence and integrate the supported findings into the V4 manuscript and reviewer response. Stage 1 already established the scaffold, original-comment preservation, entropy-failure diagnostics, exact filter specification, model inventory, and local CUDA readiness. Do not repeat that stage or regenerate the historical experiment matrices.

The main new empirical task is boundary coherence. Transformation decomposition is small. Most quantization evidence can be reused. Complexity, configuration sensitivity, security boundaries, and the conceptual fallback now need completed arguments and manuscript integration.

The proposed compute ceiling for this stage is eight cumulative hours of local GPU-job wall time, including pilots, failed attempts, and model loading. This is a recommended operational limit for the next prompt, not a measured runtime forecast or a limit previously set by the author. Pilot measurements determine whether the full study fits. CPU analysis, tests, plotting, and LaTeX work continue after model jobs finish or reach the ceiling.

## 1. Preserve Stage 1 and make validation stage-aware

Keep historical results, paperV1-paperV3, the exact requests, and Stage 1 raw/derived evidence unchanged. Record the current checkout before edits. Stage 2 may update working V4 manuscript sources, response text, and the response matrix.

Use a new versioned configuration and results directory, for example:

    configs/revision_v4/stage2_coherence.json
    results/revision_v4/stage2/
    revision_docs/REVISION_V4_STAGE2_REPORT.md

Record amendments against boundary_coherence_plan.json rather than silently replacing the original prospective design. Do not rerun the create-only scaffolder over edited documents.

The Stage 1 validator requires exactly sixteen pending response sets and pending markers in every document. Those checks describe Stage 1, not the final revision. Add Stage 2 validation that retains exact quotation and historical-evidence checks while verifying the status, evidence links, and locations of completed or unresolved answers. Keep the original Stage 1 validation record intact. Likewise, Stage 2 PDF builds must write new logs instead of overwriting Stage 1 logs and manifests.

## 2. Complete a validated boundary-coherence study

### Inventory and measurement

Stage 1 found 8,280 structural boundaries in 1,440 segmented trials spanning 240 payload groups. These are an upper bound on eligible observations, not final analysis denominators. The original full design has three arms and therefore at most 24,840 transitions, 49,680 embedding windows, and 49,680 conditional/reference LM passes before caching, matching, and exclusions.

Freeze the byte-level boundary alignment, short/empty-window treatment, ordinary pseudo-boundary placement, same-prompt donor assignment, donor reuse, truncation, and missingness rules before scoring. Preserve both overall counts and subgroup-specific exclusion rates. Short forced spans are part of the method, so excluding them can change the population being evaluated; explain that explicitly.

Use actual RankCloak transitions, matched ordinary pseudo-boundaries, and same-prompt shuffled-tail controls. Different source identities can still contain identical text. Define and record an exact-duplicate control-pair rule in advance rather than treating identical tails as informative corruption. Never replace missing same-prompt donors with unrelated topics without a separately declared analysis.

### Semantic metric choice

I recommend amending the unscored design to use a purpose-trained sentence embedding model for the primary semantic-relatedness measurement. A practical choice is sentence-transformers/all-MiniLM-L6-v2, whose official model card identifies sentence and paragraph embeddings and sentence-similarity use. Pin the exact revision, tokenizer, weights, pooling, normalization, dependencies, and truncation policy. This remains a semantic-relatedness proxy, not a validated human-coherence score.

Use an existing local copy if available. The prompt permits one public small-model acquisition up to 500 MB of required files, using an isolated environment if dependencies are missing. This is an operational allowance, not a statement of the model's exact download size. Do not download unused framework variants or replace the established generation environment. If retrieval is unavailable, the already proposed pinned DeBERTa base encoder can remain a declared fallback, but it must pass the same control-only validation and be labeled a pooled-embedding proxy. Never use the fitted steganalysis classifier as a semantic evaluator.

The second primary metric is conditional context gain under the existing different-family evaluator map: Llama-generated text scored by Qwen, Qwen by Mistral, and Mistral by Llama. Retain raw conditional tail likelihood as a secondary result. Freeze the exact prompt-only reference and boundary-straddling token treatment. Generator token IDs must not be passed to another model's tokenizer.

### Blinded pilot and decision

Select at most 36 boundary identities using the existing identity-hash rule across generator, schedule, and prompt category. Validate ordinary versus same-prompt shuffled ordinary transitions before computing RankCloak coherence scores. Record paired effects and payload-cluster intervals for the two primary metrics, model coverage, missingness, actual scoring time, and peak memory.

A positive lower 95% interval for the prespecified ordinary-minus-shuffled contrast supports control separation for that metric on this pilot. It does not establish equivalence to human judgments. Failure or inconclusive separation can reflect either limited metric sensitivity or weakly corrupted controls. Do not keep changing layers, windows, donor rules, or models until significance appears.

Predeclare at most one methodological amendment using only the control pilot. Validate an amendment on a fresh disjoint control set, disjoint in recipient and donor payloads where feasible, and retain both outcomes. If a primary metric remains unvalidated, report that limitation and keep R1.4 partly unresolved rather than claim stronger semantic evidence. Continue all independent reviewer work.

### Budget and execution

Measure the real workload before scaling. Deduplicate identical embedding inputs and identical scoring requests with exact model/configuration/input hashes. Preserve the full mapping from original units to cached results. Batch encoder work and reuse exact reference scores when valid. Any optimized LM scoring route must agree with the established scorer on a fixed validation fixture before adoption.

Reserve compute for the small transmission/quantization jobs and a safety margin. Run all eligible boundaries only if the measured forecast fits the remaining budget. Otherwise freeze a balanced hash-selected payload subset before RankCloak scoring. Keep all three models and both schedules for each selected payload and preserve the complete eligible boundary set within selected trials. Do not choose a subset using its coherence scores.

Use paired inference. Average segments within trial and repeated observations within payload before the primary overall contrast. Apply the existing 2,000 payload-cluster bootstrap with common draws across arms and models. Report that the primary intervals condition on the fixed empirical control corpus. Show donor support and reuse; add donor-aware sensitivity if broader population claims are made.

Deliverables are an eligibility and donor manifest, a frozen execution plan, pilot report, raw/derived scores, paired effects and intervals, missingness/donor tables, a coherence figure, a main-results table, and three transparently selected strong/typical/weak examples. Keep favorable and adverse findings.

## 3. Finish bounded transport diagnostics and configuration checks

Select at most one unmodified visible-text success and one failure per source model using the existing hash rule. Missing strata stay unavailable. Reuse original unmodified and composite-Markdown outcomes. Add at most thirty cover replays for isolated prefix insertion, outer trimming, trailing-space trimming, line-ending normalization, and declared wrapper removal. Count segments and token work, not only cover count, in the forecast.

Record byte changes, token changes, extracted spans, first rank differences, and exact payload outcomes. Wrapper removal must use a declared transport convention and cannot consult the original payload or optimize an offset for successful decoding. Keep original zero-recovery results unchanged. Interpret the six-case selection as an illustrative mechanism study, not a new population recovery-rate estimate.

Implement canonical configuration fingerprints using existing reproducibility helpers where possible. Test agreement, intentional mismatch, and stable canonical serialization. Present this as compatibility checking, not error correction or cryptographic authentication.

## 4. Use quantization evidence and add only what is missing

Stage 1 verified 1,920 common-history pairs and 244,440 positions, including 69,528 observed-rank changes. Identify label, representation, and population denominators before using those pooled counts for a specific claim.

Reuse these records for same-history rank sensitivity and the independent generated-path comparison. If additional fixed-message decoding evidence is useful, reconstruct the bounded Q4-to-Q8 endpoint offline from stored Q8 ranks on Q4 messages and original codec metadata. Report actual byte recovery, invalid-rank failures, and denominators. Do not infer the reverse direction from this endpoint.

Only if a stated quantitative claim needs raw neighboring-logit margins, use the proposed sixteen payload-class by codec identities, two quantizations, and up to 64 positions each. This caps that job at 32 prefixes and 2,048 observed-token evaluations. Do not run another full quantization matrix.

Explain that an observed-token decoder feeds received cover tokens into the model, not decoded payload symbols. Independent autoregressive divergence, retokenization, boundary shifts, gate desynchronization, and wrong recovered ranks have distinct propagation mechanisms.

## 5. Complete manuscript and response integration

Finish the source-based complexity comparison for arithmetic coding and grouping methods, including actual sorting, cumulative tables, finite precision, prefill and incremental model costs, repeated diagnostics, logits_all storage, and KV memory. Do not equate asymptotic cost with wall-clock performance across backends.

Integrate the precise trace-informed attacker contract, distributional explanation, support caveat for top-p controls, configuration argument, and conceptual block/ECC/retransmission fallback. RankCloak does not establish confidentiality or indistinguishability against the evaluated informed attacker. Do not claim a full-configured attacker cannot decode.

Turn the completed entropy analysis into a direct response to R1.6, keeping all six failures and the sampled-skip policy. Resolve the already-supported filter and bibliography responses. Integrate coherence and transport/quantization findings only after validation.

Preserve all thirteen exact review blocks. Complete the sixteen response sets individually with direct answer, evidence/change, and accurate manuscript location. Leave genuine unresolved markers visible. Remove obsolete Stage 1 language where it no longer describes the current evidence, while retaining an overall working-draft status if archive or other final requirements remain.

The archive audit already found a published version 2.0.0 covering V3, DOI 10.5281/zenodo.22555497. Prepare a local V4 release inventory and description that matches its contents; the existing public description incorrectly says manuscript files are excluded. Do not publish the new archive or claim a V4 DOI already exists. Final release assembly can follow the final source/evidence commit and must be checked against that exact commit.

## 6. Stage 2 completion criteria

- Every attempted experiment has explicit status, denominators, input/model/configuration identities, and a resumable record.
- Coherence metrics have documented control validation, or their failure is explicitly retained and R1.4 remains open.
- Planned runs finish within the cap or stop cleanly with a checkpoint; no partially processed convenience sample is passed off as the frozen completed sample.
- Supported evidence is integrated into the main paper, supplement, response, and cover letter.
- Stage-aware checks verify exact comments, unchanged historical evidence, numerical provenance, build results, and visual layout.
- A Stage 2 report states cumulative GPU time, full or subset scope, findings, unresolved reviewer items, and the remaining submission actions.
- Completed work is committed and pushed through the established repository workflow without force pushing or including unrelated local changes.

If this stage succeeds, the next stage should be a final scientific/editorial check and versioned release preparation, not another open-ended experimental campaign.

## Primary sources for the semantic measurement recommendation

- Official all-MiniLM-L6-v2 model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Reimers and Gurevych, Sentence-BERT, EMNLP-IJCNLP 2019: https://aclanthology.org/D19-1410/

The suitability of either metric for these short forced-span boundaries must still be checked on the declared controls.
