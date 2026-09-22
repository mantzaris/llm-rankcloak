# RankCloak V4 revision plan

Prepared for Alexander V. Mantzaris on 22 September 2026.

Repository inspected: https://github.com/mantzaris/llm-rankcloak

Inspection commit: 60d23fffe5962d4e0b652fa045d762ca10bc3dcc.

The attached reviewer letter exactly matches paperV4/response/requests.txt at this commit. The local Codex session must treat its current checkout and any later author edits as authoritative, record differences, and preserve existing work.

## 1. Recommended revision strategy

Produce a focused V4 revision that explains the mechanisms behind the existing adverse results and adds targeted validation. Preserve the artifact-concealment and exact-recovery contribution, but state that the evaluated channel is fragile and statistically distinguishable under the tested informed adversary. Neither a new steganographic architecture nor repetition of the entire primary experiment is needed to begin answering this letter.

The main empirical additions should be boundary-specific semantic-coherence evaluation, a trace-level analysis of the six strict-gate failures, and an explanation of quantization divergence that distinguishes changes under identical histories from changes after generation paths diverge. Most other requests can be addressed through source audits, existing evidence, mathematical analysis, and clearly bounded mitigation proposals.

Automated coherence scores can strengthen evidence about transitions; they cannot prove that readers perceive the transitions as natural. Likewise, a conceptual error-correction design must not be presented as demonstrated cross-model recovery.

## 2. What the repository already establishes

| Finding | Verified evidence | Consequence for V4 |
| --- | --- | --- |
| V4 currently contains the reviewer requests | paperV4/response/requests.txt | Bootstrap the manuscript, supplement, response, and cover letter from V3 without replacing the new requests. |
| V3 already separates saved-ID and visible-text recovery | paperV3/scientific_reports/main3.tex and V3 recovery tables | Retain the endpoints and denominators. Improve their explanation rather than relabeling them. |
| The Markdown operation is specific | apply_transmission_transform in rankcloak/revision_protocol.py | It normalizes line endings, strips outer whitespace, strips trailing line whitespace, and prefixes every line with a greater-than sign and a space. Describe this blockquote-style transformation accurately. |
| Strict gating failed on six attempts | entropy_generation_trials.csv and six raw entropy records | All six use ascii_b16; five are Qwen and one is Mistral. All original failures remain in the denominator. |
| Failure records contain detailed traces | results/revision_v3/generation/raw/entropy/ | Per-token entropy, eligibility, role, IDs, log probabilities, observed ranks, and progress are already available. Start offline. |
| Quantization already has a shared-history diagnostic | q8_replay_of_historical_q4_path and q4_q8_same_path_distribution_comparison in revision_v3_generation.py | Reuse it. Do not mistake independent-path disagreement for isolated logit error. Raw full logits or local rank margins may still require a small replay. |
| The model-aware detector uses saved generation likelihoods | revision_v3_analysis.py and configs/revision_v3/analysis.json | Explain its access assumptions. Saved-ID trace scores are not automatically identical to scores reconstructed from visible text. |
| Token selection sorts, while rank recovery counts | rankcloak/token_filters.py and rankcloak/rank_codec.py | Analyze selection as full sorting and known-token rank recovery as a linear scan. Do not assign identical sorting complexity to both. |
| Reference 16 is identifiable | main3.bbl, key Cai2025EntropyGuidedWatermarking | Audit the actual citation key, not a number that will change after editing. |
| A public code DOI exists | Zenodo API record 21987450, checked 22 September 2026 | The record is published. revision_docs/DOI_RELEASE_PLAN.md contains stale draft language. Verify version coverage and prepare a V4 archive update. |

The inspected public Zenodo record is https://doi.org/10.5281/zenodo.21987450. Its concept DOI is 10.5281/zenodo.21987449. The public API reported creation and last update on 20 August 2026. This establishes publication, not that the archive includes all subsequent V3 or V4 work. Inspect the archive manifest before making that claim.

### Six retained strict-gate failures

These are existing results, not proposed experiments. The token budgets below count embedding-generation positions.

| Trial suffix | Model | Payload | Prompt | Ranks consumed / requested | Tokens used / budget |
| --- | --- | --- | --- | --- | --- |
| 2b5e682d0007193b56249891 | Mistral | hmac_sha256_hex_003 | explain_everyday_system | 110 / 128 | 768 / 768 |
| 941eed91360afbdc8a26bf09 | Qwen | sha256_hex_041 | professional_project_update | 116 / 128 | 768 / 768 |
| 9ae150681b7e7347c8241d5a | Qwen | token_128_bit_hex_003 | explain_everyday_system | 54 / 64 | 384 / 384 |
| a9ff67133eeaa189fa8be8b1 | Qwen | sha256_hex_003 | explain_everyday_system | 106 / 128 | 768 / 768 |
| c4160dc01512e365fbecb170 | Qwen | hmac_sha256_hex_041 | professional_project_update | 59 / 128 | 768 / 768 |
| cbb747058efe6a8b1fbd18ec | Qwen | chacha20_poly1305_base64_041 | professional_project_update | 116 / 128 | 768 / 768 |

Raw filenames begin with entropy_trial__. Payload names in the records begin with revision_v1_. Gate thresholds are 1.9175505039890282 bits for the listed Mistral failure and 2.4325573188780316 bits for the Qwen failures. These cases establish insufficient eligible positions within the budget. They do not yet establish which linguistic patterns caused those positions.

## 3. Point-by-point work and response plan

The labels below are internal work labels. In the response letter, reproduce the editor and reviewer wording verbatim, including introductory paragraphs, Reviewer 2's acknowledgment, and Reviewer 1's instruction about preserving comments. Do not substitute the summaries in this plan for the original comments.

### R1.1. Transformations and Markdown failure

Code and evidence:

- Audit the actual transformation and the downstream retokenization and span-extraction rules. In particular, the current visible-text extraction applies saved token offsets after retokenization. Prefix insertion may therefore cause boundary misalignment as well as context and token changes.
- Trace representative failed cases through rendered bytes, transformed bytes, token IDs, extracted spans, contextual ranks, and recovered payload. Include an unmodified successful case and an unmodified retokenization failure if available.
- Decompose the Markdown operation into prefix insertion, outer trimming, trailing-space removal, and line-ending normalization. Reuse existing transformation cells when equivalent. Use a small predeclared replay only for missing comparisons.
- A diagnostic wrapper-removal arm may remove a known transport wrapper without access to the original payload. Record what it can restore and whether whitespace loss remains irreversible. Keep this new arm separate from historical zero-recovery results.

Paper correction:

Explain that semantic equivalence of text does not preserve the discrete sequence required by the decoder. Distinguish three mechanisms: a shifted span boundary, changed tokenization, and changed conditional rankings. The measured failure rate belongs to the implemented blockquote-style operation, not to all Markdown applications.

Discuss byte-preserving transport, declared reversible wrapping, and tokenizer-aware generation checks as possible boundary mitigations. Canonicalization that loses information cannot universally reconstruct the original string or token IDs. Even identical visible bytes need not recover the original generated tokenization.

Response completion criterion: exact transformation definition, concrete before/after examples, first-divergence evidence, and a clearly scoped mitigation discussion linked to manuscript locations.

### R1.2. Minor configuration desynchronization

Code and evidence:

- Audit the shared configuration: model and tokenizer hashes, quantization, backend/build, prompt rendering, BOS/EOS behavior, normalization, mask, ordering, codec, gate, and boundaries.
- Define a canonical configuration manifest and fingerprint using existing reproducibility utilities where possible. If adding a mismatch check, test matching acceptance and altered-configuration rejection without changing legacy defaults.
- Separate tokenizer segmentation changes, token-ID reassignment, vocabulary changes, and unchanged tokenization with altered logits. A minor software version update is not a quantifiable channel-error rate by itself.
- Use real alternative tokenizer versions only when locally available and precisely pinned. Label synthetic perturbations as sensitivity probes rather than version-update experiments.

Paper correction:

Model success as a sequence of conditional agreement events. The product of conditional probabilities is valid by the chain rule. A formula such as (1 - epsilon)^L requires an explicitly stated independent, identical per-step error approximation and is not an estimate for real tokenizer updates. A deterministic mismatch can instead leave every tested step unchanged or systematically break a subset of messages.

Explain configuration pinning, preflight fingerprints, and rejection of incompatible configurations. These prevent unsupported decoding attempts; they do not create cross-configuration robustness.

Response completion criterion: explicit protocol contract, conditional sensitivity argument, and honest interpretation of a mismatch preflight.

### R1.3. Zero cross-model recovery and conceptual mitigation

Code and evidence:

Reuse cross-model outcomes and source denominators. Document which model pairs, token transport, prompt, and span assumptions were tested. Do not extend zero recovery in those cells to a theorem about every possible model pair.

Paper correction:

Propose a layered fallback:

1. Identify the required decoder configuration through a pre-agreed or explicitly transmitted identifier and select the matching decoder when available.
2. Protect independently decodable blocks with an integrity check; use authentication only if an external keyed mechanism is actually specified. A checksum alone is not authentication.
3. Treat detected damaged blocks as erasures, provided boundaries and context resets remain recoverable.
4. Apply outer erasure/error correction within its stated correction radius, or retransmit failed blocks through an acknowledged channel.

Clarify that this is a proposed protocol extension, not a property of the current implementation. Ordinary payload ECC cannot restore an unrelated rank codebook or guarantee recovery after insertions/deletions and persistent synchronization loss. Repetition, redundancy, metadata, resets, and retransmission consume capacity and can add detection cues. If the matching decoder is unavailable, an explicit alternate transport is a fallback with different concealment properties.

A toy ECC simulation is optional and must remain a codec-level illustration. It is not necessary to build a new end-to-end robust system for this comment.

Response completion criterion: a concrete conceptual strategy with assumptions, costs, and failure cases, with no fabricated recovery improvement.

### R1.4. Coherence at forced-span to tail transitions

This is the highest-priority new evaluation. Whole-message readability and increased average likelihood from adding a tail are insufficient for this request.

Data design:

- Reuse the existing 1,440 paired segmented trials where feasible. Choose a balanced subset only if timing requires it, using a seeded identity-based rule fixed before scoring outcomes are inspected.
- Score actual payload-span/tail boundaries within each message. Evaluate cross-message topic continuity separately because the multi-topic condition deliberately changes prompts.
- Compare against matched ordinary continuations with pseudo-boundaries matched to forced-span lengths, plus shuffled-tail negative controls. Prefer same-prompt or same-topic shuffled tails to test more than trivial topic mismatch.
- Preserve message and payload clustering. Several segment boundaries from one payload are not independent observations.

Recommended complementary measurements:

1. Semantic relatedness between fixed windows before and after the boundary using a pinned sentence encoder or comparable semantic model. Record tokenizer, window policy, truncation, and missing/too-short cases.
2. Conditional compatibility of the tail with its preceding text under a different model family. Reuse the repository's fixed evaluator map: Llama to Qwen, Qwen to Mistral, Mistral to Llama. Compare real transitions with matched ordinary and shuffled transitions. Use equal window policies and length normalization.
3. Repetition and entity-reference continuity as interpretable secondary diagnostics. An NLI contradiction score may be supplementary but is not a universal coherence measure, especially for incomplete text windows.

Run a small blinded pilot to verify that the chosen metrics distinguish ordinary transitions from deliberately broken ones. If they do not, report the failed validation and revise the measurement before freezing the full analysis. Do not tune metrics to make RankCloak look better.

Outputs:

- One boundary-coherence figure with distributions or paired effects and payload-group confidence intervals.
- One compact table with denominators and complementary metric results.
- A few transparently selected examples spanning strong, typical, and weak measured transitions; show the boundary and source ID. These are illustrations, not participant evidence.

Response completion criterion: stronger automated boundary-specific evidence and revised claims that stop short of proving human naturalness. If results show weak transitions, say so directly and narrow the interpretation of natural tails.

### R1.5. Exact-model-aware detectability and security boundary

Code and evidence:

- Audit feature provenance and the exact access contract. V3 uses saved generation token-log-probability summaries for both labels. List availability of prompt, model, original tokenization, boundaries, and traces.
- Reuse frozen splits and predictions. Examine rank pressure and surprisal by representation and token role where those annotations exist; ordinary-control rank fields are unavailable in some historical cohorts, so do not manufacture them.
- A small no-refitting check may compare scores reconstructed from visible text with saved-trace scores. Treat this as a distinct access condition; if omitted, retain the trace-informed interpretation explicitly.
- Do not rerun the entire neural-detector training matrix merely to explain existing near-perfect AUCs.

Paper correction:

Let p_t be the declared ordinary cover distribution and q_t the distribution induced by encoding, conditional on the history. The exact-model observer can exploit systematic differences between them, for example repeated high-surprisal ranks and changes between forced spans and tails. A bounded rank range does not mean ranks follow the ordinary model's probabilities. Payload symbols are not necessarily independent or uniform; ASCII-radix representations can induce additional structure.

For an appropriate stochastic message ensemble, the sequence KL divergence decomposes into expected conditional KL terms. Specify the reference distribution, temperature, filtering, and support assumptions. Ordinary top-p controls may assign zero probability outside their support, so avoid an unsupported finite-KL calculation. Do not equate empirical ROC-AUC with a derived KL value or claim all linguistic steganography must be detectable.

State plainly that RankCloak does not provide statistical indistinguishability against the evaluated informed attacker. Surface-form concealment, payload recoverability, confidentiality, and stealth are separate claims. A detector alone does not perform decoding, but an observer who also has the receiver's complete configuration may be able to run the decoder. RankCloak does not provide encryption or secrecy from public rank mappings.

Response completion criterion: mechanism tied to observed features, accurate adversary access, and an unambiguous security boundary without dismissing high detectability.

### R1.6. Exact causes of entropy-gate failures

Start with all six historical failures and their matched ungated and moderate-gate records. Add a transparently selected set of successful strict-gate records to avoid describing failure-only associations as causes.

Compute from retained traces:

- Threshold, eligible-position fraction, consumed and remaining ranks, budget, completion status, and first sustained low-progress region.
- Longest runs below threshold, rolling progress and eligibility, entropy margins relative to threshold, token roles, surprisal, rank pressure, and repeated sequences.
- Precisely aligned linguistic windows around stalls. Use the pinned tokenizer and prefix decoding to align text; isolated token pieces alone may have misleading Unicode rendering.
- Associations with formulaic language, lists, repeated phrases, punctuation, or other patterns only when supported by the observed windows. Mark analyst coding as exploratory.

The gate consumes one requested rank at each eligible position. With L requested ranks and a budget of 6L positions, a failed trajectory that exhausts its budget has fewer than L eligible positions, hence a realized eligible fraction below 1/6. The six retained ratios satisfy that condition. This explains the accounting failure but not why linguistic contexts stayed below threshold. A calibration 75th percentile on ordinary development traces does not guarantee 25% eligibility on the different embedding trajectory.

Use one figure showing entropy, threshold, and cumulative consumed ranks for representative cases, and a complete six-case table. Extend the token budget only as a separately labeled diagnostic if it resolves a specific remaining question; preserve the original failures.

Response completion criterion: trace-to-text explanations for all six cases, matched context, and a clear distinction between observed association, capacity accounting, and demonstrated causal mechanism.

### R1.7. Computational complexity

Use source-derived operation counts rather than speculative timing claims.

Define vocabulary size V, allowed-set size A, payload rank count L, generated positions T, number of segments S, and model evaluation cost C_model at the relevant context length. Account separately for prompt prefills, incremental decoding with a KV cache, candidate processing, payload conversion, tails, and replay. Include memory for weights, KV state, and temporary vocabulary arrays.

In the inspected implementation, selecting a requested token rank uses a full stable lexicographic sort, O(A log A), while recovering the rank of an observed token counts higher logits and equal-logit lower IDs, O(A). Full-vocabulary probability and entropy calculations can still cost O(V). Tail and control routines must be audited separately; do not assume an optimized argmax if the code calls the full sorting selector. One-time mask construction decodes vocabulary pieces and is cached; report its cost separately.

For fixed-radix ASCII codecs, derive L from serialized byte length and radix using the implemented code, including padding or per-byte digit expansion. Gate skips and natural tails increase T without adding payload ranks. Segmentation adds repeated prompt prefills and continuations. The direct-subword baseline additionally uses an LM pass to obtain payload-side ranks.

Compare algorithmic components with the primary sources and actual algorithms for arithmetic-coding and grouping-based steganography. State whether probability sorting, cumulative tables, tree/group construction, and finite-precision arithmetic are included. Do not claim all alternatives are O(V), or that their constants are comparable across backends. Retain historical wall times with their original inclusive measurement scope.

A tiny local profiler can validate an unresolved bottleneck, but a hardware sweep is unnecessary.

Response completion criterion: a complexity table and equation tied to the actual implementation and established comparators, with empirical wall time kept distinct.

### R1.8. Reference 16

The inspected V3 bibliography identifies it as Cai, Ding, and Tao, Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation, arXiv:2504.12108. The arXiv record checked on 22 September 2026 lists the April 2025 preprint and no journal reference. No formal publication was verified during this inspection; that is not proof that none exists.

Recommended resolution: use the peer-reviewed ACL 2024 paper Who Wrote this Code? Watermarking for Code Generation by Lee et al. for the specific precedent of entropy-threshold selective watermarking. Retain Cai et al. as explicitly labeled historical motivation only if useful, and remove any dependence on unverified theoretical claims. V4's payload-gate definition and analysis must stand on their own. The two watermarking methods are not interchangeable algorithms.

Update bibliography metadata, in-text claims, and reference-audit records together. Keep watermark detection distinct from arbitrary-payload recovery.

Response completion criterion: verified citation metadata and a claim-specific replacement or careful justification, with no invented publication venue.

### R1.9. Complete deterministic filter rules

Generate the specification from the code that produced the retained data. The current safe_text_filter_v1 rejects:

- Empty decoded pieces.
- Pieces containing Unicode replacement character U+FFFD.
- Any character with code point below 32 except newline and tab.
- A case-insensitive occurrence of any blocked substring listed in the function. Export the exact list, including punctuation and escape sequences, into a main-text table.
- A stripped piece beginning with a hash character. The extra double-hash check in code is redundant, but record it faithfully.
- Two consecutive backslashes or at least two backslashes anywhere in the piece.

Also document individual-token decoding, UTF-8 replacement behavior, exception handling, token-mask caching, unknown-filter rejection, insufficient allowed ranks, and exact logit ties broken by ascending token ID. Distinguish this filter from sampling-time special-token exclusions and other roundtrip filters. The predicate does not automatically reject every Unicode control character, every markup fragment, or every prose-level artifact assembled across several tokens.

Add table-driven tests for each rule, near-boundary allowed cases, and tie ordering where current tests do not cover them. Report allowed-vocabulary counts and mask hashes per pinned model when available. Keep the actual full rule list in the main text as requested, with longer examples in the supplement.

Response completion criterion: a main-text specification that reproduces the actual predicate and scope. Do not silently improve the filter and describe old results as using the new behavior.

### R1.10. Quantization divergence

Reuse the paired Q4/Q8 shared-history results. These already compare both distributions on the Q4 token path, in addition to comparing independently generated sequences.

Explain the local order mechanism using z_i and perturbed logits z_i + delta_i. An ordering between two tokens can reverse when their perturbation difference exceeds their original margin. Under a uniform bound |delta_i| <= epsilon, an original pairwise gap greater than 2 epsilon is sufficient to preserve that pair's ordering. This is a sufficient condition, not an empirical bound on this backend until measured. Stable token-ID tie-breaking resolves exact ties; it cannot prevent a non-tied pair from crossing.

Use a fixed, stratified subset only if additional margins must be replayed. Log chosen-rank neighbor margins, rank inversions, first divergence, and same-path differences. Separate deterministic forced-rank selection from random sampling, where the same RNG seed can choose different tokens after probabilities change.

Distinguish three endpoints:

1. Distribution/rank changes while feeding an identical observed history.
2. Autoregressive feedback after independently generated paths first diverge.
3. Recovery when a receiver uses a different quantization on a fixed received message.

Different generated paths are compatible with exact recovery when each quantization decodes its own path. They do not alone measure cross-quantization decoder failure. An observed-token replay does not feed an erroneous recovered payload symbol back into the cover model; avoid claiming that it does. Tokenization, regenerated lead-ins, gate decisions, and boundary shifts have different propagation mechanisms.

Response completion criterion: shared-history versus free-generation explanation, existing paired evidence, and local margin diagnostics only where needed.

## 4. Execution sequence and stopping points

### Stage 1: Audit, V4 scaffold, and offline evidence

Use the accompanying first Codex prompt. Record current repository and GPU state. Preserve V1-V3 and existing results. Create the V4 manuscript, supplement, response, and cover-letter structure. Preserve every current review comment verbatim. Build the review-to-evidence matrix. Implement the minimal offline six-case gate analysis and filter-rule export. Make a bounded local GPU smoke check only to establish readiness. Produce an explicit next-stage plan with unresolved items.

Stage 1 is complete when these concrete files exist and their results are checked. It is not submission readiness.

### Stage 2: Freeze targeted analyses and finish instrumentation

Select coherence units and controls, metric definitions, paired units, confidence intervals, and exemplar-selection rules before outcomes are inspected. Validate metrics on ordinary and shuffled controls. Reuse same-history Q4/Q8 traces and decide whether local margins require a replay. Define narrowly scoped transformation diagnostics. Freeze a new V4 configuration and input hashes.

Use the connected local GPU and existing pinned weights. Measure pilot throughput and forecast total execution from actual scoring counts and lengths. The GPU model and old compute budget must be rechecked locally. Do not invent a duration from earlier project history.

### Stage 3: Execute only missing evidence

Run the frozen boundary-coherence evaluation and genuinely missing diagnostic replays. Keep local GPU inference explicit, with backend, model hashes, device identity, CUDA evidence, elapsed time, and peak memory. CPU postprocessing is appropriate; a failed GPU load must not silently become a long CPU model run. Use one resumable GPU job at a time unless the local hardware and project instructions support otherwise.

Reuse all viable original trials, detector predictions, and trace fields. Keep new diagnostic subsets separate from the original confirmatory results. Preserve failures and exact denominators.

### Stage 4: Integrate manuscript and response

Write results from the generated tables. Add the transport/configuration discussion, conceptual fallback, security analysis, exact filter specification, and implementation-specific complexity. Update the abstract and conclusion to match the evidence. Retain the existing title unless the author elects to change it.

Use main4.tex and supplementary4.tex, response_to_reviewers_v4.tex, and cover_letter_v4.tex. Adapt V3 formatting and macros, not its previous reviewer labels or assertions of completion. Every response should contain the unchanged comment, a direct answer, changes made, evidence or limitation, and final manuscript location.

Use measured language for unfavorable results. A response can be complete because a limitation is now well explained; it need not claim that the limitation was eliminated.

### Stage 5: Submission validation and archive preparation

- Rebuild manuscript, supplement, response, and cover letter. Check citations, references, figure numbering, PDF layout, clipping, and changed page/line references.
- Verify every new numerical statement against source tables. Trace examples to actual trial IDs. Ensure the ten comments and editor requirements are all covered.
- Do not impose an arbitrary page target. Keep the full filter criteria in the main text; put detailed traces, secondary plots, and complete subgroup tables in supplementary material.
- Audit the existing Zenodo files against their claimed code/data version. Prepare a versioned V4 archive with source commit, configuration, data manifests, licenses, reproducibility commands, and portable paths. Do not redistribute model weights or third-party text contrary to its license.
- Prepare any external deposit/update for author action and final DOI verification. The initial local work does not publish a release or submit to the journal.

Finish with a submission-readiness report listing any unresolved evidence, archive, or author-only action. Do not hide unresolved items behind a successful LaTeX build.

## 5. Suggested repository additions

Follow existing naming and helpers. The following paths are proposals, not files already created by this plan:

    revision_docs/REVISION_V4_PLAN.md
    revision_docs/REVISION_V4_STAGE1_REPORT.md
    paperV4/REVIEW_RESPONSE_MATRIX.md
    paperV4/scientific_reports/main4.tex
    paperV4/scientific_reports/supplementary4.tex
    paperV4/scientific_reports/figures/
    paperV4/scientific_reports/supplementary_tables/
    paperV4/response/requests.txt
    paperV4/response/response_to_reviewers_v4.tex
    paperV4/cover_letter/cover_letter_v4.tex
    configs/revision_v4/
    results/revision_v4/provenance/
    results/revision_v4/source_tables/
    results/revision_v4/figures/
    results/revision_v4/examples/

Add small V4 analysis modules and command-line entry points only where needed. Reuse loading, hashing, paired statistics, GPU accounting, and LaTeX table helpers. Avoid a parallel replacement framework.

Potential final evidence products:

| Product | Purpose |
| --- | --- |
| Boundary-coherence figure and metric table | Directly answers R1.4. |
| Six-failure table and selected entropy/progress traces | Directly answers R1.6. |
| Shared-history margin and independent-path divergence figure | Answers R1.10 if existing summaries cannot already support the explanation. |
| Full filter rule table in Methods | Directly answers R1.9. |
| Algorithmic cost comparison table | Directly answers R1.7. |
| Two transformation examples and a few boundary examples | Makes R1.1 and R1.4 understandable without substituting anecdotes for evidence. |

The response matrix should record comment ID, exact comment source, code changes, analysis artifact, manuscript location, response text location, validation, and status. Useful statuses are pending, existing evidence verified, new analysis complete, manuscript integrated, and final checked.

## 6. Primary literature and source checks

- Cai et al., Entropy-Guided Watermarking for LLMs, inspected arXiv record: https://arxiv.org/abs/2504.12108. Preprint status should be rechecked before submission.
- Lee et al., Who Wrote this Code? Watermarking for Code Generation, ACL 2024: https://aclanthology.org/2024.acl-long.268/. Peer-reviewed source for selective watermarking through entropy thresholding.
- Yan and Murawaki, Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models, EMNLP 2025: https://aclanthology.org/2025.emnlp-main.361/. Already present in the manuscript bibliography and relevant to tokenization limitations and possible mitigations.
- Ziegler et al., Neural Linguistic Steganography, EMNLP-IJCNLP 2019: https://aclanthology.org/D19-1115/.
- Shen et al., Near-imperceptible Neural Linguistic Steganography via Self-Adjusting Arithmetic Coding, EMNLP 2020: https://aclanthology.org/2020.emnlp-main.22/.
- Zhang et al., Provably Secure Generative Linguistic Steganography, Findings of ACL-IJCNLP 2021: https://aclanthology.org/2021.findings-acl.268/.
- Existing RankCloak archive: https://doi.org/10.5281/zenodo.21987450. Publication verified through https://zenodo.org/api/records/21987450; archive coverage remains to be audited.

These sources support specific technical context. They are not an instruction to add unrelated citations or accept all claims of every cited method.

## 7. Original review letter

The exact attached letter is appended below by the preparation script. This copy is the source for verbatim quotation, not the condensed work labels above.

---
Dear Dr Mantzaris,

Your manuscript, "RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text", has now been assessed.

We invite you to revise your paper, carefully addressing the comments from the reviewers and the editor. Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained. When your revision is ready, please submit the updated manuscript and a point-by-point response. This will help us move to a swift decision.

Please note that if your manuscript uses any custom or bespoke computational tool or code, or reports a new algorithm, tool, software, or a pipeline (even if individual components are not new), the underlying code must be deposited in a recognised DOI-assigning repository (e.g. zenodo) and linked either from Methods or a dedicated Code Availability section.

Editor Comments

"The manuscript presents a technically interesting and systematically evaluated framework for embedding cryptographic payloads into language-model-generated text through deterministic token-rank manipulation. Reviewer 2 considers the previous concerns largely addressed. However, several substantive issues remain before the conclusions can be fully supported. In particular, the authors should clarify the robustness boundaries under text transformations and configuration desynchronization, analyze the implications of zero cross-model recovery, explain the near-perfect detectability under an exact-model-aware attacker, strengthen semantic-coherence validation, document all deterministic filtering rules, and provide deeper analyses of entropy-gating failures, computational complexity, and quantization-induced divergence. Suggested references are optional, and the authors may select appropriate recent peer-reviewed literature."
-Chun-Wei Yang

We recommend submitting all revisions within the mentioned deadline.

If you need more time, please contact us and include your submission ID.

Kind regards,

Purva Dhamankar
---

Reviewer 2

Thank the authors for addressing most of my concerns.

Reviewer 1

The manuscript proposes a steganographic method embedding cryptographic payloads into language model outputs via deterministic token rank manipulation. While demonstrating exact recovery under strictly controlled conditions, the framework exhibits fragility against text transformations and cross-model decoding, limiting its practical utility despite comprehensive evaluations of capacity and neural steganalysis detection.

Overall, the study presents a reproducible measurement framework for linguistic steganography. However, its reliance on perfectly synchronized configurations and failure during standard markdown operations require deeper theoretical justification. Furthermore, near-perfect detectability by exact-model-aware attackers and the absence of perceptual validation weaken its broader claims. Significant revisions addressing robustness boundaries, theoretical limitations of detectability, and expanded discussions on quantization sensitivity are required before publication can be considered.

Please do not arbitrarily simplify or rephrase the review comments, as this hinders the assessment of your revisions.

1- In the abstract and introduction, the manuscript admits severe transmission fragility. While you explicitly scope this as a measurement framework, you must expand the theoretical discussion on why standard operations like markdown copy cause complete failure and discuss potential boundary mitigations.
2- Within the bounded payload methods section, assuming perfectly synchronized model configurations is highly restrictive. Please introduce a theoretical discussion addressing how minor configuration desynchronization, such as slight tokenizer version updates, theoretically impacts the overall payload recovery rate.
3- In the exact copy fragility evaluation, the zero percent recovery rate for cross-model decoding is a severe limitation. You should propose and discuss a conceptual error correction strategy or fallback mechanism to mitigate this lack of robustness during mismatched decoding.
4- Throughout the segmented multi-cover method section, the distinction between payload-bearing spans and natural tails lacks perceptual validation. You must provide stronger automated semantic coherence metrics to prove that these generated transitions appear natural and logically connected to readers.
5- Neural steganalysis results indicate near-perfect detection by the exact-model-aware attacker. Rather than a full redesign, you must deeply analyze the theoretical bottlenecks causing this glaring statistical detectability and clearly define the framework's security boundaries against this specific threat model.
6- The discussion of entropy gating notes that the strict gate condition resulted in maximum-budget payload completion failures. You should investigate and report the exact linguistic contexts or token-level characteristics that triggered these specific failures during the generation loop.
7- The computational overhead analysis reports significant performance bottlenecks. Instead of exhaustive hardware baselines, please provide a theoretical computational complexity analysis comparing your rank transcoding method against standard linguistic steganography tools to mathematically justify the observed processing costs.
8- In the reference list, citation 16 is an unverified preprint. If formally published, update the citation; otherwise, provide detailed theoretical justification for relying on its entropy-guided watermarking claims, or replace it with a peer-reviewed publication to strengthen your foundation.
9- The methods section describing the deterministic token filter leaves the rules for excluding markup fragments obscure. You must explicitly list these complete exclusion criteria in the main text so independent researchers can accurately replicate your experimental setup.
10- The paired quantization experiment shows independently generated token paths diverging significantly despite identical inputs. Please expand this section to theoretically discuss how minor quantization errors in logits are amplified by the deterministic sorting process, triggering this avalanche effect.
