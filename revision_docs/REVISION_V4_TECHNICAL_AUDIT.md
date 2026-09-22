# V4 Stage 1 technical audit

This is an internal working document. Existing-data findings, conditional arguments, and proposed extensions are distinguished below. The historical implementation and outcomes have not been changed.

## Transmission and configuration

`rankcloak/revision_protocol.py` implements `markdown_copy` and `markdown_copy_paste` by normalizing CRLF and CR to LF, stripping outer whitespace, removing trailing whitespace from each line, and adding the literal string `> ` at the start of every line. This is a particular blockquote operation. It is not a test of every Markdown editor or copy workflow.

`revision_runner.execute_robustness_decode` retokenizes the transformed text and takes the original half-open offsets from that new sequence. `revision_protocol.retokenize_message` uses the same saved-offset convention. Three mechanisms must therefore be separated. An inserted prefix can shift which tokens fall inside the extracted span. Rendering and tokenization can change the token sequence. Changed preceding tokens can alter later conditional ranks. The retained first-divergence fields locate mismatches but do not isolate causal contributions.

For example, the transformation of `  Alpha  \r\nBeta \t\n` is `> Alpha\n> Beta`. The wrapper and several whitespace changes occur together. Removing `> ` recovers `Alpha\nBeta`, which does not restore the original bytes. This example is a deterministic code illustration, not a study trial.

The next diagnostic should select at most one visible-text success and one failure per source model by the smallest seeded identity hash. Missing outcome strata remain unavailable. Reuse the original unmodified and composite-operation outcomes. Add only missing prefix-only, outer-trim-only, trailing-trim-only, line-ending-only, and declared-wrapper-removal arms. At most six covers and thirty additional cover replays are planned. Record all constituent segment counts before execution. A wrapper-removal rule may use the declared transport convention, but must not consult the source payload or search for an offset that maximizes recovery. Keep all original zero-recovery rows unchanged.

The shared contract includes model and tokenizer hashes, quantization, backend version and build, prompt bytes and rendering, BOS/EOS behavior, rank ordering, mask hash, codec and framing metadata, gate threshold and rule, and segment boundaries. A canonical JSON fingerprint can reject incompatible receivers. It prevents unsupported decoding attempts and does not create cross-configuration robustness.

As a conditional argument, success across L required agreement events has probability equal to the product of their probabilities conditional on all preceding agreements. Replacing these terms by `(1 - epsilon)^L` requires an explicit identical independent-error approximation. A tokenizer version number does not supply epsilon. Segmentation changes, token-ID reassignment, vocabulary changes, and altered logits with unchanged tokenization are different perturbations.

## Proposed fallback

This is a protocol proposal, not an implemented recovery result. Identify the exact decoder configuration through a pre-agreed identifier or explicit metadata. Use independently decodable blocks with known boundaries and context resets. Attach an integrity check to each block. A checksum detects some corruption but is not authentication. Authentication requires an explicitly specified keyed mechanism outside RankCloak.

When damaged blocks can be located, treat them as erasures and apply an outer code within its stated correction radius. For a block code with minimum distance d, a conventional sufficient condition is `2e + s < d` for e unknown symbol errors and s known erasures, assuming the block alphabet and positions remain synchronized. Otherwise request retransmission over an acknowledged channel or choose a declared alternate transport.

Payload ECC cannot reconstruct an arbitrary wrong model's rank codebook or repair persistent insertion/deletion synchronization loss. Unrecoverable boundaries and context resets defeat the proposed erasure interpretation. Metadata, redundancy, repeated contexts, integrity fields, and retries consume capacity and can add detector cues. No improvement in cross-model recovery is claimed.

## Exact-model-aware detection

`revision_v3_analysis._finite_log_probabilities` reads saved generation probabilities from ordinary-control traces and the lead-in, forced, and tail arrays of RankCloak records. `surprisal_features_from_log_probabilities` computes count, means, spread, extrema, quantiles, threshold fractions, step variation, and first/last-half means. The logistic classifier standardizes these features and chooses regularization using validation ROC-AUC. Rank-only process features are excluded because comparable control ranks were absent in the historical primary corpus.

The evaluated access condition is therefore trace informed. Reconstructing those scores independently requires the exact model/backend, rendered prompt and segment contexts, original tokenization, boundaries or reset schedule, and probability convention. A visible-text-only scorer may retokenize differently and is a distinct attacker. It has not been shown equivalent to saved-trace scoring. The saved log probabilities are untempered full-vocabulary model probabilities. Ordinary controls were sampled using temperature 0.8 and top-p 0.95. Those are different distributions and must be named explicitly in the theory.

Forced ranks do not generally follow ordinary sampling probabilities. Bounding ranks limits the candidate set but does not enforce the ordinary distribution. ASCII serialization also structures payload digits. Forced spans, tails, and sampled skips may expose additional temporal structure. These mechanisms are consistent with the measured near-perfect discrimination, but individual feature contributions are not a proof of a unique causal mechanism.

For a defined stochastic message ensemble, sequence KL divergence decomposes into expected conditional KL terms when the required support and integrability conditions hold. A top-p reference can assign zero probability to a forced choice, making finite-KL claims invalid. Empirical AUC is not a KL estimate. This argument is a security-boundary explanation, not a universal impossibility theorem for linguistic steganography.

Detection alone is not decoding. An observer who also has the receiver's complete configuration may run the decoder. RankCloak does not supply confidentiality, authentication, or statistical indistinguishability against the evaluated informed attacker. No retraining of the existing neural detector matrix is needed for this explanation.

## Entropy failures

The reproducible analysis is `scripts/analyze_revision_v4_stage1.py`. It checks all 360 RankCloak records against the full 720-row source table and exports six failures, twelve matched ungated/moderate trajectories, and three identity-selected strict successes. The selected trajectories contain 6,682 positions. Selection and input hashes are recorded in `results/revision_v4/provenance/offline_manifest.json`.

All six failures use ASCII B16 with no token filter in their retained source lineage. Five are Qwen and one is Mistral. Each consumes fewer than L ranks within 6L positions. Eligible fractions range from 0.076823 to 0.151042. This establishes a budget shortfall. It does not establish why a particular linguistic context remained below threshold. Calibration on ordinary development traces does not promise a fixed eligibility rate on the embedding trajectory.

Pinned vocabulary-only detokenization reproduces the selected recorded text. Text-window byte offsets come from full prefix decoding. The longest below-threshold runs are listed in the six-case CSV. The following observations are exploratory readings of those aligned windows, not linguistic causal labels.

| Trial suffix | Half-open run | Length | Observed context |
| --- | --- | --- | --- |
| 2b5e682d0007193b56249891 | [382, 432) | 50 | A public-services explanation moves through police contact information into a numbered fire-department item. |
| 941eed91360afbdc8a26bf09 | [419, 582) | 163 | The text starts another formatted marketing-project update with a greeting and progress bullets. |
| 9ae150681b7e7347c8241d5a | [36, 75) | 39 | A water-supply explanation introduces a source-of-water heading and source examples. |
| a9ff67133eeaa189fa8be8b1 | [322, 357) | 35 | A library explanation lists programs and events before moving to user management. |
| c4160dc01512e365fbecb170 | [501, 768) | 267 | Another CRM project update expands budget details and closes with a feedback request and sign-off. The run persists to the budget limit. |
| cbb747058efe6a8b1fbd18ec | [350, 401) | 51 | A project-proposal response introduces a structured outline and an executive-summary heading. |

These contexts can contain fluent stretches while embedding stalls. The exact windows, comparable successful trajectories, rank pressure, and rolling progress remain available for the later response. There was no new generation and no extension of the historical budgets.

## Filter and complexity

The complete literal rules are exported from the predicate AST to `results/revision_v4/source_tables/filter_rules.csv` and included in the V4 main Methods through `v4_filter_methods.tex`. The source hash and related implementation hashes are in `provenance/filter_contract.json`. Historical masks and counts are retained in `provenance/historical_filter_masks.json`. The implementation itself is unchanged.

The predicate rejects empty pieces, U+FFFD, code points below 32 except LF and TAB, its exact case-insensitive substring list, stripped pieces beginning with a hash, and two or more backslashes. It does not reject whitespace in general, DEL, every Unicode control, or markup assembled across tokens. Individual decoding uses UTF-8 replacement. Decoder exceptions produce a marker rejected by the angle-bracket rule, while exceptions reaching mask construction produce a rejected empty string. Mask caching uses model object identity and filter name. Safe-text filtering is distinct from ordinary-sampler special-token exclusions and the isolated roundtrip mask.

Let V be vocabulary size, A the allowed-set size, T generated positions, L payload ranks, and S segments. Model cost depends on prompt length and the growing KV cache. The code performs full lexicographic sorting for requested-rank selection, costing O(A log A). It counts greater scores and lower-ID ties for known-token rank recovery, costing O(A). Array conversion and mask enumeration can still cost O(V). Full-vocabulary log-softmax and filtered entropy each require a vocabulary pass. Greedy tail and lead-in selection also call the full-sort selector. Bounded quality instrumentation performs additional rank-1 and rank-B selections per forced position, so one sort per token would understate actual constants.

Encoding cost is the sum of S prompt prefills, T incremental model evaluations, and the invoked sorting and diagnostic passes. Decoding similarly adds prompt prefills and observed-token replay plus counting. Direct-subword representation adds a payload-side LM pass, and inverse transcoding adds a payload-side generation pass. For m serialized bytes, ASCII B8 uses ceiling(8m/3) ranks and B16 uses 2m. Hex-nibble coding uses one rank per displayed hex character. Gate skips and tails increase T without increasing L. Weights, KV state, retained traces, and temporary vocabulary arrays all consume memory. The backend uses `logits_all=True`, so context-sized score storage must also be included.

A later comparator table must inspect the actual arithmetic-coding and grouping algorithms, including probability sorting, cumulative tables, finite precision, and grouping construction. No universal O(V) cost or cross-backend timing advantage is inferred here. Historical inclusive wall times remain distinct from these operation counts. Primary comparator sources include [Ziegler et al.](https://aclanthology.org/D19-1115/), [Shen et al.](https://aclanthology.org/2020.emnlp-main.22/), and [Zhang et al.](https://aclanthology.org/2021.findings-acl.268/). Their detailed algorithm audit remains Stage 2 work.

## Quantization

The V3 Q8 records contain `q8_replay_of_historical_q4_path` and `q4_q8_same_path_distribution_comparison`. Stage 1 checks that the Q4 and Q8 traces use identical context and observed-token arrays across 1,920 pairs. These comparisons are distinct from each quantization generating its own sequence.

For local logits z and perturbations delta, two candidates can reverse order when their perturbation difference exceeds their original logit gap. A gap greater than 2 epsilon is sufficient to preserve that pair's order if every absolute perturbation is at most epsilon. This is a conditional mathematical bound, not an estimated backend error bound. Token-ID tie-breaking resolves exact ties and cannot prevent nearby unequal scores from crossing. After a different token is generated, independent paths acquire different histories and can diverge further. Seeded ordinary sampling can also change outcomes when probabilities change.

Observed-token replay feeds the received cover tokens into the model. It does not feed recovered payload symbols back into that cover history. Wrong ranks alone therefore do not imply the same autoregressive feedback mechanism as independently generated paths. Gate decisions, lead-in regeneration, span shifts, and retokenization introduce separate synchronization mechanisms.

The retained diagnostics include entropy, observed ranks, greedy IDs, and probability gaps to the greedy choice, but not full logit vectors or requested-rank neighbor margins. Existing evidence supports a same-history versus independent-path explanation now. If local margin measurements are needed, preselect one bounded trial per eight-class by two-codec cell, cap replay at 64 positions, and process both quantizations. This is at most 2,048 observed-token evaluations across 32 replay prefixes. No full quantization matrix is needed.

Within-quantization saved-ID recovery is not cross-quantization decoding. Detector transfer is also not decoding. For bounded Q4 messages, the saved Q8 observed ranks on the Q4 path may support an additional offline Q4-to-Q8 codec diagnostic. It must reconstruct and compare original bytes under a separately declared endpoint. The reverse direction requires matching Q4 scoring of received Q8 tokens if not already retained. Neither endpoint is asserted by the present audit.

## Bibliography and deposit

The sixteenth V3 bibliography entry is `Cai2025EntropyGuidedWatermarking`. The [arXiv primary record](https://arxiv.org/abs/2504.12108) lists the April 2025 preprint and no journal reference. Searches did not verify a formal publication. That does not prove that none exists. The V4 copy retains it explicitly as historical motivation and adds the peer-reviewed [Lee et al. ACL 2024 paper](https://aclanthology.org/2024.acl-long.268/) for selective watermarking based on an entropy threshold. Watermark detection does not imply arbitrary-payload recovery. No watermark theorem is used as a RankCloak recovery guarantee.

The [original published Zenodo record](https://zenodo.org/records/21987450) identifies DOI 10.5281/zenodo.21987450 and concept DOI 10.5281/zenodo.21987449. The downloaded 39,160,391-byte ZIP matches the published MD5. All 411 files listed in its embedded manifest pass verification. V3 and V4 configurations and results are absent from that version. Its internal offline-candidate status records its assembly state and does not negate later publication.

The concept record now resolves to [version 2.0.0](https://zenodo.org/records/22555497), DOI 10.5281/zenodo.22555497, published on 6 September 2026. Its 332,331,547-byte ZIP matches the public checksum. All 8,342 archived files match Git commit `ce853d42d6ba64065cb63c6bdfc0d825c62734cd`, with no missing files from that commit. It includes the V3 extension and 65 files under `paperV3`, despite a public description saying manuscripts are excluded. V4 is absent. Coverage and byte checks are retained in `provenance/archive_v2_coverage.json`. A future version should retain V3 and add completed V4 evidence, source identities, reproduction commands, environment and license records, and verified portable manifests. Its description must match its contents. No external deposit or submission was performed.
