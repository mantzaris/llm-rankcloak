# Revision V3 computational experiment design

The predeclared design is tracked in revision_docs/REVISION_V3_COMPUTATIONAL_PLAN.md and machine-readable configurations under configs/revision_v3/. This result namespace never modifies V2 source data.

The detector corpus is normalized with Unicode NFKC, case folding, and whitespace collapse before SHA-256 exact matching. Complete matched pairs implicated by an exact duplicate are removed. Character-boundary TF-IDF cosine similarity over 3--5 character n-grams defines near duplicates at the predeclared threshold 0.95. Payload groups linked by any near-duplicate edge form an immutable connected component. These components are assigned to train, validation, and test before feature extraction or detector fitting.

Component assignment targeted 60/20/20 train/validation/test allocation; indivisible near-duplicate components and factor balancing produced the exact realized counts recorded in the deduplication summary. Leave-one-model-family-out tests use non-target families from train and validation components and the target family only from test components. Low-FPR thresholds are selected from validation labels and then frozen. The 0.1% operating point is reported only when both validation and test contain at least 1,000 negative observations.

The human-authored secondary control uses the pinned Databricks Dolly 15k v1.0 source, automatically screened by the repository pipeline, deduplicated, checked against the detector corpus, and matched by available prompt topic and text length. It is a computational control, not a human evaluation. RankCloak-versus-human discrimination can include generic machine-versus-human cues.

The model-aware detector uses only saved token-log-probability summaries available for both RankCloak and matched ordinary controls. Rank-only fields are excluded from classifier inputs because corresponding ordinary-control ranks were not stored. Topic variability is a paired comparison of the fixed-codec segmented single-topic and multi-topic conditions; it is reported separately from recovery mode.

<!-- revision-v3-model-backed-design:start -->

The entropy-gate matrix used 120 paired experimental cells and three gate levels, producing 360 RankCloak and 360 length-matched ordinary-control generations. Eighteen independent ordinary top-p calibration traces fixed model-specific median and 75th-percentile thresholds before evaluation. Ineligible positions were ordinary top-p samples, not greedy or rank-1 choices. RankCloak and control seeds were each shared across gate levels within a cell. Both fixed-payload and fixed-token-budget estimands were computed from the same records.

The matched quantization matrix used all 1,920 historical Qwen Q4 rows and 1,920 newly generated Q8 counterparts. Historical ordinary-control seeds were read from raw Q4 records. Non-quantization contracts were hash-bound across every pair. Predeclared payload splits support Q4-to-Q8, Q8-to-Q4, and pooled held-payload detector evaluations. Generated detector corpora were locked-partition deduplicated before feature extraction, with complete matched-pair and payload-group removal used to resolve any cross-boundary duplicate component.

The two model-backed studies completed on the local RTX 5000 Ada with exact pinned GGUFs and CUDA-enabled llama-cpp-python 0.3.23. Every outcome, failure status, runtime, and peak-memory value is retained in atomic JSON ledgers; aggregate results are derived rather than manually copied.

<!-- revision-v3-model-backed-design:end -->

The recovery-mode comparison reuses only the 144-trial V1 robustness sample after verifying its source checksum. Saved token IDs, greedy lead-in regeneration, and visible-text detokenization/retokenization are separate outcomes and are not extrapolated to the full 6,480-trial corpus.
