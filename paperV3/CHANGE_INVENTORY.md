# paperV3 change inventory

This inventory maps the authoritative comments in `response/requests.txt` to the focused V3 revision. The request file is preserved byte-for-byte.

Exact revised title: `RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text`

| Editor or reviewer item | Main-manuscript change | Supplement and response coverage | Validated V3 evidence |
|---|---|---|---|
| Editorial title request | Replaced the title and PDF metadata with the exact 14-word title | Synchronized the supplement, response, cover letter, and checklist; journal-system entry remains manual | Title consistency and PDF metadata checks |
| Accurate reporting and complete limitations | Retained the V2 scientific structure; replaced superseded detector estimates and added one compact V3 Results subsection and table | Notes S11–S15 give condition-level methods, counts, provenance, and remaining scope | `results_for_manuscript.md`, `limitations_for_manuscript.md`, `claim_evidence_matrix.csv` |
| DOI-assigning repository | Preserved the current GitHub and Zenodo statements without claiming the final V3 revision is already archived | Response and checklist state the remaining manual synchronization steps | Existing DOI plus `run_manifest.json`; no Zenodo mutation in this task |
| Reviewer 2 overall assessment | Kept artifact concealment, recovery, and steganalytic classification as distinct endpoints; retained 88/144 visible-text recovery and zero markdown, paraphrase, and cross-model recovery | Overall response is brief; detailed qualifications appear under concern 1 and Notes S1, S4, and S15 | `source_tables/recovery_mode_comparison.csv` |
| Concern 1 shared configuration | Added four Introduction sentences, focused recovery-Methods wording, and one short Discussion paragraph; did not broadly recast the paper | Explains predeclared model/tokenizer/quantization/rendering/rank/filter/codec/span configuration, prompt as shared context rather than a cryptographic key, and saved tokens as one evaluated condition | Recovery implementation and V3 methods handoff |
| Concern 1 saved-token interpretation | States that retained IDs hold tokenization/history fixed while the decoder reconstructs contextual ranks and inverts the codec; distinguishes token-preserving, rendered-text, and transformed-text delivery | Most nuance is in the response letter and Supplementary Note S1 | `recovery_mode_comparison.csv`, validated primary recovery records |
| Concern 2 strict deduplication | Replaced historical detector claims with the 15,838-row pre-feature component split and zero declared cross-partition leakage | Note S11 gives normalization, threshold, component construction, counts, and limits | Deduplication audit and `manuscript_tables/deduplication.tex` |
| Concern 2 human-authored controls | Reports 250 Dolly responses as a bounded secondary corpus with domain-sensitive false positives | Notes S11–S12 document selection, matching, audit, AUCs, and exact false positives; never described as a participant study | `detector_human_metrics.csv` and human-control provenance |
| Concern 2 unseen families and adaptive attackers | Reports matched and leave-one-family-out TextCNN, DeBERTa, and exact-model-aware results; replaces the main detector figure | Note S12 gives full estimates and threat-model limits | `detector_main_metrics.csv`, `detector_holdout_macro.csv`, detector figure |
| Concern 2 low-FPR analysis | Describes validation-frozen thresholds, exact counts, matched 0.1% support, calibration drift, and unavailable smaller-set estimates | Note S12 gives exact 1% and 0.1% results; Notes S13–S14 retain unavailable conditions | Detector metrics and `manuscript_tables/detector_low_fpr.tex` |
| Concern 2 quantization | Reports paired Q4_K_M/Q8_0 recovery, a non-significant 2.71-point difference, substantial path divergence, and strong bidirectional detector transfer | Note S14 contains counts, grouped interval, exact McNemar test, sensitivity, transfer, and one-model/two-quantization scope | Quantization summaries, transfer metrics, generation validation |
| Concern 3 Cai et al. and entropy gating | Adds the verified Cai–Ding–Tao citation, distinguishes watermark marking from chosen-payload recovery, and reports the capacity–length–detectability tradeoff | Note S13 provides calibration, replay-consistent gating, conditional capacity, six retained completion failures, and detector results | Entropy tables, figures, and generation validation |
| Consolidated limitations | Retains no secrecy proof, no universal-undetectability claim, weaker visible transport, failed transformations, no participant study, bounded controls/families/quantizations/attackers, low-FPR limits, and entropy completion/capacity limits | Note S15 consolidates provenance and scope boundaries | `limitations_for_manuscript.md`, validation reports, run manifest |

## Accessibility and author-review pass

- Replaced the main and Supplementary Information abstracts with accessible high-level summaries while retaining the validated recovery, robustness, detection, and entropy tradeoff conclusions.
- Defined the human-written Dolly comparison, low false-positive-rate terminology, generating-model-aware detector, and approximately 4-bit/8-bit quantized model formats at first use.
- Applied normal punctuation to the affected Results and response headings and clarified the requested response and cover-letter wording.
- Added `paperV3/diffs.pdf`, a clearly labeled internal author-review comparison of the complete V2 and final V3 main manuscript and Supplementary Information; reproducibility details are in `paperV3/diffs/README.md`.

All new manuscript-package files are under `paperV3`; `paperV2`, `results/revision_v3`, computational code, tests, configurations, and provenance are outside the edit scope.
