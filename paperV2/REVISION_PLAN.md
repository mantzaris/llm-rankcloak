# RankCloak major-revision implementation plan

This V2 is a replacement of the partial-pilot narrative in `paperV1`, not an appendix to it. The scientific center remains RankCloak's deterministic next-token-rank transcoding; neural detection is an adverse evaluation of that method rather than a new primary contribution.

## Manuscript structure and working word budget

| Component | Target words | Revision role |
|---|---:|---|
| Abstract | <=200 | Contribution, conditional recovery, expanded evaluation, fragility, and detectability |
| Introduction | 750--900 | Motivation, related work, contribution boundary, evaluation questions |
| Results | 2,300--2,500 | Seven evidence-led subsections using completed revision evidence |
| Discussion | 750--900 | Synthesis, adverse evidence, uncertainty, and scientific value |
| Responsible use and limitations | 300--450 | Consolidated claim boundaries |
| Journal-defined main text | 4,100--4,450; hard ceiling 4,500 | Introduction + Results + Discussion + responsible-use section |
| Methods | 2,200--3,000 | Reproducible core; detailed configurations and diagnostics in the Supplement |

The main Results are organized as: (1) expanded evaluation and exact-copy recovery; (2) capacity, rank bounds, and cover-length overhead; (3) exact-copy replay and transmission fragility; (4) automated readability and model-based cover quality; (5) ablations, dynamic stopping, and topic effects; (6) neural steganalysis; and (7) computational overhead and generalizability. Supplementary Notes S1--S10 hold full methods, matrices, secondary endpoints, diagnostics, and claim-boundary material.

## Main display allocation

The main manuscript uses six display items: Table 1 (design and recovery), Figure 1 (capacity and tail overhead), Figure 2 (transmission fragility), Figure 3 (automated readability diagnostics), Table 2 (selected reviewer-requested analyses), and Figure 4 (neural detectability). Four complete figures are assigned to Supplementary Figures S1--S4: robustness, ablations, detector performance, and computational overhead.

## Major scientific changes from V1

- Replace all partial-pilot sample sizes, outcomes, figures, and conclusions with the sealed completed study.
- Define exact recovery only under saved-token exact-copy replay, then separately measure visible-text retokenization and controlled channel perturbations.
- Expand to 480 public deterministic cryptographic test vectors, eight artifact classes, three open-source 7B--8B model families, 18 English prompt templates in six categories, and secondary Spanish and Simplified Chinese trials.
- Add grouped uncertainty, multiplicity control, mixed-model diagnostics, theory checks, ablations, topic effects, computational overhead, and explicit unavailable-work accounting.
- Add established automated readability diagnostics while stating that no participant study or human rating exists.
- Add two neural detector architectures across matched and held-out regimes, with leakage, near-duplicate, sensitivity, and same-device repeatability audits.
- Moderate all practical, security, naturalness, robustness, deployment, and multilingual claims.

## Source-of-truth policy

Numerical statements are checked against authoritative CSV/JSON artifacts named by the sealed evidence package. Confirmatory, supporting, secondary, exploratory, diagnostic, unavailable, unresolved, and external-gate evidence retain those classifications. Narrative summaries are not used to override an authoritative table. A minor rounding discrepancy in a narrative token-filter summary is resolved in favor of the canonical contrast CSV and recorded in the validation audit.

## External boundaries

No human-participant study, real-world deployment, new generation run, model fitting, bootstrap, mixed model, neural training, or fresh GGUF replay is performed in this revision. DOI deposition is intentionally deferred at the author's instruction for this drafting stage.

## Implemented budget and output

The final LaTeX-aware counts are 184 words in the Abstract, 811 in the Introduction, 2,500 in Results, 764 in Discussion, 300 in Responsible use and limitations, and 2,256 in Methods. The journal-defined main-text total is 4,375 words. The compiled review manuscript is single-column and line-numbered. Its display allocation is the planned four figures and two tables; the Supplement contains Figures S1--S4 and Tables S1--S17.
