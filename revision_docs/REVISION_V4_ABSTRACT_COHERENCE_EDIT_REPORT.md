# V4 abstract and contextual-evaluation editorial correction

The pass starts at `c614f754faf3bb1a89d79689ca88342078759c15` on `main`. The only initial author edit removed the abstract's human-perception sentence. The expressly requested V3 restoration preserves that intent. Its initial patch is retained with the new receipts. There were no content conflicts or unrelated edits.

The V3 abstract is restored exactly except for the specified description of the tested transport operation. This is the complete sentence-level diff. Exact source comparison also passes.

```diff
--- V3 abstract (one sentence per display line)
+++ V4 abstract (one sentence per display line)
@@ -5 +5 @@
-Recovery from rendered cover text succeeded in 61.1 percent of the tested robustness sample, whereas Markdown transfer, paraphrasing, and decoding with another model failed.
+Recovery from rendered cover text succeeded in 61.1 percent of the tested robustness sample, whereas the tested blockquote and trimming operation, paraphrasing, and decoding with another model failed.
```

The targeted changes are complete.

- Items 2–6 update the Introduction, contextual Methods, Discussion, limitations and diagnostic captions. The completed blinded automated study is presented as evidence of contextual coherence and acceptability. Its ratings remain identified as model-based.
- Items 7A–7I update Note S5, the two scope-table cells and Note S20. The full rubric, calibration counts/errors, selection, control reuse, missingness, aggregation and bootstrap procedures are unchanged. One provenance sentence remains in main limitations and one in Note S20.
- Corresponding local edits update the main contextual-table caption, the supplementary strata caption and example labels. Literal examples are untouched. The response introduction, editor synthesis, R1.4 closing terminology and cover letter foreground the completed automated evidence. All other answers and scientific findings remain intact.
- The removed sentence was the only current use of `VanDerLee2019BestPractices`. Its uncited bibliography entry remains in the source database; the main compiled bibliography now has 48 references. No replacement prose was invented to retain it. G-Eval remains cited.

The current builder, response renderer and page-review helper now write only to `results/revision_v4/abstract_coherence_edit/`. Current TeX and editable answer records remain authoritative. Earlier scientific and document receipts, historical workflows, disclosure decisions, availability wording and submission status are unchanged. No banners or front matter were added.

Preservation and build checks passed. Scientific body values and their ordering, intervals, denominators, equations, algorithms, filter rules, figure assets, rubric and literal examples were compared with the baseline. Table values are unchanged, with only the two requested scope cells edited. All 13 exact review blocks and 16 answers remain present. All 147 compiled labels retain their identifiers and numbered values; response locations are refreshed. The 61 supplementary bookmarks and internal PDF destinations resolve. A second response render was byte-identical.

The final LaTeX passes have no warnings or unresolved references. The generated editable manuscript matches the restored abstract and current prose, embeds its bibliography and editable tables, and builds in a clean temporary directory with 183 verified input paths. All 80 PDF pages passed automatic text/bounds checks. All 35 changed or reflowed pages were visually inspected with no layout defect found. Page counts are unchanged.

| Document | Before | After | File |
| --- | --- | --- | --- |
| Main manuscript | 23 | 23 | [main4.pdf](../paperV4/scientific_reports/main4.pdf) |
| Supplement | 48 | 48 | [supplementary4.pdf](../paperV4/scientific_reports/supplementary4.pdf) |
| Response | 8 | 8 | [response_to_reviewers_v4.pdf](../paperV4/response/response_to_reviewers_v4.pdf) |
| Cover letter | 1 | 1 | [cover_letter_v4.pdf](../paperV4/cover_letter/cover_letter_v4.pdf) |
| Editable source, clean build | 21 | 21 | [main4_submission.tex](../paperV4/scientific_reports/main4_submission.tex) |

The reported comparison remains 537 complete pairs from 576 selected messages, 90.3% encoded acceptance versus 96.6% ordinary acceptance, and a paired difference of −6.3 percentage points with 95% interval [−8.7, −3.9]. These are automated contextual ratings. Neither equivalence nor indistinguishability is asserted.

[Validation receipts](../results/revision_v4/abstract_coherence_edit/validation/checks.json) and the [exact edit ledger](../results/revision_v4/abstract_coherence_edit/authorized_edits.json) retain the checks and changes. GPU time and experimental changes were zero. No archive, deposit or submission action occurred. The author should review the restored abstract and revised framing; this editorial completion does not declare journal readiness. The handoff commit is the commit containing this report, with its actual SHA supplied in the handoff message rather than embedded in itself.
