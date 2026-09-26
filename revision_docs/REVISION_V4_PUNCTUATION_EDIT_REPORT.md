# V4 punctuation cleanup

The pass starts from clean `main` at `dec80d2daff76d2fd3463950951b6012a04d6057`. It simplifies ordinary prose in the current manuscript, supplement and editable reviewer answers. The exact edit ledger records a net removal of 84 semicolons, 24 colons and 96 prose hyphens across 113 source blocks and 14 answer fields. Long clauses become separate sentences, and selected compound phrases become ordinary wording. Necessary technical punctuation remains.

The author-edited abstract, typography, title and author details are unchanged. All numeric values remain in their original order. Equations, algorithms, table bodies, figure files, complete filter rules, judging rubric and literal examples are unchanged. All 13 original review quotations and all 16 substantive answers remain present. Availability statements and submission status are unchanged. The assistance disclosure remains removed as requested.

The current builder and response renderer write to `results/revision_v4/punctuation_edit/`. Historical scripts, experiments and earlier receipts are preserved. The response was rendered twice with identical output after compiled locations were refreshed. The regenerated editable source matches the current manuscript, embeds its bibliography and editable tables, and passed the clean dependency build with 183 verified input paths.

All builds passed without LaTeX warnings, unresolved references, missing glyphs or duplicate destinations. The validator checked 147 compiled labels and all 61 supplementary bookmarks and their destinations. All 80 PDF pages passed automatic text and page-bound checks. Visual inspection covered all 70 changed or reflowed pages through the rendered contact sheets, with no clipping, overlap or broken figures/tables observed. The cover-letter prose and geometry are unchanged.

| Deliverable | Before | After | File |
| --- | --- | --- | --- |
| Main manuscript | 23 | 23 | [main4.pdf](../paperV4/scientific_reports/main4.pdf) |
| Supplement | 48 | 48 | [supplementary4.pdf](../paperV4/scientific_reports/supplementary4.pdf) |
| Reviewer response | 8 | 8 | [response_to_reviewers_v4.pdf](../paperV4/response/response_to_reviewers_v4.pdf) |
| Cover letter | 1 | 1 | [cover_letter_v4.pdf](../paperV4/cover_letter/cover_letter_v4.pdf) |

The [editable submission source](../paperV4/scientific_reports/main4_submission.tex) produced 21 pages in its isolated build. Current commands are in [DOCUMENT_BUILD.md](../paperV4/DOCUMENT_BUILD.md), and the [validation receipt](../results/revision_v4/punctuation_edit/validation/checks.json) records the final hashes. The containing Git commit identifies this editorial handoff.

GPU work and experimental changes were zero. This pass does not publish a deposit or submit to the journal.
