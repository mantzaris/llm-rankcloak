# Theory inlining and section order

Started from `ea780507574583c6c8249db31b019fd8440a6039` on `main`. Newer author wording in Discussion, limitations and availability was preserved exactly and included in the rebuilt current manuscript. The initial author patch is retained separately from the structural change.

The full text of `v4_stage2_theory.tex` is now inline in `main4.tex`, immediately before Discussion. Its wording, mathematics and `sec:v4-boundaries` label are unchanged. The former input file is retained unchanged for historical workflows but is no longer a manuscript dependency. Responsible use and limitations is now a subsection of Discussion, making Discussion the final scientific section before availability and declarations.

Response location terminology now identifies the relocated analysis under Results. All 13 quotations and 16 substantive answers are retained. The response was rendered twice with identical output. All 147 labels resolve and numbered equations, algorithms, figures and tables retain their numbers.

All four PDFs rebuilt without warnings or unresolved references. Page counts remain 23, 48, 8 and 1 for the main manuscript, supplement, response and cover letter. The editable manuscript passed its clean dependency build at 21 pages with 183 verified input paths. Seven changed pages were visually checked with no layout defects. The supplement and cover letter retain identical text and geometry.

Current outputs are [main4.tex](../paperV4/scientific_reports/main4.tex), [main4.pdf](../paperV4/scientific_reports/main4.pdf), [main4_submission.tex](../paperV4/scientific_reports/main4_submission.tex), [supplementary4.pdf](../paperV4/scientific_reports/supplementary4.pdf), [response_to_reviewers_v4.pdf](../paperV4/response/response_to_reviewers_v4.pdf) and [cover_letter_v4.pdf](../paperV4/cover_letter/cover_letter_v4.pdf). New build and preservation records are under `results/revision_v4/theory_inline/`. Earlier receipts remain unchanged. No experiments, GPU work, deposit or submission action occurred.
