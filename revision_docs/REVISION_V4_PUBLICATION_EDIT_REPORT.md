# V4 targeted publication edit

All 51 requested replacements were applied without content conflicts. The starting checkout was clean on `main` at `303a2c4442992867fed0adeb4b12061cf001c1b7`. The ending document and validation commit is `434e5f35d973e73e350388248c275d95e121fd7a`. A subsequent report-only handoff commit records this report without changing those document bytes. Its identifier can be retrieved with `git log -1 --format=%H -- revision_docs/REVISION_V4_PUBLICATION_EDIT_REPORT.md` and is supplied in the final handoff.

Applied IDs are M01–M22, S01–S24 and P01–P05. Every OLD block matched its supplied SHA256 and occurred exactly once. The [replacement allowlist](../results/revision_v4/publication_edit/authorized_replacements.json) retains the exact old and new text and the reason for each change. The edits explain the representation contribution and replay conditions more directly, distinguish detection from surface-form concealment, consolidate the Discussion and limitations, and replace the specified internal audit terminology. M07 restores the Q4_K_M identifier. M17 explicitly restores the requested substantive Codex and ChatGPT disclosure in Methods without asserting that author review is complete.

Nine additional repairs replace a comma with a semicolon in adjacent ordinary prose. No words, numerical values or captions were changed by these additional repairs.

| ID | Location and repaired clause boundary |
| --- | --- |
| A01 | Segmented method, “token sequence; it is not” |
| A02 | Detector Methods, “threshold selection; it is a corpus comparison” |
| A03 | Statistical Methods, “models were singular; no undeclared” |
| A04 | Filter ablation, after the effective-rate interval, before “both intervals included zero” |
| A05 | Detector Results, “0.1% reporting; at that target” |
| A06 | Entropy Results, “114/120 payloads; its six maximum-budget” |
| A07 | Quantization Results, after the paired interval, before “paired outcomes” |
| A08 | Overhead Results, “harness; CPU time” |
| A09 | Overhead Results, “peak allocated VRAM; DeBERTa-v3-base times” |

S24 was checked against the retained V3 validation JSON and the two specified Python sources. `failure_record_count = 0` counts files under `generation_root / "failures"`, written by the execution exception handler. It does not count budget-exhausted payloads. All six incomplete strict-gate records, their original rank counts and 768/384-token budgets, and the 114/120 completion denominator remain unchanged. The [S24 receipt](../results/revision_v4/publication_edit/validation/s24_evidence.json) identifies and hashes the inspected sources. No generation was rerun.

The current TeX and included files remain authoritative. Current build, response-render and PDF-review scripts now write only to `results/revision_v4/publication_edit/`. No manuscript renderer was run and no scientific template or analysis logic was changed. A separate validator checks this pass against the explicit replacement allowlist. The earlier presentation validator and all historical evidence remain unchanged. [DOCUMENT_BUILD.md](../paperV4/DOCUMENT_BUILD.md) gives the current commands.

The [validation receipt](../results/revision_v4/publication_edit/validation/publication_edit_check.json) records the completed checks. The author-edited abstract, title, author information, typography, availability statements and standard declarations are unchanged. All existing source labels and citation keys are preserved, with only the authorized disclosure label added. Sixteen main numbered equations, the supplementary equation, both main algorithms, Algorithm S1 and the full main-text filter specification are unchanged. Numeric tables, figure files, literal examples, the judging rubric, calibration responses, both historical semantic-pilot results and all retained data and code remain unchanged. The numerical value set across the expanded scientific sources is preserved, and the contextual estimates and intervals agree with the retained summary. No experimental or statistical analysis was rerun.

All 13 exact quotation blocks and 16 substantive answers are preserved. Only compiled response locations changed. None of the renamed supplementary headings was cited by its old title in an editable answer. Two successive ordinary response renders produced identical source bytes. The cover-letter source is unchanged. The editable manuscript was regenerated from the current source, including its author-edited abstract, embedded bibliography and three editable tables. Its clean build checked 183 input paths and required no project dependency outside the temporary build directory.

Final LaTeX passes have no undefined references or citations, missing-glyph reports, duplicate destinations or layout warnings. All 148 compiled labels resolve, existing equation/algorithm/figure/table numbers remain stable, internal PDF link targets resolve, and all 61 supplementary bookmarks land on the appropriate headings. No banner, printed contents or Reading guide was restored. Raw tool-output logs retain their native whitespace; the document and workflow source diff passes the whitespace check.

All 80 PDF pages received automated text and page-bound checks. Actual visual inspection covered all 58 changed or reflowed pages, plus the unchanged first pages of the supplement, response and cover letter, for 61 pages total. No clipping, overlap, broken figures, split captions or accidental blank pages was observed. The remaining unchanged pages were not newly claimed as visually reviewed. The [page-review receipt](../results/revision_v4/publication_edit/validation/pdf_review.json) records the exact pages and PDF hashes.

| Deliverable | Starting pages | Final pages |
| --- | --- | --- |
| [Main manuscript](../paperV4/scientific_reports/main4.pdf) | 23 | 23 |
| [Supplementary Information](../paperV4/scientific_reports/supplementary4.pdf) | 48 | 48 |
| [Response](../paperV4/response/response_to_reviewers_v4.pdf) | 8 | 8 |
| [Cover letter](../paperV4/cover_letter/cover_letter_v4.pdf) | 1 | 1 |
| [Editable manuscript](../paperV4/scientific_reports/main4_submission.tex), clean dependency build | Stale source at baseline | 21 |

Results begin on main PDF page 11. Final source/PDF hashes and build logs are in the validation receipt and its linked build records. The temporary editable-build PDF was not retained as a separate deliverable.

Additional GPU work was zero seconds. Experimental changes were zero. No archive preparation, deposit, DOI update or journal submission occurred. The existing V3-only DOI coverage and outstanding V4 deposit status are unchanged. Author review should focus on the requested Discussion and limitations replacements and the factual scope of the restored assistance disclosure. This editorial completion does not establish journal readiness or guarantee acceptance.
