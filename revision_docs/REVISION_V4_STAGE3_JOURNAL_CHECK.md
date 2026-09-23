# Scientific Reports revision requirements

Checked against the current [Scientific Reports submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines) and [revision checklist](https://www.nature.com/documents/srep-checklist-for-revised-submissions.pdf) on 23 September 2026. The original decision letter is retained in paperV4/response/requests.txt. No other journal or conference limits were applied.

The journal strongly recommends concise articles, ideally at most 11 typeset pages and about 4,500 main-text words, excluding Abstract, Methods, References and figure legends. These are not stated as universal hard page/word caps. The guidance specifies a title of at most 20 words, an abstract of at most 200 words and eight main figures/tables. It recommends page/line numbering and encourages the journal LaTeX template. The existing template was retained with concrete revision-format corrections.

The revision checklist requires editable single-column main text, separate figure uploads, editable main tables, a separate supplement, a matching title/author list, a Data Availability section, contributions and competing interests. References must be compiled into the LaTeX submission or supplied as the allowed .bbl companion. A .bib file alone is insufficient. End legends and tables are supplied in the generated upload source. The review letter additionally requires accurate claims, a point-by-point response and DOI-backed code coverage.

## Measured candidate

Counts come from results/revision_v4/stage3/validation/document_audit.json and its TeXcount logs.

| Item | Actual candidate |
| --- | --- |
| Title | 14 words, unchanged title |
| Abstract | 194 whitespace-delimited words, no citations |
| Main prose | 4,457 TeXcount text words in Introduction, Results, Discussion and limitations |
| Word-count qualification | This count excludes headers and all captions. Including the two Results table captions adds 126 words, giving 4,583. Mathematical expressions are counted separately. This is an explicit convention, not a claim about the portal's counter. |
| Complete source | 9,903 TeXcount text words including Abstract, Methods and other sections, excluding captions and headers |
| Main display items | 5 figures and 3 tables. Three existing numbered algorithm floats are reported separately. No claim is made that the checklist explicitly classifies algorithms. |
| Main bibliography | 48 active references |
| Main PDF | 25 pages, 672,893 bytes |
| Supplement | 40 pages, 16 figures, 34 tables, 1,013,737 bytes |
| Response | 8 pages, 16 finished answers, 13 exact review blocks |
| Cover letter | 1 page |
| Single-file upload source | Successfully builds to 23 pages with figures supplied separately |

The 25-page author-review PDF is not a forecast of the journal's final typeset length. Its Methods, readable algorithms and supporting definitions are not compressed to reach an arbitrary page target. The supplement remains far below the stated 50 MB guidance. Its S1–S19 navigation is preserved. Numbered-note restrictions from other Nature-family journals were not imported.

The manuscript and supplement have visible line numbers and Arabic footer page numbers. Response locations use actual compiled pages and section/table names. No exact line number was invented. The cover letter supplies author contact details, scientific fit and the retained empirical limitation. Author confirmation of prior editorial discussions, reviewer suggestions/exclusions and portal metadata remains in the checklist.

All four final builds have no missing references, clipping, blank pages or layout warnings. Every page was visually inspected. The single candidate label must be removed only after author approval. The real outstanding requirements are the author's R1.4 judgment and final approval, actual verified public V4 DOI coverage, subsequent document/DOI rebuild, and the journal portal handoff. The author should confirm how the portal treats the three algorithm floats if it requests a combined display count. A local ZIP is not the required public deposit.
