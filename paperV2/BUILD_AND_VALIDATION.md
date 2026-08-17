# Build and validation record

Validation date: 2026-08-16. Branch: `main`. HEAD and read-only `origin/main`: `cfa19d3d41af8f869837c087ec6f33fcd9f51901`.

## Repository baseline and scope

- Starting `git status --short --untracked-files=all`: clean.
- `paperV1/` had no tracked or untracked changes at the start, so revision work proceeded.
- Final scope check: every worktree addition is under `paperV2/`; no tracked file outside that directory changed.
- `git diff --exit-code -- paperV1` and `git status --short --untracked-files=all -- paperV1` both produced no output. Because the starting tree was clean, this confirms that `paperV1/` remains byte-for-byte at the starting Git state.
- No commit, staging, push, pull, reset, clean, checkout, rebase, merge, tag, or release operation was performed.
- The experiment, result, configuration, release-input, human-study, operations, and source-code trees were read only.

## Tool versions

- latexmk 4.83 (31 January 2024)
- pdfTeX 3.141592653-2.6-1.40.25, TeX Live 2023/Debian
- BibTeX 0.99d, TeX Live 2023/Debian
- TeXcount 3.1.1
- Poppler `pdftoppm` and `pdfinfo` 24.02.0

## Build commands and passes

All commands were run from the relevant source directory and wrote intermediates to the dedicated `paperV2/build/` tree.

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../build/main main2.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../build/supp supplementary2.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../build/response response_to_reviewers.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../build/cover_letter cover_letter.tex
```

latexmk ran the required dependency-driven passes, including BibTeX and repeated pdfLaTeX passes for the main manuscript. Each final target reported `All targets ... are up-to-date`. The final logs contain no LaTeX/package warning, undefined citation/reference, multiply defined label, overfull/underfull box, fatal error, or emergency-stop match. The main BibTeX log reports `warning$ -- 0`.

## Final PDFs

| Output | Pages | SHA-256 | Status |
|---|---:|---|---|
| `scientific_reports/rankcloak_scientific_reports_manuscript_v2.pdf` | 15 | `4bf159dc655ed53525a067e014ad1e91fcb2b63375c8cfca03a4038c2740c6cb` | Clean, single-column, line-numbered review manuscript |
| `scientific_reports/rankcloak_scientific_reports_supplementary_v2.pdf` | 19 | `181ee20c66e6af03545a19875589e71b8cec64aeb9c50744aa04eeb52e6d469f` | Clean; title/author list on first page; landscape detail tables rotate correctly |
| `response/response_to_reviewers.pdf` and versioned copy | 7 | `cc25b5002329eb1aafcf6325e2dcf4034ff8e6f2e0ba0a937dd0893e94afc9f9` | Clean; final manuscript/Supplement page and line references included |
| `cover_letter/cover_letter.pdf` and versioned copy | 1 | `96c798ebd7b2cf5ba81687f67a19843b0cd29ea6de5ec47d81f32b1bedc5145f` | Clean one-page letter |

The required unversioned response and cover PDFs are byte-identical to their clearly named versioned copies.

## Citation, reference, and source checks

- No unresolved citations or cross-references and no duplicate labels were found.
- The final bibliography contains no arXiv/eprint research entry. The only BibTeX `techreport` entries are the authoritative NIST standards documented in `CITATION_AUDIT.md`.
- All eight `\\includegraphics` calls use local `figures/*.pdf` paths. No LaTeX source references a figure through `results/` or `paperV1/`.
- No absolute `/home/` or `/tmp/` path occurs in publication sources or provenance manifests. The relative `results/...` paths in `FIGURE_PROVENANCE.csv` are intentionally recorded source provenance, not LaTeX figure references.
- The final main allocation is four figures and two tables. The Supplement contains Figures S1--S4 and Tables S1--S17, plus machine-readable detector split data.
- The response matrix, response letter, main text, and Supplement use consistent evidence classifications and limitations.

## Word counts

TeXcount produced: Abstract 184; Introduction 811; Results 2,500; Discussion 764; Responsible use and limitations 300; journal-defined main-text total 4,375; Methods 2,256; four main figure legends 359; full Supplement 5,949. `WORD_COUNT_REPORT.md` records the commands, full counts, and limitations.

## Figure integrity

- Eight required figure PDFs were copied, not symlinked.
- Source and destination SHA-256 values match for every figure; exact values and placements are in `FIGURE_PROVENANCE.csv`.
- `pdfinfo` reports one page for each figure. Their page boxes retain the intended 180 mm widths and recorded heights.
- `pdfimages -list` reports zero raster image objects for all eight figures.
- `pdffonts` reports two embedded/subset DejaVu font rows and zero non-embedded fonts for every figure.
- Final manuscript, Supplement, response, and cover PDFs likewise have zero non-embedded font rows and zero raster image objects.

## Rendering and visual inspection

Every page was rendered at 120 dpi with `pdftoppm` into `/tmp/rankcloak_v2_visual.Thum10`; no preview was stored in `paperV2/`. All final pages were visually inspected for margins, clipping, aspect ratio, readable type, float behavior, caption collision, table overflow, and unintended blank pages.

The first render exposed an orphaned cover-letter closing on page 2 and a touching header in the two-page landscape ablation table. The cover was condensed to one page, and the ablation header was split over two lines. The corrected cover page and Supplement pages 8, 10, and 11 were re-rendered and inspected individually. Pixel comparison showed that all other final Supplement pages were identical to their already inspected renders. The final 42-page package (15 + 19 + 7 + 1) has no detected clipping, collision, distortion, overflow, or blank float page.

## Scientific consistency checks

- Key counts, confidence intervals, detector metrics, availability totals, and classifications were cross-checked across the Abstract, main Results, tables, captions, Discussion, Supplement, response, and cover.
- A final stale-language scan found no current-pilot sample sizes or affirmative practical, secure, robust, undetectable, human-evaluated, CPU/GPU-equivalent, deployment-validated, or broad-multilingual claim. Matches for these terms are explicit denials, limitations, historical revision descriptions, or reviewer comments.
- The canonical filter is consistently identified as the safe-text filter; the roundtrip-stable Mistral condition is separately identified as unavailable.
- The canonical ablation CSV controls the token-filter rounding (`-0.001797`), as documented in `EVIDENCE_VERIFICATION.md`.
- Human data remain zero participants and zero ratings; unavailable units remain unavailable rather than failures; detector performance remains adverse evidence.

## Persistent file inventory

The pre-existing `paperV2/revisions.txt` was read but not modified. The following persistent files were created in this revision:

```text
paperV2/AI_ASSISTANCE_DISCLOSURE_DRAFT.md
paperV2/BUILD_AND_VALIDATION.md
paperV2/CHANGELOG_FROM_V1.md
paperV2/CITATION_AUDIT.md
paperV2/EVIDENCE_VERIFICATION.md
paperV2/FIGURE_PROVENANCE.csv
paperV2/REVIEWER_RESPONSE_MATRIX.md
paperV2/REVISION_PLAN.md
paperV2/UNRESOLVED_ITEMS.md
paperV2/WORD_COUNT_REPORT.md
paperV2/cover_letter/cover_letter.tex
paperV2/cover_letter/cover_letter.pdf
paperV2/cover_letter/rankcloak_cover_letter_v2.pdf
paperV2/response/response_to_reviewers.tex
paperV2/response/response_to_reviewers.pdf
paperV2/response/rankcloak_response_to_reviewers_v2.pdf
paperV2/scientific_reports/main2.tex
paperV2/scientific_reports/supplementary2.txt
paperV2/scientific_reports/supplementary2.tex
paperV2/scientific_reports/references.bib
paperV2/scientific_reports/wlscirep.cls
paperV2/scientific_reports/naturemag-doi.bst
paperV2/scientific_reports/jabbrv.sty
paperV2/scientific_reports/jabbrv-ltwa-all.ldf
paperV2/scientific_reports/jabbrv-ltwa-en.ldf
paperV2/scientific_reports/supplementary_ablation_rows.tex
paperV2/scientific_reports/supplementary_topic_rows.tex
paperV2/scientific_reports/supplementary_overhead_rows.tex
paperV2/scientific_reports/supplementary_tables/detector_split_metrics.csv
paperV2/scientific_reports/figures/ablation_summary.pdf
paperV2/scientific_reports/figures/automated_readability.pdf
paperV2/scientific_reports/figures/capacity_tail_validation.pdf
paperV2/scientific_reports/figures/computational_overhead.pdf
paperV2/scientific_reports/figures/neural_detector_performance.pdf
paperV2/scientific_reports/figures/neural_detector_performance_compact.pdf
paperV2/scientific_reports/figures/robustness_recovery.pdf
paperV2/scientific_reports/figures/robustness_recovery_compact.pdf
paperV2/scientific_reports/rankcloak_scientific_reports_manuscript_v2.pdf
paperV2/scientific_reports/rankcloak_scientific_reports_supplementary_v2.pdf
```

Dedicated, Git-ignored build artifacts were also created under `paperV2/build/`. They do not clutter the publication source directories. The exhaustive build-artifact inventory is:

```text
paperV2/build/cover_letter/cover_letter.aux
paperV2/build/cover_letter/cover_letter.fdb_latexmk
paperV2/build/cover_letter/cover_letter.fls
paperV2/build/cover_letter/cover_letter.log
paperV2/build/cover_letter/cover_letter.out
paperV2/build/cover_letter/cover_letter.pdf
paperV2/build/main/main2.aux
paperV2/build/main/main2.bbl
paperV2/build/main/main2.blg
paperV2/build/main/main2.fdb_latexmk
paperV2/build/main/main2.fls
paperV2/build/main/main2.log
paperV2/build/main/main2.out
paperV2/build/main/main2.pdf
paperV2/build/response/response_to_reviewers.aux
paperV2/build/response/response_to_reviewers.fdb_latexmk
paperV2/build/response/response_to_reviewers.fls
paperV2/build/response/response_to_reviewers.log
paperV2/build/response/response_to_reviewers.out
paperV2/build/response/response_to_reviewers.pdf
paperV2/build/supp/supplementary2.aux
paperV2/build/supp/supplementary2.fdb_latexmk
paperV2/build/supp/supplementary2.fls
paperV2/build/supp/supplementary2.log
paperV2/build/supp/supplementary2.out
paperV2/build/supp/supplementary2.pdf
```
