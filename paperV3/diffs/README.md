# RankCloak V2-to-V3 author diff

`paperV3/diffs.pdf` is an internal author-review artifact. It is not part of the journal submission package.

## Files compared

- Main manuscript: `paperV2/scientific_reports/main2.tex` versus `paperV3/scientific_reports/main3.tex`
- Supplementary Information: `paperV2/scientific_reports/supplementary2.tex` versus `paperV3/scientific_reports/supplementary3.tex`

The comparison is V2 versus the final V3 sources, not the smaller change from commit `ffc56c5`.

## Tools and commands

The diff was generated from the repository root with LATEXDIFF 1.3.2:

```bash
latexdiff --encoding=utf8 --type=UNDERLINE --subtype=SAFE --floattype=FLOATSAFE --graphics-markup=both paperV2/scientific_reports/main2.tex paperV3/scientific_reports/main3.tex > paperV3/diffs/main_diff.tex
latexdiff --encoding=utf8 --type=UNDERLINE --subtype=SAFE --floattype=FLOATSAFE --graphics-markup=both paperV2/scientific_reports/supplementary2.tex paperV3/scientific_reports/supplementary3.tex > paperV3/diffs/supplementary_diff.tex

ruby paperV3/diffs/repair_diff.rb main paperV2/scientific_reports/main2.tex paperV3/scientific_reports/main3.tex paperV3/diffs/main_diff.tex
ruby paperV3/diffs/repair_diff.rb supp paperV2/scientific_reports/supplementary2.tex paperV3/scientific_reports/supplementary3.tex paperV3/diffs/supplementary_diff.tex
```

The two diff documents, cover, and divider were compiled and assembled with:

```bash
mkdir -p paperV3/diffs/build
env TEXINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: BIBINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: BSTINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paperV3/diffs/build paperV3/diffs/main_diff.tex
env TEXINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: BIBINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: BSTINPUTS=paperV3/scientific_reports//:paperV2/scientific_reports//: latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paperV3/diffs/build paperV3/diffs/supplementary_diff.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paperV3/diffs/build paperV3/diffs/review_cover.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paperV3/diffs/build paperV3/diffs/supplement_divider.tex
pdfunite paperV3/diffs/build/review_cover.pdf paperV3/diffs/build/main_diff.pdf paperV3/diffs/build/supplement_divider.pdf paperV3/diffs/build/supplementary_diff.pdf paperV3/diffs.pdf
```

## Limited diff-only repair

`wlscirep` places the title and abstract before `\begin{document}`. LATEXDIFF therefore treated wholesale abstract replacements as preamble edits and interleaved the old and new titles too tightly. `repair_diff.rb` makes those changes explicit as red struck-through V2 text and blue underlined V3 text, separates the two title versions, enables flexible line breaking, and reduces full-width figures to 97% inside the diff only so colored frames remain within the margins. No submission-manuscript source is changed by this repair.

## Output

The final combined author-review file is `paperV3/diffs.pdf`. It contains a cover and legend, the complete main-manuscript comparison, a supplementary divider, and the complete Supplementary Information comparison.
