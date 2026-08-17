# Word-count report

Count date: 2026-08-16. Tool: TeXcount 3.1.1.

## Journal-defined main text

| Component | Words |
|---|---:|
| Abstract | 184 |
| Introduction | 811 |
| Results | 2,500 |
| Discussion | 764 |
| Responsible use and limitations | 300 |
| **Journal-defined main-text total** | **4,375** |
| Methods (excluded from the journal-defined total) | 2,256 |

The Abstract is below 200 words. The journal-defined total is below the 4,500-word working ceiling and within the planned 4,100--4,450 range.

## Other counts

| Component | Counted words |
|---|---:|
| Four main figure legends | 359 |
| Two main table captions | 82 |
| All six main display captions | 441 |
| Full main manuscript, including Abstract, headings, captions, and counted mathematical tokens | 7,520 |
| Main body prose, including Abstract but excluding headings and captions | 6,938 |
| Supplementary Information, including its Abstract, headings, captions, included table fragments, and counted mathematical tokens | 5,949 |
| Combined full main manuscript and Supplementary Information | 13,469 |

## Method and limitations

The principal command was `texcount -inc -sub=section -sum main2.tex`. Because the Scientific Reports class places the Abstract before `\\begin{document}`, TeXcount omits it from the file total; the Abstract was piped separately through `texcount -sum -` and added. The journal-defined value is the sum of TeXcount body-text counts for Introduction, Results, Discussion, and Responsible use and limitations; it excludes Abstract, Methods, Data and code availability, references, captions, headings, and back matter.

The Supplement was counted with `texcount -inc -sub=section -sum supplementary2.tex`, which follows the wrapper into `supplementary2.txt` and the three generated table fragments. Its pre-document Abstract was counted separately in the same way. The figure-legend count was obtained by extracting only the four `figure` environments and passing them to TeXcount. Raw `wc -w` was not used as the formal count. TeXcount's treatment of mathematical expressions, pseudocode, paths, and table cells can differ from a journal production count; the commands and category totals make that limitation reproducible.
