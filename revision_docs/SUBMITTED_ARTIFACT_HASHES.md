# Protected submitted-artifact hashes

These SHA-256 values were recorded before creating the revision sources. They
are the immutability gate for the submitted manuscript, Supplementary
Information, and archived submitted PDFs.

| Protected artifact | SHA-256 |
|---|---|
| `.paper/scientific_reports/main.tex` | `17e1045c098184b9472ded03e2dc16a26e451d8676e1ec982114ed8a9a545d74` |
| `.paper/scientific_reports/supplementary.tex` | `93208ad95613d4fcd9eb60d5307c364fa783505408ca384ed9fd00bd2d75b995` |
| `.paper/scientific_reports/rankcloak_scientific_reports_manuscript.pdf` | `ac90fd962f48117b8549e5488a543f42b777d58f773a81aeb5fca1038de74703` |
| `.paper/scientific_reports/rankcloak_scientific_reports_supplementary.pdf` | `d1e9f57ddbfaf4daaaf92caeea33412813796adae8aa64f539d74f3ddf2bf219` |

The revision must be written only to `main2.tex`, `supplementary2.tex`, and
`response_letter2.tex`. A final build gate must recompute all four hashes and
fail if any differs.
