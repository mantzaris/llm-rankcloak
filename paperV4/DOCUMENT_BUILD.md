# Current V4 editorial document workflow

Edit the current TeX sources and `response/editorial_answers.json`. Scientific scores, tables, figures, frozen plans and earlier stage receipts are retained unchanged. Current build and validation records go to `results/revision_v4/punctuation_edit/`.

From the repository root with the existing analysis environment and local LaTeX toolchain

```bash
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase scientific
.venv/bin/python -m scripts.render_revision_v4_editorial_response
.venv/bin/python -m scripts.render_revision_v4_editorial_response
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase correspondence
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase submission
.venv/bin/python -m scripts.validate_revision_v4_punctuation_edit
.venv/bin/python -m scripts.review_revision_v4_editorial_pdfs
```

The scientific phase resolves supplementary sidebar bookmarks first, then builds the main manuscript. No candidate stamp, reading guide or printed contents is inserted. The response renderer reads actual compiled locations and preserves the thirteen exact review blocks. Main references to supplementary notes and Algorithm S1 are explicit names, so neither document requires another document's auxiliary files. Submission generation embeds the compiled bibliography and editable tables, and tests the result with only local style dependencies in a temporary directory. It creates no upload bundle or archive.

The historical Stage 1–4 and contextual-study renderers remain available to reproduce their archived workflows. Running them on the current manuscript would restore superseded prose or write into earlier evidence namespaces. Use the editorial commands above for current documents. Any further text edit requires new builds and a fresh visual review of changed pages. PDF timestamps need not be identical across builds.

The earlier editorial, presentation-cleanup, publication-edit and disclosure-edit records and validators remain unchanged. The current pass reduces unnecessary punctuation and compound phrases in ordinary prose against commit `dec80d2daff76d2fd3463950951b6012a04d6057`. Its explicit edit ledger preserves mathematical notation, numerical values, the abstract, filter rules, judging rubric, literal examples and original reviewer quotations. It does not restore the removed assistance disclosure. Run the response renderer twice before the correspondence build to check reproducibility. The current manuscript TeX and its included files remain authoritative; the builder never restores historical prose.
