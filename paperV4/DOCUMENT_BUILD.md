# Current V4 editorial document workflow

Edit the current TeX sources and `response/editorial_answers.json`. Scientific scores, tables, figures, frozen plans and earlier stage receipts are retained unchanged. Current build and validation records go to `results/revision_v4/editorial_restructure/`.

From the repository root with the existing analysis environment and local LaTeX toolchain

```bash
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase scientific
.venv/bin/python -m scripts.render_revision_v4_editorial_response
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase correspondence
.venv/bin/python -m scripts.build_revision_v4_editorial_documents --phase submission
.venv/bin/python -m scripts.validate_revision_v4_editorial_documents
.venv/bin/python -m scripts.review_revision_v4_editorial_pdfs
```

The scientific phase resolves supplementary navigation first, then builds the main manuscript. The response renderer reads actual compiled locations and preserves the thirteen exact review blocks. Main references to supplementary notes and Algorithm S1 are explicit names, so neither document requires another document's auxiliary files. Submission generation embeds the compiled bibliography and editable tables, and tests the result with only local style dependencies in a temporary directory. It creates no upload bundle or archive.

The historical Stage 1–4 and contextual-study renderers remain available to reproduce their archived workflows. Running them on the current manuscript would restore superseded prose or write into earlier evidence namespaces. Use the editorial commands above for current documents. Any further text edit requires new builds and a fresh visual review of changed pages. PDF timestamps need not be identical across builds.
