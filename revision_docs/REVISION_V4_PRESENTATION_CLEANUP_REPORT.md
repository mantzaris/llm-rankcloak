# V4 presentation cleanup

Work began on clean `main` at `479535cca709e624540cd95c0a23bf05f0d28123`. The fetched remote matched. No applicable `AGENTS.md` was found and no newer author changes were present.

Removed the four visible candidate banners and their surrounding formatting, obsolete source-header comments, the supplementary Reading guide, printed contents and its separating page break. Note S1 now follows the normal supplementary front matter on page 1. All Notes S1–S21 and the 61 sidebar bookmarks remain.

Correspondence uses ordinary document-location wording and a normal cover-letter opening. Availability and the code-deposit answer retain the repository URL and explicitly identify the published version 2.0.0 DOI as covering V3 only. A V4 DOI-assigned deposit remains outstanding. Local archive history and the future document/deposit sequence remain in the internal author checklist.

Single-column unjustified text, existing Computer Modern fonts, page footers, numerical bracketed citations, line numbers, captions, affiliations, contact information and the AI-assistance disclosure were retained. The journal class, margins, font sizes and spacing were not changed. The [submission guidance](https://www.nature.com/srep/author-instructions/submission-guidelines) and [revision checklist](https://www.nature.com/documents/srep-checklist-for-revised-submissions.pdf) were checked on 26 September 2026. Removing printed supplementary contents implements the author's preference. The checklist's restriction on table-of-contents figures is a different provision.

| Document | Pages before | Pages after |
| --- | ---: | ---: |
| [Main manuscript](../paperV4/scientific_reports/main4.pdf) | 23 | 23 |
| [Supplementary Information](../paperV4/scientific_reports/supplementary4.pdf) | 49 | 48 |
| [Reviewer response](../paperV4/response/response_to_reviewers_v4.pdf) | 8 | 8 |
| [Cover letter](../paperV4/cover_letter/cover_letter_v4.pdf) | 1 | 1 |

The current editorial builder, renderer, validator and PDF-review tool now write to `results/revision_v4/presentation_cleanup/`. Earlier reports and receipts remain unchanged. Two ordinary response renders produced identical source without the old edition wording. [Build instructions](../paperV4/DOCUMENT_BUILD.md) document that check.

Validation confirmed unchanged scientific prose, abstracts, equations, filter rules, algorithms, display items and examples. All thirteen exact quoted review blocks and sixteen answers remain, including the unchanged 236-word R1.4 answer. Section labels and item numbering were preserved, response pages refreshed, and all retained bookmarks checked against their actual headings. Final builds have no warnings or unresolved references. An initial response URL overflow was corrected with a breakable hyperlink.

All 80 pages passed text and bounds checks. All 65 changed or reflowed pages were visually reviewed without unwanted blank front matter, clipping, overlap or broken figures/tables. The [editable manuscript](../paperV4/scientific_reports/main4_submission.tex) retains its embedded bibliography and editable tables. Its clean dependency build produced 21 pages and verified 183 input paths.

[Validation receipt](../results/revision_v4/presentation_cleanup/validation/presentation_check.json) and [visual review](../results/revision_v4/presentation_cleanup/validation/pdf_review.json) record the checks and final hashes. GPU time was zero. No experiments, statistical reruns, archive/deposit work or journal submission occurred. Presentation cleanup does not confer author approval or complete the actions in the [author checklist](../paperV4/ACTIONS_BEFORE_SUBMISSION.md).
