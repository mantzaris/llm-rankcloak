# paperV3 pre-submission checklist

Exact revised title: `RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text`

Author-review artifact: `paperV3/diffs.pdf` is for internal author inspection and is not part of the journal submission package.

## Required manual actions before journal submission

- [ ] Enter exactly `RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text` in the journal tracking system and confirm it matches the manuscript and cover letter.
- [ ] Synchronize the final V3 revision to Zenodo.
- [ ] Confirm that the archived version contains the cited computational revision.
- [ ] Synchronize the final DOI and version wording in the manuscript, response letter, repository release notes, and journal submission fields.
- [ ] Verify author name, affiliation, correspondence email, postal address, telephone number, and any required author identifiers.
- [ ] Verify the submission ID, handling-editor name, declarations, and all journal submission metadata.
- [ ] Confirm whether the journal requires a tracked-change manuscript or additional forms.
- [ ] Exclude `paperV3/diffs.pdf` and `paperV3/diffs/` from the journal submission package.

## Package validation

- [x] The exact 14-word title is synchronized in the manuscript, PDF metadata, supplement, response, cover letter, and this checklist; it has no punctuation and is not a question.
- [x] `main3.pdf`, `supplementary3.pdf`, `response_to_reviewers_v3.pdf`, and `cover_letter_v3.pdf` compile with the established LaTeX toolchain.
- [x] The abstract is at most 190 words.
- [x] The main abstract is 185 words and the Supplementary Information abstract is 79 words; neither contains quantization labels or low-FPR shorthand.
- [x] Citations and cross-references resolve; no missing assets, duplicate labels, or unresolved placeholders remain.
- [x] All pages of all four PDFs have been rendered and visually inspected for clipping, margins, page breaks, title wrapping, citations, line numbers, and response-letter references.
- [x] `paperV3/diffs.pdf` compares V2 with final V3 for both documents, visibly marks additions and deletions, includes an author-review warning and supplement divider, and has been inspected on every page.
- [x] The V3 numerical audit matches the sealed handoff, including strict-gate capacity conditional on 114 completions and the paired Q4/Q8 non-significant result.
- [x] Human-authored controls are described as a secondary corpus analysis, not a participant study or naturalness rating.
- [x] Artifact concealment, authorized recovery, and steganalytic classification remain distinct; detector classification is not described as artifact recovery.
- [x] The manuscript does not generalize its empirical result to arbitrary natural-language payloads or describe prompt context as a formal cryptographic key.
- [x] Concern 1 changes remain limited; concerns 2 and 3 receive the principal new experimental reporting.
- [x] GitHub and Zenodo statements are truthful and do not claim that final V3 synchronization has already occurred.
- [x] `paperV2`, `results/revision_v3`, computational code, tests, configurations, provenance, figures, manifests, and source tables remain unchanged.
- [x] `response/requests.txt` remains byte-for-byte unchanged.
- [x] Relevant tests have been run; any immutable historical `paperV2` bibliography-contract failures are reported rather than concealed.
- [x] The final diff is confined to the self-contained `paperV3` package and contains no transient LaTeX build files.
