# paperV3 pre-submission checklist

## Required manual actions before journal submission

- [ ] **Journal tracking-system title:** enter exactly <code>RankCloak Measures Recovery and Detectability of Synthetic Cryptographic Artifacts Encoded in Language Model Text</code> and confirm it matches the manuscript and cover letter.
- [ ] **Zenodo/final archive:** archive the final accepted computational and manuscript revision in a recognized DOI-assigning repository.
- [ ] Confirm or update the Zenodo DOI and deposited version; do not assume the current DOI already contains the final V3 commit.
- [ ] Confirm that the archived version corresponds to the code revision cited for submission.
- [ ] Synchronize the final DOI/version wording in the manuscript, response letter, repository release notes, and journal submission fields.
- [ ] Verify author name, degree/title where applicable, affiliation, correspondence email, postal address, and telephone number.
- [ ] Recheck the journal submission ID and handling-editor name.
- [ ] Confirm that no additional journal forms, declarations, or tracked-change manuscript are required.

## Repository and document checks

- [x] Exact revised title is synchronized in <code>scientific_reports/main3.tex</code>, <code>supplementary3.tex</code>, <code>response/response_to_reviewers_v3.tex</code>, <code>cover_letter/cover_letter_v3.tex</code>, their PDFs, PDF metadata where controlled, and this checklist.
- [x] Revised title contains 14 words, no punctuation, and is not a question.
- [x] <code>main3.pdf</code>, <code>supplementary3.pdf</code>, <code>response_to_reviewers_v3.pdf</code>, and <code>cover_letter_v3.pdf</code> compile with the established LaTeX toolchain.
- [x] No undefined citation, undefined reference, multiply defined label, missing figure/table, or unresolved placeholder remains.
- [x] All figures and tables are present, legible, unclipped, and referenced consistently.
- [x] All four PDFs have been rendered and visually inspected for overflow, clipping, blank pages, broken references, and title consistency.
- [x] Every response-letter location and page/table reference has been checked against the final compiled PDFs.
- [x] Numerical claims match <code>results/revision_v3</code>, including 114-case conditional strict-gate capacity and 960-case Q4/Q8 visible-recovery denominators.
- [x] Human-authored controls are described only as a bounded secondary corpus analysis, not a participant or user study.
- [x] No statistically significant or general Q8 recovery advantage is claimed.
- [x] No practical, robust, secure, deployable, or undetectable covert-channel claim is introduced.
- [x] Code Availability preserves the valid repository/DOI information without claiming that the final V3 commit is already archived.
- [x] <code>paperV2</code> is byte-for-byte unchanged.
- [x] No computational code, test, configuration, provenance, manifest, result table, or figure under <code>results/revision_v3</code> changed.
- [x] The authoritative <code>response/requests.txt</code> and its pre-existing backup remain byte-for-byte unchanged.
- [x] Relevant tests have been run and any immutable legacy <code>paperV2</code> bibliography-contract failures are reported rather than masked.
- [x] Review <code>git diff --stat</code> and every changed file; confirm changes are confined to <code>paperV3</code>.
- [ ] Commit the complete package as “Create minimal paperV3 editor revision”.
- [ ] Push the final commit to <code>origin/main</code>, verify the remote commit, and confirm a clean working tree.
