# V4 editorial restructuring report

The main manuscript now presents the method and principal findings with 14.6% less explanatory text. Results begin on page 11 rather than page 13. Supporting methods have moved into existing supplementary notes, which now have linked contents and a meaningful bookmark hierarchy. Scientific evidence and conclusions are unchanged.

Work began on clean `main` at `10a76e51e629b1df22b31c6af1c903b9cda50501`. The fetched remote matched that commit. No applicable `AGENTS.md` was found. No author edits were displaced, and no experiments, scientific analyses, archive preparation or publication actions were performed. Additional GPU time was zero.

## Length and layout

Counts use the same TeXcount convention before and after, expanding local inputs and reporting “Words in text.” Headings, captions, mathematics, abstract, availability statements and references are excluded from the explanatory subtotal. Methods includes its pseudocode. Separate count logs and exact PDF hashes are retained in the [validation receipt](../results/revision_v4/editorial_restructure/validation/editorial_check.json).

| Prose scope | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Methods | 5,571 | 4,426 | 20.6% |
| Introduction, Results and Discussion | 4,555 | 4,196 | 7.9% |
| Responsible use and limitations | 165 | 165 | 0% |
| Combined explanatory text | 10,291 | 8,787 | 14.6% |

| Document | Before pages | After pages |
| --- | ---: | ---: |
| Main manuscript | 25 | 23 |
| Supplement | 45 | 49 |
| Reviewer response | 9 | 8 |
| Cover letter | 1 | 1 |

Fonts, margins, line spacing, figure sizes, title, author details, abstract, disclosures and journal template were preserved. The five main figures, three main tables, sixteen displayed equations and two essential algorithms are unchanged.

## Movement map

| Supporting block formerly in main | Destination | Main replacement |
| --- | --- | --- |
| Protocol bookkeeping and replay checks | S1, replay modes and structured endpoints | Compact shared-contract explanation and S1 reference |
| Full tail procedure, formerly Algorithm 3 | S1, tail policies and Algorithm S1 | Stopping principle and explicit Algorithm S1 reference |
| Repeated work-unit reconciliation | Existing S2, Table S4 | Corpus summary retaining principal denominators |
| Detector normalization and threshold details | Existing S11 and S12 | Partition safeguards and named references |
| Contextual selection, full rubric, settings and retries | S20, selection, rubric and calibration subsections | Four concise Methods paragraphs |
| Calibration breakdown and missing-score bounds | S20, estimation, missingness and examples | Principal estimates, incomplete-pair count and specific references |
| Conditional agreement formula and distribution-support detail | S19, compatibility and trace-informed detection | Direct configuration and detection explanations |

Repeated discussion qualifications were consolidated. The complete deterministic filter remains unchanged in main Methods. S17 now gives a shorter account of both former pilots, their amendment, adverse results, exclusions and qualified likelihood findings. S20 retains exact full-message examples and judge instructions, with the disruption scale in new Table S36.

One description was made more precise when moving tail details. The existing dynamic helper applies `rstrip()` before checking terminal characters, making its listed double-newline suffix branch unreachable. Main text now describes the effective punctuation rule, and S1 records this implementation detail. No algorithm implementation, configuration, trajectory or numerical result changed.

## Response and validation

R1.4 is 236 words excluding its quotation and locations. It retains 537 complete pairs, 90.3% encoded versus 96.6% ordinary acceptance and the paired difference of −6.3 percentage points with 95% interval [−8.7, −3.9]. It directs readers to S20 for calibration, subgroup and missingness detail. The cover letter needed one corresponding location change.

All thirteen exact review blocks and sixteen complete answer sets passed validation. Response locations and the matrix were refreshed from compiled labels. Separate current response and claim-location ledgers preserve the earlier scientific ledgers. Automated contextual evidence remains distinct from unmeasured human perception and outstanding public coverage.

All four documents built without warnings or undefined references. All 61 supplementary contents/bookmark destinations were checked against the actual headings, including all 21 notes. No duplicate destinations were found. Every page of the four final PDFs was visually reviewed, 81 pages in total, with no clipping, broken layout or unreadable figures observed. Historical tables, filter literals, mathematical blocks, quoted examples, numerical claims and scientific records passed preservation checks.

The editable manuscript includes compiled bibliography and editable tables. Its clean dependency build produced 21 pages and verified 183 input paths without external manuscript dependencies. [Current build instructions](../paperV4/DOCUMENT_BUILD.md) use the new editorial namespace and preserve historical renderers and logs.

## Final files

- [Main manuscript](../paperV4/scientific_reports/main4.pdf)
- [Supplement](../paperV4/scientific_reports/supplementary4.pdf)
- [Reviewer response](../paperV4/response/response_to_reviewers_v4.pdf)
- [Cover letter](../paperV4/cover_letter/cover_letter_v4.pdf)
- [Editable manuscript](../paperV4/scientific_reports/main4_submission.tex)
- [Editorial evidence and records](../results/revision_v4/editorial_restructure/)

The author-review status and concrete submission prerequisites remain in [the author checklist](../paperV4/ACTIONS_BEFORE_SUBMISSION.md). This editorial pass does not change publication or journal-submission status.
