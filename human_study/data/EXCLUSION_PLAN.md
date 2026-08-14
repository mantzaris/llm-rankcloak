# Condition-blind exclusion and missing-data plan — DRAFT

## Governing rule

All participant and stimulus exclusions are executed using blinded IDs. The inclusion
file and reason codes are frozen and hashed before `blind_key.csv` is joined. Ratings
are never excluded because they are low, surprising, disagree with another rater, or
weaken a hypothesis.

## Participant-level primary exclusions

Exclude a participant/panel assignment from primary outcome models for exactly these
prespecified reasons:

| Code | Rule |
| --- | --- |
| `NO_CONSENT` | Consent was declined or no affirmative consent was recorded. |
| `UNDER_18` | Participant did not affirm age 18 or older. |
| `ENGLISH_INELIGIBLE` | Participant did not affirm the approved English-reading criterion. |
| `DUPLICATE_CONFIRMED` | Duplicate participation under the approved platform rule; retain the first complete eligible assignment unless UCF requires another rule. |
| `INCOMPLETE_REQUIRED_TASK` | Fewer than all 24 experimental message pages or their required seven ratings were submitted. |
| `BOTH_ATTENTION_CHECKS_FAILED` | Both explicit instructed-response items were incorrect. |
| `SCHEDULE_MISMATCH` | Responses cannot be reconciled to the issued blind schedule or contain duplicated/impossible item IDs. |

One failed attention check is retained in the primary analysis and flagged for a
prespecified sensitivity analysis. Compensation must follow the approved participant
terms and is not retroactively denied because of an analytic exclusion.

## Timing and response-pattern flags

Timing is not a stand-alone primary exclusion because reading speeds vary and the
platform may measure background time imperfectly. Before unblinding, flag:

- completion time below one-third of the median among otherwise eligible complete
  assignments;
- ten or more experimental pages with recorded time below five seconds; or
- identical ratings on all 168 experimental scale responses.

Report these flags and run a sensitivity analysis excluding flagged assignments. Add
them to the primary exclusion rule only if UCF-approved materials and a pre-data
amendment explicitly do so. Do not inspect condition labels when setting thresholds.

## Stimulus-level exclusions

Final safety, licensing, duplicate-text, corruption, and empty/truncated-message checks
occur before selection and upload. If a post-collection technical audit proves that a
stimulus displayed incorrectly, exclude every rating for that blinded stimulus with
code `STIMULUS_TECHNICAL_INVALID`, document the evidence without condition labels, and
do not replace it after outcomes are visible. Report the affected condition only after
the exclusion lock is joined to the blind key.

Awkwardness, artifacts, low naturalness, and suspiciousness are outcomes, not reasons
to remove a stimulus.

## Missing data

- Required survey validation should prevent item-level missing ratings.
- Do not impute ordinal outcomes.
- Incomplete required tasks receive `INCOMPLETE_REQUIRED_TASK` in the primary set.
- If the approved platform permits recoverable partial tasks, report their flow and
  include a clearly labeled available-case sensitivity analysis only.
- Missing design metadata is a pipeline error and must be resolved from the frozen
  manifests; it is not statistically imputed.

## Attrition and replacement

The recruitment target may include the UCF-approved attrition allowance. Replacement
participants receive unused precomputed panel schedules. Stop/replacement logic uses
only completion and blinded quality fields. Do not selectively add raters to conditions
or stimuli after viewing ratings.

## Cleaning lock and audit trail

Produce a row-level file containing study code, blinded stimulus ID where applicable,
inclusion flag, controlled reason codes, rule version, and timestamp. Validate counts,
hash the file, record the code revision, and only then authorize a separate process to
join condition metadata. Preserve a CONSORT-style flow count for the manuscript/SI.
