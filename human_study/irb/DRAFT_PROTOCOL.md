# DRAFT research protocol — NOT SUBMITTED OR APPROVED

**Institution:** University of Central Florida  
**Working title:** Human assessment of naturalness and suspiciousness in human- and
model-generated English messages  
**Principal investigator:** {{PI_NAME}}  
**Department:** {{UCF_DEPARTMENT}}  
**Contact:** {{PI_EMAIL}}  
**Faculty advisor, if required:** {{FACULTY_ADVISOR_IF_APPLICABLE}}  
**UCF IRB protocol:** {{UCF_IRB_PROTOCOL_NUMBER}}  
**Funding:** {{FUNDING_SOURCE}}

This document is a repository draft. It does not assert exemption, expedited review,
approval, or permission to begin. The investigator must transfer and reconcile it with
the then-current UCF Huron IRB submission and obtain UCF's determination before
any recruitment, consent, pilot, rating, or payment.

## 1. Purpose and research questions

The study will measure how adult English readers perceive short messages drawn from
eight blinded source/generation conditions. It supports a methodological evaluation of
RankCloak, an LLM rank-transcoding framework. The questions are whether conditions
differ in overall naturalness or suspiciousness and how grammaticality, fluency,
coherence, topic adherence, and completeness support those judgments.

The human study validates perceived cover-text quality. It is not a test of
cryptographic security, undetectability, participant deception susceptibility, or an
individual participant's language ability.

## 2. Design

The frozen planning design contains eight conditions and 72 English messages per
condition (576 unique stimuli). Each stimulus receives three independent ratings,
giving 1,728 experimental exposures. A balanced schedule has 72 panel slots; each slot
contains 24 experimental messages, three from each condition, plus two explicit
instructed-response attention checks. Source labels, models, codecs, and payload
metadata are hidden.

Messages are balanced within condition across six prompt categories, three templates
per category, and four eligible synthetic payload classes. Human-written controls must
be lawfully licensed and topic/length matched. All non-human conditions use frozen,
synthetic, non-operational inputs. The randomization seed, input hash, selected IDs,
and blind key are recorded by code.

The 72 slots are a design target, not authorization to enroll 72 people. The requested
sample, oversampling allowance, and replacement policy will be updated to
{{APPROVED_SAMPLE_SIZE_WITH_ATTRITION}} after simulation-based power analysis and UCF
review.

## 3. Participants

### Inclusion criteria

- At least 18 years old.
- Able to read English sufficiently to evaluate short English messages.
- Provides electronic informed consent.
- Meets the approved platform/jurisdiction eligibility criteria.

### Exclusion criteria

- Does not consent or reports being under 18.
- Duplicate enrollment under the approved duplicate-detection rule.
- Does not complete the required rating task.
- Fails both non-outcome instructed-response checks.

A single attention-check error alone does not trigger primary exclusion. Outcome
ratings, perceived condition, and agreement with study hypotheses are never exclusion
criteria. The full condition-blind rules are in `data/EXCLUSION_PLAN.md`.

Children, prisoners, and adults unable to consent are not targeted. No protected
health, education, employment, financial, or criminal information is requested.

## 4. Recruitment and compensation

No recruitment is authorized by this draft. If approved, participants would be
recruited through {{RECRUITMENT_PLATFORM}} only during
{{APPROVED_RECRUITMENT_DATES}}, using text approved by UCF. Compensation is
{{COMPENSATION_AMOUNT_AND_BASIS}}. Payment must not be contingent on particular
ratings or passing an attention check; the final handling of partial completion must
match the approved consent and platform posting.

## 5. Procedures

1. The platform presents age/English eligibility items and the approved information
   sheet.
2. A participant who declines exits before any research stimulus.
3. The participant reads the task instructions and one non-scored practice item if
   UCF approves it. No pilot participants are enrolled under this draft.
4. The participant receives one precomputed blinded schedule: 24 experimental
   messages and two attention checks.
5. For each experimental message, the participant sees its broad topic label and
   rates seven dimensions on integer scales from 1 to 7.
6. Backtracking and source guessing questions are disabled. No free text is collected.
7. The final page provides approved debriefing text and study contact information.

Estimated duration will be established from non-human timing tests and, only if
approved, a UCF-authorized pilot; the participant document retains
{{ESTIMATED_DURATION_MINUTES}} until then.

## 6. Stimulus safeguards

Stimuli are screened before instrument upload. They may not contain real credentials,
private data, real operational secrets, executable commands, individualized advice,
sexual content, graphic violence, harassment, self-harm content, or instructions for
wrongdoing. Cryptographic-artifact-like inputs are synthetic and are not visible as
real accounts or secrets. Human controls must have documented license and provenance.

Malformed, identifying, unsafe, or licensing-ineligible items are removed before the
selection manifest is frozen, not after outcomes are viewed. Replacement follows the
same blinded stratum rule.

## 7. Information disclosure and debriefing

Participants are told that the study compares qualities of English messages from
different writing/generation processes. Exact source conditions and the hidden-message
research question are withheld during rating to avoid demand characteristics. No false
statement about a message's source is made.

UCF must determine whether this limited withholding requires an incomplete-disclosure
or deception procedure and whether the proposed debrief is adequate. The investigator
will use the category and language required by UCF rather than assuming one here.

## 8. Risks and risk mitigation

Foreseeable risks are mild fatigue, boredom, frustration with awkward prose, and a
small privacy risk from online participation metadata. Content safeguards reduce
message-related risk. Participants may stop at any time. The design avoids sensitive
questions and free-text responses. The platform settings will be documented as
{{SURVEY_PLATFORM_PRIVACY_SETTINGS}}.

The investigator requests UCF's assessment of risk level; this draft does not label
the project exempt or minimal risk on UCF's behalf.

## 9. Benefits

There is no expected direct participant benefit. The research may improve scientific
understanding of perceived quality and detectability in generated English text.

## 10. Privacy, confidentiality, and data security

The survey runs on {{SURVEY_PLATFORM}}. The research team will not intentionally
collect names, email addresses, precise location, free text, or IP addresses when the
approved platform permits disabling them. Any platform worker identifier used for
eligibility/payment is stored separately from responses and replaced with a random
study code before analysis.

The response dataset contains panel-slot/study codes, blinded stimulus IDs, ratings,
attention-check results, coarse timing fields if approved, and design strata. The
condition blind key is held separately with restricted access. Files are stored in
UCF-approved encrypted storage accessible only to {{AUTHORIZED_STUDY_TEAM}}.

Identifiers/payment records are separated from analysis data. Retention is
{{DATA_RETENTION_YEARS}} years or the period required by UCF and the final publication
plan, whichever the approved protocol specifies. Public releases contain only
de-identified aggregate results and permitted stimuli/materials; no platform IDs are
released.

## 11. Analysis

Overall naturalness and suspiciousness are co-primary ordinal outcomes. The primary
analysis uses cumulative-link mixed models with fixed condition and prespecified
design covariates and crossed participant and stimulus random intercepts. The five
remaining scales are secondary. Holm adjustment is applied within the co-primary and
secondary contrast families. Effect estimates, odds ratios, and 95% confidence
intervals are reported. Exclusions are finalized while condition labels remain
blinded. Sensitivity analyses address single attention-check failures and convergence.

The simulation power analysis uses ordinal thresholds and participant/stimulus
heterogeneity; assumptions and sensitivity scenarios are documented under `power/`.

## 12. Data quality, monitoring, and withdrawal

The survey enforces valid 1--7 values and unique schedule tokens. Completion and
attention checks are evaluated without condition labels. A participant may withdraw
by closing the survey. The consent text must state whether already submitted anonymous
responses can be located and deleted; the approved platform behavior controls that
statement.

Unexpected problems, complaints, or protocol deviations will be handled under current
UCF reporting requirements. Contact language remains
{{CURRENT_UCF_IRB_CONTACT_TEXT}} until supplied from the current institutional form.

## 13. Dissemination

Results may appear in a Scientific Reports revision, Supplementary Information,
conference presentations, and a DOI-assigning repository. Only aggregate,
de-identified ratings and UCF-permitted materials will be released. Any reuse beyond
the approved scope requires the applicable UCF determination.
