# Human-study data dictionary — DRAFT

No real participant dataset exists. This dictionary distinguishes participant-facing,
restricted, raw-response, cleaning, and analysis fields so identifiers and condition
labels are not accidentally combined.

## Separation classes

1. **Payment/eligibility restricted:** platform identifier, eligibility, payment state,
   and study code. Stored separately; never released or joined by analysts beyond the
   approved duplicate/payment step.
2. **Blind key restricted:** stimulus blind ID mapped to condition/model/protocol and
   payload metadata. Hidden during collection and condition-blind cleaning.
3. **Raw response:** study code, blind stimulus, scale/rating, attention checks, and
   approved timing metadata; no condition label.
4. **Analysis:** de-identified panel/study code plus condition joined only after the
   cleaning/exclusion lock.

## Core response fields

| Field | Type / allowed values | File class | Meaning |
| --- | --- | --- | --- |
| `response_id` | unique string | raw, analysis | Random response-row identifier; not a platform ID. |
| `participant_slot_id` | `P001`-style study code | raw, analysis | Blinded scheduling/participant code. The fixture uses planned slots. |
| `presentation_order` | integer 1--26 | raw, analysis | Position within the assigned schedule. |
| `item_type` | `experimental_message`, `attention_check` | raw, analysis | Separates outcomes from instruction checks. |
| `stimulus_blind_id` | `B0001`-style or blank for checks | raw, analysis | Random blinded message identifier. |
| `scale_id` | seven scale IDs or `attention_check` | raw, analysis | Rating dimension. |
| `rating` | integer 1--7 | raw, analysis | Submitted response. |
| `attention_check_id` | declared check ID or blank | raw, analysis | Non-outcome instruction-check identifier. |
| `expected_response` | integer 1--7 or blank | restricted cleaning | Expected check response; never an outcome. |
| `response_time_ms` | nonnegative integer, if approved | raw, analysis | Page/item time from the survey platform; sensitivity flag only. |
| `synthetic_fixture` | boolean | all generated fixtures | Must be `false` for a real approved study export. |
| `include_primary` | boolean | cleaning, analysis | Condition-blind inclusion flag with reason in the exclusion log. |

## Stimulus/design fields joined after the blind lock

| Field | Type / allowed values | Meaning |
| --- | --- | --- |
| `condition` | one of eight frozen condition IDs | Source/generation condition; hidden until cleaning lock. |
| `prompt_category` | one of six design categories | Broad topic/writing category. |
| `template_id` | frozen template identifier | Prompt template used for matching/balance. |
| `payload_id` | synthetic payload identifier | Repeated/nesting identifier; contains no payload text. |
| `payload_class` | one of four eligible study classes | Balanced payload stratum. |
| `model_family` | pinned model-family ID or `human` | Source family; never participant-facing. |
| `presentation_scope` | `forced_span`, `full_message` | Whether the displayed text is payload-bearing span or full message. |
| `license_status` | controlled string | Provenance/redistribution eligibility, fixed pre-selection. |
| `safety_screen_status` | controlled string | Pre-rating content-screen result. |

## Participant flow/quality fields

| Field | Type | Storage | Meaning |
| --- | --- | --- | --- |
| `consent_response` | agree/decline | restricted flow log | Must be agree before stimulus presentation. |
| `age_18_or_older` | boolean | restricted flow log | Eligibility gate; do not collect birth date. |
| `english_reading_eligible` | boolean | restricted flow log | Participant self-attested task eligibility. |
| `completion_status` | complete/partial | cleaning | Whether all required pages were submitted. |
| `attention_checks_incorrect` | integer 0--2 | cleaning | Condition-blind check count. |
| `duplicate_status` | unique/duplicate/uncertain | restricted cleaning | Derived under approved platform rule. |
| `exclusion_reason_codes` | semicolon-delimited controlled codes | cleaning | Reasons from `EXCLUSION_PLAN.md`; never free-form outcome commentary. |
| `cleaning_lock_sha256` | SHA-256 string | audit | Hash of frozen inclusion/exclusion file before condition join. |

## Files produced by the randomization dry run

- `candidate_stimuli.csv`: identifiable experimental metadata; never participant-facing.
- `selected_stimuli_blinded.csv`: blind ID, topic label, and message only.
- `participant_schedule.csv`: participant-facing order without condition metadata.
- `blind_key.csv`: restricted join key.
- `design_audit.json`: hashes and balance totals.

No names, email addresses, IP addresses, exact locations, free text, real payload text,
or platform worker identifiers belong in the analysis file.
