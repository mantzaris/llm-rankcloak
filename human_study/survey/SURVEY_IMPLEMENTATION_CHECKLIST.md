# Survey implementation checklist — DRAFT

- [ ] UCF IRB determination and approved version date recorded.
- [ ] Every `{{PLACEHOLDER}}` in the approved participant materials resolved.
- [ ] English-only eligibility and age gate implemented before consent.
- [ ] Consent response required; declining exits without presenting stimuli.
- [ ] Condition, model, codec, payload, and source metadata hidden from participants.
- [ ] Topic label and message text rendered without platform reformatting.
- [ ] All seven integer responses required for each experimental message.
- [ ] Scale order and anchors match `instrument.json` exactly.
- [ ] Backtracking disabled to limit comparison/memory effects.
- [ ] Two attention checks inserted according to the blinded schedule.
- [ ] No free-text field or unnecessary identifier collection enabled.
- [ ] Platform timing metadata limited to approved fields.
- [ ] Participant-facing debrief appears only after the final response.
- [ ] Test accounts and synthetic fixtures removed before an approved launch.
- [ ] Export schema validated against `data/DATA_DICTIONARY.md`.
- [ ] Blind key remains outside the survey platform and analyst-cleaning dataset.

Checking this list does not authorize launch or substitute for UCF approval.
