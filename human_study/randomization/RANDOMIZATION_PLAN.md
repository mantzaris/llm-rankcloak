# Balanced selection and blinded randomization plan

## Status

Draft and synthetic-test only. Running these scripts does not authorize recruitment or
upload stimuli to a survey platform.

## Candidate manifest

Every row needs a unique stimulus ID, one of eight frozen conditions, prompt category,
template, synthetic payload ID/class, model family or `human`, presentation scope,
message text, license status, and safety-screen status. Final candidates must pass
license and safety review before selection.

## Selection balance

Within condition, candidates are grouped by six prompt categories x three templates x
four eligible payload classes. One candidate is selected by seeded shuffle from each
of the 72 strata. Each condition therefore has 72 messages: 12 per category, four per
template, 18 per payload class, and (for generated fixture conditions) 24 per model.

No outcome exists at selection. Replacement is allowed only for a documented pre-
rating safety, license, duplication, or corruption failure, using the next seeded
candidate from the same stratum.

## Blinding and allocation

Selected messages receive random blind IDs. Participant files omit condition, model,
codec, payload, and presentation-scope fields. `blind_key.csv` must remain restricted
until condition-blind cleaning and exclusions are frozen.

Each stimulus occurs in three different panel slots. Each of 72 slots has three
messages per condition (24 experimental messages). Seeded cyclic permutations prevent
duplicate exposure within slot. Order is shuffled with no adjacent same-condition
messages before labels are removed. Two instructed-response checks are inserted at
noninitial positions, producing 26 schedule rows per slot.

## Audit

The build records the seed, input/design SHA-256 hashes, output hashes, counts, and
balance summaries. The final run must reproduce selected IDs, blind mapping, and
schedules byte for byte from the frozen manifest.
