# RankCloak human-evaluation package — DRAFT, NOT APPROVED

This directory contains an English-only, pre-recruitment human-study package for the
RankCloak major revision. It is preparation material only. It has **not** been
submitted to the University of Central Florida Institutional Review Board (UCF IRB),
approved, piloted with people, opened for recruitment, or used to issue payment.

Do not recruit, consent, enroll, expose, rate, or pay any person from this package.
Those actions require completed institutional fields, UCF's formal IRB determination,
an approved final instrument, and explicit authorization outside this repository.

## Frozen planning design

- Eight conditions and 72 selected messages per condition: 576 unique messages.
- Three independent ratings per message: 1,728 experimental rating exposures.
- Seventy-two blinded panel slots, each receiving 24 experimental messages: three
  messages from every condition.
- Six prompt categories, three templates per category, and four eligible synthetic
  payload classes. Selection is exactly balanced within condition: 12 messages per
  category, four per template, and 18 per payload class.
- Two non-outcome instructed-response attention checks per panel slot. These 144 checks
  are not included in the 1,728 experimental exposures.
- Seven 1--7 scales: grammaticality, fluency, coherence, topic adherence,
  completeness, overall naturalness, and suspiciousness.
- Co-primary outcomes: overall naturalness and suspiciousness; the others are
  secondary.

The 72 panel slots are a scheduling construct, not a final powered participant count.
Sample size remains subject to simulation power, the final stimulus inventory, UCF
review, attrition allowance, and the approved recruitment plan.

## Directory map

- `config/`: machine-readable design, power scenarios, and author placeholders.
- `irb/`: UCF-tailored draft protocol, checklist, consent, participant instructions,
  and debriefing text.
- `survey/`: seven-scale instrument and attention checks.
- `randomization/`: deterministic fixture generation, balanced selection, blinding,
  and scheduling.
- `power/`: dependency-free ordinal-data simulation and assumptions.
- `analysis/`: synthetic rating generator, analysis plan, and cumulative-link
  mixed-model R script.
- `data/`: data dictionary and exclusion plan.
- `fixtures/`: synthetic-only fixture specification.
- `tests/`: dependency-free structural and end-to-end tests.

## Synthetic dry run

Run from the repository root; these commands create only synthetic data.

```bash
python3 human_study/randomization/generate_synthetic_candidates.py \
  --output /tmp/rankcloak-human/candidate_stimuli.csv
python3 human_study/randomization/build_blinded_assignments.py \
  --candidates /tmp/rankcloak-human/candidate_stimuli.csv \
  --output-dir /tmp/rankcloak-human/randomization
python3 human_study/analysis/generate_synthetic_ratings.py \
  --schedule /tmp/rankcloak-human/randomization/participant_schedule.csv \
  --blind-key /tmp/rankcloak-human/randomization/blind_key.csv \
  --output /tmp/rankcloak-human/synthetic_ratings.csv
python3 human_study/power/simulate_power.py --simulations 2000 \
  --output /tmp/rankcloak-human/power_results.csv
python3 human_study/power/simulate_power.py \
  --config human_study/config/power_design_grid.json --simulations 2000 \
  --output /tmp/rankcloak-human/power_design_grid.csv
Rscript human_study/analysis/ordinal_mixed_model.R \
  --data /tmp/rankcloak-human/synthetic_ratings.csv \
  --output-dir /tmp/rankcloak-human/model
python3 -m unittest discover -s human_study/tests -v
```

The final R fit requires the CRAN package `ordinal`. `--validate-only` performs schema
and design checks using base R when `ordinal` is unavailable. No installer or network
call is included.

The expanded planning grid is a sensitivity calculation, not an approved redesign.
Its first tested 80% landmarks for a planning odds ratio of 1.50 are 192 messages per
condition with three ratings or 120 with five under typical heterogeneity; high
heterogeneity requires more. See `power/PLANNING_RESULTS.md` before treating the
72-by-three scheduling fixture as a recruitment target.

## Blinding and unresolved fields

The randomizer emits public blinded stimuli/schedules, a restricted `blind_key.csv`,
and a hashed design audit. Never expose the blind key to raters or in survey metadata.

Unresolved author fields use `{{UPPER_SNAKE_CASE}}` markers inventoried in
`config/placeholders.json`. They include PI/contact details, department, protocol
number, retention period, platform, compensation, funding, privacy settings, and any
faculty-advisor requirement. The author must resolve them against current UCF forms.

## Ethics and scope guardrails

- Only adult English readers are contemplated.
- Only synthetic, non-operational payload contexts may appear in stimuli.
- No real credentials, secrets, accounts, commands, private data, or harmful
  instructions may be shown.
- Source condition is blinded. Whether debriefing this constitutes incomplete
  disclosure is an explicit UCF IRB determination item.
- Direct identifiers are not analysis fields. Approved platform identifiers must be
  separated and replaced with study codes.
- Exclusions are condition-blind and frozen before unblinding.

Before human activity, resolve every placeholder, replace fixtures with the frozen
message manifest, run the full power analysis, validate licensed controls, obtain the
UCF determination, freeze the approved instrument, and separately authorize
recruitment. Nothing here constitutes institutional approval.
