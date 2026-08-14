# Planning power result — synthetic design calculation only

The baseline deterministic results were generated on 2026-08-08 and the expanded
design grid on 2026-08-09, each with 2,000 replicates per scenario. They are not
participant data, a pilot, an empirical result, or a UCF-approved sample-size
determination. Reproduce them with:

```bash
python3 human_study/power/simulate_power.py --simulations 2000 \
  --output human_study/power/planning_power_results_3ratings.csv
python3 human_study/power/simulate_power.py \
  --config human_study/config/power_scenarios_5ratings.json --simulations 2000 \
  --output human_study/power/planning_power_results_5ratings.csv
python3 human_study/power/simulate_power.py \
  --config human_study/config/power_design_grid.json --simulations 2000 \
  --output human_study/power/planning_power_design_grid.csv
```

## Decision implication

The proposed 72 messages per condition with three ratings per message yielded a 0.404
rejection probability (Monte Carlo SE 0.011) for the planning common odds ratio of
1.50 at two-sided alpha 0.025 in the expanded, independently seeded grid. Five ratings
per message increased the 72-message value to 0.595. Under the typical heterogeneity
assumptions, the first tested designs exceeding 0.80 were 192 messages per condition
with three ratings (0.853) and 120 messages per condition with five ratings (0.832).
Under the higher-heterogeneity assumptions, 144 messages with five ratings yielded
0.763 and 192 with five ratings yielded 0.886. The smaller planning odds ratio of 1.25
did not reach 0.80 even at 240 messages with five ratings (0.583). The grid null
check was 0.0135 (Monte Carlo SE 0.0026), consistent with conservative calibration at
alpha 0.025. Small differences from the earlier files reflect their distinct frozen
scenario seeds.

Accordingly, this simulation does **not** justify claiming 80% power for the proposed
1,728-exposure plan, nor does it select a replacement design. Recruitment must not
begin from that plan. The 120-by-five typical design would require 4,800 experimental
exposures across eight conditions; the more conservative 192-by-five design would
require 7,680. Those are sensitivity landmarks, not approved sample sizes. Before UCF
submission, the authors must freeze a smallest effect of scientific interest and
primary contrast family, simulate the final cumulative-link mixed model or a validated
close approximation, include attrition/exclusion allowances, and obtain statistical
and IRB approval. If the 1,728-exposure design is retained for feasibility, the human
study should be framed as an estimation study with uncertainty reported, not as a
powered equivalence or non-inferiority test.

The portable planning test is intentionally simpler than the final crossed ordinal
model; therefore these values are a decision warning rather than definitive sample-size
estimates. The assumptions and omissions are documented in
`ASSUMPTIONS_AND_SENSITIVITY.md`.
