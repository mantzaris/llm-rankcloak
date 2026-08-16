# Computational-overhead figure technical note

- Authoritative source: `results/revision_v1/final_experiment_package/overhead/overhead_plot_source.csv`.
- Plotted cells are primary-stage, trial-scope, payload-group summaries with 95% payload-bootstrap intervals.
- Timing fields are inclusive wrapper measurements; encoding, generation, and supported decoding are not asserted to be perfectly isolated.
- Encoding setup uses a log axis only in the complete figure because values span several orders of magnitude.
- CPU time and repeated warm-up measurements were unavailable; wall time is not substituted for CPU time.
- Saved RAM/VRAM scopes are limited, and no kernel-exact VRAM peak is claimed.
