# Robustness figure technical note

- Authoritative source: `results/revision_v1/final_experiment_package/robustness/robustness_recovery_plot_source.csv`.
- Evidence classification: secondary evidence with diagnostic scope; 11 rows in the compact core candidate and 24 rows in the full supporting candidate.
- Analysis unit: source cover; intervals: Wilson 95% confidence intervals.
- Replay, raw-transmission, limited-canonicalization, and cross-model channels remain separate; no pooled recovery estimate is plotted.
- `Final 10% tail-only truncation` removes `ceil(10%)` of final token IDs. It does not test arbitrary truncation of payload-bearing positions.
- Limited canonicalization has partial availability: 96 observed and 48 unavailable source-cover units per displayed cell.
- Untested requested classes: isolated punctuation changes, arbitrary prefixes/suffixes, and case conversion.
- Partially represented classes: line wrapping (line-ending proxy only), email quoting (Markdown blockquote proxy only), and isolated sentence-boundary edits (paraphrase proxy only).
- Failure-mechanism labels in the upstream evidence are descriptive first-divergence categories, not causal proof.
