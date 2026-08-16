# Automated-readability figure technical note

- Authoritative source: `results/revision_v1/final_experiment_package/readability/selected_stimulus_readability_summary.csv`.
- Evidence classification: supporting automated surface diagnostics, not human evaluation; 21 plotted rows (seven conditions by three outcomes).
- All plotted rows have `human_rating_substitute=false`.
- Intervals are 95% prompt-template-cluster percentile-bootstrap intervals (18 prompt-template units; 72 stimuli per condition).
- `Surface-flag count` sums unmatched brackets, double-quote imbalance, repeated punctuation, whitespace flags, lowercase sentence starts, sentences longer than 40 words, missing terminal punctuation, and long hexadecimal/base64-like fragments.
- `Prompt similarity` is TF-IDF cosine similarity between cover text and prompt text.
- Flesch ease is an English surface heuristic. Similar values do not establish naturalness or human-perceived quality.
- No composite naturalness score or synthetic human score is computed.
