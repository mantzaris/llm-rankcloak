# Neural-detector figure technical note

- Regime source: `results/revision_v1/final_experiment_package/detectors/analysis/detector_plot_source.csv`.
- Matched confidence-interval source: `results/revision_v1/final_experiment_package/detectors/analysis/detector_extended_metrics.csv`.
- Evidence classification: confirmatory frozen endpoints plus explicitly exploratory post-freeze endpoints; 24 rows in the compact candidate and 48 rows in the full supporting candidate.
- Matched bars are payload-group bootstrap 95% confidence intervals.
- Held-out bars are minimum-to-maximum ranges across heterogeneous prespecified splits; they are not confidence intervals.
- ROC-AUC, PR-AUC, and balanced accuracy retain confirmatory frozen-upstream status.
- Precision, Brier score, and TPR at FPR <= 1% are supplementary exploratory post-freeze metrics.
- Brier score is lower-is-better. High values in the other detector panels indicate weaker concealment, not a favorable RankCloak outcome.
- The lexical near-duplicate sensitivity analysis is not plotted here and does not remove the underlying limitation.
