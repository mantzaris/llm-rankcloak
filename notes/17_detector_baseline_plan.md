# Detector Baseline Plan

## Purpose

The detector baseline is a modest feature-only steganalysis check. It is not a claim of
undetectability or strong detection performance.

## Dataset Construction

The paper suite writes `paper_cover_text_features.csv` and derives
`detector_dataset.csv` from it.

Labels:

- `label = 0`: baseline greedy cover text.
- `label = 1`: RankCloak cover text.

Dataset views:

- `detector_full_message`: baseline versus non-segmented RankCloak and segmented full messages.
- `detector_forced_prefix`: baseline rows versus segmented forced prefixes, with comparability caveats.
- `detector_nonseg_only`: baseline versus non-segmented RankCloak.
- `detector_segmented_full_only`: baseline versus segmented full messages.

## Features

The detector uses numeric and Boolean feature columns only. It does not train on text
content directly.

Candidate features include token count, character count, line count, whitespace
fraction, punctuation fraction, digit fraction, alphabetic fraction, unique-token
fraction, repeated-token fraction, token log probability, generated-rank summaries, and
artifact flags.

## Models

The dependency-free baseline is a threshold on mean token log probability.

If scikit-learn is installed, the suite also attempts:

- logistic regression with standardized features;
- random forest with shallow trees.

If scikit-learn is missing, detector output remains available through the threshold
baseline and the summary should report that the sklearn baselines were not run.

## Splits

The suite uses a deterministic stratified split. When enough rows are available, it also
attempts leave-one-prompt-family-out and leave-one-payload-class-out splits.

## Caveats

Detector results are lightweight baselines. They should not be presented as conclusive
steganalysis or as evidence of undetectability.
