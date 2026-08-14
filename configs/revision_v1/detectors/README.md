# Revision detector configuration

The detector input is CSV or JSON Lines with one raw message per row and these
required fields:

- "text": exact detector-visible text.
- "label": 0 for cover and 1 for stego.
- "payload_group_id": the independent payload grouping unit. Every message
  derived from the same payload must share this value, including paired
  controls where applicable.
- "prompt_template_id", "model_id", and "codec_id": explicit evaluation
  dimensions. Controls must use meaningful matched values, not blanks.
- "row_id": optional only when the other fields uniquely identify every row.

"default.json" prespecifies the raw-text TextCNN-equivalent and DeBERTa
classifiers. PyTorch and Transformers are lazy optional dependencies. The
transformer is offline-only by default: downloads require both
"offline_only=false"/"allow_downloads=true" in a reviewed config and the
command-line "--allow-model-downloads" switch.

If neural dependencies or cached weights are absent, the requested detector can
fall back to a character n-gram logistic smoke test. Output rows record
"requested_kind", "implementation_kind", and "implementation_status"; a
fallback is never labelled as a neural result. Outside "--smoke", the CLI exits
nonzero for fallback output unless "--accept-smoke-fallback" is explicit.

Every split keeps complete payload groups together and checks row IDs, group
IDs, and exact-text hashes for train/test overlap. Held-out-template,
leave-one-model, and leave-one-codec splits purge any training row whose
payload group occurs in the held-out partition. Metrics and confidence
intervals resample complete payload groups.
