# Intact-message contextual acceptability follow-up

This is a new study designed after the earlier fragment results. The frozen plan precedes final judging. The semantic pilots, local-likelihood study and all previous stage records remain unchanged in their original namespaces.

The primary question asks whether a complete delivered message can be followed under its authentic prompt while allowing minor awkwardness. Scores 0 and 1 are acceptable. Scores 2 and 3 indicate material disruption. Logical connectedness from 1 to 5 is secondary. These are automated model ratings, not human perception measurements.

- `plans/final_sample/` contains the population, exact selected covers, matched-control requests, all logical judge mappings and the timestamped freeze with source hashes and cost admission.
- `raw/calibration/` retains all constructed calibration responses. `plans/calibration_cases.json` contains the intended labels fixed before inference.
- `raw/controls/` contains 54 greedy control outputs. `raw/judging/` retains the blinded input, rendered chat prompt, structured response and all attempts for every unique request.
- `gpu/ledger.json` charges all model-job wall time against six hours, including loading and checks. Per-job stdout, stderr and execution receipts retain CUDA evidence.
- `plans/control_capacity_check.json` declares a technical comparison at the historical allocation. `raw/control_capacity/` preserves its outputs without replacing any frozen study control.
- `analysis/` contains the complete joined scores, hierarchical estimates, four-level distributions, selected examples and post-scoring literal-quote mapping to boundaries.
- `manuscript/` contains rendered figure/table inputs. `response_status.json` records written completion, automated empirical evidence, missing human evidence and separate archive actions.
- `validation/` contains exact source joins, independent point/interval recomputation, scientific tests, builds, document review and reproduction receipts.

Use the existing local environment. All steps below are offline and require no model weights or GPU inference.

```bash
.venv/bin/python -m scripts.analyze_revision_v4_contextual_coherence --calibration
.venv/bin/python -m scripts.analyze_revision_v4_contextual_coherence --plan final_sample
.venv/bin/python -m scripts.audit_revision_v4_contextual_coherence
.venv/bin/python -m scripts.render_revision_v4_contextual_results
```

The inference configuration is `configs/revision_v4/coherence_replacement.json`. It pins weights, embedded chat templates, decoding, retry rules, selection and analysis. Scientific inference is checkpointed by full request and contract identity. Do not rerun the prospective freezer after outcomes exist. Do not edit frozen scorer files to reproduce a different result under the same identity.

Build current documents with `scripts.build_revision_v4_contextual_documents`, first `--phase scientific`, then refresh response locations with `scripts.render_revision_v4_contextual_response`, then build `--phase correspondence` and `--phase submission`. These workflows use new logs and preserve historical stage tools and evidence. The editable manuscript is tested in a fresh temporary directory, without creating an upload bundle.

The earlier local release candidate and published V3 archive predate this follow-up. No release assembly, publication or journal submission is part of this study.
