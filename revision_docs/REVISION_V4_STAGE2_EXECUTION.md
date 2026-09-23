# Reproducing and resuming Stage 2

The frozen plans and retained checkpoints are authoritative for this execution. Do not rerun the Stage 1 scaffolder or overwrite the archived Stage 1 results. Stage 2 uses the current pinned generator environment and the existing general analysis environment. Neither was replaced.

## Completed offline setup

`rankcloak.revision_v4_coherence` performs vocabulary-only instrumentation and records all structural boundaries. `configs/revision_v4/stage2_coherence.json` records the initial design. The purpose-trained MiniLM amendment preceded all scores. The first tokenization-only preflight found Mistral's one-space rendering prefix. Its unscored inventory is retained as a diagnostic preflight, and the adjusted initial inventory was frozen before pilot scoring.

The first pilot is under `plans/pilot_initial`. Its one permitted methodological amendment is `configs/revision_v4/stage2_coherence_amendment1.json`. The fresh inventory and pilot are under `plans/amendment1` and `plans/pilot_amendment1`. Fresh controls exclude all 69 payload identities used as recipients or donors in the first pilot. The failed semantic validation is final for this stage. No additional model, layer, pooling, window or threshold search is authorized by this design.

`plans/coherence_study/freeze.json` freezes 92 selected payloads, 3,174 selected structural boundaries, 552 planned trials, and 2,750 eligible boundaries in 550 scored trials. All three models and both schedules remain in the selected structural sample. Two trials have no eligible three-arm boundary. The four selected artifact classes each contribute 23 payloads. The sample contains 8,250 arm-specific units. It is an exploratory context-gain study because semantic validation failed. The original upper bound of 24,840 transitions and 49,680 LM passes is not a measured runtime estimate.

The pilot-based full-population forecast was about 55,512 GPU-job seconds before the other-job and safety reserve. The selected-sample forecast was about 23,843 seconds, using the more conservative measured per-token rate from either pilot for each evaluator and a 1.75 multiplier. These are admission forecasts, not measured execution times. Actual times belong in the persistent ledger and Stage 2 report.

## GPU execution contract

All GPU jobs run through `scripts/run_revision_v4_stage2_job.py`. It locks the persistent ledger, checks the nominated device, counts model loading and unsuccessful attempts, enforces the cumulative 28,800-second ceiling and per-job limits, and checkpoints only its own worker. It does not stop unrelated GPU processes. The device is the RTX 5000 Ada UUID recorded in the ledger. The display GPU is not used.

A typical admitted command is shown below. Use a new job name for a resume, compute the missing-request forecast from the existing frozen requests and cached identities, and ensure that it fits the remaining ledger allowance with the necessary reserve. Never delete or reset the ledger to obtain a new allowance.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/run_revision_v4_stage2_job.py \
  --name coherence_study_qwen_resume1 \
  --forecast-seconds MEASURED_REMAINING_FORECAST \
  --max-seconds ADMITTED_HARD_LIMIT \
  -- .venv-generation-v3/bin/python -m rankcloak.revision_v4_stage2_scoring \
  --plan results/revision_v4/stage2/plans/coherence_study \
  --kind lm --model-id qwen2_5_7b_instruct_q4_k_m
```

Before a resume, compare the installed package versions and backend shared-library hashes with the retained provenance. A changed runtime requires a separate documented numerical validation and must not silently reuse these cache identities. Use the frozen corresponding IDs for Mistral and Llama. One worker runs at a time. A cache key hashes the exact scoring input, model and source/configuration contract. Contract changes, duplicate identities and corrupt request hashes are rejected. Do not start a GPU worker if all requests for it are already cached. Embedding batches passed a numerical fixture. Batched LM evaluation failed its numerical tolerance for all three evaluators, so all study LM scores retain serial batch and microbatch size one. No CPU model fallback or KV-sharing optimization was used.

## CPU reconstruction after all planned requests complete

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m rankcloak.revision_v4_stage2_analysis \
  --plan results/revision_v4/stage2/plans/coherence_study
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m rankcloak.revision_v4_quantization
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m rankcloak.revision_v4_transport analyze
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/audit_revision_v4_entropy_runs.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/audit_revision_v4_continuation_policies.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/prepare_revision_v4_stage2_text.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/render_revision_v4_stage2_results.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/render_revision_v4_stage2_response.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  scripts/build_revision_v4_stage2_documents.py
```

The join refuses an incomplete planned sample. The analysis never substitutes a convenience subset of completed rows. The result renderer uses completed derived tables. Its explicit `--preview` mode leaves an execution-pending marker and is not a final result. The Stage 2 validator rejects that preview mode in final validation. Generated document inserts may overwrite their own output files, so preserve any later author edits before rerendering.

The quantization endpoint uses only saved common-history ranks and original codec metadata. The transport worker processes only the missing segment inputs in its frozen plan. It preserves the historical raw unmodified rows as saved-token references and uses an executed unchanged-byte transformation as the visible-text baseline. This distinction was verified during CPU preparation before transport execution. A later audit found different allocated context capacities in the historical and new runs. The bounded reuse check therefore executes all 53 reused request identities at the Stage 2 capacity and requires exact rank/error agreement. These are checks of the same 25 logical cover arms, not additional transformation selections. They run after the frozen coherence jobs through the same supervisor. The validation plan and per-request results are retained under `transport`.

## Validation and handoff

Focused and relevant regression tests are recorded under `results/revision_v4/stage2/validation`. Final validation is `scripts/validate_revision_v4_stage2.py`, without `--preview`. It checks protected history, exact review blocks, answer evidence and manuscript anchors, frozen plan hashes, unchanged scorer contracts, complete planned-sample joins, numerical table claims, document builds and the cumulative GPU ceiling.

`revision_docs/REVISION_V4_RELEASE_SPEC.md` controls local release preparation. There is no publication capability in the inventory script. A final deposit must ultimately correspond to the finalized source/evidence commit and requires a real new version DOI after publication. The current DOI is not relabeled as V4 coverage.
