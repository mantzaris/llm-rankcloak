# RankCloak Scientific Reports revision V3 computational handoff

This directory is the authoritative V3 computational extension. It reuses immutable V2 sources without overwriting them, includes the completed model-backed entropy and matched-quantization studies, and contains no manuscript edits.

Primary entry points are `methods_for_manuscript.md`, `results_for_manuscript.md`, `limitations_for_manuscript.md`, `claim_evidence_matrix.csv`, `run_manifest.json`, and `test_report.md`. Open source tables are under `source_tables`; generated LaTeX tables under `manuscript_tables`; vector figures and PNG previews under `figures`; row-level detector predictions under `detector_predictions`; raw atomic model-backed records under `generation/raw`; and exact/near-duplicate audits under `deduplication`.

## Complete rerun guide

Use a fresh output directory for corpus preparation. The four model downloads are exact, pinned inputs and must pass the configured size and SHA-256 checks; substitutions fail closed.

```bash
huggingface-cli download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl --repo-type dataset --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a --local-dir /tmp/rankcloak_revision_v3_sources_repro
sha256sum /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl
.venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import_repro
.venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/build_revision_v3_generation_plans.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python -m venv .venv-generation-v3
.venv-generation-v3/bin/pip install --no-cache-dir --only-binary=:all: llama-cpp-python==0.3.23 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
.venv-generation-v3/bin/pip install --no-cache-dir nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --revision a06c33ec89c1e3402009fb47f466a89127c6d223 --local-dir models/llama3_8b
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --revision 61fd4167fff3ab01ee1cfe0da183fa27a944db48 --local-dir models/mistral_7b_instruct_v0_3
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031 --local-dir models/qwen2_5_7b_instruct
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q8_0.gguf --revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031 --local-dir models/qwen2_5_7b_instruct
sha256sum models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf models/mistral_7b_instruct_v0_3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf models/qwen2_5_7b_instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf models/qwen2_5_7b_instruct/Qwen2.5-7B-Instruct-Q8_0.gguf
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id llama3_8b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id mistral_7b_instruct_v0_3_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy_calibration --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id llama3_8b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id mistral_7b_instruct_v0_3_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase entropy --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q4_k_m --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_generation.py --phase quantization --model-id qwen2_5_7b_instruct_q8_0 --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
PYTHONPATH=. .venv-generation-v3/bin/python scripts/run_revision_v3_q4_visible_recovery.py --gpu-uuid GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf --output-dir /tmp/rankcloak_revision_v3_repro/generation
.venv/bin/python scripts/analyze_revision_v3_generation.py --output-dir /tmp/rankcloak_revision_v3_repro --generation-dir /tmp/rankcloak_revision_v3_repro/generation
.venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --prepare-only --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector surprisal --output-dir /tmp/rankcloak_revision_v3_repro
CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf .venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector textcnn --output-dir /tmp/rankcloak_revision_v3_repro
CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf .venv/bin/python scripts/run_revision_v3_generation_detectors.py --study all --detector deberta --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python -m pytest -q tests/test_revision_v3_*.py --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_v3_focused.xml
.venv/bin/python -m pytest -q --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_full.xml
.venv/bin/python scripts/finalize_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/validate_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
```

The Dolly checksum is `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`. The model sizes and hashes are recorded in `configs/revision_v3/generation_requirements.json` and verified again in `provenance/generation_preflight.json`. Generation is resumable: completed trial and recovery records are immutable and the runners refuse silent overwrite. Run Q4 model-backed quantization replay before Q8 because each Q8 record binds to the exact paired Q4 replay hash. The Q4 visible-recovery runner retokenizes and, when needed, rank-replays only the existing historical covers; it does not generate covers. No paid remote compute is part of this workflow.

Re-running analysis and finalization from unchanged authoritative ledgers regenerates source tables, LaTeX, figures, prose, source maps, and checksums. `provenance/artifact_source_map.csv` maps every publication artifact to its source and command.
