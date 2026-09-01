# RankCloak Scientific Reports revision V3 computational handoff

This directory contains the authoritative V3 computational extension. It reuses immutable V2 source trials but does not overwrite them, and it contains no manuscript edits.

Primary entry points are methods_for_manuscript.md, results_for_manuscript.md, limitations_for_manuscript.md, claim_evidence_matrix.csv, run_manifest.json, and test_report.md. Source tables are under source_tables; generated LaTeX tables are under manuscript_tables; vector figures and PNG previews are under figures; row-level predictions are under detector_predictions; deduplication and leakage ledgers are under deduplication.

## Complete rerun guide

Preparation intentionally refuses a nonempty output directory. Use a fresh path such as `/tmp/rankcloak_revision_v3_repro`; never point preparation at the authoritative V3 result directory.

```bash
huggingface-cli download databricks/databricks-dolly-15k databricks-dolly-15k.jsonl --repo-type dataset --revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a --local-dir /tmp/rankcloak_revision_v3_sources_repro
sha256sum /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl
.venv/bin/python human_study/controls/prepare_controls.py import --input /tmp/rankcloak_revision_v3_sources_repro/databricks-dolly-15k.jsonl --source-id databricks_dolly_15k_v1_pinned --acquisition-date 2026-08-31 --output-dir /tmp/rankcloak_revision_v3_dolly_import_repro
.venv/bin/python scripts/prepare_revision_v3.py --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/build_revision_v3_generation_plans.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector surprisal --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector textcnn --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/run_revision_v3_detectors.py --detector deberta --evaluation all --human-candidates /tmp/rankcloak_revision_v3_dolly_import_repro/human_control_candidates.jsonl --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python -m pytest -q --junitxml=/tmp/rankcloak_revision_v3_repro/logs/pytest_full.xml
.venv/bin/python scripts/finalize_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
.venv/bin/python scripts/validate_revision_v3.py --output-dir /tmp/rankcloak_revision_v3_repro
```

The Dolly checksum must equal `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`; the import command itself fails closed on any mismatch. The neural detector uses the locally pinned, offline DeBERTa artifact described in its fit ledgers. Select the RTX 5000 Ada device if CUDA enumeration differs.

The generation-plan command is dry-run only. It does not download weights or launch inference. Exact blocked artifact names, revisions, sizes, SHA-256 values, acquisition commands, backend installation command, storage, and runtime estimates are in `configs/revision_v3/generation_requirements.json` and `provenance/generation_preflight.json`.

Re-running finalization from unchanged metrics regenerates tables, figures, prose, and manifests. `provenance/artifact_source_map.csv` maps each publication artifact to its source and command.
