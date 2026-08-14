# Offline environment verification

These commands do not download, install, run experiments, or publish a release.

```bash
.venv/bin/python scripts/build_revision_environment_lock.py \
  --output-dir environment/revision_v1 --check
.venv/bin/python scripts/build_revision_environment_lock.py \
  --output-dir environment/revision_v1 --check --verify-model-files
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf \
CUDA_LAUNCH_BLOCKING=1 GGML_CUDA_DISABLE_GRAPHS=1 \
GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv/bin/python scripts/run_revision_matrix.py --help
analysis/revision_v1/run_with_locked_r.sh --version
.venv/bin/python scripts/build_revision_confirmatory_release_index.py --dry-run
```

The final three commands are preflight demonstrations only. Running experiments or installing the missing project-local R packages is a separate controlled action.
