"""Reproducibility manifest helpers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from .model_io import default_thread_count, llama_cpp_gpu_offload_supported


DEFAULT_MODEL_HASH_LIMIT_BYTES = 8 * 1024 * 1024 * 1024


def repo_relative_path(path: Optional[Path], project_root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(project_root.resolve()))
    except Exception:
        try:
            return str(Path(path).relative_to(project_root))
        except Exception:
            return str(path)


def package_version(distribution_name: str, import_name: Optional[str] = None) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name)
    except Exception:
        pass
    if import_name:
        try:
            module = importlib.import_module(import_name)
            return getattr(module, "__version__", None)
        except Exception:
            return None
    return None


def installed_package_versions() -> Dict[str, Optional[str]]:
    return {
        "numpy": package_version("numpy", "numpy"),
        "pandas": package_version("pandas", "pandas"),
        "matplotlib": package_version("matplotlib", "matplotlib"),
        "llama_cpp": package_version("llama-cpp-python", "llama_cpp"),
        "huggingface_hub": package_version("huggingface-hub", "huggingface_hub"),
        "nvidia_cuda_runtime_cu12": package_version("nvidia-cuda-runtime-cu12"),
        "nvidia_cublas_cu12": package_version("nvidia-cublas-cu12"),
    }


def git_commit_hash(project_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_dirty_worktree(project_root: Path) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def sha256_file(path: Path, max_bytes: int = DEFAULT_MODEL_HASH_LIMIT_BYTES) -> Optional[str]:
    if not path.exists():
        return None
    file_size = path.stat().st_size
    if file_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_line_option_int(
    command_line_args: Iterable[str], option: str, default: Optional[int] = None
) -> Optional[int]:
    """Read an integer option from a recorded argv sequence."""

    values = list(command_line_args)
    for index, value in enumerate(values):
        if value == option and index + 1 < len(values):
            try:
                return int(values[index + 1])
            except (TypeError, ValueError):
                return default
        prefix = option + "="
        if str(value).startswith(prefix):
            try:
                return int(str(value)[len(prefix) :])
            except (TypeError, ValueError):
                return default
    return default


def write_manifest(
    output_path: Path,
    project_root: Path,
    profile: str,
    output_dir: Path,
    command_line_args: Iterable[str],
    model_repo_id: str,
    model_filename: str,
    model_path: Optional[Path],
) -> Dict[str, object]:
    """Write a repo-relative reproducibility manifest."""

    command_line_args = list(command_line_args)
    n_gpu_layers_requested = command_line_option_int(
        command_line_args, "--n-gpu-layers", default=0
    )
    gpu_offload_supported = llama_cpp_gpu_offload_supported()
    model_file_size_bytes = model_path.stat().st_size if model_path and model_path.exists() else None
    model_sha256 = sha256_file(model_path) if model_path and model_path.exists() else None
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash(project_root),
        "dirty_worktree": git_dirty_worktree(project_root),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "selected_thread_count": default_thread_count(),
        "inference_backend": {
            "n_gpu_layers_requested": n_gpu_layers_requested,
            "llama_cpp_gpu_offload_supported": gpu_offload_supported,
            "gpu_backend_active": bool(n_gpu_layers_requested and gpu_offload_supported),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ggml_cuda_disable_graphs": os.environ.get("GGML_CUDA_DISABLE_GRAPHS"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "installed_package_versions": installed_package_versions(),
        "model_repo_id": model_repo_id,
        "model_filename": model_filename,
        "model_path_relative": repo_relative_path(model_path, project_root),
        "model_sha256": model_sha256,
        "model_sha256_note": (
            "skipped because the model file exceeds the local hash limit"
            if model_file_size_bytes and model_sha256 is None
            else None
        ),
        "model_file_size_bytes": model_file_size_bytes,
        "profile": profile,
        "output_dir": repo_relative_path(output_dir, project_root),
        "command_line_args": command_line_args,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

