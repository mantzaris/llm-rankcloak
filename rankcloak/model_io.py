"""Model download, loading, and tokenizer helpers for RankCloak."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LLAMA3_DIR = MODELS_DIR / "llama3_8b"


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    destination: Path


PREFERRED_LLAMA3_SPEC = ModelSpec(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    destination=LLAMA3_DIR / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
)

FALLBACK_LLAMA3_SPEC = ModelSpec(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    destination=LLAMA3_DIR / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)


def default_thread_count() -> int:
    """Return a CPU thread count that leaves one core free when possible."""

    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


def existing_llama3_model_path() -> Optional[Path]:
    """Return the first known local Llama 3 GGUF path if it exists."""

    for spec in (PREFERRED_LLAMA3_SPEC, FALLBACK_LLAMA3_SPEC):
        if spec.destination.exists():
            return spec.destination
    return None


def known_model_specs() -> Dict[str, ModelSpec]:
    """Return the preferred and fallback model specs keyed by short name."""

    return {
        "preferred": PREFERRED_LLAMA3_SPEC,
        "fallback": FALLBACK_LLAMA3_SPEC,
    }


def download_llama3_gguf(force: bool = False) -> Path:
    """Download the preferred Llama 3 GGUF, falling back to the alternate repo.

    The caller should ask before invoking this because the file is roughly
    several gigabytes. Hugging Face authentication or license failures are
    re-raised with a clear message so the notebook does not fail silently.
    """

    existing = existing_llama3_model_path()
    if existing and not force:
        return existing

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the GGUF model. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    LLAMA3_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for spec in (PREFERRED_LLAMA3_SPEC, FALLBACK_LLAMA3_SPEC):
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    local_dir=str(spec.destination.parent),
                )
            )
            if downloaded_path != spec.destination and downloaded_path.exists():
                shutil.copy2(downloaded_path, spec.destination)
            return spec.destination
        except Exception as exc:  # pragma: no cover - network/auth dependent
            failures.append(f"{spec.repo_id}/{spec.filename}: {exc}")

    message = (
        "Could not download the Llama 3 8B Instruct GGUF model.\n"
        "If Hugging Face requires a token or license acceptance, sign in with "
        "`huggingface-cli login`, accept the model terms in the browser, or "
        "manually place one of these files under models/llama3_8b/.\n"
        "Failures:\n- "
        + "\n- ".join(failures)
    )
    raise RuntimeError(message)


def preload_pip_cuda_libraries() -> List[str]:
    """Preload CUDA libraries installed by NVIDIA pip packages when present.

    CUDA llama.cpp wheels link against libcudart and cuBLAS. The NVIDIA pip
    packages install those libraries below a namespace package rather than a
    system linker path, so load them globally before importing llama_cpp.
    """

    try:
        import ctypes
        import nvidia
    except Exception:
        return []

    relative_paths = (
        Path("cuda_runtime/lib/libcudart.so.12"),
        Path("cublas/lib/libcublasLt.so.12"),
        Path("cublas/lib/libcublas.so.12"),
    )
    loaded = []
    for namespace_path in getattr(nvidia, "__path__", []):
        root = Path(namespace_path)
        for relative_path in relative_paths:
            candidate = root / relative_path
            if not candidate.exists() or str(candidate) in loaded:
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            loaded.append(str(candidate))
    return loaded


def load_llama_cpp_model(
    model_path: Optional[Path] = None,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    logits_all: bool = True,
    verbose: bool = False,
) -> Any:
    """Load a local GGUF model with optional llama.cpp GPU offload.

    ``n_gpu_layers=0`` preserves the CPU-only behavior used by the original
    experiments. ``n_gpu_layers=-1`` requests full offload; a positive value
    requests that many layers. Explicit GPU requests fail before model loading
    when the installed llama.cpp backend has no GPU-offload support.
    """

    n_gpu_layers = int(n_gpu_layers)
    if n_gpu_layers < -1:
        raise ValueError(
            "n_gpu_layers must be -1 (all), 0 (CPU), or a positive integer."
        )
    if n_gpu_layers != 0:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
        os.environ.setdefault("GGML_CUDA_DISABLE_FUSION", "1")
        os.environ.setdefault("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    preload_pip_cuda_libraries()
    try:
        from llama_cpp import Llama
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "llama-cpp-python is required for model-backed rank experiments. "
            "Install it with `pip install llama-cpp-python`. CUDA wheels also "
            "require nvidia-cuda-runtime-cu12 and nvidia-cublas-cu12."
        ) from exc

    resolved_model_path = Path(model_path) if model_path else existing_llama3_model_path()
    if resolved_model_path is None or not resolved_model_path.exists():
        expected = "\n".join(str(spec.destination) for spec in known_model_specs().values())
        raise FileNotFoundError(
            "No supported Llama 3 GGUF model was found. Expected one of:\n"
            f"{expected}"
        )

    if int(n_gpu_layers) != 0 and not llama_cpp_gpu_offload_supported():
        raise RuntimeError(
            "GPU layers were requested, but the installed llama-cpp-python build "
            "does not support GPU offload. Install a CUDA-enabled build or run with "
            "--n-gpu-layers 0."
        )

    threads = n_threads or default_thread_count()
    model_kwargs = {
        "model_path": str(resolved_model_path),
        "n_ctx": n_ctx,
        "n_threads": threads,
        "n_gpu_layers": int(n_gpu_layers),
        "logits_all": logits_all,
        "verbose": verbose,
    }
    if n_gpu_layers != 0:
        # Rank generation and recovery evaluate one payload token at a time.
        # Matching that execution shape for prompt evaluation prevents
        # intermittent CUDA batch/context drift after a context reset.
        model_kwargs.update({"n_batch": 1, "n_ubatch": 1})
    model = Llama(
        **model_kwargs,
    )
    for attribute, value in (
        ("rankcloak_model_path", str(resolved_model_path)),
        ("rankcloak_n_gpu_layers", int(n_gpu_layers)),
        ("rankcloak_n_batch", model_kwargs.get("n_batch")),
        ("rankcloak_n_ubatch", model_kwargs.get("n_ubatch")),
        ("rankcloak_gpu_offload_supported", llama_cpp_gpu_offload_supported()),
    ):
        try:
            setattr(model, attribute, value)
        except Exception:
            pass
    return model


def llama_cpp_gpu_offload_supported() -> bool:
    """Return whether the installed llama.cpp backend supports GPU offload."""

    preload_pip_cuda_libraries()
    try:
        from llama_cpp import llama_cpp as llama_cpp_api

        return bool(llama_cpp_api.llama_supports_gpu_offload())
    except Exception:
        return False


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def get_bos_token_id(model: Any) -> Optional[int]:
    """Return the BOS token id if llama-cpp-python exposes it."""

    for name in ("token_bos", "bos_token_id"):
        value = getattr(model, name, None)
        if value is None:
            continue
        try:
            token_id = int(_call_or_value(value))
        except Exception:
            continue
        if token_id >= 0:
            return token_id
    return None


def get_vocab_size(model: Any) -> Optional[int]:
    """Best-effort vocabulary size lookup for llama-cpp-python objects."""

    for candidate in (model, getattr(model, "_model", None)):
        if candidate is None:
            continue
        for name in ("n_vocab", "vocab_size"):
            value = getattr(candidate, name, None)
            if value is None:
                continue
            try:
                size = int(_call_or_value(value))
            except Exception:
                continue
            if size > 0:
                return size
    return None


def tokenize_bytes(model: Any, text_bytes: bytes, add_bos: bool = True) -> List[int]:
    """Compatibility wrapper around llama-cpp-python tokenization."""

    try:
        return list(model.tokenize(text_bytes, add_bos=add_bos))
    except TypeError:
        token_ids = list(model.tokenize(text_bytes))
        if not add_bos:
            bos_id = get_bos_token_id(model)
            if bos_id is not None and token_ids and token_ids[0] == bos_id:
                return token_ids[1:]
        return token_ids


def tokenize_payload_text(model: Any, text: str) -> List[int]:
    """Tokenize payload display text with the same leading-space convention."""

    token_ids = tokenize_bytes(model, (" " + text).encode("utf-8"), add_bos=True)
    bos_id = get_bos_token_id(model)
    if token_ids and bos_id is not None and token_ids[0] == bos_id:
        return token_ids[1:]
    if token_ids:
        return token_ids[1:]
    return []


def make_context_token_ids(model: Any, prompt: str) -> List[int]:
    """Tokenize a cover prompt/key into a non-empty autoregressive context."""

    if prompt:
        token_ids = tokenize_bytes(model, prompt.encode("utf-8"), add_bos=True)
        bos_id = get_bos_token_id(model)
        if token_ids and bos_id is not None and token_ids[0] == bos_id:
            return token_ids[1:]
        if token_ids:
            return token_ids[1:]

    bos_id = get_bos_token_id(model)
    if bos_id is None:
        raise ValueError("The model does not expose a BOS token for empty context.")
    return [bos_id]


def safe_detokenize(model: Any, token_ids: List[int]) -> str:
    """Detokenize token ids for display using replacement on UTF-8 errors."""

    try:
        raw = model.detokenize(list(map(int, token_ids)))
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception:
        return "".join("<tok:{}>".format(int(token_id)) for token_id in token_ids)


def reset_model(model: Any) -> None:
    """Reset tokens and fully clear llama.cpp's attention cache when available.

    ``Llama.reset()`` only resets its Python-side token counter. Clearing the
    underlying KV-cache data as well is important for rank replay on GPU: stale
    device-buffer contents can otherwise perturb near-tied logits after a
    context is recomputed.
    """

    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()
    context = getattr(model, "_ctx", None)
    clear_cache = getattr(context, "kv_cache_clear", None)
    if callable(clear_cache):
        clear_cache()


def evaluate_context(model: Any, context_token_ids: List[int]) -> None:
    """Reset and evaluate a context before reading next-token logits."""

    reset_model(model)
    if not context_token_ids:
        raise ValueError("Context token ids must be non-empty.")
    model.eval(list(map(int, context_token_ids)))


def get_last_logits(model: Any) -> np.ndarray:
    """Return logits for the next token after the current evaluated context."""

    n_tokens_value = getattr(model, "n_tokens", None)
    if n_tokens_value is None:
        raise AttributeError("Model does not expose n_tokens after eval().")
    n_tokens = int(_call_or_value(n_tokens_value))
    scores = getattr(model, "scores", None)
    if scores is None:
        raise AttributeError("Model does not expose scores; load with logits_all=True.")
    logits = np.asarray(scores[n_tokens - 1], dtype=np.float64)
    if logits.ndim != 1:
        logits = np.asarray(logits).reshape(-1)
    return logits
