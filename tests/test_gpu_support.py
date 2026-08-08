from pathlib import Path
from types import SimpleNamespace
import os
import sys

import pytest

from rankcloak import cli, model_io, reproducibility
from rankcloak.reproducibility import command_line_option_int


def test_load_model_forwards_gpu_layer_count(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_io, "llama_cpp_gpu_offload_supported", lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    model = model_io.load_llama_cpp_model(model_path=model_path, n_gpu_layers=-1)

    assert captured["n_gpu_layers"] == -1
    assert model.rankcloak_n_gpu_layers == -1
    assert model.rankcloak_gpu_offload_supported is True
    assert os.environ["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_explicit_gpu_request_rejects_cpu_only_build(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_io, "llama_cpp_gpu_offload_supported", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=lambda **kwargs: object()),
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    with pytest.raises(RuntimeError, match="does not support GPU offload"):
        model_io.load_llama_cpp_model(model_path=model_path, n_gpu_layers=-1)


def test_gpu_layer_count_below_minus_one_is_invalid(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=lambda **kwargs: object()),
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    with pytest.raises(ValueError, match="must be -1"):
        model_io.load_llama_cpp_model(model_path=model_path, n_gpu_layers=-2)


def test_reset_model_clears_llama_kv_cache_after_reset():
    calls = []
    model = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        _ctx=SimpleNamespace(kv_cache_clear=lambda: calls.append("clear")),
    )

    model_io.reset_model(model)

    assert calls == ["reset", "clear"]


def test_rankcloak_cli_forwards_gpu_layer_count(monkeypatch):
    captured = {}

    def fake_experiment_main(arguments):
        captured["arguments"] = arguments
        return {"ok": True}

    monkeypatch.setattr(cli, "experiment_main", fake_experiment_main)

    result = cli.main(["run", "--profile", "paper-smoke", "--n-gpu-layers", "-1"])

    assert result == {"ok": True}
    index = captured["arguments"].index("--n-gpu-layers")
    assert captured["arguments"][index + 1] == "-1"


def test_manifest_records_gpu_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(reproducibility, "llama_cpp_gpu_offload_supported", lambda: True)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    manifest = reproducibility.write_manifest(
        output_path=tmp_path / "MANIFEST.json",
        project_root=tmp_path,
        profile="paper-smoke",
        output_dir=tmp_path,
        command_line_args=["--n-gpu-layers", "-1"],
        model_repo_id="test/repo",
        model_filename=model_path.name,
        model_path=model_path,
    )

    assert manifest["inference_backend"] == {
        "n_gpu_layers_requested": -1,
        "llama_cpp_gpu_offload_supported": True,
        "gpu_backend_active": True,
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "1",
        "ggml_cuda_disable_graphs": "1",
        "cublas_workspace_config": ":4096:8",
    }
    assert manifest["model_sha256"] is not None


def test_command_line_option_int_accepts_split_and_equals_forms():
    assert command_line_option_int(["--n-gpu-layers", "-1"], "--n-gpu-layers") == -1
    assert command_line_option_int(["--n-gpu-layers=12"], "--n-gpu-layers") == 12
    assert command_line_option_int([], "--n-gpu-layers", default=0) == 0
