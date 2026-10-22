"""Seleção de backend e validação de quantização (sem carregar modelo)."""
import pytest

from inference_service.backends import (
    BACKEND_NAMES,
    BackendConfigError,
    InferenceBackend,
    create_backend,
)
from inference_service.backends.hf import QUANT_MODES, build_load_kwargs, resolve_quant
from inference_service.config import settings


def test_backend_names_documented():
    assert BACKEND_NAMES == ("tinycnn", "hf", "vllm")


def test_tinycnn_is_a_pluggable_backend():
    from inference_service.model import ModelBackend, TinyCNNBackend

    backend = create_backend("tinycnn")
    assert isinstance(backend, TinyCNNBackend)
    assert TinyCNNBackend is ModelBackend  # mesmo backend histórico, sem fork
    assert backend.kind == "tensor"
    assert isinstance(backend, InferenceBackend)
    assert backend.describe()["backend"] == "tinycnn"


async def test_vllm_backend_is_selected_without_network():
    from inference_service.backends.vllm_backend import VLLMBackend

    backend = create_backend("vllm", base_url="http://127.0.0.1:9", model="m")
    try:
        assert isinstance(backend, VLLMBackend)
        assert backend.kind == "text"
        assert backend.supports_batching is False
        assert backend.describe()["base_url"] == "http://127.0.0.1:9"
    finally:
        await backend.aclose()


def test_unknown_backend_fails_loudly():
    with pytest.raises(BackendConfigError, match="desconhecido"):
        create_backend("onnx-magic")


async def test_backend_defaults_to_env_setting(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND", "vllm")
    backend = create_backend()
    try:
        assert backend.name == "vllm"
    finally:
        await backend.aclose()


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_cpu_capable_modes(mode):
    assert resolve_quant(mode, cuda_available=False).mode == mode


@pytest.mark.parametrize("mode", ["fp16", "int8", "nf4"])
def test_cuda_only_modes_rejected_without_gpu(mode):
    with pytest.raises(BackendConfigError, match="CUDA"):
        resolve_quant(mode, cuda_available=False)
    assert resolve_quant(mode, cuda_available=True).mode == mode


def test_unknown_quant_rejected():
    with pytest.raises(BackendConfigError, match="desconhecido"):
        resolve_quant("int3", cuda_available=True)


def test_quant_table_bits():
    assert {m: s.bits for m, s in QUANT_MODES.items()} == {
        "fp32": 32,
        "fp16": 16,
        "bf16": 16,
        "int8": 8,
        "nf4": 4,
    }


def test_load_kwargs_map_to_real_bitsandbytes_configs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    fp32 = build_load_kwargs(QUANT_MODES["fp32"])
    assert fp32["dtype"] is torch.float32 and "quantization_config" not in fp32

    int8 = build_load_kwargs(QUANT_MODES["int8"])["quantization_config"]
    assert int8.load_in_8bit is True and int8.load_in_4bit is False

    nf4 = build_load_kwargs(QUANT_MODES["nf4"])["quantization_config"]
    assert nf4.load_in_4bit is True
    assert nf4.bnb_4bit_quant_type == "nf4"
    assert nf4.bnb_4bit_compute_dtype is torch.float16


def test_hf_backend_rejects_int8_on_cpu_before_loading_anything():
    pytest.importorskip("transformers")
    from inference_service.backends.hf import HuggingFaceBackend

    with pytest.raises(BackendConfigError, match="CUDA"):
        HuggingFaceBackend(quant="int8", force_cpu=True)
