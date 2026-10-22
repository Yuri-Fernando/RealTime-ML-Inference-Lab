"""Seleção de backend por configuração (``RTML_BACKEND``) ou argumento."""
from __future__ import annotations

from typing import Optional

from inference_service.backends.base import BackendConfigError
from inference_service.config import settings

BACKEND_NAMES = ("tinycnn", "hf", "vllm")


def create_backend(name: Optional[str] = None, **kwargs):
    """Instancia um backend. Importações pesadas são feitas sob demanda, então
    escolher ``tinycnn`` ou ``vllm`` não carrega torch/transformers à toa."""
    key = (name or settings.BACKEND or "").strip().lower()
    if key == "tinycnn":
        from inference_service.model import TinyCNNBackend

        return TinyCNNBackend(**kwargs)
    if key == "hf":
        from inference_service.backends.hf import HuggingFaceBackend

        return HuggingFaceBackend(**kwargs)
    if key == "vllm":
        from inference_service.backends.vllm_backend import VLLMBackend

        return VLLMBackend(**kwargs)
    raise BackendConfigError(f"backend desconhecido: {name!r} (válidos: {list(BACKEND_NAMES)})")
