"""Backends de inferência plugáveis (V0.3).

Importações pesadas (torch/transformers/httpx) ficam nos submódulos; este
pacote só reexporta o contrato e a fábrica.
"""
from inference_service.backends.base import (
    BackendConfigError,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    TextBackend,
)
from inference_service.backends.factory import BACKEND_NAMES, create_backend

__all__ = [
    "BACKEND_NAMES",
    "BackendConfigError",
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "TextBackend",
    "create_backend",
]
