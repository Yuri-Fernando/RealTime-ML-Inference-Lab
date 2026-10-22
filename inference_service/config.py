"""Centralized configuration for the inference service.

All values are overridable via environment variables so the service can be
tuned (batch size, backpressure thresholds, etc.) without code changes, e.g.
for load testing different operating points.
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime settings, read once at import time from the environment."""

    # Model / device
    INPUT_DIM: int = _env_int("RTML_INPUT_DIM", 3 * 32 * 32)  # CIFAR-like flat vector
    NUM_CLASSES: int = _env_int("RTML_NUM_CLASSES", 10)
    FORCE_CPU: bool = _env_bool("RTML_FORCE_CPU", False)
    MODEL_SEED: int = _env_int("RTML_MODEL_SEED", 1234)

    # Dynamic batching
    MAX_BATCH_SIZE: int = _env_int("RTML_MAX_BATCH_SIZE", 16)
    MAX_BATCH_WAIT_MS: float = _env_float("RTML_MAX_BATCH_WAIT_MS", 10.0)

    # Backpressure
    MAX_QUEUE_DEPTH: int = _env_int("RTML_MAX_QUEUE_DEPTH", 64)

    # Cache
    CACHE_MAX_ENTRIES: int = _env_int("RTML_CACHE_MAX_ENTRIES", 4096)

    # Request timeout (seconds) while waiting for a batched result
    REQUEST_TIMEOUT_S: float = _env_float("RTML_REQUEST_TIMEOUT_S", 5.0)

    # --- V0.3: backends de geração de texto (llm_app.py) -------------------
    # tinycnn | hf | vllm. O app histórico (app.py) ignora este valor e usa
    # sempre o TinyCNN; ele só vale para `inference_service.llm_app`.
    BACKEND: str = os.getenv("RTML_BACKEND", "hf").strip().lower()

    # HuggingFaceBackend
    HF_MODEL: str = os.getenv("RTML_HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    # fp32 | fp16 | bf16 | int8 | nf4
    HF_QUANT: str = os.getenv("RTML_HF_QUANT", "fp16").strip().lower()
    HF_LOCAL_ONLY: bool = _env_bool("RTML_HF_LOCAL_ONLY", False)
    HF_CHAT_TEMPLATE: bool = _env_bool("RTML_HF_CHAT_TEMPLATE", True)

    # VLLMBackend (servidor OpenAI-compatible externo)
    VLLM_BASE_URL: str = os.getenv("RTML_VLLM_BASE_URL", "http://127.0.0.1:8001")
    VLLM_MODEL: str = os.getenv("RTML_VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    # Batching/admissão do caminho LLM (valores próprios: geração é bem mais
    # lenta que o forward do TinyCNN)
    LLM_MAX_BATCH_SIZE: int = _env_int("RTML_LLM_MAX_BATCH_SIZE", 8)
    LLM_MAX_BATCH_WAIT_MS: float = _env_float("RTML_LLM_MAX_BATCH_WAIT_MS", 20.0)
    LLM_MAX_QUEUE_DEPTH: int = _env_int("RTML_LLM_MAX_QUEUE_DEPTH", 64)
    LLM_REQUEST_TIMEOUT_S: float = _env_float("RTML_LLM_REQUEST_TIMEOUT_S", 120.0)
    LLM_MAX_NEW_TOKENS_CAP: int = _env_int("RTML_LLM_MAX_NEW_TOKENS_CAP", 128)


settings = Settings()
