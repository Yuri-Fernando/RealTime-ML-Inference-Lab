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


settings = Settings()
