"""Real GPU telemetry, read on demand (not estimated/simulated).

Two independent data sources are used, both real:

- `torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()`: always
  available whenever CUDA is available in this process — these come
  straight from the CUDA driver via PyTorch's caching allocator, no extra
  dependency needed.
- `torch.cuda.utilization()`: PyTorch's wrapper, which itself requires
  `nvidia-ml-py` (pynvml) to talk to NVML. Without that package it raises
  `RuntimeError` — handled here as "measured but currently unavailable",
  never faked as 0 or omitted silently without a documented reason.

This module intentionally does NOT estimate anything: if a value cannot be
read from the driver/NVML in the current environment, the corresponding
Prometheus gauge is simply left unset for that scrape rather than publishing
a made-up number.
"""
from __future__ import annotations

import torch


def gpu_available() -> bool:
    return torch.cuda.is_available()


def read_memory_allocated_bytes() -> int:
    """torch.cuda.memory_allocated() for device 0 — 0 if no CUDA GPU in use."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def read_memory_reserved_bytes() -> int:
    """torch.cuda.memory_reserved() for device 0 — 0 if no CUDA GPU in use."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_reserved())


def read_utilization_percent() -> "float | None":
    """GPU utilization percent via torch.cuda.utilization() (NVML-backed).

    Returns None (not 0) when unavailable — e.g. no GPU, or nvidia-ml-py not
    installed/importable, or the driver query itself failed — so callers can
    distinguish "measured 0% idle GPU" from "could not measure".
    """
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.utilization())
    except Exception:
        # torch.cuda.utilization() raises RuntimeError when nvidia-ml-py
        # is missing or NVML otherwise can't be queried in this environment.
        return None


def update_metrics(metrics_module) -> None:
    """Push a fresh real reading into the Prometheus gauges in `metrics.py`.

    Called right before each /metrics scrape (see app.py) so the exposed
    values are as fresh as the scrape itself, not a stale background
    sampler running on its own schedule.
    """
    metrics_module.GPU_AVAILABLE.set(1 if gpu_available() else 0)
    metrics_module.GPU_MEMORY_ALLOCATED_BYTES.set(read_memory_allocated_bytes())
    metrics_module.GPU_MEMORY_RESERVED_BYTES.set(read_memory_reserved_bytes())
    utilization = read_utilization_percent()
    if utilization is not None:
        metrics_module.GPU_UTILIZATION_PERCENT.set(utilization)
