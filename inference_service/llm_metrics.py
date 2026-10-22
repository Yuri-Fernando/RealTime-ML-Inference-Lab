"""Métricas Prometheus do caminho de geração (V0.3).

Usam o MESMO ``registry`` de ``inference_service.metrics`` (um único
``/metrics`` por processo), mas ficam em módulo próprio para não alterar a
saída do app histórico (TinyCNN). Todas vêm de medições reais feitas em
``llm_app.py`` / ``llm_batching.py``.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from inference_service.metrics import registry

LLM_REQUESTS_TOTAL = Counter(
    "rtml_llm_requests_total",
    "Requisições de geração recebidas",
    ["outcome"],  # ok | backpressure | cancelled | timeout | error
    registry=registry,
)

LLM_REQUEST_LATENCY = Histogram(
    "rtml_llm_request_latency_seconds",
    "Latência ponta a ponta de /generate (fila + geração), segundos",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 120),
    registry=registry,
)

LLM_TTFT = Histogram(
    "rtml_llm_ttft_seconds",
    "Tempo até o primeiro token visto pelo cliente (espera em fila + prefill do lote)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    registry=registry,
)

LLM_TOKENS_GENERATED_TOTAL = Counter(
    "rtml_llm_tokens_generated_total",
    "Tokens gerados (completion)",
    registry=registry,
)

LLM_PROMPT_TOKENS_TOTAL = Counter(
    "rtml_llm_prompt_tokens_total",
    "Tokens de prompt processados",
    registry=registry,
)

LLM_BATCH_SIZE = Histogram(
    "rtml_llm_batch_size",
    "Requisições por lote de geração",
    buckets=(1, 2, 4, 8, 16, 32),
    registry=registry,
)

LLM_BATCH_SECONDS = Histogram(
    "rtml_llm_batch_seconds",
    "Duração de um lote de geração, segundos",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
    registry=registry,
)

LLM_TOKENS_PER_SECOND = Gauge(
    "rtml_llm_tokens_per_second",
    "Tokens/s do último lote (tokens gerados no lote / duração do lote)",
    registry=registry,
)

LLM_MODEL_MEMORY_BYTES = Gauge(
    "rtml_llm_model_memory_bytes",
    "Memória ocupada pelos pesos (get_memory_footprint) — 0 para backend remoto",
    registry=registry,
)

GPU_PEAK_ALLOCATED_BYTES = Gauge(
    "rtml_gpu_memory_peak_allocated_bytes",
    "torch.cuda.max_memory_allocated() do processo (pico de VRAM alocada)",
    registry=registry,
)

LLM_BACKEND_INFO = Gauge(
    "rtml_llm_backend_info",
    "Backend ativo (valor sempre 1; informação nos rótulos)",
    ["backend", "model", "quant"],
    registry=registry,
)
