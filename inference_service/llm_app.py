"""FastAPI de geração de texto (V0.3), com backend plugável.

    POST /generate        {"prompt": "...", "max_new_tokens": 32}
    POST /cancel/{id}     cancelamento explícito
    GET  /quality         perplexidade + saídas greedy (só backends que suportam)
    GET  /health          backend, memória, fila, contadores
    GET  /metrics         Prometheus (rtml_llm_* + GPU)

Subir: ``RTML_BACKEND=hf RTML_HF_QUANT=nf4 uvicorn inference_service.llm_app:app``.

O app histórico (``inference_service.app``, TinyCNN) é independente e continua
como era; os dois podem rodar em portas diferentes e compartilham o mesmo
módulo de admissão.

Fora do escopo desta versão (documentado no README): cache de resultados do
LLM (inflaria benchmarks com prompts repetidos), WebSocket de geração e
streaming de tokens.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from inference_service import gpu_telemetry, llm_metrics, metrics
from inference_service.backends import BackendConfigError, create_backend
from inference_service.backends.base import GenerationResult
from inference_service.config import settings
from inference_service.llm_batching import (
    BackpressureError,
    LLMBatcher,
    RequestCancelledError,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(32, ge=1, le=4096)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[float]
    latency_ms: float
    finish_reason: Optional[str]
    backend: str


class CancelResponse(BaseModel):
    request_id: str
    cancelled: bool


def create_app(backend=None, batcher: Optional[LLMBatcher] = None) -> FastAPI:
    """Cria o app. Sem argumentos, o backend vem de ``RTML_BACKEND`` na subida
    (lifespan), então importar este módulo nunca carrega um modelo."""
    state: dict = {"backend": backend, "batcher": batcher}

    def _on_batch(size: int, results: list[GenerationResult], seconds: float) -> None:
        llm_metrics.LLM_BATCH_SIZE.observe(size)
        llm_metrics.LLM_BATCH_SECONDS.observe(seconds)
        tokens = sum(r.completion_tokens for r in results)
        if seconds > 0:
            llm_metrics.LLM_TOKENS_PER_SECOND.set(tokens / seconds)

    def _on_cancelled(reason: str) -> None:
        metrics.CANCELLATIONS_TOTAL.labels(reason=reason).inc()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        be = state["backend"]
        if be is None:
            be = create_backend(settings.BACKEND)
            if getattr(be, "kind", None) != "text":
                raise BackendConfigError(
                    f"RTML_BACKEND={settings.BACKEND!r} não é um backend de texto; "
                    "use hf ou vllm (o TinyCNN é servido por inference_service.app)"
                )
            state["backend"] = be
        b = state["batcher"]
        if b is None:
            b = LLMBatcher(be)
            state["batcher"] = b
        b.on_batch_processed = _on_batch
        b.on_cancelled = _on_cancelled
        info = be.describe()
        llm_metrics.LLM_BACKEND_INFO.labels(
            backend=info.get("backend", "?"),
            model=str(info.get("model", "")),
            quant=str(info.get("quant", "")),
        ).set(1)
        llm_metrics.LLM_MODEL_MEMORY_BYTES.set(info.get("memory_footprint_bytes", 0))
        await b.start()
        try:
            yield
        finally:
            await b.stop()
            aclose = getattr(be, "aclose", None)
            if aclose is not None:
                await aclose()

    app = FastAPI(title="RealTime-ML-Inference-Lab (LLM)", lifespan=lifespan)
    app.state.lab = state

    async def run_generation(prompt: str, max_new_tokens: int, request_id: str) -> GenerateResponse:
        batcher_ = state["batcher"]
        be = state["backend"]
        start = time.perf_counter()
        metrics.QUEUE_DEPTH.set(batcher_.queue_depth)
        try:
            res = await batcher_.submit(prompt, max_new_tokens, request_id=request_id)
        except BackpressureError as exc:
            llm_metrics.LLM_REQUESTS_TOTAL.labels(outcome="backpressure").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RequestCancelledError as exc:
            llm_metrics.LLM_REQUESTS_TOTAL.labels(outcome="cancelled").inc()
            raise HTTPException(status_code=499, detail=f"request {exc} was cancelled") from exc
        except (TimeoutError, asyncio.TimeoutError) as exc:
            llm_metrics.LLM_REQUESTS_TOTAL.labels(outcome="timeout").inc()
            raise HTTPException(status_code=504, detail="generation timed out") from exc
        except Exception as exc:
            llm_metrics.LLM_REQUESTS_TOTAL.labels(outcome="error").inc()
            raise HTTPException(status_code=502, detail=f"backend error: {exc}") from exc

        latency = time.perf_counter() - start
        llm_metrics.LLM_REQUEST_LATENCY.observe(latency)
        llm_metrics.LLM_REQUESTS_TOTAL.labels(outcome="ok").inc()
        llm_metrics.LLM_TOKENS_GENERATED_TOTAL.inc(res.completion_tokens)
        llm_metrics.LLM_PROMPT_TOKENS_TOTAL.inc(res.prompt_tokens)
        if res.ttft_ms is not None:
            llm_metrics.LLM_TTFT.observe(res.ttft_ms / 1000.0)
        metrics.QUEUE_DEPTH.set(batcher_.queue_depth)
        return GenerateResponse(
            request_id=request_id,
            text=res.text,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            ttft_ms=res.ttft_ms,
            latency_ms=latency * 1000.0,
            finish_reason=res.finish_reason,
            backend=getattr(be, "name", "?"),
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        # O cliente pode fixar o id (X-Request-ID) para poder cancelar por
        # POST /cancel/{id} enquanto a geração ainda está em andamento.
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        task = asyncio.ensure_future(
            run_generation(payload.prompt, payload.max_new_tokens, request_id)
        )
        # Desconexão do cliente = cancelamento real (mesmo princípio do WS
        # do app histórico): enquanto a geração roda, vigia a conexão.
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if done:
                return task.result()
            if await request.is_disconnected():
                state["batcher"].cancel(request_id, reason="client_disconnect")
                task.cancel()
                try:
                    await task
                except (HTTPException, RequestCancelledError, asyncio.CancelledError):
                    pass
                raise HTTPException(status_code=499, detail="client disconnected")

    @app.post("/cancel/{request_id}", response_model=CancelResponse)
    async def cancel(request_id: str) -> CancelResponse:
        return CancelResponse(
            request_id=request_id,
            cancelled=state["batcher"].cancel(request_id, reason="explicit"),
        )

    @app.get("/quality")
    async def quality() -> dict:
        be = state["backend"]
        fn = getattr(be, "evaluate_quality", None)
        if fn is None:
            raise HTTPException(status_code=501, detail="backend sem avaliação de qualidade local")
        return await asyncio.to_thread(fn)

    @app.get("/health")
    async def health() -> dict:
        be = state["backend"]
        b = state["batcher"]
        info = be.describe() if be is not None else {}
        return {
            "status": "ok" if be is not None else "starting",
            **info,
            "queue_depth": b.queue_depth if b else 0,
            "max_queue_depth": b.max_queue_depth if b else 0,
            "batches_executed": getattr(be, "forward_pass_count", 0),
            "items_processed": getattr(be, "total_items_processed", 0),
            "tokens_generated_total": getattr(be, "tokens_generated_total", 0),
        }

    @app.get("/metrics", response_model=None)
    async def metrics_endpoint():
        gpu_telemetry.update_metrics(metrics)
        be = state["backend"]
        if be is not None:
            info = be.describe()
            llm_metrics.LLM_MODEL_MEMORY_BYTES.set(info.get("memory_footprint_bytes", 0))
            llm_metrics.GPU_PEAK_ALLOCATED_BYTES.set(info.get("gpu_peak_allocated_bytes", 0))
        return Response(content=metrics.render_latest(), media_type="text/plain; version=0.0.4")

    return app


# App padrão para `uvicorn inference_service.llm_app:app` (backend por env).
app = create_app()
