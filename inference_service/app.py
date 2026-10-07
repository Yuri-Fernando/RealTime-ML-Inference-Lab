"""FastAPI application: HTTP + WebSocket real-time inference endpoints.

Exposes:
  POST /infer        -> single-request HTTP inference (goes through the
                         dynamic batcher + cache like everything else)
  WS   /ws/infer      -> WebSocket inference, one JSON message in, one out
  GET  /metrics        -> Prometheus exposition format
  GET  /health         -> liveness/readiness probe (reports device + queue depth)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from inference_service import metrics
from inference_service.batching import BackpressureError, DynamicBatcher
from inference_service.cache import hash_input, result_cache
from inference_service.config import settings
from inference_service.model import model_backend

def _on_batch_processed(size: int) -> None:
    metrics.BATCH_SIZE.observe(size)
    metrics.FORWARD_PASSES_TOTAL.inc()


batcher = DynamicBatcher(model_backend=model_backend)
batcher.on_batch_processed = _on_batch_processed


@asynccontextmanager
async def lifespan(app: FastAPI):
    await batcher.start()
    yield
    await batcher.stop()


app = FastAPI(title="RealTime-ML-Inference-Lab", lifespan=lifespan)


class InferRequest(BaseModel):
    input: list[float] = Field(..., min_length=1)

    def validate_dim(self) -> None:
        if len(self.input) != settings.INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"expected input of length {settings.INPUT_DIM}, got {len(self.input)}",
            )


class InferResponse(BaseModel):
    output: list[float]
    cached: bool
    latency_ms: float


async def run_inference(values: list[float]) -> InferResponse:
    start = time.perf_counter()
    key = hash_input(values)

    cached_result = await result_cache.get(key)
    if cached_result is not None:
        metrics.CACHE_EVENTS_TOTAL.labels(result="hit").inc()
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.REQUEST_LATENCY.observe(latency_ms / 1000)
        metrics.REQUESTS_TOTAL.labels(outcome="ok").inc()
        return InferResponse(output=cached_result, cached=True, latency_ms=latency_ms)

    metrics.CACHE_EVENTS_TOTAL.labels(result="miss").inc()
    metrics.QUEUE_DEPTH.set(batcher.queue_depth)

    try:
        result = await batcher.submit(values)
    except BackpressureError as exc:
        metrics.REQUESTS_TOTAL.labels(outcome="backpressure").inc()
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        metrics.REQUESTS_TOTAL.labels(outcome="timeout").inc()
        raise HTTPException(status_code=504, detail="inference timed out") from exc

    metrics.FORWARD_PASSES_TOTAL.inc(0)  # presence; real increments happen via on_batch_processed
    await result_cache.put(key, result)

    latency_ms = (time.perf_counter() - start) * 1000
    metrics.REQUEST_LATENCY.observe(latency_ms / 1000)
    metrics.REQUESTS_TOTAL.labels(outcome="ok").inc()
    metrics.QUEUE_DEPTH.set(batcher.queue_depth)
    return InferResponse(output=result, cached=False, latency_ms=latency_ms)


@app.post("/infer", response_model=InferResponse)
async def infer(payload: InferRequest) -> InferResponse:
    payload.validate_dim()
    return await run_inference(payload.input)


@app.websocket("/ws/infer")
async def infer_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            values = data.get("input")
            if not isinstance(values, list) or len(values) != settings.INPUT_DIM:
                await websocket.send_json(
                    {"error": f"expected input of length {settings.INPUT_DIM}"}
                )
                continue
            try:
                response = await run_inference(values)
                await websocket.send_json(response.model_dump())
            except HTTPException as exc:
                await websocket.send_json({"error": exc.detail, "status_code": exc.status_code})
    except WebSocketDisconnect:
        return


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": str(model_backend.device),
        "queue_depth": batcher.queue_depth,
        "max_queue_depth": batcher.max_queue_depth,
        "cache_size": len(result_cache),
        "forward_pass_count": model_backend.forward_pass_count,
    }


@app.get("/metrics", response_model=None)
async def metrics_endpoint():
    from fastapi import Response

    return Response(content=metrics.render_latest(), media_type="text/plain; version=0.0.4")
