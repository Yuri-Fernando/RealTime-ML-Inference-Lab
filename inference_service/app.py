"""FastAPI application: HTTP + WebSocket real-time inference endpoints.

Exposes:
  POST /infer              -> single-request HTTP inference (goes through the
                               dynamic batcher + cache like everything else)
  POST /cancel/{request_id} -> explicit cancellation of an in-flight request
  WS   /ws/infer            -> WebSocket inference, one JSON message in, one
                               out; a client disconnect mid-request also
                               triggers real cancellation of that request
  GET  /metrics             -> Prometheus exposition format (incl. real GPU
                               telemetry refreshed at scrape time)
  GET  /health              -> liveness/readiness probe (device + queue depth)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from inference_service import gpu_telemetry, metrics
from inference_service.batching import (
    BackpressureError,
    DynamicBatcher,
    RequestCancelledError,
)
from inference_service.cache import hash_input, result_cache
from inference_service.config import settings
from inference_service.model import model_backend

def _on_batch_processed(size: int) -> None:
    metrics.BATCH_SIZE.observe(size)
    metrics.FORWARD_PASSES_TOTAL.inc()


def _on_cancelled(reason: str) -> None:
    metrics.CANCELLATIONS_TOTAL.labels(reason=reason).inc()


batcher = DynamicBatcher(model_backend=model_backend)
batcher.on_batch_processed = _on_batch_processed
batcher.on_cancelled = _on_cancelled


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
    request_id: str
    output: list[float]
    cached: bool
    latency_ms: float


class CancelResponse(BaseModel):
    request_id: str
    cancelled: bool


async def run_inference(values: list[float], request_id: Optional[str] = None) -> InferResponse:
    request_id = request_id or uuid.uuid4().hex
    start = time.perf_counter()
    key = hash_input(values)

    cached_result = await result_cache.get(key)
    if cached_result is not None:
        metrics.CACHE_EVENTS_TOTAL.labels(result="hit").inc()
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.REQUEST_LATENCY.observe(latency_ms / 1000)
        metrics.REQUESTS_TOTAL.labels(outcome="ok").inc()
        return InferResponse(
            request_id=request_id, output=cached_result, cached=True, latency_ms=latency_ms
        )

    metrics.CACHE_EVENTS_TOTAL.labels(result="miss").inc()
    metrics.QUEUE_DEPTH.set(batcher.queue_depth)

    try:
        result = await batcher.submit(values, request_id=request_id)
    except BackpressureError as exc:
        metrics.REQUESTS_TOTAL.labels(outcome="backpressure").inc()
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RequestCancelledError as exc:
        metrics.REQUESTS_TOTAL.labels(outcome="cancelled").inc()
        raise HTTPException(status_code=499, detail=f"request {exc} was cancelled") from exc
    except TimeoutError as exc:
        metrics.REQUESTS_TOTAL.labels(outcome="timeout").inc()
        raise HTTPException(status_code=504, detail="inference timed out") from exc

    metrics.FORWARD_PASSES_TOTAL.inc(0)  # presence; real increments happen via on_batch_processed
    await result_cache.put(key, result)

    latency_ms = (time.perf_counter() - start) * 1000
    metrics.REQUEST_LATENCY.observe(latency_ms / 1000)
    metrics.REQUESTS_TOTAL.labels(outcome="ok").inc()
    metrics.QUEUE_DEPTH.set(batcher.queue_depth)
    return InferResponse(
        request_id=request_id, output=result, cached=False, latency_ms=latency_ms
    )


@app.post("/infer", response_model=InferResponse)
async def infer(payload: InferRequest) -> InferResponse:
    payload.validate_dim()
    return await run_inference(payload.input)


@app.post("/cancel/{request_id}", response_model=CancelResponse)
async def cancel(request_id: str) -> CancelResponse:
    """Explicit cancellation: cancels a request that is still queued/in-flight.

    Real behavior, not cosmetic: if the request has not yet been picked up
    by the batching worker, it is excluded from its batch and never reaches
    the model (see DynamicBatcher._process_batch). The caller awaiting
    /infer or /ws/infer for that request_id receives a 499 response instead
    of a result.
    """
    cancelled = batcher.cancel(request_id, reason="explicit")
    return CancelResponse(request_id=request_id, cancelled=cancelled)


@app.websocket("/ws/infer")
async def infer_ws(websocket: WebSocket) -> None:
    """WebSocket inference with real cancellation on client disconnect.

    Protocol:
      - send {"input": [...], "request_id": "<optional>"} to run inference.
      - send {"cancel": "<request_id>"} (with nothing in flight on this
        connection) to cancel a request submitted earlier.
      - disconnecting the socket while a request is in flight cancels that
        in-flight request for real: the request is excluded from its batch
        and never reaches the model (see DynamicBatcher._process_batch).
        This is detected by racing the inference against a dedicated
        watcher coroutine that awaits `websocket.receive()` in a loop and
        only resolves on a "websocket.disconnect" event.
    """
    await websocket.accept()

    async def _wait_for_disconnect() -> None:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

    try:
        while True:
            data = await websocket.receive_json()

            cancel_id = data.get("cancel")
            if cancel_id is not None:
                cancelled = batcher.cancel(cancel_id, reason="explicit")
                await websocket.send_json({"cancelled": cancelled, "request_id": cancel_id})
                continue

            values = data.get("input")
            request_id = data.get("request_id") or uuid.uuid4().hex
            if not isinstance(values, list) or len(values) != settings.INPUT_DIM:
                await websocket.send_json(
                    {"error": f"expected input of length {settings.INPUT_DIM}"}
                )
                continue

            inference_task = asyncio.ensure_future(run_inference(values, request_id=request_id))
            disconnect_task = asyncio.ensure_future(_wait_for_disconnect())
            done, _pending = await asyncio.wait(
                {inference_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if inference_task in done:
                disconnect_task.cancel()
                try:
                    response = await inference_task
                    await websocket.send_json(response.model_dump())
                except HTTPException as exc:
                    await websocket.send_json(
                        {"error": exc.detail, "status_code": exc.status_code}
                    )
                continue

            # Client disconnected while inference was still in flight:
            # cancel the real in-flight request before its result (if any)
            # is ever produced/consumed.
            batcher.cancel(request_id, reason="client_disconnect")
            inference_task.cancel()
            try:
                await inference_task
            except (RequestCancelledError, HTTPException, asyncio.CancelledError):
                pass
            return
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

    # Real GPU telemetry read fresh at scrape time (not a background
    # sampler running on its own schedule) — see gpu_telemetry.py.
    gpu_telemetry.update_metrics(metrics)

    return Response(content=metrics.render_latest(), media_type="text/plain; version=0.0.4")
