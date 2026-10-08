"""End-to-end integration tests for the FastAPI app: HTTP, WebSocket,
caching, backpressure and the /metrics and /health endpoints.

Uses Starlette's TestClient (sync, backed by httpx) which drives the real
app lifespan (so the DynamicBatcher background task really starts/stops)
and the real TinyCNN model — nothing here is mocked.
"""
import asyncio
import time

import pytest
from starlette.testclient import TestClient

from inference_service import metrics
from inference_service.app import app, batcher
from inference_service.cache import result_cache
from inference_service.config import settings
from inference_service.model import model_backend


def _clear_cache_sync() -> None:
    asyncio.run(result_cache.clear())


@pytest.fixture(autouse=True)
def reset_state():
    model_backend.reset_counters()
    _clear_cache_sync()
    yield
    model_backend.reset_counters()
    _clear_cache_sync()


def _valid_input():
    return [0.1] * settings.INPUT_DIM


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "device" in body


def test_infer_http_happy_path():
    with TestClient(app) as client:
        resp = client.post("/infer", json={"input": _valid_input()})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["output"]) == settings.NUM_CLASSES
        assert body["cached"] is False
        assert body["latency_ms"] >= 0


def test_infer_http_rejects_wrong_dimension():
    with TestClient(app) as client:
        resp = client.post("/infer", json={"input": [0.1, 0.2]})
        assert resp.status_code == 422


def test_infer_http_cache_hit_on_second_identical_request():
    with TestClient(app) as client:
        payload = {"input": _valid_input()}
        first = client.post("/infer", json=payload)
        second = client.post("/infer", json=payload)

        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert first.json()["output"] == second.json()["output"]


def test_metrics_endpoint_exposes_prometheus_format():
    with TestClient(app) as client:
        client.post("/infer", json={"input": _valid_input()})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "rtml_requests_total" in text
        assert "rtml_request_latency_seconds" in text


def test_websocket_infer_roundtrip():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/infer") as ws:
            ws.send_json({"input": _valid_input()})
            response = ws.receive_json()
            assert len(response["output"]) == settings.NUM_CLASSES
            assert response["cached"] is False


def test_websocket_infer_rejects_wrong_dimension():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/infer") as ws:
            ws.send_json({"input": [0.1, 0.2]})
            response = ws.receive_json()
            assert "error" in response


def test_backpressure_returns_429_when_queue_saturated(monkeypatch):
    """Shrink max_queue_depth to 0 so even the first request is rejected,
    proving the HTTP layer surfaces BackpressureError as a real 429."""
    monkeypatch.setattr(batcher, "max_queue_depth", 0)
    with TestClient(app) as client:
        resp = client.post("/infer", json={"input": _valid_input()})
        assert resp.status_code == 429
    monkeypatch.setattr(batcher, "max_queue_depth", settings.MAX_QUEUE_DEPTH)


def test_infer_response_includes_request_id():
    with TestClient(app) as client:
        resp = client.post("/infer", json={"input": _valid_input()})
        assert resp.status_code == 200
        assert isinstance(resp.json()["request_id"], str)
        assert len(resp.json()["request_id"]) > 0


def test_cancel_unknown_request_id_returns_cancelled_false():
    with TestClient(app) as client:
        resp = client.post("/cancel/does-not-exist")
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "does-not-exist"
        assert body["cancelled"] is False


def test_websocket_disconnect_cancels_inflight_request(monkeypatch):
    """Real end-to-end proof that a WS client disconnecting mid-request
    triggers real cancellation: widen the batching window so the request
    is still queued when we close the socket, then assert the
    `rtml_cancellations_total{reason="client_disconnect"}` counter moved."""
    monkeypatch.setattr(batcher, "max_wait_ms", 2000)
    try:
        with TestClient(app) as client:
            before = metrics.CANCELLATIONS_TOTAL.labels(reason="client_disconnect")._value.get()
            with client.websocket_connect("/ws/infer") as ws:
                ws.send_json({"input": _valid_input(), "request_id": "ws-cancel-1"})
                # Deliberately close without waiting for a response — the
                # server-side handler should observe the disconnect while
                # the request is still sitting in the (slow) batch window.
            time.sleep(0.3)
            after = metrics.CANCELLATIONS_TOTAL.labels(reason="client_disconnect")._value.get()
            assert after > before
    finally:
        monkeypatch.setattr(batcher, "max_wait_ms", settings.MAX_BATCH_WAIT_MS)


def test_metrics_endpoint_exposes_gpu_telemetry_and_cancellation_metrics():
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "rtml_gpu_available" in text
        assert "rtml_gpu_memory_allocated_bytes" in text
        assert "rtml_cancellations_total" in text
