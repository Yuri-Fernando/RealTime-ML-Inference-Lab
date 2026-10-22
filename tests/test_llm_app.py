"""llm_app: contrato HTTP, admissão (429), cancelamento (499) e métricas."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from inference_service.llm_app import create_app
from inference_service.llm_batching import LLMBatcher
from tests.fakes import FakeTextBackend


def _client(backend=None, **batcher_kw):
    backend = backend or FakeTextBackend(delay_s=0.05)
    batcher_kw.setdefault("max_wait_ms", 20)
    return backend, TestClient(create_app(backend=backend, batcher=LLMBatcher(backend, **batcher_kw)))


def test_generate_returns_text_tokens_and_backend_name():
    backend, app_client = _client()
    with app_client as client:
        r = client.post("/generate", json={"prompt": "ola mundo", "max_new_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "echo:ola mundo"
    assert body["completion_tokens"] == 5 and body["prompt_tokens"] == 2
    assert body["backend"] == "fake" and body["latency_ms"] > 0
    assert body["finish_reason"] == "length"
    assert backend.total_items_processed == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "x", "max_new_tokens": 0},
        {"prompt": "x", "max_new_tokens": 99999},
        {"max_new_tokens": 3},
    ],
)
def test_invalid_payload_is_422_and_never_reaches_backend(payload):
    backend, app_client = _client()
    with app_client as client:
        assert client.post("/generate", json=payload).status_code == 422
    assert backend.total_items_processed == 0


def test_health_reports_backend_queue_and_counters():
    backend, app_client = _client()
    with app_client as client:
        client.post("/generate", json={"prompt": "a b c", "max_new_tokens": 2})
        h = client.get("/health").json()
    assert h["status"] == "ok" and h["backend"] == "fake"
    assert h["queue_depth"] == 0 and h["items_processed"] == 1 and h["batches_executed"] == 1


def test_metrics_expose_llm_series_with_real_values():
    backend, app_client = _client()
    with app_client as client:
        client.post("/generate", json={"prompt": "one two", "max_new_tokens": 7})
        text = client.get("/metrics").text
    assert 'rtml_llm_requests_total{outcome="ok"}' in text
    assert "rtml_llm_tokens_generated_total" in text
    assert "rtml_llm_ttft_seconds_count" in text
    assert "rtml_llm_batch_size_count" in text
    assert 'rtml_llm_backend_info{backend="fake"' in text
    assert "rtml_gpu_memory_peak_allocated_bytes" in text


def test_backpressure_returns_429_and_rejected_never_reaches_backend():
    backend = FakeTextBackend(delay_s=0.6)
    _, app_client = _client(backend, max_queue_depth=1, max_batch_size=1)
    with app_client as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(client.post, "/generate", json={"prompt": "lenta", "max_new_tokens": 2})
            time.sleep(0.2)
            rejected = client.post("/generate", json={"prompt": "extra", "max_new_tokens": 2})
            assert slow.result().status_code == 200
    assert rejected.status_code == 429
    assert backend.total_items_processed == 1


def test_explicit_cancel_returns_499_to_the_waiting_caller():
    backend = FakeTextBackend(delay_s=1.0)
    _, app_client = _client(backend)
    with app_client as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                client.post,
                "/generate",
                json={"prompt": "vou cancelar", "max_new_tokens": 4},
                headers={"x-request-id": "cancelme"},
            )
            time.sleep(0.4)
            c = client.post("/cancel/cancelme")
            resp = fut.result()
        assert c.json() == {"request_id": "cancelme", "cancelled": True}
        assert client.post("/cancel/cancelme").json()["cancelled"] is False
        metrics = client.get("/metrics").text
    assert resp.status_code == 499
    assert 'rtml_llm_requests_total{outcome="cancelled"}' in metrics


def test_quality_is_501_for_backends_without_local_eval():
    _, app_client = _client()
    with app_client as client:
        assert client.get("/quality").status_code == 501


def test_lifespan_rejects_tensor_backend_selected_by_env(monkeypatch):
    from inference_service.backends.base import BackendConfigError
    from inference_service.config import settings

    monkeypatch.setattr(settings, "BACKEND", "tinycnn")
    app = create_app()
    with pytest.raises(BackendConfigError, match="texto"):
        with TestClient(app):
            pass
