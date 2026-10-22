"""VLLMBackend contra um servidor MOCK do protocolo OpenAI-compatible.

Não há vLLM real aqui (ele não roda nativo no Windows; ver README e
scripts/run_vllm_wsl.sh). Mocka-se a dependência externa ausente — o
comportamento sob teste (cliente, backend, batcher, app) é real.
"""
import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from adapters.triton_vllm.vllm_client import VLLMInferenceError
from inference_service.backends.base import GenerationRequest
from inference_service.backends.vllm_backend import VLLMBackend
from inference_service.llm_app import create_app
from inference_service.llm_batching import LLMBatcher, RequestCancelledError

mock_vllm = FastAPI()
mock_vllm.state.calls = 0


@mock_vllm.get("/v1/models")
async def _models():
    return {"data": [{"id": "mock-model"}]}


@mock_vllm.post("/v1/completions")
async def _completions(request: Request):
    body = await request.json()
    mock_vllm.state.calls += 1
    prompt = body["prompt"]
    if prompt == "BOOM":
        return JSONResponse(status_code=500, content={"error": "falha simulada"})
    if prompt == "SLOW":
        await asyncio.sleep(5)
    return {
        "id": "cmpl-1",
        "choices": [{"text": f"re:{prompt}", "finish_reason": "length"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": body["max_tokens"]},
    }


def _backend() -> VLLMBackend:
    return VLLMBackend(
        base_url="http://vllm.mock",
        model="mock-model",
        transport=httpx.ASGITransport(app=mock_vllm),
    )


async def test_agenerate_parses_text_tokens_and_counts_calls():
    backend = _backend()
    try:
        res = await backend.agenerate(GenerationRequest("hello there", 6))
    finally:
        await backend.aclose()
    assert res.text == "re:hello there"
    assert res.prompt_tokens == 2 and res.completion_tokens == 6
    assert res.finish_reason == "length" and res.ttft_ms is None
    assert backend.forward_pass_count == 1 and backend.tokens_generated_total == 6


async def test_server_error_becomes_a_typed_error():
    backend = _backend()
    try:
        with pytest.raises(VLLMInferenceError, match="500"):
            await backend.agenerate(GenerationRequest("BOOM", 3))
    finally:
        await backend.aclose()


async def test_is_ready_reflects_models_endpoint():
    backend = _backend()
    try:
        assert await backend.is_ready() is True
    finally:
        await backend.aclose()
    dead = VLLMBackend(base_url="http://127.0.0.1:9", model="m")
    try:
        assert await dead.is_ready() is False
    finally:
        await dead.aclose()


async def test_client_side_batching_is_explicitly_unsupported():
    backend = _backend()
    try:
        assert backend.supports_batching is False
        with pytest.raises(NotImplementedError):
            backend.generate_batch([GenerationRequest("x")])
    finally:
        await backend.aclose()


async def test_batcher_runs_vllm_requests_concurrently_through_admission():
    backend = _backend()
    batcher = LLMBatcher(backend, max_queue_depth=16)
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(f"q{i}", 4) for i in range(6)))
    finally:
        await batcher.stop()
        await backend.aclose()
    assert [r.text for r in results] == [f"re:q{i}" for i in range(6)]
    assert backend.forward_pass_count == 6  # sem agrupamento no cliente


async def test_cancel_aborts_the_in_flight_http_call_quickly():
    backend = _backend()
    batcher = LLMBatcher(backend)
    await batcher.start()
    try:
        task = asyncio.create_task(batcher.submit("SLOW", 2, request_id="s1"))
        await asyncio.sleep(0.2)
        started = asyncio.get_running_loop().time()
        assert batcher.cancel("s1") is True
        with pytest.raises(RequestCancelledError):
            await task
        assert asyncio.get_running_loop().time() - started < 1.0  # não esperou os 5 s
    finally:
        await batcher.stop()
        await backend.aclose()


def test_app_serves_generate_through_the_vllm_backend():
    backend = _backend()
    client = TestClient(create_app(backend=backend, batcher=LLMBatcher(backend)))
    with client as c:
        r = c.post("/generate", json={"prompt": "ping pong", "max_new_tokens": 3})
        h = c.get("/health").json()
    assert r.status_code == 200 and r.json()["text"] == "re:ping pong"
    assert r.json()["backend"] == "vllm" and r.json()["ttft_ms"] is None
    assert h["backend"] == "vllm" and h["batching"].startswith("server-side")
