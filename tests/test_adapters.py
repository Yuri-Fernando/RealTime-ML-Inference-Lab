"""Tests for the Triton/vLLM adapters against a LOCAL IN-PROCESS MOCK server.

No real Triton Inference Server or vLLM server is required or used here —
that infra was not available in the session that built this repo (see
README.md "Status"). What IS real: the client code builds requests and
parses responses using the actual documented wire protocols, exercised
end-to-end over HTTP via httpx's ASGITransport against a mock server that
implements those same protocols. This is mocking an absent EXTERNAL
dependency, not the behavior under test (the client's request/response
handling, including error paths).
"""
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from adapters.triton_vllm.triton_client import TritonClient, TritonInferenceError
from adapters.triton_vllm.vllm_client import VLLMClient, VLLMInferenceError

# ---------------------------------------------------------------------------
# Mock Triton server (KServe v2 HTTP protocol)
# ---------------------------------------------------------------------------

mock_triton_app = FastAPI()


@mock_triton_app.post("/v2/models/{model_name}/infer")
async def mock_triton_infer(model_name: str, request: Request):
    body = await request.json()
    input_data = body["inputs"][0]["data"]
    if input_data == ["TRIGGER_ERROR"]:
        return JSONResponse(status_code=400, content={"error": "simulated bad input"})
    output = [x * 2 for x in input_data]
    return {
        "model_name": model_name,
        "outputs": [
            {
                "name": "OUTPUT",
                "shape": [len(output)],
                "datatype": "FP32",
                "data": output,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Mock vLLM server (OpenAI-compatible completions API)
# ---------------------------------------------------------------------------

mock_vllm_app = FastAPI()


@mock_vllm_app.post("/v1/completions")
async def mock_vllm_completions(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    if prompt == "TRIGGER_ERROR":
        return JSONResponse(status_code=500, content={"error": "simulated server error"})
    return {
        "id": "cmpl-mock",
        "choices": [
            {"text": f"echo: {prompt}", "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 2},
    }


@pytest.fixture
def triton_client():
    transport = httpx.ASGITransport(app=mock_triton_app)
    client = TritonClient(
        base_url="http://mock-triton", model_name="tinycnn", transport=transport
    )
    yield client


@pytest.fixture
def vllm_client():
    transport = httpx.ASGITransport(app=mock_vllm_app)
    client = VLLMClient(base_url="http://mock-vllm", model="mock-model", transport=transport)
    yield client


async def test_triton_client_builds_request_and_parses_response(triton_client):
    result = await triton_client.infer(values=[1.0, 2.0, 3.0], shape=[1, 3])
    assert result.model_name == "tinycnn"
    assert result.output_name == "OUTPUT"
    assert result.data == [2.0, 4.0, 6.0]
    await triton_client.aclose()


async def test_triton_client_raises_on_error_response(triton_client):
    with pytest.raises(TritonInferenceError):
        await triton_client.infer(values=["TRIGGER_ERROR"], shape=[1])
    await triton_client.aclose()


async def test_vllm_client_builds_request_and_parses_response(vllm_client):
    result = await vllm_client.complete(prompt="hello world")
    assert result.text == "echo: hello world"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 2
    await vllm_client.aclose()


async def test_vllm_client_raises_on_error_response(vllm_client):
    with pytest.raises(VLLMInferenceError):
        await vllm_client.complete(prompt="TRIGGER_ERROR")
    await vllm_client.aclose()
