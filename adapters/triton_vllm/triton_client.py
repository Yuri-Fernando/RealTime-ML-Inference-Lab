"""Triton Inference Server client using the KServe v2 HTTP/REST protocol.

Protocol reference (real, documented wire format — no special SDK needed):
  POST {base_url}/v2/models/{model_name}/infer
  Body: {
    "inputs": [{"name": ..., "shape": [...], "datatype": "FP32", "data": [...]}],
    "outputs": [{"name": ...}]
  }
  Response: {
    "model_name": ..., "outputs": [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]
  }

This client speaks that protocol over plain HTTP via httpx, so it works
against any real Triton server without requiring the `tritonclient` package.
No real Triton server was available in the session that built this repo —
see adapters/triton_vllm/__init__.py and README.md for the honesty note.
Correctness is covered by tests/test_adapters.py against a local mock.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


class TritonInferenceError(RuntimeError):
    """Raised when the Triton server returns an error response."""


@dataclass
class TritonInferResult:
    model_name: str
    output_name: str
    shape: list[int]
    data: list[float]


class TritonClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        input_name: str = "INPUT",
        output_name: str = "OUTPUT",
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def infer(self, values: list[float], shape: list[int]) -> TritonInferResult:
        payload = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": shape,
                    "datatype": "FP32",
                    "data": values,
                }
            ],
            "outputs": [{"name": self.output_name}],
        }
        url = f"/v2/models/{self.model_name}/infer"
        response = await self._client.post(url, json=payload)

        if response.status_code != 200:
            raise TritonInferenceError(
                f"Triton returned HTTP {response.status_code}: {response.text}"
            )

        body = response.json()
        if "error" in body:
            raise TritonInferenceError(f"Triton inference error: {body['error']}")

        try:
            output = next(o for o in body["outputs"] if o["name"] == self.output_name)
        except (KeyError, StopIteration) as exc:
            raise TritonInferenceError(f"malformed Triton response: {body}") from exc

        return TritonInferResult(
            model_name=body.get("model_name", self.model_name),
            output_name=output["name"],
            shape=output["shape"],
            data=output["data"],
        )
