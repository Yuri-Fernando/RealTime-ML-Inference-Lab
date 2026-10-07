"""vLLM client using the OpenAI-compatible HTTP API vLLM serves.

Protocol reference (real, documented wire format):
  POST {base_url}/v1/completions
  Body: {"model": ..., "prompt": ..., "max_tokens": ..., "temperature": ...}
  Response: {"id": ..., "choices": [{"text": ..., "finish_reason": ...}], "usage": {...}}

No real vLLM server was available in the session that built this repo —
see adapters/triton_vllm/__init__.py and README.md for the honesty note.
Correctness is covered by tests/test_adapters.py against a local mock.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


class VLLMInferenceError(RuntimeError):
    """Raised when the vLLM server returns an error response."""


@dataclass
class VLLMCompletionResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> VLLMCompletionResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._client.post("/v1/completions", json=payload)

        if response.status_code != 200:
            raise VLLMInferenceError(
                f"vLLM returned HTTP {response.status_code}: {response.text}"
            )

        body = response.json()
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError) as exc:
            raise VLLMInferenceError(f"malformed vLLM response: {body}") from exc

        usage = body.get("usage") or {}
        return VLLMCompletionResult(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
