"""VLLMBackend: cliente real de um servidor vLLM (API OpenAI-compatible).

O vLLM roda FORA deste processo (no Windows ele não roda nativo; ver
``scripts/run_vllm_wsl.sh`` e ``docker-compose.vllm.yml``). Este backend só
fala o protocolo ``POST /v1/completions`` por HTTP, reaproveitando
``adapters/triton_vllm/vllm_client.py``.

Como o vLLM faz batching contínuo no servidor, ``supports_batching`` é False:
o ``LLMBatcher`` não agrupa requisições aqui; ele apenas aplica o controle de
admissão e despacha cada requisição de forma concorrente. Cancelar uma
requisição cancela a tarefa asyncio, o que fecha a conexão HTTP (o vLLM aborta
a geração da requisição quando o cliente desconecta).

Honestidade: a correção do protocolo é coberta por testes contra um servidor
mock (``tests/test_vllm_backend.py``). Um benchmark contra um vLLM real só
existe se ``benchmarks/results/`` contiver o arquivo gerado por uma execução
real; veja o README.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import httpx

from adapters.triton_vllm.vllm_client import VLLMClient
from inference_service.backends.base import GenerationRequest, GenerationResult
from inference_service.config import settings


class VLLMBackend:
    kind = "text"
    supports_batching = False

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = (base_url or settings.VLLM_BASE_URL).rstrip("/")
        self.model = model or settings.VLLM_MODEL
        self.name = "vllm"
        self.device = "remote:vllm"
        self._client = VLLMClient(
            base_url=self.base_url, model=self.model, timeout_s=timeout_s, transport=transport
        )
        self._lock = threading.Lock()
        self.forward_pass_count = 0  # chamadas ao servidor
        self.total_items_processed = 0
        self.tokens_generated_total = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    def reset_counters(self) -> None:
        with self._lock:
            self.forward_pass_count = 0
            self.total_items_processed = 0
            self.tokens_generated_total = 0

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "device": self.device,
            "batching": "server-side (continuous batching do vLLM)",
        }

    async def agenerate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            self.forward_pass_count += 1
            self.total_items_processed += 1
        result = await self._client.complete(
            request.prompt, max_tokens=request.max_new_tokens, temperature=0.0
        )
        completion = result.completion_tokens or 0
        with self._lock:
            self.tokens_generated_total += completion
        return GenerationResult(
            text=result.text,
            prompt_tokens=result.prompt_tokens or 0,
            completion_tokens=completion,
            ttft_ms=None,  # sem streaming nesta versão: TTFT não é medido
            finish_reason=result.finish_reason,
        )

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
        cancel_events: Optional[Sequence[threading.Event]] = None,
    ) -> list[GenerationResult]:
        raise NotImplementedError(
            "VLLMBackend não agrupa no cliente; use agenerate (batching contínuo no servidor)"
        )

    async def is_ready(self) -> bool:
        """True se o servidor responde em GET /v1/models (nada é simulado)."""
        try:
            resp = await self._client._client.get("/v1/models")  # noqa: SLF001
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
