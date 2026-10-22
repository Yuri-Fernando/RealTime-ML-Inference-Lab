"""Contrato dos backends de inferência (V0.3).

Dois tipos de backend convivem no repositório:

- ``kind == "tensor"``: o ``ModelBackend`` histórico (TinyCNN), usado por
  ``inference_service/app.py``. Recebe um tensor (B, 3, 32, 32) e devolve
  probabilidades. Continua intacto.
- ``kind == "text"``: backends de geração de texto (``HuggingFaceBackend`` e
  ``VLLMBackend``), usados por ``inference_service/llm_app.py`` através do
  ``LLMBatcher``.

Os dois expõem os mesmos contadores (``forward_pass_count`` e
``total_items_processed``): é com eles que os testes provam que o batching e
o cancelamento são reais (menos chamadas ao modelo do que requisições; item
cancelado nunca chega ao modelo).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable


class BackendConfigError(RuntimeError):
    """Configuração de backend inválida ou impossível neste ambiente."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 32


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # TTFT medido no backend: tempo do início do lote até o primeiro token
    # (fim do prefill). None quando o backend não consegue medir (vLLM sem
    # streaming).
    ttft_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    cancelled: bool = False
    extra: dict = field(default_factory=dict)


@runtime_checkable
class InferenceBackend(Protocol):
    """Superfície mínima comum a qualquer backend."""

    name: str
    kind: str  # "tensor" | "text"
    device: object
    forward_pass_count: int
    total_items_processed: int

    def reset_counters(self) -> None: ...

    def describe(self) -> dict: ...


@runtime_checkable
class TextBackend(InferenceBackend, Protocol):
    """Backend de geração de texto.

    ``supports_batching`` decide o modo do ``LLMBatcher``:
    - True  -> o batcher agrupa requisições e chama ``generate_batch``
              (bloqueante, executado em thread).
    - False -> cada requisição vira uma chamada ``agenerate`` concorrente
              (o servidor remoto, ex. vLLM, faz o batching contínuo).
    """

    supports_batching: bool

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
        cancel_events: Optional[Sequence[threading.Event]] = None,
    ) -> list[GenerationResult]: ...

    async def agenerate(self, request: GenerationRequest) -> GenerationResult: ...
