"""Batcher dinâmico para geração de texto (V0.3).

Mesma ideia do ``DynamicBatcher`` (fila + janela de espera + admissão +
cancelamento), adaptada a geração:

- Admissão: reaproveita ``AdmissionController`` e as exceções
  ``BackpressureError``/``RequestCancelledError`` do batcher histórico.
- Modo com batching (``backend.supports_batching``, ex. HuggingFaceBackend):
  o worker junta até ``max_batch_size`` requisições dentro de ``max_wait_ms`` e
  chama ``backend.generate_batch`` UMA vez, numa thread (``asyncio.to_thread``)
  para não bloquear o event loop durante a geração, que dura segundos.
- Modo sem batching (ex. VLLMBackend): cada requisição vira uma tarefa
  ``backend.agenerate`` concorrente; o servidor remoto faz o batching contínuo.
- Cancelamento real: um item cancelado ANTES de entrar no lote nunca chega ao
  modelo (testado por ``total_items_processed``). Um item cancelado DURANTE a
  geração tem seu ``threading.Event`` setado e o ``StoppingCriteria`` por
  sequência do backend para de gerar tokens para ele. No modo remoto, cancelar
  cancela a tarefa asyncio (fecha a conexão HTTP).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from inference_service.admission import AdmissionController
from inference_service.backends.base import GenerationRequest, GenerationResult
from inference_service.batching import BackpressureError, RequestCancelledError
from inference_service.config import settings

__all__ = ["LLMBatcher", "BackpressureError", "RequestCancelledError"]


@dataclass
class PendingGeneration:
    request: GenerationRequest
    future: "asyncio.Future[GenerationResult]"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    task: Optional["asyncio.Task"] = None  # só no modo sem batching


class LLMBatcher:
    def __init__(
        self,
        backend,
        max_batch_size: int = settings.LLM_MAX_BATCH_SIZE,
        max_wait_ms: float = settings.LLM_MAX_BATCH_WAIT_MS,
        max_queue_depth: int = settings.LLM_MAX_QUEUE_DEPTH,
        request_timeout_s: float = settings.LLM_REQUEST_TIMEOUT_S,
        max_new_tokens_cap: int = settings.LLM_MAX_NEW_TOKENS_CAP,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.admission = AdmissionController(max_queue_depth=max_queue_depth)
        self.request_timeout_s = request_timeout_s
        self.max_new_tokens_cap = max_new_tokens_cap

        self._queue: Optional["asyncio.Queue[PendingGeneration]"] = None
        self._in_flight = 0
        self._in_flight_lock: Optional[asyncio.Lock] = None
        self._pending_by_id: dict[str, PendingGeneration] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._tasks: set[asyncio.Task] = set()
        self._stopped = False

        # Ganchos de observabilidade (definidos por llm_app.py).
        self.on_batch_processed: Optional[Callable[[int, list[GenerationResult], float], None]] = None
        self.on_cancelled: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------ ciclo de vida
    async def start(self) -> None:
        if self._worker_task is None:
            self._ensure_initialized()
            self._in_flight = 0
            self._pending_by_id = {}
            self._stopped = False
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        self._queue = None
        self._in_flight_lock = None
        self._in_flight = 0
        self._pending_by_id = {}

    def _ensure_initialized(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._in_flight_lock is None:
            self._in_flight_lock = asyncio.Lock()

    @property
    def queue_depth(self) -> int:
        return self._in_flight

    @property
    def max_queue_depth(self) -> int:
        return self.admission.max_queue_depth

    @max_queue_depth.setter
    def max_queue_depth(self, value: int) -> None:
        self.admission.max_queue_depth = value

    # ----------------------------------------------------------------- submit
    async def submit(
        self, prompt: str, max_new_tokens: int = 32, request_id: Optional[str] = None
    ) -> GenerationResult:
        """Enfileira uma geração e aguarda o resultado.

        Levanta ``BackpressureError`` (sem enfileirar) se a admissão rejeitar,
        ``RequestCancelledError`` se for cancelada via ``cancel`` e
        ``TimeoutError`` se estourar ``request_timeout_s``.
        """
        self._ensure_initialized()
        budget = max(1, min(int(max_new_tokens), self.max_new_tokens_cap))
        async with self._in_flight_lock:
            decision = self.admission.decide(self._in_flight)
            if not decision.admitted:
                raise BackpressureError(decision.in_flight, decision.max_queue_depth)
            self._in_flight += 1

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[GenerationResult]" = loop.create_future()
        pending = PendingGeneration(
            request=GenerationRequest(prompt=prompt, max_new_tokens=budget), future=future
        )
        if request_id is not None:
            pending.request_id = request_id
        self._pending_by_id[pending.request_id] = pending
        await self._queue.put(pending)

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout_s)
        except asyncio.TimeoutError as exc:
            # Estourou o tempo: sinaliza o backend para parar de gastar GPU.
            self._mark_cancelled(pending)
            raise TimeoutError(f"generation timed out after {self.request_timeout_s}s") from exc
        except asyncio.CancelledError:
            if pending.cancelled:
                raise RequestCancelledError(pending.request_id) from None
            raise
        finally:
            self._pending_by_id.pop(pending.request_id, None)
            async with self._in_flight_lock:
                self._in_flight -= 1

    def _mark_cancelled(self, pending: PendingGeneration) -> None:
        pending.cancelled = True
        pending.cancel_event.set()
        if pending.task is not None and not pending.task.done():
            pending.task.cancel()

    def cancel(self, request_id: str, reason: str = "explicit") -> bool:
        """Cancela uma requisição pendente. True se encontrou e cancelou."""
        pending = self._pending_by_id.get(request_id)
        if pending is None or pending.cancelled:
            return False
        self._mark_cancelled(pending)
        if not pending.future.done():
            pending.future.cancel()
        if self.on_cancelled is not None:
            self.on_cancelled(reason)
        return True

    # ------------------------------------------------------------------ worker
    async def _run(self) -> None:
        batching = bool(getattr(self.backend, "supports_batching", False))
        while not self._stopped:
            first = await self._queue.get()
            if not batching:
                task = asyncio.create_task(self._process_single(first))
                first.task = task
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                continue

            batch = [first]
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[PendingGeneration]) -> None:
        # Itens cancelados enquanto esperavam na fila nunca chegam ao modelo.
        live = [p for p in batch if not p.cancelled and not p.future.done()]
        if not live:
            return
        started = time.monotonic()
        try:
            results = await asyncio.to_thread(
                self.backend.generate_batch,
                [p.request for p in live],
                [p.cancel_event for p in live],
            )
        except Exception as exc:  # pragma: no cover - defensivo
            for p in live:
                if not p.future.done():
                    p.future.set_exception(exc)
            return
        batch_seconds = time.monotonic() - started
        for p, res in zip(live, results):
            if p.future.done():
                continue
            # TTFT visto pelo cliente = espera em fila + TTFT do lote.
            if res.ttft_ms is not None:
                res.ttft_ms += (started - p.enqueued_at) * 1000.0
            p.future.set_result(res)
        if self.on_batch_processed is not None:
            self.on_batch_processed(len(live), results, batch_seconds)

    async def _process_single(self, pending: PendingGeneration) -> None:
        if pending.cancelled or pending.future.done():
            return
        started = time.monotonic()
        try:
            res = await self.backend.agenerate(pending.request)
        except asyncio.CancelledError:
            return  # cancelamento: a future já foi resolvida em cancel()
        except Exception as exc:
            if not pending.future.done():
                pending.future.set_exception(exc)
            return
        if not pending.future.done():
            pending.future.set_result(res)
        if self.on_batch_processed is not None:
            self.on_batch_processed(1, [res], time.monotonic() - started)
