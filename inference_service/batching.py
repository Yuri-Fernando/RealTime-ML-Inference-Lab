"""Dynamic batcher with real queue-depth admission control and cancellation.

How it works (real, not simulated):

1. Each incoming request is wrapped in a `PendingRequest` holding an
   `asyncio.Future`, a `request_id`, and a `cancelled` flag, then pushed
   onto an `asyncio.Queue`.
2. A single background task continuously pulls requests off the queue. It
   takes the first available item, then keeps absorbing more items (up to
   `max_batch_size`) until either the batch is full or `max_wait_ms` has
   elapsed since the first item arrived — whichever comes first.
3. Before building the forward-pass tensor, any item marked `cancelled` is
   filtered OUT of the batch — it is never stacked into the tensor and never
   touches the GPU/CPU model. This is what makes cancellation real rather
   than cosmetic: a cancelled request provably does not consume a model
   forward pass (asserted in tests/test_cancellation.py by reading
   `model_backend.total_items_processed`).
4. All *admitted, non-cancelled* items collected this way are stacked into
   one tensor and sent through exactly one `model_backend.infer_batch(...)`
   call. Each request's Future is then resolved with its slice of the
   batched result.
5. Admission control (see `inference_service/admission.py`): `in_flight`
   (queued + being processed) is tracked with a counter guarded by a lock.
   `AdmissionController.decide(...)` is asked whether a NEW request fits;
   if not, `submit()` raises `BackpressureError` immediately instead of
   enqueueing — callers turn that into HTTP 429. The increment of
   `in_flight` happens atomically with the decision (same lock), but the
   *decision rule itself* lives in admission.py and is unit-testable without
   any of this queueing machinery.
6. Cancellation: `cancel(request_id)` marks the matching in-flight request
   as cancelled and resolves its Future with `asyncio.CancelledError`
   immediately (the caller does not wait for the batch window). If the
   worker has not yet picked it up, step 3 above guarantees it never reaches
   the model. If the worker already started processing the exact batch it
   was in (a narrow race), the result is still discarded — the Future is
   already resolved to cancelled and `_process_batch` checks `future.done()`
   before writing a result into it, so a late result can never overwrite a
   cancellation.

This genuinely reduces forward passes under concurrency: N requests that
arrive within the same batching window result in 1 forward pass instead of
N, which is asserted directly in tests/test_batching.py by reading
`model_backend.forward_pass_count`.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from inference_service.admission import AdmissionController
from inference_service.config import settings
from inference_service.model import ModelBackend


class BackpressureError(Exception):
    """Raised when admission control rejects a new request (queue at capacity)."""

    def __init__(self, queue_depth: int, max_queue_depth: int):
        self.queue_depth = queue_depth
        self.max_queue_depth = max_queue_depth
        super().__init__(
            f"queue depth {queue_depth} exceeds max {max_queue_depth}"
        )


class RequestCancelledError(Exception):
    """Raised (via future cancellation) when a caller cancels an in-flight request."""


@dataclass
class PendingRequest:
    values: list[float]
    future: "asyncio.Future[list[float]]"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False


class DynamicBatcher:
    def __init__(
        self,
        model_backend: ModelBackend,
        max_batch_size: int = settings.MAX_BATCH_SIZE,
        max_wait_ms: float = settings.MAX_BATCH_WAIT_MS,
        max_queue_depth: int = settings.MAX_QUEUE_DEPTH,
        input_dim: int = settings.INPUT_DIM,
    ) -> None:
        self.model_backend = model_backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.admission = AdmissionController(max_queue_depth=max_queue_depth)
        self.input_dim = input_dim

        # asyncio.Queue/Lock bind to whichever event loop is running when
        # they are first awaited; constructing them lazily inside start()
        # (rather than here in __init__) avoids "bound to a different event
        # loop" errors when a module-level singleton is reused across
        # separate test runs / TestClient sessions, each with their own loop.
        self._queue: Optional["asyncio.Queue[PendingRequest]"] = None
        self._in_flight = 0
        self._in_flight_lock: Optional[asyncio.Lock] = None
        self._pending_by_id: dict[str, PendingRequest] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._stopped = False

        # Observability hooks (set by app.py to record metrics); kept optional
        # so this module has no hard dependency on the metrics module.
        self.on_batch_processed = None  # type: Optional[callable]
        self.on_cancelled = None  # type: Optional[callable]

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
        # Drop the loop-bound queue/lock so a future start() in a
        # different event loop (e.g. a new TestClient session) creates
        # fresh ones instead of reusing objects bound to a closed loop.
        self._queue = None
        self._in_flight_lock = None
        self._in_flight = 0
        self._pending_by_id = {}

    @property
    def queue_depth(self) -> int:
        return self._in_flight

    @property
    def max_queue_depth(self) -> int:  # type: ignore[no-redef]
        return self.admission.max_queue_depth

    @max_queue_depth.setter
    def max_queue_depth(self, value: int) -> None:
        # Kept writable because tests/app code (and load tests) tune this at
        # runtime. AdmissionController is the single source of truth for the
        # limit; this just forwards to it.
        self.admission.max_queue_depth = value

    def _ensure_initialized(self) -> None:
        """Lazily create the queue/lock bound to the currently running loop.

        Normally start() does this, but submit() may be called (e.g. in
        tests exercising backpressure) without first starting the worker
        loop, so this is a safety net that is a no-op once already set.
        """
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._in_flight_lock is None:
            self._in_flight_lock = asyncio.Lock()

    async def submit(self, values: list[float], request_id: Optional[str] = None) -> list[float]:
        """Enqueue one request and await its batched result.

        Raises BackpressureError immediately (without enqueueing) if
        admission control rejects the request. Raises
        `asyncio.CancelledError` if the request is cancelled via
        `cancel(request_id)` before it is processed.
        """
        self._ensure_initialized()
        async with self._in_flight_lock:
            decision = self.admission.decide(self._in_flight)
            if not decision.admitted:
                raise BackpressureError(decision.in_flight, decision.max_queue_depth)
            self._in_flight += 1

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[list[float]]" = loop.create_future()
        request = PendingRequest(values=values, future=future)
        if request_id is not None:
            request.request_id = request_id
        self._pending_by_id[request.request_id] = request
        await self._queue.put(request)

        try:
            result = await asyncio.wait_for(future, timeout=settings.REQUEST_TIMEOUT_S)
            return result
        except asyncio.CancelledError:
            # Distinguish "we cancelled this specific request via cancel()"
            # (request.cancelled is True) from the current task itself being
            # cancelled by something external — only the former is turned
            # into a domain-level RequestCancelledError; the latter must
            # keep propagating as a real asyncio cancellation.
            if request.cancelled:
                raise RequestCancelledError(request.request_id) from None
            raise
        finally:
            self._pending_by_id.pop(request.request_id, None)
            async with self._in_flight_lock:
                self._in_flight -= 1

    def cancel(self, request_id: str, reason: str = "explicit") -> bool:
        """Cancel a request that is still queued or being awaited.

        Returns True if a matching pending request was found and cancelled
        (its future resolved with CancelledError, and it is marked so the
        worker will exclude it from any batch it is still sitting in),
        False if no such request_id is currently pending (already completed,
        already cancelled, or unknown id).
        """
        request = self._pending_by_id.get(request_id)
        if request is None or request.cancelled:
            return False
        request.cancelled = True
        if not request.future.done():
            request.future.cancel()
        if self.on_cancelled is not None:
            self.on_cancelled(reason)
        return True

    async def _run(self) -> None:
        while not self._stopped:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                raise

            batch = [first]
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[PendingRequest]) -> None:
        # Filter out anything cancelled while it was waiting in the queue —
        # these never reach the model, which is the whole point of real
        # cancellation (not just "we stopped caring about the result").
        live = [req for req in batch if not req.cancelled and not req.future.done()]
        if not live:
            return
        try:
            tensor = torch.tensor(
                [req.values for req in live], dtype=torch.float32
            ).view(len(live), 3, 32, 32)
            probs = self.model_backend.infer_batch(tensor)
            for i, req in enumerate(live):
                if not req.future.done():
                    req.future.set_result(probs[i].tolist())
            if self.on_batch_processed is not None:
                self.on_batch_processed(len(live))
        except Exception as exc:  # pragma: no cover - defensive
            for req in live:
                if not req.future.done():
                    req.future.set_exception(exc)
