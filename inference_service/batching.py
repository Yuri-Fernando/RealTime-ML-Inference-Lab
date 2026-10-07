"""Dynamic batcher with real queue-depth backpressure.

How it works (real, not simulated):

1. Each incoming request is wrapped in a `PendingRequest` holding an
   `asyncio.Future` and pushed onto an `asyncio.Queue`.
2. A single background task continuously pulls requests off the queue. It
   takes the first available item, then keeps absorbing more items (up to
   `max_batch_size`) until either the batch is full or `max_wait_ms` has
   elapsed since the first item arrived — whichever comes first.
3. All items collected this way are stacked into one tensor and sent through
   exactly one `model_backend.infer_batch(...)` call. Each request's Future
   is then resolved with its slice of the batched result.
4. Backpressure: `in_flight` (queued + being processed) is tracked with a
   counter. If a new request would push `in_flight` above
   `max_queue_depth`, `submit()` raises `BackpressureError` immediately
   instead of enqueueing — callers turn that into HTTP 429.

This genuinely reduces forward passes under concurrency: N requests that
arrive within the same batching window result in 1 forward pass instead of
N, which is asserted directly in tests/test_batching.py by reading
`model_backend.forward_pass_count`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from inference_service.config import settings
from inference_service.model import ModelBackend


class BackpressureError(Exception):
    """Raised when the pending-request queue is at capacity."""

    def __init__(self, queue_depth: int, max_queue_depth: int):
        self.queue_depth = queue_depth
        self.max_queue_depth = max_queue_depth
        super().__init__(
            f"queue depth {queue_depth} exceeds max {max_queue_depth}"
        )


@dataclass
class PendingRequest:
    values: list[float]
    future: "asyncio.Future[list[float]]"
    enqueued_at: float = field(default_factory=time.monotonic)


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
        self.max_queue_depth = max_queue_depth
        self.input_dim = input_dim

        # asyncio.Queue/Lock bind to whichever event loop is running when
        # they are first awaited; constructing them lazily inside start()
        # (rather than here in __init__) avoids "bound to a different event
        # loop" errors when a module-level singleton is reused across
        # separate test runs / TestClient sessions, each with their own loop.
        self._queue: Optional["asyncio.Queue[PendingRequest]"] = None
        self._in_flight = 0
        self._in_flight_lock: Optional[asyncio.Lock] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._stopped = False

        # Observability hooks (set by app.py to record metrics); kept optional
        # so this module has no hard dependency on the metrics module.
        self.on_batch_processed = None  # type: Optional[callable]

    async def start(self) -> None:
        if self._worker_task is None:
            self._ensure_initialized()
            self._in_flight = 0
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

    @property
    def queue_depth(self) -> int:
        return self._in_flight

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

    async def submit(self, values: list[float]) -> list[float]:
        """Enqueue one request and await its batched result.

        Raises BackpressureError immediately (without enqueueing) if the
        queue is already at capacity.
        """
        self._ensure_initialized()
        async with self._in_flight_lock:
            if self._in_flight >= self.max_queue_depth:
                raise BackpressureError(self._in_flight, self.max_queue_depth)
            self._in_flight += 1

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[list[float]]" = loop.create_future()
        request = PendingRequest(values=values, future=future)
        await self._queue.put(request)

        try:
            result = await asyncio.wait_for(future, timeout=settings.REQUEST_TIMEOUT_S)
            return result
        finally:
            async with self._in_flight_lock:
                self._in_flight -= 1

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
        try:
            tensor = torch.tensor(
                [req.values for req in batch], dtype=torch.float32
            ).view(len(batch), 3, 32, 32)
            probs = self.model_backend.infer_batch(tensor)
            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result(probs[i].tolist())
            if self.on_batch_processed is not None:
                self.on_batch_processed(len(batch))
        except Exception as exc:  # pragma: no cover - defensive
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
