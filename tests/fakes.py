"""Backends de teste (sem modelo): isolam o comportamento do batcher/app.

São dublês de backend — o componente sob teste aqui é o ``LLMBatcher`` e o
``llm_app``, não a geração em si (que é coberta com o modelo real em
``tests/test_hf_backend.py``).
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional, Sequence

from inference_service.backends.base import GenerationRequest, GenerationResult


class FakeTextBackend:
    """Backend com batching (como o HuggingFaceBackend), geração simulada por sleep."""

    kind = "text"
    supports_batching = True
    name = "fake"
    device = "cpu"

    def __init__(self, delay_s: float = 0.05, step_s: float = 0.01) -> None:
        self.delay_s = delay_s
        self.step_s = step_s
        self._lock = threading.Lock()
        self.forward_pass_count = 0
        self.total_items_processed = 0
        self.tokens_generated_total = 0
        self.batches: list[int] = []
        self.seen_requests: list[GenerationRequest] = []
        self.cancel_seen: list[bool] = []

    def reset_counters(self) -> None:
        with self._lock:
            self.forward_pass_count = 0
            self.total_items_processed = 0
            self.tokens_generated_total = 0
            self.batches.clear()
            self.seen_requests.clear()
            self.cancel_seen.clear()

    def describe(self) -> dict:
        return {"backend": self.name, "kind": self.kind, "model": "fake", "quant": "none"}

    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
        cancel_events: Optional[Sequence[threading.Event]] = None,
    ) -> list[GenerationResult]:
        events = list(cancel_events) if cancel_events else [threading.Event() for _ in requests]
        with self._lock:
            self.forward_pass_count += 1
            self.total_items_processed += len(requests)
            self.batches.append(len(requests))
            self.seen_requests.extend(requests)
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline:
            if all(e.is_set() for e in events):
                break
            time.sleep(self.step_s)
        out = []
        for r, e in zip(requests, events):
            cancelled = e.is_set()
            with self._lock:
                self.cancel_seen.append(cancelled)
            n = 1 if cancelled else r.max_new_tokens
            with self._lock:
                self.tokens_generated_total += n
            out.append(
                GenerationResult(
                    text=f"echo:{r.prompt}",
                    prompt_tokens=len(r.prompt.split()),
                    completion_tokens=n,
                    ttft_ms=1.0,
                    finish_reason="cancelled" if cancelled else "length",
                    cancelled=cancelled,
                )
            )
        return out

    async def agenerate(self, request: GenerationRequest) -> GenerationResult:  # pragma: no cover
        return (await asyncio.to_thread(self.generate_batch, [request]))[0]


class FakeRemoteBackend:
    """Backend sem batching no cliente (como o VLLMBackend)."""

    kind = "text"
    supports_batching = False
    name = "fake-remote"
    device = "remote"

    def __init__(self, delay_s: float = 0.2) -> None:
        self.delay_s = delay_s
        self.forward_pass_count = 0
        self.total_items_processed = 0
        self.tokens_generated_total = 0
        self.running = 0
        self.max_running = 0
        self.cancelled_calls = 0

    def reset_counters(self) -> None:
        self.forward_pass_count = 0
        self.total_items_processed = 0

    def describe(self) -> dict:
        return {"backend": self.name, "kind": self.kind, "model": "fake-remote"}

    def generate_batch(self, requests, cancel_events=None):  # pragma: no cover
        raise NotImplementedError

    async def agenerate(self, request: GenerationRequest) -> GenerationResult:
        self.forward_pass_count += 1
        self.total_items_processed += 1
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled_calls += 1
            raise
        finally:
            self.running -= 1
        return GenerationResult(
            text=f"remote:{request.prompt}",
            prompt_tokens=1,
            completion_tokens=request.max_new_tokens,
            ttft_ms=None,
            finish_reason="length",
        )
