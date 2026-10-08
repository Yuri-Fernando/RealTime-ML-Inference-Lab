"""Tests for real request cancellation in DynamicBatcher.

These prove cancellation is real, not cosmetic:
  - a request cancelled while still sitting in the queue (worker never
    started on it) is provably excluded from the forward pass — it does not
    increment model_backend.forward_pass_count / total_items_processed.
  - the caller's `submit()` raises RequestCancelledError instead of hanging
    or receiving a stale result.
  - cancelling an unknown/already-finished request_id is a safe no-op
    (returns False), not an error.
"""
import asyncio

import pytest

from inference_service.batching import DynamicBatcher, PendingRequest, RequestCancelledError
from inference_service.config import settings
from inference_service.model import ModelBackend


def _random_input() -> list[float]:
    import random

    return [random.random() for _ in range(settings.INPUT_DIM)]


@pytest.fixture
def model_backend():
    backend = ModelBackend()
    backend.reset_counters()
    return backend


async def test_cancel_before_worker_picks_it_up_never_touches_the_model(model_backend):
    """Start the batcher with a long batch window so the request sits in the
    queue long enough for us to cancel it before the worker drains it."""
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=16,
        max_wait_ms=2000,  # long window: the request will still be queued
        max_queue_depth=64,
    )
    await batcher.start()
    try:
        submit_task = asyncio.create_task(batcher.submit(_random_input(), request_id="req-1"))
        await asyncio.sleep(0.05)  # let it register as pending/in-flight

        cancelled = batcher.cancel("req-1")
        assert cancelled is True

        with pytest.raises(RequestCancelledError):
            await submit_task

        # The whole point: cancellation happened before the batch window
        # closed, so the model never ran for this request.
        assert model_backend.forward_pass_count == 0
        assert model_backend.total_items_processed == 0
    finally:
        await batcher.stop()


async def test_cancelled_request_excluded_from_a_batch_with_other_live_requests(model_backend):
    """Two requests in the same batch window, one cancelled: the live one
    still completes normally and the model only sees the live one."""
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=16,
        max_wait_ms=200,
        max_queue_depth=64,
    )
    await batcher.start()
    try:
        cancel_task = asyncio.create_task(
            batcher.submit(_random_input(), request_id="to-cancel")
        )
        live_task = asyncio.create_task(batcher.submit(_random_input(), request_id="to-keep"))
        await asyncio.sleep(0.02)

        assert batcher.cancel("to-cancel") is True

        with pytest.raises(RequestCancelledError):
            await cancel_task

        live_result = await live_task
        assert len(live_result) == settings.NUM_CLASSES

        # Only the live request's item should have reached the model.
        assert model_backend.total_items_processed == 1
        assert model_backend.forward_pass_count == 1
    finally:
        await batcher.stop()


async def test_cancel_unknown_request_id_is_a_safe_no_op(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend, max_batch_size=16, max_wait_ms=10, max_queue_depth=64
    )
    await batcher.start()
    try:
        assert batcher.cancel("does-not-exist") is False
    finally:
        await batcher.stop()


async def test_cancel_after_completion_is_a_safe_no_op(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend, max_batch_size=16, max_wait_ms=10, max_queue_depth=64
    )
    await batcher.start()
    try:
        result = await batcher.submit(_random_input(), request_id="finished")
        assert len(result) == settings.NUM_CLASSES
        # Already popped from _pending_by_id on completion -> no-op.
        assert batcher.cancel("finished") is False
    finally:
        await batcher.stop()


async def test_double_cancel_is_idempotent(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=16,
        max_wait_ms=2000,
        max_queue_depth=64,
    )
    await batcher.start()
    try:
        submit_task = asyncio.create_task(batcher.submit(_random_input(), request_id="req-2"))
        await asyncio.sleep(0.05)

        assert batcher.cancel("req-2") is True
        assert batcher.cancel("req-2") is False  # already cancelled -> no-op

        with pytest.raises(RequestCancelledError):
            await submit_task
    finally:
        await batcher.stop()


async def test_on_cancelled_hook_receives_reason(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=16,
        max_wait_ms=2000,
        max_queue_depth=64,
    )
    reasons: list[str] = []
    batcher.on_cancelled = reasons.append
    await batcher.start()
    try:
        submit_task = asyncio.create_task(batcher.submit(_random_input(), request_id="req-3"))
        await asyncio.sleep(0.05)
        batcher.cancel("req-3", reason="client_disconnect")
        with pytest.raises(RequestCancelledError):
            await submit_task
        assert reasons == ["client_disconnect"]
    finally:
        await batcher.stop()
