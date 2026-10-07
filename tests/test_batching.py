"""Tests for inference_service.batching: real dynamic batching and backpressure.

These tests exercise the real asyncio event loop, the real DynamicBatcher
background task, and the real TinyCNN forward pass (on CPU or GPU,
whichever is available) — nothing here is mocked.
"""
import asyncio

import pytest

from inference_service.batching import BackpressureError, DynamicBatcher
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


async def test_concurrent_requests_are_batched_into_fewer_forward_passes(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=16,
        max_wait_ms=50,
        max_queue_depth=64,
    )
    await batcher.start()
    try:
        n_requests = 10
        inputs = [_random_input() for _ in range(n_requests)]

        results = await asyncio.gather(*(batcher.submit(v) for v in inputs))

        assert len(results) == n_requests
        for r in results:
            assert len(r) == settings.NUM_CLASSES
        # The core batching claim: far fewer forward passes than requests.
        assert model_backend.forward_pass_count < n_requests
        assert model_backend.forward_pass_count >= 1
        assert model_backend.total_items_processed == n_requests
    finally:
        await batcher.stop()


async def test_single_request_still_gets_processed(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend, max_batch_size=16, max_wait_ms=20, max_queue_depth=64
    )
    await batcher.start()
    try:
        result = await batcher.submit(_random_input())
        assert len(result) == settings.NUM_CLASSES
        assert model_backend.forward_pass_count == 1
    finally:
        await batcher.stop()


async def test_batch_size_is_capped_at_max_batch_size(model_backend):
    """More requests than max_batch_size in one window must split into >1 batch."""
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=4,
        max_wait_ms=100,
        max_queue_depth=64,
    )
    await batcher.start()
    try:
        n_requests = 9  # with max_batch_size=4, needs at least 3 batches
        inputs = [_random_input() for _ in range(n_requests)]
        results = await asyncio.gather(*(batcher.submit(v) for v in inputs))

        assert len(results) == n_requests
        assert model_backend.forward_pass_count >= 3
    finally:
        await batcher.stop()


async def test_backpressure_rejects_when_queue_depth_exceeded(model_backend):
    """Force the queue to stay full by never letting the worker drain it,
    then assert the next submit() raises BackpressureError immediately."""
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=1,
        max_wait_ms=5,
        max_queue_depth=2,
    )
    # Do NOT start the worker loop: nothing drains the queue, so in-flight
    # count only grows as submit() is awaited concurrently.
    inputs = [_random_input() for _ in range(2)]

    async def hold(values):
        try:
            await asyncio.wait_for(batcher.submit(values), timeout=0.2)
        except asyncio.TimeoutError:
            pass

    holders = [asyncio.create_task(hold(v)) for v in inputs]
    await asyncio.sleep(0.05)  # let both register as in-flight

    assert batcher.queue_depth == 2

    with pytest.raises(BackpressureError):
        await batcher.submit(_random_input())

    for t in holders:
        t.cancel()
    await asyncio.gather(*holders, return_exceptions=True)


async def test_backpressure_recovers_after_worker_drains_queue(model_backend):
    batcher = DynamicBatcher(
        model_backend=model_backend,
        max_batch_size=8,
        max_wait_ms=10,
        max_queue_depth=2,
    )
    await batcher.start()
    try:
        # Fill to capacity and let the worker actually process + release them.
        results = await asyncio.gather(
            *(batcher.submit(_random_input()) for _ in range(2))
        )
        assert len(results) == 2
        assert batcher.queue_depth == 0  # released after completion

        # Queue has room again now.
        result = await batcher.submit(_random_input())
        assert len(result) == settings.NUM_CLASSES
    finally:
        await batcher.stop()
