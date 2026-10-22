"""LLMBatcher: batching de geração, admissão, cancelamento e timeout.

Os dublês (tests/fakes.py) isolam o batcher; a geração real é coberta em
tests/test_hf_backend.py.
"""
import asyncio

import pytest

from inference_service.llm_batching import (
    BackpressureError,
    LLMBatcher,
    RequestCancelledError,
)
from tests.fakes import FakeRemoteBackend, FakeTextBackend


async def _started(backend, **kw) -> LLMBatcher:
    batcher = LLMBatcher(backend, **kw)
    await batcher.start()
    return batcher


async def test_concurrent_requests_share_one_generation_batch():
    backend = FakeTextBackend(delay_s=0.05)
    batcher = await _started(backend, max_batch_size=8, max_wait_ms=80)
    try:
        results = await asyncio.gather(*(batcher.submit(f"p{i}", 4) for i in range(6)))
    finally:
        await batcher.stop()
    assert [r.text for r in results] == [f"echo:p{i}" for i in range(6)]
    assert backend.total_items_processed == 6
    # batching real: 6 requisições, menos de 6 chamadas ao backend
    assert backend.forward_pass_count < 6
    assert sum(backend.batches) == 6


async def test_batch_size_is_capped():
    backend = FakeTextBackend(delay_s=0.02)
    batcher = await _started(backend, max_batch_size=4, max_wait_ms=60)
    try:
        await asyncio.gather(*(batcher.submit(f"p{i}", 2) for i in range(10)))
    finally:
        await batcher.stop()
    assert max(backend.batches) <= 4
    assert sum(backend.batches) == 10


async def test_max_new_tokens_is_capped_by_the_batcher():
    backend = FakeTextBackend(delay_s=0.01)
    batcher = await _started(backend, max_new_tokens_cap=16, max_wait_ms=1)
    try:
        await batcher.submit("x", 10_000)
    finally:
        await batcher.stop()
    assert backend.seen_requests[0].max_new_tokens == 16


async def test_admission_rejects_when_queue_is_full():
    backend = FakeTextBackend(delay_s=0.4)
    batcher = await _started(backend, max_batch_size=1, max_wait_ms=1, max_queue_depth=2)
    try:
        a = asyncio.create_task(batcher.submit("a", 2))
        b = asyncio.create_task(batcher.submit("b", 2))
        await asyncio.sleep(0.05)
        assert batcher.queue_depth == 2
        with pytest.raises(BackpressureError) as info:
            await batcher.submit("c", 2)
        assert info.value.max_queue_depth == 2
        await asyncio.gather(a, b)
    finally:
        await batcher.stop()
    # o rejeitado nunca chegou ao backend
    assert backend.total_items_processed == 2


async def test_cancel_before_batch_never_reaches_the_backend():
    backend = FakeTextBackend(delay_s=0.3)
    batcher = await _started(backend, max_batch_size=1, max_wait_ms=1)
    try:
        first = asyncio.create_task(batcher.submit("first", 2, request_id="r1"))
        await asyncio.sleep(0.05)  # "first" já está sendo gerado
        second = asyncio.create_task(batcher.submit("second", 2, request_id="r2"))
        await asyncio.sleep(0.02)  # "second" está na fila
        assert batcher.cancel("r2") is True
        with pytest.raises(RequestCancelledError):
            await second
        await first
    finally:
        await batcher.stop()
    assert backend.total_items_processed == 1
    assert [r.prompt for r in backend.seen_requests] == ["first"]


async def test_cancel_during_generation_signals_only_that_sequence():
    backend = FakeTextBackend(delay_s=0.6)
    batcher = await _started(backend, max_batch_size=4, max_wait_ms=50)
    try:
        a = asyncio.create_task(batcher.submit("a", 8, request_id="ra"))
        b = asyncio.create_task(batcher.submit("b", 8, request_id="rb"))
        await asyncio.sleep(0.3)  # lote já em geração (janela de 50 ms passou)
        assert backend.forward_pass_count == 1
        assert batcher.cancel("ra", reason="explicit") is True
        with pytest.raises(RequestCancelledError):
            await a
        res_b = await b
    finally:
        await batcher.stop()
    assert res_b.cancelled is False and res_b.completion_tokens == 8
    # o backend viu o evento de "a" setado e o de "b" não
    assert sorted(backend.cancel_seen) == [False, True]


async def test_cancel_unknown_or_repeated_returns_false():
    backend = FakeTextBackend(delay_s=0.2)
    batcher = await _started(backend, max_wait_ms=1)
    try:
        task = asyncio.create_task(batcher.submit("a", 2, request_id="r"))
        await asyncio.sleep(0.03)
        assert batcher.cancel("desconhecido") is False
        assert batcher.cancel("r") is True
        assert batcher.cancel("r") is False
        with pytest.raises(RequestCancelledError):
            await task
    finally:
        await batcher.stop()


async def test_timeout_signals_the_backend_to_stop():
    backend = FakeTextBackend(delay_s=0.8)
    batcher = await _started(backend, max_wait_ms=1, request_timeout_s=0.15)
    try:
        with pytest.raises(TimeoutError):
            await batcher.submit("lenta", 4)
        await asyncio.sleep(0.2)
    finally:
        await batcher.stop()
    assert backend.cancel_seen == [True]


async def test_ttft_includes_queue_wait():
    backend = FakeTextBackend(delay_s=0.2)
    batcher = await _started(backend, max_batch_size=1, max_wait_ms=1)
    try:
        first = asyncio.create_task(batcher.submit("a", 2))
        await asyncio.sleep(0.02)
        res2 = await batcher.submit("b", 2)  # espera o lote de "a" terminar
        await first
    finally:
        await batcher.stop()
    assert res2.ttft_ms is not None and res2.ttft_ms > 100  # 1 ms do backend + fila


async def test_batch_hook_receives_size_results_and_duration():
    backend = FakeTextBackend(delay_s=0.03)
    batcher = await _started(backend, max_batch_size=8, max_wait_ms=60)
    seen = []
    batcher.on_batch_processed = lambda size, results, secs: seen.append((size, len(results), secs))
    try:
        await asyncio.gather(*(batcher.submit(f"p{i}", 3) for i in range(3)))
    finally:
        await batcher.stop()
    assert seen and seen[0][0] == seen[0][1] and seen[0][2] > 0


# ---- modo sem batching no cliente (backend remoto, ex. vLLM) ---------------


async def test_remote_mode_dispatches_concurrently_without_grouping():
    backend = FakeRemoteBackend(delay_s=0.2)
    batcher = await _started(backend, max_queue_depth=16)
    try:
        results = await asyncio.gather(*(batcher.submit(f"p{i}", 3) for i in range(5)))
    finally:
        await batcher.stop()
    assert [r.text for r in results] == [f"remote:p{i}" for i in range(5)]
    assert backend.max_running >= 3  # concorrência real, não serial
    assert backend.forward_pass_count == 5  # 1 chamada por requisição


async def test_remote_mode_cancel_cancels_the_in_flight_call():
    backend = FakeRemoteBackend(delay_s=2.0)
    batcher = await _started(backend)
    try:
        task = asyncio.create_task(batcher.submit("x", 2, request_id="rx"))
        await asyncio.sleep(0.1)
        assert backend.running == 1
        assert batcher.cancel("rx") is True
        with pytest.raises(RequestCancelledError):
            await task
        await asyncio.sleep(0.05)
    finally:
        await batcher.stop()
    assert backend.cancelled_calls == 1 and backend.running == 0


async def test_remote_mode_admission_applies_too():
    backend = FakeRemoteBackend(delay_s=0.3)
    batcher = await _started(backend, max_queue_depth=2)
    try:
        tasks = [asyncio.create_task(batcher.submit(f"p{i}", 1)) for i in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(BackpressureError):
            await batcher.submit("extra", 1)
        await asyncio.gather(*tasks)
    finally:
        await batcher.stop()
