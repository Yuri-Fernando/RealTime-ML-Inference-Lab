"""Tests for inference_service.cache: real hit/miss and LRU eviction behavior."""
from inference_service.cache import ResultCache, hash_input


def test_hash_input_deterministic():
    values = [0.1, 0.2, 0.3]
    assert hash_input(values) == hash_input(values)


def test_hash_input_differs_for_different_values():
    assert hash_input([0.1, 0.2]) != hash_input([0.1, 0.3])


def test_hash_input_robust_to_float_noise_within_rounding():
    a = [1 / 3, 2 / 3]
    b = [0.3333333333333333, 0.6666666666666666]
    assert hash_input(a) == hash_input(b)


async def test_cache_miss_then_hit():
    cache = ResultCache(max_entries=10)
    key = hash_input([1.0, 2.0])

    assert await cache.get(key) is None
    assert cache.misses == 1
    assert cache.hits == 0

    await cache.put(key, [9.0])
    result = await cache.get(key)

    assert result == [9.0]
    assert cache.hits == 1


async def test_cache_lru_eviction():
    cache = ResultCache(max_entries=2)
    await cache.put("a", [1.0])
    await cache.put("b", [2.0])
    await cache.put("c", [3.0])  # should evict "a" (least recently used)

    assert await cache.get("a") is None
    assert await cache.get("b") == [2.0]
    assert await cache.get("c") == [3.0]
    assert len(cache) == 2


async def test_cache_get_refreshes_recency():
    cache = ResultCache(max_entries=2)
    await cache.put("a", [1.0])
    await cache.put("b", [2.0])
    await cache.get("a")  # "a" is now most recently used
    await cache.put("c", [3.0])  # should evict "b", not "a"

    assert await cache.get("b") is None
    assert await cache.get("a") == [1.0]
    assert await cache.get("c") == [3.0]
