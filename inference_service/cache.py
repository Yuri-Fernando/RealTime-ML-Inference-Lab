"""Real result cache keyed by a hash of the (rounded) input vector.

This is a plain in-process LRU cache guarded by an asyncio lock. It is real
and testable: identical inputs produce cache hits and skip the forward
pass; different inputs are cache misses.

Floating point inputs are rounded before hashing so that two numerically
identical requests (e.g. the same JSON payload serialized twice) reliably
hash to the same key.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import Optional

from inference_service.config import settings


def hash_input(values: list[float], round_ndigits: int = 6) -> str:
    """Deterministically hash a list of floats into a cache key."""
    rounded = [round(v, round_ndigits) for v in values]
    payload = ",".join(f"{v:.{round_ndigits}f}" for v in rounded).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResultCache:
    """Thread-safe-within-event-loop LRU cache of inference results."""

    def __init__(self, max_entries: int = settings.CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._store: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[list[float]]:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    async def put(self, key: str, value: list[float]) -> None:
        async with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


result_cache = ResultCache()
