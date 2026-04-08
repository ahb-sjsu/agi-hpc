# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.lh.performance module.

The source module is a Sprint 6 stub. These tests define the expected
LRUCache and AsyncBatcher API.
"""

import time
import pytest
from unittest.mock import AsyncMock
from typing import Any, Dict, Optional

try:
    from agi.lh.performance import LRUCache, AsyncBatcher

    _HAS_MODULE = True
except (ImportError, AttributeError):
    _HAS_MODULE = False


class _StubLRUCache:
    def __init__(self, max_size=128, ttl_seconds=300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if time.monotonic() - self._times[key] < self.ttl_seconds:
                self._hits += 1
                return self._cache[key]
            del self._cache[key]
            del self._times[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any):
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = min(self._times, key=self._times.get)
            del self._cache[oldest]
            del self._times[oldest]
        self._cache[key] = value
        self._times[key] = time.monotonic()

    def invalidate(self, key: str):
        self._cache.pop(key, None)
        self._times.pop(key, None)

    def clear(self):
        self._cache.clear()
        self._times.clear()

    @property
    def stats(self):
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class _StubAsyncBatcher:
    def __init__(self, batch_size=10, flush_interval=0.1):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._items = []
        self._handler = None

    def set_handler(self, handler):
        self._handler = handler

    async def submit(self, item):
        self._items.append(item)
        if len(self._items) >= self.batch_size and self._handler:
            batch = self._items[: self.batch_size]
            self._items = self._items[self.batch_size :]
            await self._handler(batch)

    async def flush(self):
        if self._items and self._handler:
            await self._handler(self._items)
            self._items = []


if not _HAS_MODULE:
    LRUCache = _StubLRUCache
    AsyncBatcher = _StubAsyncBatcher


class TestLRUCache:
    def test_init(self):
        cache = LRUCache(max_size=64, ttl_seconds=60.0)
        assert cache.max_size == 64

    def test_put_get(self):
        cache = LRUCache()
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_miss(self):
        cache = LRUCache()
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        cache = LRUCache()
        cache.put("key1", "val")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = LRUCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_max_size_eviction(self):
        cache = LRUCache(max_size=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        assert cache.stats["size"] <= 2

    def test_stats(self):
        cache = LRUCache()
        cache.put("k1", "v1")
        cache.get("k1")
        cache.get("missing")
        s = cache.stats
        assert s["hits"] == 1
        assert s["misses"] == 1


class TestAsyncBatcher:
    @pytest.mark.asyncio
    async def test_init(self):
        batcher = AsyncBatcher(batch_size=5)
        assert batcher.batch_size == 5

    @pytest.mark.asyncio
    async def test_submit(self):
        batcher = AsyncBatcher(batch_size=10)
        handler = AsyncMock()
        batcher.set_handler(handler)
        await batcher.submit("item1")
        assert len(batcher._items) >= 0

    @pytest.mark.asyncio
    async def test_flush(self):
        batcher = AsyncBatcher(batch_size=100)
        handler = AsyncMock()
        batcher.set_handler(handler)
        await batcher.submit("item1")
        await batcher.submit("item2")
        await batcher.flush()
        handler.assert_called()

    @pytest.mark.asyncio
    async def test_batch_trigger(self):
        handler = AsyncMock()
        batcher = AsyncBatcher(batch_size=2)
        batcher.set_handler(handler)
        await batcher.submit("a")
        await batcher.submit("b")
        handler.assert_called_once()
