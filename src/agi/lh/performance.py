# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Performance optimization for the Left Hemisphere.

Provides:
- LRU cache with TTL for query results
- Async operation batcher for throughput
- Cache statistics and monitoring

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PerformanceConfig:
    """Configuration for LH performance optimizations."""

    cache_max_size: int = 128
    cache_ttl_seconds: float = 300.0
    batch_size: int = 10
    flush_interval: float = 0.1


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------


class LRUCache:
    """Thread-safe LRU cache with TTL-based expiration.

    Entries are evicted when the cache exceeds *max_size* (oldest by
    insertion time removed first) or when an entry's age exceeds
    *ttl_seconds*.
    """

    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        logger.info(
            "[lh][perf] LRUCache initialized max_size=%d ttl=%.1fs",
            max_size,
            ttl_seconds,
        )

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if absent / expired."""
        with self._lock:
            if key in self._cache:
                if time.monotonic() - self._times[key] < self.ttl_seconds:
                    self._hits += 1
                    return self._cache[key]
                # Expired — remove
                del self._cache[key]
                del self._times[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Insert or update a cache entry."""
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict oldest entry
                oldest = min(self._times, key=self._times.get)  # type: ignore[arg-type]
                del self._cache[oldest]
                del self._times[oldest]
            self._cache[key] = value
            self._times[key] = time.monotonic()

    def invalidate(self, key: str) -> None:
        """Remove a single entry."""
        with self._lock:
            self._cache.pop(key, None)
            self._times.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()
            self._times.clear()

    @property
    def stats(self) -> Dict[str, int]:
        """Return hit/miss/size statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
            }


# ---------------------------------------------------------------------------
# Async Batcher
# ---------------------------------------------------------------------------


class AsyncBatcher:
    """Batches async operations and flushes when the batch is full.

    A handler coroutine is called with a list of accumulated items
    whenever the batch reaches *batch_size* or :meth:`flush` is called.
    """

    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 0.1,
    ) -> None:
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._items: List[Any] = []
        self._handler: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None
        logger.info(
            "[lh][perf] AsyncBatcher initialized batch_size=%d interval=%.3fs",
            batch_size,
            flush_interval,
        )

    def set_handler(
        self,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Set the async handler that processes each batch."""
        self._handler = handler

    async def submit(self, item: Any) -> None:
        """Add an item. Triggers handler if batch is full."""
        self._items.append(item)
        if len(self._items) >= self.batch_size and self._handler:
            batch = self._items[: self.batch_size]
            self._items = self._items[self.batch_size :]
            await self._handler(batch)

    async def flush(self) -> None:
        """Flush remaining items through the handler."""
        if self._items and self._handler:
            await self._handler(self._items)
            self._items = []
