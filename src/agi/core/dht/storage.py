# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Storage backends for DHT.

Provides pluggable storage implementations:
- In-memory (testing)
- Redis (distributed cache)
- RocksDB (persistent, future)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage Protocol
# ---------------------------------------------------------------------------


@dataclass
class StorageEntry:
    """An entry in the storage backend."""

    key: str
    value: bytes
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # TTL in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for DHT storage backends."""

    def get(self, key: str) -> Optional[StorageEntry]:
        """Get an entry by key."""
        ...

    def put(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> StorageEntry:
        """Put an entry."""
        ...

    def delete(self, key: str) -> bool:
        """Delete an entry."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Iterate over keys with optional prefix."""
        ...


# ---------------------------------------------------------------------------
# In-Memory Backend
# ---------------------------------------------------------------------------


class InMemoryBackend:
    """
    In-memory storage backend for testing.

    Thread-safe with automatic TTL expiration.
    """

    def __init__(self, cleanup_interval: float = 60.0):
        self._data: Dict[str, StorageEntry] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

        logger.info("[dht][storage][memory] initialized")

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [
            k for k, v in self._data.items()
            if v.is_expired
        ]

        for key in expired_keys:
            del self._data[key]

        if expired_keys:
            logger.debug(
                "[dht][storage][memory] cleaned %d expired entries",
                len(expired_keys),
            )

    def get(self, key: str) -> Optional[StorageEntry]:
        """Get an entry by key."""
        with self._lock:
            self._maybe_cleanup()
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._data[key]
                return None
            return entry

    def put(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StorageEntry:
        """Put an entry."""
        with self._lock:
            existing = self._data.get(key)
            version = existing.version + 1 if existing else 1

            entry = StorageEntry(
                key=key,
                value=value,
                version=version,
                timestamp=time.time(),
                ttl=ttl,
                metadata=metadata or {},
            )

            self._data[key] = entry
            return entry

    def delete(self, key: str) -> bool:
        """Delete an entry."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._data[key]
                return False
            return True

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Iterate over keys with optional prefix."""
        with self._lock:
            self._maybe_cleanup()
            for key in list(self._data.keys()):
                if key.startswith(prefix):
                    yield key

    def size(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# Redis Backend
# ---------------------------------------------------------------------------


class RedisBackend:
    """
    Redis-based storage backend.

    Uses Redis for distributed caching with TTL support.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "dht:",
        default_ttl: Optional[int] = None,
    ):
        try:
            import redis
            self._redis = redis.from_url(url)
        except ImportError:
            raise RuntimeError("redis-py is required for RedisBackend")

        self._prefix = prefix
        self._default_ttl = default_ttl

        logger.info("[dht][storage][redis] initialized url=%s", url)

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _parse_key(self, redis_key: str) -> str:
        return redis_key.removeprefix(self._prefix)

    def get(self, key: str) -> Optional[StorageEntry]:
        """Get an entry by key."""
        redis_key = self._make_key(key)
        data = self._redis.get(redis_key)

        if data is None:
            return None

        try:
            entry_dict = json.loads(data)
            return StorageEntry(
                key=key,
                value=bytes.fromhex(entry_dict["value"]),
                version=entry_dict.get("version", 1),
                timestamp=entry_dict.get("timestamp", 0),
                ttl=entry_dict.get("ttl"),
                metadata=entry_dict.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def put(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StorageEntry:
        """Put an entry."""
        redis_key = self._make_key(key)

        # Get existing version
        existing = self.get(key)
        version = existing.version + 1 if existing else 1

        entry = StorageEntry(
            key=key,
            value=value,
            version=version,
            timestamp=time.time(),
            ttl=ttl or self._default_ttl,
            metadata=metadata or {},
        )

        entry_dict = {
            "value": value.hex(),
            "version": entry.version,
            "timestamp": entry.timestamp,
            "ttl": entry.ttl,
            "metadata": entry.metadata,
        }

        if entry.ttl:
            self._redis.setex(redis_key, entry.ttl, json.dumps(entry_dict))
        else:
            self._redis.set(redis_key, json.dumps(entry_dict))

        return entry

    def delete(self, key: str) -> bool:
        """Delete an entry."""
        redis_key = self._make_key(key)
        return self._redis.delete(redis_key) > 0

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        redis_key = self._make_key(key)
        return self._redis.exists(redis_key) > 0

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Iterate over keys with optional prefix."""
        pattern = f"{self._prefix}{prefix}*"
        for key in self._redis.scan_iter(pattern):
            yield self._parse_key(key.decode() if isinstance(key, bytes) else key)

    def size(self) -> int:
        """Get approximate number of entries."""
        pattern = f"{self._prefix}*"
        count = 0
        for _ in self._redis.scan_iter(pattern):
            count += 1
        return count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(
    backend_type: str = "memory",
    **kwargs,
) -> StorageBackend:
    """Create a storage backend.

    Args:
        backend_type: Type of backend ("memory", "redis")
        **kwargs: Backend-specific configuration

    Returns:
        Storage backend instance
    """
    if backend_type == "memory":
        return InMemoryBackend(**kwargs)
    elif backend_type == "redis":
        return RedisBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
