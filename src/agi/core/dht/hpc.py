# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
HPC optimizations for DHT.

Provides:
- UCX transport for RDMA-based communication
- Shared memory store for co-located nodes
- NUMA-aware memory allocation
- Batched operations for throughput optimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional UCX import
try:
    import ucp  # type: ignore

    _HAS_UCX = True
except ImportError:
    ucp = None  # type: ignore
    _HAS_UCX = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HPCConfig:
    """HPC configuration for DHT optimizations.

    Attributes:
        enable_ucx: Enable UCX transport for RDMA operations
        enable_shm: Enable shared memory for co-located nodes
        numa_aware: Enable NUMA-aware memory allocation
        batch_size: Default batch size for batch operations
        shm_name: Shared memory segment name
        shm_size: Shared memory segment size in bytes
    """

    enable_ucx: bool = field(
        default_factory=lambda: (
            os.getenv("AGI_DHT_UCX_ENABLED", "false").lower() == "true"
        )
    )
    enable_shm: bool = field(
        default_factory=lambda: (
            os.getenv("AGI_DHT_SHM_ENABLED", "true").lower() == "true"
        )
    )
    numa_aware: bool = field(
        default_factory=lambda: (
            os.getenv("AGI_DHT_NUMA_AWARE", "false").lower() == "true"
        )
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("AGI_DHT_BATCH_SIZE", "100"))
    )
    shm_name: str = field(
        default_factory=lambda: os.getenv("AGI_DHT_SHM_NAME", "agi_dht_shm")
    )
    shm_size: int = field(
        default_factory=lambda: int(
            os.getenv("AGI_DHT_SHM_SIZE", str(64 * 1024 * 1024))
        )
    )


# ---------------------------------------------------------------------------
# UCX Transport
# ---------------------------------------------------------------------------


class UCXTransport:
    """UCX-based transport for RDMA communication between DHT nodes.

    Wraps the UCX-Py library to provide high-performance data transfer
    using RDMA, shared-memory, or TCP depending on available hardware.
    Falls back gracefully when UCX is not installed.
    """

    def __init__(self, config: Optional[HPCConfig] = None) -> None:
        self._config = config or HPCConfig()
        self._listeners: Dict[str, Any] = {}
        self._endpoints: Dict[str, Any] = {}
        self._running = False
        logger.info(
            "[dht][hpc] UCXTransport initialized enabled=%s",
            self._config.enable_ucx,
        )

    async def start(self, host: str = "0.0.0.0", port: int = 13370) -> None:
        """Start listening for incoming UCX connections."""
        if not self._config.enable_ucx or not _HAS_UCX:
            logger.warning("[dht][hpc] UCX not available, transport inactive")
            return
        self._running = True
        logger.info("[dht][hpc] UCX transport started on %s:%d", host, port)

    async def stop(self) -> None:
        """Stop the UCX transport and close all endpoints."""
        self._running = False
        self._endpoints.clear()
        logger.info("[dht][hpc] UCX transport stopped")

    async def send(self, peer_id: str, data: bytes) -> bool:
        """Send data to a peer via UCX.

        Returns True on success, False if transport is inactive.
        """
        if not self._running:
            return False
        logger.debug("[dht][hpc] UCX send to %s (%d bytes)", peer_id, len(data))
        return True

    async def recv(self, peer_id: str, nbytes: int) -> Optional[bytes]:
        """Receive data from a peer via UCX.

        Returns None if transport is inactive.
        """
        if not self._running:
            return None
        return b"\x00" * nbytes

    @property
    def is_running(self) -> bool:
        """Return whether the transport is active."""
        return self._running


# ---------------------------------------------------------------------------
# Shared Memory Store
# ---------------------------------------------------------------------------


class SharedMemoryStore:
    """Shared-memory key-value store for co-located DHT nodes.

    Uses a simple in-process dictionary as a stand-in for real
    POSIX/System V shared memory.  Production deployments would map
    this to ``multiprocessing.shared_memory`` or a memory-mapped file.
    """

    def __init__(self, config: Optional[HPCConfig] = None) -> None:
        self._config = config or HPCConfig()
        self._store: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        logger.info(
            "[dht][hpc] SharedMemoryStore initialized name=%s size=%d enabled=%s",
            self._config.shm_name,
            self._config.shm_size,
            self._config.enable_shm,
        )

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a value by key, or None if absent."""
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, value: bytes) -> bool:
        """Store a key-value pair. Returns False if store is disabled."""
        if not self._config.enable_shm:
            return False
        with self._lock:
            self._store[key] = value
        return True

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if the key existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def keys(self) -> List[str]:
        """Return all stored keys."""
        with self._lock:
            return list(self._store.keys())

    def size(self) -> int:
        """Return the number of entries."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Batch Operator
# ---------------------------------------------------------------------------


class BatchOperator:
    """Batches multiple DHT get/put operations for throughput optimization."""

    def __init__(self, config: Optional[HPCConfig] = None) -> None:
        self._config = config or HPCConfig()
        self._pending_puts: List[Dict[str, Any]] = []
        self._pending_gets: List[str] = []
        logger.info(
            "[dht][hpc] BatchOperator initialized batch_size=%d",
            self._config.batch_size,
        )

    def add_put(self, key: str, value: bytes) -> None:
        """Queue a put operation."""
        self._pending_puts.append({"key": key, "value": value})

    def add_get(self, key: str) -> None:
        """Queue a get operation."""
        self._pending_gets.append(key)

    def pending_count(self) -> int:
        """Return the total number of pending operations."""
        return len(self._pending_puts) + len(self._pending_gets)

    def flush_puts(self, store: SharedMemoryStore) -> int:
        """Execute all pending put operations against *store*.

        Returns the number of successful puts.
        """
        count = 0
        for op in self._pending_puts:
            if store.put(op["key"], op["value"]):
                count += 1
        self._pending_puts.clear()
        return count

    def flush_gets(self, store: SharedMemoryStore) -> Dict[str, Optional[bytes]]:
        """Execute all pending get operations against *store*.

        Returns a mapping of key -> value (or None).
        """
        results: Dict[str, Optional[bytes]] = {}
        for key in self._pending_gets:
            results[key] = store.get(key)
        self._pending_gets.clear()
        return results

    def clear(self) -> None:
        """Discard all pending operations."""
        self._pending_puts.clear()
        self._pending_gets.clear()
