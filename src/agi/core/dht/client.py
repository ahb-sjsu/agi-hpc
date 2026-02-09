# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DHT Client for distributed key-value operations.

Provides a high-level API for DHT operations:
- get/put/delete with automatic routing
- Replication across multiple nodes
- Consistent hashing for load balancing

Usage:
    from agi.core.dht import DHTClient, Node

    client = DHTClient()
    client.add_node(Node("node1", "localhost", 5000))

    await client.put("key", b"value")
    value = await client.get("key")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agi.core.dht.ring import HashRing, Node
from agi.core.dht.storage import (
    StorageBackend,
    StorageEntry,
    InMemoryBackend,
    create_backend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DHTConfig:
    """Configuration for DHT client."""

    replication_factor: int = 3
    read_quorum: int = 2
    write_quorum: int = 2
    virtual_nodes: int = 150
    default_ttl: Optional[int] = None
    backend_type: str = "memory"
    backend_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DHT Client
# ---------------------------------------------------------------------------


class DHTClient:
    """
    Distributed Hash Table client.

    Provides consistent hashing with replication for distributed
    key-value storage across AGI-HPC cluster nodes.
    """

    def __init__(self, config: Optional[DHTConfig] = None):
        """
        Initialize DHT client.

        Args:
            config: DHT configuration
        """
        self.config = config or DHTConfig()

        self._ring = HashRing(default_virtual_nodes=self.config.virtual_nodes)

        # Local storage for this node (each node has its own)
        self._local_storage = create_backend(
            self.config.backend_type,
            **self.config.backend_config,
        )

        # In a real distributed system, this would track remote node connections
        self._node_backends: Dict[str, StorageBackend] = {}

        logger.info(
            "[dht] client initialized replication=%d",
            self.config.replication_factor,
        )

    def add_node(self, node: Node) -> None:
        """Add a node to the DHT ring.

        Args:
            node: Node to add
        """
        self._ring.add_node(node)

        # In a distributed system, we'd establish connection to the node
        # For now, we create a local storage backend for each node
        self._node_backends[node.node_id] = create_backend(
            self.config.backend_type,
            **self.config.backend_config,
        )

    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node from the DHT ring.

        Args:
            node_id: ID of node to remove

        Returns:
            Removed node or None
        """
        node = self._ring.remove_node(node_id)
        if node and node_id in self._node_backends:
            del self._node_backends[node_id]
        return node

    def _get_backend_for_node(self, node: Node) -> StorageBackend:
        """Get storage backend for a node."""
        return self._node_backends.get(node.node_id, self._local_storage)

    async def put(
        self,
        key: str,
        value: Union[bytes, str],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a value in the DHT.

        Args:
            key: Key to store
            value: Value to store (bytes or string)
            ttl: Time-to-live in seconds
            metadata: Optional metadata

        Returns:
            True if stored successfully on quorum nodes
        """
        if isinstance(value, str):
            value = value.encode("utf-8")

        ttl = ttl or self.config.default_ttl

        # Get nodes responsible for this key
        nodes = self._ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            # No nodes, store locally
            self._local_storage.put(key, value, ttl, metadata)
            return True

        # Write to all replica nodes
        success_count = 0
        for node in nodes:
            backend = self._get_backend_for_node(node)
            try:
                backend.put(key, value, ttl, metadata)
                success_count += 1
            except Exception as e:
                logger.warning(
                    "[dht] put failed on node %s: %s",
                    node.node_id,
                    e,
                )

        success = success_count >= self.config.write_quorum
        logger.debug(
            "[dht] put key=%s success=%d/%d quorum=%d",
            key,
            success_count,
            len(nodes),
            self.config.write_quorum,
        )

        return success

    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve a value from the DHT.

        Args:
            key: Key to retrieve

        Returns:
            Value bytes or None if not found
        """
        nodes = self._ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            # No nodes, check local storage
            entry = self._local_storage.get(key)
            return entry.value if entry else None

        # Read from replica nodes until quorum
        entries: List[StorageEntry] = []

        for node in nodes:
            backend = self._get_backend_for_node(node)
            try:
                entry = backend.get(key)
                if entry:
                    entries.append(entry)
                    if len(entries) >= self.config.read_quorum:
                        break
            except Exception as e:
                logger.warning(
                    "[dht] get failed on node %s: %s",
                    node.node_id,
                    e,
                )

        if not entries:
            return None

        # Return most recent version
        latest = max(entries, key=lambda e: (e.version, e.timestamp))
        return latest.value

    async def delete(self, key: str) -> bool:
        """Delete a key from the DHT.

        Args:
            key: Key to delete

        Returns:
            True if deleted from quorum nodes
        """
        nodes = self._ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            return self._local_storage.delete(key)

        success_count = 0
        for node in nodes:
            backend = self._get_backend_for_node(node)
            try:
                if backend.delete(key):
                    success_count += 1
            except Exception as e:
                logger.warning(
                    "[dht] delete failed on node %s: %s",
                    node.node_id,
                    e,
                )

        return success_count >= self.config.write_quorum

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the DHT.

        Args:
            key: Key to check

        Returns:
            True if key exists
        """
        nodes = self._ring.get_nodes(key, 1)

        if not nodes:
            return self._local_storage.exists(key)

        for node in nodes:
            backend = self._get_backend_for_node(node)
            try:
                if backend.exists(key):
                    return True
            except Exception:
                pass

        return False

    async def get_with_metadata(self, key: str) -> Optional[StorageEntry]:
        """Get value with full metadata.

        Args:
            key: Key to retrieve

        Returns:
            Storage entry with metadata or None
        """
        nodes = self._ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            return self._local_storage.get(key)

        for node in nodes:
            backend = self._get_backend_for_node(node)
            try:
                entry = backend.get(key)
                if entry:
                    return entry
            except Exception:
                pass

        return None

    def get_responsible_nodes(self, key: str) -> List[Node]:
        """Get nodes responsible for a key.

        Args:
            key: Key to check

        Returns:
            List of responsible nodes
        """
        return self._ring.get_nodes(key, self.config.replication_factor)

    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across nodes."""
        return self._ring.get_load_distribution()

    @property
    def node_count(self) -> int:
        """Number of nodes in the ring."""
        return self._ring.size

    @property
    def nodes(self) -> List[Node]:
        """All nodes in the ring."""
        return self._ring.get_all_nodes()

    def close(self) -> None:
        """Close the client and clean up resources."""
        logger.info("[dht] client closed")
