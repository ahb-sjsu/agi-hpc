# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Consistent hashing ring implementation.

Provides virtual node-based consistent hashing for:
- Distributed key-value storage
- Load balancing across nodes
- Minimal key redistribution on topology changes

Usage:
    from agi.core.dht import HashRing, Node

    ring = HashRing()
    ring.add_node(Node("node1", "192.168.1.1", 5000))
    ring.add_node(Node("node2", "192.168.1.2", 5000))

    node = ring.get_node("my-key")
    replicas = ring.get_nodes("my-key", count=3)
"""

from __future__ import annotations

import bisect
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node Types
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A physical node in the DHT cluster."""

    node_id: str
    address: str
    port: int
    weight: int = 1
    virtual_nodes: int = 150
    metadata: Dict = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False


@dataclass
class VirtualNode:
    """A virtual node on the hash ring."""

    hash_value: int
    node: Node
    vnode_id: int

    def __lt__(self, other):
        if isinstance(other, VirtualNode):
            return self.hash_value < other.hash_value
        return self.hash_value < other


# ---------------------------------------------------------------------------
# Hash Ring
# ---------------------------------------------------------------------------


class HashRing:
    """
    Consistent hashing ring with virtual nodes.

    Virtual nodes provide:
    - Better load distribution
    - Smoother rebalancing on node join/leave
    - Weight-based allocation

    The ring uses a 32-bit hash space with MD5 hashing.
    Each physical node creates multiple virtual nodes based on weight.
    """

    HASH_SPACE = 2**32  # 32-bit hash space

    def __init__(self, default_virtual_nodes: int = 150):
        """
        Initialize the hash ring.

        Args:
            default_virtual_nodes: Default number of virtual nodes per physical node
        """
        self._vnodes: List[VirtualNode] = []
        self._hash_positions: List[int] = []
        self._nodes: Dict[str, Node] = {}
        self._default_vnodes = default_virtual_nodes

        logger.info("[dht][ring] initialized vnodes=%d", default_virtual_nodes)

    def _hash_key(self, key: str) -> int:
        """Compute hash for a key."""
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    def add_node(self, node: Node) -> None:
        """Add a physical node to the ring.

        Args:
            node: Node to add
        """
        if node.node_id in self._nodes:
            logger.warning("[dht][ring] node already exists: %s", node.node_id)
            return

        self._nodes[node.node_id] = node

        # Create virtual nodes
        num_vnodes = node.virtual_nodes or self._default_vnodes
        num_vnodes = num_vnodes * node.weight

        for i in range(num_vnodes):
            vnode_key = f"{node.node_id}:{i}"
            hash_value = self._hash_key(vnode_key)

            vnode = VirtualNode(
                hash_value=hash_value,
                node=node,
                vnode_id=i,
            )

            # Insert maintaining sorted order
            idx = bisect.bisect_left(self._hash_positions, hash_value)
            self._hash_positions.insert(idx, hash_value)
            self._vnodes.insert(idx, vnode)

        logger.info(
            "[dht][ring] added node=%s vnodes=%d total=%d",
            node.node_id,
            num_vnodes,
            len(self._vnodes),
        )

    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a physical node from the ring.

        Args:
            node_id: ID of node to remove

        Returns:
            Removed node or None if not found
        """
        if node_id not in self._nodes:
            return None

        node = self._nodes.pop(node_id)

        # Remove all virtual nodes
        self._vnodes = [v for v in self._vnodes if v.node.node_id != node_id]
        self._hash_positions = [v.hash_value for v in self._vnodes]

        logger.info(
            "[dht][ring] removed node=%s remaining=%d",
            node_id,
            len(self._vnodes),
        )

        return node

    def get_node(self, key: str) -> Optional[Node]:
        """Get the node responsible for a key.

        Args:
            key: Key to look up

        Returns:
            Node responsible for the key, or None if ring is empty
        """
        if not self._vnodes:
            return None

        hash_value = self._hash_key(key)

        # Find the first node with hash >= key hash (clockwise walk)
        idx = bisect.bisect_left(self._hash_positions, hash_value)

        # Wrap around if necessary
        if idx >= len(self._vnodes):
            idx = 0

        return self._vnodes[idx].node

    def get_nodes(self, key: str, count: int = 3) -> List[Node]:
        """Get multiple nodes for a key (for replication).

        Args:
            key: Key to look up
            count: Number of distinct nodes to return

        Returns:
            List of unique nodes, up to count
        """
        if not self._vnodes:
            return []

        hash_value = self._hash_key(key)
        idx = bisect.bisect_left(self._hash_positions, hash_value)

        nodes: List[Node] = []
        seen: Set[str] = set()

        # Walk clockwise through the ring
        for i in range(len(self._vnodes)):
            vnode_idx = (idx + i) % len(self._vnodes)
            node = self._vnodes[vnode_idx].node

            if node.node_id not in seen:
                nodes.append(node)
                seen.add(node.node_id)

                if len(nodes) >= count:
                    break

        return nodes

    def get_partition(self, key: str) -> int:
        """Get partition ID for a key.

        Args:
            key: Key to partition

        Returns:
            Partition ID (hash bucket)
        """
        return self._hash_key(key)

    def get_keys_for_node(self, node_id: str) -> List[Tuple[int, int]]:
        """Get hash ranges assigned to a node.

        Args:
            node_id: Node ID

        Returns:
            List of (start, end) hash ranges
        """
        if node_id not in self._nodes:
            return []

        ranges = []
        prev_hash = 0 if not self._vnodes else self._vnodes[-1].hash_value

        for vnode in self._vnodes:
            if vnode.node.node_id == node_id:
                # This vnode owns the range (prev_hash, vnode.hash_value]
                ranges.append((prev_hash, vnode.hash_value))
            prev_hash = vnode.hash_value

        return ranges

    def get_all_nodes(self) -> List[Node]:
        """Get all physical nodes in the ring."""
        return list(self._nodes.values())

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        return self._nodes.get(node_id)

    @property
    def size(self) -> int:
        """Number of physical nodes in the ring."""
        return len(self._nodes)

    @property
    def vnode_count(self) -> int:
        """Number of virtual nodes in the ring."""
        return len(self._vnodes)

    def get_load_distribution(self) -> Dict[str, float]:
        """Get load distribution across nodes.

        Returns:
            Dict mapping node_id to percentage of ring owned
        """
        if not self._vnodes:
            return {}

        # Count vnodes per physical node
        vnode_counts: Dict[str, int] = {}
        for vnode in self._vnodes:
            node_id = vnode.node.node_id
            vnode_counts[node_id] = vnode_counts.get(node_id, 0) + 1

        # Convert to percentages
        total = len(self._vnodes)
        return {node_id: count / total * 100 for node_id, count in vnode_counts.items()}
