# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Distributed Hash Table (DHT) module for AGI-HPC.

Provides decentralized key-value storage with:
- Consistent hashing with virtual nodes
- Configurable replication
- Pluggable storage backends
- Automatic load balancing

Usage:
    from agi.core.dht import DHTClient, Node

    client = DHTClient()
    client.add_node(Node("node1", "localhost", 5000))
    client.add_node(Node("node2", "localhost", 5001))

    await client.put("key", b"value")
    value = await client.get("key")
"""

from agi.core.dht.ring import HashRing, Node, VirtualNode
from agi.core.dht.storage import (
    StorageBackend,
    StorageEntry,
    InMemoryBackend,
    RedisBackend,
    create_backend,
)
from agi.core.dht.client import DHTClient, DHTConfig
from agi.core.dht.state_manager import (
    StateManager,
    StateManagerConfig,
    AgentState,
    SessionState,
)

__all__ = [
    # Client
    "DHTClient",
    "DHTConfig",
    # Ring
    "HashRing",
    "Node",
    "VirtualNode",
    # Storage
    "StorageBackend",
    "StorageEntry",
    "InMemoryBackend",
    "RedisBackend",
    "create_backend",
    # State Manager
    "StateManager",
    "StateManagerConfig",
    "AgentState",
    "SessionState",
]
