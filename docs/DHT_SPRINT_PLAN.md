# DHT (Distributed Hash Table) Sprint Plan

## Overview

The Distributed Hash Table (DHT) provides decentralized key-value storage for the AGI-HPC cognitive architecture, enabling horizontal scaling across HPC clusters without central coordination bottlenecks.

## Current State Assessment

### Implemented (Scaffolding)
| Component | Status | Location |
|-----------|--------|----------|
| `dht/` directory | **Exists** | `src/agi/core/dht/` |
| Basic DHT interface | **Stub** | Placeholder implementation |

### Key Gaps
1. **No consistent hashing** - Missing ring-based node assignment
2. **No replication** - Single point of failure per key
3. **No virtual nodes** - Uneven load distribution
4. **No node discovery** - Static cluster configuration
5. **No failure detection** - No heartbeat/gossip protocol
6. **No data migration** - Manual rebalancing required
7. **No persistence** - In-memory only

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DISTRIBUTED HASH TABLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         DHT RING                                     │   │
│   │                                                                      │   │
│   │                         ┌──────┐                                     │   │
│   │                    ┌────┤Node A├────┐                               │   │
│   │                    │    └──────┘    │                               │   │
│   │               ┌────┴─┐           ┌──┴────┐                          │   │
│   │               │Node F│           │Node B │                          │   │
│   │               └────┬─┘           └──┬────┘                          │   │
│   │                    │                │                               │   │
│   │               ┌────┴─┐           ┌──┴────┐                          │   │
│   │               │Node E│           │Node C │                          │   │
│   │               └────┬─┘           └──┬────┘                          │   │
│   │                    │    ┌──────┐    │                               │   │
│   │                    └────┤Node D├────┘                               │   │
│   │                         └──────┘                                     │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────┴────────────────────────────────────┐   │
│   │                       DHT CLIENT API                                 │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │     get()    │  │    put()     │  │   delete()   │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │    scan()    │  │  replicate() │  │  migrate()   │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────┴────────────────────────────────────┐   │
│   │                         STORAGE BACKENDS                             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   In-Memory  │  │   RocksDB    │  │    Redis     │             │   │
│   │   │   (testing)  │  │ (persistent) │  │   (cache)    │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases in AGI-HPC

| Use Case | Key Pattern | Value Type | Access Pattern |
|----------|-------------|------------|----------------|
| Agent State | `agent:{agent_id}` | AgentState protobuf | Read-heavy |
| Plan Cache | `plan:{plan_id}` | PlanGraph protobuf | Write-once, read-many |
| Skill Index | `skill:{skill_hash}` | SkillDefinition | Read-heavy |
| Memory Index | `mem:{memory_type}:{key}` | Memory entry ref | Mixed |
| Session State | `session:{session_id}` | Session protobuf | Write-heavy |
| Lock Service | `lock:{resource_id}` | Lock holder info | High contention |

---

## Sprint 1: Core DHT Infrastructure

**Goal**: Implement consistent hashing with virtual nodes and basic CRUD operations.

### Tasks

#### 1.1 Consistent Hashing Ring

```python
# src/agi/core/dht/ring.py
"""Consistent hashing ring implementation."""

from __future__ import annotations

import bisect
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the DHT cluster."""

    node_id: str
    address: str
    port: int
    weight: int = 1
    virtual_nodes: int = 150
    metadata: dict = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"


@dataclass
class VirtualNode:
    """A virtual node on the hash ring."""

    hash_value: int
    node: Node
    vnode_id: int


class HashRing:
    """Consistent hashing ring with virtual nodes.

    Virtual nodes provide:
    - Better load distribution
    - Smoother rebalancing on node join/leave
    - Weight-based allocation
    """

    HASH_SPACE = 2**32  # 32-bit hash space

    def __init__(self, default_vnodes: int = 150) -> None:
        """Initialize empty hash ring."""
        self.default_vnodes = default_vnodes
        self._nodes: dict[str, Node] = {}
        self._ring: List[Tuple[int, VirtualNode]] = []
        self._ring_keys: List[int] = []  # For binary search

    def add_node(self, node: Node) -> None:
        """Add a node to the ring."""
        if node.node_id in self._nodes:
            self.remove_node(node.node_id)

        self._nodes[node.node_id] = node

        # Add virtual nodes
        vnodes = node.virtual_nodes * node.weight
        for i in range(vnodes):
            vnode_key = f"{node.node_id}:vnode:{i}"
            hash_val = self._hash(vnode_key)
            vnode = VirtualNode(hash_value=hash_val, node=node, vnode_id=i)

            # Insert maintaining sorted order
            idx = bisect.bisect_left(self._ring_keys, hash_val)
            self._ring_keys.insert(idx, hash_val)
            self._ring.insert(idx, (hash_val, vnode))

        logger.info(
            "Added node %s with %d virtual nodes",
            node.node_id,
            vnodes,
        )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring."""
        if node_id not in self._nodes:
            return

        node = self._nodes.pop(node_id)

        # Remove all virtual nodes
        self._ring = [
            (h, vn) for h, vn in self._ring
            if vn.node.node_id != node_id
        ]
        self._ring_keys = [h for h, _ in self._ring]

        logger.info("Removed node %s", node_id)

    def get_node(self, key: str) -> Optional[Node]:
        """Get the node responsible for a key."""
        if not self._ring:
            return None

        hash_val = self._hash(key)
        idx = bisect.bisect_left(self._ring_keys, hash_val)

        # Wrap around to first node if past end
        if idx == len(self._ring_keys):
            idx = 0

        return self._ring[idx][1].node

    def get_nodes(self, key: str, count: int = 3) -> List[Node]:
        """Get multiple nodes for a key (for replication)."""
        if not self._ring:
            return []

        hash_val = self._hash(key)
        idx = bisect.bisect_left(self._ring_keys, hash_val)

        seen_nodes = set()
        result = []

        # Walk ring collecting unique physical nodes
        ring_len = len(self._ring)
        for i in range(ring_len):
            actual_idx = (idx + i) % ring_len
            node = self._ring[actual_idx][1].node

            if node.node_id not in seen_nodes:
                seen_nodes.add(node.node_id)
                result.append(node)

                if len(result) >= count:
                    break

        return result

    def get_partition(self, node_id: str) -> List[Tuple[int, int]]:
        """Get hash ranges owned by a node."""
        if node_id not in self._nodes or not self._ring:
            return []

        ranges = []
        prev_hash = self._ring_keys[-1]  # Last key wraps to first

        for hash_val, vnode in self._ring:
            if vnode.node.node_id == node_id:
                ranges.append((prev_hash, hash_val))
            prev_hash = hash_val

        return ranges

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        digest = hashlib.md5(key.encode()).digest()
        return int.from_bytes(digest[:4], byteorder='big')

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def vnode_count(self) -> int:
        return len(self._ring)
```

#### 1.2 DHT Client Interface

```python
# src/agi/core/dht/client.py
"""DHT client for distributed key-value operations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from agi.core.dht.ring import HashRing, Node

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Read/write consistency levels."""
    ONE = 1       # Any single replica
    QUORUM = 2    # Majority of replicas
    ALL = 3       # All replicas


@dataclass
class DHTConfig:
    """DHT configuration."""

    replication_factor: int = 3
    read_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    write_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    timeout_ms: int = 5000
    retry_count: int = 3
    virtual_nodes: int = 150


@dataclass
class DHTEntry:
    """A DHT entry with metadata."""

    key: str
    value: bytes
    version: int = 0
    ttl_seconds: int = 0
    created_at: float = 0
    updated_at: float = 0


class DHTClient:
    """Client for distributed hash table operations."""

    def __init__(
        self,
        config: DHTConfig = None,
        ring: HashRing = None,
    ) -> None:
        """Initialize DHT client."""
        self.config = config or DHTConfig()
        self.ring = ring or HashRing(
            default_vnodes=self.config.virtual_nodes
        )
        self._node_clients: dict[str, NodeClient] = {}

    async def connect(self, bootstrap_nodes: List[str]) -> None:
        """Connect to DHT cluster via bootstrap nodes."""
        for addr in bootstrap_nodes:
            try:
                host, port = addr.split(":")
                node = Node(
                    node_id=addr,
                    address=host,
                    port=int(port),
                )
                self.ring.add_node(node)
                await self._connect_to_node(node)

                # Discover cluster topology from bootstrap node
                await self._discover_cluster(node)

            except Exception as e:
                logger.warning("Failed to connect to %s: %s", addr, e)

        if self.ring.node_count == 0:
            raise ConnectionError("Could not connect to any bootstrap nodes")

        logger.info(
            "Connected to DHT cluster with %d nodes",
            self.ring.node_count,
        )

    async def get(
        self,
        key: str,
        consistency: ConsistencyLevel = None,
    ) -> Optional[bytes]:
        """Get a value from the DHT."""
        consistency = consistency or self.config.read_consistency
        nodes = self.ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            raise RuntimeError("No nodes available")

        required = self._required_responses(consistency, len(nodes))

        # Query nodes in parallel
        tasks = [
            self._get_from_node(node, key)
            for node in nodes
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful responses
        values = [
            r for r in responses
            if isinstance(r, DHTEntry)
        ]

        if len(values) < required:
            raise RuntimeError(
                f"Consistency not met: got {len(values)}, need {required}"
            )

        # Return most recent version
        if values:
            values.sort(key=lambda e: e.version, reverse=True)
            return values[0].value

        return None

    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int = 0,
        consistency: ConsistencyLevel = None,
    ) -> bool:
        """Put a value in the DHT."""
        consistency = consistency or self.config.write_consistency
        nodes = self.ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            raise RuntimeError("No nodes available")

        required = self._required_responses(consistency, len(nodes))

        # Get current version for optimistic locking
        current = await self.get(key, ConsistencyLevel.ONE)
        version = 1
        if current:
            version += 1  # Increment version

        entry = DHTEntry(
            key=key,
            value=value,
            version=version,
            ttl_seconds=ttl_seconds,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Write to nodes in parallel
        tasks = [
            self._put_to_node(node, entry)
            for node in nodes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if r is True)

        if successes < required:
            raise RuntimeError(
                f"Write consistency not met: {successes}/{required}"
            )

        return True

    async def delete(
        self,
        key: str,
        consistency: ConsistencyLevel = None,
    ) -> bool:
        """Delete a key from the DHT."""
        consistency = consistency or self.config.write_consistency
        nodes = self.ring.get_nodes(key, self.config.replication_factor)

        if not nodes:
            raise RuntimeError("No nodes available")

        required = self._required_responses(consistency, len(nodes))

        tasks = [
            self._delete_from_node(node, key)
            for node in nodes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if r is True)

        return successes >= required

    async def scan(
        self,
        prefix: str,
        limit: int = 100,
    ) -> List[DHTEntry]:
        """Scan keys with a prefix (distributed query)."""
        # Broadcast to all nodes
        all_entries = []

        for node_id, client in self._node_clients.items():
            try:
                entries = await client.scan(prefix, limit)
                all_entries.extend(entries)
            except Exception as e:
                logger.warning("Scan failed on %s: %s", node_id, e)

        # Deduplicate by key, keeping highest version
        by_key = {}
        for entry in all_entries:
            if entry.key not in by_key or entry.version > by_key[entry.key].version:
                by_key[entry.key] = entry

        result = list(by_key.values())[:limit]
        return result

    def _required_responses(
        self,
        consistency: ConsistencyLevel,
        total_nodes: int,
    ) -> int:
        """Calculate required responses for consistency level."""
        if consistency == ConsistencyLevel.ONE:
            return 1
        elif consistency == ConsistencyLevel.QUORUM:
            return (total_nodes // 2) + 1
        else:  # ALL
            return total_nodes

    async def _connect_to_node(self, node: Node) -> None:
        """Establish connection to a node."""
        client = NodeClient(node)
        await client.connect()
        self._node_clients[node.node_id] = client

    async def _discover_cluster(self, bootstrap: Node) -> None:
        """Discover cluster topology from a node."""
        client = self._node_clients.get(bootstrap.node_id)
        if not client:
            return

        peers = await client.get_peers()
        for peer in peers:
            if peer.node_id not in self._node_clients:
                self.ring.add_node(peer)
                await self._connect_to_node(peer)

    async def _get_from_node(
        self,
        node: Node,
        key: str,
    ) -> Optional[DHTEntry]:
        """Get value from a specific node."""
        client = self._node_clients.get(node.node_id)
        if not client:
            raise RuntimeError(f"No connection to {node.node_id}")

        return await client.get(key)

    async def _put_to_node(
        self,
        node: Node,
        entry: DHTEntry,
    ) -> bool:
        """Put value to a specific node."""
        client = self._node_clients.get(node.node_id)
        if not client:
            raise RuntimeError(f"No connection to {node.node_id}")

        return await client.put(entry)

    async def _delete_from_node(
        self,
        node: Node,
        key: str,
    ) -> bool:
        """Delete key from a specific node."""
        client = self._node_clients.get(node.node_id)
        if not client:
            raise RuntimeError(f"No connection to {node.node_id}")

        return await client.delete(key)

    async def close(self) -> None:
        """Close all connections."""
        for client in self._node_clients.values():
            await client.close()
        self._node_clients.clear()
```

#### 1.3 Configuration

```yaml
# configs/dht_config.yaml
dht:
  cluster:
    name: "agi-hpc-dht"
    bootstrap_nodes:
      - "dht-node-1:7000"
      - "dht-node-2:7000"
      - "dht-node-3:7000"

  replication:
    factor: 3
    read_consistency: "quorum"
    write_consistency: "quorum"

  ring:
    virtual_nodes: 150
    hash_algorithm: "md5"

  storage:
    backend: "rocksdb"  # memory, rocksdb, redis
    path: "/var/lib/agi/dht"
    cache_size_mb: 256

  network:
    port: 7000
    timeout_ms: 5000
    retry_count: 3
    connection_pool_size: 10

  health:
    heartbeat_interval_ms: 1000
    failure_threshold: 5
    recovery_timeout_ms: 30000
```

### Acceptance Criteria
```bash
# Start 3-node DHT cluster
python -m agi.core.dht.server --port 7000 --peers dht-2:7000,dht-3:7000
python -m agi.core.dht.server --port 7000 --peers dht-1:7000,dht-3:7000
python -m agi.core.dht.server --port 7000 --peers dht-1:7000,dht-2:7000

# Test client operations
python -c "
import asyncio
from agi.core.dht.client import DHTClient

async def test():
    client = DHTClient()
    await client.connect(['localhost:7000'])

    await client.put('test:key', b'hello world')
    value = await client.get('test:key')
    print(f'Got: {value}')

    await client.close()

asyncio.run(test())
"
```

---

## Sprint 2: Node Server Implementation

**Goal**: Implement the DHT node server with storage and replication.

### Tasks

#### 2.1 DHT Node Server

```python
# src/agi/core/dht/server.py
"""DHT node server implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import grpc

from agi.core.dht.ring import HashRing, Node
from agi.core.dht.storage import StorageBackend, RocksDBBackend
from proto.dht_pb2 import (
    GetRequest, GetResponse,
    PutRequest, PutResponse,
    DeleteRequest, DeleteResponse,
    ScanRequest, ScanResponse,
    JoinRequest, JoinResponse,
    HeartbeatRequest, HeartbeatResponse,
    ReplicateRequest, ReplicateResponse,
)
from proto.dht_pb2_grpc import (
    DHTServiceServicer,
    add_DHTServiceServicer_to_server,
)

logger = logging.getLogger(__name__)


@dataclass
class DHTServerConfig:
    """DHT server configuration."""

    node_id: str
    address: str
    port: int
    storage_path: str = "/var/lib/agi/dht"
    replication_factor: int = 3
    virtual_nodes: int = 150


class DHTServer(DHTServiceServicer):
    """DHT node server."""

    def __init__(self, config: DHTServerConfig) -> None:
        """Initialize DHT server."""
        self.config = config
        self.node = Node(
            node_id=config.node_id,
            address=config.address,
            port=config.port,
            virtual_nodes=config.virtual_nodes,
        )

        self.ring = HashRing(default_vnodes=config.virtual_nodes)
        self.ring.add_node(self.node)

        self.storage = RocksDBBackend(config.storage_path)

        self._peer_clients: Dict[str, DHTClient] = {}
        self._running = False

    async def start(self, peers: List[str] = None) -> None:
        """Start the DHT server."""
        await self.storage.open()

        # Connect to peers
        if peers:
            await self._join_cluster(peers)

        # Start gRPC server
        self._server = grpc.aio.server()
        add_DHTServiceServicer_to_server(self, self._server)
        self._server.add_insecure_port(f"[::]:{self.config.port}")

        await self._server.start()
        self._running = True

        logger.info(
            "DHT node %s started on port %d",
            self.config.node_id,
            self.config.port,
        )

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._anti_entropy_loop())

    async def Get(
        self,
        request: GetRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetResponse:
        """Handle get request."""
        entry = await self.storage.get(request.key)

        if entry:
            return GetResponse(
                found=True,
                key=entry.key,
                value=entry.value,
                version=entry.version,
            )

        return GetResponse(found=False)

    async def Put(
        self,
        request: PutRequest,
        context: grpc.aio.ServicerContext,
    ) -> PutResponse:
        """Handle put request."""
        entry = DHTEntry(
            key=request.key,
            value=request.value,
            version=request.version,
            ttl_seconds=request.ttl_seconds,
        )

        success = await self.storage.put(entry)

        # Replicate to peers if primary
        if request.replicate and success:
            await self._replicate_to_peers(entry)

        return PutResponse(success=success, version=entry.version)

    async def Delete(
        self,
        request: DeleteRequest,
        context: grpc.aio.ServicerContext,
    ) -> DeleteResponse:
        """Handle delete request."""
        success = await self.storage.delete(request.key)
        return DeleteResponse(success=success)

    async def Scan(
        self,
        request: ScanRequest,
        context: grpc.aio.ServicerContext,
    ) -> ScanResponse:
        """Handle scan request."""
        entries = await self.storage.scan(request.prefix, request.limit)

        return ScanResponse(
            entries=[
                ScanResponse.Entry(
                    key=e.key,
                    value=e.value,
                    version=e.version,
                )
                for e in entries
            ]
        )

    async def Join(
        self,
        request: JoinRequest,
        context: grpc.aio.ServicerContext,
    ) -> JoinResponse:
        """Handle node join request."""
        new_node = Node(
            node_id=request.node_id,
            address=request.address,
            port=request.port,
        )

        self.ring.add_node(new_node)

        # Return current cluster topology
        peers = [
            JoinResponse.Peer(
                node_id=n.node_id,
                address=n.address,
                port=n.port,
            )
            for n in self.ring._nodes.values()
        ]

        logger.info("Node %s joined cluster", new_node.node_id)

        return JoinResponse(success=True, peers=peers)

    async def Heartbeat(
        self,
        request: HeartbeatRequest,
        context: grpc.aio.ServicerContext,
    ) -> HeartbeatResponse:
        """Handle heartbeat request."""
        return HeartbeatResponse(
            node_id=self.config.node_id,
            timestamp=time.time(),
            status="healthy",
            key_count=await self.storage.count(),
        )

    async def Replicate(
        self,
        request: ReplicateRequest,
        context: grpc.aio.ServicerContext,
    ) -> ReplicateResponse:
        """Handle replication request."""
        for entry in request.entries:
            existing = await self.storage.get(entry.key)

            # Only update if newer version
            if not existing or entry.version > existing.version:
                await self.storage.put(DHTEntry(
                    key=entry.key,
                    value=entry.value,
                    version=entry.version,
                ))

        return ReplicateResponse(
            success=True,
            replicated_count=len(request.entries),
        )

    async def _join_cluster(self, peers: List[str]) -> None:
        """Join an existing cluster."""
        for peer_addr in peers:
            try:
                channel = grpc.aio.insecure_channel(peer_addr)
                stub = DHTServiceStub(channel)

                response = await stub.Join(JoinRequest(
                    node_id=self.config.node_id,
                    address=self.config.address,
                    port=self.config.port,
                ))

                if response.success:
                    # Add all peers to our ring
                    for peer in response.peers:
                        node = Node(
                            node_id=peer.node_id,
                            address=peer.address,
                            port=peer.port,
                        )
                        self.ring.add_node(node)

                    logger.info("Joined cluster via %s", peer_addr)
                    return

            except Exception as e:
                logger.warning("Failed to join via %s: %s", peer_addr, e)

        logger.warning("Could not join existing cluster, starting new")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to peers."""
        while self._running:
            for node_id, node in list(self.ring._nodes.items()):
                if node_id == self.config.node_id:
                    continue

                try:
                    channel = grpc.aio.insecure_channel(node.endpoint)
                    stub = DHTServiceStub(channel)

                    await stub.Heartbeat(HeartbeatRequest(
                        node_id=self.config.node_id,
                        timestamp=time.time(),
                    ), timeout=1.0)

                except Exception:
                    # Mark node as potentially failed
                    pass

            await asyncio.sleep(1.0)

    async def _anti_entropy_loop(self) -> None:
        """Periodic anti-entropy for consistency repair."""
        while self._running:
            await asyncio.sleep(60.0)  # Every minute

            # Compare merkle trees with replicas
            # Sync missing or outdated entries
            pass

    async def _replicate_to_peers(self, entry: DHTEntry) -> None:
        """Replicate entry to peer nodes."""
        replica_nodes = self.ring.get_nodes(
            entry.key,
            self.config.replication_factor,
        )

        for node in replica_nodes:
            if node.node_id == self.config.node_id:
                continue

            try:
                channel = grpc.aio.insecure_channel(node.endpoint)
                stub = DHTServiceStub(channel)

                await stub.Replicate(ReplicateRequest(
                    entries=[ReplicateRequest.Entry(
                        key=entry.key,
                        value=entry.value,
                        version=entry.version,
                    )]
                ))

            except Exception as e:
                logger.warning(
                    "Replication to %s failed: %s",
                    node.node_id, e,
                )

    async def stop(self) -> None:
        """Stop the DHT server."""
        self._running = False
        await self._server.stop(5.0)
        await self.storage.close()
```

#### 2.2 Storage Backend

```python
# src/agi/core/dht/storage.py
"""DHT storage backends."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


@dataclass
class DHTEntry:
    """A DHT storage entry."""

    key: str
    value: bytes
    version: int = 1
    ttl_seconds: int = 0
    created_at: float = 0
    updated_at: float = 0

    def __post_init__(self):
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    async def open(self) -> None:
        """Open the storage."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage."""
        ...

    @abstractmethod
    async def get(self, key: str) -> Optional[DHTEntry]:
        """Get an entry."""
        ...

    @abstractmethod
    async def put(self, entry: DHTEntry) -> bool:
        """Put an entry."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an entry."""
        ...

    @abstractmethod
    async def scan(self, prefix: str, limit: int) -> List[DHTEntry]:
        """Scan entries with prefix."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total entries."""
        ...


class MemoryBackend(StorageBackend):
    """In-memory storage backend for testing."""

    def __init__(self) -> None:
        self._data: Dict[str, DHTEntry] = {}

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        self._data.clear()

    async def get(self, key: str) -> Optional[DHTEntry]:
        entry = self._data.get(key)

        # Check TTL
        if entry and entry.ttl_seconds > 0:
            if time.time() > entry.created_at + entry.ttl_seconds:
                del self._data[key]
                return None

        return entry

    async def put(self, entry: DHTEntry) -> bool:
        self._data[entry.key] = entry
        return True

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def scan(self, prefix: str, limit: int) -> List[DHTEntry]:
        result = []
        for key, entry in self._data.items():
            if key.startswith(prefix):
                result.append(entry)
                if len(result) >= limit:
                    break
        return result

    async def count(self) -> int:
        return len(self._data)


class RocksDBBackend(StorageBackend):
    """RocksDB storage backend for persistence."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._db = None

    async def open(self) -> None:
        import rocksdb

        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 300
        opts.write_buffer_size = 64 * 1024 * 1024
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 64 * 1024 * 1024

        self._db = rocksdb.DB(self.path, opts)
        logger.info("Opened RocksDB at %s", self.path)

    async def close(self) -> None:
        if self._db:
            del self._db
            self._db = None

    async def get(self, key: str) -> Optional[DHTEntry]:
        value = self._db.get(key.encode())
        if value:
            return self._deserialize(value)
        return None

    async def put(self, entry: DHTEntry) -> bool:
        self._db.put(entry.key.encode(), self._serialize(entry))
        return True

    async def delete(self, key: str) -> bool:
        self._db.delete(key.encode())
        return True

    async def scan(self, prefix: str, limit: int) -> List[DHTEntry]:
        result = []
        it = self._db.iterkeys()
        it.seek(prefix.encode())

        for key in it:
            if not key.decode().startswith(prefix):
                break

            value = self._db.get(key)
            if value:
                result.append(self._deserialize(value))

            if len(result) >= limit:
                break

        return result

    async def count(self) -> int:
        count = 0
        it = self._db.iterkeys()
        it.seek_to_first()
        for _ in it:
            count += 1
        return count

    def _serialize(self, entry: DHTEntry) -> bytes:
        import json
        return json.dumps({
            "key": entry.key,
            "value": entry.value.hex(),
            "version": entry.version,
            "ttl_seconds": entry.ttl_seconds,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }).encode()

    def _deserialize(self, data: bytes) -> DHTEntry:
        import json
        d = json.loads(data.decode())
        return DHTEntry(
            key=d["key"],
            value=bytes.fromhex(d["value"]),
            version=d["version"],
            ttl_seconds=d["ttl_seconds"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
```

### Deliverables
- [ ] DHT node gRPC server
- [ ] Memory storage backend
- [ ] RocksDB storage backend
- [ ] Node join/leave protocol
- [ ] Basic replication

---

## Sprint 3: Failure Detection and Recovery

**Goal**: Implement gossip-based failure detection and automatic recovery.

### Tasks

#### 3.1 Gossip Protocol

```python
# src/agi/core/dht/gossip.py
"""Gossip-based failure detection."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node health status."""
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class NodeState:
    """State of a node as seen by gossip."""

    node_id: str
    status: NodeStatus = NodeStatus.ALIVE
    incarnation: int = 0
    last_heartbeat: float = 0
    suspect_time: float = 0
    suspicions: Set[str] = field(default_factory=set)


class GossipProtocol:
    """SWIM-style gossip protocol for failure detection.

    Features:
    - Randomized peer selection
    - Suspicion mechanism (indirect probing)
    - Incarnation numbers for state ordering
    """

    def __init__(
        self,
        node_id: str,
        gossip_interval: float = 1.0,
        suspect_timeout: float = 5.0,
        dead_timeout: float = 30.0,
        indirect_probes: int = 3,
    ) -> None:
        """Initialize gossip protocol."""
        self.node_id = node_id
        self.gossip_interval = gossip_interval
        self.suspect_timeout = suspect_timeout
        self.dead_timeout = dead_timeout
        self.indirect_probes = indirect_probes

        self.incarnation = 0
        self._nodes: Dict[str, NodeState] = {}
        self._callbacks: List[callable] = []
        self._running = False

    def add_node(self, node_id: str) -> None:
        """Add a node to track."""
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeState(
                node_id=node_id,
                last_heartbeat=time.time(),
            )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from tracking."""
        self._nodes.pop(node_id, None)

    def on_status_change(self, callback: callable) -> None:
        """Register callback for status changes."""
        self._callbacks.append(callback)

    async def start(self, ping_fn: callable) -> None:
        """Start the gossip protocol."""
        self._ping_fn = ping_fn
        self._running = True
        asyncio.create_task(self._gossip_loop())

    async def stop(self) -> None:
        """Stop the gossip protocol."""
        self._running = False

    def receive_heartbeat(self, node_id: str, incarnation: int = 0) -> None:
        """Process received heartbeat."""
        if node_id not in self._nodes:
            self.add_node(node_id)

        state = self._nodes[node_id]

        # Update if incarnation is newer or same and not alive
        if incarnation > state.incarnation or (
            incarnation == state.incarnation and
            state.status != NodeStatus.ALIVE
        ):
            old_status = state.status
            state.status = NodeStatus.ALIVE
            state.incarnation = incarnation
            state.last_heartbeat = time.time()
            state.suspicions.clear()

            if old_status != NodeStatus.ALIVE:
                self._notify_status_change(node_id, NodeStatus.ALIVE)

    def receive_suspect(
        self,
        node_id: str,
        from_node: str,
        incarnation: int,
    ) -> None:
        """Process suspicion about a node."""
        if node_id == self.node_id:
            # We're being suspected, increment incarnation
            self.incarnation = max(self.incarnation, incarnation) + 1
            return

        if node_id not in self._nodes:
            return

        state = self._nodes[node_id]

        if incarnation >= state.incarnation:
            state.suspicions.add(from_node)

            if state.status == NodeStatus.ALIVE:
                state.status = NodeStatus.SUSPECT
                state.suspect_time = time.time()
                self._notify_status_change(node_id, NodeStatus.SUSPECT)

    async def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self._running:
            await self._do_gossip_round()
            await self._check_timeouts()
            await asyncio.sleep(self.gossip_interval)

    async def _do_gossip_round(self) -> None:
        """Perform one round of gossip."""
        if not self._nodes:
            return

        # Select random peer
        peers = list(self._nodes.keys())
        if not peers:
            return

        target = random.choice(peers)

        # Ping the target
        success = await self._ping_with_timeout(target)

        if success:
            self.receive_heartbeat(target)
        else:
            # Try indirect probing
            success = await self._indirect_ping(target)

            if not success:
                # Mark as suspect
                self.receive_suspect(target, self.node_id, self._nodes[target].incarnation)

    async def _ping_with_timeout(self, node_id: str, timeout: float = 1.0) -> bool:
        """Ping a node with timeout."""
        try:
            return await asyncio.wait_for(
                self._ping_fn(node_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def _indirect_ping(self, target: str) -> bool:
        """Try pinging target through other nodes."""
        peers = [p for p in self._nodes.keys() if p != target]

        if not peers:
            return False

        # Select random peers for indirect probing
        probe_peers = random.sample(
            peers,
            min(self.indirect_probes, len(peers)),
        )

        # Ask each peer to ping the target
        tasks = [
            self._request_indirect_ping(peer, target)
            for peer in probe_peers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(r is True for r in results)

    async def _request_indirect_ping(self, peer: str, target: str) -> bool:
        """Request a peer to ping the target."""
        # Implementation depends on network layer
        return False

    async def _check_timeouts(self) -> None:
        """Check for timed-out suspects and dead nodes."""
        now = time.time()

        for node_id, state in list(self._nodes.items()):
            if state.status == NodeStatus.SUSPECT:
                if now - state.suspect_time > self.dead_timeout:
                    state.status = NodeStatus.DEAD
                    self._notify_status_change(node_id, NodeStatus.DEAD)

            elif state.status == NodeStatus.ALIVE:
                if now - state.last_heartbeat > self.suspect_timeout:
                    state.status = NodeStatus.SUSPECT
                    state.suspect_time = now
                    self._notify_status_change(node_id, NodeStatus.SUSPECT)

    def _notify_status_change(self, node_id: str, status: NodeStatus) -> None:
        """Notify callbacks of status change."""
        logger.info("Node %s status changed to %s", node_id, status.value)

        for callback in self._callbacks:
            try:
                callback(node_id, status)
            except Exception as e:
                logger.error("Status change callback error: %s", e)
```

#### 3.2 Data Migration

```python
# src/agi/core/dht/migration.py
"""Data migration for node join/leave."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Tuple

from agi.core.dht.ring import HashRing, Node
from agi.core.dht.storage import StorageBackend, DHTEntry

logger = logging.getLogger(__name__)


@dataclass
class MigrationPlan:
    """Plan for data migration."""

    source_node: Node
    target_node: Node
    key_ranges: List[Tuple[int, int]]
    estimated_keys: int


class DataMigrator:
    """Handles data migration during cluster changes."""

    def __init__(
        self,
        ring: HashRing,
        storage: StorageBackend,
        batch_size: int = 1000,
    ) -> None:
        """Initialize migrator."""
        self.ring = ring
        self.storage = storage
        self.batch_size = batch_size

    async def on_node_join(self, new_node: Node) -> None:
        """Handle new node joining cluster."""
        # Calculate which keys should move to new node
        new_ranges = self.ring.get_partition(new_node.node_id)

        if not new_ranges:
            return

        logger.info(
            "Migrating data to new node %s (%d ranges)",
            new_node.node_id,
            len(new_ranges),
        )

        # Scan and migrate keys in ranges
        migrated = 0

        for start_hash, end_hash in new_ranges:
            entries = await self._scan_range(start_hash, end_hash)

            if entries:
                await self._migrate_entries(new_node, entries)
                migrated += len(entries)

        logger.info(
            "Migration complete: %d keys to %s",
            migrated,
            new_node.node_id,
        )

    async def on_node_leave(self, leaving_node: Node) -> None:
        """Handle node leaving cluster."""
        # Find replica nodes that should take over
        # Trigger anti-entropy to ensure consistency
        pass

    async def _scan_range(
        self,
        start_hash: int,
        end_hash: int,
    ) -> List[DHTEntry]:
        """Scan all keys in a hash range."""
        # This is simplified - real implementation would
        # need efficient range scanning
        all_entries = []

        # Scan all keys and filter by hash
        entries = await self.storage.scan("", limit=100000)

        for entry in entries:
            key_hash = self.ring._hash(entry.key)

            if start_hash < end_hash:
                if start_hash < key_hash <= end_hash:
                    all_entries.append(entry)
            else:
                # Wraps around ring
                if key_hash > start_hash or key_hash <= end_hash:
                    all_entries.append(entry)

        return all_entries

    async def _migrate_entries(
        self,
        target: Node,
        entries: List[DHTEntry],
    ) -> None:
        """Migrate entries to target node."""
        # Batch the migration
        for i in range(0, len(entries), self.batch_size):
            batch = entries[i:i + self.batch_size]

            # Send batch to target
            channel = grpc.aio.insecure_channel(target.endpoint)
            stub = DHTServiceStub(channel)

            await stub.Replicate(ReplicateRequest(
                entries=[
                    ReplicateRequest.Entry(
                        key=e.key,
                        value=e.value,
                        version=e.version,
                    )
                    for e in batch
                ]
            ))

            logger.debug(
                "Migrated batch of %d entries to %s",
                len(batch),
                target.node_id,
            )
```

### Deliverables
- [ ] SWIM gossip protocol
- [ ] Failure detection with configurable timeouts
- [ ] Node status change callbacks
- [ ] Data migration on node join
- [ ] Data rebalancing on node leave

---

## Sprint 4: Unit Tests

**Goal**: Achieve 80%+ test coverage for DHT module.

### Tasks

#### 4.1 Hash Ring Tests
- [ ] `test_add_node_creates_virtual_nodes`
- [ ] `test_remove_node_cleans_ring`
- [ ] `test_get_node_consistent`
- [ ] `test_get_nodes_returns_replicas`
- [ ] `test_partition_calculation`
- [ ] `test_load_distribution`

#### 4.2 Client Tests
- [ ] `test_put_get_basic`
- [ ] `test_delete`
- [ ] `test_consistency_levels`
- [ ] `test_scan_prefix`
- [ ] `test_ttl_expiry`
- [ ] `test_version_conflict`

#### 4.3 Server Tests
- [ ] `test_server_startup`
- [ ] `test_join_cluster`
- [ ] `test_replication`
- [ ] `test_heartbeat`

#### 4.4 Gossip Tests
- [ ] `test_failure_detection`
- [ ] `test_suspect_timeout`
- [ ] `test_dead_timeout`
- [ ] `test_incarnation_ordering`

---

## Sprint 5: Integration Testing

**Goal**: Verify DHT works with other AGI-HPC components.

### Tasks

#### 5.1 Multi-node cluster tests
- [ ] 3-node cluster operations
- [ ] Node failure and recovery
- [ ] Network partition handling
- [ ] Concurrent writes

#### 5.2 Integration with cognitive services
- [ ] LH plan caching
- [ ] Memory index storage
- [ ] Session state management
- [ ] Skill registry

#### 5.3 Docker Compose stack
```yaml
# docker-compose.dht.yaml
version: '3.8'

services:
  dht-node-1:
    build:
      context: .
      dockerfile: docker/Dockerfile.dht
    environment:
      - DHT_NODE_ID=node-1
      - DHT_PORT=7000
      - DHT_PEERS=dht-node-2:7000,dht-node-3:7000
    ports:
      - "7001:7000"
    volumes:
      - dht-data-1:/var/lib/agi/dht

  dht-node-2:
    build:
      context: .
      dockerfile: docker/Dockerfile.dht
    environment:
      - DHT_NODE_ID=node-2
      - DHT_PORT=7000
      - DHT_PEERS=dht-node-1:7000,dht-node-3:7000
    ports:
      - "7002:7000"
    volumes:
      - dht-data-2:/var/lib/agi/dht

  dht-node-3:
    build:
      context: .
      dockerfile: docker/Dockerfile.dht
    environment:
      - DHT_NODE_ID=node-3
      - DHT_PORT=7000
      - DHT_PEERS=dht-node-1:7000,dht-node-2:7000
    ports:
      - "7003:7000"
    volumes:
      - dht-data-3:/var/lib/agi/dht

volumes:
  dht-data-1:
  dht-data-2:
  dht-data-3:
```

---

## Sprint 6: Production Hardening

**Goal**: Prepare DHT for HPC deployment.

### Tasks

#### 6.1 Observability
- [ ] Prometheus metrics (ops/sec, latency, replication lag)
- [ ] Structured logging with correlation IDs
- [ ] Distributed tracing
- [ ] Grafana dashboards

#### 6.2 HPC optimizations
- [ ] UCX transport for RDMA
- [ ] Shared memory for co-located nodes
- [ ] NUMA-aware allocation
- [ ] Batch operations

#### 6.3 Security
- [ ] mTLS for node communication
- [ ] Access control
- [ ] Encryption at rest
- [ ] Audit logging

---

## File Structure After Completion

```
src/agi/core/dht/
├── __init__.py
├── ring.py              # Consistent hash ring
├── client.py            # DHT client
├── server.py            # DHT node server
├── gossip.py            # Failure detection
├── migration.py         # Data migration
├── storage/
│   ├── __init__.py
│   ├── base.py          # StorageBackend protocol
│   ├── memory.py        # In-memory backend
│   ├── rocksdb.py       # RocksDB backend
│   └── redis.py         # Redis backend
├── proto/
│   └── dht.proto        # gRPC definitions
└── config.py            # Configuration

tests/core/dht/
├── __init__.py
├── conftest.py          # Fixtures
├── test_ring.py
├── test_client.py
├── test_server.py
├── test_gossip.py
├── test_migration.py
└── integration/
    ├── test_cluster.py
    └── test_failures.py

configs/
└── dht_config.yaml
```

---

## Priority Order

1. **Sprint 1** - Critical: Core DHT infrastructure
2. **Sprint 2** - Critical: Node server for cluster operation
3. **Sprint 3** - High: Failure detection for reliability
4. **Sprint 4** - High: Tests for correctness
5. **Sprint 5** - Medium: Integration verification
6. **Sprint 6** - Low (for now): Production concerns

---

## Dependencies

```toml
# pyproject.toml additions for DHT
[project.optional-dependencies]
dht = [
    "grpcio>=1.60.0",
    "grpcio-tools>=1.60.0",
    "python-rocksdb>=0.7.0",
    "redis>=5.0",
    "xxhash>=3.4.0",
]

dht-hpc = [
    "agi-hpc[dht]",
    "ucx-py>=0.35",
]
```
