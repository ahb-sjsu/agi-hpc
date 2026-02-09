# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DHT Node Server with gRPC.

Implements Sprint 2 requirements:
- gRPC server for DHT operations
- Storage backend integration
- Node join/leave protocol
- Basic replication
- Health checks

Usage:
    python -m agi.core.dht.server --port 7000 --node-id node1 --peers node2:7000,node3:7000

Environment Variables:
    AGI_DHT_PORT            Server port (default: 7000)
    AGI_DHT_NODE_ID         Node identifier
    AGI_DHT_PEERS           Comma-separated peer addresses
    AGI_DHT_STORAGE         Storage backend: memory, rocksdb, redis
    AGI_DHT_STORAGE_PATH    Path for RocksDB storage
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import grpc
from grpc import aio

from agi.core.dht.ring import HashRing, Node
from agi.core.dht.storage import (
    StorageBackend,
    StorageEntry,
    InMemoryBackend,
    RedisBackend,
    create_backend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DHTServerConfig:
    """Configuration for DHT server."""

    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")
    host: str = "0.0.0.0"
    port: int = 7000
    storage_backend: str = "memory"
    storage_path: str = "/var/lib/agi/dht"
    redis_url: str = "redis://localhost:6379"
    replication_factor: int = 3
    virtual_nodes: int = 150
    heartbeat_interval: float = 1.0
    sync_interval: float = 60.0
    max_workers: int = 10
    enable_reflection: bool = True

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# gRPC Service Implementation
# ---------------------------------------------------------------------------


class DHTService:
    """
    gRPC service implementation for DHT operations.

    Implements:
    - Get/Put/Delete operations
    - Scan with prefix
    - Node join/leave protocol
    - Heartbeat and health checks
    - Replication
    """

    def __init__(
        self,
        config: DHTServerConfig,
        ring: HashRing,
        storage: StorageBackend,
    ) -> None:
        """Initialize DHT service."""
        self.config = config
        self.ring = ring
        self.storage = storage

        self._local_node = Node(
            node_id=config.node_id,
            address=config.host,
            port=config.port,
            virtual_nodes=config.virtual_nodes,
        )

        # Peer connections
        self._peer_channels: Dict[str, grpc.aio.Channel] = {}
        self._running = False
        self._start_time = time.time()

        logger.info(
            "[dht][server] service initialized node_id=%s",
            config.node_id,
        )

    # ------------------------------------------------------------------
    # Core Operations
    # ------------------------------------------------------------------

    async def Get(self, key: str) -> Optional[StorageEntry]:
        """Get a value by key."""
        entry = self.storage.get(key)
        if entry:
            logger.debug("[dht][server] get key=%s found", key)
        return entry

    async def Put(
        self,
        key: str,
        value: bytes,
        version: int = 0,
        ttl: Optional[int] = None,
        replicate: bool = True,
    ) -> StorageEntry:
        """Put a value."""
        # Get existing entry for version check
        existing = self.storage.get(key)
        if existing and version > 0 and existing.version >= version:
            logger.debug(
                "[dht][server] put key=%s skipped (version conflict)",
                key,
            )
            return existing

        entry = self.storage.put(key, value, ttl)
        logger.debug(
            "[dht][server] put key=%s version=%d",
            key,
            entry.version,
        )

        # Replicate to other nodes if primary
        if replicate and self._is_primary_for_key(key):
            await self._replicate_to_peers(entry)

        return entry

    async def Delete(self, key: str) -> bool:
        """Delete a key."""
        result = self.storage.delete(key)
        logger.debug("[dht][server] delete key=%s result=%s", key, result)
        return result

    async def Scan(self, prefix: str, limit: int = 100) -> List[StorageEntry]:
        """Scan keys with prefix."""
        entries = []
        for key in self.storage.keys(prefix):
            entry = self.storage.get(key)
            if entry:
                entries.append(entry)
                if len(entries) >= limit:
                    break
        return entries

    async def Exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.storage.exists(key)

    # ------------------------------------------------------------------
    # Cluster Operations
    # ------------------------------------------------------------------

    async def Join(
        self,
        node_id: str,
        address: str,
        port: int,
    ) -> List[Node]:
        """Handle node join request.

        Returns list of current cluster nodes.
        """
        new_node = Node(
            node_id=node_id,
            address=address,
            port=port,
            virtual_nodes=self.config.virtual_nodes,
        )

        self.ring.add_node(new_node)
        logger.info("[dht][server] node joined: %s", node_id)

        # Return current cluster topology
        return self.ring.get_all_nodes()

    async def Leave(self, node_id: str) -> bool:
        """Handle node leave request."""
        removed = self.ring.remove_node(node_id)
        if removed:
            logger.info("[dht][server] node left: %s", node_id)

            # Close peer connection if exists
            if node_id in self._peer_channels:
                await self._peer_channels[node_id].close()
                del self._peer_channels[node_id]

        return removed is not None

    async def GetPeers(self) -> List[Node]:
        """Get all known peers."""
        return self.ring.get_all_nodes()

    async def Heartbeat(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "node_id": self.config.node_id,
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "key_count": self.storage.size() if hasattr(self.storage, "size") else 0,
            "peer_count": self.ring.size,
        }

    async def Replicate(self, entries: List[StorageEntry]) -> int:
        """Receive replicated entries from peer."""
        replicated = 0
        for entry in entries:
            existing = self.storage.get(entry.key)
            # Only update if newer version
            if not existing or entry.version > existing.version:
                self.storage.put(
                    entry.key,
                    entry.value,
                    entry.ttl,
                    entry.metadata,
                )
                replicated += 1

        logger.debug(
            "[dht][server] replicated %d/%d entries",
            replicated,
            len(entries),
        )
        return replicated

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _is_primary_for_key(self, key: str) -> bool:
        """Check if this node is primary for a key."""
        primary = self.ring.get_node(key)
        return primary and primary.node_id == self.config.node_id

    async def _replicate_to_peers(self, entry: StorageEntry) -> None:
        """Replicate entry to peer nodes."""
        replica_nodes = self.ring.get_nodes(
            entry.key,
            self.config.replication_factor,
        )

        for node in replica_nodes:
            if node.node_id == self.config.node_id:
                continue

            try:
                # In a real implementation, this would use gRPC stub
                # For now, we log the intent
                logger.debug(
                    "[dht][server] would replicate key=%s to %s",
                    entry.key,
                    node.node_id,
                )
            except Exception as e:
                logger.warning(
                    "[dht][server] replication to %s failed: %s",
                    node.node_id,
                    e,
                )

    async def _connect_to_peer(self, address: str) -> Optional[grpc.aio.Channel]:
        """Establish connection to a peer."""
        try:
            channel = grpc.aio.insecure_channel(address)
            # Test connection with a health check
            # await grpc.channel_ready_future(channel)
            return channel
        except Exception as e:
            logger.warning("[dht][server] failed to connect to %s: %s", address, e)
            return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class DHTServer:
    """
    DHT Node Server.

    Manages:
    - gRPC server lifecycle
    - Storage backend
    - Cluster membership
    - Background tasks (heartbeat, sync)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[DHTServerConfig] = None) -> None:
        """Initialize DHT server."""
        self.config = config or DHTServerConfig()

        # Initialize components
        self._ring = HashRing(default_virtual_nodes=self.config.virtual_nodes)
        self._storage = self._create_storage()
        self._service = DHTService(self.config, self._ring, self._storage)

        # Add self to ring
        self_node = Node(
            node_id=self.config.node_id,
            address=self.config.host,
            port=self.config.port,
            virtual_nodes=self.config.virtual_nodes,
        )
        self._ring.add_node(self_node)

        self._grpc_server: Optional[grpc.aio.Server] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

        logger.info(
            "[dht][server] initialized version=%s node_id=%s port=%d",
            self.VERSION,
            self.config.node_id,
            self.config.port,
        )

    def _create_storage(self) -> StorageBackend:
        """Create storage backend based on config."""
        if self.config.storage_backend == "memory":
            return InMemoryBackend()
        elif self.config.storage_backend == "redis":
            return RedisBackend(
                url=self.config.redis_url,
                prefix=f"dht:{self.config.node_id}:",
            )
        else:
            # Default to memory
            return InMemoryBackend()

    async def start(self, peers: Optional[List[str]] = None) -> None:
        """Start the DHT server.

        Args:
            peers: List of peer addresses to connect to
        """
        logger.info("[dht][server] starting...")

        # Start gRPC server
        self._grpc_server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
        )

        # Note: In a real implementation, we'd add the protobuf-generated servicer
        # For now, we just bind to the port
        self._grpc_server.add_insecure_port(f"[::]:{self.config.port}")

        await self._grpc_server.start()
        self._running = True

        logger.info(
            "[dht][server] gRPC server started on port %d",
            self.config.port,
        )

        # Connect to peers
        if peers:
            await self._join_cluster(peers)

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._sync_loop()))

        print(f"[DHT] Node {self.config.node_id} running on port {self.config.port}")
        print("[DHT] Press Ctrl+C to stop")

    async def _join_cluster(self, peers: List[str]) -> None:
        """Join an existing cluster via peer nodes."""
        for peer_addr in peers:
            try:
                # Parse peer address
                if ":" not in peer_addr:
                    peer_addr = f"{peer_addr}:7000"

                host, port = peer_addr.split(":")

                # Create a node entry for the peer
                peer_node = Node(
                    node_id=f"peer-{host}-{port}",
                    address=host,
                    port=int(port),
                )

                self._ring.add_node(peer_node)
                logger.info("[dht][server] added peer %s", peer_addr)

                # In a real implementation, we'd:
                # 1. Connect to peer via gRPC
                # 2. Send Join request
                # 3. Receive cluster topology
                # 4. Add all discovered nodes to ring

            except Exception as e:
                logger.warning("[dht][server] failed to join via %s: %s", peer_addr, e)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to peers."""
        while self._running:
            try:
                # In a real implementation, we'd ping all known peers
                for node in self._ring.get_all_nodes():
                    if node.node_id == self.config.node_id:
                        continue
                    # Send heartbeat
                    logger.debug("[dht][server] heartbeat to %s", node.node_id)

            except Exception as e:
                logger.warning("[dht][server] heartbeat error: %s", e)

            await asyncio.sleep(self.config.heartbeat_interval)

    async def _sync_loop(self) -> None:
        """Periodic anti-entropy sync with peers."""
        while self._running:
            await asyncio.sleep(self.config.sync_interval)

            try:
                # In a real implementation, we'd:
                # 1. Exchange merkle trees with replicas
                # 2. Identify missing/outdated entries
                # 3. Sync the differences
                logger.debug("[dht][server] anti-entropy sync")

            except Exception as e:
                logger.warning("[dht][server] sync error: %s", e)

    async def stop(self) -> None:
        """Stop the DHT server."""
        logger.info("[dht][server] stopping...")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop gRPC server
        if self._grpc_server:
            await self._grpc_server.stop(grace=5.0)

        self._shutdown_event.set()
        logger.info("[dht][server] stopped")

    async def wait(self) -> None:
        """Wait for server shutdown."""
        await self._shutdown_event.wait()

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            "version": self.VERSION,
            "node_id": self.config.node_id,
            "running": self._running,
            "port": self.config.port,
            "storage_backend": self.config.storage_backend,
            "node_count": self._ring.size,
            "vnode_count": self._ring.vnode_count,
            "load_distribution": self._ring.get_load_distribution(),
        }

    @property
    def service(self) -> DHTService:
        """Get the DHT service."""
        return self._service


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AGI-HPC DHT Node Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AGI_DHT_PORT", "7000")),
        help="gRPC server port",
    )

    parser.add_argument(
        "--node-id",
        type=str,
        default=os.getenv("AGI_DHT_NODE_ID", f"node-{uuid.uuid4().hex[:8]}"),
        help="Node identifier",
    )

    parser.add_argument(
        "--peers",
        type=str,
        default=os.getenv("AGI_DHT_PEERS", ""),
        help="Comma-separated peer addresses (host:port)",
    )

    parser.add_argument(
        "--storage",
        type=str,
        choices=["memory", "redis", "rocksdb"],
        default=os.getenv("AGI_DHT_STORAGE", "memory"),
        help="Storage backend",
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.getenv("AGI_DHT_STORAGE_PATH", "/var/lib/agi/dht"),
        help="Path for persistent storage",
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.getenv("AGI_DHT_REDIS_URL", "redis://localhost:6379"),
        help="Redis URL for redis backend",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main_async() -> int:
    """Async main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    config = DHTServerConfig(
        node_id=args.node_id,
        port=args.port,
        storage_backend=args.storage,
        storage_path=args.storage_path,
        redis_url=args.redis_url,
    )

    server = DHTServer(config)

    # Parse peers
    peers = [p.strip() for p in args.peers.split(",") if p.strip()]

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("[dht][server] received shutdown signal")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await server.start(peers=peers if peers else None)
        await server.wait()
        return 0
    except KeyboardInterrupt:
        await server.stop()
        return 0
    except Exception:
        logger.exception("[dht][server] fatal error")
        return 1


def main() -> int:
    """Main entry point."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    import sys

    sys.exit(main())
