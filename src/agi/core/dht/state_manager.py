# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DHT-based State Manager for AGI-HPC.

Provides distributed state management:
- Agent state storage and retrieval
- Caching for frequently accessed data
- Session state across cluster nodes
- Atomic state updates

Usage:
    from agi.core.dht import StateManager

    manager = StateManager()

    # Store agent state
    await manager.set_agent_state("agent-1", {
        "position": [1.0, 2.0, 3.0],
        "goal": "navigate to target",
    })

    # Retrieve agent state
    state = await manager.get_agent_state("agent-1")

    # Cache computation results
    await manager.cache("query:123", result, ttl=300)
    cached = await manager.get_cached("query:123")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from agi.core.dht.client import DHTClient, DHTConfig
from agi.core.dht.ring import Node

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StateManagerConfig:
    """Configuration for the state manager."""

    # DHT settings
    replication_factor: int = 3
    read_quorum: int = 2
    write_quorum: int = 2

    # Cache settings
    default_cache_ttl: int = 300  # 5 minutes
    max_cache_ttl: int = 3600  # 1 hour

    # State settings
    state_ttl: Optional[int] = None  # No expiry by default
    state_prefix: str = "state:"
    cache_prefix: str = "cache:"
    session_prefix: str = "session:"

    # Serialization
    serializer: str = "json"  # json, msgpack, pickle


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class Serializer:
    """Base serializer for state data."""

    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data to bytes."""
        return json.dumps(data, default=str).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize bytes to data."""
        return json.loads(data.decode("utf-8"))


class MsgPackSerializer(Serializer):
    """MessagePack serializer (more efficient for binary data)."""

    @staticmethod
    def serialize(data: Any) -> bytes:
        try:
            import msgpack

            return msgpack.packb(data, use_bin_type=True)
        except ImportError:
            return Serializer.serialize(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        try:
            import msgpack

            return msgpack.unpackb(data, raw=False)
        except ImportError:
            return Serializer.deserialize(data)


def get_serializer(name: str) -> Serializer:
    """Get serializer by name."""
    if name == "msgpack":
        return MsgPackSerializer()
    return Serializer()


# ---------------------------------------------------------------------------
# State Types
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """State of an agent in the system."""

    agent_id: str
    data: Dict[str, Any]
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "data": self.data,
            "version": self.version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        return cls(
            agent_id=d.get("agent_id", ""),
            data=d.get("data", {}),
            version=d.get("version", 1),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SessionState:
    """State for a user/system session."""

    session_id: str
    user_id: str
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data": self.data,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(
            session_id=d.get("session_id", ""),
            user_id=d.get("user_id", ""),
            data=d.get("data", {}),
            created_at=d.get("created_at", time.time()),
            expires_at=d.get("expires_at"),
        )


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------


class StateManager:
    """
    DHT-based state manager for distributed state.

    Provides:
    - Agent state storage across cluster
    - Caching with TTL
    - Session management
    - Atomic state updates
    """

    def __init__(
        self,
        config: Optional[StateManagerConfig] = None,
        dht_client: Optional[DHTClient] = None,
    ):
        self._config = config or StateManagerConfig()
        self._serializer = get_serializer(self._config.serializer)

        # Create or use provided DHT client
        if dht_client is None:
            dht_config = DHTConfig(
                replication_factor=self._config.replication_factor,
                read_quorum=self._config.read_quorum,
                write_quorum=self._config.write_quorum,
            )
            dht_client = DHTClient(dht_config)

        self._dht = dht_client

        logger.info(
            "[dht][state] manager initialized replication=%d",
            self._config.replication_factor,
        )

    # ------------------------------------------------------------------ #
    # Node Management
    # ------------------------------------------------------------------ #

    def add_node(self, node: Node) -> None:
        """Add a node to the DHT cluster."""
        self._dht.add_node(node)

    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node from the cluster."""
        return self._dht.remove_node(node_id)

    # ------------------------------------------------------------------ #
    # Agent State
    # ------------------------------------------------------------------ #

    async def set_agent_state(
        self,
        agent_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store agent state.

        Args:
            agent_id: Agent identifier
            data: State data
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        key = f"{self._config.state_prefix}agent:{agent_id}"

        # Get existing state for version
        existing = await self.get_agent_state(agent_id)
        version = (existing.version + 1) if existing else 1

        state = AgentState(
            agent_id=agent_id,
            data=data,
            version=version,
            metadata=metadata or {},
        )

        value = self._serializer.serialize(state.to_dict())

        success = await self._dht.put(
            key,
            value,
            ttl=self._config.state_ttl,
            metadata={"type": "agent_state"},
        )

        if success:
            logger.debug(
                "[dht][state] set agent state agent=%s version=%d",
                agent_id,
                version,
            )

        return success

    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Retrieve agent state.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentState or None if not found
        """
        key = f"{self._config.state_prefix}agent:{agent_id}"
        value = await self._dht.get(key)

        if value is None:
            return None

        data = self._serializer.deserialize(value)
        return AgentState.from_dict(data)

    async def delete_agent_state(self, agent_id: str) -> bool:
        """Delete agent state.

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted
        """
        key = f"{self._config.state_prefix}agent:{agent_id}"
        return await self._dht.delete(key)

    async def update_agent_state(
        self,
        agent_id: str,
        updates: Dict[str, Any],
    ) -> Optional[AgentState]:
        """Update agent state (merge with existing).

        Args:
            agent_id: Agent identifier
            updates: Fields to update

        Returns:
            Updated AgentState or None if not found
        """
        existing = await self.get_agent_state(agent_id)
        if existing is None:
            return None

        # Merge updates
        existing.data.update(updates)
        existing.version += 1
        existing.timestamp = time.time()

        await self.set_agent_state(
            agent_id,
            existing.data,
            existing.metadata,
        )

        return existing

    # ------------------------------------------------------------------ #
    # Caching
    # ------------------------------------------------------------------ #

    async def cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        cache_key = f"{self._config.cache_prefix}{key}"
        ttl = min(
            ttl or self._config.default_cache_ttl,
            self._config.max_cache_ttl,
        )

        serialized = self._serializer.serialize(value)

        success = await self._dht.put(
            cache_key,
            serialized,
            ttl=ttl,
            metadata={"type": "cache"},
        )

        if success:
            logger.debug("[dht][cache] stored key=%s ttl=%d", key, ttl)

        return success

    async def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = f"{self._config.cache_prefix}{key}"
        value = await self._dht.get(cache_key)

        if value is None:
            return None

        return self._serializer.deserialize(value)

    async def invalidate_cache(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if invalidated
        """
        cache_key = f"{self._config.cache_prefix}{key}"
        return await self._dht.delete(cache_key)

    async def cache_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """Get from cache or compute and cache.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Cache TTL

        Returns:
            Cached or computed value
        """
        cached = await self.get_cached(key)
        if cached is not None:
            return cached

        value = compute_fn()
        await self.cache(key, value, ttl)
        return value

    # ------------------------------------------------------------------ #
    # Session Management
    # ------------------------------------------------------------------ #

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        data: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
    ) -> SessionState:
        """Create a new session.

        Args:
            session_id: Session identifier
            user_id: User identifier
            data: Session data
            expires_in: Expiry time in seconds

        Returns:
            Created SessionState
        """
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            data=data or {},
            expires_at=expires_at,
        )

        key = f"{self._config.session_prefix}{session_id}"
        value = self._serializer.serialize(session.to_dict())

        await self._dht.put(
            key,
            value,
            ttl=expires_in,
            metadata={"type": "session"},
        )

        logger.info(
            "[dht][session] created session=%s user=%s",
            session_id,
            user_id,
        )

        return session

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state.

        Args:
            session_id: Session identifier

        Returns:
            SessionState or None
        """
        key = f"{self._config.session_prefix}{session_id}"
        value = await self._dht.get(key)

        if value is None:
            return None

        session = SessionState.from_dict(self._serializer.deserialize(value))

        if session.is_expired():
            await self.end_session(session_id)
            return None

        return session

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> Optional[SessionState]:
        """Update session data.

        Args:
            session_id: Session identifier
            updates: Fields to update

        Returns:
            Updated SessionState or None
        """
        session = await self.get_session(session_id)
        if session is None:
            return None

        session.data.update(updates)

        key = f"{self._config.session_prefix}{session_id}"
        ttl = None
        if session.expires_at:
            ttl = int(session.expires_at - time.time())
            if ttl <= 0:
                return None

        value = self._serializer.serialize(session.to_dict())
        await self._dht.put(key, value, ttl=ttl)

        return session

    async def end_session(self, session_id: str) -> bool:
        """End a session.

        Args:
            session_id: Session identifier

        Returns:
            True if ended
        """
        key = f"{self._config.session_prefix}{session_id}"
        success = await self._dht.delete(key)

        if success:
            logger.info("[dht][session] ended session=%s", session_id)

        return success

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across nodes."""
        return self._dht.get_load_distribution()

    def close(self) -> None:
        """Close the state manager."""
        self._dht.close()
        logger.info("[dht][state] manager closed")
