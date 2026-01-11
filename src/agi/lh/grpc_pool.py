# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
gRPC Connection Pool for the Left Hemisphere.

Provides:
- Connection pooling for gRPC channels
- Health checking and automatic reconnection
- Channel lifecycle management
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar, Callable, Any
from contextlib import contextmanager

import grpc

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Stub type


# ---------------------------------------------------------------------------
# Channel Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """Configuration for a gRPC channel."""

    address: str
    options: list = field(default_factory=list)
    credentials: Optional[grpc.ChannelCredentials] = None
    connect_timeout: float = 5.0
    idle_timeout: float = 300.0  # Close idle channels after 5 minutes
    health_check_interval: float = 30.0


@dataclass
class PoolConfig:
    """Configuration for the connection pool."""

    min_channels: int = 1
    max_channels: int = 10
    channel_config: ChannelConfig = field(default_factory=lambda: ChannelConfig(""))


# ---------------------------------------------------------------------------
# Channel Wrapper
# ---------------------------------------------------------------------------


class ManagedChannel:
    """
    Wrapper around a gRPC channel with lifecycle management.

    Tracks usage, health, and provides automatic reconnection.
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self._channel: Optional[grpc.Channel] = None
        self._created_at: float = 0
        self._last_used: float = 0
        self._use_count: int = 0
        self._is_healthy: bool = False
        self._lock = threading.Lock()

    @property
    def channel(self) -> grpc.Channel:
        """Get the underlying gRPC channel, creating if necessary."""
        with self._lock:
            if self._channel is None:
                self._connect()
            self._last_used = time.monotonic()
            self._use_count += 1
            return self._channel  # type: ignore

    @property
    def is_connected(self) -> bool:
        """Check if channel is connected."""
        return self._channel is not None

    @property
    def is_idle(self) -> bool:
        """Check if channel has been idle too long."""
        if not self._channel:
            return False
        return (time.monotonic() - self._last_used) > self.config.idle_timeout

    @property
    def is_healthy(self) -> bool:
        """Check if channel is healthy."""
        return self._is_healthy

    def _connect(self) -> None:
        """Create the gRPC channel."""
        logger.info("[gRPC] Connecting to %s", self.config.address)

        if self.config.credentials:
            self._channel = grpc.secure_channel(
                self.config.address,
                self.config.credentials,
                options=self.config.options,
            )
        else:
            self._channel = grpc.insecure_channel(
                self.config.address,
                options=self.config.options,
            )

        self._created_at = time.monotonic()
        self._last_used = self._created_at
        self._is_healthy = True

        logger.info("[gRPC] Connected to %s", self.config.address)

    def check_health(self) -> bool:
        """
        Check channel health by testing connectivity.

        Returns:
            True if channel is healthy
        """
        if not self._channel:
            return False

        try:
            # Try to get channel state
            state = self._channel._channel.check_connectivity_state(True)
            self._is_healthy = state in (
                grpc.ChannelConnectivity.READY,
                grpc.ChannelConnectivity.IDLE,
            )
            return self._is_healthy
        except Exception as e:
            logger.warning(
                "[gRPC] Health check failed for %s: %s", self.config.address, e
            )
            self._is_healthy = False
            return False

    def close(self) -> None:
        """Close the channel."""
        with self._lock:
            if self._channel:
                try:
                    self._channel.close()
                except Exception as e:
                    logger.warning("[gRPC] Error closing channel: %s", e)
                finally:
                    self._channel = None
                    self._is_healthy = False
                    logger.info("[gRPC] Closed channel to %s", self.config.address)


# ---------------------------------------------------------------------------
# Stub Factory
# ---------------------------------------------------------------------------


class StubFactory(Generic[T]):
    """
    Factory for creating gRPC stubs with connection pooling.

    Usage:
        factory = StubFactory(
            SafetyServiceStub,
            ChannelConfig(address="safety:50200"),
        )

        stub = factory.get_stub()
        response = stub.CheckPlan(request)
    """

    def __init__(
        self,
        stub_class: type,
        config: ChannelConfig,
    ):
        self.stub_class = stub_class
        self.config = config
        self._channel = ManagedChannel(config)
        self._stub: Optional[T] = None
        self._lock = threading.Lock()

    def get_stub(self) -> T:
        """Get a stub instance, creating channel if necessary."""
        with self._lock:
            if self._stub is None or not self._channel.is_connected:
                self._stub = self.stub_class(self._channel.channel)
            return self._stub

    def is_healthy(self) -> bool:
        """Check if the underlying channel is healthy."""
        return self._channel.check_health()

    def close(self) -> None:
        """Close the underlying channel."""
        self._channel.close()
        self._stub = None


# ---------------------------------------------------------------------------
# Connection Pool
# ---------------------------------------------------------------------------


class ChannelPool:
    """
    Pool of gRPC channels for load balancing and resilience.

    Maintains multiple channels to a service and distributes requests.

    Usage:
        pool = ChannelPool(
            address="safety:50200",
            min_channels=2,
            max_channels=5,
        )

        with pool.get_channel() as channel:
            stub = SafetyServiceStub(channel)
            response = stub.CheckPlan(request)
    """

    def __init__(
        self,
        address: str,
        min_channels: int = 1,
        max_channels: int = 10,
        **channel_options: Any,
    ):
        self.address = address
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.channel_options = channel_options

        self._channels: list[ManagedChannel] = []
        self._current_index = 0
        self._lock = threading.Lock()
        self._closed = False

        # Create minimum channels
        config = ChannelConfig(address=address, **channel_options)
        for _ in range(min_channels):
            self._channels.append(ManagedChannel(config))

    @contextmanager
    def get_channel(self):
        """
        Get a channel from the pool.

        Uses round-robin distribution across healthy channels.
        """
        if self._closed:
            raise RuntimeError("Channel pool is closed")

        with self._lock:
            # Find a healthy channel using round-robin
            for _ in range(len(self._channels)):
                channel = self._channels[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._channels)

                if channel.is_healthy or not channel.is_connected:
                    try:
                        yield channel.channel
                        return
                    except grpc.RpcError:
                        # Mark channel as unhealthy on RPC errors
                        channel._is_healthy = False
                        raise

            # All channels unhealthy, try creating a new one
            if len(self._channels) < self.max_channels:
                config = ChannelConfig(address=self.address, **self.channel_options)
                new_channel = ManagedChannel(config)
                self._channels.append(new_channel)
                yield new_channel.channel
                return

            # No healthy channels and can't create more
            raise RuntimeError(f"No healthy channels available for {self.address}")

    def get_stub(self, stub_class: type) -> Any:
        """
        Get a stub using a channel from the pool.

        Note: The stub is tied to a specific channel. For load balancing,
        use get_channel() context manager instead.
        """
        with self._lock:
            if not self._channels:
                raise RuntimeError("Channel pool is empty")

            channel = self._channels[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._channels)
            return stub_class(channel.channel)

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all channels in the pool.

        Returns:
            Dict mapping channel index to health status
        """
        results = {}
        with self._lock:
            for i, channel in enumerate(self._channels):
                results[f"channel_{i}"] = channel.check_health()
        return results

    def cleanup_idle(self) -> int:
        """
        Close idle channels above the minimum.

        Returns:
            Number of channels closed
        """
        closed = 0
        with self._lock:
            # Keep at least min_channels
            while len(self._channels) > self.min_channels:
                # Find an idle channel
                idle_idx = None
                for i, channel in enumerate(self._channels):
                    if channel.is_idle:
                        idle_idx = i
                        break

                if idle_idx is None:
                    break

                channel = self._channels.pop(idle_idx)
                channel.close()
                closed += 1

                # Adjust current index if necessary
                if self._current_index >= len(self._channels):
                    self._current_index = 0

        if closed > 0:
            logger.info("[gRPC] Closed %d idle channels for %s", closed, self.address)

        return closed

    def close(self) -> None:
        """Close all channels in the pool."""
        self._closed = True
        with self._lock:
            for channel in self._channels:
                channel.close()
            self._channels.clear()
            logger.info("[gRPC] Closed channel pool for %s", self.address)


# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------


class ServiceRegistry:
    """
    Registry of gRPC service connections.

    Provides centralized management of all downstream service connections.

    Usage:
        registry = ServiceRegistry()
        registry.register("safety", SafetyServiceStub, "safety:50200")
        registry.register("memory", MemoryServiceStub, "memory:50110")

        safety_stub = registry.get_stub("safety")
        memory_stub = registry.get_stub("memory")
    """

    def __init__(self):
        self._factories: Dict[str, StubFactory] = {}
        self._pools: Dict[str, ChannelPool] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        stub_class: type,
        address: str,
        use_pool: bool = False,
        **options: Any,
    ) -> None:
        """
        Register a service.

        Args:
            name: Service name for lookup
            stub_class: gRPC stub class
            address: Service address (host:port)
            use_pool: Use channel pool for load balancing
            **options: Additional channel options
        """
        with self._lock:
            if use_pool:
                pool = ChannelPool(address, **options)
                self._pools[name] = pool
                logger.info(
                    "[ServiceRegistry] Registered pool: %s -> %s", name, address
                )
            else:
                config = ChannelConfig(address=address)
                factory = StubFactory(stub_class, config)
                self._factories[name] = factory
                logger.info("[ServiceRegistry] Registered: %s -> %s", name, address)

    def get_stub(self, name: str) -> Any:
        """Get a stub for the named service."""
        with self._lock:
            if name in self._factories:
                return self._factories[name].get_stub()
            if name in self._pools:
                return self._pools[name].get_stub(
                    # Need to store stub class for pools
                    type(None)  # This is a placeholder
                )
            raise KeyError(f"Unknown service: {name}")

    def get_factory(self, name: str) -> StubFactory:
        """Get the stub factory for the named service."""
        with self._lock:
            if name not in self._factories:
                raise KeyError(f"Unknown service: {name}")
            return self._factories[name]

    def get_pool(self, name: str) -> ChannelPool:
        """Get the channel pool for the named service."""
        with self._lock:
            if name not in self._pools:
                raise KeyError(f"Unknown pool: {name}")
            return self._pools[name]

    def health_check(self) -> Dict[str, bool]:
        """Check health of all registered services."""
        results = {}
        with self._lock:
            for name, factory in self._factories.items():
                results[name] = factory.is_healthy()
            for name, pool in self._pools.items():
                pool_health = pool.health_check()
                results[name] = any(pool_health.values())
        return results

    def close(self) -> None:
        """Close all service connections."""
        with self._lock:
            for factory in self._factories.values():
                factory.close()
            for pool in self._pools.values():
                pool.close()
            self._factories.clear()
            self._pools.clear()
            logger.info("[ServiceRegistry] Closed all connections")
