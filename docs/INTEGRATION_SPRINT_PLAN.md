# Integration Sprint Plan

## Overview

The Integration Sprint Plan defines how all AGI-HPC subsystems connect to form a cohesive cognitive architecture. This plan focuses on inter-service communication, data flow orchestration, end-to-end testing, and system-wide deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              AGI-HPC INTEGRATED SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                            EVENT FABRIC                                      │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │    │
│  │   │  ZMQ    │  │  Redis  │  │  NATS   │  │   UCX   │  │ Event Router    │   │    │
│  │   │ Broker  │  │ Streams │  │JetStream│  │ Backend │  │ & Subscription  │   │    │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                              │
│       ┌───────────────────────────────┼───────────────────────────────┐              │
│       │                               │                               │              │
│       ▼                               ▼                               ▼              │
│  ┌─────────────┐               ┌─────────────┐               ┌─────────────┐        │
│  │     LH      │◄─────────────►│   MEMORY    │◄─────────────►│     RH      │        │
│  │ (Reasoning) │               │  SUBSYSTEM  │               │ (Perception)│        │
│  │             │               │             │               │             │        │
│  │ ┌─────────┐ │               │ ┌─────────┐ │               │ ┌─────────┐ │        │
│  │ │Delibera-│ │               │ │Semantic │ │               │ │World    │ │        │
│  │ │tive     │ │               │ │Memory   │ │               │ │Model    │ │        │
│  │ └─────────┘ │               │ └─────────┘ │               │ └─────────┘ │        │
│  │ ┌─────────┐ │               │ ┌─────────┐ │               │ ┌─────────┐ │        │
│  │ │Planning │ │               │ │Episodic │ │               │ │Percept- │ │        │
│  │ │         │ │               │ │Memory   │ │               │ │ion      │ │        │
│  │ └─────────┘ │               │ └─────────┘ │               │ └─────────┘ │        │
│  │ ┌─────────┐ │               │ ┌─────────┐ │               │ ┌─────────┐ │        │
│  │ │Language │ │               │ │Procedur-│ │               │ │Simulat- │ │        │
│  │ │         │ │               │ │al Memory│ │               │ │ion      │ │        │
│  │ └─────────┘ │               │ └─────────┘ │               │ └─────────┘ │        │
│  └──────┬──────┘               └──────┬──────┘               └──────┬──────┘        │
│         │                             │                             │                │
│         └─────────────────────────────┼─────────────────────────────┘                │
│                                       │                                              │
│                                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                           METACOGNITION                                      │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │    │
│  │   │  Reasoning  │  │ Consistency │  │ Confidence  │  │    Anomaly      │    │    │
│  │   │   Trace     │  │   Checker   │  │ Calibration │  │   Detection     │    │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                              │
│                                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                              SAFETY                                          │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │    │
│  │   │   Reflex    │  │  Tactical   │  │  Strategic  │  │    ErisML       │    │    │
│  │   │   Layer     │  │   Layer     │  │   Layer     │  │   Integration   │    │    │
│  │   │  (<100μs)   │  │ (10-100ms)  │  │ (100ms-10s) │  │                 │    │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                              │
│                                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                           ACTION EXECUTION                                   │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │    │
│  │   │   Motor     │  │   Speech    │  │   API       │  │   Environment   │    │    │
│  │   │  Control    │  │   Output    │  │   Calls     │  │   Interaction   │    │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Current State Assessment

### Subsystem Status

| Subsystem | Proto | Services | Integration | Tests |
|-----------|-------|----------|-------------|-------|
| LH (Left Hemisphere) | Complete | Partial | Stub | Partial |
| RH (Right Hemisphere) | Stub | Stub | None | None |
| Memory | Complete | Partial | Stub | Partial |
| Metacognition | Planned | Stub | None | None |
| Event Fabric | N/A | Partial | Partial | Partial |
| Safety | Complete | Partial | Partial | Partial |
| ErisML | Complete | Functional | Functional | Partial |

### Integration Points Needed

1. **LH ↔ RH**: Perception-to-reasoning pipeline
2. **LH ↔ Memory**: Context retrieval and storage
3. **RH ↔ Memory**: World state persistence
4. **All ↔ Event Fabric**: Pub/sub messaging
5. **All ↔ Metacognition**: Reasoning trace collection
6. **All ↔ Safety**: Pre/in/post action checks
7. **Safety ↔ ErisML**: Ethical evaluation

---

## Sprint 1: Service Registry & Discovery (Weeks 1-2)

### Goals
- Centralized service registry
- Health checking and failover
- Dynamic endpoint resolution
- Configuration management

### Tasks

#### 1.1 Service Registry

```python
# src/agi/integration/registry.py
"""Service registry for AGI-HPC subsystems."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import grpc

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"


class ServiceType(Enum):
    """AGI-HPC service types."""
    # Core cognitive services
    LH_DELIBERATIVE = "lh.deliberative"
    LH_PLANNING = "lh.planning"
    LH_LANGUAGE = "lh.language"
    RH_PERCEPTION = "rh.perception"
    RH_WORLD_MODEL = "rh.world_model"
    RH_SIMULATION = "rh.simulation"

    # Memory services
    MEMORY_SEMANTIC = "memory.semantic"
    MEMORY_EPISODIC = "memory.episodic"
    MEMORY_PROCEDURAL = "memory.procedural"

    # Meta services
    METACOGNITION = "meta.cognition"

    # Safety services
    SAFETY_GATEWAY = "safety.gateway"
    SAFETY_PRE_ACTION = "safety.pre_action"
    SAFETY_IN_ACTION = "safety.in_action"
    SAFETY_POST_ACTION = "safety.post_action"
    SAFETY_ERISML = "safety.erisml"

    # Infrastructure
    EVENT_FABRIC = "fabric.broker"


@dataclass
class ServiceEndpoint:
    """A service endpoint."""

    service_type: ServiceType
    instance_id: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    version: str = "0.0.0"

    @property
    def address(self) -> str:
        """Get gRPC address."""
        return f"{self.host}:{self.port}"

    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return self.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)


class ServiceRegistry:
    """Centralized service registry."""

    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        unhealthy_threshold: float = 30.0,
    ) -> None:
        """Initialize registry."""
        self._endpoints: dict[str, ServiceEndpoint] = {}
        self._by_type: dict[ServiceType, list[str]] = {
            st: [] for st in ServiceType
        }
        self._heartbeat_interval = heartbeat_interval
        self._unhealthy_threshold = unhealthy_threshold
        self._listeners: list[Callable] = []
        self._health_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start registry background tasks."""
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service registry started")

    async def stop(self) -> None:
        """Stop registry."""
        if self._health_task:
            self._health_task.cancel()

    def register(
        self,
        service_type: ServiceType,
        host: str,
        port: int,
        instance_id: str | None = None,
        metadata: dict | None = None,
        version: str = "0.0.0",
    ) -> ServiceEndpoint:
        """Register a service endpoint."""
        import uuid

        instance_id = instance_id or str(uuid.uuid4())[:8]
        key = f"{service_type.value}:{instance_id}"

        endpoint = ServiceEndpoint(
            service_type=service_type,
            instance_id=instance_id,
            host=host,
            port=port,
            metadata=metadata or {},
            version=version,
            status=ServiceStatus.STARTING,
        )

        self._endpoints[key] = endpoint
        self._by_type[service_type].append(key)

        logger.info(
            "Registered service: %s at %s:%d",
            service_type.value, host, port,
        )

        self._notify_listeners("register", endpoint)
        return endpoint

    def deregister(self, service_type: ServiceType, instance_id: str) -> bool:
        """Deregister a service endpoint."""
        key = f"{service_type.value}:{instance_id}"

        if key in self._endpoints:
            endpoint = self._endpoints.pop(key)
            self._by_type[service_type].remove(key)
            self._notify_listeners("deregister", endpoint)
            logger.info("Deregistered service: %s", key)
            return True
        return False

    def heartbeat(
        self,
        service_type: ServiceType,
        instance_id: str,
        status: ServiceStatus = ServiceStatus.HEALTHY,
    ) -> bool:
        """Update service heartbeat."""
        key = f"{service_type.value}:{instance_id}"

        if key in self._endpoints:
            endpoint = self._endpoints[key]
            endpoint.last_heartbeat = datetime.now()
            endpoint.status = status
            return True
        return False

    def get(
        self,
        service_type: ServiceType,
        strategy: str = "round_robin",
    ) -> ServiceEndpoint | None:
        """Get a healthy endpoint for service type."""
        keys = self._by_type.get(service_type, [])
        healthy = [
            self._endpoints[k] for k in keys
            if self._endpoints[k].is_healthy()
        ]

        if not healthy:
            return None

        if strategy == "round_robin":
            # Simple round-robin
            return healthy[0]
        elif strategy == "random":
            import random
            return random.choice(healthy)
        elif strategy == "least_connections":
            # Would need connection tracking
            return healthy[0]

        return healthy[0]

    def get_all(
        self,
        service_type: ServiceType,
        include_unhealthy: bool = False,
    ) -> list[ServiceEndpoint]:
        """Get all endpoints for service type."""
        keys = self._by_type.get(service_type, [])
        endpoints = [self._endpoints[k] for k in keys]

        if not include_unhealthy:
            endpoints = [e for e in endpoints if e.is_healthy()]

        return endpoints

    def add_listener(self, callback: Callable) -> None:
        """Add registry change listener."""
        self._listeners.append(callback)

    def _notify_listeners(self, event: str, endpoint: ServiceEndpoint) -> None:
        """Notify all listeners of change."""
        for listener in self._listeners:
            try:
                listener(event, endpoint)
            except Exception as e:
                logger.error("Listener error: %s", e)

    async def _health_check_loop(self) -> None:
        """Background health checking."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)

            now = datetime.now()
            threshold = timedelta(seconds=self._unhealthy_threshold)

            for key, endpoint in self._endpoints.items():
                if now - endpoint.last_heartbeat > threshold:
                    if endpoint.status != ServiceStatus.UNHEALTHY:
                        endpoint.status = ServiceStatus.UNHEALTHY
                        logger.warning(
                            "Service unhealthy (no heartbeat): %s",
                            key,
                        )
                        self._notify_listeners("unhealthy", endpoint)
```

#### 1.2 gRPC Service Discovery

```python
# src/agi/integration/discovery.py
"""gRPC service discovery and connection management."""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

import grpc

from agi.integration.registry import ServiceRegistry, ServiceType, ServiceEndpoint

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceDiscovery:
    """Service discovery and connection pooling."""

    def __init__(self, registry: ServiceRegistry) -> None:
        """Initialize discovery."""
        self.registry = registry
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, object] = {}

    async def get_channel(
        self,
        service_type: ServiceType,
    ) -> grpc.aio.Channel | None:
        """Get or create channel to service."""
        endpoint = self.registry.get(service_type)
        if not endpoint:
            logger.warning("No healthy endpoint for %s", service_type.value)
            return None

        key = endpoint.address
        if key not in self._channels:
            self._channels[key] = grpc.aio.insecure_channel(
                endpoint.address,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                ],
            )
            logger.debug("Created channel to %s", key)

        return self._channels[key]

    async def get_stub(
        self,
        service_type: ServiceType,
        stub_class: type[T],
    ) -> T | None:
        """Get or create stub for service."""
        channel = await self.get_channel(service_type)
        if not channel:
            return None

        key = f"{service_type.value}:{stub_class.__name__}"
        if key not in self._stubs:
            self._stubs[key] = stub_class(channel)

        return self._stubs[key]

    async def close_all(self) -> None:
        """Close all channels."""
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
        self._stubs.clear()


class ServiceClient:
    """Base client with automatic discovery."""

    def __init__(
        self,
        discovery: ServiceDiscovery,
        service_type: ServiceType,
        stub_class: type,
    ) -> None:
        """Initialize client."""
        self.discovery = discovery
        self.service_type = service_type
        self.stub_class = stub_class
        self._stub = None

    async def _get_stub(self):
        """Get or refresh stub."""
        if self._stub is None:
            self._stub = await self.discovery.get_stub(
                self.service_type,
                self.stub_class,
            )
        return self._stub

    async def _call_with_retry(
        self,
        method_name: str,
        request,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Call method with automatic retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                stub = await self._get_stub()
                if stub is None:
                    raise grpc.RpcError("No healthy endpoint")

                method = getattr(stub, method_name)
                return await asyncio.wait_for(
                    method(request),
                    timeout=timeout,
                )
            except grpc.RpcError as e:
                last_error = e
                logger.warning(
                    "RPC failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                self._stub = None  # Force refresh
                await asyncio.sleep(0.5 * (attempt + 1))

        raise last_error
```

#### 1.3 Configuration Management

```python
# src/agi/integration/config.py
"""Centralized configuration management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SubsystemConfig:
    """Configuration for a subsystem."""

    enabled: bool = True
    host: str = "localhost"
    port: int = 50051
    replicas: int = 1
    resources: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Full system configuration."""

    # Subsystems
    lh: SubsystemConfig = field(default_factory=SubsystemConfig)
    rh: SubsystemConfig = field(default_factory=SubsystemConfig)
    memory: SubsystemConfig = field(default_factory=SubsystemConfig)
    metacognition: SubsystemConfig = field(default_factory=SubsystemConfig)
    safety: SubsystemConfig = field(default_factory=SubsystemConfig)
    event_fabric: SubsystemConfig = field(default_factory=SubsystemConfig)

    # Global settings
    environment: str = "development"
    log_level: str = "INFO"

    # Database connections
    postgres_dsn: str = "postgresql://agi:agi@localhost:5432/agi"
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"

    # External services
    erisml_endpoint: str = "localhost:50052"

    @classmethod
    def from_yaml(cls, path: str | Path) -> IntegrationConfig:
        """Load configuration from YAML file."""
        path = Path(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)

        config = cls()

        # Parse subsystem configs
        for name in ["lh", "rh", "memory", "metacognition", "safety", "event_fabric"]:
            if name in data:
                setattr(config, name, SubsystemConfig(**data[name]))

        # Parse global settings
        config.environment = data.get("environment", config.environment)
        config.log_level = data.get("log_level", config.log_level)
        config.postgres_dsn = data.get("postgres_dsn", config.postgres_dsn)
        config.redis_url = data.get("redis_url", config.redis_url)
        config.qdrant_url = data.get("qdrant_url", config.qdrant_url)
        config.erisml_endpoint = data.get("erisml_endpoint", config.erisml_endpoint)

        return config

    @staticmethod
    def _apply_env_overrides(data: dict) -> dict:
        """Apply environment variable overrides."""
        def replace_env(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                default = None
                if ":-" in env_var:
                    env_var, default = env_var.split(":-", 1)
                return os.environ.get(env_var, default)
            elif isinstance(value, dict):
                return {k: replace_env(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_env(v) for v in value]
            return value

        return replace_env(data)
```

### Deliverables
- [ ] Service registry with health checking
- [ ] gRPC service discovery
- [ ] Connection pooling
- [ ] Configuration management
- [ ] Environment variable overrides
- [ ] Unit tests

---

## Sprint 2: Event Fabric Integration (Weeks 3-4)

### Goals
- Connect all subsystems to event fabric
- Define standard event schemas
- Implement pub/sub patterns
- Add event tracing

### Tasks

#### 2.1 Standard Event Schema

```python
# src/agi/integration/events/schema.py
"""Standard event schemas for AGI-HPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class EventCategory(Enum):
    """Event categories."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    MEMORY = "memory"
    SAFETY = "safety"
    ACTION = "action"
    META = "meta"
    SYSTEM = "system"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EventHeader:
    """Standard event header."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    category: EventCategory = EventCategory.SYSTEM
    priority: EventPriority = EventPriority.NORMAL
    source_service: str = ""
    source_instance: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


@dataclass
class AGIEvent:
    """Base AGI event."""

    header: EventHeader
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "header": {
                "event_id": self.header.event_id,
                "event_type": self.header.event_type,
                "category": self.header.category.value,
                "priority": self.header.priority.value,
                "source_service": self.header.source_service,
                "source_instance": self.header.source_instance,
                "timestamp": self.header.timestamp.isoformat(),
                "correlation_id": self.header.correlation_id,
                "causation_id": self.header.causation_id,
                "trace_id": self.header.trace_id,
                "span_id": self.header.span_id,
            },
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AGIEvent:
        """Create from dictionary."""
        header_data = data["header"]
        header = EventHeader(
            event_id=header_data["event_id"],
            event_type=header_data["event_type"],
            category=EventCategory(header_data["category"]),
            priority=EventPriority(header_data["priority"]),
            source_service=header_data["source_service"],
            source_instance=header_data["source_instance"],
            timestamp=datetime.fromisoformat(header_data["timestamp"]),
            correlation_id=header_data.get("correlation_id"),
            causation_id=header_data.get("causation_id"),
            trace_id=header_data.get("trace_id"),
            span_id=header_data.get("span_id"),
        )
        return cls(
            header=header,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


# Specific event types
@dataclass
class PerceptionEvent(AGIEvent):
    """Perception update event."""

    def __init__(
        self,
        source: str,
        percept_type: str,
        data: dict,
        confidence: float = 1.0,
        **kwargs,
    ):
        header = EventHeader(
            event_type=f"perception.{percept_type}",
            category=EventCategory.PERCEPTION,
            source_service=source,
            **kwargs,
        )
        super().__init__(
            header=header,
            payload={
                "percept_type": percept_type,
                "data": data,
                "confidence": confidence,
            },
        )


@dataclass
class ReasoningEvent(AGIEvent):
    """Reasoning step event."""

    def __init__(
        self,
        source: str,
        reasoning_type: str,
        input_data: dict,
        output_data: dict,
        trace: list[str] | None = None,
        **kwargs,
    ):
        header = EventHeader(
            event_type=f"reasoning.{reasoning_type}",
            category=EventCategory.REASONING,
            source_service=source,
            **kwargs,
        )
        super().__init__(
            header=header,
            payload={
                "reasoning_type": reasoning_type,
                "input": input_data,
                "output": output_data,
                "trace": trace or [],
            },
        )


@dataclass
class ActionEvent(AGIEvent):
    """Action execution event."""

    def __init__(
        self,
        source: str,
        action_type: str,
        action_id: str,
        parameters: dict,
        status: str = "pending",
        **kwargs,
    ):
        header = EventHeader(
            event_type=f"action.{action_type}.{status}",
            category=EventCategory.ACTION,
            source_service=source,
            **kwargs,
        )
        super().__init__(
            header=header,
            payload={
                "action_type": action_type,
                "action_id": action_id,
                "parameters": parameters,
                "status": status,
            },
        )


@dataclass
class SafetyEvent(AGIEvent):
    """Safety decision event."""

    def __init__(
        self,
        source: str,
        decision: str,
        action_id: str,
        reason: str = "",
        violations: list[dict] | None = None,
        **kwargs,
    ):
        header = EventHeader(
            event_type=f"safety.decision.{decision.lower()}",
            category=EventCategory.SAFETY,
            priority=EventPriority.HIGH,
            source_service=source,
            **kwargs,
        )
        super().__init__(
            header=header,
            payload={
                "decision": decision,
                "action_id": action_id,
                "reason": reason,
                "violations": violations or [],
            },
        )
```

#### 2.2 Event Bus Integration

```python
# src/agi/integration/events/bus.py
"""Event bus integration for all subsystems."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Awaitable

from agi.fabric import EventFabric, ZmqBackend
from agi.integration.events.schema import AGIEvent, EventCategory

logger = logging.getLogger(__name__)

EventHandler = Callable[[AGIEvent], Awaitable[None]]


class IntegratedEventBus:
    """Event bus connecting all AGI-HPC subsystems."""

    def __init__(
        self,
        broker_address: str = "tcp://localhost:5555",
    ) -> None:
        """Initialize event bus."""
        self.broker_address = broker_address
        self._fabric: EventFabric | None = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._category_handlers: dict[EventCategory, list[EventHandler]] = {
            cat: [] for cat in EventCategory
        }
        self._running = False

    async def start(self) -> None:
        """Start event bus."""
        backend = ZmqBackend(self.broker_address)
        self._fabric = EventFabric(backend)
        await self._fabric.connect()
        self._running = True

        # Start dispatch loop
        asyncio.create_task(self._dispatch_loop())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop event bus."""
        self._running = False
        if self._fabric:
            await self._fabric.disconnect()

    async def publish(self, event: AGIEvent) -> None:
        """Publish event to bus."""
        topic = event.header.event_type
        data = json.dumps(event.to_dict())
        await self._fabric.publish(topic, data.encode())

        logger.debug(
            "Published event: %s (id=%s)",
            topic, event.header.event_id,
        )

    def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to events matching pattern."""
        if pattern not in self._handlers:
            self._handlers[pattern] = []
            asyncio.create_task(self._fabric.subscribe(pattern))

        self._handlers[pattern].append(handler)
        logger.debug("Subscribed to pattern: %s", pattern)

    def subscribe_category(
        self,
        category: EventCategory,
        handler: EventHandler,
    ) -> None:
        """Subscribe to all events in category."""
        self._category_handlers[category].append(handler)
        self.subscribe(f"{category.value}.*", handler)

    async def _dispatch_loop(self) -> None:
        """Dispatch received events to handlers."""
        while self._running:
            try:
                topic, data = await self._fabric.receive()
                event = AGIEvent.from_dict(json.loads(data.decode()))

                # Dispatch to matching handlers
                await self._dispatch_event(event)
            except Exception as e:
                logger.error("Event dispatch error: %s", e)
                await asyncio.sleep(0.1)

    async def _dispatch_event(self, event: AGIEvent) -> None:
        """Dispatch event to registered handlers."""
        tasks = []

        # Check pattern handlers
        for pattern, handlers in self._handlers.items():
            if self._matches_pattern(event.header.event_type, pattern):
                for handler in handlers:
                    tasks.append(self._safe_call(handler, event))

        # Check category handlers
        for handler in self._category_handlers.get(event.header.category, []):
            tasks.append(self._safe_call(handler, event))

        if tasks:
            await asyncio.gather(*tasks)

    @staticmethod
    def _matches_pattern(topic: str, pattern: str) -> bool:
        """Check if topic matches subscription pattern."""
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)

    @staticmethod
    async def _safe_call(handler: EventHandler, event: AGIEvent) -> None:
        """Safely call handler."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                "Handler error for %s: %s",
                event.header.event_type, e,
            )
```

#### 2.3 Subsystem Event Adapters

```python
# src/agi/integration/adapters/lh_adapter.py
"""LH subsystem event adapter."""

from __future__ import annotations

import logging

from agi.integration.events.bus import IntegratedEventBus
from agi.integration.events.schema import ReasoningEvent, ActionEvent
from agi.lh.deliberative.service import DeliberativeService
from agi.lh.planning.service import PlanningService

logger = logging.getLogger(__name__)


class LHEventAdapter:
    """Adapter connecting LH services to event bus."""

    def __init__(
        self,
        event_bus: IntegratedEventBus,
        deliberative: DeliberativeService,
        planning: PlanningService,
    ) -> None:
        """Initialize adapter."""
        self.event_bus = event_bus
        self.deliberative = deliberative
        self.planning = planning
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        # Subscribe to perception events for reasoning
        self.event_bus.subscribe(
            "perception.*",
            self._handle_perception,
        )

        # Subscribe to memory retrieval results
        self.event_bus.subscribe(
            "memory.retrieval.*",
            self._handle_memory_retrieval,
        )

        # Subscribe to safety decisions
        self.event_bus.subscribe(
            "safety.decision.*",
            self._handle_safety_decision,
        )

    async def _handle_perception(self, event) -> None:
        """Handle perception events."""
        logger.debug("LH received perception: %s", event.header.event_type)

        # Update deliberative context
        await self.deliberative.update_context(
            perception=event.payload,
        )

    async def _handle_memory_retrieval(self, event) -> None:
        """Handle memory retrieval results."""
        logger.debug("LH received memory: %s", event.header.event_type)

        # Incorporate into reasoning
        await self.deliberative.incorporate_memory(
            memories=event.payload.get("results", []),
        )

    async def _handle_safety_decision(self, event) -> None:
        """Handle safety decisions."""
        decision = event.payload.get("decision")
        action_id = event.payload.get("action_id")

        logger.info(
            "LH received safety decision: %s for %s",
            decision, action_id,
        )

        if decision == "DENY":
            await self.planning.cancel_action(action_id)
        elif decision == "DEFER":
            await self.planning.pause_action(action_id)
        elif decision == "ALLOW":
            await self.planning.execute_action(action_id)

    async def publish_reasoning_result(
        self,
        reasoning_type: str,
        input_data: dict,
        output_data: dict,
        trace: list[str],
    ) -> None:
        """Publish reasoning result event."""
        event = ReasoningEvent(
            source="lh.deliberative",
            reasoning_type=reasoning_type,
            input_data=input_data,
            output_data=output_data,
            trace=trace,
        )
        await self.event_bus.publish(event)

    async def publish_action_request(
        self,
        action_type: str,
        action_id: str,
        parameters: dict,
    ) -> None:
        """Publish action request event."""
        event = ActionEvent(
            source="lh.planning",
            action_type=action_type,
            action_id=action_id,
            parameters=parameters,
            status="requested",
        )
        await self.event_bus.publish(event)
```

### Deliverables
- [ ] Standard event schemas
- [ ] Integrated event bus
- [ ] LH event adapter
- [ ] RH event adapter
- [ ] Memory event adapter
- [ ] Safety event adapter
- [ ] Event tracing integration

---

## Sprint 3: Cognitive Pipeline (Weeks 5-6)

### Goals
- Perception-to-action pipeline
- Memory context integration
- Parallel reasoning paths
- Pipeline orchestration

### Tasks

#### 3.1 Cognitive Pipeline Orchestrator

```python
# src/agi/integration/pipeline/orchestrator.py
"""Cognitive pipeline orchestrator."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agi.integration.registry import ServiceRegistry, ServiceType
from agi.integration.discovery import ServiceDiscovery
from agi.integration.events.bus import IntegratedEventBus

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages."""
    PERCEPTION = "perception"
    MEMORY_RETRIEVAL = "memory_retrieval"
    REASONING = "reasoning"
    PLANNING = "planning"
    SAFETY_CHECK = "safety_check"
    EXECUTION = "execution"
    MEMORY_STORE = "memory_store"
    METACOGNITION = "metacognition"


@dataclass
class PipelineContext:
    """Context passed through pipeline."""

    pipeline_id: str
    started_at: datetime = field(default_factory=datetime.now)
    current_stage: PipelineStage = PipelineStage.PERCEPTION

    # Accumulated data
    perception: dict = field(default_factory=dict)
    memories: list = field(default_factory=list)
    reasoning_result: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    safety_decision: dict = field(default_factory=dict)
    execution_result: dict = field(default_factory=dict)

    # Metadata
    trace: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class CognitivePipeline:
    """Main cognitive pipeline orchestrator."""

    def __init__(
        self,
        discovery: ServiceDiscovery,
        event_bus: IntegratedEventBus,
    ) -> None:
        """Initialize pipeline."""
        self.discovery = discovery
        self.event_bus = event_bus
        self._active_pipelines: dict[str, PipelineContext] = {}

    async def process(
        self,
        input_data: dict,
        pipeline_id: str | None = None,
    ) -> PipelineContext:
        """Process input through full cognitive pipeline."""
        import uuid

        pipeline_id = pipeline_id or str(uuid.uuid4())
        context = PipelineContext(pipeline_id=pipeline_id)
        self._active_pipelines[pipeline_id] = context

        try:
            # Stage 1: Perception
            context = await self._perception_stage(context, input_data)

            # Stage 2: Memory Retrieval (parallel with perception processing)
            context = await self._memory_retrieval_stage(context)

            # Stage 3: Reasoning
            context = await self._reasoning_stage(context)

            # Stage 4: Planning
            context = await self._planning_stage(context)

            # Stage 5: Safety Check
            context = await self._safety_check_stage(context)

            # Stage 6: Execution (if approved)
            if context.safety_decision.get("decision") == "ALLOW":
                context = await self._execution_stage(context)

            # Stage 7: Memory Store
            context = await self._memory_store_stage(context)

            # Stage 8: Metacognition
            context = await self._metacognition_stage(context)

        except Exception as e:
            context.errors.append(str(e))
            logger.error("Pipeline error: %s", e)
        finally:
            del self._active_pipelines[pipeline_id]

        return context

    async def _perception_stage(
        self,
        context: PipelineContext,
        input_data: dict,
    ) -> PipelineContext:
        """Perception processing stage."""
        context.current_stage = PipelineStage.PERCEPTION
        context.trace.append(f"[{datetime.now().isoformat()}] Perception started")

        # Get RH perception service
        from proto.rh_pb2_grpc import PerceptionServiceStub
        from proto.rh_pb2 import PerceiveRequest

        stub = await self.discovery.get_stub(
            ServiceType.RH_PERCEPTION,
            PerceptionServiceStub,
        )

        if stub:
            request = PerceiveRequest(
                sensor_data=input_data.get("sensors", {}),
            )
            response = await stub.Perceive(request)
            context.perception = {
                "objects": list(response.detected_objects),
                "scene": response.scene_description,
                "confidence": response.confidence,
            }
        else:
            # Fallback: pass through raw input
            context.perception = input_data

        context.trace.append(f"[{datetime.now().isoformat()}] Perception completed")
        return context

    async def _memory_retrieval_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Memory retrieval stage."""
        context.current_stage = PipelineStage.MEMORY_RETRIEVAL
        context.trace.append(f"[{datetime.now().isoformat()}] Memory retrieval started")

        # Parallel retrieval from all memory types
        tasks = [
            self._retrieve_semantic_memory(context),
            self._retrieve_episodic_memory(context),
            self._retrieve_procedural_memory(context),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                context.memories.extend(result)
            elif isinstance(result, Exception):
                context.errors.append(f"Memory retrieval error: {result}")

        context.trace.append(f"[{datetime.now().isoformat()}] Memory retrieval completed")
        return context

    async def _retrieve_semantic_memory(
        self,
        context: PipelineContext,
    ) -> list:
        """Retrieve relevant semantic memories."""
        from proto.memory_pb2_grpc import SemanticMemoryServiceStub
        from proto.memory_pb2 import SemanticQueryRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_SEMANTIC,
            SemanticMemoryServiceStub,
        )

        if not stub:
            return []

        # Build query from perception
        query_text = context.perception.get("scene", "")

        request = SemanticQueryRequest(
            query=query_text,
            limit=10,
        )
        response = await stub.Query(request)

        return [
            {"type": "semantic", "content": m.content, "score": m.similarity}
            for m in response.results
        ]

    async def _reasoning_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Reasoning stage."""
        context.current_stage = PipelineStage.REASONING
        context.trace.append(f"[{datetime.now().isoformat()}] Reasoning started")

        from proto.lh_pb2_grpc import DeliberativeServiceStub
        from proto.lh_pb2 import ReasonRequest

        stub = await self.discovery.get_stub(
            ServiceType.LH_DELIBERATIVE,
            DeliberativeServiceStub,
        )

        if stub:
            request = ReasonRequest(
                perception=context.perception,
                memories=context.memories,
            )
            response = await stub.Reason(request)
            context.reasoning_result = {
                "conclusion": response.conclusion,
                "confidence": response.confidence,
                "trace": list(response.reasoning_trace),
            }

        context.trace.append(f"[{datetime.now().isoformat()}] Reasoning completed")
        return context

    async def _planning_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Planning stage."""
        context.current_stage = PipelineStage.PLANNING
        context.trace.append(f"[{datetime.now().isoformat()}] Planning started")

        from proto.lh_pb2_grpc import PlanningServiceStub
        from proto.lh_pb2 import PlanRequest

        stub = await self.discovery.get_stub(
            ServiceType.LH_PLANNING,
            PlanningServiceStub,
        )

        if stub:
            request = PlanRequest(
                goal=context.reasoning_result.get("conclusion", ""),
                context=context.perception,
            )
            response = await stub.CreatePlan(request)
            context.plan = {
                "plan_id": response.plan_id,
                "steps": [
                    {"action": s.action, "params": dict(s.parameters)}
                    for s in response.steps
                ],
            }

        context.trace.append(f"[{datetime.now().isoformat()}] Planning completed")
        return context

    async def _safety_check_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Safety check stage."""
        context.current_stage = PipelineStage.SAFETY_CHECK
        context.trace.append(f"[{datetime.now().isoformat()}] Safety check started")

        from proto.safety_pb2_grpc import SafetyGatewayStub
        from proto.safety_pb2 import CheckPlanRequest

        stub = await self.discovery.get_stub(
            ServiceType.SAFETY_GATEWAY,
            SafetyGatewayStub,
        )

        if stub:
            request = CheckPlanRequest(
                plan_id=context.plan.get("plan_id", ""),
                steps=context.plan.get("steps", []),
            )
            response = await stub.CheckPlan(request)
            context.safety_decision = {
                "decision": response.decision.name,
                "violations": [
                    {"rule": v.rule_id, "message": v.message}
                    for v in response.violations
                ],
            }
        else:
            # Default to DEFER without safety service
            context.safety_decision = {
                "decision": "DEFER",
                "violations": [{"rule": "no_safety_service", "message": "Safety service unavailable"}],
            }

        context.trace.append(f"[{datetime.now().isoformat()}] Safety check completed")
        return context

    async def _execution_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Action execution stage."""
        context.current_stage = PipelineStage.EXECUTION
        context.trace.append(f"[{datetime.now().isoformat()}] Execution started")

        # Execute plan steps
        for step in context.plan.get("steps", []):
            # Publish action event
            from agi.integration.events.schema import ActionEvent

            event = ActionEvent(
                source="pipeline",
                action_type=step["action"],
                action_id=f"{context.pipeline_id}:{step['action']}",
                parameters=step["params"],
                status="executing",
            )
            await self.event_bus.publish(event)

            # Execute via RH control service
            # ... execution logic ...

        context.trace.append(f"[{datetime.now().isoformat()}] Execution completed")
        return context

    async def _memory_store_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Store experience in episodic memory."""
        context.current_stage = PipelineStage.MEMORY_STORE
        context.trace.append(f"[{datetime.now().isoformat()}] Memory store started")

        from proto.memory_pb2_grpc import EpisodicMemoryServiceStub
        from proto.memory_pb2 import StoreEpisodeRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_EPISODIC,
            EpisodicMemoryServiceStub,
        )

        if stub:
            request = StoreEpisodeRequest(
                episode_id=context.pipeline_id,
                perception=context.perception,
                reasoning=context.reasoning_result,
                plan=context.plan,
                outcome=context.execution_result,
            )
            await stub.StoreEpisode(request)

        context.trace.append(f"[{datetime.now().isoformat()}] Memory store completed")
        return context

    async def _metacognition_stage(
        self,
        context: PipelineContext,
    ) -> PipelineContext:
        """Metacognitive reflection stage."""
        context.current_stage = PipelineStage.METACOGNITION
        context.trace.append(f"[{datetime.now().isoformat()}] Metacognition started")

        from proto.meta_pb2_grpc import MetacognitionServiceStub
        from proto.meta_pb2 import AnalyzeTraceRequest

        stub = await self.discovery.get_stub(
            ServiceType.METACOGNITION,
            MetacognitionServiceStub,
        )

        if stub:
            request = AnalyzeTraceRequest(
                trace=context.trace,
                reasoning_trace=context.reasoning_result.get("trace", []),
            )
            response = await stub.AnalyzeTrace(request)

            context.metrics["confidence_calibration"] = response.calibration_score
            context.metrics["consistency_score"] = response.consistency_score

        context.trace.append(f"[{datetime.now().isoformat()}] Metacognition completed")
        return context
```

### Deliverables
- [ ] Pipeline orchestrator
- [ ] Stage-by-stage processing
- [ ] Parallel memory retrieval
- [ ] Safety integration
- [ ] Metacognitive reflection
- [ ] Pipeline metrics

---

## Sprint 4: Safety Gateway Integration (Weeks 7-8)

### Goals
- Pre-action safety for all subsystems
- In-action monitoring
- Post-action audit
- ErisML integration

### Tasks

#### 4.1 Unified Safety Integration

```python
# src/agi/integration/safety/integration.py
"""Unified safety integration for all subsystems."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from agi.integration.discovery import ServiceDiscovery
from agi.integration.registry import ServiceType
from agi.integration.events.bus import IntegratedEventBus
from agi.integration.events.schema import SafetyEvent

logger = logging.getLogger(__name__)


@dataclass
class SafetyContext:
    """Context for safety-wrapped action."""

    action_id: str
    action_type: str
    parameters: dict
    agent_id: str
    pre_check_result: dict | None = None
    in_action_stream: Any = None
    post_audit_result: dict | None = None


class SafetyIntegration:
    """Unified safety integration for all actions."""

    def __init__(
        self,
        discovery: ServiceDiscovery,
        event_bus: IntegratedEventBus,
    ) -> None:
        """Initialize safety integration."""
        self.discovery = discovery
        self.event_bus = event_bus

    @asynccontextmanager
    async def safe_action(
        self,
        action_type: str,
        parameters: dict[str, Any],
        agent_id: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[SafetyContext]:
        """Context manager for safety-wrapped actions.

        Usage:
            async with safety.safe_action("move", params, "agent1") as ctx:
                if ctx.pre_check_result["decision"] == "ALLOW":
                    await execute_action(params)
        """
        import uuid

        action_id = str(uuid.uuid4())
        safety_ctx = SafetyContext(
            action_id=action_id,
            action_type=action_type,
            parameters=parameters,
            agent_id=agent_id,
        )

        try:
            # Pre-action check
            safety_ctx.pre_check_result = await self._pre_action_check(
                safety_ctx, context or {}
            )

            # Publish safety decision event
            await self.event_bus.publish(SafetyEvent(
                source="safety.integration",
                decision=safety_ctx.pre_check_result["decision"],
                action_id=action_id,
                reason=safety_ctx.pre_check_result.get("reason", ""),
                violations=safety_ctx.pre_check_result.get("violations", []),
            ))

            if safety_ctx.pre_check_result["decision"] == "DENY":
                logger.warning(
                    "Action %s denied: %s",
                    action_id,
                    safety_ctx.pre_check_result.get("reason"),
                )
                yield safety_ctx
                return

            # Start in-action monitoring
            safety_ctx.in_action_stream = await self._start_in_action_monitor(
                safety_ctx
            )

            yield safety_ctx

        finally:
            # Stop in-action monitoring
            if safety_ctx.in_action_stream:
                await self._stop_in_action_monitor(safety_ctx)

            # Post-action audit
            safety_ctx.post_audit_result = await self._post_action_audit(
                safety_ctx
            )

    async def _pre_action_check(
        self,
        ctx: SafetyContext,
        context: dict,
    ) -> dict:
        """Perform pre-action safety check."""
        from proto.safety_pb2_grpc import PreActionSafetyServiceStub
        from proto.safety_pb2 import PreActionCheckRequest

        stub = await self.discovery.get_stub(
            ServiceType.SAFETY_PRE_ACTION,
            PreActionSafetyServiceStub,
        )

        if not stub:
            logger.warning("Pre-action service unavailable, defaulting to DEFER")
            return {"decision": "DEFER", "reason": "Safety service unavailable"}

        request = PreActionCheckRequest(
            action_id=ctx.action_id,
            agent_id=ctx.agent_id,
            action_type=ctx.action_type,
            parameters=ctx.parameters,
            context=context,
        )

        response = await stub.CheckAction(request)

        return {
            "decision": response.decision.name,
            "violations": [
                {"rule": v.rule_id, "message": v.message}
                for v in response.violations
            ],
            "proof_hash": response.decision_proof_hash,
        }

    async def _start_in_action_monitor(
        self,
        ctx: SafetyContext,
    ) -> Any:
        """Start in-action safety monitoring."""
        from proto.safety_pb2_grpc import InActionSafetyServiceStub
        from proto.safety_pb2 import ActionMonitorRequest

        stub = await self.discovery.get_stub(
            ServiceType.SAFETY_IN_ACTION,
            InActionSafetyServiceStub,
        )

        if not stub:
            return None

        # Create monitoring stream
        async def send_metrics():
            while True:
                yield ActionMonitorRequest(
                    action_id=ctx.action_id,
                    metrics={"status": "running"},
                )
                await asyncio.sleep(0.1)

        return stub.MonitorAction(send_metrics())

    async def _stop_in_action_monitor(self, ctx: SafetyContext) -> None:
        """Stop in-action monitoring."""
        if ctx.in_action_stream:
            ctx.in_action_stream.cancel()

    async def _post_action_audit(self, ctx: SafetyContext) -> dict:
        """Perform post-action audit."""
        from proto.safety_pb2_grpc import PostActionSafetyServiceStub
        from proto.safety_pb2 import PostActionAuditRequest

        stub = await self.discovery.get_stub(
            ServiceType.SAFETY_POST_ACTION,
            PostActionSafetyServiceStub,
        )

        if not stub:
            return {"audited": False}

        request = PostActionAuditRequest(
            action_id=ctx.action_id,
            agent_id=ctx.agent_id,
            outcome_status="completed",
            actual_impact={},
        )

        response = await stub.AuditAction(request)

        return {
            "audited": True,
            "audit_proof_hash": response.audit_proof_hash,
            "learning_signals": len(response.learning_signals),
        }

    async def check_plan(
        self,
        plan_id: str,
        steps: list[dict],
        agent_id: str,
    ) -> dict:
        """Check entire plan through safety."""
        from proto.safety_pb2_grpc import SafetyGatewayStub
        from proto.safety_pb2 import CheckPlanRequest, PlanStep

        stub = await self.discovery.get_stub(
            ServiceType.SAFETY_GATEWAY,
            SafetyGatewayStub,
        )

        if not stub:
            return {"decision": "DEFER", "reason": "Safety gateway unavailable"}

        request = CheckPlanRequest(
            plan_id=plan_id,
            agent_id=agent_id,
            steps=[
                PlanStep(
                    step_id=str(i),
                    action_type=step["action"],
                    parameters=step.get("params", {}),
                )
                for i, step in enumerate(steps)
            ],
        )

        response = await stub.CheckPlan(request)

        return {
            "decision": response.decision.name,
            "step_decisions": [
                {"step": d.step_id, "decision": d.decision.name}
                for d in response.step_decisions
            ],
            "violations": [
                {"rule": v.rule_id, "message": v.message}
                for v in response.violations
            ],
        }
```

### Deliverables
- [ ] Unified safety integration
- [ ] Safe action context manager
- [ ] Plan-level safety checking
- [ ] Event bus integration
- [ ] Audit trail collection

---

## Sprint 5: Memory Integration (Weeks 9-10)

### Goals
- Unified memory interface
- Cross-memory retrieval
- Automatic context building
- Memory consolidation

### Tasks

#### 5.1 Unified Memory Interface

```python
# src/agi/integration/memory/unified.py
"""Unified memory interface for all memory types."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agi.integration.discovery import ServiceDiscovery
from agi.integration.registry import ServiceType

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """A memory retrieval result."""

    memory_type: str  # semantic, episodic, procedural
    content: Any
    score: float
    metadata: dict = field(default_factory=dict)
    timestamp: datetime | None = None


@dataclass
class ContextBundle:
    """Bundled context from all memory types."""

    query: str
    semantic_results: list[MemoryResult] = field(default_factory=list)
    episodic_results: list[MemoryResult] = field(default_factory=list)
    procedural_results: list[MemoryResult] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=datetime.now)

    @property
    def all_results(self) -> list[MemoryResult]:
        """Get all results combined."""
        return self.semantic_results + self.episodic_results + self.procedural_results

    def top_k(self, k: int = 10) -> list[MemoryResult]:
        """Get top-k results by score."""
        return sorted(self.all_results, key=lambda r: r.score, reverse=True)[:k]


class UnifiedMemory:
    """Unified interface to all memory subsystems."""

    def __init__(self, discovery: ServiceDiscovery) -> None:
        """Initialize unified memory."""
        self.discovery = discovery

    async def retrieve(
        self,
        query: str,
        memory_types: list[str] | None = None,
        limit_per_type: int = 10,
        min_score: float = 0.0,
    ) -> ContextBundle:
        """Retrieve context from all memory types."""
        memory_types = memory_types or ["semantic", "episodic", "procedural"]
        bundle = ContextBundle(query=query)

        tasks = []
        if "semantic" in memory_types:
            tasks.append(self._retrieve_semantic(query, limit_per_type, min_score))
        if "episodic" in memory_types:
            tasks.append(self._retrieve_episodic(query, limit_per_type, min_score))
        if "procedural" in memory_types:
            tasks.append(self._retrieve_procedural(query, limit_per_type, min_score))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Memory retrieval error: %s", result)
                continue

            mem_type = memory_types[i]
            if mem_type == "semantic":
                bundle.semantic_results = result
            elif mem_type == "episodic":
                bundle.episodic_results = result
            elif mem_type == "procedural":
                bundle.procedural_results = result

        return bundle

    async def _retrieve_semantic(
        self,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[MemoryResult]:
        """Retrieve from semantic memory."""
        from proto.memory_pb2_grpc import SemanticMemoryServiceStub
        from proto.memory_pb2 import SemanticQueryRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_SEMANTIC,
            SemanticMemoryServiceStub,
        )

        if not stub:
            return []

        request = SemanticQueryRequest(
            query=query,
            limit=limit,
            min_similarity=min_score,
        )
        response = await stub.Query(request)

        return [
            MemoryResult(
                memory_type="semantic",
                content=r.content,
                score=r.similarity,
                metadata=dict(r.metadata),
            )
            for r in response.results
        ]

    async def _retrieve_episodic(
        self,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[MemoryResult]:
        """Retrieve from episodic memory."""
        from proto.memory_pb2_grpc import EpisodicMemoryServiceStub
        from proto.memory_pb2 import EpisodicQueryRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_EPISODIC,
            EpisodicMemoryServiceStub,
        )

        if not stub:
            return []

        request = EpisodicQueryRequest(
            query=query,
            limit=limit,
        )
        response = await stub.Query(request)

        return [
            MemoryResult(
                memory_type="episodic",
                content={
                    "episode_id": r.episode_id,
                    "summary": r.summary,
                    "perception": dict(r.perception),
                    "outcome": dict(r.outcome),
                },
                score=r.relevance,
                timestamp=r.timestamp.ToDatetime() if r.HasField("timestamp") else None,
            )
            for r in response.episodes
        ]

    async def _retrieve_procedural(
        self,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[MemoryResult]:
        """Retrieve from procedural memory."""
        from proto.memory_pb2_grpc import ProceduralMemoryServiceStub
        from proto.memory_pb2 import ProceduralQueryRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_PROCEDURAL,
            ProceduralMemoryServiceStub,
        )

        if not stub:
            return []

        request = ProceduralQueryRequest(
            skill_query=query,
            limit=limit,
        )
        response = await stub.QuerySkills(request)

        return [
            MemoryResult(
                memory_type="procedural",
                content={
                    "skill_id": s.skill_id,
                    "name": s.name,
                    "steps": [dict(step) for step in s.steps],
                    "proficiency": s.proficiency,
                },
                score=s.relevance,
            )
            for s in response.skills
        ]

    async def store_experience(
        self,
        experience: dict,
    ) -> str:
        """Store experience in episodic memory."""
        from proto.memory_pb2_grpc import EpisodicMemoryServiceStub
        from proto.memory_pb2 import StoreEpisodeRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_EPISODIC,
            EpisodicMemoryServiceStub,
        )

        if not stub:
            raise RuntimeError("Episodic memory service unavailable")

        request = StoreEpisodeRequest(**experience)
        response = await stub.StoreEpisode(request)

        return response.episode_id

    async def update_skill(
        self,
        skill_id: str,
        success: bool,
        execution_data: dict,
    ) -> None:
        """Update skill proficiency in procedural memory."""
        from proto.memory_pb2_grpc import ProceduralMemoryServiceStub
        from proto.memory_pb2 import UpdateSkillRequest

        stub = await self.discovery.get_stub(
            ServiceType.MEMORY_PROCEDURAL,
            ProceduralMemoryServiceStub,
        )

        if not stub:
            return

        request = UpdateSkillRequest(
            skill_id=skill_id,
            execution_success=success,
            execution_data=execution_data,
        )
        await stub.UpdateSkill(request)
```

### Deliverables
- [ ] Unified memory interface
- [ ] Parallel cross-memory retrieval
- [ ] Context bundle building
- [ ] Experience storage
- [ ] Skill proficiency updates

---

## Sprint 6: End-to-End Testing (Weeks 11-12)

### Goals
- Integration test framework
- Scenario-based testing
- Performance benchmarks
- Chaos testing

### Tasks

#### 6.1 Integration Test Framework

```python
# tests/integration/framework.py
"""Integration test framework for AGI-HPC."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pytest

from agi.integration.pipeline.orchestrator import CognitivePipeline, PipelineContext
from agi.integration.registry import ServiceRegistry
from agi.integration.discovery import ServiceDiscovery
from agi.integration.events.bus import IntegratedEventBus

logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestCase:
    """An integration test case."""

    name: str
    description: str
    input_data: dict
    expected_stages: list[str]
    expected_outcome: dict
    timeout_seconds: float = 30.0
    tags: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Integration test result."""

    test_case: IntegrationTestCase
    passed: bool
    duration_seconds: float
    pipeline_context: PipelineContext | None = None
    error: str | None = None
    stage_times: dict[str, float] = field(default_factory=dict)


class IntegrationTestRunner:
    """Runner for integration tests."""

    def __init__(
        self,
        registry: ServiceRegistry,
        discovery: ServiceDiscovery,
        event_bus: IntegratedEventBus,
    ) -> None:
        """Initialize test runner."""
        self.registry = registry
        self.discovery = discovery
        self.event_bus = event_bus
        self.pipeline = CognitivePipeline(discovery, event_bus)

    async def run_test(self, test_case: IntegrationTestCase) -> TestResult:
        """Run a single integration test."""
        start_time = datetime.now()

        try:
            # Run pipeline with timeout
            context = await asyncio.wait_for(
                self.pipeline.process(test_case.input_data),
                timeout=test_case.timeout_seconds,
            )

            # Validate results
            passed = self._validate_result(test_case, context)

            duration = (datetime.now() - start_time).total_seconds()

            return TestResult(
                test_case=test_case,
                passed=passed,
                duration_seconds=duration,
                pipeline_context=context,
                stage_times=self._extract_stage_times(context),
            )

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_case=test_case,
                passed=False,
                duration_seconds=duration,
                error=f"Timeout after {test_case.timeout_seconds}s",
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_case=test_case,
                passed=False,
                duration_seconds=duration,
                error=str(e),
            )

    async def run_suite(
        self,
        test_cases: list[IntegrationTestCase],
        parallel: bool = False,
    ) -> list[TestResult]:
        """Run a suite of integration tests."""
        if parallel:
            tasks = [self.run_test(tc) for tc in test_cases]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in test_cases:
                result = await self.run_test(tc)
                results.append(result)
            return results

    def _validate_result(
        self,
        test_case: IntegrationTestCase,
        context: PipelineContext,
    ) -> bool:
        """Validate pipeline result against expectations."""
        # Check all expected stages were executed
        executed_stages = [s.split("]")[1].strip() for s in context.trace if "]" in s]

        for expected_stage in test_case.expected_stages:
            if not any(expected_stage in s for s in executed_stages):
                logger.error("Missing expected stage: %s", expected_stage)
                return False

        # Check expected outcome
        for key, expected_value in test_case.expected_outcome.items():
            actual_value = self._get_nested(context, key)
            if actual_value != expected_value:
                logger.error(
                    "Outcome mismatch for %s: expected %s, got %s",
                    key, expected_value, actual_value,
                )
                return False

        # Check no errors
        if context.errors:
            logger.error("Pipeline errors: %s", context.errors)
            return False

        return True

    def _extract_stage_times(self, context: PipelineContext) -> dict[str, float]:
        """Extract timing for each stage."""
        # Parse trace to extract stage durations
        times = {}
        # ... implementation ...
        return times

    @staticmethod
    def _get_nested(obj: Any, path: str) -> Any:
        """Get nested value by dot-separated path."""
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            elif hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                return None
        return obj


# Example test cases
INTEGRATION_TEST_CASES = [
    IntegrationTestCase(
        name="simple_perception_to_action",
        description="Basic perception through to action execution",
        input_data={
            "sensors": {
                "camera": {"objects": ["cup", "table"]},
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        expected_stages=[
            "Perception started",
            "Memory retrieval started",
            "Reasoning started",
            "Planning started",
            "Safety check started",
        ],
        expected_outcome={
            "safety_decision.decision": "ALLOW",
        },
        tags=["smoke", "e2e"],
    ),
    IntegrationTestCase(
        name="safety_denial",
        description="Action denied by safety system",
        input_data={
            "sensors": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "context": {
                "collision_proximity": 0.01,  # Very close!
            },
        },
        expected_stages=[
            "Perception started",
            "Safety check started",
        ],
        expected_outcome={
            "safety_decision.decision": "DENY",
        },
        tags=["safety", "e2e"],
    ),
]


# Pytest integration
@pytest.fixture
async def integration_runner():
    """Create integration test runner."""
    registry = ServiceRegistry()
    await registry.start()

    discovery = ServiceDiscovery(registry)
    event_bus = IntegratedEventBus()
    await event_bus.start()

    runner = IntegrationTestRunner(registry, discovery, event_bus)

    yield runner

    await event_bus.stop()
    await registry.stop()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_perception_to_action(integration_runner):
    """Test basic perception to action pipeline."""
    test_case = INTEGRATION_TEST_CASES[0]
    result = await integration_runner.run_test(test_case)

    assert result.passed, f"Test failed: {result.error}"
    assert result.duration_seconds < test_case.timeout_seconds
```

#### 6.2 Chaos Testing

```python
# tests/integration/chaos.py
"""Chaos testing for AGI-HPC integration."""

from __future__ import annotations

import asyncio
import random
import logging

from agi.integration.registry import ServiceRegistry, ServiceType, ServiceStatus

logger = logging.getLogger(__name__)


class ChaosMonkey:
    """Chaos testing for integration resilience."""

    def __init__(self, registry: ServiceRegistry) -> None:
        """Initialize chaos monkey."""
        self.registry = registry
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(
        self,
        failure_probability: float = 0.1,
        interval_seconds: float = 5.0,
    ) -> None:
        """Start chaos testing."""
        self._running = True
        self._task = asyncio.create_task(
            self._chaos_loop(failure_probability, interval_seconds)
        )
        logger.warning("Chaos monkey started!")

    async def stop(self) -> None:
        """Stop chaos testing."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _chaos_loop(
        self,
        failure_probability: float,
        interval_seconds: float,
    ) -> None:
        """Main chaos loop."""
        while self._running:
            await asyncio.sleep(interval_seconds)

            if random.random() < failure_probability:
                await self._inject_failure()

    async def _inject_failure(self) -> None:
        """Inject a random failure."""
        failure_types = [
            self._fail_random_service,
            self._slow_random_service,
            self._partition_service,
        ]

        failure = random.choice(failure_types)
        await failure()

    async def _fail_random_service(self) -> None:
        """Mark a random service as unhealthy."""
        service_type = random.choice(list(ServiceType))
        endpoints = self.registry.get_all(service_type, include_unhealthy=True)

        if endpoints:
            endpoint = random.choice(endpoints)
            endpoint.status = ServiceStatus.UNHEALTHY
            logger.warning(
                "CHAOS: Failed service %s:%s",
                service_type.value, endpoint.instance_id,
            )

    async def _slow_random_service(self) -> None:
        """Simulate slow service (degraded)."""
        service_type = random.choice(list(ServiceType))
        endpoints = self.registry.get_all(service_type)

        if endpoints:
            endpoint = random.choice(endpoints)
            endpoint.status = ServiceStatus.DEGRADED
            logger.warning(
                "CHAOS: Degraded service %s:%s",
                service_type.value, endpoint.instance_id,
            )

    async def _partition_service(self) -> None:
        """Simulate network partition."""
        # Mark multiple services as unhealthy
        for _ in range(random.randint(1, 3)):
            await self._fail_random_service()
```

### Deliverables
- [ ] Integration test framework
- [ ] Scenario-based test cases
- [ ] Chaos testing framework
- [ ] Performance benchmarks
- [ ] CI/CD integration

---

## Sprint 7: Observability (Weeks 13-14)

### Goals
- Distributed tracing
- Metrics aggregation
- Centralized logging
- Health dashboards

### Tasks

#### 7.1 Distributed Tracing

```python
# src/agi/integration/observability/tracing.py
"""Distributed tracing for AGI-HPC."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


class TracingService:
    """Distributed tracing for cognitive pipeline."""

    def __init__(
        self,
        service_name: str = "agi-hpc",
        otlp_endpoint: str = "localhost:4317",
    ) -> None:
        """Initialize tracing."""
        self.service_name = service_name

        # Set up OpenTelemetry
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        self.tracer = trace.get_tracer(service_name)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[trace.Span]:
        """Create a traced span."""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    @asynccontextmanager
    async def pipeline_trace(
        self,
        pipeline_id: str,
    ) -> AsyncIterator[trace.Span]:
        """Trace entire cognitive pipeline."""
        async with self.span(
            "cognitive_pipeline",
            {"pipeline.id": pipeline_id},
        ) as span:
            yield span

    @asynccontextmanager
    async def stage_trace(
        self,
        stage_name: str,
        pipeline_id: str,
    ) -> AsyncIterator[trace.Span]:
        """Trace a pipeline stage."""
        async with self.span(
            f"stage.{stage_name}",
            {
                "pipeline.id": pipeline_id,
                "stage.name": stage_name,
            },
        ) as span:
            yield span

    @asynccontextmanager
    async def service_call_trace(
        self,
        service_type: str,
        method: str,
    ) -> AsyncIterator[trace.Span]:
        """Trace a service call."""
        async with self.span(
            f"grpc.{service_type}.{method}",
            {
                "service.type": service_type,
                "rpc.method": method,
            },
        ) as span:
            yield span
```

#### 7.2 Metrics Aggregation

```python
# src/agi/integration/observability/metrics.py
"""Metrics aggregation for AGI-HPC."""

from __future__ import annotations

import logging
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway

logger = logging.getLogger(__name__)


class MetricsService:
    """Centralized metrics for AGI-HPC."""

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        pushgateway_url: str | None = None,
    ) -> None:
        """Initialize metrics."""
        self.registry = registry or CollectorRegistry()
        self.pushgateway_url = pushgateway_url

        # Pipeline metrics
        self.pipeline_duration = Histogram(
            "agi_pipeline_duration_seconds",
            "Cognitive pipeline duration",
            ["stage"],
            registry=self.registry,
        )

        self.pipeline_count = Counter(
            "agi_pipeline_total",
            "Total pipeline executions",
            ["status"],
            registry=self.registry,
        )

        # Service metrics
        self.service_calls = Counter(
            "agi_service_calls_total",
            "Total service calls",
            ["service_type", "method", "status"],
            registry=self.registry,
        )

        self.service_latency = Histogram(
            "agi_service_latency_seconds",
            "Service call latency",
            ["service_type", "method"],
            registry=self.registry,
        )

        # Safety metrics
        self.safety_decisions = Counter(
            "agi_safety_decisions_total",
            "Safety decisions by type",
            ["decision"],
            registry=self.registry,
        )

        self.safety_violations = Counter(
            "agi_safety_violations_total",
            "Safety violations by rule",
            ["rule_id"],
            registry=self.registry,
        )

        # Memory metrics
        self.memory_retrievals = Counter(
            "agi_memory_retrievals_total",
            "Memory retrievals by type",
            ["memory_type"],
            registry=self.registry,
        )

        self.memory_retrieval_latency = Histogram(
            "agi_memory_retrieval_latency_seconds",
            "Memory retrieval latency",
            ["memory_type"],
            registry=self.registry,
        )

        # Active gauges
        self.active_pipelines = Gauge(
            "agi_active_pipelines",
            "Currently active pipelines",
            registry=self.registry,
        )

        self.healthy_services = Gauge(
            "agi_healthy_services",
            "Number of healthy services",
            ["service_type"],
            registry=self.registry,
        )

    def record_pipeline_stage(self, stage: str, duration: float) -> None:
        """Record pipeline stage duration."""
        self.pipeline_duration.labels(stage=stage).observe(duration)

    def record_pipeline_complete(self, success: bool) -> None:
        """Record pipeline completion."""
        status = "success" if success else "failure"
        self.pipeline_count.labels(status=status).inc()

    def record_service_call(
        self,
        service_type: str,
        method: str,
        success: bool,
        latency: float,
    ) -> None:
        """Record service call."""
        status = "success" if success else "failure"
        self.service_calls.labels(
            service_type=service_type,
            method=method,
            status=status,
        ).inc()
        self.service_latency.labels(
            service_type=service_type,
            method=method,
        ).observe(latency)

    def record_safety_decision(self, decision: str) -> None:
        """Record safety decision."""
        self.safety_decisions.labels(decision=decision).inc()

    def record_safety_violation(self, rule_id: str) -> None:
        """Record safety violation."""
        self.safety_violations.labels(rule_id=rule_id).inc()

    def push_metrics(self, job: str = "agi-hpc") -> None:
        """Push metrics to gateway."""
        if self.pushgateway_url:
            push_to_gateway(
                self.pushgateway_url,
                job=job,
                registry=self.registry,
            )
```

### Deliverables
- [ ] OpenTelemetry tracing integration
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Centralized logging with ELK/Loki
- [ ] Alerting rules

---

## Sprint 8: Production Deployment (Weeks 15-16)

### Goals
- Kubernetes deployment
- Configuration management
- Rolling updates
- Disaster recovery

### Tasks

#### 8.1 Kubernetes Deployment

```yaml
# deploy/kubernetes/agi-hpc-system.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agi-hpc
---
# Service Registry
apiVersion: apps/v1
kind: Deployment
metadata:
  name: service-registry
  namespace: agi-hpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: service-registry
  template:
    metadata:
      labels:
        app: service-registry
    spec:
      containers:
        - name: registry
          image: agi-hpc/service-registry:latest
          ports:
            - containerPort: 8500
          env:
            - name: CONSUL_BIND_ADDR
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
---
# Event Fabric Broker
apiVersion: apps/v1
kind: Deployment
metadata:
  name: event-fabric
  namespace: agi-hpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: event-fabric
  template:
    metadata:
      labels:
        app: event-fabric
    spec:
      containers:
        - name: broker
          image: agi-hpc/event-fabric:latest
          ports:
            - containerPort: 5555
            - containerPort: 5556
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2"
---
# LH Services
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lh-deliberative
  namespace: agi-hpc
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lh-deliberative
  template:
    metadata:
      labels:
        app: lh-deliberative
        subsystem: lh
    spec:
      containers:
        - name: deliberative
          image: agi-hpc/lh-deliberative:latest
          ports:
            - containerPort: 50051
          env:
            - name: REGISTRY_URL
              value: "http://service-registry:8500"
            - name: EVENT_BROKER
              value: "tcp://event-fabric:5555"
          livenessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 10
            periodSeconds: 5
          readinessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 5
            periodSeconds: 3
---
# RH Services
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rh-perception
  namespace: agi-hpc
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rh-perception
  template:
    metadata:
      labels:
        app: rh-perception
        subsystem: rh
    spec:
      containers:
        - name: perception
          image: agi-hpc/rh-perception:latest
          ports:
            - containerPort: 50052
          resources:
            requests:
              memory: "1Gi"
              cpu: "1"
              nvidia.com/gpu: "1"
            limits:
              memory: "4Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
---
# Safety Gateway
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-gateway
  namespace: agi-hpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safety-gateway
  template:
    metadata:
      labels:
        app: safety-gateway
        subsystem: safety
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: safety-gateway
              topologyKey: kubernetes.io/hostname
      containers:
        - name: gateway
          image: agi-hpc/safety-gateway:latest
          ports:
            - containerPort: 50053
          env:
            - name: ERISML_ENDPOINT
              value: "erisml-service:50052"
            - name: POSTGRES_DSN
              valueFrom:
                secretKeyRef:
                  name: safety-secrets
                  key: postgres-dsn
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2"
---
# Memory Services
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: memory-semantic
  namespace: agi-hpc
spec:
  serviceName: memory-semantic
  replicas: 2
  selector:
    matchLabels:
      app: memory-semantic
  template:
    metadata:
      labels:
        app: memory-semantic
        subsystem: memory
    spec:
      containers:
        - name: semantic
          image: agi-hpc/memory-semantic:latest
          ports:
            - containerPort: 50054
          env:
            - name: QDRANT_URL
              value: "http://qdrant:6333"
          volumeMounts:
            - name: cache
              mountPath: /cache
  volumeClaimTemplates:
    - metadata:
        name: cache
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

#### 8.2 Helm Chart

```yaml
# deploy/helm/agi-hpc/values.yaml
global:
  imageRegistry: ""
  imagePullSecrets: []
  environment: production

serviceRegistry:
  enabled: true
  replicas: 3
  image:
    repository: agi-hpc/service-registry
    tag: latest

eventFabric:
  enabled: true
  replicas: 3
  backend: zmq  # zmq, redis, nats
  image:
    repository: agi-hpc/event-fabric
    tag: latest

lh:
  deliberative:
    enabled: true
    replicas: 2
  planning:
    enabled: true
    replicas: 2
  language:
    enabled: true
    replicas: 1

rh:
  perception:
    enabled: true
    replicas: 2
    gpu: true
  worldModel:
    enabled: true
    replicas: 1
  simulation:
    enabled: true
    replicas: 1

memory:
  semantic:
    enabled: true
    replicas: 2
    qdrantUrl: "http://qdrant:6333"
  episodic:
    enabled: true
    replicas: 2
    postgresUrl: ""
  procedural:
    enabled: true
    replicas: 1

metacognition:
  enabled: true
  replicas: 1

safety:
  gateway:
    enabled: true
    replicas: 3
  preAction:
    enabled: true
    replicas: 2
  inAction:
    enabled: true
    replicas: 2
  postAction:
    enabled: true
    replicas: 1
  erisml:
    enabled: true
    endpoint: "erisml-service:50052"

observability:
  tracing:
    enabled: true
    jaegerEndpoint: "jaeger:6831"
  metrics:
    enabled: true
    prometheusEndpoint: "prometheus:9090"
  logging:
    enabled: true
    lokiEndpoint: "loki:3100"

# External dependencies
postgresql:
  enabled: true
  auth:
    postgresPassword: ""
    database: agi

redis:
  enabled: true
  auth:
    password: ""

qdrant:
  enabled: true
  persistence:
    size: 100Gi
```

### Deliverables
- [ ] Kubernetes manifests
- [ ] Helm chart
- [ ] ConfigMaps and Secrets management
- [ ] Rolling update strategy
- [ ] Backup and restore procedures
- [ ] Disaster recovery runbook

---

## File Structure

```
src/agi/integration/
├── __init__.py
├── registry.py                 # Service registry
├── discovery.py                # Service discovery
├── config.py                   # Configuration management
├── events/
│   ├── __init__.py
│   ├── schema.py              # Event schemas
│   └── bus.py                 # Event bus
├── adapters/
│   ├── __init__.py
│   ├── lh_adapter.py          # LH event adapter
│   ├── rh_adapter.py          # RH event adapter
│   ├── memory_adapter.py      # Memory event adapter
│   └── safety_adapter.py      # Safety event adapter
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py        # Cognitive pipeline
├── safety/
│   ├── __init__.py
│   └── integration.py         # Safety integration
├── memory/
│   ├── __init__.py
│   └── unified.py             # Unified memory
└── observability/
    ├── __init__.py
    ├── tracing.py             # Distributed tracing
    ├── metrics.py             # Metrics aggregation
    └── logging.py             # Structured logging

tests/integration/
├── conftest.py
├── framework.py               # Test framework
├── chaos.py                   # Chaos testing
├── test_pipeline.py
├── test_safety_integration.py
├── test_memory_integration.py
└── benchmarks/
    ├── bench_pipeline.py
    └── bench_throughput.py

deploy/
├── kubernetes/
│   ├── agi-hpc-system.yaml
│   ├── configmaps.yaml
│   └── secrets.yaml
├── helm/
│   └── agi-hpc/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
└── docker-compose/
    └── docker-compose.yaml
```

---

## Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
integration = [
    "grpcio>=1.60.0",
    "grpcio-tools>=1.60.0",
    "pyyaml>=6.0.1",
    "httpx>=0.26.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "opentelemetry-exporter-otlp>=1.22.0",
    "prometheus-client>=0.19.0",
    "structlog>=24.1.0",
    "watchdog>=3.0.0",
]
```

---

## Quick Start

```bash
# Start infrastructure
docker compose -f deploy/docker-compose/docker-compose.yaml up -d

# Start all services
python -m agi.integration.server --config config/integration.yaml

# Run integration tests
pytest tests/integration/ -v -m integration

# Run chaos tests
python -m tests.integration.chaos --duration 60

# Deploy to Kubernetes
helm install agi-hpc deploy/helm/agi-hpc -n agi-hpc
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency | <500ms | P95 pipeline completion |
| Service availability | >99.9% | Uptime per service |
| Event delivery | >99.99% | Event fabric reliability |
| Memory retrieval | <100ms | P95 cross-memory query |
| Safety check | <100ms | P99 pre-action check |
| Test coverage | >80% | Integration test coverage |
| Recovery time | <60s | Service restart after failure |
