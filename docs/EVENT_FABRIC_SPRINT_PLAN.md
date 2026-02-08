# Event Fabric Sprint Plan

## Overview

The Event Fabric is the nervous system of AGI-HPC, providing asynchronous communication between all cognitive components. It enables loose coupling, horizontal scaling, and real-time coordination across the distributed architecture.

## Current State Assessment

### Implemented (Functional)
| Backend | Status | Description |
|---------|--------|-------------|
| `LocalBackend` | **Done** | In-process pub/sub, thread-safe, ideal for tests |
| `ZmqBackend` | **Done** | ZeroMQ PUB/SUB, requires external broker |
| `UcxBackend` | **Done** | UCX for HPC (RDMA, SHM), high-performance |

### Current API
```python
fabric = EventFabric(mode="local")  # or "zmq", "ucx"
fabric.subscribe("topic", handler_function)
fabric.publish("topic", {"key": "value"})
fabric.close()
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `AGI_FABRIC_MODE` | `local` | Backend: local, zmq, ucx |
| `AGI_FABRIC_PUB_ENDPOINT` | `tcp://fabric:5556` | ZMQ publish endpoint |
| `AGI_FABRIC_SUB_ENDPOINT` | `tcp://fabric:5555` | ZMQ subscribe endpoint |
| `AGI_FABRIC_UCX_ENDPOINT` | `tcp://fabric:13337` | UCX endpoint |
| `AGI_FABRIC_IDENTITY` | `node` | Node identity |

### Key Gaps
1. **No ZMQ broker** - Clients expect XPUB/XSUB broker
2. **No message persistence** - Events lost on crash
3. **No delivery guarantees** - At-most-once only
4. **No event schemas** - No validation
5. **No dead letter queue** - Failed events lost
6. **No observability** - No metrics/tracing
7. **No replay** - Can't replay historical events
8. **No wildcard subscriptions** - Exact topic match only

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVENT FABRIC ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      PRODUCERS                                       │   │
│   │                                                                      │   │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │   │
│   │   │   LH    │   │   RH    │   │ Safety  │   │  Meta   │            │   │
│   │   │ Planner │   │  World  │   │ Gateway │   │ cognit  │            │   │
│   │   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘            │   │
│   │        │             │             │             │                  │   │
│   └────────┼─────────────┼─────────────┼─────────────┼──────────────────┘   │
│            │             │             │             │                      │
│   ┌────────▼─────────────▼─────────────▼─────────────▼──────────────────┐   │
│   │                        FABRIC CORE                                   │   │
│   │                                                                      │   │
│   │   ┌──────────────────────────────────────────────────────────────┐  │   │
│   │   │                     MESSAGE ROUTER                            │  │   │
│   │   │                                                               │  │   │
│   │   │  • Topic-based routing                                        │  │   │
│   │   │  • Wildcard subscriptions (plan.*, safety.#)                 │  │   │
│   │   │  • Schema validation                                          │  │   │
│   │   │  • Dead letter queue                                          │  │   │
│   │   └──────────────────────────────────────────────────────────────┘  │   │
│   │                               │                                      │   │
│   │   ┌───────────────────────────┴───────────────────────────────────┐  │   │
│   │   │                      BACKENDS                                  │  │   │
│   │   │                                                                │  │   │
│   │   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │  │   │
│   │   │  │  Local   │  │   ZMQ    │  │   UCX    │  │  Redis   │       │  │   │
│   │   │  │ (test)   │  │ (multi)  │  │  (HPC)   │  │ (cloud)  │       │  │   │
│   │   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │  │   │
│   │   │                                                                │  │   │
│   │   │  ┌──────────┐  ┌──────────┐                                   │  │   │
│   │   │  │   NATS   │  │  Kafka   │                                   │  │   │
│   │   │  │(prodctn) │  │ (scale)  │                                   │  │   │
│   │   │  └──────────┘  └──────────┘                                   │  │   │
│   │   └────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                        CONSUMERS                                     │   │
│   │                                                                      │   │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │   │
│   │   │ RH Evnt │   │ Memory  │   │ Episodic│   │ Monitor │            │   │
│   │   │  Loop   │   │ Svc     │   │  Log    │   │   UI    │            │   │
│   │   └─────────┘   └─────────┘   └─────────┘   └─────────┘            │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Event Topics (Current Usage)

| Topic | Publisher | Subscriber | Description |
|-------|-----------|------------|-------------|
| `plan.step_ready` | LH PlanService | RH EventLoop | Plan step ready for execution |
| `simulation.result` | RH SimService | LH PlanService | Simulation completed |
| `perception.state_update` | Environment | RH Perception | New sensor data |
| `memory.episodic.append` | EpisodicMemory | - | Episode stored |
| `memory.semantic.write` | SemanticMemory | - | Fact stored |
| `meta.review` | Metacognition | LH | Review completed |
| `safety.decision` | SafetyGateway | Episodic, Monitor | Safety decision made |

---

## Sprint 1: ZMQ Broker Implementation

**Goal**: Provide standalone ZMQ broker for multi-node deployments.

### Tasks

#### 1.1 XPUB/XSUB Broker
- [ ] Create `fabric_broker.py` standalone process
- [ ] XPUB socket for publishers (frontend)
- [ ] XSUB socket for subscribers (backend)
- [ ] Topic forwarding between sockets
- [ ] Graceful shutdown handling

```python
# src/agi/core/events/broker.py
import zmq
import logging
import signal
import sys

logger = logging.getLogger(__name__)

class FabricBroker:
    """
    ZeroMQ XPUB/XSUB broker for Event Fabric.

    Publishers connect to frontend (XSUB).
    Subscribers connect to backend (XPUB).
    Broker forwards messages between them.
    """

    def __init__(
        self,
        frontend_addr: str = "tcp://*:5556",  # Publishers connect here
        backend_addr: str = "tcp://*:5555",   # Subscribers connect here
    ):
        self.frontend_addr = frontend_addr
        self.backend_addr = backend_addr

        self.ctx = zmq.Context.instance()

        # XSUB receives from publishers
        self.frontend = self.ctx.socket(zmq.XSUB)
        self.frontend.bind(frontend_addr)

        # XPUB sends to subscribers
        self.backend = self.ctx.socket(zmq.XPUB)
        self.backend.bind(backend_addr)

        self._running = False
        logger.info(
            "[broker] initialized frontend=%s backend=%s",
            frontend_addr,
            backend_addr,
        )

    def run(self):
        """Run the broker (blocking)."""
        self._running = True
        logger.info("[broker] starting proxy loop")

        try:
            # zmq.proxy blocks until context is terminated
            zmq.proxy(self.frontend, self.backend)
        except zmq.ContextTerminated:
            logger.info("[broker] context terminated")
        except Exception:
            logger.exception("[broker] proxy error")
        finally:
            self._running = False

    def stop(self):
        """Stop the broker."""
        logger.info("[broker] stopping")
        self.ctx.term()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Event Fabric ZMQ Broker")
    parser.add_argument("--frontend", default="tcp://*:5556", help="Publisher endpoint")
    parser.add_argument("--backend", default="tcp://*:5555", help="Subscriber endpoint")
    args = parser.parse_args()

    broker = FabricBroker(
        frontend_addr=args.frontend,
        backend_addr=args.backend,
    )

    def signal_handler(sig, frame):
        broker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    broker.run()

if __name__ == "__main__":
    main()
```

#### 1.2 Docker container for broker
- [ ] Create `Dockerfile.fabric-broker`
- [ ] Health check endpoint
- [ ] Metrics endpoint
- [ ] Docker Compose integration

```dockerfile
# docker/Dockerfile.fabric-broker
FROM python:3.11-slim

RUN pip install pyzmq

COPY src/agi/core/events/broker.py /app/broker.py

EXPOSE 5555 5556

CMD ["python", "/app/broker.py"]
```

#### 1.3 Broker high availability
- [ ] Multiple broker instances
- [ ] Client failover logic
- [ ] Broker discovery (DNS/Consul)

### Acceptance Criteria
```bash
# Start broker
python -m agi.core.events.broker --frontend tcp://*:5556 --backend tcp://*:5555

# Start publisher
AGI_FABRIC_MODE=zmq AGI_FABRIC_PUB_ENDPOINT=tcp://localhost:5556 \
  python -c "
from agi.core.events.fabric import EventFabric
f = EventFabric(mode='zmq')
f.publish('test.topic', {'msg': 'hello'})
"

# Start subscriber (in another terminal)
AGI_FABRIC_MODE=zmq AGI_FABRIC_SUB_ENDPOINT=tcp://localhost:5555 \
  python -c "
from agi.core.events.fabric import EventFabric
import time
f = EventFabric(mode='zmq')
f.subscribe('test.topic', lambda m: print(f'Received: {m}'))
time.sleep(10)
"
```

---

## Sprint 2: Redis Streams Backend

**Goal**: Add Redis Streams backend for persistent messaging and replay.

### Tasks

#### 2.1 Redis Streams adapter
- [ ] Implement `RedisBackend` class
- [ ] XADD for publishing
- [ ] XREAD with consumer groups
- [ ] Message acknowledgment
- [ ] Stream trimming (MAXLEN)

```python
# src/agi/core/events/redis_backend.py
import json
import threading
import logging
from typing import Dict, List, Optional
import redis

logger = logging.getLogger(__name__)

class RedisBackend:
    """
    Redis Streams backend for Event Fabric.

    Features:
    - Persistent message storage
    - Consumer groups for load balancing
    - Message acknowledgment
    - Replay from any point
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        stream_prefix: str = "fabric:",
        consumer_group: str = "agi-hpc",
        consumer_name: Optional[str] = None,
        max_stream_length: int = 10000,
    ):
        self.client = redis.from_url(url)
        self.stream_prefix = stream_prefix
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"consumer-{threading.current_thread().ident}"
        self.max_stream_length = max_stream_length

        self._subscribers: Dict[str, List] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None

        logger.info("[fabric][redis] initialized url=%s", url)

    def publish(self, topic: str, message: dict) -> str:
        """Publish message to Redis Stream."""
        stream_key = f"{self.stream_prefix}{topic}"
        payload = json.dumps(message)

        message_id = self.client.xadd(
            stream_key,
            {"data": payload},
            maxlen=self.max_stream_length,
            approximate=True,
        )

        logger.debug("[fabric][redis] published topic=%s id=%s", topic, message_id)
        return message_id.decode() if isinstance(message_id, bytes) else message_id

    def subscribe(self, topic: str, handler) -> None:
        """Subscribe to topic with consumer group."""
        stream_key = f"{self.stream_prefix}{topic}"

        # Create consumer group if not exists
        try:
            self.client.xgroup_create(
                stream_key,
                self.consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

        # Start consumer thread if not running
        if self._recv_thread is None or not self._recv_thread.is_alive():
            self._recv_thread = threading.Thread(
                target=self._consume_loop,
                daemon=True,
            )
            self._recv_thread.start()

        logger.info("[fabric][redis] subscribed topic=%s", topic)

    def _consume_loop(self) -> None:
        """Consumer loop reading from all subscribed streams."""
        logger.info("[fabric][redis] consumer loop started")

        while not self._stop_event.is_set():
            with self._lock:
                topics = list(self._subscribers.keys())

            if not topics:
                self._stop_event.wait(1.0)
                continue

            streams = {f"{self.stream_prefix}{t}": ">" for t in topics}

            try:
                messages = self.client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    streams,
                    count=100,
                    block=1000,
                )
            except Exception:
                logger.exception("[fabric][redis] xreadgroup error")
                continue

            for stream_key, stream_messages in messages or []:
                topic = stream_key.decode().replace(self.stream_prefix, "")
                self._process_messages(topic, stream_messages)

        logger.info("[fabric][redis] consumer loop exiting")

    def _process_messages(self, topic: str, messages: list) -> None:
        """Process messages and acknowledge."""
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))

        stream_key = f"{self.stream_prefix}{topic}"

        for message_id, fields in messages:
            try:
                data = json.loads(fields[b"data"].decode())
            except Exception:
                logger.exception("[fabric][redis] decode error id=%s", message_id)
                continue

            for handler in handlers:
                try:
                    handler(data)
                except Exception:
                    logger.exception("[fabric][redis] handler error topic=%s", topic)

            # Acknowledge message
            self.client.xack(stream_key, self.consumer_group, message_id)

    def replay(
        self,
        topic: str,
        start_id: str = "0",
        end_id: str = "+",
        count: int = 100,
    ) -> List[dict]:
        """Replay messages from stream history."""
        stream_key = f"{self.stream_prefix}{topic}"
        messages = self.client.xrange(stream_key, start_id, end_id, count=count)

        result = []
        for message_id, fields in messages:
            try:
                data = json.loads(fields[b"data"].decode())
                data["_message_id"] = message_id.decode()
                result.append(data)
            except Exception:
                continue

        return result

    def close(self) -> None:
        """Close the backend."""
        self._stop_event.set()
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        self.client.close()
        logger.info("[fabric][redis] closed")
```

#### 2.2 Message replay API
- [ ] `replay(topic, start_time, end_time)` method
- [ ] `get_stream_info(topic)` method
- [ ] Export to file (JSONL)

#### 2.3 Dead Letter Queue
- [ ] Failed message capture
- [ ] Retry logic
- [ ] DLQ monitoring

#### 2.4 Configuration
- [ ] `AGI_FABRIC_REDIS_URL` env var
- [ ] Stream max length
- [ ] Consumer group settings

---

## Sprint 3: Event Schemas and Validation

**Goal**: Define and enforce event schemas for type safety.

### Tasks

#### 3.1 Schema registry
- [ ] Define schemas using Pydantic or dataclasses
- [ ] Schema versioning
- [ ] Backward compatibility checks

```python
# src/agi/core/events/schemas.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

@dataclass
class EventEnvelope:
    """Standard envelope for all events."""
    event_id: str           # UUID
    event_type: str         # e.g., "plan.step_ready"
    source: str             # Publisher identity
    timestamp: datetime     # ISO 8601
    version: str            # Schema version
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    payload: Dict[str, Any] = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload,
        }

# Topic-specific schemas
@dataclass
class PlanStepReadyEvent:
    """plan.step_ready event payload."""
    plan_id: str
    step_id: str
    step_index: int
    description: str
    tool_id: Optional[str] = None
    params: Dict[str, str] = None
    safety_tags: List[str] = None

@dataclass
class SimulationResultEvent:
    """simulation.result event payload."""
    plan_id: str
    overall_risk: float
    approved: bool
    step_risks: List[float] = None
    violations: List[str] = None

@dataclass
class SafetyDecisionEvent:
    """safety.decision event payload."""
    plan_id: str
    step_id: str
    decision: str  # ALLOW, BLOCK, REVISE
    bond_index: Optional[float] = None
    reasons: List[str] = None
    proof_hash: Optional[str] = None

# Schema registry
EVENT_SCHEMAS = {
    "plan.step_ready": PlanStepReadyEvent,
    "simulation.result": SimulationResultEvent,
    "safety.decision": SafetyDecisionEvent,
}
```

#### 3.2 Validation middleware
- [ ] Validate on publish
- [ ] Validate on subscribe (optional)
- [ ] Schema mismatch logging

```python
# src/agi/core/events/validation.py
from dataclasses import fields, is_dataclass
from typing import Type, Any
import logging

logger = logging.getLogger(__name__)

class EventValidator:
    """Validates events against registered schemas."""

    def __init__(self, schemas: dict):
        self.schemas = schemas

    def validate(self, topic: str, payload: dict) -> bool:
        """Validate payload against schema for topic."""
        schema = self.schemas.get(topic)
        if schema is None:
            # No schema registered, allow
            return True

        try:
            self._validate_dataclass(schema, payload)
            return True
        except ValidationError as e:
            logger.warning("[validator] %s: %s", topic, e)
            return False

    def _validate_dataclass(self, schema: Type, payload: dict) -> None:
        """Validate payload matches dataclass schema."""
        if not is_dataclass(schema):
            raise ValidationError(f"Schema {schema} is not a dataclass")

        for field in fields(schema):
            if field.name not in payload:
                if field.default is field.default_factory is None:
                    raise ValidationError(f"Missing required field: {field.name}")
            else:
                # Type checking (basic)
                value = payload[field.name]
                # Could add more sophisticated type checking here

class ValidationError(Exception):
    pass
```

#### 3.3 Schema evolution
- [ ] Version field in envelope
- [ ] Migration helpers
- [ ] Deprecation warnings

---

## Sprint 4: Wildcard Subscriptions

**Goal**: Support pattern-based topic subscriptions.

### Tasks

#### 4.1 Topic patterns
- [ ] Single-level wildcard: `*` (e.g., `plan.*`)
- [ ] Multi-level wildcard: `#` (e.g., `safety.#`)
- [ ] Pattern matching engine

```python
# src/agi/core/events/patterns.py
import re
from typing import Set

class TopicMatcher:
    """Matches topics against subscription patterns."""

    def __init__(self):
        self._patterns: Set[str] = set()
        self._compiled: dict = {}

    def add_pattern(self, pattern: str) -> None:
        """Add subscription pattern."""
        self._patterns.add(pattern)
        self._compiled[pattern] = self._compile_pattern(pattern)

    def matches(self, topic: str) -> Set[str]:
        """Return all patterns that match the topic."""
        matching = set()
        for pattern, regex in self._compiled.items():
            if regex.match(topic):
                matching.add(pattern)
        return matching

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Convert pattern to regex.

        * matches exactly one level (no dots)
        # matches zero or more levels (including dots)
        """
        # Escape dots
        regex = pattern.replace(".", r"\.")
        # * matches one level
        regex = regex.replace("*", r"[^.]+")
        # # matches zero or more levels
        regex = regex.replace("#", r".*")
        return re.compile(f"^{regex}$")

# Usage example:
# matcher = TopicMatcher()
# matcher.add_pattern("plan.*")           # matches plan.created, plan.updated
# matcher.add_pattern("safety.#")         # matches safety.decision, safety.pre.check
# matcher.add_pattern("memory.semantic.write")  # exact match
```

#### 4.2 Backend integration
- [ ] Update LocalBackend
- [ ] Update ZmqBackend (use ZMQ subscriptions)
- [ ] Update RedisBackend (pattern streams)

#### 4.3 Subscription management
- [ ] Unsubscribe API
- [ ] List active subscriptions
- [ ] Subscription metrics

---

## Sprint 5: Observability

**Goal**: Add comprehensive monitoring and tracing.

### Tasks

#### 5.1 Prometheus metrics
- [ ] `fabric_messages_published_total` (counter, by topic)
- [ ] `fabric_messages_received_total` (counter, by topic)
- [ ] `fabric_message_latency_seconds` (histogram)
- [ ] `fabric_handler_errors_total` (counter)
- [ ] `fabric_queue_depth` (gauge, for Redis)

```python
# src/agi/core/events/metrics.py
from prometheus_client import Counter, Histogram, Gauge

MESSAGES_PUBLISHED = Counter(
    "fabric_messages_published_total",
    "Total messages published",
    ["topic", "backend"],
)

MESSAGES_RECEIVED = Counter(
    "fabric_messages_received_total",
    "Total messages received",
    ["topic", "backend"],
)

MESSAGE_LATENCY = Histogram(
    "fabric_message_latency_seconds",
    "Message latency from publish to receive",
    ["topic"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

HANDLER_ERRORS = Counter(
    "fabric_handler_errors_total",
    "Total handler errors",
    ["topic"],
)

QUEUE_DEPTH = Gauge(
    "fabric_queue_depth",
    "Current queue depth",
    ["topic"],
)

class MetricsMiddleware:
    """Middleware for collecting fabric metrics."""

    def __init__(self, backend, backend_name: str):
        self.backend = backend
        self.backend_name = backend_name

    def publish(self, topic: str, message: dict) -> None:
        message["_publish_time"] = time.time()
        self.backend.publish(topic, message)
        MESSAGES_PUBLISHED.labels(topic=topic, backend=self.backend_name).inc()

    def subscribe(self, topic: str, handler) -> None:
        def wrapped_handler(message: dict):
            MESSAGES_RECEIVED.labels(topic=topic, backend=self.backend_name).inc()

            publish_time = message.pop("_publish_time", None)
            if publish_time:
                latency = time.time() - publish_time
                MESSAGE_LATENCY.labels(topic=topic).observe(latency)

            try:
                handler(message)
            except Exception as e:
                HANDLER_ERRORS.labels(topic=topic).inc()
                raise

        self.backend.subscribe(topic, wrapped_handler)
```

#### 5.2 Distributed tracing
- [ ] OpenTelemetry integration
- [ ] Trace context propagation
- [ ] Span creation for publish/receive

```python
# src/agi/core/events/tracing.py
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer(__name__)

class TracingMiddleware:
    """Middleware for distributed tracing."""

    def __init__(self, backend):
        self.backend = backend

    def publish(self, topic: str, message: dict) -> None:
        with tracer.start_as_current_span(
            f"fabric.publish.{topic}",
            kind=trace.SpanKind.PRODUCER,
        ) as span:
            span.set_attribute("messaging.destination", topic)

            # Inject trace context into message
            carrier = {}
            inject(carrier)
            message["_trace_context"] = carrier

            self.backend.publish(topic, message)

    def subscribe(self, topic: str, handler) -> None:
        def traced_handler(message: dict):
            carrier = message.pop("_trace_context", {})
            ctx = extract(carrier)

            with tracer.start_as_current_span(
                f"fabric.receive.{topic}",
                context=ctx,
                kind=trace.SpanKind.CONSUMER,
            ) as span:
                span.set_attribute("messaging.destination", topic)
                handler(message)

        self.backend.subscribe(topic, traced_handler)
```

#### 5.3 Structured logging
- [ ] Event ID in logs
- [ ] Correlation ID propagation
- [ ] Log aggregation integration

---

## Sprint 6: NATS JetStream Backend

**Goal**: Add production-grade NATS backend with JetStream.

### Tasks

#### 6.1 NATS client integration
- [ ] Implement `NatsBackend` class
- [ ] JetStream for persistence
- [ ] Consumer groups (queue groups)
- [ ] Acknowledgment modes

```python
# src/agi/core/events/nats_backend.py
import json
import asyncio
import threading
import logging
from typing import Dict, List, Optional
import nats
from nats.js import JetStreamContext

logger = logging.getLogger(__name__)

class NatsBackend:
    """
    NATS JetStream backend for Event Fabric.

    Features:
    - At-least-once delivery
    - Persistent streams
    - Consumer groups
    - Replay from any point
    """

    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream_name: str = "AGI_HPC_EVENTS",
        consumer_name: Optional[str] = None,
    ):
        self.servers = servers
        self.stream_name = stream_name
        self.consumer_name = consumer_name or f"consumer-{id(self)}"

        self._nc: Optional[nats.NATS] = None
        self._js: Optional[JetStreamContext] = None
        self._subscribers: Dict[str, List] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Connect
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result()

        logger.info("[fabric][nats] initialized servers=%s", servers)

    async def _connect(self) -> None:
        """Connect to NATS and setup JetStream."""
        self._nc = await nats.connect(servers=self.servers)
        self._js = self._nc.jetstream()

        # Create stream if not exists
        try:
            await self._js.add_stream(
                name=self.stream_name,
                subjects=["fabric.>"],
                retention="limits",
                max_msgs=1000000,
                max_bytes=1024 * 1024 * 1024,  # 1GB
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

        logger.info("[fabric][nats] connected to stream %s", self.stream_name)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def publish(self, topic: str, message: dict) -> None:
        """Publish message to NATS JetStream."""
        async def _publish():
            subject = f"fabric.{topic}"
            payload = json.dumps(message).encode()
            ack = await self._js.publish(subject, payload)
            logger.debug("[fabric][nats] published %s seq=%d", topic, ack.seq)

        asyncio.run_coroutine_threadsafe(_publish(), self._loop)

    def subscribe(self, topic: str, handler) -> None:
        """Subscribe to topic with durable consumer."""
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

        async def _subscribe():
            subject = f"fabric.{topic}"
            consumer_name = f"{self.consumer_name}-{topic.replace('.', '-')}"

            try:
                sub = await self._js.pull_subscribe(
                    subject,
                    durable=consumer_name,
                    stream=self.stream_name,
                )

                # Start consumer loop
                asyncio.create_task(self._consume(topic, sub))

            except Exception:
                logger.exception("[fabric][nats] subscribe error")

        asyncio.run_coroutine_threadsafe(_subscribe(), self._loop)
        logger.info("[fabric][nats] subscribed topic=%s", topic)

    async def _consume(self, topic: str, sub) -> None:
        """Consumer loop for a subscription."""
        while not self._stop_event.is_set():
            try:
                msgs = await sub.fetch(batch=10, timeout=1)
            except asyncio.TimeoutError:
                continue
            except Exception:
                logger.exception("[fabric][nats] fetch error")
                continue

            with self._lock:
                handlers = list(self._subscribers.get(topic, []))

            for msg in msgs:
                try:
                    data = json.loads(msg.data.decode())
                    for handler in handlers:
                        handler(data)
                    await msg.ack()
                except Exception:
                    logger.exception("[fabric][nats] handler error")

    def close(self) -> None:
        """Close the backend."""
        self._stop_event.set()

        async def _close():
            if self._nc:
                await self._nc.close()

        asyncio.run_coroutine_threadsafe(_close(), self._loop).result(timeout=2)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        logger.info("[fabric][nats] closed")
```

#### 6.2 Stream management
- [ ] Stream creation on startup
- [ ] Stream trimming policies
- [ ] Multi-stream routing

#### 6.3 Consumer modes
- [ ] Push consumers (real-time)
- [ ] Pull consumers (batch processing)
- [ ] Queue groups (load balancing)

---

## Sprint 7: HPC Optimizations

**Goal**: Optimize UCX backend for HPC deployments.

### Tasks

#### 7.1 UCX broker/relay
- [ ] Create UCX relay node for many-to-many
- [ ] Connection pooling
- [ ] Automatic reconnection

#### 7.2 Shared memory optimization
- [ ] Detect same-node publishers/subscribers
- [ ] Use SHM transport for local
- [ ] Fallback to RDMA for remote

#### 7.3 Batch publishing
- [ ] Batch multiple events
- [ ] Async flush
- [ ] Configurable batch size/timeout

```python
# src/agi/core/events/batching.py
import asyncio
import threading
from typing import List, Tuple
import time

class BatchPublisher:
    """Batches multiple publishes for efficiency."""

    def __init__(
        self,
        backend,
        max_batch_size: int = 100,
        max_wait_ms: int = 10,
    ):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._batch: List[Tuple[str, dict]] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def publish(self, topic: str, message: dict) -> None:
        """Add message to batch."""
        with self._lock:
            self._batch.append((topic, message))

            if len(self._batch) >= self.max_batch_size:
                self._flush()

    def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while True:
            time.sleep(self.max_wait_ms / 1000.0)
            with self._lock:
                if self._batch and (time.time() - self._last_flush) * 1000 >= self.max_wait_ms:
                    self._flush()

    def _flush(self) -> None:
        """Flush current batch."""
        batch = self._batch
        self._batch = []
        self._last_flush = time.time()

        for topic, message in batch:
            self.backend.publish(topic, message)
```

#### 7.4 Zero-copy optimizations
- [ ] Avoid JSON for local transport
- [ ] Memory views for large payloads
- [ ] GPU-direct for tensor events

---

## Sprint 8: Testing and Documentation

**Goal**: Comprehensive testing and documentation.

### Tasks

#### 8.1 Unit tests
- [ ] `test_local_backend_pubsub`
- [ ] `test_zmq_backend_pubsub`
- [ ] `test_redis_backend_pubsub`
- [ ] `test_nats_backend_pubsub`
- [ ] `test_wildcard_matching`
- [ ] `test_schema_validation`
- [ ] `test_message_replay`

#### 8.2 Integration tests
- [ ] Multi-node ZMQ communication
- [ ] Redis Streams persistence
- [ ] Consumer group load balancing
- [ ] Failover scenarios

#### 8.3 Performance benchmarks
- [ ] Throughput (messages/sec)
- [ ] Latency (p50, p99)
- [ ] Memory usage
- [ ] CPU usage

#### 8.4 Documentation
- [ ] Backend selection guide
- [ ] Configuration reference
- [ ] Schema definition guide
- [ ] Troubleshooting guide

---

## File Structure After Completion

```
src/agi/core/events/
├── __init__.py
├── fabric.py              # EventFabric facade
├── broker.py              # ZMQ XPUB/XSUB broker
├── backends/
│   ├── __init__.py
│   ├── base.py            # FabricBackend protocol
│   ├── local.py           # LocalBackend
│   ├── zmq.py             # ZmqBackend
│   ├── ucx.py             # UcxBackend
│   ├── redis.py           # RedisBackend
│   └── nats.py            # NatsBackend
├── schemas.py             # Event schemas
├── validation.py          # Schema validation
├── patterns.py            # Wildcard topic matching
├── metrics.py             # Prometheus metrics
├── tracing.py             # OpenTelemetry tracing
├── batching.py            # Batch publishing
└── config.py              # Configuration

tests/core/events/
├── __init__.py
├── conftest.py            # Fixtures
├── test_local_backend.py
├── test_zmq_backend.py
├── test_redis_backend.py
├── test_nats_backend.py
├── test_patterns.py
├── test_validation.py
└── benchmarks/
    ├── bench_throughput.py
    └── bench_latency.py

docker/
├── Dockerfile.fabric-broker
├── docker-compose.fabric.yaml

configs/
└── fabric_config.yaml     # Fabric configuration
```

---

## Docker Compose for Development

```yaml
# docker-compose.fabric.yaml
version: '3.8'

services:
  fabric-broker:
    build:
      context: .
      dockerfile: docker/Dockerfile.fabric-broker
    ports:
      - "5555:5555"  # Subscribers
      - "5556:5556"  # Publishers
    healthcheck:
      test: ["CMD", "python", "-c", "import zmq; zmq.Context().socket(zmq.REQ).connect('tcp://localhost:5555')"]
      interval: 10s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  nats:
    image: nats:2.10-alpine
    ports:
      - "4222:4222"  # Client
      - "8222:8222"  # Monitoring
    command: ["--jetstream", "--store_dir=/data"]
    volumes:
      - nats-data:/data

volumes:
  redis-data:
  nats-data:
```

---

## Priority Order

1. **Sprint 1** - Critical: ZMQ broker enables multi-node
2. **Sprint 2** - High: Redis provides persistence
3. **Sprint 5** - High: Observability for operations
4. **Sprint 4** - Medium: Wildcards improve flexibility
5. **Sprint 3** - Medium: Schemas for safety
6. **Sprint 6** - Medium: NATS for production
7. **Sprint 7** - Low: HPC optimizations
8. **Sprint 8** - Ongoing: Testing

---

## Quick Start (After Sprint 1-2)

```bash
# Terminal 1: Start Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 2: Start ZMQ broker
python -m agi.core.events.broker

# Terminal 3: Publisher
export AGI_FABRIC_MODE=redis
python -c "
from agi.core.events.fabric import EventFabric
import time

fabric = EventFabric(mode='redis')
for i in range(10):
    fabric.publish('test.events', {'counter': i, 'time': time.time()})
    time.sleep(1)
"

# Terminal 4: Subscriber
export AGI_FABRIC_MODE=redis
python -c "
from agi.core.events.fabric import EventFabric
import time

fabric = EventFabric(mode='redis')
fabric.subscribe('test.events', lambda m: print(f'Received: {m}'))
time.sleep(30)
"

# Terminal 5: Replay
python -c "
from agi.core.events.redis_backend import RedisBackend
backend = RedisBackend()
messages = backend.replay('test.events', count=10)
for m in messages:
    print(m)
"
```

---

## Dependencies

```toml
# pyproject.toml additions for fabric
[project.optional-dependencies]
fabric = [
    # ZeroMQ
    "pyzmq>=25.0",

    # Redis
    "redis>=5.0",

    # NATS
    "nats-py>=2.6",

    # UCX (HPC)
    "ucx-py>=0.35",

    # Observability
    "prometheus-client>=0.19",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
]
```
