# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
NATS JetStream backend for Event Fabric.

Provides persistent messaging with:
- JetStream persistent message storage
- At-least-once delivery via pull consumers
- Durable consumers with batch fetch
- Queue groups for load balancing
- Stream management (create, delete, info)

Environment Variables:
    AGI_FABRIC_NATS_URL         NATS URL (default: nats://localhost:4222)
    AGI_FABRIC_NATS_STREAM      JetStream stream name (default: AGI_HPC_EVENTS)
    AGI_FABRIC_CONSUMER_GROUP   Consumer group name (default: agi-hpc)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import nats
    from nats.js.api import (
        ConsumerConfig,
        DeliverPolicy,
        RetentionPolicy,
        StorageType,
        StreamConfig,
    )
except ImportError:
    nats = None  # type: ignore

EventHandler = Callable[[dict], None]


@dataclass
class NatsBackendConfig:
    """Configuration for NATS JetStream backend."""

    servers: List[str] = field(
        default_factory=lambda: [
            os.getenv("AGI_FABRIC_NATS_URL", "nats://localhost:4222")
        ]
    )
    stream_name: str = os.getenv("AGI_FABRIC_NATS_STREAM", "AGI_HPC_EVENTS")
    consumer_group: str = os.getenv("AGI_FABRIC_CONSUMER_GROUP", "agi-hpc")
    consumer_name: Optional[str] = None
    max_msgs: int = 1000000
    max_bytes: int = 1073741824  # 1GB
    max_age_seconds: int = 604800  # 7 days
    ack_wait_seconds: int = 30
    max_deliver: int = 3
    batch_size: int = 10
    fetch_timeout_seconds: float = 1.0


@dataclass
class NatsStreamConfig:
    """Configuration for a NATS JetStream stream."""

    name: str
    subjects: List[str]
    max_msgs: int = 1000000
    max_bytes: int = 1073741824
    max_age_seconds: int = 604800
    retention: str = "limits"
    storage: str = "file"


class NatsBackend:
    """NATS JetStream backend for Event Fabric.

    Features:
    - JetStream persistent messaging
    - At-least-once delivery
    - Pull consumers with batch fetch
    - Queue groups for load balancing
    - Stream management
    """

    def __init__(self, config: Optional[NatsBackendConfig] = None) -> None:
        """Initialize NATS JetStream backend."""
        if nats is None:
            raise RuntimeError(
                "nats-py is required but not installed. "
                "Install with: pip install nats-py"
            )

        self.config = config or NatsBackendConfig()
        self.config.consumer_name = (
            self.config.consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        )

        self._nc: Optional[Any] = None  # nats.NATS client
        self._js: Optional[Any] = None  # JetStream context
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._subscriptions: Dict[str, Any] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Create dedicated asyncio event loop in daemon thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name=f"fabric-nats-loop-{self.config.consumer_name}",
            daemon=True,
        )
        self._loop_thread.start()

        # Connect via run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=10.0)

        logger.info(
            "[fabric][nats] initialized servers=%s stream=%s group=%s consumer=%s",
            self.config.servers,
            self.config.stream_name,
            self.config.consumer_group,
            self.config.consumer_name,
        )

    def _run_loop(self) -> None:
        """Run asyncio event loop in dedicated thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        """Connect to NATS and setup JetStream stream."""
        self._nc = await nats.connect(servers=self.config.servers)
        self._js = self._nc.jetstream()

        # Add or update the primary event stream with fabric.> subjects
        max_age_ns = self.config.max_age_seconds * 1_000_000_000
        await self._js.add_stream(
            name=self.config.stream_name,
            subjects=["fabric.>"],
            max_msgs=self.config.max_msgs,
            max_bytes=self.config.max_bytes,
            max_age=max_age_ns,
        )

        logger.info(
            "[fabric][nats] connected to %s, stream=%s ready",
            self.config.servers,
            self.config.stream_name,
        )

    def publish(self, topic: str, message: dict) -> None:
        """Publish message to JetStream (sync API matching FabricBackend protocol).

        Args:
            topic: Topic name
            message: Message payload dict
        """
        future = asyncio.run_coroutine_threadsafe(
            self._publish_async(topic, message), self._loop
        )
        future.result(timeout=5.0)

    async def _publish_async(self, topic: str, message: dict) -> None:
        """Async publish implementation."""
        subject = f"fabric.{topic}"
        payload = json.dumps(message).encode("utf-8")
        ack = await self._js.publish(subject, payload)
        logger.debug(
            "[fabric][nats] published subject=%s stream=%s seq=%s",
            subject,
            ack.stream,
            ack.seq,
        )

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to topic with durable pull consumer.

        Args:
            topic: Topic name
            handler: Callback function for messages
        """
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

        # Schedule async subscription setup on the event loop
        future = asyncio.run_coroutine_threadsafe(
            self._setup_subscription(topic), self._loop
        )
        future.result(timeout=10.0)

        logger.info("[fabric][nats] subscribed topic=%s", topic)

    async def _setup_subscription(self, topic: str) -> None:
        """Create a durable pull consumer for *topic* on the event loop."""
        subject = f"fabric.{topic}"
        durable = f"{self.config.consumer_group}-{topic.replace('.', '-')}"

        sub = await self._js.pull_subscribe(
            subject,
            durable=durable,
        )
        self._subscriptions[topic] = sub

        # Launch background fetch task
        task = asyncio.ensure_future(self._fetch_loop(topic, sub))
        self._consumer_tasks[topic] = task
        logger.debug(
            "[fabric][nats] pull consumer ready topic=%s durable=%s", topic, durable
        )

    async def _fetch_loop(self, topic: str, sub: Any) -> None:
        """Continuously fetch and dispatch messages for a subscription."""
        while not self._stop_event.is_set():
            try:
                msgs = await sub.fetch(
                    batch=self.config.batch_size,
                    timeout=self.config.fetch_timeout_seconds,
                )
                for msg in msgs:
                    try:
                        payload = json.loads(msg.data.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        payload = {"raw": msg.data}

                    with self._lock:
                        handlers = list(self._subscribers.get(topic, []))

                    for handler in handlers:
                        try:
                            handler(payload)
                        except Exception:
                            logger.exception(
                                "[fabric][nats] handler error topic=%s", topic
                            )

                    await msg.ack()
            except asyncio.TimeoutError:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("[fabric][nats] fetch error topic=%s", topic)
                    await asyncio.sleep(1.0)

    def close(self) -> None:
        """Shut down the NATS backend, cancel tasks and close connection."""
        self._stop_event.set()

        # Cancel consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        self._consumer_tasks.clear()

        # Close NATS connection
        if self._nc is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._nc.close(), self._loop)
                future.result(timeout=5.0)
            except Exception:
                logger.warning("[fabric][nats] error closing connection")

        # Stop the event loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5.0)
        logger.info("[fabric][nats] backend closed")
