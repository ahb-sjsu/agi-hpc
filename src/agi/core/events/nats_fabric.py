# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Async NATS JetStream Event Fabric for AGI-HPC (Phase 0 Foundation).

Provides the primary pub/sub backbone connecting all cognitive subsystems.
Built on NATS with JetStream for at-least-once persistent delivery.

Usage::

    from agi.common.event import Event
    from agi.core.events.nats_fabric import NatsEventFabric

    fabric = NatsEventFabric(servers=["nats://localhost:4222"])
    await fabric.connect()

    # Publish
    event = Event.create("lh", "lh.request.chat", {"prompt": "hello"})
    await fabric.publish("agi.lh.request.chat", event)

    # Subscribe
    async def handler(event: Event) -> None:
        print(event.payload)

    await fabric.subscribe("agi.lh.request.chat", handler)

    # Request-reply
    reply = await fabric.request("agi.safety.check.input", event, timeout=5.0)

    await fabric.disconnect()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

try:
    import nats
    from nats.aio.client import Client as NatsClient
    from nats.aio.msg import Msg as NatsMsg
    from nats.js import JetStreamContext
    from nats.js.api import StreamConfig
except ImportError:
    nats = None  # type: ignore
    NatsClient = None  # type: ignore
    NatsMsg = None  # type: ignore
    JetStreamContext = None  # type: ignore
    StreamConfig = None  # type: ignore



# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], Awaitable[None]]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EventFabricProtocol(Protocol):
    """Interface that any event fabric backend must implement."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def publish(self, subject: str, event: Event) -> None: ...

    async def subscribe(
        self, subject: str, handler: EventHandler
    ) -> None: ...

    async def request(
        self, subject: str, event: Event, timeout: float
    ) -> Event: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NatsFabricConfig:
    """Configuration for the NATS Event Fabric."""

    servers: List[str] = field(
        default_factory=lambda: ["nats://localhost:4222"]
    )
    stream_name: str = "AGI_EVENTS"
    stream_subjects: List[str] = field(
        default_factory=lambda: ["agi.>"]
    )
    max_msgs: int = 1_000_000
    max_bytes: int = 1_073_741_824  # 1 GB
    max_age_seconds: int = 604_800  # 7 days
    connect_timeout: float = 10.0
    drain_timeout: float = 5.0


# ---------------------------------------------------------------------------
# NatsEventFabric
# ---------------------------------------------------------------------------


class NatsEventFabric:
    """Async NATS JetStream Event Fabric.

    Provides connect/disconnect/publish/subscribe/request over NATS,
    with JetStream for persistent at-least-once delivery.
    """

    def __init__(
        self,
        servers: List[str] | None = None,
        config: NatsFabricConfig | None = None,
    ) -> None:
        if nats is None:
            raise RuntimeError(
                "nats-py is required but not installed. "
                "Install with: pip install nats-py"
            )
        if config is not None:
            self._config = config
        else:
            self._config = NatsFabricConfig(
                servers=servers or ["nats://localhost:4222"]
            )

        self._nc: Optional[Any] = None
        self._js: Optional[Any] = None
        self._subscriptions: Dict[str, Any] = {}
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to NATS and initialise the JetStream stream."""
        self._nc = await nats.connect(
            servers=self._config.servers,
        )
        self._js = self._nc.jetstream()

        # Create or update the JetStream stream
        # nats-py StreamConfig expects max_age in seconds (float)
        stream_cfg = StreamConfig(
            name=self._config.stream_name,
            subjects=self._config.stream_subjects,
            max_msgs=self._config.max_msgs,
            max_bytes=self._config.max_bytes,
            max_age=self._config.max_age_seconds,
        )
        await self._js.add_stream(stream_cfg)

        self._connected = True
        logger.info(
            "[nats-fabric] connected to %s, stream=%s",
            self._config.servers,
            self._config.stream_name,
        )

    async def disconnect(self) -> None:
        """Drain subscriptions and close the NATS connection."""
        if self._nc is not None and self._nc.is_connected:
            try:
                await self._nc.drain()
            except Exception:
                logger.warning("[nats-fabric] error draining connection")
            finally:
                self._connected = False
                self._subscriptions.clear()
                logger.info("[nats-fabric] disconnected")

    @property
    def is_connected(self) -> bool:
        """Return True if connected to NATS."""
        return self._connected and self._nc is not None and self._nc.is_connected

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, subject: str, event: Event) -> None:
        """Publish an Event to a NATS subject via JetStream.

        Args:
            subject: NATS subject, e.g. ``"agi.lh.request.chat"``.
            event: Event instance to publish.
        """
        if not self.is_connected:
            raise RuntimeError("[nats-fabric] not connected")

        data = event.to_bytes()
        ack = await self._js.publish(subject, data)
        logger.debug(
            "[nats-fabric] published subject=%s seq=%s event_id=%s",
            subject,
            ack.seq,
            event.id,
        )

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(self, subject: str, handler: EventHandler) -> None:
        """Subscribe to a NATS subject and dispatch Events to *handler*.

        Uses core NATS subscriptions (not JetStream pull consumers) for
        low-latency push delivery.  JetStream publish ensures persistence;
        subscribers receive messages as they arrive.

        Args:
            subject: NATS subject (wildcards supported, e.g. ``"agi.lh.>"``).
            handler: Async callback receiving an Event.
        """
        if not self.is_connected:
            raise RuntimeError("[nats-fabric] not connected")

        async def _on_msg(msg: Any) -> None:
            try:
                event = Event.from_bytes(msg.data)
            except Exception:
                logger.exception(
                    "[nats-fabric] failed to deserialise message on %s",
                    msg.subject,
                )
                return
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "[nats-fabric] handler error subject=%s event_id=%s",
                    msg.subject,
                    event.id,
                )

        sub = await self._nc.subscribe(subject, cb=_on_msg)
        self._subscriptions[subject] = sub
        logger.info("[nats-fabric] subscribed subject=%s", subject)

    # ------------------------------------------------------------------
    # Request / Reply
    # ------------------------------------------------------------------

    async def request(
        self,
        subject: str,
        event: Event,
        timeout: float = 5.0,
    ) -> Event:
        """Send a request Event and await a reply Event.

        Uses NATS core request-reply (inbox pattern).  Subjects are
        automatically prefixed with ``_rpc.`` so they are **not**
        captured by the JetStream persistent stream (whose subjects
        match ``agi.>``).  Responders should subscribe to the
        ``_rpc.``-prefixed subject.

        Args:
            subject: Logical NATS subject for the request (e.g.
                ``"agi.safety.check.input"``).  The ``_rpc.`` prefix
                is added automatically.
            event: Request Event.
            timeout: Seconds to wait for reply.

        Returns:
            The reply Event.

        Raises:
            asyncio.TimeoutError: If no reply within *timeout*.
        """
        if not self.is_connected:
            raise RuntimeError("[nats-fabric] not connected")

        rpc_subject = f"_rpc.{subject}"
        data = event.to_bytes()
        reply_msg = await self._nc.request(rpc_subject, data, timeout=timeout)
        reply_event = Event.from_bytes(reply_msg.data)
        logger.debug(
            "[nats-fabric] request-reply subject=%s reply_id=%s",
            rpc_subject,
            reply_event.id,
        )
        return reply_event
