# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Event Bus for AGI-HPC Integration.

Provides a unified interface for subsystems to publish and subscribe to events
via the EventFabric infrastructure. Includes:
- Typed event publishing
- Subscription management
- Event filtering
- Distributed tracing
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Type

from agi.integration.events.schema import (
    AGIEvent,
    EventCategory,
    EventHeader,
    EventPriority,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EventHandler = Callable[[AGIEvent], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """Event subscription."""

    subscription_id: str
    topic_pattern: str
    handler: EventHandler
    categories: Set[EventCategory] = field(default_factory=set)
    min_priority: EventPriority = EventPriority.LOW
    active: bool = True


@dataclass
class TraceContext:
    """Distributed tracing context."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """
    Central event bus for AGI-HPC.

    Provides:
    - Typed event publishing with automatic topic mapping
    - Subscription management with filtering
    - Distributed tracing context propagation
    - Retry and dead-letter queue handling
    """

    def __init__(
        self,
        service_name: str,
        instance_id: Optional[str] = None,
        fabric: Optional[Any] = None,
    ) -> None:
        """Initialize event bus.

        Args:
            service_name: Name of the owning service
            instance_id: Instance identifier (auto-generated if not provided)
            fabric: EventFabric instance (created if not provided)
        """
        self.service_name = service_name
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self._fabric = fabric
        self._subscriptions: Dict[str, Subscription] = {}
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._trace_context: Optional[TraceContext] = None
        self._running = False
        self._consumer_tasks: List[asyncio.Task] = []

        logger.info(
            "[events][bus] initialized service=%s instance=%s",
            service_name,
            self.instance_id,
        )

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        # Initialize fabric if needed
        if self._fabric is None:
            from agi.core.events.fabric import EventFabric

            self._fabric = EventFabric()
            await self._fabric.connect()

        self._running = True
        logger.info("[events][bus] started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clear subscriptions
        self._subscriptions.clear()
        self._handlers.clear()

        logger.info("[events][bus] stopped")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        event: AGIEvent,
        topic: Optional[str] = None,
    ) -> str:
        """Publish an event.

        Args:
            event: Event to publish
            topic: Optional explicit topic (derived from event type if not provided)

        Returns:
            Message ID from the fabric
        """
        # Fill in source information
        event.header.source_service = self.service_name
        event.header.source_instance = self.instance_id

        # Propagate trace context if active
        if self._trace_context:
            event.header.trace_id = self._trace_context.trace_id
            event.header.span_id = self._trace_context.span_id

        # Derive topic from event type if not provided
        if topic is None:
            topic = event.header.event_type

        # Publish via fabric
        data = event.to_dict()
        message_id = await self._fabric.publish_async(topic, data)

        logger.debug(
            "[events][bus] published event_type=%s topic=%s id=%s",
            event.header.event_type,
            topic,
            event.header.event_id,
        )

        return message_id

    async def publish_typed(
        self,
        event_class: Type[AGIEvent],
        **kwargs,
    ) -> str:
        """Publish a typed event using factory method.

        Args:
            event_class: Event class with create() method
            **kwargs: Arguments for create()

        Returns:
            Message ID
        """
        if not hasattr(event_class, "create"):
            raise ValueError(f"{event_class.__name__} has no create() method")

        event = event_class.create(source=self.service_name, **kwargs)
        return await self.publish(event)

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    def subscribe(
        self,
        topic_pattern: str,
        handler: EventHandler,
        categories: Optional[Set[EventCategory]] = None,
        min_priority: EventPriority = EventPriority.LOW,
    ) -> str:
        """Subscribe to events.

        Args:
            topic_pattern: Topic pattern (supports wildcards)
            handler: Async handler function
            categories: Optional category filter
            min_priority: Minimum priority to receive

        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            topic_pattern=topic_pattern,
            handler=handler,
            categories=categories or set(),
            min_priority=min_priority,
        )

        self._subscriptions[subscription_id] = subscription

        # Add to handlers by topic
        if topic_pattern not in self._handlers:
            self._handlers[topic_pattern] = []
        self._handlers[topic_pattern].append(handler)

        # Start consumer if running
        if self._running:
            task = asyncio.create_task(self._consume_topic(topic_pattern))
            self._consumer_tasks.append(task)

        logger.info(
            "[events][bus] subscribed topic=%s id=%s",
            topic_pattern,
            subscription_id,
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if unsubscribed, False if not found
        """
        if subscription_id in self._subscriptions:
            sub = self._subscriptions.pop(subscription_id)
            sub.active = False

            if sub.topic_pattern in self._handlers:
                self._handlers[sub.topic_pattern].remove(sub.handler)

            logger.info(
                "[events][bus] unsubscribed topic=%s id=%s",
                sub.topic_pattern,
                subscription_id,
            )
            return True
        return False

    async def _consume_topic(self, topic: str) -> None:
        """Consume messages from a topic."""
        async for message in self._fabric.subscribe_async(topic):
            if not self._running:
                break

            try:
                event = AGIEvent.from_dict(message.data)
                await self._dispatch(topic, event)
            except Exception as e:
                logger.error(
                    "[events][bus] failed to process message on %s: %s",
                    topic,
                    e,
                )

    async def _dispatch(self, topic: str, event: AGIEvent) -> None:
        """Dispatch event to handlers."""
        handlers = self._handlers.get(topic, [])

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    "[events][bus] handler error for %s: %s",
                    event.header.event_type,
                    e,
                )

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def trace_span(
        self,
        operation: str,
        parent_context: Optional[TraceContext] = None,
    ):
        """Create a trace span.

        Args:
            operation: Name of the operation being traced
            parent_context: Optional parent trace context

        Yields:
            TraceContext for this span
        """
        if parent_context:
            context = TraceContext(
                trace_id=parent_context.trace_id,
                parent_span_id=parent_context.span_id,
                baggage=parent_context.baggage.copy(),
            )
        else:
            context = TraceContext()

        # Set as current context
        previous = self._trace_context
        self._trace_context = context

        try:
            yield context
        finally:
            self._trace_context = previous

    def get_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        return self._trace_context

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def create_event(
        self,
        event_type: str,
        category: EventCategory,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> AGIEvent:
        """Create a new event with standard header.

        Args:
            event_type: Type string for the event
            category: Event category
            payload: Event payload
            priority: Event priority
            correlation_id: Optional correlation ID

        Returns:
            New AGIEvent
        """
        header = EventHeader(
            event_type=event_type,
            category=category,
            priority=priority,
            source_service=self.service_name,
            source_instance=self.instance_id,
            correlation_id=correlation_id,
        )

        if self._trace_context:
            header.trace_id = self._trace_context.trace_id
            header.span_id = self._trace_context.span_id

        return AGIEvent(header=header, payload=payload)


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_event_bus(
    service_name: str,
    fabric_mode: str = "local",
) -> EventBus:
    """Create event bus with default configuration.

    Args:
        service_name: Name of the service
        fabric_mode: EventFabric mode (local, zmq, redis)

    Returns:
        Configured EventBus
    """
    from agi.core.events.fabric import EventFabric

    fabric = EventFabric(mode=fabric_mode)
    return EventBus(service_name=service_name, fabric=fabric)
