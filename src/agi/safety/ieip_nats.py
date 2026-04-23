# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""NATS publisher for I-EIP monitor events.

Concrete :class:`~agi.safety.ieip_monitor.EventPublisher` that
converts ieip payloads into the canonical :class:`agi.common.Event`
envelope and ships them via the existing Atlas ``NatsEventFabric``.

Kept in a separate module (rather than in ``ieip_monitor``) so the
monitor itself stays import-light: unit tests and the dashboard
summarizer don't pull NATS or asyncio transitively.

The publisher is event-loop-agnostic in the direction that matters:
callers from synchronous code can fire-and-forget via
:meth:`IEIPNatsPublisher.publish`, and the actual NATS send runs on
the fabric's own loop. This matches how every other Atlas subsystem
talks to NATS from inside sync call paths (Scientist, Primer, Ego).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from agi.common.event import Event
from agi.safety.ieip_monitor import DEFAULT_NATS_SUBJECT, EventPublisher

logger = logging.getLogger(__name__)


class IEIPNatsPublisher(EventPublisher):
    """Publishes I-EIP events onto the Atlas NATS fabric.

    Parameters
    ----------
    fabric:
        A connected :class:`~agi.core.events.nats_fabric.NatsEventFabric`.
        The publisher never calls ``connect`` itself -- lifecycle is
        the caller's responsibility, so restart semantics line up with
        whatever service is hosting the monitor.
    loop:
        Optional asyncio loop to schedule the publish on. Defaults to
        ``fabric.loop`` when present, falling back to
        ``asyncio.get_event_loop()``. Set explicitly when the monitor
        runs in a thread different from where NATS was brought up.
    source:
        Value used for ``Event.source``. Defaults to ``"ieip"``.
    """

    def __init__(
        self,
        fabric: Any,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        source: str = "ieip",
    ) -> None:
        self._fabric = fabric
        self._loop = loop or getattr(fabric, "loop", None)
        self._source = source

    # ── EventPublisher protocol ------------------------------------------

    def publish(self, subject: str, payload: Mapping[str, Any]) -> None:
        """Fire-and-forget publish.

        Swallows its own exceptions and logs a warning. The monitor is
        explicitly observer-only; a publisher failure must never
        propagate back into the model's call path.
        """
        try:
            event = Event(
                source=self._source,
                type=subject,
                payload=dict(payload),
            )
            coro = self._fabric.publish(subject, event)
            loop = self._loop or asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(asyncio.ensure_future, coro)
            else:
                loop.run_until_complete(coro)
        except Exception as exc:  # pragma: no cover - fail-soft by contract
            logger.warning(
                "ieip publish to %s failed: %s: %s",
                subject or DEFAULT_NATS_SUBJECT,
                type(exc).__name__,
                exc,
            )
