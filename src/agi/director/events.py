# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Erebus's Director as a permanent node on the NATS global workspace.

Subject tree ``agi.director.*`` (see docs/SELF_DIRECTED_COGNITION.md):

    agi.director.state      out  self-model summary + heartbeat
    agi.director.cycle      out  one message per SDCC cycle (+ hash proof)
    agi.director.journal    out  each journal entry as it is written
    agi.director.goal.*     out  goal lifecycle (proposed/gated/.../done)
    agi.director.command    IN   directives / questions to the Director
    agi.director.reply      out  responses to commands / questions

``nats`` is an optional dependency (house pattern: ``try/except
ImportError``). With no broker present the node degrades to a no-op that
logs at debug level, so the Director daemon and its unit tests run with
or without NATS. Publishing is fire-and-forget; a broker hiccup never
propagates into the cognition loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger("director.events")

# Subject constants — importable without a broker.
SUBJ_STATE = "agi.director.state"
SUBJ_CYCLE = "agi.director.cycle"
SUBJ_JOURNAL = "agi.director.journal"
SUBJ_GOAL = "agi.director.goal"  # + ".{proposed,gated,accepted,dispatched,done}"
SUBJ_COMMAND = "agi.director.command"
SUBJ_REPLY = "agi.director.reply"

try:  # optional dependency
    import nats as _nats  # type: ignore

    _HAVE_NATS = True
except ImportError:  # pragma: no cover - exercised only where nats is absent
    _nats = None  # type: ignore
    _HAVE_NATS = False


CommandHandler = Callable[[dict], Awaitable[dict | None]]


class DirectorNode:
    """Thin async NATS presence for the Director.

    Usage::

        node = DirectorNode(servers="nats://127.0.0.1:4222")
        await node.connect()
        await node.publish_state(model.summary())
        await node.subscribe_commands(handler)   # handler(msg) -> reply dict
        ...
        await node.close()

    If ``nats`` is unavailable or the connection fails, the node stays in a
    disconnected state and every ``publish_*`` is a silent no-op.
    """

    def __init__(self, servers: str = "nats://127.0.0.1:4222") -> None:
        self.servers = servers
        self._nc: Any = None

    @property
    def connected(self) -> bool:
        return self._nc is not None and not getattr(self._nc, "is_closed", True)

    async def connect(self) -> bool:
        """Connect to NATS. Returns True on success, False if unavailable."""
        if not _HAVE_NATS:
            log.debug("nats not installed; Director NATS node disabled")
            return False
        try:
            self._nc = await _nats.connect(self.servers)  # type: ignore[union-attr]
            log.info("Director NATS node connected: %s", self.servers)
            return True
        except Exception as e:  # noqa: BLE001 - broker may be down; degrade gracefully
            log.warning("Director NATS connect failed (%s); running without NATS", e)
            self._nc = None
            return False

    async def _publish(self, subject: str, payload: dict) -> None:
        if not self.connected:
            return
        try:
            await self._nc.publish(subject, json.dumps(payload).encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - never let telemetry break cognition
            log.warning("publish %s failed: %s", subject, e)

    async def publish_state(self, summary: dict) -> None:
        await self._publish(SUBJ_STATE, summary)

    async def publish_cycle(self, record: dict) -> None:
        await self._publish(SUBJ_CYCLE, record)

    async def publish_journal(self, entry: dict) -> None:
        await self._publish(SUBJ_JOURNAL, entry)

    async def publish_goal(self, phase: str, goal: dict) -> None:
        await self._publish(f"{SUBJ_GOAL}.{phase}", goal)

    async def subscribe_commands(self, handler: CommandHandler) -> None:
        """Subscribe to inbound commands. ``handler`` returns an optional
        reply dict, which is published on ``agi.director.reply``.

        Phase A ships a read-only handler (status/pause/resume/run-now);
        acting commands arrive with the goal loop in Phase B."""
        if not self.connected:
            return

        async def _cb(msg: Any) -> None:
            try:
                cmd = json.loads(msg.data.decode("utf-8"))
            except Exception:  # noqa: BLE001
                cmd = {"raw": msg.data[:200].decode("utf-8", "replace")}
            try:
                reply = await handler(cmd)
            except Exception as e:  # noqa: BLE001 - a bad command must not kill the sub
                reply = {"ok": False, "error": str(e)[:200]}
            if reply is not None:
                await self._publish(SUBJ_REPLY, reply)

        await self._nc.subscribe(SUBJ_COMMAND, cb=_cb)
        log.info("Director listening on %s", SUBJ_COMMAND)

    async def close(self) -> None:
        if self.connected:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001
                pass
        self._nc = None
