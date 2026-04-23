# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS chat routing service.

Owns the chat NATS subjects and the SQLite transcript. Listens for
inbound messages, persists them, and routes based on kind:

  player_to_artemis → Primer handle_turn → reply persisted + published
                       to agi.rh.artemis.chat.out.<player_id>
  keeper_to_player  → persisted + published to chat.out.<player_id>
  keeper_to_all     → persisted + published to chat.broadcast

No business logic lives in the NATS subjects themselves; the ``kind``
field on the payload is authoritative. That keeps the wire format
legible from any client (browser DataChannel bridge, mobile PWA,
Python test).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from .store import ChatStore

log = logging.getLogger("artemis.chat.service")


SUBJECT_IN_PREFIX = "agi.rh.artemis.chat.in"        # .<player_id>
SUBJECT_OUT_PREFIX = "agi.rh.artemis.chat.out"      # .<player_id>
SUBJECT_BROADCAST = "agi.rh.artemis.chat.broadcast"

# Wire-protocol message kinds. String-valued so JSON payloads are
# self-describing without an enum class the browser side doesn't have.
MessageKind = Literal[
    "player_to_artemis",
    "artemis_to_player",
    "keeper_to_player",
    "keeper_to_all",
    "artemis_to_all",
]


@dataclass
class ChatMessage:
    """One chat message on the wire."""

    kind: MessageKind
    session_id: str
    from_id: str
    body: str
    to_id: str | None = None
    ts: float | None = None
    corr_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        payload = {
            "kind": self.kind,
            "session_id": self.session_id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "body": self.body,
            "ts": self.ts,
            "corr_id": self.corr_id,
        }
        if self.meta:
            payload["meta"] = self.meta
        return json.dumps(
            {k: v for k, v in payload.items() if v is not None},
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChatMessage":
        obj = json.loads(data)
        return cls(
            kind=str(obj["kind"]),  # type: ignore[arg-type]
            session_id=str(obj["session_id"]),
            from_id=str(obj["from_id"]),
            body=str(obj.get("body", "")),
            to_id=(None if obj.get("to_id") is None else str(obj["to_id"])),
            ts=(None if obj.get("ts") is None else float(obj["ts"])),
            corr_id=(None if obj.get("corr_id") is None else str(obj["corr_id"])),
            meta=dict(obj.get("meta") or {}),
        )


# A reply callback: given an inbound player_to_artemis message,
# produce the ARTEMIS reply body. Injected so tests can pass a
# deterministic stub. Production wires to Primer.handle_turn.
ArtemisReplyFn = Callable[[ChatMessage], Awaitable[str]]


class ChatService:
    """NATS-driven chat router with SQLite persistence.

    NATS connector is injected (same pattern as ArtemisService) so
    unit tests can drive the service without a broker.
    """

    def __init__(
        self,
        *,
        nats_url: str,
        store: ChatStore,
        reply_fn: ArtemisReplyFn,
        nats_connector: Any = None,
    ) -> None:
        self.nats_url = nats_url
        self.store = store
        self.reply_fn = reply_fn
        self._nats_connector = nats_connector
        self.nc: Any | None = None
        self._running = False

    # ── lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        self.nc = await self._connect()
        # One wildcard subscription catches every player's inbound
        # subject. Broadcast-in (keeper→all) and whispers use the
        # same subject tree with different ``kind`` values.
        await self.nc.subscribe(f"{SUBJECT_IN_PREFIX}.*", cb=self._on_in)
        self._running = True
        log.info(
            "chat service online: in=%s.* out=%s.* broadcast=%s",
            SUBJECT_IN_PREFIX,
            SUBJECT_OUT_PREFIX,
            SUBJECT_BROADCAST,
        )

    async def stop(self) -> None:
        self._running = False
        if self.nc is not None:
            try:
                await self.nc.drain()
            except Exception as e:  # noqa: BLE001
                log.warning("drain failed: %s", e)
            try:
                await self.nc.close()
            except Exception as e:  # noqa: BLE001
                log.warning("close failed: %s", e)
            self.nc = None
        log.info("chat service stopped")

    async def run_forever(self) -> None:
        while self._running:
            await asyncio.sleep(1)

    # ── NATS callbacks ─────────────────────────────────────────

    async def _on_in(self, msg: Any) -> None:
        """Handle any inbound chat message.

        Parse, persist, then dispatch to the right outbound path. Any
        exception here is logged; a single malformed message must not
        crash the service.
        """
        try:
            incoming = ChatMessage.from_bytes(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("chat in: bad JSON: %s", e)
            return
        try:
            await self.handle(incoming)
        except Exception as e:  # noqa: BLE001
            log.exception("chat handle failed: %s", e)

    # ── routing (public for tests) ─────────────────────────────

    async def handle(self, incoming: ChatMessage) -> list[ChatMessage]:
        """Route an inbound chat message.

        Returns the list of outbound messages published (useful for
        tests to assert the routing behaviour without a broker).
        """
        self._persist(incoming)
        outbound: list[ChatMessage] = []

        if incoming.kind == "player_to_artemis":
            reply_body = await self.reply_fn(incoming)
            reply = ChatMessage(
                kind="artemis_to_player",
                session_id=incoming.session_id,
                from_id="artemis",
                to_id=incoming.from_id,
                body=reply_body,
                corr_id=incoming.corr_id or _new_corr_id(),
            )
            self._persist(reply)
            await self._publish_out(reply)
            outbound.append(reply)

        elif incoming.kind == "keeper_to_player":
            if not incoming.to_id:
                log.warning("keeper_to_player missing to_id; dropping")
                return outbound
            await self._publish_out(incoming)
            outbound.append(incoming)

        elif incoming.kind in ("keeper_to_all", "artemis_to_all"):
            await self._publish_broadcast(incoming)
            outbound.append(incoming)

        else:
            log.warning("unknown chat kind: %s", incoming.kind)

        return outbound

    # ── persistence ────────────────────────────────────────────

    def _persist(self, msg: ChatMessage) -> None:
        self.store.append(
            session_id=msg.session_id,
            from_id=msg.from_id,
            to_id=msg.to_id,
            kind=msg.kind,
            body=msg.body,
            corr_id=msg.corr_id,
            ts=msg.ts,
        )

    # ── publish ────────────────────────────────────────────────

    async def _publish_out(self, msg: ChatMessage) -> None:
        if not msg.to_id:
            log.warning("outbound msg missing to_id; dropping")
            return
        subj = f"{SUBJECT_OUT_PREFIX}.{msg.to_id}"
        await self._publish(subj, msg.to_json())

    async def _publish_broadcast(self, msg: ChatMessage) -> None:
        await self._publish(SUBJECT_BROADCAST, msg.to_json())

    async def _publish(self, subject: str, payload: bytes) -> None:
        if self.nc is None:
            # In unit tests that call handle() directly without start(),
            # we simply skip the publish. ``outbound`` return value is
            # how the test inspects routing.
            return
        try:
            await self.nc.publish(subject, payload)
        except Exception as e:  # noqa: BLE001
            log.warning("publish %s failed: %s", subject, e)

    # ── connection ─────────────────────────────────────────────

    async def _connect(self) -> Any:
        if self._nats_connector is not None:
            return await self._nats_connector(self.nats_url)
        import nats  # type: ignore

        return await nats.connect(servers=[self.nats_url])


def _new_corr_id() -> str:
    return uuid.uuid4().hex[:12]


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


async def _primer_reply(msg: ChatMessage) -> str:
    """Default reply: route through ARTEMIS offline handler.

    Lazy-imported so the package can be unit-tested without pulling
    in vMOE / Primer dependencies.
    """
    from ..context import Bible, SessionState
    from ..mode import TurnRequest, handle_turn

    bible_path = Path(
        os.environ.get(
            "ARTEMIS_BIBLE_PATH",
            "/archive/artemis/bible/halyard_bible.json",
        )
    )
    bible = Bible.load(bible_path)

    # Lazy vMOE build. Chat volume is low so a per-turn rebuild is
    # fine for S1e; a persistent factory lives in ArtemisService for
    # the voice path.
    from ..mode import vmoe_caller
    from ..vmoe import default_experts, vMOE

    llm = vmoe_caller(vMOE(experts=default_experts()))

    state = SessionState(session_id=msg.session_id)
    state.load()
    req = TurnRequest(
        session_id=msg.session_id,
        turn_id=msg.corr_id or _new_corr_id(),
        speaker=msg.from_id,
        text=msg.body,
        ts=msg.ts or 0.0,
    )
    response = await handle_turn(req, state=state, bible=bible, llm=llm)
    if response is None or not response.text:
        return "(ARTEMIS silent.)"
    return response.text


def build_service_from_env() -> ChatService:
    """Construct a :class:`ChatService` from environment.

    Env:
      NATS_URL                   default nats://localhost:4222
      ARTEMIS_CHAT_DB            default /var/lib/atlas-artemis/chat.sqlite3
    """
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    db_path = os.environ.get(
        "ARTEMIS_CHAT_DB", "/var/lib/atlas-artemis/chat.sqlite3"
    )
    return ChatService(
        nats_url=nats_url,
        store=ChatStore(db_path),
        reply_fn=_primer_reply,
    )
