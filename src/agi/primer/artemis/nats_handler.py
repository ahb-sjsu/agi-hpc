# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS NATS handler — subscribes to heard, publishes say.

This module wraps :func:`handle_turn` in a NATS-driven service loop.
Bot containers publish transcript events on ``agi.rh.artemis.heard``;
this service listens, routes each event through the offline handler,
and publishes the reply (or silence) on ``agi.rh.artemis.say``.

Subjects (see ``docs/ARTEMIS.md`` §4):
  agi.rh.artemis.heard    — incoming (TurnRequest JSON)
  agi.rh.artemis.say      — outgoing (TurnResponse JSON)
  agi.rh.artemis.silence  — kill-switch toggle per session

NATS client is lazy-imported so the offline handler package can be
used without the ``nats-py`` dependency installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .context import Bible, SessionState
from .mode import LLMCaller, TurnRequest, TurnResponse, handle_turn
from .validator import ValidatorConfig

log = logging.getLogger("artemis.nats")


# Subject constants — single source of truth.
SUBJECT_HEARD = "agi.rh.artemis.heard"
SUBJECT_SAY = "agi.rh.artemis.say"
SUBJECT_SILENCE = "agi.rh.artemis.silence"

# Connection retry policy. Matches the dreaming service pattern.
_CONNECT_MAX_ATTEMPTS = 10
_CONNECT_BACKOFF_S = 5.0


class ArtemisService:
    """Event-driven ARTEMIS service.

    Responsibilities:
      - Own a single NATS connection and the three subscriptions above.
      - Maintain a process-local dict of :class:`SessionState` indexed
        by ``session_id`` (loaded from disk on first use).
      - Route each incoming ``heard`` message through :func:`handle_turn`
        and publish the resulting reply (if any) to ``say``.
      - Respond to ``silence`` messages by toggling the per-session
        kill-switch in memory; the toggle persists through restarts
        only to the extent that the session log does.

    The service deliberately does *not* own any reasoning itself —
    every decision flows through the offline ``handle_turn`` so both
    paths (unit-test, production) exercise identical code.
    """

    def __init__(
        self,
        *,
        nats_url: str,
        bible: Bible,
        llm: LLMCaller,
        validator_config: ValidatorConfig | None = None,
        keeper_approval_required: bool = True,
        nats_connector: Any = None,
    ) -> None:
        self.nats_url = nats_url
        self.bible = bible
        self.llm = llm
        self.validator_config = validator_config
        self.keeper_approval_required = keeper_approval_required
        self._nats_connector = nats_connector  # injected for tests
        self.sessions: dict[str, SessionState] = {}
        self.nc: Any | None = None
        self._running = False

    # ── lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Connect, subscribe, mark running. Returns when ready."""
        self.nc = await self._connect()
        await self.nc.subscribe(SUBJECT_HEARD, cb=self._on_heard)
        await self.nc.subscribe(SUBJECT_SILENCE, cb=self._on_silence)
        self._running = True
        log.info(
            "artemis service online: heard=%s silence=%s say=%s " "keeper_gate=%s",
            SUBJECT_HEARD,
            SUBJECT_SILENCE,
            SUBJECT_SAY,
            self.keeper_approval_required,
        )

    async def stop(self) -> None:
        """Graceful drain + close. Safe to call multiple times."""
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
        log.info("artemis service stopped")

    async def run_forever(self) -> None:
        """Block until ``stop`` is called or the process is signalled."""
        while self._running:
            await asyncio.sleep(1)

    # ── session state ──────────────────────────────────────────

    def _get_state(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            state = SessionState(session_id=session_id)
            state.load()
            self.sessions[session_id] = state
        return self.sessions[session_id]

    # ── NATS callbacks ─────────────────────────────────────────

    async def _on_heard(self, msg: Any) -> None:
        """Handle an agi.rh.artemis.heard event.

        Parsing errors are logged and dropped — a malformed event
        must not crash the service. The handler emits a silence
        outcome rather than a negative ack; there is no at-least-once
        semantics at this layer.
        """
        try:
            payload = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("heard: invalid JSON: %s", e)
            return

        try:
            req = TurnRequest(
                session_id=str(payload["session_id"]),
                turn_id=str(payload["turn_id"]),
                speaker=str(payload["speaker"]),
                text=str(payload["text"]),
                ts=float(payload["ts"]),
                partial=bool(payload.get("partial", False)),
                meta=dict(payload.get("meta") or {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning("heard: bad shape (%s): %s", e, str(payload)[:200])
            return

        state = self._get_state(req.session_id)
        try:
            response = await handle_turn(
                req,
                state=state,
                bible=self.bible,
                llm=self.llm,
                validator_config=self.validator_config,
                keeper_approval_required=self.keeper_approval_required,
            )
        except Exception as e:  # noqa: BLE001 — never let one turn kill the loop
            log.exception("handle_turn raised on %s: %s", req.turn_id, e)
            return

        if response is None:
            # Silence is a valid outcome; nothing to publish.
            return

        await self._publish_say(response)

    async def _on_silence(self, msg: Any) -> None:
        """Handle an agi.rh.artemis.silence event.

        Payload shape:
          {"session_id": str, "silenced": bool}
        ``silenced`` defaults to True — a bare silence toggle silences.
        """
        try:
            payload = json.loads(msg.data)
            session_id = str(payload["session_id"])
            silenced = bool(payload.get("silenced", True))
        except Exception as e:  # noqa: BLE001
            log.warning("silence: invalid payload: %s", e)
            return

        state = self._get_state(session_id)
        state.silenced = silenced
        log.info(
            "silence: session=%s silenced=%s (set by operator)",
            session_id,
            silenced,
        )

    # ── publish ────────────────────────────────────────────────

    async def _publish_say(self, response: TurnResponse) -> None:
        """Publish a TurnResponse to the bot on agi.rh.artemis.say."""
        payload = asdict(response)
        try:
            await self.nc.publish(
                SUBJECT_SAY,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("publish say failed for %s: %s", response.turn_id, e)

    # ── connection ─────────────────────────────────────────────

    async def _connect(self) -> Any:
        """Connect to NATS with retry. Uses an injected connector for tests."""
        if self._nats_connector is not None:
            return await self._nats_connector(self.nats_url)

        import nats  # lazy import; avoids a hard dep for unit tests

        last_err: Exception | None = None
        for attempt in range(_CONNECT_MAX_ATTEMPTS):
            try:
                nc = await nats.connect(servers=[self.nats_url])
                log.info(
                    "connected to NATS: %s (attempt %d)", self.nats_url, attempt + 1
                )
                return nc
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "NATS connect attempt %d/%d failed: %s",
                    attempt + 1,
                    _CONNECT_MAX_ATTEMPTS,
                    e,
                )
                await asyncio.sleep(_CONNECT_BACKOFF_S)
        raise RuntimeError(
            f"NATS connect failed after {_CONNECT_MAX_ATTEMPTS} attempts: {last_err}"
        )


# ─────────────────────────────────────────────────────────────────
# Config + factory
# ─────────────────────────────────────────────────────────────────


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_service_from_env() -> ArtemisService:
    """Construct an ArtemisService from process environment.

    Env vars read:
      NATS_URL                          default: nats://localhost:4222
      ARTEMIS_BIBLE_PATH                default:
                                        /archive/artemis/bible/halyard_bible.json
      ARTEMIS_KEEPER_APPROVAL_REQUIRED  default: true
      NRP_LLM_TOKEN                     bearer token for vMOE (required)

    The LLM caller is wired to a vMOE instance with the default expert
    pool. Callers wanting to override (e.g. a single-expert deployment
    for a tiny session) should construct ArtemisService directly.
    """
    from ..vmoe import default_experts, vMOE
    from .mode import vmoe_caller

    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    bible_path = Path(
        os.environ.get(
            "ARTEMIS_BIBLE_PATH",
            "/archive/artemis/bible/halyard_bible.json",
        )
    )
    keeper_gate = _bool_env("ARTEMIS_KEEPER_APPROVAL_REQUIRED", True)

    bible = Bible.load(bible_path)
    moe = vMOE(experts=default_experts())
    llm = vmoe_caller(moe)

    forbidden = bible.forbidden_phrases
    unknown = bible.unknown_texts()
    config = ValidatorConfig(
        forbidden_phrases=forbidden,
        unknown_chunks=unknown,
    )

    return ArtemisService(
        nats_url=nats_url,
        bible=bible,
        llm=llm,
        validator_config=config,
        keeper_approval_required=keeper_gate,
    )
