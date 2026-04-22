# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS LiveKit agent — the ears-and-mouth bridge.

Receives audio from a LiveKit room, runs streaming Whisper ASR, and
publishes transcripts on the Atlas NATS fabric. Consumes ARTEMIS
replies off NATS and posts them back to the room via DataChannel.

Design rules:
  - **No reasoning in this process.** Phase 2's :class:`ArtemisService`
    owns everything between "heard" and "say." This module is strictly
    an I/O adapter.
  - **Session = room.** ``session_id`` and LiveKit ``room_name`` are
    the same string end-to-end, so a multi-room deployment naturally
    isolates sessions at both NATS (subject payload filter) and
    LiveKit (room membership) layers.
  - **DataChannel, not audio, for v1.** Sending TTS into a virtual
    track is possible but deferred; the handheld-typing fiction works
    better, and text is dramatically easier to validate pre-post.

The actual LiveKit SDK imports are deferred into :meth:`run` so the
agent package is importable in environments without ``livekit-agents``
installed (unit tests use mocked I/O).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("artemis.livekit")


# Shared subject constants with nats_handler — re-declared locally to
# keep the agent importable without the whole primer.artemis chain.
SUBJECT_HEARD = "agi.rh.artemis.heard"
SUBJECT_SAY = "agi.rh.artemis.say"

# Reasonable sampling interval for the main loop while idle.
_IDLE_SLEEP_S = 1.0


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Per-session agent configuration.

    All fields have env-var defaults via :meth:`from_env` so the
    systemd / K8s deployment can inject them cleanly.
    """

    session_id: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    nats_url: str
    whisper_model: str = "large-v3"
    agent_identity: str = "artemis"
    agent_display_name: str = "ARTEMIS"

    @classmethod
    def from_env(cls, session_id: str | None = None) -> "AgentConfig":
        sid = session_id or os.environ.get("ARTEMIS_SESSION_ID", "")
        if not sid:
            raise RuntimeError("ARTEMIS_SESSION_ID is required (or pass session_id)")
        return cls(
            session_id=sid,
            livekit_url=os.environ["LIVEKIT_URL"],
            livekit_api_key=os.environ["LIVEKIT_API_KEY"],
            livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
            nats_url=os.environ.get("NATS_URL", "nats://localhost:4222"),
            whisper_model=os.environ.get("ARTEMIS_WHISPER_MODEL", "large-v3"),
            agent_identity=os.environ.get("ARTEMIS_AGENT_IDENTITY", "artemis"),
            agent_display_name=os.environ.get("ARTEMIS_DISPLAY_NAME", "ARTEMIS"),
        )


# ─────────────────────────────────────────────────────────────────
# Bridge — testable without LiveKit SDK
# ─────────────────────────────────────────────────────────────────


@dataclass
class Transcript:
    """One Whisper output — final or partial."""

    speaker: str
    text: str
    is_final: bool
    ts: float


def build_heard_payload(
    *,
    session_id: str,
    transcript: Transcript,
    turn_id: str | None = None,
) -> bytes:
    """Encode a Transcript as an ``agi.rh.artemis.heard`` JSON payload.

    Matches the schema in ``docs/ARTEMIS.md`` §4. Kept as a pure
    function so both the production agent and unit tests use the
    identical encoding.
    """
    payload = {
        "session_id": session_id,
        "turn_id": turn_id or f"t-{uuid.uuid4().hex[:12]}",
        "speaker": transcript.speaker,
        "text": transcript.text,
        "ts": transcript.ts,
        "partial": not transcript.is_final,
        "meta": {},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_say_payload(data: bytes, expected_session_id: str) -> dict | None:
    """Parse an ``agi.rh.artemis.say`` payload; drop cross-session traffic.

    Returns the parsed payload dict if it's for this session, else None.
    Malformed JSON returns None (and the caller logs).
    """
    try:
        payload = json.loads(data)
    except Exception:
        return None
    if payload.get("session_id") != expected_session_id:
        return None
    return payload


def build_datachannel_message(payload: dict) -> bytes:
    """Encode an ARTEMIS reply for LiveKit DataChannel delivery.

    Wraps the say payload in a small envelope the browser client can
    discriminate from other DataChannel traffic (future: game state,
    dice rolls, etc.).
    """
    envelope = {
        "kind": "artemis.say",
        "text": payload.get("text", ""),
        "turn_id": payload.get("turn_id"),
        "proof_hash": payload.get("proof_hash"),
        "ts": time.time(),
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────


class ArtemisLiveKitAgent:
    """LiveKit agent process — one instance per session.

    Construction is cheap (no I/O). :meth:`run` performs all connects
    and is the systemd / agent-worker entry point.

    Tests inject ``nats_client`` and ``room`` to avoid real I/O.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        nats_client: Any = None,
        room: Any = None,
        stt_factory: Any = None,
    ) -> None:
        self.config = config
        self._nats = nats_client
        self._room = room
        self._stt_factory = stt_factory
        self._running = False

    # ── testable primitives ─────────────────────────────────────

    async def on_transcript(self, transcript: Transcript) -> None:
        """Publish a finalized (or partial) transcript on NATS."""
        if self._nats is None:
            log.warning("no NATS connection; dropping transcript")
            return
        payload = build_heard_payload(
            session_id=self.config.session_id, transcript=transcript
        )
        try:
            await self._nats.publish(SUBJECT_HEARD, payload)
        except Exception as e:  # noqa: BLE001
            log.warning("heard publish failed: %s", e)

    async def on_say(self, msg: Any) -> None:
        """NATS callback for ``agi.rh.artemis.say``. Posts to DataChannel."""
        payload = parse_say_payload(msg.data, self.config.session_id)
        if payload is None:
            return  # cross-session or malformed
        if not payload.get("text"):
            return
        if self._room is None:
            log.warning("no room; dropping say reply")
            return
        wire = build_datachannel_message(payload)
        try:
            await self._room.local_participant.publish_data(wire, reliable=True)
            log.info(
                "posted reply to room: turn=%s len=%d",
                payload.get("turn_id"),
                len(wire),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("publish_data failed: %s", e)

    # ── run loop (integration surface) ──────────────────────────

    async def run(self) -> None:
        """Connect to LiveKit + NATS and run until cancelled.

        The LiveKit/NATS imports are deferred so this module stays
        importable without heavy deps.
        """
        if self._nats is None:
            import nats

            self._nats = await nats.connect(self.config.nats_url)
            log.info("nats connected: %s", self.config.nats_url)

        await self._nats.subscribe(SUBJECT_SAY, cb=self.on_say)

        if self._room is None:
            self._room = await self._connect_room()

        self._attach_track_handlers()
        self._running = True
        log.info(
            "artemis agent online: room=%s identity=%s",
            self.config.session_id,
            self.config.agent_identity,
        )
        try:
            while self._running:
                await asyncio.sleep(_IDLE_SLEEP_S)
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._running = False

    # ── plumbing (real SDK calls live here) ─────────────────────

    async def _connect_room(self) -> Any:  # pragma: no cover — I/O
        """Connect to the LiveKit room as the agent participant.

        Uses the LiveKit Python SDK. The agent needs a JWT that the
        SFU accepts; we mint our own with
        :func:`.token.mint_participant_token` using the API secret.
        """
        from livekit import rtc

        from .token import GrantOptions, mint_participant_token

        token = mint_participant_token(
            identity=self.config.agent_identity,
            room_name=self.config.session_id,
            api_key=self.config.livekit_api_key,
            api_secret=self.config.livekit_api_secret,
            grants=GrantOptions(
                can_publish=False,  # text-only in v1
                can_subscribe=True,
                can_publish_data=True,
            ),
            name=self.config.agent_display_name,
        )
        room = rtc.Room()
        await room.connect(self.config.livekit_url, token)
        return room

    def _attach_track_handlers(self) -> None:  # pragma: no cover — I/O
        """Subscribe to every audio track in the room and stream STT."""
        room = self._room

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):  # noqa: ANN001
            if getattr(track, "kind", None) != "audio":
                return
            asyncio.create_task(self._stream_track(track, participant))

    async def _stream_track(
        self, track: Any, participant: Any
    ) -> None:  # pragma: no cover — I/O
        """Run streaming ASR on one participant's audio track.

        Uses faster-whisper via the LiveKit plugin interface. On each
        finalized utterance, calls :meth:`on_transcript`.
        """
        speaker = self._format_speaker(participant)
        stt = (self._stt_factory or self._default_stt)()
        async for event in stt.stream(track):
            if getattr(event, "type", None) == "final":
                await self.on_transcript(
                    Transcript(
                        speaker=speaker,
                        text=event.text,
                        is_final=True,
                        ts=time.time(),
                    )
                )
            elif getattr(event, "type", None) == "interim":
                await self.on_transcript(
                    Transcript(
                        speaker=speaker,
                        text=event.text,
                        is_final=False,
                        ts=time.time(),
                    )
                )

    def _default_stt(self) -> Any:  # pragma: no cover
        """Construct a default STT stream (faster-whisper local)."""
        from livekit.plugins import openai  # type: ignore

        # faster-whisper is what we actually want on-Atlas; if the
        # livekit.plugins.faster_whisper plugin is present, swap here.
        return openai.STT(model=self.config.whisper_model)

    @staticmethod
    def _format_speaker(participant: Any) -> str:
        """Map a LiveKit participant to the ARTEMIS speaker convention."""
        identity = getattr(participant, "identity", "") or "unknown"
        # Keeper convention: room creator's identity starts with "keeper:".
        if identity.startswith("keeper"):
            return "keeper"
        return f"player:{identity}"

    async def _shutdown(self) -> None:  # pragma: no cover — I/O
        """Drain NATS + disconnect from the room."""
        try:
            if self._nats is not None:
                await self._nats.drain()
                await self._nats.close()
        except Exception as e:  # noqa: BLE001
            log.warning("nats shutdown: %s", e)
        try:
            if self._room is not None and hasattr(self._room, "disconnect"):
                await self._room.disconnect()
        except Exception as e:  # noqa: BLE001
            log.warning("room disconnect: %s", e)
        log.info("artemis agent stopped")
