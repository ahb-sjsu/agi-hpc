# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for ARTEMIS LiveKit agent (Phase 3).

Covers:
  - JWT minting: HS256 signature, identity+room claims, expiry, grants
  - Bridge encoding: heard payload shape, session filter, datachannel wrap
  - Agent I/O: on_transcript publishes, on_say posts to room, session
    isolation, malformed say dropped.

No real LiveKit SDK is imported — tests inject fake Room / NATS. The
actual integration happens in the agent process at runtime against the
real SFU; the unit layer verifies the logic between the I/O seams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agi.primer.artemis.livekit_agent.agent import (
    SUBJECT_HEARD,
    AgentConfig,
    ArtemisLiveKitAgent,
    Transcript,
    build_datachannel_message,
    build_heard_payload,
    parse_say_payload,
)
from agi.primer.artemis.livekit_agent.token import (
    GrantOptions,
    decode_participant_token,
    mint_participant_token,
)

# ─────────────────────────────────────────────────────────────────
# Token minting
# ─────────────────────────────────────────────────────────────────


_API_KEY = "APIkeyTest"
_API_SECRET = "supersecretsigning"


def test_token_round_trips():
    tok = mint_participant_token(
        identity="player:imogen",
        room_name="halyard-s01",
        api_key=_API_KEY,
        api_secret=_API_SECRET,
    )
    claims = decode_participant_token(tok, _API_SECRET)
    assert claims["iss"] == _API_KEY
    assert claims["sub"] == "player:imogen"
    assert claims["video"]["room"] == "halyard-s01"
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["canPublish"] is True
    assert claims["video"]["canSubscribe"] is True
    # expiry ~6h in the future (default TTL)
    assert claims["exp"] - claims["iat"] == 6 * 60 * 60


def test_token_wrong_secret_rejected():
    import jwt

    tok = mint_participant_token(
        identity="player:imogen",
        room_name="halyard-s01",
        api_key=_API_KEY,
        api_secret=_API_SECRET,
    )
    with pytest.raises(jwt.InvalidSignatureError):
        decode_participant_token(tok, "wrong-secret")


def test_token_custom_ttl_honored():
    tok = mint_participant_token(
        identity="player:sully",
        room_name="halyard-s01",
        api_key=_API_KEY,
        api_secret=_API_SECRET,
        ttl_seconds=60,
    )
    claims = decode_participant_token(tok, _API_SECRET)
    assert claims["exp"] - claims["iat"] == 60


def test_token_grants_admin_and_record_when_set():
    tok = mint_participant_token(
        identity="keeper:marsh",
        room_name="halyard-s01",
        api_key=_API_KEY,
        api_secret=_API_SECRET,
        grants=GrantOptions(room_admin=True, room_record=True),
    )
    claims = decode_participant_token(tok, _API_SECRET)
    assert claims["video"].get("roomAdmin") is True
    assert claims["video"].get("roomRecord") is True


def test_token_optional_name_embedded():
    tok = mint_participant_token(
        identity="player:imogen",
        room_name="halyard-s01",
        api_key=_API_KEY,
        api_secret=_API_SECRET,
        name="Dr Imogen Roth",
    )
    claims = decode_participant_token(tok, _API_SECRET)
    assert claims["name"] == "Dr Imogen Roth"


# ─────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────


def test_build_heard_payload_shape():
    t = Transcript(speaker="player:imogen", text="hello", is_final=True, ts=1.0)
    raw = build_heard_payload(session_id="s1", transcript=t, turn_id="t-1")
    payload = json.loads(raw)
    assert payload == {
        "session_id": "s1",
        "turn_id": "t-1",
        "speaker": "player:imogen",
        "text": "hello",
        "ts": 1.0,
        "partial": False,
        "meta": {},
    }


def test_build_heard_payload_partial_flag():
    t = Transcript(speaker="player:sully", text="welcome...", is_final=False, ts=2.0)
    payload = json.loads(
        build_heard_payload(session_id="s1", transcript=t, turn_id="t-2")
    )
    assert payload["partial"] is True


def test_build_heard_payload_auto_turn_id():
    t = Transcript(speaker="keeper", text="narration", is_final=True, ts=3.0)
    payload = json.loads(build_heard_payload(session_id="s1", transcript=t))
    assert payload["turn_id"].startswith("t-")
    assert len(payload["turn_id"]) > 3


def test_parse_say_filters_cross_session():
    raw = json.dumps({"session_id": "other", "text": "hi"}).encode()
    assert parse_say_payload(raw, "s1") is None


def test_parse_say_accepts_matching():
    raw = json.dumps({"session_id": "s1", "text": "hi", "turn_id": "t-9"}).encode()
    payload = parse_say_payload(raw, "s1")
    assert payload is not None
    assert payload["text"] == "hi"


def test_parse_say_malformed_returns_none():
    assert parse_say_payload(b"{not-json}", "s1") is None


def test_build_datachannel_envelope():
    payload = {
        "text": "The readings are nominal.",
        "turn_id": "t-1",
        "proof_hash": "abc",
    }
    wire = build_datachannel_message(payload)
    env = json.loads(wire)
    assert env["kind"] == "artemis.say"
    assert env["text"] == "The readings are nominal."
    assert env["turn_id"] == "t-1"
    assert env["proof_hash"] == "abc"
    assert "ts" in env


# ─────────────────────────────────────────────────────────────────
# Agent I/O — fakes
# ─────────────────────────────────────────────────────────────────


class FakeNATS:
    def __init__(self):
        self.published: list[tuple[str, bytes]] = []
        self.subscribed: dict[str, list] = {}
        self.closed = False
        self.drained = False

    async def publish(self, subject, data):
        self.published.append((subject, data))

    async def subscribe(self, subject, cb):
        self.subscribed.setdefault(subject, []).append(cb)

    async def drain(self):
        self.drained = True

    async def close(self):
        self.closed = True


@dataclass
class FakeLocalParticipant:
    published_data: list[bytes]

    async def publish_data(self, data: bytes, reliable: bool = True):
        self.published_data.append(data)


class FakeRoom:
    def __init__(self):
        self.local_participant = FakeLocalParticipant(published_data=[])
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


class FakeMsg:
    def __init__(self, data: bytes):
        self.data = data


def _config() -> AgentConfig:
    return AgentConfig(
        session_id="s1",
        livekit_url="wss://fake",
        livekit_api_key="k",
        livekit_api_secret="s",
        nats_url="nats://fake",
    )


# ─────────────────────────────────────────────────────────────────
# on_transcript
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_transcript_publishes_heard_event():
    nats = FakeNATS()
    agent = ArtemisLiveKitAgent(_config(), nats_client=nats, room=FakeRoom())
    t = Transcript(
        speaker="player:imogen", text="ARTEMIS, are you there?", is_final=True, ts=1.0
    )
    await agent.on_transcript(t)
    assert len(nats.published) == 1
    subj, data = nats.published[0]
    assert subj == SUBJECT_HEARD
    payload = json.loads(data)
    assert payload["session_id"] == "s1"
    assert payload["speaker"] == "player:imogen"
    assert payload["text"] == "ARTEMIS, are you there?"
    assert payload["partial"] is False


@pytest.mark.asyncio
async def test_on_transcript_swallows_no_nats():
    agent = ArtemisLiveKitAgent(_config(), nats_client=None, room=FakeRoom())
    t = Transcript(speaker="x", text="y", is_final=True, ts=1.0)
    # Must not raise even without a nats connection
    await agent.on_transcript(t)


# ─────────────────────────────────────────────────────────────────
# on_say
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_say_posts_to_room_data_channel():
    nats = FakeNATS()
    room = FakeRoom()
    agent = ArtemisLiveKitAgent(_config(), nats_client=nats, room=room)
    raw = json.dumps(
        {
            "session_id": "s1",
            "turn_id": "t-1",
            "text": "The handheld chirps.",
            "proof_hash": "abc",
        }
    ).encode()
    await agent.on_say(FakeMsg(raw))
    assert len(room.local_participant.published_data) == 1
    wire = json.loads(room.local_participant.published_data[0])
    assert wire["kind"] == "artemis.say"
    assert wire["text"] == "The handheld chirps."
    assert wire["turn_id"] == "t-1"


@pytest.mark.asyncio
async def test_on_say_ignores_other_session():
    room = FakeRoom()
    agent = ArtemisLiveKitAgent(_config(), nats_client=FakeNATS(), room=room)
    raw = json.dumps(
        {"session_id": "other", "turn_id": "t-1", "text": "hello"}
    ).encode()
    await agent.on_say(FakeMsg(raw))
    assert room.local_participant.published_data == []


@pytest.mark.asyncio
async def test_on_say_drops_malformed():
    room = FakeRoom()
    agent = ArtemisLiveKitAgent(_config(), nats_client=FakeNATS(), room=room)
    await agent.on_say(FakeMsg(b"not-json"))
    assert room.local_participant.published_data == []


@pytest.mark.asyncio
async def test_on_say_drops_empty_text():
    room = FakeRoom()
    agent = ArtemisLiveKitAgent(_config(), nats_client=FakeNATS(), room=room)
    raw = json.dumps({"session_id": "s1", "text": ""}).encode()
    await agent.on_say(FakeMsg(raw))
    assert room.local_participant.published_data == []


# ─────────────────────────────────────────────────────────────────
# Speaker naming
# ─────────────────────────────────────────────────────────────────


def test_format_speaker_keeper():
    class P:
        identity = "keeper:marsh"

    assert ArtemisLiveKitAgent._format_speaker(P()) == "keeper"


def test_format_speaker_player_identity_prefix():
    class P:
        identity = "imogen"

    assert ArtemisLiveKitAgent._format_speaker(P()) == "player:imogen"


def test_format_speaker_unknown_fallback():
    class P:
        identity = ""

    assert ArtemisLiveKitAgent._format_speaker(P()) == "player:unknown"


# ─────────────────────────────────────────────────────────────────
# AgentConfig.from_env
# ─────────────────────────────────────────────────────────────────


def test_agent_config_from_env(monkeypatch):
    monkeypatch.setenv("ARTEMIS_SESSION_ID", "halyard-s01")
    monkeypatch.setenv("LIVEKIT_URL", "wss://livekit.local")
    monkeypatch.setenv("LIVEKIT_API_KEY", "api-k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "api-s")
    monkeypatch.setenv("NATS_URL", "nats://atlas-nats:4222")
    cfg = AgentConfig.from_env()
    assert cfg.session_id == "halyard-s01"
    assert cfg.livekit_url == "wss://livekit.local"
    assert cfg.livekit_api_key == "api-k"
    assert cfg.nats_url == "nats://atlas-nats:4222"


def test_agent_config_missing_session_id_raises(monkeypatch):
    monkeypatch.delenv("ARTEMIS_SESSION_ID", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://x")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    with pytest.raises(RuntimeError, match="ARTEMIS_SESSION_ID"):
        AgentConfig.from_env()
