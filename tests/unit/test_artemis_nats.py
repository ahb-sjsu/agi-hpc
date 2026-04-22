# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for ARTEMIS NATS handler.

These tests inject a fake NATS client instead of spinning up
nats-server. End-to-end round-trip against a real broker lives in
``tests/integration/test_artemis_e2e.py``.
"""

from __future__ import annotations

import json
import time

import pytest

import agi.primer.artemis.context as _ctx
import agi.primer.artemis.mode as _mode
from agi.primer.artemis.context import Bible
from agi.primer.artemis.nats_handler import (
    SUBJECT_HEARD,
    SUBJECT_SAY,
    SUBJECT_SILENCE,
    ArtemisService,
)

# ─────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────


class FakeMsg:
    """Stand-in for nats.aio.Msg with just the ``.data`` bytes."""

    def __init__(self, data: bytes):
        self.data = data


class FakeNATS:
    """In-memory NATS double.

    Tracks subscribe callbacks and collects published messages so
    tests can assert outbound traffic. Simulates delivery by calling
    ``deliver(subject, bytes)`` from the test.
    """

    def __init__(self):
        self._subs: dict[str, list] = {}
        self.published: list[tuple[str, bytes]] = []
        self.closed = False
        self.drained = False

    async def subscribe(self, subject, cb):
        self._subs.setdefault(subject, []).append(cb)

    async def publish(self, subject, data):
        self.published.append((subject, data))

    async def drain(self):
        self.drained = True

    async def close(self):
        self.closed = True

    async def deliver(self, subject, data: bytes):
        """Test helper: dispatch a message to all subscribers."""
        for cb in self._subs.get(subject, []):
            await cb(FakeMsg(data))


class FakeLLM:
    def __init__(self, scripted):
        self._script = list(scripted)

    async def __call__(self, system, messages):
        if not self._script:
            raise RuntimeError("FakeLLM: script exhausted")
        return self._script.pop(0)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_artemis_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(_ctx, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_mode, "_PROOFS_DIR", tmp_path / "proofs")
    return tmp_path


@pytest.fixture
def fake_nats():
    return FakeNATS()


def _make_service(fake_nats, llm, *, keeper_gate=False):
    async def connector(_url):
        return fake_nats

    return ArtemisService(
        nats_url="nats://fake",
        bible=Bible(),
        llm=llm,
        keeper_approval_required=keeper_gate,
        nats_connector=connector,
    )


def _heard_payload(**overrides):
    base = {
        "session_id": "s1",
        "turn_id": "t-1",
        "speaker": "player:imogen",
        "text": "ARTEMIS, are you there?",
        "ts": time.time(),
        "partial": False,
        "meta": {},
    }
    base.update(overrides)
    return json.dumps(base).encode("utf-8")


# ─────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_subscribes_to_both_subjects(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    assert SUBJECT_HEARD in fake_nats._subs
    assert SUBJECT_SILENCE in fake_nats._subs
    assert svc._running is True
    await svc.stop()
    assert fake_nats.drained
    assert fake_nats.closed


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    await svc.stop()
    await svc.stop()  # must not raise


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heard_event_produces_say(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([("The handheld chirps.", "qwen3", 0.2)]))
    await svc.start()
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload())
    assert len(fake_nats.published) == 1
    subj, data = fake_nats.published[0]
    assert subj == SUBJECT_SAY
    payload = json.loads(data)
    assert payload["session_id"] == "s1"
    assert payload["turn_id"] == "t-1"
    assert payload["text"] == "The handheld chirps."
    assert payload["proof_hash"]
    assert payload["expert"] == "qwen3"
    await svc.stop()


@pytest.mark.asyncio
async def test_unaddressed_heard_does_not_publish(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    await fake_nats.deliver(
        SUBJECT_HEARD, _heard_payload(text="shall we head to the bridge?")
    )
    assert fake_nats.published == []
    await svc.stop()


@pytest.mark.asyncio
async def test_validator_rejection_silences(tmp_artemis_dirs, fake_nats):
    # The LLM emits an OOC break; validator rejects; no say published.
    svc = _make_service(
        fake_nats, FakeLLM([("As an AI, I cannot do that.", "qwen3", 0.1)])
    )
    await svc.start()
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload())
    assert fake_nats.published == []
    await svc.stop()


# ─────────────────────────────────────────────────────────────────
# Silence (kill-switch)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_silence_message_mutes_session(tmp_artemis_dirs, fake_nats):
    svc = _make_service(
        fake_nats, FakeLLM([("Should never be reached.", "qwen3", 0.1)])
    )
    await svc.start()
    # Silence the session.
    await fake_nats.deliver(
        SUBJECT_SILENCE,
        json.dumps({"session_id": "s1", "silenced": True}).encode(),
    )
    # Now deliver a would-be-triggering message.
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload())
    assert fake_nats.published == []
    await svc.stop()


@pytest.mark.asyncio
async def test_silence_toggle_off_restores_replies(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([("Welcome back.", "qwen3", 0.1)]))
    await svc.start()
    await fake_nats.deliver(
        SUBJECT_SILENCE,
        json.dumps({"session_id": "s1", "silenced": True}).encode(),
    )
    await fake_nats.deliver(
        SUBJECT_SILENCE,
        json.dumps({"session_id": "s1", "silenced": False}).encode(),
    )
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload())
    assert len(fake_nats.published) == 1
    await svc.stop()


@pytest.mark.asyncio
async def test_silence_defaults_to_true(tmp_artemis_dirs, fake_nats):
    # Bare silence message (no ``silenced`` key) should silence.
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    await fake_nats.deliver(SUBJECT_SILENCE, json.dumps({"session_id": "s1"}).encode())
    assert svc._get_state("s1").silenced is True
    await svc.stop()


# ─────────────────────────────────────────────────────────────────
# Malformed messages
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heard_malformed_json_is_dropped(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    await fake_nats.deliver(SUBJECT_HEARD, b"not valid json {{{")
    assert fake_nats.published == []
    assert svc._running is True  # service still alive
    await svc.stop()


@pytest.mark.asyncio
async def test_heard_missing_field_is_dropped(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    # Missing session_id.
    bad = json.dumps(
        {"turn_id": "t", "speaker": "p", "text": "ARTEMIS?", "ts": 1.0}
    ).encode()
    await fake_nats.deliver(SUBJECT_HEARD, bad)
    assert fake_nats.published == []
    await svc.stop()


@pytest.mark.asyncio
async def test_silence_malformed_is_dropped(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, FakeLLM([]))
    await svc.start()
    await fake_nats.deliver(SUBJECT_SILENCE, b"{not-json")
    # No exception, no state change.
    assert svc.sessions == {}
    await svc.stop()


# ─────────────────────────────────────────────────────────────────
# Per-session state isolation
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_sessions_have_independent_state(tmp_artemis_dirs, fake_nats):
    svc = _make_service(
        fake_nats,
        FakeLLM(
            [
                ("Reply to s1.", "qwen3", 0.1),
                ("Reply to s2.", "qwen3", 0.1),
            ]
        ),
    )
    await svc.start()
    # Silence s1 only.
    await fake_nats.deliver(
        SUBJECT_SILENCE,
        json.dumps({"session_id": "s1", "silenced": True}).encode(),
    )
    # Deliver a heard event to each session.
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload(session_id="s1"))
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload(session_id="s2"))
    # Only s2 should have produced a say.
    assert len(fake_nats.published) == 1
    payload = json.loads(fake_nats.published[0][1])
    assert payload["session_id"] == "s2"
    await svc.stop()


# ─────────────────────────────────────────────────────────────────
# Handler exceptions don't crash the service
# ─────────────────────────────────────────────────────────────────


class ExplodingLLM:
    async def __call__(self, system, messages):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_llm_error_does_not_crash_service(tmp_artemis_dirs, fake_nats):
    svc = _make_service(fake_nats, ExplodingLLM())
    await svc.start()
    await fake_nats.deliver(SUBJECT_HEARD, _heard_payload())
    # Service still alive, no publish.
    assert svc._running is True
    assert fake_nats.published == []
    # And a second turn with a working LLM would still be processed —
    # but we can't test that because the LLM is persistently broken.
    await svc.stop()


# ─────────────────────────────────────────────────────────────────
# Connection retry
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connector_is_used(tmp_artemis_dirs, fake_nats):
    """Sanity: the injected connector replaces the real ``nats.connect``."""
    calls = []

    async def connector(url):
        calls.append(url)
        return fake_nats

    svc = ArtemisService(
        nats_url="nats://x",
        bible=Bible(),
        llm=FakeLLM([]),
        nats_connector=connector,
    )
    await svc.start()
    assert calls == ["nats://x"]
    await svc.stop()
