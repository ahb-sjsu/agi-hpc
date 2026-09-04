# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the ARTEMIS chat package (store + service).

Both modules are transport-injected so the tests never touch a real
broker. The SQLite store uses ``:memory:`` which gives us FTS5
coverage without touching the filesystem.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agi.primer.artemis.chat.service import (
    SUBJECT_BROADCAST,
    SUBJECT_IN_PREFIX,
    SUBJECT_OUT_PREFIX,
    ChatMessage,
    ChatService,
)
from agi.primer.artemis.chat.store import ChatStore

# ─────────────────────────────────────────────────────────────────
# store
# ─────────────────────────────────────────────────────────────────


def _mk_store() -> ChatStore:
    return ChatStore(":memory:")


def test_store_append_and_thread_returns_rows_in_ts_order() -> None:
    s = _mk_store()
    s.append(
        session_id="sess",
        from_id="player:imogen",
        to_id="artemis",
        kind="player_to_artemis",
        body="what do I see?",
        ts=1.0,
    )
    s.append(
        session_id="sess",
        from_id="artemis",
        to_id="player:imogen",
        kind="artemis_to_player",
        body="corridor flickers",
        ts=2.0,
    )
    msgs = s.thread(session_id="sess", participant_id="player:imogen")
    assert [m.body for m in msgs] == ["what do I see?", "corridor flickers"]


def test_store_thread_player_filter_hides_other_whispers() -> None:
    s = _mk_store()
    # Keeper whispers to imogen; arlo must not see it.
    s.append(
        session_id="sess",
        from_id="keeper:gm",
        to_id="player:imogen",
        kind="keeper_to_player",
        body="you sense something",
        ts=1.0,
    )
    s.append(
        session_id="sess",
        from_id="keeper:gm",
        to_id=None,
        kind="keeper_to_all",
        body="ship lurches",
        ts=2.0,
    )
    arlo_view = s.thread(session_id="sess", participant_id="player:arlo")
    bodies = [m.body for m in arlo_view]
    assert "you sense something" not in bodies
    assert "ship lurches" in bodies  # broadcast reaches everyone


def test_store_keeper_sees_everything() -> None:
    s = _mk_store()
    s.append(
        session_id="sess",
        from_id="keeper:gm",
        to_id="player:imogen",
        kind="keeper_to_player",
        body="psst",
        ts=1.0,
    )
    s.append(
        session_id="sess",
        from_id="player:arlo",
        to_id="artemis",
        kind="player_to_artemis",
        body="status?",
        ts=2.0,
    )
    keeper_view = s.thread(session_id="sess", participant_id="keeper:gm")
    bodies = [m.body for m in keeper_view]
    assert "psst" in bodies and "status?" in bodies


def test_store_search_finds_by_fts_body() -> None:
    s = _mk_store()
    s.append(
        session_id="sess",
        from_id="player:imogen",
        to_id="artemis",
        kind="player_to_artemis",
        body="check the vacuum seal integrity",
        ts=1.0,
    )
    s.append(
        session_id="sess",
        from_id="artemis",
        to_id="player:imogen",
        kind="artemis_to_player",
        body="seal reads nominal",
        ts=2.0,
    )
    hits = s.search(query="vacuum")
    assert len(hits) == 1
    assert "vacuum" in hits[0].body
    # Multi-term FTS5 query — matches documents containing both tokens.
    hits2 = s.search(query="seal nominal")
    assert len(hits2) == 1


def test_store_search_scoped_to_session() -> None:
    s = _mk_store()
    s.append(
        session_id="sess-A",
        from_id="p",
        to_id="a",
        kind="player_to_artemis",
        body="mi-go vault",
    )
    s.append(
        session_id="sess-B",
        from_id="p",
        to_id="a",
        kind="player_to_artemis",
        body="mi-go vault",
    )
    hits_any = s.search(query="mi-go")
    hits_a = s.search(query="mi-go", session_id="sess-A")
    assert len(hits_any) == 2
    assert len(hits_a) == 1


def test_store_search_empty_query_returns_empty() -> None:
    s = _mk_store()
    s.append(
        session_id="sess",
        from_id="p",
        to_id="a",
        kind="player_to_artemis",
        body="anything",
    )
    assert s.search(query="") == []
    assert s.search(query="   ") == []


def test_store_recent_returns_oldest_first_within_window() -> None:
    s = _mk_store()
    for i in range(10):
        s.append(
            session_id="sess",
            from_id="p",
            to_id="a",
            kind="player_to_artemis",
            body=f"m{i}",
            ts=float(i),
        )
    recent = s.recent(session_id="sess", limit=3)
    # Oldest-first within the 3 most recent → m7, m8, m9.
    assert [m.body for m in recent] == ["m7", "m8", "m9"]


def test_store_count() -> None:
    s = _mk_store()
    assert s.count() == 0
    s.append(
        session_id="sess",
        from_id="p",
        to_id="a",
        kind="player_to_artemis",
        body="hi",
    )
    assert s.count() == 1
    assert s.count(session_id="sess") == 1
    assert s.count(session_id="other") == 0


# ─────────────────────────────────────────────────────────────────
# ChatMessage wire format
# ─────────────────────────────────────────────────────────────────


def test_chatmessage_roundtrip() -> None:
    m = ChatMessage(
        kind="player_to_artemis",
        session_id="sess",
        from_id="player:imogen",
        to_id="artemis",
        body="hello",
        corr_id="abc",
    )
    raw = m.to_json()
    parsed = ChatMessage.from_bytes(raw)
    assert parsed.kind == "player_to_artemis"
    assert parsed.body == "hello"
    assert parsed.corr_id == "abc"


def test_chatmessage_drops_none_fields_in_json() -> None:
    m = ChatMessage(
        kind="keeper_to_all",
        session_id="sess",
        from_id="keeper:gm",
        body="ship lurches",
    )
    obj = json.loads(m.to_json())
    assert "to_id" not in obj
    assert "corr_id" not in obj
    assert obj["body"] == "ship lurches"


# ─────────────────────────────────────────────────────────────────
# service (routing, no NATS)
# ─────────────────────────────────────────────────────────────────


class _FakeNats:
    """Minimal NATS stand-in for publish-side assertions."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))

    async def drain(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _mk_service(reply_fn=None, nc: _FakeNats | None = None) -> ChatService:
    async def _default_reply(_msg: ChatMessage) -> str:
        return "acknowledged"

    svc = ChatService(
        nats_url="nats://fake",
        store=_mk_store(),
        reply_fn=reply_fn or _default_reply,
    )
    if nc is not None:
        svc.nc = nc  # bypass start(); tests inject the publisher
    return svc


def test_service_player_to_artemis_persists_and_replies() -> None:
    nc = _FakeNats()
    svc = _mk_service(nc=nc)
    msg = ChatMessage(
        kind="player_to_artemis",
        session_id="s1",
        from_id="player:imogen",
        to_id="artemis",
        body="what do I see?",
    )
    outbound = asyncio.run(svc.handle(msg))
    assert len(outbound) == 1
    reply = outbound[0]
    assert reply.kind == "artemis_to_player"
    assert reply.to_id == "player:imogen"
    assert reply.body == "acknowledged"
    # Both sides persisted.
    all_msgs = svc.store.thread(session_id="s1", participant_id="keeper:gm")
    kinds = [m.kind for m in all_msgs]
    assert kinds == ["player_to_artemis", "artemis_to_player"]
    # Published to chat.out.<player_id>, private to the asker.
    subjects = [s for s, _ in nc.published]
    assert subjects == [f"{SUBJECT_OUT_PREFIX}.player:imogen"]


def test_service_keeper_whisper_only_goes_to_one_player() -> None:
    nc = _FakeNats()
    svc = _mk_service(nc=nc)
    msg = ChatMessage(
        kind="keeper_to_player",
        session_id="s1",
        from_id="keeper:gm",
        to_id="player:arlo",
        body="you feel a chill",
    )
    outbound = asyncio.run(svc.handle(msg))
    assert len(outbound) == 1
    assert outbound[0].to_id == "player:arlo"
    # Subject is the per-player out, NOT broadcast.
    subjects = [s for s, _ in nc.published]
    assert subjects == [f"{SUBJECT_OUT_PREFIX}.player:arlo"]
    assert SUBJECT_BROADCAST not in subjects


def test_service_keeper_broadcast_hits_broadcast_subject() -> None:
    nc = _FakeNats()
    svc = _mk_service(nc=nc)
    msg = ChatMessage(
        kind="keeper_to_all",
        session_id="s1",
        from_id="keeper:gm",
        body="power flickers",
    )
    outbound = asyncio.run(svc.handle(msg))
    assert len(outbound) == 1
    subjects = [s for s, _ in nc.published]
    assert subjects == [SUBJECT_BROADCAST]


def test_service_keeper_whisper_missing_to_id_is_dropped() -> None:
    nc = _FakeNats()
    svc = _mk_service(nc=nc)
    msg = ChatMessage(
        kind="keeper_to_player",
        session_id="s1",
        from_id="keeper:gm",
        body="oops",
    )
    outbound = asyncio.run(svc.handle(msg))
    assert outbound == []
    # Persisted (for auditing) but never published.
    assert nc.published == []


def test_service_unknown_kind_is_persisted_but_not_routed() -> None:
    nc = _FakeNats()
    svc = _mk_service(nc=nc)
    msg = ChatMessage(
        kind="some_new_kind_we_didnt_define",  # type: ignore[arg-type]
        session_id="s1",
        from_id="player:imogen",
        body="?",
    )
    outbound = asyncio.run(svc.handle(msg))
    assert outbound == []
    assert nc.published == []
    # Still persisted so the Keeper portal can see malformed traffic.
    assert svc.store.count(session_id="s1") == 1


def test_service_reply_fn_exception_does_not_lose_inbound() -> None:
    async def boom(_msg: ChatMessage) -> str:
        raise RuntimeError("LLM offline")

    nc = _FakeNats()
    svc = _mk_service(reply_fn=boom, nc=nc)
    msg = ChatMessage(
        kind="player_to_artemis",
        session_id="s1",
        from_id="player:imogen",
        to_id="artemis",
        body="what do I see?",
    )
    with pytest.raises(RuntimeError):
        asyncio.run(svc.handle(msg))
    # Inbound message was persisted before the LLM crashed — critical
    # for post-mortem & proof-chain continuity.
    assert svc.store.count(session_id="s1") == 1


def test_service_subject_constants_match_docstring() -> None:
    # Sanity: the wire contract the browser bridge relies on.
    assert SUBJECT_IN_PREFIX == "agi.rh.artemis.chat.in"
    assert SUBJECT_OUT_PREFIX == "agi.rh.artemis.chat.out"
    assert SUBJECT_BROADCAST == "agi.rh.artemis.chat.broadcast"
