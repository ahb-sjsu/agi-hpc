# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-6 tests — halyard-keeper-backend.

Covers the full keeper service surface: auth middleware,
session lifecycle, approval queue, token mint, and the aiohttp
routes wiring them together. Uses aiohttp's in-memory test
client so no network I/O happens.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from agi.halyard.keeper import (
    AiKind,
    ApprovalQueue,
    ApprovalState,
    KeeperAuthConfig,
    LiveKitConfig,
    SessionAlreadyExists,
    SessionNotFound,
    SessionRegistry,
    SessionState,
    build_app,
)
from agi.halyard.keeper._jwt import decode_participant_token
from agi.halyard.keeper.approvals import compute_approval_id
from agi.halyard.keeper.auth import ip_allowed, parse_basic_header
from agi.halyard.keeper.livekit import (
    mint_ai_token,
    mint_keeper_token,
    mint_player_token,
)

LK_DEV = LiveKitConfig(
    url="ws://127.0.0.1:7880",
    api_key="devkey",
    api_secret="secret-for-tests-only",
    ttl_seconds=3600,
)


# ─────────────────────────────────────────────────────────────────
# auth.py
# ─────────────────────────────────────────────────────────────────


def test_parse_basic_header_good() -> None:
    creds = base64.b64encode(b"alice:s3cret").decode("ascii")
    assert parse_basic_header(f"Basic {creds}") == ("alice", "s3cret")


def test_parse_basic_header_missing() -> None:
    assert parse_basic_header(None) is None
    assert parse_basic_header("") is None
    assert parse_basic_header("Bearer abc") is None


def test_parse_basic_header_malformed() -> None:
    assert parse_basic_header("Basic not_base64!!!") is None
    # Missing ':'
    encoded = base64.b64encode(b"just-user").decode("ascii")
    assert parse_basic_header(f"Basic {encoded}") is None


def test_ip_allowed_empty_allowlist_permits_everything() -> None:
    assert ip_allowed("1.2.3.4", ()) is True


def test_ip_allowed_exact_ip() -> None:
    assert ip_allowed("127.0.0.1", ("127.0.0.1/32",)) is True
    assert ip_allowed("127.0.0.2", ("127.0.0.1/32",)) is False


def test_ip_allowed_cidr() -> None:
    assert ip_allowed("10.1.2.3", ("10.0.0.0/8",)) is True
    assert ip_allowed("11.0.0.1", ("10.0.0.0/8",)) is False


def test_ip_allowed_malformed_ip_denies() -> None:
    assert ip_allowed("not-an-ip", ("10.0.0.0/8",)) is False


def test_keeper_auth_is_enabled() -> None:
    assert KeeperAuthConfig(username="", password="").is_enabled() is False
    assert KeeperAuthConfig(username="a", password="b").is_enabled() is True


# ─────────────────────────────────────────────────────────────────
# sessions.py
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_registry_create_and_get(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    s = await reg.create("halyard-s01")
    assert s.id == "halyard-s01"
    assert s.state is SessionState.OPEN
    got = await reg.get("halyard-s01")
    assert got is s


@pytest.mark.asyncio
async def test_session_registry_duplicate_create(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    await reg.create("halyard-s01")
    with pytest.raises(SessionAlreadyExists):
        await reg.create("halyard-s01")


@pytest.mark.asyncio
async def test_session_registry_not_found(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    with pytest.raises(SessionNotFound):
        await reg.get("ghost")


@pytest.mark.asyncio
async def test_session_registry_transitions(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    await reg.create("s")
    assert (await reg.pause("s")).state is SessionState.PAUSED
    assert (await reg.resume("s")).state is SessionState.OPEN
    assert (await reg.close("s")).state is SessionState.CLOSED
    # Closed can't be paused or resumed.
    from agi.halyard.keeper import InvalidTransition as _IT

    with pytest.raises(_IT):
        await reg.pause("s")


@pytest.mark.asyncio
async def test_session_registry_is_accepting_joins(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    assert reg.is_accepting_joins("unknown") is True
    await reg.create("s")
    assert reg.is_accepting_joins("s") is True
    await reg.pause("s")
    assert reg.is_accepting_joins("s") is True
    await reg.close("s")
    assert reg.is_accepting_joins("s") is False


@pytest.mark.asyncio
async def test_session_registry_ensure_idempotent(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    s1 = await reg.ensure("s")
    s2 = await reg.ensure("s")
    assert s1 is s2


@pytest.mark.asyncio
async def test_session_registry_append_log(tmp_path: Path) -> None:
    reg = SessionRegistry(archive_root=tmp_path)
    await reg.create("s")
    await reg.pause("s")
    await reg.resume("s")
    log_p = tmp_path / "keeper" / "sessions" / "s.log.jsonl"
    assert log_p.is_file()
    lines = log_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [r["action"] for r in parsed] == ["create", "pause", "resume"]


# ─────────────────────────────────────────────────────────────────
# approvals.py
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_queue_enqueue_and_list() -> None:
    q = ApprovalQueue()
    a = await q.enqueue(
        session_id="s",
        ai=AiKind.ARTEMIS,
        turn_id="t1",
        proposed_text="Hello.",
    )
    assert a.state is ApprovalState.PENDING
    pending = await q.pending_for("s")
    assert len(pending) == 1
    assert pending[0].approval_id == a.approval_id


@pytest.mark.asyncio
async def test_approval_queue_approve_plain() -> None:
    q = ApprovalQueue()
    a = await q.enqueue(
        session_id="s",
        ai=AiKind.ARTEMIS,
        turn_id="t1",
        proposed_text="Hello.",
    )
    out = await q.approve(a.approval_id)
    assert out.state is ApprovalState.APPROVED
    assert out.final_text == "Hello."
    # No longer pending.
    assert await q.pending_for("s") == []


@pytest.mark.asyncio
async def test_approval_queue_approve_with_edit() -> None:
    q = ApprovalQueue()
    a = await q.enqueue(
        session_id="s", ai=AiKind.SIGMA4, turn_id="t1",
        proposed_text="Systems nominal.",
    )
    out = await q.approve(a.approval_id, edited_text="Systems are nominal.")
    assert out.final_text == "Systems are nominal."


@pytest.mark.asyncio
async def test_approval_queue_reject() -> None:
    q = ApprovalQueue()
    a = await q.enqueue(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1",
        proposed_text="...",
    )
    out = await q.reject(a.approval_id)
    assert out.state is ApprovalState.REJECTED


@pytest.mark.asyncio
async def test_approval_queue_double_resolve_rejected() -> None:
    q = ApprovalQueue()
    a = await q.enqueue(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1",
        proposed_text="...",
    )
    await q.approve(a.approval_id)
    with pytest.raises(ValueError):
        await q.approve(a.approval_id)
    with pytest.raises(ValueError):
        await q.reject(a.approval_id)


@pytest.mark.asyncio
async def test_approval_queue_unknown_id() -> None:
    q = ApprovalQueue()
    with pytest.raises(KeyError):
        await q.approve("ap_does_not_exist")
    with pytest.raises(KeyError):
        await q.reject("ap_does_not_exist")


def test_approval_id_stable_for_same_inputs() -> None:
    a = compute_approval_id(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1", received_at=1.0,
    )
    b = compute_approval_id(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1", received_at=1.0,
    )
    assert a == b


def test_approval_id_changes_with_time() -> None:
    a = compute_approval_id(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1", received_at=1.0,
    )
    b = compute_approval_id(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1", received_at=2.0,
    )
    assert a != b


@pytest.mark.asyncio
async def test_approval_queue_fan_out() -> None:
    q = ApprovalQueue()
    listener = await q.subscribe("s")
    await q.enqueue(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1",
        proposed_text="...",
    )
    event = await asyncio.wait_for(listener.get(), timeout=0.5)
    assert event["kind"] == "approval.new"


@pytest.mark.asyncio
async def test_approval_queue_fan_out_other_session_not_delivered() -> None:
    q = ApprovalQueue()
    listener_a = await q.subscribe("sA")
    await q.enqueue(
        session_id="sB", ai=AiKind.ARTEMIS, turn_id="t1",
        proposed_text="...",
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(listener_a.get(), timeout=0.2)


# ─────────────────────────────────────────────────────────────────
# livekit.py — token mint wrappers
# ─────────────────────────────────────────────────────────────────


def test_mint_player_token_round_trips() -> None:
    tok = mint_player_token(
        config=LK_DEV, session_id="halyard-s01", identity="cross",
    )
    claims = decode_participant_token(tok, LK_DEV.api_secret)
    assert claims["sub"] == "cross"
    assert claims["video"]["room"] == "halyard-s01"
    assert claims["video"]["canPublish"] is True


def test_mint_keeper_token_prefixes_identity() -> None:
    tok = mint_keeper_token(config=LK_DEV, session_id="s", identity="abond")
    claims = decode_participant_token(tok, LK_DEV.api_secret)
    assert claims["sub"].startswith("keeper")


def test_mint_ai_token_no_av_publish() -> None:
    tok = mint_ai_token(config=LK_DEV, session_id="s", which="artemis")
    claims = decode_participant_token(tok, LK_DEV.api_secret)
    assert claims["video"]["canPublish"] is False
    assert claims["video"]["canPublishData"] is True


def test_mint_ai_token_rejects_bad_identity() -> None:
    with pytest.raises(ValueError):
        mint_ai_token(config=LK_DEV, session_id="s", which="daleks")


# ─────────────────────────────────────────────────────────────────
# app.py — HTTP routes
# ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client(tmp_path: Path):
    auth = KeeperAuthConfig(username="", password="")  # disabled
    reg = SessionRegistry(archive_root=tmp_path)
    approvals = ApprovalQueue()
    app = build_app(livekit=LK_DEV, auth=auth, sessions=reg, approvals=approvals)
    server = TestServer(app)
    async with TestClient(server) as c:
        yield c


@pytest_asyncio.fixture
async def authed_client(tmp_path: Path):
    auth = KeeperAuthConfig(username="abond", password="s3cret")
    reg = SessionRegistry(archive_root=tmp_path)
    approvals = ApprovalQueue()
    app = build_app(livekit=LK_DEV, auth=auth, sessions=reg, approvals=approvals)
    server = TestServer(app)
    async with TestClient(server) as c:
        yield c


def _basic_header(user: str, pw: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {encoded}"}


@pytest.mark.asyncio
async def test_healthz_open(client: TestClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_public_token_mint(client: TestClient) -> None:
    resp = await client.post(
        "/api/livekit/token",
        json={"session_id": "halyard-s01", "identity": "cross"},
    )
    assert resp.status == 200
    body = await resp.json()
    assert "token" in body and body["url"] == "ws://127.0.0.1:7880"
    claims = decode_participant_token(body["token"], LK_DEV.api_secret)
    assert claims["sub"] == "cross"


@pytest.mark.asyncio
async def test_public_token_mint_missing_fields(client: TestClient) -> None:
    resp = await client.post("/api/livekit/token", json={})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_public_token_mint_rejects_closed_session(
    client: TestClient, tmp_path: Path
) -> None:
    # Close the session via the keeper route first.
    # Close requires keeper auth only if enabled — this fixture has
    # auth disabled so we can hit it directly.
    await client.post("/api/sessions", json={"session_id": "closed-one"})
    await client.post("/api/sessions/closed-one/close")
    resp = await client.post(
        "/api/livekit/token",
        json={"session_id": "closed-one", "identity": "cross"},
    )
    assert resp.status == 403
    body = await resp.json()
    assert body["code"] == "session_closed"


@pytest.mark.asyncio
async def test_keeper_routes_require_auth_when_enabled(
    authed_client: TestClient,
) -> None:
    # No credentials → 401.
    resp = await authed_client.get("/api/sessions")
    assert resp.status == 401
    assert "WWW-Authenticate" in resp.headers

    # Bad credentials → 401.
    resp = await authed_client.get(
        "/api/sessions", headers=_basic_header("abond", "wrong")
    )
    assert resp.status == 401

    # Good credentials → 200.
    resp = await authed_client.get(
        "/api/sessions", headers=_basic_header("abond", "s3cret")
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_public_routes_unchanged_when_auth_enabled(
    authed_client: TestClient,
) -> None:
    """healthz and token mint are public regardless of keeper auth."""
    resp = await authed_client.get("/healthz")
    assert resp.status == 200
    resp = await authed_client.post(
        "/api/livekit/token",
        json={"session_id": "s", "identity": "cross"},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_session_lifecycle_via_routes(client: TestClient) -> None:
    # Create.
    resp = await client.post("/api/sessions", json={"session_id": "s1"})
    assert resp.status == 201

    # Duplicate.
    resp = await client.post("/api/sessions", json={"session_id": "s1"})
    assert resp.status == 409

    # List.
    resp = await client.get("/api/sessions")
    assert resp.status == 200
    body = await resp.json()
    assert any(s["id"] == "s1" for s in body["sessions"])

    # Pause.
    resp = await client.post("/api/sessions/s1/pause")
    assert resp.status == 200
    assert (await resp.json())["state"] == "paused"

    # Resume.
    resp = await client.post("/api/sessions/s1/resume")
    assert (await resp.json())["state"] == "open"

    # Close.
    resp = await client.post("/api/sessions/s1/close")
    assert (await resp.json())["state"] == "closed"

    # Can't pause after close.
    resp = await client.post("/api/sessions/s1/pause")
    assert resp.status == 409


@pytest.mark.asyncio
async def test_keeper_token_mint(client: TestClient) -> None:
    resp = await client.post(
        "/api/livekit/keeper-token",
        json={"session_id": "s1", "identity": "abond"},
    )
    assert resp.status == 200
    body = await resp.json()
    claims = decode_participant_token(body["token"], LK_DEV.api_secret)
    assert claims["sub"].startswith("keeper")


@pytest.mark.asyncio
async def test_approvals_roundtrip_via_routes(
    client: TestClient, tmp_path: Path
) -> None:
    # We can't POST approvals externally yet (NATS wiring is Sprint 7),
    # but the ApprovalQueue itself is addressable; enqueue via the
    # underlying object to exercise the routes.
    app = client.server.app
    from agi.halyard.keeper.app import KEY_APPROVALS

    approvals: ApprovalQueue = app[KEY_APPROVALS]
    item = await approvals.enqueue(
        session_id="s", ai=AiKind.ARTEMIS, turn_id="t1",
        proposed_text="Hello.",
    )

    # List.
    resp = await client.get("/api/keeper/approvals/s")
    assert resp.status == 200
    body = await resp.json()
    assert any(a["approval_id"] == item.approval_id for a in body["approvals"])

    # Approve.
    resp = await client.post(
        f"/api/keeper/approvals/{item.approval_id}/approve",
        json={},
    )
    assert resp.status == 200
    assert (await resp.json())["state"] == "approved"

    # Second approve fails.
    resp = await client.post(
        f"/api/keeper/approvals/{item.approval_id}/approve",
        json={},
    )
    assert resp.status == 409


@pytest.mark.asyncio
async def test_approval_approve_with_edit_via_route(
    client: TestClient,
) -> None:
    from agi.halyard.keeper.app import KEY_APPROVALS

    approvals: ApprovalQueue = client.server.app[KEY_APPROVALS]
    item = await approvals.enqueue(
        session_id="s", ai=AiKind.SIGMA4, turn_id="t1",
        proposed_text="Systems nominal.",
    )
    resp = await client.post(
        f"/api/keeper/approvals/{item.approval_id}/approve",
        json={"edited_text": "All systems nominal."},
    )
    body = await resp.json()
    assert body["final_text"] == "All systems nominal."


@pytest.mark.asyncio
async def test_approvals_ws_hello_and_push(client: TestClient) -> None:
    ws = await client.ws_connect("/ws/keeper/s")
    try:
        # Hello is sent first and is an empty snapshot for a fresh
        # session.
        hello = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
        assert hello["kind"] == "approvals.hello"

        # Enqueue something; the listener should receive it.
        from agi.halyard.keeper.app import KEY_APPROVALS

        approvals: ApprovalQueue = client.server.app[KEY_APPROVALS]
        item = await approvals.enqueue(
            session_id="s", ai=AiKind.ARTEMIS, turn_id="t1",
            proposed_text="Hi.",
        )
        event = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
        assert event["kind"] == "approval.new"
        assert event["approval"]["approval_id"] == item.approval_id
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_approval_not_found_404(client: TestClient) -> None:
    resp = await client.post(
        "/api/keeper/approvals/ap_missing/approve", json={}
    )
    assert resp.status == 404
