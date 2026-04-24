# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-3 tests — halyard-state aiohttp REST + WebSocket API.

Uses aiohttp's built-in test client (no live server). The tmp_path
fixture provides an isolated filesystem root for the Store.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from agi.halyard.state.api import WsBroadcaster, build_app
from agi.halyard.state.store import Store


def _sheet(pc_id: str = "cross") -> dict:
    return {
        "schema_version": "1.0",
        "session_id": "halyard-s01",
        "pc_id": pc_id,
        "identity": {
            "name": "Halden Cross",
            "age": 41,
            "origin": "Luna",
            "role": "security_officer",
            "chassis": "baseline_human",
            "credit_rating": 35,
        },
        "characteristics": {
            "str": 75, "con": 75, "siz": 65, "dex": 70,
            "app": 55, "int": 65, "pow": 55, "edu": 60,
        },
        "derived": {
            "hp_max": 14, "mp_max": 11, "san_starting": 55,
            "san_max": 65, "luck_max": 55, "move": 8,
            "build": 1, "damage_bonus": "+1D4", "dodge_base": 35,
        },
        "skills": {"firearms (handgun)": {"value": 70, "base": 20}},
        "bonds": [],
        "status": {
            "hp_current": 14, "mp_current": 11,
            "san_current": 55, "luck_current": 55,
        },
        "campaign": {"faction_loyalty": "clean"},
    }


@pytest_asyncio.fixture
async def client(tmp_path: Path):
    store = Store(archive_root=tmp_path)
    broadcaster = WsBroadcaster()
    app = build_app(store=store, broadcaster=broadcaster)
    server = TestServer(app)
    async with TestClient(server) as c:
        # Stash store on the client for setup assertions.
        c._hstore = store  # type: ignore[attr-defined]
        c._hbroadcaster = broadcaster  # type: ignore[attr-defined]
        yield c


# ─────────────────────────────────────────────────────────────────
# Basic routing
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healthz(client: TestClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status == 200
    body = await resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_list_empty_session(client: TestClient) -> None:
    resp = await client.get("/api/sheets/halyard-s01")
    assert resp.status == 200
    body = await resp.json()
    assert body["pc_ids"] == []


@pytest.mark.asyncio
async def test_get_missing_sheet_404(client: TestClient) -> None:
    resp = await client.get("/api/sheets/halyard-s01/ghost")
    assert resp.status == 404
    body = await resp.json()
    assert body["code"] == "not_found"


# ─────────────────────────────────────────────────────────────────
# Create
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_sheet(client: TestClient) -> None:
    resp = await client.post(
        "/api/sheets/halyard-s01/cross", json=_sheet()
    )
    assert resp.status == 201
    body = await resp.json()
    assert body["pc_id"] == "cross"


@pytest.mark.asyncio
async def test_create_duplicate_409(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    resp = await client.post(
        "/api/sheets/halyard-s01/cross", json=_sheet()
    )
    assert resp.status == 409
    body = await resp.json()
    assert body["code"] == "already_exists"


@pytest.mark.asyncio
async def test_create_invalid_sheet_400(client: TestClient) -> None:
    bad = _sheet()
    bad["identity"]["role"] = "xenotheologian"
    resp = await client.post("/api/sheets/halyard-s01/cross", json=bad)
    assert resp.status == 400
    body = await resp.json()
    assert body["code"] == "schema"


# ─────────────────────────────────────────────────────────────────
# Get
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_sheet_after_create(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    resp = await client.get("/api/sheets/halyard-s01/cross")
    assert resp.status == 200
    body = await resp.json()
    assert body["identity"]["name"] == "Halden Cross"


# ─────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_keeper_patch_ok(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    envelope = {
        "author": "keeper",
        "patch": [
            {"op": "replace", "path": "/status/hp_current", "value": 9}
        ],
        "reason": "took 5",
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/cross/patch", json=envelope
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["status"]["hp_current"] == 9


@pytest.mark.asyncio
async def test_player_patch_to_keeper_field_403(
    client: TestClient,
) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    envelope = {
        "author": "player",
        "author_pc_id": "cross",
        "patch": [
            {
                "op": "replace",
                "path": "/campaign/faction_loyalty",
                "value": "hollow_hand",
            }
        ],
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/cross/patch", json=envelope
    )
    assert resp.status == 403
    body = await resp.json()
    assert body["code"] == "authz_denied"


@pytest.mark.asyncio
async def test_player_patch_other_pc_403(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    await client.post(
        "/api/sheets/halyard-s01/halverson",
        json=_sheet(pc_id="halverson"),
    )
    envelope = {
        "author": "player",
        "author_pc_id": "cross",
        "patch": [
            {"op": "replace", "path": "/status/hp_current", "value": 1}
        ],
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/halverson/patch", json=envelope
    )
    assert resp.status == 403


@pytest.mark.asyncio
async def test_patch_missing_returns_404(client: TestClient) -> None:
    envelope = {
        "author": "keeper",
        "patch": [
            {"op": "replace", "path": "/status/hp_current", "value": 1}
        ],
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/ghost/patch", json=envelope
    )
    assert resp.status == 404


@pytest.mark.asyncio
async def test_patch_invalid_400(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    envelope = {
        "author": "keeper",
        "patch": [{"op": "move", "from": "/a", "path": "/b"}],
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/cross/patch", json=envelope
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["code"] == "invalid_patch"


@pytest.mark.asyncio
async def test_patch_would_violate_schema_400(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    envelope = {
        "author": "keeper",
        "patch": [
            {"op": "replace", "path": "/status/san_current", "value": 999}
        ],
    }
    resp = await client.post(
        "/api/sheets/halyard-s01/cross/patch", json=envelope
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["code"] == "schema_violation"


@pytest.mark.asyncio
async def test_bad_envelope_400(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    # Missing 'patch' list.
    resp = await client.post(
        "/api/sheets/halyard-s01/cross/patch",
        json={"author": "keeper"},
    )
    assert resp.status == 400


# ─────────────────────────────────────────────────────────────────
# WebSocket fan-out
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ws_hello_lists_session_pcs(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    ws = await client.ws_connect("/ws/sheets/halyard-s01")
    try:
        msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
        assert msg["kind"] == "session.hello"
        assert "cross" in msg["pc_ids"]
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_ws_receives_patch_update(client: TestClient) -> None:
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())
    ws = await client.ws_connect("/ws/sheets/halyard-s01")
    try:
        # Consume the hello.
        hello = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
        assert hello["kind"] == "session.hello"

        # Fire a patch via REST; WS should observe the update.
        envelope = {
            "author": "keeper",
            "patch": [
                {"op": "replace", "path": "/status/hp_current", "value": 5}
            ],
        }
        patch_resp = await client.post(
            "/api/sheets/halyard-s01/cross/patch", json=envelope
        )
        assert patch_resp.status == 200

        update = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
        assert update["kind"] == "sheet.update"
        assert update["pc_id"] == "cross"
        assert update["sheet"]["status"]["hp_current"] == 5
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_two_ws_subscribers_both_receive(
    client: TestClient,
) -> None:
    """Core acceptance criterion: two clients subscribed to the
    same session both see an update when a patch is applied."""
    await client.post("/api/sheets/halyard-s01/cross", json=_sheet())

    ws_a = await client.ws_connect("/ws/sheets/halyard-s01")
    ws_b = await client.ws_connect("/ws/sheets/halyard-s01")
    try:
        # Drain hellos.
        for ws in (ws_a, ws_b):
            msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
            assert msg["kind"] == "session.hello"

        envelope = {
            "author": "keeper",
            "patch": [
                {"op": "replace", "path": "/status/hp_current", "value": 6}
            ],
        }
        patch_resp = await client.post(
            "/api/sheets/halyard-s01/cross/patch", json=envelope
        )
        assert patch_resp.status == 200

        # Both sockets see it.
        update_a = await asyncio.wait_for(ws_a.receive_json(), timeout=1.0)
        update_b = await asyncio.wait_for(ws_b.receive_json(), timeout=1.0)
        assert update_a["kind"] == "sheet.update"
        assert update_b["kind"] == "sheet.update"
        assert update_a["sheet"]["status"]["hp_current"] == 6
    finally:
        await ws_a.close()
        await ws_b.close()


@pytest.mark.asyncio
async def test_ws_other_session_not_broadcasted(
    client: TestClient,
) -> None:
    """A WS subscribed to session A must NOT see updates from
    session B. Cross-session isolation."""
    await client.post(
        "/api/sheets/halyard-sA/cross", json=_sheet()
    )
    await client.post(
        "/api/sheets/halyard-sB/cross",
        json={**_sheet(), "session_id": "halyard-sB"},
    )

    ws_a = await client.ws_connect("/ws/sheets/halyard-sA")
    try:
        # Drain hello.
        await asyncio.wait_for(ws_a.receive_json(), timeout=1.0)

        # Patch in session B.
        envelope = {
            "author": "keeper",
            "patch": [
                {"op": "replace", "path": "/status/hp_current", "value": 2}
            ],
        }
        await client.post(
            "/api/sheets/halyard-sB/cross/patch", json=envelope
        )

        # Session A's socket should time out — nothing arrives.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws_a.receive_json(), timeout=0.3)
    finally:
        await ws_a.close()
