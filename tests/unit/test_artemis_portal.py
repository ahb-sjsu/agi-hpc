# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the ARTEMIS GM portal.

Covers auth, cache, and the HTTP endpoints. aiohttp's TestClient is
used with a dependency-injected PortalDeps so NATS and the wiki API
are never actually called.
"""

from __future__ import annotations

import time

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from agi.primer.artemis.chat.store import ChatStore
from agi.primer.artemis.portal.auth import (
    AuthError,
    extract_token,
    verify_keeper_token,
)
from agi.primer.artemis.portal.cache import SceneState, SheetCache
from agi.primer.artemis.portal.server import PortalDeps, build_app

SECRET = "test-secret-do-not-use-in-prod"


def _token(identity: str, *, secret: str = SECRET) -> str:
    claims = {"sub": identity, "iat": int(time.time())}
    return jwt.encode(claims, secret, algorithm="HS256")


# ─────────────────────────────────────────────────────────────────
# auth
# ─────────────────────────────────────────────────────────────────


def test_extract_token_prefers_bearer_then_cookie_then_query() -> None:
    assert extract_token({"Authorization": "Bearer abc"}, {}, {}) == "abc"
    assert extract_token({}, {"portal_token": "cki"}, {}) == "cki"
    assert extract_token({}, {}, {"t": "qrt"}) == "qrt"
    # Precedence: bearer beats cookie beats query.
    assert (
        extract_token({"Authorization": "Bearer a"}, {"portal_token": "b"}, {"t": "c"})
        == "a"
    )


def test_verify_keeper_token_accepts_keeper_prefix() -> None:
    p = verify_keeper_token(_token("keeper:andrew"), secret=SECRET)
    assert p.identity == "keeper:andrew"


def test_verify_keeper_token_rejects_player() -> None:
    with pytest.raises(AuthError):
        verify_keeper_token(_token("player:imogen"), secret=SECRET)


def test_verify_keeper_token_rejects_empty() -> None:
    with pytest.raises(AuthError):
        verify_keeper_token("", secret=SECRET)


def test_verify_keeper_token_rejects_wrong_secret() -> None:
    tok = _token("keeper:andrew")
    with pytest.raises(AuthError):
        verify_keeper_token(tok, secret="other-secret")


# ─────────────────────────────────────────────────────────────────
# cache
# ─────────────────────────────────────────────────────────────────


def test_sheet_cache_apply_merges_rows() -> None:
    c = SheetCache()
    c.apply("characters", [{"id": "imogen", "san": 70}])
    c.apply("characters", [{"id": "imogen", "hp": 11}])
    rows = c.rows("characters")
    assert len(rows) == 1
    assert rows[0] == {"id": "imogen", "san": 70, "hp": 11}


def test_sheet_cache_replace_all_overwrites() -> None:
    c = SheetCache()
    c.apply("characters", [{"id": "imogen", "san": 70}])
    c.replace_all("characters", [{"id": "arlo", "san": 55}])
    rows = c.rows("characters")
    assert [r["id"] for r in rows] == ["arlo"]


def test_sheet_cache_rows_sorted_by_name() -> None:
    c = SheetCache()
    c.apply("x", [{"id": "a", "name": "Zed"}, {"id": "b", "name": "Arc"}])
    assert [r["id"] for r in c.rows("x")] == ["b", "a"]


# ─────────────────────────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────────────────────────


def _mk_deps(
    *,
    published: list[tuple[str, bytes]] | None = None,
    wiki_hits: list[dict] | None = None,
) -> PortalDeps:
    published = published if published is not None else []

    async def publish(subject: str, payload: bytes) -> None:
        published.append((subject, payload))

    async def wiki(q: str, limit: int) -> list[dict]:
        return list(wiki_hits or [])

    return PortalDeps(
        chat_store=ChatStore(":memory:"),
        sheets=SheetCache(),
        scene=SceneState(),
        publish=publish,
        wiki_search=wiki,
        secret=SECRET,
    )


async def _client(aio_loop, deps: PortalDeps) -> TestClient:
    app = build_app(deps=deps)
    server = TestServer(app, loop=aio_loop)
    return TestClient(server, loop=aio_loop)


@pytest.fixture
def aio_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def _run(aio_loop, coro):
    return aio_loop.run_until_complete(coro)


def _keeper_headers(identity: str = "keeper:andrew") -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(identity)}"}


def test_http_healthz_is_public(aio_loop) -> None:
    deps = _mk_deps()

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.get("/api/healthz")
            assert r.status == 200
            body = await r.json()
            assert body["ok"] is True
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_requires_keeper_token(aio_loop) -> None:
    deps = _mk_deps()

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            # No token → 401
            r = await client.get("/api/chat/recent", params={"session": "s1"})
            assert r.status == 401
            # Player token → 401
            r2 = await client.get(
                "/api/chat/recent",
                params={"session": "s1"},
                headers={"Authorization": f"Bearer {_token('player:imogen')}"},
            )
            assert r2.status == 401
            # Keeper token → 200
            r3 = await client.get(
                "/api/chat/recent",
                params={"session": "s1"},
                headers=_keeper_headers(),
            )
            assert r3.status == 200
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_chat_search_hits_fts(aio_loop) -> None:
    deps = _mk_deps()
    deps.chat_store.append(
        session_id="s1",
        from_id="player:imogen",
        to_id="artemis",
        kind="player_to_artemis",
        body="check the vacuum seal",
    )

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.get(
                "/api/chat/search",
                params={"q": "vacuum", "session": "s1"},
                headers=_keeper_headers(),
            )
            assert r.status == 200
            body = await r.json()
            assert len(body["hits"]) == 1
            assert "vacuum" in body["hits"][0]["body"]
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_chat_whisper_publishes_and_persists(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/chat/whisper",
                json={
                    "session": "s1",
                    "to_id": "imogen",
                    "body": "you feel a chill",
                },
                headers=_keeper_headers(),
            )
            assert r.status == 200
            # Published to chat.out.player:imogen (prefix auto-added).
            assert published
            subj, _payload = published[-1]
            assert subj == "agi.rh.artemis.chat.out.player:imogen"
            # Persisted to the store too.
            assert deps.chat_store.count(session_id="s1") == 1
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_chat_whisper_missing_fields_is_400(aio_loop) -> None:
    deps = _mk_deps()

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/chat/whisper",
                json={"session": "s1"},  # no to_id / body
                headers=_keeper_headers(),
            )
            assert r.status == 400
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_chat_broadcast_publishes(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/chat/broadcast",
                json={"session": "s1", "body": "power flickers"},
                headers=_keeper_headers(),
            )
            assert r.status == 200
            subj, _payload = published[-1]
            assert subj == "agi.rh.artemis.chat.broadcast"
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_sheet_returns_cached_rows(aio_loop) -> None:
    deps = _mk_deps()
    deps.sheets.replace_all(
        "characters",
        [
            {"id": "imogen", "name": "IMOGEN ROTH", "san": 70},
            {"id": "arlo", "name": "ARLO VANCE", "san": 55},
        ],
    )

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.get(
                "/api/sheet",
                params={"name": "characters"},
                headers=_keeper_headers(),
            )
            assert r.status == 200
            body = await r.json()
            assert body["name"] == "characters"
            assert len(body["rows"]) == 2
            # Sorted by name → ARLO before IMOGEN.
            assert body["rows"][0]["id"] == "arlo"
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_wiki_search_swallows_backend_error(aio_loop) -> None:
    async def boom(q: str, limit: int):
        raise RuntimeError("wiki offline")

    deps = _mk_deps()
    deps.wiki_search = boom

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.get(
                "/api/wiki/search",
                params={"q": "mi-go"},
                headers=_keeper_headers(),
            )
            # Backend failure must NOT 5xx the portal — the UI just
            # renders an empty panel with a hint.
            assert r.status == 200
            body = await r.json()
            assert body["hits"] == []
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_scene_post_broadcasts_datachannel_payload(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/scene",
                json={"name": "Vault interior", "flags": "HOSTILE"},
                headers=_keeper_headers(),
            )
            assert r.status == 200
            subj, payload = published[-1]
            assert subj == "agi.rh.artemis.scene"
            import json

            assert json.loads(payload)["name"] == "Vault interior"
            # State is updated on the in-memory cache.
            assert deps.scene.name == "Vault interior"
        finally:
            await client.close()

    _run(aio_loop, _test())


# ─────────────────────────────────────────────────────────────────
# S1g — direct narration (silent-GM mode)
# ─────────────────────────────────────────────────────────────────


def test_http_say_publishes_to_direct_subject(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/say",
                json={"text": "The lights flicker."},
                headers=_keeper_headers(),
            )
            assert r.status == 200
            data = await r.json()
            assert data == {"ok": True, "chars": 19}
        finally:
            await client.close()

    _run(aio_loop, _test())
    # Published on the direct-say subject the avatar bridge subscribes to.
    assert published, "nothing published"
    subj, payload = published[-1]
    assert subj == "agi.rh.artemis.say.direct"
    import json

    assert json.loads(payload) == {"text": "The lights flicker."}


def test_http_say_rejects_empty(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/say",
                json={"text": "   "},
                headers=_keeper_headers(),
            )
            assert r.status == 400
        finally:
            await client.close()

    _run(aio_loop, _test())
    # No publish on empty input — keep the avatar queue clean.
    assert published == []


def test_http_say_rejects_player_token(aio_loop) -> None:
    deps = _mk_deps()

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/say",
                json={"text": "I should not be able to narrate."},
                headers={"Authorization": f"Bearer {_token('player:imogen')}"},
            )
            assert r.status == 401
        finally:
            await client.close()

    _run(aio_loop, _test())


def test_http_say_rejects_oversize(aio_loop) -> None:
    published: list[tuple[str, bytes]] = []
    deps = _mk_deps(published=published)

    async def _test():
        client = await _client(aio_loop, deps)
        await client.start_server()
        try:
            r = await client.post(
                "/api/say",
                json={"text": "x" * 2500},
                headers=_keeper_headers(),
            )
            assert r.status == 400
        finally:
            await client.close()

    _run(aio_loop, _test())
    # Cap is a belt-and-suspenders — XTTS synth of 2 kB+ blows through
    # the NATS 1 MB payload cap and makes the avatar look slow. Nothing
    # should have been published.
    assert published == []
