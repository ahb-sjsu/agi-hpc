# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Keeper-portal HTTP server.

aiohttp service that aggregates:
  - ChatStore    — SQLite+FTS5 transcript (chat.store)
  - SheetCache   — latest rows from agi.rh.artemis.sheet.* on NATS
  - Wiki search  — GET /api/wiki/search on an external memory-wiki
  - NATS publish — keeper whisper + broadcast + scene updates

Dependencies are injected so the tests can drive the endpoints with
fakes. Production wiring lives in ``build_app_from_env``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiohttp import web

from ..chat.service import (
    SUBJECT_BROADCAST,
    SUBJECT_OUT_PREFIX,
    ChatMessage,
)
from ..chat.store import ChatStore
from .auth import AuthError, KeeperPrincipal, extract_token, verify_keeper_token
from .cache import SceneState, SheetCache

log = logging.getLogger("artemis.portal.server")


NatsPublisher = Callable[[str, bytes], Awaitable[None]]
WikiSearch = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


@dataclass
class PortalDeps:
    chat_store: ChatStore
    sheets: SheetCache
    scene: SceneState
    publish: NatsPublisher
    wiki_search: WikiSearch
    secret: str


# ─────────────────────────────────────────────────────────────────
# Auth middleware
# ─────────────────────────────────────────────────────────────────


@web.middleware
async def _auth_middleware(request: web.Request, handler) -> web.StreamResponse:
    # Public surface — healthz + static assets never require auth.
    if request.path in ("/api/healthz", "/healthz"):
        return await handler(request)

    deps: PortalDeps = request.app["deps"]
    token = extract_token(
        {k: v for k, v in request.headers.items()},
        (
            {k: v.value for k, v in request.cookies.items()}
            if hasattr(request.cookies, "items")
            else dict(request.cookies)
        ),
        dict(request.query),
    )
    try:
        principal = verify_keeper_token(token, secret=deps.secret)
    except AuthError as e:
        return web.json_response({"error": str(e)}, status=401)
    request["principal"] = principal
    return await handler(request)


def _principal(request: web.Request) -> KeeperPrincipal:
    return request["principal"]


# ─────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────


async def handle_healthz(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_chat_recent(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    session = request.query.get("session") or ""
    if not session:
        return web.json_response({"error": "missing session"}, status=400)
    limit = _int_param(request, "limit", default=50, cap=500)
    msgs = deps.chat_store.recent(session_id=session, limit=limit)
    return web.json_response(
        {
            "session": session,
            "messages": [_msg_dict(m) for m in msgs],
        }
    )


async def handle_chat_thread(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    session = request.query.get("session") or ""
    participant = request.query.get("participant") or _principal(request).identity
    if not session:
        return web.json_response({"error": "missing session"}, status=400)
    limit = _int_param(request, "limit", default=500, cap=2000)
    msgs = deps.chat_store.thread(
        session_id=session,
        participant_id=participant,
        limit=limit,
    )
    return web.json_response(
        {
            "session": session,
            "participant": participant,
            "messages": [_msg_dict(m) for m in msgs],
        }
    )


async def handle_chat_search(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    q = (request.query.get("q") or "").strip()
    session = request.query.get("session")
    limit = _int_param(request, "limit", default=100, cap=500)
    msgs = deps.chat_store.search(query=q, session_id=session, limit=limit)
    return web.json_response(
        {
            "q": q,
            "session": session,
            "hits": [_msg_dict(m) for m in msgs],
        }
    )


async def handle_chat_whisper(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    payload = await _json_body(request)
    session = str(payload.get("session") or "").strip()
    to_id = str(payload.get("to_id") or "").strip()
    body = str(payload.get("body") or "").strip()
    if not session or not to_id or not body:
        return web.json_response({"error": "session/to_id/body required"}, status=400)

    msg = ChatMessage(
        kind="keeper_to_player",
        session_id=session,
        from_id=_principal(request).identity,
        to_id=to_id if to_id.startswith("player:") else f"player:{to_id}",
        body=body,
        ts=time.time(),
    )
    deps.chat_store.append(
        session_id=msg.session_id,
        from_id=msg.from_id,
        to_id=msg.to_id,
        kind=msg.kind,
        body=msg.body,
        ts=msg.ts,
    )
    subj = f"{SUBJECT_OUT_PREFIX}.{msg.to_id}"
    await deps.publish(subj, msg.to_json())
    return web.json_response({"ok": True})


async def handle_chat_broadcast(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    payload = await _json_body(request)
    session = str(payload.get("session") or "").strip()
    body = str(payload.get("body") or "").strip()
    if not session or not body:
        return web.json_response({"error": "session/body required"}, status=400)
    msg = ChatMessage(
        kind="keeper_to_all",
        session_id=session,
        from_id=_principal(request).identity,
        body=body,
        ts=time.time(),
    )
    deps.chat_store.append(
        session_id=msg.session_id,
        from_id=msg.from_id,
        to_id=None,
        kind=msg.kind,
        body=msg.body,
        ts=msg.ts,
    )
    await deps.publish(SUBJECT_BROADCAST, msg.to_json())
    return web.json_response({"ok": True})


async def handle_sheet(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    name = request.query.get("name") or "characters"
    return web.json_response(
        {
            "name": name,
            "rows": deps.sheets.rows(name),
            "available_sheets": deps.sheets.names(),
        }
    )


async def handle_wiki_search(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    q = (request.query.get("q") or "").strip()
    limit = _int_param(request, "limit", default=10, cap=50)
    if not q:
        return web.json_response({"q": q, "hits": []})
    try:
        hits = await deps.wiki_search(q, limit)
    except Exception as e:  # noqa: BLE001
        # Wiki service is optional — don't crash the portal if it's
        # down. The UI renders "wiki offline" in that case.
        log.warning("wiki search failed: %s", e)
        hits = []
    return web.json_response({"q": q, "hits": hits})


async def handle_scene_get(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    return web.json_response(deps.scene.to_dict())


async def handle_scene_post(request: web.Request) -> web.Response:
    deps: PortalDeps = request.app["deps"]
    payload = await _json_body(request)
    name = str(payload.get("name") or deps.scene.name).strip()
    flags = str(payload.get("flags") or deps.scene.flags).strip()
    deps.scene.name = name
    deps.scene.flags = flags
    # Broadcast the scene change to every open table.html via the same
    # kind="artemis.scene" DataChannel path already wired on the avatar.
    payload_out = json.dumps(
        {"kind": "artemis.scene", "name": name, "flags": flags},
        separators=(",", ":"),
    ).encode()
    await deps.publish("agi.rh.artemis.scene", payload_out)
    return web.json_response(deps.scene.to_dict())


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _msg_dict(m) -> dict[str, Any]:
    return {
        "id": m.id,
        "ts": m.ts,
        "session_id": m.session_id,
        "from_id": m.from_id,
        "to_id": m.to_id,
        "kind": m.kind,
        "body": m.body,
        "corr_id": m.corr_id,
    }


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


def _int_param(request: web.Request, name: str, *, default: int, cap: int) -> int:
    raw = request.query.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(1, min(cap, v))


# ─────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────


def build_app(*, deps: PortalDeps) -> web.Application:
    """Construct the aiohttp app with injected deps."""
    app = web.Application(middlewares=[_auth_middleware])
    app["deps"] = deps
    app.router.add_get("/api/healthz", handle_healthz)
    app.router.add_get("/api/chat/recent", handle_chat_recent)
    app.router.add_get("/api/chat/thread", handle_chat_thread)
    app.router.add_get("/api/chat/search", handle_chat_search)
    app.router.add_post("/api/chat/whisper", handle_chat_whisper)
    app.router.add_post("/api/chat/broadcast", handle_chat_broadcast)
    app.router.add_get("/api/sheet", handle_sheet)
    app.router.add_get("/api/wiki/search", handle_wiki_search)
    app.router.add_get("/api/scene", handle_scene_get)
    app.router.add_post("/api/scene", handle_scene_post)
    return app


async def run_forever(port: int, *, deps: PortalDeps) -> None:
    """Run the aiohttp app until cancelled."""
    app = build_app(deps=deps)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=port)
    await site.start()
    log.info("portal listening on 127.0.0.1:%d", port)
    try:
        while True:
            import asyncio

            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
