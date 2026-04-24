# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-keeper-backend aiohttp application.

Routes:

  Public (no auth — session-scoped):
    GET  /healthz
    POST /api/livekit/token           — mint a LiveKit JWT for a player

  Keeper-only (HTTP Basic; ``keeper.*`` route name prefix):
    GET  /api/sessions                — list
    POST /api/sessions                — create
    GET  /api/sessions/{id}           — read
    POST /api/sessions/{id}/pause
    POST /api/sessions/{id}/resume
    POST /api/sessions/{id}/close
    POST /api/livekit/keeper-token    — mint a Keeper-identity JWT
    GET  /api/keeper/approvals/{session_id}
    POST /api/keeper/approvals/{approval_id}/approve
    POST /api/keeper/approvals/{approval_id}/reject
    WS   /ws/keeper/{session_id}      — live approval feed

Sprint 6 v0: NATS wiring (subscribing artemis.say / sigma4.say
with approval flag) is stubbed out behind a feature-gate so the
service can stand up and mint tokens without the AIs running.
Sprint 7 finishes the NATS roundtrip.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

from aiohttp import WSMsgType, web

from .ai_stub import stub_reply
from .approvals import AiKind, ApprovalQueue, ApprovalState
from .auth import KeeperAuthConfig, build_keeper_middleware
from .livekit import LiveKitConfig, mint_keeper_token, mint_player_token
from .sessions import (
    InvalidTransition,
    SessionAlreadyExists,
    SessionNotFound,
    SessionRegistry,
    SessionState,
)

log = logging.getLogger("halyard.keeper.app")

# App keys.
KEY_LK: web.AppKey[LiveKitConfig] = web.AppKey("lk", LiveKitConfig)
KEY_SESSIONS: web.AppKey[SessionRegistry] = web.AppKey("sessions", SessionRegistry)
KEY_APPROVALS: web.AppKey[ApprovalQueue] = web.AppKey("approvals", ApprovalQueue)


# ─────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────


def _json_error(
    status: int, code: str, message: str = ""
) -> web.Response:
    return web.json_response(
        {"code": code, "message": message}, status=status
    )


async def _parse_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_json", "message": str(e)}),
            content_type="application/json",
        ) from None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_envelope"}),
            content_type="application/json",
        )
    return body


# ─────────────────────────────────────────────────────────────────
# Public routes
# ─────────────────────────────────────────────────────────────────


async def _healthz(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "halyard-keeper"})


async def _mint_player(request: web.Request) -> web.Response:
    """POST /api/livekit/token — mint a participant JWT.

    Body: ``{session_id, identity, name?}``.

    Public in Sprint-6 v0. The session registry's ensure() creates
    the session if missing but refuses if it's CLOSED — so a
    Keeper can evict players by closing the session.
    """
    body = await _parse_body(request)
    session_id = body.get("session_id")
    identity = body.get("identity")
    name = body.get("name")
    if not isinstance(session_id, str) or not session_id:
        return _json_error(400, "bad_envelope", "session_id required")
    if not isinstance(identity, str) or not identity:
        return _json_error(400, "bad_envelope", "identity required")

    sessions: SessionRegistry = request.app[KEY_SESSIONS]
    if not sessions.is_accepting_joins(session_id):
        return _json_error(
            403, "session_closed", f"session {session_id} is closed"
        )
    await sessions.ensure(session_id)

    lk: LiveKitConfig = request.app[KEY_LK]
    token = mint_player_token(
        config=lk,
        session_id=session_id,
        identity=identity,
        name=name if isinstance(name, str) else None,
    )
    return web.json_response({"token": token, "url": lk.url})


# ─────────────────────────────────────────────────────────────────
# Public AI-stub route — dev affordance until the real Whisper+
# LiveKit agent ingestion lands. See ai_stub.py for the design.
# ─────────────────────────────────────────────────────────────────


async def _ai_stub_turn(request: web.Request) -> web.Response:
    """POST /api/ai/{which}/stub-turn — in-persona canned reply.

    Body: ``{session_id, speaker, text}``. Returns
    ``{text, matched, ts}``. Public (no keeper auth); the request
    is harmless (no mutation, no LLM call).
    """
    which = request.match_info["which"]
    if which not in {"artemis", "sigma4"}:
        return _json_error(404, "not_found", f"unknown ai {which!r}")

    body = await _parse_body(request)
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return _json_error(400, "bad_envelope", "text required")
    if len(text) > 1024:
        return _json_error(400, "too_long", "max 1024 chars")

    reply = stub_reply(which=which, text=text)
    return web.json_response(
        {
            "text": reply.text,
            "matched": reply.matched,
            "ts": reply.ts,
            "kind": f"{which}.say",
        }
    )


# ─────────────────────────────────────────────────────────────────
# Keeper routes — registered with name prefix ``keeper.`` so the
# auth middleware picks them up.
# ─────────────────────────────────────────────────────────────────


async def _list_sessions(request: web.Request) -> web.Response:
    sessions: SessionRegistry = request.app[KEY_SESSIONS]
    items = [s.to_dict() for s in await sessions.list()]
    return web.json_response({"sessions": items})


async def _create_session(request: web.Request) -> web.Response:
    body = await _parse_body(request)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return _json_error(400, "bad_envelope", "session_id required")
    note = body.get("note", "")
    if not isinstance(note, str):
        note = str(note)
    sessions: SessionRegistry = request.app[KEY_SESSIONS]
    try:
        s = await sessions.create(session_id, note=note)
    except SessionAlreadyExists:
        return _json_error(409, "already_exists", session_id)
    return web.json_response(s.to_dict(), status=201)


async def _get_session(request: web.Request) -> web.Response:
    sessions: SessionRegistry = request.app[KEY_SESSIONS]
    try:
        s = await sessions.get(request.match_info["session_id"])
    except SessionNotFound:
        return _json_error(404, "not_found")
    return web.json_response(s.to_dict())


def _make_transition_handler(
    op: str,
) -> Any:
    async def _handler(request: web.Request) -> web.Response:
        sessions: SessionRegistry = request.app[KEY_SESSIONS]
        sid = request.match_info["session_id"]
        try:
            if op == "pause":
                s = await sessions.pause(sid)
            elif op == "resume":
                s = await sessions.resume(sid)
            else:
                s = await sessions.close(sid)
        except SessionNotFound:
            return _json_error(404, "not_found")
        except InvalidTransition as e:
            return _json_error(409, "invalid_transition", str(e))
        return web.json_response(s.to_dict())

    return _handler


async def _mint_keeper(request: web.Request) -> web.Response:
    body = await _parse_body(request)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return _json_error(400, "bad_envelope", "session_id required")
    identity = body.get("identity", "keeper")
    name = body.get("name")
    lk: LiveKitConfig = request.app[KEY_LK]
    token = mint_keeper_token(
        config=lk,
        session_id=session_id,
        identity=identity if isinstance(identity, str) else "keeper",
        name=name if isinstance(name, str) else None,
    )
    return web.json_response({"token": token, "url": lk.url})


async def _list_approvals(request: web.Request) -> web.Response:
    approvals: ApprovalQueue = request.app[KEY_APPROVALS]
    session_id = request.match_info["session_id"]
    items = await approvals.pending_for(session_id)
    return web.json_response(
        {"session_id": session_id, "approvals": [i.to_dict() for i in items]}
    )


async def _approve(request: web.Request) -> web.Response:
    approvals: ApprovalQueue = request.app[KEY_APPROVALS]
    approval_id = request.match_info["approval_id"]
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass
    edited = body.get("edited_text")
    try:
        item = await approvals.approve(
            approval_id,
            edited_text=edited if isinstance(edited, str) else None,
        )
    except KeyError:
        return _json_error(404, "not_found", approval_id)
    except ValueError as e:
        return _json_error(409, "already_resolved", str(e))
    return web.json_response(item.to_dict())


async def _reject(request: web.Request) -> web.Response:
    approvals: ApprovalQueue = request.app[KEY_APPROVALS]
    approval_id = request.match_info["approval_id"]
    try:
        item = await approvals.reject(approval_id)
    except KeyError:
        return _json_error(404, "not_found", approval_id)
    except ValueError as e:
        return _json_error(409, "already_resolved", str(e))
    return web.json_response(item.to_dict())


async def _approvals_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    approvals: ApprovalQueue = request.app[KEY_APPROVALS]
    session_id = request.match_info["session_id"]

    snapshot = await approvals.snapshot(session_id)
    await ws.send_json({"kind": "approvals.hello", "items": snapshot})

    q = await approvals.subscribe(session_id)
    # Two tasks: drain `q` → ws; watch client-side messages for close.
    async def _push():
        while True:
            event = await q.get()
            await ws.send_json(event)

    push_task = asyncio.create_task(_push())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.CLOSE:
                break
    finally:
        push_task.cancel()
        await approvals.unsubscribe(session_id, q)
    return ws


# ─────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────


def build_app(
    *,
    livekit: LiveKitConfig | None = None,
    auth: KeeperAuthConfig | None = None,
    sessions: SessionRegistry | None = None,
    approvals: ApprovalQueue | None = None,
) -> web.Application:
    """Build the aiohttp application.

    Dependencies are injectable for tests. Middlewares run in
    declaration order; keeper auth first so unauthorized calls
    fail before touching the registry.
    """
    auth_cfg = auth if auth is not None else KeeperAuthConfig.from_env()
    lk_cfg = livekit if livekit is not None else LiveKitConfig.from_env()

    app = web.Application(
        middlewares=[build_keeper_middleware(auth_cfg)],
    )
    app[KEY_LK] = lk_cfg
    app[KEY_SESSIONS] = sessions if sessions is not None else SessionRegistry()
    app[KEY_APPROVALS] = approvals if approvals is not None else ApprovalQueue()

    # Public.
    app.router.add_get("/healthz", _healthz, name="public.healthz")
    app.router.add_post(
        "/api/livekit/token", _mint_player, name="public.livekit_token"
    )
    app.router.add_post(
        "/api/ai/{which}/stub-turn", _ai_stub_turn, name="public.ai_stub_turn"
    )

    # Keeper — all names prefixed so the middleware picks them up.
    app.router.add_get(
        "/api/sessions", _list_sessions, name="keeper.sessions_list"
    )
    app.router.add_post(
        "/api/sessions", _create_session, name="keeper.sessions_create"
    )
    app.router.add_get(
        "/api/sessions/{session_id}",
        _get_session,
        name="keeper.sessions_get",
    )
    app.router.add_post(
        "/api/sessions/{session_id}/pause",
        _make_transition_handler("pause"),
        name="keeper.sessions_pause",
    )
    app.router.add_post(
        "/api/sessions/{session_id}/resume",
        _make_transition_handler("resume"),
        name="keeper.sessions_resume",
    )
    app.router.add_post(
        "/api/sessions/{session_id}/close",
        _make_transition_handler("close"),
        name="keeper.sessions_close",
    )
    app.router.add_post(
        "/api/livekit/keeper-token",
        _mint_keeper,
        name="keeper.livekit_keeper_token",
    )
    app.router.add_get(
        "/api/keeper/approvals/{session_id}",
        _list_approvals,
        name="keeper.approvals_list",
    )
    app.router.add_post(
        "/api/keeper/approvals/{approval_id}/approve",
        _approve,
        name="keeper.approvals_approve",
    )
    app.router.add_post(
        "/api/keeper/approvals/{approval_id}/reject",
        _reject,
        name="keeper.approvals_reject",
    )
    app.router.add_get(
        "/ws/keeper/{session_id}", _approvals_ws, name="keeper.approvals_ws"
    )
    return app


# re-export for tests
__all__ = [
    "build_app",
    "KEY_LK",
    "KEY_SESSIONS",
    "KEY_APPROVALS",
    "AiKind",
    "ApprovalState",
    "SessionState",
    "asdict",
]
