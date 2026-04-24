# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-state REST + WebSocket API (aiohttp).

Routes:

- ``GET  /healthz``                                 — liveness.
- ``GET  /api/sheets/{session_id}``                 — list pc_ids.
- ``GET  /api/sheets/{session_id}/{pc_id}``         — current sheet.
- ``POST /api/sheets/{session_id}/{pc_id}/patch``   — apply a patch
  (authorization via the envelope's ``author`` field; reuses the
  same authz logic as the NATS bridge).
- ``WS   /ws/sheets/{session_id}``                  — live-updates
  stream. Every sheet update for the session is fanned out.

Auth is intentionally minimal in Sprint 3 — the service is
reachable only from Atlas's Caddy front, and in production the
Keeper is the only principal who'd POST a patch over HTTP. The
halyard-keeper-backend (Sprint 6) will layer real authN/Z on top.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import WSMsgType, web

from ..bridge import StateBridge
from ..schema import validate_sheet
from ..schema.access import Author
from ..schema.patch import (
    AuthorizationError,
    InvalidPatchError,
    PatchRequest,
)
from ..store import (
    AlreadyExistsError,
    NotFoundError,
    SchemaError,
    Store,
)

log = logging.getLogger("halyard.state.api")

# aiohttp application keys — typed-ish access to stored services.
KEY_STORE: web.AppKey[Store] = web.AppKey("store", Store)
KEY_BRIDGE: web.AppKey[StateBridge] = web.AppKey("bridge", StateBridge)
KEY_BROADCASTER: web.AppKey["WsBroadcaster"] = web.AppKey(
    "broadcaster", object  # type: ignore[arg-type]
)


# ─────────────────────────────────────────────────────────────────
# WebSocket fan-out
# ─────────────────────────────────────────────────────────────────


class WsBroadcaster:
    """Tracks WS subscribers per session and fans out updates.

    Used both by the :func:`_sheets_ws` handler (which adds a
    connection) and by :class:`StateBridge` (which calls
    :meth:`broadcast` when a sheet is updated).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[web.WebSocketResponse]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, session_id: str, ws: web.WebSocketResponse
    ) -> None:
        async with self._lock:
            self._subscribers.setdefault(session_id, set()).add(ws)

    async def unregister(
        self, session_id: str, ws: web.WebSocketResponse
    ) -> None:
        async with self._lock:
            subs = self._subscribers.get(session_id)
            if subs is None:
                return
            subs.discard(ws)
            if not subs:
                self._subscribers.pop(session_id, None)

    async def broadcast(
        self, session_id: str, payload: dict[str, Any]
    ) -> int:
        """Send payload to every subscriber for the session.

        Returns the count of successful sends. Closed / failing
        sockets are dropped silently.
        """
        async with self._lock:
            # Snapshot so a concurrent register/unregister doesn't
            # fight us mid-iteration.
            subs = list(self._subscribers.get(session_id, ()))
        if not subs:
            return 0
        sent = 0
        for ws in subs:
            if ws.closed:
                continue
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception as e:  # noqa: BLE001
                log.warning("ws send failed: %s", e)
        return sent

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, ()))


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────


async def _healthz(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "halyard-state"})


async def _list_sheets(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    store: Store = request.app[KEY_STORE]
    ids = await store.list_sheets(session_id)
    return web.json_response({"session_id": session_id, "pc_ids": ids})


async def _get_sheet(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    pc_id = request.match_info["pc_id"]
    store: Store = request.app[KEY_STORE]
    try:
        sheet = await store.get(session_id, pc_id)
    except NotFoundError:
        raise web.HTTPNotFound(
            text=json.dumps({"code": "not_found", "pc_id": pc_id}),
            content_type="application/json",
        ) from None
    return web.json_response(sheet)


async def _create_sheet(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    pc_id = request.match_info["pc_id"]
    store: Store = request.app[KEY_STORE]
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_json", "message": str(e)}),
            content_type="application/json",
        ) from None
    try:
        validate_sheet(
            {**body, "session_id": session_id, "pc_id": pc_id}
        )
    except Exception as e:  # noqa: BLE001
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "schema", "message": str(e)}),
            content_type="application/json",
        ) from None
    try:
        sheet = await store.create(session_id, pc_id, body)
    except AlreadyExistsError:
        raise web.HTTPConflict(
            text=json.dumps({"code": "already_exists", "pc_id": pc_id}),
            content_type="application/json",
        ) from None
    except SchemaError as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "schema", "message": str(e)}),
            content_type="application/json",
        ) from None
    return web.json_response(sheet, status=201)


async def _patch_sheet(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    pc_id = request.match_info["pc_id"]
    store: Store = request.app[KEY_STORE]
    broadcaster: WsBroadcaster = request.app[KEY_BROADCASTER]

    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_json", "message": str(e)}),
            content_type="application/json",
        ) from None

    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_envelope"}),
            content_type="application/json",
        )

    # Envelope parsing mirrors the bridge's rules.
    author_raw = body.get("author", "keeper")
    try:
        author = Author(author_raw)
    except ValueError:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"code": "bad_envelope", "message": f"unknown author {author_raw!r}"}
            ),
            content_type="application/json",
        ) from None

    author_pc_id = body.get("author_pc_id")
    if author is Author.PLAYER and not isinstance(author_pc_id, str):
        msg = "player-origin requires author_pc_id"
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "bad_envelope", "message": msg}),
            content_type="application/json",
        )

    patch = body.get("patch")
    if not isinstance(patch, list):
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"code": "bad_envelope", "message": "missing or non-list 'patch'"}
            ),
            content_type="application/json",
        )

    req = PatchRequest(
        session_id=session_id,
        target_pc_id=pc_id,
        author=author,
        author_pc_id=author_pc_id,
        patch=patch,
        reason=str(body.get("reason", "")),
    )

    try:
        sheet = await store.patch(req)
    except NotFoundError:
        raise web.HTTPNotFound(
            text=json.dumps({"code": "not_found", "pc_id": pc_id}),
            content_type="application/json",
        ) from None
    except AuthorizationError as e:
        raise web.HTTPForbidden(
            text=json.dumps({"code": "authz_denied", "message": str(e)}),
            content_type="application/json",
        ) from None
    except InvalidPatchError as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "invalid_patch", "message": str(e)}),
            content_type="application/json",
        ) from None
    except SchemaError as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"code": "schema_violation", "message": str(e)}),
            content_type="application/json",
        ) from None

    # Fan-out to WS listeners (best-effort; we've already
    # persisted, so a fan-out failure doesn't undo the patch).
    await broadcaster.broadcast(
        session_id,
        {"kind": "sheet.update", "pc_id": pc_id, "sheet": sheet},
    )
    return web.json_response(sheet)


async def _sheets_ws(request: web.Request) -> web.WebSocketResponse:
    """Subscribe to sheet updates for a session."""
    session_id = request.match_info["session_id"]
    broadcaster: WsBroadcaster = request.app[KEY_BROADCASTER]

    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    await broadcaster.register(session_id, ws)
    try:
        # Hello message: lists current sheets so the client knows
        # what's in the session right now.
        store: Store = request.app[KEY_STORE]
        ids = await store.list_sheets(session_id)
        await ws.send_json(
            {"kind": "session.hello", "session_id": session_id, "pc_ids": ids}
        )
        async for msg in ws:
            # We don't currently accept client→server WS traffic.
            # Ignore pings / ignore messages; keep the connection
            # open until it closes or heartbeat fails.
            if msg.type == WSMsgType.CLOSE:
                break
    finally:
        await broadcaster.unregister(session_id, ws)
    return ws


# ─────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────


def build_app(
    *,
    store: Store | None = None,
    broadcaster: WsBroadcaster | None = None,
) -> web.Application:
    """Build the aiohttp application.

    Dependencies are injectable for tests. In production the
    service's entrypoint constructs a real Store (filesystem
    backed), a WsBroadcaster, and a StateBridge, and passes them
    here — the bridge wires its ``ws_broadcast`` to the
    broadcaster's ``broadcast`` method so NATS-origin updates also
    reach the browsers.
    """
    app = web.Application()
    app[KEY_STORE] = store if store is not None else Store()
    app[KEY_BROADCASTER] = (
        broadcaster if broadcaster is not None else WsBroadcaster()
    )

    app.router.add_get("/healthz", _healthz)
    app.router.add_get(
        "/api/sheets/{session_id}", _list_sheets
    )
    app.router.add_get(
        "/api/sheets/{session_id}/{pc_id}", _get_sheet
    )
    app.router.add_post(
        "/api/sheets/{session_id}/{pc_id}", _create_sheet
    )
    app.router.add_post(
        "/api/sheets/{session_id}/{pc_id}/patch", _patch_sheet
    )
    app.router.add_get(
        "/ws/sheets/{session_id}", _sheets_ws
    )
    return app
