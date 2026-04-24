# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Keeper authentication middleware.

HTTP Basic in v1 with an optional IP allow-list. Sits in front of
any route tagged as Keeper-only. The token-mint endpoint is
deliberately NOT Keeper-only — it takes a session-scoped join
request, and the session-level authz lives on the sessions
registry (closed sessions reject).

v2 (later sprints) swaps in proper session-based OAuth.
"""

from __future__ import annotations

import base64
import hmac
import ipaddress
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("halyard.keeper.auth")


@dataclass(frozen=True)
class KeeperAuthConfig:
    """Authentication configuration for Keeper routes.

    ``username`` + ``password_hash`` come from env. Empty password
    means "no auth required" — the default for dev. Production
    deployments must set real credentials.
    """

    username: str
    password: str              # plaintext — HTTP Basic is cleartext-ish
    ip_allowlist: tuple[str, ...] = ()  # CIDR strings; empty = no filter

    @classmethod
    def from_env(cls) -> "KeeperAuthConfig":
        return cls(
            username=os.environ.get("KEEPER_USERNAME", ""),
            password=os.environ.get("KEEPER_PASSWORD", ""),
            ip_allowlist=tuple(
                s.strip()
                for s in os.environ.get("KEEPER_IP_ALLOWLIST", "").split(",")
                if s.strip()
            ),
        )

    def is_enabled(self) -> bool:
        return bool(self.username and self.password)


# ─────────────────────────────────────────────────────────────────
# IP check
# ─────────────────────────────────────────────────────────────────


def ip_allowed(ip_str: str, allowlist: tuple[str, ...]) -> bool:
    """True iff ``ip_str`` is inside any CIDR in ``allowlist``.

    An empty allowlist permits everything — this is the Sprint-6
    dev default; production sets a specific list (e.g. the
    Keeper's home range plus Tailscale).
    """
    if not allowlist:
        return True
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for cidr in allowlist:
        try:
            if ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


# ─────────────────────────────────────────────────────────────────
# HTTP Basic parse
# ─────────────────────────────────────────────────────────────────


def parse_basic_header(header: str | None) -> tuple[str, str] | None:
    """Return ``(user, pass)`` from an HTTP Basic header, or None.

    Defensive: malformed headers return None rather than raising.
    """
    if not header or not header.startswith("Basic "):
        return None
    encoded = header[len("Basic ") :].strip()
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return None
    if ":" not in decoded:
        return None
    user, pw = decoded.split(":", 1)
    return user, pw


# ─────────────────────────────────────────────────────────────────
# Middleware factory
# ─────────────────────────────────────────────────────────────────


def _unauthorized() -> web.Response:
    return web.Response(
        status=401,
        headers={"WWW-Authenticate": 'Basic realm="halyard-keeper"'},
        text="unauthorized",
    )


_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]
_Middleware = Callable[[web.Request, _Handler], Awaitable[web.StreamResponse]]


def build_keeper_middleware(config: KeeperAuthConfig) -> _Middleware:
    """Build an aiohttp ``@middleware``-compatible callable.

    Only runs for routes whose name starts with ``keeper.`` — the
    app factory registers routes with that prefix to opt into
    keeper auth. Non-keeper routes bypass.
    """

    @web.middleware
    async def _middleware(
        request: web.Request,
        handler: _Handler,
    ) -> web.StreamResponse:
        route_name = request.match_info.route.name or ""
        if not route_name.startswith("keeper."):
            return await handler(request)

        # IP allow-list first — cheaper than password check and a
        # harder security floor.
        if config.ip_allowlist:
            remote = request.remote or ""
            # aiohttp returns remote as e.g. ``"127.0.0.1"``; for
            # IPv6 it may be bracketed.
            remote = remote.lstrip("[").rstrip("]")
            if not ip_allowed(remote, config.ip_allowlist):
                log.info(
                    "keeper auth: %s blocked by ip allowlist", remote
                )
                return web.Response(status=403, text="ip not permitted")

        if not config.is_enabled():
            # Dev mode — no creds configured. Log loudly so ops
            # doesn't silently run this in production.
            log.warning(
                "keeper auth DISABLED — KEEPER_USERNAME/PASSWORD unset"
            )
            return await handler(request)

        parsed = parse_basic_header(request.headers.get("Authorization"))
        if parsed is None:
            return _unauthorized()
        user, pw = parsed

        # Use hmac.compare_digest to avoid timing-attack leakage of
        # the username — even though the attack surface here is
        # minimal, it's cheap to do right.
        u_ok = hmac.compare_digest(user, config.username)
        p_ok = hmac.compare_digest(pw, config.password)
        if not (u_ok and p_ok):
            log.info("keeper auth: bad credentials")
            return _unauthorized()

        return await handler(request)

    return _middleware
