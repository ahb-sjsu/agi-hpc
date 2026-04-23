# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Keeper-only identity gate for the GM portal.

Every endpoint (except ``/api/healthz``) requires a bearer JWT signed
with the LiveKit API secret (HS256). The identity claim
(``sub`` or ``identity``) must start with ``keeper:``. Player tokens
are rejected — the portal is Keeper-only by design.

The JWT may come from:
  - Authorization: Bearer <jwt>
  - Cookie: portal_token=<jwt>
  - ?t=<jwt> URL query (useful for one-click Keeper links)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("artemis.portal.auth")


@dataclass
class KeeperPrincipal:
    identity: str  # e.g. "keeper:andrew"
    room: str | None = None


class AuthError(Exception):
    """Raised when a request fails the keeper auth gate."""


def keeper_secret() -> str:
    """Shared secret for JWT verification. Reuses the LiveKit secret."""
    sec = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not sec:
        raise RuntimeError(
            "LIVEKIT_API_SECRET not set — portal cannot verify keeper tokens"
        )
    return sec


def extract_token(
    headers: dict[str, str],
    cookies: dict[str, str],
    query: dict[str, str],
) -> str:
    """Find the JWT in a request. Returns empty string if not present."""
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if "portal_token" in cookies:
        return cookies["portal_token"].strip()
    if "t" in query and query["t"]:
        return query["t"].strip()
    return ""


def verify_keeper_token(token: str, *, secret: str | None = None) -> KeeperPrincipal:
    """Decode the JWT, validate signature + identity prefix.

    Raises :class:`AuthError` on any failure.
    """
    if not token:
        raise AuthError("missing token")
    try:
        import jwt  # type: ignore
    except ImportError as e:  # pragma: no cover — dep missing on Atlas
        raise AuthError("PyJWT not installed on server") from e
    sec = secret or keeper_secret()
    try:
        claims: dict[str, Any] = jwt.decode(token, sec, algorithms=["HS256"])
    except Exception as e:  # noqa: BLE001 — jwt raises many subtypes
        raise AuthError(f"invalid token: {e}") from e
    # LiveKit JWTs use `sub` for identity; also accept an explicit
    # `identity` field for dev-minted tokens.
    identity = str(claims.get("identity") or claims.get("sub") or "").strip()
    if not identity:
        raise AuthError("token has no identity claim")
    if not identity.startswith("keeper:"):
        raise AuthError(f"identity '{identity}' is not keeper:*")
    room = None
    video = claims.get("video")
    if isinstance(video, dict) and video.get("room"):
        room = str(video["room"])
    return KeeperPrincipal(identity=identity, room=room)
