# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""LiveKit JWT minting — self-contained.

Deliberately scoped to this package so halyard-keeper-backend has
no cross-package import dependency on files that may live on
other sprint branches. The ARTEMIS LiveKit agent
(``agi.primer.artemis.livekit_agent.token``) defines a parallel
implementation; when both packages merge to main a later refactor
can collapse them into a shared base.

LiveKit tokens are plain HS256 JWTs signed with the API secret.
Reference: https://docs.livekit.io/guides/access-tokens/
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import jwt as _jwt  # PyJWT
except ImportError:  # pragma: no cover
    _jwt = None  # type: ignore


_DEFAULT_TTL_S = 6 * 60 * 60  # 6 hours


@dataclass(frozen=True)
class GrantOptions:
    """Per-participant LiveKit grant flags.

    ``can_publish_sources`` restricts publication to a specific
    subset of sources. When None, any source is allowed (subject to
    ``can_publish``). When set to a list (e.g.
    ``["camera", "microphone"]``), the LiveKit server rejects
    publishes from any source not in the list — used here to gate
    screen-share to the GM identity at the server level.
    """

    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_publish_sources: tuple[str, ...] | None = None
    room_admin: bool = False
    room_record: bool = False


def mint_participant_token(
    *,
    identity: str,
    room_name: str,
    api_key: str,
    api_secret: str,
    ttl_seconds: int = _DEFAULT_TTL_S,
    grants: GrantOptions | None = None,
    name: str | None = None,
) -> str:
    """Mint a LiveKit join JWT.

    See :func:`agi.primer.artemis.livekit_agent.token.mint_participant_token`
    for the semantic contract; the output of both functions is
    interchangeable at the LiveKit server (same claims, same
    HS256 signature).
    """
    if _jwt is None:
        raise RuntimeError(
            "PyJWT is not installed; install with `pip install PyJWT`"
        )
    g = grants or GrantOptions()
    now = int(time.time())
    video_grant: dict = {
        "room": room_name,
        "roomJoin": True,
        "canPublish": g.can_publish,
        "canSubscribe": g.can_subscribe,
        "canPublishData": g.can_publish_data,
    }
    if g.can_publish_sources is not None:
        video_grant["canPublishSources"] = list(g.can_publish_sources)
    if g.room_admin:
        video_grant["roomAdmin"] = True
    if g.room_record:
        video_grant["roomRecord"] = True
    claims: dict = {
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "video": video_grant,
    }
    if name:
        claims["name"] = name
    return _jwt.encode(claims, api_secret, algorithm="HS256")


def decode_participant_token(token: str, api_secret: str) -> dict:
    """Verify + decode a LiveKit JWT (testing / admin tooling)."""
    if _jwt is None:
        raise RuntimeError(
            "PyJWT is not installed; install with `pip install PyJWT`"
        )
    return _jwt.decode(token, api_secret, algorithms=["HS256"])
