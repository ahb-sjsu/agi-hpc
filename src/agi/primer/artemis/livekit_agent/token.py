# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""JWT minting for LiveKit room participants.

The Keeper's portal mints one token per player per session using
:func:`mint_participant_token`. The browser client presents the token
to the LiveKit SFU, which then joins the participant to the named room
with the grants specified at mint time.

LiveKit uses plain HS256 JWTs signed with the API secret. No OAuth,
no external IDP — the API key/secret pair is all the SFU needs.

References:
  - https://docs.livekit.io/guides/access-tokens/
  - https://docs.livekit.io/home/get-started/api-keys/
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import jwt  # PyJWT
except ImportError:  # pragma: no cover
    jwt = None  # type: ignore


# Default TTL — one game session is 3–4 hours; give the token a bit of
# headroom in case the Keeper starts early or runs long. Override per
# session as needed.
_DEFAULT_TTL_S = 6 * 60 * 60  # 6 hours


@dataclass(frozen=True)
class GrantOptions:
    """Per-participant capability grants.

    Defaults reflect a player at an ARTEMIS table: can speak, can hear
    other players, cannot record. The Keeper's own token usually grants
    ``room_admin`` for muting or kicking disruptive participants.
    """

    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
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

    Args:
        identity: stable per-user identifier (e.g. ``"player:imogen"``).
            The SFU will reject a second participant joining the same
            room with the same identity.
        room_name: LiveKit room name. In ARTEMIS this is the
            ``session_id`` so the agent can filter NATS traffic.
        api_key: LiveKit API key (shared with the SFU and the agent).
        api_secret: LiveKit API secret (HS256 signing key).
        ttl_seconds: token lifetime in seconds.
        grants: per-participant capability grants.
        name: optional human-readable name visible in the room.

    Returns:
        The encoded JWT as a string. Safe to hand to a browser client.
    """
    if jwt is None:
        raise RuntimeError("PyJWT is not installed; install with `pip install PyJWT`")
    g = grants or GrantOptions()
    now = int(time.time())
    video_grant: dict = {
        "room": room_name,
        "roomJoin": True,
        "canPublish": g.can_publish,
        "canSubscribe": g.can_subscribe,
        "canPublishData": g.can_publish_data,
    }
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
    return jwt.encode(claims, api_secret, algorithm="HS256")


def decode_participant_token(token: str, api_secret: str) -> dict:
    """Decode and verify a LiveKit token (for testing / admin tooling).

    Verifies HS256 signature + expiry. Raises ``jwt.PyJWTError`` on
    failure.
    """
    if jwt is None:
        raise RuntimeError("PyJWT is not installed; install with `pip install PyJWT`")
    return jwt.decode(token, api_secret, algorithms=["HS256"])
