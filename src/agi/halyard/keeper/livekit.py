# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""LiveKit integration for halyard-keeper-backend.

Thin wrapper around
:func:`agi.primer.artemis.livekit_agent.token.mint_participant_token`.
Lives in the keeper package so the HTTP layer can depend on
``agi.halyard.keeper.livekit`` without pulling in the ARTEMIS
runtime — the helper just re-exports the token minter with
keeper-specific grant defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._jwt import GrantOptions, mint_participant_token


@dataclass(frozen=True)
class LiveKitConfig:
    """Configuration the keeper backend needs to mint tokens."""

    url: str            # the WS URL the browser connects to
    api_key: str        # matches the SFU's keys file
    api_secret: str     # "
    # TTL for minted tokens. Six hours covers a long session with
    # slack for recovery; anything shorter forces mid-session
    # re-auth, anything longer is a security footgun.
    ttl_seconds: int = 6 * 3600

    @classmethod
    def from_env(cls) -> "LiveKitConfig":
        """Build config from env. Defaults target the dev-mode
        ``livekit-server --dev`` shipped by LiveKit (``devkey`` /
        ``secret``). Production deployments MUST set real values.
        """
        return cls(
            url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
            api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
            ttl_seconds=int(
                os.environ.get("LIVEKIT_TOKEN_TTL_SECONDS", str(6 * 3600))
            ),
        )


def is_gm_identity(identity: str) -> bool:
    """True if the identity is the GM/Keeper slot.

    Mirrors the policy the web client uses to route the GM into the
    centre cell of the table grid; kept in sync deliberately so
    server-side grants match client-side UI expectations.
    """
    if not identity:
        return False
    i = identity.lower()
    return (
        i == "gm"
        or i == "keeper"
        or i.startswith("gm-")
        or i.startswith("keeper-")
        or i.startswith("keeper:")
    )


def mint_player_token(
    *,
    config: LiveKitConfig,
    session_id: str,
    identity: str,
    name: str | None = None,
) -> str:
    """Mint a player-facing LiveKit JWT.

    Grants: publish A/V, publish data (DataChannel envelopes),
    subscribe. Screen-share is GM-only — non-GM identities get
    ``canPublishSources=["camera", "microphone"]``, which the
    LiveKit server enforces. The GM gets all sources.
    """
    if is_gm_identity(identity):
        sources: tuple[str, ...] | None = None  # all sources allowed
    else:
        sources = ("camera", "microphone")
    return mint_participant_token(
        identity=identity,
        room_name=session_id,
        api_key=config.api_key,
        api_secret=config.api_secret,
        grants=GrantOptions(
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
            can_publish_sources=sources,
        ),
        name=name or identity,
        ttl_seconds=config.ttl_seconds,
    )


def mint_keeper_token(
    *,
    config: LiveKitConfig,
    session_id: str,
    identity: str = "keeper",
    name: str | None = None,
) -> str:
    """Mint the Keeper's LiveKit JWT.

    Same grants as a player in v1 — room-admin grants are not
    exercised yet. The identity prefix (``keeper``) is what the
    agent's ``_format_speaker`` uses to tag turns as Keeper-
    originated; keep the prefix stable across deployments.
    """
    identity_full = identity if identity.startswith("keeper") else f"keeper:{identity}"
    return mint_participant_token(
        identity=identity_full,
        room_name=session_id,
        api_key=config.api_key,
        api_secret=config.api_secret,
        grants=GrantOptions(
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ),
        name=name or "Keeper",
        ttl_seconds=config.ttl_seconds,
    )


def mint_ai_token(
    *,
    config: LiveKitConfig,
    session_id: str,
    which: str,  # "artemis" | "sigma-4"
) -> str:
    """Mint a token for an AI-NPC agent process.

    AIs subscribe + publish DataChannel (text replies). Audio/video
    is deliberately disabled — the handheld-typing fiction works
    better and dodges the TTS uncanny-valley failure mode.
    """
    if which not in {"artemis", "sigma-4"}:
        raise ValueError(
            f"which must be 'artemis' or 'sigma-4' (got: {which!r})"
        )
    display = "ARTEMIS" if which == "artemis" else "SIGMA-4"
    return mint_participant_token(
        identity=which,
        room_name=session_id,
        api_key=config.api_key,
        api_secret=config.api_secret,
        grants=GrantOptions(
            can_publish=False,
            can_subscribe=True,
            can_publish_data=True,
        ),
        name=display,
        ttl_seconds=config.ttl_seconds,
    )
