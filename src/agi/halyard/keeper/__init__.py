# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-keeper — Keeper-console backend for the Halyard Table.

aiohttp service that handles the Keeper-facing concerns:

- LiveKit JWT minting (delegates to
  :mod:`agi.primer.artemis.livekit_agent.token`).
- Session lifecycle (create / pause / resume / close).
- Approval queue state for both AIs.
- WebSocket live feed of pending approvals for the Keeper console.

Layout (Sprint 6):

- :mod:`.app`         — aiohttp ``build_app`` factory
- :mod:`.auth`        — HTTP Basic middleware + IP allow-list
- :mod:`.approvals`   — pending-reply queue with fan-out
- :mod:`.sessions`    — session lifecycle registry
- :mod:`.livekit`     — token-mint wrapper

Pending for Sprint 7: NATS subscription wiring so the queue
receives real ``*.say`` traffic from the AIs, stat-override
endpoint, scene trigger publish, dice broadcast.
"""

from __future__ import annotations

from .app import build_app
from .approvals import AiKind, ApprovalQueue, ApprovalState
from .auth import KeeperAuthConfig
from .livekit import LiveKitConfig, mint_keeper_token, mint_player_token
from .sessions import (
    InvalidTransition,
    SessionAlreadyExists,
    SessionNotFound,
    SessionRegistry,
    SessionState,
)

__all__ = [
    "build_app",
    "AiKind",
    "ApprovalQueue",
    "ApprovalState",
    "KeeperAuthConfig",
    "LiveKitConfig",
    "mint_keeper_token",
    "mint_player_token",
    "InvalidTransition",
    "SessionAlreadyExists",
    "SessionNotFound",
    "SessionRegistry",
    "SessionState",
]
