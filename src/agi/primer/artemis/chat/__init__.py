# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS in-session chat — routing, persistence, Keeper-searchable log.

Message kinds (single source of truth):
  player_to_artemis   — player asks ARTEMIS; routed through Primer
  artemis_to_player   — ARTEMIS reply to one player (private)
  keeper_to_player    — Keeper whispers to one player
  keeper_to_all       — Keeper broadcast (in-fiction comms relay)
  artemis_to_all      — ARTEMIS broadcast (used by silent-GM narration)

Transport subjects:
  agi.rh.artemis.chat.in.<player_id>        — player-originated inbound
  agi.rh.artemis.chat.out.<player_id>       — private outbound to one
  agi.rh.artemis.chat.broadcast             — broadcast to all

Persistence:
  SQLite with FTS5 virtual-table over the body column. Keeper portal
  (S1h) queries this with arbitrary free-text search.
"""

from __future__ import annotations

from .service import (
    SUBJECT_BROADCAST,
    SUBJECT_IN_PREFIX,
    SUBJECT_OUT_PREFIX,
    ChatMessage,
    ChatService,
    MessageKind,
    build_service_from_env,
)
from .store import ChatStore, StoredMessage

__all__ = [
    "ChatMessage",
    "ChatService",
    "ChatStore",
    "MessageKind",
    "StoredMessage",
    "SUBJECT_BROADCAST",
    "SUBJECT_IN_PREFIX",
    "SUBJECT_OUT_PREFIX",
    "build_service_from_env",
]
