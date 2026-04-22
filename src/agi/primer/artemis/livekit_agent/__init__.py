# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS LiveKit agent — Phase 3.

Bridges a LiveKit room to Atlas's NATS fabric:

- subscribes to every participant's audio track
- runs streaming Whisper ASR per track
- publishes finalized utterances on ``agi.rh.artemis.heard``
- subscribes to ``agi.rh.artemis.say`` filtered by session_id
- posts replies back into the room via LiveKit DataChannel

The agent owns **no reasoning** — the Phase 2 ``ArtemisService`` on
Atlas runs everything between hearing and replying, unchanged.

See ``docs/ARTEMIS.md`` §3 and §11 Phase 3 for the architecture.
"""

from __future__ import annotations

from .agent import AgentConfig, ArtemisLiveKitAgent
from .token import mint_participant_token

__all__ = [
    "AgentConfig",
    "ArtemisLiveKitAgent",
    "mint_participant_token",
]
