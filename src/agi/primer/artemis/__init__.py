# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS — live AI non-player character for tabletop RPG play.

Event-driven sibling of The Primer. Shares vMOE + event-log substrate,
differs in validator contract (chat safety, not code verification) and
I/O (NATS-driven, not poll-driven).

See [`docs/ARTEMIS.md`](../../../docs/ARTEMIS.md) for the plan-of-record.

This package is Phase 1: offline, text-only, unit-testable. NATS wiring
lives in Phase 2, bot container in Phase 3.
"""

from __future__ import annotations

from .mode import TurnRequest, TurnResponse, handle_turn
from .validator import DecisionProof, check_reply

__all__ = [
    "TurnRequest",
    "TurnResponse",
    "DecisionProof",
    "handle_turn",
    "check_reply",
]
