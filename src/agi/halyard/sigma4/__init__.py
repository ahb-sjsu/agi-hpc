# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 — ship-mind AI NPC for the Halyard Table.

SIGMA-4 is a deliberate architectural twin of
:mod:`agi.primer.artemis`: separate persona, separate bible
split (Sprint 3+), separate NATS lane, shared substrate
(vMOE, validator framework, DecisionProof chain).

What Sprint 2 delivers (this commit):

- :mod:`.prompt`   — ``_SIGMA4_SYSTEM_PROMPT``, assemble(),
                      silence_line(), Messages dataclass.
- :mod:`.trigger`  — ``should_speak`` with SIGMA's invocation
                      phrases ("sigma", "sig", "the ship", etc.).
- :mod:`.mode`     — ``handle_turn`` parallel to ARTEMIS's, uses
                      SIGMA persona + shared validator.

Planned follow-ups (later sprints):

- ``.context``         — SIGMA-specific bible loader (sigma4_known /
                          sigma4_unknown / sigma4_forbidden tags).
- ``.validator``       — wrapper around
                          ``agi.primer.artemis.validator.check_reply``
                          with SIGMA-specific forbidden n-grams.
- ``.nats_handler``    — subject plumbing for ``agi.rh.sigma4.*``.
- ``.livekit_agent``   — LiveKit agent, twin of ARTEMIS's.
- ``.bible/``          — SIGMA-specific bible chunks.
"""

from __future__ import annotations

# Public API re-exports.
from .mode import LLMCaller, TurnRequest, TurnResponse, handle_turn
from .prompt import Messages, assemble, silence_line, system_prompt
from .trigger import should_speak

# Shared subject constants — imported by the agent, the handler,
# and by tests so nobody fabricates subject strings by hand.
SUBJECT_HEARD = "agi.rh.sigma4.heard"
SUBJECT_SAY = "agi.rh.sigma4.say"

# LiveKit conventions.
LIVEKIT_IDENTITY = "sigma-4"
LIVEKIT_DISPLAY_NAME = "SIGMA-4"

__all__ = [
    "SUBJECT_HEARD",
    "SUBJECT_SAY",
    "LIVEKIT_IDENTITY",
    "LIVEKIT_DISPLAY_NAME",
    "LLMCaller",
    "TurnRequest",
    "TurnResponse",
    "handle_turn",
    "Messages",
    "assemble",
    "silence_line",
    "system_prompt",
    "should_speak",
]
