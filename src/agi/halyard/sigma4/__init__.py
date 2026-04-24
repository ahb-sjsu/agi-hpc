# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 — ship-mind AI NPC for the Halyard Table.

SIGMA-4 is a deliberate architectural twin of
:mod:`agi.primer.artemis`: separate persona, separate bible
split, separate NATS lane, shared substrate (vMOE, validator
framework, DecisionProof chain).

The package skeleton lives here so Sprint 0 lands a committable
structure. Actual implementations land in Sprint 2. See
``docs/HALYARD_SPRINT_PLAN.md`` §Sprint-2.

Layout (populated in Sprint 2):

- :mod:`.mode`           — ``handle_turn`` entry point
- :mod:`.prompt`         — system prompt + fallback templates
- :mod:`.context`        — bible split + session log
- :mod:`.trigger`        — should-speak policy
- :mod:`.validator`      — SIGMA-specific check-reply rules
- :mod:`.nats_handler`   — NATS subscribe / publish loop
- :mod:`.livekit_agent`  — LiveKit agent, twin of ARTEMIS's
"""

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
]
