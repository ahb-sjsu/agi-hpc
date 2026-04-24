# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Halyard Table — the live-play runtime for the Beyond the Heliopause campaign.

See ``docs/HALYARD_TABLE.md`` for the architecture of record and
``docs/HALYARD_SPRINT_PLAN.md`` for the sprint-by-sprint execution plan.

Subpackages:

- :mod:`agi.halyard.sigma4` — the SIGMA-4 ship-mind AI NPC.
- :mod:`agi.halyard.state` — the character-sheet state service.
- :mod:`agi.halyard.keeper` — the Keeper-console backend.

ARTEMIS (the other AI NPC) lives at :mod:`agi.primer.artemis` for
historical reasons; it predates this package and its existing
NATS / validator / deployment wiring is stable. SIGMA-4 is a
deliberate twin of the ARTEMIS architecture and reuses its
substrate (vMOE, validator framework, DecisionProof chain).
"""

__all__ = [
    "NAMESPACE",
    "SUBJECT_SESSION_START",
    "SUBJECT_SESSION_END",
    "SUBJECT_SESSION_SILENCE",
    "SUBJECT_SESSION_RESUME",
    "SUBJECT_SESSION_TICK",
    "SUBJECT_SHEET_PATCH_FMT",
    "SUBJECT_SHEET_UPDATE_FMT",
    "SUBJECT_SCENE_TRIGGER",
    "SUBJECT_SCENE_CUE",
    "SUBJECT_KEEPER_APPROVE",
    "SUBJECT_KEEPER_REJECT",
    "SUBJECT_KEEPER_OVERRIDE",
    "SUBJECT_KEEPER_DICE",
]

# ─────────────────────────────────────────────────────────────────
# NATS subject constants — single source of truth.
#
# These live at the package root so every subpackage, the web
# bridges, and the Keeper backend all import the same strings.
# Anyone changing a subject must change it here and grep for
# stragglers.
# ─────────────────────────────────────────────────────────────────

NAMESPACE = "agi.rh.halyard"

SUBJECT_SESSION_START = f"{NAMESPACE}.session.start"
SUBJECT_SESSION_END = f"{NAMESPACE}.session.end"
SUBJECT_SESSION_SILENCE = f"{NAMESPACE}.session.silence"
SUBJECT_SESSION_RESUME = f"{NAMESPACE}.session.resume"
SUBJECT_SESSION_TICK = f"{NAMESPACE}.session.tick"

# Per-PC sheet subjects are formatted with the pc_id; keep the
# format string so callers don't fabricate their own.
SUBJECT_SHEET_PATCH_FMT = f"{NAMESPACE}.sheet.{{pc_id}}.patch"
SUBJECT_SHEET_UPDATE_FMT = f"{NAMESPACE}.sheet.{{pc_id}}.update"

SUBJECT_SCENE_TRIGGER = f"{NAMESPACE}.scene.trigger"
SUBJECT_SCENE_CUE = f"{NAMESPACE}.scene.cue"

SUBJECT_KEEPER_APPROVE = f"{NAMESPACE}.keeper.approve"
SUBJECT_KEEPER_REJECT = f"{NAMESPACE}.keeper.reject"
SUBJECT_KEEPER_OVERRIDE = f"{NAMESPACE}.keeper.override"
SUBJECT_KEEPER_DICE = f"{NAMESPACE}.keeper.dice"
