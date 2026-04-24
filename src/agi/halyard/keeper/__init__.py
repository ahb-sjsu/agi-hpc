# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-keeper — Keeper-console backend for the Halyard Table.

Small FastAPI service that handles Keeper-authenticated
operations:

- LiveKit JWT minting (delegates to
  :mod:`agi.primer.artemis.livekit_agent.token`).
- Session lifecycle (create, start, pause, end).
- Approval queue state for both AIs.
- Keeper-authorized sheet overrides and scene triggers.

The package skeleton lives here so Sprint 0 lands a committable
structure. Actual implementations land in Sprint 6. See
``docs/HALYARD_SPRINT_PLAN.md`` §Sprint-6.

Layout (populated in Sprint 6):

- :mod:`.app`         — FastAPI application
- :mod:`.auth`        — Keeper auth (HTTP Basic v1)
- :mod:`.approvals`   — pending-reply queue state
- :mod:`.sessions`    — session lifecycle
- :mod:`.livekit`     — JWT minting + room admin
"""

__all__: list[str] = []
