# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-state — character-sheet service for the Halyard Table.

Owns live character-sheet state for a single session. Writes
accepted over NATS (``agi.rh.halyard.sheet.<pc_id>.patch``) and
REST (``POST /api/sheets/<session>/<pc_id>/patch``); updates
broadcast over NATS (``...update``) and WebSocket
(``WS /ws/sheets/<session>``).

The package skeleton lives here so Sprint 0 lands a committable
structure. Actual implementations land in Sprint 3. See
``docs/HALYARD_SPRINT_PLAN.md`` §Sprint-3.

Layout (populated in Sprint 3):

- :mod:`.schema.character_sheet_schema` — JSON Schema bindings
- :mod:`.schema.patch`   — JSON-Patch validator + authz
- :mod:`.store`          — on-disk JSON store, append-only log
- :mod:`.bridge`         — NATS bridge (patch → apply → update)
- :mod:`.api.app`        — FastAPI app (REST + WS)
"""

__all__ = [
    "SESSION_ARCHIVE_ROOT",
]

# Where session archives live on Atlas. Overridable via env var
# at service startup.
SESSION_ARCHIVE_ROOT = "/archive/halyard"
