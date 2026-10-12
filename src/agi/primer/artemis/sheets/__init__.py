# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS Sheets — Google-Sheets CSV → NATS diff stream.

The Keeper maintains character data in a Google Sheet published-to-web
as CSV. This package polls that CSV on a fixed interval, parses it
into character dicts, and publishes only the rows that changed on
``agi.rh.artemis.sheet.<sheet_name>``.

Downstream consumers:
  - Avatar agent — forwards to LiveKit DataChannel (``artemis.sheet``)
    so the table HUD updates in place.
  - Future: GM portal (S1h) reads full rows from the same NATS stream.

Subjects:
  agi.rh.artemis.sheet.<sheet_name>       — row diffs (one sheet per subject)
  agi.rh.artemis.sheet.snapshot.<name>    — full snapshot (on start / on demand)

See :mod:`.parser` for the CSV row → dict contract, and :mod:`.poller`
for the service loop.
"""

from __future__ import annotations

from .parser import (
    HUD_FIELDS,
    CharacterRow,
    parse_csv,
    row_for_hud,
)
from .poller import (
    SUBJECT_DIFF_PREFIX,
    SUBJECT_SNAPSHOT_PREFIX,
    SheetsPoller,
    build_poller_from_env,
)

__all__ = [
    "CharacterRow",
    "HUD_FIELDS",
    "parse_csv",
    "row_for_hud",
    "SUBJECT_DIFF_PREFIX",
    "SUBJECT_SNAPSHOT_PREFIX",
    "SheetsPoller",
    "build_poller_from_env",
]
