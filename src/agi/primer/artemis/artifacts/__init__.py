# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS session artifacts — player handouts and downloadables.

Static handouts (pre-session briefings, character sheets, ship-deck
diagrams) live as Markdown under :mod:`.handouts.source` and are
rendered to PDF via :func:`.generator.render_handout`. Dynamic
artifacts (per-session summaries) are produced at runtime.

The ARTIFACTS card in the player table UI calls the artifact
service (:mod:`.service`) to list and deliver these files; during
development they can also be rendered and staged manually via
``scripts/generate_handouts.py``.
"""

from __future__ import annotations

from .generator import (
    HANDOUT_SOURCE_DIR,
    HandoutError,
    HandoutMeta,
    discover_handouts,
    render_handout,
)

__all__ = [
    "HANDOUT_SOURCE_DIR",
    "HandoutError",
    "HandoutMeta",
    "discover_handouts",
    "render_handout",
]
