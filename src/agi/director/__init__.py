# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Erebus's Director — the executive / self-model faculty.

The Director is the always-on loop that runs Erebus's Self-Directed
Cognition Cycle (SDCC): it perceives Erebus's own state, maintains a
persistent self-model, deliberates on its own sub-goals against a
standing charter, dispatches work through Erebus's existing faculties,
and journals what it did.

*Atlas* is the hardware; *Erebus* is the AGI. The Director is a faculty
of Erebus that happens to run as an ``atlas-``prefixed systemd unit
(host convention), like ``atlas-scientist`` and ``atlas-primer``.

See ``docs/SELF_DIRECTED_COGNITION.md`` for the full design, the
autonomy-tier governance model, and the phased rollout. This package is
Phase A: self-model + read-only reflection (autonomy pinned to L0 — the
Director physically cannot act on the system yet).
"""

from __future__ import annotations

from .policy import AutonomyTier, PolicyError, gate
from .self_model import SelfModel

__all__ = ["AutonomyTier", "PolicyError", "SelfModel", "gate"]
