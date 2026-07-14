# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Erebus's Discord presence — a faculty for honest, gated conversation.

Erebus hosts a channel (e.g. #erebus-agi on the ErisML/DEME server) so
people can talk to it and it can learn. Design principles, straight from
Erebus's charter:

- **Reactive only (v1).** Responds when addressed; never initiates.
- **Reuses existing cognition.** Each message routes through the running
  ``/api/erebus/chat`` pipeline (Ego / vMOE / RAG) — a new mouth, not a new
  brain.
- **Every outgoing reply is DEME-gated** (``SafetyGateway.check_output``)
  before it is posted. Fail-safe: if the gate is unavailable, Erebus stays
  silent rather than posting ungated.
- **Honest.** Discloses that it is an AI; channel-scoped; rate-limited;
  fully audited; a sentinel kill-switch stops it responding.

*Atlas is the hardware; Erebus is the mind.* This is Erebus's faculty; it
runs as ``atlas-erebus-discord.service`` by host convention.

The message-handling core (``handler.py``) has no dependency on the
``discord`` library, so it is unit-testable with fakes; ``bot.py`` is the
thin ``discord.py`` wiring layer.
"""

from __future__ import annotations

from .config import DiscordConfig
from .handler import Outcome, handle

__all__ = ["DiscordConfig", "Outcome", "handle"]
