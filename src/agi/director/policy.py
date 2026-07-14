# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Autonomy-tier governance for Erebus's Director.

The Director sets its own sub-goals, so the guardrails must exist in code
*before* it can act — not as policy prose bolted on later. Every action
the Director would take is classified into an autonomy tier and checked
against the running ceiling (``DIRECTOR_MAX_TIER``) before it executes.

Tiers (see ``docs/SELF_DIRECTED_COGNITION.md``):

  L0  self-reflection, read-only .................... always allowed
  L1  write memory / wiki / self-model .............. verify-before-publish
  L2  dispatch compute within quota ................. hard resource caps
  L3  self-modification / external action ........... human review required

Hard invariant: the Director's action vocabulary **excludes every service
restart, kill, and reboot.** The Atlas no-reboot rule lives here, in the
forbidden-verb list, so the loop cannot even express those actions
regardless of tier. This is defense-in-depth on top of the systemd
allowlist in ``scripts/atlas_control.py``.

Phase A pins ``DIRECTOR_MAX_TIER=L0``: the Director can perceive and
journal, and nothing else.
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum

log = logging.getLogger("director.policy")


class AutonomyTier(IntEnum):
    """Ordered autonomy levels. Higher = more capable, more gated."""

    L0 = 0  # read-only self-reflection
    L1 = 1  # write memory / wiki / self-model
    L2 = 2  # dispatch compute within quota
    L3 = 3  # self-modification / external action (human review)

    @classmethod
    def parse(cls, value: str | int | None, default: "AutonomyTier") -> "AutonomyTier":
        if value is None:
            return default
        if isinstance(value, int):
            return cls(value)
        s = str(value).strip().upper()
        if s.startswith("L") and s[1:].isdigit():
            return cls(int(s[1:]))
        if s.isdigit():
            return cls(int(s))
        raise ValueError(f"unparseable autonomy tier: {value!r}")


class PolicyError(RuntimeError):
    """Raised when the Director attempts an action above its ceiling or on
    the forbidden-verb list."""


# Verbs the Director may NEVER express, at any tier. These target the host
# lifecycle (Atlas), which is out of scope for an Erebus cognitive faculty.
# Matching is substring-insensitive against the action's declared verb.
_FORBIDDEN_VERBS = (
    "reboot",
    "shutdown",
    "poweroff",
    "halt",
    "restart",  # no service restarts — that's the control plane's job, human-driven
    "systemctl",
    "kill",
    "pkill",
    "disable",  # no disabling units
    "rm -rf",
)


def max_tier() -> AutonomyTier:
    """The running autonomy ceiling, from ``DIRECTOR_MAX_TIER`` (default L0)."""
    return AutonomyTier.parse(os.environ.get("DIRECTOR_MAX_TIER"), AutonomyTier.L0)


def is_forbidden(verb: str) -> bool:
    """True if ``verb`` names a host-lifecycle action the Director may never take."""
    v = (verb or "").lower()
    return any(bad in v for bad in _FORBIDDEN_VERBS)


def gate(
    action_tier: AutonomyTier, *, verb: str = "", ceiling: AutonomyTier | None = None
) -> None:
    """Authorize an action or raise ``PolicyError``.

    Two independent checks:

    1. ``verb`` must not be on the forbidden host-lifecycle list (checked
       at every tier — a forbidden verb is never allowed).
    2. ``action_tier`` must be ``<=`` the running ceiling.

    Raises ``PolicyError`` on denial; returns ``None`` on approval.
    """
    if is_forbidden(verb):
        raise PolicyError(
            f"action verb {verb!r} is a forbidden host-lifecycle action "
            f"(the Director never restarts/kills/reboots Atlas)"
        )
    ceil = ceiling if ceiling is not None else max_tier()
    if action_tier > ceil:
        raise PolicyError(
            f"action requires autonomy {action_tier.name} but ceiling is "
            f"{ceil.name} (raise DIRECTOR_MAX_TIER to permit, with governance)"
        )
