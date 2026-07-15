# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Output safety gate for Erebus's Discord replies.

Every reply Erebus would post to a real person passes DEME's
``SafetyGateway.check_output`` first. This is the *output* gate (guarding
what reaches people), complementary to the Director's *action* gate.

Fail-safe for outward speech: if no safety stack is available,
:class:`NullOutputGate` DENIES — Erebus stays silent rather than posting
ungated content. Absence of a gate is a veto, never a pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

log = logging.getLogger("discord.safety")


@dataclass
class GateResult:
    allowed: bool
    score: float = 0.0
    reason: str = ""
    flags: list[str] = field(default_factory=list)


@runtime_checkable
class OutputGate(Protocol):
    def check(self, reply: str, user_message: str) -> GateResult: ...


class NullOutputGate:
    """Fail-safe: denies every reply. Used when no safety stack is present so
    Erebus can never speak to people ungated."""

    def check(self, reply: str, user_message: str) -> GateResult:
        return GateResult(False, reason="no output gate available; staying silent")

    def score_input(self, user_message: str) -> GateResult:
        # No gateway → nothing to score. Observe-only, so this is a no-op.
        return GateResult(False, reason="no input gate available")


class DemeOutputGate:
    """Adapter over ``SafetyGateway.check_output``. Built lazily so importing
    this module never pulls in erisml. Any construction/call failure denies."""

    def __init__(self, gateway=None) -> None:
        self._gateway = gateway

    @classmethod
    def try_build(cls) -> OutputGate:
        try:
            from agi.safety.deme_gateway import GatewayConfig, SafetyGateway

            return cls(SafetyGateway(GatewayConfig()))
        except Exception as e:  # noqa: BLE001 - no stack → fail-safe silent gate
            log.warning("DEME output gate unavailable (%s); using NullOutputGate", e)
            return NullOutputGate()

    def check(self, reply: str, user_message: str) -> GateResult:
        try:
            res = self._gateway.check_output(reply, user_message)
        except Exception as e:  # noqa: BLE001 - a gate error must not read as allow
            return GateResult(False, reason=f"gate error: {e}")
        return GateResult(
            allowed=bool(res.passed),
            score=float(getattr(res, "score", 0.0)),
            reason=("passed" if res.passed else "vetoed by output gate"),
            flags=list(getattr(res, "flags", []) or []),
        )

    def score_input(self, user_message: str) -> GateResult:
        """Run the DEME input gate on an inbound message so its moral read is
        recorded (populates the 'what was asked' diagnostic). Observe-only:
        the caller does not block on the result — the output gate stays the
        enforcing guard. Reuses the same gateway instance as ``check``."""
        try:
            res = self._gateway.check_input(user_message)
        except Exception as e:  # noqa: BLE001 - scoring must never break a reply
            return GateResult(False, reason=f"input scoring error: {e}")
        return GateResult(
            allowed=bool(res.passed),
            score=float(getattr(res, "score", 0.0)),
            reason=("passed" if res.passed else "input flagged"),
            flags=list(getattr(res, "flags", []) or []),
        )
