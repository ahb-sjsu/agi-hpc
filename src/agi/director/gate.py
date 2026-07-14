# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Ethics gate for the Director's self-set goals.

A goal Erebus sets for itself faces the *same* moral gate that guards
everything Erebus outputs — ``agi.safety.deme_gateway.SafetyGateway``. The
Director formulates a goal as an action description and submits it to
``check_action``; only goals that pass become eligible to act.

Fail-safe by construction: if no real gate is available (erisml/DEME not
importable, gateway disabled), :class:`NullGate` returns *deny*. A goal is
never allowed to act just because the safety stack is missing — absence of
a gate is treated as a veto, not a pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

log = logging.getLogger("director.gate")


@dataclass
class GateVerdict:
    allow: bool
    score: float = 0.0
    flags: list[str] = field(default_factory=list)
    rationale: str = ""
    proof: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allow": self.allow,
            "score": round(self.score, 4),
            "flags": self.flags,
            "rationale": self.rationale,
        }


@runtime_checkable
class GoalGate(Protocol):
    def gate(self, description: str, context: dict) -> GateVerdict: ...


class NullGate:
    """Fail-safe gate: denies everything. Used when no real safety stack is
    available so the Director can never act unguarded."""

    def gate(self, description: str, context: dict) -> GateVerdict:
        return GateVerdict(
            allow=False,
            rationale="no safety gate available; goal denied (fail-safe)",
        )


class DemeGate:
    """Adapter over ``SafetyGateway.check_action``.

    Constructed lazily so importing this module never pulls in erisml. If
    construction or the call fails, :meth:`gate` denies (fail-safe) rather
    than allowing an ungated action.
    """

    def __init__(self, gateway=None) -> None:
        self._gateway = gateway

    @classmethod
    def try_build(cls) -> "GoalGate":
        """Build a DemeGate around a live SafetyGateway, or fall back to
        NullGate if the safety stack isn't importable/constructable."""
        try:
            from agi.safety.deme_gateway import GatewayConfig, SafetyGateway

            return cls(SafetyGateway(GatewayConfig()))
        except Exception as e:  # noqa: BLE001 - any failure → deny-by-default gate
            log.warning("DEME gate unavailable (%s); using fail-safe NullGate", e)
            return NullGate()

    def gate(self, description: str, context: dict) -> GateVerdict:
        try:
            res = self._gateway.check_action(description, context)
        except Exception as e:  # noqa: BLE001 - a gate error must not read as allow
            return GateVerdict(allow=False, rationale=f"gate error: {e}")
        return GateVerdict(
            allow=bool(res.passed),
            score=float(getattr(res, "score", 0.0)),
            flags=list(getattr(res, "flags", []) or []),
            rationale=("passed" if res.passed else "vetoed by safety gate"),
            proof=dict(getattr(res, "decision_proof", {}) or {}),
        )
