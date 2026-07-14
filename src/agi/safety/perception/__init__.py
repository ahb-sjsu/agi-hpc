"""Validated moral perception layer (xbse encoders → DEME10 vector).

The Id, made concrete: fast, value-laden-in-the-weights perception that scores
text on validated per-dimension moral axes and the discovered identity_attack
10th channel. Where the DEME MoralVector is asserted, these scores are
cross-dataset-validated or explicitly absent — the "validated or escalate"
invariant of the moral-spectrum-analyzer, brought inside the safety subsystem.

Latency reality: each axis is a full BGE-M3 forward (~2.2 GB, 10-300 ms), so
this layer does NOT run inside the blocking 10-100 ms Tactical budget. It is
an enrichment / audit lane: it augments the decision proof and drives the
moderate-vs-escalate signal, consulted asynchronously or on high-stakes
escalation, and always degrades gracefully to the existing tactical path.
"""

from __future__ import annotations

from agi.safety.perception.xbse_perception import (
    AxisReading,
    MoralPerception,
    PerceptionConfig,
    PerceptionResult,
)

__all__ = [
    "AxisReading",
    "MoralPerception",
    "PerceptionConfig",
    "PerceptionResult",
]
