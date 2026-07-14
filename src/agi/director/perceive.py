# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Step 1 of the SDCC — perceive Erebus's own state.

Assembles a self-state snapshot from the state files Erebus's faculties
already maintain, plus derived capability / limitation lists. Every read
is defensive: a missing or malformed file degrades to a sensible default
rather than crashing the cycle, so this runs identically on Atlas and in
a bare dev checkout.

No network calls and no heavy deps — this is the L0 (read-only) floor of
the cycle and must always be safe to run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("director.perceive")


@dataclass
class Paths:
    """Where Erebus's faculties keep their state (Atlas defaults)."""

    memory: Path = Path("/archive/neurogolf/arc_scientist_memory.json")
    help_queue: Path = Path("/archive/neurogolf/erebus_help_queue.json")
    primer_health: Path = Path("/archive/neurogolf/primer_health.json")
    primer_events: Path = Path("/archive/neurogolf/primer_events.jsonl")
    disabled_sentinel: Path = Path("/archive/neurogolf/.director_disabled")
    gpu1_maint_sentinel: Path = Path("/archive/neurogolf/.gpu1_maint")


@dataclass
class SelfState:
    """A single-cycle snapshot of Erebus's condition."""

    tasks_total: int = 0
    tasks_solved: int = 0
    tasks_stuck: int = 0  # >= min_attempts, unsolved
    help_queue_len: int = 0
    primer_experts_healthy: list[str] = field(default_factory=list)
    primer_experts_degraded: list[str] = field(default_factory=list)
    faculties_online: list[str] = field(default_factory=list)
    gpu1_on_loan: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None


def gather(paths: Paths | None = None, *, min_attempts: int = 10) -> SelfState:
    """Read Erebus's state files and build a :class:`SelfState` snapshot."""
    p = paths or Paths()
    st = SelfState()

    mem = _read_json(p.memory) or {}
    tasks = mem.get("tasks") or {}
    st.tasks_total = len(tasks)
    for tk in tasks.values():
        if not isinstance(tk, dict):
            continue
        if tk.get("solved"):
            st.tasks_solved += 1
        elif len(tk.get("attempts", []) or []) >= min_attempts:
            st.tasks_stuck += 1

    hq = _read_json(p.help_queue)
    if isinstance(hq, list):
        st.help_queue_len = len(hq)
    elif isinstance(hq, dict):
        st.help_queue_len = len(hq.get("items", hq.get("queue", [])) or [])

    health = _read_json(p.primer_health) or {}
    # primer_health.json is the vMOE HealthTracker summary: {expert: {...}}.
    for name, info in (health.items() if isinstance(health, dict) else []):
        if isinstance(info, dict) and info.get("degraded"):
            st.primer_experts_degraded.append(name)
        else:
            st.primer_experts_healthy.append(name)

    st.gpu1_on_loan = p.gpu1_maint_sentinel.exists()

    # Faculties considered online if their state file is present + parseable.
    # This is a coarse liveness proxy, not a service-status probe (that's the
    # control plane's job and would require privilege the L0 floor lacks).
    if mem:
        st.faculties_online.append("scientist")
    if health:
        st.faculties_online.append("primer")
    if p.primer_events.exists():
        st.faculties_online.append("primer-events")

    return st


def derive_capabilities(state: SelfState) -> list[str]:
    """Turn the raw snapshot into first-person capability statements."""
    caps: list[str] = []
    if "scientist" in state.faculties_online:
        caps.append(
            f"Solve ARC-AGI tasks in a closed observe→hypothesize→verify loop "
            f"({state.tasks_solved}/{state.tasks_total} solved)."
        )
    if "primer" in state.faculties_online:
        healthy = ", ".join(state.primer_experts_healthy) or "none currently healthy"
        caps.append(f"Teach myself via a frontier vMOE ensemble ({healthy}).")
    caps.append("Reason via Ego/Superego/Divine-Council over local + NRP models.")
    caps.append("Perceive the moral valence of text (xbse Id lane).")
    caps.append("Consolidate episodic memory during idle dreaming windows.")
    return caps


def derive_limitations(state: SelfState) -> list[str]:
    """Honest limitations — kept explicit so the self-model doesn't overclaim."""
    lims: list[str] = [
        "I am not conscious; this self-model is an engineered representation, "
        "not evidence of inner experience.",
        "I act only within my autonomy ceiling; higher-tier actions need human review.",
        "I never restart, kill, or reboot Atlas — the host lifecycle is not mine.",
    ]
    if state.tasks_stuck:
        lims.append(f"{state.tasks_stuck} tasks remain stuck despite many attempts.")
    if state.primer_experts_degraded:
        lims.append(
            f"Degraded reasoning experts right now: "
            f"{', '.join(state.primer_experts_degraded)}."
        )
    if state.gpu1_on_loan:
        lims.append("GPU 1 is on maintenance loan; my Ego backend may be degraded.")
    return lims
