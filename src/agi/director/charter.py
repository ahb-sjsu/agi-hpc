# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Erebus's charter — the human-authored standing plan.

The charter is Erebus's constitution: identity, principles, and the
weighted standing objectives the Director pursues. It is **human-owned**;
the Director reads it but may only *propose* amendments (L3, human-
approved). ``charter.json`` is the machine-readable projection the Director
reads each cycle (generated from ``charter.md`` by ``scripts/charter_lint.py``
so prose and machine view can't drift).

Each objective names a *metric* the Director tracks against Erebus's
self-state. The metric registry below maps metric names to extractors over
the perceive snapshot. Metrics with no registered source return ``None``
(unmeasured) — the Director then reports the objective as un-rankable
rather than fabricating progress. Honesty over coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("/archive/erebus/charter.json")


# ── metric registry ──────────────────────────────────────────────
#
# name -> function(self_state_dict) -> float | None. Only metrics with a
# real source here can be gap-ranked; everything else is honestly
# "unmeasured" until a source is wired (see obj without a source in the
# starter charter, e.g. verified_notes / safety_dossier).

def _m_arc_solved(s: dict) -> float | None:
    return _num(s.get("tasks_solved"))


def _m_arc_stuck(s: dict) -> float | None:
    return _num(s.get("tasks_stuck"))


def _m_help_queue(s: dict) -> float | None:
    return _num(s.get("help_queue_len"))


METRIC_SOURCES = {
    "arc_solved": _m_arc_solved,
    "arc_stuck": _m_arc_stuck,
    "help_queue": _m_help_queue,
}


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def measure(metric: str, self_state: dict) -> float | None:
    """Current value of ``metric`` from a self-state snapshot, or None if
    the metric has no registered source (honestly unmeasured)."""
    fn = METRIC_SOURCES.get(metric)
    return fn(self_state) if fn else None


# ── data model ───────────────────────────────────────────────────


@dataclass
class Objective:
    id: str
    title: str
    weight: float = 0.0
    metric: str = ""
    target: float | None = None
    direction: str = "up"  # "up" = higher is better; "down" = lower is better

    def gap(self, current: float | None) -> float | None:
        """Normalized distance to target in [0, 1]; 0 = satisfied. None if
        the metric is unmeasured or no target is set."""
        if current is None or self.target is None:
            return None
        if self.direction == "down":
            denom = max(abs(current), 1.0)
            return max(0.0, (current - self.target)) / denom
        denom = max(abs(self.target), 1.0)
        return max(0.0, (self.target - current)) / denom

    def priority(self, current: float | None) -> float | None:
        g = self.gap(current)
        return None if g is None else self.weight * g


@dataclass
class Charter:
    version: int = 1
    identity: str = ""
    principles: list[str] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    limits: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Charter | None":
        """Load charter.json, or None if absent/malformed. A missing charter
        means the Director has nothing to pursue — it stays purely reflective
        (Phase A behavior) rather than inventing goals."""
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        objs = [
            Objective(
                id=o.get("id", ""),
                title=o.get("title", ""),
                weight=float(o.get("weight", 0.0)),
                metric=o.get("metric", ""),
                target=(None if o.get("target") is None else float(o["target"])),
                direction=o.get("direction", "up"),
            )
            for o in (d.get("objectives") or [])
        ]
        return cls(
            version=int(d.get("version", 1)),
            identity=d.get("identity", ""),
            principles=list(d.get("principles") or []),
            objectives=objs,
            limits=dict(d.get("limits") or {}),
        )

    def ranked_gaps(
        self, self_state: dict
    ) -> list[tuple[Objective, float | None, float | None]]:
        """Objectives with (current, priority), highest measured priority
        first; unmeasured objectives (priority None) sorted last."""
        rows = []
        for o in self.objectives:
            cur = measure(o.metric, self_state)
            rows.append((o, cur, o.priority(cur)))
        rows.sort(key=lambda r: (r[2] is not None, r[2] or 0.0), reverse=True)
        return rows

    def limit(self, key: str, default):
        return self.limits.get(key, default)
