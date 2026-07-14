# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""The Director's goal tree — its working memory for self-direction.

Persisted atomically at ``/archive/erebus/goals.json``. Each goal carries
its required autonomy tier, its (optional) dispatch action, the gate
verdict that cleared it, and provenance. Every status transition is meant
to be journaled + proof-chained by the caller (service.py) so the whole
trajectory of Erebus's self-direction is auditable.

Status lifecycle::

    proposed ─► gated ─► active ─► done
        │         │         └─► blocked (resource/dependency; retried)
        └─► rejected (gate veto)      └─► abandoned (stale/superseded)

``proposed`` goals have not cleared the ethics gate; only ``gated`` goals
are eligible to act, and only when their tier ≤ the running ceiling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

DEFAULT_PATH = Path("/archive/erebus/goals.json")


class Status(str, Enum):
    PROPOSED = "proposed"
    GATED = "gated"
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"
    REJECTED = "rejected"


@dataclass
class Goal:
    id: str
    title: str
    parent: str | None = None
    kind: str = "subgoal"  # objective | subgoal | action
    status: str = Status.PROPOSED.value
    tier: str = "L2"
    action: dict | None = None  # {"type": "teach_task", "args": {...}}
    metric: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    created_cycle: int = 0
    updated_cycle: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def key(self) -> str:
        """Dedupe key: a goal is 'the same' if it targets the same action.

        Used so the Director doesn't re-propose an identical teach_task
        every cycle while the first is still in flight."""
        if self.action:
            args = json.dumps(self.action.get("args", {}), sort_keys=True)
            return f"{self.action.get('type')}:{args}"
        return f"{self.parent}:{self.title}"


class GoalTree:
    """In-memory goal set with atomic JSON persistence."""

    def __init__(self, goals: list[Goal] | None = None, updated_cycle: int = 0) -> None:
        self.goals: list[Goal] = goals or []
        self.updated_cycle = updated_cycle

    # ── persistence ──────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "GoalTree":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return cls()
        return cls(
            goals=[Goal.from_dict(g) for g in (d.get("goals") or [])],
            updated_cycle=int(d.get("updated_cycle", 0)),
        )

    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        from agi.common.atomic_write import atomic_write_text

        atomic_write_text(
            Path(path),
            json.dumps(
                {
                    "version": 1,
                    "updated_cycle": self.updated_cycle,
                    "goals": [g.to_dict() for g in self.goals],
                },
                indent=2,
            ),
        )

    # ── queries ──────────────────────────────────────────────────

    def by_status(self, *statuses: str) -> list[Goal]:
        s = set(statuses)
        return [g for g in self.goals if g.status in s]

    def get(self, goal_id: str) -> Goal | None:
        return next((g for g in self.goals if g.id == goal_id), None)

    def has_open_key(self, key: str) -> bool:
        """True if a non-terminal goal with this action key already exists —
        so we don't stack duplicate dispatches."""
        open_ = {Status.PROPOSED.value, Status.GATED.value, Status.ACTIVE.value,
                 Status.BLOCKED.value}
        return any(g.status in open_ and g.key() == key for g in self.goals)

    # ── mutation ─────────────────────────────────────────────────

    def next_id(self) -> str:
        n = 1 + max((_id_num(g.id) for g in self.goals), default=0)
        return f"g-{n:04d}"

    def add(self, goal: Goal) -> Goal:
        if not goal.id:
            goal.id = self.next_id()
        self.goals.append(goal)
        return goal

    def transition(
        self, goal_id: str, status: str, *, cycle: int, gate: dict | None = None
    ) -> Goal | None:
        g = self.get(goal_id)
        if g is None:
            return None
        g.status = status
        g.updated_cycle = cycle
        if gate is not None:
            g.gate = gate
        return g


def _id_num(gid: str) -> int:
    try:
        return int(str(gid).split("-")[-1])
    except (ValueError, IndexError):
        return 0
