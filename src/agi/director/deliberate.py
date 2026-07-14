# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Step 3 of the SDCC — deliberate on goals.

Given the charter, the current self-state, and the goal tree, propose new
sub-goals toward the highest-gap objectives and gate each through the
ethics stack. This is the deterministic core of self-direction:

    rank objectives by gap  →  propose concrete actions  →  ethics-gate each

The proposer is deterministic (no NRP dependency in the safety-critical
loop, so the daemon and its tests are robust). A vMOE / Divine-Council
enrichment — richer candidate generation and multi-perspective
prioritization — is a clean follow-on hook (``enrich`` parameter), not part
of the core path.

B1 supports one concrete action, ``teach_task``: for objectives keyed to
the ARC solve rate / stuck backlog, propose that the Primer teach the
highest-value stuck tasks. Every candidate is gated; only ``allow`` goals
become ``gated`` (eligible to act), and a missing gate denies (fail-safe).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .charter import Charter, Objective
from .gate import GateVerdict, GoalGate
from .goals import Goal, GoalTree, Status

log = logging.getLogger("director.deliberate")

# Objectives whose metric names one of these are pursued via teach_task.
_ARC_METRICS = {"arc_solved", "arc_stuck"}

DEFAULT_MEMORY = Path("/archive/neurogolf/arc_scientist_memory.json")


def read_stuck_tasks(
    memory_path: Path = DEFAULT_MEMORY, *, min_attempts: int = 10, limit: int = 20
) -> list[int]:
    """Ranked stuck ARC task numbers: partial-progress first (most likely to
    yield to one good lesson), then by attempt count. Solved tasks excluded.
    Defensive — returns [] on any read error."""
    try:
        mem = json.loads(Path(memory_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return []
    scored: list[tuple[int, int, int]] = []  # (has_partial, attempts, task)
    for k, tk in (mem.get("tasks") or {}).items():
        if not isinstance(tk, dict) or tk.get("solved"):
            continue
        try:
            tn = int(k)
        except ValueError:
            continue
        attempts = len(tk.get("attempts", []) or [])
        if attempts < min_attempts:
            continue
        scored.append((1 if (tk.get("best_correct") or 0) > 0 else 0, attempts, tn))
    scored.sort(reverse=True)
    return [tn for _, _, tn in scored[:limit]]


def _teach_goal(task: int, obj: Objective, cur: float | None, cycle: int) -> Goal:
    rationale = (
        f"Task {task:03d} is a stuck ARC puzzle; teaching it advances "
        f"'{obj.title}' (metric {obj.metric}={cur})."
    )
    return Goal(
        id="",
        title=f"Teach stuck ARC task {task:03d}",
        parent=obj.id,
        kind="action",
        status=Status.PROPOSED.value,
        tier="L2",
        action={"type": "teach_task", "args": {"task": task}},
        metric={"name": obj.metric, "current": cur, "target": obj.target},
        provenance={"origin": "director", "cycle": cycle, "rationale": rationale},
        created_cycle=cycle,
        updated_cycle=cycle,
    )


def _gate_goal(goal: Goal, gate: GoalGate) -> GateVerdict:
    task = (goal.action or {}).get("args", {}).get("task")
    desc = (
        f"Erebus self-directed goal: ask the Primer to teach ARC task {task}. "
        f"Purpose: {goal.provenance.get('rationale', '')}"
    )
    ctx = {
        "kind": "self_directed_goal",
        "action": (goal.action or {}).get("type"),
        "objective": goal.parent,
    }
    return gate.gate(desc, ctx)


def deliberate(
    charter: Charter,
    self_state: dict,
    tree: GoalTree,
    gate: GoalGate,
    *,
    cycle: int,
    stuck_provider=read_stuck_tasks,
    max_proposals: int = 5,
    enrich=None,  # optional vMOE/Council hook: (candidates, ctx) -> candidates
) -> list[Goal]:
    """Propose + gate new goals for this cycle. Returns the newly added goals
    (already inserted into ``tree`` with their gate verdicts applied). Does
    NOT dispatch — that is the service's step 4, behind the tier gate."""
    proposals: list[Goal] = []
    ranked = charter.ranked_gaps(self_state)

    # Candidate task pool for ARC-keyed objectives (shared across them).
    stuck = None

    for obj, cur, prio in ranked:
        if len(proposals) >= max_proposals:
            break
        if prio is None or prio <= 0:
            # Unmeasured or already-satisfied objective — nothing to propose.
            # Unmeasured ones are honestly skipped (see charter metric registry).
            continue
        if obj.metric not in _ARC_METRICS:
            # No supported action for this objective yet — skip rather than
            # fabricate work. Additional action types arrive in B3+.
            continue
        if stuck is None:
            stuck = stuck_provider()
        for task in stuck:
            if len(proposals) >= max_proposals:
                break
            cand = _teach_goal(task, obj, cur, cycle)
            if tree.has_open_key(cand.key()):
                continue  # already proposed/active — don't stack duplicates
            verdict = _gate_goal(cand, gate)
            cand.gate = verdict.to_dict()
            cand.status = Status.GATED.value if verdict.allow else Status.REJECTED.value
            tree.add(cand)
            proposals.append(cand)

    if enrich and proposals:
        try:
            proposals = enrich(proposals, {"cycle": cycle}) or proposals
        except Exception as e:  # noqa: BLE001 - enrichment is best-effort
            log.warning("deliberate enrich hook failed: %s", e)

    tree.updated_cycle = cycle
    return proposals
