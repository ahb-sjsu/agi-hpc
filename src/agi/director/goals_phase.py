# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SDCC steps 3–4 orchestration — deliberate then (maybe) dispatch.

Keeps the goal loop out of the daemon plumbing in ``service.py``. Called
each cycle only when the running autonomy ceiling is ≥ L1. Behavior by
tier:

  L1  deliberate + gate → journal proposed/gated/rejected goals. No action.
  L2  additionally dispatch ``gated`` goals through the safe-channel registry
      (``teach_task``), within the resource governor's thermal / GPU-loan /
      rate budget, and retire completed teach goals.

If no charter is present the Director has nothing to pursue and this phase
is a no-op — Erebus stays purely reflective (Phase A behavior).
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import dispatch as dispatch_mod
from .charter import Charter
from .deliberate import deliberate, read_stuck_tasks
from .gate import DemeGate, GoalGate
from .goals import GoalTree, Status
from .govern import Budget, Governor
from .policy import AutonomyTier

log = logging.getLogger("director.goals")


def _is_taught(wiki_dir: Path, task: int) -> bool:
    """A teach_task goal is complete once a verified sensei note exists."""
    return (Path(wiki_dir) / f"sensei_task_{task:03d}.md").exists()


def _event(phase: str, goal) -> dict:
    return {
        "phase": phase,
        "goal": {
            "id": goal.id,
            "title": goal.title,
            "status": goal.status,
            "tier": goal.tier,
            "action": goal.action,
        },
    }


def run(cfg, self_state: dict, tier: AutonomyTier, cycle: int,
        *, gate: GoalGate | None = None, governor: Governor | None = None
        ) -> tuple[str, list[dict], dict]:
    """Run deliberation (+ dispatch if L2). Returns (note, goal_events, counts).

    ``governor`` is injectable for testing; when None a default one is built
    from the charter's limits."""
    charter = Charter.load(cfg.charter_path)
    if charter is None:
        return "", [], {}

    tree = GoalTree.load(cfg.goals_path)
    if gate is None:
        gate = DemeGate.try_build()

    def stuck_provider():
        return read_stuck_tasks(cfg.paths.memory, min_attempts=cfg.min_attempts)

    proposals = deliberate(
        charter, self_state, tree, gate,
        cycle=cycle, stuck_provider=stuck_provider, max_proposals=cfg.max_proposals,
    )
    events = [_event("proposed", g) for g in proposals]
    n_gated = sum(1 for g in proposals if g.status == Status.GATED.value)
    n_rejected = sum(1 for g in proposals if g.status == Status.REJECTED.value)
    dispatched = blocked = done = 0

    if tier >= AutonomyTier.L2:
        gov = governor
        if gov is None:
            budget = Budget(
                thermal_c=float(charter.limit("thermal_c", 82)),
                max_per_cycle=int(charter.limit("max_dispatch_per_cycle", 3)),
                max_per_day=int(charter.limit("max_dispatch_per_day", 20)),
            )
            gov = Governor(budget, rate_path=cfg.directory / "dispatch_rate.json")
        gov.start_cycle()

        # Retire completed teach goals first (so they free rate budget / stop
        # re-counting as open).
        for g in tree.by_status(Status.ACTIVE.value):
            if (g.action or {}).get("type") != "teach_task":
                continue
            task = (g.action or {}).get("args", {}).get("task")
            if task is not None and _is_taught(cfg.wiki_dir, int(task)):
                tree.transition(g.id, Status.DONE.value, cycle=cycle)
                dispatch_mod.close_directive(
                    f"teach_task:{int(task)}", cfg.directives_path
                )
                events.append(_event("done", g))
                done += 1

        # Dispatch gated goals within tier + budget.
        for g in tree.by_status(Status.GATED.value):
            if AutonomyTier.parse(g.tier, AutonomyTier.L3) > tier:
                continue
            if not dispatch_mod.is_dispatchable((g.action or {}).get("type", "")):
                continue
            dec = gov.can_dispatch()
            if not dec.ok:
                tree.transition(g.id, Status.BLOCKED.value, cycle=cycle)
                events.append(_event("blocked", g))
                blocked += 1
                log.info("dispatch blocked: %s", dec.reason)
                break  # budget/thermal exhausted for this cycle
            try:
                res = dispatch_mod.dispatch(
                    g, directives_path=cfg.directives_path, cycle=cycle, ceiling=tier
                )
            except Exception as e:  # noqa: BLE001 - a bad dispatch blocks, never crashes
                tree.transition(g.id, Status.BLOCKED.value, cycle=cycle)
                log.warning("dispatch error on %s: %s", g.id, e)
                blocked += 1
                continue
            tree.transition(g.id, Status.ACTIVE.value, cycle=cycle)
            gov.record_dispatch()
            dispatched += 1
            log.info("dispatched %s: %s", g.id, res.detail)
            events.append(_event("dispatched", g))

    tree.save(cfg.goals_path)
    counts = {
        "proposed": len(proposals), "gated": n_gated, "rejected": n_rejected,
        "dispatched": dispatched, "blocked": blocked, "done": done,
    }
    note = (
        f"goals: {counts['proposed']}p/{n_gated}gated/{n_rejected}rej/"
        f"{dispatched}disp/{blocked}blk/{done}done"
    )
    return note, events, counts
