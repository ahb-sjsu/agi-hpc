# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Step 4 of the SDCC — dispatch a gated goal into a safe channel.

A **closed whitelist** of action types. The Director cannot invent actions;
it can only fill the args of a registered type, and every dispatch re-checks
``policy.gate(tier, verb)`` (which still refuses the forbidden host-lifecycle
verbs). Actions are chosen to be additive and reversible.

B1 enables exactly one action: ``teach_task`` — the Director asks Erebus's
own Primer to teach a chosen high-value stuck task. It writes a request to
``/archive/erebus/directives.json``; a small, guarded reader in the Primer's
task selection (``_director_priority`` in ``agi.primer.service``) boosts
requested tasks to the front. This is **not** zero-integration — it needs
that reader — but the request file itself is additive and reversible (drop
the entry to undo).

Other action types are registered with their tier for documentation but are
not yet dispatchable; calling one raises ``NotImplementedError`` so the
whitelist stays honest about what is actually wired.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .policy import AutonomyTier, gate

log = logging.getLogger("director.dispatch")

DEFAULT_DIRECTIVES_PATH = Path("/archive/erebus/directives.json")

# Action type → (required tier, declared verb). The verb must not be on the
# forbidden host-lifecycle list — these are all additive data writes.
ACTION_SPECS = {
    "teach_task": (AutonomyTier.L2, "append_directive"),
    # Registered for the roadmap; not yet dispatchable (see module docstring).
    "prioritize_arc": (AutonomyTier.L2, "write_hint"),
    "schedule_dreaming": (AutonomyTier.L2, "request_window"),
    "write_note": (AutonomyTier.L1, "publish_note"),
}


@dataclass
class DispatchResult:
    ok: bool
    detail: str = ""
    reversible_ref: str = ""  # how to undo this dispatch, for the audit trail


def is_registered(action_type: str) -> bool:
    return action_type in ACTION_SPECS


def is_dispatchable(action_type: str) -> bool:
    """True if the action type is registered AND actually wired (has a
    dispatcher). Registered-but-unwired types return False."""
    return action_type in _DISPATCHERS


def dispatch(goal, *, directives_path: Path = DEFAULT_DIRECTIVES_PATH,
             cycle: int = 0, ceiling: AutonomyTier | None = None) -> DispatchResult:
    """Execute a gated goal's action. Raises on unknown/forbidden/over-tier;
    returns a :class:`DispatchResult` on success."""
    action = goal.action or {}
    atype = action.get("type", "")
    if atype not in ACTION_SPECS:
        raise ValueError(f"unknown action type: {atype!r}")
    tier, verb = ACTION_SPECS[atype]
    # Re-check policy at dispatch time (defense in depth vs the deliberate step).
    gate(tier, verb=verb, ceiling=ceiling)
    fn = _DISPATCHERS.get(atype)
    if fn is None:
        raise NotImplementedError(f"action {atype!r} registered but not yet enabled")
    return fn(action.get("args", {}), goal, directives_path, cycle)


# ── dispatchers ──────────────────────────────────────────────────


def _teach_task(args: dict, goal, directives_path: Path, cycle: int) -> DispatchResult:
    """Request that the Primer teach a specific stuck ARC task.

    Appends an idempotent request to directives.json. The Primer's
    ``_director_priority`` reader consumes ``teach_task`` entries and boosts
    those tasks in its pick order."""
    task = int(args["task"])
    reqs = _load_directives(directives_path)
    key = f"teach_task:{task}"
    if not any(r.get("key") == key and r.get("status") == "open" for r in reqs):
        reqs.append({
            "key": key,
            "type": "teach_task",
            "task": task,
            "goal_id": goal.id,
            "cycle": cycle,
            "rationale": (goal.provenance or {}).get("rationale", "")[:200],
            "status": "open",
        })
        _save_directives(directives_path, reqs)
    return DispatchResult(
        ok=True,
        detail=f"requested Primer teaching of task {task:03d}",
        reversible_ref=key,
    )


_DISPATCHERS = {
    "teach_task": _teach_task,
    # prioritize_arc / schedule_dreaming / write_note: not yet wired.
}


# ── directives file helpers (the Director's outbound request log) ─


def _load_directives(path: Path) -> list[dict]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return list(d.get("requests", [])) if isinstance(d, dict) else []
    except (FileNotFoundError, ValueError, OSError):
        return []


def _save_directives(path: Path, reqs: list[dict]) -> None:
    from agi.common.atomic_write import atomic_write_text

    atomic_write_text(Path(path), json.dumps({"requests": reqs}, indent=2))


def open_teach_tasks(path: Path = DEFAULT_DIRECTIVES_PATH) -> list[int]:
    """Task numbers with an open teach_task request — the Primer reader
    imports this to prioritize director-requested tasks."""
    return [
        int(r["task"])
        for r in _load_directives(path)
        if r.get("type") == "teach_task" and r.get("status") == "open" and "task" in r
    ]


def close_directive(key: str, path: Path = DEFAULT_DIRECTIVES_PATH) -> bool:
    """Mark a directive fulfilled (e.g. once the task is taught). Returns True
    if a matching open directive was closed."""
    reqs = _load_directives(path)
    changed = False
    for r in reqs:
        if r.get("key") == key and r.get("status") == "open":
            r["status"] = "closed"
            changed = True
    if changed:
        _save_directives(path, reqs)
    return changed
