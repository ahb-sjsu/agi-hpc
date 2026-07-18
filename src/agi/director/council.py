# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Divine-Council enrichment for the Director's deliberation (SDCC step 3).

Advisory, safety-preserving prioritization. By the time this runs, the
deterministic core of :func:`agi.director.deliberate.deliberate` has ALREADY
ethics-gated every proposal; the Council only **re-orders** those gated
proposals by strategic value toward the charter. It never adds, removes, or
invents an action, so the safety invariant (only ``gated`` goals may act, a
missing gate denies) is untouched — the worst a bad Council can do is choose a
less-useful ordering among already-approved goals.

Routing: the vMOE ``council`` hint, which prefers the Anthropic **Fable** expert
(priority 5) when ``ANTHROPIC_API_KEY`` is set, and otherwise cascades to the
NRP experts (kimi, glm-5) — see :func:`agi.primer.vmoe.default_experts`.

Disabled unless ``DIRECTOR_COUNCIL=1`` **and** a ``council``-tagged expert
exists; and it always **fails open** to the deterministic order, so it never
adds a hard dependency to the safety-critical loop.

Wire it as the ``enrich`` hook of ``deliberate(...)``::

    from .council import build_council_enrich
    proposals = deliberate(..., enrich=build_council_enrich())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Callable

log = logging.getLogger("director.council")


def _run(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, whether or not an event
    loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}
    t = threading.Thread(target=lambda: box.update(r=asyncio.run(coro)), daemon=True)
    t.start()
    t.join()
    return box.get("r")


def _extract_json_array(text: str) -> str:
    t = (text or "").strip()
    a, b = t.find("["), t.rfind("]")
    return t[a:b + 1] if 0 <= a < b else t


def _prompt(proposals: list, ctx: dict) -> list[dict[str, str]]:
    items = [
        {"id": g.id or f"idx{i}",
         "title": getattr(g, "title", ""),
         "rationale": (getattr(g, "provenance", {}) or {}).get("rationale", "")}
        for i, g in enumerate(proposals)
    ]
    return [
        {"role": "system", "content":
            "You are the Divine Council: a strategic prioritizer for an autonomous "
            "research agent (Erebus). You are given candidate goals that have ALREADY "
            "passed the ethics gate. Rank them by strategic value toward the agent's "
            "charter and long-horizon progress. Reply with ONLY a JSON array of the "
            "goal ids in recommended priority order (most valuable first). Do not add, "
            "remove, rewrite, or invent goals; use each id exactly once."},
        {"role": "user",
         "content": json.dumps({"cycle": ctx.get("cycle"), "goals": items})},
    ]


def _default_vmoe():
    try:
        from agi.primer.vmoe import vMOE
        return vMOE()
    except Exception as e:  # noqa: BLE001 - vMOE/openai optional; council is best-effort
        log.info("vMOE unavailable for council (skipping): %s", e)
        return None


def build_council_enrich(
    vmoe=None,
    *,
    hint: str = "council",
    timeout_s: float = 60.0,
    enabled: bool | None = None,
) -> Callable[[list, dict], list]:
    """Return an ``enrich(proposals, ctx) -> proposals`` for ``deliberate``.

    ``vmoe``   an ``agi.primer.vmoe.vMOE`` instance; ``None`` → lazily built.
    ``hint``   vMOE role hint used to select the Council expert pool.
    ``enabled`` overrides the ``DIRECTOR_COUNCIL`` env flag (for tests).

    The returned hook re-orders the already-gated proposals by the Council's
    recommendation and returns them; on ANY problem it returns the input
    unchanged (fail-open).
    """

    def enrich(proposals: list, ctx: dict) -> list:
        on = (enabled if enabled is not None
              else os.environ.get("DIRECTOR_COUNCIL") == "1")
        if not on or len(proposals) < 2:
            return proposals
        try:
            moe = vmoe or _default_vmoe()
            if moe is None or not moe.by_hint(hint):
                return proposals
            resp = _run(
                moe.cascade(_prompt(proposals, ctx), hint=hint, timeout_s=timeout_s)
            )
            if resp is None or not getattr(resp, "ok", False):
                return proposals
            order = json.loads(_extract_json_array(resp.content))
            rank = {str(pid): i for i, pid in enumerate(order)}
            indexed = list(enumerate(proposals))
            indexed.sort(key=lambda it: (rank.get(it[1].id or f"idx{it[0]}",
                                                   len(order) + it[0]), it[0]))
            reordered = [g for _, g in indexed]
            log.info("council reprioritized %d goals via %s",
                     len(proposals), getattr(resp, "expert", "?"))
            return reordered
        except Exception as e:  # noqa: BLE001 - advisory only; never break deliberation
            log.warning("council enrich failed (fail-open): %s", e)
            return proposals

    return enrich
