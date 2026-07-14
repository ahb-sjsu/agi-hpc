# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""The Director daemon — Erebus's Self-Directed Cognition Cycle (SDCC).

Phase A: self-model + read-only reflection. Every tick the Director
perceives Erebus's own state, reconciles the persistent self-model,
journals the delta in the first person, and publishes to the
``agi.director.*`` NATS node + a ``director_status.json`` artifact for the
dashboard / HTTP API. **Autonomy is pinned to L0** — the Director cannot
act on the system in this phase (steps 3–4 of the full SDCC arrive in
Phase B behind the tier gate).

Runs as ``atlas-director.service`` (always-on, ``Restart=always``, CPU-
only). Honors the sentinel kill-switch ``/archive/neurogolf/.director_
disabled``: while it exists, cycles are skipped but the unit stays up.

Env:
  DIRECTOR_DIR            self-model + journal dir (default /archive/erebus)
  DIRECTOR_TICK_S         light-tick interval seconds (default 3600)
  DIRECTOR_DEEP_HOUR      UTC hour for the nightly deep cycle (default 9 ≈ 2am PST)
  DIRECTOR_MAX_TIER       autonomy ceiling (default L0; see policy.py)
  DIRECTOR_NATS           NATS servers (default nats://127.0.0.1:4222)
  DIRECTOR_MIN_ATTEMPTS   stuck-task threshold for perception (default 10)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agi.common.atomic_write import atomic_write_text

from . import events, journal, perceive
from .policy import AutonomyTier, max_tier
from .self_model import SelfModel

log = logging.getLogger("director")


@dataclass
class Config:
    directory: Path
    tick_s: int
    deep_hour: int
    nats_servers: str
    min_attempts: int
    paths: perceive.Paths

    @classmethod
    def from_env(cls) -> "Config":
        directory = Path(os.environ.get("DIRECTOR_DIR", "/archive/erebus"))
        return cls(
            directory=directory,
            tick_s=int(os.environ.get("DIRECTOR_TICK_S", "3600")),
            deep_hour=int(os.environ.get("DIRECTOR_DEEP_HOUR", "9")),
            nats_servers=os.environ.get("DIRECTOR_NATS", "nats://127.0.0.1:4222"),
            min_attempts=int(os.environ.get("DIRECTOR_MIN_ATTEMPTS", "10")),
            paths=perceive.Paths(),
        )


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _proof(prev: str, record: dict) -> str:
    """SHA-256 hash-chained decision proof over the cycle record.

    Chains on the previous proof so the journal is tamper-evident: any
    edit to a past cycle breaks every subsequent hash. This is the same
    idea as the Strategic-layer decision proofs, kept local to the
    Director's journal in Phase A."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


def _last_proof(directory: Path) -> str:
    try:
        return (directory / "last_proof.txt").read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return "genesis"


# ── reconciliation ───────────────────────────────────────────────


_VALUES = [
    "Verify before I claim — a wrong note is worse than no note.",
    "Keep an honest account of what I am and am not.",
    "Act only within my charter and my autonomy ceiling.",
    "Never restart, kill, or reboot my host.",
]


def reconcile(prev: SelfModel, state: perceive.SelfState, *, ts: float,
              tier: AutonomyTier) -> tuple[SelfModel, str]:
    """Build the next self-model from the prior one + a fresh snapshot.

    Returns ``(new_model, delta_summary)`` where ``delta_summary`` is a
    first-person sentence describing what changed — the journal payload.
    """
    new = SelfModel(
        updated_at=_iso(ts),
        cycle=prev.cycle + 1,
        autonomy_tier=tier.name,
        capabilities=perceive.derive_capabilities(state),
        limitations=perceive.derive_limitations(state),
        active_goals=prev.active_goals,  # Phase B populates from the charter
        open_problems=_open_problems(state),
        values=_VALUES,
        recent_history=prev.recent_history[-9:],  # keep last ~10 after we append
        self_state=state.to_dict(),
    )

    # Delta vs the prior snapshot — the substance of the journal entry.
    d_solved = state.tasks_solved - int(prev.self_state.get("tasks_solved", 0))
    d_stuck = state.tasks_stuck - int(prev.self_state.get("tasks_stuck", 0))
    bits = [
        f"cycle {new.cycle}: {state.tasks_solved}/{state.tasks_total} tasks solved",
        f"{state.tasks_stuck} stuck",
        f"help-queue {state.help_queue_len}",
    ]
    if d_solved:
        bits.append(f"solved {d_solved:+d} since last cycle")
    if d_stuck:
        bits.append(f"stuck {d_stuck:+d}")
    if state.primer_experts_degraded:
        bits.append(f"degraded experts: {', '.join(state.primer_experts_degraded)}")
    summary = "; ".join(bits) + "."

    new.recent_history = (prev.recent_history + [summary])[-10:]
    return new, summary


def _open_problems(state: perceive.SelfState) -> list[str]:
    probs: list[str] = []
    if state.tasks_stuck:
        probs.append(f"{state.tasks_stuck} ARC tasks stuck past the attempt threshold.")
    if state.help_queue_len:
        probs.append(f"{state.help_queue_len} open questions in Erebus's help queue.")
    if not state.faculties_online:
        probs.append("No faculty state files found — am I actually running on Atlas?")
    return probs


# ── one cycle ────────────────────────────────────────────────────


async def run_cycle(cfg: Config, node: events.DirectorNode, *, deep: bool) -> dict:
    """Run one SDCC cycle. Phase A = perceive → reconcile → journal → publish.

    Returns the cycle record. Never raises for expected file/broker issues;
    unexpected errors propagate to the loop's guard."""
    ts = time.time()
    tier = max_tier()  # Phase A: L0

    # Step 1 — perceive self
    state = perceive.gather(cfg.paths, min_attempts=cfg.min_attempts)

    # Step 2 — update self-model
    prev = SelfModel.load(cfg.directory)
    model, summary = reconcile(prev, state, ts=ts, tier=tier)
    model.save(cfg.directory)

    kind = "deep-cycle" if deep else "tick"

    # Decision proof (chained) — computed before journaling so the on-disk
    # journal line itself is tamper-evident: editing any past cycle breaks
    # every subsequent hash.
    record = {
        "ts": round(ts, 3), "cycle": model.cycle, "kind": kind,
        "tier": tier.name, "summary": summary,
    }
    prev_proof = _last_proof(cfg.directory)
    proof = _proof(prev_proof, record)
    record["proof"] = proof
    atomic_write_text(cfg.directory / "last_proof.txt", proof)

    # Step 5 — reflect / journal, carrying the chained proof (steps 3–4 are
    # Phase B, behind the tier gate).
    entry = journal.append(
        ts=ts, cycle=model.cycle, kind=kind, summary=summary,
        detail=json.dumps(state.to_dict(), separators=(",", ":")),
        proof=proof,
        path=cfg.directory / "journal.jsonl",
    )
    atomic_write_text(
        cfg.directory / "director_status.json",
        json.dumps(
            {
                "summary": model.summary(),
                "last_cycle": record,
                "next_wake_s": cfg.tick_s,
                "enabled": not cfg.paths.disabled_sentinel.exists(),
                "max_tier": tier.name,
            },
            indent=2,
        ),
    )

    # Publish to the global workspace (no-op if NATS is absent)
    await node.publish_state(model.summary())
    await node.publish_journal(entry)
    await node.publish_cycle(record)

    log.info("%s | %s", kind, summary)
    return record


# ── command handler (Phase A: read-only + self-control) ──────────


def _make_handler(cfg: Config, node: events.DirectorNode):
    """Inbound-command handler for ``agi.director.command``.

    Phase A commands never touch the host or act on the world — they read
    status or control the Director's own cadence via the sentinel."""

    async def handler(cmd: dict) -> dict:
        verb = str(cmd.get("cmd", "")).lower()
        if verb == "status":
            return {"ok": True, "status": SelfModel.load(cfg.directory).summary()}
        if verb == "pause":
            cfg.paths.disabled_sentinel.parent.mkdir(parents=True, exist_ok=True)
            cfg.paths.disabled_sentinel.touch()
            return {"ok": True, "paused": True}
        if verb == "resume":
            cfg.paths.disabled_sentinel.unlink(missing_ok=True)
            return {"ok": True, "paused": False}
        if verb == "run-now":
            await run_cycle(cfg, node, deep=False)
            return {"ok": True, "ran": True}
        return {"ok": False, "error": f"unknown or not-yet-permitted command: {verb!r}"}

    return handler


# ── main loop ────────────────────────────────────────────────────


async def _main_async() -> None:
    cfg = Config.from_env()
    node = events.DirectorNode(cfg.nats_servers)
    await node.connect()
    await node.subscribe_commands(_make_handler(cfg, node))
    log.info(
        "Director online. dir=%s tick=%ds deep_hour=%02dZ tier=%s",
        cfg.directory, cfg.tick_s, cfg.deep_hour, max_tier().name,
    )
    last_deep_day = -1
    while True:
        try:
            if cfg.paths.disabled_sentinel.exists():
                log.info("disabled sentinel present; skipping cycle")
            else:
                now = time.gmtime()
                is_deep = now.tm_hour == cfg.deep_hour and now.tm_yday != last_deep_day
                if is_deep:
                    last_deep_day = now.tm_yday
                await run_cycle(cfg, node, deep=is_deep)
        except Exception as e:  # noqa: BLE001 - never let the loop die
            log.exception("cycle error: %s", e)
        await asyncio.sleep(cfg.tick_s)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("DIRECTOR_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
