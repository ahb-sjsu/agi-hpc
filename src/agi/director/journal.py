# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Step 5 of the SDCC — Erebus's first-person journal.

Append-only JSONL at ``/archive/erebus/journal.jsonl``. Each SDCC cycle
writes one entry: what Erebus perceived about itself, what it decided,
and (later phases) what it dispatched. The journal *is* the continuity of
identity across cycles — read back at the top of the next cycle and
surfaced on the dashboard.

Mirrors ``agi.primer.events`` (O_APPEND JSONL, tail-reader, never crashes
the caller). Distinct file so Erebus's reflective narrative and the
Primer's teaching telemetry don't interleave.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("director.journal")

DEFAULT_PATH = Path(
    os.environ.get("DIRECTOR_JOURNAL_PATH", "/archive/erebus/journal.jsonl")
)


def append(
    *,
    ts: float,
    cycle: int,
    kind: str,
    summary: str,
    detail: str = "",
    proof: str = "",
    path: Path | None = None,
) -> dict:
    """Append one journal entry. Returns the record (also for NATS publish).

    ``ts`` is passed in rather than read from the clock so the loop stays
    deterministic and testable. Write errors are swallowed — journaling
    must never crash the cycle."""
    target = path or DEFAULT_PATH
    record = {
        "ts": round(float(ts), 3),
        "cycle": int(cycle),
        "kind": kind,
        "summary": summary,
        "detail": detail[:2000],
        "proof": proof,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001 — journaling is best-effort
        log.warning("journal append failed: %s", e)
    return record


def tail(n: int = 20, path: Path | None = None) -> list[dict]:
    """Return the last ``n`` journal entries (newest last), or [] if none."""
    target = path or DEFAULT_PATH
    try:
        with open(target, "rb") as f:
            data = f.read()
    except (FileNotFoundError, OSError):
        return []
    out: list[dict] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-n:]
