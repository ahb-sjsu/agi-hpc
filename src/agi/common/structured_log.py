"""Structured lifecycle logger — JSON-lines per-subsystem event stream.

Complements the existing human-readable logs. Each subsystem that wants
lifecycle visibility opens a ``LifecycleLogger`` and emits events like
``cycle_start``, ``attempt_end``, ``tick_complete``. Events are written
as one JSON object per line to
``/archive/neurogolf/lifecycle/<subsystem>.jsonl`` (configurable).

Downstream consumers:
- ``/api/jobs/recent`` in ``telemetry_server.py`` tails the files and
  returns the latest N events for the dashboard.
- Ad-hoc analysis: ``jq`` queries work directly.

Why a parallel stream rather than replacing the pretty-printed logs?

1. Zero-risk migration. The existing log parsers (telemetry activity
   classifier, log-tail ring buffer, journalctl grep habits in the
   on-call playbook) keep working unchanged.
2. Machine-readable from day one. We skip the "is this JSON or plain
   text" question for everything emitted through this module.
3. Clear division of labour. The pretty-printed log is for humans at
   3am. The lifecycle stream is for programs summarizing behaviour
   over time. Different audiences, different formats.

Also see ``docs/METRICS_CONTRIBUTOR_GUIDE.md`` §3 for naming
conventions and the broader log-format roadmap.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LIFECYCLE_DIR = Path(
    os.environ.get("ATLAS_LIFECYCLE_DIR", "/archive/neurogolf/lifecycle")
)


@dataclass
class LifecycleEvent:
    """One event as persisted — matches the JSON shape on disk.

    Fields beyond the five below go into ``fields`` (free-form dict).
    Downstream consumers should not rely on a fixed set of keys in
    ``fields``; new event types add new keys over time.
    """

    ts: str  # ISO 8601 UTC with Z suffix
    subsystem: str
    event: str
    seq: int
    fields: dict[str, Any]


class LifecycleLogger:
    """Append-only JSON-lines lifecycle writer.

    Thread-safe via a simple lock on the file-write path. Single-process
    only — the Primer daemon and arc_scientist are both single-process,
    so this is fine. For multi-process use, switch to per-process files
    or introduce a NATS-backed writer.

    A sequence number is attached to every event so consumers can detect
    dropped events (e.g. if the file is rotated externally).
    """

    def __init__(
        self,
        subsystem: str,
        *,
        path: Path | str | None = None,
        lifecycle_dir: Path | None = None,
    ) -> None:
        self.subsystem = subsystem
        if path is not None:
            self.path = Path(path)
        else:
            base = Path(lifecycle_dir) if lifecycle_dir else DEFAULT_LIFECYCLE_DIR
            self.path = base / f"{subsystem}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._discover_seq()

    def _discover_seq(self) -> int:
        """Read the last line of the file (if it exists) and return the
        next sequence number. Starting fresh returns 0.

        This keeps ``seq`` monotonic across process restarts without
        requiring an external counter file."""
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                with open(self.path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    # Read the last 4KB (enough for one JSON line)
                    f.seek(max(0, size - 4096))
                    tail = f.read().decode("utf-8", errors="replace")
                for line in reversed(tail.strip().split("\n")):
                    if not line.strip():
                        continue
                    try:
                        last = json.loads(line)
                        return int(last.get("seq", -1)) + 1
                    except Exception:
                        continue
        except Exception:
            pass
        return 0

    def emit(self, event: str, **fields: Any) -> LifecycleEvent:
        """Append one lifecycle event.

        ``event`` follows the ``<subsystem>.<verb>`` convention enforced
        by making ``subsystem`` implicit — callers pass only the verb
        (``attempt_end``, not ``scientist.attempt_end``). The full name
        is reconstructed on read.
        """
        ev = LifecycleEvent(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            subsystem=self.subsystem,
            event=event,
            seq=-1,  # assigned under lock
            fields=_coerce_json_safe(fields),
        )
        with self._lock:
            ev.seq = self._seq
            self._seq += 1
            line = json.dumps(
                {
                    "ts": ev.ts,
                    "subsystem": ev.subsystem,
                    "event": ev.event,
                    "seq": ev.seq,
                    **ev.fields,
                },
                default=str,
            )
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return ev


def read_recent(
    subsystem: str,
    *,
    limit: int = 200,
    path: Path | str | None = None,
    lifecycle_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the last ``limit`` events for ``subsystem``, newest first.

    Cheap: reads the tail of the file only. Doesn't lock the writer;
    stale-read race is acceptable for dashboard consumption.
    """
    if path is not None:
        p = Path(path)
    else:
        base = Path(lifecycle_dir) if lifecycle_dir else DEFAULT_LIFECYCLE_DIR
        p = base / f"{subsystem}.jsonl"
    if not p.exists():
        return []
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Grab a generous tail; JSON lines are ~100-500 bytes
            tail_bytes = min(size, max(4096, limit * 800))
            f.seek(size - tail_bytes)
            chunk = f.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    events: list[dict[str, Any]] = []
    for line in reversed(chunk.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Partial line from mid-read; skip
            continue
        if len(events) >= limit:
            break
    return events


def _coerce_json_safe(d: dict[str, Any]) -> dict[str, Any]:
    """Best-effort conversion of non-JSON-serialisable fields so the
    writer never raises mid-emit. Tuples → lists; Path → str; objects
    with ``__dict__`` → dict; everything else → its repr.

    This is belt-and-braces — the ``default=str`` in ``json.dumps``
    handles most unknown types, but coercing up-front means the log
    line is inspectable when something goes wrong.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            try:
                json.dumps(v)  # test
                out[k] = v
            except Exception:
                out[k] = repr(v)
    return out
