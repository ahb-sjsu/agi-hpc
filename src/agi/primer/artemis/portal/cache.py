# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""In-memory caches for the GM portal.

The portal answers Keeper queries by reading a process-local snapshot
of state that's fed via NATS — we don't re-hit Google, don't re-poll,
don't re-read the SQLite chat store for every keystroke. Caches are
invalidated when new NATS events arrive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneState:
    name: str = "TRANSIT · DAY 184"
    flags: str = "NOMINAL"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "flags": self.flags}


@dataclass
class SheetCache:
    """Latest seen rows per sheet, keyed by row id.

    Fed by the ``agi.rh.artemis.sheet.snapshot.<name>`` and
    ``agi.rh.artemis.sheet.<name>`` subjects. Thread-safe because the
    NATS callback and the HTTP handler live on the same asyncio loop
    but the aiohttp server can parallelize requests.
    """

    sheets: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Merge rows into the cache for a sheet.

        A row with an unknown ``id`` is added; a row with a known id
        overwrites (diffs are idempotent).
        """
        if not name or not rows:
            return
        with self._lock:
            bucket = self.sheets.setdefault(name, {})
            for r in rows:
                rid = r.get("id")
                if not rid:
                    continue
                bucket[str(rid)] = {**bucket.get(str(rid), {}), **r}

    def replace_all(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Snapshot replace — used when a full snapshot arrives."""
        with self._lock:
            self.sheets[name] = {str(r["id"]): dict(r) for r in rows if r.get("id")}

    def rows(self, name: str) -> list[dict[str, Any]]:
        """All rows for a sheet, sorted by name/id for stable UI."""
        with self._lock:
            bucket = self.sheets.get(name, {})
            out = list(bucket.values())
        out.sort(key=lambda r: (str(r.get("name") or r.get("id") or "")))
        return out

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self.sheets.keys())
