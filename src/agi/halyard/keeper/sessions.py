# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Session registry for halyard-keeper-backend.

Holds minimal metadata per active session: id, state (open /
paused / closed), created-at, paused-at. The real "session" is
whatever's in the LiveKit room; this registry is the Keeper's
control-plane view on top.

Backed by an in-memory dict plus a per-session-id append-only log
on disk so state survives restarts. We don't need a database for
a runtime that hosts one game at a time — dict + jsonl is
sufficient and aligns with halyard-state's archive layout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger("halyard.keeper.sessions")


class SessionState(Enum):
    OPEN = "open"          # room active, players may join
    PAUSED = "paused"      # room active; AIs silenced; Keeper pause in effect
    CLOSED = "closed"      # session ended; no new joins


@dataclass
class Session:
    id: str
    state: SessionState = SessionState.OPEN
    created_at: float = field(default_factory=time.time)
    paused_at: float | None = None
    closed_at: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


# ─────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────


class SessionNotFound(KeyError):
    pass


class SessionAlreadyExists(ValueError):
    pass


class InvalidTransition(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


class SessionRegistry:
    """Per-process session registry.

    Construction does no I/O. Methods are async to match the rest
    of the halyard async surface; today's I/O is synchronous JSONL
    writes which are fast enough not to need off-loading.
    """

    def __init__(self, *, archive_root: Path | None = None) -> None:
        self._archive_root = Path(
            archive_root
            or os.environ.get("HALYARD_ARCHIVE_ROOT", "/archive/halyard")
        )
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ── public API ──

    async def create(
        self, session_id: str, *, note: str = ""
    ) -> Session:
        async with self._lock:
            if session_id in self._sessions:
                raise SessionAlreadyExists(session_id)
            s = Session(id=session_id, note=note)
            self._sessions[session_id] = s
            self._append_log(s, action="create")
            return s

    async def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise SessionNotFound(session_id)
        return self._sessions[session_id]

    async def list(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at)

    async def pause(self, session_id: str) -> Session:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                raise SessionNotFound(session_id)
            if s.state is SessionState.CLOSED:
                raise InvalidTransition(
                    f"cannot pause closed session {session_id}"
                )
            s.state = SessionState.PAUSED
            s.paused_at = time.time()
            self._append_log(s, action="pause")
            return s

    async def resume(self, session_id: str) -> Session:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                raise SessionNotFound(session_id)
            if s.state is SessionState.CLOSED:
                raise InvalidTransition(
                    f"cannot resume closed session {session_id}"
                )
            s.state = SessionState.OPEN
            s.paused_at = None
            self._append_log(s, action="resume")
            return s

    async def close(self, session_id: str) -> Session:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                raise SessionNotFound(session_id)
            s.state = SessionState.CLOSED
            s.closed_at = time.time()
            self._append_log(s, action="close")
            return s

    def is_accepting_joins(self, session_id: str) -> bool:
        """Public predicate used by the token-mint endpoint.

        Players may join a session in OPEN or PAUSED state; in
        PAUSED state the AIs are silenced but the room still
        accepts humans. CLOSED rejects joins. Unknown session =
        permissive in dev (the endpoint implicitly creates), but
        the caller decides.
        """
        s = self._sessions.get(session_id)
        if s is None:
            return True
        return s.state is not SessionState.CLOSED

    async def ensure(self, session_id: str) -> Session:
        """Get or create. Useful for the token-mint endpoint so a
        player can join without the Keeper having explicitly
        pre-registered the session (the Keeper can still ``close``
        to evict). Races are fine — we're not racing create vs.
        close."""
        try:
            return await self.get(session_id)
        except SessionNotFound:
            try:
                return await self.create(session_id)
            except SessionAlreadyExists:
                return await self.get(session_id)

    # ── persistence ──

    def _log_path(self, session_id: str) -> Path:
        return (
            self._archive_root / "keeper" / "sessions" / f"{session_id}.log.jsonl"
        )

    def _append_log(self, s: Session, *, action: str) -> None:
        try:
            p = self._log_path(s.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.time(),
                "action": action,
                **s.to_dict(),
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("session-log append failed for %s: %s", s.id, e)
