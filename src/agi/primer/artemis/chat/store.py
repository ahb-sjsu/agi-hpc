# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SQLite+FTS5 chat store for the Keeper-searchable transcript.

The store is append-only — messages are never mutated. Every chat
message (player, ARTEMIS, keeper) is written with enough metadata
that the Keeper portal (S1h) can reconstruct any thread and
free-text-search the session.

Schema::
    messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,                 -- unix seconds
      session_id TEXT NOT NULL,
      from_id TEXT NOT NULL,            -- identity prefix, e.g. "player:imogen"
      to_id TEXT,                       -- NULL for broadcasts
      kind TEXT NOT NULL,               -- MessageKind value
      body TEXT NOT NULL,
      corr_id TEXT                      -- correlates a reply to its request
    )
    INDEX messages_session_ts (session_id, ts)
    INDEX messages_participants (from_id, to_id)
    VIRTUAL TABLE messages_fts USING fts5(body, content='messages', content_rowid='id')

The FTS5 table is kept in sync via triggers.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StoredMessage:
    """Row projection returned by reads. Immutable."""

    id: int
    ts: float
    session_id: str
    from_id: str
    to_id: str | None
    kind: str
    body: str
    corr_id: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  session_id TEXT NOT NULL,
  from_id TEXT NOT NULL,
  to_id TEXT,
  kind TEXT NOT NULL,
  body TEXT NOT NULL,
  corr_id TEXT
);

CREATE INDEX IF NOT EXISTS messages_session_ts ON messages (session_id, ts);
CREATE INDEX IF NOT EXISTS messages_participants ON messages (from_id, to_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  body,
  content='messages',
  content_rowid='id'
);

-- Keep FTS index in sync with the base table.
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts (rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts (messages_fts, rowid, body)
  VALUES ('delete', old.id, old.body);
END;
"""


class ChatStore:
    """Thread-safe SQLite wrapper around the chat transcript.

    Uses a single connection guarded by a lock. Chat volume is low
    (a busy 4-hour session has maybe a few hundred messages), so the
    simple locking model is fine — we're not trying to be a general
    durable queue, just a persistent searchable log.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        # Ensure parent directory exists. In tests we use :memory:
        # which bypasses this.
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; keeps semantics explicit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── writes ──────────────────────────────────────────────────

    def append(
        self,
        *,
        session_id: str,
        from_id: str,
        to_id: str | None,
        kind: str,
        body: str,
        corr_id: str | None = None,
        ts: float | None = None,
    ) -> int:
        """Append a message; return the assigned row id.

        Defaults to the current wall clock if ``ts`` is None so that
        callers can pass their own monotonic clock or a test stub.
        """
        row_ts = time.time() if ts is None else float(ts)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages "
                "(ts, session_id, from_id, to_id, kind, body, corr_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row_ts, session_id, from_id, to_id, kind, body, corr_id),
            )
            return int(cur.lastrowid or 0)

    # ── reads ───────────────────────────────────────────────────

    def thread(
        self,
        *,
        session_id: str,
        participant_id: str,
        limit: int = 500,
    ) -> list[StoredMessage]:
        """All messages visible to ``participant_id`` in ``session_id``.

        Visibility rules:
          - Player sees messages from/to themselves, plus broadcasts.
          - Keeper sees everything (no filter).

        Keeper identity is detected by ``keeper:`` prefix; any other
        identity is treated as a player.
        """
        with self._lock:
            if participant_id.startswith("keeper:"):
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? "
                    "ORDER BY ts ASC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND ("
                    "  from_id = ? OR to_id = ? "
                    "  OR kind = 'keeper_to_all' OR kind = 'artemis_to_all'"
                    ") ORDER BY ts ASC LIMIT ?",
                    (session_id, participant_id, participant_id, limit),
                ).fetchall()
        return [_row_to_msg(r) for r in rows]

    def search(
        self,
        *,
        query: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[StoredMessage]:
        """Free-text search across the transcript. Keeper-only surface.

        Caller is responsible for authorizing access — the store has
        no notion of who's asking. Typical usage is from the Keeper
        portal, which identity-gates at the HTTP layer.

        The user's raw query is sanitized for FTS5: each whitespace-
        separated token is wrapped in double quotes (implicit AND
        between tokens). This lets Keepers search for awkward tokens
        like ``mi-go`` without hitting FTS5 operator syntax.
        """
        match = _fts_match_expr(query)
        if not match:
            return []
        with self._lock:
            if session_id is None:
                sql = (
                    "SELECT m.* FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                    "WHERE messages_fts MATCH ? ORDER BY m.ts DESC LIMIT ?"
                )
                rows = self._conn.execute(sql, (match, limit)).fetchall()
            else:
                sql = (
                    "SELECT m.* FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                    "WHERE messages_fts MATCH ? AND m.session_id = ? "
                    "ORDER BY m.ts DESC LIMIT ?"
                )
                rows = self._conn.execute(sql, (match, session_id, limit)).fetchall()
        return [_row_to_msg(r) for r in rows]

    def recent(
        self,
        *,
        session_id: str,
        limit: int = 50,
    ) -> list[StoredMessage]:
        """Most-recent messages for a session, oldest-first in the slice.

        Intended for the mobile PWA (S1f) "catch up on last N messages"
        cold-start read.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM ("
                "  SELECT * FROM messages WHERE session_id = ? "
                "  ORDER BY ts DESC LIMIT ?"
                ") ORDER BY ts ASC",
                (session_id, limit),
            ).fetchall()
        return [_row_to_msg(r) for r in rows]

    def count(self, *, session_id: str | None = None) -> int:
        with self._lock:
            if session_id is None:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM messages"
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        return int(row["n"]) if row else 0


def _fts_match_expr(query: str) -> str:
    """Sanitize a free-text query into an FTS5 MATCH expression.

    FTS5 treats characters like ``-`` and ``:`` as operators, so raw
    user text can throw ``no such column: go`` or similar. We strip
    those operators from each token and wrap tokens in double quotes
    to force phrase-literal matching. Multiple tokens are ANDed (FTS5
    default).
    """
    raw = (query or "").strip()
    if not raw:
        return ""
    tokens: list[str] = []
    for piece in raw.split():
        cleaned = piece.replace('"', "").strip()
        # Keep alphanumerics and spaces only — everything else becomes
        # a space so FTS5 tokenizes naturally inside the quoted phrase.
        cleaned = "".join(c if c.isalnum() else " " for c in cleaned).strip()
        if not cleaned:
            continue
        tokens.append(f'"{cleaned}"')
    return " ".join(tokens)


def _row_to_msg(r: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=int(r["id"]),
        ts=float(r["ts"]),
        session_id=str(r["session_id"]),
        from_id=str(r["from_id"]),
        to_id=(None if r["to_id"] is None else str(r["to_id"])),
        kind=str(r["kind"]),
        body=str(r["body"]),
        corr_id=(None if r["corr_id"] is None else str(r["corr_id"])),
    )


def iter_sessions(store: ChatStore) -> Iterable[str]:
    """Distinct session_ids in the store, most-recent first."""
    with store._lock:
        rows = store._conn.execute(
            "SELECT session_id, MAX(ts) AS last "
            "FROM messages GROUP BY session_id ORDER BY last DESC"
        ).fetchall()
    return [str(r["session_id"]) for r in rows]
