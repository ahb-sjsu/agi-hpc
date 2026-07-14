# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Audit + conversation logging for Erebus's Discord faculty.

Two append-only JSONL sinks under ``/archive/erebus/``:

- ``discord_audit.jsonl`` — every inbound message and every outbound
  decision (posted / suppressed / drafted) with the gate verdict. The
  record of what Erebus heard and what it chose to say.
- ``discord_conversations.jsonl`` — the accepted exchanges, the learning
  corpus that later feeds episodic memory / dreaming.

Writes never raise into the bot loop (best-effort).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("discord.audit")

DEFAULT_AUDIT = Path("/archive/erebus/discord_audit.jsonl")
DEFAULT_CONVOS = Path("/archive/erebus/discord_conversations.jsonl")


class AuditLog:
    def __init__(self, audit_path: Path = DEFAULT_AUDIT,
                 convo_path: Path = DEFAULT_CONVOS, now=time.time) -> None:
        self.audit_path = Path(audit_path)
        self.convo_path = Path(convo_path)
        self._now = now

    def _append(self, path: Path, record: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001 - logging must not crash the bot
            log.warning("audit append failed: %s", e)

    def inbound(self, msg, note: str = "") -> None:
        self._append(self.audit_path, {
            "ts": round(self._now(), 3), "dir": "in",
            "author_id": msg.author_id, "author": msg.author_name,
            "channel_id": msg.channel_id, "text": msg.text[:1000], "note": note,
        })

    def outbound(self, msg, reply: str, verdict, posted: bool) -> None:
        self._append(self.audit_path, {
            "ts": round(self._now(), 3), "dir": "out",
            "channel_id": msg.channel_id, "reply": reply[:1000],
            "gate_allowed": bool(getattr(verdict, "allowed", False)),
            "gate_reason": getattr(verdict, "reason", ""),
            "gate_flags": list(getattr(verdict, "flags", []) or []),
            "posted": posted,
        })

    def error(self, msg, err: str) -> None:
        self._append(self.audit_path, {
            "ts": round(self._now(), 3), "dir": "error",
            "channel_id": msg.channel_id, "error": err[:300],
        })

    def conversation(self, conv_id: str, msg, reply: str) -> None:
        self._append(self.convo_path, {
            "ts": round(self._now(), 3), "conversation_id": conv_id,
            "channel_id": msg.channel_id,
            "author_id": msg.author_id, "author": msg.author_name,
            "user": msg.text[:2000], "erebus": reply[:2000],
        })
