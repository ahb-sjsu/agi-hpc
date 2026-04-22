# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Session log + campaign-bible context assembly for ARTEMIS.

The session log is a JSONL file under ``/archive/artemis/sessions/``,
one line per turn. The bible is a curated set of text chunks tagged
``artemis_known``, ``artemis_unknown``, or ``artemis_forbidden``;
only ``artemis_known`` chunks are ever injected into the prompt.
The other tags exist so the validator can reject replies that leak
them (see ``validator.secret_leak_check``).

Phase 1 is intentionally simple:
- Session log: tail-read + naive concatenation into a ~600-token summary.
- Bible: static list of chunks loaded once per session from a JSON file
  at the path given by ``ARTEMIS_BIBLE_PATH`` (env var).

Phase 2 may swap this for L2 vector retrieval; the public interface
(:func:`assemble_context`) stays the same.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Approximate characters-per-token for the budget heuristic.
_CHARS_PER_TOKEN = 4

# Default path for session logs.
_SESSIONS_DIR = Path(
    os.environ.get("ARTEMIS_SESSIONS_DIR", "/archive/artemis/sessions")
)

# Default path for the bible chunks file.
_BIBLE_PATH = Path(
    os.environ.get("ARTEMIS_BIBLE_PATH", "/archive/artemis/bible/halyard_bible.json")
)

# Token budgets.
_BIBLE_BUDGET_TOKENS = 8_000  # headroom for the cached bible block
_SESSION_BUDGET_TOKENS = 600  # for the recent-turns summary


@dataclass
class BibleChunk:
    """One chunk of the campaign bible."""

    id: str
    tag: str  # "artemis_known" | "artemis_unknown" | "artemis_forbidden"
    title: str
    text: str

    def token_estimate(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)


@dataclass
class Bible:
    """Loaded campaign bible, partitioned by tag."""

    known: tuple[BibleChunk, ...] = field(default_factory=tuple)
    unknown: tuple[BibleChunk, ...] = field(default_factory=tuple)
    forbidden_phrases: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path | None = None) -> "Bible":
        target = path or _BIBLE_PATH
        if not target.exists():
            log.warning("artemis bible missing at %s; returning empty", target)
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("artemis bible at %s not parseable: %s", target, e)
            return cls()
        known: list[BibleChunk] = []
        unknown: list[BibleChunk] = []
        forbidden_phrases: list[str] = []
        for entry in data.get("chunks", []) or []:
            tag = entry.get("tag", "")
            chunk = BibleChunk(
                id=entry.get("id", ""),
                tag=tag,
                title=entry.get("title", ""),
                text=entry.get("text", ""),
            )
            if tag == "artemis_known":
                known.append(chunk)
            elif tag == "artemis_unknown":
                unknown.append(chunk)
        for phrase in data.get("forbidden_phrases", []) or []:
            if isinstance(phrase, str) and phrase.strip():
                forbidden_phrases.append(phrase.strip())
        return cls(
            known=tuple(known),
            unknown=tuple(unknown),
            forbidden_phrases=tuple(forbidden_phrases),
        )

    def known_texts(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.known)

    def unknown_texts(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.unknown)


@dataclass
class SessionTurn:
    """One persisted turn (heard + optional say)."""

    ts: float
    turn_id: str
    speaker: str
    text: str
    artemis_reply: str | None = None
    proof_hash: str | None = None


@dataclass
class SessionState:
    """Per-session working state held in-process by the handler."""

    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    last_proof_hash: str = "genesis"
    silenced: bool = False

    def append_turn(self, turn: SessionTurn) -> None:
        self.turns.append(turn)

    def path(self) -> Path:
        return _SESSIONS_DIR / f"{self.session_id}.jsonl"

    def load(self) -> None:
        """Load prior turns from disk. No-op if the file doesn't exist."""
        p = self.path()
        if not p.exists():
            return
        loaded: list[SessionTurn] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            loaded.append(
                SessionTurn(
                    ts=float(d.get("ts", 0)),
                    turn_id=str(d.get("turn_id", "")),
                    speaker=str(d.get("speaker", "")),
                    text=str(d.get("text", "")),
                    artemis_reply=d.get("artemis_reply"),
                    proof_hash=d.get("proof_hash"),
                )
            )
            if d.get("proof_hash"):
                self.last_proof_hash = d["proof_hash"]
        self.turns = loaded

    def persist_turn(self, turn: SessionTurn) -> None:
        """Append one turn to the session log on disk."""
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": turn.ts,
            "turn_id": turn.turn_id,
            "speaker": turn.speaker,
            "text": turn.text,
            "artemis_reply": turn.artemis_reply,
            "proof_hash": turn.proof_hash,
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


# ─────────────────────────────────────────────────────────────────
# Context assembly
# ─────────────────────────────────────────────────────────────────


def _budget_fit(chunks: list[str], budget_tokens: int) -> str:
    """Concatenate chunks up to a token budget, oldest-first."""
    out: list[str] = []
    spent = 0
    for chunk in chunks:
        cost = max(1, len(chunk) // _CHARS_PER_TOKEN)
        if spent + cost > budget_tokens:
            break
        out.append(chunk)
        spent += cost
    return "\n\n".join(out)


def session_summary(state: SessionState) -> str:
    """Reduce the last N turns to a short summary under token budget.

    Phase 1: simple last-N concatenation with a leading summary-of-summary
    line. No LLM-in-the-loop summarization; keep it deterministic and
    debuggable. Swap for a summarizer call once logs exceed 200 turns.
    """
    if not state.turns:
        return ""
    recent = state.turns[-20:]
    lines: list[str] = []
    if len(state.turns) > 20:
        lines.append(f"(Earlier this session: {len(state.turns) - 20} turns elided.)")
    for t in recent:
        lines.append(f"- {t.speaker}: {t.text}")
        if t.artemis_reply:
            lines.append(f"  ARTEMIS → {t.artemis_reply}")
    text = "\n".join(lines)
    # If over budget, drop from the top.
    while _tokens(text) > _SESSION_BUDGET_TOKENS and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


def _tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def assemble_context(
    *,
    bible: Bible,
    state: SessionState,
    bible_budget_tokens: int = _BIBLE_BUDGET_TOKENS,
) -> tuple[str, str]:
    """Return ``(bible_context, session_summary)`` for a turn.

    Bible context is deterministic and cache-friendly (stable across a
    session). Session summary is fresh each turn.
    """
    bible_ctx = _budget_fit(
        [f"## {c.title}\n\n{c.text}" for c in bible.known],
        bible_budget_tokens,
    )
    return bible_ctx, session_summary(state)
