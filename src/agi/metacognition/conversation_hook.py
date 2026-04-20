# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 4 — conversation-finalization hook.

Buffers chat turns in memory, keyed by ``conversation_id``. When a
conversation is finalized (explicit close signal OR inactivity timeout
swept lazily on each new turn), runs the detector + aggregator
pipeline so at-most-one event is emitted per conversation.

Spec: ``docs/KNOWLEDGE_GAP_MAPPING_v1_spec.md`` §7 Phase 4 — "Hook the
detector into the conversation finalization path in
``scripts/telemetry_server.py`` (triggers on session close or
inactivity timeout), emitting at most one event per conversation."

Runs entirely in the telemetry-server process. The buffer is
in-memory only by design — a crash drops in-flight conversations,
which is fine: dissatisfaction gaps are observability signal, not
load-bearing state. If durability ever matters, the event sidecar
(``dissatisfaction_events.jsonl``) is the persistence layer.

Finalization spawns a daemon thread for the detector call so a
slow LLM doesn't stall the chat handler that triggered the sweep.
Tests use ``finalize_conversation_sync`` directly for determinism.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dissatisfaction import classify_conversation
from .gap_aggregator import aggregate_event

log = logging.getLogger("metacognition.conversation_hook")

# ── constants ────────────────────────────────────────────────────

DEFAULT_IDLE_THRESHOLD_S: int = int(os.environ.get("EREBUS_CONV_IDLE_S", "600"))
"""How long a conversation may sit without new turns before being
auto-finalized. Default 10 minutes — long enough that a human pause is
not mistaken for end-of-conversation, short enough that the buffer
doesn't accumulate stale entries forever."""

DEFAULT_EGO_MODEL: str = os.environ.get("EREBUS_EGO_MODEL", "kimi")
"""The ego backend that served the conversation, recorded on the
detector signal for observability. Matches the current chat cascade
default; switch when the multi-tier ego lands."""


# ── in-memory buffer ────────────────────────────────────────────


@dataclass
class _Conversation:
    conversation_id: str
    turns: list[dict] = field(default_factory=list)
    last_ts: float = 0.0
    finalized: bool = False


class ConversationBuffer:
    """Thread-safe per-conversation turn buffer.

    One process-wide instance is used by ``default_buffer()``; tests
    construct their own to avoid cross-test pollution.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, _Conversation] = {}
        self._lock = threading.Lock()

    def record_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        ts: float | None = None,
    ) -> None:
        """Append one turn. Silently ignores turns for a conversation
        that has already been finalized — a finalized conversation is
        done, and accepting more turns would let a reopened session
        emit a second event for the same logical dialogue."""
        if not conversation_id:
            log.warning("record_turn: missing conversation_id; dropping")
            return
        ts_now = float(ts if ts is not None else time.time())
        with self._lock:
            buf = self._buffers.setdefault(
                conversation_id, _Conversation(conversation_id)
            )
            if buf.finalized:
                log.warning(
                    "record_turn: conversation %s already finalized; ignoring",
                    conversation_id,
                )
                return
            buf.turns.append({"role": role, "content": content, "ts": ts_now})
            buf.last_ts = ts_now

    def get(self, conversation_id: str) -> _Conversation | None:
        with self._lock:
            return self._buffers.get(conversation_id)

    def idle_conversations(
        self, now: float, idle_threshold_s: float
    ) -> list[_Conversation]:
        """Return every non-finalized conversation whose last turn is at
        least ``idle_threshold_s`` old. Snapshot under lock; returned
        list is stable even if more turns arrive during finalization."""
        with self._lock:
            return [
                c
                for c in self._buffers.values()
                if not c.finalized and c.turns and (now - c.last_ts) >= idle_threshold_s
            ]

    def mark_finalized(self, conversation_id: str) -> None:
        with self._lock:
            buf = self._buffers.get(conversation_id)
            if buf is not None:
                buf.finalized = True

    def drop(self, conversation_id: str) -> None:
        with self._lock:
            self._buffers.pop(conversation_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffers)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for b in self._buffers.values() if not b.finalized)


_default_buffer: ConversationBuffer | None = None
_default_lock = threading.Lock()


def default_buffer() -> ConversationBuffer:
    """Process-wide singleton buffer. Lazy so import is cheap."""
    global _default_buffer
    if _default_buffer is None:
        with _default_lock:
            if _default_buffer is None:
                _default_buffer = ConversationBuffer()
    return _default_buffer


# ── public API ──────────────────────────────────────────────────


def record_turn(
    conversation_id: str,
    role: str,
    content: str,
    *,
    ts: float | None = None,
    buffer: ConversationBuffer | None = None,
) -> None:
    """Record one chat turn in the buffer. Called by the chat handler
    on every user message and every assistant response."""
    buf = buffer if buffer is not None else default_buffer()
    buf.record_turn(conversation_id, role, content, ts=ts)


def finalize_conversation_sync(
    conversation_id: str,
    *,
    ego_model: str = DEFAULT_EGO_MODEL,
    buffer: ConversationBuffer | None = None,
    graph_path: Path | None = None,
    events_path: Path | None = None,
    llm_call: Callable[..., str] | None = None,
    detector_model: str | None = None,
) -> str | None:
    """Synchronous finalize: detector → aggregator → maybe emit.

    Returns the UKG node id on a successful emit, None otherwise.
    Marks the buffer as finalized regardless of outcome — a second
    finalize attempt is a cheap no-op.
    """
    buf = buffer if buffer is not None else default_buffer()
    conv = buf.get(conversation_id)
    if conv is None:
        log.debug("finalize_sync: no buffer for %s", conversation_id)
        return None
    if conv.finalized:
        log.debug("finalize_sync: %s already finalized", conversation_id)
        return None
    if not conv.turns:
        buf.mark_finalized(conversation_id)
        return None

    # Snapshot turns outside the lock to keep the detector call
    # non-blocking for any concurrent record_turn (which would be
    # ignored anyway once we mark_finalized below).
    turns_snapshot = list(conv.turns)
    buf.mark_finalized(conversation_id)

    try:
        signal = classify_conversation(
            conversation_id=conversation_id,
            turns=turns_snapshot,
            ego_model=ego_model,
            llm_call=llm_call,
            detector_model=detector_model,
        )
    except Exception as e:  # noqa: BLE001 — never let detector crash the sweep
        log.warning(
            "finalize_sync: detector failed conv=%s err=%r",
            conversation_id,
            str(e)[:200],
        )
        return None

    try:
        node_id = aggregate_event(
            signal,
            conversation_id=conversation_id,
            graph_path=graph_path,
            events_path=events_path,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "finalize_sync: aggregator failed conv=%s err=%r",
            conversation_id,
            str(e)[:200],
        )
        return None

    if node_id:
        log.info(
            "conversation_finalized: conv=%s emitted node=%s",
            conversation_id,
            node_id,
        )
    return node_id


def finalize_conversation(
    conversation_id: str,
    *,
    ego_model: str = DEFAULT_EGO_MODEL,
    threaded: bool = True,
    **kwargs,
) -> str | None:
    """Finalize a conversation. Defaults to a daemon thread so a slow
    detector LLM call doesn't stall the chat handler.

    Returns the node_id only when ``threaded=False``. Threaded mode
    returns None immediately (the actual result is logged and emitted
    to the events sidecar + UKG).
    """
    if not threaded:
        return finalize_conversation_sync(
            conversation_id, ego_model=ego_model, **kwargs
        )
    t = threading.Thread(
        target=finalize_conversation_sync,
        args=(conversation_id,),
        kwargs={"ego_model": ego_model, **kwargs},
        daemon=True,
        name=f"gap-finalize-{conversation_id[:12]}",
    )
    t.start()
    return None


def sweep_idle(
    *,
    ego_model: str = DEFAULT_EGO_MODEL,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    now: float | None = None,
    buffer: ConversationBuffer | None = None,
    threaded: bool = True,
    **kwargs,
) -> list[str]:
    """Finalize every conversation whose last turn is >= idle_threshold_s old.

    Called lazily from the chat handler on each incoming turn — a
    bounded sweep per request is cheaper than a dedicated background
    thread for v1 volumes and keeps startup/shutdown simple. Returns
    the conversation ids for which finalize was triggered.
    """
    buf = buffer if buffer is not None else default_buffer()
    ts_now = float(now if now is not None else time.time())
    idle = buf.idle_conversations(ts_now, idle_threshold_s)
    triggered: list[str] = []
    for conv in idle:
        finalize_conversation(
            conv.conversation_id,
            ego_model=ego_model,
            buffer=buf,
            threaded=threaded,
            **kwargs,
        )
        triggered.append(conv.conversation_id)
    return triggered
