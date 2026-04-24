# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Pending-AI-reply approval queue for halyard-keeper-backend.

When the AI NPCs run with ``keeper_approval_required=True``
(the Sprint-6 default), every proposed reply arrives on NATS as a
``{artemis,sigma4}.say`` payload with ``approval='keeper_pending'``.
Instead of being posted to Zoom/LiveKit immediately, the payload
sits in this queue. The Keeper sees pending items in their console
and chooses ✓ release / ✗ drop / ✏︎ edit.

On ✓ release, the keeper backend publishes the (possibly edited)
reply back on the same session's ``<ai>.say`` subject with
``approval='keeper_approved'``. The ARTEMIS / SIGMA agent
processes — or the web client directly, in the Sprint-6 dev
build — pick that up and ship it to the room DataChannel.

Scope for Sprint 6 v0:
  - In-memory queue, one per session.
  - Deterministic approval IDs (sha of turn_id + timestamp).
  - Subscribers (WS listeners) get a full snapshot on connect and
    incremental pushes thereafter.

Not in this sprint: NATS subscription wiring (requires the AIs'
NATS handlers to tag approval='keeper_pending' consistently), and
``✏︎ edit`` beyond a text replacement on release.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

log = logging.getLogger("halyard.keeper.approvals")


class AiKind(Enum):
    ARTEMIS = "artemis"
    SIGMA4 = "sigma4"


class ApprovalState(Enum):
    PENDING = "pending"
    APPROVED = "approved"  # released (possibly edited)
    REJECTED = "rejected"  # dropped
    EXPIRED = "expired"    # not acted on within the window


@dataclass
class PendingApproval:
    """One proposed AI reply awaiting Keeper action.

    Fields mirror the ARTEMIS / SIGMA ``TurnResponse`` shape plus
    the approval-specific fields.
    """

    approval_id: str
    session_id: str
    ai: AiKind
    turn_id: str
    proposed_text: str
    proof_hash: str
    latency_s: float
    expert: str
    received_at: float
    state: ApprovalState = ApprovalState.PENDING
    resolved_at: float | None = None
    final_text: str | None = None  # populated on APPROVED (possibly edited)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ai"] = self.ai.value
        d["state"] = self.state.value
        return d


def compute_approval_id(
    *, session_id: str, ai: AiKind, turn_id: str, received_at: float
) -> str:
    """Stable id derived from the tuple that identifies a turn.

    Determinism matters so a duplicated publish doesn't spawn two
    queue entries. We include received_at so that if the same
    turn_id somehow comes back later it's treated as a new item.
    """
    h = hashlib.sha256()
    h.update(session_id.encode("utf-8"))
    h.update(b"/")
    h.update(ai.value.encode("utf-8"))
    h.update(b"/")
    h.update(turn_id.encode("utf-8"))
    h.update(b"/")
    h.update(f"{received_at:.6f}".encode("utf-8"))
    return "ap_" + h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────
# Queue
# ─────────────────────────────────────────────────────────────────


class ApprovalQueue:
    """Per-process pending-approval store.

    One instance per keeper backend process. Session-scoped
    internally so the Keeper view only sees the session they're
    watching.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, PendingApproval] = {}
        self._by_session: dict[str, list[str]] = {}
        self._listeners: dict[str, set[asyncio.Queue[dict]]] = {}
        self._lock = asyncio.Lock()

    # ── ingest ──

    async def enqueue(
        self,
        *,
        session_id: str,
        ai: AiKind,
        turn_id: str,
        proposed_text: str,
        proof_hash: str = "",
        latency_s: float = 0.0,
        expert: str = "unknown",
    ) -> PendingApproval:
        received_at = time.time()
        approval_id = compute_approval_id(
            session_id=session_id, ai=ai, turn_id=turn_id, received_at=received_at,
        )
        item = PendingApproval(
            approval_id=approval_id,
            session_id=session_id,
            ai=ai,
            turn_id=turn_id,
            proposed_text=proposed_text,
            proof_hash=proof_hash,
            latency_s=latency_s,
            expert=expert,
            received_at=received_at,
        )
        async with self._lock:
            self._by_id[approval_id] = item
            self._by_session.setdefault(session_id, []).append(approval_id)
        await self._fan_out(
            session_id,
            {"kind": "approval.new", "approval": item.to_dict()},
        )
        return item

    # ── resolve ──

    async def approve(
        self, approval_id: str, *, edited_text: str | None = None
    ) -> PendingApproval:
        async with self._lock:
            item = self._by_id.get(approval_id)
            if item is None:
                raise KeyError(approval_id)
            if item.state is not ApprovalState.PENDING:
                raise ValueError(
                    f"approval {approval_id} already {item.state.value}"
                )
            item.state = ApprovalState.APPROVED
            item.resolved_at = time.time()
            item.final_text = (
                edited_text if edited_text is not None else item.proposed_text
            )
        await self._fan_out(
            item.session_id,
            {"kind": "approval.resolved", "approval": item.to_dict()},
        )
        return item

    async def reject(self, approval_id: str) -> PendingApproval:
        async with self._lock:
            item = self._by_id.get(approval_id)
            if item is None:
                raise KeyError(approval_id)
            if item.state is not ApprovalState.PENDING:
                raise ValueError(
                    f"approval {approval_id} already {item.state.value}"
                )
            item.state = ApprovalState.REJECTED
            item.resolved_at = time.time()
        await self._fan_out(
            item.session_id,
            {"kind": "approval.resolved", "approval": item.to_dict()},
        )
        return item

    # ── read ──

    async def pending_for(self, session_id: str) -> list[PendingApproval]:
        ids = self._by_session.get(session_id, [])
        return [
            self._by_id[i]
            for i in ids
            if self._by_id[i].state is ApprovalState.PENDING
        ]

    async def snapshot(self, session_id: str) -> list[dict[str, Any]]:
        """Last 50 items (pending + resolved) for console backfill."""
        ids = self._by_session.get(session_id, [])
        recent = ids[-50:]
        return [self._by_id[i].to_dict() for i in recent]

    # ── listener fan-out ──

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue()
        async with self._lock:
            self._listeners.setdefault(session_id, set()).add(q)
        return q

    async def unsubscribe(
        self, session_id: str, q: asyncio.Queue[dict]
    ) -> None:
        async with self._lock:
            listeners = self._listeners.get(session_id)
            if listeners is None:
                return
            listeners.discard(q)
            if not listeners:
                self._listeners.pop(session_id, None)

    async def _fan_out(
        self, session_id: str, event: dict[str, Any]
    ) -> None:
        async with self._lock:
            listeners = list(self._listeners.get(session_id, ()))
        for q in listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow listener — drop. The Keeper console refetches
                # snapshot on reconnect; losing an incremental push
                # does not lose correctness.
                log.warning("approval fan-out dropped for slow listener")
