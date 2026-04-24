# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 turn handler — the ``handle_turn`` entry point.

Parallel to :mod:`agi.primer.artemis.mode` in shape, using SIGMA-4's
persona modules (:mod:`.prompt`, :mod:`.trigger`) and the shared
validator in :mod:`agi.primer.artemis.validator`.

The validator itself is persona-agnostic — it enforces structural
rules (length, character_check for "as an AI" slips, safety flags,
forbidden n-grams) rather than persona-specific content. We pass
SIGMA-specific forbidden n-grams via the ``ValidatorConfig`` knob;
the check function stays shared.

.. note::

   The handle_turn function here is a deliberate near-duplicate of
   ``agi.primer.artemis.mode.handle_turn``, using SIGMA's prompt
   and trigger. A later sprint (see HALYARD_SPRINT_PLAN §post-sprint
   backlog) will extract the common scaffolding into a
   persona-generic base. For Sprint 2 we accept the duplication in
   exchange for not touching the stable ARTEMIS code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Shared data types from ARTEMIS — SessionState / SessionTurn / Bible
# are persona-agnostic and the two AIs can legitimately share them.
from agi.primer.artemis.context import (
    Bible,
    SessionState,
    SessionTurn,
    assemble_context,
)

# Shared validator — persona-specific rules go in the ValidatorConfig.
from agi.primer.artemis.validator import (
    DecisionProof,
    ValidatorConfig,
    check_reply,
)

from . import prompt, trigger

log = logging.getLogger("halyard.sigma4")

# Proof-chain directory. Distinct from ARTEMIS's so the two chains
# don't interleave in the archive.
_PROOFS_DIR = Path(
    os.environ.get("SIGMA4_PROOFS_DIR", "/archive/sigma4/proofs")
)

# Canonical silence sentinels. Note ``[INTERFACE_SILENT]`` is SIGMA's
# own sentinel, distinct from ARTEMIS's ``[INTERFACE_FLICKER]``.
_SILENT_SENTINEL = "[INTERFACE_SILENT]"
_SILENCE_SENTINEL = "[SILENCE]"


# ─────────────────────────────────────────────────────────────────
# Request / response types — identical shape to ARTEMIS's, so bots
# and handlers can trivially route between the two AIs.
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnRequest:
    """A ``heard`` event for SIGMA-4 (one speaker utterance)."""

    session_id: str
    turn_id: str
    speaker: str
    text: str
    ts: float
    partial: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def safety_flag(self) -> str | None:
        flag = self.meta.get("safety_flag")
        return flag if isinstance(flag, str) else None


@dataclass(frozen=True)
class TurnResponse:
    """A ``say`` event published back toward the room."""

    session_id: str
    turn_id: str
    text: str
    proof_hash: str
    latency_s: float
    expert: str
    approval: str = "auto"  # "auto" | "keeper_pending" | "keeper_approved"


# ─────────────────────────────────────────────────────────────────
# LLM caller protocol — injectable for tests.
# ─────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMCaller(Protocol):
    """Abstracted async LLM call. See the ARTEMIS LLMCaller for the
    contract — SIGMA-4 intentionally uses the same shape so the
    same vMOE instance can be targeted by either AI."""

    async def __call__(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]: ...


# ─────────────────────────────────────────────────────────────────
# Proof-chain persistence
# ─────────────────────────────────────────────────────────────────


def _chain_path(session_id: str) -> Path:
    return _PROOFS_DIR / f"{session_id}.chain"


def _append_proof(session_id: str, proof: DecisionProof) -> None:
    """Append a proof record to the session's chain file. Never raises."""
    try:
        p = _chain_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(proof.to_dict(), separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("sigma4 proof append failed for %s: %s", session_id, e)


# ─────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────


async def handle_turn(
    request: TurnRequest,
    *,
    state: SessionState,
    bible: Bible,
    llm: LLMCaller,
    validator_config: ValidatorConfig | None = None,
    keeper_approval_required: bool = False,
) -> TurnResponse | None:
    """Process one incoming turn for SIGMA-4.

    Return ``TurnResponse`` or ``None`` for silence.

    See :func:`agi.primer.artemis.mode.handle_turn` for the
    structural contract; the two handlers follow identical steps
    with SIGMA's persona swapped in.
    """
    # 1. Record the heard turn regardless of reply outcome.
    heard_turn = SessionTurn(
        ts=request.ts,
        turn_id=request.turn_id,
        speaker=request.speaker,
        text=request.text,
    )

    # 2. Partials: update log only, never generate.
    if request.partial:
        state.append_turn(heard_turn)
        return None

    # 3. Trigger policy.
    speak, _reason = trigger.should_speak(
        speaker=request.speaker,
        text=request.text,
        meta=request.meta,
        silenced=state.silenced,
        safety_flag=request.safety_flag,
    )
    if not speak:
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 4. Assemble context — reuses ARTEMIS's assembler because the
    # session-log and bible structures are shared types.
    bible_ctx, session_ctx = assemble_context(bible=bible, state=state)
    messages_obj = prompt.assemble(
        turn_text=request.text,
        speaker=request.speaker,
        bible_context=bible_ctx,
        session_summary=session_ctx,
    )

    # 5. Call the model.
    try:
        reply_text, expert_name, latency_s = await llm(
            messages_obj.system, messages_obj.messages
        )
    except Exception as e:  # noqa: BLE001
        log.warning("sigma4 llm call failed: %s", e)
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 6. Sentinel-silence handling before validation.
    stripped = (reply_text or "").strip()
    if stripped == _SILENT_SENTINEL:
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 7. X-card / pause: substitute silence sentinel before validating.
    if request.safety_flag in {"x-card", "pause"}:
        reply_text = _SILENCE_SENTINEL

    # 8. Validate; always emit the DecisionProof whether pass or fail.
    passed, proof = check_reply(
        reply_text,
        turn_id=request.turn_id,
        session_id=request.session_id,
        safety_flag=request.safety_flag,
        prev_hash=state.last_proof_hash,
        config=validator_config,
    )
    _append_proof(request.session_id, proof)
    state.last_proof_hash = proof.self_hash

    # 9. On fail: silence, audit trail is already in the chain.
    if not passed:
        heard_turn.artemis_reply = None  # shared field name; tracks either AI
        heard_turn.proof_hash = proof.self_hash
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 10. Sentinel rendering — substitute in-fiction silence lines.
    posted_text = reply_text
    if posted_text.strip() == _SILENT_SENTINEL:
        posted_text = prompt.silence_line("silent")
    elif posted_text.strip() == _SILENCE_SENTINEL:
        posted_text = prompt.silence_line("silence")

    # 11. Persist + return response.
    heard_turn.artemis_reply = posted_text
    heard_turn.proof_hash = proof.self_hash
    state.append_turn(heard_turn)
    state.persist_turn(heard_turn)

    return TurnResponse(
        session_id=request.session_id,
        turn_id=request.turn_id,
        text=posted_text,
        proof_hash=proof.self_hash,
        latency_s=round(latency_s, 3),
        expert=expert_name,
        approval="keeper_pending" if keeper_approval_required else "auto",
    )
