# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS event handler — the ``handle_turn`` entry point.

One call per incoming turn. Decides whether to reply, generates a
candidate, validates it, emits a DecisionProof, persists the turn,
and returns the reply (or None for silence).

Pure async function; no NATS, no Zoom. Phase 2 wraps this in a NATS
subscribe loop, Phase 3 hosts the bot container.

Public contract:
  - Input:  :class:`TurnRequest` (the Zoom-bot's ``heard`` event)
  - Output: :class:`TurnResponse` or ``None``
  - Side effects: session log write, DecisionProof chain append
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import prompt, trigger
from .context import Bible, SessionState, SessionTurn, assemble_context
from .validator import DecisionProof, ValidatorConfig, check_reply

log = logging.getLogger("primer.artemis")

# Lifecycle stream reuse — same pattern as the Primer proper.
try:
    from agi.common.structured_log import LifecycleLogger

    _lifecycle = LifecycleLogger("artemis")
except Exception:
    _lifecycle = None


# Directory for DecisionProof chains.
_PROOFS_DIR = Path(
    os.environ.get("ARTEMIS_PROOFS_DIR", "/archive/artemis/proofs")
)

# Canonical silence sentinels.
_FLICKER_SENTINEL = "[INTERFACE_FLICKER]"
_SILENCE_SENTINEL = "[SILENCE]"


# ─────────────────────────────────────────────────────────────────
# Request / response types
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnRequest:
    """A ``heard`` event from the Zoom bot (one speaker utterance)."""

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
    """A ``say`` event published back to the bot for posting to Zoom."""

    session_id: str
    turn_id: str
    text: str
    proof_hash: str
    latency_s: float
    expert: str
    approval: str = "auto"  # "auto" | "keeper_pending" | "keeper_approved"


# ─────────────────────────────────────────────────────────────────
# LLM caller protocol — injectable for tests
# ─────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMCaller(Protocol):
    """Abstracted async LLM call.

    Production implementation wraps a vMOE instance; tests inject a
    simple callable. Contract: given ``(system, messages)``, return
    ``(text, expert_name, latency_s)`` or raise on catastrophic error.
    Timeout / soft failures should return an empty text and a sentinel
    expert name ``"timeout"``.
    """

    async def __call__(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]: ...


# ─────────────────────────────────────────────────────────────────
# Proof chain persistence
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
        log.warning("proof append failed for %s: %s", session_id, e)


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
    """Process one incoming turn. Return a response or ``None`` for silence.

    Arguments:
      request: the incoming heard-event.
      state: the live :class:`SessionState` for this session_id.
      bible: the loaded campaign bible.
      llm: async callable that invokes the underlying model pool.
      validator_config: override of the default validator config
        (test-inject, or session-scoped rules).
      keeper_approval_required: if True, the returned response carries
        ``approval='keeper_pending'`` and the bot should surface it to
        the Keeper before posting to chat. The DecisionProof is still
        emitted so declined replies leave an audit trail.

    Side effects:
      - The turn (and reply, if any) is persisted to the session log.
      - A DecisionProof is appended to the session chain.

    Partial ASR events (``request.partial == True``) are recorded in
    state but never trigger a reply.
    """
    # 1. Record the heard turn regardless of reply.
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
    speak, reason = trigger.should_speak(
        speaker=request.speaker,
        text=request.text,
        meta=request.meta,
        silenced=state.silenced,
        safety_flag=request.safety_flag,
    )
    _emit("turn_received", request=request, speak=speak, reason=reason)
    if not speak:
        # Persist the turn; we just don't respond to it.
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 4. Assemble context.
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
        log.warning("llm call failed: %s", e)
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        _emit("llm_error", error=str(e))
        return None
    _emit(
        "ensemble_complete",
        expert=expert_name,
        latency_s=round(latency_s, 3),
        reply_len=len(reply_text or ""),
    )

    # 6. Handle sentinel replies from the model BEFORE validating.
    # The model emitted [INTERFACE_FLICKER] on purpose — that is its
    # way of saying "silent this turn." Honor it; don't validate.
    stripped = (reply_text or "").strip()
    if stripped == _FLICKER_SENTINEL:
        _emit("model_silence", sentinel="flicker")
        heard_turn.artemis_reply = None
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # X-card and related safety flags: replace any reply with the
    # mandated silence line BEFORE validation, so we're validating the
    # actual thing we'd post. This also means the validator's
    # safety_tool_check will see [SILENCE] and pass it.
    if request.safety_flag in {"x-card", "pause"}:
        reply_text = _SILENCE_SENTINEL

    # 7. Validate. Always emit the DecisionProof, whether pass or fail.
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
    _emit(
        "validator_result",
        passed=passed,
        proof_hash=proof.self_hash,
        verdict=proof.verdict,
        checks=[{"name": cr.name, "passed": cr.passed} for cr in proof.check_results],
    )

    # 8. On fail: silence, but only after the proof is logged.
    if not passed:
        heard_turn.artemis_reply = None
        heard_turn.proof_hash = proof.self_hash
        state.append_turn(heard_turn)
        state.persist_turn(heard_turn)
        return None

    # 9. Substitute the interface-flicker sentinel if the model tried
    # to use it but landed through validation (paranoid belt-and-braces).
    posted_text = reply_text
    if posted_text.strip() == _FLICKER_SENTINEL:
        posted_text = prompt.silence_line("flicker")
        _emit("sentinel_rendered", kind="flicker")
    elif posted_text.strip() == _SILENCE_SENTINEL:
        posted_text = prompt.silence_line("silence")
        _emit("sentinel_rendered", kind="silence")

    # 10. Persist with reply + return the response.
    heard_turn.artemis_reply = posted_text
    heard_turn.proof_hash = proof.self_hash
    state.append_turn(heard_turn)
    state.persist_turn(heard_turn)

    response = TurnResponse(
        session_id=request.session_id,
        turn_id=request.turn_id,
        text=posted_text,
        proof_hash=proof.self_hash,
        latency_s=round(latency_s, 3),
        expert=expert_name,
        approval="keeper_pending" if keeper_approval_required else "auto",
    )
    _emit(
        "posted" if not keeper_approval_required else "keeper_gate",
        reply=posted_text,
    )
    return response


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _emit(event: str, **fields: Any) -> None:
    """Structured-log emit that never raises."""
    if _lifecycle is None:
        return
    try:
        # LifecycleLogger signature is (event_name, **fields) per common pkg.
        _lifecycle.emit(event, **fields)
    except Exception as e:  # noqa: BLE001
        log.debug("lifecycle emit failed: %s", e)


# ─────────────────────────────────────────────────────────────────
# vMOE adapter — production LLMCaller
# ─────────────────────────────────────────────────────────────────


def vmoe_caller(
    moe: Any,
    *,
    experts: tuple[str, ...] = ("qwen3", "kimi"),
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> LLMCaller:
    """Return an :class:`LLMCaller` backed by a vMOE instance.

    Cascades across ``experts`` in order; first non-error response is
    returned. Short leash by default (300 tokens, temperature 0.7) —
    ARTEMIS is a handheld, not a chat partner.
    """

    async def _call(
        system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]:
        full = [{"role": "system", "content": system}] + messages
        last_err = ""
        for expert in experts:
            resp = await moe.call(
                expert,
                full,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if resp.ok and resp.content.strip():
                return resp.content, resp.expert, resp.latency_s
            last_err = resp.error or "empty_content"
            log.info("artemis: %s failed (%s); cascading", expert, last_err)
        raise RuntimeError(f"all experts failed: {last_err}")

    return _call
