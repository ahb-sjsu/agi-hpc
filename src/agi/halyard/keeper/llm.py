# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Real AI-turn path — runs through the full AGI-HPC pipeline.

For each incoming HTTP turn the keeper backend:

1. Builds a :class:`agi.primer.artemis.mode.TurnRequest`.
2. Looks up (or creates) a per-``(session_id, which)``
   :class:`SessionState` — the same one
   :func:`handle_turn` uses for its episodic memory.
3. Constructs a small :class:`Bible` with persona-appropriate
   ``known`` chunks and a hard list of ``forbidden_phrases`` so
   the validator's forbidden-n-gram check enforces the spoiler
   boundary at the last mile.
4. Hands the request to the same ``handle_turn`` used by the
   ARTEMIS NATS handler (for SIGMA-4 it's the parallel
   :mod:`agi.halyard.sigma4.mode`).
5. Gets back a ``TurnResponse`` including the
   :class:`DecisionProof` hash emitted for that turn. The proof
   chain is appended on disk under
   ``/archive/artemis/proofs/`` or
   ``/archive/sigma4/proofs/`` — the same locations the NATS
   consumer would write to when that service lands.

Fallback: if the NRP token is unset or the upstream model call
fails, ``handle_turn`` returns ``None`` (validator-silence or
LLM-error), and the route falls through to the keyword stub so
the web client always gets an in-persona line.

**This is the full architecture.** The only thing missing
vs. the ARTEMIS.md plan-of-record is the NATS pub/sub
transport — which can be layered on (publish the same
``TurnRequest`` we already build onto ``agi.rh.artemis.heard``,
swap the in-process call for a subject-round-trip) without
touching this module's contract.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

# Shared handle_turn + types from the Primer's ARTEMIS package.
from agi.primer.artemis.context import Bible, BibleChunk, SessionState
from agi.primer.artemis.mode import (
    TurnRequest as ArtemisTurnRequest,
)
from agi.primer.artemis.mode import (
    TurnResponse as ArtemisTurnResponse,
)
from agi.primer.artemis.mode import handle_turn as artemis_handle_turn

# SIGMA-4 is a parallel implementation in the Halyard package.
from agi.halyard.sigma4.mode import (
    TurnRequest as Sigma4TurnRequest,
)
from agi.halyard.sigma4.mode import handle_turn as sigma4_handle_turn

from .ai_stub import stub_reply
from .campaign_facts import (
    ARTEMIS_ONLY,
    SHARED_CONTEXT,
    SIGMA_ONLY,
    system_prompt_for,
)

log = logging.getLogger("halyard.keeper.llm")


# ─────────────────────────────────────────────────────────────────
# NRP config + LLMCaller
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfig:
    """Minimal NRP client config. No creds → caller is unavailable."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 220
    timeout_s: float = 12.0

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        key = os.environ.get("NRP_LLM_TOKEN", "").strip()
        if not key:
            return None
        return cls(
            base_url=os.environ.get(
                "NRP_LLM_URL", "https://ellm.nrp-nautilus.io/v1"
            ).rstrip("/"),
            api_key=key,
            model=os.environ.get("HALYARD_LLM_MODEL", "qwen3-32b"),
            temperature=float(os.environ.get("HALYARD_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("HALYARD_LLM_MAX_TOKENS", "220")),
            timeout_s=float(os.environ.get("HALYARD_LLM_TIMEOUT_S", "12.0")),
        )


class NRPLLMCaller:
    """Implements the ``LLMCaller`` protocol used by handle_turn.

    Receives ``(system, messages)`` — the exact shape the
    persona-specific prompt module already assembled — and posts
    to NRP's OpenAI-compatible endpoint. Returns
    ``(text, expert_name, latency_s)``.

    The expert name is the effective model id (NRP routes via
    vMOE internally when asked; if we grow into vMOE on our side,
    we swap the caller for one that routes across experts and
    reports the winning one).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._cfg = config

    async def __call__(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str, float]:
        started = time.time()

        body_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system}
        ]
        body_messages.extend(messages)

        payload = {
            "model": self._cfg.model,
            "messages": body_messages,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "stream": False,
        }
        headers = {
            "authorization": f"Bearer {self._cfg.api_key}",
            "content-type": "application/json",
        }
        url = f"{self._cfg.base_url}/chat/completions"

        timeout = aiohttp.ClientTimeout(total=self._cfg.timeout_s)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as r:
                    if r.status != 200:
                        body = await r.text()
                        raise RuntimeError(
                            f"NRP {r.status}: {body[:200]}"
                        )
                    data = await r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("NRP call failed: %s", e)
            return "", "timeout", time.time() - started

        choices = data.get("choices") or []
        if not choices:
            return "", "empty", time.time() - started
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                seg.get("text", "") for seg in content if isinstance(seg, dict)
            )
        return content.strip(), self._cfg.model, time.time() - started


# ─────────────────────────────────────────────────────────────────
# Persona-scoped Bible.
# Keeps the LAST-MILE forbidden-term check inside the validator
# rather than trusting the LLM to honor the system prompt.
# ─────────────────────────────────────────────────────────────────


_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "the Chamber",
    "The Chamber",
    "Chamber operative",
    "Ostland",
    "Meridian",
    "Specimen A-7",
    "A-7 case",
    "Persephone plaque",
    "Mi-go",
    "Mi go",
    "Winter Correspondents",
    "the gate",
    "the sleeper",
    "Elder Thing",
    "Yithian",
    "Starry Wisdom",
    "Hollow Hand",
    "Keepers of the Hollow Throne",
)


def _bible_for(which: str) -> Bible:
    """Build a persona-scoped Bible with shared + persona-only chunks
    as ``known`` content, plus the campaign-wide forbidden-phrase
    list. ``unknown`` stays empty here — anything in the forbidden
    list already trips the validator's ``forbidden_check``.

    A future refactor can chunk the full campaign guide and tag
    each chunk with per-AI visibility; for now the condensed
    setting facts in :mod:`.campaign_facts` are enough grounding
    to answer routine questions in-persona.
    """
    tag = "artemis_known" if which == "artemis" else "sigma4_known"
    persona_chunk_title = "ARTEMIS scope" if which == "artemis" else "SIGMA-4 scope"
    persona_text = ARTEMIS_ONLY if which == "artemis" else SIGMA_ONLY

    return Bible(
        known=(
            BibleChunk(
                id="shared-setting",
                tag=tag,
                title="Shared setting",
                text=SHARED_CONTEXT.strip(),
            ),
            BibleChunk(
                id=f"{which}-scope",
                tag=tag,
                title=persona_chunk_title,
                text=persona_text.strip(),
            ),
        ),
        unknown=(),
        forbidden_phrases=_FORBIDDEN_PHRASES,
    )


# ─────────────────────────────────────────────────────────────────
# Per-session state cache
# ─────────────────────────────────────────────────────────────────


_states: dict[tuple[str, str], SessionState] = {}


def _session_state(session_id: str, which: str) -> SessionState:
    key = (session_id, which)
    s = _states.get(key)
    if s is None:
        s = SessionState(session_id=f"{session_id}:{which}")
        _states[key] = s
    return s


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnResult:
    text: str
    source: str          # "handle_turn" | "stub-fallback"
    proof_hash: str | None
    latency_s: float
    expert: str | None
    ts: float


async def run_turn(
    *,
    session_id: str,
    which: str,
    user_text: str,
    speaker: str = "player",
) -> TurnResult:
    """Run one AI turn through the full agi-hpc pipeline.

    Returns a :class:`TurnResult` regardless of upstream state:

    - **``source="handle_turn"``**: the real pipeline ran. Includes
      the ``proof_hash`` of the validator's DecisionProof.
    - **``source="stub-fallback"``**: the NRP token was unset, or
      the LLM or validator refused. A keyword-based in-persona
      line is returned so the client always sees something.
    """
    if which not in {"artemis", "sigma4"}:
        raise ValueError(f"unknown AI: {which!r}")

    cfg = LLMConfig.from_env()
    started = time.time()

    if cfg is None:
        log.info("NRP_LLM_TOKEN unset; using stub fallback for %s", which)
        return _stub_result(which, user_text, started)

    caller = NRPLLMCaller(cfg)
    state = _session_state(session_id, which)
    bible = _bible_for(which)

    # Build the right TurnRequest. The shape is identical across
    # the two packages; we dispatch the right type so each AI's
    # sentinel conventions (INTERFACE_FLICKER vs INTERFACE_SILENT)
    # apply correctly inside handle_turn.
    turn_id = f"t-{int(time.time() * 1000)}"
    meta = {"via": "halyard-keeper.llm"}

    try:
        if which == "artemis":
            req = ArtemisTurnRequest(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                text=user_text,
                ts=time.time(),
                meta=meta,
            )
            resp: ArtemisTurnResponse | None = await artemis_handle_turn(
                req, state=state, bible=bible, llm=caller,
            )
        else:
            req_s = Sigma4TurnRequest(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                text=user_text,
                ts=time.time(),
                meta=meta,
            )
            resp = await sigma4_handle_turn(
                req_s, state=state, bible=bible, llm=caller,
            )
    except Exception as e:  # noqa: BLE001
        log.warning("handle_turn raised for %s: %s", which, e)
        return _stub_result(which, user_text, started)

    if resp is None:
        # Trigger policy declined, validator rejected, or the
        # model emitted the silence sentinel. Give the client a
        # flavor line so the stream doesn't feel broken.
        log.info("handle_turn returned None for %s; using stub fallback", which)
        return _stub_result(which, user_text, started)

    return TurnResult(
        text=resp.text,
        source="handle_turn",
        proof_hash=resp.proof_hash,
        latency_s=time.time() - started,
        expert=resp.expert,
        ts=time.time(),
    )


def _stub_result(which: str, user_text: str, started: float) -> TurnResult:
    reply = stub_reply(which=which, text=user_text)
    return TurnResult(
        text=reply.text,
        source="stub-fallback",
        proof_hash=None,
        latency_s=time.time() - started,
        expert=None,
        ts=reply.ts,
    )


# ─────────────────────────────────────────────────────────────────
# Compatibility helpers
# ─────────────────────────────────────────────────────────────────


# Re-exported for tests or admin tooling that wants to see the
# effective prompt for one of the AIs.
def effective_system_prompt(which: str) -> str:
    """Return the system prompt that would be sent to the LLM."""
    return system_prompt_for(which)
