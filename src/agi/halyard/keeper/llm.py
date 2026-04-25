# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Halyard Keeper AI-turn paths.

Two code paths, matching the campaign's design split:

- **SIGMA-4** — routes through ``agi.primer.vmoe.vMOE.cascade``.
  Fast, resilient, no validator/DecisionProof overhead. SIGMA is
  a ship-mind; the stakes are low, the priority is responsive
  dialogue. The cascade prefers non-thinking models (gpt-oss →
  glm-4.7 → kimi → qwen3) and returns the first non-empty reply.
- **ARTEMIS** — routes through the full agi-hpc pipeline:
  ``agi.primer.artemis.mode.handle_turn`` with a vMOE-backed
  :class:`VMOELLMCaller`. Bible retrieval, ErisML validator (6
  checks including secret-leak and forbidden-phrase), SHA-chained
  DecisionProof, session log. This is the "contaminated handheld"
  of the in-fiction setting — exactly the thing that deserves
  the full validation pipeline.

Both paths fall back to :func:`.ai_stub.stub_reply` if NRP creds
are unavailable or every expert fails; the web client always
gets an in-persona line, never a 500.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agi.primer.artemis.context import Bible, BibleChunk, SessionState
from agi.primer.artemis.mode import (
    TurnRequest as ArtemisTurnRequest,
)
from agi.primer.artemis.mode import (
    TurnResponse as ArtemisTurnResponse,
)
from agi.primer.artemis.mode import handle_turn as artemis_handle_turn
from agi.primer.vmoe import Expert, Response, vMOE

from .ai_stub import stub_reply
from .campaign_facts import (
    ARTEMIS_ONLY,
    SHARED_CONTEXT,
    SIGMA_ONLY,
    system_prompt_for,
)

log = logging.getLogger("halyard.keeper.llm")


# ─────────────────────────────────────────────────────────────────
# Voice bridge — publish AI replies to NATS so the avatar/TTS stack
# on Atlas (atlas-artemis-avatar.service) speaks them aloud in the
# LiveKit room. Best-effort, fire-and-forget; never blocks the reply
# path.
#
# Subjects (per agi.primer.artemis.livekit_agent.avatar_hud):
#   agi.rh.artemis.say.direct  — direct-say bridge, payload {"text": ...}
#   agi.rh.sigma4.say.direct   — same shape, parallel SIGMA-4 avatar
#                                (when deployed)
#
# Disabled if HALYARD_VOICE_BRIDGE != "1" or if nats-py is missing.
# ─────────────────────────────────────────────────────────────────


_VOICE_SUBJECTS = {
    "artemis": "agi.rh.artemis.say.direct",
    "sigma4": "agi.rh.sigma4.say.direct",
}


async def _voice_publish(which: str, text: str) -> None:
    """Best-effort publish of an AI reply to the voice/avatar NATS
    subject. Never raises; failures are logged at warning level."""
    if os.environ.get("HALYARD_VOICE_BRIDGE", "1") not in ("1", "true", "yes"):
        return
    subject = _VOICE_SUBJECTS.get(which)
    if not subject or not text or not text.strip():
        return
    try:
        import nats  # type: ignore
    except ImportError:
        log.debug("nats-py not installed; voice bridge disabled")
        return
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    try:
        nc = await nats.connect(servers=[nats_url], connect_timeout=2)
    except Exception as e:  # noqa: BLE001
        log.warning("voice bridge: NATS connect failed (%s)", e)
        return
    try:
        await nc.publish(subject, json.dumps({"text": text}).encode())
        await nc.drain()
        log.debug("voice bridge: published %d chars to %s", len(text), subject)
    except Exception as e:  # noqa: BLE001
        log.warning("voice bridge: publish failed (%s)", e)
        try:
            await nc.close()
        except Exception:  # noqa: BLE001
            pass


# ─────────────────────────────────────────────────────────────────
# Halyard-specific vMOE pool.
#
# Priority ordering favors NON-THINKING models at the front so
# cascade doesn't burn the turn on qwen3's internal reasoning
# trace when a direct reply is available. Thinking models stay in
# the pool as fallbacks for harder questions.
# ─────────────────────────────────────────────────────────────────


_NRP_BASE = "https://ellm.nrp-nautilus.io/v1"
_NRP_KEY_ENV = "NRP_LLM_TOKEN"


def _halyard_experts() -> list[Expert]:
    return [
        Expert(
            name="gpt-oss",
            model="gpt-oss",
            base_url=_NRP_BASE,
            api_key_env=_NRP_KEY_ENV,
            role_hints=frozenset({"chat", "fast", "default"}),
            timeout_s=60.0,
            priority=10,
        ),
        Expert(
            name="glm-4.7",
            model="glm-4.7",
            base_url=_NRP_BASE,
            api_key_env=_NRP_KEY_ENV,
            role_hints=frozenset({"chat", "reason"}),
            timeout_s=60.0,
            priority=20,
        ),
        Expert(
            name="kimi",
            model="kimi",
            base_url=_NRP_BASE,
            api_key_env=_NRP_KEY_ENV,
            role_hints=frozenset({"chat", "long_context"}),
            timeout_s=90.0,
            priority=30,
        ),
        Expert(
            name="qwen3",
            model="qwen3",
            base_url=_NRP_BASE,
            api_key_env=_NRP_KEY_ENV,
            role_hints=frozenset({"chat", "thinking"}),
            timeout_s=90.0,
            priority=40,
        ),
        Expert(
            name="minimax-m2",
            model="minimax-m2",
            base_url=_NRP_BASE,
            api_key_env=_NRP_KEY_ENV,
            role_hints=frozenset({"chat", "tool_use"}),
            timeout_s=90.0,
            priority=50,
        ),
    ]


@lru_cache(maxsize=1)
def _moe() -> vMOE | None:
    """Lazy, cached vMOE instance. None if NRP creds are unset."""
    if not os.environ.get(_NRP_KEY_ENV, "").strip():
        return None
    try:
        return vMOE(experts=_halyard_experts())
    except Exception as e:  # noqa: BLE001
        log.warning("vMOE construction failed: %s", e)
        return None


def _nonempty(r: Response) -> bool:
    """cascade accept — reject thinking-model empty-content responses
    so the cascade falls through to the next expert."""
    return bool(r.ok and r.content and r.content.strip())


# ─────────────────────────────────────────────────────────────────
# Per-session history — bounded deque per (session_id, which).
# ─────────────────────────────────────────────────────────────────


_HISTORY_CAP_PAIRS = 10
_histories: dict[tuple[str, str], deque[dict[str, str]]] = {}


def _hist(session_id: str, which: str) -> deque[dict[str, str]]:
    key = (session_id, which)
    buf = _histories.get(key)
    if buf is None:
        buf = deque(maxlen=_HISTORY_CAP_PAIRS * 2)
        _histories[key] = buf
    return buf


def _address(which: str, text: str) -> str:
    """Prefix the AI's name so the trigger / invocation match fires.

    The chat panel is an explicit address; the trigger policy
    doesn't know that without seeing the name in the text.
    """
    if which == "artemis" and not re.search(r"\bartemis\b", text, re.I):
        return f"ARTEMIS, {text}"
    if which == "sigma4" and not re.search(
        r"\b(sigma(-4)?|sig)\b", text, re.I
    ):
        return f"SIGMA, {text}"
    return text


# ─────────────────────────────────────────────────────────────────
# SIGMA-4 — vMOE cascade only
# ─────────────────────────────────────────────────────────────────


async def sigma4_turn(
    *, session_id: str, user_text: str
) -> TurnResult:
    moe = _moe()
    if moe is None:
        return _stub_result("sigma4", user_text, time.time())

    addressed = _address("sigma4", user_text)
    buf = _hist(session_id, "sigma4")
    buf.append({"role": "user", "content": addressed})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt_for("sigma4")}
    ]
    messages.extend(buf)

    started = time.time()
    try:
        resp = await moe.cascade(
            messages,
            accept=_nonempty,
            max_tokens=int(os.environ.get("HALYARD_LLM_MAX_TOKENS", "900")),
            temperature=float(os.environ.get("HALYARD_LLM_TEMPERATURE", "0.7")),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("sigma4 cascade raised: %s", e)
        return _stub_result("sigma4", user_text, started)

    if not resp.content.strip():
        log.info("sigma4 cascade produced no content; using stub")
        return _stub_result("sigma4", user_text, started)

    reply = resp.content.strip()
    buf.append({"role": "assistant", "content": reply})
    await _voice_publish("sigma4", reply)
    return TurnResult(
        text=reply,
        source="vmoe-cascade",
        proof_hash=None,
        latency_s=time.time() - started,
        expert=resp.expert,
        ts=time.time(),
    )


# ─────────────────────────────────────────────────────────────────
# ARTEMIS — full AGI-HPC pipeline via handle_turn
# ─────────────────────────────────────────────────────────────────


class VMOELLMCaller:
    """LLMCaller protocol adapter backed by vMOE.cascade.

    :func:`agi.primer.artemis.mode.handle_turn` expects a callable
    matching the ``LLMCaller`` protocol: ``async (system, messages)
    -> (text, expert_name, latency_s)``. This class bridges to the
    MOE so ARTEMIS's handle_turn routes through the cascade.
    """

    def __init__(
        self,
        moe: vMOE,
        *,
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> None:
        self._moe = moe
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def __call__(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]:
        full: list[dict[str, Any]] = [{"role": "system", "content": system}]
        full.extend(messages)
        started = time.time()
        try:
            resp = await self._moe.cascade(
                full,
                accept=_nonempty,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("artemis vMOE cascade raised: %s", e)
            return "", "timeout", time.time() - started
        if not resp.content.strip():
            return "", resp.expert, time.time() - started
        return resp.content.strip(), resp.expert, time.time() - started


# Forbidden phrases — validator's forbidden_check rejects replies
# containing any of these. Defense-in-depth belt to the system
# prompt's withholding rules.
_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "the Chamber", "The Chamber", "Chamber operative",
    "Ostland", "Meridian", "Specimen A-7", "A-7 case",
    "Persephone plaque", "Mi-go", "Mi go",
    "Winter Correspondents", "the gate", "the sleeper",
    "Elder Thing", "Yithian",
    "Starry Wisdom", "Hollow Hand", "Keepers of the Hollow Throne",
)


_BIBLE_JSON_DEFAULT = "/archive/halyard/bible/halyard_bible.json"


@lru_cache(maxsize=1)
def _load_json_bible() -> dict[str, Any] | None:
    path = Path(os.environ.get("HALYARD_BIBLE_PATH", _BIBLE_JSON_DEFAULT))
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load bible at %s: %s", path, e)
        return None


def _synthesize_bible(which: str) -> Bible:
    tag = "artemis_known" if which == "artemis" else "sigma4_known"
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
                title=(
                    "ARTEMIS scope" if which == "artemis" else "SIGMA-4 scope"
                ),
                text=persona_text.strip(),
            ),
        ),
        unknown=(),
        forbidden_phrases=_FORBIDDEN_PHRASES,
    )


def _bible_for(which: str) -> Bible:
    doc = _load_json_bible()
    if doc is None:
        return _synthesize_bible(which)
    known_tag = f"{which}_known"
    unknown_tag = f"{which}_unknown"
    known: list[BibleChunk] = []
    unknown: list[BibleChunk] = []
    for raw in doc.get("chunks", []):
        tag = raw.get("tag")
        if tag not in {known_tag, unknown_tag}:
            continue
        chunk = BibleChunk(
            id=str(raw.get("id", "")),
            tag=str(tag),
            title=str(raw.get("title", "")),
            text=str(raw.get("text", "")),
        )
        if tag == known_tag:
            known.append(chunk)
        else:
            unknown.append(chunk)
    forbidden = list(doc.get("forbidden_phrases", []))
    for phrase in _FORBIDDEN_PHRASES:
        if phrase not in forbidden:
            forbidden.append(phrase)
    return Bible(
        known=tuple(known),
        unknown=tuple(unknown),
        forbidden_phrases=tuple(forbidden),
    )


_artemis_states: dict[str, SessionState] = {}


def _artemis_state(session_id: str) -> SessionState:
    s = _artemis_states.get(session_id)
    if s is None:
        s = SessionState(session_id=session_id)
        _artemis_states[session_id] = s
    return s


async def artemis_turn(
    *, session_id: str, user_text: str, speaker: str = "player"
) -> TurnResult:
    moe = _moe()
    if moe is None:
        return _stub_result("artemis", user_text, time.time())

    addressed = _address("artemis", user_text)
    started = time.time()
    state = _artemis_state(session_id)
    bible = _bible_for("artemis")
    caller = VMOELLMCaller(moe)

    req = ArtemisTurnRequest(
        session_id=session_id,
        turn_id=f"t-{int(time.time() * 1000)}",
        speaker=speaker,
        text=addressed,
        ts=time.time(),
        meta={"via": "halyard-keeper.llm"},
    )
    try:
        resp: ArtemisTurnResponse | None = await artemis_handle_turn(
            req, state=state, bible=bible, llm=caller,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("artemis handle_turn raised: %s", e)
        return _stub_result("artemis", user_text, started)

    if resp is None or not resp.text.strip():
        log.info("artemis handle_turn returned empty; using stub")
        return _stub_result("artemis", user_text, started)

    await _voice_publish("artemis", resp.text)
    return TurnResult(
        text=resp.text,
        source="handle_turn",
        proof_hash=resp.proof_hash,
        latency_s=time.time() - started,
        expert=resp.expert,
        ts=time.time(),
    )


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnResult:
    text: str
    source: str          # "handle_turn" | "vmoe-cascade" | "stub-fallback"
    proof_hash: str | None
    latency_s: float
    expert: str | None
    ts: float


async def run_turn(
    *, session_id: str, which: str, user_text: str, speaker: str = "player",
) -> TurnResult:
    """Dispatcher. ARTEMIS → full pipeline. SIGMA-4 → cascade only."""
    if which == "artemis":
        return await artemis_turn(
            session_id=session_id, user_text=user_text, speaker=speaker,
        )
    if which == "sigma4":
        return await sigma4_turn(session_id=session_id, user_text=user_text)
    raise ValueError(f"unknown AI: {which!r}")


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


def effective_system_prompt(which: str) -> str:
    return system_prompt_for(which)
