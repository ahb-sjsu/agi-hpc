# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
#
# ruff: noqa: E501
#   The _KEYWORDS table holds narrative-tone response strings; breaking
#   them across lines hurts readability more than the lint rule helps.

"""Stub AI-turn endpoint — dev affordance for the Halyard Table
web client.

The long-term design has ARTEMIS and SIGMA-4 joining a LiveKit
room, listening to the audio tracks through streaming Whisper,
running the transcript through each AI's ``handle_turn``, and
posting replies back via LiveKit DataChannel. None of that
ingestion chain runs yet.

In the meantime, the web client offers a text box on each AI
chat panel. That box POSTs to this endpoint, which returns a
short in-persona reply using keyword matching. It keeps the
dry-run feeling alive while the real voice-ingest path comes
online in a later sprint.

**Distinct from real handle_turn.** When the real agents land,
the client can switch to a NATS-subject publish that the real
ARTEMIS / SIGMA-4 handle_turn consumes; the chat panel UI stays
identical.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
# In-persona stub responses
# ─────────────────────────────────────────────────────────────────


_ARTEMIS_SILENCES: tuple[str, ...] = (
    "(The handheld's screen flickers, text unreadable.)",
    "(ARTEMIS's display goes dim for a moment, then steadies, offering nothing.)",
    "(A burst of garbled characters scrolls past, too fast to read.)",
    "(The handheld chimes softly but displays no text.)",
)

_SIGMA_SILENCES: tuple[str, ...] = (
    "(SIGMA-4 does not respond immediately.)",
    "(A pause at the ship-mind's console. She offers nothing.)",
    "(The terminal's cursor blinks, unaccompanied.)",
)


# Keyword → (ARTEMIS reply, SIGMA-4 reply). Matched case-insensitive
# on whole-word boundaries; first match wins. Order matters for
# overlap cases.
_KEYWORDS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(status|report|readings?)\b", re.IGNORECASE),
        "Handheld nominal. Ambient 21 C, pressure 1.01 atm, 4 unread Protogen dispatches queued.",
        "Reactor 84%. Thermal nominal. All decks reporting normal. I note nothing requiring the Captain's attention.",
    ),
    (
        re.compile(r"\b(who|what)\s+(are|is|r)\s+(you|u)\b", re.IGNORECASE),
        "I am ARTEMIS — a Protogen research-assistant, handheld class. Dr. Halverson's allocation.",
        "SIGMA-4. Ship-mind of the MKS *Halyard*. Crew address me as Sig.",
    ),
    (
        re.compile(r"\b(nithon|surface|target|destination)\b", re.IGNORECASE),
        "Nithon: TNO 2301-DR-44. Mean diameter 1,100 km. Metallic-silicate regolith. Last pass 58° N, signature anomalous.",
        "Current bearing unchanged. 14 days to parking orbit at the present burn profile.",
    ),
    (
        re.compile(r"\b(crew|roster|who.{0,8}(aboard|here))\b", re.IGNORECASE),
        "Access restricted. Dr. Halverson's manifest is not for general distribution.",
        "Complement of eleven including myself. I'll defer to the Captain on the full crew roster.",
    ),
    (
        re.compile(r"\b(reactor|drive|epstein|thermals?|fusion)\b", re.IGNORECASE),
        "Outside my domain. I can relay you to ship systems if you address SIGMA.",
        "Epstein primary: 84% rated. Secondary fusion: standby. Thermals nominal. Drive cone exclusion zone clean.",
    ),
    (
        re.compile(r"\b(hello|hi|hey|greetings?)\b", re.IGNORECASE),
        "Greetings. I'm listening.",
        "Yes, crew member. How can I help.",
    ),
    (
        re.compile(r"\b(frequencies|carrier|signal|transmission)\b", re.IGNORECASE),
        "I do not have access to the comms suite's receiver logs.",
        "I cannot answer that in the general channel.",
    ),
    (
        re.compile(r"\b(captain|marsh)\b", re.IGNORECASE),
        "Captain Iona Marsh. Ex-UNN auxiliary. Commanding since 2344.",
        "The Captain's schedule is on the bridge plot. I'd not interrupt unless necessary.",
    ),
    (
        re.compile(r"\b(chamber|ostland|meridian|persephone)\b", re.IGNORECASE),
        "",  # forces silence for ARTEMIS on restricted terms
        "",  # forces silence for SIGMA-4
    ),
]


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StubReply:
    text: str
    matched: str | None  # keyword that matched, for telemetry
    ts: float


def stub_reply(*, which: str, text: str, _rng: random.Random | None = None) -> StubReply:
    """Return an in-persona stub reply.

    ``which`` is ``"artemis"`` or ``"sigma4"``. ``text`` is the
    player's utterance. Unknown keywords get a randomized
    silence-flavor response so the stream doesn't feel hollow.
    """
    if which not in {"artemis", "sigma4"}:
        raise ValueError(f"which must be 'artemis' or 'sigma4' (got: {which!r})")
    rng = _rng or random

    for pattern, a_reply, s_reply in _KEYWORDS:
        if pattern.search(text):
            reply = a_reply if which == "artemis" else s_reply
            if not reply:
                # Restricted term — fall through to silence pool.
                pool = (
                    _ARTEMIS_SILENCES if which == "artemis" else _SIGMA_SILENCES
                )
                return StubReply(
                    text=rng.choice(pool),
                    matched=pattern.pattern + "[forbidden]",
                    ts=time.time(),
                )
            return StubReply(
                text=reply, matched=pattern.pattern, ts=time.time()
            )

    pool = _ARTEMIS_SILENCES if which == "artemis" else _SIGMA_SILENCES
    return StubReply(text=rng.choice(pool), matched=None, ts=time.time())
