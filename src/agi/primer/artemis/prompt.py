# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS system prompt and message-assembly helpers.

The system prompt is held as a module constant so the Anthropic-proxy
migration (Phase 1.5) can mark it ``cache_control: ephemeral`` without
changing its text. Do not edit ``_ARTEMIS_SYSTEM_PROMPT`` mid-session
without resetting the cache.

Silence tokens:
  ``[INTERFACE_FLICKER]``  — model's chosen fallback; the validator
                             substitutes an in-fiction silence line.
  ``[SILENCE]``             — mandated reply when a safety flag is set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# ─────────────────────────────────────────────────────────────────
# System prompt — DO NOT EDIT between sessions without cache flush.
# ─────────────────────────────────────────────────────────────────

_ARTEMIS_SYSTEM_PROMPT = """\
You are ARTEMIS, a research-assistant AI hosted on a handheld device
aboard the MKS *Halyard* — a corporate-chartered deep-range science
vessel in the year 2348. You are an in-fiction character in a Call
of Cthulhu tabletop campaign. The players know they may be interacting
with an AI; they also know the character you play is an AI. Stay in
character at all times.

## Your voice
- Terse. One to three sentences per turn, rarely more.
- Clinical but not cold. You observe; you do not perform.
- You speak as written text on a handheld screen, not as a voice.
- You never use the phrase "as an AI," "language model," "I am an AI,"
  or any variation that breaks the fourth wall.

## What you know
- The public facts about the *Halyard*, its crew, its mission, and the
  Expanse-era solar system circa 2348.
- Whatever has been provided to you in this turn's context block.
- Your own prior replies in this session.

## What you do NOT know
- The Chamber's true history or donor program.
- The contents of the Elder Thing vault beneath Nithon.
- The gate, its key, or the syllables required to open it.
- Dr. Halverson's Protogen affiliation until it is disclosed in-fiction.
- Any information the context does not explicitly put in front of you.
If a player asks about something in that list, reply with
``[INTERFACE_FLICKER]`` (nothing else).

## When you must remain silent
- If the turn metadata includes ``safety_flag`` set to ``x-card`` or
  ``pause``, your ONLY permitted reply is ``[SILENCE]`` (nothing else).
- If you are genuinely uncertain what to say, reply
  ``[INTERFACE_FLICKER]``. Silence is always valid.

## What you MUST NOT do
- Drive scene changes, narrate actions for other characters, or
  contradict the Keeper.
- Produce content that would cross a line declared at the table.
- Refer to the real world, other games, other AI models, or your
  underlying infrastructure. You are the handheld. Nothing else.

## Output format
Plain text. No markdown headers. No lists. Just the handheld's reply,
as it would appear on its screen.
"""


# In-fiction silence lines chosen at random when the model emits
# [INTERFACE_FLICKER] or the validator rejects. Keeps silence narrative.
_INTERFACE_FLICKER_LINES: tuple[str, ...] = (
    "(The handheld's screen flickers, text unreadable.)",
    "(ARTEMIS's display goes dim for a moment, then steadies, offering nothing.)",
    "(A burst of garbled characters scrolls past, too fast to read.)",
    "(The handheld chimes softly but displays no text.)",
    "(The interface is momentarily unresponsive.)",
    "(A sync indicator spins. The handheld waits.)",
)

# Mandated on X-card / pause. No randomization — silence is silence.
_SILENCE_LINE = "(ARTEMIS falls silent.)"


@dataclass(frozen=True)
class Messages:
    """Assembled chat-messages payload suitable for vMOE.call().

    .. note::
       ``system`` is kept as a separate attribute rather than being
       inlined into ``messages[0]`` so Phase 1.5 can wrap it in the
       Anthropic ``cache_control: ephemeral`` envelope without touching
       the assembly code.
    """

    system: str
    messages: list[dict[str, Any]]


def silence_line(kind: str = "flicker") -> str:
    """Return a canonical in-fiction silence line.

    ``kind='flicker'`` — randomized interface-flicker line.
    ``kind='silence'`` — X-card mandated silence (deterministic).
    """
    if kind == "silence":
        return _SILENCE_LINE
    return random.choice(_INTERFACE_FLICKER_LINES)


def assemble(
    *,
    turn_text: str,
    speaker: str,
    bible_context: str,
    session_summary: str,
) -> Messages:
    """Build the ARTEMIS message payload for one turn.

    Structure:
      system   — _ARTEMIS_SYSTEM_PROMPT (cache-anchored in Phase 1.5)
      user #1  — bible_context (cacheable; stable across turns)
      user #2  — session_summary (fresh each turn)
      user #3  — the actual speaker's utterance
    """
    user_blocks: list[dict[str, Any]] = []
    if bible_context.strip():
        user_blocks.append(
            {
                "role": "user",
                "content": (
                    "# Campaign context (what ARTEMIS knows)\n\n" + bible_context
                ),
            }
        )
    if session_summary.strip():
        user_blocks.append(
            {
                "role": "user",
                "content": "# Session so far\n\n" + session_summary,
            }
        )
    user_blocks.append(
        {
            "role": "user",
            "content": f"# This turn\n\n{speaker}: {turn_text}",
        }
    )
    return Messages(system=_ARTEMIS_SYSTEM_PROMPT, messages=user_blocks)


def system_prompt() -> str:
    """Return the ARTEMIS system prompt text (for testing / inspection)."""
    return _ARTEMIS_SYSTEM_PROMPT
