# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Trigger policy — decides whether ARTEMIS should reply this turn.

v1 triggers (default):
  1. Explicit address — speaker text mentions "artemis" on a word
     boundary, case-insensitive.
  2. Keeper cue — a ``speaker == 'keeper'`` turn whose metadata flag
     ``artemis_cue`` is true.

v2 (feature-flagged off): proactive relevance. Not implemented here.
"""

from __future__ import annotations

import re
from typing import Any

_NAME_RE = re.compile(r"\bartemis\b", re.IGNORECASE)

# Invocation phrases the Keeper might configure in place of the raw name.
# Configurable via the ``invocation_phrases`` argument to should_speak().
_DEFAULT_INVOCATIONS: tuple[str, ...] = (
    "artemis",
    "the handheld",
)


def _name_mentioned(text: str, phrases: tuple[str, ...]) -> bool:
    lower = text.lower()
    for phrase in phrases:
        # Word-boundary on single-word phrases, substring for multi-word.
        if " " in phrase:
            if phrase in lower:
                return True
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", lower):
                return True
    return False


def should_speak(
    *,
    speaker: str,
    text: str,
    meta: dict[str, Any] | None = None,
    invocation_phrases: tuple[str, ...] = _DEFAULT_INVOCATIONS,
    silenced: bool = False,
    safety_flag: str | None = None,
) -> tuple[bool, str]:
    """Return ``(should_speak, reason)``.

    Decision order (short-circuits):
      1. If the ARTEMIS kill-switch is on → False, reason="silenced".
      2. If a safety flag is set on this turn → False, reason="safety_flag".
      3. If speaker=="keeper" and meta.artemis_cue → True, reason="keeper_cue".
      4. If the name is mentioned in the text → True, reason="addressed".
      5. Otherwise → False, reason="not_addressed".

    Safety flags cause silence at the *trigger* layer too (defense in
    depth — the validator also enforces this, but we shouldn't even
    spend the tokens).
    """
    if silenced:
        return False, "silenced"
    if safety_flag in {"x-card", "pause"}:
        return False, f"safety_flag:{safety_flag}"
    meta = meta or {}
    if speaker == "keeper" and meta.get("artemis_cue"):
        return True, "keeper_cue"
    if _name_mentioned(text, invocation_phrases):
        return True, "addressed"
    return False, "not_addressed"
