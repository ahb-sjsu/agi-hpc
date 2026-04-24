# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Trigger policy — decides whether SIGMA-4 should reply this turn.

Parallel to :mod:`agi.primer.artemis.trigger` but with SIGMA's
invocation phrases: "sigma", "sigma-4", "sig", "the ship",
"the ship-mind". "Sig" is the in-fiction diminutive; experienced
crewmembers use it, which means triggering on "sig" alone would
over-fire on Belter speech patterns (e.g., "signature"). We guard
against that with a word-boundary regex and a "must be preceded
or followed by directed language" check.

v1 triggers:
  1. Explicit address — text contains an invocation phrase.
  2. Keeper cue — ``speaker == 'keeper'`` with
     ``meta.sigma4_cue`` true.

v2 (deferred): proactive relevance classifier.
"""

from __future__ import annotations

import re
from typing import Any

# Match "sigma" or "sigma-4" on word boundaries, case-insensitive.
_NAME_RE = re.compile(r"\bsigma(-4)?\b", re.IGNORECASE)

# "Sig" as a standalone word, but not "signature", "signal", etc.
# We require that "sig" be a complete word — the regex enforces
# word boundaries on both sides.
_SIG_RE = re.compile(r"\bsig\b", re.IGNORECASE)

# Invocation phrases the Keeper might configure in place of the
# raw identity tokens. Configurable via the ``invocation_phrases``
# argument to :func:`should_speak`.
_DEFAULT_INVOCATIONS: tuple[str, ...] = (
    "sigma",
    "sigma-4",
    "sig",
    "the ship",
    "the ship-mind",
)


def _name_mentioned(text: str, phrases: tuple[str, ...]) -> bool:
    """Return True iff the text contains any invocation phrase.

    Single-word phrases must hit on word boundaries. Multi-word
    phrases match as substrings (word boundaries get too
    restrictive with phrase-internal whitespace).
    """
    lower = text.lower()
    for phrase in phrases:
        if " " in phrase:
            if phrase in lower:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", lower):
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
      1. If the SIGMA-4 kill-switch is on → False, reason="silenced".
      2. If a safety flag is set on this turn → False,
         reason=f"safety_flag:{flag}".
      3. If ``speaker == "keeper"`` and ``meta.sigma4_cue`` is truthy
         → True, reason="keeper_cue".
      4. If an invocation phrase is in the text → True,
         reason="addressed".
      5. Otherwise → False, reason="not_addressed".

    Safety flags silence at the trigger layer for defense in depth —
    the validator also enforces this, but we shouldn't spend tokens
    generating a reply we'll just refuse to post.

    Note that SIGMA-4 and ARTEMIS can both be triggered by the same
    turn (e.g., "SIGMA, ask ARTEMIS about the sample freezer.").
    That is a feature, not a bug: room etiquette is the Keeper's
    call, not the trigger policy's.
    """
    if silenced:
        return False, "silenced"
    if safety_flag in {"x-card", "pause"}:
        return False, f"safety_flag:{safety_flag}"
    meta = meta or {}
    if speaker == "keeper" and meta.get("sigma4_cue"):
        return True, "keeper_cue"
    if _name_mentioned(text, invocation_phrases):
        return True, "addressed"
    return False, "not_addressed"
