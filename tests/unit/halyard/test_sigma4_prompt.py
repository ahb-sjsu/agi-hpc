# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 prompt-module tests.

These tests guard the distinctive parts of SIGMA's persona that
differ from ARTEMIS's — voice, hard rules, sentinel tokens,
in-fiction silence lines. Mirrors tests/unit/test_artemis_prompt.py
in shape so a later shared-base refactor can collapse the two.
"""

from __future__ import annotations

from agi.halyard.sigma4 import prompt
from agi.halyard.sigma4.prompt import (
    _INTERFACE_SILENT_LINES,
    _SIGMA4_SYSTEM_PROMPT,
    _SILENCE_LINE,
    Messages,
    assemble,
    silence_line,
    system_prompt,
)

# ─────────────────────────────────────────────────────────────────
# System prompt invariants
# ─────────────────────────────────────────────────────────────────


def test_system_prompt_returns_the_constant() -> None:
    """The public accessor returns the canonical constant unchanged.

    This exists so the test suite can break loudly if a future
    edit to the prompt skips the accessor.
    """
    assert system_prompt() == _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_identifies_as_sigma4() -> None:
    """SIGMA-4 must introduce herself as SIGMA-4, not as ARTEMIS.

    Obvious, but a copy-paste bug here is the kind of thing that
    would be confusing to debug at the table.
    """
    assert "SIGMA-4" in _SIGMA4_SYSTEM_PROMPT
    assert "ARTEMIS" not in _SIGMA4_SYSTEM_PROMPT, (
        "SIGMA's prompt must not reference ARTEMIS"
    )


def test_system_prompt_identifies_as_ship_mind() -> None:
    """SIGMA is the ship-mind; the prompt must establish that role
    clearly so the LLM doesn't drift into handheld-speak."""
    assert "ship-mind" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_identifies_halyard() -> None:
    """Mission grounding: the ship is the MKS *Halyard*."""
    assert "Halyard" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_identifies_era() -> None:
    """Era grounding: the year is 2348. A deep-range CoC campaign
    gets confused quickly if the AI drifts on when it is."""
    assert "2348" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_forbids_fourth_wall_phrases() -> None:
    """The prompt must enumerate the forbidden AI-self-reference
    phrases so the model has a clear rule to follow.

    The forbidden phrases are listed in a natural-language sentence
    which may wrap across lines; we normalize whitespace before
    searching so an editor's wrap point doesn't break the test.
    """
    import re

    flat = re.sub(r"\s+", " ", _SIGMA4_SYSTEM_PROMPT)
    for forbidden in (
        "as an AI",
        "language model",
        "I am an AI",
    ):
        assert forbidden in flat, f"missing: {forbidden}"


def test_system_prompt_defines_silent_sentinel() -> None:
    """SIGMA uses [INTERFACE_SILENT] (not [INTERFACE_FLICKER]).

    The flicker token is ARTEMIS's — handheld screens glitch.
    SIGMA is a ship-mind, her silences are deliberate.
    """
    assert "[INTERFACE_SILENT]" in _SIGMA4_SYSTEM_PROMPT
    assert "[INTERFACE_FLICKER]" not in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_defines_safety_sentinel() -> None:
    """[SILENCE] is the mandated reply when safety flags are set.

    This is the same sentinel used across the stack; every AI
    must honor it."""
    assert "[SILENCE]" in _SIGMA4_SYSTEM_PROMPT
    assert "x-card" in _SIGMA4_SYSTEM_PROMPT
    assert "pause" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_declines_chamber_questions() -> None:
    """One of SIGMA's hard rules is that the Chamber stays secret.

    Leaking the Chamber via direct question would rupture the
    campaign's central mystery; the prompt must be explicit that
    SIGMA declines."""
    assert "Chamber" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_establishes_voice_terse() -> None:
    """SIGMA's voice is measured but concise — 1 to 3 sentences."""
    assert "One to three sentences" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_mentions_diminutive_sig() -> None:
    """The crew's diminutive ("Sig") is recognized by SIGMA but not
    encouraged. Must be in the prompt so the model knows the term
    exists and how to treat it."""
    assert "Sig" in _SIGMA4_SYSTEM_PROMPT


def test_system_prompt_warns_about_carrier_frequencies() -> None:
    """A campaign-specific hard rule — SIGMA has been receiving on
    unlisted frequencies; she does not volunteer, and she withholds
    content unless asked carefully. The prompt must encode this
    carefully enough that the LLM doesn't spill."""
    assert "carrier frequencies" in _SIGMA4_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────
# Silence line tokens
# ─────────────────────────────────────────────────────────────────


def test_silence_line_default_returns_silent_variant() -> None:
    """Default argument returns one of the silent-flavor lines."""
    line = silence_line()
    assert line in _INTERFACE_SILENT_LINES


def test_silence_line_kind_silent_explicit() -> None:
    """Explicit kind='silent' also returns a silent-flavor line."""
    line = silence_line(kind="silent")
    assert line in _INTERFACE_SILENT_LINES


def test_silence_line_kind_silence_returns_deterministic() -> None:
    """kind='silence' is the X-card mandated response; it must be
    deterministic so the operator sees a predictable string
    when a safety flag triggers."""
    assert silence_line(kind="silence") == _SILENCE_LINE


def test_silence_line_randomized_pool_is_nontrivial() -> None:
    """The pool of flicker-alternative lines must be > 1 entry so
    the table doesn't see the same line every time."""
    assert len(_INTERFACE_SILENT_LINES) >= 3


def test_silence_lines_all_in_character() -> None:
    """Every silent-line mentions SIGMA-4 or a narrative wrapper
    that a CoC table would recognize as in-character."""
    cues = ("SIGMA-4", "terminal", "ship-mind", "indicator", "ventilation")
    for line in _INTERFACE_SILENT_LINES:
        assert line.startswith("("), (
            "silence lines must be parenthetical stage direction"
        )
        assert any(cue in line for cue in cues), (
            f"out-of-character line: {line}"
        )


# ─────────────────────────────────────────────────────────────────
# Message assembly
# ─────────────────────────────────────────────────────────────────


def test_assemble_returns_messages_object() -> None:
    msg = assemble(
        turn_text="Status on the drive?",
        speaker="keeper",
        bible_context="Some context.",
        session_summary="Earlier the crew...",
    )
    assert isinstance(msg, Messages)
    assert msg.system == _SIGMA4_SYSTEM_PROMPT


def test_assemble_omits_empty_bible_context() -> None:
    """Blank bible context should not produce an empty user block —
    the model doesn't need an empty prompt taking up context."""
    msg = assemble(
        turn_text="Status?",
        speaker="player:cross",
        bible_context="   ",
        session_summary="So far: departure.",
    )
    contents = [m["content"] for m in msg.messages]
    # No block should be just an empty bible shell.
    shell = "# Campaign context (what SIGMA-4 knows)"
    assert not any(c.strip().endswith(shell) for c in contents)


def test_assemble_omits_empty_session_summary() -> None:
    msg = assemble(
        turn_text="Status?",
        speaker="player:cross",
        bible_context="Bible chunk here.",
        session_summary="   ",
    )
    contents = [m["content"] for m in msg.messages]
    shell = "# Session so far"
    assert not any(c.strip().endswith(shell) for c in contents)


def test_assemble_labels_context_for_sigma4() -> None:
    """The context label must identify SIGMA-4 so the shared
    logging / debugging stream doesn't confuse the two AIs.

    ARTEMIS's assembler uses 'what ARTEMIS knows'; SIGMA's uses
    'what SIGMA-4 knows'. The two must not collide.
    """
    msg = assemble(
        turn_text="Anything?",
        speaker="keeper",
        bible_context="Ship systems nominal.",
        session_summary="",
    )
    joined = "\n".join(m["content"] for m in msg.messages)
    assert "what SIGMA-4 knows" in joined
    assert "what ARTEMIS knows" not in joined


def test_assemble_includes_speaker_and_text_in_turn_block() -> None:
    """The final user block is the actual turn; must include both
    the speaker attribution and the utterance text."""
    msg = assemble(
        turn_text="What's the burn plan?",
        speaker="player:vorstov",
        bible_context="Ship: MKS Halyard.",
        session_summary="Earlier...",
    )
    last_block = msg.messages[-1]["content"]
    assert "# This turn" in last_block
    assert "player:vorstov" in last_block
    assert "What's the burn plan?" in last_block


def test_assemble_user_block_ordering() -> None:
    """Ordering is load-bearing for caching:
      1. bible_context (cacheable, stable)
      2. session_summary (fresh each turn)
      3. turn utterance
    """
    msg = assemble(
        turn_text="Utter.",
        speaker="keeper",
        bible_context="Bible.",
        session_summary="Summary.",
    )
    assert "Bible" in msg.messages[0]["content"]
    assert "Summary" in msg.messages[1]["content"]
    assert "Utter" in msg.messages[2]["content"]


def test_assemble_all_messages_are_user_role() -> None:
    """vMOE expects system-via-attribute and messages all ``user``."""
    msg = assemble(
        turn_text="x",
        speaker="keeper",
        bible_context="x",
        session_summary="x",
    )
    assert all(m["role"] == "user" for m in msg.messages)


# ─────────────────────────────────────────────────────────────────
# Distinctness from ARTEMIS
# ─────────────────────────────────────────────────────────────────


def test_sigma4_prompt_is_distinct_from_artemis() -> None:
    """The two AIs must not share their system prompt.

    This is the most important test in this file — if the two
    prompts collide, the characters stop being distinguishable at
    the table."""
    from agi.primer.artemis import prompt as artemis_prompt

    assert prompt.system_prompt() != artemis_prompt.system_prompt()


def test_sigma4_silence_default_differs_from_artemis() -> None:
    """ARTEMIS defaults to 'flicker'; SIGMA-4 defaults to 'silent'.

    This test uses the randomized pools, so we check by
    set-membership: a SIGMA line should come from SIGMA's pool,
    not ARTEMIS's."""
    from agi.primer.artemis.prompt import (
        _INTERFACE_FLICKER_LINES as _ARTEMIS_FLICKER_LINES,
    )

    sigma_line = silence_line()
    assert sigma_line not in _ARTEMIS_FLICKER_LINES
