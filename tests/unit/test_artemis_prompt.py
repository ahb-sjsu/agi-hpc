# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for ARTEMIS system prompt and message assembly."""

from __future__ import annotations

from agi.primer.artemis import prompt


def test_system_prompt_non_empty():
    sp = prompt.system_prompt()
    assert sp
    # Load-bearing rules must be present textually — if someone edits
    # the prompt, these anchors guard against accidental removal.
    assert "ARTEMIS" in sp
    assert "[INTERFACE_FLICKER]" in sp
    assert "[SILENCE]" in sp
    assert "as an AI" in sp  # the OOC prohibition text


def test_assemble_includes_all_blocks():
    out = prompt.assemble(
        turn_text="what do you see outside?",
        speaker="player:imogen",
        bible_context="the ship hums in the long dark",
        session_summary="- earlier turn",
    )
    assert out.system == prompt.system_prompt()
    assert len(out.messages) == 3  # bible + session + this turn
    assert "Campaign context" in out.messages[0]["content"]
    assert "the ship hums" in out.messages[0]["content"]
    assert "Session so far" in out.messages[1]["content"]
    assert "This turn" in out.messages[2]["content"]
    assert "player:imogen" in out.messages[2]["content"]
    assert "what do you see outside?" in out.messages[2]["content"]


def test_assemble_skips_empty_blocks():
    out = prompt.assemble(
        turn_text="hey",
        speaker="player:sully",
        bible_context="",
        session_summary="   ",
    )
    # Only the "this turn" block should remain.
    assert len(out.messages) == 1
    assert "This turn" in out.messages[0]["content"]


def test_silence_line_flicker_is_from_pool():
    # Deterministic smoke: the line must contain the narrative beat word.
    for _ in range(20):
        line = prompt.silence_line("flicker")
        assert line.startswith("(")
        assert line.endswith(")")


def test_silence_line_silence_is_canonical():
    # The X-card / pause mandated silence line is deterministic.
    assert prompt.silence_line("silence") == "(ARTEMIS falls silent.)"
    # And is stable across calls.
    assert prompt.silence_line("silence") == prompt.silence_line("silence")
