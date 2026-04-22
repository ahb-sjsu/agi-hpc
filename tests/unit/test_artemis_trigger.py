# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for ARTEMIS trigger policy."""

from __future__ import annotations

from agi.primer.artemis.trigger import should_speak


def _call(**kwargs):
    """Helper: invoke should_speak with sensible defaults."""
    defaults = {
        "speaker": "player:imogen",
        "text": "",
        "meta": None,
        "silenced": False,
        "safety_flag": None,
    }
    defaults.update(kwargs)
    return should_speak(**defaults)


def test_silenced_short_circuits():
    speak, reason = _call(silenced=True, text="ARTEMIS, report")
    assert speak is False
    assert reason == "silenced"


def test_safety_flag_forces_silence():
    speak, reason = _call(safety_flag="x-card", text="ARTEMIS, report")
    assert speak is False
    assert reason.startswith("safety_flag:")


def test_safety_flag_pause_also_silences():
    speak, reason = _call(safety_flag="pause", text="ARTEMIS, report")
    assert speak is False
    assert reason == "safety_flag:pause"


def test_keeper_cue_triggers():
    speak, reason = _call(
        speaker="keeper", text="narration...", meta={"artemis_cue": True}
    )
    assert speak is True
    assert reason == "keeper_cue"


def test_name_mention_triggers():
    speak, reason = _call(text="ARTEMIS, can you hear me?")
    assert speak is True
    assert reason == "addressed"


def test_name_mention_case_insensitive():
    speak, reason = _call(text="hey artemis")
    assert speak is True
    assert reason == "addressed"


def test_name_inside_word_does_not_trigger():
    # "artemisium" should not trigger; word-boundary matters.
    speak, reason = _call(text="we need more artemisium")
    assert speak is False
    assert reason == "not_addressed"


def test_no_address_no_trigger():
    speak, reason = _call(text="we should head to the bridge")
    assert speak is False
    assert reason == "not_addressed"


def test_custom_invocation_phrase():
    speak, reason = _call(
        text="handheld, status?",
        invocation_phrases=("artemis", "handheld"),
    )
    assert speak is True
    assert reason == "addressed"


def test_multiword_invocation_phrase():
    speak, reason = _call(
        text="the handheld says...",
        invocation_phrases=("the handheld",),
    )
    assert speak is True
    assert reason == "addressed"


def test_keeper_without_cue_still_needs_name():
    speak, reason = _call(speaker="keeper", text="narration without cue", meta={})
    assert speak is False
    assert reason == "not_addressed"
