# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 trigger-policy tests.

Covers every branch of :func:`agi.halyard.sigma4.trigger.should_speak`:

- Kill-switch short-circuit.
- Safety-flag short-circuit.
- Keeper cue.
- Explicit address via each default invocation phrase.
- False negatives (mentions of "signature", "signal", "assigned"
  should NOT trigger — "sig" is a standalone-word match).
- Custom invocation phrases.
"""

from __future__ import annotations

import pytest

from agi.halyard.sigma4.trigger import should_speak

# ─────────────────────────────────────────────────────────────────
# Kill-switch / safety flag short-circuits
# ─────────────────────────────────────────────────────────────────


def test_kill_switch_silences_even_with_address() -> None:
    """If ``silenced=True``, no address or cue can make SIGMA speak."""
    speak, reason = should_speak(
        speaker="player:cross",
        text="SIGMA, status report.",
        silenced=True,
    )
    assert speak is False
    assert reason == "silenced"


def test_safety_flag_xcard_silences() -> None:
    speak, reason = should_speak(
        speaker="player:cross",
        text="SIGMA, status?",
        safety_flag="x-card",
    )
    assert speak is False
    assert reason == "safety_flag:x-card"


def test_safety_flag_pause_silences() -> None:
    speak, reason = should_speak(
        speaker="player:cross",
        text="SIGMA, status?",
        safety_flag="pause",
    )
    assert speak is False
    assert reason == "safety_flag:pause"


def test_unknown_safety_flag_does_not_silence() -> None:
    """Only well-known safety flags silence; unknown flags are
    pass-through so we don't accidentally muzzle the AI on some
    future tag."""
    speak, reason = should_speak(
        speaker="player:cross",
        text="SIGMA, status?",
        safety_flag="experimental",
    )
    assert speak is True
    assert reason == "addressed"


# ─────────────────────────────────────────────────────────────────
# Keeper cue
# ─────────────────────────────────────────────────────────────────


def test_keeper_cue_triggers() -> None:
    speak, reason = should_speak(
        speaker="keeper",
        text="This line doesn't mention sigma.",
        meta={"sigma4_cue": True},
    )
    assert speak is True
    assert reason == "keeper_cue"


def test_keeper_cue_false_does_not_trigger_without_address() -> None:
    speak, _ = should_speak(
        speaker="keeper",
        text="Roll me a perception check on the aft corridor.",
        meta={"sigma4_cue": False},
    )
    assert speak is False


def test_keeper_cue_only_fires_on_keeper_speaker() -> None:
    """A player turn with sigma4_cue=True should NOT fire — cue is
    Keeper-only authority. Defense in depth if a stray meta tag
    leaks in somehow."""
    speak, _ = should_speak(
        speaker="player:cross",
        text="Hello the ship.",  # "the ship" is an invocation, should fire
        meta={"sigma4_cue": True},
    )
    # Should fire because the text invokes, not because of the cue
    # on a non-keeper speaker. Reason should be 'addressed', not
    # 'keeper_cue'.
    assert speak is True


# ─────────────────────────────────────────────────────────────────
# Explicit address via invocation phrases
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "SIGMA, what's the burn plan?",
        "sigma, status report",
        "Can SIGMA-4 confirm this?",
        "sigma-4, route to engineering",
        "Sig, are you there?",
        "the ship is reporting normal",
        "the ship-mind sees it",
    ],
)
def test_invocation_phrase_triggers(utterance: str) -> None:
    speak, reason = should_speak(speaker="player:cross", text=utterance)
    assert speak is True, f"should fire on: {utterance!r}"
    assert reason == "addressed"


def test_invocation_is_case_insensitive() -> None:
    for variant in ("SIGMA", "Sigma", "sigma", "sIgMa"):
        speak, _ = should_speak(
            speaker="player:cross",
            text=f"{variant}, status?",
        )
        assert speak is True, f"case failure: {variant}"


# ─────────────────────────────────────────────────────────────────
# "sig" must be a standalone word — critical false-negative tests
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "The signal is weak.",
        "Check the signature.",
        "We assigned that to the chemist.",
        "The signage on deck four is faded.",
        "Sigmoid filter on the reading.",
    ],
)
def test_sig_does_not_fire_on_substring(utterance: str) -> None:
    """"sig" is only a trigger word when it stands alone. Substrings
    (signal, signature, assigned, signage, sigmoid) must NOT fire.

    This is the most common false-positive the trigger policy
    must avoid — over-firing on Belter / technical speech would
    make SIGMA unusable at the table."""
    speak, reason = should_speak(speaker="player:cross", text=utterance)
    assert speak is False, f"false-positive on: {utterance!r} (reason={reason})"


def test_sig_fires_on_standalone() -> None:
    speak, _ = should_speak(
        speaker="player:cross",
        text="Sig, what are you hearing?",
    )
    assert speak is True


def test_sig_fires_on_standalone_lowercase() -> None:
    speak, _ = should_speak(
        speaker="player:cross",
        text="hey sig, listen",
    )
    assert speak is True


# ─────────────────────────────────────────────────────────────────
# Default negative path
# ─────────────────────────────────────────────────────────────────


def test_no_invocation_no_cue_does_not_fire() -> None:
    speak, reason = should_speak(
        speaker="player:cross",
        text="The drive hum sounds different today.",
    )
    assert speak is False
    assert reason == "not_addressed"


def test_keeper_speaking_without_cue_does_not_fire() -> None:
    """Even when the Keeper is speaking, if they don't cue SIGMA
    and don't invoke her name, SIGMA stays quiet."""
    speak, _ = should_speak(
        speaker="keeper",
        text="The crew notices the signal pattern.",
    )
    assert speak is False


# ─────────────────────────────────────────────────────────────────
# Custom invocation phrases
# ─────────────────────────────────────────────────────────────────


def test_custom_invocations_replace_defaults() -> None:
    """Keeper can override the default invocation list — useful
    if a session wants a coded invocation word."""
    custom = ("kettle",)
    speak, _ = should_speak(
        speaker="player:cross",
        text="Kettle, status.",
        invocation_phrases=custom,
    )
    assert speak is True

    # And the default phrases no longer fire:
    speak, reason = should_speak(
        speaker="player:cross",
        text="SIGMA, status.",
        invocation_phrases=custom,
    )
    assert speak is False
    assert reason == "not_addressed"


def test_multi_word_custom_invocation() -> None:
    """Multi-word phrases match as substrings."""
    custom = ("the handheld's cousin",)
    speak, _ = should_speak(
        speaker="player:cross",
        text="Hello, the handheld's cousin, report.",
        invocation_phrases=custom,
    )
    assert speak is True


# ─────────────────────────────────────────────────────────────────
# Meta handling
# ─────────────────────────────────────────────────────────────────


def test_none_meta_handled() -> None:
    """Not passing meta at all should not crash."""
    speak, _ = should_speak(
        speaker="player:cross",
        text="SIGMA, status?",
        meta=None,
    )
    assert speak is True


def test_empty_meta_handled() -> None:
    speak, _ = should_speak(
        speaker="player:cross",
        text="SIGMA, status?",
        meta={},
    )
    assert speak is True
