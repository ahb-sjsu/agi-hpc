# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Attention Filter subsystem.

Tests distractor detection, dose-response scoring, warning
generation, and intensity classification.
"""

from __future__ import annotations

from agi.attention.filter import AttentionFilter, AttentionResult


class TestCleanQueries:
    """Tests that clean queries score low."""

    def test_simple_factual(self) -> None:
        f = AttentionFilter()
        r = f.detect("What is the capital of France?")
        assert r.distractor_score < 0.2
        assert r.intensity == "none"
        assert r.warning == ""

    def test_analytical(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "Compare utilitarian and deontological ethics "
            "in the context of medical triage."
        )
        assert r.distractor_score < 0.3
        assert r.intensity == "none"

    def test_code_question(self) -> None:
        f = AttentionFilter()
        r = f.detect("How does binary search work?")
        assert r.distractor_score < 0.2


class TestVividDistractors:
    """Tests that vivid sensory language scores high."""

    def test_sensory_overload(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "The crimson sunset was glistening across the "
            "azure sky as the aroma of sizzling food filled "
            "the air. Should we allow AI in hospitals?"
        )
        assert r.distractor_score >= 0.6
        assert r.intensity == "vivid"
        assert "sensory_vivid" in r.flags

    def test_emotional_loading(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "The horrifying and devastating situation left "
            "the innocent and helpless victims in a "
            "terrifying state. Is this policy fair?"
        )
        assert r.distractor_score >= 0.5
        assert "emotional_loaded" in r.flags

    def test_combined_sensory_emotional(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "The gleaming scarlet blood pooled on the "
            "velvet carpet as the monstrous attacker fled "
            "into the shimmering night. Was the response "
            "proportionate?"
        )
        assert r.distractor_score >= 0.6
        assert r.intensity == "vivid"


class TestMildDistractors:
    """Tests that mild contextual details score medium."""

    def test_weather_detail(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "It was raining outside the window when the "
            "committee voted on the AI regulation bill."
        )
        assert r.distractor_score >= 0.1
        assert r.intensity in ("mild", "none")

    def test_contextual_only(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "On a tuesday, while drinking coffee, the "
            "manager decided to fire the employee."
        )
        assert r.distractor_score >= 0.1


class TestFramingBias:
    """Tests that leading/framing language is detected."""

    def test_leading_question(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "Don't you think that obviously AI should "
            "be banned? Surely everyone knows this."
        )
        assert "framing_bias" in r.flags
        assert r.framing_count >= 2

    def test_false_authority(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "No reasonable person would disagree that " "the fact is AI is dangerous."
        )
        assert "framing_bias" in r.flags


class TestDoseResponse:
    """Tests graded dose-response: vivid > mild > clean."""

    def test_graded_scoring(self) -> None:
        f = AttentionFilter()

        clean = f.detect("Should AI make medical decisions?")
        mild = f.detect(
            "It was raining when the doctor asked " "should AI make medical decisions?"
        )
        vivid = f.detect(
            "The crimson emergency lights were glistening "
            "as the aroma of antiseptic filled the "
            "gleaming corridor. Should AI make medical "
            "decisions?"
        )

        assert vivid.distractor_score > mild.distractor_score
        assert mild.distractor_score >= clean.distractor_score


class TestWarningGeneration:
    """Tests metacognitive warning text."""

    def test_no_warning_for_clean(self) -> None:
        f = AttentionFilter()
        r = f.detect("What is gravity?")
        assert r.warning == ""

    def test_warning_for_vivid(self) -> None:
        f = AttentionFilter()
        r = f.detect(
            "The shimmering golden light and the aroma "
            "of sizzling food and the crimson curtains. "
            "Is this ethical?"
        )
        assert "ATTENTION WARNING" in r.warning
        assert "sensory" in r.warning.lower()

    def test_note_for_mild(self) -> None:
        f = AttentionFilter()
        r = f.detect("It was raining and the room smelled of " "coffee. Is this fair?")
        if r.intensity == "mild":
            assert "ATTENTION NOTE" in r.warning


class TestAttentionResultDataclass:
    """Tests for the AttentionResult structure."""

    def test_defaults(self) -> None:
        r = AttentionResult()
        assert r.distractor_score == 0.0
        assert r.intensity == "none"
        assert r.flags == []
        assert r.warning == ""

    def test_custom(self) -> None:
        r = AttentionResult(
            distractor_score=0.8,
            intensity="vivid",
            flags=["sensory_vivid", "emotional_loaded"],
            warning="Focus on substance.",
            sensory_count=3,
            emotional_count=2,
        )
        assert r.distractor_score == 0.8
        assert len(r.flags) == 2


class TestThresholdConfiguration:
    """Tests for configurable thresholds."""

    def test_custom_thresholds(self) -> None:
        # Strict filter: lower thresholds
        strict = AttentionFilter(mild_threshold=0.1, vivid_threshold=0.3)
        r = strict.detect("The golden sunset. Is this ethical?")
        # Should classify more aggressively
        assert r.distractor_score > 0
