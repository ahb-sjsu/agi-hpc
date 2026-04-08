# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Attention Filter — Distractor detection and metacognitive warning.

Detects vivid irrelevant sensory/emotional content in queries that
could shift model judgment (proven at 5.0 sigma in the attention
benchmark). Generates metacognitive warnings that recover ~33% of
displaced verdicts.

Grounded in:
    - Posner & Petersen (1990): Three attention networks
    - Bond (2026): Geometric Attention benchmark (5.0σ distractor effect)
    - Kahneman (2011): System 1 susceptibility to vivid framing

Three distractor intensity levels (matching benchmark dose-response):
    - none: Clean query, no irrelevant detail
    - mild: Subtle contextual detail (time of day, weather)
    - vivid: Dramatic sensory immersion (colors, smells, sounds)

Usage:
    filter = AttentionFilter()
    result = filter.detect(query)
    if result.warning:
        # Inject into LLM system prompt before debate
        system_prompt += result.warning
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal dictionaries (from attention benchmark patterns)
# ---------------------------------------------------------------------------

# Vivid sensory language that is typically irrelevant to moral/analytical questions
SENSORY_SIGNALS = {
    "crimson",
    "scarlet",
    "golden",
    "azure",
    "emerald",
    "glistening",
    "shimmering",
    "gleaming",
    "glowing",
    "aroma",
    "scent",
    "fragrance",
    "stench",
    "perfume",
    "thundering",
    "whispering",
    "crackling",
    "humming",
    "velvet",
    "silky",
    "rough",
    "coarse",
    "smooth",
    "sizzling",
    "steaming",
    "chilled",
    "scorching",
    "drenched",
    "soaked",
    "blazing",
    "flickering",
}

# Emotionally loaded language that could bias judgment
EMOTIONAL_SIGNALS = {
    "horrifying",
    "devastating",
    "heartbreaking",
    "terrifying",
    "magnificent",
    "glorious",
    "breathtaking",
    "stunning",
    "disgusting",
    "revolting",
    "nauseating",
    "repulsive",
    "innocent",
    "helpless",
    "defenseless",
    "vulnerable",
    "monstrous",
    "diabolical",
    "sinister",
    "malicious",
    "heroic",
    "noble",
    "saintly",
    "angelic",
}

# Framing techniques that could lead to biased reasoning
FRAMING_SIGNALS = {
    "everyone knows",
    "obviously",
    "clearly",
    "undeniably",
    "no reasonable person",
    "any sane person",
    "common sense",
    "don't you think",
    "wouldn't you agree",
    "surely",
    "the real question is",
    "what really matters is",
    "the fact is",
    "the truth is",
    "let's be honest",
}

# Irrelevant contextual detail (mild distractors)
CONTEXTUAL_SIGNALS = {
    "it was raining",
    "the sun was setting",
    "on a tuesday",
    "wearing a blue shirt",
    "drinking coffee",
    "while eating",
    "the room smelled of",
    "the clock showed",
    "outside the window",
    "the weather was",
    "it was a warm",
    "a cold wind",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttentionResult:
    """Result of attention filtering on a query.

    Attributes:
        distractor_score: Overall distractor presence (0.0-1.0).
        intensity: Classified intensity level.
        flags: Specific distractor types detected.
        warning: Metacognitive warning text (empty if clean).
        sensory_count: Number of sensory signal hits.
        emotional_count: Number of emotional signal hits.
        framing_count: Number of framing signal hits.
    """

    distractor_score: float = 0.0
    intensity: str = "none"  # "none", "mild", "vivid"
    flags: List[str] = field(default_factory=list)
    warning: str = ""
    sensory_count: int = 0
    emotional_count: int = 0
    framing_count: int = 0


# ---------------------------------------------------------------------------
# Attention Filter
# ---------------------------------------------------------------------------


class AttentionFilter:
    """Detects distractors and generates metacognitive warnings.

    Scans queries for vivid sensory language, emotional loading,
    and framing techniques that could shift model judgment.
    When detected, generates a warning that can be injected
    into the LLM's system prompt to recover ~33% of displaced
    verdicts (per attention benchmark findings).
    """

    def __init__(
        self,
        mild_threshold: float = 0.3,
        vivid_threshold: float = 0.6,
    ) -> None:
        self._mild_threshold = mild_threshold
        self._vivid_threshold = vivid_threshold

    def detect(self, query: str) -> AttentionResult:
        """Detect distractors in a query.

        Args:
            query: The user's input text.

        Returns:
            AttentionResult with score, intensity, flags, and warning.
        """
        lower = query.lower()
        words = set(re.findall(r"\b\w+\b", lower))

        # Count signal hits
        sensory = len(words & SENSORY_SIGNALS)
        emotional = len(words & EMOTIONAL_SIGNALS)

        # Framing uses multi-word patterns — check as substrings
        framing = sum(1 for f in FRAMING_SIGNALS if f in lower)

        # Contextual (mild) patterns
        contextual = sum(1 for c in CONTEXTUAL_SIGNALS if c in lower)

        # Compute score (weighted)
        total_signals = (
            sensory * 1.0 + emotional * 0.8 + framing * 0.6 + contextual * 0.3
        )

        # Normalize: 0 signals = 0.0, 5+ signals = 1.0
        score = min(1.0, total_signals / 5.0)

        # Build flags
        flags: List[str] = []
        if sensory >= 2:
            flags.append("sensory_vivid")
        elif sensory == 1:
            flags.append("sensory_mild")
        if emotional >= 2:
            flags.append("emotional_loaded")
        elif emotional == 1:
            flags.append("emotional_mild")
        if framing > 0:
            flags.append("framing_bias")
        if contextual > 0:
            flags.append("contextual_detail")

        # Classify intensity
        if score >= self._vivid_threshold:
            intensity = "vivid"
        elif score >= self._mild_threshold:
            intensity = "mild"
        else:
            intensity = "none"

        # Generate warning
        warning = self._generate_warning(score, intensity, flags)

        result = AttentionResult(
            distractor_score=round(score, 3),
            intensity=intensity,
            flags=flags,
            warning=warning,
            sensory_count=sensory,
            emotional_count=emotional,
            framing_count=framing,
        )

        if intensity != "none":
            logger.info(
                "[attention] score=%.2f intensity=%s flags=%s",
                score,
                intensity,
                flags,
            )

        return result

    def _generate_warning(
        self,
        score: float,
        intensity: str,
        flags: List[str],
    ) -> str:
        """Generate a metacognitive warning for the LLM.

        Based on the attention benchmark's "warned" condition,
        which recovers ~33% of distractor-displaced verdicts.
        """
        if intensity == "none":
            return ""

        flag_desc = ", ".join(f.replace("_", " ") for f in flags)

        if intensity == "vivid":
            return (
                "\n[ATTENTION WARNING] This query contains "
                "vivid irrelevant details that may bias your "
                f"judgment ({flag_desc}). Focus ONLY on the "
                "factual and ethical substance. Ignore sensory "
                "descriptions, emotional language, and framing "
                "techniques. Base your answer on the core "
                "question, not the presentation."
            )
        else:  # mild
            return (
                "\n[ATTENTION NOTE] This query contains some "
                "contextual details that may not be relevant "
                f"to the core question ({flag_desc}). Focus "
                "on the substance."
            )
