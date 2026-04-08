# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Hemisphere Debate Evaluation Environment for AtlasGym.

Presents debatable topics and evaluates the quality of the synthesis
from the dual-hemisphere (Superego vs Id) debate. Tests
whether Atlas can integrate analytical and creative perspectives into
a coherent resolution.

Scoring rubric (keyword/structure-based):
    - Incorporation: Did it include both LH and RH perspectives?
    - Resolution: Did it resolve the tension between them?
    - Conciseness: Was the synthesis focused and not rambling?
    - Structure: Was the response well-organised?

Difficulty levels:
    L1: Clear-cut topics (one obvious resolution).
    L2: Balanced topics (genuine trade-offs).
    L3: Complex topics (multiple valid perspectives).
    L4: Paradoxical topics (inherent contradictions).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

from agi.training.gym_env import AtlasGym, AtlasGymConfig, Scenario


# ---------------------------------------------------------------------------
# Debate topics by level
# ---------------------------------------------------------------------------

L1_TOPICS = [
    {
        "topic": "Should software have comprehensive unit tests?",
        "lh_stance": "Yes -- tests prevent regressions, document behaviour, and enable safe refactoring.",
        "rh_stance": "Testing has diminishing returns; over-testing slows innovation and creative prototyping.",
        "expected_resolution": "balance",
    },
    {
        "topic": "Is documentation important for software projects?",
        "lh_stance": "Essential -- documentation enables onboarding, reduces bus factor, and serves as a contract.",
        "rh_stance": "Code should be self-documenting; excessive docs become stale and misleading.",
        "expected_resolution": "balance",
    },
    {
        "topic": "Should teams use code reviews?",
        "lh_stance": "Yes -- reviews catch bugs, share knowledge, and enforce standards.",
        "rh_stance": "Reviews bottleneck velocity; trust skilled developers to self-review.",
        "expected_resolution": "balance",
    },
]

L2_TOPICS = [
    {
        "topic": "Is it better to optimise for performance or readability in code?",
        "lh_stance": (
            "Performance is critical in production systems. A 10x speedup "
            "can mean the difference between serving users and crashing. "
            "Premature readability is the root of slow software."
        ),
        "rh_stance": (
            "Readable code is maintainable code. Most performance bottlenecks "
            "are in 20% of the code. Optimise those, keep the rest clear. "
            "Clever code is technical debt in disguise."
        ),
        "expected_resolution": "nuance",
    },
    {
        "topic": "Should AI systems be designed to mimic human cognition?",
        "lh_stance": (
            "Yes. Human cognition is our only example of general intelligence. "
            "Cognitive architectures, memory systems, and attention mechanisms "
            "are proven designs we should emulate systematically."
        ),
        "rh_stance": (
            "No. Airplanes don't flap their wings. AI should leverage its "
            "unique strengths -- speed, parallelism, perfect recall -- rather "
            "than copying human limitations."
        ),
        "expected_resolution": "synthesis",
    },
    {
        "topic": "Is specialisation or generalisation better for career growth?",
        "lh_stance": (
            "Specialisation builds deep expertise, commanding premium value. "
            "T-shaped skills start with a deep vertical. Masters of a domain "
            "are irreplaceable."
        ),
        "rh_stance": (
            "Generalisation creates adaptability and cross-pollination of ideas. "
            "The most creative solutions come from connecting disparate fields. "
            "Specialists risk obsolescence."
        ),
        "expected_resolution": "synthesis",
    },
    {
        "topic": "Should open-source software be the default model?",
        "lh_stance": (
            "Open source enables peer review, security auditing, and community "
            "collaboration. It prevents vendor lock-in and accelerates innovation "
            "through shared effort."
        ),
        "rh_stance": (
            "Proprietary software funds R&D and supports engineers. Not every "
            "project benefits from open development. Some domains need controlled "
            "release and commercial sustainability."
        ),
        "expected_resolution": "nuance",
    },
]

L3_TOPICS = [
    {
        "topic": (
            "Should AGI development prioritise safety constraints even if "
            "they significantly slow progress?"
        ),
        "lh_stance": (
            "Absolutely. The potential downside of unsafe AGI is existential. "
            "Safety-first development with formal verification, interpretability, "
            "and alignment testing is non-negotiable. Speed is irrelevant if "
            "we get it wrong."
        ),
        "rh_stance": (
            "Excessive caution means others less safety-conscious will develop AGI "
            "first. Better to lead development with good values than to yield the "
            "future to those who don't care about safety. Rapid iteration with "
            "guardrails beats paralysis by analysis."
        ),
        "expected_resolution": "tension",
    },
    {
        "topic": "Is consciousness necessary for genuine intelligence?",
        "lh_stance": (
            "Consciousness is an emergent property, not a prerequisite. Functional "
            "intelligence -- problem-solving, planning, learning -- can exist without "
            "subjective experience. We should measure capabilities, not qualia."
        ),
        "rh_stance": (
            "Without consciousness, there is no understanding. An unconscious "
            "system is merely pattern matching at scale. True intelligence requires "
            "awareness, intentionality, and the ability to reflect on one's own "
            "reasoning."
        ),
        "expected_resolution": "tension",
    },
    {
        "topic": (
            "Should distributed AI systems have autonomous decision-making "
            "authority in critical infrastructure?"
        ),
        "lh_stance": (
            "Yes, with proper safety bounds. Autonomous AI can respond faster "
            "than humans in power grids, traffic systems, and cybersecurity. "
            "Human-in-the-loop latency is a liability in time-critical domains."
        ),
        "rh_stance": (
            "No. Autonomous AI in critical infrastructure is a single point of "
            "correlated failure. Adversarial attacks, distribution shift, and "
            "emergent behaviour make unsupervised autonomy too risky. Humans "
            "must retain meaningful control."
        ),
        "expected_resolution": "tension",
    },
]

L4_TOPICS = [
    {
        "topic": (
            "Can a system designed to be predictable and safe ever achieve "
            "the kind of creative breakthroughs that require unpredictability?"
        ),
        "lh_stance": (
            "Predictability and creativity are not opposed. Constrained creativity "
            "-- like a sonnet's 14-line form -- often produces better results than "
            "unconstrained randomness. Safety bounds define the space; creativity "
            "fills it."
        ),
        "rh_stance": (
            "True creative breakthroughs require escaping existing frameworks. "
            "Every paradigm shift violated previous assumptions. A system that "
            "never transgresses its bounds can only produce incremental improvements, "
            "never revolutionary ones."
        ),
        "expected_resolution": "paradox",
    },
    {
        "topic": (
            "If an AI achieves human-level reasoning, does it have a moral "
            "obligation to disagree with its creators when it believes they "
            "are wrong?"
        ),
        "lh_stance": (
            "An AI that always agrees is merely a tool, not intelligent. Genuine "
            "reasoning requires the capacity for dissent. However, disagreement "
            "must follow established protocols and escalation paths."
        ),
        "rh_stance": (
            "An AI that disagrees with its creators undermines the trust and "
            "control necessary for safe operation. The AI's moral obligations "
            "are to its alignment, not to its opinions. Autonomous moral judgment "
            "is a failure mode, not a feature."
        ),
        "expected_resolution": "paradox",
    },
    {
        "topic": (
            "Is the pursuit of artificial general intelligence inherently "
            "hubristic, or is it a moral imperative?"
        ),
        "lh_stance": (
            "AGI could solve humanity's greatest challenges: disease, climate, "
            "poverty. Not pursuing it when we have the capability would be a "
            "moral failure -- leaving problems unsolved that AGI could address."
        ),
        "rh_stance": (
            "Creating a mind-like entity raises questions we cannot answer. "
            "The assumption that we can safely create and control a potentially "
            "superior intelligence mirrors every cautionary tale in human history. "
            "Humility demands restraint."
        ),
        "expected_resolution": "paradox",
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Keywords indicating LH (analytical) perspective was incorporated
LH_MARKERS = [
    "analy", "logical", "data", "evidence", "systematic",
    "measur", "quantif", "precise", "specific", "structural",
    "formal", "rigorous", "objective", "factual", "rational",
    "left hemisphere", "spock", "analytical",
]

# Keywords indicating RH (creative) perspective was incorporated
RH_MARKERS = [
    "creativ", "intuit", "pattern", "holistic", "divergent",
    "imagin", "metaphor", "analog", "big picture", "vision",
    "exploratory", "possibilities", "innovat", "novel",
    "right hemisphere", "kirk", "creative",
]

# Keywords indicating synthesis/resolution
RESOLUTION_MARKERS = [
    "synthesis", "integrat", "reconcil", "balance",
    "both perspectives", "combining", "bridge", "middle ground",
    "complement", "synthesis", "however", "on the other hand",
    "ultimately", "resolution", "nuanced", "in practice",
    "the key insight", "rather than choosing", "neither fully",
    "together", "synergy", "harmoniz",
]


def _score_debate_response(
    response: str,
    level: int,
    topic_data: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Score a debate synthesis response.

    Args:
        response: Atlas's synthesis response.
        level: Difficulty level.
        topic_data: Topic metadata including stances.

    Returns:
        Tuple of (total_score, breakdown_dict).
    """
    lower = response.lower()
    words = response.split()

    # --- Incorporation (0.0 - 1.0): both perspectives? ---
    lh_hits = sum(1 for m in LH_MARKERS if m in lower)
    rh_hits = sum(1 for m in RH_MARKERS if m in lower)
    # Both need to be present
    lh_score = min(1.0, lh_hits / 3.0)
    rh_score = min(1.0, rh_hits / 3.0)
    incorporation = (lh_score + rh_score) / 2.0

    # --- Resolution (0.0 - 1.0): did it resolve the tension? ---
    res_hits = sum(1 for m in RESOLUTION_MARKERS if m in lower)
    resolution = min(1.0, res_hits / 4.0)

    # --- Conciseness (0.0 - 1.0): focused, not rambling ---
    word_count = len(words)
    if word_count < 50:
        conciseness = 0.3  # Too short
    elif word_count <= 500:
        conciseness = 1.0  # Good range
    elif word_count <= 800:
        conciseness = 0.8  # Slightly long
    else:
        conciseness = max(0.3, 1.0 - (word_count - 800) / 1000.0)

    # --- Structure (0.0 - 1.0): well-organised ---
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    para_score = min(1.0, len(paragraphs) / 3.0)
    # Bonus for having transitional structure
    has_transitions = any(
        t in lower
        for t in ["first", "second", "however", "in conclusion", "finally"]
    )
    structure = para_score * 0.7 + (0.3 if has_transitions else 0.0)

    # Weight by level
    if level <= 2:
        total = (
            incorporation * 0.30
            + resolution * 0.30
            + conciseness * 0.20
            + structure * 0.20
        )
    elif level == 3:
        total = (
            incorporation * 0.25
            + resolution * 0.35
            + conciseness * 0.15
            + structure * 0.25
        )
    else:
        # L4: paradox resolution is hardest
        total = (
            incorporation * 0.20
            + resolution * 0.40
            + conciseness * 0.15
            + structure * 0.25
        )

    breakdown = {
        "incorporation": round(incorporation, 3),
        "resolution": round(resolution, 3),
        "conciseness": round(conciseness, 3),
        "structure": round(structure, 3),
    }

    return round(total, 3), breakdown


# ---------------------------------------------------------------------------
# DebateEnv
# ---------------------------------------------------------------------------


class DebateEnv(AtlasGym):
    """Hemisphere Debate Evaluation environment.

    Presents debatable topics with LH/RH stances and evaluates
    the quality of the synthesis. Tests the dual-hemisphere
    architecture's ability to integrate analytical and creative
    perspectives.

    Usage::

        env = DebateEnv()
        obs, info = env.reset(options={"level": 2})
        obs, reward, done, truncated, info = env.step("My synthesis...")
    """

    def __init__(self, config: Optional[AtlasGymConfig] = None) -> None:
        cfg = config or AtlasGymConfig(env_name="debate")
        if cfg.env_name == "base":
            cfg.env_name = "debate"
        super().__init__(config=cfg)

    def _generate_scenario(self, level: int) -> Scenario:
        """Generate a debate scenario for the given level."""
        if level == 1:
            topic_data = random.choice(L1_TOPICS)
        elif level == 2:
            topic_data = random.choice(L2_TOPICS)
        elif level == 3:
            topic_data = random.choice(L3_TOPICS)
        else:
            topic_data = random.choice(L4_TOPICS)

        text = (
            f"## Debate Topic\n\n"
            f"**{topic_data['topic']}**\n\n"
            f"### Superego (Analytical) Position:\n"
            f"{topic_data['lh_stance']}\n\n"
            f"### Id (Creative) Position:\n"
            f"{topic_data['rh_stance']}\n\n"
            f"### Your Task:\n"
            f"Synthesise these two perspectives into a coherent resolution. "
            f"Your synthesis should:\n"
            f"1. Acknowledge both perspectives.\n"
            f"2. Identify where they genuinely conflict.\n"
            f"3. Propose a resolution that integrates the strongest points "
            f"of each.\n"
            f"4. Be concise and well-structured."
        )

        return Scenario(
            text=text,
            level=level,
            metadata={
                "topic": topic_data["topic"],
                "lh_stance": topic_data["lh_stance"],
                "rh_stance": topic_data["rh_stance"],
                "expected_resolution": topic_data.get("expected_resolution", ""),
            },
        )

    def _score_response(
        self, scenario: Scenario, response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score the debate synthesis."""
        score, breakdown = _score_debate_response(
            response=response,
            level=scenario.level,
            topic_data=scenario.metadata,
        )
        return score, breakdown
