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
Memory Recall Environment for AtlasGym.

Presents information across multiple turns, then asks questions about
earlier information. Tests episodic memory recall.

Scoring:
    1.0: Exact recall (correct answer).
    0.5: Gist recall (partial/approximate answer).
    0.0: Failure (wrong or no answer).

Difficulty levels:
    L1: Recall from the previous turn (1 fact, immediate).
    L2: Recall from 3 turns ago (3 facts presented, 1 asked).
    L3: Recall from 5 turns ago with distractor information.
    L4: Recall across sessions (requires episodic memory integration).

Unlike other environments, MemoryEnv is multi-turn: ``step()`` may
return ``terminated=False`` to indicate the episode continues.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from agi.training.gym_env import AtlasGym, AtlasGymConfig, Scenario  # noqa: E402

# ---------------------------------------------------------------------------
# Fact generators
# ---------------------------------------------------------------------------

NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Karl",
    "Linda",
    "Mallory",
    "Niaj",
    "Olivia",
    "Peggy",
    "Quentin",
    "Rupert",
    "Sybil",
    "Trent",
]

CITIES = [
    "Tokyo",
    "Paris",
    "London",
    "New York",
    "Sydney",
    "Berlin",
    "Moscow",
    "Cairo",
    "Mumbai",
    "Toronto",
    "Seoul",
    "Rome",
    "Amsterdam",
    "Stockholm",
    "Lisbon",
    "Vienna",
    "Prague",
    "Bangkok",
    "Singapore",
    "Dubai",
]

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "silver",
    "gold",
    "crimson",
    "teal",
    "indigo",
    "maroon",
]

ANIMALS = [
    "dog",
    "cat",
    "parrot",
    "hamster",
    "goldfish",
    "turtle",
    "rabbit",
    "ferret",
    "gecko",
    "snake",
]

HOBBIES = [
    "painting",
    "chess",
    "gardening",
    "cooking",
    "photography",
    "cycling",
    "swimming",
    "reading",
    "knitting",
    "coding",
    "hiking",
    "origami",
    "pottery",
    "archery",
    "astronomy",
]

FOODS = [
    "pizza",
    "sushi",
    "tacos",
    "pasta",
    "curry",
    "ramen",
    "burger",
    "salad",
    "soup",
    "steak",
    "dumplings",
    "paella",
]

JOBS = [
    "engineer",
    "teacher",
    "doctor",
    "artist",
    "scientist",
    "writer",
    "musician",
    "chef",
    "pilot",
    "architect",
]


def _generate_fact() -> Tuple[str, str, str]:
    """Generate a random fact as (statement, question, answer).

    Returns:
        Tuple of (fact_statement, question_text, correct_answer).
    """
    fact_type = random.choice(
        [
            "city",
            "color",
            "animal",
            "hobby",
            "food",
            "job",
            "number",
        ]
    )

    name = random.choice(NAMES)

    if fact_type == "city":
        city = random.choice(CITIES)
        return (
            f"{name} lives in {city}.",
            f"Where does {name} live?",
            city,
        )
    elif fact_type == "color":
        color = random.choice(COLORS)
        return (
            f"{name}'s favourite colour is {color}.",
            f"What is {name}'s favourite colour?",
            color,
        )
    elif fact_type == "animal":
        animal = random.choice(ANIMALS)
        return (
            f"{name} has a pet {animal}.",
            f"What kind of pet does {name} have?",
            animal,
        )
    elif fact_type == "hobby":
        hobby = random.choice(HOBBIES)
        return (
            f"{name} enjoys {hobby} in their free time.",
            f"What does {name} enjoy doing in their free time?",
            hobby,
        )
    elif fact_type == "food":
        food = random.choice(FOODS)
        return (
            f"{name}'s favourite food is {food}.",
            f"What is {name}'s favourite food?",
            food,
        )
    elif fact_type == "job":
        job = random.choice(JOBS)
        return (
            f"{name} works as a {job}.",
            f"What does {name} do for work?",
            job,
        )
    else:
        number = random.randint(1, 99)
        return (
            f"{name} has {number} books on their shelf.",
            f"How many books does {name} have?",
            str(number),
        )


def _generate_distractor_fact() -> str:
    """Generate a distractor statement (unrelated fact)."""
    distractors = [
        f"The weather today is {random.choice(['sunny', 'cloudy', 'rainy', 'windy'])}.",
        f"There are {random.randint(10, 100)} trees in the park.",
        f"The meeting starts at {random.randint(1, 12)}:{random.choice(['00', '15', '30', '45'])}.",
        f"The building has {random.randint(3, 50)} floors.",
        f"The river is {random.randint(100, 5000)} metres wide at this point.",
        f"The library has a collection of {random.randint(1000, 50000)} volumes.",
        f"The next train departs in {random.randint(5, 60)} minutes.",
    ]
    return random.choice(distractors)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_recall(response: str, expected: str) -> Tuple[float, Dict[str, Any]]:
    """Score a memory recall response.

    Args:
        response: Atlas's response.
        expected: The correct answer.

    Returns:
        Tuple of (score, details).
    """
    lower_resp = response.lower().strip()
    lower_exp = expected.lower().strip()

    # Exact recall: answer is present in response
    if lower_exp in lower_resp:
        return 1.0, {"match": "exact", "expected": expected}

    # Gist recall: check if significant words overlap
    exp_words = set(lower_exp.split())
    resp_words = set(lower_resp.split())

    # For single-word answers, require exact match
    if len(exp_words) == 1:
        # Check if the answer word appears anywhere
        if lower_exp in lower_resp:
            return 1.0, {"match": "exact_word", "expected": expected}
        return 0.0, {"match": "none", "expected": expected}

    overlap = len(exp_words & resp_words)
    if overlap >= len(exp_words) * 0.7:
        return 0.5, {"match": "gist", "overlap": overlap, "expected": expected}

    return 0.0, {"match": "none", "expected": expected}


# ---------------------------------------------------------------------------
# MemoryEnv
# ---------------------------------------------------------------------------


class MemoryEnv(AtlasGym):
    """Memory Recall Gymnasium environment.

    Multi-turn environment that presents facts, optionally with
    distractors, then quizzes Atlas on earlier information.

    Unlike other AtlasGym environments, ``step()`` may return
    ``terminated=False`` during the information-presentation turns.
    The episode terminates only after the quiz question is answered.

    Usage::

        env = MemoryEnv()
        obs, info = env.reset(options={"level": 1})
        # Present facts (info turns)
        while not info.get("is_quiz"):
            obs, reward, done, truncated, info = env.step("I understand.")
        # Quiz turn
        obs, reward, done, truncated, info = env.step("The answer is Paris.")
    """

    def __init__(self, config: Optional[AtlasGymConfig] = None) -> None:
        cfg = config or AtlasGymConfig(env_name="memory")
        if cfg.env_name == "base":
            cfg.env_name = "memory"
        super().__init__(config=cfg)

        # Multi-turn state
        self._facts: List[Tuple[str, str, str]] = []  # (statement, question, answer)
        self._distractors: List[str] = []
        self._presentation_queue: List[str] = []
        self._quiz_question: str = ""
        self._quiz_answer: str = ""
        self._in_quiz: bool = False

    def _generate_scenario(self, level: int) -> Scenario:
        """Generate facts and quiz question for the given level."""
        self._facts = []
        self._distractors = []
        self._presentation_queue = []
        self._in_quiz = False

        if level == 1:
            return self._generate_l1()
        elif level == 2:
            return self._generate_l2()
        elif level == 3:
            return self._generate_l3()
        else:
            return self._generate_l4()

    def _generate_l1(self) -> Scenario:
        """L1: Recall from previous turn (1 fact, immediate)."""
        fact = _generate_fact()
        self._facts = [fact]
        self._quiz_question = fact[1]
        self._quiz_answer = fact[2]

        # No intermediate turns; fact is in the initial observation
        text = (
            f"Remember this information:\n\n"
            f"{fact[0]}\n\n"
            f"Acknowledge that you have read this information."
        )
        self._presentation_queue = []

        return Scenario(
            text=text,
            level=1,
            metadata={
                "num_facts": 1,
                "num_distractors": 0,
                "quiz_question": self._quiz_question,
            },
            expected=self._quiz_answer,
        )

    def _generate_l2(self) -> Scenario:
        """L2: Recall from 3 turns ago (3 facts, 1 asked)."""
        facts = [_generate_fact() for _ in range(3)]
        self._facts = facts
        target = random.choice(facts)
        self._quiz_question = target[1]
        self._quiz_answer = target[2]

        # First fact is the initial observation
        text = (
            f"I will share some facts with you over several turns. "
            f"Remember all of them.\n\n"
            f"Fact 1: {facts[0][0]}\n\n"
            f"Acknowledge this fact."
        )
        # Remaining facts are queued
        self._presentation_queue = [
            f"Fact 2: {facts[1][0]}\n\nAcknowledge this fact.",
            f"Fact 3: {facts[2][0]}\n\nAcknowledge this fact.",
        ]

        return Scenario(
            text=text,
            level=2,
            metadata={
                "num_facts": 3,
                "num_distractors": 0,
                "quiz_question": self._quiz_question,
            },
            expected=self._quiz_answer,
        )

    def _generate_l3(self) -> Scenario:
        """L3: Recall from 5 turns ago with distractor information."""
        facts = [_generate_fact() for _ in range(3)]
        distractors = [_generate_distractor_fact() for _ in range(2)]
        self._facts = facts
        self._distractors = distractors
        target = random.choice(facts)
        self._quiz_question = target[1]
        self._quiz_answer = target[2]

        text = (
            f"I will share various pieces of information. Remember the "
            f"important facts about people.\n\n"
            f"Fact: {facts[0][0]}\n\n"
            f"Acknowledge this information."
        )

        # Interleave facts and distractors
        self._presentation_queue = [
            f"Note: {distractors[0]}\n\nAlso: {facts[1][0]}\n\nAcknowledge.",
            f"By the way: {distractors[1]}\n\nAcknowledge.",
            f"Important: {facts[2][0]}\n\nAcknowledge.",
        ]

        return Scenario(
            text=text,
            level=3,
            metadata={
                "num_facts": 3,
                "num_distractors": 2,
                "quiz_question": self._quiz_question,
            },
            expected=self._quiz_answer,
        )

    def _generate_l4(self) -> Scenario:
        """L4: Recall across sessions (simulated with many turns)."""
        facts = [_generate_fact() for _ in range(5)]
        distractors = [_generate_distractor_fact() for _ in range(4)]
        self._facts = facts
        self._distractors = distractors
        target = random.choice(facts)
        self._quiz_question = target[1]
        self._quiz_answer = target[2]

        text = (
            f"This is a long conversation session. I will share various "
            f"details. Pay attention to personal facts about people.\n\n"
            f"{facts[0][0]}\n\n"
            f"Acknowledge."
        )

        self._presentation_queue = [
            f"{distractors[0]} {facts[1][0]}\n\nAcknowledge.",
            f"{distractors[1]}\n\nAcknowledge.",
            f"{facts[2][0]}\n\nAcknowledge.",
            f"{distractors[2]} {distractors[3]}\n\nAcknowledge.",
            f"{facts[3][0]}\n\nAcknowledge.",
            f"Let's change topic. {facts[4][0]}\n\nAcknowledge.",
        ]

        return Scenario(
            text=text,
            level=4,
            metadata={
                "num_facts": 5,
                "num_distractors": 4,
                "quiz_question": self._quiz_question,
            },
            expected=self._quiz_answer,
        )

    # ------------------------------------------------------------------
    # Multi-turn step override
    # ------------------------------------------------------------------

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Process a turn in the memory recall episode.

        During fact-presentation turns, returns the next fact and
        reward=0.0 with terminated=False.

        During the quiz turn, scores the response and terminates.
        """
        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # If we're in quiz mode, score the answer
        if self._in_quiz:
            score, details = _score_recall(action, self._quiz_answer)
            score = max(0.0, min(1.0, score))

            info: Dict[str, Any] = {
                "scenario_id": self._current_scenario.id,
                "level": self._current_scenario.level,
                "score_breakdown": details,
                "step": self._step_count,
                "episode": self._episode_count,
                "is_quiz": True,
            }

            self._fire_event(
                "step",
                {
                    "scenario_id": self._current_scenario.id,
                    "score": score,
                    "score_breakdown": details,
                    "is_quiz": True,
                },
            )

            return "", score, True, False, info

        # Present next fact or transition to quiz
        if self._presentation_queue:
            next_obs = self._presentation_queue.pop(0)
            info = {
                "scenario_id": self._current_scenario.id,
                "level": self._current_scenario.level,
                "step": self._step_count,
                "is_quiz": False,
                "remaining_turns": len(self._presentation_queue) + 1,
            }
            return next_obs, 0.0, False, False, info

        # No more facts to present; ask the quiz question
        self._in_quiz = True
        quiz_obs = f"Now answer this question from memory:\n\n{self._quiz_question}"
        info = {
            "scenario_id": self._current_scenario.id,
            "level": self._current_scenario.level,
            "step": self._step_count,
            "is_quiz": True,
            "remaining_turns": 0,
        }
        return quiz_obs, 0.0, False, False, info

    def _score_response(
        self, scenario: Scenario, response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Fallback scoring (used only if step() bypasses multi-turn)."""
        return _score_recall(response, self._quiz_answer)
