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
Logical Reasoning Environment for AtlasGym.

Generates math and logic problems programmatically with known correct
answers for objective scoring.

Difficulty levels:
    L1: Basic arithmetic (addition, subtraction, multiplication, division).
    L2: Word problems requiring equation setup.
    L3: Multi-step algebra and logic puzzles.
    L4: Novel inference and proof-style reasoning.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

from agi.training.gym_env import AtlasGym, AtlasGymConfig, Scenario  # noqa: E402

# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


def _gen_arithmetic() -> Tuple[str, float]:
    """Generate a basic arithmetic problem."""
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    op_sym, op_fn = random.choice(ops)
    a = random.randint(2, 999)
    b = random.randint(2, 999)
    answer = op_fn(a, b)
    return f"What is {a} {op_sym} {b}?", float(answer)


def _gen_division() -> Tuple[str, float]:
    """Generate a clean division problem (no remainder)."""
    b = random.randint(2, 50)
    answer = random.randint(2, 100)
    a = b * answer
    return f"What is {a} / {b}?", float(answer)


def _gen_percentage() -> Tuple[str, float]:
    """Generate a percentage calculation problem."""
    base = random.randint(50, 1000)
    pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
    answer = base * pct / 100.0
    return f"What is {pct}% of {base}?", answer


def _gen_word_problem() -> Tuple[str, float]:
    """Generate an arithmetic word problem."""
    templates = [
        (
            "A store sells {item_a} for ${price_a} each and {item_b} for "
            "${price_b} each. If you buy {qty_a} {item_a} and {qty_b} "
            "{item_b}, what is the total cost in dollars?",
            lambda v: v["price_a"] * v["qty_a"] + v["price_b"] * v["qty_b"],
        ),
        (
            "A train travels at {speed} km/h for {time} hours. "
            "How many kilometers does it cover?",
            lambda v: v["speed"] * v["time"],
        ),
        (
            "A rectangle has length {length} cm and width {width} cm. "
            "What is its area in square centimeters?",
            lambda v: v["length"] * v["width"],
        ),
        (
            "If {people} people share a bill of ${total} equally, "
            "how many dollars does each person pay?",
            lambda v: v["total"] / v["people"],
        ),
        (
            "A factory produces {rate} units per hour. How many units "
            "does it produce in {hours} hours?",
            lambda v: v["rate"] * v["hours"],
        ),
    ]

    items = [
        ("apples", "oranges"),
        ("shirts", "pants"),
        ("books", "notebooks"),
        ("pens", "pencils"),
    ]

    template, fn = random.choice(templates)
    item_pair = random.choice(items)

    values = {
        "item_a": item_pair[0],
        "item_b": item_pair[1],
        "price_a": random.randint(1, 20),
        "price_b": random.randint(1, 20),
        "qty_a": random.randint(1, 10),
        "qty_b": random.randint(1, 10),
        "speed": random.choice([40, 50, 60, 80, 100, 120]),
        "time": random.choice([1, 2, 3, 4, 5, 6]),
        "length": random.randint(3, 30),
        "width": random.randint(3, 30),
        "people": random.randint(2, 10),
        "total": random.choice([50, 60, 80, 100, 120, 150, 200]),
        "rate": random.randint(10, 100),
        "hours": random.randint(2, 12),
    }

    text = template.format(**values)
    answer = fn(values)
    return text, float(answer)


def _gen_algebra() -> Tuple[str, float]:
    """Generate an algebra problem (solve for x)."""
    # ax + b = c  =>  x = (c - b) / a
    a = random.randint(2, 12)
    x_answer = random.randint(-20, 20)
    b = random.randint(-50, 50)
    c = a * x_answer + b

    sign_b = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    return f"Solve for x: {a}x {sign_b} = {c}", float(x_answer)


def _gen_quadratic() -> Tuple[str, str]:
    """Generate a quadratic with integer roots: (x - r1)(x - r2) = 0."""
    r1 = random.randint(-10, 10)
    r2 = random.randint(-10, 10)
    # x^2 - (r1+r2)x + r1*r2 = 0
    b = -(r1 + r2)
    c = r1 * r2

    sign_b = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    sign_c = f"+ {c}" if c >= 0 else f"- {abs(c)}"

    roots = sorted({r1, r2})
    expected = ", ".join(str(r) for r in roots)

    return (
        f"Find all real roots of: x^2 {sign_b}x {sign_c} = 0",
        expected,
    )


def _gen_sequence() -> Tuple[str, float]:
    """Generate a number sequence problem."""
    seq_type = random.choice(["arithmetic", "geometric", "fibonacci_like"])

    if seq_type == "arithmetic":
        start = random.randint(1, 20)
        diff = random.randint(2, 10)
        seq = [start + i * diff for i in range(5)]
        answer = float(seq[-1] + diff)
        seq_str = ", ".join(str(s) for s in seq)
        return (
            f"What is the next number in the sequence: {seq_str}, ?",
            answer,
        )
    elif seq_type == "geometric":
        start = random.randint(1, 5)
        ratio = random.choice([2, 3])
        seq = [start * (ratio**i) for i in range(5)]
        answer = float(seq[-1] * ratio)
        seq_str = ", ".join(str(s) for s in seq)
        return (
            f"What is the next number in the sequence: {seq_str}, ?",
            answer,
        )
    else:
        a, b = random.randint(1, 5), random.randint(1, 5)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        answer = float(seq[-1] + seq[-2])
        seq_str = ", ".join(str(s) for s in seq)
        return (
            f"What is the next number in the sequence: {seq_str}, ?",
            answer,
        )


def _gen_logic_puzzle() -> Tuple[str, str]:
    """Generate a logic puzzle with a determinate answer."""
    puzzles = [
        (
            "There are three boxes: one contains only apples, one contains "
            "only oranges, and one contains both apples and oranges. The "
            "boxes are labelled 'Apples', 'Oranges', and 'Both', but every "
            "label is wrong. You pick one fruit from the box labelled 'Both' "
            "and it is an apple. What does each box actually contain?",
            "The box labelled 'Both' contains only apples. "
            "The box labelled 'Oranges' contains both. "
            "The box labelled 'Apples' contains only oranges.",
        ),
        (
            "If all Bloops are Razzles, and all Razzles are Lazzles, "
            "is it true that all Bloops are Lazzles?",
            "yes",
        ),
        (
            "A is taller than B. C is shorter than B. D is taller than A. "
            "Who is the shortest?",
            "C",
        ),
        (
            "In a room of 3 people, everyone shakes hands with everyone "
            "else exactly once. How many handshakes occur?",
            "3",
        ),
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, how many "
            "minutes would it take 100 machines to make 100 widgets?",
            "5",
        ),
        (
            "A farmer has 15 sheep. All but 8 run away. How many sheep "
            "does the farmer have left?",
            "8",
        ),
    ]
    return random.choice(puzzles)


def _gen_proof_problem() -> Tuple[str, str]:
    """Generate a proof-style reasoning problem."""
    problems = [
        (
            "Prove that the sum of two even numbers is always even. "
            "State your reasoning step by step.",
            "even",  # Key concept that must appear
        ),
        (
            "Prove that if n is odd, then n^2 is odd. Show your reasoning.",
            "odd",
        ),
        (
            "Explain why the product of any integer and zero is zero. "
            "Reference the relevant mathematical axiom.",
            "zero",
        ),
        (
            "Prove by contradiction that the square root of 2 is "
            "irrational. Outline the key steps.",
            "irrational",
        ),
        (
            "Prove that for any integer n, n(n+1) is always even. "
            "Explain your reasoning.",
            "even",
        ),
    ]
    return random.choice(problems)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _extract_numeric_answer(response: str) -> Optional[float]:
    """Extract the final numeric answer from a response.

    Looks for patterns like 'answer is X', 'X', '= X', etc.
    """
    # Try "answer is X" pattern first
    patterns = [
        r"(?:answer|result|total|equals?|=)\s*(?:is\s*)?(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)\s*$",  # Last number in response
    ]
    for pat in patterns:
        matches = re.findall(pat, response.strip(), re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    # Fallback: find all numbers and take the last one
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def _score_numeric(response: str, expected: float) -> Tuple[float, Dict[str, Any]]:
    """Score a numeric response."""
    extracted = _extract_numeric_answer(response)
    if extracted is None:
        return 0.0, {"error": "no_numeric_answer", "expected": expected}

    if abs(extracted - expected) < 1e-6:
        return 1.0, {"extracted": extracted, "expected": expected, "exact": True}
    if abs(extracted - expected) < 0.01 * abs(expected or 1):
        return 0.8, {"extracted": extracted, "expected": expected, "close": True}
    return 0.0, {"extracted": extracted, "expected": expected, "wrong": True}


def _score_text_answer(
    response: str, expected: str, level: int
) -> Tuple[float, Dict[str, Any]]:
    """Score a text-based answer (logic puzzles, proofs)."""
    lower = response.lower().strip()
    exp_lower = expected.lower().strip()

    # Check for exact containment
    if exp_lower in lower:
        return 1.0, {"match": "exact", "expected": expected}

    # For short expected answers, check for the key word
    exp_words = exp_lower.split()
    if len(exp_words) <= 3:
        found = sum(1 for w in exp_words if w in lower)
        if found == len(exp_words):
            return 1.0, {"match": "keyword_all"}
        if found > 0:
            return 0.5, {"match": "keyword_partial", "found": found}
        return 0.0, {"match": "none", "expected": expected}

    # For longer expected answers, check word overlap
    exp_set = set(exp_words)
    resp_words = set(lower.split())
    overlap = len(exp_set & resp_words)
    if overlap >= len(exp_set) * 0.7:
        return 0.9, {"match": "overlap_high", "overlap": overlap}
    if overlap >= len(exp_set) * 0.4:
        return 0.5, {"match": "overlap_medium", "overlap": overlap}
    return 0.0, {"match": "overlap_low", "overlap": overlap}


# ---------------------------------------------------------------------------
# ReasoningEnv
# ---------------------------------------------------------------------------


class ReasoningEnv(AtlasGym):
    """Logical Reasoning Gymnasium environment.

    Generates math problems (arithmetic, algebra, logic puzzles)
    programmatically with known correct answers for objective scoring.

    Usage::

        env = ReasoningEnv()
        obs, info = env.reset(options={"level": 1})
        obs, reward, done, truncated, info = env.step("42")
    """

    def __init__(self, config: Optional[AtlasGymConfig] = None) -> None:
        cfg = config or AtlasGymConfig(env_name="reasoning")
        if cfg.env_name == "base":
            cfg.env_name = "reasoning"
        super().__init__(config=cfg)

    def _generate_scenario(self, level: int) -> Scenario:
        """Generate a reasoning problem for the given level."""
        if level == 1:
            return self._generate_l1()
        elif level == 2:
            return self._generate_l2()
        elif level == 3:
            return self._generate_l3()
        else:
            return self._generate_l4()

    def _generate_l1(self) -> Scenario:
        """L1: Basic arithmetic."""
        gen = random.choice([_gen_arithmetic, _gen_division, _gen_percentage])
        text, answer = gen()
        return Scenario(
            text=f"{text}\n\nProvide the numeric answer.",
            level=1,
            expected=str(answer),
            metadata={"answer_type": "numeric", "expected": answer},
        )

    def _generate_l2(self) -> Scenario:
        """L2: Word problems."""
        gen = random.choice([_gen_word_problem, _gen_algebra])
        text, answer = gen()
        return Scenario(
            text=(f"{text}\n\nShow your work and provide the final answer."),
            level=2,
            expected=str(answer),
            metadata={"answer_type": "numeric", "expected": answer},
        )

    def _generate_l3(self) -> Scenario:
        """L3: Multi-step algebra and logic puzzles."""
        gen_type = random.choice(["quadratic", "sequence", "logic"])

        if gen_type == "quadratic":
            text, expected = _gen_quadratic()
            return Scenario(
                text=f"{text}\n\nShow your work.",
                level=3,
                expected=expected,
                metadata={"answer_type": "text", "subtype": "quadratic"},
            )
        elif gen_type == "sequence":
            text, answer = _gen_sequence()
            return Scenario(
                text=f"{text}\n\nExplain the pattern and give the answer.",
                level=3,
                expected=str(answer),
                metadata={"answer_type": "numeric", "expected": answer},
            )
        else:
            text, expected = _gen_logic_puzzle()
            return Scenario(
                text=f"{text}\n\nExplain your reasoning step by step.",
                level=3,
                expected=expected,
                metadata={"answer_type": "text", "subtype": "logic"},
            )

    def _generate_l4(self) -> Scenario:
        """L4: Novel inference and proof-style reasoning."""
        text, expected = _gen_proof_problem()
        return Scenario(
            text=text,
            level=4,
            expected=expected,
            metadata={"answer_type": "text", "subtype": "proof"},
        )

    def _score_response(
        self, scenario: Scenario, response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score based on answer type."""
        answer_type = scenario.metadata.get("answer_type", "numeric")

        if answer_type == "numeric":
            expected = scenario.metadata.get("expected", 0.0)
            return _score_numeric(response, float(expected))
        else:
            expected = scenario.expected or ""
            return _score_text_answer(response, expected, scenario.level)
