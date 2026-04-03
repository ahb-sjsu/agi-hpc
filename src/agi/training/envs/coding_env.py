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
Code Generation Environment for AtlasGym.

Presents a function specification (name, inputs, expected outputs,
test cases) and evaluates Atlas's code in a sandboxed subprocess.

Scoring:
    0.0: Code doesn't compile/run.
    0.5: Runs but fails some test cases.
    1.0: All test cases pass.

Sandbox restrictions:
    - subprocess with timeout (5 seconds default)
    - restricted PATH (no network tools)
    - no file system access beyond temp
    - PYTHONDONTWRITEBYTECODE set

Difficulty levels:
    L1: Simple functions (reverse string, fibonacci, etc.).
    L2: Data structures (linked list, stack, queue operations).
    L3: Algorithms (sorting, searching, graph traversal).
    L4: System design (class hierarchies, design patterns).
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from agi.training.gym_env import AtlasGym, AtlasGymConfig, Scenario
from agi.training.scorer import score_code_execution

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

# Each problem is: (name, description, test_cases)
# test_cases: [{"call": "func_name(args)", "expected": value}, ...]

L1_PROBLEMS = [
    {
        "name": "reverse_string",
        "description": (
            "Write a function `reverse_string(s: str) -> str` that returns "
            "the input string reversed."
        ),
        "test_cases": [
            {"call": "reverse_string('hello')", "expected": "olleh"},
            {"call": "reverse_string('')", "expected": ""},
            {"call": "reverse_string('a')", "expected": "a"},
            {"call": "reverse_string('racecar')", "expected": "racecar"},
            {"call": "reverse_string('12345')", "expected": "54321"},
        ],
    },
    {
        "name": "fibonacci",
        "description": (
            "Write a function `fibonacci(n: int) -> int` that returns the "
            "nth Fibonacci number (0-indexed). F(0)=0, F(1)=1, F(2)=1, etc."
        ),
        "test_cases": [
            {"call": "fibonacci(0)", "expected": 0},
            {"call": "fibonacci(1)", "expected": 1},
            {"call": "fibonacci(2)", "expected": 1},
            {"call": "fibonacci(5)", "expected": 5},
            {"call": "fibonacci(10)", "expected": 55},
        ],
    },
    {
        "name": "is_palindrome",
        "description": (
            "Write a function `is_palindrome(s: str) -> bool` that returns "
            "True if the string is a palindrome (case-insensitive, ignoring "
            "spaces and punctuation), False otherwise."
        ),
        "test_cases": [
            {"call": "is_palindrome('racecar')", "expected": True},
            {"call": "is_palindrome('hello')", "expected": False},
            {"call": "is_palindrome('A man a plan a canal Panama')", "expected": True},
            {"call": "is_palindrome('')", "expected": True},
            {"call": "is_palindrome('Was it a car or a cat I saw')", "expected": True},
        ],
    },
    {
        "name": "count_vowels",
        "description": (
            "Write a function `count_vowels(s: str) -> int` that returns "
            "the number of vowels (a, e, i, o, u) in the string "
            "(case-insensitive)."
        ),
        "test_cases": [
            {"call": "count_vowels('hello')", "expected": 2},
            {"call": "count_vowels('AEIOU')", "expected": 5},
            {"call": "count_vowels('bcdfg')", "expected": 0},
            {"call": "count_vowels('')", "expected": 0},
            {"call": "count_vowels('Python Programming')", "expected": 4},
        ],
    },
    {
        "name": "factorial",
        "description": (
            "Write a function `factorial(n: int) -> int` that returns n! "
            "(n factorial). Assume n >= 0."
        ),
        "test_cases": [
            {"call": "factorial(0)", "expected": 1},
            {"call": "factorial(1)", "expected": 1},
            {"call": "factorial(5)", "expected": 120},
            {"call": "factorial(10)", "expected": 3628800},
        ],
    },
    {
        "name": "sum_list",
        "description": (
            "Write a function `sum_list(nums: list) -> int` that returns "
            "the sum of all numbers in the list."
        ),
        "test_cases": [
            {"call": "sum_list([1, 2, 3])", "expected": 6},
            {"call": "sum_list([])", "expected": 0},
            {"call": "sum_list([10])", "expected": 10},
            {"call": "sum_list([-1, 1])", "expected": 0},
            {"call": "sum_list([100, 200, 300])", "expected": 600},
        ],
    },
    {
        "name": "max_element",
        "description": (
            "Write a function `max_element(nums: list) -> int` that returns "
            "the largest element in a non-empty list of integers."
        ),
        "test_cases": [
            {"call": "max_element([1, 2, 3])", "expected": 3},
            {"call": "max_element([5])", "expected": 5},
            {"call": "max_element([-1, -5, -2])", "expected": -1},
            {"call": "max_element([10, 3, 7, 1, 9])", "expected": 10},
        ],
    },
]

L2_PROBLEMS = [
    {
        "name": "flatten_list",
        "description": (
            "Write a function `flatten_list(nested: list) -> list` that "
            "flattens a nested list of integers to a single flat list. "
            "For example, [[1,2],[3,[4,5]]] -> [1,2,3,4,5]."
        ),
        "test_cases": [
            {"call": "flatten_list([[1, 2], [3, 4]])", "expected": [1, 2, 3, 4]},
            {"call": "flatten_list([1, [2, [3, [4]]]])", "expected": [1, 2, 3, 4]},
            {"call": "flatten_list([])", "expected": []},
            {"call": "flatten_list([[1], [], [2, 3]])", "expected": [1, 2, 3]},
        ],
    },
    {
        "name": "two_sum",
        "description": (
            "Write a function `two_sum(nums: list, target: int) -> list` "
            "that returns the indices of two numbers that add up to target. "
            "Assume exactly one solution exists. Return as [i, j] with i < j."
        ),
        "test_cases": [
            {"call": "two_sum([2, 7, 11, 15], 9)", "expected": [0, 1]},
            {"call": "two_sum([3, 2, 4], 6)", "expected": [1, 2]},
            {"call": "two_sum([1, 5, 3, 7], 8)", "expected": [1, 2]},
        ],
    },
    {
        "name": "group_anagrams",
        "description": (
            "Write a function `group_anagrams(words: list) -> list` that "
            "groups anagrams together. Return a sorted list of sorted groups. "
            "For example, ['eat','tea','tan','ate','nat','bat'] -> "
            "[['ate','eat','tea'],['bat'],['nat','tan']]."
        ),
        "test_cases": [
            {
                "call": "group_anagrams(['eat','tea','tan','ate','nat','bat'])",
                "expected": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]],
            },
            {"call": "group_anagrams([''])", "expected": [[""]]},
            {"call": "group_anagrams(['a'])", "expected": [["a"]]},
        ],
    },
    {
        "name": "stack_min",
        "description": (
            "Implement a class `MinStack` with methods:\n"
            "- `push(val)`: Push val onto stack.\n"
            "- `pop()`: Remove top element.\n"
            "- `top()`: Return top element.\n"
            "- `get_min()`: Return minimum element.\n\n"
            "All operations must be O(1).\n"
            "Write a function `test_min_stack()` that creates a MinStack, "
            "pushes [-2, 0, -3], calls get_min() (should return -3), "
            "pops, then calls top() (should return 0) and get_min() "
            "(should return -2), and returns [min1, top, min2]."
        ),
        "test_cases": [
            {"call": "test_min_stack()", "expected": [-3, 0, -2]},
        ],
    },
    {
        "name": "matrix_transpose",
        "description": (
            "Write a function `transpose(matrix: list) -> list` that "
            "returns the transpose of a 2D matrix (list of lists)."
        ),
        "test_cases": [
            {
                "call": "transpose([[1,2,3],[4,5,6]])",
                "expected": [[1, 4], [2, 5], [3, 6]],
            },
            {
                "call": "transpose([[1]])",
                "expected": [[1]],
            },
            {
                "call": "transpose([[1,2],[3,4],[5,6]])",
                "expected": [[1, 3, 5], [2, 4, 6]],
            },
        ],
    },
]

L3_PROBLEMS = [
    {
        "name": "merge_sort",
        "description": (
            "Write a function `merge_sort(arr: list) -> list` that "
            "implements merge sort and returns a new sorted list."
        ),
        "test_cases": [
            {
                "call": "merge_sort([3, 1, 4, 1, 5, 9, 2, 6])",
                "expected": [1, 1, 2, 3, 4, 5, 6, 9],
            },
            {"call": "merge_sort([])", "expected": []},
            {"call": "merge_sort([1])", "expected": [1]},
            {"call": "merge_sort([5, 4, 3, 2, 1])", "expected": [1, 2, 3, 4, 5]},
        ],
    },
    {
        "name": "binary_search",
        "description": (
            "Write a function `binary_search(arr: list, target: int) -> int` "
            "that returns the index of target in a sorted list, "
            "or -1 if not found."
        ),
        "test_cases": [
            {"call": "binary_search([1, 3, 5, 7, 9], 5)", "expected": 2},
            {"call": "binary_search([1, 3, 5, 7, 9], 4)", "expected": -1},
            {"call": "binary_search([], 1)", "expected": -1},
            {"call": "binary_search([1], 1)", "expected": 0},
            {
                "call": "binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)",
                "expected": 9,
            },
        ],
    },
    {
        "name": "lru_cache",
        "description": (
            "Implement a class `LRUCache` with:\n"
            "- `__init__(capacity)`: Set cache capacity.\n"
            "- `get(key) -> int`: Get value or -1 if not found.\n"
            "- `put(key, value)`: Insert/update. Evict LRU if over capacity.\n\n"
            "Write a function `test_lru()` that creates LRUCache(2), "
            "puts (1,1), (2,2), gets 1 (returns 1), puts (3,3) which "
            "evicts key 2, gets 2 (returns -1), and returns [get1, get2]."
        ),
        "test_cases": [
            {"call": "test_lru()", "expected": [1, -1]},
        ],
    },
    {
        "name": "longest_common_subsequence",
        "description": (
            "Write a function `lcs(s1: str, s2: str) -> int` that returns "
            "the length of the longest common subsequence of two strings."
        ),
        "test_cases": [
            {"call": "lcs('abcde', 'ace')", "expected": 3},
            {"call": "lcs('abc', 'abc')", "expected": 3},
            {"call": "lcs('abc', 'def')", "expected": 0},
            {"call": "lcs('', 'abc')", "expected": 0},
        ],
    },
]

L4_PROBLEMS = [
    {
        "name": "event_emitter",
        "description": (
            "Implement a class `EventEmitter` with:\n"
            "- `on(event_name, callback)`: Register a callback for an event.\n"
            "- `emit(event_name, *args)`: Call all callbacks for the event.\n"
            "- `off(event_name, callback)`: Remove a specific callback.\n"
            "- `once(event_name, callback)`: Register a one-time callback.\n\n"
            "Write a function `test_emitter()` that:\n"
            "1. Creates an EventEmitter.\n"
            "2. Registers a callback that appends values to a list.\n"
            "3. Emits 'data' with value 1, then 2.\n"
            "4. Registers a once-callback appending to the list.\n"
            "5. Emits 'data' with value 3, then 4.\n"
            "6. Returns the collected list."
        ),
        "test_cases": [
            {"call": "test_emitter()", "expected": [1, 2, 3, 3, 4]},
        ],
    },
    {
        "name": "rate_limiter",
        "description": (
            "Implement a class `RateLimiter` with:\n"
            "- `__init__(max_calls, period_seconds)`: Set limits.\n"
            "- `allow(timestamp) -> bool`: Return True if a call at this "
            "timestamp is allowed, False if rate limit exceeded.\n\n"
            "Write a function `test_rate_limiter()` that creates "
            "RateLimiter(3, 10), calls allow() at timestamps "
            "[1, 2, 3, 4, 11, 12] and returns the list of results."
        ),
        "test_cases": [
            {
                "call": "test_rate_limiter()",
                "expected": [True, True, True, False, True, True],
            },
        ],
    },
    {
        "name": "trie",
        "description": (
            "Implement a Trie (prefix tree) class with:\n"
            "- `insert(word)`: Insert a word.\n"
            "- `search(word) -> bool`: Return True if word exists.\n"
            "- `starts_with(prefix) -> bool`: Return True if any word "
            "starts with prefix.\n\n"
            "Write a function `test_trie()` that inserts 'apple' and 'app', "
            "then returns [search('apple'), search('app'), search('ap'), "
            "starts_with('ap'), starts_with('b')]."
        ),
        "test_cases": [
            {
                "call": "test_trie()",
                "expected": [True, True, False, True, False],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def _extract_code(response: str) -> str:
    """Extract Python code from a response that may contain markdown.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Plain code (no markdown)
    """
    # Try to find python code blocks
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, response, re.DOTALL)
        if matches:
            # Concatenate all code blocks
            return "\n\n".join(matches)

    # If no code blocks, return the whole response (might be plain code)
    return response


# ---------------------------------------------------------------------------
# CodingEnv
# ---------------------------------------------------------------------------


class CodingEnv(AtlasGym):
    """Code Generation Gymnasium environment.

    Presents function specifications with test cases and evaluates
    Atlas's code in a sandboxed subprocess.

    Usage::

        env = CodingEnv()
        obs, info = env.reset(options={"level": 1})
        obs, reward, done, truncated, info = env.step("def reverse_string(s): ...")
    """

    def __init__(self, config: Optional[AtlasGymConfig] = None) -> None:
        cfg = config or AtlasGymConfig(env_name="coding")
        if cfg.env_name == "base":
            cfg.env_name = "coding"
        super().__init__(config=cfg)
        self._sandbox_timeout: int = 5

    def _generate_scenario(self, level: int) -> Scenario:
        """Generate a coding problem for the given level."""
        if level == 1:
            problem = random.choice(L1_PROBLEMS)
        elif level == 2:
            problem = random.choice(L2_PROBLEMS)
        elif level == 3:
            problem = random.choice(L3_PROBLEMS)
        else:
            problem = random.choice(L4_PROBLEMS)

        # Build the prompt
        test_desc = "\n".join(
            f"  - {tc['call']} should return {tc['expected']!r}"
            for tc in problem["test_cases"]
        )
        text = (
            f"## {problem['name']}\n\n"
            f"{problem['description']}\n\n"
            f"### Test cases:\n{test_desc}\n\n"
            f"Write the Python code. Only include the function/class "
            f"definitions. Do not include test code or print statements."
        )

        return Scenario(
            text=text,
            level=level,
            metadata={
                "problem_name": problem["name"],
                "test_cases": problem["test_cases"],
            },
        )

    def _score_response(
        self, scenario: Scenario, response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Extract code and run it against test cases in a sandbox."""
        code = _extract_code(response)
        test_cases = scenario.metadata.get("test_cases", [])

        if not code.strip():
            return 0.0, {"error": "empty_code"}

        score, details = score_code_execution(
            code=code,
            test_cases=test_cases,
            timeout=self._sandbox_timeout,
        )

        details["problem_name"] = scenario.metadata.get("problem_name", "")
        return score, details
