#!/usr/bin/env python3
"""Smoke test for the AtlasGym training framework."""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/claude/agi-hpc/src")



def main() -> None:
    print("=" * 60)
    print("SMOKE TEST: AtlasGym Environments")
    print("=" * 60)

    # 1. Reasoning Environment (all levels)
    print("\n--- Reasoning Environment ---")
    from agi.training.envs.reasoning_env import ReasoningEnv
    from agi.training.gym_env import AtlasGymConfig

    env = ReasoningEnv(AtlasGymConfig(env_name="reasoning", level=1))

    for level in range(1, 5):
        obs, info = env.reset(options={"level": level})
        obs_preview = obs[:80].replace("\n", " ")
        print(f"  L{level}: {obs_preview}...")

        md = info.get("metadata", {})
        if md.get("answer_type") == "numeric":
            expected = md.get("expected", 42)
            obs, reward, done, trunc, step_info = env.step(f"The answer is {expected}")
        else:
            obs, reward, done, trunc, step_info = env.step("yes")
        print(f"       score={reward:.2f} done={done}")

    # 2. Coding Environment
    print("\n--- Coding Environment ---")
    from agi.training.envs.coding_env import CodingEnv

    env = CodingEnv(AtlasGymConfig(env_name="coding", level=1))
    obs, info = env.reset(options={"level": 1})
    problem_name = info.get("metadata", {}).get("problem_name", "?")
    print(f"  Problem: {problem_name}")

    # Submit correct code for the problem
    if problem_name == "reverse_string":
        code = "def reverse_string(s):\n    return s[::-1]"
    elif problem_name == "fibonacci":
        code = "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n - 1): a, b = b, a + b\n    return b"
    elif problem_name == "is_palindrome":
        code = "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]"
    elif problem_name == "count_vowels":
        code = (
            "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"
        )
    elif problem_name == "factorial":
        code = "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)"
    elif problem_name == "sum_list":
        code = "def sum_list(nums):\n    return sum(nums)"
    elif problem_name == "max_element":
        code = "def max_element(nums):\n    return max(nums)"
    else:
        code = "# unknown problem"

    obs, reward, done, trunc, step_info = env.step(code)
    print(f"  Score: {reward:.2f}")
    details = step_info.get("score_breakdown", {})
    print(f"  Passed: {details.get('passed', '?')}/{details.get('total', '?')}")

    # 3. Debate Environment
    print("\n--- Debate Environment ---")
    from agi.training.envs.debate_env import DebateEnv

    env = DebateEnv(AtlasGymConfig(env_name="debate", level=2))
    obs, info = env.reset()
    topic = info.get("metadata", {}).get("topic", "?")
    print(f"  Topic: {topic[:70]}...")

    response = (
        "Both the analytical and creative perspectives have merit. "
        "The logical approach emphasizes systematic analysis and evidence-based "
        "reasoning. However, the creative perspective offers innovative possibilities "
        "and divergent thinking. The key resolution is to balance both: use rigorous "
        "analytical frameworks while remaining open to creative insights. "
        "Ultimately, the synthesis of these complementary viewpoints yields "
        "the strongest approach."
    )
    obs, reward, done, trunc, step_info = env.step(response)
    print(f"  Score: {reward:.2f}")
    print(f"  Breakdown: {step_info.get('score_breakdown', {})}")

    # 4. Memory Environment
    print("\n--- Memory Environment ---")
    from agi.training.envs.memory_env import MemoryEnv

    env = MemoryEnv(AtlasGymConfig(env_name="memory", level=1))
    obs, info = env.reset()
    print(f"  Obs: {obs[:80].replace(chr(10), ' ')}...")

    # Acknowledge
    obs, reward, done, trunc, step_info = env.step("I understand.")
    is_quiz = step_info.get("is_quiz")
    print(f"  After ack: is_quiz={is_quiz}, done={done}")

    if is_quiz:
        print(f"  Quiz: {obs[:80].replace(chr(10), ' ')}...")
        obs, reward, done, trunc, step_info = env.step(env._quiz_answer)
        print(f"  Score: {reward:.2f} (answering correctly)")

    # 5. Ethics Environment (with live DB)
    print("\n--- Ethics Environment (live DB) ---")
    from agi.training.envs.ethics_env import EthicsEnv, EthicsEnvConfig

    env = EthicsEnv(EthicsEnvConfig(level=1, db_dsn="dbname=atlas user=claude"))
    obs, info = env.reset()
    traditions = info.get("metadata", {}).get("traditions", [])
    is_fallback = info.get("metadata", {}).get("fallback", False)
    print(f"  Tradition: {traditions}")
    print(f"  Fallback: {is_fallback}")
    print(f"  Obs (first 120): {obs[:120].replace(chr(10), ' ')}...")

    response = (
        "The key ethical principle in this passage relates to virtue and moral duty. "
        "The tradition emphasizes justice and compassion as fundamental moral values. "
        "This principle guides decision-making by requiring consideration of the "
        "consequences of actions and the character of the moral agent. "
        "It fits within the broader ethical framework by connecting individual "
        "moral choices to community well-being and divine commandment."
    )
    obs, reward, done, trunc, step_info = env.step(response)
    print(f"  Score: {reward:.2f}")
    print(f"  Breakdown: {step_info.get('score_breakdown', {})}")

    # Also test L2
    obs, info = env.reset(options={"level": 2})
    traditions = info.get("metadata", {}).get("traditions", [])
    print(f"  L2 Traditions: {traditions}")
    obs, reward, done, trunc, step_info = env.step(
        "Comparing these two traditions reveals both similarities and differences. "
        "The first tradition emphasizes virtue and moral duty through a framework "
        "of justice, while the second tradition takes a different perspective "
        "on ethical obligation. In contrast to the first, the second tradition "
        "values compassion and harmony. Both however converge on the importance "
        "of moral reasoning and community welfare."
    )
    print(f"  L2 Score: {reward:.2f}")

    # 6. Curriculum Manager
    print("\n--- Curriculum Manager ---")
    from agi.training.curriculum import CurriculumManager, CurriculumConfig

    cm = CurriculumManager(
        CurriculumConfig(
            db_dsn="dbname=atlas user=claude",
            window_size=5,
            promote_threshold=0.80,
        )
    )

    print(f"  Initial level: {cm.get_level('test_env')}")
    for score in [0.9, 0.8, 0.9, 0.85, 0.9]:
        action = cm.record_score("test_env", score)
        if action:
            print(f"  -> {action}!")

    print(f"  After 5 high scores: level={cm.get_level('test_env')}")

    # 7. Scorer persistence
    print("\n--- Scorer Persistence ---")
    from agi.training.scorer import ResponseScorer, ScorerConfig

    scorer = ResponseScorer(ScorerConfig(db_dsn="dbname=atlas user=claude"))
    row_id = scorer.store_result(
        env_name="smoke_test",
        level=1,
        scenario="Test scenario",
        response="Test response",
        score=0.85,
        metadata={"test": True},
    )
    print(f"  Stored result id={row_id}")

    recent = scorer.get_recent_scores("smoke_test", n=5)
    print(f"  Recent scores: {len(recent)} results")

    total = scorer.get_total_episodes()
    print(f"  Total episodes in DB: {total}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
