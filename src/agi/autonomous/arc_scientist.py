"""Autonomous ARC Scientist — self-improving solver for NeuroGolf tasks.

Implements a closed-loop scientific reasoning cycle:
  1. OBSERVE: Pick an unsolved task, study its patterns
  2. HYPOTHESIZE: Form a theory about the transformation (via LLM)
  3. EXPERIMENT: Generate candidate Python transforms
  4. EVALUATE: Security Radar — verify on ALL examples
  5. LEARN: Store what worked and what failed in episodic memory
  6. ADAPT: Update strategy weights based on accumulated experience
  7. REPEAT

The system genuinely learns: it tracks which prompt framings, task
patterns, and reasoning strategies succeed or fail, and shifts its
approach accordingly. It doesn't just retry — it reasons about WHY
something failed and tries a qualitatively different approach.

NRP guidelines respected:
  - GPU utilization >40% when using GPUs
  - No sleep infinity — jobs do real work then exit
  - Fair access via polite-submit backoff
  - Scale limited to 5 concurrent GPU pods, 10 CPU pods
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Episodic memory — the system remembers and learns
# ═══════════════════════════════════════════════════════════════

@dataclass
class Attempt:
    """Record of one attempt to solve a task."""
    task_num: int
    timestamp: str
    strategy: str          # which prompt framing / approach
    model: str             # which LLM
    verified: bool         # passed Security Radar?
    correct: int           # examples correct
    total: int             # examples total
    error: str = ""        # failure reason if any
    code: str = ""         # the Python transform
    insight: str = ""      # what we learned from this attempt


@dataclass
class TaskKnowledge:
    """Accumulated knowledge about a task."""
    task_num: int
    attempts: list = field(default_factory=list)
    solved: bool = False
    best_correct: int = 0
    best_total: int = 0
    strategies_tried: list = field(default_factory=list)
    failure_patterns: list = field(default_factory=list)  # what went wrong
    hypotheses: list = field(default_factory=list)        # theories about the task


@dataclass
class StrategyStats:
    """Performance statistics for a prompt strategy."""
    name: str
    attempts: int = 0
    successes: int = 0
    partial_successes: int = 0
    failures: int = 0
    avg_correct_ratio: float = 0.0

    @property
    def success_rate(self):
        return self.successes / max(self.attempts, 1)

    @property
    def weight(self):
        """Bayesian-inspired weight: prior + evidence."""
        # Start with uniform prior, update with evidence
        # Thompson sampling: weight ~ Beta(successes + 1, failures + 1)
        alpha = self.successes + self.partial_successes * 0.3 + 1
        beta = self.failures + 1
        return alpha / (alpha + beta)


class EpisodicMemory:
    """Persistent memory of all attempts and learned strategies."""

    def __init__(self, path: str = "arc_scientist_memory.json"):
        self.path = Path(path)
        self.tasks: dict[int, TaskKnowledge] = {}
        self.strategies: dict[str, StrategyStats] = {}
        self.global_insights: list[str] = []
        self.total_attempts: int = 0
        self.total_solves: int = 0
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for tn, tk in data.get("tasks", {}).items():
                    self.tasks[int(tn)] = TaskKnowledge(**tk)
                for name, ss in data.get("strategies", {}).items():
                    self.strategies[name] = StrategyStats(**ss)
                self.global_insights = data.get("global_insights", [])
                self.total_attempts = data.get("total_attempts", 0)
                self.total_solves = data.get("total_solves", 0)
            except Exception:
                pass

    def save(self):
        data = {
            "tasks": {str(k): asdict(v) for k, v in self.tasks.items()},
            "strategies": {k: asdict(v) for k, v in self.strategies.items()},
            "global_insights": self.global_insights,
            "total_attempts": self.total_attempts,
            "total_solves": self.total_solves,
            "last_updated": datetime.now().isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def record_attempt(self, attempt: Attempt):
        self.total_attempts += 1
        tn = attempt.task_num

        # Update task knowledge
        if tn not in self.tasks:
            self.tasks[tn] = TaskKnowledge(task_num=tn)
        tk = self.tasks[tn]
        tk.attempts.append(asdict(attempt))
        if attempt.strategy not in tk.strategies_tried:
            tk.strategies_tried.append(attempt.strategy)
        if attempt.verified:
            tk.solved = True
            self.total_solves += 1
        if attempt.correct > tk.best_correct:
            tk.best_correct = attempt.correct
            tk.best_total = attempt.total
        if attempt.error and attempt.error not in tk.failure_patterns:
            tk.failure_patterns.append(attempt.error)

        # Update strategy stats
        if attempt.strategy not in self.strategies:
            self.strategies[attempt.strategy] = StrategyStats(name=attempt.strategy)
        ss = self.strategies[attempt.strategy]
        ss.attempts += 1
        if attempt.verified:
            ss.successes += 1
        elif attempt.correct > 0:
            ss.partial_successes += 1
        else:
            ss.failures += 1
        # Running average of correct ratio
        ratio = attempt.correct / max(attempt.total, 1)
        ss.avg_correct_ratio = (
            ss.avg_correct_ratio * (ss.attempts - 1) + ratio
        ) / ss.attempts

        self.save()

    def get_unsolved_tasks(self, all_tasks: list[int]) -> list[int]:
        """Return unsolved tasks, prioritized by proximity to solution."""
        unsolved = [t for t in all_tasks if t not in self.tasks or not self.tasks[t].solved]
        # Sort by best_correct descending (closest to solving first)
        unsolved.sort(
            key=lambda t: self.tasks[t].best_correct if t in self.tasks else 0,
            reverse=True
        )
        return unsolved

    def pick_strategy(self) -> str:
        """Thompson sampling: pick strategy proportional to learned weights."""
        if not self.strategies:
            return "direct"
        # Weighted random selection
        names = list(self.strategies.keys())
        weights = [self.strategies[n].weight for n in names]
        total = sum(weights)
        if total == 0:
            return random.choice(names)
        probs = [w / total for w in weights]
        return np.random.choice(names, p=probs)

    def get_strategy_report(self) -> str:
        """Human-readable strategy performance report."""
        lines = ["Strategy Performance:"]
        ranked = sorted(self.strategies.values(),
                       key=lambda s: s.weight, reverse=True)
        for s in ranked:
            lines.append(
                f"  {s.name:20s} w={s.weight:.2f} "
                f"({s.successes}/{s.attempts} solved, "
                f"avg_ratio={s.avg_correct_ratio:.2f})"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Prompt strategies — diverse ways to ask the LLM
# ═══════════════════════════════════════════════════════════════

STRATEGIES = {
    "direct": (
        "Here are input/output grid pairs:\n\n{examples}\n"
        "Write: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Use numpy. ```python ... ```"
    ),
    "step_by_step": (
        "Analyze these grid transformations:\n\n{examples}\n"
        "Step 1: What patterns exist?\nStep 2: What changes?\n"
        "Step 3: Write the general rule as:\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Use numpy. ```python ... ```"
    ),
    "expert": (
        "You are an ARC-AGI expert. Core priors: objectness, geometry, physics.\n\n"
        "{examples}\nIdentify the prior and write:\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]```python```"
    ),
    "reverse_engineer": (
        "I lost the source code. Here are test cases:\n\n{examples}\n"
        "Reverse-engineer: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Use numpy. ```python```"
    ),
    "concise": (
        "{examples}\ndef transform(grid):\n    import numpy as np\n"
        "    # implement pattern\nComplete it. ```python```"
    ),
    "visual": (
        "Visualize colored grids. What moved? Flipped? Grew? Shrank?\n\n{examples}\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]```python```"
    ),
    "analogies": (
        "ARC-AGI grid puzzles.\n\n{examples}\n"
        "Operations: crop, tile, flip, rotate, shift, recolor, fill, mask, "
        "scale, reflect, overlay, extract, sort, gravity, flood fill.\n"
        "Write: def transform(grid: list[list[int]]) -> list[list[int]]```python```"
    ),
    "failure_aware": (
        "Previous attempts failed because: {failure_context}\n\n"
        "Try a DIFFERENT approach for these grids:\n\n{examples}\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]```python```"
    ),
}


# ═══════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════

def format_examples(task: dict, max_examples: int = 4) -> str:
    parts = []
    for i, ex in enumerate(task.get("train", [])[:max_examples]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        parts.append(f"Example {i+1}:")
        parts.append(f"  input ({inp.shape[0]}x{inp.shape[1]}): {ex['input']}")
        parts.append(f"  output ({out.shape[0]}x{out.shape[1]}): {ex['output']}")
        parts.append("")
    return "\n".join(parts)


def extract_code(response: str) -> str | None:
    if not response:
        return None
    if "```" in response:
        parts = response.split("```")
        for p in parts[1:]:
            if p.startswith("python"):
                p = p[6:]
            if "def transform" in p:
                return p.strip()
    if "def transform" in response:
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if "def transform" in line:
                return "\n".join(lines[i:])
    return None


def verify_transform(transform_fn, task: dict) -> tuple[int, int]:
    correct = total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            total += 1
            try:
                result = transform_fn(ex["input"])
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                if result == ex["output"]:
                    correct += 1
                else:
                    return correct, total
            except Exception:
                return correct, total
    return correct, total


# ═══════════════════════════════════════════════════════════════
# The Scientist — autonomous reasoning loop
# ═══════════════════════════════════════════════════════════════

class ARCScientist:
    """Autonomous agent that learns to solve ARC tasks."""

    def __init__(self, task_dir: str, memory_path: str = "arc_scientist_memory.json",
                 llm_token: str = "", llm_base_url: str = "https://ellm.nrp-nautilus.io/v1"):
        self.task_dir = Path(task_dir)
        self.memory = EpisodicMemory(memory_path)
        self.llm_token = llm_token or os.environ.get("NRP_LLM_TOKEN", "")
        self.llm_base_url = llm_base_url
        self.client = None
        self._init_client()

        # Discover available tasks
        self.all_tasks = sorted([
            int(f.stem[4:]) for f in self.task_dir.glob("task*.json")
        ])

    def _init_client(self):
        if self.llm_token:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.llm_token, base_url=self.llm_base_url)

    def _load_task(self, tn: int) -> dict:
        with open(self.task_dir / f"task{tn:03d}.json") as f:
            return json.load(f)

    def _call_llm(self, prompt: str, model: str = "kimi") -> str | None:
        if not self.client:
            return None
        extra = {}
        if model == "kimi":
            extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
        elif "qwen" in model:
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        try:
            r = self.client.chat.completions.create(
                model=model, max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            return None

    def _reflect_on_failure(self, task: dict, tn: int, code: str,
                             correct: int, total: int) -> str:
        """Ask the LLM to analyze WHY a transform failed."""
        examples = format_examples(task, max_examples=2)
        prompt = (
            f"This Python transform was supposed to solve this ARC puzzle but "
            f"only got {correct}/{total} examples correct.\n\n"
            f"Task examples:\n{examples}\n"
            f"Failed code:\n```python\n{code}\n```\n\n"
            f"In ONE sentence, what is the code doing wrong? "
            f"What pattern is it missing?"
        )
        response = self._call_llm(prompt, model="kimi")
        if response:
            return response.strip()[:200]
        return ""

    def run_cycle(self, max_attempts: int = 50, models: list[str] = None):
        """Run one full learning cycle.

        Each cycle:
          1. Pick unsolved task (prioritize near-misses)
          2. Pick strategy (Thompson sampling from learned weights)
          3. Generate transform
          4. Verify (Security Radar)
          5. If failed: reflect on why, store insight
          6. If solved: celebrate, store solution
          7. Periodically report strategy performance
        """
        if models is None:
            models = ["kimi", "qwen3"]

        unsolved = self.memory.get_unsolved_tasks(self.all_tasks)
        print(f"ARC Scientist starting cycle")
        print(f"  Tasks: {len(self.all_tasks)} total, {len(unsolved)} unsolved")
        print(f"  Memory: {self.memory.total_attempts} prior attempts, "
              f"{self.memory.total_solves} solves")
        if self.memory.strategies:
            print(self.memory.get_strategy_report())
        print()

        solved_this_cycle = 0
        attempts_this_cycle = 0

        for attempt_num in range(max_attempts):
            if not unsolved:
                print("All tasks solved!")
                break

            # 1. OBSERVE: Pick a task
            # Mix exploration (random unsolved) and exploitation (near-misses)
            if random.random() < 0.3 and any(
                t in self.memory.tasks and self.memory.tasks[t].best_correct > 0
                for t in unsolved
            ):
                # Exploitation: pick a near-miss
                near_misses = [t for t in unsolved
                               if t in self.memory.tasks
                               and self.memory.tasks[t].best_correct > 0]
                tn = random.choice(near_misses[:10])
            else:
                # Exploration: random unsolved
                tn = random.choice(unsolved[:50])

            task = self._load_task(tn)
            examples = format_examples(task)

            # 2. HYPOTHESIZE: Pick strategy
            strategy_name = self.memory.pick_strategy()

            # If we have failure history, sometimes use failure-aware strategy
            tk = self.memory.tasks.get(tn)
            if tk and tk.failure_patterns and random.random() < 0.4:
                strategy_name = "failure_aware"

            strategy_template = STRATEGIES.get(strategy_name, STRATEGIES["direct"])

            # Build prompt
            failure_context = ""
            if tk and tk.failure_patterns:
                failure_context = "; ".join(tk.failure_patterns[-3:])

            prompt = strategy_template.format(
                examples=examples,
                failure_context=failure_context or "unknown pattern"
            )

            # Pick model
            model = random.choice(models)

            # 3. EXPERIMENT: Generate transform
            print(f"[{attempt_num+1}/{max_attempts}] task{tn:03d} "
                  f"strategy={strategy_name} model={model}",
                  end=" ", flush=True)

            response = self._call_llm(prompt, model=model)
            code = extract_code(response) if response else None

            if not code:
                attempt = Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=0, total=0,
                    error="no_code_extracted"
                )
                self.memory.record_attempt(attempt)
                print("-> no code", flush=True)
                continue

            # 4. EVALUATE: Security Radar
            try:
                ns = {"np": np, "numpy": np}
                exec(code.strip(), ns)
                transform_fn = ns.get("transform")
                if not transform_fn:
                    raise ValueError("no transform function")
                correct, total = verify_transform(transform_fn, task)
            except Exception as e:
                attempt = Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=0, total=0,
                    error=f"exec_error: {str(e)[:100]}", code=code
                )
                self.memory.record_attempt(attempt)
                print(f"-> exec error", flush=True)
                continue

            verified = correct == total and total > 0

            # 5. LEARN
            if verified:
                # Success!
                solved_this_cycle += 1
                attempts_this_cycle += 1
                attempt = Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=True, correct=correct, total=total,
                    code=code
                )
                self.memory.record_attempt(attempt)
                unsolved.remove(tn)
                print(f"-> SOLVED {correct}/{total}", flush=True)

            else:
                attempts_this_cycle += 1
                # Reflect on failure
                insight = ""
                if correct > 0 and code:
                    # Near miss — worth reflecting
                    insight = self._reflect_on_failure(task, tn, code, correct, total)

                error_desc = f"partial_{correct}/{total}"
                if correct == 0:
                    error_desc = "zero_correct"

                attempt = Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=correct, total=total,
                    error=error_desc, code=code, insight=insight
                )
                self.memory.record_attempt(attempt)

                if insight:
                    print(f"-> {correct}/{total} insight: {insight[:80]}", flush=True)
                else:
                    print(f"-> {correct}/{total}", flush=True)

            # 7. Periodic reporting
            if (attempt_num + 1) % 10 == 0:
                print(f"\n--- Cycle progress: {solved_this_cycle} solved "
                      f"in {attempts_this_cycle} attempts "
                      f"({solved_this_cycle/max(attempts_this_cycle,1):.0%}) ---")
                print(self.memory.get_strategy_report())
                print()

        # Final report
        print(f"\n{'='*60}")
        print(f"Cycle complete: {solved_this_cycle} new solves "
              f"in {attempts_this_cycle} attempts")
        print(f"Cumulative: {self.memory.total_solves} total solves "
              f"/ {self.memory.total_attempts} total attempts")
        print(self.memory.get_strategy_report())

        return solved_this_cycle


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Autonomous ARC Scientist")
    ap.add_argument("--task-dir", default=".", help="Directory with task JSON files")
    ap.add_argument("--memory", default="arc_scientist_memory.json")
    ap.add_argument("--attempts", type=int, default=50, help="Attempts per cycle")
    ap.add_argument("--cycles", type=int, default=1, help="Number of learning cycles")
    ap.add_argument("--models", default="kimi,qwen3")
    args = ap.parse_args()

    scientist = ARCScientist(
        task_dir=args.task_dir,
        memory_path=args.memory,
    )

    models = [m.strip() for m in args.models.split(",")]

    for cycle in range(args.cycles):
        print(f"\n{'='*60}")
        print(f"LEARNING CYCLE {cycle + 1}/{args.cycles}")
        print(f"{'='*60}\n")

        solved = scientist.run_cycle(max_attempts=args.attempts, models=models)

        if cycle < args.cycles - 1:
            print(f"\nResting 10s before next cycle...")
            time.sleep(10)

    # Save final memory
    scientist.memory.save()
    print(f"\nMemory saved to {args.memory}")
    print(f"Total: {scientist.memory.total_solves} tasks solved "
          f"across {scientist.memory.total_attempts} attempts")


if __name__ == "__main__":
    main()
