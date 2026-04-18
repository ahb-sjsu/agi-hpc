"""Autonomous ARC Scientist (Erebus) — self-improving solver.

Implements a closed-loop scientific reasoning cycle:
  1. OBSERVE: Pick an unsolved task (clustered by similarity to solved)
  2. HYPOTHESIZE: Form a theory (via LLM, strategy chosen by Thompson sampling)
  3. EXPERIMENT: Generate candidate Python transforms
  4. EVALUATE: Security Radar — verify on ALL examples
  5. REFLECT: Structured failure classification + cross-task meta-patterns
  6. ADAPT: Update strategy weights, extract meta-patterns from failures
  7. REPEAT

Erebus's three self-requested improvements (2026-04-18):
  1. Hybrid strategy: direct first, diagnostic only on exec failure
  2. Structured reflection: classify errors as {perception, reasoning,
     execution, specification}, find cross-task patterns
  3. Task clustering: group unsolved by similarity to solved tasks,
     exploit structural priors (shape, color, symmetry)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Error taxonomy — Erebus's request #1
# ═══════════════════════════════════════════════════════════════

ERROR_TYPES = ["perception", "reasoning", "execution", "specification"]

CLASSIFY_PROMPT = (
    "An ARC-AGI transform got {correct}/{total} examples right.\n\n"
    "Task examples:\n{examples}\n"
    "Failed code:\n```python\n{code}\n```\n\n"
    "Classify the error as ONE of:\n"
    "- perception: misidentified objects, colors, boundaries, shapes\n"
    "- reasoning: wrong logic about how objects relate or transform\n"
    "- execution: code bug, off-by-one, wrong numpy operation\n"
    "- specification: solved a different valid pattern than intended\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{"error_type": "...", "diagnosis": "one sentence", '
    '"similar_to": "name a common ARC pattern this resembles"}}'
)


# ═══════════════════════════════════════════════════════════════
# Task fingerprinting — Erebus's request #3
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskFingerprint:
    """Lightweight structural signature for clustering tasks."""
    task_num: int
    input_h: int
    input_w: int
    output_h: int
    output_w: int
    same_shape: bool
    n_colors_in: int
    n_colors_out: int
    ratio_h: float
    ratio_w: float
    has_symmetry: bool
    dominant_color: int


def fingerprint_task(task: dict, tn: int) -> TaskFingerprint:
    """Extract structural signature from first training example."""
    ex = task.get("train", [{}])[0]
    inp = np.array(ex.get("input", [[]]))
    out = np.array(ex.get("output", [[]]))
    ih, iw = inp.shape if inp.ndim == 2 else (0, 0)
    oh, ow = out.shape if out.ndim == 2 else (0, 0)

    colors_in = len(set(inp.flatten())) if inp.size else 0
    colors_out = len(set(out.flatten())) if out.size else 0

    # Simple symmetry check
    has_sym = False
    if inp.size and inp.ndim == 2:
        has_sym = (np.array_equal(inp, inp[::-1]) or
                   np.array_equal(inp, inp[:, ::-1]) or
                   (ih == iw and np.array_equal(inp, inp.T)))

    dominant = int(Counter(inp.flatten()).most_common(1)[0][0]) if inp.size else 0

    return TaskFingerprint(
        task_num=tn,
        input_h=ih, input_w=iw,
        output_h=oh, output_w=ow,
        same_shape=(ih == oh and iw == ow),
        n_colors_in=colors_in,
        n_colors_out=colors_out,
        ratio_h=oh / max(ih, 1),
        ratio_w=ow / max(iw, 1),
        has_symmetry=has_sym,
        dominant_color=dominant,
    )


def task_distance(a: TaskFingerprint, b: TaskFingerprint) -> float:
    """Simple distance metric between two task fingerprints."""
    d = 0.0
    d += 0 if a.same_shape == b.same_shape else 2.0
    d += abs(a.ratio_h - b.ratio_h) + abs(a.ratio_w - b.ratio_w)
    d += abs(a.n_colors_in - b.n_colors_in) * 0.5
    d += abs(a.n_colors_out - b.n_colors_out) * 0.5
    d += abs(a.input_h - b.input_h) * 0.1 + abs(a.input_w - b.input_w) * 0.1
    d += 0 if a.has_symmetry == b.has_symmetry else 1.0
    return d


# ═══════════════════════════════════════════════════════════════
# Episodic memory
# ═══════════════════════════════════════════════════════════════

@dataclass
class Attempt:
    task_num: int
    timestamp: str
    strategy: str
    model: str
    verified: bool
    correct: int
    total: int
    error: str = ""
    error_type: str = ""       # perception|reasoning|execution|specification
    code: str = ""
    insight: str = ""
    similar_to: str = ""       # which ARC pattern this resembles


@dataclass
class TaskKnowledge:
    task_num: int
    attempts: list = field(default_factory=list)
    solved: bool = False
    best_correct: int = 0
    best_total: int = 0
    strategies_tried: list = field(default_factory=list)
    failure_patterns: list = field(default_factory=list)
    error_types: list = field(default_factory=list)   # classified errors
    hypotheses: list = field(default_factory=list)


@dataclass
class StrategyStats:
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
        alpha = self.successes + self.partial_successes * 0.3 + 1
        beta = self.failures + 1
        return alpha / (alpha + beta)


class EpisodicMemory:
    def __init__(self, path: str = "arc_scientist_memory.json"):
        self.path = Path(path)
        self.tasks: dict[int, TaskKnowledge] = {}
        self.strategies: dict[str, StrategyStats] = {}
        self.meta_patterns: list[str] = []  # cross-task error patterns
        self.global_insights: list[str] = []
        self.total_attempts: int = 0
        self.total_solves: int = 0
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for tn, tk in data.get("tasks", {}).items():
                    self.tasks[int(tn)] = TaskKnowledge(**{
                        k: v for k, v in tk.items()
                        if k in TaskKnowledge.__dataclass_fields__
                    })
                for name, ss in data.get("strategies", {}).items():
                    self.strategies[name] = StrategyStats(**{
                        k: v for k, v in ss.items()
                        if k in StrategyStats.__dataclass_fields__
                    })
                self.meta_patterns = data.get("meta_patterns", [])
                self.global_insights = data.get("global_insights", [])
                self.total_attempts = data.get("total_attempts", 0)
                self.total_solves = data.get("total_solves", 0)
            except Exception:
                pass

    def save(self):
        data = {
            "tasks": {str(k): asdict(v) for k, v in self.tasks.items()},
            "strategies": {k: asdict(v) for k, v in self.strategies.items()},
            "meta_patterns": self.meta_patterns[-20:],
            "global_insights": self.global_insights[-20:],
            "total_attempts": self.total_attempts,
            "total_solves": self.total_solves,
            "last_updated": datetime.now().isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def record_attempt(self, attempt: Attempt):
        self.total_attempts += 1
        tn = attempt.task_num

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
        if attempt.error_type and attempt.error_type not in tk.error_types:
            tk.error_types.append(attempt.error_type)

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
        ratio = attempt.correct / max(attempt.total, 1)
        ss.avg_correct_ratio = (
            ss.avg_correct_ratio * (ss.attempts - 1) + ratio
        ) / ss.attempts

        self.save()

    def detect_meta_patterns(self):
        """Find cross-task error patterns — Erebus's request #2."""
        type_counts = Counter()
        type_tasks = {}
        for tn, tk in self.tasks.items():
            for et in tk.error_types:
                type_counts[et] += 1
                type_tasks.setdefault(et, []).append(tn)

        patterns = []
        for et, count in type_counts.most_common():
            if count >= 3:
                tasks = type_tasks[et][:5]
                task_str = ", ".join(f"task{t:03d}" for t in tasks)
                patterns.append(
                    f"{et} errors ({count}x): {task_str}"
                )

        # Check for similar_to clusters
        sim_counts = Counter()
        for tk in self.tasks.values():
            for attempt in tk.attempts:
                sim = attempt.get("similar_to", "")
                if sim:
                    sim_counts[sim] += 1
        for sim, count in sim_counts.most_common(5):
            if count >= 2:
                patterns.append(f"pattern '{sim}' appears {count}x across tasks")

        self.meta_patterns = patterns
        self.save()
        return patterns

    def get_unsolved_tasks(self, all_tasks: list[int]) -> list[int]:
        unsolved = [t for t in all_tasks if t not in self.tasks or not self.tasks[t].solved]
        unsolved.sort(
            key=lambda t: self.tasks[t].best_correct if t in self.tasks else 0,
            reverse=True
        )
        return unsolved

    def pick_strategy(self) -> str:
        if not self.strategies:
            return "direct"
        names = list(self.strategies.keys())
        weights = [self.strategies[n].weight for n in names]
        total = sum(weights)
        if total == 0:
            return random.choice(names)
        probs = [w / total for w in weights]
        return np.random.choice(names, p=probs)

    def get_strategy_report(self) -> str:
        lines = ["Strategy Performance:"]
        ranked = sorted(self.strategies.values(),
                       key=lambda s: s.weight, reverse=True)
        for s in ranked:
            lines.append(
                f"  {s.name:20s} w={s.weight:.2f} "
                f"({s.successes}/{s.attempts} solved, "
                f"avg={s.avg_correct_ratio:.2f})"
            )
        if self.meta_patterns:
            lines.append("\nMeta-patterns detected:")
            for p in self.meta_patterns[:5]:
                lines.append(f"  {p}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Prompt strategies
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
    "diagnostic": (
        "Previous attempts on this ARC puzzle failed.\n"
        "Error type: {error_type}\n"
        "Diagnosis: {diagnosis}\n"
        "Similar pattern: {similar_to}\n\n"
        "Try a FUNDAMENTALLY DIFFERENT approach:\n\n{examples}\n"
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
# The Scientist — Erebus
# ═══════════════════════════════════════════════════════════════

class ARCScientist:
    """Erebus — autonomous agent that learns to solve ARC tasks."""

    def __init__(self, task_dir: str, memory_path: str = "arc_scientist_memory.json",
                 llm_token: str = "", llm_base_url: str = "https://ellm.nrp-nautilus.io/v1"):
        self.task_dir = Path(task_dir)
        self.memory = EpisodicMemory(memory_path)
        self.llm_token = llm_token or os.environ.get("NRP_LLM_TOKEN", "")
        self.llm_base_url = llm_base_url
        self.client = None
        self._init_client()

        self.all_tasks = sorted([
            int(f.stem[4:]) for f in self.task_dir.glob("task*.json")
        ])

        # Build fingerprint index for task clustering
        self.fingerprints: dict[int, TaskFingerprint] = {}
        for tn in self.all_tasks:
            try:
                task = self._load_task(tn)
                self.fingerprints[tn] = fingerprint_task(task, tn)
            except Exception:
                pass

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
        except Exception:
            return None

    def _structured_reflection(self, task: dict, tn: int, code: str,
                                correct: int, total: int) -> dict:
        """Structured failure classification — Erebus's request #1.

        Returns {error_type, diagnosis, similar_to} or empty dict.
        """
        examples = format_examples(task, max_examples=2)
        prompt = CLASSIFY_PROMPT.format(
            correct=correct, total=total,
            examples=examples, code=code
        )
        response = self._call_llm(prompt, model="kimi")
        if not response:
            return {}

        # Parse JSON from response
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                # Validate error_type
                if result.get("error_type") not in ERROR_TYPES:
                    result["error_type"] = "reasoning"
                return result
        except (json.JSONDecodeError, KeyError):
            pass

        return {"error_type": "reasoning", "diagnosis": response.strip()[:200]}

    def _pick_task_by_similarity(self, unsolved: list[int]) -> int:
        """Task clustering — Erebus's request #3.

        Pick an unsolved task similar to a recently solved one.
        """
        solved_tasks = [tn for tn, tk in self.memory.tasks.items() if tk.solved]
        if not solved_tasks or not unsolved:
            return random.choice(unsolved[:50])

        # Find unsolved tasks most similar to any solved task
        scored = []
        for utn in unsolved:
            if utn not in self.fingerprints:
                continue
            ufp = self.fingerprints[utn]
            min_dist = float("inf")
            for stn in solved_tasks:
                if stn not in self.fingerprints:
                    continue
                d = task_distance(ufp, self.fingerprints[stn])
                min_dist = min(min_dist, d)
            scored.append((min_dist, utn))

        if not scored:
            return random.choice(unsolved[:50])

        scored.sort()
        # Pick from top 10 closest (with some randomness)
        top = [tn for _, tn in scored[:10]]
        return random.choice(top)

    def run_cycle(self, max_attempts: int = 50, models: list[str] = None):
        """Run one full learning cycle with Erebus's improvements."""
        if models is None:
            models = ["kimi", "qwen3"]

        unsolved = self.memory.get_unsolved_tasks(self.all_tasks)
        print(f"Erebus starting learning cycle")
        print(f"  Tasks: {len(self.all_tasks)} total, {len(unsolved)} unsolved")
        print(f"  Memory: {self.memory.total_attempts} prior attempts, "
              f"{self.memory.total_solves} solves")
        print(f"  Fingerprints: {len(self.fingerprints)} tasks indexed")
        if self.memory.strategies:
            print(self.memory.get_strategy_report())
        print()

        solved_this_cycle = 0
        attempts_this_cycle = 0

        for attempt_num in range(max_attempts):
            if not unsolved:
                print("All tasks solved!")
                break

            # ── 1. OBSERVE: Pick a task ──
            # Three modes: similarity-guided, near-miss, exploration
            r = random.random()
            if r < 0.4 and self.memory.total_solves > 0:
                # Similarity-guided: pick task similar to one we solved
                tn = self._pick_task_by_similarity(unsolved)
            elif r < 0.6 and any(
                t in self.memory.tasks and self.memory.tasks[t].best_correct > 0
                for t in unsolved
            ):
                # Near-miss exploitation
                near_misses = [t for t in unsolved
                               if t in self.memory.tasks
                               and self.memory.tasks[t].best_correct > 0]
                tn = random.choice(near_misses[:10])
            else:
                # Pure exploration
                tn = random.choice(unsolved[:50])

            task = self._load_task(tn)
            examples = format_examples(task)
            tk = self.memory.tasks.get(tn)

            # ── 2. HYPOTHESIZE: Hybrid strategy (Erebus's request #1) ──
            # Direct first. Only use diagnostic if we have classified failures.
            if tk and tk.error_types and random.random() < 0.5:
                # We have prior classified failures — use diagnostic strategy
                strategy_name = "diagnostic"
                # Get most recent classified failure
                last_classified = {}
                for a in reversed(tk.attempts):
                    if a.get("error_type"):
                        last_classified = a
                        break
                strategy_template = STRATEGIES["diagnostic"]
                prompt = strategy_template.format(
                    examples=examples,
                    error_type=last_classified.get("error_type", "unknown"),
                    diagnosis=last_classified.get("insight", "unknown"),
                    similar_to=last_classified.get("similar_to", "unknown"),
                )
            else:
                # Thompson sampling over non-diagnostic strategies
                strategy_name = self.memory.pick_strategy()
                if strategy_name == "diagnostic":
                    strategy_name = "direct"  # don't pick diagnostic without data
                strategy_template = STRATEGIES.get(strategy_name, STRATEGIES["direct"])
                prompt = strategy_template.format(
                    examples=examples,
                    failure_context="",
                    error_type="", diagnosis="", similar_to="",
                )

            model = random.choice(models)

            # ── 3. EXPERIMENT ──
            print(f"[{attempt_num+1}/{max_attempts}] task{tn:03d} "
                  f"strategy={strategy_name} model={model}",
                  end=" ", flush=True)

            response = self._call_llm(prompt, model=model)
            code = extract_code(response) if response else None

            if not code:
                self.memory.record_attempt(Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=0, total=0,
                    error="no_code_extracted"
                ))
                print("-> no code", flush=True)
                continue

            # ── 4. EVALUATE: Security Radar ──
            try:
                ns = {"np": np, "numpy": np}
                exec(code.strip(), ns)
                transform_fn = ns.get("transform")
                if not transform_fn:
                    raise ValueError("no transform function")
                correct, total = verify_transform(transform_fn, task)
            except Exception as e:
                self.memory.record_attempt(Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=0, total=0,
                    error_type="execution",
                    error=f"exec: {str(e)[:80]}", code=code
                ))
                print("-> exec error", flush=True)
                continue

            verified = correct == total and total > 0

            # ── 5. REFLECT ──
            if verified:
                solved_this_cycle += 1
                attempts_this_cycle += 1
                self.memory.record_attempt(Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=True, correct=correct, total=total,
                    code=code
                ))
                unsolved.remove(tn)
                print(f"-> SOLVED {correct}/{total}", flush=True)
            else:
                attempts_this_cycle += 1

                # Structured reflection — classify the error
                reflection = {}
                if correct > 0 and code:
                    reflection = self._structured_reflection(
                        task, tn, code, correct, total)

                error_type = reflection.get("error_type", "")
                diagnosis = reflection.get("diagnosis", "")
                similar_to = reflection.get("similar_to", "")

                error_desc = f"partial_{correct}/{total}"
                if correct == 0:
                    error_desc = "zero_correct"

                self.memory.record_attempt(Attempt(
                    task_num=tn, timestamp=datetime.now().isoformat(),
                    strategy=strategy_name, model=model,
                    verified=False, correct=correct, total=total,
                    error=error_desc, error_type=error_type,
                    code=code, insight=diagnosis, similar_to=similar_to,
                ))

                if diagnosis:
                    print(f"-> {correct}/{total} [{error_type}] {diagnosis[:60]}",
                          flush=True)
                else:
                    print(f"-> {correct}/{total}", flush=True)

            # ── 6. ADAPT: periodic meta-pattern detection ──
            if (attempt_num + 1) % 10 == 0:
                patterns = self.memory.detect_meta_patterns()
                print(f"\n--- Progress: {solved_this_cycle} solved "
                      f"in {attempts_this_cycle} attempts "
                      f"({solved_this_cycle/max(attempts_this_cycle,1):.0%}) ---")
                print(self.memory.get_strategy_report())
                print()

        # Final report
        self.memory.detect_meta_patterns()
        print(f"\n{'='*60}")
        print(f"Cycle complete: {solved_this_cycle} new solves "
              f"in {attempts_this_cycle} attempts")
        print(f"Cumulative: {self.memory.total_solves} total solves "
              f"/ {self.memory.total_attempts} total attempts")
        print(self.memory.get_strategy_report())

        return solved_this_cycle


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Erebus — Autonomous ARC Scientist")
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
        print(f"EREBUS LEARNING CYCLE {cycle + 1}/{args.cycles}")
        print(f"{'='*60}\n")

        solved = scientist.run_cycle(max_attempts=args.attempts, models=models)

        if cycle < args.cycles - 1:
            print(f"\nConsolidating before next cycle...")
            time.sleep(5)

    scientist.memory.save()
    print(f"\nMemory saved to {args.memory}")
    print(f"Total: {scientist.memory.total_solves} tasks solved "
          f"across {scientist.memory.total_attempts} attempts")


if __name__ == "__main__":
    main()
