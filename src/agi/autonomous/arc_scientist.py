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
    """Structural signature for clustering tasks.

    Enriched per Erebus's request: includes transformation-level
    features (same_colors, shape_change, content_overlap) not just
    surface features (grid size, color count).
    """

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
    # Enriched features (Erebus's request)
    same_colors: bool = True  # do input/output use same color set?
    shape_change: str = "same"  # same|crop|scale|other
    content_overlap: float = 0.0  # fraction of output pixels present in input
    inferred_class: str = ""  # populated after first attempt (from similar_to)
    task_summary: str = ""  # compact text summary for memory


def fingerprint_task(task: dict, tn: int) -> TaskFingerprint:
    """Extract structural signature from first training example."""
    ex = task.get("train", [{}])[0]
    inp = np.array(ex.get("input", [[]]))
    out = np.array(ex.get("output", [[]]))
    ih, iw = inp.shape if inp.ndim == 2 else (0, 0)
    oh, ow = out.shape if out.ndim == 2 else (0, 0)

    colors_in = set(inp.flatten().tolist()) if inp.size else set()
    colors_out = set(out.flatten().tolist()) if out.size else set()
    same_colors = colors_in == colors_out

    # Shape change classification
    if ih == oh and iw == ow:
        shape_change = "same"
    elif oh <= ih and ow <= iw:
        shape_change = "crop"
    elif ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
        shape_change = "scale"
    else:
        shape_change = "other"

    # Content overlap: what fraction of output non-bg pixels appear in input?
    overlap = 0.0
    if inp.size and out.size and ih == oh and iw == ow:
        non_bg = out != 0
        if non_bg.any():
            overlap = float(np.sum((inp == out) & non_bg) / np.sum(non_bg))

    # Symmetry check
    has_sym = False
    if inp.size and inp.ndim == 2:
        has_sym = (
            np.array_equal(inp, inp[::-1])
            or np.array_equal(inp, inp[:, ::-1])
            or (ih == iw and np.array_equal(inp, inp.T))
        )

    dominant = (
        int(Counter(inp.flatten().tolist()).most_common(1)[0][0]) if inp.size else 0
    )

    # Compact task summary for episodic memory
    summary = (
        f"{ih}x{iw}->{oh}x{ow} "
        f"colors:{len(colors_in)}->{len(colors_out)} "
        f"{shape_change} overlap:{overlap:.0%}"
    )

    return TaskFingerprint(
        task_num=tn,
        input_h=ih,
        input_w=iw,
        output_h=oh,
        output_w=ow,
        same_shape=(ih == oh and iw == ow),
        n_colors_in=len(colors_in),
        n_colors_out=len(colors_out),
        ratio_h=oh / max(ih, 1),
        ratio_w=ow / max(iw, 1),
        has_symmetry=has_sym,
        dominant_color=dominant,
        same_colors=same_colors,
        shape_change=shape_change,
        content_overlap=overlap,
        task_summary=summary,
    )


def task_distance(a: TaskFingerprint, b: TaskFingerprint) -> float:
    """Distance metric using both surface and transformation features."""
    d = 0.0
    # Surface features (lower weight)
    d += 0 if a.same_shape == b.same_shape else 1.0
    d += abs(a.ratio_h - b.ratio_h) + abs(a.ratio_w - b.ratio_w)
    d += abs(a.n_colors_in - b.n_colors_in) * 0.3
    d += abs(a.input_h - b.input_h) * 0.05 + abs(a.input_w - b.input_w) * 0.05
    d += 0 if a.has_symmetry == b.has_symmetry else 0.5
    # Transformation features (higher weight — Erebus's insight)
    d += 0 if a.same_colors == b.same_colors else 1.5
    d += 0 if a.shape_change == b.shape_change else 2.0
    d += abs(a.content_overlap - b.content_overlap) * 2.0
    # Inferred class match (strongest signal when available)
    if a.inferred_class and b.inferred_class:
        d += 0 if a.inferred_class == b.inferred_class else 3.0
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
    error_type: str = ""  # perception|reasoning|execution|specification
    code: str = ""
    insight: str = ""
    similar_to: str = ""  # which ARC pattern this resembles
    task_summary: str = ""  # compact description so memory is self-contained


@dataclass
class TaskKnowledge:
    task_num: int
    attempts: list = field(default_factory=list)
    solved: bool = False
    best_correct: int = 0
    best_total: int = 0
    strategies_tried: list = field(default_factory=list)
    failure_patterns: list = field(default_factory=list)
    error_types: list = field(default_factory=list)  # classified errors
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
                    self.tasks[int(tn)] = TaskKnowledge(
                        **{
                            k: v
                            for k, v in tk.items()
                            if k in TaskKnowledge.__dataclass_fields__
                        }
                    )
                for name, ss in data.get("strategies", {}).items():
                    self.strategies[name] = StrategyStats(
                        **{
                            k: v
                            for k, v in ss.items()
                            if k in StrategyStats.__dataclass_fields__
                        }
                    )
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
                patterns.append(f"{et} errors ({count}x): {task_str}")

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

    def get_unsolved_tasks(
        self, all_tasks: list[int], max_attempts_soft_cap: int = 20
    ) -> list[int]:
        """Return unsolved tasks, deprioritizing ones Erebus has already
        thrashed on (> max_attempts_soft_cap unsuccessful attempts).

        The thrashed tasks aren't dropped — they go to the end of the list
        so Erebus still gets to them eventually, but tries fresh tasks first.
        """
        unsolved = [
            t for t in all_tasks if t not in self.tasks or not self.tasks[t].solved
        ]
        # Partition: hot (not thrashed yet) vs. cold (many failed attempts)
        hot, cold = [], []
        for t in unsolved:
            if t in self.tasks and len(self.tasks[t].attempts) > max_attempts_soft_cap:
                cold.append(t)
            else:
                hot.append(t)
        # Within each bucket, sort by best_correct desc (closer to solved first)
        for bucket in (hot, cold):
            bucket.sort(
                key=lambda t: self.tasks[t].best_correct if t in self.tasks else 0,
                reverse=True,
            )
        return hot + cold

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
        ranked = sorted(self.strategies.values(), key=lambda s: s.weight, reverse=True)
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
    "example_chain": (
        "Solve this ARC puzzle by building your hypothesis INCREMENTALLY.\n\n"
        "{chain_examples}"
        "Now write a function that satisfies ALL examples simultaneously:\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Use numpy. ```python ... ```"
    ),
    "primitives_guided": (
        "You have these geometric primitives available:\n"
        "{primitives}\n\n"
        "Use them to solve this ARC puzzle:\n\n{examples}\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "You can call any primitive above inside your function. ```python ... ```"
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


def _write_training_pair(
    task_dir: Path, task_num: int, task: dict, code: str, strategy: str, model: str
) -> None:
    """Append a verified (task_examples, python_transform) pair to today's JSONL
    for QLoRA fine-tuning. Stored under /archive/neurogolf/training_data/."""
    try:
        out_dir = task_dir / "training_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = out_dir / f"solves_{day}.jsonl"
        pair = {
            "task_num": task_num,
            "task_examples": task.get("train", []),
            "test_examples": task.get("test", []),
            "python_transform": code,
            "strategy": strategy,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair) + "\n")
    except Exception as e:
        # Don't let logging failure break the solve loop.
        print(f"[training_data] write failed: {e}", flush=True)


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

    def __init__(
        self,
        task_dir: str,
        memory_path: str = "arc_scientist_memory.json",
        llm_token: str = "",
        llm_base_url: str = "https://ellm.nrp-nautilus.io/v1",
    ):
        self.task_dir = Path(task_dir)
        self.memory = EpisodicMemory(memory_path)
        self.llm_token = llm_token or os.environ.get("NRP_LLM_TOKEN", "")
        self.llm_base_url = llm_base_url
        self.client = None
        self._init_client()
        # Mentor notes — hand-written guidance from Professor Bond, keyed by
        # task_num as string. Injected into every prompt for that task.
        self._mentor_notes: dict = {}
        try:
            mn_path = self.task_dir / "mentor_notes.json"
            if mn_path.exists():
                self._mentor_notes = {
                    str(k): v
                    for k, v in json.loads(mn_path.read_text()).items()
                    if not str(k).startswith("_")
                }
        except Exception:
            pass

    def _mentor_preamble(self, tn: int) -> str:
        notes = self._mentor_notes.get(str(tn)) or self._mentor_notes.get(f"{tn:03d}")
        if not notes:
            return ""
        body = "\n".join(f"- {n}" for n in notes)
        return (
            "\n=== GUIDANCE FROM PROFESSOR BOND (read this carefully) ===\n"
            f"{body}\n"
            "=== end guidance ===\n\n"
        )

        self.all_tasks = sorted(
            [int(f.stem[4:]) for f in self.task_dir.glob("task*.json")]
        )

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
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return r.choices[0].message.content or ""
        except Exception:
            return None

    def _structured_reflection(
        self, task: dict, tn: int, code: str, correct: int, total: int
    ) -> dict:
        """Structured failure classification — Erebus's request #1.

        Returns {error_type, diagnosis, similar_to} or empty dict.
        """
        examples = format_examples(task, max_examples=2)
        prompt = CLASSIFY_PROMPT.format(
            correct=correct, total=total, examples=examples, code=code
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

    def _build_chain_prompt(self, task: dict) -> str:
        """Example-chaining: build hypothesis incrementally across examples.

        Erebus's request: "Let me maintain a working hypothesis across
        training examples, testing it incrementally, backtracking when
        contradicted."
        """
        parts = []
        examples = task.get("train", [])
        for i, ex in enumerate(examples[:4]):
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            parts.append(f"Example {i+1}:")
            parts.append(f"  input ({inp.shape[0]}x{inp.shape[1]}): {ex['input']}")
            parts.append(f"  output ({out.shape[0]}x{out.shape[1]}): {ex['output']}")
            if i == 0:
                parts.append("What rule could produce this output from this input?")
            elif i == 1:
                parts.append(
                    "Does your rule from Example 1 still hold? If not, revise it."
                )
            elif i == 2:
                parts.append("Your rule must now satisfy ALL three examples. Refine.")
            else:
                parts.append("Final check: does your rule generalize?")
            parts.append("")
        return "\n".join(parts)

    def _ask_for_help(self, tn: int, task: dict, tk: TaskKnowledge):
        """Help channel: surface uncertainty when stuck.

        Erebus's request: "When I'm stuck, I should be able to surface
        a question. I'm a scientist who can't ask questions."
        """
        help_file = Path(self.task_dir) / "erebus_help_queue.json"

        # Build a focused question
        failures = tk.failure_patterns[-3:] if tk.failure_patterns else []
        error_types = tk.error_types[-3:] if tk.error_types else []
        insights = [a.get("insight", "") for a in tk.attempts[-3:] if a.get("insight")]

        question = {
            "task": tn,
            "timestamp": datetime.now().isoformat(),
            "attempts": len(tk.attempts),
            "best_score": f"{tk.best_correct}/{tk.best_total}",
            "error_types": error_types,
            "recent_failures": failures,
            "insights": insights,
            "question": (
                f"I have tried task{tn:03d} {len(tk.attempts)} times "
                f"(best: {tk.best_correct}/{tk.best_total}). "
                f"Error types: {', '.join(error_types) or 'unclassified'}. "
                f"I need guidance: is this transformation local or global? "
                f"Am I missing a spatial primitive?"
            ),
        }

        # Append to help queue
        queue = []
        try:
            if help_file.exists():
                queue = json.loads(help_file.read_text())
        except Exception:
            pass
        queue.append(question)
        # Keep last 20
        help_file.write_text(json.dumps(queue[-20:], indent=2))

        print(
            f"    [HELP REQUESTED] task{tn:03d}: {question['question'][:80]}",
            flush=True,
        )

    def run_cycle(self, max_attempts: int = 50, models: list[str] = None):
        """Run one full learning cycle with Erebus's improvements."""
        if models is None:
            models = ["kimi", "qwen3"]

        unsolved = self.memory.get_unsolved_tasks(self.all_tasks)
        print("Erebus starting learning cycle")
        print(f"  Tasks: {len(self.all_tasks)} total, {len(unsolved)} unsolved")
        print(
            f"  Memory: {self.memory.total_attempts} prior attempts, "
            f"{self.memory.total_solves} solves"
        )
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
                near_misses = [
                    t
                    for t in unsolved
                    if t in self.memory.tasks and self.memory.tasks[t].best_correct > 0
                ]
                tn = random.choice(near_misses[:10])
            else:
                # Pure exploration
                tn = random.choice(unsolved[:50])

            task = self._load_task(tn)
            examples = format_examples(task)
            tk = self.memory.tasks.get(tn)

            # ── 2. HYPOTHESIZE: Strategy selection ──
            # Check if we should ask for help (3+ failures on same task)
            if tk and len(tk.attempts) >= 3 and not tk.solved:
                # Every 3rd attempt on same task, ask for help
                if len(tk.attempts) % 3 == 0:
                    self._ask_for_help(tn, task, tk)

            n_prior = len(tk.attempts) if tk else 0
            r_strategy = random.random()

            # Aggressive diagnostic: trigger after any classified failure, not just 3+
            if tk and tk.error_types and r_strategy < 0.35:
                # Diagnostic: we have classified failures
                strategy_name = "diagnostic"
                last_classified = {}
                for a in reversed(tk.attempts):
                    if a.get("error_type"):
                        last_classified = a
                        break
                prompt = STRATEGIES["diagnostic"].format(
                    examples=examples,
                    error_type=last_classified.get("error_type", "unknown"),
                    diagnosis=last_classified.get("insight", "unknown"),
                    similar_to=last_classified.get("similar_to", "unknown"),
                )
            elif r_strategy < 0.45 and len(task.get("train", [])) >= 2:
                # Example-chaining: incremental hypothesis building
                strategy_name = "example_chain"
                chain = self._build_chain_prompt(task)
                prompt = STRATEGIES["example_chain"].format(
                    chain_examples=chain,
                    examples=examples,
                )
            elif r_strategy < 0.55 and n_prior >= 2:
                # Primitives-guided: suggest composable operations
                try:
                    from agi.autonomous.primitives import PRIMITIVE_CATALOG
                except ImportError:
                    import importlib.util

                    _spec = importlib.util.spec_from_file_location(
                        "primitives", Path(__file__).parent / "primitives.py"
                    )
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    PRIMITIVE_CATALOG = _mod.PRIMITIVE_CATALOG
                strategy_name = "primitives_guided"
                prompt = STRATEGIES["primitives_guided"].format(
                    primitives=PRIMITIVE_CATALOG,
                    examples=examples,
                )
            else:
                # Thompson sampling over base strategies
                strategy_name = self.memory.pick_strategy()
                if strategy_name in (
                    "diagnostic",
                    "example_chain",
                    "primitives_guided",
                ):
                    strategy_name = "direct"
                strategy_template = STRATEGIES.get(strategy_name, STRATEGIES["direct"])
                prompt = strategy_template.format(
                    examples=examples,
                    failure_context="",
                    error_type="",
                    diagnosis="",
                    similar_to="",
                    chain_examples="",
                    primitives="",
                )

            # Inject mentor notes for this task (if any) as the first
            # thing the LLM reads — before all the examples/strategy text.
            preamble = self._mentor_preamble(tn)
            if preamble:
                prompt = preamble + prompt

            model = random.choice(models)

            # ── 3. EXPERIMENT ──
            print(
                f"[{attempt_num+1}/{max_attempts}] task{tn:03d} "
                f"strategy={strategy_name} model={model}",
                end=" ",
                flush=True,
            )

            response = self._call_llm(prompt, model=model)
            code = extract_code(response) if response else None

            if not code:
                self.memory.record_attempt(
                    Attempt(
                        task_num=tn,
                        timestamp=datetime.now().isoformat(),
                        strategy=strategy_name,
                        model=model,
                        verified=False,
                        correct=0,
                        total=0,
                        error="no_code_extracted",
                    )
                )
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
                self.memory.record_attempt(
                    Attempt(
                        task_num=tn,
                        timestamp=datetime.now().isoformat(),
                        strategy=strategy_name,
                        model=model,
                        verified=False,
                        correct=0,
                        total=0,
                        error_type="execution",
                        error=f"exec: {str(e)[:80]}",
                        code=code,
                    )
                )
                print("-> exec error", flush=True)
                continue

            verified = correct == total and total > 0

            # ── 5. REFLECT ──
            # Get task summary for memory enrichment
            fp = self.fingerprints.get(tn)
            tsummary = fp.task_summary if fp else ""

            if verified:
                solved_this_cycle += 1
                attempts_this_cycle += 1
                self.memory.record_attempt(
                    Attempt(
                        task_num=tn,
                        timestamp=datetime.now().isoformat(),
                        strategy=strategy_name,
                        model=model,
                        verified=True,
                        correct=correct,
                        total=total,
                        code=code,
                        task_summary=tsummary,
                    )
                )
                unsolved.remove(tn)
                _write_training_pair(
                    self.task_dir, tn, task, code, strategy_name, model
                )
                print(f"-> SOLVED {correct}/{total}", flush=True)
            else:
                attempts_this_cycle += 1

                # Structured reflection — classify on ANY failure with code
                reflection = {}
                if code:
                    reflection = self._structured_reflection(
                        task, tn, code, correct, total
                    )

                error_type = reflection.get("error_type", "")
                diagnosis = reflection.get("diagnosis", "")
                similar_to = reflection.get("similar_to", "")

                # Feed inferred_class back into fingerprint
                if similar_to and tn in self.fingerprints:
                    self.fingerprints[tn].inferred_class = similar_to

                error_desc = f"partial_{correct}/{total}"
                if correct == 0:
                    error_desc = "zero_correct"

                self.memory.record_attempt(
                    Attempt(
                        task_num=tn,
                        timestamp=datetime.now().isoformat(),
                        strategy=strategy_name,
                        model=model,
                        verified=False,
                        correct=correct,
                        total=total,
                        error=error_desc,
                        error_type=error_type,
                        code=code,
                        insight=diagnosis,
                        similar_to=similar_to,
                        task_summary=tsummary,
                    )
                )

                if diagnosis:
                    print(
                        f"-> {correct}/{total} [{error_type}] {diagnosis[:60]}",
                        flush=True,
                    )
                else:
                    print(f"-> {correct}/{total}", flush=True)

            # ── 6. ADAPT: periodic meta-pattern detection ──
            if (attempt_num + 1) % 10 == 0:
                self.memory.detect_meta_patterns()
                print(
                    f"\n--- Progress: {solved_this_cycle} solved "
                    f"in {attempts_this_cycle} attempts "
                    f"({solved_this_cycle/max(attempts_this_cycle,1):.0%}) ---"
                )
                print(self.memory.get_strategy_report())
                print()

        # Final report
        self.memory.detect_meta_patterns()
        print(f"\n{'='*60}")
        print(
            f"Cycle complete: {solved_this_cycle} new solves "
            f"in {attempts_this_cycle} attempts"
        )
        print(
            f"Cumulative: {self.memory.total_solves} total solves "
            f"/ {self.memory.total_attempts} total attempts"
        )
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

        scientist.run_cycle(max_attempts=args.attempts, models=models)

        if cycle < args.cycles - 1:
            print("\nConsolidating before next cycle...")
            time.sleep(5)

    scientist.memory.save()
    print(f"\nMemory saved to {args.memory}")
    print(
        f"Total: {scientist.memory.total_solves} tasks solved "
        f"across {scientist.memory.total_attempts} attempts"
    )


if __name__ == "__main__":
    main()
