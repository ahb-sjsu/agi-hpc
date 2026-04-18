"""Erebus-specific task handlers.

These handlers are the *brain* of Erebus when it runs inside a
``nats-bursting`` persistent pool. The pool machinery (NATS connection,
JetStream consumer, ack/nak, result publishing) lives in
``nats_bursting.worker``; this module just registers what the worker
should *do* when it receives a task.

Register like this in an entrypoint::

    from nats_bursting import run_worker
    from agi.autonomous.erebus_handlers import HANDLERS
    run_worker(HANDLERS)

Pod layout: bootstrap script clones ``ahb-sjsu/neurogolf-bundle`` into
``/work/bundle``, copies ``bundle/src`` into ``/work/src``, and extracts
``bundle/data/tasks.tar.gz`` into ``/work/tasks``. The handlers assume
those paths.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(os.environ.get("TASK_DIR", "/work/tasks"))
COMPILER_DIR = Path(os.environ.get("COMPILER_DIR", "/work/src/compiler"))
LLM_BASE_URL = os.environ.get("NRP_LLM_URL", "https://ellm.nrp-nautilus.io/v1")


def _llm():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("NRP_LLM_TOKEN", ""),
                  base_url=LLM_BASE_URL, timeout=180)


def _extract_python(content: str, must_contain: tuple[str, ...]) -> str | None:
    if "```" not in content:
        return None
    for part in content.split("```"):
        s = part.lstrip()
        if s.startswith("python"):
            s = s[6:]
        if any(m in s for m in must_contain):
            return s.strip()
    return None


# ─── solve_task ──────────────────────────────────────────────────────

def handle_solve_task(task: dict) -> dict:
    """Ask an LLM to write a Python `transform` for an ARC task, verify it."""
    import numpy as np

    task_num = task["task_num"]
    model = task.get("model", "qwen3")
    tf = TASK_DIR / f"task{task_num:03d}.json"
    if not tf.exists():
        return {"error": f"task {task_num} not found"}

    task_obj = json.loads(tf.read_text())
    examples_text = "\n\n".join(
        f"Input:\n{ex['input']}\nOutput:\n{ex['output']}"
        for ex in task_obj.get("train", [])[:3]
    )
    prompt = (
        f"Solve ARC task {task_num}. Write a Python function "
        f"`def transform(grid: list[list[int]]) -> list[list[int]]`. "
        f"Return only the function in a ```python``` block.\n\n"
        f"{examples_text}"
    )
    r = _llm().chat.completions.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    code = _extract_python(r.choices[0].message.content or "",
                           ("def transform",))
    if not code:
        return {"status": "no_code", "task_num": task_num, "model": model}

    ns = {"np": np, "numpy": np}
    try:
        exec(code, ns)
        fn = ns["transform"]
        correct = total = 0
        for split in ("train", "test"):
            for ex in task_obj.get(split, []):
                total += 1
                try:
                    out = fn(ex["input"])
                    if isinstance(out, np.ndarray):
                        out = out.tolist()
                    if out == ex["output"]:
                        correct += 1
                except Exception:
                    pass
        return {
            "status": "solved" if correct == total and total > 0 else "partial",
            "task_num": task_num, "correct": correct, "total": total,
            "code": code, "model": model,
        }
    except Exception as e:
        return {"status": "exec_error", "error": str(e)[:200],
                "task_num": task_num, "model": model}


# ─── compile_attempt ─────────────────────────────────────────────────

def handle_compile_attempt(task: dict) -> dict:
    """Author + verify an ONNX compiler module for a failure cluster."""
    # Lazy import — erebus_compiler_tools is in /work/src (bundle), not in
    # the installed package.
    import sys
    sys.path.insert(0, "/work/src")
    from erebus_compiler_tools import (
        get_few_shot_modules, write_compiler_module,
    )

    cluster = task["cluster"]
    few_shot = get_few_shot_modules(compiler_dir=COMPILER_DIR,
                                    max_modules=2, max_chars_each=2500)
    sample_codes = "\n\n".join(
        f"# task{s['task']:03d}\n{s.get('code','')[:500]}"
        for s in cluster.get("sample_codes", [])[:3]
    )
    prompt = (
        f"Write an ONNX compiler module for cluster "
        f"'{cluster.get('pattern','unknown')}' "
        f"({cluster.get('n_unique_tasks',0)} tasks).\n\n"
        f"Imitate these existing modules:\n```python\n{few_shot}\n```\n\n"
        f"Similar verified transforms:\n```python\n{sample_codes}\n```\n\n"
        f"Must define detect_X and compile_X (or make_model). Opset 10 only."
    )
    r = _llm().chat.completions.create(
        model="qwen3", max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    code = _extract_python(r.choices[0].message.content or "",
                           ("def compile_", "def detect_", "def make_model"))
    if not code:
        return {"promoted": False, "reason": "no_code"}

    test_tasks = cluster.get("tasks", [])[:5]
    tag = (cluster.get("pattern", "cluster").replace(" ", "_").lower()[:30]
           + "_" + datetime.now().strftime("%Y%m%d_%H%M"))
    result = write_compiler_module(
        code, test_tasks, tag, min_solved_ratio=0.4,
        compiler_dir=COMPILER_DIR, task_dir=TASK_DIR,
    )
    if result.get("promoted"):
        result["module_source"] = code
    return result


# ─── classify_error ──────────────────────────────────────────────────

def handle_classify_error(task: dict) -> dict:
    """Structured reflection for one failed attempt."""
    examples = task.get("examples", [])[:2]
    code = task.get("code", "")
    prompt = (
        f"An ARC-AGI transform got {task.get('correct',0)}/"
        f"{task.get('total',0)} examples right.\n\n"
        f"Task examples:\n{examples}\n\n"
        f"Failed code:\n```python\n{code}\n```\n\n"
        f"Classify as JSON: "
        f'{{"error_type": "perception|reasoning|execution|specification", '
        f'"diagnosis": "...", "similar_to": "..."}}'
    )
    r = _llm().chat.completions.create(
        model="qwen3", max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = r.choices[0].message.content or ""
    try:
        s = content.find("{"); e = content.rfind("}")
        return {"classification": json.loads(content[s:e+1])}
    except Exception:
        return {"classification": {"raw": content[:500]}}


HANDLERS = {
    "solve_task": handle_solve_task,
    "compile_attempt": handle_compile_attempt,
    "classify_error": handle_classify_error,
}
