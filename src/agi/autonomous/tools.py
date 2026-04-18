"""Tool harness for Erebus — gives the scientist executable capabilities.

Instead of generating text and hoping it contains code, Erebus can now
explicitly call functions via OpenAI-compatible tool calling:

  - run_transform: execute code on a task, get verification result
  - read_task: load a task's training examples
  - read_memory: recall episodic memory for a specific task
  - search_similar: find structurally similar tasks
  - list_primitives: get the geometric primitive catalog
  - ask_for_help: post a question to the help queue
  - classify_error: run structured reflection on a failure

The LLM decides which tools to call and in what order. We execute
them and feed results back. This is the agentic loop.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Tool definitions (OpenAI function calling format)
# ═══════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_transform",
            "description": (
                "Execute a Python transform function on an ARC task. "
                "Returns verification result: how many examples passed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task number (1-400)",
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python function: "
                            "def transform(grid: list[list[int]]) -> list[list[int]]"
                        ),
                    },
                },
                "required": ["task_num", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_task",
            "description": (
                "Load an ARC task's training examples. Returns input/output "
                "grid pairs with dimensions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task number (1-400)",
                    },
                    "max_examples": {
                        "type": "integer",
                        "description": "Max training examples to return (default 3)",
                        "default": 3,
                    },
                },
                "required": ["task_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": (
                "Read Erebus's episodic memory for a specific task: "
                "prior attempts, error types, insights, best score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task number to recall memory for",
                    },
                },
                "required": ["task_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar",
            "description": (
                "Find tasks structurally similar to a given task based on "
                "fingerprint (shape, colors, transformation type). Returns "
                "the closest matches with their solve status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task to find neighbors for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar tasks to return",
                        "default": 5,
                    },
                },
                "required": ["task_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_primitives",
            "description": (
                "Get the catalog of available geometric primitives: "
                "connected_components, flood_fill, detect_symmetry, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_task",
            "description": (
                "Run automated analysis on a task: detect symmetry, "
                "color mapping, shape changes, object count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task number to analyze",
                    },
                },
                "required": ["task_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_for_help",
            "description": (
                "Post a question to Professor Bond when stuck. Include "
                "what you have tried, what you think the pattern might be, "
                "and what specific guidance would help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_num": {
                        "type": "integer",
                        "description": "Task you need help with",
                    },
                    "question": {
                        "type": "string",
                        "description": "Your specific question",
                    },
                },
                "required": ["task_num", "question"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# Tool executor
# ═══════════════════════════════════════════════════════════════

class ToolExecutor:
    """Executes tool calls from the LLM against live data."""

    def __init__(self, task_dir: str, memory=None, fingerprints=None):
        self.task_dir = Path(task_dir)
        self.memory = memory
        self.fingerprints = fingerprints or {}

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return JSON result string."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})

    def _tool_run_transform(self, task_num: int, code: str) -> dict:
        """Execute transform and verify."""
        task = self._load_task(task_num)
        if not task:
            return {"error": f"Task {task_num} not found"}

        try:
            ns = {"np": np, "numpy": np}
            exec(code.strip(), ns)
            transform_fn = ns.get("transform")
            if not transform_fn:
                return {"error": "No transform function found in code"}

            correct = total = 0
            first_failure = None
            for split in ("train", "test"):
                for i, ex in enumerate(task.get(split, [])):
                    total += 1
                    try:
                        result = transform_fn(ex["input"])
                        if isinstance(result, np.ndarray):
                            result = result.tolist()
                        if result == ex["output"]:
                            correct += 1
                        elif first_failure is None:
                            first_failure = {
                                "split": split, "index": i,
                                "expected_shape": f"{len(ex['output'])}x{len(ex['output'][0])}",
                                "got_shape": f"{len(result)}x{len(result[0])}" if result and result[0] else "empty",
                            }
                    except Exception as e:
                        if first_failure is None:
                            first_failure = {"split": split, "index": i, "error": str(e)[:100]}

            verified = correct == total and total > 0
            result = {
                "verified": verified,
                "correct": correct,
                "total": total,
                "task_num": task_num,
            }
            if first_failure and not verified:
                result["first_failure"] = first_failure
            return result

        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            return {"error": f"Execution error: {str(e)[:150]}"}

    def _tool_read_task(self, task_num: int, max_examples: int = 3) -> dict:
        """Load task examples."""
        task = self._load_task(task_num)
        if not task:
            return {"error": f"Task {task_num} not found"}

        examples = []
        for i, ex in enumerate(task.get("train", [])[:max_examples]):
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            examples.append({
                "index": i,
                "input": ex["input"],
                "input_shape": f"{inp.shape[0]}x{inp.shape[1]}",
                "output": ex["output"],
                "output_shape": f"{out.shape[0]}x{out.shape[1]}",
            })

        fp = self.fingerprints.get(task_num)
        summary = {}
        if fp:
            summary = {
                "shape_change": fp.shape_change,
                "same_colors": fp.same_colors,
                "content_overlap": f"{fp.content_overlap:.0%}",
                "has_symmetry": fp.has_symmetry,
                "inferred_class": fp.inferred_class or "unknown",
            }

        return {
            "task_num": task_num,
            "n_train": len(task.get("train", [])),
            "n_test": len(task.get("test", [])),
            "examples": examples,
            "fingerprint": summary,
        }

    def _tool_read_memory(self, task_num: int) -> dict:
        """Read episodic memory for a task."""
        if not self.memory:
            return {"error": "No memory available"}

        tk = self.memory.tasks.get(task_num)
        if not tk:
            return {"task_num": task_num, "status": "no_memory", "attempts": 0}

        # Summarize attempts (don't dump full code)
        attempt_summaries = []
        for a in tk.attempts[-5:]:  # last 5
            attempt_summaries.append({
                "strategy": a.get("strategy", ""),
                "model": a.get("model", ""),
                "verified": a.get("verified", False),
                "correct": a.get("correct", 0),
                "total": a.get("total", 0),
                "error_type": a.get("error_type", ""),
                "insight": a.get("insight", ""),
                "similar_to": a.get("similar_to", ""),
                "task_summary": a.get("task_summary", ""),
            })

        return {
            "task_num": task_num,
            "solved": tk.solved,
            "total_attempts": len(tk.attempts),
            "best_correct": tk.best_correct,
            "best_total": tk.best_total,
            "error_types": tk.error_types,
            "strategies_tried": tk.strategies_tried,
            "recent_attempts": attempt_summaries,
        }

    def _tool_search_similar(self, task_num: int, top_k: int = 5) -> dict:
        """Find similar tasks by fingerprint."""
        from agi.autonomous.arc_scientist import task_distance

        if task_num not in self.fingerprints:
            return {"error": f"No fingerprint for task {task_num}"}

        target = self.fingerprints[task_num]
        scored = []
        for tn, fp in self.fingerprints.items():
            if tn == task_num:
                continue
            d = task_distance(target, fp)
            solved = (self.memory and tn in self.memory.tasks
                      and self.memory.tasks[tn].solved) if self.memory else False
            scored.append({
                "task_num": tn,
                "distance": round(d, 2),
                "solved": solved,
                "shape_change": fp.shape_change,
                "same_colors": fp.same_colors,
                "inferred_class": fp.inferred_class or "",
                "summary": fp.task_summary,
            })

        scored.sort(key=lambda x: x["distance"])
        return {
            "query_task": task_num,
            "query_fingerprint": target.task_summary,
            "similar": scored[:top_k],
        }

    def _tool_list_primitives(self) -> dict:
        """Return the primitive catalog."""
        try:
            from agi.autonomous.primitives import PRIMITIVE_CATALOG
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "primitives", Path(__file__).parent / "primitives.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            PRIMITIVE_CATALOG = mod.PRIMITIVE_CATALOG
        return {"catalog": PRIMITIVE_CATALOG}

    def _tool_analyze_task(self, task_num: int) -> dict:
        """Run automated analysis on a task."""
        task = self._load_task(task_num)
        if not task:
            return {"error": f"Task {task_num} not found"}

        try:
            from agi.autonomous.primitives import (
                detect_symmetry, color_histogram, color_map_between,
                connected_components, find_repeating_pattern, crop_to_content,
            )
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "primitives", Path(__file__).parent / "primitives.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            detect_symmetry = mod.detect_symmetry
            color_histogram = mod.color_histogram
            color_map_between = mod.color_map_between
            connected_components = mod.connected_components
            find_repeating_pattern = mod.find_repeating_pattern
            crop_to_content = mod.crop_to_content

        results = {"task_num": task_num, "examples": []}

        for i, ex in enumerate(task.get("train", [])[:3]):
            inp = ex["input"]
            out = ex["output"]
            analysis = {
                "index": i,
                "input_symmetry": detect_symmetry(inp),
                "output_symmetry": detect_symmetry(out),
                "input_colors": color_histogram(inp),
                "output_colors": color_histogram(out),
                "color_mapping": color_map_between(inp, out),
                "input_objects": len(connected_components(inp)),
                "output_objects": len(connected_components(out)),
                "input_pattern": find_repeating_pattern(inp),
                "output_pattern": find_repeating_pattern(out),
            }
            results["examples"].append(analysis)

        return results

    def _tool_ask_for_help(self, task_num: int, question: str) -> dict:
        """Post to help queue."""
        help_file = self.task_dir / "erebus_help_queue.json"
        queue = []
        try:
            if help_file.exists():
                queue = json.loads(help_file.read_text())
        except Exception:
            pass

        from datetime import datetime
        entry = {
            "task": task_num,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "source": "tool_call",
        }
        queue.append(entry)
        help_file.write_text(json.dumps(queue[-20:], indent=2))

        return {"status": "posted", "queue_length": len(queue)}

    def _load_task(self, task_num: int) -> dict | None:
        tf = self.task_dir / f"task{task_num:03d}.json"
        if not tf.exists():
            return None
        with open(tf) as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════
# Agentic loop — multi-turn tool calling
# ═══════════════════════════════════════════════════════════════

def run_agentic_turn(client, model: str, messages: list[dict],
                     executor: ToolExecutor, max_tool_rounds: int = 5,
                     extra_body: dict = None) -> str:
    """Run a multi-turn agentic conversation with tool calling.

    The LLM can call tools, get results, reason, call more tools,
    and eventually produce a final text response.
    """
    for _ in range(max_tool_rounds):
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "max_tokens": 3000,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            # No more tool calls — return final response
            return msg.content or ""

        # Execute each tool call
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = executor.execute(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Exhausted tool rounds — ask for final answer
    messages.append({
        "role": "user",
        "content": "Please provide your final answer based on the tool results.",
    })
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=2000,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    return response.choices[0].message.content or ""
