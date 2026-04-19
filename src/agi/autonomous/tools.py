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
import os
import subprocess
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
    {
        "type": "function",
        "function": {
            "name": "query_available_gpus",
            "description": (
                "Query what GPU types are available on NRP right now. "
                "Returns GPU models, counts, VRAM, and which nodes have them. "
                "Use this to decide between heavy mode (few A100s) or "
                "swarm mode (many old GPUs)."
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
            "name": "list_compiler_modules",
            "description": (
                "List all existing ONNX compiler modules (flip, crop_bbox, "
                "color_remap, etc.) with their docstrings and exported "
                "detect_X/compile_X functions. Use this to see what the "
                "compiler already handles before writing a new module."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_compiler_module",
            "description": (
                "Read the full source of one existing compiler module. Use "
                "this as a few-shot reference when writing a new module — "
                "copy the node/init/vinfo list pattern and opset 10 ops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Module stem, e.g. 'flip' or 'crop_bbox'",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_failure_clusters",
            "description": (
                "Group today's unsolved tasks by (error_type, similar_to) "
                "pattern. Returns the biggest clusters first — these are "
                "where a new compiler module would have the most leverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "Filter to one day (YYYY-MM-DD). Default: all days.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_compiler_module",
            "description": (
                "Write a candidate ONNX compiler module, runtime-test it "
                "against a list of tasks, and promote it to the compiler "
                "directory only if it solves at least 50% of them. Pipeline: "
                "syntax-check → import-check → ONNX runtime test → promote."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source for the module. Must define compile_X() or make_model() that returns an onnx.ModelProto.",
                    },
                    "test_task_nums": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Task numbers to test the module against. Should be tasks you think the module handles.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Short tag for the module file, e.g. 'enclosure_fill' → saved as dream_enclosure_fill.py",
                    },
                },
                "required": ["code", "test_task_nums", "tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_nrp_mode",
            "description": (
                "Set the NRP compute mode. "
                "'heavy': up to 4 pods at full GPU (for A100/H100). "
                "'swarm': 5+ pods, all under 40% GPU (for many old GPUs). "
                "'auto': let the watchdog decide based on available hardware."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "heavy", "swarm"],
                        "description": "Compute mode to use",
                    },
                },
                "required": ["mode"],
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
                                "split": split,
                                "index": i,
                                "expected_shape": f"{len(ex['output'])}x{len(ex['output'][0])}",
                                "got_shape": (
                                    f"{len(result)}x{len(result[0])}"
                                    if result and result[0]
                                    else "empty"
                                ),
                            }
                    except Exception as e:
                        if first_failure is None:
                            first_failure = {
                                "split": split,
                                "index": i,
                                "error": str(e)[:100],
                            }

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
            examples.append(
                {
                    "index": i,
                    "input": ex["input"],
                    "input_shape": f"{inp.shape[0]}x{inp.shape[1]}",
                    "output": ex["output"],
                    "output_shape": f"{out.shape[0]}x{out.shape[1]}",
                }
            )

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
            attempt_summaries.append(
                {
                    "strategy": a.get("strategy", ""),
                    "model": a.get("model", ""),
                    "verified": a.get("verified", False),
                    "correct": a.get("correct", 0),
                    "total": a.get("total", 0),
                    "error_type": a.get("error_type", ""),
                    "insight": a.get("insight", ""),
                    "similar_to": a.get("similar_to", ""),
                    "task_summary": a.get("task_summary", ""),
                }
            )

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
            solved = (
                (
                    self.memory
                    and tn in self.memory.tasks
                    and self.memory.tasks[tn].solved
                )
                if self.memory
                else False
            )
            scored.append(
                {
                    "task_num": tn,
                    "distance": round(d, 2),
                    "solved": solved,
                    "shape_change": fp.shape_change,
                    "same_colors": fp.same_colors,
                    "inferred_class": fp.inferred_class or "",
                    "summary": fp.task_summary,
                }
            )

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
                "primitives", Path(__file__).parent / "primitives.py"
            )
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
                detect_symmetry,
                color_histogram,
                color_map_between,
                connected_components,
                find_repeating_pattern,
                crop_to_content,
            )
        except ImportError:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "primitives", Path(__file__).parent / "primitives.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            detect_symmetry = mod.detect_symmetry
            color_histogram = mod.color_histogram
            color_map_between = mod.color_map_between
            connected_components = mod.connected_components
            find_repeating_pattern = mod.find_repeating_pattern

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
        from agi.common.atomic_write import atomic_write_text

        atomic_write_text(help_file, json.dumps(queue[-20:], indent=2))

        return {"status": "posted", "queue_length": len(queue)}

    def _tool_query_available_gpus(self) -> dict:
        """Query NRP cluster for available GPU types — live kubectl data."""
        kubeconfig = os.path.expanduser("~/.kube/config") if os.name != "nt" else ""
        try:
            cmd = ["kubectl"]
            if kubeconfig:
                cmd += ["--kubeconfig", kubeconfig]
            cmd += [
                "get",
                "nodes",
                "-l",
                "nvidia.com/gpu.product",
                "-o",
                "custom-columns=NAME:.metadata.name,"
                "GPU:.metadata.labels.nvidia\\.com/gpu\\.product",
                "--no-headers",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return {"error": f"kubectl failed: {result.stderr[:200]}"}

            # Parse node GPU types
            gpu_counts = {}  # model -> count of nodes
            gpu_nodes = {}  # model -> list of node names
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    node = parts[0]
                    gpu = parts[1]
                    gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                    gpu_nodes.setdefault(gpu, []).append(node)

            # Known VRAM sizes
            vram = {
                "NVIDIA-A100-80GB-PCIe": 80,
                "NVIDIA-A100-SXM4-80GB": 80,
                "NVIDIA-A100-PCIE-40GB": 40,
                "NVIDIA-A100-SXM4-40GB": 40,
                "NVIDIA-H100-80GB-HBM3": 80,
                "NVIDIA-H100-SXM5-80GB": 80,
                "NVIDIA-H200-SXM-141GB": 141,
                "NVIDIA-L40": 48,
                "NVIDIA-L40S": 48,
                "NVIDIA-L4": 24,
                "NVIDIA-A10": 24,
                "NVIDIA-GeForce-RTX-3090": 24,
                "NVIDIA-GeForce-RTX-2080-Ti": 11,
                "NVIDIA-GeForce-GTX-1080-Ti": 11,
                "Tesla-T4": 16,
                "Tesla-V100-SXM2-32GB": 32,
            }

            # Build summary
            models = []
            total_datacenter = 0
            total_consumer = 0
            for gpu, count in sorted(gpu_counts.items(), key=lambda x: -x[1]):
                v = vram.get(gpu, 0)
                is_datacenter = v >= 24 and "GeForce" not in gpu and "GTX" not in gpu
                if is_datacenter:
                    total_datacenter += count
                else:
                    total_consumer += count
                models.append(
                    {
                        "model": gpu,
                        "nodes": count,
                        "vram_gb": v,
                        "datacenter": is_datacenter,
                        "example_nodes": gpu_nodes.get(gpu, [])[:3],
                    }
                )

            recommendation = ""
            if total_datacenter >= 4:
                recommendation = (
                    f"HEAVY mode recommended: {total_datacenter} datacenter GPU nodes "
                    f"available. Use 4 pods on A100/H100/L40 at full power."
                )
            elif total_consumer >= 10:
                recommendation = (
                    f"SWARM mode recommended: {total_consumer} consumer GPU nodes. "
                    f"Run many pods at <40% each."
                )
            else:
                recommendation = f"AUTO mode: {total_datacenter} datacenter + {total_consumer} consumer nodes."

            return {
                "total_gpu_nodes": sum(gpu_counts.values()),
                "datacenter_nodes": total_datacenter,
                "consumer_nodes": total_consumer,
                "models": models,
                "recommendation": recommendation,
            }
        except Exception as e:
            return {"error": str(e)[:200]}

    def _tool_set_nrp_mode(self, mode: str) -> dict:
        """Set NRP compute mode and return available GPU info."""
        import requests as _req

        try:
            r = _req.post(
                "http://localhost:8085/api/nrp/mode", json={"mode": mode}, timeout=5
            )
            result = r.json()
        except Exception as e:
            result = {"error": str(e)[:100]}

        # Also return what GPUs are currently available
        gpu_summary = {}
        try:
            r2 = _req.get("http://localhost:8085/api/nrp-burst", timeout=5)
            burst = r2.json()
            for pod in burst.get("pods", []):
                if pod.get("phase") != "Running":
                    continue
                gm = pod.get("resources", {}).get("gpu_model") or pod.get(
                    "gpu_live", {}
                ).get("vram_total_mib", "")
                if gm:
                    gpu_summary[str(gm)] = gpu_summary.get(str(gm), 0) + 1
        except Exception:
            pass

        result["available_gpus"] = gpu_summary
        result["recommendation"] = (
            "Use 'heavy' mode for A100/H100/H200 (4 pods, full power). "
            "Use 'swarm' mode for old GPUs like 1080Ti/2080Ti/T4 (many pods, light). "
            "Use 'auto' to let the watchdog decide."
        )
        return result

    def _tool_list_compiler_modules(self) -> dict:
        """List all existing compiler modules."""
        from agi.autonomous.erebus_compiler_tools import list_compiler_modules

        infos = list_compiler_modules()
        return {
            "n_modules": len(infos),
            "modules": [
                {
                    "name": m.name,
                    "docstring": m.docstring,
                    "detect_fns": m.detect_fns,
                    "compile_fns": m.compile_fns,
                    "lines": m.line_count,
                }
                for m in infos
            ],
        }

    def _tool_read_compiler_module(self, name: str) -> dict:
        """Read the full source of an existing compiler module."""
        from agi.autonomous.erebus_compiler_tools import read_compiler_module

        src = read_compiler_module(name)
        if src is None:
            return {"error": f"module '{name}' not found"}
        return {"name": name, "source": src, "lines": src.count("\n")}

    def _tool_list_failure_clusters(self, day: str = "") -> dict:
        """Cluster unsolved tasks by error_type + similar_to pattern."""
        from agi.autonomous.erebus_compiler_tools import cluster_failures

        clusters = cluster_failures(day=day or None)
        return {"n_clusters": len(clusters), "clusters": clusters[:10]}

    def _tool_write_compiler_module(
        self, code: str, test_task_nums: list[int], tag: str
    ) -> dict:
        """Syntax + import + runtime test the candidate; save on success."""
        from agi.autonomous.erebus_compiler_tools import write_compiler_module

        return write_compiler_module(code, test_task_nums, tag)

    def _load_task(self, task_num: int) -> dict | None:
        tf = self.task_dir / f"task{task_num:03d}.json"
        if not tf.exists():
            return None
        with open(tf) as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════
# Agentic loop — multi-turn tool calling
# ═══════════════════════════════════════════════════════════════


def run_agentic_turn(
    client,
    model: str,
    messages: list[dict],
    executor: ToolExecutor,
    max_tool_rounds: int = 5,
    extra_body: dict = None,
) -> str:
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
            return msg.content or ""

        # Build assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        tool_calls_list = []
        for tc in msg.tool_calls:
            tc_dict = {
                "id": tc.id if tc.id else f"call_{len(tool_calls_list)}",
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": (
                        tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments)
                    ),
                },
            }
            tool_calls_list.append(tc_dict)
        assistant_msg["tool_calls"] = tool_calls_list
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = executor.execute(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    # Exhausted tool rounds — ask for final answer
    messages.append(
        {
            "role": "user",
            "content": "Please provide your final answer based on the tool results.",
        }
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    return response.choices[0].message.content or ""
