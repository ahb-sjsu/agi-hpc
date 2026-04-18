"""ONNX-direct scientist — Erebus generates submittable ONNX models.

Skips the Python→ONNX compilation bottleneck entirely.
The LLM writes onnx.helper code that builds the model directly.

Pipeline:
  1. Pick task (similarity clustering from arc_scientist)
  2. Analyze task (detect symmetry, colors, objects)
  3. Ask LLM to write build_onnx() function
  4. Execute build_onnx() → get ModelProto
  5. Verify on ALL examples (Security Radar)
  6. If verified → save ONNX → done, submittable
  7. If failed → structured reflection → retry

This produces ONNX files that can go directly into submission.zip.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent


def grid_to_onehot(grid):
    """Convert grid to one-hot tensor [1,10,30,30]."""
    t = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= 30: break
        for c, color in enumerate(row):
            if c >= 30: break
            if 0 <= color < 10:
                t[0, color, r, c] = 1.0
    return t


def verify_onnx(model, task):
    """Verify ONNX model on ALL examples."""
    import onnxruntime
    import onnx
    buf = io.BytesIO()
    onnx.save(model, buf)
    try:
        sess = onnxruntime.InferenceSession(
            buf.getvalue(), providers=["CPUExecutionProvider"])
    except Exception:
        return 0, 0
    correct = total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            total += 1
            try:
                inp = grid_to_onehot(ex["input"])
                expected = grid_to_onehot(ex["output"])
                out = sess.run(["output"], {"input": inp})[0]
                pred = (out > 0).astype(np.float32)
                if np.array_equal(pred, expected):
                    correct += 1
                else:
                    return correct, total
            except Exception:
                return correct, total
    return correct, total


def format_examples(task, max_examples=4):
    parts = []
    for i, ex in enumerate(task.get("train", [])[:max_examples]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        parts.append(f"Example {i+1}:")
        parts.append(f"  input ({inp.shape[0]}x{inp.shape[1]}): {ex['input']}")
        parts.append(f"  output ({out.shape[0]}x{out.shape[1]}): {ex['output']}")
        parts.append("")
    return "\n".join(parts)


def extract_build_onnx(response):
    """Extract build_onnx() function from LLM response."""
    if not response:
        return None
    if "```" in response:
        parts = response.split("```")
        for p in parts[1:]:
            if p.startswith("python"):
                p = p[6:]
            if "def build_onnx" in p:
                return p.strip()
    if "def build_onnx" in response:
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if "def build_onnx" in line:
                return "\n".join(lines[i:])
    return None


def main():
    from onnx_strategies import (
        ONNX_DIRECT, ONNX_WITH_ANALYSIS, ONNX_FROM_PYTHON, ONNX_DIAGNOSTIC)

    ap = argparse.ArgumentParser(description="Erebus ONNX-direct scientist")
    ap.add_argument("--task-dir", default="/archive/neurogolf")
    ap.add_argument("--output-dir", default="solutions_onnx_direct")
    ap.add_argument("--models", default="kimi,qwen3")
    ap.add_argument("--attempts", type=int, default=200)
    ap.add_argument("--skip-solved", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("NRP_LLM_TOKEN", "")
    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

    task_dir = Path(args.task_dir)
    output_dir = task_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",")]

    # Find all tasks
    all_tasks = sorted(int(f.stem[4:]) for f in task_dir.glob("task*.json"))

    # Skip already-solved ONNX
    solved = set()
    if args.skip_solved:
        for d in ["solutions_final", "solutions_merged_latest", "solutions_safe",
                   "solutions_conv_v2", "solutions_onnx_direct"]:
            p = task_dir / d
            if p.exists():
                solved.update(int(f.stem[4:]) for f in p.glob("task*.onnx"))

    unsolved = [t for t in all_tasks if t not in solved]
    print(f"ONNX-direct scientist starting")
    print(f"  Tasks: {len(all_tasks)} total, {len(solved)} solved, {len(unsolved)} to try")
    print()

    import onnx
    n_solved = 0
    n_attempted = 0

    for attempt in range(args.attempts):
        if not unsolved:
            print("All tasks solved!")
            break

        tn = unsolved[attempt % len(unsolved)]
        with open(task_dir / f"task{tn:03d}.json") as f:
            task = json.load(f)

        examples = format_examples(task)
        model_name = models[attempt % len(models)]

        # Build prompt
        prompt = ONNX_DIRECT.format(examples=examples)

        extra = {}
        if model_name == "kimi":
            extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
        elif "qwen" in model_name:
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

        print(f"[{attempt+1}/{args.attempts}] task{tn:03d} model={model_name}",
              end=" ", flush=True)
        n_attempted += 1

        try:
            r = client.chat.completions.create(
                model=model_name, max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            response = r.choices[0].message.content or ""
        except Exception as e:
            print(f"-> LLM error", flush=True)
            continue

        code = extract_build_onnx(response)
        if not code:
            print(f"-> no build_onnx code", flush=True)
            continue

        # Execute build_onnx
        try:
            from onnx import helper, TensorProto
            ns = {"np": np, "onnx": onnx, "helper": helper,
                  "TensorProto": TensorProto}
            exec(code.strip(), ns)
            build_fn = ns.get("build_onnx")
            if not build_fn:
                print(f"-> no build_onnx function", flush=True)
                continue
            model = build_fn()
        except Exception as e:
            print(f"-> build error: {str(e)[:60]}", flush=True)
            continue

        # Verify
        correct, total = verify_onnx(model, task)
        if correct == total and total > 0:
            # Save
            onnx.save(model, str(output_dir / f"task{tn:03d}.onnx"))
            n_solved += 1
            unsolved.remove(tn)
            try:
                sys.path.insert(0, str(task_dir / "src"))
                from grammar.primitives import score_model
                s = score_model(model)
                cost = s["cost"] if s else "?"
            except Exception:
                cost = "?"
            print(f"-> SOLVED {correct}/{total} cost={cost}", flush=True)
        else:
            print(f"-> {correct}/{total}", flush=True)

        if (attempt + 1) % 20 == 0:
            print(f"\n--- Progress: {n_solved} ONNX solves in {n_attempted} attempts ---\n")

    print(f"\nDone: {n_solved} ONNX models saved to {output_dir}")
    print(f"Attempted: {n_attempted}")


if __name__ == "__main__":
    main()
