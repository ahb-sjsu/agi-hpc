"""Compile Erebus's verified transforms to ONNX and merge all solutions.

End-to-end competition pipeline:
  1. Extract verified Python transforms from Erebus's memory
  2. Try conv training on ALL unsolved tasks (GPU)
  3. Merge all ONNX sources, pick cheapest per task
  4. Build submission.zip when threshold is met

Usage:
  python compile_and_merge.py --task-dir /archive/neurogolf \
    --memory /archive/neurogolf/arc_scientist_memory.json \
    --device cuda --output-dir solutions_final
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def extract_transforms(memory_path: str, output_dir: str):
    """Extract verified Python transforms from Erebus's episodic memory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(memory_path) as f:
        mem = json.load(f)

    saved = 0
    for tn_str, tk in mem.get("tasks", {}).items():
        if not tk.get("solved"):
            continue
        for attempt in reversed(tk.get("attempts", [])):
            if attempt.get("verified") and attempt.get("code"):
                tn = int(tn_str)
                (out / f"task{tn:03d}.py").write_text(attempt["code"])
                saved += 1
                break

    print(f"Extracted {saved} verified transforms to {out}")
    return saved


def compile_all_tasks(task_dir: str, output_dir: str, device: str = "cuda",
                       num_seeds: int = 8, max_time: int = 180,
                       skip_existing: bool = True):
    """Run conv training on all tasks, save verified ONNX."""
    sys.path.insert(0, str(Path(task_dir) / "src"))
    from gpu_conv_trainer import solve_task_gpu
    from grammar.primitives import score_model, verify_model
    import onnx

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    solved = 0
    task_dir = Path(task_dir)

    for tf in sorted(task_dir.glob("task*.json")):
        tn = int(tf.stem[4:])

        if skip_existing and (out / f"task{tn:03d}.onnx").exists():
            continue

        with open(tf) as f:
            task = json.load(f)

        try:
            r = solve_task_gpu(task, tn, device, num_seeds=num_seeds,
                               max_time_s=max_time)
            if r.get("status") == "solved":
                model = r["model"]
                c, t = verify_model(model, task)
                if c == t and t > 0:
                    s = score_model(model)
                    onnx.save(model, str(out / f"task{tn:03d}.onnx"))
                    solved += 1
                    print(f"task{tn:03d}: SOLVED cost={s['cost']} "
                          f"arch={r.get('arch', '?')}", flush=True)
                else:
                    print(f"task{tn:03d}: verify_fail {c}/{t}", flush=True)
            else:
                print(f"task{tn:03d}: unsolved", flush=True)
        except Exception as e:
            print(f"task{tn:03d}: error {str(e)[:80]}", flush=True)

    print(f"\nCompiled {solved} new ONNX solutions to {out}")
    return solved


def merge_solutions(task_dir: str, output_dir: str,
                     source_dirs: list[str]):
    """Merge ONNX from multiple sources, pick cheapest per task."""
    sys.path.insert(0, str(Path(task_dir) / "src"))
    from grammar.primitives import score_model, verify_model
    import onnx

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Gather all candidates
    candidates = {}  # task_num -> [(cost, path)]
    for d in source_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.glob("task*.onnx"):
            tn = int(f.stem[4:])
            try:
                model = onnx.load(str(f))
                s = score_model(model)
                if s:
                    candidates.setdefault(tn, []).append((s["cost"], str(f)))
            except Exception:
                pass

    # Verify and pick cheapest
    total_cost = 0
    total_score = 0
    merged = 0

    for tn in sorted(candidates.keys()):
        entries = sorted(candidates[tn])  # cheapest first
        for cost, path in entries:
            try:
                model = onnx.load(path)
                task_file = Path(task_dir) / f"task{tn:03d}.json"
                if task_file.exists():
                    with open(task_file) as f:
                        task = json.load(f)
                    c, t = verify_model(model, task)
                    if c == t and t > 0:
                        shutil.copy2(path, str(out / f"task{tn:03d}.onnx"))
                        s = score_model(model)
                        total_cost += cost
                        total_score += s["score"] if s else 1.0
                        merged += 1
                        break
            except Exception:
                continue

    print(f"\nMerged: {merged} tasks")
    print(f"Total projected score: {total_score:.0f}")
    print(f"Total cost: {total_cost}")
    return merged, total_score


def build_submission(solution_dir: str, output_zip: str):
    """Build submission.zip from ONNX files."""
    import zipfile
    p = Path(solution_dir)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(p.glob("task*.onnx")):
            zf.write(f, f.name)
    n = len(list(p.glob("task*.onnx")))
    print(f"Built {output_zip}: {n} tasks")
    return n


def main():
    ap = argparse.ArgumentParser(description="Erebus competition pipeline")
    ap.add_argument("--task-dir", default="/archive/neurogolf")
    ap.add_argument("--memory", default="/archive/neurogolf/arc_scientist_memory.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="solutions_final")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-time", type=int, default=180)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--submit-threshold", type=int, default=100,
                     help="Only build submission if >= this many tasks")
    args = ap.parse_args()

    task_dir = args.task_dir

    # Step 1: Extract Erebus transforms
    print("=" * 60)
    print("STEP 1: Extract Erebus verified transforms")
    print("=" * 60)
    extract_transforms(args.memory, f"{task_dir}/solutions_erebus_transforms")

    # Step 2: Compile to ONNX
    if not args.skip_compile:
        print("\n" + "=" * 60)
        print("STEP 2: Compile all tasks to ONNX (GPU conv training)")
        print("=" * 60)
        compile_all_tasks(task_dir, f"{task_dir}/solutions_erebus_onnx",
                          args.device, args.seeds, args.max_time)

    # Step 3: Merge all sources
    print("\n" + "=" * 60)
    print("STEP 3: Merge all ONNX solutions (cheapest per task)")
    print("=" * 60)
    sources = [
        f"{task_dir}/solutions_merged_latest",
        f"{task_dir}/solutions_safe",
        f"{task_dir}/solutions_conv_v2",
        f"{task_dir}/solutions_erebus_onnx",
        f"{task_dir}/solutions_parallel",
        f"{task_dir}/solutions_dagastar",
        f"{task_dir}/solutions_gpu_atlas",
    ]
    merged, score = merge_solutions(task_dir, f"{task_dir}/{args.output_dir}",
                                     sources)

    # Step 4: Build submission if threshold met
    if merged >= args.submit_threshold:
        print("\n" + "=" * 60)
        print(f"STEP 4: Building submission ({merged} >= {args.submit_threshold})")
        print("=" * 60)
        build_submission(f"{task_dir}/{args.output_dir}",
                          f"{task_dir}/submission.zip")
        print(f"\nReady to submit: {task_dir}/submission.zip")
        print(f"Tasks: {merged}, Projected score: {score:.0f}")
    else:
        print(f"\nNot enough tasks for submission ({merged} < {args.submit_threshold})")


if __name__ == "__main__":
    main()
