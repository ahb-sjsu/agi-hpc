"""Erebus dreaming schedule — 2AM-4AM PST nightly.

During the dreaming window:
1. Switch NRP to heavy mode (4 Jobs max)
2. Submit 1 GPU Job for QLoRA fine-tuning on day's compiler successes
3. Call managed API (Qwen 397B) for deep analysis of failures
4. Synthesize new compiler modules from patterns
5. Update curriculum with new patterns
6. At 4AM: switch back to swarm mode, clean up Jobs

Outside the window: Erebus solves tasks normally.

Usage:
  python dreaming_schedule.py  (runs forever, triggers at 2AM PST)
  python dreaming_schedule.py --now  (run dreaming cycle immediately)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dreaming")

PST = timezone(timedelta(hours=-7))  # PDT in April
DREAM_START_HOUR = 2   # 2 AM PST
DREAM_END_HOUR = 4     # 4 AM PST
TASK_DIR = Path("/archive/neurogolf")
MEMORY_PATH = TASK_DIR / "arc_scientist_memory.json"
CURRICULUM_PATH = TASK_DIR / "src/compiler/CURRICULUM.md"


def is_dream_time() -> bool:
    """Check if current time is in the 2AM-4AM PST window."""
    now = datetime.now(PST)
    return DREAM_START_HOUR <= now.hour < DREAM_END_HOUR


def get_days_successes() -> list[dict]:
    """Get today's verified compiler/transform successes from Erebus memory."""
    try:
        mem = json.loads(MEMORY_PATH.read_text())
    except Exception:
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    successes = []
    for tn_str, tk in mem.get("tasks", {}).items():
        for attempt in tk.get("attempts", []):
            if (attempt.get("verified") and
                attempt.get("code") and
                attempt.get("timestamp", "").startswith(today)):
                successes.append({
                    "task": int(tn_str),
                    "code": attempt["code"],
                    "strategy": attempt.get("strategy", ""),
                    "model": attempt.get("model", ""),
                })
    return successes


def get_days_failures() -> list[dict]:
    """Get today's classified failures for deep analysis."""
    try:
        mem = json.loads(MEMORY_PATH.read_text())
    except Exception:
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    failures = []
    for tn_str, tk in mem.get("tasks", {}).items():
        for attempt in tk.get("attempts", []):
            if (not attempt.get("verified") and
                attempt.get("error_type") and
                attempt.get("timestamp", "").startswith(today)):
                failures.append({
                    "task": int(tn_str),
                    "error_type": attempt.get("error_type", ""),
                    "insight": attempt.get("insight", ""),
                    "similar_to": attempt.get("similar_to", ""),
                    "correct": attempt.get("correct", 0),
                    "total": attempt.get("total", 0),
                })
    return failures


def dream_analyze_failures(failures: list[dict]) -> str:
    """Use Qwen 397B to deeply analyze the day's failures."""
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token or not failures:
        return ""

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1",
                     timeout=60)

    # Group failures by error type
    by_type = {}
    for f in failures:
        et = f["error_type"]
        by_type.setdefault(et, []).append(f)

    summary = f"Today's {len(failures)} failures:\n"
    for et, group in by_type.items():
        tasks = [f"task{f['task']:03d}" for f in group[:5]]
        insights = [f["insight"] for f in group if f["insight"]][:3]
        summary += f"\n{et} ({len(group)}x): {', '.join(tasks)}\n"
        for ins in insights:
            summary += f"  - {ins}\n"

    prompt = (
        "You are analyzing an AI scientist's (Erebus) daily failure log.\n\n"
        f"{summary}\n\n"
        "1. What are the common root causes across these failures?\n"
        "2. What ONNX compiler patterns would fix the most failures?\n"
        "3. Prioritize: which 2-3 new compiler modules should be written next?\n"
        "4. For each suggested module: describe the ONNX ops needed and the detection logic.\n"
        "Be specific and technical. Reference opset 10 constraints."
    )

    try:
        r = client.chat.completions.create(
            model="qwen3", max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        log.warning(f"Analysis failed: {e}")
        return ""


def dream_synthesize_compiler(successes: list[dict], analysis: str) -> str:
    """Use Qwen 397B to synthesize a new compiler module from successes."""
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token or not successes:
        return ""

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1",
                     timeout=120)

    # Read existing curriculum
    curriculum = ""
    try:
        curriculum = CURRICULUM_PATH.read_text()[:3000]
    except Exception:
        pass

    # Pick the most common transform pattern from successes
    codes = "\n\n".join(
        f"# task{s['task']:03d}\n{s['code'][:500]}"
        for s in successes[:5]
    )

    prompt = (
        "You are building an ONNX compiler for ARC-AGI tasks.\n\n"
        "Here are existing compiler modules (the curriculum):\n"
        f"```\n{curriculum[:2000]}\n```\n\n"
        "Here are Python transforms that Erebus verified today:\n"
        f"```python\n{codes}\n```\n\n"
        f"Analysis of today's failures:\n{analysis[:1000]}\n\n"
        "Write a NEW compiler module that:\n"
        "1. Follows the same pattern: nodes/inits/vinfo lists → make_model()\n"
        "2. Uses only opset 10 ops (Conv, Gather, Reshape, Slice, etc.)\n"
        "3. Handles a class of tasks, not just one specific task\n"
        "4. Includes a detect_X() function that checks if a task matches\n\n"
        "Write the complete module. ```python ... ```"
    )

    try:
        r = client.chat.completions.create(
            model="qwen3", max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        log.warning(f"Synthesis failed: {e}")
        return ""


def dream_update_wiki(analysis: str, new_module: str):
    """Write dreaming results to wiki for RAG retrieval."""
    wiki_dir = Path("/home/claude/agi-hpc/wiki")
    wiki_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    entry = {
        "title": f"Erebus Dreaming Log — {today}",
        "analysis": analysis[:2000],
        "new_module": new_module[:3000] if new_module else "",
        "timestamp": datetime.now().isoformat(),
    }

    log_file = wiki_dir / f"dreaming_{today}.json"
    log_file.write_text(json.dumps(entry, indent=2))
    log.info(f"Dream log saved to {log_file}")


KIRK_SERVICES = ("atlas-id.service", "atlas-watchdog.service")


def _svc(action: str, services=KIRK_SERVICES) -> None:
    """sudo systemctl <action> <services> — claude has NOPASSWD ALL on Atlas."""
    for svc in services:
        try:
            subprocess.run(["sudo", "-n", "systemctl", action, svc],
                           check=False, capture_output=True, timeout=20)
        except Exception as e:
            log.warning(f"systemctl {action} {svc} failed: {e}")


def run_qlora_training(min_pairs: int = 10) -> None:
    """Launch dream_qlora_train.py as a subprocess on GPU 1.

    GPU 1 is Kirk's (atlas-id.service). This stops+disables Kirk and the
    watchdog before training, then re-enables+starts them when done, so
    the adapter training has exclusive use of the 32GB VRAM.
    """
    train_dir = TASK_DIR / "training_data"
    if not train_dir.exists():
        log.info("No training_data/ yet — skipping QLoRA")
        return
    pair_count = 0
    for fp in train_dir.glob("solves_*.jsonl"):
        pair_count += sum(1 for _ in fp.open())
    if pair_count < min_pairs:
        log.info(f"QLoRA skipped: {pair_count} pairs (< {min_pairs})")
        return

    script = Path(__file__).parent / "dream_qlora_train.py"
    log.info(f"Launching QLoRA training on {pair_count} pairs")

    # Free GPU 1 — stop Kirk + watchdog, then disable so they can't respawn
    log.info("Stopping Kirk + watchdog to free GPU 1")
    _svc("disable")
    _svc("stop")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"  # GV100 GPU 1 (Kirk's)
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    try:
        # Atlas GV100 is SM 7.0 (Volta) → bnb 4-bit not supported, use bf16.
        # Qwen2.5-14B in bf16 is ~28GB, fits on 32GB GV100 with LoRA adapters.
        result = subprocess.run(
            [sys.executable, "-u", str(script),
             "--base", "Qwen/Qwen2.5-14B-Instruct",
             "--quant", "none",
             "--rank", "16", "--epochs", "3", "--min-pairs", str(min_pairs)],
            env=env, timeout=5400, capture_output=True, text=True)
        log.info(f"QLoRA exit={result.returncode}")
        if result.stdout:
            log.info(f"QLoRA stdout tail:\n{result.stdout[-2000:]}")
        if result.returncode != 0 and result.stderr:
            log.warning(f"QLoRA stderr tail:\n{result.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        log.warning("QLoRA training exceeded 90 min — aborted")
    except Exception as e:
        log.warning(f"QLoRA launch failed: {e}")
    finally:
        # Always restore Kirk — even on exception — so the cortex isn't
        # stranded offline after a bad training run.
        log.info("Restoring Kirk + watchdog")
        _svc("enable")
        _svc("start")


def run_dream_cycle():
    """Execute one dreaming cycle."""
    log.info("=== EREBUS DREAMING CYCLE START ===")

    # 1. Gather day's data
    successes = get_days_successes()
    failures = get_days_failures()
    log.info(f"Day's data: {len(successes)} successes, {len(failures)} failures")

    if not successes and not failures:
        log.info("Nothing to dream about. Sleeping.")
        return

    # 2. Deep analysis of failures
    log.info("Analyzing failures with Qwen 397B...")
    analysis = dream_analyze_failures(failures)
    if analysis:
        log.info(f"Analysis: {analysis[:200]}...")

    # 3. Synthesize new compiler module
    log.info("Synthesizing compiler module...")
    new_module = dream_synthesize_compiler(successes, analysis)
    if new_module:
        # Extract code and save
        import ast
        code = None
        if "```" in new_module:
            for part in new_module.split("```"):
                if part.startswith("python"):
                    part = part[6:]
                if ("def compile_" in part or "def detect_" in part
                        or "def make_model" in part):
                    code = part.strip()
                    break

        if code:
            # Validate syntax before saving to avoid polluting compiler dir
            try:
                ast.parse(code)
            except SyntaxError as e:
                log.warning(f"Synthesized module has syntax error at line {e.lineno}: {e.msg}. Discarding.")
                code = None

        if code:
            today = datetime.now().strftime("%Y%m%d")
            module_path = TASK_DIR / f"src/compiler/dream_{today}.py"
            module_path.write_text(code)
            log.info(f"New compiler module saved to {module_path}")

            # Test it
            try:
                ns = {}
                exec(code, ns)
                funcs = [k for k in ns if k.startswith(("compile_", "detect_", "make_"))]
                log.info(f"Module defines: {funcs}")
            except Exception as e:
                log.warning(f"Module import-time error: {e}")

    # 4. Update wiki
    dream_update_wiki(analysis, new_module)

    # 5. QLoRA weight updates — the real learning.
    # Runs only if enough training pairs have accumulated.
    run_qlora_training(min_pairs=10)

    log.info("=== EREBUS DREAMING CYCLE COMPLETE ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="Run dream cycle immediately")
    args = ap.parse_args()

    if args.now:
        run_dream_cycle()
        return

    log.info("Erebus dreaming scheduler started")
    log.info(f"Dream window: {DREAM_START_HOUR}:00 - {DREAM_END_HOUR}:00 PST")

    was_dreaming = False
    while True:
        if is_dream_time():
            if not was_dreaming:
                log.info("Entering dream window")
                was_dreaming = True
                run_dream_cycle()
        else:
            if was_dreaming:
                log.info("Exiting dream window")
                was_dreaming = False

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
