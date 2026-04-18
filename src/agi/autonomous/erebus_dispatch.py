"""Atlas-side dispatcher for Erebus tasks.

Thin convenience wrapper over :class:`nats_bursting.TaskDispatcher` —
it adds Erebus-specific task builders (``solve_task`` over a task list,
``compile_attempt`` from today's failure clusters) so callers don't
have to reach into ``erebus_compiler_tools`` themselves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from nats_bursting import TaskDispatcher

NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")
STREAM = "EREBUS_TASKS"
TASK_SUBJECT = "erebus.tasks"
RESULT_PREFIX = "erebus.results."


def build_solve_tasks(task_nums: list[int], model: str) -> list[dict]:
    return [{"type": "solve_task", "task_num": tn, "model": model}
            for tn in task_nums]


def build_compile_tasks(day: str, top: int) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent))
    from erebus_compiler_tools import cluster_failures
    clusters = cluster_failures(day=day)
    classified = [c for c in clusters if c["pattern"] != "unclassified"]
    picked = (classified or clusters)[:top]
    return [{"type": "compile_attempt", "cluster": c} for c in picked]


async def run(tasks: list[dict], subject_suffix: str, wait: bool,
              timeout: int):
    async with TaskDispatcher(NATS_URL, stream=STREAM,
                              result_prefix=RESULT_PREFIX) as td:
        ids = await td.submit_many(f"{TASK_SUBJECT}.{subject_suffix}", tasks)
        print(f"dispatched {len(ids)} tasks: {ids}")
        if wait:
            results = await td.collect(ids, timeout=timeout)
            print(json.dumps(results, indent=2, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_solve = sub.add_parser("solve_task")
    sp_solve.add_argument("--tasks", type=int, nargs="+", required=True)
    sp_solve.add_argument("--model", default="qwen3")
    sp_solve.add_argument("--wait", action="store_true")
    sp_solve.add_argument("--timeout", type=int, default=300)

    sp_compile = sub.add_parser("compile_attempt")
    sp_compile.add_argument("--day",
                            default=datetime.now().strftime("%Y-%m-%d"))
    sp_compile.add_argument("--top", type=int, default=3)
    sp_compile.add_argument("--wait", action="store_true")
    sp_compile.add_argument("--timeout", type=int, default=600)

    args = ap.parse_args()
    if args.cmd == "solve_task":
        tasks = build_solve_tasks(args.tasks, args.model)
        asyncio.run(run(tasks, "solve_task", args.wait, args.timeout))
    else:
        tasks = build_compile_tasks(args.day, args.top)
        asyncio.run(run(tasks, "compile_attempt", args.wait, args.timeout))


if __name__ == "__main__":
    main()
