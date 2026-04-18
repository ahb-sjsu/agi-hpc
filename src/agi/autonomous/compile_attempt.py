"""NRP pod entrypoint: attempt to write one compiler module for a cluster.

Input:
  - env var CLUSTER_JSON (base64'd or direct JSON) describing the failing
    task cluster (error_type, similar_to, task numbers, sample codes).
  - env var NRP_LLM_TOKEN for the ELLM endpoint.

Output:
  - stdout ends with a JSON line: {"cluster":..., "promoted": bool, ...}
  - if promoted, the module source is echoed to stdout between
    <MODULE_BEGIN>...<MODULE_END> markers so the driver can fetch it
    from pod logs.

Pod resources: cpu=1, memory=2Gi (swarm-mode compliant). No GPU required.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    print(f"[compile_attempt] {msg}", flush=True)


def main():
    cluster_raw = os.environ.get("CLUSTER_JSON", "")
    if not cluster_raw:
        log("no CLUSTER_JSON env var")
        sys.exit(1)
    try:
        cluster = json.loads(cluster_raw)
    except json.JSONDecodeError as e:
        log(f"bad CLUSTER_JSON: {e}")
        sys.exit(1)

    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        log("no NRP_LLM_TOKEN env var")
        sys.exit(1)

    # Path fixups: neurogolf-bundle ships tasks and compiler at /work/src.
    # On this pod layout we expect TASK_DIR to be set too.
    task_dir = Path(os.environ.get("TASK_DIR", "/work/tasks"))
    compiler_dir = Path(os.environ.get("COMPILER_DIR", "/work/src/compiler"))

    # Pod layout: /work/src holds both compile_attempt.py and
    # erebus_compiler_tools.py (flat), so no package prefix.
    sys.path.insert(0, "/work/src")
    from erebus_compiler_tools import (
        get_few_shot_modules, write_compiler_module,
    )

    from openai import OpenAI
    client = OpenAI(api_key=token,
                    base_url="https://ellm.nrp-nautilus.io/v1",
                    timeout=180)

    few_shot = get_few_shot_modules(compiler_dir=compiler_dir,
                                    max_modules=2, max_chars_each=2500)

    sample_codes = "\n\n".join(
        f"# task{s['task']:03d}\n{s.get('code', '')[:500]}"
        for s in cluster.get("sample_codes", [])[:3]
    )

    prompt = (
        "You are an ONNX compiler author. Write a new module that solves "
        f"a cluster of ARC-AGI tasks labeled '{cluster.get('pattern', 'unknown')}' "
        f"({cluster.get('n_unique_tasks', 0)} tasks, error_type="
        f"{cluster.get('error_type', 'unknown')}).\n\n"
        "Imitate the structure of these existing modules:\n"
        f"```python\n{few_shot}\n```\n\n"
        "Here are verified Python transforms for similar tasks:\n"
        f"```python\n{sample_codes}\n```\n\n"
        "Requirements:\n"
        "1. Define detect_X(task_examples: list[dict]) -> bool\n"
        "2. Define compile_X() or make_model() returning onnx.ModelProto\n"
        "3. Use only opset 10 ops\n"
        "4. One ```python``` block, complete and runnable.\n"
    )

    log("synthesizing module via Qwen 397B")
    resp = client.chat.completions.create(
        model="qwen3", max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""

    # Extract python block
    code = None
    if "```" in content:
        for part in content.split("```"):
            stripped = part.lstrip()
            if stripped.startswith("python"):
                stripped = stripped[6:]
            if any(m in stripped for m in ("def compile_", "def detect_",
                                            "def make_model")):
                code = stripped.strip()
                break

    if not code:
        log("no python block extracted")
        print(json.dumps({"cluster": cluster.get("pattern"),
                          "promoted": False, "reason": "no_code"}))
        sys.exit(0)

    test_task_nums = cluster.get("tasks", [])[:5]
    tag = (cluster.get("pattern", "cluster").replace(" ", "_").lower()
           + "_" + datetime.now().strftime("%Y%m%d_%H%M"))

    log(f"testing against tasks {test_task_nums}")
    result = write_compiler_module(
        code, test_task_nums, tag,
        min_solved_ratio=0.4,
        compiler_dir=compiler_dir, task_dir=task_dir,
    )

    if result.get("promoted"):
        # Echo the source so the driver can pull from pod logs
        print("<MODULE_BEGIN>", flush=True)
        print(code, flush=True)
        print("<MODULE_END>", flush=True)

    # Final JSON line for result collection
    print(json.dumps({
        "cluster": cluster.get("pattern"),
        "tag": tag,
        "promoted": result.get("promoted", False),
        "solved_ratio": result.get("solved_ratio"),
        "reason": result.get("reason", ""),
        "test_tasks": test_task_nums,
    }))


if __name__ == "__main__":
    main()
