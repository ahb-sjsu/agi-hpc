"""Code validator — sandboxed execution against an ARC task.

The Primer's "verify before publish" gate. Given a candidate ``transform``
function (as Python source) and a task dict with train/test examples,
runs the function in a subprocess against every training example and
returns (all_pass, per_example_scores, diagnostic).

A Primer response only gets written to the wiki if ``all_pass`` is True.
A wrong mentor note is worse than no note (see
``feedback_sensei_verify_solutions`` in personal memory).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Outcome of running candidate code against all train examples."""

    all_pass: bool
    """True iff every training example matched exactly."""

    per_example: list[dict[str, Any]]
    """One entry per train example: {idx, correct: bool, reason?: str}."""

    diagnostic: str
    """Error trace / summary suitable for feeding back to the Primer
    as an "improve this" retry prompt. Empty string if all_pass."""


_SANDBOX_WRAPPER = r"""
import json, sys, traceback
import numpy as np  # noqa — available to candidate code

# Per-example timeout only works on Unix (signal.SIGALRM). On Windows
# we fall back to no per-example timeout; the outer subprocess wall
# clock still bounds total runtime.
try:
    import signal
    HAS_SIGALRM = hasattr(signal, "SIGALRM")
except Exception:
    signal = None
    HAS_SIGALRM = False


def _run():
    candidate_src = __CAND__
    task = __TASK__
    results = []
    ns = {}
    try:
        exec(candidate_src, ns)
    except Exception as e:
        return {"compile_error": f"{type(e).__name__}: {e}",
                "per_example": [], "all_pass": False}
    fn = ns.get("transform")
    if not callable(fn):
        return {"compile_error": "no transform() function in candidate",
                "per_example": [], "all_pass": False}
    all_pass = True
    for i, ex in enumerate(task.get("train", [])):
        got = None
        err = None
        try:
            if HAS_SIGALRM:
                signal.signal(signal.SIGALRM,
                              lambda *_: (_ for _ in ()).throw(TimeoutError("per-example 10s budget")))
                signal.alarm(10)
            got = fn(ex["input"])
            if HAS_SIGALRM:
                signal.alarm(0)
        except TimeoutError as e:
            err = f"timeout: {e}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        finally:
            if HAS_SIGALRM:
                try:
                    signal.alarm(0)
                except Exception:
                    pass
        if err:
            results.append({"idx": i, "correct": False, "reason": err})
            all_pass = False
            continue
        # Normalise outputs to plain nested lists of ints
        try:
            if hasattr(got, "tolist"):
                got = got.tolist()
            got = [[int(c) for c in row] for row in got]
        except Exception as e:
            results.append({"idx": i, "correct": False,
                            "reason": f"output not a 2D int grid: {e}"})
            all_pass = False
            continue
        exp = ex["output"]
        if got == exp:
            results.append({"idx": i, "correct": True})
        else:
            results.append({
                "idx": i,
                "correct": False,
                "reason": f"mismatch: got shape {len(got)}x{len(got[0]) if got else 0}, "
                          f"expected {len(exp)}x{len(exp[0]) if exp else 0}",
            })
            all_pass = False
    return {"per_example": results, "all_pass": all_pass, "compile_error": ""}

try:
    out = _run()
except Exception:
    out = {"per_example": [], "all_pass": False,
           "compile_error": traceback.format_exc()}

print("__PRIMER_VALIDATOR_OUT__")
print(json.dumps(out))
"""


def validate(
    code: str, task: dict[str, Any], *, wall_clock_s: int = 60
) -> ValidationResult:
    """Run ``code`` (Python source defining ``transform(grid)``) against
    ``task['train']`` in a subprocess. Returns ValidationResult.

    The subprocess has at most ``wall_clock_s`` total; each individual
    example has a 10 s SIGALRM budget inside the sandbox. The candidate
    runs with numpy available but no additional imports provided; it
    may ``import`` freely but can't escape the subprocess."""
    wrapper = _SANDBOX_WRAPPER
    wrapper = wrapper.replace("__CAND__", repr(code))
    wrapper = wrapper.replace("__TASK__", json.dumps(task))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(wrapper)
        path = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=wall_clock_s,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            all_pass=False,
            per_example=[],
            diagnostic=f"validator subprocess exceeded {wall_clock_s}s wall clock",
        )
    finally:
        try:
            path.unlink()
        except Exception:
            pass

    if proc.returncode != 0 and "__PRIMER_VALIDATOR_OUT__" not in proc.stdout:
        return ValidationResult(
            all_pass=False,
            per_example=[],
            diagnostic=f"subprocess exit {proc.returncode}: {proc.stderr[:1000]}",
        )
    # Extract the marker-separated JSON payload
    marker = "__PRIMER_VALIDATOR_OUT__"
    if marker not in proc.stdout:
        return ValidationResult(
            all_pass=False,
            per_example=[],
            diagnostic=f"no validator output marker found; stdout={proc.stdout[:500]}",
        )
    payload = proc.stdout.split(marker, 1)[1].strip()
    try:
        out = json.loads(payload)
    except json.JSONDecodeError as e:
        return ValidationResult(
            all_pass=False,
            per_example=[],
            diagnostic=f"validator JSON parse error: {e}; raw={payload[:500]}",
        )
    if out.get("compile_error"):
        return ValidationResult(
            all_pass=False,
            per_example=out.get("per_example", []),
            diagnostic=f"candidate compile error: {out['compile_error']}",
        )
    fails = [
        f"ex{r['idx']}: {r.get('reason', '?')}"
        for r in out["per_example"]
        if not r["correct"]
    ]
    return ValidationResult(
        all_pass=out["all_pass"],
        per_example=out["per_example"],
        diagnostic="" if out["all_pass"] else "; ".join(fails),
    )


def extract_code(response_text: str) -> str:
    """Pull ``def transform(...)`` code out of a Primer response.

    Accepts three common shapes:
    - Plain JSON with a ``"code"`` field.
    - Markdown with a ```python ...``` fenced block.
    - Raw Python (treated as-is if ``def transform`` is present).
    """
    # JSON-first
    text = response_text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("code"), str):
            return data["code"]
    except json.JSONDecodeError:
        pass
    # Fenced python block
    if "```" in text:
        parts = text.split("```")
        for p in parts[1:]:
            p = p.strip()
            if p.lower().startswith("python"):
                p = p[6:].lstrip()
            if "def transform" in p:
                return p.strip()
    # Raw
    if "def transform" in text:
        idx = text.index("def transform")
        return text[idx:].strip()
    return ""
