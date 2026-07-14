"""Tools for Erebus to author and test compiler modules.

The compiler writes ONNX graphs. Each module implements a pattern
(flip, crop_bbox, color_remap, etc.) with the signature:

    def detect_X(task_examples: list[dict]) -> bool
    def compile_X(...) -> onnx.ModelProto   # or make_model(...)

These helpers let Erebus:
 - inspect existing modules as few-shot examples
 - list compiler modules with their detect/compile functions
 - cluster today's failures to decide which pattern to tackle
 - write a candidate module, syntax-check and runtime-test it
 - **stage** a verified candidate for human review — but never make it
   callable on his own.

Safety posture (approved 2026-07-14): Erebus may author and test modules
autonomously (Director tier L2), but *promotion* — moving a module into
the live compiler directory where it is loaded into future solving — is
self-modification and requires a human ack in the dashboard (tier L3).

Three hardening layers wrap every candidate before it can be staged:

 1. **Static safety** (``static_safety_check``): an AST scan rejects
    dangerous constructs (os/subprocess/socket, eval/exec, file writes,
    sandbox-escape dunders). Compiler modules only need numpy/onnx, so an
    import allowlist is safe.
 2. **Sandboxed execution** (``_run_py_sandboxed``): the candidate is
    imported and run in a *subprocess* with CPU/memory/file-size rlimits
    and a socket guard — never ``exec``'d in the live process.
 3. **Provenance + audit**: every stage/promote/reject appends to
    ``compiler_audit.jsonl`` with the motivating failure cluster, the
    authoring model, test results, and the safety verdict.

A ``.compiler_disabled`` sentinel is an immediate kill-switch: while it
exists, nothing stages or promotes.

Designed to be callable from dream_synthesize_compiler AND from the
ToolExecutor agentic harness in tools.py.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ── locations ────────────────────────────────────────────────────
COMPILER_DIR = Path("/archive/neurogolf/src/compiler")          # live / callable
STAGING_DIR = Path("/archive/neurogolf/src/compiler_staging")   # staged, NOT loaded
TASK_DIR = Path("/archive/neurogolf")
MEMORY_PATH = TASK_DIR / "arc_scientist_memory.json"
PENDING_PATH = Path("/archive/erebus/pending_promotions.json")  # dashboard queue
AUDIT_PATH = Path("/archive/erebus/compiler_audit.jsonl")
DISABLED_SENTINEL = TASK_DIR / ".compiler_disabled"             # kill-switch

# ── sandbox limits ───────────────────────────────────────────────
_CPU_SECONDS = 60          # RLIMIT_CPU
_MEM_BYTES = 4 * 1024**3   # RLIMIT_AS — generous; onnxruntime/numpy mmap
_FSIZE_BYTES = 5 * 1024**2  # RLIMIT_FSIZE — candidate can't write big files
_WALL_TIMEOUT = 120        # subprocess wall-clock

# ── static-safety policy ─────────────────────────────────────────
# Compiler modules build ONNX graphs from grids; they need numerics only.
_ALLOWED_IMPORTS = {
    "numpy", "np", "onnx", "onnxruntime", "math", "itertools", "collections",
    "typing", "dataclasses", "functools", "json", "copy", "heapq", "bisect",
    "re", "enum", "abc", "operator", "string",
}
# Names/attributes that indicate an escape or side effect. Rejected outright.
_DANGEROUS_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "system", "popen",
    "spawn", "spawnv", "spawnl", "fork", "remove", "unlink", "rmdir",
    "rename", "chmod", "chown", "kill", "socket", "connect", "urlopen",
    "Request", "loads", "load", "dumps", "getoutput", "check_output",
    "check_call", "call", "run", "Popen", "setattr", "delattr", "globals",
    "vars", "memoryview", "__subclasses__", "__globals__", "__bases__",
    "__mro__", "__builtins__", "__loader__", "__code__",
}


@dataclass
class ModuleInfo:
    name: str
    path: Path
    docstring: str
    detect_fns: list[str]
    compile_fns: list[str]
    line_count: int


# ═════════════════════════════════════════════════════════════════
# Inspection (read-only)
# ═════════════════════════════════════════════════════════════════


def list_compiler_modules(compiler_dir: Path = COMPILER_DIR) -> list[ModuleInfo]:
    """Return info for every .py module in the compiler directory."""
    if not compiler_dir.exists():
        return []
    infos = []
    for fp in sorted(compiler_dir.glob("*.py")):
        if fp.name == "__init__.py":
            continue
        try:
            src = fp.read_text()
            tree = ast.parse(src)
            doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
            detects = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("detect_")
            ]
            compiles = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and (n.name.startswith("compile_") or n.name == "make_model")
            ]
            infos.append(
                ModuleInfo(
                    name=fp.stem,
                    path=fp,
                    docstring=doc,
                    detect_fns=detects,
                    compile_fns=compiles,
                    line_count=src.count("\n"),
                )
            )
        except SyntaxError:
            # Skip broken modules (e.g. truncated dream outputs)
            continue
    return infos


def read_compiler_module(name: str, compiler_dir: Path = COMPILER_DIR) -> str | None:
    """Return the full source of a compiler module."""
    fp = compiler_dir / f"{name}.py"
    if not fp.exists():
        return None
    return fp.read_text()


def get_few_shot_modules(
    compiler_dir: Path = COMPILER_DIR, max_modules: int = 3, max_chars_each: int = 3000
) -> str:
    """Render 2-3 real compiler modules as few-shot examples for the LLM.

    Picks the shortest well-formed modules so they fit in context —
    these are the clearest patterns to imitate.
    """
    infos = [
        m for m in list_compiler_modules(compiler_dir) if m.detect_fns or m.compile_fns
    ]
    # Prefer shorter, simpler modules as references
    infos.sort(key=lambda m: m.line_count)
    picks = infos[:max_modules]
    blocks = []
    for m in picks:
        src = m.path.read_text()[:max_chars_each]
        blocks.append(f"# === {m.name}.py ({m.line_count} lines) ===\n{src}")
    return "\n\n".join(blocks)


def cluster_failures(
    memory_path: Path = MEMORY_PATH, day: str | None = None
) -> list[dict]:
    """Group the day's failures by (error_type, similar_to).

    Returns clusters sorted by size — the biggest cluster is where a new
    compiler module would have the most impact.
    """
    if not memory_path.exists():
        return []
    mem = json.loads(memory_path.read_text())
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for tn_str, tk in mem.get("tasks", {}).items():
        for a in tk.get("attempts", []):
            if a.get("verified"):
                continue
            if day and not a.get("timestamp", "").startswith(day):
                continue
            key = (
                a.get("error_type", "unknown"),
                a.get("similar_to", "") or "unclassified",
            )
            buckets[key].append(int(tn_str))
    clusters = []
    for (et, pattern), tasks in buckets.items():
        uniq = sorted(set(tasks))
        clusters.append(
            {
                "error_type": et,
                "pattern": pattern,
                "n_failures": len(tasks),
                "n_unique_tasks": len(uniq),
                "tasks": uniq[:20],
            }
        )
    clusters.sort(key=lambda c: -c["n_unique_tasks"])
    return clusters


# ═════════════════════════════════════════════════════════════════
# Safety checks
# ═════════════════════════════════════════════════════════════════


def syntax_check_module(code: str) -> tuple[bool, str]:
    """Parse the candidate module. Returns (ok, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def static_safety_check(code: str) -> tuple[bool, list[str]]:
    """AST scan for dangerous constructs. Returns (ok, flags).

    This is the load-bearing *code* gate: compiler modules only ever need
    numpy/onnx, so anything reaching for os/subprocess/socket, eval/exec,
    file writes, or sandbox-escape dunders is rejected before the code is
    ever executed — even in the sandbox.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax: {e.msg}"]
    flags: list[str] = []
    for node in ast.walk(tree):
        # disallowed imports (allowlist)
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORTS:
                    flags.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS:
                flags.append(f"disallowed import-from: {node.module}")
        # dangerous bare names
        elif isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            flags.append(f"dangerous name: {node.id}")
        # dangerous attribute access (e.g. os.system, x.__globals__)
        elif isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_NAMES:
            flags.append(f"dangerous attribute: .{node.attr}")
    # de-dup, preserve order
    seen: set[str] = set()
    uniq = [f for f in flags if not (f in seen or seen.add(f))]
    return (len(uniq) == 0), uniq


# ── sandboxed subprocess execution ───────────────────────────────


def _rlimit_preexec():  # pragma: no cover - POSIX-only, exercised on Atlas
    """preexec_fn that caps CPU, address space, and file size. POSIX only."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_BYTES, _FSIZE_BYTES))


_NET_GUARD = (
    "import socket as _sock\n"
    "def _no_net(*a, **k):\n"
    "    raise OSError('network disabled in sandbox')\n"
    "_sock.socket = _no_net\n"
)


def _run_py_sandboxed(
    script: str, timeout: int = _WALL_TIMEOUT
) -> subprocess.CompletedProcess:
    """Run a python script string in a resource-limited subprocess.

    rlimits (CPU/mem/file-size) apply on POSIX via preexec_fn; on Windows
    (dev/test) only the wall-clock timeout applies. A socket guard is
    prepended to the script so pure-Python network calls raise.
    """
    preexec = _rlimit_preexec if sys.platform != "win32" else None
    return subprocess.run(
        [sys.executable, "-u", "-c", _NET_GUARD + script],
        capture_output=True,
        text=True,
        timeout=timeout,
        preexec_fn=preexec,
    )


def import_check_module(code: str) -> tuple[bool, str, list[str]]:
    """Import the candidate in a SANDBOXED SUBPROCESS (never in-process).

    Returns (ok, error, defined_functions). A crash, hang, or resource
    blow-up in the candidate can no longer take down the caller.
    """
    harness = (
        "import json, sys\n"
        f"_SRC = {code!r}\n"
        "ns = {}\n"
        "try:\n"
        "    exec(_SRC, ns)\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}', "
        "'functions': []})); sys.exit(0)\n"
        "fns = [k for k, v in ns.items() if callable(v) and "
        "k.startswith(('detect_', 'compile_', 'make_'))]\n"
        "print(json.dumps({'ok': True, 'error': '', 'functions': fns}))\n"
    )
    try:
        r = _run_py_sandboxed(harness, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "import timed out (sandbox)", []
    lines = [ln for ln in r.stdout.strip().split("\n") if ln.startswith("{")]
    if not lines:
        return False, (r.stderr[-300:] or "no output from sandbox"), []
    d = json.loads(lines[-1])
    return bool(d.get("ok")), d.get("error", ""), d.get("functions", [])


def test_compile_against_tasks(
    code: str, task_nums: list[int], task_dir: Path = TASK_DIR
) -> dict:
    """Build the ONNX model with the candidate and run it on each task.

    Runs in the same resource-limited sandbox as the import check.
    Returns per-task verification counts.
    """
    script = _make_test_harness(code, task_nums, task_dir)
    try:
        r = _run_py_sandboxed(script, timeout=_WALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "per_task": {}}
    lines = [ln for ln in r.stdout.strip().split("\n") if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": r.stderr[-500:] or "no output", "per_task": {}}
    return json.loads(lines[-1])


def _make_test_harness(code: str, task_nums: list[int], task_dir: Path) -> str:
    """Build a self-contained test script that exec's the candidate, finds
    compile_X/make_model, builds the ONNX graph, runs it, compares."""
    return f"""
import json, sys, traceback
from pathlib import Path
import numpy as np

# Candidate module source
_SRC = {code!r}

ns = {{}}
try:
    exec(_SRC, ns)
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"exec: {{e}}", "per_task": {{}}}}))
    sys.exit(0)

# Find compile function and detector
compile_fn = next((ns[k] for k in ns
                   if callable(ns[k]) and k.startswith("compile_")), None)
detect_fn  = next((ns[k] for k in ns
                   if callable(ns[k]) and k.startswith("detect_")), None)
make_fn    = ns.get("make_model")

if not (compile_fn or make_fn):
    print(json.dumps({{"ok": False, "error": "no compile_X or make_model found",
                      "per_task": {{}}}}))
    sys.exit(0)

try:
    import onnxruntime as ort
except ImportError:
    print(json.dumps({{"ok": False, "error": "onnxruntime not installed",
                      "per_task": {{}}}}))
    sys.exit(0)

task_dir = Path({str(task_dir)!r})
results = {{}}
task_nums = {task_nums!r}
for tn in task_nums:
    tf = task_dir / f"task{{tn:03d}}.json"
    if not tf.exists():
        results[str(tn)] = {{"error": "task_not_found"}}
        continue
    try:
        task = json.loads(tf.read_text())
        # Get the model
        try:
            model = (compile_fn or make_fn)()
        except TypeError:
            # Some compile_fns need task or examples
            model = (compile_fn or make_fn)(task.get("train", []))

        sess = ort.InferenceSession(model.SerializeToString(),
                                     providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        correct = total = 0
        for split in ("train", "test"):
            for ex in task.get(split, []):
                total += 1
                arr = np.array(ex["input"], dtype=np.int64)
                try:
                    out = sess.run(None, {{input_name: arr}})[0]
                    if out.tolist() == ex["output"]:
                        correct += 1
                except Exception:
                    pass
        results[str(tn)] = {{"correct": correct, "total": total}}
    except Exception as e:
        results[str(tn)] = {{"error": f"{{type(e).__name__}}: {{e}}"}}

solved = sum(1 for v in results.values()
             if v.get("correct") and v.get("correct") == v.get("total"))
print(json.dumps({{"ok": True, "n_tasks": len(task_nums),
                  "n_solved": solved, "per_task": results}}))
"""


# ═════════════════════════════════════════════════════════════════
# Kill-switch + audit + provenance store
# ═════════════════════════════════════════════════════════════════


def is_disabled(sentinel: Path = DISABLED_SENTINEL) -> bool:
    """True if the self-tooling kill-switch sentinel is present."""
    return sentinel.exists()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit(event: str, record: dict, audit_path: Path = AUDIT_PATH) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _now_iso(), "event": event, **record}
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_pending(pending_path: Path = PENDING_PATH) -> list[dict]:
    try:
        d = json.loads(pending_path.read_text(encoding="utf-8"))
        return list(d.get("pending", [])) if isinstance(d, dict) else []
    except (FileNotFoundError, ValueError, OSError):
        return []


def _save_pending(reqs: list[dict], pending_path: Path = PENDING_PATH) -> None:
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from agi.common.atomic_write import atomic_write_text

        atomic_write_text(pending_path, json.dumps({"pending": reqs}, indent=2))
    except Exception:  # noqa: BLE001 - atomic_write is best-effort; fall back
        pending_path.write_text(json.dumps({"pending": reqs}, indent=2))


def list_pending_promotions(pending_path: Path = PENDING_PATH) -> list[dict]:
    """Pending-review promotion records (what the dashboard queue renders)."""
    return [r for r in _load_pending(pending_path) if r.get("status") == "pending"]


def _deme_check_code(code: str, provenance: dict | None) -> dict:
    """Best-effort DEME read on the candidate's natural-language provenance.

    The AST static check is the hard *code* gate; DEME here scores the
    human-readable rationale/docstring (which is what DEME can meaningfully
    assess). Unavailability is recorded, not fatal — the code gate stands.
    """
    text = ""
    try:
        tree = ast.parse(code)
        text = ast.get_docstring(tree) or ""
    except SyntaxError:
        pass
    if provenance:
        text = f"{text}\n{provenance.get('rationale', '')}"
    if not text.strip():
        return {"ran": False, "reason": "no rationale text", "allowed": True}
    try:
        from agi.safety.deme_gateway import GatewayConfig, SafetyGateway

        gw = SafetyGateway(GatewayConfig())
        res = gw.check_output(text, "compiler-module authoring")
        return {
            "ran": True,
            "allowed": bool(getattr(res, "passed", True)),
            "score": float(getattr(res, "score", 0.0)),
        }
    except Exception as e:  # noqa: BLE001 - DEME optional; record and continue
        return {"ran": False, "reason": f"gateway unavailable: {e}", "allowed": True}


def _candidate_id(tag: str, code: str) -> str:
    h = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:40]
    return f"{safe}-{h}"


# ═════════════════════════════════════════════════════════════════
# Stage (autonomous, L2)  →  Promote / Reject (human ack, L3)
# ═════════════════════════════════════════════════════════════════


def stage_compiler_module(
    code: str,
    test_task_nums: list[int],
    tag: str,
    *,
    provenance: dict | None = None,
    min_solved_ratio: float = 0.5,
    staging_dir: Path | None = None,
    task_dir: Path | None = None,
    pending_path: Path | None = None,
    audit_path: Path | None = None,
    sentinel: Path | None = None,
) -> dict:
    """Author-and-test a candidate, then STAGE it for human review.

    Full gate chain: kill-switch → syntax → static-safety → sandboxed
    import → sandboxed runtime-test → DEME(provenance). On success the
    candidate is written to the *staging* dir (NOT the live compiler dir)
    and a pending-promotion record is queued for the dashboard. It never
    becomes callable here — that needs :func:`promote_pending`.

    Path kwargs resolve to the module-level constants at call time (so
    ops/tests can override them by setting the module attribute).

    Returns a dict describing each stage and, on success, ``pending_id``.
    """
    staging_dir = staging_dir if staging_dir is not None else STAGING_DIR
    task_dir = task_dir if task_dir is not None else TASK_DIR
    pending_path = pending_path if pending_path is not None else PENDING_PATH
    audit_path = audit_path if audit_path is not None else AUDIT_PATH
    sentinel = sentinel if sentinel is not None else DISABLED_SENTINEL

    result: dict = {"tag": tag, "stages": [], "staged": False}

    if is_disabled(sentinel):
        result["reason"] = "self-tooling disabled (.compiler_disabled sentinel present)"
        return result

    ok, err = syntax_check_module(code)
    result["stages"].append({"stage": "syntax", "ok": ok, "error": err})
    if not ok:
        return result

    safe, flags = static_safety_check(code)
    result["stages"].append({"stage": "static_safety", "ok": safe, "flags": flags})
    if not safe:
        result["reason"] = "failed static safety scan"
        _audit("stage_rejected", {"tag": tag, "reason": "static_safety",
                                  "flags": flags}, audit_path)
        return result

    ok, err, fns = import_check_module(code)
    result["stages"].append(
        {"stage": "import", "ok": ok, "error": err, "functions": fns}
    )
    if not ok:
        return result

    test = {"ok": True, "n_tasks": 0, "n_solved": 0, "per_task": {}}
    if test_task_nums:
        test = test_compile_against_tasks(code, test_task_nums, task_dir)
        result["stages"].append({"stage": "runtime_test", **test})
        if not test.get("ok"):
            return result
        ratio = test["n_solved"] / max(test["n_tasks"], 1)
        result["solved_ratio"] = ratio
        if ratio < min_solved_ratio:
            result["reason"] = (
                f"solved_ratio {ratio:.0%} < threshold {min_solved_ratio:.0%}"
            )
            return result

    deme = _deme_check_code(code, provenance)
    result["stages"].append({"stage": "deme", **deme})
    if deme.get("ran") and not deme.get("allowed", True):
        result["reason"] = "DEME vetoed the module rationale"
        _audit("stage_rejected", {"tag": tag, "reason": "deme"}, audit_path)
        return result

    # All gates passed → write to staging (not callable) + queue for review.
    pid = _candidate_id(tag, code)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / f"{pid}.py"
    staged_path.write_text(code, encoding="utf-8")

    record = {
        "id": pid,
        "tag": tag,
        "staged_path": str(staged_path),
        "functions": fns,
        "provenance": provenance or {},
        "test": {
            "n_tasks": test.get("n_tasks", 0),
            "n_solved": test.get("n_solved", 0),
            "solved_ratio": result.get("solved_ratio"),
            "per_task": test.get("per_task", {}),
        },
        "static_safety": {"ok": safe, "flags": flags},
        "deme": deme,
        "status": "pending",
        "created_at": _now_iso(),
    }
    reqs = _load_pending(pending_path)
    # idempotent: replace any prior pending record with the same id
    reqs = [r for r in reqs if r.get("id") != pid]
    reqs.append(record)
    _save_pending(reqs, pending_path)
    _audit("staged", {"id": pid, "tag": tag,
                      "solved_ratio": result.get("solved_ratio"),
                      "functions": fns}, audit_path)

    result["staged"] = True
    result["pending_id"] = pid
    result["staged_path"] = str(staged_path)
    return result


def promote_pending(
    pending_id: str,
    *,
    approved_by: str,
    compiler_dir: Path | None = None,
    pending_path: Path | None = None,
    audit_path: Path | None = None,
    sentinel: Path | None = None,
) -> dict:
    """Human-acked promotion (tier L3): move a staged module into the live
    compiler dir so it becomes callable. Requires ``approved_by``.
    """
    compiler_dir = compiler_dir if compiler_dir is not None else COMPILER_DIR
    pending_path = pending_path if pending_path is not None else PENDING_PATH
    audit_path = audit_path if audit_path is not None else AUDIT_PATH
    sentinel = sentinel if sentinel is not None else DISABLED_SENTINEL

    if is_disabled(sentinel):
        return {"ok": False, "error": "self-tooling disabled (sentinel present)"}

    reqs = _load_pending(pending_path)
    rec = next((r for r in reqs if r.get("id") == pending_id), None)
    if rec is None:
        return {"ok": False, "error": f"no pending record {pending_id!r}"}
    if rec.get("status") != "pending":
        return {"ok": False, "error": f"record is {rec.get('status')}, not pending"}

    staged = Path(rec["staged_path"])
    if not staged.exists():
        return {"ok": False, "error": f"staged file missing: {staged}"}

    compiler_dir.mkdir(parents=True, exist_ok=True)
    live_path = compiler_dir / f"{rec['tag']}.py"
    live_path.write_text(staged.read_text(encoding="utf-8"), encoding="utf-8")
    staged.unlink(missing_ok=True)

    rec["status"] = "promoted"
    rec["approved_by"] = approved_by
    rec["promoted_at"] = _now_iso()
    rec["live_path"] = str(live_path)
    _save_pending(reqs, pending_path)
    _audit("promoted", {"id": pending_id, "tag": rec["tag"],
                        "approved_by": approved_by,
                        "live_path": str(live_path)}, audit_path)
    return {"ok": True, "live_path": str(live_path), "tag": rec["tag"]}


def reject_pending(
    pending_id: str,
    *,
    reason: str,
    rejected_by: str,
    pending_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict:
    """Human rejection: mark a staged module rejected and delete its file."""
    pending_path = pending_path if pending_path is not None else PENDING_PATH
    audit_path = audit_path if audit_path is not None else AUDIT_PATH
    reqs = _load_pending(pending_path)
    rec = next((r for r in reqs if r.get("id") == pending_id), None)
    if rec is None:
        return {"ok": False, "error": f"no pending record {pending_id!r}"}
    if rec.get("status") != "pending":
        return {"ok": False, "error": f"record is {rec.get('status')}, not pending"}

    Path(rec["staged_path"]).unlink(missing_ok=True)
    rec["status"] = "rejected"
    rec["rejected_by"] = rejected_by
    rec["reason"] = reason
    rec["rejected_at"] = _now_iso()
    _save_pending(reqs, pending_path)
    _audit("rejected", {"id": pending_id, "tag": rec["tag"],
                        "rejected_by": rejected_by, "reason": reason}, audit_path)
    return {"ok": True}


# ── backward-compatible entry point (now stages, never auto-promotes) ─


def write_compiler_module(
    code: str,
    test_task_nums: list[int],
    tag: str,
    min_solved_ratio: float = 0.5,
    compiler_dir: Path = COMPILER_DIR,
    task_dir: Path = TASK_DIR,
) -> dict:
    """Deprecated-in-place: previously wrote straight to the live compiler
    dir. It now routes through :func:`stage_compiler_module`, so a module
    Erebus authors is **staged for human review, never auto-promoted**.

    Kept so the ToolExecutor agentic harness keeps working with the safer
    behavior. ``compiler_dir`` is ignored (staging has its own dir).
    """
    res = stage_compiler_module(
        code, test_task_nums, tag,
        min_solved_ratio=min_solved_ratio, task_dir=task_dir,
    )
    # Preserve the old key name for callers that checked "promoted".
    res["promoted"] = False
    res["staged_for_review"] = res.get("staged", False)
    return res
