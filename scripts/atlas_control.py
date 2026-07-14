"""Atlas control plane — service start/stop/restart + GPU-1 maintenance mode.

Imported by telemetry_server.py, which exposes it as:

    GET  /api/control/status            unit states, GPUs, maint flags, audit tail
    POST /api/control/service           {"unit": ..., "action": ..., "force": bool}
    POST /api/control/maint             {"mode": "enter"|"exit", "force": bool}

Security model (defense in depth):
  1. Caddy + oauth2-proxy gate the network path (as for every endpoint).
  2. In-process: control POSTs require X-Forwarded-Email in ADMIN_EMAILS
     (or a localhost client for CLI testing).
  3. sudoers is the last line: /etc/sudoers.d/atlas-control enumerates the
     exact systemctl verbs+units; nothing outside ALLOWED_UNITS is escalatable
     even if this process is compromised.
  4. Every action (allowed or refused) is appended to an audit JSONL.

Chat guard: stopping/restarting a chat-critical unit is refused while a chat
turn ran in the last CHAT_GUARD_S seconds, unless force=true. The Erebus chat
handler calls note_chat() on every turn; the timestamp also persists to disk
so a telemetry restart doesn't forget an in-flight conversation.

Maintenance mode (GPU 1) follows the sentinel-file convention
(.claude/rules/atlas-operations.md): unit stays enabled, a sentinel at
/archive/neurogolf/.gpu1_maint blocks it via ConditionPathExists, so nothing
is silently left disabled. Enter = stop atlas-id + drop sentinel + report
GPU-1 residents (killed only with force, never GPU 0). Exit = remove sentinel
only; restoring the local Ego is an explicit, separate start action.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

AUDIT_LOG = Path.home() / "logs" / "control_audit.jsonl"
LAST_CHAT_FILE = Path.home() / "logs" / ".last_chat_ts"
GPU1_SENTINEL = Path("/archive/neurogolf/.gpu1_maint")
CHAT_GUARD_S = 600

ADMIN_EMAILS = set(
    filter(None, os.environ.get("ATLAS_CONTROL_ADMINS", "agi.hpc@gmail.com").split(","))
)

# unit -> (description, allowed actions, chat_critical)
ALLOWED_UNITS = {
    "atlas-id": ("Ego local (Qwen 32B, GPU 1)", {"start", "stop", "restart"}, False),
    "atlas-superego": ("Superego (GPU 0)", {"start", "stop", "restart"}, True),
    "atlas-ego": ("Divine Council (CPU)", {"start", "stop", "restart"}, True),
    "atlas-scientist": ("Erebus ARC Scientist", {"start", "stop", "restart"}, False),
    "atlas-primer": ("The Primer (tutor)", {"start", "stop", "restart"}, False),
    "atlas-dreaming": ("Dreaming consolidator", {"start", "stop", "restart"}, False),
    "atlas-dreaming-schedule": (
        "Dreaming scheduler",
        {"start", "stop", "restart"},
        False,
    ),
    "atlas-jobs": ("Thermal job queue", {"start", "stop", "restart"}, False),
    "atlas-memory": ("Memory NATS broker", {"start", "stop", "restart"}, True),
    "atlas-safety": ("DEME Safety NATS", {"start", "stop", "restart"}, True),
    "atlas-reasoning": ("Reasoning service", {"start", "stop", "restart"}, True),
    "atlas-metacognition": ("Metacognition", {"start", "stop", "restart"}, False),
    "atlas-rag-server": ("RAG server", {"start", "stop", "restart"}, True),
    "atlas-artemis-avatar": ("Artemis avatar", {"start", "stop", "restart"}, False),
    "atlas-sigma4-avatar": ("Sigma4 avatar", {"start", "stop", "restart"}, False),
    "atlas-artemis-tts-worker": ("Artemis TTS", {"start", "stop", "restart"}, False),
    "atlas-llm-arbiter": ("Arbiter LLM (CPU 72B)", {"start", "stop", "restart"}, False),
    "atlas-backup": ("Daily backup (run now)", {"start"}, False),
}


# --------------------------------------------------------------- chat guard
def note_chat() -> None:
    try:
        LAST_CHAT_FILE.parent.mkdir(exist_ok=True)
        LAST_CHAT_FILE.touch()
    except OSError:
        pass


def last_chat_age() -> float | None:
    try:
        return time.time() - LAST_CHAT_FILE.stat().st_mtime
    except OSError:
        return None


def _chat_guard_blocks(unit: str, action: str) -> bool:
    if action == "start":
        return False
    if not ALLOWED_UNITS.get(unit, ("", set(), False))[2]:
        return False
    age = last_chat_age()
    return age is not None and age < CHAT_GUARD_S


# ------------------------------------------------------------------- audit
def _audit(kind: str, email: str, detail: dict, ok: bool, msg: str) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "email": email or "(local)",
        "ok": ok,
        "msg": msg,
        **detail,
    }
    try:
        AUDIT_LOG.parent.mkdir(exist_ok=True)
        with AUDIT_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def audit_tail(n: int = 8) -> list[dict]:
    try:
        lines = AUDIT_LOG.read_text().strip().splitlines()[-n:]
        return [json.loads(x) for x in lines]
    except (OSError, ValueError):
        return []


# ------------------------------------------------------------------ status
def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except Exception as e:  # noqa: BLE001 - status surface, never raise
        return 1, f"{type(e).__name__}: {e}"


def unit_states() -> list[dict]:
    out = []
    for unit, (desc, actions, chat_crit) in ALLOWED_UNITS.items():
        _, active = _run(
            ["systemctl", "show", f"{unit}.service", "-p", "ActiveState", "--value"]
        )
        _, sub = _run(
            ["systemctl", "show", f"{unit}.service", "-p", "SubState", "--value"]
        )
        out.append(
            {
                "unit": unit,
                "desc": desc,
                "active": active,
                "sub": sub,
                "actions": sorted(actions),
                "chat_critical": chat_crit,
            }
        )
    return out


def gpu_state() -> list[dict]:
    _, out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "mem_used_mib": int(parts[1]),
                    "mem_total_mib": int(parts[2]),
                    "util_pct": int(parts[3]),
                }
            )
    return gpus


def _gpu_pids(gpu_index: int) -> list[dict]:
    """Compute PIDs resident on one GPU (via per-GPU query)."""
    _, out = _run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    pids = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            _, cmd = _run(["ps", "-p", parts[0], "-o", "cmd="])
            pids.append(
                {"pid": int(parts[0]), "mem_mib": int(parts[1]), "cmd": cmd[:120]}
            )
    return pids


def status() -> dict:
    return {
        "units": unit_states(),
        "gpus": gpu_state(),
        "maint": {"gpu1": GPU1_SENTINEL.exists()},
        "last_chat_age_s": last_chat_age(),
        "chat_guard_s": CHAT_GUARD_S,
        "audit": audit_tail(),
    }


# ----------------------------------------------------------------- actions
def _authorized(email: str, client_ip: str) -> bool:
    if email in ADMIN_EMAILS:
        return True
    return client_ip in ("127.0.0.1", "::1") and not email


def control_service(
    unit: str, action: str, email: str, client_ip: str, force: bool = False
) -> tuple[dict, int]:
    detail = {"unit": unit, "action": action, "force": force}
    if not _authorized(email, client_ip):
        _audit("service", email, detail, False, "unauthorized")
        return {"error": "unauthorized"}, 403
    if unit not in ALLOWED_UNITS:
        _audit("service", email, detail, False, "unit not allowlisted")
        return {"error": f"unit not allowlisted: {unit}"}, 400
    if action not in ALLOWED_UNITS[unit][1]:
        _audit("service", email, detail, False, "action not allowed for unit")
        return {"error": f"action not allowed for {unit}: {action}"}, 400
    if not force and _chat_guard_blocks(unit, action):
        _audit("service", email, detail, False, "chat guard")
        return {
            "error": "chat active within guard window; retry with force=true",
            "last_chat_age_s": last_chat_age(),
        }, 409
    rc, out = _run(["sudo", "-n", "systemctl", action, f"{unit}.service"], timeout=60)
    ok = rc == 0
    _audit("service", email, detail, ok, out[:200] if out else action)
    _, active = _run(
        ["systemctl", "show", f"{unit}.service", "-p", "ActiveState", "--value"]
    )
    return (
        {
            "ok": ok,
            "unit": unit,
            "action": action,
            "active": active,
            "detail": out[:200],
        },
        200 if ok else 500,
    )


def maint_gpu1(
    mode: str, email: str, client_ip: str, force: bool = False
) -> tuple[dict, int]:
    detail = {"mode": mode, "force": force}
    if not _authorized(email, client_ip):
        _audit("maint", email, detail, False, "unauthorized")
        return {"error": "unauthorized"}, 403

    if mode == "enter":
        rc, out = _run(
            ["sudo", "-n", "systemctl", "stop", "atlas-id.service"], timeout=60
        )
        try:
            GPU1_SENTINEL.touch()
        except OSError as e:
            _audit("maint", email, detail, False, f"sentinel: {e}")
            return {"error": f"could not write sentinel: {e}"}, 500
        residents = _gpu_pids(1)
        killed = []
        if residents and force:
            for p in residents:
                # per-GPU-1 list by construction (never GPU 0); no sudo — only
                # claude-owned processes are killable from the dashboard
                rc_k, _ = _run(["kill", "-9", str(p["pid"])])
                if rc_k == 0:
                    killed.append(p["pid"])
            residents = _gpu_pids(1)
        _audit(
            "maint",
            email,
            detail,
            True,
            f"enter; stop rc={rc}; residents={len(residents)}; killed={killed}",
        )
        return {
            "ok": True,
            "mode": "enter",
            "sentinel": str(GPU1_SENTINEL),
            "gpu1_residents": residents,
            "killed": killed,
            "note": (
                "GPU 1 free"
                if not residents
                else "residual processes listed; re-run with force=true to kill"
            ),
        }, 200

    if mode == "exit":
        try:
            GPU1_SENTINEL.unlink(missing_ok=True)
        except OSError as e:
            _audit("maint", email, detail, False, f"sentinel: {e}")
            return {"error": f"could not remove sentinel: {e}"}, 500
        _audit("maint", email, detail, True, "exit; sentinel removed")
        return {
            "ok": True,
            "mode": "exit",
            "note": (
                "maintenance cleared; start atlas-id explicitly to restore Ego"
            ),
        }, 200

    _audit("maint", email, detail, False, "bad mode")
    return {"error": "mode must be enter|exit"}, 400
