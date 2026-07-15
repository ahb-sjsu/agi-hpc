#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""erebus_ctl — park Erebus so Atlas is free for research, then bring it back.

Erebus is spread across systemd faculties plus a couple of hand-launched
``llama-server`` model backends that are *not* under systemd (so ``systemctl
stop`` alone will not reclaim their VRAM). This tool does both halves and,
crucially, restores the exact pre-suspend state on resume.

    erebus suspend            # full: stop faculties + free GPU VRAM
    erebus suspend --soft     # stop faculties only, leave GPU backends up
    erebus resume             # restore whatever suspend actually stopped
    erebus status [--json]    # what's up/down, GPU headroom, suspend marker

Design decisions
----------------
* **Watchdog first.** ``atlas-watchdog`` respawns dead faculties, so suspend
  stops it before anything else and resume starts it last.
* **Faithful restore, not resurrection.** Suspend records only the faculties
  that were *actually running* (Id/Superego/Scientist/Arbiter have been off by
  operator choice for months); resume starts back exactly that set and no more.
* **Sentinels as a backstop.** The usual ``.*_disabled`` files are dropped so a
  stray ``systemctl start`` or the nightly dreaming cron can't wake heavy work
  while suspended. Resume removes them.
* **GPU backends are snapshotted, killed, and replayed.** For each GPU-resident
  ``llama-server`` we save argv + cwd + full environ, SIGTERM/SIGKILL it, and on
  resume relaunch it verbatim (detached). State file is chmod 600 (environ may
  carry HF_TOKEN).

Runs as the ``claude`` user, which has passwordless sudo for systemctl.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── layout ───────────────────────────────────────────────────────────────
STATE_DIR = Path("/archive/erebus")
STATE_FILE = STATE_DIR / "suspend_state.json"
LOG_FILE = STATE_DIR / "erebus_ctl.log"
LLAMA_LOG_DIR = STATE_DIR / "llama_logs"
NG = Path("/archive/neurogolf")

WATCHDOG = "atlas-watchdog"

# Erebus cognition / compute faculties, in stop order (heaviest-coupled last).
# Infra that must survive a suspend so the box stays observable and resumable is
# deliberately absent: telemetry (the dashboard + this button), nats, memory,
# safety, rag-server, victoriametrics, thermal, caddy, oauth2-proxy, the avatars.
COG_UNITS = [
    "atlas-director",          # executive / self-model loop
    "atlas-erebus-discord",    # Discord faculty
    "atlas-primer",            # teaching daemon (vMOE over NRP)
    "atlas-dreaming-schedule", # nightly 2-4AM QLoRA trigger
    "atlas-dreaming",          # memory consolidation
    "atlas-ego",               # Divine Council (CPU)
    "atlas-reasoning",         # Ego deliberation NATS bridge
    "atlas-metacognition",     # metacognition
    "atlas-scientist",         # ARC solver (usually already down)
    "atlas-id",                # Id local model (GPU 1)
    "atlas-superego",          # Superego model (GPU 0)
    "atlas-llm-arbiter",       # arbiter LLM (CPU)
]

# Dropped on suspend, removed on resume. Keep faculties down even if something
# tries to start them, and block the nightly dreaming/QLoRA cron.
SENTINELS = [
    NG / ".erebus_disabled",
    NG / ".dreaming_schedule_disabled",
    NG / ".discord_disabled",
    NG / ".director_disabled",
    NG / ".compiler_disabled",
]


# ── small helpers ────────────────────────────────────────────────────────
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, file=sys.stderr)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except Exception as e:  # noqa: BLE001 - control surface, never raise
        return 1, f"{type(e).__name__}: {e}"


def _is_active(unit: str) -> bool:
    rc, out = _run(["systemctl", "is-active", f"{unit}.service"], timeout=15)
    return out.strip() == "active"


def _sc(action: str, unit: str) -> tuple[bool, str]:
    rc, out = _run(["sudo", "-n", "systemctl", action, f"{unit}.service"])
    return rc == 0, out


def _gpu_rows() -> list[dict]:
    rc, out = _run(
        ["nvidia-smi",
         "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], timeout=20)
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4 and p[0].isdigit():
            rows.append({"index": int(p[0]), "used_mib": int(p[1]),
                         "total_mib": int(p[2]), "util_pct": int(p[3])})
    return rows


# ── GPU llama-server snapshot / kill / replay ────────────────────────────
def _proc_field(pid: int, key: str) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _gpu_llama_leaders() -> list[int]:
    """Thread-group leaders of every GPU-resident ``llama-server`` (both GPUs)."""
    leaders: set[int] = set()
    for gpu in (0, 1):
        rc, out = _run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-compute-apps=pid", "--format=csv,noheader,nounits"], 20)
        for line in out.splitlines():
            s = line.strip()
            if not s.isdigit():
                continue
            pid = int(s)
            tgid = _proc_field(pid, "Tgid")
            leader = int(tgid) if tgid.isdigit() else pid
            if _comm(leader).startswith("llama-server"):
                leaders.add(leader)
    return sorted(leaders)


def _snapshot_proc(pid: int) -> dict | None:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        cwd = os.readlink(f"/proc/{pid}/cwd")
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        env = {}
        for kv in raw:
            if b"=" in kv:
                k, v = kv.split(b"=", 1)
                env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    except OSError as e:
        _log(f"snapshot pid {pid} failed: {e}")
        return None
    port = model = ""
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = argv[i + 1]
        elif a == "--model" and i + 1 < len(argv):
            model = os.path.basename(argv[i + 1])
    return {"pid": pid, "argv": argv, "cwd": cwd, "env": env,
            "port": port, "model": model}


def _kill_leader(pid: int, grace: float = 12.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError as e:
        _log(f"SIGTERM {pid}: {e}")
    deadline = time.time() + grace
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(1.0)
    except ProcessLookupError:
        return True
    except OSError as e:
        _log(f"SIGKILL {pid}: {e}")
    return not Path(f"/proc/{pid}").exists()


def _replay_llama(snap: dict) -> bool:
    argv, cwd = snap.get("argv") or [], snap.get("cwd") or "/home/claude"
    if not argv:
        return False
    env = dict(snap.get("env") or {})
    env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    try:
        LLAMA_LOG_DIR.mkdir(parents=True, exist_ok=True)
        tag = snap.get("port") or snap.get("model") or "llama"
        logf = open(LLAMA_LOG_DIR / f"llama-{tag}.log", "ab")
        subprocess.Popen(argv, cwd=cwd, env=env, stdout=logf, stderr=logf,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         close_fds=True)
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"replay {snap.get('model')} :{snap.get('port')} failed: {e}")
        return False


# ── state ────────────────────────────────────────────────────────────────
def _read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.chmod(tmp, 0o600)  # environ snapshots may carry HF_TOKEN
    os.replace(tmp, STATE_FILE)


def _clear_state() -> None:
    try:
        STATE_FILE.unlink()
    except FileNotFoundError:
        pass


# ── commands ─────────────────────────────────────────────────────────────
def suspend(soft: bool = False, by: str = "cli") -> dict:
    mode = "soft" if soft else "full"
    prior = _read_state()
    if prior.get("suspended"):
        return {"ok": True, "already": True, "state": prior,
                "note": "already suspended; run 'erebus resume' to restore"}

    _log(f"SUSPEND ({mode}) requested by {by}")
    gpu_before = _gpu_rows()

    # 1) watchdog first so it can't respawn what we stop
    stopped = []
    if _is_active(WATCHDOG):
        ok, _ = _sc("stop", WATCHDOG)
        _log(f"stop {WATCHDOG}: {'ok' if ok else 'FAILED'}")
        if ok:
            stopped.append(WATCHDOG)

    # 2) sentinels (durable down-state; blocks nightly dreaming/QLoRA)
    dropped = []
    for s in SENTINELS:
        try:
            s.parent.mkdir(parents=True, exist_ok=True)
            s.touch()
            dropped.append(str(s))
        except OSError as e:
            _log(f"sentinel {s}: {e}")

    # 3) stop the faculties that are actually running (faithful restore set)
    for u in COG_UNITS:
        if _is_active(u):
            ok, out = _sc("stop", u)
            _log(f"stop {u}: {'ok' if ok else 'FAILED ' + out}")
            if ok:
                stopped.append(u)

    # 4) GPU reclaim: snapshot + kill GPU-resident llama-servers
    killed = []
    if not soft:
        for pid in _gpu_llama_leaders():
            snap = _snapshot_proc(pid)
            if not snap:
                continue
            if _kill_leader(pid):
                _log(f"killed llama-server pid {pid} "
                     f"(:{snap['port']} {snap['model']})")
                killed.append(snap)
            else:
                _log(f"could not kill llama-server pid {pid}")

    state = {"suspended": True, "mode": mode, "at": _now(), "by": by,
             "stopped_units": stopped, "sentinels": dropped,
             "killed_llama": killed, "gpu_before": gpu_before}
    _write_state(state)
    gpu_after = _gpu_rows()
    _log(f"SUSPEND complete: {len(stopped)} units, {len(killed)} GPU backends")
    return {"ok": True, "mode": mode, "stopped_units": stopped,
            "killed_llama": [{"port": k["port"], "model": k["model"]} for k in killed],
            "gpu_before": gpu_before, "gpu_after": gpu_after}


def resume(by: str = "cli") -> dict:
    state = _read_state()
    if not state.get("suspended"):
        return {"ok": True, "already": True,
                "note": "not suspended; nothing to resume"}

    _log(f"RESUME requested by {by}")

    # 1) remove sentinels so faculties may start
    for s in state.get("sentinels", []):
        try:
            Path(s).unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            _log(f"remove sentinel {s}: {e}")

    # 2) replay the GPU backends we killed (detached, verbatim)
    relaunched = []
    for snap in state.get("killed_llama", []):
        if _replay_llama(snap):
            _log(f"relaunched llama-server :{snap.get('port')} {snap.get('model')}")
            relaunched.append({"port": snap.get("port"), "model": snap.get("model")})

    # 3) start faculties back — watchdog last so it doesn't race the starts
    stopped = [u for u in state.get("stopped_units", []) if u != WATCHDOG]
    started = []
    for u in stopped:
        ok, out = _sc("start", u)
        _log(f"start {u}: {'ok' if ok else 'FAILED ' + out}")
        if ok:
            started.append(u)
    if WATCHDOG in state.get("stopped_units", []):
        ok, _ = _sc("start", WATCHDOG)
        _log(f"start {WATCHDOG}: {'ok' if ok else 'FAILED'}")
        if ok:
            started.append(WATCHDOG)

    _clear_state()
    _log(f"RESUME complete: {len(started)} units, {len(relaunched)} GPU backends")
    return {"ok": True, "started_units": started, "relaunched_llama": relaunched,
            "gpu": _gpu_rows()}


def status() -> dict:
    state = _read_state()
    live = {u: _is_active(u) for u in [WATCHDOG] + COG_UNITS}
    return {
        "suspended": bool(state.get("suspended")),
        "mode": state.get("mode"),
        "since": state.get("at"),
        "by": state.get("by"),
        "stopped_units": state.get("stopped_units", []),
        "killed_llama": [{"port": k.get("port"), "model": k.get("model")}
                         for k in state.get("killed_llama", [])],
        "live_units": live,
        "running_faculties": sorted(u for u, a in live.items() if a),
        "gpus": _gpu_rows(),
    }


# ── cli ──────────────────────────────────────────────────────────────────
def _print_status(st: dict) -> None:
    head = "SUSPENDED" if st["suspended"] else "RUNNING"
    print(f"Erebus: {head}", end="")
    if st["suspended"]:
        print(f"  (mode={st['mode']} since {st['since']} by {st['by']})")
    else:
        print()
    for g in st["gpus"]:
        free = g["total_mib"] - g["used_mib"]
        print(f"  GPU{g['index']}: {g['used_mib']:>6} / {g['total_mib']} MiB used"
              f"  ({free} MiB free, {g['util_pct']}% util)")
    up = st["running_faculties"]
    print(f"  faculties up ({len(up)}): {', '.join(up) if up else '(none)'}")
    if st["suspended"] and st["killed_llama"]:
        for k in st["killed_llama"]:
            print(f"  parked GPU backend: :{k['port']} {k['model']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="erebus", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("suspend", help="park Erebus, free the box")
    ps.add_argument("--soft", action="store_true",
                    help="stop faculties only; leave GPU backends running")
    ps.add_argument("--by", default="cli")
    pr = sub.add_parser("resume", help="restore whatever suspend stopped")
    pr.add_argument("--by", default="cli")
    pt = sub.add_parser("status", help="show suspend state + GPU headroom")
    pt.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "suspend":
        out = suspend(soft=args.soft, by=args.by)
        print(json.dumps(out, indent=2))
    elif args.cmd == "resume":
        out = resume(by=args.by)
        print(json.dumps(out, indent=2))
    else:
        st = status()
        if args.json:
            print(json.dumps(st, indent=2))
        else:
            _print_status(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
