#!/usr/bin/env python3
"""Atlas Thermal Guardian v2 — thermal protection with workload classes and placement.

v1 paused *every* process above 20% CPU whenever any CPU package crossed the
target temperature. That included the desktop user's game, which was stopped for
2–4 s every 10–80 s while Docker's daemons heated package 0 (2026-09-18).

v2 keeps the same thermal envelope but adds a policy layer:

Workload classes
  SERVICE      never paused, never re-pinned (llama-server, NATS, DB, X, ...).
  INTERACTIVE  the desktop user's foreground workloads (Steam / Proton / Wine
               games, Mod Organizer). Never paused below CRITICAL_TEMP.
  BATCH        everything else that is CPU-heavy (Docker daemons, kubectl,
               research jobs). Paused above TARGET_TEMP as in v1.

Placement (anti-affinity between INTERACTIVE and BATCH)
  Two sockets. While at least one INTERACTIVE process exists ("gaming mode"):
    - INTERACTIVE processes are pinned to INTERACTIVE_CPUS (package 1, the
      cooler socket, all threads);
    - hot BATCH processes are pinned to BATCH_CPUS (package 0), so their heat
      lands on the socket the game does not use.
  When gaming mode ends, BATCH processes get their original masks back.

Thermal policy, evaluated per package every POLL_INTERVAL seconds
  temp >= TARGET_TEMP    pause BATCH processes that may run on that package,
                         throttle RAID resync.
  temp >= CRITICAL_TEMP  pause everything that is not SERVICE, INTERACTIVE
                         included (emergency, same as v1).
  temp <= COOLDOWN_TEMP  resume everything paused because of that package.

GPU advisory: when a game and llama-server share a GPU a warning is logged
(display GPU contention causes stutter); nothing is moved automatically.

Runs as a systemd service (atlas-thermal.service, root).
  --dry-run   log every decision, send no signals, change no affinities
  --once      one evaluation and exit (for tests)
"""

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [thermal] %(message)s")
log = logging.getLogger("thermal")

# ----------------------------------------------------------------------------- policy
TARGET_TEMP = 82.0     # per package: start pausing BATCH above this
CRITICAL_TEMP = 95.0   # per package: pause everything non-SERVICE
COOLDOWN_TEMP = 75.0   # resume when the package is back below this
POLL_INTERVAL = 2.0
HOT_CPU_PCT = 20.0     # BATCH process counts as hot above this (instantaneous, all threads)
CRIT_CPU_PCT = 5.0

# Socket topology (lscpu, HP Z840, 2x E5-2690 v3):
#   NUMA node0 / package 0: CPUs 0-11,24-35     NUMA node1 / package 1: CPUs 12-23,36-47
PACKAGE_CPUS = {
    0: set(range(0, 12)) | set(range(24, 36)),
    1: set(range(12, 24)) | set(range(36, 48)),
}
INTERACTIVE_CPUS = PACKAGE_CPUS[1]
BATCH_CPUS = PACKAGE_CPUS[0]
ALL_CPUS = PACKAGE_CPUS[0] | PACKAGE_CPUS[1]

# Never paused, never re-pinned (matched against the executable name).
SERVICE = {
    "llama-server", "caddy", "oauth2-proxy", "nats-server", "python3", "postgres",
    "systemd", "sshd", "bash", "thermal_guard", "Xorg", "xfce4-session", "xfwm4",
    "pipewire", "pulseaudio", "wireplumber", "xrdp", "xrdp-sesman", "netdata",
}

# INTERACTIVE = owned by the desktop user AND command matches this.
DESKTOP_UID = 1000  # ahbond
INTERACTIVE_RE = re.compile(
    r"(steam|proton|wine|gamescope|ModOrganizer|\.exe(\s|$)|[A-Z]:\\)", re.IGNORECASE)
INTERACTIVE_MIN_CPU = 5.0   # ignore idle helpers (steamwebhelper at 0%)

RAID_THROTTLE_KB = {"target": 5000, "critical": 1000, "normal": 200000}

# ----------------------------------------------------------------------------- state
paused = {}          # pid -> (package, class, comm)
pinned_batch = {}    # pid -> original cpu set, to restore after gaming mode
_prev_cpu = {}       # pid -> (cputime_ticks, wall)
_gpu_warned_at = 0.0
CLK = os.sysconf("SC_CLK_TCK")
DRY = False


# ----------------------------------------------------------------------------- sensors
def read_temps():
    """{package_id: temp} from lm-sensors coretemp 'Package id N' lines."""
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=3)
    except Exception:
        return {}
    temps = {}
    for line in out.splitlines():
        m = re.match(r"Package id (\d+):\s+\+(\d+\.\d+)", line)
        if m:
            temps[int(m.group(1))] = float(m.group(2))
    return temps


# ----------------------------------------------------------------------------- processes
def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def list_processes():
    """[(pid, uid, comm, args, cpu_pct)] with instantaneous CPU over the last poll."""
    now = time.time()
    procs = []
    seen = set()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        stat = _read(f"/proc/{pid}/stat")
        status = _read(f"/proc/{pid}/status")
        if not stat or not status:
            continue
        try:
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])           # utime + stime, all threads
            uid = int(re.search(r"^Uid:\s+(\d+)", status, re.M).group(1))
        except (ValueError, AttributeError, IndexError):
            continue
        args = (_read(f"/proc/{pid}/cmdline") or "").replace("\0", " ").strip() or comm
        prev = _prev_cpu.get(pid)
        _prev_cpu[pid] = (ticks, now)
        seen.add(pid)
        cpu = 0.0
        if prev and now > prev[1]:
            cpu = 100.0 * (ticks - prev[0]) / CLK / (now - prev[1])
        procs.append((pid, uid, comm, args, cpu))
    for pid in list(_prev_cpu):
        if pid not in seen:
            del _prev_cpu[pid]
    return procs


def classify(pid, uid, comm, args, cpu):
    if pid == os.getpid() or pid == 1 or comm in SERVICE or comm.startswith("kworker"):
        return "SERVICE"
    if uid == DESKTOP_UID and INTERACTIVE_RE.search(args):
        return "INTERACTIVE"
    return "BATCH"


def affinity(pid):
    try:
        return os.sched_getaffinity(pid)
    except OSError:
        return None


def set_affinity(pid, cpus, why):
    """Pin every thread of pid. New threads inherit from their creator."""
    if DRY:
        log.info("DRY would pin pid %d to %s (%s)", pid, cpuset_str(cpus), why)
        return True
    ok = False
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                os.sched_setaffinity(int(tid), cpus)
                ok = True
            except OSError:
                pass
    except OSError:
        return False
    if ok:
        log.info("PINNED pid %d to %s (%s)", pid, cpuset_str(cpus), why)
    return ok


def cpuset_str(cpus):
    if cpus == ALL_CPUS:
        return "all"
    for k, v in PACKAGE_CPUS.items():
        if cpus == v:
            return f"package{k}"
    return ",".join(str(c) for c in sorted(cpus))


# ----------------------------------------------------------------------------- actions
def throttle_raid(level):
    kb = RAID_THROTTLE_KB[level]
    if DRY:
        return
    try:
        with open("/proc/sys/dev/raid/speed_limit_max", "w") as f:
            f.write(str(kb))
    except Exception:
        pass


def pause(pid, pkg, cls, comm, why):
    if pid in paused:
        return
    if DRY:
        log.warning("DRY would PAUSE pid %d %s [%s] — %s", pid, comm, cls, why)
        paused[pid] = (pkg, cls, comm)
        return
    try:
        os.kill(pid, signal.SIGSTOP)
        paused[pid] = (pkg, cls, comm)
        log.warning("PAUSED pid %d %s [%s] — %s", pid, comm, cls, why)
    except ProcessLookupError:
        pass


def resume(pid):
    pkg, cls, comm = paused.pop(pid)
    if DRY:
        log.info("DRY would RESUME pid %d %s", pid, comm)
        return
    try:
        os.kill(pid, signal.SIGCONT)
        log.info("RESUMED pid %d %s", pid, comm)
    except ProcessLookupError:
        pass


def resume_package(pkg):
    for pid, (p, _, _) in list(paused.items()):
        if p == pkg:
            resume(pid)


def gpu_advisory(gaming):
    """Warn (at most once a minute) when a game shares a GPU with llama-server."""
    global _gpu_warned_at
    if not gaming or time.time() - _gpu_warned_at < 60:
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,process_name",
             "--format=csv,noheader"], text=True, timeout=5)
    except Exception:
        return
    by_gpu = {}
    for line in out.splitlines():
        if "," not in line:
            continue
        uuid, name = [x.strip() for x in line.split(",", 1)]
        by_gpu.setdefault(uuid, []).append(name)
    for uuid, names in by_gpu.items():
        has_llm = any("llama-server" in n for n in names)
        has_game = any(INTERACTIVE_RE.search(n) for n in names)
        if has_llm and has_game:
            log.warning("GPU %s hosts both llama-server and a game — inference bursts "
                        "will stutter the game; consider CUDA_VISIBLE_DEVICES=1 for the LLM",
                        uuid[-8:])
            _gpu_warned_at = time.time()


# ----------------------------------------------------------------------------- one tick
def tick(state):
    temps = read_temps()
    if not temps:
        return
    procs = [(pid, uid, comm, args, cpu, classify(pid, uid, comm, args, cpu))
             for pid, uid, comm, args, cpu in list_processes()]
    interactive = [p for p in procs if p[5] == "INTERACTIVE" and p[4] >= INTERACTIVE_MIN_CPU]
    gaming = bool(interactive)

    # --- placement -----------------------------------------------------------
    if gaming and not state["gaming"]:
        log.info("gaming mode ON: %s", ", ".join(f"{p[2]}({p[0]})" for p in interactive[:4]))
    if not gaming and state["gaming"]:
        log.info("gaming mode OFF: restoring %d batch affinities", len(pinned_batch))
        for pid, orig in list(pinned_batch.items()):
            set_affinity(pid, orig, "gaming mode ended")
            del pinned_batch[pid]
    state["gaming"] = gaming
    if gaming:
        for pid, uid, comm, args, cpu, cls in procs:
            if cls == "INTERACTIVE" and cpu >= INTERACTIVE_MIN_CPU:
                cur = affinity(pid)
                if cur is not None and not cur <= INTERACTIVE_CPUS:
                    set_affinity(pid, INTERACTIVE_CPUS, "interactive placement")
            elif cls == "BATCH" and cpu >= HOT_CPU_PCT and pid not in pinned_batch:
                cur = affinity(pid)
                if cur is not None and cur & INTERACTIVE_CPUS:
                    if set_affinity(pid, BATCH_CPUS, f"anti-affinity, {cpu:.0f}% cpu"):
                        pinned_batch[pid] = cur
    for pid in list(pinned_batch):
        if not os.path.exists(f"/proc/{pid}"):
            del pinned_batch[pid]

    # --- thermal, per package ------------------------------------------------
    for pkg, temp in sorted(temps.items()):
        cpus = PACKAGE_CPUS.get(pkg, ALL_CPUS)
        was_hot = state["hot"].get(pkg, False)
        if temp >= CRITICAL_TEMP:
            if not state["crit"].get(pkg):
                log.critical("package %d at %.0f°C — EMERGENCY, pausing everything non-service", pkg, temp)
                state["crit"][pkg] = True
            throttle_raid("critical")
            for pid, uid, comm, args, cpu, cls in procs:
                if cls != "SERVICE" and cpu >= CRIT_CPU_PCT:
                    aff = affinity(pid)
                    if aff is None or aff & cpus:
                        pause(pid, pkg, cls, comm, f"package {pkg} at {temp:.0f}°C (critical)")
            state["hot"][pkg] = True
        elif temp >= TARGET_TEMP:
            if not was_hot:
                log.warning("package %d at %.0f°C — pausing hot batch work on it", pkg, temp)
            throttle_raid("target")
            for pid, uid, comm, args, cpu, cls in procs:
                if cls == "BATCH" and cpu >= HOT_CPU_PCT:
                    aff = affinity(pid)
                    if aff is None or aff & cpus:
                        pause(pid, pkg, cls, comm, f"package {pkg} at {temp:.0f}°C")
            state["hot"][pkg] = True
        elif temp <= COOLDOWN_TEMP and was_hot:
            log.info("package %d cooled to %.0f°C — resuming", pkg, temp)
            resume_package(pkg)
            state["hot"][pkg] = False
            state["crit"][pkg] = False
            if not any(state["hot"].values()):
                throttle_raid("normal")

    # paused processes that died
    for pid in list(paused):
        if not os.path.exists(f"/proc/{pid}"):
            del paused[pid]

    gpu_advisory(gaming)
    if state["ticks"] % 30 == 0:   # once a minute
        log.info("temps %s | gaming=%s | paused=%d | pinned_batch=%d",
                 " ".join(f"p{k}={v:.0f}" for k, v in sorted(temps.items())),
                 gaming, len(paused), len(pinned_batch))
    state["ticks"] += 1


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    DRY = args.dry_run
    log.info("Thermal guardian v2: target=%.0f critical=%.0f cooldown=%.0f interactive=%s batch=%s%s",
             TARGET_TEMP, CRITICAL_TEMP, COOLDOWN_TEMP, cpuset_str(INTERACTIVE_CPUS),
             cpuset_str(BATCH_CPUS), " DRY-RUN" if DRY else "")
    state = {"gaming": False, "hot": {}, "crit": {}, "ticks": 0}
    list_processes()            # prime the CPU sampler
    time.sleep(1.0)
    while True:
        try:
            tick(state)
        except Exception as e:  # never let the guardian die on a parse error
            log.exception("tick failed: %s", e)
        if args.once:
            if state["ticks"] < 2:      # second tick has real CPU deltas
                time.sleep(POLL_INTERVAL)
                continue
            return
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
