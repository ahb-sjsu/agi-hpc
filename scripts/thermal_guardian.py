#!/usr/bin/env python3
"""Atlas Thermal Guardian — prevents CPU thermal runaway.

Monitors CPU package temperatures. When any package exceeds the target,
aggressively throttles by:
1. Sending SIGSTOP to the heaviest CPU processes (pausing them)
2. Throttling RAID resync speed
3. Waiting for cooldown
4. Resuming processes with SIGCONT

Runs as a systemd service. Checks every 2 seconds.
"""

import logging
import os
import signal
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [thermal] %(message)s")
log = logging.getLogger("thermal")

TARGET_TEMP = 82.0     # Start throttling above this
CRITICAL_TEMP = 95.0   # Emergency: SIGSTOP everything non-essential
COOLDOWN_TEMP = 75.0   # Resume paused processes below this
POLL_INTERVAL = 2.0

# PIDs we have paused (will SIGCONT when cool)
paused_pids = set()

# Services that should NEVER be paused
PROTECTED = {"llama-server", "caddy", "oauth2-proxy", "nats-server", "python3",
             "postgres", "systemd", "sshd", "bash", "thermal_guard"}


def read_temps():
    """Read CPU package temperatures."""
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=3)
        temps = []
        for line in out.splitlines():
            if "Package" in line:
                import re
                m = re.search(r"\+(\d+\.\d+)", line)
                if m:
                    temps.append(float(m.group(1)))
        return temps
    except Exception:
        return []


def get_hot_processes(min_cpu=10.0):
    """Get PIDs of processes using > min_cpu% CPU, excluding protected."""
    try:
        out = subprocess.check_output(
            ["ps", "aux", "--sort=-%cpu"], text=True, timeout=3
        )
        pids = []
        for line in out.splitlines()[1:20]:
            parts = line.split()
            if len(parts) < 11:
                continue
            pid = int(parts[1])
            cpu = float(parts[2])
            cmd = parts[10].split("/")[-1]
            if cpu >= min_cpu and cmd not in PROTECTED and pid != os.getpid():
                pids.append((pid, cpu, cmd))
        return pids
    except Exception:
        return []


def throttle_raid(speed_kb=5000):
    """Throttle RAID resync speed."""
    try:
        with open("/proc/sys/dev/raid/speed_limit_max", "w") as f:
            f.write(str(speed_kb))
        log.info("RAID resync throttled to %d KB/s", speed_kb)
    except Exception:
        pass


def unthrottle_raid(speed_kb=200000):
    """Restore RAID resync speed."""
    try:
        with open("/proc/sys/dev/raid/speed_limit_max", "w") as f:
            f.write(str(speed_kb))
    except Exception:
        pass


def pause_process(pid, cmd):
    """Send SIGSTOP to pause a process."""
    try:
        os.kill(pid, signal.SIGSTOP)
        paused_pids.add(pid)
        log.warning("PAUSED pid %d (%s) — CPU too hot", pid, cmd)
    except ProcessLookupError:
        pass


def resume_all():
    """Send SIGCONT to all paused processes."""
    for pid in list(paused_pids):
        try:
            os.kill(pid, signal.SIGCONT)
            log.info("RESUMED pid %d", pid)
        except ProcessLookupError:
            pass
    paused_pids.clear()
    unthrottle_raid()


def main():
    log.info("Thermal guardian started: target=%.0f, critical=%.0f, cooldown=%.0f",
             TARGET_TEMP, CRITICAL_TEMP, COOLDOWN_TEMP)

    throttled = False

    while True:
        temps = read_temps()
        if not temps:
            time.sleep(POLL_INTERVAL)
            continue

        max_temp = max(temps)

        if max_temp >= CRITICAL_TEMP and not throttled:
            log.critical("CPU at %.0f — EMERGENCY THROTTLE", max_temp)
            throttle_raid(1000)
            for pid, cpu, cmd in get_hot_processes(min_cpu=5.0):
                pause_process(pid, cmd)
            throttled = True

        elif max_temp >= TARGET_TEMP and not throttled:
            log.warning("CPU at %.0f — throttling", max_temp)
            throttle_raid(5000)
            for pid, cpu, cmd in get_hot_processes(min_cpu=20.0):
                pause_process(pid, cmd)
            throttled = True

        elif max_temp <= COOLDOWN_TEMP and throttled:
            log.info("CPU cooled to %.0f — resuming", max_temp)
            resume_all()
            throttled = False

        if throttled and max_temp < TARGET_TEMP:
            # Partially cooled but not below cooldown — log but don't resume yet
            pass

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
