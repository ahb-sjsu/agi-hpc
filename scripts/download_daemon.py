#!/usr/bin/env python3
"""Atlas Download Daemon — Circuit breaker pattern for knowledge pipeline.

Runs continuously, monitors bandwidth, launches/kills downloads to
maintain target throughput. Self-heals when downloads stall or fail.

Circuit breaker states per source:
  CLOSED  — working, allow downloads
  OPEN    — failed/stalled, skip for cooldown period
  HALF    — cooldown expired, try one download to test

Usage:
    python3 scripts/download_daemon.py
    python3 scripts/download_daemon.py --target-mbps 50 --max-concurrent 6
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("dl-daemon")


class CircuitState(Enum):
    CLOSED = "closed"  # healthy, allow downloads
    OPEN = "open"  # failed, skip
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class DownloadSource:
    name: str
    urls: list  # list of URLs to download from this source
    dest: str
    method: str = "aria2c"  # aria2c, rsync, git, wget, api
    args: str = ""
    priority: int = 1
    max_concurrent: int = 1  # how many parallel downloads from this source
    expected_speed_mbps: float = 10.0
    # Circuit breaker state
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure: float = 0
    cooldown_secs: float = 300  # 5 min cooldown when circuit opens
    failure_threshold: int = 3  # open circuit after N failures
    # Tracking
    active_sessions: list = field(default_factory=list)
    bytes_downloaded: int = 0
    urls_completed: int = 0


# All knowledge sources
SOURCES = [
    DownloadSource(
        name="wikimedia",
        urls=[
            ("wikibooks", "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"),
            ("wikiversity", "https://dumps.wikimedia.org/enwikiversity/latest/enwikiversity-latest-pages-articles.xml.bz2"),
            ("wikisource", "https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2"),
            ("wikinews", "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"),
            ("wikivoyage", "https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2"),
        ],
        dest="/archive/knowledge",
        method="aria2c",
        args="-x 16 -s 16 --continue=true",
        priority=1,
        max_concurrent=3,
        expected_speed_mbps=30,
    ),
    DownloadSource(
        name="gutenberg",
        urls=[("gutenberg", "aleph.gutenberg.org::gutenberg")],
        dest="/archive/knowledge/gutenberg",
        method="rsync",
        args='-av --include="*.txt" --include="*/" --exclude="*"',
        priority=2,
        max_concurrent=4,
        expected_speed_mbps=1,
    ),
    DownloadSource(
        name="huggingface",
        urls=[
            ("c4-0", "allenai/c4 --include en/c4-train.00000-of-01024.json.gz"),
            ("c4-1", "allenai/c4 --include en/c4-train.00001-of-01024.json.gz"),
            ("c4-2", "allenai/c4 --include en/c4-train.00002-of-01024.json.gz"),
        ],
        dest="/archive/knowledge/c4",
        method="hf",
        priority=2,
        max_concurrent=2,
        expected_speed_mbps=20,
    ),
    DownloadSource(
        name="archive_org",
        urls=[
            ("ia-books", "https://archive.org/download/bookscorpus_shard_0/books_large_p1.txt"),
        ],
        dest="/archive/knowledge/ia-books",
        method="aria2c",
        args="-x 16 -s 16 --continue=true",
        priority=3,
        max_concurrent=1,
        expected_speed_mbps=10,
    ),
    DownloadSource(
        name="api_pali",
        urls=[("pali", "suttacentral")],
        dest="/archive/ethics-corpora/pali",
        method="api",
        priority=2,
        max_concurrent=1,
        expected_speed_mbps=0.1,
    ),
]


def get_net_bytes() -> int:
    try:
        with open("/proc/net/dev") as f:
            for line in f:
                if "eno1" in line:
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def get_cpu_temps() -> tuple:
    try:
        r = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5)
        temps = []
        for line in r.stdout.split("\n"):
            if "Package id" in line:
                temps.append(float(line.split("+")[1].split("\u00b0")[0]))
        return tuple(temps) if temps else (0, 0)
    except Exception:
        return (0, 0)


def tmux_alive(name: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True)
    return r.returncode == 0


def tmux_kill(name: str):
    subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)


def launch_download(source: DownloadSource, url_name: str, url: str) -> Optional[str]:
    session = f"dl-{url_name}"
    if tmux_alive(session):
        return session

    dest = os.path.join(source.dest, url_name) if source.method != "rsync" else source.dest
    os.makedirs(dest, exist_ok=True)

    if source.method == "aria2c":
        cmd = f"aria2c {source.args} --dir={dest} '{url}'"
    elif source.method == "rsync":
        cmd = f"rsync {source.args} '{url}' '{dest}/'"
    elif source.method == "git":
        if os.path.exists(os.path.join(dest, ".git")):
            cmd = f"cd '{dest}' && git pull"
        else:
            cmd = f"git clone --depth 1 '{url}' '{dest}'"
    elif source.method == "hf":
        parts = url.split(" ")
        dataset = parts[0]
        extra = " ".join(parts[1:]) if len(parts) > 1 else ""
        cmd = f"HF_HUB_ENABLE_HF_TRANSFER=1 /home/claude/env/bin/huggingface-cli download --repo-type dataset {dataset} {extra} --local-dir {dest}"
    elif source.method == "api":
        return None  # API downloads managed separately
    else:
        return None

    full_cmd = f"{cmd} 2>&1 | tee /tmp/dl_{url_name}.log; echo DL_DONE_{url_name}"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", session, full_cmd])
    log.info(f"  Launched: {session} ({source.method})")
    return session


def check_session_health(session: str, source: DownloadSource) -> bool:
    """Check if a download session is making progress."""
    log_path = f"/tmp/dl_{session.replace('dl-', '')}.log"
    if not os.path.exists(log_path):
        return True  # no log yet, give it time

    try:
        stat = os.stat(log_path)
        age = time.time() - stat.st_mtime
        # If log hasn't been written to in 120 seconds, it's stalled
        if age > 120 and stat.st_size > 0:
            return False
    except Exception:
        pass
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mbps", type=float, default=50)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--max-temp", type=float, default=82)
    args = parser.parse_args()

    log.info(f"Atlas Download Daemon (circuit breaker)")
    log.info(f"  Target: {args.target_mbps} MB/s, Max concurrent: {args.max_concurrent}")

    prev_bytes = get_net_bytes()
    prev_time = time.time()
    total_active = 0
    stall_count = 0

    while True:
        now = time.time()
        cur_bytes = get_net_bytes()
        dt = now - prev_time
        mbps = (cur_bytes - prev_bytes) / dt / 1_000_000 if dt > 0 else 0
        prev_bytes = cur_bytes
        prev_time = now

        temps = get_cpu_temps()
        max_temp = max(temps) if temps else 0

        # Thermal protection
        if max_temp > args.max_temp:
            log.warning(f"Thermal limit ({max_temp:.0f}C), pausing new downloads")
            time.sleep(30)
            continue

        # Count active downloads
        total_active = 0
        for source in SOURCES:
            source.active_sessions = [s for s in source.active_sessions if tmux_alive(s)]
            total_active += len(source.active_sessions)

        # Circuit breaker state transitions
        for source in SOURCES:
            if source.state == CircuitState.OPEN:
                if now - source.last_failure > source.cooldown_secs:
                    source.state = CircuitState.HALF_OPEN
                    source.failures = 0
                    log.info(f"  Circuit HALF_OPEN: {source.name} (cooldown expired)")

            # Check health of active sessions
            for session in list(source.active_sessions):
                if not tmux_alive(session):
                    # Completed or crashed
                    log_path = f"/tmp/dl_{session.replace('dl-', '')}.log"
                    if os.path.exists(log_path):
                        with open(log_path) as f:
                            content = f.read()
                        if f"DL_DONE_" in content:
                            source.urls_completed += 1
                            log.info(f"  Completed: {session}")
                            if source.state == CircuitState.HALF_OPEN:
                                source.state = CircuitState.CLOSED
                                log.info(f"  Circuit CLOSED: {source.name} (recovery confirmed)")
                        else:
                            source.failures += 1
                            source.last_failure = now
                            log.warning(f"  Failed: {session} (failures: {source.failures})")
                            if source.failures >= source.failure_threshold:
                                source.state = CircuitState.OPEN
                                log.error(f"  Circuit OPEN: {source.name} (threshold reached)")
                    source.active_sessions.remove(session)
                elif not check_session_health(session, source):
                    # Stalled
                    log.warning(f"  Stalled: {session}, killing")
                    tmux_kill(session)
                    source.active_sessions.remove(session)
                    source.failures += 1
                    source.last_failure = now

        # Launch new downloads if under target
        if mbps < args.target_mbps and total_active < args.max_concurrent:
            slots = args.max_concurrent - total_active
            for source in sorted(SOURCES, key=lambda s: s.priority):
                if slots <= 0:
                    break
                if source.state == CircuitState.OPEN:
                    continue

                available = len(source.active_sessions)
                can_launch = min(slots, source.max_concurrent - available)

                for url_name, url in source.urls:
                    if can_launch <= 0:
                        break
                    session = f"dl-{url_name}"
                    if tmux_alive(session) or session in source.active_sessions:
                        continue
                    # Check if already downloaded
                    dest = os.path.join(source.dest, url_name)
                    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                        continue

                    launched = launch_download(source, url_name, url)
                    if launched:
                        source.active_sessions.append(launched)
                        can_launch -= 1
                        slots -= 1
                        total_active += 1

        # Status
        circuit_status = {s.name: s.state.value for s in SOURCES}
        log.info(
            f"BW: {mbps:.1f} MB/s | Active: {total_active}/{args.max_concurrent} | "
            f"Temps: {'/'.join(f'{t:.0f}' for t in temps)}C | "
            f"Circuits: {circuit_status}"
        )

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
