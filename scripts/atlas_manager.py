#!/usr/bin/env python3
"""
Atlas AI System Manager — start | stop | status | restart | watch

Manages all Atlas AI services as a unified system:
  - Superego (Gemma 4 31B, GPU 0)
  - Id (Qwen 3 32B, GPU 1)
  - Divine Council (7 agents, 1x Gemma 4 26B-A4B MoE, --parallel 8, CPU)
  - RAG Server (Flask + embeddings)
  - Telemetry Server
  - NATS JetStream (if installed)

Usage:
    python scripts/atlas_manager.py start       # Start all services
    python scripts/atlas_manager.py stop        # Stop all services
    python scripts/atlas_manager.py restart     # Stop + start
    python scripts/atlas_manager.py status      # Health check all
    python scripts/atlas_manager.py watch       # Monitor loop (auto-restart)
    python scripts/atlas_manager.py maint       # Stop LLMs, keep infra
    python scripts/atlas_manager.py resume      # Start LLMs after maint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [atlas-mgr] %(message)s",
)
log = logging.getLogger("atlas-mgr")

# ─── Configuration ────────────────────────────────────────────

ATLAS_HOME = os.environ.get("ATLAS_HOME", "/home/claude/agi-hpc")
MODELS_DIR = os.environ.get("MODELS_DIR", "/home/claude/models")
LLAMA_BIN = os.environ.get("LLAMA_BIN", "/home/claude/llama.cpp/build/bin/llama-server")
PYTHON = os.environ.get("ATLAS_PYTHON", "/home/claude/env/bin/python3")
LOG_DIR = "/tmp"

WATCH_INTERVAL = 30  # seconds between health checks
MAX_RESTART_ATTEMPTS = 3
RESTART_COOLDOWN = 60  # seconds between restart attempts per service


@dataclass
class ServiceConfig:
    """Configuration for one Atlas service."""

    name: str  # tmux session name
    description: str
    port: int
    health_path: str  # HTTP path for health check
    start_cmd: str
    category: str  # "llm", "council", "infra"
    gpu: int | None = None  # GPU index, None = CPU
    depends_on: list[str] | None = None


# ─── Service definitions ──────────────────────────────────────

SERVICES: list[ServiceConfig] = [
    # GPU LLMs — auto-detect model files
    ServiceConfig(
        name="spock",
        description="Superego: Gemma 4 31B",
        port=8080,
        health_path="/health",
        start_cmd=(
            f"CUDA_VISIBLE_DEVICES=0 {LLAMA_BIN} "
            f"--model $(ls {MODELS_DIR}/gemma-4-31b* {MODELS_DIR}/Gemma-4-31B* 2>/dev/null | head -1) "
            f"--host 0.0.0.0 --port 8080 --ctx-size 8192 "
            f"--n-gpu-layers 99"
        ),
        category="llm",
        gpu=0,
    ),
    ServiceConfig(
        name="kirk",
        description="Id: Qwen 3 32B",
        port=8082,
        health_path="/health",
        start_cmd=(
            f"CUDA_VISIBLE_DEVICES=1 {LLAMA_BIN} "
            f"--model $(ls {MODELS_DIR}/qwen* {MODELS_DIR}/Qwen* 2>/dev/null | head -1) "
            f"--host 0.0.0.0 --port 8082 --ctx-size 8192 "
            f"--n-gpu-layers 99"
        ),
        category="llm",
        gpu=1,
    ),
    # Divine Council — single server, 7 agents, --parallel 8
    # All council members (Judge, Advocate, Synthesizer, Ethicist, Historian,
    # Futurist, Pragmatist) share this one process. ~18GB total.
    ServiceConfig(
        name="ego",
        description="Divine Council: 7 agents, 26B-A4B MoE, --parallel 8",
        port=8084,
        health_path="/health",
        start_cmd=(
            f"CUDA_VISIBLE_DEVICES= {LLAMA_BIN} "
            f"--model {MODELS_DIR}/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf "
            f"--host 127.0.0.1 --port 8084 --ctx-size 4096 "
            f"--threads 24 --parallel 8 --n-gpu-layers 0 --cont-batching"
        ),
        category="council",
    ),
    # Infrastructure
    ServiceConfig(
        name="rag",
        description="RAG Server (Flask)",
        port=8081,
        health_path="/api/search-status",
        start_cmd=(f"cd {ATLAS_HOME} && {PYTHON} atlas-rag-server.py"),
        category="infra",
        depends_on=["spock", "ego"],
    ),
    ServiceConfig(
        name="telemetry",
        description="Telemetry Server",
        port=8085,
        health_path="/api/telemetry",
        start_cmd=(f"cd {ATLAS_HOME} && {PYTHON} scripts/telemetry_server.py"),
        category="infra",
    ),
]


# ─── Helpers ──────────────────────────────────────────────────


def _run(cmd: str, timeout: int = 5) -> str:
    """Run a shell command and return stdout."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _tmux_exists(name: str) -> bool:
    """Check if a tmux session exists."""
    r = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        timeout=3,
    )
    return r.returncode == 0


def _tmux_start(name: str, cmd: str) -> bool:
    """Start a command in a new tmux session."""
    log_file = f"{LOG_DIR}/{name}.log"
    full_cmd = f'tmux new-session -d -s {name} "{cmd} 2>&1 | tee {log_file}"'
    r = subprocess.run(full_cmd, shell=True, capture_output=True, timeout=5)
    return r.returncode == 0


def _tmux_kill(name: str) -> bool:
    """Kill a tmux session."""
    r = subprocess.run(
        ["tmux", "kill-session", "-t", name],
        capture_output=True,
        timeout=5,
    )
    return r.returncode == 0


def _check_port(port: int, path: str, timeout: int = 3) -> bool:
    """Check if a service is responding on a port."""
    try:
        import urllib.request

        url = f"http://localhost:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status in (200, 404)  # 404 = server up, no /health route
    except Exception:
        return False


def _port_in_use(port: int) -> bool:
    """Check if any process is listening on a port (even without HTTP)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("localhost", port))
        s.close()
        return True
    except Exception:
        return False


def _kill_port(port: int) -> None:
    """Kill any process listening on a port."""
    out = _run(f"ss -tlnp | grep :{port}")
    if out:
        import re

        pids = re.findall(r"pid=(\d+)", out)
        for pid in pids:
            _run(f"kill -9 {pid}")
            log.info("Killed PID %s on port %d", pid, port)


# ─── Core operations ─────────────────────────────────────────


def status(services: list[ServiceConfig] | None = None) -> dict:
    """Check health of all services."""
    if services is None:
        services = SERVICES

    results = {}
    for svc in services:
        tmux_up = _tmux_exists(svc.name)
        port_up = _check_port(svc.port, svc.health_path)
        port_listening = _port_in_use(svc.port) if not port_up else True

        if port_up:
            state = "running"
        elif tmux_up and port_listening:
            state = "loading"
        elif port_listening:
            state = "orphaned"  # port in use but no tmux session
        elif tmux_up:
            state = "loading"
        else:
            state = "stopped"

        results[svc.name] = {
            "description": svc.description,
            "port": svc.port,
            "state": state,
            "category": svc.category,
        }

    return results


def start(
    services: list[ServiceConfig] | None = None,
    categories: set[str] | None = None,
) -> None:
    """Start services (respecting dependencies)."""
    if services is None:
        services = SERVICES
    if categories:
        services = [s for s in services if s.category in categories]

    # Sort by dependency: infra last (depends on LLMs)
    order = {"llm": 0, "council": 1, "infra": 2}
    services = sorted(services, key=lambda s: order.get(s.category, 1))

    for svc in services:
        if _tmux_exists(svc.name):
            log.info("%-15s already running", svc.name)
            continue

        # Check port isn't occupied by another process
        if _check_port(svc.port, svc.health_path):
            log.warning(
                "%-15s port %d already in use — killing",
                svc.name,
                svc.port,
            )
            _kill_port(svc.port)
            time.sleep(2)

        if _tmux_start(svc.name, svc.start_cmd):
            log.info(
                "%-15s started (port %d) [%s]",
                svc.name,
                svc.port,
                svc.description,
            )
        else:
            log.error("%-15s FAILED to start", svc.name)

    # Wait for services to load
    log.info("Waiting for services to initialize...")
    time.sleep(5)

    # Verify
    results = status(services)
    running = sum(1 for v in results.values() if v["state"] != "stopped")
    total = len(results)
    log.info("%d/%d services running", running, total)


def stop(
    services: list[ServiceConfig] | None = None,
    categories: set[str] | None = None,
) -> None:
    """Stop services (infra first, then LLMs)."""
    if services is None:
        services = SERVICES
    if categories:
        services = [s for s in services if s.category in categories]

    # Reverse order: infra first, then council, then LLMs
    order = {"infra": 0, "council": 1, "llm": 2}
    services = sorted(services, key=lambda s: order.get(s.category, 1))

    for svc in services:
        if _tmux_exists(svc.name):
            _tmux_kill(svc.name)
            log.info("%-15s stopped", svc.name)
        else:
            log.info("%-15s already stopped", svc.name)

    # Kill any orphaned GPU processes
    gpu_pids = _run("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    if gpu_pids:
        for pid in gpu_pids.strip().split("\n"):
            pid = pid.strip()
            if pid:
                _run(f"kill -9 {pid}")
                log.info("Killed orphaned GPU process %s", pid)

    time.sleep(3)
    log.info("All services stopped")


def restart(
    services: list[ServiceConfig] | None = None,
    categories: set[str] | None = None,
) -> None:
    """Stop then start all services."""
    stop(services, categories)
    time.sleep(3)
    start(services, categories)


def maint() -> None:
    """Enter maintenance mode: stop LLMs + council, keep infra."""
    log.info("=== ENTERING MAINTENANCE MODE ===")
    stop(categories={"llm", "council"})
    log.info("LLMs stopped. Infrastructure (RAG, telemetry) still running.")
    log.info("GPUs are free for other work.")
    log.info("Run 'atlas_manager.py resume' to restart LLMs.")


def resume() -> None:
    """Resume from maintenance: start LLMs + council."""
    log.info("=== RESUMING FROM MAINTENANCE ===")

    # Restart infra too (RAG needs LLMs)
    stop(categories={"infra"})
    time.sleep(2)
    start(categories={"llm", "council"})
    time.sleep(10)  # Wait for models to load
    start(categories={"infra"})


def watch(interval: int = WATCH_INTERVAL) -> None:
    """Monitor loop: check all services, restart any that are down."""
    log.info("=== WATCHDOG STARTED (interval=%ds) ===", interval)

    restart_counts: dict[str, int] = {}
    last_restart: dict[str, float] = {}

    def handle_signal(signum, frame):
        log.info("Watchdog stopped (signal %d)", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        results = status()
        down = [name for name, info in results.items() if info["state"] == "stopped"]

        if down:
            log.warning("DOWN: %s", ", ".join(down))

            for name in down:
                # Check cooldown
                now = time.time()
                if name in last_restart:
                    elapsed = now - last_restart[name]
                    if elapsed < RESTART_COOLDOWN:
                        log.info(
                            "%-15s cooldown (%ds remaining)",
                            name,
                            int(RESTART_COOLDOWN - elapsed),
                        )
                        continue

                # Check max attempts
                count = restart_counts.get(name, 0)
                if count >= MAX_RESTART_ATTEMPTS:
                    log.error(
                        "%-15s exceeded max restarts (%d) — giving up",
                        name,
                        MAX_RESTART_ATTEMPTS,
                    )
                    continue

                # Find the service config
                svc = next((s for s in SERVICES if s.name == name), None)
                if svc is None:
                    continue

                # Kill port if occupied
                _kill_port(svc.port)
                time.sleep(1)

                # Restart
                if _tmux_start(svc.name, svc.start_cmd):
                    log.info(
                        "%-15s RESTARTED (attempt %d/%d)",
                        name,
                        count + 1,
                        MAX_RESTART_ATTEMPTS,
                    )
                    restart_counts[name] = count + 1
                    last_restart[name] = now
                else:
                    log.error("%-15s restart FAILED", name)
        else:
            # All healthy — reset restart counts
            restart_counts.clear()

        time.sleep(interval)


# ─── CLI ──────────────────────────────────────────────────────


def print_status() -> None:
    """Pretty-print service status."""
    results = status()
    print(f"\n{'Service':>15s}  {'Port':>5s}  {'State':>8s}  Description")
    print("-" * 65)
    for name, info in results.items():
        state = info["state"]
        color = (
            "\033[32m"
            if state == "running"
            else "\033[33m" if state == "loading" else "\033[31m"
        )
        reset = "\033[0m"
        print(
            f"{name:>15s}  {info['port']:>5d}  "
            f"{color}{state:>8s}{reset}  {info['description']}"
        )

    running = sum(1 for v in results.values() if v["state"] == "running")
    print(f"\n{running}/{len(results)} services running")

    # GPU usage
    gpu_out = _run(
        "nvidia-smi --query-gpu=index,memory.used,memory.total " "--format=csv,noheader"
    )
    if gpu_out:
        print("\nGPU memory:")
        for line in gpu_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                print(f"  GPU {parts[0]}: {parts[1]} / {parts[2]}")

    # Temps
    temp_out = _run("sensors | grep Package")
    if temp_out:
        print("\nTemperatures:")
        for line in temp_out.split("\n"):
            print(f"  {line.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="atlas_manager",
        description="Atlas AI System Manager",
    )
    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "watch",
            "maint",
            "resume",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=WATCH_INTERVAL,
        help=f"Watch interval in seconds (default: {WATCH_INTERVAL})",
    )

    args = parser.parse_args()

    if args.command == "start":
        start()
        print_status()
    elif args.command == "stop":
        stop()
    elif args.command == "restart":
        restart()
        print_status()
    elif args.command == "status":
        print_status()
    elif args.command == "watch":
        watch(args.interval)
    elif args.command == "maint":
        maint()
    elif args.command == "resume":
        resume()
        print_status()


if __name__ == "__main__":
    main()
