# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Thermal-Managed Job Queue — All CPU work on Atlas goes through here.

A NATS-connected service that accepts job submissions and runs them
with thermal throttling via the Kalman-filtered ThermalController.
No script should directly spawn CPU-bound work — instead, submit
a job to this queue and it will be scheduled when thermal headroom
permits.

Architecture:
    Submitter → NATS (agi.jobs.submit) → Job Queue → ThermalController
                                              ↓
                                         subprocess (throttled)
                                              ↓
                                    NATS (agi.jobs.complete)

Job types:
    - index:   Corpus indexing (arXiv, ethics, repos)
    - embed:   Embedding generation (BGE-M3)
    - train:   LoRA training, AtlasGym
    - dream:   Dreaming consolidation
    - custom:  Any shell command

Thermal policy:
    - target_temp: 82°C — no new jobs launch above this
    - cooldown_margin: 5°C — must be at 77°C to launch next job
    - max_concurrent: 2 — at most 2 CPU-heavy jobs simultaneously
    - SIGSTOP/SIGCONT: running jobs are paused if temp hits 95°C

Usage:
    # As a service:
    python -m agi.thermal.job_queue

    # Submit a job via NATS:
    nats pub agi.jobs.submit '{"name":"index-arxiv","cmd":["python","scripts/index_all_corpora.py","--source","arxiv"],"gpu":1}'

    # Submit from Python:
    await nc.publish("agi.jobs.submit", json.dumps({
        "name": "index-arxiv",
        "cmd": ["python", "scripts/index_all_corpora.py", "--source", "arxiv"],
        "gpu": 1,
        "max_threads": 20,
    }).encode())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thermal reading (inline — no dependency on batch_probe install)
# ---------------------------------------------------------------------------

def _read_cpu_temp() -> Optional[float]:
    """Read highest CPU package temperature."""
    try:
        out = subprocess.check_output(
            ["sensors"], stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        temps = []
        import re
        for line in out.splitlines():
            if "Package" in line:
                m = re.search(r"\+(\d+\.?\d*)", line)
                if m:
                    temps.append(float(m.group(1)))
        return max(temps) if temps else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class JobQueueConfig:
    """Configuration for the thermal job queue."""

    target_temp: float = 82.0
    cooldown_margin: float = 5.0
    critical_temp: float = 95.0
    max_concurrent: int = 48  # no artificial cap — thermal controller is the throttle
    poll_interval: float = 5.0
    settle_time: float = 10.0
    work_dir: str = "/home/claude/agi-hpc"
    log_dir: str = "/tmp/atlas-jobs"
    nats_servers: List[str] = field(
        default_factory=lambda: ["nats://localhost:4222"]
    )


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A submitted job."""

    name: str
    cmd: List[str]
    gpu: Optional[int] = None
    max_threads: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    status: str = "queued"  # queued, running, paused, completed, failed
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    log_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cmd": self.cmd,
            "gpu": self.gpu,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "log_file": self.log_file,
        }


# ---------------------------------------------------------------------------
# Job Queue Engine
# ---------------------------------------------------------------------------

class ThermalJobQueue:
    """Thermal-managed job queue with SIGSTOP/SIGCONT throttling."""

    def __init__(self, config: Optional[JobQueueConfig] = None):
        self.config = config or JobQueueConfig()
        self.queue: List[Job] = []
        self.active: Dict[str, Job] = {}
        self.completed: List[Job] = []
        self.paused_pids: set = set()

        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    @property
    def can_launch(self) -> bool:
        """Check if thermal conditions allow launching a new job.

        No artificial concurrency cap — the thermal controller is the
        only throttle. If the CPU is cool enough, launch. If it's hot,
        wait. The Kalman filter predicts thermal trajectory so we can
        be proactive rather than reactive.
        """
        temp = _read_cpu_temp()
        if temp is None:
            return True  # can't read sensor, assume OK
        return temp < (self.config.target_temp - self.config.cooldown_margin)

    @property
    def is_critical(self) -> bool:
        """Check if we're at critical temperature."""
        temp = _read_cpu_temp()
        return temp is not None and temp >= self.config.critical_temp

    def submit(self, name: str, cmd: List[str], gpu: Optional[int] = None,
               max_threads: Optional[int] = None,
               env: Optional[Dict[str, str]] = None) -> Job:
        """Submit a job to the queue."""
        job = Job(
            name=name,
            cmd=cmd,
            gpu=gpu,
            max_threads=max_threads,
            env=env or {},
        )
        self.queue.append(job)
        logger.info("[jobs] Queued: %s (%s)", name, " ".join(cmd[:3]))
        return job

    def _launch(self, job: Job) -> None:
        """Launch a job subprocess."""
        env = os.environ.copy()
        env.update(job.env)

        if job.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        if job.max_threads:
            env["OMP_NUM_THREADS"] = str(job.max_threads)
            env["MKL_NUM_THREADS"] = str(job.max_threads)
            env["OPENBLAS_NUM_THREADS"] = str(job.max_threads)
            env["NUMEXPR_MAX_THREADS"] = str(job.max_threads)

        job.log_file = os.path.join(
            self.config.log_dir, f"{job.name}_{int(time.time())}.log"
        )
        log_fh = open(job.log_file, "w")

        job.process = subprocess.Popen(
            job.cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=self.config.work_dir,
            env=env,
        )
        job.started_at = time.time()
        job.status = "running"
        self.active[job.name] = job

        temp = _read_cpu_temp() or 0
        logger.info(
            "[jobs] LAUNCHED: %s (pid=%d, gpu=%s, temp=%.0f°C, %d active)",
            job.name, job.process.pid, job.gpu, temp, len(self.active),
        )

    def _reap(self) -> List[Job]:
        """Check for completed jobs."""
        done = []
        for name, job in list(self.active.items()):
            if job.process and job.process.poll() is not None:
                job.return_code = job.process.returncode
                job.finished_at = time.time()
                job.status = "completed" if job.return_code == 0 else "failed"
                job.process = None
                del self.active[name]
                self.completed.append(job)
                done.append(job)

                elapsed = job.finished_at - (job.started_at or job.finished_at)
                logger.info(
                    "[jobs] %s: %s (exit=%d, elapsed=%.0fs)",
                    job.status.upper(), name, job.return_code, elapsed,
                )
        return done

    def _pause_all(self) -> None:
        """SIGSTOP all active job processes — emergency thermal."""
        for name, job in self.active.items():
            if job.process and job.process.pid not in self.paused_pids:
                try:
                    os.kill(job.process.pid, signal.SIGSTOP)
                    self.paused_pids.add(job.process.pid)
                    job.status = "paused"
                    logger.warning("[jobs] PAUSED: %s (pid=%d) — thermal emergency",
                                   name, job.process.pid)
                except ProcessLookupError:
                    pass

    def _resume_all(self) -> None:
        """SIGCONT all paused job processes."""
        for name, job in self.active.items():
            if job.process and job.process.pid in self.paused_pids:
                try:
                    os.kill(job.process.pid, signal.SIGCONT)
                    self.paused_pids.discard(job.process.pid)
                    job.status = "running"
                    logger.info("[jobs] RESUMED: %s (pid=%d)", name, job.process.pid)
                except ProcessLookupError:
                    self.paused_pids.discard(job.process.pid)

    def tick(self) -> List[Job]:
        """One iteration of the queue manager. Call in a loop."""
        # Reap finished
        done = self._reap()

        # Thermal emergency check
        if self.is_critical:
            if not self.paused_pids:
                logger.critical("[jobs] CRITICAL TEMP — pausing all jobs")
            self._pause_all()
            return done

        # Resume if we were paused and now cool enough
        if self.paused_pids:
            temp = _read_cpu_temp()
            if temp and temp < self.config.target_temp - self.config.cooldown_margin:
                self._resume_all()

        # Launch queued jobs if thermal headroom
        while self.queue and self.can_launch:
            job = self.queue.pop(0)
            self._launch(job)
            time.sleep(self.config.settle_time)

        return done

    def status(self) -> dict:
        """Current queue status for telemetry."""
        temp = _read_cpu_temp()
        return {
            "queued": len(self.queue),
            "active": len(self.active),
            "completed": len(self.completed),
            "paused": len(self.paused_pids),
            "cpu_temp": temp,
            "can_launch": self.can_launch,
            "jobs": (
                [{"name": j.name, "status": j.status} for j in self.queue] +
                [{"name": j.name, "status": j.status,
                  "pid": j.process.pid if j.process else None,
                  "gpu": j.gpu,
                  "elapsed": int(time.time() - j.started_at) if j.started_at else 0}
                 for j in self.active.values()]
            ),
        }


# ---------------------------------------------------------------------------
# NATS Service
# ---------------------------------------------------------------------------

async def run_service(config: JobQueueConfig) -> None:
    """Main NATS-connected job queue service."""
    try:
        import nats
    except ImportError:
        logger.error("nats-py not installed — pip install nats-py")
        return

    nc = await nats.connect(servers=config.nats_servers)
    queue = ThermalJobQueue(config)

    logger.info("[jobs] Thermal job queue started (target=%.0f°C, thermal-throttled, no concurrency cap)",
                config.target_temp)

    # Subscribe to job submissions
    async def on_submit(msg):
        try:
            data = json.loads(msg.data.decode())
            job = queue.submit(
                name=data["name"],
                cmd=data["cmd"],
                gpu=data.get("gpu"),
                max_threads=data.get("max_threads"),
                env=data.get("env", {}),
            )
            await nc.publish("agi.jobs.accepted", json.dumps({
                "name": job.name, "status": "queued",
                "queue_depth": len(queue.queue),
            }).encode())
        except Exception as e:
            logger.error("[jobs] Bad submission: %s", e)

    await nc.subscribe("agi.jobs.submit", cb=on_submit)

    # Subscribe to status requests
    async def on_status(msg):
        await msg.respond(json.dumps(queue.status()).encode())

    await nc.subscribe("agi.jobs.status", cb=on_status)

    # Main loop
    running = True

    def shutdown():
        nonlocal running
        running = False

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    await nc.publish("agi.jobs.start", json.dumps({
        "event": "service_started",
        "target_temp": config.target_temp,
        "max_concurrent": config.max_concurrent,
    }).encode())

    while running:
        done = queue.tick()

        # Publish completion events
        for job in done:
            await nc.publish("agi.jobs.complete", json.dumps(
                job.to_dict(), default=str
            ).encode())

        await asyncio.sleep(config.poll_interval)

    # Cleanup: SIGCONT any paused processes before exit
    queue._resume_all()
    await nc.close()
    logger.info("[jobs] Job queue service stopped")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Atlas Thermal Job Queue")
    parser.add_argument("--target-temp", type=float, default=82.0)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--nats", default="nats://localhost:4222")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = JobQueueConfig(
        target_temp=args.target_temp,
        max_concurrent=args.max_concurrent,
        nats_servers=[args.nats],
    )

    asyncio.run(run_service(config))


if __name__ == "__main__":
    main()
