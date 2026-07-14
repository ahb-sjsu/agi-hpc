# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Resource governor + rate limits for the Director's dispatch step.

Phase A never acts, so it needs no governor. Phase B does: before any L2
dispatch the Director must confirm it won't push Atlas past its thermal
cap, won't compete with a GPU-1 maintenance loan, and won't exceed the
charter's dispatch rate limits. A goal that fails the governor is
``blocked`` (retried next cycle), not dropped.

The Director itself runs almost no compute — its reasoning is NRP shared-
pool API. This governor bounds the *dispatched* work (Primer teaching,
dreaming, etc.).

All environment probes are injectable so the whole thing is unit-testable
without a live Atlas: pass a ``temp_reader`` / ``gpu1_maint`` /
``rate_state`` and no syscalls happen.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("director.govern")

DEFAULT_RATE_PATH = Path("/archive/erebus/dispatch_rate.json")
DEFAULT_GPU1_MAINT = Path("/archive/neurogolf/.gpu1_maint")


@dataclass
class Budget:
    thermal_c: float = 82.0          # CPU package temp cap (Atlas thermal rule)
    max_per_cycle: int = 3
    max_per_day: int = 20


@dataclass
class Decision:
    ok: bool
    reason: str = ""


def read_cpu_package_temp() -> float | None:
    """Best-effort CPU package temperature via ``sensors``. None on failure —
    the governor treats an unknown temperature as *unsafe* (deny)."""
    try:
        out = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=4
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    hottest = None
    for line in out.splitlines():
        if "Package id" in line and "+" in line:
            try:
                val = float(line.split("+")[1].split("°")[0])
                hottest = val if hottest is None else max(hottest, val)
            except (IndexError, ValueError):
                continue
    return hottest


class Governor:
    """Gates dispatch on thermal, GPU-1 loan, and rate limits."""

    def __init__(
        self,
        budget: Budget | None = None,
        *,
        temp_reader=read_cpu_package_temp,
        gpu1_maint: Path = DEFAULT_GPU1_MAINT,
        rate_path: Path = DEFAULT_RATE_PATH,
        now=time.time,
    ) -> None:
        self.budget = budget or Budget()
        self._temp = temp_reader
        self._gpu1_maint = gpu1_maint
        self._rate_path = rate_path
        self._now = now

    def _load_rate(self) -> dict:
        try:
            return json.loads(self._rate_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {"day": "", "day_count": 0, "cycle_count": 0}

    def _save_rate(self, state: dict) -> None:
        try:
            from agi.common.atomic_write import atomic_write_text

            atomic_write_text(self._rate_path, json.dumps(state))
        except Exception as e:  # noqa: BLE001
            log.warning("rate save failed: %s", e)

    def start_cycle(self) -> None:
        """Reset the per-cycle counter (call once at the top of a dispatch phase)."""
        state = self._load_rate()
        state["cycle_count"] = 0
        self._save_rate(state)

    def can_dispatch(self) -> Decision:
        """Check thermal + GPU-1 loan + rate limits for the next dispatch."""
        if self._gpu1_maint.exists():
            return Decision(False, "GPU 1 on maintenance loan")
        temp = self._temp()
        if temp is None:
            return Decision(False, "CPU temperature unknown (deny-by-default)")
        if temp >= self.budget.thermal_c:
            cap = self.budget.thermal_c
            return Decision(False, f"CPU {temp:.0f}°C ≥ thermal cap {cap:.0f}°C")
        state = self._load_rate()
        today = time.strftime("%Y-%m-%d", time.gmtime(self._now()))
        if state.get("day") != today:
            state = {
                "day": today, "day_count": 0,
                "cycle_count": state.get("cycle_count", 0),
            }
            self._save_rate(state)
        if state.get("cycle_count", 0) >= self.budget.max_per_cycle:
            lim = self.budget.max_per_cycle
            return Decision(False, f"per-cycle dispatch limit ({lim}) reached")
        if state.get("day_count", 0) >= self.budget.max_per_day:
            lim = self.budget.max_per_day
            return Decision(False, f"per-day dispatch limit ({lim}) reached")
        return Decision(True)

    def record_dispatch(self) -> None:
        """Increment the rate counters after a successful dispatch."""
        state = self._load_rate()
        today = time.strftime("%Y-%m-%d", time.gmtime(self._now()))
        if state.get("day") != today:
            state = {"day": today, "day_count": 0, "cycle_count": 0}
        state["day_count"] = state.get("day_count", 0) + 1
        state["cycle_count"] = state.get("cycle_count", 0) + 1
        self._save_rate(state)
