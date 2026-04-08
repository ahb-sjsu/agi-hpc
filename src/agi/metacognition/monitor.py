# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Metacognition Monitor for AGI-HPC Phase 5.

Subscribes to ALL events via ``agi.>`` wildcard and tracks rolling
performance metrics across the entire cognitive architecture.

Tracked metrics:
    - Response latency percentiles (p50, p95, p99)
    - Token throughput (tok/s)
    - Hemisphere utilization ratio (LH vs RH)
    - Safety veto rate
    - Memory hit rate
    - User engagement signals (conversation length)

Publishes summary to:
    agi.meta.monitor.summary  (every 60 seconds)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric  # noqa: E402

# -----------------------------------------------------------------
# Metrics Store
# -----------------------------------------------------------------


@dataclass
class MetricsStore:
    """Rolling metrics store for the metacognition monitor.

    Keeps fixed-size deques of recent observations for efficient
    percentile calculation without unbounded memory growth.

    Attributes:
        window_size: Maximum observations to retain per metric.
        latencies_ms: Response latency observations.
        token_counts: Token counts per response.
        lh_requests: Count of LH requests seen.
        rh_requests: Count of RH requests seen.
        safety_checks: Count of safety checks performed.
        safety_vetoes: Count of safety vetoes (blocked responses).
        memory_queries: Count of memory queries.
        memory_hits: Count of memory queries with results.
        session_lengths: Conversation lengths per session.
        events_total: Total events observed.
    """

    window_size: int = 1000
    latencies_ms: Deque[float] = field(default_factory=deque)
    token_counts: Deque[int] = field(default_factory=deque)
    lh_requests: int = 0
    rh_requests: int = 0
    safety_checks: int = 0
    safety_vetoes: int = 0
    memory_queries: int = 0
    memory_hits: int = 0
    session_lengths: Dict[str, int] = field(default_factory=dict)
    events_total: int = 0
    _start_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.latencies_ms = deque(maxlen=self.window_size)
        self.token_counts = deque(maxlen=self.window_size)

    def record_latency(self, ms: float) -> None:
        """Record a response latency observation."""
        self.latencies_ms.append(ms)

    def record_tokens(self, count: int) -> None:
        """Record a token count observation."""
        self.token_counts.append(count)

    def percentile(self, values: Deque[float], pct: float) -> float:
        """Calculate a percentile from a deque of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def hemisphere_ratio(self) -> float:
        """LH:RH ratio. >1 means LH-dominant."""
        if self.rh_requests == 0:
            return float(self.lh_requests) if self.lh_requests > 0 else 1.0
        return self.lh_requests / self.rh_requests

    @property
    def safety_veto_rate(self) -> float:
        """Fraction of safety checks that resulted in vetoes."""
        if self.safety_checks == 0:
            return 0.0
        return self.safety_vetoes / self.safety_checks

    @property
    def memory_hit_rate(self) -> float:
        """Fraction of memory queries that returned results."""
        if self.memory_queries == 0:
            return 0.0
        return self.memory_hits / self.memory_queries

    @property
    def token_throughput(self) -> float:
        """Average tokens per second over the observation window."""
        if not self.token_counts or not self.latencies_ms:
            return 0.0
        total_tokens = sum(self.token_counts)
        total_time_s = sum(self.latencies_ms) / 1000.0
        if total_time_s == 0:
            return 0.0
        return total_tokens / total_time_s

    @property
    def avg_session_length(self) -> float:
        """Average conversation turns per session."""
        if not self.session_lengths:
            return 0.0
        return sum(self.session_lengths.values()) / len(self.session_lengths)

    def to_summary(self) -> Dict[str, Any]:
        """Produce a summary dict for publishing."""
        return {
            "uptime_seconds": round(self.uptime_seconds, 0),
            "events_total": self.events_total,
            "latency_p50_ms": round(self.percentile(self.latencies_ms, 50), 1),
            "latency_p95_ms": round(self.percentile(self.latencies_ms, 95), 1),
            "latency_p99_ms": round(self.percentile(self.latencies_ms, 99), 1),
            "token_throughput_tps": round(self.token_throughput, 1),
            "lh_requests": self.lh_requests,
            "rh_requests": self.rh_requests,
            "hemisphere_ratio": round(self.hemisphere_ratio, 2),
            "safety_checks": self.safety_checks,
            "safety_veto_rate": round(self.safety_veto_rate, 4),
            "memory_queries": self.memory_queries,
            "memory_hit_rate": round(self.memory_hit_rate, 4),
            "active_sessions": len(self.session_lengths),
            "avg_session_length": round(self.avg_session_length, 1),
        }


# -----------------------------------------------------------------
# Metacognition Monitor
# -----------------------------------------------------------------


class MetacognitionMonitor:
    """Monitors all events and tracks rolling performance metrics.

    Subscribes to ``agi.>`` (all events) and extracts telemetry
    signals from each event type. Publishes periodic summaries.

    Usage::

        monitor = MetacognitionMonitor()
        await monitor.start(fabric)
        # ... runs its own summary loop ...
        await monitor.stop()
    """

    def __init__(
        self,
        summary_interval: float = 60.0,
        window_size: int = 1000,
    ) -> None:
        self._metrics = MetricsStore(window_size=window_size)
        self._summary_interval = summary_interval
        self._fabric: Optional[NatsEventFabric] = None
        self._summary_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, fabric: NatsEventFabric) -> None:
        """Subscribe to all events and start the summary loop."""
        self._fabric = fabric

        # Subscribe to the wildcard to see ALL events
        await self._fabric.subscribe("agi.>", self._handle_event)

        self._running = True
        self._summary_task = asyncio.create_task(self._summary_loop())
        logger.info(
            "[meta-monitor] started, publishing every %.0fs", self._summary_interval
        )

    async def stop(self) -> None:
        """Cancel the summary loop."""
        self._running = False
        if self._summary_task:
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
        logger.info("[meta-monitor] stopped")

    async def _handle_event(self, event: Event) -> None:
        """Extract metrics from any passing event."""
        self._metrics.events_total += 1
        etype = event.type or ""
        payload = event.payload or {}

        # Track latency from any response event
        if "latency_ms" in payload:
            self._metrics.record_latency(payload["latency_ms"])

        # Track tokens
        if "tokens_used" in payload:
            self._metrics.record_tokens(payload["tokens_used"])

        # Track hemisphere requests
        if etype.startswith("lh.request.") or "agi.lh.request." in etype:
            self._metrics.lh_requests += 1
        elif etype.startswith("rh.request.") or "agi.rh.request." in etype:
            self._metrics.rh_requests += 1

        # Track safety checks/vetoes
        if "safety" in etype:
            self._metrics.safety_checks += 1
            if payload.get("vetoed") or payload.get("safe") is False:
                self._metrics.safety_vetoes += 1

        # Track memory queries
        if "memory.result." in etype or "memory.query." in etype:
            self._metrics.memory_queries += 1
            count = payload.get("count", 0)
            if count > 0:
                self._metrics.memory_hits += 1

        # Track session engagement
        session_id = payload.get("session_id")
        if session_id:
            self._metrics.session_lengths[session_id] = (
                self._metrics.session_lengths.get(session_id, 0) + 1
            )

    async def _summary_loop(self) -> None:
        """Periodically publish a metrics summary."""
        while self._running:
            await asyncio.sleep(self._summary_interval)
            if not self._running:
                break

            summary = self._metrics.to_summary()
            summary_event = Event.create(
                source="metacognition",
                event_type="meta.monitor.summary",
                payload=summary,
            )
            try:
                await self._fabric.publish("agi.meta.monitor.summary", summary_event)
                logger.info(
                    "[meta-monitor] summary: events=%d latency_p50=%.0fms "
                    "tps=%.1f lh/rh=%.2f veto_rate=%.4f",
                    summary["events_total"],
                    summary["latency_p50_ms"],
                    summary["token_throughput_tps"],
                    summary["hemisphere_ratio"],
                    summary["safety_veto_rate"],
                )
            except Exception:
                logger.exception("[meta-monitor] failed to publish summary")

    @property
    def metrics(self) -> MetricsStore:
        """Return current metrics store."""
        return self._metrics
