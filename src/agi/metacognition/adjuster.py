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
Metacognition Adjuster for AGI-HPC Phase 5.

Based on metrics from the Monitor, publishes adjustment events to
tune system behaviour in real time.

Adjustment rules:
    - High latency    -> reduce max_tokens
    - High veto rate  -> flag for review
    - Unbalanced hemisphere ratio -> adjust routing thresholds

Publishes to:
    agi.meta.adjust.tokens
    agi.meta.adjust.safety
    agi.meta.adjust.routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric  # noqa: E402

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class AdjusterConfig:
    """Configuration for the Metacognition Adjuster.

    Attributes:
        check_interval: Seconds between adjustment checks.
        latency_threshold_p95_ms: p95 latency above which we reduce tokens.
        min_max_tokens: Floor for max_tokens reduction.
        max_tokens_step: How much to reduce max_tokens each step.
        veto_rate_threshold: Veto rate above which we flag for review.
        hemisphere_ratio_min: Minimum LH:RH ratio (below = RH-heavy).
        hemisphere_ratio_max: Maximum LH:RH ratio (above = LH-heavy).
    """

    check_interval: float = 60.0
    latency_threshold_p95_ms: float = 30000.0
    min_max_tokens: int = 512
    max_tokens_step: int = 256
    veto_rate_threshold: float = 0.1
    hemisphere_ratio_min: float = 0.5
    hemisphere_ratio_max: float = 3.0


# -----------------------------------------------------------------
# Metacognition Adjuster
# -----------------------------------------------------------------


class MetacognitionAdjuster:
    """Publishes adjustment events based on system metrics.

    Listens to ``agi.meta.monitor.summary`` events from the Monitor
    and applies heuristic rules to tune system parameters.

    Usage::

        adjuster = MetacognitionAdjuster()
        await adjuster.start(fabric)
        # ... runs alongside monitor + reflector ...
        await adjuster.stop()
    """

    def __init__(
        self,
        config: Optional[AdjusterConfig] = None,
    ) -> None:
        self._config = config or AdjusterConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._running = False
        self._adjustments_made: int = 0
        self._current_max_tokens: int = 4096

    async def start(self, fabric: NatsEventFabric) -> None:
        """Subscribe to summary events."""
        self._fabric = fabric
        await self._fabric.subscribe("agi.meta.monitor.summary", self._on_summary)
        self._running = True
        logger.info("[meta-adjuster] started")

    async def stop(self) -> None:
        """Stop the adjuster."""
        self._running = False
        logger.info("[meta-adjuster] stopped")

    async def _on_summary(self, event: Event) -> None:
        """Evaluate metrics summary and publish adjustments."""
        payload = event.payload or {}

        await self._check_latency(payload)
        await self._check_safety(payload)
        await self._check_hemisphere_balance(payload)

    async def _check_latency(self, summary: Dict[str, Any]) -> None:
        """If p95 latency is too high, publish a token reduction event."""
        p95 = summary.get("latency_p95_ms", 0.0)
        if p95 <= self._config.latency_threshold_p95_ms:
            return

        new_max = max(
            self._current_max_tokens - self._config.max_tokens_step,
            self._config.min_max_tokens,
        )
        if new_max >= self._current_max_tokens:
            return  # already at floor

        self._current_max_tokens = new_max
        self._adjustments_made += 1

        adjust_event = Event.create(
            source="metacognition",
            event_type="meta.adjust.tokens",
            payload={
                "reason": "high_latency",
                "p95_latency_ms": round(p95, 1),
                "threshold_ms": self._config.latency_threshold_p95_ms,
                "new_max_tokens": new_max,
                "adjustment_number": self._adjustments_made,
            },
        )
        await self._fabric.publish("agi.meta.adjust.tokens", adjust_event)
        logger.info(
            "[meta-adjuster] reducing max_tokens to %d (p95=%.0fms)",
            new_max,
            p95,
        )

    async def _check_safety(self, summary: Dict[str, Any]) -> None:
        """If safety veto rate is too high, flag for review."""
        veto_rate = summary.get("safety_veto_rate", 0.0)
        checks = summary.get("safety_checks", 0)

        if veto_rate <= self._config.veto_rate_threshold or checks < 5:
            return

        self._adjustments_made += 1

        adjust_event = Event.create(
            source="metacognition",
            event_type="meta.adjust.safety",
            payload={
                "reason": "high_veto_rate",
                "veto_rate": round(veto_rate, 4),
                "threshold": self._config.veto_rate_threshold,
                "safety_checks": checks,
                "action": "flag_for_review",
                "adjustment_number": self._adjustments_made,
            },
        )
        await self._fabric.publish("agi.meta.adjust.safety", adjust_event)
        logger.warning(
            "[meta-adjuster] high veto rate %.2f%% (threshold=%.2f%%)",
            veto_rate * 100,
            self._config.veto_rate_threshold * 100,
        )

    async def _check_hemisphere_balance(self, summary: Dict[str, Any]) -> None:
        """If hemisphere ratio is unbalanced, suggest routing adjustment."""
        ratio = summary.get("hemisphere_ratio", 1.0)
        lh = summary.get("lh_requests", 0)
        rh = summary.get("rh_requests", 0)
        total = lh + rh

        if total < 10:
            return  # not enough data

        if (
            self._config.hemisphere_ratio_min
            <= ratio
            <= self._config.hemisphere_ratio_max
        ):
            return

        self._adjustments_made += 1

        if ratio > self._config.hemisphere_ratio_max:
            direction = "lh_heavy"
            suggestion = "lower LH routing threshold to send more queries to RH"
        else:
            direction = "rh_heavy"
            suggestion = "raise LH routing threshold to keep more queries in LH"

        adjust_event = Event.create(
            source="metacognition",
            event_type="meta.adjust.routing",
            payload={
                "reason": "hemisphere_imbalance",
                "direction": direction,
                "hemisphere_ratio": round(ratio, 2),
                "lh_requests": lh,
                "rh_requests": rh,
                "suggestion": suggestion,
                "adjustment_number": self._adjustments_made,
            },
        )
        await self._fabric.publish("agi.meta.adjust.routing", adjust_event)
        logger.info(
            "[meta-adjuster] hemisphere imbalance: ratio=%.2f (%s)",
            ratio,
            direction,
        )

    @property
    def adjustments_made(self) -> int:
        """Return total number of adjustments published."""
        return self._adjustments_made
