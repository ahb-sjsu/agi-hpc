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
Metacognition Reflector for AGI-HPC Phase 5.

Periodically triggers self-reflection by constructing a reflection
prompt and sending it to the LH (Gemma 4) for analysis. Stores
reflections in episodic memory.

Trigger: every N interactions (default 10).

Publishes to:
    agi.meta.reflect
    agi.memory.store.episodic (reflection storage)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

logger = logging.getLogger(__name__)

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric  # noqa: E402
from agi.meta.llm.client import LLMClient  # noqa: E402
from agi.meta.llm.config import InferenceConfig  # noqa: E402

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = (
    "You are the Metacognition module of Atlas, an AGI cognitive architecture. "
    "Your role is self-reflection: evaluating recent system performance, "
    "identifying patterns in errors or inefficiencies, and suggesting "
    "improvements. Be concise, analytical, and actionable."
)


@dataclass
class ReflectorConfig:
    """Configuration for the Metacognition Reflector.

    Attributes:
        reflection_interval: Trigger reflection every N interactions.
        llm_base_url: URL of the LLM for reflection (Gemma 4 / LH).
        llm_timeout: LLM request timeout in seconds.
        llm_model: Model name.
        max_recent_events: Number of recent events to include in prompt.
        store_reflections: Whether to persist reflections in episodic memory.
    """

    reflection_interval: int = 10
    llm_base_url: str = "http://localhost:8080"
    llm_timeout: float = 300.0
    llm_model: str = "gemma-4-31b"
    max_recent_events: int = 20
    store_reflections: bool = True


# -----------------------------------------------------------------
# Recent event buffer
# -----------------------------------------------------------------


@dataclass
class RecentEventSummary:
    """Compact summary of a recent event for reflection context."""

    source: str
    event_type: str
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: bool = False
    preview: str = ""

    def to_text(self) -> str:
        parts = [f"[{self.source}] {self.event_type}"]
        if self.latency_ms > 0:
            parts.append(f"latency={self.latency_ms:.0f}ms")
        if self.tokens_used > 0:
            parts.append(f"tokens={self.tokens_used}")
        if self.error:
            parts.append("ERROR")
        if self.preview:
            parts.append(f'"{self.preview}"')
        return " ".join(parts)


# -----------------------------------------------------------------
# Metacognition Reflector
# -----------------------------------------------------------------


class MetacognitionReflector:
    """Triggers periodic self-reflection on system performance.

    Watches the event stream, and every N interactions constructs
    a reflection prompt summarising recent activity. The prompt is
    sent to the LH (Gemma 4) for analysis. The reflection is then
    stored in episodic memory and published to agi.meta.reflect.

    Usage::

        reflector = MetacognitionReflector()
        await reflector.start(fabric)
        # ... runs alongside the monitor ...
        await reflector.stop()
    """

    def __init__(
        self,
        config: Optional[ReflectorConfig] = None,
    ) -> None:
        self._config = config or ReflectorConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._llm: Optional[LLMClient] = None
        self._recent_events: Deque[RecentEventSummary] = deque(
            maxlen=self._config.max_recent_events
        )
        self._interaction_count: int = 0
        self._reflections_count: int = 0
        self._running = False

    async def start(self, fabric: NatsEventFabric) -> None:
        """Subscribe to response events and initialise the LLM client."""
        self._fabric = fabric

        # Initialise LLM client for reflection (uses Gemma 4 / LH)
        self._llm = LLMClient(
            base_url=self._config.llm_base_url,
            timeout=self._config.llm_timeout,
            default_model=self._config.llm_model,
        )

        # Subscribe to response events (these represent completed interactions)
        await self._fabric.subscribe("agi.lh.response.>", self._on_event)
        await self._fabric.subscribe("agi.rh.response.>", self._on_event)
        await self._fabric.subscribe("agi.integration.merge", self._on_event)

        self._running = True
        logger.info(
            "[meta-reflector] started, reflecting every %d interactions",
            self._config.reflection_interval,
        )

    async def stop(self) -> None:
        """Shut down the reflector."""
        self._running = False
        if self._llm:
            await self._llm.close()
        logger.info("[meta-reflector] stopped")

    async def _on_event(self, event: Event) -> None:
        """Buffer recent events and trigger reflection when due."""
        payload = event.payload or {}

        summary = RecentEventSummary(
            source=event.source,
            event_type=event.type,
            latency_ms=payload.get("latency_ms", 0.0),
            tokens_used=payload.get("tokens_used", 0),
            error=payload.get("error", False),
            preview=str(payload.get("text", ""))[:100],
        )
        self._recent_events.append(summary)
        self._interaction_count += 1

        if self._interaction_count >= self._config.reflection_interval:
            self._interaction_count = 0
            asyncio.create_task(self._perform_reflection())

    async def _perform_reflection(self) -> None:
        """Construct and send a reflection prompt to the LLM."""
        if not self._recent_events:
            return

        t0 = time.perf_counter()

        # Build recent activity summary
        activity_lines = [e.to_text() for e in list(self._recent_events)]
        activity_text = "\n".join(activity_lines)

        # Count errors and compute stats
        events_list = list(self._recent_events)
        error_count = sum(1 for e in events_list if e.error)
        latencies = [e.latency_ms for e in events_list if e.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        prompt = (
            f"Reflect on the following {len(events_list)} recent interactions "
            f"of the Atlas AGI system.\n\n"
            f"Recent Activity:\n{activity_text}\n\n"
            f"Summary Stats:\n"
            f"- Total interactions: {len(events_list)}\n"
            f"- Errors: {error_count}\n"
            f"- Average latency: {avg_latency:.0f}ms\n"
            f"- Reflections so far: {self._reflections_count}\n\n"
            f"Please evaluate:\n"
            f"1. Overall system health and performance\n"
            f"2. Any error patterns or recurring issues\n"
            f"3. Latency trends and optimization opportunities\n"
            f"4. Hemisphere balance (are we over-relying on one side?)\n"
            f"5. Specific actionable improvements\n"
        )

        try:
            config = InferenceConfig(
                temperature=0.4,
                top_p=0.90,
                max_tokens=1024,
                system_prompt=REFLECTION_SYSTEM_PROMPT,
            )
            result = await self._llm.generate(prompt=prompt, config=config)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._reflections_count += 1

            # Publish reflection
            reflect_event = Event.create(
                source="metacognition",
                event_type="meta.reflect",
                payload={
                    "reflection_number": self._reflections_count,
                    "text": result.text,
                    "recent_events_count": len(events_list),
                    "error_count": error_count,
                    "avg_latency_ms": round(avg_latency, 1),
                    "reflection_latency_ms": round(latency_ms, 1),
                    "tokens_used": result.tokens_used,
                },
            )
            await self._fabric.publish("agi.meta.reflect", reflect_event)

            # Store reflection in episodic memory
            if self._config.store_reflections:
                episode_event = Event.create(
                    source="metacognition",
                    event_type="memory.store.episodic",
                    payload={
                        "session_id": "metacognition-reflections",
                        "user_message": prompt[:500],
                        "atlas_response": result.text[:2000],
                        "hemisphere": "lh",
                        "metadata": {
                            "type": "self_reflection",
                            "reflection_number": self._reflections_count,
                            "error_count": error_count,
                        },
                    },
                )
                await self._fabric.publish("agi.memory.store.episodic", episode_event)

            logger.info(
                "[meta-reflector] reflection #%d complete, latency=%.0fms",
                self._reflections_count,
                latency_ms,
            )

        except Exception:
            logger.exception("[meta-reflector] reflection failed")

    @property
    def reflections_count(self) -> int:
        """Return number of reflections performed."""
        return self._reflections_count
