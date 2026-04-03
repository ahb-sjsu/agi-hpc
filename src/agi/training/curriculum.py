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
Curriculum Manager for AtlasGym training.

Tracks per-environment difficulty levels and automatically promotes
or demotes based on recent performance. Persists curriculum state
in procedural memory and publishes curriculum events to NATS.

Promotion/demotion rules:
    - Promote to next level when success rate > 80% over last 20 attempts.
    - Demote when success rate < 40% over last 20 attempts.
    - Levels are clamped to [1, max_level] (default max=4).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from agi.common.event import Event
    from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig
except ImportError:
    Event = None  # type: ignore
    NatsEventFabric = None  # type: ignore
    NatsFabricConfig = None  # type: ignore

try:
    from agi.memory.procedural.store import ProceduralMemory, ProceduralMemoryConfig
except ImportError:
    ProceduralMemory = None  # type: ignore
    ProceduralMemoryConfig = None  # type: ignore

from agi.training.scorer import ResponseScorer, ScorerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Configuration for the CurriculumManager.

    Attributes:
        promote_threshold: Success rate above which to promote (0.0-1.0).
        demote_threshold: Success rate below which to demote (0.0-1.0).
        window_size: Number of recent attempts to evaluate.
        max_level: Maximum difficulty level.
        min_level: Minimum difficulty level.
        nats_servers: NATS server URLs for event publishing.
        enable_nats: Whether to publish curriculum events.
        db_dsn: PostgreSQL DSN for the scorer.
        procedural_db_path: SQLite path for procedural memory state.
        enable_procedural: Whether to persist state in procedural memory.
    """

    promote_threshold: float = 0.80
    demote_threshold: float = 0.40
    window_size: int = 20
    max_level: int = 4
    min_level: int = 1
    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    enable_nats: bool = False
    db_dsn: str = "dbname=atlas user=claude"
    procedural_db_path: str = "/home/claude/agi-hpc/data/procedural.db"
    enable_procedural: bool = False


# ---------------------------------------------------------------------------
# Per-environment state
# ---------------------------------------------------------------------------


@dataclass
class EnvCurriculumState:
    """Tracks curriculum state for a single environment.

    Attributes:
        env_name: Environment identifier.
        current_level: Current difficulty level.
        recent_scores: Sliding window of recent scores.
        total_episodes: Total episodes completed.
        promotions: Number of promotions.
        demotions: Number of demotions.
    """

    env_name: str = ""
    current_level: int = 1
    recent_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    total_episodes: int = 0
    promotions: int = 0
    demotions: int = 0

    @property
    def success_rate(self) -> float:
        """Compute success rate (score >= 0.7) over the window."""
        if not self.recent_scores:
            return 0.0
        successes = sum(1 for s in self.recent_scores if s >= 0.7)
        return successes / len(self.recent_scores)

    @property
    def avg_score(self) -> float:
        """Compute average score over the window."""
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "env_name": self.env_name,
            "current_level": self.current_level,
            "success_rate": round(self.success_rate, 3),
            "avg_score": round(self.avg_score, 3),
            "total_episodes": self.total_episodes,
            "recent_scores": list(self.recent_scores),
            "promotions": self.promotions,
            "demotions": self.demotions,
        }


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------


class CurriculumManager:
    """Manages difficulty progression across multiple training environments.

    Tracks per-environment levels, promotes/demotes based on performance,
    and optionally persists state to procedural memory and publishes
    events to NATS ``agi.meta.monitor.training``.

    Usage::

        curriculum = CurriculumManager()
        level = curriculum.get_level("ethics")         # 1
        curriculum.record_score("ethics", 0.9)
        curriculum.record_score("ethics", 0.85)
        # ... after enough high scores ...
        level = curriculum.get_level("ethics")         # 2 (promoted)
    """

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        self._config = config or CurriculumConfig()
        self._states: Dict[str, EnvCurriculumState] = {}
        self._scorer = ResponseScorer(
            config=ScorerConfig(
                db_dsn=self._config.db_dsn,
                auto_create_table=True,
            )
        )
        self._fabric: Optional[Any] = None
        self._procedural: Optional[Any] = None

        # Initialize procedural memory if enabled
        if self._config.enable_procedural and ProceduralMemory is not None:
            try:
                self._procedural = ProceduralMemory(
                    config=ProceduralMemoryConfig(
                        db_path=self._config.procedural_db_path,
                        auto_create=True,
                        seed_procedures=False,
                    )
                )
                self._load_state_from_procedural()
                logger.info("[curriculum] loaded state from procedural memory")
            except Exception:
                logger.warning("[curriculum] could not init procedural memory")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _ensure_state(self, env_name: str) -> EnvCurriculumState:
        """Get or create state for an environment."""
        if env_name not in self._states:
            self._states[env_name] = EnvCurriculumState(
                env_name=env_name,
                current_level=self._config.min_level,
                recent_scores=deque(maxlen=self._config.window_size),
            )
        return self._states[env_name]

    def get_level(self, env_name: str) -> int:
        """Get the current difficulty level for an environment.

        Args:
            env_name: Environment identifier.

        Returns:
            Current level (1 to max_level).
        """
        return self._ensure_state(env_name).current_level

    def set_level(self, env_name: str, level: int) -> None:
        """Manually set the level for an environment.

        Args:
            env_name: Environment identifier.
            level: Target level.
        """
        state = self._ensure_state(env_name)
        state.current_level = max(
            self._config.min_level,
            min(self._config.max_level, level),
        )

    # ------------------------------------------------------------------
    # Score recording and level adjustment
    # ------------------------------------------------------------------

    def record_score(self, env_name: str, score: float) -> Optional[str]:
        """Record a score and check for promotion/demotion.

        Args:
            env_name: Environment identifier.
            score: Normalised score (0.0 to 1.0).

        Returns:
            'promoted', 'demoted', or None.
        """
        state = self._ensure_state(env_name)
        state.recent_scores.append(score)
        state.total_episodes += 1

        # Check for promotion/demotion only when window is full
        if len(state.recent_scores) < self._config.window_size:
            return None

        action = None
        rate = state.success_rate

        if (
            rate >= self._config.promote_threshold
            and state.current_level < self._config.max_level
        ):
            state.current_level += 1
            state.promotions += 1
            state.recent_scores.clear()
            action = "promoted"
            logger.info(
                "[curriculum] %s promoted to level %d (rate=%.2f)",
                env_name,
                state.current_level,
                rate,
            )
        elif (
            rate <= self._config.demote_threshold
            and state.current_level > self._config.min_level
        ):
            state.current_level -= 1
            state.demotions += 1
            state.recent_scores.clear()
            action = "demoted"
            logger.info(
                "[curriculum] %s demoted to level %d (rate=%.2f)",
                env_name,
                state.current_level,
                rate,
            )

        # Persist and publish
        if action:
            self._save_state_to_procedural(env_name)
            self._fire_curriculum_event(env_name, action, rate)

        return action

    # ------------------------------------------------------------------
    # NATS integration
    # ------------------------------------------------------------------

    async def _ensure_fabric(self) -> None:
        """Lazily connect to NATS if enabled."""
        if not self._config.enable_nats or NatsEventFabric is None or Event is None:
            return
        if self._fabric is not None:
            return
        try:
            fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
            self._fabric = NatsEventFabric(config=fabric_config)
            await self._fabric.connect()
        except Exception:
            logger.warning("[curriculum] NATS connection failed")
            self._fabric = None

    def _fire_curriculum_event(self, env_name: str, action: str, rate: float) -> None:
        """Publish curriculum event to NATS."""
        if not self._config.enable_nats or Event is None:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._publish_curriculum_event(env_name, action, rate))
        except RuntimeError:
            pass

    async def _publish_curriculum_event(
        self, env_name: str, action: str, rate: float
    ) -> None:
        """Async publish curriculum event."""
        await self._ensure_fabric()
        if self._fabric is None or Event is None:
            return
        state = self._states.get(env_name)
        if state is None:
            return
        try:
            event = Event.create(
                source="training",
                event_type="meta.monitor.training",
                payload={
                    "curriculum_action": action,
                    "env_name": env_name,
                    "new_level": state.current_level,
                    "success_rate": round(rate, 3),
                    "total_episodes": state.total_episodes,
                },
            )
            await self._fabric.publish("agi.meta.monitor.training", event)
        except Exception:
            logger.debug("[curriculum] failed to publish event")

    # ------------------------------------------------------------------
    # Procedural memory persistence
    # ------------------------------------------------------------------

    def _save_state_to_procedural(self, env_name: str) -> None:
        """Save curriculum state to procedural memory."""
        if self._procedural is None:
            return
        state = self._states.get(env_name)
        if state is None:
            return
        try:
            self._procedural.store_procedure(
                name=f"curriculum_{env_name}",
                trigger=f"training|curriculum|{env_name}",
                steps=[
                    f"current_level: {state.current_level}",
                    f"total_episodes: {state.total_episodes}",
                    f"promotions: {state.promotions}",
                    f"demotions: {state.demotions}",
                    f"success_rate: {state.success_rate:.3f}",
                ],
                metadata={
                    "type": "curriculum_state",
                    "env_name": env_name,
                    "current_level": state.current_level,
                    "total_episodes": state.total_episodes,
                },
            )
        except Exception:
            logger.warning("[curriculum] failed to save state for %s", env_name)

    def _load_state_from_procedural(self) -> None:
        """Load curriculum states from procedural memory."""
        if self._procedural is None:
            return
        try:
            procedures = self._procedural.lookup("curriculum")
            for proc in procedures:
                if not proc.name.startswith("curriculum_"):
                    continue
                env_name = proc.name[len("curriculum_") :]
                meta = proc.metadata or {}
                if "current_level" in meta:
                    state = self._ensure_state(env_name)
                    state.current_level = meta["current_level"]
                    state.total_episodes = meta.get("total_episodes", 0)
                    logger.debug(
                        "[curriculum] loaded state for %s: level=%d",
                        env_name,
                        state.current_level,
                    )
        except Exception:
            logger.warning("[curriculum] failed to load state from procedural")

    # ------------------------------------------------------------------
    # Status and telemetry
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return full curriculum status across all environments.

        Returns:
            Dict with per-environment state.
        """
        return {name: state.to_dict() for name, state in self._states.items()}

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics suitable for a dashboard API.

        Returns:
            Dict with levels, recent scores, events, and totals.
        """
        metrics: Dict[str, Any] = {
            "environments": {},
            "total_episodes": 0,
            "total_promotions": 0,
            "total_demotions": 0,
        }

        for name, state in self._states.items():
            metrics["environments"][name] = {
                "current_level": state.current_level,
                "recent_scores": list(state.recent_scores)[-20:],
                "success_rate": round(state.success_rate, 3),
                "avg_score": round(state.avg_score, 3),
                "episodes": state.total_episodes,
                "promotions": state.promotions,
                "demotions": state.demotions,
            }
            metrics["total_episodes"] += state.total_episodes
            metrics["total_promotions"] += state.promotions
            metrics["total_demotions"] += state.demotions

        return metrics

    async def close(self) -> None:
        """Clean up resources."""
        if self._fabric is not None:
            try:
                await self._fabric.disconnect()
            except Exception:
                pass
            self._fabric = None
