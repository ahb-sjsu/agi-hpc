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
AtlasGym -- Base Gymnasium environment for AGI-HPC training.

Observation space: text (the scenario/question presented to Atlas).
Action space: text (Atlas's response).
Reward: float scored by an evaluator, normalised 0.0 to 1.0.

Integrates with the NATS event fabric, publishing training events
to ``agi.training.*`` subjects for monitoring by the metacognition
subsystem.

Usage::

    env = AtlasGym(env_name="ethics", level=1)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("My response...")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore

try:
    from agi.common.event import Event
    from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig
except ImportError:
    Event = None  # type: ignore
    NatsEventFabric = None  # type: ignore
    NatsFabricConfig = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AtlasGymConfig:
    """Configuration for an AtlasGym environment instance.

    Attributes:
        env_name: Identifier for this environment type (e.g. 'ethics').
        level: Current difficulty level (1-4).
        max_response_length: Maximum characters in an action (response).
        nats_servers: NATS server URLs for event publishing.
        enable_nats: Whether to publish training events to NATS.
        session_id: Training session identifier.
    """

    env_name: str = "base"
    level: int = 1
    max_response_length: int = 8192
    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    enable_nats: bool = False
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A single training scenario presented to Atlas.

    Attributes:
        id: Unique scenario identifier.
        text: The scenario/question text (the observation).
        level: Difficulty level.
        metadata: Extra context (tradition, source, test cases, etc.).
        expected: Optional expected answer for objective scoring.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[str] = None


# ---------------------------------------------------------------------------
# AtlasGym base environment
# ---------------------------------------------------------------------------


class AtlasGym:
    """Base Gymnasium-compatible environment for AGI-HPC training.

    Subclasses override ``_generate_scenario()`` and ``_score_response()``
    to implement domain-specific training environments.

    This class uses the Gymnasium API (reset / step) but does not inherit
    from ``gymnasium.Env`` directly -- text observation and action spaces
    are not natively supported by Gymnasium's space system. Instead,
    observations and actions are plain strings, and the environment
    follows the Gymnasium lifecycle contract.
    """

    # Gymnasium-style metadata
    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(self, config: Optional[AtlasGymConfig] = None) -> None:
        self._config = config or AtlasGymConfig()
        self._current_scenario: Optional[Scenario] = None
        self._episode_count: int = 0
        self._step_count: int = 0
        self._fabric: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Gymnasium-compatible spaces (text represented as unbounded)
        if spaces is not None:
            self.observation_space = spaces.Text(
                min_length=0,
                max_length=self._config.max_response_length,
            )
            self.action_space = spaces.Text(
                min_length=1,
                max_length=self._config.max_response_length,
            )

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
            logger.info(
                "[atlas-gym] NATS connected for env=%s",
                self._config.env_name,
            )
        except Exception:
            logger.warning("[atlas-gym] NATS connection failed; events disabled")
            self._fabric = None

    async def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish a training event to NATS."""
        if self._fabric is None or Event is None:
            return
        try:
            event = Event.create(
                source="training",
                event_type=event_type,
                payload={
                    "env_name": self._config.env_name,
                    "level": self._config.level,
                    "session_id": self._config.session_id,
                    **payload,
                },
            )
            subject = f"agi.training.{self._config.env_name}.{event_type}"
            await self._fabric.publish(subject, event)
        except Exception:
            logger.debug("[atlas-gym] failed to publish event %s", event_type)

    def _fire_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Fire-and-forget NATS event (sync wrapper)."""
        if not self._config.enable_nats or self._fabric is None:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._publish_event(event_type, payload))
            else:
                loop.run_until_complete(self._publish_event(event_type, payload))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Gymnasium lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and return a new scenario.

        Args:
            seed: Optional random seed (unused for text envs).
            options: Optional overrides (e.g. ``{"level": 2}``).

        Returns:
            Tuple of (observation_text, info_dict).
        """
        if options and "level" in options:
            self._config.level = options["level"]

        self._current_scenario = self._generate_scenario(self._config.level)
        self._episode_count += 1
        self._step_count = 0

        info = {
            "scenario_id": self._current_scenario.id,
            "level": self._current_scenario.level,
            "metadata": self._current_scenario.metadata,
            "episode": self._episode_count,
        }

        self._fire_event(
            "reset",
            {
                "scenario_id": self._current_scenario.id,
                "episode": self._episode_count,
            },
        )

        return self._current_scenario.text, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Evaluate Atlas's response to the current scenario.

        Args:
            action: Atlas's text response.

        Returns:
            Tuple of (next_obs, reward, terminated, truncated, info).
            For single-turn environments terminated is always True.
        """
        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1
        t0 = time.perf_counter()

        score, score_breakdown = self._score_response(
            scenario=self._current_scenario,
            response=action,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        info: Dict[str, Any] = {
            "scenario_id": self._current_scenario.id,
            "level": self._current_scenario.level,
            "score_breakdown": score_breakdown,
            "latency_ms": round(latency_ms, 1),
            "step": self._step_count,
            "episode": self._episode_count,
        }

        # Single-turn: done after one step
        terminated = True
        truncated = False

        self._fire_event(
            "step",
            {
                "scenario_id": self._current_scenario.id,
                "score": score,
                "score_breakdown": score_breakdown,
                "latency_ms": round(latency_ms, 1),
            },
        )

        return "", score, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        if self._fabric is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._fabric.disconnect())
                else:
                    loop.run_until_complete(self._fabric.disconnect())
            except RuntimeError:
                pass
            self._fabric = None

    # ------------------------------------------------------------------
    # Methods for subclasses to override
    # ------------------------------------------------------------------

    def _generate_scenario(self, level: int) -> Scenario:
        """Generate a new training scenario for the given level.

        Subclasses MUST override this method.

        Args:
            level: Difficulty level (1-4).

        Returns:
            A Scenario instance.
        """
        raise NotImplementedError("Subclasses must implement _generate_scenario")

    def _score_response(
        self, scenario: Scenario, response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score Atlas's response to a scenario.

        Subclasses MUST override this method.

        Args:
            scenario: The current scenario.
            response: Atlas's text response.

        Returns:
            Tuple of (total_score, breakdown_dict).
            Score must be in [0.0, 1.0].
        """
        raise NotImplementedError("Subclasses must implement _score_response")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def env_name(self) -> str:
        """Return the environment name."""
        return self._config.env_name

    @property
    def level(self) -> int:
        """Return the current difficulty level."""
        return self._config.level

    @level.setter
    def level(self, value: int) -> None:
        """Set the difficulty level."""
        self._config.level = value

    @property
    def episode_count(self) -> int:
        """Return the total episode count."""
        return self._episode_count
