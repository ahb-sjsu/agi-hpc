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
Right Hemisphere – ControlService

Implements the control subsystem described in:

    • Section IV.B – RH Control Node
    • Section XI   – Sensorimotor Loop (control → world model → perception)
    • Section XIV  – Simulation API (PlanStep → action sequence)

Responsibilities:
    - Translate PlanStep (symbolic action) into one or more executable
      low-level control actions.
    - Provide an interface for SimulationService to apply actions
      during short-horizon rollouts.
    - Provide an interface for RHEventLoop to perform real or simulated
      step execution (actuation loop).
    - Interface with the Environment system for simulation/hardware execution.

Now integrated with:
    - agi.env.Environment for world interaction
    - Structured action types (JointAction, CartesianAction, etc.)
    - Async execution support for real hardware
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from agi.proto_gen import plan_pb2

if TYPE_CHECKING:
    from agi.env import Environment, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ControlConfig:
    """Configuration for the control service."""

    controller_type: str = "rule_based"
    default_env: str = "mock:simple"
    action_timeout_sec: float = 30.0
    max_retries: int = 3
    safety_enabled: bool = True


@dataclass
class ActionResult:
    """Result from executing an action."""

    success: bool
    observation: Optional[np.ndarray] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class ControlService:
    """
    RH Control node with Environment integration.

    Actions produced by this service are dictionaries with fields like:

        {
            "type": "move",
            "target": [x, y, z],
            "magnitude": 0.1,
            "duration": 0.5,
        }

    SimulationService feeds these into WorldModel.rollout().
    RHEventLoop feeds them into real or simulated actuators.

    Environment Integration:
        - Uses agi.env.Environment for world interaction
        - Supports both sync and async execution
        - Converts action dicts to numpy arrays for environment stepping
    """

    def __init__(
        self,
        config: Optional[ControlConfig] = None,
        environment: Optional["Environment"] = None,
    ) -> None:
        self._config = config or ControlConfig()
        self._environment: Optional["Environment"] = environment
        self._last_observation: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0
        self._step_count: int = 0

        logger.info(
            "[RH][Control] initialized controller=%s env=%s",
            self._config.controller_type,
            self._environment.name if self._environment else "none",
        )

    # ------------------------------------------------------------------ #
    # Environment Management
    # ------------------------------------------------------------------ #

    def set_environment(self, env: "Environment") -> None:
        """Set the environment for action execution.

        Args:
            env: Environment instance
        """
        self._environment = env
        logger.info("[RH][Control] environment set: %s", env.name)

    def get_environment(self) -> Optional["Environment"]:
        """Get current environment."""
        return self._environment

    async def create_environment(
        self,
        env_name: Optional[str] = None,
    ) -> "Environment":
        """Create and set a new environment.

        Args:
            env_name: Environment name (uses default if not specified)

        Returns:
            Created environment
        """
        from agi.env import create_env

        name = env_name or self._config.default_env
        env = create_env(name)
        self._environment = env

        logger.info("[RH][Control] created environment: %s", name)
        return env

    async def reset_environment(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Initial observation or None if no environment
        """
        if not self._environment:
            logger.warning("[RH][Control] no environment to reset")
            return None

        result = await self._environment.reset(seed=seed, options=options)
        self._last_observation = result.observation
        self._episode_reward = 0.0
        self._step_count = 0

        logger.info(
            "[RH][Control] environment reset seed=%s",
            seed,
        )

        return result.observation

    # ------------------------------------------------------------------ #
    # Convert a PlanStep → list of low-level control actions
    # ------------------------------------------------------------------ #

    def translate_step(self, step: plan_pb2.PlanStep) -> List[Dict[str, Any]]:
        """
        Convert a PlanStep into an action sequence.

        The AGI-HPC architecture states that RH must:

            • convert symbolic task actions into motor primitives
            • support multi-step action expansion
            • interface with simulation and safety

        This stub follows a deterministic mapping pattern that can later
        be replaced with a full motion planner or behavior tree.
        """
        kind = step.kind.lower()  # noqa: F841
        desc = step.description.lower()

        if self._config.controller_type == "disabled":
            return [{"type": "noop", "magnitude": 0.0, "duration": 0.0}]

        # Rule-based controller: simple heuristic mapping
        if "move" in desc or "navigate" in desc:
            return self._make_move_actions(step)

        if "manipulate" in desc or "pick" in desc or "grab" in desc:
            return self._make_manipulation_actions(step)

        if "scan" in desc or "observe" in desc:
            return self._make_scan_actions(step)

        # Generic fallback
        return self._make_default_actions(step)

    # ------------------------------------------------------------------ #
    # Action construction helpers
    # ------------------------------------------------------------------ #

    def _make_move_actions(self, step: plan_pb2.PlanStep) -> List[Dict[str, Any]]:
        """
        Map symbolic movement into a simple translation.
        """
        mag = float(step.params.get("magnitude", "0.5"))
        target = step.params.get("target", "[0.0,0.0,0.0]")

        logger.debug("[RH][Control] move→actions magnitude=%s target=%s", mag, target)

        return [
            {
                "type": "move",
                "target": target,
                "magnitude": mag,
                "duration": 0.5,
            }
        ]

    def _make_manipulation_actions(
        self, step: plan_pb2.PlanStep
    ) -> List[Dict[str, Any]]:
        """
        Stub manipulation primitive.
        """
        logger.debug("[RH][Control] manipulation→actions step_id=%s", step.step_id)

        return [
            {"type": "reach", "magnitude": 0.2, "duration": 0.3},
            {"type": "grasp", "magnitude": 0.1, "duration": 0.2},
            {"type": "lift", "magnitude": 0.3, "duration": 0.5},
        ]

    def _make_scan_actions(self, step: plan_pb2.PlanStep) -> List[Dict[str, Any]]:
        """
        Camera scan / observation action.
        """
        logger.debug("[RH][Control] scan→actions step_id=%s", step.step_id)

        return [
            {"type": "rotate_camera", "magnitude": 0.2, "duration": 0.3},
            {"type": "capture_frame", "magnitude": 0.0, "duration": 0.1},
        ]

    def _make_default_actions(self, step: plan_pb2.PlanStep) -> List[Dict[str, Any]]:
        """
        Fallback action mapping if step type is unknown.
        """
        logger.debug("[RH][Control] default→actions step_id=%s", step.step_id)

        return [
            {"type": "noop", "magnitude": 0.0, "duration": 0.1},
        ]

    # ------------------------------------------------------------------ #
    # Actuator execution for real hardware or environment simulation
    # ------------------------------------------------------------------ #

    async def execute_actions(
        self,
        actions: List[Dict[str, Any]],
    ) -> List[ActionResult]:
        """
        Execute action sequence through the environment.

        In real deployments, this:

            • Sends commands to actuators via Environment interface
            • Interfaces with MuJoCo, Unity, Isaac, ROS, etc.
            • Streams back sensor data from Perception
            • Coordinates with In-Action Safety hooks

        Args:
            actions: List of action dictionaries

        Returns:
            List of ActionResult for each action
        """
        results = []

        for action_dict in actions:
            result = await self.execute_single_action(action_dict)
            results.append(result)

            if result.done:
                logger.info(
                    "[RH][Control] episode ended at step %d",
                    self._step_count,
                )
                break

        return results

    async def execute_single_action(
        self,
        action: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute a single action.

        Args:
            action: Action dictionary

        Returns:
            ActionResult with observation and reward
        """
        logger.debug("[RH][Control] executing %s", action)

        if not self._environment:
            # No environment - stub execution
            return ActionResult(
                success=True,
                info={"stub": True, "action": action},
            )

        try:
            # Convert action dict to environment action
            env_action = self._convert_to_env_action(action)

            # Execute in environment
            step_result = await self._environment.step(env_action)

            # Update tracking
            self._last_observation = step_result.observation
            self._episode_reward += step_result.reward
            self._step_count += 1

            return ActionResult(
                success=True,
                observation=step_result.observation,
                reward=step_result.reward,
                done=step_result.done,
                info=step_result.info,
            )

        except asyncio.TimeoutError:
            logger.warning("[RH][Control] action timeout")
            return ActionResult(
                success=False,
                info={"error": "timeout"},
            )
        except Exception as e:
            logger.error("[RH][Control] action failed: %s", e)
            return ActionResult(
                success=False,
                info={"error": str(e)},
            )

    def _convert_to_env_action(
        self,
        action: Dict[str, Any],
    ) -> np.ndarray:
        """
        Convert action dictionary to environment action format.

        Args:
            action: Action dictionary with type, magnitude, etc.

        Returns:
            Numpy array for environment.step()
        """
        if not self._environment:
            return np.array([0.0])

        action_space = self._environment.action_space
        action_dim = action_space.shape[0] if action_space.shape else 1

        # Parse action type and magnitude
        action_type = action.get("type", "noop")
        magnitude = action.get("magnitude", 0.0)

        if action_type == "noop":
            return np.zeros(action_dim, dtype=np.float32)

        if action_type == "move":
            # Parse target if provided
            target = action.get("target", [0.0, 0.0, 0.0])
            if isinstance(target, str):
                try:
                    import json

                    target = json.loads(target)
                except json.JSONDecodeError:
                    target = [0.0, 0.0, 0.0]

            # Create normalized direction action
            target_arr = np.array(target[:action_dim], dtype=np.float32)
            if np.linalg.norm(target_arr) > 0:
                target_arr = target_arr / np.linalg.norm(target_arr)
            return target_arr * magnitude

        if action_type in ("reach", "grasp", "lift"):
            # Manipulation primitives - return uniform action
            return np.full(action_dim, magnitude, dtype=np.float32)

        if action_type in ("rotate_camera", "capture_frame"):
            # Observation actions - minimal movement
            return np.zeros(action_dim, dtype=np.float32)

        # Default: sample from action space scaled by magnitude
        sample = self._environment.action_space.sample()
        return sample.astype(np.float32) * magnitude

    # ------------------------------------------------------------------ #
    # Observation and State Access
    # ------------------------------------------------------------------ #

    def get_observation(self) -> Optional[np.ndarray]:
        """Get the last observation from the environment."""
        return self._last_observation

    async def observe(self) -> Optional[np.ndarray]:
        """Get current observation from environment."""
        if not self._environment:
            return None
        return await self._environment.observe()

    def get_episode_reward(self) -> float:
        """Get cumulative episode reward."""
        return self._episode_reward

    def get_step_count(self) -> int:
        """Get current step count in episode."""
        return self._step_count

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Clean up resources."""
        if self._environment:
            await self._environment.close()
            self._environment = None
        logger.info("[RH][Control] closed")
