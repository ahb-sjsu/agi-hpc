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
    - Allow future replacement with:
          * PID controllers
          * Learned motor primitives
          * Reinforcement-learned low-level policies
          * Hardware drivers

This file implements a *stub controller* with the correct semantics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agi.proto_gen import plan_pb2  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class ControlService:
    """
    RH Control node.

    Actions produced by this service are dictionaries with fields like:

        {
            "type": "move",
            "target": [x, y, z],
            "magnitude": 0.1,
            "duration": 0.5,
        }

    SimulationService feeds these into WorldModel.rollout().
    RHEventLoop feeds them into real or simulated actuators.
    """

    def __init__(self, controller_type: str = "rule_based") -> None:
        self._controller_type = controller_type
        logger.info("[RH][Control] initialized controller=%s", controller_type)

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
        kind = step.kind.lower()
        desc = step.description.lower()

        if self._controller_type == "disabled":
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

    async def execute_actions(self, actions: List[Dict[str, Any]]) -> None:
        """
        Execute action sequence.

        In real deployments, this:

            • Sends commands to actuators
            • Interfaces with MuJoCo, Unity, Isaac, ROS, etc.
            • Streams back sensor data from Perception
            • Coordinates with In-Action Safety hooks

        For now, this stub just logs the actions.
        """
        for act in actions:
            logger.debug("[RH][Control] executing %s", act)
