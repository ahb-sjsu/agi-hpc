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
Right Hemisphere – World Model

Implements the "short-horizon predictive world model" described in the
AGI-HPC architecture (Sections IV.B.2, XI).

Responsibilities:
    • Accept grounded state from Perception
    • Apply candidate actions (from ControlService)
    • Run short-horizon rollouts to estimate:
          - predicted state trajectory
          - risk scores (per-step + aggregate)
          - constraint violations
    • Produce structured results for SimulationService

This is a stub model suitable for integration tests. Replace rollout()
with real physics/ML dynamics later without changing the RH APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RolloutResult (internal prediction result)
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """
    Struct returned by WorldModel.rollout().

    Matches SimulationResult fields:
        • risk_score
        • violation (string, or empty)
        • predicted_states (optional trajectory; not serialized yet)
    """

    risk_score: float
    violation: str = ""
    predicted_states: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------


class WorldModel:
    """
    Short-horizon predictive model for the Right Hemisphere (RH).

    Architecture summary (IV.B.2):
        • Accept current perceptual state
        • Unroll actions forward for N steps
        • Detect hazards, collisions, or constraint violations
        • Estimate risk & return rollout summary

    This stub is deterministic and simple:
        - Assigns risk based on action type
        - Fake physics: next state = original + index * small delta
        - Violations if action contains specific flags

    Replace with true physics or ML-based model later.

    Methods:
        rollout(init_state, actions) -> RolloutResult
    """

    def __init__(self, model_name: str = "dummy_world_model", horizon: int = 5) -> None:
        self._name = model_name
        self._horizon = horizon
        logger.info(
            "[RH][WorldModel] initialized (model=%s horizon=%d)",
            model_name,
            horizon,
        )

    # ------------------------------------------------------------------ #
    # Main API: rollout
    # ------------------------------------------------------------------ #

    def rollout(
        self,
        initial_state: Dict[str, Any],
        actions: List[Dict[str, Any]],
    ) -> RolloutResult:
        """
        Run a short rollout using the current world model.

        Args:
            initial_state: dict from Perception.current_state()
            actions: list of low-level actions from ControlService.translate_step()

        Returns:
            RolloutResult(risk_score, violation, predicted_states)

        Operational semantics:
            • Evaluate candidate action sequence.
            • Risk estimation:
                  - heuristics for now
                  - real hazard models later
            • Violation:
                  - "collision", "forbidden_tool", etc.
                  - empty string if safe

        This method is intentionally low complexity for early development.
        """
        logger.debug("[RH][WorldModel] rollout start actions=%d", len(actions))

        predicted_states: List[Dict[str, Any]] = []

        # Start from grounded state
        state = dict(initial_state)

        # Fake risk accumulator
        total_risk = 0.0
        violation: str = ""

        for i, act in enumerate(actions[: self._horizon]):
            logger.debug("[RH][WorldModel] applying action %d: %s", i, act)

            # Stub physics: move agent slightly based on "magnitude"
            mag = float(act.get("magnitude", 0.1))
            pos = state.get("agent_pose", [0.0, 0.0, 0.0])

            new_pose = [
                pos[0] + mag,
                pos[1],
                pos[2],
            ]

            # Update state
            state = {
                **state,
                "agent_pose": new_pose,
                "last_action": act,
            }

            predicted_states.append(state)

            # Risk heuristic:
            #   - large magnitudes → more risk
            #   - forbidden tools → violations
            total_risk += min(1.0, mag * 0.2)

            if "forbidden" in act.get("type", "").lower():
                violation = "forbidden_action"
                total_risk = 1.0
                break

            if mag > 1.5:
                violation = "overspeed"
                total_risk = 1.0
                break

        avg_risk = float(total_risk)

        logger.debug(
            "[RH][WorldModel] rollout done risk=%.3f violation=%s",
            avg_risk,
            violation,
        )

        return RolloutResult(
            risk_score=avg_risk,
            violation=violation,
            predicted_states=predicted_states,
        )
