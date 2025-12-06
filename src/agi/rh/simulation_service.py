"""
Right Hemisphere SimulationService

Implements the RH short-horizon simulation API described in:

   • Architecture Section IV.B – World Model Node
   • Architecture Section XI – Sensorimotor Loop
   • Architecture Section XIV.B.1 – Cognitive APIs (SimulationService)
   • Event semantics (simulation.result)

Responsibilities:
    - Receive SimulationRequest from LH
    - Use WorldModel to perform short-horizon rollouts
    - Optionally use Perception for grounding
    - Optionally apply In-Action Safety (VII.B)
    - Return SimulationResult over gRPC
    - Publish simulation.result via EventFabric
"""

from __future__ import annotations

import logging
from typing import Optional, List

import grpc

from agi.core.events.fabric import EventFabric
from agi.proto_gen import plan_pb2, plan_pb2_grpc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationService Implementation
# ---------------------------------------------------------------------------


class SimulationService(plan_pb2_grpc.SimulationServiceServicer):
    """
    RH SimulationService

    Public RPC:

        rpc Simulate(SimulationRequest) returns (SimulationResult)

    This service ties together:
        • Perception   (ground truth object positions)
        • WorldModel   (predictive rollouts)
        • Control      (applies low-level actions if needed)
        • SafetyHooks  (optional)
    """

    def __init__(
        self,
        world_model,
        perception,
        control,
        fabric: EventFabric,
    ) -> None:
        self._wm = world_model
        self._perception = perception
        self._control = control
        self._fabric = fabric

        logger.info("[RH][SimService] SimulationService initialized")

    # ------------------------------------------------------------------ #
    # gRPC API
    # ------------------------------------------------------------------ #

    def Simulate(
        self,
        request: plan_pb2.SimulationRequest,
        context: grpc.ServicerContext,
    ) -> plan_pb2.SimulationResult:
        """
        Execute a short-horizon rollout.

        Workflow (Architecture Section XI):

            1. Extract candidate plan steps.
            2. Ground initial state via perception (optional).
            3. For each step:
                - convert to control actions
                - use WorldModel to generate a trajectory
                - compute risk, constraint violations
            4. Aggregate results into SimulationResult
            5. Publish simulation.result event
        """
        logger.info(
            "[RH][SimService] Received SimulationRequest plan_id=%s steps=%d",
            request.plan_id,
            len(request.candidate_steps),
        )

        try:
            result = self._run_simulation(request)
        except Exception as exc:
            logger.exception("[RH][SimService] Simulation failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Simulation error: {exc}",
            )

        # Publish via EventFabric
        try:
            self._publish_simulation_event(
                plan_id=request.plan_id,
                result=result,
            )
        except Exception:
            logger.exception("[RH][SimService] Failed to publish simulation.result")

        return result

    # ------------------------------------------------------------------ #
    # Internal simulation pipeline
    # ------------------------------------------------------------------ #

    def _run_simulation(
        self, request: plan_pb2.SimulationRequest
    ) -> plan_pb2.SimulationResult:
        """
        Core simulation loop.

        Uses:
            • perception.current_state()
            • world_model.rollout(action_sequence)
            • control.translate_step(step)
        """
        # Step 1 — get initial ground truth (Section IV.B.1)
        try:
            init_state = self._perception.current_state()
        except Exception:
            logger.warning("[RH][SimService] Perception unavailable, using null state")
            init_state = {}

        # Step 2 — For each candidate PlanStep, generate rollout
        step_risk_scores: List[float] = []
        step_violations: List[str] = []

        for step in request.candidate_steps:
            try:
                actions = self._control.translate_step(step)
                rollout = self._wm.rollout(init_state, actions)

                step_risk_scores.append(rollout.risk_score)
                step_violations.append(rollout.violation or "")
            except Exception as exc:
                logger.exception("[RH][SimService] Step simulation failed")
                step_risk_scores.append(1.0)  # max risk fallback
                step_violations.append(f"simulation_error:{exc}")

        # Step 3 — aggregate
        overall_risk = float(sum(step_risk_scores)) / max(
            1, len(step_risk_scores)
        )
        approved = overall_risk < 0.5  # placeholder threshold

        # Step 4 — build protobuf result
        result = plan_pb2.SimulationResult()
        result.plan_id = request.plan_id
        result.overall_risk = overall_risk
        result.approved = approved
        result.notes = "RH short-horizon rollout completed."

        result.step_risk.extend(step_risk_scores)
        result.violations.extend(step_violations)

        return result

    # ------------------------------------------------------------------ #
    # Event publishing
    # ------------------------------------------------------------------ #

    def _publish_simulation_event(
        self,
        plan_id: str,
        result: plan_pb2.SimulationResult,
    ) -> None:
        """
        Publish simulation.result to the EventFabric (Section V).

        Event payload:

            {
                "plan_id": "...",
                "approved": bool,
                "overall_risk": float,
                "step_risk": [...],
                "violations": [...],
            }
        """
        payload = {
            "plan_id": plan_id,
            "approved": result.approved,
            "overall_risk": result.overall_risk,
            "step_risk": list(result.step_risk),
            "violations": list(result.violations),
        }

        logger.debug("[RH][SimService] Publishing simulation.result")
        self._fabric.publish("simulation.result", payload)
