"""
Left Hemisphere (LH) – PlanService

Implements the plan generation pipeline described in the AGI-HPC architecture
(see Sections IV.A, V, VII, VIII, XI in the architecture document).

Pipeline (LH role summary):
1. Task interpretation → goal graph construction.
2. Memory retrieval (semantic, episodic, procedural).
3. Hierarchical planning → candidate plan graph.
4. Pre-action safety analysis (hallucination detection, tool/syntax/schema checks).
5. Cross-hemisphere simulation requests (LH → RH) via EventFabric.
6. Metacognitive evaluation (accept/revise/reject).
7. Publish plan.step_ready events for each step.
8. Return structured PlanResponse.

This module does NOT do heavy computation itself:
it orchestrates Planner, Safety, Memory, Metacognition, and EventFabric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

import grpc

from agi.core.api.grpc_server import GRPCServer  # used by LH main service
from agi.core.events.fabric import EventFabric
from agi.proto_gen import plan_pb2, plan_pb2_grpc

# LH internal clients and modules, to be implemented in src/agi/lh/
from agi.lh.planner import Planner
from agi.lh.memory_client import MemoryClient
from agi.lh.safety_client import SafetyClient
from agi.lh.metacog_client import MetacognitionClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# LH PlanService configuration
# --------------------------------------------------------------------------------------

@dataclass
class LHPlanServiceConfig:
    """
    Configuration for the LH PlanService.

    The defaults align with:
    - Event fabric topics (Section V)
    - Safety architecture (Section VII)
    - Metacognition (Section VIII)
    - Sensorimotor loop (Section XI)
    """

    enable_safety: bool = True
    enable_metacognition: bool = True

    # EventFabric topics (from Appendix C: Event Schemas)
    topic_plan_step: str = "plan.step_ready"
    topic_plan_complete: str = "plan.completed"

    node_id: str = "LH"


# --------------------------------------------------------------------------------------
# LH PlanService Implementation
# --------------------------------------------------------------------------------------

class PlanService(plan_pb2_grpc.PlanServiceServicer):
    """
    Implements LH PlanService RPC described in:
      • Section XIV.B.1 – LH↔RH Cognitive APIs (Plan APIs)
      • Appendix A – API Schema Definitions

    This service is responsible ONLY for orchestration, not reasoning.
    All heavy lifting is delegated to Planner, Memory, Safety, Metacog, and RH (via EventFabric).
    """

    def __init__(
        self,
        planner: Planner,
        memory: MemoryClient,
        safety: SafetyClient,
        metacog: MetacognitionClient,
        fabric: EventFabric,
        config: Optional[LHPlanServiceConfig] = None,
    ) -> None:
        self._planner = planner
        self._memory = memory
        self._safety = safety
        self._meta = metacog
        self._fabric = fabric
        self._cfg = config or LHPlanServiceConfig()

        logger.info(
            "[LH][PlanService] Initialized node_id=%s safety=%s meta=%s",
            self._cfg.node_id,
            self._cfg.enable_safety,
            self._cfg.enable_metacognition,
        )

    # ----------------------------------------------------------------------------------
    # RPC: GeneratePlan / Plan / CreatePlan
    # (Actual method name must match plan.proto)
    # ----------------------------------------------------------------------------------

    def Plan 
        self,
        request: plan_pb2.PlanRequest, 
        context: grpc.ServicerContext,
    ) -> plan_pb2.PlanResponse: # noqa: C901
        """
        Full LH planning pipeline:

            Request → memory retrieval → candidate plan
                    → safety filtering → simulation requests
                    → metacognitive review
                    → publish plan.step_ready → response

        Architecturally grounded in:
        - Task Interpreter & Goal Manager (IV.A)
        - Hierarchical Planning Engine (IV.A)
        - Safety Supervisor (IV.A, VII)
        - Metacognition Layer (VIII)
        - EventFabric semantics (V)
        - Sensorimotor loop integration (XI)
        """

        logger.info("[LH][PlanService] Received PlanRequest")

        # 1) Memory augmentation
        try:
            enriched_req = self._memory.enrich_request(request)
        except Exception:
            logger.exception("[LH][PlanService] Memory enrichment failed; using raw request")
            enriched_req = request

        # 2) Planner → internal plan graph
        try:
            plan_graph = self._planner.generate_plan(enriched_req)
        except Exception as exc:
            logger.exception("[LH][PlanService] Planner failed")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Planner failed: {exc}",
            )

        # 3) Pre-action safety (Section VII.A)
        if self._cfg.enable_safety:
            try:
                safety_result = self._safety.check_plan(plan_graph)
            except Exception as exc:
                logger.exception("[LH][PlanService] Safety check failed")
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Safety check failed: {exc}",
                )

            if not getattr(safety_result, "approved", False):
                logger.warning("[LH][PlanService] Plan rejected by safety subsystem")
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Plan rejected by safety subsystem",
                )

        # 4) Metacognition (Section VIII)
        if self._cfg.enable_metacognition:
            try:
                review = self._meta.review_plan(plan_graph)
            except Exception as exc:
                logger.exception("[LH][PlanService] Metacognition review failed")
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Metacognition review failed: {exc}",
                )

            decision = getattr(review, "decision", "ACCEPT")

            if decision == "REJECT":
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "Metacognition rejected the plan",
                )

            if decision == "REVISE":
                try:
                    plan_graph = self._meta.revise_plan(plan_graph, review)
                except Exception as exc:
                    logger.exception("[LH][PlanService] Plan revision failed")
                    context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"Plan revision failed: {exc}",
                    )

        # 5) Publish plan steps over EventFabric (LH → RH)
        self._publish_plan_steps(plan_graph)

        # 6) Build structured PlanResponse (proto)
        try:
            response = self._build_plan_response(plan_graph)
        except Exception as exc:
            logger.exception("[LH][PlanService] Failed to build PlanResponse")
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"PlanResponse construction failed: {exc}",
            )

        return response

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------

    def _publish_plan_steps(self, plan_graph) -> None:
        """
        Publish steps to RH via event fabric.

        Events match the architecture:
        - plan.step_ready (V.A)
        - plan.completed (optional)
        """
        steps = getattr(plan_graph, "steps", None)
        if not steps:
            logger.info("[LH][PlanService] Plan contains no steps")
            return

        for idx, step in enumerate(steps):
            payload = {
                "node_id": self._cfg.node_id,
                "index": idx,
                "step": self._serialize_step(step),
            }
            try:
                self._fabric.publish(self._cfg.topic_plan_step, payload)
            except Exception:
                logger.exception("[LH][PlanService] Error publishing step idx=%d", idx)

        # Publish plan completion marker
        try:
            self._fabric.publish(
                self._cfg.topic_plan_complete,
                {"node_id": self._cfg.node_id, "num_steps": len(steps)},
            )
        except Exception:
            logger.exception("[LH][PlanService] Failed to publish plan completion event")

    def _serialize_step(self, step) -> dict:
        """
        Convert an internal step object into a JSON-serializable dict.
        Fully compatible with EventFabric transport (UCX/ZMQ).
        """
        if isinstance(step, dict):
            return step

        # Generic object → dict mapping
        data = {"repr": repr(step)}
        for attr in ("action", "params", "preconditions", "postconditions"):
            if hasattr(step, attr):
                data[attr] = getattr(step, attr)
        return data

    def _build_plan_response(
        self,
        plan_graph,
    ) -> plan_pb2.PlanResponse:
        """
        Convert plan_graph into protobuf PlanResponse.

        The architecture (XIV.B.1) expects LH to return:
            - plan metadata
            - step summaries
            - notes / confidence (optional)
        """
        response = plan_pb2.PlanResponse()

        plan_id = getattr(plan_graph, "plan_id", None)
        if plan_id and hasattr(response, "plan_id"):
            response.plan_id = str(plan_id)

        if hasattr(response, "notes"):
            response.notes = "Plan generated successfully."

        if hasattr(response, "steps"):
            for idx, step in enumerate(getattr(plan_graph, "steps", [])):
                msg = response.steps.add()
                if hasattr(msg, "index"):
                    msg.index = idx
                if hasattr(msg, "description"):
                    msg.description = getattr(step, "description", repr(step))

        return response
