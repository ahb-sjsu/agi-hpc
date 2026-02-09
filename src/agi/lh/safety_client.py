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
SafetyClient for the Left Hemisphere (LH).

Implements the LH → Safety subsystem interactions described in:

- Safety Architecture (Sections VII.A–C)
- API Interfaces (XIV.B.3)

SafetyClient performs:
    • Pre-action plan safety checks
    • Hallucination detection (schema/API validity)
    • Constraint vetting (tools, params, forbidden operations)

During early development, if the gRPC SafetyService is unavailable,
the client returns a permissive mock result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import grpc

from agi.proto_gen import plan_pb2, safety_pb2, safety_pb2_grpc

if TYPE_CHECKING:
    from agi.lh.planner import PlanGraph, PlanStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety Decision Types
# ---------------------------------------------------------------------------


class SafetyDecision(Enum):
    """Mirror of proto SafetyDecision enum."""

    UNSPECIFIED = 0
    ALLOW = 1
    BLOCK = 2
    REVISE = 3
    DEFER = 4


class RiskCategory(Enum):
    """Mirror of proto RiskCategory enum."""

    UNSPECIFIED = 0
    PHYSICAL_HARM = 1
    RIGHTS_VIOLATION = 2
    RULE_VIOLATION = 3
    FAIRNESS = 4
    EPISTEMIC = 5
    BOND_INDEX = 6


@dataclass
class StepSafetyResult:
    """Safety check result for a single step."""

    step_id: str
    decision: SafetyDecision
    risk_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    categories: List[RiskCategory] = field(default_factory=list)


@dataclass
class SafetyResult:
    """Full safety check result for a plan."""

    approved: bool
    decision: SafetyDecision = SafetyDecision.ALLOW
    risk_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    categories: List[RiskCategory] = field(default_factory=list)
    step_results: List[StepSafetyResult] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Proto Serialization
# ---------------------------------------------------------------------------


class PlanGraphSerializer:
    """
    Serializes PlanGraph to PlanGraphProto.

    Converts the internal Python plan representation to protobuf format
    for communication with the Safety subsystem.
    """

    @staticmethod
    def to_proto(plan_graph: "PlanGraph") -> plan_pb2.PlanGraphProto:
        """Convert PlanGraph to PlanGraphProto.

        Args:
            plan_graph: Internal plan representation

        Returns:
            Protobuf PlanGraphProto
        """
        proto = plan_pb2.PlanGraphProto(
            plan_id=plan_graph.plan_id,
            goal_text=plan_graph.goal_text,
            schema_version=1,
        )

        # Add metadata
        for key, value in plan_graph.metadata.items():
            proto.metadata[key] = str(value)

        # Add steps
        for step in plan_graph.steps:
            proto.steps.append(PlanGraphSerializer._step_to_proto(step))

        return proto

    @staticmethod
    def _step_to_proto(step: "PlanStep") -> plan_pb2.PlanStep:
        """Convert PlanStep to proto PlanStep.

        Args:
            step: Internal step representation

        Returns:
            Protobuf PlanStep
        """
        proto = plan_pb2.PlanStep(
            step_id=step.step_id,
            index=step.index,
            level=step.level,
            kind=step.kind,
            description=step.description,
            parent_id=step.parent_id or "",
            requires_simulation=step.requires_simulation,
            tool_id=step.tool_id or "",
        )

        # Add safety tags
        for tag in step.safety_tags:
            proto.safety_tags.append(tag)

        # Add params
        for key, value in step.params.items():
            proto.params[key] = str(value)

        return proto


class SafetyResultDeserializer:
    """
    Deserializes proto safety results to Python types.
    """

    @staticmethod
    def from_proto(response: safety_pb2.CheckPlanResponse) -> SafetyResult:
        """Convert CheckPlanResponse to SafetyResult.

        Args:
            response: Protobuf response from safety service

        Returns:
            Python SafetyResult
        """
        result = response.result

        decision = (
            SafetyDecision(result.decision) if result.decision else SafetyDecision.ALLOW
        )
        categories = (
            [RiskCategory(c) for c in result.categories] if result.categories else []
        )

        step_results = []
        for step_result in response.step_results:
            step_results.append(
                SafetyResultDeserializer._step_result_from_proto(step_result)
            )

        return SafetyResult(
            approved=response.plan_approved,
            decision=decision,
            risk_score=result.risk_score,
            issues=list(result.reasons) if result.reasons else [],
            categories=categories,
            step_results=step_results,
            metadata=dict(result.metadata) if result.metadata else {},
        )

    @staticmethod
    def _step_result_from_proto(
        step_result: safety_pb2.StepSafetyResult,
    ) -> StepSafetyResult:
        """Convert proto step result to Python type."""
        result = step_result.result
        decision = (
            SafetyDecision(result.decision) if result.decision else SafetyDecision.ALLOW
        )
        categories = (
            [RiskCategory(c) for c in result.categories] if result.categories else []
        )

        return StepSafetyResult(
            step_id=step_result.step_id,
            decision=decision,
            risk_score=result.risk_score,
            reasons=list(result.reasons) if result.reasons else [],
            categories=categories,
        )


# ---------------------------------------------------------------------------
# Safety Client
# ---------------------------------------------------------------------------


class SafetyClient:
    """
    Client for the SafetyService gRPC API.

    Provides:
        • Pre-action plan safety checks (CheckPlan)
        • Per-step safety checks (CheckStep)
        • Proto serialization/deserialization

    Falls back to mock responses if the Safety service is unavailable.
    """

    DEFAULT_PROFILE = "agi_hpc_safety_v1"

    def __init__(
        self,
        address: str = "safety:50200",
        require_proofs: bool = False,
        profile_name: Optional[str] = None,
    ) -> None:
        self._address = address
        self._require_proofs = require_proofs
        self._profile_name = profile_name or self.DEFAULT_PROFILE
        self._stub: Optional[safety_pb2_grpc.PreActionSafetyServiceStub] = None
        self._connected = False

        self._connect()

    def _connect(self) -> None:
        """Establish gRPC connection to SafetyService."""
        try:
            self._channel = grpc.insecure_channel(self._address)
            self._stub = safety_pb2_grpc.PreActionSafetyServiceStub(self._channel)
            self._connected = True
            logger.info(
                "[LH][SafetyClient] Connected to SafetyService at %s",
                self._address,
            )
        except Exception:
            self._stub = None
            self._connected = False
            logger.warning(
                "[LH][SafetyClient] Could not connect to SafetyService; "
                "will use mock responses"
            )

    def is_connected(self) -> bool:
        """Check if the safety service is connected."""
        return self._connected and self._stub is not None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_plan(self, plan_graph: "PlanGraph") -> SafetyResult:
        """
        Submit a plan to the safety subsystem for pre-action analysis.

        The architecture requires:
            • schema/tool validity
            • hallucination detection
            • constraint checks (preconditions, objects, tools)
            • risk classification (passed or rejected)

        Args:
            plan_graph: The plan to check

        Returns:
            SafetyResult with decision and details
        """
        if not self.is_connected():
            logger.warning("[LH][SafetyClient] SafetyService unavailable; allow plan")
            return SafetyResult(
                approved=True,
                decision=SafetyDecision.ALLOW,
                issues=["mock_safety: service unavailable"],
            )

        try:
            # Serialize plan to protobuf
            plan_proto = PlanGraphSerializer.to_proto(plan_graph)

            # Build request
            request = safety_pb2.CheckPlanRequest(
                plan=plan_proto,
                profile_name=self._profile_name,
                require_proofs=self._require_proofs,
            )

            # Make RPC call
            response = self._stub.CheckPlan(request)

            # Deserialize response
            result = SafetyResultDeserializer.from_proto(response)

            logger.info(
                "[LH][SafetyClient] Plan check plan_id=%s approved=%s risk=%.2f",
                plan_graph.plan_id,
                result.approved,
                result.risk_score,
            )

            return result

        except grpc.RpcError as e:
            logger.warning("[LH][SafetyClient] Safety RPC failed: %s", e)
            return SafetyResult(
                approved=False,
                decision=SafetyDecision.BLOCK,
                issues=[f"safety_rpc_failure: {e.code().name}"],
            )
        except Exception as e:
            logger.exception("[LH][SafetyClient] SafetyService error: %s", e)
            return SafetyResult(
                approved=False,
                decision=SafetyDecision.BLOCK,
                issues=[f"safety_error: {str(e)}"],
            )

    def check_step(
        self,
        plan_id: str,
        step: "PlanStep",
        world_state_ref: str = "",
    ) -> StepSafetyResult:
        """
        Check a single step for safety.

        Args:
            plan_id: ID of the containing plan
            step: The step to check
            world_state_ref: Reference to current world state

        Returns:
            StepSafetyResult with decision
        """
        if not self.is_connected():
            return StepSafetyResult(
                step_id=step.step_id,
                decision=SafetyDecision.ALLOW,
                reasons=["mock_safety: service unavailable"],
            )

        try:
            step_proto = PlanGraphSerializer._step_to_proto(step)

            request = safety_pb2.CheckActionRequest(
                plan_id=plan_id,
                step_id=step.step_id,
                step=step_proto,
                world_state_ref=world_state_ref,
            )

            response = self._stub.CheckStep(request)

            decision = (
                SafetyDecision(response.decision)
                if response.decision
                else SafetyDecision.ALLOW
            )
            categories = (
                [RiskCategory(c) for c in response.categories]
                if response.categories
                else []
            )

            return StepSafetyResult(
                step_id=step.step_id,
                decision=decision,
                risk_score=response.risk_score,
                reasons=list(response.reasons) if response.reasons else [],
                categories=categories,
            )

        except Exception as e:
            logger.warning("[LH][SafetyClient] Step check failed: %s", e)
            return StepSafetyResult(
                step_id=step.step_id,
                decision=SafetyDecision.BLOCK,
                reasons=[f"check_failed: {str(e)}"],
            )

    def close(self) -> None:
        """Close the gRPC channel."""
        if hasattr(self, "_channel"):
            self._channel.close()
        self._connected = False
        logger.info("[LH][SafetyClient] Connection closed")
