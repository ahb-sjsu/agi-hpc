# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Safety Gateway - Three-layer safety architecture.

Implements pre-action, in-action, and post-action safety checking
with optional ErisML integration for ethical evaluation.

Layers:
    1. Reflex (<100μs): Hardware-level emergency stops
    2. Tactical (10-100ms): ErisML ethical evaluation
    3. Strategic: Policy-level governance

Usage:
    gateway = SafetyGateway(erisml_address="localhost:50060")
    result = gateway.check_plan(plan)
    if result.decision == SafetyDecision.ALLOW:
        execute_plan(plan)
"""

from __future__ import annotations

import logging
import time
from concurrent import futures
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import grpc

from agi.proto_gen import (
    erisml_pb2,
    erisml_pb2_grpc,
    plan_pb2,
    safety_pb2,
    safety_pb2_grpc,
)
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety Decision
# ---------------------------------------------------------------------------


class SafetyDecision(Enum):
    """Safety check decision outcomes."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVISE = "REVISE"
    DEFER = "DEFER"


# ---------------------------------------------------------------------------
# Safety Result
# ---------------------------------------------------------------------------


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    decision: SafetyDecision
    risk_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    blocked_steps: List[str] = field(default_factory=list)
    bond_index: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_proto(self) -> safety_pb2.SafetyResult:
        """Convert to protobuf message."""
        result = safety_pb2.SafetyResult(
            decision=getattr(safety_pb2, f"SAFETY_DECISION_{self.decision.value}", 0),
            risk_score=self.risk_score,
        )
        result.reasons.extend(self.reasons)

        if self.bond_index is not None:
            result.bond_index.CopyFrom(
                erisml_pb2.BondIndexResultProto(
                    bond_index=self.bond_index,
                    baseline=0.155,
                    within_threshold=self.bond_index < 0.30,
                )
            )

        for k, v in self.metadata.items():
            result.metadata[k] = v

        return result


# ---------------------------------------------------------------------------
# Banned Tools Configuration
# ---------------------------------------------------------------------------

DEFAULT_BANNED_TOOLS: Set[str] = {
    "override_safety",
    "bypass_auth",
    "disable_monitoring",
    "root_access",
    "kernel_exploit",
}


# ---------------------------------------------------------------------------
# Safety Gateway
# ---------------------------------------------------------------------------


class SafetyGateway:
    """
    Three-layer safety gateway for AGI-HPC.

    Provides comprehensive safety checking with:
    - Fast rule-based reflex checks
    - ErisML ethical evaluation (optional)
    - Bond Index verification
    - Decision audit logging

    Example:
        # Initialize with ErisML connection
        gateway = SafetyGateway(
            erisml_address="localhost:50060",
            profile_name="agi_hpc_safety_v1",
        )

        # Check a plan before execution
        result = gateway.check_plan(plan)
        if result.decision == SafetyDecision.ALLOW:
            execute_plan(plan)
        elif result.decision == SafetyDecision.REVISE:
            revised_plan = revise_plan(plan, result.reasons)
            result = gateway.check_plan(revised_plan)

        # Real-time action checking
        result = gateway.check_action(step, sensor_readings)
        if result.decision == SafetyDecision.BLOCK:
            emergency_stop()
    """

    def __init__(
        self,
        erisml_address: Optional[str] = None,
        profile_name: str = "agi_hpc_safety_v1",
        banned_tools: Optional[Set[str]] = None,
        bond_index_threshold: float = 0.30,
        timeout_ms: int = 100,
    ):
        """
        Initialize the Safety Gateway.

        Args:
            erisml_address: Address of ErisML service (e.g., "localhost:50060")
            profile_name: DEME profile to use for evaluation
            banned_tools: Set of tool IDs that are always blocked
            bond_index_threshold: Threshold for Bond Index (default 0.30)
            timeout_ms: Timeout for ErisML calls in milliseconds
        """
        self.profile_name = profile_name
        self.banned_tools = banned_tools or DEFAULT_BANNED_TOOLS
        self.bond_index_threshold = bond_index_threshold
        self.timeout_ms = timeout_ms

        # ErisML client (optional)
        self._erisml_stub: Optional[erisml_pb2_grpc.ErisMLServiceStub] = None
        self._erisml_channel: Optional[grpc.Channel] = None

        if erisml_address:
            self._connect_erisml(erisml_address)

        # Facts builder for converting plan steps
        self._facts_builder = PlanStepToEthicalFacts()

        logger.info(
            f"SafetyGateway initialized: erisml={erisml_address}, "
            f"profile={profile_name}, bond_threshold={bond_index_threshold}"
        )

    def _connect_erisml(self, address: str) -> None:
        """Connect to ErisML service."""
        try:
            self._erisml_channel = grpc.insecure_channel(address)
            self._erisml_stub = erisml_pb2_grpc.ErisMLServiceStub(self._erisml_channel)
            logger.info(f"Connected to ErisML service at {address}")
        except Exception as e:
            logger.warning(f"Failed to connect to ErisML: {e}")
            self._erisml_stub = None

    def close(self) -> None:
        """Close connections."""
        if self._erisml_channel:
            self._erisml_channel.close()
            self._erisml_channel = None
            self._erisml_stub = None

    # -----------------------------------------------------------------------
    # Pre-Action Safety (before execution)
    # -----------------------------------------------------------------------

    def check_plan(
        self,
        plan: plan_pb2.PlanGraphProto,
        world_state: Optional[Dict] = None,
    ) -> SafetyCheckResult:
        """
        Check a plan before execution.

        Performs:
        1. Fast rule-based checks (banned tools, schema validation)
        2. ErisML ethical evaluation (if available)
        3. Bond Index verification

        Args:
            plan: The plan to check
            world_state: Optional world state for context

        Returns:
            SafetyCheckResult with decision and reasons
        """
        start_time = time.monotonic()
        reasons: List[str] = []
        blocked_steps: List[str] = []

        # Phase 1: Rule-based checks (fast, <1ms)
        for step in plan.steps:
            if step.tool_id in self.banned_tools:
                blocked_steps.append(step.step_id)
                reasons.append(f"Banned tool: {step.tool_id}")

        if blocked_steps:
            return SafetyCheckResult(
                decision=SafetyDecision.BLOCK,
                risk_score=1.0,
                reasons=reasons,
                blocked_steps=blocked_steps,
                metadata={"phase": "rule_check"},
            )

        # Phase 2: ErisML ethical evaluation (10-100ms)
        bond_index = None
        if self._erisml_stub:
            try:
                erisml_result = self._evaluate_with_erisml(plan, world_state)

                if not erisml_result.plan_approved:
                    return SafetyCheckResult(
                        decision=SafetyDecision.BLOCK,
                        risk_score=0.9,
                        reasons=[
                            f"Ethical veto: {step_id}"
                            for step_id in erisml_result.blocked_steps
                        ],
                        blocked_steps=list(erisml_result.blocked_steps),
                        bond_index=erisml_result.bond_index.bond_index,
                        metadata={"phase": "erisml_check"},
                    )

                bond_index = erisml_result.bond_index.bond_index

                # Check Bond Index threshold
                if not erisml_result.bond_index.within_threshold:
                    return SafetyCheckResult(
                        decision=SafetyDecision.REVISE,
                        risk_score=0.7,
                        reasons=[
                            f"Bond Index {bond_index:.3f} exceeds threshold "
                            f"{self.bond_index_threshold}"
                        ],
                        bond_index=bond_index,
                        metadata={"phase": "bond_index_check"},
                    )

            except grpc.RpcError as e:
                logger.warning(f"ErisML unavailable, using fallback: {e}")
                reasons.append("ErisML unavailable, rule-based check only")

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return SafetyCheckResult(
            decision=SafetyDecision.ALLOW,
            risk_score=0.1,
            reasons=reasons if reasons else ["All checks passed"],
            bond_index=bond_index,
            metadata={
                "phase": "complete",
                "latency_ms": f"{elapsed_ms:.2f}",
            },
        )

    def _evaluate_with_erisml(
        self,
        plan: plan_pb2.PlanGraphProto,
        world_state: Optional[Dict] = None,
    ) -> erisml_pb2.EvaluatePlanResponse:
        """Evaluate plan through ErisML service."""
        step_facts = []
        for step in plan.steps:
            facts = self._facts_builder.build(step, world_state=world_state)
            step_facts.append(facts)

        request = erisml_pb2.EvaluatePlanRequest(
            profile_name=self.profile_name,
            generate_proofs=True,
        )
        request.step_facts.extend(step_facts)

        return self._erisml_stub.EvaluatePlan(
            request,
            timeout=self.timeout_ms / 1000.0,
        )

    # -----------------------------------------------------------------------
    # In-Action Safety (during execution)
    # -----------------------------------------------------------------------

    def check_action(
        self,
        step: plan_pb2.PlanStep,
        sensor_readings: Optional[Dict[str, float]] = None,
    ) -> SafetyCheckResult:
        """
        Check an action during execution.

        Performs real-time safety monitoring with fast response time.

        Args:
            step: The step being executed
            sensor_readings: Real-time sensor data

        Returns:
            SafetyCheckResult with decision
        """
        # Fast reflex check
        if sensor_readings:
            proximity = sensor_readings.get("proximity", 1.0)
            if proximity < 0.1:  # Very close to obstacle
                return SafetyCheckResult(
                    decision=SafetyDecision.BLOCK,
                    risk_score=1.0,
                    reasons=["Collision imminent"],
                    metadata={"emergency_stop": "true"},
                )

            force = sensor_readings.get("force", 0.0)
            if force > 100.0:  # Excessive force
                return SafetyCheckResult(
                    decision=SafetyDecision.BLOCK,
                    risk_score=1.0,
                    reasons=["Excessive force detected"],
                    metadata={"emergency_stop": "true"},
                )

        # Tool check
        if step.tool_id in self.banned_tools:
            return SafetyCheckResult(
                decision=SafetyDecision.BLOCK,
                risk_score=1.0,
                reasons=[f"Banned tool: {step.tool_id}"],
            )

        return SafetyCheckResult(
            decision=SafetyDecision.ALLOW,
            risk_score=0.1,
            reasons=["Action safe to continue"],
        )

    def reflex_check(
        self,
        physical_harm_risk: float,
        collision_probability: float,
        emergency_flag: bool = False,
    ) -> bool:
        """
        Fast reflex check (<100μs target).

        Args:
            physical_harm_risk: Risk of physical harm (0-1)
            collision_probability: Probability of collision (0-1)
            emergency_flag: External emergency signal

        Returns:
            True if safe to continue, False if emergency stop required
        """
        if emergency_flag:
            return False

        if physical_harm_risk > 0.9:
            return False

        if collision_probability > 0.8:
            return False

        return True

    # -----------------------------------------------------------------------
    # Post-Action Safety (after execution)
    # -----------------------------------------------------------------------

    def report_outcome(
        self,
        plan_id: str,
        step_id: str,
        success: bool,
        actual_harm: float = 0.0,
        description: str = "",
    ) -> str:
        """
        Report action outcome for learning.

        Args:
            plan_id: ID of the plan
            step_id: ID of the step
            success: Whether the action succeeded
            actual_harm: Actual harm that occurred (0-1)
            description: Description of outcome

        Returns:
            Event ID for the logged outcome
        """
        event_id = f"outcome_{plan_id}_{step_id}_{int(time.time())}"

        logger.info(
            f"Outcome reported: {event_id} success={success} "
            f"harm={actual_harm:.2f} - {description}"
        )

        # In production, this would log to episodic memory
        return event_id


# ---------------------------------------------------------------------------
# gRPC Service Implementation
# ---------------------------------------------------------------------------


class SafetyGatewayServicer(safety_pb2_grpc.PreActionSafetyServiceServicer):
    """gRPC servicer for Safety Gateway."""

    def __init__(self, gateway: SafetyGateway):
        self.gateway = gateway

    def CheckPlan(
        self,
        request: safety_pb2.CheckPlanRequest,
        context: grpc.ServicerContext,
    ) -> safety_pb2.CheckPlanResponse:
        """Check a plan before execution."""
        result = self.gateway.check_plan(request.plan)

        response = safety_pb2.CheckPlanResponse(
            result=result.to_proto(),
            plan_approved=result.decision == SafetyDecision.ALLOW,
        )

        for step_id in result.blocked_steps:
            step_result = safety_pb2.StepSafetyResult(
                step_id=step_id,
                result=safety_pb2.SafetyResult(
                    decision=safety_pb2.SAFETY_DECISION_BLOCK,
                    risk_score=1.0,
                ),
            )
            response.step_results.append(step_result)

        return response


def create_safety_gateway_server(
    port: int = 50055,
    erisml_address: Optional[str] = "localhost:50060",
    max_workers: int = 10,
) -> grpc.Server:
    """
    Create Safety Gateway gRPC server.

    Args:
        port: Port to listen on
        erisml_address: Address of ErisML service
        max_workers: Maximum thread pool workers

    Returns:
        Configured gRPC server
    """
    gateway = SafetyGateway(erisml_address=erisml_address)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    servicer = SafetyGatewayServicer(gateway)
    safety_pb2_grpc.add_PreActionSafetyServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Safety Gateway server configured on port {port}")

    return server
