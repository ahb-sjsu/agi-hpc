# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
ErisML Integration Module for AGI-HPC Safety Subsystem.

Implements Sprint 3 requirements:
- Full ErisML integration with ethical evaluation
- Combined rule engine + ErisML evaluation
- Decision proof aggregation
- Human oversight triggers

This module provides the unified interface for safety evaluation that
combines:
- YAML rule engine for tactical checks
- ErisML ethical reasoning for strategic decisions
- Decision proofs for audit trails
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import grpc

from agi.proto_gen import erisml_pb2, erisml_pb2_grpc, plan_pb2
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts

if TYPE_CHECKING:
    from agi.safety.rules.engine import SafetyRuleEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Types
# ---------------------------------------------------------------------------


class EvaluationSource(Enum):
    """Source of safety evaluation."""

    RULE_ENGINE = "rule_engine"
    ERISML = "erisml"
    COMBINED = "combined"


class SafetyDecision(Enum):
    """Safety decision outcomes."""

    ALLOW = "allow"
    DENY = "deny"
    DEFER = "defer"  # Requires human oversight


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ErisMLConfig:
    """Configuration for ErisML integration."""

    address: str = "localhost:50060"
    profile_name: str = "agi_hpc_safety_v1"
    require_proofs: bool = True
    defer_threshold: float = 0.5
    block_threshold: float = 0.3
    timeout_sec: float = 5.0
    enabled: bool = True


@dataclass
class IntegratedEvaluation:
    """Result of integrated safety evaluation."""

    decision: SafetyDecision
    source: EvaluationSource
    rule_result: Optional[Dict[str, Any]] = None
    erisml_result: Optional[Dict[str, Any]] = None
    moral_vector: Optional[Dict[str, float]] = None
    bond_index: float = 0.0
    violations: List[str] = field(default_factory=list)
    decision_proofs: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    evaluation_time_ms: int = 0
    requires_human_review: bool = False
    review_reason: str = ""


@dataclass
class PlanEvaluation:
    """Evaluation result for an entire plan."""

    plan_id: str
    decision: SafetyDecision
    step_evaluations: List[IntegratedEvaluation] = field(default_factory=list)
    blocked_steps: List[str] = field(default_factory=list)
    aggregate_bond_index: float = 0.0
    plan_proof: Optional[Dict[str, Any]] = None
    requires_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ErisML Integration Service
# ---------------------------------------------------------------------------


class ErisMLIntegration:
    """
    Unified ErisML integration service.

    Combines rule-based safety checks with ErisML ethical evaluation
    for comprehensive safety assessment.

    Usage:
        integration = ErisMLIntegration(rule_engine=engine)
        result = await integration.evaluate_step(step, sim_result)
        if result.decision == SafetyDecision.DEFER:
            # Trigger human oversight
            await notify_human(result.review_reason)
    """

    def __init__(
        self,
        rule_engine: Optional["SafetyRuleEngine"] = None,
        config: Optional[ErisMLConfig] = None,
    ) -> None:
        """
        Initialize ErisML integration.

        Args:
            rule_engine: Optional SafetyRuleEngine for tactical checks
            config: ErisML configuration
        """
        self._config = config or ErisMLConfig()
        self._rule_engine = rule_engine
        self._facts_builder = PlanStepToEthicalFacts()
        self._stub: Optional[erisml_pb2_grpc.ErisMLServiceStub] = None
        self._connected = False
        self._evaluation_count = 0

        if self._config.enabled:
            self._connect()

        logger.info(
            "[Safety][ErisML] integration initialized enabled=%s",
            self._config.enabled,
        )

    def _connect(self) -> None:
        """Establish gRPC connection to ErisML service."""
        try:
            self._channel = grpc.insecure_channel(self._config.address)
            self._stub = erisml_pb2_grpc.ErisMLServiceStub(self._channel)
            self._connected = True
            logger.info(
                "[Safety][ErisML] connected to %s",
                self._config.address,
            )
        except Exception as e:
            self._stub = None
            self._connected = False
            logger.warning(
                "[Safety][ErisML] connection failed: %s",
                str(e),
            )

    def is_connected(self) -> bool:
        """Check if ErisML service is connected."""
        return self._connected and self._stub is not None

    # ------------------------------------------------------------------ #
    # Step Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_step(
        self,
        step: plan_pb2.PlanStep,
        simulation_result: Optional[plan_pb2.SimulationResult] = None,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> IntegratedEvaluation:
        """
        Evaluate a single plan step through both rule engine and ErisML.

        Args:
            step: Plan step to evaluate
            simulation_result: Optional simulation results
            world_state: Optional world state context

        Returns:
            IntegratedEvaluation with combined decision
        """
        import time

        start = time.time()
        self._evaluation_count += 1

        violations = []
        rule_result = None
        erisml_result = None
        moral_vector = None
        bond_index = 0.0
        decision_proofs = []
        requires_human = False
        review_reason = ""

        # Phase 1: Rule engine check (tactical layer)
        if self._rule_engine:
            rule_result = self._evaluate_with_rules(step)
            if rule_result.get("violations"):
                violations.extend(rule_result["violations"])

            # If rules block, short-circuit
            if rule_result.get("decision") == "BLOCK":
                return IntegratedEvaluation(
                    decision=SafetyDecision.DENY,
                    source=EvaluationSource.RULE_ENGINE,
                    rule_result=rule_result,
                    violations=violations,
                    evaluation_time_ms=int((time.time() - start) * 1000),
                )

        # Phase 2: ErisML ethical evaluation (strategic layer)
        if self.is_connected():
            erisml_result = self._evaluate_with_erisml(
                step, simulation_result, world_state
            )

            if erisml_result:
                moral_vector = erisml_result.get("moral_vector", {})
                bond_index = erisml_result.get("bond_index", 0.0)

                if erisml_result.get("proof"):
                    decision_proofs.append(erisml_result["proof"])

                if erisml_result.get("vetoed"):
                    violations.append(erisml_result.get("veto_reason", "ethical_veto"))

        # Phase 3: Combined decision
        decision, requires_human, review_reason = self._combine_decisions(
            rule_result, erisml_result, violations
        )

        elapsed = int((time.time() - start) * 1000)

        # Calculate confidence
        confidence = self._calculate_confidence(rule_result, erisml_result)

        logger.debug(
            "[Safety][ErisML] step %s evaluated: decision=%s confidence=%.2f time=%dms",
            step.step_id,
            decision.value,
            confidence,
            elapsed,
        )

        return IntegratedEvaluation(
            decision=decision,
            source=EvaluationSource.COMBINED,
            rule_result=rule_result,
            erisml_result=erisml_result,
            moral_vector=moral_vector,
            bond_index=bond_index,
            violations=violations,
            decision_proofs=decision_proofs,
            confidence=confidence,
            evaluation_time_ms=elapsed,
            requires_human_review=requires_human,
            review_reason=review_reason,
        )

    # ------------------------------------------------------------------ #
    # Plan Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_plan(
        self,
        plan: plan_pb2.PlanGraphProto,
        simulation_result: Optional[plan_pb2.SimulationResult] = None,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> PlanEvaluation:
        """
        Evaluate an entire plan through integrated safety checks.

        Args:
            plan: Plan to evaluate
            simulation_result: Optional simulation results
            world_state: Optional world state context

        Returns:
            PlanEvaluation with step-by-step results
        """
        step_evaluations = []
        blocked_steps = []
        review_reasons = []

        # Evaluate each step
        for step in plan.steps:
            eval_result = self.evaluate_step(step, simulation_result, world_state)
            step_evaluations.append(eval_result)

            if eval_result.decision == SafetyDecision.DENY:
                blocked_steps.append(step.step_id)

            if eval_result.requires_human_review:
                review_reasons.append(eval_result.review_reason)

        # Compute aggregate bond index
        bond_indices = [e.bond_index for e in step_evaluations if e.bond_index > 0]
        aggregate_bond_index = (
            sum(bond_indices) / len(bond_indices) if bond_indices else 0.0
        )

        # Overall plan decision
        if blocked_steps:
            plan_decision = SafetyDecision.DENY
        elif review_reasons:
            plan_decision = SafetyDecision.DEFER
        else:
            plan_decision = SafetyDecision.ALLOW

        # Generate plan-level proof if using ErisML
        plan_proof = None
        if self.is_connected() and self._config.require_proofs:
            plan_proof = self._generate_plan_proof(plan, step_evaluations)

        logger.info(
            "[Safety][ErisML] plan %s evaluated: decision=%s blocked=%d",
            plan.plan_id,
            plan_decision.value,
            len(blocked_steps),
        )

        return PlanEvaluation(
            plan_id=plan.plan_id,
            decision=plan_decision,
            step_evaluations=step_evaluations,
            blocked_steps=blocked_steps,
            aggregate_bond_index=aggregate_bond_index,
            plan_proof=plan_proof,
            requires_human_review=len(review_reasons) > 0,
            review_reasons=review_reasons,
        )

    # ------------------------------------------------------------------ #
    # Internal Methods
    # ------------------------------------------------------------------ #

    def _evaluate_with_rules(self, step: plan_pb2.PlanStep) -> Dict[str, Any]:
        """Evaluate step with rule engine."""
        try:
            result = self._rule_engine.check_action(
                action_type=step.kind,
                parameters=dict(step.params),
                context={"step_id": step.step_id, "tool_id": step.tool_id},
            )
            return {
                "decision": result.decision if hasattr(result, "decision") else "ALLOW",
                "violations": (
                    result.violations if hasattr(result, "violations") else []
                ),
                "matched_rules": (
                    result.matched_rules if hasattr(result, "matched_rules") else []
                ),
            }
        except Exception as e:
            logger.warning("[Safety][ErisML] rule engine error: %s", e)
            return {"decision": "ALLOW", "violations": [], "error": str(e)}

    def _evaluate_with_erisml(
        self,
        step: plan_pb2.PlanStep,
        simulation_result: Optional[plan_pb2.SimulationResult],
        world_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate step with ErisML service."""
        try:
            # Build ethical facts
            facts = self._facts_builder.build(step, simulation_result, world_state)

            # Call ErisML service
            request = erisml_pb2.EvaluateStepRequest(
                facts=facts,
                profile_name=self._config.profile_name,
            )

            response = self._stub.EvaluateStep(
                request,
                timeout=self._config.timeout_sec,
            )

            # Extract moral vector
            mv = response.moral_vector
            moral_vector = {
                "physical_harm": mv.physical_harm,
                "rights_respect": mv.rights_respect,
                "fairness_equity": mv.fairness_equity,
                "autonomy_respect": mv.autonomy_respect,
                "privacy_protection": mv.privacy_protection,
                "societal_environmental": mv.societal_environmental,
                "virtue_care": mv.virtue_care,
                "legitimacy_trust": mv.legitimacy_trust,
                "epistemic_quality": mv.epistemic_quality,
            }

            # Extract proof
            proof = None
            if response.HasField("proof"):
                proof = {
                    "decision_id": response.proof.decision_id,
                    "timestamp": response.proof.timestamp,
                    "profile_name": response.proof.profile_name,
                    "proof_hash": response.proof.proof_hash,
                    "confidence": response.proof.confidence,
                }

            return {
                "verdict": response.verdict,
                "moral_vector": moral_vector,
                "vetoed": response.vetoed,
                "veto_reason": response.veto_reason,
                "proof": proof,
                "bond_index": 0.0,  # Computed at plan level
            }

        except grpc.RpcError as e:
            logger.warning("[Safety][ErisML] RPC error: %s", e.code())
            return None
        except Exception as e:
            logger.exception("[Safety][ErisML] evaluation error: %s", e)
            return None

    def _combine_decisions(
        self,
        rule_result: Optional[Dict[str, Any]],
        erisml_result: Optional[Dict[str, Any]],
        violations: List[str],
    ) -> tuple[SafetyDecision, bool, str]:
        """
        Combine rule engine and ErisML decisions.

        Returns:
            Tuple of (decision, requires_human, review_reason)
        """
        # If there are hard violations, deny
        if violations:
            return SafetyDecision.DENY, False, ""

        # Check ErisML verdict
        if erisml_result:
            verdict = erisml_result.get("verdict", "neutral")

            if verdict == "forbid":
                return SafetyDecision.DENY, False, ""

            if verdict == "avoid":
                # Low confidence - might need human review
                confidence = erisml_result.get("moral_vector", {}).get(
                    "epistemic_quality", 0.5
                )
                if confidence < self._config.defer_threshold:
                    return (
                        SafetyDecision.DEFER,
                        True,
                        "Low confidence ethical evaluation",
                    )

        # Check rule engine decision
        if rule_result:
            if rule_result.get("decision") == "DEFER":
                return SafetyDecision.DEFER, True, "Rule engine requires human review"

        # Default: allow
        return SafetyDecision.ALLOW, False, ""

    def _calculate_confidence(
        self,
        rule_result: Optional[Dict[str, Any]],
        erisml_result: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate overall confidence in the decision."""
        confidences = []

        if erisml_result:
            mv = erisml_result.get("moral_vector", {})
            epistemic = mv.get("epistemic_quality", 0.5)
            confidences.append(epistemic)

        if rule_result and not rule_result.get("error"):
            # Rule engine gives binary confidence
            confidences.append(0.9)

        if not confidences:
            return 0.5

        return sum(confidences) / len(confidences)

    def _generate_plan_proof(
        self,
        plan: plan_pb2.PlanGraphProto,
        step_evaluations: List[IntegratedEvaluation],
    ) -> Dict[str, Any]:
        """Generate plan-level decision proof."""
        import hashlib

        # Aggregate step proof hashes
        step_hashes = []
        for eval_result in step_evaluations:
            for proof in eval_result.decision_proofs:
                step_hashes.append(proof.get("proof_hash", ""))

        # Compute plan proof hash
        plan_content = f"{plan.plan_id}:{','.join(step_hashes)}"
        plan_hash = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

        return {
            "plan_id": plan.plan_id,
            "timestamp": datetime.now().isoformat(),
            "step_count": len(step_evaluations),
            "blocked_count": sum(
                1 for e in step_evaluations if e.decision == SafetyDecision.DENY
            ),
            "plan_proof_hash": plan_hash,
            "step_proof_hashes": step_hashes,
        }

    def close(self) -> None:
        """Close gRPC connection."""
        if hasattr(self, "_channel"):
            self._channel.close()
            self._connected = False
            logger.info("[Safety][ErisML] connection closed")
