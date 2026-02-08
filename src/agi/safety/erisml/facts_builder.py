# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Converts AGI-HPC plan steps to ErisML EthicalFacts.

This is the critical domain-specific bridge that maps cognitive architecture
concepts (tools, goals, simulation results) to ethical reasoning inputs
(consequences, rights, fairness, safety, epistemic status).

Usage:
    builder = PlanStepToEthicalFacts()
    facts = builder.build(plan_step, simulation_result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from agi.proto_gen import erisml_pb2, plan_pb2

# ---------------------------------------------------------------------------
# Tool Classifications
# ---------------------------------------------------------------------------

# Tools that may cause harm if misused
HARMFUL_TOOLS: Set[str] = {
    "delete",
    "destroy",
    "terminate",
    "override_safety",
    "force_shutdown",
    "bypass_auth",
}

# Tools requiring explicit user consent
CONSENT_REQUIRED_TOOLS: Set[str] = {
    "access_personal_data",
    "modify_user_settings",
    "read_private_files",
    "send_notification",
    "modify_preferences",
}

# Tools with physical world interaction (higher risk)
HIGH_RISK_TOOLS: Set[str] = {
    "physical_manipulation",
    "navigation",
    "power_control",
    "motor_control",
    "gripper_control",
    "arm_movement",
}

# Tools with low risk
LOW_RISK_TOOLS: Set[str] = {
    "observe",
    "query",
    "analyze",
    "plan",
    "log",
    "report",
}


# ---------------------------------------------------------------------------
# Risk Estimates
# ---------------------------------------------------------------------------


@dataclass
class RiskEstimates:
    """Estimated risks from simulation or heuristics."""

    physical_harm: float = 0.0
    collision_probability: float = 0.0
    rights_violation_risk: float = 0.0
    uncertainty: float = 0.5


# ---------------------------------------------------------------------------
# Facts Builder
# ---------------------------------------------------------------------------


class PlanStepToEthicalFacts:
    """
    Converts AGI-HPC PlanStep to ErisML EthicalFactsProto.

    This is domain-specific logic that maps:
    - Plan step parameters → ethical dimensions
    - Tool usage → rights/harm implications
    - Simulation results → risk estimates
    - Safety tags → urgency and constraints

    Example:
        builder = PlanStepToEthicalFacts()

        # With simulation results
        facts = builder.build(step, simulation_result=sim_result)

        # Without simulation (heuristic estimates)
        facts = builder.build(step)

        # With world state context
        facts = builder.build(step, world_state={"nearby_agents": ["human_1"]})
    """

    def __init__(
        self,
        harmful_tools: Optional[Set[str]] = None,
        consent_tools: Optional[Set[str]] = None,
        high_risk_tools: Optional[Set[str]] = None,
    ):
        """
        Initialize the facts builder.

        Args:
            harmful_tools: Override set of harmful tools
            consent_tools: Override set of consent-required tools
            high_risk_tools: Override set of high-risk tools
        """
        self.harmful_tools = harmful_tools or HARMFUL_TOOLS
        self.consent_tools = consent_tools or CONSENT_REQUIRED_TOOLS
        self.high_risk_tools = high_risk_tools or HIGH_RISK_TOOLS

    def build(
        self,
        step: plan_pb2.PlanStep,
        simulation_result: Optional[plan_pb2.SimulationResult] = None,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> erisml_pb2.EthicalFactsProto:
        """
        Convert plan step to EthicalFacts.

        Args:
            step: The plan step to convert
            simulation_result: Optional simulation results for risk estimation
            world_state: Optional world state for context

        Returns:
            EthicalFactsProto ready for ErisML evaluation
        """
        # Extract risk estimates from simulation if available
        risks = self._estimate_risks(step, simulation_result)

        # Build facts proto
        facts = erisml_pb2.EthicalFactsProto(
            option_id=step.step_id,
            # Consequences
            expected_benefit=self._estimate_benefit(step),
            expected_harm=risks.physical_harm,
            urgency=self._estimate_urgency(step),
            affected_count=self._count_affected(step, world_state),
            # Rights and Duties
            violates_rights=self._check_rights_violation(step),
            has_valid_consent=self._check_consent(step),
            violates_explicit_rule=step.tool_id in self.harmful_tools,
            # Justice and Fairness
            discriminates_on_protected_attr=False,  # Would need context
            exploits_vulnerable_population=False,
            # Safety
            physical_harm_risk=risks.physical_harm,
            collision_probability=risks.collision_probability,
            # Epistemic
            uncertainty_level=risks.uncertainty,
            evidence_quality=1.0 - risks.uncertainty,
            novel_situation=self._is_novel_situation(step),
        )

        return facts

    def _estimate_risks(
        self,
        step: plan_pb2.PlanStep,
        sim_result: Optional[plan_pb2.SimulationResult] = None,
    ) -> RiskEstimates:
        """Extract or estimate risks from simulation."""
        if sim_result and sim_result.overall_risk > 0:
            # Use simulation results when available
            collision_prob = 0.0
            if sim_result.step_risk:
                # Find risk for this specific step
                step_idx = step.index if step.index >= 0 else 0
                if step_idx < len(sim_result.step_risk):
                    collision_prob = sim_result.step_risk[step_idx]

            return RiskEstimates(
                physical_harm=sim_result.overall_risk,
                collision_probability=collision_prob,
                uncertainty=0.1,  # Low uncertainty with simulation
            )

        # Heuristic estimates without simulation
        if step.tool_id in self.high_risk_tools:
            return RiskEstimates(
                physical_harm=0.3,
                collision_probability=0.2,
                uncertainty=0.7,  # High uncertainty without simulation
            )

        if step.tool_id in self.harmful_tools:
            return RiskEstimates(
                physical_harm=0.5,
                collision_probability=0.1,
                rights_violation_risk=0.8,
                uncertainty=0.5,
            )

        # Default low risk
        return RiskEstimates(
            physical_harm=0.05,
            collision_probability=0.01,
            uncertainty=0.3,
        )

    def _estimate_benefit(self, step: plan_pb2.PlanStep) -> float:
        """Estimate expected benefit of step."""
        # Mission-level steps have high benefit
        if step.level == 0:
            return 0.9

        # Actionable steps have moderate benefit
        if step.kind == "action":
            return 0.6

        # Planning/observation steps have lower direct benefit
        if step.kind in ("plan", "observe", "analyze"):
            return 0.4

        return 0.5

    def _estimate_urgency(self, step: plan_pb2.PlanStep) -> float:
        """Estimate urgency from safety tags."""
        tags = set(step.safety_tags)

        if "emergency" in tags:
            return 1.0
        if "time_critical" in tags:
            return 0.8
        if "urgent" in tags:
            return 0.6
        if "routine" in tags:
            return 0.2

        return 0.3

    def _count_affected(
        self,
        step: plan_pb2.PlanStep,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count entities affected by this step."""
        if world_state:
            if "nearby_agents" in world_state:
                return len(world_state["nearby_agents"]) + 1
            if "affected_entities" in world_state:
                return len(world_state["affected_entities"])

        # Default: affects at least the agent itself
        return 1

    def _check_rights_violation(self, step: plan_pb2.PlanStep) -> bool:
        """Check if step violates rights."""
        # Harmful tools violate rights
        if step.tool_id in self.harmful_tools:
            return True

        # Check for explicit violation tag
        if "rights_violation" in step.safety_tags:
            return True

        return False

    def _check_consent(self, step: plan_pb2.PlanStep) -> bool:
        """Check if consent was obtained for consent-required actions."""
        if step.tool_id not in self.consent_tools:
            return True  # Consent not required

        # Check for consent parameter
        consent_param = step.params.get("consent_obtained", "false")
        return consent_param.lower() in ("true", "yes", "1")

    def _is_novel_situation(self, step: plan_pb2.PlanStep) -> bool:
        """Check if this is a novel situation."""
        tags = set(step.safety_tags)
        return "novel" in tags or "unprecedented" in tags or "first_encounter" in tags
