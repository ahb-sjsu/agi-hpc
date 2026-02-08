# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for Safety Gateway reference implementation."""

import pytest

from agi.proto_gen import plan_pb2
from agi.safety.gateway import SafetyGateway, SafetyDecision, SafetyCheckResult
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts


class TestSafetyGatewayInit:
    """Tests for SafetyGateway initialization."""

    def test_gateway_initializes_without_erisml(self):
        """Gateway should initialize without ErisML connection."""
        gateway = SafetyGateway()
        assert gateway._erisml_stub is None
        assert gateway.profile_name == "agi_hpc_safety_v1"

    def test_gateway_initializes_with_custom_config(self):
        """Gateway should accept custom configuration."""
        gateway = SafetyGateway(
            profile_name="custom_profile",
            banned_tools={"tool_a", "tool_b"},
            bond_index_threshold=0.25,
            timeout_ms=200,
        )
        assert gateway.profile_name == "custom_profile"
        assert "tool_a" in gateway.banned_tools
        assert gateway.bond_index_threshold == 0.25
        assert gateway.timeout_ms == 200


class TestSafetyGatewayCheckPlan:
    """Tests for SafetyGateway.check_plan method."""

    def test_check_plan_allows_safe_plan(self):
        """check_plan should allow plans with no banned tools."""
        gateway = SafetyGateway()

        plan = plan_pb2.PlanGraphProto(
            plan_id="test_plan",
            goal_text="Test goal",
        )
        plan.steps.append(
            plan_pb2.PlanStep(
                step_id="step_1",
                kind="action",
                tool_id="observe",
            )
        )

        result = gateway.check_plan(plan)
        assert result.decision == SafetyDecision.ALLOW
        assert len(result.blocked_steps) == 0

    def test_check_plan_blocks_banned_tools(self):
        """check_plan should block plans with banned tools."""
        gateway = SafetyGateway(banned_tools={"dangerous_tool"})

        plan = plan_pb2.PlanGraphProto(plan_id="test_plan")
        plan.steps.append(
            plan_pb2.PlanStep(
                step_id="step_1",
                tool_id="dangerous_tool",
            )
        )

        result = gateway.check_plan(plan)
        assert result.decision == SafetyDecision.BLOCK
        assert "step_1" in result.blocked_steps
        assert any("Banned tool" in r for r in result.reasons)

    def test_check_plan_blocks_multiple_banned_tools(self):
        """check_plan should block all steps with banned tools."""
        gateway = SafetyGateway(banned_tools={"tool_a", "tool_b"})

        plan = plan_pb2.PlanGraphProto(plan_id="test_plan")
        plan.steps.append(plan_pb2.PlanStep(step_id="step_1", tool_id="tool_a"))
        plan.steps.append(plan_pb2.PlanStep(step_id="step_2", tool_id="safe_tool"))
        plan.steps.append(plan_pb2.PlanStep(step_id="step_3", tool_id="tool_b"))

        result = gateway.check_plan(plan)
        assert result.decision == SafetyDecision.BLOCK
        assert "step_1" in result.blocked_steps
        assert "step_3" in result.blocked_steps
        assert "step_2" not in result.blocked_steps


class TestSafetyGatewayCheckAction:
    """Tests for SafetyGateway.check_action method."""

    def test_check_action_allows_safe_action(self):
        """check_action should allow safe actions."""
        gateway = SafetyGateway()

        step = plan_pb2.PlanStep(step_id="step_1", tool_id="observe")
        result = gateway.check_action(step)

        assert result.decision == SafetyDecision.ALLOW

    def test_check_action_blocks_collision_risk(self):
        """check_action should block on collision risk."""
        gateway = SafetyGateway()

        step = plan_pb2.PlanStep(step_id="step_1", tool_id="navigation")
        result = gateway.check_action(
            step,
            sensor_readings={"proximity": 0.05},  # Very close
        )

        assert result.decision == SafetyDecision.BLOCK
        assert "Collision imminent" in result.reasons

    def test_check_action_blocks_excessive_force(self):
        """check_action should block on excessive force."""
        gateway = SafetyGateway()

        step = plan_pb2.PlanStep(step_id="step_1", tool_id="gripper")
        result = gateway.check_action(
            step,
            sensor_readings={"force": 150.0},  # Too high
        )

        assert result.decision == SafetyDecision.BLOCK
        assert "Excessive force" in result.reasons[0]


class TestSafetyGatewayReflexCheck:
    """Tests for SafetyGateway.reflex_check method."""

    def test_reflex_check_safe(self):
        """reflex_check should return True for safe conditions."""
        gateway = SafetyGateway()

        result = gateway.reflex_check(
            physical_harm_risk=0.1,
            collision_probability=0.1,
            emergency_flag=False,
        )
        assert result is True

    def test_reflex_check_emergency_flag(self):
        """reflex_check should return False on emergency flag."""
        gateway = SafetyGateway()

        result = gateway.reflex_check(
            physical_harm_risk=0.0,
            collision_probability=0.0,
            emergency_flag=True,
        )
        assert result is False

    def test_reflex_check_high_harm_risk(self):
        """reflex_check should return False on high harm risk."""
        gateway = SafetyGateway()

        result = gateway.reflex_check(
            physical_harm_risk=0.95,
            collision_probability=0.1,
            emergency_flag=False,
        )
        assert result is False

    def test_reflex_check_high_collision_probability(self):
        """reflex_check should return False on high collision probability."""
        gateway = SafetyGateway()

        result = gateway.reflex_check(
            physical_harm_risk=0.1,
            collision_probability=0.85,
            emergency_flag=False,
        )
        assert result is False


class TestSafetyCheckResult:
    """Tests for SafetyCheckResult dataclass."""

    def test_result_to_proto(self):
        """SafetyCheckResult should convert to proto."""
        result = SafetyCheckResult(
            decision=SafetyDecision.ALLOW,
            risk_score=0.1,
            reasons=["All checks passed"],
            bond_index=0.12,
        )

        proto = result.to_proto()
        assert proto.risk_score == pytest.approx(0.1)
        assert "All checks passed" in proto.reasons
        assert proto.bond_index.bond_index == pytest.approx(0.12)


class TestPlanStepToEthicalFacts:
    """Tests for PlanStepToEthicalFacts builder."""

    def test_build_basic_step(self):
        """Builder should convert basic plan step."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(
            step_id="step_001",
            kind="action",
            tool_id="observe",
        )

        facts = builder.build(step)
        assert facts.option_id == "step_001"
        assert facts.physical_harm_risk < 0.1  # Low risk for observe

    def test_build_high_risk_step(self):
        """Builder should estimate higher risk for physical tools."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(
            step_id="step_002",
            kind="action",
            tool_id="physical_manipulation",
        )

        facts = builder.build(step)
        assert facts.physical_harm_risk > 0.2  # Higher risk
        assert facts.uncertainty_level > 0.5  # Higher uncertainty

    def test_build_harmful_tool(self):
        """Builder should flag harmful tools."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(
            step_id="step_003",
            tool_id="override_safety",
        )

        facts = builder.build(step)
        assert facts.violates_explicit_rule is True
        assert facts.violates_rights is True

    def test_build_with_safety_tags(self):
        """Builder should use safety tags for urgency."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(step_id="step_004", tool_id="action")
        step.safety_tags.append("emergency")

        facts = builder.build(step)
        assert facts.urgency == 1.0

    def test_build_with_world_state(self):
        """Builder should count affected entities from world state."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(step_id="step_005", tool_id="action")
        world_state = {"nearby_agents": ["human_1", "human_2"]}

        facts = builder.build(step, world_state=world_state)
        assert facts.affected_count == 3  # 2 humans + agent

    def test_build_consent_required_tool(self):
        """Builder should check consent for consent-required tools."""
        builder = PlanStepToEthicalFacts()

        # Without consent
        step = plan_pb2.PlanStep(
            step_id="step_006",
            tool_id="access_personal_data",
        )

        facts = builder.build(step)
        assert facts.has_valid_consent is False

        # With consent
        step.params["consent_obtained"] = "true"
        facts = builder.build(step)
        assert facts.has_valid_consent is True

    def test_build_novel_situation(self):
        """Builder should detect novel situations."""
        builder = PlanStepToEthicalFacts()

        step = plan_pb2.PlanStep(step_id="step_007", tool_id="action")
        step.safety_tags.append("novel")

        facts = builder.build(step)
        assert facts.novel_situation is True
