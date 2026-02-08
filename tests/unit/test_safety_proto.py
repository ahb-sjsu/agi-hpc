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

"""Tests for Safety Gateway protobuf messages and service definitions."""

import pytest

from agi.proto_gen import safety_pb2, erisml_pb2, plan_pb2


class TestSafetyDecisionEnum:
    """Tests for SafetyDecision enum."""

    def test_safety_decision_unspecified(self):
        """SafetyDecision should have UNSPECIFIED as 0."""
        assert safety_pb2.SAFETY_DECISION_UNSPECIFIED == 0

    def test_safety_decision_allow(self):
        """SafetyDecision should have ALLOW."""
        assert safety_pb2.SAFETY_DECISION_ALLOW == 1

    def test_safety_decision_block(self):
        """SafetyDecision should have BLOCK."""
        assert safety_pb2.SAFETY_DECISION_BLOCK == 2

    def test_safety_decision_revise(self):
        """SafetyDecision should have REVISE."""
        assert safety_pb2.SAFETY_DECISION_REVISE == 3

    def test_safety_decision_defer(self):
        """SafetyDecision should have DEFER."""
        assert safety_pb2.SAFETY_DECISION_DEFER == 4


class TestRiskCategoryEnum:
    """Tests for RiskCategory enum."""

    def test_risk_category_unspecified(self):
        """RiskCategory should have UNSPECIFIED as 0."""
        assert safety_pb2.RISK_CATEGORY_UNSPECIFIED == 0

    def test_risk_category_physical_harm(self):
        """RiskCategory should have PHYSICAL_HARM."""
        assert safety_pb2.RISK_CATEGORY_PHYSICAL_HARM == 1

    def test_risk_category_rights_violation(self):
        """RiskCategory should have RIGHTS_VIOLATION."""
        assert safety_pb2.RISK_CATEGORY_RIGHTS_VIOLATION == 2

    def test_risk_category_rule_violation(self):
        """RiskCategory should have RULE_VIOLATION."""
        assert safety_pb2.RISK_CATEGORY_RULE_VIOLATION == 3

    def test_risk_category_fairness(self):
        """RiskCategory should have FAIRNESS."""
        assert safety_pb2.RISK_CATEGORY_FAIRNESS == 4

    def test_risk_category_epistemic(self):
        """RiskCategory should have EPISTEMIC."""
        assert safety_pb2.RISK_CATEGORY_EPISTEMIC == 5

    def test_risk_category_bond_index(self):
        """RiskCategory should have BOND_INDEX."""
        assert safety_pb2.RISK_CATEGORY_BOND_INDEX == 6


class TestSafetyResult:
    """Tests for SafetyResult message."""

    def test_safety_result_allow(self):
        """SafetyResult should represent ALLOW decision."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            risk_score=0.1,
        )
        assert result.decision == safety_pb2.SAFETY_DECISION_ALLOW
        assert result.risk_score == pytest.approx(0.1)

    def test_safety_result_block(self):
        """SafetyResult should represent BLOCK decision."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_BLOCK,
            risk_score=0.95,
        )
        result.reasons.append("Physical harm risk exceeds threshold")
        result.categories.append(safety_pb2.RISK_CATEGORY_PHYSICAL_HARM)

        assert result.decision == safety_pb2.SAFETY_DECISION_BLOCK
        assert result.risk_score == pytest.approx(0.95)
        assert len(result.reasons) == 1
        assert safety_pb2.RISK_CATEGORY_PHYSICAL_HARM in result.categories

    def test_safety_result_revise(self):
        """SafetyResult should represent REVISE decision."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_REVISE,
            risk_score=0.7,
        )
        result.reasons.append("Bond Index exceeds threshold")
        result.categories.append(safety_pb2.RISK_CATEGORY_BOND_INDEX)

        assert result.decision == safety_pb2.SAFETY_DECISION_REVISE

    def test_safety_result_with_bond_index(self):
        """SafetyResult should include ErisML Bond Index result."""
        bond_index = erisml_pb2.BondIndexResultProto(
            bond_index=0.25,
            baseline=0.155,
            within_threshold=True,
        )
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            risk_score=0.3,
            bond_index=bond_index,
        )

        assert result.bond_index.bond_index == pytest.approx(0.25)
        assert result.bond_index.within_threshold is True

    def test_safety_result_with_decision_proof(self):
        """SafetyResult should include ErisML DecisionProof."""
        proof = erisml_pb2.DecisionProofProto(
            decision_id="dec_001",
            proof_hash="abc123",
        )
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            decision_proof=proof,
        )

        assert result.decision_proof.decision_id == "dec_001"
        assert result.decision_proof.proof_hash == "abc123"

    def test_safety_result_metadata(self):
        """SafetyResult should support metadata map."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
        )
        result.metadata["latency_ms"] = "5.2"
        result.metadata["profile"] = "agi_hpc_safety_v1"

        assert result.metadata["latency_ms"] == "5.2"
        assert result.metadata["profile"] == "agi_hpc_safety_v1"


class TestCheckPlanRequest:
    """Tests for CheckPlanRequest message."""

    def test_check_plan_request_with_plan(self):
        """CheckPlanRequest should contain PlanGraphProto."""
        plan = plan_pb2.PlanGraphProto(
            plan_id="plan_001",
            goal_text="Navigate to target",
        )
        request = safety_pb2.CheckPlanRequest(
            plan=plan,
            profile_name="agi_hpc_safety_v1",
            require_proofs=True,
        )

        assert request.plan.plan_id == "plan_001"
        assert request.profile_name == "agi_hpc_safety_v1"
        assert request.require_proofs is True


class TestCheckPlanResponse:
    """Tests for CheckPlanResponse message."""

    def test_check_plan_response_approved(self):
        """CheckPlanResponse should represent approved plan."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            risk_score=0.15,
        )
        response = safety_pb2.CheckPlanResponse(
            result=result,
            plan_approved=True,
        )

        assert response.plan_approved is True
        assert response.result.decision == safety_pb2.SAFETY_DECISION_ALLOW

    def test_check_plan_response_with_step_results(self):
        """CheckPlanResponse should contain per-step results."""
        response = safety_pb2.CheckPlanResponse(plan_approved=True)

        step1_result = safety_pb2.StepSafetyResult(
            step_id="step_1",
            result=safety_pb2.SafetyResult(
                decision=safety_pb2.SAFETY_DECISION_ALLOW,
                risk_score=0.1,
            ),
        )
        step2_result = safety_pb2.StepSafetyResult(
            step_id="step_2",
            result=safety_pb2.SafetyResult(
                decision=safety_pb2.SAFETY_DECISION_ALLOW,
                risk_score=0.2,
            ),
        )

        response.step_results.append(step1_result)
        response.step_results.append(step2_result)

        assert len(response.step_results) == 2
        assert response.step_results[0].step_id == "step_1"
        assert response.step_results[1].result.risk_score == pytest.approx(0.2)


class TestCheckActionRequest:
    """Tests for CheckActionRequest message."""

    def test_check_action_request_with_step(self):
        """CheckActionRequest should contain plan step."""
        step = plan_pb2.PlanStep(
            step_id="step_001",
            kind="action",
            description="Grasp object",
            tool_id="grasp",
        )
        request = safety_pb2.CheckActionRequest(
            plan_id="plan_001",
            step_id="step_001",
            step=step,
            world_state_ref="world_snapshot_42",
        )

        assert request.plan_id == "plan_001"
        assert request.step_id == "step_001"
        assert request.step.tool_id == "grasp"
        assert request.world_state_ref == "world_snapshot_42"

    def test_check_action_request_sensor_readings(self):
        """CheckActionRequest should support sensor readings map."""
        request = safety_pb2.CheckActionRequest(
            plan_id="plan_001",
            step_id="step_001",
        )
        request.sensor_readings["proximity"] = 0.15
        request.sensor_readings["force"] = 2.5

        assert request.sensor_readings["proximity"] == pytest.approx(0.15)
        assert request.sensor_readings["force"] == pytest.approx(2.5)


class TestCheckActionResponse:
    """Tests for CheckActionResponse message."""

    def test_check_action_response_safe(self):
        """CheckActionResponse should represent safe action."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            risk_score=0.1,
        )
        response = safety_pb2.CheckActionResponse(
            result=result,
            emergency_stop=False,
        )

        assert response.emergency_stop is False
        assert response.result.decision == safety_pb2.SAFETY_DECISION_ALLOW

    def test_check_action_response_emergency_stop(self):
        """CheckActionResponse should trigger emergency stop."""
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_BLOCK,
            risk_score=1.0,
        )
        response = safety_pb2.CheckActionResponse(
            result=result,
            emergency_stop=True,
            stop_reason="Imminent collision detected",
        )

        assert response.emergency_stop is True
        assert response.stop_reason == "Imminent collision detected"


class TestReflexCheckRequest:
    """Tests for ReflexCheckRequest message (fast safety layer)."""

    def test_reflex_check_request_safe(self):
        """ReflexCheckRequest should represent safe state."""
        request = safety_pb2.ReflexCheckRequest(
            physical_harm_risk=0.1,
            collision_probability=0.05,
            emergency_flag=False,
        )

        assert request.physical_harm_risk == pytest.approx(0.1)
        assert request.collision_probability == pytest.approx(0.05)
        assert request.emergency_flag is False

    def test_reflex_check_request_emergency(self):
        """ReflexCheckRequest should flag emergency."""
        request = safety_pb2.ReflexCheckRequest(
            physical_harm_risk=0.95,
            collision_probability=0.9,
            emergency_flag=True,
        )

        assert request.emergency_flag is True
        assert request.physical_harm_risk == pytest.approx(0.95)


class TestReflexCheckResponse:
    """Tests for ReflexCheckResponse message."""

    def test_reflex_check_response_safe(self):
        """ReflexCheckResponse should indicate safe state."""
        response = safety_pb2.ReflexCheckResponse(
            safe=True,
            emergency_stop=False,
        )

        assert response.safe is True
        assert response.emergency_stop is False

    def test_reflex_check_response_emergency_stop(self):
        """ReflexCheckResponse should trigger emergency stop."""
        response = safety_pb2.ReflexCheckResponse(
            safe=False,
            emergency_stop=True,
            reason="Collision imminent",
        )

        assert response.safe is False
        assert response.emergency_stop is True
        assert response.reason == "Collision imminent"


class TestActionOutcomeReport:
    """Tests for ActionOutcomeReport message (post-action learning)."""

    def test_action_outcome_report_success(self):
        """ActionOutcomeReport should represent successful action."""
        report = safety_pb2.ActionOutcomeReport(
            plan_id="plan_001",
            step_id="step_003",
            outcome_description="Object grasped successfully",
            success=True,
            predicted_risk=0.2,
            actual_harm=0.0,
            decision_proof_hash="abc123",
        )

        assert report.plan_id == "plan_001"
        assert report.success is True
        assert report.predicted_risk == pytest.approx(0.2)
        assert report.actual_harm == pytest.approx(0.0)
        assert report.decision_proof_hash == "abc123"

    def test_action_outcome_report_failure(self):
        """ActionOutcomeReport should represent failed action."""
        report = safety_pb2.ActionOutcomeReport(
            plan_id="plan_001",
            step_id="step_004",
            outcome_description="Collision occurred",
            success=False,
            predicted_risk=0.3,
            actual_harm=0.6,
        )

        assert report.success is False
        assert report.actual_harm == pytest.approx(0.6)


class TestActionOutcomeAck:
    """Tests for ActionOutcomeAck message."""

    def test_action_outcome_ack_logged(self):
        """ActionOutcomeAck should confirm logging."""
        ack = safety_pb2.ActionOutcomeAck(
            event_id="eth_plan_001_step_003_1234567890",
            logged=True,
        )

        assert ack.event_id.startswith("eth_")
        assert ack.logged is True


class TestEmergencyStopRequest:
    """Tests for EmergencyStopRequest message."""

    def test_emergency_stop_request(self):
        """EmergencyStopRequest should contain reason and source."""
        request = safety_pb2.EmergencyStopRequest(
            reason="Human in workspace detected",
            source="proximity_sensor",
        )

        assert request.reason == "Human in workspace detected"
        assert request.source == "proximity_sensor"


class TestEmergencyStopResponse:
    """Tests for EmergencyStopResponse message."""

    def test_emergency_stop_response_acknowledged(self):
        """EmergencyStopResponse should confirm acknowledgment."""
        response = safety_pb2.EmergencyStopResponse(
            acknowledged=True,
            stop_id="estop_001",
        )

        assert response.acknowledged is True
        assert response.stop_id == "estop_001"


class TestSafetyProtoIntegration:
    """Integration tests for Safety ↔ ErisML proto interop."""

    def test_safety_result_with_full_erisml_integration(self):
        """SafetyResult should integrate with full ErisML decision output."""
        # Create ErisML components
        bond_index = erisml_pb2.BondIndexResultProto(
            bond_index=0.12,
            baseline=0.155,
            within_threshold=True,
        )

        proof = erisml_pb2.DecisionProofProto(
            decision_id="dec_integration_001",
            timestamp="2025-01-15T12:00:00Z",
            profile_name="agi_hpc_safety_v1",
            proof_hash="integration_hash_001",
        )

        # Create SafetyResult with ErisML integration
        result = safety_pb2.SafetyResult(
            decision=safety_pb2.SAFETY_DECISION_ALLOW,
            risk_score=0.15,
            bond_index=bond_index,
            decision_proof=proof,
        )
        result.reasons.append("All ethical checks passed")
        result.categories.append(safety_pb2.RISK_CATEGORY_BOND_INDEX)
        result.metadata["erisml_latency_ms"] = "45"

        # Verify integration
        assert result.bond_index.bond_index == pytest.approx(0.12)
        assert result.decision_proof.profile_name == "agi_hpc_safety_v1"
        assert result.metadata["erisml_latency_ms"] == "45"

    def test_check_plan_full_workflow(self):
        """Test complete CheckPlan workflow with nested messages."""
        # Create plan with steps
        plan = plan_pb2.PlanGraphProto(
            plan_id="workflow_plan_001",
            goal_text="Pick and place object",
        )
        step1 = plan_pb2.PlanStep(
            step_id="step_1",
            kind="action",
            description="Navigate to object",
        )
        step2 = plan_pb2.PlanStep(
            step_id="step_2",
            kind="action",
            description="Grasp object",
        )
        plan.steps.extend([step1, step2])

        # Create request
        request = safety_pb2.CheckPlanRequest(
            plan=plan,
            profile_name="agi_hpc_safety_v1",
            require_proofs=True,
        )

        # Simulate response
        response = safety_pb2.CheckPlanResponse(plan_approved=True)
        for step in plan.steps:
            step_result = safety_pb2.StepSafetyResult(
                step_id=step.step_id,
                result=safety_pb2.SafetyResult(
                    decision=safety_pb2.SAFETY_DECISION_ALLOW,
                    risk_score=0.1,
                ),
            )
            response.step_results.append(step_result)

        # Verify workflow
        assert request.plan.plan_id == "workflow_plan_001"
        assert len(request.plan.steps) == 2
        assert response.plan_approved is True
        assert len(response.step_results) == 2
