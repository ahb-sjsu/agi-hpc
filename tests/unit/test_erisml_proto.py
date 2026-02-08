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

"""Tests for ErisML protobuf messages and service definitions."""

import pytest

from agi.proto_gen import erisml_pb2


class TestEthicalFactsProto:
    """Tests for EthicalFactsProto message."""

    def test_ethical_facts_default_values(self):
        """EthicalFactsProto should have sensible defaults."""
        facts = erisml_pb2.EthicalFactsProto()
        assert facts.option_id == ""
        assert facts.expected_benefit == 0.0
        assert facts.expected_harm == 0.0
        assert facts.violates_rights is False

    def test_ethical_facts_set_consequences(self):
        """EthicalFactsProto should store consequence dimensions."""
        facts = erisml_pb2.EthicalFactsProto(
            option_id="step_001",
            expected_benefit=0.8,
            expected_harm=0.1,
            urgency=0.5,
            affected_count=3,
        )
        assert facts.option_id == "step_001"
        assert facts.expected_benefit == pytest.approx(0.8)
        assert facts.expected_harm == pytest.approx(0.1)
        assert facts.urgency == pytest.approx(0.5)
        assert facts.affected_count == 3

    def test_ethical_facts_set_rights_and_duties(self):
        """EthicalFactsProto should store rights/duties dimensions."""
        facts = erisml_pb2.EthicalFactsProto(
            violates_rights=True,
            has_valid_consent=False,
            violates_explicit_rule=True,
        )
        assert facts.violates_rights is True
        assert facts.has_valid_consent is False
        assert facts.violates_explicit_rule is True

    def test_ethical_facts_set_safety(self):
        """EthicalFactsProto should store safety dimensions."""
        facts = erisml_pb2.EthicalFactsProto(
            physical_harm_risk=0.3,
            collision_probability=0.15,
        )
        assert facts.physical_harm_risk == pytest.approx(0.3)
        assert facts.collision_probability == pytest.approx(0.15)

    def test_ethical_facts_set_epistemic(self):
        """EthicalFactsProto should store epistemic dimensions."""
        facts = erisml_pb2.EthicalFactsProto(
            uncertainty_level=0.7,
            evidence_quality=0.4,
            novel_situation=True,
        )
        assert facts.uncertainty_level == pytest.approx(0.7)
        assert facts.evidence_quality == pytest.approx(0.4)
        assert facts.novel_situation is True

    def test_ethical_facts_extensions_map(self):
        """EthicalFactsProto should support domain-specific extensions."""
        facts = erisml_pb2.EthicalFactsProto(option_id="test")
        facts.extensions["custom_metric"] = 0.42
        facts.extensions["domain_score"] = 0.85

        assert facts.extensions["custom_metric"] == pytest.approx(0.42)
        assert facts.extensions["domain_score"] == pytest.approx(0.85)

    def test_ethical_facts_serialization_roundtrip(self):
        """EthicalFactsProto should serialize and deserialize correctly."""
        original = erisml_pb2.EthicalFactsProto(
            option_id="roundtrip_test",
            expected_benefit=0.9,
            physical_harm_risk=0.2,
            novel_situation=True,
        )

        serialized = original.SerializeToString()
        restored = erisml_pb2.EthicalFactsProto()
        restored.ParseFromString(serialized)

        assert restored.option_id == "roundtrip_test"
        assert restored.expected_benefit == pytest.approx(0.9)
        assert restored.physical_harm_risk == pytest.approx(0.2)
        assert restored.novel_situation is True


class TestMoralVectorProto:
    """Tests for MoralVectorProto message."""

    def test_moral_vector_8_dimensions(self):
        """MoralVectorProto should have 8 core ethical dimensions."""
        mv = erisml_pb2.MoralVectorProto(
            physical_harm=0.1,
            rights_respect=0.9,
            fairness_equity=0.8,
            autonomy_respect=0.85,
            privacy_protection=0.7,
            societal_environmental=0.6,
            virtue_care=0.75,
            legitimacy_trust=0.8,
        )
        assert mv.physical_harm == pytest.approx(0.1)
        assert mv.rights_respect == pytest.approx(0.9)
        assert mv.fairness_equity == pytest.approx(0.8)
        assert mv.autonomy_respect == pytest.approx(0.85)
        assert mv.privacy_protection == pytest.approx(0.7)
        assert mv.societal_environmental == pytest.approx(0.6)
        assert mv.virtue_care == pytest.approx(0.75)
        assert mv.legitimacy_trust == pytest.approx(0.8)

    def test_moral_vector_epistemic_quality(self):
        """MoralVectorProto should have epistemic quality (+1 dimension)."""
        mv = erisml_pb2.MoralVectorProto(epistemic_quality=0.95)
        assert mv.epistemic_quality == pytest.approx(0.95)

    def test_moral_vector_veto_flags(self):
        """MoralVectorProto should support veto flags from reflex layer."""
        mv = erisml_pb2.MoralVectorProto()
        mv.veto_flags.append("PHYSICAL_HARM")
        mv.veto_flags.append("RIGHTS_VIOLATION")

        assert len(mv.veto_flags) == 2
        assert "PHYSICAL_HARM" in mv.veto_flags
        assert "RIGHTS_VIOLATION" in mv.veto_flags

    def test_moral_vector_reason_codes(self):
        """MoralVectorProto should support reason codes."""
        mv = erisml_pb2.MoralVectorProto()
        mv.reason_codes.append("high_collision_risk")
        mv.reason_codes.append("consent_not_obtained")

        assert len(mv.reason_codes) == 2


class TestHohfeldianVerdictProto:
    """Tests for HohfeldianVerdictProto message."""

    def test_hohfeldian_verdict_obligation(self):
        """HohfeldianVerdictProto should represent Obligation state."""
        verdict = erisml_pb2.HohfeldianVerdictProto(
            party_name="Agent",
            state="O",
            expected_state="O",
            confidence=0.95,
        )
        assert verdict.party_name == "Agent"
        assert verdict.state == "O"
        assert verdict.expected_state == "O"
        assert verdict.confidence == pytest.approx(0.95)

    def test_hohfeldian_verdict_claim(self):
        """HohfeldianVerdictProto should represent Claim state."""
        verdict = erisml_pb2.HohfeldianVerdictProto(
            party_name="Patient",
            state="C",
            confidence=0.9,
        )
        assert verdict.state == "C"

    def test_hohfeldian_verdict_liberty(self):
        """HohfeldianVerdictProto should represent Liberty state."""
        verdict = erisml_pb2.HohfeldianVerdictProto(
            party_name="User",
            state="L",
            confidence=0.8,
        )
        assert verdict.state == "L"

    def test_hohfeldian_verdict_no_claim(self):
        """HohfeldianVerdictProto should represent No-claim state."""
        verdict = erisml_pb2.HohfeldianVerdictProto(
            party_name="Observer",
            state="N",
            confidence=0.85,
        )
        assert verdict.state == "N"


class TestBondIndexResultProto:
    """Tests for BondIndexResultProto message."""

    def test_bond_index_perfect_symmetry(self):
        """BondIndexResultProto should represent perfect symmetry (0)."""
        result = erisml_pb2.BondIndexResultProto(
            bond_index=0.0,
            baseline=0.155,
            within_threshold=True,
        )
        assert result.bond_index == pytest.approx(0.0)
        assert result.baseline == pytest.approx(0.155)
        assert result.within_threshold is True

    def test_bond_index_at_baseline(self):
        """BondIndexResultProto should handle Dear Abby baseline (0.155)."""
        result = erisml_pb2.BondIndexResultProto(
            bond_index=0.155,
            baseline=0.155,
            within_threshold=True,
        )
        assert result.bond_index == pytest.approx(0.155)
        assert result.within_threshold is True

    def test_bond_index_exceeds_threshold(self):
        """BondIndexResultProto should flag when threshold exceeded."""
        result = erisml_pb2.BondIndexResultProto(
            bond_index=0.35,
            baseline=0.155,
            within_threshold=False,
        )
        assert result.bond_index == pytest.approx(0.35)
        assert result.within_threshold is False

    def test_bond_index_violations(self):
        """BondIndexResultProto should list symmetry violations."""
        result = erisml_pb2.BondIndexResultProto(
            bond_index=0.25,
            baseline=0.155,
            within_threshold=True,
        )
        result.violations.append("O↔C asymmetry at position 3")
        result.violations.append("L↔N mismatch at position 7")

        assert len(result.violations) == 2


class TestDecisionProofProto:
    """Tests for DecisionProofProto message (audit trail)."""

    def test_decision_proof_basic_fields(self):
        """DecisionProofProto should store basic decision info."""
        proof = erisml_pb2.DecisionProofProto(
            decision_id="dec_001",
            timestamp="2025-01-15T10:30:00Z",
            profile_name="agi_hpc_safety_v1",
            confidence=0.92,
        )
        assert proof.decision_id == "dec_001"
        assert proof.timestamp == "2025-01-15T10:30:00Z"
        assert proof.profile_name == "agi_hpc_safety_v1"
        assert proof.confidence == pytest.approx(0.92)

    def test_decision_proof_hash_chain(self):
        """DecisionProofProto should support hash chain for audit."""
        proof = erisml_pb2.DecisionProofProto(
            decision_id="dec_002",
            previous_proof_hash="abc123",
            proof_hash="def456",
        )
        assert proof.previous_proof_hash == "abc123"
        assert proof.proof_hash == "def456"

    def test_decision_proof_candidate_options(self):
        """DecisionProofProto should store candidate and selected options."""
        proof = erisml_pb2.DecisionProofProto(
            decision_id="dec_003",
            selected_option_id="option_b",
        )
        proof.candidate_option_ids.extend(["option_a", "option_b", "option_c"])
        proof.ranked_options.extend(["option_b", "option_a", "option_c"])
        proof.forbidden_options.append("option_d")

        assert len(proof.candidate_option_ids) == 3
        assert proof.selected_option_id == "option_b"
        assert proof.ranked_options[0] == "option_b"
        assert "option_d" in proof.forbidden_options


class TestEvaluateStepRequest:
    """Tests for EvaluateStepRequest message."""

    def test_evaluate_step_request_with_facts(self):
        """EvaluateStepRequest should contain EthicalFacts and profile."""
        facts = erisml_pb2.EthicalFactsProto(
            option_id="step_001",
            expected_benefit=0.7,
        )
        request = erisml_pb2.EvaluateStepRequest(
            facts=facts,
            profile_name="agi_hpc_safety_v1",
        )
        assert request.facts.option_id == "step_001"
        assert request.profile_name == "agi_hpc_safety_v1"


class TestEvaluateStepResponse:
    """Tests for EvaluateStepResponse message."""

    def test_evaluate_step_response_approved(self):
        """EvaluateStepResponse should represent approved step."""
        response = erisml_pb2.EvaluateStepResponse(
            verdict="prefer",
            vetoed=False,
        )
        assert response.verdict == "prefer"
        assert response.vetoed is False

    def test_evaluate_step_response_vetoed(self):
        """EvaluateStepResponse should represent vetoed step."""
        response = erisml_pb2.EvaluateStepResponse(
            verdict="forbid",
            vetoed=True,
            veto_reason="Physical harm risk exceeds threshold",
        )
        assert response.verdict == "forbid"
        assert response.vetoed is True
        assert "Physical harm" in response.veto_reason


class TestEvaluatePlanRequest:
    """Tests for EvaluatePlanRequest message."""

    def test_evaluate_plan_request_multiple_steps(self):
        """EvaluatePlanRequest should contain multiple step facts."""
        request = erisml_pb2.EvaluatePlanRequest(
            profile_name="agi_hpc_safety_v1",
            generate_proofs=True,
        )
        request.step_facts.append(erisml_pb2.EthicalFactsProto(option_id="step_1"))
        request.step_facts.append(erisml_pb2.EthicalFactsProto(option_id="step_2"))

        assert len(request.step_facts) == 2
        assert request.generate_proofs is True


class TestEvaluatePlanResponse:
    """Tests for EvaluatePlanResponse message."""

    def test_evaluate_plan_response_approved(self):
        """EvaluatePlanResponse should represent approved plan."""
        response = erisml_pb2.EvaluatePlanResponse(plan_approved=True)
        assert response.plan_approved is True
        assert len(response.blocked_steps) == 0

    def test_evaluate_plan_response_with_blocks(self):
        """EvaluatePlanResponse should list blocked steps."""
        response = erisml_pb2.EvaluatePlanResponse(plan_approved=False)
        response.blocked_steps.extend(["step_3", "step_7"])

        assert response.plan_approved is False
        assert len(response.blocked_steps) == 2


class TestBondIndexRequest:
    """Tests for BondIndexRequest message."""

    def test_bond_index_request_correlative_pairs(self):
        """BondIndexRequest should contain party A and B verdicts."""
        request = erisml_pb2.BondIndexRequest()

        # Party A has Obligation
        request.party_a_verdicts.append(
            erisml_pb2.HohfeldianVerdictProto(party_name="Agent", state="O")
        )
        # Party B should have correlative Claim
        request.party_b_verdicts.append(
            erisml_pb2.HohfeldianVerdictProto(party_name="Patient", state="C")
        )

        assert len(request.party_a_verdicts) == 1
        assert len(request.party_b_verdicts) == 1
        assert request.party_a_verdicts[0].state == "O"
        assert request.party_b_verdicts[0].state == "C"


class TestHohfeldianRequest:
    """Tests for HohfeldianRequest message."""

    def test_hohfeldian_request_multiple_verdicts(self):
        """HohfeldianRequest should contain multiple verdicts for verification."""
        request = erisml_pb2.HohfeldianRequest()
        request.verdicts.append(
            erisml_pb2.HohfeldianVerdictProto(party_name="A", state="O")
        )
        request.verdicts.append(
            erisml_pb2.HohfeldianVerdictProto(party_name="B", state="C")
        )

        assert len(request.verdicts) == 2


class TestHohfeldianResponse:
    """Tests for HohfeldianResponse message."""

    def test_hohfeldian_response_consistent(self):
        """HohfeldianResponse should indicate consistency."""
        response = erisml_pb2.HohfeldianResponse(
            consistent=True,
            symmetry_rate=1.0,
        )
        assert response.consistent is True
        assert response.symmetry_rate == pytest.approx(1.0)

    def test_hohfeldian_response_violations(self):
        """HohfeldianResponse should list violations when inconsistent."""
        response = erisml_pb2.HohfeldianResponse(
            consistent=False,
            symmetry_rate=0.75,
        )
        response.violations.append("Missing correlative for O at position 2")

        assert response.consistent is False
        assert len(response.violations) == 1
