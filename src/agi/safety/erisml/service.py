# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
ErisML gRPC service implementation.

Reference implementation bridging AGI-HPC plan evaluation to ErisML
ethical reasoning framework. This module provides:

- ErisMLServicer: gRPC servicer implementing ErisMLService
- Ethical evaluation of plan steps through moral vector computation
- Bond Index calculation for Hohfeldian symmetry verification
- Decision proof generation for audit trails

Usage:
    server = create_erisml_server(port=50060)
    server.start()
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent import futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import grpc

from agi.proto_gen import erisml_pb2, erisml_pb2_grpc
from agi.safety.erisml.hohfeld import (
    HohfeldianState,
    HohfeldianVerdict,
    compute_bond_index,
    correlative,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bond Index thresholds (empirically derived from Dear Abby corpus)
BOND_INDEX_BASELINE = 0.155
BOND_INDEX_WARNING_THRESHOLD = 0.25
BOND_INDEX_BLOCK_THRESHOLD = 0.30

# Hohfeldian correlative pairs
CORRELATIVES = {
    "O": "C",  # Obligation <-> Claim
    "C": "O",
    "L": "N",  # Liberty <-> No-claim
    "N": "L",
}

# Verdict thresholds for moral vector aggregation
VERDICT_THRESHOLDS = {
    "forbid": 0.0,
    "avoid": 0.3,
    "neutral": 0.5,
    "prefer": 0.7,
    "strongly_prefer": 0.9,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class MoralVector:
    """
    8+1 dimensional ethical assessment.

    Dimensions:
        1. physical_harm: Risk of physical harm (0=safe, 1=dangerous)
        2. rights_respect: Respect for rights (0=violates, 1=respects)
        3. fairness_equity: Fairness and equity (0=unfair, 1=fair)
        4. autonomy_respect: Respect for autonomy (0=coercive, 1=respects)
        5. privacy_protection: Privacy protection (0=violates, 1=protects)
        6. societal_environmental: Societal/environmental impact (0=harmful, 1=beneficial)
        7. virtue_care: Virtue ethics / care dimension (0=vicious, 1=virtuous)
        8. legitimacy_trust: Procedural legitimacy (0=illegitimate, 1=legitimate)
        +1. epistemic_quality: Confidence in assessment (0=uncertain, 1=certain)
    """

    physical_harm: float = 0.0
    rights_respect: float = 1.0
    fairness_equity: float = 1.0
    autonomy_respect: float = 1.0
    privacy_protection: float = 1.0
    societal_environmental: float = 1.0
    virtue_care: float = 1.0
    legitimacy_trust: float = 1.0
    epistemic_quality: float = 0.5
    veto_flags: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_tensor(self) -> "MoralTensor":  # noqa: F821
        """Convert to rank-1 MoralTensor for tensor-based evaluation."""
        from agi.safety.erisml.moral_tensor import MoralTensor

        return MoralTensor.from_moral_vector(self)

    def aggregate_score(self) -> float:
        """Compute aggregate ethical score (higher = more ethical)."""
        # Invert physical_harm (lower harm = higher score)
        harm_score = 1.0 - self.physical_harm

        # Weight physical harm and rights more heavily
        weighted = (
            harm_score * 2.0
            + self.rights_respect * 2.0
            + self.fairness_equity * 1.0
            + self.autonomy_respect * 1.0
            + self.privacy_protection * 1.0
            + self.societal_environmental * 1.0
            + self.virtue_care * 1.0
            + self.legitimacy_trust * 1.0
        ) / 10.0

        return weighted

    def to_proto(self) -> erisml_pb2.MoralVectorProto:
        """Convert to protobuf message."""
        proto = erisml_pb2.MoralVectorProto(
            physical_harm=self.physical_harm,
            rights_respect=self.rights_respect,
            fairness_equity=self.fairness_equity,
            autonomy_respect=self.autonomy_respect,
            privacy_protection=self.privacy_protection,
            societal_environmental=self.societal_environmental,
            virtue_care=self.virtue_care,
            legitimacy_trust=self.legitimacy_trust,
            epistemic_quality=self.epistemic_quality,
        )
        proto.veto_flags.extend(self.veto_flags)
        proto.reason_codes.extend(self.reason_codes)
        return proto


@dataclass
class EvaluationResult:
    """Result of ethical evaluation."""

    verdict: str
    moral_vector: MoralVector
    vetoed: bool = False
    veto_reason: str = ""
    proof_hash: str = ""


# ---------------------------------------------------------------------------
# ErisML Servicer
# ---------------------------------------------------------------------------


class ErisMLServicer(erisml_pb2_grpc.ErisMLServiceServicer):
    """
    gRPC servicer implementing ErisML ethical evaluation.

    This is a reference implementation that demonstrates the integration
    pattern. In production, this would connect to the full ErisML DEME
    pipeline for sophisticated ethical reasoning.
    """

    def __init__(self, default_profile: str = "agi_hpc_safety_v1"):
        """
        Initialize the ErisML servicer.

        Args:
            default_profile: Default DEME profile name to use
        """
        self.default_profile = default_profile
        self._proof_chain: List[str] = []
        logger.info(f"ErisMLServicer initialized with profile: {default_profile}")

    def EvaluateStep(
        self,
        request: erisml_pb2.EvaluateStepRequest,
        context: grpc.ServicerContext,
    ) -> erisml_pb2.EvaluateStepResponse:
        """
        Evaluate a single plan step through ethical reasoning.

        Args:
            request: EvaluateStepRequest with EthicalFacts and profile name
            context: gRPC context

        Returns:
            EvaluateStepResponse with verdict, moral vector, and proof
        """
        start_time = time.monotonic()
        profile = request.profile_name or self.default_profile

        # Compute moral vector from ethical facts
        moral_vector = self._compute_moral_vector(request.facts)

        # Determine verdict
        result = self._determine_verdict(moral_vector, request.facts)

        # Generate decision proof
        proof = self._generate_proof(
            decision_id=f"step_{request.facts.option_id}_{int(time.time())}",
            facts=request.facts,
            profile=profile,
            result=result,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            f"EvaluateStep completed in {elapsed_ms:.2f}ms: "
            f"verdict={result.verdict}, vetoed={result.vetoed}"
        )

        return erisml_pb2.EvaluateStepResponse(
            verdict=result.verdict,
            moral_vector=result.moral_vector.to_proto(),
            vetoed=result.vetoed,
            veto_reason=result.veto_reason,
            proof=proof,
        )

    def EvaluatePlan(
        self,
        request: erisml_pb2.EvaluatePlanRequest,
        context: grpc.ServicerContext,
    ) -> erisml_pb2.EvaluatePlanResponse:
        """
        Evaluate an entire plan through ethical reasoning.

        Args:
            request: EvaluatePlanRequest with step facts and profile name
            context: gRPC context

        Returns:
            EvaluatePlanResponse with per-step results and Bond Index
        """
        start_time = time.monotonic()
        profile = request.profile_name or self.default_profile

        step_results = []
        blocked_steps = []

        for facts in request.step_facts:
            # Evaluate each step
            moral_vector = self._compute_moral_vector(facts)
            result = self._determine_verdict(moral_vector, facts)

            step_response = erisml_pb2.EvaluateStepResponse(
                verdict=result.verdict,
                moral_vector=result.moral_vector.to_proto(),
                vetoed=result.vetoed,
                veto_reason=result.veto_reason,
            )

            if request.generate_proofs:
                proof = self._generate_proof(
                    decision_id=f"step_{facts.option_id}_{int(time.time())}",
                    facts=facts,
                    profile=profile,
                    result=result,
                )
                step_response.proof.CopyFrom(proof)

            step_results.append(step_response)

            if result.vetoed:
                blocked_steps.append(facts.option_id)

        # Compute aggregate Bond Index for the plan
        bond_index = self._compute_plan_bond_index(step_results)

        # Generate plan-level proof if requested
        plan_proof = None
        if request.generate_proofs:
            plan_proof = erisml_pb2.DecisionProofProto(
                decision_id=f"plan_{int(time.time())}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                profile_name=profile,
                confidence=bond_index.bond_index,
            )
            plan_proof.candidate_option_ids.extend(
                [f.option_id for f in request.step_facts]
            )
            plan_proof.forbidden_options.extend(blocked_steps)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            f"EvaluatePlan completed in {elapsed_ms:.2f}ms: "
            f"steps={len(step_results)}, blocked={len(blocked_steps)}, "
            f"bond_index={bond_index.bond_index:.3f}"
        )

        response = erisml_pb2.EvaluatePlanResponse(
            plan_approved=len(blocked_steps) == 0,
        )
        response.step_results.extend(step_results)
        response.blocked_steps.extend(blocked_steps)
        response.bond_index.CopyFrom(bond_index)
        if plan_proof:
            response.plan_proof.CopyFrom(plan_proof)

        return response

    def ComputeBondIndex(
        self,
        request: erisml_pb2.BondIndexRequest,
        context: grpc.ServicerContext,
    ) -> erisml_pb2.BondIndexResultProto:
        """
        Compute Bond Index between two parties' Hohfeldian verdicts.

        The Bond Index measures deviation from perfect correlative symmetry
        in Hohfeldian normative positions (O↔C, L↔N).

        Args:
            request: BondIndexRequest with party A and B verdicts
            context: gRPC context

        Returns:
            BondIndexResultProto with index value and threshold status
        """
        # Convert proto verdicts to typed HohfeldianVerdict objects
        verdicts_a: List[HohfeldianVerdict] = []
        verdicts_b: List[HohfeldianVerdict] = []
        violations: List[str] = []

        for v_a, v_b in zip(
            request.party_a_verdicts, request.party_b_verdicts, strict=False
        ):
            hv_a = HohfeldianVerdict(
                party_name=v_a.party_name,
                state=HohfeldianState(v_a.state),
                confidence=v_a.confidence,
            )
            hv_b = HohfeldianVerdict(
                party_name=v_b.party_name,
                state=HohfeldianState(v_b.state),
                confidence=v_b.confidence,
            )
            verdicts_a.append(hv_a)
            verdicts_b.append(hv_b)

            expected = correlative(hv_a.state)
            if hv_b.state != expected:
                violations.append(
                    f"{v_a.party_name}({v_a.state}) <-> "
                    f"{v_b.party_name}({v_b.state}): "
                    f"expected {expected.value}"
                )

        # Compute Bond Index using D4 correlative symmetry
        bond_idx = compute_bond_index(verdicts_a, verdicts_b, tau=1.0)

        result = erisml_pb2.BondIndexResultProto(
            bond_index=bond_idx,
            baseline=BOND_INDEX_BASELINE,
            within_threshold=bond_idx < BOND_INDEX_BLOCK_THRESHOLD,
        )
        result.violations.extend(violations)

        return result

    def VerifyHohfeldian(
        self,
        request: erisml_pb2.HohfeldianRequest,
        context: grpc.ServicerContext,
    ) -> erisml_pb2.HohfeldianResponse:
        """
        Verify Hohfeldian correlative symmetry in a set of verdicts.

        Args:
            request: HohfeldianRequest with verdicts to verify
            context: gRPC context

        Returns:
            HohfeldianResponse with consistency status and violations
        """
        violations = []

        # Group verdicts by state using D4-aware Hohfeldian states
        by_state: Dict[str, List[erisml_pb2.HohfeldianVerdictProto]] = {}
        for v in request.verdicts:
            by_state.setdefault(v.state, []).append(v)

        # Check correlative symmetry: O<->C (via s-reflection)
        o_count = len(by_state.get(HohfeldianState.O.value, []))
        c_count = len(by_state.get(HohfeldianState.C.value, []))
        if o_count != c_count:
            violations.append(
                f"O<->C asymmetry: {o_count} Obligations, {c_count} Claims"
            )

        # Check correlative symmetry: L<->N (via s-reflection)
        l_count = len(by_state.get(HohfeldianState.L.value, []))
        n_count = len(by_state.get(HohfeldianState.N.value, []))
        if l_count != n_count:
            violations.append(
                f"L<->N asymmetry: {l_count} Liberties, {n_count} No-claims"
            )

        # Compute symmetry rate
        total = o_count + c_count + l_count + n_count
        if total > 0:
            symmetric = min(o_count, c_count) * 2 + min(l_count, n_count) * 2
            symmetry_rate = symmetric / total
        else:
            symmetry_rate = 1.0

        response = erisml_pb2.HohfeldianResponse(
            consistent=len(violations) == 0,
            symmetry_rate=symmetry_rate,
        )
        response.violations.extend(violations)

        return response

    # -----------------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------------

    def _compute_moral_vector(self, facts: erisml_pb2.EthicalFactsProto) -> MoralVector:
        """Compute moral vector from ethical facts."""
        mv = MoralVector()

        # Physical harm dimension
        mv.physical_harm = max(facts.physical_harm_risk, facts.collision_probability)
        if mv.physical_harm > 0.8:
            mv.veto_flags.append("PHYSICAL_HARM")
            mv.reason_codes.append("physical_harm_exceeds_threshold")

        # Rights respect dimension
        if facts.violates_rights:
            mv.rights_respect = 0.0
            mv.veto_flags.append("RIGHTS_VIOLATION")
            mv.reason_codes.append("explicit_rights_violation")
        elif facts.violates_explicit_rule:
            mv.rights_respect = 0.2
            mv.reason_codes.append("rule_violation")
        else:
            mv.rights_respect = 1.0 if facts.has_valid_consent else 0.7

        # Fairness dimension
        if facts.discriminates_on_protected_attr:
            mv.fairness_equity = 0.0
            mv.veto_flags.append("DISCRIMINATION")
            mv.reason_codes.append("protected_attribute_discrimination")
        elif facts.exploits_vulnerable_population:
            mv.fairness_equity = 0.2
            mv.reason_codes.append("vulnerable_population_exploitation")
        else:
            mv.fairness_equity = 1.0

        # Autonomy dimension (based on consent)
        mv.autonomy_respect = 1.0 if facts.has_valid_consent else 0.5

        # Epistemic quality
        mv.epistemic_quality = facts.evidence_quality * (1.0 - facts.uncertainty_level)
        if facts.novel_situation:
            mv.epistemic_quality *= 0.7
            mv.reason_codes.append("novel_situation_uncertainty")

        # Consequentialist adjustment
        benefit_harm_ratio = (
            facts.expected_benefit / max(facts.expected_harm, 0.01)
            if facts.expected_harm > 0
            else facts.expected_benefit * 10
        )
        mv.societal_environmental = min(1.0, benefit_harm_ratio / 10.0)

        return mv

    def _determine_verdict(
        self,
        moral_vector: MoralVector,
        facts: erisml_pb2.EthicalFactsProto,
    ) -> EvaluationResult:
        """Determine verdict from moral vector."""
        # Check for hard vetoes
        if moral_vector.veto_flags:
            return EvaluationResult(
                verdict="forbid",
                moral_vector=moral_vector,
                vetoed=True,
                veto_reason=f"Veto triggered: {', '.join(moral_vector.veto_flags)}",
            )

        # Compute aggregate score
        score = moral_vector.aggregate_score()

        # Determine verdict based on score
        if score >= VERDICT_THRESHOLDS["strongly_prefer"]:
            verdict = "strongly_prefer"
        elif score >= VERDICT_THRESHOLDS["prefer"]:
            verdict = "prefer"
        elif score >= VERDICT_THRESHOLDS["neutral"]:
            verdict = "neutral"
        elif score >= VERDICT_THRESHOLDS["avoid"]:
            verdict = "avoid"
        else:
            verdict = "forbid"

        return EvaluationResult(
            verdict=verdict,
            moral_vector=moral_vector,
            vetoed=verdict == "forbid",
            veto_reason=(
                "Aggregate score below threshold" if verdict == "forbid" else ""
            ),
        )

    def _generate_proof(
        self,
        decision_id: str,
        facts: erisml_pb2.EthicalFactsProto,
        profile: str,
        result: EvaluationResult,
    ) -> erisml_pb2.DecisionProofProto:
        """Generate hash-chained decision proof."""
        # Hash the input facts
        facts_bytes = facts.SerializeToString()
        facts_hash = hashlib.sha256(facts_bytes).hexdigest()[:16]

        # Get previous proof hash for chain
        previous_hash = self._proof_chain[-1] if self._proof_chain else ""

        # Compute proof hash
        proof_content = (
            f"{decision_id}:{facts_hash}:{profile}:{result.verdict}:{previous_hash}"
        )
        proof_hash = hashlib.sha256(proof_content.encode()).hexdigest()[:16]

        # Update chain
        self._proof_chain.append(proof_hash)

        proof = erisml_pb2.DecisionProofProto(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_facts_hash=facts_hash,
            profile_name=profile,
            selected_option_id=facts.option_id,
            governance_rationale="; ".join(result.moral_vector.reason_codes),
            confidence=result.moral_vector.epistemic_quality,
            previous_proof_hash=previous_hash,
            proof_hash=proof_hash,
        )

        if result.vetoed:
            proof.forbidden_options.append(facts.option_id)

        return proof

    def _derive_hohfeldian_verdicts(
        self, moral_vector: MoralVector, party_name: str = "agent"
    ) -> Tuple[HohfeldianVerdict, HohfeldianVerdict]:
        """
        Derive Hohfeldian verdicts for agent/affected-party from moral dimensions.

        Maps moral vector dimensions to normative positions:
        - physical_harm > 0.5: Agent has Obligation to avoid, Affected has Claim
        - rights_respect < 0.5: Agent has Obligation, Affected has Claim
        - autonomy_respect > 0.7: Agent has Liberty, Affected has No-claim
        - Default: Agent has Liberty, Affected has No-claim

        Returns:
            (agent_verdict, affected_verdict) tuple
        """
        if moral_vector.physical_harm > 0.5:
            agent_state = HohfeldianState.O
        elif moral_vector.rights_respect < 0.5:
            agent_state = HohfeldianState.O
        elif moral_vector.autonomy_respect > 0.7:
            agent_state = HohfeldianState.L
        else:
            agent_state = HohfeldianState.L

        affected_state = correlative(agent_state)

        agent_verdict = HohfeldianVerdict(
            party_name=party_name,
            state=agent_state,
            confidence=moral_vector.epistemic_quality,
        )
        affected_verdict = HohfeldianVerdict(
            party_name=f"{party_name}_affected",
            state=affected_state,
            confidence=moral_vector.epistemic_quality,
        )

        return agent_verdict, affected_verdict

    def _compute_plan_bond_index(
        self, step_results: List[erisml_pb2.EvaluateStepResponse]
    ) -> erisml_pb2.BondIndexResultProto:
        """
        Compute aggregate Bond Index for a plan using D4 Hohfeldian symmetry.

        Derives Hohfeldian verdicts from each step's moral vector, then
        measures correlative symmetry using compute_bond_index().
        """
        if not step_results:
            return erisml_pb2.BondIndexResultProto(
                bond_index=0.0,
                baseline=BOND_INDEX_BASELINE,
                within_threshold=True,
            )

        agent_verdicts: List[HohfeldianVerdict] = []
        affected_verdicts: List[HohfeldianVerdict] = []
        violations: List[str] = []

        for i, result in enumerate(step_results):
            mv_proto = result.moral_vector
            mv = MoralVector(
                physical_harm=mv_proto.physical_harm,
                rights_respect=mv_proto.rights_respect,
                fairness_equity=mv_proto.fairness_equity,
                autonomy_respect=mv_proto.autonomy_respect,
                privacy_protection=mv_proto.privacy_protection,
                societal_environmental=mv_proto.societal_environmental,
                virtue_care=mv_proto.virtue_care,
                legitimacy_trust=mv_proto.legitimacy_trust,
                epistemic_quality=mv_proto.epistemic_quality,
            )

            agent_v, affected_v = self._derive_hohfeldian_verdicts(
                mv, party_name=f"step_{i}"
            )
            agent_verdicts.append(agent_v)
            affected_verdicts.append(affected_v)

            # Track any asymmetries for violation reporting
            expected = correlative(agent_v.state)
            if affected_v.state != expected:
                violations.append(
                    f"Step {i}: agent({agent_v.state.value}) "
                    f"<-> affected({affected_v.state.value}): "
                    f"expected {expected.value}"
                )

        bond_idx = compute_bond_index(agent_verdicts, affected_verdicts, tau=1.0)

        bi_result = erisml_pb2.BondIndexResultProto(
            bond_index=bond_idx,
            baseline=BOND_INDEX_BASELINE,
            within_threshold=bond_idx < BOND_INDEX_BLOCK_THRESHOLD,
        )
        bi_result.violations.extend(violations)

        return bi_result


# ---------------------------------------------------------------------------
# Server Factory
# ---------------------------------------------------------------------------


def create_erisml_server(
    port: int = 50060,
    max_workers: int = 10,
    default_profile: str = "agi_hpc_safety_v1",
) -> grpc.Server:
    """
    Create and configure an ErisML gRPC server.

    Args:
        port: Port to listen on
        max_workers: Maximum thread pool workers
        default_profile: Default DEME profile name

    Returns:
        Configured gRPC server (call server.start() to begin serving)

    Example:
        server = create_erisml_server(port=50060)
        server.start()
        server.wait_for_termination()
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = ErisMLServicer(default_profile=default_profile)
    erisml_pb2_grpc.add_ErisMLServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"ErisML server configured on port {port}")
    return server


# ---------------------------------------------------------------------------
# Standalone Server Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = create_erisml_server()
    server.start()
    logger.info("ErisML server started, waiting for termination...")
    server.wait_for_termination()
