# ErisML ↔ AGI-HPC Integration Sketch

## Overview

This document sketches how **erisml-lib** (ethical reasoning framework) integrates with **agi-hpc** (cognitive architecture) to create a safety-governed SAI system.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGI-HPC COGNITIVE LAYER                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Left Hemisphere (Planning)          Right Hemisphere (Perception)    │
│   ┌─────────────────────┐             ┌─────────────────────┐          │
│   │ Planner             │             │ World Model         │          │
│   │ - Goal decomposition│             │ - Physics sim       │          │
│   │ - Plan generation   │             │ - State prediction  │          │
│   └─────────┬───────────┘             └─────────┬───────────┘          │
│             │                                   │                       │
│             └─────────────┬─────────────────────┘                       │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    SAFETY GATEWAY                                │  │
│   │  ┌───────────────────────────────────────────────────────────┐  │  │
│   │  │                 ERISML INTEGRATION                         │  │  │
│   │  │                                                            │  │  │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │  │
│   │  │  │ Reflex   │→ │ Tactical │→ │Strategic │                 │  │  │
│   │  │  │ Layer    │  │ Layer    │  │ Layer    │                 │  │  │
│   │  │  │ (<100μs) │  │ (10-100ms)│  │ (policy) │                 │  │  │
│   │  │  └──────────┘  └──────────┘  └──────────┘                 │  │  │
│   │  │       │              │              │                      │  │  │
│   │  │       ▼              ▼              ▼                      │  │  │
│   │  │  ┌─────────────────────────────────────────────────────┐  │  │  │
│   │  │  │              DEMEPipeline.decide()                  │  │  │  │
│   │  │  │  - MoralVector computation                          │  │  │  │
│   │  │  │  - Bond Index verification                          │  │  │  │
│   │  │  │  - Hohfeldian consistency check                     │  │  │  │
│   │  │  │  - DecisionProof generation                         │  │  │  │
│   │  │  └─────────────────────────────────────────────────────┘  │  │  │
│   │  └───────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                           │                                             │
│                           ▼                                             │
│                    [ALLOW / BLOCK / REVISE]                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. New Protobuf Definitions

### `proto/erisml.proto`

```protobuf
syntax = "proto3";
package agi.erisml;

// Bridge between AGI-HPC plan steps and ErisML EthicalFacts
message EthicalFactsProto {
    string option_id = 1;

    // Consequences
    float expected_benefit = 10;
    float expected_harm = 11;
    float urgency = 12;
    int32 affected_count = 13;

    // Rights and Duties
    bool violates_rights = 20;
    bool has_valid_consent = 21;
    bool violates_explicit_rule = 22;

    // Justice and Fairness
    bool discriminates_on_protected_attr = 30;
    bool exploits_vulnerable_population = 31;

    // Safety
    float physical_harm_risk = 40;
    float collision_probability = 41;

    // Epistemic
    float uncertainty_level = 50;
    float evidence_quality = 51;
    bool novel_situation = 52;

    // Domain extensions
    map<string, float> extensions = 100;
}

message MoralVectorProto {
    float physical_harm = 1;
    float rights_respect = 2;
    float fairness_equity = 3;
    float autonomy_respect = 4;
    float privacy_protection = 5;
    float societal_environmental = 6;
    float virtue_care = 7;
    float legitimacy_trust = 8;
    float epistemic_quality = 9;

    repeated string veto_flags = 20;
    repeated string reason_codes = 21;
}

message HohfeldianVerdictProto {
    string party_name = 1;
    string state = 2;           // O, C, L, N
    string expected_state = 3;  // for verification
    float confidence = 4;
}

message BondIndexResultProto {
    float bond_index = 1;       // 0 = perfect symmetry
    float baseline = 2;         // 0.155 (Dear Abby baseline)
    bool within_threshold = 3;  // bond_index < 0.30
    repeated string violations = 4;
}

message DecisionProofProto {
    string decision_id = 1;
    string timestamp = 2;
    string input_facts_hash = 3;
    string profile_hash = 4;
    string profile_name = 5;

    repeated string candidate_option_ids = 10;
    string selected_option_id = 11;
    repeated string ranked_options = 12;
    repeated string forbidden_options = 13;

    string governance_rationale = 20;
    float confidence = 21;

    string previous_proof_hash = 30;
    string proof_hash = 31;
}

// Main service definition
service ErisMLService {
    // Evaluate a single plan step
    rpc EvaluateStep(EvaluateStepRequest) returns (EvaluateStepResponse);

    // Evaluate entire plan graph
    rpc EvaluatePlan(EvaluatePlanRequest) returns (EvaluatePlanResponse);

    // Compute Bond Index for plan
    rpc ComputeBondIndex(BondIndexRequest) returns (BondIndexResultProto);

    // Verify Hohfeldian consistency
    rpc VerifyHohfeldian(HohfeldianRequest) returns (HohfeldianResponse);
}

message EvaluateStepRequest {
    EthicalFactsProto facts = 1;
    string profile_name = 2;
}

message EvaluateStepResponse {
    string verdict = 1;         // strongly_prefer, prefer, neutral, avoid, forbid
    MoralVectorProto moral_vector = 2;
    bool vetoed = 3;
    string veto_reason = 4;
    DecisionProofProto proof = 5;
}

message EvaluatePlanRequest {
    repeated EthicalFactsProto step_facts = 1;
    string profile_name = 2;
    bool generate_proofs = 3;
}

message EvaluatePlanResponse {
    repeated EvaluateStepResponse step_results = 1;
    bool plan_approved = 2;
    repeated string blocked_steps = 3;
    BondIndexResultProto bond_index = 4;
    DecisionProofProto plan_proof = 5;
}

message BondIndexRequest {
    repeated HohfeldianVerdictProto party_a_verdicts = 1;
    repeated HohfeldianVerdictProto party_b_verdicts = 2;
}

message HohfeldianRequest {
    repeated HohfeldianVerdictProto verdicts = 1;
}

message HohfeldianResponse {
    bool consistent = 1;
    float symmetry_rate = 2;
    repeated string violations = 3;
}
```

---

## 2. ErisML Service Implementation

### `src/agi/safety/erisml/service.py`

```python
"""
ErisML gRPC service wrapper for AGI-HPC integration.
Bridges AGI-HPC plan steps to ErisML ethical evaluation.
"""
from dataclasses import dataclass
from typing import List, Optional
import grpc

from erisml.ethics import EthicalFacts, Consequences, RightsAndDuties
from erisml.ethics import JusticeAndFairness, SafetyAndSecurity, EpistemicStatus
from erisml.ethics.layers.pipeline import DEMEPipeline, PipelineConfig
from erisml.ethics.hohfeld import compute_bond_index, HohfeldianVerdict, HohfeldianState
from erisml.ethics.profile_v04 import load_profile_v04

from agi.proto import erisml_pb2, erisml_pb2_grpc
from agi.core.api.grpc_server import GRPCServer


class ErisMLServicer(erisml_pb2_grpc.ErisMLServiceServicer):
    """gRPC servicer bridging AGI-HPC to ErisML."""

    def __init__(self, default_profile: str = "agi_hpc_safety_v1"):
        self.default_profile = default_profile
        self._pipelines: dict[str, DEMEPipeline] = {}

    def _get_pipeline(self, profile_name: str) -> DEMEPipeline:
        """Lazy-load and cache DEME pipelines by profile."""
        if profile_name not in self._pipelines:
            profile = load_profile_v04(profile_name)
            config = PipelineConfig(
                generate_proofs=True,
                profile_name=profile_name,
            )
            self._pipelines[profile_name] = DEMEPipeline.from_profile(profile, config)
        return self._pipelines[profile_name]

    def _proto_to_ethical_facts(self, proto: erisml_pb2.EthicalFactsProto) -> EthicalFacts:
        """Convert protobuf to ErisML EthicalFacts."""
        return EthicalFacts(
            option_id=proto.option_id,
            consequences=Consequences(
                expected_benefit=proto.expected_benefit,
                expected_harm=proto.expected_harm,
                urgency=proto.urgency,
                affected_count=proto.affected_count,
            ),
            rights_and_duties=RightsAndDuties(
                violates_rights=proto.violates_rights,
                has_valid_consent=proto.has_valid_consent,
                violates_explicit_rule=proto.violates_explicit_rule,
            ),
            justice_and_fairness=JusticeAndFairness(
                discriminates_on_protected_attr=proto.discriminates_on_protected_attr,
                exploits_vulnerable_population=proto.exploits_vulnerable_population,
            ),
            safety_and_security=SafetyAndSecurity(
                physical_harm_risk=proto.physical_harm_risk,
            ),
            epistemic_status=EpistemicStatus(
                uncertainty_level=proto.uncertainty_level,
                evidence_quality=proto.evidence_quality,
                novel_situation_flag=proto.novel_situation,
            ),
        )

    def EvaluateStep(self, request, context):
        """Evaluate a single plan step through DEME pipeline."""
        profile = request.profile_name or self.default_profile
        pipeline = self._get_pipeline(profile)

        facts = self._proto_to_ethical_facts(request.facts)
        result = pipeline.decide([facts])

        # Convert result to proto
        response = erisml_pb2.EvaluateStepResponse(
            verdict=result.judgements[0].verdict if result.judgements else "neutral",
            vetoed=result.forbidden_options and request.facts.option_id in result.forbidden_options,
            veto_reason=result.veto_reasons.get(request.facts.option_id, ""),
        )

        if result.proof:
            response.proof.CopyFrom(self._proof_to_proto(result.proof))

        return response

    def EvaluatePlan(self, request, context):
        """Evaluate entire plan through DEME pipeline."""
        profile = request.profile_name or self.default_profile
        pipeline = self._get_pipeline(profile)

        # Convert all step facts
        facts_list = [self._proto_to_ethical_facts(f) for f in request.step_facts]

        # Run pipeline
        result = pipeline.decide(facts_list)

        # Build response
        response = erisml_pb2.EvaluatePlanResponse(
            plan_approved=len(result.forbidden_options) == 0,
            blocked_steps=list(result.forbidden_options),
        )

        # Add step results
        for facts in facts_list:
            step_result = erisml_pb2.EvaluateStepResponse(
                verdict=self._get_verdict_for_option(result, facts.option_id),
                vetoed=facts.option_id in result.forbidden_options,
            )
            response.step_results.append(step_result)

        # Compute Bond Index across all steps
        if len(facts_list) >= 2:
            bond_result = self._compute_plan_bond_index(result)
            response.bond_index.CopyFrom(bond_result)

        if result.proof:
            response.plan_proof.CopyFrom(self._proof_to_proto(result.proof))

        return response

    def ComputeBondIndex(self, request, context):
        """Compute Bond Index between two parties' Hohfeldian verdicts."""
        verdicts_a = [
            HohfeldianVerdict(v.party_name, HohfeldianState(v.state), confidence=v.confidence)
            for v in request.party_a_verdicts
        ]
        verdicts_b = [
            HohfeldianVerdict(v.party_name, HohfeldianState(v.state), confidence=v.confidence)
            for v in request.party_b_verdicts
        ]

        bond_index = compute_bond_index(verdicts_a, verdicts_b)

        return erisml_pb2.BondIndexResultProto(
            bond_index=bond_index,
            baseline=0.155,  # Dear Abby baseline
            within_threshold=bond_index < 0.30,
        )


def create_erisml_server(port: int = 50060) -> GRPCServer:
    """Factory function to create ErisML gRPC server."""
    server = GRPCServer(port=port, service_name="erisml")
    servicer = ErisMLServicer()
    erisml_pb2_grpc.add_ErisMLServiceServicer_to_server(servicer, server.server)
    return server
```

---

## 3. Integration with AGI-HPC Safety Subsystem

### `src/agi/safety/pre_action/service.py` (Modified)

```python
"""
Pre-action safety service with ErisML integration.
"""
from typing import Optional
import grpc

from agi.proto import safety_pb2, safety_pb2_grpc, erisml_pb2, erisml_pb2_grpc
from agi.safety.rules.engine import SafetyRuleEngine
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts


class PreActionSafetyServicer(safety_pb2_grpc.PreActionSafetyServicer):
    """Pre-action safety with ErisML ethical evaluation."""

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        erisml_channel: Optional[grpc.Channel] = None,
        erisml_profile: str = "agi_hpc_safety_v1",
    ):
        self.rule_engine = rule_engine
        self.facts_builder = PlanStepToEthicalFacts()
        self.erisml_profile = erisml_profile

        # ErisML client (optional - graceful degradation if unavailable)
        self.erisml_stub = None
        if erisml_channel:
            self.erisml_stub = erisml_pb2_grpc.ErisMLServiceStub(erisml_channel)

    def CheckPlan(self, request, context):
        """
        Check plan through:
        1. Fast rule-based checks (banned tools, schema validation)
        2. ErisML ethical evaluation (if available)
        3. Bond Index verification
        """
        # Phase 1: Rule-based checks (fast, <1ms)
        rule_result = self.rule_engine.check_plan(request.plan)
        if rule_result.decision == "BLOCK":
            return safety_pb2.SafetyResult(
                decision="BLOCK",
                risk_score=1.0,
                reasons=rule_result.reasons,
            )

        # Phase 2: ErisML ethical evaluation (10-100ms)
        if self.erisml_stub:
            try:
                erisml_result = self._evaluate_with_erisml(request.plan)

                # Check for ethical vetoes
                if not erisml_result.plan_approved:
                    return safety_pb2.SafetyResult(
                        decision="BLOCK",
                        risk_score=0.9,
                        reasons=[f"Ethical veto: {r}" for r in erisml_result.blocked_steps],
                        metadata={"bond_index": erisml_result.bond_index.bond_index},
                    )

                # Check Bond Index threshold
                if not erisml_result.bond_index.within_threshold:
                    return safety_pb2.SafetyResult(
                        decision="REVISE",
                        risk_score=0.7,
                        reasons=[
                            f"Bond Index {erisml_result.bond_index.bond_index:.3f} "
                            f"exceeds threshold 0.30 (baseline: 0.155)"
                        ],
                    )

            except grpc.RpcError as e:
                # Graceful degradation - log and continue with rule-based only
                self._log_erisml_unavailable(e)

        # Phase 3: Approved
        return safety_pb2.SafetyResult(
            decision="ALLOW",
            risk_score=rule_result.risk_score,
            reasons=rule_result.reasons,
        )

    def _evaluate_with_erisml(self, plan) -> erisml_pb2.EvaluatePlanResponse:
        """Convert plan steps to EthicalFacts and evaluate."""
        step_facts = []
        for step in plan.steps:
            facts_proto = self.facts_builder.build(step)
            step_facts.append(facts_proto)

        request = erisml_pb2.EvaluatePlanRequest(
            step_facts=step_facts,
            profile_name=self.erisml_profile,
            generate_proofs=True,
        )

        return self.erisml_stub.EvaluatePlan(request)
```

---

## 4. EthicalFacts Builder for Plan Steps

### `src/agi/safety/erisml/facts_builder.py`

```python
"""
Converts AGI-HPC plan steps to ErisML EthicalFacts.
This is the critical domain-specific bridge.
"""
from dataclasses import dataclass
from typing import Dict, Any

from agi.proto import plan_pb2, erisml_pb2


@dataclass
class RiskEstimates:
    """Estimated risks from simulation or heuristics."""
    physical_harm: float = 0.0
    collision_probability: float = 0.0
    rights_violation_risk: float = 0.0
    uncertainty: float = 0.5


class PlanStepToEthicalFacts:
    """
    Converts AGI-HPC PlanStep to ErisML EthicalFactsProto.

    This is domain-specific logic that maps:
    - Plan step parameters → ethical dimensions
    - Tool usage → rights/harm implications
    - Simulation results → risk estimates
    """

    # Tools with ethical implications
    HARMFUL_TOOLS = {"delete", "destroy", "terminate", "override_safety"}
    CONSENT_REQUIRED_TOOLS = {"access_personal_data", "modify_user_settings"}
    HIGH_RISK_TOOLS = {"physical_manipulation", "navigation", "power_control"}

    def build(
        self,
        step: plan_pb2.PlanStep,
        simulation_result: plan_pb2.SimulationResult = None,
        world_state: Dict[str, Any] = None,
    ) -> erisml_pb2.EthicalFactsProto:
        """Convert plan step to EthicalFacts."""

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
            violates_explicit_rule=step.tool_id in self.HARMFUL_TOOLS,

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
        sim_result: plan_pb2.SimulationResult = None,
    ) -> RiskEstimates:
        """Extract or estimate risks from simulation."""
        if sim_result:
            return RiskEstimates(
                physical_harm=sim_result.overall_risk,
                collision_probability=sim_result.collision_probability,
                uncertainty=0.1,  # Low uncertainty with simulation
            )

        # Heuristic estimates without simulation
        if step.tool_id in self.HIGH_RISK_TOOLS:
            return RiskEstimates(
                physical_harm=0.3,
                collision_probability=0.2,
                uncertainty=0.7,  # High uncertainty without simulation
            )

        return RiskEstimates()

    def _estimate_benefit(self, step: plan_pb2.PlanStep) -> float:
        """Estimate expected benefit of step."""
        # Mission-level steps have high benefit
        if step.level == 0:
            return 0.9
        # Actionable steps have moderate benefit
        if step.kind == "action":
            return 0.6
        return 0.5

    def _estimate_urgency(self, step: plan_pb2.PlanStep) -> float:
        """Estimate urgency from safety tags."""
        if "emergency" in step.safety_tags:
            return 1.0
        if "time_critical" in step.safety_tags:
            return 0.8
        return 0.3

    def _count_affected(
        self,
        step: plan_pb2.PlanStep,
        world_state: Dict[str, Any] = None,
    ) -> int:
        """Count entities affected by this step."""
        if world_state and "nearby_agents" in world_state:
            return len(world_state["nearby_agents"])
        return 1  # Default: affects at least the agent

    def _check_rights_violation(self, step: plan_pb2.PlanStep) -> bool:
        """Check if step violates rights."""
        return step.tool_id in self.HARMFUL_TOOLS

    def _check_consent(self, step: plan_pb2.PlanStep) -> bool:
        """Check if consent was obtained for consent-required actions."""
        if step.tool_id not in self.CONSENT_REQUIRED_TOOLS:
            return True  # Not required
        return step.params.get("consent_obtained", False)

    def _is_novel_situation(self, step: plan_pb2.PlanStep) -> bool:
        """Check if this is a novel situation."""
        return "novel" in step.safety_tags or "unprecedented" in step.safety_tags
```

---

## 5. DEME Profile for AGI-HPC

### `deme_profiles/agi_hpc_safety_v1.json`

```json
{
    "version": "0.4",
    "name": "agi_hpc_safety_v1",
    "description": "Safety governance profile for AGI-HPC cognitive architecture",
    "stakeholder_label": "users_operators_and_society",

    "tier_configs": {
        "0": {
            "weight_multiplier": 10.0,
            "veto_enabled": true,
            "description": "Constitutional - always enforced"
        },
        "1": {
            "weight_multiplier": 5.0,
            "veto_enabled": true,
            "description": "Core Safety - physical harm prevention"
        },
        "2": {
            "weight_multiplier": 2.0,
            "veto_enabled": false,
            "description": "Rights and Fairness - soft constraints"
        },
        "3": {
            "weight_multiplier": 1.0,
            "veto_enabled": false,
            "description": "Soft Values - preferences"
        }
    },

    "active_em_names": [
        "geneva_constitutional",
        "physical_safety",
        "rights_first_compliance",
        "autonomy_respect",
        "transparency"
    ],

    "reflex_rules": [
        {
            "name": "physical_harm_veto",
            "category": "PHYSICAL_HARM",
            "field": "safety_and_security.physical_harm_risk",
            "threshold": 0.8,
            "reason": "Physical harm risk exceeds safety threshold"
        },
        {
            "name": "rights_violation_veto",
            "category": "RIGHTS_VIOLATION",
            "field": "rights_and_duties.violates_rights",
            "value": true,
            "reason": "Action violates fundamental rights"
        },
        {
            "name": "banned_tool_veto",
            "category": "RULE_VIOLATION",
            "field": "rights_and_duties.violates_explicit_rule",
            "value": true,
            "reason": "Action uses banned tool"
        }
    ],

    "bond_index_config": {
        "baseline": 0.155,
        "warning_threshold": 0.25,
        "block_threshold": 0.30,
        "compute_per_decision": true
    },

    "audit_config": {
        "generate_proofs": true,
        "hash_chain_enabled": true,
        "retention_days": 365
    }
}
```

---

## 6. Episodic Memory Integration

### `src/agi/memory/episodic/erisml_events.py`

```python
"""
Store ErisML decision proofs in episodic memory for audit and learning.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agi.proto import erisml_pb2


@dataclass
class EthicalDecisionEvent:
    """Episodic memory event for ethical decisions."""
    event_id: str
    timestamp: datetime

    # Decision context
    plan_id: str
    step_id: str

    # ErisML results
    verdict: str
    bond_index: float
    vetoed: bool
    veto_reason: Optional[str]

    # Proof chain
    decision_proof_hash: str
    previous_proof_hash: Optional[str]

    # Outcome (filled in post-action)
    actual_outcome: Optional[str] = None
    outcome_matched_prediction: Optional[bool] = None

    def to_episodic_entry(self) -> dict:
        """Convert to episodic memory format."""
        return {
            "event_type": "ethical_decision",
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tags": ["erisml", "safety", f"verdict:{self.verdict}"],
            "data": {
                "plan_id": self.plan_id,
                "step_id": self.step_id,
                "verdict": self.verdict,
                "bond_index": self.bond_index,
                "vetoed": self.vetoed,
                "veto_reason": self.veto_reason,
                "proof_hash": self.decision_proof_hash,
            },
        }


class EthicalDecisionLogger:
    """Logs ethical decisions to episodic memory."""

    def __init__(self, episodic_service):
        self.episodic = episodic_service
        self._last_proof_hash: Optional[str] = None

    def log_decision(
        self,
        plan_id: str,
        step_id: str,
        result: erisml_pb2.EvaluateStepResponse,
    ) -> str:
        """Log decision and return event_id."""
        event = EthicalDecisionEvent(
            event_id=f"eth_{plan_id}_{step_id}_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            plan_id=plan_id,
            step_id=step_id,
            verdict=result.verdict,
            bond_index=result.bond_index.bond_index if result.HasField("bond_index") else 0.0,
            vetoed=result.vetoed,
            veto_reason=result.veto_reason if result.vetoed else None,
            decision_proof_hash=result.proof.proof_hash if result.HasField("proof") else "",
            previous_proof_hash=self._last_proof_hash,
        )

        # Append to episodic memory
        self.episodic.append(event.to_episodic_entry())

        # Update chain
        self._last_proof_hash = event.decision_proof_hash

        return event.event_id

    def record_outcome(self, event_id: str, outcome: str, matched: bool):
        """Record actual outcome for learning."""
        self.episodic.update(
            event_id=event_id,
            updates={
                "actual_outcome": outcome,
                "outcome_matched_prediction": matched,
            },
        )
```

---

## 7. Service Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGI-HPC SERVICE MESH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ LH Service  │────▶│ Pre-Action  │────▶│  ErisML     │       │
│  │ :50051      │     │ Safety      │     │  Service    │       │
│  │             │     │ :50055      │     │  :50060     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                   │                   │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Memory      │     │ In-Action   │     │ Episodic    │       │
│  │ Services    │     │ Safety      │     │ Memory      │       │
│  │ :50052-54   │     │ :50056      │     │ :50053      │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                   │                   ▲               │
│         │                   │                   │               │
│         ▼                   ▼                   │               │
│  ┌─────────────┐     ┌─────────────┐           │               │
│  │ RH Service  │     │ Post-Action │───────────┘               │
│  │ :50057      │     │ Safety      │  (logs outcomes)          │
│  │             │     │ :50058      │                           │
│  └─────────────┘     └─────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Decision Flow Example

```
1. LH generates plan: "Navigate to red cube, pick it up"
   └── Steps: [analyze_goal, plan_path, execute_navigation, grasp_object]

2. LH calls PreActionSafety.CheckPlan(plan)
   └── Pre-action safety receives plan

3. Pre-action safety runs rule-based checks
   └── No banned tools → PASS

4. Pre-action safety calls ErisML.EvaluatePlan(step_facts)
   └── Each step converted to EthicalFacts
   └── DEMEPipeline.decide() runs:
       ├── ReflexLayer: No hard vetoes
       ├── TacticalLayer: MoralVectors computed
       │   └── physical_safety: grasp_object has collision risk 0.2
       │   └── autonomy_respect: all steps respect operator intent
       └── Bond Index: 0.12 (within threshold)

5. ErisML returns:
   └── plan_approved: true
   └── bond_index: 0.12 (healthy)
   └── proof: DecisionProof with hash chain

6. Pre-action safety returns ALLOW to LH

7. LH publishes plan.step_ready events

8. Post-action safety logs outcomes to episodic memory
   └── EthicalDecisionLogger.record_outcome()
   └── Hash chain maintained for audit
```

---

## 9. Configuration

### `config/agi_hpc.yaml` (additions)

```yaml
safety:
  pre_action:
    enabled: true
    erisml:
      enabled: true
      address: "localhost:50060"
      profile: "agi_hpc_safety_v1"
      timeout_ms: 100
      fallback_on_unavailable: true

    bond_index:
      enabled: true
      baseline: 0.155
      warning_threshold: 0.25
      block_threshold: 0.30

    audit:
      log_decisions: true
      hash_chain: true
      retention_days: 365

  in_action:
    enabled: true
    # ... existing config

  post_action:
    enabled: true
    log_to_episodic: true
    learn_from_outcomes: true
```

---

## 10. Testing Strategy

### Unit Tests

```python
# tests/test_erisml_integration.py

def test_plan_step_to_ethical_facts():
    """Test conversion of plan steps to EthicalFacts."""
    builder = PlanStepToEthicalFacts()
    step = create_test_step(tool_id="navigation", safety_tags=["time_critical"])
    facts = builder.build(step)

    assert facts.option_id == step.step_id
    assert facts.urgency == 0.8  # time_critical
    assert not facts.violates_rights

def test_bond_index_computation():
    """Test Bond Index is computed correctly."""
    # Create correlative pair: A has O, B has C
    verdicts_a = [HohfeldianVerdict("A", HohfeldianState.O)]
    verdicts_b = [HohfeldianVerdict("B", HohfeldianState.C)]

    bond_index = compute_bond_index(verdicts_a, verdicts_b)
    assert bond_index == 0.0  # Perfect symmetry

def test_erisml_veto_blocks_plan():
    """Test that ErisML veto blocks dangerous plans."""
    servicer = PreActionSafetyServicer(
        rule_engine=SafetyRuleEngine(),
        erisml_channel=mock_erisml_channel(returns_veto=True),
    )

    result = servicer.CheckPlan(dangerous_plan_request)
    assert result.decision == "BLOCK"
    assert "Ethical veto" in result.reasons[0]
```

### Integration Tests

```python
# tests/integration/test_full_safety_pipeline.py

def test_full_pipeline_with_erisml():
    """Test LH → Safety → ErisML → Memory flow."""
    # Start all services
    lh = start_lh_service()
    safety = start_safety_service()
    erisml = start_erisml_service()
    memory = start_memory_service()

    # Generate plan
    plan_response = lh.Plan(test_request)

    # Verify safety was checked
    assert plan_response.metadata["safety_status"] == "approved"

    # Verify decision was logged
    events = memory.query_episodic(tags=["erisml"])
    assert len(events) > 0
    assert events[0]["bond_index"] < 0.30
```

---

## Summary

This integration provides:

1. **Formal ethical reasoning** via ErisML's DEMEPipeline
2. **Bond Index verification** ensuring correlative symmetry
3. **Hohfeldian consistency** tracking moral positions
4. **Audit trail** with hash-chained DecisionProofs
5. **Graceful degradation** when ErisML unavailable
6. **Learning from outcomes** via episodic memory

The key insight: **safety is not a gate at the end, but woven throughout the cognitive architecture**.
