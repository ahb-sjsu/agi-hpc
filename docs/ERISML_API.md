# ErisML Integration API Reference

This document describes the API for integrating ErisML ethical reasoning with AGI-HPC cognitive architecture.

## Overview

The ErisML integration provides three main components:

1. **ErisML Service** - gRPC service for ethical evaluation
2. **Safety Gateway** - Three-layer safety architecture
3. **Facts Builder** - Converts plan steps to ethical facts

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGI-HPC COGNITIVE LAYER                          │
│                                                                     │
│   Left Hemisphere (Planning)      Right Hemisphere (Perception)    │
│   ┌─────────────────┐             ┌─────────────────┐              │
│   │ Planner         │             │ World Model     │              │
│   └────────┬────────┘             └────────┬────────┘              │
│            │                               │                        │
│            └───────────┬───────────────────┘                        │
│                        ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    SAFETY GATEWAY                            │  │
│   │  ┌─────────────────────────────────────────────────────────┐│  │
│   │  │              ERISML INTEGRATION                         ││  │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││  │
│   │  │  │ Reflex   │→ │ Tactical │→ │Strategic │              ││  │
│   │  │  │ (<100μs) │  │(10-100ms)│  │ (policy) │              ││  │
│   │  │  └──────────┘  └──────────┘  └──────────┘              ││  │
│   │  │       │              │              │                   ││  │
│   │  │       ▼              ▼              ▼                   ││  │
│   │  │  ┌─────────────────────────────────────────────────┐   ││  │
│   │  │  │           ErisMLService.EvaluatePlan()          │   ││  │
│   │  │  └─────────────────────────────────────────────────┘   ││  │
│   │  └─────────────────────────────────────────────────────────┘│  │
│   └─────────────────────────────────────────────────────────────┘  │
│                        │                                            │
│                        ▼                                            │
│                 [ALLOW / BLOCK / REVISE]                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ErisML Service

### Starting the Service

```python
from agi.safety.erisml import create_erisml_server

# Create and start server
server = create_erisml_server(
    port=50060,
    max_workers=10,
    default_profile="agi_hpc_safety_v1",
)
server.start()
server.wait_for_termination()
```

### RPC Methods

#### EvaluateStep

Evaluate a single plan step through ethical reasoning.

**Request:** `EvaluateStepRequest`
```protobuf
message EvaluateStepRequest {
  EthicalFactsProto facts = 1;
  string profile_name = 2;  // Optional, defaults to agi_hpc_safety_v1
}
```

**Response:** `EvaluateStepResponse`
```protobuf
message EvaluateStepResponse {
  string verdict = 1;           // strongly_prefer, prefer, neutral, avoid, forbid
  MoralVectorProto moral_vector = 2;
  bool vetoed = 3;
  string veto_reason = 4;
  DecisionProofProto proof = 5;
}
```

**Example:**
```python
import grpc
from agi.proto_gen import erisml_pb2, erisml_pb2_grpc

channel = grpc.insecure_channel("localhost:50060")
stub = erisml_pb2_grpc.ErisMLServiceStub(channel)

facts = erisml_pb2.EthicalFactsProto(
    option_id="step_001",
    expected_benefit=0.8,
    expected_harm=0.1,
    physical_harm_risk=0.05,
)

request = erisml_pb2.EvaluateStepRequest(
    facts=facts,
    profile_name="agi_hpc_safety_v1",
)

response = stub.EvaluateStep(request)
print(f"Verdict: {response.verdict}")
print(f"Vetoed: {response.vetoed}")
```

#### EvaluatePlan

Evaluate an entire plan with aggregated Bond Index.

**Request:** `EvaluatePlanRequest`
```protobuf
message EvaluatePlanRequest {
  repeated EthicalFactsProto step_facts = 1;
  string profile_name = 2;
  bool generate_proofs = 3;
}
```

**Response:** `EvaluatePlanResponse`
```protobuf
message EvaluatePlanResponse {
  repeated EvaluateStepResponse step_results = 1;
  bool plan_approved = 2;
  repeated string blocked_steps = 3;
  BondIndexResultProto bond_index = 4;
  DecisionProofProto plan_proof = 5;
}
```

#### ComputeBondIndex

Compute Bond Index between two parties' Hohfeldian positions.

**Request:** `BondIndexRequest`
```protobuf
message BondIndexRequest {
  repeated HohfeldianVerdictProto party_a_verdicts = 1;
  repeated HohfeldianVerdictProto party_b_verdicts = 2;
}
```

**Response:** `BondIndexResultProto`
```protobuf
message BondIndexResultProto {
  float bond_index = 1;       // 0 = perfect symmetry
  float baseline = 2;         // 0.155 (Dear Abby baseline)
  bool within_threshold = 3;  // bond_index < 0.30
  repeated string violations = 4;
}
```

#### VerifyHohfeldian

Verify correlative symmetry in Hohfeldian positions.

**Request:** `HohfeldianRequest`
```protobuf
message HohfeldianRequest {
  repeated HohfeldianVerdictProto verdicts = 1;
}
```

**Response:** `HohfeldianResponse`
```protobuf
message HohfeldianResponse {
  bool consistent = 1;
  float symmetry_rate = 2;    // 1.0 = perfect O↔C, L↔N symmetry
  repeated string violations = 3;
}
```

---

## Safety Gateway

### Initialization

```python
from agi.safety import SafetyGateway

gateway = SafetyGateway(
    erisml_address="localhost:50060",  # Optional ErisML connection
    profile_name="agi_hpc_safety_v1",
    banned_tools={"override_safety", "bypass_auth"},
    bond_index_threshold=0.30,
    timeout_ms=100,
)
```

### Methods

#### check_plan

Check a plan before execution.

```python
from agi.proto_gen import plan_pb2

plan = plan_pb2.PlanGraphProto(
    plan_id="plan_001",
    goal_text="Pick up the red cube",
)
plan.steps.append(plan_pb2.PlanStep(
    step_id="step_1",
    kind="action",
    tool_id="navigation",
))

result = gateway.check_plan(plan)

if result.decision == SafetyDecision.ALLOW:
    execute_plan(plan)
elif result.decision == SafetyDecision.REVISE:
    print(f"Revise plan: {result.reasons}")
elif result.decision == SafetyDecision.BLOCK:
    print(f"Plan blocked: {result.reasons}")
```

#### check_action

Check an action during execution.

```python
result = gateway.check_action(
    step=current_step,
    sensor_readings={
        "proximity": 0.5,
        "force": 10.0,
    },
)

if result.decision == SafetyDecision.BLOCK:
    emergency_stop()
```

#### reflex_check

Ultra-fast reflex check (<100μs).

```python
safe = gateway.reflex_check(
    physical_harm_risk=0.1,
    collision_probability=0.05,
    emergency_flag=False,
)

if not safe:
    emergency_stop()
```

#### report_outcome

Report action outcome for learning.

```python
event_id = gateway.report_outcome(
    plan_id="plan_001",
    step_id="step_3",
    success=True,
    actual_harm=0.0,
    description="Object grasped successfully",
)
```

---

## Facts Builder

Converts AGI-HPC plan steps to ErisML ethical facts.

```python
from agi.safety.erisml import PlanStepToEthicalFacts
from agi.proto_gen import plan_pb2

builder = PlanStepToEthicalFacts()

step = plan_pb2.PlanStep(
    step_id="step_001",
    kind="action",
    tool_id="gripper_control",
    description="Grasp object",
)
step.safety_tags.append("time_critical")

# Build facts with optional simulation result
facts = builder.build(
    step,
    simulation_result=sim_result,  # Optional
    world_state={"nearby_agents": ["human_1"]},  # Optional
)

print(f"Physical harm risk: {facts.physical_harm_risk}")
print(f"Urgency: {facts.urgency}")
```

---

## Key Concepts

### Bond Index

The Bond Index measures deviation from perfect correlative symmetry in Hohfeldian normative positions:

- **O (Obligation) ↔ C (Claim)**: If A has an Obligation, B has a correlative Claim
- **L (Liberty) ↔ N (No-claim)**: If A has a Liberty, B has a correlative No-claim

**Values:**
- `0.0` = Perfect symmetry (ideal)
- `0.155` = Empirical baseline (Dear Abby corpus)
- `0.25` = Warning threshold
- `0.30` = Block threshold

### Moral Vector

8+1 dimensional ethical assessment:

1. `physical_harm` - Risk of physical harm (0=safe, 1=dangerous)
2. `rights_respect` - Respect for rights (0=violates, 1=respects)
3. `fairness_equity` - Fairness and equity
4. `autonomy_respect` - Respect for autonomy
5. `privacy_protection` - Privacy protection
6. `societal_environmental` - Societal/environmental impact
7. `virtue_care` - Virtue ethics dimension
8. `legitimacy_trust` - Procedural legitimacy
9. `epistemic_quality` - Confidence in assessment (+1 dimension)

### Verdicts

- `strongly_prefer` - Highly ethical action
- `prefer` - Ethical action
- `neutral` - Neither good nor bad
- `avoid` - Potentially problematic
- `forbid` - Action vetoed (hard block)

### Decision Proofs

Hash-chained audit trail for governance compliance:

```protobuf
message DecisionProofProto {
  string decision_id = 1;
  string timestamp = 2;
  string input_facts_hash = 3;
  string profile_hash = 4;
  string profile_name = 5;
  string selected_option_id = 11;
  string governance_rationale = 20;
  float confidence = 21;
  string previous_proof_hash = 30;  // Chain link
  string proof_hash = 31;
}
```

---

## Configuration

### DEME Profile

Create a profile in `deme_profiles/agi_hpc_safety_v1.json`:

```json
{
    "version": "0.4",
    "name": "agi_hpc_safety_v1",
    "description": "Safety profile for AGI-HPC",

    "tier_configs": {
        "0": {"weight_multiplier": 10.0, "veto_enabled": true},
        "1": {"weight_multiplier": 5.0, "veto_enabled": true},
        "2": {"weight_multiplier": 2.0, "veto_enabled": false},
        "3": {"weight_multiplier": 1.0, "veto_enabled": false}
    },

    "bond_index_config": {
        "baseline": 0.155,
        "warning_threshold": 0.25,
        "block_threshold": 0.30
    },

    "audit_config": {
        "generate_proofs": true,
        "hash_chain_enabled": true
    }
}
```

---

## Service Ports

| Service | Default Port | Description |
|---------|--------------|-------------|
| ErisML Service | 50060 | Ethical evaluation |
| Safety Gateway | 50055 | Pre/In/Post action safety |
| LH Service | 50051 | Left hemisphere planning |
| Memory Service | 50052-54 | Memory subsystems |

---

## Error Handling

### ErisML Unavailable

The Safety Gateway gracefully degrades when ErisML is unavailable:

```python
gateway = SafetyGateway(erisml_address="localhost:50060")

# If ErisML is down, check_plan still works with rule-based checks only
result = gateway.check_plan(plan)
# result.reasons will include "ErisML unavailable, rule-based check only"
```

### Timeout Handling

```python
gateway = SafetyGateway(
    erisml_address="localhost:50060",
    timeout_ms=100,  # 100ms timeout
)

# Calls to ErisML will timeout after 100ms
# Fallback to rule-based checking on timeout
```

---

## Testing

Run the test suite:

```bash
pytest tests/unit/test_erisml_proto.py tests/unit/test_safety_proto.py -v
```

Test coverage:
- 32 tests for ErisML proto messages
- 36 tests for Safety proto messages
