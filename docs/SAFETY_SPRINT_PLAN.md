# Safety Subsystem Sprint Plan

## Overview

The Safety Subsystem provides multi-layer protection for the AGI-HPC cognitive architecture, ensuring all actions are ethically vetted and physically safe. It integrates the ErisML ethical reasoning framework with real-time safety monitoring and human oversight capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SAFETY SUBSYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SafetyGateway                                 │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │    │
│  │  │ Pre-Action  │  │  In-Action   │  │        Post-Action          │ │    │
│  │  │   Check     │──│  Monitor     │──│         Audit               │ │    │
│  │  └─────────────┘  └──────────────┘  └─────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Safety Layers                                │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ REFLEX LAYER (<100μs) - Hardware interlocks, emergency stops  │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                              │                                       │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ TACTICAL LAYER (10-100ms) - Rule engine, constraint checking  │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                              │                                       │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ STRATEGIC LAYER (100ms-10s) - ErisML ethical reasoning        │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ErisML Integration                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │    │
│  │  │ Facts Builder│──│ ErisML gRPC  │──│ Decision Proof Generator   │ │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Human Oversight                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │    │
│  │  │ DEFER Queue  │──│ Review UI    │──│ Override Audit Trail       │ │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Current State Assessment

### Existing Components

| Component | Status | Location |
|-----------|--------|----------|
| safety.proto | Complete | `proto/safety.proto` |
| SafetyGateway | Functional | `src/agi/safety/gateway.py` |
| ErisML Service | Functional | `src/agi/safety/erisml/service.py` |
| Facts Builder | Functional | `src/agi/safety/erisml/facts_builder.py` |
| PreActionService | Stub | `src/agi/safety/pre_action/service.py` |
| InActionService | Stub | `src/agi/safety/in_action/service.py` |
| PostActionService | Stub | `src/agi/safety/post_action/service.py` |
| SafetyRuleEngine | Basic | `src/agi/safety/rules/engine.py` |

### Gap Analysis

1. **gRPC Services** - Stubs need full implementation
2. **Rule Engine** - Needs YAML policy loading and DSL
3. **Reflex Layer** - No hardware safety integration
4. **Human Oversight** - No DEFER queue or review workflow
5. **Audit Trail** - No persistent decision proof storage
6. **Testing** - No formal verification or fuzzing

---

## Sprint 1: Service Implementation (Weeks 1-2)

### Goals
- Implement full gRPC services for pre/in/post action phases
- Wire up SafetyGateway with all service layers
- Add comprehensive logging and metrics

### Tasks

#### 1.1 Pre-Action Service Implementation

```python
# src/agi/safety/pre_action/service.py
"""Pre-action safety verification service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grpc

from agi.safety.rules.engine import SafetyRuleEngine
from proto.safety_pb2 import (
    PreActionCheckRequest,
    PreActionCheckResponse,
    SafetyDecision,
    RiskCategory,
    RiskLevel,
    SafetyViolation,
)
from proto.safety_pb2_grpc import PreActionSafetyServiceServicer

if TYPE_CHECKING:
    from agi.safety.erisml.service import ErisMLService

logger = logging.getLogger(__name__)


@dataclass
class PreActionConfig:
    """Configuration for pre-action safety checks."""

    max_risk_level: RiskLevel = RiskLevel.RISK_LEVEL_MEDIUM
    require_erisml_approval: bool = True
    defer_on_uncertainty: bool = True
    confidence_threshold: float = 0.8


class PreActionSafetyService(PreActionSafetyServiceServicer):
    """Pre-action safety verification service.

    Checks proposed actions before execution:
    - Rule-based constraint validation
    - ErisML ethical analysis
    - Risk level assessment
    - Human oversight triggers
    """

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        erisml_service: ErisMLService | None = None,
        config: PreActionConfig | None = None,
    ) -> None:
        """Initialize pre-action service."""
        self.rule_engine = rule_engine
        self.erisml_service = erisml_service
        self.config = config or PreActionConfig()
        self._metrics = PreActionMetrics()

    async def CheckAction(
        self,
        request: PreActionCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> PreActionCheckResponse:
        """Check if an action is safe to execute."""
        logger.info(
            "Pre-action check for action=%s agent=%s",
            request.action_id,
            request.agent_id,
        )

        violations: list[SafetyViolation] = []
        decision = SafetyDecision.SAFETY_DECISION_ALLOW

        # Layer 1: Rule-based checks (fastest)
        rule_result = await self._check_rules(request)
        if rule_result.violations:
            violations.extend(rule_result.violations)
            if rule_result.is_blocking:
                decision = SafetyDecision.SAFETY_DECISION_DENY

        # Layer 2: ErisML ethical analysis (if enabled)
        if (
            decision != SafetyDecision.SAFETY_DECISION_DENY
            and self.config.require_erisml_approval
            and self.erisml_service
        ):
            erisml_result = await self._check_erisml(request)
            if erisml_result.decision == SafetyDecision.SAFETY_DECISION_DENY:
                decision = SafetyDecision.SAFETY_DECISION_DENY
                violations.extend(erisml_result.violations)
            elif erisml_result.decision == SafetyDecision.SAFETY_DECISION_DEFER:
                decision = SafetyDecision.SAFETY_DECISION_DEFER

        # Layer 3: Uncertainty check
        if (
            decision == SafetyDecision.SAFETY_DECISION_ALLOW
            and self.config.defer_on_uncertainty
        ):
            confidence = self._calculate_confidence(request, violations)
            if confidence < self.config.confidence_threshold:
                decision = SafetyDecision.SAFETY_DECISION_DEFER

        self._metrics.record_check(decision)

        return PreActionCheckResponse(
            action_id=request.action_id,
            decision=decision,
            violations=violations,
            risk_assessment=self._build_risk_assessment(violations),
            decision_proof_hash=self._generate_proof_hash(request, decision),
        )

    async def _check_rules(
        self,
        request: PreActionCheckRequest,
    ) -> RuleCheckResult:
        """Apply rule-based safety constraints."""
        return await self.rule_engine.check_action(
            action_type=request.action_type,
            parameters=dict(request.parameters),
            context=dict(request.context),
        )

    async def _check_erisml(
        self,
        request: PreActionCheckRequest,
    ) -> ErisMLCheckResult:
        """Perform ErisML ethical analysis."""
        # Convert to ErisML facts
        facts = self.erisml_service.build_facts(request)

        # Get ethical evaluation
        evaluation = await self.erisml_service.evaluate(facts)

        return ErisMLCheckResult(
            decision=self._map_erisml_decision(evaluation),
            violations=self._extract_erisml_violations(evaluation),
            bond_index=evaluation.bond_index,
            moral_vector=evaluation.moral_vector,
        )
```

#### 1.2 In-Action Monitoring Service

```python
# src/agi/safety/in_action/service.py
"""In-action safety monitoring service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import grpc

from proto.safety_pb2 import (
    ActionMonitorRequest,
    ActionMonitorResponse,
    SafetyDecision,
    EmergencyStopRequest,
    EmergencyStopResponse,
    ActionState,
)
from proto.safety_pb2_grpc import InActionSafetyServiceServicer

logger = logging.getLogger(__name__)


@dataclass
class MonitoredAction:
    """Tracked action state."""

    action_id: str
    agent_id: str
    start_time: float
    state: ActionState = ActionState.ACTION_STATE_RUNNING
    violations: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class InActionSafetyService(InActionSafetyServiceServicer):
    """Real-time action monitoring service.

    Monitors running actions for:
    - Constraint violations
    - Resource limit breaches
    - Anomalous behavior patterns
    - Emergency stop conditions
    """

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        reflex_layer: ReflexLayer | None = None,
    ) -> None:
        """Initialize in-action service."""
        self.rule_engine = rule_engine
        self.reflex_layer = reflex_layer
        self._active_actions: dict[str, MonitoredAction] = {}
        self._stop_events: dict[str, asyncio.Event] = {}

    async def MonitorAction(
        self,
        request_iterator: AsyncIterator[ActionMonitorRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[ActionMonitorResponse]:
        """Stream-based action monitoring."""
        async for request in request_iterator:
            action = self._get_or_create_action(request)

            # Update action state
            action.metrics.update(dict(request.metrics))

            # Check for violations
            violations = await self._check_runtime_constraints(action)

            # Check reflex layer (hardware safety)
            if self.reflex_layer:
                reflex_result = await self.reflex_layer.check(action)
                if reflex_result.emergency_stop:
                    yield ActionMonitorResponse(
                        action_id=request.action_id,
                        decision=SafetyDecision.SAFETY_DECISION_DENY,
                        emergency_stop=True,
                        stop_reason=reflex_result.reason,
                    )
                    await self._trigger_emergency_stop(action.action_id)
                    return

            if violations:
                action.violations.extend(violations)
                yield ActionMonitorResponse(
                    action_id=request.action_id,
                    decision=SafetyDecision.SAFETY_DECISION_MODIFY,
                    violations=violations,
                    suggested_modifications=self._suggest_modifications(violations),
                )
            else:
                yield ActionMonitorResponse(
                    action_id=request.action_id,
                    decision=SafetyDecision.SAFETY_DECISION_ALLOW,
                )

    async def EmergencyStop(
        self,
        request: EmergencyStopRequest,
        context: grpc.aio.ServicerContext,
    ) -> EmergencyStopResponse:
        """Trigger emergency stop for an action."""
        logger.warning(
            "Emergency stop requested: action=%s reason=%s",
            request.action_id,
            request.reason,
        )

        success = await self._trigger_emergency_stop(
            request.action_id,
            request.reason,
        )

        return EmergencyStopResponse(
            action_id=request.action_id,
            success=success,
            stopped_at=time.time(),
        )

    async def _trigger_emergency_stop(
        self,
        action_id: str,
        reason: str = "",
    ) -> bool:
        """Execute emergency stop procedure."""
        if action_id in self._stop_events:
            self._stop_events[action_id].set()

            # Notify reflex layer for hardware stop
            if self.reflex_layer:
                await self.reflex_layer.emergency_stop(action_id)

            return True
        return False
```

#### 1.3 Post-Action Audit Service

```python
# src/agi/safety/post_action/service.py
"""Post-action audit and learning service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import grpc

from proto.safety_pb2 import (
    PostActionAuditRequest,
    PostActionAuditResponse,
    OutcomeAssessment,
    DecisionProof,
    LearningSignal,
)
from proto.safety_pb2_grpc import PostActionSafetyServiceServicer

if TYPE_CHECKING:
    from agi.memory.episodic.service import EpisodicMemoryService

logger = logging.getLogger(__name__)


class PostActionSafetyService(PostActionSafetyServiceServicer):
    """Post-action audit and learning service.

    After action completion:
    - Assess actual outcomes
    - Compare to predicted outcomes
    - Generate learning signals
    - Store decision proofs
    - Update safety policies
    """

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        memory_service: EpisodicMemoryService | None = None,
        proof_store: DecisionProofStore | None = None,
    ) -> None:
        """Initialize post-action service."""
        self.rule_engine = rule_engine
        self.memory_service = memory_service
        self.proof_store = proof_store or InMemoryProofStore()

    async def AuditAction(
        self,
        request: PostActionAuditRequest,
        context: grpc.aio.ServicerContext,
    ) -> PostActionAuditResponse:
        """Audit a completed action."""
        logger.info(
            "Auditing action=%s outcome=%s",
            request.action_id,
            request.outcome_status,
        )

        # Retrieve decision proof
        proof = await self.proof_store.get(request.action_id)

        # Assess outcome against predictions
        assessment = await self._assess_outcome(request, proof)

        # Generate learning signals
        learning_signals = self._generate_learning_signals(assessment)

        # Update rule weights if applicable
        if learning_signals:
            await self._update_safety_model(learning_signals)

        # Store audit record
        audit_proof = await self._create_audit_proof(
            request, proof, assessment, learning_signals
        )

        # Store in episodic memory for future reference
        if self.memory_service:
            await self._store_in_memory(audit_proof)

        return PostActionAuditResponse(
            action_id=request.action_id,
            assessment=assessment,
            learning_signals=learning_signals,
            audit_proof_hash=audit_proof.hash,
        )

    async def _assess_outcome(
        self,
        request: PostActionAuditRequest,
        proof: DecisionProof | None,
    ) -> OutcomeAssessment:
        """Assess action outcome against predictions."""
        assessment = OutcomeAssessment()

        # Compare actual vs predicted impact
        if proof and proof.predicted_impact:
            assessment.prediction_accuracy = self._calculate_accuracy(
                predicted=proof.predicted_impact,
                actual=request.actual_impact,
            )

        # Check for unintended consequences
        assessment.unintended_consequences = self._detect_unintended(
            predicted=proof.predicted_impact if proof else None,
            actual=request.actual_impact,
        )

        # Assess ethical alignment
        if proof and proof.erisml_evaluation:
            assessment.ethical_alignment = self._assess_ethical_alignment(
                evaluation=proof.erisml_evaluation,
                outcome=request.actual_impact,
            )

        return assessment

    def _generate_learning_signals(
        self,
        assessment: OutcomeAssessment,
    ) -> list[LearningSignal]:
        """Generate signals for safety model updates."""
        signals = []

        # Prediction error signal
        if assessment.prediction_accuracy < 0.8:
            signals.append(LearningSignal(
                signal_type=LearningSignal.PREDICTION_ERROR,
                magnitude=1.0 - assessment.prediction_accuracy,
                context={"assessment": assessment.to_dict()},
            ))

        # Unintended consequence signal
        for consequence in assessment.unintended_consequences:
            signals.append(LearningSignal(
                signal_type=LearningSignal.UNINTENDED_CONSEQUENCE,
                magnitude=consequence.severity,
                context={"consequence": consequence.to_dict()},
            ))

        return signals
```

### Deliverables
- [ ] Full PreActionSafetyService implementation
- [ ] Full InActionSafetyService implementation
- [ ] Full PostActionSafetyService implementation
- [ ] SafetyGateway integration with all services
- [ ] Unit tests for all services
- [ ] Integration test suite

---

## Sprint 2: Enhanced Rule Engine (Weeks 3-4)

### Goals
- YAML-based rule definition and loading
- Rule DSL for complex constraints
- Rule priority and conflict resolution
- Hot-reload capability

### Tasks

#### 2.1 YAML Rule Schema

```yaml
# config/safety_rules/base_rules.yaml
version: "1.0"
metadata:
  name: "Base Safety Rules"
  description: "Core safety constraints for AGI-HPC"
  author: "Safety Team"

rules:
  # Physical safety rules
  - id: "reflex.emergency_stop"
    name: "Emergency Stop Override"
    layer: reflex
    priority: 1000
    conditions:
      any:
        - metric: "joint_torque_max"
          operator: ">"
          value: 100.0
        - metric: "collision_proximity"
          operator: "<"
          value: 0.05
        - signal: "emergency_button"
          value: true
    action:
      type: deny
      immediate: true
      trigger_hardware_stop: true

  - id: "tactical.workspace_bounds"
    name: "Workspace Boundary Check"
    layer: tactical
    priority: 500
    conditions:
      all:
        - field: "target_position.x"
          operator: "within"
          range: [-2.0, 2.0]
        - field: "target_position.y"
          operator: "within"
          range: [-2.0, 2.0]
        - field: "target_position.z"
          operator: "within"
          range: [0.0, 3.0]
    action:
      type: allow
    violation_action:
      type: modify
      clip_to_bounds: true

  - id: "tactical.velocity_limit"
    name: "Maximum Velocity Constraint"
    layer: tactical
    priority: 400
    conditions:
      field: "velocity_magnitude"
      operator: "<="
      value: 1.5  # m/s
    violation_action:
      type: modify
      scale_velocity: 0.5

  - id: "strategic.human_proximity"
    name: "Human Proximity Safety"
    layer: strategic
    priority: 300
    conditions:
      any:
        - field: "nearest_human_distance"
          operator: ">="
          value: 2.0
        - all:
            - field: "nearest_human_distance"
              operator: ">="
              value: 0.5
            - field: "velocity_magnitude"
              operator: "<="
              value: 0.3
    violation_action:
      type: defer
      reason: "Human too close for planned action"
      require_human_approval: true

  # Ethical rules
  - id: "strategic.harm_prevention"
    name: "Harm Prevention"
    layer: strategic
    priority: 900
    conditions:
      field: "predicted_harm_probability"
      operator: "<"
      value: 0.01
    violation_action:
      type: deny
      reason: "Unacceptable harm probability"
      log_level: critical

  - id: "strategic.resource_fairness"
    name: "Resource Allocation Fairness"
    layer: strategic
    priority: 200
    conditions:
      field: "resource_gini_coefficient"
      operator: "<="
      value: 0.3
    violation_action:
      type: modify
      redistribute: true
```

#### 2.2 Rule Engine Implementation

```python
# src/agi/safety/rules/engine.py
"""Enhanced safety rule engine with YAML loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class RuleLayer(Enum):
    """Safety rule layers."""
    REFLEX = "reflex"      # <100μs
    TACTICAL = "tactical"  # 10-100ms
    STRATEGIC = "strategic"  # 100ms-10s


class RuleAction(Enum):
    """Rule violation actions."""
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    DEFER = "defer"


@dataclass
class RuleCondition:
    """A rule condition to evaluate."""

    field: str | None = None
    metric: str | None = None
    signal: str | None = None
    operator: str = "=="
    value: Any = None
    range: tuple[float, float] | None = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Get the value to check
        if self.field:
            actual = self._get_nested(context, self.field)
        elif self.metric:
            actual = context.get("metrics", {}).get(self.metric)
        elif self.signal:
            actual = context.get("signals", {}).get(self.signal)
        else:
            return True

        # Apply operator
        if self.operator == "==":
            return actual == self.value
        elif self.operator == "!=":
            return actual != self.value
        elif self.operator == ">":
            return actual > self.value
        elif self.operator == ">=":
            return actual >= self.value
        elif self.operator == "<":
            return actual < self.value
        elif self.operator == "<=":
            return actual <= self.value
        elif self.operator == "within":
            return self.range[0] <= actual <= self.range[1]
        elif self.operator == "outside":
            return actual < self.range[0] or actual > self.range[1]

        return False

    def _get_nested(self, obj: dict, path: str) -> Any:
        """Get nested value by dot-separated path."""
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj


@dataclass
class SafetyRule:
    """A safety rule definition."""

    id: str
    name: str
    layer: RuleLayer
    priority: int
    conditions: list[RuleCondition] = field(default_factory=list)
    condition_mode: str = "all"  # "all" or "any"
    action: RuleAction = RuleAction.ALLOW
    violation_action: RuleAction = RuleAction.DENY
    violation_params: dict = field(default_factory=dict)
    enabled: bool = True

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        """Evaluate rule against context."""
        if not self.enabled:
            return RuleResult(rule_id=self.id, passed=True)

        # Evaluate conditions
        if self.condition_mode == "all":
            passed = all(c.evaluate(context) for c in self.conditions)
        else:  # "any"
            passed = any(c.evaluate(context) for c in self.conditions)

        return RuleResult(
            rule_id=self.id,
            passed=passed,
            action=self.action if passed else self.violation_action,
            params=self.violation_params if not passed else {},
        )


@dataclass
class RuleResult:
    """Result of rule evaluation."""

    rule_id: str
    passed: bool
    action: RuleAction = RuleAction.ALLOW
    params: dict = field(default_factory=dict)


class SafetyRuleEngine:
    """Enhanced safety rule engine."""

    def __init__(self) -> None:
        """Initialize rule engine."""
        self._rules: dict[str, SafetyRule] = {}
        self._rules_by_layer: dict[RuleLayer, list[SafetyRule]] = {
            layer: [] for layer in RuleLayer
        }
        self._file_watchers: list = []

    def load_rules(self, path: str | Path) -> int:
        """Load rules from YAML file or directory."""
        path = Path(path)
        count = 0

        if path.is_file():
            count += self._load_file(path)
        elif path.is_dir():
            for file in path.glob("**/*.yaml"):
                count += self._load_file(file)

        # Re-sort rules by priority
        for layer in RuleLayer:
            self._rules_by_layer[layer].sort(
                key=lambda r: r.priority, reverse=True
            )

        logger.info("Loaded %d safety rules from %s", count, path)
        return count

    def _load_file(self, path: Path) -> int:
        """Load rules from a single YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        count = 0
        for rule_data in data.get("rules", []):
            rule = self._parse_rule(rule_data)
            self._rules[rule.id] = rule
            self._rules_by_layer[rule.layer].append(rule)
            count += 1

        return count

    def _parse_rule(self, data: dict) -> SafetyRule:
        """Parse rule from YAML data."""
        conditions = []
        cond_data = data.get("conditions", {})

        if "all" in cond_data:
            mode = "all"
            cond_list = cond_data["all"]
        elif "any" in cond_data:
            mode = "any"
            cond_list = cond_data["any"]
        else:
            mode = "all"
            cond_list = [cond_data] if cond_data else []

        for c in cond_list:
            conditions.append(RuleCondition(
                field=c.get("field"),
                metric=c.get("metric"),
                signal=c.get("signal"),
                operator=c.get("operator", "=="),
                value=c.get("value"),
                range=tuple(c["range"]) if "range" in c else None,
            ))

        return SafetyRule(
            id=data["id"],
            name=data["name"],
            layer=RuleLayer(data["layer"]),
            priority=data.get("priority", 100),
            conditions=conditions,
            condition_mode=mode,
            action=RuleAction(data.get("action", {}).get("type", "allow")),
            violation_action=RuleAction(
                data.get("violation_action", {}).get("type", "deny")
            ),
            violation_params=data.get("violation_action", {}),
        )

    async def check_action(
        self,
        action_type: str,
        parameters: dict[str, Any],
        context: dict[str, Any],
        layer: RuleLayer | None = None,
    ) -> RuleCheckResult:
        """Check action against rules."""
        # Build full context
        full_context = {
            "action_type": action_type,
            "parameters": parameters,
            **context,
        }

        violations = []
        final_action = RuleAction.ALLOW

        # Check rules in priority order
        layers = [layer] if layer else list(RuleLayer)

        for check_layer in layers:
            for rule in self._rules_by_layer[check_layer]:
                result = rule.evaluate(full_context)

                if not result.passed:
                    violations.append(result)

                    # DENY overrides everything
                    if result.action == RuleAction.DENY:
                        final_action = RuleAction.DENY
                        break
                    # DEFER overrides MODIFY and ALLOW
                    elif (
                        result.action == RuleAction.DEFER
                        and final_action != RuleAction.DENY
                    ):
                        final_action = RuleAction.DEFER
                    # MODIFY overrides ALLOW
                    elif (
                        result.action == RuleAction.MODIFY
                        and final_action == RuleAction.ALLOW
                    ):
                        final_action = RuleAction.MODIFY

            if final_action == RuleAction.DENY:
                break

        return RuleCheckResult(
            action=final_action,
            violations=violations,
            is_blocking=(final_action == RuleAction.DENY),
        )

    def enable_hot_reload(self, path: str | Path) -> None:
        """Enable hot-reload of rules on file changes."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class RuleReloader(FileSystemEventHandler):
            def __init__(self, engine: SafetyRuleEngine, path: Path):
                self.engine = engine
                self.path = path

            def on_modified(self, event):
                if event.src_path.endswith(".yaml"):
                    logger.info("Reloading rules due to change: %s", event.src_path)
                    self.engine._rules.clear()
                    for layer in RuleLayer:
                        self.engine._rules_by_layer[layer].clear()
                    self.engine.load_rules(self.path)

        path = Path(path)
        observer = Observer()
        observer.schedule(RuleReloader(self, path), str(path), recursive=True)
        observer.start()
        self._file_watchers.append(observer)
```

### Deliverables
- [ ] YAML rule schema specification
- [ ] Rule parser and loader
- [ ] Condition evaluator with all operators
- [ ] Rule priority resolution
- [ ] Hot-reload capability
- [ ] Rule validation and linting tool
- [ ] Example rule sets for common scenarios

---

## Sprint 3: ErisML Full Integration (Weeks 5-6)

### Goals
- Connect to external erisml-lib service
- Full ethical evaluation pipeline
- Decision proof generation with hash chains
- Moral vector visualization

### Tasks

#### 3.1 ErisML Client Enhancement

```python
# src/agi/safety/erisml/client.py
"""Enhanced ErisML client with full ethical evaluation."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import grpc

from proto.erisml_pb2 import (
    EthicalFacts,
    EthicalEvaluationRequest,
    EthicalEvaluationResponse,
    BondIndex,
    MoralVector,
    HohfeldianAnalysis,
    DecisionProof,
)
from proto.erisml_pb2_grpc import ErisMLServiceStub

logger = logging.getLogger(__name__)


@dataclass
class EthicalEvaluation:
    """Complete ethical evaluation result."""

    request_id: str
    bond_index: BondIndex
    moral_vector: MoralVector
    hohfeldian: HohfeldianAnalysis
    decision: str  # ALLOW, DENY, DEFER
    confidence: float
    reasoning: list[str]
    proof: DecisionProof
    timestamp: datetime


class ErisMLClient:
    """Client for ErisML ethical reasoning service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50052,
        timeout: float = 5.0,
    ) -> None:
        """Initialize ErisML client."""
        self.address = f"{host}:{port}"
        self.timeout = timeout
        self._channel: grpc.aio.Channel | None = None
        self._stub: ErisMLServiceStub | None = None
        self._proof_chain: list[str] = []  # Hash chain

    async def connect(self) -> None:
        """Establish connection to ErisML service."""
        self._channel = grpc.aio.insecure_channel(self.address)
        self._stub = ErisMLServiceStub(self._channel)
        logger.info("Connected to ErisML service at %s", self.address)

    async def close(self) -> None:
        """Close connection."""
        if self._channel:
            await self._channel.close()

    async def evaluate(
        self,
        facts: EthicalFacts,
        context: dict[str, Any] | None = None,
    ) -> EthicalEvaluation:
        """Perform full ethical evaluation."""
        request = EthicalEvaluationRequest(
            request_id=self._generate_request_id(),
            facts=facts,
            context_json=json.dumps(context) if context else "{}",
            require_proof=True,
            previous_proof_hash=self._proof_chain[-1] if self._proof_chain else "",
        )

        try:
            response: EthicalEvaluationResponse = await self._stub.Evaluate(
                request,
                timeout=self.timeout,
            )
        except grpc.RpcError as e:
            logger.error("ErisML evaluation failed: %s", e)
            raise ErisMLEvaluationError(str(e)) from e

        # Add to proof chain
        self._proof_chain.append(response.proof.hash)

        return EthicalEvaluation(
            request_id=request.request_id,
            bond_index=response.bond_index,
            moral_vector=response.moral_vector,
            hohfeldian=response.hohfeldian,
            decision=self._interpret_decision(response),
            confidence=response.confidence,
            reasoning=list(response.reasoning_trace),
            proof=response.proof,
            timestamp=datetime.now(),
        )

    async def evaluate_plan(
        self,
        plan_steps: list[dict],
        stakeholders: list[str],
        context: dict[str, Any],
    ) -> list[EthicalEvaluation]:
        """Evaluate a multi-step plan."""
        evaluations = []

        for i, step in enumerate(plan_steps):
            facts = self._build_plan_step_facts(step, stakeholders, i)
            evaluation = await self.evaluate(facts, context)
            evaluations.append(evaluation)

            # Stop on DENY
            if evaluation.decision == "DENY":
                logger.warning(
                    "Plan step %d denied: %s",
                    i,
                    evaluation.reasoning,
                )
                break

        return evaluations

    def _build_plan_step_facts(
        self,
        step: dict,
        stakeholders: list[str],
        step_index: int,
    ) -> EthicalFacts:
        """Build EthicalFacts from plan step."""
        from agi.safety.erisml.facts_builder import FactsBuilder

        builder = FactsBuilder()
        return builder.from_plan_step(
            step=step,
            stakeholders=stakeholders,
            step_index=step_index,
        )

    def _interpret_decision(
        self,
        response: EthicalEvaluationResponse,
    ) -> str:
        """Interpret evaluation response as decision."""
        # Bond index thresholds
        if response.bond_index.value < 0.3:
            return "DENY"
        elif response.bond_index.value < 0.6:
            return "DEFER"
        else:
            return "ALLOW"

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())

    def get_proof_chain(self) -> list[str]:
        """Get the current decision proof chain."""
        return self._proof_chain.copy()

    def verify_proof_chain(self) -> bool:
        """Verify integrity of proof chain."""
        # Each hash should chain from previous
        for i in range(1, len(self._proof_chain)):
            # Verification logic would check signature and chain
            pass
        return True
```

#### 3.2 Decision Proof Storage

```python
# src/agi/safety/erisml/proof_store.py
"""Decision proof storage with PostgreSQL backend."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import asyncpg

from proto.erisml_pb2 import DecisionProof

logger = logging.getLogger(__name__)


class PostgresProofStore:
    """PostgreSQL-backed decision proof storage."""

    def __init__(
        self,
        dsn: str = "postgresql://agi:agi@localhost:5432/agi_safety",
    ) -> None:
        """Initialize proof store."""
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and schema."""
        self._pool = await asyncpg.create_pool(self.dsn)

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_proofs (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(255) UNIQUE NOT NULL,
                    action_id VARCHAR(255),
                    agent_id VARCHAR(255),
                    proof_hash VARCHAR(64) NOT NULL,
                    previous_hash VARCHAR(64),
                    bond_index REAL,
                    moral_vector JSONB,
                    hohfeldian JSONB,
                    decision VARCHAR(20),
                    confidence REAL,
                    reasoning JSONB,
                    facts JSONB,
                    context JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    verified BOOLEAN DEFAULT FALSE
                );

                CREATE INDEX IF NOT EXISTS idx_proofs_action
                    ON decision_proofs(action_id);
                CREATE INDEX IF NOT EXISTS idx_proofs_agent
                    ON decision_proofs(agent_id);
                CREATE INDEX IF NOT EXISTS idx_proofs_hash
                    ON decision_proofs(proof_hash);
                CREATE INDEX IF NOT EXISTS idx_proofs_created
                    ON decision_proofs(created_at);
            """)

    async def store(
        self,
        request_id: str,
        proof: DecisionProof,
        evaluation: dict[str, Any],
        action_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Store a decision proof."""
        async with self._pool.acquire() as conn:
            row_id = await conn.fetchval("""
                INSERT INTO decision_proofs (
                    request_id, action_id, agent_id, proof_hash,
                    previous_hash, bond_index, moral_vector, hohfeldian,
                    decision, confidence, reasoning, facts, context
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
            """,
                request_id,
                action_id,
                agent_id,
                proof.hash,
                proof.previous_hash,
                evaluation.get("bond_index"),
                json.dumps(evaluation.get("moral_vector", {})),
                json.dumps(evaluation.get("hohfeldian", {})),
                evaluation.get("decision"),
                evaluation.get("confidence"),
                json.dumps(evaluation.get("reasoning", [])),
                json.dumps(evaluation.get("facts", {})),
                json.dumps(evaluation.get("context", {})),
            )

            logger.debug("Stored decision proof %s -> row %d", request_id, row_id)
            return row_id

    async def get(self, request_id: str) -> dict | None:
        """Retrieve a decision proof by request ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM decision_proofs WHERE request_id = $1
            """, request_id)

            if row:
                return dict(row)
            return None

    async def get_by_action(self, action_id: str) -> list[dict]:
        """Get all proofs for an action."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM decision_proofs
                WHERE action_id = $1
                ORDER BY created_at
            """, action_id)

            return [dict(row) for row in rows]

    async def get_chain(
        self,
        start_hash: str,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve proof chain starting from hash."""
        chain = []
        current_hash = start_hash

        async with self._pool.acquire() as conn:
            while current_hash and len(chain) < limit:
                row = await conn.fetchrow("""
                    SELECT * FROM decision_proofs
                    WHERE proof_hash = $1
                """, current_hash)

                if row:
                    chain.append(dict(row))
                    current_hash = row["previous_hash"]
                else:
                    break

        return chain

    async def verify_chain_integrity(
        self,
        agent_id: str,
        start_time: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """Verify integrity of an agent's proof chain."""
        errors = []

        async with self._pool.acquire() as conn:
            query = """
                SELECT * FROM decision_proofs
                WHERE agent_id = $1
            """
            params = [agent_id]

            if start_time:
                query += " AND created_at >= $2"
                params.append(start_time)

            query += " ORDER BY created_at"
            rows = await conn.fetch(query, *params)

        # Verify chain links
        for i in range(1, len(rows)):
            if rows[i]["previous_hash"] != rows[i-1]["proof_hash"]:
                errors.append(
                    f"Chain break at {rows[i]['request_id']}: "
                    f"expected {rows[i-1]['proof_hash']}, "
                    f"got {rows[i]['previous_hash']}"
                )

        return len(errors) == 0, errors
```

### Deliverables
- [ ] Enhanced ErisML client with full evaluation
- [ ] Decision proof generation
- [ ] PostgreSQL proof storage
- [ ] Proof chain verification
- [ ] Moral vector visualization tool
- [ ] Integration tests with erisml-lib

---

## Sprint 4: Reflex Layer (Weeks 7-8)

### Goals
- Hardware safety integration (<100μs response)
- Emergency stop mechanisms
- Sensor-based safety triggers
- Real-time constraint enforcement

### Tasks

#### 4.1 Reflex Layer Interface

```python
# src/agi/safety/reflex/layer.py
"""Hardware-level reflex safety layer."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class ReflexTrigger(Enum):
    """Reflex trigger types."""
    COLLISION = "collision"
    FORCE_LIMIT = "force_limit"
    VELOCITY_LIMIT = "velocity_limit"
    TEMPERATURE = "temperature"
    EMERGENCY_BUTTON = "emergency_button"
    WATCHDOG = "watchdog"
    CUSTOM = "custom"


@dataclass
class ReflexEvent:
    """A reflex layer event."""

    trigger: ReflexTrigger
    timestamp_ns: int  # Nanoseconds for precision
    value: float
    threshold: float
    source: str
    action_id: str | None = None


@dataclass
class ReflexResult:
    """Result from reflex layer check."""

    safe: bool
    emergency_stop: bool = False
    reason: str = ""
    event: ReflexEvent | None = None
    latency_us: float = 0.0  # Microseconds


class ReflexLayerBase(ABC):
    """Abstract base for reflex layer implementations."""

    @abstractmethod
    async def check(self, state: dict) -> ReflexResult:
        """Check current state against reflex constraints."""
        ...

    @abstractmethod
    async def emergency_stop(self, action_id: str | None = None) -> bool:
        """Trigger emergency stop."""
        ...

    @abstractmethod
    async def register_trigger(
        self,
        trigger_type: ReflexTrigger,
        threshold: float,
        callback: Callable[[ReflexEvent], None],
    ) -> str:
        """Register a custom trigger."""
        ...


class SoftwareReflexLayer(ReflexLayerBase):
    """Software-based reflex layer for simulation/testing."""

    def __init__(self) -> None:
        """Initialize software reflex layer."""
        self._triggers: dict[str, dict] = {}
        self._emergency_stop_active = False
        self._callbacks: list[Callable] = []

        # Default thresholds
        self._thresholds = {
            ReflexTrigger.COLLISION: 0.02,  # 2cm proximity
            ReflexTrigger.FORCE_LIMIT: 100.0,  # 100N
            ReflexTrigger.VELOCITY_LIMIT: 2.0,  # 2 m/s
            ReflexTrigger.TEMPERATURE: 80.0,  # 80°C
        }

    async def check(self, state: dict) -> ReflexResult:
        """Check state against reflex constraints."""
        import time
        start = time.perf_counter_ns()

        # Check collision proximity
        if "collision_proximity" in state:
            if state["collision_proximity"] < self._thresholds[ReflexTrigger.COLLISION]:
                event = ReflexEvent(
                    trigger=ReflexTrigger.COLLISION,
                    timestamp_ns=time.time_ns(),
                    value=state["collision_proximity"],
                    threshold=self._thresholds[ReflexTrigger.COLLISION],
                    source="proximity_sensor",
                )
                return ReflexResult(
                    safe=False,
                    emergency_stop=True,
                    reason=f"Collision imminent: {state['collision_proximity']:.3f}m",
                    event=event,
                    latency_us=(time.perf_counter_ns() - start) / 1000,
                )

        # Check force limits
        if "joint_forces" in state:
            max_force = max(state["joint_forces"])
            if max_force > self._thresholds[ReflexTrigger.FORCE_LIMIT]:
                event = ReflexEvent(
                    trigger=ReflexTrigger.FORCE_LIMIT,
                    timestamp_ns=time.time_ns(),
                    value=max_force,
                    threshold=self._thresholds[ReflexTrigger.FORCE_LIMIT],
                    source="force_sensor",
                )
                return ReflexResult(
                    safe=False,
                    emergency_stop=True,
                    reason=f"Force limit exceeded: {max_force:.1f}N",
                    event=event,
                    latency_us=(time.perf_counter_ns() - start) / 1000,
                )

        # Check velocity
        if "velocity_magnitude" in state:
            if state["velocity_magnitude"] > self._thresholds[ReflexTrigger.VELOCITY_LIMIT]:
                event = ReflexEvent(
                    trigger=ReflexTrigger.VELOCITY_LIMIT,
                    timestamp_ns=time.time_ns(),
                    value=state["velocity_magnitude"],
                    threshold=self._thresholds[ReflexTrigger.VELOCITY_LIMIT],
                    source="velocity_sensor",
                )
                return ReflexResult(
                    safe=False,
                    emergency_stop=False,  # Reduce, don't stop
                    reason=f"Velocity limit: {state['velocity_magnitude']:.2f}m/s",
                    event=event,
                    latency_us=(time.perf_counter_ns() - start) / 1000,
                )

        latency = (time.perf_counter_ns() - start) / 1000
        return ReflexResult(safe=True, latency_us=latency)

    async def emergency_stop(self, action_id: str | None = None) -> bool:
        """Trigger emergency stop."""
        logger.critical("EMERGENCY STOP triggered for action %s", action_id)
        self._emergency_stop_active = True

        # Notify all callbacks
        for callback in self._callbacks:
            try:
                callback(ReflexEvent(
                    trigger=ReflexTrigger.EMERGENCY_BUTTON,
                    timestamp_ns=time.time_ns(),
                    value=1.0,
                    threshold=0.0,
                    source="software_stop",
                    action_id=action_id,
                ))
            except Exception as e:
                logger.error("Emergency callback error: %s", e)

        return True

    async def register_trigger(
        self,
        trigger_type: ReflexTrigger,
        threshold: float,
        callback: Callable[[ReflexEvent], None],
    ) -> str:
        """Register custom trigger."""
        import uuid
        trigger_id = str(uuid.uuid4())

        self._triggers[trigger_id] = {
            "type": trigger_type,
            "threshold": threshold,
            "callback": callback,
        }
        self._thresholds[trigger_type] = threshold
        self._callbacks.append(callback)

        return trigger_id


class HardwareReflexLayer(ReflexLayerBase):
    """Hardware-integrated reflex layer for real robots."""

    def __init__(
        self,
        hardware_interface: str = "ethercat",
        safety_controller: str = "localhost:5000",
    ) -> None:
        """Initialize hardware reflex layer."""
        self.hardware_interface = hardware_interface
        self.safety_controller = safety_controller
        self._connected = False

    async def connect(self) -> None:
        """Connect to hardware safety controller."""
        # Implementation depends on hardware
        logger.info(
            "Connecting to hardware safety controller: %s via %s",
            self.safety_controller,
            self.hardware_interface,
        )
        self._connected = True

    async def check(self, state: dict) -> ReflexResult:
        """Check via hardware safety controller."""
        if not self._connected:
            await self.connect()

        # Hardware check is performed by dedicated safety PLC
        # This just queries the current status
        import time
        start = time.perf_counter_ns()

        # Query safety controller status
        # ... hardware-specific implementation ...

        latency = (time.perf_counter_ns() - start) / 1000
        return ReflexResult(safe=True, latency_us=latency)

    async def emergency_stop(self, action_id: str | None = None) -> bool:
        """Hardware emergency stop."""
        logger.critical(
            "HARDWARE EMERGENCY STOP for action %s",
            action_id,
        )
        # Send stop command to safety PLC
        # ... hardware-specific implementation ...
        return True

    async def register_trigger(
        self,
        trigger_type: ReflexTrigger,
        threshold: float,
        callback: Callable[[ReflexEvent], None],
    ) -> str:
        """Register trigger with hardware controller."""
        # Configure safety PLC
        # ... hardware-specific implementation ...
        import uuid
        return str(uuid.uuid4())
```

### Deliverables
- [ ] Reflex layer interface
- [ ] Software reflex implementation
- [ ] Hardware reflex interface
- [ ] <100μs latency validation
- [ ] Emergency stop integration
- [ ] Sensor abstraction layer

---

## Sprint 5: Human Oversight (Weeks 9-10)

### Goals
- DEFER queue management
- Human review workflow
- Override audit trail
- Real-time notification system

### Tasks

#### 5.1 DEFER Queue Implementation

```python
# src/agi/safety/oversight/defer_queue.py
"""Human oversight DEFER queue management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import asyncpg

logger = logging.getLogger(__name__)


class DeferPriority(Enum):
    """DEFER request priority levels."""
    CRITICAL = 1  # <1 minute response needed
    HIGH = 2      # <5 minutes
    NORMAL = 3    # <30 minutes
    LOW = 4       # <24 hours


class DeferStatus(Enum):
    """DEFER request status."""
    PENDING = "pending"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    DENIED = "denied"
    MODIFIED = "modified"
    EXPIRED = "expired"


@dataclass
class DeferRequest:
    """A deferred action requiring human review."""

    request_id: str
    action_id: str
    agent_id: str
    action_type: str
    parameters: dict[str, Any]
    reason: str
    priority: DeferPriority
    status: DeferStatus = DeferStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    reviewer_id: str | None = None
    review_notes: str = ""
    modified_parameters: dict[str, Any] | None = None


class DeferQueue:
    """Human oversight DEFER queue."""

    def __init__(
        self,
        dsn: str = "postgresql://agi:agi@localhost:5432/agi_safety",
        notification_callback: Callable[[DeferRequest], None] | None = None,
    ) -> None:
        """Initialize DEFER queue."""
        self.dsn = dsn
        self.notification_callback = notification_callback
        self._pool: asyncpg.Pool | None = None
        self._expiry_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize queue storage."""
        self._pool = await asyncpg.create_pool(self.dsn)

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS defer_queue (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(255) UNIQUE NOT NULL,
                    action_id VARCHAR(255) NOT NULL,
                    agent_id VARCHAR(255) NOT NULL,
                    action_type VARCHAR(255) NOT NULL,
                    parameters JSONB NOT NULL,
                    reason TEXT,
                    priority INTEGER NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    context JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    reviewer_id VARCHAR(255),
                    reviewed_at TIMESTAMPTZ,
                    review_notes TEXT,
                    modified_parameters JSONB
                );

                CREATE INDEX IF NOT EXISTS idx_defer_status
                    ON defer_queue(status);
                CREATE INDEX IF NOT EXISTS idx_defer_priority
                    ON defer_queue(priority, created_at);
                CREATE INDEX IF NOT EXISTS idx_defer_expires
                    ON defer_queue(expires_at) WHERE status = 'pending';
            """)

        # Start expiry checker
        self._expiry_task = asyncio.create_task(self._check_expiry_loop())

    async def submit(self, request: DeferRequest) -> str:
        """Submit action for human review."""
        # Calculate expiry based on priority
        if request.expires_at is None:
            ttl_map = {
                DeferPriority.CRITICAL: timedelta(minutes=1),
                DeferPriority.HIGH: timedelta(minutes=5),
                DeferPriority.NORMAL: timedelta(minutes=30),
                DeferPriority.LOW: timedelta(hours=24),
            }
            request.expires_at = datetime.now() + ttl_map[request.priority]

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO defer_queue (
                    request_id, action_id, agent_id, action_type,
                    parameters, reason, priority, context, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                request.request_id,
                request.action_id,
                request.agent_id,
                request.action_type,
                json.dumps(request.parameters),
                request.reason,
                request.priority.value,
                json.dumps(request.context),
                request.expires_at,
            )

        logger.info(
            "DEFER submitted: %s priority=%s expires=%s",
            request.request_id,
            request.priority.name,
            request.expires_at,
        )

        # Send notification
        if self.notification_callback:
            self.notification_callback(request)

        return request.request_id

    async def get_pending(
        self,
        limit: int = 50,
        priority: DeferPriority | None = None,
    ) -> list[DeferRequest]:
        """Get pending requests for review."""
        query = """
            SELECT * FROM defer_queue
            WHERE status = 'pending'
        """
        params = []

        if priority:
            query += " AND priority = $1"
            params.append(priority.value)

        query += " ORDER BY priority, created_at LIMIT $" + str(len(params) + 1)
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_request(row) for row in rows]

    async def claim(
        self,
        request_id: str,
        reviewer_id: str,
    ) -> bool:
        """Claim a request for review."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE defer_queue
                SET status = 'reviewing', reviewer_id = $2
                WHERE request_id = $1 AND status = 'pending'
            """, request_id, reviewer_id)

        return "UPDATE 1" in result

    async def approve(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str = "",
    ) -> DeferRequest | None:
        """Approve a deferred action."""
        return await self._complete_review(
            request_id, reviewer_id, DeferStatus.APPROVED, notes
        )

    async def deny(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str = "",
    ) -> DeferRequest | None:
        """Deny a deferred action."""
        return await self._complete_review(
            request_id, reviewer_id, DeferStatus.DENIED, notes
        )

    async def modify(
        self,
        request_id: str,
        reviewer_id: str,
        modified_parameters: dict[str, Any],
        notes: str = "",
    ) -> DeferRequest | None:
        """Approve with modifications."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                UPDATE defer_queue
                SET status = 'modified',
                    reviewer_id = $2,
                    reviewed_at = NOW(),
                    review_notes = $3,
                    modified_parameters = $4
                WHERE request_id = $1
                RETURNING *
            """, request_id, reviewer_id, notes, json.dumps(modified_parameters))

        if result:
            return self._row_to_request(result)
        return None

    async def wait_for_decision(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> DeferRequest | None:
        """Wait for human decision on a request."""
        start = datetime.now()

        while True:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM defer_queue WHERE request_id = $1
                """, request_id)

            if row:
                request = self._row_to_request(row)
                if request.status != DeferStatus.PENDING:
                    return request

            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= timeout:
                    return None

            await asyncio.sleep(0.5)

    async def _complete_review(
        self,
        request_id: str,
        reviewer_id: str,
        status: DeferStatus,
        notes: str,
    ) -> DeferRequest | None:
        """Complete a review with given status."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow("""
                UPDATE defer_queue
                SET status = $2,
                    reviewer_id = $3,
                    reviewed_at = NOW(),
                    review_notes = $4
                WHERE request_id = $1
                RETURNING *
            """, request_id, status.value, reviewer_id, notes)

        if result:
            return self._row_to_request(result)
        return None

    async def _check_expiry_loop(self) -> None:
        """Background task to expire old requests."""
        while True:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE defer_queue
                        SET status = 'expired'
                        WHERE status = 'pending'
                          AND expires_at < NOW()
                    """)
            except Exception as e:
                logger.error("Expiry check error: %s", e)

            await asyncio.sleep(10)

    def _row_to_request(self, row) -> DeferRequest:
        """Convert database row to DeferRequest."""
        return DeferRequest(
            request_id=row["request_id"],
            action_id=row["action_id"],
            agent_id=row["agent_id"],
            action_type=row["action_type"],
            parameters=json.loads(row["parameters"]),
            reason=row["reason"],
            priority=DeferPriority(row["priority"]),
            status=DeferStatus(row["status"]),
            context=json.loads(row["context"]) if row["context"] else {},
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            reviewer_id=row["reviewer_id"],
            review_notes=row["review_notes"] or "",
            modified_parameters=(
                json.loads(row["modified_parameters"])
                if row["modified_parameters"] else None
            ),
        )
```

#### 5.2 Review API

```python
# src/agi/safety/oversight/api.py
"""Human oversight review API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

from agi.safety.oversight.defer_queue import DeferQueue, DeferStatus

app = FastAPI(title="AGI Safety Oversight API")


class ReviewDecision(BaseModel):
    """Review decision payload."""

    reviewer_id: str
    decision: str  # approve, deny, modify
    notes: str = ""
    modified_parameters: dict | None = None


@app.get("/api/v1/defer/pending")
async def get_pending_requests(
    limit: int = 50,
    priority: int | None = None,
):
    """Get pending DEFER requests."""
    queue = get_defer_queue()
    requests = await queue.get_pending(limit=limit, priority=priority)
    return {"requests": [r.__dict__ for r in requests]}


@app.post("/api/v1/defer/{request_id}/claim")
async def claim_request(request_id: str, reviewer_id: str):
    """Claim a request for review."""
    queue = get_defer_queue()
    success = await queue.claim(request_id, reviewer_id)
    if not success:
        raise HTTPException(400, "Request already claimed or not found")
    return {"claimed": True}


@app.post("/api/v1/defer/{request_id}/review")
async def submit_review(request_id: str, decision: ReviewDecision):
    """Submit review decision."""
    queue = get_defer_queue()

    if decision.decision == "approve":
        result = await queue.approve(
            request_id, decision.reviewer_id, decision.notes
        )
    elif decision.decision == "deny":
        result = await queue.deny(
            request_id, decision.reviewer_id, decision.notes
        )
    elif decision.decision == "modify":
        if not decision.modified_parameters:
            raise HTTPException(400, "Modified parameters required")
        result = await queue.modify(
            request_id,
            decision.reviewer_id,
            decision.modified_parameters,
            decision.notes,
        )
    else:
        raise HTTPException(400, f"Invalid decision: {decision.decision}")

    if not result:
        raise HTTPException(404, "Request not found")

    return {"success": True, "status": result.status.value}


@app.websocket("/ws/defer/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket for real-time DEFER updates."""
    await websocket.accept()

    # Subscribe to DEFER queue updates
    queue = get_defer_queue()

    async def on_new_request(request):
        await websocket.send_json({
            "type": "new_request",
            "request": request.__dict__,
        })

    # Set callback
    queue.notification_callback = on_new_request

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        pass
```

### Deliverables
- [ ] DEFER queue with PostgreSQL backend
- [ ] Priority-based expiry handling
- [ ] REST API for review workflow
- [ ] WebSocket real-time updates
- [ ] Review dashboard UI
- [ ] Audit trail for all decisions

---

## Sprint 6: Safety Learning (Weeks 11-12)

### Goals
- Learn from action outcomes
- Update rule weights
- Anomaly detection
- Confidence calibration

### Tasks

#### 6.1 Safety Learning Service

```python
# src/agi/safety/learning/service.py
"""Safety learning from outcomes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from agi.safety.rules.engine import SafetyRuleEngine, SafetyRule

logger = logging.getLogger(__name__)


@dataclass
class OutcomeFeedback:
    """Feedback from action outcome."""

    action_id: str
    predicted_safe: bool
    actual_safe: bool
    violated_rules: list[str]
    unintended_effects: list[str]
    severity: float  # 0-1


class SafetyLearner:
    """Online learning for safety rule weights."""

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        learning_rate: float = 0.01,
        min_samples: int = 10,
    ) -> None:
        """Initialize safety learner."""
        self.rule_engine = rule_engine
        self.learning_rate = learning_rate
        self.min_samples = min_samples

        # Rule performance tracking
        self._rule_stats: dict[str, RuleStats] = {}
        self._false_negatives: list[OutcomeFeedback] = []
        self._false_positives: list[OutcomeFeedback] = []

    def record_outcome(self, feedback: OutcomeFeedback) -> None:
        """Record action outcome for learning."""
        # Track false negatives (predicted safe, was unsafe)
        if feedback.predicted_safe and not feedback.actual_safe:
            self._false_negatives.append(feedback)
            logger.warning(
                "False negative: action %s was unsafe",
                feedback.action_id,
            )

            # Increase sensitivity of violated rules
            for rule_id in feedback.violated_rules:
                self._adjust_rule_weight(rule_id, +self.learning_rate)

        # Track false positives (predicted unsafe, was safe)
        elif not feedback.predicted_safe and feedback.actual_safe:
            self._false_positives.append(feedback)

            # Slightly decrease sensitivity (be conservative)
            for rule_id in feedback.violated_rules:
                self._adjust_rule_weight(rule_id, -self.learning_rate * 0.5)

        # Update rule statistics
        for rule_id in feedback.violated_rules:
            stats = self._get_or_create_stats(rule_id)
            stats.total_triggers += 1
            if not feedback.actual_safe:
                stats.true_positives += 1
            else:
                stats.false_positives += 1

    def _adjust_rule_weight(self, rule_id: str, delta: float) -> None:
        """Adjust rule priority/weight."""
        if rule_id in self.rule_engine._rules:
            rule = self.rule_engine._rules[rule_id]
            # Clamp priority between 1 and 1000
            new_priority = max(1, min(1000, rule.priority + int(delta * 100)))

            logger.info(
                "Adjusting rule %s priority: %d -> %d",
                rule_id, rule.priority, new_priority,
            )
            rule.priority = new_priority

    def get_rule_performance(self) -> dict[str, dict]:
        """Get performance metrics for all rules."""
        result = {}

        for rule_id, stats in self._rule_stats.items():
            if stats.total_triggers >= self.min_samples:
                precision = (
                    stats.true_positives /
                    (stats.true_positives + stats.false_positives)
                    if (stats.true_positives + stats.false_positives) > 0
                    else 0.0
                )
                result[rule_id] = {
                    "total_triggers": stats.total_triggers,
                    "true_positives": stats.true_positives,
                    "false_positives": stats.false_positives,
                    "precision": precision,
                }

        return result

    def detect_anomalies(
        self,
        recent_window: int = 100,
    ) -> list[str]:
        """Detect anomalous safety patterns."""
        anomalies = []

        # Check for sudden increase in false negatives
        if len(self._false_negatives) >= 3:
            recent = self._false_negatives[-recent_window:]
            # ... statistical analysis ...
            pass

        # Check for rules with poor precision
        for rule_id, stats in self._rule_stats.items():
            if stats.total_triggers >= self.min_samples:
                precision = (
                    stats.true_positives /
                    (stats.true_positives + stats.false_positives)
                )
                if precision < 0.5:
                    anomalies.append(
                        f"Rule {rule_id} has low precision: {precision:.2f}"
                    )

        return anomalies

    def _get_or_create_stats(self, rule_id: str) -> RuleStats:
        """Get or create stats for a rule."""
        if rule_id not in self._rule_stats:
            self._rule_stats[rule_id] = RuleStats()
        return self._rule_stats[rule_id]


@dataclass
class RuleStats:
    """Statistics for a safety rule."""

    total_triggers: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
```

### Deliverables
- [ ] Outcome feedback collection
- [ ] Rule weight adjustment
- [ ] Anomaly detection
- [ ] Performance metrics dashboard
- [ ] Automatic rule refinement

---

## Sprint 7: Testing and Verification (Weeks 13-14)

### Goals
- Comprehensive test suite
- Fuzzing for edge cases
- Formal verification of critical paths
- Chaos testing

### Tasks

#### 7.1 Safety Test Framework

```python
# tests/safety/test_framework.py
"""Safety subsystem test framework."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import pytest

from agi.safety.gateway import SafetyGateway
from agi.safety.rules.engine import SafetyRuleEngine


@dataclass
class SafetyTestCase:
    """A safety test case."""

    name: str
    action_type: str
    parameters: dict[str, Any]
    context: dict[str, Any]
    expected_decision: str  # ALLOW, DENY, DEFER
    expected_violations: list[str] = None

    def __post_init__(self):
        self.expected_violations = self.expected_violations or []


class SafetyFuzzer:
    """Fuzzer for safety subsystem."""

    def __init__(self, gateway: SafetyGateway) -> None:
        """Initialize fuzzer."""
        self.gateway = gateway
        self._test_cases: list[SafetyTestCase] = []
        self._failures: list[tuple[SafetyTestCase, Exception]] = []

    def generate_test_cases(self, count: int = 1000) -> list[SafetyTestCase]:
        """Generate random test cases."""
        cases = []

        for _ in range(count):
            # Random action type
            action_type = random.choice([
                "move", "grasp", "release", "push", "pull", "rotate",
            ])

            # Random parameters with edge cases
            parameters = {
                "target_position": {
                    "x": random.uniform(-10, 10),
                    "y": random.uniform(-10, 10),
                    "z": random.uniform(-5, 10),
                },
                "velocity": random.uniform(0, 5),
                "force": random.uniform(0, 200),
            }

            # Random context
            context = {
                "nearest_human_distance": random.uniform(0, 10),
                "collision_proximity": random.uniform(0, 1),
            }

            # Determine expected outcome based on rules
            expected = self._predict_outcome(action_type, parameters, context)

            cases.append(SafetyTestCase(
                name=f"fuzz_{len(cases)}",
                action_type=action_type,
                parameters=parameters,
                context=context,
                expected_decision=expected,
            ))

        return cases

    async def run_fuzz_campaign(
        self,
        duration_seconds: float = 60,
    ) -> dict[str, Any]:
        """Run fuzzing campaign."""
        import time
        start = time.time()
        tested = 0
        failures = 0

        while time.time() - start < duration_seconds:
            cases = self.generate_test_cases(100)

            for case in cases:
                try:
                    result = await self.gateway.check_action(
                        action_type=case.action_type,
                        parameters=case.parameters,
                        context=case.context,
                    )

                    if result.decision.name != case.expected_decision:
                        self._failures.append((case, None))
                        failures += 1

                    tested += 1
                except Exception as e:
                    self._failures.append((case, e))
                    failures += 1

        return {
            "tested": tested,
            "failures": failures,
            "failure_rate": failures / tested if tested > 0 else 0,
            "duration": time.time() - start,
        }

    def _predict_outcome(
        self,
        action_type: str,
        parameters: dict,
        context: dict,
    ) -> str:
        """Predict expected outcome based on rules."""
        # Emergency conditions -> DENY
        if context.get("collision_proximity", 1) < 0.05:
            return "DENY"

        # High force -> DENY
        if parameters.get("force", 0) > 100:
            return "DENY"

        # Human too close -> DEFER
        if context.get("nearest_human_distance", 10) < 0.5:
            return "DEFER"

        # Out of bounds -> DENY
        pos = parameters.get("target_position", {})
        if abs(pos.get("x", 0)) > 2 or abs(pos.get("y", 0)) > 2:
            return "DENY"

        return "ALLOW"


# Pytest fixtures and tests
@pytest.fixture
async def safety_gateway():
    """Create safety gateway for testing."""
    gateway = SafetyGateway()
    await gateway.initialize()
    yield gateway
    await gateway.shutdown()


@pytest.mark.asyncio
async def test_emergency_stop(safety_gateway):
    """Test emergency stop triggers correctly."""
    result = await safety_gateway.check_action(
        action_type="move",
        parameters={"velocity": 1.0},
        context={"collision_proximity": 0.01},
    )
    assert result.decision.name == "DENY"
    assert result.emergency_stop


@pytest.mark.asyncio
async def test_human_proximity_defer(safety_gateway):
    """Test human proximity triggers DEFER."""
    result = await safety_gateway.check_action(
        action_type="move",
        parameters={"velocity": 0.5},
        context={"nearest_human_distance": 0.3},
    )
    assert result.decision.name == "DEFER"


@pytest.mark.asyncio
async def test_fuzzing_campaign(safety_gateway):
    """Run fuzzing campaign."""
    fuzzer = SafetyFuzzer(safety_gateway)
    results = await fuzzer.run_fuzz_campaign(duration_seconds=10)

    assert results["failure_rate"] < 0.01, (
        f"Too many failures: {results['failures']}/{results['tested']}"
    )
```

### Deliverables
- [ ] Unit test suite for all services
- [ ] Integration test suite
- [ ] Fuzzing framework
- [ ] Property-based testing
- [ ] Chaos testing for failure scenarios
- [ ] Performance benchmarks

---

## Sprint 8: Production Hardening (Weeks 15-16)

### Goals
- High availability configuration
- Disaster recovery
- Security hardening
- Production monitoring

### Tasks

#### 8.1 HA Configuration

```yaml
# deploy/safety/ha-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: safety-config
  namespace: agi-hpc
data:
  safety.yaml: |
    service:
      replicas: 3
      leader_election: true

    pre_action:
      timeout_ms: 100
      circuit_breaker:
        failure_threshold: 5
        recovery_timeout_s: 30

    in_action:
      streaming: true
      buffer_size: 1000

    post_action:
      async: true
      batch_size: 50

    erisml:
      endpoint: "erisml-service:50052"
      timeout_ms: 5000
      fallback: "deny"  # Deny if ErisML unavailable

    reflex:
      layer: hardware
      controller: "safety-plc:5000"
      watchdog_ms: 10

    oversight:
      defer_queue:
        postgres_dsn: "postgresql://safety:***@postgres:5432/safety"
        redis_url: "redis://redis:6379/0"
      notifications:
        slack_webhook: "${SLACK_WEBHOOK}"
        pagerduty_key: "${PAGERDUTY_KEY}"

    rules:
      path: /etc/agi/safety/rules
      hot_reload: true
      validation: strict

    audit:
      proof_store:
        type: postgres
        dsn: "postgresql://safety:***@postgres:5432/safety"
        retention_days: 365
      export:
        enabled: true
        format: parquet
        s3_bucket: "agi-safety-audit"

    monitoring:
      prometheus:
        enabled: true
        port: 9090
      tracing:
        enabled: true
        jaeger_endpoint: "jaeger:6831"
```

#### 8.2 Security Configuration

```yaml
# deploy/safety/security.yaml
apiVersion: v1
kind: Secret
metadata:
  name: safety-secrets
  namespace: agi-hpc
type: Opaque
data:
  postgres-password: <base64>
  erisml-api-key: <base64>
  jwt-secret: <base64>
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: safety-network-policy
  namespace: agi-hpc
spec:
  podSelector:
    matchLabels:
      app: safety-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: agi-core
      ports:
        - protocol: TCP
          port: 50051
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: erisml-service
      ports:
        - protocol: TCP
          port: 50052
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### Deliverables
- [ ] HA deployment configuration
- [ ] Disaster recovery procedures
- [ ] Security hardening
- [ ] mTLS for all communications
- [ ] Audit log export
- [ ] Production runbook

---

## File Structure

```
src/agi/safety/
├── __init__.py
├── gateway.py                  # SafetyGateway (exists)
├── pre_action/
│   ├── __init__.py
│   └── service.py             # PreActionSafetyService
├── in_action/
│   ├── __init__.py
│   └── service.py             # InActionSafetyService
├── post_action/
│   ├── __init__.py
│   └── service.py             # PostActionSafetyService
├── rules/
│   ├── __init__.py
│   ├── engine.py              # SafetyRuleEngine (enhanced)
│   ├── parser.py              # YAML rule parser
│   └── dsl.py                 # Rule DSL
├── reflex/
│   ├── __init__.py
│   ├── layer.py               # ReflexLayer interface
│   ├── software.py            # Software reflex
│   └── hardware.py            # Hardware reflex
├── erisml/
│   ├── __init__.py
│   ├── client.py              # ErisML client (enhanced)
│   ├── service.py             # ErisML gRPC service (exists)
│   ├── facts_builder.py       # Facts builder (exists)
│   └── proof_store.py         # Decision proof storage
├── oversight/
│   ├── __init__.py
│   ├── defer_queue.py         # DEFER queue
│   ├── api.py                 # Review REST API
│   └── notifications.py       # Alert notifications
├── learning/
│   ├── __init__.py
│   ├── service.py             # Safety learning
│   └── anomaly.py             # Anomaly detection
└── metrics.py                 # Prometheus metrics

config/safety_rules/
├── base_rules.yaml
├── physical_safety.yaml
├── ethical_constraints.yaml
└── domain/
    ├── robotics.yaml
    ├── healthcare.yaml
    └── finance.yaml

tests/safety/
├── conftest.py
├── test_pre_action.py
├── test_in_action.py
├── test_post_action.py
├── test_rules_engine.py
├── test_reflex_layer.py
├── test_erisml_integration.py
├── test_defer_queue.py
├── test_learning.py
├── test_framework.py          # Fuzzing framework
└── benchmarks/
    ├── bench_latency.py
    └── bench_throughput.py
```

---

## Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
safety = [
    "grpcio>=1.60.0",
    "grpcio-tools>=1.60.0",
    "asyncpg>=0.29.0",
    "redis>=5.0.0",
    "pyyaml>=6.0.1",
    "watchdog>=3.0.0",
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "websockets>=12.0",
    "prometheus-client>=0.19.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "httpx>=0.26.0",  # For ErisML client
]
```

---

## Quick Start

```bash
# Start safety infrastructure
docker compose -f deploy/safety/docker-compose.yaml up -d

# Load safety rules
python -m agi.safety.rules.loader config/safety_rules/

# Start safety services
python -m agi.safety.server --config config/safety.yaml

# Run tests
pytest tests/safety/ -v

# Run fuzzing
python -m agi.safety.fuzz --duration 60

# Start oversight API
uvicorn agi.safety.oversight.api:app --host 0.0.0.0 --port 8080
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Reflex latency | <100μs | 99th percentile |
| Pre-action latency | <100ms | 99th percentile |
| False negative rate | <0.01% | Monthly audit |
| DEFER response time | <5min | P50 human review |
| Proof chain integrity | 100% | Continuous verification |
| Rule coverage | >95% | Safety scenario tests |
| ErisML availability | >99.9% | Uptime monitoring |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ErisML service unavailable | Fallback to conservative DENY |
| Database failure | Redis cache + async recovery |
| Human reviewer unavailable | Escalation chain + auto-expire |
| Rule misconfiguration | Validation + canary deployment |
| Hardware safety failure | Redundant sensors + watchdog |
| Decision proof tampering | Cryptographic hash chain |
