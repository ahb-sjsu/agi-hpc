# Metacognition Subsystem Sprint Plan

## Overview

Metacognition is the "thinking about thinking" layer of AGI-HPC. It provides self-monitoring, plan review, confidence estimation, and anomaly detection to ensure the cognitive system operates reliably and can recognize its own limitations.

## Current State Assessment

### Implemented (Scaffolding Complete)
| File | Status | Description |
|------|--------|-------------|
| `meta/service.py` | **Stub** | Basic gRPC skeleton, placeholder evaluate() |
| `lh/metacog_client.py` | **Stub** | Returns ACCEPT always, gRPC wiring present |
| `meta.proto` | **TODO** | Just contains TODO marker |

### Current Functionality
- `MetacognitionEngine.evaluate()` - Placeholder with basic heuristics
- Returns: confidence, issues list, decision (ACCEPT/REVISE/REJECT)
- Publishes `meta.review` events to EventFabric

### Key Gaps
1. **No proto definitions** - meta.proto is empty
2. **No reasoning trace analysis** - Just checks for TODO string
3. **No confidence calibration** - Returns hardcoded 0.8
4. **No LLM-based reflection** - No actual reasoning about reasoning
5. **No learning from outcomes** - No feedback loop

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      METACOGNITION SUBSYSTEM                                 │
│                          Port: 50070                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    METACOGNITION ENGINE                              │   │
│   │                                                                      │   │
│   │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│   │   │   Reasoning   │  │  Consistency  │  │   Anomaly     │           │   │
│   │   │   Analyzer    │  │   Checker     │  │   Detector    │           │   │
│   │   │               │  │               │  │               │           │   │
│   │   │ - Trace eval  │  │ - Memory xref │  │ - Drift       │           │   │
│   │   │ - Logic check │  │ - RH verify   │  │ - Outliers    │           │   │
│   │   │ - Coherence   │  │ - Safety sync │  │ - Uncertainty │           │   │
│   │   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │   │
│   │           │                  │                  │                    │   │
│   │   ┌───────┴──────────────────┴──────────────────┴───────┐           │   │
│   │   │              CONFIDENCE ESTIMATOR                    │           │   │
│   │   │                                                      │           │   │
│   │   │  Bayesian confidence calibration with historical     │           │   │
│   │   │  outcome feedback for well-calibrated predictions    │           │   │
│   │   └───────────────────────────┬──────────────────────────┘           │   │
│   │                               │                                      │   │
│   │   ┌───────────────────────────▼──────────────────────────┐           │   │
│   │   │              DECISION MAKER                           │           │   │
│   │   │                                                       │           │   │
│   │   │   ACCEPT: Confidence > 0.7, no critical issues       │           │   │
│   │   │   REVISE: Confidence 0.4-0.7, addressable issues     │           │   │
│   │   │   REJECT: Confidence < 0.4, critical issues          │           │   │
│   │   │   DEFER:  Requires human oversight                    │           │   │
│   │   └───────────────────────────────────────────────────────┘           │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                      LLM REFLECTION                                  │   │
│   │                                                                      │   │
│   │   Uses language model for:                                          │   │
│   │   - Reasoning chain verification                                    │   │
│   │   - Plan critique and improvement suggestions                       │   │
│   │   - Explanation generation for decisions                            │   │
│   │   - Creative problem-solving on REVISE                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sprint 1: Proto Definitions and Core Service

**Goal**: Define complete proto API and wire up gRPC service properly.

### Tasks

#### 1.1 Define meta.proto
- [ ] `MetaReviewRequest` - Plan, reasoning trace, simulation result
- [ ] `MetaReviewResponse` - Decision, confidence, issues, suggestions
- [ ] `ReviseRequest` - Plan + review feedback
- [ ] `RevisedPlan` - Modified plan with change explanations
- [ ] `IntrospectionQuery` - Query internal state
- [ ] `IntrospectionResponse` - Current confidence, recent decisions
- [ ] `CalibrationEvent` - Outcome feedback for calibration

```protobuf
// proto/meta.proto
syntax = "proto3";
package agi.meta.v1;

import "plan.proto";

// =============================================================================
// Metacognition Service
// =============================================================================

service MetacognitionService {
  // Review a plan before execution
  rpc ReviewPlan(MetaReviewRequest) returns (MetaReviewResponse);

  // Request plan revision based on review
  rpc RevisePlan(ReviseRequest) returns (RevisedPlanResponse);

  // Query current metacognitive state
  rpc Introspect(IntrospectionQuery) returns (IntrospectionResponse);

  // Report outcome for calibration
  rpc ReportOutcome(OutcomeReport) returns (OutcomeReportResponse);
}

// =============================================================================
// Review Messages
// =============================================================================

message MetaReviewRequest {
  // The plan to review
  agi.plan.v1.PlanGraphProto plan = 1;

  // Reasoning trace from planner
  ReasoningTrace reasoning_trace = 2;

  // Simulation results from RH (optional)
  agi.plan.v1.SimulationResult simulation = 3;

  // Memory context used (optional)
  MemoryContext memory_context = 4;

  // Request ID for correlation
  string request_id = 5;
}

message ReasoningTrace {
  // Ordered reasoning steps
  repeated ReasoningStep steps = 1;

  // Total reasoning time in ms
  int64 reasoning_time_ms = 2;

  // Model used for reasoning
  string model_id = 3;
}

message ReasoningStep {
  int32 index = 1;
  string step_type = 2;  // "decompose", "verify", "select", "conclude"
  string content = 3;
  float confidence = 4;
  repeated string evidence = 5;
}

message MemoryContext {
  int32 semantic_hits = 1;
  int32 episodic_hits = 2;
  int32 skill_hits = 3;
  float avg_similarity = 4;
}

message MetaReviewResponse {
  // Decision: ACCEPT, REVISE, REJECT, DEFER
  MetaDecision decision = 1;

  // Confidence in this review (0.0 - 1.0)
  float confidence = 2;

  // Issues found during review
  repeated MetaIssue issues = 3;

  // Suggestions for improvement (if REVISE)
  repeated string suggestions = 4;

  // Explanation of the decision
  string explanation = 5;

  // Review took this long (ms)
  int64 review_time_ms = 6;
}

enum MetaDecision {
  META_DECISION_UNSPECIFIED = 0;
  META_DECISION_ACCEPT = 1;
  META_DECISION_REVISE = 2;
  META_DECISION_REJECT = 3;
  META_DECISION_DEFER = 4;  // Requires human oversight
}

message MetaIssue {
  // Issue severity
  IssueSeverity severity = 1;

  // Issue category
  string category = 2;

  // Human-readable description
  string description = 3;

  // Which step(s) affected
  repeated string affected_steps = 4;

  // Suggested fix
  string suggested_fix = 5;
}

enum IssueSeverity {
  ISSUE_SEVERITY_UNSPECIFIED = 0;
  ISSUE_SEVERITY_INFO = 1;
  ISSUE_SEVERITY_WARNING = 2;
  ISSUE_SEVERITY_ERROR = 3;
  ISSUE_SEVERITY_CRITICAL = 4;
}

// =============================================================================
// Revision Messages
// =============================================================================

message ReviseRequest {
  // Original plan
  agi.plan.v1.PlanGraphProto original_plan = 1;

  // Review that triggered revision
  MetaReviewResponse review = 2;

  // Maximum revision attempts
  int32 max_attempts = 3;
}

message RevisedPlanResponse {
  // The revised plan
  agi.plan.v1.PlanGraphProto revised_plan = 1;

  // What was changed
  repeated PlanChange changes = 2;

  // Did revision succeed?
  bool success = 3;

  // If failed, why
  string failure_reason = 4;
}

message PlanChange {
  string change_type = 1;  // "added", "removed", "modified"
  string step_id = 2;
  string description = 3;
  string rationale = 4;
}

// =============================================================================
// Introspection Messages
// =============================================================================

message IntrospectionQuery {
  // What to introspect
  repeated string aspects = 1;  // "confidence", "recent_decisions", "anomalies"

  // Time window (seconds)
  int32 time_window_s = 2;
}

message IntrospectionResponse {
  // Current overall confidence
  float current_confidence = 1;

  // Recent decision summary
  DecisionSummary recent_decisions = 2;

  // Detected anomalies
  repeated Anomaly anomalies = 3;

  // Current calibration quality
  float calibration_score = 4;

  // System health indicators
  map<string, float> health_metrics = 5;
}

message DecisionSummary {
  int32 total_reviews = 1;
  int32 accepts = 2;
  int32 revises = 3;
  int32 rejects = 4;
  int32 defers = 5;
  float avg_confidence = 6;
}

message Anomaly {
  string anomaly_type = 1;  // "confidence_drift", "outcome_mismatch", "latency_spike"
  string description = 2;
  float severity = 3;
  int64 detected_at = 4;
}

// =============================================================================
// Calibration Messages
// =============================================================================

message OutcomeReport {
  // Which review this is for
  string request_id = 1;

  // What actually happened
  bool plan_succeeded = 2;
  float completion_percentage = 3;
  string outcome_description = 4;

  // What was predicted
  float predicted_confidence = 5;
  MetaDecision decision_made = 6;
}

message OutcomeReportResponse {
  // Updated calibration score
  float new_calibration_score = 1;

  // Feedback incorporated
  bool feedback_accepted = 2;
}
```

#### 1.2 Wire up gRPC service
- [ ] Generate proto stubs
- [ ] Implement `MetacognitionServiceServicer`
- [ ] Configure port 50070
- [ ] Add health checking
- [ ] EventFabric integration

#### 1.3 Configuration
- [ ] Create `configs/meta_config.yaml`
- [ ] Confidence thresholds (accept/revise/reject)
- [ ] LLM settings for reflection
- [ ] Calibration parameters

```yaml
# configs/meta_config.yaml
service:
  host: "0.0.0.0"
  port: 50070

thresholds:
  accept_confidence: 0.7
  revise_confidence: 0.4
  max_revision_attempts: 3

llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
  max_tokens: 2000
  temperature: 0.3

calibration:
  window_size: 100  # Recent outcomes to consider
  update_rate: 0.1  # Learning rate for calibration
  min_samples: 10   # Minimum before calibrating

anomaly_detection:
  confidence_drift_threshold: 0.15
  latency_spike_multiplier: 3.0
  outcome_mismatch_threshold: 0.3
```

### Acceptance Criteria
```bash
# Start Metacognition service
python -m agi.meta.service --port 50070

# Test review RPC
grpcurl -plaintext -d '{}' localhost:50070 agi.meta.v1.MetacognitionService/ReviewPlan
```

---

## Sprint 2: Reasoning Trace Analysis

**Goal**: Implement intelligent analysis of LH reasoning traces.

### Tasks

#### 2.1 Reasoning step classification
- [ ] Identify step types (decompose, verify, select, conclude)
- [ ] Detect logical fallacies
- [ ] Check chain-of-thought coherence
- [ ] Identify unsupported conclusions

```python
# src/agi/meta/reasoning_analyzer.py
from dataclasses import dataclass
from enum import Enum
from typing import List

class StepType(Enum):
    DECOMPOSE = "decompose"  # Breaking down problem
    VERIFY = "verify"        # Checking facts
    SELECT = "select"        # Choosing between options
    CONCLUDE = "conclude"    # Drawing conclusions
    UNKNOWN = "unknown"

@dataclass
class StepAnalysis:
    step_index: int
    step_type: StepType
    coherent_with_previous: bool
    evidence_supported: bool
    issues: List[str]
    confidence_modifier: float

class ReasoningAnalyzer:
    """Analyzes reasoning traces for quality and coherence."""

    def analyze_trace(self, trace: ReasoningTrace) -> TraceAnalysis:
        """Analyze complete reasoning trace."""
        step_analyses = []
        issues = []

        for i, step in enumerate(trace.steps):
            analysis = self._analyze_step(step, trace.steps[:i])
            step_analyses.append(analysis)

            if not analysis.coherent_with_previous:
                issues.append(f"Step {i}: Incoherent with previous reasoning")
            if not analysis.evidence_supported:
                issues.append(f"Step {i}: Conclusion not supported by evidence")

        overall_confidence = self._compute_trace_confidence(step_analyses)

        return TraceAnalysis(
            step_analyses=step_analyses,
            issues=issues,
            overall_confidence=overall_confidence,
        )

    def _analyze_step(
        self, step: ReasoningStep, previous: List[ReasoningStep]
    ) -> StepAnalysis:
        """Analyze individual reasoning step."""
        # Classify step type
        step_type = self._classify_step(step)

        # Check coherence with previous steps
        coherent = self._check_coherence(step, previous)

        # Check if evidence supports conclusion
        evidence_supported = self._check_evidence(step)

        return StepAnalysis(
            step_index=step.index,
            step_type=step_type,
            coherent_with_previous=coherent,
            evidence_supported=evidence_supported,
            issues=[],
            confidence_modifier=1.0 if coherent and evidence_supported else 0.8,
        )
```

#### 2.2 Logic verification
- [ ] Check for circular reasoning
- [ ] Detect contradictions
- [ ] Verify causal claims
- [ ] Validate quantitative assertions

#### 2.3 Evidence quality assessment
- [ ] Score evidence relevance
- [ ] Check source reliability
- [ ] Detect speculation vs fact
- [ ] Flag unverified assumptions

#### 2.4 LLM-assisted analysis (optional)
- [ ] Use LLM to critique reasoning
- [ ] Generate improvement suggestions
- [ ] Identify blind spots

---

## Sprint 3: Consistency Checking

**Goal**: Cross-verify plans against memory, simulation, and safety results.

### Tasks

#### 3.1 Memory consistency
- [ ] Verify facts cited in plan exist in semantic memory
- [ ] Check for contradictions with stored knowledge
- [ ] Validate skill references in procedural memory
- [ ] Cross-reference similar past episodes

```python
# src/agi/meta/consistency_checker.py
class ConsistencyChecker:
    """Checks plan consistency against memory and simulation."""

    def __init__(
        self,
        semantic_client: SemanticMemoryClient,
        episodic_client: EpisodicMemoryClient,
        procedural_client: ProceduralMemoryClient,
    ):
        self.semantic = semantic_client
        self.episodic = episodic_client
        self.procedural = procedural_client

    async def check_memory_consistency(
        self, plan: PlanGraph, memory_context: MemoryContext
    ) -> ConsistencyResult:
        """Verify plan is consistent with memory."""
        issues = []

        # Check each step's tool references
        for step in plan.steps:
            if step.tool_id:
                skill = await self.procedural.get_skill(step.tool_id)
                if not skill:
                    issues.append(MetaIssue(
                        severity=IssueSeverity.ERROR,
                        category="missing_skill",
                        description=f"Tool {step.tool_id} not found in procedural memory",
                        affected_steps=[step.step_id],
                    ))
                else:
                    # Check preconditions
                    if not self._preconditions_met(step, skill):
                        issues.append(MetaIssue(
                            severity=IssueSeverity.WARNING,
                            category="precondition_violation",
                            description=f"Preconditions for {step.tool_id} may not be met",
                            affected_steps=[step.step_id],
                        ))

        # Check for similar failed episodes
        similar_failures = await self.episodic.search(
            task_description=plan.goal,
            success_only=False,
        )
        for episode in similar_failures:
            if not episode.outcome.success:
                issues.append(MetaIssue(
                    severity=IssueSeverity.INFO,
                    category="similar_failure",
                    description=f"Similar task failed before: {episode.outcome.outcome_description}",
                    suggested_fix="Consider alternative approach",
                ))

        return ConsistencyResult(issues=issues)
```

#### 3.2 Simulation consistency
- [ ] Verify simulation was run for risky steps
- [ ] Check simulation risk against plan assumptions
- [ ] Flag steps that failed simulation
- [ ] Verify physical feasibility

#### 3.3 Safety consistency
- [ ] Verify safety checks were performed
- [ ] Check for bypassed safety steps
- [ ] Validate Bond Index is acceptable
- [ ] Ensure Decision Proofs exist

#### 3.4 Inter-step consistency
- [ ] Verify step ordering makes sense
- [ ] Check precondition/postcondition chains
- [ ] Detect missing intermediate steps
- [ ] Validate resource availability

---

## Sprint 4: Confidence Calibration

**Goal**: Implement well-calibrated confidence estimation with outcome feedback.

### Tasks

#### 4.1 Bayesian confidence model
- [ ] Prior confidence from heuristics
- [ ] Update based on check results
- [ ] Calibrate against historical outcomes
- [ ] Uncertainty quantification

```python
# src/agi/meta/confidence.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ConfidenceFactors:
    reasoning_quality: float      # 0-1
    memory_consistency: float     # 0-1
    simulation_success: float     # 0-1
    safety_approval: float        # 0-1
    novelty_penalty: float        # 0-1 (lower for novel situations)

class ConfidenceEstimator:
    """Bayesian confidence estimation with calibration."""

    def __init__(self, calibration_window: int = 100):
        self.calibration_window = calibration_window
        self.outcome_history: List[Tuple[float, bool]] = []

        # Calibration parameters (learned from outcomes)
        self.calibration_a = 1.0  # Scale
        self.calibration_b = 0.0  # Bias

    def estimate_confidence(self, factors: ConfidenceFactors) -> float:
        """Estimate calibrated confidence from factors."""
        # Weighted combination of factors
        raw_confidence = (
            0.3 * factors.reasoning_quality +
            0.2 * factors.memory_consistency +
            0.2 * factors.simulation_success +
            0.2 * factors.safety_approval +
            0.1 * factors.novelty_penalty
        )

        # Apply calibration
        calibrated = self._calibrate(raw_confidence)

        return np.clip(calibrated, 0.0, 1.0)

    def _calibrate(self, raw: float) -> float:
        """Apply learned calibration function."""
        # Platt scaling: sigmoid(a * raw + b)
        return 1 / (1 + np.exp(-(self.calibration_a * raw + self.calibration_b)))

    def update_from_outcome(self, predicted_confidence: float, success: bool):
        """Update calibration based on actual outcome."""
        self.outcome_history.append((predicted_confidence, success))

        # Keep window size
        if len(self.outcome_history) > self.calibration_window:
            self.outcome_history = self.outcome_history[-self.calibration_window:]

        # Refit calibration parameters
        if len(self.outcome_history) >= 10:
            self._fit_calibration()

    def _fit_calibration(self):
        """Fit calibration parameters using logistic regression."""
        confidences = np.array([c for c, _ in self.outcome_history])
        outcomes = np.array([1.0 if s else 0.0 for _, s in self.outcome_history])

        # Simple gradient descent for Platt scaling
        for _ in range(100):
            preds = 1 / (1 + np.exp(-(self.calibration_a * confidences + self.calibration_b)))
            error = preds - outcomes
            grad_a = np.mean(error * confidences)
            grad_b = np.mean(error)
            self.calibration_a -= 0.1 * grad_a
            self.calibration_b -= 0.1 * grad_b

    def calibration_score(self) -> float:
        """Compute Expected Calibration Error (ECE)."""
        if len(self.outcome_history) < 10:
            return 0.0

        # Bin predictions and compute calibration error
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        total = len(self.outcome_history)

        for i in range(len(bins) - 1):
            in_bin = [
                (c, s) for c, s in self.outcome_history
                if bins[i] <= c < bins[i + 1]
            ]
            if in_bin:
                avg_conf = np.mean([c for c, _ in in_bin])
                avg_acc = np.mean([1.0 if s else 0.0 for _, s in in_bin])
                ece += len(in_bin) / total * abs(avg_conf - avg_acc)

        return 1.0 - ece  # Return as score (higher is better)
```

#### 4.2 Outcome feedback loop
- [ ] Receive `OutcomeReport` after execution
- [ ] Update calibration parameters
- [ ] Track prediction accuracy
- [ ] Alert on calibration drift

#### 4.3 Uncertainty decomposition
- [ ] Aleatoric uncertainty (inherent randomness)
- [ ] Epistemic uncertainty (lack of knowledge)
- [ ] Model uncertainty (LLM reliability)
- [ ] Report uncertainty breakdown

---

## Sprint 5: Anomaly Detection

**Goal**: Detect unusual patterns that might indicate problems.

### Tasks

#### 5.1 Confidence drift detection
- [ ] Track rolling confidence average
- [ ] Alert on significant drops
- [ ] Detect sudden spikes (overconfidence)
- [ ] Seasonal/cyclical patterns

```python
# src/agi/meta/anomaly_detector.py
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Anomaly:
    anomaly_type: str
    description: str
    severity: float
    detected_at: int
    context: dict

class AnomalyDetector:
    """Detects anomalies in metacognitive behavior."""

    def __init__(
        self,
        confidence_drift_threshold: float = 0.15,
        latency_spike_multiplier: float = 3.0,
        outcome_mismatch_threshold: float = 0.3,
        window_size: int = 50,
    ):
        self.confidence_drift_threshold = confidence_drift_threshold
        self.latency_spike_multiplier = latency_spike_multiplier
        self.outcome_mismatch_threshold = outcome_mismatch_threshold

        self.confidence_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.decision_history = deque(maxlen=window_size)

    def check_confidence_drift(self, current_confidence: float) -> Optional[Anomaly]:
        """Check for unusual confidence drift."""
        self.confidence_history.append(current_confidence)

        if len(self.confidence_history) < 10:
            return None

        recent = list(self.confidence_history)[-10:]
        older = list(self.confidence_history)[-20:-10] if len(self.confidence_history) >= 20 else recent

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        drift = abs(recent_mean - older_mean)

        if drift > self.confidence_drift_threshold:
            direction = "dropped" if recent_mean < older_mean else "spiked"
            return Anomaly(
                anomaly_type="confidence_drift",
                description=f"Confidence {direction} by {drift:.2f} (from {older_mean:.2f} to {recent_mean:.2f})",
                severity=min(1.0, drift / self.confidence_drift_threshold),
                detected_at=int(time.time() * 1000),
                context={"recent_mean": recent_mean, "older_mean": older_mean},
            )

        return None

    def check_latency_spike(self, review_time_ms: int) -> Optional[Anomaly]:
        """Check for unusual latency spikes."""
        self.latency_history.append(review_time_ms)

        if len(self.latency_history) < 10:
            return None

        history = list(self.latency_history)[:-1]
        mean_latency = np.mean(history)
        std_latency = np.std(history) or 1.0

        z_score = (review_time_ms - mean_latency) / std_latency

        if z_score > self.latency_spike_multiplier:
            return Anomaly(
                anomaly_type="latency_spike",
                description=f"Review took {review_time_ms}ms (expected ~{mean_latency:.0f}ms)",
                severity=min(1.0, z_score / 5.0),
                detected_at=int(time.time() * 1000),
                context={"review_time_ms": review_time_ms, "expected_ms": mean_latency},
            )

        return None

    def check_decision_pattern(self, decision: MetaDecision) -> Optional[Anomaly]:
        """Check for unusual decision patterns."""
        self.decision_history.append(decision)

        if len(self.decision_history) < 20:
            return None

        recent = list(self.decision_history)[-10:]
        reject_rate = sum(1 for d in recent if d == MetaDecision.REJECT) / len(recent)

        if reject_rate > 0.5:
            return Anomaly(
                anomaly_type="high_reject_rate",
                description=f"Rejection rate is {reject_rate:.0%} in last 10 reviews",
                severity=reject_rate,
                detected_at=int(time.time() * 1000),
                context={"reject_rate": reject_rate},
            )

        return None
```

#### 5.2 Outcome mismatch detection
- [ ] Compare predictions to outcomes
- [ ] Detect systematic over/under-confidence
- [ ] Alert on repeated failures
- [ ] Track failure patterns

#### 5.3 Behavioral anomalies
- [ ] Unusual decision patterns (too many rejects)
- [ ] Reasoning pattern changes
- [ ] Memory miss rate spikes
- [ ] Safety trigger frequency

---

## Sprint 6: LLM-Based Reflection

**Goal**: Use language model for deep plan critique and improvement.

### Tasks

#### 6.1 Plan critique prompts
- [ ] Design critique prompt template
- [ ] Multi-turn dialogue for clarification
- [ ] Structured output parsing
- [ ] Issue prioritization

```python
# src/agi/meta/llm_reflection.py
class LLMReflector:
    """Uses LLM for plan critique and reflection."""

    CRITIQUE_PROMPT = """You are a metacognitive reviewer for an AI planning system.

Review the following plan and identify any issues:

GOAL: {goal}

PLAN:
{plan_steps}

REASONING TRACE:
{reasoning_trace}

SIMULATION RESULT:
- Risk Score: {risk_score}
- Violations: {violations}

Analyze the plan for:
1. Logical coherence - Does the reasoning make sense?
2. Completeness - Are any steps missing?
3. Safety - Are there any safety concerns?
4. Feasibility - Can this plan actually be executed?
5. Efficiency - Is there a simpler approach?

Provide your analysis in the following JSON format:
{{
    "issues": [
        {{
            "severity": "critical|warning|info",
            "category": "logic|completeness|safety|feasibility|efficiency",
            "description": "...",
            "affected_steps": ["step_id1", "step_id2"],
            "suggested_fix": "..."
        }}
    ],
    "overall_assessment": "...",
    "confidence": 0.0-1.0,
    "recommendation": "ACCEPT|REVISE|REJECT"
}}
"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def critique_plan(
        self,
        plan: PlanGraph,
        reasoning_trace: ReasoningTrace,
        simulation: SimulationResult,
    ) -> LLMCritique:
        """Get LLM critique of plan."""
        prompt = self.CRITIQUE_PROMPT.format(
            goal=plan.goal,
            plan_steps=self._format_steps(plan.steps),
            reasoning_trace=self._format_trace(reasoning_trace),
            risk_score=simulation.overall_risk,
            violations=simulation.violations,
        )

        response = await self.llm.complete(prompt, json_mode=True)
        return self._parse_critique(response)

    async def suggest_revision(
        self,
        plan: PlanGraph,
        issues: List[MetaIssue],
    ) -> RevisedPlan:
        """Get LLM suggestion for plan revision."""
        prompt = self.REVISION_PROMPT.format(
            plan=self._format_plan(plan),
            issues=self._format_issues(issues),
        )

        response = await self.llm.complete(prompt, json_mode=True)
        return self._parse_revision(response)
```

#### 6.2 Explanation generation
- [ ] Generate human-readable explanations
- [ ] Explain decision rationale
- [ ] Summarize issues found
- [ ] Provide actionable suggestions

#### 6.3 Creative problem-solving
- [ ] Suggest alternative approaches
- [ ] Generate revised plans
- [ ] Identify root causes
- [ ] Learn from past solutions

---

## Sprint 7: Unit Tests

**Goal**: Achieve 80%+ test coverage for metacognition module.

### Tasks

#### 7.1 Core engine tests
- [ ] `test_metacognition_engine_accept`
- [ ] `test_metacognition_engine_revise`
- [ ] `test_metacognition_engine_reject`
- [ ] `test_metacognition_engine_defer`

#### 7.2 Reasoning analyzer tests
- [ ] `test_reasoning_step_classification`
- [ ] `test_coherence_detection`
- [ ] `test_evidence_validation`
- [ ] `test_fallacy_detection`

#### 7.3 Consistency checker tests
- [ ] `test_memory_consistency`
- [ ] `test_simulation_consistency`
- [ ] `test_safety_consistency`
- [ ] `test_step_ordering`

#### 7.4 Confidence estimator tests
- [ ] `test_confidence_estimation`
- [ ] `test_calibration_update`
- [ ] `test_calibration_score`
- [ ] `test_uncertainty_decomposition`

#### 7.5 Anomaly detector tests
- [ ] `test_confidence_drift_detection`
- [ ] `test_latency_spike_detection`
- [ ] `test_decision_pattern_anomaly`
- [ ] `test_outcome_mismatch`

### Test Infrastructure
```python
# tests/meta/conftest.py
import pytest
from agi.meta.engine import MetacognitionEngine
from agi.meta.confidence import ConfidenceEstimator
from agi.meta.anomaly_detector import AnomalyDetector

@pytest.fixture
def meta_engine():
    return MetacognitionEngine()

@pytest.fixture
def confidence_estimator():
    return ConfidenceEstimator(calibration_window=50)

@pytest.fixture
def anomaly_detector():
    return AnomalyDetector()

@pytest.fixture
def sample_plan():
    """Create sample plan for testing."""
    return create_test_plan(num_steps=5)

@pytest.fixture
def sample_reasoning_trace():
    """Create sample reasoning trace."""
    return create_test_trace(num_steps=10)
```

---

## Sprint 8: Integration and Production

**Goal**: Integrate with LH and prepare for production.

### Tasks

#### 8.1 LH integration
- [ ] Update `metacog_client.py` with full API
- [ ] Add revision loop in `plan_service.py`
- [ ] Configure retry logic
- [ ] Handle DEFER decisions

#### 8.2 EventFabric events
- [ ] Publish `meta.review.started`
- [ ] Publish `meta.review.completed`
- [ ] Publish `meta.anomaly.detected`
- [ ] Publish `meta.calibration.updated`

#### 8.3 Observability
- [ ] Prometheus metrics
- [ ] Decision latency histogram
- [ ] Confidence distribution
- [ ] Anomaly counts

#### 8.4 Persistence
- [ ] Store review history in PostgreSQL
- [ ] Calibration state persistence
- [ ] Anomaly log
- [ ] Decision audit trail

---

## File Structure After Completion

```
src/agi/meta/
├── __init__.py
├── service.py              # gRPC service entrypoint
├── engine.py               # MetacognitionEngine
├── reasoning_analyzer.py   # Reasoning trace analysis
├── consistency_checker.py  # Memory/simulation consistency
├── confidence.py           # Confidence estimation
├── anomaly_detector.py     # Anomaly detection
├── llm_reflection.py       # LLM-based reflection
├── decision_maker.py       # Final decision logic
├── persistence/
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy models
│   └── store.py            # Review history storage
└── config.py               # Configuration

tests/meta/
├── __init__.py
├── conftest.py             # Fixtures
├── test_engine.py
├── test_reasoning_analyzer.py
├── test_consistency_checker.py
├── test_confidence.py
├── test_anomaly_detector.py
└── test_llm_reflection.py

configs/
└── meta_config.yaml        # Metacognition configuration
```

---

## Priority Order

1. **Sprint 1** - Critical: Proto definitions and service bootstrap
2. **Sprint 4** - High: Confidence calibration (core value)
3. **Sprint 2** - High: Reasoning analysis
4. **Sprint 3** - High: Consistency checking
5. **Sprint 5** - Medium: Anomaly detection
6. **Sprint 6** - Medium: LLM reflection
7. **Sprint 7** - High: Testing
8. **Sprint 8** - Medium: Integration

---

## Quick Start (After Sprint 1-4)

```bash
# Terminal 1: Start Metacognition service
python -m agi.meta.service --port 50070

# Terminal 2: Test review
python -c "
import grpc
from agi.proto_gen import meta_pb2, meta_pb2_grpc, plan_pb2

channel = grpc.insecure_channel('localhost:50070')
stub = meta_pb2_grpc.MetacognitionServiceStub(channel)

# Create test request
request = meta_pb2.MetaReviewRequest(
    plan=plan_pb2.PlanGraphProto(
        plan_id='test-001',
        goal='Navigate to red cube',
    ),
    reasoning_trace=meta_pb2.ReasoningTrace(
        steps=[
            meta_pb2.ReasoningStep(
                index=0,
                step_type='decompose',
                content='First identify red cube location',
                confidence=0.9,
            ),
        ],
    ),
)

response = stub.ReviewPlan(request)
print(f'Decision: {meta_pb2.MetaDecision.Name(response.decision)}')
print(f'Confidence: {response.confidence:.2f}')
print(f'Issues: {len(response.issues)}')
for issue in response.issues:
    print(f'  - [{issue.category}] {issue.description}')
"
```

---

## Dependencies

```toml
# pyproject.toml additions for metacognition
[project.optional-dependencies]
meta = [
    # Core
    "numpy>=1.24",

    # LLM
    "anthropic>=0.18",
    "openai>=1.0",

    # Database
    "asyncpg>=0.29.0",
    "sqlalchemy[asyncio]>=2.0",

    # Metrics
    "prometheus-client>=0.19",
]
```
