# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Reasoning Trace Analyzer for Metacognition.

Implements Sprint 2 requirements:
- Reasoning step classification
- Logic verification
- Evidence quality assessment
- Coherence checking

This module analyzes reasoning traces from the LH planner to:
- Classify step types (decompose, verify, select, conclude)
- Detect logical fallacies
- Check chain-of-thought coherence
- Identify unsupported conclusions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Types
# ---------------------------------------------------------------------------


class StepType(Enum):
    """Types of reasoning steps."""

    DECOMPOSE = "decompose"  # Breaking down problem
    VERIFY = "verify"  # Checking facts
    SELECT = "select"  # Choosing between options
    CONCLUDE = "conclude"  # Drawing conclusions
    HYPOTHESIZE = "hypothesize"  # Forming hypotheses
    EVALUATE = "evaluate"  # Evaluating options
    RECALL = "recall"  # Retrieving from memory
    CALCULATE = "calculate"  # Numerical computation
    UNKNOWN = "unknown"


class IssueType(Enum):
    """Types of reasoning issues."""

    CIRCULAR_REASONING = "circular_reasoning"
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"
    CONTRADICTION = "contradiction"
    MISSING_EVIDENCE = "missing_evidence"
    LOGICAL_GAP = "logical_gap"
    OVERGENERALIZATION = "overgeneralization"
    FALSE_DICHOTOMY = "false_dichotomy"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    CONFIRMATION_BIAS = "confirmation_bias"
    HASTY_CONCLUSION = "hasty_conclusion"
    INCOHERENT = "incoherent"


class IssueSeverity(Enum):
    """Severity levels for issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ReasoningStep:
    """A single step in a reasoning trace."""

    index: int
    step_type: str
    content: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ReasoningTrace:
    """Complete reasoning trace from planner."""

    steps: List[ReasoningStep]
    reasoning_time_ms: int = 0
    model_id: str = ""
    goal: str = ""


@dataclass
class StepAnalysis:
    """Analysis result for a single reasoning step."""

    step_index: int
    step_type: StepType
    coherent_with_previous: bool
    evidence_supported: bool
    issues: List["ReasoningIssue"]
    confidence_modifier: float = 1.0


@dataclass
class ReasoningIssue:
    """An issue found in reasoning."""

    issue_type: IssueType
    severity: IssueSeverity
    description: str
    affected_steps: List[int]
    suggested_fix: str = ""


@dataclass
class TraceAnalysis:
    """Complete analysis of a reasoning trace."""

    step_analyses: List[StepAnalysis]
    issues: List[ReasoningIssue]
    overall_confidence: float
    reasoning_quality: float
    coherence_score: float
    evidence_score: float


# ---------------------------------------------------------------------------
# Step Type Classifier
# ---------------------------------------------------------------------------


class StepTypeClassifier:
    """Classifies reasoning steps by their type."""

    # Keywords associated with each step type
    STEP_KEYWORDS = {
        StepType.DECOMPOSE: [
            "break down",
            "divide",
            "split",
            "decompose",
            "parts",
            "components",
            "sub-tasks",
            "steps are",
            "first",
            "then",
        ],
        StepType.VERIFY: [
            "verify",
            "check",
            "confirm",
            "validate",
            "ensure",
            "is correct",
            "is valid",
            "matches",
            "consistent",
        ],
        StepType.SELECT: [
            "choose",
            "select",
            "pick",
            "prefer",
            "best option",
            "between",
            "option a",
            "option b",
            "alternatives",
        ],
        StepType.CONCLUDE: [
            "therefore",
            "thus",
            "hence",
            "conclude",
            "finally",
            "in conclusion",
            "as a result",
            "so",
            "this means",
        ],
        StepType.HYPOTHESIZE: [
            "assume",
            "suppose",
            "if",
            "hypothesis",
            "might",
            "could be",
            "possibly",
            "perhaps",
            "likely",
        ],
        StepType.EVALUATE: [
            "evaluate",
            "assess",
            "analyze",
            "compare",
            "weigh",
            "pros and cons",
            "trade-off",
            "consider",
        ],
        StepType.RECALL: [
            "remember",
            "recall",
            "from memory",
            "previously",
            "known",
            "according to",
            "as per",
        ],
        StepType.CALCULATE: [
            "calculate",
            "compute",
            "sum",
            "multiply",
            "divide",
            "equals",
            "result is",
            "formula",
            "equation",
        ],
    }

    def classify(self, step: ReasoningStep) -> StepType:
        """Classify a reasoning step by its content."""
        content_lower = step.content.lower()

        # Check explicit step_type if provided
        if step.step_type:
            try:
                return StepType(step.step_type.lower())
            except ValueError:
                pass

        # Classify by keywords
        scores = {step_type: 0 for step_type in StepType}

        for step_type, keywords in self.STEP_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    scores[step_type] += 1

        # Return type with highest score
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type

        return StepType.UNKNOWN


# ---------------------------------------------------------------------------
# Coherence Checker
# ---------------------------------------------------------------------------


class CoherenceChecker:
    """Checks coherence between reasoning steps."""

    def check_coherence(
        self,
        step: ReasoningStep,
        previous_steps: List[ReasoningStep],
    ) -> Tuple[bool, List[ReasoningIssue]]:
        """Check if step is coherent with previous steps."""
        issues = []

        if not previous_steps:
            return True, []

        # Check for logical connection
        has_connection = self._has_logical_connection(step, previous_steps)
        if not has_connection:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.LOGICAL_GAP,
                    severity=IssueSeverity.WARNING,
                    description=f"Step {step.index} lacks clear connection to previous reasoning",
                    affected_steps=[step.index],
                    suggested_fix="Add explicit transition or reference to previous steps",
                )
            )

        # Check for contradictions
        contradiction = self._find_contradiction(step, previous_steps)
        if contradiction:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.CONTRADICTION,
                    severity=IssueSeverity.ERROR,
                    description=f"Step {step.index} contradicts step {contradiction}",
                    affected_steps=[step.index, contradiction],
                    suggested_fix="Resolve contradiction between steps",
                )
            )

        # Check for circular reasoning
        if self._is_circular(step, previous_steps):
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.CIRCULAR_REASONING,
                    severity=IssueSeverity.ERROR,
                    description=f"Step {step.index} appears to use circular reasoning",
                    affected_steps=[step.index],
                    suggested_fix="Break the circular dependency with independent evidence",
                )
            )

        is_coherent = (
            len(
                [
                    i
                    for i in issues
                    if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]
                ]
            )
            == 0
        )
        return is_coherent, issues

    def _has_logical_connection(
        self,
        step: ReasoningStep,
        previous: List[ReasoningStep],
    ) -> bool:
        """Check if step references or builds on previous steps."""
        content_lower = step.content.lower()

        # Connection words
        connection_words = [
            "therefore",
            "thus",
            "hence",
            "because",
            "since",
            "as a result",
            "consequently",
            "from this",
            "based on",
            "building on",
            "given that",
            "following",
            "next",
        ]

        for word in connection_words:
            if word in content_lower:
                return True

        # Check for reference to previous step concepts
        for prev_step in previous[-3:]:  # Check last 3 steps
            prev_words = set(prev_step.content.lower().split())
            curr_words = set(content_lower.split())

            # Look for significant word overlap
            common = prev_words & curr_words
            # Filter out common words
            common = {
                w
                for w in common
                if len(w) > 4 and w not in ["that", "this", "with", "from", "have"]
            }

            if len(common) >= 2:
                return True

        return False

    def _find_contradiction(
        self,
        step: ReasoningStep,
        previous: List[ReasoningStep],
    ) -> Optional[int]:
        """Find if step contradicts any previous step."""
        # Negation patterns
        negation_pairs = [
            ("is", "is not"),
            ("can", "cannot"),
            ("should", "should not"),
            ("will", "will not"),
            ("true", "false"),
            ("yes", "no"),
            ("possible", "impossible"),
            ("valid", "invalid"),
        ]

        content_lower = step.content.lower()

        for prev_step in previous:
            prev_lower = prev_step.content.lower()

            # Check for negation of same statement
            for pos, neg in negation_pairs:
                if pos in content_lower and neg in prev_lower:
                    # Check if referring to same subject
                    if self._same_subject(content_lower, prev_lower):
                        return prev_step.index
                if neg in content_lower and pos in prev_lower:
                    if self._same_subject(content_lower, prev_lower):
                        return prev_step.index

        return None

    def _same_subject(self, text1: str, text2: str) -> bool:
        """Check if two texts refer to the same subject."""
        # Simple noun overlap check
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Filter to likely nouns (longer words)
        nouns1 = {w for w in words1 if len(w) > 4 and w.isalpha()}
        nouns2 = {w for w in words2 if len(w) > 4 and w.isalpha()}

        return len(nouns1 & nouns2) >= 2

    def _is_circular(
        self,
        step: ReasoningStep,
        previous: List[ReasoningStep],
    ) -> bool:
        """Check for circular reasoning patterns."""
        content_lower = step.content.lower()

        # Look for self-referential patterns
        for prev_step in previous[:3]:  # Check early steps
            prev_lower = prev_step.content.lower()

            # If current step is very similar to early step
            words1 = set(content_lower.split())
            words2 = set(prev_lower.split())
            common = words1 & words2
            similarity = len(common) / max(len(words1), len(words2), 1)

            if similarity > 0.8 and step.index - prev_step.index > 3:
                return True

        return False


# ---------------------------------------------------------------------------
# Evidence Validator
# ---------------------------------------------------------------------------


class EvidenceValidator:
    """Validates evidence quality in reasoning."""

    def validate(self, step: ReasoningStep) -> Tuple[bool, List[ReasoningIssue]]:
        """Validate evidence for a reasoning step."""
        issues = []

        # Check if conclusion step has evidence
        if self._is_conclusion(step) and not step.evidence:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.UNSUPPORTED_CONCLUSION,
                    severity=IssueSeverity.WARNING,
                    description=f"Step {step.index} draws conclusion without explicit evidence",
                    affected_steps=[step.index],
                    suggested_fix="Add supporting evidence or references",
                )
            )

        # Check for speculation markers
        if self._has_speculation(step) and step.confidence > 0.8:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.OVERGENERALIZATION,
                    severity=IssueSeverity.INFO,
                    description=f"Step {step.index} uses speculative language but has high confidence",
                    affected_steps=[step.index],
                    suggested_fix="Adjust confidence or provide stronger evidence",
                )
            )

        # Check for hasty conclusions
        if self._is_hasty_conclusion(step):
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.HASTY_CONCLUSION,
                    severity=IssueSeverity.WARNING,
                    description=f"Step {step.index} may be drawing hasty conclusion",
                    affected_steps=[step.index],
                    suggested_fix="Add intermediate reasoning steps",
                )
            )

        is_valid = (
            len(
                [
                    i
                    for i in issues
                    if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]
                ]
            )
            == 0
        )
        return is_valid, issues

    def _is_conclusion(self, step: ReasoningStep) -> bool:
        """Check if step is a conclusion."""
        content_lower = step.content.lower()
        conclusion_markers = [
            "therefore",
            "thus",
            "hence",
            "conclude",
            "in conclusion",
            "finally",
            "as a result",
        ]
        return any(marker in content_lower for marker in conclusion_markers)

    def _has_speculation(self, step: ReasoningStep) -> bool:
        """Check if step contains speculative language."""
        content_lower = step.content.lower()
        speculation_markers = [
            "might",
            "maybe",
            "possibly",
            "could be",
            "perhaps",
            "probably",
            "likely",
            "unlikely",
            "seems",
            "appears",
            "guess",
            "assume",
        ]
        return any(marker in content_lower for marker in speculation_markers)

    def _is_hasty_conclusion(self, step: ReasoningStep) -> bool:
        """Check for hasty conclusion patterns."""
        content_lower = step.content.lower()

        # Short conclusions with strong claims
        if len(step.content) < 50:
            strong_claims = ["always", "never", "all", "none", "everyone", "nobody"]
            if any(claim in content_lower for claim in strong_claims):
                return True

        return False


# ---------------------------------------------------------------------------
# Main Reasoning Analyzer
# ---------------------------------------------------------------------------


class ReasoningAnalyzer:
    """Main class for analyzing reasoning traces."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        use_llm: bool = False,
    ) -> None:
        """Initialize reasoning analyzer.

        Args:
            min_confidence: Minimum acceptable confidence score
            use_llm: Whether to use LLM for deeper analysis
        """
        self.min_confidence = min_confidence
        self.use_llm = use_llm

        self._classifier = StepTypeClassifier()
        self._coherence_checker = CoherenceChecker()
        self._evidence_validator = EvidenceValidator()

        logger.info("[meta][analyzer] initialized min_confidence=%.2f", min_confidence)

    def analyze_trace(self, trace: ReasoningTrace) -> TraceAnalysis:
        """Analyze complete reasoning trace.

        Args:
            trace: Reasoning trace to analyze

        Returns:
            Complete analysis with issues and scores
        """
        step_analyses = []
        all_issues = []

        for i, step in enumerate(trace.steps):
            analysis = self._analyze_step(step, trace.steps[:i])
            step_analyses.append(analysis)
            all_issues.extend(analysis.issues)

        # Compute aggregate scores
        coherence_score = self._compute_coherence_score(step_analyses)
        evidence_score = self._compute_evidence_score(step_analyses)
        overall_confidence = self._compute_trace_confidence(
            step_analyses, coherence_score, evidence_score
        )
        reasoning_quality = self._compute_quality_score(step_analyses, all_issues)

        # Add global issues
        global_issues = self._check_global_issues(trace, step_analyses)
        all_issues.extend(global_issues)

        logger.debug(
            "[meta][analyzer] analyzed trace: %d steps, %d issues, confidence=%.2f",
            len(trace.steps),
            len(all_issues),
            overall_confidence,
        )

        return TraceAnalysis(
            step_analyses=step_analyses,
            issues=all_issues,
            overall_confidence=overall_confidence,
            reasoning_quality=reasoning_quality,
            coherence_score=coherence_score,
            evidence_score=evidence_score,
        )

    def _analyze_step(
        self,
        step: ReasoningStep,
        previous: List[ReasoningStep],
    ) -> StepAnalysis:
        """Analyze individual reasoning step."""
        # Classify step type
        step_type = self._classifier.classify(step)

        # Check coherence
        coherent, coherence_issues = self._coherence_checker.check_coherence(
            step, previous
        )

        # Validate evidence
        evidence_valid, evidence_issues = self._evidence_validator.validate(step)

        # Combine issues
        issues = coherence_issues + evidence_issues

        # Compute confidence modifier
        confidence_modifier = 1.0
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                confidence_modifier *= 0.5
            elif issue.severity == IssueSeverity.ERROR:
                confidence_modifier *= 0.7
            elif issue.severity == IssueSeverity.WARNING:
                confidence_modifier *= 0.9

        return StepAnalysis(
            step_index=step.index,
            step_type=step_type,
            coherent_with_previous=coherent,
            evidence_supported=evidence_valid,
            issues=issues,
            confidence_modifier=confidence_modifier,
        )

    def _check_global_issues(
        self,
        trace: ReasoningTrace,
        step_analyses: List[StepAnalysis],
    ) -> List[ReasoningIssue]:
        """Check for global trace-level issues."""
        issues = []

        # Check for missing conclusion
        has_conclusion = any(a.step_type == StepType.CONCLUDE for a in step_analyses)
        if not has_conclusion and len(trace.steps) > 2:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.LOGICAL_GAP,
                    severity=IssueSeverity.WARNING,
                    description="Reasoning trace lacks explicit conclusion",
                    affected_steps=[len(trace.steps) - 1],
                    suggested_fix="Add concluding step that summarizes the reasoning",
                )
            )

        # Check for too short trace
        if len(trace.steps) < 2 and trace.goal:
            issues.append(
                ReasoningIssue(
                    issue_type=IssueType.HASTY_CONCLUSION,
                    severity=IssueSeverity.WARNING,
                    description="Reasoning trace is too short for the goal complexity",
                    affected_steps=[0],
                    suggested_fix="Expand reasoning with intermediate steps",
                )
            )

        # Check for monotonic confidence drop
        if len(step_analyses) >= 3:
            confidences = [a.confidence_modifier for a in step_analyses]
            if all(
                c1 > c2 for c1, c2 in zip(confidences, confidences[1:], strict=False)
            ):
                issues.append(
                    ReasoningIssue(
                        issue_type=IssueType.INCOHERENT,
                        severity=IssueSeverity.INFO,
                        description="Confidence drops monotonically throughout reasoning",
                        affected_steps=list(range(len(step_analyses))),
                        suggested_fix="Review reasoning chain for accumulating uncertainties",
                    )
                )

        return issues

    def _compute_coherence_score(self, step_analyses: List[StepAnalysis]) -> float:
        """Compute overall coherence score."""
        if not step_analyses:
            return 1.0

        coherent_count = sum(1 for a in step_analyses if a.coherent_with_previous)
        return coherent_count / len(step_analyses)

    def _compute_evidence_score(self, step_analyses: List[StepAnalysis]) -> float:
        """Compute overall evidence support score."""
        if not step_analyses:
            return 1.0

        supported_count = sum(1 for a in step_analyses if a.evidence_supported)
        return supported_count / len(step_analyses)

    def _compute_trace_confidence(
        self,
        step_analyses: List[StepAnalysis],
        coherence: float,
        evidence: float,
    ) -> float:
        """Compute overall trace confidence."""
        if not step_analyses:
            return 0.5

        # Product of step confidence modifiers
        step_confidence = 1.0
        for analysis in step_analyses:
            step_confidence *= analysis.confidence_modifier

        # Combine with coherence and evidence scores
        overall = 0.4 * step_confidence + 0.3 * coherence + 0.3 * evidence

        return max(0.0, min(1.0, overall))

    def _compute_quality_score(
        self,
        step_analyses: List[StepAnalysis],
        issues: List[ReasoningIssue],
    ) -> float:
        """Compute reasoning quality score."""
        if not step_analyses:
            return 0.5

        # Base score from step analysis
        base_score = 1.0

        # Penalize based on issue severity
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 0.3
            elif issue.severity == IssueSeverity.ERROR:
                base_score -= 0.15
            elif issue.severity == IssueSeverity.WARNING:
                base_score -= 0.05

        # Bonus for good step type variety
        step_types = {a.step_type for a in step_analyses}
        if len(step_types) >= 3:
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    async def analyze_trace_with_llm(
        self,
        trace: ReasoningTrace,
    ) -> TraceAnalysis:
        """Analyze trace with LLM assistance.

        Args:
            trace: Reasoning trace to analyze

        Returns:
            Enhanced analysis with LLM insights
        """
        # Start with regular analysis
        analysis = self.analyze_trace(trace)

        if not self.use_llm:
            return analysis

        # TODO: Add LLM-based critique
        # This would use the LLM infrastructure to:
        # 1. Review the reasoning chain
        # 2. Identify logical fallacies
        # 3. Suggest improvements
        # 4. Provide natural language explanation

        return analysis
