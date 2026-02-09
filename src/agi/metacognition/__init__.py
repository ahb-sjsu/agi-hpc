# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Metacognition Module for AGI-HPC.

The Metacognition module provides "thinking about thinking" capabilities:
- Reasoning trace analysis
- Consistency checking
- Confidence estimation
- Anomaly detection
- LLM-based reflection

Usage:
    from agi.metacognition import ReasoningAnalyzer, ReasoningTrace

    analyzer = ReasoningAnalyzer()
    analysis = analyzer.analyze_trace(trace)
    print(f"Confidence: {analysis.overall_confidence}")
    print(f"Issues: {len(analysis.issues)}")
"""

from agi.metacognition.reasoning_analyzer import (
    ReasoningAnalyzer,
    ReasoningTrace,
    ReasoningStep,
    TraceAnalysis,
    StepAnalysis,
    ReasoningIssue,
    StepType,
    IssueType,
    IssueSeverity,
    StepTypeClassifier,
    CoherenceChecker,
    EvidenceValidator,
)
from agi.metacognition.consistency_checker import (
    ConsistencyChecker,
    ConsistencyCheckResult,
    ConsistencyLevel,
    ComponentType,
    ComponentState,
    Inconsistency,
    InconsistencyType,
)

__all__ = [
    # Main analyzer
    "ReasoningAnalyzer",
    # Consistency checker
    "ConsistencyChecker",
    "ConsistencyCheckResult",
    "ConsistencyLevel",
    "ComponentType",
    "ComponentState",
    "Inconsistency",
    "InconsistencyType",
    # Data types
    "ReasoningTrace",
    "ReasoningStep",
    "TraceAnalysis",
    "StepAnalysis",
    "ReasoningIssue",
    # Enums
    "StepType",
    "IssueType",
    "IssueSeverity",
    # Components
    "StepTypeClassifier",
    "CoherenceChecker",
    "EvidenceValidator",
]
