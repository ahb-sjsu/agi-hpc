# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Standard Event Schemas for AGI-HPC.

Provides unified event types for all subsystem communication via EventFabric:
- Perception events (RH → LH)
- Reasoning events (LH internal)
- Planning events (LH → Safety)
- Memory events (Memory subsystem)
- Safety events (Safety → All)
- Action events (All → Environment)
- Meta events (Metacognition)
- System events (Infrastructure)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventCategory(Enum):
    """Event categories."""

    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    MEMORY = "memory"
    SAFETY = "safety"
    ACTION = "action"
    META = "meta"
    SYSTEM = "system"


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Base Event Classes
# ---------------------------------------------------------------------------


@dataclass
class EventHeader:
    """Standard event header with distributed tracing support."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    category: EventCategory = EventCategory.SYSTEM
    priority: EventPriority = EventPriority.NORMAL
    source_service: str = ""
    source_instance: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    version: str = "1.0"


@dataclass
class AGIEvent:
    """Base AGI event with header and payload."""

    header: EventHeader
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "header": {
                "event_id": self.header.event_id,
                "event_type": self.header.event_type,
                "category": self.header.category.value,
                "priority": self.header.priority.value,
                "source_service": self.header.source_service,
                "source_instance": self.header.source_instance,
                "timestamp": self.header.timestamp.isoformat(),
                "correlation_id": self.header.correlation_id,
                "causation_id": self.header.causation_id,
                "trace_id": self.header.trace_id,
                "span_id": self.header.span_id,
                "version": self.header.version,
            },
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AGIEvent":
        """Create from dictionary."""
        header_data = data["header"]
        header = EventHeader(
            event_id=header_data["event_id"],
            event_type=header_data["event_type"],
            category=EventCategory(header_data["category"]),
            priority=EventPriority(header_data["priority"]),
            source_service=header_data["source_service"],
            source_instance=header_data["source_instance"],
            timestamp=datetime.fromisoformat(header_data["timestamp"]),
            correlation_id=header_data.get("correlation_id"),
            causation_id=header_data.get("causation_id"),
            trace_id=header_data.get("trace_id"),
            span_id=header_data.get("span_id"),
            version=header_data.get("version", "1.0"),
        )
        return cls(
            header=header,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Perception Events
# ---------------------------------------------------------------------------


class PerceptionEventType(Enum):
    """Perception event types."""

    STATE_UPDATE = "perception.state_update"
    OBJECT_DETECTED = "perception.object_detected"
    OBJECT_LOST = "perception.object_lost"
    SCENE_CHANGE = "perception.scene_change"
    ANOMALY = "perception.anomaly"


@dataclass
class PerceptionEvent(AGIEvent):
    """Event from perception pipeline."""

    @classmethod
    def create(
        cls,
        source: str,
        percept_type: str,
        data: Dict,
        confidence: float = 1.0,
        source_instance: str = "",
        **kwargs,
    ) -> "PerceptionEvent":
        """Create perception event."""
        header = EventHeader(
            event_type=f"perception.{percept_type}",
            category=EventCategory.PERCEPTION,
            source_service=source,
            source_instance=source_instance,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "percept_type": percept_type,
                "data": data,
                "confidence": confidence,
            },
        )


# ---------------------------------------------------------------------------
# Reasoning Events
# ---------------------------------------------------------------------------


class ReasoningEventType(Enum):
    """Reasoning event types."""

    STEP = "reasoning.step"
    COMPLETE = "reasoning.complete"
    ERROR = "reasoning.error"
    TIMEOUT = "reasoning.timeout"


@dataclass
class ReasoningEvent(AGIEvent):
    """Event from reasoning process."""

    @classmethod
    def create(
        cls,
        source: str,
        step_type: str,
        content: str,
        confidence: float = 0.8,
        step_index: int = 0,
        **kwargs,
    ) -> "ReasoningEvent":
        """Create reasoning event."""
        header = EventHeader(
            event_type=ReasoningEventType.STEP.value,
            category=EventCategory.REASONING,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "step_type": step_type,
                "content": content,
                "confidence": confidence,
                "step_index": step_index,
            },
        )


# ---------------------------------------------------------------------------
# Planning Events
# ---------------------------------------------------------------------------


class PlanningEventType(Enum):
    """Planning event types."""

    PLAN_CREATED = "planning.plan_created"
    PLAN_UPDATED = "planning.plan_updated"
    PLAN_VALIDATED = "planning.plan_validated"
    PLAN_APPROVED = "planning.plan_approved"
    PLAN_REJECTED = "planning.plan_rejected"
    PLAN_EXECUTION_STARTED = "planning.execution_started"
    PLAN_EXECUTION_COMPLETED = "planning.execution_completed"
    PLAN_EXECUTION_FAILED = "planning.execution_failed"


@dataclass
class PlanningEvent(AGIEvent):
    """Event from planning system."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: PlanningEventType,
        plan_id: str,
        goal: str = "",
        steps: Optional[List[Dict]] = None,
        **kwargs,
    ) -> "PlanningEvent":
        """Create planning event."""
        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.PLANNING,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "plan_id": plan_id,
                "goal": goal,
                "steps": steps or [],
            },
        )


# ---------------------------------------------------------------------------
# Memory Events
# ---------------------------------------------------------------------------


class MemoryEventType(Enum):
    """Memory event types."""

    SEMANTIC_STORE = "memory.semantic.store"
    SEMANTIC_RETRIEVE = "memory.semantic.retrieve"
    EPISODIC_STORE = "memory.episodic.store"
    EPISODIC_RETRIEVE = "memory.episodic.retrieve"
    PROCEDURAL_STORE = "memory.procedural.store"
    PROCEDURAL_RETRIEVE = "memory.procedural.retrieve"
    CONSOLIDATION = "memory.consolidation"


@dataclass
class MemoryEvent(AGIEvent):
    """Event from memory subsystem."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: MemoryEventType,
        memory_type: str,
        key: str = "",
        query: str = "",
        results: Optional[List[Dict]] = None,
        **kwargs,
    ) -> "MemoryEvent":
        """Create memory event."""
        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.MEMORY,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "memory_type": memory_type,
                "key": key,
                "query": query,
                "results": results or [],
            },
        )


# ---------------------------------------------------------------------------
# Safety Events
# ---------------------------------------------------------------------------


class SafetyEventType(Enum):
    """Safety event types."""

    CHECK_REQUESTED = "safety.check_requested"
    CHECK_PASSED = "safety.check_passed"
    CHECK_FAILED = "safety.check_failed"
    VIOLATION = "safety.violation"
    EMERGENCY_STOP = "safety.emergency_stop"
    OVERRIDE_REQUESTED = "safety.override_requested"
    OVERRIDE_GRANTED = "safety.override_granted"
    OVERRIDE_DENIED = "safety.override_denied"


@dataclass
class SafetyEvent(AGIEvent):
    """Event from safety subsystem."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: SafetyEventType,
        action_id: str,
        risk_score: float = 0.0,
        violations: Optional[List[str]] = None,
        decision: str = "",
        **kwargs,
    ) -> "SafetyEvent":
        """Create safety event."""
        priority = EventPriority.NORMAL
        if event_type in [SafetyEventType.EMERGENCY_STOP, SafetyEventType.VIOLATION]:
            priority = EventPriority.CRITICAL

        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.SAFETY,
            source_service=source,
            priority=priority,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "action_id": action_id,
                "risk_score": risk_score,
                "violations": violations or [],
                "decision": decision,
            },
        )


# ---------------------------------------------------------------------------
# Action Events
# ---------------------------------------------------------------------------


class ActionEventType(Enum):
    """Action event types."""

    REQUESTED = "action.requested"
    STARTED = "action.started"
    PROGRESS = "action.progress"
    COMPLETED = "action.completed"
    FAILED = "action.failed"
    CANCELLED = "action.cancelled"


@dataclass
class ActionEvent(AGIEvent):
    """Event from action execution."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: ActionEventType,
        action_id: str,
        action_type: str = "",
        parameters: Optional[Dict] = None,
        result: Optional[Dict] = None,
        error: str = "",
        **kwargs,
    ) -> "ActionEvent":
        """Create action event."""
        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.ACTION,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "action_id": action_id,
                "action_type": action_type,
                "parameters": parameters or {},
                "result": result or {},
                "error": error,
            },
        )


# ---------------------------------------------------------------------------
# Meta Events
# ---------------------------------------------------------------------------


class MetaEventType(Enum):
    """Metacognition event types."""

    REVIEW_STARTED = "meta.review_started"
    REVIEW_COMPLETED = "meta.review_completed"
    ANOMALY_DETECTED = "meta.anomaly_detected"
    CALIBRATION_UPDATED = "meta.calibration_updated"
    CONFIDENCE_ALERT = "meta.confidence_alert"


@dataclass
class MetaEvent(AGIEvent):
    """Event from metacognition."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: MetaEventType,
        target_id: str = "",
        confidence: float = 0.0,
        decision: str = "",
        issues: Optional[List[str]] = None,
        **kwargs,
    ) -> "MetaEvent":
        """Create meta event."""
        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.META,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "target_id": target_id,
                "confidence": confidence,
                "decision": decision,
                "issues": issues or [],
            },
        )


# ---------------------------------------------------------------------------
# System Events
# ---------------------------------------------------------------------------


class SystemEventType(Enum):
    """System event types."""

    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    SERVICE_HEALTHY = "system.service_healthy"
    SERVICE_UNHEALTHY = "system.service_unhealthy"
    CONFIG_UPDATED = "system.config_updated"
    ERROR = "system.error"
    WARNING = "system.warning"


@dataclass
class SystemEvent(AGIEvent):
    """System infrastructure event."""

    @classmethod
    def create(
        cls,
        source: str,
        event_type: SystemEventType,
        message: str = "",
        details: Optional[Dict] = None,
        **kwargs,
    ) -> "SystemEvent":
        """Create system event."""
        header = EventHeader(
            event_type=event_type.value,
            category=EventCategory.SYSTEM,
            source_service=source,
            **kwargs,
        )
        return cls(
            header=header,
            payload={
                "message": message,
                "details": details or {},
            },
        )
