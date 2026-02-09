# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Integration Events Module.

Provides standard event schemas and event bus for AGI-HPC subsystem communication.
"""

from agi.integration.events.schema import (
    AGIEvent,
    EventCategory,
    EventHeader,
    EventPriority,
    PerceptionEvent,
    PerceptionEventType,
    ReasoningEvent,
    ReasoningEventType,
    PlanningEvent,
    PlanningEventType,
    MemoryEvent,
    MemoryEventType,
    SafetyEvent,
    SafetyEventType,
    ActionEvent,
    ActionEventType,
    MetaEvent,
    MetaEventType,
    SystemEvent,
    SystemEventType,
)

from agi.integration.events.bus import (
    EventBus,
    Subscription,
    TraceContext,
    create_event_bus,
)

__all__ = [
    # Base classes
    "AGIEvent",
    "EventCategory",
    "EventHeader",
    "EventPriority",
    # Perception
    "PerceptionEvent",
    "PerceptionEventType",
    # Reasoning
    "ReasoningEvent",
    "ReasoningEventType",
    # Planning
    "PlanningEvent",
    "PlanningEventType",
    # Memory
    "MemoryEvent",
    "MemoryEventType",
    # Safety
    "SafetyEvent",
    "SafetyEventType",
    # Action
    "ActionEvent",
    "ActionEventType",
    # Meta
    "MetaEvent",
    "MetaEventType",
    # System
    "SystemEvent",
    "SystemEventType",
    # Event Bus
    "EventBus",
    "Subscription",
    "TraceContext",
    "create_event_bus",
]
