# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Integration Module for AGI-HPC.

Provides inter-subsystem communication infrastructure:
- Standard event schemas
- Event bus for pub/sub messaging
- Service registry (upcoming)
- Distributed tracing
"""

from agi.integration.events import (
    AGIEvent,
    EventBus,
    EventCategory,
    EventHeader,
    EventPriority,
    create_event_bus,
)

__all__ = [
    "AGIEvent",
    "EventBus",
    "EventCategory",
    "EventHeader",
    "EventPriority",
    "create_event_bus",
]
