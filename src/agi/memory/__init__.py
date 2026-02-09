# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Memory Subsystem for AGI-HPC.

Provides three types of memory:
- Semantic Memory: Facts, concepts, schemas, knowledge graph
- Episodic Memory: Experiences, events, task executions
- Procedural Memory: Skills, action sequences, learned procedures

Plus a unified interface:
- Unified Memory: Combined query interface for planning context enrichment
"""

from agi.memory.unified import (
    UnifiedMemoryService,
    UnifiedMemoryServicer,
    UnifiedMemoryConfig,
    PlanningContext,
)

__all__ = [
    # Unified Memory
    "UnifiedMemoryService",
    "UnifiedMemoryServicer",
    "UnifiedMemoryConfig",
    "PlanningContext",
]
