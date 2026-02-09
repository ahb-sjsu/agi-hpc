# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unified Memory Service.

Provides combined query interface for planning context enrichment,
aggregating results from Semantic, Episodic, and Procedural memory.
"""

from agi.memory.unified.service import (
    UnifiedMemoryService,
    UnifiedMemoryServicer,
    UnifiedMemoryConfig,
    PlanningContext,
)

__all__ = [
    "UnifiedMemoryService",
    "UnifiedMemoryServicer",
    "UnifiedMemoryConfig",
    "PlanningContext",
]
