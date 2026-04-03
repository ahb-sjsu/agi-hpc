# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Memory Subsystem for AGI-HPC.

Provides three types of memory:
- Semantic Memory: Facts, concepts, schemas, knowledge graph
- Episodic Memory: Experiences, events, task executions
- Procedural Memory: Skills, action sequences, learned procedures

Phase 2 (NATS-connected, Atlas-native):
- EpisodicMemory: PostgreSQL/pgvector conversation history
- ProceduralMemory: SQLite-backed learned behaviours
- SemanticMemory: pgvector wrapper around RAGSearcher
- MemoryService: NATS-connected memory broker

Legacy (Sprint 1-6, gRPC-based):
- UnifiedMemoryService: Combined query interface (requires grpc)
"""

from __future__ import annotations

__all__ = [
    # Phase 2 (NATS-connected)
    "EpisodicMemory",
    "ProceduralMemory",
    "SemanticMemory",
    "MemoryService",
    # Legacy (gRPC-based)
    "UnifiedMemoryService",
    "UnifiedMemoryServicer",
    "UnifiedMemoryConfig",
    "PlanningContext",
]

# Phase 2 stores (Atlas-native)
try:
    from agi.memory.episodic.store import EpisodicMemory
except ImportError:
    pass

try:
    from agi.memory.procedural.store import ProceduralMemory
except ImportError:
    pass

try:
    from agi.memory.semantic.store import SemanticMemory
except ImportError:
    pass

try:
    from agi.memory.nats_service import MemoryService
except ImportError:
    pass

# Legacy unified interface (gRPC-based, Sprint 1-6)
try:
    from agi.memory.unified import (
        UnifiedMemoryService,
        UnifiedMemoryServicer,
        UnifiedMemoryConfig,
        PlanningContext,
    )
except ImportError:
    pass
