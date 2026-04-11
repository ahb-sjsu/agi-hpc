# AGI-HPC Knowledge Extraction & Graph
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Structured knowledge extraction from documents and knowledge graph storage."""

from __future__ import annotations

__all__ = [
    "Entity",
    "ExtractedKnowledge",
    "KnowledgeExtractionConfig",
    "KnowledgeExtractor",
    "KnowledgeGraph",
    "KnowledgeGraphConfig",
    "Relationship",
]

try:
    from agi.memory.knowledge.extractor import (
        Entity,
        ExtractedKnowledge,
        KnowledgeExtractionConfig,
        KnowledgeExtractor,
        Relationship,
    )
except ImportError:
    pass

try:
    from agi.memory.knowledge.graph import KnowledgeGraph, KnowledgeGraphConfig
except ImportError:
    pass
