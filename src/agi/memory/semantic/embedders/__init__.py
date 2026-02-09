# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Embedding models for semantic memory.

Available embedders:
- SentenceTransformerEmbedder: Local models (all-MiniLM-L6-v2, etc.)
- OpenAIEmbedder: OpenAI API (text-embedding-3-small, etc.)
"""

from agi.memory.semantic.embedders.base import (
    EmbeddingModel,
    BaseEmbedder,
    EmbeddingResult,
    compute_text_hash,
)

__all__ = [
    "EmbeddingModel",
    "BaseEmbedder",
    "EmbeddingResult",
    "compute_text_hash",
]

# Optional imports
try:
    from agi.memory.semantic.embedders.sentence_transformer import (
        SentenceTransformerEmbedder,
    )

    __all__.append("SentenceTransformerEmbedder")
except ImportError:
    pass

try:
    from agi.memory.semantic.embedders.openai import OpenAIEmbedder

    __all__.append("OpenAIEmbedder")
except ImportError:
    pass
