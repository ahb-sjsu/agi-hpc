# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Semantic Memory module for AGI-HPC.

Provides vector-based knowledge storage and retrieval:
- Facts and concepts with embeddings
- Similarity search with filtering
- Multiple embedding model support
- Qdrant vector database backend

Usage:
    from agi.memory.semantic import SemanticMemoryClient

    client = SemanticMemoryClient()
    await client.store_fact("The sun is a star", domain="astronomy")
    results = await client.search("What is the sun?")
"""

from agi.memory.semantic.client import (
    SemanticMemoryClient,
    Fact,
    SearchResult,
)
from agi.memory.semantic.qdrant_store import (
    QdrantVectorStore,
    QdrantConfig,
    VectorSearchResult,
    create_vector_store,
)
from agi.memory.semantic.embedders import (
    EmbeddingModel,
    BaseEmbedder,
    EmbeddingResult,
)

__all__ = [
    # Client
    "SemanticMemoryClient",
    "Fact",
    "SearchResult",
    # Vector Store
    "QdrantVectorStore",
    "QdrantConfig",
    "VectorSearchResult",
    "create_vector_store",
    # Embedders
    "EmbeddingModel",
    "BaseEmbedder",
    "EmbeddingResult",
]

# Optional embedder exports
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
