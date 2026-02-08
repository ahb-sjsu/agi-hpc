# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Semantic Memory Client for AGI-HPC.

High-level API for storing and retrieving semantic knowledge:
- Facts and concepts
- Entity relationships
- Tool schemas
- Vector similarity search

Usage:
    from agi.memory.semantic import SemanticMemoryClient

    client = SemanticMemoryClient()

    # Store a fact
    await client.store_fact(
        content="The sky is blue due to Rayleigh scattering",
        domain="physics",
        entity_type="fact",
    )

    # Search for related facts
    results = await client.search("What color is the sky?", limit=5)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agi.memory.semantic.embedders.base import EmbeddingModel
from agi.memory.semantic.qdrant_store import (
    QdrantVectorStore,
    QdrantConfig,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A semantic fact stored in memory."""

    id: str
    content: str
    domain: str = "general"
    entity_type: str = "fact"
    source: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to storage payload."""
        return {
            "content": self.content,
            "domain": self.domain,
            "entity_type": self.entity_type,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    @classmethod
    def from_search_result(cls, result: VectorSearchResult) -> "Fact":
        """Create from search result."""
        payload = result.payload
        return cls(
            id=result.id,
            content=payload.get("content", ""),
            domain=payload.get("domain", "general"),
            entity_type=payload.get("entity_type", "fact"),
            source=payload.get("source", ""),
            confidence=payload.get("confidence", 1.0),
            timestamp=payload.get("timestamp", 0.0),
            metadata={
                k: v for k, v in payload.items()
                if k not in (
                    "content", "domain", "entity_type",
                    "source", "confidence", "timestamp"
                )
            },
        )


@dataclass
class SearchResult:
    """Result from semantic search."""

    fact: Fact
    score: float
    vector: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Semantic Memory Client
# ---------------------------------------------------------------------------


class SemanticMemoryClient:
    """
    High-level client for semantic memory operations.

    Combines embedding model with vector store for:
    - Storing facts with automatic embedding
    - Semantic similarity search
    - Filtered retrieval
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        store: Optional[QdrantVectorStore] = None,
        store_config: Optional[QdrantConfig] = None,
    ):
        """
        Initialize semantic memory client.

        Args:
            embedder: Embedding model (creates default if not provided)
            store: Vector store (creates default if not provided)
            store_config: Config for vector store
        """
        # Create default embedder if not provided
        if embedder is None:
            embedder = self._create_default_embedder()
        self._embedder = embedder

        # Create default store if not provided
        if store is None:
            config = store_config or QdrantConfig(
                vector_size=embedder.dimension
            )
            store = QdrantVectorStore(config)
        self._store = store

        logger.info(
            "[memory][semantic] client initialized embedder=%s",
            embedder.model_name,
        )

    def _create_default_embedder(self) -> EmbeddingModel:
        """Create default embedding model."""
        # Try sentence-transformers first (local)
        try:
            from agi.memory.semantic.embedders.sentence_transformer import (
                SentenceTransformerEmbedder,
            )
            return SentenceTransformerEmbedder()
        except (ImportError, RuntimeError):
            pass

        # Fall back to OpenAI
        try:
            from agi.memory.semantic.embedders.openai import OpenAIEmbedder
            return OpenAIEmbedder()
        except (ImportError, RuntimeError):
            pass

        raise RuntimeError(
            "No embedding model available. Install sentence-transformers "
            "or set OPENAI_API_KEY."
        )

    async def store_fact(
        self,
        content: str,
        domain: str = "general",
        entity_type: str = "fact",
        source: str = "",
        confidence: float = 1.0,
        fact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        """Store a fact in semantic memory.

        Args:
            content: The fact content
            domain: Knowledge domain (e.g., "physics", "history")
            entity_type: Type of entity (e.g., "fact", "concept", "entity")
            source: Source of the fact
            confidence: Confidence score (0-1)
            fact_id: Optional custom ID
            metadata: Additional metadata

        Returns:
            Stored fact with ID
        """
        # Generate ID if not provided
        fact_id = fact_id or str(uuid.uuid4())

        # Create fact
        fact = Fact(
            id=fact_id,
            content=content,
            domain=domain,
            entity_type=entity_type,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Generate embedding
        embedding = await self._embedder.embed_single(content)

        # Store in vector database
        self._store.upsert(
            id=fact_id,
            vector=embedding,
            payload=fact.to_payload(),
        )

        logger.debug("[memory][semantic] stored fact id=%s", fact_id)

        return fact

    async def store_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Fact]:
        """Store multiple facts in batch.

        Args:
            facts: List of fact dicts with content, domain, etc.

        Returns:
            List of stored facts
        """
        if not facts:
            return []

        # Extract contents for batch embedding
        contents = [f["content"] for f in facts]

        # Batch embed
        embeddings = await self._embedder.embed(contents)

        # Prepare for batch upsert
        result_facts = []
        ids = []
        vectors = []
        payloads = []

        for fact_dict, embedding in zip(facts, embeddings):
            fact_id = fact_dict.get("id") or str(uuid.uuid4())
            fact = Fact(
                id=fact_id,
                content=fact_dict["content"],
                domain=fact_dict.get("domain", "general"),
                entity_type=fact_dict.get("entity_type", "fact"),
                source=fact_dict.get("source", ""),
                confidence=fact_dict.get("confidence", 1.0),
                metadata=fact_dict.get("metadata", {}),
            )

            ids.append(fact_id)
            vectors.append(embedding)
            payloads.append(fact.to_payload())
            result_facts.append(fact)

        # Batch upsert
        self._store.upsert_batch(ids, vectors, payloads)

        logger.debug("[memory][semantic] batch stored %d facts", len(facts))

        return result_facts

    async def search(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for semantically similar facts.

        Args:
            query: Search query text
            limit: Maximum results to return
            domain: Filter by domain
            entity_type: Filter by entity type
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = await self._embedder.embed_single(query)

        # Build filters
        filters = {}
        if domain:
            filters["domain"] = domain
        if entity_type:
            filters["entity_type"] = entity_type

        # Search
        results = self._store.search(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=min_score,
            filter_conditions=filters if filters else None,
        )

        return [
            SearchResult(
                fact=Fact.from_search_result(r),
                score=r.score,
                vector=r.vector,
            )
            for r in results
        ]

    async def get(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        result = self._store.get(fact_id)
        if result is None:
            return None
        return Fact.from_search_result(result)

    async def delete(self, fact_ids: Union[str, List[str]]) -> None:
        """Delete facts by ID."""
        self._store.delete(fact_ids)

    async def clear(self) -> None:
        """Clear all facts from memory."""
        self._store.clear()

    def count(self) -> int:
        """Get total number of stored facts."""
        return self._store.count()

    def close(self) -> None:
        """Close the client."""
        self._store.close()
        logger.info("[memory][semantic] client closed")
