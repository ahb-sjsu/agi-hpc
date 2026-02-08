# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Qdrant vector store for semantic memory.

Provides persistent vector storage with:
- Cosine similarity search
- Metadata filtering
- Batch upsert/delete operations
- Collection management

Environment Variables:
    QDRANT_URL          Qdrant server URL (default: localhost:6333)
    QDRANT_API_KEY      API key for Qdrant Cloud (optional)
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        PointStruct,
        VectorParams,
        SearchRequest,
        Record,
    )
except ImportError:
    QdrantClient = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector store."""

    url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "localhost:6333")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )
    collection_name: str = "semantic_facts"
    vector_size: int = 768  # Default for all-mpnet-base-v2
    distance: str = "cosine"
    on_disk: bool = False


# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------


class QdrantVectorStore:
    """
    Qdrant-backed vector store for semantic memory.

    Features:
    - Persistent storage
    - Cosine/Euclidean/Dot product similarity
    - Metadata filtering
    - Batch operations
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.config = config or QdrantConfig()
        self._client: Optional[QdrantClient] = None
        self._connected = False

        logger.info(
            "[memory][qdrant] initialized url=%s collection=%s",
            self.config.url,
            self.config.collection_name,
        )

    def connect(self) -> None:
        """Connect to Qdrant server."""
        if self._connected:
            return

        # Parse URL
        if "://" in self.config.url:
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
            )
        else:
            # host:port format
            parts = self.config.url.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 6333

            self._client = QdrantClient(
                host=host,
                port=port,
                api_key=self.config.api_key,
            )

        self._connected = True
        logger.info("[memory][qdrant] connected to %s", self.config.url)

    def ensure_collection(self) -> None:
        """Ensure collection exists with correct schema."""
        if not self._connected:
            self.connect()

        # Check if collection exists
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.collection_name in collection_names:
            logger.debug(
                "[memory][qdrant] collection %s exists",
                self.config.collection_name,
            )
            return

        # Create collection
        distance = Distance.COSINE
        if self.config.distance.lower() == "euclidean":
            distance = Distance.EUCLID
        elif self.config.distance.lower() == "dot":
            distance = Distance.DOT

        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
                on_disk=self.config.on_disk,
            ),
        )

        logger.info(
            "[memory][qdrant] created collection %s size=%d",
            self.config.collection_name,
            self.config.vector_size,
        )

    def upsert(
        self,
        id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """Insert or update a single vector."""
        if not self._connected:
            self.connect()
            self.ensure_collection()

        self._client.upsert(
            collection_name=self.config.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug("[memory][qdrant] upserted id=%s", id)

    def upsert_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Batch insert or update vectors."""
        if not self._connected:
            self.connect()
            self.ensure_collection()

        points = [
            PointStruct(id=id, vector=vector, payload=payload)
            for id, vector, payload in zip(ids, vectors, payloads)
        ]

        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

        logger.debug("[memory][qdrant] batch upserted %d points", len(points))

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Maximum results to return
            score_threshold: Minimum score threshold
            filter_conditions: Metadata filters (field: value)

        Returns:
            List of search results
        """
        if not self._connected:
            self.connect()
            self.ensure_collection()

        # Build filter
        search_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filter_conditions.items()
            ]
            search_filter = Filter(must=must_conditions)

        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
        )

        return [
            VectorSearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                vector=r.vector,
            )
            for r in results
        ]

    def get(self, id: str) -> Optional[VectorSearchResult]:
        """Retrieve a vector by ID."""
        if not self._connected:
            self.connect()

        results = self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            return None

        r = results[0]
        return VectorSearchResult(
            id=str(r.id),
            score=1.0,
            payload=r.payload or {},
            vector=r.vector,
        )

    def delete(self, ids: Union[str, List[str]]) -> None:
        """Delete vectors by ID."""
        if not self._connected:
            self.connect()

        if isinstance(ids, str):
            ids = [ids]

        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=ids),
        )

        logger.debug("[memory][qdrant] deleted %d points", len(ids))

    def count(self) -> int:
        """Get total number of vectors."""
        if not self._connected:
            self.connect()

        info = self._client.get_collection(self.config.collection_name)
        return info.points_count

    def clear(self) -> None:
        """Delete all vectors in collection."""
        if not self._connected:
            self.connect()

        # Recreate collection
        self._client.delete_collection(self.config.collection_name)
        self.ensure_collection()

        logger.info("[memory][qdrant] cleared collection")

    def close(self) -> None:
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False
        logger.info("[memory][qdrant] closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vector_store(
    url: Optional[str] = None,
    collection: str = "semantic_facts",
    vector_size: int = 768,
) -> QdrantVectorStore:
    """Create a Qdrant vector store.

    Args:
        url: Qdrant server URL
        collection: Collection name
        vector_size: Vector dimension

    Returns:
        Configured QdrantVectorStore
    """
    config = QdrantConfig(
        url=url or os.getenv("QDRANT_URL", "localhost:6333"),
        collection_name=collection,
        vector_size=vector_size,
    )
    store = QdrantVectorStore(config)
    store.connect()
    store.ensure_collection()
    return store
