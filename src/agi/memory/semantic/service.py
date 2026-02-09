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
Semantic Memory Service (AGI-HPC)

Responsibilities:
- Store vector embeddings + metadata
- Support filtered nearest-neighbor search
- Provide Write() + Query() over gRPC
- EventFabric notifications (optional)

Now integrated with:
- QdrantVectorStore for production vector search
- Embedding models (SentenceTransformer/OpenAI)
- SemanticMemoryClient for high-level operations
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import Any, Dict, List, Optional

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

from agi.proto_gen.memory_pb2 import (
    SemanticWriteResponse,
    SemanticQueryResponse,
    SemanticHit,
    SemanticEntry,
)
from agi.proto_gen.memory_pb2_grpc import (
    SemanticServiceServicer,
    add_SemanticServiceServicer_to_server,
)

from agi.memory.semantic.client import SemanticMemoryClient, Fact
from agi.memory.semantic.qdrant_store import QdrantVectorStore, QdrantConfig
from agi.memory.semantic.embedders.base import EmbeddingModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector Store Adapter
# ---------------------------------------------------------------------------


class VectorStoreAdapter:
    """
    Adapter bridging gRPC proto types to the new QdrantVectorStore.

    Provides the legacy write/query interface while using modern infrastructure.
    """

    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: Optional[EmbeddingModel] = None,
    ):
        self._store = store
        self._embedder = embedder
        self._loop = asyncio.new_event_loop()

    def write(self, entries: List[SemanticEntry]) -> None:
        """Write entries to vector store.

        Args:
            entries: Proto SemanticEntry messages
        """
        if not entries:
            return

        ids = []
        vectors = []
        payloads = []

        for entry in entries:
            ids.append(entry.id)
            vectors.append(self._decode_embedding(entry.embedding))
            payloads.append(
                {
                    "content": entry.text,
                    "domain": entry.domain or "general",
                    "entity_type": entry.entity_type or "fact",
                    "source": entry.source or "",
                    "confidence": entry.confidence if entry.confidence else 1.0,
                    "metadata": dict(entry.metadata) if entry.metadata else {},
                }
            )

        self._store.upsert_batch(ids, vectors, payloads)
        logger.debug("[semantic] wrote %d entries", len(entries))

    def query(
        self,
        query_embedding: bytes,
        top_k: int,
        domain: str = "",
        min_score: float = 0.0,
    ) -> List[SemanticHit]:
        """Query for similar entries.

        Args:
            query_embedding: Query vector as bytes
            top_k: Maximum results
            domain: Optional domain filter
            min_score: Minimum similarity threshold

        Returns:
            List of SemanticHit proto messages
        """
        query_vector = self._decode_embedding(query_embedding)

        filters = {}
        if domain:
            filters["domain"] = domain

        results = self._store.search(
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
            filter_conditions=filters if filters else None,
        )

        hits = []
        for result in results:
            payload = result.payload
            entry = SemanticEntry(
                id=result.id,
                text=payload.get("content", ""),
                embedding=self._encode_embedding(result.vector or []),
                domain=payload.get("domain", ""),
                entity_type=payload.get("entity_type", ""),
                source=payload.get("source", ""),
                confidence=payload.get("confidence", 1.0),
            )
            hits.append(SemanticHit(entry=entry, score=result.score))

        return hits

    async def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        domain: str = "",
    ) -> List[SemanticHit]:
        """Search using text query (generates embedding automatically).

        Args:
            query_text: Natural language query
            top_k: Maximum results
            domain: Optional domain filter

        Returns:
            List of SemanticHit proto messages
        """
        if not self._embedder:
            raise RuntimeError("Embedder required for text search")

        embedding = await self._embedder.embed_single(query_text)
        return self.query(
            self._encode_embedding(embedding),
            top_k,
            domain,
        )

    @staticmethod
    def _decode_embedding(data: bytes) -> List[float]:
        """Decode bytes to float vector."""
        if not data:
            return []
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    @staticmethod
    def _encode_embedding(vector: List[float]) -> bytes:
        """Encode float vector to bytes."""
        if not vector:
            return b""
        return struct.pack(f"{len(vector)}f", *vector)


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------


class SemanticMemServicer(SemanticServiceServicer):
    """gRPC servicer for semantic memory operations."""

    def __init__(
        self,
        store: VectorStoreAdapter,
        fabric: EventFabric,
        client: Optional[SemanticMemoryClient] = None,
    ):
        self._store = store
        self._fabric = fabric
        self._client = client
        self._loop = asyncio.new_event_loop()

    def Write(self, request, context):
        """Handle Write RPC."""
        self._store.write(list(request.entries))
        self._fabric.publish(
            "memory.semantic.write",
            {
                "count": len(request.entries),
            },
        )
        return SemanticWriteResponse(ids=[e.id for e in request.entries])

    def Query(self, request, context):
        """Handle Query RPC."""
        hits = self._store.query(
            request.query_embedding,
            request.top_k,
            getattr(request, "domain", ""),
            getattr(request, "min_score", 0.0),
        )
        return SemanticQueryResponse(hits=hits)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SemanticMemoryService:
    """
    Semantic Memory Service with integrated vector store.

    Uses QdrantVectorStore for production vector similarity search
    and optional embedding models for automatic text embedding.
    """

    def __init__(self, config_path: str = "configs/memory_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()

        # Initialize vector store
        qdrant_config = self._build_qdrant_config()
        self._qdrant_store = QdrantVectorStore(qdrant_config)

        # Initialize embedder (optional, for text-based queries)
        self._embedder = self._create_embedder()

        # Create adapter for gRPC interface
        self._store_adapter = VectorStoreAdapter(
            self._qdrant_store,
            self._embedder,
        )

        # High-level client for internal use
        self._client = (
            SemanticMemoryClient(
                embedder=self._embedder,
                store=self._qdrant_store,
            )
            if self._embedder
            else None
        )

        self.grpc = GRPCServer(self.config.rpc_port)

    def _build_qdrant_config(self) -> QdrantConfig:
        """Build Qdrant configuration from service config."""
        qdrant_cfg = getattr(self.config, "qdrant", None) or {}

        return QdrantConfig(
            host=qdrant_cfg.get("host", "localhost"),
            port=qdrant_cfg.get("port", 6333),
            collection_name=qdrant_cfg.get("collection", "semantic_memory"),
            vector_size=qdrant_cfg.get("vector_size", 384),
            use_grpc=qdrant_cfg.get("use_grpc", True),
            prefer_grpc=qdrant_cfg.get("prefer_grpc", True),
        )

    def _create_embedder(self) -> Optional[EmbeddingModel]:
        """Create embedding model based on configuration."""
        embedder_cfg = getattr(self.config, "embedder", None) or {}
        embedder_type = embedder_cfg.get("type", "sentence_transformer")

        try:
            if embedder_type == "sentence_transformer":
                from agi.memory.semantic.embedders.sentence_transformer import (
                    SentenceTransformerEmbedder,
                )

                return SentenceTransformerEmbedder(
                    model_name=embedder_cfg.get("model", "all-MiniLM-L6-v2"),
                )
            elif embedder_type == "openai":
                from agi.memory.semantic.embedders.openai import OpenAIEmbedder

                return OpenAIEmbedder(
                    model_name=embedder_cfg.get("model", "text-embedding-3-small"),
                )
        except (ImportError, RuntimeError) as e:
            logger.warning("[semantic] embedder initialization failed: %s", e)

        return None

    def run(self) -> None:
        """Start the semantic memory service."""
        servicer = SemanticMemServicer(
            self._store_adapter,
            self.fabric,
            self._client,
        )
        add_SemanticServiceServicer_to_server(servicer, self.grpc.server)

        logger.info(
            "[semantic] service running on port %d",
            self.config.rpc_port,
        )
        print("[MEM-SEM] Semantic Memory service running...")

        self.grpc.start()
        self.grpc.wait()

    def close(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._qdrant_store.close()
        logger.info("[semantic] service closed")


def main():
    logging.basicConfig(level=logging.INFO)
    SemanticMemoryService().run()


if __name__ == "__main__":
    main()
