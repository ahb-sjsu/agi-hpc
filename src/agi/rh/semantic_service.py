"""
Semantic Memory Service

Implements the semantic memory subsystem described in the AGI-HPC
architecture:

    • Stores text + metadata as vectorized "knowledge items"
    • Supports semantic search over stored items
    • Exposes a gRPC SemanticMemoryService API
    • Pluggable backend (in-memory by default; Qdrant/FAISS later)

Public RPCs (as assumed by agi.lh.memory_client):

    rpc SemanticSearch(SemanticQuery) returns (SemanticResult);

Additional useful RPCs (for ingestion / admin):

    rpc UpsertItem(UpsertRequest) returns (UpsertResponse);
    rpc DeleteItem(DeleteRequest) returns (DeleteResponse);
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from agi.core.api.grpc_server import GRPCServer, GRPCServerConfig
from agi.proto_gen import memory_pb2, memory_pb2_grpc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend data model
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeItem:
    """
    A single semantic memory item.

    In a production deployment, this would correspond to a row in a
    vector DB (Qdrant/FAISS/PGVector/etc.).
    """

    item_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    # Simple embedding representation for the in-memory backend:
    # bag-of-words with term weights (pseudo TF).
    embedding: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend interface + in-memory implementation
# ---------------------------------------------------------------------------


class SemanticBackend:
    """
    Abstract backend for semantic memory.

    Implementations must provide:
        • upsert(item)
        • delete(item_id)
        • search(query_text, top_k, filters)
    """

    def upsert(self, item: KnowledgeItem) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, item_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemorySemanticBackend(SemanticBackend):
    """
    Simple in-memory semantic backend using bag-of-words + cosine similarity.

    This is NOT meant for very large scale, but is perfectly adequate for
    development and unit/integration tests. It also provides a clear
    reference for swapping in Qdrant/FAISS later.
    """

    def __init__(self) -> None:
        self._items: Dict[str, KnowledgeItem] = {}
        logger.info("[semantic] Using InMemorySemanticBackend")

    # -------------------------- public API ---------------------------------

    def upsert(self, item: KnowledgeItem) -> None:
        if not item.embedding:
            item.embedding = self._embed(item.text)
        self._items[item.item_id] = item

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        if not self._items:
            return []

        query_emb = self._embed(query_text)
        filt = filters or {}

        scored: List[Tuple[KnowledgeItem, float]] = []

        for item in self._items.values():
            if not self._passes_filters(item, filt):
                continue

            score = self._cosine(query_emb, item.embedding)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k <= 0:
            return scored
        return scored[:top_k]

    # -------------------------- internals ----------------------------------

    def _embed(self, text: str) -> Dict[str, float]:
        """
        Extremely simple embedding: lowercased bag-of-words with
        sqrt(count) weighting.

        Replace this with a real embedding model when integrating
        production infrastructure.
        """
        tokens = text.lower().split()
        counts: Dict[str, int] = {}
        for tok in tokens:
            if tok:
                counts[tok] = counts.get(tok, 0) + 1

        return {t: math.sqrt(c) for t, c in counts.items()}

    def _cosine(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> float:
        if not a or not b:
            return 0.0

        # dot product
        dot = 0.0
        for k, va in a.items():
            vb = b.get(k)
            if vb is not None:
                dot += va * vb

        # norms
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0

        return dot / (na * nb)

    def _passes_filters(
        self,
        item: KnowledgeItem,
        filters: Dict[str, str],
    ) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if item.metadata.get(k) != v:
                return False
        return True


# ---------------------------------------------------------------------------
# Service config + loader
# ---------------------------------------------------------------------------


@dataclass
class SemanticServiceConfig:
    port: int = 50110
    backend: str = "in_memory"  # future: "qdrant", "faiss", etc.


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_from_dict(cfg: Dict[str, Any]) -> SemanticServiceConfig:
    svc_cfg = cfg.get("semantic", {})
    return SemanticServiceConfig(
        port=int(svc_cfg.get("grpc_port", 50110)),
        backend=str(svc_cfg.get("backend", "in_memory")),
    )


def create_backend(cfg: SemanticServiceConfig) -> SemanticBackend:
    if cfg.backend == "in_memory":
        return InMemorySemanticBackend()
    # Hook for future backends:
    # if cfg.backend == "qdrant": return QdrantBackend(...)
    # if cfg.backend == "faiss": return FaissBackend(...)
    logger.warning(
        "[semantic] Unknown backend '%s', falling back to in-memory", cfg.backend
    )
    return InMemorySemanticBackend()


# ---------------------------------------------------------------------------
# gRPC SemanticMemoryService implementation
# ---------------------------------------------------------------------------


class SemanticMemoryService(memory_pb2_grpc.SemanticMemoryServiceServicer):
    """
    gRPC service implementation for Semantic Memory.

    Depends on a SemanticBackend for storage and search.
    """

    def __init__(self, backend: SemanticBackend) -> None:
        self._backend = backend
        logger.info("[semantic] SemanticMemoryService initialized")

    # --------------------- RPC: SemanticSearch -----------------------------

    def SemanticSearch(
        self,
        request: memory_pb2.SemanticQuery,
        context,
    ) -> memory_pb2.SemanticResult:
        """
        Entry point for LH MemoryClient.enrich_request, etc.

        Request fields (expected in memory.proto):
            string text = 1;
            int32 top_k = 2;
            map<string, string> filters = 3;
        """
        text = request.text
        top_k = request.top_k or 5
        filters = dict(request.filters)

        logger.info(
            "[semantic] SemanticSearch text_len=%d top_k=%d",
            len(text),
            top_k,
        )

        hits = self._backend.search(text, top_k, filters)

        resp = memory_pb2.SemanticResult()
        for item, score in hits:
            hit = resp.hits.add()
            hit.id = item.item_id
            hit.score = float(score)
            hit.text = item.text
            for k, v in item.metadata.items():
                hit.metadata[k] = v

        return resp

    # --------------------- RPC: UpsertItem --------------------------------

    def UpsertItem(
        self,
        request: memory_pb2.UpsertRequest,
        context,
    ) -> memory_pb2.UpsertResponse:
        """
        Ingest or update a single knowledge item.

        UpsertRequest (expected):
            string id = 1;
            string text = 2;
            map<string, string> metadata = 3;
        """
        kid = request.id or request.text[:32] or "item"
        text = request.text
        metadata = dict(request.metadata)

        logger.info("[semantic] UpsertItem id=%s text_len=%d", kid, len(text))

        item = KnowledgeItem(
            item_id=kid,
            text=text,
            metadata=metadata,
        )
        self._backend.upsert(item)

        return memory_pb2.UpsertResponse(ok=True)

    # --------------------- RPC: DeleteItem --------------------------------

    def DeleteItem(
        self,
        request: memory_pb2.DeleteRequest,
        context,
    ) -> memory_pb2.DeleteResponse:
        """
        Delete a knowledge item by id.
        """
        kid = request.id
        logger.info("[semantic] DeleteItem id=%s", kid)
        self._backend.delete(kid)
        return memory_pb2.DeleteResponse(ok=True)


# ---------------------------------------------------------------------------
# Service bootstrap
# ---------------------------------------------------------------------------


def run_semantic_service(config_path: str) -> None:
    cfg_dict = load_yaml(config_path)

    # Logging
    logging.basicConfig(
        level=cfg_dict.get("logging", {}).get("level", "INFO").upper(),
        format="[%(levelname)s][%(name)s] %(message)s",
    )

    svc_cfg = config_from_dict(cfg_dict)
    backend = create_backend(svc_cfg)
    service = SemanticMemoryService(backend)

    server_cfg = GRPCServerConfig(port=svc_cfg.port)
    server = GRPCServer(server_cfg)

    memory_pb2_grpc.add_SemanticMemoryServiceServicer_to_server(
        service,
        server._server,
    )

    server.add_signal_handlers()
    logger.info("[semantic] Starting SemanticMemoryService on port %d", svc_cfg.port)
    server.start()
    server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Memory Service")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to semantic memory YAML config.",
    )
    args = parser.parse_args()

    try:
        run_semantic_service(args.config)
    except Exception:
        logger.exception("[semantic] Fatal error in SemanticMemoryService")
        sys.exit(1)


if __name__ == "__main__":
    main()
