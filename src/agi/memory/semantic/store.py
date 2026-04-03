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
Semantic Memory Store for AGI-HPC Phase 2.

Thin wrapper around the existing RAGSearcher from ``src/agi/lh/rag.py``,
providing a uniform memory-tier interface for the Memory Service broker.
Also adds a ``store()`` method for inserting new semantic entries.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

from agi.lh.rag import RAGConfig, RAGResult, RAGSearcher  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SemanticMemoryConfig:
    """Configuration for the semantic memory wrapper.

    Attributes:
        db_dsn: PostgreSQL connection string.
        embed_model_name: Sentence-transformer model for encoding.
        embed_device: Device for embedding inference.
        top_k: Default number of search results.
        score_threshold: Minimum similarity score for results.
    """

    db_dsn: str = "dbname=atlas user=claude"
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    top_k: int = 6
    score_threshold: float = 0.0


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Uniform semantic memory interface wrapping RAGSearcher.

    Provides ``search()`` for retrieval and ``store()`` for inserting
    new semantic entries into the pgvector ``chunks`` table.

    Usage::

        memory = SemanticMemory()
        results = memory.search("How does Paxos consensus work?")
        for r in results:
            print(f"[{r.repo}/{r.file}] {r.score:.3f}")

        memory.store(
            text="Raft is a consensus algorithm...",
            embedding=[0.1, 0.2, ...],  # 1024-dim
            metadata={"repo": "agi-hpc", "file_path": "docs/raft.md"},
        )
    """

    def __init__(self, config: Optional[SemanticMemoryConfig] = None) -> None:
        self._config = config or SemanticMemoryConfig()
        rag_config = RAGConfig(
            db_dsn=self._config.db_dsn,
            embed_model_name=self._config.embed_model_name,
            embed_device=self._config.embed_device,
            top_k=self._config.top_k,
            score_threshold=self._config.score_threshold,
        )
        self._searcher = RAGSearcher(config=rag_config)
        logger.info("[semantic] initialised with dsn=%s", self._config.db_dsn)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RAGResult]:
        """Search semantic memory for relevant document chunks.

        Args:
            query: Natural language search query.
            top_k: Number of results (overrides config default).

        Returns:
            List of RAGResult ordered by descending similarity.
        """
        results = self._searcher.search(query, top_k=top_k)
        logger.debug(
            "[semantic] search query=%r results=%d",
            query[:60],
            len(results),
        )
        return results

    def format_context(self, results: List[RAGResult]) -> str:
        """Format search results into a context string.

        Args:
            results: RAGResult list from search().

        Returns:
            Formatted context string for prompt injection.
        """
        return self._searcher.format_context(results)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new semantic entry in the chunks table.

        Args:
            text: Text content to store.
            embedding: 1024-dim embedding vector.
            metadata: Dict with optional 'repo' and 'file_path' keys.

        Returns:
            The UUID of the stored chunk.
        """
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )

        meta = metadata or {}
        repo = meta.get("repo", "manual")
        file_path = meta.get("file_path", "")
        chunk_id = str(uuid.uuid4())
        emb_str = str(embedding)

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (id, repo, file_path, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (chunk_id, repo, file_path, text, emb_str),
                )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("[semantic] failed to store chunk")
            raise

        logger.debug(
            "[semantic] stored chunk id=%s repo=%s",
            chunk_id[:8],
            repo,
        )
        return chunk_id
