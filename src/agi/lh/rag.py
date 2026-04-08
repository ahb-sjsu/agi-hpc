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
RAG (Retrieval-Augmented Generation) searcher for AGI-HPC Left Hemisphere.

Provides pgvector search against the Atlas knowledge base
(44K+ chunks from 27 research repositories).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


@dataclass
class RAGResult:
    """Single retrieval result from pgvector search.

    Attributes:
        repo: Source repository name.
        file: File path within the repository.
        text: Retrieved text chunk.
        score: Cosine similarity score (0-1, higher is better).
    """

    repo: str = ""
    file: str = ""
    text: str = ""
    score: float = 0.0


@dataclass
class RAGConfig:
    """Configuration for the RAG searcher.

    Attributes:
        db_dsn: PostgreSQL connection string.
        embed_model_name: Sentence-transformer model for query encoding.
        embed_device: Device for embedding inference (cpu/cuda).
        top_k: Default number of results to return.
        score_threshold: Minimum similarity score for results.
    """

    db_dsn: str = "dbname=atlas user=claude"
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    top_k: int = 6
    score_threshold: float = 0.0


class RAGSearcher:
    """Retrieval-Augmented Generation searcher using pgvector.

    Loads a sentence-transformer model and queries PostgreSQL with
    pgvector for semantically similar document chunks.

    Usage::

        searcher = RAGSearcher()
        results = searcher.search("How does Paxos consensus work?")
        for r in results:
            print(f"[{r.repo}/{r.file}] score={r.score:.3f}")
            print(r.text)
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required but not installed. "
                "Install with: pip install psycopg2-binary"
            )
        self._config = config or RAGConfig()
        self._embed_model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embed_model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(
                "[rag] loading embedding model '%s' on %s",
                self._config.embed_model_name,
                self._config.embed_device,
            )
            self._embed_model = SentenceTransformer(
                self._config.embed_model_name,
                device=self._config.embed_device,
            )
            logger.info("[rag] embedding model ready")
        return self._embed_model

    def search(self, query: str, top_k: Optional[int] = None) -> List[RAGResult]:
        """Search pgvector for relevant document chunks.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (overrides config default).

        Returns:
            List of RAGResult ordered by descending similarity score.
        """
        top_k = top_k or self._config.top_k
        model = self._ensure_model()

        # Encode query
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        emb_str = str(q_emb.tolist())

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT repo, file_path, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (emb_str, emb_str, top_k),
                )
                results = [
                    RAGResult(repo=r[0], file=r[1], text=r[2], score=float(r[3]))
                    for r in cur.fetchall()
                ]
            conn.close()
        except Exception:
            logger.exception("[rag] search failed")
            return []

        # Filter by threshold
        if self._config.score_threshold > 0:
            results = [r for r in results if r.score >= self._config.score_threshold]

        logger.debug(
            "[rag] query=%r top_k=%d results=%d",
            query[:60],
            top_k,
            len(results),
        )
        return results

    def format_context(self, results: List[RAGResult]) -> str:
        """Format RAG results into a context string for prompt injection.

        Args:
            results: List of RAGResult from search().

        Returns:
            Formatted string suitable for insertion into a system prompt.
        """
        if not results:
            return ""

        parts = []
        for r in results:
            parts.append(f"[{r.repo}/{r.file}] (relevance: {r.score:.3f})\n{r.text}")
        return "\n\n".join(parts)
