# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unified search across all 3.3M corpus documents.

Searches three corpora in parallel and merges results with
corpus-aware normalization:

  - chunks (112K): Code from 27 research repos
  - ethics_chunks (2.4M): Cross-civilizational ethics corpus
  - publications (824K): Academic paper metadata

Each corpus uses PCA-384 IVFFlat as the primary search method.
Results are normalized to a common [0, 1] score range because
raw cosine distances are not comparable across corpora with
different embedding distributions.

Usage::

    from agi.common.unified_search import UnifiedSearcher

    searcher = UnifiedSearcher()
    results = searcher.search("What does Confucius say about justice?")
    for r in results:
        print(f"[{r['corpus']}] {r['title']} score={r['score']:.3f}")
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

DB_DSN = "dbname=atlas user=claude"
PCA_PATH = "/home/claude/agi-hpc/data/pca_rotation_384.pkl"


@dataclass
class CorpusConfig:
    """Configuration for a searchable corpus."""

    table: str
    name: str
    content_col: str
    title_expr: str  # SQL expression for display title
    metadata_cols: list  # columns to include in results
    ivfflat_probes: int = 10
    weight: float = 1.0  # relative weight in merged results


# Corpus definitions
CORPORA = [
    CorpusConfig(
        table="chunks",
        name="code",
        content_col="content",
        title_expr="repo || '/' || file_path",
        metadata_cols=["repo", "file_path"],
        ivfflat_probes=10,
        weight=1.0,
    ),
    CorpusConfig(
        table="ethics_chunks",
        name="ethics",
        content_col="content",
        title_expr="tradition || ': ' || coalesce(source_ref, 'unknown')",
        metadata_cols=["tradition", "language", "period", "source_ref"],
        ivfflat_probes=20,
        weight=1.0,
    ),
    CorpusConfig(
        table="publications",
        name="publications",
        content_col="title",
        title_expr="title || ' by ' || coalesce(author, 'unknown')",
        metadata_cols=["title", "author", "year", "topic", "language"],
        ivfflat_probes=15,
        weight=0.8,
    ),
    CorpusConfig(
        table="arxiv_papers",
        name="arxiv",
        content_col="abstract",
        title_expr="title",
        metadata_cols=["arxiv_id", "categories", "datestamp"],
        ivfflat_probes=20,
        weight=1.0,
    ),
    CorpusConfig(
        table="academic_papers",
        name="academic",
        content_col="abstract",
        title_expr="title || ' (' || coalesce(source, '') || ')'",
        metadata_cols=["source", "authors", "year", "doi"],
        ivfflat_probes=20,
        weight=0.9,
    ),
    CorpusConfig(
        table="wikipedia_chunks",
        name="wikipedia",
        content_col="content",
        title_expr="article_title",
        metadata_cols=["article_title"],
        ivfflat_probes=30,
        weight=0.7,  # lower weight — general knowledge, not research-specific
    ),
    CorpusConfig(
        table="gutenberg_chunks",
        name="gutenberg",
        content_col="content",
        title_expr="book_title",
        metadata_cols=["book_id", "book_title"],
        ivfflat_probes=20,
        weight=0.5,  # lowest weight — literature, not technical
    ),
]


class UnifiedSearcher:
    """Search across all 3.3M corpus documents.

    Uses PCA-384 IVFFlat on each corpus, with per-corpus score
    normalization via min-max scaling to ensure fair ranking across
    corpora with different similarity distributions.

    Args:
        db_dsn: PostgreSQL connection string.
        pca_path: Path to PCA rotation matrix.
        corpora: List of CorpusConfig (defaults to all three).
        per_corpus_k: How many candidates to retrieve per corpus before merging.
    """

    def __init__(
        self,
        db_dsn: str = DB_DSN,
        pca_path: str = PCA_PATH,
        corpora: Optional[List[CorpusConfig]] = None,
        per_corpus_k: int = 10,
    ) -> None:
        self.db_dsn = db_dsn
        self.corpora = corpora or CORPORA
        self.per_corpus_k = per_corpus_k

        # Load PCA
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
            self._pca_components = pca["components"].T.astype(np.float32)
            self._pca_mean = pca["mean"].astype(np.float32)
            logger.info(
                "[unified-search] PCA-384 loaded (%.1f%% variance)",
                pca["variance_captured"] * 100,
            )

    def _pca_project(self, embedding: np.ndarray) -> np.ndarray:
        """Project 1024-dim embedding to PCA-384, L2-normalized."""
        centered = embedding.astype(np.float32) - self._pca_mean
        projected = centered @ self._pca_components
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected = projected / norm
        return projected

    def search(
        self,
        query: str = "",
        embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        corpora_filter: Optional[List[str]] = None,
        tradition_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[dict]:
        """Search all corpora and return merged, normalized results.

        Args:
            query: Text query (used for FTS fallback).
            embedding: Pre-computed 1024-dim embedding. If None, vector
                search is skipped and only FTS is used.
            top_k: Total results to return after merging.
            corpora_filter: List of corpus names to search (e.g. ["ethics"]).
                Default searches all.
            tradition_filter: Filter ethics corpus by tradition.
            language_filter: Filter by language.

        Returns:
            List of result dicts with normalized scores, sorted best-first.
        """
        if self._pca_components is None:
            logger.warning("[unified-search] No PCA model, returning empty")
            return []

        # PCA project the query
        q_pca = None
        if embedding is not None:
            q_pca = self._pca_project(embedding)
        pca_str = str(q_pca.tolist()) if q_pca is not None else None

        all_results = []

        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()

            for corpus in self.corpora:
                if corpora_filter and corpus.name not in corpora_filter:
                    continue

                results = self._search_corpus(
                    cur,
                    corpus,
                    pca_str,
                    query,
                    self.per_corpus_k,
                    tradition_filter=tradition_filter,
                    language_filter=language_filter,
                )
                all_results.extend(results)

            conn.close()
        except Exception as e:
            logger.error("[unified-search] DB error: %s", e)
            return []

        # Normalize scores across corpora
        normalized = self._normalize_and_merge(all_results, top_k)
        return normalized

    def _search_corpus(
        self,
        cur,
        corpus: CorpusConfig,
        pca_str: Optional[str],
        query: str,
        top_k: int,
        tradition_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[dict]:
        """Search a single corpus via PCA-384 IVFFlat."""
        results = []

        # Build WHERE clause for filters
        where_parts = ["embedding_pca384 IS NOT NULL"]
        params = []

        if tradition_filter and corpus.table == "ethics_chunks":
            where_parts.append("tradition = %s")
            params.append(tradition_filter)
        if language_filter and "language" in corpus.metadata_cols:
            where_parts.append("language = %s")
            params.append(language_filter)

        where_clause = " AND ".join(where_parts)

        if pca_str is not None:
            # Vector search
            cur.execute("SET ivfflat.probes = %s" % corpus.ivfflat_probes)

            meta_cols = ", ".join(corpus.metadata_cols)
            sql = (
                "SELECT id, %s AS title, %s, %s, "
                "1 - (embedding_pca384 <=> %%s::vector) AS score "
                "FROM %s WHERE %s "
                "ORDER BY embedding_pca384 <=> %%s::vector "
                "LIMIT %%s"
                % (
                    corpus.title_expr,
                    corpus.content_col,
                    meta_cols,
                    corpus.table,
                    where_clause,
                )
            )
            cur.execute(sql, [pca_str] + params + [pca_str, top_k])

            for row in cur.fetchall():
                idx = 0
                result = {
                    "id": row[idx],
                    "title": row[idx + 1],
                    "content": str(row[idx + 2])[:1500],
                    "raw_score": float(row[idx + 3 + len(corpus.metadata_cols)]),
                    "corpus": corpus.name,
                    "table": corpus.table,
                    "weight": corpus.weight,
                }
                # Add metadata
                for i, col in enumerate(corpus.metadata_cols):
                    result[col] = row[idx + 3 + i]
                results.append(result)
        elif query:
            # FTS fallback
            cur.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name=%s AND column_name='tsv'",
                (corpus.table,),
            )
            if cur.fetchone()[0] > 0:
                sql = (
                    "SELECT id, %s AS title, %s, "
                    "ts_rank(tsv, plainto_tsquery('english', %%s)) AS score "
                    "FROM %s WHERE tsv @@ plainto_tsquery('english', %%s) "
                    "AND %s "
                    "ORDER BY score DESC LIMIT %%s"
                    % (
                        corpus.title_expr,
                        corpus.content_col,
                        corpus.table,
                        where_clause,
                    )
                )
                cur.execute(sql, [query, query] + params + [top_k])
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "content": str(row[2])[:1500],
                        "raw_score": float(row[3]),
                        "corpus": corpus.name,
                        "table": corpus.table,
                        "weight": corpus.weight,
                    })

        return results

    def _normalize_and_merge(
        self, results: List[dict], top_k: int
    ) -> List[dict]:
        """Normalize scores per corpus and merge.

        Uses min-max normalization per corpus so that scores from different
        corpora (which have different similarity distributions) are comparable.
        Then applies corpus weight and sorts.
        """
        if not results:
            return []

        # Group by corpus
        by_corpus: dict[str, List[dict]] = {}
        for r in results:
            by_corpus.setdefault(r["corpus"], []).append(r)

        # Min-max normalize within each corpus
        normalized = []
        for corpus_name, corpus_results in by_corpus.items():
            scores = [r["raw_score"] for r in corpus_results]
            min_s = min(scores)
            max_s = max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0

            for r in corpus_results:
                # Normalized to [0, 1] within corpus, then weighted
                norm_score = (r["raw_score"] - min_s) / range_s
                r["score"] = norm_score * r["weight"]
                r["normalized_score"] = norm_score
                normalized.append(r)

        # Sort by weighted normalized score
        normalized.sort(key=lambda x: -x["score"])

        # Clean up internal fields
        for r in normalized:
            r.pop("raw_score", None)
            r.pop("weight", None)
            r.pop("table", None)

        return normalized[:top_k]

    def search_text(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Convenience: embed query text and search.

        Requires sentence-transformers to be available.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("BAAI/bge-m3", device="cpu")
            embedding = model.encode([query], normalize_embeddings=True)[0]
            return self.search(
                query=query, embedding=embedding, top_k=top_k, **kwargs
            )
        except ImportError:
            logger.warning("sentence-transformers not available, using FTS only")
            return self.search(query=query, top_k=top_k, **kwargs)

    def stats(self) -> dict:
        """Return corpus statistics."""
        if psycopg2 is None:
            return {}
        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()
            result = {}
            for corpus in self.corpora:
                cur.execute(
                    "SELECT count(*) FROM %s WHERE embedding_pca384 IS NOT NULL"
                    % corpus.table
                )
                result[corpus.name] = cur.fetchone()[0]
            result["total"] = sum(result.values())
            conn.close()
            return result
        except Exception:
            return {}
