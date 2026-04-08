# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Hybrid search combining vector similarity and full-text search via
Reciprocal Rank Fusion (RRF).

Implements the 3-tier search cascade:
  Tier 1: Compiled wiki article lookup (instant, exact match)
  Tier 2: Hybrid RRF(vector + FTS) on chunks (semantic + keyword)
  Tier 3: Vector-only on large corpora (ethics, cross-lingual)

RRF merges ranked lists without score calibration::

    RRF_score(doc) = sum(1 / (k + rank_i)) for each retriever i

Reference: Cormack, Clarke, Butt (2009). "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods."

Usage::

    from agi.common.hybrid_search import HybridSearcher, SearchResult

    searcher = HybridSearcher(db_dsn="dbname=atlas user=claude")
    results = searcher.search("How does the safety gateway work?", top_k=6)
    for r in results:
        print(f"[{r.source}] {r.repo}/{r.file} score={r.score:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore


# ------------------------------------------------------------------ #
# Data classes                                                        #
# ------------------------------------------------------------------ #


def extract_keywords(content: str, file_path: str = "") -> str:
    """Extract normalized keywords from code/text content.

    Extracts identifiers (class names, function names, imports),
    normalizes CamelCase and snake_case, and weights structural
    elements higher.

    Args:
        content: Raw text content.
        file_path: File path for language-aware extraction.

    Returns:
        Space-separated normalized keyword string for tsvector indexing.
    """
    import re

    keywords = []
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()

        # Python: imports (high weight — repeat for boosting)
        if stripped.startswith(("import ", "from ")):
            modules = re.findall(r"[\w.]+", stripped)
            for m in modules:
                parts = m.split(".")
                keywords.extend(parts * 2)  # double weight

        # Python: class/def definitions (high weight)
        match = re.match(r"(?:class|def)\s+(\w+)", stripped)
        if match:
            name = match.group(1)
            keywords.append(name)
            keywords.append(name)  # double weight
            # Split CamelCase
            keywords.extend(_split_identifier(name))

        # General: extract identifiers
        identifiers = re.findall(r"\b[a-zA-Z_]\w{2,}\b", stripped)
        for ident in identifiers:
            keywords.extend(_split_identifier(ident))

    # Add file path components
    if file_path:
        parts = re.split(r"[/\\.]", file_path)
        keywords.extend(p.lower() for p in parts if len(p) > 2)

    # Deduplicate while preserving weight (count)
    return " ".join(keywords)


def _split_identifier(name: str) -> list[str]:
    """Split CamelCase and snake_case identifiers into words.

    Examples:
        TurboQuantKV -> [turbo, quant, kv]
        safety_gateway -> [safety, gateway]
        BGE_M3 -> [bge, m3]
    """
    import re

    # snake_case
    parts = name.split("_")
    words = []
    for part in parts:
        # CamelCase split
        tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", part)
        words.extend(t.lower() for t in tokens if len(t) > 1)
    return words


def llm_extract_keywords(
    content: str,
    file_path: str = "",
    llm_url: str = "http://localhost:8080",
    timeout: float = 30.0,
) -> str:
    """Use Spock (Gemma 4) to extract structured keywords from content.

    The LLM identifies conceptual keywords, synonyms, and importance
    that regex-based extraction misses. Falls back to regex extraction
    on failure.

    Args:
        content: Raw text/code content (truncated to ~2000 chars).
        file_path: File path for context.
        llm_url: Spock/LLM server URL.
        timeout: Request timeout in seconds.

    Returns:
        Space-separated keyword string.
    """
    try:
        import requests

        prompt = (
            "Extract the most important technical keywords from this code/text. "
            "Include: class names, function names, concepts, algorithms, "
            "libraries, and architectural patterns. "
            "Output ONLY a comma-separated list of keywords, nothing else.\n\n"
            f"File: {file_path}\n"
            f"Content:\n{content[:2000]}"
        )

        resp = requests.post(
            f"{llm_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            keywords = data["choices"][0]["message"]["content"]
            # Clean up: remove quotes, normalize
            keywords = keywords.replace('"', "").replace("'", "")
            words = [w.strip().lower() for w in keywords.split(",")]
            return " ".join(w for w in words if w)
    except Exception as e:
        logger.debug("[keywords] LLM extraction failed: %s, falling back to regex", e)

    # Fallback to regex extraction
    return extract_keywords(content, file_path)


@dataclass
class SearchResult:
    """A single search result with provenance tracking."""

    chunk_id: str
    repo: str
    file: str
    content: str
    score: float
    source: str = ""  # "wiki", "vector", "fts", "hybrid"
    vector_rank: Optional[int] = None
    fts_rank: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "repo": self.repo,
            "file": self.file,
            "text": self.content,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class WikiArticle:
    """A compiled wiki article."""

    slug: str
    title: str
    content: str
    repo: Optional[str] = None
    backlinks: List[str] = field(default_factory=list)
    path: Optional[str] = None


# ------------------------------------------------------------------ #
# Reciprocal Rank Fusion                                              #
# ------------------------------------------------------------------ #

RRF_K = 60  # Standard RRF constant (from the original paper)


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
) -> Dict[str, float]:
    """Compute RRF scores from multiple ranked lists of document IDs.

    Args:
        ranked_lists: List of ranked ID lists (best first).
        k: RRF constant (default 60).

    Returns:
        Dict mapping doc_id to RRF score, sorted descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


# ------------------------------------------------------------------ #
# Wiki lookup                                                         #
# ------------------------------------------------------------------ #


class WikiIndex:
    """Simple filesystem-based wiki article index.

    Scans a directory of markdown files and provides keyword lookup.

    Args:
        wiki_dir: Path to the compiled wiki directory.
    """

    def __init__(self, wiki_dir: str = "/archive/wiki") -> None:
        self.wiki_dir = Path(wiki_dir)
        self._articles: Dict[str, WikiArticle] = {}
        self._keywords: Dict[str, List[str]] = {}  # keyword -> [slug, ...]
        self._loaded = False

    def load(self) -> int:
        """Load all wiki articles from disk. Returns count."""
        if not self.wiki_dir.exists():
            logger.info("[wiki] directory %s does not exist", self.wiki_dir)
            return 0

        self._articles.clear()
        self._keywords.clear()

        for md_file in sorted(self.wiki_dir.glob("**/*.md")):
            slug = md_file.stem
            content = md_file.read_text(encoding="utf-8", errors="replace")

            # Extract title from first heading
            title = slug
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Extract backlinks [[slug]]
            import re

            backlinks = re.findall(r"\[\[([^\]]+)\]\]", content)

            article = WikiArticle(
                slug=slug,
                title=title,
                content=content,
                path=str(md_file),
                backlinks=backlinks,
            )
            self._articles[slug] = article

            # Index keywords from title and slug
            for word in slug.replace("-", " ").replace("_", " ").lower().split():
                if len(word) > 2:
                    self._keywords.setdefault(word, []).append(slug)
            for word in title.lower().split():
                if len(word) > 2:
                    self._keywords.setdefault(word, []).append(slug)

        self._loaded = True
        logger.info("[wiki] loaded %d articles from %s", len(self._articles), self.wiki_dir)
        return len(self._articles)

    def lookup(self, query: str, top_k: int = 3) -> List[WikiArticle]:
        """Find wiki articles matching a query.

        Uses simple keyword overlap scoring.

        Args:
            query: Search query.
            top_k: Max articles to return.

        Returns:
            Matched WikiArticle list, best first.
        """
        if not self._loaded:
            self.load()

        if not self._articles:
            return []

        query_words = set(query.lower().split())
        slug_scores: Dict[str, float] = {}

        for word in query_words:
            for slug in self._keywords.get(word, []):
                slug_scores[slug] = slug_scores.get(slug, 0.0) + 1.0

        if not slug_scores:
            return []

        ranked = sorted(slug_scores.items(), key=lambda x: -x[1])
        return [self._articles[slug] for slug, _ in ranked[:top_k] if slug in self._articles]

    @property
    def article_count(self) -> int:
        return len(self._articles)


# ------------------------------------------------------------------ #
# Hybrid Searcher                                                     #
# ------------------------------------------------------------------ #


class HybridSearcher:
    """3-tier search cascade: wiki → hybrid RRF → vector-only.

    Args:
        db_dsn: PostgreSQL connection string.
        wiki_dir: Path to compiled wiki directory.
        pca_components: PCA rotation matrix (1024, 384) or None.
        pca_mean: PCA mean vector (1024,) or None.
        hamming_index: Dict with binary_db, pca384_db, chunk_ids, chunk_data.
        vector_weight: Weight for vector results in RRF (default 1.0).
        fts_weight: Weight for FTS results in RRF (default 1.0).
    """

    def __init__(
        self,
        db_dsn: str = "dbname=atlas user=claude",
        wiki_dir: str = "/archive/wiki",
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
        hamming_index: Optional[dict] = None,
        vector_weight: float = 1.0,
        fts_weight: float = 1.0,
    ) -> None:
        self.db_dsn = db_dsn
        self.wiki = WikiIndex(wiki_dir)
        self.wiki.load()
        self._pca_components = pca_components
        self._pca_mean = pca_mean
        self._hamming = hamming_index
        self._vector_weight = vector_weight
        self._fts_weight = fts_weight

    def search(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
        top_k: int = 6,
        table: str = "chunks",
    ) -> List[SearchResult]:
        """Run the 3-tier search cascade.

        Args:
            query: Text query.
            embedding: Pre-computed 1024-dim embedding (optional).
            top_k: Number of results.
            table: DB table to search.

        Returns:
            List of SearchResult, best first.
        """
        results: List[SearchResult] = []

        # Tier 1: Wiki lookup
        wiki_results = self._search_wiki(query, top_k=2)
        if wiki_results:
            results.extend(wiki_results)
            logger.debug("[hybrid] wiki hit: %d articles", len(wiki_results))

        # Tier 2: Hybrid RRF (vector + FTS)
        remaining = top_k - len(results)
        if remaining > 0:
            # Get more candidates than needed for RRF fusion
            n_candidates = remaining * 5
            vector_ids = self._search_vector(query, embedding, n_candidates, table)
            fts_ids = self._search_fts(query, n_candidates, table)

            # RRF fusion
            seen_ids = {r.chunk_id for r in results}
            fused = self._fuse_rrf(vector_ids, fts_ids, remaining, seen_ids, table)
            results.extend(fused)

        return results[:top_k]

    def _search_wiki(self, query: str, top_k: int = 2) -> List[SearchResult]:
        """Tier 1: Wiki article lookup."""
        articles = self.wiki.lookup(query, top_k=top_k)
        results = []
        for art in articles:
            results.append(
                SearchResult(
                    chunk_id="wiki:" + art.slug,
                    repo=art.repo or "wiki",
                    file=art.slug + ".md",
                    content=art.content[:2000],  # truncate for context window
                    score=1.0,
                    source="wiki",
                )
            )
        return results

    def _search_vector(
        self,
        query: str,
        embedding: Optional[np.ndarray],
        top_k: int,
        table: str,
    ) -> List[str]:
        """Vector search returning ranked chunk IDs."""
        if self._hamming and table == "chunks":
            return self._search_hamming(embedding, top_k)

        if self._pca_components is not None and embedding is not None:
            return self._search_pgvector_pca(embedding, top_k, table)

        return []

    def _search_hamming(self, embedding: Optional[np.ndarray], top_k: int) -> List[str]:
        """GPU Hamming funnel on in-memory index."""
        if embedding is None or self._hamming is None:
            return []

        try:
            from turboquant_pro.cuda_search import gpu_hamming_search, pack_binary

            # PCA project
            centered = embedding.astype(np.float32) - self._pca_mean
            projected = centered @ self._pca_components
            norm = np.linalg.norm(projected)
            if norm > 1e-10:
                projected = projected / norm

            q_binary = (projected > 0).astype(np.uint8)
            q_packed = pack_binary(q_binary[np.newaxis, :])[0]

            coarse_idx, _ = gpu_hamming_search(
                q_packed, self._hamming["binary_db"], top_k=200
            )

            # PCA-384 rerank
            candidate_pca = self._hamming["pca384_db"][coarse_idx]
            scores = candidate_pca @ projected
            rerank_order = np.argsort(scores)[::-1][:top_k]

            chunk_ids = self._hamming["chunk_ids"]
            return [chunk_ids[coarse_idx[i]] for i in rerank_order]
        except Exception as e:
            logger.warning("[hybrid] hamming search failed: %s", e)
            return []

    def _search_pgvector_pca(
        self, embedding: np.ndarray, top_k: int, table: str
    ) -> List[str]:
        """pgvector IVFFlat on PCA-384."""
        centered = embedding.astype(np.float32) - self._pca_mean
        projected = centered @ self._pca_components
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected = projected / norm
        pca_str = str(projected.tolist())

        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()
            cur.execute("SET ivfflat.probes = 10")
            cur.execute(
                "SELECT id FROM %s ORDER BY embedding_pca384 <=> %%s::vector LIMIT %%s"
                % table,
                (pca_str, top_k),
            )
            ids = [r[0] for r in cur.fetchall()]
            conn.close()
            return ids
        except Exception as e:
            logger.warning("[hybrid] pgvector search failed: %s", e)
            return []

    def _search_fts(self, query: str, top_k: int, table: str = "chunks") -> List[str]:
        """Full-text search via tsvector."""
        if psycopg2 is None:
            return []

        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM %s "
                "WHERE tsv @@ plainto_tsquery('english', %%s) "
                "ORDER BY ts_rank(tsv, plainto_tsquery('english', %%s)) DESC "
                "LIMIT %%s" % table,
                (query, query, top_k),
            )
            ids = [r[0] for r in cur.fetchall()]
            conn.close()
            return ids
        except Exception as e:
            logger.warning("[hybrid] FTS search failed: %s", e)
            return []

    def _fuse_rrf(
        self,
        vector_ids: List[str],
        fts_ids: List[str],
        top_k: int,
        seen_ids: set,
        table: str,
    ) -> List[SearchResult]:
        """Fuse vector and FTS results with RRF, fetch content."""
        # Weight the ranked lists
        weighted_lists = []
        for _ in range(max(1, int(self._vector_weight))):
            weighted_lists.append(vector_ids)
        for _ in range(max(1, int(self._fts_weight))):
            weighted_lists.append(fts_ids)

        rrf_scores = reciprocal_rank_fusion(weighted_lists)

        # Filter already-seen IDs
        ranked_ids = [
            (doc_id, score)
            for doc_id, score in rrf_scores.items()
            if doc_id not in seen_ids
        ][:top_k]

        if not ranked_ids:
            return []

        # Fetch content from DB
        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()
            id_list = [r[0] for r in ranked_ids]
            placeholders = ",".join(["%s"] * len(id_list))
            cur.execute(
                "SELECT id, repo, file_path, content FROM %s WHERE id IN (%s)"
                % (table, placeholders),
                id_list,
            )
            rows = {r[0]: r for r in cur.fetchall()}
            conn.close()
        except Exception as e:
            logger.warning("[hybrid] content fetch failed: %s", e)
            return []

        # Build results with provenance
        results = []
        vector_set = set(vector_ids)
        fts_set = set(fts_ids)
        for doc_id, rrf_score in ranked_ids:
            if doc_id not in rows:
                continue
            row = rows[doc_id]
            in_vector = doc_id in vector_set
            in_fts = doc_id in fts_set
            if in_vector and in_fts:
                source = "hybrid"
            elif in_vector:
                source = "vector"
            else:
                source = "fts"

            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    repo=row[1],
                    file=row[2],
                    content=row[3],
                    score=rrf_score,
                    source=source,
                    vector_rank=vector_ids.index(doc_id) if in_vector else None,
                    fts_rank=fts_ids.index(doc_id) if in_fts else None,
                )
            )

        return results

    def stats(self) -> dict:
        """Return search system statistics."""
        return {
            "wiki_articles": self.wiki.article_count,
            "hamming_ready": self._hamming is not None,
            "pca_ready": self._pca_components is not None,
            "vector_weight": self._vector_weight,
            "fts_weight": self._fts_weight,
        }
