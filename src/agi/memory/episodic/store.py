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
Episodic Memory Store for AGI-HPC Phase 2.

PostgreSQL-backed conversation history with pgvector semantic search.
Stores episodes (user/response pairs) with session tracking, hemisphere
routing metadata, and optional embedding vectors for similarity recall.

Schema:
    CREATE TABLE episodes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id TEXT NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        user_message TEXT,
        atlas_response TEXT,
        hemisphere TEXT,          -- 'lh', 'rh', 'both'
        safety_flags JSONB DEFAULT '{}',
        quality_score FLOAT,
        metadata JSONB DEFAULT '{}',
        embedding vector(1024)
    );
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """A single conversation episode.

    Attributes:
        id: Unique episode identifier (UUID).
        session_id: Conversation session identifier.
        timestamp: UTC creation time.
        user_message: The user's input.
        atlas_response: Atlas system response.
        hemisphere: Which hemisphere handled the request ('lh', 'rh', 'both').
        safety_flags: Safety check results as JSON.
        quality_score: Quality assessment score (0.0 - 1.0).
        metadata: Arbitrary extra metadata.
        embedding: Optional 1024-dim embedding vector.
    """

    id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_message: str = ""
    atlas_response: str = ""
    hemisphere: str = "lh"
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else str(self.timestamp)
            ),
            "user_message": self.user_message,
            "atlas_response": self.atlas_response,
            "hemisphere": self.hemisphere,
            "safety_flags": self.safety_flags,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EpisodicMemoryConfig:
    """Configuration for the episodic memory store.

    Attributes:
        db_dsn: PostgreSQL connection string.
        table_name: Name of the episodes table.
        auto_create_table: Whether to create the table on init.
    """

    db_dsn: str = "dbname=atlas user=claude"
    table_name: str = "episodes"
    auto_create_table: bool = True


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_message TEXT,
    atlas_response TEXT,
    hemisphere TEXT,
    safety_flags JSONB DEFAULT '{}',
    quality_score FLOAT,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_id
    ON episodes (session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
    ON episodes (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_embedding
    ON episodes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
"""


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """PostgreSQL-backed episodic memory store.

    Stores conversation episodes and supports recall by recency,
    session, or semantic similarity (via pgvector embeddings).

    Usage::

        memory = EpisodicMemory()
        memory.ensure_table()

        episode_id = memory.store_episode(
            session_id="sess-123",
            user_msg="How does Paxos work?",
            response="Paxos is a consensus algorithm...",
            hemisphere="lh",
        )
        recent = memory.recall_recent("sess-123", n=5)
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None) -> None:
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required but not installed. "
                "Install with: pip install psycopg2-binary"
            )
        self._config = config or EpisodicMemoryConfig()
        if self._config.auto_create_table:
            self.ensure_table()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create the episodes table and indexes if they do not exist."""
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                # Ensure pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(CREATE_TABLE_SQL)
            conn.close()
            logger.info(
                "[episodic] ensured table '%s' exists",
                self._config.table_name,
            )
        except Exception:
            logger.exception("[episodic] failed to create table")
            raise

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_episode(
        self,
        session_id: str,
        user_msg: str,
        response: str,
        hemisphere: str = "lh",
        metadata: Optional[Dict[str, Any]] = None,
        safety_flags: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.0,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Store a conversation episode.

        Args:
            session_id: Conversation session identifier.
            user_msg: The user's input message.
            response: Atlas system response.
            hemisphere: Which hemisphere handled it ('lh', 'rh', 'both').
            metadata: Optional extra metadata dict.
            safety_flags: Optional safety check results.
            quality_score: Quality assessment (0.0 - 1.0).
            embedding: Optional 1024-dim embedding vector.

        Returns:
            The UUID of the stored episode.
        """
        episode_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {})
        safety_json = json.dumps(safety_flags or {})
        emb_str = str(embedding) if embedding else None

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO episodes
                        (id, session_id, user_message, atlas_response,
                         hemisphere, safety_flags, quality_score,
                         metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        session_id,
                        user_msg,
                        response,
                        hemisphere,
                        safety_json,
                        quality_score,
                        meta_json,
                        emb_str,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("[episodic] failed to store episode")
            raise

        logger.debug(
            "[episodic] stored episode id=%s session=%s hemisphere=%s",
            episode_id[:8],
            session_id[:8],
            hemisphere,
        )
        return episode_id

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall_recent(self, session_id: str, n: int = 10) -> List[Episode]:
        """Recall the N most recent episodes for a session.

        Args:
            session_id: Conversation session identifier.
            n: Maximum number of episodes to return.

        Returns:
            List of Episode objects ordered by timestamp descending.
        """
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, session_id, timestamp, user_message,
                           atlas_response, hemisphere, safety_flags,
                           quality_score, metadata
                    FROM episodes
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (session_id, n),
                )
                rows = cur.fetchall()
            conn.close()
        except Exception:
            logger.exception("[episodic] recall_recent failed")
            return []

        return [self._row_to_episode(row) for row in rows]

    def recall_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Episode]:
        """Recall episodes similar to a query embedding.

        Uses pgvector cosine distance for semantic search across all
        stored episodes that have an embedding.

        Args:
            query_embedding: 1024-dim query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of Episode objects ordered by similarity descending.
        """
        emb_str = str(query_embedding)

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, session_id, timestamp, user_message,
                           atlas_response, hemisphere, safety_flags,
                           quality_score, metadata
                    FROM episodes
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (emb_str, top_k),
                )
                rows = cur.fetchall()
            conn.close()
        except Exception:
            logger.exception("[episodic] recall_similar failed")
            return []

        return [self._row_to_episode(row) for row in rows]

    def get_session_history(self, session_id: str) -> List[Episode]:
        """Retrieve full conversation history for a session.

        Args:
            session_id: Conversation session identifier.

        Returns:
            All episodes for the session, ordered by timestamp ascending.
        """
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, session_id, timestamp, user_message,
                           atlas_response, hemisphere, safety_flags,
                           quality_score, metadata
                    FROM episodes
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
            conn.close()
        except Exception:
            logger.exception("[episodic] get_session_history failed")
            return []

        return [self._row_to_episode(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: tuple) -> Episode:
        """Convert a database row to an Episode dataclass."""
        safety = row[6] if row[6] else {}
        if isinstance(safety, str):
            safety = json.loads(safety)
        meta = row[8] if row[8] else {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Episode(
            id=str(row[0]),
            session_id=row[1],
            timestamp=row[2],
            user_message=row[3] or "",
            atlas_response=row[4] or "",
            hemisphere=row[5] or "lh",
            safety_flags=safety,
            quality_score=float(row[7]) if row[7] is not None else 0.0,
            metadata=meta,
        )
