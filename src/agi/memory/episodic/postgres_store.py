# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
PostgreSQL storage backend for episodic memory.

Provides persistent episode storage with:
- Temporal indexing for range queries
- JSONB for flexible metadata
- Decision Proof hash chain storage
- Episode similarity search with embeddings

Environment Variables:
    POSTGRES_URL        PostgreSQL connection URL
    POSTGRES_HOST       PostgreSQL host (default: localhost)
    POSTGRES_PORT       PostgreSQL port (default: 5432)
    POSTGRES_DB         Database name (default: agi_memory)
    POSTGRES_USER       Username (default: postgres)
    POSTGRES_PASSWORD   Password
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:
    psycopg2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL storage."""

    host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    database: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "agi_memory")
    )
    user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres")
    )
    password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "")
    )

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """An episode record."""

    episode_id: str
    task_description: str
    task_type: str = ""
    scenario_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    outcome_success: Optional[bool] = None
    outcome_description: str = ""
    completion_percentage: float = 0.0
    total_duration_ms: int = 0
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def from_row(cls, row: Dict) -> "Episode":
        """Create from database row."""
        return cls(
            episode_id=str(row.get("episode_id", "")),
            task_description=row.get("task_description", ""),
            task_type=row.get("task_type", ""),
            scenario_id=row.get("scenario_id", ""),
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            outcome_success=row.get("outcome_success"),
            outcome_description=row.get("outcome_description", ""),
            completion_percentage=row.get("completion_percentage", 0.0),
            total_duration_ms=row.get("total_duration_ms", 0),
            insights=row.get("insights", []),
            metadata=row.get("metadata", {}),
            embedding=row.get("embedding"),
        )


@dataclass
class EpisodeStep:
    """A step within an episode."""

    episode_id: str
    step_index: int
    step_id: str = ""
    description: str = ""
    tool_id: str = ""
    succeeded: Optional[bool] = None
    failure_reason: str = ""
    duration_ms: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEvent:
    """An event within an episode."""

    episode_id: str
    step_index: int
    event_type: str
    timestamp_ms: int
    payload: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionProof:
    """A decision proof for governance audit trail."""

    proof_id: str
    episode_id: str
    step_id: str
    timestamp_ms: int
    decision: str  # ALLOW, BLOCK, REVISE
    bond_index: float = 0.0
    moral_vector: Dict[str, float] = field(default_factory=dict)
    previous_proof_hash: str = ""
    proof_hash: str = ""
    signature: Optional[bytes] = None

    def compute_hash(self) -> str:
        """Compute hash for this proof."""
        content = json.dumps({
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "timestamp_ms": self.timestamp_ms,
            "decision": self.decision,
            "bond_index": self.bond_index,
            "moral_vector": self.moral_vector,
            "previous_proof_hash": self.previous_proof_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# PostgreSQL Store
# ---------------------------------------------------------------------------


class PostgresEpisodicStore:
    """
    PostgreSQL-backed episodic memory store.

    Features:
    - Episodes with steps and events
    - Temporal range queries
    - Decision proof chain
    - Similarity search with embeddings
    """

    SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS episodes (
        episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        task_description TEXT NOT NULL,
        task_type VARCHAR(100),
        scenario_id VARCHAR(255),
        start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        end_time TIMESTAMPTZ,
        outcome_success BOOLEAN,
        outcome_description TEXT,
        completion_percentage FLOAT DEFAULT 0,
        total_duration_ms BIGINT DEFAULT 0,
        insights TEXT[],
        metadata JSONB DEFAULT '{}',
        embedding VECTOR(768),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS episode_steps (
        id SERIAL PRIMARY KEY,
        episode_id UUID REFERENCES episodes(episode_id) ON DELETE CASCADE,
        step_index INT NOT NULL,
        step_id VARCHAR(255),
        description TEXT,
        tool_id VARCHAR(255),
        succeeded BOOLEAN,
        failure_reason TEXT,
        duration_ms BIGINT DEFAULT 0,
        params JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS episode_events (
        id SERIAL PRIMARY KEY,
        episode_id UUID REFERENCES episodes(episode_id) ON DELETE CASCADE,
        step_index INT,
        event_type VARCHAR(100) NOT NULL,
        timestamp_ms BIGINT NOT NULL,
        payload JSONB DEFAULT '{}',
        tags JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS decision_proofs (
        proof_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        episode_id UUID REFERENCES episodes(episode_id),
        step_id VARCHAR(255),
        timestamp_ms BIGINT NOT NULL,
        decision VARCHAR(50) NOT NULL,
        bond_index FLOAT,
        moral_vector JSONB,
        previous_proof_hash VARCHAR(64),
        proof_hash VARCHAR(64) NOT NULL,
        signature BYTEA,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_time, end_time);
    CREATE INDEX IF NOT EXISTS idx_episodes_task_type ON episodes(task_type);
    CREATE INDEX IF NOT EXISTS idx_episode_steps_episode ON episode_steps(episode_id);
    CREATE INDEX IF NOT EXISTS idx_episode_events_episode ON episode_events(episode_id);
    CREATE INDEX IF NOT EXISTS idx_decision_proofs_episode ON decision_proofs(episode_id);
    CREATE INDEX IF NOT EXISTS idx_decision_proofs_hash ON decision_proofs(proof_hash);
    """

    def __init__(self, config: Optional[PostgresConfig] = None):
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )

        self.config = config or PostgresConfig()
        self._conn = None
        self._connected = False

        logger.info(
            "[memory][episodic][postgres] initialized host=%s db=%s",
            self.config.host,
            self.config.database,
        )

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        if self._connected:
            return

        self._conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )
        self._connected = True

        logger.info(
            "[memory][episodic][postgres] connected to %s",
            self.config.host,
        )

    def init_schema(self) -> None:
        """Initialize database schema."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute(self.SCHEMA)
        self._conn.commit()

        logger.info("[memory][episodic][postgres] schema initialized")

    def store_episode(self, episode: Episode) -> str:
        """Store an episode."""
        if not self._connected:
            self.connect()

        episode_id = episode.episode_id or str(uuid.uuid4())

        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episodes (
                    episode_id, task_description, task_type, scenario_id,
                    start_time, end_time, outcome_success, outcome_description,
                    completion_percentage, total_duration_ms, insights, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (episode_id) DO UPDATE SET
                    task_description = EXCLUDED.task_description,
                    end_time = EXCLUDED.end_time,
                    outcome_success = EXCLUDED.outcome_success,
                    outcome_description = EXCLUDED.outcome_description,
                    completion_percentage = EXCLUDED.completion_percentage,
                    total_duration_ms = EXCLUDED.total_duration_ms,
                    insights = EXCLUDED.insights,
                    metadata = EXCLUDED.metadata
            """, (
                episode_id,
                episode.task_description,
                episode.task_type,
                episode.scenario_id,
                episode.start_time or datetime.utcnow(),
                episode.end_time,
                episode.outcome_success,
                episode.outcome_description,
                episode.completion_percentage,
                episode.total_duration_ms,
                episode.insights,
                Json(episode.metadata),
            ))
        self._conn.commit()

        logger.debug("[memory][episodic][postgres] stored episode=%s", episode_id)
        return episode_id

    def store_step(self, step: EpisodeStep) -> None:
        """Store an episode step."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episode_steps (
                    episode_id, step_index, step_id, description,
                    tool_id, succeeded, failure_reason, duration_ms, params
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                step.episode_id,
                step.step_index,
                step.step_id,
                step.description,
                step.tool_id,
                step.succeeded,
                step.failure_reason,
                step.duration_ms,
                Json(step.params),
            ))
        self._conn.commit()

    def store_event(self, event: EpisodeEvent) -> None:
        """Store an episode event."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episode_events (
                    episode_id, step_index, event_type,
                    timestamp_ms, payload, tags
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                event.episode_id,
                event.step_index,
                event.event_type,
                event.timestamp_ms,
                Json(event.payload),
                Json(event.tags),
            ))
        self._conn.commit()

    def store_decision_proof(self, proof: DecisionProof) -> str:
        """Store a decision proof."""
        if not self._connected:
            self.connect()

        proof_id = proof.proof_id or str(uuid.uuid4())
        proof_hash = proof.proof_hash or proof.compute_hash()

        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO decision_proofs (
                    proof_id, episode_id, step_id, timestamp_ms,
                    decision, bond_index, moral_vector,
                    previous_proof_hash, proof_hash, signature
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                proof_id,
                proof.episode_id,
                proof.step_id,
                proof.timestamp_ms,
                proof.decision,
                proof.bond_index,
                Json(proof.moral_vector),
                proof.previous_proof_hash,
                proof_hash,
                proof.signature,
            ))
        self._conn.commit()

        logger.debug(
            "[memory][episodic][postgres] stored proof=%s hash=%s",
            proof_id,
            proof_hash[:16],
        )
        return proof_id

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM episodes WHERE episode_id = %s
            """, (episode_id,))
            row = cur.fetchone()
            if not row:
                return None

            columns = [desc[0] for desc in cur.description]
            return Episode.from_row(dict(zip(columns, row)))

    def query_episodes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        task_type: Optional[str] = None,
        scenario_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """Query episodes with filters."""
        if not self._connected:
            self.connect()

        conditions = []
        params = []

        if start_time:
            conditions.append("start_time >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("start_time <= %s")
            params.append(end_time)
        if task_type:
            conditions.append("task_type = %s")
            params.append(task_type)
        if scenario_id:
            conditions.append("scenario_id = %s")
            params.append(scenario_id)
        if success is not None:
            conditions.append("outcome_success = %s")
            params.append(success)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM episodes
                WHERE {where_clause}
                ORDER BY start_time DESC
                LIMIT %s
            """, params)

            columns = [desc[0] for desc in cur.description]
            return [
                Episode.from_row(dict(zip(columns, row)))
                for row in cur.fetchall()
            ]

    def get_episode_steps(self, episode_id: str) -> List[EpisodeStep]:
        """Get steps for an episode."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM episode_steps
                WHERE episode_id = %s
                ORDER BY step_index
            """, (episode_id,))

            columns = [desc[0] for desc in cur.description]
            return [
                EpisodeStep(
                    episode_id=str(row[columns.index("episode_id")]),
                    step_index=row[columns.index("step_index")],
                    step_id=row[columns.index("step_id")] or "",
                    description=row[columns.index("description")] or "",
                    tool_id=row[columns.index("tool_id")] or "",
                    succeeded=row[columns.index("succeeded")],
                    failure_reason=row[columns.index("failure_reason")] or "",
                    duration_ms=row[columns.index("duration_ms")] or 0,
                    params=row[columns.index("params")] or {},
                )
                for row in cur.fetchall()
            ]

    def get_episode_events(self, episode_id: str) -> List[EpisodeEvent]:
        """Get events for an episode."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM episode_events
                WHERE episode_id = %s
                ORDER BY timestamp_ms
            """, (episode_id,))

            columns = [desc[0] for desc in cur.description]
            return [
                EpisodeEvent(
                    episode_id=str(row[columns.index("episode_id")]),
                    step_index=row[columns.index("step_index")] or 0,
                    event_type=row[columns.index("event_type")],
                    timestamp_ms=row[columns.index("timestamp_ms")],
                    payload=row[columns.index("payload")] or {},
                    tags=row[columns.index("tags")] or {},
                )
                for row in cur.fetchall()
            ]

    def verify_proof_chain(self, episode_id: str) -> bool:
        """Verify decision proof chain integrity."""
        if not self._connected:
            self.connect()

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM decision_proofs
                WHERE episode_id = %s
                ORDER BY timestamp_ms
            """, (episode_id,))

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        if not rows:
            return True

        prev_hash = ""
        for row in rows:
            data = dict(zip(columns, row))

            # Verify previous hash matches
            if data["previous_proof_hash"] != prev_hash:
                logger.warning(
                    "[memory][episodic] proof chain broken at %s",
                    data["proof_id"],
                )
                return False

            # Verify computed hash
            proof = DecisionProof(
                proof_id=str(data["proof_id"]),
                episode_id=str(data["episode_id"]),
                step_id=data["step_id"] or "",
                timestamp_ms=data["timestamp_ms"],
                decision=data["decision"],
                bond_index=data["bond_index"] or 0.0,
                moral_vector=data["moral_vector"] or {},
                previous_proof_hash=data["previous_proof_hash"] or "",
            )
            computed_hash = proof.compute_hash()

            if computed_hash != data["proof_hash"]:
                logger.warning(
                    "[memory][episodic] proof hash mismatch at %s",
                    data["proof_id"],
                )
                return False

            prev_hash = data["proof_hash"]

        return True

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._connected = False
        logger.info("[memory][episodic][postgres] closed")
