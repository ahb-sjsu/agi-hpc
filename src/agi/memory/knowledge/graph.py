# AGI-HPC Knowledge Graph
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Knowledge graph storage for entities and relationships.

Uses PostgreSQL for persistence with optional SQLite fallback for testing.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from agi.memory.knowledge.extractor import (
    Entity,
    ExtractedKnowledge,
    Relationship,
)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the knowledge graph."""

    db_dsn: str = "dbname=atlas user=claude"
    table_prefix: str = "knowledge"
    use_sqlite: bool = False
    sqlite_path: str = ":memory:"


class KnowledgeGraph:
    """Store and query a graph of entities and relationships."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None) -> None:
        self._config = config or KnowledgeGraphConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._pg_conn = None

        if self._config.use_sqlite:
            self._init_sqlite()
        else:
            self._init_postgres()

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(self._config.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                source_doc TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entity_name
                ON knowledge_entities(name);

            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_doc TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rel_subject
                ON knowledge_relationships(subject);
            CREATE INDEX IF NOT EXISTS idx_rel_object
                ON knowledge_relationships(object);

            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id TEXT PRIMARY KEY,
                source_path TEXT,
                summary TEXT,
                key_concepts TEXT,
                entity_count INTEGER,
                relationship_count INTEGER,
                ingested_at TEXT
            );
            """)

    def _init_postgres(self) -> None:
        try:
            import psycopg2

            self._pg_conn = psycopg2.connect(self._config.db_dsn)
            self._pg_conn.autocommit = True
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_entities (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name TEXT NOT NULL,
                        entity_type TEXT,
                        description TEXT,
                        source_doc TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_entity_name
                        ON knowledge_entities(name);

                    CREATE TABLE IF NOT EXISTS knowledge_relationships (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        confidence FLOAT DEFAULT 1.0,
                        source_doc TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_rel_subject
                        ON knowledge_relationships(subject);
                    CREATE INDEX IF NOT EXISTS idx_rel_object
                        ON knowledge_relationships(object);

                    CREATE TABLE IF NOT EXISTS knowledge_documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_path TEXT,
                        summary TEXT,
                        key_concepts TEXT[],
                        entity_count INT,
                        relationship_count INT,
                        ingested_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """)
        except ImportError:
            logger.warning("psycopg2 not available, falling back to SQLite")
            self._config.use_sqlite = True
            self._init_sqlite()

    def store(self, knowledge: ExtractedKnowledge) -> str:
        """Store extracted knowledge. Returns document ID."""
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if self._config.use_sqlite:
            return self._store_sqlite(knowledge, doc_id, now)
        return self._store_postgres(knowledge, doc_id)

    def _store_sqlite(
        self, knowledge: ExtractedKnowledge, doc_id: str, now: str
    ) -> str:
        assert self._conn is not None
        cur = self._conn.cursor()

        for e in knowledge.entities:
            cur.execute(
                "INSERT OR IGNORE INTO knowledge_entities "
                "(id, name, entity_type, description, source_doc, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    e.name,
                    e.entity_type,
                    e.description,
                    e.source_doc,
                    now,
                ),
            )

        for r in knowledge.relationships:
            cur.execute(
                "INSERT INTO knowledge_relationships "
                "(id, subject, predicate, object, confidence, source_doc, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    r.subject,
                    r.predicate,
                    r.object,
                    r.confidence,
                    r.source_doc,
                    now,
                ),
            )

        import json

        cur.execute(
            "INSERT INTO knowledge_documents "
            "(id, source_path, summary, key_concepts, entity_count, "
            "relationship_count, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                knowledge.source_path,
                knowledge.summary,
                json.dumps(knowledge.key_concepts),
                len(knowledge.entities),
                len(knowledge.relationships),
                now,
            ),
        )
        self._conn.commit()
        logger.info(
            "Stored knowledge doc %s: %d entities, %d relationships",
            doc_id,
            len(knowledge.entities),
            len(knowledge.relationships),
        )
        return doc_id

    def _store_postgres(self, knowledge: ExtractedKnowledge, doc_id: str) -> str:
        assert self._pg_conn is not None
        with self._pg_conn.cursor() as cur:
            for e in knowledge.entities:
                cur.execute(
                    "INSERT INTO knowledge_entities "
                    "(name, entity_type, description, source_doc) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (e.name, e.entity_type, e.description, e.source_doc),
                )

            for r in knowledge.relationships:
                cur.execute(
                    "INSERT INTO knowledge_relationships "
                    "(subject, predicate, object, confidence, source_doc) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (r.subject, r.predicate, r.object, r.confidence, r.source_doc),
                )

            cur.execute(
                "INSERT INTO knowledge_documents "
                "(id, source_path, summary, key_concepts, entity_count, "
                "relationship_count) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    doc_id,
                    knowledge.source_path,
                    knowledge.summary,
                    knowledge.key_concepts,
                    len(knowledge.entities),
                    len(knowledge.relationships),
                ),
            )
        return doc_id

    def query_entity(self, name: str) -> Optional[Entity]:
        """Look up an entity by name."""
        if self._config.use_sqlite:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT name, entity_type, description, source_doc "
                "FROM knowledge_entities WHERE name = ?",
                (name,),
            ).fetchone()
            if row:
                return Entity(
                    name=row["name"],
                    entity_type=row["entity_type"],
                    description=row["description"],
                    source_doc=row["source_doc"],
                )
            return None

        assert self._pg_conn is not None
        with self._pg_conn.cursor() as cur:
            cur.execute(
                "SELECT name, entity_type, description, source_doc "
                "FROM knowledge_entities WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return Entity(
                    name=row[0],
                    entity_type=row[1],
                    description=row[2],
                    source_doc=row[3],
                )
        return None

    def query_relationships(
        self, entity: str, predicate: Optional[str] = None
    ) -> list[Relationship]:
        """Find relationships involving an entity."""
        if self._config.use_sqlite:
            assert self._conn is not None
            if predicate:
                rows = self._conn.execute(
                    "SELECT subject, predicate, object, confidence, source_doc "
                    "FROM knowledge_relationships "
                    "WHERE (subject = ? OR object = ?) AND predicate = ?",
                    (entity, entity, predicate),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT subject, predicate, object, confidence, source_doc "
                    "FROM knowledge_relationships "
                    "WHERE subject = ? OR object = ?",
                    (entity, entity),
                ).fetchall()
        else:
            assert self._pg_conn is not None
            with self._pg_conn.cursor() as cur:
                if predicate:
                    cur.execute(
                        "SELECT subject, predicate, object, confidence, source_doc "
                        "FROM knowledge_relationships "
                        "WHERE (subject = %s OR object = %s) AND predicate = %s",
                        (entity, entity, predicate),
                    )
                else:
                    cur.execute(
                        "SELECT subject, predicate, object, confidence, source_doc "
                        "FROM knowledge_relationships "
                        "WHERE subject = %s OR object = %s",
                        (entity, entity),
                    )
                rows = cur.fetchall()

        return [
            Relationship(
                subject=r[0] if isinstance(r, tuple) else r["subject"],
                predicate=r[1] if isinstance(r, tuple) else r["predicate"],
                object=r[2] if isinstance(r, tuple) else r["object"],
                confidence=r[3] if isinstance(r, tuple) else r["confidence"],
                source_doc=r[4] if isinstance(r, tuple) else r["source_doc"],
            )
            for r in rows
        ]

    def get_stats(self) -> dict[str, int]:
        """Return counts of entities, relationships, and documents."""
        if self._config.use_sqlite:
            assert self._conn is not None
            entities = self._conn.execute(
                "SELECT COUNT(*) FROM knowledge_entities"
            ).fetchone()[0]
            rels = self._conn.execute(
                "SELECT COUNT(*) FROM knowledge_relationships"
            ).fetchone()[0]
            docs = self._conn.execute(
                "SELECT COUNT(*) FROM knowledge_documents"
            ).fetchone()[0]
        else:
            assert self._pg_conn is not None
            with self._pg_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM knowledge_entities")
                entities = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM knowledge_relationships")
                rels = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM knowledge_documents")
                docs = cur.fetchone()[0]

        return {"entities": entities, "relationships": rels, "documents": docs}
