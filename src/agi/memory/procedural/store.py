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
Procedural Memory Store for AGI-HPC Phase 2.

SQLite-backed store of learned behaviours and procedures.
Each procedure has a trigger pattern, ordered steps, and
success/failure tracking for reinforcement-style learning.

Schema:
    procedures(
        name TEXT PRIMARY KEY,
        trigger_pattern TEXT NOT NULL,
        procedure_steps TEXT NOT NULL,    -- JSON array of step strings
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        last_used TEXT,                   -- ISO 8601 timestamp
        metadata TEXT DEFAULT '{}'        -- JSON object
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class Procedure:
    """A learned procedure (behaviour pattern).

    Attributes:
        name: Unique procedure name/identifier.
        trigger_pattern: Pattern that activates this procedure.
        procedure_steps: Ordered list of step descriptions.
        success_count: Number of successful executions.
        failure_count: Number of failed executions.
        last_used: ISO 8601 timestamp of last use.
        metadata: Extra metadata dict.
    """

    name: str = ""
    trigger_pattern: str = ""
    procedure_steps: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Return the success rate as a float between 0 and 1."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "name": self.name,
            "trigger_pattern": self.trigger_pattern,
            "procedure_steps": self.procedure_steps,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used": self.last_used,
            "metadata": self.metadata,
            "success_rate": self.success_rate,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProceduralMemoryConfig:
    """Configuration for the procedural memory store.

    Attributes:
        db_path: Path to the SQLite database file.
        auto_create: Whether to create schema on init.
        seed_procedures: Whether to seed with built-in procedures.
    """

    db_path: str = "/home/claude/agi-hpc/data/procedural.db"
    auto_create: bool = True
    seed_procedures: bool = True


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS procedures (
    name TEXT PRIMARY KEY,
    trigger_pattern TEXT NOT NULL,
    procedure_steps TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_used TEXT,
    metadata TEXT DEFAULT '{}'
);
"""

# ---------------------------------------------------------------------------
# Built-in seed procedures
# ---------------------------------------------------------------------------

SEED_PROCEDURES = [
    {
        "name": "repo_search",
        "trigger_pattern": "user asks about a repo|repository|codebase|source code",
        "procedure_steps": [
            "Extract repository name or topic from user query",
            "Search semantic memory with repo filter",
            "Rank results by relevance score",
            "Format top results as context for LLM",
            "Generate response with RAG context",
        ],
        "metadata": {"category": "retrieval", "priority": "high"},
    },
    {
        "name": "error_diagnosis",
        "trigger_pattern": "user reports error|exception|traceback|bug|crash",
        "procedure_steps": [
            "Extract error type and message from user input",
            "Search semantic memory for similar errors",
            "Search episodic memory for past resolutions",
            "Route to LH for analytical diagnosis",
            "Suggest fix with confidence score",
        ],
        "metadata": {"category": "debugging", "priority": "high"},
    },
    {
        "name": "creative_exploration",
        "trigger_pattern": "brainstorm|imagine|what if|design|explore possibilities",
        "procedure_steps": [
            "Detect creative intent from keywords",
            "Route to RH hemisphere for divergent thinking",
            "Retrieve related concepts from semantic memory",
            "Generate multiple perspectives",
            "Synthesize into coherent exploration",
        ],
        "metadata": {"category": "creative", "priority": "medium"},
    },
    {
        "name": "session_context_recall",
        "trigger_pattern": "earlier|before|we discussed|you said|last time",
        "procedure_steps": [
            "Extract temporal reference from user query",
            "Query episodic memory for session history",
            "Retrieve relevant prior exchanges",
            "Inject context into current prompt",
            "Generate continuity-aware response",
        ],
        "metadata": {"category": "context", "priority": "high"},
    },
    {
        "name": "comparison_analysis",
        "trigger_pattern": "compare|difference|versus|vs|trade-off|pros and cons",
        "procedure_steps": [
            "Extract entities to compare from user query",
            "Search semantic memory for each entity",
            "Route to LH for structured comparison",
            "Generate comparison matrix or table",
            "Summarize key trade-offs",
        ],
        "metadata": {"category": "analysis", "priority": "medium"},
    },
]


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------


class ProceduralMemory:
    """SQLite-backed procedural memory store.

    Stores learned behaviours as trigger-pattern / procedure-steps pairs,
    with success/failure tracking for reinforcement-style learning.

    Usage::

        memory = ProceduralMemory()
        memory.store_procedure(
            name="repo_search",
            trigger="user asks about a repository",
            steps=["search semantic memory", "format results"],
        )
        matches = memory.lookup("Tell me about the agi-hpc repo")
        memory.record_outcome("repo_search", success=True)
    """

    def __init__(self, config: Optional[ProceduralMemoryConfig] = None) -> None:
        self._config = config or ProceduralMemoryConfig()
        self._db_path = self._config.db_path

        # Ensure parent directory exists
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        if self._config.auto_create:
            self._ensure_schema()
        if self._config.seed_procedures:
            self._seed_initial_procedures()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the procedures table if it does not exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
        conn.close()
        logger.info("[procedural] ensured schema at %s", self._db_path)

    def _seed_initial_procedures(self) -> None:
        """Insert built-in seed procedures (skip if already present)."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        seeded = 0
        for proc in SEED_PROCEDURES:
            cursor.execute("SELECT 1 FROM procedures WHERE name = ?", (proc["name"],))
            if cursor.fetchone() is None:
                cursor.execute(
                    """
                    INSERT INTO procedures
                        (name, trigger_pattern, procedure_steps, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        proc["name"],
                        proc["trigger_pattern"],
                        json.dumps(proc["procedure_steps"]),
                        json.dumps(proc.get("metadata", {})),
                    ),
                )
                seeded += 1
        conn.commit()
        conn.close()
        if seeded:
            logger.info("[procedural] seeded %d built-in procedures", seeded)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_procedure(
        self,
        name: str,
        trigger: str,
        steps: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store or update a procedure.

        Args:
            name: Unique procedure name/identifier.
            trigger: Trigger pattern (pipe-separated keywords/phrases).
            steps: Ordered list of procedure step descriptions.
            metadata: Optional extra metadata.
        """
        meta_json = json.dumps(metadata or {})
        steps_json = json.dumps(steps)

        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT INTO procedures
                (name, trigger_pattern, procedure_steps, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                trigger_pattern = excluded.trigger_pattern,
                procedure_steps = excluded.procedure_steps,
                metadata = excluded.metadata
            """,
            (name, trigger, steps_json, meta_json),
        )
        conn.commit()
        conn.close()

        logger.debug(
            "[procedural] stored procedure name=%s trigger=%r steps=%d",
            name,
            trigger[:50],
            len(steps),
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, trigger_text: str) -> List[Procedure]:
        """Find procedures whose trigger pattern matches the input text.

        Performs case-insensitive substring matching against each
        pipe-separated trigger phrase. Results are ordered by match
        count (best matches first), then by success rate.

        Args:
            trigger_text: Input text to match against trigger patterns.

        Returns:
            List of matching Procedure objects.
        """
        lower_text = trigger_text.lower()

        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("""
            SELECT name, trigger_pattern, procedure_steps,
                   success_count, failure_count, last_used, metadata
            FROM procedures
            """)
        rows = cursor.fetchall()
        conn.close()

        scored: List[tuple[int, float, Procedure]] = []
        for row in rows:
            proc = self._row_to_procedure(row)
            triggers = [t.strip().lower() for t in proc.trigger_pattern.split("|")]
            match_count = sum(1 for t in triggers if t in lower_text)
            if match_count > 0:
                scored.append((match_count, proc.success_rate, proc))

        # Sort: most matches first, then highest success rate
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored]

    def get_all(self) -> List[Procedure]:
        """Return all stored procedures.

        Returns:
            List of all Procedure objects.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("""
            SELECT name, trigger_pattern, procedure_steps,
                   success_count, failure_count, last_used, metadata
            FROM procedures
            ORDER BY name
            """)
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_procedure(row) for row in rows]

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def record_outcome(self, name: str, success: bool) -> None:
        """Record a success or failure for a procedure.

        Updates the success/failure counters and last_used timestamp.

        Args:
            name: Procedure name.
            success: Whether the procedure succeeded.
        """
        now = datetime.now(timezone.utc).isoformat()
        col = "success_count" if success else "failure_count"

        conn = sqlite3.connect(self._db_path)
        conn.execute(
            f"""
            UPDATE procedures
            SET {col} = {col} + 1,
                last_used = ?
            WHERE name = ?
            """,
            (now, name),
        )
        conn.commit()
        conn.close()

        logger.debug(
            "[procedural] recorded %s for '%s'",
            "success" if success else "failure",
            name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_procedure(row: tuple) -> Procedure:
        """Convert a database row to a Procedure dataclass."""
        steps_raw = row[2]
        if isinstance(steps_raw, str):
            steps = json.loads(steps_raw)
        else:
            steps = steps_raw or []
        meta_raw = row[6]
        if isinstance(meta_raw, str):
            meta = json.loads(meta_raw)
        else:
            meta = meta_raw or {}
        return Procedure(
            name=row[0],
            trigger_pattern=row[1],
            procedure_steps=steps,
            success_count=row[3] or 0,
            failure_count=row[4] or 0,
            last_used=row[5] or "",
            metadata=meta,
        )
