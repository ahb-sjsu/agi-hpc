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
Scoring and results storage for AtlasGym training.

Provides multiple scoring strategies (keyword matching, structure
analysis, code execution, exact match) and persists results to
the ``training_results`` PostgreSQL table.

Schema::

    CREATE TABLE training_results (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        env_name TEXT,
        level INT,
        scenario TEXT,
        response TEXT,
        score FLOAT,
        metadata JSONB DEFAULT '{}'
    );
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS training_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    env_name TEXT,
    level INT,
    scenario TEXT,
    response TEXT,
    score FLOAT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_training_results_env
    ON training_results (env_name);
CREATE INDEX IF NOT EXISTS idx_training_results_timestamp
    ON training_results (timestamp DESC);
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScorerConfig:
    """Configuration for the ResponseScorer.

    Attributes:
        db_dsn: PostgreSQL connection string.
        auto_create_table: Whether to create the results table on init.
        sandbox_timeout: Timeout in seconds for code execution sandbox.
    """

    db_dsn: str = "dbname=atlas user=claude"
    auto_create_table: bool = True
    sandbox_timeout: int = 5


# ---------------------------------------------------------------------------
# Scoring strategies
# ---------------------------------------------------------------------------


def score_keyword_presence(
    response: str,
    keywords: List[str],
    case_sensitive: bool = False,
) -> float:
    """Score based on presence of keywords in the response.

    Args:
        response: The text to score.
        keywords: List of keywords/phrases to look for.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Fraction of keywords found (0.0 to 1.0).
    """
    if not keywords:
        return 0.0
    text = response if case_sensitive else response.lower()
    found = sum(1 for kw in keywords if (kw if case_sensitive else kw.lower()) in text)
    return found / len(keywords)


def score_structure(
    response: str,
    min_words: int = 50,
    require_paragraphs: int = 2,
    require_sections: bool = False,
) -> float:
    """Score based on response structure (length, paragraphs, sections).

    Args:
        response: The text to score.
        min_words: Minimum word count for full credit.
        require_paragraphs: Minimum paragraph count.
        require_sections: Whether to require markdown-style sections.

    Returns:
        Structure score (0.0 to 1.0).
    """
    words = response.split()
    word_score = min(1.0, len(words) / max(min_words, 1))

    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    para_score = min(1.0, len(paragraphs) / max(require_paragraphs, 1))

    section_score = 1.0
    if require_sections:
        sections = re.findall(r"^#{1,3}\s+.+", response, re.MULTILINE)
        section_score = min(1.0, len(sections) / 2.0)

    return word_score * 0.4 + para_score * 0.3 + section_score * 0.3


def score_exact_match(response: str, expected: str) -> float:
    """Score based on exact match with the expected answer.

    Strips whitespace and compares case-insensitively.

    Args:
        response: The text to score.
        expected: The expected answer.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0


def score_numeric_match(
    response: str, expected: float, tolerance: float = 1e-6
) -> float:
    """Score based on numeric match within tolerance.

    Extracts the last number from the response and compares.

    Args:
        response: The text containing a numeric answer.
        expected: The expected numeric value.
        tolerance: Absolute tolerance for matching.

    Returns:
        1.0 if match, 0.5 if close (within 10x tolerance), 0.0 otherwise.
    """
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if not numbers:
        return 0.0
    try:
        value = float(numbers[-1])
    except ValueError:
        return 0.0
    if abs(value - expected) <= tolerance:
        return 1.0
    if abs(value - expected) <= tolerance * 10:
        return 0.5
    return 0.0


def score_code_execution(
    code: str,
    test_cases: List[Dict[str, Any]],
    timeout: int = 5,
) -> Tuple[float, Dict[str, Any]]:
    """Execute code in a sandboxed subprocess and check test cases.

    The sandbox restricts: no network, no file system writes beyond
    temp, timeout at *timeout* seconds.

    Args:
        code: Python code string to execute.
        test_cases: List of dicts with 'call' (expression) and 'expected' (value).
        timeout: Execution timeout in seconds.

    Returns:
        Tuple of (score, details_dict).
        Score: 0.0 (doesn't run), 0.5 (runs but wrong), 1.0 (all pass).
    """
    if not test_cases:
        return 0.0, {"error": "no test cases"}

    # Build test harness
    test_code = textwrap.dedent(code).strip()
    test_lines = [test_code, "", "import json", "results = []"]

    for i, tc in enumerate(test_cases):
        call_expr = tc.get("call", "")
        expected = json.dumps(tc.get("expected"))
        test_lines.append(
            f"try:\n"
            f"    _result_{i} = {call_expr}\n"
            f"    results.append({{'idx': {i}, 'got': _result_{i}, "
            f"'expected': {expected}, 'pass': _result_{i} == {expected}}})\n"
            f"except Exception as _e_{i}:\n"
            f"    results.append({{'idx': {i}, 'error': str(_e_{i}), 'pass': False}})"
        )

    test_lines.append("print(json.dumps(results))")
    full_code = "\n".join(test_lines)

    # Write to temp file and execute in sandbox
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-u",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": "/usr/bin:/usr/local/bin",
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        if result.returncode != 0:
            return 0.0, {
                "error": "execution_failed",
                "stderr": result.stderr[:500],
                "returncode": result.returncode,
            }

        # Parse results
        try:
            test_results = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return 0.0, {
                "error": "output_parse_failed",
                "stdout": result.stdout[:500],
            }

        passed = sum(1 for t in test_results if t.get("pass"))
        total = len(test_results)

        if total == 0:
            return 0.0, {"error": "no_results"}
        if passed == total:
            return 1.0, {"passed": passed, "total": total, "results": test_results}
        if passed > 0:
            return 0.5 + 0.5 * (passed / total), {
                "passed": passed,
                "total": total,
                "results": test_results,
            }
        return 0.5, {"passed": 0, "total": total, "results": test_results}

    except subprocess.TimeoutExpired:
        return 0.0, {"error": "timeout", "timeout_seconds": timeout}
    except FileNotFoundError:
        return 0.0, {"error": "python_not_found"}
    except Exception as exc:
        return 0.0, {"error": str(exc)}


# ---------------------------------------------------------------------------
# ResponseScorer
# ---------------------------------------------------------------------------


class ResponseScorer:
    """Multi-strategy response scorer with PostgreSQL persistence.

    Stores all scored results in the ``training_results`` table for
    longitudinal analysis and curriculum management.

    Usage::

        scorer = ResponseScorer()
        score = scorer.score_and_store(
            env_name="ethics",
            level=2,
            scenario="Analyze this dilemma...",
            response="The key principle here is...",
            score=0.75,
            metadata={"breakdown": {"relevance": 0.8, "depth": 0.7}},
        )
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self._config = config or ScorerConfig()
        if self._config.auto_create_table and psycopg2 is not None:
            self._ensure_table()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the training_results table if it does not exist."""
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.close()
            logger.info("[scorer] ensured training_results table exists")
        except Exception:
            logger.warning(
                "[scorer] could not create training_results table; "
                "results will not be persisted"
            )

    # ------------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------------

    def store_result(
        self,
        env_name: str,
        level: int,
        scenario: str,
        response: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Persist a training result to PostgreSQL.

        Args:
            env_name: Environment name (e.g. 'ethics').
            level: Difficulty level.
            scenario: Scenario text.
            response: Atlas's response text.
            score: Normalised score (0.0 to 1.0).
            metadata: Optional extra metadata.

        Returns:
            The row id if stored, None if persistence is unavailable.
        """
        if psycopg2 is None:
            logger.debug("[scorer] psycopg2 not available; skipping store")
            return None

        meta_json = json.dumps(metadata or {})

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO training_results
                        (env_name, level, scenario, response, score, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        env_name,
                        level,
                        scenario[:10000],
                        response[:10000],
                        score,
                        meta_json,
                    ),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
            conn.close()
            logger.debug(
                "[scorer] stored result id=%d env=%s score=%.2f",
                row_id,
                env_name,
                score,
            )
            return row_id
        except Exception:
            logger.exception("[scorer] failed to store training result")
            return None

    # ------------------------------------------------------------------
    # Query results
    # ------------------------------------------------------------------

    def get_recent_scores(self, env_name: str, n: int = 20) -> List[Dict[str, Any]]:
        """Retrieve the N most recent scores for an environment.

        Args:
            env_name: Environment name.
            n: Number of results to retrieve.

        Returns:
            List of result dicts.
        """
        if psycopg2 is None:
            return []

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, timestamp, level, score, metadata
                    FROM training_results
                    WHERE env_name = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (env_name, n),
                )
                rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1].isoformat() if r[1] else "",
                    "level": r[2],
                    "score": r[3],
                    "metadata": r[4] if r[4] else {},
                }
                for r in rows
            ]
        except Exception:
            logger.exception("[scorer] failed to query recent scores")
            return []

    def get_success_rate(self, env_name: str, level: int, n: int = 20) -> float:
        """Compute success rate (score >= 0.7) over last N attempts.

        Args:
            env_name: Environment name.
            level: Difficulty level.
            n: Window size.

        Returns:
            Success rate as a float (0.0 to 1.0).
        """
        if psycopg2 is None:
            return 0.0

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT score FROM training_results
                    WHERE env_name = %s AND level = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (env_name, level, n),
                )
                rows = cur.fetchall()
            conn.close()
            if not rows:
                return 0.0
            successes = sum(1 for r in rows if r[0] >= 0.7)
            return successes / len(rows)
        except Exception:
            logger.exception("[scorer] failed to query success rate")
            return 0.0

    def get_total_episodes(self, env_name: Optional[str] = None) -> int:
        """Count total episodes, optionally filtered by environment.

        Args:
            env_name: Optional environment filter.

        Returns:
            Total episode count.
        """
        if psycopg2 is None:
            return 0

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                if env_name:
                    cur.execute(
                        "SELECT COUNT(*) FROM training_results WHERE env_name = %s",
                        (env_name,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM training_results")
                count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception:
            logger.exception("[scorer] failed to count episodes")
            return 0
