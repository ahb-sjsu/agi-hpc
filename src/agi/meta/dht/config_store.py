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
Config Version Store for AGI-HPC.

Stores configuration versions in PostgreSQL, enabling distributed
config changes across subsystems. Each config entry is keyed by a
string key and stores an arbitrary JSON value with a monotonically
increasing version number.

Usage::

    store = ConfigStore(dsn="dbname=atlas user=claude")
    await store.init_db()

    await store.put("safety.veto_threshold", 0.3, version=1)
    value, version = await store.get("safety.veto_threshold")
    keys = await store.list_keys()

    await store.close()

Phase 6 (DHT Service Registry + Final Polish).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS config_store (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL DEFAULT '{}'::jsonb,
    version     INTEGER NOT NULL DEFAULT 1,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_UPSERT_SQL = """
INSERT INTO config_store (key, value, version, updated_at)
VALUES ($1, $2, $3, $4)
ON CONFLICT (key)
DO UPDATE SET
    value = EXCLUDED.value,
    version = EXCLUDED.version,
    updated_at = EXCLUDED.updated_at;
"""

_SELECT_SQL = """
SELECT value, version FROM config_store WHERE key = $1;
"""

_LIST_KEYS_SQL = """
SELECT key FROM config_store ORDER BY key;
"""

_DELETE_SQL = """
DELETE FROM config_store WHERE key = $1;
"""

_SELECT_ALL_SQL = """
SELECT key, value, version, updated_at FROM config_store ORDER BY key;
"""


# ---------------------------------------------------------------------------
# DSN parsing
# ---------------------------------------------------------------------------


def _parse_dsn(dsn: str) -> Dict[str, Any]:
    """Parse a libpq-style or URI DSN into asyncpg keyword args.

    asyncpg does not accept libpq-style ``"dbname=atlas user=claude"``
    DSNs. This helper converts them to keyword arguments.

    URI-style (``"postgresql://..."`` ) is passed through as ``dsn=``.
    """
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        return {"dsn": dsn}
    kwargs: Dict[str, Any] = {}
    key_map = {
        "dbname": "database",
        "user": "user",
        "password": "password",
        "host": "host",
        "port": "port",
    }
    for token in dsn.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pg_key = key_map.get(key, key)
        if pg_key == "port":
            kwargs[pg_key] = int(value)
        else:
            kwargs[pg_key] = value
    return kwargs


# ---------------------------------------------------------------------------
# Config Store
# ---------------------------------------------------------------------------


class ConfigStore:
    """Versioned configuration store backed by PostgreSQL.

    Stores configuration key-value pairs with version tracking,
    enabling distributed config changes across AGI-HPC subsystems.

    Args:
        dsn: PostgreSQL connection string (libpq-style or URI).
    """

    def __init__(self, dsn: str = "dbname=atlas user=claude") -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required: pip install asyncpg")
        self._dsn = dsn
        self._pg_kwargs = _parse_dsn(dsn)
        self._pool: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create the connection pool and ensure the table exists."""
        self._pool = await asyncpg.create_pool(
            min_size=1, max_size=5, **self._pg_kwargs
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        logger.info("[config-store] database initialised (config_store table)")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("[config-store] connection pool closed")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def put(
        self,
        key: str,
        value: Any,
        version: int = 1,
    ) -> None:
        """Store a configuration value with version.

        Args:
            key: Configuration key (e.g. ``"safety.veto_threshold"``).
            value: JSON-serialisable value.
            version: Version number (should be monotonically increasing).
        """
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT_SQL,
                key,
                json.dumps(value),
                version,
                now,
            )
        logger.info("[config-store] put key=%s version=%d", key, version)

    async def get(self, key: str) -> Optional[Tuple[Any, int]]:
        """Retrieve a configuration value and its version.

        Args:
            key: Configuration key.

        Returns:
            Tuple of ``(value, version)`` or ``None`` if not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SELECT_SQL, key)
        if row is None:
            return None
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        return (value, row["version"])

    async def list_keys(self) -> List[str]:
        """Return all configuration keys."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_LIST_KEYS_SQL)
        return [row["key"] for row in rows]

    async def delete(self, key: str) -> bool:
        """Delete a configuration entry.

        Returns:
            ``True`` if the key was found and removed.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(_DELETE_SQL, key)
        return result == "DELETE 1"

    async def list_all(self) -> List[Dict[str, Any]]:
        """Return all configuration entries with full details.

        Returns:
            List of dicts with ``key``, ``value``, ``version``, ``updated_at``.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_ALL_SQL)
        results = []
        for row in rows:
            value = row["value"]
            if isinstance(value, str):
                value = json.loads(value)
            results.append(
                {
                    "key": row["key"],
                    "value": value,
                    "version": row["version"],
                    "updated_at": row["updated_at"].isoformat(),
                }
            )
        return results
