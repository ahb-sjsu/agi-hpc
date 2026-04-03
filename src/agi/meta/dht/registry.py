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
DHT Service Registry for AGI-HPC.

Provides a local service registry using a simplified Kademlia-like
interface. Each subsystem registers on startup, heartbeats every 10
seconds, and the registry stores state in PostgreSQL for persistence
across restarts.

Statuses:
    ``"healthy"``   -- service is responding and heartbeat is fresh
    ``"degraded"``  -- no heartbeat received in 60 seconds
    ``"unknown"``   -- newly registered, not yet checked

Usage::

    registry = ServiceRegistry(dsn="dbname=atlas user=claude")
    await registry.init_db()

    await registry.register("safety", 50055, {"phase": 3})
    info = await registry.lookup("safety")
    all_services = await registry.list_all()
    await registry.health_check()
    await registry.deregister("safety")

Phase 6 (DHT Service Registry + Final Polish).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_STATUS_HEALTHY = "healthy"
_STATUS_DEGRADED = "degraded"
_STATUS_UNKNOWN = "unknown"

_STALE_THRESHOLD_S = 60.0  # seconds without heartbeat -> degraded


@dataclass
class ServiceInfo:
    """Descriptor for a registered subsystem.

    Attributes:
        name: Unique service name (e.g. ``"safety"``, ``"memory"``).
        host: Hostname or IP address the service is listening on.
        port: TCP port the service is listening on.
        status: Current health status (``"healthy"``, ``"degraded"``,
            ``"unknown"``).
        last_heartbeat: UTC timestamp of last heartbeat.
        metadata: Arbitrary JSON-serialisable metadata (phase, version, etc.).
    """

    name: str = ""
    host: str = "localhost"
    port: int = 0
    status: str = _STATUS_UNKNOWN
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        d = asdict(self)
        d["last_heartbeat"] = self.last_heartbeat.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceInfo:
        """Construct from a dictionary (e.g. parsed JSON)."""
        data = dict(data)
        ts = data.get("last_heartbeat")
        if isinstance(ts, str):
            data["last_heartbeat"] = datetime.fromisoformat(ts)
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS service_registry (
    name            TEXT PRIMARY KEY,
    host            TEXT NOT NULL DEFAULT 'localhost',
    port            INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'unknown',
    last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

_UPSERT_SQL = """
INSERT INTO service_registry (name, host, port, status, last_heartbeat, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (name)
DO UPDATE SET
    host = EXCLUDED.host,
    port = EXCLUDED.port,
    status = EXCLUDED.status,
    last_heartbeat = EXCLUDED.last_heartbeat,
    metadata = EXCLUDED.metadata;
"""

_SELECT_ONE_SQL = """
SELECT name, host, port, status, last_heartbeat, metadata
FROM service_registry
WHERE name = $1;
"""

_SELECT_ALL_SQL = """
SELECT name, host, port, status, last_heartbeat, metadata
FROM service_registry
ORDER BY name;
"""

_DELETE_SQL = """
DELETE FROM service_registry WHERE name = $1;
"""

_UPDATE_STATUS_SQL = """
UPDATE service_registry
SET status = $2
WHERE name = $1;
"""

_UPDATE_HEARTBEAT_SQL = """
UPDATE service_registry
SET last_heartbeat = $2, status = 'healthy'
WHERE name = $1;
"""


# ---------------------------------------------------------------------------
# DSN parsing
# ---------------------------------------------------------------------------


def _parse_dsn(dsn: str) -> Dict[str, Any]:
    """Parse a libpq-style or URI DSN into asyncpg keyword args.

    asyncpg does not accept libpq-style ``"dbname=atlas user=claude"``
    DSNs. This helper converts them to keyword arguments that
    ``asyncpg.create_pool()`` and ``asyncpg.connect()`` accept.

    URI-style (``"postgresql://..."`` ) is passed through as ``dsn=``.
    """
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        return {"dsn": dsn}
    # Parse libpq-style: "key=value key=value"
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
# Service Registry
# ---------------------------------------------------------------------------


class ServiceRegistry:
    """Local service registry backed by PostgreSQL.

    Implements a simplified Kademlia-like interface for service
    discovery within the AGI-HPC cognitive architecture.

    Args:
        dsn: PostgreSQL connection string (libpq-style or URI).
            Libpq-style: ``"dbname=atlas user=claude"``
            URI-style: ``"postgresql://claude@localhost/atlas"``
        host: Default host for registered services.
        stale_threshold_s: Seconds after which a service with no
            heartbeat is marked degraded.
    """

    def __init__(
        self,
        dsn: str = "dbname=atlas user=claude",
        host: str = "localhost",
        stale_threshold_s: float = _STALE_THRESHOLD_S,
    ) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required: pip install asyncpg")
        self._dsn = dsn
        self._pg_kwargs = _parse_dsn(dsn)
        self._host = host
        self._stale_threshold_s = stale_threshold_s
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
        logger.info("[dht-registry] database initialised (service_registry table)")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("[dht-registry] connection pool closed")

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------

    async def register(
        self,
        service_name: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        host: Optional[str] = None,
    ) -> ServiceInfo:
        """Register (or update) a subsystem in the registry.

        Args:
            service_name: Unique service identifier.
            port: TCP port.
            metadata: Arbitrary metadata (phase, version, etc.).
            host: Override host (defaults to ``self._host``).

        Returns:
            The newly registered ``ServiceInfo``.
        """
        now = datetime.now(timezone.utc)
        info = ServiceInfo(
            name=service_name,
            host=host or self._host,
            port=port,
            status=_STATUS_HEALTHY,
            last_heartbeat=now,
            metadata=metadata or {},
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT_SQL,
                info.name,
                info.host,
                info.port,
                info.status,
                info.last_heartbeat,
                json.dumps(info.metadata),
            )
        logger.info(
            "[dht-registry] registered %s at %s:%d",
            service_name,
            info.host,
            port,
        )
        return info

    async def deregister(self, service_name: str) -> bool:
        """Remove a service from the registry.

        Returns:
            ``True`` if the service was found and removed.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(_DELETE_SQL, service_name)
        removed = result == "DELETE 1"
        if removed:
            logger.info("[dht-registry] deregistered %s", service_name)
        else:
            logger.warning("[dht-registry] deregister: %s not found", service_name)
        return removed

    async def lookup(self, service_name: str) -> Optional[ServiceInfo]:
        """Find a service by name.

        Returns:
            ``ServiceInfo`` if found, ``None`` otherwise.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SELECT_ONE_SQL, service_name)
        if row is None:
            return None
        return _row_to_info(row)

    async def list_all(self) -> List[ServiceInfo]:
        """Return all registered services."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_ALL_SQL)
        return [_row_to_info(r) for r in rows]

    async def heartbeat(self, service_name: str) -> bool:
        """Record a heartbeat for a service, marking it healthy.

        Returns:
            ``True`` if the service was found and updated.
        """
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            result = await conn.execute(_UPDATE_HEARTBEAT_SQL, service_name, now)
        updated = result == "UPDATE 1"
        if updated:
            logger.debug("[dht-registry] heartbeat from %s", service_name)
        else:
            logger.warning("[dht-registry] heartbeat: %s not found", service_name)
        return updated

    async def health_check(self) -> Dict[str, str]:
        """Ping all registered services and update their status.

        For each registered service:
        - If last_heartbeat is older than ``stale_threshold_s``, mark degraded.
        - Optionally attempt an HTTP GET to ``http://{host}:{port}/health``.

        Returns:
            Dict mapping service name to new status.
        """
        services = await self.list_all()
        now = datetime.now(timezone.utc)
        results: Dict[str, str] = {}

        for svc in services:
            elapsed = (now - svc.last_heartbeat).total_seconds()
            if elapsed > self._stale_threshold_s:
                new_status = _STATUS_DEGRADED
            else:
                # Try HTTP health probe
                new_status = await self._probe_service(svc)

            if new_status != svc.status:
                async with self._pool.acquire() as conn:
                    await conn.execute(_UPDATE_STATUS_SQL, svc.name, new_status)
                logger.info(
                    "[dht-registry] %s status: %s -> %s",
                    svc.name,
                    svc.status,
                    new_status,
                )
            results[svc.name] = new_status

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _probe_service(self, svc: ServiceInfo) -> str:
        """Attempt an HTTP health probe against a service.

        Returns ``"healthy"`` if the probe succeeds (HTTP 200),
        otherwise returns the current status unchanged.
        """
        if aiohttp is None:
            # Cannot probe without aiohttp; rely on heartbeat age
            return svc.status

        url = f"http://{svc.host}:{svc.port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        return _STATUS_HEALTHY
                    return svc.status
        except Exception:
            logger.debug(
                "[dht-registry] health probe failed for %s at %s",
                svc.name,
                url,
            )
            return svc.status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_info(row: Any) -> ServiceInfo:
    """Convert an asyncpg Record to a ServiceInfo."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ServiceInfo(
        name=row["name"],
        host=row["host"],
        port=row["port"],
        status=row["status"],
        last_heartbeat=row["last_heartbeat"],
        metadata=meta if isinstance(meta, dict) else {},
    )
