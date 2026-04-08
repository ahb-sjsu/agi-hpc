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
DHT NATS Service for AGI-HPC.

Connects the DHT Service Registry to the NATS Event Fabric, enabling
distributed service discovery and heartbeat management across subsystems.

Subscribes to:
    ``agi.dht.register``    -- service registration requests
    ``agi.dht.deregister``  -- service removal requests
    ``agi.dht.lookup``      -- service discovery queries
    ``agi.dht.heartbeat``   -- heartbeat pings from services

Publishes:
    ``agi.dht.status``      -- full registry state every 30 seconds

Detects stale services (no heartbeat in 60s) and marks them degraded.

Usage::

    service = DhtNatsService()
    await service.start()
    # ... runs until stopped ...
    await service.stop()

Phase 6 (DHT Service Registry + Final Polish).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402
from agi.meta.dht.registry import ServiceRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DhtServiceConfig:
    """Configuration for the DHT NATS Service.

    Attributes:
        nats_servers: NATS server URLs.
        dsn: PostgreSQL connection string for the registry.
        host: Default host for registered services.
        status_interval_s: Seconds between registry status publishes.
        health_check_interval_s: Seconds between health checks.
        stale_threshold_s: Seconds without heartbeat before marking degraded.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    dsn: str = "dbname=atlas user=claude"
    host: str = "localhost"
    status_interval_s: float = 30.0
    health_check_interval_s: float = 30.0
    stale_threshold_s: float = 60.0

    @classmethod
    def from_yaml(cls, path: str) -> DhtServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        dht = data.get("dht", data)
        nats_cfg = dht.get("nats", {})
        pg_cfg = dht.get("postgresql", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            dsn=pg_cfg.get("dsn", "dbname=atlas user=claude"),
            host=dht.get("host", "localhost"),
            status_interval_s=dht.get("status_interval_s", 30.0),
            health_check_interval_s=dht.get("health_check_interval_s", 30.0),
            stale_threshold_s=dht.get("stale_threshold_s", 60.0),
        )


# ---------------------------------------------------------------------------
# NATS subjects
# ---------------------------------------------------------------------------

_SUBJECT_REGISTER = "agi.dht.register"
_SUBJECT_DEREGISTER = "agi.dht.deregister"
_SUBJECT_LOOKUP = "agi.dht.lookup"
_SUBJECT_HEARTBEAT = "agi.dht.heartbeat"
_SUBJECT_STATUS = "agi.dht.status"


# ---------------------------------------------------------------------------
# DHT NATS Service
# ---------------------------------------------------------------------------


class DhtNatsService:
    """NATS-connected DHT service for distributed service discovery.

    Bridges the ``ServiceRegistry`` (PostgreSQL-backed) with the NATS
    event fabric so that all subsystems can register, discover, and
    heartbeat through the messaging layer.
    """

    def __init__(self, config: Optional[DhtServiceConfig] = None) -> None:
        self._config = config or DhtServiceConfig()
        self._registry: Optional[ServiceRegistry] = None
        self._fabric: Optional[NatsEventFabric] = None
        self._running = False
        self._status_task: Optional[asyncio.Task[None]] = None
        self._health_task: Optional[asyncio.Task[None]] = None

    @property
    def registry(self) -> Optional[ServiceRegistry]:
        """Return the underlying ServiceRegistry."""
        return self._registry

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to NATS + PostgreSQL and start processing."""
        logger.info("[dht-service] starting Phase 6 DHT Service")

        # Initialise registry
        self._registry = ServiceRegistry(
            dsn=self._config.dsn,
            host=self._config.host,
            stale_threshold_s=self._config.stale_threshold_s,
        )
        await self._registry.init_db()

        # Connect to NATS
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Subscribe to DHT subjects
        await self._fabric.subscribe(_SUBJECT_REGISTER, self._handle_register)
        await self._fabric.subscribe(_SUBJECT_DEREGISTER, self._handle_deregister)
        await self._fabric.subscribe(_SUBJECT_LOOKUP, self._handle_lookup)
        await self._fabric.subscribe(_SUBJECT_HEARTBEAT, self._handle_heartbeat)

        self._running = True

        # Start periodic tasks
        self._status_task = asyncio.create_task(self._status_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            "[dht-service] ready -- listening on agi.dht.{register,deregister,lookup,heartbeat}"
        )

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False

        for task in (self._status_task, self._health_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._fabric:
            await self._fabric.disconnect()

        if self._registry:
            await self._registry.close()

        logger.info("[dht-service] stopped")

    # ------------------------------------------------------------------
    # NATS handlers
    # ------------------------------------------------------------------

    async def _handle_register(self, event: Event) -> None:
        """Handle service registration request.

        Expected payload keys:
            ``service_name`` (str): The service to register.
            ``port`` (int): TCP port.
            ``metadata`` (dict, optional): Extra metadata.
            ``host`` (str, optional): Host override.
        """
        payload = event.payload
        service_name = payload.get("service_name", "")
        port = payload.get("port", 0)
        metadata = payload.get("metadata", {})
        host = payload.get("host")

        if not service_name or not port:
            logger.warning(
                "[dht-service] register: missing service_name or port in %s",
                event.id[:8],
            )
            return

        info = await self._registry.register(
            service_name=service_name,
            port=port,
            metadata=metadata,
            host=host,
        )
        logger.info(
            "[dht-service] registered %s at %s:%d via NATS",
            service_name,
            info.host,
            port,
        )

    async def _handle_deregister(self, event: Event) -> None:
        """Handle service removal request.

        Expected payload keys:
            ``service_name`` (str): The service to remove.
        """
        service_name = event.payload.get("service_name", "")
        if not service_name:
            logger.warning(
                "[dht-service] deregister: missing service_name in %s",
                event.id[:8],
            )
            return
        await self._registry.deregister(service_name)

    async def _handle_lookup(self, event: Event) -> None:
        """Handle service discovery query.

        Expected payload keys:
            ``service_name`` (str): The service to look up.

        Publishes the result back on ``agi.dht.lookup.result``.
        """
        service_name = event.payload.get("service_name", "")
        if not service_name:
            logger.warning(
                "[dht-service] lookup: missing service_name in %s",
                event.id[:8],
            )
            return

        info = await self._registry.lookup(service_name)
        result_payload: Dict[str, Any] = {
            "service_name": service_name,
            "found": info is not None,
        }
        if info:
            result_payload["service"] = info.to_dict()

        result_event = Event.create(
            source="dht",
            event_type="dht.lookup.result",
            payload=result_payload,
            trace_id=event.trace_id,
        )
        await self._fabric.publish("agi.dht.lookup.result", result_event)

    async def _handle_heartbeat(self, event: Event) -> None:
        """Handle heartbeat from a service.

        Expected payload keys:
            ``service_name`` (str): The heartbeating service.
        """
        service_name = event.payload.get("service_name", "")
        if not service_name:
            return
        await self._registry.heartbeat(service_name)

    # ------------------------------------------------------------------
    # Periodic tasks
    # ------------------------------------------------------------------

    async def _status_loop(self) -> None:
        """Periodically publish full registry state to agi.dht.status."""
        while self._running:
            try:
                await asyncio.sleep(self._config.status_interval_s)
                await self._publish_status()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[dht-service] status publish error")

    async def _health_check_loop(self) -> None:
        """Periodically run health checks on all registered services."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval_s)
                results = await self._registry.health_check()
                logger.debug("[dht-service] health check: %s", results)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[dht-service] health check error")

    async def _publish_status(self) -> None:
        """Publish current registry state."""
        if not self._fabric or not self._fabric.is_connected:
            return

        services = await self._registry.list_all()
        status_event = Event.create(
            source="dht",
            event_type="dht.status",
            payload={
                "services": [s.to_dict() for s in services],
                "total": len(services),
                "healthy": sum(1 for s in services if s.status == "healthy"),
                "degraded": sum(1 for s in services if s.status == "degraded"),
            },
        )
        await self._fabric.publish(_SUBJECT_STATUS, status_event)
        logger.debug("[dht-service] published status: %d services", len(services))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the DHT Service until interrupted."""
    if config_path:
        config = DhtServiceConfig.from_yaml(config_path)
    else:
        config = DhtServiceConfig()

    service = DhtNatsService(config=config)
    await service.start()

    try:
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the DHT Service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AGI-HPC DHT Service Registry (Phase 6)"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to dht_config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(run_service(args.config))
    except KeyboardInterrupt:
        logger.info("[dht-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
