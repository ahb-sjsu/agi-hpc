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
Episodic Memory Service (AGI-HPC)

Responsibilities:
- Append observations/actions/simulations/safety events
- Query episodes or ranges with filtering
- Provide Append() + Query() RPCs
- Decision proof chain for governance

Now integrated with:
- PostgreSQL for production storage
- Decision Proof chain for audit trails
- EpisodicMemoryClient for high-level operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

from agi.proto_gen.memory_pb2 import (
    EpisodicAppendResponse,
    EpisodicQueryResponse,
    EpisodicEvent,
)
from agi.proto_gen.memory_pb2_grpc import (
    EpisodicServiceServicer,
    add_EpisodicServiceServicer_to_server,
)

from agi.memory.episodic.client import EpisodicMemoryClient
from agi.memory.episodic.postgres_store import (
    PostgresEpisodicStore,
    PostgresConfig,
    Episode,
    EpisodeStep,
    EpisodeEvent as EpisodeEventData,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage Adapter
# ---------------------------------------------------------------------------


class EpisodicStoreAdapter:
    """
    Adapter bridging gRPC proto types to PostgresEpisodicStore.

    Provides the legacy append/query interface while using PostgreSQL.
    Falls back to JSONL log if PostgreSQL is unavailable.
    """

    def __init__(
        self,
        postgres_store: Optional[PostgresEpisodicStore] = None,
        fallback_path: Optional[Path] = None,
    ):
        self._postgres = postgres_store
        self._use_postgres = postgres_store is not None
        self._fallback_path = fallback_path or Path("data/episodic")
        self._fallback_path.mkdir(parents=True, exist_ok=True)

    def append(self, events: List[EpisodicEvent]) -> None:
        """Append events to storage.

        Args:
            events: Proto EpisodicEvent messages
        """
        if not events:
            return

        if self._use_postgres and self._postgres:
            for event in events:
                event_data = EpisodeEventData(
                    episode_id=event.episode_id,
                    step_index=event.step_index,
                    event_type=event.event_type,
                    timestamp_ms=event.timestamp_ms,
                    payload=self._parse_payload(event.payload_uri),
                    tags=dict(event.tags) if event.tags else {},
                )
                try:
                    self._postgres.store_event(event_data)
                except Exception as e:
                    logger.warning(
                        "[episodic] postgres write failed: %s, using fallback",
                        e,
                    )
                    self._append_fallback([event])
        else:
            self._append_fallback(events)

        logger.debug("[episodic] appended %d events", len(events))

    def _append_fallback(self, events: List[EpisodicEvent]) -> None:
        """Append to JSONL fallback."""
        ts = int(time.time() * 1000)
        fname = self._fallback_path / f"events_{ts}.jsonl"
        with fname.open("a", encoding="utf8") as f:
            for e in events:
                f.write(
                    json.dumps(
                        {
                            "episode_id": e.episode_id,
                            "step_index": e.step_index,
                            "event_type": e.event_type,
                            "timestamp_ms": e.timestamp_ms,
                            "payload": e.payload_uri,
                            "tags": dict(e.tags) if e.tags else {},
                        }
                    )
                    + "\n"
                )

    def query(
        self,
        episode_id: str = "",
        event_type: str = "",
        start_time_ms: int = 0,
        end_time_ms: int = 0,
        limit: int = 1000,
    ) -> List[EpisodicEvent]:
        """Query events with filtering.

        Args:
            episode_id: Filter by episode
            event_type: Filter by event type
            start_time_ms: Start time filter
            end_time_ms: End time filter
            limit: Maximum results

        Returns:
            List of EpisodicEvent proto messages
        """
        if self._use_postgres and self._postgres:
            return self._query_postgres(
                episode_id, event_type, start_time_ms, end_time_ms, limit
            )
        return self._query_fallback(
            episode_id, event_type, start_time_ms, end_time_ms, limit
        )

    def _query_postgres(
        self,
        episode_id: str,
        event_type: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> List[EpisodicEvent]:
        """Query from PostgreSQL."""
        try:
            if episode_id:
                events_data = self._postgres.get_episode_events(episode_id)
            else:
                # Query recent episodes and their events
                episodes = self._postgres.query_episodes(limit=limit)
                events_data = []
                for ep in episodes:
                    events_data.extend(self._postgres.get_episode_events(ep.episode_id))

            # Apply filters
            results = []
            for e in events_data[:limit]:
                if event_type and e.event_type != event_type:
                    continue
                if start_time_ms and e.timestamp_ms < start_time_ms:
                    continue
                if end_time_ms and e.timestamp_ms > end_time_ms:
                    continue

                results.append(
                    EpisodicEvent(
                        episode_id=e.episode_id,
                        step_index=e.step_index,
                        event_type=e.event_type,
                        timestamp_ms=e.timestamp_ms,
                        payload_uri=json.dumps(e.payload) if e.payload else "",
                        tags=e.tags or {},
                    )
                )

            return results

        except Exception as e:
            logger.warning("[episodic] postgres query failed: %s, using fallback", e)
            return self._query_fallback(
                episode_id, event_type, start_time_ms, end_time_ms, limit
            )

    def _query_fallback(
        self,
        episode_id: str,
        event_type: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> List[EpisodicEvent]:
        """Query from JSONL fallback."""
        results = []
        for file in sorted(
            self._fallback_path.glob("*.jsonl"),
            reverse=True,
        ):
            with file.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if episode_id and r.get("episode_id") != episode_id:
                        continue
                    if event_type and r.get("event_type") != event_type:
                        continue
                    if start_time_ms and r.get("timestamp_ms", 0) < start_time_ms:
                        continue
                    if end_time_ms and r.get("timestamp_ms", 0) > end_time_ms:
                        continue

                    results.append(
                        EpisodicEvent(
                            episode_id=r.get("episode_id", ""),
                            step_index=r.get("step_index", 0),
                            event_type=r.get("event_type", ""),
                            timestamp_ms=r.get("timestamp_ms", 0),
                            payload_uri=r.get("payload", ""),
                            tags=r.get("tags", {}),
                        )
                    )

                    if len(results) >= limit:
                        return results

        return results

    @staticmethod
    def _parse_payload(payload_uri: str) -> Dict[str, Any]:
        """Parse payload from URI or JSON string."""
        if not payload_uri:
            return {}
        try:
            return json.loads(payload_uri)
        except json.JSONDecodeError:
            return {"uri": payload_uri}


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------


class EpisodicMemServicer(EpisodicServiceServicer):
    """gRPC servicer for episodic memory operations."""

    def __init__(
        self,
        store: EpisodicStoreAdapter,
        fabric: EventFabric,
        client: Optional[EpisodicMemoryClient] = None,
    ):
        self._store = store
        self._fabric = fabric
        self._client = client

    def Append(self, request, context):
        """Handle Append RPC."""
        events = list(request.events)
        self._store.append(events)
        self._fabric.publish(
            "memory.episodic.append",
            {
                "count": len(events),
            },
        )
        return EpisodicAppendResponse()

    def Query(self, request, context):
        """Handle Query RPC."""
        events = self._store.query(
            episode_id=getattr(request, "episode_id", ""),
            event_type=getattr(request, "event_type", ""),
            start_time_ms=getattr(request, "start_time_ms", 0),
            end_time_ms=getattr(request, "end_time_ms", 0),
            limit=request.limit or 1000,
        )
        return EpisodicQueryResponse(events=events)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EpisodicMemoryService:
    """
    Episodic Memory Service with PostgreSQL integration.

    Uses PostgreSQL for production storage with JSONL fallback.
    Supports decision proof chains for governance audit trails.
    """

    def __init__(self, config_path: str = "configs/memory_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()

        # Initialize PostgreSQL store (if available)
        self._postgres_store = self._create_postgres_store()

        # Create adapter
        self._store_adapter = EpisodicStoreAdapter(
            postgres_store=self._postgres_store,
            fallback_path=Path(
                getattr(self.config, "episodic_data_path", "data/episodic")
            ),
        )

        # High-level client for internal use
        self._client = (
            EpisodicMemoryClient(
                store=self._postgres_store,
                auto_init_schema=False,
            )
            if self._postgres_store
            else None
        )

        self.grpc = GRPCServer(self.config.rpc_port + 1)

    def _create_postgres_store(self) -> Optional[PostgresEpisodicStore]:
        """Create PostgreSQL store from configuration."""
        postgres_cfg = getattr(self.config, "postgres", None) or {}

        if not postgres_cfg.get("enabled", True):
            logger.info("[episodic] postgres disabled, using fallback")
            return None

        try:
            config = PostgresConfig(
                host=postgres_cfg.get("host", "localhost"),
                port=postgres_cfg.get("port", 5432),
                database=postgres_cfg.get("database", "agi_memory"),
                user=postgres_cfg.get("user", "postgres"),
                password=postgres_cfg.get("password", ""),
            )
            store = PostgresEpisodicStore(config)

            # Initialize schema if requested
            if postgres_cfg.get("init_schema", True):
                try:
                    store.init_schema()
                except Exception as e:
                    logger.warning("[episodic] schema init failed: %s", e)

            return store

        except Exception as e:
            logger.warning("[episodic] postgres initialization failed: %s", e)
            return None

    def run(self) -> None:
        """Start the episodic memory service."""
        servicer = EpisodicMemServicer(
            self._store_adapter,
            self.fabric,
            self._client,
        )
        add_EpisodicServiceServicer_to_server(servicer, self.grpc.server)

        storage_type = "postgres" if self._postgres_store else "jsonl"
        logger.info(
            "[episodic] service running on port %d (storage=%s)",
            self.config.rpc_port + 1,
            storage_type,
        )
        print("[MEM-EPI] Episodic Memory service running...")

        self.grpc.start()
        self.grpc.wait()

    def close(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        if self._postgres_store:
            self._postgres_store.close()
        logger.info("[episodic] service closed")


def main():
    logging.basicConfig(level=logging.INFO)
    EpisodicMemoryService().run()


if __name__ == "__main__":
    main()
