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
Memory Service Broker for AGI-HPC Phase 2.

NATS-connected service that routes memory requests to the appropriate
tier (semantic, episodic, procedural) and publishes results.

Subscribes to:
    agi.memory.store.episodic    -- store a conversation episode
    agi.memory.store.procedural  -- store a learned procedure
    agi.memory.query.semantic    -- search semantic memory (RAG)
    agi.memory.query.episodic    -- search episodic memory
    agi.memory.query.procedural  -- lookup procedures

Publishes results to:
    agi.memory.result.{tier}

Telemetry published to:
    agi.meta.monitor.memory

Usage::

    service = MemoryService()
    await service.start()
    # ... runs until stopped ...
    await service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402
from agi.memory.episodic.store import EpisodicMemory, EpisodicMemoryConfig  # noqa: E402
from agi.memory.procedural.store import (  # noqa: E402
    ProceduralMemory,
    ProceduralMemoryConfig,
)
from agi.memory.semantic.store import SemanticMemory, SemanticMemoryConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MemoryServiceConfig:
    """Configuration for the Memory Service broker.

    Attributes:
        nats_servers: NATS server URLs.
        port: Service port (for health/metrics endpoint).
        db_dsn: PostgreSQL connection string for episodic + semantic.
        embed_model_name: Sentence-transformer model name.
        embed_device: Device for embedding inference.
        sqlite_path: SQLite database path for procedural memory.
        semantic_top_k: Default semantic search results.
        episodic_recall_n: Default episodic recall count.
        seed_procedures: Whether to seed built-in procedures.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    port: int = 50300
    db_dsn: str = "dbname=atlas user=claude"
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    sqlite_path: str = "/home/claude/agi-hpc/data/procedural.db"
    semantic_top_k: int = 6
    episodic_recall_n: int = 10
    seed_procedures: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> MemoryServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        mem = data.get("memory", data)
        nats_cfg = mem.get("nats", {})
        pg_cfg = mem.get("postgresql", {})
        embed_cfg = mem.get("embedding", {})
        proc_cfg = mem.get("procedural", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            port=mem.get("port", 50300),
            db_dsn=pg_cfg.get("dsn", "dbname=atlas user=claude"),
            embed_model_name=embed_cfg.get("model", "BAAI/bge-m3"),
            embed_device=embed_cfg.get("device", "cpu"),
            sqlite_path=proc_cfg.get(
                "db_path", "/home/claude/agi-hpc/data/procedural.db"
            ),
            semantic_top_k=mem.get("semantic_top_k", 6),
            episodic_recall_n=mem.get("episodic_recall_n", 10),
            seed_procedures=proc_cfg.get("seed", True),
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass
class MemoryTelemetry:
    """Accumulates memory service metrics."""

    queries_total: int = 0
    stores_total: int = 0
    semantic_queries: int = 0
    episodic_queries: int = 0
    procedural_queries: int = 0
    episodic_stores: int = 0
    procedural_stores: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        total = self.queries_total + self.stores_total
        if total == 0:
            return 0.0
        return self.total_latency_ms / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries_total": self.queries_total,
            "stores_total": self.stores_total,
            "semantic_queries": self.semantic_queries,
            "episodic_queries": self.episodic_queries,
            "procedural_queries": self.procedural_queries,
            "episodic_stores": self.episodic_stores,
            "procedural_stores": self.procedural_stores,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Memory Service
# ---------------------------------------------------------------------------


class MemoryService:
    """NATS-connected memory service broker.

    Routes memory requests to the appropriate tier and publishes
    results back through the event fabric.
    """

    # Subjects this service listens on
    STORE_SUBJECTS = {
        "agi.memory.store.episodic": "_handle_store_episodic",
        "agi.memory.store.procedural": "_handle_store_procedural",
    }
    QUERY_SUBJECTS = {
        "agi.memory.query.semantic": "_handle_query_semantic",
        "agi.memory.query.episodic": "_handle_query_episodic",
        "agi.memory.query.procedural": "_handle_query_procedural",
    }

    def __init__(self, config: Optional[MemoryServiceConfig] = None) -> None:
        self._config = config or MemoryServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._semantic: Optional[SemanticMemory] = None
        self._episodic: Optional[EpisodicMemory] = None
        self._procedural: Optional[ProceduralMemory] = None
        self._telemetry = MemoryTelemetry()
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and initialise all memory tiers."""
        logger.info("[memory-service] starting Phase 2 Memory Service")

        # Initialise NATS fabric
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Initialise memory tiers
        self._episodic = EpisodicMemory(
            config=EpisodicMemoryConfig(
                db_dsn=self._config.db_dsn,
                auto_create_table=True,
            )
        )
        logger.info("[memory-service] episodic memory ready")

        self._procedural = ProceduralMemory(
            config=ProceduralMemoryConfig(
                db_path=self._config.sqlite_path,
                auto_create=True,
                seed_procedures=self._config.seed_procedures,
            )
        )
        logger.info("[memory-service] procedural memory ready")

        self._semantic = SemanticMemory(
            config=SemanticMemoryConfig(
                db_dsn=self._config.db_dsn,
                embed_model_name=self._config.embed_model_name,
                embed_device=self._config.embed_device,
                top_k=self._config.semantic_top_k,
            )
        )
        logger.info("[memory-service] semantic memory ready")

        # Subscribe to all subjects
        all_subjects = {**self.STORE_SUBJECTS, **self.QUERY_SUBJECTS}
        for subject, handler_name in all_subjects.items():
            handler = getattr(self, handler_name)
            await self._fabric.subscribe(subject, handler)

        self._running = True
        logger.info(
            "[memory-service] ready -- subscribed to %d subjects",
            len(all_subjects),
        )

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[memory-service] stopped")

    # ------------------------------------------------------------------
    # Store handlers
    # ------------------------------------------------------------------

    async def _handle_store_episodic(self, event: Event) -> None:
        """Handle agi.memory.store.episodic events.

        Expected payload:
            session_id (str): Session identifier.
            user_message (str): User's input.
            atlas_response (str): System response.
            hemisphere (str): 'lh', 'rh', or 'both'.
            metadata (dict, optional): Extra metadata.
            safety_flags (dict, optional): Safety check results.
            quality_score (float, optional): Quality score.
            embedding (list, optional): 1024-dim vector.
        """
        t0 = time.perf_counter()
        trace_id = event.trace_id

        try:
            p = event.payload
            episode_id = self._episodic.store_episode(
                session_id=p.get("session_id", "unknown"),
                user_msg=p.get("user_message", ""),
                response=p.get("atlas_response", ""),
                hemisphere=p.get("hemisphere", "lh"),
                metadata=p.get("metadata"),
                safety_flags=p.get("safety_flags"),
                quality_score=p.get("quality_score", 0.0),
                embedding=p.get("embedding"),
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.stores_total += 1
            self._telemetry.episodic_stores += 1
            self._telemetry.total_latency_ms += latency_ms

            # Publish result
            result_event = Event.create(
                source="memory",
                event_type="memory.result.episodic",
                payload={
                    "action": "store",
                    "episode_id": episode_id,
                    "session_id": p.get("session_id", "unknown"),
                    "latency_ms": round(latency_ms, 1),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.result.episodic", result_event)

            logger.info(
                "[memory-service] stored episode id=%s trace=%s %.0fms",
                episode_id[:8],
                trace_id[:8],
                latency_ms,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[memory-service] error storing episode trace=%s",
                trace_id[:8],
            )

    async def _handle_store_procedural(self, event: Event) -> None:
        """Handle agi.memory.store.procedural events.

        Expected payload:
            name (str): Procedure name.
            trigger (str): Trigger pattern.
            steps (list[str]): Procedure steps.
            metadata (dict, optional): Extra metadata.
        """
        t0 = time.perf_counter()
        trace_id = event.trace_id

        try:
            p = event.payload
            self._procedural.store_procedure(
                name=p.get("name", ""),
                trigger=p.get("trigger", ""),
                steps=p.get("steps", []),
                metadata=p.get("metadata"),
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.stores_total += 1
            self._telemetry.procedural_stores += 1
            self._telemetry.total_latency_ms += latency_ms

            result_event = Event.create(
                source="memory",
                event_type="memory.result.procedural",
                payload={
                    "action": "store",
                    "name": p.get("name", ""),
                    "latency_ms": round(latency_ms, 1),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.result.procedural", result_event)

            logger.info(
                "[memory-service] stored procedure name=%s trace=%s",
                p.get("name", ""),
                trace_id[:8],
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[memory-service] error storing procedure trace=%s",
                trace_id[:8],
            )

    # ------------------------------------------------------------------
    # Query handlers
    # ------------------------------------------------------------------

    async def _handle_query_semantic(self, event: Event) -> None:
        """Handle agi.memory.query.semantic events.

        Expected payload:
            query (str): Natural language search query.
            top_k (int, optional): Number of results.
        """
        t0 = time.perf_counter()
        trace_id = event.trace_id

        try:
            p = event.payload
            query = p.get("query", "")
            top_k = p.get("top_k", self._config.semantic_top_k)

            results = self._semantic.search(query, top_k=top_k)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.queries_total += 1
            self._telemetry.semantic_queries += 1
            self._telemetry.total_latency_ms += latency_ms

            result_event = Event.create(
                source="memory",
                event_type="memory.result.semantic",
                payload={
                    "action": "query",
                    "query": query[:200],
                    "results": [
                        {
                            "repo": r.repo,
                            "file": r.file,
                            "text": r.text[:500],
                            "score": r.score,
                        }
                        for r in results
                    ],
                    "count": len(results),
                    "latency_ms": round(latency_ms, 1),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.result.semantic", result_event)

            logger.info(
                "[memory-service] semantic query=%r results=%d trace=%s %.0fms",
                query[:40],
                len(results),
                trace_id[:8],
                latency_ms,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[memory-service] error querying semantic trace=%s",
                trace_id[:8],
            )

    async def _handle_query_episodic(self, event: Event) -> None:
        """Handle agi.memory.query.episodic events.

        Expected payload:
            session_id (str, optional): Session to recall from.
            n (int, optional): Number of recent episodes.
            query_embedding (list, optional): For similarity search.
            top_k (int, optional): For similarity search.
            mode (str, optional): 'recent', 'similar', or 'history'.
        """
        t0 = time.perf_counter()
        trace_id = event.trace_id

        try:
            p = event.payload
            mode = p.get("mode", "recent")
            episodes: list = []

            if mode == "similar" and p.get("query_embedding"):
                episodes = self._episodic.recall_similar(
                    query_embedding=p["query_embedding"],
                    top_k=p.get("top_k", 5),
                )
            elif mode == "history" and p.get("session_id"):
                episodes = self._episodic.get_session_history(
                    session_id=p["session_id"]
                )
            else:
                # Default: recent episodes for session
                episodes = self._episodic.recall_recent(
                    session_id=p.get("session_id", "unknown"),
                    n=p.get("n", self._config.episodic_recall_n),
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.queries_total += 1
            self._telemetry.episodic_queries += 1
            self._telemetry.total_latency_ms += latency_ms

            result_event = Event.create(
                source="memory",
                event_type="memory.result.episodic",
                payload={
                    "action": "query",
                    "mode": mode,
                    "episodes": [ep.to_dict() for ep in episodes],
                    "count": len(episodes),
                    "latency_ms": round(latency_ms, 1),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.result.episodic", result_event)

            logger.info(
                "[memory-service] episodic query mode=%s results=%d trace=%s %.0fms",
                mode,
                len(episodes),
                trace_id[:8],
                latency_ms,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[memory-service] error querying episodic trace=%s",
                trace_id[:8],
            )

    async def _handle_query_procedural(self, event: Event) -> None:
        """Handle agi.memory.query.procedural events.

        Expected payload:
            trigger_text (str): Text to match against procedure triggers.
        """
        t0 = time.perf_counter()
        trace_id = event.trace_id

        try:
            p = event.payload
            trigger_text = p.get("trigger_text", "")

            procedures = self._procedural.lookup(trigger_text)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.queries_total += 1
            self._telemetry.procedural_queries += 1
            self._telemetry.total_latency_ms += latency_ms

            result_event = Event.create(
                source="memory",
                event_type="memory.result.procedural",
                payload={
                    "action": "query",
                    "trigger_text": trigger_text[:200],
                    "procedures": [proc.to_dict() for proc in procedures],
                    "count": len(procedures),
                    "latency_ms": round(latency_ms, 1),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.result.procedural", result_event)

            logger.info(
                "[memory-service] procedural query=%r results=%d trace=%s %.0fms",
                trigger_text[:40],
                len(procedures),
                trace_id[:8],
                latency_ms,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[memory-service] error querying procedural trace=%s",
                trace_id[:8],
            )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> MemoryTelemetry:
        """Return current telemetry snapshot."""
        return self._telemetry

    async def _publish_telemetry(self, trace_id: str = "") -> None:
        """Publish telemetry to the monitoring subject."""
        if self._fabric and self._fabric.is_connected:
            telemetry_event = Event.create(
                source="memory",
                event_type="meta.monitor.memory",
                payload=self._telemetry.to_dict(),
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.meta.monitor.memory", telemetry_event)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the Memory Service until interrupted."""
    if config_path:
        config = MemoryServiceConfig.from_yaml(config_path)
    else:
        config = MemoryServiceConfig()

    service = MemoryService(config=config)
    await service.start()

    try:
        # Run forever until cancelled
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the Memory Service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AGI-HPC Memory Service (Phase 2)")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to memory_config.yaml",
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
        logger.info("[memory-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
