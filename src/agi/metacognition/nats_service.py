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
Metacognition NATS Service for AGI-HPC Phase 5.

Main service that runs the Monitor, Reflector, and Adjuster together.
Provides a single entry point for the entire metacognition subsystem.

Components:
    - Monitor:   subscribes to agi.> and tracks rolling metrics
    - Reflector: every N interactions, sends self-reflection to LLM
    - Adjuster:  based on metrics, publishes tuning adjustments
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402
from agi.metacognition.monitor import MetacognitionMonitor  # noqa: E402
from agi.metacognition.reflector import (  # noqa: E402
    MetacognitionReflector,
    ReflectorConfig,
)
from agi.metacognition.adjuster import (  # noqa: E402
    MetacognitionAdjuster,
    AdjusterConfig,
)

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class MetacognitionServiceConfig:
    """Configuration for the Metacognition NATS service.

    Attributes:
        nats_servers: NATS server URLs.
        summary_interval: Seconds between monitor summary publications.
        reflection_interval: Interactions between reflections.
        llm_base_url: LLM URL for reflection (Gemma 4).
        llm_timeout: LLM request timeout.
        llm_model: Model name for reflection.
        latency_threshold_p95_ms: Adjuster latency threshold.
        veto_rate_threshold: Adjuster safety veto threshold.
        enable_reflector: Whether to run the reflector.
        enable_adjuster: Whether to run the adjuster.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    summary_interval: float = 60.0
    reflection_interval: int = 10
    llm_base_url: str = "http://localhost:8080"
    llm_timeout: float = 300.0
    llm_model: str = "gemma-4-31b"
    latency_threshold_p95_ms: float = 30000.0
    veto_rate_threshold: float = 0.1
    enable_reflector: bool = True
    enable_adjuster: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> MetacognitionServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        meta = data.get("metacognition", data)
        nats_cfg = meta.get("nats", {})
        llm_cfg = meta.get("llm", {})
        adj_cfg = meta.get("adjuster", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            summary_interval=meta.get("summary_interval", 60.0),
            reflection_interval=meta.get("reflection_interval", 10),
            llm_base_url=llm_cfg.get("base_url", "http://localhost:8080"),
            llm_timeout=llm_cfg.get("timeout", 300.0),
            llm_model=llm_cfg.get("model", "gemma-4-31b"),
            latency_threshold_p95_ms=adj_cfg.get("latency_threshold_p95_ms", 30000.0),
            veto_rate_threshold=adj_cfg.get("veto_rate_threshold", 0.1),
            enable_reflector=meta.get("enable_reflector", True),
            enable_adjuster=meta.get("enable_adjuster", True),
        )


# -----------------------------------------------------------------
# Metacognition NATS Service
# -----------------------------------------------------------------


class MetacognitionService:
    """Main metacognition NATS service.

    Orchestrates the Monitor, Reflector, and Adjuster on a single
    NATS connection.

    Usage::

        service = MetacognitionService()
        await service.start()
        # ... runs until stopped ...
        await service.stop()
    """

    def __init__(
        self,
        config: Optional[MetacognitionServiceConfig] = None,
    ) -> None:
        self._config = config or MetacognitionServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._monitor: Optional[MetacognitionMonitor] = None
        self._reflector: Optional[MetacognitionReflector] = None
        self._adjuster: Optional[MetacognitionAdjuster] = None
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and start all metacognition components."""
        logger.info("[meta-service] starting Phase 5 Metacognition Service")

        # Initialise NATS fabric
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Start Monitor
        self._monitor = MetacognitionMonitor(
            summary_interval=self._config.summary_interval,
        )
        await self._monitor.start(self._fabric)

        # Start Reflector
        if self._config.enable_reflector:
            reflector_config = ReflectorConfig(
                reflection_interval=self._config.reflection_interval,
                llm_base_url=self._config.llm_base_url,
                llm_timeout=self._config.llm_timeout,
                llm_model=self._config.llm_model,
            )
            self._reflector = MetacognitionReflector(config=reflector_config)
            await self._reflector.start(self._fabric)

        # Start Adjuster
        if self._config.enable_adjuster:
            adjuster_config = AdjusterConfig(
                latency_threshold_p95_ms=self._config.latency_threshold_p95_ms,
                veto_rate_threshold=self._config.veto_rate_threshold,
            )
            self._adjuster = MetacognitionAdjuster(config=adjuster_config)
            await self._adjuster.start(self._fabric)

        self._running = True
        logger.info(
            "[meta-service] ready -- monitor=%s reflector=%s adjuster=%s",
            "on",
            "on" if self._config.enable_reflector else "off",
            "on" if self._config.enable_adjuster else "off",
        )

    async def stop(self) -> None:
        """Stop all components and disconnect."""
        self._running = False
        if self._adjuster:
            await self._adjuster.stop()
        if self._reflector:
            await self._reflector.stop()
        if self._monitor:
            await self._monitor.stop()
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[meta-service] stopped")

    @property
    def monitor(self) -> Optional[MetacognitionMonitor]:
        """Return the monitor instance."""
        return self._monitor

    @property
    def reflector(self) -> Optional[MetacognitionReflector]:
        """Return the reflector instance."""
        return self._reflector

    @property
    def adjuster(self) -> Optional[MetacognitionAdjuster]:
        """Return the adjuster instance."""
        return self._adjuster


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the Metacognition Service until interrupted."""
    if config_path:
        config = MetacognitionServiceConfig.from_yaml(config_path)
    else:
        config = MetacognitionServiceConfig()

    service = MetacognitionService(config=config)
    await service.start()

    try:
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the Metacognition Service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AGI-HPC Metacognition Service (Phase 5)"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to meta_config.yaml",
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
        logger.info("[meta-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
